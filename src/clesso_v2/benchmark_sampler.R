##############################################################################
##
## benchmark_sampler.R -- Performance benchmarks for CLESSO v2 sampler
##
## Generates synthetic observation data and profiles every major code path
## in clesso_sampler.R to identify bottlenecks.
##
## Usage:
##   source("benchmark_sampler.R")
##
## Output:
##   Console timing report + optional Rprof flame-graph data saved to
##   clesso_v2/output/sampler_profile.out
##
##############################################################################

library(data.table)
library(microbenchmark)

cat("=== CLESSO v2 Sampler Benchmark Suite ===\n\n")

# ---------------------------------------------------------------------------
# 0. Source the sampler
# ---------------------------------------------------------------------------
this_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    if (nchar(getwd()) > 0) getwd()
    else stop("Cannot determine script directory.")
  }
)

sampler_path <- file.path(this_dir, "clesso_sampler.R")
if (!file.exists(sampler_path)) {
  sampler_path <- file.path(this_dir, "src", "clesso_v2", "clesso_sampler.R")
}
source(sampler_path)


# ===========================================================================
# 1. Generate synthetic observation data at several scales
# ===========================================================================
generate_synthetic_obs <- function(n_sites    = 500,
                                   n_species  = 200,
                                   obs_per_site_mean = 20,
                                   seed       = 42) {
  set.seed(seed)

  ## Create sites on a grid
  lons <- runif(n_sites, 110, 155)
  lats <- runif(n_sites, -44, -10)
  site_ids <- paste0(round(lons, 2), ":", round(lats, 2))

  species_pool <- paste0("sp_", sprintf("%04d", seq_len(n_species)))

  ## Poisson-distributed records per site
  n_records <- rpois(n_sites, obs_per_site_mean)
  n_records[n_records < 1] <- 1

  ## Build observation table
  obs_list <- lapply(seq_len(n_sites), function(i) {
    n <- n_records[i]
    ## Sample species with a log-series-like abundance distribution
    sp_probs <- 1 / seq_len(n_species)
    sp_probs <- sp_probs / sum(sp_probs)
    sp <- sample(species_pool, n, replace = TRUE, prob = sp_probs)

    richness_i <- length(unique(sp))

    data.table(
      site_id   = site_ids[i],
      species   = sp,
      longitude = lons[i],
      latitude  = lats[i],
      eventDate = as.character(as.Date("2000-01-01") + sample(0:6570, n, replace = TRUE)),
      nRecords  = n,
      nRecords_exDateLocDups = n,
      nSiteVisits = sample(1:10, 1),
      richness  = richness_i
    )
  })

  rbindlist(obs_list)
}

## Several benchmark scales
scales <- list(
  small  = list(n_sites = 100,  n_species = 50,   obs_per_site = 10),
  medium = list(n_sites = 500,  n_species = 200,  obs_per_site = 20),
  large  = list(n_sites = 2000, n_species = 500,  obs_per_site = 30)
)

cat("Generating synthetic datasets...\n")
datasets <- lapply(names(scales), function(sz) {
  s <- scales[[sz]]
  dt <- generate_synthetic_obs(
    n_sites           = s$n_sites,
    n_species         = s$n_species,
    obs_per_site_mean = s$obs_per_site,
    seed              = 42
  )
  cat(sprintf("  %-7s: %6d records, %4d sites, %4d species\n",
              sz, nrow(dt), length(unique(dt$site_id)),
              length(unique(dt$species))))
  dt
})
names(datasets) <- names(scales)


# ===========================================================================
# 2. Micro-benchmarks: isolate hot-spots within the sampler
# ===========================================================================
cat("\n--- Micro-benchmarks ---\n\n")

obs_med <- datasets$medium
setDT(obs_med)
setkey(obs_med, site_id)

# ---------------------------------------------------------------------------
# 2a. Within-site: per-site lapply pair drawing
#     (the inner lapply that draws 2 records from each site)
# ---------------------------------------------------------------------------
cat("[2a] Within-site: per-site pair drawing via lapply vs vectorised\n")

## Setup: build site index once
site_index <- obs_med[, .(rows = list(.I)), by = site_id]
setkey(site_index, site_id)
site_counts <- obs_med[, .N, by = site_id]
eligible_sites <- site_counts[N >= 2]$site_id
eligible_obs <- obs_med[site_id %in% eligible_sites]
setkey(eligible_obs, site_id)
site_index_e <- eligible_obs[, .(rows = list(.I)), by = site_id]
setkey(site_index_e, site_id)

## Current approach: lapply
n_test <- 10000L
site_cap <- eligible_obs[, .(.N), by = site_id]
site_cap[, capacity := N * (N - 1L) / 2]
site_cap[, weight := capacity / sum(capacity)]
sampled_sites_test <- sample(site_cap$site_id, n_test,
                              replace = TRUE, prob = site_cap$weight)

bench_lapply <- function() {
  rbindlist(lapply(sampled_sites_test, function(sid) {
    rows <- site_index_e[.(sid), rows][[1]]
    if (length(rows) < 2) return(NULL)
    idx <- sample(rows, 2, replace = FALSE)
    data.table(idx1 = idx[1], idx2 = idx[2])
  }))
}

## Vectorised alternative: pre-expand row lists, then sample in bulk
bench_vectorised <- function() {
  ## Look up row lists for all sampled sites at once
  site_rows <- site_index_e[.(sampled_sites_test), rows]
  lens <- vapply(site_rows, length, 0L)
  valid <- lens >= 2L

  ## For valid sites, draw 2 indices
  n_valid <- sum(valid)
  if (n_valid == 0) return(data.table(idx1 = integer(0), idx2 = integer(0)))

  valid_rows <- site_rows[valid]
  idx1 <- integer(n_valid)
  idx2 <- integer(n_valid)
  for (j in seq_len(n_valid)) {
    r <- valid_rows[[j]]
    s <- sample.int(length(r), 2L)
    idx1[j] <- r[s[1L]]
    idx2[j] <- r[s[2L]]
  }
  data.table(idx1 = idx1, idx2 = idx2)
}

mb_2a <- microbenchmark(
  lapply_current   = bench_lapply(),
  vectorised_alt   = bench_vectorised(),
  times = 5
)
cat("  Results (10k site draws):\n")
print(mb_2a)


# ---------------------------------------------------------------------------
# 2b. Key-based deduplication: paste0 vs bit64/digest
# ---------------------------------------------------------------------------
cat("\n[2b] Pair-key deduplication: paste0 vs integer key\n")

n_keys <- 100000L
s1 <- sample.int(nrow(obs_med), n_keys, replace = TRUE)
s2 <- sample.int(nrow(obs_med), n_keys, replace = TRUE)

bench_paste_key <- function() {
  sp1 <- obs_med$species[s1]
  sp2 <- obs_med$species[s2]
  sid <- obs_med$site_id[s1]
  d1 <- obs_med$eventDate[s1]
  d2 <- obs_med$eventDate[s2]
  key <- paste0(sid, ":",
                pmin(as.character(sp1), as.character(sp2)), ":",
                pmax(as.character(sp1), as.character(sp2)), ":",
                pmin(as.character(d1), as.character(d2)), ":",
                pmax(as.character(d1), as.character(d2)))
  duplicated(key)
}

bench_integer_key <- function() {
  ## Use row indices as key: order-invariant by sorting
  k1 <- pmin(s1, s2)
  k2 <- pmax(s1, s2)
  duplicated(data.table(k1, k2))
}

mb_2b <- microbenchmark(
  paste0_key  = bench_paste_key(),
  integer_key = bench_integer_key(),
  times = 5
)
cat("  Results (100k pairs):\n")
print(mb_2b)


# ---------------------------------------------------------------------------
# 2c. Between-site Phase 1: growing attempted_keys vector
#     Benchmarks the O(n²) pattern of c(keys, new_keys) + duplicated()
# ---------------------------------------------------------------------------
cat("\n[2c] Between-site: growing key vector vs environment-based set\n")

bench_growing_vec <- function(n_iters = 20, batch_size = 5000) {
  attempted_keys <- character(0)
  for (i in seq_len(n_iters)) {
    new_keys <- paste0(sample.int(1e6, batch_size), "~",
                       sample.int(1e6, batch_size))
    all_keys <- c(attempted_keys, new_keys)
    is_dup <- duplicated(all_keys)
    is_dup <- is_dup[(length(attempted_keys) + 1):length(all_keys)]
    attempted_keys <- c(attempted_keys, new_keys[!is_dup])
  }
  length(attempted_keys)
}

bench_env_set <- function(n_iters = 20, batch_size = 5000) {
  seen <- new.env(hash = TRUE, parent = emptyenv(), size = n_iters * batch_size)
  total <- 0L
  for (i in seq_len(n_iters)) {
    new_keys <- paste0(sample.int(1e6, batch_size), "~",
                       sample.int(1e6, batch_size))
    is_dup <- vapply(new_keys, function(k) exists(k, envir = seen, inherits = FALSE), logical(1))
    novel <- new_keys[!is_dup]
    for (k in novel) assign(k, TRUE, envir = seen)
    total <- total + length(novel)
  }
  total
}

mb_2c <- microbenchmark(
  growing_vector = bench_growing_vec(),
  env_hashset    = bench_env_set(),
  times = 3
)
cat("  Results (20 iters x 5000 batch):\n")
print(mb_2c)


# ---------------------------------------------------------------------------
# 2d. Between-site Phase 2: species-stratified match sampling
#     lapply per-species vs vectorised batch draw
# ---------------------------------------------------------------------------
cat("\n[2d] Between-site Phase 2: per-species lapply vs batch draw\n")

obs_med[, obs_idx := .I]
species_index <- obs_med[, .(obs_indices = list(obs_idx), n = .N), by = species]
species_multi <- species_index[n >= 2]
setkey(species_multi, species)

n_sp_test <- 5000L
sp_sample <- sample(species_multi$species, n_sp_test, replace = TRUE,
                     prob = species_multi$n / sum(species_multi$n))

bench_lapply_phase2 <- function() {
  rbindlist(lapply(sp_sample, function(sp) {
    idx_pool <- species_multi[.(sp), obs_indices][[1]]
    if (length(idx_pool) < 2) return(NULL)
    pair_idx <- sample(idx_pool, 2, replace = FALSE)
    ri <- pair_idx[1]
    rj <- pair_idx[2]
    if (obs_med$site_id[ri] == obs_med$site_id[rj]) return(NULL)
    data.table(ri = ri, rj = rj)
  }), fill = TRUE)
}

bench_vectorised_phase2 <- function() {
  ## Batch: look up all index pools, draw pairs in a loop but build one DT
  idx_pools <- species_multi[.(sp_sample), obs_indices]
  ri <- integer(n_sp_test)
  rj <- integer(n_sp_test)
  valid <- logical(n_sp_test)
  for (j in seq_len(n_sp_test)) {
    pool <- idx_pools[[j]]
    if (length(pool) < 2) next
    pair <- sample(pool, 2L, replace = FALSE)
    if (obs_med$site_id[pair[1]] == obs_med$site_id[pair[2]]) next
    ri[j] <- pair[1]
    rj[j] <- pair[2]
    valid[j] <- TRUE
  }
  data.table(ri = ri[valid], rj = rj[valid])
}

mb_2d <- microbenchmark(
  lapply_per_species = bench_lapply_phase2(),
  vectorised_batch   = bench_vectorised_phase2(),
  times = 5
)
cat("  Results (5000 species draws):\n")
print(mb_2d)

## Cleanup
obs_med[, obs_idx := NULL]


# ===========================================================================
# 3. Full sampler timing at each scale
# ===========================================================================
cat("\n--- Full sampler timing (per-scale) ---\n\n")

pair_targets <- list(
  small  = list(n_within = 5000,   n_between = 5000),
  medium = list(n_within = 50000,  n_between = 50000),
  large  = list(n_within = 200000, n_between = 200000)
)

full_timings <- list()

for (sz in names(datasets)) {
  obs <- datasets[[sz]]
  tgt <- pair_targets[[sz]]

  cat(sprintf("Scale: %s (%d records) -- within=%d, between=%d\n",
              sz, nrow(obs), tgt$n_within, tgt$n_between))

  t0 <- proc.time()
  result <- clesso_sampler(
    obs_dt              = obs,
    n_within            = tgt$n_within,
    n_between           = tgt$n_between,
    within_min_recs     = 2,
    within_match_ratio  = 0.5,
    between_match_ratio = 0.5,
    balance_weights     = TRUE,
    seed                = 42,
    species_thresh      = 500,
    cores               = 1
  )
  elapsed <- proc.time() - t0
  full_timings[[sz]] <- elapsed

  cat(sprintf("  --> %.2f seconds (user: %.2f, system: %.2f)\n\n",
              elapsed["elapsed"], elapsed["user.self"], elapsed["sys.self"]))
}


# ===========================================================================
# 4. Rprof-based profiling of the medium dataset
# ===========================================================================
cat("\n--- Rprof profiling (medium dataset) ---\n")

output_dir <- file.path(this_dir, "output")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
prof_file <- file.path(output_dir, "sampler_profile.out")

obs_prof <- datasets$medium

Rprof(prof_file, interval = 0.01)
invisible(clesso_sampler(
  obs_dt              = obs_prof,
  n_within            = 50000,
  n_between           = 50000,
  within_min_recs     = 2,
  within_match_ratio  = 0.5,
  between_match_ratio = 0.5,
  balance_weights     = TRUE,
  seed                = 42,
  species_thresh      = 500,
  cores               = 1
))
Rprof(NULL)

prof_summary <- summaryRprof(prof_file)

cat("\n--- Top 20 time-consuming functions (by self.time) ---\n")
top_self <- head(prof_summary$by.self, 20)
print(top_self)

cat("\n--- Top 20 functions (by total.time) ---\n")
top_total <- head(prof_summary$by.total, 20)
print(top_total)


# ===========================================================================
# 5. Granular timing: instrument individual sampler stages
# ===========================================================================
cat("\n--- Granular stage timing (medium dataset) ---\n")

obs_gran <- copy(datasets$medium)
setDT(obs_gran)

## --- Within-site timing breakdown ---
cat("\n  [Within-site breakdown]\n")

t_start <- proc.time()

## 5a. Site eligibility check
t_a <- proc.time()
site_counts <- obs_gran[, .N, by = site_id]
eligible_sites <- site_counts[N >= 2]$site_id
eligible_obs <- obs_gran[site_id %in% eligible_sites]
setkey(eligible_obs, site_id)
t_a <- proc.time() - t_a
cat(sprintf("    Site eligibility:    %.4f s\n", t_a["elapsed"]))

## 5b. Build site index
t_b <- proc.time()
site_index_g <- eligible_obs[, .(rows = list(.I)), by = site_id]
setkey(site_index_g, site_id)
t_b <- proc.time() - t_b
cat(sprintf("    Build site index:    %.4f s\n", t_b["elapsed"]))

## 5c. Capacity computation
t_c <- proc.time()
site_cap_g <- eligible_obs[, .(.N), by = site_id]
site_cap_g[, capacity := N * (N - 1L) / 2]
site_cap_g[, weight := capacity / sum(capacity)]
t_c <- proc.time() - t_c
cat(sprintf("    Capacity weights:    %.4f s\n", t_c["elapsed"]))

## 5d. Site sampling (draw 50k sites)
n_draw <- 100000L
t_d <- proc.time()
sampled_sites_g <- sample(site_cap_g$site_id, n_draw,
                           replace = TRUE, prob = site_cap_g$weight)
t_d <- proc.time() - t_d
cat(sprintf("    Sample sites (100k): %.4f s\n", t_d["elapsed"]))

## 5e. Pair drawing via lapply
t_e <- proc.time()
pairs_raw <- rbindlist(lapply(sampled_sites_g, function(sid) {
  rows <- site_index_g[.(sid), rows][[1]]
  if (length(rows) < 2) return(NULL)
  idx <- sample(rows, 2, replace = FALSE)
  data.table(idx1 = idx[1], idx2 = idx[2])
}))
t_e <- proc.time() - t_e
cat(sprintf("    lapply pair draw:    %.4f s  *** LIKELY BOTTLENECK ***\n", t_e["elapsed"]))

## 5f. Build batch data.table
t_f <- proc.time()
if (nrow(pairs_raw) > 0) {
  rec1 <- eligible_obs[pairs_raw$idx1]
  rec2 <- eligible_obs[pairs_raw$idx2]
  batch <- data.table(
    site_i = rec1$site_id, site_j = rec2$site_id,
    species_i = rec1$species, species_j = rec2$species,
    y = as.integer(rec1$species != rec2$species)
  )
}
t_f <- proc.time() - t_f
cat(sprintf("    Build batch DT:      %.4f s\n", t_f["elapsed"]))

## 5g. Deduplication via paste0 key
t_g <- proc.time()
if (nrow(pairs_raw) > 0) {
  batch[, pair_key := paste0(
    site_i, ":",
    pmin(as.character(species_i), as.character(species_j)), ":",
    pmax(as.character(species_i), as.character(species_j))
  )]
  deduped <- batch[!duplicated(pair_key)]
}
t_g <- proc.time() - t_g
cat(sprintf("    Deduplication:       %.4f s\n", t_g["elapsed"]))

## --- Between-site Phase 1 timing breakdown ---
cat("\n  [Between-site Phase 1 breakdown]\n")

n_obs <- nrow(obs_gran)

## 5h. Random pair generation
t_h <- proc.time()
nm <- 100000L
s1 <- sample.int(n_obs, nm, replace = TRUE)
s2 <- sample.int(n_obs, nm, replace = TRUE)
same_site <- obs_gran$site_id[s1] == obs_gran$site_id[s2]
s1 <- s1[!same_site]
s2 <- s2[!same_site]
t_h <- proc.time() - t_h
cat(sprintf("    Random pair gen:     %.4f s\n", t_h["elapsed"]))

## 5i. Key creation + dedup (simulating the growing vector)
t_i <- proc.time()
key <- fifelse(s1 <= s2,
               paste(s1, s2, sep = "~"),
               paste(s2, s1, sep = "~"))
dup <- duplicated(key)
t_i <- proc.time() - t_i
cat(sprintf("    Key create+dedup:    %.4f s\n", t_i["elapsed"]))

## 5j. Batch data.table construction
t_j <- proc.time()
batch_btwn <- data.table(
  site_i    = obs_gran$site_id[s1],
  site_j    = obs_gran$site_id[s2],
  species_i = obs_gran$species[s1],
  species_j = obs_gran$species[s2],
  y         = as.integer(obs_gran$species[s1] != obs_gran$species[s2])
)
t_j <- proc.time() - t_j
cat(sprintf("    Batch DT build:      %.4f s\n", t_j["elapsed"]))

## --- Between-site Phase 2 timing breakdown ---
cat("\n  [Between-site Phase 2 breakdown]\n")

obs_gran[, obs_idx := .I]
species_index_g <- obs_gran[, .(obs_indices = list(obs_idx), n = .N), by = species]
species_multi_g <- species_index_g[n >= 2]
setkey(species_multi_g, species)

n_sp_bench <- 10000L
sp_sample_g <- sample(species_multi_g$species, n_sp_bench, replace = TRUE,
                       prob = species_multi_g$n / sum(species_multi_g$n))

## 5k. Per-species lapply (match generation)
t_k <- proc.time()
match_batch <- rbindlist(lapply(sp_sample_g, function(sp) {
  idx_pool <- species_multi_g[.(sp), obs_indices][[1]]
  if (length(idx_pool) < 2) return(NULL)
  pair_idx <- sample(idx_pool, 2, replace = FALSE)
  ri <- pair_idx[1]
  rj <- pair_idx[2]
  if (obs_gran$site_id[ri] == obs_gran$site_id[rj]) return(NULL)
  data.table(ri = ri, rj = rj)
}), fill = TRUE)
t_k <- proc.time() - t_k
cat(sprintf("    Phase2 lapply (10k): %.4f s  *** LIKELY BOTTLENECK ***\n", t_k["elapsed"]))

obs_gran[, obs_idx := NULL]


# ===========================================================================
# 6. Summary report
# ===========================================================================
cat("\n")
cat("===================================================================\n")
cat("  BENCHMARK SUMMARY\n")
cat("===================================================================\n\n")

cat("Full sampler timings:\n")
for (sz in names(full_timings)) {
  cat(sprintf("  %-7s: %.2f s elapsed\n", sz, full_timings[[sz]]["elapsed"]))
}

cat("\nMicro-benchmark medians (milliseconds):\n")
mb_results <- list(
  `2a: within pair draw` = mb_2a,
  `2b: key dedup`        = mb_2b,
  `2c: growing key vec`  = mb_2c,
  `2d: phase2 match`     = mb_2d
)
for (nm in names(mb_results)) {
  mb <- mb_results[[nm]]
  summ <- summary(mb)
  cat(sprintf("  %s\n", nm))
  for (i in seq_len(nrow(summ))) {
    cat(sprintf("    %-20s: %8.1f ms (median)\n", summ$expr[i], summ$median[i]))
  }
}

cat("\nKnown bottlenecks (prioritised):\n")
cat("  1. Within-site lapply per-site pair drawing -- R-level loop over\n")
cat("     potentially 100k+ sites; creates tiny data.tables each call.\n")
cat("  2. Between-site Phase 1: growing `attempted_keys` via c() -- O(n^2)\n")
cat("     copy pattern. Use an environment hash set instead.\n")
cat("  3. Between-site Phase 2: per-species lapply -- same R-level loop\n")
cat("     overhead; should batch species lookups and build one DT.\n")
cat("  4. String-based pair key deduplication (paste0) -- much slower than\n")
cat("     integer index-based dedup via data.table.\n")
cat("  5. Within-site: capacity weights recomputed each iteration (move\n")
cat("     outside the loop).\n")
cat("\n  See clesso_sampler_optimised.R for implementations of these fixes.\n")

cat(sprintf("\nRprof data saved to: %s\n", prof_file))
cat("  View with: summaryRprof('sampler_profile.out')\n")
cat("  Or:        profvis::profvis(prof = 'sampler_profile.out')\n")
cat("\n=== Benchmark complete ===\n")
