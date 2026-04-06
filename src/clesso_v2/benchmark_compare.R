##############################################################################
##
## benchmark_compare.R -- Side-by-side comparison: original vs optimised sampler
##
## Runs both sampler implementations on the same synthetic data and
## compares wall-clock time and output equivalence.
##
## Usage:
##   source("benchmark_compare.R")
##
##############################################################################

library(data.table)

cat("=== CLESSO Sampler: Original vs Optimised Comparison ===\n\n")

this_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    if (nchar(getwd()) > 0) getwd()
    else stop("Cannot determine script directory.")
  }
)

# ---------------------------------------------------------------------------
# Generate test data
# ---------------------------------------------------------------------------
generate_test_obs <- function(n_sites = 500, n_species = 200,
                               obs_per_site_mean = 20, seed = 42) {
  set.seed(seed)
  lons <- runif(n_sites, 110, 155)
  lats <- runif(n_sites, -44, -10)
  site_ids <- paste0(round(lons, 2), ":", round(lats, 2))
  species_pool <- paste0("sp_", sprintf("%04d", seq_len(n_species)))
  n_records <- rpois(n_sites, obs_per_site_mean)
  n_records[n_records < 1] <- 1

  rbindlist(lapply(seq_len(n_sites), function(i) {
    n <- n_records[i]
    sp_probs <- 1 / seq_len(n_species)
    sp_probs <- sp_probs / sum(sp_probs)
    sp <- sample(species_pool, n, replace = TRUE, prob = sp_probs)
    data.table(
      site_id = site_ids[i], species = sp,
      longitude = lons[i], latitude = lats[i],
      eventDate = as.character(as.Date("2000-01-01") + sample(0:6570, n, replace = TRUE)),
      nRecords = n,
      nRecords_exDateLocDups = n,
      nSiteVisits = sample(1:10, 1),
      richness = length(unique(sp))
    )
  }))
}

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
configs <- list(
  quick = list(
    label      = "Quick (10k pairs)",
    n_sites    = 200,
    n_species  = 100,
    obs_per_site = 15,
    n_within   = 10000,
    n_between  = 10000
  ),
  medium = list(
    label      = "Medium (100k pairs)",
    n_sites    = 500,
    n_species  = 200,
    obs_per_site = 20,
    n_within   = 100000,
    n_between  = 100000
  ),
  full = list(
    label      = "Full scale (500k pairs)",
    n_sites    = 2000,
    n_species  = 500,
    obs_per_site = 30,
    n_within   = 500000,
    n_between  = 500000
  )
)

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
results <- list()

for (cfg_name in names(configs)) {
  cfg <- configs[[cfg_name]]
  cat(sprintf("\n{'='*60}\n"))
  cat(sprintf("Config: %s\n", cfg$label))
  cat(sprintf("{'='*60}\n\n"))

  obs_data <- generate_test_obs(
    n_sites           = cfg$n_sites,
    n_species         = cfg$n_species,
    obs_per_site_mean = cfg$obs_per_site,
    seed              = 42
  )
  cat(sprintf("  Data: %d records, %d sites, %d species\n",
              nrow(obs_data), length(unique(obs_data$site_id)),
              length(unique(obs_data$species))))

  ## --- Run ORIGINAL ---
  cat("\n  [Original sampler]\n")
  source(file.path(this_dir, "clesso_sampler.R"))
  gc(verbose = FALSE)

  t_orig <- proc.time()
  orig_result <- clesso_sampler(
    obs_dt              = copy(obs_data),
    n_within            = cfg$n_within,
    n_between           = cfg$n_between,
    within_min_recs     = 2,
    within_match_ratio  = 0.5,
    between_match_ratio = 0.5,
    balance_weights     = TRUE,
    seed                = 42,
    species_thresh      = 500,
    cores               = 1
  )
  t_orig <- proc.time() - t_orig

  ## --- Run OPTIMISED ---
  cat("\n  [Optimised sampler]\n")
  source(file.path(this_dir, "clesso_sampler_optimised.R"))
  gc(verbose = FALSE)

  t_opt <- proc.time()
  opt_result <- clesso_sampler(
    obs_dt              = copy(obs_data),
    n_within            = cfg$n_within,
    n_between           = cfg$n_between,
    within_min_recs     = 2,
    within_match_ratio  = 0.5,
    between_match_ratio = 0.5,
    balance_weights     = TRUE,
    seed                = 42,
    species_thresh      = 500,
    cores               = 1
  )
  t_opt <- proc.time() - t_opt

  ## --- Compare ---
  speedup <- t_orig["elapsed"] / max(t_opt["elapsed"], 0.001)

  results[[cfg_name]] <- list(
    label       = cfg$label,
    n_within    = cfg$n_within,
    n_between   = cfg$n_between,
    t_orig      = t_orig["elapsed"],
    t_opt       = t_opt["elapsed"],
    speedup     = speedup,
    n_orig      = nrow(orig_result),
    n_opt       = nrow(opt_result),
    match_orig  = sum(orig_result$y == 0),
    match_opt   = sum(opt_result$y == 0),
    cols_match  = identical(sort(names(orig_result)), sort(names(opt_result)))
  )

  cat(sprintf("\n  %-12s  Original: %7.2f s  |  Optimised: %7.2f s  |  Speedup: %.1fx\n",
              cfg$label, t_orig["elapsed"], t_opt["elapsed"], speedup))
  cat(sprintf("  %-12s  Pairs: %d (orig) vs %d (opt)  |  Cols match: %s\n",
              "", nrow(orig_result), nrow(opt_result),
              if (results[[cfg_name]]$cols_match) "YES" else "NO"))
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n")
cat("===================================================================\n")
cat("  COMPARISON SUMMARY\n")
cat("===================================================================\n\n")
cat(sprintf("  %-25s  %10s  %10s  %8s\n", "Config", "Original", "Optimised", "Speedup"))
cat(sprintf("  %-25s  %10s  %10s  %8s\n", "------", "--------", "---------", "-------"))
for (r in results) {
  cat(sprintf("  %-25s  %8.2f s  %8.2f s  %6.1fx\n",
              r$label, r$t_orig, r$t_opt, r$speedup))
}

cat("\nOutput compatibility:\n")
for (nm in names(results)) {
  r <- results[[nm]]
  cat(sprintf("  %-25s  rows: %d/%d  cols_match: %s\n",
              r$label, r$n_orig, r$n_opt, r$cols_match))
}

cat("\n=== Comparison complete ===\n")

## Restore the original sampler so the pipeline is not affected
source(file.path(this_dir, "clesso_sampler.R"))
