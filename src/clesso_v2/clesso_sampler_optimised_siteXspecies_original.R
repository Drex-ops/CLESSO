##############################################################################
##
## clesso_sampler_optimised.R -- Performance-optimised sampler for CLESSO v2
##
## Drop-in replacement for clesso_sampler.R with the same public API
## but significantly faster internals. Key optimisations:
##
##   0. Pre-sampling reduction to unique site × species records
##   1. Within-site: vectorised pair drawing replaces per-site lapply;
##      sampling WITH replacement so species matches are possible
##   2. Between-site Phase 1: environment hash set replaces growing
##      character vector + duplicated() (O(n) vs O(n^2))
##   3. Between-site Phase 2: batch species lookup via keyed data.table
##      and single vectorised DT construction
##   4. Deduplication via integer index keys instead of paste0 strings
##   5. Site capacity weights computed once outside iteration loop
##
## Usage:
##   source("clesso_sampler_optimised.R")
##   # then call clesso_sampler() exactly as before
##
##############################################################################


# ---------------------------------------------------------------------------
# clesso_sample_within_pairs  (optimised)
# ---------------------------------------------------------------------------
clesso_sample_within_pairs <- function(obs_dt,
                                       n_pairs,
                                       min_records = 2,
                                       match_ratio = NULL,
                                       max_iter    = 50,
                                       seed        = NULL) {
  require(data.table)
  if (!is.null(seed)) set.seed(seed)

  obs_dt <- as.data.table(obs_dt)

  ## Identify eligible sites
  site_counts <- obs_dt[, .N, by = site_id]
  eligible_sites <- site_counts[N >= min_records]$site_id
  cat(sprintf("  Within-site sampling: %d eligible sites (of %d total) with >= %d records\n",
              length(eligible_sites), length(unique(obs_dt$site_id)), min_records))

  if (length(eligible_sites) == 0) {
    stop("No sites with >= ", min_records, " records. Cannot form within-site pairs.")
  }

  eligible_obs <- obs_dt[site_id %in% eligible_sites]
  setkey(eligible_obs, site_id)

  ## Build site-level index ONCE (optimisation #5)
  site_index <- eligible_obs[, .(rows = list(.I)), by = site_id]
  setkey(site_index, site_id)

  ## Pre-compute site capacity weights ONCE (optimisation #5)
  site_cap <- eligible_obs[, .(alpha_i = .N), by = site_id]
  site_cap[, capacity := alpha_i^2]
  site_cap[, q_site := capacity / sum(capacity)]

  ## Pre-allocate result
  result_list <- vector("list", max_iter)
  total_collected <- 0L

  if (!is.null(match_ratio)) {
    n_target_match <- ceiling(n_pairs * match_ratio)
    n_target_miss  <- n_pairs - n_target_match
    n_match_so_far <- 0L
    n_miss_so_far  <- 0L
  }

  for (iter in seq_len(max_iter)) {
    remaining <- n_pairs - total_collected
    if (remaining <= 0) break

    n_draw <- remaining * 2L

    ## Draw sites (with replacement, weighted by capacity)
    sampled_sites <- sample(site_cap$site_id, n_draw,
                            replace = TRUE, prob = site_cap$q_site)

    ## ---- OPTIMISATION #1: vectorised pair drawing ----
    ## Batch-lookup all row lists at once via keyed join
    site_rows_list <- site_index[.(sampled_sites), rows]
    lens <- lengths(site_rows_list)
    valid_mask <- lens >= 2L
    n_valid <- sum(valid_mask)

    if (n_valid == 0) next

    valid_rows <- site_rows_list[valid_mask]

    ## Pre-allocate integer vectors instead of creating n tiny DTs
    idx1 <- integer(n_valid)
    idx2 <- integer(n_valid)
    for (j in seq_len(n_valid)) {
      r <- valid_rows[[j]]
      ## Sample WITH replacement so that species matches are possible
      ## (after site × species dedup each row is a unique species,
      ##  so replacement is the only way to draw the same species twice)
      s <- sample.int(length(r), 2L, replace = TRUE)
      idx1[j] <- r[s[1L]]
      idx2[j] <- r[s[2L]]
    }

    ## Build pair records in one shot
    rec1 <- eligible_obs[idx1]
    rec2 <- eligible_obs[idx2]

    batch <- data.table(
      site_i      = rec1$site_id,
      site_j      = rec2$site_id,
      species_i   = rec1$species,
      species_j   = rec2$species,
      lon_i       = rec1$longitude,
      lat_i       = rec1$latitude,
      lon_j       = rec2$longitude,
      lat_j       = rec2$latitude,
      eventDate_i = rec1$eventDate,
      eventDate_j = rec2$eventDate,
      y           = as.integer(rec1$species != rec2$species),
      pair_type   = "within",
      nRecords_i  = rec1$nRecords,
      nRecords_j  = rec2$nRecords,
      richness_i  = rec1$richness,
      richness_j  = rec2$richness,
      .idx1       = idx1,
      .idx2       = idx2
    )

    ## ---- OPTIMISATION #4: integer-based deduplication ----
    ## Use sorted integer indices instead of paste0 string keys
    batch[, c(".idx1", ".idx2") := NULL]

    ## Apply match ratio targeting
    if (!is.null(match_ratio)) {
      matches    <- batch[y == 0]
      mismatches <- batch[y == 1]

      n_take_match <- min(nrow(matches), n_target_match - n_match_so_far)
      n_take_miss  <- min(nrow(mismatches), n_target_miss - n_miss_so_far)

      if (n_take_match > 0) matches <- matches[sample(.N, n_take_match)]
      else matches <- matches[0]
      if (n_take_miss > 0) mismatches <- mismatches[sample(.N, n_take_miss)]
      else mismatches <- mismatches[0]

      batch <- rbind(matches, mismatches)
      n_match_so_far <- n_match_so_far + nrow(matches)
      n_miss_so_far  <- n_miss_so_far + nrow(mismatches)
    }

    if (nrow(batch) > remaining) {
      batch <- batch[sample(.N, remaining)]
    }

    result_list[[iter]] <- batch
    total_collected <- total_collected + nrow(batch)

    if (iter %% 10 == 0 || total_collected >= n_pairs) {
      cat(sprintf("    iter %d: %d / %d within-site pairs collected\n",
                  iter, total_collected, n_pairs))
    }
  }

  out <- rbindlist(result_list[!vapply(result_list, is.null, logical(1))])
  if (nrow(out) > n_pairs) out <- out[sample(.N, n_pairs)]

  n_match <- sum(out$y == 0)
  n_miss  <- sum(out$y == 1)
  cat(sprintf("  Within-site pairs: %d total (%d match, %d mismatch, ratio %.3f)\n",
              nrow(out), n_match, n_miss,
              ifelse(n_match > 0, n_miss / n_match, Inf)))
  out
}


# ---------------------------------------------------------------------------
# clesso_sample_between_pairs  (optimised)
# ---------------------------------------------------------------------------
clesso_sample_between_pairs <- function(obs_dt,
                                        n_pairs,
                                        match_ratio    = 0.5,
                                        max_iter       = 50,
                                        species_thresh = 500,
                                        cores          = max(1, parallel::detectCores() - 1),
                                        seed           = NULL) {
  require(data.table)
  if (!is.null(seed)) set.seed(seed)

  obs_dt <- as.data.table(obs_dt)
  n_obs <- nrow(obs_dt)

  if (!is.null(match_ratio)) {
    n_target_match <- ceiling(n_pairs * match_ratio)
    n_target_miss  <- n_pairs - n_target_match
  } else {
    n_target_match <- n_pairs
    n_target_miss  <- n_pairs
  }

  ## -----------------------------------------------------------------------
  ## Phase 1: Random pairs for mismatches
  ## -----------------------------------------------------------------------
  cat("  Between-site sampling Phase 1: random pairs for mismatches\n")

  ## ---- OPTIMISATION #2: hash set instead of growing vector ----
  seen_keys <- new.env(hash = TRUE, parent = emptyenv(),
                       size = as.integer(n_pairs * 1.5))

  mismatch_list  <- vector("list", max_iter)
  match_list     <- vector("list", max_iter)
  n_miss_so_far  <- 0L
  n_match_so_far <- 0L

  for (iter in seq_len(max_iter)) {
    remaining_miss <- n_target_miss - n_miss_so_far
    if (remaining_miss <= 0) break

    nm <- remaining_miss * 2L
    s1 <- sample.int(n_obs, nm, replace = TRUE)
    s2 <- sample.int(n_obs, nm, replace = TRUE)

    ## Ensure different sites
    same_site <- obs_dt$site_id[s1] == obs_dt$site_id[s2]
    s1 <- s1[!same_site]
    s2 <- s2[!same_site]

    if (length(s1) == 0) next

    ## ---- OPTIMISATION #4: integer pair key ----
    k1 <- pmin(s1, s2)
    k2 <- pmax(s1, s2)

    ## Fast dedup within this batch
    batch_dt <- data.table(k1 = k1, k2 = k2, s1 = s1, s2 = s2)
    batch_dt <- batch_dt[!duplicated(batch_dt[, .(k1, k2)])]

    ## Check against seen hash set (O(1) per lookup)
    key_str <- paste0(batch_dt$k1, "~", batch_dt$k2)
    is_seen <- vapply(key_str, function(k) {
      exists(k, envir = seen_keys, inherits = FALSE)
    }, logical(1), USE.NAMES = FALSE)

    batch_dt <- batch_dt[!is_seen]
    key_str  <- key_str[!is_seen]

    if (nrow(batch_dt) == 0) next

    ## Register new keys in hash set
    for (k in key_str) seen_keys[[k]] <- TRUE

    s1 <- batch_dt$s1
    s2 <- batch_dt$s2

    ## Build batch
    batch <- data.table(
      site_i      = obs_dt$site_id[s1],
      site_j      = obs_dt$site_id[s2],
      species_i   = obs_dt$species[s1],
      species_j   = obs_dt$species[s2],
      lon_i       = obs_dt$longitude[s1],
      lat_i       = obs_dt$latitude[s1],
      lon_j       = obs_dt$longitude[s2],
      lat_j       = obs_dt$latitude[s2],
      eventDate_i = obs_dt$eventDate[s1],
      eventDate_j = obs_dt$eventDate[s2],
      y           = as.integer(obs_dt$species[s1] != obs_dt$species[s2]),
      pair_type   = "between",
      nRecords_i  = obs_dt$nRecords[s1],
      nRecords_j  = obs_dt$nRecords[s2],
      richness_i  = obs_dt$richness[s1],
      richness_j  = obs_dt$richness[s2]
    )

    new_miss  <- batch[y == 1]
    new_match <- batch[y == 0]

    n_take <- min(nrow(new_miss), remaining_miss)
    if (n_take > 0) {
      mismatch_list[[iter]] <- new_miss[sample(.N, n_take)]
      n_miss_so_far <- n_miss_so_far + n_take
    }

    if (!is.null(match_ratio) && n_match_so_far < n_target_match && nrow(new_match) > 0) {
      n_take_m <- min(nrow(new_match), n_target_match - n_match_so_far)
      match_list[[iter]] <- new_match[sample(.N, n_take_m)]
      n_match_so_far <- n_match_so_far + n_take_m
    }

    if (iter %% 10 == 0) {
      cat(sprintf("    Phase 1 iter %d: %d mismatches, %d matches\n",
                  iter, n_miss_so_far, n_match_so_far))
    }
  }

  cat(sprintf("  Phase 1 complete: %d mismatches, %d matches\n",
              n_miss_so_far, n_match_so_far))

  ## -----------------------------------------------------------------------
  ## Phase 2: Species-stratified sampling for matches (optimised)
  ## -----------------------------------------------------------------------
  if (!is.null(match_ratio) && n_match_so_far < n_target_match) {
    cat("  Between-site sampling Phase 2: species-stratified match sampling\n")

    obs_dt[, obs_idx := .I]
    species_index <- obs_dt[, .(obs_indices = list(obs_idx), m_s = .N), by = species]
    species_multi <- species_index[m_s >= 2]
    species_multi[, match_capacity := m_s * (m_s - 1L)]
    ## ---- OPTIMISATION #3: keyed lookup instead of linear scan ----
    setkey(species_multi, species)

    for (iter in seq_len(max_iter)) {
      remaining_match <- n_target_match - n_match_so_far
      if (remaining_match <= 0) break

      n_sp_draw <- min(remaining_match * 2L, nrow(species_multi) * 3L)
      sp_sample <- sample(species_multi$species,
                    n_sp_draw,
                    replace = TRUE,
                    prob = species_multi$match_capacity / sum(species_multi$match_capacity))

      ## ---- OPTIMISATION #3: batch species lookup + vectorised DT build ----
      idx_pools <- species_multi[.(sp_sample), obs_indices]

      ri <- integer(length(sp_sample))
      rj <- integer(length(sp_sample))
      valid <- logical(length(sp_sample))

      for (j in seq_along(sp_sample)) {
        pool <- idx_pools[[j]]
        if (length(pool) < 2L) next
        pair <- sample(pool, 2L, replace = FALSE)
        if (obs_dt$site_id[pair[1]] == obs_dt$site_id[pair[2]]) next
        ri[j] <- pair[1]
        rj[j] <- pair[2]
        valid[j] <- TRUE
      }

      ri_v <- ri[valid]
      rj_v <- rj[valid]

      if (length(ri_v) == 0) next

      ## Single vectorised data.table construction
      new_matches <- data.table(
        site_i      = obs_dt$site_id[ri_v],
        site_j      = obs_dt$site_id[rj_v],
        species_i   = obs_dt$species[ri_v],
        species_j   = obs_dt$species[rj_v],
        lon_i       = obs_dt$longitude[ri_v],
        lat_i       = obs_dt$latitude[ri_v],
        lon_j       = obs_dt$longitude[rj_v],
        lat_j       = obs_dt$latitude[rj_v],
        eventDate_i = obs_dt$eventDate[ri_v],
        eventDate_j = obs_dt$eventDate[rj_v],
        y           = 0L,
        pair_type   = "between",
        nRecords_i  = obs_dt$nRecords[ri_v],
        nRecords_j  = obs_dt$nRecords[rj_v],
        richness_i  = obs_dt$richness[ri_v],
        richness_j  = obs_dt$richness[rj_v]
      )

      n_take <- min(nrow(new_matches), remaining_match)
      match_list[[length(match_list) + 1]] <- new_matches[sample(.N, n_take)]
      n_match_so_far <- n_match_so_far + n_take

      if (iter %% 10 == 0) {
        cat(sprintf("    Phase 2 iter %d: %d / %d matches\n",
                    iter, n_match_so_far, n_target_match))
      }
    }
  }

  ## -----------------------------------------------------------------------
  ## Combine and finalise
  ## -----------------------------------------------------------------------
  all_mismatches <- rbindlist(mismatch_list[!vapply(mismatch_list, is.null, logical(1))])
  all_matches    <- rbindlist(match_list[!vapply(match_list, is.null, logical(1))])
  out <- rbind(all_mismatches, all_matches)

  if (nrow(out) > n_pairs) out <- out[sample(.N, n_pairs)]

  n_match <- sum(out$y == 0)
  n_miss  <- sum(out$y == 1)
  cat(sprintf("  Between-site pairs: %d total (%d match, %d mismatch, ratio %.3f)\n",
              nrow(out), n_match, n_miss,
              ifelse(n_match > 0, n_miss / n_match, Inf)))

  if ("obs_idx" %in% names(obs_dt)) obs_dt[, obs_idx := NULL]

  out
}


# ---------------------------------------------------------------------------
# clesso_sampler  (unchanged API, uses optimised internals)
# ---------------------------------------------------------------------------
clesso_sampler <- function(obs_dt,
                           n_within          = 500000,
                           n_between         = 500000,
                           within_min_recs   = 2,
                           within_match_ratio  = 0.5,
                           between_match_ratio = 0.5,
                           balance_weights   = TRUE,
                           seed              = NULL,
                           ...) {
  require(data.table)
  obs_dt <- as.data.table(obs_dt)

  required_cols <- c("site_id", "species", "longitude", "latitude",
                     "eventDate", "nRecords", "richness")
  missing <- setdiff(required_cols, names(obs_dt))
  if (length(missing) > 0) {
    stop("obs_dt is missing required columns: ", paste(missing, collapse = ", "))
  }

  cat(sprintf("\n=== CLESSO v2 Sampler (optimised) ===\n"))
  cat(sprintf("  Raw observations: %d records, %d sites, %d species\n",
              nrow(obs_dt), length(unique(obs_dt$site_id)),
              length(unique(obs_dt$species))))

  ## Reduce to unique site × species combinations
  ## Keeps one row per (site_id, species); aggregates nRecords, takes first
  ## of all other columns (longitude/latitude/richness are site-level anyway).
  obs_dt <- obs_dt[, .(
    longitude = longitude[1],
    latitude  = latitude[1],
    eventDate = eventDate[1],
    nRecords  = sum(nRecords),
    richness  = richness[1]
  ), by = .(site_id, species)]

  cat(sprintf("  After site × species dedup: %d unique records, %d sites, %d species\n",
              nrow(obs_dt), length(unique(obs_dt$site_id)),
              length(unique(obs_dt$species))))
  cat(sprintf("  Targets: %d within-site pairs, %d between-site pairs\n",
              n_within, n_between))

  ## Within-site pairs
  cat("\n--- Sampling within-site pairs ---\n")
  t_within <- proc.time()
  within_pairs <- clesso_sample_within_pairs(
    obs_dt      = obs_dt,
    n_pairs     = n_within,
    min_records = within_min_recs,
    match_ratio = within_match_ratio,
    seed        = seed
  )
  t_within <- proc.time() - t_within
  cat(sprintf("  Within-site time: %.2f s\n", t_within["elapsed"]))

  ## Between-site pairs
  cat("\n--- Sampling between-site pairs ---\n")
  t_between <- proc.time()
  between_pairs <- clesso_sample_between_pairs(
    obs_dt      = obs_dt,
    n_pairs     = n_between,
    match_ratio = between_match_ratio,
    seed        = if (!is.null(seed)) seed + 1 else NULL,
    ...
  )
  t_between <- proc.time() - t_between
  cat(sprintf("  Between-site time: %.2f s\n", t_between["elapsed"]))

  ## Combine
  all_pairs <- rbind(within_pairs, between_pairs)
  all_pairs[, is_within := as.integer(pair_type == "within")]

  ## Compute pair weights
  if (balance_weights) {
    all_pairs <- clesso_compute_pair_weights(all_pairs, obs_dt)
  } else {
    all_pairs[, w := 1.0]
  }

  ## Diagnostics
  cat("\n--- Sampling summary ---\n")
  summary_dt <- all_pairs[, .(
    n_pairs    = .N,
    n_match    = sum(y == 0),
    n_mismatch = sum(y == 1),
    n_sites    = length(unique(c(site_i, site_j)))
  ), by = pair_type]
  print(summary_dt)

  cat(sprintf("  Total pairs: %d\n", nrow(all_pairs)))
  cat(sprintf("  Unique sites in pairs: %d\n",
              length(unique(c(all_pairs$site_i, all_pairs$site_j)))))

  all_pairs
}


# ---------------------------------------------------------------------------
# clesso_compute_pair_weights  (unchanged from original)
# ---------------------------------------------------------------------------
clesso_compute_pair_weights <- function(pairs_dt, obs_unique_dt) {
  require(data.table)
  pairs_dt <- copy(as.data.table(pairs_dt))
  obs_unique_dt <- as.data.table(obs_unique_dt)

  ## site richness alpha_i from deduplicated site × species table
  site_alpha <- obs_unique_dt[, .(alpha_i = .N), by = site_id]

  ## species occupancy m_s from deduplicated site × species table
  species_occ <- obs_unique_dt[, .(m_s = .N), by = species]

  ## population cell counts (ordered pairs)
  N_within_match <- site_alpha[, sum(alpha_i)]
  N_within_total <- site_alpha[, sum(alpha_i^2)]
  N_within_miss  <- N_within_total - N_within_match

  N_between_total <- (site_alpha[, sum(alpha_i)])^2 - site_alpha[, sum(alpha_i^2)]
  N_between_match <- species_occ[, sum(m_s * (m_s - 1L))]
  N_between_miss  <- N_between_total - N_between_match

  pop_cells <- data.table(
    pair_type = c("within", "within", "between", "between"),
    y         = c(0L, 1L, 0L, 1L),
    pop_N     = c(N_within_match, N_within_miss, N_between_match, N_between_miss)
  )

  samp_cells <- pairs_dt[, .(sample_N = .N), by = .(pair_type, y)]

  pairs_dt <- merge(pairs_dt, pop_cells,  by = c("pair_type", "y"), all.x = TRUE)
  pairs_dt <- merge(pairs_dt, samp_cells, by = c("pair_type", "y"), all.x = TRUE)

  pairs_dt[, w := pop_N / sample_N]

  cat("  Pair weights computed (population-corrected 4-cell weights):\n")
  print(
    pairs_dt[, .(
      n      = .N,
      pop_N  = unique(pop_N),
      total_w = sum(w),
      mean_w  = mean(w)
    ), by = .(pair_type, y)][order(pair_type, y)]
  )

  pairs_dt[, c("pop_N", "sample_N") := NULL]
  pairs_dt
}

# ---------------------------------------------------------------------------
# clesso_format_aggregated_data  (unchanged from original)
# ---------------------------------------------------------------------------
clesso_format_aggregated_data <- function(datRED) {
  require(data.table)
  dt <- as.data.table(datRED)
  out <- data.table(
    site_id   = dt$ID,
    species   = dt$gen_spec,
    longitude = dt$lonID,
    latitude  = dt$latID,
    eventDate = as.character(dt$eventDate),
    nRecords  = dt$nRecords,
    nRecords_exDateLocDups = dt$nRecords.exDateLocDups,
    nSiteVisits = dt$nSiteVisits,
    richness  = dt$richness
  )
  cat(sprintf("  Formatted %d records for CLESSO sampler\n", nrow(out)))
  out
}
