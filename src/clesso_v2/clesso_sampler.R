##############################################################################
##
## clesso_sampler.R — Observation-Pair Sampler for CLESSO v2
##
## Samples both WITHIN-site and BETWEEN-site observation pairs to
## support joint estimation of alpha (richness) and beta (turnover).
##
## The key extension over the original obsPairSampler is the addition
## of within-site pairs, which isolate alpha (because S_{i,i} = 1 when
## i == j). Between-site pairs continue to inform turnover (beta),
## modulated by alpha at each site.
##
## Functions:
##   clesso_sample_within_pairs()  — within-site pair sampling
##   clesso_sample_between_pairs() — between-site pair sampling
##   clesso_sampler()              — main orchestrator
##
## See main.tex for the full mathematical description.
##
##############################################################################

# ---------------------------------------------------------------------------
# clesso_sample_within_pairs
#
# Sample observation pairs from the SAME site. These pairs identify
# alpha (richness) because when i == j, turnover S = 1 and:
#   p_{i,i} = 1 / alpha_i
#
# Parameters:
#   obs_dt      - data.table of observation records with columns:
#                 site_id, species, longitude, latitude, eventDate, ...
#   n_pairs     - total number of within-site pairs to sample
#   min_records - minimum records at a site to be eligible for within-
#                 site sampling (default: 2)
#   match_ratio - target proportion of match pairs (0-1). If NULL,
#                 natural sampling proportions are used. Setting e.g.
#                 0.5 aims for equal matches/mismatches.
#   max_iter    - maximum sampling iterations to fill quota (default: 50)
#   seed        - optional random seed for reproducibility
#
# Returns:
#   data.table with columns:
#     site_i, site_j  - site IDs (equal for within-site)
#     species_i, species_j - species drawn at each side of pair
#     lon_i, lat_i, lon_j, lat_j - coordinates
#     eventDate_i, eventDate_j - observation dates
#     y               - 0 if match, 1 if mismatch
#     pair_type       - "within"
#     nRecords_i, nRecords_j - record counts at each site
#     richness_i, richness_j - observed richness at each site
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

  ## Identify eligible sites (those with >= min_records observations)
  site_counts <- obs_dt[, .N, by = site_id]
  eligible_sites <- site_counts[N >= min_records]$site_id
  cat(sprintf("  Within-site sampling: %d eligible sites (of %d total) with >= %d records\n",
              length(eligible_sites), length(unique(obs_dt$site_id)), min_records))

  if (length(eligible_sites) == 0) {
    stop("No sites with >= ", min_records, " records. Cannot form within-site pairs.")
  }

  ## Subset to eligible sites and index by site_id for fast lookup
  eligible_obs <- obs_dt[site_id %in% eligible_sites]
  setkey(eligible_obs, site_id)

  ## Build site-level index: list of row indices per site
  site_index <- eligible_obs[, .(rows = list(.I)), by = site_id]
  setkey(site_index, site_id)

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

    ## Over-sample to account for natural match/mismatch proportions
    n_draw <- remaining * 2L  # draw more than needed, filter later

    ## Sample sites proportional to their pair capacity: n*(n-1)/2
    site_pair_cap <- eligible_obs[, .(.N), by = site_id]
    site_pair_cap[, capacity := N * (N - 1L) / 2]
    ## Normalise to sampling weights
    site_pair_cap[, weight := capacity / sum(capacity)]

    ## Draw sites (with replacement, weighted by capacity)
    sampled_sites <- sample(site_pair_cap$site_id, n_draw,
                            replace = TRUE, prob = site_pair_cap$weight)

    ## For each sampled site, draw 2 records (without replacement within site)
    pairs_dt <- rbindlist(lapply(sampled_sites, function(sid) {
      rows <- site_index[.(sid), rows][[1]]
      if (length(rows) < 2) return(NULL)
      idx <- sample(rows, 2, replace = FALSE)
      data.table(idx1 = idx[1], idx2 = idx[2])
    }))

    if (nrow(pairs_dt) == 0) next

    ## Build pair records
    rec1 <- eligible_obs[pairs_dt$idx1]
    rec2 <- eligible_obs[pairs_dt$idx2]

    batch <- data.table(
      site_i     = rec1$site_id,
      site_j     = rec2$site_id,
      species_i  = rec1$species,
      species_j  = rec2$species,
      lon_i      = rec1$longitude,
      lat_i      = rec1$latitude,
      lon_j      = rec2$longitude,
      lat_j      = rec2$latitude,
      eventDate_i = rec1$eventDate,
      eventDate_j = rec2$eventDate,
      y          = as.integer(rec1$species != rec2$species),
      pair_type  = "within",
      nRecords_i = rec1$nRecords,
      nRecords_j = rec2$nRecords,
      richness_i = rec1$richness,
      richness_j = rec2$richness
    )

    ## Remove duplicate pairs (order-invariant key: site + sorted species + sorted dates)
    batch[, pair_key := paste0(
      site_i, ":",
      pmin(as.character(species_i), as.character(species_j)), ":",
      pmax(as.character(species_i), as.character(species_j)), ":",
      pmin(as.character(eventDate_i), as.character(eventDate_j)), ":",
      pmax(as.character(eventDate_i), as.character(eventDate_j))
    )]
    batch <- batch[!duplicated(pair_key)]
    batch[, pair_key := NULL]

    ## Apply match ratio targeting if requested
    if (!is.null(match_ratio)) {
      matches    <- batch[y == 0]
      mismatches <- batch[y == 1]

      ## Take up to the remaining quota for each type
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

    ## Cap at remaining
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
# clesso_sample_between_pairs
#
# Sample observation pairs from DIFFERENT sites. These inform turnover
# (beta), modulated by alpha at each site:
#   p_{i,j} = S_{i,j} * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
#
# This is conceptually similar to the original obsPairSampler but
# produces output in the new clesso format.
#
# Parameters:
#   obs_dt         - data.table of observation records (same as above)
#   n_pairs        - total number of between-site pairs to sample
#   match_ratio    - target proportion of match pairs (0-1). Default 0.5
#                    for balanced sampling. NULL = natural proportions.
#   max_iter       - maximum sampling iterations (default: 50)
#   species_thresh - species count above which parallel matching is chunked
#   cores          - number of cores for parallel match sampling
#   seed           - optional random seed
#
# Returns:
#   data.table with same schema as clesso_sample_within_pairs output
#   but pair_type = "between"
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

  ## Target counts
  if (!is.null(match_ratio)) {
    n_target_match <- ceiling(n_pairs * match_ratio)
    n_target_miss  <- n_pairs - n_target_match
  } else {
    n_target_match <- n_pairs  # will fill naturally
    n_target_miss  <- n_pairs
  }

  ## -----------------------------------------------------------------------
  ## Phase 1: Random pairs — fills mismatches (most random pairs mismatch)
  ## -----------------------------------------------------------------------
  cat("  Between-site sampling Phase 1: random pairs for mismatches\n")

  attempted_keys <- character(0)
  mismatch_list  <- vector("list", max_iter)
  match_list     <- vector("list", max_iter)
  n_miss_so_far  <- 0L
  n_match_so_far <- 0L

  for (iter in seq_len(max_iter)) {
    remaining_miss <- n_target_miss - n_miss_so_far
    if (remaining_miss <= 0) break

    nm <- remaining_miss * 2L  # oversample
    s1 <- sample.int(n_obs, nm, replace = TRUE)
    s2 <- sample.int(n_obs, nm, replace = TRUE)

    ## Ensure different sites
    same_site <- obs_dt$site_id[s1] == obs_dt$site_id[s2]
    s1 <- s1[!same_site]
    s2 <- s2[!same_site]

    if (length(s1) == 0) next

    ## Create order-invariant key and deduplicate
    key <- fifelse(s1 <= s2,
                   paste(s1, s2, sep = "~"),
                   paste(s2, s1, sep = "~"))

    all_keys <- c(attempted_keys, key)
    is_dup <- duplicated(all_keys)
    is_dup <- is_dup[(length(attempted_keys) + 1):length(all_keys)]

    s1  <- s1[!is_dup]
    s2  <- s2[!is_dup]
    key <- key[!is_dup]
    attempted_keys <- c(attempted_keys, key)

    if (length(s1) == 0) next

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

    ## Split into matches and mismatches
    new_miss  <- batch[y == 1]
    new_match <- batch[y == 0]

    ## Keep mismatches up to quota
    n_take <- min(nrow(new_miss), remaining_miss)
    if (n_take > 0) {
      mismatch_list[[iter]] <- new_miss[sample(.N, n_take)]
      n_miss_so_far <- n_miss_so_far + n_take
    }

    ## Opportunistically collect matches from random sampling
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
  ## Phase 2: Species-stratified sampling for matches
  ## (same species at different sites)
  ## -----------------------------------------------------------------------
  if (!is.null(match_ratio) && n_match_so_far < n_target_match) {
    cat("  Between-site sampling Phase 2: species-stratified match sampling\n")

    ## Index observations by species
    obs_dt[, obs_idx := .I]
    species_index <- obs_dt[, .(obs_indices = list(obs_idx), n = .N), by = species]
    ## Only species observed at >= 2 different sites can produce matches
    species_multi <- species_index[n >= 2]

    for (iter in seq_len(max_iter)) {
      remaining_match <- n_target_match - n_match_so_far
      if (remaining_match <= 0) break

      ## Sample species proportional to number of observations
      sp_sample <- sample(species_multi$species,
                          min(remaining_match * 2L, nrow(species_multi)),
                          replace = TRUE,
                          prob = species_multi$n / sum(species_multi$n))

      match_batch_list <- lapply(sp_sample, function(sp) {
        idx_pool <- species_multi[species == sp, obs_indices][[1]]
        if (length(idx_pool) < 2) return(NULL)

        pair_idx <- sample(idx_pool, 2, replace = FALSE)
        ri <- pair_idx[1]
        rj <- pair_idx[2]

        ## Ensure different sites
        if (obs_dt$site_id[ri] == obs_dt$site_id[rj]) return(NULL)

        data.table(
          site_i      = obs_dt$site_id[ri],
          site_j      = obs_dt$site_id[rj],
          species_i   = obs_dt$species[ri],
          species_j   = obs_dt$species[rj],
          lon_i       = obs_dt$longitude[ri],
          lat_i       = obs_dt$latitude[ri],
          lon_j       = obs_dt$longitude[rj],
          lat_j       = obs_dt$latitude[rj],
          eventDate_i = obs_dt$eventDate[ri],
          eventDate_j = obs_dt$eventDate[rj],
          y           = 0L,
          pair_type   = "between",
          nRecords_i  = obs_dt$nRecords[ri],
          nRecords_j  = obs_dt$nRecords[rj],
          richness_i  = obs_dt$richness[ri],
          richness_j  = obs_dt$richness[rj]
        )
      })

      new_matches <- rbindlist(match_batch_list[!vapply(match_batch_list, is.null, logical(1))])
      if (nrow(new_matches) == 0) next

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

  ## Clean up temp column
  if ("obs_idx" %in% names(obs_dt)) obs_dt[, obs_idx := NULL]

  out
}


# ---------------------------------------------------------------------------
# clesso_sampler — Main entry point
#
# Orchestrates both within-site and between-site sampling, combining
# results into a single paired dataset ready for clesso_prepare_model_data().
#
# Parameters:
#   obs_dt           - data.table of aggregated observations. Required columns:
#                        site_id   - unique site identifier (e.g. "151.05:-33.85")
#                        species   - species name or ID
#                        longitude - site longitude
#                        latitude  - site latitude
#                        eventDate - observation date (character or Date)
#                        nRecords  - total records at the site
#                        richness  - observed species richness at the site
#   n_within         - number of within-site pairs to sample
#   n_between        - number of between-site pairs to sample
#   within_min_recs  - minimum records at a site for within-site eligibility
#   within_match_ratio - target match/total ratio for within pairs (NULL = natural)
#   between_match_ratio - target match/total ratio for between pairs (0.5 = balanced)
#   balance_weights  - if TRUE, compute inverse-frequency weights so that
#                      within and between contributions are balanced in the
#                      likelihood (prevents one type from dominating)
#   seed             - optional random seed
#   ...              - additional args passed to between-site sampler
#
# Returns:
#   data.table combining within and between pairs with columns:
#     site_i, site_j, species_i, species_j,
#     lon_i, lat_i, lon_j, lat_j,
#     eventDate_i, eventDate_j,
#     y (0=match, 1=mismatch),
#     pair_type ("within" or "between"),
#     is_within (1 or 0),
#     w (pair weight),
#     nRecords_i, nRecords_j,
#     richness_i, richness_j
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

  ## Validate required columns
  required_cols <- c("site_id", "species", "longitude", "latitude",
                     "eventDate", "nRecords", "richness")
  missing <- setdiff(required_cols, names(obs_dt))
  if (length(missing) > 0) {
    stop("obs_dt is missing required columns: ", paste(missing, collapse = ", "))
  }

  cat(sprintf("\n=== CLESSO v2 Sampler ===\n"))
  cat(sprintf("  Observations: %d records, %d sites, %d species\n",
              nrow(obs_dt), length(unique(obs_dt$site_id)),
              length(unique(obs_dt$species))))
  cat(sprintf("  Targets: %d within-site pairs, %d between-site pairs\n",
              n_within, n_between))

  ## -----------------------------------------------------------------------
  ## Sample within-site pairs
  ## -----------------------------------------------------------------------
  cat("\n--- Sampling within-site pairs ---\n")
  within_pairs <- clesso_sample_within_pairs(
    obs_dt      = obs_dt,
    n_pairs     = n_within,
    min_records = within_min_recs,
    match_ratio = within_match_ratio,
    seed        = seed
  )

  ## -----------------------------------------------------------------------
  ## Sample between-site pairs
  ## -----------------------------------------------------------------------
  cat("\n--- Sampling between-site pairs ---\n")
  between_pairs <- clesso_sample_between_pairs(
    obs_dt      = obs_dt,
    n_pairs     = n_between,
    match_ratio = between_match_ratio,
    seed        = if (!is.null(seed)) seed + 1 else NULL,
    ...
  )

  ## -----------------------------------------------------------------------
  ## Combine
  ## -----------------------------------------------------------------------
  all_pairs <- rbind(within_pairs, between_pairs)
  all_pairs[, is_within := as.integer(pair_type == "within")]

  ## -----------------------------------------------------------------------
  ## Compute pair weights
  ## -----------------------------------------------------------------------
  if (balance_weights) {
    all_pairs <- clesso_compute_pair_weights(all_pairs)
  } else {
    all_pairs[, w := 1.0]
  }

  ## -----------------------------------------------------------------------
  ## Diagnostics
  ## -----------------------------------------------------------------------
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
# clesso_compute_pair_weights
#
# Compute weights that balance the likelihood contributions from:
#   1. Within vs between pairs (so neither dominates)
#   2. Match vs mismatch within each type (prevents class imbalance)
#
# The weighting scheme gives equal total weight to within and between
# groups, and within each group, equal total weight to matches and
# mismatches. Individual pair weights are then 1/n for that cell.
#
# Parameters:
#   pairs_dt - data.table with columns: pair_type, y
#
# Returns:
#   pairs_dt with added column 'w'
# ---------------------------------------------------------------------------
clesso_compute_pair_weights <- function(pairs_dt) {
  pairs_dt <- copy(pairs_dt)

  ## 4-cell design: (within/between) x (match/mismatch)
  ## Each cell gets equal total weight of 1.0, so total weight = 4.0
  ## Individual weight = 1.0 / count_in_cell
  pairs_dt[, cell := paste0(pair_type, "_", y)]
  cell_counts <- pairs_dt[, .N, by = cell]

  ## Target: each cell contributes equally
  target_weight_per_cell <- 1.0

  pairs_dt[cell_counts, on = "cell",
           w := target_weight_per_cell / N]

  cat(sprintf("  Pair weights computed (4-cell balancing):\n"))
  weight_summary <- pairs_dt[, .(n = .N, total_w = sum(w), mean_w = mean(w)),
                              by = .(pair_type, y)]
  setorder(weight_summary, pair_type, y)
  print(weight_summary)

  pairs_dt[, cell := NULL]
  pairs_dt
}


# ---------------------------------------------------------------------------
# clesso_format_aggregated_data
#
# Convert the output of siteAggregator (shared/R/site_aggregator.R) to
# the data.table format expected by clesso_sampler(). This is a
# convenience wrapper so existing pipelines can feed into the new sampler.
#
# Parameters:
#   datRED - data.frame from siteAggregator()
#
# Returns:
#   data.table with standardised column names for clesso_sampler()
# ---------------------------------------------------------------------------
clesso_format_aggregated_data <- function(datRED) {
  require(data.table)
  dt <- as.data.table(datRED)

  ## Map column names from siteAggregator output to clesso schema
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
