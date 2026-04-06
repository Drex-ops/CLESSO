##############################################################################
##
## clesso_sampler_optimised.R -- Hex-grid spatially balanced sampler
##
## Replaces capacity-weighted sampling with a hex-grid balanced scheme:
##
##   1. A ~2deg hexagonal grid is overlaid on all site locations.
##   2. Within-site pairs: equal quota per hex cell; within each hex,
##      sites sampled proportional to alpha^2 (preserves local signal).
##   3. Between-site pairs: 50% within-hex (equal quota per hex) +
##      50% between-hex (hex-pairs sampled uniformly, one site from each).
##   4. Inverse-probability weights (IPW) correct for the spatial bias
##      introduced by hex balancing, so the loss landscape remains
##      population-correct.
##
## Usage:
##   source("clesso_sampler_optimised.R")
##   # then call clesso_sampler() exactly as before
##
##############################################################################


# ---------------------------------------------------------------------------
# assign_hex_grid -- Create hex grid and assign sites to hex cells
# ---------------------------------------------------------------------------
#' @param site_dt data.table with columns site_id, longitude, latitude
#'                (one row per unique site)
#' @param hex_size numeric, approximate hex diameter in degrees (default 2.0)
#' @param return_grid logical, if TRUE return a list with $assign (data.table)
#'                    and $hex_grid (sf object with geometry for every occupied hex)
#' @return data.table with columns site_id, hex_id (or list if return_grid=TRUE)
assign_hex_grid <- function(site_dt, hex_size = 2.0, return_grid = FALSE) {
  require(sf)
  require(data.table)

  ## Build point geometry for every unique site
  pts <- st_as_sf(site_dt, coords = c("longitude", "latitude"), crs = 4326)

  ## Bounding box expanded slightly so edge sites are captured
  bb <- st_bbox(pts)
  bb["xmin"] <- bb["xmin"] - hex_size
  bb["xmax"] <- bb["xmax"] + hex_size
  bb["ymin"] <- bb["ymin"] - hex_size
  bb["ymax"] <- bb["ymax"] + hex_size

  ## Create hex grid over bbox  (square = FALSE -> hexagons)
  hex_grid <- st_make_grid(
    st_as_sfc(bb, crs = 4326),
    cellsize = hex_size,
    square   = FALSE
  )
  hex_grid <- st_sf(hex_id = seq_along(hex_grid), geometry = hex_grid)

  ## Spatial join: which hex contains each point?
  joined <- st_join(pts, hex_grid, join = st_within)

  out <- data.table(
    site_id = joined$site_id,
    hex_id  = joined$hex_id
  )

  ## Remove any NAs (points outside grid -- shouldn't happen with expansion)
  out <- out[!is.na(hex_id)]

  ## Keep only hexes with at least one site
  occupied_ids <- unique(out$hex_id)
  n_hex <- length(occupied_ids)
  cat(sprintf("  Hex grid: %d occupied hexes (size %.1f deg)\n", n_hex, hex_size))

  if (return_grid) {
    hex_grid_occupied <- hex_grid[hex_grid$hex_id %in% occupied_ids, ]
    return(list(assign = out, hex_grid = hex_grid_occupied))
  }
  out
}


# ---------------------------------------------------------------------------
# clesso_sample_within_pairs  (hex-balanced)
# ---------------------------------------------------------------------------
clesso_sample_within_pairs <- function(obs_dt,
                                       n_pairs,
                                       min_records   = 2,
                                       match_ratio   = 0.5,
                                       max_iter      = 50,
                                       hex_assign    = NULL,
                                       seed          = NULL,
                                       design_w_cap_pctl = 0.99) {
  require(data.table)
  if (!is.null(seed)) set.seed(seed)

  obs_dt <- as.data.table(obs_dt)
  ## Factor-proof: ensure character site_id and species for name-based lookup
  if (is.factor(obs_dt$site_id)) obs_dt[, site_id := as.character(site_id)]
  if (is.factor(obs_dt$species)) obs_dt[, species := as.character(species)]

  ## ---- Site-level stats ----
  site_stats <- obs_dt[, .(m_i = .N, n_recs = sum(nRecords)),
                       by = .(site_id, longitude, latitude)]
  site_stats <- site_stats[m_i >= min_records]

  if (nrow(site_stats) == 0) {
    warning("No sites with >= min_records species. Returning empty.")
    return(data.table())
  }

  ## Merge hex assignments
  if (!is.null(hex_assign)) {
    site_stats <- merge(site_stats, hex_assign, by = "site_id", all.x = TRUE)
    ## Sites without hex assignment (shouldn't happen) get dropped
    n_no_hex <- sum(is.na(site_stats$hex_id))
    if (n_no_hex > 0) {
      cat(sprintf("  Warning: %d sites without hex assignment dropped\n", n_no_hex))
      site_stats <- site_stats[!is.na(hex_id)]
    }
  } else {
    ## Fallback: single hex (original behaviour)
    site_stats[, hex_id := 1L]
  }

  ## ---- Capacity per site: m_i^2 (number of ordered species-pair draws) ----
  ## NOTE: m_i is the observed species count (an observable quantity),
  ## not a model-predicted alpha.  Using m_i^2 as capacity means high-
  ## richness sites contribute more within-site pairs per draw.
  site_stats[, capacity := as.numeric(m_i)^2]

  ## ---- Hex quotas ----
  hex_summary <- site_stats[, .(
    n_sites    = .N,
    tot_cap    = sum(capacity)
  ), by = hex_id]

  n_hex <- nrow(hex_summary)
  base_quota <- ceiling(n_pairs / n_hex)
  hex_summary[, quota := base_quota]

  ## Cap quotas for hexes that can't fill them
  ## Max possible within-site pairs for a hex ~ sum(alpha_i^2) across its sites
  ## Use a generous multiple since we sample with replacement
  hex_summary[, max_pairs := as.integer(pmin(tot_cap * 2, .Machine$integer.max / 2))]
  hex_summary[, quota := pmin(quota, max_pairs)]

  ## Redistribute surplus from capped hexes
  total_assigned <- sum(hex_summary$quota)
  deficit <- n_pairs - total_assigned
  if (deficit > 0) {
    uncapped <- hex_summary[quota == base_quota]
    if (nrow(uncapped) > 0) {
      extra <- ceiling(deficit / nrow(uncapped))
      hex_summary[quota == base_quota, quota := quota + extra]
    }
  }

  cat(sprintf("  Within-site: %d hexes, base quota %d pairs/hex\n",
              n_hex, base_quota))

  ## ---- Match/mismatch targets ----
  if (!is.null(match_ratio)) {
    target_match_frac <- match_ratio
  } else {
    target_match_frac <- 0.5
  }

  ## ---- Pre-build ALL per-site obs pools (once, not per hex) ----
  obs_dt[, obs_row := .I]

  ## Merge hex info onto obs_dt for fast hex-site lookup
  obs_hex <- merge(
    obs_dt[, .(obs_row, site_id)],
    hex_assign[, .(site_id, hex_id)],
    by = "site_id", all.x = TRUE
  )
  obs_hex <- obs_hex[!is.na(hex_id)]

  ## Per-site pools (global, built once)
  all_site_pools <- split(obs_dt$obs_row, obs_dt$site_id)
  all_pool_sizes <- vapply(all_site_pools, length, 0L)

  ## Per-hex obs row lookup (list of obs_row vectors keyed by hex_id)
  hex_obs_rows <- split(obs_hex$obs_row, obs_hex$hex_id)

  ## ---- Sample pairs per hex ----
  ## Also track total drawn match/mismatch (pre-quota) for retention rates
  all_hex_pairs <- vector("list", n_hex)
  total_drawn_match <- 0L
  total_drawn_mismatch <- 0L

  for (h_idx in seq_len(n_hex)) {
    if (h_idx %% 100 == 0 || h_idx == n_hex) {
      cat(sprintf("    Within-site progress: %d / %d hexes\n", h_idx, n_hex))
    }
    hid    <- hex_summary$hex_id[h_idx]
    h_quota <- hex_summary$quota[h_idx]
    if (h_quota <= 0) next

    ## Sites in this hex (already computed in site_stats)
    h_sites <- site_stats[hex_id == hid]
    if (nrow(h_sites) == 0) next

    ## Filter to eligible sites (>= 2 obs)
    eligible_sites <- h_sites$site_id[all_pool_sizes[h_sites$site_id] >= 2L]
    if (length(eligible_sites) == 0) next

    ## Site sampling weights (capacity = m_i^2) for eligible sites
    h_site_wt <- h_sites[site_id %in% eligible_sites, .(site_id, capacity)]
    h_site_wt[, prob := capacity / sum(capacity)]

    pool_sizes_h <- all_pool_sizes[eligible_sites]

    ## Target match/mismatch counts for this hex
    n_match_target <- ceiling(h_quota * target_match_frac)
    n_miss_target  <- h_quota - n_match_target
    n_match_got    <- 0L
    n_miss_got     <- 0L

    pair_list <- vector("list", 10L)

    for (iter in seq_len(10L)) {
      remaining <- h_quota - n_match_got - n_miss_got
      if (remaining <= 0) break

      ## Draw sites with replacement, weighted by capacity
      n_draw <- min(remaining * 3L, 20000L)
      drawn_site_idx <- sample.int(nrow(h_site_wt), n_draw,
                                    replace = TRUE, prob = h_site_wt$prob)
      drawn_site_ids <- h_site_wt$site_id[drawn_site_idx]

      ## For each drawn site, pick 2 obs from its pool
      idx1 <- integer(n_draw)
      idx2 <- integer(n_draw)

      ## Vectorised random offsets within each site's pool
      ps <- pool_sizes_h[drawn_site_ids]
      r1 <- sample.int(1000000L, n_draw, replace = TRUE) %% ps + 1L
      r2 <- sample.int(1000000L, n_draw, replace = TRUE) %% ps + 1L

      ## Look up actual obs_row per site (grouped for efficiency)
      for (sid in unique(drawn_site_ids)) {
        pool <- all_site_pools[[sid]]
        mask <- drawn_site_ids == sid
        idx1[mask] <- pool[r1[mask]]
        idx2[mask] <- pool[r2[mask]]
      }

      ## Build batch using global obs_dt row indices
      batch <- data.table(
        site_i      = obs_dt$site_id[idx1],
        site_j      = obs_dt$site_id[idx2],
        species_i   = obs_dt$species[idx1],
        species_j   = obs_dt$species[idx2],
        lon_i       = obs_dt$longitude[idx1],
        lat_i       = obs_dt$latitude[idx1],
        lon_j       = obs_dt$longitude[idx2],
        lat_j       = obs_dt$latitude[idx2],
        eventDate_i = obs_dt$eventDate[idx1],
        eventDate_j = obs_dt$eventDate[idx2],
        y           = as.integer(obs_dt$species[idx1] != obs_dt$species[idx2]),
        pair_type   = "within",
        nRecords_i  = obs_dt$nRecords[idx1],
        nRecords_j  = obs_dt$nRecords[idx2],
        richness_i  = obs_dt$richness[idx1],
        richness_j  = obs_dt$richness[idx2],
        hex_id      = hid
      )

      ## Split by match/mismatch and respect quotas
      ## Track all drawn counts for retention rate computation
      matches    <- batch[y == 0]
      mismatches <- batch[y == 1]
      total_drawn_match    <- total_drawn_match    + nrow(matches)
      total_drawn_mismatch <- total_drawn_mismatch + nrow(mismatches)

      n_take_match <- min(nrow(matches), n_match_target - n_match_got)
      n_take_miss  <- min(nrow(mismatches), n_miss_target - n_miss_got)

      keep <- rbind(
        if (n_take_match > 0) matches[sample(.N, n_take_match)] else NULL,
        if (n_take_miss > 0)  mismatches[sample(.N, n_take_miss)] else NULL
      )

      if (!is.null(keep) && nrow(keep) > 0) {
        pair_list[[iter]] <- keep
        n_match_got <- n_match_got + n_take_match
        n_miss_got  <- n_miss_got + n_take_miss
      }
    }

    hex_pairs <- rbindlist(pair_list[!vapply(pair_list, is.null, logical(1))])
    if (nrow(hex_pairs) > 0) {
      ## Add per-pair site capacity for design weight computation.
      ## The proposal draws site i from hex h with prob m_i^2 / sum_h(m_j^2).
      ## Target G_W(i) = 1/H * 1/n_h (uniform hex, uniform site).
      ## Design weight a_W(i) = G_W(i) / q_W(i) = sum_h(m_j^2) / (n_h * m_i^2)
      ## where n_h = number of eligible sites in this hex.
      n_h_eligible <- length(eligible_sites)
      hex_total_cap <- sum(h_site_wt$capacity)
      ## Look up m_i² for the drawn site (site_i = site_j for within-site pairs)
      m_i_sq <- h_site_wt$capacity[match(hex_pairs$site_i, h_site_wt$site_id)]
      hex_pairs[, design_w := hex_total_cap / (n_h_eligible * m_i_sq)]
      all_hex_pairs[[h_idx]] <- hex_pairs
    }
  }

  ## Clean up temp column
  if ("obs_row" %in% names(obs_dt)) obs_dt[, obs_row := NULL]

  out <- rbindlist(all_hex_pairs[!vapply(all_hex_pairs, is.null, logical(1))])

  if (nrow(out) > n_pairs) out <- out[sample(.N, n_pairs)]

  ## Add stratum index: 0 = within-site
  out[, stratum := 0L]

  ## Normalise design weights so mean = 1 within this stratum
  out[, design_w := design_w / mean(design_w, na.rm = TRUE)]

  ## ---- Winsorise design weights at the chosen percentile ----
  if (!is.null(design_w_cap_pctl) && design_w_cap_pctl < 1.0) {
    cap_val <- quantile(out$design_w, probs = design_w_cap_pctl, na.rm = TRUE)
    n_capped <- sum(out$design_w > cap_val, na.rm = TRUE)
    pre_max  <- max(out$design_w, na.rm = TRUE)
    out[design_w > cap_val, design_w := cap_val]
    out[, design_w := design_w / mean(design_w, na.rm = TRUE)]   # re-normalise
    ess_post <- sum(out$design_w)^2 / sum(out$design_w^2)
    cat(sprintf("  Within-site design_w truncation (p%.1f): cap=%.3f, capped %d pairs (%.1f%%), pre-max=%.1f, post-ESS=%.0f/%.0f (%.1f%%)\n",
                design_w_cap_pctl * 100, cap_val, n_capped,
                100 * n_capped / nrow(out), pre_max,
                ess_post, nrow(out), 100 * ess_post / nrow(out)))
  }

  ## Compute retention rates for class-conditioned (50/50) sampling
  ## r_s_0 = P(retain | Y=0, W), r_s_1 = P(retain | Y=1, W)
  n_match <- sum(out$y == 0)
  n_miss  <- sum(out$y == 1)
  r_W_0 <- if (total_drawn_match > 0) n_match / total_drawn_match else 1.0
  r_W_1 <- if (total_drawn_mismatch > 0) n_miss / total_drawn_mismatch else 1.0
  attr(out, "retention_rates") <- list(r_0 = r_W_0, r_1 = r_W_1,
                                        drawn_match = total_drawn_match,
                                        drawn_mismatch = total_drawn_mismatch)

  cat(sprintf("  Within-site pairs: %d total (%d match, %d mismatch, ratio %.3f)\n",
              nrow(out), n_match, n_miss,
              ifelse(n_match > 0, n_miss / n_match, Inf)))
  cat(sprintf("  Within-site hex usage: %d / %d hexes contributed pairs\n",
              length(unique(out$hex_id)), n_hex))
  cat(sprintf("  Within-site retention rates: r_0=%.4f, r_1=%.4f (drawn: %d match, %d mismatch)\n",
              r_W_0, r_W_1, total_drawn_match, total_drawn_mismatch))
  cat(sprintf("  Within-site design_w: mean=%.3f, min=%.3f, max=%.3f\n",
              mean(out$design_w), min(out$design_w), max(out$design_w)))

  out
}


# ---------------------------------------------------------------------------
# clesso_sample_between_pairs  (hex-balanced, 3-tier distance)
# ---------------------------------------------------------------------------
#' Sample between-site pairs using a 3-tier hex distance scheme:
#'   Tier 1 (same hex):      pairs where both sites share the same hex
#'   Tier 2 (neighbour hex): pairs where sites are in adjacent hexes
#'   Tier 3 (distant hex):   pairs from any non-adjacent hex combination
#'
#' @param obs_dt         data.table of observations (site x species)
#' @param n_pairs        total number of between-site pairs to draw
#' @param match_ratio    target fraction of species-match pairs (y==0)
#' @param max_iter       max sampling iterations per tier
#' @param species_thresh unused (kept for API compatibility)
#' @param cores          unused (kept for API compatibility)
#' @param hex_assign     data.table(site_id, hex_id)
#' @param hex_grid       sf object with hex polygons (needed for neighbor lookup)
#' @param frac_same      fraction of pairs from same hex      (default 0.5)
#' @param frac_neighbour fraction of pairs from neighbour hexes (default 0.3)
#' @param frac_distant   fraction of pairs from distant hexes   (default 0.2)
#' @param seed           random seed
clesso_sample_between_pairs <- function(obs_dt,
                                        n_pairs,
                                        match_ratio    = 0.5,
                                        max_iter       = 50,
                                        species_thresh = 500,
                                        cores          = max(1, parallel::detectCores() - 1),
                                        hex_assign     = NULL,
                                        hex_grid       = NULL,
                                        frac_same      = 0.5,
                                        frac_neighbour = 0.3,
                                        frac_distant   = 0.2,
                                        seed           = NULL,
                                        design_w_cap_pctl = 0.99) {
  require(data.table)
  if (!is.null(seed)) set.seed(seed)

  obs_dt <- as.data.table(obs_dt)
  ## Factor-proof: ensure character site_id and species for name-based lookup
  if (is.factor(obs_dt$site_id)) obs_dt[, site_id := as.character(site_id)]
  if (is.factor(obs_dt$species)) obs_dt[, species := as.character(species)]

  ## Merge hex assignments
  if (!is.null(hex_assign)) {
    if (!"hex_id" %in% names(obs_dt)) {
      obs_dt <- merge(obs_dt, hex_assign, by = "site_id", all.x = TRUE)
    }
    obs_dt <- obs_dt[!is.na(hex_id)]
  } else {
    obs_dt[, hex_id := 1L]
  }

  n_obs <- nrow(obs_dt)

  ## Validate tier fractions
  frac_sum <- frac_same + frac_neighbour + frac_distant
  if (abs(frac_sum - 1.0) > 1e-6) {
    cat(sprintf("  Warning: tier fractions sum to %.3f, normalising to 1.0\n", frac_sum))
    frac_same      <- frac_same / frac_sum
    frac_neighbour <- frac_neighbour / frac_sum
    frac_distant   <- frac_distant / frac_sum
  }

  n_same_target      <- ceiling(n_pairs * frac_same)
  n_neighbour_target <- ceiling(n_pairs * frac_neighbour)
  n_distant_target   <- n_pairs - n_same_target - n_neighbour_target

  cat(sprintf("  Between-site tiers: %d same-hex + %d neighbour-hex + %d distant-hex\n",
              n_same_target, n_neighbour_target, n_distant_target))

  ## ---- Match/mismatch targets (global) ----
  if (!is.null(match_ratio)) {
    target_match_frac <- match_ratio
  } else {
    target_match_frac <- 0.5
  }

  ## ---- Hex-level setup ----
  all_hexes <- sort(unique(obs_dt$hex_id))
  n_all_hex <- length(all_hexes)

  ## Reorder obs_dt by hex_id for contiguous obs blocks
  setorder(obs_dt, hex_id)
  obs_dt[, obs_idx := .I]

  ## Per-hex obs lookup
  hex_obs_dt <- obs_dt[, .(n_obs = .N, start_idx = min(obs_idx)), by = hex_id]
  setkey(hex_obs_dt, hex_id)
  hex_obs_split <- split(obs_dt$obs_idx, obs_dt$hex_id)

  ## Per-hex site-obs counts (for cross-site probability)
  hex_site_obs <- obs_dt[, .(n_obs = .N), by = .(hex_id, site_id)]
  hex_site_counts <- obs_dt[, .(n_sites = length(unique(site_id))), by = hex_id]

  ## Per-site observation count lookup (for design weight computation)
  ## n_obs_site[site_id] gives the number of obs (= m_i = species count) at that site
  site_obs_count <- obs_dt[, .N, by = site_id]
  setkey(site_obs_count, site_id)
  ## Per-hex total obs (N_h) and site count (n_h) lookups
  setkey(hex_site_counts, hex_id)

  ## ---- Build hex neighbour lookup ----
  neighbour_map <- list()  # hex_id -> vector of neighbour hex_ids
  if (!is.null(hex_grid)) {
    cat("  Building hex neighbour lookup (st_touches)...\n")
    ## st_touches returns a sparse list of which hexes share an edge/vertex
    touches <- st_touches(hex_grid)
    for (i in seq_len(nrow(hex_grid))) {
      hid <- hex_grid$hex_id[i]
      nb_indices <- touches[[i]]
      if (length(nb_indices) > 0) {
        nb_hex_ids <- hex_grid$hex_id[nb_indices]
        ## Keep only occupied hexes
        nb_hex_ids <- nb_hex_ids[nb_hex_ids %in% all_hexes]
        if (length(nb_hex_ids) > 0) {
          neighbour_map[[as.character(hid)]] <- nb_hex_ids
        }
      }
    }
    n_with_nb <- sum(vapply(neighbour_map, length, 0L) > 0)
    cat(sprintf("  %d hexes have occupied neighbours (of %d occupied)\n",
                n_with_nb, n_all_hex))
  } else {
    cat("  Warning: no hex_grid supplied, neighbour tier falls back to distant\n")
  }

  ## =========================================================================
  ## Helper: build pair data.table from two obs index vectors
  ## =========================================================================
  build_pair_batch <- function(r1, r2, hex_ids = NULL, hex_ids_j = NULL) {
    dt <- data.table(
      site_i      = obs_dt$site_id[r1],
      site_j      = obs_dt$site_id[r2],
      species_i   = obs_dt$species[r1],
      species_j   = obs_dt$species[r2],
      lon_i       = obs_dt$longitude[r1],
      lat_i       = obs_dt$latitude[r1],
      lon_j       = obs_dt$longitude[r2],
      lat_j       = obs_dt$latitude[r2],
      eventDate_i = obs_dt$eventDate[r1],
      eventDate_j = obs_dt$eventDate[r2],
      y           = as.integer(obs_dt$species[r1] != obs_dt$species[r2]),
      pair_type   = "between",
      nRecords_i  = obs_dt$nRecords[r1],
      nRecords_j  = obs_dt$nRecords[r2],
      richness_i  = obs_dt$richness[r1],
      richness_j  = obs_dt$richness[r2],
      hex_id      = if (!is.null(hex_ids)) hex_ids else obs_dt$hex_id[r1],
      hex_id_j    = if (!is.null(hex_ids_j)) hex_ids_j else obs_dt$hex_id[r2]
    )
    dt
  }

  ## =========================================================================
  ## Helper: compute between-site design weights for a tier.
  ## Target G(i,j) = 1/(n_units) * 1/(n_h_a * n_h_b)  [uniform unit, uniform site]
  ## Proposal q(i,j) ∝ (n_obs_i / N_h_a) * (n_obs_j / N_h_b)
  ## Design weight a = G/q ∝ (N_h_a * N_h_b) / (n_h_a * n_h_b * n_obs_i * n_obs_j)
  ## Normalised so mean = 1 within the tier.
  ##
  ## For same-hex tier (h_a = h_b = h): similar structure but the proposal
  ## draws 2 obs from the SAME hex pool, so n_obs_i * n_obs_j / N_h^2.
  ## =========================================================================
  compute_between_design_w <- function(dt) {
    if (nrow(dt) == 0) return(numeric(0))
    ## Look up per-site obs counts (cast to double to avoid integer overflow)
    n_obs_i <- as.double(site_obs_count[.(dt$site_i), N])
    n_obs_j <- as.double(site_obs_count[.(dt$site_j), N])
    ## Look up per-hex totals
    N_h_a   <- as.double(hex_obs_dt[.(dt$hex_id), n_obs])
    N_h_b   <- as.double(hex_obs_dt[.(dt$hex_id_j), n_obs])
    n_h_a   <- as.double(hex_site_counts[.(dt$hex_id), n_sites])
    n_h_b   <- as.double(hex_site_counts[.(dt$hex_id_j), n_sites])
    ## Raw design weight: proportional to G/q
    raw_w <- (N_h_a * N_h_b) / (n_h_a * n_h_b * n_obs_i * n_obs_j)
    raw_w[is.na(raw_w) | !is.finite(raw_w)] <- 1.0
    ## Normalise to mean 1 — return numeric vector (not data.table)
    raw_w / mean(raw_w)
  }

  ## =========================================================================
  ## Helper: quota-based sampling within a specific set of candidate obs pairs
  ## Returns a data.table of pairs with balanced match/mismatch
  ## =========================================================================
  sample_tier <- function(target_n, tier_name, draw_fn, max_iter_tier = 50L) {
    n_match_target <- ceiling(target_n * target_match_frac)
    n_miss_target  <- target_n - n_match_target
    n_match_got    <- 0L
    n_miss_got     <- 0L
    ## Track all drawn (pre-quota) for retention rates
    drawn_match    <- 0L
    drawn_mismatch <- 0L

    result_list <- vector("list", max_iter_tier)

    for (it in seq_len(max_iter_tier)) {
      remaining <- target_n - n_match_got - n_miss_got
      if (remaining <= 0) break

      batch <- draw_fn(remaining)
      if (is.null(batch) || nrow(batch) == 0) next

      new_match <- batch[y == 0]
      new_miss  <- batch[y == 1]
      drawn_match    <- drawn_match    + nrow(new_match)
      drawn_mismatch <- drawn_mismatch + nrow(new_miss)

      n_take_match <- min(nrow(new_match), n_match_target - n_match_got)
      n_take_miss  <- min(nrow(new_miss),  n_miss_target  - n_miss_got)

      keep <- rbind(
        if (n_take_match > 0) new_match[sample(.N, n_take_match)] else NULL,
        if (n_take_miss > 0)  new_miss[sample(.N, n_take_miss)]  else NULL
      )

      if (!is.null(keep) && nrow(keep) > 0) {
        result_list[[it]] <- keep
        n_match_got <- n_match_got + n_take_match
        n_miss_got  <- n_miss_got  + n_take_miss
      }

      if (it %% 10 == 0) {
        cat(sprintf("    [%s] iter %d: %d match, %d mismatch\n",
                    tier_name, it, n_match_got, n_miss_got))
      }
    }

    out <- rbindlist(result_list[!vapply(result_list, is.null, logical(1))])
    ## Compute retention rates for class-conditioned sampling
    r_0 <- if (drawn_match > 0)    sum(out$y == 0) / drawn_match    else 1.0
    r_1 <- if (drawn_mismatch > 0) sum(out$y == 1) / drawn_mismatch else 1.0
    attr(out, "retention_rates") <- list(r_0 = r_0, r_1 = r_1,
                                          drawn_match = drawn_match,
                                          drawn_mismatch = drawn_mismatch)
    cat(sprintf("  %s: %d pairs (%d match, %d mismatch)  retent. r_0=%.4f, r_1=%.4f\n",
                tier_name, nrow(out), sum(out$y == 0), sum(out$y == 1), r_0, r_1))
    out
  }

  ## ===========================================================================
  ## Tier 1: Same-hex pairs (local beta diversity)
  ## ===========================================================================
  cat("\n  --- Tier 1: Same-hex between-site pairs ---\n")

  multi_site_hexes <- hex_site_counts[n_sites >= 2]$hex_id
  cat(sprintf("  %d hexes have >= 2 sites\n", length(multi_site_hexes)))

  ## Per-hex quota (equal allocation across hexes)
  tier1_quota <- ceiling(n_same_target / max(length(multi_site_hexes), 1L))
  tier1_match_target <- ceiling(tier1_quota * target_match_frac)
  tier1_miss_target  <- tier1_quota - tier1_match_target

  tier1_list <- vector("list", length(multi_site_hexes))
  n_t1_skipped <- 0L
  tier1_drawn_match <- 0L
  tier1_drawn_mismatch <- 0L

  for (h_idx in seq_along(multi_site_hexes)) {
    hid <- multi_site_hexes[h_idx]
    h_rows <- hex_obs_split[[as.character(hid)]]
    n_h <- length(h_rows)
    if (n_h < 2) { n_t1_skipped <- n_t1_skipped + 1L; next }

    ## Cross-site probability
    h_site_obs <- hex_site_obs[hex_id == hid]
    cross_site_prob <- 1 - sum(h_site_obs$n_obs^2) / n_h^2
    if (cross_site_prob < 0.001) { n_t1_skipped <- n_t1_skipped + 1L; next }

    n_miss_got  <- 0L
    n_match_got <- 0L
    hex_pair_list <- vector("list", 10L)

    for (iter in seq_len(10L)) {
      remaining <- tier1_quota - n_miss_got - n_match_got
      if (remaining <= 0) break

      nm <- as.integer(min(remaining / cross_site_prob * 2 + 10, 50000))
      r1 <- h_rows[sample.int(n_h, nm, replace = TRUE)]
      r2 <- h_rows[sample.int(n_h, nm, replace = TRUE)]

      same_site <- obs_dt$site_id[r1] == obs_dt$site_id[r2]
      r1 <- r1[!same_site]; r2 <- r2[!same_site]
      if (length(r1) == 0) break

      batch <- build_pair_batch(r1, r2, hex_ids = rep(hid, length(r1)),
                               hex_ids_j = rep(hid, length(r1)))

      new_miss  <- batch[y == 1]
      new_match <- batch[y == 0]
      tier1_drawn_match    <- tier1_drawn_match    + nrow(new_match)
      tier1_drawn_mismatch <- tier1_drawn_mismatch + nrow(new_miss)
      n_take_miss  <- min(nrow(new_miss),  tier1_miss_target  - n_miss_got)
      n_take_match <- min(nrow(new_match), tier1_match_target - n_match_got)

      keep <- rbind(
        if (n_take_miss > 0)  new_miss[sample(.N, n_take_miss)]   else NULL,
        if (n_take_match > 0) new_match[sample(.N, n_take_match)] else NULL
      )
      if (!is.null(keep) && nrow(keep) > 0) {
        hex_pair_list[[iter]] <- keep
        n_miss_got  <- n_miss_got  + n_take_miss
        n_match_got <- n_match_got + n_take_match
      }
    }

    hex_dt <- rbindlist(hex_pair_list[!vapply(hex_pair_list, is.null, logical(1))])
    if (nrow(hex_dt) > 0) tier1_list[[h_idx]] <- hex_dt
  }

  tier1_pairs <- rbindlist(tier1_list[!vapply(tier1_list, is.null, logical(1))])

  ## -- Add stratum index and design weights for Tier 1 --
  if (nrow(tier1_pairs) > 0) {
    tier1_pairs[, stratum := 1L]
    tier1_pairs[, design_w := compute_between_design_w(tier1_pairs)]
    ## Winsorise Tier 1 design weights
    if (!is.null(design_w_cap_pctl) && design_w_cap_pctl < 1.0 && nrow(tier1_pairs) > 0) {
      cap_val <- quantile(tier1_pairs$design_w, probs = design_w_cap_pctl, na.rm = TRUE)
      n_capped <- sum(tier1_pairs$design_w > cap_val, na.rm = TRUE)
      pre_max  <- max(tier1_pairs$design_w, na.rm = TRUE)
      tier1_pairs[design_w > cap_val, design_w := cap_val]
      tier1_pairs[, design_w := design_w / mean(design_w, na.rm = TRUE)]
      ess_post <- sum(tier1_pairs$design_w)^2 / sum(tier1_pairs$design_w^2)
      cat(sprintf("    Tier 1 design_w truncation (p%.1f): cap=%.3f, capped %d pairs (%.1f%%), pre-max=%.1f, post-ESS=%.0f/%.0f (%.1f%%)\n",
                  design_w_cap_pctl * 100, cap_val, n_capped,
                  100 * n_capped / nrow(tier1_pairs), pre_max,
                  ess_post, nrow(tier1_pairs), 100 * ess_post / nrow(tier1_pairs)))
    }
  }
  ## -- Compute Tier 1 retention rates --
  tier1_r0 <- if (tier1_drawn_match > 0)    sum(tier1_pairs$y == 0) / tier1_drawn_match else 1

  tier1_r1 <- if (tier1_drawn_mismatch > 0) sum(tier1_pairs$y == 1) / tier1_drawn_mismatch else 1

  cat(sprintf("  Tier 1 (same-hex): %d pairs (%d match, %d mismatch)\n",
              nrow(tier1_pairs), sum(tier1_pairs$y == 0), sum(tier1_pairs$y == 1)))
  if (n_t1_skipped > 0)
    cat(sprintf("    Skipped %d hexes (too sparse for cross-site pairs)\n", n_t1_skipped))

  ## ===========================================================================
  ## Tier 2: Neighbour-hex pairs (regional beta diversity)
  ## ===========================================================================
  cat("\n  --- Tier 2: Neighbour-hex between-site pairs ---\n")

  ## Build list of (hex_a, hex_b) neighbour pairs where both are occupied
  nb_pairs_list <- list()
  for (hid_chr in names(neighbour_map)) {
    hid <- as.integer(hid_chr)
    for (nb in neighbour_map[[hid_chr]]) {
      ## Only store each pair once (lower id first)
      if (hid < nb) nb_pairs_list[[length(nb_pairs_list) + 1]] <- c(hid, nb)
    }
  }

  if (length(nb_pairs_list) > 0) {
    nb_pairs_mat <- do.call(rbind, nb_pairs_list)
    n_nb_pairs <- nrow(nb_pairs_mat)
    cat(sprintf("  %d unique neighbour hex-pairs available\n", n_nb_pairs))

    ## Sampling: uniformly draw neighbour hex-pairs, then one obs from each hex
    tier2_draw <- function(remaining) {
      nm <- min(remaining * 3L, 100000L)
      pair_idx <- sample.int(n_nb_pairs, nm, replace = TRUE)
      h1 <- nb_pairs_mat[pair_idx, 1]
      h2 <- nb_pairs_mat[pair_idx, 2]

      ## Random obs from each hex
      n1 <- hex_obs_dt[.(h1), n_obs]
      st1 <- hex_obs_dt[.(h1), start_idx]
      n2 <- hex_obs_dt[.(h2), n_obs]
      st2 <- hex_obs_dt[.(h2), start_idx]

      s1 <- st1 + sample.int(max(n1), length(n1), replace = TRUE) %% n1
      s2 <- st2 + sample.int(max(n2), length(n2), replace = TRUE) %% n2

      valid <- !is.na(s1) & !is.na(s2)
      s1 <- s1[valid]; s2 <- s2[valid]; h1 <- h1[valid]
      if (length(s1) == 0) return(NULL)

      ## Ensure different sites (almost guaranteed, but check)
      diff_site <- obs_dt$site_id[s1] != obs_dt$site_id[s2]
      s1 <- s1[diff_site]; s2 <- s2[diff_site]; h1 <- h1[diff_site]
      h2_valid <- h2[valid]; h2_valid <- h2_valid[diff_site]
      if (length(s1) == 0) return(NULL)

      build_pair_batch(s1, s2, hex_ids = h1, hex_ids_j = h2_valid)
    }

    tier2_pairs <- sample_tier(n_neighbour_target, "Tier 2 (neighbour-hex)",
                               tier2_draw, max_iter_tier = max_iter)

    ## -- Add stratum index and design weights for Tier 2 --
    if (nrow(tier2_pairs) > 0) {
      tier2_pairs[, stratum := 2L]
      tier2_pairs[, design_w := compute_between_design_w(tier2_pairs)]
      ## Winsorise Tier 2 design weights
      if (!is.null(design_w_cap_pctl) && design_w_cap_pctl < 1.0 && nrow(tier2_pairs) > 0) {
        cap_val <- quantile(tier2_pairs$design_w, probs = design_w_cap_pctl, na.rm = TRUE)
        n_capped <- sum(tier2_pairs$design_w > cap_val, na.rm = TRUE)
        pre_max  <- max(tier2_pairs$design_w, na.rm = TRUE)
        tier2_pairs[design_w > cap_val, design_w := cap_val]
        tier2_pairs[, design_w := design_w / mean(design_w, na.rm = TRUE)]
        ess_post <- sum(tier2_pairs$design_w)^2 / sum(tier2_pairs$design_w^2)
        cat(sprintf("    Tier 2 design_w truncation (p%.1f): cap=%.3f, capped %d pairs (%.1f%%), pre-max=%.1f, post-ESS=%.0f/%.0f (%.1f%%)\n",
                    design_w_cap_pctl * 100, cap_val, n_capped,
                    100 * n_capped / nrow(tier2_pairs), pre_max,
                    ess_post, nrow(tier2_pairs), 100 * ess_post / nrow(tier2_pairs)))
      }
    }
    ## -- Extract Tier 2 retention rates --
    tier2_retention <- attr(tier2_pairs, "retention_rates")
  } else {
    cat("  No neighbour hex-pairs available; redirecting quota to distant tier\n")
    tier2_pairs <- data.table()
    n_distant_target <- n_distant_target + n_neighbour_target
  }

  ## ===========================================================================
  ## Tier 3: Distant-hex pairs (continental beta diversity)
  ## ===========================================================================
  cat("\n  --- Tier 3: Distant-hex between-site pairs ---\n")

  ## Build set of "distant" hex-pairs: all pairs except same-hex and neighbours
  ## For efficiency, we don't enumerate them; instead we sample any two hexes
  ## and reject if they are the same or neighbours.
  neighbour_set <- new.env(hash = TRUE, parent = emptyenv())
  for (hid_chr in names(neighbour_map)) {
    hid <- as.integer(hid_chr)
    for (nb in neighbour_map[[hid_chr]]) {
      key <- paste0(min(hid, nb), "_", max(hid, nb))
      assign(key, TRUE, envir = neighbour_set)
    }
  }

  is_distant <- function(h1, h2) {
    h1 != h2 & !vapply(seq_along(h1), function(i) {
      key <- paste0(min(h1[i], h2[i]), "_", max(h1[i], h2[i]))
      exists(key, envir = neighbour_set, inherits = FALSE)
    }, logical(1))
  }

  tier3_draw <- function(remaining) {
    nm <- min(remaining * 4L, 150000L)
    h1 <- sample(all_hexes, nm, replace = TRUE)
    h2 <- sample(all_hexes, nm, replace = TRUE)

    keep <- is_distant(h1, h2)
    h1 <- h1[keep]; h2 <- h2[keep]
    if (length(h1) == 0) return(NULL)

    n1 <- hex_obs_dt[.(h1), n_obs]
    st1 <- hex_obs_dt[.(h1), start_idx]
    n2 <- hex_obs_dt[.(h2), n_obs]
    st2 <- hex_obs_dt[.(h2), start_idx]

    s1 <- st1 + sample.int(max(n1), length(n1), replace = TRUE) %% n1
    s2 <- st2 + sample.int(max(n2), length(n2), replace = TRUE) %% n2

    valid <- !is.na(s1) & !is.na(s2)
    s1 <- s1[valid]; s2 <- s2[valid]; h1 <- h1[valid]
    h2_valid <- h2[keep]; h2_valid <- h2_valid[valid]
    if (length(s1) == 0) return(NULL)

    diff_site <- obs_dt$site_id[s1] != obs_dt$site_id[s2]
    s1 <- s1[diff_site]; s2 <- s2[diff_site]; h1 <- h1[diff_site]
    h2_valid <- h2_valid[diff_site]
    if (length(s1) == 0) return(NULL)

    build_pair_batch(s1, s2, hex_ids = h1, hex_ids_j = h2_valid)
  }

  tier3_pairs <- sample_tier(n_distant_target, "Tier 3 (distant-hex)",
                             tier3_draw, max_iter_tier = max_iter)

  ## -- Add stratum index and design weights for Tier 3 --
  if (nrow(tier3_pairs) > 0) {
    tier3_pairs[, stratum := 3L]
    tier3_pairs[, design_w := compute_between_design_w(tier3_pairs)]
    ## Winsorise Tier 3 design weights
    if (!is.null(design_w_cap_pctl) && design_w_cap_pctl < 1.0 && nrow(tier3_pairs) > 0) {
      cap_val <- quantile(tier3_pairs$design_w, probs = design_w_cap_pctl, na.rm = TRUE)
      n_capped <- sum(tier3_pairs$design_w > cap_val, na.rm = TRUE)
      pre_max  <- max(tier3_pairs$design_w, na.rm = TRUE)
      tier3_pairs[design_w > cap_val, design_w := cap_val]
      tier3_pairs[, design_w := design_w / mean(design_w, na.rm = TRUE)]
      ess_post <- sum(tier3_pairs$design_w)^2 / sum(tier3_pairs$design_w^2)
      cat(sprintf("    Tier 3 design_w truncation (p%.1f): cap=%.3f, capped %d pairs (%.1f%%), pre-max=%.1f, post-ESS=%.0f/%.0f (%.1f%%)\n",
                  design_w_cap_pctl * 100, cap_val, n_capped,
                  100 * n_capped / nrow(tier3_pairs), pre_max,
                  ess_post, nrow(tier3_pairs), 100 * ess_post / nrow(tier3_pairs)))
    }
  }
  ## -- Extract Tier 3 retention rates --
  tier3_retention <- attr(tier3_pairs, "retention_rates")

  ## ===========================================================================
  ## Phase 2: Species-stratified match boost (if needed)
  ## ===========================================================================
  all_tier_pairs <- rbind(tier1_pairs, tier2_pairs, tier3_pairs, fill = TRUE)
  total_match_got <- sum(all_tier_pairs$y == 0)
  total_match_needed <- ceiling(n_pairs * target_match_frac) - total_match_got

  if (total_match_needed > 0 && !is.null(match_ratio)) {
    cat(sprintf("\n  Phase 2: species-stratified match boost (%d more needed)\n",
                total_match_needed))

    obs_dt[, obs_idx := .I]
    species_index <- obs_dt[, .(obs_indices = list(obs_idx), m_s = .N), by = species]
    species_multi <- species_index[m_s >= 2]
    species_multi[, match_capacity := m_s * (m_s - 1L)]
    setkey(species_multi, species)

    match_boost_list <- vector("list", max_iter)
    n_boost_got <- 0L

    for (iter in seq_len(max_iter)) {
      remaining_match <- total_match_needed - n_boost_got
      if (remaining_match <= 0) break

      n_sp_draw <- min(remaining_match * 2L, nrow(species_multi) * 3L)
      sp_sample <- sample(species_multi$species,
                          n_sp_draw,
                          replace = TRUE,
                          prob = species_multi$match_capacity / sum(species_multi$match_capacity))

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

      ri_v <- ri[valid]; rj_v <- rj[valid]
      if (length(ri_v) == 0) next

      new_matches <- build_pair_batch(ri_v, rj_v)
      new_matches[, y := 0L]

      n_take <- min(nrow(new_matches), remaining_match)
      match_boost_list[[iter]] <- new_matches[sample(.N, n_take)]
      n_boost_got <- n_boost_got + n_take
    }

    boost_dt <- rbindlist(match_boost_list[!vapply(match_boost_list, is.null, logical(1))])
    cat(sprintf("  Phase 2 species-strat boost: %d extra matches\n", nrow(boost_dt)))

    ## -- Match-boost stratum: stratum 4, unit design weight --
    if (nrow(boost_dt) > 0) {
      boost_dt[, stratum  := 4L]
      boost_dt[, design_w := 1.0]
    }
  } else {
    boost_dt <- data.table()
  }

  ## Combine all between-site pairs
  ## Tag tier for diagnostics (legacy label)
  if (nrow(tier1_pairs) > 0) tier1_pairs[, between_tier := "same_hex"]
  if (nrow(tier2_pairs) > 0) tier2_pairs[, between_tier := "neighbour_hex"]
  if (nrow(tier3_pairs) > 0) tier3_pairs[, between_tier := "distant_hex"]
  if (nrow(boost_dt) > 0)    boost_dt[, between_tier := "match_boost"]
  out <- rbind(tier1_pairs, tier2_pairs, tier3_pairs, boost_dt, fill = TRUE)

  ## Ensure design_w and stratum columns exist even if some tiers empty
  if (!"stratum" %in% names(out))  out[, stratum  := NA_integer_]
  if (!"design_w" %in% names(out)) out[, design_w := 1.0]

  if (nrow(out) > n_pairs) out <- out[sample(.N, n_pairs)]

  n_match <- sum(out$y == 0)
  n_miss  <- sum(out$y == 1)
  cat(sprintf("\n  Between-site final: %d total (%d match, %d mismatch, ratio %.3f)\n",
              nrow(out), n_match, n_miss,
              ifelse(n_match > 0, n_miss / n_match, Inf)))

  ## Print tier breakdown
  if ("between_tier" %in% names(out)) {
    cat("  Tier breakdown:\n")
    print(out[, .(.N, n_match = sum(y == 0), n_miss = sum(y == 1)), by = between_tier])
  }

  if ("obs_idx" %in% names(obs_dt)) obs_dt[, obs_idx := NULL]

  ## Store per-stratum retention rates as attribute
  between_retention <- list(
    tier1 = list(r_0 = tier1_r0, r_1 = tier1_r1),
    tier2 = if (exists("tier2_retention") && !is.null(tier2_retention))
              tier2_retention else list(r_0 = NA, r_1 = NA),
    tier3 = if (exists("tier3_retention") && !is.null(tier3_retention))
              tier3_retention else list(r_0 = NA, r_1 = NA)
  )
  attr(out, "retention_rates") <- between_retention

  out
}


# ---------------------------------------------------------------------------
# clesso_sampler_diagnostics  -- PDF report of sampling balance
# ---------------------------------------------------------------------------
#' Generate a multi-page PDF summarising the spatial balance of a sample
#' drawn by the hex-balanced sampler.
#'
#' @param pairs_dt   data.table of sampled pairs (output of clesso_sampler)
#' @param obs_dt     data.table of the full observation set (unique site x species)
#' @param hex_assign data.table(site_id, hex_id)
#' @param hex_grid   sf object with hex polygons (from assign_hex_grid)
#' @param pdf_path   file path for the output PDF
clesso_sampler_diagnostics <- function(pairs_dt, obs_dt, hex_assign, hex_grid,
                                       pdf_path = "sampler_diagnostics.pdf") {
  require(data.table)
  require(sf)

  pairs_dt  <- as.data.table(pairs_dt)
  obs_dt    <- as.data.table(obs_dt)
  hex_assign <- as.data.table(hex_assign)

  ## ---- Derived tables ----

  # Site-level summaries
  site_stats <- obs_dt[, .(
    alpha_i   = .N,
    n_records = sum(nRecords)
  ), by = .(site_id, longitude, latitude)]
  site_stats <- merge(site_stats, hex_assign, by = "site_id", all.x = TRUE)

  # Hex-level observation summaries (from full population)
  hex_obs <- site_stats[!is.na(hex_id), .(
    n_sites     = .N,
    n_obs       = sum(alpha_i),
    mean_alpha  = mean(alpha_i),
    median_alpha = as.double(median(alpha_i)),
    total_records = sum(n_records)
  ), by = hex_id]

  # Hex-level sample summaries (from drawn pairs)
  hex_samp_within <- pairs_dt[pair_type == "within", .(
    n_within    = .N,
    n_within_match = sum(y == 0),
    n_within_miss  = sum(y == 1)
  ), by = hex_id]

  hex_samp_between <- pairs_dt[pair_type == "between", .(
    n_between    = .N,
    n_between_match = sum(y == 0),
    n_between_miss  = sum(y == 1)
  ), by = hex_id]

  # All unique sites appearing in the sample
  sampled_sites_i <- unique(pairs_dt[, .(site_id = site_i)])
  sampled_sites_j <- unique(pairs_dt[, .(site_id = site_j)])
  sampled_sites   <- unique(rbind(sampled_sites_i, sampled_sites_j))
  sampled_sites   <- merge(sampled_sites, hex_assign, by = "site_id", all.x = TRUE)
  hex_samp_sites  <- sampled_sites[!is.na(hex_id), .(n_sampled_sites = .N), by = hex_id]

  # Weight summaries per hex
  if ("w" %in% names(pairs_dt)) {
    hex_weights_within <- pairs_dt[pair_type == "within", .(
      mean_w_within  = mean(w),
      median_w_within = median(w),
      sum_w_within   = sum(w)
    ), by = hex_id]
    hex_weights_between <- pairs_dt[pair_type == "between", .(
      mean_w_between  = mean(w),
      median_w_between = median(w),
      sum_w_between   = sum(w)
    ), by = hex_id]
  }

  # Merge everything onto hex_obs
  hex_all <- copy(hex_obs)
  hex_all <- merge(hex_all, hex_samp_within,  by = "hex_id", all.x = TRUE)
  hex_all <- merge(hex_all, hex_samp_between, by = "hex_id", all.x = TRUE)
  hex_all <- merge(hex_all, hex_samp_sites,   by = "hex_id", all.x = TRUE)
  if ("w" %in% names(pairs_dt)) {
    hex_all <- merge(hex_all, hex_weights_within,  by = "hex_id", all.x = TRUE)
    hex_all <- merge(hex_all, hex_weights_between, by = "hex_id", all.x = TRUE)
  }
  # Fill NAs with 0 for count columns
  for (col in c("n_within", "n_within_match", "n_within_miss",
                 "n_between", "n_between_match", "n_between_miss",
                 "n_sampled_sites")) {
    if (col %in% names(hex_all)) {
      set(hex_all, which(is.na(hex_all[[col]])), col, 0L)
    }
  }
  hex_all[, n_total_pairs := n_within + n_between]

  # Merge hex_all data onto hex_grid sf for maps
  hex_grid_data <- merge(hex_grid, hex_all, by = "hex_id", all.x = TRUE)

  ## ---- Colour helpers ----
  map_fill <- function(hex_sf, values, title, log_scale = FALSE,
                       col_ramp = NULL) {
    if (is.null(col_ramp)) {
      col_ramp <- colorRampPalette(
        c("#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
          "#6baed6", "#4292c6", "#2171b5", "#084594")
      )
    }
    n_cols <- 100
    cols <- col_ramp(n_cols)

    vals <- values
    if (log_scale) vals <- log1p(vals)
    vals[is.na(vals)] <- 0
    vmin <- min(vals, na.rm = TRUE)
    vmax <- max(vals, na.rm = TRUE)
    if (vmax == vmin) vmax <- vmin + 1
    idx <- pmin(n_cols, pmax(1, round((vals - vmin) / (vmax - vmin) * (n_cols - 1)) + 1))

    plot(st_geometry(hex_sf), col = cols[idx], border = "grey50", lwd = 0.3,
         main = title)

    ## Legend
    legend_vals <- if (log_scale) {
      pretty(range(values, na.rm = TRUE), n = 5)
    } else {
      pretty(range(values, na.rm = TRUE), n = 5)
    }
    legend_cols_idx <- pmin(n_cols, pmax(1, round(
      (if (log_scale) log1p(legend_vals) else legend_vals - vmin) /
        (vmax - vmin) * (n_cols - 1)
    ) + 1))
    legend_cols_idx[legend_cols_idx < 1] <- 1
    legend("bottomleft", legend = format(legend_vals, big.mark = ","),
           fill = cols[legend_cols_idx], bty = "n", cex = 0.7,
           title = if (log_scale) "(log scale)" else NULL)
  }

  heat_ramp <- colorRampPalette(
    c("#fff5f0", "#fee0d2", "#fcbba1", "#fc9272",
      "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d")
  )
  green_ramp <- colorRampPalette(
    c("#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b",
      "#74c476", "#41ab5d", "#238b45", "#005a32")
  )

  ## ---- Open PDF ----
  pdf(pdf_path, width = 11, height = 8.5)
  on.exit(dev.off(), add = TRUE)

  ## ===== PAGE 1: Overview text summary =====
  par(mar = c(0, 1, 2, 1))
  plot.new()
  title("CLESSO Hex-Balanced Sampler Diagnostics", cex.main = 1.4)

  n_within  <- sum(pairs_dt$pair_type == "within")
  n_between <- sum(pairs_dt$pair_type == "between")
  n_hexes   <- nrow(hex_all)
  n_pop_sites <- nrow(site_stats)

  txt <- c(
    sprintf("Date: %s", Sys.time()),
    "",
    sprintf("Population: %s unique site x species records",
            format(nrow(obs_dt), big.mark = ",")),
    sprintf("            %s sites, %s species",
            format(n_pop_sites, big.mark = ","),
            format(length(unique(obs_dt$species)), big.mark = ",")),
    sprintf("Hex grid:   %d occupied hexes", n_hexes),
    sprintf("Sites/hex:  min=%d  median=%d  max=%d  mean=%.1f",
            min(hex_all$n_sites), as.integer(median(hex_all$n_sites)),
            max(hex_all$n_sites), mean(hex_all$n_sites)),
    "",
    sprintf("Within-site pairs:  %s  (match=%s, mismatch=%s)",
            format(n_within, big.mark = ","),
            format(sum(pairs_dt$pair_type == "within" & pairs_dt$y == 0), big.mark = ","),
            format(sum(pairs_dt$pair_type == "within" & pairs_dt$y == 1), big.mark = ",")),
    sprintf("Between-site pairs: %s  (match=%s, mismatch=%s)",
            format(n_between, big.mark = ","),
            format(sum(pairs_dt$pair_type == "between" & pairs_dt$y == 0), big.mark = ","),
            format(sum(pairs_dt$pair_type == "between" & pairs_dt$y == 1), big.mark = ",")),
    sprintf("Total pairs:        %s", format(nrow(pairs_dt), big.mark = ",")),
    sprintf("Unique sites in sample: %s / %s (%.1f%%)",
            format(nrow(sampled_sites), big.mark = ","),
            format(n_pop_sites, big.mark = ","),
            100 * nrow(sampled_sites) / n_pop_sites),
    "",
    sprintf("Hexes contributing within-site pairs:  %d / %d",
            sum(hex_all$n_within > 0), n_hexes),
    sprintf("Hexes contributing between-site pairs: %d / %d",
            sum(hex_all$n_between > 0), n_hexes)
  )
  if ("w" %in% names(pairs_dt)) {
    txt <- c(txt, "",
      sprintf("Weight range (within):  %.2e to %.2e  (median %.2e)",
              min(pairs_dt[pair_type == "within"]$w),
              max(pairs_dt[pair_type == "within"]$w),
              median(pairs_dt[pair_type == "within"]$w)),
      sprintf("Weight range (between): %.2e to %.2e  (median %.2e)",
              min(pairs_dt[pair_type == "between"]$w),
              max(pairs_dt[pair_type == "between"]$w),
              median(pairs_dt[pair_type == "between"]$w))
    )
  }
  if ("between_tier" %in% names(pairs_dt)) {
    bp <- pairs_dt[pair_type == "between"]
    tc <- bp[, .N, by = between_tier]
    setorder(tc, between_tier)
    txt <- c(txt, "",
      "Between-site tier breakdown:")
    for (r in seq_len(nrow(tc))) {
      txt <- c(txt, sprintf("  %-15s  %s  (%.1f%%)",
                             tc$between_tier[r],
                             format(tc$N[r], big.mark = ","),
                             100 * tc$N[r] / sum(tc$N)))
    }
  }
  text(0.02, seq(0.92, by = -0.04, length.out = length(txt)),
       txt, adj = 0, family = "mono", cex = 0.75)

  ## ===== PAGE 2: Observation maps (population) =====
  par(mfrow = c(2, 2), mar = c(1, 1, 3, 1))

  map_fill(hex_grid_data, hex_grid_data$n_sites,
           "Population: Sites per hex", log_scale = TRUE)
  map_fill(hex_grid_data, hex_grid_data$n_obs,
           "Population: Observations per hex (site x spp)", log_scale = TRUE,
           col_ramp = green_ramp)
  map_fill(hex_grid_data, hex_grid_data$mean_alpha,
           "Population: Mean species richness per hex")
  map_fill(hex_grid_data, hex_grid_data$total_records,
           "Population: Total records per hex", log_scale = TRUE,
           col_ramp = heat_ramp)

  ## ===== PAGE 3: Sample maps =====
  par(mfrow = c(2, 2), mar = c(1, 1, 3, 1))

  map_fill(hex_grid_data, hex_grid_data$n_total_pairs,
           "Sample: Total pairs per hex", log_scale = TRUE)
  map_fill(hex_grid_data, hex_grid_data$n_within,
           "Sample: Within-site pairs per hex", log_scale = TRUE,
           col_ramp = green_ramp)
  map_fill(hex_grid_data, hex_grid_data$n_between,
           "Sample: Between-site pairs per hex", log_scale = TRUE,
           col_ramp = heat_ramp)
  map_fill(hex_grid_data, hex_grid_data$n_sampled_sites,
           "Sample: Unique sites sampled per hex", log_scale = TRUE)

  ## ===== PAGE 4: Sample vs Population comparison (scatter) =====
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

  # Sites: population vs sampled
  plot(hex_all$n_sites, hex_all$n_sampled_sites,
       xlab = "Sites in hex (population)", ylab = "Sites sampled in hex",
       main = "Site coverage per hex", pch = 16, cex = 0.7, col = "#2171b5")
  abline(0, 1, lty = 2, col = "grey50")
  abline(lm(n_sampled_sites ~ n_sites, data = hex_all), col = "red", lwd = 1.5)
  legend("topleft", legend = sprintf("r = %.2f",
         cor(hex_all$n_sites, hex_all$n_sampled_sites, use = "complete.obs")),
         bty = "n", cex = 0.8)

  # Observations vs total pairs
  plot(hex_all$n_obs, hex_all$n_total_pairs,
       xlab = "Observations in hex (population)", ylab = "Total pairs sampled",
       main = "Pairs vs observations per hex",
       pch = 16, cex = 0.7, col = "#238b45", log = "xy")
  abline(lm(log(n_total_pairs + 1) ~ log(n_obs + 1), data = hex_all),
         col = "red", lwd = 1.5, untf = TRUE)
  legend("topleft", legend = sprintf("r(log) = %.2f",
         cor(log1p(hex_all$n_obs), log1p(hex_all$n_total_pairs),
             use = "complete.obs")),
         bty = "n", cex = 0.8)

  # Sampling rate: pairs / obs
  hex_all[, samp_rate := n_total_pairs / n_obs]
  plot(hex_all$n_obs, hex_all$samp_rate,
       xlab = "Observations in hex (population)",
       ylab = "Sampling rate (pairs / obs)",
       main = "Sampling rate by hex density",
       pch = 16, cex = 0.7, col = "#cb181d", log = "x")
  abline(h = median(hex_all$samp_rate, na.rm = TRUE), lty = 2, col = "grey50")
  text(max(hex_all$n_obs) * 0.8,
       median(hex_all$samp_rate, na.rm = TRUE) * 1.15,
       sprintf("median = %.3f", median(hex_all$samp_rate, na.rm = TRUE)),
       cex = 0.7)

  # Sampling coverage: fraction of sites sampled per hex
  hex_all[, site_coverage := n_sampled_sites / n_sites]
  plot(hex_all$n_sites, hex_all$site_coverage,
       xlab = "Sites in hex (population)",
       ylab = "Fraction of sites sampled",
       main = "Site coverage by hex size",
       pch = 16, cex = 0.7, col = "#6a3d9a", log = "x",
       ylim = c(0, 1))
  abline(h = median(hex_all$site_coverage, na.rm = TRUE), lty = 2, col = "grey50")

  ## ===== PAGE 5: Histograms - sample balance =====
  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

  # Within-site pairs per hex
  hist(hex_all$n_within, breaks = 30, col = "#9ecae1", border = "white",
       main = "Within-site pairs per hex", xlab = "Pairs", ylab = "Hexes")
  abline(v = median(hex_all$n_within, na.rm = TRUE), lty = 2, col = "red", lwd = 2)

  # Between-site pairs per hex
  hist(hex_all$n_between, breaks = 30, col = "#a1d99b", border = "white",
       main = "Between-site pairs per hex", xlab = "Pairs", ylab = "Hexes")
  abline(v = median(hex_all$n_between, na.rm = TRUE), lty = 2, col = "red", lwd = 2)

  # Total pairs per hex
  hist(hex_all$n_total_pairs, breaks = 30, col = "#fcbba1", border = "white",
       main = "Total pairs per hex", xlab = "Pairs", ylab = "Hexes")
  abline(v = median(hex_all$n_total_pairs, na.rm = TRUE), lty = 2, col = "red", lwd = 2)

  # Population observations per hex (log)
  hist(log10(hex_all$n_obs), breaks = 30, col = "#c6dbef", border = "white",
       main = "Population obs per hex (log10)", xlab = "log10(obs)", ylab = "Hexes")
  abline(v = median(log10(hex_all$n_obs), na.rm = TRUE), lty = 2, col = "red", lwd = 2)

  # Sites per hex (population)
  hist(hex_all$n_sites, breaks = 30, col = "#bcbddc", border = "white",
       main = "Sites per hex (population)", xlab = "Sites", ylab = "Hexes")
  abline(v = median(hex_all$n_sites, na.rm = TRUE), lty = 2, col = "red", lwd = 2)

  # Mean richness per hex
  hist(hex_all$mean_alpha, breaks = 30, col = "#fdd0a2", border = "white",
       main = "Mean species richness per hex", xlab = "Mean alpha", ylab = "Hexes")
  abline(v = median(hex_all$mean_alpha, na.rm = TRUE), lty = 2, col = "red", lwd = 2)

  ## ===== PAGE 6: Weight distributions =====
  if ("w" %in% names(pairs_dt)) {
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

    # Within-site weight distribution
    w_within <- pairs_dt[pair_type == "within"]$w
    if (length(w_within) > 0) {
      hist(log10(w_within + 1), breaks = 50, col = "#9ecae1", border = "white",
           main = "Within-site weights (log10)", xlab = "log10(w + 1)", ylab = "Pairs")
      abline(v = log10(median(w_within) + 1), lty = 2, col = "red", lwd = 2)
    } else {
      plot.new(); title("No within-site pairs")
    }

    # Between-site weight distribution
    w_between <- pairs_dt[pair_type == "between"]$w
    if (length(w_between) > 0) {
      hist(log10(w_between + 1), breaks = 50, col = "#a1d99b", border = "white",
           main = "Between-site weights (log10)", xlab = "log10(w + 1)", ylab = "Pairs")
      abline(v = log10(median(w_between) + 1), lty = 2, col = "red", lwd = 2)
    } else {
      plot.new(); title("No between-site pairs")
    }

    # Mean weight per hex (within)
    if ("mean_w_within" %in% names(hex_all)) {
      valid_w <- hex_all[!is.na(mean_w_within) & mean_w_within > 0]
      if (nrow(valid_w) > 0) {
        map_fill(hex_grid_data, hex_grid_data$mean_w_within,
                 "Mean within-site weight per hex", log_scale = TRUE)
      } else {
        plot.new(); title("No within-site weights")
      }
    } else {
      plot.new(); title("No within-site weights")
    }

    # Mean weight per hex (between)
    if ("mean_w_between" %in% names(hex_all)) {
      valid_w <- hex_all[!is.na(mean_w_between) & mean_w_between > 0]
      if (nrow(valid_w) > 0) {
        map_fill(hex_grid_data, hex_grid_data$mean_w_between,
                 "Mean between-site weight per hex", log_scale = TRUE,
                 col_ramp = heat_ramp)
      } else {
        plot.new(); title("No between-site weights")
      }
    } else {
      plot.new(); title("No between-site weights")
    }
  }

  ## ===== PAGE 7: Per-site sampling frequency =====
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

  # How many times each site appears in pairs
  site_freq_i <- pairs_dt[, .N, by = site_i]
  site_freq_j <- pairs_dt[, .N, by = site_j]
  setnames(site_freq_i, c("site_id", "n_i"))
  setnames(site_freq_j, c("site_id", "n_j"))
  site_freq <- merge(site_freq_i, site_freq_j, by = "site_id", all = TRUE)
  for (col in c("n_i", "n_j")) set(site_freq, which(is.na(site_freq[[col]])), col, 0L)
  site_freq[, n_total := n_i + n_j]

  # Merge richness
  site_alpha_pop <- obs_dt[, .(alpha_i = .N), by = site_id]
  site_freq <- merge(site_freq, site_alpha_pop, by = "site_id", all.x = TRUE)
  site_freq <- merge(site_freq, hex_assign, by = "site_id", all.x = TRUE)

  hist(log10(site_freq$n_total), breaks = 50, col = "#9ecae1", border = "white",
       main = "Site sampling frequency (log10)", xlab = "log10(n_pairs)", ylab = "Sites")

  plot(site_freq$alpha_i, site_freq$n_total,
       xlab = "Site species richness", ylab = "Times sampled in pairs",
       main = "Sampling freq vs richness", pch = ".", col = "#2171b580",
       log = "xy")
  abline(lm(log(n_total) ~ log(alpha_i), data = site_freq[alpha_i > 0 & n_total > 0]),
         col = "red", lwd = 1.5, untf = TRUE)

  # Richness distribution: sampled vs population
  alpha_pop <- site_alpha_pop$alpha_i
  alpha_samp <- site_freq$alpha_i

  hist(alpha_pop, breaks = 60, col = adjustcolor("#2171b5", 0.5), border = NA,
       main = "Species richness: population vs sampled",
       xlab = "Species richness", ylab = "Sites", xlim = c(0, quantile(alpha_pop, 0.99)))
  hist(alpha_samp, breaks = 60, col = adjustcolor("#cb181d", 0.5), border = NA, add = TRUE)
  legend("topright",
         legend = c(sprintf("Population (n=%s)", format(length(alpha_pop), big.mark = ",")),
                    sprintf("Sampled (n=%s)", format(length(alpha_samp), big.mark = ","))),
         fill = adjustcolor(c("#2171b5", "#cb181d"), 0.5), bty = "n", cex = 0.8)

  # Site coverage map (fraction sampled per hex)
  map_fill(hex_grid_data, hex_grid_data$n_sampled_sites / hex_grid_data$n_sites,
           "Site coverage fraction per hex",
           col_ramp = green_ramp)

  ## ===== PAGE 8: Geographic distribution of sampled pairs =====
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

  # Within-site pair locations
  within_pairs <- pairs_dt[pair_type == "within"]
  if (nrow(within_pairs) > 0) {
    n_plot <- min(nrow(within_pairs), 10000)
    idx <- sample(nrow(within_pairs), n_plot)
    plot(within_pairs$lon_i[idx], within_pairs$lat_i[idx],
         pch = ".", col = adjustcolor("#2171b5", 0.3),
         xlab = "Longitude", ylab = "Latitude",
         main = sprintf("Within-site pair locations (n=%s)",
                        format(nrow(within_pairs), big.mark = ",")),
         asp = 1)
    plot(st_geometry(hex_grid), add = TRUE, border = "grey70", lwd = 0.3)
  }

  # Between-site pairs: show connections
  between_pairs <- pairs_dt[pair_type == "between"]
  if (nrow(between_pairs) > 0) {
    n_plot <- min(nrow(between_pairs), 5000)
    idx <- sample(nrow(between_pairs), n_plot)
    plot(c(between_pairs$lon_i[idx], between_pairs$lon_j[idx]),
         c(between_pairs$lat_i[idx], between_pairs$lat_j[idx]),
         type = "n",
         xlab = "Longitude", ylab = "Latitude",
         main = sprintf("Between-site pair connections (n=%s of %s)",
                        format(n_plot, big.mark = ","),
                        format(nrow(between_pairs), big.mark = ",")),
         asp = 1)
    plot(st_geometry(hex_grid), add = TRUE, border = "grey90", lwd = 0.3)
    segments(between_pairs$lon_i[idx], between_pairs$lat_i[idx],
             between_pairs$lon_j[idx], between_pairs$lat_j[idx],
             col = adjustcolor("#cb181d", 0.05), lwd = 0.3)
    points(between_pairs$lon_i[idx], between_pairs$lat_i[idx],
           pch = ".", col = adjustcolor("#2171b5", 0.3))
  }

  ## ===== PAGE 9: Between-site tier breakdown (3-tier) =====
  if ("between_tier" %in% names(pairs_dt)) {
    bp <- pairs_dt[pair_type == "between"]
    tier_counts <- bp[, .N, by = between_tier]
    setorder(tier_counts, between_tier)

    par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

    # Tier bar chart
    barplot(tier_counts$N, names.arg = tier_counts$between_tier,
            col = c("#4292c6", "#41ab5d", "#ef3b2c")[seq_len(nrow(tier_counts))],
            main = "Between-site pairs by tier",
            ylab = "Pairs", xlab = "Tier")
    tier_pct <- sprintf("%.1f%%", 100 * tier_counts$N / sum(tier_counts$N))
    text(seq_len(nrow(tier_counts)) * 1.2 - 0.5, tier_counts$N * 0.5,
         tier_pct, cex = 0.9, font = 2)

    # Geographic distance distribution by tier
    if (all(c("lon_i", "lat_i", "lon_j", "lat_j") %in% names(bp))) {
      # Haversine-approximate distance in km
      bp[, dist_km := {
        dlat <- (lat_j - lat_i) * pi / 180
        dlon <- (lon_j - lon_i) * pi / 180
        a <- sin(dlat / 2)^2 + cos(lat_i * pi / 180) * cos(lat_j * pi / 180) * sin(dlon / 2)^2
        6371 * 2 * atan2(sqrt(a), sqrt(1 - a))
      }]

      tier_labels <- sort(unique(bp$between_tier))
      tier_cols <- c(same_hex = "#4292c6", neighbour_hex = "#41ab5d", distant_hex = "#ef3b2c")
      xmax <- quantile(bp$dist_km, 0.99, na.rm = TRUE)

      # Overlapping histograms
      plot(1, type = "n", xlim = c(0, xmax), ylim = c(0, 1),
           xlab = "Distance (km)", ylab = "Density",
           main = "Pair distance distribution by tier")
      for (tier in tier_labels) {
        d <- bp[between_tier == tier]$dist_km
        if (length(d) > 10) {
          dens <- density(d, from = 0, to = xmax, n = 256)
          dens$y <- dens$y / max(dens$y)
          lines(dens$x, dens$y, col = tier_cols[tier], lwd = 2)
        }
      }
      legend("topright", legend = tier_labels, col = tier_cols[tier_labels],
             lwd = 2, bty = "n", cex = 0.8)

      # Distance boxplots by tier
      boxplot(dist_km ~ between_tier, data = bp,
              col = tier_cols[sort(unique(bp$between_tier))],
              main = "Distance by tier (km)",
              xlab = "Tier", ylab = "Distance (km)",
              outline = FALSE)
    } else {
      plot.new(); title("No coordinates for distance calc")
      plot.new(); title("")
    }

    # Geographic connections coloured by tier (subsample)
    if (all(c("lon_i", "lat_i", "lon_j", "lat_j") %in% names(bp))) {
      n_plot <- min(nrow(bp), 6000)
      idx <- sample(nrow(bp), n_plot)
      bp_sub <- bp[idx]

      plot(c(bp_sub$lon_i, bp_sub$lon_j),
           c(bp_sub$lat_i, bp_sub$lat_j),
           type = "n",
           xlab = "Longitude", ylab = "Latitude",
           main = "Between-site pairs coloured by tier",
           asp = 1)
      plot(st_geometry(hex_grid), add = TRUE, border = "grey90", lwd = 0.3)

      tier_cols_alpha <- c(same_hex = adjustcolor("#4292c6", 0.08),
                           neighbour_hex = adjustcolor("#41ab5d", 0.08),
                           distant_hex = adjustcolor("#ef3b2c", 0.08))
      # Draw distant first (background), then neighbour, then same_hex (foreground)
      for (tier in c("distant_hex", "neighbour_hex", "same_hex")) {
        bp_t <- bp_sub[between_tier == tier]
        if (nrow(bp_t) > 0) {
          segments(bp_t$lon_i, bp_t$lat_i, bp_t$lon_j, bp_t$lat_j,
                   col = tier_cols_alpha[tier], lwd = 0.4)
        }
      }
      legend("bottomleft",
             legend = c("same_hex", "neighbour_hex", "distant_hex"),
             col = c("#4292c6", "#41ab5d", "#ef3b2c"),
             lty = 1, lwd = 2, bty = "n", cex = 0.7)
    }
  }

  cat(sprintf("  Diagnostics PDF saved: %s\n", pdf_path))
  invisible(hex_all)
}


# ---------------------------------------------------------------------------
# clesso_sampler  (hex-balanced version, same API)
# ---------------------------------------------------------------------------
clesso_sampler <- function(obs_dt,
                           n_within          = 500000,
                           n_between         = 500000,
                           within_min_recs   = 2,
                           within_match_ratio  = 0.5,
                           between_match_ratio = 0.5,
                           balance_weights   = TRUE,
                           hex_size          = 2.0,
                           between_frac_same      = 0.5,
                           between_frac_neighbour = 0.3,
                           between_frac_distant   = 0.2,
                           design_w_cap_pctl = 0.99,
                           diagnostics_pdf   = NULL,
                           seed              = NULL,
                           ...) {
  require(data.table)
  require(sf)
  obs_dt <- as.data.table(obs_dt)

  required_cols <- c("site_id", "species", "longitude", "latitude",
                     "eventDate", "nRecords", "richness")
  missing <- setdiff(required_cols, names(obs_dt))
  if (length(missing) > 0) {
    stop("obs_dt is missing required columns: ", paste(missing, collapse = ", "))
  }

  cat(sprintf("\n=== CLESSO v2 Sampler (hex-balanced, %.1f deg hexes) ===\n", hex_size))
  cat(sprintf("  Raw observations: %d records, %d sites, %d species\n",
              nrow(obs_dt), length(unique(obs_dt$site_id)),
              length(unique(obs_dt$species))))

  ## Reduce to unique site x species combinations
  ## Keeps one row per (site_id, species); aggregates nRecords, takes first
  ## of all other columns (longitude/latitude/richness are site-level anyway).
  obs_dt <- obs_dt[, .(
    longitude = longitude[1],
    latitude  = latitude[1],
    eventDate = eventDate[1],
    nRecords  = sum(nRecords),
    richness  = richness[1]
  ), by = .(site_id, species)]

  ## CRITICAL: convert factor columns to character for reliable named-vector
  ## indexing (factor levels cause positional rather than name-based lookup)
  if (is.factor(obs_dt$site_id))  obs_dt[, site_id := as.character(site_id)]
  if (is.factor(obs_dt$species))  obs_dt[, species := as.character(species)]

  cat(sprintf("  After site x species dedup: %d unique records, %d sites, %d species\n",
              nrow(obs_dt), length(unique(obs_dt$site_id)),
              length(unique(obs_dt$species))))
  cat(sprintf("  Targets: %d within-site pairs, %d between-site pairs\n",
              n_within, n_between))

  ## ---- Build hex grid assignment ----
  cat("\n--- Building hex grid ---\n")
  site_locs <- obs_dt[, .(longitude = longitude[1], latitude = latitude[1]),
                      by = site_id]
  hex_result <- assign_hex_grid(site_locs, hex_size = hex_size, return_grid = TRUE)
  hex_assign <- hex_result$assign
  hex_grid   <- hex_result$hex_grid

  ## Show hex occupancy stats
  hex_stats <- hex_assign[, .N, by = hex_id]
  cat(sprintf("  Sites per hex: min %d, median %d, max %d, mean %.1f\n",
              min(hex_stats$N), as.integer(median(hex_stats$N)),
              max(hex_stats$N), mean(hex_stats$N)))

  ## Within-site pairs
  cat("\n--- Sampling within-site pairs (hex-balanced) ---\n")
  t_within <- proc.time()
  within_pairs <- clesso_sample_within_pairs(
    obs_dt            = obs_dt,
    n_pairs           = n_within,
    min_records       = within_min_recs,
    match_ratio       = within_match_ratio,
    hex_assign        = hex_assign,
    seed              = seed,
    design_w_cap_pctl = design_w_cap_pctl
  )
  t_within <- proc.time() - t_within
  cat(sprintf("  Within-site time: %.2f s\n", t_within["elapsed"]))

  ## Between-site pairs (3-tier: same-hex / neighbour-hex / distant-hex)
  cat("\n--- Sampling between-site pairs (3-tier hex-balanced) ---\n")
  t_between <- proc.time()
  between_pairs <- clesso_sample_between_pairs(
    obs_dt            = obs_dt,
    n_pairs           = n_between,
    match_ratio       = between_match_ratio,
    hex_assign        = hex_assign,
    hex_grid          = hex_grid,
    frac_same         = between_frac_same,
    frac_neighbour    = between_frac_neighbour,
    frac_distant      = between_frac_distant,
    seed              = if (!is.null(seed)) seed + 1 else NULL,
    design_w_cap_pctl = design_w_cap_pctl,
    ...
  )
  t_between <- proc.time() - t_between
  cat(sprintf("  Between-site time: %.2f s\n", t_between["elapsed"]))

  ## Combine
  all_pairs <- rbind(within_pairs, between_pairs, fill = TRUE)
  all_pairs[, is_within := as.integer(pair_type == "within")]

  ## Design weights (already computed inside each sampler stage):
  ##  - within_pairs has stratum=0, design_w from hex-capacity IPW
  ##  - between_pairs has stratum=1/2/3/4, design_w from pair-level proposal
  ## No longer call clesso_compute_pair_weights (the old hex-IPW function).
  ## For backward compat, copy design_w into 'w' column.
  if (!"design_w" %in% names(all_pairs)) all_pairs[, design_w := 1.0]
  all_pairs[is.na(design_w), design_w := 1.0]
  all_pairs[, w := design_w]

  ## Collect retention rates for retrospective correction
  within_retention  <- attr(within_pairs,  "retention_rates")
  between_retention <- attr(between_pairs, "retention_rates")
  retention_rates <- list(
    within  = within_retention,
    between = between_retention
  )
  attr(all_pairs, "retention_rates") <- retention_rates

  ## Diagnostics
  cat("\n--- Sampling summary ---\n")
  summary_dt <- all_pairs[, .(
    n_pairs    = .N,
    n_match    = sum(y == 0),
    n_mismatch = sum(y == 1),
    n_sites    = length(unique(c(site_i, site_j)))
  ), by = pair_type]
  print(summary_dt)

  ## Per-hex diagnostics
  if ("hex_id" %in% names(all_pairs)) {
    cat("\n  Per-hex pair counts (top 20):\n")
    hex_diag <- all_pairs[, .(
      n_within  = sum(pair_type == "within"),
      n_between = sum(pair_type == "between"),
      n_total   = .N
    ), by = hex_id][order(-n_total)]
    print(head(hex_diag, 20))

    cat(sprintf("\n  Hex pair count stats: min=%d, median=%d, max=%d\n",
                min(hex_diag$n_total), as.integer(median(hex_diag$n_total)),
                max(hex_diag$n_total)))
  }

  cat(sprintf("  Total pairs: %d\n", nrow(all_pairs)))
  cat(sprintf("  Unique sites in pairs: %d\n",
              length(unique(c(all_pairs$site_i, all_pairs$site_j)))))

  ## ---- Generate diagnostic PDF if requested ----
  if (!is.null(diagnostics_pdf)) {
    cat("\n--- Generating sampler diagnostics PDF ---\n")
    tryCatch(
      clesso_sampler_diagnostics(
        pairs_dt   = all_pairs,
        obs_dt     = obs_dt,
        hex_assign = hex_assign,
        hex_grid   = hex_grid,
        pdf_path   = diagnostics_pdf
      ),
      error = function(e) {
        warning("Diagnostics PDF generation failed: ", e$message)
      }
    )
  }

  all_pairs
}


# ---------------------------------------------------------------------------
# clesso_compute_pair_weights  (hex IPW, stratum-normalised)
# ---------------------------------------------------------------------------
#' Compute per-pair importance weights for the hex-balanced sampler.
#'
#' Design rationale:
#' -  The sampler already controls the balance across the four strata
#'    (within-match, within-mismatch, between-match, between-mismatch)
#'    via the match_ratio parameters.  There is no need to re-inflate
#'    weights by the population cell sizes (pop_N / sample_N), which
#'    would create extreme variance (the between-mismatch population is
#'    ~10 trillion vs within-match ~3 million → ratio ~3 million).
#'
#' -  The weights ONLY correct for within-stratum geographic imbalance
#'    introduced by the hex-balanced sampler, which over-samples sparse
#'    hexes relative to their population contribution.
#'
#' -  For each stratum the IPW = hex_capacity / mean(hex_capacity),
#'    normalised so that the mean weight within each stratum is 1.
#'    This keeps the loss scale stable and ESS high.
#'
#' -  The NN training code applies a further alpha-harmonic-mean
#'    correction to match pairs to adjust for within-cell richness bias.
#'    That operates on these weights.
#'
#' @param pairs_dt   data.table of sampled pairs (with hex_id column)
#' @param obs_unique_dt data.table of unique site×species records
#' @param hex_assign data.table(site_id, hex_id)
#' @return pairs_dt with added column 'w' (pair weight, mean ~1 per stratum)
clesso_compute_pair_weights <- function(pairs_dt, obs_unique_dt,
                                        hex_assign = NULL) {
  require(data.table)
  pairs_dt <- copy(as.data.table(pairs_dt))
  obs_unique_dt <- as.data.table(obs_unique_dt)

  ## Site alpha (for hex capacity computation)
  site_alpha <- obs_unique_dt[, .(alpha_i = .N), by = site_id]

  ## ---- IPW correction for hex balancing ----
  ## The hex-balanced sampler gives each hex roughly equal quota (uniform
  ## in hex-space).  A pair drawn from hex h should be weighted by:
  ##   IPW(h) = capacity_h / mean(capacity)
  ## so that dense hexes recover their population importance and the
  ## expected weighted loss is unbiased.  After normalisation per-stratum
  ## the mean weight is 1, keeping loss scale and ESS healthy.

  if (!is.null(hex_assign) && "hex_id" %in% names(pairs_dt)) {
    site_alpha_hex <- merge(site_alpha, hex_assign, by = "site_id", all.x = TRUE)

    ## Within-site capacity per hex: sum(alpha_i^2) -- proportional to
    ## number of possible within-site pairs
    hex_within_cap <- site_alpha_hex[!is.na(hex_id),
                                     .(hex_cap_within = sum(as.numeric(alpha_i)^2)),
                                     by = hex_id]

    ## Between-site capacity per hex: sum(alpha_i) -- proportional to
    ## contribution to cross-site pairs
    hex_between_cap <- site_alpha_hex[!is.na(hex_id),
                                      .(hex_cap_between = sum(as.numeric(alpha_i))),
                                      by = hex_id]

    hex_cap <- merge(hex_within_cap, hex_between_cap, by = "hex_id", all = TRUE)
    pairs_dt <- merge(pairs_dt, hex_cap, by = "hex_id", all.x = TRUE)

    ## IPW: capacity / mean_capacity  (normalised so mean = 1 per pair_type)
    mean_within_cap  <- mean(hex_within_cap$hex_cap_within)
    mean_between_cap <- mean(hex_between_cap$hex_cap_between)

    pairs_dt[pair_type == "within",
             w := hex_cap_within / mean_within_cap]
    pairs_dt[pair_type == "between",
             w := hex_cap_between / mean_between_cap]

    ## Handle NAs (sites without hex assignment)
    pairs_dt[is.na(w), w := 1.0]

    ## Diagnostic output
    cat(sprintf("  IPW stats (within):  min=%.3f  median=%.3f  max=%.3f\n",
        min(pairs_dt[pair_type == "within"]$w, na.rm = TRUE),
        median(pairs_dt[pair_type == "within"]$w, na.rm = TRUE),
        max(pairs_dt[pair_type == "within"]$w, na.rm = TRUE)))
    cat(sprintf("  IPW stats (between): min=%.3f  median=%.3f  max=%.3f\n",
        min(pairs_dt[pair_type == "between"]$w, na.rm = TRUE),
        median(pairs_dt[pair_type == "between"]$w, na.rm = TRUE),
        max(pairs_dt[pair_type == "between"]$w, na.rm = TRUE)))
  } else {
    pairs_dt[, w := 1.0]
  }

  ## ESS diagnostic
  w_all <- pairs_dt$w
  ess_all <- sum(w_all)^2 / sum(w_all^2)
  cat(sprintf("  Pair weights computed (hex IPW, stratum-normalised):\n"))
  cat(sprintf("  Overall ESS: %.0f / %d = %.1f%%\n",
      ess_all, nrow(pairs_dt), 100 * ess_all / nrow(pairs_dt)))
  print(
    pairs_dt[, .(
      n      = .N,
      mean_w  = mean(w),
      min_w   = min(w),
      max_w   = max(w),
      ess     = sum(w)^2 / sum(w^2),
      ess_pct = round(100 * sum(w)^2 / sum(w^2) / .N, 1)
    ), by = .(pair_type, y)][order(pair_type, y)]
  )

  ## Clean up temp columns
  drop_cols <- intersect(names(pairs_dt),
                         c("hex_cap_within", "hex_cap_between"))
  if (length(drop_cols) > 0) pairs_dt[, (drop_cols) := NULL]

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
