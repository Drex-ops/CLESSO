##############################################################################
##
## predict_temporal.R — Temporal prediction from a fitted STresiduals GDM
##
## Purpose:
##   Given a fitted GDM from the reca_STresiduals pipeline and a set of
##   lon/lat points with paired years (baseline → target), compute the
##   temporal component of ecological distance and dissimilarity.
##
##   Spatial and substrate predictors are zeroed out; only temporal
##   environmental variables contribute to the prediction.
##
## Prerequisites (must be sourced before calling):
##   gdm_functions.R   (I_spline)
##   gen_windows.R      (gen_windows)
##   utils.R            (inv.logit, ObsTrans)
##   + the 'arrow' package
##
## Usage:
##   source("src/shared/R/utils.R")
##   source("src/shared/R/gdm_functions.R")
##   source("src/shared/R/gen_windows.R")
##   source("src/shared/R/predict_temporal.R")
##
##   result <- predict_temporal_gdm(
##     fit_path     = "output/AVES_1mil_30climWin_STresid_biAverage_fittedGDM.RData",
##     points       = data.frame(lon = c(149.1, 133.8),
##                               lat = c(-35.3, -23.7),
##                               year1 = c(1960, 1960),
##                               year2 = c(2010, 2010)),
##     npy_src      = "/Volumes/PortableSSD/CLIMATE/geonpy",
##     python_exe   = ".venv/bin/python3",
##     pyper_script = "src/shared/python/pyper.py"
##   )
##
## Returns:
##   A data.frame with columns:
##     lon, lat, year1, year2,
##     temporal_distance  — temporal ecological distance (sum of temporal
##                          I-spline contributions, excluding intercept)
##     linear_predictor   — intercept + temporal_distance
##     predicted_prob     — inv.logit(linear_predictor) [mismatch probability]
##     dissimilarity      — ObsTrans-corrected temporal dissimilarity [0, 1]
##
##   Attributes (access via attr(result, "...")):
##     "spline_table"       — full covariate matrix (n_points × n_total_splines),
##                            zeros for spatial/substrate, values for temporal
##     "pred_contributions" — per-predictor summed contribution (n_points × n_preds)
##     "raw_temporal_env"   — raw extracted temporal env values (pre-spline)
##     "fit_metadata"       — key metadata from the fit object
##
##############################################################################

# ---------------------------------------------------------------------------
# predict_temporal_gdm
#
# Full pipeline: extract temporal env → I-spline → predict.
#
# Parameters:
#   fit_path       - path to *_fittedGDM.RData (used if 'fit' is NULL)
#   fit            - the fit list object directly (if already loaded)
#   points         - data.frame with columns: lon, lat, year1, year2
#                    Optionally include month1, month2 (default: 'month')
#   npy_src        - path to directory containing geonpy .npy files
#   python_exe     - path to Python executable
#   pyper_script   - path to pyper.py
#   feather_tmpdir - temp directory for feather exchange files (default: tempdir())
#   month          - default month for climate extraction when month1/month2
#                    are not in 'points' (default: 6 = June, mid-year)
#   verbose        - print progress messages (default: TRUE)
# ---------------------------------------------------------------------------
predict_temporal_gdm <- function(
    fit_path       = NULL,
    fit            = NULL,
    points,
    npy_src,
    python_exe     = NULL,
    pyper_script   = NULL,
    feather_tmpdir = tempdir(),
    month          = 6L,
    verbose        = TRUE
) {

  ## ==== 1. Load and validate fit object ================================
  if (is.null(fit) && is.null(fit_path))
    stop("Provide either 'fit' (list) or 'fit_path' (path to *_fittedGDM.RData).")

  if (is.null(fit)) {
    if (!file.exists(fit_path)) stop(paste("fit_path not found:", fit_path))
    env <- new.env()
    load(fit_path, envir = env)
    fit <- env$fit
    if (is.null(fit)) stop("The .RData file does not contain a 'fit' object.")
  }

  required <- c("intercept", "predictors", "coefficients", "quantiles",
                 "splines", "climate_window", "env_params", "w_ratio")
  missing  <- setdiff(required, names(fit))
  if (length(missing) > 0)
    stop(paste("fit object missing required metadata fields:",
               paste(missing, collapse = ", "),
               "\n  Re-run the model with the updated run_obsGDM.R to include metadata."))

  ## ==== 2. Validate points =============================================
  if (!is.data.frame(points)) points <- as.data.frame(points)
  needed <- c("lon", "lat", "year1", "year2")
  miss   <- setdiff(needed, names(points))
  if (length(miss) > 0)
    stop(paste("'points' data.frame missing columns:", paste(miss, collapse = ", ")))
  n_pts <- nrow(points)
  if (verbose) cat(sprintf("=== Temporal GDM Prediction (%d point-pairs) ===\n", n_pts))

  ## ==== 3. Identify temporal predictors ================================
  temp_idx <- grep("^temp_", fit$predictors)
  if (length(temp_idx) == 0)
    stop("No temporal predictors (names starting with 'temp_') found.\n",
         "  This function is designed for reca_STresiduals GDM fits.")

  n_preds <- length(fit$predictors)
  if (verbose) {
    cat(sprintf("  Model: %s | climate window = %d yr | w = %.4f\n",
                fit$species_group, fit$climate_window, fit$w_ratio))
    cat(sprintf("  Predictors: %d total, %d temporal\n",
                n_preds, length(temp_idx)))
  }

  ## ==== 4. Build temporal extraction parameters ========================
  ## Reconstruct the same temporal_params used in the training pipeline
  c_yr         <- fit$climate_window
  geonpy_start <- if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L

  temporal_params <- list()
  for (ep in fit$env_params) {
    temporal_params[[length(temporal_params) + 1]] <- list(
      variables = ep$variables,
      mstat     = ep$mstat,
      cstat     = ep$cstat,
      window    = c_yr,
      prefix    = paste0("temp_", ep$cstat)
    )
  }

  ## ==== 5. Build site-pair table (same site, two times) ================
  m1 <- if ("month1" %in% names(points)) as.integer(points$month1) else rep(as.integer(month), n_pts)
  m2 <- if ("month2" %in% names(points)) as.integer(points$month2) else rep(as.integer(month), n_pts)

  pairs <- data.frame(
    Lon1   = points$lon,
    Lat1   = points$lat,
    year1  = as.integer(points$year1),
    month1 = m1,
    Lon2   = points$lon,
    Lat2   = points$lat,
    year2  = as.integer(points$year2),
    month2 = m2
  )

  ## Validate: years must allow climate window extraction
  min_year <- min(c(pairs$year1, pairs$year2))
  if ((min_year - c_yr) < geonpy_start)
    stop(sprintf(
      "Earliest year (%d) - climate window (%d) = %d, which is before geonpy start (%d).\n  Use later years or a shorter climate window.",
      min_year, c_yr, min_year - c_yr, geonpy_start))

  ## ==== 6. Extract temporal environmental data =========================
  if (verbose) cat(sprintf("  Extracting temporal env (window = %d yr)...\n", c_yr))
  require(arrow)

  env_parts <- list()
  for (j in seq_along(temporal_params)) {
    tp <- temporal_params[[j]]
    if (verbose)
      cat(sprintf("    [%d/%d] %s (%s)\n", j, length(temporal_params),
                   tp$prefix, paste(tp$variables, collapse = ", ")))

    raw <- gen_windows(
      pairs        = pairs,
      variables    = tp$variables,
      mstat        = tp$mstat,
      cstat        = tp$cstat,
      window       = tp$window,
      npy_src      = npy_src,
      start_year   = geonpy_start,
      python_exe   = python_exe,
      pyper_script = pyper_script,
      feather_tmpdir = feather_tmpdir
    )

    ## Keep env columns only (skip first 8 coordinate columns)
    env_cols <- raw[, 9:ncol(raw), drop = FALSE]
    colnames(env_cols) <- paste(tp$prefix, colnames(env_cols), sep = "_")
    env_parts[[j]] <- env_cols
  }

  env_all <- do.call(cbind, env_parts)

  ## ==== 7. Split site-1 / site-2 and clean column names ================
  idx_1  <- grep("_1$", names(env_all))
  idx_2  <- grep("_2$", names(env_all))
  env_s1 <- env_all[, idx_1, drop = FALSE]
  env_s2 <- env_all[, idx_2, drop = FALSE]

  ## Strip date-range patterns (e.g. "191101-201712_") to match fit$predictors
  strip_daterange <- function(x) gsub("\\d{6}-\\d{6}_", "", x)
  names(env_s1) <- strip_daterange(names(env_s1))
  names(env_s2) <- strip_daterange(names(env_s2))

  if (verbose) cat(sprintf("  Extracted %d temporal env columns per site\n", ncol(env_s1)))

  ## Check for NA and sentinel values
  any_na   <- any(is.na(env_s1)) || any(is.na(env_s2))
  any_sent <- any(env_s1 == -9999, na.rm = TRUE) || any(env_s2 == -9999, na.rm = TRUE)
  if (any_na)   warning("NA values found in extracted env data. Affected rows will have NA predictions.")
  if (any_sent) warning("-9999 sentinel values in extracted data (possible ocean/missing grid cells).")

  ## ==== 8. Apply I-splines and build full covariate table ==============
  result <- .predict_from_temporal_env(fit, env_s1, env_s2, temp_idx, verbose)

  ## ==== 9. Prepend point identifiers ===================================
  result <- cbind(
    data.frame(
      lon   = points$lon,
      lat   = points$lat,
      year1 = points$year1,
      year2 = points$year2,
      stringsAsFactors = FALSE
    ),
    result
  )

  ## Attach raw env as attribute
  attr(result, "raw_temporal_env") <- cbind(env_s1, env_s2)
  attr(result, "fit_metadata") <- list(
    species_group         = fit$species_group,
    climate_window        = fit$climate_window,
    w_ratio               = fit$w_ratio,
    intercept             = fit$intercept,
    D2                    = fit$D2,
    nagelkerke_r2         = fit$nagelkerke_r2,
    n_temporal_predictors = length(temp_idx),
    n_total_predictors    = n_preds,
    n_total_splines       = sum(fit$splines),
    predictors_used       = fit$predictors[temp_idx]
  )

  if (verbose) {
    cat(sprintf("  Temporal distance range: [%.4f, %.4f]\n",
                min(result$temporal_distance, na.rm = TRUE),
                max(result$temporal_distance, na.rm = TRUE)))
    cat(sprintf("  Dissimilarity range:     [%.4f, %.4f]\n",
                min(result$dissimilarity, na.rm = TRUE),
                max(result$dissimilarity, na.rm = TRUE)))
    cat("=== Prediction complete ===\n")
  }

  result
}


# ---------------------------------------------------------------------------
# predict_temporal_from_env
#
# Lower-level function: apply the GDM temporal prediction from
# pre-extracted raw environmental values (skips the gen_windows step).
#
# Parameters:
#   fit    - the fit list object (must include predictors, coefficients,
#            quantiles, splines, intercept, w_ratio)
#   env_s1 - data.frame of site-1 (baseline year) raw env values.
#            Column names must match fit$predictors for temporal predictors,
#            with date-range patterns already stripped.
#   env_s2 - data.frame of site-2 (prediction year) raw env values.
#            Same columns as env_s1 but with _1 replaced by _2.
#   verbose - print progress (default: TRUE)
#
# Returns:
#   Same structure as predict_temporal_gdm but without lon/lat/year columns.
# ---------------------------------------------------------------------------
predict_temporal_from_env <- function(fit, env_s1, env_s2, verbose = TRUE) {

  required <- c("intercept", "predictors", "coefficients", "quantiles",
                 "splines", "w_ratio")
  missing  <- setdiff(required, names(fit))
  if (length(missing) > 0)
    stop(paste("fit object missing fields:", paste(missing, collapse = ", ")))

  temp_idx <- grep("^temp_", fit$predictors)
  if (length(temp_idx) == 0)
    stop("No temporal predictors (names starting with 'temp_') found in fit.")

  .predict_from_temporal_env(fit, env_s1, env_s2, temp_idx, verbose)
}


# ---------------------------------------------------------------------------
# .predict_from_temporal_env  (internal)
#
# Shared implementation for both predict_temporal_gdm and
# predict_temporal_from_env.
# ---------------------------------------------------------------------------
.predict_from_temporal_env <- function(fit, env_s1, env_s2, temp_idx, verbose) {

  n_pts         <- nrow(env_s1)
  n_preds       <- length(fit$predictors)
  total_splines <- sum(fit$splines)
  csp           <- c(0, cumsum(fit$splines))

  ## ---- Full spline covariate table: zeros everywhere, fill temporal ----
  spline_table <- matrix(0, nrow = n_pts, ncol = total_splines)

  ## Name columns to match training pipeline
  spl_names <- character(total_splines)
  k <- 0L
  for (i in seq_along(fit$predictors)) {
    for (s in seq_len(fit$splines[i])) {
      k <- k + 1L
      spl_names[k] <- paste0(fit$predictors[i], "_spl", s)
    }
  }
  colnames(spline_table) <- spl_names

  ## Per-predictor contribution tracking
  pred_contrib <- matrix(0, nrow = n_pts, ncol = n_preds)
  colnames(pred_contrib) <- fit$predictors

  matched <- 0L
  for (i in temp_idx) {
    pred_name <- fit$predictors[i]
    ns        <- fit$splines[i]
    coefs_i   <- fit$coefficients[(csp[i] + 1):(csp[i] + ns)]
    quants_i  <- fit$quantiles[(csp[i] + 1):(csp[i] + ns)]

    ## Match predictor name to extracted env columns
    ## Predictor ends in _1 (from training pipeline X1 column naming)
    col_1 <- which(names(env_s1) == pred_name)
    pred_2 <- sub("_1$", "_2", pred_name)
    col_2 <- which(names(env_s2) == pred_2)

    if (length(col_1) == 0 || length(col_2) == 0) {
      warning(sprintf("  Cannot match predictor '%s' to extracted env columns. Skipping.", pred_name))
      next
    }

    v1 <- env_s1[, col_1]
    v2 <- env_s2[, col_2]
    matched <- matched + 1L

    ## Compute I-spline basis values and fill covariate table
    contrib_i <- rep(0, n_pts)
    for (sp in seq_len(ns)) {
      ## Knot selection logic matches splineData / splineData_fast
      if (sp == 1L) {
        q1 <- quants_i[1]; q2 <- quants_i[1]; q3 <- quants_i[min(2, ns)]
      } else if (sp == ns) {
        q1 <- quants_i[max(1, ns - 1)]; q2 <- quants_i[ns]; q3 <- quants_i[ns]
      } else {
        q1 <- quants_i[sp - 1]; q2 <- quants_i[sp]; q3 <- quants_i[sp + 1]
      }

      spl_diff <- abs(I_spline(v1, q1, q2, q3) - I_spline(v2, q1, q2, q3))

      ## Store in full covariate table
      spline_table[, csp[i] + sp] <- spl_diff

      ## Accumulate weighted contribution for this predictor
      contrib_i <- contrib_i + coefs_i[sp] * spl_diff
    }

    pred_contrib[, i] <- contrib_i
  }

  if (verbose) cat(sprintf("  Matched %d of %d temporal predictors\n", matched, length(temp_idx)))

  ## ---- Compute predictions ----

  ## Temporal ecological distance (contribution from temporal predictors only)
  ## This is the dot product of the full spline table with all coefficients;
  ## because spatial/substrate columns are zero, only temporal contributes.
  temporal_distance <- as.numeric(spline_table %*% fit$coefficients)

  ## Full linear predictor (intercept + temporal distance)
  eta <- fit$intercept + temporal_distance

  ## Predicted mismatch probability (logit link)
  predicted_prob <- inv.logit(eta)

  ## Baseline probability (intercept only = no ecological distance)
  p0 <- inv.logit(fit$intercept)

  ## Temporal dissimilarity via observation transform
  dissim <- ObsTrans(p0, fit$w_ratio, predicted_prob)

  ## ---- Assemble output ----
  result <- data.frame(
    temporal_distance = temporal_distance,
    linear_predictor  = eta,
    predicted_prob    = predicted_prob,
    dissimilarity     = dissim$out,
    stringsAsFactors  = FALSE
  )

  attr(result, "spline_table")       <- spline_table
  attr(result, "pred_contributions") <- pred_contrib

  result
}


# ---------------------------------------------------------------------------
# summarise_temporal_gdm
#
# Print a human-readable summary of the temporal components of a fitted
# GDM object.
#
# Parameters:
#   fit      - the fit list object (or path to *_fittedGDM.RData)
#   fit_path - alternative: path to .RData file
# ---------------------------------------------------------------------------
summarise_temporal_gdm <- function(fit = NULL, fit_path = NULL) {

  if (is.null(fit)) {
    if (is.null(fit_path)) stop("Provide 'fit' or 'fit_path'.")
    env <- new.env(); load(fit_path, envir = env); fit <- env$fit
  }

  cat("========================================\n")
  cat("  Fitted GDM Summary (temporal focus)\n")
  cat("========================================\n")

  if (!is.null(fit$species_group))  cat(sprintf("  Species group  : %s\n", fit$species_group))
  if (!is.null(fit$climate_window)) cat(sprintf("  Climate window : %d years\n", fit$climate_window))
  if (!is.null(fit$nMatch))         cat(sprintf("  nMatch         : %s\n", format(fit$nMatch, big.mark = ",")))
  if (!is.null(fit$w_ratio))        cat(sprintf("  w (miss/match) : %.4f\n", fit$w_ratio))
  if (!is.null(fit$D2))             cat(sprintf("  D² (deviance)  : %.4f\n", fit$D2))
  if (!is.null(fit$nagelkerke_r2))  cat(sprintf("  Nagelkerke R²  : %.4f\n", fit$nagelkerke_r2))
  if (!is.null(fit$biAverage))      cat(sprintf("  biAverage      : %s\n", fit$biAverage))
  if (!is.null(fit$decomposition))  cat(sprintf("  Decomposition  : %s\n", fit$decomposition))
  cat(sprintf("  Intercept      : %.6f\n", fit$intercept))
  cat(sprintf("  Sample size    : %s\n", format(fit$sample, big.mark = ",")))

  n_preds   <- length(fit$predictors)
  temp_idx  <- grep("^temp_", fit$predictors)
  spat_idx  <- grep("^spat_", fit$predictors)
  subs_idx  <- setdiff(seq_len(n_preds), c(temp_idx, spat_idx))
  csp       <- c(0, cumsum(fit$splines))

  cat(sprintf("\n  Predictors: %d total\n", n_preds))
  cat(sprintf("    Spatial    : %d\n", length(spat_idx)))
  cat(sprintf("    Substrate  : %d\n", length(subs_idx)))
  cat(sprintf("    Temporal   : %d\n", length(temp_idx)))

  ## Temporal predictor details
  if (length(temp_idx) > 0) {
    cat("\n  Temporal predictors:\n")
    for (i in temp_idx) {
      ns     <- fit$splines[i]
      coefs  <- fit$coefficients[(csp[i] + 1):(csp[i] + ns)]
      quants <- fit$quantiles[(csp[i] + 1):(csp[i] + ns)]
      total  <- sum(coefs)
      cat(sprintf("    %-35s  coef_sum = %8.5f  quantiles = [%.2f, %.2f, %.2f]\n",
                  fit$predictors[i], total, quants[1], quants[2], quants[3]))
    }

    ## Total temporal coefficient weight
    all_coefs <- fit$coefficients
    all_coefs[is.na(all_coefs)] <- 0
    temp_coef_sum  <- sum(abs(all_coefs[unlist(lapply(temp_idx, function(i) (csp[i]+1):(csp[i]+fit$splines[i])))]))
    total_coef_sum <- sum(abs(all_coefs))
    cat(sprintf("\n  Temporal coefficient weight: %.4f / %.4f (%.1f%%)\n",
                temp_coef_sum, total_coef_sum,
                100 * temp_coef_sum / max(total_coef_sum, 1e-10)))
  }

  cat("========================================\n")
  invisible(fit)
}


# ===========================================================================
# Example usage (uncomment to run):
# ===========================================================================
#
# ## Source dependencies
# source("src/shared/R/utils.R")
# source("src/shared/R/gdm_functions.R")
# source("src/shared/R/gen_windows.R")
# source("src/shared/R/predict_temporal.R")
#
# ## Inspect the fitted model
# summarise_temporal_gdm(
#   fit_path = "src/reca_STresiduals/output/AVES_1mil_30climWin_STresid_biAverage_fittedGDM.RData"
# )
#
# ## Define prediction points
# pts <- data.frame(
#   lon   = c(149.13, 133.87, 151.21),
#   lat   = c(-35.28, -23.70, -33.87),
#   year1 = c(1960,   1960,   1960),
#   year2 = c(2010,   2010,   2010)
# )
#
# ## Run temporal prediction
# result <- predict_temporal_gdm(
#   fit_path     = "src/reca_STresiduals/output/AVES_1mil_30climWin_STresid_biAverage_fittedGDM.RData",
#   points       = pts,
#   npy_src      = "/Volumes/PortableSSD/CLIMATE/geonpy",
#   python_exe   = ".venv/bin/python3",
#   pyper_script = "src/shared/python/pyper.py"
# )
#
# print(result)
#
# ## Access the full spline covariate table
# spline_tab <- attr(result, "spline_table")
#
# ## Access per-predictor contributions
# contrib <- attr(result, "pred_contributions")
#
# ## Access raw env values (pre-spline)
# raw_env <- attr(result, "raw_temporal_env")
