##############################################################################
##
## debug_temporal_prediction.R
##
## Steps through predict_temporal_gdm manually (one year-pair) to identify
## where NAs arise.  Run interactively -- each section prints diagnostics.
##
##############################################################################

cat("=== DEBUG: Stepping through temporal prediction ===\n\n")

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))

project_root <- config$project_root
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_temporal.R"))

library(raster)
library(arrow)

# ---------------------------------------------------------------------------
# 1. Load fit object and check predictor metadata
# ---------------------------------------------------------------------------
cat("--- STEP 1: Load fit ---\n")
load(config$fit_path)  # loads 'fit'

cat(sprintf("  Predictors: %d\n", length(fit$predictors)))
cat(sprintf("  Coefficients: %d (NA count: %d)\n",
            length(fit$coefficients), sum(is.na(fit$coefficients))))
cat(sprintf("  Quantiles: %d (NA count: %d)\n",
            length(fit$quantiles), sum(is.na(fit$quantiles))))
cat(sprintf("  Splines: %s\n", paste(fit$splines, collapse = ", ")))
cat(sprintf("  add_modis: %s\n", fit$add_modis))
if (isTRUE(fit$add_modis)) {
  cat(sprintf("  modis_variables: %s\n", paste(fit$modis_variables, collapse = ", ")))
  cat(sprintf("  modis_year_range: %d–%d\n", fit$modis_year_range[1], fit$modis_year_range[2]))
}

## Show all temporal predictors and their quantiles
temp_idx <- grep("^temp_", fit$predictors)
csp      <- c(0, cumsum(fit$splines))
cat(sprintf("\n  Temporal predictors (%d):\n", length(temp_idx)))
for (i in temp_idx) {
  ns <- fit$splines[i]
  q  <- fit$quantiles[(csp[i] + 1):(csp[i] + ns)]
  co <- fit$coefficients[(csp[i] + 1):(csp[i] + ns)]
  cat(sprintf("    [%d] %-35s  quantiles=[%s]  coefs=[%s]\n",
              i, fit$predictors[i],
              paste(sprintf("%.4f", q), collapse = ", "),
              paste(sprintf("%.6f", co), collapse = ", ")))
}

# ---------------------------------------------------------------------------
# 2. Sample a small set of points
# ---------------------------------------------------------------------------
cat("\n--- STEP 2: Sample 10 test points ---\n")
ras <- raster(config$reference_raster)
set.seed(42)
samp <- as.data.frame(sampleRandom(ras, size = 10, na.rm = TRUE, xy = TRUE))
colnames(samp)[1:2] <- c("lon", "lat")
n_pts <- nrow(samp)
cat(sprintf("  Points: %d  (lon range: [%.2f, %.2f])\n",
            n_pts, min(samp$lon), max(samp$lon)))
print(samp[, 1:2])

## One year pair for testing
baseline_year <- if (isTRUE(config$add_modis)) config$modis_start_year else 1950L
target_year   <- baseline_year + 5L
cat(sprintf("\n  Test pair: %d -> %d\n", baseline_year, target_year))

# ---------------------------------------------------------------------------
# 3. Build the site-pair table (same as predict_temporal_gdm step 5)
# ---------------------------------------------------------------------------
cat("\n--- STEP 3: Build pairs table ---\n")
c_yr         <- fit$climate_window
geonpy_start <- if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L

pairs <- data.frame(
  Lon1   = samp$lon,
  Lat1   = samp$lat,
  year1  = as.integer(baseline_year),
  month1 = 6L,
  Lon2   = samp$lon,
  Lat2   = samp$lat,
  year2  = as.integer(target_year),
  month2 = 6L
)
cat(sprintf("  Pairs: %d rows, min year: %d, climate window: %d -> needs data from %d\n",
            nrow(pairs), baseline_year, c_yr, baseline_year - c_yr))

# ---------------------------------------------------------------------------
# 4. Extract climate temporal env (gen_windows)
# ---------------------------------------------------------------------------
cat("\n--- STEP 4: Extract climate temporal env via gen_windows ---\n")

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

env_parts <- list()
for (j in seq_along(temporal_params)) {
  tp <- temporal_params[[j]]
  cat(sprintf("  [%d/%d] %s (%s)...", j, length(temporal_params),
              tp$prefix, paste(tp$variables, collapse = ", ")))

  raw <- gen_windows(
    pairs        = pairs,
    variables    = tp$variables,
    mstat        = tp$mstat,
    cstat        = tp$cstat,
    window       = tp$window,
    npy_src      = config$npy_src,
    start_year   = geonpy_start,
    python_exe   = config$python_exe,
    pyper_script = config$pyper_script
  )

  env_cols <- raw[, 9:ncol(raw), drop = FALSE]
  colnames(env_cols) <- paste(tp$prefix, colnames(env_cols), sep = "_")

  na_count <- sum(is.na(env_cols))
  cat(sprintf(" %d cols, %d NAs\n", ncol(env_cols), na_count))
  if (na_count > 0) {
    for (cn in names(env_cols)) {
      nn <- sum(is.na(env_cols[[cn]]))
      if (nn > 0) cat(sprintf("    %s: %d/%d NAs\n", cn, nn, n_pts))
    }
  }
  env_parts[[j]] <- env_cols
}

env_all <- do.call(cbind, env_parts)
cat(sprintf("\n  env_all (climate only): %d rows × %d cols, total NAs: %d\n",
            nrow(env_all), ncol(env_all), sum(is.na(env_all))))

# ---------------------------------------------------------------------------
# 5. Extract MODIS temporal variables
# ---------------------------------------------------------------------------
if (isTRUE(fit$add_modis)) {
  cat("\n--- STEP 5: Extract MODIS temporal variables ---\n")
  modis_dir   <- config$modis_dir
  modis_res   <- config$modis_resolution
  modis_vars  <- fit$modis_variables
  modis_range <- fit$modis_year_range

  pts_sp <- sp::SpatialPoints(data.frame(samp$lon, samp$lat))

  modis_yr1 <- data.frame(matrix(NA_real_, n_pts, length(modis_vars)))
  modis_yr2 <- data.frame(matrix(NA_real_, n_pts, length(modis_vars)))
  colnames(modis_yr1) <- paste0("temp_modis_", modis_vars, "_1")
  colnames(modis_yr2) <- paste0("temp_modis_", modis_vars, "_2")

  yr1_clamped <- pmin(pmax(as.integer(baseline_year), modis_range[1]), modis_range[2])
  yr2_clamped <- pmin(pmax(as.integer(target_year),   modis_range[1]), modis_range[2])
  unique_yrs  <- sort(unique(c(yr1_clamped, yr2_clamped)))
  cat(sprintf("  Clamped years: yr1=%d, yr2=%d\n", yr1_clamped, yr2_clamped))

  for (vi in seq_along(modis_vars)) {
    mv <- modis_vars[vi]
    for (yr in unique_yrs) {
      fname <- file.path(modis_dir,
                         paste0("modis_", yr, "_", mv, "_", modis_res, "_COG.tif"))
      cat(sprintf("  Loading: %s ... ", basename(fname)))
      if (!file.exists(fname)) { cat("NOT FOUND\n"); next }

      ras_m <- raster::raster(fname)
      vals  <- raster::extract(ras_m, pts_sp)
      na_ct <- sum(is.na(vals))
      cat(sprintf("extracted %d values, %d NAs (%.0f%%)\n",
                  length(vals), na_ct, 100 * na_ct / length(vals)))
      if (na_ct > 0) {
        cat(sprintf("    NA at point indices: %s\n",
                    paste(which(is.na(vals)), collapse = ", ")))
        cat(sprintf("    NA point coords:\n"))
        na_idx <- which(is.na(vals))
        for (ii in na_idx) {
          cat(sprintf("      [%d] lon=%.4f, lat=%.4f\n", ii, samp$lon[ii], samp$lat[ii]))
        }
      }

      idx1 <- which(yr1_clamped == yr)
      if (length(idx1) > 0) modis_yr1[idx1, vi] <- vals[idx1]

      idx2 <- which(yr2_clamped == yr)
      if (length(idx2) > 0) modis_yr2[idx2, vi] <- vals[idx2]
    }
  }

  cat(sprintf("\n  modis_yr1 NAs: %d / %d\n", sum(is.na(modis_yr1)), prod(dim(modis_yr1))))
  cat(sprintf("  modis_yr2 NAs: %d / %d\n", sum(is.na(modis_yr2)), prod(dim(modis_yr2))))
  print(head(modis_yr1))
  print(head(modis_yr2))

  env_all <- cbind(env_all, modis_yr1, modis_yr2)
  cat(sprintf("\n  env_all (with MODIS): %d rows × %d cols, total NAs: %d\n",
              nrow(env_all), ncol(env_all), sum(is.na(env_all))))
} else {
  cat("\n--- STEP 5: MODIS not enabled, skipping ---\n")
}

# ---------------------------------------------------------------------------
# 6. Split into env_s1 / env_s2 and clean names
# ---------------------------------------------------------------------------
cat("\n--- STEP 6: Split env_s1 / env_s2 ---\n")
idx_1  <- grep("_1$", names(env_all))
idx_2  <- grep("_2$", names(env_all))
env_s1 <- env_all[, idx_1, drop = FALSE]
env_s2 <- env_all[, idx_2, drop = FALSE]

strip_daterange <- function(x) gsub("\\d{6}-\\d{6}_", "", x)
names(env_s1) <- strip_daterange(names(env_s1))
names(env_s2) <- strip_daterange(names(env_s2))

cat(sprintf("  env_s1 cols: %s\n", paste(names(env_s1), collapse = ", ")))
cat(sprintf("  env_s2 cols: %s\n", paste(names(env_s2), collapse = ", ")))
cat(sprintf("  env_s1 NAs: %d, env_s2 NAs: %d\n",
            sum(is.na(env_s1)), sum(is.na(env_s2))))

# ---------------------------------------------------------------------------
# 7. Match predictors and test I_spline
# ---------------------------------------------------------------------------
cat("\n--- STEP 7: Match predictors and test I_spline ---\n")

n_preds       <- length(fit$predictors)
total_splines <- sum(fit$splines)

for (i in temp_idx) {
  pred_name <- fit$predictors[i]
  ns        <- fit$splines[i]
  coefs_i   <- fit$coefficients[(csp[i] + 1):(csp[i] + ns)]
  quants_i  <- fit$quantiles[(csp[i] + 1):(csp[i] + ns)]

  col_1 <- which(names(env_s1) == pred_name)
  pred_2 <- sub("_1$", "_2", pred_name)
  col_2 <- which(names(env_s2) == pred_2)

  if (length(col_1) == 0 || length(col_2) == 0) {
    cat(sprintf("  [%d] %-35s  NO MATCH (col_1=%d, col_2=%d)\n",
                i, pred_name, length(col_1), length(col_2)))
    cat(sprintf("       Looking for '%s' in env_s1 names: %s\n",
                pred_name, paste(names(env_s1), collapse = ", ")))
    next
  }

  v1 <- env_s1[, col_1]
  v2 <- env_s2[, col_2]
  na1 <- sum(is.na(v1)); na2 <- sum(is.na(v2))

  cat(sprintf("  [%d] %-35s  matched | v1 NAs: %d, v2 NAs: %d",
              i, pred_name, na1, na2))

  ## Try I_spline to see if it crashes
  for (sp in seq_len(ns)) {
    if (sp == 1L) {
      q1 <- quants_i[1]; q2 <- quants_i[1]; q3 <- quants_i[min(2, ns)]
    } else if (sp == ns) {
      q1 <- quants_i[max(1, ns - 1)]; q2 <- quants_i[ns]; q3 <- quants_i[ns]
    } else {
      q1 <- quants_i[sp - 1]; q2 <- quants_i[sp]; q3 <- quants_i[sp + 1]
    }

    result_ok <- tryCatch({
      spl1 <- I_spline(v1, q1, q2, q3)
      spl2 <- I_spline(v2, q1, q2, q3)
      spl_diff <- abs(spl1 - spl2)
      list(ok = TRUE,
           spl1_na = sum(is.na(spl1)),
           spl2_na = sum(is.na(spl2)),
           diff_na = sum(is.na(spl_diff)))
    }, error = function(e) {
      list(ok = FALSE, msg = conditionMessage(e))
    })

    if (!result_ok$ok) {
      cat(sprintf("\n       *** SPLINE %d CRASHED: %s", sp, result_ok$msg))
      cat(sprintf("\n       q1=%.4f, q2=%.4f, q3=%.4f", q1, q2, q3))
      cat(sprintf("\n       v1 sample: %s", paste(sprintf("%.4f", head(v1, 5)), collapse = ", ")))
      cat(sprintf("\n       v2 sample: %s", paste(sprintf("%.4f", head(v2, 5)), collapse = ", ")))
      cat(sprintf("\n       v1 has NAs: %s, v2 has NAs: %s",
                  any(is.na(v1)), any(is.na(v2))))
    } else if (result_ok$diff_na > 0) {
      cat(sprintf("\n       spline %d: spl1_na=%d, spl2_na=%d, diff_na=%d",
                  sp, result_ok$spl1_na, result_ok$spl2_na, result_ok$diff_na))
    }
  }
  cat("\n")
}

# ---------------------------------------------------------------------------
# 8. Try the full matrix multiply (coefficients may have NAs)
# ---------------------------------------------------------------------------
cat("\n--- STEP 8: Check coefficient vector ---\n")
na_coefs <- which(is.na(fit$coefficients))
if (length(na_coefs) > 0) {
  cat(sprintf("  WARNING: %d NA coefficients at positions: %s\n",
              length(na_coefs), paste(na_coefs, collapse = ", ")))
  cat("  These correspond to predictors:\n")
  for (pos in na_coefs) {
    ## Find which predictor this belongs to
    for (pi in seq_along(fit$predictors)) {
      sp_start <- csp[pi] + 1
      sp_end   <- csp[pi] + fit$splines[pi]
      if (pos >= sp_start && pos <= sp_end) {
        cat(sprintf("    coef[%d] -> %s (spline %d)\n",
                    pos, fit$predictors[pi], pos - csp[pi]))
        break
      }
    }
  }
} else {
  cat("  All coefficients are finite (no NAs)\n")
}

cat("\n=== DEBUG complete ===\n")
cat("If you see CRASHED lines above, those are the source of the error.\n")
cat("The likely fix: I_spline needs to handle NA input values gracefully.\n")
