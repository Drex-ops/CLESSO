##############################################################################
##
## run_obsGDM_window_sweep.R  --  Climate window tuning for RECA obsGDM
##
## Runs Steps 1-5 once (data load, w estimation, data prep, site×species
## matrix, observation-pair sampling) then loops Step 6 (env extraction ->
## spline -> fit -> diagnostics) over a range of climate windows (3–60 yr).
##
## Only model-fit statistics are saved (no full fit objects), collected into
## a single summary table for downstream climate-window selection.
##
## Usage:
##   1. Edit config.R as normal
##   2. source("run_obsGDM_window_sweep.R")
##
##############################################################################

cat("=== RECA obsGDM -- Climate Window Sweep ===\n")
cat("Loading configuration...\n")

# ---------------------------------------------------------------------------
# Source configuration and modules
# ---------------------------------------------------------------------------
this_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    if (nchar(getwd()) > 0) getwd()
    else stop("Cannot determine script directory.")
  }
)

config_path <- file.path(this_dir, "config.R")
if (!file.exists(config_path)) {
  config_path <- file.path(this_dir, "src", "reca_STresiduals", "config.R")
}
source(config_path)
save_config_snapshot()
source(file.path(config$r_dir, "utils.R"))
source(file.path(config$r_dir, "gdm_functions.R"))
source(file.path(config$r_dir, "site_aggregator.R"))
source(file.path(config$r_dir, "site_richness_extractor.R"))
source(file.path(config$r_dir, "obs_pair_sampler.R"))
source(file.path(config$r_dir, "gen_windows.R"))
source(file.path(config$r_dir, "run_chunked_env.R"))
source(file.path(config$r_dir, "plotting.R"))

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------
load_packages()
rasterOptions(tmpdir = config$raster_tmpdir)

# ---------------------------------------------------------------------------
# Climate window range to sweep
# ---------------------------------------------------------------------------
window_range <- 3L:60L
cat(sprintf("  Will sweep %d climate windows: %d–%d yr\n",
            length(window_range), min(window_range), max(window_range)))

# ===========================================================================
# STEP 1: Load and aggregate observations
# ===========================================================================
cat("\n--- Step 1: Load and aggregate observations ---\n")

obs_file <- file.path(config$data_dir, config$obs_csv)
if (!file.exists(obs_file)) stop(paste("Observation file not found:", obs_file))
dat <- read.csv(obs_file)
cat(sprintf("  Loaded %d records from %s\n", nrow(dat), config$obs_csv))

ras_sp <- raster(config$reference_raster)
res    <- res(ras_sp)[1]
box    <- extent(ras_sp)

datRED <- siteAggregator(dat, res, box)

test   <- is.na(extract(ras_sp, datRED[, c("lonID", "latID")]))
datRED <- datRED[!test, ]

agg_file <- file.path(config$run_output_dir,
                       paste0(config$species_group, "_aggregated_basicFilt.RData"))
save(datRED, file = agg_file)
cat(sprintf("  Aggregated data: %d site-species-date records\n", nrow(datRED)))

# ===========================================================================
# STEP 2: Estimate mismatch/match ratio (w)
# ===========================================================================
cat("\n--- Step 2: Estimate mismatch/match ratio ---\n")
w <- estimate_w(datRED, nSamples = config$w_estimation_samples)

# ===========================================================================
# STEP 3: Prepare data for sampling
# ===========================================================================
cat("\n--- Step 3: Prepare data for sampling ---\n")

datRED <- datRED[datRED$eventDate != "", ]
date_test <- as.Date(datRED$eventDate) > (config$min_date %m+% years(config$date_offset_years))
datRED    <- datRED[date_test, ]
date_test <- as.Date(datRED$eventDate) < config$max_date
datRED    <- datRED[date_test, ]
datRED    <- droplevels(datRED)
cat(sprintf("  After date filter: %d records\n", nrow(datRED)))

data <- data.frame(
  ID        = datRED$ID,
  Latitude  = datRED$latID,
  Longitude = datRED$lonID,
  species   = datRED$gen_spec,
  nRecords  = datRED$nRecords,
  nRecords.exDateLocDups = datRED$nRecords.exDateLocDups,
  nSiteVisits = datRED$nSiteVisits,
  richness  = datRED$richness,
  stringsAsFactors = FALSE
)

LocDups <- paste(data$ID, data$species, sep = ":")
test    <- duplicated(LocDups)
data    <- data[!test, ]
data    <- data[order(data$ID), ]
data$row.count <- 1:nrow(data)
data$species   <- as.factor(data$species)

LocDups <- as.factor(paste(datRED$ID, datRED$gen_spec, sep = ":"))
ones    <- rep(1, nrow(datRED))
count   <- bySum(ones, LocDups)

# ===========================================================================
# STEP 4: Build site × species matrix
# ===========================================================================
cat("\n--- Step 4: Build site x species matrix ---\n")
cl <- makeCluster(config$cores_to_use)
registerDoSNOW(cl)

frog.auGrid <- site.richness.extractor.bigData(frog.auGrid = data)

frog.auGrid <- data.frame(
  ID        = datRED$ID,
  Latitude  = datRED$latID,
  Longitude = datRED$lonID,
  species   = datRED$gen_spec,
  eventDate = as.character(datRED$eventDate),
  nRecords  = datRED$nRecords,
  nRecords.exDateLocDups = datRED$nRecords.exDateLocDups,
  nSiteVisits = datRED$nSiteVisits,
  richness  = datRED$richness,
  stringsAsFactors = FALSE,
  Site.Richness = datRED$richness
)

## Build full-record proxy (maps each row of frog.auGrid to its site)
full_site_map <- match(as.character(frog.auGrid$ID), site_levels)
m2 <- site_species_proxy(site_sp_matrix, full_site_map)
rm(m1); gc()

# ===========================================================================
# STEP 5: Run observation-pair sampler
# ===========================================================================
cat("\n--- Step 5: Run observation-pair sampler ---\n")
obsPairs_out <- obsPairSampler.bigData.RECA(
  frog.auGrid, config$nMatch, m1 = m2,
  richness = TRUE, speciesThreshold = config$species_threshold,
  coresToUse = config$cores_to_use
)
registerDoSEQ()

obspairs_file <- file.path(config$run_output_dir,
  paste0("ObsPairsTable_RECA_", config$species_group, "_WindowSweep.rds"))
saveRDS(obsPairs_out, file = obspairs_file)

## Keep original in memory for the loop
obsPairs_orig <- obsPairs_out

## Pre-extract substrate (time-invariant, only needs doing once)
cat("\n--- Pre-extracting substrate (shared across all windows) ---\n")
ext_data_orig <- obsPairs_orig[, 2:9]
pnt1     <- SpatialPoints(data.frame(ext_data_orig[, 1], ext_data_orig[, 2]))
pnt2     <- SpatialPoints(data.frame(ext_data_orig[, 5], ext_data_orig[, 6]))
subs_brk <- brick(config$substrate_raster)
env1_subs_all <- extract(subs_brk, pnt1)
env2_subs_all <- extract(subs_brk, pnt2)
colnames(env1_subs_all) <- paste0(colnames(env1_subs_all), "_1")
colnames(env2_subs_all) <- paste0(colnames(env2_subs_all), "_2")
cat(sprintf("  Substrate columns: %d per site\n", ncol(env1_subs_all)))

# ===========================================================================
# STEP 6: Loop over climate windows
# ===========================================================================
cat("\n")
cat("###########################################################################\n")
cat("##  CLIMATE WINDOW SWEEP: Looping Step 6 over windows ", min(window_range),
    "–", max(window_range), " yr\n")
cat("###########################################################################\n")

## Storage for summary statistics
sweep_results <- data.frame(
  climate_window  = integer(),
  n_pairs_raw     = integer(),
  n_pairs_clean   = integer(),
  n_pairs_used    = integer(),
  intercept       = numeric(),
  D2              = numeric(),
  nagelkerke_r2   = numeric(),
  null_deviance   = numeric(),
  residual_deviance = numeric(),
  AIC             = numeric(),
  n_predictors    = integer(),
  n_nonzero_coefs = integer(),
  total_coef_sum  = numeric(),
  temporal_coef_sum  = numeric(),
  spatial_coef_sum   = numeric(),
  substrate_coef_sum = numeric(),
  temporal_weight_pct = numeric(),
  elapsed_min     = numeric(),
  status          = character(),
  stringsAsFactors = FALSE
)

sweep_t0 <- proc.time()

for (wi in seq_along(window_range)) {
  c_yr <- window_range[wi]

  cat(sprintf("\n=== Window %d/%d: %d yr ===\n",
              wi, length(window_range), c_yr))
  iter_t0 <- proc.time()

  ## -----------------------------------------------------------------
  ## 6a: Temporal filter and env extraction
  ## -----------------------------------------------------------------
  obsPairs_out      <- obsPairs_orig
  earliest_year_all <- pmin(obsPairs_out$year1, obsPairs_out$year2)
  tst <- (earliest_year_all - c_yr) >= config$geonpy_start_year
  obsPairs_out <- obsPairs_out[tst, ]
  ext_data     <- obsPairs_out[, 2:9]
  n_pairs_raw  <- nrow(obsPairs_out)
  cat(sprintf("  Pairs after temporal filter: %d\n", n_pairs_raw))

  if (n_pairs_raw < 100) {
    cat("  [SKIP] Too few pairs -- skipping this window\n")
    sweep_results <- rbind(sweep_results, data.frame(
      climate_window = c_yr, n_pairs_raw = n_pairs_raw,
      n_pairs_clean = NA, n_pairs_used = NA, intercept = NA,
      D2 = NA, nagelkerke_r2 = NA, null_deviance = NA,
      residual_deviance = NA, AIC = NA, n_predictors = NA,
      n_nonzero_coefs = NA, total_coef_sum = NA,
      temporal_coef_sum = NA, spatial_coef_sum = NA,
      substrate_coef_sum = NA, temporal_weight_pct = NA,
      elapsed_min = NA, status = "too_few_pairs",
      stringsAsFactors = FALSE
    ))
    next
  }

  save_prefix <- paste0(config$species_group, "_",
                        format(config$nMatch / 1e6, nsmall = 0), "mil_",
                        c_yr, "climWin_STresid_")
  if (config$biAverage)              save_prefix <- paste0(save_prefix, "biAverage_")
  if (config$decomposition != "none") save_prefix <- paste0(config$decomposition, "_", save_prefix)

  env_file <- file.path(config$run_output_dir, paste0(save_prefix, "ObsEnvTable.RData"))
  get_env  <- !config$skip_existing_env || !file.exists(env_file)

  status <- tryCatch({

    if (get_env) {
      cat("  Extracting environmental data ...\n")

      earliest_year  <- pmin(ext_data[, 3], ext_data[, 7])
      earliest_month <- ifelse(ext_data[, 3] <= ext_data[, 7],
                               ext_data[, 4], ext_data[, 8])

      spatial_pairs <- data.frame(
        Lon1 = ext_data[, 1], Lat1 = ext_data[, 2],
        year1 = earliest_year, month1 = earliest_month,
        Lon2 = ext_data[, 5], Lat2 = ext_data[, 6],
        year2 = earliest_year, month2 = earliest_month
      )

      temporal_pairs <- data.frame(
        Lon1 = ext_data[, 5], Lat1 = ext_data[, 6],
        year1 = earliest_year, month1 = earliest_month,
        Lon2 = ext_data[, 5], Lat2 = ext_data[, 6],
        year2 = ext_data[, 7], month2 = ext_data[, 8]
      )

      spatial_params  <- list()
      temporal_params <- list()
      for (ep in config$env_params) {
        spatial_params[[length(spatial_params) + 1]] <- list(
          variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
          window = c_yr, prefix = paste0("spat_", ep$cstat)
        )
        temporal_params[[length(temporal_params) + 1]] <- list(
          variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
          window = c_yr, prefix = paste0("temp_", ep$cstat)
        )
      }

      n_workers <- min(length(spatial_params) + length(temporal_params),
                       config$cores_to_use)
      cl <- makeCluster(n_workers)
      registerDoSNOW(cl)

      env_spatA <- run_chunked_env(spatial_pairs, spatial_params, "Spatial-A")

      if (config$biAverage) {
        env_spatB <- run_chunked_env(spatial_pairs, spatial_params, "Spatial-B",
                                      swap_sites = TRUE)
        env_spatial <- (env_spatA + env_spatB) / 2
      } else {
        env_spatial <- env_spatA
      }

      env_temporal <- run_chunked_env(temporal_pairs, temporal_params, "Temporal")

      stopCluster(cl)
      registerDoSEQ()

      env_spat1 <- env_spatial[, grep("_1$", names(env_spatial))]
      env_spat2 <- env_spatial[, grep("_2$", names(env_spatial))]
      env_temp1 <- env_temporal[, grep("_1$", names(env_temporal))]
      env_temp2 <- env_temporal[, grep("_2$", names(env_temporal))]

      ## Subset the pre-extracted substrate to the filtered rows
      keep_rows <- which(tst)
      env1_subs <- env1_subs_all[keep_rows, , drop = FALSE]
      env2_subs <- env2_subs_all[keep_rows, , drop = FALSE]

      parts <- list(obsPairs_out,
                    env_spat1, env1_subs, env_temp1,
                    env_spat2, env2_subs, env_temp2)
      parts <- parts[!sapply(parts, is.null)]
      obsPairs_out <- do.call(cbind, parts)

      rm(env_spatial, env_spatA, env_spat1, env_spat2,
         env_temporal, env_temp1, env_temp2, env1_subs, env2_subs)
      if (exists("env_spatB")) rm(env_spatB)
      gc()

      save(obsPairs_out, file = env_file)
      cat(sprintf("  Saved env table: %s\n", basename(env_file)))
    } else {
      cat("  Loading existing env table ...\n")
      load(env_file)
    }

    ## -----------------------------------------------------------------
    ## 6b: Clean NA and sentinel values
    ## -----------------------------------------------------------------
    env_cols  <- 23:ncol(obsPairs_out)
    test_na   <- is.na(rowSums(obsPairs_out[, env_cols]))
    obsPairs_out <- obsPairs_out[!test_na, ]

    sentinel_test <- rep(0, nrow(obsPairs_out))
    for (col in env_cols) {
      sentinel_test <- sentinel_test + (obsPairs_out[, col] == -9999)
    }
    obsPairs_out <- obsPairs_out[sentinel_test == 0, ]
    n_pairs_clean <- nrow(obsPairs_out)
    cat(sprintf("  After cleaning: %d pairs\n", n_pairs_clean))

    ## -----------------------------------------------------------------
    ## 6c: Decomposition filter
    ## -----------------------------------------------------------------
    if (config$decomposition == "v3") {
      siteID_1  <- paste(obsPairs_out$Lon1, obsPairs_out$Lat1, sep = "~")
      siteID_2  <- paste(obsPairs_out$Lon2, obsPairs_out$Lat2, sep = "~")
      same_site <- siteID_1 == siteID_2
      same_time <- obsPairs_out$year1 == obsPairs_out$year2
      same_time[same_site] <- FALSE
      keep <- same_site | same_time
    } else {
      keep <- rep(TRUE, nrow(obsPairs_out))
    }
    n_pairs_used <- sum(keep)

    ## -----------------------------------------------------------------
    ## 6d: I-spline transformation
    ## -----------------------------------------------------------------
    cat("  Computing I-spline basis ...\n")
    toSpline <- obsPairs_out[, env_cols]
    splined  <- splineData_fast(toSpline)

    if (config$decomposition == "v3") {
      splined_new <- splined[keep, ]
    } else {
      splined_new <- splined
    }

    ## -----------------------------------------------------------------
    ## 6e: Fit GDM
    ## -----------------------------------------------------------------
    cat("  Fitting GDM ...\n")
    match_response <- if (config$decomposition == "v3") {
      obsPairs_out$Match[keep]
    } else {
      obsPairs_out$Match
    }

    mod_ready <- cbind(Match = match_response, as.data.frame(splined_new))
    colnames(mod_ready) <- gsub("191101-201712_", "", colnames(mod_ready))

    f1      <- paste(colnames(mod_ready)[-1], collapse = "+")
    formula <- as.formula(paste(colnames(mod_ready)[1], "~", f1, sep = ""))
    obsGDM_1 <- fitGDM(formula = formula, data = mod_ready)

    ## -----------------------------------------------------------------
    ## 6f: Collect statistics
    ## -----------------------------------------------------------------
    gdm_dev <- RsqGLM(obs = obsGDM_1$y, pred = fitted(obsGDM_1))
    D2 <- (obsGDM_1$null.deviance - obsGDM_1$deviance) / obsGDM_1$null.deviance

    cat(sprintf("  D² = %.4f  |  Nagelkerke R² = %.4f  |  AIC = %.1f\n",
                D2, gdm_dev$Nagelkerke, AIC(obsGDM_1)))

    ## Coefficient breakdown by predictor type
    coefs <- coef(obsGDM_1)[-1]
    coefs[is.na(coefs)] <- 0
    pred_names <- gsub("_spl\\d+$", "", names(coefs))

    temporal_mask  <- grepl("^temp_", pred_names)
    spatial_mask   <- grepl("^spat_", pred_names)
    substrate_mask <- !temporal_mask & !spatial_mask

    iter_elapsed <- (proc.time() - iter_t0)["elapsed"]

    total_sum     <- sum(coefs)
    temporal_sum  <- sum(coefs[temporal_mask])
    spatial_sum   <- sum(coefs[spatial_mask])
    substrate_sum <- sum(coefs[substrate_mask])
    temporal_pct  <- if (total_sum > 0) 100 * temporal_sum / total_sum else NA_real_

    sweep_results <<- rbind(sweep_results, data.frame(
      climate_window    = c_yr,
      n_pairs_raw       = n_pairs_raw,
      n_pairs_clean     = n_pairs_clean,
      n_pairs_used      = n_pairs_used,
      intercept         = coef(obsGDM_1)[1],
      D2                = D2,
      nagelkerke_r2     = gdm_dev$Nagelkerke,
      null_deviance     = obsGDM_1$null.deviance,
      residual_deviance = obsGDM_1$deviance,
      AIC               = AIC(obsGDM_1),
      n_predictors      = length(unique(pred_names)),
      n_nonzero_coefs   = sum(coefs != 0),
      total_coef_sum    = total_sum,
      temporal_coef_sum = temporal_sum,
      spatial_coef_sum  = spatial_sum,
      substrate_coef_sum = substrate_sum,
      temporal_weight_pct = temporal_pct,
      elapsed_min       = iter_elapsed / 60,
      status            = "ok",
      stringsAsFactors  = FALSE
    ))

    "ok"
  }, error = function(e) {
    msg <- conditionMessage(e)
    warning(sprintf("  [ERROR] Window %d yr failed: %s", c_yr, msg))
    sweep_results <<- rbind(sweep_results, data.frame(
      climate_window = c_yr, n_pairs_raw = n_pairs_raw,
      n_pairs_clean = NA, n_pairs_used = NA, intercept = NA,
      D2 = NA, nagelkerke_r2 = NA, null_deviance = NA,
      residual_deviance = NA, AIC = NA, n_predictors = NA,
      n_nonzero_coefs = NA, total_coef_sum = NA,
      temporal_coef_sum = NA, spatial_coef_sum = NA,
      substrate_coef_sum = NA, temporal_weight_pct = NA,
      elapsed_min = NA, status = paste0("error: ", msg),
      stringsAsFactors = FALSE
    ))
    msg
  })

  ## Progress update
  total_elapsed <- (proc.time() - sweep_t0)["elapsed"]
  rate <- wi / total_elapsed
  remaining <- (length(window_range) - wi) / rate
  cat(sprintf("  [Progress] %d/%d done | %.1f min elapsed | ~%.1f min remaining\n",
              wi, length(window_range), total_elapsed / 60, remaining / 60))

  ## Incremental save after each iteration
  sweep_csv <- file.path(config$run_output_dir,
    paste0(config$species_group, "_window_sweep_stats.csv"))
  write.csv(sweep_results, sweep_csv, row.names = FALSE)

  gc()
}

# ===========================================================================
# 7. Final summary
# ===========================================================================
total_time <- (proc.time() - sweep_t0)["elapsed"]

cat("\n")
cat("###########################################################################\n")
cat("##  SWEEP COMPLETE\n")
cat("###########################################################################\n")
cat(sprintf("  Total time: %.1f min\n", total_time / 60))
cat(sprintf("  Windows attempted: %d  |  Successful: %d  |  Failed: %d\n",
            nrow(sweep_results),
            sum(sweep_results$status == "ok"),
            sum(sweep_results$status != "ok")))

## Save final CSV
sweep_csv <- file.path(config$run_output_dir,
  paste0(config$species_group, "_window_sweep_stats.csv"))
write.csv(sweep_results, sweep_csv, row.names = FALSE)
cat(sprintf("  Saved: %s\n", basename(sweep_csv)))

## Save RDS with full results + run metadata
sweep_rds <- file.path(config$run_output_dir,
  paste0(config$species_group, "_window_sweep_stats.rds"))
saveRDS(list(
  results        = sweep_results,
  species_group  = config$species_group,
  nMatch         = config$nMatch,
  biAverage      = config$biAverage,
  decomposition  = config$decomposition,
  w_ratio        = w,
  window_range   = window_range,
  total_time_min = total_time / 60,
  run_timestamp  = Sys.time()
), file = sweep_rds)
cat(sprintf("  Saved: %s\n", basename(sweep_rds)))

# ---------------------------------------------------------------------------
# 8. Diagnostic plots
# ---------------------------------------------------------------------------
cat("\n--- Generating sweep diagnostic plots ---\n")

ok_rows <- sweep_results[sweep_results$status == "ok", ]

if (nrow(ok_rows) >= 3) {
  pdf_file <- file.path(config$run_output_dir,
    paste0(config$species_group, "_window_sweep_plots.pdf"))
  pdf(pdf_file, width = 14, height = 16)

  par(mfrow = c(4, 2), mar = c(5, 5, 3, 2), oma = c(0, 0, 3, 0))

  ## Panel 1: D² vs climate window
  plot(ok_rows$climate_window, ok_rows$D2, type = "b", pch = 19, col = "#B2182B",
       xlab = "Climate Window (years)", ylab = expression(D^2),
       main = "Deviance Explained")
  best_D2 <- ok_rows$climate_window[which.max(ok_rows$D2)]
  abline(v = best_D2, lty = 2, col = "grey50")
  text(best_D2, max(ok_rows$D2, na.rm = TRUE),
       labels = sprintf("best = %d yr", best_D2), pos = 4, cex = 0.8)

  ## Panel 2: Nagelkerke R² vs climate window
  plot(ok_rows$climate_window, ok_rows$nagelkerke_r2, type = "b", pch = 19, col = "#2166AC",
       xlab = "Climate Window (years)", ylab = "Nagelkerke R²",
       main = "Nagelkerke R²")
  best_nag <- ok_rows$climate_window[which.max(ok_rows$nagelkerke_r2)]
  abline(v = best_nag, lty = 2, col = "grey50")
  text(best_nag, max(ok_rows$nagelkerke_r2, na.rm = TRUE),
       labels = sprintf("best = %d yr", best_nag), pos = 4, cex = 0.8)

  ## Panel 3: AIC vs climate window
  plot(ok_rows$climate_window, ok_rows$AIC, type = "b", pch = 19, col = "#238B45",
       xlab = "Climate Window (years)", ylab = "AIC",
       main = "AIC (lower is better)")
  best_aic <- ok_rows$climate_window[which.min(ok_rows$AIC)]
  abline(v = best_aic, lty = 2, col = "grey50")
  text(best_aic, min(ok_rows$AIC, na.rm = TRUE),
       labels = sprintf("best = %d yr", best_aic), pos = 4, cex = 0.8)

  ## Panel 4: Temporal weight %
  plot(ok_rows$climate_window, ok_rows$temporal_weight_pct, type = "b", pch = 19, col = "#7570B3",
       xlab = "Climate Window (years)", ylab = "Temporal Weight (%)",
       main = "Temporal Coefficient Weight")

  ## Panel 5: Total coefficient sum
  plot(ok_rows$climate_window, ok_rows$total_coef_sum, type = "b", pch = 19, col = "grey30",
       xlab = "Climate Window (years)", ylab = "Total Coef Sum",
       main = "Total Coefficient Sum")

  ## Panel 6: Number of non-zero coefficients
  plot(ok_rows$climate_window, ok_rows$n_nonzero_coefs, type = "b", pch = 19, col = "#D95F02",
       xlab = "Climate Window (years)", ylab = "Non-zero Coefficients",
       main = "Active Coefficients")

  ## Panel 7: Pairs available
  plot(ok_rows$climate_window, ok_rows$n_pairs_used / 1e3, type = "b", pch = 19, col = "#1B9E77",
       xlab = "Climate Window (years)", ylab = "Pairs (×1000)",
       main = "Usable Observation Pairs")

  ## Panel 8: Elapsed time per window
  plot(ok_rows$climate_window, ok_rows$elapsed_min, type = "b", pch = 19, col = "#E7298A",
       xlab = "Climate Window (years)", ylab = "Elapsed (min)",
       main = "Computation Time per Window")

  mtext(sprintf("Climate Window Sweep -- %s | nMatch = %s | biAverage = %s | decomp = %s",
                config$species_group,
                format(config$nMatch, big.mark = ","),
                config$biAverage,
                config$decomposition),
        outer = TRUE, cex = 1.0, font = 2)

  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_file)))

  ## Print best-window summary
  cat("\n--- Best windows ---\n")
  cat(sprintf("  Max D²:           %d yr (D² = %.4f)\n", best_D2, max(ok_rows$D2, na.rm = TRUE)))
  cat(sprintf("  Max Nagelkerke:   %d yr (R² = %.4f)\n", best_nag, max(ok_rows$nagelkerke_r2, na.rm = TRUE)))
  cat(sprintf("  Min AIC:          %d yr (AIC = %.1f)\n", best_aic, min(ok_rows$AIC, na.rm = TRUE)))
} else {
  cat("  [SKIP] Too few successful windows for plots\n")
}

cat(sprintf("\n=== Window sweep complete (%.1f min) ===\n", total_time / 60))
