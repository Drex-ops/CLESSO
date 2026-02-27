##############################################################################
##
## run_obsGDM.R  —  Single parameterised entry point for RECA obsGDM
##
## This replaces the 13 duplicated run_obsGDM_*.r scripts from OLD_RECA.
## All variant behaviour (AVES/PLANTS/VAS, biAverage, v2/v3 decomposition,
## date filtering, sample sizes, env variables) is controlled via config.R.
##
## Usage:
##   1. Edit config.R (or set environment variables) for your run
##   2. source("run_obsGDM.R")
##
## Pipeline:
##   Load data → siteAggregator → date filter → site.richness.extractor →
##   obsPairSampler → gen_windows (env extraction) → anomalies →
##   substrate extraction → splineData → fitGDM → diagnostics → save
##
##############################################################################

cat("=== RECA obsGDM ===\n")
cat("Loading configuration...\n")

# ---------------------------------------------------------------------------
# Source configuration and modules
# ---------------------------------------------------------------------------
this_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),           # works when sourced
  error = function(e) {
    if (nchar(getwd()) > 0) getwd()      # fallback: current working directory
    else stop("Cannot determine script directory.")
  }
)
source(file.path(this_dir, "src", "reca_version","config.R"))
source(file.path(config$r_dir, "utils.R"))
source(file.path(config$r_dir, "gdm_functions.R"))
source(file.path(config$r_dir, "site_aggregator.R"))
source(file.path(config$r_dir, "site_richness_extractor.R"))
source(file.path(config$r_dir, "obs_pair_sampler.R"))
source(file.path(config$r_dir, "gen_windows.R"))
source(file.path(config$r_dir, "plotting.R"))

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------
load_packages()
rasterOptions(tmpdir = config$raster_tmpdir)

# ===========================================================================
# STEP 1: Load and aggregate observations
# ===========================================================================
cat("\n--- Step 1: Load and aggregate observations ---\n")

obs_file <- file.path(config$data_dir, config$obs_csv)
if (!file.exists(obs_file)) stop(paste("Observation file not found:", obs_file))
dat <- read.csv(obs_file)
cat(sprintf("  Loaded %d records from %s\n", nrow(dat), config$obs_csv))

## Get grid parameters from reference raster
ras_sp <- raster(config$reference_raster)
res    <- res(ras_sp)[1]
box    <- extent(ras_sp)

## Aggregate to grid cells
datRED <- siteAggregator(dat, res, box)

## Clean sites where substrate gives NAs
test   <- is.na(extract(ras_sp, datRED[, c("lonID", "latID")]))
datRED <- datRED[!test, ]

## Save aggregated data
agg_file <- file.path(config$output_dir,
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

## Date filter
datRED <- datRED[datRED$eventDate != "", ]
date_test <- as.Date(datRED$eventDate) > (config$min_date %m+% years(config$date_offset_years))
datRED    <- datRED[date_test, ]
date_test <- as.Date(datRED$eventDate) < config$max_date
datRED    <- datRED[date_test, ]
datRED    <- droplevels(datRED)
cat(sprintf("  After date filter: %d records\n", nrow(datRED)))

## Prepare site × species data
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

## Reduce to unique site × species
LocDups <- paste(data$ID, data$species, sep = ":")
test    <- duplicated(LocDups)
data    <- data[!test, ]
data    <- data[order(data$ID), ]
data$row.count <- 1:nrow(data)
data$species   <- as.factor(data$species)

## Count records per unique site×species for m1 expansion
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

## Restore full dataset
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

## Expand m1 back to full dataset
row.nums <- 1:nrow(m1)
row.vect <- rep(row.nums, count)
m2       <- m1[row.vect, ]
gc()

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

obspairs_file <- file.path(config$output_dir,
  paste0("ObsPairsTable_RECA_", config$species_group, "_WindowTestRuns.rds"))
saveRDS(obsPairs_out, file = obspairs_file)

## Keep original in memory for the loop (avoids OneDrive read issues)
obsPairs_orig <- obsPairs_out

# ===========================================================================
# STEP 6: Loop over climate/weather window combinations
# ===========================================================================
cat("\n--- Step 6: Climate window loop ---\n")
biol_group <- config$species_group

for (c_yr in config$c_yrs) {
  for (w_yr in config$w_yrs) {

    cat(sprintf("\n>>> Climate: %d yrs | Weather: %d yrs\n", c_yr, w_yr))

    ## Reload obs pairs from in-memory copy and apply temporal filter
    obsPairs_out <- obsPairs_orig
    tst_1 <- as.Date(paste0(obsPairs_out$year1, "-01-01")) >
               (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
    tst_2 <- as.Date(paste0(obsPairs_out$year2, "-01-01")) >
               (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
    obsPairs_out <- obsPairs_out[tst_1 & tst_2, ]
    ext_data     <- obsPairs_out[, 2:9]

    save_prefix <- make_save_prefix(config, c_yr, w_yr)
    env_file    <- file.path(config$output_dir, paste0(save_prefix, "ObsEnvTable.RData"))

    # ------------------------------------------------------------------
    # STEP 6a: Environmental data extraction
    # ------------------------------------------------------------------
    get_env <- !config$skip_existing_env || !file.exists(env_file)

    if (get_env) {
      cat("  Extracting environmental data...\n")

      ## Build env params for this window combination
      init_params <- list()
      for (ep in config$env_params) {
        ## Climate window version
        init_params[[length(init_params) + 1]] <- list(
          variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
          window = c_yr, prefix = paste0(ep$cstat, "Xbr_", c_yr)
        )
        ## Weather window version
        init_params[[length(init_params) + 1]] <- list(
          variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
          window = w_yr, prefix = paste0(ep$cstat, "Xbr_", w_yr)
        )
      }

      ## Parallel env extraction
      n_workers <- min(length(init_params), config$cores_to_use)
      cl <- makeCluster(n_workers)
      registerDoSNOW(cl)

      env_outA <- foreach(x = 1:length(init_params), .combine = "cbind",
                          .packages = "arrow") %dopar% {
        out <- gen_windows(
          pairs        = ext_data,
          variables    = init_params[[x]]$variables,
          mstat        = init_params[[x]]$mstat,
          cstat        = init_params[[x]]$cstat,
          window       = init_params[[x]]$window,
          npy_src      = config$npy_src,
          start_year   = config$geonpy_start_year,
          python_exe   = config$python_exe,
          pyper_script = config$pyper_script,
          feather_tmpdir = config$feather_tmpdir
        )
        colnames(out) <- paste(init_params[[x]]$prefix, colnames(out), sep = "_")
        out[, 9:ncol(out)]
      }

      ## Bidirectional averaging (if enabled)
      if (config$biAverage) {
        cat("  Computing bidirectional average...\n")
        env_outB <- foreach(x = 1:length(init_params), .combine = "cbind",
                            .packages = "arrow") %dopar% {
          out <- gen_windows(
            pairs        = ext_data[, c(1, 2, 7, 8, 5, 6, 3, 4)],  # swap sites
            variables    = init_params[[x]]$variables,
            mstat        = init_params[[x]]$mstat,
            cstat        = init_params[[x]]$cstat,
            window       = init_params[[x]]$window,
            npy_src      = config$npy_src,
            start_year   = config$geonpy_start_year,
            python_exe   = config$python_exe,
            pyper_script = config$pyper_script,
            feather_tmpdir = config$feather_tmpdir
          )
          colnames(out) <- paste(init_params[[x]]$prefix, colnames(out), sep = "_")
          out[, 9:ncol(out)]
        }
        env_out <- (env_outA + env_outB) / 2
      } else {
        env_out <- env_outA
      }

      stopCluster(cl)
      registerDoSEQ()

      ## Split into site 1 / site 2
      env_out1 <- env_out[, grep("_1$", names(env_out))]
      env_out2 <- env_out[, grep("_2$", names(env_out))]

      ## Make anomalies (weather - climate)
      w_cols_1 <- grep(paste0("_", w_yr, "_"), colnames(env_out1))
      c_cols_1 <- grep(paste0("_", c_yr, "_"), colnames(env_out1))
      if (length(w_cols_1) == length(c_cols_1) && length(w_cols_1) > 0) {
        env_anom_out1 <- env_out1[, w_cols_1] - env_out1[, c_cols_1]
        colnames(env_anom_out1) <- gsub("191101-201712", "anom", colnames(env_anom_out1))
      } else {
        env_anom_out1 <- NULL
      }

      w_cols_2 <- grep(paste0("_", w_yr, "_"), colnames(env_out2))
      c_cols_2 <- grep(paste0("_", c_yr, "_"), colnames(env_out2))
      if (length(w_cols_2) == length(c_cols_2) && length(w_cols_2) > 0) {
        env_anom_out2 <- env_out2[, w_cols_2] - env_out2[, c_cols_2]
        colnames(env_anom_out2) <- gsub("191101-201712", "anom", colnames(env_anom_out2))
      } else {
        env_anom_out2 <- NULL
      }

      ## Extract substrate
      pnt1     <- SpatialPoints(data.frame(ext_data$Lon1, ext_data$Lat1))
      pnt2     <- SpatialPoints(data.frame(ext_data$Lon2, ext_data$Lat2))
      subs_brk <- brick(config$substrate_raster)
      env1_subs <- extract(subs_brk, pnt1)
      env2_subs <- extract(subs_brk, pnt2)
      colnames(env1_subs) <- paste0(colnames(env1_subs), "_1")
      colnames(env2_subs) <- paste0(colnames(env2_subs), "_2")

      ## Temperature range
      tnn_c1 <- grep(paste0("min.*_", c_yr, ".*TNn.*_1"), colnames(env_out1), ignore.case = TRUE)
      txx_c1 <- grep(paste0("max.*_", c_yr, ".*TXx.*_1"), colnames(env_out1), ignore.case = TRUE)
      tnn_w1 <- grep(paste0("min.*_", w_yr, ".*TNn.*_1"), colnames(env_out1), ignore.case = TRUE)
      txx_w1 <- grep(paste0("max.*_", w_yr, ".*TXx.*_1"), colnames(env_out1), ignore.case = TRUE)

      if (length(tnn_c1) > 0 && length(txx_c1) > 0) {
        Trng_1 <- abs(env_out1[, c(tnn_c1[1], tnn_w1[1])] - env_out1[, c(txx_c1[1], txx_w1[1])])
        names(Trng_1) <- c("Trng_15_191101-201712_1", "Trng_1_191101-201712_1")
      } else {
        Trng_1 <- NULL
      }

      tnn_c2 <- grep(paste0("min.*_", c_yr, ".*TNn.*_2"), colnames(env_out2), ignore.case = TRUE)
      txx_c2 <- grep(paste0("max.*_", c_yr, ".*TXx.*_2"), colnames(env_out2), ignore.case = TRUE)
      tnn_w2 <- grep(paste0("min.*_", w_yr, ".*TNn.*_2"), colnames(env_out2), ignore.case = TRUE)
      txx_w2 <- grep(paste0("max.*_", w_yr, ".*TXx.*_2"), colnames(env_out2), ignore.case = TRUE)

      if (length(tnn_c2) > 0 && length(txx_c2) > 0) {
        Trng_2 <- abs(env_out2[, c(tnn_c2[1], tnn_w2[1])] - env_out2[, c(txx_c2[1], txx_w2[1])])
        names(Trng_2) <- c("Trng_15_191101-201712_2", "Trng_1_191101-201712_2")
      } else {
        Trng_2 <- NULL
      }

      ## Combine everything
      parts <- list(obsPairs_out, env_out1, env1_subs, Trng_1, env_anom_out1,
                    env_out2, env2_subs, Trng_2, env_anom_out2)
      parts <- parts[!sapply(parts, is.null)]
      obsPairs_out <- do.call(cbind, parts)

      ## Clean up
      rm(env_out, env_outA, env_out1, env_out2, env1_subs, env2_subs)
      if (exists("env_outB")) rm(env_outB)
      gc()

      ## Save
      save(obsPairs_out, file = env_file)
      cat(sprintf("  Saved env table: %s\n", basename(env_file)))
    } else {
      cat("  Loading existing env table...\n")
      load(env_file)
    }

    # ------------------------------------------------------------------
    # STEP 6b: Clean NA and sentinel values
    # ------------------------------------------------------------------
    env_cols  <- 23:ncol(obsPairs_out)
    test_na   <- is.na(rowSums(obsPairs_out[, env_cols]))
    obsPairs_out <- obsPairs_out[!test_na, ]

    ## Remove -9999 sentinel values
    sentinel_test <- rep(0, nrow(obsPairs_out))
    for (col in env_cols) {
      sentinel_test <- sentinel_test + (obsPairs_out[, col] == -9999)
    }
    obsPairs_out <- obsPairs_out[sentinel_test == 0, ]
    cat(sprintf("  After cleaning: %d pairs\n", nrow(obsPairs_out)))

    # ------------------------------------------------------------------
    # STEP 6c: Apply decomposition filter (v3 only keeps same_site | same_time)
    # ------------------------------------------------------------------
    if (config$decomposition == "v3") {
      siteID_1  <- paste(obsPairs_out$Lon1, obsPairs_out$Lat1, sep = "~")
      siteID_2  <- paste(obsPairs_out$Lon2, obsPairs_out$Lat2, sep = "~")
      same_site <- siteID_1 == siteID_2
      same_time <- obsPairs_out$year1 == obsPairs_out$year2
      same_time[same_site] <- FALSE
      keep <- same_site | same_time
      cat(sprintf("  v3 decomposition: keeping %d of %d pairs (same_site|same_time)\n",
                  sum(keep), length(keep)))
    } else {
      keep <- rep(TRUE, nrow(obsPairs_out))
    }

    # ------------------------------------------------------------------
    # STEP 6d: I-spline transformation
    # ------------------------------------------------------------------
    cat("  Computing I-spline basis...\n")
    toSpline <- obsPairs_out[, env_cols]
    ## Remove substrate-only columns if present (handled separately in some variants)
    ## This preserves the structure from the original scripts
    splined  <- splineData(toSpline)

    if (config$decomposition == "v3") {
      splined_new <- splined[keep, ]
    } else {
      splined_new <- splined
    }

    # ------------------------------------------------------------------
    # STEP 6e: Fit GDM
    # ------------------------------------------------------------------
    cat("  Fitting GDM...\n")
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

    # ------------------------------------------------------------------
    # STEP 6f: Diagnostics and save
    # ------------------------------------------------------------------
    out_prefix <- file.path(config$output_dir, save_prefix)

    ## Deviance
    gdm_dev <- RsqGLM(obs = obsGDM_1$y, pred = fitted(obsGDM_1))
    save(gdm_dev, file = paste0(out_prefix, "DevianceCalcs.RData"))

    D2 <- (obsGDM_1$null.deviance - obsGDM_1$deviance) / obsGDM_1$null.deviance
    cat(sprintf("  Deviance explained: %.4f\n", D2))
    cat(sprintf("  Nagelkerke R²:     %.4f\n", gdm_dev$Nagelkerke))
    save(D2, file = paste0(out_prefix, "D2_deviance.RData"))

    ## Coefficients
    coefs <- coef(obsGDM_1)
    save(coefs, file = paste0(out_prefix, "coefficients.RData"))

    ## Build fit object for plotting
    fit <- list()
    fit$intercept    <- coef(obsGDM_1)[1]
    fit$sample       <- nrow(mod_ready)
    fit$predictors   <- gsub("191101-201712_", "",
                             gsub("_spl1", "", colnames(splined)[grep("_spl1", colnames(splined))]))
    fit$coefficients <- coef(obsGDM_1)[-1]
    fit$coefficients[is.na(fit$coefficients)] <- 0

    ## Compute quantiles from the data
    nc  <- ncol(toSpline)
    nc2 <- nc / 2
    X1  <- toSpline[, 1:nc2]
    X2  <- toSpline[, (nc2 + 1):nc]
    nms <- names(X1); names(X2) <- nms
    sv  <- c(rep(1, nrow(X1)), rep(2, nrow(X2)))
    XX  <- rbind(X1, X2)
    fit$quantiles  <- unlist(lapply(1:ncol(XX), function(x) quantile(XX[, x], c(0, 0.5, 1))))
    fit$splines    <- rep(3, ncol(XX))
    fit$predicted  <- fitted(obsGDM_1)
    fit$ecological <- obsGDM_1$linear.predictors
    save(fit, file = paste0(out_prefix, "fittedGDM.RData"))

    ## Diagnostic plots
    tiff(paste0(out_prefix, "GDM-ObsDiag.tif"),
         height = 6, width = 6, units = "in", res = 200, compression = "lzw")
    obs.gdm.plot(obsGDM_1, save_prefix, w, Is = fit$intercept)
    dev.off()

    pdf(paste0(out_prefix, "GDM-gdmDiag.pdf"))
    gdm.spline.plot(fit)
    dev.off()

    cat(sprintf("  Saved: %s*\n", save_prefix))

  }  # end w_yr loop
}  # end c_yr loop

cat("\n=== RECA obsGDM complete ===\n")
