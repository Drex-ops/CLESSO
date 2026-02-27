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
source(file.path(this_dir, "src", "reca_STresiduals","config.R"))
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
# STEP 6: Environmental variable extraction (spatial-temporal residuals)
#
# Approach:
#   For each observation pair we extract env variables in two ways:
#   SPATIAL  – both sites evaluated at the earliest year of the pair
#              (difference captures purely spatial variation)
#   TEMPORAL – site 2 evaluated at the earliest year AND at its
#              observation year (difference captures temporal change
#              at that location)
# ===========================================================================
cat("\n--- Step 6: Environmental variable extraction (ST residuals) ---\n")
biol_group <- config$species_group
c_yr       <- config$climate_window

## Apply temporal filter: ensure earliest_year - climate_window >= geonpy_start_year
obsPairs_out      <- obsPairs_orig
earliest_year_all <- pmin(obsPairs_out$year1, obsPairs_out$year2)
tst <- (earliest_year_all - c_yr) >= config$geonpy_start_year
obsPairs_out <- obsPairs_out[tst, ]
ext_data     <- obsPairs_out[, 2:9]
cat(sprintf("  Pairs after temporal filter: %d\n", nrow(obsPairs_out)))

save_prefix <- paste0(config$species_group, "_",
                      format(config$nMatch / 1e6, nsmall = 0), "mil_",
                      c_yr, "climWin_STresid_")
if (config$biAverage)              save_prefix <- paste0(save_prefix, "biAverage_")
if (config$decomposition != "none") save_prefix <- paste0(config$decomposition, "_", save_prefix)

env_file <- file.path(config$output_dir, paste0(save_prefix, "ObsEnvTable.RData"))

# ------------------------------------------------------------------
# STEP 6a: Environmental data extraction
# ------------------------------------------------------------------
get_env <- !config$skip_existing_env || !file.exists(env_file)

if (get_env) {
  cat("  Extracting environmental data...\n")

  ## Per-pair earliest year and corresponding month
  earliest_year  <- pmin(ext_data[, 3], ext_data[, 7])   # min(year1, year2)
  earliest_month <- ifelse(ext_data[, 3] <= ext_data[, 7],
                           ext_data[, 4], ext_data[, 8])  # month of earlier obs

  ## ----- SPATIAL pairs -----
  ## Both sites at the earliest year: difference = purely spatial
  spatial_pairs <- data.frame(
    Lon1   = ext_data[, 1],
    Lat1   = ext_data[, 2],
    year1  = earliest_year,
    month1 = earliest_month,
    Lon2   = ext_data[, 5],
    Lat2   = ext_data[, 6],
    year2  = earliest_year,
    month2 = earliest_month
  )

  ## ----- TEMPORAL pairs -----
  ## Site 2 at earliest year vs. site 2 at observation year:
  ## difference = temporal change at site 2
  temporal_pairs <- data.frame(
    Lon1   = ext_data[, 5],   # site 2 coords
    Lat1   = ext_data[, 6],
    year1  = earliest_year,   # baseline (earliest year)
    month1 = earliest_month,
    Lon2   = ext_data[, 5],   # same site 2 coords
    Lat2   = ext_data[, 6],
    year2  = ext_data[, 7],   # actual observation year2
    month2 = ext_data[, 8]    # actual observation month2
  )

  ## Build env param lists for the fixed climate window
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

  ## --- Parallel extraction: SPATIAL component ---
  n_workers <- min(length(spatial_params) + length(temporal_params),
                   config$cores_to_use)
  cl <- makeCluster(n_workers)
  registerDoSNOW(cl)

  env_spatA <- foreach(x = 1:length(spatial_params), .combine = "cbind",
                       .packages = "arrow") %dopar% {
    out <- gen_windows(
      pairs        = spatial_pairs,
      variables    = spatial_params[[x]]$variables,
      mstat        = spatial_params[[x]]$mstat,
      cstat        = spatial_params[[x]]$cstat,
      window       = spatial_params[[x]]$window,
      npy_src      = config$npy_src,
      start_year   = config$geonpy_start_year,
      python_exe   = config$python_exe,
      pyper_script = config$pyper_script,
      feather_tmpdir = config$feather_tmpdir
    )
    colnames(out) <- paste(spatial_params[[x]]$prefix, colnames(out), sep = "_")
    out[, 9:ncol(out)]
  }

  ## Bidirectional averaging (spatial only – temporal pairs share the same site)
  if (config$biAverage) {
    cat("  Computing bidirectional average (spatial)...\n")
    env_spatB <- foreach(x = 1:length(spatial_params), .combine = "cbind",
                         .packages = "arrow") %dopar% {
      out <- gen_windows(
        pairs        = spatial_pairs[, c(5, 6, 7, 8, 1, 2, 3, 4)],  # swap sites
        variables    = spatial_params[[x]]$variables,
        mstat        = spatial_params[[x]]$mstat,
        cstat        = spatial_params[[x]]$cstat,
        window       = spatial_params[[x]]$window,
        npy_src      = config$npy_src,
        start_year   = config$geonpy_start_year,
        python_exe   = config$python_exe,
        pyper_script = config$pyper_script,
        feather_tmpdir = config$feather_tmpdir
      )
      colnames(out) <- paste(spatial_params[[x]]$prefix, colnames(out), sep = "_")
      out[, 9:ncol(out)]
    }
    env_spatial <- (env_spatA + env_spatB) / 2
  } else {
    env_spatial <- env_spatA
  }

  ## --- Parallel extraction: TEMPORAL component ---
  cat("  Extracting temporal environmental data...\n")
  env_temporal <- foreach(x = 1:length(temporal_params), .combine = "cbind",
                          .packages = "arrow") %dopar% {
    out <- gen_windows(
      pairs        = temporal_pairs,
      variables    = temporal_params[[x]]$variables,
      mstat        = temporal_params[[x]]$mstat,
      cstat        = temporal_params[[x]]$cstat,
      window       = temporal_params[[x]]$window,
      npy_src      = config$npy_src,
      start_year   = config$geonpy_start_year,
      python_exe   = config$python_exe,
      pyper_script = config$pyper_script,
      feather_tmpdir = config$feather_tmpdir
    )
    colnames(out) <- paste(temporal_params[[x]]$prefix, colnames(out), sep = "_")
    out[, 9:ncol(out)]
  }

  stopCluster(cl)
  registerDoSEQ()

  ## Split spatial env into site 1 / site 2
  env_spat1 <- env_spatial[, grep("_1$", names(env_spatial))]
  env_spat2 <- env_spatial[, grep("_2$", names(env_spatial))]

  ## Split temporal env into baseline (earliest yr) / observation year
  env_temp1 <- env_temporal[, grep("_1$", names(env_temporal))]
  env_temp2 <- env_temporal[, grep("_2$", names(env_temporal))]

  ## Extract substrate (time-invariant, spatial only)
  pnt1     <- SpatialPoints(data.frame(ext_data[, 1], ext_data[, 2]))
  pnt2     <- SpatialPoints(data.frame(ext_data[, 5], ext_data[, 6]))
  subs_brk <- brick(config$substrate_raster)
  env1_subs <- extract(subs_brk, pnt1)
  env2_subs <- extract(subs_brk, pnt2)
  colnames(env1_subs) <- paste0(colnames(env1_subs), "_1")
  colnames(env2_subs) <- paste0(colnames(env2_subs), "_2")

  ## Combine: obsPairs metadata + spatial env + substrate + temporal env
  parts <- list(obsPairs_out,
                env_spat1, env1_subs,
                env_spat2, env2_subs,
                env_temp1, env_temp2)
  parts <- parts[!sapply(parts, is.null)]
  obsPairs_out <- do.call(cbind, parts)

  ## Clean up
  rm(env_spatial, env_spatA, env_spat1, env_spat2,
     env_temporal, env_temp1, env_temp2, env1_subs, env2_subs)
  if (exists("env_spatB")) rm(env_spatB)
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



cat("\n=== RECA obsGDM complete ===\n")
