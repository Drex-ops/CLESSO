##############################################################################
##
## run_obsGDM.R  --  Single parameterised entry point for RECA obsGDM
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
##   Load data -> siteAggregator -> date filter -> site.richness.extractor ->
##   obsPairSampler -> gen_windows (env extraction) -> anomalies ->
##   substrate extraction -> splineData -> fitGDM -> diagnostics -> save
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

## Locate config.R: if this_dir is already the reca_STresiduals folder, use it
## directly; otherwise assume this_dir is the project root.
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

## Build full-record proxy (maps each row of frog.auGrid to its site)
## This replaces the old m2 = m1[row.vect, ] expansion that caused OOM.
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

## ---------------------------------------------------------------------------
## MODIS validation (early exit if misconfigured)
## ---------------------------------------------------------------------------
if (config$add_modis) {
  cat("  MODIS land cover extraction enabled.\n")
  cat(sprintf("  MODIS directory: %s\n", config$modis_dir))
  if (!dir.exists(config$modis_dir)) {
    stop(paste("MODIS directory not found:", config$modis_dir))
  }
  ## Check that the date range of the model is within MODIS coverage
  model_min_year <- as.integer(format(
    config$min_date %m+% years(config$date_offset_years), "%Y"))
  model_max_year <- as.integer(format(config$max_date, "%Y"))
  if (model_min_year < config$modis_start_year) {
    stop(sprintf(
      paste0("ADD_MODIS requires data from %d onwards, but config$min_date + offset gives %d. ",
             "Set min_date/date_offset_years so the earliest model year >= %d, or disable ADD_MODIS."),
      config$modis_start_year, model_min_year, config$modis_start_year
    ))
  }
  if (model_max_year > config$modis_end_year) {
    warning(sprintf(
      paste0("config$max_date (%s) exceeds MODIS end year (%d). ",
             "Pairs with years > %d will be clamped to %d for MODIS extraction."),
      config$max_date, config$modis_end_year,
      config$modis_end_year, config$modis_end_year
    ))
  }
  cat(sprintf("  Model year range: %d\u2013%d  |  MODIS coverage: %d\u2013%d\n",
              model_min_year, model_max_year,
              config$modis_start_year, config$modis_end_year))
}

## ---------------------------------------------------------------------------
## Condition raster validation (early exit if misconfigured)
## ---------------------------------------------------------------------------
if (config$add_condition) {
  cat("  Condition raster extraction enabled.\n")
  cat(sprintf("  Condition TIF: %s\n", config$condition_tif_path))
  if (!file.exists(config$condition_tif_path)) {
    stop(paste("Condition TIF not found:", config$condition_tif_path))
  }
  model_min_year <- as.integer(format(
    config$min_date %m+% years(config$date_offset_years), "%Y"))
  model_max_year <- as.integer(format(config$max_date, "%Y"))
  if (model_max_year > config$condition_end_year) {
    warning(sprintf(
      paste0("config$max_date (%s) exceeds condition end year (%d). ",
             "Pairs with years > %d will be clamped to %d for condition extraction."),
      config$max_date, config$condition_end_year,
      config$condition_end_year, config$condition_end_year
    ))
  }
  cat(sprintf("  Model year range: %d\u2013%d  |  Condition coverage: %d\u2013%d\n",
              model_min_year, model_max_year,
              config$condition_start_year, config$condition_end_year))
}

## Apply temporal filter: ensure earliest_year - climate_window >= geonpy_start_year
obsPairs_out      <- obsPairs_orig
earliest_year_all <- pmin(obsPairs_out$year1, obsPairs_out$year2)
tst <- (earliest_year_all - c_yr) >= config$geonpy_start_year
obsPairs_out <- obsPairs_out[tst, ]

## Additional MODIS temporal filter: both years must be >= modis_start_year
if (config$add_modis) {
  modis_tst <- obsPairs_out$year1 >= config$modis_start_year &
               obsPairs_out$year2 >= config$modis_start_year
  n_removed <- sum(!modis_tst)
  if (n_removed > 0) {
    cat(sprintf("  MODIS filter: removing %d pairs with years before %d\n",
                n_removed, config$modis_start_year))
  }
  obsPairs_out <- obsPairs_out[modis_tst, ]
}

## Additional condition temporal filter: both years must be >= condition_start_year
if (config$add_condition) {
  cond_tst <- obsPairs_out$year1 >= config$condition_start_year &
              obsPairs_out$year2 >= config$condition_start_year
  n_removed <- sum(!cond_tst)
  if (n_removed > 0) {
    cat(sprintf("  Condition filter: removing %d pairs with years before %d\n",
                n_removed, config$condition_start_year))
  }
  obsPairs_out <- obsPairs_out[cond_tst, ]
}

ext_data     <- obsPairs_out[, 2:9]
cat(sprintf("  Pairs after temporal filter: %d\n", nrow(obsPairs_out)))

save_prefix <- paste0(config$species_group, "_",
                      format(config$nMatch / 1e6, nsmall = 0), "mil_",
                      c_yr, "climWin_STresid_")
if (config$biAverage)              save_prefix <- paste0(save_prefix, "biAverage_")
if (config$decomposition != "none") save_prefix <- paste0(config$decomposition, "_", save_prefix)
if (config$add_modis)              save_prefix <- paste0(save_prefix, config$modis_suffix)
if (config$add_condition)          save_prefix <- paste0(save_prefix, config$condition_suffix)

env_file <- file.path(config$run_output_dir, paste0(save_prefix, "ObsEnvTable.RData"))

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

  env_spatA <- run_chunked_env(spatial_pairs, spatial_params, "Spatial-A")

  ## Bidirectional averaging (spatial only – temporal pairs share the same site)
  if (config$biAverage) {
    cat("  Computing bidirectional average (spatial)...\n")
    env_spatB <- run_chunked_env(spatial_pairs, spatial_params, "Spatial-B",
                                  swap_sites = TRUE)
    env_spatial <- (env_spatA + env_spatB) / 2
  } else {
    env_spatial <- env_spatA
  }

  ## --- Parallel extraction: TEMPORAL component ---
  cat("  Extracting temporal environmental data...\n")
  env_temporal <- run_chunked_env(temporal_pairs, temporal_params, "Temporal")

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

  ## ------------------------------------------------------------------
  ## STEP 6a-MODIS: Extract MODIS as spatial + temporal predictors
  ##
  ## MODIS land cover is treated the same way as climate variables:
  ##   Spatial component -- both sites at the earliest year
  ##     spat_modis_{var}_1 = MODIS(site1, earliest_year)
  ##     spat_modis_{var}_2 = MODIS(site2, earliest_year)
  ##   Temporal component -- site 2 at baseline vs observation year
  ##     temp_modis_{var}_1 = MODIS(site2, earliest_year)
  ##     temp_modis_{var}_2 = MODIS(site2, year2)
  ##
  ## File pattern: modis_{year}_{variable}_{resolution}_COG.tif
  ## Years outside MODIS range are clamped to boundary year.
  ## ------------------------------------------------------------------
  modis_spat_1 <- NULL; modis_spat_2 <- NULL
  modis_temp_1 <- NULL; modis_temp_2 <- NULL

  if (config$add_modis) {
    cat("  Extracting MODIS land cover (spatial + temporal)...\n")

    ## Helper: load MODIS raster for a given variable and year
    load_modis_raster <- function(varname, year) {
      yr_clamped <- min(max(year, config$modis_start_year), config$modis_end_year)
      fname <- file.path(config$modis_dir,
                         paste0("modis_", yr_clamped, "_", varname, "_",
                                config$modis_resolution, "_COG.tif"))
      if (!file.exists(fname)) stop(sprintf("MODIS raster not found: %s", fname))
      raster(fname)
    }

    ## Clamp years to MODIS range
    earliest_year_clamped <- pmin(pmax(earliest_year,
                                       config$modis_start_year), config$modis_end_year)
    year2_clamped <- pmin(pmax(ext_data[, 7],
                               config$modis_start_year), config$modis_end_year)
    all_modis_years <- sort(unique(c(earliest_year_clamped, year2_clamped)))

    ## Pre-load MODIS rasters (var × year)
    cat(sprintf("  Pre-loading MODIS rasters for %d years × %d variables...\n",
                length(all_modis_years), length(config$modis_variables)))
    modis_cache <- list()
    for (mv in config$modis_variables) {
      for (yr in all_modis_years) {
        key <- paste0(mv, "_", yr)
        modis_cache[[key]] <- load_modis_raster(mv, yr)
      }
    }

    n_pairs <- nrow(ext_data)
    n_vars  <- length(config$modis_variables)
    modis_pnt1 <- SpatialPoints(data.frame(ext_data[, 1], ext_data[, 2]))
    modis_pnt2 <- SpatialPoints(data.frame(ext_data[, 5], ext_data[, 6]))

    ## --- SPATIAL MODIS ---
    ## Site 1 at earliest_year -> spat_modis_{var}_1
    ## Site 2 at earliest_year -> spat_modis_{var}_2
    ms1_fwd <- matrix(NA_real_, nrow = n_pairs, ncol = n_vars)
    ms2_fwd <- matrix(NA_real_, nrow = n_pairs, ncol = n_vars)
    colnames(ms1_fwd) <- paste0("spat_modis_", config$modis_variables, "_1")
    colnames(ms2_fwd) <- paste0("spat_modis_", config$modis_variables, "_2")

    for (vi in seq_along(config$modis_variables)) {
      mv <- config$modis_variables[vi]
      cat(sprintf("    Spatial %s...\n", mv))
      for (yr in sort(unique(earliest_year_clamped))) {
        idx <- which(earliest_year_clamped == yr)
        ras <- modis_cache[[paste0(mv, "_", yr)]]
        ms1_fwd[idx, vi] <- extract(ras, modis_pnt1[idx])
        ms2_fwd[idx, vi] <- extract(ras, modis_pnt2[idx])
      }
    }

    if (config$biAverage) {
      cat("  Computing bidirectional average (MODIS spatial)...\n")
      ms1_rev <- matrix(NA_real_, nrow = n_pairs, ncol = n_vars)
      ms2_rev <- matrix(NA_real_, nrow = n_pairs, ncol = n_vars)
      for (vi in seq_along(config$modis_variables)) {
        mv <- config$modis_variables[vi]
        for (yr in sort(unique(earliest_year_clamped))) {
          idx <- which(earliest_year_clamped == yr)
          ras <- modis_cache[[paste0(mv, "_", yr)]]
          ms1_rev[idx, vi] <- extract(ras, modis_pnt2[idx])  # site2 -> _1
          ms2_rev[idx, vi] <- extract(ras, modis_pnt1[idx])  # site1 -> _2
        }
      }
      ms1_fwd <- (ms1_fwd + ms1_rev) / 2
      ms2_fwd <- (ms2_fwd + ms2_rev) / 2
      colnames(ms1_fwd) <- paste0("spat_modis_", config$modis_variables, "_1")
      colnames(ms2_fwd) <- paste0("spat_modis_", config$modis_variables, "_2")
    }

    modis_spat_1 <- as.data.frame(ms1_fwd)
    modis_spat_2 <- as.data.frame(ms2_fwd)

    ## --- TEMPORAL MODIS ---
    ## Site 2 at earliest_year -> temp_modis_{var}_1 (baseline)
    ## Site 2 at obs year2     -> temp_modis_{var}_2 (target)
    mt1 <- matrix(NA_real_, nrow = n_pairs, ncol = n_vars)
    mt2 <- matrix(NA_real_, nrow = n_pairs, ncol = n_vars)
    colnames(mt1) <- paste0("temp_modis_", config$modis_variables, "_1")
    colnames(mt2) <- paste0("temp_modis_", config$modis_variables, "_2")

    for (vi in seq_along(config$modis_variables)) {
      mv <- config$modis_variables[vi]
      cat(sprintf("    Temporal %s...\n", mv))

      ## Baseline (site2 at earliest_year)
      for (yr in sort(unique(earliest_year_clamped))) {
        idx <- which(earliest_year_clamped == yr)
        ras <- modis_cache[[paste0(mv, "_", yr)]]
        mt1[idx, vi] <- extract(ras, modis_pnt2[idx])
      }

      ## Target (site2 at obs year2)
      for (yr in sort(unique(year2_clamped))) {
        idx <- which(year2_clamped == yr)
        ras <- modis_cache[[paste0(mv, "_", yr)]]
        mt2[idx, vi] <- extract(ras, modis_pnt2[idx])
      }
    }

    modis_temp_1 <- as.data.frame(mt1)
    modis_temp_2 <- as.data.frame(mt2)

    cat(sprintf("  MODIS extraction complete: %d spatial + %d temporal covariates per site\n",
                n_vars, n_vars))

    rm(modis_cache, ms1_fwd, ms2_fwd, mt1, mt2, modis_pnt1, modis_pnt2)
    if (exists("ms1_rev")) rm(ms1_rev, ms2_rev)
    gc()
  }

  ## ------------------------------------------------------------------
  ## STEP 6a-CONDITION: Extract condition raster (spatial + temporal)
  ##
  ## Condition is a single multi-band GeoTIFF where band i = year
  ## (config$condition_start_year + i - 1).  It is treated the same
  ## way as MODIS:
  ##   Spatial component -- both sites at the earliest year
  ##     spat_condition_1 = Condition(site1, earliest_year)
  ##     spat_condition_2 = Condition(site2, earliest_year)
  ##   Temporal component -- site 2 at baseline vs observation year
  ##     temp_condition_1 = Condition(site2, earliest_year)
  ##     temp_condition_2 = Condition(site2, year2)
  ##
  ## Years outside the condition range are clamped to the boundary year.
  ## ------------------------------------------------------------------
  cond_spat_1 <- NULL; cond_spat_2 <- NULL
  cond_temp_1 <- NULL; cond_temp_2 <- NULL

  if (config$add_condition) {
    cat("  Extracting condition raster (spatial + temporal)...\n")

    ## Load the multi-band TIF as a brick (one layer per year)
    cond_brick <- brick(config$condition_tif_path)
    cond_start <- config$condition_start_year
    cond_end   <- config$condition_end_year

    ## Helper: get band index for a given year (vectorised, clamped)
    cond_band_idx <- function(year) {
      yr_clamped <- pmin(pmax(year, cond_start), cond_end)
      yr_clamped - cond_start + 1L
    }

    ## Clamp years
    earliest_year_cond <- pmin(pmax(earliest_year, cond_start), cond_end)
    year2_cond         <- pmin(pmax(ext_data[, 7], cond_start), cond_end)

    n_pairs <- nrow(ext_data)
    cond_pnt1 <- SpatialPoints(data.frame(ext_data[, 1], ext_data[, 2]))
    cond_pnt2 <- SpatialPoints(data.frame(ext_data[, 5], ext_data[, 6]))

    cond_var <- config$condition_variable  # "condition"

    ## --- SPATIAL CONDITION ---
    cs1 <- rep(NA_real_, n_pairs)
    cs2 <- rep(NA_real_, n_pairs)

    cat("    Spatial condition...\n")
    for (yr in sort(unique(earliest_year_cond))) {
      idx <- which(earliest_year_cond == yr)
      band <- cond_band_idx(yr)
      ras  <- cond_brick[[band]]
      cs1[idx] <- extract(ras, cond_pnt1[idx])
      cs2[idx] <- extract(ras, cond_pnt2[idx])
    }

    if (config$biAverage) {
      cat("  Computing bidirectional average (condition spatial)...\n")
      cs1_rev <- rep(NA_real_, n_pairs)
      cs2_rev <- rep(NA_real_, n_pairs)
      for (yr in sort(unique(earliest_year_cond))) {
        idx <- which(earliest_year_cond == yr)
        band <- cond_band_idx(yr)
        ras  <- cond_brick[[band]]
        cs1_rev[idx] <- extract(ras, cond_pnt2[idx])  # site2 -> _1
        cs2_rev[idx] <- extract(ras, cond_pnt1[idx])  # site1 -> _2
      }
      cs1 <- (cs1 + cs1_rev) / 2
      cs2 <- (cs2 + cs2_rev) / 2
      rm(cs1_rev, cs2_rev)
    }

    cond_spat_1 <- data.frame(spat_condition_1 = cs1)
    cond_spat_2 <- data.frame(spat_condition_2 = cs2)

    ## --- TEMPORAL CONDITION ---
    ct1 <- rep(NA_real_, n_pairs)
    ct2 <- rep(NA_real_, n_pairs)

    cat("    Temporal condition (baseline)...\n")
    for (yr in sort(unique(earliest_year_cond))) {
      idx <- which(earliest_year_cond == yr)
      band <- cond_band_idx(yr)
      ras  <- cond_brick[[band]]
      ct1[idx] <- extract(ras, cond_pnt2[idx])
    }

    cat("    Temporal condition (target)...\n")
    for (yr in sort(unique(year2_cond))) {
      idx <- which(year2_cond == yr)
      band <- cond_band_idx(yr)
      ras  <- cond_brick[[band]]
      ct2[idx] <- extract(ras, cond_pnt2[idx])
    }

    cond_temp_1 <- data.frame(temp_condition_1 = ct1)
    cond_temp_2 <- data.frame(temp_condition_2 = ct2)

    cat(sprintf("  Condition extraction complete: 1 spatial + 1 temporal covariate per site\n"))

    rm(cond_brick, cs1, cs2, ct1, ct2, cond_pnt1, cond_pnt2)
    gc()
  }

  ## Combine: obsPairs metadata + ALL site-1/baseline cols + ALL site-2/obs cols
  ## IMPORTANT: splineData_fast splits at the midpoint, so all _1 columns must
  ## come first and all _2 columns second, in matching order.
  parts <- list(obsPairs_out,
                env_spat1, modis_spat_1, cond_spat_1, env1_subs, env_temp1, modis_temp_1, cond_temp_1,
                env_spat2, modis_spat_2, cond_spat_2, env2_subs, env_temp2, modis_temp_2, cond_temp_2)
  parts <- parts[!sapply(parts, is.null)]
  obsPairs_out <- do.call(cbind, parts)

  ## Clean up climate/substrate/MODIS/condition env objects
  rm(env_spatial, env_spatA, env_spat1, env_spat2,
     env_temporal, env_temp1, env_temp2, env1_subs, env2_subs)
  if (exists("env_spatB")) rm(env_spatB)
  if (exists("modis_spat_1")) rm(modis_spat_1, modis_spat_2,
                                  modis_temp_1, modis_temp_2)
  if (exists("cond_spat_1"))  rm(cond_spat_1, cond_spat_2,
                                  cond_temp_1, cond_temp_2)
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
cat("  Computing I-spline basis (fast)...\n")
toSpline <- obsPairs_out[, env_cols]
splined  <- splineData_fast(toSpline)

if (config$decomposition == "v3") {
  splined_new <- splined[keep, ]
} else {
  splined_new <- splined
}

# ------------------------------------------------------------------
# STEP 6d2: Pre-compute metadata & free memory before fitting
# ------------------------------------------------------------------
## Quantiles (needed for fit object; compute now so we can free toSpline)
nc  <- ncol(toSpline)
nc2 <- nc / 2
X1  <- toSpline[, 1:nc2]
X2  <- toSpline[, (nc2 + 1):nc]
nms <- names(X1); names(X2) <- nms
XX  <- rbind(X1, X2)
sp_quant      <- unlist(lapply(1:ncol(XX), function(x) quantile(XX[, x], c(0, 0.5, 1))))
n_spline_vars <- ncol(XX)
rm(X1, X2, XX, nms)

## Predictor names (from spline column names)
predictors_vec <- gsub("191101-201712_", "",
                        gsub("_spl1", "", colnames(splined)[grep("_spl1", colnames(splined))]))

## Total pair count (before v3 filter)
n_pairs_total <- nrow(obsPairs_out)

## Build mod_ready before freeing source objects
match_response <- if (config$decomposition == "v3") {
  obsPairs_out$Match[keep]
} else {
  obsPairs_out$Match
}
mod_ready <- cbind(Match = match_response, as.data.frame(splined_new))
colnames(mod_ready) <- gsub("191101-201712_", "", colnames(mod_ready))

## FREE large intermediate objects to reclaim memory for GLM fitting
cat("  Freeing intermediate objects to reduce memory...\n")
rm(obsPairs_out, toSpline, splined, splined_new, match_response)
if (exists("keep") && is.logical(keep)) rm(keep)
invisible(gc())

# ------------------------------------------------------------------
# STEP 6e: Fit GDM (with optional subsampling for very large datasets)
# ------------------------------------------------------------------
if (!is.null(config$max_fit_pairs) && nrow(mod_ready) > config$max_fit_pairs) {
  cat(sprintf("  Subsampling %d -> %d pairs for fitting\n",
              nrow(mod_ready), config$max_fit_pairs))
  idx <- sample.int(nrow(mod_ready), config$max_fit_pairs)
  mod_ready <- mod_ready[idx, ]
  rm(idx)
}

cat(sprintf("  Fitting GDM on %d pairs...\n", nrow(mod_ready)))
f1      <- paste(colnames(mod_ready)[-1], collapse = "+")
formula <- as.formula(paste(colnames(mod_ready)[1], "~", f1, sep = ""))
obsGDM_1 <- fitGDM(formula = formula, data = mod_ready)

# ------------------------------------------------------------------
# STEP 6f: Diagnostics and save
# ------------------------------------------------------------------
out_prefix <- file.path(config$run_output_dir, save_prefix)

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

## Build fit object for plotting and metadata
    fit <- list()
    fit$intercept    <- coef(obsGDM_1)[1]
    fit$sample       <- nrow(mod_ready)
    fit$predictors   <- predictors_vec
    fit$coefficients <- coef(obsGDM_1)[-1]
    fit$coefficients[is.na(fit$coefficients)] <- 0
    fit$quantiles    <- sp_quant
    fit$splines      <- rep(3, n_spline_vars)
    fit$predicted    <- fitted(obsGDM_1)
    fit$ecological   <- obsGDM_1$linear.predictors

    ## ---- Run metadata ----
    fit$species_group   <- config$species_group
    fit$climate_window  <- c_yr
    fit$nMatch          <- config$nMatch
    fit$w_ratio         <- w
    fit$biAverage       <- config$biAverage
    fit$decomposition   <- config$decomposition
    fit$date_range      <- c(as.character(config$min_date), as.character(config$max_date))
    fit$date_offset_yrs <- config$date_offset_years
    fit$obs_csv         <- config$obs_csv
    fit$n_pairs         <- n_pairs_total
    fit$D2              <- D2
    fit$nagelkerke_r2   <- gdm_dev$Nagelkerke
    fit$env_params      <- config$env_params
    fit$substrate_raster <- basename(config$substrate_raster)
    fit$reference_raster <- basename(config$reference_raster)
    fit$grid_resolution  <- config$grid_resolution
    fit$geonpy_start_year <- config$geonpy_start_year
    fit$add_modis        <- config$add_modis
    if (config$add_modis) {
      fit$modis_variables  <- config$modis_variables
      fit$modis_year_range <- c(config$modis_start_year, config$modis_end_year)
    }
    fit$add_condition    <- config$add_condition
    if (config$add_condition) {
      fit$condition_variable   <- config$condition_variable
      fit$condition_year_range <- c(config$condition_start_year, config$condition_end_year)
      fit$condition_tif_path   <- config$condition_tif_path
    }
    fit$run_timestamp   <- Sys.time()
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
