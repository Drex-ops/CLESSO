setwd("/Users/andrewhoskins/Library/CloudStorage/OneDrive-NAILSMA/CODE/TERN-biodiversity-index")

# Load config
source("src/clesso_v2/clesso_config.R")

# Load modules
source(file.path(clesso_config$r_dir, "utils.R"))
source(file.path(clesso_config$r_dir, "gdm_functions.R"))
source(file.path(clesso_config$r_dir, "site_aggregator.R"))
source(file.path(clesso_config$r_dir, "gen_windows.R"))
source(file.path(clesso_config$clesso_dir, "clesso_sampler_optimised.R"))
source(file.path(clesso_config$clesso_dir, "clesso_prepare_data.R"))
load_packages()

# Step 1: Load and aggregate observations
obs_file <- file.path(clesso_config$data_dir, clesso_config$obs_csv)
dat <- read.csv(obs_file)
ras_sp  <- raster(clesso_config$reference_raster)
res_deg <- res(ras_sp)[1]
box     <- extent(ras_sp)
datRED  <- siteAggregator(dat, res_deg, box)
test    <- is.na(extract(ras_sp, datRED[, c("lonID", "latID")]))
datRED  <- datRED[!test, ]

# Step 2: Date filter and format
datRED <- datRED[datRED$eventDate != "", ]
datRED <- datRED[as.Date(datRED$eventDate) >= clesso_config$min_date, ]
datRED <- datRED[as.Date(datRED$eventDate) <  clesso_config$max_date, ]
datRED <- droplevels(datRED)
obs_dt <- clesso_format_aggregated_data(datRED)

# Step 3: Sample pairs (THIS uses the new optimised sampler)
diag_pdf <- file.path(clesso_config$output_dir,
  paste0("sampler_diagnostics_", clesso_config$run_id, ".pdf"))
pairs_dt <- clesso_sampler(
  obs_dt              = obs_dt,
  n_within            = clesso_config$n_within,
  n_between           = clesso_config$n_between,
  within_min_recs     = clesso_config$within_min_records,
  within_match_ratio  = clesso_config$within_match_ratio,
  between_match_ratio = clesso_config$between_match_ratio,
  balance_weights     = clesso_config$balance_weights,
  seed                = clesso_config$seed,
  species_thresh      = clesso_config$species_threshold,
  cores               = clesso_config$cores,
  diagnostics_pdf     = diag_pdf
)

# ===========================================================================
# STEP 4: Extract environmental covariates
#
# For the alpha (richness) model: site-level covariates extracted at each
# unique site location (substrate, climate, etc.)
#
# For the turnover (beta) model: pairwise env distances computed via
# I-spline transformations of |env_i - env_j|
#
# NOTE: Environmental extraction depends on available climate data.
# This step uses substrate rasters (always available) and optionally
# geonpy climate data. Adapt this section to your data availability.
# ===========================================================================
cat("\n--- Step 4: Extract environmental covariates ---\n")

## Extract substrate values at all unique site locations
unique_sites <- unique(pairs_dt[, .(
  site_id = site_i, lon = lon_i, lat = lat_i
)])
unique_sites_j <- unique(pairs_dt[, .(
  site_id = site_j, lon = lon_j, lat = lat_j
)])
setnames(unique_sites_j, c("site_id", "lon", "lat"))
unique_sites <- unique(rbind(unique_sites, unique_sites_j))
unique_sites <- unique(unique_sites, by = "site_id")

## Substrate extraction
subs_raster <- tryCatch(
  raster::brick(clesso_config$substrate_raster),
  error = function(e) {
    warning(paste("Could not load substrate raster:", e$message))
    NULL
  }
)

if (!is.null(subs_raster)) {
  subs_vals <- raster::extract(subs_raster,
                                unique_sites[, .(lon, lat)])
  if (is.matrix(subs_vals)) {
    subs_dt <- as.data.table(subs_vals)
    names(subs_dt) <- paste0("subs_", seq_len(ncol(subs_dt)))
  } else {
    subs_dt <- data.table(subs_1 = subs_vals)
  }
  site_covs <- cbind(unique_sites[, .(site_id)], subs_dt)
  cat(sprintf("  Extracted %d substrate variables at %d sites\n",
              ncol(subs_dt), nrow(site_covs)))
} else {
  ## Fallback: no substrate data, use coords only
  site_covs <- NULL
  cat("  No substrate data available. Using lon/lat for alpha model.\n")
}

## Environmental site table for turnover (between-site pairs)
## This provides per-site env values that clesso_build_turnover_X
## will difference for between-site pairs.
if (!is.null(subs_raster)) {
  env_site_table <- cbind(unique_sites[, .(site_id)], subs_dt)
  cat(sprintf("  Env site table: %d sites x %d env variables\n",
              nrow(env_site_table), ncol(subs_dt)))
} else {
  env_site_table <- NULL
  cat("  No env site table. Turnover model uses geographic distance only.\n")
}

## ---- Climate variable extraction via geonpy (fixed 30-year average) ----
## Extract a 30-year climate average centred on 2010 for every unique site.
## pyper.py expects pairwise input (x1,y1,year1,month1,x2,y2,year2,month2),
## so we pair each site with itself and use only the _1 output columns.
if (length(clesso_config$env_params) > 0 &&
    file.exists(clesso_config$pyper_script) &&
    dir.exists(clesso_config$npy_src)) {

  cat("  Extracting climate variables (30-year mean centred on 2010)...\n")
  require(arrow)

  ## Fixed reference: year 2010, month 6 (mid-year), window = config value
  clim_year   <- 2010L
  clim_month  <- 6L
  clim_window <- clesso_config$climate_window  # default 30 years

  ## Build dummy pairs (each site paired with itself)
  clim_pairs <- data.frame(
    Lon1   = unique_sites$lon,
    Lat1   = unique_sites$lat,
    year1  = rep(clim_year, nrow(unique_sites)),
    month1 = rep(clim_month, nrow(unique_sites)),
    Lon2   = unique_sites$lon,
    Lat2   = unique_sites$lat,
    year2  = rep(clim_year, nrow(unique_sites)),
    month2 = rep(clim_month, nrow(unique_sites))
  )

  ## Iterate through env_params groups (same structure as reca_STresiduals)
  clim_parts <- list()
  for (j in seq_along(clesso_config$env_params)) {
    ep <- clesso_config$env_params[[j]]
    cat(sprintf("    [%d/%d] %s | mstat=%s cstat=%s\n",
                j, length(clesso_config$env_params),
                paste(ep$variables, collapse = ", "),
                ep$mstat, ep$cstat))

    raw <- gen_windows(
      pairs          = clim_pairs,
      variables      = ep$variables,
      mstat          = ep$mstat,
      cstat          = ep$cstat,
      window         = clim_window,
      npy_src        = clesso_config$npy_src,
      start_year     = clesso_config$geonpy_start_year,
      python_exe     = clesso_config$python_exe,
      pyper_script   = clesso_config$pyper_script,
      feather_tmpdir = clesso_config$feather_tmpdir
    )

    ## Keep only _1 columns (site-level values; skip first 8 coord columns)
    env_cols <- raw[, 9:ncol(raw), drop = FALSE]
    idx_1    <- grep("_1$", names(env_cols))
    site_env <- env_cols[, idx_1, drop = FALSE]

    ## Strip date-range patterns (e.g. "191101-201712_") and _1 suffix
    names(site_env) <- gsub("\\d{6}-\\d{6}_", "", names(site_env))
    names(site_env) <- gsub("_1$", "", names(site_env))

    ## Prefix with cstat to keep names unique across groups
    names(site_env) <- paste0(ep$cstat, "_", names(site_env))

    clim_parts[[j]] <- site_env
  }

  clim_dt <- as.data.table(do.call(cbind, clim_parts))

  ## Check for sentinels / NA
  n_na   <- sum(is.na(clim_dt))
  n_sent <- sum(clim_dt == -9999, na.rm = TRUE)
  if (n_na > 0)   warning(sprintf("  %d NA values in climate extraction.", n_na))
  if (n_sent > 0)  warning(sprintf("  %d sentinel (-9999) values in climate extraction.", n_sent))

  cat(sprintf("  Extracted %d climate variables at %d sites\n",
              ncol(clim_dt), nrow(clim_dt)))

  ## Append to site_covs (alpha model)
  if (!is.null(site_covs)) {
    site_covs <- cbind(site_covs, clim_dt)
  } else {
    site_covs <- cbind(unique_sites[, .(site_id)], clim_dt)
  }

  ## Append to env_site_table (turnover / beta model)
  if (!is.null(env_site_table)) {
    env_site_table <- cbind(env_site_table, clim_dt)
  } else {
    env_site_table <- cbind(unique_sites[, .(site_id)], clim_dt)
  }

  cat(sprintf("  Final site_covs: %d columns | env_site_table: %d columns\n",
              ncol(site_covs) - 1, ncol(env_site_table) - 1))

} else {
  cat("  Skipping climate extraction (env_params empty or geonpy not available).\n")
}


# Save the new pairs
pairs_file <- file.path(clesso_config$output_dir,
  paste0("clesso_pairs_", clesso_config$run_id, ".rds"))
saveRDS(pairs_dt, file = pairs_file)
cat(sprintf("Saved %d pairs to %s\n", nrow(pairs_dt), pairs_file))

# Now run the export
source("src/clesso_nn/export_for_nn.R")