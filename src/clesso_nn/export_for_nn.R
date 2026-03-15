##############################################################################
##
## export_for_nn.R -- Export CLESSO R pipeline data for the NN model
##
## Run this AFTER Steps 1-4 of run_clesso.R have completed (i.e. you have
## pairs_dt, site_covs, env_site_table, and site_obs_richness in memory).
##
## This script exports everything the Python NN pipeline needs as feather
## files in a designated export directory.
##
## Usage (from within a running R session after Steps 1-4):
##   source("src/clesso_nn/export_for_nn.R")
##
## Or standalone (re-runs Steps 1-4 then exports):
##   Rscript src/clesso_nn/export_for_nn.R
##
##############################################################################

cat("=== Exporting CLESSO data for neural network pipeline ===\n")

# ---------------------------------------------------------------------------
# Check if we're running standalone or inside run_clesso.R
# ---------------------------------------------------------------------------
if (!exists("pairs_dt") || !exists("clesso_config")) {
  cat("  pairs_dt not found -- running R pipeline Steps 1-4 first...\n")

  ## Source run_clesso.R up to Step 4
  ## We'll source config and modules, run Steps 1-4, then export.
  this_dir <- "/Users/andrewhoskins/Library/CloudStorage/OneDrive-NAILSMA/CODE/TERN-biodiversity-index/src/"

  clesso_dir <- "/Users/andrewhoskins/Library/CloudStorage/OneDrive-NAILSMA/CODE/TERN-biodiversity-index/src/clesso_v2"

  config_path <- file.path(clesso_dir, "clesso_config.R")
  if (!file.exists(config_path)) {
    stop("Cannot find clesso_config.R at: ", config_path,
         "\nSet working directory to project root or source from run_clesso.R")
  }
  source(config_path)

  source(file.path(clesso_config$r_dir, "utils.R"))
  source(file.path(clesso_config$r_dir, "gdm_functions.R"))
  source(file.path(clesso_config$r_dir, "site_aggregator.R"))
  source(file.path(clesso_config$r_dir, "gen_windows.R"))
  source(file.path(clesso_config$clesso_dir, "clesso_sampler_optimised.R"))
  source(file.path(clesso_config$clesso_dir, "clesso_prepare_data.R"))

  load_packages()

  ## -----------------------------------------------------------------
  ## Try to find saved pairs from a previous run
  ## -----------------------------------------------------------------
  output_base <- dirname(clesso_config$output_dir)  # parent of run-specific dir
  saved_pairs <- list.files(output_base, pattern = "^clesso_pairs_.*\\.rds$",
                            recursive = TRUE, full.names = TRUE)

  ## Filter to the current species group
  sp_pattern <- paste0("clesso_pairs_", clesso_config$species_group)
  saved_pairs <- saved_pairs[grepl(sp_pattern, basename(saved_pairs))]

  if (length(saved_pairs) > 0) {
    ## Load the most recent one (by file modification time)
    saved_pairs <- saved_pairs[order(file.mtime(saved_pairs), decreasing = TRUE)]
    cat(sprintf("  Found saved pairs from previous run: %s\n", saved_pairs[1]))
    pairs_dt <- readRDS(saved_pairs[1])
    if (!inherits(pairs_dt, "data.table")) pairs_dt <- as.data.table(pairs_dt)
    cat(sprintf("  Loaded %d pairs (%d columns)\n", nrow(pairs_dt), ncol(pairs_dt)))

  } else {
    ## -- No saved data: run Steps 1-3 from scratch --
    cat("  No saved pairs found. Running Steps 1-3 from scratch...\n\n")

    # ---- STEP 1: Load and aggregate observations ----
    cat("--- Step 1: Load and aggregate observations ---\n")
    obs_file <- file.path(clesso_config$data_dir, clesso_config$obs_csv)
    if (!file.exists(obs_file)) stop(paste("Observation file not found:", obs_file))
    dat <- read.csv(obs_file)
    cat(sprintf("  Loaded %d records from %s\n", nrow(dat), clesso_config$obs_csv))

    ras_sp  <- raster(clesso_config$reference_raster)
    res_deg <- res(ras_sp)[1]
    box     <- extent(ras_sp)
    datRED  <- siteAggregator(dat, res_deg, box)
    test    <- is.na(extract(ras_sp, datRED[, c("lonID", "latID")]))
    datRED  <- datRED[!test, ]
    cat(sprintf("  Aggregated data: %d site-species-date records\n", nrow(datRED)))

    # ---- STEP 2: Date filter and format ----
    cat("\n--- Step 2: Date filter and format ---\n")
    datRED <- datRED[datRED$eventDate != "", ]
    datRED <- datRED[as.Date(datRED$eventDate) >= clesso_config$min_date, ]
    datRED <- datRED[as.Date(datRED$eventDate) <  clesso_config$max_date, ]
    datRED <- droplevels(datRED)
    cat(sprintf("  After date filter: %d records\n", nrow(datRED)))
    obs_dt <- clesso_format_aggregated_data(datRED)

    # ---- STEP 3: Sample observation pairs ----
    cat("\n--- Step 3: CLESSO observation-pair sampling ---\n")
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
      cores               = clesso_config$cores
    )

    ## Save pairs for future reuse
    pairs_file <- file.path(clesso_config$output_dir,
      paste0("clesso_pairs_", clesso_config$run_id, ".rds"))
    saveRDS(pairs_dt, file = pairs_file)
    cat(sprintf("  Pairs saved to %s\n", pairs_file))
  }

  ## -----------------------------------------------------------------
  ## Step 4: Compute observed richness per site
  ## -----------------------------------------------------------------
  if (!exists("site_obs_richness") || is.null(site_obs_richness)) {
    if (exists("obs_dt") && !is.null(obs_dt)) {
      site_obs_richness <- obs_dt[, .(S_obs = uniqueN(species)), by = .(site_id)]
    } else {
      cat("  (obs_dt not available; site_obs_richness will be derived from pairs)\n")
      site_obs_richness <- NULL
    }
  }

  ## -----------------------------------------------------------------
  ## Step 4b: Reconstruct unique_sites and site_covs from pairs_dt
  ## -----------------------------------------------------------------
  require(data.table)
  unique_sites <- unique(pairs_dt[, .(site_id = site_i, lon = lon_i, lat = lat_i)])
  unique_sites_j <- unique(pairs_dt[, .(site_id = site_j, lon = lon_j, lat = lat_j)])
  setnames(unique_sites_j, c("site_id", "lon", "lat"))
  unique_sites <- unique(rbind(unique_sites, unique_sites_j), by = "site_id")

  ## Extract substrate values if available
  subs_raster <- tryCatch(
    raster::brick(clesso_config$substrate_raster),
    error = function(e) { cat("  Substrate raster not available. Using coords only.\n"); NULL }
  )
  if (!is.null(subs_raster)) {
    subs_vals <- raster::extract(subs_raster, unique_sites[, .(lon, lat)])
    if (is.matrix(subs_vals)) {
      subs_dt <- as.data.table(subs_vals)
      names(subs_dt) <- paste0("subs_", seq_len(ncol(subs_dt)))
    } else {
      subs_dt <- data.table(subs_1 = subs_vals)
    }
    site_covs      <- cbind(unique_sites[, .(site_id)], subs_dt)
    env_site_table <- cbind(unique_sites[, .(site_id)], subs_dt)
  } else {
    site_covs      <- NULL
    env_site_table <- NULL
  }

  ## -----------------------------------------------------------------
  ## Step 4c: Extract climate variables via geonpy (30-year average)
  ## -----------------------------------------------------------------
  ## This mirrors the climate extraction block in run_clesso.R (Step 4).
  ## Requires env_params, pyper_script, and npy_src from clesso_config.
  if (length(clesso_config$env_params) > 0 &&
      file.exists(clesso_config$pyper_script) &&
      dir.exists(clesso_config$npy_src)) {

    cat("  Extracting climate variables (30-year mean centred on 2010)...\n")
    require(arrow)

    clim_year   <- 2010L
    clim_month  <- 6L
    clim_window <- clesso_config$climate_window

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

      ## Keep only _1 columns (site-level; skip first 8 coord columns)
      env_cols <- raw[, 9:ncol(raw), drop = FALSE]
      idx_1    <- grep("_1$", names(env_cols))
      site_env <- env_cols[, idx_1, drop = FALSE]

      ## Strip date-range patterns and _1 suffix
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
    if (n_sent > 0) warning(sprintf("  %d sentinel (-9999) values in climate extraction.", n_sent))

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
    if (length(clesso_config$env_params) == 0)
      cat("    Reason: clesso_config$env_params is empty\n")
    if (!file.exists(clesso_config$pyper_script))
      cat(sprintf("    Reason: pyper_script not found: %s\n", clesso_config$pyper_script))
    if (!dir.exists(clesso_config$npy_src))
      cat(sprintf("    Reason: npy_src dir not found: %s\n", clesso_config$npy_src))
  }

  cat(sprintf("  Standalone setup complete: %d pairs, %d unique sites\n",
              nrow(pairs_dt), nrow(unique_sites)))
}

# ---------------------------------------------------------------------------
# Step 4d: Extract effort / detectability rasters at site locations
# ---------------------------------------------------------------------------
## The 6 raw effort rasters (ESRI BIL .flt format) live in the Effort_data_preper
## outputs directory.  We extract values at each unique site and include them in
## site_covariates so the NN effort‑net can use them.

effort_raster_dir <- Sys.getenv(
  "EFFORT_RASTER_DIR",
  unset = "/Users/andrewhoskins/Library/Mobile Documents/com~apple~CloudDocs/CODE/Effort_data_preper/outputs"
)

effort_layer_names <- c(
  "ala_record_count",
  "ala_record_smoothed",
  "dist_to_nearest_institution",
  "hub_influence_unweighted",
  "hub_influence_ecology_weighted",
  "road_density_km_per_km2"
)

if (dir.exists(effort_raster_dir)) {
  cat("  Extracting effort / detectability rasters at site locations...\n")
  require(raster)

  ## Rebuild unique_sites if not already in scope (e.g. sourced from run_clesso)
  if (!exists("unique_sites") || is.null(unique_sites)) {
    unique_sites <- unique(pairs_dt[, .(site_id = site_i, lon = lon_i, lat = lat_i)])
    unique_sites_j <- unique(pairs_dt[, .(site_id = site_j, lon = lon_j, lat = lat_j)])
    setnames(unique_sites_j, c("site_id", "lon", "lat"))
    unique_sites <- unique(rbind(unique_sites, unique_sites_j), by = "site_id")
  }

  effort_parts <- list()
  for (lyr in effort_layer_names) {
    flt_path <- file.path(effort_raster_dir, paste0(lyr, ".flt"))
    if (!file.exists(flt_path)) {
      cat(sprintf("    WARNING: effort raster not found: %s\n", flt_path))
      next
    }
    ras <- raster(flt_path)
    vals <- raster::extract(ras, unique_sites[, .(lon, lat)])
    effort_parts[[lyr]] <- vals
    n_na <- sum(is.na(vals))
    cat(sprintf("    %s  — %d values extracted (%d NA)\n", lyr, length(vals), n_na))
  }

  if (length(effort_parts) > 0) {
    effort_dt <- as.data.table(effort_parts)

    ## Append effort columns to site_covs (so they end up in site_covariates.feather)
    if (!is.null(site_covs)) {
      site_covs <- cbind(site_covs, effort_dt)
    } else {
      site_covs <- cbind(unique_sites[, .(site_id)], effort_dt)
    }
    cat(sprintf("  Effort extraction complete: %d layers appended to site_covs\n",
                ncol(effort_dt)))
  }
} else {
  cat(sprintf("  Skipping effort raster extraction (dir not found: %s)\n",
              effort_raster_dir))
}

# ---------------------------------------------------------------------------
# Create export directory
# ---------------------------------------------------------------------------
require(arrow)
require(data.table)

export_dir <- file.path(clesso_config$output_dir, "nn_export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)
cat(sprintf("  Export directory: %s\n", export_dir))

# ---------------------------------------------------------------------------
# 1. Export pairs (the main training data)
# ---------------------------------------------------------------------------
cat("  Exporting pairs_dt...\n")

## Ensure weight column exists
if (!"w" %in% names(pairs_dt)) {
  cat("  Computing pair weights...\n")
  source(file.path(clesso_config$clesso_dir, "clesso_prepare_data.R"))
  ## Simple uniform weights if function not available
  pairs_dt[, w := 1.0]
}

## Add is_within if not present
if (!"is_within" %in% names(pairs_dt)) {
  pairs_dt[, is_within := as.integer(pair_type == "within")]
}

## Select columns needed by NN
export_cols <- c("site_i", "site_j", "species_i", "species_j",
                 "lon_i", "lat_i", "lon_j", "lat_j",
                 "y", "pair_type", "is_within", "w")
## Only keep columns that exist
export_cols <- intersect(export_cols, names(pairs_dt))
pairs_export <- pairs_dt[, ..export_cols]

arrow::write_feather(pairs_export, file.path(export_dir, "pairs.feather"))
cat(sprintf("  Saved %d pairs (%d cols)\n", nrow(pairs_export), ncol(pairs_export)))

# ---------------------------------------------------------------------------
# 2. Export site covariates (for alpha model)
# ---------------------------------------------------------------------------
cat("  Exporting site covariates...\n")

## Build unique site table with all covariates
unique_sites_export <- unique(pairs_dt[, .(
  site_id = site_i, lon = lon_i, lat = lat_i
)])
unique_sites_j_export <- unique(pairs_dt[, .(
  site_id = site_j, lon = lon_j, lat = lat_j
)])
setnames(unique_sites_j_export, c("site_id", "lon", "lat"))
unique_sites_export <- unique(rbind(unique_sites_export, unique_sites_j_export),
                               by = "site_id")

## Merge site_covs if available
if (exists("site_covs") && !is.null(site_covs)) {
  site_covs_dt <- as.data.table(site_covs)
  if ("site_id" %in% names(site_covs_dt)) {
    unique_sites_export <- merge(unique_sites_export, site_covs_dt,
                                  by = "site_id", all.x = TRUE)
  }
}

arrow::write_feather(unique_sites_export, file.path(export_dir, "site_covariates.feather"))
cat(sprintf("  Saved %d sites with %d columns\n",
            nrow(unique_sites_export), ncol(unique_sites_export)))

# ---------------------------------------------------------------------------
# 3. Export env site table (for turnover / beta model)
# ---------------------------------------------------------------------------
cat("  Exporting env site table...\n")

if (exists("env_site_table") && !is.null(env_site_table)) {
  env_dt <- as.data.table(env_site_table)
  arrow::write_feather(env_dt, file.path(export_dir, "env_site_table.feather"))
  cat(sprintf("  Saved %d sites with %d env columns\n",
              nrow(env_dt), ncol(env_dt) - 1))
} else {
  cat("  No env_site_table available. NN will use geographic distance only.\n")
}

# ---------------------------------------------------------------------------
# 4. Export observed richness per site
# ---------------------------------------------------------------------------
cat("  Exporting observed richness...\n")

if (exists("site_obs_richness") && !is.null(site_obs_richness)) {
  arrow::write_feather(as.data.table(site_obs_richness),
                       file.path(export_dir, "site_obs_richness.feather"))
  cat(sprintf("  Saved observed richness for %d sites\n",
              nrow(site_obs_richness)))
} else {
  ## Compute from obs_dt if available
  if (exists("obs_dt")) {
    sor <- obs_dt[, .(S_obs = uniqueN(species)), by = .(site_id)]
    arrow::write_feather(sor, file.path(export_dir, "site_obs_richness.feather"))
    cat(sprintf("  Computed and saved observed richness for %d sites\n", nrow(sor)))
  } else {
    cat("  No observed richness available.\n")
  }
}

# ---------------------------------------------------------------------------
# 5. Export metadata (for reproducibility)
# ---------------------------------------------------------------------------
cat("  Exporting metadata...\n")

## Identify which effort columns actually made it into site_covs
effort_cols_in_export <- if (exists("effort_parts") && length(effort_parts) > 0)
                            names(effort_parts) else character(0)

metadata <- list(
  species_group       = clesso_config$species_group,
  run_id              = clesso_config$run_id,
  n_pairs             = nrow(pairs_dt),
  n_sites             = nrow(unique_sites_export),
  min_date            = as.character(clesso_config$min_date),
  max_date            = as.character(clesso_config$max_date),
  n_within            = sum(pairs_dt$pair_type == "within"),
  n_between           = sum(pairs_dt$pair_type == "between"),
  geo_distance        = clesso_config$geo_distance,
  export_date         = as.character(Sys.time()),
  alpha_cov_cols      = if (exists("site_covs") && !is.null(site_covs))
                          setdiff(names(site_covs), c("site_id", effort_cols_in_export))
                        else c("lon", "lat"),
  env_cov_cols        = if (exists("env_site_table") && !is.null(env_site_table))
                          setdiff(names(env_site_table), "site_id") else character(0),
  effort_cov_cols     = effort_cols_in_export
)

saveRDS(metadata, file.path(export_dir, "metadata.rds"))
## Also save as JSON for Python
json_str <- jsonlite::toJSON(metadata, auto_unbox = TRUE, pretty = TRUE)
writeLines(json_str, file.path(export_dir, "metadata.json"))

cat(sprintf("\n=== Export complete: %s ===\n", export_dir))
cat("  Files:\n")
cat("    pairs.feather\n")
cat("    site_covariates.feather\n")
if (exists("env_site_table") && !is.null(env_site_table)) cat("    env_site_table.feather\n")
cat("    site_obs_richness.feather\n")
cat("    metadata.json / metadata.rds\n")
cat(sprintf("\n  Next: run the NN pipeline:\n"))
cat(sprintf("    python src/clesso_nn/run_clesso_nn.py --export-dir %s\n\n", export_dir))
