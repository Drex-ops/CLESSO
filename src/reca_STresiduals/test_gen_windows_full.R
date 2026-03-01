##############################################################################
##
## test_gen_windows_full.R  —  Full-scale env extraction test (STresiduals)
##
## Runs the complete Step 6a environmental data extraction on the full
## obsPairs table using the spatial-temporal residual decomposition:
##   - SPATIAL pairs: both sites at earliest year
##   - TEMPORAL pairs: site 2 at earliest year vs observation year
##   - biAverage (if enabled): swap sites for spatial only
##
## Uses the same parallel foreach + chunked approach as run_obsGDM.R
## with timestamped progress logging.
##
## Usage:
##   source("test_gen_windows_full.R")
##
##############################################################################

cat("=== Full-Scale Step 6a Env Extraction Test (STresiduals) ===\n\n")

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
source(file.path(config$r_dir, "utils.R"))
source(file.path(config$r_dir, "gdm_functions.R"))
source(file.path(config$r_dir, "gen_windows.R"))
source(file.path(config$r_dir, "run_chunked_env.R"))

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------
load_packages()
rasterOptions(tmpdir = config$raster_tmpdir)

# ---------------------------------------------------------------------------
# Load the saved obsPairs table
# ---------------------------------------------------------------------------
obspairs_file <- file.path(
  config$output_dir,
  paste0("ObsPairsTable_RECA_", config$species_group, "_WindowTestRuns.rds")
)
if (!file.exists(obspairs_file)) stop(paste("ObsPairs file not found:", obspairs_file))
obsPairs_out <- readRDS(obspairs_file)

cat(sprintf("ObsPairs table dimensions: %d rows x %d columns\n",
            nrow(obsPairs_out), ncol(obsPairs_out)))

# ---------------------------------------------------------------------------
# Apply temporal filter (same as run_obsGDM.R STresiduals step 6)
# ---------------------------------------------------------------------------
c_yr <- config$climate_window

earliest_year_all <- pmin(obsPairs_out$year1, obsPairs_out$year2)
tst <- (earliest_year_all - c_yr) >= config$geonpy_start_year
obsPairs_out <- obsPairs_out[tst, ]
ext_data     <- obsPairs_out[, 2:9]

cat(sprintf("Climate window: %d yrs\n", c_yr))
cat(sprintf("After temporal filter: %d rows\n\n", nrow(obsPairs_out)))

# ---------------------------------------------------------------------------
# Build spatial and temporal pair frames
# ---------------------------------------------------------------------------
earliest_year  <- pmin(ext_data[, 3], ext_data[, 7])
earliest_month <- ifelse(ext_data[, 3] <= ext_data[, 7],
                         ext_data[, 4], ext_data[, 8])

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

temporal_pairs <- data.frame(
  Lon1   = ext_data[, 5],
  Lat1   = ext_data[, 6],
  year1  = earliest_year,
  month1 = earliest_month,
  Lon2   = ext_data[, 5],
  Lat2   = ext_data[, 6],
  year2  = ext_data[, 7],
  month2 = ext_data[, 8]
)

# ---------------------------------------------------------------------------
# Build env param lists (spatial + temporal prefixes)
# ---------------------------------------------------------------------------
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

n_spat_params <- length(spatial_params)
n_temp_params <- length(temporal_params)
n_workers     <- min(n_spat_params + n_temp_params, config$cores_to_use)

cat(sprintf("Spatial env-param sets:  %d\n", n_spat_params))
cat(sprintf("Temporal env-param sets: %d\n", n_temp_params))
cat(sprintf("Workers: %d  |  biAverage: %s\n", n_workers, config$biAverage))
cat(sprintf("Total rows: %d  |  Chunk size: %d\n\n", nrow(spatial_pairs), config$chunk_size))

# ===========================================================================
# Start cluster
# ===========================================================================
cl <- makeCluster(n_workers)
registerDoSNOW(cl)

# ===========================================================================
# Warm-up: 3 chunks of spatial to verify progress & estimate timing
# ===========================================================================
cat("=====================================================\n")
cat("WARM-UP: Testing first 3 chunks (spatial) to verify feedback\n")
cat("=====================================================\n")

warmup_rows <- min(3 * config$chunk_size, nrow(spatial_pairs))
warmup_data <- spatial_pairs[1:warmup_rows, ]
t0_warmup   <- proc.time()
warmup <- run_chunked_env(warmup_data, spatial_params, "warmup",
                          swap_sites = FALSE)
warmup_elapsed <- (proc.time() - t0_warmup)["elapsed"]
warmup_rate    <- warmup_rows / warmup_elapsed

est_spatial_min  <- (nrow(spatial_pairs) / warmup_rate) / 60
est_biavg_min    <- if (config$biAverage) est_spatial_min else 0
est_temporal_min <- est_spatial_min  # roughly same scale
est_total_min    <- est_spatial_min + est_biavg_min + est_temporal_min

cat(sprintf("\n  Warmup: %d rows in %.1f sec (%.1f rows/sec)\n",
            warmup_rows, warmup_elapsed, warmup_rate))
cat(sprintf("  Estimated full run: ~%.0f min (~%.1f hours)\n",
            est_total_min, est_total_min / 60))
cat(sprintf("    Spatial-A: ~%.0f min\n", est_spatial_min))
if (config$biAverage) cat(sprintf("    Spatial-B: ~%.0f min\n", est_biavg_min))
cat(sprintf("    Temporal:  ~%.0f min\n\n", est_temporal_min))
rm(warmup)
gc(verbose = FALSE)

# ===========================================================================
# Spatial-A: Normal site order (full data)
# ===========================================================================
cat("=====================================================\n")
cat("SPATIAL-A: Full extraction\n")
cat("=====================================================\n")
t0_spatA   <- proc.time()
env_spatA  <- run_chunked_env(spatial_pairs, spatial_params, "Spatial-A",
                               swap_sites = FALSE)
elapsed_spatA <- (proc.time() - t0_spatA)["elapsed"]

# ===========================================================================
# Spatial-B: Swapped sites (bidirectional averaging)
# ===========================================================================
elapsed_spatB <- 0
if (config$biAverage) {
  cat("\n=====================================================\n")
  cat("SPATIAL-B: biAverage (swapped sites)\n")
  cat("=====================================================\n")
  t0_spatB   <- proc.time()
  env_spatB  <- run_chunked_env(spatial_pairs, spatial_params, "Spatial-B",
                                 swap_sites = TRUE)
  elapsed_spatB <- (proc.time() - t0_spatB)["elapsed"]

  cat("Computing bidirectional average...\n")
  env_spatial <- (env_spatA + env_spatB) / 2
  rm(env_spatA, env_spatB)
} else {
  env_spatial <- env_spatA
  rm(env_spatA)
}

# ===========================================================================
# Temporal: Full extraction
# ===========================================================================
cat("\n=====================================================\n")
cat("TEMPORAL: Full extraction\n")
cat("=====================================================\n")
t0_temp     <- proc.time()
env_temporal <- run_chunked_env(temporal_pairs, temporal_params, "Temporal",
                                 swap_sites = FALSE)
elapsed_temp <- (proc.time() - t0_temp)["elapsed"]

stopCluster(cl)
registerDoSEQ()
gc(verbose = FALSE)

# ===========================================================================
# Summary
# ===========================================================================
total_elapsed <- elapsed_spatA + elapsed_spatB + elapsed_temp

cat("\n=====================================================\n")
cat("FULL-SCALE EXTRACTION SUMMARY (STresiduals)\n")
cat("=====================================================\n\n")
cat(sprintf("  Rows extracted    : %d\n", nrow(spatial_pairs)))
cat(sprintf("  Climate window    : %d years\n", c_yr))
cat(sprintf("  Spatial params    : %d\n", n_spat_params))
cat(sprintf("  Temporal params   : %d\n", n_temp_params))
cat(sprintf("  Workers           : %d\n", n_workers))
cat(sprintf("  biAverage         : %s\n\n", config$biAverage))

cat(sprintf("  Spatial output    : %d x %d\n", nrow(env_spatial), ncol(env_spatial)))
cat(sprintf("  Temporal output   : %d x %d\n\n", nrow(env_temporal), ncol(env_temporal)))

cat(sprintf("  Spatial-A time    : %7.1f sec  (%5.1f min)\n", elapsed_spatA, elapsed_spatA / 60))
if (config$biAverage) {
  cat(sprintf("  Spatial-B time    : %7.1f sec  (%5.1f min)\n", elapsed_spatB, elapsed_spatB / 60))
}
cat(sprintf("  Temporal time     : %7.1f sec  (%5.1f min)\n", elapsed_temp, elapsed_temp / 60))
cat(sprintf("  Total time        : %7.1f sec  (%5.1f min)\n\n",
            total_elapsed, total_elapsed / 60))

per_row <- total_elapsed / nrow(spatial_pairs)
cat(sprintf("  Throughput        : %.6f sec/row\n", per_row))
cat(sprintf("  Est. 1M rows      : %.1f min\n", per_row * 1e6 / 60))
cat(sprintf("  Est. 2M rows      : %.1f min\n", per_row * 2e6 / 60))

cat("\n=== Full-scale test complete ===\n")
