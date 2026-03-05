##############################################################################
##
## test_step6a_env_extraction.R  --  Smoke-test environmental data extraction
##
## Tests step 6a (spatial + temporal env extraction via run_chunked_env)
## on small subsets (1000 rows each) before running the full dataset.
## This confirms the pipeline is working end-to-end and gives timing
## estimates without committing to the full (potentially hours-long) run.
##
## Three phases:
##   Phase 1: 1000 spatial rows only
##   Phase 2: 1000 temporal rows only
##   Phase 3: Full dataset (spatial + temporal + biAverage)
##            -- only runs if phases 1 & 2 succeed and user opts in
##
## Usage:
##   source("test_step6a_env_extraction.R")
##
##   Or set RUN_FULL=TRUE env var to auto-proceed to phase 3:
##     Sys.setenv(RUN_FULL = "TRUE")
##     source("test_step6a_env_extraction.R")
##
##############################################################################

cat("=== Step 6a Environment Extraction Test ===\n\n")

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
source(file.path(this_dir, "config.R"))
source(file.path(config$r_dir, "utils.R"))
source(file.path(config$r_dir, "gen_windows.R"))
source(file.path(config$r_dir, "run_chunked_env.R"))

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------
load_packages()
rasterOptions(tmpdir = config$raster_tmpdir)

# ---------------------------------------------------------------------------
# Configuration for this test
# ---------------------------------------------------------------------------
N_TEST_ROWS  <- 1000L          # rows per test phase
TEST_CHUNK   <- 500L           # smaller chunks for quick feedback
RUN_FULL_ENV <- as.logical(Sys.getenv("RUN_FULL", unset = "FALSE"))

cat(sprintf("  Test rows:   %d (spatial) + %d (temporal)\n", N_TEST_ROWS, N_TEST_ROWS))
cat(sprintf("  Chunk size:  %d (test) / %d (full)\n", TEST_CHUNK, config$chunk_size))
cat(sprintf("  biAverage:   %s\n", config$biAverage))
cat(sprintf("  Cores:       %d\n", config$cores_to_use))
cat(sprintf("  Run full:    %s\n\n", RUN_FULL_ENV))

# ---------------------------------------------------------------------------
# Load the saved obsPairs table (from steps 1-5)
# ---------------------------------------------------------------------------
obspairs_file <- file.path(
  config$output_dir,
  paste0("ObsPairsTable_RECA_", config$species_group, "_WindowTestRuns.rds")
)
if (!file.exists(obspairs_file)) stop(paste("ObsPairs file not found:", obspairs_file))
obsPairs_orig <- readRDS(obspairs_file)
cat(sprintf("Loaded ObsPairs table: %d rows x %d cols\n", nrow(obsPairs_orig), ncol(obsPairs_orig)))

# ---------------------------------------------------------------------------
# Apply temporal filter (mirrors run_obsGDM.R step 6 preamble)
# ---------------------------------------------------------------------------
c_yr <- config$climate_window

earliest_year_all <- pmin(obsPairs_orig$year1, obsPairs_orig$year2)
tst <- (earliest_year_all - c_yr) >= config$geonpy_start_year
obsPairs_out <- obsPairs_orig[tst, ]
ext_data     <- obsPairs_out[, 2:9]
cat(sprintf("After temporal filter: %d rows\n\n", nrow(obsPairs_out)))

if (nrow(obsPairs_out) < N_TEST_ROWS) {
  stop(sprintf("Only %d rows after temporal filter -- fewer than the %d test rows requested.",
               nrow(obsPairs_out), N_TEST_ROWS))
}

# ---------------------------------------------------------------------------
# Build spatial and temporal pair frames (same logic as run_obsGDM.R)
# ---------------------------------------------------------------------------
earliest_year  <- pmin(ext_data[, 3], ext_data[, 7])
earliest_month <- ifelse(ext_data[, 3] <= ext_data[, 7],
                         ext_data[, 4], ext_data[, 8])

spatial_pairs_full <- data.frame(
  Lon1   = ext_data[, 1],
  Lat1   = ext_data[, 2],
  year1  = earliest_year,
  month1 = earliest_month,
  Lon2   = ext_data[, 5],
  Lat2   = ext_data[, 6],
  year2  = earliest_year,
  month2 = earliest_month
)

temporal_pairs_full <- data.frame(
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
# Build env param lists (fixed climate window, spatial/temporal prefixes)
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

# ---------------------------------------------------------------------------
# Helper: run a phase with timing and error handling
# ---------------------------------------------------------------------------
run_phase <- function(label, pairs_data, params, n_rows, chunk_size,
                      swap_sites = FALSE) {
  cat(sprintf("=====================================================\n"))
  cat(sprintf("  %s  (%d rows, chunk=%d)\n", label, n_rows, chunk_size))
  cat(sprintf("=====================================================\n"))

  subset <- pairs_data[1:n_rows, , drop = FALSE]

  ## Quick sanity checks on the subset
  cat(sprintf("  Year range:  %d – %d\n", min(c(subset$year1, subset$year2)),
              max(c(subset$year1, subset$year2))))
  cat(sprintf("  Lon range:   %.2f – %.2f\n", min(c(subset$Lon1, subset$Lon2)),
              max(c(subset$Lon1, subset$Lon2))))
  cat(sprintf("  Lat range:   %.2f – %.2f\n", min(c(subset$Lat1, subset$Lat2)),
              max(c(subset$Lat1, subset$Lat2))))

  n_workers <- min(length(params), config$cores_to_use)
  cat(sprintf("  Creating cluster with %d workers...\n", n_workers))
  flush.console()
  cl <- makeCluster(n_workers)
  registerDoSNOW(cl)
  cat("  Cluster ready. Starting env extraction...\n")
  flush.console()

  t0 <- proc.time()
  result <- tryCatch({
    out <- run_chunked_env(subset, params, label,
                           swap_sites = swap_sites,
                           chunk_size = chunk_size)
    elapsed <- (proc.time() - t0)["elapsed"]
    list(success = TRUE, data = out, elapsed = elapsed, error = NULL)
  }, error = function(e) {
    elapsed <- (proc.time() - t0)["elapsed"]
    list(success = FALSE, data = NULL, elapsed = elapsed, error = conditionMessage(e))
  })

  cat("  Shutting down cluster...\n")
  flush.console()
  stopCluster(cl)
  registerDoSEQ()

  if (result$success) {
    cat(sprintf("\n  SUCCESS: %d rows x %d cols in %.2f sec (%.2f min)\n",
                nrow(result$data), ncol(result$data), result$elapsed, result$elapsed / 60))
    cat(sprintf("  Rate: %.1f rows/sec\n", n_rows / result$elapsed))
    cat(sprintf("  Estimated full (%d rows): %.1f min\n\n",
                nrow(pairs_data),
                (nrow(pairs_data) / n_rows) * result$elapsed / 60))

    ## Print first few cols/rows as sanity check
    cat("  Sample output (first 5 rows, first 6 cols):\n")
    print(head(result$data[, 1:min(6, ncol(result$data))], 5))
    cat("\n")

    ## Check for NAs and -9999 sentinels
    n_na      <- sum(is.na(result$data))
    n_sentinel <- sum(result$data == -9999, na.rm = TRUE)
    cat(sprintf("  NAs: %d  |  -9999 sentinels: %d\n\n", n_na, n_sentinel))
  } else {
    cat(sprintf("\n  FAILED after %.2f sec\n", result$elapsed))
    cat(sprintf("  Error: %s\n\n", result$error))
  }

  result
}

# ===========================================================================
# PHASE 1: Spatial extraction (1000 rows)
# ===========================================================================
cat("\n")
phase1 <- run_phase("Phase 1: SPATIAL (test)",
                     spatial_pairs_full, spatial_params,
                     N_TEST_ROWS, TEST_CHUNK)

# ===========================================================================
# PHASE 2: Temporal extraction (1000 rows)
# ===========================================================================
phase2 <- run_phase("Phase 2: TEMPORAL (test)",
                     temporal_pairs_full, temporal_params,
                     N_TEST_ROWS, TEST_CHUNK)

# ===========================================================================
# PHASE 2b: Spatial biAverage (swap_sites=TRUE) if biAverage is enabled
# ===========================================================================
if (config$biAverage && phase1$success) {
  phase2b <- run_phase("Phase 2b: SPATIAL-B biAvg (test)",
                       spatial_pairs_full, spatial_params,
                       N_TEST_ROWS, TEST_CHUNK,
                       swap_sites = TRUE)
  if (phase2b$success) {
    env_spatial_test <- (phase1$data + phase2b$data) / 2
    cat("  biAverage spatial test: OK (averaged A and B)\n\n")
  }
} else {
  phase2b <- list(success = TRUE)
}

# ===========================================================================
# Summary
# ===========================================================================
cat("=====================================================\n")
cat("  TEST SUMMARY\n")
cat("=====================================================\n")
cat(sprintf("  Phase 1 (Spatial, %d rows):       %s  (%.2f sec)\n",
            N_TEST_ROWS,
            if (phase1$success) "PASS" else "FAIL",
            phase1$elapsed))
cat(sprintf("  Phase 2 (Temporal, %d rows):      %s  (%.2f sec)\n",
            N_TEST_ROWS,
            if (phase2$success) "PASS" else "FAIL",
            phase2$elapsed))
if (config$biAverage) {
  cat(sprintf("  Phase 2b (Spatial-B biAvg, %d):   %s  (%.2f sec)\n",
              N_TEST_ROWS,
              if (phase2b$success) "PASS" else "FAIL",
              phase2b$elapsed))
}

all_pass <- phase1$success && phase2$success && phase2b$success

if (all_pass) {
  n_full        <- nrow(spatial_pairs_full)
  spatial_rate  <- N_TEST_ROWS / phase1$elapsed
  temporal_rate <- N_TEST_ROWS / phase2$elapsed
  biavg_time    <- if (config$biAverage && phase2b$success) phase2b$elapsed else 0
  biavg_rate    <- if (biavg_time > 0) N_TEST_ROWS / biavg_time else Inf

  est_spatial_min  <- (n_full / spatial_rate) / 60
  est_temporal_min <- (n_full / temporal_rate) / 60
  est_biavg_min    <- if (is.finite(biavg_rate)) (n_full / biavg_rate) / 60 else 0
  est_total_min    <- est_spatial_min + est_temporal_min + est_biavg_min

  cat(sprintf("\n  Full dataset: %d rows\n", n_full))
  cat(sprintf("  Estimated full run time:\n"))
  cat(sprintf("    Spatial:   %.1f min\n", est_spatial_min))
  cat(sprintf("    Temporal:  %.1f min\n", est_temporal_min))
  if (config$biAverage) {
    cat(sprintf("    biAvg-B:   %.1f min\n", est_biavg_min))
  }
  cat(sprintf("    TOTAL:     %.1f min (~%.1f hours)\n\n", est_total_min, est_total_min / 60))
} else {
  cat("\n  One or more phases FAILED -- fix errors before running full extraction.\n\n")
}

# ===========================================================================
# PHASE 3: Full dataset (optional)
# ===========================================================================
if (all_pass && RUN_FULL_ENV) {
  cat("=====================================================\n")
  cat("  PHASE 3: FULL DATASET EXTRACTION\n")
  cat("=====================================================\n\n")

  t0_full <- proc.time()

  ## Spatial A
  n_workers <- min(length(spatial_params) + length(temporal_params),
                   config$cores_to_use)
  cl <- makeCluster(n_workers)
  registerDoSNOW(cl)

  cat("--- Spatial-A (full) ---\n")
  env_spatA <- run_chunked_env(spatial_pairs_full, spatial_params, "Spatial-A")

  if (config$biAverage) {
    cat("--- Spatial-B biAverage (full) ---\n")
    env_spatB <- run_chunked_env(spatial_pairs_full, spatial_params, "Spatial-B",
                                  swap_sites = TRUE)
    env_spatial <- (env_spatA + env_spatB) / 2
    rm(env_spatB)
  } else {
    env_spatial <- env_spatA
  }
  rm(env_spatA)

  cat("--- Temporal (full) ---\n")
  env_temporal <- run_chunked_env(temporal_pairs_full, temporal_params, "Temporal")

  stopCluster(cl)
  registerDoSEQ()

  elapsed_full <- (proc.time() - t0_full)["elapsed"]
  cat(sprintf("\n  FULL EXTRACTION COMPLETE: %.1f sec (%.1f min)\n",
              elapsed_full, elapsed_full / 60))
  cat(sprintf("  Spatial:  %d x %d\n", nrow(env_spatial), ncol(env_spatial)))
  cat(sprintf("  Temporal: %d x %d\n", nrow(env_temporal), ncol(env_temporal)))

  ## Split and combine (same as run_obsGDM.R)
  env_spat1 <- env_spatial[, grep("_1$", names(env_spatial))]
  env_spat2 <- env_spatial[, grep("_2$", names(env_spatial))]
  env_temp1 <- env_temporal[, grep("_1$", names(env_temporal))]
  env_temp2 <- env_temporal[, grep("_2$", names(env_temporal))]

  ## Substrate
  pnt1     <- SpatialPoints(data.frame(ext_data[, 1], ext_data[, 2]))
  pnt2     <- SpatialPoints(data.frame(ext_data[, 5], ext_data[, 6]))
  subs_brk <- brick(config$substrate_raster)
  env1_subs <- extract(subs_brk, pnt1)
  env2_subs <- extract(subs_brk, pnt2)
  colnames(env1_subs) <- paste0(colnames(env1_subs), "_1")
  colnames(env2_subs) <- paste0(colnames(env2_subs), "_2")

  ## Combine
  save_prefix <- paste0(config$species_group, "_",
                        format(config$nMatch / 1e6, nsmall = 0), "mil_",
                        c_yr, "climWin_STresid_")
  if (config$biAverage) save_prefix <- paste0(save_prefix, "biAverage_")
  if (config$decomposition != "none") save_prefix <- paste0(config$decomposition, "_", save_prefix)
  if (config$add_modis) save_prefix <- paste0(save_prefix, config$modis_suffix)

  parts <- list(obsPairs_out,
                env_spat1, env1_subs,
                env_spat2, env2_subs,
                env_temp1, env_temp2)
  parts <- parts[!sapply(parts, is.null)]
  obsPairs_combined <- do.call(cbind, parts)

  env_file <- file.path(config$output_dir, paste0(save_prefix, "ObsEnvTable.RData"))
  save(obsPairs_combined, file = env_file)
  cat(sprintf("  Saved: %s\n", basename(env_file)))
  cat(sprintf("  Final table: %d rows x %d cols\n\n", nrow(obsPairs_combined), ncol(obsPairs_combined)))

  rm(env_spatial, env_temporal, env_spat1, env_spat2, env_temp1, env_temp2)
  gc(verbose = FALSE)
} else if (all_pass && !RUN_FULL_ENV) {
  cat("To proceed with the full extraction, re-run with:\n")
  cat("  Sys.setenv(RUN_FULL = \"TRUE\")\n")
  cat("  source(\"test_step6a_env_extraction.R\")\n\n")
}

cat("=== Test complete ===\n")
