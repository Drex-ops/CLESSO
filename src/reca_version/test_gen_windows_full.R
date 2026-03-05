##############################################################################
##
## test_gen_windows_full.R  --  Full-scale env extraction test (~2M rows)
##
## Runs the complete Step 6a environmental data extraction on the full
## obsPairs table (all rows after temporal filter) using the same parallel
## foreach approach as run_obsGDM.R, with timestamped progress logging
## from each worker.
##
## If biAverage is enabled in config, both directions (A and B) are run.
##
## Usage:
##   source("test_gen_windows_full.R")
##
##############################################################################

cat("=== Full-Scale Step 6a Env Extraction Test ===\n\n")

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
  config_path <- file.path(this_dir, "src", "reca_version", "config.R")
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
# Apply temporal filter (same as run_obsGDM.R step 6)
# ---------------------------------------------------------------------------
tst_1 <- as.Date(paste0(obsPairs_out$year1, "-01-01")) >
           (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
tst_2 <- as.Date(paste0(obsPairs_out$year2, "-01-01")) >
           (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
obsPairs_out <- obsPairs_out[tst_1 & tst_2, ]
ext_data     <- obsPairs_out[, 2:9]

cat(sprintf("After temporal filter: %d rows x %d cols\n\n",
            nrow(ext_data), ncol(ext_data)))

# ---------------------------------------------------------------------------
# Build env params (identical to run_obsGDM.R)
# ---------------------------------------------------------------------------
c_yr <- config$c_yrs[1]
w_yr <- config$w_yrs[1]

init_params <- list()
for (ep in config$env_params) {
  init_params[[length(init_params) + 1]] <- list(
    variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
    window = c_yr, prefix = paste0(ep$cstat, "Xbr_", c_yr)
  )
  init_params[[length(init_params) + 1]] <- list(
    variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
    window = w_yr, prefix = paste0(ep$cstat, "Xbr_", w_yr)
  )
}

n_params  <- length(init_params)
n_workers <- min(n_params, config$cores_to_use)

cat(sprintf("Climate window: %d yrs  |  Weather window: %d yrs\n", c_yr, w_yr))
cat(sprintf("Env-param sets: %d  |  Workers: %d  |  biAverage: %s\n",
            n_params, n_workers, config$biAverage))
cat(sprintf("Total rows to extract: %d\n", nrow(ext_data)))
cat(sprintf("Chunk size: %d rows\n\n", config$chunk_size))

# ===========================================================================
# Start cluster (reused across warm-up and full run)
# ===========================================================================
cl <- makeCluster(n_workers)
registerDoSNOW(cl)

# ===========================================================================
# Warm-up: 3 chunks to verify progress feedback & estimate timing
# ===========================================================================
cat("=====================================================\n")
cat("WARM-UP: Testing first 3 chunks to verify feedback\n")
cat("=====================================================\n")

warmup_rows <- min(3 * config$chunk_size, nrow(ext_data))
ext_data_warmup <- ext_data[1:warmup_rows, ]
t0_warmup <- proc.time()
warmup <- run_chunked_env(ext_data_warmup, init_params, "warmup",
                          swap_sites = FALSE)
warmup_elapsed <- (proc.time() - t0_warmup)["elapsed"]

cat(sprintf("  Estimated full Direction A: %.1f min for %d rows\n\n",
            (warmup_elapsed / warmup_rows) * nrow(ext_data) / 60,
            nrow(ext_data)))
rm(warmup)
gc(verbose = FALSE)

# ===========================================================================
# Direction A: Normal site order (full data)
# ===========================================================================
t0_A <- proc.time()
env_outA  <- run_chunked_env(ext_data, init_params, "A", swap_sites = FALSE)
elapsed_A <- (proc.time() - t0_A)["elapsed"]

# ===========================================================================
# Direction B: Swapped sites (bidirectional averaging)
# ===========================================================================
if (config$biAverage) {
  t0_B <- proc.time()
  env_outB  <- run_chunked_env(ext_data, init_params, "B", swap_sites = TRUE)
  elapsed_B <- (proc.time() - t0_B)["elapsed"]

  cat("Computing bidirectional average...\n")
  env_out <- (env_outA + env_outB) / 2
  rm(env_outA, env_outB)
} else {
  env_out <- env_outA
  rm(env_outA)
  elapsed_B <- 0
}

stopCluster(cl)
registerDoSEQ()
gc(verbose = FALSE)

# ===========================================================================
# Summary
# ===========================================================================
total_elapsed <- elapsed_A + elapsed_B

cat("\n=====================================================\n")
cat("FULL-SCALE EXTRACTION SUMMARY\n")
cat("=====================================================\n\n")
cat(sprintf("  Rows extracted    : %d\n", nrow(ext_data)))
cat(sprintf("  Env-param sets    : %d\n", n_params))
cat(sprintf("  Workers           : %d\n", n_workers))
cat(sprintf("  biAverage         : %s\n", config$biAverage))
cat(sprintf("  Output dimensions : %d x %d\n\n", nrow(env_out), ncol(env_out)))
cat(sprintf("  Direction A time  : %7.1f sec  (%5.1f min)\n", elapsed_A, elapsed_A / 60))
if (config$biAverage) {
  cat(sprintf("  Direction B time  : %7.1f sec  (%5.1f min)\n", elapsed_B, elapsed_B / 60))
}
cat(sprintf("  Total time        : %7.1f sec  (%5.1f min)\n\n",
            total_elapsed, total_elapsed / 60))

per_row <- total_elapsed / nrow(ext_data)
cat(sprintf("  Throughput        : %.6f sec/row\n", per_row))
cat(sprintf("  Est. 1M rows      : %.1f min\n", per_row * 1e6 / 60))
cat(sprintf("  Est. 2M rows      : %.1f min\n", per_row * 2e6 / 60))

cat("\n=== Full-scale test complete ===\n")
