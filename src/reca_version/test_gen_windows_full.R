##############################################################################
##
## test_gen_windows_full.R  —  Full-scale env extraction test (~2M rows)
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
source(file.path(this_dir, "src", "reca_version", "config.R"))
source(file.path(config$r_dir, "utils.R"))
source(file.path(config$r_dir, "gdm_functions.R"))
source(file.path(config$r_dir, "gen_windows.R"))

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
cat(sprintf("Total rows to extract: %d\n\n", nrow(ext_data)))

# ---------------------------------------------------------------------------
# Chunked parallel extraction settings
# ---------------------------------------------------------------------------
CHUNK_SIZE <- 10000   # rows per chunk — adjust for progress granularity vs overhead
n_total    <- nrow(ext_data)
chunk_idx  <- split(1:n_total, ceiling(1:n_total / CHUNK_SIZE))
n_chunks   <- length(chunk_idx)

cat(sprintf("Chunk size: %d rows  |  Total chunks: %d\n\n", CHUNK_SIZE, n_chunks))

# ---------------------------------------------------------------------------
# Helper: run one direction (all env-params) on full data, chunk by chunk
# ---------------------------------------------------------------------------
run_chunked_extraction <- function(ext_data, init_params, direction_label,
                                   swap_sites = FALSE, cl) {
  n_total    <- nrow(ext_data)
  n_params   <- length(init_params)
  local_idx  <- split(1:n_total, ceiling(1:n_total / CHUNK_SIZE))
  local_nchunks <- length(local_idx)

  cat(sprintf("[%s] === Direction %s: Starting chunked extraction (%d rows, %d chunks) ===\n",
              format(Sys.time(), "%H:%M:%S"), direction_label, n_total, local_nchunks))

  chunk_results <- vector("list", local_nchunks)
  t0 <- proc.time()

  for (i in seq_along(local_idx)) {
    rows <- local_idx[[i]]
    if (swap_sites) {
      ext_chunk <- ext_data[rows, c(1, 2, 7, 8, 5, 6, 3, 4)]
    } else {
      ext_chunk <- ext_data[rows, ]
    }

    chunk_out <- foreach(x = 1:n_params, .combine = "cbind",
                         .packages = "arrow",
                         .export = c("gen_windows", "config")) %dopar% {
      out <- gen_windows(
        pairs        = ext_chunk,
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

    chunk_results[[i]] <- chunk_out

    ## Progress reporting
    elapsed    <- (proc.time() - t0)["elapsed"]
    rows_done  <- max(rows)
    pct        <- 100 * rows_done / n_total
    rate       <- rows_done / elapsed
    remaining  <- (n_total - rows_done) / rate
    cat(sprintf("  [%s] (%s) Chunk %d/%d | %d/%d rows (%.1f%%) | %.1fs elapsed | est. %.1f min remaining\n",
                format(Sys.time(), "%H:%M:%S"), direction_label,
                i, local_nchunks, rows_done, n_total, pct,
                elapsed, remaining / 60))
  }

  total_elapsed <- (proc.time() - t0)["elapsed"]
  env_out <- do.call(rbind, chunk_results)

  cat(sprintf("[%s] Direction %s COMPLETE: %.1f sec (%.1f min) — output %d x %d\n",
              format(Sys.time(), "%H:%M:%S"), direction_label,
              total_elapsed, total_elapsed / 60,
              nrow(env_out), ncol(env_out)))

  list(result = env_out, elapsed = total_elapsed)
}

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

warmup_rows <- min(3 * CHUNK_SIZE, nrow(ext_data))
ext_data_warmup <- ext_data[1:warmup_rows, ]
warmup <- run_chunked_extraction(ext_data_warmup, init_params, "warmup",
                                  swap_sites = FALSE, cl = cl)

cat(sprintf("  Estimated full Direction A: %.1f min for %d rows\n\n",
            (warmup$elapsed / warmup_rows) * nrow(ext_data) / 60,
            nrow(ext_data)))
rm(warmup)
gc(verbose = FALSE)

# ===========================================================================
# Direction A: Normal site order (full data)
# ===========================================================================
resultA <- run_chunked_extraction(ext_data, init_params, "A",
                                   swap_sites = FALSE, cl = cl)
env_outA  <- resultA$result
elapsed_A <- resultA$elapsed

# ===========================================================================
# Direction B: Swapped sites (bidirectional averaging)
# ===========================================================================
if (config$biAverage) {
  resultB <- run_chunked_extraction(ext_data, init_params, "B",
                                     swap_sites = TRUE, cl = cl)
  env_outB  <- resultB$result
  elapsed_B <- resultB$elapsed

  cat("Computing bidirectional average...\n")
  env_out <- (env_outA + env_outB) / 2
  rm(env_outA, env_outB, resultA, resultB)
} else {
  env_out <- env_outA
  rm(env_outA, resultA)
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
