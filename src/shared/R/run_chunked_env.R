##############################################################################
##
## run_chunked_env.R  —  Chunked parallel environmental data extraction
##
## Extracts environmental data in chunks of configurable size, running
## gen_windows() for each env-param set in parallel via foreach/doSNOW.
## Reports per-chunk progress on the main process.
##
## Requires:
##   - gen_windows() already sourced
##   - config list (for npy_src, geonpy_start_year, python_exe,
##                  pyper_script, feather_tmpdir, chunk_size)
##   - A doSNOW cluster already registered (makeCluster + registerDoSNOW)
##
##############################################################################

# ---------------------------------------------------------------------------
# run_chunked_env
#
# Parameters:
#   pairs_data      - data.frame with 8 columns: Lon1 Lat1 year1 month1
#                     Lon2 Lat2 year2 month2
#   params          - list of env-param lists, each with:
#                     variables, mstat, cstat, window, prefix
#   direction_label - string label for progress messages (e.g. "A", "B",
#                     "Spatial-A", "Temporal")
#   swap_sites      - if TRUE, swap site-1/site-2 columns before extraction
#   chunk_size      - number of rows per chunk (default: config$chunk_size
#                     if available, otherwise 10000)
# ---------------------------------------------------------------------------
run_chunked_env <- function(pairs_data, params, direction_label,
                            swap_sites = FALSE,
                            chunk_size = NULL) {

  ## Determine chunk size: explicit arg > config > default 10000
  if (is.null(chunk_size)) {
    chunk_size <- if (exists("config") && !is.null(config$chunk_size)) {
      config$chunk_size
    } else {
      10000L
    }
  }

  n_total       <- nrow(pairs_data)
  local_idx     <- split(1:n_total, ceiling(1:n_total / chunk_size))
  local_nchunks <- length(local_idx)
  n_params      <- length(params)

  cat(sprintf("  [%s] %s: %d rows, %d chunks (size %d), %d env-params\n",
              format(Sys.time(), "%H:%M:%S"), direction_label,
              n_total, local_nchunks, chunk_size, n_params))
  flush.console()

  chunk_results <- vector("list", local_nchunks)
  t0 <- proc.time()

  for (i in seq_along(local_idx)) {
    rows <- local_idx[[i]]
    cat(sprintf("  [%s] (%s) Starting chunk %d/%d (rows %d-%d)...\n",
                format(Sys.time(), "%H:%M:%S"), direction_label,
                i, local_nchunks, min(rows), max(rows)))
    flush.console()

    if (swap_sites) {
      ext_chunk <- pairs_data[rows, c(1, 2, 7, 8, 5, 6, 3, 4)]
    } else {
      ext_chunk <- pairs_data[rows, ]
    }

    chunk_out <- foreach(x = 1:n_params, .combine = "cbind",
                         .packages = "arrow",
                         .export = c("gen_windows", "config")) %dopar% {
      out <- gen_windows(
        pairs        = ext_chunk,
        variables    = params[[x]]$variables,
        mstat        = params[[x]]$mstat,
        cstat        = params[[x]]$cstat,
        window       = params[[x]]$window,
        npy_src      = config$npy_src,
        start_year   = config$geonpy_start_year,
        python_exe   = config$python_exe,
        pyper_script = config$pyper_script,
        feather_tmpdir = config$feather_tmpdir
      )
      colnames(out) <- paste(params[[x]]$prefix, colnames(out), sep = "_")
      out[, 9:ncol(out)]
    }

    chunk_results[[i]] <- chunk_out

    elapsed   <- (proc.time() - t0)["elapsed"]
    rows_done <- max(rows)
    pct       <- 100 * rows_done / n_total
    rate      <- rows_done / elapsed
    remaining <- (n_total - rows_done) / rate
    cat(sprintf("  [%s] (%s) Chunk %d/%d | %d/%d rows (%.1f%%) | %.1fs elapsed | est. %.1f min remaining\n",
                format(Sys.time(), "%H:%M:%S"), direction_label,
                i, local_nchunks, rows_done, n_total, pct,
                elapsed, remaining / 60))
    flush.console()
  }

  total_elapsed <- (proc.time() - t0)["elapsed"]
  env_out <- do.call(rbind, chunk_results)
  cat(sprintf("  [%s] %s COMPLETE: %.1f sec (%.1f min) — %d x %d\n",
              format(Sys.time(), "%H:%M:%S"), direction_label,
              total_elapsed, total_elapsed / 60,
              nrow(env_out), ncol(env_out)))
  flush.console()
  env_out
}
