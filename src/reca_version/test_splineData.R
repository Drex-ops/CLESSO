##############################################################################
##
## test_splineData.R  --  Performance & accuracy comparison:
##                        splineData() vs splineData_fast()
##
## Tests:
##   1. Accuracy   -- verify identical output on small, medium, and full data
##   2. Performance -- benchmark both functions at 1K, 10K, 100K, full dataset
##   3. Memory      -- report peak memory usage
##
## Usage:
##   source("test_splineData.R")
##
##############################################################################

cat("=== splineData vs splineData_fast: Performance & Accuracy ===\n\n")

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

load_packages()


# ---------------------------------------------------------------------------
# Build test data from the real obsPairs env-extracted table
# ---------------------------------------------------------------------------
cat("Loading data...\n")

## Try to load a previously saved env table (post-step 6a) for realistic data
## Fall back to constructing synthetic data if not available
env_file <- list.files(config$output_dir, pattern = "ObsEnvTable\\.RData$",
                       full.names = TRUE)

if (length(env_file) > 0) {
  cat(sprintf("  Loading real env table: %s\n", basename(env_file[1])))
  load(env_file[1])  # loads obsPairs_out
  env_cols  <- 23:ncol(obsPairs_out)
  full_data <- obsPairs_out[, env_cols]
  cat(sprintf("  Full data: %d rows x %d env columns\n", nrow(full_data), ncol(full_data)))
} else {
  ## Build from obsPairs RDS + extracting a few env params (quick synthetic)
  cat("  No saved env table found. Building synthetic test data...\n")
  obspairs_file <- file.path(
    config$output_dir,
    paste0("ObsPairsTable_RECA_", config$species_group, "_WindowTestRuns.rds")
  )
  if (!file.exists(obspairs_file)) stop("Cannot find obsPairs file: ", obspairs_file)
  obsPairs_out <- readRDS(obspairs_file)

  ## Create synthetic env columns (site1 and site2 halves, must be even)
  n   <- nrow(obsPairs_out)
  nc2 <- 30  # 30 predictors -> 60 columns (site1 + site2)
  set.seed(42)
  full_data <- as.data.frame(matrix(rnorm(n * nc2 * 2), nrow = n, ncol = nc2 * 2))
  colnames(full_data) <- c(paste0("env_", 1:nc2, "_1"), paste0("env_", 1:nc2, "_2"))
  cat(sprintf("  Synthetic data: %d rows x %d columns\n", nrow(full_data), ncol(full_data)))
}

## Verify even column count
stopifnot(ncol(full_data) %% 2 == 0)
nc2 <- ncol(full_data) / 2

cat(sprintf("  Predictors: %d  |  Spline bases: %d per predictor = %d total columns\n",
            nc2, 3, nc2 * 3))


# ===========================================================================
# TEST 1: ACCURACY -- verify identical results
# ===========================================================================
cat("\n=====================================================\n")
cat("TEST 1: Accuracy comparison\n")
cat("=====================================================\n\n")

test_sizes <- c(100, 1000, 10000)
for (sz in test_sizes) {
  n_use <- min(sz, nrow(full_data))
  test_input <- full_data[1:n_use, ]

  suppressMessages({
    out_old  <- splineData(test_input)
    out_fast <- splineData_fast(test_input)
  })

  ## Check dimensions
  dim_match <- identical(dim(out_old), dim(out_fast))

  ## Check column names
  names_match <- identical(colnames(out_old), colnames(out_fast))

  ## Check values -- allow tiny floating-point tolerance
  max_diff <- max(abs(out_old - out_fast), na.rm = TRUE)
  mean_diff <- mean(abs(out_old - out_fast), na.rm = TRUE)
  values_match <- max_diff < 1e-12

  ## Check NA positions match
  na_match <- identical(is.na(out_old), is.na(out_fast))

  status <- if (dim_match && names_match && values_match && na_match) "PASS" else "FAIL"

  cat(sprintf("  n=%6d | dims: %s | names: %s | max_diff: %.2e | mean_diff: %.2e | NAs: %s | %s\n",
              n_use,
              ifelse(dim_match, "OK", "MISMATCH"),
              ifelse(names_match, "OK", "MISMATCH"),
              max_diff, mean_diff,
              ifelse(na_match, "OK", "MISMATCH"),
              status))

  if (!values_match) {
    cat("    WARNING: Values differ! Investigating...\n")
    diffs <- abs(out_old - out_fast)
    worst_idx <- which(diffs == max_diff, arr.ind = TRUE)[1, ]
    cat(sprintf("    Worst diff at [%d, %d]: old=%.15f  fast=%.15f\n",
                worst_idx[1], worst_idx[2],
                out_old[worst_idx[1], worst_idx[2]],
                out_fast[worst_idx[1], worst_idx[2]]))
  }
}


# ===========================================================================
# TEST 2: PERFORMANCE -- benchmark at increasing sizes
# ===========================================================================
cat("\n=====================================================\n")
cat("TEST 2: Performance benchmark\n")
cat("=====================================================\n\n")

bench_sizes <- c(1000, 10000, 100000)
if (nrow(full_data) > 100000) {
  bench_sizes <- c(bench_sizes, nrow(full_data))
}
n_reps <- 3  # repeats per size (use best time)

results <- data.frame(
  n_rows     = integer(),
  n_cols     = integer(),
  time_old   = numeric(),
  time_fast  = numeric(),
  speedup    = numeric(),
  stringsAsFactors = FALSE
)

for (sz in bench_sizes) {
  n_use <- min(sz, nrow(full_data))
  test_input <- full_data[1:n_use, ]

  cat(sprintf("  Benchmarking n=%d (%d cols)...\n", n_use, ncol(test_input)))

  ## Time the original
  times_old <- numeric(n_reps)
  for (r in 1:n_reps) {
    gc(verbose = FALSE)
    t0 <- proc.time()
    suppressMessages(out_old <- splineData(test_input))
    times_old[r] <- (proc.time() - t0)["elapsed"]
  }
  best_old <- min(times_old)

  ## Time the fast version
  times_fast <- numeric(n_reps)
  for (r in 1:n_reps) {
    gc(verbose = FALSE)
    t0 <- proc.time()
    out_fast <- splineData_fast(test_input)
    times_fast[r] <- (proc.time() - t0)["elapsed"]
  }
  best_fast <- min(times_fast)

  speedup <- best_old / best_fast

  results <- rbind(results, data.frame(
    n_rows    = n_use,
    n_cols    = ncol(test_input),
    time_old  = best_old,
    time_fast = best_fast,
    speedup   = speedup
  ))

  cat(sprintf("    splineData:      %.3f sec (best of %d)\n", best_old, n_reps))
  cat(sprintf("    splineData_fast: %.3f sec (best of %d)\n", best_fast, n_reps))
  cat(sprintf("    Speedup:         %.1fx\n\n", speedup))
}


# ===========================================================================
# TEST 3: Memory profile (single run at largest size)
# ===========================================================================
cat("=====================================================\n")
cat("TEST 3: Memory profile\n")
cat("=====================================================\n\n")

n_use <- min(max(bench_sizes), nrow(full_data))
test_input <- full_data[1:n_use, ]
input_mb <- object.size(test_input) / 1024^2

cat(sprintf("  Input size: %d rows x %d cols = %.1f MB\n", n_use, ncol(test_input), input_mb))

## Original
gc(reset = TRUE, verbose = FALSE)
suppressMessages(out_old <- splineData(test_input))
mem_old <- gc(verbose = FALSE)
peak_old <- sum(mem_old[, 6])  # max used (Mb)
out_old_mb <- object.size(out_old) / 1024^2
rm(out_old)

## Fast
gc(reset = TRUE, verbose = FALSE)
out_fast <- splineData_fast(test_input)
mem_fast <- gc(verbose = FALSE)
peak_fast <- sum(mem_fast[, 6])  # max used (Mb)
out_fast_mb <- object.size(out_fast) / 1024^2
rm(out_fast)

cat(sprintf("  Output size: %.1f MB\n", out_fast_mb))
cat(sprintf("  Peak memory (splineData):      %.1f MB\n", peak_old))
cat(sprintf("  Peak memory (splineData_fast): %.1f MB\n", peak_fast))


# ===========================================================================
# SUMMARY
# ===========================================================================
cat("\n=====================================================\n")
cat("PERFORMANCE SUMMARY\n")
cat("=====================================================\n\n")

cat(sprintf("  %-10s  %10s  %14s  %14s  %10s\n",
            "Rows", "Cols", "Old (sec)", "Fast (sec)", "Speedup"))
cat(paste(rep("-", 66), collapse = ""), "\n")
for (i in seq_len(nrow(results))) {
  r <- results[i, ]
  cat(sprintf("  %-10s  %10d  %14.3f  %14.3f  %9.1fx\n",
              format(r$n_rows, big.mark = ","), r$n_cols,
              r$time_old, r$time_fast, r$speedup))
}
cat(paste(rep("-", 66), collapse = ""), "\n")

## Extrapolate to full dataset size if we didn't test it
if (nrow(full_data) > max(bench_sizes)) {
  ## Use the two largest benchmarks to estimate scaling
  last2 <- tail(results, 2)
  rate_old  <- last2$time_old[2] / last2$n_rows[2]
  rate_fast <- last2$time_fast[2] / last2$n_rows[2]
  est_old   <- rate_old * nrow(full_data)
  est_fast  <- rate_fast * nrow(full_data)
  cat(sprintf("\n  Estimated for full dataset (%s rows):\n",
              format(nrow(full_data), big.mark = ",")))
  cat(sprintf("    splineData:      ~%.1f sec (%.1f min)\n", est_old, est_old / 60))
  cat(sprintf("    splineData_fast: ~%.1f sec (%.1f min)\n", est_fast, est_fast / 60))
  cat(sprintf("    Est. speedup:    ~%.1fx\n", est_old / est_fast))
}

cat("\n=== Test complete ===\n")
