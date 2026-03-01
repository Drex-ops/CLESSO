##############################################################################
##
## test_gen_windows.R  —  Performance benchmarks for Step 6a env extraction
##
## Tests:
##   1) Sequential (non-parallel) gen_windows for 100 / 1,000 / 10,000 rows
##   2) Parallel foreach loop with chunk sizes of 100 and 1,000 rows
##   3) Reports obsPairs table dimensions
##   4) Extrapolates estimated time for 1M and 2M records
##
## Usage:
##   source("test_gen_windows.R")
##
##############################################################################

cat("=== Step 6a Performance Benchmark ===\n\n")

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

cat(sprintf("ObsPairs table dimensions: %d rows x %d columns\n\n",
            nrow(obsPairs_out), ncol(obsPairs_out)))

# ---------------------------------------------------------------------------
# Apply temporal filter (same as run_obsGDM.R step 6)
# ---------------------------------------------------------------------------
tst_1 <- as.Date(paste0(obsPairs_out$year1, "-01-01")) >
           (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
tst_2 <- as.Date(paste0(obsPairs_out$year2, "-01-01")) >
           (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
obsPairs_out <- obsPairs_out[tst_1 & tst_2, ]
ext_data_full <- obsPairs_out[, 2:9]

cat(sprintf("After temporal filter: %d rows available for extraction\n\n",
            nrow(ext_data_full)))

# ---------------------------------------------------------------------------
# Pick a single env-param set for benchmarking (first climate window)
# ---------------------------------------------------------------------------
c_yr <- config$c_yrs[1]
w_yr <- config$w_yrs[1]

## Build env params (same logic as run_obsGDM.R)
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

## Use the first env-param for the sequential tests
test_param <- init_params[[1]]

cat(sprintf("Benchmark env-param: vars=%s  mstat=%s  cstat=%s  window=%d\n",
            paste(test_param$variables, collapse = ","),
            test_param$mstat, test_param$cstat, test_param$window))
cat(sprintf("Total env-param sets (for parallel test): %d\n", length(init_params)))
cat(sprintf("Cores available: %d\n\n", config$cores_to_use))

# ===========================================================================
# Helper: run a single sequential gen_windows call and return timing
# ===========================================================================
run_sequential_test <- function(n_rows, ext_data_full, param) {
  ext_sub <- ext_data_full[1:min(n_rows, nrow(ext_data_full)), ]
  actual_n <- nrow(ext_sub)

  cat(sprintf("  [SEQ] n=%d ... ", actual_n))
  gc(verbose = FALSE)
  t0 <- proc.time()

  out <- gen_windows(
    pairs        = ext_sub,
    variables    = param$variables,
    mstat        = param$mstat,
    cstat        = param$cstat,
    window       = param$window,
    npy_src      = config$npy_src,
    start_year   = config$geonpy_start_year,
    python_exe   = config$python_exe,
    pyper_script = config$pyper_script,
    feather_tmpdir = config$feather_tmpdir
  )

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("%.2f sec  (%.4f sec/row)\n", elapsed, elapsed / actual_n))

  list(n = actual_n, elapsed = elapsed, per_row = elapsed / actual_n,
       out_dim = dim(out))
}

# ===========================================================================
# Helper: run the full parallel foreach loop on a subset of rows
# ===========================================================================
run_parallel_test <- function(n_rows, ext_data_full, init_params) {
  ext_sub <- ext_data_full[1:min(n_rows, nrow(ext_data_full)), ]
  actual_n <- nrow(ext_sub)

  cat(sprintf("  [PAR] n=%d, %d env-params, %d workers ... ",
              actual_n, length(init_params),
              min(length(init_params), config$cores_to_use)))
  gc(verbose = FALSE)

  n_workers <- min(length(init_params), config$cores_to_use)
  cl <- makeCluster(n_workers)
  registerDoSNOW(cl)

  t0 <- proc.time()

  env_outA <- foreach(x = 1:length(init_params), .combine = "cbind",
                      .packages = "arrow",
                      .export = c("gen_windows", "config")) %dopar% {
    out <- gen_windows(
      pairs        = ext_sub,
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

  elapsed <- (proc.time() - t0)["elapsed"]
  stopCluster(cl)
  registerDoSEQ()

  cat(sprintf("%.2f sec  (%.4f sec/row)\n", elapsed, elapsed / actual_n))

  list(n = actual_n, elapsed = elapsed, per_row = elapsed / actual_n,
       out_dim = dim(env_outA))
}

# ===========================================================================
# TEST 1: Sequential extraction — 100, 1000, 10000 records
# ===========================================================================
cat("=====================================================\n")
cat("TEST 1: Sequential gen_windows (single env-param)\n")
cat("=====================================================\n")

seq_sizes   <- c(100, 1000, 10000)
seq_results <- list()

for (n in seq_sizes) {
  if (n > nrow(ext_data_full)) {
    cat(sprintf("  [SEQ] Skipping n=%d (only %d rows available)\n", n, nrow(ext_data_full)))
    next
  }
  res <- run_sequential_test(n, ext_data_full, test_param)
  seq_results[[as.character(n)]] <- res
}

# ===========================================================================
# TEST 2: Sequential loop over ALL env-params (fair comparison to parallel)
# ===========================================================================
cat("\n=====================================================\n")
cat("TEST 2: Sequential loop (all env-params, one at a time)\n")
cat("=====================================================\n")

seq_all_sizes   <- c(100, 1000, 10000)
seq_all_results <- list()

for (n in seq_all_sizes) {
  if (n > nrow(ext_data_full)) {
    cat(sprintf("  [SEQ-ALL] Skipping n=%d (only %d rows available)\n", n, nrow(ext_data_full)))
    next
  }
  ext_sub  <- ext_data_full[1:min(n, nrow(ext_data_full)), ]
  actual_n <- nrow(ext_sub)
  cat(sprintf("  [SEQ-ALL] n=%d, %d env-params ... ", actual_n, length(init_params)))
  gc(verbose = FALSE)
  t0 <- proc.time()

  parts <- list()
  for (x in seq_along(init_params)) {
    out <- gen_windows(
      pairs        = ext_sub,
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
    parts[[x]] <- out[, 9:ncol(out)]
  }
  env_outA <- do.call(cbind, parts)

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("%.2f sec  (%.4f sec/row)\n", elapsed, elapsed / actual_n))
  seq_all_results[[as.character(n)]] <- list(
    n = actual_n, elapsed = elapsed, per_row = elapsed / actual_n,
    out_dim = dim(env_outA)
  )
}

# ===========================================================================
# TEST 3: Parallel foreach loop — 100, 1000 and 10000 records per chunk
# ===========================================================================
cat("\n=====================================================\n")
cat("TEST 3: Parallel foreach (all env-params)\n")
cat("=====================================================\n")

par_sizes   <- c(100, 1000, 10000)
par_results <- list()

for (n in par_sizes) {
  if (n > nrow(ext_data_full)) {
    cat(sprintf("  [PAR] Skipping n=%d (only %d rows available)\n", n, nrow(ext_data_full)))
    next
  }
  res <- run_parallel_test(n, ext_data_full, init_params)
  par_results[[as.character(n)]] <- res
}

# ===========================================================================
# SUMMARY: Performance metrics & extrapolation
# ===========================================================================
cat("\n=====================================================\n")
cat("PERFORMANCE SUMMARY\n")
cat("=====================================================\n\n")

cat("--- ObsPairs table ---\n")
cat(sprintf("  Dimensions (after temporal filter): %d rows x %d cols\n\n",
            nrow(ext_data_full), ncol(ext_data_full)))

divider <- paste(rep("-", 78), collapse = "")

## Test 1: Sequential single env-param
cat("--- Test 1: Sequential (single env-param) ---\n")
cat(sprintf("  %-10s  %10s  %12s  %14s  %14s\n",
            "N rows", "Time (s)", "sec/row", "Est 1M (min)", "Est 2M (min)"))
cat(divider, "\n")
for (nm in names(seq_results)) {
  r <- seq_results[[nm]]
  cat(sprintf("  %-10d  %10.2f  %12.6f  %14.1f  %14.1f\n",
              r$n, r$elapsed, r$per_row,
              r$per_row * 1e6 / 60, r$per_row * 2e6 / 60))
}

## Test 2: Sequential ALL env-params
cat("\n--- Test 2: Sequential loop (ALL 10 env-params, one-by-one) ---\n")
cat(sprintf("  %-10s  %10s  %12s  %14s  %14s\n",
            "N rows", "Time (s)", "sec/row", "Est 1M (min)", "Est 2M (min)"))
cat(divider, "\n")
for (nm in names(seq_all_results)) {
  r <- seq_all_results[[nm]]
  cat(sprintf("  %-10d  %10.2f  %12.6f  %14.1f  %14.1f\n",
              r$n, r$elapsed, r$per_row,
              r$per_row * 1e6 / 60, r$per_row * 2e6 / 60))
}

## Test 3: Parallel ALL env-params
cat("\n--- Test 3: Parallel foreach (ALL 10 env-params) ---\n")
cat(sprintf("  %-10s  %10s  %12s  %14s  %14s\n",
            "N rows", "Time (s)", "sec/row", "Est 1M (min)", "Est 2M (min)"))
cat(divider, "\n")
for (nm in names(par_results)) {
  r <- par_results[[nm]]
  cat(sprintf("  %-10d  %10.2f  %12.6f  %14.1f  %14.1f\n",
              r$n, r$elapsed, r$per_row,
              r$per_row * 1e6 / 60, r$per_row * 2e6 / 60))
}

## Speedup comparison (Test 2 vs Test 3)
cat("\n--- Speedup: Sequential-all vs Parallel-all ---\n")
common_sizes <- intersect(names(seq_all_results), names(par_results))
if (length(common_sizes) > 0) {
  cat(sprintf("  %-10s  %12s  %12s  %10s\n", "N rows", "Seq (s)", "Par (s)", "Speedup"))
  cat(divider, "\n")
  for (nm in common_sizes) {
    s <- seq_all_results[[nm]]
    p <- par_results[[nm]]
    cat(sprintf("  %-10d  %12.2f  %12.2f  %10.2fx\n",
                s$n, s$elapsed, p$elapsed, s$elapsed / p$elapsed))
  }
}

## Output column dimensions
cat("\n--- Output dimensions ---\n")
for (nm in names(seq_results)) {
  r <- seq_results[[nm]]
  cat(sprintf("  SEQ-1   n=%-6s -> output %d x %d\n", nm, r$out_dim[1], r$out_dim[2]))
}
for (nm in names(seq_all_results)) {
  r <- seq_all_results[[nm]]
  cat(sprintf("  SEQ-ALL n=%-6s -> output %d x %d\n", nm, r$out_dim[1], r$out_dim[2]))
}
for (nm in names(par_results)) {
  r <- par_results[[nm]]
  cat(sprintf("  PAR     n=%-6s -> output %d x %d\n", nm, r$out_dim[1], r$out_dim[2]))
}

cat("\n=== Benchmark complete ===\n")
