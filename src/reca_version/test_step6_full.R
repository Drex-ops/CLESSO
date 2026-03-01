##############################################################################
##
## test_step6_full.R  —  Full Step 6 pipeline benchmark (single iteration)
##
## Runs all sub-steps of Step 6 once (first climate/weather window combo)
## with per-step performance metrics. No climate/weather loop — just a
## single pass to measure each sub-step's cost.
##
## Sub-steps:
##   6a  Environmental data extraction (chunked, parallel, with progress)
##   6b  Clean NA and sentinel values
##   6c  Decomposition filter (v3)
##   6d  I-spline transformation
##   6e  Fit GDM
##   6f  Diagnostics and save
##
## Usage:
##   source("test_step6_full.R")
##
##############################################################################

cat("=== Step 6 Full Pipeline Benchmark (single iteration) ===\n\n")

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
source(file.path(config$r_dir, "plotting.R"))

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------
load_packages()
rasterOptions(tmpdir = config$raster_tmpdir)

# ---------------------------------------------------------------------------
# Timing collector
# ---------------------------------------------------------------------------
timings <- list()
record_time <- function(step_name, elapsed) {
  timings[[step_name]] <<- elapsed
  cat(sprintf("  >> %s: %.2f sec (%.2f min)\n\n", step_name, elapsed, elapsed / 60))
}

# ---------------------------------------------------------------------------
# Load the saved obsPairs table
# ---------------------------------------------------------------------------
obspairs_file <- file.path(
  config$output_dir,
  paste0("ObsPairsTable_RECA_", config$species_group, "_WindowTestRuns.rds")
)
if (!file.exists(obspairs_file)) stop(paste("ObsPairs file not found:", obspairs_file))
obsPairs_out <- readRDS(obspairs_file)

cat(sprintf("ObsPairs table: %d rows x %d columns\n", nrow(obsPairs_out), ncol(obsPairs_out)))

# ---------------------------------------------------------------------------
# Apply temporal filter (same as run_obsGDM.R step 6 preamble)
# ---------------------------------------------------------------------------
c_yr <- config$c_yrs[1]
w_yr <- config$w_yrs[1]

cat(sprintf("Climate window: %d yrs  |  Weather window: %d yrs\n", c_yr, w_yr))

tst_1 <- as.Date(paste0(obsPairs_out$year1, "-01-01")) >
           (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
tst_2 <- as.Date(paste0(obsPairs_out$year2, "-01-01")) >
           (as.Date("1911-01-01") %m+% years(max(config$c_yrs)))
obsPairs_out <- obsPairs_out[tst_1 & tst_2, ]
ext_data     <- obsPairs_out[, 2:9]

save_prefix <- make_save_prefix(config, c_yr, w_yr)

cat(sprintf("After temporal filter: %d rows\n", nrow(obsPairs_out)))
cat(sprintf("Save prefix: %s\n\n", save_prefix))

# ===========================================================================
# STEP 6a: Environmental data extraction (chunked parallel with progress)
# ===========================================================================
cat("=====================================================\n")
cat("STEP 6a: Environmental data extraction\n")
cat("=====================================================\n")

t0_6a <- proc.time()

## Build env params
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

n_params   <- length(init_params)
n_workers  <- min(n_params, config$cores_to_use)

cat(sprintf("  Env-param sets: %d  |  Workers: %d  |  Chunk size: %d\n",
            n_params, n_workers, config$chunk_size))

cl <- makeCluster(n_workers)
registerDoSNOW(cl)

env_outA <- run_chunked_env(ext_data, init_params, "A", swap_sites = FALSE)

if (config$biAverage) {
  env_outB <- run_chunked_env(ext_data, init_params, "B", swap_sites = TRUE)
  env_out  <- (env_outA + env_outB) / 2
  rm(env_outB)
} else {
  env_out <- env_outA
}
rm(env_outA)

stopCluster(cl)
registerDoSEQ()

## Split into site 1 / site 2
env_out1 <- env_out[, grep("_1$", names(env_out))]
env_out2 <- env_out[, grep("_2$", names(env_out))]

## Make anomalies (weather - climate)
w_cols_1 <- grep(paste0("_", w_yr, "_"), colnames(env_out1))
c_cols_1 <- grep(paste0("_", c_yr, "_"), colnames(env_out1))
if (length(w_cols_1) == length(c_cols_1) && length(w_cols_1) > 0) {
  env_anom_out1 <- env_out1[, w_cols_1] - env_out1[, c_cols_1]
  colnames(env_anom_out1) <- gsub("191101-201712", "anom", colnames(env_anom_out1))
} else {
  env_anom_out1 <- NULL
}

w_cols_2 <- grep(paste0("_", w_yr, "_"), colnames(env_out2))
c_cols_2 <- grep(paste0("_", c_yr, "_"), colnames(env_out2))
if (length(w_cols_2) == length(c_cols_2) && length(w_cols_2) > 0) {
  env_anom_out2 <- env_out2[, w_cols_2] - env_out2[, c_cols_2]
  colnames(env_anom_out2) <- gsub("191101-201712", "anom", colnames(env_anom_out2))
} else {
  env_anom_out2 <- NULL
}

## Extract substrate
pnt1      <- SpatialPoints(data.frame(ext_data$Lon1, ext_data$Lat1))
pnt2      <- SpatialPoints(data.frame(ext_data$Lon2, ext_data$Lat2))
subs_brk  <- brick(config$substrate_raster)
env1_subs <- extract(subs_brk, pnt1)
env2_subs <- extract(subs_brk, pnt2)
colnames(env1_subs) <- paste0(colnames(env1_subs), "_1")
colnames(env2_subs) <- paste0(colnames(env2_subs), "_2")

## Temperature range
tnn_c1 <- grep(paste0("min.*_", c_yr, ".*TNn.*_1"), colnames(env_out1), ignore.case = TRUE)
txx_c1 <- grep(paste0("max.*_", c_yr, ".*TXx.*_1"), colnames(env_out1), ignore.case = TRUE)
tnn_w1 <- grep(paste0("min.*_", w_yr, ".*TNn.*_1"), colnames(env_out1), ignore.case = TRUE)
txx_w1 <- grep(paste0("max.*_", w_yr, ".*TXx.*_1"), colnames(env_out1), ignore.case = TRUE)

if (length(tnn_c1) > 0 && length(txx_c1) > 0) {
  Trng_1 <- abs(env_out1[, c(tnn_c1[1], tnn_w1[1])] - env_out1[, c(txx_c1[1], txx_w1[1])])
  names(Trng_1) <- c("Trng_15_191101-201712_1", "Trng_1_191101-201712_1")
} else {
  Trng_1 <- NULL
}

tnn_c2 <- grep(paste0("min.*_", c_yr, ".*TNn.*_2"), colnames(env_out2), ignore.case = TRUE)
txx_c2 <- grep(paste0("max.*_", c_yr, ".*TXx.*_2"), colnames(env_out2), ignore.case = TRUE)
tnn_w2 <- grep(paste0("min.*_", w_yr, ".*TNn.*_2"), colnames(env_out2), ignore.case = TRUE)
txx_w2 <- grep(paste0("max.*_", w_yr, ".*TXx.*_2"), colnames(env_out2), ignore.case = TRUE)

if (length(tnn_c2) > 0 && length(txx_c2) > 0) {
  Trng_2 <- abs(env_out2[, c(tnn_c2[1], tnn_w2[1])] - env_out2[, c(txx_c2[1], txx_w2[1])])
  names(Trng_2) <- c("Trng_15_191101-201712_2", "Trng_1_191101-201712_2")
} else {
  Trng_2 <- NULL
}

## Combine everything
parts <- list(obsPairs_out, env_out1, env1_subs, Trng_1, env_anom_out1,
              env_out2, env2_subs, Trng_2, env_anom_out2)
parts <- parts[!sapply(parts, is.null)]
obsPairs_out <- do.call(cbind, parts)

rm(env_out, env_out1, env_out2, env1_subs, env2_subs)
gc(verbose = FALSE)

elapsed_6a <- (proc.time() - t0_6a)["elapsed"]
cat(sprintf("  obsPairs_out after 6a: %d rows x %d cols\n", nrow(obsPairs_out), ncol(obsPairs_out)))
record_time("6a_env_extraction", elapsed_6a)

# ===========================================================================
# STEP 6b: Clean NA and sentinel values
# ===========================================================================
cat("=====================================================\n")
cat("STEP 6b: Clean NA and sentinel values\n")
cat("=====================================================\n")

t0_6b <- proc.time()

env_cols  <- 23:ncol(obsPairs_out)
test_na   <- is.na(rowSums(obsPairs_out[, env_cols]))
n_before  <- nrow(obsPairs_out)
obsPairs_out <- obsPairs_out[!test_na, ]
cat(sprintf("  Removed %d rows with NA (%.1f%%)\n",
            n_before - nrow(obsPairs_out),
            100 * (n_before - nrow(obsPairs_out)) / n_before))

## Remove -9999 sentinel values
sentinel_test <- rep(0, nrow(obsPairs_out))
for (col in env_cols) {
  sentinel_test <- sentinel_test + (obsPairs_out[, col] == -9999)
}
n_before2 <- nrow(obsPairs_out)
obsPairs_out <- obsPairs_out[sentinel_test == 0, ]
cat(sprintf("  Removed %d rows with -9999 sentinels\n", n_before2 - nrow(obsPairs_out)))
cat(sprintf("  After cleaning: %d pairs\n", nrow(obsPairs_out)))

elapsed_6b <- (proc.time() - t0_6b)["elapsed"]
record_time("6b_clean_na_sentinel", elapsed_6b)

# ===========================================================================
# STEP 6c: Decomposition filter
# ===========================================================================
cat("=====================================================\n")
cat("STEP 6c: Decomposition filter\n")
cat("=====================================================\n")

t0_6c <- proc.time()

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
  cat(sprintf("  Decomposition '%s': keeping all %d pairs\n",
              config$decomposition, nrow(obsPairs_out)))
}

elapsed_6c <- (proc.time() - t0_6c)["elapsed"]
record_time("6c_decomposition_filter", elapsed_6c)

# ===========================================================================
# STEP 6d: I-spline transformation
# ===========================================================================
cat("=====================================================\n")
cat("STEP 6d: I-spline transformation\n")
cat("=====================================================\n")

t0_6d <- proc.time()

toSpline <- obsPairs_out[, env_cols]
splined  <- splineData_fast(toSpline)

if (config$decomposition == "v3") {
  splined_new <- splined[keep, ]
} else {
  splined_new <- splined
}

cat(sprintf("  Splined data: %d rows x %d cols\n", nrow(splined_new), ncol(splined_new)))

elapsed_6d <- (proc.time() - t0_6d)["elapsed"]
record_time("6d_ispline_transform", elapsed_6d)

# ===========================================================================
# STEP 6e: Fit GDM
# ===========================================================================
cat("=====================================================\n")
cat("STEP 6e: Fit GDM\n")
cat("=====================================================\n")

t0_6e <- proc.time()

match_response <- if (config$decomposition == "v3") {
  obsPairs_out$Match[keep]
} else {
  obsPairs_out$Match
}

mod_ready <- cbind(Match = match_response, as.data.frame(splined_new))
colnames(mod_ready) <- gsub("191101-201712_", "", colnames(mod_ready))

f1      <- paste(colnames(mod_ready)[-1], collapse = "+")
formula <- as.formula(paste(colnames(mod_ready)[1], "~", f1, sep = ""))
obsGDM_1 <- fitGDM(formula = formula, data = mod_ready)

D2 <- (obsGDM_1$null.deviance - obsGDM_1$deviance) / obsGDM_1$null.deviance
cat(sprintf("  Deviance explained: %.4f\n", D2))

elapsed_6e <- (proc.time() - t0_6e)["elapsed"]
record_time("6e_fit_gdm", elapsed_6e)

# ===========================================================================
# STEP 6f: Diagnostics and save
# ===========================================================================
cat("=====================================================\n")
cat("STEP 6f: Diagnostics and save\n")
cat("=====================================================\n")

t0_6f <- proc.time()

out_prefix <- file.path(config$output_dir, save_prefix)

## Deviance R²
gdm_dev <- RsqGLM(obs = obsGDM_1$y, pred = fitted(obsGDM_1))
save(gdm_dev, file = paste0(out_prefix, "DevianceCalcs.RData"))
cat(sprintf("  Nagelkerke R²: %.4f\n", gdm_dev$Nagelkerke))
save(D2, file = paste0(out_prefix, "D2_deviance.RData"))

## Coefficients
coefs <- coef(obsGDM_1)
save(coefs, file = paste0(out_prefix, "coefficients.RData"))

## Build fit object
fit <- list()
fit$intercept    <- coef(obsGDM_1)[1]
fit$sample       <- nrow(mod_ready)
fit$predictors   <- gsub("191101-201712_", "",
                         gsub("_spl1", "", colnames(splined)[grep("_spl1", colnames(splined))]))
fit$coefficients <- coef(obsGDM_1)[-1]
fit$coefficients[is.na(fit$coefficients)] <- 0

nc  <- ncol(toSpline)
nc2 <- nc / 2
X1  <- toSpline[, 1:nc2]
X2  <- toSpline[, (nc2 + 1):nc]
nms <- names(X1); names(X2) <- nms
XX  <- rbind(X1, X2)
fit$quantiles  <- unlist(lapply(1:ncol(XX), function(x) quantile(XX[, x], c(0, 0.5, 1))))
fit$splines    <- rep(3, ncol(XX))
fit$predicted  <- fitted(obsGDM_1)
fit$ecological <- obsGDM_1$linear.predictors
save(fit, file = paste0(out_prefix, "fittedGDM.RData"))

## Diagnostic plots
## w is needed for obs.gdm.plot — estimate a default if not available
if (!exists("w")) {
  w <- 1  # placeholder; in full pipeline this comes from step 2
  cat("  Note: 'w' not available from step 2, using placeholder w=1 for plot\n")
}

tiff(paste0(out_prefix, "GDM-ObsDiag.tif"),
     height = 6, width = 6, units = "in", res = 200, compression = "lzw")
obs.gdm.plot(obsGDM_1, save_prefix, w, Is = fit$intercept)
dev.off()

pdf(paste0(out_prefix, "GDM-gdmDiag.pdf"))
gdm.spline.plot(fit)
dev.off()

cat(sprintf("  Saved outputs with prefix: %s\n", save_prefix))

## Save env table
env_file <- file.path(config$output_dir, paste0(save_prefix, "ObsEnvTable.RData"))
save(obsPairs_out, file = env_file)
cat(sprintf("  Saved env table: %s\n", basename(env_file)))

elapsed_6f <- (proc.time() - t0_6f)["elapsed"]
record_time("6f_diagnostics_save", elapsed_6f)

# ===========================================================================
# PERFORMANCE SUMMARY
# ===========================================================================
total_elapsed <- sum(unlist(timings))

cat("\n=====================================================\n")
cat("STEP 6 PERFORMANCE SUMMARY\n")
cat("=====================================================\n\n")

cat(sprintf("  %-30s  %10s  %10s  %8s\n", "Step", "Time (s)", "Time (min)", "% Total"))
cat(paste(rep("-", 68), collapse = ""), "\n")
for (nm in names(timings)) {
  t <- timings[[nm]]
  cat(sprintf("  %-30s  %10.2f  %10.2f  %7.1f%%\n",
              nm, t, t / 60, 100 * t / total_elapsed))
}
cat(paste(rep("-", 68), collapse = ""), "\n")
cat(sprintf("  %-30s  %10.2f  %10.2f  %7.1f%%\n",
            "TOTAL", total_elapsed, total_elapsed / 60, 100))

cat(sprintf("\n  Rows processed            : %d\n", nrow(obsPairs_out)))
cat(sprintf("  Final obsPairs dimensions : %d x %d\n", nrow(obsPairs_out), ncol(obsPairs_out)))
cat(sprintf("  Deviance explained (D²)   : %.4f\n", D2))
cat(sprintf("  Nagelkerke R²             : %.4f\n", gdm_dev$Nagelkerke))

cat("\n=== Step 6 benchmark complete ===\n")
