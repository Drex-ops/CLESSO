##############################################################################
##
## run_clesso_beta_fixAlpha.R — Beta-only (turnover) pipeline with fixed alpha
##
## Two-stage workflow:
##   Stage 1: Run run_clesso_alpha.R to estimate alpha per site
##   Stage 2: This script takes those fixed alpha values and estimates
##            turnover (beta) from between-site pairs only
##
## Pipeline:
##   1. Load data → siteAggregator
##   2. Date filter → format for CLESSO sampler
##   3. Load fixed alpha estimates (from prior clesso_alpha run)
##   4. Sample between-site observation pairs only
##   5. Extract pairwise environmental covariates (turnover X matrix)
##   6. Prepare beta-only TMB model data
##   7. Compile and fit the TMB model (clesso_beta_fixAlpha.cpp)
##   8. Diagnostics and output
##
## Usage:
##   1. First run: source("run_clesso_alpha.R")
##   2. Edit clesso_config.R if needed
##   3. source("run_clesso_beta_fixAlpha.R")
##
##############################################################################

cat("=== CLESSO Beta-Only (Fixed Alpha) Pipeline ===\n")
cat("Loading configuration and modules...\n")

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

## Locate config
config_path <- file.path(this_dir, "clesso_config.R")
if (!file.exists(config_path)) {
  config_path <- file.path(this_dir, "src", "clesso_v2", "clesso_config.R")
}
source(config_path)

## Source shared modules
source(file.path(clesso_config$r_dir, "utils.R"))
source(file.path(clesso_config$r_dir, "gdm_functions.R"))
source(file.path(clesso_config$r_dir, "site_aggregator.R"))

## Source CLESSO v2 modules
source(file.path(clesso_config$clesso_dir, "clesso_sampler_optimised.R"))
source(file.path(clesso_config$clesso_dir, "clesso_prepare_data.R"))
source(file.path(clesso_config$clesso_dir, "clesso_predict.R"))

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------
load_packages()
rasterOptions(tmpdir = clesso_config$raster_tmpdir)


# ===========================================================================
# STEP 1: Load and aggregate observations
# ===========================================================================
cat("\n--- Step 1: Load and aggregate observations ---\n")

obs_file <- file.path(clesso_config$data_dir, clesso_config$obs_csv)
if (!file.exists(obs_file)) stop(paste("Observation file not found:", obs_file))
dat <- read.csv(obs_file)
cat(sprintf("  Loaded %d records from %s\n", nrow(dat), clesso_config$obs_csv))

## Reference raster for grid
ras_sp <- raster(clesso_config$reference_raster)
res_deg <- res(ras_sp)[1]
box     <- extent(ras_sp)

## Aggregate to grid cells
datRED <- siteAggregator(dat, res_deg, box)

## Clean sites where reference raster gives NAs
test   <- is.na(extract(ras_sp, datRED[, c("lonID", "latID")]))
datRED <- datRED[!test, ]
cat(sprintf("  Aggregated data: %d site-species-date records\n", nrow(datRED)))


# ===========================================================================
# STEP 2: Date filter and format for CLESSO sampler
# ===========================================================================
cat("\n--- Step 2: Date filter and format ---\n")

datRED <- datRED[datRED$eventDate != "", ]
date_test <- as.Date(datRED$eventDate) >= clesso_config$min_date
datRED    <- datRED[date_test, ]
date_test <- as.Date(datRED$eventDate) < clesso_config$max_date
datRED    <- datRED[date_test, ]
datRED    <- droplevels(datRED)
cat(sprintf("  After date filter: %d records\n", nrow(datRED)))

## Convert to CLESSO format
obs_dt <- clesso_format_aggregated_data(datRED)


# ===========================================================================
# STEP 3: Load fixed alpha estimates from prior clesso_alpha run
# ===========================================================================
cat("\n--- Step 3: Load fixed alpha estimates ---\n")

## Look for alpha results file
alpha_results_file <- file.path(clesso_config$output_dir,
  paste0("clesso_alpha_results_", clesso_config$species_group, ".rds"))

## Allow override via environment variable
alpha_results_file <- Sys.getenv("CLESSO_ALPHA_RESULTS",
                                  unset = alpha_results_file)

if (!file.exists(alpha_results_file)) {
  stop(paste("Alpha results file not found:", alpha_results_file,
             "\nRun run_clesso_alpha.R first, or set CLESSO_ALPHA_RESULTS env var."))
}

alpha_results <- readRDS(alpha_results_file)
alpha_est_dt  <- alpha_results$alpha_estimates
cat(sprintf("  Loaded alpha estimates for %d sites from:\n    %s\n",
            nrow(alpha_est_dt), alpha_results_file))
cat(sprintf("  Alpha: mean = %.1f, range = [%.1f, %.1f]\n",
            mean(alpha_est_dt$alpha_est),
            min(alpha_est_dt$alpha_est),
            max(alpha_est_dt$alpha_est)))


# ===========================================================================
# STEP 4: Sample BETWEEN-SITE observation pairs only
# ===========================================================================
cat("\n--- Step 4: Between-site observation-pair sampling ---\n")

between_pairs <- clesso_sample_between_pairs(
  obs_dt      = obs_dt,
  n_pairs     = clesso_config$n_between,
  match_ratio = clesso_config$between_match_ratio,
  seed        = if (!is.null(clesso_config$seed)) clesso_config$seed + 1 else NULL,
  species_thresh = clesso_config$species_threshold,
  cores       = clesso_config$cores
)

## Add expected columns
between_pairs[, is_within := 0L]

## Compute pair weights (match vs mismatch balancing)
n_match <- sum(between_pairs$y == 0)
n_miss  <- sum(between_pairs$y == 1)
if (n_match > 0 && n_miss > 0) {
  between_pairs[y == 0, w := 0.5 / n_match]
  between_pairs[y == 1, w := 0.5 / n_miss]
  cat(sprintf("  Pair weights: match w=%.6f (%d pairs), mismatch w=%.6f (%d pairs)\n",
              0.5 / n_match, n_match, 0.5 / n_miss, n_miss))
} else {
  between_pairs[, w := 1.0 / .N]
  cat("  Warning: only one class present. Using uniform weights.\n")
}

pairs_dt <- between_pairs

## Save pairs
pairs_file <- file.path(clesso_config$output_dir,
  paste0("clesso_beta_fixAlpha_pairs_", clesso_config$species_group, ".rds"))
saveRDS(pairs_dt, file = pairs_file)
cat(sprintf("  Between-site pairs saved to %s\n", pairs_file))


# ===========================================================================
# STEP 5: Build site table and map alpha values
# ===========================================================================
cat("\n--- Step 5: Build site table and map fixed alpha ---\n")

## Collect all unique sites from the between-site pairs
unique_sites_i <- unique(pairs_dt[, .(site_id = site_i, lon = lon_i, lat = lat_i)])
unique_sites_j <- unique(pairs_dt[, .(site_id = site_j, lon = lon_j, lat = lat_j)])
setnames(unique_sites_j, c("site_id", "lon", "lat"))
unique_sites <- unique(rbind(unique_sites_i, unique_sites_j), by = "site_id")

## Assign contiguous 0-based site indices
unique_sites[, site_index := .I - 1L]
setkey(unique_sites, site_id)

## Map fixed alpha values to the site table
unique_sites[alpha_est_dt, alpha_fixed := i.alpha_est, on = .(site_id)]

## Check for sites without alpha estimates
n_missing_alpha <- sum(is.na(unique_sites$alpha_fixed))
if (n_missing_alpha > 0) {
  cat(sprintf("  WARNING: %d sites have no alpha estimate. Using median alpha as fallback.\n",
              n_missing_alpha))
  median_alpha <- median(alpha_est_dt$alpha_est, na.rm = TRUE)
  unique_sites[is.na(alpha_fixed), alpha_fixed := median_alpha]
} else {
  cat(sprintf("  All %d sites matched to alpha estimates.\n", nrow(unique_sites)))
}

site_table <- unique_sites
nSites <- nrow(site_table)

cat(sprintf("  Site table: %d sites, alpha range = [%.1f, %.1f]\n",
            nSites,
            min(site_table$alpha_fixed),
            max(site_table$alpha_fixed)))


# ===========================================================================
# STEP 6: Extract pairwise environmental covariates (turnover X matrix)
# ===========================================================================
cat("\n--- Step 6: Extract environmental covariates for turnover ---\n")

## Map pair site IDs to 0-based indices
pairs_dt[site_table, site_i_idx := i.site_index, on = .(site_i = site_id)]
pairs_dt[site_table, site_j_idx := i.site_index, on = .(site_j = site_id)]

stopifnot(!anyNA(pairs_dt$site_i_idx), !anyNA(pairs_dt$site_j_idx))

## Substrate extraction for env_site_table
subs_raster <- tryCatch(
  raster::brick(clesso_config$substrate_raster),
  error = function(e) {
    warning(paste("Could not load substrate raster:", e$message))
    NULL
  }
)

env_site_table <- NULL
if (!is.null(subs_raster)) {
  subs_vals <- raster::extract(subs_raster,
                                site_table[, .(lon, lat)])
  if (is.matrix(subs_vals)) {
    subs_dt <- as.data.table(subs_vals)
    names(subs_dt) <- paste0("subs_", seq_len(ncol(subs_dt)))
  } else {
    subs_dt <- data.table(subs_1 = subs_vals)
  }
  env_site_table <- cbind(site_table[, .(site_id)], subs_dt)
  cat(sprintf("  Env site table: %d sites x %d env variables\n",
              nrow(env_site_table), ncol(subs_dt)))
} else {
  cat("  No substrate data. Turnover model uses geographic distance only.\n")
}

## Build the turnover X matrix (I-spline transformed env distances)
turnover_info <- clesso_build_turnover_X(
  pairs_dt       = pairs_dt,
  env_site_table = env_site_table,
  geo_distance   = clesso_config$geo_distance,
  n_splines      = clesso_config$n_splines
)
X <- turnover_info$X
Kbeta <- ncol(X)

cat(sprintf("  X matrix: %d pairs x %d turnover covariates\n", nrow(X), Kbeta))


# ===========================================================================
# STEP 7: Assemble TMB data and fit the model
# ===========================================================================
cat("\n--- Step 7: Compile and fit beta-only TMB model ---\n")

## Alpha vector ordered by site_index
alpha_vec <- site_table$alpha_fixed[order(site_table$site_index)]

## TMB data list
data_list <- list(
  y           = as.numeric(pairs_dt$y),
  site_i      = as.integer(pairs_dt$site_i_idx),
  site_j      = as.integer(pairs_dt$site_j_idx),
  X           = X,
  w           = as.numeric(pairs_dt$w),
  alpha_fixed = as.numeric(alpha_vec)
)

## Initial parameter values
parameters <- list(
  eta0_raw = 0,
  beta_raw = rep(log(0.01), Kbeta)
)

## Store model data for later
model_data <- list(
  data_list     = data_list,
  parameters    = parameters,
  site_table    = site_table,
  turnover_info = turnover_info,
  alpha_source  = alpha_results_file,
  pairs_dt      = pairs_dt
)

## Save model data
model_data_file <- file.path(clesso_config$output_dir,
  paste0("clesso_beta_fixAlpha_model_data_", clesso_config$species_group, ".rds"))
saveRDS(model_data, file = model_data_file)
cat(sprintf("  Model data saved to %s\n", model_data_file))

cat(sprintf("\n--- Beta-only model data summary ---\n"))
cat(sprintf("  Between-site pairs: %d (%d match, %d mismatch)\n",
            nrow(pairs_dt), sum(pairs_dt$y == 0), sum(pairs_dt$y == 1)))
cat(sprintf("  Sites:  %d\n", nSites))
cat(sprintf("  X dims: %d x %d (turnover covariates)\n", nrow(X), Kbeta))
cat(sprintf("  Fixed alpha: mean = %.1f, range = [%.1f, %.1f]\n",
            mean(alpha_vec), min(alpha_vec), max(alpha_vec)))

## --- Compile TMB ---
library(TMB)

cpp_file     <- file.path(clesso_config$clesso_dir, "clesso_beta_fixAlpha.cpp")
cpp_basename <- tools::file_path_sans_ext(basename(cpp_file))

dll_path <- file.path(clesso_config$clesso_dir,
                       paste0(cpp_basename, .Platform$dynlib.ext))
if (!file.exists(dll_path)) {
  cat("  Compiling beta-only TMB model...\n")
  compile(cpp_file)
} else {
  cat("  Beta-only TMB model already compiled.\n")
}

## Load the DLL
dyn.load(dynlib(file.path(clesso_config$clesso_dir, cpp_basename)))

## Build TMB objective function
## No random effects in this model — purely fixed-effect optimisation
obj <- MakeADFun(
  data       = data_list,
  parameters = parameters,
  DLL        = cpp_basename,
  silent     = TRUE
)

## Optimise
cat("  Fitting beta-only model...\n")
t_fit <- proc.time()
fit <- nlminb(
  start     = obj$par,
  objective = obj$fn,
  gradient  = obj$gr,
  control   = list(
    eval.max = clesso_config$tmb_eval_max,
    iter.max = clesso_config$tmb_iter_max
  )
)
t_fit <- proc.time() - t_fit

cat(sprintf("  Convergence: %d (message: %s)\n", fit$convergence, fit$message))
cat(sprintf("  Final objective: %.4f\n", fit$objective))
cat(sprintf("  Fitting time: %.1f seconds\n", t_fit["elapsed"]))

## Standard errors
rep <- sdreport(obj)


# ===========================================================================
# STEP 8: Extract results and diagnostics
# ===========================================================================
cat("\n--- Step 8: Results and diagnostics ---\n")

est <- summary(rep, "report")

## Turnover intercept
eta0_rows <- grep("^eta0$", rownames(est))
if (length(eta0_rows) > 0) {
  cat(sprintf("  Turnover intercept (eta0): %.4f (SE: %.4f)\n",
              est[eta0_rows, "Estimate"], est[eta0_rows, "Std. Error"]))
}

## Turnover coefficients
beta_rows <- grep("^beta$", rownames(est))
if (length(beta_rows) > 0) {
  cat("  Turnover coefficients (beta):\n")
  beta_est <- est[beta_rows, , drop = FALSE]
  if (!is.null(turnover_info$col_names)) {
    rownames(beta_est) <- turnover_info$col_names
  }
  print(beta_est)
}

## Predicted similarity summary
S_rows <- grep("^S_pred$", rownames(est))
if (length(S_rows) > 0) {
  S_est <- est[S_rows, "Estimate"]
  cat(sprintf("  Predicted similarity (S): mean = %.3f, range = [%.3f, %.3f]\n",
              mean(S_est), min(S_est), max(S_est)))
}

## Save full results
results <- list(
  fit            = fit,
  sdreport       = rep,
  beta_est       = if (length(beta_rows) > 0) est[beta_rows, , drop = FALSE] else NULL,
  eta0_est       = if (length(eta0_rows) > 0) est[eta0_rows, , drop = FALSE] else NULL,
  model_data     = model_data,
  config         = clesso_config,
  alpha_source   = alpha_results_file
)

results_file <- file.path(clesso_config$output_dir,
  paste0("clesso_beta_fixAlpha_results_", clesso_config$species_group, ".rds"))
saveRDS(results, file = results_file)

cat(sprintf("\n=== CLESSO beta-only (fixed alpha) pipeline complete ===\n"))
cat(sprintf("  Results saved to %s\n", results_file))


# ===========================================================================
# STEP 9: Predict turnover for training pairs (demonstration)
# ===========================================================================
cat("\n--- Step 9: Turnover prediction (training set) ---\n")

## Extract fitted beta and eta0 on natural scale
beta_fitted <- est[beta_rows, "Estimate"]
eta0_fitted <- est[eta0_rows, "Estimate"]

## Compute predicted eta and S for all training pairs
eta_train <- rep(eta0_fitted, nrow(X)) + as.numeric(X %*% beta_fitted)
S_train   <- exp(-eta_train)

## Compute predicted p_match using fixed alpha
ai_train <- alpha_vec[pairs_dt$site_i_idx + 1L]
aj_train <- alpha_vec[pairs_dt$site_j_idx + 1L]
pmatch_train <- S_train * (ai_train + aj_train) / (2 * ai_train * aj_train)
pmatch_train <- pmin(pmax(pmatch_train, 0), 1)

## Summary
cat(sprintf("  Training set predictions (%d pairs):\n", nrow(pairs_dt)))
cat(sprintf("    eta:     mean = %.3f, range = [%.3f, %.3f]\n",
            mean(eta_train), min(eta_train), max(eta_train)))
cat(sprintf("    S:       mean = %.3f, range = [%.3f, %.3f]\n",
            mean(S_train), min(S_train), max(S_train)))
cat(sprintf("    p_match: mean = %.4f, range = [%.4f, %.4f]\n",
            mean(pmatch_train), min(pmatch_train), max(pmatch_train)))

## Classification accuracy (threshold = 0.5)
pred_class <- as.integer(pmatch_train < 0.5)  # 0=match, 1=mismatch
accuracy   <- mean(pred_class == pairs_dt$y)
cat(sprintf("    Accuracy (threshold 0.5): %.1f%%\n", accuracy * 100))

## Attach predictions to results
results$train_predictions <- data.table(
  site_i  = pairs_dt$site_i,
  site_j  = pairs_dt$site_j,
  y       = pairs_dt$y,
  eta     = eta_train,
  S       = S_train,
  p_match = pmatch_train
)

## Re-save with predictions
saveRDS(results, file = results_file)
cat(sprintf("  Updated results saved to %s\n", results_file))
