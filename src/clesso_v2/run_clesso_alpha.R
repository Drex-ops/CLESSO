##############################################################################
##
## run_clesso_alpha.R — Alpha-only (richness) pipeline for CLESSO
##
## Simplified pipeline that uses ONLY within-site observation pairs to
## estimate species richness (alpha) per site — no turnover model.
##
## Pipeline:
##   1. Load data → siteAggregator
##   2. Date filter → format for CLESSO sampler
##   3. Sample within-site observation pairs only
##   4. Extract site-level environmental covariates
##   5. Prepare alpha-only TMB model data
##   6. Compile and fit the TMB model (clesso_alpha.cpp)
##   7. Diagnostics and output
##   8. Predict alpha at training sites
##
## Usage:
##   1. Edit clesso_config.R (or set CLESSO_* environment variables)
##   2. source("run_clesso_alpha.R")
##
##############################################################################

cat("=== CLESSO Alpha-Only Pipeline ===\n")
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
source(file.path(clesso_config$r_dir, "site_aggregator.R"))

## Source CLESSO v2 modules (sampler + data prep + predict)
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
# STEP 3: Sample WITHIN-SITE observation pairs only
# ===========================================================================
cat("\n--- Step 3: Within-site observation-pair sampling ---\n")

within_pairs <- clesso_sample_within_pairs(
  obs_dt      = obs_dt,
  n_pairs     = clesso_config$n_within,
  min_records = clesso_config$within_min_records,
  match_ratio = clesso_config$within_match_ratio,
  seed        = clesso_config$seed
)

## Add columns expected by downstream functions
within_pairs[, is_within := 1L]

## Compute pair weights (match vs mismatch balancing)
n_match <- sum(within_pairs$y == 0)
n_miss  <- sum(within_pairs$y == 1)
if (n_match > 0 && n_miss > 0) {
  within_pairs[y == 0, w := 0.5 / n_match]
  within_pairs[y == 1, w := 0.5 / n_miss]
  cat(sprintf("  Pair weights: match w=%.6f (%d pairs), mismatch w=%.6f (%d pairs)\n",
              0.5 / n_match, n_match, 0.5 / n_miss, n_miss))
} else {
  within_pairs[, w := 1.0 / .N]
  cat("  Warning: only one class present. Using uniform weights.\n")
}

pairs_dt <- within_pairs

## Save pairs
pairs_file <- file.path(clesso_config$output_dir,
  paste0("clesso_alpha_pairs_", clesso_config$species_group, ".rds"))
saveRDS(pairs_dt, file = pairs_file)
cat(sprintf("  Within-site pairs saved to %s\n", pairs_file))


# ===========================================================================
# STEP 4: Extract site-level environmental covariates
# ===========================================================================
cat("\n--- Step 4: Extract site-level environmental covariates ---\n")

## All unique sites from the within-site pairs
unique_sites <- unique(pairs_dt[, .(
  site_id = site_i, lon = lon_i, lat = lat_i
)])
unique_sites <- unique(unique_sites, by = "site_id")

## Substrate extraction
subs_raster <- tryCatch(
  raster::brick(clesso_config$substrate_raster),
  error = function(e) {
    warning(paste("Could not load substrate raster:", e$message))
    NULL
  }
)

if (!is.null(subs_raster)) {
  subs_vals <- raster::extract(subs_raster,
                                unique_sites[, .(lon, lat)])
  if (is.matrix(subs_vals)) {
    subs_dt <- as.data.table(subs_vals)
    names(subs_dt) <- paste0("subs_", seq_len(ncol(subs_dt)))
  } else {
    subs_dt <- data.table(subs_1 = subs_vals)
  }
  site_covs <- cbind(unique_sites[, .(site_id)], subs_dt)
  cat(sprintf("  Extracted %d substrate variables at %d sites\n",
              ncol(subs_dt), nrow(site_covs)))
} else {
  site_covs <- NULL
  cat("  No substrate data available. Using lon/lat for alpha model.\n")
}


# ===========================================================================
# STEP 5: Prepare alpha-only TMB model data
# ===========================================================================
cat("\n--- Step 5: Prepare alpha-only model data ---\n")

## Build site table and Z matrix (reuse the existing function)
site_info <- clesso_build_site_table(
  pairs_dt    = pairs_dt,
  site_covs   = site_covs,
  standardize = clesso_config$standardize_Z
)

site_table <- site_info$site_table
Z          <- site_info$Z

## Map pair site IDs to 0-based indices
pairs_dt <- as.data.table(pairs_dt)
pairs_dt[site_table, site_i_idx := i.site_index, on = .(site_i = site_id)]

stopifnot(!anyNA(pairs_dt$site_i_idx))

## Build alpha spline basis (if requested)
alpha_spline_info <- NULL
use_splines <- clesso_config$use_alpha_splines

if (use_splines) {
  spline_label <- ifelse(clesso_config$alpha_spline_type == "regression",
                         "regression spline", "P-spline")
  cat(sprintf("\n--- Building alpha %s basis ---\n", spline_label))
  Z_for_splines <- site_info$Z_raw
  if (is.null(Z_for_splines)) Z_for_splines <- Z

  alpha_spline_info <- clesso_build_alpha_splines(
    Z_raw          = Z_for_splines,
    n_knots        = clesso_config$alpha_n_knots,
    spline_deg     = clesso_config$alpha_spline_deg,
    pen_order      = clesso_config$alpha_pen_order,
    cov_names      = site_info$alpha_cov_cols,
    knot_positions = clesso_config$alpha_knot_positions
  )

  B_alpha <- alpha_spline_info$B_alpha
  S_alpha <- alpha_spline_info$S_alpha
  K_basis <- ncol(B_alpha)
  n_lambda_blocks <- alpha_spline_info$n_covariates

} else {
  ## Dummy matrices for TMB (use_alpha_splines=0 means these are ignored)
  nSites_tmp <- nrow(site_table)
  B_alpha <- matrix(0, nrow = nSites_tmp, ncol = 1)
  S_alpha <- matrix(0, nrow = 1, ncol = 1)
  K_basis <- 1L
  n_lambda_blocks <- 1L
  storage.mode(B_alpha) <- "double"
  storage.mode(S_alpha) <- "double"
}

nSites <- nrow(site_table)
Kalpha <- ncol(Z)

## Assemble TMB data list (alpha-only: no X, no site_j, no is_within)
data_list <- list(
  y                 = as.numeric(pairs_dt$y),
  site_i            = as.integer(pairs_dt$site_i_idx),
  w                 = as.numeric(pairs_dt$w),
  Z                 = Z,
  B_alpha           = B_alpha,
  S_alpha           = S_alpha,
  alpha_block_sizes = if (use_splines) as.integer(alpha_spline_info$n_bases_per_cov)
                      else as.integer(1),
  use_alpha_splines = as.integer(use_splines)
)

## Initial parameter values
parameters <- list(
  alpha0           = log(clesso_config$alpha_init - 1),
  theta_alpha      = rep(0, Kalpha),
  b_alpha          = rep(0, K_basis),
  log_lambda_alpha = rep(log(1.0), n_lambda_blocks),
  u_site           = rep(0, nSites),
  log_sigma_u      = log(0.5)
)

## Store model data for prediction later
model_data <- list(
  data_list  = data_list,
  parameters = parameters,
  site_info  = site_info,
  alpha_info = list(
    cov_cols          = site_info$alpha_cov_cols,
    z_center          = site_info$z_center,
    z_scale           = site_info$z_scale,
    use_alpha_splines = use_splines,
    spline_type       = clesso_config$alpha_spline_type,
    spline_info       = alpha_spline_info
  ),
  pairs_dt = pairs_dt
)

## Save model data
model_data_file <- file.path(clesso_config$output_dir,
  paste0("clesso_alpha_model_data_", clesso_config$species_group, ".rds"))
saveRDS(model_data, file = model_data_file)
cat(sprintf("  Model data saved to %s\n", model_data_file))

cat(sprintf("\n--- Alpha-only model data summary ---\n"))
cat(sprintf("  Within-site pairs: %d (%d match, %d mismatch)\n",
            nrow(pairs_dt), sum(pairs_dt$y == 0), sum(pairs_dt$y == 1)))
cat(sprintf("  Sites:  %d\n", nSites))
cat(sprintf("  Z dims: %d x %d (alpha linear covariates)\n", nSites, Kalpha))
if (use_splines) {
  cat(sprintf("  B_alpha dims: %d x %d (alpha spline basis)\n",
              nrow(B_alpha), ncol(B_alpha)))
  cat(sprintf("  Smoothing blocks: %d\n", n_lambda_blocks))
} else {
  cat("  Alpha splines: disabled (linear model only)\n")
}


# ===========================================================================
# STEP 6: Compile and fit alpha-only TMB model
# ===========================================================================
cat("\n--- Step 6: Compile and fit alpha-only TMB model ---\n")

library(TMB)

cpp_file     <- file.path(clesso_config$clesso_dir, "clesso_alpha.cpp")
cpp_basename <- tools::file_path_sans_ext(basename(cpp_file))

## Compile (only if needed)
dll_path <- file.path(clesso_config$clesso_dir,
                       paste0(cpp_basename, .Platform$dynlib.ext))
if (!file.exists(dll_path)) {
  cat("  Compiling alpha-only TMB model...\n")
  compile(cpp_file)
} else {
  cat("  Alpha-only TMB model already compiled.\n")
}

## Load the DLL
dyn.load(dynlib(file.path(clesso_config$clesso_dir, cpp_basename)))

## Build TMB objective function
## u_site is always a random effect.
## When alpha splines are enabled, b_alpha coefficients are also random.
## When splines are disabled, map b_alpha and log_lambda_alpha to NA.
## Regression spline mode: b_alpha as fixed effects, penalty disabled.
random_effects <- "u_site"
tmb_map <- list()

if (!use_splines) {
  K_dummy <- length(parameters$b_alpha)
  n_lam   <- length(parameters$log_lambda_alpha)
  tmb_map$b_alpha          <- factor(rep(NA, K_dummy))
  tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
} else if (clesso_config$alpha_spline_type == "regression") {
  ## Regression splines: b_alpha as fixed effects, no smoothness penalty
  n_lam <- length(parameters$log_lambda_alpha)
  tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
  cat("  Regression spline mode: b_alpha as fixed effects, no smoothness penalty\n")
}

obj <- MakeADFun(
  data       = data_list,
  parameters = parameters,
  random     = random_effects,
  map        = tmb_map,
  DLL        = cpp_basename,
  silent     = TRUE
)

## Optimise
cat("  Fitting alpha-only model...\n")
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
## getReportCovariance = FALSE avoids building the dense covariance matrix
## of all ADREPORT'd quantities, which can exceed memory limits.
rep <- sdreport(obj, getReportCovariance = FALSE)


# ===========================================================================
# STEP 7: Extract results and diagnostics
# ===========================================================================
cat("\n--- Step 7: Results and diagnostics ---\n")

est <- summary(rep, "report")

## Alpha (richness) estimates per site
## alpha_site and log_alpha_site are reported via REPORT() (not ADREPORT)
## to avoid the huge delta-method Jacobian. Retrieve via obj$report().
rpt <- obj$report()

alpha_estimates <- data.table(
  site_id       = site_table$site_id,
  site_index    = site_table$site_index,
  lon           = site_table$lon,
  lat           = site_table$lat,
  alpha_est     = rpt$alpha_site,
  log_alpha_est = rpt$log_alpha_site
)

cat(sprintf("  Alpha estimates: mean = %.1f, range = [%.1f, %.1f]\n",
            mean(alpha_estimates$alpha_est),
            min(alpha_estimates$alpha_est),
            max(alpha_estimates$alpha_est)))

## Alpha regression coefficients (linear terms)
theta_rows <- grep("^theta_alpha", rownames(est))
if (length(theta_rows) > 0) {
  cat("  Alpha linear coefficients (theta):\n")
  theta_est <- est[theta_rows, , drop = FALSE]
  if (!is.null(site_info$alpha_cov_cols)) {
    rownames(theta_est) <- site_info$alpha_cov_cols
  }
  print(theta_est)
}

## Alpha intercept
alpha0_rows <- grep("^alpha0$", rownames(est))
if (length(alpha0_rows) > 0) {
  cat(sprintf("  Alpha intercept (alpha0): %.4f (SE: %.4f)\n",
              est[alpha0_rows, "Estimate"], est[alpha0_rows, "Std. Error"]))
}

## Sigma_u
sigma_rows <- grep("^sigma_u$", rownames(est))
if (length(sigma_rows) > 0) {
  cat(sprintf("  Site RE sigma_u: %.4f (SE: %.4f)\n",
              est[sigma_rows, "Estimate"], est[sigma_rows, "Std. Error"]))
}

## Alpha spline coefficients and smoothing parameters
if (use_splines) {
  b_alpha_rows <- grep("^b_alpha$", rownames(est))
  lambda_rows  <- grep("^lambda_alpha", rownames(est))

  if (length(lambda_rows) > 0) {
    cat("  Alpha smoothing parameters (lambda):\n")
    lambda_est <- est[lambda_rows, , drop = FALSE]
    if (!is.null(site_info$alpha_cov_cols)) {
      rownames(lambda_est) <- site_info$alpha_cov_cols
    }
    print(lambda_est)
  }

  if (length(b_alpha_rows) > 0) {
    cat(sprintf("  Alpha spline coefficients: %d total\n", length(b_alpha_rows)))
  }
}

## Save full results
results <- list(
  fit             = fit,
  sdreport        = rep,
  alpha_estimates = alpha_estimates,
  theta_est       = if (length(theta_rows) > 0) est[theta_rows, , drop = FALSE] else NULL,
  model_data      = model_data,
  config          = clesso_config
)

results_file <- file.path(clesso_config$output_dir,
  paste0("clesso_alpha_results_", clesso_config$species_group, ".rds"))
saveRDS(results, file = results_file)

cat(sprintf("\n=== CLESSO alpha-only pipeline complete ===\n"))
cat(sprintf("  Results saved to %s\n", results_file))


# ===========================================================================
# STEP 8: Predict alpha at training sites (demonstration)
# ===========================================================================
cat("\n--- Step 8: Alpha prediction (training set) ---\n")

## Predict alpha at all training sites using clesso_predict_alpha
train_sites <- model_data$site_info$site_table
alpha_pred <- clesso_predict_alpha(
  new_sites  = train_sites,
  results    = results,
  include_re = TRUE
)

cat(sprintf("  Alpha predictions at %d training sites:\n", nrow(alpha_pred)))
cat(sprintf("    mean = %.1f, median = %.1f, range = [%.1f, %.1f]\n",
            mean(alpha_pred$alpha), median(alpha_pred$alpha),
            min(alpha_pred$alpha), max(alpha_pred$alpha)))

## Attach predictions to results
results$alpha_pred <- alpha_pred

## Re-save with predictions
saveRDS(results, file = results_file)
cat(sprintf("  Updated results saved to %s\n", results_file))
