##############################################################################
##
## run_clesso.R -- End-to-end pipeline for CLESSO v2
##
## Pipeline:
##   1. Load data -> siteAggregator
##   2. Date filter -> format for CLESSO sampler
##   3. Sample within-site + between-site observation pairs
##   4. Extract environmental covariates (site-level for alpha, pair-level
##      for turnover)
##   5. Prepare TMB model data (X, Z matrices)
##   6. Compile and fit the TMB model
##   7. Diagnostics and output
##
## Usage:
##   1. Edit clesso_config.R (or set CLESSO_* environment variables)
##   2. source("run_clesso.R")
##
##############################################################################

cat("=== CLESSO v2 Pipeline ===\n")
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

## Source shared modules (old methods preserved)
source(file.path(clesso_config$r_dir, "utils.R"))
source(file.path(clesso_config$r_dir, "gdm_functions.R"))
source(file.path(clesso_config$r_dir, "site_aggregator.R"))
source(file.path(clesso_config$r_dir, "gen_windows.R"))

## Source CLESSO v2 modules
source(file.path(clesso_config$clesso_dir, "clesso_sampler_optimised.R"))
source(file.path(clesso_config$clesso_dir, "clesso_prepare_data.R"))
source(file.path(clesso_config$clesso_dir, "clesso_predict.R"))
source(file.path(clesso_config$clesso_dir, "clesso_iterative.R"))
source(file.path(clesso_config$clesso_dir, "clesso_progress_logger.R"))

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

## Aggregate to grid cells (existing shared method)
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
# STEP 3: Sample within-site + between-site observation pairs
# ===========================================================================
cat("\n--- Step 3: CLESSO observation-pair sampling ---\n")

pairs_dt <- clesso_sampler(
  obs_dt              = obs_dt,
  n_within            = clesso_config$n_within,
  n_between           = clesso_config$n_between,
  within_min_recs     = clesso_config$within_min_records,
  within_match_ratio  = clesso_config$within_match_ratio,
  between_match_ratio = clesso_config$between_match_ratio,
  balance_weights     = clesso_config$balance_weights,
  seed                = clesso_config$seed,
  species_thresh      = clesso_config$species_threshold,
  cores               = clesso_config$cores
)

## Save pairs
pairs_file <- file.path(clesso_config$output_dir,
  paste0("clesso_pairs_", clesso_config$run_id, ".rds"))
saveRDS(pairs_dt, file = pairs_file)
cat(sprintf("  Pairs saved to %s\n", pairs_file))

## Compute observed species richness per site (lower bound for alpha)
## Use the pre-filter richness from siteAggregator (counts ALL species ever
## observed at a site, not just those remaining after date filtering).
cat("  Computing observed richness per site (lower bound)...\n")
site_obs_richness <- unique(obs_dt[, .(site_id, S_obs = richness)])
cat(sprintf("  Observed richness: mean = %.1f, range = [%d, %d] across %d sites\n",
            mean(site_obs_richness$S_obs),
            min(site_obs_richness$S_obs),
            max(site_obs_richness$S_obs),
            nrow(site_obs_richness)))


# ===========================================================================
# STEP 4: Extract environmental covariates
#
# For the alpha (richness) model: site-level covariates extracted at each
# unique site location (substrate, climate, etc.)
#
# For the turnover (beta) model: pairwise env distances computed via
# I-spline transformations of |env_i - env_j|
#
# NOTE: Environmental extraction depends on available climate data.
# This step uses substrate rasters (always available) and optionally
# geonpy climate data. Adapt this section to your data availability.
# ===========================================================================
cat("\n--- Step 4: Extract environmental covariates ---\n")

## Extract substrate values at all unique site locations
unique_sites <- unique(pairs_dt[, .(
  site_id = site_i, lon = lon_i, lat = lat_i
)])
unique_sites_j <- unique(pairs_dt[, .(
  site_id = site_j, lon = lon_j, lat = lat_j
)])
setnames(unique_sites_j, c("site_id", "lon", "lat"))
unique_sites <- unique(rbind(unique_sites, unique_sites_j))
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
  ## Fallback: no substrate data, use coords only
  site_covs <- NULL
  cat("  No substrate data available. Using lon/lat for alpha model.\n")
}

## Environmental site table for turnover (between-site pairs)
## This provides per-site env values that clesso_build_turnover_X
## will difference for between-site pairs.
if (!is.null(subs_raster)) {
  env_site_table <- cbind(unique_sites[, .(site_id)], subs_dt)
  cat(sprintf("  Env site table: %d sites x %d env variables\n",
              nrow(env_site_table), ncol(subs_dt)))
} else {
  env_site_table <- NULL
  cat("  No env site table. Turnover model uses geographic distance only.\n")
}

## ---- Climate variable extraction via geonpy (fixed 30-year average) ----
## Extract a 30-year climate average centred on 2010 for every unique site.
## pyper.py expects pairwise input (x1,y1,year1,month1,x2,y2,year2,month2),
## so we pair each site with itself and use only the _1 output columns.
if (length(clesso_config$env_params) > 0 &&
    file.exists(clesso_config$pyper_script) &&
    dir.exists(clesso_config$npy_src)) {

  cat("  Extracting climate variables (30-year mean centred on 2010)...\n")
  require(arrow)

  ## Fixed reference: year 2010, month 6 (mid-year), window = config value
  clim_year   <- 2010L
  clim_month  <- 6L
  clim_window <- clesso_config$climate_window  # default 30 years

  ## Build dummy pairs (each site paired with itself)
  clim_pairs <- data.frame(
    Lon1   = unique_sites$lon,
    Lat1   = unique_sites$lat,
    year1  = rep(clim_year, nrow(unique_sites)),
    month1 = rep(clim_month, nrow(unique_sites)),
    Lon2   = unique_sites$lon,
    Lat2   = unique_sites$lat,
    year2  = rep(clim_year, nrow(unique_sites)),
    month2 = rep(clim_month, nrow(unique_sites))
  )

  ## Iterate through env_params groups (same structure as reca_STresiduals)
  clim_parts <- list()
  for (j in seq_along(clesso_config$env_params)) {
    ep <- clesso_config$env_params[[j]]
    cat(sprintf("    [%d/%d] %s | mstat=%s cstat=%s\n",
                j, length(clesso_config$env_params),
                paste(ep$variables, collapse = ", "),
                ep$mstat, ep$cstat))

    raw <- gen_windows(
      pairs          = clim_pairs,
      variables      = ep$variables,
      mstat          = ep$mstat,
      cstat          = ep$cstat,
      window         = clim_window,
      npy_src        = clesso_config$npy_src,
      start_year     = clesso_config$geonpy_start_year,
      python_exe     = clesso_config$python_exe,
      pyper_script   = clesso_config$pyper_script,
      feather_tmpdir = clesso_config$feather_tmpdir
    )

    ## Keep only _1 columns (site-level values; skip first 8 coord columns)
    env_cols <- raw[, 9:ncol(raw), drop = FALSE]
    idx_1    <- grep("_1$", names(env_cols))
    site_env <- env_cols[, idx_1, drop = FALSE]

    ## Strip date-range patterns (e.g. "191101-201712_") and _1 suffix
    names(site_env) <- gsub("\\d{6}-\\d{6}_", "", names(site_env))
    names(site_env) <- gsub("_1$", "", names(site_env))

    ## Prefix with cstat to keep names unique across groups
    names(site_env) <- paste0(ep$cstat, "_", names(site_env))

    clim_parts[[j]] <- site_env
  }

  clim_dt <- as.data.table(do.call(cbind, clim_parts))

  ## Check for sentinels / NA
  n_na   <- sum(is.na(clim_dt))
  n_sent <- sum(clim_dt == -9999, na.rm = TRUE)
  if (n_na > 0)   warning(sprintf("  %d NA values in climate extraction.", n_na))
  if (n_sent > 0)  warning(sprintf("  %d sentinel (-9999) values in climate extraction.", n_sent))

  cat(sprintf("  Extracted %d climate variables at %d sites\n",
              ncol(clim_dt), nrow(clim_dt)))

  ## Append to site_covs (alpha model)
  if (!is.null(site_covs)) {
    site_covs <- cbind(site_covs, clim_dt)
  } else {
    site_covs <- cbind(unique_sites[, .(site_id)], clim_dt)
  }

  ## Append to env_site_table (turnover / beta model)
  if (!is.null(env_site_table)) {
    env_site_table <- cbind(env_site_table, clim_dt)
  } else {
    env_site_table <- cbind(unique_sites[, .(site_id)], clim_dt)
  }

  cat(sprintf("  Final site_covs: %d columns | env_site_table: %d columns\n",
              ncol(site_covs) - 1, ncol(env_site_table) - 1))

} else {
  cat("  Skipping climate extraction (env_params empty or geonpy not available).\n")
}


# ===========================================================================
# STEP 5: Prepare TMB model data
# ===========================================================================
cat("\n--- Step 5: Prepare model data ---\n")

model_data <- clesso_prepare_model_data(
  pairs_dt              = pairs_dt,
  site_covs             = site_covs,
  env_site_table        = env_site_table,
  geo_distance          = clesso_config$geo_distance,
  n_splines             = clesso_config$n_splines,
  standardize_Z         = clesso_config$standardize_Z,
  alpha_init            = clesso_config$alpha_init,
  use_alpha_splines     = clesso_config$use_alpha_splines,
  alpha_n_knots         = clesso_config$alpha_n_knots,
  alpha_spline_deg      = clesso_config$alpha_spline_deg,
  alpha_pen_order       = clesso_config$alpha_pen_order,
  alpha_knot_positions  = clesso_config$alpha_knot_positions,
  site_obs_richness     = site_obs_richness,
  alpha_lower_bound_lambda = clesso_config$alpha_lower_bound_lambda
)

## Save model data
model_data_file <- file.path(clesso_config$output_dir,
  paste0("clesso_model_data_", clesso_config$run_id, ".rds"))
saveRDS(model_data, file = model_data_file)
cat(sprintf("  Model data saved to %s\n", model_data_file))


# ===========================================================================
# STEP 6: Compile and fit TMB model
# ===========================================================================
cat("\n--- Step 6: Compile and fit TMB model ---\n")

library(TMB)

use_iterative <- isTRUE(clesso_config$iterative_fitting)

if (use_iterative) {
  ## ---- Strategy 2: Alternating block-coordinate descent ----
  cat("  Using ITERATIVE (alternating alpha/beta) fitting strategy\n")

  progress_log_file <- file.path(clesso_config$output_dir,
    paste0("clesso_progress_", clesso_config$run_id, ".log"))
  cat(sprintf("  Progress log: %s\n", progress_log_file))
  cat("  Monitor with:  tail -f", progress_log_file, "\n")

  iter_result <- clesso_fit_iterative(
    model_data     = model_data,
    config         = clesso_config,
    max_iter       = clesso_config$iterative_max_iter %||% 20L,
    tol            = clesso_config$iterative_tol %||% 1e-4,
    verbose        = TRUE,
    progress_log   = progress_log_file
  )

  ## Unpack into the same variables used downstream
  obj <- iter_result$obj
  fit <- list(
    par         = iter_result$par,
    objective   = iter_result$objective,
    convergence = iter_result$convergence,
    message     = sprintf("iterative %d cycles, %s",
                          iter_result$iterations,
                          if (iter_result$convergence == 0) "converged" else "max_iter")
  )
  rep <- iter_result$sdreport

  cat(sprintf("  Convergence: %d (message: %s)\n", fit$convergence, fit$message))
  cat(sprintf("  Final objective: %.4f\n", fit$objective))

} else {
  ## ---- Standard joint optimisation ----
  cat("  Using JOINT optimisation strategy\n")

  cpp_file <- file.path(clesso_config$clesso_dir, "clesso_v2.cpp")
  cpp_basename <- tools::file_path_sans_ext(basename(cpp_file))

  ## Compile (only if needed)
  dll_path <- file.path(clesso_config$clesso_dir,
                         paste0(cpp_basename, .Platform$dynlib.ext))
  if (!file.exists(dll_path)) {
    cat("  Compiling TMB model...\n")
    compile(cpp_file)
  } else {
    cat("  TMB model already compiled.\n")
  }

  ## Load the DLL
  dyn.load(dynlib(file.path(clesso_config$clesso_dir, cpp_basename)))

  ## Build TMB objective function
  ## u_site is always a random effect. When alpha splines are enabled,
  ## b_alpha coefficients are also random (penalised via lambda * b'Sb).
  ## When splines are disabled, we map b_alpha and log_lambda_alpha to NA
  ## so TMB holds them fixed at their initial values (zeros/dummy).
  ##
  ## Regression spline mode (alpha_spline_type = "regression"):
  ## Spline basis is used but without the smoothness penalty -- b_alpha are
  ## estimated as fixed effects and log_lambda_alpha is mapped to NA.
  random_effects <- "u_site"
  tmb_map <- list()

  if (!clesso_config$use_alpha_splines) {
    ## Fix dummy spline parameters so they are not estimated
    K_dummy <- length(model_data$parameters$b_alpha)
    n_lam   <- length(model_data$parameters$log_lambda_alpha)
    tmb_map$b_alpha          <- factor(rep(NA, K_dummy))
    tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
  } else if (clesso_config$alpha_spline_type == "regression") {
    ## Regression splines: b_alpha stays as fixed effects (not in random),
    ## and the smoothness penalty is disabled by mapping log_lambda_alpha to NA.
    n_lam <- length(model_data$parameters$log_lambda_alpha)
    tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
    cat("  Regression spline mode: b_alpha as fixed effects, no smoothness penalty\n")
  }

  obj <- MakeADFun(
    data       = model_data$data_list,
    parameters = model_data$parameters,
    random     = random_effects,
    map        = tmb_map,
    DLL        = cpp_basename,
    silent     = TRUE
  )

  ## Optimise
  ## Set up progress logger so convergence can be monitored in real time.
  ## From another terminal run:  tail -f <output_dir>/clesso_progress_<run_id>.log
  progress_log_file <- file.path(clesso_config$output_dir,
    paste0("clesso_progress_", clesso_config$run_id, ".log"))
  logger <- clesso_make_logger(obj, progress_log_file,
                               print_every = 10L,
                               phase_label = "joint")
  cat(sprintf("  Progress log: %s\n", progress_log_file))
  cat("  Monitor with:  tail -f", progress_log_file, "\n")
  cat("  Fitting model...\n")
  fit <- nlminb(
    start     = obj$par,
    objective = logger$fn,
    gradient  = logger$gr,
    control   = list(
      eval.max = clesso_config$tmb_eval_max,
      iter.max = clesso_config$tmb_iter_max
    )
  )
  logger$close()

  cat(sprintf("  Convergence: %d (message: %s)\n", fit$convergence, fit$message))
  cat(sprintf("  Final objective: %.4f\n", fit$objective))

  ## Standard errors
  ## getReportCovariance = FALSE avoids building the dense covariance matrix
  ## of all ADREPORT'd quantities, which can exceed memory limits.
  rep <- sdreport(obj, getReportCovariance = FALSE)
}


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
  site_id       = model_data$site_info$site_table$site_id,
  site_index    = model_data$site_info$site_table$site_index,
  lon           = model_data$site_info$site_table$lon,
  lat           = model_data$site_info$site_table$lat,
  S_obs         = model_data$data_list$S_obs,
  alpha_est     = rpt$alpha_site,
  log_alpha_est = rpt$log_alpha_site
)

cat(sprintf("  Alpha estimates: mean = %.1f, range = [%.1f, %.1f]\n",
            mean(alpha_estimates$alpha_est),
            min(alpha_estimates$alpha_est),
            max(alpha_estimates$alpha_est)))

## Check lower bound violations
if (any(alpha_estimates$S_obs > 0)) {
  n_violations <- sum(alpha_estimates$alpha_est < alpha_estimates$S_obs - 0.5)
  if (n_violations > 0) {
    cat(sprintf("  WARNING: %d sites have alpha_est < S_obs (lower bound violations)\n",
                n_violations))
  } else {
    cat("  All alpha estimates >= observed richness (lower bound satisfied)\n")
  }
}

## Beta (turnover) coefficients
beta_rows <- grep("^beta$", rownames(est))
eta0_rows <- grep("^eta0$", rownames(est))

cat("  Turnover coefficients (beta):\n")
if (length(beta_rows) > 0) {
  beta_est <- est[beta_rows, , drop = FALSE]
  if (!is.null(model_data$turnover_info$col_names)) {
    rownames(beta_est) <- model_data$turnover_info$col_names
  }
  print(beta_est)
}

## Alpha regression coefficients (linear terms)
theta_rows <- grep("^theta_alpha", rownames(est))
if (length(theta_rows) > 0) {
  cat("  Alpha linear coefficients (theta):\n")
  theta_est <- est[theta_rows, , drop = FALSE]
  if (!is.null(model_data$alpha_info$cov_cols)) {
    rownames(theta_est) <- model_data$alpha_info$cov_cols
  }
  print(theta_est)
}

## Alpha spline coefficients and smoothing parameters
if (clesso_config$use_alpha_splines) {
  b_alpha_rows <- grep("^b_alpha$", rownames(est))
  lambda_rows  <- grep("^lambda_alpha", rownames(est))

  if (length(lambda_rows) > 0) {
    cat("  Alpha smoothing parameters (lambda):\n")
    lambda_est <- est[lambda_rows, , drop = FALSE]
    if (!is.null(model_data$alpha_info$cov_cols)) {
      rownames(lambda_est) <- model_data$alpha_info$cov_cols
    }
    print(lambda_est)
  }

  if (length(b_alpha_rows) > 0) {
    cat(sprintf("  Alpha spline coefficients: %d total\n", length(b_alpha_rows)))
    b_alpha_est <- est[b_alpha_rows, , drop = FALSE]
    if (!is.null(model_data$alpha_info$spline_info)) {
      rownames(b_alpha_est) <- model_data$alpha_info$spline_info$col_names
    }
  }
}

## Save full results
results <- list(
  fit             = fit,
  sdreport        = rep,
  alpha_estimates = alpha_estimates,
  beta_est        = if (length(beta_rows) > 0) est[beta_rows, , drop = FALSE] else NULL,
  eta0_est        = if (length(eta0_rows) > 0) est[eta0_rows, , drop = FALSE] else NULL,
  theta_est       = if (length(theta_rows) > 0) est[theta_rows, , drop = FALSE] else NULL,
  model_data      = model_data,
  config          = clesso_config,
  config_snapshot = clesso_snapshot_config(),
  run_id          = clesso_config$run_id
)

results_file <- file.path(clesso_config$output_dir,
  paste0("clesso_results_", clesso_config$run_id, ".rds"))
saveRDS(results, file = results_file)

cat(sprintf("\n=== CLESSO v2 pipeline complete ===\n"))
cat(sprintf("  Results saved to %s\n", results_file))


# ===========================================================================
# STEP 8: Predict alpha and beta for training sites/pairs (demonstration)
#
# This step shows how to use clesso_predict() to obtain predictions.
# Replace the inputs below with new site data for out-of-sample prediction.
# ===========================================================================
cat("\n--- Step 8: Prediction (training-set demonstration) ---\n")

## Predict alpha at all training sites
train_sites <- model_data$site_info$site_table
pred <- clesso_predict(
  results        = results,
  new_sites      = train_sites,
  new_pairs      = NULL,
  include_re     = TRUE,
  compute_pmatch = FALSE
)

cat(sprintf("  Alpha predictions at %d training sites written to results.\n",
            nrow(pred$alpha_pred)))

## Attach predictions to results
results$predictions <- pred

## Re-save with predictions
saveRDS(results, file = results_file)
cat(sprintf("  Updated results saved to %s\n", results_file))
