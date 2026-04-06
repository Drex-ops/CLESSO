##############################################################################
##
## clesso_iterative.R -- Alternating block-coordinate optimisation for CLESSO v2
##
## Strategy:
##   Iterate until convergence:
##     1. Hold alpha fixed, optimise turnover (beta) parameters
##     2. Hold beta  fixed, optimise richness (alpha) parameters
##
## This avoids the full joint Hessian and is especially helpful when
## the model has many spline coefficients in both sub-models.
##
## Parameters are split into two blocks:
##   Beta block:  eta0_raw, beta_raw
##   Alpha block: alpha0, theta_alpha, b_alpha, log_lambda_alpha,
##                u_site, log_sigma_u
##
## TMB's `map` argument is used to fix a block: parameters mapped to
## factor(NA) are held at their current values.
##
##############################################################################

# ---------------------------------------------------------------------------
# clesso_fit_iterative
#
# Alternating block-coordinate descent for a pre-built CLESSO TMB model.
#
# Arguments:
#   model_data       - output from clesso_prepare_model_data()
#   config           - clesso_config list (for spline settings, etc.)
#   max_iter         - maximum number of full alpha/beta cycles (default 20)
#   tol              - convergence tolerance on relative change in
#                      negative log-likelihood (default 1e-4)
#   nlminb_control   - control list passed to nlminb() for each sub-problem
#   verbose          - print iteration diagnostics (default TRUE)
#   warm_start       - optional named list of parameter values to start from
#                      (e.g. from a previous run). Names must match TMB
#                      parameter names.
#
# Returns:
#   list with:
#     par            - final combined parameter vector
#     parameters     - final parameter list (same format as model_data$parameters)
#     objective      - final negative log-likelihood
#     convergence    - 0 if converged, 1 if max_iter reached
#     iterations     - number of full cycles completed
#     trace          - data.frame of per-iteration diagnostics
#     obj            - final TMB object (with all parameters free)
#     sdreport       - sdreport from final model (or NULL if it fails)
# ---------------------------------------------------------------------------
clesso_fit_iterative <- function(model_data,
                                 config,
                                 max_iter       = 20,
                                 tol            = 1e-4,
                                 nlminb_control = NULL,
                                 verbose        = TRUE,
                                 warm_start     = NULL,
                                 progress_log   = NULL) {

  library(TMB)

  ## ---- Default nlminb control ----
  if (is.null(nlminb_control)) {
    nlminb_control <- list(
      eval.max = config$tmb_eval_max %||% 4000L,
      iter.max = config$tmb_iter_max %||% 4000L
    )
  }

  ## ---- Compile & load DLL (idempotent) ----
  cpp_file     <- file.path(config$clesso_dir, "clesso_v2.cpp")
  cpp_basename <- tools::file_path_sans_ext(basename(cpp_file))
  dll_path     <- file.path(config$clesso_dir,
                             paste0(cpp_basename, .Platform$dynlib.ext))
  if (!file.exists(dll_path)) {
    if (verbose) cat("  Compiling TMB model...\n")
    compile(cpp_file)
  }
  if (!is.loaded(cpp_basename)) {
    dyn.load(dynlib(file.path(config$clesso_dir, cpp_basename)))
  }

  ## ---- Current parameter values (mutable copy) ----
  pars <- model_data$parameters
  if (!is.null(warm_start)) {
    for (nm in names(warm_start)) {
      if (nm %in% names(pars)) {
        pars[[nm]] <- warm_start[[nm]]
      }
    }
  }

  ## ---- Identify which block each TMB parameter belongs to ----
  ##  Beta block:  eta0_raw, beta_raw
  ##  Alpha block: everything else
  beta_names  <- c("eta0_raw", "beta_raw")
  alpha_names <- setdiff(names(pars), beta_names)

  ## ---- Build base map (from config, for spline settings) ----
  base_map <- list()
  if (!config$use_alpha_splines) {
    K_dummy <- length(pars$b_alpha)
    n_lam   <- length(pars$log_lambda_alpha)
    base_map$b_alpha          <- factor(rep(NA, K_dummy))
    base_map$log_lambda_alpha <- factor(rep(NA, n_lam))
  } else if (config$alpha_spline_type == "regression") {
    n_lam <- length(pars$log_lambda_alpha)
    base_map$log_lambda_alpha <- factor(rep(NA, n_lam))
  }

  ## Helper: build map that fixes a set of parameter names at current values
  make_map <- function(fix_names) {
    m <- base_map
    for (nm in fix_names) {
      if (nm %in% names(base_map)) next  # already mapped to NA
      m[[nm]] <- factor(rep(NA, length(pars[[nm]])))
    }
    m
  }

  ## Random effects: u_site is always random (when alpha block is free)
  ## When alpha block is fixed, u_site is also fixed via map, so

  ## we set random = character(0) for the beta sub-problem.

  ## ---- Iteration trace ----
  trace <- data.frame(
    iter       = integer(0),
    phase      = character(0),
    nll_before = numeric(0),
    nll_after  = numeric(0),
    converge   = integer(0),
    n_par      = integer(0),
    stringsAsFactors = FALSE
  )

  ## ---- Initial full objective ----
  obj_full <- MakeADFun(
    data       = model_data$data_list,
    parameters = pars,
    random     = "u_site",
    map        = base_map,
    DLL        = cpp_basename,
    silent     = TRUE
  )
  nll_prev <- obj_full$fn(obj_full$par)
  if (verbose) cat(sprintf("\n=== CLESSO Iterative Fitting (max %d cycles, tol=%.1e) ===\n",
                           max_iter, tol))
  if (verbose) cat(sprintf("  Initial NLL: %.4f\n\n", nll_prev))

  converged <- FALSE

  for (iter in seq_len(max_iter)) {

    ## ==================================================================
    ## Phase 1: Fix alpha, optimise beta
    ## ==================================================================
    if (verbose) cat(sprintf("--- Cycle %d, Phase 1: Optimise BETA (fix alpha) ---\n", iter))

    map_fix_alpha <- make_map(alpha_names)

    obj_beta <- MakeADFun(
      data       = model_data$data_list,
      parameters = pars,
      random     = character(0),   # u_site fixed via map
      map        = map_fix_alpha,
      DLL        = cpp_basename,
      silent     = TRUE
    )

    nll_before_beta <- obj_beta$fn(obj_beta$par)

    ## Progress logging for beta phase
    if (!is.null(progress_log)) {
      beta_logger <- clesso_make_logger(
        obj_beta, progress_log,
        print_every = 10L,
        phase_label = sprintf("beta_cycle%d", iter))
      beta_fn <- beta_logger$fn
      beta_gr <- beta_logger$gr
    } else {
      beta_fn <- obj_beta$fn
      beta_gr <- obj_beta$gr
      beta_logger <- NULL
    }

    fit_beta <- nlminb(
      start     = obj_beta$par,
      objective = beta_fn,
      gradient  = beta_gr,
      control   = nlminb_control
    )
    if (!is.null(beta_logger)) beta_logger$close()

    ## Update pars with optimised beta values
    par_full_beta <- obj_beta$env$parList(fit_beta$par)
    for (nm in beta_names) {
      pars[[nm]] <- par_full_beta[[nm]]
    }

    nll_after_beta <- fit_beta$objective
    trace <- rbind(trace, data.frame(
      iter = iter, phase = "beta",
      nll_before = nll_before_beta, nll_after = nll_after_beta,
      converge = fit_beta$convergence,
      n_par = length(fit_beta$par),
      stringsAsFactors = FALSE
    ))

    if (verbose) {
      cat(sprintf("  NLL: %.4f -> %.4f (conv=%d, %d pars)\n",
                  nll_before_beta, nll_after_beta, fit_beta$convergence,
                  length(fit_beta$par)))
    }

    ## ==================================================================
    ## Phase 2: Fix beta, optimise alpha
    ## ==================================================================
    if (verbose) cat(sprintf("--- Cycle %d, Phase 2: Optimise ALPHA (fix beta) ---\n", iter))

    map_fix_beta <- make_map(beta_names)

    ## u_site is random in the alpha phase
    obj_alpha <- MakeADFun(
      data       = model_data$data_list,
      parameters = pars,
      random     = "u_site",
      map        = map_fix_beta,
      DLL        = cpp_basename,
      silent     = TRUE
    )

    nll_before_alpha <- obj_alpha$fn(obj_alpha$par)

    ## Progress logging for alpha phase
    if (!is.null(progress_log)) {
      alpha_logger <- clesso_make_logger(
        obj_alpha, progress_log,
        print_every = 10L,
        phase_label = sprintf("alpha_cycle%d", iter))
      alpha_fn <- alpha_logger$fn
      alpha_gr <- alpha_logger$gr
    } else {
      alpha_fn <- obj_alpha$fn
      alpha_gr <- obj_alpha$gr
      alpha_logger <- NULL
    }

    fit_alpha <- nlminb(
      start     = obj_alpha$par,
      objective = alpha_fn,
      gradient  = alpha_gr,
      control   = nlminb_control
    )
    if (!is.null(alpha_logger)) alpha_logger$close()

    ## Update pars with optimised alpha values
    par_full_alpha <- obj_alpha$env$parList(fit_alpha$par)
    for (nm in alpha_names) {
      pars[[nm]] <- par_full_alpha[[nm]]
    }
    ## Also update u_site from the random effect mode
    pars$u_site <- par_full_alpha$u_site

    nll_after_alpha <- fit_alpha$objective
    trace <- rbind(trace, data.frame(
      iter = iter, phase = "alpha",
      nll_before = nll_before_alpha, nll_after = nll_after_alpha,
      converge = fit_alpha$convergence,
      n_par = length(fit_alpha$par),
      stringsAsFactors = FALSE
    ))

    if (verbose) {
      cat(sprintf("  NLL: %.4f -> %.4f (conv=%d, %d pars + RE)\n",
                  nll_before_alpha, nll_after_alpha, fit_alpha$convergence,
                  length(fit_alpha$par)))
    }

    ## ==================================================================
    ## Check convergence
    ## ==================================================================
    nll_curr <- nll_after_alpha
    rel_change <- abs(nll_curr - nll_prev) / (abs(nll_prev) + 1e-8)

    if (verbose) {
      cat(sprintf("  Cycle %d complete: NLL %.4f -> %.4f (rel change = %.2e)\n\n",
                  iter, nll_prev, nll_curr, rel_change))
    }

    if (rel_change < tol) {
      converged <- TRUE
      if (verbose) cat(sprintf("*** Converged at cycle %d (rel change %.2e < tol %.1e) ***\n\n", iter, rel_change, tol))
      break
    }

    nll_prev <- nll_curr
  }

  if (!converged && verbose) {
    cat(sprintf("*** Did not converge after %d cycles (rel change = %.2e) ***\n\n",
                max_iter, rel_change))
  }

  ## ---- Final model with all parameters free (for sdreport) ----
  if (verbose) cat("--- Building final model for sdreport ---\n")

  obj_final <- MakeADFun(
    data       = model_data$data_list,
    parameters = pars,
    random     = "u_site",
    map        = base_map,
    DLL        = cpp_basename,
    silent     = TRUE
  )

  ## Evaluate at converged values (no further optimisation)
  final_nll <- obj_final$fn(obj_final$par)

  ## sdreport (may fail for very large models, wrapped in tryCatch)
  rep <- tryCatch(
    sdreport(obj_final, getReportCovariance = FALSE),
    error = function(e) {
      warning(paste("sdreport failed:", e$message))
      NULL
    }
  )

  if (verbose) {
    cat(sprintf("  Final NLL: %.4f\n", final_nll))
    cat(sprintf("  Convergence: %s (%d cycles)\n",
                if (converged) "YES" else "NO", iter))
  }

  list(
    par         = obj_final$par,
    parameters  = pars,
    objective   = final_nll,
    convergence = as.integer(!converged),
    iterations  = iter,
    trace       = trace,
    obj         = obj_final,
    sdreport    = rep
  )
}
