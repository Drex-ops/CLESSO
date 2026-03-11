##############################################################################
##
## clesso_hessian_check.R -- Hessian-based parameter identifiability diagnostics
##
## Loads a fitted CLESSO results .rds and examines the Hessian of the
## marginal negative log-likelihood (fixed effects, random effects
## integrated out) to diagnose parameter identifiability issues.
##
## Diagnostics produced:
##   1. Eigenvalue spectrum of the fixed-effect Hessian
##      - Near-zero eigenvalues => flat directions => non-identifiable
##   2. Condition number (ratio of max to min eigenvalue)
##   3. "Sloppy" parameter directions: eigenvectors for the smallest
##      eigenvalues, showing which parameter combinations are poorly
##      determined by the data
##   4. Fixed-effect correlation matrix (from inverse Hessian)
##      - Near ±1 off-diagonals => confounded parameter pairs
##   5. Per-parameter marginal SE / identifiability flag
##
## Usage:
##   Interactive:
##     results_file <- "output/<run_id>/clesso_results_<run_id>.rds"
##     source("clesso_hessian_check.R")
##
##   Command line:
##     Rscript clesso_hessian_check.R <path_to_results.rds>
##
##   As a function (after sourcing):
##     diag <- clesso_hessian_diagnostics(results, rebuild_obj = FALSE)
##
##############################################################################

suppressPackageStartupMessages({
  library(TMB)
  library(Matrix)
})


# ===========================================================================
# clesso_hessian_diagnostics
#
# Main function.  Extracts or recomputes the fixed-effect Hessian and runs
# identifiability diagnostics.
#
# Arguments:
#   results       - A CLESSO results list (as loaded from the .rds file).
#                   Must contain `sdreport` and/or `model_data` + `config`.
#   rebuild_obj   - If TRUE, rebuild the TMB AD object from model_data and
#                   recompute the Hessian via obj$he().  Slower but does not
#                   rely on the sdreport covariance being invertible.
#                   If FALSE (default), extract the Hessian from sdreport.
#   eigen_tol     - Eigenvalues smaller than this fraction of the largest
#                   eigenvalue are flagged as "practically zero" (default 1e-6).
#   corr_tol      - Absolute correlations above this threshold are flagged
#                   as problematic pairwise confounding (default 0.95).
#   verbose       - Print results to console (default TRUE).
#   plot          - If TRUE (default), produce diagnostic plots.
#
# Returns (invisibly):
#   A list with:
#     hessian         - the fixed-effect Hessian matrix
#     eigenvalues     - eigenvalues (sorted ascending)
#     eigenvectors    - corresponding eigenvectors (columns)
#     condition_number - ratio of max to min absolute eigenvalue
#     par_names       - parameter names
#     sloppy_modes    - data.frame of flagged near-zero eigenvalue modes
#     cor_matrix      - correlation matrix (from inverse Hessian)
#     flagged_pairs   - data.frame of highly correlated parameter pairs
#     par_summary     - per-parameter summary (estimate, SE, identifiable?)
#     pdHess          - logical: was the Hessian positive-definite?
# ===========================================================================
clesso_hessian_diagnostics <- function(results,
                                        rebuild_obj = FALSE,
                                        eigen_tol   = 1e-6,
                                        corr_tol    = 0.95,
                                        verbose     = TRUE,
                                        plot        = TRUE) {

  ## ------------------------------------------------------------------
  ## 1. Obtain the fixed-effect Hessian
  ## ------------------------------------------------------------------
  H         <- NULL
  par_names <- NULL
  par_est   <- NULL
  pdHess    <- NULL

  if (!rebuild_obj && !is.null(results$sdreport)) {
    ## --- Path A: extract from sdreport ---
    rep <- results$sdreport

    pdHess    <- rep$pdHess
    par_est   <- rep$par.fixed
    par_names <- names(par_est)

    ## sdreport stores cov.fixed = H^{-1}.  Invert to get H.
    ## If the Hessian wasn't PD, cov.fixed may not exist or may be
    ## unreliable -- fall back to rebuilding the TMB object.
    cov_fixed <- tryCatch(rep$cov.fixed, error = function(e) NULL)

    if (!is.null(cov_fixed) && isTRUE(pdHess)) {
      H <- tryCatch(solve(cov_fixed), error = function(e) {
        if (verbose) cat("  [!] Could not invert cov.fixed; will rebuild TMB object.\n")
        NULL
      })
    }

    if (is.null(H)) {
      if (verbose) cat("  sdreport covariance not usable (pdHess =", pdHess,
                       "); rebuilding TMB object...\n")
      rebuild_obj <- TRUE
    }
  } else {
    rebuild_obj <- TRUE
  }

  if (rebuild_obj) {
    ## --- Path B: reconstruct TMB object and compute Hessian ---
    if (verbose) cat("  Rebuilding TMB object from model_data...\n")

    config     <- results$config
    model_data <- results$model_data

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

    ## Backward compatibility: older results may lack S_obs and
    ## lambda_lower_bound (added in the lower-bound penalty update).
    ## Inject harmless defaults so the current .cpp template is satisfied.
    nSites <- nrow(model_data$data_list$Z)
    if (is.null(model_data$data_list$S_obs)) {
      model_data$data_list$S_obs <- rep(0, nSites)
      if (verbose) cat("  [compat] Injected S_obs (zeros) for older results.\n")
    }
    if (is.null(model_data$data_list$lambda_lower_bound)) {
      model_data$data_list$lambda_lower_bound <- 0
      if (verbose) cat("  [compat] Injected lambda_lower_bound = 0 for older results.\n")
    }

    ## Reconstruct map (same logic as run_clesso.R / clesso_iterative.R)
    tmb_map <- list()
    if (!isTRUE(config$use_alpha_splines)) {
      K_dummy <- length(model_data$parameters$b_alpha)
      n_lam   <- length(model_data$parameters$log_lambda_alpha)
      tmb_map$b_alpha          <- factor(rep(NA, K_dummy))
      tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
    } else if (identical(config$alpha_spline_type, "regression")) {
      n_lam <- length(model_data$parameters$log_lambda_alpha)
      tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
    }

    ## Inject the converged parameter values back. The sdreport stores
    ## fixed effects in par.fixed; random effects can be pulled from
    ## the sdreport or from the saved parameter list.
    pars <- model_data$parameters

    ## Overwrite with converged values if we have them from a previous
    ## iterative fit (stored in results$fit$par or sdreport).
    if (!is.null(results$sdreport)) {
      ## Fixed effects: reconstruct parameter list from sdreport
      rep        <- results$sdreport
      fixed_est  <- rep$par.fixed
      random_est <- rep$par.random
    }

    obj <- MakeADFun(
      data       = model_data$data_list,
      parameters = pars,
      random     = "u_site",
      map        = tmb_map,
      DLL        = cpp_basename,
      silent     = TRUE
    )

    ## Set parameters to the converged values
    if (!is.null(results$sdreport)) {
      par_at_mle <- results$sdreport$par.fixed
    } else if (!is.null(results$fit$par)) {
      par_at_mle <- results$fit$par
    } else {
      par_at_mle <- obj$par
    }

    ## Evaluate at the MLE (this also sets internal state)
    obj$fn(par_at_mle)

    par_est   <- par_at_mle
    par_names <- names(par_at_mle)

    ## Compute the Hessian of the *marginal* NLL (random effects integrated out)
    if (verbose) cat("  Computing Hessian (this may take a moment)...\n")
    H <- tryCatch(
      optimHess(par_at_mle, fn = obj$fn, gr = obj$gr),
      error = function(e) {
        if (verbose) cat("  [!] optimHess failed:", e$message, "\n")
        if (verbose) cat("  Falling back to numDeriv::hessian...\n")
        numDeriv::hessian(obj$fn, par_at_mle)
      }
    )

    pdHess <- all(eigen(H, symmetric = TRUE, only.values = TRUE)$values > 0)
  }

  ## Label rows/columns
  if (is.null(par_names)) par_names <- paste0("p", seq_len(ncol(H)))
  ## TMB uses duplicated names for vector parameters; make unique
  par_labels <- make.unique(par_names, sep = "_")
  rownames(H) <- colnames(H) <- par_labels

  ## ------------------------------------------------------------------
  ## 2. Eigendecomposition
  ## ------------------------------------------------------------------
  eig <- eigen(H, symmetric = TRUE)
  ev  <- eig$values
  ev_order <- order(ev)   # ascending
  ev_sorted <- ev[ev_order]
  vec_sorted <- eig$vectors[, ev_order, drop = FALSE]
  rownames(vec_sorted) <- par_labels

  max_ev <- max(abs(ev))
  min_ev <- min(abs(ev))
  cond_num <- if (min_ev > 0) max_ev / min_ev else Inf

  ## ------------------------------------------------------------------
  ## 3. Flag sloppy modes (near-zero eigenvalues)
  ## ------------------------------------------------------------------
  threshold <- eigen_tol * max_ev
  sloppy_idx <- which(abs(ev_sorted) < threshold)

  sloppy_modes <- data.frame(
    mode_index  = integer(0),
    eigenvalue  = numeric(0),
    dominant_param = character(0),
    dominant_weight = numeric(0),
    stringsAsFactors = FALSE
  )

  if (length(sloppy_idx) > 0) {
    for (k in sloppy_idx) {
      v <- vec_sorted[, k]
      dom <- which.max(abs(v))
      sloppy_modes <- rbind(sloppy_modes, data.frame(
        mode_index     = k,
        eigenvalue     = ev_sorted[k],
        dominant_param = par_labels[dom],
        dominant_weight = v[dom],
        stringsAsFactors = FALSE
      ))
    }
  }

  ## ------------------------------------------------------------------
  ## 4. Correlation matrix from inverse Hessian (= covariance)
  ## ------------------------------------------------------------------
  cor_matrix <- NULL
  flagged_pairs <- data.frame(
    param_i     = character(0),
    param_j     = character(0),
    correlation = numeric(0),
    stringsAsFactors = FALSE
  )

  cov_fixed <- tryCatch(solve(H), error = function(e) NULL)

  if (!is.null(cov_fixed)) {
    se <- sqrt(pmax(diag(cov_fixed), 0))
    D_inv <- ifelse(se > 0, 1 / se, 0)
    cor_matrix <- diag(D_inv) %*% cov_fixed %*% diag(D_inv)
    rownames(cor_matrix) <- colnames(cor_matrix) <- par_labels

    ## Flag high correlations
    for (i in seq_len(nrow(cor_matrix) - 1)) {
      for (j in (i + 1):ncol(cor_matrix)) {
        if (abs(cor_matrix[i, j]) > corr_tol) {
          flagged_pairs <- rbind(flagged_pairs, data.frame(
            param_i     = par_labels[i],
            param_j     = par_labels[j],
            correlation = cor_matrix[i, j],
            stringsAsFactors = FALSE
          ))
        }
      }
    }
  }

  ## ------------------------------------------------------------------
  ## 5. Per-parameter summary
  ## ------------------------------------------------------------------
  se_vec <- if (!is.null(cov_fixed)) sqrt(pmax(diag(cov_fixed), 0)) else rep(NA, length(par_est))

  par_summary <- data.frame(
    parameter    = par_labels,
    estimate     = par_est,
    std_error    = se_vec,
    rel_se       = ifelse(abs(par_est) > 1e-12, se_vec / abs(par_est), NA),
    identifiable = se_vec < Inf & !is.na(se_vec) & se_vec > 0,
    stringsAsFactors = FALSE
  )
  rownames(par_summary) <- NULL

  ## ------------------------------------------------------------------
  ## 6. Console output
  ## ------------------------------------------------------------------
  if (verbose) {
    cat("\n")
    cat("=" |> rep(72) |> paste(collapse = ""), "\n")
    cat("  CLESSO Hessian Identifiability Diagnostics\n")
    cat("=" |> rep(72) |> paste(collapse = ""), "\n\n")

    cat(sprintf("  Number of fixed-effect parameters: %d\n", length(par_est)))
    cat(sprintf("  Hessian positive-definite: %s\n",
                if (isTRUE(pdHess)) "YES" else "NO  *** POTENTIAL ISSUE ***"))
    cat(sprintf("  Condition number: %.2e\n", cond_num))
    cat(sprintf("  Eigenvalue range: [%.4e, %.4e]\n", min(ev_sorted), max(ev_sorted)))

    ## Eigenvalue summary
    cat("\n--- Eigenvalue spectrum (smallest 10) ---\n")
    n_show <- min(10, length(ev_sorted))
    for (k in seq_len(n_show)) {
      flag <- if (abs(ev_sorted[k]) < threshold) " *** NEAR-ZERO ***" else ""
      cat(sprintf("  [%2d] eigenvalue = %12.4e%s\n", k, ev_sorted[k], flag))

      ## Show the top contributing parameters for flagged modes
      if (abs(ev_sorted[k]) < threshold) {
        v <- vec_sorted[, k]
        top_idx <- order(abs(v), decreasing = TRUE)[1:min(5, length(v))]
        for (ti in top_idx) {
          cat(sprintf("         %s: %.4f\n", par_labels[ti], v[ti]))
        }
      }
    }

    ## Sloppy modes summary
    if (nrow(sloppy_modes) > 0) {
      cat(sprintf("\n*** %d SLOPPY MODE(S) DETECTED (eigenvalue < %.1e × max) ***\n",
                  nrow(sloppy_modes), eigen_tol))
      cat("  These directions in parameter space are poorly constrained by the data.\n")
      cat("  The dominant parameter in each sloppy direction is listed above.\n")
      cat("  Consider:\n")
      cat("    - Removing or combining redundant covariates\n")
      cat("    - Adding stronger priors / penalties\n")
      cat("    - Checking whether the data inform all model components\n")
    } else {
      cat("\n  No sloppy modes detected -- all eigenvalues well above threshold.\n")
    }

    ## Flagged correlations
    if (nrow(flagged_pairs) > 0) {
      cat(sprintf("\n--- Highly correlated parameter pairs (|r| > %.2f) ---\n", corr_tol))
      for (r in seq_len(nrow(flagged_pairs))) {
        cat(sprintf("  %s  <->  %s : r = %.4f\n",
                    flagged_pairs$param_i[r],
                    flagged_pairs$param_j[r],
                    flagged_pairs$correlation[r]))
      }
      cat("  High correlations suggest these parameters trade off against each other.\n")
    } else {
      cat(sprintf("\n  No parameter pairs with |correlation| > %.2f.\n", corr_tol))
    }

    ## Parameters with large relative SE
    cat("\n--- Parameter summary ---\n")
    print(par_summary, digits = 4, row.names = FALSE)

    cat("\n")
    cat("=" |> rep(72) |> paste(collapse = ""), "\n")
  }

  ## ------------------------------------------------------------------
  ## 7. Plots
  ## ------------------------------------------------------------------
  if (plot) {
    op <- par(no.readonly = TRUE)
    on.exit(par(op), add = TRUE)

    ## --- Plot A: Eigenvalue spectrum ---
    par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

    plot(seq_along(ev_sorted), log10(pmax(abs(ev_sorted), 1e-300)),
         type = "b", pch = 19, cex = 0.8,
         col = ifelse(abs(ev_sorted) < threshold, "red", "steelblue"),
         xlab = "Mode index (sorted by eigenvalue)",
         ylab = expression(log[10]("|eigenvalue|")),
         main = "Hessian Eigenvalue Spectrum")
    abline(h = log10(threshold), lty = 2, col = "red")
    legend("bottomright", legend = c("OK", "Near-zero", "Threshold"),
           col = c("steelblue", "red", "red"), pch = c(19, 19, NA),
           lty = c(NA, NA, 2), bty = "n", cex = 0.8)

    ## --- Plot B: Correlation matrix heatmap ---
    if (!is.null(cor_matrix) && ncol(cor_matrix) <= 60) {
      n_par <- ncol(cor_matrix)
      image(1:n_par, 1:n_par, cor_matrix[n_par:1, ],
            col = colorRampPalette(c("#B2182B", "white", "#2166AC"))(101),
            zlim = c(-1, 1),
            xaxt = "n", yaxt = "n",
            xlab = "", ylab = "",
            main = "Fixed-Effect Correlation Matrix")
      axis(1, at = 1:n_par, labels = par_labels, las = 2, cex.axis = 0.6)
      axis(2, at = 1:n_par, labels = rev(par_labels), las = 2, cex.axis = 0.6)
    } else if (!is.null(cor_matrix)) {
      ## Too many parameters for labels -- just show the image
      n_par <- ncol(cor_matrix)
      image(1:n_par, 1:n_par, cor_matrix[n_par:1, ],
            col = colorRampPalette(c("#B2182B", "white", "#2166AC"))(101),
            zlim = c(-1, 1),
            xlab = "Parameter index", ylab = "Parameter index",
            main = "Fixed-Effect Correlation Matrix")
    }
  }

  ## ------------------------------------------------------------------
  ## Return
  ## ------------------------------------------------------------------
  out <- list(
    hessian          = H,
    eigenvalues      = ev_sorted,
    eigenvectors     = vec_sorted,
    condition_number = cond_num,
    par_names        = par_labels,
    par_estimates    = par_est,
    sloppy_modes     = sloppy_modes,
    cor_matrix       = cor_matrix,
    cov_fixed        = cov_fixed,
    flagged_pairs    = flagged_pairs,
    par_summary      = par_summary,
    pdHess           = pdHess
  )

  invisible(out)
}


# ===========================================================================
# clesso_profile_sloppy
#
# For a given sloppy mode, profile the NLL along that eigenvector direction.
# Useful for confirming the parameter is genuinely flat (not just numerically
# noisy) and visualising how far the likelihood is from being informative.
#
# Arguments:
#   results    - fitted clesso results
#   diag       - output from clesso_hessian_diagnostics()
#   mode_index - which sloppy mode to profile (index into diag$eigenvalues)
#   range      - how far to perturb in each direction (in units of
#                the eigenvalue-implied SE, default ±3)
#   n_points   - number of points to evaluate (default 51)
#   plot       - produce a profile plot (default TRUE)
#
# Returns (invisibly):
#   data.frame with columns: step, delta, nll
# ===========================================================================
clesso_profile_sloppy <- function(results, diag,
                                   mode_index = 1,
                                   range = 3,
                                   n_points = 51,
                                   plot = TRUE) {

  config     <- results$config
  model_data <- results$model_data

  ## Rebuild TMB object
  cpp_file     <- file.path(config$clesso_dir, "clesso_v2.cpp")
  cpp_basename <- tools::file_path_sans_ext(basename(cpp_file))
  dll_path     <- file.path(config$clesso_dir,
                             paste0(cpp_basename, .Platform$dynlib.ext))
  if (!file.exists(dll_path)) compile(cpp_file)
  if (!is.loaded(cpp_basename)) {
    dyn.load(dynlib(file.path(config$clesso_dir, cpp_basename)))
  }

  tmb_map <- list()
  if (!isTRUE(config$use_alpha_splines)) {
    K_dummy <- length(model_data$parameters$b_alpha)
    n_lam   <- length(model_data$parameters$log_lambda_alpha)
    tmb_map$b_alpha          <- factor(rep(NA, K_dummy))
    tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
  } else if (identical(config$alpha_spline_type, "regression")) {
    n_lam <- length(model_data$parameters$log_lambda_alpha)
    tmb_map$log_lambda_alpha <- factor(rep(NA, n_lam))
  }

  obj <- MakeADFun(
    data       = model_data$data_list,
    parameters = model_data$parameters,
    random     = "u_site",
    map        = tmb_map,
    DLL        = cpp_basename,
    silent     = TRUE
  )

  par_mle <- diag$par_estimates

  ## Direction = eigenvector for the requested mode
  direction <- diag$eigenvectors[, mode_index]
  ev        <- diag$eigenvalues[mode_index]

  ## Implied SE along this direction: 1/sqrt(eigenvalue)
  step_se <- if (ev > 0) 1 / sqrt(ev) else 1.0

  deltas <- seq(-range * step_se, range * step_se, length.out = n_points)
  nll_values <- numeric(n_points)

  for (i in seq_along(deltas)) {
    par_try <- par_mle + deltas[i] * direction
    nll_values[i] <- obj$fn(par_try)
  }

  nll_profile <- data.frame(
    step  = seq_along(deltas),
    delta = deltas,
    nll   = nll_values
  )

  if (plot) {
    plot(deltas, nll_values, type = "l", lwd = 2, col = "steelblue",
         xlab = paste0("Perturbation along mode ", mode_index),
         ylab = "Negative log-likelihood",
         main = paste0("NLL Profile Along Sloppy Mode ", mode_index,
                       "\n(eigenvalue = ", formatC(ev, format = "e", digits = 2), ")"))
    abline(v = 0, lty = 2, col = "grey40")
    abline(h = min(nll_values) + 1.92, lty = 3, col = "red")
    legend("topright", legend = c("NLL", "MLE", "+1.92 (95% CI)"),
           col = c("steelblue", "grey40", "red"),
           lty = c(1, 2, 3), lwd = c(2, 1, 1), bty = "n", cex = 0.8)
  }

  invisible(nll_profile)
}


# ===========================================================================
# Script execution (when sourced or run from command line)
# ===========================================================================
if (sys.nframe() == 0L || !interactive()) {
  ## Command-line usage
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1) {
    results_file <- args[1]
  }
}

## If results_file is set (either from CLI or pre-set before sourcing),
## run the diagnostics interactively.
if (exists("results_file") && nchar(results_file) > 0) {
  if (!file.exists(results_file)) {
    stop(paste("Results file not found:", results_file))
  }

  cat(sprintf("Loading results from: %s\n", results_file))
  results <- readRDS(results_file)

  diag <- clesso_hessian_diagnostics(results)

  ## If there are sloppy modes, profile the first one
  if (nrow(diag$sloppy_modes) > 0) {
    cat("\nProfiling the first sloppy mode...\n")
    prof <- clesso_profile_sloppy(results, diag, mode_index = 1)
  }
}
