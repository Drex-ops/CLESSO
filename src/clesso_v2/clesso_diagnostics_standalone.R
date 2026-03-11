##############################################################################
##
## clesso_diagnostics_standalone.R
##
## Self-contained diagnostic & function-shape plots for a CLESSO results
## .rds file.  Does NOT source clesso_config.R or any other project file —
## everything is read from the results object itself plus inlined helpers.
##
## Generates all the same sections as clesso_diagnostics.R:
##   A. MODEL DIAGNOSTICS
##      A1. Convergence summary
##      A2. Residual histograms + observed vs predicted
##      A3. Classification metrics (AUC)
##      A4. Random effects distribution + QQ
##      A5. Spatial map + histogram of alpha
##   B. FITTED FUNCTION SHAPES
##      B1. Alpha spline partial-effect curves g_k(z) (log & alpha scale)
##      B2. Beta (turnover) I-spline response curves
##      B3. Similarity decay curves S = exp(-eta)
##      B4. Beta coefficient bar chart
##      B5. Variable importance (summed beta)
##      B6. Alpha effects with uncertainty ribbons
##
## Usage:
##   Rscript clesso_diagnostics_standalone.R <path_to_results.rds>
##
## Or interactively:
##   results_file <- "output/clesso_results_AVES.rds"
##   source("clesso_diagnostics_standalone.R")
##
##############################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(splines)
})

# ===========================================================================
# Inlined helper functions (no external source() required)
# ===========================================================================

## --- clesso_extract_params ---
## Handles both:
##   (a) newer format: sdreport with ADREPORT'd parameters
##   (b) older format: predictions$params list + fit$par + beta_est/theta_est/eta0_est
clesso_extract_params <- function(results) {

  ## --- Try newer sdreport-based extraction first ---
  sdr <- results$sdreport
  can_use_sdr <- FALSE
  if (!is.null(sdr)) {
    est <- tryCatch(summary(sdr, "report"), error = function(e) NULL)
    if (!is.null(est) && "alpha0" %in% rownames(est)) {
      can_use_sdr <- TRUE
    }
  }

  if (can_use_sdr) {
    est_r <- summary(sdr, "random")

    alpha0      <- unname(est[grep("^alpha0$", rownames(est)), "Estimate"])
    theta_alpha <- unname(est[grep("^theta_alpha$", rownames(est)), "Estimate"])
    sigma_u     <- unname(est[grep("^sigma_u$", rownames(est)), "Estimate"])
    eta0        <- unname(est[grep("^eta0$", rownames(est)), "Estimate"])
    beta        <- unname(est[grep("^beta$", rownames(est)), "Estimate"])

    b_alpha_idx  <- grep("^b_alpha$", rownames(est))
    b_alpha      <- if (length(b_alpha_idx) > 0) unname(est[b_alpha_idx, "Estimate"])
                    else numeric(0)
    lambda_idx   <- grep("^lambda_alpha$", rownames(est))
    lambda_alpha <- if (length(lambda_idx) > 0) unname(est[lambda_idx, "Estimate"])
                    else numeric(0)
    u_idx  <- grep("^u_site$", rownames(est_r))
    u_site <- if (length(u_idx) > 0) unname(est_r[u_idx, "Estimate"]) else numeric(0)

    alpha0_se      <- unname(est[grep("^alpha0$", rownames(est)), "Std. Error"])
    theta_alpha_se <- unname(est[grep("^theta_alpha$", rownames(est)), "Std. Error"])
    beta_se        <- unname(est[grep("^beta$", rownames(est)), "Std. Error"])
    eta0_se        <- unname(est[grep("^eta0$", rownames(est)), "Std. Error"])

    return(list(
      alpha0 = alpha0, theta_alpha = theta_alpha, b_alpha = b_alpha,
      sigma_u = sigma_u, u_site = u_site, eta0 = eta0, beta = beta,
      lambda_alpha = lambda_alpha,
      alpha0_se = alpha0_se, theta_alpha_se = theta_alpha_se,
      beta_se = beta_se, eta0_se = eta0_se
    ))
  }

  ## --- Fallback: older/pre-run-id format ---
  ## Try predictions$params (already extracted)
  pp <- results$predictions$params
  if (!is.null(pp)) {
    return(list(
      alpha0         = pp$alpha0,
      theta_alpha    = pp$theta_alpha,
      b_alpha        = pp$b_alpha %||% numeric(0),
      sigma_u        = pp$sigma_u,
      u_site         = pp$u_site %||% numeric(0),
      eta0           = pp$eta0,
      beta           = pp$beta,
      lambda_alpha   = pp$lambda_alpha %||% numeric(0),
      alpha0_se      = pp$alpha0_se %||% NA_real_,
      theta_alpha_se = pp$theta_alpha_se %||% NA_real_,
      beta_se        = pp$beta_se %||% NA_real_,
      eta0_se        = pp$eta0_se %||% NA_real_
    ))
  }

  ## Last resort: reconstruct from fit$par + stored tables
  fit_par <- results$fit$par
  alpha0  <- unname(fit_par[names(fit_par) == "alpha0"])
  theta_alpha <- unname(fit_par[names(fit_par) == "theta_alpha"])
  b_alpha <- unname(fit_par[names(fit_par) == "b_alpha"])
  sigma_u <- if ("log_sigma_u" %in% names(fit_par))
               exp(unname(fit_par["log_sigma_u"])) else NA_real_
  eta0    <- if (!is.null(results$eta0_est))
               results$eta0_est[1, "Estimate"] else NA_real_
  beta    <- if (!is.null(results$beta_est))
               results$beta_est[, "Estimate"] else numeric(0)
  beta_se <- if (!is.null(results$beta_est))
               results$beta_est[, "Std. Error"] else NA_real_
  theta_se <- if (!is.null(results$theta_est))
                results$theta_est[, "Std. Error"] else NA_real_
  eta0_se <- if (!is.null(results$eta0_est))
               results$eta0_est[1, "Std. Error"] else NA_real_

  list(
    alpha0 = alpha0, theta_alpha = theta_alpha, b_alpha = b_alpha,
    sigma_u = sigma_u, u_site = numeric(0), eta0 = unname(eta0),
    beta = unname(beta), lambda_alpha = numeric(0),
    alpha0_se = NA_real_, theta_alpha_se = unname(theta_se),
    beta_se = unname(beta_se), eta0_se = unname(eta0_se)
  )
}

## --- I_spline (from gdm_functions.R) ---
I_spline <- function(predVal, q1, q2, q3) {
  outVal <- rep(NA_real_, length(predVal))
  ok <- !is.na(predVal)
  pv <- predVal[ok]
  ov <- rep(NA_real_, length(pv))
  ov[pv <= q1] <- 0
  ov[pv >= q3] <- 1
  ov[pv > q1 & pv <= q2] <-
    ((pv[pv > q1 & pv <= q2] - q1)^2) / ((q2 - q1) * (q3 - q1))
  ov[pv > q2 & pv < q3] <-
    1 - ((q3 - pv[pv > q2 & pv < q3])^2) / ((q3 - q2) * (q3 - q1))
  outVal[ok] <- ov
  outVal
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
pal_blue   <- "#2166AC"
pal_red    <- "#B2182B"
pal_orange <- "#E08214"
pal_green  <- "#1B7837"
pal_grey   <- "#636363"
pal_light  <- "#D1E5F0"


# ===========================================================================
# 1. Locate the results file
# ===========================================================================
if (!exists("results_file")) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1) {
    results_file <- args[1]
  } else {
    this_dir <- tryCatch(
      dirname(sys.frame(1)$ofile),
      error = function(e) {
        if (requireNamespace("rstudioapi", quietly = TRUE) &&
            rstudioapi::isAvailable()) {
          dirname(rstudioapi::getActiveDocumentContext()$path)
        } else getwd()
      }
    )
    candidates <- list.files(file.path(this_dir, "output"),
                             pattern = "^clesso_results_.*\\.rds$",
                             full.names = TRUE)
    if (length(candidates) == 0)
      stop("No results file found. Pass the path as an argument or set `results_file`.")
    results_file <- candidates[which.max(file.info(candidates)$mtime)]
    cat(sprintf("Auto-detected results file: %s\n", results_file))
  }
}

if (!file.exists(results_file))
  stop(paste("Results file not found:", results_file))

cat(sprintf("\n=== CLESSO Standalone Diagnostics ===\n"))
cat(sprintf("  Results file: %s\n", results_file))

results <- readRDS(results_file)


# ===========================================================================
# 2. Extract metadata
# ===========================================================================
config_snap <- results$config_snapshot
config      <- results$config  # may also exist

if (!is.null(config_snap)) {
  species_grp <- config_snap$species_group %||% "unknown"
  run_id      <- results$run_id %||% config_snap$run_id %||% species_grp
  cat(sprintf("  Run ID       : %s\n", run_id))
  cat(sprintf("  Species group: %s\n", species_grp))
  cat(sprintf("  Snapshot time: %s\n", config_snap$snapshot_time %||% "N/A"))

  cat("\n  --- Config at time of fitting ---\n")
  config_keys <- c("n_within", "n_between", "n_splines", "geo_distance",
                   "use_alpha_splines", "alpha_spline_type", "alpha_n_knots",
                   "alpha_spline_deg", "alpha_init", "standardize_Z",
                   "alpha_lower_bound_lambda", "balance_weights",
                   "min_date", "max_date", "climate_window", "seed")
  for (k in config_keys) {
    v <- config_snap[[k]]
    if (!is.null(v)) cat(sprintf("    %-30s: %s\n", k, paste(v, collapse = ", ")))
  }
  cat("\n")
} else {
  species_grp <- if (!is.null(config$species_group)) config$species_group
                 else sub("^clesso_results_", "",
                          sub("\\.rds$", "", basename(results_file)))
  run_id <- results$run_id %||% species_grp
  cat(sprintf("  Species group (inferred): %s\n", species_grp))
  cat("  (No config snapshot embedded — older results format)\n\n")
}

out_dir  <- dirname(results_file)
pdf_file <- file.path(out_dir, paste0("clesso_diagnostics_", run_id, ".pdf"))


# ===========================================================================
# 3. Extract components
# ===========================================================================
model_data  <- results$model_data
params      <- clesso_extract_params(results)
alpha_est   <- results$alpha_estimates
alpha_pred  <- results$alpha_predictions %||%
              results$predictions$alpha_pred   # older format
pairs_dt    <- model_data$pairs_dt

cat(sprintf("  Convergence : %d  (objective = %.4f)\n",
            results$fit$convergence, results$fit$objective))
cat(sprintf("  Alpha estimates : %d sites\n", nrow(alpha_est)))
cat(sprintf("  Alpha range     : [%.2f, %.2f], mean = %.2f\n",
            min(alpha_est$alpha_est), max(alpha_est$alpha_est),
            mean(alpha_est$alpha_est)))


# ===========================================================================
# 4. Open PDF
# ===========================================================================
cat(sprintf("\n  Generating diagnostics → %s\n", pdf_file))
pdf(pdf_file, width = 12, height = 9)


# ===========================================================================
# A. MODEL DIAGNOSTICS
# ===========================================================================

# ---------------------------------------------------------------------------
# A1. Convergence & parameter summary text page
# ---------------------------------------------------------------------------
cat("--- A1. Convergence summary ---\n")

est  <- summary(results$sdreport, "report")

cat(sprintf("  alpha0        = %.4f (SE %.4f)\n", params$alpha0, params$alpha0_se))
cat(sprintf("  sigma_u       = %.4f\n", params$sigma_u))
cat(sprintf("  eta0 (exp)    = %.4f (SE %.4f)\n", params$eta0, params$eta0_se))

n_beta     <- length(params$beta)
beta_names <- if (!is.null(model_data$turnover_info$col_names))
                model_data$turnover_info$col_names else paste0("beta_", seq_len(n_beta))
for (i in seq_len(n_beta))
  cat(sprintf("  beta[%s] = %.6f (SE %.6f)\n", beta_names[i],
              params$beta[i], params$beta_se[i]))

par(mar = c(1, 1, 3, 1))
plot.new()
title(main = sprintf("CLESSO v2 - %s - Model Summary", species_grp), cex.main = 1.4)

summary_lines <- c(
  sprintf("Results file: %s", basename(results_file)),
  sprintf("Convergence: %d (%s)", results$fit$convergence, results$fit$message),
  sprintf("Final objective: %.4f", results$fit$objective),
  "",
  sprintf("alpha0 (intercept): %.4f  (SE %.4f)", params$alpha0, params$alpha0_se),
  sprintf("sigma_u (site RE SD): %.4f", params$sigma_u),
  sprintf("eta0 (turnover intercept): %.4f  (SE %.4f)", params$eta0, params$eta0_se),
  "",
  sprintf("Number of sites: %d", nrow(model_data$site_info$site_table)),
  sprintf("Alpha covariates (Z): %s",
          paste(model_data$alpha_info$cov_cols, collapse = ", ")),
  sprintf("Alpha splines: %s",
          if (!is.null(model_data$alpha_info$use_alpha_splines) &&
              model_data$alpha_info$use_alpha_splines) "enabled" else "disabled"),
  sprintf("Turnover variables: %d  (I-spline columns: %d)",
          length(model_data$turnover_info$splines),
          sum(model_data$turnover_info$splines)),
  "",
  "Beta (turnover) coefficients:"
)
for (i in seq_len(n_beta)) {
  summary_lines <- c(summary_lines,
    sprintf("  %s = %.6f (SE %.6f)", beta_names[i],
            params$beta[i], params$beta_se[i]))
}

alpha_info <- model_data$alpha_info
cov_cols   <- alpha_info$cov_cols
if (length(params$theta_alpha) > 0) {
  summary_lines <- c(summary_lines, "", "Theta (alpha linear coefficients):")
  for (i in seq_along(params$theta_alpha)) {
    nm <- if (!is.null(cov_cols)) cov_cols[i] else paste0("theta_", i)
    summary_lines <- c(summary_lines,
      sprintf("  %s = %.4f (SE %.4f)", nm, params$theta_alpha[i],
              params$theta_alpha_se[i]))
  }
}

## Add config info if we have it
if (!is.null(config_snap)) {
  summary_lines <- c(summary_lines, "",
    sprintf("Config snapshot: %s", format(config_snap$snapshot_time)),
    sprintf("alpha_lower_bound_lambda: %s",
            config_snap$alpha_lower_bound_lambda %||% "N/A"),
    sprintf("seed: %s", config_snap$seed %||% "N/A"))
}

text(0.05, 0.95, paste(summary_lines, collapse = "\n"),
     adj = c(0, 1), family = "mono", cex = 0.70)


# ---------------------------------------------------------------------------
# A2. Residual analysis
# ---------------------------------------------------------------------------
cat("--- A2. Residual analysis ---\n")

site_table <- model_data$site_info$site_table

## Compute training-set predictions from pre-built matrices
X_train    <- model_data$data_list$X
site_i_idx <- model_data$data_list$site_i  # 0-based
site_j_idx <- model_data$data_list$site_j
alpha_all  <- alpha_est$alpha_est
alpha_i    <- alpha_all[site_i_idx + 1L]
alpha_j    <- alpha_all[site_j_idx + 1L]

eta_train <- params$eta0 + as.numeric(X_train %*% params$beta)
S_train   <- exp(-eta_train)

p_hat <- S_train * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
p_hat <- pmin(pmax(p_hat, 0), 1)

y_obs   <- pairs_dt$y
resid_r <- y_obs - p_hat

par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

is_within  <- as.logical(pairs_dt$is_within)
is_between <- !is_within

hist(resid_r[is_within], breaks = 80, col = adjustcolor(pal_blue, 0.5),
     border = NA, main = "Residuals - Within-Site Pairs",
     xlab = "Observed - Predicted", ylab = "Frequency")
abline(v = 0, col = pal_red, lwd = 2, lty = 2)
mtext(sprintf("n = %d | mean = %.4f | sd = %.4f",
              sum(is_within), mean(resid_r[is_within], na.rm = TRUE),
              sd(resid_r[is_within], na.rm = TRUE)),
      side = 3, line = 0, cex = 0.7)

hist(resid_r[is_between], breaks = 80, col = adjustcolor(pal_orange, 0.5),
     border = NA, main = "Residuals - Between-Site Pairs",
     xlab = "Observed - Predicted", ylab = "Frequency")
abline(v = 0, col = pal_red, lwd = 2, lty = 2)
mtext(sprintf("n = %d | mean = %.4f | sd = %.4f",
              sum(is_between), mean(resid_r[is_between], na.rm = TRUE),
              sd(resid_r[is_between], na.rm = TRUE)),
      side = 3, line = 0, cex = 0.7)

## Observed vs predicted scatter (subsampled)
set.seed(123)
n_sub   <- min(20000, length(p_hat))
idx_sub <- sample(length(p_hat), n_sub)

plot(p_hat[idx_sub], y_obs[idx_sub],
     pch = 16, cex = 0.3,
     col = ifelse(is_within[idx_sub],
                  adjustcolor(pal_blue, 0.2),
                  adjustcolor(pal_orange, 0.2)),
     xlab = "Predicted p_match", ylab = "Observed (0/1)",
     main = "Observed vs Predicted")
abline(0, 1, col = pal_red, lwd = 2, lty = 2)
legend("topleft", legend = c("Within-site", "Between-site"),
       pch = 16, col = c(pal_blue, pal_orange), cex = 0.8, bg = "white")

## Predicted p_match density by pair type
d_within  <- density(p_hat[is_within],  from = 0, to = 1, na.rm = TRUE)
d_between <- density(p_hat[is_between], from = 0, to = 1, na.rm = TRUE)

plot(d_within, col = pal_blue, lwd = 2,
     xlim = c(0, 1), ylim = c(0, max(d_within$y, d_between$y) * 1.1),
     main = "Predicted p_match Distribution",
     xlab = "Predicted p_match", ylab = "Density")
lines(d_between, col = pal_orange, lwd = 2)
legend("topright", legend = c("Within-site", "Between-site"),
       col = c(pal_blue, pal_orange), lwd = 2, cex = 0.8, bg = "white")


# ---------------------------------------------------------------------------
# A3. AUC / classification accuracy
# ---------------------------------------------------------------------------
cat("--- A3. Classification metrics ---\n")

compute_auc <- function(pred, obs) {
  idx1 <- which(obs == 1); idx0 <- which(obs == 0)
  if (length(idx1) == 0 || length(idx0) == 0) return(NA)
  n1 <- min(length(idx1), 50000); n0 <- min(length(idx0), 50000)
  s1 <- pred[sample(idx1, n1)]; s0 <- pred[sample(idx0, n0)]
  mean(outer(s1, s0, ">")) + 0.5 * mean(outer(s1, s0, "=="))
}

auc_val <- tryCatch(compute_auc(1 - p_hat, y_obs), error = function(e) NA)
cat(sprintf("  AUC (mismatch prediction): %.4f\n", auc_val))

par(mfrow = c(1, 1), mar = c(5, 5, 4, 2))
thresh <- 0.5
y_pred_class <- as.integer(p_hat < thresh)
conf_mat     <- table(Predicted = y_pred_class, Observed = y_obs)
accuracy     <- sum(diag(conf_mat)) / sum(conf_mat)

plot.new()
title(main = sprintf("Classification Summary - %s", species_grp), cex.main = 1.2)
class_lines <- c(
  sprintf("AUC (mismatch prediction): %.4f", auc_val),
  sprintf("Accuracy at threshold %.2f: %.4f", thresh, accuracy),
  "", "Confusion matrix:",
  capture.output(print(conf_mat))
)
text(0.05, 0.8, paste(class_lines, collapse = "\n"),
     adj = c(0, 1), family = "mono", cex = 0.9)


# ---------------------------------------------------------------------------
# A4. Site random effects distribution
# ---------------------------------------------------------------------------
cat("--- A4. Random effects distribution ---\n")

par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

u_site <- params$u_site
if (length(u_site) > 0 && !all(u_site == 0)) {
  hist(u_site, breaks = 60, col = adjustcolor(pal_green, 0.5), border = NA,
       main = "Site Random Effects (u_i)",
       xlab = expression(u[i]), ylab = "Frequency")
  abline(v = 0, col = pal_red, lwd = 2, lty = 2)
  mtext(sprintf("sigma_u = %.4f | n = %d | mean = %.4f",
                params$sigma_u, length(u_site), mean(u_site)),
        side = 3, line = 0, cex = 0.7)

  qqnorm(u_site, main = "QQ Plot: Site Random Effects",
         pch = 16, cex = 0.4, col = adjustcolor(pal_green, 0.5))
  qqline(u_site, col = pal_red, lwd = 2)
} else {
  plot.new(); text(0.5, 0.5, "No site random effects estimated", cex = 1.2)
  plot.new()
}


# ---------------------------------------------------------------------------
# A5. Spatial map of fitted alpha
# ---------------------------------------------------------------------------
cat("--- A5. Spatial map of alpha ---\n")

par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

if (all(c("lon", "lat") %in% names(alpha_est))) {
  n_col <- 256
  alpha_breaks <- seq(min(alpha_est$alpha_est, na.rm = TRUE),
                      max(alpha_est$alpha_est, na.rm = TRUE),
                      length.out = n_col + 1)
  alpha_cols <- colorRampPalette(c("#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
                                    "#FC8D59", "#EF6548", "#D7301F", "#990000"))(n_col)
  alpha_cut <- cut(alpha_est$alpha_est, breaks = alpha_breaks,
                   include.lowest = TRUE, labels = FALSE)

  plot(alpha_est$lon, alpha_est$lat,
       pch = 15, cex = 0.5, col = alpha_cols[alpha_cut],
       xlab = "Longitude", ylab = "Latitude",
       main = sprintf("Fitted Alpha (Richness) - %s", species_grp), asp = 1)

  legend_vals <- pretty(range(alpha_est$alpha_est, na.rm = TRUE), n = 5)
  legend("bottomleft", legend = legend_vals,
         pch = 15, col = alpha_cols[findInterval(legend_vals, alpha_breaks)],
         cex = 0.65, bg = "white", title = "Alpha")
} else {
  plot.new()
  text(0.5, 0.5, "No lon/lat in alpha estimates", cex = 1.2)
}

hist(alpha_est$alpha_est, breaks = 60,
     col = adjustcolor(pal_orange, 0.5), border = NA,
     main = "Distribution of Fitted Alpha",
     xlab = "Alpha (species richness)", ylab = "Sites")
abline(v = mean(alpha_est$alpha_est), col = pal_red, lwd = 2, lty = 2)
mtext(sprintf("mean = %.1f | median = %.1f | range = [%.1f, %.1f]",
              mean(alpha_est$alpha_est), median(alpha_est$alpha_est),
              min(alpha_est$alpha_est), max(alpha_est$alpha_est)),
      side = 3, line = 0, cex = 0.7)

## Alpha vs S_obs (if available)
if ("S_obs" %in% names(alpha_est) && any(alpha_est$S_obs > 0)) {
  par(mfrow = c(1, 1), mar = c(5, 5, 4, 2))
  plot(alpha_est$S_obs, alpha_est$alpha_est,
       pch = 16, cex = 0.3, col = adjustcolor(pal_blue, 0.15),
       xlab = "Observed species count (S_obs)",
       ylab = expression(hat(alpha)),
       main = sprintf("Alpha vs S_obs - %s", species_grp))
  abline(0, 1, col = pal_red, lwd = 2, lty = 2)
  n_below <- sum(alpha_est$alpha_est < alpha_est$S_obs - 0.01)
  legend("topleft", legend = sprintf("%d sites with alpha < S_obs", n_below),
         text.col = pal_red, cex = 0.9, bg = "white")
}


# ===========================================================================
# B. FITTED FUNCTION SHAPES
# ===========================================================================

# ---------------------------------------------------------------------------
# B1. Alpha spline partial-effect curves g_k(z)
# ---------------------------------------------------------------------------
cat("\n--- B1. Alpha spline partial-effect curves ---\n")

use_splines <- !is.null(alpha_info$use_alpha_splines) && alpha_info$use_alpha_splines
spline_info <- alpha_info$spline_info
n_alpha_cov <- length(cov_cols)
Z_raw       <- model_data$site_info$Z_raw

if (use_splines && !is.null(spline_info) && length(params$b_alpha) > 0) {

  n_cov <- spline_info$n_covariates
  per_page <- 9  # 3x3 grid

  n_bases   <- spline_info$n_bases_per_cov
  cum_bases <- c(0, cumsum(n_bases))

  ## --- Pages: partial effects on log(alpha-1) scale ---
  for (k in seq_len(n_cov)) {
    if ((k - 1) %% per_page == 0)
      par(mfrow = c(3, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
    x_range <- range(Z_raw[, k], na.rm = TRUE)
    x_grid  <- seq(x_range[1], x_range[2], length.out = 300)

    B_k <- splines::bs(
      x_grid,
      knots          = spline_info$knot_list[[k]],
      degree         = spline_info$spline_deg,
      Boundary.knots = spline_info$boundary_list[[k]],
      intercept      = FALSE
    )

    idx_k <- (cum_bases[k] + 1):cum_bases[k + 1]
    b_k   <- params$b_alpha[idx_k]
    g_k   <- as.numeric(B_k %*% b_k)

    theta_k  <- params$theta_alpha[k]
    z_center <- alpha_info$z_center[k]
    z_scale  <- alpha_info$z_scale[k]
    lin_k    <- theta_k * (x_grid - z_center) / z_scale

    total_k <- lin_k + g_k

    y_range <- range(c(total_k, lin_k, g_k), na.rm = TRUE)
    pad <- diff(y_range) * 0.1; if (pad == 0) pad <- 0.5
    y_range <- y_range + c(-pad, pad)

    plot(x_grid, total_k, type = "l", col = pal_blue, lwd = 3,
         xlim = x_range, ylim = y_range,
         xlab = cov_cols[k], ylab = "Partial effect on log(alpha-1)",
         main = sprintf("g_%d: %s", k, cov_cols[k]))
    lines(x_grid, lin_k, col = pal_grey, lwd = 2, lty = 2)
    lines(x_grid, g_k, col = pal_orange, lwd = 2, lty = 3)
    abline(h = 0, col = "grey70", lty = 3)
    abline(v = spline_info$knot_list[[k]], col = "grey80", lty = 3, lwd = 0.5)
    rug(Z_raw[, k], col = adjustcolor("black", 0.05), quiet = TRUE)
    legend("topright",
           legend = c("Total (linear + spline)", "Linear only", "Spline only"),
           col = c(pal_blue, pal_grey, pal_orange),
           lwd = c(3, 2, 2), lty = c(1, 2, 3), cex = 0.65, bg = "white")
  }

  ## --- Pages: on alpha (richness) scale ---
  for (k in seq_len(n_cov)) {
    if ((k - 1) %% per_page == 0)
      par(mfrow = c(3, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
    x_range <- range(Z_raw[, k], na.rm = TRUE)
    x_grid  <- seq(x_range[1], x_range[2], length.out = 300)

    B_k <- splines::bs(
      x_grid,
      knots          = spline_info$knot_list[[k]],
      degree         = spline_info$spline_deg,
      Boundary.knots = spline_info$boundary_list[[k]],
      intercept      = FALSE
    )

    idx_k <- (cum_bases[k] + 1):cum_bases[k + 1]
    b_k   <- params$b_alpha[idx_k]
    g_k   <- as.numeric(B_k %*% b_k)

    theta_k  <- params$theta_alpha[k]
    z_center <- alpha_info$z_center[k]
    z_scale  <- alpha_info$z_scale[k]
    lin_k    <- theta_k * (x_grid - z_center) / z_scale

    total_k    <- lin_k + g_k
    alpha_pred_k <- exp(params$alpha0 + total_k) + 1

    plot(x_grid, alpha_pred_k, type = "l", col = pal_blue, lwd = 3,
         xlim = x_range,
         xlab = cov_cols[k], ylab = "Predicted alpha (richness)",
         main = sprintf("Alpha response: %s (others at mean)", cov_cols[k]))
    rug(Z_raw[, k], col = adjustcolor("black", 0.05), quiet = TRUE)
    abline(v = spline_info$knot_list[[k]], col = "grey80", lty = 3, lwd = 0.5)
  }

} else if (n_alpha_cov > 0) {
  ## No splines — linear-only alpha effects
  per_page <- 9
  for (k in seq_len(n_alpha_cov)) {
    if ((k - 1) %% per_page == 0)
      par(mfrow = c(3, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
    x_range <- range(Z_raw[, k], na.rm = TRUE)
    x_grid  <- seq(x_range[1], x_range[2], length.out = 300)
    theta_k  <- params$theta_alpha[k]
    z_center <- alpha_info$z_center[k]
    z_scale  <- alpha_info$z_scale[k]
    lin_k      <- theta_k * (x_grid - z_center) / z_scale
    alpha_pred_k <- exp(params$alpha0 + lin_k) + 1
    plot(x_grid, alpha_pred_k, type = "l", col = pal_blue, lwd = 3,
         xlim = x_range,
         xlab = cov_cols[k], ylab = "Predicted alpha (richness)",
         main = sprintf("Alpha response: %s (linear)", cov_cols[k]))
    rug(Z_raw[, k], col = adjustcolor("black", 0.05), quiet = TRUE)
  }
}


# ---------------------------------------------------------------------------
# B2. Beta (turnover) I-spline response curves
# ---------------------------------------------------------------------------
cat("--- B2. Beta I-spline response curves ---\n")

turnover_info  <- model_data$turnover_info
splines_vec    <- turnover_info$splines
quant_vec      <- turnover_info$quantiles
x_colnames     <- turnover_info$col_names
n_turn_vars    <- length(splines_vec)
var_base_names <- unique(gsub("_spl[0-9]+$", "", x_colnames))
csp            <- c(0, cumsum(splines_vec))

per_page_tv <- 9  # 3x3 grid
basis_cols <- c(pal_orange, pal_green, pal_red, "#7570B3", "#E7298A", "#66A61E")

for (v in seq_len(n_turn_vars)) {
  if ((v - 1) %% per_page_tv == 0)
    par(mfrow = c(3, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
  ns   <- splines_vec[v]
  quan <- quant_vec[(csp[v] + 1):(csp[v] + ns)]

  d_range <- c(0, max(quan) * 1.2)
  d_grid  <- seq(d_range[1], d_range[2], length.out = 300)

  spl_vals <- matrix(NA, nrow = length(d_grid), ncol = ns)
  for (sp in seq_len(ns)) {
    if (sp == 1) {
      q1 <- quan[1]; q2 <- quan[1]; q3 <- quan[2]
    } else if (sp == ns) {
      q1 <- quan[ns - 1]; q2 <- quan[ns]; q3 <- quan[ns]
    } else {
      q1 <- quan[sp - 1]; q2 <- quan[sp]; q3 <- quan[sp + 1]
    }
    spl_vals[, sp] <- I_spline(d_grid, q1, q2, q3)
  }

  beta_idx <- (csp[v] + 1):csp[v + 1]
  beta_v   <- params$beta[beta_idx]
  response <- spl_vals %*% beta_v

  y_max <- max(c(response, spl_vals %*% abs(beta_v)), na.rm = TRUE)
  if (y_max == 0) y_max <- 1

  plot(d_grid, response, type = "l", col = pal_blue, lwd = 3,
       xlim = d_range, ylim = c(0, y_max * 1.1),
       xlab = sprintf("|d %s|", var_base_names[v]),
       ylab = "Contribution to eta (turnover)",
       main = sprintf("Turnover response: %s", var_base_names[v]))

  for (sp in seq_len(ns)) {
    lines(d_grid, spl_vals[, sp] * beta_v[sp],
          col = adjustcolor(basis_cols[(sp - 1) %% length(basis_cols) + 1], 0.6),
          lwd = 1.5, lty = 2)
  }
  abline(v = quan, col = "grey70", lty = 3)
  mtext(sprintf("beta = [%s]", paste(sprintf("%.4f", beta_v), collapse = ", ")),
        side = 3, line = 0, cex = 0.55)
  legend("topleft",
         legend = c("Composite", paste0("Basis ", seq_len(ns), " (*b)")),
         col = c(pal_blue, basis_cols[seq_len(ns)]),
         lwd = c(3, rep(1.5, ns)), lty = c(1, rep(2, ns)),
         cex = 0.55, bg = "white")
}


# ---------------------------------------------------------------------------
# B3. Similarity decay curves: S = exp(-eta)
# ---------------------------------------------------------------------------
cat("--- B3. Composite turnover similarity curves ---\n")

for (v in seq_len(n_turn_vars)) {
  if ((v - 1) %% per_page_tv == 0)
    par(mfrow = c(3, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
  ns   <- splines_vec[v]
  quan <- quant_vec[(csp[v] + 1):(csp[v] + ns)]

  d_range <- c(0, max(quan) * 1.2)
  d_grid  <- seq(d_range[1], d_range[2], length.out = 300)

  spl_vals <- matrix(NA, nrow = length(d_grid), ncol = ns)
  for (sp in seq_len(ns)) {
    if (sp == 1) {
      q1 <- quan[1]; q2 <- quan[1]; q3 <- quan[2]
    } else if (sp == ns) {
      q1 <- quan[ns - 1]; q2 <- quan[ns]; q3 <- quan[ns]
    } else {
      q1 <- quan[sp - 1]; q2 <- quan[sp]; q3 <- quan[sp + 1]
    }
    spl_vals[, sp] <- I_spline(d_grid, q1, q2, q3)
  }

  beta_idx <- (csp[v] + 1):csp[v + 1]
  beta_v   <- params$beta[beta_idx]

  eta_v <- params$eta0 + as.numeric(spl_vals %*% beta_v)
  S_v   <- exp(-eta_v)

  plot(d_grid, S_v, type = "l", col = pal_blue, lwd = 3,
       xlim = d_range, ylim = c(0, 1),
       xlab = sprintf("|d %s|", var_base_names[v]),
       ylab = "Compositional Similarity (S)",
       main = sprintf("Similarity decay: %s", var_base_names[v]))
  abline(h = exp(-params$eta0), col = pal_grey, lty = 2, lwd = 1.5)
  abline(v = quan, col = "grey80", lty = 3)
  legend("topright",
         legend = c("S = exp(-eta)", sprintf("S0 = %.3f", exp(-params$eta0))),
         col = c(pal_blue, pal_grey), lwd = c(3, 1.5), lty = c(1, 2),
         cex = 0.7, bg = "white")
}


# ---------------------------------------------------------------------------
# B4. Beta coefficient bar chart
# ---------------------------------------------------------------------------
cat("--- B4. Beta coefficient bar chart ---\n")

par(mfrow = c(1, 1), mar = c(5, 10, 4, 2))

ord <- order(abs(params$beta))
bp <- barplot(params$beta[ord], horiz = TRUE, las = 1,
              names.arg = beta_names[ord],
              col = ifelse(params$beta[ord] > 0,
                          adjustcolor(pal_blue, 0.7),
                          adjustcolor(pal_red, 0.7)),
              border = NA,
              xlab = "Beta coefficient (exp-transformed)",
              main = sprintf("Turnover Coefficients - %s", species_grp),
              cex.names = 0.6)

## Draw error bars only where SE is finite
se_ord <- params$beta_se[ord]
se_ok  <- is.finite(se_ord)
if (any(se_ok)) {
  segments(params$beta[ord[se_ok]] - 1.96 * se_ord[se_ok], bp[se_ok],
           params$beta[ord[se_ok]] + 1.96 * se_ord[se_ok], bp[se_ok],
           col = pal_grey, lwd = 1.5)
}
abline(v = 0, col = pal_red, lwd = 1.5, lty = 2)


# ---------------------------------------------------------------------------
# B5. Variable importance (summed beta per variable)
# ---------------------------------------------------------------------------
cat("--- B5. Variable importance (summed beta) ---\n")

var_importance <- numeric(n_turn_vars)
names(var_importance) <- var_base_names

for (v in seq_len(n_turn_vars)) {
  beta_idx <- (csp[v] + 1):csp[v + 1]
  var_importance[v] <- sum(params$beta[beta_idx])
}

par(mfrow = c(1, 1), mar = c(5, 10, 4, 2))
ord_imp <- order(var_importance)

barplot(var_importance[ord_imp], horiz = TRUE, las = 1,
        col = colorRampPalette(c(pal_light, pal_blue))(n_turn_vars),
        border = NA,
        xlab = "Sum of I-spline coefficients",
        main = sprintf("Variable Importance (Turnover) - %s", species_grp),
        cex.names = 0.7)


# ---------------------------------------------------------------------------
# B6. Alpha effects with uncertainty ribbons
# ---------------------------------------------------------------------------
cat("--- B6. Alpha effects with uncertainty ribbons ---\n")

if (use_splines && !is.null(spline_info) && length(params$b_alpha) > 0) {

  ## Try to get b_alpha SEs from sdreport; fall back to NaN (no ribbons)
  b_alpha_se <- NULL
  if (!is.null(results$sdreport)) {
    est_all <- tryCatch(summary(results$sdreport, "report"), error = function(e) NULL)
    if (!is.null(est_all)) {
      b_alpha_rows <- grep("^b_alpha$", rownames(est_all))
      if (length(b_alpha_rows) > 0)
        b_alpha_se <- est_all[b_alpha_rows, "Std. Error"]
    }
  }
  ## If still NULL, try predictions$params (may have been stored)
  if (is.null(b_alpha_se)) {
    pp_se <- results$predictions$params$b_alpha_se
    if (!is.null(pp_se)) b_alpha_se <- pp_se
  }
  ## Default: zero SEs (no CI ribbon, but curves still drawn)
  if (is.null(b_alpha_se)) b_alpha_se <- rep(0, length(params$b_alpha))

  if (length(b_alpha_se) > 0) {
    b_alpha_se[!is.finite(b_alpha_se)] <- 0
    b_alpha_se[!is.finite(b_alpha_se)] <- 0

    n_cov <- spline_info$n_covariates
    per_page_ci <- 9  # 3x3 grid

    for (k in seq_len(n_cov)) {
      if ((k - 1) %% per_page_ci == 0)
        par(mfrow = c(3, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
      x_range <- range(Z_raw[, k], na.rm = TRUE)
      x_grid  <- seq(x_range[1], x_range[2], length.out = 300)

      B_k <- splines::bs(
        x_grid,
        knots          = spline_info$knot_list[[k]],
        degree         = spline_info$spline_deg,
        Boundary.knots = spline_info$boundary_list[[k]],
        intercept      = FALSE
      )

      idx_k <- (cum_bases[k] + 1):cum_bases[k + 1]
      b_k   <- params$b_alpha[idx_k]
      se_k  <- b_alpha_se[idx_k]

      g_k   <- as.numeric(B_k %*% b_k)
      g_se  <- sqrt(as.numeric(B_k^2 %*% se_k^2))
      g_se[!is.finite(g_se)] <- 0

      theta_k  <- params$theta_alpha[k]
      z_center <- alpha_info$z_center[k]
      z_scale  <- alpha_info$z_scale[k]
      lin_k    <- theta_k * (x_grid - z_center) / z_scale

      total_k <- lin_k + g_k
      upper   <- total_k + 1.96 * g_se
      lower   <- total_k - 1.96 * g_se

      finite_vals <- c(upper[is.finite(upper)], lower[is.finite(lower)],
                       total_k[is.finite(total_k)])
      if (length(finite_vals) == 0) { plot.new(); next }
      y_range <- range(finite_vals)
      pad <- diff(y_range) * 0.1; if (pad == 0) pad <- 0.5
      y_range <- y_range + c(-pad, pad)

      plot(x_grid, total_k, type = "n",
           xlim = x_range, ylim = y_range,
           xlab = cov_cols[k], ylab = "Partial effect on log(alpha-1)",
           main = sprintf("g_%d: %s (+/- 95%% CI)", k, cov_cols[k]))

      polygon(c(x_grid, rev(x_grid)), c(upper, rev(lower)),
              col = adjustcolor(pal_blue, 0.15), border = NA)
      lines(x_grid, total_k, col = pal_blue, lwd = 3)
      lines(x_grid, lin_k, col = pal_grey, lwd = 2, lty = 2)
      abline(h = 0, col = "grey70", lty = 3)
      rug(Z_raw[, k], col = adjustcolor("black", 0.05), quiet = TRUE)

      legend("topright",
             legend = c("Total (lin + spline)", "Linear only", "95% CI"),
             col = c(pal_blue, pal_grey, adjustcolor(pal_blue, 0.3)),
             lwd = c(3, 2, 8), lty = c(1, 2, 1), cex = 0.6, bg = "white")
    }
  }
}


# ===========================================================================
# Close PDF
# ===========================================================================
dev.off()

cat(sprintf("\n  Diagnostics saved to: %s\n", pdf_file))
cat("=== CLESSO standalone diagnostics complete ===\n")