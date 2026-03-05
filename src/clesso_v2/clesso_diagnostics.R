##############################################################################
##
## clesso_diagnostics.R -- Diagnostic and function-shape plots for CLESSO v2
##
## Loads a fitted CLESSO results object and generates:
##
##   A. MODEL DIAGNOSTICS
##      1. Convergence summary table
##      2. Residual histograms (within-site & between-site pairs)
##      3. Observed vs predicted p_match (scatter + hexbin)
##      4. Random effects (u_site) distribution
##      5. Spatial map of fitted alpha (richness)
##
##   B. FITTED FUNCTION SHAPES
##      6. Alpha spline partial-effect curves g_k(z) for each covariate
##      7. Beta (turnover) I-spline response curves per variable
##      8. Composite turnover function: eta as a function of env distance
##      9. Alpha–beta interaction surface (predicted p_match)
##
## Usage:
##   source("clesso_diagnostics.R")
##
## Reads the results from:
##   output/clesso_results_<species_group>.rds
##
##############################################################################

cat("=== CLESSO v2 Diagnostics & Function Shape Plots ===\n\n")

# ---------------------------------------------------------------------------
# 0. Source config and dependencies
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())

config_path <- file.path(this_dir, "clesso_config.R")
if (!file.exists(config_path)) {
  config_path <- file.path(this_dir, "src", "clesso_v2", "clesso_config.R")
}
source(config_path)

source(file.path(clesso_config$r_dir, "utils.R"))
source(file.path(clesso_config$r_dir, "gdm_functions.R"))
source(file.path(clesso_config$clesso_dir, "clesso_predict.R"))

library(data.table)
library(splines)

# ---------------------------------------------------------------------------
# 1. Load results
# ---------------------------------------------------------------------------
## Results file lookup: use run_id from config (which includes the timestamp).
## To load a specific prior run, set CLESSO_RUN_ID env var before sourcing
## the config, or set clesso_config$run_id manually.
## Also accept CLESSO_RESULTS_FILE env var for a direct path override.
results_file <- Sys.getenv("CLESSO_RESULTS_FILE", unset = "")
if (nchar(results_file) == 0) {
  results_file <- file.path(clesso_config$output_dir,
    paste0("clesso_results_", clesso_config$run_id, ".rds"))
}

if (!file.exists(results_file)) stop(paste("Results not found:", results_file))
results <- readRDS(results_file)

model_data  <- results$model_data
config      <- results$config
params      <- clesso_extract_params(results)

species_grp <- config$species_group
run_id      <- results$run_id %||% config$run_id %||% species_grp
out_dir     <- config$output_dir

cat(sprintf("  Loaded results for: %s  (run_id: %s)\n", species_grp, run_id))
cat(sprintf("  Convergence: %d  (objective = %.4f)\n",
            results$fit$convergence, results$fit$objective))

## Print config snapshot if available
if (!is.null(results$config_snapshot)) {
  cat("  Config snapshot recorded at:", format(results$config_snapshot$snapshot_time), "\n")
}

# ---------------------------------------------------------------------------
# Helper: nice colour palette
# ---------------------------------------------------------------------------
pal_blue   <- "#2166AC"
pal_red    <- "#B2182B"
pal_orange <- "#E08214"
pal_green  <- "#1B7837"
pal_grey   <- "#636363"
pal_light  <- "#D1E5F0"


# ===========================================================================
# A. MODEL DIAGNOSTICS
# ===========================================================================

pdf_diag <- file.path(out_dir, paste0(run_id, "_clesso_diagnostics.pdf"))
pdf(pdf_diag, width = 12, height = 9)

# ---------------------------------------------------------------------------
# A1. Convergence & parameter summary
# ---------------------------------------------------------------------------
cat("\n--- A1. Convergence summary ---\n")

est <- summary(results$sdreport, "report")

## Print key parameters
cat(sprintf("  alpha0        = %.4f (SE %.4f)\n", params$alpha0, params$alpha0_se))
cat(sprintf("  sigma_u       = %.4f\n", params$sigma_u))
cat(sprintf("  eta0 (exp)    = %.4f (SE %.4f)\n", params$eta0, params$eta0_se))
n_beta <- length(params$beta)
for (i in seq_len(n_beta)) {
  nm <- if (!is.null(model_data$turnover_info$col_names))
          model_data$turnover_info$col_names[i] else paste0("beta_", i)
  cat(sprintf("  beta[%s] = %.6f (SE %.6f)\n", nm, params$beta[i], params$beta_se[i]))
}

## Summary text page
par(mar = c(1, 1, 3, 1))
plot.new()
title(main = sprintf("CLESSO v2 - %s - Model Summary", species_grp), cex.main = 1.4)

summary_lines <- c(
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
  sprintf("Alpha splines: %s", if (model_data$alpha_info$use_alpha_splines) "enabled" else "disabled"),
  sprintf("Turnover variables: %d  (I-spline columns: %d)",
          length(model_data$turnover_info$splines),
          sum(model_data$turnover_info$splines)),
  "",
  "Beta (turnover) coefficients:"
)

for (i in seq_len(n_beta)) {
  nm <- if (!is.null(model_data$turnover_info$col_names))
          model_data$turnover_info$col_names[i] else paste0("beta_", i)
  summary_lines <- c(summary_lines,
    sprintf("  %s = %.6f (SE %.6f)", nm, params$beta[i], params$beta_se[i]))
}

if (length(params$theta_alpha) > 0) {
  summary_lines <- c(summary_lines, "", "Theta (alpha linear coefficients):")
  for (i in seq_along(params$theta_alpha)) {
    nm <- if (!is.null(model_data$alpha_info$cov_cols))
            model_data$alpha_info$cov_cols[i] else paste0("theta_", i)
    summary_lines <- c(summary_lines,
      sprintf("  %s = %.4f (SE %.4f)", nm, params$theta_alpha[i], params$theta_alpha_se[i]))
  }
}

text(0.05, 0.95, paste(summary_lines, collapse = "\n"),
     adj = c(0, 1), family = "mono", cex = 0.75)


# ---------------------------------------------------------------------------
# A2. Residual analysis
# ---------------------------------------------------------------------------
cat("--- A2. Residual analysis ---\n")

pairs_dt    <- model_data$pairs_dt
site_table  <- model_data$site_info$site_table

## Compute training-set predictions directly from pre-built matrices
## (avoids needing to reconstruct env_site_table for turnover prediction)
X_train <- model_data$data_list$X
Z_train <- model_data$data_list$Z

## Alpha per site: log(alpha-1) = alpha0 + Z*theta + B*b_alpha + u
site_i_idx <- model_data$data_list$site_i  # 0-based
site_j_idx <- model_data$data_list$site_j
alpha_all  <- results$alpha_estimates$alpha_est      # already on natural scale
alpha_i    <- alpha_all[site_i_idx + 1L]
alpha_j    <- alpha_all[site_j_idx + 1L]

## Turnover: eta = eta0 + X*beta,  S = exp(-eta)
eta_train <- params$eta0 + as.numeric(X_train %*% params$beta)
S_train   <- exp(-eta_train)

## Full probability of match: p = S * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
p_hat <- S_train * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
p_hat <- pmin(pmax(p_hat, 0), 1)

y_obs   <- pairs_dt$y  # 0 = match, 1 = mismatch
resid_r <- y_obs - p_hat

## --- Residual histogram ---
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

## --- Observed vs predicted scatter ---
## Subsample for readability
set.seed(123)
n_sub <- min(20000, length(p_hat))
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

## --- Predicted p_match distribution by pair type ---
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

## Simple AUC approximation (Wilcoxon–Mann–Whitney)
compute_auc <- function(pred, obs) {
  ## obs = 1 for "event" (match), 0 for non-event
  idx1 <- which(obs == 1)
  idx0 <- which(obs == 0)
  if (length(idx1) == 0 || length(idx0) == 0) return(NA)

  ## Sample for speed
  n1 <- min(length(idx1), 50000)
  n0 <- min(length(idx0), 50000)
  s1 <- pred[sample(idx1, n1)]
  s0 <- pred[sample(idx0, n0)]

  ## Proportion of concordant pairs
  mean(outer(s1, s0, ">")) + 0.5 * mean(outer(s1, s0, "=="))
}

## For p_match, higher means more likely to be a match (y=0 in CLESSO convention?)
## Let's determine convention from data: check if matches have lower or higher y
## CLESSO: y=0 means match (same species), y=1 means mismatch
## So for AUC re "correctly classifying matches", use 1-p_hat for y
auc_val <- tryCatch(compute_auc(1 - p_hat, y_obs), error = function(e) NA)
cat(sprintf("  AUC (mismatch prediction): %.4f\n", auc_val))

par(mfrow = c(1, 1), mar = c(5, 5, 4, 2))

## Classification performance at threshold = 0.5
thresh <- 0.5
y_pred_class <- as.integer(p_hat < thresh)  # predict mismatch if p_match < 0.5
conf_mat <- table(Predicted = y_pred_class, Observed = y_obs)
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)

plot.new()
title(main = sprintf("Classification Summary - %s", species_grp), cex.main = 1.2)
class_lines <- c(
  sprintf("AUC (mismatch prediction): %.4f", auc_val),
  sprintf("Accuracy at threshold %.2f: %.4f", thresh, accuracy),
  "",
  "Confusion matrix:",
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

  ## QQ plot
  qqnorm(u_site, main = "QQ Plot: Site Random Effects",
         pch = 16, cex = 0.4, col = adjustcolor(pal_green, 0.5))
  qqline(u_site, col = pal_red, lwd = 2)
} else {
  plot.new()
  text(0.5, 0.5, "No site random effects estimated", cex = 1.2)
  plot.new()
}


# ---------------------------------------------------------------------------
# A5. Spatial map of fitted alpha
# ---------------------------------------------------------------------------
cat("--- A5. Spatial map of alpha ---\n")

alpha_est <- results$alpha_estimates

par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

## Alpha on natural scale
n_col <- 256
alpha_breaks <- seq(min(alpha_est$alpha_est, na.rm = TRUE),
                    max(alpha_est$alpha_est, na.rm = TRUE),
                    length.out = n_col + 1)
alpha_cols <- colorRampPalette(c("#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
                                  "#FC8D59", "#EF6548", "#D7301F", "#990000"))(n_col)
alpha_cut <- cut(alpha_est$alpha_est, breaks = alpha_breaks, include.lowest = TRUE, labels = FALSE)

plot(alpha_est$lon, alpha_est$lat,
     pch = 15, cex = 0.5,
     col = alpha_cols[alpha_cut],
     xlab = "Longitude", ylab = "Latitude",
     main = sprintf("Fitted Alpha (Richness) - %s", species_grp),
     asp = 1)

## Add colour bar
legend_vals <- pretty(range(alpha_est$alpha_est, na.rm = TRUE), n = 5)
legend("bottomleft",
       legend = legend_vals,
       pch = 15, col = alpha_cols[findInterval(legend_vals, alpha_breaks)],
       cex = 0.65, bg = "white", title = "Alpha")

## Alpha histogram
hist(alpha_est$alpha_est, breaks = 60,
     col = adjustcolor(pal_orange, 0.5), border = NA,
     main = "Distribution of Fitted Alpha",
     xlab = "Alpha (species richness)", ylab = "Sites")
abline(v = mean(alpha_est$alpha_est), col = pal_red, lwd = 2, lty = 2)
mtext(sprintf("mean = %.1f | median = %.1f | range = [%.1f, %.1f]",
              mean(alpha_est$alpha_est), median(alpha_est$alpha_est),
              min(alpha_est$alpha_est), max(alpha_est$alpha_est)),
      side = 3, line = 0, cex = 0.7)


# ===========================================================================
# B. FITTED FUNCTION SHAPES
# ===========================================================================

# ---------------------------------------------------------------------------
# B1. Alpha spline partial-effect curves g_k(z)
# ---------------------------------------------------------------------------
cat("\n--- B1. Alpha spline partial-effect curves ---\n")

alpha_info  <- model_data$alpha_info
use_splines <- alpha_info$use_alpha_splines
spline_info <- alpha_info$spline_info
cov_cols    <- alpha_info$cov_cols
n_alpha_cov <- length(cov_cols)

if (use_splines && !is.null(spline_info) && length(params$b_alpha) > 0) {

  ## Number of covariates (possibly > 2 if substrate variables exist)
  n_cov <- spline_info$n_covariates

  ## Determine grid layout
  n_cols_layout <- min(n_cov, 3)
  n_rows_layout <- ceiling(n_cov / n_cols_layout)
  par(mfrow = c(n_rows_layout, n_cols_layout), mar = c(5, 5, 4, 2))

  ## B coefficients are stacked: first n_bases[1] for cov 1, then n_bases[2] for cov 2, etc.
  n_bases <- spline_info$n_bases_per_cov
  cum_bases <- c(0, cumsum(n_bases))

  ## Get raw covariate values for x-axis ranges
  Z_raw <- model_data$site_info$Z_raw

  for (k in seq_len(n_cov)) {
    x_range <- range(Z_raw[, k], na.rm = TRUE)
    x_grid  <- seq(x_range[1], x_range[2], length.out = 300)

    ## Build B-spline basis on the grid
    B_k <- splines::bs(
      x_grid,
      knots          = spline_info$knot_list[[k]],
      degree         = spline_info$spline_deg,
      Boundary.knots = spline_info$boundary_list[[k]],
      intercept      = FALSE
    )

    ## Extract b_alpha coefficients for this covariate
    idx_k <- (cum_bases[k] + 1):cum_bases[k + 1]
    b_k   <- params$b_alpha[idx_k]

    ## Partial effect: g_k(z) = B_k(z) %*% b_k
    g_k <- as.numeric(B_k %*% b_k)

    ## Also compute linear-only effect for comparison
    theta_k  <- params$theta_alpha[k]
    z_center <- alpha_info$z_center[k]
    z_scale  <- alpha_info$z_scale[k]
    lin_k    <- theta_k * (x_grid - z_center) / z_scale

    ## Combined partial effect (linear + spline)
    total_k <- lin_k + g_k

    ## Plot
    y_range <- range(c(total_k, lin_k, g_k), na.rm = TRUE)
    pad <- diff(y_range) * 0.1
    if (pad == 0) pad <- 0.5
    y_range <- y_range + c(-pad, pad)

    plot(x_grid, total_k, type = "l", col = pal_blue, lwd = 3,
         xlim = x_range, ylim = y_range,
         xlab = cov_cols[k], ylab = "Partial effect on log(alpha-1)",
         main = sprintf("g_%d: %s", k, cov_cols[k]))

    lines(x_grid, lin_k, col = pal_grey, lwd = 2, lty = 2)
    lines(x_grid, g_k, col = pal_orange, lwd = 2, lty = 3)
    abline(h = 0, col = "grey70", lty = 3)

    ## Show knot positions
    abline(v = spline_info$knot_list[[k]], col = "grey80", lty = 3, lwd = 0.5)

    ## Rug of data values
    rug(Z_raw[, k], col = adjustcolor("black", 0.05), quiet = TRUE)

    legend("topright",
           legend = c("Total (linear + spline)", "Linear only", "Spline only"),
           col = c(pal_blue, pal_grey, pal_orange),
           lwd = c(3, 2, 2), lty = c(1, 2, 3), cex = 0.65, bg = "white")
  }

  ## --- Also show on alpha scale (exp(g) + 1 contribution) ---
  par(mfrow = c(n_rows_layout, n_cols_layout), mar = c(5, 5, 4, 2))

  for (k in seq_len(n_cov)) {
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

    ## On alpha scale, show exp(alpha0 + total_k) + 1
    ## (holding other covariates at their mean = 0 after standardization)
    total_k    <- lin_k + g_k
    alpha_pred <- exp(params$alpha0 + total_k) + 1

    plot(x_grid, alpha_pred, type = "l", col = pal_blue, lwd = 3,
         xlim = x_range,
         xlab = cov_cols[k], ylab = "Predicted alpha (richness)",
         main = sprintf("Alpha response: %s (others at mean)", cov_cols[k]))

    rug(Z_raw[, k], col = adjustcolor("black", 0.05), quiet = TRUE)
    abline(v = spline_info$knot_list[[k]], col = "grey80", lty = 3, lwd = 0.5)
  }

} else {
  ## No splines -- just show linear effects on alpha
  par(mfrow = c(1, min(n_alpha_cov, 3)), mar = c(5, 5, 4, 2))

  Z_raw <- model_data$site_info$Z_raw

  for (k in seq_len(n_alpha_cov)) {
    x_range <- range(Z_raw[, k], na.rm = TRUE)
    x_grid  <- seq(x_range[1], x_range[2], length.out = 300)

    theta_k  <- params$theta_alpha[k]
    z_center <- alpha_info$z_center[k]
    z_scale  <- alpha_info$z_scale[k]

    lin_k      <- theta_k * (x_grid - z_center) / z_scale
    alpha_pred <- exp(params$alpha0 + lin_k) + 1

    plot(x_grid, alpha_pred, type = "l", col = pal_blue, lwd = 3,
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

turnover_info <- model_data$turnover_info
splines_vec   <- turnover_info$splines
quant_vec     <- turnover_info$quantiles
x_colnames    <- turnover_info$col_names
n_turn_vars   <- length(splines_vec)

## Extract variable base names from column names
## e.g., "subs_1_spl1", "subs_1_spl2", "subs_1_spl3" -> "subs_1"
var_base_names <- unique(gsub("_spl[0-9]+$", "", x_colnames))

## Cumulative spline indices
csp <- c(0, cumsum(splines_vec))

n_cols_layout <- min(n_turn_vars, 3)
n_rows_layout <- ceiling(n_turn_vars / n_cols_layout)
par(mfrow = c(n_rows_layout, n_cols_layout), mar = c(5, 5, 4, 2))

for (v in seq_len(n_turn_vars)) {
  ns   <- splines_vec[v]
  quan <- quant_vec[(csp[v] + 1):(csp[v] + ns)]

  ## Generate a grid of environmental distances for this variable
  ## The I-spline maps raw difference to a monotonic basis
  d_range <- c(0, max(quan) * 1.2)
  d_grid  <- seq(d_range[1], d_range[2], length.out = 300)

  ## Compute I-spline basis values at each grid point
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

  ## Get the beta coefficients for this variable's splines
  beta_idx <- (csp[v] + 1):csp[v + 1]
  beta_v   <- params$beta[beta_idx]

  ## Composite response: weighted sum of I-spline bases
  response <- spl_vals %*% beta_v

  ## Plot individual spline bases (scaled by beta)
  y_max <- max(c(response, spl_vals %*% abs(beta_v)), na.rm = TRUE)
  if (y_max == 0) y_max <- 1

  plot(d_grid, response, type = "l", col = pal_blue, lwd = 3,
       xlim = d_range, ylim = c(0, y_max * 1.1),
       xlab = sprintf("|d %s|", var_base_names[v]),
       ylab = "Contribution to eta (turnover)",
       main = sprintf("Turnover response: %s", var_base_names[v]))

  ## Show individual basis functions (scaled)
  basis_cols <- c(pal_orange, pal_green, pal_red, "#7570B3", "#E7298A", "#66A61E")
  for (sp in seq_len(ns)) {
    lines(d_grid, spl_vals[, sp] * beta_v[sp],
          col = adjustcolor(basis_cols[(sp - 1) %% length(basis_cols) + 1], 0.6),
          lwd = 1.5, lty = 2)
  }

  ## Mark quantile positions
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
# B3. Full turnover response: S = exp(-eta) vs total env distance
# ---------------------------------------------------------------------------
cat("--- B3. Composite turnover similarity curve ---\n")

## For an intuitive plot, compute S along one variable at a time
## while holding others at zero (i.e., same-site for that variable)

par(mfrow = c(n_rows_layout, n_cols_layout), mar = c(5, 5, 4, 2))

for (v in seq_len(n_turn_vars)) {
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

beta_vals <- params$beta
beta_ses  <- params$beta_se
beta_names <- if (!is.null(model_data$turnover_info$col_names)) {
                model_data$turnover_info$col_names
              } else paste0("beta_", seq_along(beta_vals))

## Order by magnitude
ord <- order(abs(beta_vals))
bp <- barplot(beta_vals[ord], horiz = TRUE, las = 1,
              names.arg = beta_names[ord],
              col = ifelse(beta_vals[ord] > 0,
                          adjustcolor(pal_blue, 0.7),
                          adjustcolor(pal_red, 0.7)),
              border = NA,
              xlab = "Beta coefficient (exp-transformed)",
              main = sprintf("Turnover Coefficients - %s", species_grp),
              cex.names = 0.6)

## Error bars
segments(beta_vals[ord] - 1.96 * beta_ses[ord], bp,
         beta_vals[ord] + 1.96 * beta_ses[ord], bp,
         col = pal_grey, lwd = 1.5)
abline(v = 0, col = pal_red, lwd = 1.5, lty = 2)


# ---------------------------------------------------------------------------
# B5. Variable importance: sum of I-spline coefficients per variable
# ---------------------------------------------------------------------------
cat("--- B5. Variable importance (summed beta) ---\n")

## Sum absolute beta coefficients per variable as a simple importance metric
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
# B6. Alpha partial-effect curves on log(alpha) scale with uncertainty
# ---------------------------------------------------------------------------
cat("--- B6. Alpha effects with uncertainty ribbons ---\n")

## Use delta method approximation for spline uncertainty
## SE of g_k(z) at point x = |B_k(x)| * SE(b_k)
## This is approximate but useful

if (use_splines && !is.null(spline_info) && length(params$b_alpha) > 0) {

  ## Get SEs for b_alpha from sdreport
  est_all <- summary(results$sdreport, "report")
  b_alpha_rows <- grep("^b_alpha$", rownames(est_all))

  if (length(b_alpha_rows) > 0) {
    b_alpha_se <- est_all[b_alpha_rows, "Std. Error"]
    ## Replace any non-finite SEs with 0 (no ribbon for those)
    b_alpha_se[!is.finite(b_alpha_se)] <- 0

    n_cov <- spline_info$n_covariates
    n_cols_layout <- min(n_cov, 3)
    n_rows_layout <- ceiling(n_cov / n_cols_layout)
    par(mfrow = c(n_rows_layout, n_cols_layout), mar = c(5, 5, 4, 2))

    for (k in seq_len(n_cov)) {
      x_range <- range(Z_raw[, k], na.rm = TRUE)
      x_grid  <- seq(x_range[1], x_range[2], length.out = 300)

      B_k <- splines::bs(
        x_grid,
        knots          = spline_info$knot_list[[k]],
        degree         = spline_info$spline_deg,
        Boundary.knots = spline_info$boundary_list[[k]],
        intercept      = FALSE
      )

      idx_k   <- (cum_bases[k] + 1):cum_bases[k + 1]
      b_k     <- params$b_alpha[idx_k]
      se_k    <- b_alpha_se[idx_k]

      g_k     <- as.numeric(B_k %*% b_k)
      ## Approximate SE of g_k(z): sqrt(B^2 %*% se^2)
      g_se    <- sqrt(as.numeric(B_k^2 %*% se_k^2))
      g_se[!is.finite(g_se)] <- 0

      theta_k  <- params$theta_alpha[k]
      z_center <- alpha_info$z_center[k]
      z_scale  <- alpha_info$z_scale[k]
      lin_k    <- theta_k * (x_grid - z_center) / z_scale

      total_k <- lin_k + g_k
      upper   <- total_k + 1.96 * g_se
      lower   <- total_k - 1.96 * g_se

      ## Guard against non-finite ylim
      finite_vals <- c(upper[is.finite(upper)], lower[is.finite(lower)],
                       total_k[is.finite(total_k)])
      if (length(finite_vals) == 0) { plot.new(); next }
      y_range <- range(finite_vals)
      pad <- diff(y_range) * 0.1
      if (pad == 0) pad <- 0.5
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


# ---------------------------------------------------------------------------
# Close PDF
# ---------------------------------------------------------------------------
dev.off()
cat(sprintf("\n  Saved diagnostics PDF: %s\n", basename(pdf_diag)))

cat(sprintf("\n=== CLESSO v2 diagnostics complete ===\n"))
