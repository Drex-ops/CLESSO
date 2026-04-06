##############################################################################
##
## plot_coefficient_barplots.R
##
## For each model setting type (Base, MODIS, Condition), produce grouped
## barplots showing coefficient effect sizes (sum of I-spline coefficients
## per predictor) across all biological groups, plus a model-fit summary.
##
## Outputs to: output/model_output_viz/
##
## Usage:
##   Rscript plot_coefficient_barplots.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Coefficient Effect-Size Barplots\n")
cat("###########################################################################\n\n")

this_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("--file=", "", file_arg[1])))
  } else {
    getwd()
  }
})
output_dir <- file.path(this_dir, "output")
viz_dir    <- file.path(output_dir, "model_output_viz")
if (!dir.exists(viz_dir)) dir.create(viz_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Discover run folders
# ---------------------------------------------------------------------------
groups <- c("AMP", "AVES", "HYM", "MAM", "REP", "VAS")

group_labels <- c(
  AMP  = "Amphibians",
  AVES = "Birds",
  HYM  = "Hymenoptera",
  MAM  = "Mammals",
  REP  = "Reptiles",
  VAS  = "Vascular Plants"
)

group_colours <- c(
  AMP  = "#E41A1C",
  AVES = "#377EB8",
  HYM  = "#FF7F00",
  MAM  = "#4DAF4A",
  REP  = "#984EA3",
  VAS  = "#A65628"
)

model_types <- c("base", "MODIS", "COND")
model_labels <- c(base = "Base (climate + substrate only)",
                  MODIS = "MODIS (+ remote sensing)",
                  COND  = "Condition (+ landscape condition)")

find_run_folder <- function(group, variant = "base") {
  pattern <- switch(variant,
    base  = sprintf("^%s_\\d{8}T\\d{6}$", group),
    COND  = sprintf("^%s_COND_\\d{8}T\\d{6}$", group),
    MODIS = sprintf("^%s_MODIS_\\d{8}T\\d{6}$", group)
  )
  dirs <- list.dirs(output_dir, full.names = FALSE, recursive = FALSE)
  matches <- dirs[grepl(pattern, dirs)]
  if (length(matches) == 0) return(NULL)
  file.path(output_dir, sort(matches, decreasing = TRUE)[1])
}

# ---------------------------------------------------------------------------
# 2. Load all fitted GDMs and extract effect sizes
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDMs ---\n")

## Helper: extract effect sizes from a fit object
extract_effects <- function(fit) {
  predictors  <- fit$predictors
  n_splines   <- fit$splines
  coefs       <- fit$coefficients
  n_pred      <- length(predictors)

  effects <- numeric(n_pred)
  names(effects) <- predictors
  idx <- 1
  for (i in seq_len(n_pred)) {
    ns <- n_splines[i]
    spline_vals <- coefs[idx:(idx + ns - 1)]
    ## Only sum finite values; replace Inf/NaN with 0
    spline_vals[!is.finite(spline_vals)] <- 0
    effects[i] <- sum(spline_vals)
    idx <- idx + ns
  }
  effects
}

## Storage: list of model_type -> list of group -> effects vector
all_effects <- list()
all_metadata <- list()

for (mt in model_types) {
  all_effects[[mt]] <- list()
  all_metadata[[mt]] <- list()

  for (grp in groups) {
    run_dir <- find_run_folder(grp, mt)
    if (is.null(run_dir)) next

    ## Find the fittedGDM file
    fit_files <- list.files(run_dir, pattern = "fittedGDM\\.RData$", full.names = TRUE)
    if (length(fit_files) == 0) next

    env <- new.env()
    load(fit_files[1], envir = env)
    if (!exists("fit", envir = env)) next

    fit <- env$fit
    effects <- extract_effects(fit)

    all_effects[[mt]][[grp]] <- effects
    all_metadata[[mt]][[grp]] <- list(
      D2 = fit$D2,
      R2 = fit$nagelkerke_r2,
      n_pairs = fit$n_pairs,
      n_pred = length(fit$predictors),
      intercept = fit$intercept
    )

    cat(sprintf("  %-5s %-6s : %d predictors, D2=%.4f, R2=%.4f\n",
                grp, mt, length(fit$predictors), fit$D2, fit$nagelkerke_r2))
  }
}
cat("\n")

# ---------------------------------------------------------------------------
# 3. Categorise predictors for cleaner labels
# ---------------------------------------------------------------------------
## Create human-readable labels and domain categories
predictor_labels <- c(
  spat_mean_mean_PT_1   = "Mean Precip (spatial)",
  spat_min_TNn_1        = "Min TNn (spatial)",
  spat_min_FWPT_1       = "Min FWPT (spatial)",
  spat_max_max_PT_1     = "Max Precip (spatial)",
  spat_max_FWPT_1       = "Max FWPT (spatial)",
  spat_max_FD_1         = "Max Frost Days (spatial)",
  spat_max_TXx_1        = "Max TXx (spatial)",
  spat_max_TNn_1        = "Max TNn (spatial)",
  spat_max_PD_1         = "Max Precip Days (spatial)",
  SLT_mean_1            = "Silt",
  AWC_mean_1            = "Avail. Water Cap.",
  PHC_mean_1            = "pH (mean)",
  ECE_mean_1            = "Elec. Conductivity",
  ELVR1000_range_1      = "Elevation Range",
  NTO_mean_1            = "Total Nitrogen",
  PHC_range_1           = "pH (range)",
  SND_range_1           = "Sand (range)",
  temp_mean_mean_PT_1   = "Mean Precip (temporal)",
  temp_min_TNn_1        = "Min TNn (temporal)",
  temp_min_FWPT_1       = "Min FWPT (temporal)",
  temp_max_max_PT_1     = "Max Precip (temporal)",
  temp_max_FWPT_1       = "Max FWPT (temporal)",
  temp_max_FD_1         = "Max Frost Days (temporal)",
  temp_max_TXx_1        = "Max TXx (temporal)",
  temp_max_TNn_1        = "Max TNn (temporal)",
  temp_max_PD_1         = "Max Precip Days (temporal)",
  spat_modis_nontree_1  = "MODIS Non-tree (spatial)",
  spat_modis_nonveget_1 = "MODIS Non-veg (spatial)",
  spat_modis_treecov_1  = "MODIS Tree Cov (spatial)",
  temp_modis_nontree_1  = "MODIS Non-tree (temporal)",
  temp_modis_nonveget_1 = "MODIS Non-veg (temporal)",
  temp_modis_treecov_1  = "MODIS Tree Cov (temporal)",
  spat_condition_1      = "Condition (spatial)",
  temp_condition_1      = "Condition (temporal)"
)

predictor_domain <- c(
  spat_mean_mean_PT_1   = "Climate (spatial)",
  spat_min_TNn_1        = "Climate (spatial)",
  spat_min_FWPT_1       = "Climate (spatial)",
  spat_max_max_PT_1     = "Climate (spatial)",
  spat_max_FWPT_1       = "Climate (spatial)",
  spat_max_FD_1         = "Climate (spatial)",
  spat_max_TXx_1        = "Climate (spatial)",
  spat_max_TNn_1        = "Climate (spatial)",
  spat_max_PD_1         = "Climate (spatial)",
  SLT_mean_1            = "Substrate",
  AWC_mean_1            = "Substrate",
  PHC_mean_1            = "Substrate",
  ECE_mean_1            = "Substrate",
  ELVR1000_range_1      = "Substrate",
  NTO_mean_1            = "Substrate",
  PHC_range_1           = "Substrate",
  SND_range_1           = "Substrate",
  temp_mean_mean_PT_1   = "Climate (temporal)",
  temp_min_TNn_1        = "Climate (temporal)",
  temp_min_FWPT_1       = "Climate (temporal)",
  temp_max_max_PT_1     = "Climate (temporal)",
  temp_max_FWPT_1       = "Climate (temporal)",
  temp_max_FD_1         = "Climate (temporal)",
  temp_max_TXx_1        = "Climate (temporal)",
  temp_max_TNn_1        = "Climate (temporal)",
  temp_max_PD_1         = "Climate (temporal)",
  spat_modis_nontree_1  = "MODIS (spatial)",
  spat_modis_nonveget_1 = "MODIS (spatial)",
  spat_modis_treecov_1  = "MODIS (spatial)",
  temp_modis_nontree_1  = "MODIS (temporal)",
  temp_modis_nonveget_1 = "MODIS (temporal)",
  temp_modis_treecov_1  = "MODIS (temporal)",
  spat_condition_1      = "Condition (spatial)",
  temp_condition_1      = "Condition (temporal)"
)

domain_colours <- c(
  "Climate (spatial)"   = "#2166AC",
  "Climate (temporal)"  = "#92C5DE",
  "Substrate"           = "#8C510A",
  "MODIS (spatial)"     = "#1B7837",
  "MODIS (temporal)"    = "#7FBF7B",
  "Condition (spatial)"  = "#C51B7D",
  "Condition (temporal)" = "#DE77AE"
)

get_label <- function(pred) {
  lab <- predictor_labels[pred]
  ifelse(is.na(lab), pred, lab)
}

# ===========================================================================
# 4. Generate barplots -- one PDF per model type
# ===========================================================================

for (mt in model_types) {
  effects_list <- all_effects[[mt]]
  meta_list    <- all_metadata[[mt]]

  if (length(effects_list) == 0) {
    cat(sprintf("  [SKIP] %s -- no fitted models found\n", mt))
    next
  }

  active_groups <- names(effects_list)
  n_groups <- length(active_groups)

  ## Union of all predictors across groups for this model type
  all_preds <- unique(unlist(lapply(effects_list, names)))

  ## Order predictors by domain, then by name
  pred_domains <- predictor_domain[all_preds]
  pred_domains[is.na(pred_domains)] <- "Other"
  domain_order <- c("Climate (spatial)", "Substrate", "Climate (temporal)",
                    "MODIS (spatial)", "MODIS (temporal)",
                    "Condition (spatial)", "Condition (temporal)", "Other")
  pred_order <- order(match(pred_domains, domain_order), all_preds)
  all_preds <- all_preds[pred_order]
  n_pred <- length(all_preds)

  ## Build effects matrix [n_groups x n_pred]
  effects_mat <- matrix(0, nrow = n_groups, ncol = n_pred,
                        dimnames = list(active_groups, all_preds))
  for (grp in active_groups) {
    eff <- effects_list[[grp]]
    shared <- intersect(names(eff), all_preds)
    effects_mat[grp, shared] <- eff[shared]
  }

  ## Cap extreme values (numerical instability in some base models)
  cap_val <- max(effects_mat[effects_mat < 10], na.rm = TRUE) * 1.1
  effects_mat[effects_mat > cap_val] <- cap_val

  ## Pretty labels
  pred_labels <- sapply(all_preds, get_label)

  ## ---- PDF ----
  pdf_file <- file.path(viz_dir, sprintf("coefficient_effects_%s.pdf", mt))
  pdf(pdf_file, width = 16, height = max(10, n_pred * 0.4))

  ## --- Plot 1: Grouped horizontal barplot ---
  par(mar = c(5, 12, 5, 2))

  ## Transpose so predictors are along y-axis, groups are side-by-side
  bp_mat <- t(effects_mat[rev(active_groups), , drop = FALSE])

  ## Colour bars by group
  bp_cols <- group_colours[rev(active_groups)]

  barplot(t(bp_mat), beside = TRUE, horiz = TRUE,
          names.arg = pred_labels,
          col = bp_cols,
          border = NA,
          las = 1, cex.names = 0.7, cex.axis = 0.8,
          xlab = "Effect Size (sum of I-spline coefficients)",
          main = sprintf("GDM Coefficient Effect Sizes -- %s\n%d biological groups",
                         model_labels[mt], n_groups))

  ## Domain separators
  ## Add domain colour indicators on the left margin
  domains_ordered <- predictor_domain[all_preds]
  domains_ordered[is.na(domains_ordered)] <- "Other"

  legend("topright",
         legend = c(group_labels[rev(active_groups)]),
         fill = bp_cols,
         border = NA, cex = 0.75, bg = "white",
         title = "Biological Group")

  ## --- Plot 2: Faceted by domain category ---
  unique_domains <- unique(domains_ordered)
  unique_domains <- intersect(domain_order, unique_domains)  # maintain order

  par(mfrow = c(length(unique_domains), 1),
      mar = c(3, 12, 2, 1), oma = c(3, 0, 4, 0))

  for (dom in unique_domains) {
    dom_preds <- all_preds[domains_ordered == dom]
    dom_labels <- sapply(dom_preds, get_label)
    n_dom <- length(dom_preds)

    dom_mat <- effects_mat[active_groups, dom_preds, drop = FALSE]
    bp_mat2 <- t(dom_mat[rev(active_groups), , drop = FALSE])

    barplot(t(bp_mat2), beside = TRUE, horiz = TRUE,
            names.arg = dom_labels,
            col = group_colours[rev(active_groups)],
            border = NA,
            las = 1, cex.names = 0.75, cex.axis = 0.75,
            main = dom)
  }

  mtext(sprintf("GDM Coefficient Effect Sizes by Domain -- %s", model_labels[mt]),
        outer = TRUE, cex = 1.1, line = 1.5)
  mtext("Effect Size (sum of I-spline coefficients)", side = 1, outer = TRUE, line = 1)

  ## --- Plot 3: Stacked bar -- total effect by domain per group ---
  par(mfrow = c(1, 1), mar = c(5, 8, 5, 12), xpd = TRUE)

  ## Compute total effect per domain per group
  domain_totals <- matrix(0, nrow = length(unique_domains), ncol = n_groups,
                          dimnames = list(unique_domains, active_groups))
  for (dom in unique_domains) {
    dom_preds <- all_preds[domains_ordered == dom]
    for (grp in active_groups) {
      domain_totals[dom, grp] <- sum(effects_mat[grp, dom_preds])
    }
  }

  dom_cols <- domain_colours[unique_domains]
  dom_cols[is.na(dom_cols)] <- "grey60"

  bp <- barplot(domain_totals, beside = FALSE,
                col = dom_cols, border = NA,
                names.arg = group_labels[active_groups],
                las = 2, cex.names = 0.8,
                ylab = "Total Effect Size",
                main = sprintf("Total Effect Size by Domain -- %s", model_labels[mt]))

  legend("topright", inset = c(-0.18, 0),
         legend = unique_domains,
         fill = dom_cols,
         border = NA, cex = 0.7, bg = "white",
         title = "Domain")

  ## --- Plot 4: Model fit summary table ---
  par(mar = c(2, 2, 4, 2), xpd = FALSE)
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1))
  title(main = sprintf("Model Fit Summary -- %s", model_labels[mt]), cex.main = 1.2)

  ## Table header
  y_pos <- 0.85
  x_cols <- c(0.05, 0.22, 0.38, 0.54, 0.70, 0.85)
  headers <- c("Group", "D2", "Nagelkerke R2", "N pairs", "N predictors", "Intercept")
  for (j in seq_along(headers)) {
    text(x_cols[j], y_pos, headers[j], font = 2, cex = 0.9, adj = 0)
  }

  ## Table rows
  for (i in seq_along(active_groups)) {
    grp <- active_groups[i]
    m <- meta_list[[grp]]
    y_pos <- y_pos - 0.08
    text(x_cols[1], y_pos, sprintf("%s (%s)", group_labels[grp], grp), cex = 0.85, adj = 0,
         col = group_colours[grp])
    text(x_cols[2], y_pos, sprintf("%.4f", m$D2), cex = 0.85, adj = 0)
    text(x_cols[3], y_pos, sprintf("%.4f", m$R2), cex = 0.85, adj = 0)
    text(x_cols[4], y_pos, sprintf("%d", m$n_pairs), cex = 0.85, adj = 0)
    text(x_cols[5], y_pos, sprintf("%d", m$n_pred), cex = 0.85, adj = 0)
    text(x_cols[6], y_pos, sprintf("%.4f", m$intercept), cex = 0.85, adj = 0)
  }

  dev.off()
  cat(sprintf("  Saved: %s (%d groups, %d predictors)\n", basename(pdf_file), n_groups, n_pred))
}

# ===========================================================================
# 5. Cross-model comparison: one combined summary
# ===========================================================================
cat("\n--- Generating cross-model comparison ---\n")

pdf_compare <- file.path(viz_dir, "coefficient_effects_model_comparison.pdf")
pdf(pdf_compare, width = 16, height = 10)

## --- Plot A: D2 and R2 across model types ---
par(mfrow = c(1, 2), mar = c(8, 5, 4, 1))

## Collect D2 values
for (metric_name in c("D2", "R2")) {
  metric_field <- if (metric_name == "R2") "R2" else "D2"

  values <- list()
  for (mt in model_types) {
    for (grp in groups) {
      m <- all_metadata[[mt]][[grp]]
      if (is.null(m)) next
      values[[length(values) + 1]] <- data.frame(
        group = grp, model = mt, value = m[[metric_field]],
        stringsAsFactors = FALSE)
    }
  }
  df <- do.call(rbind, values)
  if (nrow(df) == 0) next

  ## Grouped barplot: groups on x-axis, model types side-by-side
  active_mt <- intersect(model_types, unique(df$model))
  active_grp <- intersect(groups, unique(df$group))

  mat <- matrix(NA, nrow = length(active_mt), ncol = length(active_grp),
                dimnames = list(active_mt, active_grp))
  for (r in seq_len(nrow(df))) {
    mat[df$model[r], df$group[r]] <- df$value[r]
  }

  mt_cols <- c(base = "#4393C3", MODIS = "#2CA02C", COND = "#D62728")

  barplot(mat, beside = TRUE,
          col = mt_cols[active_mt], border = NA,
          names.arg = group_labels[active_grp],
          las = 2, cex.names = 0.85,
          ylab = metric_name,
          main = sprintf("%s by Group and Model Type", metric_name))

  legend("topright",
         legend = model_labels[active_mt],
         fill = mt_cols[active_mt],
         border = NA, cex = 0.7, bg = "white")
}

## --- Plot B: Per-group comparison of effect sizes across model types ---
## For each group, show how each predictor's effect changes across model types
for (grp in groups) {
  ## Collect effects for this group across model types
  grp_effects <- list()
  for (mt in model_types) {
    if (!is.null(all_effects[[mt]][[grp]])) {
      grp_effects[[mt]] <- all_effects[[mt]][[grp]]
    }
  }
  if (length(grp_effects) < 2) next

  ## Union of predictors
  preds <- unique(unlist(lapply(grp_effects, names)))
  pred_doms <- predictor_domain[preds]
  pred_doms[is.na(pred_doms)] <- "Other"
  p_order <- order(match(pred_doms, domain_order), preds)
  preds <- preds[p_order]

  ## Effects matrix [model_types x predictors]
  active_mt <- names(grp_effects)
  mat <- matrix(0, nrow = length(active_mt), ncol = length(preds),
                dimnames = list(active_mt, preds))
  for (mt in active_mt) {
    shared <- intersect(names(grp_effects[[mt]]), preds)
    mat[mt, shared] <- grp_effects[[mt]][shared]
  }

  ## Cap extremes
  cap2 <- max(mat[mat < 10], na.rm = TRUE) * 1.1
  mat[mat > cap2] <- cap2

  p_labels <- sapply(preds, get_label)

  par(mfrow = c(1, 1), mar = c(5, 12, 5, 2))

  mt_cols <- c(base = "#4393C3", MODIS = "#2CA02C", COND = "#D62728")
  bp_mat <- t(mat[rev(active_mt), , drop = FALSE])

  barplot(t(bp_mat), beside = TRUE, horiz = TRUE,
          names.arg = p_labels,
          col = mt_cols[rev(active_mt)],
          border = NA,
          las = 1, cex.names = 0.65, cex.axis = 0.8,
          xlab = "Effect Size",
          main = sprintf("%s (%s) -- Effect Sizes Across Model Types",
                         group_labels[grp], grp))

  legend("topright",
         legend = model_labels[rev(active_mt)],
         fill = mt_cols[rev(active_mt)],
         border = NA, cex = 0.7, bg = "white")
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_compare)))

cat("\n=== Coefficient barplots complete ===\n")
