##############################################################################
##
## plot_combined_timeseries.R
##
## For each biological group, plot the long baseline (1950-2017) mean
## similarity timeseries and overlay the condition-model timeseries,
## offset so that the COND curve's first year value matches the
## corresponding year's value on the longer baseline timeseries.
##
## Produces:
##   A) Site-level combined timeseries (one page per group + one summary)
##   B) IBRA-level combined timeseries (one page per region per group)
##
## Usage:
##   Rscript plot_combined_timeseries.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Combined Timeseries: Base + Condition (offset)\n")
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
groups <- c("AMP", "MAM", "REP", "VAS") #, "AVES", "HYM")
prefix <- "sans_HYM_AVES" # for file naming only

## Find the latest run folder for each group × variant
find_run_folder <- function(group, variant = "base") {
  pattern <- switch(variant,
    base = sprintf("^%s_\\d{8}T\\d{6}$", group),
    COND = sprintf("^%s_COND_\\d{8}T\\d{6}$", group),
    MODIS = sprintf("^%s_MODIS_\\d{8}T\\d{6}$", group)
  )
  dirs <- list.dirs(output_dir, full.names = FALSE, recursive = FALSE)
  matches <- dirs[grepl(pattern, dirs)]
  if (length(matches) == 0) return(NULL)
  ## Return the latest (alphabetically last = newest timestamp)
  file.path(output_dir, sort(matches, decreasing = TRUE)[1])
}

## Build a registry of available data
registry <- list()
for (grp in groups) {
  base_dir <- find_run_folder(grp, "base")
  cond_dir <- find_run_folder(grp, "COND")

  base_ts   <- if (!is.null(base_dir)) file.path(base_dir, paste0(grp, "_test_timeseries_results.rds")) else NULL
  cond_ts   <- if (!is.null(cond_dir)) file.path(cond_dir, paste0(grp, "_test_timeseries_results.rds")) else NULL
  base_ibra <- if (!is.null(base_dir)) file.path(base_dir, paste0(grp, "_ibra_timeseries_results.rds")) else NULL
  cond_ibra <- if (!is.null(cond_dir)) file.path(cond_dir, paste0(grp, "_ibra_timeseries_results.rds")) else NULL

  registry[[grp]] <- list(
    base_ts   = if (!is.null(base_ts)   && file.exists(base_ts))   base_ts   else NULL,
    cond_ts   = if (!is.null(cond_ts)   && file.exists(cond_ts))   cond_ts   else NULL,
    base_ibra = if (!is.null(base_ibra) && file.exists(base_ibra)) base_ibra else NULL,
    cond_ibra = if (!is.null(cond_ibra) && file.exists(cond_ibra)) cond_ibra else NULL
  )
}

## Report what we found
cat("--- Available data ---\n")
for (grp in groups) {
  r <- registry[[grp]]
  cat(sprintf("  %-5s  base_ts=%s  cond_ts=%s  base_ibra=%s  cond_ibra=%s\n",
              grp,
              if (!is.null(r$base_ts))   "YES" else "---",
              if (!is.null(r$cond_ts))   "YES" else "---",
              if (!is.null(r$base_ibra)) "YES" else "---",
              if (!is.null(r$cond_ibra)) "YES" else "---"))
}
cat("\n")

# ---------------------------------------------------------------------------
# 2. Colour palette for groups
# ---------------------------------------------------------------------------
group_colours <- c(
  AMP  = "#E41A1C",   # red
  AVES = "#377EB8",   # blue
  HYM  = "#FF7F00",   # orange
  MAM  = "#4DAF4A",   # green
  REP  = "#984EA3",   # purple
  VAS  = "#A65628"    # brown
)

## Full readable names for legend labels
group_full_names <- c(
  AMP  = "Amphibians",
  AVES = "Birds",
  HYM  = "Hymenoptera (ants, bees, wasps)",
  MAM  = "Mammals",
  REP  = "Reptiles",
  VAS  = "Vascular plants"
)

# ===========================================================================
# A) SITE-LEVEL COMBINED TIMESERIES
# ===========================================================================
cat("=== A) Site-level combined timeseries ===\n\n")

## ---- Per-group pages + all-groups summary ----
pdf_file <- file.path(viz_dir, sprintf("combined_timeseries_base_vs_cond_%s.pdf", prefix))
pdf(pdf_file, width = 14, height = 8)

## Storage for summary plot
summary_data <- list()

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ts)) {
    cat(sprintf("  [SKIP] %s -- no base timeseries\n", grp))
    next
  }

  base <- readRDS(r$base_ts)
  base_years <- c(base$baseline_year, base$target_years)
  base_sim   <- c(1.0, base$mean_sim)  # similarity=1 at baseline

  has_cond <- !is.null(r$cond_ts)
  if (has_cond) {
    cond <- readRDS(r$cond_ts)
    cond_years_raw <- c(cond$baseline_year, cond$target_years)
    cond_sim_raw   <- c(1.0, cond$mean_sim)

    ## Offset: find the base value at the COND baseline year
    align_year <- cond$baseline_year    # e.g. 2000
    ## Find the nearest or exact year in the base timeseries
    align_idx <- which.min(abs(base_years - align_year))
    base_val_at_align <- base_sim[align_idx]

    ## COND similarity starts at 1.0 (perfect similarity to its own baseline)
    ## Offset = base_value_at_align_year - 1.0
    offset <- base_val_at_align - 1.0
    cond_sim_offset <- cond_sim_raw + offset
  }

  ## Store for summary
  summary_data[[grp]] <- list(
    base_years = base_years,
    base_sim   = base_sim,
    has_cond   = has_cond,
    cond_years = if (has_cond) cond_years_raw else NULL,
    cond_sim   = if (has_cond) cond_sim_offset else NULL
  )

  ## ---- Per-group plot ----
  par(mar = c(5, 5, 4, 2))

  y_vals <- base_sim
  if (has_cond) y_vals <- c(y_vals, cond_sim_offset)
  y_range <- range(y_vals, na.rm = TRUE)
  pad <- diff(y_range) * 0.05
  y_range <- y_range + c(-pad, pad)

  plot(base_years, base_sim, type = "l", lwd = 3,
       col = group_colours[grp],
       xlim = range(base_years), ylim = y_range,
       xlab = "Year", ylab = "Mean Temporal Similarity",
       main = sprintf("%s -- Mean Temporal Similarity\nClimate-only model (1950 baseline) vs Climate + Condition model (aligned at %s)",
                      group_full_names[grp], if (has_cond) as.character(cond$baseline_year) else "N/A"),
       cex.main = 1.0)

  ## Add +/-1 SD band for base
  base_sd <- apply(base$mat_sim, 2, sd, na.rm = TRUE)
  base_sd_full <- c(0, base_sd)
  polygon(c(base_years, rev(base_years)),
          c(base_sim + base_sd_full, rev(pmax(0, base_sim - base_sd_full))),
          col = adjustcolor(group_colours[grp], alpha.f = 0.1), border = NA)

  if (has_cond) {
    ## Offset COND line
    lines(cond_years_raw, cond_sim_offset, lwd = 3, lty = 2,
          col = adjustcolor(group_colours[grp], alpha.f = 0.7))

    ## +/-1 SD band for COND
    cond_sd <- apply(cond$mat_sim, 2, sd, na.rm = TRUE)
    cond_sd_full <- c(0, cond_sd)
    polygon(c(cond_years_raw, rev(cond_years_raw)),
            c(cond_sim_offset + cond_sd_full, rev(pmax(0, cond_sim_offset - cond_sd_full))),
            col = adjustcolor(group_colours[grp], alpha.f = 0.08), border = NA)

    ## Vertical alignment marker
    abline(v = align_year, lty = 3, col = "grey50")
    points(align_year, base_val_at_align, pch = 16, cex = 1.5, col = "grey30")

    legend("bottomleft",
           legend = c(sprintf("Climate-only model (1950 baseline, %d sites)", nrow(base$sites)),
                      sprintf("Climate + Condition model (aligned at %d, %d sites)",
                              cond$baseline_year, nrow(cond$sites)),
                      "Standard deviation", "Alignment point"),
           col = c(group_colours[grp],
                   adjustcolor(group_colours[grp], alpha.f = 0.7),
                   adjustcolor(group_colours[grp], alpha.f = 0.2),
                   "grey30"),
           lwd = c(3, 3, 8, NA), lty = c(1, 2, 1, NA),
           pch = c(NA, NA, NA, 16),
           cex = 0.85, bg = "white")
  } else {
    legend("bottomleft",
           legend = c(sprintf("Climate-only model (1950 baseline, %d sites)", nrow(base$sites)),
                      "Climate + Condition model: not available"),
           col = c(group_colours[grp], "grey70"),
           lwd = c(3, 1), lty = c(1, 3), cex = 0.85, bg = "white")
  }

  cat(sprintf("  %s: plotted (cond=%s)\n", grp, if (has_cond) "YES" else "NO"))
}

## ---- Summary: all groups on one plot ----
par(mar = c(5, 5, 4, 2))
all_y <- unlist(lapply(summary_data, function(d) c(d$base_sim, d$cond_sim)))
y_range <- range(all_y, na.rm = TRUE)
pad <- diff(y_range) * 0.05
y_range <- y_range + c(-pad, pad)

plot(NA, xlim = c(1950, 2017), ylim = y_range,
     xlab = "Year", ylab = "Mean Temporal Similarity",
     main = "All Groups -- Mean Temporal Similarity\nClimate-only model (solid) vs Climate + Condition model (dashed)",
     cex.main = 1.0)

leg_labels <- character()
leg_cols   <- character()
leg_lty    <- integer()

for (grp in names(summary_data)) {
  d <- summary_data[[grp]]
  lines(d$base_years, d$base_sim, col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5)
  leg_labels <- c(leg_labels, sprintf("%s (climate-only)", group_full_names[grp]))
  leg_cols   <- c(leg_cols, group_colours[grp])
  leg_lty    <- c(leg_lty, 1L)

  if (d$has_cond) {
    lines(d$cond_years, d$cond_sim, col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5, lty = 2)
    leg_labels <- c(leg_labels, sprintf("%s (climate + condition)", group_full_names[grp]))
    leg_cols   <- c(leg_cols, group_colours[grp])
    leg_lty    <- c(leg_lty, 2L)
  }
}

## ---- Mean of all base timeseries (black line + SD band) ----
## Interpolate each group's base series onto a common year grid, then average
common_years <- seq(1950, 2017)
base_matrix  <- matrix(NA_real_, nrow = length(summary_data), ncol = length(common_years))
for (i in seq_along(summary_data)) {
  d <- summary_data[[i]]
  base_matrix[i, ] <- approx(d$base_years, d$base_sim,
                              xout = common_years, rule = 1)$y
}
mean_base_sim <- colMeans(base_matrix, na.rm = TRUE)
sd_base_sim   <- apply(base_matrix, 2, sd, na.rm = TRUE)

## SD band for base mean
polygon(c(common_years, rev(common_years)),
        c(mean_base_sim + sd_base_sim, rev(pmax(0, mean_base_sim - sd_base_sim))),
        col = adjustcolor("black", alpha.f = 0.10), border = NA)
lines(common_years, mean_base_sim, col = "black", lwd = 4)

leg_labels <- c(leg_labels, "Mean across all groups (climate-only)")
leg_cols   <- c(leg_cols, "black")
leg_lty    <- c(leg_lty, 1L)

## ---- Mean of all COND timeseries (black dashed line + SD band) ----
cond_groups <- Filter(function(d) d$has_cond, summary_data)
if (length(cond_groups) > 0) {
  ## Determine common year range across all COND series
  cond_min_yr <- min(sapply(cond_groups, function(d) min(d$cond_years)))
  cond_max_yr <- max(sapply(cond_groups, function(d) max(d$cond_years)))
  common_cond_years <- seq(cond_min_yr, cond_max_yr)

  cond_matrix <- matrix(NA_real_, nrow = length(cond_groups), ncol = length(common_cond_years))
  for (i in seq_along(cond_groups)) {
    d <- cond_groups[[i]]
    cond_matrix[i, ] <- approx(d$cond_years, d$cond_sim,
                                xout = common_cond_years, rule = 1)$y
  }
  mean_cond_sim <- colMeans(cond_matrix, na.rm = TRUE)
  sd_cond_sim   <- apply(cond_matrix, 2, sd, na.rm = TRUE)

  ## SD band for cond mean
  polygon(c(common_cond_years, rev(common_cond_years)),
          c(mean_cond_sim + sd_cond_sim, rev(pmax(0, mean_cond_sim - sd_cond_sim))),
          col = adjustcolor("black", alpha.f = 0.06), border = NA)
  lines(common_cond_years, mean_cond_sim, col = "black", lwd = 4, lty = 2)

  leg_labels <- c(leg_labels, "Mean across all groups (climate + condition)")
  leg_cols   <- c(leg_cols, "black")
  leg_lty    <- c(leg_lty, 2L)
}

## Vertical alignment marker at year 2000
align_year_summary <- 2000
align_idx_summary  <- which(common_years == align_year_summary)
abline(v = align_year_summary, lty = 3, col = "grey50")
if (length(align_idx_summary) == 1) {
  points(align_year_summary, mean_base_sim[align_idx_summary],
         pch = 16, cex = 1.5, col = "grey30")
}

legend("bottomleft", legend = leg_labels, col = leg_cols,
       lwd = ifelse(leg_cols == "black", 4, 2.5),
       lty = leg_lty, cex = 0.7, bg = "white", ncol = 2)

dev.off()
cat(sprintf("\n  Saved: %s\n\n", basename(pdf_file)))


# ===========================================================================
# A2) SITE-LEVEL ENV-ONLY TIMESERIES
# ===========================================================================
cat("=== A2) Site-level env-only timeseries ===\n\n")

## Fixed y-axis lower bound for env-only and cond-aligned plots
global_base_ymin <- 0.6
cat(sprintf("  Global y-min: %.1f\n\n", global_base_ymin))

pdf_env <- file.path(viz_dir, sprintf("env_only_timeseries_%s.pdf", prefix))
pdf(pdf_env, width = 14, height = 8)

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ts)) {
    cat(sprintf("  [SKIP] %s -- no base timeseries\n", grp))
    next
  }

  base <- readRDS(r$base_ts)
  base_years <- c(base$baseline_year, base$target_years)
  base_sim   <- c(1.0, base$mean_sim)

  par(mar = c(5, 5, 4, 2))

  plot(base_years, base_sim, type = "l", lwd = 3,
       col = group_colours[grp],
       xlim = range(base_years), ylim = c(global_base_ymin, 1),
       xlab = "Year", ylab = "Mean Temporal Similarity",
       main = sprintf("%s -- Mean Temporal Similarity\nClimate-only model (1950 baseline)",
                      group_full_names[grp]),
       cex.main = 1.0)

  ## +/-1 SD band
  base_sd <- apply(base$mat_sim, 2, sd, na.rm = TRUE)
  base_sd_full <- c(0, base_sd)
  polygon(c(base_years, rev(base_years)),
          c(base_sim + base_sd_full, rev(pmax(0, base_sim - base_sd_full))),
          col = adjustcolor(group_colours[grp], alpha.f = 0.1), border = NA)

  legend("bottomleft",
         legend = c(sprintf("Climate-only model (1950 baseline, %d sites)", nrow(base$sites)),
                    "Standard deviation"),
         col = c(group_colours[grp],
                 adjustcolor(group_colours[grp], alpha.f = 0.2)),
         lwd = c(3, 8), lty = c(1, 1),
         cex = 0.85, bg = "white")

  cat(sprintf("  %s: plotted (env-only)\n", grp))
}

## ---- Summary: all groups env-only ----
par(mar = c(5, 5, 4, 2))

plot(NA, xlim = c(1950, 2017), ylim = c(global_base_ymin, 1),
     xlab = "Year", ylab = "Mean Temporal Similarity",
     main = "All Groups -- Mean Temporal Similarity\nClimate-only model",
     cex.main = 1.0)

leg_labels <- character()
leg_cols   <- character()
leg_lty    <- integer()

for (grp in names(summary_data)) {
  d <- summary_data[[grp]]
  lines(d$base_years, d$base_sim, col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5)
  leg_labels <- c(leg_labels, sprintf("%s (climate-only)", group_full_names[grp]))
  leg_cols   <- c(leg_cols, group_colours[grp])
  leg_lty    <- c(leg_lty, 1L)
}

## Mean of all base timeseries
common_years_env <- seq(1950, 2017)
base_matrix_env  <- matrix(NA_real_, nrow = length(summary_data), ncol = length(common_years_env))
for (i in seq_along(summary_data)) {
  d <- summary_data[[i]]
  base_matrix_env[i, ] <- approx(d$base_years, d$base_sim,
                                  xout = common_years_env, rule = 1)$y
}
mean_base_env <- colMeans(base_matrix_env, na.rm = TRUE)
sd_base_env   <- apply(base_matrix_env, 2, sd, na.rm = TRUE)

polygon(c(common_years_env, rev(common_years_env)),
        c(mean_base_env + sd_base_env, rev(pmax(0, mean_base_env - sd_base_env))),
        col = adjustcolor("black", alpha.f = 0.10), border = NA)
lines(common_years_env, mean_base_env, col = "black", lwd = 4)

leg_labels <- c(leg_labels, "Mean across all groups (climate-only)")
leg_cols   <- c(leg_cols, "black")
leg_lty    <- c(leg_lty, 1L)

legend("bottomleft", legend = leg_labels, col = leg_cols,
       lwd = ifelse(leg_cols == "black", 4, 2.5),
       lty = leg_lty, cex = 0.7, bg = "white", ncol = 2)

dev.off()
cat(sprintf("\n  Saved: %s\n\n", basename(pdf_env)))


# ===========================================================================
# A3) SITE-LEVEL CONDITION-ALIGNED TIMESERIES (x-axis from condition start)
# ===========================================================================
cat("=== A3) Site-level condition-aligned timeseries ===\n\n")

pdf_aligned <- file.path(viz_dir, sprintf("cond_aligned_timeseries_%s.pdf", prefix))
pdf(pdf_aligned, width = 14, height = 8)

## Storage for summary plot
summary_aligned <- list()

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ts)) {
    cat(sprintf("  [SKIP] %s -- no base timeseries\n", grp))
    next
  }

  base <- readRDS(r$base_ts)
  base_years <- c(base$baseline_year, base$target_years)
  base_sim   <- c(1.0, base$mean_sim)

  has_cond <- !is.null(r$cond_ts)
  if (!has_cond) {
    cat(sprintf("  [SKIP] %s -- no condition timeseries for alignment\n", grp))
    next
  }

  cond <- readRDS(r$cond_ts)
  cond_years_raw <- c(cond$baseline_year, cond$target_years)
  cond_sim_raw   <- c(1.0, cond$mean_sim)

  align_year <- cond$baseline_year
  align_idx  <- which.min(abs(base_years - align_year))
  base_val_at_align <- base_sim[align_idx]

  ## Offset COND to align with base at the condition baseline year
  offset <- base_val_at_align - 1.0
  cond_sim_offset <- cond_sim_raw + offset

  ## Restrict base to years >= align_year
  base_keep <- base_years >= align_year
  base_years_clipped <- base_years[base_keep]
  base_sim_clipped   <- base_sim[base_keep]

  ## Store for summary
  summary_aligned[[grp]] <- list(
    base_years = base_years_clipped,
    base_sim   = base_sim_clipped,
    cond_years = cond_years_raw,
    cond_sim   = cond_sim_offset,
    align_year = align_year
  )

  ## ---- Shared values for all style variants ----
  base_sd <- apply(base$mat_sim, 2, sd, na.rm = TRUE)
  base_sd_full <- c(0, base_sd)
  base_sd_clipped <- base_sd_full[base_keep]

  cond_sd <- apply(cond$mat_sim, 2, sd, na.rm = TRUE)
  cond_sd_full <- c(0, cond_sd)

  x_range <- range(c(base_years_clipped, cond_years_raw))

  ## Helper: draw a base plot frame (reused by each variant)
  draw_base_frame <- function(subtitle) {
    par(mar = c(5, 5, 4, 2))
    plot(base_years_clipped, base_sim_clipped, type = "l", lwd = 3,
         col = group_colours[grp],
         xlim = x_range, ylim = c(global_base_ymin, 1),
         xlab = "Year", ylab = "Mean Temporal Similarity",
         main = sprintf("%s -- Mean Temporal Similarity [%s]\nClimate-only (1950 baseline) vs Climate + Condition (aligned at %d)",
                        group_full_names[grp], subtitle, align_year),
         cex.main = 1.0)
    ## Base SD band (always solid fill)
    polygon(c(base_years_clipped, rev(base_years_clipped)),
            c(base_sim_clipped + base_sd_clipped,
              rev(pmax(0, base_sim_clipped - base_sd_clipped))),
            col = adjustcolor(group_colours[grp], alpha.f = 0.15), border = NA)
    ## Alignment marker
    abline(v = align_year, lty = 3, col = "grey50")
    points(align_year, base_val_at_align, pch = 16, cex = 1.5, col = "grey30")
  }

  ## ================================================================
  ## STYLE A: Stippled (hatched) fill for condition SD band
  ## ================================================================
  draw_base_frame("Style A: stippled condition SD")

  ## Condition SD band with cross-hatching
  polygon(c(cond_years_raw, rev(cond_years_raw)),
          c(cond_sim_offset + cond_sd_full,
            rev(pmax(0, cond_sim_offset - cond_sd_full))),
          col = NA,
          border = adjustcolor(group_colours[grp], alpha.f = 0.4),
          density = 15, angle = 45, lwd = 0.5)
  ## Second pass at opposite angle for cross-hatch effect
  polygon(c(cond_years_raw, rev(cond_years_raw)),
          c(cond_sim_offset + cond_sd_full,
            rev(pmax(0, cond_sim_offset - cond_sd_full))),
          col = NA,
          border = adjustcolor(group_colours[grp], alpha.f = 0.4),
          density = 15, angle = -45, lwd = 0.5)

  lines(cond_years_raw, cond_sim_offset, lwd = 3, lty = 2,
        col = adjustcolor(group_colours[grp], alpha.f = 0.7))

  legend("bottomleft",
         legend = c(sprintf("Climate-only model (%d sites)", nrow(base$sites)),
                    sprintf("Clim + Condition model (%d sites)", nrow(cond$sites)),
                    "Base SD (shaded)", "Condition SD (stippled)", "Alignment point"),
         col = c(group_colours[grp],
                 adjustcolor(group_colours[grp], alpha.f = 0.7),
                 adjustcolor(group_colours[grp], alpha.f = 0.15),
                 adjustcolor(group_colours[grp], alpha.f = 0.4),
                 "grey30"),
         lwd = c(3, 3, 8, 8, NA), lty = c(1, 2, 1, 1, NA),
         pch = c(NA, NA, NA, NA, 16),
         cex = 0.85, bg = "white")

  ## ================================================================
  ## STYLE B: Contrasting colour for condition (blue-grey)
  ## ================================================================
  cond_col <- "#4A708B"   # steel-blue, distinct from any group colour

  draw_base_frame("Style B: contrasting colour")

  ## Condition SD band in contrasting colour
  polygon(c(cond_years_raw, rev(cond_years_raw)),
          c(cond_sim_offset + cond_sd_full,
            rev(pmax(0, cond_sim_offset - cond_sd_full))),
          col = adjustcolor(cond_col, alpha.f = 0.12), border = NA)

  lines(cond_years_raw, cond_sim_offset, lwd = 3, lty = 2,
        col = adjustcolor(cond_col, alpha.f = 0.85))

  legend("bottomleft",
         legend = c(sprintf("Climate-only model (%d sites)", nrow(base$sites)),
                    sprintf("Clim + Condition model (%d sites)", nrow(cond$sites)),
                    "Base SD", "Condition SD", "Alignment point"),
         col = c(group_colours[grp],
                 adjustcolor(cond_col, alpha.f = 0.85),
                 adjustcolor(group_colours[grp], alpha.f = 0.15),
                 adjustcolor(cond_col, alpha.f = 0.25),
                 "grey30"),
         lwd = c(3, 3, 8, 8, NA), lty = c(1, 2, 1, 1, NA),
         pch = c(NA, NA, NA, NA, 16),
         cex = 0.85, bg = "white")

  ## ================================================================
  ## STYLE C: Outlined-only condition SD (dashed border, no fill)
  ## ================================================================
  draw_base_frame("Style C: outlined condition SD")

  ## Condition SD band as dashed outline only
  polygon(c(cond_years_raw, rev(cond_years_raw)),
          c(cond_sim_offset + cond_sd_full,
            rev(pmax(0, cond_sim_offset - cond_sd_full))),
          col = NA,
          border = adjustcolor(group_colours[grp], alpha.f = 0.45),
          lty = 3, lwd = 1.5)

  lines(cond_years_raw, cond_sim_offset, lwd = 3, lty = 2,
        col = adjustcolor(group_colours[grp], alpha.f = 0.7))

  legend("bottomleft",
         legend = c(sprintf("Climate-only model (%d sites)", nrow(base$sites)),
                    sprintf("Clim + Condition model (%d sites)", nrow(cond$sites)),
                    "Base SD (shaded)", "Condition SD (outline)", "Alignment point"),
         col = c(group_colours[grp],
                 adjustcolor(group_colours[grp], alpha.f = 0.7),
                 adjustcolor(group_colours[grp], alpha.f = 0.15),
                 adjustcolor(group_colours[grp], alpha.f = 0.45),
                 "grey30"),
         lwd = c(3, 3, 8, 1.5, NA), lty = c(1, 2, 1, 3, NA),
         pch = c(NA, NA, NA, NA, 16),
         cex = 0.85, bg = "white")

  cat(sprintf("  %s: plotted 3 style variants (cond-aligned)\n", grp))
}

## ---- Summary: all groups condition-aligned ----
if (length(summary_aligned) > 0) {
  par(mar = c(5, 5, 4, 2))

  ## x range: from earliest align_year to latest year
  x_min <- min(sapply(summary_aligned, function(d) d$align_year))
  x_max <- max(sapply(summary_aligned, function(d) max(c(d$base_years, d$cond_years))))

  plot(NA, xlim = c(x_min, x_max), ylim = c(global_base_ymin, 1),
       xlab = "Year", ylab = "Mean Temporal Similarity",
       main = "All Groups -- Mean Temporal Similarity\nClimate-only (solid) vs Climate + Condition (dashed)",
       cex.main = 1.0)

  leg_labels <- character()
  leg_cols   <- character()
  leg_lty    <- integer()

  for (grp in names(summary_aligned)) {
    d <- summary_aligned[[grp]]
    lines(d$base_years, d$base_sim,
          col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5)
    leg_labels <- c(leg_labels, sprintf("%s (climate-only)", group_full_names[grp]))
    leg_cols   <- c(leg_cols, group_colours[grp])
    leg_lty    <- c(leg_lty, 1L)

    lines(d$cond_years, d$cond_sim,
          col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5, lty = 2)
    leg_labels <- c(leg_labels, sprintf("%s (climate + condition)", group_full_names[grp]))
    leg_cols   <- c(leg_cols, group_colours[grp])
    leg_lty    <- c(leg_lty, 2L)
  }

  ## Mean across groups
  common_aligned_years <- seq(x_min, x_max)
  base_mat_a <- matrix(NA_real_, nrow = length(summary_aligned), ncol = length(common_aligned_years))
  cond_mat_a <- matrix(NA_real_, nrow = length(summary_aligned), ncol = length(common_aligned_years))
  for (i in seq_along(summary_aligned)) {
    d <- summary_aligned[[i]]
    base_mat_a[i, ] <- approx(d$base_years, d$base_sim,
                               xout = common_aligned_years, rule = 1)$y
    cond_mat_a[i, ] <- approx(d$cond_years, d$cond_sim,
                               xout = common_aligned_years, rule = 1)$y
  }
  mean_base_a <- colMeans(base_mat_a, na.rm = TRUE)
  sd_base_a   <- apply(base_mat_a, 2, sd, na.rm = TRUE)
  mean_cond_a <- colMeans(cond_mat_a, na.rm = TRUE)
  sd_cond_a   <- apply(cond_mat_a, 2, sd, na.rm = TRUE)

  polygon(c(common_aligned_years, rev(common_aligned_years)),
          c(mean_base_a + sd_base_a, rev(pmax(0, mean_base_a - sd_base_a))),
          col = adjustcolor("black", alpha.f = 0.10), border = NA)
  lines(common_aligned_years, mean_base_a, col = "black", lwd = 4)
  leg_labels <- c(leg_labels, "Mean across all groups (climate-only)")
  leg_cols   <- c(leg_cols, "black")
  leg_lty    <- c(leg_lty, 1L)

  polygon(c(common_aligned_years, rev(common_aligned_years)),
          c(mean_cond_a + sd_cond_a, rev(pmax(0, mean_cond_a - sd_cond_a))),
          col = adjustcolor("black", alpha.f = 0.06), border = NA)
  lines(common_aligned_years, mean_cond_a, col = "black", lwd = 4, lty = 2)
  leg_labels <- c(leg_labels, "Mean across all groups (climate + condition)")
  leg_cols   <- c(leg_cols, "black")
  leg_lty    <- c(leg_lty, 2L)

  ## Vertical alignment marker
  abline(v = x_min, lty = 3, col = "grey50")

  legend("bottomleft", legend = leg_labels, col = leg_cols,
         lwd = ifelse(leg_cols == "black", 4, 2.5),
         lty = leg_lty, cex = 0.7, bg = "white", ncol = 2)
}

dev.off()
cat(sprintf("\n  Saved: %s\n\n", basename(pdf_aligned)))


# ===========================================================================
# B) IBRA-LEVEL COMBINED TIMESERIES
# ===========================================================================
cat("=== B) IBRA-level combined timeseries ===\n\n")

pdf_ibra <- file.path(viz_dir, sprintf("combined_timeseries_IBRA_base_vs_cond_%s.pdf", prefix))

pdf(pdf_ibra, width = 14, height = 8)

## ---- Pre-compute per-group ylim across all IBRA regions ----
## Consistent within each group; different between groups for max variation.
group_ylims <- list()
for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ibra)) next
  tmp_base <- readRDS(r$base_ibra)
  tmp_base_years <- c(tmp_base$baseline_year, tmp_base$target_years)
  tmp_cond <- NULL
  if (!is.null(r$cond_ibra)) tmp_cond <- readRDS(r$cond_ibra)

  all_y_grp <- c()
  for (reg in names(tmp_base$region_results)) {
    br <- tmp_base$region_results[[reg]]
    if (is.null(br) || all(is.na(br$mat_sim))) next
    base_mean_tmp <- c(1.0, colMeans(br$mat_sim, na.rm = TRUE))
    all_y_grp <- c(all_y_grp, base_mean_tmp)
    if (!is.null(tmp_cond) && reg %in% names(tmp_cond$region_results)) {
      cr <- tmp_cond$region_results[[reg]]
      if (!is.null(cr) && !all(is.na(cr$mat_sim))) {
        cond_mean_tmp <- c(1.0, colMeans(cr$mat_sim, na.rm = TRUE))
        align_idx_tmp <- which.min(abs(tmp_base_years - tmp_cond$baseline_year))
        offset_tmp <- base_mean_tmp[align_idx_tmp] - 1.0
        all_y_grp <- c(all_y_grp, cond_mean_tmp + offset_tmp)
      }
    }
  }
  yr <- range(all_y_grp, na.rm = TRUE)
  pad <- diff(yr) * 0.05
  group_ylims[[grp]] <- yr + c(-pad, pad)
  cat(sprintf("  %s ylim: [%.4f, %.4f]\n", grp, group_ylims[[grp]][1], group_ylims[[grp]][2]))
}
cat("\n")

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ibra)) {
    cat(sprintf("  [SKIP] %s -- no base IBRA timeseries\n", grp))
    next
  }

  base_ibra <- readRDS(r$base_ibra)
  base_target_years <- base_ibra$target_years
  base_all_years    <- c(base_ibra$baseline_year, base_target_years)

  has_cond_ibra <- !is.null(r$cond_ibra)
  cond_ibra <- NULL
  if (has_cond_ibra) {
    cond_ibra <- readRDS(r$cond_ibra)
    cond_target_years <- cond_ibra$target_years
    cond_all_years    <- c(cond_ibra$baseline_year, cond_target_years)
  }

  ## Get all IBRA regions from the base results
  base_regions <- names(base_ibra$region_results)

  ## Use the pre-computed per-group ylim
  grp_ylim <- group_ylims[[grp]]

  ## ---- Per-region plot ----
  for (reg in base_regions) {
    br <- base_ibra$region_results[[reg]]
    if (is.null(br) || all(is.na(br$mat_sim))) next

    ## Base: mean similarity per year (prepend 1.0 for baseline year)
    base_mean <- c(1.0, colMeans(br$mat_sim, na.rm = TRUE))
    base_sd   <- c(0,   apply(br$mat_sim, 2, sd, na.rm = TRUE))

    ## Condition
    cond_plotted <- FALSE
    if (has_cond_ibra && reg %in% names(cond_ibra$region_results)) {
      cr <- cond_ibra$region_results[[reg]]
      if (!is.null(cr) && !all(is.na(cr$mat_sim))) {
        cond_mean_raw <- c(1.0, colMeans(cr$mat_sim, na.rm = TRUE))
        cond_sd_raw   <- c(0,   apply(cr$mat_sim, 2, sd, na.rm = TRUE))

        ## Offset
        align_year <- cond_ibra$baseline_year
        align_idx  <- which.min(abs(base_all_years - align_year))
        base_val   <- base_mean[align_idx]
        offset     <- base_val - 1.0
        cond_mean_offset <- cond_mean_raw + offset
        cond_plotted <- TRUE
      }
    }

    par(mar = c(5, 5, 4, 2))
    plot(base_all_years, base_mean, type = "l", lwd = 3,
         col = group_colours[grp],
         xlim = range(base_all_years), ylim = grp_ylim,
         xlab = "Year", ylab = "Mean Temporal Similarity",
         main = sprintf("%s -- %s\nBase vs Condition (offset) | %d base sites",
                        grp, reg, br$n_sites),
         cex.main = 0.95)

    ## SD band
    polygon(c(base_all_years, rev(base_all_years)),
            c(base_mean + base_sd, rev(pmax(0, base_mean - base_sd))),
            col = adjustcolor(group_colours[grp], alpha.f = 0.1), border = NA)

    if (cond_plotted) {
      lines(cond_all_years, cond_mean_offset, lwd = 3, lty = 2,
            col = adjustcolor(group_colours[grp], alpha.f = 0.7))
      polygon(c(cond_all_years, rev(cond_all_years)),
              c(cond_mean_offset + cond_sd_raw, rev(pmax(0, cond_mean_offset - cond_sd_raw))),
              col = adjustcolor(group_colours[grp], alpha.f = 0.08), border = NA)
      abline(v = align_year, lty = 3, col = "grey50")
    }
  }

  cat(sprintf("  %s: %d IBRA regions plotted (cond=%s)\n",
              grp, length(base_regions), if (has_cond_ibra) "YES" else "NO"))
}

dev.off()
cat(sprintf("\n  Saved: %s\n\n", basename(pdf_ibra)))


# ===========================================================================
# C) IBRA-LEVEL ALL-GROUPS SUMMARY (one page per IBRA region)
# ===========================================================================
cat("=== C) IBRA-level all-groups summary ===\n\n")

## First, collect all IBRA data into a structure: ibra_data[[region]][[group]]
ibra_all_data <- list()

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ibra)) next

  base_ibra <- readRDS(r$base_ibra)
  base_all_years <- c(base_ibra$baseline_year, base_ibra$target_years)

  has_cond_ibra <- !is.null(r$cond_ibra)
  cond_ibra <- NULL
  if (has_cond_ibra) {
    cond_ibra <- readRDS(r$cond_ibra)
    cond_all_years <- c(cond_ibra$baseline_year, cond_ibra$target_years)
  }

  for (reg in names(base_ibra$region_results)) {
    br <- base_ibra$region_results[[reg]]
    if (is.null(br) || all(is.na(br$mat_sim))) next

    base_mean <- c(1.0, colMeans(br$mat_sim, na.rm = TRUE))

    entry <- list(
      base_years = base_all_years,
      base_sim   = base_mean,
      n_sites    = br$n_sites,
      has_cond   = FALSE,
      cond_years = NULL,
      cond_sim   = NULL
    )

    if (has_cond_ibra && reg %in% names(cond_ibra$region_results)) {
      cr <- cond_ibra$region_results[[reg]]
      if (!is.null(cr) && !all(is.na(cr$mat_sim))) {
        cond_mean_raw <- c(1.0, colMeans(cr$mat_sim, na.rm = TRUE))
        align_year <- cond_ibra$baseline_year
        align_idx  <- which.min(abs(base_all_years - align_year))
        offset     <- base_mean[align_idx] - 1.0
        entry$has_cond   <- TRUE
        entry$cond_years <- cond_all_years
        entry$cond_sim   <- cond_mean_raw + offset
      }
    }

    if (is.null(ibra_all_data[[reg]])) ibra_all_data[[reg]] <- list()
    ibra_all_data[[reg]][[grp]] <- entry
  }
}

## Now plot one page per IBRA region
all_ibra_regions <- sort(names(ibra_all_data))
cat(sprintf("  Found %d IBRA regions with data\n", length(all_ibra_regions)))

## ---- Compute GLOBAL y-range across all IBRA regions and groups ----
global_y_vals <- unlist(lapply(ibra_all_data, function(reg_data) {
  unlist(lapply(reg_data, function(d) c(d$base_sim, d$cond_sim)))
}))
global_y_range <- range(global_y_vals, na.rm = TRUE)
global_pad <- diff(global_y_range) * 0.05
global_y_range <- global_y_range + c(-global_pad, global_pad)
cat(sprintf("  Global y-range for IBRA plots: [%.4f, %.4f]\n",
            global_y_range[1], global_y_range[2]))

pdf_ibra_allgrp <- file.path(viz_dir, sprintf("combined_timeseries_IBRA_all_groups_%s.pdf", prefix))
pdf(pdf_ibra_allgrp, width = 14, height = 8)

for (reg in all_ibra_regions) {
  reg_data <- ibra_all_data[[reg]]
  if (length(reg_data) == 0) next

  ## Use global y range so all IBRA pages are comparable
  y_range <- global_y_range

  par(mar = c(5, 5, 4, 2))
  plot(NA, xlim = c(1950, 2017), ylim = y_range,
       xlab = "Year", ylab = "Mean Temporal Similarity",
       main = sprintf("%s -- All Groups\nClimate-only (solid) vs Climate + Condition (dashed)",
                      reg),
       cex.main = 1.0)

  leg_labels <- character()
  leg_cols   <- character()
  leg_lty    <- integer()

  for (grp in names(reg_data)) {
    d <- reg_data[[grp]]
    lines(d$base_years, d$base_sim,
          col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5)
    leg_labels <- c(leg_labels, sprintf("%s (climate-only)", group_full_names[grp]))
    leg_cols   <- c(leg_cols, group_colours[grp])
    leg_lty    <- c(leg_lty, 1L)

    if (d$has_cond) {
      lines(d$cond_years, d$cond_sim,
            col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5, lty = 2)
      leg_labels <- c(leg_labels, sprintf("%s (climate + condition)", group_full_names[grp]))
      leg_cols   <- c(leg_cols, group_colours[grp])
      leg_lty    <- c(leg_lty, 2L)
    }
  }

  ## Mean line across groups for this IBRA region
  common_years <- seq(1950, 2017)
  base_mat <- matrix(NA_real_, nrow = length(reg_data), ncol = length(common_years))
  for (i in seq_along(reg_data)) {
    d <- reg_data[[i]]
    base_mat[i, ] <- approx(d$base_years, d$base_sim,
                             xout = common_years, rule = 1)$y
  }
  mean_base <- colMeans(base_mat, na.rm = TRUE)
  sd_base   <- apply(base_mat, 2, sd, na.rm = TRUE)

  polygon(c(common_years, rev(common_years)),
          c(mean_base + sd_base, rev(pmax(0, mean_base - sd_base))),
          col = adjustcolor("black", alpha.f = 0.10), border = NA)
  lines(common_years, mean_base, col = "black", lwd = 4)
  leg_labels <- c(leg_labels, "Mean across all groups (climate-only)")
  leg_cols   <- c(leg_cols, "black")
  leg_lty    <- c(leg_lty, 1L)

  ## Mean COND line
  cond_entries <- Filter(function(d) d$has_cond, reg_data)
  if (length(cond_entries) > 0) {
    cond_min <- min(sapply(cond_entries, function(d) min(d$cond_years)))
    cond_max <- max(sapply(cond_entries, function(d) max(d$cond_years)))
    common_cond <- seq(cond_min, cond_max)
    cond_mat <- matrix(NA_real_, nrow = length(cond_entries), ncol = length(common_cond))
    for (i in seq_along(cond_entries)) {
      d <- cond_entries[[i]]
      cond_mat[i, ] <- approx(d$cond_years, d$cond_sim,
                               xout = common_cond, rule = 1)$y
    }
    mean_cond <- colMeans(cond_mat, na.rm = TRUE)
    sd_cond   <- apply(cond_mat, 2, sd, na.rm = TRUE)

    polygon(c(common_cond, rev(common_cond)),
            c(mean_cond + sd_cond, rev(pmax(0, mean_cond - sd_cond))),
            col = adjustcolor("black", alpha.f = 0.06), border = NA)
    lines(common_cond, mean_cond, col = "black", lwd = 4, lty = 2)
    leg_labels <- c(leg_labels, "Mean across all groups (climate + condition)")
    leg_cols   <- c(leg_cols, "black")
    leg_lty    <- c(leg_lty, 2L)
  }

  ## Alignment marker at 2000
  abline(v = 2000, lty = 3, col = "grey50")
  idx_2000 <- which(common_years == 2000)
  if (length(idx_2000) == 1) {
    points(2000, mean_base[idx_2000], pch = 16, cex = 1.5, col = "grey30")
  }

  legend("bottomleft", legend = leg_labels, col = leg_cols,
         lwd = ifelse(leg_cols == "black", 4, 2.5),
         lty = leg_lty, cex = 0.65, bg = "white", ncol = 2)
}

dev.off()
cat(sprintf("  Saved: %s\n\n", basename(pdf_ibra_allgrp)))


# ===========================================================================
# D) EXPORT TIMESERIES AS JSON
# ===========================================================================
cat("=== D) Exporting timeseries as JSON ===\n\n")

library(jsonlite)

## --- D1. Australia-wide timeseries JSON ---
aus_json_data <- list()
for (grp in names(summary_data)) {
  d <- summary_data[[grp]]
  entry <- list(
    group      = grp,
    group_name = unname(group_full_names[grp]),
    base = list(
      years      = d$base_years,
      similarity = round(d$base_sim, 6)
    )
  )
  if (d$has_cond) {
    entry$condition <- list(
      years      = d$cond_years,
      similarity = round(d$cond_sim, 6)
    )
  }
  aus_json_data[[grp]] <- entry
}

## Add the cross-group mean
common_years_out <- seq(1950, 2017)
aus_json_data[["_mean"]] <- list(
  group      = "_mean",
  group_name = "Mean across all groups",
  base = list(
    years      = common_years_out,
    similarity = round(colMeans(
      do.call(rbind, lapply(summary_data, function(d)
        approx(d$base_years, d$base_sim, xout = common_years_out, rule = 1)$y
      )), na.rm = TRUE), 6)
  )
)
## Add mean cond if available
cond_grps <- Filter(function(d) d$has_cond, summary_data)
if (length(cond_grps) > 0) {
  cond_min <- min(sapply(cond_grps, function(d) min(d$cond_years)))
  cond_max <- max(sapply(cond_grps, function(d) max(d$cond_years)))
  cond_yrs <- seq(cond_min, cond_max)
  aus_json_data[["_mean"]]$condition <- list(
    years      = cond_yrs,
    similarity = round(colMeans(
      do.call(rbind, lapply(cond_grps, function(d)
        approx(d$cond_years, d$cond_sim, xout = cond_yrs, rule = 1)$y
      )), na.rm = TRUE), 6)
  )
}

aus_json_path <- file.path(viz_dir, "timeseries_australia_wide.json")
writeLines(toJSON(aus_json_data, pretty = TRUE, auto_unbox = TRUE), aus_json_path)
cat(sprintf("  Saved: %s\n", basename(aus_json_path)))

## --- D2. IBRA-level timeseries JSON ---
ibra_json_data <- list()
for (reg in sort(names(ibra_all_data))) {
  reg_data <- ibra_all_data[[reg]]
  reg_entry <- list(region = reg, groups = list())

  for (grp in names(reg_data)) {
    d <- reg_data[[grp]]
    g_entry <- list(
      group      = grp,
      group_name = unname(group_full_names[grp]),
      n_sites    = d$n_sites,
      base = list(
        years      = d$base_years,
        similarity = round(d$base_sim, 6)
      )
    )
    if (d$has_cond) {
      g_entry$condition <- list(
        years      = d$cond_years,
        similarity = round(d$cond_sim, 6)
      )
    }
    reg_entry$groups[[grp]] <- g_entry
  }

  ## Cross-group mean for this region
  common_yrs <- seq(1950, 2017)
  base_vals <- do.call(rbind, lapply(reg_data, function(d)
    approx(d$base_years, d$base_sim, xout = common_yrs, rule = 1)$y
  ))
  reg_entry$mean_base <- list(
    years      = common_yrs,
    similarity = round(colMeans(base_vals, na.rm = TRUE), 6)
  )

  cond_entries <- Filter(function(d) d$has_cond, reg_data)
  if (length(cond_entries) > 0) {
    c_min <- min(sapply(cond_entries, function(d) min(d$cond_years)))
    c_max <- max(sapply(cond_entries, function(d) max(d$cond_years)))
    c_yrs <- seq(c_min, c_max)
    cond_vals <- do.call(rbind, lapply(cond_entries, function(d)
      approx(d$cond_years, d$cond_sim, xout = c_yrs, rule = 1)$y
    ))
    reg_entry$mean_condition <- list(
      years      = c_yrs,
      similarity = round(colMeans(cond_vals, na.rm = TRUE), 6)
    )
  }

  ibra_json_data[[reg]] <- reg_entry
}

ibra_json_path <- file.path(viz_dir, "timeseries_ibra.json")
writeLines(toJSON(ibra_json_data, pretty = TRUE, auto_unbox = TRUE), ibra_json_path)
cat(sprintf("  Saved: %s\n\n", basename(ibra_json_path)))

cat("=== Combined timeseries complete ===\n")
