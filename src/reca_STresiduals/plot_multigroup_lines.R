##############################################################################
##
## plot_multigroup_lines.R
##
## Plot the long 1950-2017 baseline timeseries (modis=FALSE, cond=FALSE)
## with one line per biological group.
##
## Produces:
##   A) Combined site-level: all groups' mean similarity on one plot
##   B) Per-IBRA-region: all groups' mean similarity per IBRA region
##
## Usage:
##   Rscript plot_multigroup_lines.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Multi-Group Line Plots (Base 1950-2017)\n")
cat("###########################################################################\n\n")

this_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  ## When run via Rscript, try commandArgs to find the script path
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
# 1. Discover base run folders
# ---------------------------------------------------------------------------
groups <- c("AMP", "AVES", "HYM", "MAM", "REP", "VAS")

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

group_colours <- c(
  AMP  = "#E41A1C",   # red
  AVES = "#377EB8",   # blue
  HYM  = "#FF7F00",   # orange
  MAM  = "#4DAF4A",   # green
  REP  = "#984EA3",   # purple
  VAS  = "#A65628"    # brown
)

group_labels <- c(
  AMP  = "Amphibians",
  AVES = "Birds",
  HYM  = "Hymenoptera",
  MAM  = "Mammals",
  REP  = "Reptiles",
  VAS  = "Vascular Plants"
)

# ---------------------------------------------------------------------------
# 2. Load site-level timeseries for each group
# ---------------------------------------------------------------------------
cat("--- Loading site-level timeseries ---\n")
site_data <- list()

for (grp in groups) {
  base_dir <- find_run_folder(grp, "base")
  if (is.null(base_dir)) {
    cat(sprintf("  [SKIP] %s -- no base run folder\n", grp))
    next
  }
  ts_file <- file.path(base_dir, paste0(grp, "_test_timeseries_results.rds"))
  if (!file.exists(ts_file)) {
    cat(sprintf("  [SKIP] %s -- timeseries RDS not found\n", grp))
    next
  }
  ts <- readRDS(ts_file)
  site_data[[grp]] <- list(
    years    = c(ts$baseline_year, ts$target_years),
    mean_sim = c(1.0, ts$mean_sim),
    sd_sim   = c(0, apply(ts$mat_sim, 2, sd, na.rm = TRUE)),
    n_sites  = nrow(ts$sites),
    metadata = ts$fit_metadata
  )
  cat(sprintf("  %-5s : loaded (%d years, %d sites)\n", grp,
              length(ts$target_years), nrow(ts$sites)))
}

active_groups <- names(site_data)
cat(sprintf("\n  Active groups: %d / %d\n\n", length(active_groups), length(groups)))

# ===========================================================================
# A) COMBINED SITE-LEVEL PLOT
# ===========================================================================
cat("=== A) Combined site-level multi-group plot ===\n")

pdf_site <- file.path(viz_dir, "multigroup_timeseries_site_level.pdf")
pdf(pdf_site, width = 14, height = 8)

## ---- Plot 1: Lines only (clean) ----
par(mar = c(5, 5, 4, 2))

all_y <- unlist(lapply(site_data, function(d) d$mean_sim))
y_range <- range(all_y, na.rm = TRUE)
pad <- diff(y_range) * 0.05
y_range <- y_range + c(-pad, pad)

plot(NA, xlim = c(1950, 2017), ylim = y_range,
     xlab = "Year", ylab = "Mean Temporal Similarity (vs 1950 baseline)",
     main = "Temporal Biodiversity Similarity -- All Groups\n30 yr climate window | 1950 baseline",
     cex.main = 1.1)
grid(col = "grey90")

for (grp in active_groups) {
  d <- site_data[[grp]]
  lines(d$years, d$mean_sim, col = group_colours[grp], lwd = 3)
}

legend("topleft",
       legend = sprintf("%s (%s, n=%d)", group_labels[active_groups],
                        active_groups, sapply(site_data[active_groups], function(d) d$n_sites)),
       col = group_colours[active_groups],
       lwd = 3, cex = 0.85, bg = "white")

## ---- Plot 2: Lines with +/-1 SD ribbons ----
par(mar = c(5, 5, 4, 2))

## Expand y-range to include SD bands
all_y_sd <- unlist(lapply(site_data, function(d) c(d$mean_sim + d$sd_sim, d$mean_sim - d$sd_sim)))
y_range2 <- range(all_y_sd, na.rm = TRUE)
pad2 <- diff(y_range2) * 0.05
y_range2 <- y_range2 + c(-pad2, pad2)

plot(NA, xlim = c(1950, 2017), ylim = y_range2,
     xlab = "Year", ylab = "Mean Temporal Similarity (vs 1950 baseline)",
     main = "Temporal Biodiversity Similarity -- All Groups (+/-1 SD)\n30 yr climate window | 1950 baseline",
     cex.main = 1.1)
grid(col = "grey90")

## Draw SD bands first (back to front)
for (grp in rev(active_groups)) {
  d <- site_data[[grp]]
  polygon(c(d$years, rev(d$years)),
          c(d$mean_sim + d$sd_sim, rev(pmax(0, d$mean_sim - d$sd_sim))),
          col = adjustcolor(group_colours[grp], alpha.f = 0.08), border = NA)
}

## Then lines on top
for (grp in active_groups) {
  d <- site_data[[grp]]
  lines(d$years, d$mean_sim, col = group_colours[grp], lwd = 3)
}

legend("topleft",
       legend = sprintf("%s (%s)", group_labels[active_groups], active_groups),
       col = group_colours[active_groups],
       lwd = 3, cex = 0.85, bg = "white")

## ---- Plot 3: Rate of change (year-on-year delta) ----
par(mar = c(5, 5, 4, 2))

## Compute deltas
delta_data <- list()
for (grp in active_groups) {
  d <- site_data[[grp]]
  delta_data[[grp]] <- list(
    years = d$years[-1],
    delta = diff(d$mean_sim)
  )
}

all_delta <- unlist(lapply(delta_data, function(d) d$delta))
y_range3 <- range(all_delta, na.rm = TRUE)
pad3 <- max(abs(y_range3)) * 0.1
y_range3 <- c(-max(abs(y_range3)) - pad3, max(abs(y_range3)) + pad3)

plot(NA, xlim = c(1950, 2017), ylim = y_range3,
     xlab = "Year", ylab = expression(Delta * " Mean Similarity / year"),
     main = "Year-on-Year Change in Mean Similarity -- All Groups",
     cex.main = 1.1)
grid(col = "grey90")
abline(h = 0, lty = 2, col = "grey40")

for (grp in active_groups) {
  d <- delta_data[[grp]]
  ## Smoothed trend (loess)
  if (length(d$years) > 10) {
    lo <- loess(d$delta ~ d$years, span = 0.3)
    lines(d$years, predict(lo), col = group_colours[grp], lwd = 2.5)
  }
}

legend("topleft",
       legend = sprintf("%s (%s)", group_labels[active_groups], active_groups),
       col = group_colours[active_groups],
       lwd = 2.5, cex = 0.85, bg = "white")

dev.off()
cat(sprintf("  Saved: %s\n\n", basename(pdf_site)))


# ===========================================================================
# B) IBRA-LEVEL MULTI-GROUP PLOTS
# ===========================================================================
cat("=== B) IBRA-level multi-group plots ===\n")

## Load IBRA timeseries for each group
cat("--- Loading IBRA timeseries ---\n")
ibra_data <- list()

for (grp in groups) {
  base_dir <- find_run_folder(grp, "base")
  if (is.null(base_dir)) next
  ibra_file <- file.path(base_dir, paste0(grp, "_ibra_timeseries_results.rds"))
  if (!file.exists(ibra_file)) {
    cat(sprintf("  [SKIP] %s -- IBRA timeseries not found\n", grp))
    next
  }
  ibra_data[[grp]] <- readRDS(ibra_file)
  n_reg <- sum(!sapply(ibra_data[[grp]]$region_results, is.null))
  cat(sprintf("  %-5s : loaded (%d regions)\n", grp, n_reg))
}

ibra_groups <- names(ibra_data)
cat(sprintf("\n  Active groups for IBRA: %d\n\n", length(ibra_groups)))

if (length(ibra_groups) < 1) {
  cat("  No IBRA data available -- skipping.\n")
} else {
  ## Collect all region names across all groups
  all_regions <- unique(unlist(lapply(ibra_data, function(d) names(d$region_results))))
  all_regions <- sort(all_regions)
  cat(sprintf("  Total unique IBRA regions: %d\n\n", length(all_regions)))

  ## ---- Summary plot: one line per group, median across all regions ----
  pdf_ibra_summary <- file.path(viz_dir, "multigroup_timeseries_IBRA_summary.pdf")
  pdf(pdf_ibra_summary, width = 14, height = 8)

  ## Compute national median-of-region-means for each group
  par(mar = c(5, 5, 4, 2))

  national_data <- list()
  for (grp in ibra_groups) {
    ib <- ibra_data[[grp]]
    base_years <- c(ib$baseline_year, ib$target_years)

    ## For each region, compute mean similarity per year
    region_means <- list()
    for (reg in names(ib$region_results)) {
      rr <- ib$region_results[[reg]]
      if (is.null(rr) || all(is.na(rr$mat_sim))) next
      region_means[[reg]] <- c(1.0, colMeans(rr$mat_sim, na.rm = TRUE))
    }

    if (length(region_means) == 0) next

    ## Stack into matrix and compute median across regions
    mat <- do.call(rbind, region_means)
    national_data[[grp]] <- list(
      years      = base_years,
      median_sim = apply(mat, 2, median, na.rm = TRUE),
      q25_sim    = apply(mat, 2, quantile, 0.25, na.rm = TRUE),
      q75_sim    = apply(mat, 2, quantile, 0.75, na.rm = TRUE),
      n_regions  = nrow(mat)
    )
  }

  ## Plot: median across IBRA regions
  all_y <- unlist(lapply(national_data, function(d) d$median_sim))
  y_range <- range(all_y, na.rm = TRUE)
  pad <- diff(y_range) * 0.05
  y_range <- y_range + c(-pad, pad)

  plot(NA, xlim = c(1950, 2017), ylim = y_range,
       xlab = "Year",
       ylab = "Median of IBRA Region Mean Similarities",
       main = "Temporal Biodiversity Similarity -- Median Across IBRA Regions\nAll Groups | 30 yr climate window | 1950 baseline",
       cex.main = 1.0)
  grid(col = "grey90")

  for (grp in names(national_data)) {
    d <- national_data[[grp]]
    lines(d$years, d$median_sim, col = group_colours[grp], lwd = 3)
  }

  legend("topleft",
         legend = sprintf("%s (%s, %d regions)", group_labels[names(national_data)],
                          names(national_data),
                          sapply(national_data, function(d) d$n_regions)),
         col = group_colours[names(national_data)],
         lwd = 3, cex = 0.85, bg = "white")

  ## Plot: median with IQR ribbon
  par(mar = c(5, 5, 4, 2))
  all_y2 <- unlist(lapply(national_data, function(d) c(d$q25_sim, d$q75_sim)))
  y_range2 <- range(all_y2, na.rm = TRUE)
  pad2 <- diff(y_range2) * 0.05
  y_range2 <- y_range2 + c(-pad2, pad2)

  plot(NA, xlim = c(1950, 2017), ylim = y_range2,
       xlab = "Year",
       ylab = "Median of IBRA Region Mean Similarities",
       main = "Temporal Biodiversity Similarity -- Median +/- IQR Across IBRA Regions\nAll Groups | 30 yr climate window | 1950 baseline",
       cex.main = 1.0)
  grid(col = "grey90")

  for (grp in rev(names(national_data))) {
    d <- national_data[[grp]]
    polygon(c(d$years, rev(d$years)),
            c(d$q75_sim, rev(d$q25_sim)),
            col = adjustcolor(group_colours[grp], alpha.f = 0.1), border = NA)
  }
  for (grp in names(national_data)) {
    d <- national_data[[grp]]
    lines(d$years, d$median_sim, col = group_colours[grp], lwd = 3)
  }

  legend("topleft",
         legend = sprintf("%s (%s)", group_labels[names(national_data)], names(national_data)),
         col = group_colours[names(national_data)],
         lwd = 3, cex = 0.85, bg = "white")

  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_ibra_summary)))


  ## ---- Per-region plots: small-multiple grid (6 per page) ----
  pdf_ibra_regions <- file.path(viz_dir, "multigroup_timeseries_IBRA_per_region.pdf")
  pdf(pdf_ibra_regions, width = 16, height = 11)

  ## Compute per-region mean similarity for each group
  region_group_data <- list()
  for (reg in all_regions) {
    rg <- list()
    for (grp in ibra_groups) {
      rr <- ibra_data[[grp]]$region_results[[reg]]
      if (is.null(rr) || all(is.na(rr$mat_sim))) next
      rg[[grp]] <- list(
        years    = c(ibra_data[[grp]]$baseline_year, ibra_data[[grp]]$target_years),
        mean_sim = c(1.0, colMeans(rr$mat_sim, na.rm = TRUE))
      )
    }
    if (length(rg) > 0) region_group_data[[reg]] <- rg
  }

  plotted_regions <- names(region_group_data)
  n_plotted <- length(plotted_regions)
  panels_per_page <- 6
  n_pages <- ceiling(n_plotted / panels_per_page)

  for (page in seq_len(n_pages)) {
    start_idx <- (page - 1) * panels_per_page + 1
    end_idx   <- min(page * panels_per_page, n_plotted)
    page_regs <- plotted_regions[start_idx:end_idx]

    par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

    for (reg in page_regs) {
      rg <- region_group_data[[reg]]

      all_y <- unlist(lapply(rg, function(d) d$mean_sim))
      y_range <- range(all_y, na.rm = TRUE)
      if (!all(is.finite(y_range)) || diff(y_range) == 0) { plot.new(); next }
      pad <- diff(y_range) * 0.05
      y_range <- y_range + c(-pad, pad)

      plot(NA, xlim = c(1950, 2017), ylim = y_range,
           xlab = "", ylab = "Mean Similarity",
           main = reg, cex.main = 0.9)
      grid(col = "grey92")

      for (grp in names(rg)) {
        lines(rg[[grp]]$years, rg[[grp]]$mean_sim,
              col = group_colours[grp], lwd = 2)
      }

      ## Small legend (group codes only)
      legend("topleft", legend = names(rg), col = group_colours[names(rg)],
             lwd = 2, cex = 0.55, bg = adjustcolor("white", alpha.f = 0.8), bty = "n")
    }

    ## Fill remaining panels
    remaining <- panels_per_page - length(page_regs)
    if (remaining > 0) for (k in seq_len(remaining)) plot.new()

    mtext(sprintf("Multi-Group Temporal Similarity by IBRA Region (1950 baseline) -- page %d/%d",
                  page, n_pages),
          outer = TRUE, cex = 0.9)
  }

  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_ibra_regions)))
}

cat("\n=== Multi-group line plots complete ===\n")
