##############################################################################
##
## test_predict_temporal_timeseries.R
##
## Test script: load a fitted STresiduals GDM, sample 100 random non-NA
## points from the reference raster, predict temporal biodiversity change
## for a time series of year-pairs (1950 vs 1951, 1950 vs 1952, ...,
## 1950 vs 2017), and plot the trajectories.
##
##############################################################################

cat("=== Test: Temporal GDM Time-Series Prediction ===\n\n")

# ---------------------------------------------------------------------------
# 0. Source config and set parameters
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))

project_root     <- config$project_root
fit_path         <- config$fit_path
ref_raster       <- config$reference_raster
npy_src          <- config$npy_src
python_exe       <- config$python_exe
pyper_script     <- config$pyper_script
modis_dir        <- if (isTRUE(config$add_modis)) config$modis_dir else NULL
modis_resolution <- if (isTRUE(config$add_modis)) config$modis_resolution else "1km"

## Time-series parameters
## The climate .npy data has a fixed year range (geonpy_start_year to
## geonpy_end_year).  When MODIS is enabled we also want to stay within
## the MODIS year range.  Use the intersection of both constraints.
climate_end <- config$geonpy_end_year      # e.g. 2017

if (isTRUE(config$add_modis)) {
  baseline_year <- config$modis_start_year
  last_year     <- min(config$modis_end_year, climate_end)
  target_years  <- (baseline_year + 1L):last_year
  cat(sprintf("  MODIS enabled -> time-series %d–%d (capped by climate data end %d)\n",
              baseline_year, last_year, climate_end))
} else {
  baseline_year <- 1950L
  target_years  <- 1951L:climate_end
}

# ---------------------------------------------------------------------------
# 1. Source dependencies
# ---------------------------------------------------------------------------
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_temporal.R"))

library(raster)
library(arrow)

# ---------------------------------------------------------------------------
# 2. Load fit object and print summary
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)   # loads 'fit'
summarise_temporal_gdm(fit)

# ---------------------------------------------------------------------------
# 3. Sample 1000 random non-NA points from reference raster
# ---------------------------------------------------------------------------
cat("\n--- Sampling 1000 random non-NA points from reference raster ---\n")
if (!file.exists(ref_raster)) stop(paste("Reference raster not found:", ref_raster))
ras <- raster(ref_raster)

set.seed(42)
samp <- as.data.frame(sampleRandom(ras, size = 1000, na.rm = TRUE, xy = TRUE))
colnames(samp)[1:2] <- c("lon", "lat")
n_sites <- nrow(samp)
cat(sprintf("  Sampled %d points (lon [%.2f, %.2f], lat [%.2f, %.2f])\n",
            n_sites, min(samp$lon), max(samp$lon),
            min(samp$lat), max(samp$lat)))

# ---------------------------------------------------------------------------
# 4. Run predictions for each target year
# ---------------------------------------------------------------------------
n_years <- length(target_years)
cat(sprintf("\n--- Running time-series predictions: %d -> {%d..%d} (%d steps, %d sites) ---\n",
            baseline_year, min(target_years), max(target_years), n_years, n_sites))

## Storage: matrix [n_sites x n_years] for each output variable
mat_distance <- matrix(NA_real_, nrow = n_sites, ncol = n_years)
mat_dissim   <- matrix(NA_real_, nrow = n_sites, ncol = n_years)
mat_prob     <- matrix(NA_real_, nrow = n_sites, ncol = n_years)
colnames(mat_distance) <- colnames(mat_dissim) <- colnames(mat_prob) <- target_years

t0 <- proc.time()

for (yi in seq_along(target_years)) {
  yr <- target_years[yi]
  elapsed <- (proc.time() - t0)["elapsed"]
  if (yi > 1) {
    rate <- (yi - 1) / elapsed
    eta  <- (n_years - yi + 1) / rate
    cat(sprintf("  [%s] Year %d (%d/%d) | %.0fs elapsed | est. %.1f min remaining\n",
                format(Sys.time(), "%H:%M:%S"), yr, yi, n_years, elapsed, eta / 60))
  } else {
    cat(sprintf("  [%s] Year %d (%d/%d)\n", format(Sys.time(), "%H:%M:%S"), yr, yi, n_years))
  }
  flush.console()

  pts <- data.frame(
    lon   = samp$lon,
    lat   = samp$lat,
    year1 = baseline_year,
    year2 = yr
  )

  res <- tryCatch(
    predict_temporal_gdm(
      fit              = fit,
      points           = pts,
      npy_src          = npy_src,
      python_exe       = python_exe,
      pyper_script     = pyper_script,
      modis_dir        = modis_dir,
      modis_resolution = modis_resolution,
      verbose          = FALSE
    ),
    error = function(e) {
      cat(sprintf("  *** Year %d FAILED: %s\n", yr, conditionMessage(e)))
      NULL
    }
  )

  if (!is.null(res)) {
    mat_distance[, yi] <- res$temporal_distance
    mat_dissim[, yi]   <- res$dissimilarity
    mat_prob[, yi]     <- res$predicted_prob
  }
}

total_time <- (proc.time() - t0)["elapsed"]
cat(sprintf("\n  Time-series complete: %d years x %d sites in %.1f min\n",
            n_years, n_sites, total_time / 60))

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
cat("\n--- Time-series summary ---\n")
mean_dissim <- colMeans(mat_dissim, na.rm = TRUE)
n_ok <- sum(is.finite(mean_dissim))
cat(sprintf("  Valid year-columns: %d / %d\n", n_ok, n_years))
if (n_ok == 0) {
  cat("  WARNING: All predictions failed -- no valid results to plot.\n")
  cat("  Re-run with verbose=TRUE in predict_temporal_gdm to diagnose.\n")
  cat(sprintf("\n=== Time-series test complete (%.1f min) ===\n", total_time / 60))
  q(save = "no")
}
cat(sprintf("  Mean dissimilarity range: [%.4f (%d), %.4f (%d)]\n",
            min(mean_dissim, na.rm = TRUE), target_years[which.min(mean_dissim)],
            max(mean_dissim, na.rm = TRUE), target_years[which.max(mean_dissim)]))

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
cat("\n--- Generating plots ---\n")
out_dir <- config$run_output_dir
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## MODIS suffix for output filenames (derived from fit metadata)
modis_tag <- if (isTRUE(fit$add_modis)) "_MODIS" else ""

## Colour palette for sites (by latitude, south=blue -> north=red)
lat_order  <- order(samp$lat)
lat_pal    <- colorRampPalette(c("#2166AC", "#67A9CF", "#D1E5F0",
                                  "#FDDBC7", "#EF8A62", "#B2182B"))(n_sites)
site_cols  <- character(n_sites)
site_cols[lat_order] <- lat_pal

## ============ Plot 1: Spaghetti plot -- all sites =========================
pdf_spag <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_test_timeseries_spaghetti.pdf"))
pdf(pdf_spag, width = 12, height = 7)

par(mar = c(5, 5, 4, 2))
y_range <- range(mat_dissim, na.rm = TRUE)
plot(NA, xlim = range(target_years), ylim = y_range,
     xlab = "Year", ylab = "Temporal Dissimilarity",
     main = sprintf("Temporal Biodiversity Dissimilarity (%d baseline)\n%s | %d sites | %d yr climate window",
                    baseline_year, fit$species_group, n_sites, fit$climate_window),
     cex.main = 1.1)

for (i in seq_len(n_sites)) {
  lines(target_years, mat_dissim[i, ], col = adjustcolor(site_cols[i], alpha.f = 0.4), lwd = 0.8)
}

## Mean trajectory
lines(target_years, mean_dissim, col = "black", lwd = 3)

## Confidence band (mean +/- 1 SD)
sd_dissim <- apply(mat_dissim, 2, sd, na.rm = TRUE)
polygon(c(target_years, rev(target_years)),
        c(mean_dissim + sd_dissim, rev(mean_dissim - sd_dissim)),
        col = adjustcolor("grey40", alpha.f = 0.15), border = NA)

legend("topleft",
       legend = c("Individual sites", "Mean", "Mean +/- 1 SD"),
       col = c("grey50", "black", adjustcolor("grey40", alpha.f = 0.3)),
       lwd = c(1, 3, 8), lty = c(1, 1, 1), cex = 0.9, bg = "white")

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_spag)))

## ============ Plot 2: Mean + quantile ribbon =============================
pdf_ribbon <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_test_timeseries_ribbon.pdf"))
pdf(pdf_ribbon, width = 12, height = 7)

par(mar = c(5, 5, 4, 2))
q10 <- apply(mat_dissim, 2, quantile, 0.10, na.rm = TRUE)
q25 <- apply(mat_dissim, 2, quantile, 0.25, na.rm = TRUE)
q50 <- apply(mat_dissim, 2, quantile, 0.50, na.rm = TRUE)
q75 <- apply(mat_dissim, 2, quantile, 0.75, na.rm = TRUE)
q90 <- apply(mat_dissim, 2, quantile, 0.90, na.rm = TRUE)

plot(NA, xlim = range(target_years), ylim = range(c(q10, q90), na.rm = TRUE),
     xlab = "Year", ylab = "Temporal Dissimilarity",
     main = sprintf("Temporal Dissimilarity Trajectory (%d baseline)\n%s | quantile ribbons across %d sites",
                    baseline_year, fit$species_group, n_sites),
     cex.main = 1.1)

## 10-90% band
polygon(c(target_years, rev(target_years)),
        c(q90, rev(q10)),
        col = adjustcolor("#2166AC", alpha.f = 0.15), border = NA)
## 25-75% band
polygon(c(target_years, rev(target_years)),
        c(q75, rev(q25)),
        col = adjustcolor("#2166AC", alpha.f = 0.25), border = NA)
## Median
lines(target_years, q50, col = "#2166AC", lwd = 3)
## Mean
lines(target_years, mean_dissim, col = "#B2182B", lwd = 2, lty = 2)

legend("topleft",
       legend = c("Median", "Mean", "25th–75th pctile", "10th–90th pctile"),
       col = c("#2166AC", "#B2182B",
               adjustcolor("#2166AC", alpha.f = 0.4),
               adjustcolor("#2166AC", alpha.f = 0.2)),
       lwd = c(3, 2, 8, 8), lty = c(1, 2, 1, 1), cex = 0.9, bg = "white")

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_ribbon)))

## ============ Plot 3: Heatmap (sites × years) ============================
pdf_heat <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_test_timeseries_heatmap.pdf"))
pdf(pdf_heat, width = 14, height = 8)

par(mar = c(5, 5, 4, 6))
heat_pal <- colorRampPalette(c("#2166AC", "#67A9CF", "#D1E5F0",
                                "#FDDBC7", "#EF8A62", "#B2182B"))(100)

## Reorder sites by latitude (south at bottom)
ord <- order(samp$lat)
image(x = target_years, y = seq_len(n_sites),
      z = t(mat_dissim[ord, ]),
      col = heat_pal,
      xlab = "Year", ylab = "Site (ordered by latitude, south -> north)",
      main = sprintf("Temporal Dissimilarity Heatmap (%d baseline)\n%s | %d yr climate window",
                     baseline_year, fit$species_group, fit$climate_window),
      cex.main = 1.1, axes = FALSE)
axis(1)
axis(2, at = seq(1, n_sites, length.out = 5),
     labels = sprintf("%.1f°", samp$lat[ord][seq(1, n_sites, length.out = 5)]),
     las = 1, cex.axis = 0.8)
box()

## Colour bar
fields_available <- requireNamespace("fields", quietly = TRUE)
if (fields_available) {
  fields::image.plot(legend.only = TRUE, zlim = range(mat_dissim, na.rm = TRUE),
                     col = heat_pal, legend.mar = 4,
                     legend.lab = "Dissimilarity")
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_heat)))

## ============ Plot 4: Multi-panel -- ecological distance + dissimilarity + rate
pdf_multi <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_test_timeseries_multipanel.pdf"))
pdf(pdf_multi, width = 12, height = 12)

par(mfrow = c(3, 1), mar = c(4.5, 5, 3, 2))

## Panel A: Mean temporal ecological distance
mean_dist <- colMeans(mat_distance, na.rm = TRUE)
sd_dist   <- apply(mat_distance, 2, sd, na.rm = TRUE)

plot(target_years, mean_dist, type = "l", lwd = 3, col = "#238B45",
     xlab = "Year", ylab = "Temporal Ecological Distance",
     main = sprintf("Mean Temporal Ecological Distance (%d baseline) | %s",
                    baseline_year, fit$species_group))
polygon(c(target_years, rev(target_years)),
        c(mean_dist + sd_dist, rev(pmax(0, mean_dist - sd_dist))),
        col = adjustcolor("#238B45", alpha.f = 0.15), border = NA)

## Panel B: Mean dissimilarity
plot(target_years, mean_dissim, type = "l", lwd = 3, col = "#B2182B",
     xlab = "Year", ylab = "Temporal Dissimilarity",
     main = sprintf("Mean Temporal Dissimilarity (%d baseline) | %s",
                    baseline_year, fit$species_group))
polygon(c(target_years, rev(target_years)),
        c(mean_dissim + sd_dissim, rev(pmax(0, mean_dissim - sd_dissim))),
        col = adjustcolor("#B2182B", alpha.f = 0.15), border = NA)

## Panel C: Year-on-year rate of change in dissimilarity
delta_dissim <- diff(mean_dissim)
delta_years  <- target_years[-1]
plot(delta_years, delta_dissim, type = "h", lwd = 2,
     col = ifelse(delta_dissim >= 0, "#B2182B", "#2166AC"),
     xlab = "Year", ylab = expression(Delta * " Dissimilarity / year"),
     main = sprintf("Year-on-Year Change in Mean Dissimilarity | %s",
                    fit$species_group))
abline(h = 0, lty = 2, col = "grey50")
## Smoothed trend
if (length(delta_years) > 10) {
  lo <- loess(delta_dissim ~ delta_years, span = 0.3)
  lines(delta_years, predict(lo), col = "black", lwd = 2)
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_multi)))

## ============ Plot 5: Selected site trajectories =========================
pdf_sites <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_test_timeseries_selected_sites.pdf"))
pdf(pdf_sites, width = 12, height = 8)

par(mar = c(5, 5, 4, 8), xpd = TRUE)

## Pick 10 sites evenly spaced by latitude
sel_idx <- lat_order[round(seq(1, n_sites, length.out = 10))]
sel_pal <- colorRampPalette(c("#2166AC", "#D1E5F0", "#FDDBC7", "#B2182B"))(10)

y_range_sel <- range(mat_dissim[sel_idx, ], na.rm = TRUE)
plot(NA, xlim = range(target_years), ylim = y_range_sel,
     xlab = "Year", ylab = "Temporal Dissimilarity",
     main = sprintf("Selected Site Trajectories (%d baseline)\n%s | 10 sites by latitude",
                    baseline_year, fit$species_group),
     cex.main = 1.1)

for (k in seq_along(sel_idx)) {
  lines(target_years, mat_dissim[sel_idx[k], ], col = sel_pal[k], lwd = 2)
}

## Legend outside plot
legend("topright", inset = c(-0.15, 0),
       legend = sprintf("%.1f°, %.1f°", samp$lon[sel_idx], samp$lat[sel_idx]),
       col = sel_pal, lwd = 2, cex = 0.7, bg = "white",
       title = "lon, lat")

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_sites)))

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
rds_file <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_test_timeseries_results.rds"))
saveRDS(list(
  baseline_year  = baseline_year,
  target_years   = target_years,
  sites          = samp[, c("lon", "lat")],
  mat_distance   = mat_distance,
  mat_dissim     = mat_dissim,
  mat_prob       = mat_prob,
  mean_dissim    = mean_dissim,
  fit_metadata   = list(
    species_group  = fit$species_group,
    climate_window = fit$climate_window,
    w_ratio        = fit$w_ratio,
    intercept      = fit$intercept,
    D2             = fit$D2
  )
), file = rds_file)
cat(sprintf("  Saved: %s\n", basename(rds_file)))

cat(sprintf("\n=== Time-series test complete (%.1f min) ===\n", total_time / 60))
