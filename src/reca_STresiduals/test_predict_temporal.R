##############################################################################
##
## test_predict_temporal.R
##
## Test script: load a fitted STresiduals GDM, sample 100 random non-NA
## points from the reference raster, predict temporal biodiversity change
## between 1950 and 2017, and plot the results.
##
##############################################################################

cat("=== Test: Temporal GDM Prediction ===\n\n")

# ---------------------------------------------------------------------------
# 0. Source config
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))

project_root <- config$project_root
fit_path     <- config$fit_path
ref_raster   <- config$reference_raster
npy_src      <- config$npy_src
python_exe   <- config$python_exe
pyper_script <- config$pyper_script

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
# 3. Sample 100 random non-NA points from reference raster
# ---------------------------------------------------------------------------
cat("\n--- Sampling 100 random non-NA points from reference raster ---\n")
if (!file.exists(ref_raster)) stop(paste("Reference raster not found:", ref_raster))
ras <- raster(ref_raster)

set.seed(42)
## sampleRandom returns coords + values; na.rm=TRUE skips ocean/NA cells
samp <- as.data.frame(sampleRandom(ras, size = 100, na.rm = TRUE, xy = TRUE))
colnames(samp)[1:2] <- c("lon", "lat")
cat(sprintf("  Sampled %d points (extent: lon [%.2f, %.2f], lat [%.2f, %.2f])\n",
            nrow(samp),
            min(samp$lon), max(samp$lon),
            min(samp$lat), max(samp$lat)))

## Build prediction points: same locations, 1950 -> 2017
pts <- data.frame(
  lon   = samp$lon,
  lat   = samp$lat,
  year1 = 1950L,
  year2 = 2017L
)

# ---------------------------------------------------------------------------
# 4. Run temporal prediction
# ---------------------------------------------------------------------------
cat("\n--- Running temporal prediction (1950 -> 2017) ---\n")
result <- predict_temporal_gdm(
  fit          = fit,
  points       = pts,
  npy_src      = npy_src,
  python_exe   = python_exe,
  pyper_script = pyper_script,
  verbose      = TRUE
)

cat("\n--- Results summary ---\n")
print(summary(result[, c("temporal_distance", "linear_predictor",
                          "predicted_prob", "dissimilarity")]))

# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
cat("\n--- Generating plots ---\n")
out_dir <- config$run_output_dir
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## Colour palette: blue (low change) -> red (high change)
n_cols  <- 100
pal     <- colorRampPalette(c("#2166AC", "#67A9CF", "#D1E5F0",
                               "#FDDBC7", "#EF8A62", "#B2182B"))(n_cols)

## Map dissimilarity to colour index
diss       <- result$dissimilarity
diss_range <- range(diss, na.rm = TRUE)
col_idx    <- pmax(1, pmin(n_cols,
                round((diss - diss_range[1]) / max(diff(diss_range), 1e-10) * (n_cols - 1)) + 1))
pt_cols    <- pal[col_idx]

## ---- Plot 1: Map of temporal dissimilarity ----
pdf_map <- file.path(out_dir, paste0(fit$species_group, "_temporal_dissimilarity_map.pdf"))
pdf(pdf_map, width = 10, height = 8)

plot(ras, col = grey(0.9), legend = FALSE, axes = TRUE,
     main = sprintf("Temporal Biodiversity Dissimilarity (1950 -> 2017)\n%s | climate window = %d yr",
                    fit$species_group, fit$climate_window),
     xlab = "Longitude", ylab = "Latitude")
points(result$lon, result$lat, pch = 19, cex = 1.5, col = pt_cols)

## Colour legend
legend_vals <- pretty(diss_range, n = 5)
legend_cols <- pal[pmax(1, pmin(n_cols,
  round((legend_vals - diss_range[1]) / max(diff(diss_range), 1e-10) * (n_cols - 1)) + 1))]
legend("bottomleft",
       legend = sprintf("%.3f", legend_vals),
       col    = legend_cols,
       pch    = 19, pt.cex = 1.5,
       title  = "Dissimilarity",
       bg     = "white", cex = 0.8)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_map)))

## ---- Plot 2: Multi-panel diagnostics ----
pdf_diag <- file.path(out_dir, paste0(fit$species_group, "_temporal_prediction_diagnostics.pdf"))
pdf(pdf_diag, width = 10, height = 10)

par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3, 1))

## Panel A: Histogram of temporal ecological distance
hist(result$temporal_distance, breaks = 20, col = "#67A9CF", border = "white",
     main = "Temporal Ecological Distance",
     xlab = "Distance (1950 -> 2017)", ylab = "Frequency")
abline(v = mean(result$temporal_distance, na.rm = TRUE), col = "red", lwd = 2, lty = 2)
legend("topright", legend = sprintf("mean = %.4f", mean(result$temporal_distance, na.rm = TRUE)),
       col = "red", lty = 2, lwd = 2, cex = 0.8)

## Panel B: Histogram of dissimilarity
hist(result$dissimilarity, breaks = 20, col = "#EF8A62", border = "white",
     main = "Temporal Dissimilarity",
     xlab = "Dissimilarity (1950 -> 2017)", ylab = "Frequency")
abline(v = mean(result$dissimilarity, na.rm = TRUE), col = "red", lwd = 2, lty = 2)
legend("topright", legend = sprintf("mean = %.4f", mean(result$dissimilarity, na.rm = TRUE)),
       col = "red", lty = 2, lwd = 2, cex = 0.8)

## Panel C: Dissimilarity vs latitude
plot(result$lat, result$dissimilarity, pch = 19, col = pt_cols, cex = 1.2,
     xlab = "Latitude", ylab = "Dissimilarity",
     main = "Dissimilarity vs Latitude")
if (nrow(result) > 3) {
  lo <- loess(dissimilarity ~ lat, data = result)
  ord <- order(result$lat)
  lines(result$lat[ord], predict(lo)[ord], col = "black", lwd = 2)
}

## Panel D: Per-predictor contributions (stacked bar-like summary)
contrib <- attr(result, "pred_contributions")
temp_idx <- grep("^temp_", colnames(contrib))
if (length(temp_idx) > 0) {
  temp_contrib <- contrib[, temp_idx, drop = FALSE]
  mean_contrib <- colMeans(temp_contrib, na.rm = TRUE)
  mean_contrib <- sort(mean_contrib, decreasing = TRUE)
  ## Shorten names for display
  short_names <- gsub("temp_", "", names(mean_contrib))
  short_names <- substr(short_names, 1, 25)

  barplot(mean_contrib, names.arg = short_names, las = 2, col = "#2166AC",
          main = "Mean Predictor Contributions",
          ylab = "Mean contribution to temporal distance",
          cex.names = 0.65)
} else {
  plot.new()
  text(0.5, 0.5, "No temporal predictors", cex = 1.2)
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_diag)))

## ---- Plot 3: Dissimilarity on longitude × latitude (bubble plot) ----
pdf_bubble <- file.path(out_dir, paste0(fit$species_group, "_temporal_bubble_map.pdf"))
pdf(pdf_bubble, width = 10, height = 8)

## Scale point sizes by dissimilarity
sz <- 1 + 3 * (diss - diss_range[1]) / max(diff(diss_range), 1e-10)

plot(ras, col = grey(seq(0.95, 0.85, length = 10)), legend = FALSE, axes = TRUE,
     main = sprintf("Temporal Change Magnitude (1950 -> 2017)\n%s | bubble size ∝ dissimilarity",
                    fit$species_group),
     xlab = "Longitude", ylab = "Latitude")
points(result$lon, result$lat, pch = 21, cex = sz, bg = pt_cols, col = "grey30", lwd = 0.5)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_bubble)))

cat("\n=== Test complete ===\n")
