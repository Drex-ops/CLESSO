##############################################################################
##
## predict_spatial_2017_PCA_vs_MDS.R
##
## Generate spatial biological-community maps for 2017 using two methods:
##
##   A. Transform -> PCA -> RGB   (fast, approximate)
##   B. Landmark MDS -> RGB      (rigorous, calibrated dissimilarity)
##
## Both use the same fitted STresiduals GDM and spatial+substrate
## environmental data extracted at ref_year = 2017.  The PCA approach
## is computed first; its I-spline transform is reused by the LMDS
## approach to avoid repeating the expensive climate extraction step.
##
## Outputs (in output directory):
##   - PCA RGB GeoTIFF + map PDF + PC maps PDF
##   - LMDS RGB GeoTIFF + map PDF
##   - Side-by-side comparison PDF
##   - Summary statistics PDF
##
##############################################################################

cat("=== Spatial Biological Map for 2017: PCA vs Landmark MDS ===\n\n")

# ---------------------------------------------------------------------------
# 0.  Source config and set parameters
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))

project_root <- config$project_root
fit_path     <- config$fit_path
ref_raster   <- config$reference_raster
subs_raster  <- config$substrate_raster
npy_src      <- config$npy_src
python_exe   <- config$python_exe
pyper_script <- config$pyper_script
out_dir      <- config$output_dir

## ---- Reference year: 2017 ----
ref_year  <- 2017L
ref_month <- 6L

## ---- PCA parameters ----
pca_method   <- "prcomp"
n_components <- 3L
stretch      <- 2            # percentile stretch for RGB mapping

## ---- Landmark MDS parameters ----
n_landmarks       <- 500L
landmark_method   <- "stratified"   # spatially even coverage
use_dissimilarity <- TRUE           # TRUE = full calibration (intercept -> logit -> ObsTrans)

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1.  Source dependencies
# ---------------------------------------------------------------------------
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_spatial.R"))

library(raster)
library(arrow)

# ---------------------------------------------------------------------------
# 2.  Load fitted model
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)

cat(sprintf("  Model: %s | climate window = %d yr\n",
            fit$species_group, fit$climate_window))
cat(sprintf("  Predictors: %d total (%d spatial, %d substrate, %d temporal)\n",
            length(fit$predictors),
            length(grep("^spat_", fit$predictors)),
            length(setdiff(seq_along(fit$predictors),
                           c(grep("^spat_", fit$predictors),
                             grep("^temp_", fit$predictors)))),
            length(grep("^temp_", fit$predictors))))
cat(sprintf("  Mapping year: %d\n\n", ref_year))

## MODIS suffix for output filenames (derived from fit metadata)
modis_tag <- if (isTRUE(fit$add_modis)) "_MODIS" else ""

# ===========================================================================
# A.  Transform -> PCA -> RGB
# ===========================================================================
cat(sprintf("\n######  APPROACH A: Transform -> PCA -> RGB (ref_year = %d)  ######\n\n",
    ref_year))

result_pca <- predict_spatial_rgb(
  fit          = fit,
  ref_raster   = ref_raster,
  subs_raster  = subs_raster,
  npy_src      = npy_src,
  python_exe   = python_exe,
  pyper_script = pyper_script,
  ref_year     = ref_year,
  ref_month    = ref_month,
  modis_dir    = if (isTRUE(fit$add_modis)) config$modis_dir else NULL,
  modis_resolution = config$modis_resolution,
  pca_method   = pca_method,
  n_components = n_components,
  stretch      = stretch,
  verbose      = TRUE
)

## Save PCA RGB GeoTIFF
tag_pca <- sprintf("%s_%dyr_ref%d%s_PCA", fit$species_group, fit$climate_window, ref_year, modis_tag)
writeRaster(result_pca$rgb_stack,
            file.path(out_dir, paste0(tag_pca, "_RGB.tif")),
            format = "GTiff", options = "COMPRESS=LZW",
            overwrite = TRUE, datatype = "INT1U")
cat(sprintf("  Saved: %s_RGB.tif\n", tag_pca))

## Save PCA RDS (for later reuse)
saveRDS(list(
  pca_scores         = result_pca$pca_scores,
  variance_explained = result_pca$variance_explained,
  coords             = result_pca$coords,
  rgb_vals           = result_pca$rgb_vals,
  active_cols        = result_pca$active_cols,
  predictor_info     = attr(result_pca$transformed, "predictor_info"),
  fit_metadata       = list(
    species_group  = fit$species_group,
    climate_window = fit$climate_window,
    ref_year       = ref_year,
    pca_method     = pca_method,
    stretch        = stretch
  )
), file = file.path(out_dir, paste0(tag_pca, "_prediction.rds")))
cat(sprintf("  Saved: %s_prediction.rds\n", tag_pca))

## ---- PCA RGB Map PDF ----
pdf_pca <- file.path(out_dir, paste0(tag_pca, "_RGB_map.pdf"))
pdf(pdf_pca, width = 14, height = 10)
par(mar = c(2, 2, 4, 1))

plotRGB(result_pca$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
        main = sprintf(
          "Biological Space (PCA of GDM-transformed environment)\n%s | %d yr climate window | ref year %d",
          fit$species_group, fit$climate_window, ref_year),
        axes = TRUE)

if (!is.null(result_pca$variance_explained)) {
  ve <- result_pca$variance_explained
  legend("bottomleft",
         legend = c(sprintf("R = PC1 (%.1f%%)", ve[1]),
                    sprintf("G = PC2 (%.1f%%)", ve[2]),
                    sprintf("B = PC3 (%.1f%%)", ve[3]),
                    sprintf("Total: %.1f%%", sum(ve))),
         fill = c("red", "green", "blue", "grey50"),
         cex = 0.8, bg = "white")
}
dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_pca)))

## ---- PCA Individual PC Maps + Scree Plot ----
pdf_pcs <- file.path(out_dir, paste0(tag_pca, "_PC_maps.pdf"))
pdf(pdf_pcs, width = 16, height = 14)
par(mfrow = c(2, 2), mar = c(3, 3, 4, 5))

pc_pals <- list(
  colorRampPalette(c("#F7F7F7", "#B2182B"))(100),
  colorRampPalette(c("#F7F7F7", "#238B45"))(100),
  colorRampPalette(c("#F7F7F7", "#2166AC"))(100)
)
pc_names <- c("PC1 (Red)", "PC2 (Green)", "PC3 (Blue)")

for (k in 1:3) {
  score_ras <- raster(ref_raster); values(score_ras) <- NA
  score_ras[result_pca$coords$cell] <- result_pca$pca_scores[, k]
  ve_str <- if (!is.null(result_pca$variance_explained))
    sprintf(" (%.1f%%)", result_pca$variance_explained[k]) else ""
  plot(score_ras, col = pc_pals[[k]],
       main = sprintf("%s%s -- %s | ref %d", pc_names[k], ve_str,
                      fit$species_group, ref_year),
       cex.main = 1.0)
}

if (!is.null(result_pca$pca)) {
  n_show <- min(15, length(result_pca$pca$sdev))
  var_pct <- 100 * result_pca$pca$sdev^2 / sum(result_pca$pca$sdev^2)
  cum_pct <- cumsum(var_pct)
  barplot(var_pct[1:n_show], names.arg = 1:n_show,
          col = c("red", "green", "blue", rep("grey70", n_show - 3)),
          xlab = "Principal Component", ylab = "Variance Explained (%)",
          main = "PCA Scree Plot", ylim = c(0, max(var_pct) * 1.2))
  lines(seq_len(n_show) * 1.2 - 0.5, cum_pct[1:n_show],
        type = "b", pch = 19, col = "black", lwd = 2)
  axis(4, at = pretty(c(0, 100)), labels = paste0(pretty(c(0, 100)), "%"))
  mtext("Cumulative %", side = 4, line = 2.5)
}
dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_pcs)))

# ===========================================================================
# B.  Landmark MDS -> RGB  (reuses transform from A)
# ===========================================================================
cat(sprintf("\n\n######  APPROACH B: Landmark MDS -> RGB (ref_year = %d)  ######\n\n",
    ref_year))

result_lmds <- predict_spatial_lmds(
  fit               = fit,
  ref_raster        = ref_raster,
  transformed       = result_pca$transformed,
  coords            = result_pca$coords,
  n_landmarks       = n_landmarks,
  landmark_method   = landmark_method,
  n_components      = n_components,
  stretch           = stretch,
  use_dissimilarity = use_dissimilarity,
  verbose           = TRUE
)

## Save LMDS RGB GeoTIFF
dissim_tag <- if (use_dissimilarity) "dissim" else "ecodist"
tag_lmds <- sprintf("%s_%dyr_ref%d%s_LMDS_%dk_%s",
                    fit$species_group, fit$climate_window, ref_year,
                    modis_tag, n_landmarks, dissim_tag)
writeRaster(result_lmds$rgb_stack,
            file.path(out_dir, paste0(tag_lmds, "_RGB.tif")),
            format = "GTiff", options = "COMPRESS=LZW",
            overwrite = TRUE, datatype = "INT1U")
cat(sprintf("  Saved: %s_RGB.tif\n", tag_lmds))

## ---- LMDS RGB Map PDF ----
pdf_lmds <- file.path(out_dir, paste0(tag_lmds, "_RGB_map.pdf"))
pdf(pdf_lmds, width = 14, height = 10)
par(mar = c(2, 2, 4, 1))

plotRGB(result_lmds$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
        main = sprintf(
          "Biological Space (Landmark MDS of calibrated GDM dissimilarity)\n%s | %d yr | ref %d | %d landmarks (%s)",
          fit$species_group, fit$climate_window, ref_year,
          n_landmarks, landmark_method),
        axes = TRUE)

if (!is.null(result_lmds$variance_explained)) {
  ve <- result_lmds$variance_explained
  legend("bottomleft",
         legend = c(sprintf("R=Dim1 (%.1f%%)", ve[1]),
                    if (length(ve) >= 2) sprintf("G=Dim2 (%.1f%%)", ve[2]) else "G=Dim2",
                    if (length(ve) >= 3) sprintf("B=Dim3 (%.1f%%)", ve[3]) else "B=Dim3",
                    sprintf("Total: %.1f%%", sum(ve)),
                    sprintf("Neg eigenvals: %d/%d",
                            result_lmds$n_neg_eigenvalues, n_landmarks)),
         fill = c("red", "green", "blue", "grey50", NA),
         border = c(rep("black", 4), NA),
         cex = 0.7, bg = "white")
}
dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_lmds)))

# ===========================================================================
# COMPARISON PLOTS
# ===========================================================================
cat("\n\n--- Generating comparison plots ---\n")

# ---------------------------------------------------------------------------
# Plot 1: Side-by-side PCA vs LMDS
# ---------------------------------------------------------------------------
pdf_compare <- file.path(out_dir, sprintf("%s_%dyr_ref%d%s_PCA_vs_LMDS.pdf",
                                           fit$species_group, fit$climate_window, ref_year,
                                           modis_tag))
pdf(pdf_compare, width = 18, height = 10)
par(mfrow = c(1, 2), mar = c(2, 2, 4, 1))

## A: PCA
plotRGB(result_pca$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
        main = sprintf("A: PCA -> RGB\n%s | %d yr | ref %d",
                       fit$species_group, fit$climate_window, ref_year),
        axes = TRUE)
if (!is.null(result_pca$variance_explained)) {
  ve <- result_pca$variance_explained
  legend("bottomleft",
         legend = c(sprintf("R=PC1 (%.1f%%)", ve[1]),
                    sprintf("G=PC2 (%.1f%%)", ve[2]),
                    sprintf("B=PC3 (%.1f%%)", ve[3]),
                    sprintf("Total: %.1f%%", sum(ve))),
         fill = c("red", "green", "blue", "grey50"),
         cex = 0.7, bg = "white")
}

## B: LMDS
plotRGB(result_lmds$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
        main = sprintf("B: Landmark MDS -> RGB (%d landmarks, %s)\n%s | %d yr | ref %d",
                       n_landmarks, dissim_tag,
                       fit$species_group, fit$climate_window, ref_year),
        axes = TRUE)
if (!is.null(result_lmds$variance_explained)) {
  ve <- result_lmds$variance_explained
  legend("bottomleft",
         legend = c(sprintf("R=Dim1 (%.1f%%)", ve[1]),
                    if (length(ve) >= 2) sprintf("G=Dim2 (%.1f%%)", ve[2]) else "G=Dim2",
                    if (length(ve) >= 3) sprintf("B=Dim3 (%.1f%%)", ve[3]) else "B=Dim3",
                    sprintf("Total: %.1f%%", sum(ve)),
                    sprintf("Neg eigenvals: %d/%d",
                            result_lmds$n_neg_eigenvalues, n_landmarks)),
         fill = c("red", "green", "blue", "grey50", NA),
         border = c(rep("black", 4), NA),
         cex = 0.7, bg = "white")
}
dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_compare)))

# ---------------------------------------------------------------------------
# Plot 2: Summary diagnostics
# ---------------------------------------------------------------------------
pdf_summary <- file.path(out_dir, sprintf("%s_%dyr_ref%d%s_PCA_vs_LMDS_summary.pdf",
                                           fit$species_group, fit$climate_window, ref_year,
                                           modis_tag))
pdf(pdf_summary, width = 16, height = 12)
par(mfrow = c(2, 2), mar = c(4, 4, 4, 2))

## Panel 1: PCA Scree plot
if (!is.null(result_pca$pca)) {
  n_show <- min(15, length(result_pca$pca$sdev))
  var_pct <- 100 * result_pca$pca$sdev^2 / sum(result_pca$pca$sdev^2)
  cum_pct <- cumsum(var_pct)
  barplot(var_pct[1:n_show], names.arg = 1:n_show,
          col = c("red", "green", "blue", rep("grey70", n_show - 3)),
          xlab = "Principal Component", ylab = "Variance (%)",
          main = sprintf("PCA Scree Plot (ref %d)", ref_year),
          ylim = c(0, max(var_pct) * 1.2))
  lines(seq_len(n_show) * 1.2 - 0.5, cum_pct[1:n_show],
        type = "b", pch = 19, col = "black", lwd = 2)
  axis(4, at = pretty(c(0, 100)), labels = paste0(pretty(c(0, 100)), "%"))
  mtext("Cumulative %", side = 4, line = 2.5)
}

## Panel 2: LMDS Eigenvalue spectrum
evals <- result_lmds$eigenvalues
n_show <- min(30, length(evals))
cols <- rep("grey70", n_show)
cols[evals[1:n_show] > 0] <- "#2166AC"
cols[evals[1:n_show] < 0] <- "#B2182B"
cols[1] <- "red"; if (n_show >= 2) cols[2] <- "green"; if (n_show >= 3) cols[3] <- "blue"

barplot(evals[1:n_show], names.arg = 1:n_show, col = cols,
        main = sprintf("LMDS Eigenvalue Spectrum (%d landmarks, ref %d)",
                       n_landmarks, ref_year),
        xlab = "Dimension", ylab = "Eigenvalue")
abline(h = 0, lty = 2, col = "grey40")
legend("topright",
       legend = c(sprintf("Positive: %d", sum(evals > 1e-10)),
                  sprintf("Negative: %d", sum(evals < -1e-10)),
                  sprintf("Top 3 = %.1f%%", sum(result_lmds$variance_explained))),
       fill = c("#2166AC", "#B2182B", "grey50"),
       cex = 0.8, bg = "white")

## Panel 3: Landmark distribution map
plot(raster(ref_raster),
     col = colorRampPalette(c("grey95", "grey80"))(20),
     legend = FALSE,
     main = sprintf("Landmark distribution (%s, %d landmarks)",
                    landmark_method, n_landmarks))
lm_coords <- result_lmds$landmark_coords
points(lm_coords$lon, lm_coords$lat, pch = 20, col = "#D62728", cex = 0.5)

## Panel 4: Landmark pairwise dissimilarity distribution
d_tri <- result_lmds$D_LL[upper.tri(result_lmds$D_LL)]
if (use_dissimilarity) {
  hist(d_tri, breaks = 50, col = "#74ADD1", border = "white",
       main = "Landmark Pairwise Dissimilarity",
       xlab = "Calibrated Dissimilarity (ObsTrans)", ylab = "Count")
} else {
  hist(d_tri, breaks = 50, col = "#74ADD1", border = "white",
       main = "Landmark Pairwise Ecological Distance",
       xlab = "Ecological Distance (Manhattan)", ylab = "Count")
}
abline(v = median(d_tri), lty = 2, col = "red", lwd = 2)
legend("topright", legend = sprintf("Median: %.3f", median(d_tri)),
       lty = 2, col = "red", lwd = 2, cex = 0.8, bg = "white")

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_summary)))

# ---------------------------------------------------------------------------
# Plot 3: Predictor contribution barplot (shared)
# ---------------------------------------------------------------------------
pdf_contrib <- file.path(out_dir, sprintf("%s_%dyr_ref%d%s_spatial_predictor_contributions.pdf",
                                           fit$species_group, fit$climate_window, ref_year,
                                           modis_tag))
pdf(pdf_contrib, width = 12, height = 8)

pred_info <- attr(result_pca$transformed, "predictor_info")
active_preds <- pred_info[pred_info$matched & pred_info$type != "temporal", ]

if (nrow(active_preds) > 0) {
  csp <- c(0, cumsum(fit$splines))
  active_preds$coef_sum <- sapply(active_preds$idx, function(i) {
    sum(fit$coefficients[(csp[i] + 1):(csp[i] + fit$splines[i])])
  })
  active_preds <- active_preds[order(-active_preds$coef_sum), ]

  par(mar = c(5, 12, 4, 2))
  bar_cols <- ifelse(active_preds$type == "spatial", "#2166AC", "#D95F02")

  barplot(rev(active_preds$coef_sum),
          horiz = TRUE, las = 1,
          names.arg = rev(active_preds$predictor),
          col = rev(bar_cols), border = NA,
          xlab = "Coefficient Sum (biological importance)",
          main = sprintf("Spatial Predictor Importance -- %s | %d yr | ref %d",
                         fit$species_group, fit$climate_window, ref_year),
          cex.names = 0.7)

  legend("bottomright",
         legend = c("Spatial climate", "Substrate"),
         fill = c("#2166AC", "#D95F02"),
         cex = 0.9, bg = "white")
}
dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_contrib)))

# ---------------------------------------------------------------------------
# Save comparison RDS
# ---------------------------------------------------------------------------
rds_compare <- file.path(out_dir, sprintf("%s_%dyr_ref%d%s_PCA_vs_LMDS.rds",
                                           fit$species_group, fit$climate_window, ref_year,
                                           modis_tag))
saveRDS(list(
  pca = list(
    variance_explained = result_pca$variance_explained,
    n_active_cols      = length(result_pca$active_cols),
    method             = pca_method
  ),
  lmds = list(
    variance_explained = result_lmds$variance_explained,
    n_landmarks        = n_landmarks,
    landmark_method    = landmark_method,
    use_dissimilarity  = use_dissimilarity,
    n_neg_eigenvalues  = result_lmds$n_neg_eigenvalues,
    eigenvalues        = result_lmds$eigenvalues
  ),
  fit_metadata = list(
    species_group  = fit$species_group,
    climate_window = fit$climate_window,
    ref_year       = ref_year,
    stretch        = stretch
  )
), file = rds_compare)
cat(sprintf("  Saved: %s\n", basename(rds_compare)))

cat(sprintf("\n=== Spatial maps for %d complete (PCA + LMDS) ===\n", ref_year))
