##############################################################################
##
## predict_spatial_comparison.R
##
## Compare three spatial prediction approaches from a fitted STresiduals GDM:
##
##   A. Transform → PCA → RGB   (predict_spatial_rgb)
##   B. Landmark MDS → RGB      (predict_spatial_lmds)
##   C. Dissimilarity from ref  (spatial_dissimilarity_from_ref)
##
## A is the standard fast approach (no pairwise calibration).
## B is the rigorous pairwise approach via Nyström-approximated MDS.
## C produces a continuous dissimilarity map from chosen reference sites.
##
## The script runs A first, then passes the pre-computed transform to B
## and C to avoid repeating the expensive climate extraction.
##
## Outputs:
##   - RGB GeoTIFFs for A and B
##   - Side-by-side comparison PDF (A vs B)
##   - Dissimilarity rasters + map PDF for C
##   - Summary statistics PDF
##
##############################################################################

cat("=== Spatial Prediction Comparison: PCA vs Landmark MDS vs Reference ===\n\n")

# ---------------------------------------------------------------------------
# 0. Paths and parameters
# ---------------------------------------------------------------------------
project_root <- tryCatch(
  normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", ".."), mustWork = FALSE),
  error = function(e) getwd()
)
if (!dir.exists(project_root)) project_root <- getwd()

fit_path     <- file.path(project_root,
  "src/reca_STresiduals/output/AVES_1mil_30climWin_STresid_biAverage_fittedGDM.RData")
ref_raster   <- file.path(project_root,
  "data/FWPT_mean_Cmax_mean_1946_1975.flt")
subs_raster  <- file.path(project_root,
  "data/SUBS_brk_AVES.grd")
npy_src      <- "/Volumes/PortableSSD/CLIMATE/geonpy"
python_exe   <- file.path(project_root, ".venv/bin/python3")
pyper_script <- file.path(project_root, "src/shared/python/pyper.py")

ref_year  <- 1960L
ref_month <- 6L

## Landmark MDS parameters
n_landmarks     <- 500L
landmark_method <- "stratified"   # spatially even coverage
use_dissimilarity <- TRUE         # TRUE = full calibration; FALSE = raw Manhattan

## PCA parameters (for approach A)
pca_method   <- "prcomp"
n_components <- 3L
stretch      <- 2

## Reference pixels for approach C  (lon, lat)
## Example: tropical (Darwin), arid (Alice Springs), temperate (Sydney),
##          Mediterranean (Perth), sub-alpine (Kosciuszko region)
ref_sites <- data.frame(
  lon   = c(130.85, 133.88, 151.21, 115.86, 148.30),
  lat   = c(-12.46, -23.70, -33.87, -31.95, -36.45),
  label = c("Darwin", "Alice_Springs", "Sydney", "Perth", "Kosciuszko"),
  stringsAsFactors = FALSE
)

out_dir <- file.path(project_root, "src/reca_STresiduals/output")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Source dependencies
# ---------------------------------------------------------------------------
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_spatial.R"))

library(raster)
library(arrow)

# ---------------------------------------------------------------------------
# 2. Load fitted model
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)

cat(sprintf("  Model: %s | climate window = %d yr | %d predictors\n",
            fit$species_group, fit$climate_window, length(fit$predictors)))

# ===========================================================================
# A.  Transform → PCA → RGB
# ===========================================================================
cat("\n\n######  APPROACH A: Transform → PCA → RGB  ######\n\n")

result_pca <- predict_spatial_rgb(
  fit          = fit,
  ref_raster   = ref_raster,
  subs_raster  = subs_raster,
  npy_src      = npy_src,
  python_exe   = python_exe,
  pyper_script = pyper_script,
  ref_year     = ref_year,
  ref_month    = ref_month,
  pca_method   = pca_method,
  n_components = n_components,
  stretch      = stretch,
  verbose      = TRUE
)

## Save RGB GeoTIFF
tag_a <- sprintf("%s_%dyr_ref%d_PCA", fit$species_group, fit$climate_window, ref_year)
writeRaster(result_pca$rgb_stack,
            file.path(out_dir, paste0(tag_a, "_RGB.tif")),
            format = "GTiff", options = "COMPRESS=LZW",
            overwrite = TRUE, datatype = "INT1U")
cat(sprintf("  Saved: %s_RGB.tif\n", tag_a))

# ===========================================================================
# B.  Landmark MDS → RGB  (reuses pre-computed transform from A)
# ===========================================================================
cat("\n\n######  APPROACH B: Landmark MDS → RGB  ######\n\n")

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

## Save RGB GeoTIFF
dissim_tag <- if (use_dissimilarity) "dissim" else "ecodist"
tag_b <- sprintf("%s_%dyr_ref%d_LMDS_%dk_%s",
                 fit$species_group, fit$climate_window, ref_year,
                 n_landmarks, dissim_tag)
writeRaster(result_lmds$rgb_stack,
            file.path(out_dir, paste0(tag_b, "_RGB.tif")),
            format = "GTiff", options = "COMPRESS=LZW",
            overwrite = TRUE, datatype = "INT1U")
cat(sprintf("  Saved: %s_RGB.tif\n", tag_b))

# ===========================================================================
# C.  Dissimilarity from reference sites
# ===========================================================================
cat("\n\n######  APPROACH C: Dissimilarity from Reference Sites  ######\n\n")

result_ref <- spatial_dissimilarity_from_ref(
  fit         = fit,
  ref_pixels  = ref_sites,
  ref_raster  = ref_raster,
  transformed = result_pca$transformed,
  coords      = result_pca$coords,
  verbose     = TRUE
)

## Save dissimilarity rasters
tag_c <- sprintf("%s_%dyr_ref%d_dissim_from_ref",
                 fit$species_group, fit$climate_window, ref_year)
writeRaster(result_ref$dissim_stack,
            file.path(out_dir, paste0(tag_c, ".tif")),
            format = "GTiff", options = "COMPRESS=LZW",
            overwrite = TRUE)
cat(sprintf("  Saved: %s.tif\n", tag_c))

# ===========================================================================
# PLOTS
# ===========================================================================
cat("\n\n--- Generating comparison plots ---\n")

# ---------------------------------------------------------------------------
# Plot 1: Side-by-side A vs B
# ---------------------------------------------------------------------------
pdf_compare <- file.path(out_dir, sprintf("%s_%dyr_ref%d_PCA_vs_LMDS.pdf",
                                           fit$species_group, fit$climate_window, ref_year))
pdf(pdf_compare, width = 18, height = 10)

par(mfrow = c(1, 2), mar = c(2, 2, 4, 1))

## A: PCA
plotRGB(result_pca$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
        main = sprintf("A: Transform → PCA → RGB\n%s | %d yr | ref %d",
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
        main = sprintf("B: Landmark MDS → RGB (%d landmarks, %s)\n%s | %d yr | ref %d",
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
# Plot 2: Reference site dissimilarity maps
# ---------------------------------------------------------------------------
pdf_ref <- file.path(out_dir, sprintf("%s_%dyr_ref%d_dissim_from_ref_maps.pdf",
                                       fit$species_group, fit$climate_window, ref_year))

n_refs <- nlayers(result_ref$dissim_stack)
n_cols <- min(n_refs, 3)
n_rows <- ceiling(n_refs / n_cols)

pdf(pdf_ref, width = 7 * n_cols, height = 6 * n_rows)
par(mfrow = c(n_rows, n_cols), mar = c(3, 3, 4, 5))

dissim_pal <- colorRampPalette(c("#313695", "#4575B4", "#74ADD1", "#ABD9E9",
                                  "#FEE090", "#FDAE61", "#F46D43", "#D73027",
                                  "#A50026"))(100)

for (r in seq_len(n_refs)) {
  ri <- result_ref$ref_info[r, ]

  plot(result_ref$dissim_stack[[r]], col = dissim_pal,
       zlim = c(0, 1),
       main = sprintf("Dissimilarity from %s\n(%.2f°E, %.2f°S)",
                      ri$label, ri$lon, abs(ri$lat)),
       cex.main = 1.0)

  ## Mark the reference point
  points(ri$lon, ri$lat, pch = 21, bg = "white", col = "black",
         cex = 2, lwd = 2)
  text(ri$lon, ri$lat, ri$label, pos = 3, cex = 0.8, font = 2, offset = 0.8)
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_ref)))

# ---------------------------------------------------------------------------
# Plot 3: Eigenvalue spectrum from LMDS + summary stats
# ---------------------------------------------------------------------------
pdf_summary <- file.path(out_dir, sprintf("%s_%dyr_ref%d_prediction_summary.pdf",
                                           fit$species_group, fit$climate_window, ref_year))
pdf(pdf_summary, width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 4, 2))

## Panel 1: Eigenvalue spectrum (LMDS)
evals <- result_lmds$eigenvalues
n_show <- min(30, length(evals))
cols <- rep("grey70", n_show)
cols[evals[1:n_show] > 0] <- "#2166AC"
cols[evals[1:n_show] < 0] <- "#B2182B"
cols[1] <- "red"; if (n_show >= 2) cols[2] <- "green"; if (n_show >= 3) cols[3] <- "blue"

barplot(evals[1:n_show], names.arg = 1:n_show, col = cols,
        main = sprintf("LMDS Eigenvalue Spectrum (%d landmarks)", n_landmarks),
        xlab = "Dimension", ylab = "Eigenvalue")
abline(h = 0, lty = 2, col = "grey40")
legend("topright",
       legend = c(sprintf("Positive: %d", sum(evals > 1e-10)),
                  sprintf("Negative: %d", sum(evals < -1e-10)),
                  sprintf("Top 3 = %.1f%%", sum(result_lmds$variance_explained))),
       fill = c("#2166AC", "#B2182B", "grey50"),
       cex = 0.8, bg = "white")

## Panel 2: Landmark distribution map
plot(raster(ref_raster),
     col = colorRampPalette(c("grey95", "grey80"))(20),
     legend = FALSE,
     main = sprintf("Landmark distribution (%s, %d landmarks)",
                    landmark_method, n_landmarks))
lm_coords <- result_lmds$landmark_coords
points(lm_coords$lon, lm_coords$lat, pch = 20, col = "#D62728", cex = 0.5)

## Panel 3: Comparison histogram — ecological distances from landmarks
## (shows distribution of pairwise distances in the landmark set)
d_tri <- result_lmds$D_LL[upper.tri(result_lmds$D_LL)]
if (use_dissimilarity) {
  hist(d_tri, breaks = 50, col = "#74ADD1", border = "white",
       main = "Landmark Pairwise Dissimilarity Distribution",
       xlab = "Calibrated Dissimilarity (ObsTrans)", ylab = "Count")
} else {
  hist(d_tri, breaks = 50, col = "#74ADD1", border = "white",
       main = "Landmark Pairwise Ecological Distance",
       xlab = "Ecological Distance (Manhattan)", ylab = "Count")
}
abline(v = median(d_tri), lty = 2, col = "red", lwd = 2)
legend("topright", legend = sprintf("Median: %.3f", median(d_tri)),
       lty = 2, col = "red", lwd = 2, cex = 0.8, bg = "white")

## Panel 4: Reference dissimilarity summary
if (n_refs > 0) {
  ref_stats <- data.frame(
    site   = result_ref$ref_info$label,
    median = sapply(seq_len(n_refs), function(r) median(result_ref$dissim_mat[, r], na.rm = TRUE)),
    mean   = sapply(seq_len(n_refs), function(r) mean(result_ref$dissim_mat[, r], na.rm = TRUE)),
    sd     = sapply(seq_len(n_refs), function(r) sd(result_ref$dissim_mat[, r], na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
  ref_stats <- ref_stats[order(ref_stats$median), ]

  par(mar = c(5, 10, 4, 2))
  bp <- barplot(rev(ref_stats$median), horiz = TRUE, las = 1,
                names.arg = rev(ref_stats$site),
                col = "#F46D43", border = NA,
                xlab = "Median Dissimilarity (to all other pixels)",
                main = "Reference Site Median Dissimilarity",
                xlim = c(0, max(ref_stats$median + ref_stats$sd) * 1.1))

  ## Error bars (±sd)
  segments(rev(ref_stats$median - ref_stats$sd), bp,
           rev(ref_stats$median + ref_stats$sd), bp,
           lwd = 2, col = "grey30")
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_summary)))

## Save full results RDS
rds_file <- file.path(out_dir, sprintf("%s_%dyr_ref%d_spatial_comparison.rds",
                                        fit$species_group, fit$climate_window, ref_year))
saveRDS(list(
  approach_A = list(
    variance_explained = result_pca$variance_explained,
    n_active_cols      = length(result_pca$active_cols)
  ),
  approach_B = list(
    variance_explained = result_lmds$variance_explained,
    n_landmarks        = n_landmarks,
    landmark_method    = landmark_method,
    use_dissimilarity  = use_dissimilarity,
    n_neg_eigenvalues  = result_lmds$n_neg_eigenvalues,
    eigenvalues        = result_lmds$eigenvalues
  ),
  approach_C = list(
    ref_info = result_ref$ref_info,
    dissim_summary = data.frame(
      label  = result_ref$ref_info$label,
      median = sapply(seq_len(n_refs), function(r) median(result_ref$dissim_mat[, r], na.rm = TRUE)),
      mean   = sapply(seq_len(n_refs), function(r) mean(result_ref$dissim_mat[, r], na.rm = TRUE)),
      sd     = sapply(seq_len(n_refs), function(r) sd(result_ref$dissim_mat[, r], na.rm = TRUE)),
      min    = sapply(seq_len(n_refs), function(r) min(result_ref$dissim_mat[, r], na.rm = TRUE)),
      max    = sapply(seq_len(n_refs), function(r) max(result_ref$dissim_mat[, r], na.rm = TRUE)),
      stringsAsFactors = FALSE
    )
  ),
  fit_metadata = list(
    species_group  = fit$species_group,
    climate_window = fit$climate_window,
    ref_year       = ref_year,
    pca_method     = pca_method,
    stretch        = stretch
  )
), file = rds_file)
cat(sprintf("  Saved: %s\n", basename(rds_file)))

cat("\n=== Spatial prediction comparison complete ===\n")
