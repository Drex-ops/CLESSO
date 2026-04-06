##############################################################################
##
## predict_spatial_eastern_aus.R
##
## Produce spatial Landmark-MDS/RGB biological-community maps cropped to
## eastern Australia (longitude >= western QLD border, ~138°E).
##
## The script runs predict_spatial_lmds() for the full continent, then:
##   1. Crops MDS scores to the eastern region
##   2. Re-stretches RGB values to the edges of the cropped area
##   3. Writes a cropped GeoTIFF + map PDFs
##
## If an existing PCA prediction RDS is found, its I-spline transform is
## reused to skip the expensive climate extraction step.
##
## Usage (command line):
##   Rscript predict_spatial_eastern_aus.R
##
## Environment variables (all inherited from config.R):
##   RECA_SPECIES_GROUP, RECA_WINDOW, RECA_MODEL_TYPE, etc.
##
## Optional overrides:
##   EAST_AUS_LON_MIN    — western longitude boundary (default: 138.0)
##   EAST_AUS_STRETCH    — percentile stretch for RGB (default: 2)
##   EAST_AUS_REF_YEAR   — reference year (default: 2017)
##   EAST_AUS_LANDMARKS  — number of LMDS landmarks (default: 500)
##
##############################################################################

cat("=== Spatial Biological Map (LMDS) — Eastern Australia ===\n\n")

# ---------------------------------------------------------------------------
# 0.  Source config and set parameters
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))
save_config_snapshot()

project_root <- config$project_root
fit_path     <- config$fit_path
ref_raster   <- config$reference_raster
subs_raster  <- config$substrate_raster
npy_src      <- config$npy_src
python_exe   <- config$python_exe
pyper_script <- config$pyper_script
out_dir      <- config$run_output_dir

## ---- Eastern Australia boundary ----
lon_min <- as.numeric(Sys.getenv("EAST_AUS_LON_MIN", unset = "138.0"))

## ---- Reference year ----
ref_year  <- as.integer(Sys.getenv("EAST_AUS_REF_YEAR", unset = "2017"))
ref_month <- 6L

## ---- LMDS parameters ----
n_landmarks       <- as.integer(Sys.getenv("EAST_AUS_LANDMARKS", unset = "500"))
landmark_method   <- "stratified"
use_dissimilarity <- TRUE
n_components      <- 3L
stretch           <- as.numeric(Sys.getenv("EAST_AUS_STRETCH", unset = "2"))

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

cat(sprintf("  Eastern boundary: lon >= %.1f°E\n", lon_min))
cat(sprintf("  Reference year:   %d\n", ref_year))
cat(sprintf("  Landmarks:        %d (%s)\n", n_landmarks, landmark_method))
cat(sprintf("  RGB stretch:      %.1f%% percentile\n\n", stretch))

# ---------------------------------------------------------------------------
# 1.  Source dependencies
# ---------------------------------------------------------------------------
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_spatial.R"))

library(raster)

# ---------------------------------------------------------------------------
# 2.  Load fitted model
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)

cat(sprintf("  Model: %s | climate window = %d yr\n",
            fit$species_group, fit$climate_window))
cat(sprintf("  Mapping year: %d\n\n", ref_year))

modis_tag <- if (isTRUE(fit$add_modis)) "_MODIS" else ""
cond_tag  <- if (isTRUE(fit$add_condition)) "_COND" else ""
dissim_tag <- if (use_dissimilarity) "dissim" else "ecodist"
tag_base  <- sprintf("%s_%dyr_ref%d%s%s",
                     fit$species_group, fit$climate_window,
                     ref_year, modis_tag, cond_tag)

# ---------------------------------------------------------------------------
# 3.  Run Landmark MDS for full Australia
# ---------------------------------------------------------------------------
## Check for a pre-computed PCA result whose I-spline transform we can reuse
pca_tag  <- sprintf("%s_%dyr_ref%d%s_PCA", fit$species_group,
                    fit$climate_window, ref_year, modis_tag)
rds_pca  <- file.path(out_dir, paste0(pca_tag, "_prediction.rds"))

## We need the pre-computed transform + coords to skip climate extraction.
## The PCA result contains the full rgb_stack which has the same grid as our
## ref_raster; however, it may not store the raw I-spline transform.
## We'll try loading the full PCA result_pca object via predict_spatial_rgb().

## Strategy: always call predict_spatial_lmds() with its own extraction.
## If a PCA prediction exists, we load its transform to speed things up.
pre_transform <- NULL
pre_coords    <- NULL

if (file.exists(rds_pca)) {
  cat(sprintf("  Found PCA RDS: %s\n", basename(rds_pca)))
  cat("  (LMDS will still compute its own landmarks + distance matrices)\n\n")
}

cat(sprintf("\n######  Landmark MDS (ref_year = %d, k = %d)  ######\n\n",
            ref_year, n_landmarks))

result_lmds <- predict_spatial_lmds(
  fit               = fit,
  ref_raster        = ref_raster,
  subs_raster       = subs_raster,
  npy_src           = npy_src,
  python_exe        = python_exe,
  pyper_script      = pyper_script,
  ref_year          = ref_year,
  ref_month         = ref_month,
  modis_dir         = if (isTRUE(fit$add_modis)) config$modis_dir else NULL,
  modis_resolution  = config$modis_resolution,
  condition_tif     = if (isTRUE(fit$add_condition)) config$condition_tif_path else NULL,
  n_landmarks       = n_landmarks,
  landmark_method   = landmark_method,
  n_components      = n_components,
  stretch           = stretch,
  use_dissimilarity = use_dissimilarity,
  verbose           = TRUE
)

mds_scores <- result_lmds$mds_scores
coords     <- result_lmds$coords
var_expl   <- result_lmds$variance_explained
n_neg      <- result_lmds$n_neg_eigenvalues

cat(sprintf("\n  Full-Australia: %d pixels, %d MDS dimensions\n",
            nrow(mds_scores), ncol(mds_scores)))

## Save full-Australia LMDS RDS (for potential reuse)
tag_lmds_full <- sprintf("%s_LMDS_%dk_%s", tag_base, n_landmarks, dissim_tag)
rds_lmds_full <- file.path(out_dir, paste0(tag_lmds_full, "_prediction.rds"))
saveRDS(list(
  mds_scores         = mds_scores,
  variance_explained = var_expl,
  eigenvalues        = result_lmds$eigenvalues,
  n_neg_eigenvalues  = n_neg,
  coords             = coords,
  rgb_vals           = result_lmds$rgb_vals,
  landmark_idx       = result_lmds$landmark_idx,
  landmark_coords    = result_lmds$landmark_coords,
  n_landmarks        = n_landmarks,
  use_dissimilarity  = use_dissimilarity,
  fit_metadata       = list(
    species_group    = fit$species_group,
    climate_window   = fit$climate_window,
    ref_year         = ref_year,
    landmark_method  = landmark_method,
    stretch          = stretch
  )
), file = rds_lmds_full)
cat(sprintf("  Saved full LMDS RDS: %s\n", basename(rds_lmds_full)))

## Also save full-Australia GeoTIFF
writeRaster(result_lmds$rgb_stack,
            file.path(out_dir, paste0(tag_lmds_full, "_RGB.tif")),
            format = "GTiff", options = "COMPRESS=LZW",
            overwrite = TRUE, datatype = "INT1U")
cat(sprintf("  Saved full LMDS GeoTIFF: %s_RGB.tif\n", tag_lmds_full))

# ---------------------------------------------------------------------------
# 4.  Crop to eastern Australia
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Cropping to eastern Australia (lon >= %.1f°E) ---\n", lon_min))

east_idx <- which(coords$lon >= lon_min)
cat(sprintf("  Pixels in crop:  %d / %d  (%.1f%%)\n",
            length(east_idx), nrow(coords),
            100 * length(east_idx) / nrow(coords)))

if (length(east_idx) == 0) stop("No pixels found east of the boundary — check lon_min")

scores_east <- mds_scores[east_idx, , drop = FALSE]
coords_east <- coords[east_idx, , drop = FALSE]

# ---------------------------------------------------------------------------
# 5.  Re-stretch RGB to cropped region values
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Re-stretching RGB to eastern values (%.1f%% percentile) ---\n", stretch))

n_comp <- ncol(scores_east)
rgb_east <- matrix(NA_real_, nrow = nrow(scores_east), ncol = n_comp)

for (k in seq_len(n_comp)) {
  v  <- scores_east[, k]
  lo <- quantile(v, stretch / 100, na.rm = TRUE)
  hi <- quantile(v, 1 - stretch / 100, na.rm = TRUE)
  if (hi <= lo) { lo <- min(v, na.rm = TRUE); hi <- max(v, na.rm = TRUE) }
  v_scaled <- (v - lo) / (hi - lo)
  v_scaled <- pmin(pmax(v_scaled, 0), 1)
  rgb_east[, k] <- round(v_scaled * 255)
  cat(sprintf("  Dim%d: lo=%.4f  hi=%.4f  (stretch range for eastern region)\n", k, lo, hi))
}

# ---------------------------------------------------------------------------
# 6.  Build cropped raster and extent
# ---------------------------------------------------------------------------
cat("\n--- Building cropped RGB raster ---\n")

## Determine the crop extent from the actual pixel coordinates plus a buffer
lat_range <- range(coords_east$lat, na.rm = TRUE)
lon_range <- range(coords_east$lon, na.rm = TRUE)
buf <- 0.5   # degrees buffer for cleaner map edges

crop_ext <- extent(lon_range[1] - buf, lon_range[2] + buf,
                   lat_range[1] - buf, lat_range[2] + buf)

## Create a full-extent template, fill in cropped pixels, then crop
template <- raster(ref_raster)

r_layer <- template; values(r_layer) <- NA
g_layer <- template; values(g_layer) <- NA
b_layer <- template; values(b_layer) <- NA

r_layer[coords_east$cell] <- rgb_east[, 1]
g_layer[coords_east$cell] <- rgb_east[, 2]
if (n_comp >= 3) b_layer[coords_east$cell] <- rgb_east[, 3]

rgb_stack <- stack(r_layer, g_layer, b_layer)
names(rgb_stack) <- c("MDS1_red", "MDS2_green", "MDS3_blue")

## Crop the raster to the bounding box of eastern pixels
rgb_crop <- crop(rgb_stack, crop_ext)

cat(sprintf("  Cropped extent: [%.1f, %.1f] × [%.1f, %.1f]\n",
            xmin(crop_ext), xmax(crop_ext), ymin(crop_ext), ymax(crop_ext)))
cat(sprintf("  Raster dims: %d × %d\n", nrow(rgb_crop), ncol(rgb_crop)))

# ---------------------------------------------------------------------------
# 7.  Save cropped GeoTIFF
# ---------------------------------------------------------------------------
tag_east <- sprintf("%s_LMDS_%dk_%s_eastAus", tag_base, n_landmarks, dissim_tag)
tif_path <- file.path(out_dir, paste0(tag_east, "_RGB.tif"))

writeRaster(rgb_crop, tif_path,
            format = "GTiff", options = "COMPRESS=LZW",
            overwrite = TRUE, datatype = "INT1U")
cat(sprintf("\n  Saved GeoTIFF: %s\n", basename(tif_path)))

# ---------------------------------------------------------------------------
# 8.  Map PDF — eastern Australia LMDS RGB
# ---------------------------------------------------------------------------
pdf_path <- file.path(out_dir, paste0(tag_east, "_RGB_map.pdf"))
pdf(pdf_path, width = 10, height = 14)
par(mar = c(3, 3, 5, 1))

plotRGB(rgb_crop, r = 1, g = 2, b = 3, stretch = "none",
        main = sprintf(
          "Biological Space (LMDS) — Eastern Australia\n%s | %d yr window | ref %d | %dk landmarks%s%s",
          fit$species_group, fit$climate_window, ref_year, n_landmarks,
          ifelse(nzchar(modis_tag), " + MODIS", ""),
          ifelse(nzchar(cond_tag),  " + COND",  "")),
        axes = TRUE, cex.main = 1.1)

## Variance + info legend
legend("bottomleft",
       legend = c(
         sprintf("R = Dim1 (%.1f%%)", var_expl[1]),
         if (length(var_expl) >= 2) sprintf("G = Dim2 (%.1f%%)", var_expl[2]) else "G = Dim2",
         if (length(var_expl) >= 3) sprintf("B = Dim3 (%.1f%%)", var_expl[3]) else "B = Dim3",
         sprintf("Total: %.1f%%", sum(var_expl)),
         sprintf("Neg eigenvals: %d/%d", n_neg, n_landmarks),
         "",
         sprintf("Stretch: %.1f%% percentile", stretch),
         sprintf("Crop: lon >= %.1f\u00B0E", lon_min)),
       fill = c("red", "green", "blue", "grey50", NA, NA, NA, NA),
       border = c(rep("black", 4), rep(NA, 4)),
       cex = 0.75, bg = "white")

dev.off()
cat(sprintf("  Saved PDF: %s\n", basename(pdf_path)))

# ---------------------------------------------------------------------------
# 9.  Individual MDS dimension maps — eastern Australia
# ---------------------------------------------------------------------------
pdf_dims <- file.path(out_dir, paste0(tag_east, "_dim_maps.pdf"))
pdf(pdf_dims, width = 10, height = 14)
par(mfrow = c(min(n_comp, 3), 1), mar = c(3, 3, 4, 5))

dim_pals <- list(
  colorRampPalette(c("#F7F7F7", "#B2182B"))(100),
  colorRampPalette(c("#F7F7F7", "#238B45"))(100),
  colorRampPalette(c("#F7F7F7", "#2166AC"))(100)
)
dim_names <- c("Dim1 (Red)", "Dim2 (Green)", "Dim3 (Blue)")

for (k in seq_len(min(n_comp, 3))) {
  score_ras <- raster(ref_raster); values(score_ras) <- NA
  score_ras[coords_east$cell] <- scores_east[, k]
  score_ras <- crop(score_ras, crop_ext)
  ve_str <- if (!is.null(var_expl) && length(var_expl) >= k)
    sprintf(" (%.1f%%)", var_expl[k]) else ""
  plot(score_ras, col = dim_pals[[k]],
       main = sprintf("%s%s — Eastern Australia\n%s | ref %d",
                      dim_names[k], ve_str, fit$species_group, ref_year),
       cex.main = 1.0, axes = TRUE)
}
dev.off()
cat(sprintf("  Saved PDF: %s\n", basename(pdf_dims)))

# ---------------------------------------------------------------------------
# 10. Full-Australia LMDS map PDF (for comparison)
# ---------------------------------------------------------------------------
pdf_full <- file.path(out_dir, paste0(tag_lmds_full, "_RGB_map.pdf"))
pdf(pdf_full, width = 14, height = 10)
par(mar = c(2, 2, 4, 1))

plotRGB(result_lmds$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
        main = sprintf(
          "Biological Space (LMDS) — Full Australia\n%s | %d yr | ref %d | %dk landmarks (%s)",
          fit$species_group, fit$climate_window, ref_year,
          n_landmarks, landmark_method),
        axes = TRUE)

if (!is.null(var_expl)) {
  legend("bottomleft",
         legend = c(sprintf("R=Dim1 (%.1f%%)", var_expl[1]),
                    if (length(var_expl) >= 2) sprintf("G=Dim2 (%.1f%%)", var_expl[2]) else "G=Dim2",
                    if (length(var_expl) >= 3) sprintf("B=Dim3 (%.1f%%)", var_expl[3]) else "B=Dim3",
                    sprintf("Total: %.1f%%", sum(var_expl)),
                    sprintf("Neg eigenvals: %d/%d", n_neg, n_landmarks)),
         fill = c("red", "green", "blue", "grey50", NA),
         border = c(rep("black", 4), NA),
         cex = 0.8, bg = "white")
}
dev.off()
cat(sprintf("  Saved PDF: %s\n", basename(pdf_full)))

# ---------------------------------------------------------------------------
# 11. Save eastern Australia LMDS RDS
# ---------------------------------------------------------------------------
rds_east <- file.path(out_dir, paste0(tag_east, "_prediction.rds"))
saveRDS(list(
  mds_scores         = scores_east,
  variance_explained = var_expl,
  eigenvalues        = result_lmds$eigenvalues,
  n_neg_eigenvalues  = n_neg,
  coords             = coords_east,
  rgb_vals           = rgb_east,
  crop_extent        = crop_ext,
  lon_min            = lon_min,
  stretch            = stretch,
  n_landmarks        = n_landmarks,
  use_dissimilarity  = use_dissimilarity,
  fit_metadata       = list(
    species_group   = fit$species_group,
    climate_window  = fit$climate_window,
    ref_year        = ref_year,
    landmark_method = landmark_method
  )
), file = rds_east)
cat(sprintf("  Saved RDS: %s\n", basename(rds_east)))

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat(sprintf("\n=== Eastern Australia LMDS maps complete ===\n"))
cat(sprintf("  Output directory: %s\n", out_dir))
cat(sprintf("  Files:\n"))
cat(sprintf("    %s\n", basename(tif_path)))
cat(sprintf("    %s\n", basename(pdf_path)))
cat(sprintf("    %s\n", basename(pdf_dims)))
cat(sprintf("    %s\n", basename(pdf_full)))
cat(sprintf("    %s\n", basename(rds_east)))
cat(sprintf("    %s\n", basename(rds_lmds_full)))
