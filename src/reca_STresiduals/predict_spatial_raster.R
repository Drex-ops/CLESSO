##############################################################################
##
## predict_spatial_raster.R
##
## Generate a biological-space RGB map from a fitted STresiduals GDM.
##
## For each pixel, the spatial+substrate environmental variables are
## transformed through the model's I-spline basis and weighted by
## coefficients. PCA on the transformed space produces 3 components
## mapped to R, G, B channels — pixels with similar colour = similar
## predicted biological community.
##
## Outputs:
##   - 3-band GeoTIFF (RGB)
##   - RGB composite map PDF
##   - PC individual maps PDF
##   - Per-predictor contribution barplot PDF
##
##############################################################################

cat("=== Spatial GDM → RGB Biological Turnover Map ===\n\n")

# ---------------------------------------------------------------------------
# 0. Source config and set parameters
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

## Reference year for spatial climate extraction
## (mid-point of the reference climate period 1946-1975)
ref_year  <- 1960L
ref_month <- 6L

## PCA parameters
pca_method   <- "prcomp"
n_components <- 3L
stretch      <- 2    # percentile stretch

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
# 2. Load fit object
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)

cat(sprintf("  Model: %s | climate window = %d yr\n",
            fit$species_group, fit$climate_window))
cat(sprintf("  Predictors: %d total (%d spatial climate, %d substrate, %d temporal)\n",
            length(fit$predictors),
            length(grep("^spat_", fit$predictors)),
            length(setdiff(seq_along(fit$predictors),
                           c(grep("^spat_", fit$predictors),
                             grep("^temp_", fit$predictors)))),
            length(grep("^temp_", fit$predictors))))

# ---------------------------------------------------------------------------
# 3. Run spatial RGB prediction
# ---------------------------------------------------------------------------
result <- predict_spatial_rgb(
  fit          = fit,
  ref_raster   = ref_raster,
  subs_raster  = subs_raster,
  env_spatial  = NULL,       # will extract from geonpy
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

# ---------------------------------------------------------------------------
# 4. Save RGB GeoTIFF
# ---------------------------------------------------------------------------
cat("\n--- Saving outputs ---\n")

tif_file <- file.path(out_dir, sprintf("%s_%dyr_ref%d_spatial_RGB.tif",
                                        fit$species_group, fit$climate_window, ref_year))
writeRaster(result$rgb_stack, tif_file, format = "GTiff",
            options = c("COMPRESS=LZW"), overwrite = TRUE, datatype = "INT1U")
cat(sprintf("  Saved: %s\n", basename(tif_file)))

## Save RDS with full results
rds_file <- file.path(out_dir, sprintf("%s_%dyr_ref%d_spatial_prediction.rds",
                                        fit$species_group, fit$climate_window, ref_year))
saveRDS(list(
  pca_scores         = result$pca_scores,
  variance_explained = result$variance_explained,
  coords             = result$coords,
  rgb_vals           = result$rgb_vals,
  active_cols        = result$active_cols,
  predictor_info     = attr(result$transformed, "predictor_info"),
  fit_metadata       = list(
    species_group  = fit$species_group,
    climate_window = fit$climate_window,
    ref_year       = ref_year,
    pca_method     = pca_method,
    stretch        = stretch
  )
), file = rds_file)
cat(sprintf("  Saved: %s\n", basename(rds_file)))

# ---------------------------------------------------------------------------
# 5. Plot 1: RGB composite map
# ---------------------------------------------------------------------------
cat("\n--- Generating plots ---\n")

pdf_rgb <- file.path(out_dir, sprintf("%s_%dyr_ref%d_spatial_RGB_map.pdf",
                                       fit$species_group, fit$climate_window, ref_year))
pdf(pdf_rgb, width = 14, height = 10)

par(mar = c(2, 2, 4, 1))

## Build RGB image from raster
r_ras <- result$rgb_stack[[1]]
g_ras <- result$rgb_stack[[2]]
b_ras <- result$rgb_stack[[3]]

plotRGB(stack(r_ras, g_ras, b_ras), r = 1, g = 2, b = 3,
        stretch = "none",
        main = sprintf("Biological Space (RGB from PCA of GDM-transformed environment)\n%s | %d yr climate window | ref year %d",
                       fit$species_group, fit$climate_window, ref_year),
        axes = TRUE)

## Add variance explained annotation
if (!is.null(result$variance_explained)) {
  ve <- result$variance_explained
  legend("bottomleft",
         legend = c(sprintf("R = PC1 (%.1f%%)", ve[1]),
                    sprintf("G = PC2 (%.1f%%)", ve[2]),
                    sprintf("B = PC3 (%.1f%%)", ve[3]),
                    sprintf("Total: %.1f%%", sum(ve))),
         fill = c("red", "green", "blue", "grey50"),
         cex = 0.8, bg = "white")
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_rgb)))

# ---------------------------------------------------------------------------
# 6. Plot 2: Individual PC maps
# ---------------------------------------------------------------------------
pdf_pcs <- file.path(out_dir, sprintf("%s_%dyr_ref%d_spatial_PC_maps.pdf",
                                       fit$species_group, fit$climate_window, ref_year))
pdf(pdf_pcs, width = 16, height = 14)

par(mfrow = c(2, 2), mar = c(3, 3, 4, 5))

## Colour palettes
pc_pals <- list(
  colorRampPalette(c("#F7F7F7", "#B2182B"))(100),   # PC1 red
  colorRampPalette(c("#F7F7F7", "#238B45"))(100),   # PC2 green
  colorRampPalette(c("#F7F7F7", "#2166AC"))(100)    # PC3 blue
)
pc_names <- c("PC1 (Red)", "PC2 (Green)", "PC3 (Blue)")

for (k in 1:3) {
  ## Create continuous raster from PCA scores
  score_ras <- raster(ref_raster)
  values(score_ras) <- NA
  score_ras[result$coords$cell] <- result$pca_scores[, k]

  ve_str <- if (!is.null(result$variance_explained)) {
    sprintf(" (%.1f%% var)", result$variance_explained[k])
  } else ""

  plot(score_ras, col = pc_pals[[k]],
       main = sprintf("%s%s\n%s | %d yr | ref %d",
                      pc_names[k], ve_str,
                      fit$species_group, fit$climate_window, ref_year),
       cex.main = 1.0)
}

## Panel 4: Scree plot
if (!is.null(result$pca)) {
  n_show <- min(15, length(result$pca$sdev))
  var_pct <- 100 * result$pca$sdev^2 / sum(result$pca$sdev^2)
  cum_pct <- cumsum(var_pct)

  barplot(var_pct[1:n_show], names.arg = 1:n_show,
          col = c("red", "green", "blue", rep("grey70", n_show - 3)),
          xlab = "Principal Component", ylab = "Variance Explained (%)",
          main = "PCA Scree Plot",
          ylim = c(0, max(var_pct) * 1.2))
  lines(seq_len(n_show) * 1.2 - 0.5, cum_pct[1:n_show],
        type = "b", pch = 19, col = "black", lwd = 2)
  axis(4, at = pretty(c(0, 100)), labels = paste0(pretty(c(0, 100)), "%"))
  mtext("Cumulative %", side = 4, line = 2.5)
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_pcs)))

# ---------------------------------------------------------------------------
# 7. Plot 3: Predictor contribution barplot
# ---------------------------------------------------------------------------
pdf_contrib <- file.path(out_dir, sprintf("%s_%dyr_ref%d_spatial_predictor_contributions.pdf",
                                           fit$species_group, fit$climate_window, ref_year))
pdf(pdf_contrib, width = 12, height = 8)

pred_info <- attr(result$transformed, "predictor_info")
active_preds <- pred_info[pred_info$matched & pred_info$type != "temporal", ]

if (nrow(active_preds) > 0) {
  ## Sum of coefficient weights per predictor
  csp <- c(0, cumsum(fit$splines))
  active_preds$coef_sum <- sapply(active_preds$idx, function(i) {
    sum(fit$coefficients[(csp[i] + 1):(csp[i] + fit$splines[i])])
  })
  active_preds <- active_preds[order(-active_preds$coef_sum), ]

  par(mar = c(5, 12, 4, 2))
  bar_cols <- ifelse(active_preds$type == "spatial", "#2166AC", "#D95F02")

  bp <- barplot(rev(active_preds$coef_sum),
                horiz = TRUE, las = 1,
                names.arg = rev(active_preds$predictor),
                col = rev(bar_cols), border = NA,
                xlab = "Coefficient Sum (biological importance)",
                main = sprintf("Spatial Predictor Importance — %s | %d yr",
                               fit$species_group, fit$climate_window),
                cex.names = 0.7)

  legend("bottomright",
         legend = c("Spatial climate", "Substrate"),
         fill = c("#2166AC", "#D95F02"),
         cex = 0.9, bg = "white")
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_contrib)))

# ---------------------------------------------------------------------------
# 8. Plot 4: PCA loading structure (which predictors drive each PC)
# ---------------------------------------------------------------------------
if (!is.null(result$pca)) {
  pdf_loadings <- file.path(out_dir, sprintf("%s_%dyr_ref%d_spatial_PCA_loadings.pdf",
                                              fit$species_group, fit$climate_window, ref_year))
  pdf(pdf_loadings, width = 14, height = 10)

  loadings <- result$pca$rotation[, 1:3]

  ## Aggregate loadings by predictor (sum absolute loadings across splines)
  active_names <- colnames(result$transformed_active)
  pred_base <- gsub("_spl\\d+$", "", active_names)
  unique_preds <- unique(pred_base)

  load_by_pred <- matrix(0, nrow = length(unique_preds), ncol = 3)
  for (p in seq_along(unique_preds)) {
    mask <- pred_base == unique_preds[p]
    for (pc in 1:3) {
      load_by_pred[p, pc] <- sum(abs(loadings[mask, pc]))
    }
  }
  rownames(load_by_pred) <- unique_preds
  colnames(load_by_pred) <- c("PC1", "PC2", "PC3")

  ## Sort by total loading contribution
  ord <- order(-rowSums(load_by_pred))
  load_by_pred <- load_by_pred[ord, , drop = FALSE]

  par(mar = c(5, 12, 4, 2))
  n_show <- min(nrow(load_by_pred), 20)
  load_show <- load_by_pred[seq_len(n_show), , drop = FALSE]

  barplot(t(rev(as.data.frame(t(load_show)))),
          horiz = TRUE, beside = FALSE,
          names.arg = rownames(load_show),
          col = c("red", "green", "blue"),
          las = 1, cex.names = 0.65,
          xlab = "Sum of |PCA Loadings|",
          main = sprintf("Predictor Contributions to PC1-3 — %s",
                         fit$species_group))

  legend("bottomright",
         legend = c("PC1 (Red)", "PC2 (Green)", "PC3 (Blue)"),
         fill = c("red", "green", "blue"),
         cex = 0.85, bg = "white")

  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_loadings)))
}

cat("\n=== Spatial prediction complete ===\n")
