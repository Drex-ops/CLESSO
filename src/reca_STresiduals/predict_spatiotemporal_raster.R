##############################################################################
##
## predict_spatiotemporal_raster.R
##
## Generate spatio-temporal biological maps from a fitted STresiduals GDM.
## Combines spatial biological position with temporal biological change.
##
## Four mapping modes are available (controlled by `mode` parameter):
##
##   Mode 1: "raw"   — Simple concatenation of spatial position + absolute
##                      temporal |I(yr1) - I(yr2)| → PCA → RGB.
##                      (Original behaviour for backward compatibility.)
##
##   Mode 2: "signed" — Same as raw, but uses SIGNED temporal differences
##                      I(yr2) - I(yr1) to preserve direction of change.
##
##   Mode 3: "alpha"  — Normalised blocks + α-weighted mixing.
##                      Both spatial and temporal blocks are standardised
##                      to unit column variance, then weighted by
##                      sqrt(1-α) and sqrt(α). PCA → RGB.
##                      α = 0 → spatial only; α = 1 → temporal only.
##
##   Mode 4: "hsl"    — Two-stage HSL bivariate map:
##                      Hue ← spatial community type (atan2 of PC1, PC2)
##                      Saturation ← spatial distinctiveness
##                      Lightness ← temporal change magnitude
##
## Outputs (per mode):
##   - 3-band GeoTIFF (RGB)
##   - RGB composite map PDF
##   - Mode-specific diagnostics (PC maps, HSL channels, etc.)
##   - Predictor contribution barplot(s) PDF
##
##############################################################################

cat("=== Spatio-temporal GDM → RGB Biological Turnover Map ===\n\n")

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

## ---- Year pair ----
year1 <- 1950L
year2 <- 2017L

## Reference year for spatial climate extraction
ref_year  <- year1
ref_month <- 6L

## ---- Which modes to run ----
## Set to TRUE/FALSE to control which are produced
run_raw    <- TRUE    # Mode 1: original unweighted concatenation (absolute diffs)
run_signed <- TRUE    # Mode 2: signed temporal differences
run_alpha  <- TRUE    # Mode 3: α-weighted mixing (multiple α values)
run_hsl    <- TRUE    # Mode 4: HSL bivariate map

## Mode 3 settings
alpha_values <- c(0.25, 0.5, 0.75)   # sweep of mixing weights

## Mode 4 settings
sat_fixed    <- 0.85                  # fixed saturation (NULL for variable)
light_range  <- c(0.25, 0.90)        # dark = more change
light_invert <- TRUE                  # TRUE = dark = more change

## PCA parameters (modes 1-3)
pca_method   <- "prcomp"
n_components <- 3L
stretch      <- 2

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
cat(sprintf("  Year pair: %d → %d\n\n", year1, year2))

# ---------------------------------------------------------------------------
# Helper:  plot an RGB result (shared by modes 1–3)
# ---------------------------------------------------------------------------
.plot_rgb_composite <- function(result, tag, mode_label) {
  pdf_rgb <- file.path(out_dir, paste0(tag, "_RGB_map.pdf"))
  pdf(pdf_rgb, width = 14, height = 10)
  par(mar = c(2, 2, 4, 1))

  plotRGB(result$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
          main = sprintf(
            "%s\n%s | %d yr | %d → %d",
            mode_label, fit$species_group, fit$climate_window, year1, year2),
          axes = TRUE)

  if (!is.null(result$variance_explained)) {
    ve <- result$variance_explained
    info <- c(sprintf("R=PC1 (%.1f%%)", ve[1]),
              sprintf("G=PC2 (%.1f%%)", ve[2]),
              sprintf("B=PC3 (%.1f%%)", ve[3]),
              sprintf("Total: %.1f%%", sum(ve)),
              sprintf("Spat/subs cols: %d", result$n_spat_active),
              sprintf("Temporal cols: %d", result$n_temp_active))
    if (!is.null(result$alpha))
      info <- c(info, sprintf("alpha = %.2f", result$alpha))
    if (!is.null(result$signed_temporal))
      info <- c(info, sprintf("signed = %s", result$signed_temporal))

    n_info <- length(info)
    legend("bottomleft",
           legend = info,
           fill = c("red", "green", "blue", "grey50",
                     rep(NA, n_info - 4)),
           border = c(rep("black", 4), rep(NA, n_info - 4)),
           cex = 0.7, bg = "white")
  }
  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_rgb)))
}

.plot_pc_maps <- function(result, tag) {
  pdf_pcs <- file.path(out_dir, paste0(tag, "_PC_maps.pdf"))
  pdf(pdf_pcs, width = 16, height = 14)
  par(mfrow = c(2, 2), mar = c(3, 3, 4, 5))

  pc_pals <- list(
    colorRampPalette(c("#F7F7F7", "#B2182B"))(100),
    colorRampPalette(c("#F7F7F7", "#238B45"))(100),
    colorRampPalette(c("#F7F7F7", "#2166AC"))(100))
  pc_names <- c("PC1 (Red)", "PC2 (Green)", "PC3 (Blue)")

  for (k in 1:3) {
    score_ras <- raster(ref_raster); values(score_ras) <- NA
    score_ras[result$coords$cell] <- result$pca_scores[, k]
    ve_str <- if (!is.null(result$variance_explained))
      sprintf(" (%.1f%%)", result$variance_explained[k]) else ""
    plot(score_ras, col = pc_pals[[k]],
         main = sprintf("%s%s — %d → %d", pc_names[k], ve_str, year1, year2),
         cex.main = 1.0)
  }

  if (!is.null(result$pca)) {
    n_show <- min(15, length(result$pca$sdev))
    var_pct <- 100 * result$pca$sdev^2 / sum(result$pca$sdev^2)
    cum_pct <- cumsum(var_pct)
    barplot(var_pct[1:n_show], names.arg = 1:n_show,
            col = c("red", "green", "blue", rep("grey70", n_show - 3)),
            xlab = "Principal Component", ylab = "Variance (%)",
            main = "Scree Plot", ylim = c(0, max(var_pct) * 1.2))
    lines(seq_len(n_show) * 1.2 - 0.5, cum_pct[1:n_show],
          type = "b", pch = 19, col = "black", lwd = 2)
    axis(4, at = pretty(c(0, 100)), labels = paste0(pretty(c(0, 100)), "%"))
    mtext("Cumulative %", side = 4, line = 2.5)
  }
  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_pcs)))
}

.save_rgb_tif <- function(result, tag) {
  tif <- file.path(out_dir, paste0(tag, "_RGB.tif"))
  writeRaster(result$rgb_stack, tif, format = "GTiff",
              options = "COMPRESS=LZW", overwrite = TRUE, datatype = "INT1U")
  cat(sprintf("  Saved: %s\n", basename(tif)))
}

# ===========================================================================
# MODE 1: Raw (original) — absolute temporal diffs, unweighted concatenation
# ===========================================================================
if (run_raw) {
  cat("\n\n######  MODE 1: Raw (absolute diffs, unweighted)  ######\n\n")
  result_raw <- predict_spatiotemporal_rgb(
    fit = fit, ref_raster = ref_raster, subs_raster = subs_raster,
    npy_src = npy_src, python_exe = python_exe, pyper_script = pyper_script,
    year1 = year1, year2 = year2, ref_year = ref_year, ref_month = ref_month,
    signed_temporal = FALSE, alpha = NULL,
    pca_method = pca_method, n_components = n_components, stretch = stretch,
    verbose = TRUE)

  tag_raw <- sprintf("%s_%dyr_%d_to_%d_ST_raw",
                     fit$species_group, fit$climate_window, year1, year2)
  .save_rgb_tif(result_raw, tag_raw)
  .plot_rgb_composite(result_raw, tag_raw,
                      "Spatio-temporal PCA (raw absolute diffs, unweighted)")
  .plot_pc_maps(result_raw, tag_raw)
}

# ===========================================================================
# MODE 2: Signed — signed temporal diffs, unweighted concatenation
# ===========================================================================
if (run_signed) {
  cat("\n\n######  MODE 2: Signed temporal diffs  ######\n\n")
  result_signed <- predict_spatiotemporal_rgb(
    fit = fit, ref_raster = ref_raster, subs_raster = subs_raster,
    npy_src = npy_src, python_exe = python_exe, pyper_script = pyper_script,
    year1 = year1, year2 = year2, ref_year = ref_year, ref_month = ref_month,
    signed_temporal = TRUE, alpha = NULL,
    pca_method = pca_method, n_components = n_components, stretch = stretch,
    verbose = TRUE)

  tag_signed <- sprintf("%s_%dyr_%d_to_%d_ST_signed",
                        fit$species_group, fit$climate_window, year1, year2)
  .save_rgb_tif(result_signed, tag_signed)
  .plot_rgb_composite(result_signed, tag_signed,
                      "Spatio-temporal PCA (signed diffs, unweighted)")
  .plot_pc_maps(result_signed, tag_signed)
}

# ===========================================================================
# MODE 3: Alpha-weighted — normalised blocks with mixing parameter
# ===========================================================================
if (run_alpha) {
  for (a in alpha_values) {
    cat(sprintf("\n\n######  MODE 3: Alpha = %.2f  ######\n\n", a))
    result_a <- predict_spatiotemporal_rgb(
      fit = fit, ref_raster = ref_raster, subs_raster = subs_raster,
      npy_src = npy_src, python_exe = python_exe, pyper_script = pyper_script,
      year1 = year1, year2 = year2, ref_year = ref_year, ref_month = ref_month,
      signed_temporal = TRUE, alpha = a, normalise_blocks = TRUE,
      pca_method = pca_method, n_components = n_components, stretch = stretch,
      verbose = TRUE)

    tag_a <- sprintf("%s_%dyr_%d_to_%d_ST_alpha%.0f",
                     fit$species_group, fit$climate_window, year1, year2, a * 100)
    .save_rgb_tif(result_a, tag_a)
    .plot_rgb_composite(result_a, tag_a,
                        sprintf("Spatio-temporal PCA (alpha=%.2f, signed, normalised)", a))
    .plot_pc_maps(result_a, tag_a)
  }
}

# ===========================================================================
# MODE 4: HSL bivariate map
# ===========================================================================
if (run_hsl) {
  cat("\n\n######  MODE 4: HSL Bivariate Map  ######\n\n")

  ## Can reuse transforms from mode 1 or 2 if available
  ## Otherwise HSL function extracts from scratch
  pre_spat <- if (exists("result_raw")) result_raw$transformed_spatial else NULL
  pre_temp <- if (exists("result_raw")) result_raw$transformed_temporal else NULL
  pre_coords <- if (exists("result_raw")) result_raw$coords else NULL

  result_hsl <- predict_spatiotemporal_hsl(
    fit = fit, ref_raster = ref_raster, subs_raster = subs_raster,
    npy_src = npy_src, python_exe = python_exe, pyper_script = pyper_script,
    year1 = year1, year2 = year2, ref_year = ref_year, ref_month = ref_month,
    trans_spatial = pre_spat, trans_temporal = pre_temp, coords = pre_coords,
    signed_temporal = FALSE,
    sat_fixed = sat_fixed, light_range = light_range,
    light_invert = light_invert, stretch = stretch,
    verbose = TRUE)

  tag_hsl <- sprintf("%s_%dyr_%d_to_%d_ST_HSL",
                     fit$species_group, fit$climate_window, year1, year2)

  ## Save RGB GeoTIFF
  .save_rgb_tif(result_hsl, tag_hsl)

  ## Save diagnostic rasters (hue, saturation, lightness, magnitude)
  writeRaster(result_hsl$magnitude_raster,
              file.path(out_dir, paste0(tag_hsl, "_temporal_magnitude.tif")),
              format = "GTiff", options = "COMPRESS=LZW", overwrite = TRUE)
  cat(sprintf("  Saved: %s\n", paste0(tag_hsl, "_temporal_magnitude.tif")))

  ## ---- HSL Map PDF ----
  pdf_hsl <- file.path(out_dir, paste0(tag_hsl, "_map.pdf"))
  pdf(pdf_hsl, width = 14, height = 10)
  par(mar = c(2, 2, 4, 1))

  plotRGB(result_hsl$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
          main = sprintf(
            paste0("HSL Bivariate Map: Hue = community type, Lightness = temporal change\n",
                   "%s | %d yr | %d → %d"),
            fit$species_group, fit$climate_window, year1, year2),
          axes = TRUE)

  ve <- result_hsl$spatial_ve
  legend("bottomleft",
         legend = c(sprintf("Hue: spatial PC1 (%.1f%%) + PC2 (%.1f%%)", ve[1], ve[2]),
                    sprintf("Saturation: %.2f (fixed)", sat_fixed),
                    sprintf("Lightness: %s = more change",
                            if (light_invert) "darker" else "brighter"),
                    sprintf("L range: [%.2f, %.2f]", light_range[1], light_range[2])),
         cex = 0.75, bg = "white")
  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_hsl)))

  ## ---- HSL Channels PDF ----
  pdf_channels <- file.path(out_dir, paste0(tag_hsl, "_channels.pdf"))
  pdf(pdf_channels, width = 18, height = 12)
  par(mfrow = c(2, 2), mar = c(3, 3, 4, 5))

  ## Hue (circular colour wheel)
  hue_pal <- colorRampPalette(c("red", "yellow", "green", "cyan",
                                 "blue", "magenta", "red"))(256)
  plot(result_hsl$hue_raster, col = hue_pal, zlim = c(0, 1),
       main = sprintf("Hue (community type)\nSpatial PC1/PC2, var = %.1f%%",
                      sum(ve)),
       cex.main = 1.0)

  ## Saturation
  if (is.null(sat_fixed)) {
    plot(result_hsl$sat_raster,
         col = colorRampPalette(c("grey90", "#2B8CBE"))(100),
         main = "Saturation (spatial distinctiveness)", cex.main = 1.0)
  } else {
    plot(result_hsl$light_raster,
         col = colorRampPalette(c("#2B2B2B", "#F7F7F7"))(100),
         main = sprintf("Lightness (%.2f = high change)", light_range[1]),
         cex.main = 1.0)
  }

  ## Temporal magnitude
  mag_pal <- colorRampPalette(c("#F7F7F7", "#FEE08B", "#F46D43",
                                 "#D73027", "#A50026"))(100)
  plot(result_hsl$magnitude_raster, col = mag_pal,
       main = "Temporal Change Magnitude\n(Euclidean norm of temporal spline diffs)",
       cex.main = 1.0)

  ## Histogram of temporal magnitude
  mag_vals <- result_hsl$temporal_magnitude
  hist(mag_vals, breaks = 80, col = "#F46D43", border = "white",
       main = "Distribution of Temporal Change Magnitude",
       xlab = "Magnitude", ylab = "Pixel count")
  abline(v = median(mag_vals, na.rm = TRUE), lty = 2, col = "black", lwd = 2)
  legend("topright",
         legend = c(sprintf("Median: %.4f", median(mag_vals, na.rm = TRUE)),
                    sprintf("Mean: %.4f", mean(mag_vals, na.rm = TRUE)),
                    sprintf("SD: %.4f", sd(mag_vals, na.rm = TRUE))),
         cex = 0.8, bg = "white")

  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_channels)))
}

# ---------------------------------------------------------------------------
# Predictor contribution barplot (shared across modes)
# ---------------------------------------------------------------------------
cat("\n\n--- Predictor contributions ---\n")

tag_base <- sprintf("%s_%dyr_%d_to_%d_spatiotemporal",
                    fit$species_group, fit$climate_window, year1, year2)
pdf_contrib <- file.path(out_dir, paste0(tag_base, "_predictor_contributions.pdf"))
pdf(pdf_contrib, width = 14, height = 10)

csp <- c(0, cumsum(fit$splines))
pred_df <- data.frame(
  predictor = fit$predictors,
  idx       = seq_along(fit$predictors),
  type      = ifelse(grepl("^spat_", fit$predictors), "spatial",
              ifelse(grepl("^temp_", fit$predictors), "temporal", "substrate")),
  coef_sum  = sapply(seq_along(fit$predictors), function(i) {
    sum(fit$coefficients[(csp[i] + 1):(csp[i] + fit$splines[i])])
  }),
  stringsAsFactors = FALSE
)
pred_df <- pred_df[pred_df$coef_sum > 0, ]
pred_df <- pred_df[order(-pred_df$coef_sum), ]

if (nrow(pred_df) > 0) {
  par(mar = c(5, 14, 4, 2))
  bar_cols <- ifelse(pred_df$type == "spatial", "#2166AC",
              ifelse(pred_df$type == "temporal", "#D62728", "#D95F02"))
  barplot(rev(pred_df$coef_sum), horiz = TRUE, las = 1,
          names.arg = rev(pred_df$predictor),
          col = rev(bar_cols), border = NA,
          xlab = "Coefficient Sum (biological importance)",
          main = sprintf("Predictor Importance — %s | %d yr | %d → %d",
                         fit$species_group, fit$climate_window, year1, year2),
          cex.names = 0.65)
  legend("bottomright",
         legend = c("Spatial climate", "Substrate", "Temporal climate"),
         fill = c("#2166AC", "#D95F02", "#D62728"), cex = 0.9, bg = "white")
}
dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_contrib)))

# ---------------------------------------------------------------------------
# Side-by-side comparison if multiple modes were run
# ---------------------------------------------------------------------------
results_list <- list()
if (run_raw    && exists("result_raw"))    results_list[["Mode 1: Raw (|diff|)"]]        <- result_raw
if (run_signed && exists("result_signed")) results_list[["Mode 2: Signed (diff)"]]        <- result_signed
if (run_hsl    && exists("result_hsl"))    results_list[["Mode 4: HSL"]]                   <- result_hsl

## Add alpha results
if (run_alpha) {
  for (a in alpha_values) {
    var_name <- sprintf("result_a")  # last one in loop
  }
  # The alpha results are saved per-iteration; for comparison we re-run the last one
  # (or reference result_a from the loop — it holds the final alpha value)
  if (exists("result_a"))
    results_list[[sprintf("Mode 3: alpha=%.2f", alpha_values[length(alpha_values)])]] <- result_a
}

if (length(results_list) >= 2) {
  cat("\n--- Side-by-side comparison ---\n")
  n_maps <- length(results_list)
  pdf_compare <- file.path(out_dir, paste0(tag_base, "_mode_comparison.pdf"))
  pdf(pdf_compare, width = 7 * min(n_maps, 3), height = 6 * ceiling(n_maps / 3))
  par(mfrow = c(ceiling(n_maps / 3), min(n_maps, 3)), mar = c(2, 2, 4, 1))

  for (nm in names(results_list)) {
    res <- results_list[[nm]]
    plotRGB(res$rgb_stack, r = 1, g = 2, b = 3, stretch = "none",
            main = sprintf("%s\n%d → %d", nm, year1, year2),
            axes = TRUE)
  }
  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_compare)))
}

cat(sprintf("\n=== All spatio-temporal predictions complete (%d → %d) ===\n", year1, year2))
