##############################################################################
##
## animate_spatiotemporal_dual_panel_1950_to_2017.R
##
## Animated GIF with two side-by-side panels per frame:
##
##   LEFT:  Temporal biodiversity dissimilarity map  (1950 -> year2)
##   RIGHT: PCA spatial community map                (year2)
##
## The animation sweeps year2 from 1955 to 2017 (5-year steps), showing
## how both temporal change accumulates and how the spatial community
## pattern shifts through time.
##
## Design:
##   1. Pre-compute the final year (1950 -> 2017) to establish:
##      - Fixed colour scale for temporal dissimilarity (global max)
##      - Reference PCA rotation matrix (prcomp on 2017 I-spline data)
##      - Fixed RGB bounds from the reference PCA scores
##   2. Single-pass loop: compute temporal + spatial -> render PNG frame
##   3. Assemble frames into animated GIF (magick)
##
## Consistency:
##   - Temporal panel uses a FIXED colour scale [0, global_max]
##   - PCA spatial panels project each year's I-spline transform through
##     the SAME rotation matrix from the 2017 reference year.  This
##     guarantees perfectly stable colours across frames (no MDS sign/
##     rotation ambiguity, no Procrustes alignment needed).
##   - RGB mapping uses FIXED percentile bounds from the reference year
##
## NOTE: This is computationally intensive (~14 years × climate
## extraction × PCA projection).  Expect several hours of runtime.
##
##############################################################################

cat("=== Dual-Panel Animation: Temporal Change + PCA Spatial ===\n")
cat("=== 1950 -> 2017 ===\n\n")

# ---------------------------------------------------------------------------
# 0.  Paths and parameters
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

## ---- Year range (5-year steps for speed) ----
year1     <- 1950L
year2_seq <- seq(1955L, 2015L, by = 5L)
year2_seq <- c(year2_seq, 2017L)       # always include 2017 as the final year

## ---- Chunk sizes ----
temporal_chunk_size <- 10000L    # pixels per temporal prediction chunk
spatial_chunk_size  <- 50000L    # pixels per spatial climate extraction chunk

## ---- PCA parameters ----
n_components      <- 3L
stretch           <- 2

## ---- Animation settings ----
gif_fps      <- 4
gif_loop     <- 0
frame_width  <- 2400           # total width (two panels)
frame_height <- 1000
ref_month    <- 6L

## ---- Output ----
out_dir   <- file.path(project_root, "src/reca_STresiduals/output")
frame_dir <- file.path(out_dir, "animation_dual_frames")
if (!dir.exists(frame_dir)) dir.create(frame_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1.  Source dependencies
# ---------------------------------------------------------------------------
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_temporal.R"))
source(file.path(project_root, "src/shared/R/predict_spatial.R"))

library(raster)
library(arrow)

# ---------------------------------------------------------------------------
# 2.  Load fitted model
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)
summarise_temporal_gdm(fit)

# ---------------------------------------------------------------------------
# 3.  Reference raster, pixel coordinates, substrate (extracted ONCE)
# ---------------------------------------------------------------------------
cat("\n--- Setting up reference raster and substrate ---\n")
ras <- raster(ref_raster)
subs_brick <- brick(subs_raster)

vals   <- getValues(ras)
non_na <- which(!is.na(vals))
xy     <- xyFromCell(ras, non_na)
n_pixels <- nrow(xy)

coords_df <- data.frame(
  lon  = xy[, 1],
  lat  = xy[, 2],
  cell = non_na,
  stringsAsFactors = FALSE
)

cat(sprintf("  Non-NA pixels: %s\n", format(n_pixels, big.mark = ",")))

## Extract substrate values (same for every year)
cat("  Extracting substrate...\n")
subs_vals <- raster::extract(subs_brick, xy)
colnames(subs_vals) <- paste0(colnames(subs_vals), "_1")
cat(sprintf("  Substrate: %d layers\n", ncol(subs_vals)))

## Build spatial extraction params from fit
spatial_params <- list()
for (ep in fit$env_params) {
  spatial_params[[length(spatial_params) + 1]] <- list(
    variables = ep$variables,
    mstat     = ep$mstat,
    cstat     = ep$cstat,
    window    = fit$climate_window,
    prefix    = paste0("spat_", ep$cstat)
  )
}
geonpy_start <- if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L

## Helper: make raster from values
make_raster <- function(values, template, cell_indices) {
  r <- raster(template)
  r[] <- NA
  r[cell_indices] <- values
  r
}

# ---------------------------------------------------------------------------
# 4.  Helper functions
# ---------------------------------------------------------------------------

## --- 4a.  Extract spatial climate for a given year ----------------------
extract_spatial_climate <- function(year, verbose = TRUE) {
  if (verbose) cat(sprintf("  Extracting spatial climate (year = %d)...\n", year))

  n_chunks <- ceiling(n_pixels / spatial_chunk_size)
  env_parts_list <- vector("list", n_chunks)

  for (ch in seq_len(n_chunks)) {
    row_start <- (ch - 1) * spatial_chunk_size + 1
    row_end   <- min(ch * spatial_chunk_size, n_pixels)
    chunk_xy  <- xy[row_start:row_end, , drop = FALSE]
    n_ch      <- nrow(chunk_xy)

    if (verbose && n_chunks > 1)
      cat(sprintf("    Climate chunk %d/%d (%d px)\n", ch, n_chunks, n_ch))

    pairs <- data.frame(
      Lon1   = chunk_xy[, 1],
      Lat1   = chunk_xy[, 2],
      year1  = rep(as.integer(year), n_ch),
      month1 = rep(as.integer(ref_month), n_ch),
      Lon2   = chunk_xy[, 1],
      Lat2   = chunk_xy[, 2],
      year2  = rep(as.integer(year), n_ch),
      month2 = rep(as.integer(ref_month), n_ch)
    )

    chunk_env_parts <- list()
    for (j in seq_along(spatial_params)) {
      sp <- spatial_params[[j]]
      raw <- gen_windows(
        pairs          = pairs,
        variables      = sp$variables,
        mstat          = sp$mstat,
        cstat          = sp$cstat,
        window         = sp$window,
        npy_src        = npy_src,
        start_year     = geonpy_start,
        python_exe     = python_exe,
        pyper_script   = pyper_script,
        feather_tmpdir = tempdir()
      )
      env_cols <- raw[, grep("_1$", names(raw)), drop = FALSE]
      colnames(env_cols) <- paste(sp$prefix, colnames(env_cols), sep = "_")
      colnames(env_cols) <- gsub("\\d{6}-\\d{6}_", "", colnames(env_cols))
      chunk_env_parts[[j]] <- env_cols
    }

    env_parts_list[[ch]] <- do.call(cbind, chunk_env_parts)
  }

  do.call(rbind, env_parts_list)
}

## --- 4b.  I-spline transform for a given year's spatial env ----------------
##     Returns the active (non-zero-var) columns of the transformed matrix,
##     plus good-pixel mask and coordinates.
ispline_transform_spatial <- function(env_spatial, verbose = TRUE) {

  ## Combine with substrate
  env_df <- cbind(as.data.frame(env_spatial), as.data.frame(subs_vals))

  ## Remove bad rows
  na_rows      <- is.na(rowSums(env_df))
  sentinel_rows <- apply(env_df, 1, function(r) any(r == -9999, na.rm = TRUE))
  bad_rows     <- na_rows | sentinel_rows
  good_mask    <- !bad_rows

  env_clean    <- env_df[good_mask, , drop = FALSE]
  coords_clean <- coords_df[good_mask, , drop = FALSE]

  ## I-spline transform
  transformed <- transform_spatial_gdm(
    fit = fit, env_df = env_clean,
    weight_by_coef = TRUE, spatial_only = TRUE, verbose = verbose
  )

  ## Active columns (non-zero variance)
  col_var     <- apply(transformed, 2, var, na.rm = TRUE)
  active_cols <- which(col_var > 1e-15)
  trans_act   <- transformed[, active_cols, drop = FALSE]

  if (verbose) cat(sprintf("    Active columns: %d / %d\n",
                            length(active_cols), ncol(transformed)))

  list(
    transformed  = trans_act,
    active_cols  = active_cols,
    coords       = coords_clean,
    good_mask    = good_mask
  )
}

## --- 4c.  Project data onto reference PCA rotation ----------------------
##     Uses the FIXED rotation matrix and centre from the reference year.
##     This is deterministic -- no sign/dimension ambiguity.
project_pca_scores <- function(trans_act, pca_rotation, pca_center, verbose = TRUE) {

  ## Match columns: new data may have different active columns.  We
  ## project only the columns that exist in both.  Missing columns
  ## contribute zero (centred value = 0 − 0 = 0).
  centred <- sweep(trans_act, 2, pca_center[colnames(trans_act)])
  scores  <- centred %*% pca_rotation[colnames(trans_act), , drop = FALSE]

  if (verbose) cat(sprintf("    Projected %d pixels × %d PCs\n",
                            nrow(scores), ncol(scores)))
  scores
}

## --- 4d.  Map PCA scores to RGB with fixed bounds ----------------------
scores_to_rgb <- function(scores, lo_bounds, hi_bounds) {
  m   <- ncol(scores)
  rgb <- matrix(NA_real_, nrow = nrow(scores), ncol = m)
  for (k in seq_len(m)) {
    v <- (scores[, k] - lo_bounds[k]) / (hi_bounds[k] - lo_bounds[k])
    rgb[, k] <- round(pmin(pmax(v, 0), 1) * 255)
  }
  rgb
}

## --- 4e.  Compute temporal dissimilarity (chunked) ---------------------
compute_temporal_dissim <- function(yr2, verbose = TRUE) {
  pts <- data.frame(
    lon   = coords_df$lon,
    lat   = coords_df$lat,
    year1 = rep(year1, n_pixels),
    year2 = rep(yr2, n_pixels)
  )

  chunks     <- split(seq_len(n_pixels), ceiling(seq_len(n_pixels) / temporal_chunk_size))
  all_dissim <- rep(NA_real_, n_pixels)

  for (ci in seq_along(chunks)) {
    rows <- chunks[[ci]]
    chunk_result <- tryCatch(
      predict_temporal_gdm(
        fit            = fit,
        points         = pts[rows, ],
        npy_src        = npy_src,
        python_exe     = python_exe,
        pyper_script   = pyper_script,
        feather_tmpdir = tempdir(),
        verbose        = FALSE
      ),
      error = function(e) {
        warning(sprintf("  Year %d, chunk %d failed: %s", yr2, ci, conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(chunk_result))
      all_dissim[rows] <- chunk_result$dissimilarity
  }
  all_dissim
}

# ---------------------------------------------------------------------------
# 5.  Pre-compute reference year (2017)
#     -> temporal dissim max for colour scale
#     -> PCA reference rotation matrix, RGB bounds
# ---------------------------------------------------------------------------
cat("\n\n===== PRE-COMPUTING REFERENCE YEAR (2017) =====\n\n")

## 5a. Temporal dissimilarity 1950 -> 2017 (maximum extent)
cat("--- 5a. Temporal dissimilarity (1950 -> 2017) ---\n")
t0_ref <- proc.time()
dissim_2017 <- compute_temporal_dissim(2017L, verbose = TRUE)
valid_2017  <- !is.na(dissim_2017)
temporal_zlim_max <- max(dissim_2017[valid_2017]) * 1.05
cat(sprintf("  max = %.4f -> zlim_max = %.4f\n",
            max(dissim_2017[valid_2017]), temporal_zlim_max))

## 5b. PCA spatial at 2017 (reference rotation matrix)
cat("\n--- 5b. PCA spatial (2017) -- computing reference rotation ---\n")
env_spatial_2017 <- extract_spatial_climate(2017L, verbose = TRUE)

## I-spline transform
ref_transform <- ispline_transform_spatial(env_spatial_2017, verbose = TRUE)

## Run PCA on the reference year
cat("  Running prcomp() on 2017 I-spline data...\n")
ref_pca <- prcomp(ref_transform$transformed, center = TRUE, scale. = FALSE,
                   rank. = n_components)

## Save the rotation matrix and centre -- these define the FIXED projection
pca_rotation <- ref_pca$rotation[, seq_len(n_components), drop = FALSE]
pca_center   <- ref_pca$center  # named vector (column means)

ref_scores <- ref_pca$x[, seq_len(n_components), drop = FALSE]
ref_coords <- ref_transform$coords

ref_var_explained <- 100 * ref_pca$sdev[seq_len(n_components)]^2 /
                     sum(ref_pca$sdev^2)
cat(sprintf("  Variance explained: PC1=%.1f%%, PC2=%.1f%%, PC3=%.1f%% (total=%.1f%%)\n",
            ref_var_explained[1], ref_var_explained[2], ref_var_explained[3],
            sum(ref_var_explained)))

## Compute fixed RGB bounds from reference year (percentile stretch)
rgb_lo <- numeric(n_components)
rgb_hi <- numeric(n_components)
for (k in seq_len(n_components)) {
  rgb_lo[k] <- quantile(ref_scores[, k], stretch / 100, na.rm = TRUE)
  rgb_hi[k] <- quantile(ref_scores[, k], 1 - stretch / 100, na.rm = TRUE)
  if (rgb_hi[k] <= rgb_lo[k]) {
    rgb_lo[k] <- min(ref_scores[, k], na.rm = TRUE)
    rgb_hi[k] <- max(ref_scores[, k], na.rm = TRUE)
  }
}
cat(sprintf("  RGB bounds fixed from 2017: lo=[%.3f, %.3f, %.3f], hi=[%.3f, %.3f, %.3f]\n",
            rgb_lo[1], rgb_lo[2], rgb_lo[3], rgb_hi[1], rgb_hi[2], rgb_hi[3]))

ref_elapsed <- (proc.time() - t0_ref)["elapsed"]
cat(sprintf("\n  Reference pre-computation: %.1f min\n", ref_elapsed / 60))

# ---------------------------------------------------------------------------
# 6.  Colour palettes for rendering
# ---------------------------------------------------------------------------
## Temporal dissimilarity palette
dissim_pal <- colorRampPalette(c(
  "#F7F7F7", "#FEF0D9", "#FDD49E", "#FDBB84",
  "#FC8D59", "#EF6548", "#D7301F", "#B30000", "#7F0000"
))(256)

# ---------------------------------------------------------------------------
# 7.  Main loop: compute + render each year
# ---------------------------------------------------------------------------
cat(sprintf("\n\n===== MAIN LOOP: %d -> %d (%d frames) =====\n\n",
            min(year2_seq), max(year2_seq), length(year2_seq)))

stats_list <- vector("list", length(year2_seq))
t0_total   <- proc.time()

for (yi in seq_along(year2_seq)) {
  yr2    <- year2_seq[yi]
  t0_yr  <- proc.time()

  cat(sprintf("\n[%s] === Frame %d/%d: %d -> %d ===\n",
              format(Sys.time(), "%H:%M:%S"), yi, length(year2_seq), year1, yr2))
  flush.console()

  ## ------ 7a. Temporal dissimilarity ------------------------------------
  cat("  Computing temporal dissimilarity...\n")
  if (yr2 == 2017L) {
    all_dissim <- dissim_2017   # reuse pre-computed
    cat("    (reusing pre-computed 2017)\n")
  } else {
    all_dissim <- compute_temporal_dissim(yr2, verbose = FALSE)
  }

  valid     <- !is.na(all_dissim)
  yr_mean   <- mean(all_dissim[valid])
  yr_median <- median(all_dissim[valid])
  yr_max    <- max(all_dissim[valid])
  cat(sprintf("    mean=%.4f, median=%.4f, max=%.4f\n", yr_mean, yr_median, yr_max))

  ## ------ 7b. PCA spatial (projected through reference rotation) ----------
  cat("  Computing PCA spatial...\n")
  if (yr2 == 2017L) {
    pca_scores_yr <- ref_scores
    pca_coords_yr <- ref_coords
    ve <- ref_var_explained
    cat("    (reusing pre-computed 2017)\n")
  } else {
    env_spatial_yr <- extract_spatial_climate(yr2, verbose = FALSE)
    yr_transform   <- ispline_transform_spatial(env_spatial_yr, verbose = FALSE)

    ## Project through the FIXED reference rotation matrix
    pca_scores_yr <- project_pca_scores(yr_transform$transformed,
                                         pca_rotation, pca_center, verbose = FALSE)
    pca_coords_yr <- yr_transform$coords
    ve <- ref_var_explained   # same rotation -> same variance structure
  }

  ## Apply fixed RGB mapping
  rgb_vals <- scores_to_rgb(pca_scores_yr, rgb_lo, rgb_hi)

  ## Build PCA RGB raster stack
  r_layer <- make_raster(rgb_vals[, 1], ras, pca_coords_yr$cell)
  g_layer <- make_raster(rgb_vals[, 2], ras, pca_coords_yr$cell)
  b_layer <- if (n_components >= 3) make_raster(rgb_vals[, 3], ras, pca_coords_yr$cell) else
    make_raster(rep(0, nrow(rgb_vals)), ras, pca_coords_yr$cell)
  pca_rgb <- stack(r_layer, g_layer, b_layer)

  ## ------ 7c. Render dual-panel PNG frame --------------------------------
  frame_file <- file.path(frame_dir, sprintf("frame_%04d_%d_to_%d.png",
                                              yi, year1, yr2))
  png(frame_file, width = frame_width, height = frame_height, res = 150)

  layout(matrix(1:2, nrow = 1), widths = c(1, 1))

  ## LEFT: Temporal dissimilarity
  par(mar = c(2, 2, 3, 4.5))
  ras_temporal <- make_raster(all_dissim, ras, non_na)
  plot(ras_temporal, col = dissim_pal, zlim = c(0, temporal_zlim_max),
       axes = TRUE, box = TRUE,
       main = sprintf("Temporal Change: %d -> %d", year1, yr2),
       cex.main = 1.3,
       legend.args = list(text = "Dissimilarity", side = 4, line = 2.5, cex = 0.8))

  ## Year label overlay
  usr <- par("usr")
  text(usr[2] - (usr[2] - usr[1]) * 0.05,
       usr[3] + (usr[4] - usr[3]) * 0.08,
       labels = yr2, cex = 2.5, font = 2, adj = c(1, 0), col = "#333333")
  text(usr[1] + (usr[2] - usr[1]) * 0.02,
       usr[3] + (usr[4] - usr[3]) * 0.08,
       labels = sprintf("mean = %.4f", yr_mean),
       cex = 0.85, adj = c(0, 0), col = "#555555")

  ## RIGHT: PCA spatial community
  par(mar = c(2, 2, 3, 1))
  plotRGB(pca_rgb, r = 1, g = 2, b = 3, stretch = "none",
          main = sprintf("PCA Spatial Community: %d", yr2),
          axes = TRUE)

  ## Variance annotation
  legend("bottomleft",
         legend = c(sprintf("R=PC1 (%.0f%%)", ve[1]),
                    sprintf("G=PC2 (%.0f%%)", ve[2]),
                    sprintf("B=PC3 (%.0f%%)", ve[3])),
         fill = c("red", "green", "blue"),
         cex = 0.65, bg = adjustcolor("white", alpha.f = 0.8), border = NA)

  dev.off()

  ## ------ 7d. Track statistics ------------------------------------------
  stats_list[[yi]] <- data.frame(
    year2         = yr2,
    dissim_mean   = yr_mean,
    dissim_median = yr_median,
    dissim_max    = yr_max,
    pca_ve_pc1    = ve[1],
    pca_ve_pc2    = if (length(ve) >= 2) ve[2] else NA,
    pca_ve_pc3    = if (length(ve) >= 3) ve[3] else NA,
    stringsAsFactors = FALSE
  )

  elapsed_yr    <- (proc.time() - t0_yr)["elapsed"]
  elapsed_total <- (proc.time() - t0_total)["elapsed"]
  est_remain    <- elapsed_total / yi * (length(year2_seq) - yi)

  cat(sprintf("  Frame saved: %s (%.0fs, est. %.1f min remaining)\n",
              basename(frame_file), elapsed_yr, est_remain / 60))
  flush.console()
}

total_compute <- (proc.time() - t0_total)["elapsed"]
cat(sprintf("\n  Main loop complete: %d frames in %.1f min\n",
            length(year2_seq), total_compute / 60))

## Save stats
stats_df  <- do.call(rbind, stats_list)
stats_csv <- file.path(out_dir, sprintf("%s_%dyr_%d_to_%d_dual_animation_stats.csv",
                                         fit$species_group, fit$climate_window,
                                         year1, max(year2_seq)))
write.csv(stats_df, stats_csv, row.names = FALSE)
cat(sprintf("  Saved: %s\n", basename(stats_csv)))

# ---------------------------------------------------------------------------
# 8.  Assemble animated GIF
# ---------------------------------------------------------------------------
cat("\n--- Assembling animated GIF ---\n")

if (!requireNamespace("magick", quietly = TRUE)) {
  cat("  Installing 'magick' package...\n")
  install.packages("magick")
}
library(magick)

frame_files <- sort(list.files(frame_dir, pattern = "^frame_.*\\.png$", full.names = TRUE))
cat(sprintf("  Reading %d frames...\n", length(frame_files)))

frames <- image_read(frame_files)

## Compress: quantise to 256 colours + inter-frame optimisation
cat("  Quantising to 256 colours...\n")
frames <- image_quantize(frames, max = 256, dither = TRUE)

delay <- as.integer(round(100 / gif_fps))
cat(sprintf("  Animating: delay = %d cs, optimize = TRUE...\n", delay))
frames <- image_animate(frames, delay = delay, loop = gif_loop, optimize = TRUE)

gif_file <- file.path(out_dir, sprintf("%s_%dyr_%d_to_%d_dual_temporal_spatial.gif",
                                        fit$species_group, fit$climate_window,
                                        year1, max(year2_seq)))
image_write(frames, path = gif_file)

fsize <- file.size(gif_file) / 1024^2
cat(sprintf("  Saved: %s (%.1f MB)\n", basename(gif_file), fsize))

# ---------------------------------------------------------------------------
# 9.  Summary timeline plot
# ---------------------------------------------------------------------------
cat("\n--- Generating summary timeline ---\n")

pdf_timeline <- file.path(out_dir, sprintf("%s_%dyr_%d_to_%d_dual_timeline.pdf",
                                            fit$species_group, fit$climate_window,
                                            year1, max(year2_seq)))
pdf(pdf_timeline, width = 14, height = 8)

par(mfrow = c(2, 1), mar = c(4, 5, 3, 5))

## Panel 1: temporal dissimilarity over time
plot(stats_df$year2, stats_df$dissim_mean, type = "l", lwd = 2, col = "#D7301F",
     xlab = "Target Year", ylab = "Mean Dissimilarity",
     main = sprintf("Temporal Biodiversity Change from %d  |  %s  |  %d yr window",
                    year1, fit$species_group, fit$climate_window),
     xlim = c(year1, max(year2_seq)),
     ylim = c(0, max(stats_df$dissim_max) * 1.1))

lines(stats_df$year2, stats_df$dissim_median, lty = 2, col = "#FC8D59", lwd = 1.5)
lines(stats_df$year2, stats_df$dissim_max, lty = 1, col = "#7F0000", lwd = 1)

legend("topleft",
       legend = c("Mean", "Median", "Max"),
       col = c("#D7301F", "#FC8D59", "#7F0000"),
       lty = c(1, 2, 1), lwd = c(2, 1.5, 1), cex = 0.85, bg = "white")

## Panel 2: PCA variance explained over time
plot(stats_df$year2, stats_df$pca_ve_pc1, type = "l", lwd = 2, col = "red",
     xlab = "Target Year", ylab = "PCA Variance Explained (%)",
     main = "PCA Variance Explained by Component (fixed rotation from 2017)",
     xlim = c(year1, max(year2_seq)),
     ylim = c(0, max(stats_df$pca_ve_pc1, na.rm = TRUE) * 1.2))

if (!all(is.na(stats_df$pca_ve_pc2)))
  lines(stats_df$year2, stats_df$pca_ve_pc2, lwd = 2, col = "green")
if (!all(is.na(stats_df$pca_ve_pc3)))
  lines(stats_df$year2, stats_df$pca_ve_pc3, lwd = 2, col = "blue")

legend("topright",
       legend = c("PC1", "PC2", "PC3"),
       col = c("red", "green", "blue"),
       lwd = 2, cex = 0.85, bg = "white")

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_timeline)))

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat(sprintf("\n=== Dual-panel animation pipeline complete ===\n"))
cat(sprintf("  Frames:   %s/\n", basename(frame_dir)))
cat(sprintf("  GIF:      %s\n", basename(gif_file)))
cat(sprintf("  Timeline: %s\n", basename(pdf_timeline)))
cat(sprintf("  Stats:    %s\n", basename(stats_csv)))
cat(sprintf("  Total time: %.1f min (reference) + %.1f min (main loop) = %.1f min\n",
            ref_elapsed / 60, total_compute / 60, (ref_elapsed + total_compute) / 60))
