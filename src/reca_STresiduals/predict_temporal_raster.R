##############################################################################
##
## predict_temporal_raster.R
##
## Full-coverage raster prediction: extract every non-NA pixel from the
## reference raster, predict temporal biodiversity change between two years
## using a fitted STresiduals GDM, and write output as GeoTIFF rasters.
##
## Outputs (in output directory):
##   - *_temporal_dissimilarity.tif
##   - *_temporal_distance.tif
##   - *_predicted_prob.tif
##   - *_raster_maps.pdf   (plotted maps)
##
##############################################################################

cat("=== Raster Temporal GDM Prediction ===\n\n")

# ---------------------------------------------------------------------------
# 0. Source config and set parameters
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))
save_config_snapshot()

project_root <- config$project_root
fit_path     <- config$fit_path
ref_raster   <- config$reference_raster
npy_src      <- config$npy_src
python_exe   <- config$python_exe
pyper_script <- config$pyper_script
out_dir      <- config$run_output_dir

## Prediction years
year1 <- 1950L
year2 <- 2017L

## Chunk size: how many pixels to send to gen_windows at once.
## Larger = faster but more memory.  10 000 is a safe default.
chunk_size <- 10000L

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
# 2. Load fit object
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)   # loads 'fit'
summarise_temporal_gdm(fit)

# ---------------------------------------------------------------------------
# 3. Extract ALL non-NA pixel coordinates from reference raster
# ---------------------------------------------------------------------------
cat("\n--- Extracting all non-NA pixel coordinates ---\n")
if (!file.exists(ref_raster)) stop(paste("Reference raster not found:", ref_raster))
ras <- raster(ref_raster)

## Get all cell values -- NA for ocean/missing
vals <- getValues(ras)
non_na <- which(!is.na(vals))
coords <- xyFromCell(ras, non_na)
n_pixels <- nrow(coords)

cat(sprintf("  Raster: %d x %d cells, res = %.4f°\n", ncol(ras), nrow(ras), res(ras)[1]))
cat(sprintf("  Non-NA pixels: %s (of %s total)\n",
            format(n_pixels, big.mark = ","),
            format(ncell(ras), big.mark = ",")))
cat(sprintf("  Extent: lon [%.2f, %.2f], lat [%.2f, %.2f]\n",
            min(coords[, 1]), max(coords[, 1]),
            min(coords[, 2]), max(coords[, 2])))

## Build prediction points table
pts <- data.frame(
  lon   = coords[, 1],
  lat   = coords[, 2],
  year1 = rep(year1, n_pixels),
  year2 = rep(year2, n_pixels)
)

# ---------------------------------------------------------------------------
# 4. Chunked temporal prediction
#
#    predict_temporal_gdm calls gen_windows sequentially per env param.
#    For very large pixel counts we chunk the full set to keep memory
#    and per-call sizes manageable.
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Running chunked prediction (%d -> %d) ---\n", year1, year2))
cat(sprintf("  Chunk size: %s pixels\n", format(chunk_size, big.mark = ",")))

chunks     <- split(seq_len(n_pixels), ceiling(seq_len(n_pixels) / chunk_size))
n_chunks   <- length(chunks)
t0         <- proc.time()

## Pre-allocate result vectors
all_distance    <- rep(NA_real_, n_pixels)
all_linear_pred <- rep(NA_real_, n_pixels)
all_prob        <- rep(NA_real_, n_pixels)
all_dissim      <- rep(NA_real_, n_pixels)

for (ci in seq_along(chunks)) {
  rows <- chunks[[ci]]
  pct  <- round(100 * max(rows) / n_pixels, 1)

  cat(sprintf("  [%s] Chunk %d/%d (pixels %s–%s, %.1f%%)...\n",
              format(Sys.time(), "%H:%M:%S"), ci, n_chunks,
              format(min(rows), big.mark = ","),
              format(max(rows), big.mark = ","), pct))
  flush.console()

  chunk_result <- tryCatch(
    predict_temporal_gdm(
      fit              = fit,
      points           = pts[rows, ],
      npy_src          = npy_src,
      python_exe       = python_exe,
      pyper_script     = pyper_script,
      modis_dir        = config$modis_dir,
      modis_resolution = config$modis_resolution,
      verbose          = FALSE
    ),
    error = function(e) {
      warning(sprintf("Chunk %d failed: %s", ci, conditionMessage(e)))
      NULL
    }
  )

  if (!is.null(chunk_result)) {
    all_distance[rows]    <- chunk_result$temporal_distance
    all_linear_pred[rows] <- chunk_result$linear_predictor
    all_prob[rows]        <- chunk_result$predicted_prob
    all_dissim[rows]      <- chunk_result$dissimilarity
  }

  elapsed   <- (proc.time() - t0)["elapsed"]
  done      <- max(rows)
  rate      <- done / elapsed
  remaining <- (n_pixels - done) / rate
  cat(sprintf("  [%s]   %.0fs elapsed | est. %.1f min remaining\n",
              format(Sys.time(), "%H:%M:%S"), elapsed, remaining / 60))
  flush.console()
}

total_time <- (proc.time() - t0)["elapsed"]
cat(sprintf("\n  Prediction complete: %s pixels in %.1f min (%.0f pixels/sec)\n",
            format(n_pixels, big.mark = ","), total_time / 60, n_pixels / total_time))

# ---------------------------------------------------------------------------
# 5. Build output rasters
# ---------------------------------------------------------------------------
cat("\n--- Building output rasters ---\n")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## MODIS suffix for output filenames (derived from fit metadata)
modis_tag <- if (isTRUE(fit$add_modis)) "_MODIS" else ""

prefix <- sprintf("%s_%dyr_%d_to_%d%s",
                  fit$species_group, fit$climate_window, year1, year2, modis_tag)

## Helper: create a raster from prediction values at non-NA cell positions
make_raster <- function(values, template, cell_indices) {
  r <- raster(template)
  r[] <- NA
  r[cell_indices] <- values
  r
}

ras_dissim   <- make_raster(all_dissim,      ras, non_na)
ras_distance <- make_raster(all_distance,    ras, non_na)
ras_prob     <- make_raster(all_prob,        ras, non_na)

## Write GeoTIFFs
tif_dissim   <- file.path(out_dir, paste0(prefix, "_temporal_dissimilarity.tif"))
tif_distance <- file.path(out_dir, paste0(prefix, "_temporal_distance.tif"))
tif_prob     <- file.path(out_dir, paste0(prefix, "_predicted_prob.tif"))

writeRaster(ras_dissim,   tif_dissim,   format = "GTiff", overwrite = TRUE)
writeRaster(ras_distance, tif_distance, format = "GTiff", overwrite = TRUE)
writeRaster(ras_prob,     tif_prob,     format = "GTiff", overwrite = TRUE)

cat(sprintf("  Saved: %s\n", basename(tif_dissim)))
cat(sprintf("  Saved: %s\n", basename(tif_distance)))
cat(sprintf("  Saved: %s\n", basename(tif_prob)))

## Save the raw results table and cell indices for later use
rds_file <- file.path(out_dir, paste0(prefix, "_prediction_results.rds"))
saveRDS(list(
  cell_indices    = non_na,
  lon             = pts$lon,
  lat             = pts$lat,
  year1           = year1,
  year2           = year2,
  temporal_distance = all_distance,
  linear_predictor  = all_linear_pred,
  predicted_prob    = all_prob,
  dissimilarity     = all_dissim,
  fit_metadata      = list(
    species_group  = fit$species_group,
    climate_window = fit$climate_window,
    w_ratio        = fit$w_ratio,
    intercept      = fit$intercept,
    D2             = fit$D2,
    nagelkerke_r2  = fit$nagelkerke_r2
  )
), file = rds_file)
cat(sprintf("  Saved: %s\n", basename(rds_file)))

# ---------------------------------------------------------------------------
# 6. Summary statistics
# ---------------------------------------------------------------------------
cat("\n--- Prediction summary ---\n")
valid <- !is.na(all_dissim)
cat(sprintf("  Valid pixels: %s / %s (%.1f%%)\n",
            format(sum(valid), big.mark = ","),
            format(n_pixels, big.mark = ","),
            100 * sum(valid) / n_pixels))
cat(sprintf("  Temporal distance:  mean = %.4f, sd = %.4f, range = [%.4f, %.4f]\n",
            mean(all_distance[valid]), sd(all_distance[valid]),
            min(all_distance[valid]), max(all_distance[valid])))
cat(sprintf("  Dissimilarity:      mean = %.4f, sd = %.4f, range = [%.4f, %.4f]\n",
            mean(all_dissim[valid]), sd(all_dissim[valid]),
            min(all_dissim[valid]), max(all_dissim[valid])))
cat(sprintf("  Predicted prob:     mean = %.4f, sd = %.4f, range = [%.4f, %.4f]\n",
            mean(all_prob[valid]), sd(all_prob[valid]),
            min(all_prob[valid]), max(all_prob[valid])))

# ---------------------------------------------------------------------------
# 7. Plot raster maps
# ---------------------------------------------------------------------------
cat("\n--- Generating raster maps ---\n")

## Colour palettes
pal_div   <- colorRampPalette(c("#2166AC", "#67A9CF", "#D1E5F0",
                                 "#FDDBC7", "#EF8A62", "#B2182B"))
pal_seq   <- colorRampPalette(c("#F7FCF5", "#C7E9C0", "#74C476",
                                 "#238B45", "#00441B"))

pdf_maps <- file.path(out_dir, paste0(prefix, "_raster_maps.pdf"))
pdf(pdf_maps, width = 12, height = 14)

layout(matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE),
       widths = c(1, 1), heights = c(1, 1))
par(mar = c(2, 2, 3, 4))

## Map 1: Temporal dissimilarity
plot(ras_dissim, col = pal_div(100), axes = TRUE,
     main = sprintf("Temporal Dissimilarity (%d -> %d)\n%s | %d yr window",
                    year1, year2, fit$species_group, fit$climate_window),
     cex.main = 1.1)

## Map 2: Temporal ecological distance
plot(ras_distance, col = pal_seq(100), axes = TRUE,
     main = sprintf("Temporal Ecological Distance (%d -> %d)\n%s | %d yr window",
                    year1, year2, fit$species_group, fit$climate_window),
     cex.main = 1.1)

## Map 3: Predicted mismatch probability
plot(ras_prob, col = pal_div(100), axes = TRUE,
     main = sprintf("Predicted Mismatch Probability (%d -> %d)\n%s | %d yr window",
                    year1, year2, fit$species_group, fit$climate_window),
     cex.main = 1.1)

## Map 4: Histogram of dissimilarity
par(mar = c(5, 5, 3, 2))
hist(all_dissim[valid], breaks = 50, col = "#67A9CF", border = "white",
     main = "Distribution of Temporal Dissimilarity",
     xlab = "Dissimilarity", ylab = "Pixel count", cex.main = 1.1)
abline(v = mean(all_dissim[valid]), col = "red", lwd = 2, lty = 2)
abline(v = median(all_dissim[valid]), col = "darkgreen", lwd = 2, lty = 2)
legend("topright",
       legend = c(sprintf("mean = %.4f", mean(all_dissim[valid])),
                  sprintf("median = %.4f", median(all_dissim[valid]))),
       col = c("red", "darkgreen"), lty = 2, lwd = 2, cex = 0.9)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_maps)))

## Also save individual high-res TIFFs of the maps
tif_map <- file.path(out_dir, paste0(prefix, "_dissimilarity_map.tif"))
tiff(tif_map, width = 10, height = 8, units = "in", res = 300, compression = "lzw")
plot(ras_dissim, col = pal_div(100), axes = TRUE,
     main = sprintf("Temporal Dissimilarity (%d -> %d)  |  %s  |  %d yr window",
                    year1, year2, fit$species_group, fit$climate_window))
dev.off()
cat(sprintf("  Saved: %s\n", basename(tif_map)))

cat(sprintf("\n=== Raster prediction complete (%.1f min) ===\n", total_time / 60))
