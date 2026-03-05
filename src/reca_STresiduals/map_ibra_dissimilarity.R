##############################################################################
##
## map_ibra_dissimilarity.R
##
## Aggregate a temporal dissimilarity raster by IBRA region (mean, sd, max)
## and generate choropleth maps.
##
##############################################################################

cat("=== IBRA Region Dissimilarity Maps ===\n\n")

# ---------------------------------------------------------------------------
# 0. Source config and derive paths
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))

## Prediction years (must match predict_temporal_raster.R settings)
year1 <- 1950L
year2 <- 2017L

## Dissimilarity TIF produced by predict_temporal_raster.R
modis_tag <- if (isTRUE(config$add_modis)) "_MODIS" else ""
dissim_prefix <- sprintf("%s_%dyr_%d_to_%d%s",
                         config$species_group, config$climate_window, year1, year2,
                         modis_tag)
dissim_tif <- file.path(config$output_dir,
                        paste0(dissim_prefix, "_temporal_dissimilarity.tif"))
ibra_shp   <- file.path(config$data_dir, "ibra51_reg", "ibra51_regions.shp")
out_dir    <- config$output_dir
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## Label used in plot titles
map_label <- sprintf("%s | %d -> %d | %d yr climate window",
                     config$species_group, year1, year2, config$climate_window)

# ---------------------------------------------------------------------------
# 1. Load packages
# ---------------------------------------------------------------------------
library(raster)
library(sf)
library(dplyr)

# ---------------------------------------------------------------------------
# 2. Load raster and shapefile
# ---------------------------------------------------------------------------
cat("--- Loading dissimilarity raster ---\n")
if (!file.exists(dissim_tif)) stop("Raster not found: ", dissim_tif)
ras <- raster(dissim_tif)
cat(sprintf("  %d x %d, res %.4f, range [%.4f, %.4f]\n",
            ncol(ras), nrow(ras), xres(ras),
            cellStats(ras, "min"), cellStats(ras, "max")))

cat("\n--- Loading IBRA shapefile ---\n")
if (!file.exists(ibra_shp)) stop("Shapefile not found: ", ibra_shp)
ibra <- st_read(ibra_shp, quiet = TRUE)

cat("  Repairing invalid geometries ...\n")
ibra <- st_make_valid(ibra)

cat("  Dissolving by REG_NAME ...\n")
ibra_dissolved <- ibra %>%
  group_by(REG_NAME) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

## Match CRS
ibra_dissolved <- st_transform(ibra_dissolved, st_crs(projection(ras)))
n_regions <- nrow(ibra_dissolved)
cat(sprintf("  %d dissolved IBRA regions\n", n_regions))

# ---------------------------------------------------------------------------
# 3. Extract raster values per region
# ---------------------------------------------------------------------------
cat("\n--- Extracting raster values per region ---\n")
t0 <- proc.time()

## Convert to Spatial for raster::extract
ibra_sp <- as(ibra_dissolved, "Spatial")
extr <- raster::extract(ras, ibra_sp)   # list of numeric vectors

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Extraction complete in %.1f s\n", elapsed))

# ---------------------------------------------------------------------------
# 4. Compute zonal statistics
# ---------------------------------------------------------------------------
cat("\n--- Computing zonal statistics ---\n")

stats <- data.frame(
  REG_NAME   = ibra_dissolved$REG_NAME,
  mean_dissim = NA_real_,
  sd_dissim   = NA_real_,
  max_dissim  = NA_real_,
  median_dissim = NA_real_,
  n_pixels    = NA_integer_,
  stringsAsFactors = FALSE
)

for (i in seq_len(n_regions)) {
  vals <- extr[[i]]
  vals <- vals[!is.na(vals)]
  stats$n_pixels[i] <- length(vals)
  if (length(vals) > 0) {
    stats$mean_dissim[i]   <- mean(vals)
    stats$sd_dissim[i]     <- sd(vals)
    stats$max_dissim[i]    <- max(vals)
    stats$median_dissim[i] <- median(vals)
  }
}

## Print top/bottom 5
stats_sorted <- stats[order(-stats$mean_dissim), ]
cat("\n  Top 5 regions by mean dissimilarity:\n")
print(head(stats_sorted[, c("REG_NAME", "mean_dissim", "sd_dissim", "max_dissim", "n_pixels")], 5),
      row.names = FALSE)
cat("\n  Bottom 5 regions by mean dissimilarity:\n")
print(tail(stats_sorted[, c("REG_NAME", "mean_dissim", "sd_dissim", "max_dissim", "n_pixels")], 5),
      row.names = FALSE)

# ---------------------------------------------------------------------------
# 5. Join statistics to polygons
# ---------------------------------------------------------------------------
ibra_stats <- ibra_dissolved %>%
  left_join(stats, by = "REG_NAME")

## Save as CSV and GeoPackage
csv_file <- file.path(out_dir, paste0("ibra_dissimilarity_stats", modis_tag, ".csv"))
write.csv(st_drop_geometry(ibra_stats), csv_file, row.names = FALSE)
cat(sprintf("\n  Saved: %s\n", basename(csv_file)))

gpkg_file <- file.path(out_dir, paste0("ibra_dissimilarity_stats", modis_tag, ".gpkg"))
st_write(ibra_stats, gpkg_file, delete_dsn = TRUE, quiet = TRUE)
cat(sprintf("  Saved: %s\n", basename(gpkg_file)))

# ---------------------------------------------------------------------------
# 6. Choropleth maps
# ---------------------------------------------------------------------------
cat("\n--- Generating maps ---\n")

## Diverging palette (cool -> warm)
make_pal <- function(n = 100) {
  colorRampPalette(c("#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7",
                      "#FDDBC7", "#EF8A62", "#B2182B"))(n)
}

## Shared plot function
plot_ibra_map <- function(ibra_sf, col_name, title, legend_title, pal, out_pdf) {
  vals <- ibra_sf[[col_name]]
  valid <- !is.na(vals)
  if (sum(valid) == 0) { cat("  [SKIP] No valid values for", col_name, "\n"); return() }

  brk <- seq(min(vals[valid]), max(vals[valid]), length.out = length(pal) + 1)
  cols <- pal[findInterval(vals, brk, all.inside = TRUE)]
  cols[!valid] <- "grey80"

  pdf(out_pdf, width = 14, height = 10)
  par(mar = c(2, 2, 4, 1))

  plot(st_geometry(ibra_sf), col = cols, border = "grey40", lwd = 0.3,
       main = title, cex.main = 1.1)

  ## Add region labels for extreme values (top/bottom 5)
  centroids <- st_coordinates(st_centroid(ibra_sf, of_largest_polygon = TRUE))
  ranked <- order(-vals)
  label_idx <- ranked[c(1:5, (length(ranked)-4):length(ranked))]
  label_idx <- label_idx[valid[label_idx]]

  if (length(label_idx) > 0) {
    text(centroids[label_idx, 1], centroids[label_idx, 2],
         labels = ibra_sf$REG_NAME[label_idx],
         cex = 0.45, font = 2, col = "black")
  }

  ## Colour bar
  fields_ok <- requireNamespace("fields", quietly = TRUE)
  if (fields_ok) {
    fields::image.plot(legend.only = TRUE,
                       zlim = range(vals[valid]),
                       col = pal, legend.mar = 4,
                       legend.lab = legend_title,
                       smallplot = c(0.82, 0.84, 0.15, 0.85))
  }

  dev.off()
  cat(sprintf("  Saved: %s\n", basename(out_pdf)))
}

pal100 <- make_pal(100)

## Map 1: Mean dissimilarity
plot_ibra_map(
  ibra_stats, "mean_dissim",
  title = paste0("Mean Temporal Dissimilarity by IBRA Region\n", map_label),
  legend_title = "Mean Dissimilarity",
  pal = pal100,
  out_pdf = file.path(out_dir, paste0("ibra_map_mean_dissim", modis_tag, ".pdf"))
)

## Map 2: SD dissimilarity
plot_ibra_map(
  ibra_stats, "sd_dissim",
  title = paste0("SD of Temporal Dissimilarity by IBRA Region\n", map_label),
  legend_title = "SD Dissimilarity",
  pal = pal100,
  out_pdf = file.path(out_dir, paste0("ibra_map_sd_dissim", modis_tag, ".pdf"))
)

## Map 3: Max dissimilarity
plot_ibra_map(
  ibra_stats, "max_dissim",
  title = paste0("Max Temporal Dissimilarity by IBRA Region\n", map_label),
  legend_title = "Max Dissimilarity",
  pal = pal100,
  out_pdf = file.path(out_dir, paste0("ibra_map_max_dissim", modis_tag, ".pdf"))
)

## Map 4: Median dissimilarity
plot_ibra_map(
  ibra_stats, "median_dissim",
  title = paste0("Median Temporal Dissimilarity by IBRA Region\n", map_label),
  legend_title = "Median Dissimilarity",
  pal = pal100,
  out_pdf = file.path(out_dir, paste0("ibra_map_median_dissim", modis_tag, ".pdf"))
)

# ---------------------------------------------------------------------------
# 7. Combined 4-panel map
# ---------------------------------------------------------------------------
cat("\n--- Generating combined 4-panel map ---\n")

pdf_combined <- file.path(out_dir, paste0("ibra_map_combined_4panel", modis_tag, ".pdf"))
pdf(pdf_combined, width = 18, height = 14)

par(mfrow = c(2, 2), mar = c(2, 2, 3, 5), oma = c(0, 0, 3, 0))

for (stat_col in c("mean_dissim", "sd_dissim", "max_dissim", "median_dissim")) {
  vals <- ibra_stats[[stat_col]]
  valid <- !is.na(vals)
  if (sum(valid) == 0) { plot.new(); next }

  brk <- seq(min(vals[valid]), max(vals[valid]), length.out = 101)
  cols <- pal100[findInterval(vals, brk, all.inside = TRUE)]
  cols[!valid] <- "grey80"

  nice_name <- switch(stat_col,
    mean_dissim   = "Mean",
    sd_dissim     = "Std Dev",
    max_dissim    = "Maximum",
    median_dissim = "Median"
  )

  plot(st_geometry(ibra_stats), col = cols, border = "grey40", lwd = 0.3,
       main = nice_name, cex.main = 1.2)

  fields_ok <- requireNamespace("fields", quietly = TRUE)
  if (fields_ok) {
    fields::image.plot(legend.only = TRUE,
                       zlim = range(vals[valid]),
                       col = pal100, legend.mar = 3.5,
                       smallplot = c(0.88, 0.90, 0.15, 0.85))
  }
}

mtext(paste0("Temporal Dissimilarity by IBRA Region -- ", map_label),
      outer = TRUE, cex = 1.1, font = 2)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_combined)))

cat("\n=== Done ===\n")
