##############################################################################
##
## test_predict_temporal_timeseries_ibra.R
##
## Load a fitted STresiduals GDM, load IBRA regions shapefile, and for each
## region sample up to 1000 random non-NA pixels from the reference raster.
## Predict temporal biodiversity change for a time series of year-pairs
## (1950 vs 1951 … 1950 vs 2017) and generate a ribbon plot per region.
##
##############################################################################

cat("=== IBRA Region Temporal GDM Time-Series ===\n\n")

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
ibra_shp     <- file.path(project_root,
  "data/ibra51_reg/ibra51_regions.shp")
npy_src      <- "/Volumes/PortableSSD/CLIMATE/geonpy"
python_exe   <- file.path(project_root, ".venv/bin/python3")
pyper_script <- file.path(project_root, "src/shared/python/pyper.py")

## Time-series parameters
baseline_year  <- 1950L
target_years   <- 1951L:2017L
pixels_per_reg <- 1000L   # max pixels to sample per region

## Output
out_dir <- file.path(project_root, "src/reca_STresiduals/output")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Source dependencies
# ---------------------------------------------------------------------------
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_temporal.R"))

library(raster)
library(arrow)
library(sf)

# ---------------------------------------------------------------------------
# 2. Load fit object and print summary
# ---------------------------------------------------------------------------
cat("--- Loading fitted GDM ---\n")
if (!file.exists(fit_path)) stop(paste("Fit file not found:", fit_path))
load(fit_path)   # loads 'fit'
summarise_temporal_gdm(fit)

# ---------------------------------------------------------------------------
# 3. Load reference raster and IBRA shapefile
# ---------------------------------------------------------------------------
cat("\n--- Loading reference raster ---\n")
if (!file.exists(ref_raster)) stop(paste("Reference raster not found:", ref_raster))
ras <- raster(ref_raster)
cat(sprintf("  Reference raster: %d x %d (%.4f res)\n",
            ncol(ras), nrow(ras), xres(ras)))

cat("\n--- Loading IBRA regions shapefile ---\n")
if (!file.exists(ibra_shp)) stop(paste("IBRA shapefile not found:", ibra_shp))
ibra <- st_read(ibra_shp, quiet = TRUE)

## Fix any invalid geometries before dissolving
cat("  Repairing invalid geometries ...\n")
ibra <- st_make_valid(ibra)

## Dissolve multipart polygons into one polygon per REG_NAME
cat("  Dissolving multipart polygons by REG_NAME ...\n")
ibra_dissolved <- ibra %>%
  dplyr::group_by(REG_NAME) %>%
  dplyr::summarise(geometry = sf::st_union(geometry), .groups = "drop")

## Transform to same CRS as raster (longlat WGS84 likely already compatible)
ras_crs <- crs(ras)
ibra_dissolved <- st_transform(ibra_dissolved, st_crs(ras_crs))

region_names <- sort(ibra_dissolved$REG_NAME)
n_regions    <- length(region_names)
cat(sprintf("  IBRA regions: %d dissolved polygons\n", n_regions))

# ---------------------------------------------------------------------------
# 4. Sample pixels per region
# ---------------------------------------------------------------------------
cat("\n--- Sampling pixels per IBRA region ---\n")
set.seed(42)

## Convert raster to SpatialPointsDataFrame of all non-NA cells
cat("  Extracting all non-NA cells from reference raster ...\n")
all_cells <- which(!is.na(values(ras)))
all_xy    <- xyFromCell(ras, all_cells)
all_pts   <- st_as_sf(data.frame(lon = all_xy[, 1], lat = all_xy[, 2]),
                       coords = c("lon", "lat"), crs = st_crs(ras_crs))
cat(sprintf("  Total non-NA cells: %d\n", length(all_cells)))

## Spatial join: assign each cell to an IBRA region
cat("  Performing spatial join (cells → regions) ...\n")
join_t0 <- proc.time()
cell_region <- st_join(all_pts, ibra_dissolved["REG_NAME"], left = FALSE)
join_time <- (proc.time() - join_t0)["elapsed"]
cat(sprintf("  Spatial join complete: %d cells matched in %.1f s\n",
            nrow(cell_region), join_time))

## Add lon/lat back from coordinates
coords <- st_coordinates(cell_region)
cell_region$lon <- coords[, 1]
cell_region$lat <- coords[, 2]
cell_region <- st_drop_geometry(cell_region)

## Sample up to pixels_per_reg per region
region_samples <- list()
for (reg in region_names) {
  reg_pts <- cell_region[cell_region$REG_NAME == reg, , drop = FALSE]
  n_avail <- nrow(reg_pts)
  if (n_avail == 0) {
    cat(sprintf("  [SKIP] %s — no non-NA pixels\n", reg))
    next
  }
  n_samp <- min(n_avail, pixels_per_reg)
  idx <- sample(seq_len(n_avail), n_samp)
  region_samples[[reg]] <- reg_pts[idx, c("lon", "lat")]
  cat(sprintf("  %-40s : %d / %d pixels sampled\n", reg, n_samp, n_avail))
}

## Remove any empty regions
active_regions <- names(region_samples)
n_active       <- length(active_regions)
cat(sprintf("\n  Active regions with samples: %d / %d\n", n_active, n_regions))

# ---------------------------------------------------------------------------
# 5. Run time-series predictions per region
# ---------------------------------------------------------------------------
n_years <- length(target_years)
cat(sprintf("\n--- Running time-series predictions: %d baseline → {%d..%d} ---\n",
            baseline_year, min(target_years), max(target_years)))
cat(sprintf("    %d regions × up to %d pixels × %d years\n",
            n_active, pixels_per_reg, n_years))

## Storage: list of matrices [n_sites x n_years] per region
results <- list()

total_t0 <- proc.time()

for (ri in seq_along(active_regions)) {
  reg <- active_regions[ri]
  samp <- region_samples[[reg]]
  n_sites <- nrow(samp)

  cat(sprintf("\n=== Region %d/%d: %s (%d sites) ===\n",
              ri, n_active, reg, n_sites))

  mat_distance <- matrix(NA_real_, nrow = n_sites, ncol = n_years)
  mat_dissim   <- matrix(NA_real_, nrow = n_sites, ncol = n_years)
  mat_prob     <- matrix(NA_real_, nrow = n_sites, ncol = n_years)
  colnames(mat_distance) <- colnames(mat_dissim) <- colnames(mat_prob) <- target_years

  reg_t0 <- proc.time()

  for (yi in seq_along(target_years)) {
    yr <- target_years[yi]

    if (yi %% 10 == 1 || yi == n_years) {
      elapsed <- (proc.time() - reg_t0)["elapsed"]
      if (yi > 1) {
        rate <- (yi - 1) / elapsed
        eta  <- (n_years - yi + 1) / rate
        cat(sprintf("  [%s] Year %d (%d/%d) | %.0fs elapsed | ~%.1f min left\n",
                    format(Sys.time(), "%H:%M:%S"), yr, yi, n_years, elapsed, eta / 60))
      } else {
        cat(sprintf("  [%s] Year %d (%d/%d)\n",
                    format(Sys.time(), "%H:%M:%S"), yr, yi, n_years))
      }
      flush.console()
    }

    pts <- data.frame(
      lon   = samp$lon,
      lat   = samp$lat,
      year1 = baseline_year,
      year2 = yr
    )

    res <- tryCatch(
      predict_temporal_gdm(
        fit          = fit,
        points       = pts,
        npy_src      = npy_src,
        python_exe   = python_exe,
        pyper_script = pyper_script,
        verbose      = FALSE
      ),
      error = function(e) {
        warning(sprintf("[%s] Year %d failed: %s", reg, yr, conditionMessage(e)))
        NULL
      }
    )

    if (!is.null(res)) {
      mat_distance[, yi] <- res$temporal_distance
      mat_dissim[, yi]   <- res$dissimilarity
      mat_prob[, yi]     <- res$predicted_prob
    }
  }

  reg_time <- (proc.time() - reg_t0)["elapsed"]
  cat(sprintf("  Region '%s' complete in %.1f min\n", reg, reg_time / 60))

  results[[reg]] <- list(
    sites        = samp,
    mat_distance = mat_distance,
    mat_dissim   = mat_dissim,
    mat_prob     = mat_prob,
    n_sites      = n_sites
  )
}

total_time <- (proc.time() - total_t0)["elapsed"]
cat(sprintf("\n--- All regions complete: %.1f min total ---\n", total_time / 60))

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
rds_file <- file.path(out_dir, "ibra_timeseries_results.rds")
saveRDS(list(
  baseline_year  = baseline_year,
  target_years   = target_years,
  region_results = results,
  fit_metadata   = list(
    species_group  = fit$species_group,
    climate_window = fit$climate_window,
    w_ratio        = fit$w_ratio,
    intercept      = fit$intercept,
    D2             = fit$D2
  )
), file = rds_file)
cat(sprintf("  Saved results: %s\n", basename(rds_file)))

# ---------------------------------------------------------------------------
# 7. Per-region ribbon plots (multi-page PDF)
# ---------------------------------------------------------------------------
cat("\n--- Generating per-region ribbon plots ---\n")

pdf_ribbon <- file.path(out_dir, "ibra_timeseries_ribbon_per_region.pdf")
pdf(pdf_ribbon, width = 12, height = 7)

for (reg in active_regions) {
  r <- results[[reg]]
  if (is.null(r) || all(is.na(r$mat_dissim))) next

  q10 <- apply(r$mat_dissim, 2, quantile, 0.10, na.rm = TRUE)
  q25 <- apply(r$mat_dissim, 2, quantile, 0.25, na.rm = TRUE)
  q50 <- apply(r$mat_dissim, 2, quantile, 0.50, na.rm = TRUE)
  q75 <- apply(r$mat_dissim, 2, quantile, 0.75, na.rm = TRUE)
  q90 <- apply(r$mat_dissim, 2, quantile, 0.90, na.rm = TRUE)
  mean_d <- colMeans(r$mat_dissim, na.rm = TRUE)

  y_range <- range(c(q10, q90), na.rm = TRUE)
  if (all(is.finite(y_range)) && diff(y_range) > 0) {
    ## Add 5% padding
    pad <- diff(y_range) * 0.05
    y_range <- y_range + c(-pad, pad)
  } else {
    next
  }

  par(mar = c(5, 5, 4, 2))
  plot(NA, xlim = range(target_years), ylim = y_range,
       xlab = "Year", ylab = "Temporal Dissimilarity",
       main = sprintf("%s\nTemporal Dissimilarity (%d baseline) | %s | %d sites",
                      reg, baseline_year, fit$species_group, r$n_sites),
       cex.main = 1.0)

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
  lines(target_years, mean_d, col = "#B2182B", lwd = 2, lty = 2)

  legend("topleft",
         legend = c("Median", "Mean", "25th-75th pctile", "10th-90th pctile"),
         col = c("#2166AC", "#B2182B",
                 adjustcolor("#2166AC", alpha.f = 0.4),
                 adjustcolor("#2166AC", alpha.f = 0.2)),
         lwd = c(3, 2, 8, 8), lty = c(1, 2, 1, 1), cex = 0.85, bg = "white")
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_ribbon)))

# ---------------------------------------------------------------------------
# 8. Summary comparison: all regions on one plot
# ---------------------------------------------------------------------------
cat("\n--- Generating summary comparison plot ---\n")

## Compute per-region median trajectory and final-year median dissimilarity
region_medians <- data.frame(
  region       = character(),
  final_dissim = numeric(),
  stringsAsFactors = FALSE
)

median_trajectories <- list()

for (reg in active_regions) {
  r <- results[[reg]]
  if (is.null(r) || all(is.na(r$mat_dissim))) next
  meds <- apply(r$mat_dissim, 2, median, na.rm = TRUE)
  median_trajectories[[reg]] <- meds
  region_medians <- rbind(region_medians, data.frame(
    region       = reg,
    final_dissim = meds[length(meds)],
    stringsAsFactors = FALSE
  ))
}

## Order by final dissimilarity (highest change first)
region_medians <- region_medians[order(-region_medians$final_dissim), ]

## Colour palette: gradient from low change (blue) to high change (red)
n_plotted <- nrow(region_medians)
change_pal <- colorRampPalette(c("#2166AC", "#67A9CF", "#D1E5F0",
                                  "#FDDBC7", "#EF8A62", "#B2182B"))(n_plotted)
## Assign colours in order of change (highest = red)
reg_colours <- setNames(change_pal, region_medians$region)

## --- Plot A: All median trajectories overlaid ---
pdf_all <- file.path(out_dir, "ibra_timeseries_all_regions.pdf")
pdf(pdf_all, width = 14, height = 9)

par(mar = c(5, 5, 4, 2))
y_all <- range(unlist(lapply(median_trajectories, range, na.rm = TRUE)), na.rm = TRUE)
pad   <- diff(y_all) * 0.05
y_all <- y_all + c(-pad, pad)

plot(NA, xlim = range(target_years), ylim = y_all,
     xlab = "Year", ylab = "Median Temporal Dissimilarity",
     main = sprintf("IBRA Region Comparison — Median Temporal Dissimilarity (%d baseline)\n%s | %d regions",
                    baseline_year, fit$species_group, n_plotted),
     cex.main = 1.0)

for (reg in names(median_trajectories)) {
  lines(target_years, median_trajectories[[reg]],
        col = adjustcolor(reg_colours[reg], alpha.f = 0.7), lwd = 1.5)
}

## Highlight top 5 and bottom 5
top5 <- head(region_medians$region, 5)
bot5 <- tail(region_medians$region, 5)
for (reg in c(top5, bot5)) {
  lines(target_years, median_trajectories[[reg]],
        col = reg_colours[reg], lwd = 2.5)
}

legend("topleft",
       legend = c(paste("Highest:", top5), "", paste("Lowest:", bot5)),
       col    = c(reg_colours[top5], NA, reg_colours[bot5]),
       lwd    = c(rep(2.5, 5), NA, rep(2.5, 5)),
       lty    = c(rep(1, 5), NA, rep(1, 5)),
       cex    = 0.65, bg = "white", ncol = 1)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_all)))

## --- Plot B: Bar chart of final-year median dissimilarity by region ---
pdf_bar <- file.path(out_dir, "ibra_timeseries_final_dissim_barplot.pdf")
pdf(pdf_bar, width = 14, height = max(8, n_plotted * 0.22))

par(mar = c(5, 14, 4, 2))
bp <- barplot(rev(region_medians$final_dissim),
              horiz = TRUE, las = 1,
              names.arg = rev(region_medians$region),
              col = rev(reg_colours[region_medians$region]),
              border = NA,
              xlab = sprintf("Median Temporal Dissimilarity (%d → %d)",
                             baseline_year, max(target_years)),
              main = sprintf("IBRA Region Ranking — %s | %d yr climate window",
                             fit$species_group, fit$climate_window),
              cex.names = 0.55, cex.main = 1.0)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_bar)))

# ---------------------------------------------------------------------------
# 9. Multi-page small-multiple ribbon grid (6 per page)
# ---------------------------------------------------------------------------
cat("\n--- Generating small-multiple ribbon grid ---\n")

pdf_grid <- file.path(out_dir, "ibra_timeseries_ribbon_grid.pdf")
pdf(pdf_grid, width = 16, height = 11)

## Use the ranked order (highest change first)
ranked_regions <- region_medians$region
panels_per_page <- 6
n_pages <- ceiling(n_plotted / panels_per_page)

for (page in seq_len(n_pages)) {
  start_idx <- (page - 1) * panels_per_page + 1
  end_idx   <- min(page * panels_per_page, n_plotted)
  page_regs <- ranked_regions[start_idx:end_idx]

  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

  for (reg in page_regs) {
    r <- results[[reg]]
    if (is.null(r) || all(is.na(r$mat_dissim))) {
      plot.new()
      next
    }

    q10 <- apply(r$mat_dissim, 2, quantile, 0.10, na.rm = TRUE)
    q25 <- apply(r$mat_dissim, 2, quantile, 0.25, na.rm = TRUE)
    q50 <- apply(r$mat_dissim, 2, quantile, 0.50, na.rm = TRUE)
    q75 <- apply(r$mat_dissim, 2, quantile, 0.75, na.rm = TRUE)
    q90 <- apply(r$mat_dissim, 2, quantile, 0.90, na.rm = TRUE)

    y_range <- range(c(q10, q90), na.rm = TRUE)
    if (!all(is.finite(y_range)) || diff(y_range) == 0) {
      plot.new()
      next
    }
    pad <- diff(y_range) * 0.05
    y_range <- y_range + c(-pad, pad)

    plot(NA, xlim = range(target_years), ylim = y_range,
         xlab = "", ylab = "Dissimilarity",
         main = sprintf("%s (n=%d)", reg, r$n_sites),
         cex.main = 0.9)

    polygon(c(target_years, rev(target_years)),
            c(q90, rev(q10)),
            col = adjustcolor(reg_colours[reg], alpha.f = 0.15), border = NA)
    polygon(c(target_years, rev(target_years)),
            c(q75, rev(q25)),
            col = adjustcolor(reg_colours[reg], alpha.f = 0.30), border = NA)
    lines(target_years, q50, col = reg_colours[reg], lwd = 2)
  }

  ## Fill remaining panels on last page
  remaining <- panels_per_page - length(page_regs)
  if (remaining > 0) for (k in seq_len(remaining)) plot.new()

  mtext(sprintf("IBRA Temporal Dissimilarity — %s (%d baseline) — page %d/%d",
                fit$species_group, baseline_year, page, n_pages),
        outer = TRUE, cex = 0.9)
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_grid)))

cat(sprintf("\n=== IBRA time-series analysis complete (%.1f min) ===\n",
            total_time / 60))
