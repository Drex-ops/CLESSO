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
# 0. Source config and set parameters
# ---------------------------------------------------------------------------
this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
source(file.path(this_dir, "config.R"))

project_root <- config$project_root
fit_path     <- config$fit_path
ref_raster   <- config$reference_raster
ibra_shp     <- file.path(config$data_dir, "ibra51_reg", "ibra51_regions.shp")
npy_src      <- config$npy_src
python_exe   <- config$python_exe
pyper_script <- config$pyper_script
out_dir      <- config$run_output_dir

## Time-series parameters
## When MODIS or Condition data is used, restrict prediction to the
## year range covered by those layers.  Otherwise use the full climate range.
climate_end <- config$geonpy_end_year      # e.g. 2017

if (isTRUE(config$add_modis)) {
  baseline_year <- config$modis_start_year
  last_year     <- min(config$modis_end_year, climate_end)
  target_years  <- (baseline_year + 1L):last_year
  cat(sprintf("  MODIS enabled -> time-series %d–%d\n", baseline_year, last_year))
} else if (isTRUE(config$add_condition)) {
  baseline_year <- config$condition_start_year
  last_year     <- min(config$condition_end_year, climate_end)
  target_years  <- (baseline_year + 1L):last_year
  cat(sprintf("  Condition enabled -> time-series %d–%d\n", baseline_year, last_year))
} else {
  baseline_year <- 1950L
  target_years  <- 1951L:climate_end
}

pixels_per_reg <- 1000L   # max pixels to sample per region

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
cat("  Performing spatial join (cells -> regions) ...\n")
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
    cat(sprintf("  [SKIP] %s -- no non-NA pixels\n", reg))
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
#
# OPTIMISATION NOTES
#   The original version called predict_temporal_gdm() once per year per
#   region (67 years × 5 env groups = 335 Python subprocess calls per
#   region, ~28 000 total).  This version:
#
#     A) Year-batches:  all target years for a region are packed into a
#        single mega-table (n_sites × n_years rows) and passed to
#        predict_temporal_gdm in ONE call.  This cuts Python subprocess
#        overhead by ~67×.
#
#     B) Region parallelism:  regions are processed across CPU cores
#        with parallel::mclapply (Unix) or sequential fallback (Windows).
#        Set RECA_IBRA_CORES env var to control (default: cores_to_use
#        from config, capped at 8 to avoid disk I/O saturation).
# ---------------------------------------------------------------------------
n_years <- length(target_years)
cat(sprintf("\n--- Running time-series predictions: %d baseline -> {%d..%d} ---\n",
            baseline_year, min(target_years), max(target_years)))
cat(sprintf("    %d regions × up to %d pixels × %d years\n",
            n_active, pixels_per_reg, n_years))

## Number of cores for region-level parallelism
n_cores <- as.integer(Sys.getenv("RECA_IBRA_CORES",
               unset = as.character(min(config$cores_to_use, 8L))))
use_parallel <- (n_cores > 1L) && (.Platform$OS.type == "unix")
if (use_parallel) {
  cat(sprintf("    Parallel mode: %d cores (mclapply)\n", n_cores))
} else {
  cat("    Sequential mode\n")
}

## ---- Worker function: process one region (batched over all years) ----
process_region <- function(reg) {
  samp    <- region_samples[[reg]]
  n_sites <- nrow(samp)

  ## Build mega-table: replicate every site for every target year
  pts_all <- data.frame(
    lon   = rep(samp$lon,   times = n_years),
    lat   = rep(samp$lat,   times = n_years),
    year1 = baseline_year,
    year2 = rep(target_years, each = n_sites)
  )

  res <- tryCatch(
    predict_temporal_gdm(
      fit          = fit,
      points       = pts_all,
      npy_src      = npy_src,
      python_exe   = python_exe,
      pyper_script = pyper_script,
      verbose      = FALSE
    ),
    error = function(e) {
      warning(sprintf("[%s] prediction failed: %s", reg, conditionMessage(e)))
      NULL
    }
  )

  if (is.null(res)) return(NULL)

  ## Reshape from long vector back to [n_sites × n_years] matrices
  mat_distance <- matrix(res$temporal_distance, nrow = n_sites, ncol = n_years)
  mat_dissim   <- matrix(res$dissimilarity,      nrow = n_sites, ncol = n_years)
  mat_prob     <- matrix(res$predicted_prob,      nrow = n_sites, ncol = n_years)
  mat_sim      <- 1 - mat_dissim
  colnames(mat_distance) <- colnames(mat_dissim) <- colnames(mat_prob) <- target_years
  colnames(mat_sim) <- target_years

  list(
    sites        = samp,
    mat_distance = mat_distance,
    mat_dissim   = mat_dissim,
    mat_sim      = mat_sim,
    mat_prob     = mat_prob,
    n_sites      = n_sites
  )
}

## ---- Run across all regions ----
total_t0 <- proc.time()

if (use_parallel) {
  results_list <- parallel::mclapply(
    active_regions, process_region, mc.cores = n_cores
  )
} else {
  results_list <- vector("list", n_active)
  for (ri in seq_along(active_regions)) {
    reg <- active_regions[ri]
    n_sites <- nrow(region_samples[[reg]])
    cat(sprintf("\n=== Region %d/%d: %s (%d sites × %d years = %d pairs) ===\n",
                ri, n_active, reg, n_sites, n_years, n_sites * n_years))
    reg_t0 <- proc.time()
    results_list[[ri]] <- process_region(reg)
    reg_time <- (proc.time() - reg_t0)["elapsed"]
    cat(sprintf("  Region '%s' complete in %.1f min\n", reg, reg_time / 60))
  }
}

## Convert to named list
results <- setNames(results_list, active_regions)

total_time <- (proc.time() - total_t0)["elapsed"]
cat(sprintf("\n--- All regions complete: %.1f min total ---\n", total_time / 60))

## Overall NA summary across all regions
cat("\n--- NA Summary ---\n")
for (reg in active_regions) {
  r <- results[[reg]]
  if (is.null(r)) next
  total_c <- prod(dim(r$mat_sim))
  na_c    <- sum(is.na(r$mat_sim))
  cat(sprintf("  %-40s : %5.1f%% NA  (%d / %d)\n",
              reg, 100 * na_c / max(total_c, 1), na_c, total_c))
}

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
## MODIS suffix for output filenames (derived from fit metadata)
modis_tag <- if (isTRUE(fit$add_modis)) "_MODIS" else ""

rds_file <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_ibra_timeseries_results.rds"))
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

pdf_ribbon <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_ibra_timeseries_ribbon_per_region.pdf"))
pdf(pdf_ribbon, width = 12, height = 7)

for (reg in active_regions) {
  r <- results[[reg]]
  if (is.null(r) || all(is.na(r$mat_sim))) next

  ## Count valid (non-NA) values per year; skip years with < 3
  n_valid <- colSums(!is.na(r$mat_sim))
  ok_yrs  <- n_valid >= 3
  if (sum(ok_yrs) < 2) next    # need at least 2 plottable years

  q10 <- apply(r$mat_sim, 2, quantile, 0.10, na.rm = TRUE)
  q25 <- apply(r$mat_sim, 2, quantile, 0.25, na.rm = TRUE)
  q50 <- apply(r$mat_sim, 2, quantile, 0.50, na.rm = TRUE)
  q75 <- apply(r$mat_sim, 2, quantile, 0.75, na.rm = TRUE)
  q90 <- apply(r$mat_sim, 2, quantile, 0.90, na.rm = TRUE)
  mean_d <- colMeans(r$mat_sim, na.rm = TRUE)

  ## Replace NaN/Inf from all-NA columns with NA for safe plotting
  q10[!is.finite(q10)]       <- NA
  q25[!is.finite(q25)]       <- NA
  q50[!is.finite(q50)]       <- NA
  q75[!is.finite(q75)]       <- NA
  q90[!is.finite(q90)]       <- NA
  mean_d[!is.finite(mean_d)] <- NA

  y_range <- range(c(q10[ok_yrs], q90[ok_yrs]), na.rm = TRUE)
  if (all(is.finite(y_range)) && diff(y_range) > 0) {
    ## Add 5% padding
    pad <- diff(y_range) * 0.05
    y_range <- y_range + c(-pad, pad)
  } else {
    next
  }

  par(mar = c(5, 5, 4, 2))
  plot(NA, xlim = range(target_years), ylim = y_range,
       xlab = "Year", ylab = "Temporal Similarity",
       main = sprintf("%s\nTemporal Similarity (%d baseline) | %s | %d sites",
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

## Compute per-region median trajectory and final-year median similarity
region_medians <- data.frame(
  region    = character(),
  final_sim = numeric(),
  stringsAsFactors = FALSE
)

median_trajectories <- list()

for (reg in active_regions) {
  r <- results[[reg]]
  if (is.null(r) || all(is.na(r$mat_sim))) next
  meds <- apply(r$mat_sim, 2, median, na.rm = TRUE)
  median_trajectories[[reg]] <- meds
  region_medians <- rbind(region_medians, data.frame(
    region    = reg,
    final_sim = meds[length(meds)],
    stringsAsFactors = FALSE
  ))
}

## Order by final similarity (lowest similarity = most change, first)
region_medians <- region_medians[order(region_medians$final_sim), ]

## Colour palette: gradient from low similarity (red) to high similarity (blue)
n_plotted <- nrow(region_medians)
change_pal <- colorRampPalette(c("#B2182B", "#EF8A62", "#FDDBC7",
                                  "#D1E5F0", "#67A9CF", "#2166AC"))(n_plotted)
## Assign colours in order of change (highest = red)
reg_colours <- setNames(change_pal, region_medians$region)

## --- Plot A: All median trajectories overlaid ---
pdf_all <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_ibra_timeseries_all_regions.pdf"))
pdf(pdf_all, width = 14, height = 9)

par(mar = c(5, 5, 4, 2))
y_all <- range(unlist(lapply(median_trajectories, range, na.rm = TRUE)), na.rm = TRUE)
pad   <- diff(y_all) * 0.05
y_all <- y_all + c(-pad, pad)

plot(NA, xlim = range(target_years), ylim = y_all,
     xlab = "Year", ylab = "Median Temporal Similarity",
     main = sprintf("IBRA Region Comparison -- Median Temporal Similarity (%d baseline)\n%s | %d regions",
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
       legend = c(paste("Lowest:", top5), "", paste("Highest:", bot5)),
       col    = c(reg_colours[top5], NA, reg_colours[bot5]),
       lwd    = c(rep(2.5, 5), NA, rep(2.5, 5)),
       lty    = c(rep(1, 5), NA, rep(1, 5)),
       cex    = 0.65, bg = "white", ncol = 1)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_all)))

## --- Plot B: Bar chart of final-year median similarity by region ---
pdf_bar <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_ibra_timeseries_final_sim_barplot.pdf"))
pdf(pdf_bar, width = 14, height = max(8, n_plotted * 0.22))

par(mar = c(5, 14, 4, 2))
bp <- barplot(rev(region_medians$final_sim),
              horiz = TRUE, las = 1,
              names.arg = rev(region_medians$region),
              col = rev(reg_colours[region_medians$region]),
              border = NA,
              xlab = sprintf("Median Temporal Similarity (%d -> %d)",
                             baseline_year, max(target_years)),
              main = sprintf("IBRA Region Ranking -- %s | %d yr climate window",
                             fit$species_group, fit$climate_window),
              cex.names = 0.55, cex.main = 1.0)

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_bar)))

# ---------------------------------------------------------------------------
# 9. Multi-page small-multiple ribbon grid (6 per page)
# ---------------------------------------------------------------------------
cat("\n--- Generating small-multiple ribbon grid ---\n")

pdf_grid <- file.path(out_dir, paste0(fit$species_group, modis_tag, "_ibra_timeseries_ribbon_grid.pdf"))
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
    if (is.null(r) || all(is.na(r$mat_sim))) {
      plot.new()
      next
    }

    q10 <- apply(r$mat_sim, 2, quantile, 0.10, na.rm = TRUE)
    q25 <- apply(r$mat_sim, 2, quantile, 0.25, na.rm = TRUE)
    q50 <- apply(r$mat_sim, 2, quantile, 0.50, na.rm = TRUE)
    q75 <- apply(r$mat_sim, 2, quantile, 0.75, na.rm = TRUE)
    q90 <- apply(r$mat_sim, 2, quantile, 0.90, na.rm = TRUE)

    ## Sanitise non-finite values from all-NA columns
    q10[!is.finite(q10)] <- NA
    q25[!is.finite(q25)] <- NA
    q50[!is.finite(q50)] <- NA
    q75[!is.finite(q75)] <- NA
    q90[!is.finite(q90)] <- NA

    ok_yrs  <- colSums(!is.na(r$mat_sim)) >= 3
    y_range <- range(c(q10[ok_yrs], q90[ok_yrs]), na.rm = TRUE)
    if (!all(is.finite(y_range)) || diff(y_range) == 0) {
      plot.new()
      next
    }
    pad <- diff(y_range) * 0.05
    y_range <- y_range + c(-pad, pad)

    plot(NA, xlim = range(target_years), ylim = y_range,
         xlab = "", ylab = "Similarity",
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

  mtext(sprintf("IBRA Temporal Similarity -- %s (%d baseline) -- page %d/%d",
                fit$species_group, baseline_year, page, n_pages),
        outer = TRUE, cex = 0.9)
}

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_grid)))

cat(sprintf("\n=== IBRA time-series analysis complete (%.1f min) ===\n",
            total_time / 60))
