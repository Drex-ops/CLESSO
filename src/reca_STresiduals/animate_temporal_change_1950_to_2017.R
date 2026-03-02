##############################################################################
##
## animate_temporal_change_1950_to_2017.R
##
## Create an animated GIF showing the spatial pattern of temporal
## biodiversity change accumulating from a 1950 baseline to each
## successive year (1951, 1952, ..., 2017).
##
## For each year pair 1950 → Y, the script:
##   1. Computes temporal dissimilarity at every pixel via the fitted
##      STresiduals GDM (same pipeline as predict_temporal_raster.R).
##   2. Renders a map frame (PNG) with a FIXED colour scale [0, global_max]
##      so all frames are directly comparable.
##   3. Assembles all frames into an animated GIF (magick package).
##
## Outputs:
##   - Individual frame PNGs in output/animation_frames/
##   - Animated GIF: output/<species>_temporal_change_1950_to_2017.gif
##   - Summary CSV with per-year statistics
##
## NOTE: This is computationally intensive (67 year-pairs × ~275k pixels
## × climate extraction).  Expect several hours of runtime.
## A progress log is printed after each year.
##
##############################################################################

cat("=== Animated Temporal Biodiversity Change: 1950 → 2017 ===\n\n")

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
npy_src      <- "/Volumes/PortableSSD/CLIMATE/geonpy"
python_exe   <- file.path(project_root, ".venv/bin/python3")
pyper_script <- file.path(project_root, "src/shared/python/pyper.py")

## ---- Year range ----
year1     <- 1950L
year2_seq <- seq(1955L, 2015L, by = 5L)
year2_seq <- c(year2_seq, 2017L)       # always include 2017 as the final year

## ---- Chunk size for pixel processing ----
chunk_size <- 10000L

## ---- Animation settings ----
gif_fps      <- 4            # frames per second
gif_loop     <- 0            # 0 = loop forever
frame_width  <- 1200         # pixels
frame_height <- 1000

## ---- Output ----
out_dir   <- file.path(project_root, "src/reca_STresiduals/output")
frame_dir <- file.path(out_dir, "animation_frames")
if (!dir.exists(frame_dir)) dir.create(frame_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1.  Source dependencies
# ---------------------------------------------------------------------------
source(file.path(project_root, "src/shared/R/utils.R"))
source(file.path(project_root, "src/shared/R/gdm_functions.R"))
source(file.path(project_root, "src/shared/R/gen_windows.R"))
source(file.path(project_root, "src/shared/R/predict_temporal.R"))

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
# 3.  Reference raster and pixel coordinates (computed once)
# ---------------------------------------------------------------------------
cat("\n--- Setting up reference raster ---\n")
ras <- raster(ref_raster)
vals <- getValues(ras)
non_na <- which(!is.na(vals))
coords <- xyFromCell(ras, non_na)
n_pixels <- nrow(coords)

cat(sprintf("  Raster: %d × %d cells | Non-NA pixels: %s\n",
            ncol(ras), nrow(ras), format(n_pixels, big.mark = ",")))

## Helper: build raster from values
make_raster <- function(values, template, cell_indices) {
  r <- raster(template)
  r[] <- NA
  r[cell_indices] <- values
  r
}

# ---------------------------------------------------------------------------
# 4.  PASS 1: Compute dissimilarity for all year-pairs
#     (Store results in memory / on disk; track global max for fixed scale)
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Pass 1: Computing temporal dissimilarity for %d year-pairs ---\n",
            length(year2_seq)))
cat(sprintf("  Baseline year: %d\n", year1))
cat(sprintf("  Target years:  %d → %d\n", min(year2_seq), max(year2_seq)))
cat(sprintf("  Chunk size:    %s pixels\n\n", format(chunk_size, big.mark = ",")))

## Storage for per-year results
dissim_list  <- vector("list", length(year2_seq))
stats_list   <- vector("list", length(year2_seq))
global_max   <- 0
t0_total     <- proc.time()

for (yi in seq_along(year2_seq)) {
  yr2 <- year2_seq[yi]
  t0_yr <- proc.time()

  cat(sprintf("[%s] Year %d → %d (%d/%d)...\n",
              format(Sys.time(), "%H:%M:%S"), year1, yr2, yi, length(year2_seq)))
  flush.console()

  ## Build points table
  pts <- data.frame(
    lon   = coords[, 1],
    lat   = coords[, 2],
    year1 = rep(year1, n_pixels),
    year2 = rep(yr2, n_pixels)
  )

  ## Chunked prediction
  chunks <- split(seq_len(n_pixels), ceiling(seq_len(n_pixels) / chunk_size))
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

    if (!is.null(chunk_result)) {
      all_dissim[rows] <- chunk_result$dissimilarity
    }
  }

  ## Store
  dissim_list[[yi]] <- all_dissim

  ## Track statistics
  valid <- !is.na(all_dissim)
  yr_mean   <- mean(all_dissim[valid])
  yr_median <- median(all_dissim[valid])
  yr_sd     <- sd(all_dissim[valid])
  yr_max    <- max(all_dissim[valid])
  yr_min    <- min(all_dissim[valid])

  stats_list[[yi]] <- data.frame(
    year2  = yr2,
    mean   = yr_mean,
    median = yr_median,
    sd     = yr_sd,
    min    = yr_min,
    max    = yr_max,
    n_valid = sum(valid),
    stringsAsFactors = FALSE
  )

  ## Update global max for fixed colour scale
  if (yr_max > global_max) global_max <- yr_max

  elapsed_yr <- (proc.time() - t0_yr)["elapsed"]
  elapsed_total <- (proc.time() - t0_total)["elapsed"]
  est_remaining <- elapsed_total / yi * (length(year2_seq) - yi)

  cat(sprintf("  [%s]   mean=%.4f, max=%.4f | %.0fs | est. %.1f min remaining\n",
              format(Sys.time(), "%H:%M:%S"), yr_mean, yr_max,
              elapsed_yr, est_remaining / 60))
  flush.console()
}

total_compute <- (proc.time() - t0_total)["elapsed"]
cat(sprintf("\n  Pass 1 complete: %d year-pairs in %.1f min\n",
            length(year2_seq), total_compute / 60))
cat(sprintf("  Global dissimilarity max: %.4f\n", global_max))

## Save summary stats
stats_df <- do.call(rbind, stats_list)
stats_csv <- file.path(out_dir, sprintf("%s_%dyr_%d_to_%d_animation_stats.csv",
                                         fit$species_group, fit$climate_window,
                                         year1, max(year2_seq)))
write.csv(stats_df, stats_csv, row.names = FALSE)
cat(sprintf("  Saved: %s\n", basename(stats_csv)))

# ---------------------------------------------------------------------------
# 5.  PASS 2: Render map frames with fixed colour scale
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Pass 2: Rendering %d map frames ---\n", length(year2_seq)))

## Colour palette: white → yellow → orange → red → dark red
dissim_pal <- colorRampPalette(c(
  "#F7F7F7", "#FEF0D9", "#FDD49E", "#FDBB84",
  "#FC8D59", "#EF6548", "#D7301F", "#B30000", "#7F0000"
))(256)

## Pad global max by 5% for headroom
zlim_max <- global_max * 1.05

for (yi in seq_along(year2_seq)) {
  yr2 <- year2_seq[yi]

  ## Build raster
  ras_frame <- make_raster(dissim_list[[yi]], ras, non_na)

  ## Render PNG frame
  frame_file <- file.path(frame_dir, sprintf("frame_%04d_%d_to_%d.png",
                                              yi, year1, yr2))
  png(frame_file, width = frame_width, height = frame_height, res = 150)

  par(mar = c(2, 2, 3.5, 5), bg = "white")

  plot(ras_frame, col = dissim_pal, zlim = c(0, zlim_max),
       axes = TRUE, box = TRUE,
       main = sprintf("Temporal Biodiversity Change: %d → %d\n%s | %d yr climate window",
                      year1, yr2, fit$species_group, fit$climate_window),
       cex.main = 1.1,
       legend.args = list(text = "Dissimilarity", side = 4, line = 2.5, cex = 0.9))

  ## Year label overlay (large, bottom-right)
  usr <- par("usr")
  text(usr[2] - (usr[2] - usr[1]) * 0.05,
       usr[3] + (usr[4] - usr[3]) * 0.08,
       labels = yr2, cex = 2.5, font = 2, adj = c(1, 0), col = "#333333")

  ## Mean dissimilarity annotation
  valid <- !is.na(dissim_list[[yi]])
  yr_mean <- mean(dissim_list[[yi]][valid])
  text(usr[1] + (usr[2] - usr[1]) * 0.02,
       usr[3] + (usr[4] - usr[3]) * 0.08,
       labels = sprintf("mean = %.4f", yr_mean),
       cex = 0.9, adj = c(0, 0), col = "#555555")

  dev.off()

  if (yi %% 10 == 0 || yi == length(year2_seq))
    cat(sprintf("  Rendered frame %d/%d (%d → %d)\n", yi, length(year2_seq), year1, yr2))
}

cat(sprintf("  All %d frames saved to: %s\n", length(year2_seq), basename(frame_dir)))

# ---------------------------------------------------------------------------
# 6.  Assemble animated GIF
# ---------------------------------------------------------------------------
cat("\n--- Assembling animated GIF ---\n")

if (!requireNamespace("magick", quietly = TRUE)) {
  cat("  NOTE: 'magick' package not installed. Installing...\n")
  install.packages("magick")
}
library(magick)

## Read frames in order
frame_files <- sort(list.files(frame_dir, pattern = "^frame_.*\\.png$", full.names = TRUE))
cat(sprintf("  Reading %d frames...\n", length(frame_files)))

frames <- image_read(frame_files)

## --- Compression & optimisation ---
## 1. Quantise to 256 colours (GIF's native maximum palette depth).
##    This is the single biggest file-size reducer — from 24-bit RGB
##    down to an 8-bit indexed palette per frame.
cat("  Quantising to 256 colours...\n")
frames <- image_quantize(frames, max = 256, dither = TRUE)

## 2. Animate with frame-level optimisation.
##    optimize = TRUE enables GIF inter-frame diffing so only changed
##    pixels are stored in each frame (huge savings when background is static).
delay <- as.integer(round(100 / gif_fps))
cat(sprintf("  Animating: delay = %d cs (%.0f fps), optimize = TRUE...\n", delay, gif_fps))
frames <- image_animate(frames, delay = delay, loop = gif_loop, optimize = TRUE)

## Write GIF
gif_file <- file.path(out_dir, sprintf("%s_%dyr_%d_to_%d_temporal_change.gif",
                                        fit$species_group, fit$climate_window,
                                        year1, max(year2_seq)))
image_write(frames, path = gif_file)
cat(sprintf("  Saved: %s\n", basename(gif_file)))

## Report file size
fsize <- file.size(gif_file)
cat(sprintf("  GIF size: %.1f MB\n", fsize / 1024^2))

# ---------------------------------------------------------------------------
# 7.  Bonus: Summary timeline plot (mean dissimilarity over time)
# ---------------------------------------------------------------------------
cat("\n--- Generating summary timeline plot ---\n")

pdf_timeline <- file.path(out_dir, sprintf("%s_%dyr_%d_to_%d_temporal_timeline.pdf",
                                            fit$species_group, fit$climate_window,
                                            year1, max(year2_seq)))
pdf(pdf_timeline, width = 12, height = 6)

par(mar = c(5, 5, 4, 5))

## Mean dissimilarity over time
plot(stats_df$year2, stats_df$mean, type = "l", lwd = 2, col = "#D7301F",
     xlab = "Target Year", ylab = "Mean Temporal Dissimilarity",
     main = sprintf("Mean Temporal Biodiversity Change from %d\n%s | %d yr climate window",
                    year1, fit$species_group, fit$climate_window),
     xlim = c(year1, max(year2_seq)),
     ylim = c(0, max(stats_df$mean + stats_df$sd) * 1.1))

## ± 1 SD band
polygon(c(stats_df$year2, rev(stats_df$year2)),
        c(stats_df$mean + stats_df$sd, rev(pmax(stats_df$mean - stats_df$sd, 0))),
        col = adjustcolor("#D7301F", alpha.f = 0.15), border = NA)

## Median line
lines(stats_df$year2, stats_df$median, lty = 2, lwd = 1.5, col = "#FC8D59")

## Max line on secondary axis
par(new = TRUE)
plot(stats_df$year2, stats_df$max, type = "l", lwd = 1.5, col = "#7F0000",
     axes = FALSE, xlab = "", ylab = "",
     xlim = c(year1, max(year2_seq)),
     ylim = c(0, max(stats_df$max) * 1.1))
axis(4, col = "#7F0000", col.axis = "#7F0000")
mtext("Max Dissimilarity", side = 4, line = 3, col = "#7F0000")

legend("topleft",
       legend = c("Mean ± SD", "Median", "Max (right axis)"),
       col = c("#D7301F", "#FC8D59", "#7F0000"),
       lty = c(1, 2, 1), lwd = c(2, 1.5, 1.5),
       fill = c(adjustcolor("#D7301F", alpha.f = 0.15), NA, NA),
       border = c("black", NA, NA),
       cex = 0.85, bg = "white")

dev.off()
cat(sprintf("  Saved: %s\n", basename(pdf_timeline)))

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat(sprintf("\n=== Animation pipeline complete ===\n"))
cat(sprintf("  Frames:   %s\n", frame_dir))
cat(sprintf("  GIF:      %s\n", basename(gif_file)))
cat(sprintf("  Timeline: %s\n", basename(pdf_timeline)))
cat(sprintf("  Stats:    %s\n", basename(stats_csv)))
cat(sprintf("  Total compute time: %.1f min\n", total_compute / 60))
