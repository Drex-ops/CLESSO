##############################################################################
##
## plot_radar_map_hex.R
##
## Places mini radar plots on a hexagonal grid over Australia.  Each hex
## cell aggregates sites that fall within it, computing mean temporal
## change at the 2017 epoch for each biological group.
##
## Produces 4 PDFs:
##   1) All of Australia – all groups
##   2) All of Australia – sans AVES & HYM
##   3) Eastern states   – all groups
##   4) Eastern states   – sans AVES & HYM
##
## Usage:
##   Rscript plot_radar_map_hex.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Hex-Grid Radar Map: Temporal Change at 2017 Epoch\n")
cat("###########################################################################\n\n")

# ---------------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------------
PLOT_DISSIMILARITY <- TRUE

metric_name <- if (PLOT_DISSIMILARITY) "Change in Community Composition" else "Similarity in Community Composition"
metric_tag  <- if (PLOT_DISSIMILARITY) "dissim" else "sim"

## Hex cell size in degrees (approximate).  Smaller = more cells, finer detail.
HEX_CELLSIZE <- 3.0     # ~550 km across at mid-latitudes

## Radar size as fraction of plot width for each mini radar
RADAR_SIZE <- 0.075

## Size of the coloured dot at the tip of each radar axis
AXIS_DOT_CEX <- 0.8

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
this_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("--file=", "", file_arg[1])))
  } else {
    getwd()
  }
})
output_dir <- file.path(this_dir, "output")
viz_dir    <- file.path(output_dir, "model_output_viz")
data_dir   <- file.path(dirname(dirname(this_dir)), "data")
if (!dir.exists(viz_dir)) dir.create(viz_dir, recursive = TRUE)

## Required packages
for (pkg in c("sf", "dplyr", "fmsb")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
library(sf)
library(dplyr)
library(fmsb)

# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------
all_groups <- c("AMP", "AVES", "HYM", "MAM", "REP", "VAS")

group_colours <- c(
  AMP  = "#E41A1C",
  AVES = "#377EB8",
  HYM  = "#FF7F00",
  MAM  = "#4DAF4A",
  REP  = "#984EA3",
  VAS  = "#A65628"
)

group_full_names <- c(
  AMP  = "Amphibians",
  AVES = "Birds",
  HYM  = "Hymenoptera",
  MAM  = "Mammals",
  REP  = "Reptiles",
  VAS  = "Vascular plants"
)

# ---------------------------------------------------------------------------
# Discover run folders and load IBRA timeseries data
# ---------------------------------------------------------------------------
find_run_folder <- function(group, variant = "base") {
  pattern <- switch(variant,
    base  = sprintf("^%s_\\d{8}T\\d{6}$", group),
    COND  = sprintf("^%s_COND_\\d{8}T\\d{6}$", group),
    MODIS = sprintf("^%s_MODIS_\\d{8}T\\d{6}$", group)
  )
  dirs <- list.dirs(output_dir, full.names = FALSE, recursive = FALSE)
  matches <- dirs[grepl(pattern, dirs)]
  if (length(matches) == 0) return(NULL)
  file.path(output_dir, sort(matches, decreasing = TRUE)[1])
}

## Load all IBRA results — we will extract individual sites from them
ibra_data <- list()
for (grp in all_groups) {
  base_dir <- find_run_folder(grp, "base")
  cond_dir <- find_run_folder(grp, "COND")
  base_f <- if (!is.null(base_dir)) file.path(base_dir, paste0(grp, "_ibra_timeseries_results.rds")) else NULL
  cond_f <- if (!is.null(cond_dir)) file.path(cond_dir, paste0(grp, "_ibra_timeseries_results.rds")) else NULL
  ibra_data[[grp]] <- list(
    base = if (!is.null(base_f) && file.exists(base_f)) readRDS(base_f) else NULL,
    cond = if (!is.null(cond_f) && file.exists(cond_f)) readRDS(cond_f) else NULL
  )
  cat(sprintf("  %-5s  base=%s  cond=%s\n", grp,
              if (!is.null(ibra_data[[grp]]$base)) "YES" else "---",
              if (!is.null(ibra_data[[grp]]$cond)) "YES" else "---"))
}
cat("\n")

# ---------------------------------------------------------------------------
# Extract all site-level data: lon, lat, 2017 value — per group × model
# ---------------------------------------------------------------------------
cat("--- Extracting site-level 2017 values from IBRA results ---\n")

## Returns a data.frame with columns: lon, lat, value
extract_site_values <- function(group, model_type = "base") {
  dat <- ibra_data[[group]][[model_type]]
  if (is.null(dat)) return(data.frame(lon = numeric(), lat = numeric(), value = numeric()))

  all_years <- c(dat$baseline_year, dat$target_years)
  idx_2017  <- which(all_years == 2017)
  if (length(idx_2017) == 0) idx_2017 <- length(all_years)
  col_idx <- idx_2017 - 1L  # column in mat_sim (excludes baseline)

  rows <- list()
  for (reg in names(dat$region_results)) {
    rr <- dat$region_results[[reg]]
    if (is.null(rr) || is.null(rr$sites) || all(is.na(rr$mat_sim))) next
    if (col_idx < 1 || col_idx > ncol(rr$mat_sim)) next

    sim_vals <- rr$mat_sim[, col_idx]
    vals <- if (PLOT_DISSIMILARITY) 1.0 - sim_vals else sim_vals

    rows[[length(rows) + 1L]] <- data.frame(
      lon   = rr$sites$lon,
      lat   = rr$sites$lat,
      value = vals,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0) return(data.frame(lon = numeric(), lat = numeric(), value = numeric()))
  do.call(rbind, rows)
}

site_data <- list()
for (grp in all_groups) {
  site_data[[grp]] <- list(
    base = extract_site_values(grp, "base"),
    cond = extract_site_values(grp, "cond")
  )
  cat(sprintf("  %-5s  base: %d sites,  cond: %d sites\n",
              grp, nrow(site_data[[grp]]$base), nrow(site_data[[grp]]$cond)))
}
cat("\n")

# ---------------------------------------------------------------------------
# Load IBRA shapefile for the coastline outline
# ---------------------------------------------------------------------------
ibra_shp <- file.path(data_dir, "ibra51_reg", "ibra51_regions.shp")
cat("--- Loading IBRA shapefile (for outline) ---\n")
if (!file.exists(ibra_shp)) stop("Shapefile not found: ", ibra_shp)
ibra_sf <- st_read(ibra_shp, quiet = TRUE)
ibra_sf <- st_make_valid(ibra_sf)

## Ensure CRS is WGS84 (lon/lat) — sites are in EPSG:4326
ibra_sf <- st_transform(ibra_sf, 4326)

## Dissolve into a single Australia outline
aus_outline <- st_union(ibra_sf)
cat("  Australia outline created\n\n")

# ---------------------------------------------------------------------------
# Build the hex grid
# ---------------------------------------------------------------------------
cat(sprintf("--- Building hex grid (cell size = %.1f degrees) ---\n", HEX_CELLSIZE))

## Create a hex grid covering the bounding box of Australia
hex_grid <- st_make_grid(aus_outline, cellsize = HEX_CELLSIZE, square = FALSE)
hex_grid <- st_sf(hex_id = seq_along(hex_grid), geometry = hex_grid)

## Clip to the Australia outline so we only keep hexes with land
hex_grid <- hex_grid[st_intersects(hex_grid, aus_outline, sparse = FALSE)[, 1], ]
hex_grid$hex_id <- seq_len(nrow(hex_grid))

## Compute centroids
hex_centroids <- st_centroid(hex_grid)
hex_coords    <- st_coordinates(hex_centroids)
hex_grid$cx   <- hex_coords[, 1]
hex_grid$cy   <- hex_coords[, 2]

cat(sprintf("  %d hex cells cover Australia\n\n", nrow(hex_grid)))

# ---------------------------------------------------------------------------
# Assign sites to hex cells and compute mean value per cell × group × model
# ---------------------------------------------------------------------------
cat("--- Assigning sites to hex cells ---\n")

## For each group × model_type, produce a data.frame:
##   hex_id, mean_value
assign_to_hex <- function(sites_df) {
  if (nrow(sites_df) == 0) return(data.frame(hex_id = integer(), mean_value = numeric()))

  pts <- st_as_sf(sites_df, coords = c("lon", "lat"), crs = 4326)
  ## Spatial join: which hex does each site fall in?
  joined <- st_join(pts, hex_grid[, c("hex_id", "geometry")], join = st_within)

  ## Aggregate: mean value per hex
  agg <- joined %>%
    st_drop_geometry() %>%
    filter(!is.na(hex_id)) %>%
    group_by(hex_id) %>%
    summarise(mean_value = mean(value, na.rm = TRUE),
              n_sites    = n(),
              .groups    = "drop")
  as.data.frame(agg)
}

hex_vals <- list()
for (grp in all_groups) {
  hex_vals[[grp]] <- list(
    base = assign_to_hex(site_data[[grp]]$base),
    cond = assign_to_hex(site_data[[grp]]$cond)
  )
  cat(sprintf("  %-5s  base: %d hexes,  cond: %d hexes\n",
              grp, nrow(hex_vals[[grp]]$base), nrow(hex_vals[[grp]]$cond)))
}
cat("\n")

# ---------------------------------------------------------------------------
# Identify which hex cells fall in eastern states
# ---------------------------------------------------------------------------
## Eastern states = lon >= 138 AND lat >= -44 (rough bounding box)
## A hex is "eastern" if its centroid is east of 138°E
eastern_hex_ids <- hex_grid$hex_id[hex_grid$cx >= 138]
cat(sprintf("  Eastern-state hex cells: %d / %d\n\n",
            length(eastern_hex_ids), nrow(hex_grid)))

# ---------------------------------------------------------------------------
# Pre-compute global y_max for consistent radar scales
# ---------------------------------------------------------------------------
compute_global_limits <- function(groups_to_plot, hex_ids) {
  all_v <- c()
  for (grp in groups_to_plot) {
    for (mt in c("base", "cond")) {
      hv <- hex_vals[[grp]][[mt]]
      vals <- hv$mean_value[hv$hex_id %in% hex_ids]
      all_v <- c(all_v, vals)
    }
  }
  all_v <- all_v[!is.na(all_v)]
  if (length(all_v) == 0) return(c(0, 1))
  if (PLOT_DISSIMILARITY) {
    c(0, max(0.01, ceiling(max(all_v) * 20) / 20 + 0.05))
  } else {
    c(max(0, floor(min(all_v) * 20) / 20 - 0.05), 1.0)
  }
}

# ===========================================================================
# MINI RADAR FUNCTION
# ===========================================================================
draw_mini_radar <- function(groups_to_plot, hex_id, y_limits) {

  n_ax   <- length(groups_to_plot)
  labels <- unname(group_full_names[groups_to_plot])
  y_min  <- y_limits[1]
  y_max  <- y_limits[2]

  ## Look up values for this hex
  base_vals <- sapply(groups_to_plot, function(grp) {
    hv <- hex_vals[[grp]]$base
    row <- hv[hv$hex_id == hex_id, ]
    if (nrow(row) == 0) NA_real_ else row$mean_value[1]
  })
  cond_vals <- sapply(groups_to_plot, function(grp) {
    hv <- hex_vals[[grp]]$cond
    row <- hv[hv$hex_id == hex_id, ]
    if (nrow(row) == 0) NA_real_ else row$mean_value[1]
  })
  has_cond <- !all(is.na(cond_vals))

  all_vals <- c(base_vals, cond_vals)
  all_vals <- all_vals[!is.na(all_vals)]
  if (length(all_vals) == 0) return(invisible(FALSE))

  base_clean <- ifelse(is.na(base_vals), y_min, base_vals)
  cond_clean <- ifelse(is.na(cond_vals), y_min, cond_vals)

  if (has_cond) {
    df <- as.data.frame(rbind(rep(y_max, n_ax), rep(y_min, n_ax),
                              base_clean, cond_clean))
  } else {
    df <- as.data.frame(rbind(rep(y_max, n_ax), rep(y_min, n_ax),
                              base_clean))
  }
  colnames(df) <- labels

  base_fill <- adjustcolor("#2166AC", alpha.f = 0.40)
  cond_fill <- adjustcolor("#B2182B", alpha.f = 0.30)
  base_bdr  <- "#2166AC"
  cond_bdr  <- "#B2182B"

  pcols  <- if (has_cond) c(base_bdr, cond_bdr) else base_bdr
  pfcols <- if (has_cond) c(base_fill, cond_fill) else base_fill
  plwd   <- if (has_cond) c(0.6, 0.6) else 0.6
  plty   <- if (has_cond) c(1, 2) else 1

  par(mar = c(0, 0, 0, 0))
  radarchart(df,
             axistype   = 0,
             pty        = 32,
             pcol       = pcols,
             pfcol      = pfcols,
             plwd       = plwd,
             plty       = plty,
             cglcol     = "grey60",
             cglty      = 1,
             cglwd      = 0.5,
             axislabcol = NA,
             vlabels    = rep("", n_ax),
             vlcex      = 0,
             calcex     = 0,
             title      = "")

  ## Draw coloured dots at the tip of each axis
  angles_deg <- seq(from = 90, by = 360 / n_ax, length.out = n_ax)
  angles_rad <- angles_deg * pi / 180
  for (i in seq_len(n_ax)) {
    tip_x <- cos(angles_rad[i]) * 1.0
    tip_y <- sin(angles_rad[i]) * 1.0
    points(tip_x, tip_y, pch = 16,
           col = group_colours[groups_to_plot[i]],
           cex = AXIS_DOT_CEX)
  }
  invisible(TRUE)
}

# ===========================================================================
# MAP DRAWING FUNCTION
# ===========================================================================
draw_hex_radar_map <- function(hex_ids, groups_to_plot, title,
                               xlim = NULL, ylim = NULL) {

  if (is.null(xlim)) {
    bb <- st_bbox(aus_outline)
    xlim <- c(bb["xmin"], bb["xmax"])
    ylim <- c(bb["ymin"], bb["ymax"])
  }

  y_limits <- compute_global_limits(groups_to_plot, hex_ids)

  ## --- Base map ---
  par(mar = c(3, 3, 4, 1), xpd = FALSE)

  ## Draw the hex grid as a light background
  plot(st_geometry(hex_grid), col = "grey95", border = "grey85", lwd = 0.15,
       xlim = xlim, ylim = ylim,
       main = title, cex.main = 0.95,
       axes = TRUE)

  ## Overlay Australia outline
  plot(aus_outline, col = NA, border = "grey40", lwd = 0.6, add = TRUE)

  ## Highlight hex cells with data
  data_hex <- hex_grid[hex_grid$hex_id %in% hex_ids, ]
  plot(st_geometry(data_hex), col = adjustcolor("grey88", alpha.f = 0.5),
       border = "grey65", lwd = 0.3, add = TRUE)

  ## --- Place mini radars ---
  usr <- par("usr")
  plt <- par("plt")
  fig <- par("fig")

  usr_to_ndc_x <- function(x) {
    frac <- (x - usr[1]) / (usr[2] - usr[1])
    plt_x <- plt[1] + frac * (plt[2] - plt[1])
    fig[1] + plt_x * (fig[2] - fig[1])
  }
  usr_to_ndc_y <- function(y) {
    frac <- (y - usr[3]) / (usr[4] - usr[3])
    plt_y <- plt[3] + frac * (plt[4] - plt[3])
    fig[3] + plt_y * (fig[4] - fig[3])
  }

  radar_hw <- RADAR_SIZE / 2
  n_drawn <- 0L

  for (hid in hex_ids) {
    row <- hex_grid[hex_grid$hex_id == hid, ]
    cx <- row$cx[1]
    cy <- row$cy[1]

    ## Check this hex has at least one group with data
    has_any <- any(sapply(groups_to_plot, function(grp) {
      hv <- hex_vals[[grp]]$base
      hid %in% hv$hex_id
    }))
    if (!has_any) next

    nx <- usr_to_ndc_x(cx)
    ny <- usr_to_ndc_y(cy)

    x1 <- max(0, nx - radar_hw)
    x2 <- min(1, nx + radar_hw)
    y1 <- max(0, ny - radar_hw)
    y2 <- min(1, ny + radar_hw)
    if (x2 - x1 < 0.01 || y2 - y1 < 0.01) next

    op <- par(fig = c(x1, x2, y1, y2), new = TRUE)
    ok <- tryCatch(draw_mini_radar(groups_to_plot, hid, y_limits),
                   error = function(e) FALSE)
    par(op)
    if (isTRUE(ok)) n_drawn <- n_drawn + 1L
  }

  ## --- Combined legend (top-right, below title): group labels on top, model type below ---
  n_g <- length(groups_to_plot)
  ## Total rows: n_g group rows + divider + 2 model rows
  n_total_rows <- n_g + 3
  row_h <- 1.0 / (n_total_rows + 1)   # height per row in [0,1] space
  leg_height <- 0.02 * n_total_rows + 0.03
  leg_width  <- 0.18
  leg_x2 <- fig[2] - 0.01
  leg_x1 <- leg_x2 - leg_width
  leg_y2 <- fig[4] - 0.09
  leg_y1 <- leg_y2 - leg_height

  op <- par(fig = c(leg_x1, leg_x2, leg_y1, leg_y2), new = TRUE, mar = c(0,0,0,0))
  plot.new()
  rect(0, 0, 1, 1, col = adjustcolor("white", alpha.f = 0.85), border = "grey70")

  ## Group legend: coloured dot + name
  for (i in seq_len(n_g)) {
    yy <- 1.0 - row_h * i
    grp <- groups_to_plot[i]
    points(0.08, yy, pch = 16, col = group_colours[grp], cex = 0.8)
    text(0.15, yy, group_full_names[grp], adj = c(0, 0.5), cex = 0.55)
  }

  ## Divider line
  div_y <- 1.0 - row_h * (n_g + 0.5)
  segments(0.05, div_y, 0.95, div_y, col = "grey70", lwd = 0.5)

  ## Model-type legend below
  m1_y <- 1.0 - row_h * (n_g + 1.5)
  m2_y <- 1.0 - row_h * (n_g + 2.5)
  segments(0.05, m1_y, 0.20, m1_y, col = "#2166AC", lwd = 2)
  text(0.25, m1_y, "Climate-only", adj = c(0, 0.5), cex = 0.48)
  segments(0.05, m2_y, 0.20, m2_y, col = "#B2182B", lwd = 2, lty = 2)
  text(0.25, m2_y, "Climate + Condition", adj = c(0, 0.5), cex = 0.48)
  par(op)

  cat(sprintf("  %d radars placed on hex map\n", n_drawn))
}

# ===========================================================================
# PRODUCE THE FOUR PDFs
# ===========================================================================
group_sets <- list(
  list(groups = all_groups,
       suffix = "all_groups"),
  list(groups = c("AMP", "MAM", "REP", "VAS"),
       suffix = "sans_AVES_HYM")
)

scope_sets <- list(
  list(hex_ids   = hex_grid$hex_id,
       scope_tag = "all_australia",
       scope_lbl = "All of Australia",
       xlim      = NULL,
       ylim      = NULL),
  list(hex_ids   = eastern_hex_ids,
       scope_tag = "eastern_states",
       scope_lbl = "Eastern States",
       xlim      = c(138, 154),
       ylim      = c(-44, -10))
)

for (gs in group_sets) {
  for (ss in scope_sets) {

    fname <- sprintf("radar_map_hex_2017_%s_%s_%s.pdf",
                     metric_tag, ss$scope_tag, gs$suffix)
    pdf_path <- file.path(viz_dir, fname)
    cat(sprintf("\n=== Generating: %s ===\n", fname))

    pdf(pdf_path, width = 16, height = 14)

    ttl <- sprintf("%s -- Hex Grid\nMean Temporal %s at 2017 (vs 1950 baseline)\nHex cell size: %.1f°",
                   ss$scope_lbl, metric_name, HEX_CELLSIZE)

    draw_hex_radar_map(hex_ids        = ss$hex_ids,
                       groups_to_plot = gs$groups,
                       title          = ttl,
                       xlim           = ss$xlim,
                       ylim           = ss$ylim)

    dev.off()
    cat(sprintf("  Saved: %s\n", basename(pdf_path)))
  }
}

cat("\n=== Hex-grid radar map plots complete ===\n")
