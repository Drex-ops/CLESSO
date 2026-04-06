##############################################################################
##
## plot_radar_map.R
##
## Places mini radar plots on a stylised map of Australia, one radar per
## IBRA region positioned at the region centroid.  Each radar shows the
## 2017-epoch temporal change across biological groups.
##
## Produces four PDFs:
##   1) All of Australia – all groups
##   2) All of Australia – sans AVES & HYM
##   3) Eastern states   – all groups
##   4) Eastern states   – sans AVES & HYM
##
## Usage:
##   Rscript plot_radar_map.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Radar Map: Temporal Change at 2017 Epoch\n")
cat("###########################################################################\n\n")

# ---------------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------------
PLOT_DISSIMILARITY <- TRUE

metric_name <- if (PLOT_DISSIMILARITY) "Change in Community Composition" else "Similarity in Community Composition"
metric_tag  <- if (PLOT_DISSIMILARITY) "dissim" else "sim"

## Radar size (fraction of the plot width occupied by each mini radar)
RADAR_SIZE <- 0.07   # adjust up/down to make radars bigger/smaller

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
# IBRA → State mapping
# ---------------------------------------------------------------------------
target_states <- c("VIC", "ACT", "QLD", "NSW", "TAS")
state_label   <- paste(sort(target_states), collapse = ", ")

ibra_state_map <- list(
  "Arnhem Coast"                  = "NT",
  "Arnhem Plateau"                = "NT",
  "Australian Alps"               = c("ACT", "NSW", "VIC"),
  "Avon Wheatbelt"                = "WA",
  "Ben Lomond"                    = "TAS",
  "Brigalow Belt North"           = "QLD",
  "Brigalow Belt South"           = c("NSW", "QLD"),
  "Broken Hill Complex"           = c("NSW", "SA"),
  "Burt Plain"                    = "NT",
  "Cape York Peninsula"           = "QLD",
  "Carnarvon"                     = "WA",
  "Central Arnhem"                = "NT",
  "Central Kimberley"             = "WA",
  "Central Mackay Coast"          = "QLD",
  "Central Ranges"                = c("NT", "SA", "WA"),
  "Channel Country"               = c("NSW", "NT", "QLD", "SA"),
  "Cobar Peneplain"               = "NSW",
  "Coolgardie"                    = "WA",
  "Daly Basin"                    = "NT",
  "Dampierland"                   = "WA",
  "Darling Riverine Plains"       = c("NSW", "QLD"),
  "Darwin Coastal"                = "NT",
  "Davenport Murchison Ranges"    = "NT",
  "Desert Uplands"                = "QLD",
  "Einasleigh Uplands"            = "QLD",
  "Esperance Plains"              = "WA",
  "Eyre Yorke Block"              = "SA",
  "Finke"                         = c("NT", "SA"),
  "Flinders"                      = c("TAS", "VIC"),
  "Flinders Lofty Block"          = "SA",
  "Gascoyne"                      = "WA",
  "Gawler"                        = "SA",
  "Geraldton Sandplains"          = "WA",
  "Gibson Desert"                 = "WA",
  "Great Sandy Desert"            = c("NT", "WA"),
  "Great Victoria Desert"         = c("SA", "WA"),
  "Gulf Coastal"                  = "NT",
  "Gulf Fall and Uplands"         = c("NT", "QLD"),
  "Gulf Plains"                   = c("NT", "QLD"),
  "Hampton"                       = c("SA", "WA"),
  "Jarrah Forest"                 = "WA",
  "Kanmantoo"                     = "SA",
  "King"                          = "TAS",
  "Little Sandy Desert"           = "WA",
  "MacDonnell Ranges"             = "NT",
  "Mallee"                        = "WA",
  "Mitchell Grass Downs"          = c("NT", "QLD"),
  "Mount Isa Inlier"              = c("NT", "QLD"),
  "Mulga Lands"                   = c("NSW", "QLD"),
  "Murchison"                     = "WA",
  "Murray Darling Depression"     = c("NSW", "SA", "VIC"),
  "NSW North Coast"               = c("NSW", "QLD"),
  "NSW South Western Slopes"      = c("NSW", "VIC"),
  "Nandewar"                      = c("NSW", "QLD"),
  "Naracoorte Coastal Plain"      = c("SA", "VIC"),
  "New England Tableland"         = c("NSW", "QLD"),
  "Northern Kimberley"            = "WA",
  "Nullarbor"                     = c("SA", "WA"),
  "Ord Victoria Plain"            = c("NT", "WA"),
  "Pilbara"                       = "WA",
  "Pine Creek"                    = "NT",
  "Riverina"                      = c("NSW", "SA", "VIC"),
  "Simpson Strzelecki Dunefields" = c("NSW", "NT", "QLD", "SA"),
  "South East Coastal Plain"      = "VIC",
  "South East Corner"             = c("NSW", "VIC"),
  "South Eastern Highlands"       = c("ACT", "NSW", "VIC"),
  "South Eastern Queensland"      = c("NSW", "QLD"),
  "Stony Plains"                  = c("NT", "SA"),
  "Sturt Plateau"                 = "NT",
  "Swan Coastal Plain"            = "WA",
  "Sydney Basin"                  = c("ACT", "NSW"),
  "Tanami"                        = c("NT", "WA"),
  "Tasmanian Central Highlands"   = "TAS",
  "Tasmanian Northern Midlands"   = "TAS",
  "Tasmanian Northern Slopes"     = "TAS",
  "Tasmanian South East"          = "TAS",
  "Tasmanian Southern Ranges"     = "TAS",
  "Tasmanian West"                = "TAS",
  "Tiwi Cobourg"                  = "NT",
  "Victoria Bonaparte"            = c("NT", "WA"),
  "Victorian Midlands"            = "VIC",
  "Victorian Volcanic Plain"      = c("SA", "VIC"),
  "Warren"                        = "WA",
  "Wet Tropics"                   = "QLD",
  "Yalgoo"                        = "WA"
)

region_in_states <- function(region_name, states = target_states) {
  mapped <- ibra_state_map[[region_name]]
  if (is.null(mapped)) return(FALSE)
  any(mapped %in% states)
}

# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------
all_groups <- c("AMP", "AVES", "HYM", "MAM", "REP", "VAS")

group_colours <- c(
  AMP  = "#E41A1C",   # red
  AVES = "#377EB8",   # blue
  HYM  = "#FF7F00",   # orange
  MAM  = "#4DAF4A",   # green
  REP  = "#984EA3",   # purple
  VAS  = "#A65628"    # brown
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
# Get 2017 value (similarity or dissimilarity) for a group/region
# ---------------------------------------------------------------------------
get_2017_val <- function(group, model_type = "base", regions = NULL) {
  dat <- ibra_data[[group]][[model_type]]
  if (is.null(dat)) return(NA_real_)

  all_years <- c(dat$baseline_year, dat$target_years)
  idx_2017  <- which(all_years == 2017)
  if (length(idx_2017) == 0) idx_2017 <- length(all_years)
  col_idx <- idx_2017 - 1L

  if (is.null(regions)) regions <- names(dat$region_results)

  vals <- c()
  for (reg in regions) {
    rr <- dat$region_results[[reg]]
    if (is.null(rr) || all(is.na(rr$mat_sim))) next
    if (col_idx < 1 || col_idx > ncol(rr$mat_sim)) next
    vals <- c(vals, rr$mat_sim[, col_idx])
  }

  if (length(vals) == 0) return(NA_real_)
  sim <- mean(vals, na.rm = TRUE)
  if (PLOT_DISSIMILARITY) 1.0 - sim else sim
}

# ---------------------------------------------------------------------------
# Collect all IBRA region names present in any group
# ---------------------------------------------------------------------------
all_ibra_regions <- sort(unique(unlist(lapply(all_groups, function(grp) {
  regs <- character()
  if (!is.null(ibra_data[[grp]]$base))
    regs <- c(regs, names(ibra_data[[grp]]$base$region_results))
  if (!is.null(ibra_data[[grp]]$cond))
    regs <- c(regs, names(ibra_data[[grp]]$cond$region_results))
  regs
}))))
eastern_regions <- all_ibra_regions[sapply(all_ibra_regions, region_in_states)]

cat(sprintf("  Total IBRA regions in data: %d\n", length(all_ibra_regions)))
cat(sprintf("  Eastern-state regions:      %d\n\n", length(eastern_regions)))

# ---------------------------------------------------------------------------
# Load IBRA shapefile for the base map + centroids
# ---------------------------------------------------------------------------
ibra_shp <- file.path(data_dir, "ibra51_reg", "ibra51_regions.shp")
cat("--- Loading IBRA shapefile ---\n")
if (!file.exists(ibra_shp)) stop("Shapefile not found: ", ibra_shp)
ibra_sf <- st_read(ibra_shp, quiet = TRUE)
ibra_sf <- st_make_valid(ibra_sf)

## Dissolve to one polygon per REG_NAME
ibra_dissolved <- ibra_sf %>%
  group_by(REG_NAME) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

## Compute centroids
centroids <- st_centroid(ibra_dissolved)
centroid_coords <- st_coordinates(centroids)
centroid_df <- data.frame(
  REG_NAME = ibra_dissolved$REG_NAME,
  lon      = centroid_coords[, 1],
  lat      = centroid_coords[, 2],
  stringsAsFactors = FALSE
)
cat(sprintf("  %d IBRA polygons loaded, %d centroids\n\n", nrow(ibra_dissolved), nrow(centroid_df)))

# ---------------------------------------------------------------------------
# Pre-compute a global y_max for consistent radar scales across all regions
# ---------------------------------------------------------------------------
compute_global_limits <- function(groups_to_plot, regions) {
  all_v <- c()
  for (reg in regions) {
    for (grp in groups_to_plot) {
      all_v <- c(all_v,
                 get_2017_val(grp, "base", reg),
                 get_2017_val(grp, "cond", reg))
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
# MINI RADAR DRAWING FUNCTION (no titles, no legends, compact)
# ===========================================================================
## Draws a small radar into the CURRENT viewport / subplot region.
## Uses a GLOBAL y_min / y_max for consistent scale.
draw_mini_radar <- function(groups_to_plot, region, y_limits) {

  n_ax   <- length(groups_to_plot)
  labels <- unname(group_full_names[groups_to_plot])
  y_min  <- y_limits[1]
  y_max  <- y_limits[2]

  base_vals <- sapply(groups_to_plot, function(grp) get_2017_val(grp, "base", region))
  cond_vals <- sapply(groups_to_plot, function(grp) get_2017_val(grp, "cond", region))
  has_cond  <- !all(is.na(cond_vals))

  all_vals  <- c(base_vals, cond_vals)
  all_vals  <- all_vals[!is.na(all_vals)]
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
  plwd   <- if (has_cond) c(1.5, 1.5) else 1.5
  plty   <- if (has_cond) c(1, 2) else 1

  par(mar = c(0, 0, 0, 0))
  radarchart(df,
             axistype  = 0,
             pcol      = pcols,
             pfcol     = pfcols,
             plwd      = plwd,
             plty      = plty,
             cglcol    = "grey60",
             cglty     = 1,
             cglwd     = 0.5,
             axislabcol = NA,
             vlcex     = 0,          # hide vertex labels for the mini version
             calcex    = 0,
             title     = "")

  invisible(TRUE)
}

# ===========================================================================
# MAP DRAWING FUNCTION
# ===========================================================================
## Draws the base map, then overlays mini radars at centroids.
## regions_to_plot: character vector of IBRA region names to include
## groups_to_plot:  character vector of group codes
## title:           main title
## xlim, ylim:      optional map extent overrides
draw_radar_map <- function(regions_to_plot, groups_to_plot, title,
                           xlim = NULL, ylim = NULL) {

  ## Which dissolved polygons to show on the base map?
  ## Show all polygons for context but only place radars on selected regions.
  if (is.null(xlim)) {
    bb <- st_bbox(ibra_dissolved)
    xlim <- c(bb["xmin"], bb["xmax"])
    ylim <- c(bb["ymin"], bb["ymax"])
  }

  ## Pre-compute global limits for consistent scale
  y_limits <- compute_global_limits(groups_to_plot, regions_to_plot)

  ## --- Draw the base map ---
  par(mar = c(3, 3, 4, 1), xpd = FALSE)
  plot(st_geometry(ibra_dissolved), col = "grey92", border = "grey70", lwd = 0.3,
       xlim = xlim, ylim = ylim,
       main = title, cex.main = 0.95,
       axes = TRUE)

  ## Highlight the regions that have data
  has_data_idx <- which(ibra_dissolved$REG_NAME %in% regions_to_plot)
  if (length(has_data_idx) > 0) {
    plot(st_geometry(ibra_dissolved[has_data_idx, ]),
         col = "grey85", border = "grey55", lwd = 0.5, add = TRUE)
  }

  ## --- Overlay mini radars ---
  ## We use subplot() to place each radar.  We need to convert geographic

  ## coords to user coords (they're the same since we're using longlat).
  ## subplot() from TeachingDemos would be ideal but to keep dependencies
  ## light we use par("fig") + par("plt") coordinate math.

  ## Get the plot region in user coordinates
  usr <- par("usr")         # c(x1, x2, y1, y2)
  plt <- par("plt")         # c(left, right, bottom, top) as fraction of figure
  fig <- par("fig")         # c(left, right, bottom, top) as fraction of device

  ## Convert user coords to normalised device coords (ndc)
  usr_to_ndc_x <- function(x) {
    frac <- (x - usr[1]) / (usr[2] - usr[1])   # fraction within plot region
    plt_x <- plt[1] + frac * (plt[2] - plt[1]) # fraction within figure
    fig[1] + plt_x * (fig[2] - fig[1])          # fraction within device
  }
  usr_to_ndc_y <- function(y) {
    frac <- (y - usr[3]) / (usr[4] - usr[3])
    plt_y <- plt[3] + frac * (plt[4] - plt[3])
    fig[3] + plt_y * (fig[4] - fig[3])
  }

  ## Radar half-width in NDC
  radar_hw <- RADAR_SIZE / 2

  n_drawn <- 0L
  for (reg in regions_to_plot) {
    ci <- which(centroid_df$REG_NAME == reg)
    if (length(ci) == 0) next

    cx <- centroid_df$lon[ci[1]]
    cy <- centroid_df$lat[ci[1]]

    ## Convert to NDC
    nx <- usr_to_ndc_x(cx)
    ny <- usr_to_ndc_y(cy)

    ## Clamp so the subplot stays within [0,1]
    x1 <- max(0, nx - radar_hw)
    x2 <- min(1, nx + radar_hw)
    y1 <- max(0, ny - radar_hw)
    y2 <- min(1, ny + radar_hw)

    ## Skip if too small (region outside visible extent)
    if (x2 - x1 < 0.01 || y2 - y1 < 0.01) next

    ## Draw the mini radar in a subplot
    op <- par(fig = c(x1, x2, y1, y2), new = TRUE)
    ok <- tryCatch({
      draw_mini_radar(groups_to_plot, reg, y_limits)
    }, error = function(e) FALSE)
    par(op)
    if (isTRUE(ok)) n_drawn <- n_drawn + 1L
  }

  ## --- Legend in the bottom-left corner ---
  ## Draw a small legend subplot
  leg_x1 <- fig[1] + 0.01
  leg_x2 <- leg_x1 + 0.18
  leg_y1 <- fig[3] + 0.01
  leg_y2 <- leg_y1 + 0.08 + 0.015 * length(groups_to_plot)

  op <- par(fig = c(leg_x1, leg_x2, leg_y1, leg_y2), new = TRUE, mar = c(0,0,0,0))
  plot.new()
  rect(0, 0, 1, 1, col = adjustcolor("white", alpha.f = 0.85), border = "grey70")

  ## Show group colour key as coloured dots
  n_g <- length(groups_to_plot)
  y_pos <- seq(0.9, 0.1, length.out = n_g)
  for (i in seq_len(n_g)) {
    grp <- groups_to_plot[i]
    points(0.08, y_pos[i], pch = 16, col = group_colours[grp], cex = 1.2)
    text(0.15, y_pos[i], group_full_names[grp], adj = c(0, 0.5), cex = 0.55)
  }
  par(op)

  ## --- Model legend (base vs cond) at bottom-right ---
  leg2_x2 <- fig[2] - 0.01
  leg2_x1 <- leg2_x2 - 0.14
  leg2_y1 <- fig[3] + 0.01
  leg2_y2 <- leg2_y1 + 0.06

  op <- par(fig = c(leg2_x1, leg2_x2, leg2_y1, leg2_y2), new = TRUE, mar = c(0,0,0,0))
  plot.new()
  rect(0, 0, 1, 1, col = adjustcolor("white", alpha.f = 0.85), border = "grey70")
  segments(0.05, 0.7, 0.20, 0.7, col = "#2166AC", lwd = 2)
  text(0.25, 0.7, "Climate-only", adj = c(0, 0.5), cex = 0.48)
  segments(0.05, 0.3, 0.20, 0.3, col = "#B2182B", lwd = 2, lty = 2)
  text(0.25, 0.3, "Climate + Condition", adj = c(0, 0.5), cex = 0.48)
  par(op)

  cat(sprintf("  %d radars placed on map\n", n_drawn))
}

# ===========================================================================
# PRODUCE THE FOUR PDF VARIANTS
# ===========================================================================
group_sets <- list(
  list(groups = all_groups,
       suffix = "all_groups"),
  list(groups = c("AMP", "MAM", "REP", "VAS"),
       suffix = "sans_AVES_HYM")
)

## Eastern states: crop to lon >= 138, lat roughly -45 to -10
scope_sets <- list(
  list(regions   = all_ibra_regions,
       scope_tag = "all_australia",
       scope_lbl = "All of Australia",
       xlim      = NULL,
       ylim      = NULL),
  list(regions   = eastern_regions,
       scope_tag = "eastern_states",
       scope_lbl = sprintf("Eastern States (%s)", state_label),
       xlim      = c(138, 154),
       ylim      = c(-44, -10))
)

for (gs in group_sets) {
  for (ss in scope_sets) {

    fname <- sprintf("radar_map_2017_%s_%s_%s.pdf", metric_tag, ss$scope_tag, gs$suffix)
    pdf_path <- file.path(viz_dir, fname)
    cat(sprintf("\n=== Generating: %s ===\n", fname))

    pdf(pdf_path, width = 16, height = 14)

    ttl <- sprintf("%s\nMean Temporal %s at 2017 (vs 1950 baseline)",
                   ss$scope_lbl, metric_name)

    draw_radar_map(regions_to_plot = ss$regions,
                   groups_to_plot  = gs$groups,
                   title           = ttl,
                   xlim            = ss$xlim,
                   ylim            = ss$ylim)

    dev.off()
    cat(sprintf("  Saved: %s\n", basename(pdf_path)))
  }
}

cat("\n=== Radar map plots complete ===\n")
