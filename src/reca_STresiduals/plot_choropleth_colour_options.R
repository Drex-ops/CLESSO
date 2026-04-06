##############################################################################
##
## plot_choropleth_colour_options.R
##
## Generate a comparison PDF showing the VAS temporal-change choropleth
## under multiple colour ramps × data transforms, so we can pick the
## option that best reveals variation across IBRA regions.
##
## Colour ramps (8):
##   Single-hue:  White→Red, White→Blue, White→Green, White→Purple
##   Multi-hue:   Heat, Viridis, YlOrRd (YellowOrangeRed), Spectral (div)
##
## Transforms (4):
##   Linear, Log(x+ε), Square-root, Quantile (rank-based)
##
## Produces one multi-page PDF with 8 ramps × 4 transforms = 32 panels
## (one per page) plus a 4×2 "sampler" overview page for each transform.
##
## Usage:
##   Rscript plot_choropleth_colour_options.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  VAS Choropleth: Colour Ramp & Transform Options\n")
cat("###########################################################################\n\n")

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

for (pkg in c("sf", "dplyr", "grDevices")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
library(sf)
library(dplyr)

# ---------------------------------------------------------------------------
# IBRA → State mapping  (for eastern-state scope)
# ---------------------------------------------------------------------------
target_states <- c("VIC", "ACT", "QLD", "NSW", "TAS")

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
# Load VAS ibra timeseries data
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

vas_dir <- find_run_folder("VAS", "base")
if (is.null(vas_dir)) stop("No VAS base run folder found!")
vas_rds <- file.path(vas_dir, "VAS_ibra_timeseries_results.rds")
if (!file.exists(vas_rds)) stop("VAS ibra timeseries results not found: ", vas_rds)

cat(sprintf("  Loading: %s\n", vas_rds))
vas_data <- readRDS(vas_rds)

# ---------------------------------------------------------------------------
# Get 2017 dissimilarity per region
# ---------------------------------------------------------------------------
get_2017_dissim <- function(region) {
  all_years <- c(vas_data$baseline_year, vas_data$target_years)
  idx_2017  <- which(all_years == 2017)
  if (length(idx_2017) == 0) idx_2017 <- length(all_years)
  col_idx <- idx_2017 - 1L

  rr <- vas_data$region_results[[region]]
  if (is.null(rr) || all(is.na(rr$mat_sim))) return(NA_real_)
  if (col_idx < 1 || col_idx > ncol(rr$mat_sim)) return(NA_real_)

  sim <- mean(rr$mat_sim[, col_idx], na.rm = TRUE)
  1.0 - sim  # dissimilarity
}

# Collect all regions from the VAS data
all_regions <- names(vas_data$region_results)
eastern_regions <- all_regions[sapply(all_regions, region_in_states)]

cat(sprintf("  VAS regions total: %d, eastern: %d\n", length(all_regions), length(eastern_regions)))

# Build value table for ALL australia
raw_vals <- sapply(all_regions, get_2017_dissim)
val_df_all <- data.frame(REG_NAME = all_regions, value = raw_vals, stringsAsFactors = FALSE)

# Build value table for eastern states
raw_vals_east <- sapply(eastern_regions, get_2017_dissim)
val_df_east <- data.frame(REG_NAME = eastern_regions, value = raw_vals_east, stringsAsFactors = FALSE)

cat(sprintf("  Non-NA values: all=%d, eastern=%d\n\n",
            sum(!is.na(val_df_all$value)), sum(!is.na(val_df_east$value))))

# ---------------------------------------------------------------------------
# Load IBRA shapefile
# ---------------------------------------------------------------------------
ibra_shp <- file.path(data_dir, "ibra51_reg", "ibra51_regions.shp")
cat("--- Loading IBRA shapefile ---\n")
if (!file.exists(ibra_shp)) stop("Shapefile not found: ", ibra_shp)
ibra_sf <- st_read(ibra_shp, quiet = TRUE)
ibra_sf <- st_make_valid(ibra_sf)

ibra_dissolved <- ibra_sf %>%
  group_by(REG_NAME) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

cat(sprintf("  %d dissolved IBRA polygons\n\n", nrow(ibra_dissolved)))

# ===========================================================================
# COLOUR RAMPS  (8 options)
# ===========================================================================
n_cols <- 100

colour_ramps <- list(

  ## --- Single-hue sequential ---
  "White → Red" = list(
    cols = colorRampPalette(c("white", "#FEE0D2", "#FC9272", "#DE2D26", "#67000D"))(n_cols),
    type = "sequential"
  ),
  "White → Blue" = list(
    cols = colorRampPalette(c("white", "#DEEBF7", "#9ECAE1", "#3182BD", "#08306B"))(n_cols),
    type = "sequential"
  ),
  "White → Green" = list(
    cols = colorRampPalette(c("white", "#E5F5E0", "#A1D99B", "#31A354", "#00441B"))(n_cols),
    type = "sequential"
  ),
  "White → Purple" = list(
    cols = colorRampPalette(c("white", "#EFEDF5", "#BCBDDC", "#756BB1", "#3F007D"))(n_cols),
    type = "sequential"
  ),

  ## --- Multi-hue sequential / diverging ---
  "Heat" = list(
    cols = colorRampPalette(c("#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C",
                              "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"))(n_cols),
    type = "sequential"
  ),
  "Viridis" = list(
    cols = colorRampPalette(c("#440154", "#482878", "#3E4A89", "#31688E",
                              "#26828E", "#1F9E89", "#35B779", "#6DCD59",
                              "#B4DE2C", "#FDE725"))(n_cols),
    type = "sequential"
  ),
  "YlOrRd" = list(
    cols = colorRampPalette(c("#FFFFB2", "#FED976", "#FEB24C", "#FD8D3C",
                              "#FC4E2A", "#E31A1C", "#B10026"))(n_cols),
    type = "sequential"
  ),
  "Spectral (div)" = list(
    cols = colorRampPalette(c("#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4",
                              "#E6F598", "#FEE08B", "#FDAE61", "#F46D43",
                              "#D53E4F", "#9E0142"))(n_cols),
    type = "diverging"
  )
)

# ===========================================================================
# TRANSFORMS (4 options)
# ===========================================================================
## Each transform returns a *transformed* copy of the value vector and a
## label-formatting function (for the colour-bar tick labels).

apply_transform <- function(vals, transform_name) {
  v <- vals
  v_nona <- v[!is.na(v)]
  eps <- max(min(v_nona[v_nona > 0]) * 0.01, 1e-9)

  switch(transform_name,
    "Linear" = list(
      vals  = v,
      vmin  = 0,
      vmax  = max(v_nona, na.rm = TRUE),
      fmt   = function(x) sprintf("%.4f", x),
      label = "Linear scale"
    ),
    "Log" = {
      vt <- ifelse(is.na(v), NA, log10(v + eps))
      vt_nona <- vt[!is.na(vt)]
      list(
        vals  = vt,
        vmin  = min(vt_nona),
        vmax  = max(vt_nona),
        fmt   = function(x) sprintf("%.2f", 10^x - eps),
        label = "log10(x + eps)"
      )
    },
    "Sqrt" = {
      vt <- ifelse(is.na(v), NA, sqrt(pmax(v, 0)))
      vt_nona <- vt[!is.na(vt)]
      list(
        vals  = vt,
        vmin  = 0,
        vmax  = max(vt_nona),
        fmt   = function(x) sprintf("%.4f", x^2),
        label = "sqrt(x)"
      )
    },
    "Quantile" = {
      ## Rank-based mapping: each value gets its quantile position [0,1]
      vt <- ifelse(is.na(v), NA, rank(v, ties.method = "average", na.last = "keep"))
      n_valid <- sum(!is.na(vt))
      vt <- ifelse(is.na(vt), NA, (vt - 1) / max(n_valid - 1, 1))
      list(
        vals  = vt,
        vmin  = 0,
        vmax  = 1,
        fmt   = function(x) sprintf("Q%.0f%%", x * 100),
        label = "Quantile (rank)"
      )
    }
  )
}

# ===========================================================================
# CHOROPLETH DRAWING FUNCTION  (colour-ramp aware)
# ===========================================================================
val_to_col <- function(v, vmin, vmax, ramp_cols) {
  if (is.na(v)) return("grey85")
  idx <- round((v - vmin) / max(vmax - vmin, 1e-9) * (length(ramp_cols) - 1)) + 1L
  idx <- max(1L, min(length(ramp_cols), idx))
  ramp_cols[idx]
}

draw_choropleth <- function(values_df, regions, title,
                            xlim = NULL, ylim = NULL,
                            vmin = 0, vmax = 1,
                            ramp_cols = NULL,
                            tick_fmt = function(x) sprintf("%.3f", x),
                            scale_label = "Change in Community Composition") {
  if (is.null(ramp_cols)) ramp_cols <- colour_ramps[["White → Red"]]$cols

  plot_sf <- ibra_dissolved %>%
    left_join(values_df, by = "REG_NAME")

  plot_sf$fill <- sapply(seq_len(nrow(plot_sf)), function(i) {
    reg <- plot_sf$REG_NAME[i]
    if (!(reg %in% regions)) return("grey92")
    val_to_col(plot_sf$value[i], vmin, vmax, ramp_cols)
  })

  if (is.null(xlim)) {
    bb <- st_bbox(ibra_dissolved)
    xlim <- c(bb["xmin"], bb["xmax"])
    ylim <- c(bb["ymin"], bb["ymax"])
  }

  par(mar = c(3, 3, 5, 7))
  plot(st_geometry(plot_sf), col = plot_sf$fill, border = "grey50", lwd = 0.3,
       xlim = xlim, ylim = ylim,
       main = title, cex.main = 0.85,
       axes = TRUE)

  ## Colour bar legend
  usr <- par("usr")
  bar_x1 <- usr[2] + (usr[2] - usr[1]) * 0.02
  bar_x2 <- usr[2] + (usr[2] - usr[1]) * 0.04
  bar_y1 <- usr[3] + (usr[4] - usr[3]) * 0.15
  bar_y2 <- usr[4] - (usr[4] - usr[3]) * 0.15

  n_bar <- 50
  bar_ys <- seq(bar_y1, bar_y2, length.out = n_bar + 1)

  par(xpd = TRUE)
  for (k in seq_len(n_bar)) {
    v <- vmin + (k - 0.5) / n_bar * (vmax - vmin)
    rect(bar_x1, bar_ys[k], bar_x2, bar_ys[k + 1],
         col = val_to_col(v, vmin, vmax, ramp_cols), border = NA)
  }
  rect(bar_x1, bar_y1, bar_x2, bar_y2, col = NA, border = "grey30", lwd = 0.5)

  n_ticks <- 5
  tick_vals <- seq(vmin, vmax, length.out = n_ticks)
  tick_ys   <- seq(bar_y1, bar_y2, length.out = n_ticks)
  for (k in seq_len(n_ticks)) {
    segments(bar_x2, tick_ys[k], bar_x2 + (bar_x2 - bar_x1) * 0.3, tick_ys[k],
             col = "grey30", lwd = 0.5)
    text(bar_x2 + (bar_x2 - bar_x1) * 0.5, tick_ys[k],
         tick_fmt(tick_vals[k]), adj = c(0, 0.5), cex = 0.6)
  }

  text((bar_x1 + bar_x2) / 2, bar_y2 + (usr[4] - usr[3]) * 0.03,
       scale_label, cex = 0.55, srt = 0)
  par(xpd = FALSE)
}

# ===========================================================================
# SAMPLER PAGE: 4×2 grid showing all ramps for a single transform
# ===========================================================================
draw_sampler_page <- function(values_df, regions, transform_name,
                              xlim = NULL, ylim = NULL,
                              scope_label = "") {
  ## Apply transform
  tr <- apply_transform(values_df$value, transform_name)
  df_t <- data.frame(REG_NAME = values_df$REG_NAME, value = tr$vals,
                     stringsAsFactors = FALSE)

  par(mfrow = c(4, 2), mar = c(1.5, 1.5, 3, 4), oma = c(0, 0, 3, 0))

  for (ramp_name in names(colour_ramps)) {
    ramp <- colour_ramps[[ramp_name]]

    plot_sf <- ibra_dissolved %>%
      left_join(df_t, by = "REG_NAME")

    plot_sf$fill <- sapply(seq_len(nrow(plot_sf)), function(i) {
      reg <- plot_sf$REG_NAME[i]
      if (!(reg %in% regions)) return("grey92")
      val_to_col(plot_sf$value[i], tr$vmin, tr$vmax, ramp$cols)
    })

    bb <- st_bbox(ibra_dissolved)
    xl <- if (!is.null(xlim)) xlim else c(bb["xmin"], bb["xmax"])
    yl <- if (!is.null(ylim)) ylim else c(bb["ymin"], bb["ymax"])

    plot(st_geometry(plot_sf), col = plot_sf$fill, border = "grey50", lwd = 0.2,
         xlim = xl, ylim = yl,
         main = ramp_name, cex.main = 0.9,
         axes = FALSE)
    box(lwd = 0.3)

    ## Mini colour bar
    usr <- par("usr")
    bx1 <- usr[2] + (usr[2] - usr[1]) * 0.02
    bx2 <- usr[2] + (usr[2] - usr[1]) * 0.05
    by1 <- usr[3] + (usr[4] - usr[3]) * 0.15
    by2 <- usr[4] - (usr[4] - usr[3]) * 0.15
    nb  <- 30
    bys <- seq(by1, by2, length.out = nb + 1)
    par(xpd = TRUE)
    for (k in seq_len(nb)) {
      v <- tr$vmin + (k - 0.5) / nb * (tr$vmax - tr$vmin)
      rect(bx1, bys[k], bx2, bys[k + 1],
           col = val_to_col(v, tr$vmin, tr$vmax, ramp$cols), border = NA)
    }
    rect(bx1, by1, bx2, by2, col = NA, border = "grey30", lwd = 0.4)
    ## Min / max labels
    text(bx2 + (bx2 - bx1) * 0.3, by1, tr$fmt(tr$vmin), cex = 0.45, adj = c(0, 0.5))
    text(bx2 + (bx2 - bx1) * 0.3, by2, tr$fmt(tr$vmax), cex = 0.45, adj = c(0, 0.5))
    par(xpd = FALSE)
  }

  mtext(sprintf("VAS — %s  |  Transform: %s  [%s]",
                scope_label, transform_name, tr$label),
        outer = TRUE, cex = 1.0, line = 1)
}

# ===========================================================================
# GENERATE THE PDF
# ===========================================================================
transform_names <- c("Linear", "Log", "Sqrt", "Quantile")
ramp_names      <- names(colour_ramps)

## We produce two PDFs: All-Australia and Eastern States
scopes <- list(
  list(val_df    = val_df_all,
       regions   = all_regions,
       scope_tag = "all_australia",
       scope_lbl = "All of Australia",
       xlim      = NULL,
       ylim      = NULL),
  list(val_df    = val_df_east,
       regions   = eastern_regions,
       scope_tag = "eastern_states",
       scope_lbl = "Eastern States",
       xlim      = c(138, 154),
       ylim      = c(-44, -10))
)

for (sc in scopes) {
  fname <- sprintf("VAS_choropleth_colour_options_%s.pdf", sc$scope_tag)
  pdf_path <- file.path(viz_dir, fname)
  cat(sprintf("\n=== Generating: %s ===\n", fname))

  pdf(pdf_path, width = 16, height = 10)

  ## ---- Sampler overview pages (one per transform) ----
  for (tname in transform_names) {
    cat(sprintf("  Sampler page: %s\n", tname))
    draw_sampler_page(sc$val_df, sc$regions, tname,
                      xlim = sc$xlim, ylim = sc$ylim,
                      scope_label = sc$scope_lbl)
  }

  ## ---- Full-size individual pages: each ramp × each transform ----
  par(mfrow = c(1, 1))
  for (tname in transform_names) {
    tr <- apply_transform(sc$val_df$value, tname)
    df_t <- data.frame(REG_NAME = sc$val_df$REG_NAME, value = tr$vals,
                       stringsAsFactors = FALSE)

    for (rname in ramp_names) {
      ramp <- colour_ramps[[rname]]
      ttl <- sprintf("VAS — %s\nColour: %s  |  Transform: %s  [%s]\nTemporal Dissimilarity at 2017 (vs 1950 baseline)",
                     sc$scope_lbl, rname, tname, tr$label)
      draw_choropleth(df_t, sc$regions, ttl,
                      xlim = sc$xlim, ylim = sc$ylim,
                      vmin = tr$vmin, vmax = tr$vmax,
                      ramp_cols = ramp$cols,
                      tick_fmt = tr$fmt,
                      scale_label = "Dissimilarity")
      cat(sprintf("  Page: %s × %s\n", rname, tname))
    }
  }

  dev.off()
  cat(sprintf("  Saved: %s\n", basename(pdf_path)))
}

cat("\n=== VAS colour options complete ===\n")
