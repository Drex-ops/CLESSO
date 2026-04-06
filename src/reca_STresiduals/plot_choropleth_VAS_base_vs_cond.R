##############################################################################
##
## plot_choropleth_VAS_base_vs_cond.R
##
## Choropleth maps of VAS temporal change (dissimilarity) at the 2017 epoch.
## White-to-red colour ramp, linear scale.
##
## Produces a single 4-page PDF:
##   1. All Australia  — Climate-only model  (base, 1950 baseline)
##   2. All Australia  — Climate + Condition model (COND baseline)
##   3. Eastern States — Climate-only model  (base, 1950 baseline)
##   4. Eastern States — Climate + Condition model (COND baseline)
##
## All 4 pages share one colour scale so they are directly comparable.
##
## Usage:
##   Rscript plot_choropleth_VAS_base_vs_cond.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  VAS Choropleth: Climate-only vs Climate + Condition\n")
cat("###########################################################################\n\n")

PLOT_DISSIMILARITY <- TRUE
metric_name <- if (PLOT_DISSIMILARITY) "Change in Community Composition" else "Similarity in Community Composition"

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

for (pkg in c("sf", "dplyr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
library(sf)
library(dplyr)

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
# Load VAS base and COND IBRA timeseries
# ---------------------------------------------------------------------------
find_run_folder <- function(group, variant = "base") {
  pattern <- switch(variant,
    base  = sprintf("^%s_\\d{8}T\\d{6}$", group),
    COND  = sprintf("^%s_COND_\\d{8}T\\d{6}$", group)
  )
  dirs <- list.dirs(output_dir, full.names = FALSE, recursive = FALSE)
  matches <- dirs[grepl(pattern, dirs)]
  if (length(matches) == 0) return(NULL)
  file.path(output_dir, sort(matches, decreasing = TRUE)[1])
}

base_dir <- find_run_folder("VAS", "base")
cond_dir <- find_run_folder("VAS", "COND")

base_rds <- if (!is.null(base_dir)) file.path(base_dir, "VAS_ibra_timeseries_results.rds") else NULL
cond_rds <- if (!is.null(cond_dir)) file.path(cond_dir, "VAS_ibra_timeseries_results.rds") else NULL

vas_base <- if (!is.null(base_rds) && file.exists(base_rds)) readRDS(base_rds) else NULL
vas_cond <- if (!is.null(cond_rds) && file.exists(cond_rds)) readRDS(cond_rds) else NULL

cat(sprintf("  VAS base: %s  (baseline year: %s)\n",
            if (!is.null(vas_base)) "YES" else "---",
            if (!is.null(vas_base)) as.character(vas_base$baseline_year) else "N/A"))
cat(sprintf("  VAS cond: %s  (baseline year: %s)\n\n",
            if (!is.null(vas_cond)) "YES" else "---",
            if (!is.null(vas_cond)) as.character(vas_cond$baseline_year) else "N/A"))

if (is.null(vas_base)) stop("VAS base IBRA timeseries not found!")

# ---------------------------------------------------------------------------
# Extract 2017 dissimilarity per region for a given model
# ---------------------------------------------------------------------------
get_2017_dissim <- function(dat, region) {
  if (is.null(dat)) return(NA_real_)
  all_years <- c(dat$baseline_year, dat$target_years)
  idx_2017  <- which(all_years == 2017)
  if (length(idx_2017) == 0) idx_2017 <- length(all_years)
  col_idx <- idx_2017 - 1L

  rr <- dat$region_results[[region]]
  if (is.null(rr) || all(is.na(rr$mat_sim))) return(NA_real_)
  if (col_idx < 1 || col_idx > ncol(rr$mat_sim)) return(NA_real_)

  sim <- mean(rr$mat_sim[, col_idx], na.rm = TRUE)
  1.0 - sim  # dissimilarity
}

# ---------------------------------------------------------------------------
# Collect regions
# ---------------------------------------------------------------------------
all_regions <- sort(unique(c(
  names(vas_base$region_results),
  if (!is.null(vas_cond)) names(vas_cond$region_results) else character()
)))
eastern_regions <- all_regions[sapply(all_regions, region_in_states)]

cat(sprintf("  Total IBRA regions: %d,  Eastern: %d\n\n",
            length(all_regions), length(eastern_regions)))

# ---------------------------------------------------------------------------
# Build value vectors
# ---------------------------------------------------------------------------
base_vals_all  <- sapply(all_regions, function(r) get_2017_dissim(vas_base, r))
cond_vals_all  <- sapply(all_regions, function(r) get_2017_dissim(vas_cond, r))
base_vals_east <- sapply(eastern_regions, function(r) get_2017_dissim(vas_base, r))
cond_vals_east <- sapply(eastern_regions, function(r) get_2017_dissim(vas_cond, r))

cat(sprintf("  Base all-Aus:  %d non-NA of %d\n", sum(!is.na(base_vals_all)), length(base_vals_all)))
cat(sprintf("  Cond all-Aus:  %d non-NA of %d\n", sum(!is.na(cond_vals_all)), length(cond_vals_all)))
cat(sprintf("  Base eastern:  %d non-NA of %d\n", sum(!is.na(base_vals_east)), length(base_vals_east)))
cat(sprintf("  Cond eastern:  %d non-NA of %d\n\n", sum(!is.na(cond_vals_east)), length(cond_vals_east)))

# ---------------------------------------------------------------------------
# Shared colour scale across all 4 pages
# ---------------------------------------------------------------------------
all_vals <- c(base_vals_all, cond_vals_all, base_vals_east, cond_vals_east)
all_vals <- all_vals[!is.na(all_vals)]
vmin <- 0
vmax <- ceiling(max(all_vals) * 100) / 100
cat(sprintf("  Shared colour scale: [%.4f, %.4f]\n\n", vmin, vmax))

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

# ---------------------------------------------------------------------------
# White-to-red colour ramp (linear)
# ---------------------------------------------------------------------------
n_cols  <- 100
wr_ramp <- colorRampPalette(c("white", "#FEE0D2", "#FC9272", "#DE2D26", "#67000D"))(n_cols)

val_to_col <- function(v, vmin, vmax) {
  if (is.na(v)) return("grey85")
  idx <- round((v - vmin) / max(vmax - vmin, 1e-9) * (n_cols - 1)) + 1L
  idx <- max(1L, min(n_cols, idx))
  wr_ramp[idx]
}

# ===========================================================================
# CHOROPLETH DRAWING FUNCTION
# ===========================================================================
draw_choropleth <- function(values_df, regions, title,
                            xlim = NULL, ylim = NULL,
                            vmin = 0, vmax = 1) {

  plot_sf <- ibra_dissolved %>%
    left_join(values_df, by = "REG_NAME")

  plot_sf$fill <- sapply(seq_len(nrow(plot_sf)), function(i) {
    reg <- plot_sf$REG_NAME[i]
    if (!(reg %in% regions)) return("grey92")
    val_to_col(plot_sf$value[i], vmin, vmax)
  })

  if (is.null(xlim)) {
    bb <- st_bbox(ibra_dissolved)
    xlim <- c(bb["xmin"], bb["xmax"])
    ylim <- c(bb["ymin"], bb["ymax"])
  }

  par(mar = c(3, 3, 4, 6))
  plot(st_geometry(plot_sf), col = plot_sf$fill, border = "grey50", lwd = 0.3,
       xlim = xlim, ylim = ylim,
       main = title, cex.main = 1.0,
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
         col = val_to_col(v, vmin, vmax), border = NA)
  }
  rect(bar_x1, bar_y1, bar_x2, bar_y2, col = NA, border = "grey30", lwd = 0.5)

  n_ticks <- 5
  tick_vals <- seq(vmin, vmax, length.out = n_ticks)
  tick_ys   <- seq(bar_y1, bar_y2, length.out = n_ticks)
  for (k in seq_len(n_ticks)) {
    segments(bar_x2, tick_ys[k], bar_x2 + (bar_x2 - bar_x1) * 0.3, tick_ys[k],
             col = "grey30", lwd = 0.5)
    text(bar_x2 + (bar_x2 - bar_x1) * 0.5, tick_ys[k],
         sprintf("%.4f", tick_vals[k]), adj = c(0, 0.5), cex = 0.65)
  }

  text((bar_x1 + bar_x2) / 2, bar_y2 + (usr[4] - usr[3]) * 0.03,
       metric_name, cex = 0.6, srt = 0)
  par(xpd = FALSE)
}

# ===========================================================================
# PRODUCE THE 4-PAGE PDF
# ===========================================================================
base_baseline <- vas_base$baseline_year   # e.g. 1950
cond_baseline <- if (!is.null(vas_cond)) vas_cond$baseline_year else "N/A"

fname    <- "choropleth_VAS_base_vs_cond_2017.pdf"
pdf_path <- file.path(viz_dir, fname)
cat(sprintf("=== Generating: %s ===\n", fname))

pdf(pdf_path, width = 14, height = 10)

## --- Page 1: All Australia, Climate-only ---
df1 <- data.frame(REG_NAME = all_regions, value = base_vals_all, stringsAsFactors = FALSE)
draw_choropleth(df1, all_regions,
  title = sprintf("Vascular Plants - All of Australia\nClimate-only model: %s at 2017 (vs %d baseline)",
                  metric_name, base_baseline),
  vmin = vmin, vmax = vmax)
cat("  Page 1: All Australia — Climate-only\n")

## --- Page 2: All Australia, Climate + Condition ---
df2 <- data.frame(REG_NAME = all_regions, value = cond_vals_all, stringsAsFactors = FALSE)
draw_choropleth(df2, all_regions,
  title = sprintf("Vascular Plants - All of Australia\nClimate + Condition model: %s at 2017 (vs %s baseline)",
                  metric_name, cond_baseline),
  vmin = vmin, vmax = vmax)
cat("  Page 2: All Australia — Climate + Condition\n")

## --- Page 3: Eastern States, Climate-only ---
df3 <- data.frame(REG_NAME = eastern_regions, value = base_vals_east, stringsAsFactors = FALSE)
draw_choropleth(df3, eastern_regions,
  title = sprintf("Vascular Plants - Eastern States (%s)\nClimate-only model: %s at 2017 (vs %d baseline)",
                  state_label, metric_name, base_baseline),
  xlim = c(138, 154), ylim = c(-44, -10),
  vmin = vmin, vmax = vmax)
cat("  Page 3: Eastern States — Climate-only\n")

## --- Page 4: Eastern States, Climate + Condition ---
df4 <- data.frame(REG_NAME = eastern_regions, value = cond_vals_east, stringsAsFactors = FALSE)
draw_choropleth(df4, eastern_regions,
  title = sprintf("Vascular Plants - Eastern States (%s)\nClimate + Condition model: %s at 2017 (vs %s baseline)",
                  state_label, metric_name, cond_baseline),
  xlim = c(138, 154), ylim = c(-44, -10),
  vmin = vmin, vmax = vmax)
cat("  Page 4: Eastern States — Climate + Condition\n")

dev.off()
cat(sprintf("\n  Saved: %s\n", basename(pdf_path)))
cat("\n=== VAS base vs condition choropleth complete ===\n")
