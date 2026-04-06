##############################################################################
##
## plot_choropleth_temporal_change.R
##
## Choropleth maps of temporal change (dissimilarity) at the 2017 epoch,
## coloured by IBRA region using a white-to-red colour ramp.
##
## For each scope (All Australia / Eastern States) and each group set
## (all groups / sans AVES & HYM), produces a multi-page PDF containing:
##   - One map per biological group
##   - A "Median across groups" combined map
##   - A "Sum across groups" combined map
##
## Produces 4 PDFs total.
##
## Usage:
##   Rscript plot_choropleth_temporal_change.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Choropleth Maps: Temporal Change at 2017 Epoch\n")
cat("###########################################################################\n\n")

# ---------------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------------
## Always plot dissimilarity (1 - similarity) for these maps so that
## "more red" = "more change".  Set FALSE if you prefer similarity scale.
PLOT_DISSIMILARITY <- TRUE

metric_name <- if (PLOT_DISSIMILARITY) "Change in Community Composition" else "Similarity in Community Composition"
metric_tag  <- if (PLOT_DISSIMILARITY) "dissim" else "sim"

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
# Group definitions
# ---------------------------------------------------------------------------
all_groups <- c("AMP", "AVES", "HYM", "MAM", "REP", "VAS")

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
# Get 2017 value per region (mean across sites within that region)
# ---------------------------------------------------------------------------
get_2017_val <- function(group, model_type = "base", region) {
  dat <- ibra_data[[group]][[model_type]]
  if (is.null(dat)) return(NA_real_)

  all_years <- c(dat$baseline_year, dat$target_years)
  idx_2017  <- which(all_years == 2017)
  if (length(idx_2017) == 0) idx_2017 <- length(all_years)
  col_idx <- idx_2017 - 1L

  rr <- dat$region_results[[region]]
  if (is.null(rr) || all(is.na(rr$mat_sim))) return(NA_real_)
  if (col_idx < 1 || col_idx > ncol(rr$mat_sim)) return(NA_real_)

  sim <- mean(rr$mat_sim[, col_idx], na.rm = TRUE)
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
# Build a data table: value per (region × group) — base model only
# ---------------------------------------------------------------------------
cat("--- Computing 2017 values per region × group ---\n")
val_table <- expand.grid(
  REG_NAME = all_ibra_regions,
  group    = all_groups,
  stringsAsFactors = FALSE
)
val_table$value <- mapply(function(reg, grp) get_2017_val(grp, "base", reg),
                          val_table$REG_NAME, val_table$group)
cat(sprintf("  %d region × group entries, %d non-NA\n\n",
            nrow(val_table), sum(!is.na(val_table$value))))

# ---------------------------------------------------------------------------
# White-to-red colour ramp
# ---------------------------------------------------------------------------
n_cols  <- 100
wr_ramp <- colorRampPalette(c("white", "#FEE0D2", "#FC9272", "#DE2D26", "#67000D"))(n_cols)

## Map a value to a colour given a fixed range [vmin, vmax]
val_to_col <- function(v, vmin, vmax) {
  if (is.na(v)) return("grey85")
  idx <- round((v - vmin) / max(vmax - vmin, 1e-9) * (n_cols - 1)) + 1L
  idx <- max(1L, min(n_cols, idx))
  wr_ramp[idx]
}

# ===========================================================================
# CHOROPLETH DRAWING FUNCTION
# ===========================================================================
## Draws a single choropleth page.
## `values_df`: data.frame with columns REG_NAME and value
## `regions`:   character vector of regions to highlight (others are grey)
## `title`:     plot title
## `xlim`, `ylim`: optional crop
## `vmin`, `vmax`: colour scale range (shared across pages within a PDF)
draw_choropleth <- function(values_df, regions, title,
                            xlim = NULL, ylim = NULL,
                            vmin = 0, vmax = 1) {

  ## Join values onto the dissolved polygons
  plot_sf <- ibra_dissolved %>%
    left_join(values_df, by = "REG_NAME")

  ## Assign fill colours
  plot_sf$fill <- sapply(seq_len(nrow(plot_sf)), function(i) {
    reg <- plot_sf$REG_NAME[i]
    if (!(reg %in% regions)) return("grey92")
    val_to_col(plot_sf$value[i], vmin, vmax)
  })

  ## Determine extent
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

  ## Draw colour bar legend on the right
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

  ## Tick labels
  n_ticks <- 5
  tick_vals <- seq(vmin, vmax, length.out = n_ticks)
  tick_ys   <- seq(bar_y1, bar_y2, length.out = n_ticks)
  for (k in seq_len(n_ticks)) {
    segments(bar_x2, tick_ys[k], bar_x2 + (bar_x2 - bar_x1) * 0.3, tick_ys[k],
             col = "grey30", lwd = 0.5)
    text(bar_x2 + (bar_x2 - bar_x1) * 0.5, tick_ys[k],
         sprintf("%.3f", tick_vals[k]), adj = c(0, 0.5), cex = 0.65)
  }

  ## Label above the bar
  text((bar_x1 + bar_x2) / 2, bar_y2 + (usr[4] - usr[3]) * 0.03,
       metric_name, cex = 0.6, srt = 0)
  par(xpd = FALSE)
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

    fname <- sprintf("choropleth_temporal_change_2017_%s_%s_%s.pdf",
                     metric_tag, ss$scope_tag, gs$suffix)
    pdf_path <- file.path(viz_dir, fname)
    cat(sprintf("\n=== Generating: %s ===\n", fname))

    ## --- Compute a shared colour scale across all pages in this PDF ---
    ## Gather all per-group values for these regions
    all_vals <- c()
    for (grp in gs$groups) {
      for (reg in ss$regions) {
        all_vals <- c(all_vals, get_2017_val(grp, "base", reg))
      }
    }
    ## Also compute cross-group median & sum per region for scale
    for (reg in ss$regions) {
      grp_vals <- sapply(gs$groups, function(g) get_2017_val(g, "base", reg))
      grp_vals <- grp_vals[!is.na(grp_vals)]
      if (length(grp_vals) > 0) {
        all_vals <- c(all_vals, median(grp_vals), sum(grp_vals))
      }
    }
    all_vals <- all_vals[!is.na(all_vals)]

    ## Per-group pages share one scale; combined pages get their own.
    ## (sum has a very different range, so we split scales)
    vmin_grp <- 0
    vmax_grp <- if (length(all_vals) > 0) {
      max(sapply(gs$groups, function(grp) {
        gv <- sapply(ss$regions, function(r) get_2017_val(grp, "base", r))
        max(gv, na.rm = TRUE)
      }), na.rm = TRUE)
    } else 1

    ## Median scale
    med_vals <- sapply(ss$regions, function(reg) {
      gv <- sapply(gs$groups, function(g) get_2017_val(g, "base", reg))
      gv <- gv[!is.na(gv)]
      if (length(gv) > 0) median(gv) else NA_real_
    })
    vmin_med <- 0
    vmax_med <- max(med_vals, na.rm = TRUE)

    ## Sum scale
    sum_vals <- sapply(ss$regions, function(reg) {
      gv <- sapply(gs$groups, function(g) get_2017_val(g, "base", reg))
      gv <- gv[!is.na(gv)]
      if (length(gv) > 0) sum(gv) else NA_real_
    })
    vmin_sum <- 0
    vmax_sum <- max(sum_vals, na.rm = TRUE)

    ## Pad maxes slightly
    vmax_grp <- ceiling(vmax_grp * 100) / 100
    vmax_med <- ceiling(vmax_med * 100) / 100
    vmax_sum <- ceiling(vmax_sum * 100) / 100

    pdf(pdf_path, width = 14, height = 10)

    ## ---- Per-group pages ----
    for (grp in gs$groups) {
      df <- data.frame(
        REG_NAME = ss$regions,
        value    = sapply(ss$regions, function(r) get_2017_val(grp, "base", r)),
        stringsAsFactors = FALSE
      )
      ttl <- sprintf("%s -- %s\nMean Temporal %s at 2017 (vs 1950 baseline)",
                     group_full_names[grp], ss$scope_lbl, metric_name)
      draw_choropleth(df, ss$regions, ttl,
                      xlim = ss$xlim, ylim = ss$ylim,
                      vmin = vmin_grp, vmax = vmax_grp)
      cat(sprintf("  Page: %s (%d regions with data)\n",
                  grp, sum(!is.na(df$value))))
    }

    ## ---- Median across groups ----
    df_med <- data.frame(
      REG_NAME = ss$regions,
      value    = med_vals,
      stringsAsFactors = FALSE
    )
    ttl_med <- sprintf("Median across groups -- %s\nTemporal %s at 2017 (vs 1950 baseline)\nGroups: %s",
                       ss$scope_lbl, metric_name,
                       paste(group_full_names[gs$groups], collapse = ", "))
    draw_choropleth(df_med, ss$regions, ttl_med,
                    xlim = ss$xlim, ylim = ss$ylim,
                    vmin = vmin_med, vmax = vmax_med)
    cat(sprintf("  Page: Median (%d regions)\n", sum(!is.na(df_med$value))))

    ## ---- Sum across groups ----
    df_sum <- data.frame(
      REG_NAME = ss$regions,
      value    = sum_vals,
      stringsAsFactors = FALSE
    )
    ttl_sum <- sprintf("Sum across groups -- %s\nTemporal %s at 2017 (vs 1950 baseline)\nGroups: %s",
                       ss$scope_lbl, metric_name,
                       paste(group_full_names[gs$groups], collapse = ", "))
    draw_choropleth(df_sum, ss$regions, ttl_sum,
                    xlim = ss$xlim, ylim = ss$ylim,
                    vmin = vmin_sum, vmax = vmax_sum)
    cat(sprintf("  Page: Sum (%d regions)\n", sum(!is.na(df_sum$value))))

    dev.off()
    cat(sprintf("  Saved: %s\n", basename(pdf_path)))
  }
}

cat("\n=== Choropleth maps complete ===\n")
