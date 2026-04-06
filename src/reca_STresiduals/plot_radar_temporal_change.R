##############################################################################
##
## plot_radar_temporal_change.R
##
## Radar (spider) plots of temporal change at the 2017 epoch for each
## biological group.  Each axis of the radar represents one group.  The
## plotted value is the mean temporal similarity at 2017 (relative to the
## 1950 baseline, so 1.0 = no change), or dissimilarity (1 − similarity)
## if PLOT_DISSIMILARITY is set to TRUE.
##
## Produces four multi-page PDFs:
##   1) All IBRA regions, all groups
##   2) All IBRA regions, sans AVES & HYM
##   3) Eastern-state IBRA regions only, all groups
##   4) Eastern-state IBRA regions only, sans AVES & HYM
##
## Each PDF contains:
##   - One page per IBRA region (base model + condition model overlaid)
##   - A final "All of Australia" (or "Eastern States") summary page
##
## Usage:
##   Rscript plot_radar_temporal_change.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Radar Plots: Temporal Change at 2017 Epoch\n")
cat("###########################################################################\n\n")

# ---------------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------------
## Set to TRUE to plot dissimilarity (1 - similarity) instead of similarity.
## When TRUE, larger values = more change, outer ring = max observed change.
## When FALSE (default), larger values = less change, outer ring = 1.0.
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
if (!dir.exists(viz_dir)) dir.create(viz_dir, recursive = TRUE)

## Install fmsb if needed (provides radarchart)
if (!requireNamespace("fmsb", quietly = TRUE)) {
  install.packages("fmsb", repos = "https://cloud.r-project.org")
}
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
# Discover run folders and load IBRA data
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

## Load all IBRA results into a master structure:
##   ibra_data[[group]][["base"|"cond"]] -> the loaded RDS
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
# Extract 2017 similarity for a given region (or all regions for site-level)
# ---------------------------------------------------------------------------
## Returns the mean similarity at 2017 for a specific IBRA region,
## or across a set of regions (for the summary page).
## model_type: "base" or "cond"
get_2017_sim <- function(group, model_type = "base", regions = NULL) {
  dat <- ibra_data[[group]][[model_type]]
  if (is.null(dat)) return(NA_real_)

  all_years <- c(dat$baseline_year, dat$target_years)
  idx_2017  <- which(all_years == 2017)
  if (length(idx_2017) == 0) {
    ## Fall back to last year
    idx_2017 <- length(all_years)
  }
  ## For IBRA data the baseline (year index 1) has similarity = 1.0,

  ## and target years are columns in mat_sim.  mat_sim is sites × target_years
  ## (without the baseline), so index is shifted by -1.
  col_idx <- idx_2017 - 1L  # column in mat_sim

  if (is.null(regions)) {
    ## Use ALL regions in the dataset
    regions <- names(dat$region_results)
  }

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

## For a single region, get the mean similarity at 2017
get_2017_sim_region <- function(group, model_type = "base", region) {
  get_2017_sim(group, model_type, regions = region)
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

cat(sprintf("  Total IBRA regions found: %d\n", length(all_ibra_regions)))
cat(sprintf("  Eastern-state regions:    %d\n\n", length(eastern_regions)))

# ===========================================================================
# RADAR PLOT FUNCTION
# ===========================================================================
## Draw a single radar chart for one IBRA (or summary).
## `groups_to_plot`: character vector of group codes to include
## `region`: single region name, or NULL for summary (all supplied regions)
## `summary_regions`: vector of region names for the summary page
## `title`: plot title
draw_radar <- function(groups_to_plot, region = NULL,
                       summary_regions = NULL, title = "") {

  labels <- unname(group_full_names[groups_to_plot])
  n_ax   <- length(groups_to_plot)

  ## Gather base & cond values
  base_vals <- sapply(groups_to_plot, function(grp) {
    if (!is.null(region)) {
      get_2017_sim_region(grp, "base", region)
    } else {
      get_2017_sim(grp, "base", summary_regions)
    }
  })

  cond_vals <- sapply(groups_to_plot, function(grp) {
    if (!is.null(region)) {
      get_2017_sim_region(grp, "cond", region)
    } else {
      get_2017_sim(grp, "cond", summary_regions)
    }
  })

  has_cond <- !all(is.na(cond_vals))

  ## Determine axis limits
  all_vals <- c(base_vals, cond_vals)
  all_vals <- all_vals[!is.na(all_vals)]
  if (length(all_vals) == 0) {
    cat(sprintf("    [SKIP] %s -- no data\n", title))
    return(invisible(NULL))
  }

  if (PLOT_DISSIMILARITY) {
    ## Dissimilarity: 0 = no change, higher = more change.  Outer ring = max observed.
    y_min <- 0
    y_max <- max(0.01, ceiling(max(all_vals) * 20) / 20 + 0.05)  # round up with padding
  } else {
    ## Similarity: 1 = no change, lower = more change.  Outer ring = 1.0.
    y_max <- 1.0
    y_min <- max(0, floor(min(all_vals) * 20) / 20 - 0.05)
  }

  ## fmsb::radarchart needs a data.frame where row 1 = max, row 2 = min, row 3+ = data
  ## Replace NAs with y_min so the polygon doesn't break
  base_clean <- ifelse(is.na(base_vals), y_min, base_vals)
  cond_clean <- ifelse(is.na(cond_vals), y_min, cond_vals)

  if (has_cond) {
    df <- as.data.frame(rbind(
      rep(y_max, n_ax),
      rep(y_min, n_ax),
      base_clean,
      cond_clean
    ))
  } else {
    df <- as.data.frame(rbind(
      rep(y_max, n_ax),
      rep(y_min, n_ax),
      base_clean
    ))
  }
  colnames(df) <- labels

  ## Colours
  base_col <- adjustcolor("#2166AC", alpha.f = 0.35)  # blue fill
  cond_col <- adjustcolor("#B2182B", alpha.f = 0.25)  # red fill
  base_bdr <- "#2166AC"
  cond_bdr <- "#B2182B"

  pcols <- if (has_cond) c(base_bdr, cond_bdr) else base_bdr
  pfcols <- if (has_cond) c(base_col, cond_col) else base_col
  plwd  <- if (has_cond) c(2.5, 2.5) else 2.5
  plty  <- if (has_cond) c(1, 2) else 1

  par(mar = c(2, 2, 4, 2))
  radarchart(df,
             axistype  = 1,
             pcol      = pcols,
             pfcol     = pfcols,
             plwd      = plwd,
             plty      = plty,
             cglcol    = "grey70",
             cglty     = 1,
             cglwd     = 0.8,
             axislabcol = "grey40",
             vlcex     = 1.0,
             calcex    = 0.8,
             title     = title,
             cex.main  = 1.0)

  ## Add data-value labels at each vertex for the base model
  ## Calculate vertex positions (evenly spaced angles starting from top)
  angles <- seq(pi/2, pi/2 + 2*pi, length.out = n_ax + 1)[1:n_ax]
  for (i in seq_len(n_ax)) {
    if (!is.na(base_vals[i])) {
      ## Normalise to 0-1 within [y_min, y_max] for radial positioning
      r <- (base_vals[i] - y_min) / (y_max - y_min)
      x <- r * cos(angles[i])
      y <- r * sin(angles[i])
      text(x, y, sprintf("%.3f", base_vals[i]),
           cex = 0.7, col = base_bdr, pos = 3, offset = 0.3)
    }
  }

  ## Legend
  if (has_cond) {
    legend("bottomright",
           legend = c("Climate-only model", "Climate + Condition model"),
           col = c(base_bdr, cond_bdr), lwd = 2.5, lty = c(1, 2),
           fill = c(base_col, cond_col), border = c(base_bdr, cond_bdr),
           cex = 0.75, bg = "white")
  } else {
    legend("bottomright",
           legend = "Climate-only model",
           col = base_bdr, lwd = 2.5, lty = 1,
           fill = base_col, border = base_bdr,
           cex = 0.75, bg = "white")
  }
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

scope_sets <- list(
  list(regions   = all_ibra_regions,
       scope_tag = "all_australia",
       scope_lbl = "All of Australia"),
  list(regions   = eastern_regions,
       scope_tag = "eastern_states",
       scope_lbl = sprintf("Eastern States (%s)", state_label))
)

for (gs in group_sets) {
  for (ss in scope_sets) {

    fname <- sprintf("radar_temporal_change_2017_%s_%s_%s.pdf", metric_tag, ss$scope_tag, gs$suffix)
    pdf_path <- file.path(viz_dir, fname)
    cat(sprintf("=== Generating: %s ===\n", fname))

    pdf(pdf_path, width = 9, height = 9)

    n_plotted <- 0L

    ## ---- One page per IBRA region ----
    for (reg in ss$regions) {
      reg_states <- paste(ibra_state_map[[reg]], collapse = ", ")
      ttl <- sprintf("%s (%s)\nMean Temporal %s at 2017 (vs 1950 baseline)",
                     reg, reg_states, metric_name)
      draw_radar(groups_to_plot  = gs$groups,
                 region          = reg,
                 title           = ttl)
      n_plotted <- n_plotted + 1L
    }

    ## ---- Summary page: all regions aggregated ----
    ttl <- sprintf("%s\nMean Temporal %s at 2017 (vs 1950 baseline)",
                   ss$scope_lbl, metric_name)
    draw_radar(groups_to_plot  = gs$groups,
               summary_regions = ss$regions,
               title           = ttl)
    n_plotted <- n_plotted + 1L

    dev.off()
    cat(sprintf("  -> %d pages, saved: %s\n\n", n_plotted, basename(pdf_path)))
  }
}

cat("=== Radar plots complete ===\n")
