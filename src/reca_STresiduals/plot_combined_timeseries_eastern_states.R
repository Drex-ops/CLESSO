##############################################################################
##
## plot_combined_timeseries_eastern_states.R
##
## State-filtered version of plot_combined_timeseries.R.
## Only includes IBRA regions that intersect with VIC, ACT, QLD, NSW, or TAS.
##
## Produces:
##   A) Site-level combined timeseries (one page per group + one summary)
##      — aggregated from IBRA data, filtered to eastern-state regions only
##   B) IBRA-level combined timeseries per group (filtered to eastern states)
##   C) IBRA-level all-groups summary (filtered to eastern states)
##   D) JSON exports:
##      - timeseries_site_level_eastern_states.json  (filtered)
##      - timeseries_ibra_eastern_states.json         (filtered)
##
## Usage:
##   Rscript plot_combined_timeseries_eastern_states.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Combined Timeseries: Eastern States (VIC, ACT, QLD, NSW, TAS)\n")
cat("###########################################################################\n\n")

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

# ---------------------------------------------------------------------------
# 0. IBRA → State mapping and filter
# ---------------------------------------------------------------------------
# Each IBRA region is mapped to ALL states/territories whose boundaries it
# intersects.  A region is included if it overlaps ANY of the target states.
# Source: IBRA7 spatial join (same as generate_combined_temporal_change.py)

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

## Filter function: TRUE if region overlaps any target state
region_in_states <- function(region_name, states = target_states) {
  mapped <- ibra_state_map[[region_name]]
  if (is.null(mapped)) return(FALSE)
  any(mapped %in% states)
}

cat(sprintf("  Target states: %s\n", state_label))
cat(sprintf("  IBRA regions mapped: %d total, %d in target states\n\n",
            length(ibra_state_map),
            sum(sapply(names(ibra_state_map), region_in_states))))

# ---------------------------------------------------------------------------
# 1. Discover run folders
# ---------------------------------------------------------------------------
groups <- c("AMP", "AVES", "HYM", "MAM", "REP", "VAS")
prefix <- "eastern_states"

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

registry <- list()
for (grp in groups) {
  base_dir <- find_run_folder(grp, "base")
  cond_dir <- find_run_folder(grp, "COND")

  base_ibra <- if (!is.null(base_dir)) file.path(base_dir, paste0(grp, "_ibra_timeseries_results.rds")) else NULL
  cond_ibra <- if (!is.null(cond_dir)) file.path(cond_dir, paste0(grp, "_ibra_timeseries_results.rds")) else NULL

  registry[[grp]] <- list(
    base_ibra = if (!is.null(base_ibra) && file.exists(base_ibra)) base_ibra else NULL,
    cond_ibra = if (!is.null(cond_ibra) && file.exists(cond_ibra)) cond_ibra else NULL
  )
}

cat("--- Available data ---\n")
for (grp in groups) {
  r <- registry[[grp]]
  cat(sprintf("  %-5s  base_ibra=%s  cond_ibra=%s\n",
              grp,
              if (!is.null(r$base_ibra)) "YES" else "---",
              if (!is.null(r$cond_ibra)) "YES" else "---"))
}
cat("\n")

# ---------------------------------------------------------------------------
# 2. Colour palette and group names
# ---------------------------------------------------------------------------
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
  HYM  = "Hymenoptera (ants, bees, wasps)",
  MAM  = "Mammals",
  REP  = "Reptiles",
  VAS  = "Vascular plants"
)

# ===========================================================================
# A) SITE-LEVEL COMBINED TIMESERIES (eastern states only, from IBRA data)
# ===========================================================================
cat(sprintf("=== A) Site-level combined timeseries (filtered: %s) ===\n\n", state_label))

pdf_file <- file.path(viz_dir, sprintf("combined_timeseries_base_vs_cond_%s.pdf", prefix))
pdf(pdf_file, width = 14, height = 8)

summary_data <- list()

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ibra)) {
    cat(sprintf("  [SKIP] %s -- no base IBRA data\n", grp))
    next
  }

  ## --- Aggregate base model from filtered IBRA regions ---
  base_ibra_dat <- readRDS(r$base_ibra)
  base_years    <- c(base_ibra_dat$baseline_year, base_ibra_dat$target_years)
  n_target_yrs  <- length(base_ibra_dat$target_years)

  ## Concatenate mat_sim from all eastern-state regions
  base_mats <- list()
  for (reg in names(base_ibra_dat$region_results)) {
    if (!region_in_states(reg)) next
    br <- base_ibra_dat$region_results[[reg]]
    if (is.null(br) || all(is.na(br$mat_sim))) next
    base_mats[[length(base_mats) + 1L]] <- br$mat_sim
  }
  if (length(base_mats) == 0) {
    cat(sprintf("  [SKIP] %s -- no data in eastern-state IBRA regions\n", grp))
    next
  }
  base_mat_all  <- do.call(rbind, base_mats)          # sites × years
  base_mean_sim <- colMeans(base_mat_all, na.rm = TRUE)
  base_sim      <- c(1.0, base_mean_sim)
  n_base_sites  <- nrow(base_mat_all)

  ## --- Aggregate condition model from filtered IBRA regions ---
  has_cond <- !is.null(r$cond_ibra)
  cond_years_raw    <- NULL
  cond_sim_offset   <- NULL
  cond_mat_all      <- NULL
  n_cond_sites      <- 0L
  cond_baseline_yr  <- NULL

  if (has_cond) {
    cond_ibra_dat <- readRDS(r$cond_ibra)
    cond_years_raw   <- c(cond_ibra_dat$baseline_year, cond_ibra_dat$target_years)
    cond_baseline_yr <- cond_ibra_dat$baseline_year

    cond_mats <- list()
    for (reg in names(cond_ibra_dat$region_results)) {
      if (!region_in_states(reg)) next
      cr <- cond_ibra_dat$region_results[[reg]]
      if (is.null(cr) || all(is.na(cr$mat_sim))) next
      cond_mats[[length(cond_mats) + 1L]] <- cr$mat_sim
    }
    if (length(cond_mats) > 0) {
      cond_mat_all     <- do.call(rbind, cond_mats)
      cond_mean_sim    <- colMeans(cond_mat_all, na.rm = TRUE)
      cond_sim_raw     <- c(1.0, cond_mean_sim)
      n_cond_sites     <- nrow(cond_mat_all)

      ## Align condition to base at the condition baseline year
      align_year <- cond_baseline_yr
      align_idx  <- which.min(abs(base_years - align_year))
      base_val_at_align <- base_sim[align_idx]
      offset <- base_val_at_align - 1.0
      cond_sim_offset <- cond_sim_raw + offset
    } else {
      has_cond <- FALSE
    }
  }

  summary_data[[grp]] <- list(
    base_years = base_years,
    base_sim   = base_sim,
    has_cond   = has_cond,
    cond_years = cond_years_raw,
    cond_sim   = cond_sim_offset
  )

  ## ---- Per-group plot ----
  par(mar = c(5, 5, 4, 2))

  y_vals <- base_sim
  if (has_cond) y_vals <- c(y_vals, cond_sim_offset)
  y_range <- range(y_vals, na.rm = TRUE)
  pad <- diff(y_range) * 0.05
  y_range <- y_range + c(-pad, pad)

  plot(base_years, base_sim, type = "l", lwd = 3,
       col = group_colours[grp],
       xlim = range(base_years), ylim = y_range,
       xlab = "Year", ylab = "Mean Temporal Similarity",
       main = sprintf("%s -- Mean Temporal Similarity (%s)\nClimate-only model (1950 baseline) vs Climate + Condition model (aligned at %s)",
                      group_full_names[grp], state_label,
                      if (has_cond) as.character(cond_baseline_yr) else "N/A"),
       cex.main = 1.0)

  base_sd <- apply(base_mat_all, 2, sd, na.rm = TRUE)
  base_sd_full <- c(0, base_sd)
  polygon(c(base_years, rev(base_years)),
          c(base_sim + base_sd_full, rev(pmax(0, base_sim - base_sd_full))),
          col = adjustcolor(group_colours[grp], alpha.f = 0.1), border = NA)

  if (has_cond) {
    lines(cond_years_raw, cond_sim_offset, lwd = 3, lty = 2,
          col = adjustcolor(group_colours[grp], alpha.f = 0.7))
    cond_sd <- apply(cond_mat_all, 2, sd, na.rm = TRUE)
    cond_sd_full <- c(0, cond_sd)
    polygon(c(cond_years_raw, rev(cond_years_raw)),
            c(cond_sim_offset + cond_sd_full, rev(pmax(0, cond_sim_offset - cond_sd_full))),
            col = adjustcolor(group_colours[grp], alpha.f = 0.08), border = NA)
    abline(v = align_year, lty = 3, col = "grey50")
    points(align_year, base_val_at_align, pch = 16, cex = 1.5, col = "grey30")

    legend("bottomleft",
           legend = c(sprintf("Climate-only model (1950 baseline, %d sites)", n_base_sites),
                      sprintf("Climate + Condition model (aligned at %d, %d sites)",
                              cond_baseline_yr, n_cond_sites),
                      "Standard deviation", "Alignment point"),
           col = c(group_colours[grp],
                   adjustcolor(group_colours[grp], alpha.f = 0.7),
                   adjustcolor(group_colours[grp], alpha.f = 0.2),
                   "grey30"),
           lwd = c(3, 3, 8, NA), lty = c(1, 2, 1, NA),
           pch = c(NA, NA, NA, 16),
           cex = 0.85, bg = "white")
  } else {
    legend("bottomleft",
           legend = c(sprintf("Climate-only model (1950 baseline, %d sites)", n_base_sites),
                      "Climate + Condition model: not available"),
           col = c(group_colours[grp], "grey70"),
           lwd = c(3, 1), lty = c(1, 3), cex = 0.85, bg = "white")
  }

  cat(sprintf("  %s: %d sites from eastern-state IBRA regions (cond=%s)\n",
              grp, n_base_sites, if (has_cond) "YES" else "NO"))
}

## ---- Summary: all groups on one plot ----
par(mar = c(5, 5, 4, 2))
all_y <- unlist(lapply(summary_data, function(d) c(d$base_sim, d$cond_sim)))
y_range <- range(all_y, na.rm = TRUE)
pad <- diff(y_range) * 0.05
y_range <- y_range + c(-pad, pad)

plot(NA, xlim = c(1950, 2017), ylim = y_range,
     xlab = "Year", ylab = "Mean Temporal Similarity",
     main = sprintf("All Groups -- Mean Temporal Similarity (%s)\nClimate-only model (solid) vs Climate + Condition model (dashed)",
                    state_label),
     cex.main = 1.0)

leg_labels <- character()
leg_cols   <- character()
leg_lty    <- integer()

for (grp in names(summary_data)) {
  d <- summary_data[[grp]]
  lines(d$base_years, d$base_sim, col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5)
  leg_labels <- c(leg_labels, sprintf("%s (climate-only)", group_full_names[grp]))
  leg_cols   <- c(leg_cols, group_colours[grp])
  leg_lty    <- c(leg_lty, 1L)

  if (d$has_cond) {
    lines(d$cond_years, d$cond_sim, col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5, lty = 2)
    leg_labels <- c(leg_labels, sprintf("%s (climate + condition)", group_full_names[grp]))
    leg_cols   <- c(leg_cols, group_colours[grp])
    leg_lty    <- c(leg_lty, 2L)
  }
}

## Mean of all base timeseries
common_years <- seq(1950, 2017)
base_matrix  <- matrix(NA_real_, nrow = length(summary_data), ncol = length(common_years))
for (i in seq_along(summary_data)) {
  d <- summary_data[[i]]
  base_matrix[i, ] <- approx(d$base_years, d$base_sim,
                              xout = common_years, rule = 1)$y
}
mean_base_sim <- colMeans(base_matrix, na.rm = TRUE)
sd_base_sim   <- apply(base_matrix, 2, sd, na.rm = TRUE)

polygon(c(common_years, rev(common_years)),
        c(mean_base_sim + sd_base_sim, rev(pmax(0, mean_base_sim - sd_base_sim))),
        col = adjustcolor("black", alpha.f = 0.10), border = NA)
lines(common_years, mean_base_sim, col = "black", lwd = 4)
leg_labels <- c(leg_labels, "Mean across all groups (climate-only)")
leg_cols   <- c(leg_cols, "black")
leg_lty    <- c(leg_lty, 1L)

## Mean of all COND timeseries
cond_groups <- Filter(function(d) d$has_cond, summary_data)
if (length(cond_groups) > 0) {
  cond_min_yr <- min(sapply(cond_groups, function(d) min(d$cond_years)))
  cond_max_yr <- max(sapply(cond_groups, function(d) max(d$cond_years)))
  common_cond_years <- seq(cond_min_yr, cond_max_yr)

  cond_matrix <- matrix(NA_real_, nrow = length(cond_groups), ncol = length(common_cond_years))
  for (i in seq_along(cond_groups)) {
    d <- cond_groups[[i]]
    cond_matrix[i, ] <- approx(d$cond_years, d$cond_sim,
                                xout = common_cond_years, rule = 1)$y
  }
  mean_cond_sim <- colMeans(cond_matrix, na.rm = TRUE)
  sd_cond_sim   <- apply(cond_matrix, 2, sd, na.rm = TRUE)

  polygon(c(common_cond_years, rev(common_cond_years)),
          c(mean_cond_sim + sd_cond_sim, rev(pmax(0, mean_cond_sim - sd_cond_sim))),
          col = adjustcolor("black", alpha.f = 0.06), border = NA)
  lines(common_cond_years, mean_cond_sim, col = "black", lwd = 4, lty = 2)
  leg_labels <- c(leg_labels, "Mean across all groups (climate + condition)")
  leg_cols   <- c(leg_cols, "black")
  leg_lty    <- c(leg_lty, 2L)
}

## Alignment marker at 2000
abline(v = 2000, lty = 3, col = "grey50")
idx_2000 <- which(common_years == 2000)
if (length(idx_2000) == 1) {
  points(2000, mean_base_sim[idx_2000], pch = 16, cex = 1.5, col = "grey30")
}

legend("bottomleft", legend = leg_labels, col = leg_cols,
       lwd = ifelse(leg_cols == "black", 4, 2.5),
       lty = leg_lty, cex = 0.7, bg = "white", ncol = 2)

dev.off()
cat(sprintf("\n  Saved: %s\n\n", basename(pdf_file)))


# ===========================================================================
# B) IBRA-LEVEL COMBINED TIMESERIES — EASTERN STATES ONLY
# ===========================================================================
cat(sprintf("=== B) IBRA-level combined timeseries (filtered: %s) ===\n\n", state_label))

pdf_ibra <- file.path(viz_dir, sprintf("combined_timeseries_IBRA_base_vs_cond_%s.pdf", prefix))
pdf(pdf_ibra, width = 14, height = 8)

## ---- Pre-compute per-group ylim across eastern-state IBRA regions ----
## Consistent within each group; different between groups for max variation.
group_ylims <- list()
for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ibra)) next
  tmp_base <- readRDS(r$base_ibra)
  tmp_base_years <- c(tmp_base$baseline_year, tmp_base$target_years)
  tmp_cond <- NULL
  if (!is.null(r$cond_ibra)) tmp_cond <- readRDS(r$cond_ibra)

  all_y_grp <- c()
  for (reg in names(tmp_base$region_results)) {
    if (!region_in_states(reg)) next
    br <- tmp_base$region_results[[reg]]
    if (is.null(br) || all(is.na(br$mat_sim))) next
    base_mean_tmp <- c(1.0, colMeans(br$mat_sim, na.rm = TRUE))
    all_y_grp <- c(all_y_grp, base_mean_tmp)
    if (!is.null(tmp_cond) && reg %in% names(tmp_cond$region_results)) {
      cr <- tmp_cond$region_results[[reg]]
      if (!is.null(cr) && !all(is.na(cr$mat_sim))) {
        cond_mean_tmp <- c(1.0, colMeans(cr$mat_sim, na.rm = TRUE))
        align_idx_tmp <- which.min(abs(tmp_base_years - tmp_cond$baseline_year))
        offset_tmp <- base_mean_tmp[align_idx_tmp] - 1.0
        all_y_grp <- c(all_y_grp, cond_mean_tmp + offset_tmp)
      }
    }
  }
  yr <- range(all_y_grp, na.rm = TRUE)
  pad <- diff(yr) * 0.05
  group_ylims[[grp]] <- yr + c(-pad, pad)
  cat(sprintf("  %s ylim: [%.4f, %.4f]\n", grp, group_ylims[[grp]][1], group_ylims[[grp]][2]))
}
cat("\n")

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ibra)) {
    cat(sprintf("  [SKIP] %s -- no base IBRA timeseries\n", grp))
    next
  }

  base_ibra <- readRDS(r$base_ibra)
  base_target_years <- base_ibra$target_years
  base_all_years    <- c(base_ibra$baseline_year, base_target_years)

  has_cond_ibra <- !is.null(r$cond_ibra)
  cond_ibra <- NULL
  if (has_cond_ibra) {
    cond_ibra <- readRDS(r$cond_ibra)
    cond_target_years <- cond_ibra$target_years
    cond_all_years    <- c(cond_ibra$baseline_year, cond_target_years)
  }

  ## Filter to eastern-state IBRA regions
  base_regions <- names(base_ibra$region_results)
  base_regions <- base_regions[sapply(base_regions, region_in_states)]

  ## Use the pre-computed per-group ylim
  grp_ylim <- group_ylims[[grp]]

  n_plotted <- 0L
  for (reg in base_regions) {
    br <- base_ibra$region_results[[reg]]
    if (is.null(br) || all(is.na(br$mat_sim))) next

    base_mean <- c(1.0, colMeans(br$mat_sim, na.rm = TRUE))
    base_sd   <- c(0,   apply(br$mat_sim, 2, sd, na.rm = TRUE))

    cond_plotted <- FALSE
    if (has_cond_ibra && reg %in% names(cond_ibra$region_results)) {
      cr <- cond_ibra$region_results[[reg]]
      if (!is.null(cr) && !all(is.na(cr$mat_sim))) {
        cond_mean_raw <- c(1.0, colMeans(cr$mat_sim, na.rm = TRUE))
        cond_sd_raw   <- c(0,   apply(cr$mat_sim, 2, sd, na.rm = TRUE))
        align_year <- cond_ibra$baseline_year
        align_idx  <- which.min(abs(base_all_years - align_year))
        base_val   <- base_mean[align_idx]
        offset     <- base_val - 1.0
        cond_mean_offset <- cond_mean_raw + offset
        cond_plotted <- TRUE
      }
    }

    ## States label for this region
    reg_states <- paste(ibra_state_map[[reg]], collapse = ", ")

    par(mar = c(5, 5, 4, 2))
    plot(base_all_years, base_mean, type = "l", lwd = 3,
         col = group_colours[grp],
         xlim = range(base_all_years), ylim = grp_ylim,
         xlab = "Year", ylab = "Mean Temporal Similarity",
         main = sprintf("%s -- %s (%s)\nClimate-only vs Climate + Condition | %d sites",
                        group_full_names[grp], reg, reg_states, br$n_sites),
         cex.main = 0.95)

    polygon(c(base_all_years, rev(base_all_years)),
            c(base_mean + base_sd, rev(pmax(0, base_mean - base_sd))),
            col = adjustcolor(group_colours[grp], alpha.f = 0.1), border = NA)

    if (cond_plotted) {
      lines(cond_all_years, cond_mean_offset, lwd = 3, lty = 2,
            col = adjustcolor(group_colours[grp], alpha.f = 0.7))
      polygon(c(cond_all_years, rev(cond_all_years)),
              c(cond_mean_offset + cond_sd_raw, rev(pmax(0, cond_mean_offset - cond_sd_raw))),
              col = adjustcolor(group_colours[grp], alpha.f = 0.08), border = NA)
      abline(v = align_year, lty = 3, col = "grey50")
    }

    n_plotted <- n_plotted + 1L
  }

  cat(sprintf("  %s: %d eastern-state IBRA regions plotted (cond=%s)\n",
              grp, n_plotted, if (has_cond_ibra) "YES" else "NO"))
}

dev.off()
cat(sprintf("\n  Saved: %s\n\n", basename(pdf_ibra)))


# ===========================================================================
# C) IBRA-LEVEL ALL-GROUPS SUMMARY — EASTERN STATES ONLY
# ===========================================================================
cat(sprintf("=== C) IBRA all-groups summary (filtered: %s) ===\n\n", state_label))

## Collect all IBRA data (filtered to eastern states)
ibra_all_data <- list()

for (grp in groups) {
  r <- registry[[grp]]
  if (is.null(r$base_ibra)) next

  base_ibra <- readRDS(r$base_ibra)
  base_all_years <- c(base_ibra$baseline_year, base_ibra$target_years)

  has_cond_ibra <- !is.null(r$cond_ibra)
  cond_ibra <- NULL
  if (has_cond_ibra) {
    cond_ibra <- readRDS(r$cond_ibra)
    cond_all_years <- c(cond_ibra$baseline_year, cond_ibra$target_years)
  }

  for (reg in names(base_ibra$region_results)) {
    ## FILTER: only eastern-state regions
    if (!region_in_states(reg)) next

    br <- base_ibra$region_results[[reg]]
    if (is.null(br) || all(is.na(br$mat_sim))) next

    base_mean <- c(1.0, colMeans(br$mat_sim, na.rm = TRUE))

    entry <- list(
      base_years = base_all_years,
      base_sim   = base_mean,
      n_sites    = br$n_sites,
      has_cond   = FALSE,
      cond_years = NULL,
      cond_sim   = NULL
    )

    if (has_cond_ibra && reg %in% names(cond_ibra$region_results)) {
      cr <- cond_ibra$region_results[[reg]]
      if (!is.null(cr) && !all(is.na(cr$mat_sim))) {
        cond_mean_raw <- c(1.0, colMeans(cr$mat_sim, na.rm = TRUE))
        align_year <- cond_ibra$baseline_year
        align_idx  <- which.min(abs(base_all_years - align_year))
        offset     <- base_mean[align_idx] - 1.0
        entry$has_cond   <- TRUE
        entry$cond_years <- cond_all_years
        entry$cond_sim   <- cond_mean_raw + offset
      }
    }

    if (is.null(ibra_all_data[[reg]])) ibra_all_data[[reg]] <- list()
    ibra_all_data[[reg]][[grp]] <- entry
  }
}

all_ibra_regions <- sort(names(ibra_all_data))
cat(sprintf("  Found %d IBRA regions in eastern states\n", length(all_ibra_regions)))

## ---- Compute GLOBAL y-range across all IBRA regions and groups ----
global_y_vals <- unlist(lapply(ibra_all_data, function(reg_data) {
  unlist(lapply(reg_data, function(d) c(d$base_sim, d$cond_sim)))
}))
global_y_range <- range(global_y_vals, na.rm = TRUE)
global_pad <- diff(global_y_range) * 0.05
global_y_range <- global_y_range + c(-global_pad, global_pad)
cat(sprintf("  Global y-range for IBRA plots: [%.4f, %.4f]\n",
            global_y_range[1], global_y_range[2]))

pdf_ibra_allgrp <- file.path(viz_dir, sprintf("combined_timeseries_IBRA_all_groups_%s.pdf", prefix))
pdf(pdf_ibra_allgrp, width = 14, height = 8)

for (reg in all_ibra_regions) {
  reg_data <- ibra_all_data[[reg]]
  if (length(reg_data) == 0) next

  ## Use global y range so all IBRA pages are comparable
  y_range <- global_y_range

  reg_states <- paste(ibra_state_map[[reg]], collapse = ", ")

  par(mar = c(5, 5, 4, 2))
  plot(NA, xlim = c(1950, 2017), ylim = y_range,
       xlab = "Year", ylab = "Mean Temporal Similarity",
       main = sprintf("%s (%s) -- All Groups\nClimate-only (solid) vs Climate + Condition (dashed)",
                      reg, reg_states),
       cex.main = 1.0)

  leg_labels <- character()
  leg_cols   <- character()
  leg_lty    <- integer()

  for (grp in names(reg_data)) {
    d <- reg_data[[grp]]
    lines(d$base_years, d$base_sim,
          col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5)
    leg_labels <- c(leg_labels, sprintf("%s (climate-only)", group_full_names[grp]))
    leg_cols   <- c(leg_cols, group_colours[grp])
    leg_lty    <- c(leg_lty, 1L)

    if (d$has_cond) {
      lines(d$cond_years, d$cond_sim,
            col = adjustcolor(group_colours[grp], alpha.f = 0.7), lwd = 1.5, lty = 2)
      leg_labels <- c(leg_labels, sprintf("%s (climate + condition)", group_full_names[grp]))
      leg_cols   <- c(leg_cols, group_colours[grp])
      leg_lty    <- c(leg_lty, 2L)
    }
  }

  ## Mean base line
  common_years <- seq(1950, 2017)
  base_mat <- matrix(NA_real_, nrow = length(reg_data), ncol = length(common_years))
  for (i in seq_along(reg_data)) {
    d <- reg_data[[i]]
    base_mat[i, ] <- approx(d$base_years, d$base_sim,
                             xout = common_years, rule = 1)$y
  }
  mean_base <- colMeans(base_mat, na.rm = TRUE)
  sd_base   <- apply(base_mat, 2, sd, na.rm = TRUE)

  polygon(c(common_years, rev(common_years)),
          c(mean_base + sd_base, rev(pmax(0, mean_base - sd_base))),
          col = adjustcolor("black", alpha.f = 0.10), border = NA)
  lines(common_years, mean_base, col = "black", lwd = 4)
  leg_labels <- c(leg_labels, "Mean across all groups (climate-only)")
  leg_cols   <- c(leg_cols, "black")
  leg_lty    <- c(leg_lty, 1L)

  ## Mean COND line
  cond_entries <- Filter(function(d) d$has_cond, reg_data)
  if (length(cond_entries) > 0) {
    cond_min <- min(sapply(cond_entries, function(d) min(d$cond_years)))
    cond_max <- max(sapply(cond_entries, function(d) max(d$cond_years)))
    common_cond <- seq(cond_min, cond_max)
    cond_mat <- matrix(NA_real_, nrow = length(cond_entries), ncol = length(common_cond))
    for (i in seq_along(cond_entries)) {
      d <- cond_entries[[i]]
      cond_mat[i, ] <- approx(d$cond_years, d$cond_sim,
                               xout = common_cond, rule = 1)$y
    }
    mean_cond <- colMeans(cond_mat, na.rm = TRUE)
    sd_cond   <- apply(cond_mat, 2, sd, na.rm = TRUE)

    polygon(c(common_cond, rev(common_cond)),
            c(mean_cond + sd_cond, rev(pmax(0, mean_cond - sd_cond))),
            col = adjustcolor("black", alpha.f = 0.06), border = NA)
    lines(common_cond, mean_cond, col = "black", lwd = 4, lty = 2)
    leg_labels <- c(leg_labels, "Mean across all groups (climate + condition)")
    leg_cols   <- c(leg_cols, "black")
    leg_lty    <- c(leg_lty, 2L)
  }

  ## Alignment marker
  abline(v = 2000, lty = 3, col = "grey50")
  idx_2000 <- which(common_years == 2000)
  if (length(idx_2000) == 1) {
    points(2000, mean_base[idx_2000], pch = 16, cex = 1.5, col = "grey30")
  }

  legend("bottomleft", legend = leg_labels, col = leg_cols,
         lwd = ifelse(leg_cols == "black", 4, 2.5),
         lty = leg_lty, cex = 0.65, bg = "white", ncol = 2)
}

dev.off()
cat(sprintf("  Saved: %s\n\n", basename(pdf_ibra_allgrp)))


# ===========================================================================
# D) EXPORT TIMESERIES AS JSON
# ===========================================================================
cat("=== D) Exporting timeseries as JSON ===\n\n")

library(jsonlite)

## --- D1. Site-level timeseries JSON (eastern states only) ---
aus_json_data <- list()
for (grp in names(summary_data)) {
  d <- summary_data[[grp]]
  entry <- list(
    group      = grp,
    group_name = unname(group_full_names[grp]),
    base = list(
      years      = d$base_years,
      similarity = round(d$base_sim, 6)
    )
  )
  if (d$has_cond) {
    entry$condition <- list(
      years      = d$cond_years,
      similarity = round(d$cond_sim, 6)
    )
  }
  aus_json_data[[grp]] <- entry
}

## Cross-group mean
common_years_out <- seq(1950, 2017)
aus_json_data[["_mean"]] <- list(
  group      = "_mean",
  group_name = "Mean across all groups",
  base = list(
    years      = common_years_out,
    similarity = round(colMeans(
      do.call(rbind, lapply(summary_data, function(d)
        approx(d$base_years, d$base_sim, xout = common_years_out, rule = 1)$y
      )), na.rm = TRUE), 6)
  )
)
cond_grps <- Filter(function(d) d$has_cond, summary_data)
if (length(cond_grps) > 0) {
  cond_min <- min(sapply(cond_grps, function(d) min(d$cond_years)))
  cond_max <- max(sapply(cond_grps, function(d) max(d$cond_years)))
  cond_yrs <- seq(cond_min, cond_max)
  aus_json_data[["_mean"]]$condition <- list(
    years      = cond_yrs,
    similarity = round(colMeans(
      do.call(rbind, lapply(cond_grps, function(d)
        approx(d$cond_years, d$cond_sim, xout = cond_yrs, rule = 1)$y
      )), na.rm = TRUE), 6)
  )
}

aus_json_path <- file.path(viz_dir, sprintf("timeseries_site_level_%s.json", prefix))
writeLines(toJSON(aus_json_data, pretty = TRUE, auto_unbox = TRUE), aus_json_path)
cat(sprintf("  Saved: %s\n", basename(aus_json_path)))

## --- D2. IBRA-level timeseries JSON (eastern states only) ---
ibra_json_data <- list()
for (reg in sort(names(ibra_all_data))) {
  reg_data <- ibra_all_data[[reg]]
  reg_states <- paste(ibra_state_map[[reg]], collapse = ", ")
  reg_entry <- list(region = reg, states = reg_states, groups = list())

  for (grp in names(reg_data)) {
    d <- reg_data[[grp]]
    g_entry <- list(
      group      = grp,
      group_name = unname(group_full_names[grp]),
      n_sites    = d$n_sites,
      base = list(
        years      = d$base_years,
        similarity = round(d$base_sim, 6)
      )
    )
    if (d$has_cond) {
      g_entry$condition <- list(
        years      = d$cond_years,
        similarity = round(d$cond_sim, 6)
      )
    }
    reg_entry$groups[[grp]] <- g_entry
  }

  ## Cross-group mean
  common_yrs <- seq(1950, 2017)
  base_vals <- do.call(rbind, lapply(reg_data, function(d)
    approx(d$base_years, d$base_sim, xout = common_yrs, rule = 1)$y
  ))
  reg_entry$mean_base <- list(
    years      = common_yrs,
    similarity = round(colMeans(base_vals, na.rm = TRUE), 6)
  )

  cond_entries <- Filter(function(d) d$has_cond, reg_data)
  if (length(cond_entries) > 0) {
    c_min <- min(sapply(cond_entries, function(d) min(d$cond_years)))
    c_max <- max(sapply(cond_entries, function(d) max(d$cond_years)))
    c_yrs <- seq(c_min, c_max)
    cond_vals <- do.call(rbind, lapply(cond_entries, function(d)
      approx(d$cond_years, d$cond_sim, xout = c_yrs, rule = 1)$y
    ))
    reg_entry$mean_condition <- list(
      years      = c_yrs,
      similarity = round(colMeans(cond_vals, na.rm = TRUE), 6)
    )
  }

  ibra_json_data[[reg]] <- reg_entry
}

ibra_json_path <- file.path(viz_dir, sprintf("timeseries_ibra_%s.json", prefix))
writeLines(toJSON(ibra_json_data, pretty = TRUE, auto_unbox = TRUE), ibra_json_path)
cat(sprintf("  Saved: %s\n\n", basename(ibra_json_path)))

cat(sprintf("=== Combined timeseries complete (eastern states: %s) ===\n", state_label))
