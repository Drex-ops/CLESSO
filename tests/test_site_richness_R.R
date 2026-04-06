#!/usr/bin/env Rscript
##############################################################################
##
## Test: Verify site richness is computed correctly
##
## Ground-truth strategy:
##   1. Load raw ALA VAS data
##   2. Bin to 0.05° grid using SAME logic as siteAggregator
##   3. Compute species richness per cell directly (uniqueN)
##   4. Run siteAggregator() and compare datRED$richness
##   5. Run site.richness.extractor.bigData() and compare Site.Richness
##   6. Run clesso_format_aggregated_data() and compare obs_dt$richness
##
## Usage:
##   Rscript tests/test_site_richness_R.R
##
##############################################################################

library(data.table)
library(raster)
library(Matrix)

## --------------------------------------------------------------------------
## Setup paths
## --------------------------------------------------------------------------
## Detect script location robustly (works with Rscript and source())
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
if (length(script_path) == 0) {
  ## Fallback: assume working directory is the project root
  project_root <- getwd()
} else {
  project_root <- normalizePath(file.path(dirname(script_path), ".."))
}
setwd(project_root)

## Source shared utilities
source("src/shared/R/utils.R")
source("src/shared/R/site_aggregator.R")
source("src/shared/R/site_richness_extractor.R")
source("src/clesso_v2/clesso_sampler.R")

cat("=" , rep("=", 69), "\n", sep = "")
cat("  Site Richness Verification Test\n")
cat("=", rep("=", 69), "\n\n", sep = "")

## --------------------------------------------------------------------------
## STEP 1: Load raw data + reference raster
## --------------------------------------------------------------------------
cat("--- Step 1: Load raw data ---\n")

obs_csv <- "data/ala_vas_2026-03-03.csv"
ref_raster_path <- "data/FWPT_mean_Cmax_mean_1946_1975.flt"

dat <- fread(obs_csv)
cat(sprintf("  Raw records: %s\n", format(nrow(dat), big.mark = ",")))

ras <- raster(ref_raster_path)
res_deg <- res(ras)[1]
box <- extent(ras)
cat(sprintf("  Grid resolution: %.2f°\n", res_deg))
cat(sprintf("  Extent: [%.2f, %.2f] × [%.2f, %.2f]\n",
            box[1], box[2], box[3], box[4]))

## --------------------------------------------------------------------------
## STEP 2: Ground truth — compute richness directly from raw data
## --------------------------------------------------------------------------
cat("\n--- Step 2: Compute ground-truth richness ---\n")

## Replicate the siteAggregator grid binning
latBRKS <- seq(box[3], box[4], by = res_deg)
lonBRKS <- seq(box[1], box[2], by = res_deg)
os <- res_deg / 2
latCENT <- seq(box[3] + os, box[4] - os, by = res_deg)
lonCENT <- seq(box[1] + os, box[2] - os, by = res_deg)

dat[, lonID := lonCENT[as.numeric(cut(decimalLongitude, breaks = lonBRKS))]]
dat[, latID := latCENT[as.numeric(cut(decimalLatitude,  breaks = latBRKS))]]
dat[, site_id := paste(lonID, latID, sep = ":")]

## Drop records that fall outside the grid
dat <- dat[!is.na(lonID) & !is.na(latID)]

## Ground truth: unique species per site (all dates)
truth <- dat[, .(richness_truth = uniqueN(scientificName)), by = site_id]
cat(sprintf("  Ground-truth sites: %s\n", format(nrow(truth), big.mark = ",")))
cat(sprintf("  Richness: mean=%.1f, median=%.0f, max=%d\n",
            mean(truth$richness_truth),
            median(truth$richness_truth),
            max(truth$richness_truth)))

## --------------------------------------------------------------------------
## STEP 3: Run siteAggregator and compare
## --------------------------------------------------------------------------
cat("\n--- Step 3: Test siteAggregator() ---\n")

## siteAggregator expects a data.frame with specific columns
dat_df <- as.data.frame(dat)
datRED <- siteAggregator(dat_df, res_deg, box)
datRED_dt <- as.data.table(datRED)

## siteAggregator richness is per-site, replicated per record
agg_richness <- unique(datRED_dt[, .(site_id = ID, richness_agg = richness)])

## Merge with ground truth
cmp1 <- merge(truth, agg_richness, by = "site_id", all = FALSE)
cat(sprintf("  Matched sites: %d\n", nrow(cmp1)))
cat(sprintf("  Exact matches: %d / %d (%.1f%%)\n",
            sum(cmp1$richness_truth == cmp1$richness_agg),
            nrow(cmp1),
            100 * mean(cmp1$richness_truth == cmp1$richness_agg)))

## Any mismatches?
mismatches1 <- cmp1[richness_truth != richness_agg]
if (nrow(mismatches1) > 0) {
  cat(sprintf("  *** MISMATCH: %d sites differ ***\n", nrow(mismatches1)))
  mismatches1[, diff := richness_truth - richness_agg]
  cat(sprintf("  Max diff: %d, mean diff: %.2f\n",
              max(abs(mismatches1$diff)), mean(mismatches1$diff)))
  ## Show a few examples
  cat("  Examples:\n")
  print(head(mismatches1[order(-abs(diff))], 10))
} else {
  cat("  *** PASS: siteAggregator richness matches ground truth ***\n")
}

## --------------------------------------------------------------------------
## STEP 4: Run site.richness.extractor.bigData and compare
## --------------------------------------------------------------------------
cat("\n--- Step 4: Test site.richness.extractor.bigData() ---\n")

## Prepare input (same as run_obsGDM.R step 3)
## Filter dates as RECA pipeline does.
## Note: siteAggregator may return eventDate as IDate (integer) or character,
## so coerce to Date first, then filter on NAs.
datRED_filt <- copy(datRED_dt)
datRED_filt[, ed := as.Date(as.character(eventDate))]
datRED_filt <- datRED_filt[!is.na(ed)]
datRED_filt <- datRED_filt[ed >= as.Date("1970-01-01") & ed < as.Date("2018-01-01")]

## Reduce to unique site x species like RECA does
data_for_extractor <- data.frame(
  ID        = datRED_filt$ID,
  Latitude  = datRED_filt$latID,
  Longitude = datRED_filt$lonID,
  species   = datRED_filt$gen_spec,
  nRecords  = datRED_filt$nRecords,
  nRecords.exDateLocDups = datRED_filt$nRecords.exDateLocDups,
  nSiteVisits = datRED_filt$nSiteVisits,
  richness  = datRED_filt$richness,
  stringsAsFactors = FALSE
)

LocDups <- paste(data_for_extractor$ID, data_for_extractor$species, sep = ":")
test <- duplicated(LocDups)
data_for_extractor <- data_for_extractor[!test, ]
data_for_extractor <- data_for_extractor[order(data_for_extractor$ID), ]
data_for_extractor$species <- as.factor(data_for_extractor$species)

## Run extractor
frog.auGrid <- site.richness.extractor.bigData(frog.auGrid = data_for_extractor)

## Extract site-level richness from the extractor (Site.Richness)
ext_richness <- unique(data.table(
  site_id = frog.auGrid$ID,
  richness_ext = frog.auGrid$Site.Richness
))

## Ground truth for date-filtered sites:
## After date filtering, some species might be removed from a site.
## The extractor operates on post-filter data, so ground truth here =
## unique species per site in the date-filtered dataset.
truth_filt <- as.data.table(data_for_extractor)[,
  .(richness_truth_filt = uniqueN(species)), by = .(site_id = ID)]

cmp2 <- merge(truth_filt, ext_richness, by = "site_id", all = FALSE)
cat(sprintf("  Matched sites: %d\n", nrow(cmp2)))
cat(sprintf("  Exact matches: %d / %d (%.1f%%)\n",
            sum(cmp2$richness_truth_filt == cmp2$richness_ext),
            nrow(cmp2),
            100 * mean(cmp2$richness_truth_filt == cmp2$richness_ext)))

mismatches2 <- cmp2[richness_truth_filt != richness_ext]
if (nrow(mismatches2) > 0) {
  cat(sprintf("  *** MISMATCH: %d sites differ ***\n", nrow(mismatches2)))
  mismatches2[, diff := richness_truth_filt - richness_ext]
  cat(sprintf("  Max diff: %d, mean diff: %.2f\n",
              max(abs(mismatches2$diff)), mean(mismatches2$diff)))
  print(head(mismatches2[order(-abs(diff))], 10))
} else {
  cat("  *** PASS: site.richness.extractor Site.Richness matches ground truth ***\n")
}

## --------------------------------------------------------------------------
## STEP 5: Test clesso_format_aggregated_data() richness pass-through
## --------------------------------------------------------------------------
cat("\n--- Step 5: Test clesso_format_aggregated_data() ---\n")

## Apply date filter to datRED (as run_clesso.R does)
datRED_clesso <- copy(datRED_dt)
datRED_clesso[, ed := as.Date(as.character(eventDate))]
datRED_clesso <- datRED_clesso[!is.na(ed)]
datRED_clesso <- datRED_clesso[ed >= as.Date("1970-01-01") & ed < as.Date("2018-01-01")]
datRED_clesso <- droplevels(as.data.frame(datRED_clesso))

obs_dt <- clesso_format_aggregated_data(datRED_clesso)

## obs_dt$richness should carry through from datRED$richness (pre-filter)
clesso_richness <- unique(obs_dt[, .(site_id, richness_clesso = richness)])

## Compare to siteAggregator richness (pre-date-filter truth)
cmp3 <- merge(agg_richness, clesso_richness, by = "site_id", all = FALSE)
cat(sprintf("  Matched sites: %d\n", nrow(cmp3)))
cat(sprintf("  Exact matches: %d / %d (%.1f%%)\n",
            sum(cmp3$richness_agg == cmp3$richness_clesso),
            nrow(cmp3),
            100 * mean(cmp3$richness_agg == cmp3$richness_clesso)))

mismatches3 <- cmp3[richness_agg != richness_clesso]
if (nrow(mismatches3) > 0) {
  cat(sprintf("  *** MISMATCH: %d sites differ ***\n", nrow(mismatches3)))
} else {
  cat("  *** PASS: clesso_format_aggregated_data richness matches siteAggregator ***\n")
}

## --------------------------------------------------------------------------
## STEP 6: Test that the pre-filter richness is higher than or equal to
## the post-filter uniqueN(species) — it should always be >= because
## filtering can only remove species, never add them.
## --------------------------------------------------------------------------
cat("\n--- Step 6: Verify pre-filter richness >= post-filter uniqueN ---\n")

post_filter_rich <- obs_dt[, .(richness_post = uniqueN(species)), by = site_id]
pre_vs_post <- merge(clesso_richness, post_filter_rich, by = "site_id")
pre_vs_post[, passes := richness_clesso >= richness_post]
n_fail <- sum(!pre_vs_post$passes)
cat(sprintf("  Sites where pre-filter >= post-filter: %d / %d\n",
            sum(pre_vs_post$passes), nrow(pre_vs_post)))
if (n_fail > 0) {
  cat(sprintf("  *** FAIL: %d sites have pre-filter < post-filter ***\n", n_fail))
  print(head(pre_vs_post[passes == FALSE]))
} else {
  cat("  *** PASS: pre-filter richness always >= post-filter uniqueN ***\n")
}

## --------------------------------------------------------------------------
## Summary
## --------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("  SUMMARY\n")
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("  1. siteAggregator richness vs ground truth:       %s\n",
            ifelse(nrow(mismatches1) == 0, "PASS", "FAIL")))
cat(sprintf("  2. site.richness.extractor Site.Richness:         %s\n",
            ifelse(nrow(mismatches2) == 0, "PASS", "FAIL")))
cat(sprintf("  3. clesso_format_aggregated_data pass-through:    %s\n",
            ifelse(nrow(mismatches3) == 0, "PASS", "FAIL")))
cat(sprintf("  4. Pre-filter >= post-filter richness:            %s\n",
            ifelse(n_fail == 0, "PASS", "FAIL")))
cat(rep("=", 70), "\n")
