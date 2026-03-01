##############################################################################
##
## Site Aggregator for RECA obsGDM
##
## Assigns raw ALA observations to a regular grid and computes site-level
## summary statistics (record counts, visit counts, richness).
##
## Ported from: OLD_RECA/code/siteAggregator.R
## Change: replaced ffbase::bySum with data.table equivalent (in utils.R)
##
##############################################################################

siteAggregator <- function(dat, res, box) {
  ## Set breaks
  latBRKS <- seq(box[3], box[4], by = res)
  lonBRKS <- seq(box[1], box[2], by = res)

  ## Calculate centroids
  os      <- res / 2
  latCENT <- seq(box[3] + os, box[4] - os, by = res)
  lonCENT <- seq(box[1] + os, box[2] - os, by = res)

  ## Bin observations
  latCUT <- cut(dat$decimalLatitude,  breaks = latBRKS)
  lonCUT <- cut(dat$decimalLongitude, breaks = lonBRKS)

  ## Reduce data.frame to required columns
  datRED <- data.frame(
    RAW_latdec     = dat$decimalLatitude,
    RAW_longdec    = dat$decimalLongitude,
    crdUncertainty = dat$coordinateUncertaintyInMeters,
    gen_spec       = as.character(dat$scientificName),
    eventDate      = dat$eventDate
  )
  gc()

  ## Assign centroids
  datRED$lonID <- lonCENT[as.numeric(lonCUT)]
  datRED$latID <- latCENT[as.numeric(latCUT)]

  rm(latCUT, lonCUT, latCENT, lonCENT)
  gc()

  ## Remove duplicate location-species records
  n1    <- nrow(datRED)
  dupID <- paste(datRED$RAW_latdec, datRED$RAW_longdec, datRED$gen_spec, dat$eventDate, sep = ":")
  test  <- duplicated(dupID)
  datRED <- datRED[!test, ]
  cat("Duplicate records removed:", n1 - nrow(datRED), "\n")

  ## Create site IDs
  datRED$ID <- paste(datRED$lonID, datRED$latID, sep = ":")
  datRED$ID <- as.factor(datRED$ID)
  datRED    <- datRED[order(datRED$ID), ]

  ## Number of unique location-species records per site
  ones   <- rep(1, nrow(datRED))
  counts <- bySum(ones, datRED$ID)
  datRED$nRecords <- rep(counts, counts)
  rm(ones, test)
  gc()

  ## Number of records excluding same date-location duplicates
  DateLocDups <- paste(datRED$RAW_latdec, datRED$RAW_longdec, datRED$eventDate, sep = ":")
  test     <- duplicated(DateLocDups)
  recs     <- datRED$ID[!test]
  ones     <- rep(1, length(recs))
  recCounts <- bySum(ones, recs)
  datRED$nRecords.exDateLocDups <- rep(recCounts, counts)
  rm(recs, test, DateLocDups, recCounts, ones)
  gc()

  ## Number of unique observation locations per site
  sites <- paste(datRED$ID, datRED$RAW_latdec, datRED$RAW_longdec, sep = ":")
  id    <- datRED$ID
  test  <- duplicated(sites)
  sites <- sites[!test]
  id    <- id[!test]
  ones  <- rep(1, length(id))
  siteCounts <- bySum(ones, id)
  datRED$nSiteVisits <- rep(siteCounts, counts)
  rm(ones, sites, id, test, siteCounts, counts)
  gc()

  ## Reduce to unique site-species-eventDate records
  species <- paste(datRED$ID, datRED$gen_spec, datRED$eventDate, sep = ":")
  test    <- duplicated(species)
  datRED  <- datRED[!test, ]

  ## Species richness per site
  LocDups   <- paste(datRED$ID, datRED$gen_spec, sep = ":")
  test      <- duplicated(LocDups)
  recs      <- datRED$ID[!test]
  ones      <- rep(1, length(recs))
  recCounts <- bySum(ones, recs)
  ones      <- rep(1, nrow(datRED))
  counts    <- bySum(ones, datRED$ID)
  datRED$richness <- rep(recCounts, counts)
  rm(ones, counts, test)
  gc()

  datRED
}
