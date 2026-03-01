##############################################################################
##
## Site Richness Extractor (big data version) for RECA obsGDM
##
## Builds a sparse site × species matrix (m1) using parallel processing.
## This matrix is used by the obsPairSampler to compute shared species
## counts (Sørensen-type dissimilarity).
##
## Ported from: site-richness-extractor-bigData.R (CSIRO network)
## Changes:
##   - Replaced ffbase::bySum with data.table equivalent (from utils.R)
##   - Uses existing parallel cluster instead of creating new one
##   - Still assigns m1 and species.list.df to .GlobalEnv (legacy behaviour)
##
##############################################################################

site.richness.extractor.bigData <- function(frog.auGrid,
                                            sitesPerIteration = 500000,
                                            cl = NULL) {
  require(doSNOW)
  require(foreach)
  require(parallel)
  require(data.table)
  require(Matrix)

  ## Give species unique numeric ID
  if (!is.factor(frog.auGrid$species)) {
    frog.auGrid$species <- as.factor(frog.auGrid$species)
  }
  frog.auGrid$species.ID <- as.numeric(frog.auGrid$species)

  ## Species list
  species.list         <- unique(frog.auGrid$species)
  species.list         <- species.list[order(species.list)]
  species.list.numeric <- as.numeric(species.list)
  species.list.df      <- data.frame(species = species.list, species.ID = species.list.numeric)
  nc <- length(species.list)

  cat("Building site x species matrix:", nc, "species,", nrow(frog.auGrid), "records\n")

  ## Build sparse matrix in chunks via parallel processing
  mat <- data.table(species = frog.auGrid$species.ID,
                    rows    = frog.auGrid$row.count,
                    site    = as.numeric(frog.auGrid$ID))
  setkey(mat, site)
  unqsFull1      <- unique(mat$site)
  unqsFull1.lgth <- length(unqsFull1)

  if (unqsFull1.lgth >= sitesPerIteration) {
    AAA <- seq(1, unqsFull1.lgth, by = sitesPerIteration)
    BBB <- seq(sitesPerIteration, unqsFull1.lgth, by = sitesPerIteration)
    if (length(BBB) < length(AAA)) BBB <- c(BBB, unqsFull1.lgth)
  } else {
    AAA <- 1
    BBB <- unqsFull1.lgth
  }

  for (z in 1:length(AAA)) {
    cat(sprintf("  Chunk %d of %d\n", z, length(AAA)))
    unqsFull <- unqsFull1[AAA[z]:BBB[z]]
    cs       <- ceiling(length(unqsFull) / getDoParWorkers())
    chunks   <- rep(1:getDoParWorkers(), each = cs)
    chunks   <- chunks[1:length(unqsFull)]

    ij <- foreach(x = 1:getDoParWorkers(), .combine = "rbind",
                  .packages = "data.table") %dopar% {
      testFunc <- function(tt) {
        ln <- length(tt$rows)
        a  <- rep(tt$rows, ln)
        b  <- rep(tt$species, each = ln)
        cbind(a, b)
      }
      unqs <- unqsFull[chunks == x]
      do.call(rbind, lapply(unqs, function(y) { tt <- mat[J(y)]; testFunc(tt) }))
    }

    nrA <- ij[nrow(ij), 1]
    if (z == 1) {
      nrB <- nrA
      m1  <- sparseMatrix(i = ij[, 1], j = ij[, 2], dims = c(nrA, nc))
    } else {
      nrC <- ij[1, 1]
      nrB <- nrA - (nrC - 1)
      ii  <- ij[, 1] - (nrC - 1)
      m0  <- sparseMatrix(i = ii, j = ij[, 2], dims = c(nrB, nc))
      rm(ii)
      m1 <- rBind(m1, m0)
      rm(m0)
      gc()
    }
    rm(ij)
    gc()
  }

  rm(mat)
  gc()

  ## Attach site-level richness
  ones    <- rep(1, nrow(frog.auGrid))
  siteRich <- bySum(ones, frog.auGrid$ID)
  frog.auGrid$Site.Richness <- rep(siteRich, siteRich)

  ## Assign to global environment (legacy behaviour for downstream functions)
  assign("m1", m1, pos = .GlobalEnv)
  assign("species.list.df", species.list.df, pos = .GlobalEnv)

  frog.auGrid
}
