##############################################################################
##
## Site Richness Extractor (big data version) for RECA obsGDM
##
## Builds a site × species presence matrix and exposes it to downstream
## code (obsPairSampler) via a lightweight proxy that supports `[i, ]`
## subsetting without materialising the full record × species matrix.
##
## The proxy stores:
##   site_sp  -- n_sites × n_species sparse binary matrix (~100 MB)
##   site_map -- record -> site index vector (~25 MB)
##
## When the sampler requests m1[samp, ], the proxy returns
## site_sp[site_map[samp], ], which is a real sparse matrix.
## This avoids storing the 3M × 20K matrix with 578M+ nonzeros (~7 GB)
## that caused OOM on large taxa (VAS, INVERT).
##
## v3 (2026-03-03): Proxy-based approach.  Never materialises full m1.
## v2 (2026-03-03): Sparse algebra (still OOM on m1 materialisation).
##
## Assigns to .GlobalEnv: m1 (proxy), site_sp_matrix, site_levels,
##   species.list.df  (all legacy).
##
##############################################################################

# ---------------------------------------------------------------------------
# Proxy class: record × species lookup via site_sp + site_map
# ---------------------------------------------------------------------------

#' Create a proxy that behaves like a record × species matrix
#' but only stores the compact site × species matrix + a mapping vector.
site_species_proxy <- function(site_sp, record_site_map) {
  obj <- list(site_sp = site_sp, site_map = record_site_map)
  class(obj) <- "site_species_proxy"
  obj
}

#' Subset operator: m1[i, ] -> site_sp[site_map[i], ]
`[.site_species_proxy` <- function(x, i, j, ..., drop = FALSE) {
  if (missing(j)) {
    x$site_sp[x$site_map[i], , drop = drop]
  } else {
    x$site_sp[x$site_map[i], j, drop = drop]
  }
}

nrow.site_species_proxy <- function(x) length(x$site_map)
ncol.site_species_proxy <- function(x) ncol(x$site_sp)
dim.site_species_proxy  <- function(x) c(length(x$site_map), ncol(x$site_sp))

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

site.richness.extractor.bigData <- function(frog.auGrid,
                                            sitesPerIteration = 500000,
                                            cl = NULL) {
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
  nr <- nrow(frog.auGrid)

  cat("Building site x species matrix:", nc, "species,", nr, "records\n")

  ## ------------------------------------------------------------------
  ## Build compact site × species presence matrix
  ## ------------------------------------------------------------------

  ## 1. Record -> species indicator (exactly nr nonzeros)
  cat("  Step 1/3: record-species indicator\n")
  m_rec <- sparseMatrix(
    i = seq_len(nr),
    j = frog.auGrid$species.ID,
    x = 1,
    dims = c(nr, nc)
  )

  ## 2. Record -> site indicator (exactly nr nonzeros)
  cat("  Step 2/3: record-site indicator\n")
  site_fac <- as.factor(frog.auGrid$ID)
  site_ids <- as.integer(site_fac)
  n_sites  <- nlevels(site_fac)
  m_site <- sparseMatrix(
    i = seq_len(nr),
    j = site_ids,
    x = 1,
    dims = c(nr, n_sites)
  )

  ## 3. Site × species presence (n_sites × nc)
  cat(sprintf("  Step 3/3: site-level presence (%d sites × %d species)\n",
              n_sites, nc))
  site_sp <- crossprod(m_site, m_rec)       # = t(M_site) %*% M_rec
  site_sp@x <- rep(1, length(site_sp@x))   # counts -> binary presence
  rm(m_rec, m_site); gc()

  cat(sprintf("  Done. site_sp: %d × %d, %s nonzeros (proxy avoids %s-row expansion)\n",
              nrow(site_sp), ncol(site_sp),
              format(length(site_sp@x), big.mark = ","),
              format(nr, big.mark = ",")))

  ## Build proxy: record -> site -> species
  m1 <- site_species_proxy(site_sp, site_ids)

  ## Attach site-level richness (unique species count per site)
  ## Use rowSums on the binary site×species matrix for true species count,
  ## NOT bySum on records (which gives record count, not species count).
  site_richness <- as.integer(rowSums(site_sp))  # n_sites vector
  ## Expand back to per-record: each record gets its site's richness
  frog.auGrid$Site.Richness <- site_richness[site_ids]

  ## Assign to global environment (legacy behaviour for downstream functions)
  assign("m1", m1, pos = .GlobalEnv)
  assign("site_sp_matrix", site_sp, pos = .GlobalEnv)
  assign("site_levels", levels(site_fac), pos = .GlobalEnv)
  assign("species.list.df", species.list.df, pos = .GlobalEnv)

  frog.auGrid
}
