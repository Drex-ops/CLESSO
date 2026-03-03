##############################################################################
##
## download_ala.R — Download observation records from Atlas of Living
##                  Australia (ALA) via the galah package
##
## Downloads all occurrence records for a specified taxonomic group,
## applies standard quality filters, and writes a CSV formatted for
## the CLESSO run_* pipelines (compatible with siteAggregator).
##
## Usage:
##   1. Set your ALA-registered email (required):
##        Sys.setenv(ALA_EMAIL = "you@example.com")
##      or pass it as an argument to download_ala_occurrences().
##
##   2. Source this file and call download_ala_occurrences():
##        source("download_ala.R")
##        download_ala_occurrences()
##
##   3. Or run from the command line with env vars:
##        CLESSO_SPECIES_GROUP=VAS Rscript download_ala.R
##
## The output CSV has the same columns as the existing filtered_data
## files expected by run_clesso.R / run_clesso_alpha.R:
##   occurrenceID, scientificName, taxonRank, eventDate,
##   decimalLatitude, decimalLongitude, coordinateUncertaintyInMeters
##
## Supported species groups (set via CLESSO_SPECIES_GROUP or the
## `species_group` argument):
##   AVES    — Birds (class Aves)
##   VAS     — Vascular plants (kingdom Plantae, class Equisetopsida)
##   PLANTAE — All plants (kingdom Plantae, incl. bryophytes)
##   MAM     — Mammals (class Mammalia)
##   REP     — Reptiles (class Reptilia)
##   AMP     — Amphibians (class Amphibia)
##   FISH    — Actinopterygii (ray-finned fishes)
##   INSECT  — Insects (class Insecta)
##   ARACH   — Arachnids (class Arachnida)
##
## You can also pass an arbitrary taxon name via the `taxon` argument.
##
##############################################################################

# ---------------------------------------------------------------------------
# Install / load galah
# ---------------------------------------------------------------------------
if (!requireNamespace("galah", quietly = TRUE)) {
  cat("Installing galah package from CRAN...\n")
  install.packages("galah", repos = "https://cloud.r-project.org")
}
library(galah)

# ---------------------------------------------------------------------------
# Predefined taxonomic group mappings
#
# NOTE: ALA's Australian backbone taxonomy does not reliably resolve
# "Tracheophyta" — it maps to a NZ concept (NZOR-6-33408) returning
# very few records. Instead, vascular plants are queried via
# identify("Plantae") + filter(class == "Equisetopsida"), which is
# ALA's classification for all vascular plants (~29.7M records).
#
# Each entry has:
#   taxon  — name passed to galah::identify()
#   rank   — taxonomic rank (informational)
#   filter_class — optional extra filter on the `class` field
#                  (used when identify() is at a higher rank)
# ---------------------------------------------------------------------------
GROUP_TAXA <- list(
  AVES    = list(taxon = "Aves",           rank = "class",   filter_class = NULL),
  VAS     = list(taxon = "Plantae",        rank = "kingdom", filter_class = "Equisetopsida"),
  PLANTAE = list(taxon = "Plantae",        rank = "kingdom", filter_class = NULL),
  MAM     = list(taxon = "Mammalia",       rank = "class",   filter_class = NULL),
  REP     = list(taxon = "Reptilia",       rank = "class",   filter_class = NULL),
  AMP     = list(taxon = "Amphibia",       rank = "class",   filter_class = NULL),
  FISH    = list(taxon = "Actinopterygii", rank = "class",   filter_class = NULL),
  INSECT  = list(taxon = "Insecta",        rank = "class",   filter_class = NULL),
  ARACH   = list(taxon = "Arachnida",      rank = "class",   filter_class = NULL)
)

# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

#' Download ALA occurrence records for a taxonomic group
#'
#' @param species_group Character code from GROUP_TAXA (e.g. "AVES", "VAS").
#'   Ignored if \code{taxon} is supplied.
#' @param taxon  Optional character string — a taxon name to pass to
#'   \code{galah::identify()} directly. Overrides \code{species_group}.
#' @param email  ALA-registered email address. If NULL, reads from
#'   \code{Sys.getenv("ALA_EMAIL")}.
#' @param min_year  Earliest year to include (default 1970).
#' @param max_year  Latest year to include (exclusive; default current year + 1).
#' @param min_coord_uncertainty  Discard records with coordinate uncertainty
#'   above this value (metres). Default 10000 (10 km). Set NULL to skip.
#' @param apply_ala_profile  Apply the ALA data quality profile? Default TRUE.
#' @param basis_of_record  Character vector of accepted basis-of-record values.
#'   Default: human observations and machine observations only.
#' @param output_dir  Directory to write the CSV to.
#' @param output_file  Output filename. If NULL, auto-generated as
#'   \code{ala_<group>_<date>.csv}.
#' @param dry_run  If TRUE, print the expected record count and return
#'   without downloading. Useful for checking the query before committing
#'   to a large download.
#'
#' @return Invisibly returns the file path of the written CSV.
download_ala_occurrences <- function(
    species_group = Sys.getenv("CLESSO_SPECIES_GROUP", unset = "AVES"),
    taxon         = NULL,
    email         = NULL,
    min_year      = 1970L,
    max_year      = as.integer(format(Sys.Date(), "%Y")) + 1L,
    min_coord_uncertainty = 10000,
    apply_ala_profile     = TRUE,
    basis_of_record = c("HUMAN_OBSERVATION", "MACHINE_OBSERVATION"),
    output_dir    = NULL,
    output_file   = NULL,
    dry_run       = FALSE
) {

  # -----------------------------------------------------------------------
  # 0. Configure galah
  # -----------------------------------------------------------------------
  if (is.null(email)) {
    email <- Sys.getenv("ALA_EMAIL", unset = NA)
    if (is.na(email) || email == "") {
      stop("An ALA-registered email is required.\n",
           "  Set it via: Sys.setenv(ALA_EMAIL = 'you@example.com')\n",
           "  or pass email = 'you@example.com' to this function.\n",
           "  Register free at: https://auth.ala.org.au/userdetails/registration/createAccount")
    }
  }

  galah_config(atlas = "Australia", email = email, verbose = FALSE)
  cat("=== ALA Download Script ===\n")
  cat(sprintf("  Atlas  : Australia (ALA)\n"))
  cat(sprintf("  Email  : %s\n", email))


  # -----------------------------------------------------------------------
  # 1. Resolve taxonomic group
  # -----------------------------------------------------------------------
  filter_class <- NULL

  if (!is.null(taxon)) {
    taxon_name  <- taxon
    group_label <- gsub(" ", "_", taxon)
  } else {
    species_group <- toupper(species_group)
    if (!species_group %in% names(GROUP_TAXA)) {
      stop(sprintf("Unknown species_group '%s'. Valid options: %s\n  Or supply taxon = '<name>' directly.",
                   species_group, paste(names(GROUP_TAXA), collapse = ", ")))
    }
    taxon_name   <- GROUP_TAXA[[species_group]]$taxon
    filter_class <- GROUP_TAXA[[species_group]]$filter_class
    group_label  <- species_group
  }

  cat(sprintf("  Taxon  : %s\n", taxon_name))
  if (!is.null(filter_class)) cat(sprintf("  Class  : %s (additional filter)\n", filter_class))
  cat(sprintf("  Group  : %s\n", group_label))
  cat(sprintf("  Years  : %d – %d\n", min_year, max_year - 1L))

  # -----------------------------------------------------------------------
  # 2. Build query
  # -----------------------------------------------------------------------
  cat("\n--- Building query ---\n")

  ## Helper: build a base query with taxonomic + standard filters
  build_base_query <- function() {
    q <- galah_call() |>
      identify(taxon_name) |>
      filter(
        year >= min_year,
        year <  max_year,
        basisOfRecord == basis_of_record
      )
    ## Apply additional class filter if needed (e.g. VAS = Equisetopsida)
    if (!is.null(filter_class)) {
      q <- q |> filter(class == filter_class)
    }
    q
  }

  q <- build_base_query()

  ## Apply ALA quality profile (removes dubious coords, duplicates, etc.)
  if (apply_ala_profile) {
    q <- q |> apply_profile(ALA)
    cat("  ALA data quality profile applied\n")
  }

  ## Select the columns we need (matching existing CSV schema)
  q <- q |>
    select(
      occurrenceID,
      scientificName,
      taxonRank,
      eventDate,
      decimalLatitude,
      decimalLongitude,
      coordinateUncertaintyInMeters
    )

  # -----------------------------------------------------------------------
  # 3. Check record count (and optionally bail for dry run)
  # -----------------------------------------------------------------------
  cat("\n--- Checking record count ---\n")

  n_records <- build_base_query() |>
    count() |>
    collect()

  n_expected <- n_records$count[1]
  cat(sprintf("  Estimated records: %s\n", format(n_expected, big.mark = ",")))

  if (dry_run) {
    cat("\n  [DRY RUN] Skipping download. Set dry_run = FALSE to proceed.\n")
    return(invisible(NULL))
  }

  if (n_expected > 50e6) {
    cat(sprintf(
      "\n  WARNING: Very large download (%s records). This may take a long time.\n",
      format(n_expected, big.mark = ",")))
    cat("  Consider adding additional filters (e.g. narrower date range or region).\n")
  }

  # -----------------------------------------------------------------------
  # 4. Download occurrences
  # -----------------------------------------------------------------------
  cat("\n--- Downloading occurrences (this may take a while) ---\n")

  occ <- q |> collect()

  cat(sprintf("  Downloaded %s records\n", format(nrow(occ), big.mark = ",")))

  # -----------------------------------------------------------------------
  # 5. Post-processing and quality filters
  # -----------------------------------------------------------------------
  cat("\n--- Post-processing ---\n")
  n_raw <- nrow(occ)

  ## 5a. Remove records with missing coordinates
  occ <- occ[!is.na(occ$decimalLatitude) & !is.na(occ$decimalLongitude), ]
  n_after_coords <- nrow(occ)
  if (n_raw - n_after_coords > 0) {
    cat(sprintf("  Removed %d records with missing coordinates\n",
                n_raw - n_after_coords))
  }

  ## 5b. Remove records with missing or empty scientificName
  occ <- occ[!is.na(occ$scientificName) & nchar(occ$scientificName) > 0, ]
  n_after_name <- nrow(occ)
  if (n_after_coords - n_after_name > 0) {
    cat(sprintf("  Removed %d records with missing scientificName\n",
                n_after_coords - n_after_name))
  }

  ## 5c. Remove records with missing eventDate
  ## galah returns eventDate as POSIXct; convert to Date string
  occ$eventDate <- as.character(as.Date(occ$eventDate))
  occ <- occ[!is.na(occ$eventDate) & occ$eventDate != "NA", ]
  n_after_date <- nrow(occ)
  if (n_after_name - n_after_date > 0) {
    cat(sprintf("  Removed %d records with missing eventDate\n",
                n_after_name - n_after_date))
  }

  ## 5d. Filter by coordinate uncertainty
  if (!is.null(min_coord_uncertainty)) {
    ## Keep records where uncertainty is <= threshold OR is NA (many records
    ## have no uncertainty value — these are typically fine)
    occ <- occ[is.na(occ$coordinateUncertaintyInMeters) |
               occ$coordinateUncertaintyInMeters <= min_coord_uncertainty, ]
    n_after_uncert <- nrow(occ)
    if (n_after_date - n_after_uncert > 0) {
      cat(sprintf("  Removed %d records with coordinate uncertainty > %d m\n",
                  n_after_date - n_after_uncert, min_coord_uncertainty))
    }
  }

  ## 5e. Restrict to mainland Australia + Tasmania bounding box
  ##     (rough filter to remove e.g. offshore territory records)
  AUS_LAT_MIN  <- -44.0
  AUS_LAT_MAX  <- -10.0
  AUS_LON_MIN  <- 112.0
  AUS_LON_MAX  <- 154.0

  in_bounds <- occ$decimalLatitude  >= AUS_LAT_MIN &
               occ$decimalLatitude  <= AUS_LAT_MAX &
               occ$decimalLongitude >= AUS_LON_MIN &
               occ$decimalLongitude <= AUS_LON_MAX
  n_before_bbox <- nrow(occ)
  occ <- occ[in_bounds, ]
  if (n_before_bbox - nrow(occ) > 0) {
    cat(sprintf("  Removed %d records outside Australian mainland bounding box\n",
                n_before_bbox - nrow(occ)))
  }

  ## 5f. Ensure coordinateUncertaintyInMeters is numeric (replace NA → NA_real_)
  occ$coordinateUncertaintyInMeters <- as.numeric(occ$coordinateUncertaintyInMeters)

  cat(sprintf("\n  Final record count: %s\n", format(nrow(occ), big.mark = ",")))

  # -----------------------------------------------------------------------
  # 6. Format output to match existing CSV schema
  # -----------------------------------------------------------------------
  ## The run_* scripts expect CSV with these columns:
  ##   occurrenceID, scientificName, taxonRank, eventDate,
  ##   decimalLatitude, decimalLongitude, coordinateUncertaintyInMeters
  ##
  ## siteAggregator() reads from this via:
  ##   dat$scientificName, dat$eventDate, dat$decimalLatitude,
  ##   dat$decimalLongitude, dat$coordinateUncertaintyInMeters

  out <- data.frame(
    occurrenceID                  = as.character(occ$occurrenceID),
    scientificName                = as.character(occ$scientificName),
    taxonRank                     = as.character(occ$taxonRank),
    eventDate                     = occ$eventDate,
    decimalLatitude               = as.numeric(occ$decimalLatitude),
    decimalLongitude              = as.numeric(occ$decimalLongitude),
    coordinateUncertaintyInMeters = as.numeric(occ$coordinateUncertaintyInMeters),
    stringsAsFactors = FALSE
  )

  # -----------------------------------------------------------------------
  # 7. Write CSV
  # -----------------------------------------------------------------------
  if (is.null(output_dir)) {
    ## Default: project data/ directory (same level as existing CSV)
    output_dir <- tryCatch({
      proj_root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", ".."),
                                  mustWork = FALSE)
      file.path(proj_root, "data")
    }, error = function(e) {
      file.path(getwd(), "data")
    })
  }

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  if (is.null(output_file)) {
    output_file <- sprintf("ala_%s_%s.csv",
                           tolower(group_label),
                           format(Sys.Date(), "%Y-%m-%d"))
  }

  out_path <- file.path(output_dir, output_file)
  write.csv(out, file = out_path, row.names = FALSE)
  cat(sprintf("\n--- Output written to: %s ---\n", out_path))
  cat(sprintf("  Rows: %s  |  Columns: %d\n", format(nrow(out), big.mark = ","), ncol(out)))

  # -----------------------------------------------------------------------
  # 8. Print summary statistics
  # -----------------------------------------------------------------------
  cat("\n--- Summary ---\n")
  cat(sprintf("  Unique species     : %s\n",
              format(length(unique(out$scientificName)), big.mark = ",")))
  cat(sprintf("  Date range         : %s to %s\n",
              min(out$eventDate, na.rm = TRUE),
              max(out$eventDate, na.rm = TRUE)))
  cat(sprintf("  Latitude range     : %.2f to %.2f\n",
              min(out$decimalLatitude), max(out$decimalLatitude)))
  cat(sprintf("  Longitude range    : %.2f to %.2f\n",
              min(out$decimalLongitude), max(out$decimalLongitude)))

  n_with_uncert <- sum(!is.na(out$coordinateUncertaintyInMeters))
  cat(sprintf("  Coord uncertainty  : %d of %d records have values\n",
              n_with_uncert, nrow(out)))
  if (n_with_uncert > 0) {
    cat(sprintf("    Median: %.0f m, 95th pctl: %.0f m\n",
                median(out$coordinateUncertaintyInMeters, na.rm = TRUE),
                quantile(out$coordinateUncertaintyInMeters, 0.95, na.rm = TRUE)))
  }

  cat(sprintf("\n  To use with CLESSO, set:\n"))
  cat(sprintf("    Sys.setenv(CLESSO_OBS_CSV = \"%s\")\n", output_file))
  cat(sprintf("    Sys.setenv(CLESSO_SPECIES_GROUP = \"%s\")\n", group_label))

  invisible(out_path)
}


# ---------------------------------------------------------------------------
# Convenience: count records without downloading
# ---------------------------------------------------------------------------

#' Check how many records are available for a taxonomic group
#'
#' @inheritParams download_ala_occurrences
#' @return Tibble with the count
count_ala_occurrences <- function(
    species_group = "AVES",
    taxon         = NULL,
    email         = NULL,
    min_year      = 1970L,
    max_year      = as.integer(format(Sys.Date(), "%Y")) + 1L,
    basis_of_record = c("HUMAN_OBSERVATION", "MACHINE_OBSERVATION")
) {

  if (is.null(email)) {
    email <- Sys.getenv("ALA_EMAIL", unset = NA)
    if (is.na(email) || email == "") {
      stop("Set ALA_EMAIL environment variable or pass email argument.")
    }
  }
  galah_config(atlas = "Australia", email = email, verbose = FALSE)

  if (!is.null(taxon)) {
    taxon_name   <- taxon
    filter_class <- NULL
  } else {
    species_group <- toupper(species_group)
    if (!species_group %in% names(GROUP_TAXA)) {
      stop(sprintf("Unknown species_group '%s'.", species_group))
    }
    taxon_name   <- GROUP_TAXA[[species_group]]$taxon
    filter_class <- GROUP_TAXA[[species_group]]$filter_class
  }

  q <- galah_call() |>
    identify(taxon_name) |>
    filter(
      year >= min_year,
      year <  max_year,
      basisOfRecord == basis_of_record
    )
  if (!is.null(filter_class)) {
    q <- q |> filter(class == filter_class)
  }

  counts <- q |>
    count() |>
    collect()

  cat(sprintf("Records available for %s (%d–%d): %s\n",
              taxon_name, min_year, max_year - 1L,
              format(counts$count[1], big.mark = ",")))
  counts
}


# ===========================================================================
# CLI entry point — run only when invoked via Rscript (not source())
# ===========================================================================
if (!interactive() && sys.nframe() == 0L) {
  ## Only auto-run if executed directly as a script
  if (Sys.getenv("ALA_EMAIL", unset = "") != "") {
    download_ala_occurrences()
  } else {
    cat("download_ala.R loaded. Call download_ala_occurrences() to begin.\n")
    cat("  Required: Sys.setenv(ALA_EMAIL = 'you@example.com')\n")
    cat("  Optional: Sys.setenv(CLESSO_SPECIES_GROUP = 'VAS')\n")
  }
}
