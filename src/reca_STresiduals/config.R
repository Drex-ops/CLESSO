##############################################################################
##
## RECA obsGDM Configuration
##
## All paths and run parameters are defined here. Override any setting by
## setting the corresponding environment variable before sourcing this file,
## or by editing the defaults below.
##
##############################################################################

# ---------------------------------------------------------------------------
# Helper: read env var with fallback default
# ---------------------------------------------------------------------------
env_or_default <- function(var, default) {
  val <- Sys.getenv(var, unset = NA)
  if (is.na(val)) default else val
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

config <- list()

## Root directory for this project (two levels up from this config file)
config$project_root <- env_or_default(
  "RECA_PROJECT_ROOT",
  normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", ".."), mustWork = FALSE)
)

## Shared R module directory
config$r_dir <- file.path(config$project_root, "src", "shared", "R")

## Python executable (for gen_windows -> pyper.py calls)
## Default to the project venv Python so pyarrow/geonpy are available
config$python_exe <- env_or_default(
  "RECA_PYTHON_EXE",
  file.path(config$project_root, ".venv", "bin", "python3")
)

## Path to pyper.py script
config$pyper_script <- env_or_default(
	"RECA_PYPER_SCRIPT",
	file.path(config$project_root, "src", "shared", "python", "pyper.py")
)

## Input data directory (ALA CSV files, substrate rasters, etc.)
config$data_dir <- env_or_default("RECA_DATA_DIR", file.path(config$project_root, "data"))

## AWAP geonpy .npy files directory
config$npy_src <- env_or_default("RECA_NPY_SRC", "/Volumes/PortableSSD/CLIMATE/geonpy")

## Substrate raster brick file (.grd)
config$substrate_raster <- env_or_default(
  "RECA_SUBSTRATE_RASTER",
  file.path(config$data_dir, "SUBS_brk_VAS.grd")
)

## Reference raster for grid alignment (.flt)
config$reference_raster <- env_or_default(
  "RECA_REFERENCE_RASTER",
  file.path(config$data_dir, "FWPT_mean_Cmax_mean_1946_1975.flt")
)

## Output directory
config$output_dir <- env_or_default("RECA_OUTPUT_DIR", file.path(config$project_root, "src", "reca_STresiduals", "output"))

## Raster temp directory (rasterOptions(tmpdir=...))
config$raster_tmpdir <- env_or_default("RECA_RASTER_TMPDIR", tempdir())

## Temp directory for feather exchange files
config$feather_tmpdir <- env_or_default("RECA_FEATHER_TMPDIR", tempdir())

# ---------------------------------------------------------------------------
# Run parameters (defaults for AVES SpatTemp biAverage v3)
# ---------------------------------------------------------------------------

## Species group: "AVES", "PLANTS", or "VAS"
config$species_group <- env_or_default("RECA_SPECIES_GROUP", "VAS")

## Input observations CSV filename (relative to data_dir)
config$obs_csv <- env_or_default("RECA_OBS_CSV", "ala_vas_2026-03-03.csv")

## Number of observation-pair matches to sample
config$nMatch <- as.integer(env_or_default("RECA_NMATCH", "1000000"))

## Climate window years to iterate over (kept for compatibility)
config$c_yrs <- eval(parse(text = env_or_default("RECA_C_YRS", "seq(61, 75, by = 2)")))

## Weather window years to iterate over (kept for compatibility)
config$w_yrs <- eval(parse(text = env_or_default("RECA_W_YRS", "c(1)")))

## Fixed climate window size in years for spatial-temporal residual extraction.
## Used as the averaging window for both spatial and temporal env extraction.
config$climate_window <- as.integer(env_or_default("RECA_CLIMATE_WINDOW", "30"))

## Use bidirectional averaging of env extraction (biAverage)
config$biAverage <- as.logical(env_or_default("RECA_BIAVERAGE", "TRUE"))

## Decomposition variant: "none", "v2" (spatial-temporal), or "v3" (same_site|same_time only)
config$decomposition <- env_or_default("RECA_DECOMPOSITION", "none")

## Minimum observation date filter (observations before this date are excluded)
config$min_date <- as.Date(env_or_default("RECA_MIN_DATE", "2000-01-01"))

## Maximum observation date filter
config$max_date <- as.Date(env_or_default("RECA_MAX_DATE", "2018-01-01"))

## Date offset in years added to min_date for temporal filtering
## (observations need env data going back c_yr years from their date)
config$date_offset_years <- as.integer(env_or_default("RECA_DATE_OFFSET_YEARS", "1"))

## Grid resolution for site aggregation (degrees)
config$grid_resolution <- as.numeric(env_or_default("RECA_GRID_RESOLUTION", "0.01"))

## geonpy start/end year (range of monthly climate data in .npy files)
config$geonpy_start_year <- as.integer(env_or_default("RECA_GEONPY_START_YEAR", "1911"))
config$geonpy_end_year   <- as.integer(env_or_default("RECA_GEONPY_END_YEAR",   "2017"))

## Parallel settings
config$cores_to_use <- as.integer(env_or_default(
  "RECA_CORES",
  as.character(max(1, parallel::detectCores() - 1))
))

## Chunk size for chunked env extraction (rows per chunk)
config$chunk_size <- as.integer(env_or_default("RECA_CHUNK_SIZE", "10000"))

## Species threshold for switching parallel strategies in obsPairSampler
config$species_threshold <- as.integer(env_or_default("RECA_SPECIES_THRESHOLD", "500"))

## Number of samples for estimating mismatch/match ratio (w)
config$w_estimation_samples <- as.integer(env_or_default("RECA_W_SAMPLES", "8000000"))

## Skip env extraction if output file already exists
config$skip_existing_env <- as.logical(env_or_default("RECA_SKIP_EXISTING_ENV", "TRUE"))

# ---------------------------------------------------------------------------
# MODIS land cover extraction (2001–2020 only)
#
# When enabled, MODIS nontree / nonveget / treecov rasters are extracted
# for each observation-pair year and added to the covariate set.
# File naming convention:  modis_{year}_{variable}_{resolution}_COG.tif
#   e.g.  data/modis/modis_2015_treecov_1km_COG.tif
# ---------------------------------------------------------------------------
config$add_modis        <- as.logical(env_or_default("RECA_ADD_MODIS", "TRUE"))
config$modis_dir        <- env_or_default("RECA_MODIS_DIR",
                             file.path(config$data_dir, "modis"))
config$modis_variables  <- c("nontree", "nonveget", "treecov")
config$modis_start_year <- 2001L
config$modis_end_year   <- 2020L
config$modis_resolution <- "1km"   # used in filename pattern

## Convenience suffix for output file naming ("_MODIS" or "")
config$modis_suffix <- if (config$add_modis) "MODIS_" else ""

## Maximum pairs to use for GDM fitting (NULL = no limit, use all).
## Set to e.g. 1000000 if the GLM runs out of memory.
config$max_fit_pairs <- NULL

# ---------------------------------------------------------------------------
# Environmental variable extraction parameters
#
# Each element is a list with:
#   variables - character vector of geonpy variable names (without .npy)
#   mstat     - monthly summary statistic ('mean', 'min', 'max')
#   cstat     - climatology summary statistic ('mean', 'min', 'max')
# ---------------------------------------------------------------------------
config$env_params <- list(
  list(variables = c("mean_PT_191101-201712"),
       mstat = "mean", cstat = "mean"),
  list(variables = c("TNn_191101-201712", "FWPT_191101-201712"),
       mstat = "mean", cstat = "min"),
  list(variables = c("max_PT_191101-201712", "FWPT_191101-201712"),
       mstat = "mean", cstat = "max"),
  list(variables = c("FD_191101-201712", "TXx_191101-201712"),
       mstat = "mean", cstat = "max"),
  list(variables = c("TNn_191101-201712", "PD_191101-201712"),
       mstat = "mean", cstat = "max")
)

# ---------------------------------------------------------------------------
# Derived: fitted model path
# ---------------------------------------------------------------------------
config$fit_filename <- paste0(
  config$species_group, "_",
  format(config$nMatch / 1e6, nsmall = 0), "mil_",
  config$climate_window, "climWin_STresid_",
  if (config$biAverage) "biAverage_" else "",
  config$modis_suffix,
  "fittedGDM.RData"
)
config$fit_path <- file.path(config$output_dir, config$fit_filename)

# ---------------------------------------------------------------------------
# Create output directory if needed
# ---------------------------------------------------------------------------
if (!dir.exists(config$output_dir)) {
  dir.create(config$output_dir, recursive = TRUE)
}

cat("RECA config loaded.\n")
cat("  Species group  :", config$species_group, "\n")
cat("  Fit file       :", config$fit_filename, "\n")
cat("  Data dir       :", config$data_dir, "\n")
cat("  Output dir     :", config$output_dir, "\n")
cat("  nMatch         :", config$nMatch, "\n")
cat("  climate_window :", config$climate_window, "years\n")
cat("  c_yrs          :", paste(config$c_yrs, collapse = ", "), "\n")
cat("  w_yrs          :", paste(config$w_yrs, collapse = ", "), "\n")
cat("  biAverage      :", config$biAverage, "\n")
cat("  decomposition  :", config$decomposition, "\n")
cat("  add_modis      :", config$add_modis, "\n")
cat("  cores          :", config$cores_to_use, "\n")
