##############################################################################
##
## clesso_config.R -- Configuration for CLESSO v2 runs
##
## Follows the same pattern as src/reca_STresiduals/config.R but adds
## parameters specific to the joint alpha-beta sampling and modelling.
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
clesso_config <- list()

## Root directory (three levels up from this config file => project root)
clesso_config$project_root <- env_or_default(
  "CLESSO_PROJECT_ROOT",
  normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", ".."), mustWork = FALSE)
)

## Shared R module directory
clesso_config$r_dir <- file.path(clesso_config$project_root, "src", "shared", "R")

## CLESSO v2 source directory
clesso_config$clesso_dir <- file.path(clesso_config$project_root, "src", "clesso_v2")

## Data directory
clesso_config$data_dir <- env_or_default(
  "CLESSO_DATA_DIR",
  file.path(clesso_config$project_root, "data")
)

## Output directory
clesso_config$output_dir <- env_or_default(
  "CLESSO_OUTPUT_DIR",
  file.path(clesso_config$clesso_dir, "output")
)

## Python executable (for env extraction via pyper.py)
clesso_config$python_exe <- env_or_default(
  "CLESSO_PYTHON_EXE",
  file.path(clesso_config$project_root, ".venv", "bin", "python3")
)

## Path to pyper.py
clesso_config$pyper_script <- env_or_default(
  "CLESSO_PYPER_SCRIPT",
  file.path(clesso_config$project_root, "src", "shared", "python", "pyper.py")
)

## AWAP geonpy .npy files directory
clesso_config$npy_src <- env_or_default(
  "CLESSO_NPY_SRC",
  "/Volumes/PortableSSD/CLIMATE/geonpy"
)

## Substrate raster (.grd)
clesso_config$substrate_raster <- env_or_default(
  "CLESSO_SUBSTRATE_RASTER",
  file.path(clesso_config$data_dir, "SUBS_brk_VAS.grd")
)

## Reference raster for grid alignment (.flt)
clesso_config$reference_raster <- env_or_default(
  "CLESSO_REFERENCE_RASTER",
  file.path(clesso_config$data_dir, "FWPT_mean_Cmax_mean_1946_1975.flt")
)

## Raster temp directory
clesso_config$raster_tmpdir <- env_or_default("CLESSO_RASTER_TMPDIR", tempdir())

## Temp directory for feather exchange
clesso_config$feather_tmpdir <- env_or_default("CLESSO_FEATHER_TMPDIR", tempdir())

# ---------------------------------------------------------------------------
# Species / data parameters
# ---------------------------------------------------------------------------

## Species group
clesso_config$species_group <- env_or_default("CLESSO_SPECIES_GROUP", "VAS")

## Input CSV filename (relative to data_dir)
clesso_config$obs_csv <- env_or_default("CLESSO_OBS_CSV", "ala_vas_2026-03-03.csv")

## Date bounds for observation filtering
clesso_config$min_date <- as.Date(env_or_default("CLESSO_MIN_DATE", "1970-01-01"))
clesso_config$max_date <- as.Date(env_or_default("CLESSO_MAX_DATE", "2018-01-01"))

## Grid resolution for site aggregation (degrees)
clesso_config$grid_resolution <- as.numeric(env_or_default("CLESSO_GRID_RESOLUTION", "0.01"))

# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

## Number of within-site pairs to sample
clesso_config$n_within <- as.integer(env_or_default("CLESSO_N_WITHIN", "500000"))

## Number of between-site pairs to sample
clesso_config$n_between <- as.integer(env_or_default("CLESSO_N_BETWEEN", "500000"))

## Minimum records per site for within-site pair eligibility
clesso_config$within_min_records <- as.integer(env_or_default("CLESSO_WITHIN_MIN_RECS", "2"))

## Target match ratio for within-site pairs (NULL string = natural ratio)
clesso_config$within_match_ratio <- {
  val <- env_or_default("CLESSO_WITHIN_MATCH_RATIO", "0.5")
  if (val == "NULL") NULL else as.numeric(val)
}

## Target match ratio for between-site pairs
clesso_config$between_match_ratio <- {
  val <- env_or_default("CLESSO_BETWEEN_MATCH_RATIO", "0.5")
  if (val == "NULL") NULL else as.numeric(val)
}

## Whether to compute balancing weights for within/between pairs
clesso_config$balance_weights <- as.logical(env_or_default("CLESSO_BALANCE_WEIGHTS", "TRUE"))

## Random seed for reproducibility (NULL = no seed)
clesso_config$seed <- {
  val <- env_or_default("CLESSO_SEED", "42")
  if (val == "NULL") NULL else as.integer(val)
}

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

## Number of I-spline bases per variable (for turnover model)
clesso_config$n_splines <- as.integer(env_or_default("CLESSO_N_SPLINES", "3"))

## Include geographic distance in turnover model
clesso_config$geo_distance <- as.logical(env_or_default("CLESSO_GEO_DISTANCE", "TRUE"))

## Standardize alpha covariates (Z)
clesso_config$standardize_Z <- as.logical(env_or_default("CLESSO_STANDARDIZE_Z", "TRUE"))

## Initial alpha estimate (species richness prior)
clesso_config$alpha_init <- as.numeric(env_or_default("CLESSO_ALPHA_INIT", "20"))

## ----------- Alpha spline settings -----------
## Use spline smooth terms for the alpha (richness) sub-model.
## When TRUE, g_k(z) are B-spline terms; when FALSE, linear only.
clesso_config$use_alpha_splines <- as.logical(env_or_default("CLESSO_USE_ALPHA_SPLINES", "TRUE"))

## Spline type: "penalised" (P-spline, default) or "regression" (fixed knots,
## no smoothness penalty -- coefficients estimated as fixed effects).
## For regression splines the knot number/positions fully determine the
## flexibility; for P-splines the penalty parameter lambda is estimated.
clesso_config$alpha_spline_type <- tolower(env_or_default("CLESSO_ALPHA_SPLINE_TYPE", "regression"))
stopifnot(clesso_config$alpha_spline_type %in% c("penalised", "regression"))

## Number of interior knots per alpha covariate spline
clesso_config$alpha_n_knots <- as.integer(env_or_default("CLESSO_ALPHA_N_KNOTS", "10"))

## B-spline degree (3 = cubic, the standard choice)
clesso_config$alpha_spline_deg <- as.integer(env_or_default("CLESSO_ALPHA_SPLINE_DEG", "3"))

## Difference penalty order (2 = penalise 2nd differences, standard P-spline)
## Only used when alpha_spline_type = "penalised".
clesso_config$alpha_pen_order <- as.integer(env_or_default("CLESSO_ALPHA_PEN_ORDER", "2"))

## User-specified interior knot positions (optional).
## When NULL (default), knots are placed at equally-spaced quantiles of the
## covariate values. To set custom positions, provide a list of numeric
## vectors -- one per alpha covariate -- via the R environment or by
## assigning directly after sourcing this config:
##   clesso_config$alpha_knot_positions <- list(c(10,20,30), c(0.1,0.5,0.9))
## When set, alpha_n_knots is ignored and the number of knots is determined
## by the length of each vector.
clesso_config$alpha_knot_positions <- NULL

## --- Alpha lower bound (observed richness penalty) ---
## Penalty weight for soft lower bound on alpha: discourages alpha < S_obs
## (observed species count). Uses a smooth one-sided hinge so the model
## still estimates *total* richness from covariates (no offset), and
## predictions at new sites (where S_obs is unknown) remain unbiased.
## Set to 0 to disable. Recommended range: 1-100.
clesso_config$alpha_lower_bound_lambda <- as.numeric(
  env_or_default("CLESSO_ALPHA_LB_LAMBDA", "10")
)

## Climate window size in years (for env extraction)
clesso_config$climate_window <- as.integer(env_or_default("CLESSO_CLIMATE_WINDOW", "30"))

## geonpy start year
clesso_config$geonpy_start_year <- as.integer(env_or_default("CLESSO_GEONPY_START_YEAR", "1911"))

# ---------------------------------------------------------------------------
# TMB optimisation
# ---------------------------------------------------------------------------
clesso_config$tmb_eval_max <- as.integer(env_or_default("CLESSO_TMB_EVAL_MAX", "4000"))
clesso_config$tmb_iter_max <- as.integer(env_or_default("CLESSO_TMB_ITER_MAX", "4000"))

# ---------------------------------------------------------------------------
# Iterative (alternating alpha/beta) fitting
# ---------------------------------------------------------------------------
## When TRUE, use block-coordinate descent instead of joint optimisation.
## This alternates between fixing alpha and fitting beta, then fixing beta
## and fitting alpha, which avoids the full joint Hessian and is faster
## when the model has many spline coefficients.
clesso_config$iterative_fitting <- as.logical(
  env_or_default("CLESSO_ITERATIVE_FITTING", "FALSE")
)
## Maximum number of full alpha/beta cycles
clesso_config$iterative_max_iter <- as.integer(
  env_or_default("CLESSO_ITERATIVE_MAX_ITER", "20")
)
## Convergence tolerance on relative change in negative log-likelihood
clesso_config$iterative_tol <- as.numeric(
  env_or_default("CLESSO_ITERATIVE_TOL", "1e-4")
)

# ---------------------------------------------------------------------------
# Parallel settings
# ---------------------------------------------------------------------------
clesso_config$cores <- as.integer(env_or_default(
  "CLESSO_CORES",
  as.character(max(1, parallel::detectCores() - 1))
))

## Species threshold for chunked parallel match sampling
clesso_config$species_threshold <- as.integer(env_or_default("CLESSO_SPECIES_THRESHOLD", "500"))

## Chunk size for env extraction
clesso_config$chunk_size <- as.integer(env_or_default("CLESSO_CHUNK_SIZE", "10000"))

# ---------------------------------------------------------------------------
# Environmental variable extraction parameters
# (same format as reca_STresiduals config)
# ---------------------------------------------------------------------------
clesso_config$env_params <- list(
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
# Run identifier
# ---------------------------------------------------------------------------
## Unique run_id stamped on every output file. Default format:
##   <species_group>_<YYYYMMDD_HHMMSS>
## Override with CLESSO_RUN_ID env var for reproducible naming.
clesso_config$run_id <- env_or_default(
  "CLESSO_RUN_ID",
  paste0(clesso_config$species_group, "_", format(Sys.time(), "%Y%m%d_%H%M%S"))
)

# ---------------------------------------------------------------------------
# Create output directory (run-specific sub-folder)
# ---------------------------------------------------------------------------
clesso_config$output_dir <- file.path(clesso_config$output_dir,
                                       clesso_config$run_id)
if (!dir.exists(clesso_config$output_dir)) {
  dir.create(clesso_config$output_dir, recursive = TRUE)
}

# ---------------------------------------------------------------------------
# Config snapshot helper
# ---------------------------------------------------------------------------
## Returns a plain list (no functions/environments) that can be serialised
## alongside model results so every output file records how it was produced.
clesso_snapshot_config <- function(cfg = clesso_config) {
  snap <- cfg
  ## Strip items that don't serialise cleanly
  snap$alpha_knot_positions <- if (is.null(cfg$alpha_knot_positions)) "auto" else cfg$alpha_knot_positions
  snap$snapshot_time <- Sys.time()
  snap$R_version     <- paste0(R.version$major, ".", R.version$minor)
  snap
}

cat("\n=== CLESSO v2 config loaded ===\n")
cat("  Run ID          :", clesso_config$run_id, "\n")
cat("  Species group   :", clesso_config$species_group, "\n")
cat("  Data dir        :", clesso_config$data_dir, "\n")
cat("  Output dir      :", clesso_config$output_dir, "\n")
cat("  n_within pairs  :", clesso_config$n_within, "\n")
cat("  n_between pairs :", clesso_config$n_between, "\n")
cat("  within min recs :", clesso_config$within_min_records, "\n")
cat("  balance weights :", clesso_config$balance_weights, "\n")
cat("  n_splines       :", clesso_config$n_splines, "\n")
cat("  geo_distance    :", clesso_config$geo_distance, "\n")
cat("  alpha splines   :", clesso_config$use_alpha_splines, "\n")
cat("  alpha spline typ:", clesso_config$alpha_spline_type, "\n")
cat("  alpha n_knots   :", clesso_config$alpha_n_knots, "\n")
cat("  alpha spline deg:", clesso_config$alpha_spline_deg, "\n")
cat("  alpha pen order :", clesso_config$alpha_pen_order,
    ifelse(clesso_config$alpha_spline_type == "regression", "(unused)", ""), "\n")
cat("  alpha knot pos  :", if (is.null(clesso_config$alpha_knot_positions)) "auto (quantile)"
                           else "user-specified", "\n")
cat("  alpha lb lambda :", clesso_config$alpha_lower_bound_lambda, "\n")
cat("  climate_window  :", clesso_config$climate_window, "years\n")
cat("  cores           :", clesso_config$cores, "\n")
cat("  seed            :", clesso_config$seed, "\n")
