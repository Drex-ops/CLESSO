##############################################################################
##
## run_full_pipeline.R  --  End-to-end RECA obsGDM pipeline
##
## Runs the following steps in sequence, all writing to the same unique
## run folder under ./output/:
##
##   1. run_obsGDM.R                         — fit the GDM
##   2. predict_temporal_raster.R            — temporal dissimilarity map
##   3. predict_spatial_2017_PCA_vs_MDS.R    — spatial PCA vs LMDS map
##   4. map_ibra_dissimilarity.R             — IBRA region summary
##   5. test_predict_temporal_timeseries.R   — time-series diagnostics
##
## Usage:
##   1. Edit config.R (or set environment variables) for your run
##   2. Rscript run_full_pipeline.R
##      -or-  source("run_full_pipeline.R")
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  RECA Full Pipeline\n")
cat("###########################################################################\n\n")

pipeline_t0 <- proc.time()

# ---------------------------------------------------------------------------
# Resolve script directory
# ---------------------------------------------------------------------------
this_dir <- tryCatch({
  ## Prefer explicit env var (set by run_batch.R or user)
  env_dir <- Sys.getenv("RECA_SCRIPT_DIR", unset = "")
  if (nchar(env_dir) > 0) env_dir
  else dirname(sys.frame(1)$ofile)
}, error = function(e) {
    if (nchar(getwd()) > 0) getwd()
    else stop("Cannot determine script directory.")
  }
)

# ---------------------------------------------------------------------------
# Source config once to lock in the run ID
# ---------------------------------------------------------------------------
source(file.path(this_dir, "config.R"))
run_id <- config$run_id

cat(sprintf("  Run ID           : %s\n", run_id))
cat(sprintf("  Run output dir   : %s\n", config$run_output_dir))
cat(sprintf("  Species group    : %s\n", config$species_group))
cat(sprintf("  add_modis        : %s\n", config$add_modis))
cat(sprintf("  add_condition    : %s\n", config$add_condition))
cat("\n")

## Pin RECA_RUN_ID so every subsequent script that re-sources config.R
## reuses the same run folder instead of generating a new timestamp.
Sys.setenv(RECA_RUN_ID = run_id)

# ---------------------------------------------------------------------------
# Helper: run a pipeline step in a clean-ish environment
# ---------------------------------------------------------------------------
run_step <- function(step_num, label, script_file) {
  cat("\n")
  cat("###########################################################################\n")
  cat(sprintf("##  Step %d: %s\n", step_num, label))
  cat("###########################################################################\n\n")

  script_path <- file.path(this_dir, script_file)
  if (!file.exists(script_path)) stop(paste("Script not found:", script_path))

  step_t0 <- proc.time()
  source(script_path, local = new.env(parent = globalenv()))
  step_time <- (proc.time() - step_t0)["elapsed"]

  cat(sprintf("\n  [Step %d complete: %.1f min]\n", step_num, step_time / 60))
  invisible(step_time)
}

# ===========================================================================
# Pipeline steps
# ===========================================================================

step_times <- numeric(6)

step_times[1] <- run_step(1, "Fit GDM (run_obsGDM.R)",
                           "run_obsGDM.R")

step_times[2] <- run_step(2, "Predict temporal raster (predict_temporal_raster.R)",
                           "predict_temporal_raster.R")

step_times[3] <- run_step(3, "Predict spatial PCA vs MDS (predict_spatial_2017_PCA_vs_MDS.R)",
                           "predict_spatial_2017_PCA_vs_MDS.R")

step_times[4] <- run_step(4, "Map IBRA dissimilarity (map_ibra_dissimilarity.R)",
                           "map_ibra_dissimilarity.R")

step_times[5] <- run_step(5, "Time-series diagnostics (test_predict_temporal_timeseries.R)",
                           "test_predict_temporal_timeseries.R")

step_times[6] <- run_step(6, "Time-series diagnostics IBRA (test_predict_temporal_timeseries_ibra.R)",
                           "test_predict_temporal_timeseries_ibra.R")

# ===========================================================================
# Summary
# ===========================================================================
total_time <- (proc.time() - pipeline_t0)["elapsed"]

cat("\n")
cat("###########################################################################\n")
cat("##  PIPELINE COMPLETE\n")
cat("###########################################################################\n")
cat(sprintf("  Run ID         : %s\n", run_id))
cat(sprintf("  Output folder  : %s\n", file.path("output", run_id)))
cat(sprintf("  Total time     : %.1f min\n", total_time / 60))
cat("\n  Step timings:\n")

step_labels <- c(
  "Fit GDM",
  "Predict temporal raster",
  "Predict spatial PCA vs MDS",
  "Map IBRA dissimilarity",
  "Time-series diagnostics",
  "Time-series diagnostics IBRA"
)
for (i in seq_along(step_times)) {
  cat(sprintf("    %d. %-35s %6.1f min\n", i, step_labels[i], step_times[i] / 60))
}
cat(sprintf("    %-37s %6.1f min\n", "TOTAL", total_time / 60))
cat("\n")
