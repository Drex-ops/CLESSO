##############################################################################
##
## run_cond_timeseries.R
##
## Re-run ONLY the timeseries prediction steps (Steps 5 & 6 from the
## pipeline) for existing COND run folders that already have fitted GDMs
## but are missing timeseries results.
##
## This sets the appropriate environment variables so that config.R picks
## up the correct species group, condition flag, and run ID, then sources
## the timeseries scripts.
##
## Usage:
##   Rscript run_cond_timeseries.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  Re-run timeseries steps for COND run folders\n")
cat("###########################################################################\n\n")

this_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    if (nchar(getwd()) > 0) getwd()
    else stop("Cannot determine script directory.")
  }
)

output_dir <- file.path(this_dir, "output")

# ---------------------------------------------------------------------------
# Define COND run folders and their species configs
# ---------------------------------------------------------------------------
cond_runs <- list(
  list(run_id = "AMP_COND_20260312T070314",  species_group = "AMP",  obs_csv = "ala_amp_2026-03-03.csv",  substrate = "SUBS_brk.grd"),
  list(run_id = "AVES_COND_20260312T071316", species_group = "AVES", obs_csv = "ala_aves_2026-03-04.csv", substrate = "SUBS_brk_AVES.grd"),
  list(run_id = "MAM_COND_20260312T075249",  species_group = "MAM",  obs_csv = "ala_mam_2026-03-03.csv",  substrate = "SUBS_brk.grd"),
  list(run_id = "REP_COND_20260312T080417",  species_group = "REP",  obs_csv = "ala_rep_2026-03-03.csv",  substrate = "SUBS_brk.grd"),
  list(run_id = "VAS_COND_20260312T081417",  species_group = "VAS",  obs_csv = "ala_vas_2026-03-03.csv",  substrate = "SUBS_brk_VAS.grd")
)

n_total <- length(cond_runs)
run_log <- data.frame(
  run_num       = integer(),
  species_group = character(),
  status        = character(),
  elapsed_min   = numeric(),
  stringsAsFactors = FALSE
)

batch_t0 <- proc.time()

for (ri in seq_along(cond_runs)) {
  cr <- cond_runs[[ri]]

  ## Check the run folder exists and has a fitted GDM
  run_dir <- file.path(output_dir, cr$run_id)
  if (!dir.exists(run_dir)) {
    cat(sprintf("\n  [SKIP] Run folder not found: %s\n", cr$run_id))
    run_log <- rbind(run_log, data.frame(
      run_num = ri, species_group = cr$species_group,
      status = "SKIP: folder missing", elapsed_min = 0, stringsAsFactors = FALSE))
    next
  }

  ## Check if timeseries results already exist
  existing_ts <- list.files(run_dir, pattern = "_test_timeseries_results\\.rds$")
  if (length(existing_ts) > 0) {
    cat(sprintf("\n  [SKIP] Timeseries already exists for %s: %s\n", cr$run_id, existing_ts[1]))
    run_log <- rbind(run_log, data.frame(
      run_num = ri, species_group = cr$species_group,
      status = "SKIP: already done", elapsed_min = 0, stringsAsFactors = FALSE))
    next
  }

  cat("\n")
  cat("###########################################################################\n")
  cat(sprintf("##  %d / %d : %s (timeseries steps)\n", ri, n_total, cr$run_id))
  cat("###########################################################################\n\n")

  ## Set environment variables
  Sys.setenv(RECA_SPECIES_GROUP    = cr$species_group)
  Sys.setenv(RECA_OBS_CSV          = cr$obs_csv)
  Sys.setenv(RECA_SUBSTRATE_RASTER = cr$substrate)
  Sys.setenv(RECA_ADD_MODIS        = "FALSE")
  Sys.setenv(RECA_ADD_CONDITION    = "TRUE")
  Sys.setenv(RECA_MIN_DATE         = "2000-01-01")
  Sys.setenv(RECA_RUN_ID           = cr$run_id)
  Sys.setenv(RECA_SCRIPT_DIR       = this_dir)

  run_t0 <- proc.time()
  status <- tryCatch({
    ## Step 5: Site-level timeseries
    cat("--- Running Step 5: test_predict_temporal_timeseries.R ---\n\n")
    source(file.path(this_dir, "test_predict_temporal_timeseries.R"),
           local = new.env(parent = globalenv()))

    ## Step 6: IBRA timeseries
    cat("\n--- Running Step 6: test_predict_temporal_timeseries_ibra.R ---\n\n")
    source(file.path(this_dir, "test_predict_temporal_timeseries_ibra.R"),
           local = new.env(parent = globalenv()))

    "OK"
  }, error = function(e) {
    cat(sprintf("\n  *** ERROR: %s\n", conditionMessage(e)))
    paste("FAILED:", conditionMessage(e))
  })

  run_elapsed <- (proc.time() - run_t0)["elapsed"]
  run_log <- rbind(run_log, data.frame(
    run_num = ri, species_group = cr$species_group,
    status = status, elapsed_min = run_elapsed / 60, stringsAsFactors = FALSE))

  cat(sprintf("\n  [%s complete: %s — %.1f min]\n", cr$run_id, status, run_elapsed / 60))

  ## Flush memory
  if (requireNamespace("raster", quietly = TRUE)) raster::removeTmpFiles(h = 0)
  gc(verbose = FALSE, full = TRUE)
  gc(verbose = FALSE, full = TRUE)
}

# ---------------------------------------------------------------------------
# Clean up env vars
# ---------------------------------------------------------------------------
for (v in c("RECA_SPECIES_GROUP", "RECA_OBS_CSV", "RECA_SUBSTRATE_RASTER",
            "RECA_ADD_MODIS", "RECA_ADD_CONDITION", "RECA_MIN_DATE",
            "RECA_RUN_ID", "RECA_SCRIPT_DIR")) {
  Sys.unsetenv(v)
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
batch_time <- (proc.time() - batch_t0)["elapsed"]

cat("\n")
cat("###########################################################################\n")
cat("##  COND TIMESERIES RE-RUN COMPLETE\n")
cat("###########################################################################\n\n")

cat(sprintf("  Total: %d | OK: %d | Failed: %d | Skipped: %d\n",
            n_total,
            sum(run_log$status == "OK"),
            sum(grepl("^FAILED", run_log$status)),
            sum(grepl("^SKIP", run_log$status))))
cat(sprintf("  Total time: %.1f min\n\n", batch_time / 60))

for (i in seq_len(nrow(run_log))) {
  r <- run_log[i, ]
  cat(sprintf("  %d  %-6s  %6.1f min  %s\n",
              r$run_num, r$species_group, r$elapsed_min, r$status))
}
cat("\n")
