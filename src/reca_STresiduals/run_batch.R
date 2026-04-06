##############################################################################
##
## run_batch.R  --  Loop over species groups × covariate combinations
##
## Outer loop : observation datasets (species groups)
## Inner loop : covariate toggles (modis × condition)
##
## Each iteration sets environment variables so that config.R picks them up,
## then sources run_full_pipeline.R which runs the full 6-step pipeline.
##
## Usage:
##   Rscript run_batch.R
##
##############################################################################

cat("\n")
cat("###########################################################################\n")
cat("##  RECA BATCH RUN\n")
cat("###########################################################################\n\n")

batch_t0 <- proc.time()

# ---------------------------------------------------------------------------
# Resolve script directory
# ---------------------------------------------------------------------------
this_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    if (nchar(getwd()) > 0) getwd()
    else stop("Cannot determine script directory.")
  }
)

# ---------------------------------------------------------------------------
# Define species-group configurations
# ---------------------------------------------------------------------------
species_configs <- list(
  #list(obs_csv = "ala_amp_2026-03-03.csv",  species_group = "AMP",  substrate = "SUBS_brk.grd"),
  #list(obs_csv = "ala_aves_2026-03-04.csv", species_group = "AVES", substrate = "SUBS_brk_AVES.grd"),
  list(obs_csv = "ala_hym_2026-03-11.csv",  species_group = "HYM",  substrate = "SUBS_brk.grd")#,
  #list(obs_csv = "ala_mam_2026-03-03.csv",  species_group = "MAM",  substrate = "SUBS_brk.grd"),
  #list(obs_csv = "ala_rep_2026-03-03.csv",  species_group = "REP",  substrate = "SUBS_brk.grd"),
  #list(obs_csv = "ala_vas_2026-03-03.csv",  species_group = "VAS",  substrate = "SUBS_brk_VAS.grd")
)

# ---------------------------------------------------------------------------
# Define covariate combinations
# ---------------------------------------------------------------------------
covariate_combos <- list(
  list(add_modis = FALSE, add_condition = FALSE),
  #list(add_modis = TRUE,  add_condition = FALSE),
  list(add_modis = FALSE, add_condition = TRUE)
)

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
n_species <- length(species_configs)
n_combos  <- length(covariate_combos)
n_total   <- n_species * n_combos

run_log <- data.frame(
  run_num       = integer(),
  species_group = character(),
  add_modis     = logical(),
  add_condition = logical(),
  status        = character(),
  elapsed_min   = numeric(),
  stringsAsFactors = FALSE
)

run_counter <- 0

# ---------------------------------------------------------------------------
# Nested loop
# ---------------------------------------------------------------------------
for (sp in species_configs) {
  for (cv in covariate_combos) {
    run_counter <- run_counter + 1

    label <- sprintf("%s | modis=%s cond=%s",
                     sp$species_group,
                     ifelse(cv$add_modis, "Y", "N"),
                     ifelse(cv$add_condition, "Y", "N"))

    cat("\n")
    cat("###########################################################################\n")
    cat(sprintf("##  BATCH RUN %d / %d : %s\n", run_counter, n_total, label))
    cat("###########################################################################\n\n")

    ## ---- Set environment variables for config.R ----------------------------
    Sys.setenv(RECA_SPECIES_GROUP    = sp$species_group)
    Sys.setenv(RECA_OBS_CSV          = sp$obs_csv)
    Sys.setenv(RECA_SUBSTRATE_RASTER = sp$substrate)
    Sys.setenv(RECA_ADD_MODIS        = toupper(as.character(cv$add_modis)))
    Sys.setenv(RECA_ADD_CONDITION    = toupper(as.character(cv$add_condition)))

    ## Set RECA_MIN_DATE: 2000 when condition or modis data is used,
    ## otherwise 1950 (pre-satellite era data is fine without those layers)
    if (cv$add_modis || cv$add_condition) {
      Sys.setenv(RECA_MIN_DATE = "2000-01-01")
    } else {
      Sys.setenv(RECA_MIN_DATE = "1950-01-01")
    }

    ## Clear RECA_RUN_ID so each iteration gets a fresh timestamp
    Sys.unsetenv("RECA_RUN_ID")

    ## Tell run_full_pipeline.R where scripts live (avoids sys.frame issues)
    Sys.setenv(RECA_SCRIPT_DIR = this_dir)

    ## ---- Run the pipeline --------------------------------------------------
    run_t0 <- proc.time()
    status <- tryCatch({
      source(file.path(this_dir, "run_full_pipeline.R"),
             local = new.env(parent = globalenv()))
      "OK"
    }, error = function(e) {
      cat(sprintf("\n  *** BATCH ERROR: %s\n", conditionMessage(e)))
      ## Print the call stack for debugging
      calls <- sys.calls()
      if (length(calls) > 0) {
        cat("  Traceback (most recent last):\n")
        for (ci in seq_along(calls)) {
          line <- tryCatch(deparse(calls[[ci]], width.cutoff = 120)[1],
                           error = function(x) "<unparseable>")
          cat(sprintf("    %d: %s\n", ci, substr(line, 1, 120)))
        }
      }
      paste("FAILED:", conditionMessage(e))
    })
    run_elapsed <- (proc.time() - run_t0)["elapsed"]

    run_log <- rbind(run_log, data.frame(
      run_num       = run_counter,
      species_group = sp$species_group,
      add_modis     = cv$add_modis,
      add_condition = cv$add_condition,
      status        = status,
      elapsed_min   = run_elapsed / 60,
      stringsAsFactors = FALSE
    ))

    cat(sprintf("\n  [Batch run %d/%d complete: %s — %.1f min]\n",
                run_counter, n_total, status, run_elapsed / 60))

    ## ---- Flush memory between runs ----------------------------------------
    ## Remove all objects created by the pipeline in the sourced environment,
    ## drop raster temp files, and force a full garbage collection pass.
    cat("  Flushing memory ...\n")
    if (requireNamespace("raster", quietly = TRUE)) {
      raster::removeTmpFiles(h = 0)   # remove ALL raster temp files
    }
    gc(verbose = FALSE, full = TRUE)
    gc(verbose = FALSE, full = TRUE)   # second pass catches weak refs
    cat(sprintf("  Memory after GC: %.0f MB used\n",
                sum(gc(verbose = FALSE)[, 2])))
  }
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
cat("##  BATCH COMPLETE\n")
cat("###########################################################################\n\n")

cat(sprintf("  Total combinations : %d\n", n_total))
cat(sprintf("  Succeeded          : %d\n", sum(run_log$status == "OK")))
cat(sprintf("  Failed             : %d\n", sum(run_log$status != "OK")))
cat(sprintf("  Total time         : %.1f min (%.1f hr)\n\n",
            batch_time / 60, batch_time / 3600))

cat("  Run details:\n")
cat(sprintf("  %-4s  %-6s  %-6s  %-6s  %-8s  %s\n",
            "#", "Group", "MODIS", "COND", "Minutes", "Status"))
cat("  ", strrep("-", 60), "\n")
for (i in seq_len(nrow(run_log))) {
  r <- run_log[i, ]
  cat(sprintf("  %-4d  %-6s  %-6s  %-6s  %6.1f    %s\n",
              r$run_num, r$species_group,
              ifelse(r$add_modis, "Y", "N"),
              ifelse(r$add_condition, "Y", "N"),
              r$elapsed_min, r$status))
}
cat("\n")

## Save the log
log_file <- file.path(this_dir, "output", sprintf("batch_log_%s.rds",
                      format(Sys.time(), "%Y%m%dT%H%M%S")))
if (!dir.exists(dirname(log_file))) dir.create(dirname(log_file), recursive = TRUE)
saveRDS(run_log, log_file)
cat(sprintf("  Batch log saved: %s\n\n", log_file))
