##############################################################################
##
## clesso_progress_logger.R -- Progress-logging wrapper for nlminb
##
## Creates a wrapper around TMB's obj$fn and obj$gr that logs each
## evaluation to a file in real time. You can monitor the log from
## another terminal with:
##
##   tail -f <output_dir>/clesso_progress_<run_id>.log
##
## The log is a tab-separated file with columns:
##   timestamp  iter  n_eval  objective  grad_max  delta_obj  elapsed_sec
##
## Usage:
##   logger <- clesso_make_logger(obj, log_path, print_every = 10)
##
##   fit <- nlminb(
##     start     = obj$par,
##     objective = logger$fn,
##     gradient  = logger$gr,
##     control   = list(...)
##   )
##
##   logger$close()   # flush and close the log file
##
##############################################################################

# ---------------------------------------------------------------------------
# clesso_make_logger
#
# Wraps TMB obj$fn and obj$gr with progress logging.
#
# Arguments:
#   obj           - TMB MakeADFun object
#   log_path      - file path for the progress log
#   print_every   - print a summary to console every N function evaluations
#                   (default 10; set 0 to suppress console output)
#   phase_label   - optional label prepended to log lines (e.g. "joint",
#                   "beta_cycle3")
#
# Returns:
#   list with:
#     fn       - wrapped objective function
#     gr       - wrapped gradient function
#     close    - function to flush/close the log
#     summary  - function returning a data.frame of the trace so far
# ---------------------------------------------------------------------------
clesso_make_logger <- function(obj, log_path,
                               print_every = 10L,
                               phase_label = "") {

  ## State variables (shared via closure)
  n_fn   <- 0L
  n_gr   <- 0L
  n_iter <- 0L       # incremented each time gradient is called
  best_obj  <- Inf
  prev_obj  <- NA_real_
  t_start   <- proc.time()["elapsed"]

  ## Open log file (append if it already exists from a previous phase)
  log_con <- file(log_path, open = "a")

  ## Write header if file is empty/new
  if (file.info(log_path)$size <= 1) {
    writeLines(paste0(
      "timestamp\tphase\titer\tn_fn\tn_gr\tobjective\t",
      "grad_max\tdelta_obj\tbest_obj\telapsed_sec"
    ), con = log_con)
    flush(log_con)
  }

  ## Internal: write one log line
  write_log <- function(obj_val, grad_max) {
    delta <- if (is.na(prev_obj)) NA_real_ else obj_val - prev_obj
    elapsed <- as.numeric(proc.time()["elapsed"] - t_start, units = "secs")
    line <- sprintf(
      "%s\t%s\t%d\t%d\t%d\t%.6f\t%.4e\t%.4e\t%.6f\t%.1f",
      format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      phase_label,
      n_iter,
      n_fn,
      n_gr,
      obj_val,
      if (is.na(grad_max)) 0 else grad_max,
      if (is.na(delta)) 0 else delta,
      best_obj,
      elapsed
    )
    writeLines(line, con = log_con)
    flush(log_con)
  }

  ## Wrapped fn
  logged_fn <- function(par) {
    val <- obj$fn(par)
    n_fn <<- n_fn + 1L

    if (is.finite(val) && val < best_obj) best_obj <<- val

    ## Only log on fn calls (gradient logged separately)
    val
  }

  ## Wrapped gr
  logged_gr <- function(par) {
    g <- obj$gr(par)
    n_gr <<- n_gr + 1L
    n_iter <<- n_iter + 1L

    ## Current objective (use last fn value from TMB cache)
    cur_obj <- tryCatch(obj$fn(par), error = function(e) NA_real_)
    grad_max <- max(abs(g))

    ## Log every gradient evaluation
    write_log(cur_obj, grad_max)

    ## Console output
    if (print_every > 0 && (n_iter %% print_every == 0 || n_iter == 1)) {
      elapsed <- as.numeric(proc.time()["elapsed"] - t_start, units = "secs")
      delta_str <- if (is.na(prev_obj)) "   NA   " else sprintf("%+.4e", cur_obj - prev_obj)
      cat(sprintf(
        "  [%s] iter=%d  fn=%d  obj=%.4f  delta=%s  |grad|=%.3e  (%.0fs)\n",
        format(Sys.time(), "%H:%M:%S"),
        n_iter, n_fn, cur_obj, delta_str, grad_max, elapsed
      ))
    }

    prev_obj <<- cur_obj
    g
  }

  ## Close
  close_log <- function() {
    elapsed <- as.numeric(proc.time()["elapsed"] - t_start, units = "secs")
    writeLines(sprintf(
      "# %s  DONE  phase=%s  iters=%d  fn_evals=%d  gr_evals=%d  best_obj=%.6f  elapsed=%.0fs",
      format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      phase_label, n_iter, n_fn, n_gr, best_obj, elapsed
    ), con = log_con)
    flush(log_con)
    close(log_con)
  }

  ## Summary
  get_summary <- function() {
    data.frame(
      phase      = phase_label,
      iterations = n_iter,
      fn_evals   = n_fn,
      gr_evals   = n_gr,
      best_obj   = best_obj,
      elapsed_s  = as.numeric(proc.time()["elapsed"] - t_start)
    )
  }

  list(
    fn      = logged_fn,
    gr      = logged_gr,
    close   = close_log,
    summary = get_summary
  )
}
