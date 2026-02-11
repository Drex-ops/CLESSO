##############################################################################
##
## gen_windows - Climate window extraction (R-to-Python bridge)
##
## Writes obs-pair coordinates to a feather file, calls pyper.py via
## system(), reads back the extracted climate window summaries.
##
## Merged from: OLD_RECA/code/dynowindow.R and OLD_RECA/code/gen_windows2.r
## Changes:
##   - Removed hardcoded CSIRO UNC paths
##   - All paths (python exe, pyper.py, temp dir, npy source) read from config
##   - Uses arrow::write_feather / arrow::read_feather (feather pkg is archived)
##   - Cleans up temp files
##
##############################################################################

gen_windows <- function(pairs, variables, mstat, cstat, window,
                        pairs_dst = NULL, npy_src = NULL,
                        start_year = NULL,
                        python_exe = NULL,
                        pyper_script = NULL,
                        feather_tmpdir = NULL) {

  ## Resolve paths from config if not passed directly
  if (!exists("config", envir = .GlobalEnv) && (is.null(python_exe) || is.null(pyper_script))) {
    stop("Either pass python_exe/pyper_script arguments, or ensure 'config' is loaded in the global environment.")
  }
  cfg <- if (exists("config", envir = .GlobalEnv)) get("config", envir = .GlobalEnv) else list()

  exe          <- if (!is.null(python_exe))    python_exe    else cfg$python_exe
  pyfile       <- if (!is.null(pyper_script))  pyper_script  else cfg$pyper_script
  tmpdir       <- if (!is.null(feather_tmpdir)) feather_tmpdir else cfg$feather_tmpdir
  src          <- if (!is.null(npy_src))       npy_src       else cfg$npy_src
  start_year   <- if (!is.null(start_year))    start_year    else cfg$geonpy_start_year

  if (is.null(exe) || exe == "") stop("Python executable path not configured.")
  if (is.null(pyfile) || !file.exists(pyfile)) stop(paste("pyper.py not found at:", pyfile))

  require(arrow)

  type_pairs <- class(pairs)
  if (type_pairs != "character") {
    ## data.frame-like input
    if (is.null(pairs_dst)) {
      pairs_dst <- tempfile(tmpdir = tmpdir, fileext = ".feather")
    }

    col_class <- rep(c("numeric", "numeric", "integer", "integer"), 2)
    for (i in 1:8) {
      class(pairs[, i]) <- col_class[i]
    }

    min_year <- min(c(pairs[, 3], pairs[, 7]))
    if ((min_year - window) < start_year) {
      stop(sprintf("Found year: %s. Cannot build climate windows before %s",
                   min_year, start_year))
    }

    arrow::write_feather(as.data.frame(pairs), pairs_dst)
  } else {
    pairs_dst <- pairs
  }

  ## Build command line call
  if (Sys.info()["sysname"] == "Windows") {
    pyfile <- gsub("/", "\\\\", pyfile)
  }

  variables_str <- paste(variables, collapse = " ")

  call <- sprintf('%s "%s" -f %s -s %s -m %s -e %s -w %s',
                  exe, pyfile, pairs_dst, cstat, mstat, variables_str, window)

  if (!is.null(src)) {
    call <- sprintf("%s -src %s", call, src)
  }

  ## Execute Python
  output_fp <- system(call, intern = TRUE)

  ## Read result
  output <- tryCatch({
    arrow::read_feather(output_fp)
  }, error = function(e) e)

  if (!length(grep("data.frame", class(output)))) {
    stop(sprintf("Could not read output file. Error: %s", output_fp))
  }

  ## Clean up temp files
  tryCatch(file.remove(output_fp), error = function(e) NULL)
  if (type_pairs != "character") {
    tryCatch(file.remove(pairs_dst), error = function(e) NULL)
  }

  as.data.frame(output)
}
