##############################################################################
##
## Shared utility functions for RECA obsGDM
##
## Extracted from the inline definitions duplicated across all run scripts.
## Source this file once; all run/prediction scripts use these.
##
##############################################################################

# ---------------------------------------------------------------------------
# Load required packages
# ---------------------------------------------------------------------------
load_packages <- function() {
  pkgs <- c("raster", "nnls", "lubridate", "data.table",
            "doSNOW", "foreach", "parallel", "Matrix")
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(paste("Required package not installed:", pkg))
    }
    library(pkg, character.only = TRUE)
  }
}

# ---------------------------------------------------------------------------
# bySum replacement (ffbase is archived from CRAN)
#
# Drop-in replacement for ffbase::bySum using data.table.
# Counts / sums `x` grouped by `by`, returning results in factor-level order.
# ---------------------------------------------------------------------------
bySum <- function(x, by) {
  if (!is.factor(by)) by <- as.factor(by)
  dt <- data.table::data.table(x = x, by = by)
  agg <- dt[, .(s = sum(x)), keyby = by]
  # Return in same order as factor levels
  agg$s
}

# ---------------------------------------------------------------------------
# RsqGLM - Pseudo-R² measures for GLMs
#
# Based on Nagelkerke (1991) and Allison (2014).
# Version 1.2 (3 Jan 2015) - originally from run_obsGDM scripts.
# ---------------------------------------------------------------------------
RsqGLM <- function(obs = NULL, pred = NULL, model = NULL) {
  model.provided <- !is.null(model)

  if (model.provided) {
    if (!("glm" %in% class(model))) stop("'model' must be of class 'glm'.")
    obs  <- model$y
    pred <- model$fitted.values
  } else {
    if (is.null(obs) || is.null(pred))
      stop("Provide either 'obs' and 'pred', or a 'model' of class 'glm'.")
    if (length(obs) != length(pred))
      stop("'obs' and 'pred' must be the same length.")
    logit <- log(pred / (1 - pred))
    model <- glm(obs ~ logit, family = "binomial")
  }

  null.mod   <- glm(obs ~ 1, family = family(model))
  loglike.M  <- as.numeric(logLik(model))
  loglike.0  <- as.numeric(logLik(null.mod))
  N          <- length(obs)

  CoxSnell   <- 1 - exp(-(2 / N) * (loglike.M - loglike.0))
  Nagelkerke <- CoxSnell / (1 - exp((2 * N^(-1)) * loglike.0))
  McFadden   <- 1 - (loglike.M / loglike.0)
  Tjur       <- mean(pred[obs == 1]) - mean(pred[obs == 0])
  sqPearson  <- cor(obs, pred)^2

  list(CoxSnell = CoxSnell, Nagelkerke = Nagelkerke,
       McFadden = McFadden, Tjur = Tjur, sqPearson = sqPearson)
}

# ---------------------------------------------------------------------------
# inv.logit - Inverse logit (logistic) function
# ---------------------------------------------------------------------------
inv.logit <- function(x) {
  exp(x) / (1 + exp(x))
}

# ---------------------------------------------------------------------------
# ObsTrans - Transform observation match probability to dissimilarity
#
# Parameters:
#   p0 - baseline match probability (intercept on probability scale)
#   w  - mismatch/match ratio weight
#   p  - predicted match probability vector
# ---------------------------------------------------------------------------
ObsTrans <- function(p0, w, p) {
  prw  <- (p * w) / ((1 - p) + (p * w))
  p0w  <- (p0 * w) / ((1 - p0) + (p0 * w))
  out  <- 1 - ((1 - prw) / (1 - p0w))
  list(prw = prw, out = out)
}

# ---------------------------------------------------------------------------
# estimate_w - Estimate mismatch/match ratio from observation data
#
# Parameters:
#   datRED    - aggregated observation data.frame (from siteAggregator)
#   nSamples  - number of random pairs to draw (default 8e6)
# ---------------------------------------------------------------------------
estimate_w <- function(datRED, nSamples = 8e6) {
  idx <- 1:nrow(datRED)
  s1 <- sample(idx, nSamples, replace = TRUE)
  s2 <- sample(idx, nSamples, replace = TRUE)
  miss <- datRED$gen_spec[s1] != datRED$gen_spec[s2]

  missCount  <- sum(miss)
  matchCount <- nSamples - missCount
  propMiss   <- missCount / nSamples
  propMatch  <- matchCount / nSamples
  w <- propMiss / propMatch

  cat(sprintf("Estimated w (miss/match ratio): %.4f\n", w))
  cat(sprintf("  miss proportion : %.4f\n", propMiss))
  cat(sprintf("  match proportion: %.4f\n", propMatch))

  w
}

# ---------------------------------------------------------------------------
# make_save_prefix - Construct standardised output file prefix
# ---------------------------------------------------------------------------
make_save_prefix <- function(config, c_yr, w_yr) {
  prefix <- paste0(config$species_group, "_",
                   format(config$nMatch / 1e6, nsmall = 0), "mil_",
                   c_yr, "climYr_",
                   w_yr, "weathYr_")
  if (config$biAverage) prefix <- paste0(prefix, "biAverage_")
  if (config$decomposition != "none") prefix <- paste0(config$decomposition, "_", prefix)
  prefix
}
