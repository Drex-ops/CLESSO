##############################################################################
##
## Core GDM functions for RECA obsGDM
##
## Recovered from the external CSIRO scripts:
##   - fitGDM.R          (I_spline, splineData, fitGDM)
##   - nnnpls.fit2_(good).R  (nnls.fit - NNLS method for glm)
##   - negexp_GDMlink.R  (negexp - negative exponential link for glm)
##
## These were originally sourced from CSIRO network drives:
##   U:\users\hos06j\R scripts\
##   Y:\MOD\G\GDM\_tools\
##
##############################################################################

# ---------------------------------------------------------------------------
# I_spline - Calculate I-Spline basis value
#
# Quadratic I-spline with 3 knots (q1, q2, q3).
# Returns 0 below q1, 1 above q3, quadratic between.
# ---------------------------------------------------------------------------
I_spline <- function(predVal, q1, q2, q3) {
  outVal <- rep(NA, length(predVal))
  outVal[predVal <= q1] <- 0
  outVal[predVal >= q3] <- 1
  outVal[predVal > q1 & predVal <= q2] <-
    ((predVal[predVal > q1 & predVal <= q2] - q1)^2) / ((q2 - q1) * (q3 - q1))
  outVal[predVal > q2 & predVal < q3] <-
    1 - ((q3 - predVal[predVal > q2 & predVal < q3])^2) / ((q3 - q2) * (q3 - q1))
  outVal
}

# ---------------------------------------------------------------------------
# splineData - Create I-spline transformed site-pair distance table
#
# Takes a matrix X where the first half of columns are site-1 values and
# the second half are site-2 values. Applies I-spline basis transformation
# to the stacked data, then returns |spline(site1) - spline(site2)|.
#
# Parameters:
#   X         - matrix/data.frame with even number of columns (site1 | site2)
#   splines   - integer vector: number of spline bases per predictor
#               (default: 3 per predictor, using 0%, 50%, 100% quantiles)
#   quantiles - numeric vector of knot positions (length = sum(splines))
#               (default: computed from data as 0%, 50%, 100% quantiles)
# ---------------------------------------------------------------------------
splineData <- function(X, splines = NULL, quantiles = NULL) {
  nc  <- ncol(X)
  nc2 <- nc / 2
  if (nc %% 2 != 0) stop("X must be a matrix with even columns")

  X1 <- X[, 1:nc2]
  X2 <- X[, (nc2 + 1):nc]
  nms <- colnames(X1)
  colnames(X2) <- nms

  ## Stack sites
  sv <- c(rep(1, nrow(X1)), rep(2, nrow(X2)))
  XX <- rbind(X1, X2)

  ## Default splines: 3 per predictor
  if (is.null(splines)) {
    message("No splines specified. Using 3 splines (0%, 50%, 100% quantiles)")
    splines <- rep(3, ncol(XX))
  }

  ## Default quantiles
  if (is.null(quantiles)) {
    if (!all(splines == 3)) stop("Must specify quantile positions if all(splines) != 3")
    quantiles <- unlist(lapply(1:ncol(XX), function(x) {
      quantile(XX[, x], c(0, 0.5, 1))
    }))
  }

  if (length(quantiles) != sum(splines)) {
    stop("Number of quantiles must equal number of splines")
  }

  ## Build spline basis
  csp <- c(0, cumsum(splines))
  out.tab <- c()
  for (col in 1:ncol(XX)) {
    ns      <- splines[col]
    predVal <- XX[, col]
    quan    <- quantiles[(csp[col] + 1):(csp[col] + ns)]
    for (sp in 1:ns) {
      if (sp == 1)              spl <- I_spline(predVal, quan[1], quan[1], quan[2])
      else if (sp == ns)        spl <- I_spline(predVal, quan[ns - 1], quan[ns], quan[ns])
      else                      spl <- I_spline(predVal, quan[sp - 1], quan[sp], quan[sp + 1])
      out.tab <- cbind(out.tab, spl)
      if (anyNA(spl)) { warning(paste("NA in spline: col", col, "spline", sp)); break }
    }
  }

  ## Name columns
  NMS   <- rep(nms, splines)
  SPNMS <- paste0("spl", unlist(lapply(splines, function(x) 1:x)))
  colnames(out.tab) <- paste(NMS, SPNMS, sep = "_")

  ## Return absolute difference between site-1 and site-2 spline values
  XX1 <- out.tab[sv == 1, ]
  XX2 <- out.tab[sv == 2, ]
  abs(XX1 - XX2)
}

# ---------------------------------------------------------------------------
# negexp - Negative exponential GDM link function for glm()
#
# Link:         g(mu)  = -log(1 - mu)
# Inverse link: g^-1(eta) = 1 - exp(-eta)
# Derivative:   d(mu)/d(eta) = exp(-eta)
# ---------------------------------------------------------------------------
negexp <- function() {
  linkfun  <- function(mu)  -log(1 - mu)
  linkinv  <- function(eta) 1 - exp(-eta)
  mu.eta   <- function(eta) exp(-eta)
  valideta <- function(eta) all(is.finite(eta))
  link     <- "negexp"
  structure(list(linkfun = linkfun, linkinv = linkinv,
                 mu.eta = mu.eta, valideta = valideta, name = link),
            class = "link-glm")
}

# ---------------------------------------------------------------------------
# nnnpls (non-negative constrained least squares)
#
# This is called internally by nnls.fit via nnls::nnnpls.
# It requires the 'nnls' package.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# nnls.fit - Non-negative least squares fitting method for glm()
#
# Drop-in replacement for glm.fit that constrains all coefficients (except
# the intercept) to be non-negative via NNLS. This is essential for GDM
# because I-spline coefficients must be >= 0 to maintain monotonicity.
#
# Usage: glm(..., method = 'nnls.fit')
# ---------------------------------------------------------------------------
nnls.fit <- function(x, y, weights = rep(1, nobs), start = NULL,
                     etastart = NULL, mustart = NULL, offset = rep(0, nobs),
                     family = gaussian(), control = list(), intercept = TRUE) {
  require(nnls)
  control  <- do.call("glm.control", control)
  x        <- as.matrix(x)
  xnames   <- dimnames(x)[[2L]]
  ynames   <- if (is.matrix(y)) rownames(y) else names(y)
  conv     <- FALSE
  nobs     <- NROW(y)
  nvars    <- ncol(x)
  EMPTY    <- nvars == 0

  if (is.null(weights)) weights <- rep.int(1, nobs)
  if (is.null(offset))  offset  <- rep.int(0, nobs)

  variance <- family$variance
  linkinv  <- family$linkinv
  if (!is.function(variance) || !is.function(linkinv))
    stop("'family' argument seems not to be a valid family object", call. = FALSE)
  dev.resids <- family$dev.resids
  aic        <- family$aic
  mu.eta     <- family$mu.eta
  unless.null <- function(x, if.null) if (is.null(x)) if.null else x
  valideta   <- unless.null(family$valideta, function(eta) TRUE)
  validmu    <- unless.null(family$validmu,  function(mu)  TRUE)

  if (is.null(mustart)) {
    eval(family$initialize)
  } else {
    mukeep <- mustart
    eval(family$initialize)
    mustart <- mukeep
  }

  if (EMPTY) {
    eta <- rep.int(0, nobs) + offset
    if (!valideta(eta)) stop("invalid linear predictor values in empty model", call. = FALSE)
    mu <- linkinv(eta)
    if (!validmu(mu)) stop("invalid fitted means in empty model", call. = FALSE)
    dev       <- sum(dev.resids(y, mu, weights))
    w         <- ((weights * mu.eta(eta)^2) / variance(mu))^0.5
    residuals <- (y - mu) / mu.eta(eta)
    good      <- rep(TRUE, length(residuals))
    boundary  <- conv <- TRUE
    coef      <- numeric()
    iter      <- 0L
  } else {
    coefold <- NULL
    eta <- if (!is.null(etastart)) etastart
           else if (!is.null(start)) {
             if (length(start) != nvars)
               stop(gettextf("length of 'start' should equal %d", nvars), domain = NA)
             coefold <- start
             offset + as.vector(if (NCOL(x) == 1L) x * start else x %*% start)
           } else family$linkfun(mustart)
    mu <- linkinv(eta)
    if (!(validmu(mu) && valideta(eta)))
      stop("cannot find valid starting values: please specify some", call. = FALSE)
    devold   <- sum(dev.resids(y, mu, weights))
    boundary <- conv <- FALSE

    for (iter in 1L:control$maxit) {
      good <- weights > 0
      varmu <- variance(mu)[good]
      if (any(is.na(varmu))) stop("NAs in V(mu)")
      if (any(varmu == 0))   stop("0s in V(mu)")
      mu.eta.val <- mu.eta(eta)
      if (any(is.na(mu.eta.val[good]))) stop("NAs in d(mu)/d(eta)")
      good <- (weights > 0) & (mu.eta.val != 0)
      if (all(!good)) { conv <- FALSE; warning("no observations informative at iteration ", iter); break }

      z <- (eta - offset)[good] + (y - mu)[good] / mu.eta.val[good]
      w <- sqrt((weights[good] * mu.eta.val[good]^2) / variance(mu)[good])
      ngoodobs <- as.integer(nobs - sum(!good))

      ## NNLS: constrain all coefficients except intercept to be >= 0
      fit <- nnnpls(x[good, , drop = FALSE] * w, z * w,
                    con = c(-1, rep(1, ncol(x) - 1)))
      fit$coefficients <- fit$x

      ## QR decomposition for diagnostics
      QR         <- qr(x[good, , drop = FALSE] * w, tol = min(1e-07, control$epsilon / 1000))
      fit$qr     <- QR$qr
      fit$rank   <- QR$rank
      fit$pivot  <- QR$pivot
      fit$qraux  <- QR$qraux
      fit$effects <- fit$fitted

      if (any(!is.finite(fit$coefficients))) {
        conv <- FALSE
        warning(gettextf("non-finite coefficients at iteration %d", iter), domain = NA)
        break
      }
      if (nobs < fit$rank)
        stop(gettextf("X matrix has rank %d, but only %d observations", fit$rank, nobs), domain = NA)

      start[fit$pivot] <- fit$coefficients
      eta <- drop(x %*% start)
      mu  <- linkinv(eta <- eta + offset)
      dev <- sum(dev.resids(y, mu, weights))

      if (control$trace) cat("Deviance =", dev, "Iterations -", iter, "\n")

      boundary <- FALSE
      if (!is.finite(dev)) {
        if (is.null(coefold))
          stop("no valid set of coefficients has been found: please supply starting values", call. = FALSE)
        warning("step size truncated due to divergence", call. = FALSE)
        ii <- 1
        while (!is.finite(dev)) {
          if (ii > control$maxit) stop("inner loop 1; cannot correct step size", call. = FALSE)
          ii    <- ii + 1
          start <- (start + coefold) / 2
          eta   <- drop(x %*% start)
          mu    <- linkinv(eta <- eta + offset)
          dev   <- sum(dev.resids(y, mu, weights))
        }
        boundary <- TRUE
        if (control$trace) cat("Step halved: new deviance =", dev, "\n")
      }

      if (!(valideta(eta) && validmu(mu))) {
        if (is.null(coefold))
          stop("no valid set of coefficients has been found: please supply starting values", call. = FALSE)
        warning("step size truncated: out of bounds", call. = FALSE)
        ii <- 1
        while (!(valideta(eta) && validmu(mu))) {
          if (ii > control$maxit) stop("inner loop 2; cannot correct step size", call. = FALSE)
          ii    <- ii + 1
          start <- (start + coefold) / 2
          eta   <- drop(x %*% start)
          mu    <- linkinv(eta <- eta + offset)
        }
        boundary <- TRUE
        dev <- sum(dev.resids(y, mu, weights))
        if (control$trace) cat("Step halved: new deviance =", dev, "\n")
      }

      if (((dev - devold) / (0.1 + abs(dev)) >= control$epsilon) & (iter > 1)) {
        if (is.null(coefold))
          stop("no valid set of coefficients has been found: please supply starting values", call. = FALSE)
        warning("step size truncated due to increasing deviance", call. = FALSE)
        ii <- 1
        while ((dev - devold) / (0.1 + abs(dev)) > -control$epsilon) {
          if (ii > control$maxit) break
          ii    <- ii + 1
          start <- (start + coefold) / 2
          eta   <- drop(x %*% start)
          mu    <- linkinv(eta <- eta + offset)
          dev   <- sum(dev.resids(y, mu, weights))
        }
        if (ii > control$maxit) warning("inner loop 3; cannot correct step size")
        else if (control$trace) cat("Step halved: new deviance =", dev, "\n")
      }

      if (abs(dev - devold) / (0.1 + abs(dev)) < control$epsilon) {
        conv <- TRUE
        coef <- start
        break
      } else {
        devold  <- dev
        coef    <- coefold <- start
      }
    }

    if (!conv) warning("nnls.fit: algorithm did not converge. Try increasing the maximum iterations", call. = FALSE)
    if (boundary) warning("nnls.fit: algorithm stopped at boundary value", call. = FALSE)

    eps <- 10 * .Machine$double.eps
    if (family$family == "binomial") {
      if (any(mu > 1 - eps) || any(mu < eps))
        warning("nnls.fit: fitted probabilities numerically 0 or 1 occurred", call. = FALSE)
    }
    if (family$family == "poisson") {
      if (any(mu < eps))
        warning("nnls.fit: fitted rates numerically 0 occurred", call. = FALSE)
    }

    if (fit$rank < nvars) coef[fit$pivot][seq.int(fit$rank + 1, nvars)] <- NA
    xxnames   <- xnames[fit$pivot]
    residuals <- (y - mu) / mu.eta(eta)
    fit$qr    <- as.matrix(fit$qr)
    nr        <- min(sum(good), nvars)
    if (nr < nvars) {
      Rmat <- diag(nvars)
      Rmat[1L:nr, 1L:nvars] <- fit$qr[1L:nr, 1L:nvars]
    } else {
      Rmat <- fit$qr[1L:nvars, 1L:nvars]
    }
    Rmat <- as.matrix(Rmat)
    Rmat[row(Rmat) > col(Rmat)] <- 0
    names(coef)        <- xnames
    colnames(fit$qr)   <- xxnames
    dimnames(Rmat)     <- list(xxnames, xxnames)
  }

  names(residuals) <- ynames
  names(mu)        <- ynames
  names(eta)       <- ynames
  wt               <- rep.int(0, nobs)
  wt[good]         <- w^2
  names(wt)        <- ynames
  names(weights)   <- ynames
  names(y)         <- ynames
  if (!EMPTY)
    names(fit$effects) <- c(xxnames[seq_len(fit$rank)], rep.int("", sum(good) - fit$rank))

  wtdmu   <- if (intercept) sum(weights * y) / sum(weights) else linkinv(offset)
  nulldev <- sum(dev.resids(y, wtdmu, weights))
  n.ok    <- nobs - sum(weights == 0)
  nulldf  <- n.ok - as.integer(intercept)
  rank    <- if (EMPTY) 0 else fit$rank
  resdf   <- n.ok - rank
  aic.model <- aic(y, n, mu, weights, dev) + 2 * rank

  list(coefficients = coef, residuals = residuals, fitted.values = mu,
       effects = if (!EMPTY) fit$effects, R = if (!EMPTY) Rmat,
       rank = rank,
       qr = if (!EMPTY) structure(fit[c("qr", "rank", "qraux", "pivot", "tol")], class = "qr"),
       family = family, linear.predictors = eta, deviance = dev,
       aic = aic.model, null.deviance = nulldev, iter = iter,
       weights = wt, prior.weights = weights, df.residual = resdf,
       df.null = nulldf, y = y, converged = conv, boundary = boundary)
}

# ---------------------------------------------------------------------------
# fitGDM - Fit a GDM using GLM with NNLS constraints
#
# The run scripts use a logit link (family = binomial()). The original
# fitGDM.R used the negexp link. Both versions are supported here via
# the `link` parameter.
#
# Parameters:
#   formula - model formula
#   data    - data.frame of Match ~ spline predictors
#   link    - "logit" (default, matches run scripts) or "negexp"
# ---------------------------------------------------------------------------
fitGDM <- function(formula = NULL, data = NULL, link = "logit") {
  fam <- if (link == "negexp") {
    binomial(link = negexp())
  } else {
    binomial()
  }
  fit <- glm(formula, family = fam, data = data,
             control = list(maxit = 500), method = "nnls.fit")
  fit
}
