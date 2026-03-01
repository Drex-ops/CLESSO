##############################################################################
##
## Observation-pair diagnostic plot for obsGDM (logit link version)
##
## 4-panel diagnostic: binned obs vs predicted, ecological distance vs miss
## ratio, density curves of match/mismatch, and (optional) dissimilarity
## transform validation.
##
## Ported from: OLD_RECA/code/obs.gdm.plot_logit.R
## Changes: minimal (formatting only)
##
##############################################################################

obs.gdm.plot <- function(model, title, w, Is) {
  par(mfcol = c(2, 2))

  ## Panel 1: Binned observed vs predicted miss ratio
  plot(fitted(model), model$data$Match,
       xlab = "Predicted Miss ratio", ylab = "Observed binned Miss ratio",
       type = "n", ylim = c(0, 1), xlim = c(0, 1), main = title)

  sqs <- seq(0.05, 0.95, by = 0.1)
  rat <- unlist(lapply(sqs, function(x) {
    data <- model$data$Match[fitted(model) >= (x - 0.05) & fitted(model) <= (x + 0.05)]
    length(data[data == 1]) / length(data)
  }))
  points(sqs, rat, pch = 20, cex = 1, col = rgb(0, 0, 1))
  overlayX <- overlayY <- seq(min(fitted(model)), max(fitted(model)), length = 200)
  lines(overlayX, overlayY, lwd = 1, lty = "dashed")

  ## Panel 2: Ecological distance vs binned miss ratio
  plot(model$linear.predictors, model$data$Match,
       xlab = "Predicted Ecological Distance",
       ylab = "Observed Binned Miss ratio", type = "n")

  overlayX <- seq(min(model$linear.predictors), max(model$linear.predictors), length = 200)
  overlayY <- inv.logit(overlayX)

  ecoR <- range(model$linear.predictors)
  tt <- try(seq(ecoR[1], ecoR[2], by = 0.1), silent = TRUE)
  sqs <- if (!inherits(tt, "try-error")) seq(ecoR[1], ecoR[2], by = 0.1) else seq(ecoR[1], 10, by = 0.1)

  rat <- unlist(lapply(sqs, function(x) {
    data <- model$data$Match[model$linear.predictors >= (x - 0.05) & model$linear.predictors <= (x + 0.05)]
    length(data[data == 1]) / length(data)
  }))
  points(sqs, rat, pch = 20, cex = 1, col = rgb(0, 0, 1))
  lines(overlayX, overlayY, lwd = 2, lty = "dashed", col = "green")

  ## Panel 3: Density of ecological distance for matches/mismatches
  match_dens <- density(x = model$linear.predictors[model$data$Match == 0])
  miss_dens  <- density(x = model$linear.predictors[model$data$Match == 1])
  mx   <- max(c(match_dens$y, miss_dens$y))
  xrng <- range(model$linear.predictors)
  plot(1, 1, type = "n", xlim = xrng, ylim = c(0, mx),
       xlab = "Predicted Ecological Distance", ylab = "Density")
  lines(match_dens, col = "red")
  lines(miss_dens,  col = "blue")
  legend("topright", legend = c("match", "miss"), col = c("red", "blue"), lty = 1)

  ## Panel 4: Placeholder (dissimilarity transform validation - requires
  ## Sorensen data which is not always available)
  plot.new()
  text(0.5, 0.5, "Panel 4: Dissimilarity transform\n(requires Sorensen data)", cex = 0.8)
}

# ---------------------------------------------------------------------------
# gdm.spline.plot - Pure R replacement for the Gdm01-dependent version
#
# The original version called .C("GetPredictorPlotData", ..., PACKAGE="Gdm01")
# which requires the Gdm01 Windows DLL. This version reimplements the same
# I-spline plotting in pure R using our I_spline function.
#
# Parameters:
#   model - list with $predictors, $coefficients, $quantiles, $splines
# ---------------------------------------------------------------------------
gdm.spline.plot <- function(model, plot.layout = c(2, 2),
                            plot.color = rgb(0, 0, 1),
                            plot.linewidth = 2) {
  PSAMPLE <- 200

  par(mfrow = plot.layout)

  preds      <- length(model$predictors)
  splineindex <- 1

  ## First pass: find max y for consistent axis scaling
  predmax <- 0
  si <- 1
  for (i in 1:preds) {
    ns <- model$splines[i]
    coeffs <- model$coefficients[si:(si + ns - 1)]
    if (sum(coeffs) > 0) {
      quants <- model$quantiles[si:(si + ns - 1)]
      xvals  <- seq(min(quants), max(quants), length = PSAMPLE)
      yvals  <- rep(0, PSAMPLE)
      for (sp in 1:ns) {
        if (sp == 1)       spl <- I_spline(xvals, quants[1], quants[1], quants[min(2, ns)])
        else if (sp == ns) spl <- I_spline(xvals, quants[max(1, ns - 1)], quants[ns], quants[ns])
        else               spl <- I_spline(xvals, quants[sp - 1], quants[sp], quants[sp + 1])
        yvals <- yvals + coeffs[sp] * spl
      }
      predmax <- max(predmax, max(yvals))
    }
    si <- si + ns
  }

  ## Second pass: plot each predictor
  si <- 1
  for (i in 1:preds) {
    ns     <- model$splines[i]
    coeffs <- model$coefficients[si:(si + ns - 1)]
    if (sum(coeffs) > 0) {
      quants <- model$quantiles[si:(si + ns - 1)]
      xvals  <- seq(min(quants), max(quants), length = PSAMPLE)
      yvals  <- rep(0, PSAMPLE)
      for (sp in 1:ns) {
        if (sp == 1)       spl <- I_spline(xvals, quants[1], quants[1], quants[min(2, ns)])
        else if (sp == ns) spl <- I_spline(xvals, quants[max(1, ns - 1)], quants[ns], quants[ns])
        else               spl <- I_spline(xvals, quants[sp - 1], quants[sp], quants[sp + 1])
        yvals <- yvals + coeffs[sp] * spl
      }
      plot(xvals, yvals,
           xlab = model$predictors[i],
           ylab = paste0("f(", model$predictors[i], ")"),
           ylim = c(0, predmax), type = "l",
           col = plot.color, lwd = plot.linewidth)
    }
    si <- si + ns
  }
}
