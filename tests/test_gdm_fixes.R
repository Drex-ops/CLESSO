##############################################################################
## Quick smoke tests for the 2026-03-12 numerical stability fixes:
##   1. I_spline degenerate knot guards
##   2. Ridge penalty in fitGDM
##   3. nnls.fit inner-loop-3 revert
##   4. Case weights pass-through
##############################################################################

cat("\n=== Testing numerical stability fixes ===\n\n")
source("src/shared/R/gdm_functions.R")

errors <- 0

## ---- Test 1: I_spline normal case ----
x <- seq(0, 10, by = 0.5)
r <- I_spline(x, 2, 5, 8)
if (all(!is.na(r) & r >= 0 & r <= 1)) {
  cat("PASS: I_spline normal case\n")
} else {
  cat("FAIL: I_spline normal case\n"); errors <- errors + 1
}

## ---- Test 2: Fully collapsed knots ----
r2 <- I_spline(x, 5, 5, 5)
if (all(r2 == 0)) {
  cat("PASS: I_spline fully collapsed knots (q1==q2==q3) -> all zero\n")
} else {
  cat("FAIL: I_spline fully collapsed knots\n"); errors <- errors + 1
}

## ---- Test 3: Left-collapsed knots (q1==q2) ----
r3 <- I_spline(x, 3, 3, 7)
if (all(!is.na(r3) & r3 >= 0 & r3 <= 1)) {
  cat("PASS: I_spline left-collapsed (q1==q2)\n")
} else {
  cat("FAIL: I_spline left-collapsed\n"); errors <- errors + 1
}

## ---- Test 4: Right-collapsed knots (q2==q3) ----
r4 <- I_spline(x, 2, 8, 8)
if (all(!is.na(r4) & r4 >= 0 & r4 <= 1)) {
  cat("PASS: I_spline right-collapsed (q2==q3)\n")
} else {
  cat("FAIL: I_spline right-collapsed\n"); errors <- errors + 1
}

## ---- Test 5: NA passthrough ----
r5 <- I_spline(c(1, NA, 5), 2, 5, 8)
if (is.na(r5[2]) && !is.na(r5[1]) && !is.na(r5[3])) {
  cat("PASS: I_spline NA handling\n")
} else {
  cat("FAIL: I_spline NA handling\n"); errors <- errors + 1
}

## ---- Test 6: fitGDM accepts weights and lambda ----
set.seed(42)
n <- 200
x1 <- runif(n)
x2 <- runif(n)
y  <- rbinom(n, 1, plogis(-1 + 2 * abs(x1 - x2)))
spl1 <- abs(x1 - x2)
spl2 <- (abs(x1 - x2))^2
dd <- data.frame(Match = y, pred_spl1 = spl1, pred_spl2 = spl2)

## Without weights/lambda (original behaviour)
fit0 <- fitGDM(Match ~ pred_spl1 + pred_spl2, data = dd)
cat(sprintf("  fit0 D2: %.4f  coefs: %s\n", 
            1 - fit0$deviance/fit0$null.deviance,
            paste(round(coef(fit0), 3), collapse = ", ")))

## With lambda
fit1 <- fitGDM(Match ~ pred_spl1 + pred_spl2, data = dd, lambda = 1.0)
cat(sprintf("  fit1 D2: %.4f  coefs: %s (lambda=1)\n",
            1 - fit1$deviance/fit1$null.deviance,
            paste(round(coef(fit1), 3), collapse = ", ")))

## Ridge should shrink coefficients toward zero
if (sum(abs(coef(fit1)[-1])) <= sum(abs(coef(fit0)[-1]))) {
  cat("PASS: Ridge penalty shrinks coefficients\n")
} else {
  cat("FAIL: Ridge penalty did not shrink coefficients\n"); errors <- errors + 1
}

## With case weights
wts <- ifelse(y == 0, 3, 1)
wts <- wts / mean(wts)
fit2 <- fitGDM(Match ~ pred_spl1 + pred_spl2, data = dd, weights = wts)
cat(sprintf("  fit2 D2: %.4f  intercept: %.3f (weighted)\n",
            1 - fit2$deviance/fit2$null.deviance, coef(fit2)[1]))

## Weighted intercept should differ from unweighted
if (abs(coef(fit2)[1] - coef(fit0)[1]) > 0.01) {
  cat("PASS: Case weights shift intercept\n")
} else {
  cat("FAIL: Case weights did not shift intercept\n"); errors <- errors + 1
}

## Both together
fit3 <- fitGDM(Match ~ pred_spl1 + pred_spl2, data = dd,
               weights = wts, lambda = 0.5)
cat(sprintf("  fit3 D2: %.4f  coefs: %s (weighted + ridge)\n",
            1 - fit3$deviance/fit3$null.deviance,
            paste(round(coef(fit3), 3), collapse = ", ")))
if (!is.null(fit3) && fit3$converged) {
  cat("PASS: fitGDM with weights + lambda converges\n")
} else {
  cat("FAIL: fitGDM with weights + lambda did not converge\n"); errors <- errors + 1
}

## ---- Summary ----
cat(sprintf("\n=== %d tests, %d failures ===\n", 8, errors))
if (errors > 0) quit(status = 1)
