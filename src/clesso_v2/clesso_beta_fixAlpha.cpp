// clesso_beta_fixAlpha.cpp — TMB template for beta-only (turnover) model
//                            with fixed (pre-estimated) alpha values
//
// Companion to clesso_alpha.cpp. This template takes alpha (richness) as
// FIXED DATA from a prior estimation step and estimates only the turnover
// (beta) coefficients from between-site observation pairs.
//
// Model:
//   eta_{i,j} = eta0 + X_{i,j} * beta       [linear predictor for turnover]
//   S_{i,j}   = exp(-eta_{i,j})              [compositional similarity, 0-1]
//
// Likelihood (between-site pairs only):
//   P(match) = S_{i,j} * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
//   y ~ Bernoulli(p_match)   where y=0 is match, y=1 is mismatch
//
// Alpha values are passed as DATA (not estimated). No site random effects,
// no alpha covariates, no spline machinery — just turnover estimation.

#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // =========================================================================
  // DATA
  // =========================================================================

  DATA_VECTOR(y);              // 0=match, 1=mismatch (length P, between-site only)
  DATA_IVECTOR(site_i);        // 0-based site index for pair endpoint i (length P)
  DATA_IVECTOR(site_j);        // 0-based site index for pair endpoint j (length P)

  DATA_MATRIX(X);              // pairwise covariates for turnover (P x Kbeta)
  DATA_VECTOR(w);              // pair weights (length P)

  DATA_VECTOR(alpha_fixed);    // fixed alpha values per site (length nSites)
                               // from a prior clesso_alpha fit

  const int P     = y.size();
  const int Kbeta = X.cols();

  // =========================================================================
  // PARAMETERS
  // =========================================================================

  PARAMETER(eta0_raw);                 // intercept (log scale, exp → eta0 > 0)
  PARAMETER_VECTOR(beta_raw);          // turnover coefficients (length Kbeta,
                                       // exp → beta >= 0, monotonicity)

  // ---- Transforms / constraints ----
  Type eta0 = exp(eta0_raw);
  vector<Type> beta(Kbeta);
  for (int k = 0; k < Kbeta; k++) beta(k) = exp(beta_raw(k));

  // =========================================================================
  // NEGATIVE LOG-LIKELIHOOD
  // =========================================================================
  Type nll = Type(0.0);

  const Type eps = Type(1e-10);

  for (int p = 0; p < P; p++) {
    int i = site_i(p);
    int j = site_j(p);

    // Alpha from fixed data (not estimated)
    Type ai = alpha_fixed(i);
    Type aj = alpha_fixed(j);

    // eta_{i,j} = eta0 + X_{i,j} * beta
    Type eta = eta0;
    for (int k = 0; k < Kbeta; k++) {
      eta += X(p, k) * beta(k);
    }

    // Compositional similarity
    Type S = exp(-eta);

    // P(match) = S * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
    Type p_match = S * (ai + aj) / (Type(2.0) * ai * aj);

    // Clamp to (eps, 1 - eps)
    p_match = CppAD::CondExpLt(p_match, eps, eps, p_match);
    p_match = CppAD::CondExpGt(p_match, Type(1.0) - eps, Type(1.0) - eps, p_match);

    // Weighted Bernoulli log-likelihood
    Type ll = (Type(1.0) - y(p)) * log(p_match) + y(p) * log(Type(1.0) - p_match);
    nll -= w(p) * ll;
  }

  // =========================================================================
  // REPORTS
  // =========================================================================
  ADREPORT(beta);
  ADREPORT(eta0);

  // Report predicted similarity for each pair (useful for diagnostics)
  // Use REPORT (not ADREPORT) to avoid huge Jacobian when P is large.
  vector<Type> S_pred(P);
  vector<Type> eta_pred(P);
  for (int p = 0; p < P; p++) {
    Type eta_p = eta0;
    for (int k = 0; k < Kbeta; k++) {
      eta_p += X(p, k) * beta(k);
    }
    eta_pred(p) = eta_p;
    S_pred(p) = exp(-eta_p);
  }
  REPORT(eta_pred);
  REPORT(S_pred);

  return nll;
}
