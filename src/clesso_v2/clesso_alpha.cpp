// clesso_alpha.cpp — TMB template for alpha-only (richness) model
//
// Simplified version of clesso_v2.cpp that uses ONLY within-site pairs
// to estimate species richness (alpha) per site.
//
// Model:
//   log(alpha*_s) = alpha0 + Z_s * theta_alpha + B_s * b_alpha + u_s
//   alpha_s       = exp(log(alpha*_s)) + 1         [ensures alpha > 1]
//
// Likelihood (within-site pairs only):
//   P(match) = 1 / alpha_s
//   y ~ Bernoulli(p_match)   where y=0 is match, y=1 is mismatch
//
// No turnover (beta) component — no X matrix, no eta, no S.

#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // =========================================================================
  // DATA
  // =========================================================================

  DATA_VECTOR(y);              // 0=match, 1=mismatch (length P, within-site only)
  DATA_IVECTOR(site_i);        // 0-based site index for each pair (length P)
  DATA_VECTOR(w);              // pair weights (length P)

  DATA_MATRIX(Z);              // site covariates for alpha, linear (nSites x Kalpha)

  // ---- Alpha spline data ----
  DATA_MATRIX(B_alpha);        // B-spline basis matrix (nSites x K_basis_total)
  DATA_MATRIX(S_alpha);        // block-diagonal penalty matrix (K_basis x K_basis)
  DATA_IVECTOR(alpha_block_sizes);  // basis functions per covariate block
  DATA_INTEGER(use_alpha_splines);  // 1 = use spline basis, 0 = linear only

  const int P      = y.size();
  const int nSites = Z.rows();
  const int Kalpha = Z.cols();
  const int K_basis = B_alpha.cols();

  // =========================================================================
  // PARAMETERS
  // =========================================================================

  // ---- Alpha / richness (linear) ----
  PARAMETER(alpha0);                   // intercept for log-alpha
  PARAMETER_VECTOR(theta_alpha);       // length Kalpha

  // ---- Alpha / richness (spline) ----
  PARAMETER_VECTOR(b_alpha);           // spline coefficients (length K_basis)
  PARAMETER_VECTOR(log_lambda_alpha);  // log smoothing params, one per block

  // ---- Site-level random effect ----
  PARAMETER_VECTOR(u_site);            // length nSites
  PARAMETER(log_sigma_u);              // log SD of u_site

  // ---- Transforms ----
  Type sigma_u = exp(log_sigma_u);

  // =========================================================================
  // COMPUTE LOG-ALPHA PER SITE
  // =========================================================================
  //
  // log(alpha*_s) = alpha0 + Z_s * theta_alpha + B_s * b_alpha + u_s
  // alpha_s       = exp(log(alpha*_s)) + 1
  //
  vector<Type> log_alpha_site(nSites);
  vector<Type> alpha_site(nSites);

  for (int s = 0; s < nSites; s++) {
    Type linpred = alpha0 + u_site(s);

    // Linear terms
    for (int k = 0; k < Kalpha; k++) {
      linpred += Z(s, k) * theta_alpha(k);
    }

    // Spline smooth terms
    if (use_alpha_splines == 1) {
      for (int b = 0; b < K_basis; b++) {
        linpred += B_alpha(s, b) * b_alpha(b);
      }
    }

    log_alpha_site(s) = linpred;
    alpha_site(s) = exp(linpred) + Type(1.0);   // alpha > 1
  }

  // =========================================================================
  // NEGATIVE LOG-LIKELIHOOD
  // =========================================================================
  Type nll = Type(0.0);

  // --- Random effect prior: u_s ~ N(0, sigma_u) ---
  nll -= sum(dnorm(u_site, Type(0.0), sigma_u, true));

  // --- P-spline penalty: (1/2) * sum_k lambda_k * b_k' S_k b_k ---
  if (use_alpha_splines == 1 && K_basis > 0) {
    int n_lambda = log_lambda_alpha.size();

    if (n_lambda == 1) {
      // Single smoothing parameter for all bases
      Type lambda = exp(log_lambda_alpha(0));
      // Efficient sparse b' S b via matrix-vector product
      vector<Type> Sb = S_alpha * b_alpha;
      Type pen = (b_alpha * Sb).sum();
      nll += Type(0.5) * lambda * pen;

    } else {
      // Per-covariate-block penalties
      // Use sparse mat-vec on full S then accumulate per block
      vector<Type> Sb = S_alpha * b_alpha;
      int offset = 0;
      for (int blk = 0; blk < n_lambda; blk++) {
        Type lambda_k = exp(log_lambda_alpha(blk));
        int bsz = alpha_block_sizes(blk);
        Type pen_k = Type(0.0);
        for (int i = 0; i < bsz; i++) {
          pen_k += b_alpha(offset + i) * Sb(offset + i);
        }
        nll += Type(0.5) * lambda_k * pen_k;
        offset += bsz;
      }
    }
  }

  // --- Bernoulli likelihood: within-site pairs only ---
  //
  // P(match) = 1 / alpha_s
  // ll_p = (1 - y_p) * log(p) + y_p * log(1 - p)
  //
  const Type eps = Type(1e-10);

  for (int p = 0; p < P; p++) {
    int s = site_i(p);
    Type ai = alpha_site(s);

    Type p_match = Type(1.0) / ai;

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
  // Use REPORT (not ADREPORT) for large per-site vectors to avoid
  // building the delta-method Jacobian in sdreport(), which can
  // exceed memory limits when nSites is large. Retrieve via obj$report().
  REPORT(alpha_site);
  REPORT(log_alpha_site);
  ADREPORT(theta_alpha);
  ADREPORT(alpha0);
  ADREPORT(sigma_u);

  if (use_alpha_splines == 1 && K_basis > 0) {
    ADREPORT(b_alpha);
    vector<Type> lambda_alpha(log_lambda_alpha.size());
    for (int k = 0; k < log_lambda_alpha.size(); k++) {
      lambda_alpha(k) = exp(log_lambda_alpha(k));
    }
    ADREPORT(lambda_alpha);
  }

  return nll;
}
