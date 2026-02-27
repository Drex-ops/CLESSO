#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ---- Data ----
  DATA_VECTOR(y);              // 0=match, 1=mismatch (length P)
  DATA_IVECTOR(site_i);        // 0-based site index (length P)
  DATA_IVECTOR(site_j);        // 0-based site index (length P)
  DATA_IVECTOR(is_within);     // 1 within-site, 0 between-site (length P)

  DATA_MATRIX(X);              // pairwise covariates for turnover (P x Kbeta)
  DATA_VECTOR(w);              // pair weights (length P)

  DATA_MATRIX(Z);              // site covariates for alpha (nSites x Kalpha)

  const int P = y.size();
  const int Kbeta  = X.cols();
  const int nSites = Z.rows();
  const int Kalpha = Z.cols();

  // ---- Parameters: beta/turnover ----
  PARAMETER(eta0_raw);
  PARAMETER_VECTOR(beta_raw);          // length Kbeta

  // ---- Parameters: alpha/richness ----
  PARAMETER(alpha0);                   // intercept for log-alpha
  PARAMETER_VECTOR(theta_alpha);       // length Kalpha
  PARAMETER_VECTOR(u_site);            // length nSites (random effect)
  PARAMETER(log_sigma_u);              // SD of u_site

  // ---- Transforms / constraints ----
  // Turnover coefficients constrained >= 0 to keep eta>=0
  Type eta0 = exp(eta0_raw);
  vector<Type> beta(Kbeta);
  for(int k=0; k<Kbeta; k++) beta(k) = exp(beta_raw(k));

  Type sigma_u = exp(log_sigma_u);

  // Compute log alpha and alpha per site
  vector<Type> log_alpha_site(nSites);
  vector<Type> alpha_site(nSites);

  for(int s=0; s<nSites; s++){
    Type linpred = alpha0 + u_site(s);
    for(int k=0; k<Kalpha; k++){
      linpred += Z(s,k) * theta_alpha(k);
    }
    log_alpha_site(s) = linpred;

    // enforce alpha > 1 (avoids p_within>1); you can change +1 to +eps if preferred
    alpha_site(s) = exp(linpred) + Type(1.0);
  }

  // ---- Negative log-likelihood ----
  Type nll = 0.0;

  // Prior on u_site ~ Normal(0, sigma_u)
  nll -= sum(dnorm(u_site, Type(0.0), sigma_u, true));

  const Type eps = Type(1e-10);

  for(int p=0; p<P; p++){
    int i = site_i(p);
    int j = site_j(p);

    Type ai = alpha_site(i);
    Type aj = alpha_site(j);

    // eta_ij = eta0 + X_ij %*% beta
    Type eta = eta0;
    for(int k=0; k<Kbeta; k++){
      eta += X(p,k) * beta(k);
    }

    Type S = exp(-eta);

    Type p_match;
    if(is_within(p) == 1){
      // within-site: p = 1/alpha_i
      p_match = Type(1.0) / ai;
    } else {
      // between-site:
      p_match = S * (ai + aj) / (Type(2.0) * ai * aj);
    }

    // clamp to (eps, 1-eps)
    p_match = CppAD::CondExpLt(p_match, eps, eps, p_match);
    p_match = CppAD::CondExpGt(p_match, Type(1.0) - eps, Type(1.0) - eps, p_match);

    // weighted Bernoulli contribution
    Type ll = (Type(1.0) - y(p)) * log(p_match) + y(p) * log(Type(1.0) - p_match);
    nll -= w(p) * ll;
  }

  // ---- Reports ----
  ADREPORT(alpha_site);
  ADREPORT(log_alpha_site);
  ADREPORT(theta_alpha);
  ADREPORT(alpha0);
  ADREPORT(sigma_u);

  ADREPORT(beta);
  ADREPORT(eta0);

  return nll;
}