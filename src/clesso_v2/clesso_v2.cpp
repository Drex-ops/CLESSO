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

  DATA_MATRIX(Z);              // site covariates for alpha, linear (nSites x Kalpha)

  // ---- Alpha spline data ----
  // B_alpha: pre-computed B-spline basis matrix (nSites x K_basis_total)
  // Each column is a basis function; columns are grouped by covariate.
  DATA_MATRIX(B_alpha);

  // S_alpha: block-diagonal penalty matrix for alpha splines
  // (K_basis_total x K_basis_total). Typically a 2nd-order difference
  // penalty per covariate block, stacked block-diagonally.
  DATA_MATRIX(S_alpha);

  // alpha_block_sizes: number of basis functions per covariate block
  // (length = number of alpha covariates with spline terms)
  DATA_IVECTOR(alpha_block_sizes);

  // use_alpha_splines: flag (1 = use spline basis, 0 = linear only)
  DATA_INTEGER(use_alpha_splines);

  const int P = y.size();
  const int Kbeta  = X.cols();
  const int nSites = Z.rows();
  const int Kalpha = Z.cols();
  const int K_basis = B_alpha.cols();   // 0 if no splines

  // ---- Parameters: beta/turnover ----
  PARAMETER(eta0_raw);
  PARAMETER_VECTOR(beta_raw);          // length Kbeta

  // ---- Parameters: alpha/richness (linear) ----
  PARAMETER(alpha0);                   // intercept for log-alpha
  PARAMETER_VECTOR(theta_alpha);       // length Kalpha (linear coefficients)

  // ---- Parameters: alpha/richness (spline) ----
  PARAMETER_VECTOR(b_alpha);           // spline coefficients (length K_basis)
  PARAMETER_VECTOR(log_lambda_alpha);  // log smoothing params, one per covariate block

  // ---- Parameters: site-level random effect ----
  PARAMETER_VECTOR(u_site);            // length nSites (random effect)
  PARAMETER(log_sigma_u);              // SD of u_site

  // ---- Transforms / constraints ----
  // Turnover coefficients constrained >= 0 to keep eta>=0
  Type eta0 = exp(eta0_raw);
  vector<Type> beta(Kbeta);
  for(int k=0; k<Kbeta; k++) beta(k) = exp(beta_raw(k));

  Type sigma_u = exp(log_sigma_u);

  // ---- Compute log-alpha per site ----
  // log(alpha_i) = alpha0 + Z_i * theta_alpha + B_i * b_alpha + u_i
  // where B_i * b_alpha = sum_k g_k(z_{k,i}) are the smooth terms
  vector<Type> log_alpha_site(nSites);
  vector<Type> alpha_site(nSites);

  for(int s=0; s<nSites; s++){
    Type linpred = alpha0 + u_site(s);

    // Linear terms
    for(int k=0; k<Kalpha; k++){
      linpred += Z(s,k) * theta_alpha(k);
    }

    // Spline smooth terms
    if(use_alpha_splines == 1){
      for(int b=0; b<K_basis; b++){
        linpred += B_alpha(s,b) * b_alpha(b);
      }
    }

    log_alpha_site(s) = linpred;

    // enforce alpha > 1 (avoids p_within>1)
    alpha_site(s) = exp(linpred) + Type(1.0);
  }

  // ---- Negative log-likelihood ----
  Type nll = 0.0;

  // Prior on u_site ~ Normal(0, sigma_u)
  nll -= sum(dnorm(u_site, Type(0.0), sigma_u, true));

  // ---- P-spline smoothness penalty on alpha spline coefficients ----
  // Penalty: lambda/2 * b' S b (added to nll as positive contribution)
  // S_alpha is block-diagonal; log_lambda_alpha has one entry per block.
  // For simplicity we apply the full quadratic form with a single lambda
  // when there's one block, or use the block structure via the pre-built
  // composite penalty matrix weighted by per-block lambdas in R.
  // Here we support per-block lambdas via: pen = sum_k lambda_k * b'S_k b
  // But since S_alpha is already the sum of lambda-weighted blocks
  // assembled in R, we just need: pen = b' S_alpha b when
  // n_lambda_blocks == 1. For multiple blocks, R pre-multiplies.
  //
  // FLEXIBLE APPROACH: we pass the un-weighted S_alpha and apply lambda here.
  // For K covariates with n_bases_per_cov bases each, log_lambda_alpha
  // has K entries. We use DATA to tell us block sizes.

  if(use_alpha_splines == 1 && K_basis > 0){
    int n_lambda = log_lambda_alpha.size();

    if(n_lambda == 1){
      // Single smoothing parameter for all spline bases
      Type lambda = exp(log_lambda_alpha(0));
      // b' S b
      Type pen = Type(0.0);
      for(int i=0; i<K_basis; i++){
        for(int j=0; j<K_basis; j++){
          pen += b_alpha(i) * S_alpha(i,j) * b_alpha(j);
        }
      }
      nll += Type(0.5) * lambda * pen;

    } else {
      // Per-covariate-block penalties using actual block sizes
      int offset = 0;
      for(int blk=0; blk<n_lambda; blk++){
        Type lambda_k = exp(log_lambda_alpha(blk));
        int bsz = alpha_block_sizes(blk);
        Type pen_k = Type(0.0);
        for(int i=0; i<bsz; i++){
          for(int j=0; j<bsz; j++){
            pen_k += b_alpha(offset+i) * S_alpha(offset+i, offset+j) * b_alpha(offset+j);
          }
        }
        nll += Type(0.5) * lambda_k * pen_k;
        offset += bsz;
      }
    }
  }

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

  // Spline reports
  if(use_alpha_splines == 1 && K_basis > 0){
    ADREPORT(b_alpha);
    vector<Type> lambda_alpha(log_lambda_alpha.size());
    for(int k=0; k<log_lambda_alpha.size(); k++){
      lambda_alpha(k) = exp(log_lambda_alpha(k));
    }
    ADREPORT(lambda_alpha);
  }

  ADREPORT(beta);
  ADREPORT(eta0);

  return nll;
}