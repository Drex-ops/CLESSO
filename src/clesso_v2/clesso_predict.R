##############################################################################
##
## clesso_predict.R -- Prediction functions for CLESSO v2
##
## Given a fitted CLESSO model (from run_clesso.R) and new site / pair data,
## predict the alpha (richness) and beta (turnover/similarity) components.
##
## Functions:
##   clesso_extract_params()  -- extract fitted parameters from TMB results
##   clesso_predict_alpha()   -- predict richness (alpha) at new sites
##   clesso_predict_beta()    -- predict turnover similarity for site pairs
##   clesso_predict()         -- unified prediction for sites and/or pairs
##
## Depends on:
##   - splines::bs()                   (B-spline basis at new sites)
##   - I_spline() / splineData_fast()  (from shared/R/gdm_functions.R)
##
##############################################################################


# ---------------------------------------------------------------------------
# clesso_extract_params
#
# Extract fitted parameter estimates from a CLESSO results object
# (as saved by run_clesso.R).
#
# Parameters:
#   results - list returned by run_clesso.R (or loaded from RDS), containing
#             at minimum: sdreport, model_data, config
#
# Returns:
#   list with named parameter vectors:
#     alpha0       - scalar intercept for log-alpha
#     theta_alpha  - vector of linear alpha coefficients (length Kalpha)
#     b_alpha      - vector of spline alpha coefficients (length K_basis)
#     sigma_u      - scalar SD of site random effect
#     u_site       - vector of fitted site random effects (length nSites)
#     eta0         - scalar turnover intercept (exp-transformed)
#     beta         - vector of turnover coefficients (exp-transformed)
#     lambda_alpha - vector of smoothing parameters (if splines used)
# ---------------------------------------------------------------------------
clesso_extract_params <- function(results) {
  rep   <- results$sdreport
  est   <- summary(rep, "report")
  est_r <- summary(rep, "random")

  ## Scalar / named extractions from ADREPORT
  .get <- function(name) {
    idx <- grep(paste0("^", name, "$"), rownames(est))
    if (length(idx) == 0) return(NULL)
    setNames(est[idx, "Estimate"], paste0(name, seq_along(idx)))
  }

  alpha0      <- unname(est[grep("^alpha0$", rownames(est)), "Estimate"])
  theta_alpha <- unname(est[grep("^theta_alpha$", rownames(est)), "Estimate"])
  sigma_u     <- unname(est[grep("^sigma_u$", rownames(est)), "Estimate"])
  eta0        <- unname(est[grep("^eta0$", rownames(est)), "Estimate"])
  beta_raw    <- est[grep("^beta$", rownames(est)), "Estimate"]
  beta        <- unname(beta_raw)

  ## Spline coefficients
  b_alpha_idx <- grep("^b_alpha$", rownames(est))
  b_alpha     <- if (length(b_alpha_idx) > 0) unname(est[b_alpha_idx, "Estimate"])
                 else numeric(0)

  lambda_idx  <- grep("^lambda_alpha$", rownames(est))
  lambda_alpha <- if (length(lambda_idx) > 0) unname(est[lambda_idx, "Estimate"])
                  else numeric(0)

  ## Site random effects from the random parameter block
  u_idx    <- grep("^u_site$", rownames(est_r))
  u_site   <- if (length(u_idx) > 0) unname(est_r[u_idx, "Estimate"])
              else numeric(0)

  ## Standard errors (useful for uncertainty)
  alpha0_se      <- unname(est[grep("^alpha0$", rownames(est)), "Std. Error"])
  theta_alpha_se <- unname(est[grep("^theta_alpha$", rownames(est)), "Std. Error"])
  beta_se        <- unname(est[grep("^beta$", rownames(est)), "Std. Error"])
  eta0_se        <- unname(est[grep("^eta0$", rownames(est)), "Std. Error"])

  list(
    alpha0       = alpha0,
    theta_alpha  = theta_alpha,
    b_alpha      = b_alpha,
    sigma_u      = sigma_u,
    u_site       = u_site,
    eta0         = eta0,
    beta         = beta,
    lambda_alpha = lambda_alpha,
    ## SEs
    alpha0_se      = alpha0_se,
    theta_alpha_se = theta_alpha_se,
    beta_se        = beta_se,
    eta0_se        = eta0_se
  )
}


# ---------------------------------------------------------------------------
# clesso_predict_alpha
#
# Predict species richness (alpha) at a set of sites given their covariate
# values and the fitted CLESSO model.
#
# Model:
#   log(alpha_i - 1) = alpha0 + Z_i * theta + B_i * b + u_i
#   alpha_i = exp(log(alpha_i - 1)) + 1
#
# Parameters:
#   new_sites   - data.frame / data.table with columns matching the alpha
#                 covariates used in model fitting. Must include the same
#                 covariate columns as were used for Z. Optionally include
#                 'site_id', 'lon', 'lat'.
#   results     - fitted CLESSO results object (from run_clesso.R)
#   params      - (optional) pre-extracted params from clesso_extract_params()
#   include_re  - logical: include site random effects for sites that were
#                 in the training data? (default TRUE). New sites get u=0.
#
# Returns:
#   data.table with columns:
#     site_id (if available), lon, lat, log_alpha, alpha, u_site
# ---------------------------------------------------------------------------
clesso_predict_alpha <- function(new_sites,
                                 results,
                                 params      = NULL,
                                 include_re  = TRUE) {
  require(data.table)
  require(splines)

  if (is.null(params)) params <- clesso_extract_params(results)
  model_data <- results$model_data

  ## -----------------------------------------------------------------------
  ## 1. Identify covariate columns and prepare Z_new
  ## -----------------------------------------------------------------------
  alpha_info   <- model_data$alpha_info
  cov_cols     <- alpha_info$cov_cols
  z_center     <- alpha_info$z_center
  z_scale      <- alpha_info$z_scale
  use_splines  <- alpha_info$use_alpha_splines
  spline_info  <- alpha_info$spline_info

  new_sites <- as.data.table(new_sites)
  n_new     <- nrow(new_sites)

  ## Check required columns
  missing_cols <- setdiff(cov_cols, names(new_sites))
  if (length(missing_cols) > 0) {
    stop(sprintf("new_sites is missing required covariate columns: %s",
                 paste(missing_cols, collapse = ", ")))
  }

  ## Extract raw covariate values (un-standardised)
  Z_raw_new <- as.matrix(new_sites[, ..cov_cols])
  storage.mode(Z_raw_new) <- "double"

  ## Standardise using training-set centering/scaling
  if (!is.null(z_center) && !is.null(z_scale)) {
    Z_new <- scale(Z_raw_new, center = z_center, scale = z_scale)
    Z_new <- unname(as.matrix(Z_new))
  } else {
    Z_new <- unname(Z_raw_new)
  }

  ## -----------------------------------------------------------------------
  ## 2. Linear component: alpha0 + Z_new %*% theta_alpha
  ## -----------------------------------------------------------------------
  linpred <- rep(params$alpha0, n_new) + as.numeric(Z_new %*% params$theta_alpha)

  ## -----------------------------------------------------------------------
  ## 3. Spline smooth component: B_new %*% b_alpha
  ## -----------------------------------------------------------------------
  if (use_splines && !is.null(spline_info) && length(params$b_alpha) > 0) {
    n_cov <- spline_info$n_covariates

    B_blocks <- vector("list", n_cov)
    for (k in seq_len(n_cov)) {
      x_new <- Z_raw_new[, k]

      ## Reconstruct the B-spline basis at new data points using the
      ## same knots and boundary knots from the training fit
      B_blocks[[k]] <- splines::bs(
        x_new,
        knots          = spline_info$knot_list[[k]],
        degree         = spline_info$spline_deg,
        Boundary.knots = spline_info$boundary_list[[k]],
        intercept      = FALSE
      )
    }

    B_new <- do.call(cbind, B_blocks)
    storage.mode(B_new) <- "double"

    linpred <- linpred + as.numeric(B_new %*% params$b_alpha)
  }

  ## -----------------------------------------------------------------------
  ## 4. Site random effects (for known sites)
  ## -----------------------------------------------------------------------
  u_vec <- rep(0, n_new)

  if (include_re && length(params$u_site) > 0 && "site_id" %in% names(new_sites)) {
    ## Match new sites to training sites by site_id
    train_site_table <- model_data$site_info$site_table
    match_idx <- match(new_sites$site_id, train_site_table$site_id)
    known     <- !is.na(match_idx)

    if (any(known)) {
      site_indices <- train_site_table$site_index[match_idx[known]]
      ## site_index is 0-based, u_site is 1-indexed in R
      u_vec[known] <- params$u_site[site_indices + 1L]
      cat(sprintf("  Alpha predict: %d / %d sites matched to training (using random effects)\n",
                  sum(known), n_new))
    }
  }

  linpred <- linpred + u_vec

  ## -----------------------------------------------------------------------
  ## 5. Transform to alpha scale
  ## -----------------------------------------------------------------------
  ## alpha = exp(linpred) + 1  (ensures alpha > 1)
  log_alpha_m1 <- linpred
  alpha        <- exp(linpred) + 1

  ## -----------------------------------------------------------------------
  ## 6. Assemble output
  ## -----------------------------------------------------------------------
  out <- data.table(
    log_alpha = log_alpha_m1,
    alpha     = alpha,
    u_site    = u_vec
  )

  ## Attach identifiers if available
  if ("site_id" %in% names(new_sites)) out[, site_id := new_sites$site_id]
  if ("lon" %in% names(new_sites))     out[, lon := new_sites$lon]
  if ("lat" %in% names(new_sites))     out[, lat := new_sites$lat]

  ## Reorder columns nicely
  id_cols   <- intersect(c("site_id", "lon", "lat"), names(out))
  val_cols  <- setdiff(names(out), id_cols)
  setcolorder(out, c(id_cols, val_cols))

  out[]
}


# ---------------------------------------------------------------------------
# clesso_predict_beta
#
# Predict compositional turnover (similarity) between site pairs.
#
# Model sub-components:
#   eta_{i,j} = eta0 + X_{i,j} %*% beta
#   S_{i,j}   = exp(-eta_{i,j})        (compositional similarity, 0 to 1)
#
# Optionally combines with alpha predictions to compute full p_match:
#   p_{i,j} = S_{i,j} * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
#
# Parameters:
#   new_pairs      - data.frame / data.table with pair information.
#                    Must include coordinate columns for geographic distance:
#                      lon_i, lat_i, lon_j, lat_j
#                    And optionally environmental columns:
#                      env_i_<name>, env_j_<name>  (pairwise env values)
#                    OR provide env_site_table for lookup.
#   results        - fitted CLESSO results object (from run_clesso.R)
#   params         - (optional) pre-extracted params from clesso_extract_params()
#   env_site_table - (optional) data.table with site_id + env columns for
#                    looking up environmental values per site. If provided,
#                    new_pairs must have 'site_i' and 'site_j' columns.
#   alpha_i        - (optional) pre-computed alpha values for site i (length = nrow(new_pairs))
#   alpha_j        - (optional) pre-computed alpha values for site j
#   geo_distance   - include geographic distance in turnover (default: from config)
#   splineData_fn  - (optional) custom spline function
#
# Returns:
#   data.table with columns:
#     eta (linear predictor), S (similarity = exp(-eta)),
#     and optionally p_match (if alpha values are provided)
# ---------------------------------------------------------------------------
clesso_predict_beta <- function(new_pairs,
                                results,
                                params         = NULL,
                                env_site_table = NULL,
                                alpha_i        = NULL,
                                alpha_j        = NULL,
                                geo_distance   = NULL,
                                splineData_fn  = NULL) {
  require(data.table)

  if (is.null(params))       params <- clesso_extract_params(results)
  model_data   <- results$model_data
  config       <- results$config
  turnover_info <- model_data$turnover_info

  new_pairs <- as.data.table(new_pairs)
  n_pairs   <- nrow(new_pairs)

  if (is.null(geo_distance)) geo_distance <- config$geo_distance

  ## -----------------------------------------------------------------------
  ## 1. Build pairwise I-spline covariate matrix X_new
  ##    using the same quantiles and structure as training
  ## -----------------------------------------------------------------------
  ## We need the same covariate columns paired as site1|site2

  ## Determine env variables from training
  ## The turnover_info$col_names encodes var_spl1, var_spl2, ... patterns
  ## The splines and quantiles vectors tell us how to reconstruct X
  splines_vec  <- turnover_info$splines
  quant_vec    <- turnover_info$quantiles

  ## Determine variable names from X column names
  ## Column names are like "varname_spl1", "varname_spl2", etc.
  x_colnames   <- turnover_info$col_names
  n_vars       <- length(splines_vec)

  ## Extract per-pair env values
  ## Strategy A: env columns directly in new_pairs (env_i_*, env_j_*)
  ## Strategy B: Look up from env_site_table via site_i / site_j
  i_cols <- grep("^env_i_", names(new_pairs), value = TRUE)
  j_cols <- grep("^env_j_", names(new_pairs), value = TRUE)

  if (length(i_cols) > 0 && length(j_cols) > 0) {
    env_i <- as.matrix(new_pairs[, ..i_cols])
    env_j <- as.matrix(new_pairs[, ..j_cols])
    var_names <- gsub("^env_i_", "", i_cols)

  } else if (!is.null(env_site_table)) {
    env_site_table <- as.data.table(env_site_table)
    stopifnot("site_id" %in% names(env_site_table))
    stopifnot(all(c("site_i", "site_j") %in% names(new_pairs)))

    env_cols <- setdiff(names(env_site_table), "site_id")
    env_i <- as.matrix(env_site_table[match(new_pairs$site_i, env_site_table$site_id),
                                       ..env_cols])
    env_j <- as.matrix(env_site_table[match(new_pairs$site_j, env_site_table$site_id),
                                       ..env_cols])
    var_names <- env_cols

  } else {
    ## No environmental data -- geo distance only
    env_i <- matrix(nrow = n_pairs, ncol = 0)
    env_j <- matrix(nrow = n_pairs, ncol = 0)
    var_names <- character(0)
  }

  ## Add geographic coordinates if requested
  if (geo_distance) {
    stopifnot(all(c("lon_i", "lat_i", "lon_j", "lat_j") %in% names(new_pairs)))
    geo_i <- cbind(new_pairs$lon_i, new_pairs$lat_i)
    geo_j <- cbind(new_pairs$lon_j, new_pairs$lat_j)
    colnames(geo_i) <- colnames(geo_j) <- c("geo_lon", "geo_lat")
    env_i <- cbind(env_i, geo_i)
    env_j <- cbind(env_j, geo_j)
    var_names <- c(var_names, "geo_lon", "geo_lat")
  }

  n_vars_new <- ncol(env_i)

  if (n_vars_new != n_vars) {
    stop(sprintf("Number of turnover variables (%d) does not match fitted model (%d)",
                 n_vars_new, n_vars))
  }

  ## Build site1|site2 matrix for splineData
  colnames(env_i) <- var_names
  colnames(env_j) <- var_names
  site_pair_matrix <- cbind(env_i, env_j)

  ## Apply I-spline transformation with TRAINING quantiles
  if (is.null(splineData_fn)) {
    if (exists("splineData_fast", mode = "function")) {
      splineData_fn <- splineData_fast
    } else if (exists("splineData", mode = "function")) {
      splineData_fn <- splineData
    } else {
      stop("splineData function not found. Source shared/R/gdm_functions.R first.")
    }
  }

  X_new <- splineData_fn(site_pair_matrix,
                         splines   = splines_vec,
                         quantiles = quant_vec)

  storage.mode(X_new) <- "double"

  ## -----------------------------------------------------------------------
  ## 2. Compute eta and similarity S
  ## -----------------------------------------------------------------------
  ## eta = eta0 + X %*% beta
  eta <- rep(params$eta0, n_pairs) + as.numeric(X_new %*% params$beta)

  ## S = exp(-eta) -- compositional similarity
  S <- exp(-eta)

  ## -----------------------------------------------------------------------
  ## 3. Optionally compute p_match using alpha values
  ## -----------------------------------------------------------------------
  p_match <- NULL

  if (!is.null(alpha_i) && !is.null(alpha_j)) {
    stopifnot(length(alpha_i) == n_pairs, length(alpha_j) == n_pairs)
    ## p_match = S * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
    p_match <- S * (alpha_i + alpha_j) / (2 * alpha_i * alpha_j)
    ## Clamp to [0, 1]
    p_match <- pmin(pmax(p_match, 0), 1)
  }

  ## -----------------------------------------------------------------------
  ## 4. Assemble output
  ## -----------------------------------------------------------------------
  out <- data.table(
    eta = eta,
    S   = S
  )

  if (!is.null(p_match)) out[, p_match := p_match]

  ## Attach pair identifiers if available
  if ("site_i" %in% names(new_pairs)) out[, site_i := new_pairs$site_i]
  if ("site_j" %in% names(new_pairs)) out[, site_j := new_pairs$site_j]

  id_cols  <- intersect(c("site_i", "site_j"), names(out))
  val_cols <- setdiff(names(out), id_cols)
  if (length(id_cols) > 0) setcolorder(out, c(id_cols, val_cols))

  out[]
}


# ---------------------------------------------------------------------------
# clesso_predict
#
# Unified prediction function. Given a fitted CLESSO model and new site /
# pair data, predict alpha, beta, or both.
#
# Parameters:
#   results        - fitted CLESSO results object (from run_clesso.R)
#   new_sites      - (optional) data.table/data.frame for alpha prediction.
#                    Must have the same covariate columns as training.
#   new_pairs      - (optional) data.table/data.frame for beta prediction.
#                    Must have lon_i/lat_i/lon_j/lat_j and optionally
#                    env_i_*/env_j_* or use env_site_table.
#   env_site_table - (optional) site-level env table for turnover lookup.
#   include_re     - include site random effects for known sites (default TRUE)
#   compute_pmatch - if TRUE and both alpha and beta are predicted, compute
#                    the full p_match probability (default TRUE)
#   geo_distance   - include geographic distance in turnover (default: from config)
#   splineData_fn  - optional custom spline function
#
# Returns:
#   list with:
#     alpha_pred - data.table of alpha predictions (NULL if new_sites not given)
#     beta_pred  - data.table of beta predictions (NULL if new_pairs not given)
#     params     - extracted parameter list
# ---------------------------------------------------------------------------
clesso_predict <- function(results,
                           new_sites      = NULL,
                           new_pairs      = NULL,
                           env_site_table = NULL,
                           include_re     = TRUE,
                           compute_pmatch = TRUE,
                           geo_distance   = NULL,
                           splineData_fn  = NULL) {

  params <- clesso_extract_params(results)

  ## -----------------------------------------------------------------------
  ## Alpha prediction
  ## -----------------------------------------------------------------------
  alpha_pred <- NULL
  if (!is.null(new_sites)) {
    cat("--- Predicting alpha (richness) ---\n")
    alpha_pred <- clesso_predict_alpha(
      new_sites  = new_sites,
      results    = results,
      params     = params,
      include_re = include_re
    )
    cat(sprintf("  Predicted alpha at %d sites: mean = %.1f, range = [%.1f, %.1f]\n",
                nrow(alpha_pred),
                mean(alpha_pred$alpha),
                min(alpha_pred$alpha),
                max(alpha_pred$alpha)))
  }

  ## -----------------------------------------------------------------------
  ## Beta prediction
  ## -----------------------------------------------------------------------
  beta_pred <- NULL
  if (!is.null(new_pairs)) {
    new_pairs <- as.data.table(new_pairs)

    ## If we predicted alpha and can match site IDs to pairs,
    ## automatically supply alpha_i and alpha_j for p_match
    alpha_i_vec <- NULL
    alpha_j_vec <- NULL

    if (compute_pmatch && !is.null(alpha_pred) &&
        "site_id" %in% names(alpha_pred) &&
        all(c("site_i", "site_j") %in% names(new_pairs))) {

      ## Look up alpha for each pair endpoint
      match_i <- match(new_pairs$site_i, alpha_pred$site_id)
      match_j <- match(new_pairs$site_j, alpha_pred$site_id)

      if (!anyNA(match_i) && !anyNA(match_j)) {
        alpha_i_vec <- alpha_pred$alpha[match_i]
        alpha_j_vec <- alpha_pred$alpha[match_j]
        cat("  Combining alpha and beta for p_match computation.\n")
      } else {
        n_miss <- sum(is.na(match_i)) + sum(is.na(match_j))
        cat(sprintf("  Warning: %d pair endpoints not found in alpha predictions. Skipping p_match.\n",
                    n_miss))
      }
    }

    cat("--- Predicting beta (turnover) ---\n")
    beta_pred <- clesso_predict_beta(
      new_pairs      = new_pairs,
      results        = results,
      params         = params,
      env_site_table = env_site_table,
      alpha_i        = alpha_i_vec,
      alpha_j        = alpha_j_vec,
      geo_distance   = geo_distance,
      splineData_fn  = splineData_fn
    )
    cat(sprintf("  Predicted turnover for %d pairs: mean S = %.3f, range = [%.3f, %.3f]\n",
                nrow(beta_pred),
                mean(beta_pred$S),
                min(beta_pred$S),
                max(beta_pred$S)))

    if ("p_match" %in% names(beta_pred)) {
      cat(sprintf("  Predicted p_match: mean = %.3f, range = [%.3f, %.3f]\n",
                  mean(beta_pred$p_match),
                  min(beta_pred$p_match),
                  max(beta_pred$p_match)))
    }
  }

  list(
    alpha_pred = alpha_pred,
    beta_pred  = beta_pred,
    params     = params
  )
}
