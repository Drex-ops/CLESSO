##############################################################################
##
## clesso_prepare_data.R — Prepare model inputs for CLESSO v2 TMB model
##
## Transforms the paired dataset from clesso_sampler() into the matrices
## and vectors required by the TMB model (clesso_v2.cpp):
##   - y:         response vector (0=match, 1=mismatch)
##   - site_i/j:  0-based site indices
##   - is_within: within-site indicator
##   - X:         pairwise covariate matrix for turnover (beta)
##   - Z:         site-level covariate matrix for richness (alpha)
##   - B_alpha:   B-spline basis matrix for alpha smooth terms
##   - S_alpha:   penalty matrix for alpha splines
##   - w:         pair weights
##
## Functions:
##   clesso_build_site_table()       — unique site table with covariates
##   clesso_build_alpha_splines()    — P-spline basis + penalty for alpha
##   clesso_build_turnover_X()       — pairwise |env_i - env_j| matrix
##   clesso_prepare_model_data()     — main orchestrator → TMB data list
##
##############################################################################

# ---------------------------------------------------------------------------
# clesso_build_site_table
#
# Build a table of unique sites with their coordinates and any site-level
# covariates for the alpha (richness) sub-model.
#
# Parameters:
#   pairs_dt    - data.table from clesso_sampler()
#   site_covs   - optional data.table/data.frame of site-level covariates.
#                 Must have a 'site_id' column. Other columns are treated
#                 as alpha-model covariates (Z matrix).
#                 If NULL, only coordinates are used.
#   standardize - whether to centre and scale covariate columns (default TRUE)
#
# Returns:
#   list with:
#     site_table - data.table: site_id, site_index (0-based), lon, lat, + covariates
#     Z          - numeric matrix (nSites x Kalpha) of alpha covariates
#     z_center   - centering values used (for prediction)
#     z_scale    - scaling values used (for prediction)
# ---------------------------------------------------------------------------
clesso_build_site_table <- function(pairs_dt, site_covs = NULL,
                                    standardize = TRUE) {
  require(data.table)

  ## Collect all unique sites from the pairs
  sites_i <- pairs_dt[, .(site_id = site_i, lon = lon_i, lat = lat_i)]
  sites_j <- pairs_dt[, .(site_id = site_j, lon = lon_j, lat = lat_j)]
  all_sites <- unique(rbind(sites_i, sites_j))
  all_sites <- unique(all_sites, by = "site_id")

  ## Assign 0-based contiguous indices
  all_sites[, site_index := .I - 1L]
  setkey(all_sites, site_id)

  cat(sprintf("  Site table: %d unique sites\n", nrow(all_sites)))

  ## -----------------------------------------------------------------------
  ## Build Z matrix (alpha covariates)
  ## -----------------------------------------------------------------------
  if (!is.null(site_covs)) {
    site_covs <- as.data.table(site_covs)
    stopifnot("site_id" %in% names(site_covs))

    ## Merge site covariates (keep all sites, allow NA for missing)
    merged <- merge(all_sites, site_covs, by = "site_id", all.x = TRUE, sort = FALSE)
    setorder(merged, site_index)

    alpha_cov_cols <- setdiff(names(site_covs), "site_id")

    if (length(alpha_cov_cols) == 0) {
      warning("site_covs has no covariate columns besides site_id. Using coordinates only.")
      alpha_cov_cols <- c("lon", "lat")
      Z_raw <- as.matrix(merged[, .(lon, lat)])
    } else {
      Z_raw <- as.matrix(merged[, ..alpha_cov_cols])
    }

    ## Check for NAs
    na_count <- sum(is.na(Z_raw))
    if (na_count > 0) {
      warning(sprintf("%d NA values in site covariates. Sites with missing covariates may cause issues.", na_count))
    }
  } else {
    ## Default: use lon/lat as alpha covariates
    cat("  No site_covs provided. Using lon/lat as alpha covariates.\n")
    setorder(all_sites, site_index)
    alpha_cov_cols <- c("lon", "lat")
    Z_raw <- as.matrix(all_sites[, .(lon, lat)])
    merged <- copy(all_sites)
  }

  ## Standardize
  z_center <- NULL
  z_scale  <- NULL
  if (standardize && ncol(Z_raw) > 0) {
    z_center <- colMeans(Z_raw, na.rm = TRUE)
    z_scale  <- apply(Z_raw, 2, sd, na.rm = TRUE)
    z_scale[z_scale == 0] <- 1  # avoid division by zero
    Z <- scale(Z_raw, center = z_center, scale = z_scale)
    Z <- unname(as.matrix(Z))
  } else {
    Z <- unname(as.matrix(Z_raw))
  }

  storage.mode(Z) <- "double"

  list(
    site_table     = merged,
    Z              = Z,
    Z_raw          = Z_raw,
    alpha_cov_cols = alpha_cov_cols,
    z_center       = z_center,
    z_scale        = z_scale
  )
}


# ---------------------------------------------------------------------------
# clesso_build_alpha_splines
#
# Build penalised B-spline (P-spline) basis matrices for the alpha
# (richness) sub-model. Each alpha covariate gets its own B-spline basis
# and second-order difference penalty, stacked block-diagonally.
#
# This implements the g_k(·) smooth terms from the model specification:
#   log alpha_i = alpha0 + sum_k g_k(z_{k,i}) + u_i
#
# Uses the splines package (base R) to generate B-spline bases, and a
# 2nd-order difference penalty matrix (P-spline; Eilers & Marx 1996)
# for each covariate block.
#
# Parameters:
#   Z_raw        - un-standardised site covariate matrix (nSites x Kalpha)
#   n_knots      - number of interior knots per covariate (default 10)
#   spline_deg   - B-spline degree (default 3 = cubic)
#   pen_order    - difference penalty order (default 2)
#   cov_names    - optional covariate names for labelling
#
# Returns:
#   list with:
#     B_alpha       - numeric matrix (nSites x K_basis_total)
#     S_alpha       - numeric matrix (K_basis_total x K_basis_total),
#                     block-diagonal penalty matrix
#     n_bases_per_cov - integer vector: bases per covariate
#     n_covariates  - number of covariates with spline terms
#     knot_list     - list of knot vectors (for prediction on new data)
#     boundary_list - list of boundary knots per covariate
# ---------------------------------------------------------------------------
clesso_build_alpha_splines <- function(Z_raw,
                                       n_knots        = 10,
                                       spline_deg     = 3,
                                       pen_order      = 2,
                                       cov_names      = NULL,
                                       knot_positions = NULL) {
  require(splines)

  if (!is.matrix(Z_raw)) Z_raw <- as.matrix(Z_raw)
  n_sites <- nrow(Z_raw)
  n_cov   <- ncol(Z_raw)

  if (is.null(cov_names)) {
    cov_names <- if (!is.null(colnames(Z_raw))) colnames(Z_raw)
                 else paste0("z", seq_len(n_cov))
  }

  ## Validate user-supplied knot positions
  if (!is.null(knot_positions)) {
    if (!is.list(knot_positions) || length(knot_positions) != n_cov) {
      stop(sprintf(
        "knot_positions must be a list of length %d (one numeric vector per covariate), got length %d",
        n_cov, length(knot_positions)))
    }
  }

  B_list    <- vector("list", n_cov)
  S_list    <- vector("list", n_cov)
  knot_list <- vector("list", n_cov)
  bnd_list  <- vector("list", n_cov)
  n_bases   <- integer(n_cov)

  for (k in seq_len(n_cov)) {
    x <- Z_raw[, k]

    ## Determine knot positions
    bnd <- range(x, na.rm = TRUE)
    ## Add small buffer to boundaries to avoid edge issues
    bnd_buf <- bnd + c(-1, 1) * diff(bnd) * 0.001

    if (!is.null(knot_positions)) {
      ## User-specified interior knots for this covariate
      interior_knots <- sort(as.numeric(knot_positions[[k]]))
      n_knots_k <- length(interior_knots)
    } else {
      ## Default: equally-spaced quantiles
      interior_knots <- quantile(x, probs = seq(0, 1, length.out = n_knots + 2)[-c(1, n_knots + 2)],
                                 na.rm = TRUE)
      n_knots_k <- n_knots
    }

    ## B-spline basis
    B_k <- splines::bs(x,
                       knots     = interior_knots,
                       degree    = spline_deg,
                       Boundary.knots = bnd_buf,
                       intercept = FALSE)  # no intercept (alpha0 handles it)

    n_bases_k <- ncol(B_k)
    n_bases[k] <- n_bases_k

    ## 2nd-order difference penalty matrix
    ## D is the (n_bases - pen_order) x n_bases difference matrix
    D <- diff(diag(n_bases_k), differences = pen_order)
    S_k <- crossprod(D)  # D'D: n_bases_k x n_bases_k

    B_list[[k]] <- B_k
    S_list[[k]] <- S_k
    knot_list[[k]] <- interior_knots
    bnd_list[[k]]  <- bnd_buf

    cat(sprintf("    Covariate '%s': %d B-spline bases (degree %d, %d interior knots%s)\n",
                cov_names[k], n_bases_k, spline_deg, n_knots_k,
                if (!is.null(knot_positions)) " [user-specified]" else ""))
  }

  ## Stack into block-diagonal structure
  K_total <- sum(n_bases)
  B_alpha <- do.call(cbind, B_list)
  storage.mode(B_alpha) <- "double"

  ## Build block-diagonal penalty matrix
  S_alpha <- matrix(0, nrow = K_total, ncol = K_total)
  offset <- 0L
  for (k in seq_len(n_cov)) {
    idx <- (offset + 1):(offset + n_bases[k])
    S_alpha[idx, idx] <- S_list[[k]]
    offset <- offset + n_bases[k]
  }
  storage.mode(S_alpha) <- "double"

  ## Name columns
  col_names <- unlist(lapply(seq_len(n_cov), function(k) {
    paste0(cov_names[k], "_spl", seq_len(n_bases[k]))
  }))
  colnames(B_alpha) <- col_names

  cat(sprintf("  Alpha spline basis: %d sites x %d total bases (%d covariates)\n",
              n_sites, K_total, n_cov))

  list(
    B_alpha         = B_alpha,
    S_alpha         = S_alpha,
    n_bases_per_cov = n_bases,
    n_covariates    = n_cov,
    knot_list       = knot_list,
    boundary_list   = bnd_list,
    spline_deg      = spline_deg,
    pen_order       = pen_order,
    cov_names       = cov_names,
    col_names       = col_names
  )
}


# ---------------------------------------------------------------------------
# clesso_build_turnover_X
#
# Build the pairwise covariate matrix X for the turnover (beta) sub-model.
# For between-site pairs: X contains |spline(env_i) - spline(env_j)| using
# I-spline transformations (GDM-style monotonic basis).
# For within-site pairs: X = 0 (no environmental distance).
#
# Parameters:
#   pairs_dt       - data.table from clesso_sampler()
#   env_data       - data.table/data.frame of environmental variables per
#                    observation/site in the pairs. Must have same number
#                    of rows as pairs_dt. Columns are:
#                      env_i_* : env values at site i
#                      env_j_* : env values at site j
#                    OR provide env_site_table + use site lookup.
#   env_site_table - alternative: data.table with site_id + env columns.
#                    Environmental distances are computed from this.
#   geo_distance   - if TRUE, include geographic distance (default TRUE)
#   n_splines      - number of I-spline bases per env variable (default 3)
#   spline_quantiles - pre-computed quantile positions (NULL = auto)
#   splineData_fn  - spline function to use (default: splineData_fast from
#                    shared/R/gdm_functions.R)
#
# Returns:
#   list with:
#     X         - numeric matrix (nPairs x Kbeta)
#     splines   - integer vector of spline counts per variable
#     quantiles - numeric vector of knot positions
#     col_names - column names for X
# ---------------------------------------------------------------------------
clesso_build_turnover_X <- function(pairs_dt,
                                    env_data       = NULL,
                                    env_site_table = NULL,
                                    geo_distance   = TRUE,
                                    n_splines      = 3,
                                    spline_quantiles = NULL,
                                    splineData_fn  = NULL) {
  require(data.table)
  pairs_dt <- as.data.table(pairs_dt)
  n_pairs <- nrow(pairs_dt)

  ## -----------------------------------------------------------------------
  ## Build site-pair env values (site1 | site2 format for splineData)
  ## -----------------------------------------------------------------------

  if (!is.null(env_data)) {
    ## env_data provided directly (same row order as pairs_dt)
    env_data <- as.data.table(env_data)
    stopifnot(nrow(env_data) == n_pairs)

    ## Expect columns like env_i_varname and env_j_varname
    i_cols <- grep("^env_i_", names(env_data), value = TRUE)
    j_cols <- grep("^env_j_", names(env_data), value = TRUE)

    if (length(i_cols) == 0 || length(j_cols) == 0) {
      stop("env_data must have columns prefixed 'env_i_' and 'env_j_'")
    }

    env_i <- as.matrix(env_data[, ..i_cols])
    env_j <- as.matrix(env_data[, ..j_cols])
    var_names <- gsub("^env_i_", "", i_cols)

  } else if (!is.null(env_site_table)) {
    ## Look up env values from site table
    env_site_table <- as.data.table(env_site_table)
    stopifnot("site_id" %in% names(env_site_table))

    env_cols <- setdiff(names(env_site_table), "site_id")
    if (length(env_cols) == 0) stop("env_site_table has no environmental columns")

    ## Map pairs to site env values
    env_i <- as.matrix(env_site_table[match(pairs_dt$site_i, env_site_table$site_id),
                                       ..env_cols])
    env_j <- as.matrix(env_site_table[match(pairs_dt$site_j, env_site_table$site_id),
                                       ..env_cols])
    var_names <- env_cols

  } else {
    ## No env data — use geographic distance only
    if (!geo_distance) {
      stop("Must provide env_data, env_site_table, or set geo_distance=TRUE")
    }
    env_i <- matrix(nrow = n_pairs, ncol = 0)
    env_j <- matrix(nrow = n_pairs, ncol = 0)
    var_names <- character(0)
  }

  ## -----------------------------------------------------------------------
  ## Add geographic distance if requested
  ## -----------------------------------------------------------------------
  if (geo_distance) {
    geo_i <- cbind(pairs_dt$lon_i, pairs_dt$lat_i)
    geo_j <- cbind(pairs_dt$lon_j, pairs_dt$lat_j)
    colnames(geo_i) <- colnames(geo_j) <- c("geo_lon", "geo_lat")

    env_i <- cbind(env_i, geo_i)
    env_j <- cbind(env_j, geo_j)
    var_names <- c(var_names, "geo_lon", "geo_lat")
  }

  n_vars <- ncol(env_i)

  if (n_vars == 0) {
    warning("No covariates for turnover model. Returning empty X matrix.")
    return(list(
      X = matrix(0, nrow = n_pairs, ncol = 0),
      splines = integer(0),
      quantiles = numeric(0),
      col_names = character(0)
    ))
  }

  ## -----------------------------------------------------------------------
  ## Apply I-spline transformation via splineData
  ## -----------------------------------------------------------------------

  ## Build the stacked site1|site2 matrix expected by splineData
  colnames(env_i) <- var_names
  colnames(env_j) <- var_names
  site_pair_matrix <- cbind(env_i, env_j)

  splines_vec <- rep(as.integer(n_splines), n_vars)

  ## Try to use splineData_fast if available, else fall back
  if (is.null(splineData_fn)) {
    if (exists("splineData_fast", mode = "function")) {
      splineData_fn <- splineData_fast
    } else if (exists("splineData", mode = "function")) {
      splineData_fn <- splineData
    } else {
      stop("splineData function not found. Source shared/R/gdm_functions.R first.")
    }
  }

  cat(sprintf("  Building turnover X matrix: %d vars x %d splines = %d columns\n",
              n_vars, n_splines, sum(splines_vec)))

  X <- splineData_fn(site_pair_matrix,
                     splines = splines_vec,
                     quantiles = spline_quantiles)

  ## For within-site pairs, environmental distance should be 0
  ## (splineData computes |spline(s1) - spline(s2)| which should be 0
  ## when coords are identical, but enforce it explicitly)
  within_idx <- which(pairs_dt$pair_type == "within")
  if (length(within_idx) > 0) {
    ## Verify: within-site pairs should have near-zero env distance
    max_within_X <- max(abs(X[within_idx, ]))
    if (max_within_X > 1e-6) {
      cat(sprintf("  Note: max within-site X value = %.6f (forcing to 0)\n",
                  max_within_X))
    }
    X[within_idx, ] <- 0
  }

  storage.mode(X) <- "double"
  col_names <- colnames(X)

  ## Extract the quantiles that were used (for prediction)
  ## Re-derive since splineData may compute them internally
  if (is.null(spline_quantiles)) {
    spline_quantiles <- unlist(lapply(1:n_vars, function(v) {
      vals <- c(env_i[, v], env_j[, v])
      quantile(vals, seq(0, 1, length.out = n_splines))
    }))
  }

  list(
    X         = X,
    splines   = splines_vec,
    quantiles = spline_quantiles,
    col_names = col_names
  )
}


# ---------------------------------------------------------------------------
# clesso_prepare_model_data
#
# Main function: takes paired observations + covariates and produces
# the complete data and parameter lists for TMB.
#
# Parameters:
#   pairs_dt           - data.table from clesso_sampler()
#   site_covs          - site-level covariates for alpha (see clesso_build_site_table)
#   env_data           - pairwise env data (see clesso_build_turnover_X)
#   env_site_table     - site-level env data (see clesso_build_turnover_X)
#   geo_distance       - include geographic distance in turnover model
#   n_splines          - number of I-spline bases per predictor (turnover)
#   standardize_Z      - standardize alpha covariates
#   alpha_init         - initial estimate for mean alpha (default 20)
#   splineData_fn      - optional spline function override
#   use_alpha_splines  - if TRUE, build P-spline smooth terms for alpha
#   alpha_n_knots      - number of interior knots per alpha covariate spline
#   alpha_spline_deg   - B-spline degree for alpha smooths (default 3)
#   alpha_pen_order    - difference penalty order (default 2)
#
# Returns:
#   list with:
#     data_list      - list for TMB MakeADFun(data = ...)
#     parameters     - list for TMB MakeADFun(parameters = ...)
#     site_info      - site table + mapping
#     turnover_info  - spline metadata for X
#     alpha_info     - covariate metadata for Z + spline metadata
#     pairs_dt       - input pairs with site indices added
# ---------------------------------------------------------------------------
clesso_prepare_model_data <- function(pairs_dt,
                                      site_covs          = NULL,
                                      env_data           = NULL,
                                      env_site_table     = NULL,
                                      geo_distance       = TRUE,
                                      n_splines          = 3,
                                      standardize_Z      = TRUE,
                                      alpha_init         = 20,
                                      splineData_fn      = NULL,
                                      use_alpha_splines  = FALSE,
                                      alpha_n_knots      = 10,
                                      alpha_spline_deg   = 3,
                                      alpha_pen_order    = 2,
                                      alpha_knot_positions = NULL) {
  require(data.table)
  pairs_dt <- as.data.table(copy(pairs_dt))

  cat("\n=== Preparing CLESSO v2 model data ===\n")

  ## -----------------------------------------------------------------------
  ## 1. Build site table and Z matrix
  ## -----------------------------------------------------------------------
  cat("\n--- Building site table and alpha covariates (Z) ---\n")
  site_info <- clesso_build_site_table(pairs_dt, site_covs,
                                       standardize = standardize_Z)
  site_table <- site_info$site_table
  Z <- site_info$Z

  ## -----------------------------------------------------------------------
  ## 2. Map pair site IDs to 0-based indices
  ## -----------------------------------------------------------------------
  pairs_dt[site_table, site_i_idx := i.site_index, on = .(site_i = site_id)]
  pairs_dt[site_table, site_j_idx := i.site_index, on = .(site_j = site_id)]

  stopifnot(!anyNA(pairs_dt$site_i_idx), !anyNA(pairs_dt$site_j_idx))

  ## -----------------------------------------------------------------------
  ## 3. Build turnover covariate matrix X
  ## -----------------------------------------------------------------------
  cat("\n--- Building turnover covariates (X) ---\n")
  turnover_info <- clesso_build_turnover_X(
    pairs_dt       = pairs_dt,
    env_data       = env_data,
    env_site_table = env_site_table,
    geo_distance   = geo_distance,
    n_splines      = n_splines,
    splineData_fn  = splineData_fn
  )
  X <- turnover_info$X

  ## -----------------------------------------------------------------------
  ## 4. Build alpha spline basis (if requested)
  ## -----------------------------------------------------------------------
  alpha_spline_info <- NULL

  if (use_alpha_splines) {
    cat("\n--- Building alpha P-spline basis ---\n")

    ## Use un-standardised covariate values for spline basis construction
    ## (standardisation is for the linear Z; splines work on natural scale)
    Z_for_splines <- site_info$Z_raw
    if (is.null(Z_for_splines)) {
      warning("Z_raw not available in site_info; using standardised Z for splines")
      Z_for_splines <- Z
    }

    alpha_spline_info <- clesso_build_alpha_splines(
      Z_raw          = Z_for_splines,
      n_knots        = alpha_n_knots,
      spline_deg     = alpha_spline_deg,
      pen_order      = alpha_pen_order,
      cov_names      = site_info$alpha_cov_cols,
      knot_positions = alpha_knot_positions
    )

    B_alpha <- alpha_spline_info$B_alpha
    S_alpha <- alpha_spline_info$S_alpha
    K_basis <- ncol(B_alpha)
    n_lambda_blocks <- alpha_spline_info$n_covariates

  } else {
    ## No splines: pass dummy 1-column matrices to TMB (0-col not portable)
    ## The use_alpha_splines=0 flag tells TMB to ignore these
    nSites_tmp <- nrow(site_table)
    B_alpha <- matrix(0, nrow = nSites_tmp, ncol = 1)
    S_alpha <- matrix(0, nrow = 1, ncol = 1)
    K_basis <- 1L    # dummy dimension; ignored when use_alpha_splines=0
    n_lambda_blocks <- 1L
    storage.mode(B_alpha) <- "double"
    storage.mode(S_alpha) <- "double"
  }

  ## -----------------------------------------------------------------------
  ## 5. Assemble TMB data list
  ## -----------------------------------------------------------------------
  nSites <- nrow(site_table)
  Kbeta  <- ncol(X)
  Kalpha <- ncol(Z)

  data_list <- list(
    y                = as.numeric(pairs_dt$y),
    site_i           = as.integer(pairs_dt$site_i_idx),
    site_j           = as.integer(pairs_dt$site_j_idx),
    is_within        = as.integer(pairs_dt$is_within),
    X                = X,
    w                = as.numeric(pairs_dt$w),
    Z                = Z,
    B_alpha           = B_alpha,
    S_alpha           = S_alpha,
    alpha_block_sizes = if (use_alpha_splines) as.integer(alpha_spline_info$n_bases_per_cov)
                        else as.integer(1),
    use_alpha_splines = as.integer(use_alpha_splines)
  )

  ## -----------------------------------------------------------------------
  ## 6. Initial parameter values
  ## -----------------------------------------------------------------------
  parameters <- list(
    eta0_raw        = 0,
    beta_raw        = rep(log(0.01), Kbeta),
    alpha0          = log(alpha_init - 1),
    theta_alpha     = rep(0, Kalpha),
    b_alpha         = rep(0, K_basis),
    log_lambda_alpha = if (n_lambda_blocks > 0) rep(log(1.0), n_lambda_blocks) else numeric(0),
    u_site          = rep(0, nSites),
    log_sigma_u     = log(0.5)
  )

  ## -----------------------------------------------------------------------
  ## 7. Summary
  ## -----------------------------------------------------------------------
  cat(sprintf("\n--- Model data summary ---\n"))
  cat(sprintf("  Pairs:  %d (%d within, %d between)\n",
              nrow(pairs_dt), sum(pairs_dt$is_within), sum(!pairs_dt$is_within)))
  cat(sprintf("  Sites:  %d\n", nSites))
  cat(sprintf("  X dims: %d x %d (turnover covariates)\n", nrow(X), Kbeta))
  cat(sprintf("  Z dims: %d x %d (alpha linear covariates)\n", nrow(Z), Kalpha))
  if (use_alpha_splines) {
    cat(sprintf("  B_alpha dims: %d x %d (alpha spline basis)\n",
                nrow(B_alpha), K_basis))
    cat(sprintf("  Smoothing blocks: %d (one lambda per covariate)\n",
                n_lambda_blocks))
  } else {
    cat("  Alpha splines: disabled (linear model only)\n")
  }

  list(
    data_list        = data_list,
    parameters       = parameters,
    site_info        = site_info,
    turnover_info    = turnover_info,
    alpha_info       = list(
      cov_cols          = site_info$alpha_cov_cols,
      z_center          = site_info$z_center,
      z_scale           = site_info$z_scale,
      use_alpha_splines = use_alpha_splines,
      spline_info       = alpha_spline_info
    ),
    pairs_dt         = pairs_dt
  )
}
