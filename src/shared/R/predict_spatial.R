##############################################################################
##
## predict_spatial.R -- Spatial GDM prediction & biological-space RGB mapping
##
## Purpose:
##   Given a fitted STresiduals GDM and environmental rasters, transform
##   each pixel through the fitted I-spline basis for spatial predictors
##   (climate + substrate) and produce an RGB "biological turnover" map.
##
## Three prediction approaches are provided:
##
##   A. Transform -> PCA -> RGB  [predict_spatial_rgb / predict_spatiotemporal_rgb]
##      Standard approach (same as gdm::gdm.transform -> PCA):
##      Transform each pixel independently through fitted I-splines weighted
##      by coefficients.  PCA on the transformed n x p matrix -> 3 PCs -> RGB.
##      Pixels with similar colour = similar predicted biological community.
##      Complexity: O(n · p) for transform + O(n · p²) for PCA ≈ seconds.
##      Caveat: skips the nonlinear calibration (intercept -> logit -> ObsTrans).
##
##   B. Landmark MDS -> RGB  [predict_spatial_lmds]
##      Select k landmark pixels (~500).  Compute true GDM dissimilarity
##      (ecological distance -> logit -> ObsTrans) between all landmark pairs
##      (k × k) and from every pixel to each landmark (n × k).
##      Classical MDS on the k × k matrix + Nyström extension embeds all
##      pixels into ordination space -> 3 dims -> RGB.
##      Preserves the full nonlinear calibration.
##      Complexity: O(n · k · p) ≈ minutes for n=275k, k=500.
##
##   C. Dissimilarity from reference  [spatial_dissimilarity_from_ref]
##      Pick one or more reference pixels.  For each, compute calibrated
##      GDM dissimilarity to every other pixel -> continuous raster [0, 1].
##      Answers: "How different is each pixel from this location?"
##      Complexity: O(n · p) per reference pixel ≈ sub-second.
##
## Mathematical basis:
##   GDM ecological distance decomposes into a Manhattan (L1) distance in
##   the β-weighted I-spline space:
##
##     d(i,j) = Σ_k Σ_s β_ks |I_s(x_ik) - I_s(x_jk)|
##
##   Because β ≥ 0 (NNLS), the β-weighted transform is monotone, so:
##     d(i,j) = Σ |t_ik − t_jk|   where t = β · I_spline(env)
##
##   This means each pixel can be transformed independently -> O(n . p),
##   and pairwise distances are just L1 norms in the transformed space.
##
## Prerequisites:
##   gdm_functions.R   (I_spline)
##   utils.R            (inv.logit, ObsTrans)
##
## Functions:
##   transform_spatial_gdm()         -- I-spline transform all pixels (core)
##   transform_temporal_change_gdm() -- temporal change (signed or absolute)
##   predict_spatial_rgb()           -- [A] Transform -> PCA -> RGB
##   predict_spatiotemporal_rgb()    -- [A] Spatio-temporal -> PCA -> RGB
##                                      Options: signed diffs, α-weighted mixing
##   predict_spatiotemporal_hsl()    -- [D] HSL bivariate map:
##                                      Hue = community type (spatial PCA)
##                                      Lightness = temporal change magnitude
##   .hsl_to_rgb_vec()              -- (internal) HSL -> RGB conversion
##   predict_spatial_lmds()          -- [B] Landmark MDS -> RGB
##   spatial_dissimilarity_from_ref()-- [C] Dissimilarity from ref pixel(s)
##   spatial_dissimilarity()         -- Pairwise dissimilarity (env1 vs env2)
##
##############################################################################


# ---------------------------------------------------------------------------
# .extract_modis_columns  (internal helper)
#
# Extract MODIS land cover values at pixel locations for a given year.
# Returns a data.frame with columns named "{col_prefix}_{var}{col_suffix}".
#
# Used by the predict_*() functions when the fitted model includes MODIS
# covariates (fit$add_modis == TRUE).
#
# Parameters:
#   fit            - fitted GDM list (must contain modis_variables,
#                    modis_year_range)
#   xy             - matrix [n × 2] of lon/lat coordinates
#   year           - the prediction year (will be clamped to MODIS range)
#   modis_dir      - path to directory containing MODIS COG TIFs
#   modis_resolution - resolution string for filename (e.g. "1km")
#   col_prefix     - column name prefix, e.g. "spat_modis" or "temp_modis"
#   col_suffix     - column name suffix, e.g. "_1" or "_2"
#   verbose        - print progress
#
# Returns:
#   A data.frame with columns named "{col_prefix}_{var}{col_suffix}"
#   (one per MODIS variable).
# ---------------------------------------------------------------------------
.extract_modis_columns <- function(
    fit,
    xy,
    year,
    modis_dir,
    modis_resolution = "1km",
    col_prefix       = "spat_modis",
    col_suffix       = "_1",
    verbose          = TRUE
) {
  modis_vars   <- fit$modis_variables
  modis_range  <- fit$modis_year_range  # c(start, end)
  yr_clamped   <- min(max(as.integer(year), modis_range[1]), modis_range[2])

  n_pixels <- nrow(xy)
  result   <- data.frame(matrix(NA_real_, nrow = n_pixels,
                                 ncol = length(modis_vars)))
  colnames(result) <- paste0(col_prefix, "_", modis_vars, col_suffix)

  if (verbose) cat(sprintf("  Extracting MODIS (%s, year = %d -> clamped %d)...\n",
                            col_prefix, as.integer(year), yr_clamped))

  pts <- sp::SpatialPoints(as.data.frame(xy))

  for (vi in seq_along(modis_vars)) {
    mv    <- modis_vars[vi]
    fname <- file.path(modis_dir,
                       paste0("modis_", yr_clamped, "_", mv, "_",
                              modis_resolution, "_COG.tif"))
    if (!file.exists(fname)) {
      warning(sprintf("MODIS raster not found: %s -- filling with NA", fname))
      next
    }
    ras <- raster::raster(fname)
    result[, vi] <- raster::extract(ras, pts)
    if (verbose) cat(sprintf("    %s_%s%s: %d non-NA values\n",
                              col_prefix, mv, col_suffix,
                              sum(!is.na(result[, vi]))))
  }

  result
}


# ---------------------------------------------------------------------------
# .extract_condition_columns  (internal helper)
#
# Extract site-condition values at pixel locations for a given year from a
# single multi-band GeoTIFF (band i = condition_start_year + i - 1).
# Returns a data.frame with a single column named
#   "{col_prefix}{col_suffix}"  (e.g. "spat_condition_1").
#
# Parameters:
#   fit            - fitted GDM list (must contain condition_variable,
#                    condition_year_range, condition_tif_path)
#   xy             - matrix [n × 2] of lon/lat coordinates
#   year           - the prediction year (clamped to condition range)
#   condition_tif  - path to the multi-band condition GeoTIFF
#                    (overrides fit$condition_tif_path when provided)
#   col_prefix     - column name prefix, e.g. "spat_condition" or
#                    "temp_condition"
#   col_suffix     - column name suffix, e.g. "_1" or "_2"
#   verbose        - print progress
#
# Returns:
#   A data.frame with one column per condition variable (currently just one).
# ---------------------------------------------------------------------------
.extract_condition_columns <- function(
    fit,
    xy,
    year,
    condition_tif  = NULL,
    col_prefix     = "spat_condition",
    col_suffix     = "_1",
    verbose        = TRUE
) {
  cond_range <- fit$condition_year_range           # c(start, end)
  yr_clamped <- min(max(as.integer(year), cond_range[1]), cond_range[2])
  band_idx   <- yr_clamped - cond_range[1] + 1L

  tif_path <- if (!is.null(condition_tif)) condition_tif else fit$condition_tif_path
  if (is.null(tif_path) || !file.exists(tif_path)) {
    warning(sprintf("Condition TIF not found: %s -- filling with NA", tif_path))
    n_pixels <- nrow(xy)
    result   <- data.frame(matrix(NA_real_, nrow = n_pixels, ncol = 1))
    colnames(result) <- paste0(col_prefix, col_suffix)
    return(result)
  }

  if (verbose) cat(sprintf("  Extracting condition (%s, year = %d -> clamped %d, band %d)...\n",
                            col_prefix, as.integer(year), yr_clamped, band_idx))

  brk <- raster::brick(tif_path)
  ras <- brk[[band_idx]]
  pts <- sp::SpatialPoints(as.data.frame(xy))
  vals <- raster::extract(ras, pts)

  result <- data.frame(vals)
  colnames(result) <- paste0(col_prefix, col_suffix)

  if (verbose) cat(sprintf("    %s%s: %d non-NA values\n",
                            col_prefix, col_suffix,
                            sum(!is.na(vals))))
  result
}


# ---------------------------------------------------------------------------
# transform_spatial_gdm
#
# Transform environmental values at each pixel through the fitted I-spline
# basis functions for spatial predictors (spatial climate + substrate).
#
# This is the core operation: each pixel i gets mapped to a vector of
# β-weighted I-spline values, placing it in "GDM biological space".
#
# Parameters:
#   fit       - fitted GDM list object (with predictors, coefficients,
#               quantiles, splines)
#   env_df    - data.frame with ONE row per pixel and columns matching
#               spatial+substrate predictor names (with _1 suffix).
#               Column names must match fit$predictors for spatial/substrate.
#   weight_by_coef - if TRUE (default), multiply each spline column by its
#               coefficient. This gives the "biological importance" weighting.
#               Set to FALSE for unweighted spline values.
#   spatial_only - if TRUE (default), only transform spatial (spat_*) and
#               substrate predictors. Temporal predictors get zero columns.
#   verbose   - print progress
#
# Returns:
#   A matrix [n_pixels × total_splines] of transformed values.
#   Only spatial/substrate columns are filled; temporal columns are zero.
#   Attribute "predictor_info" contains a data.frame of predictor metadata.
# ---------------------------------------------------------------------------
transform_spatial_gdm <- function(
    fit,
    env_df,
    weight_by_coef = TRUE,
    spatial_only   = TRUE,
    verbose        = TRUE
) {

  n_pts         <- nrow(env_df)
  n_preds       <- length(fit$predictors)
  total_splines <- sum(fit$splines)
  csp           <- c(0, cumsum(fit$splines))

  ## Identify predictor types
  temp_idx <- grep("^temp_", fit$predictors)
  spat_idx <- grep("^spat_", fit$predictors)
  subs_idx <- setdiff(seq_len(n_preds), c(temp_idx, spat_idx))

  if (spatial_only) {
    active_idx <- c(spat_idx, subs_idx)
  } else {
    active_idx <- seq_len(n_preds)
  }

  if (verbose) {
    cat(sprintf("  Transforming %d pixels through %d active predictors (%d spline cols)\n",
                n_pts, length(active_idx), sum(fit$splines[active_idx])))
  }

  ## Pre-allocate output
  result <- matrix(0, nrow = n_pts, ncol = total_splines)

  ## Column names
  spl_names <- character(total_splines)
  k <- 0L
  for (i in seq_len(n_preds)) {
    for (s in seq_len(fit$splines[i])) {
      k <- k + 1L
      spl_names[k] <- paste0(fit$predictors[i], "_spl", s)
    }
  }
  colnames(result) <- spl_names

  ## Track predictor info
  pred_info <- data.frame(
    idx       = seq_len(n_preds),
    predictor = fit$predictors,
    type      = ifelse(seq_len(n_preds) %in% temp_idx, "temporal",
                       ifelse(seq_len(n_preds) %in% spat_idx, "spatial", "substrate")),
    n_splines = fit$splines,
    coef_sum  = NA_real_,
    matched   = FALSE,
    stringsAsFactors = FALSE
  )

  matched <- 0L
  for (i in active_idx) {
    pred_name <- fit$predictors[i]
    ns        <- fit$splines[i]
    coefs_i   <- fit$coefficients[(csp[i] + 1):(csp[i] + ns)]
    quants_i  <- fit$quantiles[(csp[i] + 1):(csp[i] + ns)]
    pred_info$coef_sum[i] <- sum(coefs_i)

    ## Match predictor name to env columns
    col_match <- which(names(env_df) == pred_name)
    if (length(col_match) == 0) {
      ## Try without _1 suffix
      pred_base <- sub("_1$", "", pred_name)
      col_match <- which(names(env_df) == pred_base)
    }

    if (length(col_match) == 0) {
      if (verbose) cat(sprintf("    [SKIP] Cannot match: %s\n", pred_name))
      next
    }

    vals <- env_df[, col_match[1]]
    matched <- matched + 1L
    pred_info$matched[i] <- TRUE

    ## Apply I-spline basis
    for (sp in seq_len(ns)) {
      if (sp == 1L) {
        q1 <- quants_i[1]; q2 <- quants_i[1]; q3 <- quants_i[min(2, ns)]
      } else if (sp == ns) {
        q1 <- quants_i[max(1, ns - 1)]; q2 <- quants_i[ns]; q3 <- quants_i[ns]
      } else {
        q1 <- quants_i[sp - 1]; q2 <- quants_i[sp]; q3 <- quants_i[sp + 1]
      }

      spl_val <- I_spline(vals, q1, q2, q3)

      if (weight_by_coef) {
        result[, csp[i] + sp] <- spl_val * coefs_i[sp]
      } else {
        result[, csp[i] + sp] <- spl_val
      }
    }
  }

  if (verbose) cat(sprintf("  Matched %d / %d active predictors\n", matched, length(active_idx)))

  attr(result, "predictor_info") <- pred_info
  result
}


# ---------------------------------------------------------------------------
# predict_spatial_rgb
#
# Full pipeline: extract environmental data per pixel -> I-spline transform
# -> PCA -> assign RGB colours -> return rasters.
#
# Parameters:
#   fit          - fitted GDM list object
#   ref_raster   - path or RasterLayer for the reference raster (defines grid)
#   subs_raster  - path or RasterBrick for substrate layers
#   env_spatial  - EITHER: a pre-extracted data.frame of spatial climate env
#                  values with columns matching spat_* predictor names,
#                  OR NULL to extract from geonpy (requires npy_src etc.)
#   npy_src      - path to geonpy .npy directory (required if env_spatial is NULL)
#   python_exe   - Python executable (required if env_spatial is NULL)
#   pyper_script - path to pyper.py (required if env_spatial is NULL)
#   ref_year     - reference year for spatial climate extraction (default: 1975)
#   ref_month    - reference month (default: 6)
#   pca_method   - "prcomp" (standard PCA, default), "robust" (robust PCA via
#                  MASS::cov.mcd), or "none" (return raw transformed values)
#   n_components - number of PCA components (default: 3 for RGB)
#   stretch      - percentile stretch for RGB: linearly map [p, 100-p] -> [0,1]
#                  (default: 2 = 2nd/98th percentile stretch)
#   chunk_size   - pixels per chunk for climate extraction (default: 50000)
#   verbose      - print progress
#
# Returns:
#   A list with:
#     $rgb_stack    - RasterBrick with 3 layers (R, G, B) in [0, 255]
#     $pca          - the prcomp object (or NULL if pca_method = "none")
#     $pca_scores   - matrix of PCA scores [n_pixels × n_components]
#     $transformed  - the full transformed matrix from transform_spatial_gdm
#     $coords       - data.frame with lon, lat, cell index of non-NA pixels
#     $env_df       - the combined environmental data.frame
#     $variance_explained - variance explained by each PC (%)
# ---------------------------------------------------------------------------
predict_spatial_rgb <- function(
    fit,
    ref_raster,
    subs_raster,
    env_spatial      = NULL,
    npy_src          = NULL,
    python_exe       = NULL,
    pyper_script     = NULL,
    ref_year         = 1975L,
    ref_month        = 6L,
    modis_dir        = NULL,
    modis_resolution = "1km",
    condition_tif    = NULL,
    pca_method       = "prcomp",
    n_components     = 3L,
    stretch          = 2,
    chunk_size       = 50000L,
    verbose          = TRUE
) {

  cat("=== Spatial GDM Prediction -> RGB ===\n\n")
  t0 <- proc.time()

  ## ---- 1. Load rasters ------------------------------------------------
  if (verbose) cat("--- 1. Loading rasters ---\n")
  if (is.character(ref_raster))  ref_raster  <- raster::raster(ref_raster)
  if (is.character(subs_raster)) subs_raster <- raster::brick(subs_raster)

  ## Non-NA pixel coordinates
  cell_idx <- which(!is.na(raster::values(ref_raster)))
  xy       <- raster::xyFromCell(ref_raster, cell_idx)
  n_pixels <- length(cell_idx)
  cat(sprintf("  Non-NA pixels: %d\n", n_pixels))

  coords <- data.frame(
    lon  = xy[, 1],
    lat  = xy[, 2],
    cell = cell_idx,
    stringsAsFactors = FALSE
  )

  ## ---- 2. Extract substrate values ------------------------------------
  if (verbose) cat("\n--- 2. Extracting substrate values ---\n")
  subs_vals <- raster::extract(subs_raster, xy)
  colnames(subs_vals) <- paste0(colnames(subs_vals), "_1")
  cat(sprintf("  Substrate: %d layers × %d pixels\n", ncol(subs_vals), n_pixels))

  ## ---- 3. Extract spatial climate values ------------------------------
  ## Identify spatial climate predictors
  spat_preds <- fit$predictors[grep("^spat_", fit$predictors)]

  if (is.null(env_spatial)) {
    if (is.null(npy_src) || is.null(python_exe) || is.null(pyper_script))
      stop("env_spatial is NULL -- must provide npy_src, python_exe, pyper_script for climate extraction.")

    if (verbose) cat(sprintf("\n--- 3. Extracting spatial climate (ref year = %d) ---\n", ref_year))
    require(arrow)

    ## Build spatial extraction params (same site, same year = purely spatial)
    spatial_params <- list()
    for (ep in fit$env_params) {
      spatial_params[[length(spatial_params) + 1]] <- list(
        variables = ep$variables,
        mstat     = ep$mstat,
        cstat     = ep$cstat,
        window    = fit$climate_window,
        prefix    = paste0("spat_", ep$cstat)
      )
    }

    ## Process in chunks to manage memory
    n_chunks <- ceiling(n_pixels / chunk_size)
    env_parts_list <- vector("list", n_chunks)

    for (ch in seq_len(n_chunks)) {
      row_start <- (ch - 1) * chunk_size + 1
      row_end   <- min(ch * chunk_size, n_pixels)
      chunk_xy  <- xy[row_start:row_end, , drop = FALSE]
      n_ch      <- nrow(chunk_xy)

      if (verbose) cat(sprintf("  Chunk %d/%d (%d pixels) ...\n", ch, n_chunks, n_ch))

      ## Same-site same-year pairs (spatial only)
      pairs <- data.frame(
        Lon1   = chunk_xy[, 1],
        Lat1   = chunk_xy[, 2],
        year1  = rep(as.integer(ref_year), n_ch),
        month1 = rep(as.integer(ref_month), n_ch),
        Lon2   = chunk_xy[, 1],
        Lat2   = chunk_xy[, 2],
        year2  = rep(as.integer(ref_year), n_ch),
        month2 = rep(as.integer(ref_month), n_ch)
      )

      chunk_env_parts <- list()
      for (j in seq_along(spatial_params)) {
        sp <- spatial_params[[j]]

        raw <- gen_windows(
          pairs        = pairs,
          variables    = sp$variables,
          mstat        = sp$mstat,
          cstat        = sp$cstat,
          window       = sp$window,
          npy_src      = npy_src,
          start_year   = if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L,
          python_exe   = python_exe,
          pyper_script = pyper_script
        )

        ## Keep site-1 columns only (same site, so _1 and _2 are identical)
        env_cols <- raw[, grep("_1$", names(raw)), drop = FALSE]
        colnames(env_cols) <- paste(sp$prefix, colnames(env_cols), sep = "_")
        ## Strip date-range patterns
        colnames(env_cols) <- gsub("\\d{6}-\\d{6}_", "", colnames(env_cols))
        chunk_env_parts[[j]] <- env_cols
      }

      env_parts_list[[ch]] <- do.call(cbind, chunk_env_parts)
    }

    env_spatial <- do.call(rbind, env_parts_list)
    cat(sprintf("  Spatial climate: %d columns × %d pixels\n", ncol(env_spatial), nrow(env_spatial)))

  } else {
    if (verbose) cat("\n--- 3. Using pre-extracted spatial climate data ---\n")
    cat(sprintf("  Spatial climate: %d columns × %d pixels\n", ncol(env_spatial), nrow(env_spatial)))
  }

  ## ---- 3b. Extract MODIS land cover (spatial component) ----------------
  modis_spat_vals <- NULL
  if (isTRUE(fit$add_modis)) {
    if (is.null(modis_dir)) {
      ## Try to pick up from config in the calling environment
      if (exists("config", envir = parent.frame()) &&
          !is.null(parent.frame()$config$modis_dir)) {
        modis_dir        <- parent.frame()$config$modis_dir
        modis_resolution <- parent.frame()$config$modis_resolution
      } else {
        warning("fit$add_modis is TRUE but modis_dir not provided -- MODIS predictors will be skipped")
      }
    }
    if (!is.null(modis_dir)) {
      if (verbose) cat(sprintf("\n--- 3b. Extracting MODIS spatial (year = %d) ---\n", ref_year))
      modis_spat_vals <- .extract_modis_columns(
        fit              = fit,
        xy               = xy,
        year             = ref_year,
        modis_dir        = modis_dir,
        modis_resolution = modis_resolution,
        col_prefix       = "spat_modis",
        col_suffix       = "_1",
        verbose          = verbose
      )
      cat(sprintf("  MODIS spatial: %d columns × %d pixels\n",
                  ncol(modis_spat_vals), nrow(modis_spat_vals)))
    }
  }

  ## ---- 3c. Extract condition raster (spatial component) ----------------
  cond_spat_vals <- NULL
  if (isTRUE(fit$add_condition)) {
    if (is.null(condition_tif)) {
      if (exists("config", envir = parent.frame()) &&
          !is.null(parent.frame()$config$condition_tif_path)) {
        condition_tif <- parent.frame()$config$condition_tif_path
      } else if (!is.null(fit$condition_tif_path)) {
        condition_tif <- fit$condition_tif_path
      } else {
        warning("fit$add_condition is TRUE but condition_tif not provided -- condition predictors will be skipped")
      }
    }
    if (!is.null(condition_tif)) {
      if (verbose) cat(sprintf("\n--- 3c. Extracting condition spatial (year = %d) ---\n", ref_year))
      cond_spat_vals <- .extract_condition_columns(
        fit           = fit,
        xy            = xy,
        year          = ref_year,
        condition_tif = condition_tif,
        col_prefix    = "spat_condition",
        col_suffix    = "_1",
        verbose       = verbose
      )
      cat(sprintf("  Condition spatial: %d columns × %d pixels\n",
                  ncol(cond_spat_vals), nrow(cond_spat_vals)))
    }
  }

  ## ---- 4. Combine env data --------------------------------------------
  if (verbose) cat("\n--- 4. Combining environmental data ---\n")
  env_df <- cbind(env_spatial, as.data.frame(subs_vals))
  if (!is.null(modis_spat_vals)) env_df <- cbind(env_df, modis_spat_vals)
  if (!is.null(cond_spat_vals))  env_df <- cbind(env_df, cond_spat_vals)

  ## Remove pixels with NA environment
  na_rows <- is.na(rowSums(env_df))
  sentinel_rows <- apply(env_df, 1, function(r) any(r == -9999, na.rm = TRUE))
  bad_rows <- na_rows | sentinel_rows

  if (any(bad_rows)) {
    cat(sprintf("  Removing %d pixels with NA/sentinel values (%d remain)\n",
                sum(bad_rows), sum(!bad_rows)))
    env_df <- env_df[!bad_rows, , drop = FALSE]
    coords <- coords[!bad_rows, , drop = FALSE]
    n_pixels <- nrow(env_df)
  }

  ## ---- 5. I-spline transform ------------------------------------------
  if (verbose) cat("\n--- 5. I-spline transformation ---\n")
  transformed <- transform_spatial_gdm(
    fit            = fit,
    env_df         = env_df,
    weight_by_coef = TRUE,
    spatial_only   = TRUE,
    verbose        = verbose
  )

  ## Drop zero-variance columns (temporal cols are all zero; also any
  ## constant spatial cols)
  col_var <- apply(transformed, 2, var, na.rm = TRUE)
  active_cols <- which(col_var > 1e-15)
  transformed_active <- transformed[, active_cols, drop = FALSE]

  if (verbose) cat(sprintf("  Active (non-zero-variance) columns: %d / %d\n",
                            length(active_cols), ncol(transformed)))

  ## ---- 6. PCA ---------------------------------------------------------
  if (verbose) cat(sprintf("\n--- 6. PCA (%s, %d components) ---\n", pca_method, n_components))

  pca_obj <- NULL
  var_explained <- NULL

  if (pca_method == "none") {
    ## No PCA, just use the first n_components columns
    scores <- transformed_active[, seq_len(min(n_components, ncol(transformed_active))), drop = FALSE]
  } else if (pca_method == "robust") {
    ## Robust PCA via MASS
    if (!requireNamespace("MASS", quietly = TRUE))
      stop("Package 'MASS' required for robust PCA")
    centre <- colMeans(transformed_active, na.rm = TRUE)
    scaled <- scale(transformed_active, center = centre, scale = FALSE)
    cov_rob <- MASS::cov.mcd(scaled, quantile.used = floor(0.75 * nrow(scaled)))
    eig     <- eigen(cov_rob$cov, symmetric = TRUE)
    loadings <- eig$vectors[, seq_len(n_components), drop = FALSE]
    scores   <- scaled %*% loadings
    var_explained <- 100 * eig$values[seq_len(n_components)] / sum(eig$values)
  } else {
    ## Standard PCA
    pca_obj <- prcomp(transformed_active, center = TRUE, scale. = FALSE,
                       rank. = n_components)
    scores  <- pca_obj$x[, seq_len(n_components), drop = FALSE]
    var_explained <- 100 * pca_obj$sdev[seq_len(n_components)]^2 / sum(pca_obj$sdev^2)
  }

  if (!is.null(var_explained) && verbose) {
    cat(sprintf("  Variance explained: PC1=%.1f%%, PC2=%.1f%%, PC3=%.1f%% (total=%.1f%%)\n",
                var_explained[1], var_explained[2], var_explained[3], sum(var_explained)))
  }

  ## ---- 7. Map to RGB [0, 255] with percentile stretch -----------------
  if (verbose) cat(sprintf("\n--- 7. Mapping to RGB (stretch = %g%% percentile) ---\n", stretch))

  rgb_vals <- matrix(NA_real_, nrow = n_pixels, ncol = n_components)
  for (k in seq_len(n_components)) {
    v <- scores[, k]
    lo <- quantile(v, stretch / 100, na.rm = TRUE)
    hi <- quantile(v, 1 - stretch / 100, na.rm = TRUE)
    if (hi <= lo) { lo <- min(v, na.rm = TRUE); hi <- max(v, na.rm = TRUE) }
    v_scaled <- (v - lo) / (hi - lo)
    v_scaled <- pmin(pmax(v_scaled, 0), 1)
    rgb_vals[, k] <- round(v_scaled * 255)
  }

  ## ---- 8. Build output rasters ----------------------------------------
  if (verbose) cat("\n--- 8. Building RGB rasters ---\n")

  template <- raster::raster(ref_raster)

  r_layer <- template; raster::values(r_layer) <- NA
  g_layer <- template; raster::values(g_layer) <- NA
  b_layer <- template; raster::values(b_layer) <- NA

  r_layer[coords$cell] <- rgb_vals[, 1]
  g_layer[coords$cell] <- rgb_vals[, 2]
  b_layer[coords$cell] <- rgb_vals[, 3]

  rgb_stack <- raster::stack(r_layer, g_layer, b_layer)
  names(rgb_stack) <- c("PC1_red", "PC2_green", "PC3_blue")

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("\n=== Spatial prediction complete (%.1f s) ===\n", elapsed))

  list(
    rgb_stack          = rgb_stack,
    pca                = pca_obj,
    pca_scores         = scores,
    transformed        = transformed,
    transformed_active = transformed_active,
    coords             = coords,
    env_df             = env_df,
    variance_explained = var_explained,
    rgb_vals           = rgb_vals,
    active_cols        = active_cols
  )
}


# ---------------------------------------------------------------------------
# transform_temporal_change_gdm
#
# For each pixel, compute the temporal I-spline CHANGE between year1 and
# year2: |I_spline(env@year1) - I_spline(env@year2)|, weighted by β.
#
# This complements transform_spatial_gdm(): the spatial transform gives a
# pixel its "biological position" while this gives its "temporal shift".
#
# Parameters:
#   fit      - fitted GDM list object
#   env_yr1  - data.frame: temporal climate env at year1 per pixel.
#              Column names must match temporal predictor names (temp_*_1).
#   env_yr2  - data.frame: temporal climate env at year2 per pixel.
#              Column names must match temporal predictor names (temp_*_2).
#   weight_by_coef - if TRUE, weight by fitted coefficients (default: TRUE)
#   signed   - if TRUE, compute signed difference I(yr2) - I(yr1) instead
#              of absolute |I(yr1) - I(yr2)|. Signed differences preserve
#              the direction of biological change (positive = increase at
#              yr2, negative = decrease). Default: FALSE (absolute, for
#              backward compatibility with GDM dissimilarity convention).
#   verbose  - print progress
#
# Returns:
#   Matrix [n_pixels × total_splines]. Only temporal predictor columns are
#   filled; spatial/substrate columns are zero.
# ---------------------------------------------------------------------------
transform_temporal_change_gdm <- function(
    fit,
    env_yr1,
    env_yr2,
    weight_by_coef = TRUE,
    signed         = FALSE,
    verbose        = TRUE
) {

  n_pts         <- nrow(env_yr1)
  n_preds       <- length(fit$predictors)
  total_splines <- sum(fit$splines)
  csp           <- c(0, cumsum(fit$splines))

  temp_idx <- grep("^temp_", fit$predictors)

  if (verbose) cat(sprintf("  Transforming temporal change: %d pixels × %d temporal predictors (%s)\n",
                            n_pts, length(temp_idx),
                            if (signed) "signed" else "absolute"))

  result <- matrix(0, nrow = n_pts, ncol = total_splines)

  ## Column names
  spl_names <- character(total_splines)
  k <- 0L
  for (i in seq_len(n_preds)) {
    for (s in seq_len(fit$splines[i])) {
      k <- k + 1L
      spl_names[k] <- paste0(fit$predictors[i], "_spl", s)
    }
  }
  colnames(result) <- spl_names

  matched <- 0L
  for (i in temp_idx) {
    pred_name <- fit$predictors[i]
    ns        <- fit$splines[i]
    coefs_i   <- fit$coefficients[(csp[i] + 1):(csp[i] + ns)]
    quants_i  <- fit$quantiles[(csp[i] + 1):(csp[i] + ns)]

    ## Match to env columns
    col_1 <- which(names(env_yr1) == pred_name)
    if (length(col_1) == 0) col_1 <- which(names(env_yr1) == sub("_1$", "", pred_name))

    pred_2 <- sub("_1$", "_2", pred_name)
    col_2 <- which(names(env_yr2) == pred_2)
    if (length(col_2) == 0) col_2 <- which(names(env_yr2) == sub("_2$", "", pred_2))
    if (length(col_2) == 0) col_2 <- which(names(env_yr2) == pred_name)

    if (length(col_1) == 0 || length(col_2) == 0) {
      if (verbose) cat(sprintf("    [SKIP] Cannot match temporal predictor: %s\n", pred_name))
      next
    }

    v1 <- env_yr1[, col_1[1]]
    v2 <- env_yr2[, col_2[1]]
    matched <- matched + 1L

    for (sp in seq_len(ns)) {
      if (sp == 1L) {
        q1 <- quants_i[1]; q2 <- quants_i[1]; q3 <- quants_i[min(2, ns)]
      } else if (sp == ns) {
        q1 <- quants_i[max(1, ns - 1)]; q2 <- quants_i[ns]; q3 <- quants_i[ns]
      } else {
        q1 <- quants_i[sp - 1]; q2 <- quants_i[sp]; q3 <- quants_i[sp + 1]
      }

      spl_yr1 <- I_spline(v1, q1, q2, q3)
      spl_yr2 <- I_spline(v2, q1, q2, q3)
      spl_diff <- if (signed) (spl_yr2 - spl_yr1) else abs(spl_yr1 - spl_yr2)

      if (weight_by_coef) {
        result[, csp[i] + sp] <- spl_diff * coefs_i[sp]
      } else {
        result[, csp[i] + sp] <- spl_diff
      }
    }
  }

  if (verbose) cat(sprintf("  Matched %d / %d temporal predictors\n", matched, length(temp_idx)))
  result
}


# ---------------------------------------------------------------------------
# predict_spatiotemporal_rgb
#
# Full spatio-temporal prediction: combine spatial "biological position" with
# temporal "biological change" into a single transformed space, then PCA -> RGB.
#
# For each pixel we build a vector comprising:
#   [spatial I-spline positions (weighted)] + [substrate I-spline positions] +
#   [temporal I-spline |year1 - year2| changes (weighted)]
#
# Pixels with the same colour share similar biological communities AND have
# undergone similar temporal change.
#
# Parameters:
#   fit          - fitted GDM list
#   ref_raster   - reference raster (defines grid)
#   subs_raster  - substrate raster brick
#   npy_src, python_exe, pyper_script - for geonpy climate extraction
#   year1        - start year for temporal pairs (integer)
#   year2        - end year for temporal pairs (integer)
#   ref_year     - reference year for spatial climate (default: year1)
#   ref_month    - reference month (default: 6)
#   env_spatial  - pre-extracted spatial env (or NULL to extract)
#   env_temporal_yr1 - pre-extracted temporal env at year1 (or NULL)
#   env_temporal_yr2 - pre-extracted temporal env at year2 (or NULL)
#   signed_temporal - if TRUE, use signed temporal differences
#                  I(yr2) - I(yr1) instead of |I(yr1) - I(yr2)|.
#                  Preserves direction of change (default: FALSE).
#   alpha    - mixing weight for spatial vs temporal, in [0, 1].
#              0 = spatial only, 1 = temporal only. Default: NULL (no
#              normalisation -- original behaviour, simple concatenation).
#              When set, both blocks are normalised to unit column
#              variance before weighting by sqrt(1-alpha) and sqrt(alpha).
#   normalise_blocks - if TRUE AND alpha is not NULL, normalise each block
#              (spatial, temporal) to unit column variance before weighting.
#              Default: TRUE. Set FALSE to skip normalisation even when
#              alpha is specified (useful for sensitivity checks).
#   pca_method   - "prcomp" (default), "robust", or "none"
#   n_components - PCA components (default: 3)
#   stretch      - percentile stretch for RGB (default: 2)
#   chunk_size   - pixels per extraction chunk (default: 50000)
#   verbose      - print progress
#
# Returns:
#   A list with:
#     $rgb_stack          - 3-band RasterBrick [0, 255]
#     $pca                - prcomp object (or NULL)
#     $pca_scores         - PCA scores [n_pixels × 3]
#     $transformed_spatial  - spatial+substrate I-spline matrix
#     $transformed_temporal - temporal change I-spline matrix
#     $transformed_full     - concatenated active columns (input to PCA)
#     $coords             - lon, lat, cell for non-NA pixels
#     $variance_explained - PC variance %
#     $rgb_vals           - RGB values [n_pixels × 3]
#     $year1, $year2      - the year pair used
#     $alpha              - mixing weight used (NULL if unweighted)
#     $signed_temporal    - whether signed differences were used
# ---------------------------------------------------------------------------
predict_spatiotemporal_rgb <- function(
    fit,
    ref_raster,
    subs_raster,
    npy_src,
    python_exe,
    pyper_script,
    year1,
    year2,
    ref_year         = year1,
    ref_month        = 6L,
    env_spatial       = NULL,
    env_temporal_yr1  = NULL,
    env_temporal_yr2  = NULL,
    modis_dir        = NULL,
    modis_resolution = "1km",
    condition_tif    = NULL,
    signed_temporal  = FALSE,
    alpha            = NULL,
    normalise_blocks = TRUE,
    pca_method       = "prcomp",
    n_components     = 3L,
    stretch          = 2,
    chunk_size       = 50000L,
    verbose          = TRUE
) {

  cat(sprintf("=== Spatio-temporal GDM Prediction -> RGB (%d -> %d) ===\n\n", year1, year2))
  t0 <- proc.time()

  ## ---- 1. Load rasters ------------------------------------------------
  if (verbose) cat("--- 1. Loading rasters ---\n")
  if (is.character(ref_raster))  ref_raster  <- raster::raster(ref_raster)
  if (is.character(subs_raster)) subs_raster <- raster::brick(subs_raster)

  cell_idx <- which(!is.na(raster::values(ref_raster)))
  xy       <- raster::xyFromCell(ref_raster, cell_idx)
  n_pixels <- length(cell_idx)
  cat(sprintf("  Non-NA pixels: %d\n", n_pixels))

  coords <- data.frame(lon = xy[, 1], lat = xy[, 2], cell = cell_idx,
                        stringsAsFactors = FALSE)

  ## ---- 2. Extract substrate -------------------------------------------
  if (verbose) cat("\n--- 2. Extracting substrate ---\n")
  subs_vals <- raster::extract(subs_raster, xy)
  colnames(subs_vals) <- paste0(colnames(subs_vals), "_1")
  cat(sprintf("  Substrate: %d layers\n", ncol(subs_vals)))

  ## ---- 3. Extract spatial climate at ref_year -------------------------
  geonpy_start <- if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L
  c_yr <- fit$climate_window

  .extract_climate_chunked <- function(xy, year, month, param_list, label) {
    n <- nrow(xy)
    n_chunks <- ceiling(n / chunk_size)
    parts <- vector("list", n_chunks)

    for (ch in seq_len(n_chunks)) {
      r1 <- (ch - 1) * chunk_size + 1
      r2 <- min(ch * chunk_size, n)
      chunk_xy <- xy[r1:r2, , drop = FALSE]
      n_ch <- nrow(chunk_xy)

      if (verbose && n_chunks > 1)
        cat(sprintf("    %s chunk %d/%d (%d px)\n", label, ch, n_chunks, n_ch))

      pairs <- data.frame(
        Lon1 = chunk_xy[, 1], Lat1 = chunk_xy[, 2],
        year1 = rep(as.integer(year), n_ch), month1 = rep(as.integer(month), n_ch),
        Lon2 = chunk_xy[, 1], Lat2 = chunk_xy[, 2],
        year2 = rep(as.integer(year), n_ch), month2 = rep(as.integer(month), n_ch)
      )

      chunk_parts <- list()
      for (j in seq_along(param_list)) {
        sp <- param_list[[j]]
        raw <- gen_windows(
          pairs = pairs, variables = sp$variables, mstat = sp$mstat,
          cstat = sp$cstat, window = sp$window, npy_src = npy_src,
          start_year = geonpy_start, python_exe = python_exe,
          pyper_script = pyper_script
        )
        env_cols <- raw[, grep("_1$", names(raw)), drop = FALSE]
        colnames(env_cols) <- paste(sp$prefix, colnames(env_cols), sep = "_")
        colnames(env_cols) <- gsub("\\d{6}-\\d{6}_", "", colnames(env_cols))
        chunk_parts[[j]] <- env_cols
      }
      parts[[ch]] <- do.call(cbind, chunk_parts)
    }
    do.call(rbind, parts)
  }

  ## Build param lists
  spatial_params <- list()
  temporal_params <- list()
  for (ep in fit$env_params) {
    spatial_params[[length(spatial_params) + 1]] <- list(
      variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
      window = c_yr, prefix = paste0("spat_", ep$cstat)
    )
    temporal_params[[length(temporal_params) + 1]] <- list(
      variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
      window = c_yr, prefix = paste0("temp_", ep$cstat)
    )
  }

  if (is.null(env_spatial)) {
    if (verbose) cat(sprintf("\n--- 3. Extracting spatial climate (ref year = %d) ---\n", ref_year))
    require(arrow)
    env_spatial <- .extract_climate_chunked(xy, ref_year, ref_month,
                                             spatial_params, "Spatial")
    cat(sprintf("  Spatial climate: %d cols × %d pixels\n", ncol(env_spatial), nrow(env_spatial)))
  } else {
    if (verbose) cat("\n--- 3. Using pre-extracted spatial climate ---\n")
  }

  ## ---- 4. Extract temporal climate at year1 and year2 -----------------
  if (is.null(env_temporal_yr1)) {
    if (verbose) cat(sprintf("\n--- 4a. Extracting temporal climate at year1 = %d ---\n", year1))
    env_temporal_yr1 <- .extract_climate_chunked(xy, year1, ref_month,
                                                   temporal_params, "Temporal-yr1")
    cat(sprintf("  Temporal yr1: %d cols × %d pixels\n", ncol(env_temporal_yr1), nrow(env_temporal_yr1)))
  }

  if (is.null(env_temporal_yr2)) {
    if (verbose) cat(sprintf("\n--- 4b. Extracting temporal climate at year2 = %d ---\n", year2))
    env_temporal_yr2 <- .extract_climate_chunked(xy, year2, ref_month,
                                                   temporal_params, "Temporal-yr2")
    cat(sprintf("  Temporal yr2: %d cols × %d pixels\n", ncol(env_temporal_yr2), nrow(env_temporal_yr2)))
  }

  ## ---- 4c. Extract MODIS land cover (spatial + temporal) ---------------
  modis_spat_vals <- NULL
  modis_temp_yr1  <- NULL
  modis_temp_yr2  <- NULL
  if (isTRUE(fit$add_modis)) {
    if (is.null(modis_dir)) {
      if (exists("config", envir = parent.frame()) &&
          !is.null(parent.frame()$config$modis_dir)) {
        modis_dir        <- parent.frame()$config$modis_dir
        modis_resolution <- parent.frame()$config$modis_resolution
      } else {
        warning("fit$add_modis is TRUE but modis_dir not provided -- MODIS predictors will be skipped")
      }
    }
    if (!is.null(modis_dir)) {
      if (verbose) cat(sprintf("\n--- 4c. Extracting MODIS (spatial @ %d, temporal %d -> %d) ---\n",
                                ref_year, year1, year2))
      ## Spatial MODIS at ref_year
      modis_spat_vals <- .extract_modis_columns(
        fit = fit, xy = xy, year = ref_year,
        modis_dir = modis_dir, modis_resolution = modis_resolution,
        col_prefix = "spat_modis", col_suffix = "_1", verbose = verbose
      )
      ## Temporal MODIS at year1 (baseline -- uses _1 suffix)
      modis_temp_yr1 <- .extract_modis_columns(
        fit = fit, xy = xy, year = year1,
        modis_dir = modis_dir, modis_resolution = modis_resolution,
        col_prefix = "temp_modis", col_suffix = "_1", verbose = verbose
      )
      ## Temporal MODIS at year2 (target -- uses _1 suffix for matching convention)
      modis_temp_yr2 <- .extract_modis_columns(
        fit = fit, xy = xy, year = year2,
        modis_dir = modis_dir, modis_resolution = modis_resolution,
        col_prefix = "temp_modis", col_suffix = "_1", verbose = verbose
      )
      cat(sprintf("  MODIS: %d spatial + %d×2 temporal columns\n",
                  ncol(modis_spat_vals), ncol(modis_temp_yr1)))
    }
  }

  ## ---- 4d. Extract condition raster (spatial + temporal) ---------------
  cond_spat_vals <- NULL
  cond_temp_yr1  <- NULL
  cond_temp_yr2  <- NULL
  if (isTRUE(fit$add_condition)) {
    if (is.null(condition_tif)) {
      if (exists("config", envir = parent.frame()) &&
          !is.null(parent.frame()$config$condition_tif_path)) {
        condition_tif <- parent.frame()$config$condition_tif_path
      } else if (!is.null(fit$condition_tif_path)) {
        condition_tif <- fit$condition_tif_path
      } else {
        warning("fit$add_condition is TRUE but condition_tif not provided -- condition predictors will be skipped")
      }
    }
    if (!is.null(condition_tif)) {
      if (verbose) cat(sprintf("\n--- 4d. Extracting condition (spatial @ %d, temporal %d -> %d) ---\n",
                                ref_year, year1, year2))
      cond_spat_vals <- .extract_condition_columns(
        fit = fit, xy = xy, year = ref_year,
        condition_tif = condition_tif,
        col_prefix = "spat_condition", col_suffix = "_1", verbose = verbose
      )
      cond_temp_yr1 <- .extract_condition_columns(
        fit = fit, xy = xy, year = year1,
        condition_tif = condition_tif,
        col_prefix = "temp_condition", col_suffix = "_1", verbose = verbose
      )
      cond_temp_yr2 <- .extract_condition_columns(
        fit = fit, xy = xy, year = year2,
        condition_tif = condition_tif,
        col_prefix = "temp_condition", col_suffix = "_1", verbose = verbose
      )
      cat(sprintf("  Condition: %d spatial + %d×2 temporal columns\n",
                  ncol(cond_spat_vals), ncol(cond_temp_yr1)))
    }
  }

  ## ---- 5. Clean NAs and sentinels across all env ----------------------
  if (verbose) cat("\n--- 5. Cleaning data ---\n")
  env_spatial_df   <- as.data.frame(env_spatial)
  env_subs_df      <- as.data.frame(subs_vals)
  env_tmpyr1_df    <- as.data.frame(env_temporal_yr1)
  env_tmpyr2_df    <- as.data.frame(env_temporal_yr2)

  all_env <- cbind(env_spatial_df, env_subs_df, env_tmpyr1_df, env_tmpyr2_df)
  if (!is.null(modis_spat_vals))  all_env <- cbind(all_env, modis_spat_vals)
  if (!is.null(modis_temp_yr1))   all_env <- cbind(all_env, modis_temp_yr1)
  if (!is.null(modis_temp_yr2))   all_env <- cbind(all_env, modis_temp_yr2)
  if (!is.null(cond_spat_vals))   all_env <- cbind(all_env, cond_spat_vals)
  if (!is.null(cond_temp_yr1))    all_env <- cbind(all_env, cond_temp_yr1)
  if (!is.null(cond_temp_yr2))    all_env <- cbind(all_env, cond_temp_yr2)
  na_rows <- is.na(rowSums(all_env))
  sentinel_rows <- apply(all_env, 1, function(r) any(r == -9999, na.rm = TRUE))
  bad_rows <- na_rows | sentinel_rows

  if (any(bad_rows)) {
    cat(sprintf("  Removing %d pixels with NA/sentinel (%d remain)\n",
                sum(bad_rows), sum(!bad_rows)))
    env_spatial_df <- env_spatial_df[!bad_rows, , drop = FALSE]
    env_subs_df    <- env_subs_df[!bad_rows, , drop = FALSE]
    env_tmpyr1_df  <- env_tmpyr1_df[!bad_rows, , drop = FALSE]
    env_tmpyr2_df  <- env_tmpyr2_df[!bad_rows, , drop = FALSE]
    if (!is.null(modis_spat_vals)) modis_spat_vals <- modis_spat_vals[!bad_rows, , drop = FALSE]
    if (!is.null(modis_temp_yr1))  modis_temp_yr1  <- modis_temp_yr1[!bad_rows, , drop = FALSE]
    if (!is.null(modis_temp_yr2))  modis_temp_yr2  <- modis_temp_yr2[!bad_rows, , drop = FALSE]
    if (!is.null(cond_spat_vals))  cond_spat_vals  <- cond_spat_vals[!bad_rows, , drop = FALSE]
    if (!is.null(cond_temp_yr1))   cond_temp_yr1   <- cond_temp_yr1[!bad_rows, , drop = FALSE]
    if (!is.null(cond_temp_yr2))   cond_temp_yr2   <- cond_temp_yr2[!bad_rows, , drop = FALSE]
    coords         <- coords[!bad_rows, , drop = FALSE]
    n_pixels       <- nrow(coords)
  }

  ## ---- 6. I-spline transforms -----------------------------------------
  if (verbose) cat("\n--- 6a. Spatial+substrate I-spline transform ---\n")
  env_spat_combined <- cbind(env_spatial_df, env_subs_df)
  if (!is.null(modis_spat_vals)) env_spat_combined <- cbind(env_spat_combined, modis_spat_vals)
  if (!is.null(cond_spat_vals))  env_spat_combined <- cbind(env_spat_combined, cond_spat_vals)
  trans_spatial <- transform_spatial_gdm(
    fit = fit, env_df = env_spat_combined,
    weight_by_coef = TRUE, spatial_only = TRUE, verbose = verbose
  )

  if (verbose) cat(sprintf("\n--- 6b. Temporal change I-spline transform (%s) ---\n",
                            if (signed_temporal) "signed" else "absolute"))
  ## Add MODIS temporal columns to the temporal env data frames
  if (!is.null(modis_temp_yr1)) env_tmpyr1_df <- cbind(env_tmpyr1_df, modis_temp_yr1)
  if (!is.null(modis_temp_yr2)) env_tmpyr2_df <- cbind(env_tmpyr2_df, modis_temp_yr2)
  ## Add condition temporal columns to the temporal env data frames
  if (!is.null(cond_temp_yr1)) env_tmpyr1_df <- cbind(env_tmpyr1_df, cond_temp_yr1)
  if (!is.null(cond_temp_yr2)) env_tmpyr2_df <- cbind(env_tmpyr2_df, cond_temp_yr2)
  trans_temporal <- transform_temporal_change_gdm(
    fit = fit, env_yr1 = env_tmpyr1_df, env_yr2 = env_tmpyr2_df,
    weight_by_coef = TRUE, signed = signed_temporal, verbose = verbose
  )

  ## ---- 7. Combine and PCA ---------------------------------------------
  ## Concatenate: spatial columns from trans_spatial + temporal columns from trans_temporal
  trans_full <- trans_spatial + trans_temporal   # same shape, non-overlapping non-zero cols

  ## Drop zero-variance columns
  col_var <- apply(trans_full, 2, var, na.rm = TRUE)
  active_cols <- which(col_var > 1e-15)
  trans_active <- trans_full[, active_cols, drop = FALSE]

  ## Identify which active columns are spatial vs temporal
  active_names <- colnames(trans_active)
  is_temporal_col <- grepl("^temp_", active_names)
  n_spat_active <- sum(!is_temporal_col)
  n_temp_active <- sum(is_temporal_col)

  ## Alpha-weighted normalisation (Option C: controlled mixing)
  if (!is.null(alpha)) {
    if (alpha < 0 || alpha > 1) stop("alpha must be in [0, 1]")
    spat_cols <- which(!is_temporal_col)
    temp_cols <- which(is_temporal_col)

    if (normalise_blocks && length(spat_cols) > 0 && length(temp_cols) > 0) {
      spat_sd <- apply(trans_active[, spat_cols, drop = FALSE], 2, sd, na.rm = TRUE)
      temp_sd <- apply(trans_active[, temp_cols, drop = FALSE], 2, sd, na.rm = TRUE)
      spat_sd[spat_sd == 0] <- 1
      temp_sd[temp_sd == 0] <- 1
      trans_active[, spat_cols] <- sweep(trans_active[, spat_cols, drop = FALSE], 2, spat_sd, "/")
      trans_active[, temp_cols] <- sweep(trans_active[, temp_cols, drop = FALSE], 2, temp_sd, "/")
    }

    trans_active[, spat_cols] <- sqrt(1 - alpha) * trans_active[, spat_cols, drop = FALSE]
    trans_active[, temp_cols] <- sqrt(alpha)     * trans_active[, temp_cols, drop = FALSE]

    if (verbose) cat(sprintf("  Alpha mixing: spatial weight = %.2f, temporal weight = %.2f\n",
                              sqrt(1 - alpha), sqrt(alpha)))
  }

  if (verbose) {
    cat(sprintf("\n--- 7. PCA (%s) ---\n", pca_method))
    cat(sprintf("  Active columns: %d (%d spatial/substrate + %d temporal)\n",
                length(active_cols), n_spat_active, n_temp_active))
  }

  pca_obj <- NULL
  var_explained <- NULL

  if (pca_method == "none") {
    scores <- trans_active[, seq_len(min(n_components, ncol(trans_active))), drop = FALSE]
  } else if (pca_method == "robust") {
    if (!requireNamespace("MASS", quietly = TRUE))
      stop("Package 'MASS' required for robust PCA")
    centre <- colMeans(trans_active, na.rm = TRUE)
    scaled <- scale(trans_active, center = centre, scale = FALSE)
    cov_rob <- MASS::cov.mcd(scaled, quantile.used = floor(0.75 * nrow(scaled)))
    eig <- eigen(cov_rob$cov, symmetric = TRUE)
    loadings <- eig$vectors[, seq_len(n_components), drop = FALSE]
    scores <- scaled %*% loadings
    var_explained <- 100 * eig$values[seq_len(n_components)] / sum(eig$values)
  } else {
    pca_obj <- prcomp(trans_active, center = TRUE, scale. = FALSE,
                       rank. = n_components)
    scores <- pca_obj$x[, seq_len(n_components), drop = FALSE]
    var_explained <- 100 * pca_obj$sdev[seq_len(n_components)]^2 / sum(pca_obj$sdev^2)
  }

  if (!is.null(var_explained) && verbose)
    cat(sprintf("  Variance: PC1=%.1f%%, PC2=%.1f%%, PC3=%.1f%% (total=%.1f%%)\n",
                var_explained[1], var_explained[2], var_explained[3], sum(var_explained)))

  ## ---- 8. RGB mapping -------------------------------------------------
  if (verbose) cat(sprintf("\n--- 8. RGB mapping (stretch = %g%%) ---\n", stretch))
  rgb_vals <- matrix(NA_real_, nrow = n_pixels, ncol = n_components)
  for (k in seq_len(n_components)) {
    v <- scores[, k]
    lo <- quantile(v, stretch / 100, na.rm = TRUE)
    hi <- quantile(v, 1 - stretch / 100, na.rm = TRUE)
    if (hi <= lo) { lo <- min(v, na.rm = TRUE); hi <- max(v, na.rm = TRUE) }
    v_scaled <- pmin(pmax((v - lo) / (hi - lo), 0), 1)
    rgb_vals[, k] <- round(v_scaled * 255)
  }

  ## ---- 9. Build rasters -----------------------------------------------
  if (verbose) cat("\n--- 9. Building RGB rasters ---\n")
  template <- raster::raster(ref_raster)

  r_layer <- template; raster::values(r_layer) <- NA
  g_layer <- template; raster::values(g_layer) <- NA
  b_layer <- template; raster::values(b_layer) <- NA

  r_layer[coords$cell] <- rgb_vals[, 1]
  g_layer[coords$cell] <- rgb_vals[, 2]
  b_layer[coords$cell] <- rgb_vals[, 3]

  rgb_stack <- raster::stack(r_layer, g_layer, b_layer)
  names(rgb_stack) <- c("PC1_red", "PC2_green", "PC3_blue")

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("\n=== Spatio-temporal prediction complete (%.1f s) ===\n", elapsed))

  list(
    rgb_stack            = rgb_stack,
    pca                  = pca_obj,
    pca_scores           = scores,
    transformed_spatial  = trans_spatial,
    transformed_temporal = trans_temporal,
    transformed_full     = trans_active,
    coords               = coords,
    env_spatial          = env_spatial_df,
    env_temporal_yr1     = env_tmpyr1_df,
    env_temporal_yr2     = env_tmpyr2_df,
    variance_explained   = var_explained,
    rgb_vals             = rgb_vals,
    active_cols          = active_cols,
    n_spat_active        = n_spat_active,
    n_temp_active        = n_temp_active,
    year1                = year1,
    year2                = year2,
    ref_year             = ref_year,
    alpha                = alpha,
    signed_temporal      = signed_temporal
  )
}


# ---------------------------------------------------------------------------
# .hsl_to_rgb_vec  (internal helper)
#
# Convert H, S, L vectors to R, G, B vectors (all in [0, 1]).
# H is in [0, 1] (fraction of 360°), S in [0, 1], L in [0, 1].
# ---------------------------------------------------------------------------
.hsl_to_rgb_vec <- function(h, s, l) {
  n <- length(h)
  r <- g <- b <- numeric(n)

  for (i in seq_len(n)) {
    if (is.na(h[i]) || is.na(s[i]) || is.na(l[i])) {
      r[i] <- g[i] <- b[i] <- NA
      next
    }
    if (s[i] == 0) {
      r[i] <- g[i] <- b[i] <- l[i]
      next
    }

    q_val <- if (l[i] < 0.5) l[i] * (1 + s[i]) else l[i] + s[i] - l[i] * s[i]
    p_val <- 2 * l[i] - q_val

    .hue2rgb <- function(p, q, t) {
      if (t < 0) t <- t + 1
      if (t > 1) t <- t - 1
      if (t < 1/6) return(p + (q - p) * 6 * t)
      if (t < 1/2) return(q)
      if (t < 2/3) return(p + (q - p) * (2/3 - t) * 6)
      return(p)
    }

    r[i] <- .hue2rgb(p_val, q_val, h[i] + 1/3)
    g[i] <- .hue2rgb(p_val, q_val, h[i])
    b[i] <- .hue2rgb(p_val, q_val, h[i] - 1/3)
  }

  cbind(r = r, g = g, b = b)
}


# ---------------------------------------------------------------------------
# predict_spatiotemporal_hsl
#
# Two-stage spatio-temporal map using a bivariate colour scheme:
#
#   - Hue       ← spatial community type (from atan2 of spatial PC1, PC2)
#   - Saturation ← spatial distinctiveness (distance from PCA centroid)
#   - Lightness  ← temporal change magnitude (sum of |temporal spline diffs|)
#
# This gives a single map where COLOUR tells you what community type a pixel
# belongs to, and BRIGHTNESS tells you how much biological change has occurred
# between year1 and year2. Dark pixels = large temporal change.
#
# The function reuses the same extraction and transform machinery as
# predict_spatiotemporal_rgb(). If pre-computed transforms are provided via
# `trans_spatial` and `trans_temporal`, the expensive climate extraction is
# skipped entirely.
#
# Parameters:
#   fit            - fitted GDM list
#   ref_raster     - reference raster (path or RasterLayer)
#   subs_raster    - substrate raster (path or RasterBrick)
#   npy_src, python_exe, pyper_script - for geonpy extraction
#   year1, year2   - temporal pair
#   ref_year       - spatial climate reference year (default: year1)
#   ref_month      - reference month (default: 6)
#   env_spatial, env_temporal_yr1, env_temporal_yr2 - pre-extracted env (or NULL)
#   trans_spatial   - pre-computed spatial+substrate I-spline transform (or NULL)
#   trans_temporal  - pre-computed temporal change I-spline transform (or NULL)
#   coords         - data.frame with lon, lat, cell (required if transforms
#                    are pre-computed)
#   signed_temporal - use signed temporal diffs (default: FALSE)
#   sat_fixed      - if not NULL, use this fixed saturation [0,1] (default: 0.85)
#                    Set NULL to derive saturation from spatial distance to centroid.
#   light_range    - c(min, max) lightness range (default: c(0.25, 0.90)).
#                    Low L = dark = more change.
#   light_invert   - if TRUE (default), high change -> dark (low L).
#                    Set FALSE for high change -> bright.
#   stretch        - percentile stretch for lightness (default: 2)
#   chunk_size     - pixels per chunk (default: 50000)
#   verbose        - print progress
#
# Returns:
#   List with:
#     $rgb_stack      - 3-band RasterBrick [0, 255] (from HSL conversion)
#     $hue, $sat, $light - raw HSL values [0, 1] per pixel
#     $spatial_pca    - prcomp object from spatial-only PCA
#     $temporal_magnitude - temporal change magnitude per pixel
#     $transformed_spatial, $transformed_temporal
#     $coords, $year1, $year2
# ---------------------------------------------------------------------------
predict_spatiotemporal_hsl <- function(
    fit,
    ref_raster       = NULL,
    subs_raster      = NULL,
    npy_src          = NULL,
    python_exe       = NULL,
    pyper_script     = NULL,
    year1            = NULL,
    year2            = NULL,
    ref_year         = year1,
    ref_month        = 6L,
    env_spatial      = NULL,
    env_temporal_yr1 = NULL,
    env_temporal_yr2 = NULL,
    modis_dir        = NULL,
    modis_resolution = "1km",
    condition_tif    = NULL,
    trans_spatial     = NULL,
    trans_temporal    = NULL,
    coords           = NULL,
    signed_temporal  = FALSE,
    sat_fixed        = 0.85,
    light_range      = c(0.25, 0.90),
    light_invert     = TRUE,
    stretch          = 2,
    chunk_size       = 50000L,
    verbose          = TRUE
) {

  cat(sprintf("=== Spatio-temporal HSL Map (%s -> %s) ===\n\n",
              if (!is.null(year1)) year1 else "?",
              if (!is.null(year2)) year2 else "?"))
  t0 <- proc.time()

  ## ==================================================================
  ## 1.  Obtain transforms  (reuse if provided)
  ## ==================================================================
  if (is.null(trans_spatial) || is.null(trans_temporal)) {

    if (is.null(year1) || is.null(year2))
      stop("year1 and year2 required when transforms are not pre-computed")
    if (is.null(ref_raster) || is.null(subs_raster))
      stop("ref_raster and subs_raster required when transforms are not pre-computed")

    if (verbose) cat("--- 1. Computing transforms from scratch ---\n")
    if (is.character(ref_raster))  ref_raster  <- raster::raster(ref_raster)
    if (is.character(subs_raster)) subs_raster <- raster::brick(subs_raster)

    cell_idx <- which(!is.na(raster::values(ref_raster)))
    xy       <- raster::xyFromCell(ref_raster, cell_idx)
    n_pixels <- length(cell_idx)
    coords   <- data.frame(lon = xy[, 1], lat = xy[, 2], cell = cell_idx,
                            stringsAsFactors = FALSE)

    subs_vals <- raster::extract(subs_raster, xy)
    colnames(subs_vals) <- paste0(colnames(subs_vals), "_1")

    geonpy_start <- if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L
    c_yr <- fit$climate_window

    .extract_chunked <- function(xy, year, month, param_list, label) {
      n <- nrow(xy)
      n_ch <- ceiling(n / chunk_size)
      parts <- vector("list", n_ch)
      for (ch in seq_len(n_ch)) {
        r1 <- (ch - 1) * chunk_size + 1
        r2 <- min(ch * chunk_size, n)
        cxy <- xy[r1:r2, , drop = FALSE]; nc <- nrow(cxy)
        if (verbose && n_ch > 1) cat(sprintf("    %s chunk %d/%d\n", label, ch, n_ch))
        pairs <- data.frame(
          Lon1 = cxy[, 1], Lat1 = cxy[, 2],
          year1 = as.integer(year), month1 = as.integer(month),
          Lon2 = cxy[, 1], Lat2 = cxy[, 2],
          year2 = as.integer(year), month2 = as.integer(month))
        cp <- list()
        for (j in seq_along(param_list)) {
          sp <- param_list[[j]]
          raw <- gen_windows(pairs = pairs, variables = sp$variables,
                             mstat = sp$mstat, cstat = sp$cstat,
                             window = sp$window, npy_src = npy_src,
                             start_year = geonpy_start,
                             python_exe = python_exe,
                             pyper_script = pyper_script)
          ec <- raw[, grep("_1$", names(raw)), drop = FALSE]
          colnames(ec) <- paste(sp$prefix, colnames(ec), sep = "_")
          colnames(ec) <- gsub("\\d{6}-\\d{6}_", "", colnames(ec))
          cp[[j]] <- ec
        }
        parts[[ch]] <- do.call(cbind, cp)
      }
      do.call(rbind, parts)
    }

    spatial_params <- temporal_params <- list()
    for (ep in fit$env_params) {
      spatial_params[[length(spatial_params) + 1]] <- list(
        variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
        window = c_yr, prefix = paste0("spat_", ep$cstat))
      temporal_params[[length(temporal_params) + 1]] <- list(
        variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
        window = c_yr, prefix = paste0("temp_", ep$cstat))
    }

    require(arrow)
    if (is.null(env_spatial))
      env_spatial <- .extract_chunked(xy, ref_year, ref_month, spatial_params, "Spatial")
    if (is.null(env_temporal_yr1))
      env_temporal_yr1 <- .extract_chunked(xy, year1, ref_month, temporal_params, "Temp-yr1")
    if (is.null(env_temporal_yr2))
      env_temporal_yr2 <- .extract_chunked(xy, year2, ref_month, temporal_params, "Temp-yr2")

    ## MODIS extraction (spatial + temporal, if model includes MODIS)
    modis_spat_vals <- NULL
    modis_temp_yr1  <- NULL
    modis_temp_yr2  <- NULL
    if (isTRUE(fit$add_modis)) {
      if (is.null(modis_dir)) {
        if (exists("config", envir = parent.frame()) &&
            !is.null(parent.frame()$config$modis_dir)) {
          modis_dir        <- parent.frame()$config$modis_dir
          modis_resolution <- parent.frame()$config$modis_resolution
        } else {
          warning("fit$add_modis is TRUE but modis_dir not provided -- MODIS predictors will be skipped")
        }
      }
      if (!is.null(modis_dir)) {
        if (verbose) cat("  Extracting MODIS (spatial + temporal)...\n")
        modis_spat_vals <- .extract_modis_columns(
          fit = fit, xy = xy, year = ref_year,
          modis_dir = modis_dir, modis_resolution = modis_resolution,
          col_prefix = "spat_modis", col_suffix = "_1", verbose = verbose
        )
        modis_temp_yr1 <- .extract_modis_columns(
          fit = fit, xy = xy, year = year1,
          modis_dir = modis_dir, modis_resolution = modis_resolution,
          col_prefix = "temp_modis", col_suffix = "_1", verbose = verbose
        )
        modis_temp_yr2 <- .extract_modis_columns(
          fit = fit, xy = xy, year = year2,
          modis_dir = modis_dir, modis_resolution = modis_resolution,
          col_prefix = "temp_modis", col_suffix = "_1", verbose = verbose
        )
      }
    }

    ## Condition extraction (spatial + temporal, if model includes condition)
    cond_spat_vals <- NULL
    cond_temp_yr1  <- NULL
    cond_temp_yr2  <- NULL
    if (isTRUE(fit$add_condition)) {
      if (is.null(condition_tif)) {
        if (exists("config", envir = parent.frame()) &&
            !is.null(parent.frame()$config$condition_tif_path)) {
          condition_tif <- parent.frame()$config$condition_tif_path
        } else if (!is.null(fit$condition_tif_path)) {
          condition_tif <- fit$condition_tif_path
        } else {
          warning("fit$add_condition is TRUE but condition_tif not provided -- condition predictors will be skipped")
        }
      }
      if (!is.null(condition_tif)) {
        if (verbose) cat("  Extracting condition (spatial + temporal)...\n")
        cond_spat_vals <- .extract_condition_columns(
          fit = fit, xy = xy, year = ref_year,
          condition_tif = condition_tif,
          col_prefix = "spat_condition", col_suffix = "_1", verbose = verbose
        )
        cond_temp_yr1 <- .extract_condition_columns(
          fit = fit, xy = xy, year = year1,
          condition_tif = condition_tif,
          col_prefix = "temp_condition", col_suffix = "_1", verbose = verbose
        )
        cond_temp_yr2 <- .extract_condition_columns(
          fit = fit, xy = xy, year = year2,
          condition_tif = condition_tif,
          col_prefix = "temp_condition", col_suffix = "_1", verbose = verbose
        )
      }
    }

    ## Clean
    env_all <- cbind(as.data.frame(env_spatial), as.data.frame(subs_vals),
                     as.data.frame(env_temporal_yr1), as.data.frame(env_temporal_yr2))
    if (!is.null(modis_spat_vals)) env_all <- cbind(env_all, as.data.frame(modis_spat_vals))
    if (!is.null(modis_temp_yr1))  env_all <- cbind(env_all, as.data.frame(modis_temp_yr1))
    if (!is.null(modis_temp_yr2))  env_all <- cbind(env_all, as.data.frame(modis_temp_yr2))
    if (!is.null(cond_spat_vals))  env_all <- cbind(env_all, as.data.frame(cond_spat_vals))
    if (!is.null(cond_temp_yr1))   env_all <- cbind(env_all, as.data.frame(cond_temp_yr1))
    if (!is.null(cond_temp_yr2))   env_all <- cbind(env_all, as.data.frame(cond_temp_yr2))
    bad <- is.na(rowSums(env_all)) |
      apply(env_all, 1, function(r) any(r == -9999, na.rm = TRUE))
    if (any(bad)) {
      env_spatial      <- as.data.frame(env_spatial)[!bad, , drop = FALSE]
      subs_vals        <- as.data.frame(subs_vals)[!bad, , drop = FALSE]
      env_temporal_yr1 <- as.data.frame(env_temporal_yr1)[!bad, , drop = FALSE]
      env_temporal_yr2 <- as.data.frame(env_temporal_yr2)[!bad, , drop = FALSE]
      if (!is.null(modis_spat_vals)) modis_spat_vals <- as.data.frame(modis_spat_vals)[!bad, , drop = FALSE]
      if (!is.null(modis_temp_yr1))  modis_temp_yr1  <- as.data.frame(modis_temp_yr1)[!bad, , drop = FALSE]
      if (!is.null(modis_temp_yr2))  modis_temp_yr2  <- as.data.frame(modis_temp_yr2)[!bad, , drop = FALSE]
      if (!is.null(cond_spat_vals))  cond_spat_vals  <- as.data.frame(cond_spat_vals)[!bad, , drop = FALSE]
      if (!is.null(cond_temp_yr1))   cond_temp_yr1   <- as.data.frame(cond_temp_yr1)[!bad, , drop = FALSE]
      if (!is.null(cond_temp_yr2))   cond_temp_yr2   <- as.data.frame(cond_temp_yr2)[!bad, , drop = FALSE]
      coords           <- coords[!bad, , drop = FALSE]
    }

    env_spat_subs <- cbind(as.data.frame(env_spatial), as.data.frame(subs_vals))
    if (!is.null(modis_spat_vals)) env_spat_subs <- cbind(env_spat_subs, as.data.frame(modis_spat_vals))
    if (!is.null(cond_spat_vals))  env_spat_subs <- cbind(env_spat_subs, as.data.frame(cond_spat_vals))
    trans_spatial <- transform_spatial_gdm(
      fit = fit, env_df = env_spat_subs,
      weight_by_coef = TRUE, spatial_only = TRUE, verbose = verbose)

    ## Add MODIS temporal columns before temporal transform
    env_yr1_df <- as.data.frame(env_temporal_yr1)
    env_yr2_df <- as.data.frame(env_temporal_yr2)
    if (!is.null(modis_temp_yr1)) env_yr1_df <- cbind(env_yr1_df, modis_temp_yr1)
    if (!is.null(modis_temp_yr2)) env_yr2_df <- cbind(env_yr2_df, modis_temp_yr2)
    ## Add condition temporal columns before temporal transform
    if (!is.null(cond_temp_yr1)) env_yr1_df <- cbind(env_yr1_df, cond_temp_yr1)
    if (!is.null(cond_temp_yr2)) env_yr2_df <- cbind(env_yr2_df, cond_temp_yr2)
    trans_temporal <- transform_temporal_change_gdm(
      fit = fit, env_yr1 = env_yr1_df, env_yr2 = env_yr2_df,
      weight_by_coef = TRUE, signed = signed_temporal, verbose = verbose)

  } else {
    if (is.null(coords)) stop("coords required when transforms are pre-computed")
    if (verbose) cat("--- 1. Using pre-computed transforms ---\n")
  }

  n_pixels <- nrow(trans_spatial)

  ## ==================================================================
  ## 2.  Spatial PCA -> Hue + Saturation
  ## ==================================================================
  if (verbose) cat("\n--- 2. Spatial PCA for hue ---\n")

  col_var_s <- apply(trans_spatial, 2, var, na.rm = TRUE)
  active_s  <- which(col_var_s > 1e-15)
  trans_s   <- trans_spatial[, active_s, drop = FALSE]

  spat_pca <- prcomp(trans_s, center = TRUE, scale. = FALSE, rank. = 2)
  pc1 <- spat_pca$x[, 1]
  pc2 <- spat_pca$x[, 2]

  ## Hue = angle in PC1–PC2 space, mapped to [0, 1]
  hue <- (atan2(pc2, pc1) + pi) / (2 * pi)

  ## Saturation = distance from centroid in PC1–PC2 (biological distinctiveness)
  if (is.null(sat_fixed)) {
    dist_from_centre <- sqrt(pc1^2 + pc2^2)
    sat_lo <- quantile(dist_from_centre, 0.02, na.rm = TRUE)
    sat_hi <- quantile(dist_from_centre, 0.98, na.rm = TRUE)
    if (sat_hi <= sat_lo) { sat_lo <- min(dist_from_centre, na.rm = TRUE)
                             sat_hi <- max(dist_from_centre, na.rm = TRUE) }
    sat <- pmin(pmax((dist_from_centre - sat_lo) / (sat_hi - sat_lo), 0.1), 1.0)
  } else {
    sat <- rep(sat_fixed, n_pixels)
    dist_from_centre <- sqrt(pc1^2 + pc2^2)
  }

  ve_spat <- 100 * spat_pca$sdev[1:2]^2 / sum(spat_pca$sdev^2)
  if (verbose) cat(sprintf("  Spatial PC1=%.1f%%, PC2=%.1f%%\n", ve_spat[1], ve_spat[2]))

  ## ==================================================================
  ## 3.  Temporal magnitude -> Lightness
  ## ==================================================================
  if (verbose) cat("\n--- 3. Temporal magnitude for lightness ---\n")

  col_var_t  <- apply(trans_temporal, 2, var, na.rm = TRUE)
  active_t   <- which(col_var_t > 1e-15)
  trans_t    <- trans_temporal[, active_t, drop = FALSE]

  ## Temporal magnitude = Euclidean norm of temporal change vector
  temp_magnitude <- sqrt(rowSums(trans_t^2))

  ## Map to lightness range with percentile stretch
  lo_t <- quantile(temp_magnitude, stretch / 100, na.rm = TRUE)
  hi_t <- quantile(temp_magnitude, 1 - stretch / 100, na.rm = TRUE)
  if (hi_t <= lo_t) { lo_t <- min(temp_magnitude, na.rm = TRUE)
                       hi_t <- max(temp_magnitude, na.rm = TRUE) }

  scaled_mag <- pmin(pmax((temp_magnitude - lo_t) / (hi_t - lo_t), 0), 1)

  ## Map to lightness: high change -> low L (dark) or high L depending on invert
  if (light_invert) {
    light <- light_range[2] - scaled_mag * (light_range[2] - light_range[1])
  } else {
    light <- light_range[1] + scaled_mag * (light_range[2] - light_range[1])
  }

  if (verbose) {
    cat(sprintf("  Temporal magnitude -- min: %.4f, median: %.4f, max: %.4f\n",
                min(temp_magnitude, na.rm = TRUE),
                median(temp_magnitude, na.rm = TRUE),
                max(temp_magnitude, na.rm = TRUE)))
    cat(sprintf("  Lightness range: [%.2f, %.2f] (%s = more change)\n",
                light_range[1], light_range[2],
                if (light_invert) "darker" else "brighter"))
  }

  ## ==================================================================
  ## 4.  HSL -> RGB
  ## ==================================================================
  if (verbose) cat("\n--- 4. HSL -> RGB conversion ---\n")
  rgb_raw <- .hsl_to_rgb_vec(hue, sat, light)
  rgb_vals <- round(rgb_raw * 255)

  ## ==================================================================
  ## 5.  Build rasters
  ## ==================================================================
  if (verbose) cat("\n--- 5. Building RGB rasters ---\n")
  if (is.null(ref_raster)) {
    if (!is.null(fit$ref_raster_path))
      ref_raster <- raster::raster(fit$ref_raster_path)
    else
      stop("ref_raster required for raster output")
  }
  if (is.character(ref_raster)) ref_raster <- raster::raster(ref_raster)
  template <- raster::raster(ref_raster)

  mk_layer <- function(vals) {
    ly <- template; raster::values(ly) <- NA
    ly[coords$cell] <- vals; ly
  }

  r_layer <- mk_layer(rgb_vals[, 1])
  g_layer <- mk_layer(rgb_vals[, 2])
  b_layer <- mk_layer(rgb_vals[, 3])

  rgb_stack <- raster::stack(r_layer, g_layer, b_layer)
  names(rgb_stack) <- c("HSL_red", "HSL_green", "HSL_blue")

  ## Also build individual channel rasters for diagnostics
  hue_layer   <- mk_layer(hue)
  sat_layer   <- mk_layer(sat)
  light_layer <- mk_layer(light)
  mag_layer   <- mk_layer(temp_magnitude)

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("\n=== HSL spatio-temporal map complete (%.1f s) ===\n", elapsed))

  list(
    rgb_stack            = rgb_stack,
    hue                  = hue,
    sat                  = sat,
    light                = light,
    hue_raster           = hue_layer,
    sat_raster           = sat_layer,
    light_raster         = light_layer,
    magnitude_raster     = mag_layer,
    spatial_pca          = spat_pca,
    spatial_ve           = ve_spat,
    dist_from_centre     = dist_from_centre,
    temporal_magnitude   = temp_magnitude,
    transformed_spatial  = trans_spatial,
    transformed_temporal = trans_temporal,
    coords               = coords,
    rgb_vals             = rgb_vals,
    year1                = year1,
    year2                = year2,
    ref_year             = ref_year,
    signed_temporal      = signed_temporal
  )
}


# ---------------------------------------------------------------------------
#
# Given a fitted GDM and a pair of environmental data.frames, compute
# the spatial ecological distance and dissimilarity between each pair
# of corresponding rows (pixel i from env1 vs pixel i from env2).
#
# Uses only spatial + substrate predictors.
#
# Parameters:
#   fit    - fitted GDM list
#   env1   - data.frame: environmental values at site set 1
#   env2   - data.frame: environmental values at site set 2
#            (same rows, same columns as env1)
#
# Returns:
#   data.frame with:
#     spatial_distance, linear_predictor, predicted_prob, dissimilarity
# ---------------------------------------------------------------------------
spatial_dissimilarity <- function(fit, env1, env2) {

  n_pts         <- nrow(env1)
  n_preds       <- length(fit$predictors)
  total_splines <- sum(fit$splines)
  csp           <- c(0, cumsum(fit$splines))

  temp_idx <- grep("^temp_", fit$predictors)
  spat_idx <- grep("^spat_", fit$predictors)
  subs_idx <- setdiff(seq_len(n_preds), c(temp_idx, spat_idx))
  active_idx <- c(spat_idx, subs_idx)

  ## Build pairwise spline distance (|spline(site1) - spline(site2)|)
  spline_diff <- matrix(0, nrow = n_pts, ncol = total_splines)

  for (i in active_idx) {
    pred_name <- fit$predictors[i]
    ns        <- fit$splines[i]
    coefs_i   <- fit$coefficients[(csp[i] + 1):(csp[i] + ns)]
    quants_i  <- fit$quantiles[(csp[i] + 1):(csp[i] + ns)]

    ## Match columns
    col1 <- which(names(env1) == pred_name)
    if (length(col1) == 0) col1 <- which(names(env1) == sub("_1$", "", pred_name))
    col2 <- which(names(env2) == pred_name)
    if (length(col2) == 0) col2 <- which(names(env2) == sub("_1$", "", pred_name))

    if (length(col1) == 0 || length(col2) == 0) next

    v1 <- env1[, col1[1]]
    v2 <- env2[, col2[1]]

    for (sp in seq_len(ns)) {
      if (sp == 1L)       { q1 <- quants_i[1]; q2 <- quants_i[1]; q3 <- quants_i[min(2, ns)] }
      else if (sp == ns)  { q1 <- quants_i[max(1, ns - 1)]; q2 <- quants_i[ns]; q3 <- quants_i[ns] }
      else                { q1 <- quants_i[sp - 1]; q2 <- quants_i[sp]; q3 <- quants_i[sp + 1] }

      spline_diff[, csp[i] + sp] <- abs(I_spline(v1, q1, q2, q3) - I_spline(v2, q1, q2, q3))
    }
  }

  ## Ecological distance
  spatial_distance <- as.numeric(spline_diff %*% fit$coefficients)
  eta <- fit$intercept + spatial_distance
  predicted_prob <- inv.logit(eta)
  p0 <- inv.logit(fit$intercept)
  dissim <- ObsTrans(p0, fit$w_ratio, predicted_prob)

  data.frame(
    spatial_distance = spatial_distance,
    linear_predictor = eta,
    predicted_prob   = predicted_prob,
    dissimilarity    = dissim$out,
    stringsAsFactors = FALSE
  )
}


# ---------------------------------------------------------------------------
# predict_spatial_lmds
#
# Landmark MDS ordination using true GDM-calibrated dissimilarities.
#
# The transform->PCA approach (predict_spatial_rgb) maps each pixel into
# biological space independently and then ordinates via PCA. This preserves
# relative positions but does NOT pass through the nonlinear calibration
# (intercept -> logit -> ObsTrans).
#
# This function instead:
#   1. Selects k landmark pixels (random, stratified, or custom)
#   2. Computes the full calibrated dissimilarity (ecological distance ->
#      logit -> ObsTrans) between all landmark pairs (k×k) and from every
#      pixel to each landmark (n×k)
#   3. Runs classical MDS (eigendecomposition of the double-centred
#      squared-distance matrix) on the k×k landmark matrix
#   4. Projects all n pixels into MDS space via Nyström extension
#   5. Maps MDS dimensions -> RGB
#
# Complexity: O(k² · p) for D_LL + O(n · k · p) for D_NL + O(k³) for MDS
# With k = 500, n = 275k, p ≈ 51: the n·k·p term ≈ 7 billion operations
# but is vectorised (≈ 30 s in R).
#
# The function accepts a pre-computed β-weighted I-spline transform
# (from a prior predict_spatial_rgb() call) to avoid repeating the
# expensive climate extraction step.
#
# Parameters:
#   fit          - fitted GDM list
#   ref_raster   - reference raster (defines grid; path or RasterLayer)
#   subs_raster  - substrate raster brick (path or RasterBrick)
#   env_spatial, npy_src, python_exe, pyper_script, ref_year, ref_month,
#   chunk_size   - same as predict_spatial_rgb()
#   transformed  - pre-computed β-weighted I-spline transform matrix from
#                  transform_spatial_gdm(weight_by_coef=TRUE). If NULL,
#                  the function will compute it from scratch.
#   coords       - data.frame with lon, lat, cell matching rows of
#                  `transformed` (required when transformed is provided)
#   n_landmarks  - number of landmark pixels (default: 500)
#   landmark_method - "random" (default), "stratified" (spatial k-means),
#                     or "custom" (user-supplied indices)
#   landmark_idx - integer vector of row indices (when method = "custom")
#   n_components - MDS dimensions (default: 3 for RGB)
#   stretch      - percentile stretch for RGB (default: 2)
#   use_dissimilarity - if TRUE (default), apply the full nonlinear
#                  calibration (logit -> ObsTrans) before MDS. If FALSE,
#                  use raw ecological distance (Manhattan in transformed
#                  space). The raw distance is already Manhattan and may
#                  embed better with fewer negative eigenvalues.
#   seed         - random seed for landmark selection (default: 42)
#   verbose      - print progress
#
# Returns:
#   List with:
#     $rgb_stack          - 3-band RasterBrick [0, 255]
#     $mds_scores         - matrix [n_pixels × n_components]
#     $landmark_idx       - row indices of selected landmarks
#     $landmark_coords    - lon/lat/cell of landmarks
#     $landmark_mds       - MDS coordinates of landmarks [k × m]
#     $D_LL               - k × k distance matrix (dissimilarity or eco dist)
#     $eigenvalues        - all eigenvalues from MDS of D_LL
#     $variance_explained - % variance per MDS dimension (from eigenvalues)
#     $coords, $rgb_vals  - as above
# ---------------------------------------------------------------------------
predict_spatial_lmds <- function(
    fit,
    ref_raster      = NULL,
    subs_raster     = NULL,
    env_spatial     = NULL,
    npy_src         = NULL,
    python_exe      = NULL,
    pyper_script    = NULL,
    ref_year        = 1975L,
    ref_month       = 6L,
    modis_dir       = NULL,
    modis_resolution = "1km",
    condition_tif   = NULL,
    transformed     = NULL,
    coords          = NULL,
    n_landmarks     = 500L,
    landmark_method = "random",
    landmark_idx    = NULL,
    n_components    = 3L,
    stretch         = 2,
    use_dissimilarity = TRUE,
    chunk_size      = 50000L,
    seed            = 42L,
    verbose         = TRUE
) {

  cat("=== Spatial GDM Prediction -> Landmark MDS -> RGB ===\n\n")
  t0 <- proc.time()

  ## ==================================================================
  ## 1.  Obtain β-weighted I-spline transform
  ## ==================================================================
  if (is.null(transformed)) {
    if (verbose) cat("--- 1. Computing I-spline transform from scratch ---\n")
    if (is.null(ref_raster) || is.null(subs_raster))
      stop("Must provide ref_raster and subs_raster when 'transformed' is NULL")

    if (is.character(ref_raster))  ref_raster  <- raster::raster(ref_raster)
    if (is.character(subs_raster)) subs_raster <- raster::brick(subs_raster)

    cell_idx <- which(!is.na(raster::values(ref_raster)))
    xy       <- raster::xyFromCell(ref_raster, cell_idx)
    n_pixels <- length(cell_idx)
    coords   <- data.frame(lon = xy[, 1], lat = xy[, 2], cell = cell_idx,
                            stringsAsFactors = FALSE)

    ## Substrate
    subs_vals <- raster::extract(subs_raster, xy)
    colnames(subs_vals) <- paste0(colnames(subs_vals), "_1")

    ## Spatial climate
    if (is.null(env_spatial)) {
      if (is.null(npy_src) || is.null(python_exe) || is.null(pyper_script))
        stop("Must provide npy_src, python_exe, pyper_script for climate extraction")
      require(arrow)

      spatial_params <- list()
      for (ep in fit$env_params) {
        spatial_params[[length(spatial_params) + 1]] <- list(
          variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
          window = fit$climate_window, prefix = paste0("spat_", ep$cstat))
      }

      geonpy_start <- if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L
      n_chunks <- ceiling(n_pixels / chunk_size)
      env_parts <- vector("list", n_chunks)

      for (ch in seq_len(n_chunks)) {
        r1 <- (ch - 1) * chunk_size + 1
        r2 <- min(ch * chunk_size, n_pixels)
        chunk_xy <- xy[r1:r2, , drop = FALSE]; n_ch <- nrow(chunk_xy)
        if (verbose) cat(sprintf("  Climate chunk %d/%d (%d px)\n", ch, n_chunks, n_ch))
        pairs <- data.frame(
          Lon1 = chunk_xy[, 1], Lat1 = chunk_xy[, 2],
          year1 = as.integer(ref_year), month1 = as.integer(ref_month),
          Lon2 = chunk_xy[, 1], Lat2 = chunk_xy[, 2],
          year2 = as.integer(ref_year), month2 = as.integer(ref_month))
        chunk_parts <- list()
        for (j in seq_along(spatial_params)) {
          sp <- spatial_params[[j]]
          raw <- gen_windows(pairs = pairs, variables = sp$variables,
                             mstat = sp$mstat, cstat = sp$cstat,
                             window = sp$window, npy_src = npy_src,
                             start_year = geonpy_start,
                             python_exe = python_exe,
                             pyper_script = pyper_script)
          env_cols <- raw[, grep("_1$", names(raw)), drop = FALSE]
          colnames(env_cols) <- paste(sp$prefix, colnames(env_cols), sep = "_")
          colnames(env_cols) <- gsub("\\d{6}-\\d{6}_", "", colnames(env_cols))
          chunk_parts[[j]] <- env_cols
        }
        env_parts[[ch]] <- do.call(cbind, chunk_parts)
      }
      env_spatial <- do.call(rbind, env_parts)
    }

    ## MODIS extraction (spatial component, if model includes MODIS)
    modis_spat_vals <- NULL
    if (isTRUE(fit$add_modis)) {
      if (is.null(modis_dir)) {
        if (exists("config", envir = parent.frame()) &&
            !is.null(parent.frame()$config$modis_dir)) {
          modis_dir        <- parent.frame()$config$modis_dir
          modis_resolution <- parent.frame()$config$modis_resolution
        } else {
          warning("fit$add_modis is TRUE but modis_dir not provided -- MODIS predictors will be skipped")
        }
      }
      if (!is.null(modis_dir)) {
        if (verbose) cat("  Extracting MODIS spatial...\n")
        modis_spat_vals <- .extract_modis_columns(
          fit = fit, xy = xy, year = ref_year,
          modis_dir = modis_dir, modis_resolution = modis_resolution,
          col_prefix = "spat_modis", col_suffix = "_1", verbose = verbose
        )
      }
    }

    ## Condition extraction (spatial component, if model includes condition)
    cond_spat_vals <- NULL
    if (isTRUE(fit$add_condition)) {
      if (is.null(condition_tif)) {
        if (exists("config", envir = parent.frame()) &&
            !is.null(parent.frame()$config$condition_tif_path)) {
          condition_tif <- parent.frame()$config$condition_tif_path
        } else if (!is.null(fit$condition_tif_path)) {
          condition_tif <- fit$condition_tif_path
        } else {
          warning("fit$add_condition is TRUE but condition_tif not provided -- condition predictors will be skipped")
        }
      }
      if (!is.null(condition_tif)) {
        if (verbose) cat("  Extracting condition spatial...\n")
        cond_spat_vals <- .extract_condition_columns(
          fit = fit, xy = xy, year = ref_year,
          condition_tif = condition_tif,
          col_prefix = "spat_condition", col_suffix = "_1", verbose = verbose
        )
      }
    }

    env_df <- cbind(as.data.frame(env_spatial), as.data.frame(subs_vals))
    if (!is.null(modis_spat_vals)) env_df <- cbind(env_df, modis_spat_vals)
    if (!is.null(cond_spat_vals))  env_df <- cbind(env_df, cond_spat_vals)
    na_rows <- is.na(rowSums(env_df))
    sentinel <- apply(env_df, 1, function(r) any(r == -9999, na.rm = TRUE))
    bad <- na_rows | sentinel
    if (any(bad)) {
      env_df <- env_df[!bad, , drop = FALSE]
      coords <- coords[!bad, , drop = FALSE]
    }

    transformed <- transform_spatial_gdm(fit = fit, env_df = env_df,
                                          weight_by_coef = TRUE,
                                          spatial_only = TRUE, verbose = verbose)
  } else {
    if (is.null(coords)) stop("Must provide coords when 'transformed' is provided")
    if (verbose) cat("--- 1. Using pre-computed I-spline transform ---\n")
  }

  n_pixels <- nrow(transformed)

  ## Active (non-zero-variance) columns only
  col_var     <- apply(transformed, 2, var, na.rm = TRUE)
  active_cols <- which(col_var > 1e-15)
  trans_act   <- transformed[, active_cols, drop = FALSE]
  p_act       <- ncol(trans_act)

  if (verbose) cat(sprintf("  Pixels: %d | Active cols: %d\n", n_pixels, p_act))

  ## ==================================================================
  ## 2.  Select landmarks
  ## ==================================================================
  if (verbose) cat(sprintf("\n--- 2. Selecting %d landmarks (%s) ---\n",
                            n_landmarks, landmark_method))
  n_landmarks <- min(n_landmarks, n_pixels)

  if (landmark_method == "custom" && !is.null(landmark_idx)) {
    lm_idx <- landmark_idx[landmark_idx >= 1 & landmark_idx <= n_pixels]
  } else if (landmark_method == "stratified") {
    set.seed(seed)
    km <- kmeans(coords[, c("lon", "lat")], centers = n_landmarks,
                 nstart = 3, iter.max = 50)
    lm_idx <- integer(n_landmarks)
    for (cl in seq_len(n_landmarks)) {
      members <- which(km$cluster == cl)
      dx <- coords$lon[members] - km$centers[cl, 1]
      dy <- coords$lat[members] - km$centers[cl, 2]
      lm_idx[cl] <- members[which.min(dx^2 + dy^2)]
    }
  } else {
    set.seed(seed)
    lm_idx <- sort(sample.int(n_pixels, n_landmarks))
  }
  k <- length(lm_idx)
  cat(sprintf("  Selected %d landmarks\n", k))

  trans_lm <- trans_act[lm_idx, , drop = FALSE]

  ## ==================================================================
  ## 3.  Distance matrices
  ## ==================================================================
  ## Helper: ecological distance -> calibrated dissimilarity
  p0 <- inv.logit(fit$intercept)

  .eco_to_dissim <- function(d_vec) {
    eta   <- fit$intercept + d_vec
    p_hat <- inv.logit(eta)
    ObsTrans(p0, fit$w_ratio, p_hat)$out
  }

  ## 3a. D_LL (k × k)
  if (verbose) cat(sprintf("\n--- 3a. Landmark distance matrix (%d × %d) ---\n", k, k))
  D_LL <- matrix(0, nrow = k, ncol = k)
  for (i in seq_len(k - 1)) {
    for (j in (i + 1):k) {
      eco_d <- sum(abs(trans_lm[i, ] - trans_lm[j, ]))
      D_LL[i, j] <- eco_d
      D_LL[j, i] <- eco_d
    }
  }
  if (use_dissimilarity) {
    D_LL_raw <- D_LL
    D_LL <- matrix(.eco_to_dissim(as.vector(D_LL)), nrow = k, ncol = k)
    diag(D_LL) <- 0
  }

  ## 3b. D_NL (n × k)
  if (verbose) cat(sprintf("\n--- 3b. Pixel-to-landmark distances (%d × %d) ---\n", n_pixels, k))
  D_NL <- matrix(0, nrow = n_pixels, ncol = k)
  for (l in seq_len(k)) {
    if (verbose && l %% 100 == 0)
      cat(sprintf("    Landmark %d / %d ...\n", l, k))
    D_NL[, l] <- rowSums(abs(sweep(trans_act, 2, trans_lm[l, ])))
  }
  if (use_dissimilarity) {
    D_NL <- matrix(.eco_to_dissim(as.vector(D_NL)), nrow = n_pixels, ncol = k)
  }

  ## ==================================================================
  ## 4.  Classical MDS on landmarks + Nyström extension
  ## ==================================================================
  if (verbose) cat(sprintf("\n--- 4. Classical MDS + Nyström (%d components) ---\n", n_components))

  D2_LL     <- D_LL^2
  col_means <- colMeans(D2_LL)
  grand_mean <- mean(D2_LL)

  ## Double-centre: B = -0.5 * H * D² * H
  B_LL <- -0.5 * (sweep(sweep(D2_LL, 2, col_means), 1, col_means) + grand_mean)

  eig    <- eigen(B_LL, symmetric = TRUE)
  pos_ix <- which(eig$values > 1e-10)
  n_pos  <- length(pos_ix)
  m      <- min(n_components, n_pos)

  if (m < n_components) {
    if (verbose)
      cat(sprintf("  Note: only %d positive eigenvalues (requested %d)\n", n_pos, n_components))
    if (m == 0) stop("No positive eigenvalues -- distances may be degenerate")
  }

  lambda_m <- eig$values[pos_ix[1:m]]
  V_m      <- eig$vectors[, pos_ix[1:m], drop = FALSE]

  ## Landmark MDS coordinates
  lm_mds <- V_m %*% diag(sqrt(lambda_m), nrow = m)

  var_explained <- 100 * lambda_m / sum(pmax(eig$values, 0))
  n_neg <- sum(eig$values < -1e-10)
  if (verbose) {
    cat(sprintf("  Eigenvalue summary: %d positive, %d negative (of %d)\n",
                n_pos, n_neg, k))
    cat(sprintf("  Variance: Dim1=%.1f%%", var_explained[1]))
    if (m >= 2) cat(sprintf(", Dim2=%.1f%%", var_explained[2]))
    if (m >= 3) cat(sprintf(", Dim3=%.1f%%", var_explained[3]))
    cat(sprintf(" (total=%.1f%%)\n", sum(var_explained)))
  }

  ## Nyström extension:
  ##   y_i = -0.5 * Λ_m^{-1/2} * V_m' * (d²_i - μ_col)
  D2_NL   <- D_NL^2
  centred <- sweep(D2_NL, 2, col_means)
  scores  <- -0.5 * centred %*% V_m %*% diag(1 / sqrt(lambda_m), nrow = m)

  ## ==================================================================
  ## 5.  RGB mapping
  ## ==================================================================
  if (verbose) cat(sprintf("\n--- 5. Mapping to RGB (stretch = %g%%) ---\n", stretch))
  rgb_vals <- matrix(NA_real_, nrow = n_pixels, ncol = m)
  for (kk in seq_len(m)) {
    v  <- scores[, kk]
    lo <- quantile(v, stretch / 100, na.rm = TRUE)
    hi <- quantile(v, 1 - stretch / 100, na.rm = TRUE)
    if (hi <= lo) { lo <- min(v, na.rm = TRUE); hi <- max(v, na.rm = TRUE) }
    rgb_vals[, kk] <- round(pmin(pmax((v - lo) / (hi - lo), 0), 1) * 255)
  }

  ## ==================================================================
  ## 6.  Build rasters
  ## ==================================================================
  if (verbose) cat("\n--- 6. Building RGB rasters ---\n")
  if (is.null(ref_raster)) {
    if (!is.null(fit$ref_raster_path))
      ref_raster <- raster::raster(fit$ref_raster_path)
    else
      stop("ref_raster required for raster output")
  }
  if (is.character(ref_raster)) ref_raster <- raster::raster(ref_raster)
  template <- raster::raster(ref_raster)

  mk_layer <- function(vals) {
    ly <- template; raster::values(ly) <- NA
    ly[coords$cell] <- vals; ly
  }

  r_layer <- mk_layer(rgb_vals[, 1])
  g_layer <- if (m >= 2) mk_layer(rgb_vals[, 2]) else mk_layer(rep(0, n_pixels))
  b_layer <- if (m >= 3) mk_layer(rgb_vals[, 3]) else mk_layer(rep(0, n_pixels))

  rgb_stack <- raster::stack(r_layer, g_layer, b_layer)
  names(rgb_stack) <- c("MDS1_red", "MDS2_green", "MDS3_blue")

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("\n=== Landmark MDS prediction complete (%.1f s) ===\n", elapsed))

  list(
    rgb_stack          = rgb_stack,
    mds_scores         = scores,
    landmark_idx       = lm_idx,
    landmark_coords    = coords[lm_idx, ],
    landmark_mds       = lm_mds,
    D_LL               = D_LL,
    D_NL               = D_NL,
    eigenvalues        = eig$values,
    variance_explained = var_explained,
    n_neg_eigenvalues  = n_neg,
    coords             = coords,
    rgb_vals           = rgb_vals,
    n_landmarks        = k,
    n_components       = m,
    use_dissimilarity  = use_dissimilarity
  )
}


# ---------------------------------------------------------------------------
# spatial_dissimilarity_from_ref
#
# Compute calibrated GDM dissimilarity from one or more reference pixels
# to every other pixel. Returns a raster per reference pixel.
#
# This answers: "How biologically different is each pixel from this
# reference location?"
#
# Uses the pre-computed β-weighted I-spline transform: ecological distance
# is the Manhattan (L1) distance in transformed space (since all β ≥ 0
# from NNLS), which is then passed through intercept -> logit -> ObsTrans
# to give calibrated dissimilarity.
#
# Parameters:
#   fit          - fitted GDM list
#   ref_pixels   - reference pixels, specified as ONE of:
#                  (a) data.frame with lon, lat -- snapped to nearest pixel
#                  (b) integer vector of ROW INDICES into `transformed`
#   ref_raster   - reference raster (path or RasterLayer) for output grid
#   transformed  - pre-computed β-weighted I-spline transform matrix
#                  [n_pixels × total_splines] from transform_spatial_gdm().
#                  If NULL, will be computed (requires subs_raster, etc.)
#   coords       - data.frame with lon, lat, cell matching transformed rows
#   subs_raster  - substrate raster (required if transformed is NULL)
#   env_spatial, npy_src, python_exe, pyper_script, ref_year, ref_month,
#   chunk_size   - same as predict_spatial_rgb()
#   verbose      - print progress
#
# Returns:
#   List with:
#     $dissim_stack  - RasterStack (one layer per reference pixel, values
#                      are calibrated dissimilarity [0, 1])
#     $eco_dist_stack - RasterStack (raw ecological distance, unbounded)
#     $ref_info      - data.frame with lon, lat, cell, row_idx of refs
#     $coords        - all pixel coords
# ---------------------------------------------------------------------------
spatial_dissimilarity_from_ref <- function(
    fit,
    ref_pixels,
    ref_raster      = NULL,
    transformed     = NULL,
    coords          = NULL,
    subs_raster     = NULL,
    env_spatial     = NULL,
    npy_src         = NULL,
    python_exe      = NULL,
    pyper_script    = NULL,
    ref_year        = 1975L,
    ref_month       = 6L,
    modis_dir       = NULL,
    modis_resolution = "1km",
    condition_tif   = NULL,
    chunk_size      = 50000L,
    verbose         = TRUE
) {

  cat("=== Dissimilarity from Reference Site(s) ===\n\n")
  t0 <- proc.time()

  ## ---- 1. Get transformed matrix --------------------------------------
  if (is.null(transformed)) {
    if (verbose) cat("--- 1. Computing I-spline transform from scratch ---\n")
    if (is.null(ref_raster) || is.null(subs_raster))
      stop("Must provide ref_raster and subs_raster when 'transformed' is NULL")

    if (is.character(ref_raster))  ref_raster  <- raster::raster(ref_raster)
    if (is.character(subs_raster)) subs_raster <- raster::brick(subs_raster)

    cell_idx <- which(!is.na(raster::values(ref_raster)))
    xy       <- raster::xyFromCell(ref_raster, cell_idx)
    coords   <- data.frame(lon = xy[, 1], lat = xy[, 2], cell = cell_idx,
                            stringsAsFactors = FALSE)

    subs_vals <- raster::extract(subs_raster, xy)
    colnames(subs_vals) <- paste0(colnames(subs_vals), "_1")

    if (is.null(env_spatial)) {
      if (is.null(npy_src) || is.null(python_exe) || is.null(pyper_script))
        stop("Must provide npy_src, python_exe, pyper_script for extraction")
      require(arrow)

      spatial_params <- list()
      for (ep in fit$env_params) {
        spatial_params[[length(spatial_params) + 1]] <- list(
          variables = ep$variables, mstat = ep$mstat, cstat = ep$cstat,
          window = fit$climate_window, prefix = paste0("spat_", ep$cstat))
      }

      geonpy_start <- if (!is.null(fit$geonpy_start_year)) fit$geonpy_start_year else 1911L
      n_pixels <- length(cell_idx)
      n_chunks <- ceiling(n_pixels / chunk_size)
      env_parts <- vector("list", n_chunks)

      for (ch in seq_len(n_chunks)) {
        r1 <- (ch - 1) * chunk_size + 1
        r2 <- min(ch * chunk_size, n_pixels)
        chunk_xy <- xy[r1:r2, , drop = FALSE]; n_ch <- nrow(chunk_xy)
        if (verbose) cat(sprintf("  Climate chunk %d/%d (%d px)\n", ch, n_chunks, n_ch))
        pairs <- data.frame(
          Lon1 = chunk_xy[, 1], Lat1 = chunk_xy[, 2],
          year1 = as.integer(ref_year), month1 = as.integer(ref_month),
          Lon2 = chunk_xy[, 1], Lat2 = chunk_xy[, 2],
          year2 = as.integer(ref_year), month2 = as.integer(ref_month))
        chunk_parts <- list()
        for (j in seq_along(spatial_params)) {
          sp <- spatial_params[[j]]
          raw <- gen_windows(pairs = pairs, variables = sp$variables,
                             mstat = sp$mstat, cstat = sp$cstat,
                             window = sp$window, npy_src = npy_src,
                             start_year = geonpy_start,
                             python_exe = python_exe,
                             pyper_script = pyper_script)
          env_cols <- raw[, grep("_1$", names(raw)), drop = FALSE]
          colnames(env_cols) <- paste(sp$prefix, colnames(env_cols), sep = "_")
          colnames(env_cols) <- gsub("\\d{6}-\\d{6}_", "", colnames(env_cols))
          chunk_parts[[j]] <- env_cols
        }
        env_parts[[ch]] <- do.call(cbind, chunk_parts)
      }
      env_spatial <- do.call(rbind, env_parts)
    }

    ## MODIS extraction (spatial component, if model includes MODIS)
    modis_spat_vals <- NULL
    if (isTRUE(fit$add_modis)) {
      if (is.null(modis_dir)) {
        if (exists("config", envir = parent.frame()) &&
            !is.null(parent.frame()$config$modis_dir)) {
          modis_dir        <- parent.frame()$config$modis_dir
          modis_resolution <- parent.frame()$config$modis_resolution
        }
      }
      if (!is.null(modis_dir)) {
        if (verbose) cat("  Extracting MODIS spatial...\n")
        modis_spat_vals <- .extract_modis_columns(
          fit = fit, xy = xy, year = ref_year,
          modis_dir = modis_dir, modis_resolution = modis_resolution,
          col_prefix = "spat_modis", col_suffix = "_1", verbose = verbose
        )
      }
    }

    ## Condition extraction (spatial component, if model includes condition)
    cond_spat_vals <- NULL
    if (isTRUE(fit$add_condition)) {
      if (is.null(condition_tif)) {
        if (exists("config", envir = parent.frame()) &&
            !is.null(parent.frame()$config$condition_tif_path)) {
          condition_tif <- parent.frame()$config$condition_tif_path
        } else if (!is.null(fit$condition_tif_path)) {
          condition_tif <- fit$condition_tif_path
        }
      }
      if (!is.null(condition_tif)) {
        if (verbose) cat("  Extracting condition spatial...\n")
        cond_spat_vals <- .extract_condition_columns(
          fit = fit, xy = xy, year = ref_year,
          condition_tif = condition_tif,
          col_prefix = "spat_condition", col_suffix = "_1", verbose = verbose
        )
      }
    }

    env_df <- cbind(as.data.frame(env_spatial), as.data.frame(subs_vals))
    if (!is.null(modis_spat_vals)) env_df <- cbind(env_df, modis_spat_vals)
    if (!is.null(cond_spat_vals))  env_df <- cbind(env_df, cond_spat_vals)
    na_rows <- is.na(rowSums(env_df))
    sentinel <- apply(env_df, 1, function(r) any(r == -9999, na.rm = TRUE))
    bad <- na_rows | sentinel
    if (any(bad)) {
      env_df <- env_df[!bad, , drop = FALSE]
      coords <- coords[!bad, , drop = FALSE]
    }

    transformed <- transform_spatial_gdm(fit = fit, env_df = env_df,
                                          weight_by_coef = TRUE,
                                          spatial_only = TRUE, verbose = verbose)
  } else {
    if (is.null(coords)) stop("Must provide coords when 'transformed' is provided")
    if (verbose) cat("--- 1. Using pre-computed I-spline transform ---\n")
  }

  n_pixels <- nrow(transformed)

  ## Active columns
  col_var     <- apply(transformed, 2, var, na.rm = TRUE)
  active_cols <- which(col_var > 1e-15)
  trans_act   <- transformed[, active_cols, drop = FALSE]

  ## ---- 2. Resolve reference pixel indices -----------------------------
  if (verbose) cat("\n--- 2. Resolving reference pixels ---\n")

  if (is.data.frame(ref_pixels) && all(c("lon", "lat") %in% names(ref_pixels))) {
    ## Snap lon/lat to nearest pixel in coords
    ref_row_idx <- integer(nrow(ref_pixels))
    for (r in seq_len(nrow(ref_pixels))) {
      dx <- coords$lon - ref_pixels$lon[r]
      dy <- coords$lat - ref_pixels$lat[r]
      ref_row_idx[r] <- which.min(dx^2 + dy^2)
    }
    ref_info <- data.frame(
      lon     = coords$lon[ref_row_idx],
      lat     = coords$lat[ref_row_idx],
      cell    = coords$cell[ref_row_idx],
      row_idx = ref_row_idx,
      stringsAsFactors = FALSE
    )
    if (!is.null(ref_pixels$label))
      ref_info$label <- ref_pixels$label
    else
      ref_info$label <- paste0("ref_", seq_along(ref_row_idx))

  } else if (is.numeric(ref_pixels)) {
    ref_row_idx <- as.integer(ref_pixels)
    ref_row_idx <- ref_row_idx[ref_row_idx >= 1 & ref_row_idx <= n_pixels]
    ref_info <- data.frame(
      lon     = coords$lon[ref_row_idx],
      lat     = coords$lat[ref_row_idx],
      cell    = coords$cell[ref_row_idx],
      row_idx = ref_row_idx,
      label   = paste0("ref_", seq_along(ref_row_idx)),
      stringsAsFactors = FALSE
    )
  } else {
    stop("ref_pixels must be a data.frame with lon/lat columns or an integer vector of row indices")
  }

  n_refs <- nrow(ref_info)
  cat(sprintf("  Reference pixels: %d\n", n_refs))
  for (r in seq_len(n_refs)) {
    cat(sprintf("    [%d] %s: lon=%.3f, lat=%.3f (row %d, cell %d)\n",
                r, ref_info$label[r], ref_info$lon[r], ref_info$lat[r],
                ref_info$row_idx[r], ref_info$cell[r]))
  }

  ## ---- 3. Compute dissimilarity from each reference -------------------
  if (verbose) cat("\n--- 3. Computing dissimilarities ---\n")

  p0 <- inv.logit(fit$intercept)

  eco_dist_mat <- matrix(NA_real_, nrow = n_pixels, ncol = n_refs)
  dissim_mat   <- matrix(NA_real_, nrow = n_pixels, ncol = n_refs)

  for (r in seq_len(n_refs)) {
    ri <- ref_info$row_idx[r]
    if (verbose) cat(sprintf("  %s: computing Manhattan distances ...\n", ref_info$label[r]))

    ## Manhattan distance from reference to all pixels
    eco_d <- rowSums(abs(sweep(trans_act, 2, trans_act[ri, ])))

    ## Calibrated dissimilarity
    eta   <- fit$intercept + eco_d
    p_hat <- inv.logit(eta)
    d_obs <- ObsTrans(p0, fit$w_ratio, p_hat)$out

    eco_dist_mat[, r] <- eco_d
    dissim_mat[, r]   <- d_obs
  }

  ## ---- 4. Build rasters -----------------------------------------------
  if (verbose) cat("\n--- 4. Building rasters ---\n")

  if (is.null(ref_raster)) {
    if (!is.null(fit$ref_raster_path))
      ref_raster <- raster::raster(fit$ref_raster_path)
    else
      stop("ref_raster required for raster output")
  }
  if (is.character(ref_raster)) ref_raster <- raster::raster(ref_raster)
  template <- raster::raster(ref_raster)

  dissim_layers   <- list()
  eco_dist_layers <- list()

  for (r in seq_len(n_refs)) {
    ly_d <- template; raster::values(ly_d) <- NA
    ly_d[coords$cell] <- dissim_mat[, r]
    dissim_layers[[r]] <- ly_d

    ly_e <- template; raster::values(ly_e) <- NA
    ly_e[coords$cell] <- eco_dist_mat[, r]
    eco_dist_layers[[r]] <- ly_e
  }

  dissim_stack   <- raster::stack(dissim_layers)
  eco_dist_stack <- raster::stack(eco_dist_layers)
  names(dissim_stack)   <- ref_info$label
  names(eco_dist_stack) <- paste0(ref_info$label, "_ecodist")

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("\n=== Dissimilarity-from-reference complete (%.1f s) ===\n", elapsed))

  list(
    dissim_stack   = dissim_stack,
    eco_dist_stack = eco_dist_stack,
    dissim_mat     = dissim_mat,
    eco_dist_mat   = eco_dist_mat,
    ref_info       = ref_info,
    coords         = coords
  )
}
