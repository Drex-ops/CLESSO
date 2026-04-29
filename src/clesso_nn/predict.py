"""
predict.py -- Prediction utilities for trained CLESSO NN model.

Load a trained model checkpoint and produce:
  - Alpha (richness) predictions for arbitrary sites
  - Beta (turnover / similarity) predictions for site pairs
  - Combined match-probability predictions
  - Monotonicity verification plots

Predictions are exportable as CSV/feather for use in R or other tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .model import CLESSONet, TransformBetaNet


# --------------------------------------------------------------------------
# Load trained model
# --------------------------------------------------------------------------

def load_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[CLESSONet, dict]:
    """Load a trained CLESSO NN model from checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Infer beta_no_intercept from state_dict if not saved in config
    beta_no_intercept = cfg.get("beta_no_intercept", None)
    if beta_no_intercept is None:
        sd = ckpt["model_state_dict"]
        beta_no_intercept = "beta_net.encoders.0.0.bias" not in sd

    model = CLESSONet(
        K_alpha=cfg["K_alpha"],
        K_env=cfg["K_env"],
        alpha_hidden=cfg["alpha_hidden"],
        alpha_dropout=cfg["alpha_dropout"],
        beta_dropout=cfg["beta_dropout"],
        alpha_activation=cfg["alpha_activation"],
        alpha_lb_lambda=cfg.get("alpha_lb_lambda", 0.0),
        alpha_regression_lambda=cfg.get("alpha_regression_lambda", 0.0),
        beta_type=cfg.get("beta_type", "transform"),
        beta_no_intercept=beta_no_intercept,
        transform_n_knots=cfg.get("transform_n_knots", 32),
        transform_g_knots=cfg.get("transform_g_knots", 16),
        K_effort=cfg.get("K_effort", 0),
        effort_hidden=cfg.get("effort_hidden", [64, 32]),
        effort_dropout=cfg.get("effort_dropout", 0.1),
        effort_mode=cfg.get("effort_mode", "additive"),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_alpha(
    model: CLESSONet,
    site_covariates: pd.DataFrame | np.ndarray,
    ckpt: dict,
    device: str = "cpu",
    mode: str = "env_only",
    effort_covariates: pd.DataFrame | np.ndarray | None = None,
) -> np.ndarray:
    """Predict species richness α for each site.

    Args:
        model: trained CLESSONet
        site_covariates: (n_sites, K_alpha) array or DataFrame of site
            covariates (same columns as training). Will be standardised
            using the stored training means/stds.
        ckpt: checkpoint dict (for normalisation stats)
        mode: "env_only" (true richness, default), "full" (env+effort),
              or "effort_only" (effort component logit only — for diagnostics)
        effort_covariates: (n_sites, K_effort) effort features.  Required
            when mode="full" or "effort_only" and model has effort_net.
            Will be standardised using stored w_mean / w_std.

    Returns:
        alpha: (n_sites,) array.  For mode="effort_only", returns raw
               effort logit (not softplus-shifted).
    """
    stats = ckpt["site_data_stats"]
    z_mean = np.array(stats["z_mean"], dtype=np.float32)
    z_std = np.array(stats["z_std"], dtype=np.float32)

    if isinstance(site_covariates, pd.DataFrame):
        # Select columns in training order
        cov_names = stats["alpha_cov_names"]

        # Add Fourier features if the model was trained with them
        fourier_n_freq = stats.get("fourier_n_frequencies", 0)
        fourier_max_wl = stats.get("fourier_max_wavelength", 40.0)
        if fourier_n_freq and fourier_n_freq > 0 and "lon" in site_covariates.columns:
            from .dataset import compute_fourier_features
            lon_deg = site_covariates["lon"].values.astype(np.float32)
            lat_deg = site_covariates["lat"].values.astype(np.float32)
            ff, ff_names = compute_fourier_features(
                lon_deg, lat_deg, fourier_n_freq, fourier_max_wl)
            # Add Fourier columns to the DataFrame copy
            site_covariates = site_covariates.copy()
            for k, name in enumerate(ff_names):
                site_covariates[name] = ff[:, k]

        Z_raw = site_covariates[cov_names].values.astype(np.float32)
    else:
        Z_raw = np.asarray(site_covariates, dtype=np.float32)

    Z = (Z_raw - z_mean) / z_std
    np.nan_to_num(Z, copy=False, nan=0.0)
    Z_t = torch.from_numpy(Z).to(device)

    # Prepare effort tensor if needed
    W_t = None
    if mode in ("full", "effort_only") and model.effort_net is not None:
        w_mean = np.array(stats.get("w_mean", []), dtype=np.float32)
        w_std = np.array(stats.get("w_std", []), dtype=np.float32)
        if effort_covariates is not None:
            if isinstance(effort_covariates, pd.DataFrame):
                effort_names = stats.get("effort_cov_names", [])
                W_raw = effort_covariates[effort_names].values.astype(np.float32)
            else:
                W_raw = np.asarray(effort_covariates, dtype=np.float32)
            W = (W_raw - w_mean) / w_std
            np.nan_to_num(W, copy=False, nan=0.0)
            W_t = torch.from_numpy(W).to(device)

    if mode == "env_only":
        alpha = model._compute_alpha_env_only(Z_t).cpu().numpy()
    elif mode == "full":
        alpha = model._compute_alpha(Z_t, W_t).cpu().numpy()
    elif mode == "effort_only":
        # Return raw effort logit (for mapping effort component)
        if model.effort_net is not None and W_t is not None:
            alpha = model.effort_net(W_t).squeeze(-1).cpu().numpy()
        else:
            alpha = np.zeros(Z_t.shape[0], dtype=np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return alpha


# --------------------------------------------------------------------------
# Beta (turnover / similarity) prediction
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_beta(
    model: CLESSONet,
    env_i: np.ndarray,
    env_j: np.ndarray,
    ckpt: dict,
    device: str = "cpu",
    geo_i: Optional[np.ndarray] = None,
    geo_j: Optional[np.ndarray] = None,
    lon_i: Optional[np.ndarray] = None,
    lat_i: Optional[np.ndarray] = None,
    lon_j: Optional[np.ndarray] = None,
    lat_j: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Predict turnover η and similarity S for site pairs.

    Args:
        model: trained CLESSONet
        env_i, env_j: (n_pairs, K_env_cols) environmental covariate arrays
            for sites i and j respectively (raw, un-standardised)
        ckpt: checkpoint dict
        geo_i, geo_j: (n_pairs, 2) lon/lat arrays (OLD geo approach — |Δlon|, |Δlat|)
        lon_i, lat_i, lon_j, lat_j: (n_pairs,) raw lon/lat in degrees
            for haversine geo_dist (NEW approach — include_geo_dist_in_beta)

    Returns:
        dict with:
            eta:        (n_pairs,) turnover values
            similarity: (n_pairs,) S = exp(-eta)
    """
    stats = ckpt["site_data_stats"]

    # Standardise env covariates and compute |diff|
    parts = []
    if env_i is not None and env_i.shape[1] > 0:
        e_mean = np.array(stats["e_mean"], dtype=np.float32)
        e_std = np.array(stats["e_std"], dtype=np.float32)
        ei_std = (env_i.astype(np.float32) - e_mean) / e_std
        ej_std = (env_j.astype(np.float32) - e_mean) / e_std
        parts.append(np.abs(ei_std - ej_std))

    if geo_i is not None:
        geo_mean = np.array(stats["geo_mean"], dtype=np.float32)
        geo_std = np.array(stats["geo_std"], dtype=np.float32)
        gi_std = (geo_i.astype(np.float32) - geo_mean) / geo_std
        gj_std = (geo_j.astype(np.float32) - geo_mean) / geo_std
        parts.append(np.abs(gi_std - gj_std))

    # Haversine geographic distance (NEW approach)
    if stats.get("include_geo_dist_in_beta", False):
        geo_dist_scale = stats.get("geo_dist_scale")
        if lon_i is not None and geo_dist_scale:
            from .dataset import haversine_km
            d_km = haversine_km(
                np.asarray(lon_i, dtype=np.float32),
                np.asarray(lat_i, dtype=np.float32),
                np.asarray(lon_j, dtype=np.float32),
                np.asarray(lat_j, dtype=np.float32),
            )
            d_norm = (d_km / geo_dist_scale).reshape(-1, 1)
            parts.append(d_norm)

    env_diff = np.concatenate(parts, axis=1) if parts else np.zeros((len(env_i), 0), dtype=np.float32)
    np.nan_to_num(env_diff, copy=False, nan=0.0)

    env_diff_t = torch.from_numpy(env_diff).to(device)

    if isinstance(model.beta_net, TransformBetaNet):
        # TransformBetaNet needs raw standardised per-site env values
        e_mean = np.array(stats["e_mean"], dtype=np.float32)
        e_std = np.array(stats["e_std"], dtype=np.float32)
        ei_std_t = torch.from_numpy(
            ((env_i.astype(np.float32) - e_mean) / e_std)).to(device)
        ej_std_t = torch.from_numpy(
            ((env_j.astype(np.float32) - e_mean) / e_std)).to(device)
        np.nan_to_num(ei_std_t.numpy() if device == "cpu" else ei_std_t.cpu().numpy(),
                      copy=False, nan=0.0)
        np.nan_to_num(ej_std_t.numpy() if device == "cpu" else ej_std_t.cpu().numpy(),
                      copy=False, nan=0.0)
        # Extract geo_dist if present
        geo_dist_t = None
        if stats.get("include_geo_dist_in_beta", False) and lon_i is not None:
            geo_dist_t = env_diff_t[:, -1:]
            # env_i/env_j are only the env columns (not geo)
        eta = model.beta_net(ei_std_t, ej_std_t, geo_dist=geo_dist_t).cpu().numpy()
    else:
        eta = model.beta_net(env_diff_t).cpu().numpy()
    similarity = np.exp(-eta)

    return {"eta": eta, "similarity": similarity}


# --------------------------------------------------------------------------
# Full prediction (alpha + beta combined)
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_full(
    model: CLESSONet,
    site_covariates_i: pd.DataFrame | np.ndarray,
    site_covariates_j: pd.DataFrame | np.ndarray,
    env_i: np.ndarray,
    env_j: np.ndarray,
    ckpt: dict,
    device: str = "cpu",
    geo_i: Optional[np.ndarray] = None,
    geo_j: Optional[np.ndarray] = None,
    lon_i: Optional[np.ndarray] = None,
    lat_i: Optional[np.ndarray] = None,
    lon_j: Optional[np.ndarray] = None,
    lat_j: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Predict alpha for both sites and beta for each pair.

    Returns dict with: alpha_i, alpha_j, eta, similarity, p_match
    """
    alpha_i = predict_alpha(model, site_covariates_i, ckpt, device)
    alpha_j = predict_alpha(model, site_covariates_j, ckpt, device)
    beta = predict_beta(model, env_i, env_j, ckpt, device, geo_i, geo_j,
                        lon_i, lat_i, lon_j, lat_j)

    # p_match for between-site pairs
    S = beta["similarity"]
    p_match = S * (alpha_i + alpha_j) / (2.0 * alpha_i * alpha_j)
    p_match = np.clip(p_match, 1e-7, 1.0 - 1e-7)

    return {
        "alpha_i": alpha_i,
        "alpha_j": alpha_j,
        "eta": beta["eta"],
        "similarity": beta["similarity"],
        "p_match": p_match,
    }


# --------------------------------------------------------------------------
# Monotonicity check
# --------------------------------------------------------------------------

@torch.no_grad()
def check_monotonicity(
    model: CLESSONet,
    K_env: int,
    n_points: int = 200,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Sweep each env distance dimension from 0 → max and verify η is non-decreasing.

    For TransformBetaNet, sweeps raw env values while holding the other site
    fixed at 0. For other beta nets, sweeps |env_diff| directly.

    Returns a dict mapping dimension index to (distances, eta_values).
    Useful for plotting response curves.
    """
    results = {}

    if isinstance(model.beta_net, TransformBetaNet):
        # Sweep raw env value for site i while site j stays at 0
        for dim in range(K_env):
            env_i = torch.zeros(n_points, K_env, device=device)
            env_j = torch.zeros(n_points, K_env, device=device)
            env_i[:, dim] = torch.linspace(-3, 3, n_points)
            # env_j stays at 0 → distance is |T(x) - T(0)|

            eta = model.beta_net(env_i, env_j).cpu().numpy()
            raw_values = env_i[:, dim].cpu().numpy()

            # Check: as raw value moves away from 0 (in either direction),
            # eta should be non-decreasing. Check positive direction.
            half = n_points // 2
            diffs_pos = np.diff(eta[half:])
            diffs_neg = np.diff(eta[:half+1][::-1])  # reverse: moving away from 0
            is_monotone = np.all(diffs_pos >= -1e-6) and np.all(diffs_neg >= -1e-6)

            results[dim] = {
                "distances": raw_values,
                "eta": eta,
                "is_monotone": is_monotone,
                "max_violation": float(min(np.min(diffs_pos), np.min(diffs_neg)))
                    if len(diffs_pos) > 0 else 0.0,
            }
    else:
        for dim in range(K_env):
            grid = torch.zeros(n_points, K_env, device=device)
            grid[:, dim] = torch.linspace(0, 5, n_points)

            eta = model.beta_net(grid).cpu().numpy()
            distances = grid[:, dim].cpu().numpy()

            diffs = np.diff(eta)
            is_monotone = np.all(diffs >= -1e-6)

            results[dim] = {
                "distances": distances,
                "eta": eta,
                "is_monotone": is_monotone,
                "max_violation": float(np.min(diffs)) if len(diffs) > 0 else 0.0,
            }

    return results


@torch.no_grad()
def predict_transform(
    model: CLESSONet,
    site_env: np.ndarray,
    ckpt: dict,
    device: str = "cpu",
) -> np.ndarray:
    """Apply T_k transforms to raw env values (TransformBetaNet only).

    Useful for spatial prediction: PCA on transformed space → RGB map.

    Args:
        model: trained CLESSONet with TransformBetaNet
        site_env: (N, K_env) raw environmental covariates (un-standardised)
        ckpt: checkpoint dict

    Returns:
        transformed: (N, K_env) T_k(standardised env_k) for each dimension

    Raises:
        TypeError: if model does not use TransformBetaNet
    """
    if not isinstance(model.beta_net, TransformBetaNet):
        raise TypeError("predict_transform requires a model with TransformBetaNet")

    stats = ckpt["site_data_stats"]
    e_mean = np.array(stats["e_mean"], dtype=np.float32)
    e_std = np.array(stats["e_std"], dtype=np.float32)
    env_std = (site_env.astype(np.float32) - e_mean) / e_std
    np.nan_to_num(env_std, copy=False, nan=0.0)

    env_t = torch.from_numpy(env_std).to(device)
    transformed = model.beta_net.transform_site(env_t).cpu().numpy()
    return transformed


# --------------------------------------------------------------------------
# Export predictions
# --------------------------------------------------------------------------

def export_alpha_predictions(
    model: CLESSONet,
    site_covariates: pd.DataFrame,
    ckpt: dict,
    output_path: str | Path,
    device: str = "cpu",
):
    """Predict alpha for all sites and save as feather."""
    alpha = predict_alpha(model, site_covariates, ckpt, device)
    result = site_covariates[["site_id"]].copy()
    result["alpha_nn"] = alpha
    import pyarrow.feather as pf
    pf.write_feather(result, str(output_path))
    print(f"Saved alpha predictions ({len(result)} sites) to {output_path}")
    return result
