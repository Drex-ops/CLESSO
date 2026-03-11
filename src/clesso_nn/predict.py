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

from .model import CLESSONet


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

    model = CLESSONet(
        K_alpha=cfg["K_alpha"],
        K_env=cfg["K_env"],
        alpha_hidden=cfg["alpha_hidden"],
        beta_hidden=cfg.get("beta_hidden", [64, 32, 16]),
        alpha_dropout=cfg["alpha_dropout"],
        beta_dropout=cfg["beta_dropout"],
        alpha_activation=cfg["alpha_activation"],
        alpha_lb_lambda=cfg.get("alpha_lb_lambda", 0.0),
        alpha_regression_lambda=cfg.get("alpha_regression_lambda", 0.0),
        beta_type=cfg.get("beta_type", "deep"),
        beta_n_knots=cfg.get("beta_n_knots", 32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


# --------------------------------------------------------------------------
# Alpha (richness) prediction
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_alpha(
    model: CLESSONet,
    site_covariates: pd.DataFrame | np.ndarray,
    ckpt: dict,
    device: str = "cpu",
) -> np.ndarray:
    """Predict species richness α for each site.

    Args:
        model: trained CLESSONet
        site_covariates: (n_sites, K_alpha) array or DataFrame of site
            covariates (same columns as training). Will be standardised
            using the stored training means/stds.
        ckpt: checkpoint dict (for normalisation stats)

    Returns:
        alpha: (n_sites,) array of richness estimates
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
    alpha = model.alpha_net(Z_t).cpu().numpy()
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
) -> dict[str, np.ndarray]:
    """Predict turnover η and similarity S for site pairs.

    Args:
        model: trained CLESSONet
        env_i, env_j: (n_pairs, K_env_cols) environmental covariate arrays
            for sites i and j respectively (raw, un-standardised)
        ckpt: checkpoint dict
        geo_i, geo_j: (n_pairs, 2) lon/lat arrays (if geo_distance was used)

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

    env_diff = np.concatenate(parts, axis=1) if parts else np.zeros((len(env_i), 0), dtype=np.float32)
    np.nan_to_num(env_diff, copy=False, nan=0.0)

    env_diff_t = torch.from_numpy(env_diff).to(device)
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
) -> dict[str, np.ndarray]:
    """Predict alpha for both sites and beta for each pair.

    Returns dict with: alpha_i, alpha_j, eta, similarity, p_match
    """
    alpha_i = predict_alpha(model, site_covariates_i, ckpt, device)
    alpha_j = predict_alpha(model, site_covariates_j, ckpt, device)
    beta = predict_beta(model, env_i, env_j, ckpt, device, geo_i, geo_j)

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

    Returns a dict mapping dimension index to (distances, eta_values).
    Useful for plotting response curves.
    """
    results = {}
    for dim in range(K_env):
        # Create input: one dimension varies 0→5, others held at 0
        grid = torch.zeros(n_points, K_env, device=device)
        grid[:, dim] = torch.linspace(0, 5, n_points)

        eta = model.beta_net(grid).cpu().numpy()
        distances = grid[:, dim].cpu().numpy()

        # Check monotonicity
        diffs = np.diff(eta)
        is_monotone = np.all(diffs >= -1e-6)

        results[dim] = {
            "distances": distances,
            "eta": eta,
            "is_monotone": is_monotone,
            "max_violation": float(np.min(diffs)) if len(diffs) > 0 else 0.0,
        }

    return results


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
