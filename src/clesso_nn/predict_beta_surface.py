#!/usr/bin/env python3
"""
predict_beta_surface.py
=======================

Generate a spatial biological-community map of Australia using
Landmark MDS applied to the trained CLESSO NN beta (turnover) model.

Algorithm (follows the R predict_spatial_lmds approach):
  1. Extract all 17 env covariates at every land cell (same as alpha surface)
  2. Standardise env + geo using training statistics
  3. Select k landmark pixels via spatial k-means (stratified)
  4. Compute pairwise η between all landmark pairs (k×k)
     using the trained AdditiveBetaNet
  5. Compute η from every pixel to each landmark (n×k)
  6. Convert η to dissimilarity: D = 1 - exp(-η)
  7. Classical MDS on landmark D matrix + Nyström extension for all pixels
  8. Map 3 MDS dimensions → RGB
  9. Write 3-band GeoTIFF + PDF map

Usage:
    python predict_beta_surface.py                           # defaults
    python predict_beta_surface.py --n-landmarks 1000        # more landmarks
    python predict_beta_surface.py --output beta_surface.tif

Requirements:
    torch, numpy, rasterio, geonpy, matplotlib, pyshp, sklearn
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

# ──────────────────────────────────────────────────────────────────────────
# Default paths (relative to project root)
# ──────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _default_checkpoint():
    """Prefer calibrated checkpoint when it exists."""
    base = PROJECT_ROOT / "src" / "clesso_nn" / "output" / "VAS_hexBalance_nn" / "best_model.pt"
    calibrated = base.parent / "best_model_calibrated.pt"
    return calibrated if calibrated.exists() else base


DEFAULTS = dict(
    checkpoint=_default_checkpoint(),
    reference_raster=PROJECT_ROOT / "data" / "FWPT_mean_Cmax_mean_1946_1975.flt",
    substrate_raster=PROJECT_ROOT / "data" / "SUBS_brk_VAS.grd",
    npy_src="/Volumes/PortableSSD/CLIMATE/geonpy",
    output=None,  # derived from checkpoint dir at runtime
    n_landmarks=500,
    n_components=9,
    stretch=2.0,       # percentile stretch for RGB
    seed=42,
    batch_size=50_000,
    climate_year=2010,
    climate_month=6,
    climate_window=30,
    geonpy_start_year=1911,
)

# Date-range suffix on geonpy .npy files (matches training data period)
NPY_DATE_RANGE = "191101-201712"

# Stat prefix → (mstat_func, cstat_func) mapping used by geonpy
# mstat is always np.mean (monthly aggregation); cstat varies.
CSTAT_FUNCS = {
    "mean": (np.mean, np.mean),
    "min":  (np.mean, np.min),
    "max":  (np.mean, np.max),
}


def derive_climate_vars(cov_names: list[str]) -> list[tuple[str, str, callable, callable]]:
    """Derive climate extraction specs from the checkpoint's covariate names.

    Convention (matching run_clesso.R):
        column name  = "{cstat}_{varname}"
        npy file     = "{varname}_{NPY_DATE_RANGE}.npy"
        mstat        = np.mean  (always)
        cstat        = np.mean / np.min / np.max  (from prefix)

    Non-climate names (subs_*) are skipped.

    Returns: list of (col_name, npy_basename, mstat_func, cstat_func)
    """
    skip_prefixes = ("subs_",)
    specs = []
    for col in cov_names:
        if any(col.startswith(pfx) for pfx in skip_prefixes):
            continue
        parts = col.split("_", 1)
        if len(parts) != 2 or parts[0] not in CSTAT_FUNCS:
            continue  # not a climate column
        cstat_prefix, varname = parts
        mstat_func, cstat_func = CSTAT_FUNCS[cstat_prefix]
        npy_basename = f"{varname}_{NPY_DATE_RANGE}"
        specs.append((col, npy_basename, mstat_func, cstat_func))
    return specs


# ──────────────────────────────────────────────────────────────────────────
# Reuse covariate extraction from alpha surface script
# ──────────────────────────────────────────────────────────────────────────

def load_reference_grid(path):
    """Load reference raster → land mask + coordinate grids."""
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        height, width = src.height, src.width
        nodata = src.nodata

    if nodata is not None:
        mask = (data != nodata) & ~np.isnan(data)
    else:
        mask = ~np.isnan(data)

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    lons = transform.c + (cols + 0.5) * transform.a
    lats = transform.f + (rows + 0.5) * transform.e

    n_valid = int(mask.sum())
    print(f"  Reference grid: {height}×{width}, {n_valid:,} land cells "
          f"({100 * n_valid / data.size:.1f}%)")

    return dict(
        height=height, width=width, transform=transform,
        mask=mask, lons=lons, lats=lats, n_valid=n_valid,
    )


def extract_substrate(path, mask):
    """Read 6-band substrate raster at land cells → (n_valid, 6) float32."""
    with rasterio.open(path) as src:
        n_bands = src.count
        print(f"  Substrate raster: {n_bands} bands")
        all_bands = src.read()
        nodata = src.nodata

    cube = np.moveaxis(all_bands, 0, -1).astype(np.float32)
    vals = cube[mask]
    if nodata is not None:
        vals[vals == nodata] = np.nan
    return vals


def extract_climate(npy_src, lons, lats, climate_specs,
                    climate_year, climate_month,
                    climate_window, start_year):
    """Extract climate variables at land cells → dict[name → (n,) array].

    Args:
        climate_specs: list of (col_name, npy_basename, mstat_func, cstat_func)
                       as returned by derive_climate_vars()
    """
    from geonpy.geonpy import Geonpy, calc_climatology_window, gen_multi_index_slice

    npy_src = Path(npy_src)
    n_pts = len(lons)
    pts = np.column_stack([lons, lats])

    year_mon = np.array([[climate_year, climate_month]])
    window_months = climate_window * 12
    dim_idx = gen_multi_index_slice(year_mon, window_months, st_year=start_year)
    dim_idx = np.broadcast_to(dim_idx, (n_pts, window_months))

    print(f"  Climate window: {climate_window} years ending "
          f"{climate_year}/{climate_month:02d} ({window_months} months)")

    results = {}
    n_vars = len(climate_specs)
    for i, (col_name, npy_name, mstat, cstat) in enumerate(climate_specs, 1):
        npy_path = npy_src / f"{npy_name}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Climate file not found: {npy_path}")
        print(f"    [{i}/{n_vars}] {col_name} ← {npy_name}", end="  ")
        t0 = time.time()
        g = Geonpy(str(npy_path))
        raw = g.read_points(pts, dim_idx=dim_idx)
        vals = calc_climatology_window(raw, mstat, cstat)
        results[col_name] = vals.astype(np.float32)
        dt = time.time() - t0
        print(f"({dt:.1f}s)")
        del g
    return results


# ──────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device="cpu"):
    """Load trained CLESSO NN model from checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from clesso_nn.model import CLESSONet, TransformBetaNet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Infer beta_no_intercept from state_dict if not saved in config
    beta_no_intercept = cfg.get("beta_no_intercept", None)
    if beta_no_intercept is None:
        sd = ckpt["model_state_dict"]
        beta_no_intercept = "beta_net.encoders.0.0.bias" not in sd and \
                            "beta_net.dim_nets.0.0.bias" not in sd

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
        beta_no_intercept=beta_no_intercept,
        transform_n_knots=cfg.get("transform_n_knots", 32),
        transform_g_knots=cfg.get("transform_g_knots", 16),
        K_effort=cfg.get("K_effort", 0),
        effort_hidden=cfg.get("effort_hidden", [64, 32]),
        effort_dropout=cfg.get("effort_dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


# ──────────────────────────────────────────────────────────────────────────
# Standardise env covariates (matching training pipeline)
# ──────────────────────────────────────────────────────────────────────────

def standardise_env(env_raw, geo_raw, ckpt):
    """Standardise env + geo covariates using training stats.

    Args:
        env_raw: (n, 15) raw env covariates (subs + climate) in training order
        geo_raw: (n, 2) raw lon, lat (ignored if model was trained without geo)
        ckpt: checkpoint dict

    Returns:
        E_std: (n, 15) standardised env covariates
        G_std: (n, 2) standardised geographic coordinates, or None if geo excluded
    """
    stats = ckpt["site_data_stats"]
    e_mean = np.array(stats["e_mean"], dtype=np.float32)
    e_std = np.array(stats["e_std"], dtype=np.float32)

    E_std = (env_raw - e_mean) / e_std
    np.nan_to_num(E_std, copy=False, nan=0.0)

    # Geo may be absent if model was trained with include_geo_in_beta=False
    if "geo_mean" in stats and stats["geo_mean"] is not None:
        geo_mean = np.array(stats["geo_mean"], dtype=np.float32)
        geo_std = np.array(stats["geo_std"], dtype=np.float32)
        G_std = (geo_raw - geo_mean) / geo_std
        np.nan_to_num(G_std, copy=False, nan=0.0)
    else:
        G_std = None

    return E_std, G_std


# ──────────────────────────────────────────────────────────────────────────
# Landmark selection (stratified k-means)
# ──────────────────────────────────────────────────────────────────────────

def select_landmarks_stratified(lons, lats, n_landmarks, seed=42):
    """Select spatially even landmarks using MiniBatchKMeans on (lon, lat).

    Returns: integer indices into the land-cell arrays.
    """
    from sklearn.cluster import MiniBatchKMeans

    coords = np.column_stack([lons, lats])
    n_landmarks = min(n_landmarks, len(lons))

    print(f"  Running MiniBatchKMeans ({n_landmarks} clusters)...")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=n_landmarks, random_state=seed,
        batch_size=min(10_000, len(lons)), n_init=3, max_iter=50,
    )
    km.fit(coords)
    dt = time.time() - t0
    print(f"  Clustering done ({dt:.1f}s)")

    # Pick pixel closest to each cluster centre
    lm_idx = np.empty(n_landmarks, dtype=np.int64)
    for cl in range(n_landmarks):
        members = np.where(km.labels_ == cl)[0]
        dx = coords[members, 0] - km.cluster_centers_[cl, 0]
        dy = coords[members, 1] - km.cluster_centers_[cl, 1]
        lm_idx[cl] = members[np.argmin(dx ** 2 + dy ** 2)]

    return lm_idx


# ──────────────────────────────────────────────────────────────────────────
# Compute η via the beta net
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_eta_pairwise(model, E_std_a, G_std_a, E_std_b, G_std_b,
                         device="cpu", batch_size=100_000,
                         lons_a=None, lats_a=None, lons_b=None, lats_b=None,
                         geo_dist_scale=None):
    """Compute η = beta_net(|env_diff|) for pairs (a[i], b[i]).

    Args:
        E_std_a, E_std_b: (n, 15) standardised env covariates
        G_std_a, G_std_b: (n, 2)  standardised geo coordinates, or None if no geo
        lons_a, lats_a, lons_b, lats_b: raw lon/lat in degrees for haversine geo_dist
        geo_dist_scale: float, if provided, append haversine_km / scale to env_diff

    Returns:
        eta: (n,) float32 array
    """
    n = E_std_a.shape[0]

    from clesso_nn.model import TransformBetaNet
    _is_transform = isinstance(model.beta_net, TransformBetaNet)

    if _is_transform:
        # TransformBetaNet takes (env_i, env_j, geo_dist)
        if G_std_a is not None and G_std_b is not None:
            env_a = np.hstack([E_std_a, G_std_a]).astype(np.float32)
            env_b = np.hstack([E_std_b, G_std_b]).astype(np.float32)
        else:
            env_a = E_std_a.astype(np.float32)
            env_b = E_std_b.astype(np.float32)

        geo_d_norm = None
        if geo_dist_scale is not None and lons_a is not None:
            from clesso_nn.dataset import haversine_km
            d_km = haversine_km(lons_a, lats_a, lons_b, lats_b).astype(np.float32)
            geo_d_norm = (d_km / geo_dist_scale).reshape(-1, 1)

        eta = np.empty(n, dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ei_t = torch.from_numpy(env_a[start:end]).to(device)
            ej_t = torch.from_numpy(env_b[start:end]).to(device)
            gd_t = None
            if geo_d_norm is not None:
                gd_t = torch.from_numpy(geo_d_norm[start:end]).to(device)
            eta[start:end] = model.beta_net(ei_t, ej_t, geo_dist=gd_t).cpu().numpy()
        return eta

    # --- Non-transform beta nets: pass |env_diff| ---
    if G_std_a is not None and G_std_b is not None:
        env_diff = np.abs(
            np.hstack([E_std_a, G_std_a]) - np.hstack([E_std_b, G_std_b])
        ).astype(np.float32)
    else:
        env_diff = np.abs(E_std_a - E_std_b).astype(np.float32)

    # Append haversine geographic distance if configured
    if geo_dist_scale is not None and lons_a is not None:
        from clesso_nn.dataset import haversine_km
        d_km = haversine_km(lons_a, lats_a, lons_b, lats_b).astype(np.float32)
        d_norm = (d_km / geo_dist_scale).reshape(-1, 1)
        env_diff = np.hstack([env_diff, d_norm])

    eta = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        diff_t = torch.from_numpy(env_diff[start:end]).to(device)
        eta[start:end] = model.beta_net(diff_t).cpu().numpy()
    return eta


# ──────────────────────────────────────────────────────────────────────────
# Landmark MDS + Nyström extension
# ──────────────────────────────────────────────────────────────────────────

def landmark_mds(D_LL, D_NL, n_components=3):
    """Classical MDS on landmark distance matrix + Nyström extension.

    Args:
        D_LL: (k, k) landmark–landmark dissimilarity matrix
        D_NL: (n, k) pixel–landmark dissimilarity matrix
        n_components: number of MDS dimensions

    Returns:
        scores:  (n, m) MDS coordinates for all pixels
        lm_mds:  (k, m) MDS coordinates for landmarks
        eigenvalues: all eigenvalues from MDS
        var_explained: % variance per dimension
    """
    k = D_LL.shape[0]

    # Double-centre the squared distance matrix
    D2_LL = D_LL ** 2
    col_means = D2_LL.mean(axis=0)
    grand_mean = D2_LL.mean()
    B_LL = -0.5 * (D2_LL - col_means[None, :] - col_means[:, None] + grand_mean)

    # Eigendecomposition
    eigenvalues, V = np.linalg.eigh(B_LL)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    pos_mask = eigenvalues > 1e-10
    n_pos = pos_mask.sum()
    n_neg = (eigenvalues < -1e-10).sum()
    m = min(n_components, n_pos)

    print(f"  Eigenvalue summary: {n_pos} positive, {n_neg} negative (of {k})")
    if m < n_components:
        print(f"  Note: only {n_pos} positive eigenvalues (requested {n_components})")
    if m == 0:
        raise ValueError("No positive eigenvalues — distances may be degenerate")

    lambda_m = eigenvalues[:m]
    V_m = V[:, :m]

    # Landmark MDS coordinates
    lm_mds = V_m * np.sqrt(lambda_m)[None, :]

    # Variance explained
    pos_total = eigenvalues[eigenvalues > 0].sum()
    var_explained = 100.0 * lambda_m / pos_total

    for i in range(m):
        print(f"  Dim {i + 1}: eigenvalue={lambda_m[i]:.2f}, "
              f"variance={var_explained[i]:.1f}%")
    print(f"  Total variance (top {m}): {var_explained.sum():.1f}%")

    # Nyström extension: project all n pixels into MDS space
    # y_i = -0.5 * Λ_m^{-1/2} * V_m' * (d²_i - μ_col)
    D2_NL = D_NL ** 2
    centred = D2_NL - col_means[None, :]
    scores = -0.5 * centred @ V_m * (1.0 / np.sqrt(lambda_m))[None, :]

    return scores, lm_mds, eigenvalues, var_explained, n_neg


# ──────────────────────────────────────────────────────────────────────────
# RGB mapping with percentile stretch
# ──────────────────────────────────────────────────────────────────────────

def scores_to_rgb(scores, stretch=2.0):
    """Map MDS scores to 0–255 RGB via percentile stretch.

    Args:
        scores: (n, 3) MDS coordinates
        stretch: percentile to clip at each end (e.g., 2 = 2nd to 98th)

    Returns:
        rgb: (n, 3) uint8 array
    """
    m = scores.shape[1]
    rgb = np.zeros((scores.shape[0], 3), dtype=np.uint8)
    for k in range(min(m, 3)):
        v = scores[:, k]
        lo = np.percentile(v, stretch)
        hi = np.percentile(v, 100 - stretch)
        if hi <= lo:
            lo, hi = v.min(), v.max()
        if hi <= lo:
            continue  # constant dimension → leave channel at 0
        scaled = np.clip((v - lo) / (hi - lo), 0, 1)
        rgb[:, k] = (scaled * 255).astype(np.uint8)
    return rgb


# ──────────────────────────────────────────────────────────────────────────
# Write 3-band RGB GeoTIFF
# ──────────────────────────────────────────────────────────────────────────

def write_rgb_geotiff(output_path, rgb_vals, mask, grid):
    """Write 3-band RGB GeoTIFF."""
    height, width = grid["height"], grid["width"]
    out = np.zeros((3, height, width), dtype=np.uint8)
    for band in range(3):
        layer = np.zeros((height, width), dtype=np.uint8)
        layer[mask] = rgb_vals[:, band]
        out[band] = layer

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 3,
        "crs": "EPSG:4326",
        "transform": grid["transform"],
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out)
        dst.update_tags(
            DESCRIPTION="CLESSO NN beta turnover (Landmark MDS → RGB)",
            MODEL="CLESSONet VAS — AdditiveBetaNet",
            RED="MDS Dimension 1",
            GREEN="MDS Dimension 2",
            BLUE="MDS Dimension 3",
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Written: {output_path} ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────
# PDF map generation (2 pages)
# ──────────────────────────────────────────────────────────────────────────

def plot_beta_map(rgb_vals, mds_scores, grid, eigenvalues, var_explained,
                  n_neg, n_landmarks, pdf_path,
                  shapefile_path=None, epoch=0, val_loss=0.0,
                  lm_lons=None, lm_lats=None, stretch=2.0,
                  extra_triplets=None):
    """Produce a multi-page PDF with beta surface maps and diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = grid["height"], grid["width"]
    mask = grid["mask"]
    transform = grid["transform"]

    x_min = transform.c
    y_max = transform.f
    x_max = x_min + width * transform.a
    y_min = y_max + height * transform.e
    extent = [x_min, x_max, y_min, y_max]

    # Build 2D RGB image (H, W, 3) with white ocean
    rgb_img = np.full((height, width, 3), 230, dtype=np.uint8)
    rgb_img[mask] = rgb_vals

    # Load IBRA boundaries (optional)
    boundary_shapes = None
    if shapefile_path is not None:
        shapefile_path = Path(shapefile_path)
        if shapefile_path.exists():
            try:
                import shapefile as shp
                sf = shp.Reader(str(shapefile_path))
                boundary_shapes = sf.shapes()
                print(f"  Loaded {len(boundary_shapes)} IBRA boundaries")
            except Exception as e:
                print(f"  Warning: could not read shapefile: {e}")

    m = min(mds_scores.shape[1], 3)
    dim_labels = ["Dim 1 (Red)", "Dim 2 (Green)", "Dim 3 (Blue)"]
    dim_colors = ["#B2182B", "#238B45", "#2166AC"]

    with PdfPages(str(pdf_path)) as pdf:
        # ─── Page 1: Main RGB map ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(rgb_img, extent=extent, origin="upper",
                  interpolation="nearest", aspect="equal")

        # Overlay IBRA boundaries
        if boundary_shapes is not None:
            for shape in boundary_shapes:
                for i_part in range(len(shape.parts)):
                    i_start = shape.parts[i_part]
                    i_end = (shape.parts[i_part + 1]
                             if i_part + 1 < len(shape.parts)
                             else len(shape.points))
                    pts = np.array(shape.points[i_start:i_end])
                    if len(pts) > 2:
                        ax.plot(pts[:, 0], pts[:, 1],
                                color="#333333", linewidth=0.25, alpha=0.4)

        # Mark landmarks
        if lm_lons is not None and lm_lats is not None:
            ax.scatter(lm_lons, lm_lats, s=1.5, c="white",
                       edgecolors="black", linewidths=0.3, alpha=0.6,
                       zorder=5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Longitude (°E)", fontsize=11)
        ax.set_ylabel("Latitude (°S)", fontsize=11)
        ax.set_title(
            f"CLESSO NN — Biological Space (Landmark MDS of β turnover)\n"
            f"Epoch {epoch} | Val loss {val_loss:.4f} | "
            f"{n_landmarks} landmarks | {stretch}% stretch",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, linewidth=0.3, alpha=0.3, color="grey")

        # Legend
        legend_text = []
        for i in range(m):
            legend_text.append(f"  {dim_labels[i]}: {var_explained[i]:.1f}%")
        legend_text.append(f"  Total: {var_explained[:m].sum():.1f}%")
        legend_text.append(f"  Neg eigenvalues: {n_neg}/{n_landmarks}")
        ax.text(0.02, 0.02, "\n".join(legend_text),
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="grey"))

        fig.tight_layout()
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

        # ─── Page 2: Individual dimension maps ───────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for k in range(3):
            ax = axes[k]
            if k < m:
                dim_2d = np.full((height, width), np.nan, dtype=np.float32)
                dim_2d[mask] = mds_scores[:, k]
                vlo = np.percentile(mds_scores[:, k], stretch)
                vhi = np.percentile(mds_scores[:, k], 100 - stretch)

                cmap = plt.cm.RdYlBu_r.copy()
                cmap.set_bad(color="#e8e8e8")
                im = ax.imshow(dim_2d, extent=extent, origin="upper",
                               cmap=cmap, vmin=vlo, vmax=vhi,
                               interpolation="nearest", aspect="equal")
                fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
                ax.set_title(f"{dim_labels[k]} ({var_explained[k]:.1f}%)",
                             fontsize=11, fontweight="bold",
                             color=dim_colors[k])
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "(not available)", ha="center",
                        va="center", transform=ax.transAxes)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig.suptitle("MDS Dimension Maps", fontsize=13, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ─── Page 3: Eigenvalue spectrum + landmark distribution ─────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Eigenvalue spectrum
        ax_eig = axes[0]
        n_show = min(40, len(eigenvalues))
        evals_show = eigenvalues[:n_show]
        colors = []
        for i, ev in enumerate(evals_show):
            if i == 0:
                colors.append("#B2182B")
            elif i == 1:
                colors.append("#238B45")
            elif i == 2:
                colors.append("#2166AC")
            elif ev > 0:
                colors.append("#74ADD1")
            else:
                colors.append("#F4A582")
        ax_eig.bar(range(n_show), evals_show, color=colors, edgecolor="white",
                   linewidth=0.3)
        ax_eig.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_eig.set_xlabel("Dimension", fontsize=11)
        ax_eig.set_ylabel("Eigenvalue", fontsize=11)
        ax_eig.set_title(
            f"Eigenvalue Spectrum ({n_landmarks} landmarks)\n"
            f"{(eigenvalues > 1e-10).sum()} positive, "
            f"{(eigenvalues < -1e-10).sum()} negative",
            fontsize=11, fontweight="bold")

        # Right: Landmark distribution
        ax_lm = axes[1]
        bg_2d = np.full((height, width), np.nan, dtype=np.float32)
        bg_2d[mask] = 0.8
        ax_lm.imshow(bg_2d, extent=extent, origin="upper",
                     cmap="Greys", vmin=0, vmax=1,
                     interpolation="nearest", aspect="equal")
        if lm_lons is not None:
            ax_lm.scatter(lm_lons, lm_lats, s=4, c="#D62728",
                         edgecolors="black", linewidths=0.2, zorder=5)
        ax_lm.set_xlim(x_min, x_max)
        ax_lm.set_ylim(y_min, y_max)
        ax_lm.set_xlabel("Longitude (°E)", fontsize=11)
        ax_lm.set_ylabel("Latitude (°S)", fontsize=11)
        ax_lm.set_title(
            f"Landmark Distribution (stratified, n={n_landmarks})",
            fontsize=11, fontweight="bold")

        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ─── Extra triplet pages (dims 4-6, 7-9, …) ────────────
        if extra_triplets:
            for dim_start, trip_rgb in extra_triplets:
                dim_end = min(dim_start + 3, mds_scores.shape[1])
                n_trip = dim_end - dim_start
                trip_channels = ["Red", "Green", "Blue"][:n_trip]
                trip_var = var_explained[dim_start:dim_end]

                # RGB map page
                trip_img = np.full((height, width, 3), 230, dtype=np.uint8)
                trip_img[mask] = trip_rgb

                fig, ax = plt.subplots(figsize=(12, 10))
                ax.imshow(trip_img, extent=extent, origin="upper",
                          interpolation="nearest", aspect="equal")

                if boundary_shapes is not None:
                    for shape in boundary_shapes:
                        for i_part in range(len(shape.parts)):
                            i_start = shape.parts[i_part]
                            i_end = (shape.parts[i_part + 1]
                                     if i_part + 1 < len(shape.parts)
                                     else len(shape.points))
                            pts = np.array(shape.points[i_start:i_end])
                            if len(pts) > 2:
                                ax.plot(pts[:, 0], pts[:, 1],
                                        color="#333333", linewidth=0.25,
                                        alpha=0.4)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("Longitude (°E)", fontsize=11)
                ax.set_ylabel("Latitude (°S)", fontsize=11)
                ax.set_title(
                    f"Biological Space — Dims {dim_start+1}–"
                    f"{dim_start+n_trip} "
                    f"({sum(trip_var):.1f}% variance)\n"
                    f"Epoch {epoch} | {n_landmarks} landmarks | "
                    f"{stretch}% stretch",
                    fontsize=13, fontweight="bold")
                ax.grid(True, linewidth=0.3, alpha=0.3, color="grey")

                legend_lines = []
                for i in range(n_trip):
                    legend_lines.append(
                        f"  Dim {dim_start+i+1} ({trip_channels[i]}): "
                        f"{trip_var[i]:.1f}%")
                legend_lines.append(f"  Total: {sum(trip_var):.1f}%")
                ax.text(0.02, 0.02, "\n".join(legend_lines),
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="white", alpha=0.85,
                                  edgecolor="grey"))

                fig.tight_layout()
                pdf.savefig(fig, dpi=200)
                plt.close(fig)

                # Individual dimension maps page
                dim_colors_t = ["#B2182B", "#238B45", "#2166AC"]
                fig, axes = plt.subplots(1, 3, figsize=(16, 6))
                for kk in range(3):
                    ax = axes[kk]
                    d = dim_start + kk
                    if d < mds_scores.shape[1]:
                        dim_2d = np.full((height, width), np.nan,
                                         dtype=np.float32)
                        dim_2d[mask] = mds_scores[:, d]
                        vlo = np.percentile(mds_scores[:, d], stretch)
                        vhi = np.percentile(mds_scores[:, d],
                                            100 - stretch)
                        cmap = plt.cm.RdYlBu_r.copy()
                        cmap.set_bad(color="#e8e8e8")
                        im = ax.imshow(dim_2d, extent=extent,
                                       origin="upper", cmap=cmap,
                                       vmin=vlo, vmax=vhi,
                                       interpolation="nearest",
                                       aspect="equal")
                        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
                        dv = trip_var[kk] if kk < len(trip_var) else 0
                        ax.set_title(
                            f"Dim {d+1} ({trip_channels[kk]}) "
                            f"({dv:.1f}%)",
                            fontsize=11, fontweight="bold",
                            color=dim_colors_t[kk])
                    else:
                        ax.axis("off")
                        ax.text(0.5, 0.5, "(not available)",
                                ha="center", va="center",
                                transform=ax.transAxes)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)

                fig.suptitle(
                    f"MDS Dimensions {dim_start+1}–{dim_start+n_trip}",
                    fontsize=13, fontweight="bold")
                fig.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

    print(f"  Written: {pdf_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict biological community (beta turnover) surface using Landmark MDS"
    )
    parser.add_argument("--checkpoint", type=str,
                        default=str(DEFAULTS["checkpoint"]))
    parser.add_argument("--reference-raster", type=str,
                        default=str(DEFAULTS["reference_raster"]))
    parser.add_argument("--substrate-raster", type=str,
                        default=str(DEFAULTS["substrate_raster"]))
    parser.add_argument("--npy-src", type=str,
                        default=str(DEFAULTS["npy_src"]))
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Output GeoTIFF path (default: <checkpoint_dir>/beta_surface.tif)")
    parser.add_argument("--n-landmarks", type=int,
                        default=DEFAULTS["n_landmarks"])
    parser.add_argument("--n-components", type=int,
                        default=DEFAULTS["n_components"])
    parser.add_argument("--stretch", type=float,
                        default=DEFAULTS["stretch"])
    parser.add_argument("--seed", type=int,
                        default=DEFAULTS["seed"])
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULTS["batch_size"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    t_start = time.time()

    # Derive output path from checkpoint dir if not explicitly provided
    if args.output is None:
        args.output = str(Path(args.checkpoint).parent / "beta_surface.tif")

    print("=" * 70)
    print("  CLESSO NN — Beta Surface (Landmark MDS)")
    print("=" * 70)

    # ── 1. Load model ────────────────────────────────────────────────
    print("\n[1/8] Loading model checkpoint...")
    model, ckpt = load_model(args.checkpoint, args.device)
    stats = ckpt["site_data_stats"]
    env_cov_names = stats["env_cov_names"]
    print(f"  Model epoch: {ckpt['epoch']}")
    has_geo = "geo_mean" in stats and stats["geo_mean"] is not None
    has_geo_dist = stats.get("include_geo_dist_in_beta", False)
    geo_dist_scale = stats.get("geo_dist_scale") if has_geo_dist else None
    n_geo = 2 if has_geo else 0
    n_geo_dist = 1 if has_geo_dist else 0
    print(f"  K_env: {ckpt['config']['K_env']} "
          f"({len(env_cov_names)} env + {n_geo} geo + {n_geo_dist} geo_dist)")
    print(f"  Val loss: {ckpt['val_loss']:.4f}")
    if has_geo_dist:
        print(f"  Geo distance scale: {geo_dist_scale:.1f} km")

    # ── 2. Load reference grid ───────────────────────────────────────
    print("\n[2/8] Loading reference grid...")
    grid = load_reference_grid(args.reference_raster)
    mask = grid["mask"]
    n_valid = grid["n_valid"]

    # ── 3. Extract covariates ────────────────────────────────────────
    print("\n[3/8] Extracting covariates...")
    lons_valid = grid["lons"][mask].astype(np.float32)
    lats_valid = grid["lats"][mask].astype(np.float32)
    print(f"  Coordinates: lon [{lons_valid.min():.2f}, {lons_valid.max():.2f}], "
          f"lat [{lats_valid.min():.2f}, {lats_valid.max():.2f}]")

    # Substrate PCA (6 bands)
    subs_vals = extract_substrate(args.substrate_raster, mask)

    # Derive which climate variables to extract from the checkpoint
    climate_specs = derive_climate_vars(env_cov_names)
    print(f"  Climate variables to extract ({len(climate_specs)}): "
          f"{[s[0] for s in climate_specs]}")

    # Climate
    climate = extract_climate(
        args.npy_src, lons_valid, lats_valid,
        climate_specs=climate_specs,
        climate_year=DEFAULTS["climate_year"],
        climate_month=DEFAULTS["climate_month"],
        climate_window=DEFAULTS["climate_window"],
        start_year=DEFAULTS["geonpy_start_year"],
    )

    # Assemble env covariates in training order (15 cols)
    env_arrays = {}
    for i in range(subs_vals.shape[1]):
        env_arrays[f"subs_{i + 1}"] = subs_vals[:, i]
    env_arrays.update(climate)

    env_raw = np.column_stack([env_arrays[name] for name in env_cov_names])
    geo_raw = np.column_stack([lons_valid, lats_valid])
    print(f"  Env matrix: {env_raw.shape}, Geo: {'included' if has_geo else 'excluded'}")

    # ── 4. Standardise ──────────────────────────────────────────────
    print("\n[4/8] Standardising covariates...")
    E_std, G_std = standardise_env(env_raw, geo_raw, ckpt)

    # Check for NaN cells
    nan_env = np.isnan(env_raw).any(axis=1)
    nan_geo = np.isnan(geo_raw).any(axis=1)
    nan_mask = nan_env | nan_geo
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  Warning: {n_nan:,} cells have NaN covariates — excluded from MDS")

    # ── 5. Select landmarks ─────────────────────────────────────────
    print(f"\n[5/8] Selecting {args.n_landmarks} landmarks (stratified)...")
    lm_idx = select_landmarks_stratified(
        lons_valid, lats_valid, args.n_landmarks, args.seed
    )
    k = len(lm_idx)
    print(f"  Selected {k} landmarks")

    E_lm = E_std[lm_idx]
    G_lm = G_std[lm_idx] if G_std is not None else None
    lons_lm = lons_valid[lm_idx]
    lats_lm = lats_valid[lm_idx]

    # ── 6. Compute distance matrices ────────────────────────────────
    print(f"\n[6/8] Computing distance matrices...")

    # 6a. D_LL: landmark × landmark (k × k)
    print(f"  6a. Landmark–landmark η ({k}×{k} = {k * (k - 1) // 2:,} pairs)...")
    t0 = time.time()

    # Build all landmark pair indices
    i_idx, j_idx = np.triu_indices(k, k=1)
    n_pairs = len(i_idx)

    eta_ll = compute_eta_pairwise(
        model,
        E_std_a=E_lm[i_idx], G_std_a=G_lm[i_idx] if G_lm is not None else None,
        E_std_b=E_lm[j_idx], G_std_b=G_lm[j_idx] if G_lm is not None else None,
        device=args.device, batch_size=args.batch_size,
        lons_a=lons_lm[i_idx], lats_a=lats_lm[i_idx],
        lons_b=lons_lm[j_idx], lats_b=lats_lm[j_idx],
        geo_dist_scale=geo_dist_scale,
    )

    # Convert to dissimilarity: D = 1 - exp(-η)
    dissim_ll = 1.0 - np.exp(-eta_ll)
    D_LL = np.zeros((k, k), dtype=np.float32)
    D_LL[i_idx, j_idx] = dissim_ll
    D_LL[j_idx, i_idx] = dissim_ll

    dt = time.time() - t0
    print(f"  D_LL done ({dt:.1f}s) — median dissim: {np.median(dissim_ll):.4f}, "
          f"range: [{dissim_ll.min():.4f}, {dissim_ll.max():.4f}]")

    # 6b. D_NL: all pixels × landmarks (n × k)
    print(f"  6b. Pixel–landmark η ({n_valid:,} × {k})...")
    t0 = time.time()

    D_NL = np.zeros((n_valid, k), dtype=np.float32)
    for l in range(k):
        if l % 100 == 0:
            print(f"    Landmark {l}/{k}...", end="\r")

        # Broadcast landmark l against all pixels
        E_lm_l = np.broadcast_to(E_lm[l:l + 1], E_std.shape)
        G_lm_l = (
            np.broadcast_to(G_lm[l:l + 1], G_std.shape)
            if G_lm is not None and G_std is not None
            else None
        )

        # Broadcast landmark lon/lat for haversine distance
        lons_lm_l = np.full(n_valid, lons_lm[l], dtype=np.float32)
        lats_lm_l = np.full(n_valid, lats_lm[l], dtype=np.float32)

        eta_nl = compute_eta_pairwise(
            model, E_std, G_std, E_lm_l, G_lm_l,
            device=args.device, batch_size=args.batch_size,
            lons_a=lons_valid, lats_a=lats_valid,
            lons_b=lons_lm_l, lats_b=lats_lm_l,
            geo_dist_scale=geo_dist_scale,
        )
        D_NL[:, l] = 1.0 - np.exp(-eta_nl)

    dt = time.time() - t0
    print(f"  D_NL done ({dt:.1f}s)                              ")

    # ── 7. Classical MDS + Nyström ──────────────────────────────────
    print(f"\n[7/8] Classical MDS + Nyström extension "
          f"({args.n_components} components)...")
    scores, lm_mds, eigenvalues, var_explained, n_neg = landmark_mds(
        D_LL.astype(np.float64), D_NL.astype(np.float64),
        n_components=args.n_components,
    )

    # ── 8. RGB mapping + output ─────────────────────────────────────
    print(f"\n[8/8] Generating RGB map and outputs...")
    scores_f32 = scores.astype(np.float32)
    rgb_vals = scores_to_rgb(scores_f32[:, :3], stretch=args.stretch)

    # Write GeoTIFF (dims 1–3)
    write_rgb_geotiff(args.output, rgb_vals, mask, grid)

    # Additional RGB triplets for higher MDS dimensions
    extra_triplets = []
    m_avail = scores_f32.shape[1]
    for d_start in (3, 6):
        if d_start < m_avail:
            d_end = min(d_start + 3, m_avail)
            trip_scores = np.zeros((scores_f32.shape[0], 3), dtype=np.float32)
            trip_scores[:, :d_end - d_start] = scores_f32[:, d_start:d_end]
            trip_rgb = scores_to_rgb(trip_scores, stretch=args.stretch)
            extra_triplets.append((d_start, trip_rgb))

            suffix = f"_dims{d_start+1}-{d_start+3}"
            trip_path = str(Path(args.output).with_suffix('')) + suffix + ".tif"
            write_rgb_geotiff(trip_path, trip_rgb, mask, grid)

    # Generate PDF map
    pdf_path = Path(args.output).with_suffix(".pdf")
    shapefile_path = PROJECT_ROOT / "data" / "ibra51_reg" / "ibra51_regions.shp"

    plot_beta_map(
        rgb_vals, scores_f32, grid,
        eigenvalues, var_explained, n_neg,
        n_landmarks=k, pdf_path=pdf_path,
        shapefile_path=shapefile_path,
        epoch=ckpt["epoch"], val_loss=ckpt["val_loss"],
        lm_lons=lons_valid[lm_idx], lm_lats=lats_valid[lm_idx],
        stretch=args.stretch,
        extra_triplets=extra_triplets,
    )

    dt = time.time() - t_start
    print(f"\nDone in {dt:.0f}s ({dt / 60:.1f} min)")
    print(f"Outputs:")
    print(f"  GeoTIFF: {args.output}")
    for d_start, _ in extra_triplets:
        suffix = f"_dims{d_start+1}-{d_start+3}"
        print(f"  GeoTIFF: {Path(args.output).with_suffix('')}{suffix}.tif")
    print(f"  PDF map: {pdf_path}")


if __name__ == "__main__":
    main()
