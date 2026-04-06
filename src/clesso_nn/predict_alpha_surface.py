#!/usr/bin/env python3
"""
predict_alpha_surface.py
========================

Predict species richness (alpha) across all of Australia and write a GeoTIFF.

Uses the trained CLESSO NN model checkpoint together with:
  - Substrate PCA raster  (6 bands from SUBS_brk_VAS.grd)
  - AWAP climate grids    (9 variables from geonpy .npy files)
  - Grid coordinates      (lon, lat derived from reference raster)

The alpha covariates are determined dynamically from the checkpoint's
alpha_cov_names — no hard-coded variable list. Climate variables are
derived using the naming convention: column "{cstat}_{varname}" maps
to geonpy file "{varname}_191101-201712.npy".

The output is a single-band GeoTIFF at the same resolution and extent
as the reference raster (0.05°, 670×813, WGS84).

Usage:
    python predict_alpha_surface.py                         # defaults
    python predict_alpha_surface.py --checkpoint path.pt    # custom checkpoint
    python predict_alpha_surface.py --output alpha.tif      # custom output path

Requirements:
    torch, numpy, rasterio, geonpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
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
    batch_size=50_000,
    climate_year=2010,
    climate_month=6,
    climate_window=30,  # years
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


def derive_climate_vars(alpha_cov_names: list[str]) -> list[tuple[str, str, callable, callable]]:
    """Derive climate extraction specs from the checkpoint's alpha_cov_names.

    Convention (matching run_clesso.R):
        column name  = "{cstat}_{varname}"
        npy file     = "{varname}_{NPY_DATE_RANGE}.npy"
        mstat        = np.mean  (always)
        cstat        = np.mean / np.min / np.max  (from prefix)

    Non-climate names (lon, lat, subs_*, fourier_*) are skipped.

    Returns: list of (col_name, npy_basename, mstat_func, cstat_func)
    """
    skip_prefixes = ("lon", "lat", "subs_", "fourier_")
    specs = []
    for col in alpha_cov_names:
        if any(col == pfx or col.startswith(pfx) for pfx in skip_prefixes):
            continue
        # Split on first underscore: cstat_prefix, varname
        parts = col.split("_", 1)
        if len(parts) != 2 or parts[0] not in CSTAT_FUNCS:
            raise ValueError(
                f"Cannot derive climate extraction for column '{col}'. "
                f"Expected format '{{mean|min|max}}_{{varname}}'.")
        cstat_prefix, varname = parts
        mstat_func, cstat_func = CSTAT_FUNCS[cstat_prefix]
        npy_basename = f"{varname}_{NPY_DATE_RANGE}"
        specs.append((col, npy_basename, mstat_func, cstat_func))
    return specs


# ──────────────────────────────────────────────────────────────────────────
# Step 1: Load reference raster → land mask + coordinate grids
# ──────────────────────────────────────────────────────────────────────────

def load_reference_grid(path: str | Path) -> dict:
    """Load reference raster and return grid metadata + land mask."""
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        height, width = src.height, src.width
        nodata = src.nodata

    # Land mask: valid where not nodata and not NaN
    if nodata is not None:
        mask = (data != nodata) & ~np.isnan(data)
    else:
        mask = ~np.isnan(data)

    # Build lon/lat grids (cell centres)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    lons = transform.c + (cols + 0.5) * transform.a  # x origin + col * pixel_width
    lats = transform.f + (rows + 0.5) * transform.e  # y origin + row * pixel_height (negative)

    n_valid = int(mask.sum())
    print(f"  Reference grid: {height}×{width}, {n_valid:,} land cells "
          f"({100 * n_valid / data.size:.1f}%)")
    print(f"  Extent: {lons[mask].min():.3f}–{lons[mask].max():.3f}°E, "
          f"{lats[mask].max():.3f}–{lats[mask].min():.3f}°S")

    return dict(
        height=height, width=width, transform=transform,
        mask=mask, lons=lons, lats=lats, n_valid=n_valid,
    )


# ──────────────────────────────────────────────────────────────────────────
# Step 2: Extract substrate PCA (6 bands)
# ──────────────────────────────────────────────────────────────────────────

def extract_substrate(path: str | Path, mask: np.ndarray) -> np.ndarray:
    """Read 6-band substrate raster and extract values at land cells.

    Returns: (n_valid, 6) float32 array.
    """
    with rasterio.open(path) as src:
        n_bands = src.count
        print(f"  Substrate raster: {n_bands} bands")
        all_bands = src.read()  # (bands, height, width)
        nodata = src.nodata

    # Stack to (height, width, bands)
    cube = np.moveaxis(all_bands, 0, -1).astype(np.float32)

    # Extract at valid cells
    vals = cube[mask]  # (n_valid, 6)

    # Replace nodata / extreme values with NaN
    if nodata is not None:
        vals[vals == nodata] = np.nan

    n_nan = np.isnan(vals).any(axis=1).sum()
    if n_nan > 0:
        print(f"  Warning: {n_nan} land cells have NaN substrate values")

    return vals


# ──────────────────────────────────────────────────────────────────────────
# Step 3: Extract climate variables from geonpy
# ──────────────────────────────────────────────────────────────────────────

def extract_climate(npy_src: str | Path, lons: np.ndarray, lats: np.ndarray,
                    climate_specs: list[tuple[str, str, callable, callable]],
                    climate_year: int, climate_month: int,
                    climate_window: int, start_year: int) -> dict[str, np.ndarray]:
    """Extract climate variables at land cell coordinates.

    Uses geonpy read_points + calc_climatology_window — identical to pyper.py
    so that the extraction exactly matches how training data was produced.

    Args:
        npy_src:         directory containing geonpy .npy files
        lons, lats:      (n_valid,) coordinate arrays of land cells
        climate_specs:   list of (col_name, npy_basename, mstat_func, cstat_func)
                         as returned by derive_climate_vars()
        climate_year:    centre year for climate window (2010)
        climate_month:   centre month (6)
        climate_window:  window length in years (30)
        start_year:      geonpy temporal origin year (1911)

    Returns: dict mapping column name → (n_valid,) float32 array
    """
    from geonpy.geonpy import Geonpy, calc_climatology_window, gen_multi_index_slice

    npy_src = Path(npy_src)
    n_pts = len(lons)

    # Build coordinate array (lon, lat) for read_points
    pts = np.column_stack([lons, lats])

    # Build time window indices — same for all cells
    # gen_multi_index_slice expects (n_pts, 2) with [year, month]
    year_mon = np.array([[climate_year, climate_month]])  # single row
    window_months = climate_window * 12
    dim_idx = gen_multi_index_slice(year_mon, window_months, st_year=start_year)
    # dim_idx shape: (1, 360) — broadcast to all points
    dim_idx = np.broadcast_to(dim_idx, (n_pts, window_months))

    print(f"  Climate window: {climate_window} years ending "
          f"{climate_year}/{climate_month:02d}")
    print(f"  Window months: {window_months}, index range: "
          f"{dim_idx[0, 0]}–{dim_idx[0, -1]}")
    print(f"  Points: {n_pts:,}")

    results = {}
    n_vars = len(climate_specs)

    for i, (col_name, npy_name, mstat, cstat) in enumerate(climate_specs, 1):
        npy_path = npy_src / f"{npy_name}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Climate file not found: {npy_path}")

        print(f"    [{i}/{n_vars}] {col_name} ← {npy_name}", end="  ")
        t0 = time.time()

        g = Geonpy(str(npy_path))

        # Extract raw time-series at each point: (n_pts, 360)
        raw = g.read_points(pts, dim_idx=dim_idx)

        # Compute climatology: monthly stat → yearly stat
        vals = calc_climatology_window(raw, mstat, cstat)
        results[col_name] = vals.astype(np.float32)

        dt = time.time() - t0
        print(f"range: {np.nanmin(vals):.4f} – {np.nanmax(vals):.4f}  "
              f"({dt:.1f}s)")

        del g  # release memory-mapped file

    return results


# ──────────────────────────────────────────────────────────────────────────
# Step 3b: Extract effort rasters (ESRI BIL float32)
# ──────────────────────────────────────────────────────────────────────────

def extract_effort_rasters(
    effort_dir: str | Path,
    effort_names: list[str],
    mask: np.ndarray,
) -> np.ndarray:
    """Read effort rasters (.flt ESRI BIL) and extract at land cells.

    Each raster is expected to be a single-band .flt file with matching .hdr,
    at the same resolution/extent as the reference grid.

    Returns: (n_valid, K_effort) float32 array
    """
    effort_dir = Path(effort_dir)
    cols = []

    for name in effort_names:
        flt_path = effort_dir / f"{name}.flt"
        if not flt_path.exists():
            # Try .tif too
            flt_path = effort_dir / f"{name}.tif"
        if not flt_path.exists():
            raise FileNotFoundError(
                f"Effort raster not found: {effort_dir / name}.flt (or .tif)")

        with rasterio.open(flt_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata

        if nodata is not None:
            data[data == nodata] = np.nan

        vals = data[mask]
        cols.append(vals)
        n_nan = np.isnan(vals).sum()
        print(f"    {name}: range [{np.nanmin(vals):.3g}, {np.nanmax(vals):.3g}], "
              f"NaN: {n_nan}")

    return np.column_stack(cols)


# ──────────────────────────────────────────────────────────────────────────
# Step 4: Load model and predict
# ──────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    """Load trained CLESSO NN model from checkpoint."""
    # Import model class
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from clesso_nn.model import CLESSONet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Infer beta_no_intercept from state_dict if not saved in config
    # (for checkpoints saved before this flag was added)
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


@torch.no_grad()
def predict_alpha_batched(model, Z: np.ndarray, batch_size: int,
                          device: str = "cpu",
                          W: np.ndarray | None = None,
                          mode: str = "env_only") -> np.ndarray:
    """Predict alpha in batches to manage memory.

    Args:
        Z: (n_cells, K_alpha) standardised covariate array
        W: (n_cells, K_effort) standardised effort array (or None)
        mode: "env_only" (true richness), "full" (env+effort),
              "effort_only" (effort logit component)

    Returns: (n_cells,) alpha predictions
    """
    n = Z.shape[0]
    alpha = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Z_batch = torch.from_numpy(Z[start:end]).to(device)
        W_batch = None
        if W is not None:
            W_batch = torch.from_numpy(W[start:end]).to(device)

        if mode == "env_only":
            alpha[start:end] = model._compute_alpha_env_only(Z_batch).cpu().numpy()
        elif mode == "full":
            alpha[start:end] = model._compute_alpha(Z_batch, W_batch).cpu().numpy()
        elif mode == "effort_only":
            if model.effort_net is not None and W_batch is not None:
                alpha[start:end] = model.effort_net(W_batch).squeeze(-1).cpu().numpy()
            else:
                alpha[start:end] = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        if (start // batch_size) % 10 == 0:
            pct = 100 * end / n
            print(f"    Predicted {end:,}/{n:,} cells ({pct:.0f}%)", end="\r")

    print(f"    Predicted {n:,}/{n:,} cells (100%)    ")
    return alpha


# ──────────────────────────────────────────────────────────────────────────
# Step 5: Write GeoTIFF output
# ──────────────────────────────────────────────────────────────────────────

def write_geotiff(output_path: str | Path, alpha: np.ndarray,
                  grid: dict, nodata: float = -9999.0):
    """Write alpha predictions as a single-band GeoTIFF."""
    height, width = grid["height"], grid["width"]
    mask = grid["mask"]

    # Create output raster (start with nodata everywhere)
    out = np.full((height, width), nodata, dtype=np.float32)
    out[mask] = alpha

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(
            DESCRIPTION="CLESSO NN predicted species richness (alpha)",
            MODEL="CLESSONet VAS",
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Written: {output_path} ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────
# Step 6a: Generate covariate diagnostic PDF
# ──────────────────────────────────────────────────────────────────────────

def plot_covariate_maps(covariate_arrays: dict, cov_names: list[str],
                        grid: dict, pdf_path: str | Path,
                        shapefile_path: str | Path | None = None,
                        ncols: int = 3):
    """Produce a multi-page PDF with a map of every env covariate.

    Each page shows up to ncols × 3 panels (default 3×3 = 9 per page).
    Each panel shows the raw (unstandardised) covariate on a diverging or
    sequential colour scale, with summary stats in the title.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize

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

    # Read IBRA boundaries if available
    boundary_shapes = None
    if shapefile_path is not None:
        shapefile_path = Path(shapefile_path)
        if shapefile_path.exists():
            try:
                import shapefile as shp
                sf = shp.Reader(str(shapefile_path))
                boundary_shapes = sf.shapes()
            except Exception:
                pass

    def _draw_boundaries(ax):
        if boundary_shapes is None:
            return
        for shape in boundary_shapes:
            pts = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(parts) - 1):
                seg = pts[parts[k]:parts[k + 1]]
                ax.plot(seg[:, 0], seg[:, 1], color="#444444",
                        linewidth=0.25, alpha=0.5)

    nrows_per_page = 3
    panels_per_page = nrows_per_page * ncols
    n_covs = len(cov_names)
    n_pages = math.ceil(n_covs / panels_per_page)

    print(f"  Covariate diagnostic PDF: {n_covs} covariates, "
          f"{n_pages} page(s) → {pdf_path.name}")

    with PdfPages(str(pdf_path)) as pdf:
        for page_idx in range(n_pages):
            start = page_idx * panels_per_page
            end = min(start + panels_per_page, n_covs)
            page_names = cov_names[start:end]
            n_on_page = len(page_names)
            n_rows = math.ceil(n_on_page / ncols)

            fig, axes = plt.subplots(
                n_rows, ncols,
                figsize=(5.5 * ncols, 4.5 * n_rows),
                squeeze=False,
            )
            fig.suptitle(
                f"CLESSO NN — Input Covariates (page {page_idx + 1}/{n_pages})",
                fontsize=14, fontweight="bold", y=0.995,
            )

            for k, name in enumerate(page_names):
                row, col = divmod(k, ncols)
                ax = axes[row, col]

                vals_1d = covariate_arrays[name]
                arr_2d = np.full((height, width), np.nan, dtype=np.float32)
                arr_2d[mask] = vals_1d

                valid = vals_1d[~np.isnan(vals_1d)]
                if len(valid) == 0:
                    ax.set_title(f"{name}\n(all NaN)", fontsize=9)
                    ax.axis("off")
                    continue

                p1, p99 = np.nanpercentile(valid, [1, 99])
                vmin, vmax = p1, p99
                if vmin == vmax:
                    vmin -= 1
                    vmax += 1

                # Use diverging cmap if data spans zero, sequential otherwise
                if vmin < 0 < vmax:
                    abs_v = max(abs(vmin), abs(vmax))
                    cmap = plt.cm.RdBu_r.copy()
                    vmin, vmax = -abs_v, abs_v
                else:
                    cmap = plt.cm.viridis.copy()
                cmap.set_bad(color="#e8e8e8")

                im = ax.imshow(
                    arr_2d, extent=extent, origin="upper",
                    cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                    interpolation="nearest", aspect="equal",
                )
                _draw_boundaries(ax)
                fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

                n_nan = np.isnan(vals_1d).sum()
                title = (f"{name}\n"
                         f"mean={np.nanmean(valid):.2f}  "
                         f"sd={np.nanstd(valid):.2f}  "
                         f"NaN={n_nan:,}")
                ax.set_title(title, fontsize=8)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=6)

            # Turn off unused panels
            for k in range(n_on_page, n_rows * ncols):
                row, col = divmod(k, ncols)
                axes[row, col].axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"  Written: {pdf_path}")


# ──────────────────────────────────────────────────────────────────────────
# Step 6b: Generate PDF map
# ──────────────────────────────────────────────────────────────────────────

def plot_alpha_map(alpha_raster: np.ndarray, grid: dict,
                   pdf_path: str | Path, nodata: float = -9999.0,
                   shapefile_path: str | Path | None = None,
                   epoch: int = 0, val_loss: float = 0.0,
                   alpha_full: np.ndarray | None = None,
                   alpha_effort: np.ndarray | None = None):
    """Produce a publication-quality PDF map of the alpha surface.

    Includes:
      - Main map with log-scaled colour ramp
      - IBRA bioregion boundaries (if shapefile provided)
      - Histogram of alpha distribution
      - Summary statistics
      - (If effort present) Full alpha map, effort logit map, difference map
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize, BoundaryNorm
    from matplotlib.cm import ScalarMappable
    import matplotlib.ticker as mticker
    import matplotlib.patheffects as pe

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = grid["height"], grid["width"]
    mask = grid["mask"]
    transform = grid["transform"]

    # Build 2D alpha raster
    alpha_2d = np.full((height, width), np.nan, dtype=np.float32)
    alpha_flat = np.full(int(mask.sum()), np.nan, dtype=np.float32)
    alpha_flat[:] = alpha_raster
    alpha_2d[mask] = alpha_flat

    # Geographic extent (pixel edges, not centres)
    x_min = transform.c
    y_max = transform.f
    x_max = x_min + width * transform.a
    y_min = y_max + height * transform.e   # e is negative
    extent = [x_min, x_max, y_min, y_max]

    # Valid alpha values for statistics
    valid = alpha_flat[~np.isnan(alpha_flat)]

    # ── Colour scale ─────────────────────────────────────────────────
    # Use a linear colour map truncated at sensible bounds
    vmin, vmax = 1.0, max(np.nanpercentile(valid, 99.5), 10.0)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e8e8e8")  # light grey for ocean / nodata

    # ── Read IBRA boundary shapefile ─────────────────────────────────
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

    with PdfPages(str(pdf_path)) as pdf:
        # ─────────────────────────────────────────────────────────────
        # Page 1: Main map
        # ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(
            alpha_2d,
            extent=extent,
            origin="upper",
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            interpolation="nearest",
            aspect="equal",
        )

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
                                color="#333333", linewidth=0.25, alpha=0.5)

        # Colourbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=30)
        cbar.set_label("Predicted species richness (α)", fontsize=11)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Longitude (°E)", fontsize=11)
        ax.set_ylabel("Latitude (°S)", fontsize=11)
        ax.set_title(
            f"CLESSO NN — Predicted Vascular Plant Richness (α)\n"
            f"Epoch {epoch} | Val loss {val_loss:.4f} | "
            f"{len(valid):,} cells @ 0.05°",
            fontsize=13, fontweight="bold",
        )

        # Grid lines
        ax.grid(True, linewidth=0.3, alpha=0.4, color="grey")

        fig.tight_layout()
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────────
        # Page 2: Distribution + summary stats
        # ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                 gridspec_kw={"width_ratios": [2, 1]})

        # Left: histogram
        ax_hist = axes[0]
        bins = np.linspace(max(valid.min(), 1.0), valid.max(), 80)
        ax_hist.hist(valid, bins=bins, color="steelblue", edgecolor="white",
                     linewidth=0.3, alpha=0.9)
        ax_hist.set_xlabel("Predicted α (species richness)", fontsize=11)
        ax_hist.set_ylabel("Number of grid cells", fontsize=11)
        ax_hist.set_title("Distribution of predicted α", fontsize=12,
                          fontweight="bold")
        ax_hist.axvline(np.median(valid), color="red", linestyle="--",
                        linewidth=1.2, label=f"median = {np.median(valid):.2f}")
        ax_hist.axvline(np.mean(valid), color="orange", linestyle="-.",
                        linewidth=1.2, label=f"mean = {np.mean(valid):.2f}")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, axis="y", linewidth=0.3, alpha=0.4)

        # Right: summary statistics table
        ax_tab = axes[1]
        ax_tab.axis("off")
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        rows = [
            ["Total land cells", f"{len(valid):,}"],
            ["Min", f"{valid.min():.2f}"],
        ]
        for p in pcts:
            rows.append([f"{p}th percentile", f"{np.percentile(valid, p):.2f}"])
        rows += [
            ["Max", f"{valid.max():.2f}"],
            ["Mean", f"{np.mean(valid):.2f}"],
            ["Std dev", f"{np.std(valid):.2f}"],
            ["", ""],
            ["Cells with α > 5",
             f"{(valid > 5).sum():,} ({100 * (valid > 5).mean():.1f}%)"],
            ["Cells with α > 10",
             f"{(valid > 10).sum():,} ({100 * (valid > 10).mean():.1f}%)"],
        ]
        table = ax_tab.table(
            cellText=rows,
            colLabels=["Statistic", "Value"],
            loc="center",
            cellLoc="left",
            colWidths=[0.55, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        # Style header row
        for j in range(2):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        fig.suptitle("Alpha Surface — Summary Statistics", fontsize=13,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────────
        # Pages 3–5: Effort decomposition (if effort surfaces provided)
        # ─────────────────────────────────────────────────────────────
        if alpha_full is not None or alpha_effort is not None:
            print("  Adding effort pages to PDF...")

            # Helper: build 2D raster from flat valid-pixel array
            def _to_2d(flat_arr):
                arr_2d = np.full((height, width), np.nan, dtype=np.float32)
                arr_2d[mask] = flat_arr
                return arr_2d

            # Helper: draw IBRA boundaries on an axes
            def _draw_boundaries(ax):
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
                                        color="#333333", linewidth=0.25, alpha=0.5)

            # ── Page 3: Full alpha (env + effort) map ────────────────
            if alpha_full is not None:
                full_2d = _to_2d(alpha_full)
                full_valid = alpha_full[~np.isnan(alpha_full)]

                fig, ax = plt.subplots(figsize=(12, 10))
                im = ax.imshow(
                    full_2d, extent=extent, origin="upper",
                    cmap=cmap,
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    interpolation="nearest", aspect="equal",
                )
                _draw_boundaries(ax)
                cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=30)
                cbar.set_label("Predicted α (env + effort)", fontsize=11)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("Longitude (°E)", fontsize=11)
                ax.set_ylabel("Latitude (°S)", fontsize=11)
                ax.set_title(
                    f"CLESSO NN — α (env + effort)\n"
                    f"Mean {np.nanmean(full_valid):.1f} | "
                    f"Median {np.nanmedian(full_valid):.1f} | "
                    f"Range [{np.nanmin(full_valid):.1f}, {np.nanmax(full_valid):.1f}]",
                    fontsize=13, fontweight="bold",
                )
                ax.grid(True, linewidth=0.3, alpha=0.4, color="grey")
                fig.tight_layout()
                pdf.savefig(fig, dpi=200)
                plt.close(fig)

            # ── Page 4: Effort logit map ─────────────────────────────
            if alpha_effort is not None:
                eff_2d = _to_2d(alpha_effort)
                eff_valid = alpha_effort[~np.isnan(alpha_effort)]

                # Diverging colour map centred on 0
                eff_abs = max(abs(np.nanpercentile(eff_valid, 1)),
                              abs(np.nanpercentile(eff_valid, 99)))
                eff_abs = max(eff_abs, 0.01)  # avoid zero range
                eff_cmap = plt.cm.RdBu_r.copy()
                eff_cmap.set_bad(color="#e8e8e8")

                fig, ax = plt.subplots(figsize=(12, 10))
                im = ax.imshow(
                    eff_2d, extent=extent, origin="upper",
                    cmap=eff_cmap,
                    norm=Normalize(vmin=-eff_abs, vmax=eff_abs),
                    interpolation="nearest", aspect="equal",
                )
                _draw_boundaries(ax)
                cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=30)
                cbar.set_label("Effort logit (EffortNet output)", fontsize=11)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("Longitude (°E)", fontsize=11)
                ax.set_ylabel("Latitude (°S)", fontsize=11)
                ax.set_title(
                    f"CLESSO NN — Effort Logit Surface\n"
                    f"Mean {np.nanmean(eff_valid):.3f} | "
                    f"SD {np.nanstd(eff_valid):.3f} | "
                    f"Range [{np.nanmin(eff_valid):.3f}, {np.nanmax(eff_valid):.3f}]",
                    fontsize=13, fontweight="bold",
                )
                ax.grid(True, linewidth=0.3, alpha=0.4, color="grey")
                fig.tight_layout()
                pdf.savefig(fig, dpi=200)
                plt.close(fig)

            # ── Page 5: Side-by-side env-only vs full + histogram ────
            if alpha_full is not None and alpha_effort is not None:
                fig, axes = plt.subplots(2, 2, figsize=(16, 14))

                # Top-left: env-only alpha
                ax = axes[0, 0]
                im = ax.imshow(
                    alpha_2d, extent=extent, origin="upper",
                    cmap=cmap,
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    interpolation="nearest", aspect="equal",
                )
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
                ax.set_title("α env-only (true richness)", fontsize=11, fontweight="bold")
                ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                ax.set_aspect("equal")

                # Top-right: full alpha (env + effort)
                ax = axes[0, 1]
                full_2d = _to_2d(alpha_full)
                im = ax.imshow(
                    full_2d, extent=extent, origin="upper",
                    cmap=cmap,
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    interpolation="nearest", aspect="equal",
                )
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
                ax.set_title("α full (env + effort)", fontsize=11, fontweight="bold")
                ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                ax.set_aspect("equal")

                # Bottom-left: effort logit
                ax = axes[1, 0]
                eff_2d = _to_2d(alpha_effort)
                eff_valid = alpha_effort[~np.isnan(alpha_effort)]
                eff_abs_v = max(abs(np.nanpercentile(eff_valid, 1)),
                                abs(np.nanpercentile(eff_valid, 99)), 0.01)
                eff_cmap2 = plt.cm.RdBu_r.copy()
                eff_cmap2.set_bad(color="#e8e8e8")
                im = ax.imshow(
                    eff_2d, extent=extent, origin="upper",
                    cmap=eff_cmap2,
                    norm=Normalize(vmin=-eff_abs_v, vmax=eff_abs_v),
                    interpolation="nearest", aspect="equal",
                )
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
                ax.set_title("Effort logit", fontsize=11, fontweight="bold")
                ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                ax.set_aspect("equal")

                # Bottom-right: difference map (full − env-only), own colour scale
                ax = axes[1, 1]
                diff_arr = alpha_full - alpha_raster  # absolute difference
                diff_2d = _to_2d(diff_arr)
                diff_valid = diff_arr[~np.isnan(diff_arr)]
                diff_abs_v = max(abs(np.nanpercentile(diff_valid, 1)),
                                 abs(np.nanpercentile(diff_valid, 99)), 0.01)
                diff_cmap = plt.cm.PuOr_r.copy()
                diff_cmap.set_bad(color="#e8e8e8")
                im = ax.imshow(
                    diff_2d, extent=extent, origin="upper",
                    cmap=diff_cmap,
                    norm=Normalize(vmin=-diff_abs_v, vmax=diff_abs_v),
                    interpolation="nearest", aspect="equal",
                )
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
                diff_mean = np.nanmean(diff_valid)
                diff_std  = np.nanstd(diff_valid)
                ax.set_title(
                    f"Δα (full − env-only)  μ={diff_mean:+.1f}  σ={diff_std:.1f}",
                    fontsize=10, fontweight="bold",
                )
                ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                ax.set_aspect("equal")

                fig.suptitle(
                    f"CLESSO NN — Effort Decomposition (Epoch {epoch})",
                    fontsize=14, fontweight="bold",
                )
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, dpi=200)
                plt.close(fig)

    print(f"  Written: {pdf_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict species richness (alpha) surface for Australia"
    )
    parser.add_argument("--checkpoint", type=str,
                        default=str(DEFAULTS["checkpoint"]),
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--reference-raster", type=str,
                        default=str(DEFAULTS["reference_raster"]),
                        help="Reference raster defining the grid")
    parser.add_argument("--substrate-raster", type=str,
                        default=str(DEFAULTS["substrate_raster"]),
                        help="Substrate PCA raster (6 bands)")
    parser.add_argument("--npy-src", type=str,
                        default=str(DEFAULTS["npy_src"]),
                        help="Directory containing geonpy .npy climate files")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Output GeoTIFF path (default: <checkpoint_dir>/alpha_surface.tif)")
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULTS["batch_size"],
                        help="Prediction batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu, mps, cuda)")
    parser.add_argument("--effort-input-dir", type=str, default=None,
                        help="Directory containing effort INPUT rasters (.flt). "
                             "Output surfaces are written alongside --output.")
    args = parser.parse_args()

    t_start = time.time()

    # Derive output path from checkpoint dir if not explicitly provided
    if args.output is None:
        args.output = str(Path(args.checkpoint).parent / "alpha_surface.tif")

    print("=" * 70)
    print("  CLESSO NN — Alpha Surface Prediction")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────
    print("\n[1/5] Loading model checkpoint...")
    model, ckpt = load_model(args.checkpoint, args.device)
    stats = ckpt["site_data_stats"]
    cov_names = stats["alpha_cov_names"]
    z_mean = np.array(stats["z_mean"], dtype=np.float32)
    z_std = np.array(stats["z_std"], dtype=np.float32)
    print(f"  Model epoch: {ckpt['epoch']}")
    print(f"  K_alpha: {ckpt['config']['K_alpha']} covariates: {cov_names}")
    print(f"  Val loss: {ckpt['val_loss']:.4f}")

    # ── Load reference grid ──────────────────────────────────────────
    print("\n[2/5] Loading reference grid...")
    grid = load_reference_grid(args.reference_raster)
    mask = grid["mask"]
    n_valid = grid["n_valid"]

    # ── Extract covariates ───────────────────────────────────────────
    print("\n[3/5] Extracting covariates...")

    # Coordinates
    lons_valid = grid["lons"][mask].astype(np.float32)
    lats_valid = grid["lats"][mask].astype(np.float32)
    print(f"  Coordinates: lon [{lons_valid.min():.2f}, {lons_valid.max():.2f}], "
          f"lat [{lats_valid.min():.2f}, {lats_valid.max():.2f}]")

    # Substrate PCA
    subs_vals = extract_substrate(args.substrate_raster, mask)

    # Derive which climate variables to extract from the checkpoint
    climate_specs = derive_climate_vars(cov_names)
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

    # ── Assemble covariate matrix ────────────────────────────────────
    print("\n[4/5] Assembling covariates and predicting alpha...")

    # Build in the EXACT order the model expects (matching alpha_cov_names)
    covariate_arrays = {
        "lon": lons_valid,
        "lat": lats_valid,
    }
    for i in range(subs_vals.shape[1]):
        covariate_arrays[f"subs_{i + 1}"] = subs_vals[:, i]
    covariate_arrays.update(climate)

    # Add Fourier features if the model was trained with them
    fourier_n_freq = stats.get("fourier_n_frequencies", 0)
    fourier_max_wl = stats.get("fourier_max_wavelength", 40.0)
    if fourier_n_freq and fourier_n_freq > 0:
        from clesso_nn.dataset import compute_fourier_features
        ff, ff_names = compute_fourier_features(
            lons_valid, lats_valid, fourier_n_freq, fourier_max_wl)
        for k, name in enumerate(ff_names):
            covariate_arrays[name] = ff[:, k]
        print(f"  Fourier features: {len(ff_names)} "
              f"({fourier_n_freq} freqs, max λ={fourier_max_wl}°)")

    # Stack in training order
    Z_raw = np.column_stack([covariate_arrays[name] for name in cov_names])
    print(f"  Covariate matrix: {Z_raw.shape} (cells × features)")

    # Check for NaNs before standardisation
    nan_mask = np.isnan(Z_raw).any(axis=1)
    n_nan_cells = nan_mask.sum()
    if n_nan_cells > 0:
        print(f"  Warning: {n_nan_cells:,} cells have NaN covariates "
              f"({100 * n_nan_cells / n_valid:.1f}%) — will predict as nodata")

    # Standardise using training statistics
    Z = (Z_raw - z_mean) / z_std
    np.nan_to_num(Z, copy=False, nan=0.0)

    # ── Extract effort rasters (if model has effort_net) ─────────────
    W = None
    effort_names = stats.get("effort_cov_names", [])
    has_effort = model.effort_net is not None and len(effort_names) > 0
    if has_effort:
        effort_dir = args.effort_input_dir

        # Auto-detect effort input dir if not explicitly supplied
        if not effort_dir:
            _search_paths = [
                Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"
                    "/CODE/Effort_data_preper/outputs",
                PROJECT_ROOT / "data" / "effort",
            ]
            for candidate in _search_paths:
                if candidate.is_dir():
                    effort_dir = str(candidate)
                    print(f"  [effort] Auto-detected input dir: {effort_dir}")
                    break

        if effort_dir:
            print(f"  Extracting {len(effort_names)} effort rasters...")
            W_raw = extract_effort_rasters(effort_dir, effort_names, mask)
            w_mean = np.array(stats["w_mean"], dtype=np.float32)
            w_std = np.array(stats["w_std"], dtype=np.float32)
            W = (W_raw - w_mean) / w_std
            np.nan_to_num(W, copy=False, nan=0.0)
            print(f"  Effort matrix: {W.shape} (cells × effort features)")
        else:
            print("  WARNING: Model has effort_net but no effort input dir "
                  "found. Effort component will be zeroed.")

    # ── Predict ──────────────────────────────────────────────────────
    # Always produce env_only surface (true richness)
    print("  Predicting env-only alpha (true richness)...")
    alpha = predict_alpha_batched(model, Z, args.batch_size, args.device,
                                  mode="env_only")

    # Mark NaN-covariate cells as NaN in output
    if n_nan_cells > 0:
        alpha[nan_mask] = np.nan

    print(f"  Alpha (env-only) range: {np.nanmin(alpha):.2f} – {np.nanmax(alpha):.2f}")
    print(f"  Alpha (env-only) mean:  {np.nanmean(alpha):.2f}")
    print(f"  Alpha (env-only) median: {np.nanmedian(alpha):.2f}")

    # If effort available, also produce full and effort-only surfaces
    alpha_full = None
    alpha_effort = None
    if has_effort and W is not None:
        print("  Predicting full alpha (env + effort)...")
        alpha_full = predict_alpha_batched(model, Z, args.batch_size,
                                           args.device, W=W, mode="full")
        if n_nan_cells > 0:
            alpha_full[nan_mask] = np.nan
        print(f"  Alpha (full) range: {np.nanmin(alpha_full):.2f} – "
              f"{np.nanmax(alpha_full):.2f}")

        print("  Predicting effort logit component...")
        alpha_effort = predict_alpha_batched(model, Z, args.batch_size,
                                             args.device, W=W, mode="effort_only")
        if n_nan_cells > 0:
            alpha_effort[nan_mask] = np.nan
        print(f"  Effort logit range: {np.nanmin(alpha_effort):.3f} – "
              f"{np.nanmax(alpha_effort):.3f}")

    # ── Write output ─────────────────────────────────────────────────
    print("\n[5/6] Writing GeoTIFF(s)...")
    output_path = Path(args.output)
    write_geotiff(output_path, alpha, grid)

    if alpha_full is not None:
        full_path = output_path.with_stem(output_path.stem + "_full")
        write_geotiff(full_path, alpha_full, grid)

    if alpha_effort is not None:
        effort_path = output_path.with_stem(output_path.stem + "_effort_logit")
        write_geotiff(effort_path, alpha_effort, grid)

    # ── Generate PDF map ─────────────────────────────────────────────
    pdf_path = output_path.with_suffix(".pdf")
    print("\n[6/7] Generating PDF map...")
    shapefile_path = PROJECT_ROOT / "data" / "ibra51_reg" / "ibra51_regions.shp"
    plot_alpha_map(
        alpha, grid, pdf_path,
        shapefile_path=shapefile_path,
        epoch=ckpt["epoch"],
        val_loss=ckpt["val_loss"],
        alpha_full=alpha_full,
        alpha_effort=alpha_effort,
    )

    # ── Generate covariate diagnostic PDF ────────────────────────────
    cov_pdf_path = output_path.with_stem(output_path.stem + "_covariates").with_suffix(".pdf")
    print("\n[7/7] Generating covariate diagnostic PDF...")
    plot_covariate_maps(
        covariate_arrays, cov_names, grid, cov_pdf_path,
        shapefile_path=shapefile_path,
    )

    dt = time.time() - t_start
    print(f"\nDone in {dt:.0f}s ({dt / 60:.1f} min)")
    print(f"Outputs:")
    print(f"  GeoTIFF (env-only): {output_path}")
    if alpha_full is not None:
        print(f"  GeoTIFF (full):     {full_path}")
    if alpha_effort is not None:
        print(f"  GeoTIFF (effort):   {effort_path}")
    print(f"  PDF map: {pdf_path}")
    print(f"  PDF covariates: {cov_pdf_path}")


if __name__ == "__main__":
    main()
