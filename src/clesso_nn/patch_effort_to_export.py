#!/usr/bin/env python3
"""
patch_effort_to_export.py
─────────────────────────
Add effort / detectability raster values to an *existing* nn_export directory,
without touching the pairs or re-running the R pipeline.

This reads the site_covariates.feather, extracts the 6 effort raster values at
each site's (lon, lat) coordinates, appends them as new columns, and overwrites
the feather file.  metadata.json is updated with `effort_cov_cols`.

Usage
─────
    python src/clesso_nn/patch_effort_to_export.py <export_dir> [--effort-raster-dir <dir>]

Example
───────
    python src/clesso_nn/patch_effort_to_export.py \
        src/clesso_v2/output/VAS_20260310_092634/nn_export

If --effort-raster-dir is not given, it defaults to the Effort_data_preper
outputs directory.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

# ── Effort raster layer names (must have matching .flt + .hdr files) ────────
EFFORT_LAYERS = [
    "ala_record_count",
    "ala_record_smoothed",
    "dist_to_nearest_institution",
    "hub_influence_unweighted",
    "hub_influence_ecology_weighted",
    "road_density_km_per_km2",
]

DEFAULT_EFFORT_DIR = (
    "/Users/andrewhoskins/Library/Mobile Documents/"
    "com~apple~CloudDocs/CODE/Effort_data_preper/outputs"
)


# ── ESRI BIL reader ────────────────────────────────────────────────────────
def _parse_hdr(hdr_path: Path) -> dict:
    """Parse an ESRI BIL .hdr sidecar into a dict."""
    info: dict = {}
    with open(hdr_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                info[parts[0].upper()] = parts[1]
    return {
        "nrows": int(info["NROWS"]),
        "ncols": int(info["NCOLS"]),
        "ulx": float(info["ULXMAP"]),
        "uly": float(info["ULYMAP"]),
        "xdim": float(info["XDIM"]),
        "ydim": float(info["YDIM"]),
        "nodata": float(info.get("NODATA", "-3.4e+38")),
    }


def _read_bil(flt_path: Path) -> tuple[np.ndarray, dict]:
    """Read an ESRI BIL float32 raster and its grid parameters."""
    hdr_path = flt_path.with_suffix(".hdr")
    if not hdr_path.exists():
        raise FileNotFoundError(f"Header not found: {hdr_path}")
    info = _parse_hdr(hdr_path)
    data = np.fromfile(flt_path, dtype="<f4").reshape(info["nrows"], info["ncols"])
    return data, info


def _extract_at_coords(
    data: np.ndarray, info: dict, lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Extract raster values at (lon, lat) coordinates via nearest-pixel lookup."""
    cols = np.round((lons - info["ulx"]) / info["xdim"]).astype(int)
    rows = np.round((info["uly"] - lats) / info["ydim"]).astype(int)

    valid = (
        (rows >= 0)
        & (rows < info["nrows"])
        & (cols >= 0)
        & (cols < info["ncols"])
    )
    values = np.full(len(lons), np.nan, dtype=np.float32)
    values[valid] = data[rows[valid], cols[valid]]

    # Replace nodata sentinel with NaN
    nodata = info["nodata"]
    values[np.isclose(values, nodata, atol=1e30)] = np.nan

    return values


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch effort raster values into an existing nn_export directory."
    )
    parser.add_argument(
        "export_dir",
        type=Path,
        help="Path to the nn_export directory (contains site_covariates.feather)",
    )
    parser.add_argument(
        "--effort-raster-dir",
        type=Path,
        default=Path(DEFAULT_EFFORT_DIR),
        help="Directory containing the .flt effort raster files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing anything",
    )
    args = parser.parse_args()

    export_dir: Path = args.export_dir
    effort_dir: Path = args.effort_raster_dir

    # ── Validate paths ──────────────────────────────────────────────────
    sc_path = export_dir / "site_covariates.feather"
    meta_path = export_dir / "metadata.json"

    if not sc_path.exists():
        sys.exit(f"ERROR: {sc_path} not found. Is this an nn_export directory?")
    if not effort_dir.is_dir():
        sys.exit(f"ERROR: Effort raster directory not found: {effort_dir}")

    # ── Load site covariates ────────────────────────────────────────────
    sc = pf.read_feather(str(sc_path))
    n_sites = len(sc)
    print(f"Loaded {n_sites:,} sites from {sc_path.name}")
    print(f"  Existing columns: {list(sc.columns)}")

    # Check if effort columns already present
    existing_effort = [c for c in EFFORT_LAYERS if c in sc.columns]
    if existing_effort:
        print(f"\n  WARNING: {len(existing_effort)} effort column(s) already present: "
              f"{existing_effort}")
        print("  These will be OVERWRITTEN with fresh raster extractions.\n")
        # Drop them so we can re-add cleanly
        sc = sc.drop(columns=existing_effort)

    lons = sc["lon"].values.astype(np.float64)
    lats = sc["lat"].values.astype(np.float64)

    # ── Extract each effort raster ──────────────────────────────────────
    added_layers: list[str] = []
    for layer_name in EFFORT_LAYERS:
        flt_path = effort_dir / f"{layer_name}.flt"
        if not flt_path.exists():
            print(f"  SKIP: {flt_path.name} not found")
            continue

        data, info = _read_bil(flt_path)
        values = _extract_at_coords(data, info, lons, lats)

        n_valid = np.sum(~np.isnan(values))
        n_nan = np.sum(np.isnan(values))
        vmin = np.nanmin(values) if n_valid > 0 else float("nan")
        vmax = np.nanmax(values) if n_valid > 0 else float("nan")
        print(
            f"  {layer_name:40s}  valid={n_valid:,}  nan={n_nan:,}  "
            f"range=[{vmin:.4g}, {vmax:.4g}]"
        )

        sc[layer_name] = values
        added_layers.append(layer_name)

    if not added_layers:
        sys.exit("ERROR: No effort layers were extracted. Check --effort-raster-dir.")

    print(f"\nAdded {len(added_layers)} effort columns to site_covariates")
    print(f"  Final columns ({len(sc.columns)}): {list(sc.columns)}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # ── Write updated site_covariates.feather ───────────────────────────
    pf.write_feather(sc, str(sc_path))
    print(f"\n  Wrote {sc_path}")

    # ── Update metadata.json ────────────────────────────────────────────
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["effort_cov_cols"] = added_layers

    # Also update alpha_cov_cols to exclude effort columns (if present)
    if "alpha_cov_cols" in meta:
        meta["alpha_cov_cols"] = [
            c for c in meta["alpha_cov_cols"] if c not in set(added_layers)
        ]

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path} (effort_cov_cols: {added_layers})")

    print("\nDone! You can now train with effort features using this export directory.")
    print(f"  python src/clesso_nn/run_clesso_nn.py --export-dir {export_dir}")


if __name__ == "__main__":
    main()
