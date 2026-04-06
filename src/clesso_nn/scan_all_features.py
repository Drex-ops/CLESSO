#!/usr/bin/env python3
"""
scan_all_features.py — Add all available geonpy climate features to nn_export,
then run feature selection to identify the optimal subset.

This is a one-shot driver script that:
    1. Backs up current nn_export feather files
    2. Discovers all standard geonpy variables not yet in the export
    3. Extracts them for all sites (via pyper.py) and adds to feather files
    4. Runs the full feature selection report

Usage:
    # Full scan: add all missing variables (mstat=mean, cstat=mean) + report
    python scan_all_features.py

    # Add with different cstat (e.g. max)
    python scan_all_features.py --cstat max

    # Add multiple cstat variants (comprehensive but slow)
    python scan_all_features.py --cstat mean max min

    # Dry run: show what would be added without extracting
    python scan_all_features.py --dry-run

    # Skip feature selection (just add the features)
    python scan_all_features.py --no-select

    # Custom sample size for feature selection
    python scan_all_features.py --n-sample 100000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

# Import from sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manage_env_features import (
    PROJECT_ROOT,
    DEFAULT_EXPORT_DIR,
    DEFAULT_NPY_SRC,
    PYPER_SCRIPT,
    DEFAULT_WINDOW,
    discover_npy_variables,
    load_export,
    backup_files,
    save_export,
    extract_climate_for_sites,
)


def get_current_env_columns(meta: dict) -> set[str]:
    """Get all env column names currently in the export."""
    alpha = set(meta.get("alpha_cov_cols", []))
    env = set(meta.get("env_cov_cols", []))
    return alpha | env


def identify_missing_variables(
    npy_src: str,
    current_cols: set[str],
    standard_only: bool = True,
) -> list[dict]:
    """Find geonpy variables that have no derived column in the export.

    Returns list of variable metadata dicts for variables not yet represented.
    """
    available = discover_npy_variables(npy_src)

    if standard_only:
        available = [v for v in available if v["standard"]]

    # A variable is "missing" if none of its possible derived names appear
    # in the current columns. Derived name pattern: {cstat}_{var_name}
    # We check if var_name appears as a substring of any current column.
    missing = []
    for v in available:
        var_name = v["var_name"]
        # Check if any column contains this variable name
        in_export = any(var_name in col for col in current_cols)
        if not in_export:
            missing.append(v)

    return missing


def add_variables_batch(
    export_dir: Path,
    variables: list[dict],
    mstat: str,
    cstat: str,
    npy_src: str,
    window: int = DEFAULT_WINDOW,
) -> list[str]:
    """Extract and add a batch of variables to the feather files.

    All variables must use the same mstat and cstat (single pyper call).
    Returns list of added column names.
    """
    sc, est, meta = load_export(export_dir)
    alpha_cols = meta.get("alpha_cov_cols", [])
    env_cols = meta.get("env_cov_cols", [])

    # Get basenames for pyper
    npy_basenames = [v["basename"] for v in variables]
    var_names = [v["var_name"] for v in variables]

    # Expected column names
    expected = [f"{cstat}_{vn}" for vn in var_names]

    # Check for existing columns
    existing = set(sc.columns) | set(est.columns)
    already_exist = [c for c in expected if c in existing]
    if already_exist:
        print(f"  Skipping {len(already_exist)} columns that already exist: "
              f"{already_exist[:5]}{'...' if len(already_exist) > 5 else ''}")
        # Filter to only new ones
        new_mask = [c not in existing for c in expected]
        npy_basenames = [b for b, m in zip(npy_basenames, new_mask) if m]
        var_names = [v for v, m in zip(var_names, new_mask) if m]
        expected = [c for c, m in zip(expected, new_mask) if m]

    if not npy_basenames:
        print("  Nothing new to add for this cstat.")
        return []

    # Extract
    lon = sc["lon"].values.astype(np.float32)
    lat = sc["lat"].values.astype(np.float32)

    print(f"\n  Extracting {len(npy_basenames)} variable(s) at {len(lon):,} sites...")
    print(f"  mstat={mstat}, cstat={cstat}, window={window}yr")
    print(f"  Variables: {var_names}")

    t0 = time.time()
    site_env = extract_climate_for_sites(
        lon=lon, lat=lat,
        variables=npy_basenames,
        mstat=mstat,
        cstat=cstat,
        npy_src=npy_src,
        window=window,
    )
    elapsed = time.time() - t0
    print(f"  Extraction took {elapsed:.1f}s")
    print(f"  Columns returned: {list(site_env.columns)}")

    # Validate
    assert len(site_env) == len(sc), (
        f"Row mismatch: {len(site_env)} vs {len(sc)}"
    )

    # Add to DataFrames
    for col in site_env.columns:
        sc[col] = site_env[col].values
        est[col] = site_env[col].values

    # Update metadata
    for col in site_env.columns:
        if col not in alpha_cols:
            alpha_cols.append(col)
        if col not in env_cols:
            env_cols.append(col)
    meta["alpha_cov_cols"] = alpha_cols
    meta["env_cov_cols"] = env_cols

    save_export(export_dir, sc, est, meta)

    # Print stats
    for col in site_env.columns:
        vals = site_env[col].dropna()
        n_nan = site_env[col].isna().sum()
        print(f"    {col:<30s} mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"NaN={n_nan}")

    return list(site_env.columns)


def run_feature_selection(export_dir: Path, n_sample: int, max_features: int):
    """Run the feature selection report."""
    script = Path(__file__).resolve().parent / "select_env_features.py"
    cmd = [
        sys.executable, str(script),
        "--export-dir", str(export_dir),
        "report",
        "--n-sample", str(n_sample),
        "--max-features", str(max_features),
    ]
    print(f"\n{'='*70}")
    print("Running feature selection...")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add all available geonpy features and run feature selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR),
        help="Path to nn_export directory",
    )
    parser.add_argument(
        "--npy-src", type=str, default=DEFAULT_NPY_SRC,
        help="Path to geonpy .npy files directory",
    )
    parser.add_argument(
        "--mstat", type=str, default="mean",
        choices=["mean", "min", "max"],
        help="Monthly statistic (default: mean)",
    )
    parser.add_argument(
        "--cstat", type=str, nargs="+", default=["mean"],
        choices=["mean", "min", "max", "ptp"],
        help="Climatology statistic(s). Can specify multiple (default: mean)",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Climate window in years (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be added without extracting",
    )
    parser.add_argument(
        "--no-select", action="store_true",
        help="Skip feature selection (just add features)",
    )
    parser.add_argument(
        "--n-sample", type=int, default=100_000,
        help="Pairs to sample for feature selection (default: 100k)",
    )
    parser.add_argument(
        "--max-features", type=int, default=20,
        help="Max features to recommend (default: 20)",
    )
    parser.add_argument(
        "--include-alternative", action="store_true",
        help="Include variables with alternative date ranges (mask variables)",
    )

    args = parser.parse_args()
    export_dir = Path(args.export_dir)

    # ── Step 0: Show current state ──
    sc, est, meta = load_export(export_dir)
    current_cols = get_current_env_columns(meta)
    subs_cols = [c for c in current_cols if c.startswith("subs_")]
    climate_cols = [c for c in current_cols if not c.startswith("subs_")]

    print(f"\n{'='*70}")
    print(f"SCAN ALL FEATURES — geonpy → nn_export → feature selection")
    print(f"{'='*70}")
    print(f"\nExport: {export_dir}")
    print(f"Sites:  {sc.shape[0]:,}")
    print(f"Current features: {len(current_cols)} "
          f"({len(subs_cols)} substrate, {len(climate_cols)} climate)")
    print(f"  Climate: {sorted(climate_cols)}")

    # ── Step 1: Discover missing variables ──
    standard_only = not args.include_alternative
    missing = identify_missing_variables(
        args.npy_src, current_cols, standard_only=standard_only,
    )

    print(f"\nAvailable variables not yet in export: {len(missing)}")
    for v in missing:
        print(f"  {v['var_name']:<25s} ({v['filename']})")

    if not missing:
        print("\nAll available variables are already in the export!")
        if not args.no_select:
            run_feature_selection(export_dir, args.n_sample, args.max_features)
        return

    # Show what will be added
    total_new = 0
    for cstat in args.cstat:
        new_cols = [f"{cstat}_{v['var_name']}" for v in missing]
        existing = set(sc.columns) | set(est.columns)
        truly_new = [c for c in new_cols if c not in existing]
        total_new += len(truly_new)
        print(f"\n  cstat={cstat}: {len(truly_new)} new columns")
        for c in truly_new[:10]:
            print(f"    + {c}")
        if len(truly_new) > 10:
            print(f"    ... and {len(truly_new) - 10} more")

    print(f"\nTotal new columns to add: {total_new}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # ── Step 2: Backup ──
    print(f"\nBacking up current files...")
    backup_files(export_dir)

    # ── Step 3: Extract and add, one cstat at a time ──
    all_added = []
    for cstat in args.cstat:
        print(f"\n{'─'*70}")
        print(f"Processing cstat={cstat}...")
        print(f"{'─'*70}")

        added = add_variables_batch(
            export_dir=export_dir,
            variables=missing,
            mstat=args.mstat,
            cstat=cstat,
            npy_src=args.npy_src,
            window=args.window,
        )
        all_added.extend(added)

    # ── Step 4: Summary ──
    sc2, est2, meta2 = load_export(export_dir)
    new_total = get_current_env_columns(meta2)
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Added {len(all_added)} new columns")
    print(f"  site_covariates: {sc2.shape[0]:,} × {sc2.shape[1]}")
    print(f"  env_site_table:  {est2.shape[0]:,} × {est2.shape[1]}")
    print(f"  Total env features: {len(new_total)} "
          f"({sum(1 for c in new_total if c.startswith('subs_'))} substrate, "
          f"{sum(1 for c in new_total if not c.startswith('subs_'))} climate)")

    # ── Step 5: Feature selection ──
    if not args.no_select:
        run_feature_selection(export_dir, args.n_sample, args.max_features)
    else:
        print("\nSkipping feature selection (--no-select). Run manually:")
        print(f"  python src/clesso_nn/select_env_features.py report")


if __name__ == "__main__":
    main()
