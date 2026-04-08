#!/usr/bin/env python3
"""
manage_env_features.py — List, add, and remove environmental features
from nn_export feather files.

Usage:
    # List all available geonpy variables (from the SSD)
    python manage_env_features.py list-available

    # Show what's currently in the nn_export feather files
    python manage_env_features.py show

    # Add a new climate feature (extracts from geonpy via pyper.py)
    python manage_env_features.py add-climate Precip --mstat mean --cstat mean
    python manage_env_features.py add-climate SolarMJ TempMean --mstat mean --cstat max

    # Remove a feature column from both feather files
    python manage_env_features.py remove max_PD min_FWPT

    # Undo all changes (restore from backup)
    python manage_env_features.py restore

Notes:
    - Adds columns to BOTH site_covariates.feather (alpha) and
      env_site_table.feather (beta).  Updates metadata.json too.
    - Backups are created automatically before any modification
      (*.feather.bak, metadata.json.bak).
    - Climate extraction requires: geonpy .npy files on SSD, pyper.py,
      and site lon/lat in site_covariates.feather.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from clesso_nn.config import ACTIVE_PROFILE

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXPORT_DIR = (
    PROJECT_ROOT / "src" / "clesso_v2" / "output"
    / "VAS_20260310_092634" / "nn_export"
)
DEFAULT_NPY_SRC = ACTIVE_PROFILE["climate_npy_dir"]
PYPER_SCRIPT = PROJECT_ROOT / "src" / "shared" / "python" / "pyper.py"

# Climate window defaults (30-year mean centred on 2010)
DEFAULT_WINDOW = 30
DEFAULT_YEAR = 2010
DEFAULT_MONTH = 6

# .npy files that use _191101-201712 date range
STANDARD_DATE_RANGE = "191101-201712"

# Files that use different date ranges (e.g. _19210101-20180831)
# These need special handling for the start_year param
ALTERNATIVE_DATE_RANGES = {
    "19210101-20180831": 1921,
    "19410101-20180831": 1941,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_npy_variables(npy_src: str) -> list[dict]:
    """Scan the geonpy directory and return metadata for each .npy file.

    Returns list of dicts with keys:
        filename, basename, date_range, start_year, standard
    """
    npy_dir = Path(npy_src)
    if not npy_dir.exists():
        print(f"ERROR: geonpy directory not found: {npy_dir}")
        print("       Is the PortableSSD mounted?")
        sys.exit(1)

    results = []
    for f in sorted(npy_dir.glob("*.npy")):
        # Skip macOS resource fork files (._*)
        if f.name.startswith("._"):
            continue
        basename = f.stem  # e.g. "FD_191101-201712" or "lt0_mask_19210101-20180831"
        # Check for standard date range
        if STANDARD_DATE_RANGE in basename:
            var_name = basename.replace(f"_{STANDARD_DATE_RANGE}", "")
            results.append({
                "filename": f.name,
                "basename": basename,
                "var_name": var_name,
                "date_range": STANDARD_DATE_RANGE,
                "start_year": 1911,
                "standard": True,
            })
        else:
            # Try alternative date ranges
            for dr, sy in ALTERNATIVE_DATE_RANGES.items():
                if dr in basename:
                    var_name = basename.replace(f"_{dr}", "")
                    results.append({
                        "filename": f.name,
                        "basename": basename,
                        "var_name": var_name,
                        "date_range": dr,
                        "start_year": sy,
                        "standard": False,
                    })
                    break
            else:
                # Unknown format
                results.append({
                    "filename": f.name,
                    "basename": basename,
                    "var_name": basename,
                    "date_range": "unknown",
                    "start_year": None,
                    "standard": False,
                })

    return results


def load_export(export_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load site_covariates, env_site_table, and metadata from nn_export."""
    sc = pf.read_feather(export_dir / "site_covariates.feather")
    est = pf.read_feather(export_dir / "env_site_table.feather")
    meta_path = export_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return sc, est, meta


def backup_files(export_dir: Path):
    """Create .bak copies of feather files and metadata."""
    for name in ["site_covariates.feather", "env_site_table.feather", "metadata.json"]:
        src = export_dir / name
        dst = export_dir / f"{name}.bak"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Backed up: {name} → {name}.bak")
        elif src.exists() and dst.exists():
            print(f"  Backup already exists: {name}.bak (not overwritten)")


def save_export(export_dir: Path, sc: pd.DataFrame, est: pd.DataFrame, meta: dict):
    """Write updated feather files and metadata."""
    pf.write_feather(sc, export_dir / "site_covariates.feather")
    pf.write_feather(est, export_dir / "env_site_table.feather")
    with open(export_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)


def get_current_env_cols(meta: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (alpha_cov_cols, env_cov_cols, effort_cov_cols) from metadata."""
    return (
        meta.get("alpha_cov_cols", []),
        meta.get("env_cov_cols", []),
        meta.get("effort_cov_cols", []),
    )


def extract_climate_for_sites(
    lon: np.ndarray,
    lat: np.ndarray,
    variables: list[str],
    mstat: str,
    cstat: str,
    npy_src: str,
    window: int = DEFAULT_WINDOW,
    year: int = DEFAULT_YEAR,
    month: int = DEFAULT_MONTH,
) -> pd.DataFrame:
    """Extract climate values at site locations using pyper.py.

    Creates a fake "pairs" feather (site paired with itself) so pyper
    can extract values, then returns a DataFrame with one row per site.
    """
    n = len(lon)

    # Build pairs array: each site paired with itself
    # Format: x1, y1, year1, month1, x2, y2, year2, month2
    pairs_arr = np.column_stack([
        lon, lat,
        np.full(n, year, dtype=np.float32),
        np.full(n, month, dtype=np.float32),
        lon, lat,
        np.full(n, year, dtype=np.float32),
        np.full(n, month, dtype=np.float32),
    ]).astype(np.float32)

    pairs_df = pd.DataFrame(pairs_arr, columns=[
        "x_1", "y_1", "year_1", "month_1",
        "x_2", "y_2", "year_2", "month_2",
    ])

    # Write to temp feather
    with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as tmp_in:
        tmp_in_path = tmp_in.name
    pf.write_feather(pairs_df, tmp_in_path)

    # Build pyper.py command
    python_exe = sys.executable
    cmd_args = [
        python_exe, str(PYPER_SCRIPT),
        "-f", str(tmp_in_path),
        "-e", *variables,
        "-s", str(cstat),
        "-m", str(mstat),
        "-w", str(window),
        "-src", str(npy_src),
    ]
    print(f"  Running pyper: {' '.join(str(a) for a in cmd_args)}")
    proc = subprocess.run(cmd_args, capture_output=True, text=True)
    result_path = proc.stdout.strip()

    if not result_path or not Path(result_path).exists():
        os.unlink(tmp_in_path)
        print("  ERROR: pyper.py failed or returned no output path.")
        if proc.stderr:
            print(f"  stderr: {proc.stderr.strip()}")
        print(f"  Command was: {' '.join(str(a) for a in cmd_args)}")
        sys.exit(1)

    # Read result
    result_df = pf.read_feather(result_path)

    # Cleanup temp files
    os.unlink(tmp_in_path)
    os.unlink(result_path)

    # Extract _1 columns (site values; _2 are identical since self-paired)
    env_cols = [c for c in result_df.columns if c.endswith("_1") and c not in
                ("x_1", "y_1", "year_1", "month_1")]
    site_env = result_df[env_cols].copy()

    # Clean column names: strip date range + _1 suffix, prefix with cstat
    clean_names = {}
    for col in site_env.columns:
        name = col
        # Remove date range patterns
        import re
        name = re.sub(r"\d{6}-\d{6}_", "", name)
        name = re.sub(r"\d{8}-\d{8}_", "", name)
        # Remove _1 suffix
        name = re.sub(r"_1$", "", name)
        # Prefix with cstat
        name = f"{cstat}_{name}"
        clean_names[col] = name
    site_env = site_env.rename(columns=clean_names)

    return site_env


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list_available(args):
    """List all available .npy variables on the SSD."""
    variables = discover_npy_variables(args.npy_src)

    # Get currently used variables from the export
    sc, est, meta = load_export(Path(args.export_dir))
    current_alpha = set(meta.get("alpha_cov_cols", []))
    current_env = set(meta.get("env_cov_cols", []))
    current_all = current_alpha | current_env

    print(f"\n{'='*80}")
    print(f"Available geonpy .npy variables: {args.npy_src}")
    print(f"{'='*80}\n")

    # Group by date range
    standard = [v for v in variables if v["standard"]]
    other = [v for v in variables if not v["standard"]]

    print(f"Standard variables ({STANDARD_DATE_RANGE}, start_year=1911):")
    print(f"  {'Variable':<25} {'Filename':<40} {'In export?'}")
    print(f"  {'─'*25} {'─'*40} {'─'*12}")
    for v in standard:
        # Check if any derived column from this var is in the export
        in_export = any(v["var_name"] in col for col in current_all)
        marker = "✓ (in use)" if in_export else ""
        print(f"  {v['var_name']:<25} {v['filename']:<40} {marker}")

    if other:
        print(f"\nOther variables (alternative date ranges):")
        print(f"  {'Variable':<25} {'Filename':<45} {'Date range':<25} {'start_year'}")
        print(f"  {'─'*25} {'─'*45} {'─'*25} {'─'*10}")
        for v in other:
            print(f"  {v['var_name']:<25} {v['filename']:<45} {v['date_range']:<25} {v['start_year']}")

    print(f"\nTotal: {len(variables)} .npy files ({len(standard)} standard, {len(other)} other)")

    # Show possible derived features
    print(f"\n{'─'*80}")
    print("To add a variable, combine it with an mstat (monthly summary)")
    print("and a cstat (climatology summary across years):")
    print()
    print("  mstat options: mean, min, max")
    print("  cstat options: mean, min, max, ptp (range)")
    print()
    print("  Example: add-climate Precip --mstat mean --cstat mean")
    print("           → column name: mean_Precip")
    print()
    print("  Example: add-climate SolarMJ TempMean --mstat mean --cstat max")
    print("           → column names: max_SolarMJ, max_TempMean")


def cmd_show(args):
    """Show current columns in the nn_export feather files."""
    export_dir = Path(args.export_dir)
    sc, est, meta = load_export(export_dir)

    alpha_cols, env_cols, effort_cols = get_current_env_cols(meta)

    # Separate substrate from climate
    subs_cols = [c for c in alpha_cols if c.startswith("subs_")]
    climate_cols = [c for c in alpha_cols if not c.startswith("subs_")]

    print(f"\n{'='*70}")
    print(f"nn_export: {export_dir}")
    print(f"{'='*70}")
    print(f"\nSites: {sc.shape[0]:,}  |  site_covariates cols: {sc.shape[1]}")
    print(f"                      |  env_site_table cols:  {est.shape[1]}")

    print(f"\n── Substrate ({len(subs_cols)} columns) ──")
    for c in subs_cols:
        vals = sc[c].dropna()
        print(f"  {c:<25} mean={vals.mean():8.2f}  std={vals.std():8.2f}  "
              f"range=[{vals.min():.2f}, {vals.max():.2f}]")

    print(f"\n── Climate ({len(climate_cols)} columns) ──")
    for c in climate_cols:
        vals = sc[c].dropna()
        in_env = "α+β" if c in est.columns else "α only"
        print(f"  {c:<25} mean={vals.mean():8.2f}  std={vals.std():8.2f}  "
              f"range=[{vals.min():.2f}, {vals.max():.2f}]  [{in_env}]")

    print(f"\n── Effort ({len(effort_cols)} columns) ──")
    for c in effort_cols:
        if c in sc.columns:
            vals = sc[c].dropna()
            print(f"  {c:<40} mean={vals.mean():10.2f}  std={vals.std():10.2f}")

    # Check for columns in feather but NOT in metadata
    sc_extra = set(sc.columns) - {"site_id", "lon", "lat"} - set(alpha_cols) - set(effort_cols)
    if sc_extra:
        print(f"\n── Extra columns in site_covariates (not in metadata) ──")
        for c in sorted(sc_extra):
            print(f"  {c}")

    est_extra = set(est.columns) - {"site_id"} - set(env_cols)
    if est_extra:
        print(f"\n── Extra columns in env_site_table (not in metadata) ──")
        for c in sorted(est_extra):
            print(f"  {c}")

    # Check for backup files
    has_backup = (export_dir / "site_covariates.feather.bak").exists()
    if has_backup:
        print(f"\n  ℹ Backup files exist (.bak) — use 'restore' to revert changes.")


def cmd_add_climate(args):
    """Add new climate variable(s) extracted from geonpy."""
    export_dir = Path(args.export_dir)
    sc, est, meta = load_export(export_dir)
    alpha_cols, env_cols, effort_cols = get_current_env_cols(meta)

    # Resolve variable basenames to .npy filenames
    available = discover_npy_variables(args.npy_src)
    avail_map = {v["var_name"]: v for v in available}

    npy_basenames = []
    for var in args.variables:
        if var in avail_map:
            npy_basenames.append(avail_map[var]["basename"])
        else:
            # Try exact match against basename
            exact = [v for v in available if v["basename"] == var]
            if exact:
                npy_basenames.append(exact[0]["basename"])
            else:
                print(f"ERROR: Variable '{var}' not found in {args.npy_src}")
                print(f"  Available: {', '.join(sorted(avail_map.keys()))}")
                sys.exit(1)

    # What column names will be produced?
    expected_cols = [f"{args.cstat}_{var}" for var in args.variables]

    # Check for duplicates
    existing = set(sc.columns) | set(est.columns)
    dupes = [c for c in expected_cols if c in existing]
    if dupes:
        print(f"WARNING: These columns already exist: {dupes}")
        if not args.force:
            print("  Use --force to overwrite.")
            sys.exit(1)
        print("  --force: will overwrite existing columns.")

    # Backup before modification
    backup_files(export_dir)

    # Extract climate values
    lon = sc["lon"].values.astype(np.float32)
    lat = sc["lat"].values.astype(np.float32)

    print(f"\nExtracting {len(npy_basenames)} variable(s) at {len(lon):,} sites...")
    print(f"  Variables: {', '.join(args.variables)}")
    print(f"  mstat={args.mstat}, cstat={args.cstat}, window={args.window}yr")

    site_env = extract_climate_for_sites(
        lon=lon, lat=lat,
        variables=npy_basenames,
        mstat=args.mstat,
        cstat=args.cstat,
        npy_src=args.npy_src,
        window=args.window,
    )

    print(f"  Extracted columns: {list(site_env.columns)}")

    # Validate shapes
    assert len(site_env) == len(sc), (
        f"Row mismatch: extracted {len(site_env)} but feather has {len(sc)}"
    )

    # Add to site_covariates
    for col in site_env.columns:
        if col in sc.columns and args.force:
            sc = sc.drop(columns=[col])
        sc[col] = site_env[col].values

    # Add to env_site_table
    for col in site_env.columns:
        if col in est.columns and args.force:
            est = est.drop(columns=[col])
        est[col] = site_env[col].values

    # Update metadata
    for col in site_env.columns:
        if col not in alpha_cols:
            alpha_cols.append(col)
        if col not in env_cols:
            env_cols.append(col)
    meta["alpha_cov_cols"] = alpha_cols
    meta["env_cov_cols"] = env_cols

    # Save
    save_export(export_dir, sc, est, meta)

    print(f"\n✓ Added {len(site_env.columns)} column(s) to both feather files.")
    print(f"  site_covariates: {sc.shape[1]} cols")
    print(f"  env_site_table:  {est.shape[1]} cols")

    # Quick stats
    for col in site_env.columns:
        vals = site_env[col].dropna()
        n_na = site_env[col].isna().sum()
        print(f"  {col:<25} mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]  NaN={n_na}")


def cmd_remove(args):
    """Remove columns from both feather files."""
    export_dir = Path(args.export_dir)
    sc, est, meta = load_export(export_dir)
    alpha_cols, env_cols, effort_cols = get_current_env_cols(meta)

    # Validate columns exist
    all_removable = set(sc.columns) - {"site_id", "lon", "lat"}
    not_found = [c for c in args.columns if c not in all_removable]
    if not_found:
        print(f"ERROR: Columns not found (or not removable): {not_found}")
        print(f"  Removable columns: {sorted(all_removable)}")
        sys.exit(1)

    # Warn about effort columns
    effort_remove = [c for c in args.columns if c in effort_cols]
    if effort_remove:
        print(f"WARNING: Removing effort column(s): {effort_remove}")
        print("  This will affect EffortNet features.")

    # Warn about substrate columns
    subs_remove = [c for c in args.columns if c.startswith("subs_")]
    if subs_remove:
        print(f"WARNING: Removing substrate column(s): {subs_remove}")

    # Backup
    backup_files(export_dir)

    # Remove from DataFrames
    cols_to_remove = set(args.columns)
    removed_sc = [c for c in args.columns if c in sc.columns]
    removed_est = [c for c in args.columns if c in est.columns]

    sc = sc.drop(columns=[c for c in cols_to_remove if c in sc.columns])
    est = est.drop(columns=[c for c in cols_to_remove if c in est.columns])

    # Update metadata
    meta["alpha_cov_cols"] = [c for c in alpha_cols if c not in cols_to_remove]
    meta["env_cov_cols"] = [c for c in env_cols if c not in cols_to_remove]
    meta["effort_cov_cols"] = [c for c in effort_cols if c not in cols_to_remove]

    save_export(export_dir, sc, est, meta)

    print(f"\n✓ Removed columns:")
    for c in args.columns:
        from_files = []
        if c in removed_sc:
            from_files.append("site_covariates")
        if c in removed_est:
            from_files.append("env_site_table")
        print(f"  {c:<25} from {', '.join(from_files) if from_files else '(not found in feather)'}")
    print(f"  site_covariates: {sc.shape[1]} cols")
    print(f"  env_site_table:  {est.shape[1]} cols")


def cmd_restore(args):
    """Restore feather files from .bak backups."""
    export_dir = Path(args.export_dir)

    restored = False
    for name in ["site_covariates.feather", "env_site_table.feather", "metadata.json"]:
        bak = export_dir / f"{name}.bak"
        dst = export_dir / name
        if bak.exists():
            shutil.copy2(bak, dst)
            print(f"  Restored: {name} ← {name}.bak")
            restored = True
        else:
            print(f"  No backup found: {name}.bak")

    if restored:
        sc, est, meta = load_export(export_dir)
        print(f"\n✓ Restored to backup state.")
        print(f"  site_covariates: {sc.shape[1]} cols, {sc.shape[0]:,} rows")
        print(f"  env_site_table:  {est.shape[1]} cols")
    else:
        print("\n✗ No backup files found. Nothing to restore.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manage environmental features in nn_export feather files.",
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

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # -- list-available --
    sub.add_parser("list-available", aliases=["ls"],
                   help="List all available geonpy .npy variables")

    # -- show --
    sub.add_parser("show", aliases=["info"],
                   help="Show current columns in nn_export feather files")

    # -- add-climate --
    add_p = sub.add_parser("add-climate", aliases=["add"],
                           help="Add climate variable(s) from geonpy")
    add_p.add_argument("variables", nargs="+",
                       help="Variable name(s) matching .npy basenames "
                            "(e.g. Precip, SolarMJ, TempMean)")
    add_p.add_argument("--mstat", default="mean", choices=["mean", "min", "max"],
                       help="Monthly statistic (default: mean)")
    add_p.add_argument("--cstat", default="mean", choices=["mean", "min", "max", "ptp"],
                       help="Climatology statistic (default: mean)")
    add_p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                       help=f"Climate window in years (default: {DEFAULT_WINDOW})")
    add_p.add_argument("--force", action="store_true",
                       help="Overwrite if column already exists")

    # -- remove --
    rm_p = sub.add_parser("remove", aliases=["rm"],
                          help="Remove column(s) from feather files")
    rm_p.add_argument("columns", nargs="+",
                      help="Column name(s) to remove")

    # -- restore --
    sub.add_parser("restore",
                   help="Restore feather files from .bak backups")

    args = parser.parse_args()

    if args.command in ("list-available", "ls"):
        cmd_list_available(args)
    elif args.command in ("show", "info"):
        cmd_show(args)
    elif args.command in ("add-climate", "add"):
        cmd_add_climate(args)
    elif args.command in ("remove", "rm"):
        cmd_remove(args)
    elif args.command == "restore":
        cmd_restore(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
