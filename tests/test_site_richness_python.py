#!/usr/bin/env python3
"""
Test: Verify site richness is correct in the Python NN pipeline.

Ground-truth strategy:
  1. Load raw ALA VAS data
  2. Bin to 0.05° grid using SAME break/centroid logic as R siteAggregator
  3. Compute species richness per cell directly
  4. Compare with:
     a) Original exported site_obs_richness.feather
     b) Corrected site_obs_richness_CORRECTED.feather
     c) What the NN dataset.py actually loads (SiteData.S_obs)
  5. Verify the NN model sees the correct S_obs values

Usage:
    python tests/test_site_richness_python.py
"""

import sys
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  Site Richness Verification Test (Python / NN Pipeline)")
print("=" * 70)

n_pass = 0
n_fail = 0

def check(name, condition, detail=""):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f"  *** PASS: {name} ***")
    else:
        n_fail += 1
        print(f"  *** FAIL: {name} ***")
    if detail:
        print(f"           {detail}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: Compute ground-truth richness from raw ALA
# ──────────────────────────────────────────────────────────────────────────
print("\n--- Step 1: Ground-truth richness from raw ALA ---")

ala_path = project_root / "data" / "ala_vas_2026-03-03.csv"
ala = pd.read_csv(ala_path, low_memory=False)
print(f"  Raw records: {len(ala):,}")

# Grid parameters from the reference raster header
# FWPT_mean_Cmax_mean_1946_1975.hdr:
#   ULXMAP=112.95, ULYMAP=-10.1, XDIM=0.05, YDIM=0.05
#   NROWS=670, NCOLS=813
res_deg = 0.05
xmin = 112.95 - res_deg / 2   # 112.925  (left edge)
xmax = 112.95 + (813 - 1) * res_deg + res_deg / 2  # 153.575
ymin = -10.1 - (670 - 1) * res_deg - res_deg / 2   # -43.575
ymax = -10.1 + res_deg / 2                          # -10.075

# Build breaks and centroids (matching R's siteAggregator exactly)
lon_breaks = np.arange(xmin, xmax + res_deg, res_deg)
lat_breaks = np.arange(ymin, ymax + res_deg, res_deg)
lon_centroids = lon_breaks[:-1] + res_deg / 2
lat_centroids = lat_breaks[:-1] + res_deg / 2

def bin_to_centroid(vals, breaks, centroids):
    """Replicate R's cut() + centroid lookup."""
    bin_idx = np.searchsorted(breaks, vals, side='right') - 1
    # Out of range → NaN
    out = np.full(len(vals), np.nan)
    valid = (bin_idx >= 0) & (bin_idx < len(centroids))
    out[valid] = centroids[bin_idx[valid]]
    return out

ala["lonID"] = bin_to_centroid(ala["decimalLongitude"].values, lon_breaks, lon_centroids)
ala["latID"] = bin_to_centroid(ala["decimalLatitude"].values, lat_breaks, lat_centroids)

# Drop records outside grid
ala = ala.dropna(subset=["lonID", "latID"])

# Build site_id matching R's paste(lonID, latID, sep=":")
# R's formatting: default print of numeric. Need to match precision.
ala["lonID_r"] = ala["lonID"].round(10)
ala["latID_r"] = ala["latID"].round(10)
ala["site_id"] = ala["lonID_r"].astype(str) + ":" + ala["latID_r"].astype(str)

# Ground truth: all-time richness per site (no date filter)
truth_all = ala.groupby("site_id")["scientificName"].nunique().reset_index()
truth_all.columns = ["site_id", "richness_truth"]
print(f"  Ground-truth sites (all dates): {len(truth_all):,}")
print(f"  Richness: mean={truth_all['richness_truth'].mean():.1f}, "
      f"median={truth_all['richness_truth'].median():.0f}, "
      f"max={truth_all['richness_truth'].max()}")

# Ground truth: date-filtered (1970–2018)
ala["eventDate"] = pd.to_datetime(ala["eventDate"], errors="coerce")
ala_filt = ala[(ala["eventDate"] >= "1970-01-01") & (ala["eventDate"] < "2018-01-01")]
truth_filt = ala_filt.groupby("site_id")["scientificName"].nunique().reset_index()
truth_filt.columns = ["site_id", "richness_truth_filt"]
print(f"  Ground-truth sites (1970–2018): {len(truth_filt):,}")
print(f"  Richness: mean={truth_filt['richness_truth_filt'].mean():.1f}, "
      f"median={truth_filt['richness_truth_filt'].median():.0f}, "
      f"max={truth_filt['richness_truth_filt'].max()}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: Compare with ORIGINAL exported S_obs
# ──────────────────────────────────────────────────────────────────────────
print("\n--- Step 2: Compare with original site_obs_richness.feather ---")

export_dir = project_root / "src" / "clesso_v2" / "output" / "VAS_20260310_092634" / "nn_export"
orig = feather.read_feather(export_dir / "site_obs_richness.feather")
print(f"  Original file: {len(orig)} sites, S_obs mean={orig['S_obs'].mean():.2f}")

merged_orig = truth_filt.merge(orig, on="site_id", how="inner")
print(f"  Matched: {len(merged_orig)} sites")

if len(merged_orig) > 0:
    ratio = merged_orig["richness_truth_filt"] / merged_orig["S_obs"].clip(lower=0.01)
    mean_ratio = ratio.mean()
    print(f"  Mean truth/original ratio: {mean_ratio:.1f}x")
    check("Original S_obs is under-counted (known bug)",
          mean_ratio > 5,
          f"Expected ratio >> 1 due to subsampled obs_dt; got {mean_ratio:.1f}x")
else:
    print("  WARNING: no matching sites (different ID format?)")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: Compare with CORRECTED S_obs
# ──────────────────────────────────────────────────────────────────────────
print("\n--- Step 3: Compare with corrected site_obs_richness_CORRECTED.feather ---")

corrected_path = export_dir / "site_obs_richness_CORRECTED.feather"
if corrected_path.exists():
    corrected = feather.read_feather(corrected_path)
    print(f"  Corrected file: {len(corrected)} sites, S_obs mean={corrected['S_obs'].mean():.1f}")

    merged_corr = truth_filt.merge(corrected, on="site_id", how="inner")
    print(f"  Matched: {len(merged_corr)} sites")

    if len(merged_corr) > 0:
        exact_match = (merged_corr["richness_truth_filt"] == merged_corr["S_obs"]).mean()
        close_match = (np.abs(merged_corr["richness_truth_filt"] - merged_corr["S_obs"]) <= 1).mean()
        ratio_corr = (merged_corr["richness_truth_filt"] / merged_corr["S_obs"].clip(lower=1)).mean()
        corr = merged_corr["richness_truth_filt"].corr(merged_corr["S_obs"])

        print(f"  Exact match: {100*exact_match:.1f}%")
        print(f"  Within ±1:   {100*close_match:.1f}%")
        print(f"  Mean ratio:  {ratio_corr:.3f}")
        print(f"  Correlation: {corr:.4f}")

        check("Corrected S_obs matches ground truth (correlation > 0.99)",
              corr > 0.99,
              f"r = {corr:.4f}")
        check("Corrected S_obs mean ratio close to 1.0",
              0.9 < ratio_corr < 1.1,
              f"ratio = {ratio_corr:.3f}")
else:
    print("  WARNING: Corrected file not found!")
    n_fail += 1

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: Verify what the NN dataset.py actually loads
# ──────────────────────────────────────────────────────────────────────────
print("\n--- Step 4: Verify NN dataset.py loads correct S_obs ---")

from src.clesso_nn.dataset import load_export, SiteData

data = load_export(export_dir)

# Build SiteData (same as run_clesso_nn.py)
sd = SiteData(
    site_covariates=data["site_covariates"],
    env_site_table=data.get("env_site_table"),
    site_obs_richness=data["site_obs_richness"],
    metadata=data["metadata"],
)

# Extract S_obs per site
s_obs_loaded = sd.S_obs.numpy()
n_sites = sd.n_sites
n_with_sobs = (s_obs_loaded > 0).sum()

print(f"  NN sites: {n_sites}")
print(f"  Sites with S_obs > 0: {n_with_sobs}")
print(f"  S_obs: mean={s_obs_loaded[s_obs_loaded > 0].mean():.1f}, "
      f"median={np.median(s_obs_loaded[s_obs_loaded > 0]):.0f}, "
      f"max={s_obs_loaded.max():.0f}")

# Map back to site_ids for comparison
idx_to_sid = {v: k for k, v in sd.site_id_to_idx.items()}
nn_richness = pd.DataFrame({
    "site_id": [idx_to_sid[i] for i in range(n_sites)],
    "S_obs_nn": s_obs_loaded,
})
nn_nonzero = nn_richness[nn_richness["S_obs_nn"] > 0]

# Compare with corrected file
if corrected_path.exists():
    merged_nn = nn_nonzero.merge(corrected, on="site_id", how="inner")
    if len(merged_nn) > 0:
        nn_exact = (merged_nn["S_obs_nn"] == merged_nn["S_obs"]).mean()
        nn_corr = merged_nn["S_obs_nn"].corr(merged_nn["S_obs"])
        print(f"  NN vs corrected file: exact match {100*nn_exact:.1f}%, r={nn_corr:.4f}")
        check("NN dataset loads corrected S_obs values",
              nn_exact > 0.99,
              f"Exact match = {100*nn_exact:.1f}%")
    else:
        print("  WARNING: No overlap between NN sites and corrected file")
        n_fail += 1

# Compare with ground truth
merged_nn_truth = nn_nonzero.merge(truth_filt, on="site_id", how="inner")
if len(merged_nn_truth) > 0:
    nn_truth_corr = merged_nn_truth["S_obs_nn"].corr(merged_nn_truth["richness_truth_filt"])
    nn_truth_ratio = (merged_nn_truth["S_obs_nn"] / merged_nn_truth["richness_truth_filt"].clip(lower=1)).mean()
    print(f"  NN S_obs vs ground truth: r={nn_truth_corr:.4f}, mean ratio={nn_truth_ratio:.3f}")
    check("NN S_obs matches ground truth (r > 0.99)",
          nn_truth_corr > 0.99,
          f"r = {nn_truth_corr:.4f}")
    check("NN S_obs mean ratio close to 1.0",
          0.9 < nn_truth_ratio < 1.1,
          f"ratio = {nn_truth_ratio:.3f}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: Verify S_obs is reasonable (not the old subsampled values)
# ──────────────────────────────────────────────────────────────────────────
print("\n--- Step 5: Sanity checks ---")

mean_sobs = s_obs_loaded[s_obs_loaded > 0].mean()
check("Mean S_obs > 10 (not subsampled)",
      mean_sobs > 10,
      f"mean = {mean_sobs:.1f} (old bug had mean ≈ 2.7)")

check("Max S_obs > 100 (realistic for VAS)",
      s_obs_loaded.max() > 100,
      f"max = {s_obs_loaded.max():.0f}")

# Check that alpha_regression_lambda is disabled
from src.clesso_nn.config import CLESSONNConfig
cfg = CLESSONNConfig()
check("alpha_regression_lambda is 0 (disabled)",
      cfg.alpha_regression_lambda == 0,
      f"alpha_regression_lambda = {cfg.alpha_regression_lambda}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Passed: {n_pass}")
print(f"  Failed: {n_fail}")
print("=" * 70)

sys.exit(0 if n_fail == 0 else 1)
