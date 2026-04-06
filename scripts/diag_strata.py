#!/usr/bin/env python3
"""Diagnostic: investigate stratum characteristics and AUC validity."""
import pandas as pd
import numpy as np

export_dir = "src/clesso_v2/output/VAS_20260316_181756/nn_export"
pairs = pd.read_feather(f"{export_dir}/pairs.feather")

bp = pairs[pairs.is_within == 0].copy()

# Geographic distance
bp["geo_dist"] = np.sqrt((bp.lon_i - bp.lon_j)**2 + (bp.lat_i - bp.lat_j)**2)

print("=== Geographic distance by stratum ===")
for s in sorted(bp.stratum.unique()):
    d = bp.geo_dist[bp.stratum == s]
    print(f"  stratum {s}: mean={d.mean():.4f}, med={d.median():.4f}, std={d.std():.4f}")

print("\n=== Geographic distance by stratum x y ===")
for s in sorted(bp.stratum.unique()):
    for yv in [0, 1]:
        mask = (bp.stratum == s) & (bp.y == yv)
        if mask.sum() == 0:
            continue
        d = bp.geo_dist[mask]
        print(f"  stratum {s}, y={yv}: n={mask.sum():6d}, geo_dist mean={d.mean():.4f} med={d.median():.4f}")

print("\n=== Design weight by stratum ===")
for s in sorted(bp.stratum.unique()):
    w = bp.design_w[bp.stratum == s]
    print(f"  stratum {s}: w_mean={w.mean():.4f}, w_med={w.median():.4f}, w_min={w.min():.6f}, w_max={w.max():.4f}")

# Check pair_type breakdown 
print("\n=== pair_type by stratum ===")
for s in sorted(bp.stratum.unique()):
    mask = bp.stratum == s
    pts = bp.pair_type[mask].value_counts()
    print(f"  stratum {s}:")
    for pt, cnt in pts.items():
        print(f"    {pt}: {cnt:,}")

# Env diff magnitudes by stratum (compute from site covariates)
env_cols = ["subs_1", "subs_2", "subs_3", "subs_4", "subs_5", "subs_6",
            "mean_mean_PT", "min_TNn", "min_FWPT", "max_max_PT", "max_FWPT",
            "max_FD", "max_TXx", "max_TNn", "max_PD"]

sites = pd.read_feather(f"{export_dir}/site_covariates.feather")
env_data = pd.read_feather(f"{export_dir}/env_site_table.feather")

# Compute env_diff norm for a sample of pairs from each stratum
print("\n=== Env diff magnitude by stratum (L2 norm, sampled 10k per stratum) ===")
rng = np.random.default_rng(42)
for s in sorted(bp.stratum.unique()):
    mask = bp.stratum == s
    idx = bp.index[mask]
    if len(idx) > 10000:
        idx = rng.choice(idx, 10000, replace=False)
    sub = bp.loc[idx]
    # Get env covariates for sites i and j
    ei = env_data.loc[sub.site_i.values]
    ej = env_data.loc[sub.site_j.values]
    if len(env_cols) <= len(ei.columns):
        cols_avail = [c for c in env_cols if c in ei.columns]
        ediff = np.abs(ei[cols_avail].values - ej[cols_avail].values)
        enorm = np.sqrt((ediff**2).sum(axis=1))
        print(f"  stratum {s}: env_L2 mean={enorm.mean():.4f}, med={np.median(enorm):.4f}, std={enorm.std():.4f}")
    else:
        print(f"  stratum {s}: could not compute (columns mismatch)")
