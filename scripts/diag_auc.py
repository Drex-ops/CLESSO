#!/usr/bin/env python3
"""Diagnostic: check AUC calculation and data balance."""
import json, numpy as np, pandas as pd

export_dir = "src/clesso_v2/output/VAS_20260316_181756/nn_export"
pairs = pd.read_feather(f"{export_dir}/pairs.feather")
with open(f"{export_dir}/metadata.json") as f:
    meta = json.load(f)

print("=== Metadata ===")
for k, v in meta.items():
    if k not in ("alpha_cov_names", "env_diff_names"):
        print(f"  {k}: {v}")

print("\n=== Between-site pairs ===")
bp = pairs[pairs["is_within"] == 0]
print(f"Total: {len(bp):,}")
print(f"  y=0 (match):    {(bp.y==0).sum():,}  ({(bp.y==0).mean():.4f})")
print(f"  y=1 (mismatch): {(bp.y==1).sum():,}  ({(bp.y==1).mean():.4f})")

print("\nStratum distribution (between-site):")
if "stratum" in bp.columns:
    for s in sorted(bp.stratum.unique()):
        mask = bp.stratum == s
        n = mask.sum()
        m = (bp.y[mask] == 0).sum()
        print(f"  stratum {s}: n={n:,}, match={m:,} ({m/n:.4f})")
else:
    print("  No stratum column in data")

print("\n=== Within-site pairs ===")
wp = pairs[pairs["is_within"] == 1]
print(f"Total: {len(wp):,}")
print(f"  y=0 (match):    {(wp.y==0).sum():,}  ({(wp.y==0).mean():.4f})")
print(f"  y=1 (mismatch): {(wp.y==1).sum():,}  ({(wp.y==1).mean():.4f})")

# Retention rates
print("\n=== Retention rates ===")
rr = meta.get("retention_rates", {})
if rr:
    for k, v in sorted(rr.items()):
        print(f"  {k}: {v}")
else:
    print("  No retention rates found")

# Check weight distribution
if "weight" in bp.columns:
    print("\n=== Between-site weight stats ===")
    w = bp["weight"]
    print(f"  mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")
    for s in sorted(bp.stratum.unique()) if "stratum" in bp.columns else []:
        mask = bp.stratum == s
        ws = bp.weight[mask]
        print(f"  stratum {s}: mean={ws.mean():.4f}, std={ws.std():.4f}")

# Check columns present
print(f"\n=== Columns: {list(pairs.columns)} ===")
