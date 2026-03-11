#!/usr/bin/env python3
"""Check within-site vs between-site y distributions and weight distributions."""
import numpy as np
import pyarrow.feather as feather
from pathlib import Path

export_dir = Path('src/clesso_v2/output/VAS_20260310_092634/nn_export')
pairs = feather.read_feather(export_dir / 'pairs.feather')

print("=== Within-site pairs ===")
w = pairs[pairs['is_within'] == 1]
print(f"  N = {len(w):,}")
print(f"  y=0 (match): {(w['y']==0).sum():,} ({(w['y']==0).mean()*100:.1f}%)")
print(f"  y=1 (mismatch): {(w['y']==1).sum():,} ({(w['y']==1).mean()*100:.1f}%)")
if 'w' in w.columns:
    print(f"  weight: mean={w['w'].mean():.4f}, std={w['w'].std():.4f}, min={w['w'].min():.4f}, max={w['w'].max():.4f}")
print()

print("=== Between-site pairs ===")
b = pairs[pairs['is_within'] == 0]
print(f"  N = {len(b):,}")
print(f"  y=0 (match): {(b['y']==0).sum():,} ({(b['y']==0).mean()*100:.1f}%)")
print(f"  y=1 (mismatch): {(b['y']==1).sum():,} ({(b['y']==1).mean()*100:.1f}%)")
if 'w' in b.columns:
    print(f"  weight: mean={b['w'].mean():.4f}, std={b['w'].std():.4f}, min={b['w'].min():.4f}, max={b['w'].max():.4f}")
print()

# Check observed richness
rich_path = export_dir / 'site_obs_richness.feather'
if rich_path.exists():
    rich = feather.read_feather(rich_path)
    print("=== Observed Richness ===")
    print(f"  Sites: {len(rich)}")
    print(f"  S_obs: mean={rich['S_obs'].mean():.1f}, median={rich['S_obs'].median():.1f}, min={rich['S_obs'].min()}, max={rich['S_obs'].max()}")
    print(f"  S_obs quantiles: 25%={rich['S_obs'].quantile(0.25):.0f}, 75%={rich['S_obs'].quantile(0.75):.0f}")
else:
    print("No site_obs_richness.feather found!")

print()
print("=== Overall stats ===")
print(f"  Total pairs: {len(pairs):,}")
print(f"  Within fraction: {len(w)/len(pairs)*100:.1f}%")
print(f"  Overall y=0: {(pairs['y']==0).sum():,} ({(pairs['y']==0).mean()*100:.1f}%)")
print(f"  Overall y=1: {(pairs['y']==1).sum():,} ({(pairs['y']==1).mean()*100:.1f}%)")

# Expected match rate for within-site
if rich_path.exists():
    expected_p = (1 / rich['S_obs']).mean()
    print(f"\n  Expected within-site match rate (1/S_obs avg): {expected_p:.4f}")
    print(f"  Actual within-site match rate: {(w['y']==0).mean():.4f}")
