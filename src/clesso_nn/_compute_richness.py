#!/usr/bin/env python3
"""Derive site_obs_richness from within-site pair match rates.

For a site with S equally-abundant species, the probability of two
random observations matching is 1/S. So:
    S_obs ≈ n_within_pairs / n_within_matches

This is the standard plug-in Simpson diversity estimator and provides
a reasonable approximation when the original observation data is not
available.
"""
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path

export_dir = Path('src/clesso_v2/output/VAS_20260310_092634/nn_export')
pairs = feather.read_feather(export_dir / 'pairs.feather')

# Filter to within-site pairs
within = pairs[pairs['is_within'] == 1].copy()
print(f"Within-site pairs: {len(within):,}")

# Compute per-site match rate
site_stats = within.groupby('site_i').agg(
    n_pairs=('y', 'count'),
    n_matches=('y', lambda x: (x == 0).sum()),
    n_mismatches=('y', lambda x: (x == 1).sum()),
).reset_index()
site_stats.columns = ['site_id', 'n_pairs', 'n_matches', 'n_mismatches']

# Match rate = n_matches / n_pairs ≈ 1/S
# So S_obs ≈ n_pairs / n_matches
# Guard against division by zero (sites with no matches → very high richness)
site_stats['match_rate'] = site_stats['n_matches'] / site_stats['n_pairs']
site_stats['S_obs'] = np.where(
    site_stats['n_matches'] > 0,
    site_stats['n_pairs'] / site_stats['n_matches'],
    site_stats['n_pairs']  # fallback: use n_pairs as lower bound
)

print(f"\nSites with richness estimates: {len(site_stats)}")
print(f"S_obs: mean={site_stats['S_obs'].mean():.1f}, median={site_stats['S_obs'].median():.1f}")
print(f"  min={site_stats['S_obs'].min():.1f}, max={site_stats['S_obs'].max():.1f}")
print(f"  25th={site_stats['S_obs'].quantile(0.25):.1f}, 75th={site_stats['S_obs'].quantile(0.75):.1f}")
print(f"\nMatch rate: mean={site_stats['match_rate'].mean():.4f}")
print(f"  Sites with 0 matches: {(site_stats['n_matches']==0).sum()}")

# Export
out = site_stats[['site_id', 'S_obs']].copy()
feather.write_feather(out, str(export_dir / 'site_obs_richness.feather'))
print(f"\nSaved {len(out)} sites to {export_dir / 'site_obs_richness.feather'}")

# Sanity check: distribution
print("\nS_obs distribution:")
for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    print(f"  {q*100:.0f}th: {site_stats['S_obs'].quantile(q):.1f}")
