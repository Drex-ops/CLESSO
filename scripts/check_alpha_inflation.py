#!/usr/bin/env python3
"""Quick diagnostic: what's driving alpha so high?"""
import pyarrow.feather as pf
import numpy as np
from pathlib import Path

export = Path('src/clesso_v2/output/VAS_20260316_053959/nn_export')

# 1. Site observed richness (S_obs) — the lower bound target
sobs = pf.read_feather(export / 'site_obs_richness.feather')
print('=== S_obs (site observed richness) ===')
print(f'Columns: {list(sobs.columns)}')
s = sobs.iloc[:, -1] if sobs.shape[1] > 1 else sobs.iloc[:, 0]
print(f'  mean:   {s.mean():.1f}')
print(f'  median: {s.median():.1f}')
print(f'  min:    {s.min():.1f}')
print(f'  max:    {s.max():.1f}')
print(f'  p25:    {s.quantile(0.25):.1f}')
print(f'  p75:    {s.quantile(0.75):.1f}')
print(f'  p95:    {s.quantile(0.95):.1f}')
print(f'  p99:    {s.quantile(0.99):.1f}')
print(f'  >100:   {(s > 100).sum()}/{len(s)} sites')
print(f'  >500:   {(s > 500).sum()}/{len(s)} sites')
print(f'  >1000:  {(s > 1000).sum()}/{len(s)} sites')
print()

# 2. Within-site pair match rates → implied alpha
pairs = pf.read_feather(export / 'pairs.feather')
within = pairs[pairs['is_within'] == 1]
print('=== Within-site pair stats ===')
print(f'Total within-site pairs: {len(within):,}')
print(f'Overall match rate (y=0): {(within["y"] == 0).mean():.4f}')
print(f'Unique sites: {within["site_i"].nunique()}')
print()

# Per-site match rate → implied alpha = 1/match_rate
site_match = within.groupby('site_i')['y'].apply(lambda g: (g == 0).mean())
implied_alpha = 1.0 / site_match.replace(0, np.nan)
print('=== Implied alpha from within-site match rate (1/match_rate) ===')
print(f'  mean:   {implied_alpha.mean():.1f}')
print(f'  median: {implied_alpha.median():.1f}')
print(f'  min:    {implied_alpha.min():.1f}')
print(f'  max:    {implied_alpha.max():.1f}')
print(f'  p75:    {implied_alpha.quantile(0.75):.1f}')
print(f'  p95:    {implied_alpha.quantile(0.95):.1f}')
