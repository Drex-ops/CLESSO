#!/usr/bin/env python3
"""Quick data analysis for beta collapse debugging."""
import json, numpy as np, pyarrow.feather as feather
from pathlib import Path

export_dir = Path('src/clesso_v2/output/VAS_20260310_092634/nn_export')
print('Export dir:', export_dir)

pairs = feather.read_feather(export_dir / 'pairs.feather')
n = len(pairs)
n_within = (pairs['is_within'] == 1).sum()
n_between = (pairs['is_within'] == 0).sum()
print(f'Total pairs: {n:,}')
print(f'Within-site: {n_within:,} ({n_within/n*100:.1f}%)')
print(f'Between-site: {n_between:,} ({n_between/n*100:.1f}%)')
print()

y_between = pairs.loc[pairs['is_within']==0, 'y']
print('Between-site y distribution:')
print(f'  y=0 (match): {(y_between==0).sum():,} ({(y_between==0).mean()*100:.1f}%)')
print(f'  y=1 (mismatch): {(y_between==1).sum():,} ({(y_between==1).mean()*100:.1f}%)')
print()

env = feather.read_feather(export_dir / 'env_site_table.feather')
env_cols = [c for c in env.columns if c != 'site_id']
print('Env columns:', env_cols)
print()
for c in env_cols:
    vals = env[c].dropna()
    print(f'  {c}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}')

with open(export_dir / 'metadata.json') as f:
    meta = json.load(f)
print()
print('K_env:', meta.get('K_env'))
print('K_alpha:', meta.get('K_alpha'))
