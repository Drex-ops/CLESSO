#!/usr/bin/env python3
"""Verify corrected S_obs file format and overlap with training sites."""

import pyarrow.feather as feather
from pathlib import Path

export_dir = Path(__file__).parent.parent / 'clesso_v2/output/VAS_20260310_092634/nn_export'

# Check corrected file
corrected = feather.read_feather(export_dir / 'site_obs_richness_CORRECTED.feather')
print('CORRECTED file:')
print(f'  Columns: {list(corrected.columns)}')
print(f'  Shape: {corrected.shape}')
print(f'  S_obs: mean={corrected.S_obs.mean():.1f}, median={corrected.S_obs.median():.1f}, max={int(corrected.S_obs.max())}')
print()

# Check original file for comparison
original = feather.read_feather(export_dir / 'site_obs_richness.feather')
print('ORIGINAL file:')
print(f'  Columns: {list(original.columns)}')
print(f'  Shape: {original.shape}')
print(f'  S_obs: mean={original.S_obs.mean():.1f}, median={original.S_obs.median():.1f}, max={int(original.S_obs.max())}')
print()

# Check overlap with training sites
site_cov = feather.read_feather(export_dir / 'site_covariates.feather')
train_ids = set(site_cov.site_id)
corrected_ids = set(corrected.site_id)
original_ids = set(original.site_id)

print(f'Training sites: {len(train_ids)}')
print(f'Overlap with corrected: {len(train_ids & corrected_ids)}')
print(f'Overlap with original: {len(train_ids & original_ids)}')
print(f'Train sites missing from corrected: {len(train_ids - corrected_ids)}')
