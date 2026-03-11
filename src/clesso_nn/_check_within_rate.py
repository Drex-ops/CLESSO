#!/usr/bin/env python3
"""Check within-site match rate and what it implies for alpha."""
import sys
import torch
import numpy as np
import pyarrow.feather as feather
from pathlib import Path

export_dir = Path("src/clesso_v2/output/VAS_20260310_092634/nn_export")

# Load pairs
pairs = feather.read_feather(export_dir / "pairs.feather")

within = pairs[pairs["is_within"] == True]
between = pairs[pairs["is_within"] == False]

print(f"Total pairs: {len(pairs)}")
print(f"Within-site: {len(within)}")
print(f"Between-site: {len(between)}")

print(f"\n--- Within-site y values ---")
print(within["y"].value_counts().sort_index())
# y=0 = match, y=1 = mismatch

n_within_match = (within["y"] == 0).sum()
n_within_mismatch = (within["y"] == 1).sum()
print(f"\n  matches: {n_within_match}")
print(f"  mismatches: {n_within_mismatch}")
if len(within) > 0:
    match_rate = n_within_match / len(within)
    print(f"  match rate: {match_rate:.4f}")
    print(f"  → implied alpha = 1/match_rate = {1.0 / match_rate:.2f}")

print(f"\n--- Between-site y values ---")
print(between["y"].value_counts().sort_index())

print(f"\n--- Check species columns ---")
print(f"  Columns: {list(pairs.columns)}")
print(f"\n  Within-site samples:")
print(within[["site_i", "site_j", "species_i", "species_j", "y"]].head(20))

# Check per-site within match rate
print(f"\n--- Per-site within-site match rate ---")
within_sites = within.groupby("site_i")
match_rates = []
for site, grp in within_sites:
    mr = (grp["y"] == 0).mean()
    match_rates.append({"site": site, "n_pairs": len(grp), "match_rate": mr, "implied_alpha": 1.0 / max(mr, 0.001)})

import pandas as pd
mr_df = pd.DataFrame(match_rates)
print(f"  N sites with within pairs: {len(mr_df)}")
print(f"\n  Match rate distribution:")
print(mr_df["match_rate"].describe())
print(f"\n  Implied alpha distribution:")
print(mr_df["implied_alpha"].describe())
print(f"\n  Top 20 by implied alpha:")
top = mr_df.nlargest(20, "implied_alpha")
print(top.to_string(index=False))

# Compare with S_obs
rich_df = feather.read_feather(export_dir / "site_obs_richness.feather")
mr_df = mr_df.merge(rich_df.rename(columns={"site_id": "site"}), on="site", how="left")
print(f"\n--- Match-rate alpha vs S_obs ---")
print(f"  Correlation: {mr_df['implied_alpha'].corr(mr_df['S_obs']):.4f}")
print(f"  Mean ratio (implied/S_obs): {(mr_df['implied_alpha'] / mr_df['S_obs']).mean():.3f}")
print(f"  Median ratio: {(mr_df['implied_alpha'] / mr_df['S_obs']).median():.3f}")
