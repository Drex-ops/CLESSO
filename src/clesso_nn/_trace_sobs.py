#!/usr/bin/env python3
"""Trace S_obs from raw ALA data through the pipeline to training data."""
import sys
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
from collections import Counter

print("=" * 70)
print("  Tracing S_obs: Raw ALA → Exported Training Data")
print("=" * 70)

# ── 1. Raw ALA data ──────────────────────────────────────────────────
print("\n[1] Raw ALA data (ala_vas_2026-03-03.csv)...")
# Read a chunk to be fast — just need structure
ala = pd.read_csv("data/ala_vas_2026-03-03.csv", low_memory=False)
n_raw = len(ala)
print(f"  Total records: {n_raw:,}")
print(f"  Columns: {list(ala.columns)}")
print(f"  Unique species: {ala['scientificName'].nunique():,}")
print(f"  Date range: {ala['eventDate'].min()} → {ala['eventDate'].max()}")
print(f"  Lat range: [{ala['decimalLatitude'].min():.4f}, {ala['decimalLatitude'].max():.4f}]")
print(f"  Lon range: [{ala['decimalLongitude'].min():.4f}, {ala['decimalLongitude'].max():.4f}]")

# Aggregate to 0.05° grid cells (matching CLESSO site_id format: "lon:lat")
ala["lon_r"] = (ala["decimalLongitude"] / 0.05).round() * 0.05
ala["lat_r"] = (ala["decimalLatitude"] / 0.05).round() * 0.05
ala["site_id"] = ala["lon_r"].round(2).astype(str) + ":" + ala["lat_r"].round(2).astype(str)

# Species per cell (raw, no filtering)
raw_richness = ala.groupby("site_id")["scientificName"].nunique()
print(f"\n  Grid cells (raw): {len(raw_richness):,}")
print(f"  Species per cell (raw ALA):")
print(f"    Mean:    {raw_richness.mean():.1f}")
print(f"    Median:  {raw_richness.median():.1f}")
print(f"    Max:     {raw_richness.max()}")
for p in [75, 90, 95, 99]:
    print(f"    {p}th:    {raw_richness.quantile(p/100):.0f}")

# Show top cells
print(f"\n  Top 20 cells by species count (raw ALA):")
top20 = raw_richness.nlargest(20)
for site, n in top20.items():
    print(f"    {site}: {n} species")

# ── 2. Exported S_obs ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[2] Exported S_obs (site_obs_richness.feather)...")
export_dir = Path("src/clesso_v2/output/VAS_20260310_092634/nn_export")
rich_df = feather.read_feather(export_dir / "site_obs_richness.feather")
print(f"  Sites with S_obs: {len(rich_df):,}")
print(f"  S_obs distribution:")
print(f"    Mean:    {rich_df['S_obs'].mean():.2f}")
print(f"    Median:  {rich_df['S_obs'].median():.2f}")
print(f"    Max:     {rich_df['S_obs'].max():.0f}")
for p in [75, 90, 95, 99]:
    print(f"    {p}th:    {rich_df['S_obs'].quantile(p/100):.1f}")

# ── 3. Match exported site_ids with raw ALA ──────────────────────────
print("\n" + "=" * 70)
print("[3] Comparing site_ids...")
export_sites = set(rich_df["site_id"])
raw_sites = set(raw_richness.index)
overlap = export_sites & raw_sites
print(f"  Exported sites: {len(export_sites):,}")
print(f"  Raw ALA sites: {len(raw_sites):,}")
print(f"  Overlap: {len(overlap):,}")

# For overlapping sites, compare richness
if overlap:
    compare = []
    for s in list(overlap)[:5000]:  # sample
        compare.append({
            "site_id": s,
            "raw_ALA_richness": raw_richness[s],
            "exported_S_obs": rich_df.loc[rich_df["site_id"] == s, "S_obs"].values[0]
        })
    comp_df = pd.DataFrame(compare)
    comp_df["ratio"] = comp_df["raw_ALA_richness"] / comp_df["exported_S_obs"]
    print(f"\n  Richness comparison (n={len(comp_df)}):")
    print(f"    Raw ALA mean:   {comp_df['raw_ALA_richness'].mean():.1f}")
    print(f"    Exported mean:  {comp_df['exported_S_obs'].mean():.2f}")
    print(f"    Mean ratio:     {comp_df['ratio'].mean():.2f}")
    print(f"    Median ratio:   {comp_df['ratio'].median():.2f}")

    # Show worst mismatches
    comp_df["diff"] = comp_df["raw_ALA_richness"] - comp_df["exported_S_obs"]
    biggest = comp_df.nlargest(10, "diff")
    print(f"\n  Biggest gaps (raw ALA >> exported S_obs):")
    print(biggest[["site_id", "raw_ALA_richness", "exported_S_obs", "ratio"]].to_string(index=False))

# ── 4. Check what obs_dt looks like (the filtered data) ─────────────
print("\n" + "=" * 70)
print("[4] Checking filtered/processed observation data...")

# Look for the filtered data
filtered_path = Path("data/filtered_data_2018-11-20.csv")
if filtered_path.exists():
    filt = pd.read_csv(filtered_path, nrows=5)
    print(f"  filtered_data columns: {list(filt.columns)}")
    filt_full = pd.read_csv(filtered_path, low_memory=False)
    print(f"  filtered_data rows: {len(filt_full):,}")
else:
    print("  filtered_data_2018-11-20.csv not found")

# Check the pairs to see how many unique species appear per site
print("\n" + "=" * 70)
print("[5] Within-site species diversity from pairs...")
pairs = feather.read_feather(export_dir / "pairs.feather")
within = pairs[pairs["is_within"] == True]
print(f"  Within-site pairs: {len(within):,}")

# Count unique species per site from within-site pairs
species_per_site_i = within.groupby("site_i")["species_i"].nunique().rename("n_species_from_pairs")
print(f"  Sites with within-pairs: {len(species_per_site_i):,}")
print(f"  Species per site (from pairs):")
print(f"    Mean:    {species_per_site_i.mean():.2f}")
print(f"    Median:  {species_per_site_i.median():.2f}")
print(f"    Max:     {species_per_site_i.max()}")
for p in [75, 90, 95, 99]:
    print(f"    {p}th:    {species_per_site_i.quantile(p/100):.0f}")

# Check S_obs = uniqueN(species) — does this match what pairs show?
# The pairs file has species_i and species_j
# For within-site pairs, both should be at the same site
# S_obs should be the # of unique species at each site
# Let's verify
sample_site = within["site_i"].value_counts().idxmax()
site_pairs = within[within["site_i"] == sample_site]
unique_sp = set(site_pairs["species_i"].unique()) | set(site_pairs["species_j"].unique())
s_obs_val = rich_df.loc[rich_df["site_id"] == sample_site, "S_obs"].values
print(f"\n  Sample site: {sample_site}")
print(f"    Unique species from pairs: {len(unique_sp)}")
print(f"    S_obs in export: {s_obs_val}")
print(f"    N within-pairs at this site: {len(site_pairs)}")

# ── 6. Check the R export logic for S_obs computation ───────────────
print("\n" + "=" * 70)
print("[6] How S_obs is really computed...")
print("  In export_for_nn.R line 123:")
print('    site_obs_richness <- obs_dt[, .(S_obs = uniqueN(species)), by = .(site_id)]')
print("  Where obs_dt is the observation-level data BEFORE pair sampling")
print("  Key question: is obs_dt the same as the raw ALA data, or is it filtered?")
print("  If filtered/subsampled, that explains the low S_obs values")

# Check S_obs values — fractional S_obs?
frac = rich_df[rich_df["S_obs"] != rich_df["S_obs"].astype(int)]
print(f"\n  Fractional S_obs values: {len(frac)} ({100*len(frac)/len(rich_df):.1f}%)")
if len(frac) > 0:
    print("  Examples:")
    print(frac.head(10))
    print("  This suggests S_obs may be computed differently (e.g., average across sub-samples)")
