#!/usr/bin/env python3
"""Compute S_obs correctly from raw ALA data and compare with exported S_obs."""
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path

print("=" * 70)
print("  Recomputing S_obs from Raw ALA Data")
print("=" * 70)

# ── 1. Load raw ALA ──────────────────────────────────────────────────
print("\n[1] Loading raw ALA data...")
ala = pd.read_csv("data/ala_vas_2026-03-03.csv", low_memory=False)
print(f"  Records: {len(ala):,}")

# Apply date filter matching metadata: 1970-01-01 to 2018-01-01
ala["eventDate"] = pd.to_datetime(ala["eventDate"], errors="coerce")
ala = ala[(ala["eventDate"] >= "1970-01-01") & (ala["eventDate"] < "2018-01-01")]
print(f"  After date filter (1970–2018): {len(ala):,}")

# Aggregate to 0.05° grid
ala["lon_r"] = (ala["decimalLongitude"] / 0.05).round() * 0.05
ala["lat_r"] = (ala["decimalLatitude"] / 0.05).round() * 0.05
ala["site_id"] = ala["lon_r"].round(2).astype(str) + ":" + ala["lat_r"].round(2).astype(str)

# Compute TRUE observed richness = uniqueN(species) per site
true_richness = ala.groupby("site_id")["scientificName"].nunique().reset_index()
true_richness.columns = ["site_id", "S_obs_true"]
print(f"  Grid cells: {len(true_richness):,}")

print(f"\n  TRUE S_obs distribution:")
print(f"    Mean:    {true_richness['S_obs_true'].mean():.1f}")
print(f"    Median:  {true_richness['S_obs_true'].median():.0f}")
print(f"    Max:     {true_richness['S_obs_true'].max()}")
for p in [75, 90, 95, 99]:
    print(f"    {p}th:    {true_richness['S_obs_true'].quantile(p/100):.0f}")

# ── 2. Compare with exported S_obs ───────────────────────────────────
print("\n[2] Comparing with exported S_obs...")
export_dir = Path("src/clesso_v2/output/VAS_20260310_092634/nn_export")
exported = feather.read_feather(export_dir / "site_obs_richness.feather")
print(f"  Exported sites: {len(exported):,}")

merged = true_richness.merge(exported, on="site_id", how="inner")
print(f"  Matched sites: {len(merged):,}")

merged["ratio"] = merged["S_obs_true"] / merged["S_obs"]
print(f"\n  Comparison (n={len(merged)}):")
print(f"    Mean TRUE S_obs:     {merged['S_obs_true'].mean():.1f}")
print(f"    Mean exported S_obs: {merged['S_obs'].mean():.2f}")
print(f"    Mean ratio (true/exported): {merged['ratio'].mean():.1f}x")
print(f"    Median ratio: {merged['ratio'].median():.1f}x")

# ── 3. Check where the discrepancy comes from ────────────────────────
print("\n[3] Sample of high-discrepancy sites...")
top_disc = merged.nlargest(20, "ratio")
print(f"  {'site_id':<20} {'S_obs_true':>12} {'S_obs_export':>13} {'ratio':>8}")
for _, row in top_disc.iterrows():
    print(f"  {row['site_id']:<20} {row['S_obs_true']:>12} {row['S_obs']:>13.2f} {row['ratio']:>8.1f}")

# ── 4. Correlation between TRUE and exported ────────────────────────
print(f"\n[4] Correlation between TRUE and exported S_obs:")
corr = merged["S_obs_true"].corr(merged["S_obs"])
print(f"    Pearson r: {corr:.4f}")

# ── 5. What sites have S_obs == 1 in export but many species in truth?
print("\n[5] Sites with exported S_obs ≈ 1 but many true species:")
low_export = merged[(merged["S_obs"] < 1.5) & (merged["S_obs_true"] > 50)]
print(f"  Found {len(low_export)} such sites")
if len(low_export) > 0:
    print(low_export[["site_id", "S_obs_true", "S_obs", "ratio"]].head(10).to_string(index=False))

# ── 6. Re-export TRUE richness ───────────────────────────────────────
print("\n[6] Saving CORRECTED S_obs...")
output_path = export_dir / "site_obs_richness_CORRECTED.feather"
corrected = true_richness.rename(columns={"S_obs_true": "S_obs"})
feather.write_feather(corrected, output_path)
print(f"  Written: {output_path}")
print(f"  Sites: {len(corrected):,}")
print(f"  Mean S_obs: {corrected['S_obs'].mean():.1f}")
