#!/usr/bin/env python3
"""Diagnostic: alpha–beta covariate collinearity analysis.

For each shared variable that appears in both the alpha (site-level richness)
and beta (pairwise turnover) pathways, this script quantifies the structural
collinearity between site-level values and pairwise absolute differences.

Outputs:
  - Correlation matrix + heatmap (PNG)
  - VIF (Variance Inflation Factor) for each feature
  - PCA explained-variance summary and top loadings

Usage:
    python scripts/diag_collinearity.py [export_dir]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

# ── Export directory ──────────────────────────────────────────────────────
if len(sys.argv) > 1:
    export_dir = sys.argv[1]
else:
    export_dir = "nn_export"

export_dir = Path(export_dir)
assert export_dir.exists(), f"Export directory not found: {export_dir}"

# ── Load data ─────────────────────────────────────────────────────────────
print(f"Loading from {export_dir} ...")
pairs = pd.read_feather(export_dir / "pairs.feather")
site_cov = pd.read_feather(export_dir / "site_covariates.feather")
env_table = pd.read_feather(export_dir / "env_site_table.feather")
with open(export_dir / "metadata.json") as f:
    meta = json.load(f)

# ── Identify shared variables ────────────────────────────────────────────
# Alpha source: numeric columns in site_covariates (excluding site_id, lon, lat, effort)
# Beta source:  numeric columns in env_site_table  (excluding site_id)
effort_cols = set(meta.get("effort_cov_cols", []))
coord_cols = {"lon", "lat"}

alpha_candidates = {
    c for c in site_cov.columns
    if c != "site_id"
    and c not in effort_cols
    and c not in coord_cols
    and pd.api.types.is_numeric_dtype(site_cov[c])
}
beta_candidates = {
    c for c in env_table.columns
    if c != "site_id"
    and pd.api.types.is_numeric_dtype(env_table[c])
}

shared = sorted(alpha_candidates & beta_candidates)
alpha_only = sorted(alpha_candidates - beta_candidates)
beta_only = sorted(beta_candidates - alpha_candidates)

print(f"\n=== Variable assignment ===")
print(f"  Shared (alpha AND beta): {len(shared)}")
for v in shared:
    print(f"    {v}")
print(f"  Alpha-only:              {len(alpha_only)}")
for v in alpha_only:
    print(f"    {v}")
print(f"  Beta-only:               {len(beta_only)}")
for v in beta_only:
    print(f"    {v}")

if not shared:
    print("\nNo shared variables between alpha and beta — nothing to analyse.")
    sys.exit(0)

# ── Build site lookup ─────────────────────────────────────────────────────
site_ids = site_cov["site_id"].values
site_id_to_idx = {sid: i for i, sid in enumerate(site_ids)}

# Align env_table to same site order
env_aligned = env_table.set_index("site_id").reindex(site_ids)

# Raw values for shared variables (unstandardised — correlations are
# scale-invariant so standardisation happens inside the pair matrix)
alpha_vals = site_cov[shared].values.astype(np.float64)  # (n_sites, K_shared)
beta_vals = env_aligned[shared].values.astype(np.float64)

# ── Sample pairs (between-site only) ─────────────────────────────────────
between = pairs[pairs["is_within"] == 0]
MAX_PAIRS = 200_000
rng = np.random.default_rng(42)
if len(between) > MAX_PAIRS:
    sample_idx = rng.choice(len(between), MAX_PAIRS, replace=False)
    between = between.iloc[sample_idx]
n_pairs = len(between)
print(f"\n  Using {n_pairs:,} between-site pairs for analysis")

idx_i = np.array([site_id_to_idx[s] for s in between["site_i"]], dtype=np.int64)
idx_j = np.array([site_id_to_idx[s] for s in between["site_j"]], dtype=np.int64)

# ── Build combined feature matrix (pairs × 2K_shared) ────────────────────
# For each shared variable X:
#   alpha_mean = (X_i + X_j) / 2   — representative site-level signal
#   beta_diff  = |X_i - X_j|       — pairwise difference (beta input)
K = len(shared)
alpha_mean_cols = []
beta_diff_cols = []
col_names = []

for k, var in enumerate(shared):
    xi = alpha_vals[idx_i, k]
    xj = alpha_vals[idx_j, k]
    alpha_mean_cols.append((xi + xj) / 2.0)
    beta_diff_cols.append(np.abs(xi - xj))

alpha_mean_mat = np.column_stack(alpha_mean_cols)  # (n_pairs, K)
beta_diff_mat = np.column_stack(beta_diff_cols)    # (n_pairs, K)

# Standardise each column (zero mean, unit variance)
def standardise(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

alpha_mean_std = standardise(alpha_mean_mat)
beta_diff_std = standardise(beta_diff_mat)

# Combined matrix: [alpha_mean_1..K | beta_diff_1..K]
combined = np.column_stack([alpha_mean_std, beta_diff_std])
alpha_names = [f"{v}_alpha_mean" for v in shared]
beta_names = [f"{v}_beta_diff" for v in shared]
all_names = alpha_names + beta_names

# ══════════════════════════════════════════════════════════════════════════
#  Phase 1: Correlation Analysis
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CORRELATION ANALYSIS")
print("=" * 60)

corr = np.corrcoef(combined, rowvar=False)  # (2K, 2K)

# Same-variable correlations: cor(alpha_mean_k, beta_diff_k)
print("\n=== Same-variable α–β correlations ===")
print(f"  {'Variable':<25s}  {'cor(α_mean, β_diff)':>20s}")
print("  " + "-" * 47)
same_var_cors = []
for k, var in enumerate(shared):
    r = corr[k, K + k]
    same_var_cors.append(r)
    flag = " ***" if abs(r) > 0.5 else " *" if abs(r) > 0.3 else ""
    print(f"  {var:<25s}  {r:>20.4f}{flag}")

med_cor = np.median(np.abs(same_var_cors))
max_cor = np.max(np.abs(same_var_cors))
print(f"\n  Median |cor| = {med_cor:.4f},  Max |cor| = {max_cor:.4f}")

# Cross-variable: highest alpha-beta cross-correlations
print("\n=== Top 10 cross-variable α–β correlations ===")
cross = []
for a in range(K):
    for b in range(K):
        if a == b:
            continue
        cross.append((shared[a], shared[b], corr[a, K + b]))
cross.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"  {'α variable':<20s}  {'β variable':<20s}  {'cor':>8s}")
print("  " + "-" * 50)
for a_name, b_name, r in cross[:10]:
    print(f"  {a_name:<20s}  {b_name:<20s}  {r:>8.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  Phase 2: VIF
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  VARIANCE INFLATION FACTOR (VIF)")
print("=" * 60)

# VIF_j = 1 / (1 - R²_j), where R²_j is from regressing feature j on all others
from numpy.linalg import lstsq

def compute_vif(X):
    """Compute VIF for each column of X (assumed centred/scaled)."""
    n, p = X.shape
    vifs = np.empty(p)
    for j in range(p):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        # Add intercept
        others_with_int = np.column_stack([np.ones(n), others])
        coef, _, _, _ = lstsq(others_with_int, y, rcond=None)
        y_hat = others_with_int @ coef
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vifs

vifs = compute_vif(combined)

print(f"\n  {'Feature':<30s}  {'VIF':>8s}  {'Flag':>6s}")
print("  " + "-" * 48)
for i, name in enumerate(all_names):
    flag = "HIGH" if vifs[i] > 10 else "mod" if vifs[i] > 5 else ""
    print(f"  {name:<30s}  {vifs[i]:>8.2f}  {flag:>6s}")

n_high = sum(1 for v in vifs if v > 10)
n_mod = sum(1 for v in vifs if 5 < v <= 10)
print(f"\n  VIF > 10 (severe): {n_high}")
print(f"  VIF 5–10 (moderate): {n_mod}")

# ══════════════════════════════════════════════════════════════════════════
#  Phase 3: PCA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PCA — SHARED ALPHA + BETA FEATURES")
print("=" * 60)

from numpy.linalg import svd

# PCA on standardised combined matrix
U, S, Vt = svd(combined - combined.mean(axis=0), full_matrices=False)
eigenvalues = S ** 2 / (n_pairs - 1)
explained = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained)

print(f"\n  {'PC':<6s}  {'Var explained':>14s}  {'Cumulative':>12s}")
print("  " + "-" * 36)
for i in range(min(10, len(explained))):
    print(f"  PC{i+1:<3d}  {explained[i]:>14.4f}  {cumulative[i]:>12.4f}")

# Milestones
for threshold in [0.90, 0.95, 0.99]:
    n_pc = int(np.searchsorted(cumulative, threshold) + 1)
    print(f"  → {threshold:.0%} variance explained by first {n_pc} PCs")

# Top loadings for PC1–3
print("\n=== Top loadings (absolute) ===")
for pc_idx in range(min(3, Vt.shape[0])):
    loadings = Vt[pc_idx]
    order = np.argsort(np.abs(loadings))[::-1]
    print(f"\n  PC{pc_idx+1} (explains {explained[pc_idx]:.1%}):")
    for rank in range(min(6, len(order))):
        j = order[rank]
        print(f"    {all_names[j]:<30s}  {loadings[j]:>+8.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  Phase 4: Heatmap
# ══════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(max(8, K * 0.7 + 2), max(7, K * 0.7 + 1)))

    # Reorder: alpha block then beta block — highlight the cross-block
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr, cmap="RdBu_r", norm=norm, aspect="equal")

    ax.set_xticks(range(2 * K))
    ax.set_yticks(range(2 * K))
    ax.set_xticklabels(all_names, rotation=90, fontsize=7)
    ax.set_yticklabels(all_names, fontsize=7)

    # Draw block separators
    ax.axhline(K - 0.5, color="black", linewidth=1.5)
    ax.axvline(K - 0.5, color="black", linewidth=1.5)
    ax.set_title("α (site-level mean) vs β (pairwise |diff|) correlation", fontsize=10)

    # Block labels
    ax.text(K / 2, -1.5, "α_mean", ha="center", fontsize=9, fontweight="bold")
    ax.text(K + K / 2, -1.5, "β_diff", ha="center", fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")

    # Annotate same-variable cells in the cross-block
    for k in range(K):
        ax.text(K + k, k, f"{corr[k, K+k]:.2f}", ha="center", va="center",
                fontsize=6, fontweight="bold",
                color="white" if abs(corr[k, K+k]) > 0.6 else "black")

    fig.tight_layout()
    out_path = export_dir / "diag_collinearity.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  Heatmap saved → {out_path}")
    plt.close(fig)

except ImportError:
    print("\n  [matplotlib not available — skipping heatmap]")

print("\nDone.")
