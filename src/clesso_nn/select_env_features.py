#!/usr/bin/env python3
"""
select_env_features.py — Pre-processing feature selection for CLESSO NN.

Runs a battery of diagnostics on the current env features in nn_export to
identify the best non-collinear, informative feature set.

Usage:
    # Full report on current env features
    python select_env_features.py report

    # Full report including candidate features from geonpy (wider scan)
    python select_env_features.py report --scan-geonpy

    # Recommend a non-collinear subset (prints manage_env_features.py commands)
    python select_env_features.py recommend

Steps performed:
    1. NaN / coverage check — flags features with high missingness
    2. Near-zero variance filter — flags near-constant features
    3. Pairwise correlation matrix — flags highly correlated pairs
    4. VIF (Variance Inflation Factor) — iteratively removes collinear features
    5. Univariate relevance — ranks features by their |env_diff| correlation
       with compositional turnover (y) in the training pairs
    6. Recommended feature set — non-collinear + informative

Outputs a PDF diagnostics report to the nn_export directory plus a
text summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXPORT_DIR = (
    PROJECT_ROOT / "src" / "clesso_v2" / "output"
    / "VAS_20260310_092634" / "nn_export"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_env_data(export_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load env_site_table, site_covariates, and metadata."""
    est = pf.read_feather(export_dir / "env_site_table.feather")
    sc = pf.read_feather(export_dir / "site_covariates.feather")
    with open(export_dir / "metadata.json") as f:
        meta = json.load(f)
    return est, sc, meta


def get_env_matrix(est: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extract numeric env columns from env_site_table as a DataFrame."""
    env_cols = [c for c in est.columns
                if c != "site_id" and pd.api.types.is_numeric_dtype(est[c])
                and not c.startswith("subs_")]
    climate_cols = env_cols
    subs_cols = [c for c in est.columns if c.startswith("subs_")]
    all_cols = subs_cols + climate_cols
    return est[all_cols], all_cols


def load_pairs_sample(export_dir: Path, n_sample: int = 200_000,
                      seed: int = 42) -> pd.DataFrame:
    """Load a random sample of training pairs."""
    pairs = pf.read_feather(export_dir / "pairs.feather")
    if len(pairs) > n_sample:
        pairs = pairs.sample(n=n_sample, random_state=seed)
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# 2. Diagnostic functions
# ═══════════════════════════════════════════════════════════════════════════

def check_coverage(X: pd.DataFrame) -> pd.DataFrame:
    """Check NaN rates and basic stats per feature.

    Returns DataFrame with columns: n_nan, pct_nan, mean, std, min, max.
    """
    rows = []
    for col in X.columns:
        vals = X[col].values
        n_nan = int(np.isnan(vals).sum())
        valid = vals[~np.isnan(vals)]
        rows.append({
            "feature": col,
            "n_nan": n_nan,
            "pct_nan": 100.0 * n_nan / len(vals),
            "n_valid": len(valid),
            "mean": float(np.mean(valid)) if len(valid) else np.nan,
            "std": float(np.std(valid)) if len(valid) else np.nan,
            "min": float(np.min(valid)) if len(valid) else np.nan,
            "max": float(np.max(valid)) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows).set_index("feature")


def check_near_zero_variance(X: pd.DataFrame,
                              freq_ratio_thresh: float = 19.0,
                              unique_pct_thresh: float = 10.0) -> pd.DataFrame:
    """Identify near-zero variance features.

    Uses caret-style criteria:
        - freq_ratio: ratio of most-common to second-most-common value (high → bad)
        - unique_pct: number of unique values as % of total (low → bad)

    A feature is flagged if freq_ratio > thresh AND unique_pct < thresh.
    """
    rows = []
    for col in X.columns:
        vals = X[col].dropna().values
        n = len(vals)
        unique_vals, counts = np.unique(vals, return_counts=True)
        n_unique = len(unique_vals)
        unique_pct = 100.0 * n_unique / n if n > 0 else 0

        if len(counts) >= 2:
            sorted_counts = np.sort(counts)[::-1]
            freq_ratio = sorted_counts[0] / max(sorted_counts[1], 1)
        else:
            freq_ratio = float("inf")

        flagged = (freq_ratio > freq_ratio_thresh) and (unique_pct < unique_pct_thresh)

        rows.append({
            "feature": col,
            "n_unique": n_unique,
            "unique_pct": unique_pct,
            "freq_ratio": freq_ratio,
            "std": float(np.std(vals)),
            "nzv_flag": flagged,
        })
    return pd.DataFrame(rows).set_index("feature")


def compute_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation (NaN-safe)."""
    X_filled = X.fillna(X.mean())
    return X_filled.corr(method="pearson")


def find_correlated_pairs(corr: pd.DataFrame,
                           threshold: float = 0.85) -> pd.DataFrame:
    """Find all feature pairs with |r| > threshold."""
    cols = corr.columns.tolist()
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                rows.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "abs_corr": abs(r),
                    "corr": r,
                })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return df


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    VIF_k = 1 / (1 - R²_k)  where R²_k is from regressing feature k
    on all other features.

    Uses numpy least-squares directly (no statsmodels dependency).
    """
    X_filled = X.fillna(X.mean())
    X_std = (X_filled - X_filled.mean()) / X_filled.std().replace(0, 1)
    vals = X_std.values.astype(np.float64)
    n, p = vals.shape

    vifs = []
    for k in range(p):
        y_k = vals[:, k]
        X_others = np.delete(vals, k, axis=1)
        X_others = np.column_stack([np.ones(n), X_others])

        # Least squares: y_k ~ X_others
        try:
            coef, residuals, rank, sv = np.linalg.lstsq(X_others, y_k, rcond=None)
            y_pred = X_others @ coef
            ss_res = np.sum((y_k - y_pred) ** 2)
            ss_tot = np.sum((y_k - y_k.mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            vif = 1.0 / max(1.0 - r2, 1e-12)
        except np.linalg.LinAlgError:
            vif = float("inf")
        vifs.append(vif)

    return pd.DataFrame({"feature": X.columns, "VIF": vifs}).set_index("feature")


def iterative_vif_removal(X: pd.DataFrame,
                           vif_threshold: float = 10.0,
                           max_iter: int = 50) -> tuple[list[str], list[str]]:
    """Iteratively remove the highest-VIF feature until all VIF < threshold.

    Returns (kept_features, removed_features_in_order).
    """
    remaining = list(X.columns)
    removed = []

    for _ in range(max_iter):
        if len(remaining) <= 2:
            break
        vif_df = compute_vif(X[remaining])
        max_vif_feat = vif_df["VIF"].idxmax()
        max_vif_val = vif_df["VIF"].max()

        if max_vif_val <= vif_threshold:
            break
        removed.append((max_vif_feat, max_vif_val))
        remaining.remove(max_vif_feat)

    return remaining, removed


# ═══════════════════════════════════════════════════════════════════════════
# 3. Univariate relevance to turnover
# ═══════════════════════════════════════════════════════════════════════════

def univariate_relevance(est: pd.DataFrame, pairs: pd.DataFrame,
                          env_cols: list[str]) -> pd.DataFrame:
    """Rank features by how well |env_diff| predicts match/mismatch.

    For each env variable k, computes:
        d_k = |standardised_env_i_k - standardised_env_j_k|
    Then measures:
        - Point-biserial correlation with y (1=mismatch, 0=match)
        - AUC of d_k alone as a classifier for y
        - Mann-Whitney U p-value between match vs mismatch distributions
    """
    # Build site_id → index
    site_ids = est["site_id"].values
    id_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    # Standardise env
    subs_cols = [c for c in env_cols if c.startswith("subs_")]
    climate_cols = [c for c in env_cols if not c.startswith("subs_")]
    all_cols = subs_cols + climate_cols

    E = est[all_cols].values.astype(np.float64)
    e_mean = np.nanmean(E, axis=0)
    e_std = np.nanstd(E, axis=0)
    e_std[e_std == 0] = 1.0
    E_std = (E - e_mean) / e_std
    np.nan_to_num(E_std, copy=False, nan=0.0)

    # Map pair sites to indices
    valid_mask = pairs["site_i"].isin(id_to_idx) & pairs["site_j"].isin(id_to_idx)
    pairs_v = pairs[valid_mask].copy()
    idx_i = np.array([id_to_idx[s] for s in pairs_v["site_i"]])
    idx_j = np.array([id_to_idx[s] for s in pairs_v["site_j"]])
    y = pairs_v["y"].values.astype(np.float64)

    # Only use between-site pairs for relevance (within-site → env_diff=0 by design)
    between = pairs_v["is_within"].values == 0
    idx_i_b = idx_i[between]
    idx_j_b = idx_j[between]
    y_b = y[between]

    rows = []
    for k, col in enumerate(all_cols):
        d_k = np.abs(E_std[idx_i_b, k] - E_std[idx_j_b, k])

        # Point-biserial correlation (y_b ∈ {0,1}, d_k continuous)
        r_pb, p_pb = sp_stats.pointbiserialr(y_b, d_k)

        # AUC (higher d_k should predict mismatch=1)
        try:
            auc = roc_auc_score(y_b, d_k)
        except ValueError:
            auc = 0.5

        # Mann-Whitney: are mismatches' d_k > matches' d_k?
        d_match = d_k[y_b == 0]
        d_mismatch = d_k[y_b == 1]
        try:
            u_stat, p_mw = sp_stats.mannwhitneyu(d_mismatch, d_match,
                                                  alternative="greater")
        except ValueError:
            p_mw = 1.0

        # Mean difference
        mean_match = np.mean(d_match)
        mean_mismatch = np.mean(d_mismatch)

        rows.append({
            "feature": col,
            "type": "substrate" if col.startswith("subs_") else "climate",
            "r_pointbiserial": r_pb,
            "auc": auc,
            "mw_pvalue": p_mw,
            "mean_d_match": mean_match,
            "mean_d_mismatch": mean_mismatch,
            "ratio_mismatch_match": mean_mismatch / max(mean_match, 1e-12),
        })

    df = pd.DataFrame(rows).set_index("feature")
    df = df.sort_values("auc", ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Multivariate importance via Random Forest
# ═══════════════════════════════════════════════════════════════════════════

def rf_importance(est: pd.DataFrame, pairs: pd.DataFrame,
                   env_cols: list[str],
                   n_sample: int = 100_000,
                   seed: int = 42) -> pd.DataFrame:
    """Quick Random Forest on |env_diff| → y to get feature importances.

    Uses a small RF (100 trees, max_depth=8) on a subsample of between-site
    pairs. Returns feature importances (Gini + permutation).
    """
    # Build index
    site_ids = est["site_id"].values
    id_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    subs_cols = [c for c in env_cols if c.startswith("subs_")]
    climate_cols = [c for c in env_cols if not c.startswith("subs_")]
    all_cols = subs_cols + climate_cols

    E = est[all_cols].values.astype(np.float64)
    e_mean = np.nanmean(E, axis=0)
    e_std = np.nanstd(E, axis=0)
    e_std[e_std == 0] = 1.0
    E_std = (E - e_mean) / e_std
    np.nan_to_num(E_std, copy=False, nan=0.0)

    # Between-site pairs only
    valid = pairs["site_i"].isin(id_to_idx) & pairs["site_j"].isin(id_to_idx)
    pairs_v = pairs[valid].copy()
    between = pairs_v["is_within"].values == 0
    pairs_b = pairs_v[between]

    if len(pairs_b) > n_sample:
        pairs_b = pairs_b.sample(n=n_sample, random_state=seed)

    idx_i = np.array([id_to_idx[s] for s in pairs_b["site_i"]])
    idx_j = np.array([id_to_idx[s] for s in pairs_b["site_j"]])
    y = pairs_b["y"].values.astype(int)

    X_diff = np.abs(E_std[idx_i] - E_std[idx_j]).astype(np.float32)

    # Train RF
    print("  Training Random Forest for importance estimation...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=50,
        n_jobs=-1, random_state=seed, class_weight="balanced",
    )
    rf.fit(X_diff, y)

    # Gini importance
    gini_imp = rf.feature_importances_

    # Quick OOB-style AUC via 3-fold CV
    print("  Running 3-fold CV for AUC and permutation importance...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    aucs = []
    perm_importances = np.zeros(len(all_cols))

    for train_idx, test_idx in cv.split(X_diff, y):
        rf_cv = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=50,
            n_jobs=-1, random_state=seed, class_weight="balanced",
        )
        rf_cv.fit(X_diff[train_idx], y[train_idx])
        prob = rf_cv.predict_proba(X_diff[test_idx])[:, 1]
        base_auc = roc_auc_score(y[test_idx], prob)
        aucs.append(base_auc)

        # Permutation importance
        for k in range(X_diff.shape[1]):
            X_perm = X_diff[test_idx].copy()
            rng = np.random.default_rng(seed + k)
            X_perm[:, k] = rng.permutation(X_perm[:, k])
            perm_prob = rf_cv.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score(y[test_idx], perm_prob)
            perm_importances[k] += (base_auc - perm_auc) / 3

    print(f"  RF 3-fold AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    df = pd.DataFrame({
        "feature": all_cols,
        "gini_importance": gini_imp,
        "perm_importance_auc_drop": perm_importances,
    }).set_index("feature")
    df = df.sort_values("perm_importance_auc_drop", ascending=False)
    return df, np.mean(aucs)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Recommendation engine
# ═══════════════════════════════════════════════════════════════════════════

def recommend_features(
    coverage: pd.DataFrame,
    nzv: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    vif_kept: list[str],
    relevance: pd.DataFrame,
    rf_imp: pd.DataFrame | None,
    max_features: int = 15,
    nan_threshold: float = 5.0,
    nzv_remove: bool = True,
    corr_threshold: float = 0.85,
) -> tuple[list[str], pd.DataFrame]:
    """Combine all diagnostics to produce a ranked recommended feature set.

    Strategy:
        1. Remove features with >nan_threshold% NaN
        2. Remove near-zero variance features (if flagged and nzv_remove=True)
        3. Filter to VIF-surviving features (non-collinear set)
        4. Rank by composite score = mean rank across AUC + r_pointbiserial + perm_importance
        5. Return top max_features
    """
    all_feats = list(coverage.index)

    # Step 1: Remove high-NaN
    high_nan = set(coverage[coverage["pct_nan"] > nan_threshold].index)

    # Step 2: Remove NZV
    nzv_feats = set(nzv[nzv["nzv_flag"]].index) if nzv_remove else set()

    # Step 3: VIF survivors
    vif_set = set(vif_kept)

    # Build candidate set: must pass NaN + NZV + VIF
    candidates = [f for f in all_feats
                  if f not in high_nan
                  and f not in nzv_feats
                  and f in vif_set]

    # Step 4: Rank by composite score
    # We use the relevance AUC + permutation importance AUC drop
    rank_df = pd.DataFrame(index=candidates)

    if len(candidates) == 0:
        return [], rank_df

    # AUC rank (higher = better)
    if "auc" in relevance.columns:
        rel_sub = relevance.reindex(candidates)
        rank_df["auc"] = rel_sub["auc"]
        rank_df["auc_rank"] = rank_df["auc"].rank(ascending=False)
    else:
        rank_df["auc_rank"] = 1

    # Point-biserial rank
    if "r_pointbiserial" in relevance.columns:
        rel_sub = relevance.reindex(candidates)
        rank_df["r_pb"] = rel_sub["r_pointbiserial"].abs()
        rank_df["r_pb_rank"] = rank_df["r_pb"].rank(ascending=False)
    else:
        rank_df["r_pb_rank"] = 1

    # Permutation importance rank
    if rf_imp is not None and "perm_importance_auc_drop" in rf_imp.columns:
        imp_sub = rf_imp.reindex(candidates)
        rank_df["perm_imp"] = imp_sub["perm_importance_auc_drop"]
        rank_df["perm_imp_rank"] = rank_df["perm_imp"].rank(ascending=False)
    else:
        rank_df["perm_imp_rank"] = 1

    # Composite rank (lower = better)
    rank_df["composite_rank"] = (
        rank_df["auc_rank"] + rank_df["r_pb_rank"] + rank_df["perm_imp_rank"]
    ) / 3
    rank_df = rank_df.sort_values("composite_rank")

    recommended = list(rank_df.index[:max_features])
    return recommended, rank_df


# ═══════════════════════════════════════════════════════════════════════════
# 6. PDF report generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_report_pdf(
    export_dir: Path,
    coverage: pd.DataFrame,
    nzv: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    vif_df: pd.DataFrame,
    vif_kept: list[str],
    vif_removed: list,
    relevance: pd.DataFrame,
    rf_imp: pd.DataFrame | None,
    rf_auc: float | None,
    recommended: list[str],
    rank_df: pd.DataFrame,
):
    """Write a multi-page PDF with feature selection diagnostics."""
    pdf_path = export_dir / "feature_selection_report.pdf"
    print(f"\n  Writing PDF report: {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        # --- Page 1: Summary ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        lines = [
            "CLESSO NN — Feature Selection Report",
            "=" * 50,
            f"Export directory: {export_dir}",
            f"Total features analysed: {len(coverage)}",
            f"  Substrate: {sum(1 for c in coverage.index if c.startswith('subs_'))}",
            f"  Climate:   {sum(1 for c in coverage.index if not c.startswith('subs_'))}",
            "",
            f"Near-zero variance flagged: {nzv['nzv_flag'].sum()}",
            f"High NaN (>5%):            {(coverage['pct_nan'] > 5).sum()}",
            f"Correlated pairs (|r|>0.85): {len(corr_pairs)}",
            f"VIF survivors (VIF<10):    {len(vif_kept)}/{len(coverage)}",
            "",
            f"RF baseline AUC (3-fold):  {rf_auc:.4f}" if rf_auc else "",
            "",
            "Recommended features:",
        ]
        for i, feat in enumerate(recommended, 1):
            auc_val = relevance.loc[feat, "auc"] if feat in relevance.index else "?"
            lines.append(f"  {i:2d}. {feat:<30s} (AUC={auc_val:.4f})" if isinstance(auc_val, float) else f"  {i:2d}. {feat}")
        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 2: Coverage table ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Feature Coverage & Basic Statistics", fontsize=12, fontweight="bold", loc="left")
        table_data = coverage[["n_nan", "pct_nan", "mean", "std", "min", "max"]].round(3)
        table = ax.table(
            cellText=table_data.values,
            rowLabels=table_data.index,
            colLabels=table_data.columns,
            cellLoc="right", rowLoc="left", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(range(len(table_data.columns) + 1))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 3: Correlation heatmap ---
        fig, ax = plt.subplots(figsize=(11, 9))
        n = len(corr_matrix)
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr_matrix.index, fontsize=7)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = corr_matrix.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        ax.set_title("Pairwise Correlation Matrix", fontsize=12, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 4: VIF bar chart ---
        fig, ax = plt.subplots(figsize=(11, 6))
        vif_sorted = vif_df.sort_values("VIF", ascending=True)
        colors = ["#2196F3" if f in vif_kept else "#EF5350"
                  for f in vif_sorted.index]
        bars = ax.barh(range(len(vif_sorted)), vif_sorted["VIF"].clip(upper=100),
                       color=colors)
        ax.set_yticks(range(len(vif_sorted)))
        ax.set_yticklabels(vif_sorted.index, fontsize=8)
        ax.axvline(10, color="red", linestyle="--", alpha=0.7, label="VIF = 10 threshold")
        ax.set_xlabel("VIF (clipped at 100)")
        ax.set_title("Variance Inflation Factor (blue=kept, red=removed)", fontsize=12, fontweight="bold")
        ax.legend()
        # Annotate values
        for i, (feat, row) in enumerate(vif_sorted.iterrows()):
            val = row["VIF"]
            ax.text(min(val, 100) + 0.5, i, f"{val:.1f}", va="center", fontsize=7)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 5: Univariate relevance ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 7))

        # AUC bar chart
        ax = axes[0]
        rel_sorted = relevance.sort_values("auc", ascending=True)
        colors_rel = ["#4CAF50" if f in recommended else "#9E9E9E"
                      for f in rel_sorted.index]
        ax.barh(range(len(rel_sorted)), rel_sorted["auc"], color=colors_rel)
        ax.set_yticks(range(len(rel_sorted)))
        ax.set_yticklabels(rel_sorted.index, fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("AUC (|env_diff_k| → y)")
        ax.set_title("Univariate AUC", fontsize=10, fontweight="bold")

        # Point-biserial r
        ax = axes[1]
        ax.barh(range(len(rel_sorted)), rel_sorted["r_pointbiserial"].abs(),
                color=colors_rel)
        ax.set_yticks(range(len(rel_sorted)))
        ax.set_yticklabels(rel_sorted.index, fontsize=8)
        ax.set_xlabel("|r_pointbiserial|")
        ax.set_title("Point-Biserial Correlation", fontsize=10, fontweight="bold")

        fig.suptitle("Univariate Relevance to Turnover (green=recommended)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 6: RF importance ---
        if rf_imp is not None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 7))

            imp_sorted = rf_imp.sort_values("perm_importance_auc_drop", ascending=True)
            colors_rf = ["#4CAF50" if f in recommended else "#9E9E9E"
                         for f in imp_sorted.index]

            ax = axes[0]
            ax.barh(range(len(imp_sorted)), imp_sorted["gini_importance"],
                    color=colors_rf)
            ax.set_yticks(range(len(imp_sorted)))
            ax.set_yticklabels(imp_sorted.index, fontsize=8)
            ax.set_xlabel("Gini Importance")
            ax.set_title("Gini Importance", fontsize=10, fontweight="bold")

            ax = axes[1]
            ax.barh(range(len(imp_sorted)), imp_sorted["perm_importance_auc_drop"],
                    color=colors_rf)
            ax.set_yticks(range(len(imp_sorted)))
            ax.set_yticklabels(imp_sorted.index, fontsize=8)
            ax.set_xlabel("AUC Drop (permutation)")
            ax.set_title("Permutation Importance", fontsize=10, fontweight="bold")

            fig.suptitle(f"Random Forest Feature Importance (AUC={rf_auc:.4f})",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 7: Recommended set summary ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        lines = [
            "Recommended Feature Set",
            "=" * 50,
            "",
            f"{'#':<4s} {'Feature':<30s} {'AUC':<8s} {'|r_pb|':<8s} {'Perm Imp':<10s} {'VIF':<8s} {'Rank':<8s}",
            "-" * 80,
        ]
        for i, feat in enumerate(recommended, 1):
            auc = relevance.loc[feat, "auc"] if feat in relevance.index else np.nan
            rpb = abs(relevance.loc[feat, "r_pointbiserial"]) if feat in relevance.index else np.nan
            pimp = rf_imp.loc[feat, "perm_importance_auc_drop"] if rf_imp is not None and feat in rf_imp.index else np.nan
            vif = vif_df.loc[feat, "VIF"] if feat in vif_df.index else np.nan
            rank = rank_df.loc[feat, "composite_rank"] if feat in rank_df.index else np.nan
            lines.append(
                f"{i:<4d} {feat:<30s} {auc:<8.4f} {rpb:<8.4f} {pimp:<10.6f} {vif:<8.1f} {rank:<8.1f}"
                if not any(np.isnan(x) for x in [auc, rpb, pimp, vif, rank])
                else f"{i:<4d} {feat:<30s} ..."
            )

        lines.extend([
            "",
            "-" * 80,
            "",
            "To apply this set with manage_env_features.py:",
        ])

        # Which features to remove?
        current_all = set(coverage.index)
        rec_set = set(recommended)
        to_remove = current_all - rec_set
        if to_remove:
            lines.append(f"  python manage_env_features.py remove {' '.join(sorted(to_remove))}")
        lines.append("")
        lines.append("  (Or add new features first with manage_env_features.py add-climate)")

        text = "\n".join(lines)
        ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"  PDF saved: {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Commands
# ═══════════════════════════════════════════════════════════════════════════

def cmd_report(args):
    """Run full feature selection diagnostics and produce a PDF report."""
    export_dir = Path(args.export_dir)
    print(f"\nFeature selection report for: {export_dir}")

    # Load data
    print("\n1. Loading data...")
    est, sc, meta = load_env_data(export_dir)
    X, env_cols = get_env_matrix(est)
    print(f"   {X.shape[0]:,} sites × {X.shape[1]} env features")
    print(f"   Features: {list(X.columns)}")

    pairs = load_pairs_sample(export_dir, n_sample=args.n_sample)
    print(f"   {len(pairs):,} training pairs loaded")

    # Step 1: Coverage
    print("\n2. Checking coverage & NaN rates...")
    coverage = check_coverage(X)
    n_high_nan = (coverage["pct_nan"] > 5).sum()
    print(f"   Features with >5% NaN: {n_high_nan}")
    if n_high_nan:
        for feat in coverage[coverage["pct_nan"] > 5].index:
            print(f"     ⚠ {feat}: {coverage.loc[feat, 'pct_nan']:.1f}% NaN")

    # Step 2: Near-zero variance
    print("\n3. Near-zero variance check...")
    nzv = check_near_zero_variance(X)
    n_nzv = nzv["nzv_flag"].sum()
    print(f"   NZV flagged: {n_nzv}")
    if n_nzv:
        for feat in nzv[nzv["nzv_flag"]].index:
            print(f"     ⚠ {feat}: unique_pct={nzv.loc[feat, 'unique_pct']:.1f}%, "
                  f"freq_ratio={nzv.loc[feat, 'freq_ratio']:.0f}, "
                  f"std={nzv.loc[feat, 'std']:.6f}")

    # Step 3: Correlation matrix
    print("\n4. Pairwise correlations...")
    corr_matrix = compute_correlation_matrix(X)
    corr_pairs_df = find_correlated_pairs(corr_matrix, threshold=args.corr_threshold)
    print(f"   Highly correlated pairs (|r|>{args.corr_threshold}): {len(corr_pairs_df)}")
    for _, row in corr_pairs_df.iterrows():
        print(f"     {row['feature_a']} ~ {row['feature_b']}: r={row['corr']:.3f}")

    # Step 4: VIF
    print("\n5. Variance Inflation Factor...")
    vif_df = compute_vif(X)
    print(f"   Initial VIF values:")
    for feat in vif_df.sort_values("VIF", ascending=False).index:
        flag = " ⚠ HIGH" if vif_df.loc[feat, "VIF"] > 10 else ""
        print(f"     {feat:<30s} VIF={vif_df.loc[feat, 'VIF']:>8.1f}{flag}")

    print(f"\n   Iterative VIF removal (threshold={args.vif_threshold})...")
    vif_kept, vif_removed = iterative_vif_removal(X, vif_threshold=args.vif_threshold)
    if vif_removed:
        print(f"   Removed {len(vif_removed)} features:")
        for feat, val in vif_removed:
            print(f"     ✗ {feat} (VIF={val:.1f})")
    print(f"   Kept {len(vif_kept)}: {vif_kept}")

    # Step 5: Univariate relevance
    print("\n6. Univariate relevance to turnover...")
    relevance = univariate_relevance(est, pairs, env_cols)
    print(f"   {'Feature':<30s} {'AUC':>8s}  {'|r_pb|':>8s}  {'MW p':>10s}")
    print(f"   {'─'*30} {'─'*8}  {'─'*8}  {'─'*10}")
    for feat, row in relevance.iterrows():
        print(f"   {feat:<30s} {row['auc']:>8.4f}  {abs(row['r_pointbiserial']):>8.4f}  "
              f"{row['mw_pvalue']:>10.2e}")

    # Step 6: Random Forest importance
    print("\n7. Random Forest importance estimation...")
    rf_imp, rf_auc = rf_importance(est, pairs, env_cols,
                                    n_sample=min(args.n_sample, 100_000))

    print(f"\n   {'Feature':<30s} {'Gini':>10s}  {'Perm ΔAU':>10s}")
    print(f"   {'─'*30} {'─'*10}  {'─'*10}")
    for feat, row in rf_imp.iterrows():
        print(f"   {feat:<30s} {row['gini_importance']:>10.4f}  "
              f"{row['perm_importance_auc_drop']:>10.6f}")

    # Step 7: Recommendation
    print("\n8. Computing recommendation...")
    recommended, rank_df = recommend_features(
        coverage, nzv, corr_matrix, vif_kept, relevance, rf_imp,
        max_features=args.max_features,
    )

    # Print recommendation
    print(f"\n{'='*70}")
    print(f"RECOMMENDED FEATURE SET ({len(recommended)} features)")
    print(f"{'='*70}")
    for i, feat in enumerate(recommended, 1):
        auc = relevance.loc[feat, "auc"] if feat in relevance.index else np.nan
        print(f"  {i:2d}. {feat:<30s}  AUC={auc:.4f}")

    # What's being dropped?
    current_set = set(env_cols)
    subs_in_current = {c for c in current_set if c.startswith("subs_")}
    climate_in_current = current_set - subs_in_current
    rec_set = set(recommended)
    dropped = current_set - rec_set
    if dropped:
        print(f"\n  Would drop: {sorted(dropped)}")

    # Generate PDF
    generate_report_pdf(
        export_dir, coverage, nzv, corr_matrix, corr_pairs_df,
        vif_df, vif_kept, vif_removed, relevance, rf_imp, rf_auc,
        recommended, rank_df,
    )

    print(f"\n✓ Feature selection report complete.")

    # Optionally apply the recommendation
    if getattr(args, 'apply', False):
        apply_recommendation(export_dir, recommended, env_cols)


def cmd_recommend(args):
    """Quick recommendation without full PDF (faster)."""
    export_dir = Path(args.export_dir)
    est, sc, meta = load_env_data(export_dir)
    X, env_cols = get_env_matrix(est)
    pairs = load_pairs_sample(export_dir, n_sample=50_000)

    print(f"Analysing {len(env_cols)} features on {len(pairs):,} pairs...")

    coverage = check_coverage(X)
    nzv = check_near_zero_variance(X)
    corr_matrix = compute_correlation_matrix(X)
    vif_kept, _ = iterative_vif_removal(X, vif_threshold=args.vif_threshold)
    relevance = univariate_relevance(est, pairs, env_cols)

    # Skip RF for speed
    recommended, rank_df = recommend_features(
        coverage, nzv, corr_matrix, vif_kept, relevance, None,
        max_features=args.max_features,
    )

    print(f"\nRecommended features ({len(recommended)}):")
    for i, f in enumerate(recommended, 1):
        auc = relevance.loc[f, "auc"] if f in relevance.index else "?"
        print(f"  {i:2d}. {f:<30s}  AUC={auc:.4f}" if isinstance(auc, float) else f"  {i}. {f}")

    # Print manage_env_features.py commands
    to_remove = set(env_cols) - set(recommended)
    if to_remove:
        print(f"\nTo apply:")
        print(f"  python src/clesso_nn/manage_env_features.py remove {' '.join(sorted(to_remove))}")

    # Optionally apply
    if getattr(args, 'apply', False):
        apply_recommendation(export_dir, recommended, env_cols)


def cmd_apply(args):
    """Run feature selection and apply the recommended set (remove the rest)."""
    export_dir = Path(args.export_dir)
    est, sc, meta = load_env_data(export_dir)
    X, env_cols = get_env_matrix(est)
    pairs = load_pairs_sample(export_dir, n_sample=args.n_sample)

    print(f"Analysing {len(env_cols)} features on {len(pairs):,} pairs...")

    coverage = check_coverage(X)
    nzv = check_near_zero_variance(X)
    corr_matrix = compute_correlation_matrix(X)
    vif_kept, _ = iterative_vif_removal(X, vif_threshold=args.vif_threshold)
    relevance = univariate_relevance(est, pairs, env_cols)

    # RF importance for better ranking
    print("\nRunning Random Forest importance...")
    rf_imp, rf_auc = rf_importance(est, pairs, env_cols,
                                    n_sample=min(args.n_sample, 100_000))

    recommended, rank_df = recommend_features(
        coverage, nzv, corr_matrix, vif_kept, relevance, rf_imp,
        max_features=args.max_features,
    )

    print(f"\nRecommended features ({len(recommended)}):")
    for i, f in enumerate(recommended, 1):
        auc = relevance.loc[f, "auc"] if f in relevance.index else np.nan
        print(f"  {i:2d}. {f:<30s}  AUC={auc:.4f}")

    apply_recommendation(export_dir, recommended, env_cols)


def apply_recommendation(export_dir: Path, recommended: list[str],
                         env_cols: list[str]):
    """Remove non-recommended env columns from feather files.

    Backs up files first, then drops columns not in the recommended set
    from both site_covariates.feather and env_site_table.feather.
    Updates metadata.json accordingly.
    """
    rec_set = set(recommended)
    to_remove = [c for c in env_cols if c not in rec_set]

    if not to_remove:
        print("\n✓ All features are already in the recommended set. Nothing to remove.")
        return

    print(f"\n{'='*70}")
    print(f"APPLYING RECOMMENDATION — removing {len(to_remove)} features")
    print(f"{'='*70}")
    print(f"  Keeping:  {sorted(rec_set & set(env_cols))}")
    print(f"  Removing: {sorted(to_remove)}")

    # Backup
    for name in ["site_covariates.feather", "env_site_table.feather", "metadata.json"]:
        src = export_dir / name
        dst = export_dir / f"{name}.bak"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Backed up: {name} → {name}.bak")

    # Load current data
    est = pf.read_feather(export_dir / "env_site_table.feather")
    sc = pf.read_feather(export_dir / "site_covariates.feather")
    with open(export_dir / "metadata.json") as f:
        meta = json.load(f)

    # Remove columns
    drop_sc = [c for c in to_remove if c in sc.columns]
    drop_est = [c for c in to_remove if c in est.columns]
    sc = sc.drop(columns=drop_sc)
    est = est.drop(columns=drop_est)

    # Update metadata
    remove_set = set(to_remove)
    meta["alpha_cov_cols"] = [c for c in meta.get("alpha_cov_cols", []) if c not in remove_set]
    meta["env_cov_cols"] = [c for c in meta.get("env_cov_cols", []) if c not in remove_set]

    # Save
    pf.write_feather(sc, export_dir / "site_covariates.feather")
    pf.write_feather(est, export_dir / "env_site_table.feather")
    with open(export_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n✓ Applied. Feather files updated:")
    print(f"  site_covariates: {sc.shape[0]:,} × {sc.shape[1]}")
    print(f"  env_site_table:  {est.shape[0]:,} × {est.shape[1]}")
    print(f"  Removed {len(drop_sc)} cols from site_covariates, {len(drop_est)} from env_site_table")
    print(f"\n  To undo: python src/clesso_nn/manage_env_features.py restore")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Feature selection pre-processing for CLESSO NN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR),
        help="Path to nn_export directory",
    )

    sub = parser.add_subparsers(dest="command")

    # -- report --
    rpt = sub.add_parser("report", help="Full diagnostics report with PDF")
    rpt.add_argument("--n-sample", type=int, default=200_000,
                     help="Number of pairs to sample for analysis (default: 200k)")
    rpt.add_argument("--max-features", type=int, default=15,
                     help="Maximum features to recommend (default: 15)")
    rpt.add_argument("--vif-threshold", type=float, default=10.0,
                     help="VIF threshold for collinearity removal (default: 10)")
    rpt.add_argument("--corr-threshold", type=float, default=0.85,
                     help="Correlation threshold for flagging pairs (default: 0.85)")
    rpt.add_argument("--apply", action="store_true",
                     help="Apply the recommendation: remove non-selected features from feathers")

    # -- recommend --
    rec = sub.add_parser("recommend", help="Quick feature recommendation (no PDF)")
    rec.add_argument("--max-features", type=int, default=15,
                     help="Maximum features to recommend (default: 15)")
    rec.add_argument("--vif-threshold", type=float, default=10.0,
                     help="VIF threshold for collinearity removal (default: 10)")
    rec.add_argument("--apply", action="store_true",
                     help="Apply the recommendation: remove non-selected features from feathers")

    # -- apply --
    app = sub.add_parser("apply", help="Run selection and apply: remove non-recommended features")
    app.add_argument("--n-sample", type=int, default=100_000,
                     help="Number of pairs to sample (default: 100k)")
    app.add_argument("--max-features", type=int, default=15,
                     help="Maximum features to recommend (default: 15)")
    app.add_argument("--vif-threshold", type=float, default=10.0,
                     help="VIF threshold (default: 10)")

    args = parser.parse_args()
    if args.command == "report":
        cmd_report(args)
    elif args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "apply":
        cmd_apply(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
