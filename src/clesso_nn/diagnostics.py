#!/usr/bin/env python3
"""
diagnostics.py -- Diagnostic and function-shape plots for CLESSO NN.

Mirrors the R script clesso_v2/clesso_diagnostics.R but for the neural
network model. Loads a trained checkpoint and the export data, then
generates a multi-page PDF with:

  A. MODEL DIAGNOSTICS
     1. Training curves (loss, accuracy, alpha stats, LR)
     2. Residual analysis (within / between)
     3. Observed vs predicted p_match
     4. Classification metrics (AUC, confusion matrix)
     5. Spatial map of fitted alpha (richness)
     6. Alpha vs observed richness comparison

  B. FITTED FUNCTION SHAPES
     7. Beta (turnover) monotone response curves per env dimension
     8. Similarity decay curves S = exp(-eta)
     9. Alpha partial-dependence plots (site covariates → richness)
    10. Variable importance bar chart

Usage:
    python src/clesso_nn/diagnostics.py \\
        --export-dir path/to/nn_export \\
        --checkpoint  path/to/best_model.pt

    Or after run_clesso_nn.py has completed:
    python src/clesso_nn/diagnostics.py \\
        --export-dir src/clesso_v2/output/VAS_.../nn_export
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Guard matplotlib backend before import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

import pandas as pd
import torch

# Ensure project root on path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.clesso_nn.config import CLESSONNConfig
from src.clesso_nn.dataset import SiteData, load_export, make_dataloaders, collate_fn
from src.clesso_nn.model import CLESSONet
from src.clesso_nn.predict import (
    check_monotonicity,
    load_model,
    predict_alpha,
)

# --------------------------------------------------------------------------
# Colour palette (matching R diagnostics)
# --------------------------------------------------------------------------
PAL_BLUE = "#2166AC"
PAL_RED = "#B2182B"
PAL_ORANGE = "#E08214"
PAL_GREEN = "#1B7837"
PAL_GREY = "#636363"
PAL_LIGHT = "#D1E5F0"


# ==========================================================================
# Helper utilities
# ==========================================================================

def _auc_wilcoxon(y_true: np.ndarray, y_score: np.ndarray, max_n: int = 50_000) -> float:
    """Approximate AUC via Wilcoxon statistic (sampled for speed)."""
    idx1 = np.where(y_true == 1)[0]
    idx0 = np.where(y_true == 0)[0]
    if len(idx1) == 0 or len(idx0) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    s1 = y_score[rng.choice(idx1, min(len(idx1), max_n), replace=False)]
    s0 = y_score[rng.choice(idx0, min(len(idx0), max_n), replace=False)]
    # Fraction of concordant pairs
    return float(np.mean(s1[:, None] > s0[None, :]) +
                 0.5 * np.mean(s1[:, None] == s0[None, :]))


def _safe_density(vals, lo=None, hi=None, num=300):
    """Kernel density estimate (Gaussian)."""
    from scipy.stats import gaussian_kde
    if lo is None:
        lo = np.nanmin(vals)
    if hi is None:
        hi = np.nanmax(vals)
    x = np.linspace(lo, hi, num)
    try:
        kde = gaussian_kde(vals[np.isfinite(vals)])
        return x, kde(x)
    except Exception:
        return x, np.zeros_like(x)


# ==========================================================================
# Main diagnostics driver
# ==========================================================================

def run_diagnostics(
    export_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    """Generate the full diagnostics PDF."""

    export_dir = Path(export_dir).resolve()
    t0_total = time.time()

    # ------------------------------------------------------------------
    # 0. Resolve paths
    # ------------------------------------------------------------------
    if checkpoint_path is None:
        # Default: look in standard NN output location
        cfg = CLESSONNConfig()
        checkpoint_path = cfg.output_dir / "best_model.pt"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_dir is None:
        output_dir = checkpoint_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CLESSO NN Diagnostics")
    print("=" * 60)
    print(f"  Export dir:  {export_dir}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Output dir:  {output_dir}")

    # ------------------------------------------------------------------
    # 1. Load model and data
    # ------------------------------------------------------------------
    print("\n--- Loading model and data ---")
    model, ckpt = load_model(checkpoint_path, device="cpu")
    model.eval()
    data = load_export(export_dir)

    # Infer include_geo_in_beta from checkpoint: if no geo names in stats, geo was excluded
    stats = ckpt["site_data_stats"]
    has_geo = any("geo" in n for n in stats.get("env_cov_names", []))
    data["metadata"]["include_geo_in_beta"] = has_geo

    # Pass through geo distance and Fourier settings from checkpoint
    data["metadata"]["include_geo_dist_in_beta"] = stats.get("include_geo_dist_in_beta", False)
    data["metadata"]["fourier_n_frequencies"] = stats.get("fourier_n_frequencies", 0)
    data["metadata"]["fourier_max_wavelength"] = stats.get("fourier_max_wavelength", 40.0)

    site_data = SiteData(
        site_covariates=data["site_covariates"],
        env_site_table=data["env_site_table"],
        site_obs_richness=data["site_obs_richness"],
        metadata=data["metadata"],
        effort_cov_names=stats.get("effort_cov_names") or None,
    )

    # Override geo_dist_scale with checkpoint value for exact reproducibility
    # (SiteData recomputes from random samples, which should match but this is safer)
    ckpt_geo_dist_scale = stats.get("geo_dist_scale")
    if ckpt_geo_dist_scale is not None and site_data.include_geo_dist_in_beta:
        site_data.geo_dist_scale = ckpt_geo_dist_scale

    stats = ckpt["site_data_stats"]
    cfg_model = ckpt["config"]
    best_epoch = ckpt.get("epoch", "?")
    best_val_loss = ckpt.get("val_loss", float("nan"))

    species = data["metadata"].get("species_group", "Unknown")
    n_sites = len(data["site_covariates"])
    n_pairs = len(data["pairs"])
    print(f"  Species:    {species}")
    print(f"  Sites:      {n_sites:,}")
    print(f"  Pairs:      {n_pairs:,}")
    print(f"  Best epoch: {best_epoch}  (val_loss={best_val_loss:.6f})")

    # ------------------------------------------------------------------
    # 2. Compute predictions on all pairs
    # ------------------------------------------------------------------
    print("\n--- Computing predictions on all pairs ---")

    # Alpha for all sites
    alpha_all = predict_alpha(model, data["site_covariates"], ckpt, "cpu")
    site_ids = data["site_covariates"]["site_id"].values
    site_id_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    # Match probability for each pair
    pairs = data["pairs"]
    idx_i = np.array([site_id_to_idx[s] for s in pairs["site_i"]])
    idx_j = np.array([site_id_to_idx[s] for s in pairs["site_j"]])

    alpha_i = alpha_all[idx_i]
    alpha_j = alpha_all[idx_j]

    # Compute env_diff for each pair
    env_i = site_data.get_env_at_site(torch.from_numpy(idx_i.astype(np.int64)))
    env_j = site_data.get_env_at_site(torch.from_numpy(idx_j.astype(np.int64)))
    env_diff = torch.abs(env_i - env_j).numpy().astype(np.float32)

    # Append haversine geographic distance if model was trained with it
    if site_data.include_geo_dist_in_beta and site_data.lon_deg is not None:
        from clesso_nn.dataset import haversine_km
        geo_dist = haversine_km(
            site_data.lon_deg[idx_i], site_data.lat_deg[idx_i],
            site_data.lon_deg[idx_j], site_data.lat_deg[idx_j],
        )
        geo_dist_norm = (geo_dist / site_data.geo_dist_scale).astype(np.float32)
        env_diff = np.column_stack([env_diff, geo_dist_norm])

    is_within = pairs["is_within"].values.astype(bool)
    env_diff[is_within] = 0.0

    with torch.no_grad():
        eta = model.beta_net(torch.from_numpy(env_diff)).cpu().numpy()
    similarity = np.exp(-eta)

    # p_match
    p_within = 1.0 / alpha_i
    p_between = similarity * (alpha_i + alpha_j) / (2.0 * alpha_i * alpha_j)
    p_match = np.where(is_within, p_within, p_between)
    p_match = np.clip(p_match, 1e-7, 1 - 1e-7)

    y_obs = pairs["y"].values.astype(np.float32)
    residuals = y_obs - p_match

    print(f"  p_match range: [{p_match.min():.4f}, {p_match.max():.4f}]")
    print(f"  alpha range:   [{alpha_all.min():.1f}, {alpha_all.max():.1f}]")

    # ------------------------------------------------------------------
    # 3. Load training log if available
    # ------------------------------------------------------------------
    # Support joint (single log), two-stage (stage1 + stage2 logs),
    # and cyclic (single log with cycle/phase columns)
    log_path = checkpoint_path.parent / "training_progress.log"
    s1_log_path = checkpoint_path.parent / "training_progress_stage1.log"
    s2_log_path = checkpoint_path.parent / "training_progress_stage2.log"
    cyc_log_path = checkpoint_path.parent / "training_progress_cyclic.log"
    ft_log_path = checkpoint_path.parent / "training_progress_finetune.log"

    train_log = None       # joint-mode log
    stage1_log = None      # two-stage: alpha-only on within-site
    stage2_log = None      # two-stage: beta-only on between-site
    cyclic_log = None      # cyclic: alternating alpha/beta per cycle
    finetune_log = None    # cyclic_finetune: Phase 2 geo fine-tuning
    is_two_stage = s1_log_path.exists() and s2_log_path.exists()
    is_cyclic = cyc_log_path.exists()
    is_cyclic_finetune = is_cyclic and ft_log_path.exists()

    if is_cyclic:
        cyclic_log = pd.read_csv(cyc_log_path)
        n_cycles = cyclic_log["cycle"].nunique()
        n_alpha = len(cyclic_log[cyclic_log["phase"] == "alpha"])
        n_beta = len(cyclic_log[cyclic_log["phase"] == "beta"])
        print(f"  Training log (cyclic): {n_cycles} cycles, "
              f"{n_alpha} alpha epochs, {n_beta} beta epochs")
        if is_cyclic_finetune:
            finetune_log = pd.read_csv(ft_log_path)
            print(f"  Training log (finetune Phase 2): {len(finetune_log)} epochs")
    elif is_two_stage:
        stage1_log = pd.read_csv(s1_log_path)
        stage2_log = pd.read_csv(s2_log_path)
        print(f"  Training logs: Stage 1 ({len(stage1_log)} epochs), "
              f"Stage 2 ({len(stage2_log)} epochs)")
    elif log_path.exists():
        train_log = pd.read_csv(log_path)
        print(f"  Training log: {len(train_log)} epochs")

    # ------------------------------------------------------------------
    # Open PDF
    # ------------------------------------------------------------------
    pdf_path = output_dir / f"{species}_nn_diagnostics.pdf"
    print(f"\n--- Generating diagnostics: {pdf_path.name} ---")

    with PdfPages(str(pdf_path)) as pdf:

        # ==============================================================
        # A1. Model summary text page
        # ==============================================================
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.axis("off")

        env_names = stats.get("env_cov_names", [])
        alpha_names = stats.get("alpha_cov_names", [])
        geo_names = ["geo_lon", "geo_lat"] if site_data.geo is not None else []
        geo_dist_names = ["geo_dist_km"] if getattr(site_data, "include_geo_dist_in_beta", False) else []

        training_mode = cfg_model.get("training_mode", "joint")

        summary = [
            f"CLESSO Neural Network -- {species} -- Model Summary",
            "",
            f"Training mode:   {training_mode}",
            f"Best epoch:      {best_epoch}",
            f"Best val loss:   {best_val_loss:.6f}",
            f"Total pairs:     {n_pairs:,}",
            f"Total sites:     {n_sites:,}",
            "",
            f"K_alpha (site covariates): {cfg_model['K_alpha']}",
            f"  Columns: {', '.join(alpha_names)}",
            f"K_env (pairwise env):      {cfg_model['K_env']}",
            f"  Env: {', '.join(env_names)}",
            f"  Geo: {', '.join(geo_names + geo_dist_names)}",
            "",
            f"Alpha network:  {cfg_model['alpha_hidden']}  "
            f"(dropout={cfg_model['alpha_dropout']}, act={cfg_model['alpha_activation']})",
            f"Beta network:   {cfg_model['beta_hidden']}  "
            f"(monotone, dropout={cfg_model['beta_dropout']})",
            f"Alpha LB lambda: {cfg_model.get('alpha_lb_lambda', 'N/A')}",
        ]

        # Two-stage specific summary lines
        if is_two_stage:
            s1_epochs = len(stage1_log) if stage1_log is not None else 0
            s2_epochs = len(stage2_log) if stage2_log is not None else 0
            s2_final_auc = float(stage2_log["val_auc"].iloc[-1]) if (
                stage2_log is not None and "val_auc" in stage2_log.columns
            ) else float("nan")
            s2_final_eta = float(stage2_log["val_eta_mean"].iloc[-1]) if (
                stage2_log is not None and "val_eta_mean" in stage2_log.columns
            ) else float("nan")
            summary += [
                "",
                f"--- Two-Stage Training ---",
                f"Stage 1 (alpha-only, within-site):  {s1_epochs} epochs",
                f"Stage 2 (beta-only, between-site):  {s2_epochs} epochs",
                f"Stage 2 final AUC:    {s2_final_auc:.4f}",
                f"Stage 2 final η mean: {s2_final_eta:.4f}",
            ]

        # Cyclic specific summary lines
        if is_cyclic and cyclic_log is not None:
            cyc_alpha = cyclic_log[cyclic_log["phase"] == "alpha"]
            cyc_beta = cyclic_log[cyclic_log["phase"] == "beta"]
            _nc = cyclic_log["cycle"].nunique()
            # Best beta AUC across all cycles
            _best_auc = float(cyc_beta["val_auc"].max()) if "val_auc" in cyc_beta.columns else float("nan")
            _best_cycle = int(cyc_beta.loc[cyc_beta["val_loss"].idxmin(), "cycle"]) if len(cyc_beta) > 0 else 0
            summary += [
                "",
                f"--- Cyclic Block-Coordinate Descent ---",
                f"Total cycles:       {_nc}",
                f"Alpha epochs/cycle: {len(cyc_alpha) // max(_nc, 1)}",
                f"Beta epochs/cycle:  {len(cyc_beta) // max(_nc, 1)}",
                f"Best beta AUC:      {_best_auc:.4f}",
                f"Best cycle:         {_best_cycle}",
            ]

        # Cyclic finetune Phase 2 summary
        if is_cyclic_finetune and finetune_log is not None:
            ft_epochs = len(finetune_log)
            ft_best_auc = float(finetune_log["val_auc"].max()) if "val_auc" in finetune_log.columns else float("nan")
            ft_best_loss = float(finetune_log["val_loss"].min()) if "val_loss" in finetune_log.columns else float("nan")
            summary += [
                "",
                f"--- Phase 2: Geographic Fine-tuning ---",
                f"Finetune epochs:     {ft_epochs}",
                f"Best finetune AUC:   {ft_best_auc:.4f}",
                f"Best finetune loss:  {ft_best_loss:.6f}",
            ]

        summary += [
            "",
            f"Alpha: mean={alpha_all.mean():.1f}  median={np.median(alpha_all):.1f}  "
            f"range=[{alpha_all.min():.1f}, {alpha_all.max():.1f}]",
            f"Eta (between-site): mean={eta[~is_within].mean():.4f}  "
            f"max={eta[~is_within].max():.4f}",
            f"Parameters: alpha_net={sum(p.numel() for p in model.alpha_net.parameters()):,}  "
            f"beta_net={sum(p.numel() for p in model.beta_net.parameters()):,}  "
            f"total={sum(p.numel() for p in model.parameters()):,}",
        ]

        ax.text(0.05, 0.95, "\n".join(summary), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # A2. Training curves
        # ==============================================================
        if is_cyclic and cyclic_log is not None and len(cyclic_log) > 1:
            # ---- Cyclic mode: alternating alpha/beta phases ----
            import matplotlib.cm as cm

            cyc_alpha = cyclic_log[cyclic_log["phase"] == "alpha"].copy()
            cyc_beta = cyclic_log[cyclic_log["phase"] == "beta"].copy()
            n_cyc = cyclic_log["cycle"].nunique()
            cmap = plt.get_cmap("viridis", n_cyc)

            # --- Page 1: Alpha phase across cycles ---
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

            # Alpha val loss per cycle
            ax1 = fig.add_subplot(gs[0, 0])
            for c in sorted(cyc_alpha["cycle"].unique()):
                sub = cyc_alpha[cyc_alpha["cycle"] == c]
                ax1.plot(sub["phase_epoch"], sub["val_loss"],
                         color=cmap(c - 1), lw=1.2, alpha=0.8, label=f"C{c}")
            ax1.set_xlabel("Phase epoch"); ax1.set_ylabel("Val loss")
            ax1.set_title("Alpha Phase: Validation Loss per Cycle")
            ax1.legend(fontsize=6, ncol=2)

            # Alpha accuracy per cycle
            ax2 = fig.add_subplot(gs[0, 1])
            for c in sorted(cyc_alpha["cycle"].unique()):
                sub = cyc_alpha[cyc_alpha["cycle"] == c]
                ax2.plot(sub["phase_epoch"], sub["val_accuracy"],
                         color=cmap(c - 1), lw=1.2, alpha=0.8, label=f"C{c}")
            ax2.set_xlabel("Phase epoch"); ax2.set_ylabel("Accuracy")
            ax2.set_title("Alpha Phase: Validation Accuracy per Cycle")
            ax2.set_ylim(0.5, 1.0)
            ax2.legend(fontsize=6, ncol=2)

            # Alpha mean trajectory across global epochs
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(cyc_alpha["global_epoch"], cyc_alpha["alpha_mean"],
                     color=PAL_BLUE, lw=1.5, label="Mean α")
            ax3.fill_between(
                cyc_alpha["global_epoch"],
                cyc_alpha["alpha_min"].astype(float),
                cyc_alpha["alpha_max"].astype(float),
                alpha=0.12, color=PAL_BLUE,
            )
            # Mark cycle boundaries
            for c in sorted(cyc_alpha["cycle"].unique()):
                sub = cyc_alpha[cyc_alpha["cycle"] == c]
                ax3.axvline(sub["global_epoch"].iloc[0], color=PAL_GREY,
                            ls=":", lw=0.8, alpha=0.5)
            ax3.set_xlabel("Global epoch"); ax3.set_ylabel("Alpha (richness)")
            ax3.set_title("Alpha: Mean Richness Across All Cycles")

            # Alpha LB penalty
            ax4 = fig.add_subplot(gs[1, 1])
            if "train_lb" in cyc_alpha.columns:
                ax4.plot(cyc_alpha["global_epoch"], cyc_alpha["train_lb"],
                         color=PAL_ORANGE, lw=1.2)
            ax4.set_xlabel("Global epoch"); ax4.set_ylabel("LB penalty")
            ax4.set_title("Alpha Phase: Lower-Bound Penalty")

            fig.suptitle(f"Cyclic Training: Alpha Phases -- {species}", fontsize=13)
            pdf.savefig(fig); plt.close(fig)

            # --- Page 2: Beta phase across cycles ---
            if len(cyc_beta) > 1:
                fig = plt.figure(figsize=(14, 10))
                gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

                # Beta val loss per cycle
                ax1 = fig.add_subplot(gs[0, 0])
                for c in sorted(cyc_beta["cycle"].unique()):
                    sub = cyc_beta[cyc_beta["cycle"] == c]
                    ax1.plot(sub["phase_epoch"], sub["val_loss"],
                             color=cmap(c - 1), lw=1.2, alpha=0.8, label=f"C{c}")
                ax1.set_xlabel("Phase epoch"); ax1.set_ylabel("Val loss")
                ax1.set_title("Beta Phase: Validation Loss per Cycle")
                ax1.legend(fontsize=6, ncol=2)

                # Beta AUC per cycle
                ax2 = fig.add_subplot(gs[0, 1])
                if "val_auc" in cyc_beta.columns:
                    for c in sorted(cyc_beta["cycle"].unique()):
                        sub = cyc_beta[cyc_beta["cycle"] == c]
                        ax2.plot(sub["phase_epoch"], sub["val_auc"],
                                 color=cmap(c - 1), lw=1.2, alpha=0.8, label=f"C{c}")
                    ax2.axhline(0.5, color=PAL_GREY, ls=":", lw=1)
                    ax2.set_ylim(0.45, max(0.85, cyc_beta["val_auc"].max() + 0.05))
                ax2.set_xlabel("Phase epoch"); ax2.set_ylabel("AUC")
                ax2.set_title("Beta Phase: Between-Site AUC per Cycle")
                ax2.legend(fontsize=6, ncol=2)

                # Beta eta mean trajectory
                ax3 = fig.add_subplot(gs[1, 0])
                if "val_eta_mean" in cyc_beta.columns:
                    ax3.plot(cyc_beta["global_epoch"], cyc_beta["val_eta_mean"],
                             color=PAL_BLUE, lw=1.5, label="Mean η")
                    if "val_eta_max" in cyc_beta.columns:
                        ax3.plot(cyc_beta["global_epoch"], cyc_beta["val_eta_max"],
                                 color=PAL_ORANGE, lw=1, ls="--", label="Max η")
                    for c in sorted(cyc_beta["cycle"].unique()):
                        sub = cyc_beta[cyc_beta["cycle"] == c]
                        ax3.axvline(sub["global_epoch"].iloc[0], color=PAL_GREY,
                                    ls=":", lw=0.8, alpha=0.5)
                    ax3.legend(fontsize=8)
                ax3.set_xlabel("Global epoch"); ax3.set_ylabel("η (turnover)")
                ax3.set_title("Beta: Eta Across All Cycles")

                # Per-cycle summary: best val_loss per cycle
                ax4 = fig.add_subplot(gs[1, 1])
                cycles_list = sorted(cyc_beta["cycle"].unique())
                best_losses = [cyc_beta[cyc_beta["cycle"] == c]["val_loss"].min()
                               for c in cycles_list]
                best_aucs = [cyc_beta[cyc_beta["cycle"] == c]["val_auc"].max()
                             if "val_auc" in cyc_beta.columns else 0
                             for c in cycles_list]
                ax4.bar(cycles_list, best_losses, color=PAL_BLUE, alpha=0.7,
                        label="Best val_loss")
                ax4.set_xlabel("Cycle"); ax4.set_ylabel("Best val_loss")
                ax4.set_title("Per-Cycle Best Beta Loss")
                ax4b = ax4.twinx()
                ax4b.plot(cycles_list, best_aucs, color=PAL_GREEN, lw=2,
                          marker="o", ms=5, label="Best AUC")
                ax4b.set_ylabel("Best AUC", color=PAL_GREEN)
                ax4.legend(loc="upper left", fontsize=8)
                ax4b.legend(loc="upper right", fontsize=8)

                fig.suptitle(f"Cyclic Training: Beta Phases -- {species}", fontsize=13)
                pdf.savefig(fig); plt.close(fig)

            # ---- Phase 2 finetune curves (if cyclic_finetune) ----
            if is_cyclic_finetune and finetune_log is not None and len(finetune_log) > 1:
                ft = finetune_log
                fig = plt.figure(figsize=(14, 10))
                gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

                # Finetune val loss
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(ft["epoch"], ft["val_loss"],
                         color=PAL_RED, lw=1.5, label="Val loss")
                if "train_loss" in ft.columns:
                    ax1.plot(ft["epoch"], ft["train_loss"],
                             color=PAL_BLUE, lw=1.5, alpha=0.7, label="Train loss")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
                ax1.set_title("Phase 2 (Geo Fine-tune): Loss")
                ax1.legend(fontsize=8)

                # Finetune AUC
                ax2 = fig.add_subplot(gs[0, 1])
                if "val_auc" in ft.columns:
                    ax2.plot(ft["epoch"], ft["val_auc"],
                             color=PAL_GREEN, lw=1.5)
                ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
                ax2.set_title("Phase 2: Between-Site AUC")

                # Finetune eta stats
                ax3 = fig.add_subplot(gs[1, 0])
                if "val_eta_mean" in ft.columns:
                    ax3.plot(ft["epoch"], ft["val_eta_mean"],
                             color=PAL_BLUE, lw=1.5, label="Mean η")
                if "val_eta_std" in ft.columns:
                    ax3.fill_between(
                        ft["epoch"],
                        ft["val_eta_mean"].astype(float) - ft["val_eta_std"].astype(float),
                        ft["val_eta_mean"].astype(float) + ft["val_eta_std"].astype(float),
                        alpha=0.15, color=PAL_BLUE, label="±1 std",
                    )
                ax3.set_xlabel("Epoch"); ax3.set_ylabel("η (turnover)")
                ax3.set_title("Phase 2: Eta Stats")
                ax3.legend(fontsize=8)

                # Finetune alpha stats
                ax4 = fig.add_subplot(gs[1, 1])
                if "alpha_mean" in ft.columns:
                    ax4.plot(ft["epoch"], ft["alpha_mean"],
                             color=PAL_BLUE, lw=1.5, label="Mean α")
                    if "alpha_min" in ft.columns:
                        ax4.fill_between(
                            ft["epoch"],
                            ft["alpha_min"].astype(float),
                            ft["alpha_max"].astype(float),
                            alpha=0.15, color=PAL_BLUE, label="[min, max]",
                        )
                ax4.set_xlabel("Epoch"); ax4.set_ylabel("Alpha (richness)")
                ax4.set_title("Phase 2: Alpha Stats")
                ax4.legend(fontsize=8)

                fig.suptitle(f"Phase 2 Geo Fine-tuning -- {species}", fontsize=13)
                pdf.savefig(fig); plt.close(fig)

        elif is_two_stage and stage1_log is not None and len(stage1_log) > 1:
            # ---- Two-stage mode: Stage 1 (alpha-only) ----
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

            s1 = stage1_log

            # Stage 1 loss
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(s1["epoch"], s1["train_loss"],
                     color=PAL_BLUE, lw=1.5, label="Train loss")
            ax1.plot(s1["epoch"], s1["val_loss"],
                     color=PAL_RED, lw=1.5, label="Val loss")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
            ax1.set_title("Stage 1 (Alpha-Only): Loss")
            ax1.legend(fontsize=8)

            # Stage 1 BCE vs LB
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(s1["epoch"], s1["train_bce"],
                     color=PAL_BLUE, lw=1.5, label="Train BCE")
            ax2.plot(s1["epoch"], s1["val_bce"],
                     color=PAL_RED, lw=1.5, label="Val BCE")
            if "train_lb" in s1.columns:
                ax2.plot(s1["epoch"], s1["train_lb"],
                         color=PAL_ORANGE, lw=1.5, ls="--", label="LB penalty")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss component")
            ax2.set_title("Stage 1: BCE and LB Penalty")
            ax2.legend(fontsize=8)

            # Stage 1 accuracy
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(s1["epoch"], s1["val_accuracy"],
                     color=PAL_GREEN, lw=1.5)
            ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy")
            ax3.set_title("Stage 1: Validation Accuracy (within-site)")
            ax3.set_ylim(0, 1)

            # Stage 1 alpha trajectory
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(s1["epoch"], s1["alpha_mean"],
                     color=PAL_BLUE, lw=1.5, label="Mean α")
            ax4.fill_between(
                s1["epoch"],
                s1["alpha_min"].astype(float),
                s1["alpha_max"].astype(float),
                alpha=0.15, color=PAL_BLUE, label="[min, max]",
            )
            ax4.set_xlabel("Epoch"); ax4.set_ylabel("Alpha (richness)")
            ax4.set_title("Stage 1: Alpha During Training")
            ax4.legend(fontsize=8)

            fig.suptitle(f"Stage 1: Alpha Training on Within-Site Pairs -- {species}",
                         fontsize=13)
            pdf.savefig(fig); plt.close(fig)

            # ---- Two-stage mode: Stage 2 (beta-only) ----
            if stage2_log is not None and len(stage2_log) > 1:
                s2 = stage2_log
                fig = plt.figure(figsize=(14, 10))
                gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

                # Stage 2 loss
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(s2["epoch"], s2["train_loss"],
                         color=PAL_BLUE, lw=1.5, label="Train loss")
                ax1.plot(s2["epoch"], s2["val_loss"],
                         color=PAL_RED, lw=1.5, label="Val loss")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (importance-weighted)")
                ax1.set_title("Stage 2 (Beta-Only): Loss")
                ax1.legend(fontsize=8)

                # Stage 2 eta trajectory
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.plot(s2["epoch"], s2["val_eta_mean"],
                         color=PAL_BLUE, lw=1.5, label="Mean η")
                if "val_eta_min" in s2.columns and "val_eta_max" in s2.columns:
                    ax2.fill_between(
                        s2["epoch"],
                        s2["val_eta_min"].astype(float),
                        s2["val_eta_max"].astype(float),
                        alpha=0.15, color=PAL_BLUE, label="[min, max]",
                    )
                if "val_eta_std" in s2.columns:
                    ax2.plot(s2["epoch"], s2["val_eta_std"],
                             color=PAL_ORANGE, lw=1, ls="--", label="Std η")
                ax2.set_xlabel("Epoch"); ax2.set_ylabel("η (turnover)")
                ax2.set_title("Stage 2: Eta (Turnover) During Training")
                ax2.legend(fontsize=8)

                # Stage 2 AUC
                ax3 = fig.add_subplot(gs[1, 0])
                if "val_auc" in s2.columns:
                    ax3.plot(s2["epoch"], s2["val_auc"],
                             color=PAL_GREEN, lw=1.5)
                    ax3.axhline(0.5, color=PAL_GREY, ls=":", lw=1,
                                label="Random (0.5)")
                    ax3.set_ylim(0.45, max(0.85, s2["val_auc"].max() + 0.05))
                    ax3.legend(fontsize=8)
                ax3.set_xlabel("Epoch"); ax3.set_ylabel("AUC")
                ax3.set_title("Stage 2: Between-Site AUC")

                # Stage 2 beta gradient norm
                ax4 = fig.add_subplot(gs[1, 1])
                if "beta_grad_norm" in s2.columns:
                    ax4.plot(s2["epoch"], s2["beta_grad_norm"],
                             color=PAL_BLUE, lw=1.5)
                    ax4.set_yscale("log")
                ax4.set_xlabel("Epoch"); ax4.set_ylabel("β gradient norm")
                ax4.set_title("Stage 2: Beta Gradient Norm")

                fig.suptitle(
                    f"Stage 2: Beta Training on Between-Site Pairs -- {species}",
                    fontsize=13)
                pdf.savefig(fig); plt.close(fig)

        elif train_log is not None and len(train_log) > 1:
            # ---- Joint mode (legacy) ----
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

            # Loss curves
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(train_log["epoch"], train_log["train_loss"],
                     color=PAL_BLUE, lw=1.5, label="Train loss")
            ax1.plot(train_log["epoch"], train_log["val_loss"],
                     color=PAL_RED, lw=1.5, label="Val loss")
            ax1.axvline(best_epoch, color=PAL_GREY, ls="--", lw=1, alpha=0.6)
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
            ax1.set_title("Training & Validation Loss")
            ax1.legend(fontsize=8)

            # BCE vs LB penalty
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(train_log["epoch"], train_log["train_bce"],
                     color=PAL_BLUE, lw=1.5, label="Train BCE")
            ax2.plot(train_log["epoch"], train_log["val_bce"],
                     color=PAL_RED, lw=1.5, label="Val BCE")
            if "train_lb" in train_log.columns:
                ax2.plot(train_log["epoch"], train_log["train_lb"],
                         color=PAL_ORANGE, lw=1.5, ls="--", label="Train LB penalty")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss component")
            ax2.set_title("BCE and Lower-Bound Penalty")
            ax2.legend(fontsize=8)

            # Accuracy
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(train_log["epoch"], train_log["val_accuracy"],
                     color=PAL_GREEN, lw=1.5)
            ax3.axvline(best_epoch, color=PAL_GREY, ls="--", lw=1, alpha=0.6)
            ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy")
            ax3.set_title("Validation Accuracy")
            ax3.set_ylim(0, 1)

            # Alpha stats over training
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(train_log["epoch"], train_log["alpha_mean"],
                     color=PAL_BLUE, lw=1.5, label="Mean α")
            ax4.fill_between(
                train_log["epoch"],
                train_log["alpha_min"].astype(float),
                train_log["alpha_max"].astype(float),
                alpha=0.15, color=PAL_BLUE, label="[min, max]",
            )
            ax4.axvline(best_epoch, color=PAL_GREY, ls="--", lw=1, alpha=0.6)
            ax4.set_xlabel("Epoch"); ax4.set_ylabel("Alpha (richness)")
            ax4.set_title("Alpha Statistics During Training")
            ax4.legend(fontsize=8)

            fig.suptitle(f"CLESSO NN Training Curves -- {species}", fontsize=13)
            pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # A3. Residual analysis
        # ==============================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Within-site residuals
        ax = axes[0, 0]
        r_w = residuals[is_within]
        ax.hist(r_w, bins=80, color=PAL_BLUE, alpha=0.5, edgecolor="none")
        ax.axvline(0, color=PAL_RED, lw=2, ls="--")
        ax.set_title("Residuals -- Within-Site Pairs")
        ax.set_xlabel("Observed − Predicted")
        ax.text(0.02, 0.95, f"n={len(r_w):,}\nmean={r_w.mean():.4f}\nsd={r_w.std():.4f}",
                transform=ax.transAxes, fontsize=8, va="top")

        # Between-site residuals
        ax = axes[0, 1]
        r_b = residuals[~is_within]
        ax.hist(r_b, bins=80, color=PAL_ORANGE, alpha=0.5, edgecolor="none")
        ax.axvline(0, color=PAL_RED, lw=2, ls="--")
        ax.set_title("Residuals -- Between-Site Pairs")
        ax.set_xlabel("Observed − Predicted")
        ax.text(0.02, 0.95, f"n={len(r_b):,}\nmean={r_b.mean():.4f}\nsd={r_b.std():.4f}",
                transform=ax.transAxes, fontsize=8, va="top")

        # Observed vs predicted scatter (subsampled)
        ax = axes[1, 0]
        rng = np.random.default_rng(123)
        n_sub = min(20000, len(p_match))
        idx_sub = rng.choice(len(p_match), n_sub, replace=False)
        colors = np.where(is_within[idx_sub], PAL_BLUE, PAL_ORANGE)
        ax.scatter(p_match[idx_sub], y_obs[idx_sub],
                   c=colors, s=2, alpha=0.15, rasterized=True)
        ax.plot([0, 1], [0, 1], color=PAL_RED, lw=2, ls="--")
        ax.set_xlabel("Predicted p_match"); ax.set_ylabel("Observed (0/1)")
        ax.set_title("Observed vs Predicted")
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)

        # Predicted p_match density by pair type
        ax = axes[1, 1]
        x_w, d_w = _safe_density(p_match[is_within], lo=0, hi=1)
        x_b, d_b = _safe_density(p_match[~is_within], lo=0, hi=1)
        ax.plot(x_w, d_w, color=PAL_BLUE, lw=2, label="Within-site")
        ax.plot(x_b, d_b, color=PAL_ORANGE, lw=2, label="Between-site")
        ax.set_xlabel("Predicted p_match"); ax.set_ylabel("Density")
        ax.set_title("Predicted p_match Distribution")
        ax.legend(fontsize=8)

        fig.suptitle(f"CLESSO NN Residual Analysis -- {species}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # A4. Classification metrics (AUC, confusion)
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # AUC: y=1 is mismatch, score = 1 - p_match (higher = more likely mismatch)
        auc = _auc_wilcoxon(y_obs, 1 - p_match)

        # Confusion at threshold 0.5
        pred_class = (p_match < 0.5).astype(int)  # predict mismatch when p_match < 0.5
        tp = np.sum((pred_class == 1) & (y_obs == 1))
        fp = np.sum((pred_class == 1) & (y_obs == 0))
        fn = np.sum((pred_class == 0) & (y_obs == 1))
        tn = np.sum((pred_class == 0) & (y_obs == 0))
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

        ax = axes[0]
        ax.axis("off")
        lines = [
            f"Classification Summary -- {species}",
            "",
            f"AUC (mismatch prediction):  {auc:.4f}",
            f"Accuracy at threshold 0.5:  {accuracy:.4f}",
            "",
            "Confusion matrix (rows=predicted, cols=observed):",
            f"                Match(0)    Mismatch(1)",
            f"  Pred Match      {tn:>8,}      {fn:>8,}",
            f"  Pred Mismatch   {fp:>8,}      {tp:>8,}",
            "",
            f"  Precision (mismatch): {tp / max(tp + fp, 1):.4f}",
            f"  Recall    (mismatch): {tp / max(tp + fn, 1):.4f}",
            f"  Specificity (match):  {tn / max(tn + fp, 1):.4f}",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, va="top", fontfamily="monospace")

        # AUC-ROC curve approximation
        ax = axes[1]
        thresholds = np.linspace(0, 1, 200)
        tpr_list = []
        fpr_list = []
        for t in thresholds:
            pred_mis = (p_match < t)
            tpr_list.append(np.sum(pred_mis & (y_obs == 1)) / max(np.sum(y_obs == 1), 1))
            fpr_list.append(np.sum(pred_mis & (y_obs == 0)) / max(np.sum(y_obs == 0), 1))
        ax.plot(fpr_list, tpr_list, color=PAL_BLUE, lw=2)
        ax.plot([0, 1], [0, 1], color=PAL_GREY, ls="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve (AUC = {auc:.4f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # A5. Spatial map of fitted alpha
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        lons = data["site_covariates"]["lon"].values
        lats = data["site_covariates"]["lat"].values

        # Map
        ax = axes[0]
        sc = ax.scatter(lons, lats, c=alpha_all, s=1.5, cmap="YlOrRd",
                        vmin=np.percentile(alpha_all, 2),
                        vmax=np.percentile(alpha_all, 98),
                        rasterized=True)
        fig.colorbar(sc, ax=ax, label="Alpha (richness)", shrink=0.8)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"Fitted Alpha (Richness) -- {species}")
        ax.set_aspect("equal")

        # Histogram
        ax = axes[1]
        ax.hist(alpha_all, bins=80, color=PAL_ORANGE, alpha=0.6, edgecolor="none")
        ax.axvline(alpha_all.mean(), color=PAL_RED, lw=2, ls="--",
                   label=f"mean={alpha_all.mean():.1f}")
        ax.axvline(np.median(alpha_all), color=PAL_BLUE, lw=2, ls=":",
                   label=f"median={np.median(alpha_all):.1f}")
        ax.set_xlabel("Alpha (richness)"); ax.set_ylabel("Sites")
        ax.set_title("Distribution of Fitted Alpha")
        ax.legend(fontsize=9)
        ax.text(0.98, 0.95,
                f"range=[{alpha_all.min():.1f}, {alpha_all.max():.1f}]",
                transform=ax.transAxes, fontsize=8, ha="right", va="top")

        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # A6. Alpha vs observed richness
        # ==============================================================
        if data["site_obs_richness"] is not None:
            obs_rich = data["site_obs_richness"]
            s_obs_map = dict(zip(obs_rich["site_id"], obs_rich["S_obs"]))
            s_obs = np.array([s_obs_map.get(sid, np.nan) for sid in site_ids],
                             dtype=np.float32)
            mask = np.isfinite(s_obs) & (s_obs > 0)

            if mask.any():
                fig, axes = plt.subplots(1, 3, figsize=(16, 6))

                # Scatter
                ax = axes[0]
                ax.scatter(s_obs[mask], alpha_all[mask], s=2, alpha=0.2,
                           color=PAL_BLUE, rasterized=True)
                lo = min(s_obs[mask].min(), alpha_all[mask].min())
                hi = max(s_obs[mask].max(), alpha_all[mask].max())
                ax.plot([lo, hi], [lo, hi], color=PAL_RED, lw=2, ls="--",
                        label="1:1 line")
                corr = np.corrcoef(alpha_all[mask], s_obs[mask])[0, 1]
                ax.set_xlabel("Observed S_obs")
                ax.set_ylabel("Predicted Alpha")
                ax.set_title(f"Alpha vs Observed Richness (r={corr:.3f})")
                ax.legend(fontsize=8)

                # Ratio histogram
                ax = axes[1]
                ratio = alpha_all[mask] / s_obs[mask]
                ax.hist(ratio, bins=80, color=PAL_GREEN, alpha=0.6, edgecolor="none")
                ax.axvline(1, color=PAL_RED, lw=2, ls="--")
                ax.axvline(np.median(ratio), color=PAL_BLUE, lw=2, ls=":",
                           label=f"median={np.median(ratio):.2f}")
                ax.set_xlabel("Alpha / S_obs"); ax.set_ylabel("Sites")
                ax.set_title("Predicted / Observed Richness Ratio")
                ax.legend(fontsize=8)

                # Spatial map of ratio
                ax = axes[2]
                fin = mask
                sc = ax.scatter(lons[fin], lats[fin], c=np.log2(ratio),
                                s=1.5, cmap="RdBu_r",
                                vmin=-2, vmax=2, rasterized=True)
                fig.colorbar(sc, ax=ax, label="log₂(α / S_obs)", shrink=0.8)
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                ax.set_title("Spatial Pattern: Alpha vs Observed")
                ax.set_aspect("equal")

                fig.suptitle(f"Alpha vs Observed Richness -- {species}", fontsize=13)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # B1. Beta monotone response curves per env dimension
        # ==============================================================
        print("  Generating beta response curves...")
        K_env = cfg_model["K_env"]
        all_env_names = list(env_names) + geo_names + geo_dist_names
        # Pad names if needed
        while len(all_env_names) < K_env:
            all_env_names.append(f"dim_{len(all_env_names)}")

        mono = check_monotonicity(model, K_env, n_points=500, device="cpu")

        # Layout: up to 3 cols per row
        n_dims = K_env
        n_cols = min(n_dims, 3)
        n_rows = int(np.ceil(n_dims / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for dim in range(n_dims):
            ax = axes[dim // n_cols, dim % n_cols]
            info = mono[dim]
            d = info["distances"]
            e = info["eta"]
            ax.plot(d, e, color=PAL_BLUE, lw=2.5)
            status = "✓ monotone" if info["is_monotone"] else "✗ violation"
            ax.set_title(f"{all_env_names[dim]}  [{status}]", fontsize=9)
            ax.set_xlabel(f"|Δ {all_env_names[dim]}| (standardised)")
            ax.set_ylabel("η (turnover)")
            ax.axhline(0, color=PAL_GREY, ls=":", lw=0.5)

        # Hide unused axes
        for i in range(n_dims, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        fig.suptitle(f"Beta Turnover Response Curves -- {species}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # B2. Similarity decay curves S = exp(-eta)
        # ==============================================================
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for dim in range(n_dims):
            ax = axes[dim // n_cols, dim % n_cols]
            info = mono[dim]
            d = info["distances"]
            S = np.exp(-info["eta"])
            ax.plot(d, S, color=PAL_BLUE, lw=2.5)
            ax.set_ylim(0, 1)
            ax.set_title(all_env_names[dim], fontsize=9)
            ax.set_xlabel(f"|Δ {all_env_names[dim]}| (standardised)")
            ax.set_ylabel("Similarity S = exp(−η)")

            # Baseline S at distance=0
            S0 = S[0]
            ax.axhline(S0, color=PAL_GREY, ls="--", lw=1,
                       label=f"S₀={S0:.3f}")
            ax.legend(fontsize=7, loc="upper right")

        for i in range(n_dims, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        fig.suptitle(f"Compositional Similarity Decay -- {species}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # B3. Alpha partial-dependence plots
        # ==============================================================
        print("  Generating alpha partial-dependence plots...")
        K_alpha = cfg_model["K_alpha"]
        z_mean = np.array(stats["z_mean"], dtype=np.float32)
        z_std = np.array(stats["z_std"], dtype=np.float32)

        # Raw covariate values for the rug / range
        # Need to add Fourier features if model was trained with them
        site_cov_df = data["site_covariates"].copy()
        fourier_n_freq = stats.get("fourier_n_frequencies", 0)
        fourier_max_wl = stats.get("fourier_max_wavelength", 40.0)
        if fourier_n_freq and fourier_n_freq > 0 and "lon" in site_cov_df.columns:
            from clesso_nn.dataset import compute_fourier_features
            lon_deg = site_cov_df["lon"].values.astype(np.float32)
            lat_deg = site_cov_df["lat"].values.astype(np.float32)
            ff, ff_names = compute_fourier_features(
                lon_deg, lat_deg, fourier_n_freq, fourier_max_wl)
            for kk, name in enumerate(ff_names):
                site_cov_df[name] = ff[:, kk]

        Z_raw = site_cov_df[alpha_names].values.astype(np.float32)

        n_alpha = len(alpha_names)
        n_cols_a = min(n_alpha, 3)
        n_rows_a = int(np.ceil(n_alpha / n_cols_a))

        # -- Partial dependence on log(alpha-1) scale --
        fig, axes = plt.subplots(n_rows_a, n_cols_a,
                                 figsize=(5 * n_cols_a, 4 * n_rows_a),
                                 squeeze=False)

        for k in range(n_alpha):
            ax = axes[k // n_cols_a, k % n_cols_a]

            x_lo, x_hi = np.nanpercentile(Z_raw[:, k], [1, 99])
            x_grid = np.linspace(x_lo, x_hi, 300)

            # Standardise: vary dimension k, hold others at mean (=0 in Z space)
            Z_grid = np.zeros((300, K_alpha), dtype=np.float32)
            Z_grid[:, k] = (x_grid - z_mean[k]) / z_std[k]

            with torch.no_grad():
                alpha_pred = model._compute_alpha_env_only(
                    torch.from_numpy(Z_grid)).cpu().numpy()

            ax.plot(x_grid, alpha_pred, color=PAL_BLUE, lw=2.5)
            ax.set_xlabel(alpha_names[k])
            ax.set_ylabel("Predicted α (richness)")
            ax.set_title(f"Alpha response: {alpha_names[k]}", fontsize=9)

            # Rug plot showing data distribution
            rug_vals = Z_raw[:, k]
            rug_sub = rug_vals[::max(1, len(rug_vals)//500)]
            rug_y = ax.get_ylim()[0]
            ax.plot(rug_sub, np.full(len(rug_sub), rug_y),
                    "|", color="k", alpha=0.05, ms=5)

        for i in range(n_alpha, n_rows_a * n_cols_a):
            axes[i // n_cols_a, i % n_cols_a].set_visible(False)

        fig.suptitle(f"Alpha Partial-Dependence (others at mean) -- {species}",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # -- Same but with 2D interaction: alpha vs two covariates --
        if n_alpha >= 2:
            # Pick the two most important (by range of partial effect)
            pd_ranges = []
            for k in range(n_alpha):
                x_lo, x_hi = np.nanpercentile(Z_raw[:, k], [5, 95])
                Z_lo = np.zeros((1, K_alpha), dtype=np.float32)
                Z_hi = np.zeros((1, K_alpha), dtype=np.float32)
                Z_lo[0, k] = (x_lo - z_mean[k]) / z_std[k]
                Z_hi[0, k] = (x_hi - z_mean[k]) / z_std[k]
                with torch.no_grad():
                    a_lo = model._compute_alpha_env_only(
                        torch.from_numpy(Z_lo)).item()
                    a_hi = model._compute_alpha_env_only(
                        torch.from_numpy(Z_hi)).item()
                pd_ranges.append(abs(a_hi - a_lo))

            top2 = np.argsort(pd_ranges)[-2:][::-1]
            k1, k2 = top2[0], top2[1]

            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            n_grid = 80
            x1_range = np.linspace(*np.nanpercentile(Z_raw[:, k1], [2, 98]), n_grid)
            x2_range = np.linspace(*np.nanpercentile(Z_raw[:, k2], [2, 98]), n_grid)
            X1, X2 = np.meshgrid(x1_range, x2_range)

            Z_2d = np.zeros((n_grid * n_grid, K_alpha), dtype=np.float32)
            Z_2d[:, k1] = ((X1.ravel()) - z_mean[k1]) / z_std[k1]
            Z_2d[:, k2] = ((X2.ravel()) - z_mean[k2]) / z_std[k2]

            with torch.no_grad():
                alpha_2d = model._compute_alpha_env_only(
                    torch.from_numpy(Z_2d)).cpu().numpy()
            ALPHA = alpha_2d.reshape(n_grid, n_grid)

            cs = ax.contourf(X1, X2, ALPHA, levels=20, cmap="YlOrRd")
            fig.colorbar(cs, ax=ax, label="Predicted α")
            ax.set_xlabel(alpha_names[k1])
            ax.set_ylabel(alpha_names[k2])
            ax.set_title(
                f"Alpha Interaction Surface: {alpha_names[k1]} × {alpha_names[k2]}\n"
                f"(other covariates at mean) -- {species}",
                fontsize=10)

            pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # B4. Variable importance (beta network)
        # ==============================================================
        print("  Computing variable importance...")

        # Importance = range of eta when sweeping each dimension 0→5
        importance = np.zeros(n_dims)
        for dim in range(n_dims):
            info = mono[dim]
            importance[dim] = info["eta"].max() - info["eta"].min()

        fig, ax = plt.subplots(figsize=(10, max(4, n_dims * 0.4)))
        order = np.argsort(importance)
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_dims))
        names_ordered = [all_env_names[i] for i in order]
        ax.barh(range(n_dims), importance[order], color=colors, edgecolor="none")
        ax.set_yticks(range(n_dims))
        ax.set_yticklabels(names_ordered, fontsize=8)
        ax.set_xlabel("η range (turnover contribution)")
        ax.set_title(f"Variable Importance (Turnover) -- {species}")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # B5. Multi-dimensional beta interaction (2D heatmap)
        # ==============================================================
        if n_dims >= 2:
            # Pick the two most important turnover dims
            top2_beta = np.argsort(importance)[-2:][::-1]
            d1, d2 = top2_beta[0], top2_beta[1]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            n_grid = 80
            r1 = np.linspace(0, 5, n_grid)
            r2 = np.linspace(0, 5, n_grid)
            G1, G2 = np.meshgrid(r1, r2)

            env_grid = np.zeros((n_grid * n_grid, K_env), dtype=np.float32)
            env_grid[:, d1] = G1.ravel()
            env_grid[:, d2] = G2.ravel()

            with torch.no_grad():
                eta_2d = model.beta_net(torch.from_numpy(env_grid)).cpu().numpy()
            ETA = eta_2d.reshape(n_grid, n_grid)
            SIM = np.exp(-ETA)

            # Eta heatmap
            ax = axes[0]
            cs = ax.contourf(G1, G2, ETA, levels=20, cmap="YlOrRd")
            fig.colorbar(cs, ax=ax, label="η (turnover)")
            ax.set_xlabel(all_env_names[d1])
            ax.set_ylabel(all_env_names[d2])
            ax.set_title("Turnover η")

            # Similarity heatmap
            ax = axes[1]
            cs = ax.contourf(G1, G2, SIM, levels=20, cmap="YlGnBu")
            fig.colorbar(cs, ax=ax, label="S = exp(−η)")
            ax.set_xlabel(all_env_names[d1])
            ax.set_ylabel(all_env_names[d2])
            ax.set_title("Similarity S")

            fig.suptitle(
                f"Beta Interaction: {all_env_names[d1]} × {all_env_names[d2]} -- {species}",
                fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # C1. Eta distribution (between-site pairs)
        # ==============================================================
        print("  Generating eta distribution diagnostic...")
        eta_between = eta[~is_within]
        sim_between = similarity[~is_within]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Eta histogram
        ax = axes[0, 0]
        ax.hist(eta_between, bins=100, color=PAL_BLUE, alpha=0.6, edgecolor="none")
        ax.axvline(eta_between.mean(), color=PAL_RED, lw=2, ls="--",
                   label=f"mean={eta_between.mean():.3f}")
        ax.axvline(np.median(eta_between), color=PAL_ORANGE, lw=2, ls=":",
                   label=f"median={np.median(eta_between):.3f}")
        ax.set_xlabel("η (turnover)")
        ax.set_ylabel("Between-site pairs")
        ax.set_title("Distribution of η (Between-Site Pairs)")
        ax.legend(fontsize=8)
        ax.text(0.98, 0.95,
                f"n={len(eta_between):,}\nmax={eta_between.max():.3f}\n"
                f"% > 0.1: {100*(eta_between > 0.1).mean():.1f}%\n"
                f"% > 1.0: {100*(eta_between > 1.0).mean():.1f}%",
                transform=ax.transAxes, fontsize=8, ha="right", va="top")

        # Similarity histogram
        ax = axes[0, 1]
        ax.hist(sim_between, bins=100, color=PAL_GREEN, alpha=0.6, edgecolor="none")
        ax.axvline(sim_between.mean(), color=PAL_RED, lw=2, ls="--",
                   label=f"mean={sim_between.mean():.3f}")
        ax.set_xlabel("S = exp(−η)")
        ax.set_ylabel("Between-site pairs")
        ax.set_title("Distribution of Compositional Similarity")
        ax.legend(fontsize=8)

        # Eta vs geographic distance (if available)
        ax = axes[1, 0]
        # Determine geographic distance from env_diff based on model configuration
        has_geo_dist_beta = getattr(site_data, "include_geo_dist_in_beta", False)
        has_old_geo = site_data.geo is not None
        geo_d_all = None

        if has_geo_dist_beta:
            # NEW approach: haversine distance is the LAST column of env_diff
            geo_d_all = env_diff[:, -1]  # already normalised by geo_dist_scale
        elif has_old_geo:
            # OLD approach: last 2 columns are |Δlon|, |Δlat| (standardised)
            geo_cols = env_diff[:, -2:]
            geo_d_all = np.sqrt((geo_cols ** 2).sum(axis=1))

        if geo_d_all is not None:
            between_mask = ~is_within
            geo_d_b = geo_d_all[between_mask]
            rng = np.random.default_rng(42)
            n_sub = min(20000, len(eta_between))
            idx = rng.choice(len(eta_between), n_sub, replace=False)
            ax.scatter(geo_d_b[idx], eta_between[idx], s=1, alpha=0.1,
                       color=PAL_BLUE, rasterized=True)
            ax.set_xlabel("Geographic distance (standardised)")
            ax.set_ylabel("η (turnover)")
            ax.set_title("η vs Geographic Distance")
        else:
            ax.text(0.5, 0.5, "No geographic distance available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("η vs Geographic Distance")

        # Eta CDF
        ax = axes[1, 1]
        eta_sorted = np.sort(eta_between)
        cdf = np.arange(1, len(eta_sorted) + 1) / len(eta_sorted)
        # Subsample for plotting efficiency
        step = max(1, len(eta_sorted) // 2000)
        ax.plot(eta_sorted[::step], cdf[::step], color=PAL_BLUE, lw=2)
        ax.axhline(0.5, color=PAL_GREY, ls=":", lw=1)
        ax.axvline(np.median(eta_between), color=PAL_RED, ls="--", lw=1,
                   label=f"median={np.median(eta_between):.3f}")
        ax.set_xlabel("η (turnover)")
        ax.set_ylabel("Cumulative proportion")
        ax.set_title("CDF of η (Between-Site Pairs)")
        ax.legend(fontsize=8)

        fig.suptitle(f"Turnover (η) Diagnostics -- {species}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # C2. Per-dimension turnover contribution
        # ==============================================================
        print("  Generating per-dimension contribution breakdown...")
        # For each between-site pair, compute the contribution of each dimension
        # by evaluating eta with only that dimension non-zero
        between_env = env_diff[~is_within]
        n_between_sub = min(5000, len(between_env))
        rng = np.random.default_rng(99)
        sub_idx = rng.choice(len(between_env), n_between_sub, replace=False)
        between_sub = between_env[sub_idx]

        dim_contributions = np.zeros((n_between_sub, n_dims))
        with torch.no_grad():
            for dim in range(n_dims):
                single = np.zeros_like(between_sub)
                single[:, dim] = between_sub[:, dim]
                dim_contributions[:, dim] = model.beta_net(
                    torch.from_numpy(single)).cpu().numpy().ravel()

        mean_contrib = dim_contributions.mean(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart of mean per-dimension contribution
        ax = axes[0]
        order = np.argsort(mean_contrib)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_dims))
        ax.barh(range(n_dims), mean_contrib[order],
                color=colors, edgecolor="none")
        ax.set_yticks(range(n_dims))
        ax.set_yticklabels([all_env_names[i] for i in order], fontsize=8)
        ax.set_xlabel("Mean η contribution (single-dim)")
        ax.set_title("Per-Dimension Turnover Contribution")

        # Stacked proportion (relative contribution)
        ax = axes[1]
        total_contrib = mean_contrib.sum()
        if total_contrib > 0:
            pct = 100 * mean_contrib / total_contrib
            pct_order = pct[order]
            cumul = np.zeros(n_dims)
            colors_r = [colors[i] for i in range(n_dims)]
            for i, dim_i in enumerate(order):
                ax.barh(0, pct[dim_i], left=cumul[0] if i == 0 else cumul[i],
                        color=colors[i], edgecolor="none",
                        label=f"{all_env_names[dim_i]} ({pct[dim_i]:.1f}%)")
                if i == 0:
                    cumul[i] = pct[dim_i]
                else:
                    cumul[i] = cumul[i-1] + pct[dim_i]
            ax.set_xlim(0, 100)
            ax.set_xlabel("% of total turnover")
            ax.set_title("Relative Contributions")
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.set_yticks([])

        fig.suptitle(f"Per-Dimension Turnover Contributions -- {species}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.85, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # C3. Classification performance by pair type
        # ==============================================================
        print("  Generating pair-type classification breakdown...")
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        for i_ax, (label, mask_sel, color) in enumerate([
            ("Within-Site", is_within, PAL_BLUE),
            ("Between-Site", ~is_within, PAL_ORANGE),
            ("All Pairs", np.ones(len(y_obs), dtype=bool), PAL_GREEN),
        ]):
            ax = axes[i_ax]
            y_sel = y_obs[mask_sel]
            p_sel = p_match[mask_sel]

            auc_sel = _auc_wilcoxon(y_sel, 1 - p_sel)
            acc_sel = np.mean((p_sel >= 0.5) == (y_sel == 0))

            # Mini ROC curve
            thresholds = np.linspace(0, 1, 200)
            tpr_l, fpr_l = [], []
            for t in thresholds:
                pred_mis = (p_sel < t)
                tpr_l.append(np.sum(pred_mis & (y_sel == 1)) / max(np.sum(y_sel == 1), 1))
                fpr_l.append(np.sum(pred_mis & (y_sel == 0)) / max(np.sum(y_sel == 0), 1))
            ax.plot(fpr_l, tpr_l, color=color, lw=2)
            ax.plot([0, 1], [0, 1], color=PAL_GREY, ls="--", lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"{label}\nAUC={auc_sel:.4f}  Acc={acc_sel:.4f}")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.text(0.98, 0.05, f"n={len(y_sel):,}\nmatch={int(y_sel.sum()):,}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom")

        fig.suptitle(f"ROC by Pair Type -- {species}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close(fig)

        # ==============================================================
        # C4. Predicted p_match vs alpha (between-site pairs)
        # ==============================================================
        print("  Generating p_match vs alpha diagnostic...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # p_match vs harmonic mean alpha for between-site pairs
        ax = axes[0]
        alpha_h = 2.0 * alpha_i[~is_within] * alpha_j[~is_within] / (
            alpha_i[~is_within] + alpha_j[~is_within])
        p_b = p_match[~is_within]
        n_sub = min(20000, len(alpha_h))
        rng = np.random.default_rng(77)
        idx_sub = rng.choice(len(alpha_h), n_sub, replace=False)
        ax.scatter(alpha_h[idx_sub], p_b[idx_sub], s=1, alpha=0.1,
                   color=PAL_BLUE, rasterized=True)
        ax.set_xlabel("Harmonic mean α (pair)")
        ax.set_ylabel("Predicted p_match")
        ax.set_title("p_match vs Alpha (Between-Site)")
        ax.set_ylim(0, 1)

        # Similarity vs alpha
        ax = axes[1]
        ax.scatter(alpha_h[idx_sub], sim_between[idx_sub], s=1, alpha=0.1,
                   color=PAL_GREEN, rasterized=True)
        ax.set_xlabel("Harmonic mean α (pair)")
        ax.set_ylabel("Similarity S = exp(−η)")
        ax.set_title("Similarity vs Alpha (Between-Site)")
        ax.set_ylim(0, 1)

        fig.suptitle(f"p_match and Similarity Structure -- {species}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0_total
    print(f"\n  Saved: {pdf_path}")
    print(f"  Completed in {elapsed:.1f}s")


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLESSO NN Diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--export-dir", required=True,
        help="Path to the nn_export directory (feather files from R)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to best_model.pt. Default: auto-detect from output dir",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save the diagnostics PDF. Default: same as checkpoint dir",
    )

    args = parser.parse_args()
    run_diagnostics(args.export_dir, args.checkpoint, args.output_dir)
