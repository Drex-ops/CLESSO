#!/usr/bin/env python3
"""
Training-log dashboard for CLESSO-NN cyclic training.

Reads a training_progress_cyclic.log CSV (the file written live by train.py)
and produces a multi-panel figure showing the key metrics with coloured
warning bands so you can spot problems at a glance.

Usage
-----
    # Default: reads from the standard output directory
    python src/clesso_nn/plot_training_log.py

    # Custom log path
    python src/clesso_nn/plot_training_log.py /path/to/training_progress_cyclic.log

    # Continuous monitoring (re-plot every N seconds)
    python src/clesso_nn/plot_training_log.py --watch 30

    # Save to file instead of interactive display
    python src/clesso_nn/plot_training_log.py --save training_dashboard.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_LOG = (
    Path(__file__).resolve().parent / "output" / "VAS_hexBalance_nn"
    / "training_progress_cyclic.log"
)
BATCH_SIZE = 8192  # for ESS ratio calculation


# ── Colour palette ────────────────────────────────────────────────────
GREEN  = "#2ecc71"
AMBER  = "#f39c12"
RED    = "#e74c3c"
BLUE   = "#3498db"
PURPLE = "#9b59b6"
GREY   = "#95a5a6"
BG_GREEN  = "#eafaf1"
BG_AMBER  = "#fef9e7"
BG_RED    = "#fdedec"

# ── Thresholds ────────────────────────────────────────────────────────
ESS_HEALTHY   = 0.30  # ESS/batch > 30% → green
ESS_WATCH     = 0.10  # 10-30% → amber; <10% → red

AUC_GOOD      = 0.70
AUC_OK        = 0.60
AUC_BAD       = 0.55

GRAD_VANISH   = 1e-8  # gradient norm below this → vanishing


def read_log(path: Path) -> pd.DataFrame:
    """Read the CSV log, handling wrapped long lines if present."""
    df = pd.read_csv(path)
    # Ensure numeric columns are numeric (some may have trailing whitespace)
    for col in df.columns:
        if col not in ("phase", "timestamp"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_phase_shading(ax, df: pd.DataFrame):
    """Shade alpha phases in light blue, beta in light orange behind the data."""
    prev_phase = None
    start_ep = None
    for _, row in df.iterrows():
        ep = row["global_epoch"]
        phase = row["phase"]
        if phase != prev_phase:
            if prev_phase is not None:
                colour = "#d6eaf8" if prev_phase == "alpha" else "#fdebd0"
                ax.axvspan(start_ep - 0.5, ep - 0.5,
                           alpha=0.25, color=colour, lw=0)
            start_ep = ep
            prev_phase = phase
    # Final span
    if prev_phase is not None:
        colour = "#d6eaf8" if prev_phase == "alpha" else "#fdebd0"
        ax.axvspan(start_ep - 0.5, df["global_epoch"].max() + 0.5,
                   alpha=0.25, color=colour, lw=0)


def _add_cycle_boundaries(ax, df: pd.DataFrame):
    """Add thin vertical dashed lines at cycle boundaries."""
    cycles = df["cycle"].unique()
    for c in cycles[1:]:
        ep = df.loc[df["cycle"] == c, "global_epoch"].min()
        ax.axvline(ep - 0.5, color=GREY, ls="--", lw=0.5, alpha=0.5)


def _add_ess_bands(ax, batch_size: int):
    """Add horizontal background bands for ESS health thresholds."""
    ymax = batch_size * 1.05
    ax.axhspan(0, ESS_WATCH * batch_size,
               color=BG_RED, alpha=0.4, zorder=0)
    ax.axhspan(ESS_WATCH * batch_size, ESS_HEALTHY * batch_size,
               color=BG_AMBER, alpha=0.4, zorder=0)
    ax.axhspan(ESS_HEALTHY * batch_size, ymax,
               color=BG_GREEN, alpha=0.3, zorder=0)


def _add_auc_bands(ax):
    """Add horizontal background bands for AUC thresholds."""
    ax.axhspan(0.0, AUC_BAD,  color=BG_RED,   alpha=0.4, zorder=0)
    ax.axhspan(AUC_BAD, AUC_OK, color=BG_AMBER, alpha=0.4, zorder=0)
    ax.axhspan(AUC_OK, AUC_GOOD, color=BG_GREEN, alpha=0.2, zorder=0)
    ax.axhspan(AUC_GOOD, 1.0,  color=BG_GREEN, alpha=0.4, zorder=0)


def _status_badge(ax, text: str, colour: str, x: float = 0.98, y: float = 0.92):
    """Draw a small coloured status badge in data-axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=colour, alpha=0.7,
                      ec="none"),
            color="white")


def plot_dashboard(df: pd.DataFrame, batch_size: int = BATCH_SIZE,
                   save_path: str | None = None):
    """
    Produce an 8-panel training dashboard (4×2).

    Panels
    ------
    1. Val loss (alpha + beta on same axes)
    2. AUC  (beta only, with threshold bands)
    3. Alpha mean  (with min/max whiskers)
    4. ESS  (with batch-fraction bands)
    5. Eta mean (LOG SCALE, train + val, with collapse detection)
    6. Eta spatial variation (val_eta_std, linear)
    7. Anti-collapse regulariser (eta_ac_loss)
    8. Gradient norms  (alpha + beta, log scale)
    """

    alpha = df[df["phase"] == "alpha"].copy()
    beta  = df[df["phase"] == "beta"].copy()

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), dpi=120)
    fig.suptitle("CLESSO-NN Cyclic Training Dashboard", fontsize=13,
                 fontweight="bold", y=0.98)

    # Add a subtitle with latest epoch and timestamp
    latest = df.iloc[-1]
    subtitle = (f"Global epoch {int(latest['global_epoch'])}  |  "
                f"Cycle {int(latest['cycle'])}  |  "
                f"{latest['timestamp']}")
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9, color=GREY)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)

    # ── Panel 1: Validation Loss ──────────────────────────────────────
    ax = axes[0, 0]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    if len(alpha):
        ax.plot(alpha["global_epoch"], alpha["val_loss"],
                "o-", ms=2, lw=1.2, color=BLUE, label="Alpha val_loss")
    if len(beta):
        ax.plot(beta["global_epoch"], beta["val_loss"],
                "o-", ms=2, lw=1.2, color=PURPLE, label="Beta val_loss")
    ax.set_ylabel("Validation Loss", fontsize=9)
    ax.set_title("Validation Loss", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

    # Status: check if beta loss is trending down over last 3 cycles
    if len(beta) >= 20:
        recent_end = beta.groupby("cycle")["val_loss"].last()
        if len(recent_end) >= 3:
            slope = np.polyfit(range(len(recent_end[-3:])), recent_end.values[-3:], 1)[0]
            if slope < -1e-6:
                _status_badge(ax, "IMPROVING", GREEN)
            elif slope > 1e-5:
                _status_badge(ax, "WORSENING", RED)
            else:
                _status_badge(ax, "PLATEAU", AMBER)

    # ── Panel 2: AUC ─────────────────────────────────────────────────
    ax = axes[0, 1]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    _add_auc_bands(ax)
    has_s123 = "val_auc_s123" in beta.columns if len(beta) else False
    if len(beta):
        # Overall AUC (muted when s123 is available)
        overall_alpha = 0.35 if has_s123 else 1.0
        ax.plot(beta["global_epoch"], beta["val_auc"],
                "o-", ms=2, lw=1.0, color=GREY, alpha=overall_alpha,
                label="AUC (all strata)")
        # Per-stratum AUC (s123) – primary metric when available
        if has_s123:
            ax.plot(beta["global_epoch"], beta["val_auc_s123"],
                    "o-", ms=2, lw=1.4, color=BLUE, label="AUC s1-3")
            cycle_end_s123 = beta.groupby("cycle").last().reset_index()
            ax.plot(cycle_end_s123["global_epoch"], cycle_end_s123["val_auc_s123"],
                    "D", ms=5, color="navy", zorder=5, label="Cycle end (s1-3)")
        # Add end-of-cycle markers for overall AUC
        cycle_end = beta.groupby("cycle").last().reset_index()
        if not has_s123:
            ax.plot(cycle_end["global_epoch"], cycle_end["val_auc"],
                    "D", ms=5, color="navy", zorder=5, label="Cycle end")
        ax.legend(fontsize=7)
        # Current AUC badge – prefer s123 when available
        if has_s123:
            current_auc_s123 = beta["val_auc_s123"].iloc[-1]
            lbl = f"AUC s1-3 {current_auc_s123:.3f}"
            if current_auc_s123 >= AUC_GOOD:
                _status_badge(ax, lbl, GREEN)
            elif current_auc_s123 >= AUC_OK:
                _status_badge(ax, lbl, AMBER)
            else:
                _status_badge(ax, lbl, RED)
        else:
            current_auc = beta["val_auc"].iloc[-1]
            if current_auc >= AUC_GOOD:
                _status_badge(ax, f"AUC {current_auc:.3f}", GREEN)
            elif current_auc >= AUC_OK:
                _status_badge(ax, f"AUC {current_auc:.3f}", AMBER)
            else:
                _status_badge(ax, f"AUC {current_auc:.3f}", RED)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Validation AUC (beta phase)", fontsize=10, fontweight="bold")
    # Dynamic y-axis: include data range even when AUC is very low
    if len(beta):
        auc_cols = ["val_auc"]
        if has_s123:
            auc_cols.append("val_auc_s123")
        auc_min = beta[auc_cols].min().min()
        auc_max = beta[auc_cols].max().max()
        ax.set_ylim(min(0.20, auc_min - 0.02),
                    max(0.75, auc_max + 0.02))
    else:
        ax.set_ylim(0.20, 0.75)

    # ── Panel 3: Alpha (species richness) ────────────────────────────
    ax = axes[1, 0]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    if len(alpha):
        ax.plot(alpha["global_epoch"], alpha["alpha_mean"],
                "o-", ms=2, lw=1.2, color=BLUE, label="Mean α")
        # ±1 std band
        ax.fill_between(alpha["global_epoch"],
                        alpha["alpha_mean"] - alpha["alpha_std"],
                        alpha["alpha_mean"] + alpha["alpha_std"],
                        alpha=0.25, color=BLUE, label="±1 SD")
        # Min/max whiskers
        ax.fill_between(alpha["global_epoch"],
                        alpha["alpha_min"], alpha["alpha_max"],
                        alpha=0.10, color=BLUE, label="Min–Max")

        # Annotate latest value
        last = alpha.iloc[-1]
        ax.annotate(f"μ={last['alpha_mean']:.0f}\nσ={last['alpha_std']:.0f}",
                    xy=(last["global_epoch"], last["alpha_mean"]),
                    fontsize=7, color=BLUE, ha="left",
                    xytext=(5, 0), textcoords="offset points")

        ax.legend(fontsize=7, loc="upper left")
        # Badge: is alpha drifting?
        cycle_means = alpha.groupby("cycle")["alpha_mean"].last()
        if len(cycle_means) >= 3:
            cv = cycle_means.std() / cycle_means.mean()
            if cv < 0.02:
                _status_badge(ax, "STABLE", GREEN)
            elif cv < 0.10:
                _status_badge(ax, "DRIFTING", AMBER)
            else:
                _status_badge(ax, "UNSTABLE", RED)
    ax.set_ylabel("Alpha (species richness)", fontsize=9)
    ax.set_title("Alpha: Mean ± SD  [Min, Max]", fontsize=10, fontweight="bold")

    # ── Panel 4: ESS ─────────────────────────────────────────────────
    ax = axes[1, 1]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    _add_ess_bands(ax, batch_size)
    ax.plot(df["global_epoch"], df["weight_ess"],
            "o-", ms=2, lw=1.2, color=BLUE)
    # Reference lines
    ax.axhline(ESS_HEALTHY * batch_size, color=GREEN, ls="--", lw=0.8,
               alpha=0.6, label=f"Healthy ({ESS_HEALTHY:.0%})")
    ax.axhline(ESS_WATCH * batch_size, color=RED, ls="--", lw=0.8,
               alpha=0.6, label=f"Danger ({ESS_WATCH:.0%})")
    ax.set_ylabel("Effective Sample Size", fontsize=9)
    ax.set_title("Weight ESS", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, batch_size * 1.05)
    # Status badge
    min_ess_ratio = df["weight_ess"].min() / batch_size
    if min_ess_ratio >= ESS_HEALTHY:
        _status_badge(ax, f"MIN {min_ess_ratio:.0%}", GREEN)
    elif min_ess_ratio >= ESS_WATCH:
        _status_badge(ax, f"MIN {min_ess_ratio:.0%}", AMBER)
    else:
        _status_badge(ax, f"MIN {min_ess_ratio:.0%} ⚠", RED)

    # ── Panel 5: Eta mean (LOG SCALE — critical collapse diagnostic) ─
    ax = axes[2, 0]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    if len(beta):
        epochs = beta["global_epoch"]

        # Train eta (from the beta training loop)
        train_eta = beta["train_eta_mean"]
        valid_train = train_eta.notna() & (train_eta > 0)
        if valid_train.any():
            ax.semilogy(epochs[valid_train], train_eta[valid_train],
                        "o-", ms=2, lw=1.2, color=PURPLE, label="Train η mean")

        # Val eta
        val_eta = beta["val_eta_mean"]
        valid_val = val_eta.notna() & (val_eta > 0)
        if valid_val.any():
            ax.semilogy(epochs[valid_val], val_eta[valid_val],
                        "s-", ms=2, lw=1.2, color=BLUE, label="Val η mean")

        # Healthy zone shading (η ∈ [0.5, 10] is ecologically reasonable)
        ax.axhspan(0.5, 10.0, color=BG_GREEN, alpha=0.3, zorder=0)
        ax.axhspan(0.01, 0.5, color=BG_AMBER, alpha=0.3, zorder=0)
        ax.axhspan(0, 0.01, color=BG_RED, alpha=0.3, zorder=0)

        # Reference lines
        ax.axhline(10.0, color=RED, ls=":", lw=0.8, alpha=0.5, label="η ceiling")
        ax.axhline(0.5, color=AMBER, ls=":", lw=0.8, alpha=0.5, label="η low")
        ax.axhline(1.0, color=GREEN, ls=":", lw=0.8, alpha=0.3)

        # Annotate latest value
        last_b = beta.iloc[-1]
        last_train_eta = last_b.get("train_eta_mean", np.nan)
        last_val_eta = last_b.get("val_eta_mean", np.nan)
        label_parts = []
        if pd.notna(last_train_eta) and last_train_eta > 0:
            label_parts.append(f"train={last_train_eta:.4f}")
        if pd.notna(last_val_eta) and last_val_eta > 0:
            label_parts.append(f"val={last_val_eta:.4f}")
        if label_parts:
            ref_val = last_val_eta if pd.notna(last_val_eta) else last_train_eta
            ax.annotate("\n".join(label_parts),
                        xy=(last_b["global_epoch"], max(ref_val, 1e-6)),
                        fontsize=7, color=BLUE, ha="left",
                        xytext=(5, 0), textcoords="offset points")

        ax.legend(fontsize=7, loc="upper right")

        # Badge: multi-level collapse detection
        if len(beta) >= 3:
            recent_eta = beta["val_eta_mean"].tail(5)
            recent_mean = recent_eta.mean()
            # Check monotonic decline (collapse signature)
            is_declining = len(recent_eta) >= 3 and all(
                recent_eta.iloc[i] >= recent_eta.iloc[i + 1]
                for i in range(len(recent_eta) - 1))
            if recent_mean > 8.0:
                _status_badge(ax, "SATURATED ⚠", RED)
            elif recent_mean < 0.05:
                _status_badge(ax, "COLLAPSED ⚠", RED)
            elif recent_mean < 0.5:
                _status_badge(ax, "COLLAPSING ⚠", RED)
            elif is_declining and recent_mean < 2.0:
                _status_badge(ax, "DECLINING", AMBER)
            elif recent_mean >= 0.5:
                _status_badge(ax, f"η={recent_mean:.2f}", GREEN)

    ax.set_ylabel("η mean (log scale)", fontsize=9)
    ax.set_xlabel("Global Epoch", fontsize=9)
    ax.set_title("Eta Mean — Collapse Diagnostic (log scale)",
                 fontsize=10, fontweight="bold")

    # ── Panel 6: Eta spatial variation (val_eta_std, linear) ─────────
    ax = axes[2, 1]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    if len(beta):
        epochs = beta["global_epoch"]
        eta_std = beta["val_eta_std"]
        valid = eta_std.notna()
        if valid.any():
            ax.plot(epochs[valid], eta_std[valid],
                    "o-", ms=2, lw=1.2, color=BLUE, label="Val η SD")

        # Also show val_eta_min and val_eta_max range
        if "val_eta_min" in beta.columns and "val_eta_max" in beta.columns:
            eta_min = beta["val_eta_min"]
            eta_max = beta["val_eta_max"]
            valid_range = eta_min.notna() & eta_max.notna()
            if valid_range.any():
                ax2 = ax.twinx()
                ax2.fill_between(epochs[valid_range],
                                 eta_min[valid_range], eta_max[valid_range],
                                 alpha=0.15, color=PURPLE, label="η range")
                ax2.plot(epochs[valid_range], eta_max[valid_range],
                         "-", lw=0.5, color=PURPLE, alpha=0.5)
                ax2.plot(epochs[valid_range], eta_min[valid_range],
                         "-", lw=0.5, color=PURPLE, alpha=0.5)
                ax2.set_ylabel("η min / max", fontsize=8, color=PURPLE)
                ax2.tick_params(labelsize=7, colors=PURPLE)
                ax2.set_yscale("log")

        # Annotate latest
        last_b = beta.iloc[-1]
        if pd.notna(last_b.get("val_eta_std")):
            ax.annotate(f"SD={last_b['val_eta_std']:.3f}",
                        xy=(last_b["global_epoch"], last_b["val_eta_std"]),
                        fontsize=7, color=BLUE, ha="left",
                        xytext=(5, 0), textcoords="offset points")

        ax.legend(fontsize=7, loc="upper left")
        # Badge: η variation should be non-trivial for a healthy model
        if len(beta) >= 5:
            recent_std = eta_std.tail(5).mean()
            if recent_std < 0.01:
                _status_badge(ax, "NO VARIATION ⚠", RED)
            elif recent_std < 0.1:
                _status_badge(ax, "LOW VARIATION", AMBER)
            else:
                _status_badge(ax, f"SD={recent_std:.2f}", GREEN)

    ax.set_ylabel("η SD (spatial variation)", fontsize=9)
    ax.set_xlabel("Global Epoch", fontsize=9)
    ax.set_title("Eta Spatial Variation", fontsize=10, fontweight="bold")

    # ── Panel 7: Anti-collapse regulariser (eta_ac_loss) ─────────────
    ax = axes[3, 0]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    has_ac = "eta_ac_loss" in df.columns
    if has_ac and len(beta):
        ac_vals = beta["eta_ac_loss"]
        valid_ac = ac_vals.notna() & (ac_vals != 0)
        if valid_ac.any():
            ax.plot(beta.loc[valid_ac, "global_epoch"],
                    ac_vals[valid_ac],
                    "o-", ms=2, lw=1.2, color=PURPLE, label="η anti-collapse")
            # Annotate latest
            last_ac = ac_vals[valid_ac].iloc[-1] if valid_ac.any() else 0
            ax.annotate(f"{last_ac:.4f}",
                        xy=(beta.loc[valid_ac, "global_epoch"].iloc[-1], last_ac),
                        fontsize=7, color=PURPLE, ha="left",
                        xytext=(5, 0), textcoords="offset points")
            ax.legend(fontsize=7)
            # Badge: if ac_loss is large, the regulariser is fighting collapse
            if last_ac > 0.1:
                _status_badge(ax, "ACTIVE ⚠", AMBER)
            elif last_ac > 0.01:
                _status_badge(ax, "MODERATE", GREEN)
            else:
                _status_badge(ax, "NEGLIGIBLE", GREEN)
        else:
            ax.text(0.5, 0.5, "No eta_ac_loss data yet",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=GREY)
    elif not has_ac:
        ax.text(0.5, 0.5, "eta_ac_loss column not in log\n(pre-fix run?)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=RED)
    else:
        ax.text(0.5, 0.5, "No beta data yet",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=GREY)
    ax.set_ylabel("Anti-collapse loss", fontsize=9)
    ax.set_xlabel("Global Epoch", fontsize=9)
    ax.set_title("Eta Anti-Collapse Regulariser", fontsize=10, fontweight="bold")

    # ── Panel 8: Gradient Norms (log scale) ──────────────────────────
    ax = axes[3, 1]
    _add_phase_shading(ax, df)
    _add_cycle_boundaries(ax, df)
    if len(alpha):
        valid_ag = alpha[alpha["alpha_grad_norm"].notna() &
                         (alpha["alpha_grad_norm"] > 0)]
        if len(valid_ag):
            ax.semilogy(valid_ag["global_epoch"], valid_ag["alpha_grad_norm"],
                        "o-", ms=2, lw=1.2, color=BLUE, label="Alpha grad")
    if len(beta):
        valid_bg = beta[beta["beta_grad_norm"].notna() &
                        (beta["beta_grad_norm"] > 0)]
        if len(valid_bg):
            ax.semilogy(valid_bg["global_epoch"], valid_bg["beta_grad_norm"],
                        "o-", ms=2, lw=1.2, color=PURPLE, label="Beta grad")
    # Vanishing gradient line
    ax.axhline(GRAD_VANISH, color=RED, ls="--", lw=0.8, alpha=0.6,
               label=f"Vanishing ({GRAD_VANISH:.0e})")
    ax.axhline(1e-5, color=AMBER, ls="--", lw=0.8, alpha=0.4,
               label="Effectively zero (1e-5)")
    ax.legend(fontsize=7)
    ax.set_ylabel("Gradient Norm (log)", fontsize=9)
    ax.set_xlabel("Global Epoch", fontsize=9)
    ax.set_title("Gradient Norms", fontsize=10, fontweight="bold")
    # Badge
    if len(beta):
        latest_bg = beta["beta_grad_norm"].dropna().iloc[-1] if len(beta) else 0
        if latest_bg < GRAD_VANISH:
            _status_badge(ax, "VANISHING ⚠", RED)
        elif latest_bg < 1e-5:
            _status_badge(ax, "NEAR ZERO ⚠", RED)
        elif latest_bg < 1e-3:
            _status_badge(ax, "SMALL", AMBER)
        else:
            _status_badge(ax, "OK", GREEN)

    # ── Legend for phase shading ──────────────────────────────────────
    alpha_patch = mpatches.Patch(color="#d6eaf8", alpha=0.5, label="Alpha phase")
    beta_patch  = mpatches.Patch(color="#fdebd0", alpha=0.5, label="Beta phase")
    fig.legend(handles=[alpha_patch, beta_patch], loc="lower center",
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.003))

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualise CLESSO-NN cyclic training log")
    parser.add_argument(
        "log_path", nargs="?", default=str(DEFAULT_LOG),
        help="Path to training_progress_cyclic.log "
             f"(default: {DEFAULT_LOG})")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size for ESS ratio (default: {BATCH_SIZE})")
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save to file (PNG/PDF) instead of interactive display")
    parser.add_argument(
        "--watch", type=int, default=None, metavar="SECS",
        help="Re-plot every SECS seconds (live monitoring mode)")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    if args.watch:
        plt.ion()  # interactive mode for live updates
        fig = None
        print(f"Watching {log_path} every {args.watch}s  (Ctrl+C to stop)")
        try:
            while True:
                df = read_log(log_path)
                if fig is not None:
                    plt.close(fig)
                fig = plot_dashboard(df, batch_size=args.batch_size,
                                     save_path=args.save)
                plt.pause(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        df = read_log(log_path)
        plot_dashboard(df, batch_size=args.batch_size, save_path=args.save)


if __name__ == "__main__":
    main()
