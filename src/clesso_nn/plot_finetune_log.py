#!/usr/bin/env python3
"""
Training-log dashboard for CLESSO-NN fine-tune phase.

Reads a training_progress_finetune.log CSV (written by train_finetune_joint
or train_finetune_geo) and produces a multi-panel figure showing the key
metrics with coloured warning bands.

Supports both "joint" and "geo" finetune log formats — auto-detected from
column names.

Usage
-----
    # Default: reads from the standard output directory
    python src/clesso_nn/plot_finetune_log.py

    # Custom log path
    python src/clesso_nn/plot_finetune_log.py /path/to/training_progress_finetune.log

    # Continuous monitoring (re-plot every N seconds)
    python src/clesso_nn/plot_finetune_log.py --watch 30

    # Save to file instead of interactive display
    python src/clesso_nn/plot_finetune_log.py --save finetune_dashboard.png
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
    / "training_progress_finetune.log"
)
BATCH_SIZE = 8192  # for ESS ratio calculation


# ── Colour palette ────────────────────────────────────────────────────
GREEN  = "#2ecc71"
AMBER  = "#f39c12"
RED    = "#e74c3c"
BLUE   = "#3498db"
PURPLE = "#9b59b6"
GREY   = "#95a5a6"
NAVY   = "#2c3e50"
BG_GREEN  = "#eafaf1"
BG_AMBER  = "#fef9e7"
BG_RED    = "#fdedec"

# ── Thresholds ────────────────────────────────────────────────────────
ESS_HEALTHY   = 0.30  # ESS/batch > 30% → green
ESS_WATCH     = 0.10  # 10-30% → amber; <10% → red

AUC_GOOD      = 0.70
AUC_OK        = 0.60
AUC_BAD       = 0.55


def read_log(path: Path) -> pd.DataFrame:
    """Read the CSV log, handling wrapped long lines if present."""
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in ("timestamp",):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _detect_mode(df: pd.DataFrame) -> str:
    """Detect whether this is a 'joint' or 'geo' finetune log."""
    if "lr_alpha" in df.columns:
        return "joint"
    elif "lr_existing" in df.columns:
        return "geo"
    return "unknown"


def _status_badge(ax, text: str, colour: str, x: float = 0.98, y: float = 0.92):
    """Draw a small coloured status badge in data-axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=colour, alpha=0.7,
                      ec="none"),
            color="white")


def _add_auc_bands(ax):
    """Add horizontal background bands for AUC thresholds."""
    ax.axhspan(0.0, AUC_BAD,  color=BG_RED,   alpha=0.4, zorder=0)
    ax.axhspan(AUC_BAD, AUC_OK, color=BG_AMBER, alpha=0.4, zorder=0)
    ax.axhspan(AUC_OK, AUC_GOOD, color=BG_GREEN, alpha=0.2, zorder=0)
    ax.axhspan(AUC_GOOD, 1.0,  color=BG_GREEN, alpha=0.4, zorder=0)


def _add_ess_bands(ax, batch_size: int):
    """Add horizontal background bands for ESS health thresholds."""
    ymax = batch_size * 1.05
    ax.axhspan(0, ESS_WATCH * batch_size,
               color=BG_RED, alpha=0.4, zorder=0)
    ax.axhspan(ESS_WATCH * batch_size, ESS_HEALTHY * batch_size,
               color=BG_AMBER, alpha=0.4, zorder=0)
    ax.axhspan(ESS_HEALTHY * batch_size, ymax,
               color=BG_GREEN, alpha=0.3, zorder=0)


def plot_finetune_dashboard(df: pd.DataFrame, batch_size: int = BATCH_SIZE,
                            save_path: str | None = None):
    """
    Produce a 6-panel training dashboard (3×2) for finetune phase.

    Panels
    ------
    1. Validation Loss (train + val, with best marker)
    2. AUC  (with threshold bands; s1-3 if available)
    3. Alpha distribution (mean ± std, min/max)
    4. ESS  (with batch-fraction bands)
    5. Eta mean + std (turnover diagnostic)
    6. Learning rates + accuracy
    """

    mode = _detect_mode(df)
    mode_label = {"joint": "Joint", "geo": "Geographic"}.get(mode, "Unknown")

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), dpi=120)
    fig.suptitle(f"CLESSO-NN Fine-tune Dashboard  ({mode_label} mode)",
                 fontsize=13, fontweight="bold", y=0.98)

    # Subtitle with latest epoch and timestamp
    latest = df.iloc[-1]
    n_epochs = int(latest["epoch"])
    subtitle = f"Epoch {n_epochs}  |  {latest.get('timestamp', '')}"
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9, color=GREY)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)

    epochs = df["epoch"]

    # ── Panel 1: Validation Loss ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, df["train_loss"], "o-", ms=2, lw=1.0,
            color=PURPLE, alpha=0.6, label="Train loss")
    ax.plot(epochs, df["val_loss"], "o-", ms=2, lw=1.2,
            color=BLUE, label="Val loss (all pairs)")

    # Mark best epoch
    best_idx = df["val_loss"].idxmin()
    if pd.notna(best_idx):
        best_row = df.loc[best_idx]
        ax.axvline(best_row["epoch"], color=GREEN, ls="--", lw=0.8, alpha=0.5)
        ax.plot(best_row["epoch"], best_row["val_loss"],
                "*", ms=12, color=GREEN, zorder=5, label="Best")
        ax.annotate(f"best={best_row['val_loss']:.4f}\nep {int(best_row['epoch'])}",
                    xy=(best_row["epoch"], best_row["val_loss"]),
                    fontsize=7, color=GREEN, ha="left",
                    xytext=(8, 8), textcoords="offset points")

    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("Train / Val Loss", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

    # Status badge: trend over last 10 epochs
    if len(df) >= 10:
        recent = df["val_loss"].tail(10)
        slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
        if slope < -1e-5:
            _status_badge(ax, "IMPROVING", GREEN)
        elif slope > 1e-4:
            _status_badge(ax, "WORSENING", RED)
        else:
            _status_badge(ax, "PLATEAU", AMBER)

    # ── Panel 2: AUC ─────────────────────────────────────────────────
    ax = axes[0, 1]
    _add_auc_bands(ax)
    has_s123 = "val_auc_s123" in df.columns and df["val_auc_s123"].notna().any()
    has_auc = "val_auc" in df.columns and df["val_auc"].notna().any()

    if has_auc:
        overall_alpha = 0.35 if has_s123 else 1.0
        ax.plot(epochs, df["val_auc"],
                "o-", ms=2, lw=1.0, color=GREY, alpha=overall_alpha,
                label="AUC (all strata)")
    if has_s123:
        ax.plot(epochs, df["val_auc_s123"],
                "o-", ms=2, lw=1.4, color=BLUE, label="AUC s1-3")
        # Best AUC marker
        best_auc_idx = df["val_auc_s123"].idxmax()
        if pd.notna(best_auc_idx):
            best_auc_row = df.loc[best_auc_idx]
            ax.plot(best_auc_row["epoch"], best_auc_row["val_auc_s123"],
                    "*", ms=12, color=GREEN, zorder=5, label="Best")

    primary_auc_col = "val_auc_s123" if has_s123 else "val_auc"
    if primary_auc_col in df.columns and df[primary_auc_col].notna().any():
        current_auc = df[primary_auc_col].iloc[-1]
        lbl = f"AUC {'s1-3 ' if has_s123 else ''}{current_auc:.3f}"
        if current_auc >= AUC_GOOD:
            _status_badge(ax, lbl, GREEN)
        elif current_auc >= AUC_OK:
            _status_badge(ax, lbl, AMBER)
        else:
            _status_badge(ax, lbl, RED)

    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Validation AUC (between-site)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    # Dynamic y-axis
    auc_cols = [c for c in ["val_auc", "val_auc_s123"] if c in df.columns]
    if auc_cols:
        auc_min = df[auc_cols].min().min()
        auc_max = df[auc_cols].max().max()
        if pd.notna(auc_min) and pd.notna(auc_max):
            ax.set_ylim(min(0.20, auc_min - 0.02), max(0.75, auc_max + 0.02))

    # ── Panel 3: Alpha ────────────────────────────────────────────────
    ax = axes[1, 0]
    if "alpha_mean" in df.columns:
        ax.plot(epochs, df["alpha_mean"],
                "o-", ms=2, lw=1.2, color=BLUE, label="Mean α")
        if "alpha_std" in df.columns:
            ax.fill_between(epochs,
                            df["alpha_mean"] - df["alpha_std"],
                            df["alpha_mean"] + df["alpha_std"],
                            alpha=0.25, color=BLUE, label="±1 SD")
        if "alpha_min" in df.columns and "alpha_max" in df.columns:
            ax.fill_between(epochs,
                            df["alpha_min"], df["alpha_max"],
                            alpha=0.10, color=BLUE, label="Min–Max")

        # Annotate latest
        last = df.iloc[-1]
        ax.annotate(f"μ={last['alpha_mean']:.0f}\nσ={last.get('alpha_std', 0):.0f}",
                    xy=(last["epoch"], last["alpha_mean"]),
                    fontsize=7, color=BLUE, ha="left",
                    xytext=(5, 0), textcoords="offset points")

        # Stability badge
        if len(df) >= 10:
            recent_means = df["alpha_mean"].tail(10)
            cv = recent_means.std() / recent_means.mean() if recent_means.mean() > 0 else 0
            if cv < 0.02:
                _status_badge(ax, "STABLE", GREEN)
            elif cv < 0.10:
                _status_badge(ax, "DRIFTING", AMBER)
            else:
                _status_badge(ax, "UNSTABLE", RED)

    ax.set_ylabel("Alpha (species richness)", fontsize=9)
    ax.set_title("Alpha: Mean ± SD  [Min, Max]", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel 4: ESS ─────────────────────────────────────────────────
    ax = axes[1, 1]
    _add_ess_bands(ax, batch_size)
    if "weight_ess" in df.columns:
        ax.plot(epochs, df["weight_ess"],
                "o-", ms=2, lw=1.2, color=BLUE)
        ax.axhline(ESS_HEALTHY * batch_size, color=GREEN, ls="--", lw=0.8,
                    alpha=0.6, label=f"Healthy ({ESS_HEALTHY:.0%})")
        ax.axhline(ESS_WATCH * batch_size, color=RED, ls="--", lw=0.8,
                    alpha=0.6, label=f"Danger ({ESS_WATCH:.0%})")
        ax.set_ylim(0, batch_size * 1.05)
        ax.legend(fontsize=7, loc="lower right")

        # Status badge
        min_ess = df["weight_ess"].min()
        ratio = min_ess / batch_size if batch_size > 0 else 0
        if ratio >= ESS_HEALTHY:
            _status_badge(ax, f"MIN {ratio:.0%}", GREEN)
        elif ratio >= ESS_WATCH:
            _status_badge(ax, f"MIN {ratio:.0%}", AMBER)
        else:
            _status_badge(ax, f"MIN {ratio:.0%}", RED)

    ax.set_ylabel("Effective Sample Size", fontsize=9)
    ax.set_title("Weight ESS", fontsize=10, fontweight="bold")

    # ── Panel 5: Eta (turnover mean + std) ───────────────────────────
    ax = axes[2, 0]
    if "val_eta_mean" in df.columns:
        eta_mean = df["val_eta_mean"]
        valid = eta_mean.notna() & (eta_mean > 0)
        if valid.any():
            ax.semilogy(epochs[valid], eta_mean[valid],
                        "o-", ms=2, lw=1.2, color=BLUE, label="Val η mean")

        # Healthy zone shading
        ax.axhspan(0.5, 10.0, color=BG_GREEN, alpha=0.3, zorder=0)
        ax.axhspan(0.01, 0.5, color=BG_AMBER, alpha=0.3, zorder=0)
        ax.axhspan(0, 0.01, color=BG_RED, alpha=0.3, zorder=0)
        ax.axhline(10.0, color=RED, ls=":", lw=0.8, alpha=0.5, label="η ceiling")
        ax.axhline(0.5, color=AMBER, ls=":", lw=0.8, alpha=0.5, label="η low")

        # η SD on secondary axis
        if "val_eta_std" in df.columns:
            eta_std = df["val_eta_std"]
            valid_std = eta_std.notna() & (eta_std > 0)
            if valid_std.any():
                ax2 = ax.twinx()
                ax2.plot(epochs[valid_std], eta_std[valid_std],
                         "s-", ms=2, lw=1.0, color=PURPLE, alpha=0.7,
                         label="Val η SD")
                ax2.set_ylabel("η SD", fontsize=8, color=PURPLE)
                ax2.tick_params(labelsize=7, colors=PURPLE)

        # Annotate latest
        if valid.any():
            last_eta = eta_mean[valid].iloc[-1]
            ax.annotate(f"η={last_eta:.3f}",
                        xy=(epochs[valid].iloc[-1], last_eta),
                        fontsize=7, color=BLUE, ha="left",
                        xytext=(5, 0), textcoords="offset points")

        ax.legend(fontsize=7, loc="upper right")

        # Badge
        if valid.sum() >= 3:
            recent_eta = eta_mean[valid].tail(5).mean()
            if recent_eta > 8.0:
                _status_badge(ax, "SATURATED", RED)
            elif recent_eta < 0.05:
                _status_badge(ax, "COLLAPSED", RED)
            elif recent_eta < 0.5:
                _status_badge(ax, "LOW", AMBER)
            else:
                _status_badge(ax, f"η={recent_eta:.2f}", GREEN)

    ax.set_ylabel("η mean (log scale)", fontsize=9)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_title("Eta Mean — Turnover Diagnostic", fontsize=10, fontweight="bold")

    # ── Panel 6: Learning rates + accuracy ───────────────────────────
    ax = axes[2, 1]

    # LR on left axis
    if mode == "joint" and "lr_alpha" in df.columns:
        ax.semilogy(epochs, df["lr_alpha"], "-", lw=1.2,
                     color=BLUE, label="LR alpha")
        ax.semilogy(epochs, df["lr_beta"], "-", lw=1.2,
                     color=PURPLE, label="LR beta")
    elif mode == "geo" and "lr_existing" in df.columns:
        ax.semilogy(epochs, df["lr_existing"], "-", lw=1.2,
                     color=BLUE, label="LR existing")
        ax.semilogy(epochs, df["lr_new"], "-", lw=1.2,
                     color=PURPLE, label="LR new geo")

    ax.set_ylabel("Learning Rate (log)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    # Accuracy on right axis
    if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(epochs, df["val_accuracy"], "o-", ms=2, lw=1.0,
                 color=GREEN, alpha=0.7, label="Val accuracy")
        ax2.set_ylabel("Accuracy", fontsize=8, color=GREEN)
        ax2.tick_params(labelsize=7, colors=GREEN)
        ax2.set_ylim(0.4, 1.0)
        ax2.legend(fontsize=7, loc="lower right")

        # Accuracy badge
        current_acc = df["val_accuracy"].iloc[-1]
        if pd.notna(current_acc):
            if current_acc >= 0.75:
                _status_badge(ax, f"Acc {current_acc:.1%}", GREEN, x=0.50)
            elif current_acc >= 0.60:
                _status_badge(ax, f"Acc {current_acc:.1%}", AMBER, x=0.50)
            else:
                _status_badge(ax, f"Acc {current_acc:.1%}", RED, x=0.50)

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_title("Learning Rates + Accuracy", fontsize=10, fontweight="bold")

    # ── Summary box ──────────────────────────────────────────────────
    best_val = df["val_loss"].min()
    best_ep = int(df.loc[df["val_loss"].idxmin(), "epoch"]) if pd.notna(df["val_loss"].idxmin()) else "?"
    elapsed = df["elapsed_sec"].max() if "elapsed_sec" in df.columns else 0
    summary_parts = [
        f"Mode: {mode_label}",
        f"Epochs: {n_epochs}",
        f"Best val_loss: {best_val:.4f} (ep {best_ep})",
    ]
    if has_s123:
        best_auc_val = df["val_auc_s123"].max()
        summary_parts.append(f"Best AUC s1-3: {best_auc_val:.3f}")
    elif has_auc:
        best_auc_val = df["val_auc"].max()
        summary_parts.append(f"Best AUC: {best_auc_val:.3f}")
    if elapsed > 0:
        summary_parts.append(f"Time: {elapsed/60:.1f} min")
    summary_text = "  |  ".join(summary_parts)
    fig.text(0.5, 0.005, summary_text, ha="center", fontsize=8,
             color=NAVY, fontstyle="italic")

    fig.tight_layout(rect=[0, 0.025, 1, 0.95])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualise CLESSO-NN fine-tune training log")
    parser.add_argument(
        "log_path", nargs="?", default=str(DEFAULT_LOG),
        help="Path to training_progress_finetune.log "
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
        plt.ion()
        fig = None
        print(f"Watching {log_path} every {args.watch}s  (Ctrl+C to stop)")
        try:
            while True:
                df = read_log(log_path)
                if fig is not None:
                    plt.close(fig)
                fig = plot_finetune_dashboard(df, batch_size=args.batch_size,
                                              save_path=args.save)
                plt.pause(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        df = read_log(log_path)
        plot_finetune_dashboard(df, batch_size=args.batch_size,
                                save_path=args.save)


if __name__ == "__main__":
    main()
