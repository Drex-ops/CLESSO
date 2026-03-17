#!/usr/bin/env python3
"""Quick 2-cycle diagnostic: test different config tweaks and report eta health.

Usage:
    python scripts/quick_eta_test.py [--label LABEL] [--override KEY=VAL ...]

Examples:
    python scripts/quick_eta_test.py --label baseline
    python scripts/quick_eta_test.py --label no_lb --override alpha_lower_bound_lambda=0.0
    python scripts/quick_eta_test.py --label low_lb --override alpha_lower_bound_lambda=0.01
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

# Ensure project root on path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch


def run_test(label: str, overrides: dict):
    from src.clesso_nn.config import CLESSONNConfig
    from src.clesso_nn.dataset import SiteData, load_export
    from src.clesso_nn.model import CLESSONet
    from src.clesso_nn.train import train_cyclic

    print("=" * 60)
    print(f"  Quick Eta Test: {label}")
    print("=" * 60)

    cfg = CLESSONNConfig()

    # Force 2 cycles only
    cfg.max_cycles = 2

    # Apply overrides
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            print(f"  WARNING: unknown config key '{k}', skipping")
            continue
        # Auto-cast to the right type
        current = getattr(cfg, k)
        if isinstance(current, bool):
            v = v.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            v = int(v)
        elif isinstance(current, float):
            v = float(v)
        setattr(cfg, k, v)
        print(f"  Override: {k} = {v}")

    # Use a test-specific output dir so we don't clobber real runs
    cfg.run_id = f"_test_{label}"

    device = cfg.resolve_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    print("\nLoading data...")
    t0 = time.time()
    data = load_export(cfg.export_dir)
    data["metadata"]["include_geo_in_beta"] = cfg.include_geo_in_beta
    data["metadata"]["exclude_coords_from_alpha"] = cfg.exclude_coords_from_alpha
    data["metadata"]["include_geo_dist_in_beta"] = cfg.include_geo_dist_in_beta
    data["metadata"]["fourier_n_frequencies"] = cfg.fourier_n_frequencies
    data["metadata"]["fourier_max_wavelength"] = cfg.fourier_max_wavelength

    # Auto-detect effort
    _effort_names = cfg.effort_cov_names
    if not _effort_names:
        meta_effort = data["metadata"].get("effort_cov_cols", [])
        if meta_effort:
            _effort_names = list(meta_effort)

    site_data = SiteData(
        site_covariates=data["site_covariates"],
        env_site_table=data["env_site_table"],
        site_obs_richness=data["site_obs_richness"],
        metadata=data["metadata"],
        effort_cov_names=_effort_names if _effort_names else None,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s  "
          f"(K_alpha={site_data.K_alpha}, K_env={site_data.K_env})")

    # Create model
    model = CLESSONet(
        K_alpha=site_data.K_alpha,
        K_env=site_data.K_env,
        alpha_hidden=cfg.alpha_hidden,
        beta_hidden=cfg.beta_hidden,
        alpha_dropout=cfg.alpha_dropout,
        beta_dropout=cfg.beta_dropout,
        alpha_activation=cfg.alpha_activation,
        alpha_lb_lambda=cfg.alpha_lower_bound_lambda,
        alpha_anchor_lambda=cfg.alpha_anchor_lambda,
        alpha_anchor_tolerance=cfg.alpha_anchor_tolerance,
        alpha_regression_lambda=cfg.alpha_regression_lambda,
        beta_type=cfg.beta_type,
        beta_n_knots=cfg.beta_n_knots,
        beta_no_intercept=cfg.beta_no_intercept,
        K_effort=site_data.K_effort,
        effort_hidden=cfg.effort_hidden,
        effort_dropout=cfg.effort_dropout,
        effort_mode=cfg.effort_mode,
    )
    model.eta_smoothness_lambda = cfg.eta_smoothness_lambda
    model.effort_penalty = cfg.effort_penalty

    K_alpha_env = sum(1 for c in site_data.alpha_cov_names
                      if not c.startswith("fourier_"))
    K_env_env = len(site_data.env_cov_names)
    model.geo_penalty_alpha = cfg.geo_penalty_alpha
    model.geo_penalty_beta = cfg.geo_penalty_beta
    model.K_alpha_env = K_alpha_env
    model.K_env_env = K_env_env

    # Print sigmoid_shift for beta net
    if hasattr(model.beta_net, "sigmoid_shift"):
        print(f"  Beta sigmoid_shift (init): {model.beta_net.sigmoid_shift.item():.3f}")

    # Train 2 cycles
    print(f"\nTraining 2 cycles (alpha×{cfg.cycle_alpha_epochs} + beta×{cfg.cycle_beta_epochs})...")
    print(f"  alpha_lower_bound_lambda = {cfg.alpha_lower_bound_lambda}")
    print(f"  beta_lr_mult = {cfg.beta_lr_mult}")
    print(f"  stage2_beta_grad_scale = {cfg.stage2_beta_grad_scale}")
    print(f"  eta_smoothness_lambda = {cfg.eta_smoothness_lambda}")
    print()

    state = train_cyclic(
        model=model,
        pairs=data["pairs"],
        site_data=site_data,
        config=cfg,
    )

    # Read the log and print summary
    log_path = cfg.output_dir / "training_progress_cyclic.log"
    if log_path.exists():
        print("\n" + "=" * 60)
        print(f"  RESULTS: {label}")
        print("=" * 60)
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            ep = row["global_epoch"]
            phase = row["phase"]
            if phase == "alpha":
                print(f"  ep={ep:>3s}  alpha  "
                      f"bce={float(row['train_bce']):8.6f}  "
                      f"lb={float(row['train_lb']):8.6f}  "
                      f"alpha_mean={float(row['alpha_mean']):8.1f}  "
                      f"alpha_max={float(row['alpha_max']):8.1f}")
            else:
                eta_mean = float(row["val_eta_mean"])
                eta_min = float(row["val_eta_min"])
                beta_grad = float(row["beta_grad_norm"])
                auc = float(row["val_auc"])
                print(f"  ep={ep:>3s}  beta   "
                      f"eta_mean={eta_mean:7.4f}  "
                      f"eta_min={eta_min:8.6f}  "
                      f"beta_grad={beta_grad:.6f}  "
                      f"auc={auc:.4f}")

        # Summary verdict
        beta_rows = [r for r in rows if r["phase"] == "beta"]
        if beta_rows:
            last_eta = float(beta_rows[-1]["val_eta_mean"])
            last_grad = float(beta_rows[-1]["beta_grad_norm"])
            last_auc = float(beta_rows[-1]["val_auc"])
            alpha_rows = [r for r in rows if r["phase"] == "alpha"]
            last_alpha = float(alpha_rows[-1]["alpha_mean"]) if alpha_rows else 0

            print()
            if last_eta > 9.0:
                print(f"  ❌ ETA SATURATED (η={last_eta:.2f})")
            elif last_eta > 5.0:
                print(f"  ⚠️  ETA HIGH (η={last_eta:.2f})")
            else:
                print(f"  ✅ ETA HEALTHY (η={last_eta:.2f})")

            if last_grad < 1e-4:
                print(f"  ❌ BETA GRADIENT DEAD ({last_grad:.2e})")
            else:
                print(f"  ✅ BETA GRADIENT ALIVE ({last_grad:.2e})")

            if last_auc > 0.55:
                print(f"  ✅ AUC LEARNING ({last_auc:.4f})")
            else:
                print(f"  ❌ AUC NEAR RANDOM ({last_auc:.4f})")

            if last_alpha > 100:
                print(f"  ⚠️  ALPHA INFLATED (mean={last_alpha:.0f})")
            else:
                print(f"  ✅ ALPHA REASONABLE (mean={last_alpha:.0f})")

    # Cleanup test output
    import shutil
    test_dir = cfg.output_dir
    if "_test_" in str(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

    print()
    return state


def main():
    parser = argparse.ArgumentParser(description="Quick 2-cycle eta diagnostic")
    parser.add_argument("--label", default="baseline", help="Test label")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides as KEY=VAL")
    args = parser.parse_args()

    overrides = {}
    for item in args.override:
        if "=" in item:
            k, v = item.split("=", 1)
            overrides[k] = v
        else:
            print(f"WARNING: ignoring malformed override '{item}' (need KEY=VAL)")

    run_test(args.label, overrides)


if __name__ == "__main__":
    main()
