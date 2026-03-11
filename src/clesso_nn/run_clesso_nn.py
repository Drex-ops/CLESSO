#!/usr/bin/env python3
"""
run_clesso_nn.py -- End-to-end pipeline for CLESSO neural network model.

Pipeline:
  1. Load data exported from R pipeline (feather files from export_for_nn.R)
  2. Build site data structures and pair dataset
  3. Create AlphaNet + MonotoneBetaNet architecture
  4. Train with early stopping and progress logging
  5. Evaluate and export predictions
  6. Verify monotonicity of the beta (turnover) network

Usage:
  python -m src.clesso_nn.run_clesso_nn --export-dir path/to/nn_export

  Or from project root:
  python src/clesso_nn/run_clesso_nn.py --export-dir path/to/nn_export

Prerequisites:
  1. Run R pipeline Steps 1-4 (run_clesso.R up to env extraction)
  2. Run export_for_nn.R to produce feather files
  3. pip install torch pandas pyarrow numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main(export_dir: str, config_overrides: dict | None = None):
    """Run the full CLESSO NN pipeline."""

    # Add project root to path if needed
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.clesso_nn.config import CLESSONNConfig
    from src.clesso_nn.dataset import SiteData, load_export, make_dataloaders
    from src.clesso_nn.model import CLESSONet, expand_model_for_geo
    from src.clesso_nn.predict import (
        check_monotonicity,
        export_alpha_predictions,
        predict_alpha,
    )
    from src.clesso_nn.train import train, train_two_stage, train_cyclic, train_finetune_geo

    print("=" * 60)
    print("  CLESSO Neural Network Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    cfg = CLESSONNConfig()
    cfg.export_dir = Path(export_dir).resolve()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    device = cfg.resolve_device()
    print(f"\nDevice:     {device}")
    print(f"Export dir: {cfg.export_dir}")
    print(f"Output dir: {cfg.output_dir}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ------------------------------------------------------------------
    # Step 1: Load exported data
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading exported data ---")
    t0 = time.time()
    data = load_export(cfg.export_dir)
    print(f"  Pairs:       {len(data['pairs']):,}")
    print(f"  Sites:       {len(data['site_covariates']):,}")
    if data["env_site_table"] is not None:
        env_cols = [c for c in data["env_site_table"].columns if c != "site_id"]
        print(f"  Env columns: {len(env_cols)} ({', '.join(env_cols[:5])}{'...' if len(env_cols) > 5 else ''})")
    if data["site_obs_richness"] is not None:
        print(f"  Obs richness: {len(data['site_obs_richness']):,} sites")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Build site data and dataloaders
    # ------------------------------------------------------------------
    print("\n--- Step 2: Building datasets ---")
    t0 = time.time()

    # Pass config flags into metadata so SiteData can see them
    data["metadata"]["include_geo_in_beta"] = cfg.include_geo_in_beta
    data["metadata"]["exclude_coords_from_alpha"] = cfg.exclude_coords_from_alpha

    # For cyclic_finetune mode, Phase 1 uses env-only (no Fourier, no geo_dist).
    # Phase 2 will build a separate SiteData with geo features enabled.
    if cfg.training_mode == "cyclic_finetune":
        data["metadata"]["include_geo_dist_in_beta"] = False
        data["metadata"]["fourier_n_frequencies"] = 0
        data["metadata"]["fourier_max_wavelength"] = cfg.fourier_max_wavelength
        print("  [cyclic_finetune] Phase 1: env+substrate only "
              "(no Fourier, no geo_dist)")
    else:
        data["metadata"]["include_geo_dist_in_beta"] = cfg.include_geo_dist_in_beta
        data["metadata"]["fourier_n_frequencies"] = cfg.fourier_n_frequencies
        data["metadata"]["fourier_max_wavelength"] = cfg.fourier_max_wavelength

    site_data = SiteData(
        site_covariates=data["site_covariates"],
        env_site_table=data["env_site_table"],
        site_obs_richness=data["site_obs_richness"],
        metadata=data["metadata"],
    )

    print(f"  K_alpha (site covs):     {site_data.K_alpha}")
    print(f"  K_env   (pairwise env):  {site_data.K_env}")
    print(f"  Total pairs: {len(data['pairs']):,}")
    print(f"  Built in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Create model
    # ------------------------------------------------------------------
    print("\n--- Step 3: Creating model ---")
    model = CLESSONet(
        K_alpha=site_data.K_alpha,
        K_env=site_data.K_env,
        alpha_hidden=cfg.alpha_hidden,
        beta_hidden=cfg.beta_hidden,
        alpha_dropout=cfg.alpha_dropout,
        beta_dropout=cfg.beta_dropout,
        alpha_activation=cfg.alpha_activation,
        alpha_lb_lambda=cfg.alpha_lower_bound_lambda,
        alpha_regression_lambda=cfg.alpha_regression_lambda,
        beta_type=cfg.beta_type,
        beta_n_knots=cfg.beta_n_knots,
    )
    model.eta_smoothness_lambda = cfg.eta_smoothness_lambda

    n_params = sum(p.numel() for p in model.parameters())
    n_alpha = sum(p.numel() for p in model.alpha_net.parameters())
    n_beta = sum(p.numel() for p in model.beta_net.parameters())
    print(f"  Total parameters: {n_params:,}")
    print(f"  Alpha network:    {n_alpha:,}")
    print(f"  Beta network:     {n_beta:,}")
    print(f"  Architecture:")
    print(f"    Alpha: {cfg.alpha_hidden} (dropout={cfg.alpha_dropout})")
    if cfg.beta_type == "additive":
        print(f"    Beta:  additive, {cfg.beta_n_knots} knots/dim × {site_data.K_env} dims (dropout={cfg.beta_dropout})")
    else:
        print(f"    Beta:  deep monotone {cfg.beta_hidden} (dropout={cfg.beta_dropout})")
    print(f"    Beta LR multiplier: {cfg.beta_lr_mult}×")

    # ------------------------------------------------------------------
    # Step 4: Train
    # ------------------------------------------------------------------
    print(f"\n--- Step 4: Training (mode={cfg.training_mode}) ---")
    t0 = time.time()

    if cfg.training_mode == "cyclic_finetune":
        # ============================================================
        # Phase 1: Damped cyclic on env+substrate only
        # ============================================================
        print("\n" + "=" * 60)
        print("  PHASE 1: Damped Cyclic on Environmental Features")
        print("=" * 60)
        state = train_cyclic(
            model=model,
            pairs=data["pairs"],
            site_data=site_data,
            config=cfg,
        )
        phase1_loss = state.best_val_loss
        print(f"\n  Phase 1 complete: best_val_loss = {phase1_loss:.6f}")

        # Save Phase 1 checkpoint separately
        phase1_ckpt = cfg.output_dir / "best_model_phase1.pt"
        import shutil
        if (cfg.output_dir / "best_model.pt").exists():
            shutil.copy2(cfg.output_dir / "best_model.pt", phase1_ckpt)
            print(f"  Phase 1 checkpoint saved: {phase1_ckpt}")

        # ============================================================
        # Phase 2: Expand model + fine-tune with geographic features
        # ============================================================
        # Build new SiteData WITH Fourier + geo_dist
        print("\n--- Building Phase 2 dataset (with geographic features) ---")
        meta_phase2 = dict(data["metadata"])  # copy
        meta_phase2["include_geo_dist_in_beta"] = cfg.include_geo_dist_in_beta
        meta_phase2["fourier_n_frequencies"] = cfg.fourier_n_frequencies
        meta_phase2["fourier_max_wavelength"] = cfg.fourier_max_wavelength

        site_data_geo = SiteData(
            site_covariates=data["site_covariates"],
            env_site_table=data["env_site_table"],
            site_obs_richness=data["site_obs_richness"],
            metadata=meta_phase2,
        )
        print(f"  Phase 2 K_alpha: {site_data_geo.K_alpha}  "
              f"K_env: {site_data_geo.K_env}")

        # Expand model architecture to accommodate new features
        model_expanded = expand_model_for_geo(
            model, site_data_geo.K_alpha, site_data_geo.K_env, cfg,
        )

        # Fine-tune with geographic features
        state = train_finetune_geo(
            model=model_expanded,
            pairs=data["pairs"],
            site_data=site_data_geo,
            config=cfg,
            best_phase1_loss=phase1_loss,
        )

        # Use expanded model and geo site_data going forward
        model = model_expanded
        site_data = site_data_geo

    elif cfg.training_mode == "two_stage":
        state = train_two_stage(
            model=model,
            pairs=data["pairs"],
            site_data=site_data,
            config=cfg,
        )
    elif cfg.training_mode == "cyclic":
        state = train_cyclic(
            model=model,
            pairs=data["pairs"],
            site_data=site_data,
            config=cfg,
        )
    else:
        # Joint training (legacy — prone to beta collapse at large alpha)
        train_loader, val_loader, train_ds, val_ds = make_dataloaders(
            pairs=data["pairs"],
            site_data=site_data,
            val_fraction=cfg.val_fraction,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            use_unit_weights=cfg.use_unit_weights,
        )
        state = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            site_data=site_data,
            config=cfg,
        )

    training_time = time.time() - t0
    print(f"  Training time: {training_time:.0f}s ({training_time/60:.1f} min)")

    # ------------------------------------------------------------------
    # Step 5: Export predictions
    # ------------------------------------------------------------------
    print("\n--- Step 5: Exporting predictions ---")

    # Move model to CPU for prediction/export
    model.to("cpu")

    # Alpha predictions for all training sites
    alpha_path = cfg.output_dir / "alpha_predictions.feather"
    export_alpha_predictions(
        model, data["site_covariates"], 
        torch.load(cfg.output_dir / "best_model.pt", map_location="cpu", weights_only=False),
        alpha_path, device="cpu",
    )

    # Compare with observed richness
    if data["site_obs_richness"] is not None:
        ckpt = torch.load(cfg.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
        alpha_pred = predict_alpha(model, data["site_covariates"], ckpt, "cpu")

        obs_rich = data["site_obs_richness"]
        site_ids = data["site_covariates"]["site_id"].values
        s_obs_map = dict(zip(obs_rich["site_id"], obs_rich["S_obs"]))
        s_obs = np.array([s_obs_map.get(sid, 0) for sid in site_ids], dtype=np.float32)

        mask = s_obs > 0
        if mask.any():
            corr = np.corrcoef(alpha_pred[mask], s_obs[mask])[0, 1]
            ratio = np.mean(alpha_pred[mask] / s_obs[mask])
            below = np.mean(alpha_pred[mask] < s_obs[mask])
            print(f"  Alpha vs S_obs correlation: {corr:.3f}")
            print(f"  Mean alpha/S_obs ratio:     {ratio:.2f}")
            print(f"  Fraction alpha < S_obs:     {below:.1%}")

    # ------------------------------------------------------------------
    # Step 6: Monotonicity verification
    # ------------------------------------------------------------------
    print("\n--- Step 6: Monotonicity verification ---")
    mono = check_monotonicity(model, site_data.K_env, n_points=500, device="cpu")

    all_monotone = True
    for dim, info in mono.items():
        status = "OK" if info["is_monotone"] else f"VIOLATION (min_diff={info['max_violation']:.6f})"
        if not info["is_monotone"]:
            all_monotone = False
        dim_name = ""
        env_names = site_data.env_cov_names + (["geo_lon", "geo_lat"] if site_data.geo is not None else [])
        geo_dist_names = ["geo_dist_km"] if site_data.include_geo_dist_in_beta else []
        env_names = env_names + geo_dist_names
        if dim < len(env_names):
            dim_name = f" ({env_names[dim]})"
        print(f"  Dim {dim}{dim_name}: {status}")

    if all_monotone:
        print("  All dimensions monotone.")
    else:
        print("  WARNING: Some dimensions show monotonicity violations.")

    # Save monotonicity data for plotting
    mono_data = {}
    for dim, info in mono.items():
        mono_data[str(dim)] = {
            "distances": info["distances"].tolist(),
            "eta": info["eta"].tolist(),
            "is_monotone": bool(info["is_monotone"]),
        }
    with open(cfg.output_dir / "monotonicity.json", "w") as f:
        json.dump(mono_data, f)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CLESSO NN Pipeline Complete")
    print("=" * 60)
    print(f"  Best epoch:     {state.best_epoch}")
    print(f"  Best val loss:  {state.best_val_loss:.6f}")
    print(f"  Training time:  {training_time:.0f}s")
    print(f"  Output dir:     {cfg.output_dir}")
    print(f"\n  Files:")
    print(f"    best_model.pt              -- trained model checkpoint")
    print(f"    training_progress.log      -- per-epoch training log")
    print(f"    alpha_predictions.feather  -- richness estimates per site")
    print(f"    monotonicity.json          -- turnover response curves")
    print()

    # ------------------------------------------------------------------
    # Post-training: Diagnostics + Surface Predictions
    # ------------------------------------------------------------------
    import subprocess

    checkpoint_path = str(cfg.output_dir / "best_model.pt")

    # 1. Diagnostics
    print("\n" + "=" * 60)
    print("  Running diagnostics...")
    print("=" * 60)
    try:
        from src.clesso_nn.diagnostics import run_diagnostics
        run_diagnostics(
            export_dir=str(cfg.export_dir),
            checkpoint_path=checkpoint_path,
            output_dir=str(cfg.output_dir),
        )
    except Exception as e:
        print(f"  WARNING: Diagnostics failed: {e}")

    # 2. Alpha surface prediction
    print("\n" + "=" * 60)
    print("  Predicting alpha surface...")
    print("=" * 60)
    try:
        alpha_cmd = [
            sys.executable,
            str(project_root / "src" / "clesso_nn" / "predict_alpha_surface.py"),
            "--checkpoint", checkpoint_path,
        ]
        subprocess.run(alpha_cmd, check=True)
    except Exception as e:
        print(f"  WARNING: Alpha surface prediction failed: {e}")

    # 3. Beta surface prediction
    print("\n" + "=" * 60)
    print("  Predicting beta surface...")
    print("=" * 60)
    try:
        beta_cmd = [
            sys.executable,
            str(project_root / "src" / "clesso_nn" / "predict_beta_surface.py"),
            "--checkpoint", checkpoint_path,
        ]
        subprocess.run(beta_cmd, check=True)
    except Exception as e:
        print(f"  WARNING: Beta surface prediction failed: {e}")

    print("\n" + "=" * 60)
    print("  All post-training steps complete.")
    print("=" * 60)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLESSO Neural Network Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _default_export = str(
        Path(__file__).resolve().parent.parent
        / "clesso_v2/output/VAS_20260310_092634/nn_export"
    )
    parser.add_argument(
        "--export-dir", default=_default_export,
        help="Path to directory with feather files from export_for_nn.R",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None,
        help="Max epochs (joint mode), or both stage1 & stage2 max epochs (two-stage mode)")
    parser.add_argument("--patience", type=int, default=None,
        help="Early stopping patience (applied to both stages in two-stage mode)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["joint", "two_stage"],
        help="Training mode: 'joint' or 'two_stage' (default: two_stage)")
    parser.add_argument(
        "--alpha-hidden", type=str, default=None,
        help="Alpha hidden dims as comma-separated ints, e.g. '64,32,16'",
    )
    parser.add_argument(
        "--beta-hidden", type=str, default=None,
        help="Beta hidden dims as comma-separated ints, e.g. '64,32,16'",
    )

    args = parser.parse_args()

    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.epochs is not None:
        overrides["max_epochs"] = args.epochs
        overrides["stage1_max_epochs"] = args.epochs
        overrides["stage2_max_epochs"] = args.epochs
    if args.patience is not None:
        overrides["patience"] = args.patience
        overrides["stage1_patience"] = args.patience
        overrides["stage2_patience"] = args.patience
    if args.device is not None:
        overrides["device"] = args.device
    if args.run_id is not None:
        overrides["run_id"] = args.run_id
    if args.mode is not None:
        overrides["training_mode"] = args.mode
    if args.alpha_hidden is not None:
        overrides["alpha_hidden"] = [int(x) for x in args.alpha_hidden.split(",")]
    if args.beta_hidden is not None:
        overrides["beta_hidden"] = [int(x) for x in args.beta_hidden.split(",")]

    main(args.export_dir, overrides if overrides else None)
