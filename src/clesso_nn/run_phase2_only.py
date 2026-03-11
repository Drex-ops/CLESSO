#!/usr/bin/env python3
"""
run_phase2_only.py -- Re-run Phase 2 (geo finetune) using saved Phase 1 checkpoint.

Usage:
    python src/clesso_nn/run_phase2_only.py
"""
import sys
import time
import json
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.clesso_nn.config import CLESSONNConfig
from src.clesso_nn.dataset import SiteData, load_export, make_dataloaders
from src.clesso_nn.model import CLESSONet, expand_model_for_geo
from src.clesso_nn.predict import check_monotonicity, export_alpha_predictions, predict_alpha
from src.clesso_nn.train import train_finetune_geo


def main():
    cfg = CLESSONNConfig()
    cfg.export_dir = Path(project_root / "src/clesso_v2/output/VAS_20260310_092634/nn_export").resolve()
    device = cfg.resolve_device()

    print("=" * 60)
    print("  Phase 2 Only: Fine-tune with Geographic Features")
    print("=" * 60)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    print("\n--- Loading data ---")
    data = load_export(cfg.export_dir)
    print(f"  Pairs: {len(data['pairs']):,}  Sites: {len(data['site_covariates']):,}")

    # Load Phase 1 checkpoint
    phase1_ckpt_path = cfg.output_dir / "best_model_phase1.pt"
    print(f"\n--- Loading Phase 1 checkpoint: {phase1_ckpt_path} ---")
    ckpt = torch.load(phase1_ckpt_path, map_location="cpu", weights_only=False)
    phase1_loss = ckpt["val_loss"]
    K_alpha_p1 = ckpt["config"]["K_alpha"]
    K_env_p1 = ckpt["config"]["K_env"]
    print(f"  Phase 1: K_alpha={K_alpha_p1}, K_env={K_env_p1}, val_loss={phase1_loss:.6f}")

    # Rebuild Phase 1 model and load weights
    model_p1 = CLESSONet(
        K_alpha=K_alpha_p1, K_env=K_env_p1,
        alpha_hidden=cfg.alpha_hidden, beta_hidden=cfg.beta_hidden,
        alpha_dropout=cfg.alpha_dropout, beta_dropout=cfg.beta_dropout,
        alpha_activation=cfg.alpha_activation,
        alpha_lb_lambda=cfg.alpha_lower_bound_lambda,
        alpha_regression_lambda=cfg.alpha_regression_lambda,
        beta_type=cfg.beta_type, beta_n_knots=cfg.beta_n_knots,
    )
    model_p1.load_state_dict(ckpt["model_state_dict"])
    print("  Phase 1 model restored successfully")

    # Build Phase 2 SiteData with geographic features
    print("\n--- Building Phase 2 dataset ---")
    data["metadata"]["include_geo_in_beta"] = cfg.include_geo_in_beta
    data["metadata"]["include_geo_dist_in_beta"] = cfg.include_geo_dist_in_beta
    data["metadata"]["fourier_n_frequencies"] = cfg.fourier_n_frequencies
    data["metadata"]["fourier_max_wavelength"] = cfg.fourier_max_wavelength
    data["metadata"]["exclude_coords_from_alpha"] = cfg.exclude_coords_from_alpha

    site_data_geo = SiteData(
        site_covariates=data["site_covariates"],
        env_site_table=data["env_site_table"],
        site_obs_richness=data["site_obs_richness"],
        metadata=data["metadata"],
    )
    print(f"  Phase 2 K_alpha: {site_data_geo.K_alpha}  K_env: {site_data_geo.K_env}")

    # Expand model
    model = expand_model_for_geo(model_p1, site_data_geo.K_alpha, site_data_geo.K_env, cfg)

    # Fine-tune
    t0 = time.time()
    state = train_finetune_geo(
        model=model,
        pairs=data["pairs"],
        site_data=site_data_geo,
        config=cfg,
        best_phase1_loss=phase1_loss,
    )
    elapsed = time.time() - t0
    print(f"\n  Phase 2 training time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Export predictions
    print("\n--- Exporting predictions ---")
    model.to("cpu")
    alpha_path = cfg.output_dir / "alpha_predictions.feather"
    ckpt2 = torch.load(cfg.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    export_alpha_predictions(model, data["site_covariates"], ckpt2, alpha_path, device="cpu")

    # Monotonicity check
    print("\n--- Monotonicity verification ---")
    mono = check_monotonicity(model, site_data_geo.K_env, n_points=500, device="cpu")
    env_names = site_data_geo.env_cov_names + (["geo_lon", "geo_lat"] if site_data_geo.geo is not None else [])
    geo_dist_names = ["geo_dist_km"] if site_data_geo.include_geo_dist_in_beta else []
    env_names = env_names + geo_dist_names
    for dim, info in mono.items():
        name = env_names[dim] if dim < len(env_names) else f"dim_{dim}"
        status = "OK" if info["is_monotone"] else f"VIOLATION"
        print(f"  Dim {dim} ({name}): {status}")

    mono_data = {}
    for dim, info in mono.items():
        mono_data[str(dim)] = {
            "distances": info["distances"].tolist(),
            "eta": info["eta"].tolist(),
            "is_monotone": bool(info["is_monotone"]),
        }
    with open(cfg.output_dir / "monotonicity.json", "w") as f:
        json.dump(mono_data, f)

    print(f"\n  Done. Best phase2 val_loss: {state.best_val_loss:.6f}")
    print(f"  Phase 1 val_loss: {phase1_loss:.6f}")


if __name__ == "__main__":
    main()
