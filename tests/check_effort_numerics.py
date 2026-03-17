#!/usr/bin/env python3
"""Quick numeric check: does effort actually change alpha?"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import numpy as np

# Load checkpoint
ckpt = torch.load(
    project_root / "src/clesso_nn/output/VAS_nn/best_model.pt",
    map_location="cpu", weights_only=False,
)
cfg = ckpt["config"]
stats = ckpt["site_data_stats"]

# Rebuild model
from src.clesso_nn.model import CLESSONet
model = CLESSONet(
    K_alpha=cfg["K_alpha"], K_env=cfg["K_env"],
    alpha_hidden=cfg["alpha_hidden"],
    beta_hidden=cfg.get("beta_hidden", [64, 32, 16]),
    alpha_dropout=cfg["alpha_dropout"],
    beta_dropout=cfg["beta_dropout"],
    alpha_activation=cfg["alpha_activation"],
    beta_type=cfg.get("beta_type", "deep"),
    beta_n_knots=cfg.get("beta_n_knots", 32),
    beta_no_intercept=cfg.get("beta_no_intercept", True),
    K_effort=cfg.get("K_effort", 0),
    effort_hidden=cfg.get("effort_hidden", [64, 32]),
    effort_dropout=cfg.get("effort_dropout", 0.1),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- Test 1: synthetic data ---
Z = torch.zeros(1, cfg["K_alpha"])
W_zero = torch.zeros(1, cfg["K_effort"])
W_high = torch.full((1, cfg["K_effort"]), 2.0)
W_low = torch.full((1, cfg["K_effort"]), -2.0)

with torch.no_grad():
    env_logit = model.alpha_net.logit(Z)
    eff_zero = model.effort_net(W_zero)
    eff_high = model.effort_net(W_high)
    eff_low = model.effort_net(W_low)
    a_env = model._compute_alpha_env_only(Z)
    a_full_zero = model._compute_alpha(Z, W_zero)
    a_full_high = model._compute_alpha(Z, W_high)
    a_full_low = model._compute_alpha(Z, W_low)

print("=== Shapes ===")
print(f"  env_logit: {env_logit.shape}, effort: {eff_zero.shape}")

print("\n=== Raw logits (synthetic, mean covariates) ===")
print(f"  env_logit:     {env_logit.item():.4f}")
print(f"  effort(zero):  {eff_zero.item():.4f}")
print(f"  effort(+2std): {eff_high.item():.4f}")
print(f"  effort(-2std): {eff_low.item():.4f}")

print("\n=== Alpha values (synthetic) ===")
print(f"  env-only:            {a_env.item():.2f}")
print(f"  full (effort=0):     {a_full_zero.item():.2f}")
print(f"  full (effort=+2std): {a_full_high.item():.2f}")
print(f"  full (effort=-2std): {a_full_low.item():.2f}")

# --- Test 2: real site data ---
print("\n=== Real site data ===")
from src.clesso_nn.dataset import SiteData, load_export

data = load_export(project_root / "src/clesso_v2/output/VAS_20260310_092634/nn_export")
effort_names = stats.get("effort_cov_names", [])
sd = SiteData(
    site_covariates=data["site_covariates"],
    env_site_table=data["env_site_table"],
    site_obs_richness=data["site_obs_richness"],
    metadata=data["metadata"],
    effort_cov_names=effort_names,
)
W_real = sd.W
Z_real = sd.Z

with torch.no_grad():
    eff_logit = model.effort_net(W_real).squeeze(-1)
    env_logit_all = model.alpha_net.logit(Z_real)
    alpha_envonly = model._compute_alpha_env_only(Z_real)
    alpha_full = model._compute_alpha(Z_real, W_real)

diff = alpha_full - alpha_envonly

print(f"  env_logit:    mean={env_logit_all.mean():.3f}  std={env_logit_all.std():.3f}  range=[{env_logit_all.min():.3f}, {env_logit_all.max():.3f}]")
print(f"  effort_logit: mean={eff_logit.mean():.3f}  std={eff_logit.std():.3f}  range=[{eff_logit.min():.3f}, {eff_logit.max():.3f}]")
print(f"  alpha env_only: mean={alpha_envonly.mean():.1f}  range=[{alpha_envonly.min():.1f}, {alpha_envonly.max():.1f}]")
print(f"  alpha full:     mean={alpha_full.mean():.1f}  range=[{alpha_full.min():.1f}, {alpha_full.max():.1f}]")
print(f"  diff (full-env): mean={diff.mean():.2f}  std={diff.std():.2f}  max_abs={diff.abs().max():.2f}")
print(f"  % sites with |diff| > 1:  {100*(diff.abs() > 1).float().mean():.1f}%")
print(f"  % sites with |diff| > 10: {100*(diff.abs() > 10).float().mean():.1f}%")
