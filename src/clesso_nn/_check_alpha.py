#!/usr/bin/env python3
"""Quick diagnostic: compare predicted alpha at training sites vs S_obs."""
import sys
import torch
import numpy as np
sys.path.insert(0, "src")
from clesso_nn.model import CLESSONet
from clesso_nn.dataset import load_export, SiteData

# Load model
ckpt = torch.load("src/clesso_nn/output/VAS_nn/best_model.pt",
                   map_location="cpu", weights_only=False)
cfg = ckpt["config"]
model = CLESSONet(
    K_alpha=cfg["K_alpha"], K_env=cfg["K_env"],
    alpha_hidden=cfg["alpha_hidden"],
    beta_hidden=cfg.get("beta_hidden", [64, 32, 16]),
    alpha_dropout=cfg["alpha_dropout"],
    beta_dropout=cfg["beta_dropout"],
    alpha_activation=cfg["alpha_activation"],
    beta_type=cfg.get("beta_type", "deep"),
    beta_n_knots=cfg.get("beta_n_knots", 32),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load site data
export_dir = "src/clesso_v2/output/VAS_20260310_092634/nn_export"
data = load_export(export_dir)
sd = SiteData(data["site_covariates"], data["env_site_table"],
              data["site_obs_richness"], data["metadata"])

# Predict alpha at all training sites
with torch.no_grad():
    alpha_pred = model.alpha_net(sd.Z).numpy().ravel()

S_obs = sd.S_obs.numpy()

print("=" * 60)
print("  Alpha Diagnostic: Predicted vs Observed at Training Sites")
print("=" * 60)

print(f"\nN sites: {len(alpha_pred)}")
print(f"\n{'':>30} {'Observed':>12} {'Predicted':>12}")
print(f"{'Range':>30} [{S_obs.min():.2f}, {S_obs.max():.2f}]  [{alpha_pred.min():.2f}, {alpha_pred.max():.2f}]")
print(f"{'Mean':>30} {S_obs.mean():12.2f} {alpha_pred.mean():12.2f}")
print(f"{'Median':>30} {np.median(S_obs):12.2f} {np.median(alpha_pred):12.2f}")
print(f"{'Std':>30} {S_obs.std():12.2f} {alpha_pred.std():12.2f}")

# Ratio
ratio = alpha_pred / np.maximum(S_obs, 1.0)
print(f"\n--- Ratio (predicted / observed) ---")
print(f"  Mean:   {ratio.mean():.3f}")
print(f"  Median: {np.median(ratio):.3f}")
print(f"  5th:    {np.percentile(ratio, 5):.3f}")
print(f"  95th:   {np.percentile(ratio, 95):.3f}")

above = (alpha_pred > S_obs).sum()
below = (alpha_pred < S_obs).sum()
equal = (alpha_pred == S_obs).sum()
print(f"\n  pred > obs: {above}/{len(alpha_pred)} ({100*above/len(alpha_pred):.1f}%)")
print(f"  pred < obs: {below}/{len(alpha_pred)} ({100*below/len(alpha_pred):.1f}%)")

corr = np.corrcoef(S_obs, alpha_pred)[0, 1]
print(f"\n  Correlation:  {corr:.4f}")
print(f"  R²:           {corr**2:.4f}")

# Show top-20 highest observed richness
print(f"\n--- Top 20 observed-richness sites ---")
print(f"  {'S_obs':>8} {'pred':>8} {'ratio':>8}")
high_idx = np.argsort(S_obs)[-20:][::-1]
for i in high_idx:
    print(f"  {S_obs[i]:8.1f} {alpha_pred[i]:8.2f} {ratio[i]:8.3f}")

# Distribution of S_obs
print(f"\n--- S_obs distribution ---")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {pct:3d}th: {np.percentile(S_obs, pct):.2f}")
print(f"  sites with S_obs == 1: {(S_obs == 1).sum()} ({100*(S_obs == 1).mean():.1f}%)")
print(f"  sites with S_obs <= 2: {(S_obs <= 2).sum()} ({100*(S_obs <= 2).mean():.1f}%)")

# Check loss components from config
print(f"\n--- Training regularisation ---")
print(f"  alpha_lb_lambda (lower-bound penalty):    {cfg.get('alpha_lb_lambda', 'N/A')}")
print(f"  alpha_regression_lambda (MSE to S_obs):   {cfg.get('alpha_regression_lambda', 'N/A')}")
print(f"  Model alpha_lb_lambda:    {model.alpha_lb_lambda}")
print(f"  Model alpha_regression_lambda: {model.alpha_regression_lambda}")

# Check what S_obs actually represents
print(f"\n--- Checking base data ---")
import pandas as pd
import pyarrow.feather as feather
rich_df = feather.read_feather(f"{export_dir}/site_obs_richness.feather")
print(f"  site_obs_richness columns: {list(rich_df.columns)}")
print(f"  S_obs describe:\n{rich_df['S_obs'].describe()}")

# Check how S_obs is computed - look at export_for_nn.R logic
pairs = feather.read_feather(f"{export_dir}/pairs.feather")
print(f"\n  Pairs columns: {list(pairs.columns)}")
print(f"  Total pairs: {len(pairs)}")
print(f"  Within-site pairs: {pairs['is_within'].sum()}")
print(f"  Between-site pairs: {(~pairs['is_within'].astype(bool)).sum()}")
print(f"  y distribution: {pairs['y'].value_counts().to_dict()}")
