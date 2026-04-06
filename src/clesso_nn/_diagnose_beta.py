#!/usr/bin/env python3
"""Diagnose beta model steepness."""
import torch
import numpy as np
from src.clesso_nn.predict import load_model
from src.clesso_nn.dataset import load_export, SiteData

ckpt_path = "src/clesso_nn/output/VAS_nn/best_model.pt"
model, ckpt = load_model(ckpt_path, device="cpu")
model.eval()

stats = ckpt["site_data_stats"]
env_mean = torch.tensor(stats["e_mean"], dtype=torch.float32)
env_std = torch.tensor(stats["e_std"], dtype=torch.float32)
K = model.beta_net.K_env
print(f"K_env = {K}")

# 1. Per-dimension response curves
print("\n=== PER-DIMENSION RESPONSE CURVES ===")
print("(sweeping standardised |Δx_k| from 0 to 5)")
x_vals = torch.linspace(0, 5, 20)
for k in range(K):
    inp = torch.zeros(20, K)
    inp[:, k] = x_vals
    with torch.no_grad():
        eta = model.beta_net(inp)
    sim = torch.exp(-eta)
    vals = [0, 4, 8, 12, 19]
    parts = [f"eta({x_vals[v]:.1f})={eta[v]:.3f}" for v in vals]
    print(f"  dim {k:2d}: {' '.join(parts)} | S(5)={sim[19]:.4f}")

# 2. Between-site eta on random pairs
print("\n=== BETWEEN-SITE ETA DISTRIBUTION (5000 random pairs) ===")
export_dir = "src/clesso_v2/output/VAS_20260310_092634/nn_export"
data = load_export(export_dir)

# Infer include_geo_in_beta from checkpoint
has_geo = any("geo" in n for n in stats.get("env_cov_names", []))
data["metadata"]["include_geo_in_beta"] = has_geo

site_data = SiteData(
    site_covariates=data["site_covariates"],
    env_site_table=data["env_site_table"],
    site_obs_richness=data["site_obs_richness"],
    metadata=data["metadata"],
)

# standardise same as training uses (get_env_at_site)
# need to do it the same way SiteData does
n = site_data.n_sites
all_idx = torch.arange(n)
E_full = site_data.get_env_at_site(all_idx)  # already standardised in SiteData
print(f"  E_full shape: {E_full.shape} (should be n_sites x {K})")

np.random.seed(42)
idx_i = np.random.randint(0, n, 10000)
idx_j = np.random.randint(0, n, 10000)
mask = idx_i != idx_j
idx_i, idx_j = idx_i[mask][:5000], idx_j[mask][:5000]

env_diff = torch.abs(E_full[idx_i] - E_full[idx_j])
with torch.no_grad():
    eta = model.beta_net(env_diff)
    sim = torch.exp(-eta)

eta_np = eta.numpy()
sim_np = sim.numpy()
print(f"  eta: mean={eta_np.mean():.3f}  median={np.median(eta_np):.3f}  "
      f"p5={np.percentile(eta_np, 5):.3f}  p95={np.percentile(eta_np, 95):.3f}  "
      f"min={eta_np.min():.3f}  max={eta_np.max():.3f}")
print(f"  sim: mean={sim_np.mean():.4f}  median={np.median(sim_np):.4f}  "
      f"p5={np.percentile(sim_np, 5):.4f}  p95={np.percentile(sim_np, 95):.4f}")

# Fraction hitting the clamp
n_at_max = (eta_np >= 9.9).sum()
print(f"  eta at clamp (>=9.9): {n_at_max}/{len(eta_np)} ({100*n_at_max/len(eta_np):.1f}%)")

# 3. Env diff magnitudes
diff_np = env_diff.numpy()
print("\n=== RAW ENV DIFF MAGNITUDES (standardised) ===")
for k in range(diff_np.shape[1]):
    d = diff_np[:, k]
    print(f"  dim {k:2d}: mean={d.mean():.3f}  median={np.median(d):.3f}  "
          f"p95={np.percentile(d, 95):.3f}  max={d.max():.3f}")

# 4. Per-dim contribution to total eta
print("\n=== PER-DIM CONTRIBUTION TO TOTAL ETA ===")
total_eta = np.zeros(5000)
with torch.no_grad():
    for k in range(K):
        x_k = env_diff[:, k : k + 1]
        f_k = model.beta_net.dim_nets[k](x_k).squeeze(-1).numpy()
        print(f"  dim {k:2d}: mean_f_k={f_k.mean():.3f}  max_f_k={f_k.max():.3f}")
        total_eta += f_k
print(f"  total (before clamp): mean={total_eta.mean():.3f}  max={total_eta.max():.3f}")

# 5. What fraction of the total is near zero?
print("\n=== ETA PERCENTILES ===")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:2d}: eta={np.percentile(eta_np, p):.3f}  sim={np.exp(-np.percentile(eta_np, p)):.4f}")
