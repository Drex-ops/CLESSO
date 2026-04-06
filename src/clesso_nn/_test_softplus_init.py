#!/usr/bin/env python3
"""Quick check of softplus beta init."""
import torch
import numpy as np
from src.clesso_nn.model import CLESSONet

m = CLESSONet(K_alpha=17, K_env=17, beta_type="additive", beta_n_knots=32)
m.eval()  # disable dropout for clean baseline subtraction

x_vals = torch.linspace(0, 5, 20)
print("Per-dimension response (untrained init):")
for k in [0, 5, 10, 16]:
    inp = torch.zeros(20, 17)
    inp[:, k] = x_vals
    with torch.no_grad():
        eta = m.beta_net(inp)
    pts = [0, 2, 5, 9, 14, 19]
    labels = [f"{x_vals[i]:.1f}" for i in pts]
    vals = [f"{eta[i]:.3f}" for i in pts]
    print(f"  dim {k:2d}: " + "  ".join(f"eta({l})={v}" for l, v in zip(labels, vals)))

diff = torch.rand(1000, 17) * 2
with torch.no_grad():
    eta = m.beta_net(diff)
e = eta.numpy()
print(f"\nRandom pairs (0-2s range):")
print(f"  mean={e.mean():.3f}  median={np.median(e):.3f}  max={e.max():.3f}")
print(f"  eta>0.1: {(e>0.1).sum()}/1000  eta>0.5: {(e>0.5).sum()}/1000  eta>1: {(e>1).sum()}/1000")
print("All OK")
