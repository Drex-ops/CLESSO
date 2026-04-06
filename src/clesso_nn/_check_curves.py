"""Quick check of beta response curve shapes."""
import json
import numpy as np

meta = json.load(open("src/clesso_v2/output/VAS_20260310_092634/nn_export/metadata.json"))
env_cols = meta["env_cov_cols"] + ["geo_x", "geo_y"]

with open("src/clesso_nn/output/VAS_nn/monotonicity.json") as f:
    data = json.load(f)

print("Dim | Variable       | eta_range | R2_linear | Shape               | Cumulative at 25/50/75%")
print("=" * 95)
for dim_key in sorted(data.keys(), key=int):
    d = data[dim_key]
    etas = np.array(d["eta"])
    dists = np.array(d["distances"])
    eta_range = etas[-1] - etas[0]
    if eta_range < 0.01:
        continue
    # R^2 of linear fit
    coeffs = np.polyfit(dists, etas, 1)
    linear_pred = np.polyval(coeffs, dists)
    ss_res = np.sum((etas - linear_pred) ** 2)
    ss_tot = np.sum((etas - np.mean(etas)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    idx = int(dim_key)
    name = env_cols[idx] if idx < len(env_cols) else f"dim_{dim_key}"
    if r2 > 0.999:
        shape = "LINEAR"
    elif r2 > 0.99:
        shape = "nearly linear"
    elif r2 > 0.95:
        shape = "slightly curved"
    else:
        shape = "STRONGLY curved"
    n = len(etas)
    e25 = etas[n // 4] / max(etas[-1], 1e-10)
    e50 = etas[n // 2] / max(etas[-1], 1e-10)
    e75 = etas[3 * n // 4] / max(etas[-1], 1e-10)
    print(f"{idx:>3} | {name:<14} | {eta_range:>9.3f} | {r2:.6f}  | {shape:<19} | {e25:.2f} / {e50:.2f} / {e75:.2f}")
