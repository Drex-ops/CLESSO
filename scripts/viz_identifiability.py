#!/usr/bin/env python3
"""Visualisation: alpha–beta identifiability in the p_{i,j} model.

Demonstrates that a single observed match probability p_{i,j} admits
an infinite surface of (α_i, α_j, η) solutions, and that only the
within-site pairs (which isolate α) break this degeneracy.

Model:
    p_{i,j} = e^{-η} · (α_i + α_j) / (2·α_i·α_j)

    Symmetric case (α_i = α_j = α):
        p = e^{-η} / α    →    η = −ln(p·α)

Usage:
    python3 scripts/viz_identifiability.py [output.png]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/identifiability.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────
ALPHA_MIN, ALPHA_MAX = 1.01, 600   # α > 1 by model constraint
ETA_MIN, ETA_MAX = 0.0, 10.0       # η ∈ [0, 10] by clamp

# ── Helper: solve for η given p, α_i, α_j ───────────────────────────────
def eta_from_p(p, alpha_i, alpha_j):
    """η = −ln(p · 2·α_i·α_j / (α_i + α_j))"""
    arg = p * 2.0 * alpha_i * alpha_j / (alpha_i + alpha_j)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = -np.log(arg)
    return eta

def p_from_alpha_eta(alpha_i, alpha_j, eta):
    """p = e^{-η} · (α_i + α_j) / (2·α_i·α_j)"""
    return np.exp(-eta) * (alpha_i + alpha_j) / (2.0 * alpha_i * alpha_j)


# ══════════════════════════════════════════════════════════════════════════
#  Figure: 3 panels
# ══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 5.5))
fig.suptitle(
    r"Identifiability of $\alpha$ and $\beta$ in $\,p_{i,j} = e^{-\eta}\,"
    r"\frac{\alpha_i + \alpha_j}{2\,\alpha_i\,\alpha_j}$",
    fontsize=13, fontweight="bold", y=0.98,
)

# ──────────────────────────────────────────────────────────────────────────
#  Panel A: Symmetric case — α–η tradeoff curves for fixed p
# ──────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(131)

alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, 2000)
p_values = [0.001, 0.005, 0.01, 0.02, 0.05]
colors_a = plt.cm.viridis(np.linspace(0.15, 0.85, len(p_values)))

for p_val, col in zip(p_values, colors_a):
    eta = -np.log(p_val * alpha_grid)  # symmetric: η = −ln(p·α)
    valid = (eta >= ETA_MIN) & (eta <= ETA_MAX)
    ax1.plot(alpha_grid[valid], eta[valid], color=col, linewidth=2,
             label=f"$p = {p_val}$")

ax1.set_xlabel(r"$\alpha$ (species richness)", fontsize=11)
ax1.set_ylabel(r"$\eta$ (turnover)", fontsize=11)
ax1.set_title("A.  Symmetric case  ($\\alpha_i = \\alpha_j$)", fontsize=11,
              fontweight="bold")
ax1.set_xlim(1, ALPHA_MAX)
ax1.set_ylim(ETA_MIN, ETA_MAX)
ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

# Annotate the tradeoff
ax1.annotate(
    "Each curve = ∞ solutions\nfor a single $p_{i,j}$",
    xy=(80, 4.2), fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
)
ax1.annotate(
    "", xy=(250, 1.5), xytext=(80, 3.9),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
)

# Show direction of tradeoff
ax1.annotate(
    r"$\alpha \!\uparrow$  offset by  $\eta \!\downarrow$",
    xy=(200, 2.0), fontsize=8, color="0.35", fontstyle="italic",
    rotation=-35,
)

ax1.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────────────────────
#  Panel B: Asymmetric case — solution surface for fixed p
# ──────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(132)

p_fixed = 0.01
n_grid = 300
ai = np.linspace(ALPHA_MIN, ALPHA_MAX, n_grid)
aj = np.linspace(ALPHA_MIN, ALPHA_MAX, n_grid)
AI, AJ = np.meshgrid(ai, aj)
ETA = eta_from_p(p_fixed, AI, AJ)

# Mask infeasible (η outside [0, 10])
ETA_masked = np.where((ETA >= ETA_MIN) & (ETA <= ETA_MAX), ETA, np.nan)

im = ax2.contourf(AI, AJ, ETA_masked, levels=20, cmap="magma_r")
contour_lines = ax2.contour(AI, AJ, ETA_masked, levels=[1, 2, 3, 4, 5, 6, 7, 8],
                            colors="white", linewidths=0.6, alpha=0.7)
ax2.clabel(contour_lines, fontsize=7, fmt=r"$\eta$=%.0f", colors="white")

cbar = fig.colorbar(im, ax=ax2, shrink=0.85, pad=0.02)
cbar.set_label(r"$\eta$ (turnover)", fontsize=10)

ax2.set_xlabel(r"$\alpha_i$", fontsize=11)
ax2.set_ylabel(r"$\alpha_j$", fontsize=11)
ax2.set_title(f"B.  Solution surface for $p = {p_fixed}$", fontsize=11,
              fontweight="bold")

# Mark example solutions that all give the same p
example_alphas = [(10, 10), (50, 50), (100, 200), (300, 300)]
for ai_ex, aj_ex in example_alphas:
    eta_ex = eta_from_p(p_fixed, ai_ex, aj_ex)
    if ETA_MIN <= eta_ex <= ETA_MAX:
        ax2.plot(ai_ex, aj_ex, "o", color="cyan", markersize=5, markeredgecolor="white",
                 markeredgewidth=1.0, zorder=5)

ax2.annotate(
    f"Every coloured point\ngives $p = {p_fixed}$\n(2D manifold of solutions)",
    xy=(350, 40), fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
)

ax2.grid(True, alpha=0.15, color="white")

# ──────────────────────────────────────────────────────────────────────────
#  Panel C: How within-site pairs break the degeneracy
# ──────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(133)

# Show the solution manifold as a shaded region vs. a constrained solution
alpha_grid_c = np.linspace(ALPHA_MIN, ALPHA_MAX, 2000)
p_obs = 0.01  # observed p for a between-site pair

# The full curve of (α, η) solutions (symmetric case for clarity)
eta_curve = -np.log(p_obs * alpha_grid_c)
valid = (eta_curve >= ETA_MIN) & (eta_curve <= ETA_MAX)

# ── Scenario 1: Between-site only (full curve = degenerate) ──
ax3.fill_between(alpha_grid_c[valid], 0, eta_curve[valid],
                 color="red", alpha=0.10, label="Degenerate region")
ax3.plot(alpha_grid_c[valid], eta_curve[valid], color="red", linewidth=2.5,
         alpha=0.7, label=f"Between only ($p_{{i,j}}={p_obs}$)")

# ── Scenario 2: Within-site pair pins α ──
# If within-site p_{i,i} = 1/α_i is observed, α_i is identified
alpha_true = 80
eta_true = -np.log(p_obs * alpha_true)

# Within-site observation gives α = 80 ± some uncertainty
alpha_lo, alpha_hi = 60, 105  # plausible range from within-site data
eta_lo = -np.log(p_obs * alpha_hi)
eta_hi = -np.log(p_obs * alpha_lo)

# Shade the constrained region
ax3.axvspan(alpha_lo, alpha_hi, color="blue", alpha=0.08)
ax3.axvline(alpha_true, color="blue", linewidth=1.5, linestyle="--", alpha=0.7)

# The constrained solution
ax3.plot(alpha_true, eta_true, "s", color="blue", markersize=10, zorder=5,
         markeredgecolor="white", markeredgewidth=1.5,
         label=f"Identified solution")

# Show the narrow band of remaining solutions
alpha_band = alpha_grid_c[(alpha_grid_c >= alpha_lo) & (alpha_grid_c <= alpha_hi)]
eta_band = -np.log(p_obs * alpha_band)
band_valid = (eta_band >= ETA_MIN) & (eta_band <= ETA_MAX)
ax3.plot(alpha_band[band_valid], eta_band[band_valid], color="blue", linewidth=4,
         alpha=0.5, label="Constrained range")

ax3.set_xlabel(r"$\alpha$ (species richness)", fontsize=11)
ax3.set_ylabel(r"$\eta$ (turnover)", fontsize=11)
ax3.set_title("C.  Within-site pairs break degeneracy", fontsize=11,
              fontweight="bold")
ax3.set_xlim(1, ALPHA_MAX)
ax3.set_ylim(ETA_MIN, ETA_MAX)

# Annotations
ax3.annotate(
    "Between-site pair alone:\n∞ solutions on red curve",
    xy=(300, 2.5), fontsize=8, color="red",
    bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8),
)

ax3.annotate(
    r"Within-site $p_{i,i}=1/\alpha_i$" "\npins richness →\nidentifies turnover",
    xy=(alpha_true + 10, eta_true + 0.7), fontsize=8, color="blue",
    bbox=dict(boxstyle="round,pad=0.3", fc="lavender", ec="blue", alpha=0.8),
)

ax3.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
ax3.grid(True, alpha=0.3)

# ── Layout & save ────────────────────────────────────────────────────────
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close(fig)

# ── Print summary statistics ─────────────────────────────────────────────
print("\n=== Degrees of freedom per observation ===")
print("  Between-site pair (i≠j): 3 unknowns (α_i, α_j, η_ij), 1 equation → 2 DOF")
print("  Within-site pair  (i=i): 1 unknown  (α_i),              1 equation → 0 DOF")
print()
print("  Without within-site pairs:")
print("    N between-site pairs connecting M sites →")
print("    M unknown α's + N unknown η's, but only N equations")
print("    → (M + N) − N = M free parameters (all α's unidentified)")
print()
print("  With within-site pairs:")
print("    Each site with within-site data → α_i pinned directly")
print("    Then between-site pairs identify η_ij given known α's")
print("    → Full identification when every site has within-site data")

# Extra: quantify the tradeoff magnitude
print("\n=== Example: how much can α and η compensate? ===")
for p_ex in [0.001, 0.005, 0.01]:
    print(f"\n  p = {p_ex}:")
    for alpha in [5, 20, 50, 100, 200, 500]:
        eta = -np.log(p_ex * alpha)
        if ETA_MIN <= eta <= ETA_MAX:
            S = np.exp(-eta)
            print(f"    α={alpha:>4d}  →  η={eta:.2f}  (Sørensen={S:.4f}, "
                  f"i.e. {S*100:.1f}% species shared)")
