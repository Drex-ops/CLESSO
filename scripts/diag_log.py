"""Analyze training log and print summary."""
import pandas as pd
import numpy as np

log = pd.read_csv("src/clesso_nn/output/VAS_hexBalance_nn/training_progress_cyclic.log")
log.columns = log.columns.str.strip()

alpha = log[log["phase"] == "alpha"]
beta  = log[log["phase"] == "beta"]

print("=" * 70)
print("TRAINING LOG SUMMARY")
print("=" * 70)

# Per-cycle end-of-phase summaries
print("\n-- Alpha phase (end-of-phase per cycle) --")
print(f"{'Cyc':>3} {'val_loss':>9} {'alpha_mean':>11} {'alpha_std':>10} {'alpha_max':>10} {'val_bce':>8}")
for c in sorted(alpha["cycle"].unique()):
    sub = alpha[alpha["cycle"] == c]
    r = sub.iloc[-1]
    print(f"{c:3d} {r['val_loss']:9.4f} {r['alpha_mean']:11.1f} {r['alpha_std']:10.1f} {r['alpha_max']:10.1f} {r['val_bce']:8.4f}")

print("\n-- Beta phase (end-of-phase per cycle) --")
print(f"{'Cyc':>3} {'val_loss':>9} {'auc_s123':>9} {'auc_all':>8} {'eta_mean':>9} {'eta_std':>8} {'eta_max':>8} {'grad_norm':>10} {'ac_loss':>8}")
for c in sorted(beta["cycle"].unique()):
    sub = beta[beta["cycle"] == c]
    r = sub.iloc[-1]
    print(f"{c:3d} {r['val_loss']:9.4f} {r['val_auc_s123']:9.4f} {r['val_auc']:8.4f} "
          f"{r['val_eta_mean']:9.4f} {r['val_eta_std']:8.4f} {r['val_eta_max']:8.4f} "
          f"{r['beta_grad_norm']:10.6f} {r['eta_ac_loss']:8.4f}")

print("\n-- Within-cycle beta dynamics (first vs last epoch) --")
print(f"{'Cyc':>3} {'loss_start':>11} {'loss_end':>9} {'d_loss':>9} {'auc_start':>10} {'auc_end':>8} {'d_AUC':>8}")
for c in sorted(beta["cycle"].unique()):
    sub = beta[beta["cycle"] == c]
    r0, r1 = sub.iloc[0], sub.iloc[-1]
    dl = r1["val_loss"] - r0["val_loss"]
    da = r1["val_auc_s123"] - r0["val_auc_s123"]
    print(f"{c:3d} {r0['val_loss']:11.4f} {r1['val_loss']:9.4f} {dl:+9.4f} "
          f"{r0['val_auc_s123']:10.4f} {r1['val_auc_s123']:8.4f} {da:+8.4f}")

# Cross-cycle best
print(f"\n-- Best values across all cycles --")
print(f"Best beta val_loss:  {beta['val_loss'].min():.4f} (epoch {beta.loc[beta['val_loss'].idxmin(), 'global_epoch']:.0f})")
print(f"Best AUC s1-3:       {beta['val_auc_s123'].max():.4f} (epoch {beta.loc[beta['val_auc_s123'].idxmax(), 'global_epoch']:.0f})")
print(f"Best AUC overall:    {beta['val_auc'].max():.4f}")

# Trend analysis
print(f"\n-- Cross-cycle trend (end-of-phase beta) --")
ends = beta.groupby("cycle").last()
for col, label in [("val_loss", "Beta val_loss"), ("val_auc_s123", "AUC s1-3"),
                    ("val_eta_mean", "Eta mean"), ("beta_grad_norm", "Grad norm")]:
    vals = ends[col].values
    if len(vals) >= 3:
        slope = np.polyfit(range(len(vals)), vals, 1)[0]
        print(f"  {label:20s}: {vals[0]:.4f} -> {vals[-1]:.4f}  slope={slope:+.6f}/cycle")

# Alpha inflation check
print(f"\n-- Alpha inflation check --")
alpha_ends = alpha.groupby("cycle").last()
print(f"  alpha_std:  {alpha_ends['alpha_std'].iloc[0]:.1f} -> {alpha_ends['alpha_std'].iloc[-1]:.1f}")
print(f"  alpha_max:  {alpha_ends['alpha_max'].iloc[0]:.1f} -> {alpha_ends['alpha_max'].iloc[-1]:.1f}")
print(f"  alpha_mean: {alpha_ends['alpha_mean'].iloc[0]:.1f} -> {alpha_ends['alpha_mean'].iloc[-1]:.1f}")
cv0 = alpha_ends['alpha_std'].iloc[0] / alpha_ends['alpha_mean'].iloc[0]
cv1 = alpha_ends['alpha_std'].iloc[-1] / alpha_ends['alpha_mean'].iloc[-1]
print(f"  CV (std/mean): {cv0:.2f} -> {cv1:.2f}")
