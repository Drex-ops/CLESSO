"""
train.py -- Training loop for CLESSO NN.

Handles:
  - Training with Adam + learning rate scheduling
  - Validation and early stopping
  - Per-epoch progress logging to file (tail-able)
  - Model checkpointing
  - Two-stage training (alpha-only → beta-only) for gradient stability
  - Cyclic block-coordinate descent (alpha↔beta alternating cycles)

Importance Weighting (class-balanced hex-IPW scheme)
=====================================================
Every per-pair loss term is weighted by w_final to correct for sampling biases.

1. **Hex-IPW weight** (``w_ipw = batch["weight"]``):
   Comes from the R sampler (``clesso_sampler_optimised.R``).  The hex-
   balanced sampler gives equal quotas to each hex, so dense hexes are
   under-sampled relative to their population contribution.  The IPW
   weight = hex_capacity / mean(hex_capacity), normalised so mean ~1
   per stratum (pair_type × y).  This corrects geographic imbalance
   while keeping the weight scale stable.

2. **Alpha-harmonic-mean correction** (``w_alpha = 1/ā_h`` for matches,
   ``1`` for mismatches):
   Within a given cell, pairs from high-richness sites are over-
   represented because the probability of drawing a *match* pair is
   proportional to ``m_s(m_s-1)/α²``.  Dividing by the harmonic mean
   of the two site alphas corrects this conditional bias.

3. **Class-balanced normalisation**:
   After combining factors 1+2, the y=0 (match) and y=1 (mismatch)
   weight groups are normalised separately so each contributes equally
   to the BCE loss.  Without this, the 1/alpha correction (~150×
   down-weight on matches) would make the model learn almost entirely
   from mismatches.  Within each class, relative weights still reflect
   geographic and richness corrections.

4. **Hard-pair mining weight** (``w_mining = batch["mining_iw"]``, optional):
   When hard-pair mining is active, the miner re-samples the training set
   to over-represent hard examples.  ``mining_iw`` is the inverse of the
   over-sampling ratio, preserving unbiasedness.

``compute_loss()`` in model.py applies factors 1-3.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .config import CLESSONNConfig
from .dataset import SiteData, make_dataloaders, HardPairMiner
from .model import CLESSONet


# --------------------------------------------------------------------------
# Platform helpers
# --------------------------------------------------------------------------

def _monitor_hint(path: Path) -> str:
    """Return a platform-appropriate command to tail a log file."""
    if sys.platform == "win32":
        return f"Get-Content -Wait -Path {path}"
    return f"tail -f {path}"


# --------------------------------------------------------------------------
# Weight stabilisation helpers
# --------------------------------------------------------------------------

def _stabilise_weights(
    w: torch.Tensor,
    clamp_factor: float = 0.0,
    log_normalise: bool = False,
) -> tuple[torch.Tensor, float]:
    """Optional stabilisation of combined importance weights.

    Args:
        w: raw combined importance weights (pop × alpha × mining), shape (B,)
        clamp_factor: if > 0, clamp to median(w) × clamp_factor to limit
            the influence of outlier pairs.  Typical values: 5–20.
        log_normalise: if True, project weights into log-space and softmax
            back, which compresses heavy tails.  The resulting weights sum
            to 1, so loss = sum(w * bce) (no division by sum(w)).

    Returns:
        (w_stable, ess) where ess = (sum w)^2 / sum(w^2)  ∈ [1, B].
        ESS near B means weights are roughly uniform; ESS << B means a
        few pairs dominate the batch.
    """
    if clamp_factor > 0:
        cap = w.median() * clamp_factor
        w = w.clamp(max=cap.item())

    if log_normalise:
        # Softmax in log-space: preserves ordering, compresses magnitude
        log_w = torch.log(w + 1e-12)
        w = torch.softmax(log_w, dim=0) * w.numel()  # scale so mean ≈ 1

    # Effective sample size: (Σw)² / Σ(w²)
    sum_w = w.sum()
    ess = (sum_w ** 2 / (w ** 2).sum()).item()

    return w, ess


# --------------------------------------------------------------------------
# Training state
# --------------------------------------------------------------------------

class TrainingState:
    """Mutable container for tracking training progress."""

    def __init__(self):
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: list[dict] = []


# --------------------------------------------------------------------------
# Single epoch
# --------------------------------------------------------------------------

def train_one_epoch(
    model: CLESSONet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    site_data: SiteData,
    device: str,
    log_every: int = 10,
    weight_clamp_factor: float = 0.0,
    weight_log_normalise: bool = False,
) -> dict:
    """Train for one epoch. Returns dict of mean metrics."""
    model.train()
    Z = site_data.Z.to(device)
    S_obs = site_data.S_obs.to(device)
    W = site_data.W.to(device) if site_data.W is not None else None
    anchor_richness = site_data.anchor_richness.to(device) if site_data.anchor_richness.any() else None

    total_loss = 0.0
    total_bce = 0.0
    total_lb = 0.0
    total_anchor = 0.0
    total_alpha_reg = 0.0
    total_eta_smooth = 0.0
    total_eta_ac = 0.0
    total_effort = 0.0
    total_richness_anchor = 0.0
    total_beta_grad = 0.0
    total_alpha_grad = 0.0
    total_ess = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        z_i = Z[batch["site_i"]]
        z_j = Z[batch["site_j"]]
        w_i = W[batch["site_i"]] if W is not None else None
        w_j = W[batch["site_j"]] if W is not None else None

        result = model.compute_loss(batch, z_i, z_j, S_obs, w_i=w_i, w_j=w_j,
                                    anchor_richness=anchor_richness,
                                    weight_clamp_factor=weight_clamp_factor,
                                    weight_log_normalise=weight_log_normalise)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()

        # Track gradient norms before clipping
        beta_grads = [p.grad.norm().item() for p in model.beta_net.parameters()
                      if p.grad is not None]
        alpha_grads = [p.grad.norm().item() for p in model.alpha_net.parameters()
                       if p.grad is not None]
        total_beta_grad += sum(beta_grads) / max(len(beta_grads), 1)
        total_alpha_grad += sum(alpha_grads) / max(len(alpha_grads), 1)

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        total_bce += result["bce_loss"].item()
        total_lb += result["lb_penalty"].item()
        total_anchor += result["anchor_penalty"].item()
        total_alpha_reg += result["alpha_reg_loss"].item()
        total_eta_smooth += result["eta_smooth_loss"].item()
        total_eta_ac += result["eta_ac_loss"].item()
        total_effort += result["effort_loss"].item()
        total_richness_anchor += result["richness_anchor_loss"].item()
        total_ess += result["weight_ess"]
        n_batches += 1

        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"    batch {i+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  "
                  f"bce={result['bce_loss'].item():.4f}  "
                  f"lb={result['lb_penalty'].item():.4f}  "
                  f"anc={result['anchor_penalty'].item():.4f}  "
                  f"areg={result['alpha_reg_loss'].item():.4f}")

    return {
        "loss": total_loss / n_batches,
        "bce_loss": total_bce / n_batches,
        "lb_penalty": total_lb / n_batches,
        "anchor_penalty": total_anchor / n_batches,
        "alpha_reg_loss": total_alpha_reg / n_batches,
        "eta_smooth_loss": total_eta_smooth / n_batches,
        "eta_ac_loss": total_eta_ac / n_batches,
        "effort_loss": total_effort / n_batches,
        "richness_anchor_loss": total_richness_anchor / n_batches,
        "beta_grad_norm": total_beta_grad / n_batches,
        "alpha_grad_norm": total_alpha_grad / n_batches,
        "weight_ess": total_ess / n_batches,
    }


@torch.no_grad()
def validate(
    model: CLESSONet,
    loader: DataLoader,
    site_data: SiteData,
    device: str,
    weight_clamp_factor: float = 0.0,
    weight_log_normalise: bool = False,
) -> dict:
    """Evaluate on validation set. Returns dict of mean metrics."""
    model.eval()
    Z = site_data.Z.to(device)
    S_obs = site_data.S_obs.to(device)
    W = site_data.W.to(device) if site_data.W is not None else None
    anchor_richness = site_data.anchor_richness.to(device) if site_data.anchor_richness.any() else None

    total_loss = 0.0
    total_bce = 0.0
    total_lb = 0.0
    total_anchor = 0.0
    total_alpha_reg = 0.0
    total_eta_smooth = 0.0
    total_eta_ac = 0.0
    total_effort = 0.0
    total_richness_anchor = 0.0
    total_ess = 0.0
    n_batches = 0

    # Also track prediction stats
    all_alpha_i = []
    all_y = []
    all_p = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z_i = Z[batch["site_i"]]
        z_j = Z[batch["site_j"]]
        w_i = W[batch["site_i"]] if W is not None else None
        w_j = W[batch["site_j"]] if W is not None else None

        result = model.compute_loss(batch, z_i, z_j, S_obs, w_i=w_i, w_j=w_j,
                                    anchor_richness=anchor_richness,
                                    weight_clamp_factor=weight_clamp_factor,
                                    weight_log_normalise=weight_log_normalise)

        total_loss += result["loss"].item()
        total_bce += result["bce_loss"].item()
        total_lb += result["lb_penalty"].item()
        total_anchor += result["anchor_penalty"].item()
        total_alpha_reg += result["alpha_reg_loss"].item()
        total_eta_smooth += result["eta_smooth_loss"].item()
        total_eta_ac += result["eta_ac_loss"].item()
        total_effort += result["effort_loss"].item()
        total_richness_anchor += result["richness_anchor_loss"].item()
        total_ess += result["weight_ess"]
        n_batches += 1

        all_alpha_i.append(result["alpha_i"].cpu())
        all_y.append(batch["y"].cpu())
        all_p.append(result["p_match"].cpu())

    all_alpha_i = torch.cat(all_alpha_i)
    all_y = torch.cat(all_y)
    all_p = torch.cat(all_p)

    # Classification accuracy (threshold 0.5)
    pred_y = (all_p < 0.5).float()  # p_match < 0.5 → predict mismatch (y=1)
    accuracy = (pred_y == all_y).float().mean().item()

    return {
        "loss": total_loss / n_batches,
        "bce_loss": total_bce / n_batches,
        "lb_penalty": total_lb / n_batches,
        "anchor_penalty": total_anchor / n_batches,
        "alpha_reg_loss": total_alpha_reg / n_batches,
        "eta_smooth_loss": total_eta_smooth / n_batches,
        "eta_ac_loss": total_eta_ac / n_batches,
        "effort_loss": total_effort / n_batches,
        "richness_anchor_loss": total_richness_anchor / n_batches,
        "accuracy": accuracy,
        "alpha_mean": all_alpha_i.mean().item(),
        "alpha_std": all_alpha_i.std().item(),
        "alpha_min": all_alpha_i.min().item(),
        "alpha_max": all_alpha_i.max().item(),
        "weight_ess": total_ess / n_batches,
    }


# --------------------------------------------------------------------------
# Full training loop
# --------------------------------------------------------------------------

def train(
    model: CLESSONet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    site_data: SiteData,
    config: CLESSONNConfig,
) -> TrainingState:
    """Full training loop with early stopping and logging.

    Returns TrainingState with history and best model info.
    """
    device = config.resolve_device()
    print(f"Training on device: {device}")
    model = model.to(device)

    # ---- Separate parameter groups ----
    # Beta weight_raw params get NO weight decay (L2 decay pushes them
    # toward 0 → softplus(0) = 0.693 uniformly, causing dimensional collapse).
    # Beta also gets a higher LR via beta_lr_mult to compete with alpha gradients.
    beta_lr = config.learning_rate * config.beta_lr_mult

    alpha_params = list(model.alpha_net.parameters())
    beta_weight_raw = [p for n, p in model.beta_net.named_parameters()
                       if "weight_raw" in n]
    beta_other = [p for n, p in model.beta_net.named_parameters()
                  if "weight_raw" not in n]

    param_groups = [
        {"params": alpha_params,
         "lr": config.learning_rate,
         "weight_decay": config.weight_decay},
        {"params": beta_weight_raw,
         "lr": beta_lr,
         "weight_decay": 0.0},            # CRITICAL: no weight decay on weight_raw
        {"params": beta_other,
         "lr": beta_lr,
         "weight_decay": config.weight_decay},
    ]
    if model.effort_net is not None:
        param_groups.append({
            "params": list(model.effort_net.parameters()),
            "lr": config.resolved_effort_lr,
            "weight_decay": config.resolved_effort_wd,
        })
        print(f"  EffortNet LR: {config.resolved_effort_lr:.2e}  "
              f"WD: {config.resolved_effort_wd:.2e}")
    optimizer = Adam(param_groups)
    # Cosine annealing with warm restarts:
    # T_0 = 30 epochs per cycle, T_mult = 2 (double cycle length each restart)
    # Gives cycles of 30, 60, 120, ... epochs — avoids premature LR death
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6,
    )

    # Gradient amplification for beta parameters.
    # The chain rule through exp(-η) attenuates beta gradients by ≈p,
    # making them ~1000× smaller than alpha gradients. We amplify them
    # via gradient hooks so the optimizer sees comparable updates.
    beta_grad_scale = config.beta_grad_scale
    if beta_grad_scale != 1.0:
        for param in model.beta_net.parameters():
            param.register_hook(lambda grad, s=beta_grad_scale: grad * s)
        print(f"Beta gradient amplification: {beta_grad_scale}×")

    state = TrainingState()
    checkpoint_path = config.output_dir / "best_model.pt"

    # Set up progress log file
    log_path = config.output_dir / "training_progress.log"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "train_loss", "train_bce", "train_lb", "train_anchor", "train_areg",
        "train_effort", "train_richness_anchor",
        "val_loss", "val_bce", "val_lb", "val_anchor", "val_areg", "val_effort",
        "val_richness_anchor",
        "val_accuracy",
        "alpha_mean", "alpha_std", "alpha_min", "alpha_max",
        "lr", "beta_grad_norm", "alpha_grad_norm",
        "weight_ess",
        "elapsed_sec", "timestamp",
    ])
    log_writer.writeheader()
    log_file.flush()
    print(f"Progress log: {log_path}")
    print(f"Monitor with:  {_monitor_hint(log_path)}\n")

    t_start = time.time()

    try:
        for epoch in range(1, config.max_epochs + 1):
            state.epoch = epoch
            t_epoch = time.time()

            print(f"=== Epoch {epoch}/{config.max_epochs} ===")

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, site_data,
                device, log_every=config.log_every,
                weight_clamp_factor=config.weight_clamp_factor,
                weight_log_normalise=config.weight_log_normalise,
            )

            # Validate
            val_metrics = validate(model, val_loader, site_data, device,
                                   weight_clamp_factor=config.weight_clamp_factor,
                                   weight_log_normalise=config.weight_log_normalise)

            # Learning rate scheduling (cosine annealing steps per epoch)
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t_start
            epoch_time = time.time() - t_epoch

            # Print summary
            print(
                f"  train: loss={train_metrics['loss']:.4f} "
                f"bce={train_metrics['bce_loss']:.4f} "
                f"lb={train_metrics['lb_penalty']:.4f} "
                f"areg={train_metrics['alpha_reg_loss']:.4f} "
                f"∇α={train_metrics['alpha_grad_norm']:.4f} "
                f"∇β={train_metrics['beta_grad_norm']:.4f}"
            )
            print(
                f"  val:   loss={val_metrics['loss']:.4f} "
                f"bce={val_metrics['bce_loss']:.4f} "
                f"acc={val_metrics['accuracy']:.3f} "
                f"α=[{val_metrics['alpha_min']:.0f}, {val_metrics['alpha_mean']:.0f}, "
                f"{val_metrics['alpha_max']:.0f}]"
            )
            print(f"  lr={current_lr:.2e}  epoch_time={epoch_time:.0f}s  total={elapsed:.0f}s")

            # Log to file
            log_writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_bce": f"{train_metrics['bce_loss']:.6f}",
                "train_lb": f"{train_metrics['lb_penalty']:.6f}",
                "train_anchor": f"{train_metrics['anchor_penalty']:.6f}",
                "train_areg": f"{train_metrics['alpha_reg_loss']:.6f}",
                "train_effort": f"{train_metrics['effort_loss']:.6f}",
                "train_richness_anchor": f"{train_metrics['richness_anchor_loss']:.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_bce": f"{val_metrics['bce_loss']:.6f}",
                "val_lb": f"{val_metrics['lb_penalty']:.6f}",
                "val_anchor": f"{val_metrics['anchor_penalty']:.6f}",
                "val_areg": f"{val_metrics['alpha_reg_loss']:.6f}",
                "val_effort": f"{val_metrics['effort_loss']:.6f}",
                "val_richness_anchor": f"{val_metrics['richness_anchor_loss']:.6f}",
                "val_accuracy": f"{val_metrics['accuracy']:.4f}",
                "alpha_mean": f"{val_metrics['alpha_mean']:.2f}",
                "alpha_std": f"{val_metrics['alpha_std']:.2f}",
                "alpha_min": f"{val_metrics['alpha_min']:.2f}",
                "alpha_max": f"{val_metrics['alpha_max']:.2f}",
                "lr": f"{current_lr:.2e}",
                "beta_grad_norm": f"{train_metrics['beta_grad_norm']:.6f}",
                "alpha_grad_norm": f"{train_metrics['alpha_grad_norm']:.6f}",
                "weight_ess": f"{train_metrics['weight_ess']:.1f}",
                "elapsed_sec": f"{elapsed:.0f}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            log_file.flush()

            # Track history
            state.history.append({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": current_lr,
                "beta_grad_norm": train_metrics["beta_grad_norm"],
                "alpha_grad_norm": train_metrics["alpha_grad_norm"],
                "elapsed": elapsed,
            })

            # Early stopping
            if val_metrics["loss"] < state.best_val_loss:
                state.best_val_loss = val_metrics["loss"]
                state.best_epoch = epoch
                state.patience_counter = 0
                # Save checkpoint
                _save_checkpoint(model, optimizer, epoch, val_metrics["loss"],
                                 site_data, config, checkpoint_path)
                print(f"  *** New best model saved (val_loss={state.best_val_loss:.6f}) ***")
            else:
                state.patience_counter += 1
                if state.patience_counter >= config.patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {config.patience} epochs)")
                    break

            print()

    finally:
        log_file.close()

    print(f"\nTraining complete. Best epoch: {state.best_epoch} "
          f"(val_loss={state.best_val_loss:.6f})")
    print(f"Best model saved to: {checkpoint_path}")

    # Restore best model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Restored best model weights.\n")

    return state


# ==========================================================================
# Two-stage training: alpha-only → beta-only
# ==========================================================================
#
# Mirrors the original CLESSO two-stage approach:
#   Stage 1 (clesso_alpha.cpp): within-site pairs only → learn α per site
#   Stage 2 (clesso_beta_fixAlpha.cpp): between-site pairs, α fixed → learn β
#
# Why two-stage?
# In the joint model, α inflates to ~1000 (matching the sampled pair
# frequencies rather than true richness).  This makes dp/dη ≈ 1/α ≈ 0.001,
# starving β of gradient.  Two-stage training:
#   1. Anchors α to realistic values (≈ S_obs, forced by lb_penalty)
#   2. Gives β uncontested, importance-weighted gradients
#
# Importance weighting in Stage 2:
# Match (y=0) gradient w.r.t. η is always +1 (independent of α).
# Mismatch (y=1) gradient w.r.t. η is -p/(1-p) ≈ -1/α_h.
# With 50/50 sampled match/mismatch, matches dominate by factor α.
# Weighting matches by 1/α_h balances the two, giving net gradient ≈ 0
# at η=0 — exactly what we need for learning to begin from the signal.
# ==========================================================================


@torch.no_grad()
def _stage2_validate(
    model: CLESSONet,
    loader: DataLoader,
    site_data: SiteData,
    device: str,
    eps: float = 1e-7,
    weight_clamp_factor: float = 0.0,
    weight_log_normalise: bool = False,
) -> dict:
    """Stage 2: validate beta on between-site pairs."""
    model.eval()
    Z = site_data.Z.to(device)
    W = site_data.W.to(device) if site_data.W is not None else None

    total_loss = 0.0
    all_eta = []
    all_p = []
    all_y = []
    all_stratum = []
    n_batches = 0

    # Retention rate keys for retrospective correction
    _rr_keys = {
        1: ("between_tier1_r0", "between_tier1_r1"),
        2: ("between_tier2_r0", "between_tier2_r1"),
        3: ("between_tier3_r0", "between_tier3_r1"),
    }
    rr = model.retention_rates

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z_i = Z[batch["site_i"]]
        z_j = Z[batch["site_j"]]
        w_i = W[batch["site_i"]] if W is not None else None
        w_j = W[batch["site_j"]] if W is not None else None

        fwd = model.forward(z_i, z_j, batch["env_diff"], batch["is_within"],
                            w_i=w_i, w_j=w_j,
                            env_i=batch.get("env_i"),
                            env_j=batch.get("env_j"),
                            geo_dist=batch.get("geo_dist"))
        p = fwd["p_match"].clamp(eps, 1.0 - eps)
        y = batch["y"]
        eta = fwd["eta"]
        stratum = batch["stratum"]

        # ---- Retrospective Bernoulli correction per stratum ----
        p_corr = p.clone()
        for s_idx, (r0_key, r1_key) in _rr_keys.items():
            mask_s = stratum == s_idx
            if not mask_s.any():
                continue
            r0 = rr.get(r0_key, 1.0)
            r1 = rr.get(r1_key, 1.0)
            if r0 != 1.0 or r1 != 1.0:
                p_s = p[mask_s]
                p_corr[mask_s] = (
                    r0 * p_s / (r0 * p_s + r1 * (1.0 - p_s) + eps)
                ).clamp(eps, 1.0 - eps)

        log_p_corr = torch.log(p_corr)
        log_1m_p_corr = torch.log(1.0 - p_corr)
        bce = -(1.0 - y) * log_p_corr - y * log_1m_p_corr

        # Design weights (proper IPW from sampler)
        imp_w = batch["design_w"].clone()

        # Stabilise and compute ESS
        imp_w, _ess = _stabilise_weights(
            imp_w, weight_clamp_factor, weight_log_normalise)

        loss = (imp_w * bce).sum() / imp_w.sum()

        total_loss += loss.item()
        all_eta.append(eta.cpu())
        all_p.append(p.cpu())
        all_y.append(y.cpu())
        all_stratum.append(stratum.cpu())
        n_batches += 1

    all_eta = torch.cat(all_eta)
    all_p = torch.cat(all_p)
    all_y = torch.cat(all_y)
    all_stratum = torch.cat(all_stratum)

    # AUC for between-site classification
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_y.numpy(), 1 - all_p.numpy())
    except (ValueError, ImportError):
        auc = 0.5

    # Per-stratum AUC: strata 1-3 only (excludes match-boost stratum 4)
    auc_s123 = 0.5
    try:
        from sklearn.metrics import roc_auc_score as _roc
        mask_s123 = (all_stratum >= 1) & (all_stratum <= 3)
        if mask_s123.any():
            y_s123 = all_y[mask_s123].numpy()
            p_s123 = all_p[mask_s123].numpy()
            if len(np.unique(y_s123)) == 2:
                auc_s123 = _roc(y_s123, 1 - p_s123)
    except (ValueError, ImportError):
        pass

    return {
        "loss": total_loss / max(n_batches, 1),
        "eta_mean": all_eta.mean().item(),
        "eta_std": all_eta.std().item(),
        "eta_min": all_eta.min().item(),
        "eta_max": all_eta.max().item(),
        "auc": auc,
        "auc_s123": auc_s123,
    }


def _save_checkpoint(model, optimizer, epoch, val_loss, site_data, config, path):
    """Save model checkpoint with metadata."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": {
            "K_alpha": site_data.K_alpha,
            "K_env": site_data.K_env,
            "K_effort": site_data.K_effort,
            "alpha_hidden": config.alpha_hidden,
            "beta_hidden": config.beta_hidden,
            "alpha_dropout": config.alpha_dropout,
            "beta_dropout": config.beta_dropout,
            "alpha_activation": config.alpha_activation,
            "alpha_lb_lambda": config.alpha_lower_bound_lambda,
            "alpha_anchor_lambda": config.alpha_anchor_lambda,
            "alpha_anchor_tolerance": config.alpha_anchor_tolerance,
            "alpha_regression_lambda": config.alpha_regression_lambda,
            "beta_type": config.beta_type,
            "beta_n_knots": config.beta_n_knots,
            "beta_no_intercept": config.beta_no_intercept,
            "transform_n_knots": config.transform_n_knots,
            "transform_g_knots": config.transform_g_knots,
            "training_mode": config.training_mode,
            "geo_penalty_alpha": getattr(config, "geo_penalty_alpha", 0.0),
            "geo_penalty_beta": getattr(config, "geo_penalty_beta", 0.0),
            "effort_hidden": list(config.effort_hidden),
            "effort_dropout": config.effort_dropout,
            "effort_mode": config.effort_mode,
        },
        "site_data_stats": {
            "z_mean": site_data.z_mean.tolist(),
            "z_std": site_data.z_std.tolist(),
            "e_mean": site_data.e_mean.tolist() if hasattr(site_data, "e_mean") and site_data.e_mean is not None else None,
            "e_std": site_data.e_std.tolist() if hasattr(site_data, "e_std") and site_data.e_std is not None else None,
            "w_mean": site_data.w_mean.tolist() if site_data.w_mean is not None else None,
            "w_std": site_data.w_std.tolist() if site_data.w_std is not None else None,
            "geo_mean": site_data.geo_mean.tolist() if hasattr(site_data, "geo_mean") and site_data.geo_mean is not None else None,
            "geo_std": site_data.geo_std.tolist() if hasattr(site_data, "geo_std") and site_data.geo_std is not None else None,
            "geo_dist_scale": site_data.geo_dist_scale,
            "include_geo_dist_in_beta": site_data.include_geo_dist_in_beta,
            "fourier_n_frequencies": site_data.fourier_n_frequencies,
            "fourier_max_wavelength": site_data.fourier_max_wavelength,
            "alpha_cov_names": site_data.alpha_cov_names,
            "env_cov_names": site_data.env_cov_names,
            "effort_cov_names": site_data.effort_cov_names,
        },
    }, path)


def train_two_stage(
    model: CLESSONet,
    pairs,  # pd.DataFrame — all pairs, will be split internally
    site_data: SiteData,
    config: CLESSONNConfig,
) -> TrainingState:
    """Two-stage training: alpha-only on within-site, then beta-only on between-site.

    Stage 1: Freeze beta, train alpha on within-site pairs with BCE + lb_penalty.
             Alpha converges to ≈ S_obs per site (realistic richness).

    Stage 2: Freeze alpha, train beta on between-site pairs with importance-
             weighted BCE.  Match pairs are weighted by 1/α_h to balance the
             gradient (which otherwise favours matches by factor α).
             Beta learns non-trivial turnover response curves.

    Returns TrainingState (combined) with best model having both stages' weights.
    """
    import pandas as pd

    device = config.resolve_device()
    print(f"Training on device: {device}")
    model = model.to(device)

    # Split pairs by type
    within_pairs = pairs[pairs["is_within"] == 1].reset_index(drop=True)
    between_pairs = pairs[pairs["is_within"] == 0].reset_index(drop=True)
    print(f"  Within-site pairs:  {len(within_pairs):,}")
    print(f"  Between-site pairs: {len(between_pairs):,}")

    state = TrainingState()

    # ==================================================================
    # STAGE 1: Alpha-only on within-site pairs
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STAGE 1: Training alpha on within-site pairs")
    print("=" * 60)

    # Freeze beta, unfreeze alpha (+ effort_net)
    for p in model.beta_net.parameters():
        p.requires_grad = False
    for p in model.alpha_net.parameters():
        p.requires_grad = True
    if model.effort_net is not None:
        for p in model.effort_net.parameters():
            p.requires_grad = True

    # Dataloaders (within-site only, population weights from sampler)
    s1_train, s1_val, _, _ = make_dataloaders(
        within_pairs, site_data,
        val_fraction=config.val_fraction,
        batch_size=config.batch_size,
        seed=config.seed,
        use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )

    s1_alpha_params = list(model.alpha_net.parameters())
    s1_param_groups = [
        {"params": s1_alpha_params,
         "lr": config.learning_rate,
         "weight_decay": config.weight_decay},
    ]
    if model.effort_net is not None:
        s1_param_groups.append({
            "params": list(model.effort_net.parameters()),
            "lr": config.resolved_effort_lr,
            "weight_decay": config.resolved_effort_wd,
        })
        print(f"  EffortNet LR: {config.resolved_effort_lr:.2e}  "
              f"WD: {config.resolved_effort_wd:.2e}")
    s1_optimizer = Adam(s1_param_groups)
    s1_scheduler = CosineAnnealingWarmRestarts(
        s1_optimizer, T_0=30, T_mult=2, eta_min=1e-6,
    )

    s1_checkpoint = config.output_dir / "best_model_stage1.pt"
    s1_log_path = config.output_dir / "training_progress_stage1.log"
    s1_log_file = open(s1_log_path, "w", newline="")
    s1_log_writer = csv.DictWriter(s1_log_file, fieldnames=[
        "epoch", "train_loss", "train_bce", "train_lb", "train_anchor", "train_effort",
        "val_loss", "val_bce", "val_effort", "val_accuracy",
        "alpha_mean", "alpha_std", "alpha_min", "alpha_max",
        "lr", "alpha_grad_norm", "weight_ess",
        "elapsed_sec", "timestamp",
    ])
    s1_log_writer.writeheader()
    s1_log_file.flush()
    print(f"  Stage 1 log: {s1_log_path}")
    print(f"  Monitor with:  {_monitor_hint(s1_log_path)}\n")

    s1_best_loss = float("inf")
    s1_patience_ctr = 0
    t_start = time.time()

    try:
        for epoch in range(1, config.stage1_max_epochs + 1):
            print(f"  S1 Epoch {epoch}/{config.stage1_max_epochs}")

            train_m = train_one_epoch(
                model, s1_train, s1_optimizer, site_data,
                device, log_every=config.log_every,
                weight_clamp_factor=config.weight_clamp_factor,
                weight_log_normalise=config.weight_log_normalise,
            )
            val_m = validate(model, s1_val, site_data, device,
                             weight_clamp_factor=config.weight_clamp_factor,
                             weight_log_normalise=config.weight_log_normalise)
            s1_scheduler.step(epoch)
            lr = s1_optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start

            print(
                f"    train: loss={train_m['loss']:.4f} bce={train_m['bce_loss']:.4f} "
                f"lb={train_m['lb_penalty']:.4f}  "
                f"val: loss={val_m['loss']:.4f} acc={val_m['accuracy']:.3f} "
                f"α=[{val_m['alpha_min']:.0f}, {val_m['alpha_mean']:.0f}, {val_m['alpha_max']:.0f}]"
            )

            s1_log_writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_m['loss']:.6f}",
                "train_bce": f"{train_m['bce_loss']:.6f}",
                "train_lb": f"{train_m['lb_penalty']:.6f}",
                "train_anchor": f"{train_m['anchor_penalty']:.6f}",
                "train_effort": f"{train_m['effort_loss']:.6f}",
                "val_loss": f"{val_m['loss']:.6f}",
                "val_bce": f"{val_m['bce_loss']:.6f}",
                "val_effort": f"{val_m['effort_loss']:.6f}",
                "val_accuracy": f"{val_m['accuracy']:.4f}",
                "alpha_mean": f"{val_m['alpha_mean']:.2f}",
                "alpha_std": f"{val_m['alpha_std']:.2f}",
                "alpha_min": f"{val_m['alpha_min']:.2f}",
                "alpha_max": f"{val_m['alpha_max']:.2f}",
                "lr": f"{lr:.2e}",
                "alpha_grad_norm": f"{train_m['alpha_grad_norm']:.6f}",
                "weight_ess": f"{train_m['weight_ess']:.1f}",
                "elapsed_sec": f"{elapsed:.0f}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            s1_log_file.flush()

            if val_m["loss"] < s1_best_loss:
                s1_best_loss = val_m["loss"]
                state.best_epoch = epoch
                s1_patience_ctr = 0
                _save_checkpoint(model, s1_optimizer, epoch, val_m["loss"],
                                 site_data, config, s1_checkpoint)
                print(f"    *** Stage 1 best (val_loss={s1_best_loss:.6f}) ***")
            else:
                s1_patience_ctr += 1
                if s1_patience_ctr >= config.stage1_patience:
                    print(f"    Stage 1 early stopping (no improvement for "
                          f"{config.stage1_patience} epochs)")
                    break
            print()
    finally:
        s1_log_file.close()

    # Restore best Stage 1 model
    ckpt = torch.load(s1_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    s1_time = time.time() - t_start
    print(f"\n  Stage 1 complete: best epoch {state.best_epoch}, "
          f"val_loss={s1_best_loss:.6f}, time={s1_time:.0f}s")

    # Report alpha distribution after Stage 1
    with torch.no_grad():
        model.eval()
        Z = site_data.Z.to(device)
        W = site_data.W.to(device) if site_data.W is not None else None
        all_alpha = model._compute_alpha(Z, W).cpu()
        print(f"  Alpha after Stage 1: mean={all_alpha.mean():.1f}, "
              f"std={all_alpha.std():.1f}, "
              f"min={all_alpha.min():.1f}, max={all_alpha.max():.1f}")

    # ==================================================================
    # STAGE 2: Beta-only on between-site pairs (alpha frozen)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STAGE 2: Training beta on between-site pairs (alpha frozen)")
    print("=" * 60)

    # Freeze alpha (+ effort_net), unfreeze beta
    for p in model.alpha_net.parameters():
        p.requires_grad = False
    if model.effort_net is not None:
        for p in model.effort_net.parameters():
            p.requires_grad = False
    for p in model.beta_net.parameters():
        p.requires_grad = True

    # Dataloaders (between-site only, population weights from sampler)
    s2_train, s2_val, s2_train_ds, _ = make_dataloaders(
        between_pairs, site_data,
        val_fraction=config.val_fraction,
        batch_size=config.batch_size,
        seed=config.seed,
        use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )

    # Hard-pair mining
    use_hard_mining = config.hard_mining_lambda > 0.0
    hard_miner = None
    if use_hard_mining:
        hard_miner = HardPairMiner(s2_train_ds, config)
        print(f"  Hard-pair mining: \u03bb_hm={config.hard_mining_lambda}, "
              f"bins={config.hard_mining_n_bins}, "
              f"bin_weights={config.hard_mining_bin_weights}, "
              f"a_max={config.hard_mining_a_max}")

    beta_lr = config.learning_rate * config.beta_lr_mult
    beta_weight_raw = [p for n, p in model.beta_net.named_parameters()
                       if "weight_raw" in n]
    beta_other = [p for n, p in model.beta_net.named_parameters()
                  if "weight_raw" not in n]

    s2_optimizer = Adam([
        {"params": beta_weight_raw,
         "lr": beta_lr, "weight_decay": 0.0},
        {"params": beta_other,
         "lr": beta_lr, "weight_decay": config.weight_decay},
    ])
    s2_scheduler = CosineAnnealingWarmRestarts(
        s2_optimizer, T_0=30, T_mult=2, eta_min=1e-6,
    )

    # Beta gradient amplification (moderate — importance weights handle the balance)
    s2_grad_scale = config.stage2_beta_grad_scale
    if s2_grad_scale != 1.0:
        for param in model.beta_net.parameters():
            param.register_hook(lambda g, s=s2_grad_scale: g * s)
        print(f"  Stage 2 gradient amplification: {s2_grad_scale}×")

    s2_checkpoint = config.output_dir / "best_model.pt"  # final model
    s2_log_path = config.output_dir / "training_progress_stage2.log"
    s2_log_file = open(s2_log_path, "w", newline="")
    s2_log_writer = csv.DictWriter(s2_log_file, fieldnames=[
        "epoch", "train_loss", "train_eta_mean",
        "val_loss", "val_eta_mean", "val_eta_std", "val_eta_min", "val_eta_max",
        "val_auc", "lr", "beta_grad_norm", "eta_ac_loss", "weight_ess",
        "elapsed_sec", "timestamp",
    ])
    s2_log_writer.writeheader()
    s2_log_file.flush()
    print(f"  Stage 2 log: {s2_log_path}")
    print(f"  Monitor with:  {_monitor_hint(s2_log_path)}\n")

    s2_best_loss = float("inf")
    s2_patience_ctr = 0
    t_start2 = time.time()

    try:
        for epoch in range(1, config.stage2_max_epochs + 1):
            print(f"  S2 Epoch {epoch}/{config.stage2_max_epochs}")

            # Refresh hard-pair scores periodically
            mining_active = (
                use_hard_mining
                and epoch > config.hard_mining_warmup_cycles  # reuse as warmup epochs
            )
            if mining_active:
                should_refresh = (
                    epoch == config.hard_mining_warmup_cycles + 1
                    or (config.hard_mining_refresh_every > 0
                        and (epoch - 1) % config.hard_mining_refresh_every == 0)
                )
                if should_refresh:
                    hard_miner.refresh_scores(model, site_data, device,
                                              batch_size=config.batch_size * 2)
                    s2_train_active = hard_miner.make_dataloader(
                        batch_size=config.batch_size)
                    if epoch == config.hard_mining_warmup_cycles + 1:
                        print(f"    Hard-pair mining activated (epoch {epoch})")
            else:
                s2_train_active = s2_train

            # ---- Train (importance-weighted) ----
            model.beta_net.train()
            model.alpha_net.eval()
            Z = site_data.Z.to(device)
            W = site_data.W.to(device) if site_data.W is not None else None

            total_loss_t = 0.0
            total_eta_t = 0.0
            total_eta_ac_t = 0.0
            total_bg = 0.0
            total_ess_t = 0.0
            nb = 0
            eps = 1e-7

            # Retention rate keys for retrospective correction
            _rr_keys = {
                1: ("between_tier1_r0", "between_tier1_r1"),
                2: ("between_tier2_r0", "between_tier2_r1"),
                3: ("between_tier3_r0", "between_tier3_r1"),
            }
            rr = model.retention_rates

            for bi, batch in enumerate(s2_train_active):
                batch = {k: v.to(device) for k, v in batch.items()}
                z_i = Z[batch["site_i"]]
                z_j = Z[batch["site_j"]]
                w_i = W[batch["site_i"]] if W is not None else None
                w_j = W[batch["site_j"]] if W is not None else None

                fwd = model.forward(z_i, z_j, batch["env_diff"], batch["is_within"],
                                    w_i=w_i, w_j=w_j,
                                    env_i=batch.get("env_i"),
                                    env_j=batch.get("env_j"),
                                    geo_dist=batch.get("geo_dist"))
                p = fwd["p_match"].clamp(eps, 1.0 - eps)
                y = batch["y"]
                eta = fwd["eta"]
                stratum = batch["stratum"]

                # ---- Retrospective Bernoulli correction per stratum ----
                p_corr = p.clone()
                for s_idx, (r0_key, r1_key) in _rr_keys.items():
                    mask_s = stratum == s_idx
                    if not mask_s.any():
                        continue
                    r0 = rr.get(r0_key, 1.0)
                    r1 = rr.get(r1_key, 1.0)
                    if r0 != 1.0 or r1 != 1.0:
                        p_s = p[mask_s]
                        p_corr[mask_s] = (
                            r0 * p_s / (r0 * p_s + r1 * (1.0 - p_s) + eps)
                        ).clamp(eps, 1.0 - eps)

                log_p_corr = torch.log(p_corr)
                log_1m_p_corr = torch.log(1.0 - p_corr)
                bce = -(1.0 - y) * log_p_corr - y * log_1m_p_corr

                # Design weights (proper IPW from sampler) + optional mining correction
                imp_w = batch["design_w"].clone()
                if mining_active and "mining_iw" in batch:
                    imp_w = imp_w * batch["mining_iw"]

                # Stabilise and compute ESS
                imp_w, batch_ess = _stabilise_weights(
                    imp_w, config.weight_clamp_factor,
                    config.weight_log_normalise)
                total_ess_t += batch_ess

                loss = (imp_w * bce).sum() / imp_w.sum()

                # Anti-collapse regularizer: -λ · mean(log(η + ε))
                eta_ac_loss = torch.tensor(0.0, device=p.device)
                if model.eta_anti_collapse_lambda > 0:
                    eta_ac_loss = (
                        -model.eta_anti_collapse_lambda
                        * torch.log(eta + 1e-4).mean()
                    )
                    loss = loss + eta_ac_loss

                # Geographic parameter L2 penalty (beta only during Stage 2)
                if model.geo_penalty_beta > 0 and model.K_env_env > 0:
                    _, beta_geo_l2 = model.geo_param_penalty(
                        K_alpha_env=model.K_alpha_env,
                        K_env_env=model.K_env_env)
                    loss = loss + model.geo_penalty_beta * beta_geo_l2

                s2_optimizer.zero_grad()
                loss.backward()

                bg = [pp.grad.norm().item() for pp in model.beta_net.parameters()
                      if pp.grad is not None]
                total_bg += sum(bg) / max(len(bg), 1)

                nn.utils.clip_grad_norm_(model.beta_net.parameters(), max_norm=5.0)
                s2_optimizer.step()

                total_loss_t += loss.item()
                total_eta_t += eta.mean().item()
                total_eta_ac_t += eta_ac_loss.item()
                nb += 1

                if config.log_every > 0 and (bi + 1) % config.log_every == 0:
                    print(f"    batch {bi+1}/{len(s2_train)}  "
                          f"loss={loss.item():.4f}  "
                          f"η_mean={eta.mean().item():.4f}  "
                          f"η_max={eta.max().item():.4f}")

            train_m2 = {
                "loss": total_loss_t / max(nb, 1),
                "eta_mean": total_eta_t / max(nb, 1),
                "eta_ac_loss": total_eta_ac_t / max(nb, 1),
                "beta_grad_norm": total_bg / max(nb, 1),
                "weight_ess": total_ess_t / max(nb, 1),
            }

            # ---- Validate ----
            val_m2 = _stage2_validate(model, s2_val, site_data, device,
                                     weight_clamp_factor=config.weight_clamp_factor,
                                     weight_log_normalise=config.weight_log_normalise)
            s2_scheduler.step(epoch)
            lr = s2_optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start2

            print(
                f"    train: loss={train_m2['loss']:.4f} η_mean={train_m2['eta_mean']:.4f} "
                f"∇β={train_m2['beta_grad_norm']:.4f}  "
                f"val: loss={val_m2['loss']:.4f} η=[{val_m2['eta_min']:.2f}, "
                f"{val_m2['eta_mean']:.2f}, {val_m2['eta_max']:.2f}] "
                f"AUC={val_m2['auc']:.3f}"
            )

            s2_log_writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_m2['loss']:.6f}",
                "train_eta_mean": f"{train_m2['eta_mean']:.6f}",
                "val_loss": f"{val_m2['loss']:.6f}",
                "val_eta_mean": f"{val_m2['eta_mean']:.6f}",
                "val_eta_std": f"{val_m2['eta_std']:.6f}",
                "val_eta_min": f"{val_m2['eta_min']:.6f}",
                "val_eta_max": f"{val_m2['eta_max']:.6f}",
                "val_auc": f"{val_m2['auc']:.4f}",
                "lr": f"{lr:.2e}",
                "beta_grad_norm": f"{train_m2['beta_grad_norm']:.6f}",
                "eta_ac_loss": f"{train_m2['eta_ac_loss']:.6f}",
                "weight_ess": f"{train_m2['weight_ess']:.1f}",
                "elapsed_sec": f"{elapsed:.0f}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            s2_log_file.flush()

            state.history.append({
                "stage": 2,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_m2.items()},
                **{f"val_{k}": v for k, v in val_m2.items()},
                "lr": lr,
            })

            if val_m2["loss"] < s2_best_loss:
                s2_best_loss = val_m2["loss"]
                state.best_val_loss = s2_best_loss
                state.best_epoch = epoch
                s2_patience_ctr = 0
                _save_checkpoint(model, s2_optimizer, epoch, val_m2["loss"],
                                 site_data, config, s2_checkpoint)
                print(f"    *** Stage 2 best (val_loss={s2_best_loss:.6f}) ***")
            else:
                s2_patience_ctr += 1
                if s2_patience_ctr >= config.stage2_patience:
                    print(f"    Stage 2 early stopping (no improvement for "
                          f"{config.stage2_patience} epochs)")
                    break
            print()
    finally:
        s2_log_file.close()

    # Restore best Stage 2 model
    ckpt = torch.load(s2_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    s2_time = time.time() - t_start2

    print(f"\n  Stage 2 complete: best epoch {state.best_epoch}, "
          f"val_loss={s2_best_loss:.6f}, time={s2_time:.0f}s")
    print(f"\n  Two-stage training complete.")
    print(f"  Final model saved to: {s2_checkpoint}")

    return state


# ==========================================================================
# Cyclic block-coordinate descent: alpha ↔ beta alternating
# ==========================================================================
#
# Mirrors clesso_iterative.R: alternating block-coordinate optimisation.
#
#   for cycle in 1..max_cycles:
#     Phase A: freeze beta, train alpha on within-site pairs for N_a epochs
#     Phase B: freeze alpha, train beta on between-site pairs for N_b epochs
#     Check convergence: relative change in combined val loss < tol
#
# Key differences from two_stage:
# - Alpha and beta get to see each other's updated parameters repeatedly
# - Alpha can re-adjust after beta has learnt turnover
# - Better suited when alpha/beta interact (e.g. alpha adjusting to
#   compensate for turnover)
# ==========================================================================


def train_cyclic(
    model: CLESSONet,
    pairs,  # pd.DataFrame — all pairs, will be split internally
    site_data: SiteData,
    config: CLESSONNConfig,
) -> TrainingState:
    """Cyclic block-coordinate descent: alternate alpha and beta training.

    Each cycle:
      Phase A: freeze beta, train alpha on within-site pairs (N_a epochs)
      Phase B: freeze alpha, train beta on between-site pairs (N_b epochs)

    Repeats until relative change in best val loss < cycle_tol, or max_cycles.

    Returns TrainingState with overall best model (selected by Stage 2 val loss).
    """
    import pandas as pd

    device = config.resolve_device()
    print(f"Training on device: {device}")
    model = model.to(device)

    # Split pairs by type
    within_pairs = pairs[pairs["is_within"] == 1].reset_index(drop=True)
    between_pairs = pairs[pairs["is_within"] == 0].reset_index(drop=True)
    print(f"  Within-site pairs:  {len(within_pairs):,}")
    print(f"  Between-site pairs: {len(between_pairs):,}")

    # Build dataloaders once (reused across cycles)
    s1_train, s1_val, _, _ = make_dataloaders(
        within_pairs, site_data,
        val_fraction=config.val_fraction, batch_size=config.batch_size,
        seed=config.seed, use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )
    s2_train, s2_val, s2_train_ds, _ = make_dataloaders(
        between_pairs, site_data,
        val_fraction=config.val_fraction, batch_size=config.batch_size,
        seed=config.seed, use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )

    # Hard-pair mining (between-site pairs only)
    use_hard_mining = config.hard_mining_lambda > 0.0
    hard_miner = None
    if use_hard_mining:
        hard_miner = HardPairMiner(s2_train_ds, config)
        print(f"  Hard-pair mining: λ_hm={config.hard_mining_lambda}, "
              f"bins={config.hard_mining_n_bins}, "
              f"bin_weights={config.hard_mining_bin_weights}, "
              f"a_max={config.hard_mining_a_max}, "
              f"warmup={config.hard_mining_warmup_cycles} cycles")

    state = TrainingState()

    # Checkpoint paths
    best_model_path = config.output_dir / "best_model.pt"
    stage1_ckpt_path = config.output_dir / "best_model_stage1.pt"

    # Progress log — single file for all cycles
    log_path = config.output_dir / "training_progress_cyclic.log"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "cycle", "phase", "phase_epoch", "global_epoch",
        "train_loss", "val_loss",
        # Alpha phase columns
        "train_bce", "train_lb", "train_anchor", "train_effort",
        "val_bce", "val_effort", "val_accuracy",
        "alpha_mean", "alpha_std", "alpha_min", "alpha_max",
        "alpha_grad_norm",
        # Beta phase columns
        "train_eta_mean", "val_eta_mean", "val_eta_std",
        "val_eta_min", "val_eta_max", "val_auc", "val_auc_s123",
        "beta_grad_norm", "eta_ac_loss",
        # Common
        "weight_ess",
        "lr", "elapsed_sec", "timestamp",
    ])
    log_writer.writeheader()
    log_file.flush()
    print(f"  Cyclic log: {log_path}")
    print(f"  Monitor with:  {_monitor_hint(log_path)}\n")

    # ---- Set up optimizers (persistent across cycles for momentum state) ----

    # Alpha optimizer (effort_net gets its own param group with higher LR)
    alpha_opt_params = list(model.alpha_net.parameters())
    alpha_param_groups = [
        {"params": alpha_opt_params,
         "lr": config.learning_rate,
         "weight_decay": config.weight_decay},
    ]
    if model.effort_net is not None:
        alpha_param_groups.append({
            "params": list(model.effort_net.parameters()),
            "lr": config.resolved_effort_lr,
            "weight_decay": config.resolved_effort_wd,
        })
        print(f"  EffortNet LR: {config.resolved_effort_lr:.2e}  "
              f"WD: {config.resolved_effort_wd:.2e}")
    alpha_optimizer = Adam(alpha_param_groups)
    # Step-based LR decay: halve every 200 global epochs
    alpha_scheduler = torch.optim.lr_scheduler.StepLR(
        alpha_optimizer, step_size=200, gamma=0.5,
    )

    # Beta optimizer
    beta_lr = config.learning_rate * config.beta_lr_mult
    beta_weight_raw = [p for n, p in model.beta_net.named_parameters()
                       if "weight_raw" in n]
    beta_other = [p for n, p in model.beta_net.named_parameters()
                  if "weight_raw" not in n]
    beta_optimizer = Adam([
        {"params": beta_weight_raw, "lr": beta_lr, "weight_decay": 0.0},
        {"params": beta_other, "lr": beta_lr, "weight_decay": config.weight_decay},
    ])
    beta_scheduler = torch.optim.lr_scheduler.StepLR(
        beta_optimizer, step_size=200, gamma=0.5,
    )

    # Beta gradient amplification
    beta_grad_scale = config.beta_grad_scale
    if beta_grad_scale != 1.0:
        for param in model.beta_net.parameters():
            param.register_hook(lambda g, s=beta_grad_scale: g * s)
        print(f"  Beta gradient amplification: {beta_grad_scale}×")
    else:
        print("  Beta gradient amplification: OFF (beta_grad_scale=1.0)")

    t_start = time.time()
    global_epoch = 0
    prev_best_loss = float("inf")
    overall_best_loss = float("inf")

    # ---- Cycle parameters ----
    max_cycles = config.max_cycles
    n_alpha_epochs = config.cycle_alpha_epochs
    n_beta_epochs = config.cycle_beta_epochs
    tol = config.cycle_tol

    print(f"\n{'='*60}")
    print(f"  Cyclic Block-Coordinate Descent (Damped)")
    print(f"  max_cycles={max_cycles}, "
          f"alpha_epochs/cycle={n_alpha_epochs}, "
          f"beta_epochs/cycle={n_beta_epochs}, "
          f"tol={tol:.1e}")
    beta_patience = config.cycle_beta_patience
    print(f"  Total epoch budget: {max_cycles * (n_alpha_epochs + n_beta_epochs)}")
    print(f"  Beta patience (per phase): {beta_patience} epochs")
    print(f"{'='*60}\n")

    converged = False

    try:
        for cycle in range(1, max_cycles + 1):

            # ==============================================================
            # Phase A: Train alpha on within-site pairs (beta frozen)
            # ==============================================================
            if cycle == 1 or cycle % 10 == 0:
                print(f"{'─'*60}")
                print(f"  Cycle {cycle}/{max_cycles} — Phase A: "
                      f"Optimise ALPHA ({n_alpha_epochs} ep, beta frozen)")
                print(f"{'─'*60}")

            for p in model.beta_net.parameters():
                p.requires_grad = False
            for p in model.alpha_net.parameters():
                p.requires_grad = True
            if model.effort_net is not None:
                for p in model.effort_net.parameters():
                    p.requires_grad = True

            s1_best_loss_this_cycle = float("inf")

            for ep in range(1, n_alpha_epochs + 1):
                global_epoch += 1

                train_m = train_one_epoch(
                    model, s1_train, alpha_optimizer, site_data,
                    device, log_every=config.log_every,
                    weight_clamp_factor=config.weight_clamp_factor,
                    weight_log_normalise=config.weight_log_normalise,
                )
                val_m = validate(model, s1_val, site_data, device,
                                 weight_clamp_factor=config.weight_clamp_factor,
                                 weight_log_normalise=config.weight_log_normalise)
                alpha_scheduler.step()
                lr = alpha_optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start

                # Compact logging: first, last, and every 5th epoch
                if ep == 1 or ep == n_alpha_epochs or (ep % 5 == 0 and n_alpha_epochs >= 10):
                    print(
                        f"    A-{ep:3d}/{n_alpha_epochs}  "
                        f"loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.3f}  "
                        f"α=[{val_m['alpha_min']:.0f}, {val_m['alpha_mean']:.0f}, "
                        f"{val_m['alpha_max']:.0f}]"
                    )

                log_writer.writerow({
                    "cycle": cycle, "phase": "alpha",
                    "phase_epoch": ep, "global_epoch": global_epoch,
                    "train_loss": f"{train_m['loss']:.6f}",
                    "val_loss": f"{val_m['loss']:.6f}",
                    "train_bce": f"{train_m['bce_loss']:.6f}",
                    "train_lb": f"{train_m['lb_penalty']:.6f}",
                    "train_anchor": f"{train_m['anchor_penalty']:.6f}",
                    "train_effort": f"{train_m['effort_loss']:.6f}",
                    "val_bce": f"{val_m['bce_loss']:.6f}",
                    "val_effort": f"{val_m['effort_loss']:.6f}",
                    "val_accuracy": f"{val_m['accuracy']:.4f}",
                    "alpha_mean": f"{val_m['alpha_mean']:.2f}",
                    "alpha_std": f"{val_m['alpha_std']:.2f}",
                    "alpha_min": f"{val_m['alpha_min']:.2f}",
                    "alpha_max": f"{val_m['alpha_max']:.2f}",
                    "alpha_grad_norm": f"{train_m['alpha_grad_norm']:.6f}",
                    # Beta columns blank for alpha phase
                    "train_eta_mean": "", "val_eta_mean": "", "val_eta_std": "",
                    "val_eta_min": "", "val_eta_max": "", "val_auc": "", "val_auc_s123": "",
                    "beta_grad_norm": "", "eta_ac_loss": "",
                    "weight_ess": f"{train_m['weight_ess']:.1f}",
                    "lr": f"{lr:.2e}",
                    "elapsed_sec": f"{elapsed:.0f}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                log_file.flush()

                if val_m["loss"] < s1_best_loss_this_cycle:
                    s1_best_loss_this_cycle = val_m["loss"]
                    _save_checkpoint(model, alpha_optimizer, global_epoch,
                                     val_m["loss"], site_data, config,
                                     stage1_ckpt_path)

            # Report alpha distribution at key cycles
            if cycle == 1 or cycle % 10 == 0 or cycle == max_cycles:
                with torch.no_grad():
                    model.eval()
                    Z = site_data.Z.to(device)
                    W = site_data.W.to(device) if site_data.W is not None else None
                    all_alpha = model._compute_alpha(Z, W).cpu()
                    print(f"  Alpha after cycle {cycle}: mean={all_alpha.mean():.1f}, "
                          f"std={all_alpha.std():.1f}, "
                          f"min={all_alpha.min():.1f}, max={all_alpha.max():.1f}")

            # ==============================================================
            # Phase B: Train beta on between-site pairs (alpha frozen)
            # ==============================================================
            mining_active = (
                use_hard_mining
                and cycle > config.hard_mining_warmup_cycles
            )
            mining_label = " + hard mining" if mining_active else ""

            if cycle == 1 or cycle % 10 == 0:
                print(f"\n{'─'*60}")
                print(f"  Cycle {cycle}/{max_cycles} — Phase B: "
                      f"Optimise BETA ({n_beta_epochs} ep, alpha frozen{mining_label})")
                print(f"{'─'*60}")

            for p in model.alpha_net.parameters():
                p.requires_grad = False
            if model.effort_net is not None:
                for p in model.effort_net.parameters():
                    p.requires_grad = False
            for p in model.beta_net.parameters():
                p.requires_grad = True

            # Refresh hard-pair scores and rebuild mined dataloader
            if mining_active:
                hard_miner.refresh_scores(model, site_data, device,
                                          batch_size=config.batch_size * 2)
                s2_train_mined = hard_miner.make_dataloader(
                    batch_size=config.batch_size)
            else:
                s2_train_mined = s2_train

            s2_best_loss_this_cycle = float("inf")
            s2_patience_ctr = 0
            eps = 1e-7

            for ep in range(1, n_beta_epochs + 1):
                global_epoch += 1

                # Mid-phase refresh of hardness scores
                if (mining_active
                        and ep > 1
                        and config.hard_mining_refresh_every > 0
                        and (ep - 1) % config.hard_mining_refresh_every == 0):
                    hard_miner.refresh_scores(model, site_data, device,
                                              batch_size=config.batch_size * 2)
                    s2_train_mined = hard_miner.make_dataloader(
                        batch_size=config.batch_size)

                # ---- Train (importance-weighted) ----
                model.beta_net.train()
                model.alpha_net.eval()
                Z = site_data.Z.to(device)
                W = site_data.W.to(device) if site_data.W is not None else None

                total_loss_t = 0.0
                total_eta_t = 0.0
                total_eta_ac_t = 0.0
                total_bg = 0.0
                total_ess_t = 0.0
                nb = 0

                # Retention rate keys for retrospective correction
                _rr_keys = {
                    1: ("between_tier1_r0", "between_tier1_r1"),
                    2: ("between_tier2_r0", "between_tier2_r1"),
                    3: ("between_tier3_r0", "between_tier3_r1"),
                }
                rr = model.retention_rates

                for bi, batch in enumerate(s2_train_mined):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    z_i = Z[batch["site_i"]]
                    z_j = Z[batch["site_j"]]
                    w_i = W[batch["site_i"]] if W is not None else None
                    w_j = W[batch["site_j"]] if W is not None else None

                    fwd = model.forward(z_i, z_j, batch["env_diff"],
                                        batch["is_within"],
                                        w_i=w_i, w_j=w_j,
                                        env_i=batch.get("env_i"),
                                        env_j=batch.get("env_j"),
                                        geo_dist=batch.get("geo_dist"))
                    p = fwd["p_match"].clamp(eps, 1.0 - eps)
                    y = batch["y"]
                    eta = fwd["eta"]
                    stratum = batch["stratum"]

                    # ---- Retrospective Bernoulli correction per stratum ----
                    # Apply p → p_samp = r0*p / (r0*p + r1*(1-p))
                    # for each between-site tier using sampler retention rates.
                    p_corr = p.clone()
                    for s_idx, (r0_key, r1_key) in _rr_keys.items():
                        mask_s = stratum == s_idx
                        if not mask_s.any():
                            continue
                        r0 = rr.get(r0_key, 1.0)
                        r1 = rr.get(r1_key, 1.0)
                        if r0 != 1.0 or r1 != 1.0:
                            p_s = p[mask_s]
                            p_corr[mask_s] = (
                                r0 * p_s / (r0 * p_s + r1 * (1.0 - p_s) + eps)
                            ).clamp(eps, 1.0 - eps)

                    log_p_corr = torch.log(p_corr)
                    log_1m_p_corr = torch.log(1.0 - p_corr)
                    bce = -(1.0 - y) * log_p_corr - y * log_1m_p_corr

                    # Design weights (proper IPW from sampler) + optional mining correction
                    imp_w = batch["design_w"].clone()
                    if mining_active and "mining_iw" in batch:
                        imp_w = imp_w * batch["mining_iw"]

                    # Stabilise and compute ESS
                    imp_w, batch_ess = _stabilise_weights(
                        imp_w, config.weight_clamp_factor,
                        config.weight_log_normalise)
                    total_ess_t += batch_ess

                    loss = (imp_w * bce).sum() / imp_w.sum()

                    # Anti-collapse regularizer: -λ · mean(log(η + ε))
                    eta_ac_loss = torch.tensor(0.0, device=p.device)
                    if model.eta_anti_collapse_lambda > 0:
                        eta_ac_loss = (
                            -model.eta_anti_collapse_lambda
                            * torch.log(eta + 1e-4).mean()
                        )
                        loss = loss + eta_ac_loss

                    # Geographic parameter L2 penalty (beta only during Phase B)
                    if model.geo_penalty_beta > 0 and model.K_env_env > 0:
                        _, beta_geo_l2 = model.geo_param_penalty(
                            K_alpha_env=model.K_alpha_env,
                            K_env_env=model.K_env_env)
                        loss = loss + model.geo_penalty_beta * beta_geo_l2

                    beta_optimizer.zero_grad()
                    loss.backward()

                    bg = [pp.grad.norm().item()
                          for pp in model.beta_net.parameters()
                          if pp.grad is not None]
                    avg_bg = sum(bg) / max(len(bg), 1)
                    total_bg += avg_bg

                    # One-time hook verification (first batch, first beta epoch, first cycle)
                    if bi == 0 and ep == 1 and cycle == 1:
                        print(f"  [HOOK CHECK] beta_grad_scale={config.beta_grad_scale}, "
                              f"avg_param_grad_norm={avg_bg:.6f}  "
                              f"(expect ~{0.0005 * config.beta_grad_scale:.4f} if hooks firing)")

                    nn.utils.clip_grad_norm_(
                        model.beta_net.parameters(), max_norm=5.0)
                    beta_optimizer.step()

                    total_loss_t += loss.item()
                    total_eta_t += eta.mean().item()
                    total_eta_ac_t += eta_ac_loss.item()
                    nb += 1

                    if (config.log_every > 0
                            and (bi + 1) % config.log_every == 0):
                        print(f"      batch {bi+1}/{len(s2_train_mined)}  "
                              f"loss={loss.item():.4f}  "
                              f"η_mean={eta.mean().item():.4f}  "
                              f"η_max={eta.max().item():.4f}")

                train_m2 = {
                    "loss": total_loss_t / max(nb, 1),
                    "eta_mean": total_eta_t / max(nb, 1),
                    "eta_ac_loss": total_eta_ac_t / max(nb, 1),
                    "beta_grad_norm": total_bg / max(nb, 1),
                    "weight_ess": total_ess_t / max(nb, 1),
                }

                val_m2 = _stage2_validate(
                    model, s2_val, site_data, device,
                    weight_clamp_factor=config.weight_clamp_factor,
                    weight_log_normalise=config.weight_log_normalise)
                beta_scheduler.step()
                lr = beta_optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start

                # Compact logging: first, last, and every 5th epoch
                if ep == 1 or ep == n_beta_epochs or (ep % 5 == 0 and n_beta_epochs >= 10):
                    print(
                        f"    B-{ep:3d}/{n_beta_epochs}  "
                        f"loss={val_m2['loss']:.4f}  "
                        f"η=[{val_m2['eta_min']:.2f}, "
                        f"{val_m2['eta_mean']:.2f}, "
                        f"{val_m2['eta_max']:.2f}]  "
                        f"AUC={val_m2['auc']:.3f} "
                        f"AUC_s123={val_m2['auc_s123']:.3f}"
                    )

                log_writer.writerow({
                    "cycle": cycle, "phase": "beta",
                    "phase_epoch": ep, "global_epoch": global_epoch,
                    "train_loss": f"{train_m2['loss']:.6f}",
                    "val_loss": f"{val_m2['loss']:.6f}",
                    # Alpha columns blank
                    "train_bce": "", "train_lb": "", "train_anchor": "", "train_effort": "",
                    "val_bce": "", "val_effort": "",
                    "val_accuracy": "",
                    "alpha_mean": "", "alpha_std": "",
                    "alpha_min": "", "alpha_max": "",
                    "alpha_grad_norm": "",
                    # Beta phase columns
                    "train_eta_mean": f"{train_m2['eta_mean']:.6f}",
                    "val_eta_mean": f"{val_m2['eta_mean']:.6f}",
                    "val_eta_std": f"{val_m2['eta_std']:.6f}",
                    "val_eta_min": f"{val_m2['eta_min']:.6f}",
                    "val_eta_max": f"{val_m2['eta_max']:.6f}",
                    "val_auc": f"{val_m2['auc']:.4f}",
                    "val_auc_s123": f"{val_m2['auc_s123']:.4f}",
                    "beta_grad_norm": f"{train_m2['beta_grad_norm']:.6f}",
                    "eta_ac_loss": f"{train_m2['eta_ac_loss']:.6f}",
                    "weight_ess": f"{train_m2['weight_ess']:.1f}",
                    "lr": f"{lr:.2e}",
                    "elapsed_sec": f"{elapsed:.0f}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                log_file.flush()

                # Save best model (Stage 2 loss is the overall metric)
                if val_m2["loss"] < s2_best_loss_this_cycle:
                    s2_best_loss_this_cycle = val_m2["loss"]
                    s2_patience_ctr = 0

                    if val_m2["loss"] < overall_best_loss:
                        overall_best_loss = val_m2["loss"]
                        state.best_val_loss = overall_best_loss
                        state.best_epoch = global_epoch
                        _save_checkpoint(
                            model, beta_optimizer, global_epoch,
                            val_m2["loss"], site_data, config,
                            best_model_path)
                        print(f"      *** New overall best "
                              f"(val_loss={overall_best_loss:.6f}) ***")
                else:
                    s2_patience_ctr += 1
                    if beta_patience > 0 and s2_patience_ctr >= beta_patience:
                        print(f"      Beta early stopping at epoch {ep}/{n_beta_epochs} "
                              f"(no improvement for {beta_patience} epochs)")
                        break

            # ==============================================================
            # Check cycle convergence
            # ==============================================================
            cycle_loss = s2_best_loss_this_cycle
            if prev_best_loss < float("inf"):
                rel_change = abs(cycle_loss - prev_best_loss) / (
                    abs(prev_best_loss) + 1e-8)
            else:
                rel_change = float("inf")

            elapsed = time.time() - t_start
            # Compact cycle summary: every cycle for first 5, then every 10th
            if cycle <= 5 or cycle % 10 == 0 or rel_change < tol:
                print(f"\n  Cycle {cycle}: "
                      f"beta_loss={cycle_loss:.6f}, "
                      f"rel_change={rel_change:.2e}, "
                      f"best_overall={overall_best_loss:.6f}, "
                      f"beta_epochs_used={ep}/{n_beta_epochs}, "
                      f"time={elapsed:.0f}s")

            state.history.append({
                "cycle": cycle,
                "beta_val_loss": cycle_loss,
                "rel_change": rel_change,
                "global_epoch": global_epoch,
                "elapsed": elapsed,
            })

            if rel_change < tol:
                converged = True
                print(f"\n  *** Converged at cycle {cycle} "
                      f"(rel_change {rel_change:.2e} < tol {tol:.1e}) ***")
                break

            prev_best_loss = cycle_loss

    finally:
        log_file.close()

    total_time = time.time() - t_start

    if not converged:
        print(f"\n  Max cycles ({max_cycles}) reached "
              f"(final rel_change={rel_change:.2e})")

    # Restore overall best model
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"\n  Cyclic training complete.")
    print(f"  Best global epoch: {state.best_epoch}")
    print(f"  Best val loss:     {state.best_val_loss:.6f}")
    print(f"  Total cycles:      {cycle}")
    print(f"  Converged:         {converged}")
    print(f"  Training time:     {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final model:       {best_model_path}")

    return state


# --------------------------------------------------------------------------
# Phase 2: Fine-tune with geographic features
# --------------------------------------------------------------------------

def train_finetune_geo(
    model: CLESSONet,
    pairs,  # pd.DataFrame — all pairs
    site_data: SiteData,  # NEW site_data with Fourier + geo_dist features
    config: CLESSONNConfig,
    best_phase1_loss: float = float("inf"),
) -> TrainingState:
    """Fine-tune a Phase-1-trained model after expanding with geographic features.

    Joint training of alpha + beta with differential learning rates:
      - Existing (env) params: finetune_lr (small, preserve learned signal)
      - New geo params: finetune_lr * finetune_new_param_lr_mult (larger, learn quickly)

    Uses importance-weighted BCE on ALL pairs (within + between) so both alpha
    and beta receive gradients simultaneously.
    """
    device = config.resolve_device()
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Fine-tune with Geographic Features")
    print(f"{'='*60}")
    print(f"  K_alpha: {site_data.K_alpha}  K_env: {site_data.K_env}")
    freeze_existing = getattr(config, 'finetune_freeze_existing', False)
    if freeze_existing:
        print(f"  Mode: FROZEN existing params — training new geo params only")
        print(f"  LR (new geo): {config.finetune_lr * config.finetune_new_param_lr_mult:.1e}")
    else:
        print(f"  LR (existing): {config.finetune_lr:.1e}  "
              f"LR (new geo): {config.finetune_lr * config.finetune_new_param_lr_mult:.1e}")
    print(f"  Max epochs: {config.finetune_max_epochs}  "
          f"Patience: {config.finetune_patience}")

    # Geographic parameter L2 penalty (reads strengths from model attributes)
    if model.geo_penalty_alpha > 0 or model.geo_penalty_beta > 0:
        print(f"  Geo L2 penalty: alpha={model.geo_penalty_alpha}, "
              f"beta={model.geo_penalty_beta}")

    model = model.to(device)

    # Count env-only dim_nets (from Phase 1) vs new geo dim_net(s)
    K_env_total = site_data.K_env
    n_env_dims = len(site_data.env_cov_names)  # pure env dimensions
    if hasattr(site_data, 'geo') and site_data.geo is not None:
        n_env_dims += site_data.geo.shape[1]
    n_new_dims = K_env_total - n_env_dims  # geo_dist dim_net(s)

    # Count env-only alpha input columns (non-Fourier)
    K_alpha_env = sum(1 for c in site_data.alpha_cov_names
                      if not c.startswith("fourier_"))

    # Separate parameters into existing vs new
    existing_params = []
    new_geo_params = []

    # Alpha params
    for p in model.alpha_net.parameters():
        if freeze_existing:
            p.requires_grad = False
        else:
            p.requires_grad = True
            existing_params.append(p)

    # Effort params — optionally frozen via finetune_freeze_effort
    # When trainable, effort gets its own param group with dedicated LR.
    freeze_effort = getattr(config, 'finetune_freeze_effort', False)
    effort_params = []
    if hasattr(model, 'effort_net') and model.effort_net is not None:
        for p in model.effort_net.parameters():
            if freeze_effort:
                p.requires_grad = False
            else:
                p.requires_grad = True
                effort_params.append(p)

    # Beta params: existing per-dim components vs new geo components
    from src.clesso_nn.model import AdditiveBetaNet, FactoredDeepBetaNet

    beta = model.beta_net
    if isinstance(beta, AdditiveBetaNet):
        for k, dk in enumerate(beta.dim_nets):
            if k < n_env_dims:
                for p in dk.parameters():
                    if freeze_existing:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                        existing_params.append(p)
            else:
                for p in dk.parameters():
                    p.requires_grad = True
                    new_geo_params.append(p)

    elif isinstance(beta, FactoredDeepBetaNet):
        # Per-dim encoders + projectors: existing vs new
        for k in range(beta.K_env):
            parts = list(beta.encoders[k].parameters()) + list(beta.dim_projectors[k].parameters())
            if k < n_env_dims:
                for p in parts:
                    if freeze_existing:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                        existing_params.append(p)
            else:
                for p in parts:
                    p.requires_grad = True
                    new_geo_params.append(p)
        # Interaction net: treat as existing (shared across all dims)
        for p in beta.interaction_net.parameters():
            if freeze_existing:
                p.requires_grad = False
            else:
                p.requires_grad = True
                existing_params.append(p)

    else:
        # Deep beta or unknown — treat all as existing
        for p in beta.parameters():
            if freeze_existing:
                p.requires_grad = False
            else:
                p.requires_grad = True
                existing_params.append(p)

    n_existing = sum(p.numel() for p in existing_params)
    n_effort = sum(p.numel() for p in effort_params)
    n_new = sum(p.numel() for p in new_geo_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Freeze existing: {freeze_existing}")
    print(f"  Trainable existing params: {n_existing:,}")
    print(f"  Trainable effort params:   {n_effort:,}")
    print(f"  Trainable new geo params:  {n_new:,}")
    print(f"  Frozen params:             {n_frozen:,}")

    new_lr = config.finetune_lr * config.finetune_new_param_lr_mult
    param_groups = []
    if existing_params:
        param_groups.append(
            {"params": existing_params, "lr": config.finetune_lr,
             "weight_decay": config.weight_decay},
        )
    if effort_params:
        param_groups.append(
            {"params": effort_params,
             "lr": config.resolved_effort_lr,
             "weight_decay": config.resolved_effort_wd},
        )
        print(f"  EffortNet LR (finetune): {config.resolved_effort_lr:.2e}  "
              f"WD: {config.resolved_effort_wd:.2e}")
    if new_geo_params:
        param_groups.append(
            {"params": new_geo_params, "lr": new_lr,
             "weight_decay": config.weight_decay},
        )

    if not param_groups:
        print("  WARNING: No trainable parameters for Phase 2. "
              "Skipping fine-tuning.")
        return TrainingState()

    optimizer = Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Build dataloaders with NEW site_data (includes Fourier + geo_dist)
    import pandas as pd

    between_pairs = pairs[pairs["is_within"] == 0].reset_index(drop=True)

    # All-pairs dataloaders for joint training
    all_train, all_val, _, _ = make_dataloaders(
        pairs, site_data,
        val_fraction=config.val_fraction, batch_size=config.batch_size,
        seed=config.seed, use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )
    # Between-site val loader for AUC tracking
    _, s2_val, _, _ = make_dataloaders(
        between_pairs, site_data,
        val_fraction=config.val_fraction, batch_size=config.batch_size,
        seed=config.seed, use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )

    # Progress log
    log_path = config.output_dir / "training_progress_finetune.log"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "train_loss", "val_loss",
        "val_bce", "val_effort", "val_accuracy",
        "alpha_mean", "alpha_std", "alpha_min", "alpha_max",
        "val_eta_mean", "val_eta_std", "val_auc",
        "weight_ess",
        "lr_existing", "lr_new", "elapsed_sec", "timestamp",
    ])
    log_writer.writeheader()
    log_file.flush()
    print(f"  Finetune log: {log_path}")

    # Checkpoint paths
    best_model_path = config.output_dir / "best_model.pt"
    phase2_ckpt_path = config.output_dir / "best_model_phase2.pt"

    state = TrainingState()
    # Note: Phase 1 best_val_loss used between-site-only metric, which is
    # incomparable with our all-pairs metric. Start fresh for Phase 2.
    best_val_loss = float("inf")
    patience_ctr = 0
    eps = 1e-7
    t_start = time.time()

    Z = site_data.Z.to(device)
    S_obs = site_data.S_obs.to(device)
    W = site_data.W.to(device) if site_data.W is not None else None
    anchor_richness = site_data.anchor_richness.to(device) if site_data.anchor_richness.any() else None

    for epoch in range(1, config.finetune_max_epochs + 1):
        # ---- Train ----
        model.train()
        total_loss_t = 0.0
        total_ess_t = 0.0
        nb = 0

        for batch in all_train:
            batch = {k: v.to(device) for k, v in batch.items()}
            z_i = Z[batch["site_i"]]
            z_j = Z[batch["site_j"]]
            w_i = W[batch["site_i"]] if W is not None else None
            w_j = W[batch["site_j"]] if W is not None else None

            fwd = model.forward(z_i, z_j, batch["env_diff"], batch["is_within"],
                                w_i=w_i, w_j=w_j,
                                env_i=batch.get("env_i"),
                                env_j=batch.get("env_j"),
                                geo_dist=batch.get("geo_dist"))
            p = fwd["p_match"].clamp(eps, 1.0 - eps)
            y = batch["y"]
            stratum = batch["stratum"]

            # ---- Retrospective Bernoulli correction per stratum ----
            _rr_keys = {
                0: ("within_r0", "within_r1"),
                1: ("between_tier1_r0", "between_tier1_r1"),
                2: ("between_tier2_r0", "between_tier2_r1"),
                3: ("between_tier3_r0", "between_tier3_r1"),
            }
            rr = model.retention_rates
            p_corr = p.clone()
            for s_idx, (r0_key, r1_key) in _rr_keys.items():
                mask_s = stratum == s_idx
                if not mask_s.any():
                    continue
                r0 = rr.get(r0_key, 1.0)
                r1 = rr.get(r1_key, 1.0)
                if r0 != 1.0 or r1 != 1.0:
                    p_s = p[mask_s]
                    p_corr[mask_s] = (
                        r0 * p_s / (r0 * p_s + r1 * (1.0 - p_s) + eps)
                    ).clamp(eps, 1.0 - eps)

            log_p_corr = torch.log(p_corr)
            log_1m_p_corr = torch.log(1.0 - p_corr)
            bce = -(1.0 - y) * log_p_corr - y * log_1m_p_corr

            # Design weights (proper IPW from sampler)
            imp_w = batch["design_w"].clone()

            # Stabilise and compute ESS
            imp_w, batch_ess = _stabilise_weights(
                imp_w, config.weight_clamp_factor,
                config.weight_log_normalise)
            total_ess_t += batch_ess

            bce_weighted = (imp_w * bce).sum() / imp_w.sum()

            # Pre-compute α_env for lb + anchor + richness_anchor penalties
            alpha_env_i = alpha_env_j = None
            if model.alpha_lb_lambda > 0.0 or model.alpha_anchor_lambda > 0.0 \
                    or model.richness_anchor_lambda > 0.0:
                alpha_env_i = model._compute_alpha_env_only(z_i)
                alpha_env_j = model._compute_alpha_env_only(z_j)

            # Alpha lower-bound penalty on α_env (log-space: scale-invariant)
            lb_penalty = torch.tensor(0.0, device=device)
            if model.alpha_lb_lambda > 0.0 and alpha_env_i is not None:
                s_i_lb = S_obs[batch["site_i"]]
                s_j_lb = S_obs[batch["site_j"]]
                eps_lb = 1.0
                log_gap_i = torch.log(s_i_lb.clamp(min=eps_lb)) - torch.log(alpha_env_i)
                log_gap_j = torch.log(s_j_lb.clamp(min=eps_lb)) - torch.log(alpha_env_j)
                viol_i = torch.nn.functional.softplus(10.0 * log_gap_i) / 10.0
                viol_j = torch.nn.functional.softplus(10.0 * log_gap_j) / 10.0
                lb_penalty = model.alpha_lb_lambda * (
                    viol_i.pow(2).mean() + viol_j.pow(2).mean())

            # Alpha anchor penalty on α_env (effort-stripped true richness)
            anchor_penalty = torch.tensor(0.0, device=device)
            if model.alpha_anchor_lambda > 0.0 and alpha_env_i is not None:
                s_i = S_obs[batch["site_i"]].clamp(min=1.0)
                s_j = S_obs[batch["site_j"]].clamp(min=1.0)
                tau = model.alpha_anchor_tolerance
                gap_i = (torch.log(alpha_env_i) - torch.log(s_i)).abs() - tau
                gap_j = (torch.log(alpha_env_j) - torch.log(s_j)).abs() - tau
                viol_i_a = torch.nn.functional.relu(gap_i)
                viol_j_a = torch.nn.functional.relu(gap_j)
                anchor_penalty = model.alpha_anchor_lambda * (
                    viol_i_a.pow(2).mean() + viol_j_a.pow(2).mean())

            # Richness anchor penalty (external raster-based soft constraint)
            richness_anchor_loss = torch.tensor(0.0, device=device)
            if model.richness_anchor_lambda > 0.0 and alpha_env_i is not None \
                    and anchor_richness is not None:
                anc_i = anchor_richness[batch["site_i"]]
                anc_j = anchor_richness[batch["site_j"]]
                valid_i = anc_i > 1.0
                valid_j = anc_j > 1.0
                if valid_i.any():
                    ra_i = (torch.log(alpha_env_i[valid_i]) - torch.log(anc_i[valid_i])).pow(2).mean()
                else:
                    ra_i = torch.tensor(0.0, device=device)
                if valid_j.any():
                    ra_j = (torch.log(alpha_env_j[valid_j]) - torch.log(anc_j[valid_j])).pow(2).mean()
                else:
                    ra_j = torch.tensor(0.0, device=device)
                richness_anchor_loss = model.richness_anchor_lambda * (ra_i + ra_j) / 2.0

            loss = bce_weighted + lb_penalty + anchor_penalty + richness_anchor_loss

            # Geographic parameter L2 penalty (separate from weight decay)
            if model.geo_penalty_alpha > 0 or model.geo_penalty_beta > 0:
                alpha_geo_l2, beta_geo_l2 = model.geo_param_penalty(
                    K_alpha_env=model.K_alpha_env, K_env_env=model.K_env_env)
                loss = loss + model.geo_penalty_alpha * alpha_geo_l2 \
                            + model.geo_penalty_beta * beta_geo_l2

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_t += loss.item()
            total_ess_t += batch_ess
            nb += 1

        train_loss = total_loss_t / max(nb, 1)
        train_ess = total_ess_t / max(nb, 1)

        # ---- Validate (all pairs for overall loss, between-site for AUC) ----
        val_m = validate(model, all_val, site_data, device,
                         weight_clamp_factor=config.weight_clamp_factor,
                         weight_log_normalise=config.weight_log_normalise)
        val_s2 = _stage2_validate(model, s2_val, site_data, device,
                                  weight_clamp_factor=config.weight_clamp_factor,
                                  weight_log_normalise=config.weight_log_normalise)
        scheduler.step()

        lr_exist = optimizer.param_groups[0]["lr"]
        lr_new_val = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_exist
        elapsed = time.time() - t_start

        # Log
        log_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_m['loss']:.6f}",
            "val_bce": f"{val_m['bce_loss']:.6f}",
            "val_effort": f"{val_m['effort_loss']:.6f}",
            "val_accuracy": f"{val_m['accuracy']:.4f}",
            "alpha_mean": f"{val_m['alpha_mean']:.2f}",
            "alpha_std": f"{val_m['alpha_std']:.2f}",
            "alpha_min": f"{val_m['alpha_min']:.2f}",
            "alpha_max": f"{val_m['alpha_max']:.2f}",
            "val_eta_mean": f"{val_s2['eta_mean']:.6f}",
            "val_eta_std": f"{val_s2['eta_std']:.6f}",
            "val_auc": f"{val_s2['auc']:.4f}",
            "weight_ess": f"{train_ess:.1f}",
            "lr_existing": f"{lr_exist:.2e}",
            "lr_new": f"{lr_new_val:.2e}",
            "elapsed_sec": f"{elapsed:.0f}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        log_file.flush()

        # Print progress
        if epoch == 1 or epoch % 5 == 0 or epoch == config.finetune_max_epochs:
            print(f"  FT-{epoch:3d}/{config.finetune_max_epochs}  "
                  f"beta_loss={val_s2['loss']:.6f}  "
                  f"AUC={val_s2['auc']:.3f}  "
                  f"α=[{val_m['alpha_min']:.0f},{val_m['alpha_mean']:.0f},"
                  f"{val_m['alpha_max']:.0f}]  "
                  f"η=[{val_s2['eta_min']:.2f},{val_s2['eta_mean']:.2f},"
                  f"{val_s2['eta_max']:.2f}]")

        # Save best — use between-site loss (comparable with Phase 1 metric)
        metric = val_s2["loss"]
        if metric < best_val_loss:
            best_val_loss = metric
            state.best_val_loss = best_val_loss
            state.best_epoch = epoch
            patience_ctr = 0
            _save_checkpoint(model, optimizer, epoch, metric,
                             site_data, config, best_model_path)
            _save_checkpoint(model, optimizer, epoch, metric,
                             site_data, config, phase2_ckpt_path)
            print(f"      *** New best (beta_loss={best_val_loss:.6f}, "
                  f"AUC={val_s2['auc']:.3f}) ***")
        else:
            patience_ctr += 1
            if patience_ctr >= config.finetune_patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {config.finetune_patience} epochs)")
                break

    log_file.close()
    total_time = time.time() - t_start

    # Restore best model
    if phase2_ckpt_path.exists():
        ckpt = torch.load(phase2_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    elif best_model_path.exists():
        try:
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            print("  Warning: Could not restore from best_model.pt (shape mismatch)")

    print(f"\n  Fine-tuning complete.")
    print(f"  Best epoch:        {state.best_epoch}")
    print(f"  Best val loss (between-site): {state.best_val_loss:.6f}")
    print(f"  Phase 1 val loss:  {best_phase1_loss:.6f}")
    improvement = (best_phase1_loss - state.best_val_loss) / best_phase1_loss * 100
    print(f"  Improvement:       {improvement:+.2f}%")
    print(f"  Training time:     {total_time:.0f}s ({total_time/60:.1f} min)")

    return state


# --------------------------------------------------------------------------
# Joint fine-tune — alpha+beta together on same env/effort variables
# --------------------------------------------------------------------------

def train_finetune_joint(
    model: CLESSONet,
    pairs,  # pd.DataFrame — all pairs
    site_data: SiteData,
    config: CLESSONNConfig,
    best_phase1_loss: float = float("inf"),
) -> TrainingState:
    """Joint fine-tune after cyclic Phase 1 — all params trainable, same features.

    Unlike train_finetune_geo which expands the model with geographic
    features, this trains alpha + beta *jointly* on the exact same
    env/effort variables used in Phase 1.  The idea is to let both
    networks find a joint optimum now that cycling has given each a
    good initialisation.

    Uses importance-weighted BCE on ALL pairs (within + between) so both
    alpha and beta receive gradients simultaneously.
    """
    import pandas as pd

    device = config.resolve_device()
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Joint Fine-tune (same env/effort variables)")
    print(f"{'='*60}")

    alpha_lr = config.finetune_lr * config.finetune_joint_alpha_lr_mult
    beta_lr = config.finetune_lr * config.finetune_joint_beta_lr_mult
    print(f"  LR alpha:  {alpha_lr:.2e}")
    print(f"  LR beta:   {beta_lr:.2e}")
    if model.effort_net is not None:
        print(f"  LR effort: {config.resolved_effort_lr:.2e}")
    print(f"  Max epochs: {config.finetune_max_epochs}  "
          f"Patience: {config.finetune_patience}")

    model = model.to(device)

    # All parameters trainable
    for p in model.parameters():
        p.requires_grad = True

    # Build differential LR param groups
    alpha_params = list(model.alpha_net.parameters())
    beta_params = list(model.beta_net.parameters())
    effort_params = []
    if model.effort_net is not None:
        effort_params = list(model.effort_net.parameters())

    param_groups = [
        {"params": alpha_params, "lr": alpha_lr,
         "weight_decay": config.weight_decay},
        {"params": beta_params, "lr": beta_lr,
         "weight_decay": config.weight_decay},
    ]
    if effort_params:
        param_groups.append(
            {"params": effort_params,
             "lr": config.resolved_effort_lr,
             "weight_decay": config.resolved_effort_wd},
        )

    n_alpha = sum(p.numel() for p in alpha_params)
    n_beta = sum(p.numel() for p in beta_params)
    n_effort = sum(p.numel() for p in effort_params)
    print(f"  Alpha params: {n_alpha:,}  Beta params: {n_beta:,}  "
          f"Effort params: {n_effort:,}")

    optimizer = Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Build dataloaders on SAME site_data (no geo expansion)
    between_pairs = pairs[pairs["is_within"] == 0].reset_index(drop=True)

    all_train, all_val, _, _ = make_dataloaders(
        pairs, site_data,
        val_fraction=config.val_fraction, batch_size=config.batch_size,
        seed=config.seed, use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )
    _, s2_val, _, _ = make_dataloaders(
        between_pairs, site_data,
        val_fraction=config.val_fraction, batch_size=config.batch_size,
        seed=config.seed, use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )

    # Progress log
    log_path = config.output_dir / "training_progress_finetune.log"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "train_loss", "val_loss",
        "val_bce", "val_effort", "val_accuracy",
        "alpha_mean", "alpha_std", "alpha_min", "alpha_max",
        "val_eta_mean", "val_eta_std", "val_auc", "val_auc_s123",
        "weight_ess",
        "lr_alpha", "lr_beta", "elapsed_sec", "timestamp",
    ])
    log_writer.writeheader()
    log_file.flush()
    print(f"  Joint finetune log: {log_path}")

    # Checkpoint paths
    best_model_path = config.output_dir / "best_model.pt"
    phase2_ckpt_path = config.output_dir / "best_model_phase2.pt"

    state = TrainingState()
    best_val_loss = float("inf")
    patience_ctr = 0
    eps = 1e-7
    t_start = time.time()

    Z = site_data.Z.to(device)
    S_obs = site_data.S_obs.to(device)
    W = site_data.W.to(device) if site_data.W is not None else None
    anchor_richness = site_data.anchor_richness.to(device) if site_data.anchor_richness.any() else None

    for epoch in range(1, config.finetune_max_epochs + 1):
        # ---- Train ----
        model.train()
        total_loss_t = 0.0
        total_ess_t = 0.0
        nb = 0

        for batch in all_train:
            batch = {k: v.to(device) for k, v in batch.items()}
            z_i = Z[batch["site_i"]]
            z_j = Z[batch["site_j"]]
            w_i = W[batch["site_i"]] if W is not None else None
            w_j = W[batch["site_j"]] if W is not None else None

            fwd = model.forward(z_i, z_j, batch["env_diff"], batch["is_within"],
                                w_i=w_i, w_j=w_j,
                                env_i=batch.get("env_i"),
                                env_j=batch.get("env_j"),
                                geo_dist=batch.get("geo_dist"))
            p = fwd["p_match"].clamp(eps, 1.0 - eps)
            y = batch["y"]
            stratum = batch["stratum"]

            # ---- Retrospective Bernoulli correction per stratum ----
            _rr_keys = {
                0: ("within_r0", "within_r1"),
                1: ("between_tier1_r0", "between_tier1_r1"),
                2: ("between_tier2_r0", "between_tier2_r1"),
                3: ("between_tier3_r0", "between_tier3_r1"),
            }
            rr = model.retention_rates
            p_corr = p.clone()
            for s_idx, (r0_key, r1_key) in _rr_keys.items():
                mask_s = stratum == s_idx
                if not mask_s.any():
                    continue
                r0 = rr.get(r0_key, 1.0)
                r1 = rr.get(r1_key, 1.0)
                if r0 != 1.0 or r1 != 1.0:
                    p_s = p[mask_s]
                    p_corr[mask_s] = (
                        r0 * p_s / (r0 * p_s + r1 * (1.0 - p_s) + eps)
                    ).clamp(eps, 1.0 - eps)

            log_p_corr = torch.log(p_corr)
            log_1m_p_corr = torch.log(1.0 - p_corr)
            bce = -(1.0 - y) * log_p_corr - y * log_1m_p_corr

            # Design weights
            imp_w = batch["design_w"].clone()
            imp_w, batch_ess = _stabilise_weights(
                imp_w, config.weight_clamp_factor,
                config.weight_log_normalise)
            total_ess_t += batch_ess

            bce_weighted = (imp_w * bce).sum() / imp_w.sum()

            # Pre-compute α_env for lb + anchor + richness_anchor penalties
            alpha_env_i = alpha_env_j = None
            if model.alpha_lb_lambda > 0.0 or model.alpha_anchor_lambda > 0.0 \
                    or model.richness_anchor_lambda > 0.0:
                alpha_env_i = model._compute_alpha_env_only(z_i)
                alpha_env_j = model._compute_alpha_env_only(z_j)

            # Alpha lower-bound penalty on α_env (true richness must ≥ S_obs)
            lb_penalty = torch.tensor(0.0, device=device)
            if model.alpha_lb_lambda > 0.0 and alpha_env_i is not None:
                s_i_lb = S_obs[batch["site_i"]]
                s_j_lb = S_obs[batch["site_j"]]
                eps_lb = 1.0
                log_gap_i = torch.log(s_i_lb.clamp(min=eps_lb)) - torch.log(alpha_env_i)
                log_gap_j = torch.log(s_j_lb.clamp(min=eps_lb)) - torch.log(alpha_env_j)
                viol_i = torch.nn.functional.softplus(10.0 * log_gap_i) / 10.0
                viol_j = torch.nn.functional.softplus(10.0 * log_gap_j) / 10.0
                lb_penalty = model.alpha_lb_lambda * (
                    viol_i.pow(2).mean() + viol_j.pow(2).mean())

            # Alpha anchor penalty on α_env (effort-stripped true richness)
            anchor_penalty = torch.tensor(0.0, device=device)
            if model.alpha_anchor_lambda > 0.0 and alpha_env_i is not None:
                s_i = S_obs[batch["site_i"]].clamp(min=1.0)
                s_j = S_obs[batch["site_j"]].clamp(min=1.0)
                tau = model.alpha_anchor_tolerance
                gap_i = (torch.log(alpha_env_i) - torch.log(s_i)).abs() - tau
                gap_j = (torch.log(alpha_env_j) - torch.log(s_j)).abs() - tau
                viol_i_a = torch.nn.functional.relu(gap_i)
                viol_j_a = torch.nn.functional.relu(gap_j)
                anchor_penalty = model.alpha_anchor_lambda * (
                    viol_i_a.pow(2).mean() + viol_j_a.pow(2).mean())

            # Richness anchor penalty (external raster-based soft constraint)
            richness_anchor_loss = torch.tensor(0.0, device=device)
            if model.richness_anchor_lambda > 0.0 and alpha_env_i is not None \
                    and anchor_richness is not None:
                anc_i = anchor_richness[batch["site_i"]]
                anc_j = anchor_richness[batch["site_j"]]
                valid_i = anc_i > 1.0
                valid_j = anc_j > 1.0
                if valid_i.any():
                    ra_i = (torch.log(alpha_env_i[valid_i]) - torch.log(anc_i[valid_i])).pow(2).mean()
                else:
                    ra_i = torch.tensor(0.0, device=device)
                if valid_j.any():
                    ra_j = (torch.log(alpha_env_j[valid_j]) - torch.log(anc_j[valid_j])).pow(2).mean()
                else:
                    ra_j = torch.tensor(0.0, device=device)
                richness_anchor_loss = model.richness_anchor_lambda * (ra_i + ra_j) / 2.0

            loss = bce_weighted + lb_penalty + anchor_penalty + richness_anchor_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_t += loss.item()
            nb += 1

        train_loss = total_loss_t / max(nb, 1)
        train_ess = total_ess_t / max(nb, 1)

        # ---- Validate ----
        val_m = validate(model, all_val, site_data, device,
                         weight_clamp_factor=config.weight_clamp_factor,
                         weight_log_normalise=config.weight_log_normalise)
        val_s2 = _stage2_validate(model, s2_val, site_data, device,
                                  weight_clamp_factor=config.weight_clamp_factor,
                                  weight_log_normalise=config.weight_log_normalise)
        scheduler.step()

        lr_a = optimizer.param_groups[0]["lr"]
        lr_b = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t_start

        # Log
        log_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_m['loss']:.6f}",
            "val_bce": f"{val_m['bce_loss']:.6f}",
            "val_effort": f"{val_m['effort_loss']:.6f}",
            "val_accuracy": f"{val_m['accuracy']:.4f}",
            "alpha_mean": f"{val_m['alpha_mean']:.2f}",
            "alpha_std": f"{val_m['alpha_std']:.2f}",
            "alpha_min": f"{val_m['alpha_min']:.2f}",
            "alpha_max": f"{val_m['alpha_max']:.2f}",
            "val_eta_mean": f"{val_s2['eta_mean']:.6f}",
            "val_eta_std": f"{val_s2['eta_std']:.6f}",
            "val_auc": f"{val_s2['auc']:.4f}",
            "val_auc_s123": f"{val_s2['auc_s123']:.4f}",
            "weight_ess": f"{train_ess:.1f}",
            "lr_alpha": f"{lr_a:.2e}",
            "lr_beta": f"{lr_b:.2e}",
            "elapsed_sec": f"{elapsed:.0f}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        log_file.flush()

        # Print progress
        if epoch == 1 or epoch % 5 == 0 or epoch == config.finetune_max_epochs:
            print(f"  JF-{epoch:3d}/{config.finetune_max_epochs}  "
                  f"loss={val_m['loss']:.6f}  beta_loss={val_s2['loss']:.6f}  "
                  f"AUC_s123={val_s2['auc_s123']:.3f}  "
                  f"α=[{val_m['alpha_min']:.0f},{val_m['alpha_mean']:.0f},"
                  f"{val_m['alpha_max']:.0f}]  "
                  f"η=[{val_s2['eta_min']:.2f},{val_s2['eta_mean']:.2f},"
                  f"{val_s2['eta_max']:.2f}]")

        # Save best — use between-site loss
        metric = val_s2["loss"]
        if metric < best_val_loss:
            best_val_loss = metric
            state.best_val_loss = best_val_loss
            state.best_epoch = epoch
            patience_ctr = 0
            _save_checkpoint(model, optimizer, epoch, metric,
                             site_data, config, best_model_path)
            _save_checkpoint(model, optimizer, epoch, metric,
                             site_data, config, phase2_ckpt_path)
            print(f"      *** New best (beta_loss={best_val_loss:.6f}, "
                  f"AUC_s123={val_s2['auc_s123']:.3f}) ***")
        else:
            patience_ctr += 1
            if patience_ctr >= config.finetune_patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {config.finetune_patience} epochs)")
                break

    log_file.close()
    total_time = time.time() - t_start

    # Restore best model
    if phase2_ckpt_path.exists():
        ckpt = torch.load(phase2_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    elif best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"\n  Joint fine-tuning complete.")
    print(f"  Best epoch:        {state.best_epoch}")
    print(f"  Best val loss (between-site): {state.best_val_loss:.6f}")
    print(f"  Phase 1 val loss:  {best_phase1_loss:.6f}")
    improvement = (best_phase1_loss - state.best_val_loss) / best_phase1_loss * 100
    print(f"  Improvement:       {improvement:+.2f}%")
    print(f"  Training time:     {total_time:.0f}s ({total_time/60:.1f} min)")

    return state


# --------------------------------------------------------------------------
# Calibration phase — design-weighted loss only, no mining, no boost
# --------------------------------------------------------------------------

def train_calibration(
    model: CLESSONet,
    pairs,  # pd.DataFrame — all pairs
    site_data: SiteData,
    config: CLESSONNConfig,
) -> TrainingState:
    """Calibration phase: short fine-tune with design-weighted composite loss.

    No hard-pair mining, no match-boost auxiliary loss.
    Uses a small learning rate and design weights only to de-bias
    probability estimates after quota sampling.
    """
    import pandas as pd

    device = config.resolve_device()
    print(f"\n{'='*60}")
    print(f"  CALIBRATION PHASE")
    print(f"{'='*60}")
    print(f"  Epochs: {config.calibration_epochs}  "
          f"LR: {config.calibration_lr:.1e}  "
          f"Patience: {config.calibration_patience}")

    model = model.to(device)

    # Disable match-boost for calibration
    saved_boost_lambda = model.match_boost_lambda
    model.match_boost_lambda = 0.0

    # Use ALL pairs (within + between)
    cal_train, cal_val, _, _ = make_dataloaders(
        pairs, site_data,
        val_fraction=config.val_fraction, batch_size=config.batch_size,
        seed=config.seed, use_unit_weights=False,
        use_design_weights=getattr(config, 'use_design_weights', True),
    )

    # Small LR, all parameters unfrozen
    optimizer = Adam(model.parameters(), lr=config.calibration_lr,
                     weight_decay=config.weight_decay)

    state = TrainingState()
    cal_ckpt_path = config.output_dir / "best_model_calibrated.pt"

    # Progress log
    log_path = config.output_dir / "training_progress_calibration.log"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "train_loss", "val_loss", "train_bce", "val_bce",
    ])
    log_writer.writeheader()

    patience_ctr = 0

    Z = site_data.Z.to(device)
    S_obs = site_data.S_obs.to(device)
    W = site_data.W.to(device) if site_data.W is not None else None
    anchor_richness = site_data.anchor_richness.to(device) if site_data.anchor_richness.any() else None

    for epoch in range(1, config.calibration_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        n_batches = 0

        for batch in cal_train:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            z_i = Z[batch["site_i"]]
            z_j = Z[batch["site_j"]]
            w_i = W[batch["site_i"]] if W is not None else None
            w_j = W[batch["site_j"]] if W is not None else None

            result = model.compute_loss(
                batch, z_i, z_j, S_obs,
                w_i=w_i, w_j=w_j,
                anchor_richness=anchor_richness,
            )
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += result["bce_loss"].item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_bce = epoch_bce / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_bce = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in cal_val:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                z_i = Z[batch["site_i"]]
                z_j = Z[batch["site_j"]]
                w_i = W[batch["site_i"]] if W is not None else None
                w_j = W[batch["site_j"]] if W is not None else None
                result = model.compute_loss(
                    batch, z_i, z_j, S_obs,
                    w_i=w_i, w_j=w_j,
                    anchor_richness=anchor_richness,
                )
                val_loss += result["loss"].item()
                val_bce += result["bce_loss"].item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_bce = val_bce / max(n_val, 1)

        log_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{avg_train_loss:.6f}",
            "val_loss": f"{avg_val_loss:.6f}",
            "train_bce": f"{avg_train_bce:.6f}",
            "val_bce": f"{avg_val_bce:.6f}",
        })
        log_file.flush()

        if epoch % 10 == 1 or epoch == config.calibration_epochs:
            print(f"  Cal epoch {epoch:3d}: loss={avg_train_loss:.6f}  "
                  f"val={avg_val_loss:.6f}  bce={avg_train_bce:.6f}")

        # Early stopping
        if avg_val_loss < state.best_val_loss:
            state.best_val_loss = avg_val_loss
            state.best_epoch = epoch
            patience_ctr = 0
            _save_checkpoint(model, optimizer, epoch, avg_val_loss,
                             site_data, config, cal_ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= config.calibration_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    log_file.close()

    # Restore best calibrated model
    if cal_ckpt_path.exists():
        ckpt = torch.load(cal_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Restore boost lambda
    model.match_boost_lambda = saved_boost_lambda

    print(f"\n  Calibration complete: best_val_loss = {state.best_val_loss:.6f} "
          f"(epoch {state.best_epoch})")

    return state
