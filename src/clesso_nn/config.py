"""
config.py -- Default configuration for CLESSO neural-network pipeline.

Mirrors the key settings from clesso_v2/clesso_config.R but in Python.
Override by editing this file or passing a dict to the training functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env(var: str, default: str) -> str:
    return os.environ.get(var, default)


@dataclass
class CLESSONNConfig:
    """Configuration for the CLESSO neural-network model."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root: Path = field(default_factory=lambda: Path(
        _env("CLESSO_PROJECT_ROOT",
             str(Path(__file__).resolve().parents[2]))
    ))

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def output_dir(self) -> Path:
        d = self.project_root / "src" / "clesso_nn" / "output" / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Run identification
    # ------------------------------------------------------------------
    species_group: str = _env("CLESSO_SPECIES_GROUP", "VAS")
    run_id: str = _env("CLESSO_RUN_ID", "VAS_nn")

    # ------------------------------------------------------------------
    # Data (exported from R pipeline Steps 1-4)
    # ------------------------------------------------------------------
    export_dir: Optional[Path] = None  # set after R export; contains feather files

    # ------------------------------------------------------------------
    # Alpha network architecture
    # ------------------------------------------------------------------
    alpha_hidden: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    alpha_dropout: float = 0.1
    alpha_activation: str = "silu"  # "relu", "gelu", "silu"

    # ------------------------------------------------------------------
    # Beta (turnover) network architecture -- monotone network
    # ------------------------------------------------------------------
    beta_type: str = "factored"  # "additive" | "factored" (per-dim + interactions) | "deep"
    beta_hidden: list[int] = field(default_factory=lambda: [128, 64, 32])  # only for beta_type="deep"
    beta_n_knots: int = 32       # per-dimension knots for beta_type="additive"
    beta_dropout: float = 0.1
    beta_lr_mult: float = 10.0  # LR multiplier for beta network (relative to base LR)
    beta_grad_scale: float = 100.0  # gradient amplification for beta params (via hooks)
    include_geo_in_beta: bool = False  # OLD: include raw lon/lat diffs in beta (deprecated)
    include_geo_dist_in_beta: bool = False  # NEW: include haversine distance (km) in beta
    exclude_coords_from_alpha: bool = False  # exclude raw lon/lat from alpha covariates

    # Fourier positional encoding for alpha model
    fourier_n_frequencies: int = 0       # number of frequency octaves (0 = disabled)
    fourier_max_wavelength: float = 40.0 # degrees; wavelengths = max/2^k for k=0..N-1

    # Eta smoothness penalty: lambda * mean(eta^2) on between-site pairs.
    # Penalises large eta to encourage gradual turnover.
    # Set to 0.0 (disabled) — the softplus activation in the beta net now
    # provides gradual onset naturally without needing this penalty.
    eta_smoothness_lambda: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 500
    patience: int = 40          # early stopping patience (epochs without improvement)
    val_fraction: float = 0.1   # fraction of pairs held out for validation
    seed: int = 42
    use_unit_weights: bool = True  # ignore stored weights; use w=1 (natural loss)

    # Soft alpha lower-bound penalty: lambda * sum [softplus(S_obs - alpha)]^2
    # Note: S_obs should be the TRUE observed richness from raw data, not the
    # subsampled pair-source richness. Use site_obs_richness_CORRECTED.feather.
    alpha_lower_bound_lambda: float = 1.0

    # Direct alpha regression: lambda * MSE(alpha, S_obs)
    # DISABLED (0.0) — this was pulling alpha toward the incorrect S_obs values.
    # With corrected S_obs, the lower-bound penalty alone is sufficient.
    alpha_regression_lambda: float = 0.0

    # ------------------------------------------------------------------
    # Two-stage training
    # ------------------------------------------------------------------
    # "joint" = original single-stage (alpha+beta together, prone to beta collapse)
    # "two_stage" = Stage 1: alpha-only on within-site, Stage 2: beta-only on between-site
    # "cyclic" = Alternating block-coordinate descent: cycle alpha→beta→alpha→... until converged
    # "cyclic_finetune" = Phase 1: damped cyclic on env-only, Phase 2: fine-tune with geo features
    training_mode: str = "cyclic"

    stage1_max_epochs: int = 300
    stage1_patience: int = 30

    stage2_max_epochs: int = 300
    stage2_patience: int = 40
    stage2_beta_grad_scale: float = 10.0  # less aggressive than joint (importance weights handle balance)

    # ------------------------------------------------------------------
    # Cyclic (block-coordinate descent) training
    # ------------------------------------------------------------------
    # Damped approach: small alpha step → 2× beta steps → repeat.
    # Keeps alpha changes incremental so beta can adapt without oscillation.
    max_cycles: int = 100
    cycle_alpha_epochs: int = 5    # small alpha step per cycle
    cycle_beta_epochs: int = 10    # 2× beta steps per cycle
    cycle_tol: float = 1e-4        # relative change convergence threshold

    # ------------------------------------------------------------------
    # Hard-pair mining (between-site pairs only)
    # ------------------------------------------------------------------
    # Importance-corrected hard-pair mining for beta training.
    # Mining strength λ_hm ∈ [0,1]: 0 = disabled (default), 1 = pure hard mining.
    # q(p) = (1-λ_hm)·π(p) + λ_hm·h*(p)  with importance correction w* = w·π(p)/q(p).
    hard_mining_lambda: float = 0.3         # mining strength / mixing proportion (0 = off)
    hard_mining_warmup_cycles: int = 3      # cycles before enabling mining
    hard_mining_n_bins: int = 3             # difficulty bins (e.g. easy/medium/hard)
    hard_mining_bin_weights: list[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.5],  # mass per bin (easy→hard)
    )
    hard_mining_a_max: float = 5.0          # truncation cap for π(p)/q(p)
    hard_mining_refresh_every: int = 1      # re-score pairs every N beta epochs within a phase

    # ------------------------------------------------------------------
    # Fine-tuning phase (Phase 2 of cyclic_finetune)
    # ------------------------------------------------------------------
    # After Phase 1 (damped cyclic on env-only), Phase 2 adds geographic
    # features (Fourier harmonics → alpha, haversine geo_dist → beta)
    # and fine-tunes with a smaller learning rate.
    finetune_max_epochs: int = 100
    finetune_lr: float = 5e-4         # base LR for existing (env) params
    finetune_new_param_lr_mult: float = 5.0  # LR multiplier for new geo params
    finetune_patience: int = 30       # early stopping patience
    finetune_freeze_existing: bool = True  # freeze Phase 1 params; only train new geo params

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: str = "mps"  # "auto", "cpu", "mps", "cuda"

    def resolve_device(self) -> str:
        import torch
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_every: int = 10  # print training stats every N batches
    progress_log: bool = True  # write per-epoch log to output_dir
