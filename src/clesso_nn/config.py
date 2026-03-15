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
    beta_no_intercept: bool = True  # if True, remove bias from first MonotoneLinear in additive/factored nets
    beta_lr_mult: float = 10.0  # LR multiplier for beta network (relative to base LR)
    beta_grad_scale: float = 100.0  # gradient amplification for beta params (via hooks)
    include_geo_in_beta: bool = False  # OLD: include raw lon/lat diffs in beta (deprecated)
    include_geo_dist_in_beta: bool = False  # NEW: include haversine distance (km) in beta
    exclude_coords_from_alpha: bool = False  # exclude raw lon/lat from alpha covariates

    # Fourier positional encoding for alpha model
    fourier_n_frequencies: int = 0       # number of frequency octaves (0 = disabled)
    fourier_max_wavelength: float = 10.0 # degrees; wavelengths = max/2^k for k=0..N-1

    # ------------------------------------------------------------------
    # Effort / detectability network (additive decomposition in alpha)
    # ------------------------------------------------------------------
    # When effort_cov_names is non-empty, a separate EffortNet is created:
    #   α = softplus(EnvNet(env) + EffortNet(effort)) + 1
    # This cleanly separates observation-process artifacts from true richness.
    # At prediction time, the effort component can be zeroed to obtain "true richness".
    #
    # Note: for future exploration, a multiplicative decomposition
    #   α_eff = α_true / detection_prob 
    # could replace the additive form. That would require EffortNet to output
    # a sigmoid-bounded detection probability rather than a raw logit offset.
    effort_cov_names: list[str] = field(default_factory=list)  # column names in site_covariates (empty = disabled)
    effort_hidden: list[int] = field(default_factory=lambda: [64, 32])
    effort_dropout: float = 0.1
    effort_penalty: float = 0.0  # L2 penalty on all effort_net parameters (0 = disabled)

    # Path to effort raster directory (for surface prediction)
    effort_raster_dir: Optional[Path] = None

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
    hard_mining_lambda: float = 0.0         # mining strength / mixing proportion (0 = off)
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
    finetune_freeze_effort: bool = False     # freeze EffortNet params in Phase 2 (False = keep training)

    # Geographic parameter L2 penalty (separate from weight_decay).
    # Penalises geo-related parameters strongly so the model only uses
    # spatial information when site-level covariates cannot explain the
    # pattern.  Applied as: lambda * sum(param^2) added to the loss.
    #   geo_penalty_alpha: L2 on alpha-net weights connecting to Fourier columns
    #   geo_penalty_beta:  L2 on beta-net geo dim_net / encoder parameters
    # Set to 0.0 to disable.
    geo_penalty_alpha: float = 0.2    # strong default — spatial alpha is expensive
    geo_penalty_beta: float = 1.0     # strong default — spatial beta is expensive

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
