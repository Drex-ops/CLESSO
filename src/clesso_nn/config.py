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


# ======================================================================
# Path profiles — switch via  CLESSO_PROFILE  env var ("local" | "remote")
# ======================================================================
PATH_PROFILES: dict[str, dict] = {
    # Mac / local development machine
    "local": {
        "export_dir": Path(
            "/Users/andrewhoskins/Library/CloudStorage/OneDrive-NAILSMA"
            "/CODE/TERN-biodiversity-index/src/clesso_v2/output"
            "/VAS_20260316_181756/nn_export"
        ),
        "effort_search_paths": [
            str(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"
                "/CODE/Effort_data_preper/outputs"),
            "data/effort",
        ],
        "richness_anchor_path": Path(
            str(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"
                "/CODE/Richness_anchor_clesso/output/richness_anchor_lower.tif")
        ),
        "climate_npy_dir": "/Volumes/PortableSSD/CLIMATE/geonpy",
        "device": "mps",
    },
    # Windows / remote compute machine
    "remote": {
        "export_dir": None,                   # TODO: fill in Windows path to nn_export directory
        "effort_search_paths": [
            None,                             # TODO: fill in Windows path to Effort_data_preper/outputs
            "data/effort",
        ],
        "richness_anchor_path": None,         # TODO: fill in Windows path to richness_anchor_lower.tif
        "climate_npy_dir": None,              # TODO: fill in Windows path to CLIMATE/geonpy
        "device": "auto",
    },
}

ACTIVE_PROFILE = PATH_PROFILES[os.environ.get("CLESSO_PROFILE", "local")]
# ======================================================================


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
    run_id: str = _env("CLESSO_RUN_ID", "VAS_hexBalance_nn")

    # ------------------------------------------------------------------
    # Data (exported from R pipeline Steps 1-4)
    # ------------------------------------------------------------------
    export_dir: Optional[Path] = field(default_factory=lambda: ACTIVE_PROFILE["export_dir"])  # set after R export; contains feather files

    # ------------------------------------------------------------------
    # Alpha network architecture
    # ------------------------------------------------------------------
    alpha_hidden: list[int] = field(default_factory=lambda: [64,32]) #[256, 128, 64, 32])
    alpha_dropout: float = 0.1
    alpha_activation: str = "silu"  # "relu", "gelu", "silu"

    # ------------------------------------------------------------------
    # Beta (turnover) network architecture -- monotone network
    # ------------------------------------------------------------------
    beta_type: str = "transform"  # "additive" | "factored" | "transform" | "deep"
    beta_hidden: list[int] = field(default_factory=lambda: [128, 64, 32])  # only for beta_type="deep"
    beta_n_knots: int = 32       # per-dimension knots for beta_type="additive"
    beta_dropout: float = 0.1
    beta_no_intercept: bool = True  # if True, remove bias from first MonotoneLinear in additive/factored/transform nets
    transform_n_knots: int = 32  # hidden units in T_k transform nets (beta_type="transform")
    transform_g_knots: int = 16  # hidden units in g_k weighting nets (beta_type="transform")
    beta_lr_mult: float = 50.0  # LR multiplier for beta network (relative to base LR)
    beta_grad_scale: float = 100.0  # gradient amplification for beta params (via hooks; sigmoid provides healthy grads)
    include_geo_in_beta: bool = False  # OLD: include raw lon/lat diffs in beta (deprecated)
    include_geo_dist_in_beta: bool = False  # NEW: include haversine distance (km) in beta
    exclude_coords_from_alpha: bool = True  # exclude raw lon/lat from alpha covariates

    # Fourier positional encoding for alpha model
    fourier_n_frequencies: int = 0       # number of frequency octaves (0 = disabled)
    fourier_max_wavelength: float = 10.0 # degrees; wavelengths = max/2^k for k=0..N-1

    # ------------------------------------------------------------------
    # Effort / detectability network (additive decomposition in alpha)
    # ------------------------------------------------------------------
    # Master switch: set use_effort=False to disable the EffortNet even
    # when the dataset contains effort covariates.
    use_effort: bool = True

    # When effort_cov_names is non-empty, a separate EffortNet is created.
    # effort_mode controls how the effort offset combines with the env logit:
    #
    #   "additive":       α = softplus(env_logit + effort_logit) + 1
    #       Effort adds/subtracts a fixed number of apparent species.
    #
    #   "multiplicative": α_obs = (softplus(env_logit) + 1) · σ(effort_logit)
    #       Effort acts as a general (0, 1) modifier on env-predicted richness.
    #       NOTE: can push α below 1 for very low capture fractions.
    #
    #   "completeness" (default):
    #       α_true = 1 + softplus(env_logit)
    #       c_i    = σ(effort_logit)   ∈ (0, 1)   (completeness fraction)
    #       α_obs  = 1 + c_i · softplus(env_logit)
    #                = 1 + c_i · (α_true − 1)
    #       Effort can only *reduce* apparent richness below true richness.
    #       α_obs ≥ 1 is guaranteed.  More constrained than multiplicative:
    #       effort is a pure capture/completeness fraction, not a general
    #       multiplicative modifier.
    #
    # At prediction time, the effort component can be zeroed to obtain
    # "true richness".
    effort_mode: str = "completeness"  # "additive", "multiplicative", or "completeness"
    effort_cov_names: list[str] = field(default_factory=list)  # column names in site_covariates (empty = disabled)
    # Drop these effort covariates before building EffortNet.
    # Endogenous covariates (e.g. record count) are confounded with
    # apparent richness and can defeat the effort/env decomposition.
    effort_drop_cov_names: list[str] = field(default_factory=lambda: [
        "ala_record_count", "ala_record_smoothed",
    ])
    effort_hidden: list[int] = field(default_factory=lambda: [64, 32])
    effort_dropout: float = 0.1
    effort_penalty: float = 0.0  # L2 penalty on all effort_net parameters (0 = disabled)
    effort_lr: Optional[float] = None           # learning rate for EffortNet (None → 3× learning_rate)
    effort_weight_decay: Optional[float] = None  # weight decay for EffortNet (None → inherits weight_decay)

    # Path to effort INPUT rasters (the raw .flt/.hdr files used for
    # surface prediction and training-data extraction).  These are READ,
    # never written.  Output surfaces go to the model output directory.
    # If None, auto-detected from the search paths below.
    effort_input_dir: Optional[Path] = None

    # Default search paths for effort INPUT rasters (tried in order when
    # effort_input_dir is None and the model has effort features).
    _effort_input_search_paths: list[str] = field(
        default_factory=lambda: ACTIVE_PROFILE["effort_search_paths"]
    )

    # Eta smoothness penalty: lambda * mean(eta^2) on between-site pairs.
    # Penalises large eta to encourage gradual turnover and prevent the
    # degenerate solution where eta→∞ for all pairs (predicting p≈0 everywhere).
    # Acts as a restoring force that keeps eta near zero unless the data
    # provides strong evidence for high turnover.
    # DISABLED by default: the retrospective correction + anti-collapse
    # regularizer provide a more principled solution.  Set > 0 to re-enable.
    eta_smoothness_lambda: float = 0.0

    # Eta anti-collapse regularizer: -lambda * mean(log(eta + eps)) on
    # between-site pairs.  Applies a logarithmic barrier that strongly
    # resists eta → 0 while exerting negligible force at moderate eta.
    # Acts as a safety net alongside the retrospective correction.
    # Recommended: 0.01–0.1.  Set to 0.0 to disable.
    eta_anti_collapse_lambda: float = 0.0

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

    # ------------------------------------------------------------------
    # Composite likelihood weighting
    # ------------------------------------------------------------------
    # Stratum weights λ_s for the composite loss:
    #   stratum 0 = within-site (W)
    #   stratum 1 = same-hex between-site (S)
    #   stratum 2 = neighbour-hex between-site (N)
    #   stratum 3 = distant-hex between-site (D)
    # Match-boost pairs (stratum 4) are handled separately via match_boost_lambda.
    stratum_weights: list[float] = field(
        default_factory=lambda: [0.25, 0.35, 0.25, 0.15]
    )

    # Auxiliary match-boost shaping loss (stratum 4).
    # This λ controls the weight of match-boosted pairs as an auxiliary BCE
    # that helps the model see rare matches without distorting the main loss.
    match_boost_lambda: float = 0.1

    # Drop match-boost pairs (stratum 4) entirely before training.
    # When True, stratum-4 pairs are removed from the pairs DataFrame
    # in preprocessing so they never enter any dataloader or loss.
    exclude_match_boost: bool = True

    # Use design weights from the sampler (True) or unit weights (False).
    # Design weights correct for unequal selection probabilities in the
    # hex-balanced sampler.  If False, all design_w are set to 1.0.
    use_design_weights: bool = True

    # Weight stabilisation (applied after combining pop + alpha + mining weights)
    # Clamp: cap extreme weights at clamp_factor × batch median (0 = disabled)
    weight_clamp_factor: float = 0.0
    # Log-normalise: shift weights into log-space, softmax back to positive
    # weights. Compresses heavy tails but changes the weighting scheme.
    weight_log_normalise: bool = False

    # Soft alpha lower-bound penalty on α_env (effort-stripped true richness):
    #   λ · mean[ softplus(κ · (log S_obs − log α_env)) / κ ]²
    # Ensures predicted true richness ≥ observed richness.  Applied to α_env
    # (not α_obs), so EffortNet can freely explain the observation gap.
    # Note: S_obs should be the TRUE observed richness from raw data, not the
    # subsampled pair-source richness. Use site_obs_richness_CORRECTED.feather.
    # Set alpha_lower_bound_lambda = 0.0 to disable the penalty entirely.
    alpha_lower_bound_lambda: float = 1.0  # 0.0 = disabled

    # Alpha anchor penalty (log-normal prior on α_env):
    # Penalises α_env (effort-stripped true richness) that drifts far from S_obs.
    # Works in log-space with a dead-zone tolerance:
    #   penalty = λ * mean[ max(0, |log(α_env) - log(S_obs)| - τ)² ]
    # where τ = alpha_anchor_tolerance (log-scale slack).
    #
    # Applied to α_env (not α_obs), so EffortNet can independently explain
    # the observation gap.  An undersampled site with S_obs=20 can have
    # α_env up to exp(τ)×20 ≈ 40 (for τ≈0.7) without penalty.
    # Beyond that, the quadratic penalty pushes back.
    #
    # This prevents the runaway α inflation that kills beta gradients
    # and AUC in block-coordinate training.
    # 
    # 0.7 is a reasonable default that allows α_env up to ~2× S_obs without penalty, which is sufficient slack for moderate undersampling while still penalising extreme inflation.
    # 1.1 is a looser tolerance that allows α_env up to ~3× S_obs without penalty, which may be needed for very undersampled datasets but provides less regularisation against inflation.
    # 1.6 is a very loose tolerance that allows α_env up to ~5× S_obs without penalty, which may be necessary for extremely undersampled datasets but provides minimal regularisation against inflation.
    # 0.0 is a hard anchor that penalises any deviation from S_obs, which may be too strict given the uncertainty in S_obs and the flexibility of EffortNet.
    #
    # Set alpha_anchor_lambda = 0.0 to disable the penalty entirely.
    alpha_anchor_lambda: float = 0.0   # 0.0 = disabled
    alpha_anchor_tolerance: float = 1.1 # log-space slack (e.g. 0.7 → ~2× S_obs, 1.1 → ~3× S_obs)

    # Direct alpha regression: lambda * MSE(alpha, S_obs)
    # DISABLED (0.0) — this was pulling alpha toward the incorrect S_obs values.
    # With corrected S_obs, the lower-bound penalty alone is sufficient.
    alpha_regression_lambda: float = 0.0

    # Richness anchor (auxiliary raster-based soft constraint on α_env):
    # An external raster provides expected richness values at certain
    # locations.  The model is softly pushed toward these values via:
    #   penalty = λ * mean[ (log(α_env) - log(anchor))² ]
    # Only applied where the raster has valid data (pixel value > 1;
    # pixels with value 0 are treated as no-data).
    # Set richness_anchor_lambda = 0.0 or richness_anchor_path = None to disable.
    richness_anchor_path: Optional[Path] = field(default_factory=lambda: ACTIVE_PROFILE["richness_anchor_path"])
    richness_anchor_lambda: float = 1.0  # 0.0 = disabled

    # ------------------------------------------------------------------
    # Two-stage training
    # ------------------------------------------------------------------
    # "joint" = original single-stage (alpha+beta together, prone to beta collapse)
    # "two_stage" = Stage 1: alpha-only on within-site, Stage 2: beta-only on between-site
    # "cyclic" = Alternating block-coordinate descent: cycle alpha→beta→alpha→... until converged
    # "cyclic_finetune" = Phase 1: damped cyclic on env-only, Phase 2: fine-tune with geo features
    training_mode: str = "cyclic_finetune"  # "joint" | "two_stage" | "cyclic" | "cyclic_finetune"

    stage1_max_epochs: int = 300
    stage1_patience: int = 30

    stage2_max_epochs: int = 300
    stage2_patience: int = 40
    stage2_beta_grad_scale: float = 1.0  # sigmoid provides healthy gradients; no amplification needed

    # ------------------------------------------------------------------
    # Cyclic (block-coordinate descent) training
    # ------------------------------------------------------------------
    # Damped approach: small alpha step → 2× beta steps → repeat.
    # Keeps alpha changes incremental so beta can adapt without oscillation.
    max_cycles: int = 20
    cycle_alpha_epochs: int = 5    # small alpha step per cycle
    cycle_beta_epochs: int = 20    # more beta steps per cycle to adapt to alpha shift
    cycle_beta_patience: int = 5   # within-phase patience: stop beta early if no val_loss improvement
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
    # finetune_mode controls what Phase 2 does after cyclic Phase 1:
    #   "geo"   = add geographic features (Fourier → alpha, haversine → beta)
    #             and fine-tune with differential LR (original behaviour)
    #   "joint" = joint alpha+beta training on the SAME env/effort variables
    #             used in Phase 1 — no model expansion, all params trainable
    finetune_mode: str = "joint"      # "geo" | "joint"
    finetune_max_epochs: int = 100
    finetune_lr: float = 5e-4         # base LR for existing (env) params
    finetune_new_param_lr_mult: float = 5.0  # LR multiplier for new geo params (geo mode only)
    finetune_patience: int = 30       # early stopping patience
    finetune_freeze_existing: bool = True  # freeze Phase 1 params; only train new geo params (geo mode)
    finetune_freeze_effort: bool = False     # freeze EffortNet params in Phase 2 (False = keep training)
    # Joint finetune uses these LR multipliers relative to finetune_lr:
    finetune_joint_beta_lr_mult: float = 10.0  # beta gets bigger LR (relative to finetune_lr)
    finetune_joint_alpha_lr_mult: float = 1.0  # alpha gets base finetune_lr

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
    # Calibration phase (final phase after main training)
    # ------------------------------------------------------------------
    # Short phase with design-weighted loss only (no mining, no boost),
    # small LR, to de-bias probability estimates after quota sampling.
    calibration_epochs: int = 50
    calibration_lr: float = 1e-4
    calibration_patience: int = 20

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: str = field(default_factory=lambda: ACTIVE_PROFILE["device"])  # "auto", "cpu", "mps", "cuda"

    def resolve_device(self) -> str:
        import torch
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def resolved_effort_lr(self) -> float:
        """EffortNet learning rate: explicit value or 3× base LR."""
        if self.effort_lr is not None:
            return self.effort_lr
        return 3.0 * self.learning_rate

    @property
    def resolved_effort_wd(self) -> float:
        """EffortNet weight decay: explicit value or inherits base weight_decay."""
        if self.effort_weight_decay is not None:
            return self.effort_weight_decay
        return self.weight_decay

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_every: int = 10  # print training stats every N batches
    progress_log: bool = True  # write per-epoch log to output_dir
