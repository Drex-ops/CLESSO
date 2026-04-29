"""
model.py -- Neural network architecture for CLESSO.

Three components:
  1. AlphaNet:         site covariates → richness (α > 1)
  2. Beta network:     env site values → turnover η ≥ 0
                       (TransformBetaNet or FactoredDeepBetaNet)
  3. CLESSONet:        combines both, computes match probability and loss

The beta network enforces monotone non-decreasing response in environmental
distance — the same constraint achieved by I-splines + non-negative
coefficients in the TMB model, but with greater flexibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _sigmoid_clamp(x: torch.Tensor, shift: torch.Tensor,
                   max_val: float = 10.0) -> torch.Tensor:
    """Smooth bounded output: maps ℝ → (0, max_val) via shifted sigmoid.

    Uses  max_val · σ(x − shift)  which:
      - Has gradient  max_val · σ · (1−σ) > 0  everywhere.
      - Peak gradient at σ=0.5 (x=shift), tapering symmetrically.
      - Even at η = 0.99·max_val, gradient ≈ 0.1   (vs tanh ≈ 0.0002).
      - Even at η = 0.999·max_val, gradient ≈ 0.01  (still trainable).

    ``shift`` is a learnable parameter controlling the midpoint; initialised
    so η starts near 1–2 (ecological mid-range, far from both boundaries).

    Gradient comparison at η=9 (out of 10):
      hard clamp:  0.0  (dead)
      tanh clamp:  0.02 (dying)
      sigmoid:     0.9  (healthy)
    """
    return max_val * torch.sigmoid(x - shift)


# --------------------------------------------------------------------------
# Alpha network: site covariates → α (species richness)
# --------------------------------------------------------------------------

class AlphaNet(nn.Module):
    """MLP mapping site-level covariates to species richness α > 1.

    Architecture:
        input (K_alpha) → [Linear→Act→Dropout] × L → Linear(1)
        output = softplus(raw) + 1   (ensures α > 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (64, 32, 16),
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), act_fn(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialise the output layer bias so that softplus(bias) + 1 ≈ target.
        # For vascular plants with median S_obs ≈ 3, we want alpha ≈ 3.
        # softplus(1.1) ≈ 1.74, + 1 = 2.74.
        # This is a gentle starting point; the model learns site-specific values.
        output_layer = self.net[-1]  # last Linear
        with torch.no_grad():
            output_layer.bias.fill_(1.1)  # softplus(1.1) + 1 ≈ 2.7

    def logit(self, z: torch.Tensor) -> torch.Tensor:
        """Return raw (pre-softplus) logit for additive decomposition.

        Args:
            z: (batch, K_alpha) standardised site covariates
        Returns:
            raw: (batch,) unbounded logit
        """
        return self.net(z).squeeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, K_alpha) standardised site covariates
        Returns:
            alpha: (batch,) positive richness values > 1
        """
        raw = self.logit(z)
        # softplus ensures positivity; +1 ensures alpha > 1
        return F.softplus(raw) + 1.0


# --------------------------------------------------------------------------
# Effort / detectability network (additive offset in alpha)
# --------------------------------------------------------------------------

class EffortNet(nn.Module):
    """Small MLP mapping effort/detectability covariates to a raw logit offset.

    Used in additive decomposition:
        α = softplus(AlphaNet.logit(env) + EffortNet(effort)) + 1

    Architecture:
        input (K_effort) → [Linear→SiLU→Dropout] × L → Linear(1)
        output = raw logit (no activation — added to env logit before softplus)

    Output bias initialised to 0 so effort starts neutral (env carries
    the initial prediction).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Output bias = 0 → effort starts neutral
        # (softplus(env_logit + 0) = softplus(env_logit))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Args:
            w: (batch, K_effort) standardised effort covariates
        Returns:
            offset: (batch,) raw logit offset
        """
        return self.net(w).squeeze(-1)


# --------------------------------------------------------------------------
# Monotone beta network: |env_diff| → η ≥ 0  (compositional turnover)
# --------------------------------------------------------------------------

class MonotoneLinear(nn.Module):
    """Linear layer with non-negative weights (monotone w.r.t. inputs).

    Weights are stored as unconstrained parameters and passed through
    softplus to enforce w ≥ 0. This means:
        output = softplus(W_raw) @ input + bias
    guaranteeing that output is non-decreasing in every input dimension
    (when composed with monotone activations and preceding monotone layers).

    Reference: Sill (1997), Daniels & Velikova (2010), monotone network literature.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._init_weights()

    def _init_weights(self):
        # Initialise so that softplus(w_raw) ≈ small positive value
        nn.init.normal_(self.weight_raw, mean=0.5, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.weight_raw)  # (out, in), all ≥ 0
        out = F.linear(x, w, self.bias)
        return out


class TransformBetaNet(nn.Module):
    """Transform-first beta network: learn T_k, then take distances.

    This is the closest neural analog of GDM's I-spline structure.
    For each covariate k, learn a monotone 1D transform T_k(x), then define

        η = sigmoid_clamp(Σ_k g_k(|T_k(x_{k,i}) - T_k(x_{k,j})|), shift, 10)

    where T_k is a monotone per-dimension transform and g_k is a monotone
    per-dimension weighting network (both using MonotoneLinear + Softplus).

    Key difference from older additive variants:
        Additive (now removed): f_k(|x_i - x_j|)     — learns on pre-computed distances
        TransformBetaNet:  g_k(|T_k(x_i) - T_k(x_j)|) — learns transform first

    The transform-first approach lets the network discover that e.g. the
    difference between 10°C and 15°C matters more than between 25°C and 30°C,
    which is exactly what GDM's I-splines achieve.

    Monotonicity guarantee:
        - T_k is monotone (MonotoneLinear + Softplus composition)
        - |T_k(x_i) - T_k(x_j)| is non-negative
        - If x_i and x_j move further apart in the raw env space, T_k being
          monotone means |T_k(x_i) - T_k(x_j)| ≥ previous value
        - g_k is monotone and origin-anchored (g_k(0) = 0)
        - The sum of non-negative monotone terms is monotone and non-negative
    """

    def __init__(
        self,
        input_dim: int,
        n_transform_knots: int = 32,
        n_g_knots: int = 16,
        dropout: float = 0.1,
        no_intercept: bool = False,
        n_geo_dims: int = 0,
    ):
        super().__init__()
        self.K_env = input_dim
        self.n_transform_knots = n_transform_knots
        self.n_g_knots = n_g_knots
        self.no_intercept = no_intercept
        self.n_geo_dims = n_geo_dims
        self.eta_max = 10.0

        # Learnable shift for sigmoid output
        self.sigmoid_shift = nn.Parameter(torch.tensor(1.75))

        # Per-dimension monotone transform: raw env scalar → transformed scalar
        # T_k: scalar → n_transform_knots → 1
        self.transform_nets = nn.ModuleList()
        for _ in range(input_dim):
            self.transform_nets.append(nn.Sequential(
                MonotoneLinear(1, n_transform_knots, bias=True),
                nn.Softplus(),
                nn.Dropout(dropout),
                MonotoneLinear(n_transform_knots, 1, bias=True),
            ))

        # Per-dimension monotone weighting: |T_k(x_i) - T_k(x_j)| → contribution
        # g_k: scalar → n_g_knots → 1, origin-anchored via baseline subtraction
        self.weight_nets = nn.ModuleList()
        for _ in range(input_dim):
            self.weight_nets.append(nn.Sequential(
                MonotoneLinear(1, n_g_knots, bias=not no_intercept),
                nn.Softplus(),
                nn.Dropout(dropout),
                MonotoneLinear(n_g_knots, 1, bias=False),
            ))

        # Optional geo distance sub-nets (haversine km, already a scalar distance)
        # These use the same g_k-style monotone network directly on distance.
        self.geo_nets = nn.ModuleList()
        for _ in range(n_geo_dims):
            self.geo_nets.append(nn.Sequential(
                MonotoneLinear(1, n_g_knots, bias=not no_intercept),
                nn.Softplus(),
                nn.Dropout(dropout),
                MonotoneLinear(n_g_knots, 1, bias=False),
            ))

        self._init_weights()
        self._calibrate_sigmoid_shift()

    def _init_weights(self):
        """Initialise transform and weighting networks."""
        # Transform nets: moderate init so T_k starts near-linear
        for net in self.transform_nets:
            first_layer = True
            for m in net.modules():
                if isinstance(m, MonotoneLinear):
                    if first_layer:
                        nn.init.uniform_(m.weight_raw, -0.5, 0.5)
                        if m.bias is not None:
                            n = m.bias.shape[0]
                            m.bias.data.copy_(torch.linspace(-3.0, 0.0, n))
                        first_layer = False
                    else:
                        # Output layer: moderate so T_k has ~unit-scale output
                        nn.init.constant_(m.weight_raw, -2.0)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        # Weight nets: small init (per-dimension monotone weighting)
        for net_list in [self.weight_nets, self.geo_nets]:
            for net in net_list:
                first_layer = True
                for m in net.modules():
                    if isinstance(m, MonotoneLinear):
                        if first_layer:
                            nn.init.uniform_(m.weight_raw, -0.5, 0.5)
                            if m.bias is not None:
                                n = m.bias.shape[0]
                                m.bias.data.copy_(torch.linspace(-3.0, 0.0, n))
                            first_layer = False
                        else:
                            nn.init.constant_(m.weight_raw, -3.0)

    def _calibrate_sigmoid_shift(self, target_eta: float = 1.5):
        """Set sigmoid_shift so η ≈ target_eta for typical standardised input.

        Simulates a pair of sites 0.5σ apart in each env dimension.
        """
        with torch.no_grad():
            # Simulate site_i at +0.25σ and site_j at -0.25σ (total diff 0.5σ)
            env_i = torch.ones(64, self.K_env) * 0.25
            env_j = torch.ones(64, self.K_env) * -0.25
            total = torch.zeros(64)
            zero = torch.zeros(1, 1)
            for k in range(self.K_env):
                t_i = self.transform_nets[k](env_i[:, k:k+1])
                t_j = self.transform_nets[k](env_j[:, k:k+1])
                delta_k = torch.abs(t_i - t_j)
                g_k = self.weight_nets[k](delta_k)
                g_k_0 = self.weight_nets[k](zero)
                total += (g_k - g_k_0).squeeze(-1)
            mean_raw = total.mean().item()
            t = target_eta / self.eta_max
            logit_t = math.log(t / (1.0 - t))
            self.sigmoid_shift.data.fill_(mean_raw - logit_t)

    def transform_site(self, env: torch.Tensor) -> torch.Tensor:
        """Apply per-dimension transforms T_k to raw (standardised) env values.

        Useful for spatial prediction (PCA on transformed space → RGB) and
        spline visualisation.

        Args:
            env: (N, K_env) standardised environmental covariates
        Returns:
            transformed: (N, K_env) T_k(env_k) for each dimension k
        """
        parts = []
        for k in range(self.K_env):
            t_k = self.transform_nets[k](env[:, k:k+1])  # (N, 1)
            parts.append(t_k)
        return torch.cat(parts, dim=-1)  # (N, K_env)

    def forward(
        self,
        env_i: torch.Tensor,
        env_j: torch.Tensor,
        geo_dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            env_i: (B, K_env) standardised env covariates for site i
            env_j: (B, K_env) standardised env covariates for site j
            geo_dist: (B, n_geo_dims) normalised geographic distance(s), or None
        Returns:
            eta: (B,) non-negative turnover values η ∈ (0, eta_max)
        """
        device = env_i.device
        total = torch.zeros(env_i.shape[0], device=device)
        zero = torch.zeros(1, 1, device=device)

        for k in range(self.K_env):
            # Apply monotone transform to each site's env value
            t_i = self.transform_nets[k](env_i[:, k:k+1])   # (B, 1)
            t_j = self.transform_nets[k](env_j[:, k:k+1])   # (B, 1)
            # Distance in transformed space
            delta_k = torch.abs(t_i - t_j)                   # (B, 1)
            # Monotone weighting, origin-anchored
            g_k = self.weight_nets[k](delta_k)                # (B, 1)
            g_k_0 = self.weight_nets[k](zero)                 # (1, 1)
            total = total + (g_k - g_k_0).squeeze(-1)         # (B,)

        # Optional geographic distance dimensions
        if geo_dist is not None and self.n_geo_dims > 0:
            for g in range(self.n_geo_dims):
                d_g = geo_dist[:, g:g+1]                      # (B, 1)
                h_g = self.geo_nets[g](d_g)                   # (B, 1)
                h_g_0 = self.geo_nets[g](zero)                # (1, 1)
                total = total + (h_g - h_g_0).squeeze(-1)     # (B,)

        return _sigmoid_clamp(total, self.sigmoid_shift, max_val=self.eta_max)


class FactoredDeepBetaNet(nn.Module):
    """Factored deep beta network: per-dimension monotone encoders + monotone interaction.

    Combines per-dimension specialisation with cross-dimension
    interactions, using a factored encoder + projector architecture.

    Architecture:
        1. Per-dim monotone encoder: |Δx_k| → h_k  (private, positive weights via Softplus)
        2. Per-dim monotone projection: h_k → s_k ≥ 0  (additive baseline)
        3. Monotone shared interaction: concat(h_1,...,h_K) → scalar  (monotone, positive weights)
        4. Output: η = (Σ_k s_k) + interaction - baseline, clamped to [0, 10]

    Monotonicity guarantee:
        - Each encoder is monotone in its input (positive weights + monotone activations)
        - Each s_k is monotone and ≥ 0 (via baseline subtraction)
        - The shared interaction network has positive weights (MonotoneLinear), so
          it's monotone in its inputs. Since each h_k is monotone in |Δx_k| and the
          composition of monotone functions is monotone, the interaction term is monotone.
        - The sum of monotone non-negative terms is monotone and non-negative.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden: int = 16,
        encoder_depth: int = 2,
        interaction_hidden: list[int] | None = None,
        dropout: float = 0.0,  # dropout disabled: baseline subtraction is incompatible
        no_intercept: bool = False,
    ):
        super().__init__()
        self.K_env = input_dim
        self.encoder_hidden = encoder_hidden
        self.no_intercept = no_intercept
        self.eta_max = 10.0

        # Learnable shift for sigmoid output (bounded η with healthy gradients)
        self.sigmoid_shift = nn.Parameter(torch.tensor(1.75))

        if interaction_hidden is None:
            interaction_hidden = [32]

        # --- Per-dimension monotone encoders ---
        # Each maps |Δx_k| (scalar) → h_k (encoder_hidden-dim, all ≥ 0 via ReLU)
        # No dropout: baseline subtraction requires deterministic forward pass.
        # no_intercept=True removes bias from first encoder layer.
        self.encoders = nn.ModuleList()
        for _ in range(input_dim):
            layers: list[nn.Module] = []
            in_dim = 1
            first_layer = True
            for _ in range(encoder_depth):
                out_dim = encoder_hidden
                layers.append(MonotoneLinear(in_dim, out_dim, bias=(not no_intercept) if first_layer else True))
                layers.append(nn.Softplus())
                in_dim = out_dim
                first_layer = False
            # Final layer + ReLU ensures h_k ≥ 0 element-wise
            layers.append(MonotoneLinear(in_dim, encoder_hidden))
            layers.append(nn.ReLU())
            self.encoders.append(nn.Sequential(*layers))

        # --- Per-dimension scalar projections (additive component) ---
        self.dim_projectors = nn.ModuleList([
            MonotoneLinear(encoder_hidden, 1, bias=False) for _ in range(input_dim)
        ])

        # --- Monotone shared interaction network ---
        # Takes concatenated encoder outputs (all ≥ 0), produces scalar via
        # MonotoneLinear layers (positive weights) → monotone in all inputs.
        # No dropout for same reason as encoders.
        interact_layers: list[nn.Module] = []
        in_dim = input_dim * encoder_hidden
        for h_dim in interaction_hidden:
            interact_layers.append(MonotoneLinear(in_dim, h_dim))
            interact_layers.append(nn.Softplus())
            in_dim = h_dim
        interact_layers.append(MonotoneLinear(in_dim, 1, bias=False))
        self.interaction_net = nn.Sequential(*interact_layers)

        self._init_weights()
        self._calibrate_sigmoid_shift()

    def _calibrate_sigmoid_shift(self, target_eta: float = 1.5):
        """Set sigmoid_shift so η ≈ target_eta for typical standardised input.

        Runs a no-grad forward pass with |Δx| = 0.5 (half a standard deviation)
        to measure the raw network total, then solves for the shift that maps
        that total to the desired initial η.
        """
        with torch.no_grad():
            x = torch.ones(64, self.K_env) * 0.5
            zero = torch.zeros(1, 1)

            # Additive component
            additive = torch.zeros(64)
            encodings = []
            for k in range(self.K_env):
                h_k = self.encoders[k](x[:, k:k+1])
                encodings.append(h_k)
                s_k = self.dim_projectors[k](h_k).squeeze(-1)
                s_k_0 = self.dim_projectors[k](self.encoders[k](zero)).squeeze(-1)
                additive += (s_k - s_k_0)

            # Interaction component
            concat_h = torch.cat(encodings, dim=-1)
            interaction = self.interaction_net(concat_h).squeeze(-1)
            h_zeros = [self.encoders[k](zero) for k in range(self.K_env)]
            concat_h0 = torch.cat(h_zeros, dim=-1)
            interaction_0 = self.interaction_net(concat_h0).squeeze(-1)
            interaction = interaction - interaction_0

            mean_raw = (additive + interaction).mean().item()
            t = target_eta / self.eta_max
            logit_t = math.log(t / (1.0 - t))
            self.sigmoid_shift.data.fill_(mean_raw - logit_t)

    def _init_weights(self):
        """Initialise for moderate initial eta values."""
        for enc in self.encoders:
            first_layer = True
            for m in enc.modules():
                if isinstance(m, MonotoneLinear):
                    if first_layer:
                        nn.init.uniform_(m.weight_raw, -0.5, 0.5)
                        if m.bias is not None:
                            n = m.bias.shape[0]
                            m.bias.data.copy_(torch.linspace(-3.0, 0.0, n))
                        first_layer = False
                    else:
                        nn.init.uniform_(m.weight_raw, -1.0, 0.5)
                        if m.bias is not None:
                            nn.init.uniform_(m.bias, -1.0, 0.0)

        # Projectors: very small init to compensate for multi-layer encoder
        # magnitudes.  Encoder outputs ~20 after 3 layers of compounding;
        # softplus(-6) ≈ 0.0025 → per-dim ≈ 16*0.0025*2 ≈ 0.08, total ≈ 1.2.
        for proj in self.dim_projectors:
            nn.init.constant_(proj.weight_raw, -6.0)

        # Interaction net: very small init so interaction starts near zero.
        # With K_env*encoder_hidden inputs, even small weights aggregate fast.
        # softplus(-6) ≈ 0.0025; with 240 inputs → pre-act ≈ 0.6 → manageable.
        for m in self.interaction_net.modules():
            if isinstance(m, MonotoneLinear):
                nn.init.constant_(m.weight_raw, -6.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, env_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            env_diff: (B, K_env) absolute environmental differences, all ≥ 0
        Returns:
            eta: (B,) turnover values ≥ 0, clamped to [0, 10]
        """
        device = env_diff.device

        # 1. Per-dimension encoding + additive projection (baseline-subtracted)
        encodings = []
        additive = torch.zeros(env_diff.shape[0], device=device)
        zero = torch.zeros(1, 1, device=device)

        for k in range(self.K_env):
            x_k = env_diff[:, k:k + 1]                # (B, 1)
            h_k = self.encoders[k](x_k)                # (B, encoder_hidden)
            encodings.append(h_k)

            s_k = self.dim_projectors[k](h_k).squeeze(-1)              # (B,)
            s_k_0 = self.dim_projectors[k](self.encoders[k](zero)).squeeze(-1)  # (1,)
            additive = additive + (s_k - s_k_0)

        # 2. Monotone interaction (baseline-subtracted)
        concat_h = torch.cat(encodings, dim=-1)         # (B, K_env * encoder_hidden)
        interaction = self.interaction_net(concat_h).squeeze(-1)  # (B,)

        # Baseline: value at all-zeros input
        zero_k = torch.zeros(1, 1, device=device)
        h_zeros = [self.encoders[k](zero_k) for k in range(self.K_env)]
        concat_h0 = torch.cat(h_zeros, dim=-1)          # (1, K_env * encoder_hidden)
        interaction_0 = self.interaction_net(concat_h0).squeeze(-1)  # (1,)

        interaction = interaction - interaction_0

        # 3. Combine additive + interaction
        eta = additive + interaction

        return _sigmoid_clamp(eta, self.sigmoid_shift, max_val=self.eta_max)


# --------------------------------------------------------------------------
# Model expansion for geographic fine-tuning
# --------------------------------------------------------------------------

def expand_model_for_geo(
    model_phase1: CLESSONet,
    K_alpha_new: int,
    K_env_new: int,
    config,
) -> CLESSONet:
    """Create an expanded model and transfer Phase 1 weights.

    Phase 1 was trained on env+substrate only.  Phase 2 adds:
      - Fourier positional encoding features to alpha (expands input layer)
      - Haversine geo_distance dimension to beta (adds new dim_net)

    Weight transfer strategy:
      - AlphaNet first layer: copy existing columns, zero-init new Fourier columns
        (so initial alpha output is identical to Phase 1)
      - AlphaNet subsequent layers: copy exactly
      - BetaNet existing dim_nets: copy exactly
      - BetaNet new dim_net(s): fresh initialisation (random)
    """
    # Build new model with expanded dimensions
    model_new = CLESSONet(
        K_alpha=K_alpha_new,
        K_env=K_env_new,
        alpha_hidden=config.alpha_hidden,
        alpha_dropout=config.alpha_dropout,
        beta_dropout=config.beta_dropout,
        alpha_activation=config.alpha_activation,
        alpha_lb_lambda=config.alpha_lower_bound_lambda,
        alpha_anchor_lambda=config.alpha_anchor_lambda,
        alpha_anchor_tolerance=config.alpha_anchor_tolerance,
        alpha_regression_lambda=config.alpha_regression_lambda,
        beta_type=config.beta_type,
        beta_no_intercept=config.beta_no_intercept,
        transform_n_knots=getattr(config, 'transform_n_knots', 32),
        transform_g_knots=getattr(config, 'transform_g_knots', 16),
        K_effort=model_phase1.effort_net.net[0].in_features if model_phase1.effort_net else 0,
        effort_hidden=config.effort_hidden,
        effort_dropout=config.effort_dropout,
        effort_mode=getattr(model_phase1, 'effort_mode', 'additive'),
    )
    model_new.eta_smoothness_lambda = config.eta_smoothness_lambda
    model_new.eta_anti_collapse_lambda = config.eta_anti_collapse_lambda

    K_alpha_old = model_phase1.alpha_net.net[0].in_features
    K_env_old = model_phase1.beta_net.K_env

    with torch.no_grad():
        # ---- Transfer AlphaNet weights ----
        old_layers = list(model_phase1.alpha_net.net.children())
        new_layers = list(model_new.alpha_net.net.children())

        # First Linear layer: expand input dimension
        # Copy existing weights for env+substrate columns, zero-init Fourier columns
        first_old = old_layers[0]
        first_new = new_layers[0]
        first_new.weight.zero_()  # zero entire weight matrix
        first_new.weight[:, :K_alpha_old].copy_(first_old.weight)
        first_new.bias.copy_(first_old.bias)

        # All subsequent layers: copy exactly (same dimensions)
        for old_m, new_m in zip(old_layers[1:], new_layers[1:]):
            if isinstance(old_m, nn.Linear):
                new_m.weight.copy_(old_m.weight)
                new_m.bias.copy_(old_m.bias)

        # ---- Transfer BetaNet weights ----
        beta_old = model_phase1.beta_net
        beta_new = model_new.beta_net

        if isinstance(beta_old, FactoredDeepBetaNet):
            # Copy per-dimension encoders + projectors
            for k in range(K_env_old):
                beta_new.encoders[k].load_state_dict(beta_old.encoders[k].state_dict())
                beta_new.dim_projectors[k].load_state_dict(beta_old.dim_projectors[k].state_dict())
            # Interaction net is rebuilt with new K_env dims → fresh init
            # (input size changes with K_env, so weights can't be transferred)

        elif isinstance(beta_old, TransformBetaNet):
            # Copy per-dimension transform and weight nets
            for k in range(K_env_old):
                beta_new.transform_nets[k].load_state_dict(beta_old.transform_nets[k].state_dict())
                beta_new.weight_nets[k].load_state_dict(beta_old.weight_nets[k].state_dict())
            # New geo_nets keep fresh initialisation

        else:
            # Unknown beta variant — skip beta transfer
            print("  [expand_model_for_geo] WARNING: Beta weight transfer not supported "
                  f"for {type(beta_old).__name__}, using fresh init")

        # New dim_nets / encoders (K_env_old .. K_env_new-1) keep fresh initialisation

        # ---- Transfer EffortNet weights (if present) ----
        if model_phase1.effort_net is not None and model_new.effort_net is not None:
            model_new.effort_net.load_state_dict(model_phase1.effort_net.state_dict())
        model_new.effort_penalty = model_phase1.effort_penalty

    n_new_alpha = K_alpha_new - K_alpha_old
    n_new_beta = K_env_new - K_env_old
    print(f"  [expand_model_for_geo] Alpha: {K_alpha_old} → {K_alpha_new} "
          f"(+{n_new_alpha} Fourier features, zero-init)")
    print(f"  [expand_model_for_geo] Beta:  {K_env_old} → {K_env_new} "
          f"(+{n_new_beta} geo dim_net(s), fresh init)")

    return model_new


# --------------------------------------------------------------------------
# Combined CLESSO model
# --------------------------------------------------------------------------

class CLESSONet(nn.Module):
    """Joint alpha-beta model for CLESSO.

    Combines AlphaNet (richness) and a monotone beta network (turnover) with the
    CLESSO likelihood:
        within-site:   p_match = 1 / α_i
        between-site:  p_match = S_ij * (α_i + α_j) / (2 * α_i * α_j)
    where S_ij = exp(−η_ij).

    Loss = weighted binary cross-entropy + optional alpha lower-bound penalty.
    """

    def __init__(
        self,
        K_alpha: int,
        K_env: int,
        alpha_hidden: list[int] = (64, 32, 16),
        alpha_dropout: float = 0.1,
        beta_dropout: float = 0.1,
        alpha_activation: str = "relu",
        alpha_lb_lambda: float = 10.0,
        alpha_anchor_lambda: float = 1.0,
        alpha_anchor_tolerance: float = 1.6,
        alpha_regression_lambda: float = 1.0,
        beta_type: str = "transform",
        beta_no_intercept: bool = False,
        transform_n_knots: int = 32,
        transform_g_knots: int = 16,
        K_effort: int = 0,
        effort_hidden: list[int] = (64, 32),
        effort_dropout: float = 0.1,
        effort_mode: str = "additive",
        eps: float = 1e-7,
    ):
        super().__init__()
        if effort_mode not in ("additive", "multiplicative", "completeness"):
            raise ValueError(
                f"effort_mode must be 'additive', 'multiplicative', or "
                f"'completeness', got {effort_mode!r}"
            )
        self.effort_mode = effort_mode
        self.alpha_net = AlphaNet(K_alpha, alpha_hidden, alpha_dropout, alpha_activation)

        # Effort / detectability subnetwork
        if K_effort > 0:
            self.effort_net = EffortNet(K_effort, effort_hidden, effort_dropout)
        else:
            self.effort_net = None

        if beta_type == "factored":
            self.beta_net = FactoredDeepBetaNet(
                K_env, encoder_hidden=16, encoder_depth=2,
                interaction_hidden=[32], dropout=0.0,
                no_intercept=beta_no_intercept,
            )
        elif beta_type == "transform":
            self.beta_net = TransformBetaNet(
                K_env, n_transform_knots=transform_n_knots,
                n_g_knots=transform_g_knots, dropout=beta_dropout,
                no_intercept=beta_no_intercept,
            )
        else:
            raise ValueError(f"Unknown beta_type: {beta_type!r}. "
                             "Use 'transform' or 'factored'.")

        self.alpha_lb_lambda = alpha_lb_lambda
        self.alpha_anchor_lambda = alpha_anchor_lambda
        self.alpha_anchor_tolerance = alpha_anchor_tolerance
        self.alpha_regression_lambda = alpha_regression_lambda
        self.eta_smoothness_lambda = 0.0  # set by train() from config
        self.eta_anti_collapse_lambda = 0.0  # set by train() from config
        self.richness_anchor_lambda = 0.0  # set externally from config
        self.eps = eps

        # Geographic parameter L2 penalty — set externally after construction.
        # K_alpha_env / K_env_env mark the boundary between env and geo params.
        # When > 0 and lambda > 0, geo_param_penalty() is added to compute_loss.
        self.geo_penalty_alpha = 0.0
        self.geo_penalty_beta = 0.0
        self.K_alpha_env = 0   # number of non-geo alpha input columns
        self.K_env_env = 0     # number of non-geo beta env dimensions
        self.effort_penalty = 0.0  # L2 penalty on effort_net parameters

        # Composite likelihood parameters — set externally from config
        self.stratum_weights: list[float] = [0.25, 0.35, 0.25, 0.15]
        self.match_boost_lambda: float = 0.1
        # Per-stratum retention rates for retrospective correction.
        # Keys: "within_r0", "within_r1", "between_tier1_r0", etc.
        # Set after loading metadata.
        self.retention_rates: dict[str, float] = {}

    def geo_param_penalty(self, K_alpha_env: int = 0, K_env_env: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute separate L2 penalties on geographic parameters.

        Args:
            K_alpha_env: number of *non-geo* input columns in alpha's first
                         layer (env+substrate). Columns K_alpha_env: are
                         Fourier geo features whose weights get penalised.
            K_env_env:   number of *non-geo* env dimensions in beta.
                         Beta dim_nets/encoders at indices K_env_env: are
                         geo dim_nets whose parameters get penalised.

        Returns:
            (alpha_geo_l2, beta_geo_l2): scalar tensors, each is
            sum(param**2) over the relevant geo parameters.
            Caller multiplies by the desired lambda.
        """
        device = next(self.parameters()).device
        alpha_geo_l2 = torch.tensor(0.0, device=device)
        beta_geo_l2 = torch.tensor(0.0, device=device)

        # --- Alpha: penalise first-layer weights for Fourier columns ---
        K_alpha_total = self.alpha_net.net[0].in_features
        if K_alpha_env > 0 and K_alpha_env < K_alpha_total:
            # First Linear layer weights: (hidden, K_alpha_total)
            first_layer = self.alpha_net.net[0]
            geo_weights = first_layer.weight[:, K_alpha_env:]  # columns for Fourier
            alpha_geo_l2 = geo_weights.pow(2).sum()

        # --- Beta: penalise parameters of geo dim_nets / encoders ---
        if K_env_env > 0 and K_env_env < getattr(self.beta_net, 'K_env', 0):
            if isinstance(self.beta_net, FactoredDeepBetaNet):
                for k in range(K_env_env, self.beta_net.K_env):
                    for p in self.beta_net.encoders[k].parameters():
                        beta_geo_l2 = beta_geo_l2 + p.pow(2).sum()
                    for p in self.beta_net.dim_projectors[k].parameters():
                        beta_geo_l2 = beta_geo_l2 + p.pow(2).sum()
            elif isinstance(self.beta_net, TransformBetaNet):
                # Penalise geo_nets parameters (not env transform/weight nets)
                for g in range(self.beta_net.n_geo_dims):
                    for p in self.beta_net.geo_nets[g].parameters():
                        beta_geo_l2 = beta_geo_l2 + p.pow(2).sum()

        return alpha_geo_l2, beta_geo_l2

    def effort_param_penalty(self) -> torch.Tensor:
        """Compute L2 penalty on all effort_net parameters.

        Returns scalar tensor: sum(param**2) over all effort_net weights.
        Caller multiplies by self.effort_penalty.
        """
        device = next(self.parameters()).device
        if self.effort_net is None:
            return torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        for p in self.effort_net.parameters():
            total = total + p.pow(2).sum()
        return total

    def _compute_alpha(
        self,
        z: torch.Tensor,
        w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute alpha with effort decomposition.

        Additive mode:
            α = softplus(env_logit + effort_logit) + 1

        Multiplicative mode:
            α_obs = α_env · σ(effort_logit)
            Effort acts as a general modifier ∈ (0, 1).
            Can push apparent richness below α_env but also below 1
            for very low capture fractions.

        Completeness mode:
            α_true = 1 + softplus(f_α(z))
            c_i = σ(f_ω(w)) ∈ (0, 1)   -- capture / completeness fraction
            α_obs = 1 + c_i · softplus(f_α(z)) = 1 + c_i · (α_true - 1)
            Effort can only *reduce* apparent richness below true richness;
            α_obs ≥ 1 is guaranteed.  When c → 1, α_obs → α_true.
            At initialisation σ(0) = 0.5, so α_obs ≈ 1 + 0.5·(α_true - 1).

        Args:
            z: (B, K_alpha) standardised env site covariates
            w: (B, K_effort) standardised effort covariates, or None
        Returns:
            alpha: (B,) positive richness values > 1
        """
        if self.effort_mode == "completeness":
            env_sp = F.softplus(self.alpha_net.logit(z))  # α_true − 1
            if self.effort_net is not None and w is not None:
                c = torch.sigmoid(self.effort_net(w))  # completeness ∈ (0,1)
                return 1.0 + c * env_sp
            return 1.0 + env_sp  # no effort → full completeness
        elif self.effort_mode == "multiplicative":
            alpha_env = F.softplus(self.alpha_net.logit(z)) + 1.0
            if self.effort_net is not None and w is not None:
                effort_logit = self.effort_net(w)
                capture_frac = torch.sigmoid(effort_logit)  # range (0, 1)
                return alpha_env * capture_frac
            return alpha_env
        else:  # additive
            raw_env = self.alpha_net.logit(z)
            if self.effort_net is not None and w is not None:
                raw_effort = self.effort_net(w)
                raw = raw_env + raw_effort
            else:
                raw = raw_env
            return F.softplus(raw) + 1.0

    def _compute_alpha_env_only(self, z: torch.Tensor) -> torch.Tensor:
        """Compute alpha from env covariates only (effort zeroed).

        Used at prediction time to get 'true richness' without
        observation-process artifacts.
        """
        return self.alpha_net(z)  # softplus(logit) + 1

    def forward(
        self,
        z_i: torch.Tensor,       # (B, K_alpha): site covariates for site i
        z_j: torch.Tensor,       # (B, K_alpha): site covariates for site j
        env_diff: torch.Tensor,  # (B, K_env):   |env_i - env_j|
        is_within: torch.Tensor, # (B,):         1=within, 0=between
        w_i: torch.Tensor | None = None,  # (B, K_effort): effort covs site i
        w_j: torch.Tensor | None = None,  # (B, K_effort): effort covs site j
        env_i: torch.Tensor | None = None,  # (B, K_env): raw standardised env site i
        env_j: torch.Tensor | None = None,  # (B, K_env): raw standardised env site j
        geo_dist: torch.Tensor | None = None,  # (B, n_geo): geo distance(s)
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning alpha, eta, similarity, and match probability.

        Returns dict with keys:
            alpha_i, alpha_j : (B,) richness estimates (including effort)
            eta              : (B,) turnover (only meaningful for between-site)
            similarity       : (B,) S = exp(-eta)
            p_match          : (B,) probability of species match
        """
        alpha_i = self._compute_alpha(z_i, w_i)
        alpha_j = self._compute_alpha(z_j, w_j)
        if isinstance(self.beta_net, TransformBetaNet):
            eta = self.beta_net(env_i, env_j, geo_dist=geo_dist)
        else:
            eta = self.beta_net(env_diff)
        similarity = torch.exp(-eta)

        # --- Log-space match probability ---
        # Computing log(p) directly avoids underflow when α is large.
        # Critical for beta gradients: d[log(p_between)]/dη = -1 always,
        # regardless of alpha magnitude (no more vanishing gradients).
        #
        # Within-site:  p = 1/α  →  log(p) = -log(α)
        # Between-site: p = S·(α_i+α_j)/(2·α_i·α_j)
        #               log(p) = -η + log(α_i+α_j) - log(2) - log(α_i) - log(α_j)
        log_p_within = -torch.log(alpha_i)
        log_p_between = (
            -eta
            + torch.log(alpha_i + alpha_j)
            - math.log(2.0)
            - torch.log(alpha_i)
            - torch.log(alpha_j)
        )

        is_w = is_within.bool()
        log_p_match = torch.where(is_w, log_p_within, log_p_between)
        # Clamp log(p) to [log(eps), log(1-eps)] for numerical safety
        log_eps = math.log(self.eps)
        log_1m_eps = math.log(1.0 - self.eps)
        log_p_match = log_p_match.clamp(log_eps, log_1m_eps)

        # Materialise p for diagnostics / other uses
        p_match = log_p_match.exp()
        # log(1-p): use log1p(-p) for numerical stability when p is small
        log_1m_p = torch.log1p(-p_match)
        log_1m_p = log_1m_p.clamp(min=log_eps)

        return {
            "alpha_i": alpha_i,
            "alpha_j": alpha_j,
            "eta": eta,
            "similarity": similarity,
            "p_match": p_match,
            "log_p_match": log_p_match,
            "log_1m_p": log_1m_p,
        }

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        S_obs: torch.Tensor,      # (n_sites,) observed richness
        w_i: torch.Tensor | None = None,
        w_j: torch.Tensor | None = None,
        anchor_richness: torch.Tensor | None = None,  # (n_sites,) anchor richness from raster
        weight_clamp_factor: float = 0.0,
        weight_log_normalise: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Composite likelihood loss with per-stratum weighting.

        Loss = Σ_s λ_s · L_s  +  λ_boost · L_boost  +  penalties

        ALL strata use the same retrospective Bernoulli correction:
          p → p_samp = r₀·p / (r₀·p + r₁·(1−p))
        using per-stratum retention rates (r₀, r₁) from the sampler.

        The key insight is that this correction simultaneously fixes
        the gradient asymmetry that causes eta collapse.  For between-site
        pairs with α~100, the raw BCE has match gradient ~200× larger than
        mismatch.  But when the sampler enforces ~50:50 quotas, r₀ ≈ 1.0
        and r₁ << 1 (because mismatches are far more abundant in the
        population).  The retrospective correction maps p ≈ 0.01 →
        p_samp ≈ 0.5, making both match and mismatch BCE gradients O(1).

        Strata:
          0 = within-site (W)
          1 = same-hex between-site (S)  — uses between_tier1_r0/r1
          2 = neighbour-hex between-site (N)  — uses between_tier2_r0/r1
          3 = distant-hex between-site (D)  — uses between_tier3_r0/r1
          4 = match-boost (auxiliary, not in main composite likelihood)

        Args:
            batch: dict with y, is_within, design_w, stratum, env_diff, site_i, site_j
            z_i, z_j: (B, K_alpha) site covariates
            S_obs: (n_sites,) observed richness per site
            w_i, w_j: (B, K_effort) effort covariates or None

        Returns:
            dict with loss components and forward outputs
        """
        fwd = self.forward(z_i, z_j, batch["env_diff"], batch["is_within"],
                           w_i=w_i, w_j=w_j,
                           env_i=batch.get("env_i"),
                           env_j=batch.get("env_j"),
                           geo_dist=batch.get("geo_dist"))

        y = batch["y"]
        design_w = batch["design_w"]
        stratum = batch["stratum"]
        p = fwd["p_match"]
        eps = 1e-7

        log_p = fwd["log_p_match"]
        log_1m_p = fwd["log_1m_p"]

        # ------------------------------------------------------------------
        # Per-element BCE on raw p (before any correction)
        # ------------------------------------------------------------------
        bce_per_pair = -(1.0 - y) * log_p - y * log_1m_p

        # ------------------------------------------------------------------
        # Retrospective correction for ALL strata:
        #
        # Every stratum uses p → p_samp = r₀·p / (r₀·p + r₁·(1−p))
        # with per-stratum retention rates from the sampler.
        #
        # This simultaneously:
        #   1. Corrects the sampling bias (quota 50:50 ≠ population freq)
        #   2. Eliminates the between-site gradient asymmetry:
        #      When r₀ ≈ 1.0 and r₁ << 1, p ≈ 0.01 maps to p_samp ≈ 0.5,
        #      making d(BCE)/dη balanced for match and mismatch pairs.
        # ------------------------------------------------------------------

        lambda_s = self.stratum_weights  # [λ_W, λ_S, λ_N, λ_D]
        total_bce = torch.tensor(0.0, device=p.device)
        stratum_bce_parts = {}
        weight_ess_parts = {}

        # Retention rate keys per stratum
        _rr_keys = {
            0: ("within_r0", "within_r1"),
            1: ("between_tier1_r0", "between_tier1_r1"),
            2: ("between_tier2_r0", "between_tier2_r1"),
            3: ("between_tier3_r0", "between_tier3_r1"),
        }

        rr = self.retention_rates

        for s_idx in range(4):
            mask_s = stratum == s_idx
            if not mask_s.any():
                continue

            p_s = p[mask_s]
            y_s = y[mask_s]
            dw_s = design_w[mask_s]

            r0_key, r1_key = _rr_keys[s_idx]
            r0_s = rr.get(r0_key, 1.0)
            r1_s = rr.get(r1_key, 1.0)

            if r0_s != 1.0 or r1_s != 1.0:
                # Retrospective p_samp = r0*p / (r0*p + r1*(1-p))
                p_samp = r0_s * p_s / (r0_s * p_s + r1_s * (1.0 - p_s) + eps)
                log_ps = torch.log(p_samp.clamp(min=eps))
                log_1ps = torch.log((1.0 - p_samp).clamp(min=eps))
                bce_s = -(1.0 - y_s) * log_ps - y_s * log_1ps
            else:
                bce_s = bce_per_pair[mask_s]

            w_sum_s = dw_s.sum()
            L_s = (dw_s * bce_s).sum() / w_sum_s.clamp(min=eps)
            total_bce = total_bce + lambda_s[s_idx] * L_s
            stratum_bce_parts[f"bce_s{s_idx}"] = L_s.detach()
            ess_s = (w_sum_s ** 2 / (dw_s ** 2).sum()).item()
            weight_ess_parts[f"ess_s{s_idx}"] = ess_s

        # ------------------------------------------------------------------
        # Match-boost auxiliary loss (stratum 4)
        # ------------------------------------------------------------------
        boost_loss = torch.tensor(0.0, device=p.device)
        mask_boost = stratum == 4
        if mask_boost.any() and self.match_boost_lambda > 0:
            # Unweighted BCE on boosted match pairs (they are all y=0)
            bce_boost = bce_per_pair[mask_boost]
            boost_loss = self.match_boost_lambda * bce_boost.mean()
            stratum_bce_parts["bce_boost"] = bce_boost.mean().detach()

        bce_loss = total_bce + boost_loss

        # Overall ESS (across all main strata)
        main_mask = stratum < 4
        if main_mask.any():
            w_main = design_w[main_mask]
            _sum_w = w_main.sum()
            weight_ess = (_sum_w ** 2 / (w_main ** 2).sum()).item()
        else:
            weight_ess = 0.0

        # ------------------------------------------------------------------
        # Alpha penalties (unchanged)
        # ------------------------------------------------------------------

        # Pre-compute α_env (effort-stripped true richness) for lb + anchor + richness_anchor penalties
        alpha_env_i = alpha_env_j = None
        if (self.alpha_lb_lambda > 0.0 or self.alpha_anchor_lambda > 0.0
                or self.richness_anchor_lambda > 0.0) and S_obs is not None:
            alpha_env_i = self._compute_alpha_env_only(z_i)
            alpha_env_j = self._compute_alpha_env_only(z_j)

        # Soft lower-bound penalty on α_env: true richness must ≥ S_obs (log-space)
        lb_penalty = torch.tensor(0.0, device=p.device)
        if self.alpha_lb_lambda > 0.0 and alpha_env_i is not None:
            s_i_lb = S_obs[batch["site_i"]]
            s_j_lb = S_obs[batch["site_j"]]

            eps_lb = 1.0
            log_gap_i = torch.log(s_i_lb.clamp(min=eps_lb)) - torch.log(alpha_env_i)
            log_gap_j = torch.log(s_j_lb.clamp(min=eps_lb)) - torch.log(alpha_env_j)
            violation_i = F.softplus(10.0 * log_gap_i) / 10.0
            violation_j = F.softplus(10.0 * log_gap_j) / 10.0
            lb_penalty = self.alpha_lb_lambda * (
                violation_i.pow(2).mean() + violation_j.pow(2).mean()
            )

        # Direct alpha regression: MSE(alpha, S_obs)
        alpha_reg_loss = torch.tensor(0.0, device=p.device)
        if self.alpha_regression_lambda > 0.0 and S_obs is not None:
            alpha_i_obs = S_obs[batch["site_i"]]
            alpha_j_obs = S_obs[batch["site_j"]]
            mask_i = alpha_i_obs > 0
            mask_j = alpha_j_obs > 0
            if mask_i.any():
                reg_i = (fwd["alpha_i"][mask_i] - alpha_i_obs[mask_i]).pow(2).mean()
            else:
                reg_i = torch.tensor(0.0, device=p.device)
            if mask_j.any():
                reg_j = (fwd["alpha_j"][mask_j] - alpha_j_obs[mask_j]).pow(2).mean()
            else:
                reg_j = torch.tensor(0.0, device=p.device)
            alpha_reg_loss = self.alpha_regression_lambda * (reg_i + reg_j) / 2.0

        # Eta smoothness penalty (disabled by default; pushes η→0)
        eta_smooth_loss = torch.tensor(0.0, device=p.device)
        if self.eta_smoothness_lambda > 0.0:
            between_mask = batch["is_within"] == 0
            if between_mask.any():
                eta_between = fwd["eta"][between_mask]
                eta_smooth_loss = self.eta_smoothness_lambda * eta_between.pow(2).mean()

        # Eta anti-collapse regularizer: -λ * mean(log(η + ε))
        # Logarithmic barrier that strongly resists η→0 while exerting
        # negligible force at moderate η (e.g. at η=2, gradient ≈ -0.005).
        eta_ac_loss = torch.tensor(0.0, device=p.device)
        if self.eta_anti_collapse_lambda > 0.0:
            between_mask_ac = batch["is_within"] == 0
            if between_mask_ac.any():
                eta_between_ac = fwd["eta"][between_mask_ac]
                eta_ac_loss = -self.eta_anti_collapse_lambda * torch.log(
                    eta_between_ac + 1e-4
                ).mean()

        # Alpha anchor penalty (log-normal prior with dead-zone tolerance)
        # Applied to α_env (effort-stripped true richness), NOT α_obs.
        # This lets EffortNet freely explain the gap between α_env and S_obs
        # while preventing runaway α_env inflation.
        anchor_penalty = torch.tensor(0.0, device=p.device)
        if self.alpha_anchor_lambda > 0.0 and alpha_env_i is not None:
            s_i = S_obs[batch["site_i"]].clamp(min=1.0)
            s_j = S_obs[batch["site_j"]].clamp(min=1.0)

            tau = self.alpha_anchor_tolerance
            gap_i = (torch.log(alpha_env_i) - torch.log(s_i)).abs() - tau
            gap_j = (torch.log(alpha_env_j) - torch.log(s_j)).abs() - tau
            viol_i = F.relu(gap_i)
            viol_j = F.relu(gap_j)
            anchor_penalty = self.alpha_anchor_lambda * (
                viol_i.pow(2).mean() + viol_j.pow(2).mean()
            )

        # Richness anchor penalty (external raster-based soft constraint)
        # Pushes α_env toward anchor values at sites with valid anchor data.
        #   penalty = λ * mean[ (log(α_env) - log(anchor))² ]
        richness_anchor_loss = torch.tensor(0.0, device=p.device)
        if self.richness_anchor_lambda > 0.0 and alpha_env_i is not None and anchor_richness is not None:
            anc_i = anchor_richness[batch["site_i"]]
            anc_j = anchor_richness[batch["site_j"]]
            valid_i = anc_i > 1.0
            valid_j = anc_j > 1.0
            if valid_i.any():
                ra_i = (torch.log(alpha_env_i[valid_i]) - torch.log(anc_i[valid_i])).pow(2).mean()
            else:
                ra_i = torch.tensor(0.0, device=p.device)
            if valid_j.any():
                ra_j = (torch.log(alpha_env_j[valid_j]) - torch.log(anc_j[valid_j])).pow(2).mean()
            else:
                ra_j = torch.tensor(0.0, device=p.device)
            richness_anchor_loss = self.richness_anchor_lambda * (ra_i + ra_j) / 2.0

        loss = bce_loss + lb_penalty + alpha_reg_loss + eta_smooth_loss + eta_ac_loss + anchor_penalty + richness_anchor_loss

        # Effort net L2 penalty
        effort_loss = torch.tensor(0.0, device=p.device)
        if self.effort_penalty > 0 and self.effort_net is not None:
            effort_loss = self.effort_penalty * self.effort_param_penalty()
            loss = loss + effort_loss

        # Geographic parameter L2 penalty
        geo_loss = torch.tensor(0.0, device=p.device)
        if (self.geo_penalty_alpha > 0 or self.geo_penalty_beta > 0) \
                and (self.K_alpha_env > 0 or self.K_env_env > 0):
            alpha_geo_l2, beta_geo_l2 = self.geo_param_penalty(
                K_alpha_env=self.K_alpha_env, K_env_env=self.K_env_env)
            geo_loss = self.geo_penalty_alpha * alpha_geo_l2 \
                     + self.geo_penalty_beta * beta_geo_l2
            loss = loss + geo_loss

        return {
            "loss": loss,
            "bce_loss": bce_loss,
            "lb_penalty": lb_penalty,
            "anchor_penalty": anchor_penalty,
            "alpha_reg_loss": alpha_reg_loss,
            "eta_smooth_loss": eta_smooth_loss,
            "eta_ac_loss": eta_ac_loss,
            "effort_loss": effort_loss,
            "geo_loss": geo_loss,
            "boost_loss": boost_loss,
            "richness_anchor_loss": richness_anchor_loss,
            "weight_ess": weight_ess,
            **stratum_bce_parts,
            **weight_ess_parts,
            **fwd,
        }
