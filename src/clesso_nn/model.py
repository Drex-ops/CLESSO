"""
model.py -- Neural network architecture for CLESSO.

Three components:
  1. AlphaNet:         site covariates → richness (α > 1)
  2. MonotoneBetaNet:  |env_i − env_j| → turnover η ≥ 0  (monotone network)
  3. CLESSONet:        combines both, computes match probability and loss

The monotone beta network enforces that increasing environmental distance
always produces equal or greater compositional dissimilarity — the same
constraint achieved by I-splines + non-negative coefficients in the TMB
model, but with greater flexibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, K_alpha) standardised site covariates
        Returns:
            alpha: (batch,) positive richness values > 1
        """
        raw = self.net(z).squeeze(-1)
        # softplus ensures positivity; +1 ensures alpha > 1
        return F.softplus(raw) + 1.0


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


class MonotoneBetaNet(nn.Module):
    """Deep monotone network mapping environmental distances → turnover η.

    Architecture:
        input |env_diff| (K_env) → [MonotoneLinear→ReLU] × L → MonotoneLinear(1)
        output = softplus(raw)   (ensures η ≥ 0)

    Because:
      - All inputs are |env_i − env_j| ≥ 0
      - All weights are non-negative (via MonotoneLinear)
      - All activations (ReLU) are monotone non-decreasing

    The composition is monotone non-decreasing in each input dimension.
    This guarantees: more environmental distance → higher η → lower
    similarity S = exp(−η), matching the ecological constraint from GDM.

    NOTE: This deep variant is prone to dimensional collapse (all weights
    converging to the same value under weight decay). Prefer AdditiveBetaNet
    which mirrors GDM's additive I-spline structure.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (64, 32, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                MonotoneLinear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(MonotoneLinear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, env_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            env_diff: (batch, K_env) non-negative pairwise env distances
        Returns:
            eta: (batch,) non-negative turnover values
        """
        raw = self.net(env_diff).squeeze(-1)
        return F.softplus(raw)


class AdditiveBetaNet(nn.Module):
    """Additive monotone beta network: η = Σ_k f_k(|Δx_k|).

    Each f_k is an independent per-dimension monotone network:
        f_k: scalar |Δx_k| → MonotoneLinear(n_knots) → Softplus → MonotoneLinear(1)

    Uses **softplus** (not ReLU) as the hidden activation. Softplus has no
    dead zone — it always has a nonzero gradient, giving naturally gradual
    onset of turnover rather than the hard on/off threshold behavior of ReLU.
    Softplus is still monotone non-decreasing, so the monotonicity guarantee
    is preserved.

    This directly mirrors GDM's additive I-spline + non-negative coefficient
    structure, but with learned smooth transformations. Each dimension MUST
    develop its own response curve, preventing the dimensional collapse that
    plagues the deep MonotoneBetaNet.
    """

    def __init__(
        self,
        input_dim: int,
        n_knots: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K_env = input_dim
        self.n_knots = n_knots

        # Per-dimension monotone transformation: scalar → n_knots → 1
        # Softplus activation instead of ReLU: no dead zone, gradual onset
        self.dim_nets = nn.ModuleList()
        for _ in range(input_dim):
            self.dim_nets.append(nn.Sequential(
                MonotoneLinear(1, n_knots, bias=True),
                nn.Softplus(),
                nn.Dropout(dropout),
                MonotoneLinear(n_knots, 1, bias=False),  # no output bias → f_k(0)≈0
            ))

        self._init_knots()

    def _init_knots(self):
        """Initialise per-dimension networks for gradual response onset.

        With softplus activation, every neuron contributes at all input values
        (no dead zone). The biases control where each neuron's contribution
        accelerates. We spread biases so responses ramp smoothly from ~0.1σ
        to ~3σ of standardised environmental distance.

        Second-layer weights are initialised small so each dimension
        contributes ~0.3 at max distance, giving total η ≈ 5 across 17 dims.
        """
        for net in self.dim_nets:
            first_layer = True
            for m in net.modules():
                if isinstance(m, MonotoneLinear):
                    if first_layer:
                        # First layer: controls knot positions
                        nn.init.uniform_(m.weight_raw, -0.5, 0.5)
                        if m.bias is not None:
                            n = m.bias.shape[0]
                            knots = torch.linspace(-3.0, 0.0, n)
                            m.bias.data.copy_(knots)
                        first_layer = False
                    else:
                        # Second layer (output): very small so per-dim
                        # contribution is modest (~0.3 at max distance).
                        # softplus(-3) ≈ 0.05, so 32 neurons × 0.05 ≈ 1.6
                        # but each neuron's delta is small → f_k(max) ≈ 0.3
                        nn.init.constant_(m.weight_raw, -3.0)

    def forward(self, env_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            env_diff: (batch, K_env) non-negative pairwise env distances
        Returns:
            eta: (batch,) non-negative turnover values η ≥ 0
        """
        total = torch.zeros(env_diff.shape[0], device=env_diff.device)
        # Zero-input baseline (shared across batch) — computed once per forward
        zero = torch.zeros(1, 1, device=env_diff.device)
        for k in range(self.K_env):
            x_k = env_diff[:, k : k + 1]         # (batch, 1)
            g_k = self.dim_nets[k](x_k)           # (batch, 1)
            g_k_0 = self.dim_nets[k](zero)         # (1, 1) — baseline at zero
            f_k = g_k - g_k_0                      # f_k(0) = 0 exactly
            total = total + f_k.squeeze(-1)         # (batch,)
        # Clamp η to [0, η_max] to prevent runaway.
        # η=10 → S=exp(-10)≈4.5e-5, more than enough for any ecological scenario.
        # Without this clamp, the model can push η→∞ for all between-site pairs,
        # learning the degenerate solution p≈0 everywhere.
        return total.clamp(min=0.0, max=10.0)


class FactoredDeepBetaNet(nn.Module):
    """Factored deep beta network: per-dimension monotone encoders + monotone interaction.

    Combines the per-dimension specialisation of AdditiveBetaNet with cross-dimension
    interactions, without the collapse problems of the fully-connected MonotoneBetaNet.

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
    ):
        super().__init__()
        self.K_env = input_dim
        self.encoder_hidden = encoder_hidden

        if interaction_hidden is None:
            interaction_hidden = [32]

        # --- Per-dimension monotone encoders ---
        # Each maps |Δx_k| (scalar) → h_k (encoder_hidden-dim, all ≥ 0 via ReLU)
        # No dropout: baseline subtraction requires deterministic forward pass.
        self.encoders = nn.ModuleList()
        for _ in range(input_dim):
            layers: list[nn.Module] = []
            in_dim = 1
            for _ in range(encoder_depth):
                out_dim = encoder_hidden
                layers.append(MonotoneLinear(in_dim, out_dim))
                layers.append(nn.Softplus())
                in_dim = out_dim
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

        return eta.clamp(min=0.0, max=10.0)


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
        beta_hidden=config.beta_hidden,
        alpha_dropout=config.alpha_dropout,
        beta_dropout=config.beta_dropout,
        alpha_activation=config.alpha_activation,
        alpha_lb_lambda=config.alpha_lower_bound_lambda,
        alpha_regression_lambda=config.alpha_regression_lambda,
        beta_type=config.beta_type,
        beta_n_knots=config.beta_n_knots,
    )
    model_new.eta_smoothness_lambda = config.eta_smoothness_lambda

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

        if isinstance(beta_old, AdditiveBetaNet):
            # Copy per-dimension spline nets
            for k in range(K_env_old):
                beta_new.dim_nets[k].load_state_dict(beta_old.dim_nets[k].state_dict())

        elif isinstance(beta_old, FactoredDeepBetaNet):
            # Copy per-dimension encoders + projectors
            for k in range(K_env_old):
                beta_new.encoders[k].load_state_dict(beta_old.encoders[k].state_dict())
                beta_new.dim_projectors[k].load_state_dict(beta_old.dim_projectors[k].state_dict())
            # Interaction net is rebuilt with new K_env dims → fresh init
            # (input size changes with K_env, so weights can't be transferred)

        else:
            # Deep beta or unknown — skip beta transfer
            print("  [expand_model_for_geo] WARNING: Beta weight transfer not supported "
                  f"for {type(beta_old).__name__}, using fresh init")

        # New dim_nets / encoders (K_env_old .. K_env_new-1) keep fresh initialisation

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

    Combines AlphaNet (richness) and MonotoneBetaNet (turnover) with the
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
        beta_hidden: list[int] = (64, 32, 16),
        alpha_dropout: float = 0.1,
        beta_dropout: float = 0.1,
        alpha_activation: str = "relu",
        alpha_lb_lambda: float = 10.0,
        alpha_regression_lambda: float = 1.0,
        beta_type: str = "additive",
        beta_n_knots: int = 32,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha_net = AlphaNet(K_alpha, alpha_hidden, alpha_dropout, alpha_activation)

        if beta_type == "additive":
            self.beta_net = AdditiveBetaNet(K_env, beta_n_knots, beta_dropout)
        elif beta_type == "factored":
            self.beta_net = FactoredDeepBetaNet(
                K_env, encoder_hidden=16, encoder_depth=2,
                interaction_hidden=[32], dropout=0.0,
            )
        elif beta_type == "deep":
            self.beta_net = MonotoneBetaNet(K_env, beta_hidden, beta_dropout)
        else:
            raise ValueError(f"Unknown beta_type: {beta_type!r}. "
                             "Use 'additive', 'factored', or 'deep'.")

        self.alpha_lb_lambda = alpha_lb_lambda
        self.alpha_regression_lambda = alpha_regression_lambda
        self.eta_smoothness_lambda = 0.0  # set by train() from config
        self.eps = eps

    def forward(
        self,
        z_i: torch.Tensor,       # (B, K_alpha): site covariates for site i
        z_j: torch.Tensor,       # (B, K_alpha): site covariates for site j
        env_diff: torch.Tensor,  # (B, K_env):   |env_i - env_j|
        is_within: torch.Tensor, # (B,):         1=within, 0=between
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning alpha, eta, similarity, and match probability.

        Returns dict with keys:
            alpha_i, alpha_j : (B,) richness estimates
            eta              : (B,) turnover (only meaningful for between-site)
            similarity       : (B,) S = exp(-eta)
            p_match          : (B,) probability of species match
        """
        alpha_i = self.alpha_net(z_i)
        alpha_j = self.alpha_net(z_j)
        eta = self.beta_net(env_diff)
        similarity = torch.exp(-eta)

        # Match probability
        p_within = 1.0 / alpha_i
        p_between = similarity * (alpha_i + alpha_j) / (2.0 * alpha_i * alpha_j)

        p_match = torch.where(is_within.bool(), p_within, p_between)
        p_match = p_match.clamp(self.eps, 1.0 - self.eps)

        return {
            "alpha_i": alpha_i,
            "alpha_j": alpha_j,
            "eta": eta,
            "similarity": similarity,
            "p_match": p_match,
        }

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        S_obs: torch.Tensor,      # (n_sites,) observed richness
    ) -> dict[str, torch.Tensor]:
        """Compute weighted BCE loss + alpha penalties.

        Loss = weighted_BCE + lb_penalty + alpha_regression

        Where:
          - weighted_BCE:     standard binary cross-entropy weighted by pair weights
          - lb_penalty:       soft lower-bound: λ_lb * mean[softplus(S_obs-α)]²
          - alpha_regression: direct MSE anchor: λ_reg * MSE(α, S_obs)
            Prevents alpha from collapsing to degenerate values.

        Args:
            batch: dict from dataloader with y, is_within, weight, env_diff, site_i, site_j
            z_i: (B, K_alpha) site covariates for site i
            z_j: (B, K_alpha) site covariates for site j
            S_obs: (n_sites,) observed species count per site

        Returns:
            dict with loss, bce_loss, lb_penalty, alpha_reg_loss, and forward outputs
        """
        fwd = self.forward(z_i, z_j, batch["env_diff"], batch["is_within"])

        y = batch["y"]
        w = batch["weight"]
        p = fwd["p_match"]

        # Weighted binary cross-entropy
        # y=0 means match → log(p), y=1 means mismatch → log(1-p)
        bce = -(1.0 - y) * torch.log(p) - y * torch.log(1.0 - p)
        bce_loss = (w * bce).sum() / w.sum()

        # Soft lower-bound penalty: discourage alpha < S_obs
        lb_penalty = torch.tensor(0.0, device=p.device)
        if self.alpha_lb_lambda > 0.0 and S_obs is not None:
            alpha_i_obs = S_obs[batch["site_i"]]
            alpha_j_obs = S_obs[batch["site_j"]]

            # softplus(S_obs - alpha)^2 — only penalises when alpha < S_obs
            violation_i = F.softplus(10.0 * (alpha_i_obs - fwd["alpha_i"])) / 10.0
            violation_j = F.softplus(10.0 * (alpha_j_obs - fwd["alpha_j"])) / 10.0
            lb_penalty = self.alpha_lb_lambda * (
                violation_i.pow(2).mean() + violation_j.pow(2).mean()
            )

        # Direct alpha regression: MSE(alpha, S_obs)
        # Anchors alpha to observed richness, breaking the p≈0.5 equilibrium
        alpha_reg_loss = torch.tensor(0.0, device=p.device)
        if self.alpha_regression_lambda > 0.0 and S_obs is not None:
            alpha_i_obs = S_obs[batch["site_i"]]
            alpha_j_obs = S_obs[batch["site_j"]]
            # Only penalise sites with known richness (S_obs > 0)
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

        # Eta smoothness penalty: penalise large eta to encourage gradual
        # turnover ramps rather than step functions.
        eta_smooth_loss = torch.tensor(0.0, device=p.device)
        if self.eta_smoothness_lambda > 0.0:
            between_mask = batch["is_within"] == 0
            if between_mask.any():
                eta_between = fwd["eta"][between_mask]
                eta_smooth_loss = self.eta_smoothness_lambda * eta_between.pow(2).mean()

        loss = bce_loss + lb_penalty + alpha_reg_loss + eta_smooth_loss

        return {
            "loss": loss,
            "bce_loss": bce_loss,
            "lb_penalty": lb_penalty,
            "alpha_reg_loss": alpha_reg_loss,
            "eta_smooth_loss": eta_smooth_loss,
            **fwd,
        }
