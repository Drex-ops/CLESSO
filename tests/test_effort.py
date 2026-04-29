#!/usr/bin/env python3
"""Smoke tests for effort/detectability additive decomposition."""
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from clesso_nn.model import CLESSONet, EffortNet


def test_no_effort_backward_compat():
    """Model without effort should work identically to before."""
    m = CLESSONet(K_alpha=17, K_env=15, beta_type="transform")
    m.eval()
    assert m.effort_net is None
    z = torch.randn(4, 17)
    with torch.no_grad():
        alpha1 = m._compute_alpha_env_only(z)
        alpha2 = m._compute_alpha(z, w=None)
    assert alpha1.shape == (4,)
    assert torch.allclose(alpha1, alpha2), "env_only and no-effort should match"
    print("PASS: No effort (backward compat)")


def test_effort_model_construction():
    """Model with effort should create EffortNet."""
    m = CLESSONet(
        K_alpha=17, K_env=15, beta_type="transform",
        K_effort=6, effort_hidden=[64, 32], effort_dropout=0.1,
    )
    assert m.effort_net is not None
    assert isinstance(m.effort_net, EffortNet)
    n_params = sum(p.numel() for p in m.effort_net.parameters())
    print(f"  EffortNet params: {n_params}")
    assert n_params > 0
    print("PASS: Effort model construction")


def test_effort_alpha_decomposition():
    """Additive decomposition should work correctly."""
    m = CLESSONet(
        K_alpha=17, K_env=15, beta_type="transform",
        K_effort=6, effort_hidden=[64, 32], effort_dropout=0.1,
    )
    z = torch.randn(4, 17)
    w = torch.randn(4, 6)

    alpha_env = m._compute_alpha_env_only(z)
    alpha_full = m._compute_alpha(z, w)

    assert alpha_env.shape == (4,)
    assert alpha_full.shape == (4,)
    assert (alpha_env >= 1.0).all(), "Alpha env should be >= 1 (softplus + 1)"
    assert (alpha_full >= 1.0).all(), "Alpha full should be >= 1"

    # With zero-init output bias, effort logit starts near 0
    # so full should be close to env_only
    diff = (alpha_full - alpha_env).abs().max().item()
    print(f"  Initial env-vs-full max diff: {diff:.4f}")
    print("PASS: Alpha decomposition")


def test_forward_with_effort():
    """Forward pass should accept w_i/w_j."""
    m = CLESSONet(
        K_alpha=17, K_env=15, beta_type="transform",
        K_effort=6, effort_hidden=[64, 32], effort_dropout=0.1,
    )
    z_i = torch.randn(8, 17)
    z_j = torch.randn(8, 17)
    w_i = torch.randn(8, 6)
    w_j = torch.randn(8, 6)
    env_diff = torch.rand(8, 15)
    is_within = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)

    fwd = m.forward(z_i, z_j, env_diff, is_within, w_i=w_i, w_j=w_j)
    assert "p_match" in fwd
    assert "alpha_i" in fwd
    assert fwd["alpha_i"].shape == (8,)
    assert fwd["p_match"].shape == (8,)
    print("PASS: Forward with effort")


def test_compute_loss_with_effort():
    """compute_loss should include effort_loss."""
    m = CLESSONet(
        K_alpha=17, K_env=15, beta_type="transform",
        K_effort=6, effort_hidden=[64, 32], effort_dropout=0.1,
    )
    z_i = torch.randn(8, 17)
    z_j = torch.randn(8, 17)
    w_i = torch.randn(8, 6)
    w_j = torch.randn(8, 6)

    batch = {
        "site_i": torch.arange(8),
        "site_j": torch.arange(8),
        "y": torch.randint(0, 2, (8,)).float(),
        "is_within": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32),
        "weight": torch.ones(8),
        "design_w": torch.ones(8),
        "stratum": torch.zeros(8, dtype=torch.long),
        "env_diff": torch.rand(8, 15),
    }
    S_obs = torch.ones(8) * 50

    # Without penalty
    result = m.compute_loss(batch, z_i, z_j, S_obs, w_i=w_i, w_j=w_j)
    assert "effort_loss" in result
    assert result["effort_loss"].item() == 0.0, "Should be zero without penalty"

    # With penalty
    m.effort_penalty = 0.01
    result2 = m.compute_loss(batch, z_i, z_j, S_obs, w_i=w_i, w_j=w_j)
    assert result2["effort_loss"].item() > 0, "Should be nonzero with penalty"
    print(f"  effort_loss with penalty: {result2['effort_loss'].item():.6f}")
    print("PASS: compute_loss with effort")


def test_effort_net_zero_init():
    """EffortNet output bias should start at 0 (neutral)."""
    net = EffortNet(input_dim=6, hidden_dims=[64, 32], dropout=0.1)
    # The last layer output bias should be 0
    last_layer = net.net[-1]
    assert last_layer.bias is not None
    assert last_layer.bias.item() == 0.0
    # With random input, output should be non-zero but small
    x = torch.randn(100, 6)
    out = net(x)
    assert out.shape == (100,)
    print(f"  EffortNet initial output: mean={out.mean().item():.4f}, "
          f"std={out.std().item():.4f}")
    print("PASS: EffortNet zero init")


def test_completeness_mode():
    """Completeness mode: α_obs = 1 + c · softplus(env_logit), c ∈ (0,1)."""
    m = CLESSONet(
        K_alpha=17, K_env=15, beta_type="transform",
        K_effort=6, effort_hidden=[64, 32], effort_dropout=0.1,
        effort_mode="completeness",
    )
    m.eval()
    z = torch.randn(32, 17)
    w = torch.randn(32, 6)

    alpha_true = m._compute_alpha_env_only(z)  # 1 + softplus(logit)
    alpha_obs = m._compute_alpha(z, w)

    assert (alpha_true >= 1.0).all(), "α_true must be ≥ 1"
    assert (alpha_obs >= 1.0).all(), "α_obs must be ≥ 1 (completeness guarantee)"
    assert (alpha_obs <= alpha_true + 1e-5).all(), (
        "α_obs must be ≤ α_true (completeness can only reduce richness)")

    # Without effort, α should equal α_true
    alpha_no_effort = m._compute_alpha(z, w=None)
    assert torch.allclose(alpha_no_effort, alpha_true), (
        "Without effort covariates, completeness mode should give α_true")
    print(f"  α_true range: [{alpha_true.min():.2f}, {alpha_true.max():.2f}]")
    print(f"  α_obs  range: [{alpha_obs.min():.2f}, {alpha_obs.max():.2f}]")
    print("PASS: Completeness mode")


def test_multiplicative_mode():
    """Multiplicative mode: α_obs = (softplus(logit)+1) · σ(effort_logit)."""
    m = CLESSONet(
        K_alpha=17, K_env=15, beta_type="transform",
        K_effort=6, effort_hidden=[64, 32], effort_dropout=0.1,
        effort_mode="multiplicative",
    )
    m.eval()
    z = torch.randn(32, 17)
    w = torch.randn(32, 6)

    alpha_env = m._compute_alpha_env_only(z)
    alpha_obs = m._compute_alpha(z, w)

    assert (alpha_env >= 1.0).all(), "α_env must be ≥ 1"
    # Multiplicative can go below 1 (no +1 outside product)
    assert (alpha_obs > 0.0).all(), "α_obs must be > 0 in multiplicative mode"
    print(f"  α_env range: [{alpha_env.min():.2f}, {alpha_env.max():.2f}]")
    print(f"  α_obs range: [{alpha_obs.min():.2f}, {alpha_obs.max():.2f}]")
    print("PASS: Multiplicative mode")


def test_gradient_flow():
    """Gradients should flow through effort_net during backprop."""
    m = CLESSONet(
        K_alpha=17, K_env=15, beta_type="transform",
        K_effort=6, effort_hidden=[64, 32], effort_dropout=0.1,
    )
    m.effort_penalty = 0.01

    z_i = torch.randn(8, 17)
    z_j = torch.randn(8, 17)
    w_i = torch.randn(8, 6)
    w_j = torch.randn(8, 6)
    batch = {
        "site_i": torch.arange(8),
        "site_j": torch.arange(8),
        "y": torch.randint(0, 2, (8,)).float(),
        "is_within": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32),
        "weight": torch.ones(8),
        "design_w": torch.ones(8),
        "stratum": torch.zeros(8, dtype=torch.long),
        "env_diff": torch.rand(8, 15),
    }
    S_obs = torch.ones(8) * 50

    result = m.compute_loss(batch, z_i, z_j, S_obs, w_i=w_i, w_j=w_j)
    result["loss"].backward()

    # Check effort_net has gradients
    has_grad = False
    for p in m.effort_net.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "EffortNet should receive gradients"
    print("PASS: Gradient flow through effort_net")


if __name__ == "__main__":
    test_no_effort_backward_compat()
    test_effort_model_construction()
    test_effort_alpha_decomposition()
    test_forward_with_effort()
    test_compute_loss_with_effort()
    test_effort_net_zero_init()
    test_completeness_mode()
    test_multiplicative_mode()
    test_gradient_flow()
    print("\nALL TESTS PASSED")
