"""Quick smoke tests for TransformBetaNet."""
import torch
from src.clesso_nn.model import TransformBetaNet, CLESSONet


def test_creation():
    tbn = TransformBetaNet(input_dim=5, n_transform_knots=16, n_g_knots=8)
    assert tbn.K_env == 5
    assert len(tbn.transform_nets) == 5
    assert len(tbn.weight_nets) == 5
    print("PASS: creation")


def test_forward():
    tbn = TransformBetaNet(input_dim=5, n_transform_knots=16, n_g_knots=8)
    B = 32
    env_i = torch.randn(B, 5)
    env_j = torch.randn(B, 5)
    eta = tbn(env_i, env_j)
    assert eta.shape == (B,), f"Expected ({B},), got {eta.shape}"
    assert (eta >= 0).all(), "eta has negative values!"
    assert (eta <= 10).all(), "eta exceeds eta_max!"
    print(f"PASS: forward, eta range [{eta.min().item():.4f}, {eta.max().item():.4f}]")


def test_same_site():
    """Same-site pairs in eval mode: all T_k contributions should be 0 (before sigmoid)."""
    tbn = TransformBetaNet(input_dim=5, n_transform_knots=16, n_g_knots=8)
    tbn.eval()  # disable dropout
    env_same = torch.randn(32, 5)
    with torch.no_grad():
        eta = tbn(env_same, env_same)
    # All per-dim contributions g_k(|T(x)-T(x)|) - g_k(0) = g_k(0) - g_k(0) = 0
    # But sigmoid_clamp(0, shift) ≈ 10*σ(-shift) > 0.
    # This is fine: within-site match probability is 1/α, not exp(-η).
    # Verify all pairs get the same eta (since contributions are all 0).
    assert (eta.max() - eta.min()).item() < 1e-5, \
        f"Same-site etas should be identical, spread = {(eta.max()-eta.min()).item():.6f}"
    print(f"PASS: same-site, eta = {eta[0].item():.4f} (uniform, sigmoid floor)")


def test_transform_site():
    tbn = TransformBetaNet(input_dim=5, n_transform_knots=16, n_g_knots=8)
    env = torch.randn(10, 5)
    transformed = tbn.transform_site(env)
    assert transformed.shape == (10, 5)
    print("PASS: transform_site")


def test_clessonet_integration():
    model = CLESSONet(K_alpha=8, K_env=5, beta_type="transform",
                      transform_n_knots=16, transform_g_knots=8)
    assert isinstance(model.beta_net, TransformBetaNet)

    B = 32
    z_i = torch.randn(B, 8)
    z_j = torch.randn(B, 8)
    env_i = torch.randn(B, 5)
    env_j = torch.randn(B, 5)
    env_diff = torch.abs(env_i - env_j)
    is_within = torch.zeros(B)

    fwd = model.forward(z_i, z_j, env_diff, is_within, env_i=env_i, env_j=env_j)
    assert fwd["eta"].shape == (B,)
    assert fwd["p_match"].shape == (B,)
    print(f"PASS: CLESSONet forward, eta range [{fwd['eta'].min().item():.4f}, {fwd['eta'].max().item():.4f}]")


def test_gradients():
    tbn = TransformBetaNet(input_dim=5, n_transform_knots=16, n_g_knots=8)
    env_i = torch.randn(4, 5, requires_grad=True)
    env_j = torch.randn(4, 5, requires_grad=True)
    eta = tbn(env_i, env_j)
    eta.sum().backward()
    assert env_i.grad is not None
    assert (env_i.grad != 0).any()
    print("PASS: gradients flow")


def test_geo_dist():
    tbn = TransformBetaNet(input_dim=5, n_transform_knots=16, n_g_knots=8, n_geo_dims=1)
    B = 32
    env_i = torch.randn(B, 5)
    env_j = torch.randn(B, 5)
    geo_dist = torch.rand(B, 1)
    eta = tbn(env_i, env_j, geo_dist=geo_dist)
    assert eta.shape == (B,)
    print("PASS: geo_dist")


def test_monotonicity():
    """T_k monotonicity: increasing input → non-decreasing output."""
    tbn = TransformBetaNet(input_dim=3, n_transform_knots=16, n_g_knots=8)
    tbn.eval()  # disable dropout for deterministic monotonicity check
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    with torch.no_grad():
        for k in range(3):
            t_k = tbn.transform_nets[k](x).squeeze(-1)
            diffs = torch.diff(t_k)
            assert (diffs >= -1e-6).all(), f"T_{k} not monotone!"
    print("PASS: T_k monotonicity")


if __name__ == "__main__":
    test_creation()
    test_forward()
    test_same_site()
    test_transform_site()
    test_clessonet_integration()
    test_gradients()
    test_geo_dist()
    test_monotonicity()
    print("\nALL TESTS PASSED")
