"""Tests for ITG turbulence compute functions."""

import numpy as np
import pytest

from desc.examples import get
from desc.grid import LinearGrid


@pytest.mark.unit
def test_gx_reference_quantities():
    """Test GX reference quantities B_ref and L_ref."""
    eq = get("DSHAPE")
    data = eq.compute(["gx_B_reference", "gx_L_reference", "a"])

    # B_ref = 2|ψ_b|/a²
    psi_b = np.abs(eq.Psi) / (2 * np.pi)
    expected_B_ref = 2 * psi_b / data["a"] ** 2

    assert np.isfinite(data["gx_B_reference"])
    assert np.isfinite(data["gx_L_reference"])
    np.testing.assert_allclose(data["gx_B_reference"], expected_B_ref, rtol=1e-10)
    np.testing.assert_allclose(data["gx_L_reference"], data["a"], rtol=1e-10)


@pytest.mark.unit
def test_gx_bmag_order_unity():
    """Test that bmag is O(1) for well-normalized equilibria."""
    for eq_name in ["DSHAPE", "HELIOTRON"]:
        eq = get(eq_name)
        rho = np.linspace(0.1, 1, 5)
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["gx_bmag"], grid=grid)

        assert np.isfinite(data["gx_bmag"]).all(), f"Non-finite bmag for {eq_name}"
        assert np.all(data["gx_bmag"] > 0.1), f"bmag too small for {eq_name}"
        assert np.all(data["gx_bmag"] < 10), f"bmag too large for {eq_name}"


# GX coefficient names used in multiple tests
GX_COEFFICIENT_NAMES = [
    "gx_bmag",
    "gx_gds2",
    "gx_gds21_over_shat",
    "gx_gds22_over_shat_squared",
    "gx_gbdrift",
    "gx_cvdrift",
    "gx_gbdrift0_over_shat",
    "gx_gradpar",
]


@pytest.mark.unit
@pytest.mark.parametrize("eq_name", ["DSHAPE", "HELIOTRON"])
def test_gx_coefficients(eq_name):
    """Test all GX coefficients on tokamak and stellarator equilibria."""
    eq = get(eq_name)
    rho = np.linspace(0.2, 0.8, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

    data = eq.compute(GX_COEFFICIENT_NAMES, grid=grid)

    for name in GX_COEFFICIENT_NAMES:
        assert np.isfinite(data[name]).all(), f"Non-finite values for {name}"

    # Positivity checks for squared quantities
    assert np.all(data["gx_bmag"] > 0)
    assert np.all(data["gx_gds2"] > 0)
    assert np.all(data["gx_gds22_over_shat_squared"] > 0)


@pytest.mark.unit
def test_itg_proxy():
    """Test ITG proxy compute functions."""
    for eq_name in ["DSHAPE", "HELIOTRON"]:
        eq = get(eq_name)
        rho = np.linspace(0.2, 0.8, 3)
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["ITG proxy integrand", "ITG proxy"], grid=grid)

        # Integrand should be finite and positive (sigmoid + 0.2 >= 0.2)
        assert np.isfinite(data["ITG proxy integrand"]).all(), f"Non-finite for {eq_name}"
        assert np.all(data["ITG proxy integrand"] >= 0), f"Negative integrand for {eq_name}"

        # Scalar proxy should be finite and positive
        assert np.isfinite(data["ITG proxy"]), f"Non-finite proxy for {eq_name}"
        assert data["ITG proxy"] > 0, f"Non-positive proxy for {eq_name}"


@pytest.mark.unit
def test_compute_arclength_via_gradpar():
    """Test arclength computation using gradpar."""
    from desc.compute._turbulence import compute_arclength_via_gradpar

    # Test 1: Constant gradpar should give linear arclength
    npoints = 101
    theta_pest = np.linspace(-np.pi, np.pi, npoints)
    gradpar_const = np.ones(npoints) * 2.0  # constant gradpar

    arclength = compute_arclength_via_gradpar(gradpar_const, theta_pest)

    # Arclength should start at 0
    assert arclength[0] == 0.0

    # Arclength should be monotonically increasing (gradpar > 0)
    assert np.all(np.diff(arclength) > 0)

    # For constant gradpar=2, dl/dθ = 1/2, so total length = π
    expected_length = np.pi  # (2π range) / 2
    np.testing.assert_allclose(arclength[-1], expected_length, rtol=1e-3)

    # Test 2: Varying gradpar
    gradpar_var = 1.0 + 0.5 * np.sin(theta_pest)
    arclength_var = compute_arclength_via_gradpar(gradpar_var, theta_pest)

    assert arclength_var[0] == 0.0
    assert np.all(np.diff(arclength_var) > 0)  # Still monotonic
    assert np.isfinite(arclength_var).all()


@pytest.mark.unit
def test_compute_arclength_2d():
    """Test arclength computation with multiple field lines."""
    from desc.compute._turbulence import compute_arclength_via_gradpar

    npoints = 51
    num_alpha = 3
    theta_pest = np.linspace(-np.pi, np.pi, npoints)

    # Different gradpar for each field line
    gradpar_2d = np.ones((npoints, num_alpha))
    gradpar_2d[:, 0] = 1.0
    gradpar_2d[:, 1] = 2.0
    gradpar_2d[:, 2] = 0.5

    arclength_2d = compute_arclength_via_gradpar(gradpar_2d, theta_pest)

    # Shape should be preserved
    assert arclength_2d.shape == (npoints, num_alpha)

    # Each field line should start at 0
    np.testing.assert_array_equal(arclength_2d[0, :], 0.0)

    # Each field line should be monotonically increasing
    for a in range(num_alpha):
        assert np.all(np.diff(arclength_2d[:, a]) > 0)

    # Field line with larger gradpar should have shorter total length
    # (dl/dθ = 1/gradpar)
    assert arclength_2d[-1, 1] < arclength_2d[-1, 0] < arclength_2d[-1, 2]


@pytest.mark.unit
def test_resample_to_uniform_arclength():
    """Test resampling to uniform arclength grid."""
    from desc.compute._turbulence import (
        compute_arclength_via_gradpar,
        resample_to_uniform_arclength,
    )

    # Create test data with known properties
    npoints_in = 201
    npoints_out = 96
    theta_pest = np.linspace(-np.pi, np.pi, npoints_in)

    # Varying gradpar creates non-uniform arclength
    gradpar = 1.0 + 0.3 * np.cos(theta_pest)
    arclength = compute_arclength_via_gradpar(gradpar, theta_pest)

    # Create test data: smooth functions
    nfeatures = 3
    data = np.zeros((nfeatures, npoints_in))
    data[0] = np.sin(theta_pest)  # Oscillating
    data[1] = theta_pest / np.pi  # Linear
    data[2] = np.ones(npoints_in)  # Constant

    z_uniform, data_uniform = resample_to_uniform_arclength(arclength, data, npoints_out)

    # Output shape checks
    assert z_uniform.shape == (npoints_out,)
    assert data_uniform.shape == (nfeatures, npoints_out)

    # z should be uniformly spaced in [-pi, pi)
    dz = z_uniform[1] - z_uniform[0]
    np.testing.assert_allclose(np.diff(z_uniform), dz, rtol=1e-10)

    # Constant function should remain constant after resampling
    np.testing.assert_allclose(data_uniform[2], 1.0, rtol=1e-3)

    # All output should be finite
    assert np.isfinite(data_uniform).all()


@pytest.mark.unit
def test_solve_poloidal_turns_for_length():
    """Test solving for poloidal turns to achieve target length."""
    from desc.compute._turbulence import solve_poloidal_turns_for_length

    # Create a simple length function: length = C * poloidal_turns
    # where C depends on gradpar and theta range
    # For gradpar=1: dl/dθ = 1, and θ range = 2*pi * poloidal_turns
    # So length = 2*pi * poloidal_turns
    def simple_length_fn(poloidal_turns):
        return 2 * np.pi * poloidal_turns

    # Solve for poloidal_turns = target_length / (2*pi)
    target_length = 10.0
    expected_turns = target_length / (2 * np.pi)

    poloidal_turns = solve_poloidal_turns_for_length(
        simple_length_fn, target_length, x0_guess=expected_turns
    )

    np.testing.assert_allclose(poloidal_turns, expected_turns, rtol=1e-6)


@pytest.mark.unit
def test_solve_poloidal_turns_nonlinear():
    """Test solving for poloidal turns with nonlinear length function."""
    from desc.compute._turbulence import solve_poloidal_turns_for_length

    # Nonlinear length function: length = poloidal_turns^2 + poloidal_turns
    def nonlinear_length_fn(poloidal_turns):
        return poloidal_turns**2 + poloidal_turns

    # Solve for target = 6: poloidal_turns^2 + poloidal_turns = 6
    # (poloidal_turns - 2)(poloidal_turns + 3) = 0 -> poloidal_turns = 2
    target_length = 6.0
    expected_turns = 2.0

    poloidal_turns = solve_poloidal_turns_for_length(
        nonlinear_length_fn, target_length, x0_guess=1.5
    )

    np.testing.assert_allclose(poloidal_turns, expected_turns, rtol=1e-6)


@pytest.mark.unit
def test_solve_poloidal_turns_error():
    """Test that solver raises error when target is out of bracket."""
    from desc.compute._turbulence import solve_poloidal_turns_for_length

    # Length function that saturates: length = tanh(poloidal_turns)
    # Max achievable length ~ 1
    def saturating_length_fn(poloidal_turns):
        return np.tanh(poloidal_turns)

    # Target length > max achievable should raise ValueError
    with pytest.raises(ValueError, match="Could not find poloidal_turns"):
        solve_poloidal_turns_for_length(saturating_length_fn, target_length=2.0)


# =============================================================================
# BatchNorm Tests
# =============================================================================


@pytest.mark.unit
def test_batch_norm_1d():
    """Test BatchNorm1d function correctness."""
    from desc.backend import jnp
    from desc.compute._turbulence import _batch_norm_1d

    batch, channels, length = 2, 4, 10
    x = jnp.ones((batch, channels, length)) * 2.0  # All values are 2.0
    gamma = jnp.ones(channels) * 0.5  # Scale by 0.5
    beta = jnp.ones(channels) * 1.0  # Shift by 1.0
    running_mean = jnp.ones(channels) * 1.0  # Mean is 1.0
    running_var = jnp.ones(channels) * 0.25  # Variance is 0.25

    output = _batch_norm_1d(x, gamma, beta, running_mean, running_var)

    # Expected: scale * (x - mean) / sqrt(var + eps) + shift
    # = 0.5 * (2.0 - 1.0) / sqrt(0.25 + 1e-5) + 1.0
    # ≈ 0.5 * 1.0 / 0.5 + 1.0 = 1.0 + 1.0 = 2.0
    expected = jnp.ones((batch, channels, length)) * 2.0
    np.testing.assert_allclose(np.array(output), np.array(expected), rtol=1e-5)

    # Test with different values per channel
    gamma = jnp.array([1.0, 2.0, 3.0, 4.0])
    beta = jnp.array([0.1, 0.2, 0.3, 0.4])
    running_mean = jnp.array([0.5, 1.0, 1.5, 2.0])
    running_var = jnp.array([0.25, 0.5, 0.75, 1.0])

    output2 = _batch_norm_1d(x, gamma, beta, running_mean, running_var)
    assert output2.shape == (batch, channels, length)
    assert np.isfinite(np.array(output2)).all()


@pytest.mark.unit
def test_cyclic_invariant_forward_with_batchnorm():
    """Test CNN forward pass with BatchNorm weights and detection logic."""
    from desc.backend import jnp
    from desc.compute._turbulence import _cyclic_invariant_forward, _has_batch_norm

    # Test BatchNorm detection logic
    weights_no_bn = {
        "conv_layers.0.weight": None,
        "conv_layers.0.bias": None,
    }
    assert not _has_batch_norm(weights_no_bn)

    weights_old_bn = {
        "conv_layers.0.weight": None,
        "conv_layers.0.bn.weight": None,
    }
    assert _has_batch_norm(weights_old_bn)

    weights_new_bn = {
        "conv_layers.0.weight": None,
        "conv_batch_norms.0.weight": None,
    }
    assert _has_batch_norm(weights_new_bn)

    # Create weights with BatchNorm
    np.random.seed(42)
    conv_channels = [7, 16, 32, 16, 32, 16]
    kernel_sizes = [3, 3, 3, 3, 3]
    fc_dims = [16 + 2, 32, 16]

    weights_bn = {}
    for i in range(5):
        weights_bn[f"conv_layers.{i}.weight"] = jnp.array(
            np.random.randn(conv_channels[i + 1], conv_channels[i], kernel_sizes[i]) * 0.1
        ).astype(jnp.float32)
        weights_bn[f"conv_layers.{i}.bias"] = jnp.zeros(conv_channels[i + 1], dtype=jnp.float32)
        # Add BatchNorm parameters
        weights_bn[f"conv_batch_norms.{i}.weight"] = jnp.ones(conv_channels[i + 1], dtype=jnp.float32)
        weights_bn[f"conv_batch_norms.{i}.bias"] = jnp.zeros(conv_channels[i + 1], dtype=jnp.float32)
        weights_bn[f"conv_batch_norms.{i}.running_mean"] = jnp.zeros(conv_channels[i + 1], dtype=jnp.float32)
        weights_bn[f"conv_batch_norms.{i}.running_var"] = jnp.ones(conv_channels[i + 1], dtype=jnp.float32)

    weights_bn["fc_layers.0.weight"] = jnp.array(
        np.random.randn(fc_dims[1], fc_dims[0]) * 0.1
    ).astype(jnp.float32)
    weights_bn["fc_layers.0.bias"] = jnp.zeros(fc_dims[1], dtype=jnp.float32)
    weights_bn["fc_batch_norms.0.weight"] = jnp.ones(fc_dims[1], dtype=jnp.float32)
    weights_bn["fc_batch_norms.0.bias"] = jnp.zeros(fc_dims[1], dtype=jnp.float32)
    weights_bn["fc_batch_norms.0.running_mean"] = jnp.zeros(fc_dims[1], dtype=jnp.float32)
    weights_bn["fc_batch_norms.0.running_var"] = jnp.ones(fc_dims[1], dtype=jnp.float32)

    weights_bn["fc_layers.1.weight"] = jnp.array(
        np.random.randn(fc_dims[2], fc_dims[1]) * 0.1
    ).astype(jnp.float32)
    weights_bn["fc_layers.1.bias"] = jnp.zeros(fc_dims[2], dtype=jnp.float32)
    weights_bn["fc_batch_norms.1.weight"] = jnp.ones(fc_dims[2], dtype=jnp.float32)
    weights_bn["fc_batch_norms.1.bias"] = jnp.zeros(fc_dims[2], dtype=jnp.float32)
    weights_bn["fc_batch_norms.1.running_mean"] = jnp.zeros(fc_dims[2], dtype=jnp.float32)
    weights_bn["fc_batch_norms.1.running_var"] = jnp.ones(fc_dims[2], dtype=jnp.float32)

    weights_bn["output_layer.weight"] = jnp.array(
        np.random.randn(1, fc_dims[2]) * 0.1
    ).astype(jnp.float32)
    weights_bn["output_layer.bias"] = jnp.zeros(1, dtype=jnp.float32)

    # Verify BatchNorm is detected
    assert _has_batch_norm(weights_bn)

    # Test forward pass
    batch = 2
    signals = jnp.array(np.random.randn(batch, 7, 96)).astype(jnp.float32)
    scalars = jnp.array(np.random.randn(batch, 2)).astype(jnp.float32)

    output = _cyclic_invariant_forward(signals, scalars, weights_bn)

    assert output.shape == (batch, 1)
    assert np.isfinite(np.array(output)).all()


@pytest.mark.unit
def test_cyclic_invariant_forward_shape():
    """Test CNN forward pass produces correct output shape."""
    from desc.backend import jnp
    from desc.compute._turbulence import _cyclic_invariant_forward

    # Use helper function to create mock weights
    weights = _make_mock_weights()

    # Test forward pass
    batch = 3
    np.random.seed(42)
    signals = jnp.array(np.random.randn(batch, 7, 96)).astype(jnp.float32)
    scalars = jnp.array(np.random.randn(batch, 2)).astype(jnp.float32)

    output = _cyclic_invariant_forward(signals, scalars, weights)

    assert output.shape == (batch, 1)
    assert np.isfinite(np.array(output)).all()


@pytest.mark.slow
def test_jax_vs_pytorch_forward_pass():
    """Test that JAX forward pass matches PyTorch CyclicInvariantNet output.

    Runs in subprocess due to JAX/PyTorch conflicts on some systems.
    """
    import subprocess
    import sys
    import textwrap

    pytest.importorskip("torch")

    script = textwrap.dedent("""
        import numpy as np
        import torch
        import torch.nn as nn
        from desc.backend import jnp
        from desc.compute._turbulence import _cyclic_invariant_forward

        class CyclicNet(nn.Module):
            def __init__(self):
                super().__init__()
                ch = [7, 16, 32, 16, 32, 16]
                self.convs = nn.ModuleList([
                    nn.Conv1d(ch[i], ch[i+1], 3, padding="same", padding_mode="circular")
                    for i in range(5)
                ])
                self.pools = nn.ModuleList([nn.MaxPool1d(2, 2) for _ in range(5)])
                self.fc1, self.fc2, self.out = nn.Linear(18, 32), nn.Linear(32, 16), nn.Linear(16, 1)

            def forward(self, x, s):
                x = x.transpose(1, 2)
                for conv, pool in zip(self.convs, self.pools):
                    x = pool(torch.relu(conv(x)))
                return self.out(torch.relu(self.fc2(torch.relu(self.fc1(torch.cat([x.mean(-1), s], -1))))))

        torch.manual_seed(42)
        np.random.seed(42)
        model = CyclicNet().eval()
        sd = model.state_dict()

        w = {}
        for i in range(5):
            w[f"conv_layers.{i}.weight"] = jnp.array(sd[f"convs.{i}.weight"].numpy())
            w[f"conv_layers.{i}.bias"] = jnp.array(sd[f"convs.{i}.bias"].numpy())
        w["fc_layers.0.weight"], w["fc_layers.0.bias"] = jnp.array(sd["fc1.weight"].numpy()), jnp.array(sd["fc1.bias"].numpy())
        w["fc_layers.1.weight"], w["fc_layers.1.bias"] = jnp.array(sd["fc2.weight"].numpy()), jnp.array(sd["fc2.bias"].numpy())
        w["output_layer.weight"], w["output_layer.bias"] = jnp.array(sd["out.weight"].numpy()), jnp.array(sd["out.bias"].numpy())

        sig, sca = np.random.randn(2, 96, 7).astype(np.float32), np.random.randn(2, 2).astype(np.float32)
        with torch.no_grad():
            y_pt = model(torch.from_numpy(sig), torch.from_numpy(sca)).numpy()
        y_jax = np.array(_cyclic_invariant_forward(jnp.array(sig.transpose(0,2,1)), jnp.array(sca), w))
        assert np.allclose(y_jax, y_pt, rtol=1e-4, atol=1e-4), f"max_diff={np.max(np.abs(y_jax-y_pt))}"
    """)

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        pytest.fail(f"JAX vs PyTorch mismatch:\n{result.stdout}\n{result.stderr}")


@pytest.mark.unit
def test_ensemble_forward():
    """Test ensemble forward pass averages predictions correctly."""
    from desc.backend import jnp
    from desc.compute._turbulence import _cyclic_invariant_forward, _ensemble_forward

    np.random.seed(42)

    # Create two sets of weights with different values
    def make_weights():
        conv_channels = [7, 8, 8, 8, 8, 8]
        kernel_sizes = [3, 3, 3, 3, 3]
        fc_dims = [8 + 2, 8, 4]
        weights = {}
        for i in range(5):
            weights[f"conv_layers.{i}.weight"] = jnp.array(
                np.random.randn(conv_channels[i + 1], conv_channels[i], kernel_sizes[i])
            ).astype(jnp.float32)
            weights[f"conv_layers.{i}.bias"] = jnp.array(
                np.random.randn(conv_channels[i + 1])
            ).astype(jnp.float32)
        weights["fc_layers.0.weight"] = jnp.array(np.random.randn(fc_dims[1], fc_dims[0])).astype(
            jnp.float32
        )
        weights["fc_layers.0.bias"] = jnp.array(np.random.randn(fc_dims[1])).astype(jnp.float32)
        weights["fc_layers.1.weight"] = jnp.array(np.random.randn(fc_dims[2], fc_dims[1])).astype(
            jnp.float32
        )
        weights["fc_layers.1.bias"] = jnp.array(np.random.randn(fc_dims[2])).astype(jnp.float32)
        weights["output_layer.weight"] = jnp.array(np.random.randn(1, fc_dims[2])).astype(
            jnp.float32
        )
        weights["output_layer.bias"] = jnp.array(np.random.randn(1)).astype(jnp.float32)
        return weights

    weights1 = make_weights()
    weights2 = make_weights()
    weights_list = [weights1, weights2]

    # Test input
    batch = 2
    signals = jnp.array(np.random.randn(batch, 7, 96)).astype(jnp.float32)
    scalars = jnp.array(np.random.randn(batch, 2)).astype(jnp.float32)

    # Ensemble forward
    ensemble_output = _ensemble_forward(signals, scalars, weights_list)

    # Individual forward passes
    out1 = _cyclic_invariant_forward(signals, scalars, weights1)
    out2 = _cyclic_invariant_forward(signals, scalars, weights2)
    expected_mean = (np.array(out1) + np.array(out2)) / 2

    np.testing.assert_allclose(np.array(ensemble_output), expected_mean, rtol=1e-6)

    # Test return_std=True
    mean_output, std_output = _ensemble_forward(
        signals, scalars, weights_list, return_std=True
    )
    expected_std = np.std([np.array(out1), np.array(out2)], axis=0)
    np.testing.assert_allclose(np.array(mean_output), expected_mean, rtol=1e-6)
    np.testing.assert_allclose(np.array(std_output), expected_std, rtol=1e-6)
    assert std_output.shape == mean_output.shape


@pytest.mark.unit
def test_jit_forward_matches_original():
    """Test JIT-compiled forward matches non-JIT version."""
    from desc.backend import jnp
    from desc.compute._turbulence import (
        _cyclic_invariant_forward,
        _make_jit_forward,
    )

    np.random.seed(42)

    # Create mock weights
    conv_channels = [7, 8, 8, 8, 8, 8]
    kernel_sizes = [3, 3, 3, 3, 3]
    fc_dims = [8 + 2, 8, 4]
    weights = {}
    for i in range(5):
        weights[f"conv_layers.{i}.weight"] = jnp.array(
            np.random.randn(conv_channels[i + 1], conv_channels[i], kernel_sizes[i])
        ).astype(jnp.float32)
        weights[f"conv_layers.{i}.bias"] = jnp.array(
            np.random.randn(conv_channels[i + 1])
        ).astype(jnp.float32)
    weights["fc_layers.0.weight"] = jnp.array(
        np.random.randn(fc_dims[1], fc_dims[0])
    ).astype(jnp.float32)
    weights["fc_layers.0.bias"] = jnp.array(np.random.randn(fc_dims[1])).astype(
        jnp.float32
    )
    weights["fc_layers.1.weight"] = jnp.array(
        np.random.randn(fc_dims[2], fc_dims[1])
    ).astype(jnp.float32)
    weights["fc_layers.1.bias"] = jnp.array(np.random.randn(fc_dims[2])).astype(
        jnp.float32
    )
    weights["output_layer.weight"] = jnp.array(
        np.random.randn(1, fc_dims[2])
    ).astype(jnp.float32)
    weights["output_layer.bias"] = jnp.array(np.random.randn(1)).astype(jnp.float32)

    # Create JIT-compiled forward function
    jit_forward = _make_jit_forward(weights)

    # Test input
    signals = jnp.array(np.random.randn(2, 7, 96)).astype(jnp.float32)
    scalars = jnp.array(np.random.randn(2, 2)).astype(jnp.float32)

    # Compare outputs
    original = _cyclic_invariant_forward(signals, scalars, weights)
    jitted = jit_forward(signals, scalars)

    np.testing.assert_allclose(np.array(original), np.array(jitted), rtol=1e-5)


# =============================================================================
# NNITGProxy Objective Tests
# =============================================================================

# Use nz_internal=101 for all NNITGProxy tests (vs 1001 default).
# This exercises the full code path (arclength, resampling, CNN) at ~10x less cost.
_TEST_NZ_INTERNAL = 101


def _make_mock_weights():
    """Create mock CNN weights for testing."""
    from desc.backend import jnp

    np.random.seed(42)
    conv_channels = [7, 16, 32, 16, 32, 16]
    kernel_sizes = [3, 3, 3, 3, 3]
    fc_dims = [16 + 2, 32, 16]  # +2 for scalars

    weights = {}
    for i in range(5):
        weights[f"conv_layers.{i}.weight"] = jnp.array(
            np.random.randn(conv_channels[i + 1], conv_channels[i], kernel_sizes[i]) * 0.1
        ).astype(jnp.float32)
        weights[f"conv_layers.{i}.bias"] = jnp.zeros(conv_channels[i + 1], dtype=jnp.float32)

    weights["fc_layers.0.weight"] = jnp.array(
        np.random.randn(fc_dims[1], fc_dims[0]) * 0.1
    ).astype(jnp.float32)
    weights["fc_layers.0.bias"] = jnp.zeros(fc_dims[1], dtype=jnp.float32)
    weights["fc_layers.1.weight"] = jnp.array(
        np.random.randn(fc_dims[2], fc_dims[1]) * 0.1
    ).astype(jnp.float32)
    weights["fc_layers.1.bias"] = jnp.zeros(fc_dims[2], dtype=jnp.float32)
    weights["output_layer.weight"] = jnp.array(
        np.random.randn(1, fc_dims[2]) * 0.1
    ).astype(jnp.float32)
    weights["output_layer.bias"] = jnp.zeros(1, dtype=jnp.float32)
    return weights


def _make_mock_weights_tuple():
    """Create mock weights and JIT-compiled forward function for testing."""
    from desc.compute._turbulence import _make_jit_forward

    weights = _make_mock_weights()
    jit_forward = _make_jit_forward(weights)
    return weights, jit_forward


@pytest.fixture(scope="module")
def dshape_eq():
    """Module-scoped DSHAPE equilibrium (loaded once, shared across tests)."""
    return get("DSHAPE")


@pytest.fixture(scope="module")
def single_model_obj(dshape_eq):
    """Built NNITGProxy with 2 rho, 2 alpha, single model, nz_internal=101."""
    pytest.importorskip("torch")
    from unittest.mock import patch

    from desc.objectives._turbulence import NNITGProxy

    mock_weights_tuple = _make_mock_weights_tuple()
    with patch(
        "desc.objectives._turbulence._load_nn_weights",
        return_value=mock_weights_tuple,
    ):
        obj = NNITGProxy(
            dshape_eq, rho=[0.4, 0.6], alpha=[0, np.pi / 2],
            model_path="/fake/path.pt", nz_internal=_TEST_NZ_INTERNAL,
        )
        obj.build(use_jit=False, verbose=0)
    return obj


@pytest.fixture(scope="module")
def ensemble_obj(dshape_eq):
    """Built NNITGProxy with 2 rho, 2 alpha, ensemble of 2 models, nz_internal=101."""
    pytest.importorskip("torch")
    from unittest.mock import patch

    from desc.compute._turbulence import _make_jit_forward
    from desc.objectives._turbulence import NNITGProxy

    mock_weights = _make_mock_weights()
    np.random.seed(123)
    mock_weights2 = _make_mock_weights()
    jit_forward1 = _make_jit_forward(mock_weights)
    jit_forward2 = _make_jit_forward(mock_weights2)

    with patch(
        "desc.objectives._turbulence._load_ensemble_weights_cached",
        return_value=([mock_weights, mock_weights2], [jit_forward1, jit_forward2]),
    ):
        obj = NNITGProxy(
            dshape_eq, rho=[0.4, 0.6], alpha=[0, np.pi / 2],
            ensemble_dir="/fake/dir", ensemble_csv="/fake/results.csv",
            nz_internal=_TEST_NZ_INTERNAL,
        )
        obj.build(use_jit=False, verbose=0)
    return obj


@pytest.mark.unit
def test_nnitgproxy_init(dshape_eq):
    """Test NNITGProxy construction and parameter validation."""
    pytest.importorskip("torch")
    from desc.objectives._turbulence import NNITGProxy

    # Test basic construction with mock model_path
    obj = NNITGProxy(dshape_eq, rho=0.5, model_path="/fake/path.pt")
    assert obj._npoints == 96
    assert obj._nz_internal == 1001
    assert obj._target_flux_tube_length == 75.4
    assert obj._a_over_LT == 3.0
    assert obj._a_over_Ln == 1.0
    np.testing.assert_array_equal(obj._rho, [0.5])

    # Test with custom parameters
    obj2 = NNITGProxy(
        dshape_eq,
        rho=[0.3, 0.5, 0.7],
        alpha=[0, np.pi],
        npoints=64,
        nz_internal=501,
        target_flux_tube_length=62.8,
        a_over_LT=2.5,
        a_over_Ln=0.5,
        model_path="/fake/path.pt",
    )
    np.testing.assert_array_equal(obj2._rho, [0.3, 0.5, 0.7])
    np.testing.assert_array_equal(obj2._alpha, [0, np.pi])
    assert obj2._npoints == 64
    assert obj2._target_flux_tube_length == 62.8

    # Test error when ensemble_dir specified without ensemble_csv
    with pytest.raises(ValueError, match="ensemble_csv must be specified"):
        NNITGProxy(dshape_eq, rho=0.5, ensemble_dir="/fake/dir")


@pytest.mark.unit
def test_nnitgproxy_build(dshape_eq):
    """Test NNITGProxy build method with mock weights."""
    pytest.importorskip("torch")
    from unittest.mock import patch

    from desc.compute._turbulence import _make_jit_forward
    from desc.objectives._turbulence import NNITGProxy

    mock_weights = _make_mock_weights()
    mock_jit_forward = _make_jit_forward(mock_weights)

    with patch(
        "desc.objectives._turbulence._load_nn_weights",
        return_value=(mock_weights, mock_jit_forward),
    ):
        obj = NNITGProxy(
            dshape_eq, rho=0.5, model_path="/fake/path.pt",
            nz_internal=_TEST_NZ_INTERNAL,
        )
        obj.build(use_jit=False, verbose=0)

    # Check constants are populated
    assert "rho" in obj.constants
    assert "alpha" in obj.constants
    assert "toroidal_turns" in obj.constants
    assert "target_length" in obj.constants
    assert "a" in obj.constants
    assert hasattr(obj, "_models") and len(obj._models) > 0
    assert "a_over_LT" in obj.constants
    assert "a_over_Ln" in obj.constants

    # Check values are reasonable
    assert obj.constants["toroidal_turns"] > 0
    assert obj.constants["target_length"] > 0
    assert obj.constants["a"] > 0
    assert len(obj._models) == 1
    assert callable(obj._models[0])

    # Check dimension
    assert obj._dim_f == 1  # Single rho value


@pytest.mark.unit
def test_nnitgproxy_compute(single_model_obj):
    """Test NNITGProxy compute: basic, return_signals, return_per_alpha, both."""
    pytest.importorskip("torch")
    obj = single_model_obj
    params = obj.things[0].params_dict
    num_rho, num_alpha = 2, 2

    # --- Basic output ---
    Q = obj.compute(params)
    assert Q.shape == (num_rho,)
    assert np.isfinite(Q).all()
    assert np.all(Q > 0)

    # --- return_signals ---
    Q_sig, signals_info = obj.compute(params, return_signals=True)
    assert Q_sig.shape == (num_rho,)
    assert np.isfinite(Q_sig).all()
    assert "z" in signals_info
    assert "signals" in signals_info
    assert "feature_names" in signals_info
    z = signals_info["z"]
    assert z.shape == (96,)
    np.testing.assert_allclose(z[0], -np.pi, rtol=1e-5)
    signals = signals_info["signals"]
    assert signals.shape == (num_rho, num_alpha, 7, 96)
    assert np.isfinite(signals).all()
    assert len(signals_info["feature_names"]) == 7
    assert signals_info["feature_names"][0] == "bmag"

    # --- return_per_alpha ---
    Q_pa = obj.compute(params, return_per_alpha=True)
    assert Q_pa.shape == (num_rho, num_alpha)
    assert np.isfinite(Q_pa).all()
    assert np.all(Q_pa > 0)
    # Mean over alpha should match default result
    np.testing.assert_allclose(np.mean(Q_pa, axis=-1), Q, rtol=1e-5)

    # --- return_per_alpha + return_signals ---
    Q_both, sig_both = obj.compute(
        params, return_signals=True, return_per_alpha=True
    )
    assert Q_both.shape == (num_rho, num_alpha)
    assert sig_both["signals"].shape == (num_rho, num_alpha, 7, 96)
    assert np.isfinite(sig_both["signals"]).all()


@pytest.mark.unit
def test_nnitgproxy_solve_length(dshape_eq):
    """Test NNITGProxy compute with solve_length_at_compute=True path."""
    pytest.importorskip("torch")
    from unittest.mock import patch

    from desc.objectives._turbulence import NNITGProxy

    mock_weights_tuple = _make_mock_weights_tuple()

    with patch(
        "desc.objectives._turbulence._load_nn_weights",
        return_value=mock_weights_tuple,
    ):
        obj = NNITGProxy(
            dshape_eq, rho=[0.5], alpha=[0], model_path="/fake/path.pt",
            solve_length_at_compute=True, nz_internal=_TEST_NZ_INTERNAL,
        )
        obj.build(use_jit=False, verbose=0)

    Q = obj.compute(obj.things[0].params_dict)
    assert Q.shape == (1,)
    assert np.isfinite(Q).all()
    assert Q[0] > 0


@pytest.mark.unit
def test_nnitgproxy_ensemble_std(ensemble_obj):
    """Test return_std with ensemble: basic, per_alpha, signals."""
    pytest.importorskip("torch")
    obj = ensemble_obj
    params = obj.things[0].params_dict
    num_rho, num_alpha = 2, 2

    # --- return_std basic ---
    Q, Q_std = obj.compute(params, return_std=True)
    assert Q.shape == (num_rho,)
    assert Q_std.shape == (num_rho,)
    assert np.isfinite(Q).all()
    assert np.isfinite(Q_std).all()
    assert np.all(Q > 0)
    assert np.all(Q_std >= 0)

    # --- return_std + return_per_alpha ---
    Q_pa, Q_std_pa = obj.compute(params, return_std=True, return_per_alpha=True)
    assert Q_pa.shape == (num_rho, num_alpha)
    assert Q_std_pa.shape == (num_rho, num_alpha)

    # --- return_std + return_signals ---
    Q_sig, Q_std_sig, signals_info = obj.compute(
        params, return_std=True, return_signals=True
    )
    assert Q_sig.shape == (num_rho,)
    assert Q_std_sig.shape == (num_rho,)
    assert "signals" in signals_info


@pytest.mark.unit
def test_nnitgproxy_single_model_std(single_model_obj):
    """Test return_std with single model returns zero std."""
    pytest.importorskip("torch")
    obj = single_model_obj
    Q, Q_std = obj.compute(obj.things[0].params_dict, return_std=True)
    assert Q.shape == (2,)
    assert Q_std.shape == (2,)
    np.testing.assert_array_equal(Q_std, 0)
