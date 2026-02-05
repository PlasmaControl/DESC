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
def test_itg_proxy_objective_import():
    """Test that ITGProxy objective can be imported."""
    from desc.objectives import ITGProxy

    eq = get("DSHAPE")
    obj = ITGProxy(eq, rho=0.5, nturns=1, nzetaperturn=20)
    assert obj._nturns == 1
    assert obj._nzetaperturn == 20


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
# CNN Layer Primitives Tests
# =============================================================================


@pytest.mark.unit
def test_circular_pad_1d():
    """Test circular padding wraps values correctly."""
    from desc.backend import jnp
    from desc.compute._turbulence import _circular_pad_1d

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    padded = _circular_pad_1d(x, pad_left=2, pad_right=2)

    # Expected: [4, 5, 1, 2, 3, 4, 5, 1, 2]
    expected = jnp.array([4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0])
    np.testing.assert_array_equal(np.array(padded), np.array(expected))


@pytest.mark.unit
def test_circular_pad_1d_batched():
    """Test circular padding works with batched inputs."""
    from desc.backend import jnp
    from desc.compute._turbulence import _circular_pad_1d

    # Shape (batch=2, channels=3, length=4)
    x = jnp.arange(24).reshape(2, 3, 4).astype(float)
    padded = _circular_pad_1d(x, pad_left=1, pad_right=1)

    # Shape should be (2, 3, 6)
    assert padded.shape == (2, 3, 6)

    # Check wrapping for first batch, first channel: [0,1,2,3] -> [3,0,1,2,3,0]
    np.testing.assert_array_equal(np.array(padded[0, 0, :]), [3, 0, 1, 2, 3, 0])


@pytest.mark.unit
def test_conv1d_circular_shape():
    """Test conv1d with circular padding produces correct output shape."""
    from desc.backend import jnp
    from desc.compute._turbulence import _conv1d_circular

    batch, in_ch, length = 2, 7, 96
    out_ch, kernel_size = 16, 3

    x = jnp.ones((batch, in_ch, length))
    weight = jnp.ones((out_ch, in_ch, kernel_size))
    bias = jnp.zeros(out_ch)

    out = _conv1d_circular(x, weight, bias, kernel_size)

    # With circular padding='same', output length should equal input length
    assert out.shape == (batch, out_ch, length)


@pytest.mark.unit
def test_max_pool_1d():
    """Test max pooling reduces spatial dimension correctly."""
    from desc.backend import jnp
    from desc.compute._turbulence import _max_pool_1d

    batch, channels, length = 2, 16, 96
    x = jnp.arange(batch * channels * length).reshape(batch, channels, length).astype(float)

    pooled = _max_pool_1d(x, pool_size=2, stride=2)

    # Length should be halved
    assert pooled.shape == (batch, channels, length // 2)

    # After 5 pooling layers: 96 -> 48 -> 24 -> 12 -> 6 -> 3
    x_five_pools = x
    for _ in range(5):
        x_five_pools = _max_pool_1d(x_five_pools)
    assert x_five_pools.shape == (batch, channels, 3)


@pytest.mark.unit
def test_global_avg_pool_1d():
    """Test global average pooling reduces to correct shape."""
    from desc.backend import jnp
    from desc.compute._turbulence import _global_avg_pool_1d

    batch, channels, length = 2, 32, 3
    x = jnp.ones((batch, channels, length))

    pooled = _global_avg_pool_1d(x)

    assert pooled.shape == (batch, channels)
    # All ones should average to 1
    np.testing.assert_allclose(np.array(pooled), 1.0)


# =============================================================================
# CNN Forward Pass Tests
# =============================================================================


@pytest.mark.unit
def test_cyclic_invariant_forward_shape():
    """Test CNN forward pass produces correct output shape."""
    from desc.backend import jnp
    from desc.compute._turbulence import _cyclic_invariant_forward

    # Create random weights matching CyclicInvariantNet architecture
    np.random.seed(42)

    # Architecture hyperparameters (typical values)
    conv_channels = [7, 16, 32, 16, 32, 16]  # input + 5 layers
    kernel_sizes = [3, 3, 3, 3, 3]
    fc_dims = [16 + 2, 32, 16]  # +2 for scalars

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
    weights["output_layer.weight"] = jnp.array(np.random.randn(1, fc_dims[2])).astype(jnp.float32)
    weights["output_layer.bias"] = jnp.array(np.random.randn(1)).astype(jnp.float32)

    # Test forward pass
    batch = 3
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


# =============================================================================
# NNITGProxy Objective Tests
# =============================================================================


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


@pytest.mark.unit
def test_nnitgproxy_init():
    """Test NNITGProxy construction and parameter validation."""
    from desc.objectives._turbulence import NNITGProxy

    eq = get("DSHAPE")

    # Test basic construction with mock model_path
    obj = NNITGProxy(eq, rho=0.5, model_path="/fake/path.pt")
    assert obj._npoints == 96
    assert obj._nz_internal == 1001
    assert obj._target_flux_tube_length == 75.4
    assert obj._a_over_LT == 3.0
    assert obj._a_over_Ln == 1.0
    np.testing.assert_array_equal(obj._rho, [0.5])

    # Test with custom parameters
    obj2 = NNITGProxy(
        eq,
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
        NNITGProxy(eq, rho=0.5, ensemble_dir="/fake/dir")


@pytest.mark.unit
def test_nnitgproxy_build():
    """Test NNITGProxy build method with mock weights."""
    from unittest.mock import patch

    from desc.objectives._turbulence import NNITGProxy

    eq = get("DSHAPE")
    mock_weights = _make_mock_weights()

    # Patch _load_nn_weights to return mock weights
    with patch(
        "desc.objectives._turbulence._load_nn_weights", return_value=mock_weights
    ):
        obj = NNITGProxy(eq, rho=0.5, model_path="/fake/path.pt")
        obj.build(use_jit=False, verbose=0)

    # Check constants are populated
    assert "rho" in obj.constants
    assert "alpha" in obj.constants
    assert "poloidal_turns" in obj.constants
    assert "target_length" in obj.constants
    assert "a" in obj.constants
    assert "nn_weights" in obj.constants
    assert "a_over_LT" in obj.constants
    assert "a_over_Ln" in obj.constants

    # Check values are reasonable
    assert obj.constants["poloidal_turns"] > 0
    assert obj.constants["target_length"] > 0
    assert obj.constants["a"] > 0
    assert obj.constants["nn_weights"] is not None
    assert obj.constants["ensemble_weights"] is None

    # Check dimension
    assert obj._dim_f == 1  # Single rho value


@pytest.mark.unit
def test_nnitgproxy_compute():
    """Test NNITGProxy compute method produces valid output."""
    from unittest.mock import patch

    from desc.objectives._turbulence import NNITGProxy

    eq = get("DSHAPE")
    mock_weights = _make_mock_weights()

    with patch(
        "desc.objectives._turbulence._load_nn_weights", return_value=mock_weights
    ):
        obj = NNITGProxy(eq, rho=0.5, model_path="/fake/path.pt")
        obj.build(use_jit=False, verbose=0)

    # Compute Q
    Q = obj.compute(obj.things[0].params_dict)

    # Check output shape and validity
    assert Q.shape == (1,)  # Single rho value
    assert np.isfinite(Q).all()
    assert Q[0] > 0  # Q should be positive (exp of log_Q)


@pytest.mark.unit
def test_nnitgproxy_compute_return_signals():
    """Test NNITGProxy compute with return_signals=True."""
    from unittest.mock import patch

    from desc.objectives._turbulence import NNITGProxy

    eq = get("DSHAPE")
    mock_weights = _make_mock_weights()

    with patch(
        "desc.objectives._turbulence._load_nn_weights", return_value=mock_weights
    ):
        obj = NNITGProxy(
            eq, rho=[0.4, 0.6], alpha=[0, np.pi / 2], model_path="/fake/path.pt"
        )
        obj.build(use_jit=False, verbose=0)

    # Compute Q with signals
    Q, signals_info = obj.compute(obj.things[0].params_dict, return_signals=True)

    # Check Q output
    assert Q.shape == (2,)  # Two rho values
    assert np.isfinite(Q).all()

    # Check signals_info structure
    assert "z" in signals_info
    assert "signals" in signals_info
    assert "feature_names" in signals_info

    # Check z coordinates
    z = signals_info["z"]
    assert z.shape == (96,)  # Default npoints
    np.testing.assert_allclose(z[0], -np.pi, rtol=1e-5)

    # Check signals shape: (num_rho, num_alpha, 7, npoints)
    signals = signals_info["signals"]
    assert signals.shape == (2, 2, 7, 96)
    assert np.isfinite(signals).all()

    # Check feature names
    assert len(signals_info["feature_names"]) == 7
    assert signals_info["feature_names"][0] == "bmag"
