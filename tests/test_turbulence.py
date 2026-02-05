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


@pytest.mark.unit
def test_gx_coefficients_tokamak():
    """Test all GX coefficients on tokamak equilibrium."""
    eq = get("DSHAPE")
    rho = np.linspace(0.2, 0.8, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

    quantities = [
        "gx_bmag",
        "gx_gds2",
        "gx_gds21_over_shat",
        "gx_gds22_over_shat_squared",
        "gx_gbdrift",
        "gx_cvdrift",
        "gx_gbdrift0_over_shat",
        "gx_gradpar",
    ]
    data = eq.compute(quantities, grid=grid)

    for name in quantities:
        assert np.isfinite(data[name]).all(), f"Non-finite values for {name}"

    # Positivity checks for squared quantities
    assert np.all(data["gx_bmag"] > 0)
    assert np.all(data["gx_gds2"] > 0)
    assert np.all(data["gx_gds22_over_shat_squared"] > 0)


@pytest.mark.unit
def test_gx_coefficients_stellarator():
    """Test all GX coefficients on stellarator equilibrium."""
    eq = get("HELIOTRON")
    rho = np.linspace(0.2, 0.8, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

    quantities = [
        "gx_bmag",
        "gx_gds2",
        "gx_gds21_over_shat",
        "gx_gds22_over_shat_squared",
        "gx_gbdrift",
        "gx_cvdrift",
        "gx_gbdrift0_over_shat",
        "gx_gradpar",
    ]
    data = eq.compute(quantities, grid=grid)

    for name in quantities:
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
def test_compute_flux_tube_length():
    """Test flux tube length computation."""
    from desc.compute._turbulence import (
        compute_arclength_via_gradpar,
        compute_flux_tube_length,
    )

    npoints = 101
    theta_pest = np.linspace(-np.pi, np.pi, npoints)

    # Constant gradpar = 2: dl/dθ = 1/2, so total length = pi
    gradpar = np.ones(npoints) * 2.0
    length = compute_flux_tube_length(gradpar, theta_pest)

    # Should equal the last value of cumulative arclength
    arclength = compute_arclength_via_gradpar(gradpar, theta_pest)
    np.testing.assert_allclose(length, arclength[-1], rtol=1e-10)

    # Expected value for constant gradpar
    expected_length = np.pi
    np.testing.assert_allclose(length, expected_length, rtol=1e-3)


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
