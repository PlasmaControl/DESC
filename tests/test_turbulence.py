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
