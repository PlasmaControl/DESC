"""Tests for ITG turbulence compute functions."""

import numpy as np
import pytest

from desc.examples import get
from desc.grid import LinearGrid

@pytest.mark.unit
def test_gx_B_reference():
    """Test GX reference magnetic field B_ref = 2|ψ_b|/a²."""
    eq = get("DSHAPE")
    data = eq.compute(["gx_B_reference", "a"])

    psi_b = np.abs(eq.Psi) / (2 * np.pi)
    expected = 2 * psi_b / data["a"] ** 2

    assert np.isfinite(data["gx_B_reference"])
    np.testing.assert_allclose(data["gx_B_reference"], expected, rtol=1e-10)

@pytest.mark.unit
def test_gx_L_reference():
    """Test GX reference length L_ref = a."""
    eq = get("DSHAPE")
    data = eq.compute(["gx_L_reference", "a"])

    assert np.isfinite(data["gx_L_reference"])
    np.testing.assert_allclose(data["gx_L_reference"], data["a"], rtol=1e-10)

@pytest.mark.unit
def test_gx_bmag():
    """Test normalized magnetic field bmag = |B|/B_ref."""
    eq = get("DSHAPE")
    rho = np.linspace(0.1, 1, 5)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute(["gx_bmag", "gx_B_reference", "|B|"], grid=grid)

    expected = data["|B|"] / data["gx_B_reference"]

    assert np.isfinite(data["gx_bmag"]).all()
    np.testing.assert_allclose(data["gx_bmag"], expected, rtol=1e-10)
    # bmag should be O(1) for well-normalized equilibrium
    assert np.all(data["gx_bmag"] > 0.1)
    assert np.all(data["gx_bmag"] < 10)

@pytest.mark.unit
def test_gx_bmag_stellarator():
    """Test bmag on 3D stellarator equilibrium."""
    eq = get("HELIOTRON")
    rho = np.linspace(0.1, 1, 5)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute(["gx_bmag"], grid=grid)

    assert np.isfinite(data["gx_bmag"]).all()
    assert np.all(data["gx_bmag"] > 0)

@pytest.mark.unit
def test_gx_gds2():
    """Test gds2 = |∇α|² × L_ref² × s."""
    eq = get("DSHAPE")
    rho = np.linspace(0.2, 0.8, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute(["gx_gds2", "grad(alpha)", "a", "rho"], grid=grid)

    # Manual calculation
    from desc.utils import dot
    grad_alpha_sq = dot(data["grad(alpha)"], data["grad(alpha)"])
    L_ref = data["a"]
    s = data["rho"] ** 2
    expected = grad_alpha_sq * L_ref**2 * s

    assert np.isfinite(data["gx_gds2"]).all()
    np.testing.assert_allclose(data["gx_gds2"], expected, rtol=1e-10)
    assert np.all(data["gx_gds2"] > 0)  # Should be positive (squared quantity)

@pytest.mark.unit
def test_gx_gds21_over_shat():
    """Test gds21/shat = σ_Bxy × (∇α·∇ψ) / B_ref."""
    eq = get("DSHAPE")
    rho = np.linspace(0.2, 0.8, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute(
        ["gx_gds21_over_shat", "grad(alpha)", "grad(psi)", "a"], grid=grid
    )

    # Manual calculation
    from desc.utils import dot
    sigma_Bxy = -1
    psi_b = eq.Psi / (2 * np.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    grad_alpha_dot_grad_psi = dot(data["grad(alpha)"], data["grad(psi)"])
    expected = sigma_Bxy * grad_alpha_dot_grad_psi / B_ref

    assert np.isfinite(data["gx_gds21_over_shat"]).all()
    np.testing.assert_allclose(data["gx_gds21_over_shat"], expected, rtol=1e-10)

@pytest.mark.unit
def test_gx_gds22_over_shat_squared():
    """Test gds22/shat² = |∇ψ|² / (L_ref² × B_ref² × s)."""
    eq = get("DSHAPE")
    rho = np.linspace(0.2, 0.8, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute(["gx_gds22_over_shat_squared", "|grad(psi)|^2", "a", "rho"], grid=grid)

    # Manual calculation
    psi_b = eq.Psi / (2 * np.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    L_ref = data["a"]
    s = data["rho"] ** 2
    expected = data["|grad(psi)|^2"] / (L_ref**2 * B_ref**2 * s)

    assert np.isfinite(data["gx_gds22_over_shat_squared"]).all()
    np.testing.assert_allclose(data["gx_gds22_over_shat_squared"], expected, rtol=1e-10)
    assert np.all(data["gx_gds22_over_shat_squared"] > 0)  # Should be positive

@pytest.mark.unit
def test_gx_gradient_coefficients_stellarator():
    """Test gradient coefficients on 3D stellarator equilibrium."""
    eq = get("HELIOTRON")
    rho = np.linspace(0.2, 0.8, 3)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute(
        ["gx_gds2", "gx_gds21_over_shat", "gx_gds22_over_shat_squared"], grid=grid
    )

    assert np.isfinite(data["gx_gds2"]).all()
    assert np.isfinite(data["gx_gds21_over_shat"]).all()
    assert np.isfinite(data["gx_gds22_over_shat_squared"]).all()
    assert np.all(data["gx_gds2"] > 0)
    assert np.all(data["gx_gds22_over_shat_squared"] > 0)
