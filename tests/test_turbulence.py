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
