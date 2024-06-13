"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.test_plotting import tol_1d

from desc.equilibrium.coords import rtz_grid
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import ObjectiveFunction
from desc.objectives._neoclassical import EffectiveRipple, GammaC


@pytest.mark.unit
def test_field_line_average():
    """Test that field line average converges to surface average."""
    eq = get("W7-X")
    grid = rtz_grid(
        eq,
        np.array([0, 0.5]),
        np.array([0]),
        np.linspace(0, 40 * np.pi, 200),
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
    )
    data = eq.compute(["<L|r,a>", "<G|r,a>", "V_r(r)"], grid=grid)
    np.testing.assert_allclose(
        data["<L|r,a>"] / data["<G|r,a>"], data["V_r(r)"] / (4 * np.pi**2), rtol=1e-3
    )


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_effective_ripple():
    """Test effective ripple with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = rtz_grid(
        eq,
        rho,
        np.array([0]),
        np.linspace(0, 20 * np.pi, 1000),
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
    )
    data = eq.compute("effective ripple", grid=grid)
    assert np.isfinite(data["effective ripple"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["effective ripple"]), marker="o")
    return fig


@pytest.mark.unit
def test_ad_ripple():
    """Make sure we can differentiate."""
    eq = get("ESTELL")
    grid = LinearGrid(L=1, M=2, N=2, NFP=eq.NFP, sym=eq.sym, axis=False)
    eq.change_resolution(2, 2, 2, 4, 4, 4)

    obj = ObjectiveFunction(EffectiveRipple(eq, grid=grid))
    obj.build(verbose=0)
    g = obj.grad(obj.x())
    assert not np.any(np.isnan(g))  # FIXME


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c():
    """Test Γ_c with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = rtz_grid(
        eq,
        rho,
        np.array([0]),
        np.linspace(0, 20 * np.pi, 1000),
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
    )
    data = eq.compute("Gamma_c", grid=grid)
    assert np.isfinite(data["Gamma_c"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["Gamma_c"]), marker="o")
    return fig


@pytest.mark.unit
def test_ad_gamma():
    """Make sure we can differentiate."""
    eq = get("ESTELL")
    grid = LinearGrid(L=1, M=2, N=2, NFP=eq.NFP, sym=eq.sym, axis=False)
    eq.change_resolution(2, 2, 2, 4, 4, 4)

    obj = ObjectiveFunction(GammaC(eq, grid=grid))
    obj.build(verbose=0)
    g = obj.grad(obj.x())
    assert not np.any(np.isnan(g))  # FIXME
