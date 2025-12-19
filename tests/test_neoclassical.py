"""Test neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.test_plotting import tol_1d

from desc.examples import get
from desc.external import NeoIO
from desc.grid import Grid, LinearGrid
from desc.integrals import Bounce2D


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
@pytest.mark.parametrize("nufft_eps", [0, 1e-6])
def test_effective_ripple_2D(nufft_eps):
    """Test effective ripple with W7-X against NEO."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    num_transit = 10
    data = eq.compute(
        "effective ripple 3/2",
        grid=grid,
        angle=Bounce2D.angle(eq, X=32, Y=32, rho=rho),
        Y_B=128,
        num_transit=num_transit,
        num_well=20 * num_transit,
        surf_batch_size=1 if (nufft_eps == 0) else 2,
        nufft_eps=nufft_eps,
    )

    assert data["effective ripple 3/2"].ndim == 1
    assert np.isfinite(data["effective ripple 3/2"]).all()
    eps_32 = grid.compress(data["effective ripple 3/2"])
    # NEO file generated from DESC equlibrium on 2025-10-23 17:47:07.280264.
    neo_rho, neo_eps_32 = NeoIO.read("tests/inputs/neo_out.W7-X")
    np.testing.assert_allclose(eps_32, np.interp(rho, neo_rho, neo_eps_32), rtol=0.11)

    fig, ax = plt.subplots()
    ax.plot(rho, eps_32, marker="o")
    ax.plot(neo_rho, neo_eps_32)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_effective_ripple_1D():
    """Test effective ripple 1D with W7-X against NEO."""
    eq = get("W7-X")
    Y_B = 100
    num_transit = 10
    num_well = 20 * num_transit
    rho = np.linspace(0, 1, 10)
    alpha = np.array([0])
    zeta = np.linspace(0, num_transit * 2 * np.pi, num_transit * Y_B)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    data = eq.compute(
        "old effective ripple", grid=grid, num_well=num_well, surf_batch_size=2
    )

    assert np.isfinite(data["old effective ripple"]).all()
    np.testing.assert_allclose(
        data["old effective ripple 3/2"] ** (2 / 3),
        data["old effective ripple"],
        err_msg="Bug in source grid logic in eq.compute.",
    )
    eps_32 = grid.compress(data["old effective ripple 3/2"])
    neo_rho, neo_eps_32 = NeoIO.read("tests/inputs/neo_out.W7-X")
    np.testing.assert_allclose(eps_32, np.interp(rho, neo_rho, neo_eps_32), rtol=0.1)

    fig, ax = plt.subplots()
    ax.plot(rho, eps_32, marker="o")
    ax.plot(neo_rho, neo_eps_32)
    return fig


@pytest.mark.unit
@pytest.mark.slow
def test_fieldline_average():
    """Test that fieldline average converges to surface average."""
    rho = np.array([1])
    alpha = np.array([0])
    eq = get("DSHAPE")
    iota_grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = iota_grid.compress(eq.compute("iota", grid=iota_grid)["iota"]).item()
    # For axisymmetric devices, one poloidal transit must be exact.
    zeta = np.linspace(0, 2 * np.pi / iota, 25)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    data = eq.compute(
        ["fieldline length", "fieldline length/volume", "V_r(r)"], grid=grid
    )
    np.testing.assert_allclose(
        data["fieldline length"] / data["fieldline length/volume"],
        data["V_r(r)"] / (4 * np.pi**2),
        rtol=1e-3,
    )
    assert np.all(data["fieldline length"] > 0)
    assert np.all(data["fieldline length/volume"] > 0)

    # Otherwise, many toroidal transits are necessary to sample surface.
    eq = get("W7-X")
    zeta = np.linspace(0, 40 * np.pi, 300)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    data = eq.compute(
        ["fieldline length", "fieldline length/volume", "V_r(r)"], grid=grid
    )
    np.testing.assert_allclose(
        data["fieldline length"] / data["fieldline length/volume"],
        data["V_r(r)"] / (4 * np.pi**2),
        rtol=2e-3,
    )
    assert np.all(data["fieldline length"] > 0)
    assert np.all(data["fieldline length/volume"] > 0)
