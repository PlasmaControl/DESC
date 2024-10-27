"""Tests for deprecated compute functions for neoclassical transport."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.test_plotting import tol_1d

from desc.equilibrium.coords import get_rtz_grid
from desc.examples import get
from desc.grid import LinearGrid

from .test_neoclassical import NeoIO


@pytest.mark.unit
def test_fieldline_average():
    """Test that fieldline average converges to surface average."""
    rho = np.array([1])
    alpha = np.array([0])
    eq = get("DSHAPE")
    iota_grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = iota_grid.compress(eq.compute("iota", grid=iota_grid)["iota"]).item()
    # For axisymmetric devices, one poloidal transit must be exact.
    zeta = np.linspace(0, 2 * np.pi / iota, 25)
    grid = get_rtz_grid(eq, rho, alpha, zeta, coordinates="raz")
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
    grid = get_rtz_grid(eq, rho, alpha, zeta, coordinates="raz")
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


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_effective_ripple_1D():
    """Test effective ripple 1D with W7-X against NEO."""
    Y_B = 100
    num_transit = 10
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = get_rtz_grid(
        eq,
        rho,
        poloidal=np.array([0]),
        toroidal=np.linspace(0, num_transit * 2 * np.pi, num_transit * Y_B),
        coordinates="raz",
    )
    data = eq.compute(
        "deprecated(effective ripple)", grid=grid, num_well=20 * num_transit
    )

    assert np.isfinite(data["deprecated(effective ripple)"]).all()
    np.testing.assert_allclose(
        data["deprecated(effective ripple 3/2)"] ** (2 / 3),
        data["deprecated(effective ripple)"],
        err_msg="Bug in source grid logic in eq.compute.",
    )
    eps_32 = grid.compress(data["deprecated(effective ripple 3/2)"])
    neo_rho, neo_eps_32 = NeoIO.read("tests/inputs/neo_out.w7x")
    np.testing.assert_allclose(eps_32, np.interp(rho, neo_rho, neo_eps_32), rtol=0.16)

    fig, ax = plt.subplots()
    ax.plot(rho, eps_32, marker="o")
    ax.plot(neo_rho, neo_eps_32)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c_1D():
    """Test Γ_c Nemov 1D with W7-X."""
    Y_B = 100
    num_transit = 10
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = get_rtz_grid(
        eq,
        rho,
        poloidal=np.array([0]),
        toroidal=np.linspace(0, num_transit * 2 * np.pi, num_transit * Y_B),
        coordinates="raz",
    )
    data = eq.compute("deprecated(Gamma_c)", grid=grid, num_well=20 * num_transit)
    assert np.isfinite(data["deprecated(Gamma_c)"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["deprecated(Gamma_c)"]), marker="o")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c_Velasco_1D():
    """Test Γ_c Velasco 1D with W7-X."""
    Y_B = 100
    num_transit = 10
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = get_rtz_grid(
        eq,
        rho,
        poloidal=np.array([0]),
        toroidal=np.linspace(0, num_transit * 2 * np.pi, num_transit * Y_B),
        coordinates="raz",
    )
    data = eq.compute(
        "deprecated(Gamma_c Velasco)", grid=grid, num_well=20 * num_transit
    )
    assert np.isfinite(data["deprecated(Gamma_c Velasco)"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["deprecated(Gamma_c Velasco)"]), marker="o")
    return fig
