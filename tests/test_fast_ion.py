"""Test fast ion compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.test_plotting import tol_1d

from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.integrals import Bounce2D


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
@pytest.mark.parametrize("nufft_eps", [0, 1e-7])
def test_Gamma_c_Nemov_2D(nufft_eps):
    """Test Γ_c Nemov with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(1e-12, 1, 10)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    num_transit = 10
    data = eq.compute(
        "Gamma_c",
        grid=grid,
        theta=Bounce2D.compute_theta(eq, X=32, Y=64, rho=rho),
        Y_B=128,
        num_transit=num_transit,
        num_well=20 * num_transit,
        nufft_eps=nufft_eps,
    )
    assert np.isfinite(data["Gamma_c"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["Gamma_c"]), marker="o")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
@pytest.mark.parametrize("nufft_eps", [0, 1e-7])
def test_Gamma_c_Velasco_2D(nufft_eps):
    """Test Γ_c Velasco with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(1e-12, 1, 10)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    num_transit = 10
    data = eq.compute(
        "Gamma_c Velasco",
        grid=grid,
        theta=Bounce2D.compute_theta(eq, X=32, Y=64, rho=rho),
        Y_B=128,
        num_transit=num_transit,
        num_well=20 * num_transit,
        nufft_eps=nufft_eps,
    )
    assert np.isfinite(data["Gamma_c Velasco"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["Gamma_c Velasco"]), marker="o")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c_Nemov_1D():
    """Test Γ_c Nemov 1D with W7-X."""
    eq = get("W7-X")
    Y_B = 100
    num_transit = 10
    num_well = 20 * num_transit
    rho = np.linspace(1e-12, 1, 10)
    alpha = np.array([0])
    zeta = np.linspace(0, num_transit * 2 * np.pi, num_transit * Y_B)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    data = eq.compute("old Gamma_c", grid=grid, num_well=num_well, surf_batch_size=2)

    assert np.isfinite(data["old Gamma_c"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["old Gamma_c"]), marker="o")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c_Velasco_1D():
    """Test Γ_c Velasco 1D with W7-X."""
    eq = get("W7-X")
    Y_B = 100
    num_transit = 10
    num_well = 20 * num_transit
    rho = np.linspace(1e-12, 1, 10)
    alpha = np.array([0])
    zeta = np.linspace(0, num_transit * 2 * np.pi, num_transit * Y_B)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    data = eq.compute("old Gamma_c Velasco", grid=grid, num_well=num_well)

    assert np.isfinite(data["old Gamma_c Velasco"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["old Gamma_c Velasco"]), marker="o")
    return fig
