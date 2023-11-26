"""Test cylindrical coordinate implementation"""

# import pickle

import numpy as np
import pytest

# from scipy.io import netcdf_file
# from scipy.signal import convolve2d

from desc.compute import data_index, rpz2xyz_vec
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.examples import get
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierRZToroidalSurface,
    FourierXYZCurve,
    ZernikeRZToroidalSection,
)
from desc.grid import LinearGrid, QuadratureGrid


def theta_pinch_B_ana_cal(rho, Psi0, a0, a1):
    """B(rho) for a theta_pinch configuration which r(rho) has only
    R_1^1 and R_3^1 Zernike components.

    Args:
        rho (np.array): _description_
        Psi0 (float): _description_
        a0 (float): R_1^2 coefficient
        a1 (float): R_3^2 coefficient

    Returns:
        _type_: _description_
    """
    return (
        Psi0
        / np.pi
        / (a0 + a1 * (3 * rho**2 - 2))
        / (a0 - 2 * a1 + 9 * a1 * rho**2)
    )


@pytest.fixture(scope="session")
def grid_3d():
    return LinearGrid(L=3, M=3, N=3)


@pytest.fixture(scope="session")
def const_B(grid_3d):
    eq = Equilibrium()
    B = np.zeros_like(grid_3d.nodes)
    B[:, 1] = eq.Psi / np.pi
    return eq, B, grid_3d


@pytest.fixture(scope="session")
def non_const_B(request, grid_3d):
    eq = Equilibrium(L=3, M=3)
    rho = grid_3d.nodes[:, 0]
    theta = grid_3d.nodes[:, 1]
    # a0, a1 = request.param
    a0, a1 = 0.8, 0.2
    R = 10 + (a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.cos(theta)
    Z = -(a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.sin(theta)
    eq.set_initial_guess(grid_3d, R, Z)
    B = np.zeros_like(grid_3d.nodes)
    B[:, 1] = theta_pinch_B_ana_cal(rho, eq.Psi, a0, a1)
    return eq, B, grid_3d


@pytest.fixture(scope="session")
def theta_pinch(grid_3d):
    eq = None
    B_ana = None
    grid = grid_3d
    return eq, B_ana, grid


@pytest.fixture(scope="session")
def screw_pinch():
    pass


@pytest.mark.mirror_unit
def test_construct_equilibrium():
    """make sure equilibrium can be constructed and can compute something"""
    eq = Equilibrium()
    data = eq.compute(["|B|"])


@pytest.mark.mirror_unit
@pytest.mark.parametrize(
    "config",
    ["const_B", "non_const_B"],
)
def test_compute_B_straight_coord(config, request):
    eq, B_ana, grid = request.getfixturevalue(config)
    B = eq.compute("B", grid=grid)["B"]
    np.testing.assert_allclose(B, B_ana)


@pytest.mark.mirror_regression
def test_theta_pinch():
    pass


@pytest.mark.mirror_regression
def test_screw_pinch():
    pass
