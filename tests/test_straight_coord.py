"""Test cylindrical coordinate implementation"""

# import pickle

import numpy as np
import pytest

# from scipy.io import netcdf_file
# from scipy.signal import convolve2d

# from desc.compute import data_index, rpz2xyz_vec
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.profiles import PowerSeriesProfile

# from desc.examples import get
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierRZToroidalSurface,
    FourierXYZCurve,
    ZernikeRZToroidalSection,
)
from desc.grid import LinearGrid, QuadratureGrid
from .ana_util import modes_gen
from .ana_straight import model_const_B, model_screw_pinch1, model_theta_pinch1
from .ana_straight_3d import model_mirror1, model_mirror_iota1, model_mirror_3d1


@pytest.fixture(scope="session")
def grid_3d():
    return LinearGrid(
        rho=np.linspace(0.1, 1, 5), M=5, N=5
    )  # Carefull on axis, might generate nan when compute


@pytest.fixture(scope="session")
def const_B(grid_3d):
    eq = Equilibrium()
    ana_model = model_const_B(eq.Psi, 1, 10)
    return eq, ana_model, grid_3d


@pytest.fixture(scope="session")
def theta_pinch1(grid_3d):
    eq = Equilibrium(L=3, M=3)
    rho = grid_3d.nodes[:, 0]
    theta = grid_3d.nodes[:, 1]
    # a0, a1 = request.param
    a0, a1 = 0.8, 0.2
    R = 10 + (a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.cos(theta)
    Z = -(a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.sin(theta)
    eq.set_initial_guess(grid_3d, R, Z)
    ana_model = model_theta_pinch1(eq.Psi, a0, a1, 10)
    return eq, ana_model, grid_3d


@pytest.fixture(scope="session")
def screw_pinch1(grid_3d):
    i0, i2 = 1, 1
    iota = PowerSeriesProfile(params=[i0, i2], modes=[0, 2])
    eq = Equilibrium(L=3, M=3, iota=iota)
    rho = grid_3d.nodes[:, 0]
    theta = grid_3d.nodes[:, 1]
    # a0, a1 = request.param
    a0, a1 = 0.8, 0.2
    R = 10 + (a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.cos(theta)
    Z = -(a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.sin(theta)
    eq.set_initial_guess(grid_3d, R, Z)
    ana_model = model_screw_pinch1(eq.Psi, a0, a1, 10, i0, i2)
    return eq, ana_model, grid_3d


@pytest.fixture(scope="session")
def mirror1(grid_3d):
    a0 = 0.8
    a1 = 0.2
    b1 = -0.1
    iota = PowerSeriesProfile(params=[0, 0], modes=[0, 2])
    eq = Equilibrium(L=3, M=3, N=1, iota=iota, sym=None)
    ana_model = model_mirror1(eq.Psi, a0, a1, b1, 10)
    rlmn = modes_gen(ana_model.modes["R_lmn"], eq.R_basis)
    zlmn = modes_gen(ana_model.modes["Z_lmn"], eq.Z_basis)
    eq.R_lmn = rlmn
    eq.Z_lmn = zlmn
    return eq, ana_model, grid_3d


@pytest.fixture(scope="session")
def mirror_iota1(grid_3d):
    a0 = 0.8
    a1 = 0.2
    b1 = -0.1
    i0 = 1
    i2 = 1
    Psi = 1.0
    ana_model = model_mirror_iota1(Psi, a0, a1, b1, 10, i0, i2)
    iota = PowerSeriesProfile(
        params=ana_model.modes["iota"]["params"], modes=ana_model.modes["iota"]["modes"]
    )
    eq = Equilibrium(Psi, L=3, M=3, N=1, iota=iota, sym=None)
    rlmn = modes_gen(ana_model.modes["R_lmn"], eq.R_basis)
    zlmn = modes_gen(ana_model.modes["Z_lmn"], eq.Z_basis)
    eq.R_lmn = rlmn
    eq.Z_lmn = zlmn
    return eq, ana_model, grid_3d

@pytest.fixture(scope="session")
def mirror_3d1(grid_3d):
    a0 = 0.8
    a1 = 0.2
    b1 = -0.1
    c1 = 0.1
    c2 = 0.2
    Psi = 1.0
    ana_model = model_mirror_3d1(Psi, a0, a1, b1, 10, c1, c2)
    iota = PowerSeriesProfile(
        params=ana_model.modes["iota"]["params"], modes=ana_model.modes["iota"]["modes"]
    )
    eq = Equilibrium(Psi, L=3, M=3, N=1, iota=iota, sym=None)
    rlmn = modes_gen(ana_model.modes["R_lmn"], eq.R_basis)
    zlmn = modes_gen(ana_model.modes["Z_lmn"], eq.Z_basis)
    eq.R_lmn = rlmn
    eq.Z_lmn = zlmn
    return eq, ana_model, grid_3d

@pytest.fixture(scope="session")
def theta_pinch2(grid_3d):
    eq = None
    B_ana = None
    grid = grid_3d
    return eq, B_ana, grid


@pytest.fixture(scope="session")
def screw_pinch2():
    pass


@pytest.mark.mirror_unit
def test_construct_equilibrium():
    """make sure equilibrium can be constructed and can compute something"""
    eq = Equilibrium()
    data = eq.compute(["|B|"])


@pytest.mark.mirror_unit
@pytest.mark.parametrize(
    "config",
    ["const_B", "theta_pinch1"],
)
def test_comput_ortho_force_balance_straight_coord(config, request):
    eq, ana, grid = request.getfixturevalue(config)
    names = ["|J|", "|F|", "|B|"]
    data = eq.compute(names, grid=grid)
    np.testing.assert_allclose(
        data["|J|"] * data["|B|"], data["|F|"], atol=1e-15, rtol=1e-15
    )


@pytest.mark.mirror_unit
@pytest.mark.parametrize(
    "name, func_name",
    [("B", "B_vec_ana_cal"), ("J", "j_vec_ana_cal"), ("F", "gradp_vec_ana_cal")],
)
@pytest.mark.parametrize(
    "config",
    ["const_B", "theta_pinch1", "screw_pinch1"],
)
def test_compute_against_ana_straight_coord(config, request, name, func_name):
    eq, ana, grid = request.getfixturevalue(config)
    names = [name]
    data = eq.compute(names, grid=grid)
    rtz = grid.nodes
    data_ana = getattr(ana, func_name)(rtz)
    np.testing.assert_allclose(data[name], data_ana, atol=1e-6, rtol=1e-10)


@pytest.mark.mirror_unit
@pytest.mark.parametrize(
    "name, func_name",
    [("B", "B_vec_ana_cal"), ("J", "j_vec_ana_cal"), ("F", "gradp_vec_ana_cal")],
)
@pytest.mark.parametrize(
    "config",
    ["mirror1", "mirror_iota1"],
)
def test_compute_against_ana_straight_coord_3D(config, request, name, func_name):
    eq, ana, grid = request.getfixturevalue(config)
    names = [name]
    data = eq.compute(names, grid=grid)
    rtz = grid.nodes
    data_ana = getattr(ana, func_name)(rtz)
    np.testing.assert_allclose(data[name], data_ana, atol=1e-6, rtol=1e-10)

@pytest.mark.mirror_unit
@pytest.mark.parametrize(
    "name, func_name",
    [("B", "B_vec_ana_cal")],
)
@pytest.mark.parametrize(
    "config",
    ["mirror_3d1"],
)
def test_compute_B_against_ana_straight_coord_3D(config, request, name, func_name):
    eq, ana, grid = request.getfixturevalue(config)
    names = [name]
    data = eq.compute(names, grid=grid)
    rtz = grid.nodes
    data_ana = getattr(ana, func_name)(rtz)
    np.testing.assert_allclose(data[name], data_ana, atol=1e-6, rtol=1e-10)


@pytest.mark.mirror_regression
def test_solved_theta_pinch(theta_pinch):
    pass
