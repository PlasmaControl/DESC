"""Test cylindrical coordinate implementation (Still periodic in zeta)"""

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile

from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.continuation import solve_continuation_automatic
from .ana_straight_eq import model_theta_pinch_eq1


@pytest.fixture(scope="session")
def grid_3d():
    return LinearGrid(
        rho=np.linspace(0.1, 1, 5), M=5, N=5
    )  # Carefull on axis, might generate nan when compute


@pytest.fixture(scope="session")
def theta_pinch_eq1(grid_3d):
    R = 7
    a = 1.2
    Psi0 = 0.9
    grid = grid_3d
    model = model_theta_pinch_eq1(R, a, Psi0)
    params = model.params
    p = PowerSeriesProfile(
        params=params["pressure"]["params"],
        modes=params["pressure"]["modes"],
        # sym=params["sym"],
    )
    iota = PowerSeriesProfile(params=[0, 0])
    surf = FourierRZToroidalSurface(
        R_lmn=params["Rb_lmn"]["R_lmn"],
        modes_R=params["Rb_lmn"]["modes_R"],
        Z_lmn=params["Zb_lmn"]["Z_lmn"],
        modes_Z=params["Zb_lmn"]["modes_Z"],
        sym=params["sym"],
    )
    eq = Equilibrium(
        surface=surf,
        pressure=p,
        iota=iota,
        Psi=Psi0,
        NFP=1,
        L=5,
        M=3,
        N=1,
        L_grid=10,
        M_grid=6,
        N_grid=2,
        sym=params["sym"],
        method="jitable",
    )

    return eq, model, grid


@pytest.fixture(scope="session")
def screw_pinch2():
    pass


@pytest.mark.solve
@pytest.mark.mirror_regression
@pytest.mark.parametrize(
    "config",
    ["theta_pinch_eq1"],
)
def test_solved_theta_pinch_direct(config, request):
    eq, model, grid = request.getfixturevalue(config)
    assert isinstance(model, model_theta_pinch_eq1)
    eq.solve(
        verbose=3,
        ftol=1e-6,
        xtol=1e-16,
        gtol=1e-16,
        maxiter=100,
    )
    data = eq.compute(["R", "Z"], grid=grid)
    rho = np.linalg.norm(np.stack((data["R"] - model.R, data["Z"]), axis=1), axis=1)
    rho_ana = model.radial(grid.nodes)
    np.testing.assert_allclose(rho, rho_ana, atol=1e-2, rtol=1e-3)


@pytest.mark.slow
@pytest.mark.solve
@pytest.mark.mirror_regression
@pytest.mark.parametrize(
    "config",
    ["theta_pinch_eq1"],
)
def test_solved_theta_pinch_auto(config, request):
    eq, model, grid = request.getfixturevalue(config)
    assert isinstance(model, model_theta_pinch_eq1)
    eqfam = solve_continuation_automatic(
        eq,
        ftol=1e-6,
        xtol=1e-16,
        gtol=1e-16,
    )
    data = eq.compute(["R", "Z"], grid=grid)
    rho = np.linalg.norm(np.stack((data["R"] - model.R, data["Z"]), axis=1), axis=1)
    rho_ana = model.radial(grid.nodes)
    np.testing.assert_allclose(rho, rho_ana, atol=1e-2, rtol=1e-3)
