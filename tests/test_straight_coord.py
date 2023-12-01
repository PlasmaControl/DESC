"""Test cylindrical coordinate implementation"""

# import pickle

import numpy as np
import pytest
from scipy.constants import mu_0

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
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ana_model(ABC):
    @abstractmethod
    def j_vec_ana_cal(self):
        pass
    @abstractmethod
    def B_vec_ana_cal(self):
        pass
    @abstractmethod
    def gradp_vec_ana_cal(self):
        pass

@dataclass
class model_const_B(ana_model):
    """
        Const B configuration
    """
    Psi: float
    a: float
    R: float

    def xyz(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        return np.stack(
            (self.R + rho * np.cos(theta), zeta, -rho * np.sin(theta)), axis=1
        )

    def B_vec_ana_cal(self, rtz):
        B = np.zeros_like(rtz)
        B[:, 1] = self.Psi / np.pi / self.a**2
        return B

    def p_ana_cal(self, rtz):
        return np.zeros_like(rtz)

    def j_vec_ana_cal(self, rtz):
        return np.zeros_like(rtz)

    def gradp_vec_ana_cal(self, rtz):
        return np.zeros_like(rtz)


@dataclass
class model_theta_pinch1(ana_model):
    """
        Theta_pinch configuration: w/o p
        Rlmn: 0,0,0->R; 1,1,0->a0; 3,1,0->a1
        Zlmn: 1,-1,0->-a0; 3,-1,0->-a1
    """
    Psi: float
    a0: float
    a1: float
    R: float

    def xyz(self, rtz):  # this is wrong
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        return np.stack(
            (self.R + rho * np.cos(theta), zeta, -rho * np.sin(theta)), axis=1
        )

    def B_ana_cal(self, rho_in):
        a0 = self.a0
        a1 = self.a1
        Psi0 = self.Psi
        if len(rho_in.shape) > 1:
            rho_in = rho[:, 0]
        else:
            rho = rho_in
        return (
            Psi0
            / np.pi
            / (a0 + a1 * (3 * rho**2 - 2))
            / (a0 - 2 * a1 + 9 * a1 * rho**2)
        )

    def B_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        B = self.B_ana_cal(rho)
        return np.stack((np.zeros_like(rho), B, np.zeros_like(rho)), axis=1)

    def p_ana_cal(self, rho_in):
        a0 = self.a0
        a1 = self.a1
        Psi = self.Psi
        if len(rho_in.shape) > 1:
            rho_in = rho[:, 0]
        else:
            rho = rho_in
        term1 = (a0 - 2 * a1 + 3 * a1 * rho**2) ** 2
        term2 = (a0 - 2 * a1 + 9 * a1 * rho**2) ** 2

        result = -1 / (2 * term1 * term2) * Psi**2 / np.pi**2 / mu_0

        return result

    def j_ana_cal(self, rho_in):
        a0 = self.a0
        a1 = self.a1
        Psi = self.Psi
        if len(rho_in.shape) > 1:
            rho_in = rho[:, 0]
        else:
            rho = rho_in
        numerator = 12 * a1 * rho * (2 * a0 - 4 * a1 + 9 * a1 * rho**2)
        denominator = (3 * a1 * rho**2 + a0 - 2 * a1) ** 2 * (
            9 * a1 * rho**2 + a0 - 2 * a1
        ) ** 3

        result = numerator / denominator * Psi / np.pi / mu_0

        return result

    def j_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        j = self.j_ana_cal(rho)
        return np.stack(
            (-j * np.sin(theta), np.zeros_like(rho), -j * np.cos(theta)), axis=1
        )

    def gradp_ana_cal(self, rho):
        return self.j_ana_cal(rho) * self.B_ana_cal(rho)

    def gradp_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        F = self.gradp_ana_cal(rho)
        return np.stack(
            (F * np.cos(theta), np.zeros_like(rho), -F * np.sin(theta)), axis=1
        )


@dataclass
class model_screw_pinch1(ana_model):
    """
        screw pinch configuration: w/o p
        Rlmn: 0,0,0->R; 1,1,0->a0; 3,1,0->a1
        Zlmn: 1,-1,0->-a0; 3,-1,0->-a1
        iota: i0+i2*rho^2
        Model calculated analytically in Mathematica
    """
    Psi: float
    a0: float
    a1: float
    R: float
    i0: float
    i2: float

    def B_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        i0 = self.i0
        i2 = self.i2
        psi0 = self.Psi / np.pi / 2
        B0 = -(2 * psi0 * rho * (i0 + i2 * rho**2) * np.sin(theta)) / (a0 + a1 * (-2 + 9 * rho**2))
        B1 = (2 * psi0) / ((a0 + a1 * (-2 + 3 * rho**2)) * (a0 + a1 * (-2 + 9 * rho**2)))
        B2 = -(2 * psi0 * rho * (i0 + i2 * rho**2) * np.cos(theta)) / (a0 + a1 * (-2 + 9 * rho**2))
        return np.stack((B0,B1,B2), axis=1)
    
    def j_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        i0 = self.i0
        i2 = self.i2
        psi0 = self.Psi / np.pi / 2
        mu0 = mu_0
        j0 = -(24 * a1 * psi0 * rho * (2 * a0 + a1 * (-4 + 9 * rho**2)) * np.sin(theta)) / (mu0 * (a0 + a1 * (-2 + 3 * rho**2))**2 * (a0 + a1 * (-2 + 9 * rho**2))**3)
        j1 = (4 * psi0 * (a0**2 * (i0 + 2 * i2 * rho**2) + 2 * a0 * a1 * (i0 * (-2 + 3 * rho**2) + i2 * rho**2 * (-4 + 9 * rho**2)) + a1**2 * (2 * i2 * rho**2 * (4 - 18 * rho**2 + 27 * rho**4) + i0 * (4 - 12 * rho**2 + 27 * rho**4)))) / (mu0 * (a0 + a1 * (-2 + 3 * rho**2)) * (a0 + a1 * (-2 + 9 * rho**2))**3)
        j2 = -(24 * a1 * psi0 * rho * (2 * a0 + a1 * (-4 + 9 * rho**2)) * np.cos(theta)) / (mu0 * (a0 + a1 * (-2 + 3 * rho**2))**2 * (a0 + a1 * (-2 + 9 * rho**2))**3)
        return np.stack((j0,j1,j2), axis=1)
    
    def gradp_vec_ana_cal(self,rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        i0 = self.i0
        i2 = self.i2
        psi0 = self.Psi / np.pi / 2
        mu0 = mu_0
        complex_expr = 8 * psi0**2 * rho * (-12 * (a0 - 2 * a1) * a1 + (a0 - 2 * a1)**4 * i0**2 + 3 * (2 * a1 * (-9 * a1 + 2 * (a0 - 2 * a1)**3 * i0**2) + (a0 - 2 * a1)**4 * i0 * i2) * rho**2 + 2 * (a0 - 2 * a1)**2 * (36 * a1**2 * i0**2 + 21 * (a0 - 2 * a1) * a1 * i0 * i2 + (a0 - 2 * a1)**2 * i2**2) * rho**4 + 6 * (a0 - 2 * a1) * a1 * (36 * a1**2 * i0**2 + 42 * (a0 - 2 * a1) * a1 * i0 * i2 + 5 * (a0 - 2 * a1)**2 * i2**2) * rho**6 + 9 * a1**2 * (27 * a1**2 * i0**2 + 78 * (a0 - 2 * a1) * a1 * i0 * i2 + 20 * (a0 - 2 * a1)**2 * i2**2) * rho**8 + 243 * a1**3 * i2 * (3 * a1 * i0 + 2 * a0 * i2 - 4 * a1 * i2) * rho**10 + 486 * a1**4 * i2**2 * rho**12)

        f0 = -complex_expr * np.cos(theta) / (mu0 * (a0 + a1 * (-2 + 3 * rho**2))**3 * (a0 + a1 * (-2 + 9 * rho**2))**4)
        f1 = np.zeros_like(rho)
        f2 = complex_expr * np.sin(theta) / (mu0 * (a0 + a1 * (-2 + 3 * rho**2))**3 * (a0 + a1 * (-2 + 9 * rho**2))**4)
        return np.stack((f0,f1,f2), axis=1)


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
    ["const_B", "theta_pinch1","screw_pinch1"],
)
def test_compute_against_ana_straight_coord(config, request, name, func_name):
    eq, ana, grid = request.getfixturevalue(config)
    names = [name]
    data = eq.compute(names, grid=grid)
    rtz = grid.nodes
    data_ana = getattr(ana, func_name)(rtz)
    np.testing.assert_allclose(data[name], data_ana, atol=1e-6, rtol=1e-10)


@pytest.mark.mirror_regression
def test_solved_theta_pinch(theta_pinch):
    pass
