import unittest
import numpy as np
from netCDF4 import Dataset
import desc.io
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import (
    GenericObjective,
    Energy,
    Volume,
    AspectRatio,
    ToroidalCurrent,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
    MercierStability,
    MercierShear,
    MercierCurr,
    MercierWell,
    MercierGeod,
)

default_range = (0.05, 1)
default_rtol = 1e-2
default_atol = 1e-6


def all_close(
    y1, y2, rho, rho_range=default_range, rtol=default_rtol, atol=default_atol
):
    """
    Test that the values of y1 and y2, over the indices defined by the given range,
    are closer than the given tolerance.

    Parameters
    ----------
    y1 : ndarray
        values to compare
    y2 : ndarray
        values to compare
    rho : ndarray
        rho values
    rho_range : (float, float)
        the range of rho values to compare
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance
    """
    minimum, maximum = rho_range
    interval = np.where((minimum < rho) & (rho < maximum))
    np.testing.assert_allclose(y1[interval], y2[interval], rtol=rtol, atol=atol)


def get_desc_eq(name):
    """
    Parameters
    ----------
    name : str
        Name of the equilibrium.

    Returns
    -------
    eq : Equilibrium
        DESC equilibrium.
    """
    return desc.io.load("examples/DESC/" + name + "_output.h5")[-1]


def get_vmec_data(name, quantity):
    """
    Parameters
    ----------
    name : str
        Name of the equilibrium.
    quantity: str
        Name of the quantity to return.

    Returns
    -------
    :rtype: (ndarray, ndarray)
    rho : ndarray
        Radial coordinate.
    quantity : ndarray
        Variable from VMEC output.
    """
    f = Dataset("examples/VMEC/wout_" + name + ".nc")
    rho = np.sqrt(f.variables["phi"] / np.array(f.variables["phi"])[-1])
    quantity = np.asarray(f.variables[quantity])
    return rho, quantity


def get_grid(eq, rho):
    """
    Parameters
    ----------
    eq : Equilibrium
        DESC equilibrium.
    rho: ndarray
        Rho values over which the grid will be defined.

    Returns
    -------
    :rtype: (ndarray, ndarray)
    grid : LinearGrid
        Defined over rho array with equilibrium resolution.
    """
    return LinearGrid(
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=1,
        sym=False,
        rho=np.atleast_1d(rho),
    )


class TestObjectiveFunction(unittest.TestCase):
    """Test ObjectiveFunction class."""

    def test_generic(self):
        eq = Equilibrium()
        obj = GenericObjective("sqrt(g)", eq=eq)
        kwargs = {"R_lmn": eq.R_lmn, "Z_lmn": eq.Z_lmn}
        B = obj.compute(**kwargs)
        np.testing.assert_allclose(B, eq.compute("sqrt(g)")["sqrt(g)"])

    def test_volume(self):
        eq = Equilibrium()
        obj = Volume(target=10 * np.pi ** 2, weight=1 / np.pi ** 2, eq=eq)
        V = obj.compute(eq.R_lmn, eq.Z_lmn)
        np.testing.assert_allclose(V, 10)

    def test_aspect_ratio(self):
        eq = Equilibrium()
        obj = AspectRatio(target=5, weight=2, eq=eq)
        AR = obj.compute(eq.R_lmn, eq.Z_lmn)
        np.testing.assert_allclose(AR, 10)

    def test_energy(self):
        eq = Equilibrium()
        obj = Energy(target=0, weight=(4 * np.pi * 1e-7), eq=eq)
        W = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.p_l, eq.Psi)
        np.testing.assert_allclose(W, 10)

    def test_toroidal_current(self):
        eq = Equilibrium()
        obj = ToroidalCurrent(target=1, weight=2, eq=eq)
        I = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
        np.testing.assert_allclose(I, -2)

    def test_qs_boozer(self):
        eq = Equilibrium()
        obj = QuasisymmetryBoozer(eq=eq)
        fb = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
        np.testing.assert_allclose(fb, 0)

    def test_qs_twoterm(self):
        eq = Equilibrium()
        obj = QuasisymmetryTwoTerm(eq=eq)
        fc = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
        np.testing.assert_allclose(fc, 0)

    def test_qs_tp(self):
        eq = Equilibrium()
        obj = QuasisymmetryTripleProduct(eq=eq)
        ft = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
        np.testing.assert_allclose(ft, 0)

    def test_mercier_stability(self):
        eq = Equilibrium()
        obj = MercierStability(eq=eq)
        DMerc = obj.compute(
            eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.c_l, eq.Psi
        )
        np.testing.assert_allclose(DMerc, 0, err_msg="should be 0 in vacuum")

        def test(name, rho_range=default_range, rtol=default_rtol, atol=default_atol):
            rho, vmec = get_vmec_data(name, "DMerc")
            eq = get_desc_eq(name)
            grid = get_grid(eq, rho)
            obj = MercierStability(eq=eq, grid=grid)
            DMerc = obj.compute(
                eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.c_l, eq.Psi
            )
            all_close(DMerc, vmec, rho, rho_range, rtol, atol)

        test("DSHAPE", (0.175, 0.8))
        test("DSHAPE", (0.8, 1), atol=5e-2)
        test("HELIOTRON", (0.1, 0.275), rtol=11e-2)
        test("HELIOTRON", (0.275, 0.975), rtol=5e-2)

    def test_mercier_shear(self):
        eq = Equilibrium()
        obj = MercierShear(eq=eq)
        DShear = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
        np.testing.assert_allclose(DShear, 0, err_msg="should be 0 in vacuum")

        def test(name, rho_range=default_range, rtol=default_rtol, atol=default_atol):
            rho, vmec = get_vmec_data(name, "DShear")
            eq = get_desc_eq(name)
            grid = get_grid(eq, rho)
            obj = MercierShear(eq=eq, grid=grid)
            DShear = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
            assert np.all(
                DShear[np.isfinite(DShear)] >= 0
            ), "DShear should always have a stabilizing effect."
            all_close(DShear, vmec, rho, rho_range, rtol, atol)

        test("DSHAPE", (0, 1), 1e-12, 0)
        test("HELIOTRON", (0, 1), 1e-12, 0)

    def test_mercier_curr(self):
        eq = Equilibrium()
        obj = MercierCurr(eq=eq)
        DCurr = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
        np.testing.assert_allclose(DCurr, 0, err_msg="should be 0 in vacuum")

        def test(name, rho_range=default_range, rtol=default_rtol, atol=default_atol):
            rho, vmec = get_vmec_data(name, "DCurr")
            eq = get_desc_eq(name)
            grid = get_grid(eq, rho)
            obj = MercierCurr(eq=eq, grid=grid)
            DCurr = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
            all_close(DCurr, vmec, rho, rho_range, rtol, atol)

        test("DSHAPE", (0.075, 0.975))
        test("HELIOTRON", (0.16, 0.9), rtol=62e-3)

    def test_mercier_well(self):
        eq = Equilibrium()
        obj = MercierWell(eq=eq)
        DWell = obj.compute(
            eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.c_l, eq.Psi
        )
        np.testing.assert_allclose(DWell, 0, err_msg="should be 0 in vacuum")

        def test(name, rho_range=default_range, rtol=default_rtol, atol=default_atol):
            rho, vmec = get_vmec_data(name, "DWell")
            eq = get_desc_eq(name)
            grid = get_grid(eq, rho)
            obj = MercierWell(eq=eq, grid=grid)
            DWell = obj.compute(
                eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.c_l, eq.Psi
            )
            all_close(DWell, vmec, rho, rho_range, rtol, atol)

        test("DSHAPE", (0.11, 0.8))
        test("HELIOTRON", (0.01, 0.45), rtol=176e-3)
        test("HELIOTRON", (0.45, 0.6), atol=6e-1)
        test("HELIOTRON", (0.6, 0.99))

    def test_mercier_geod(self):
        eq = Equilibrium()
        obj = MercierGeod(eq=eq)
        DGeod = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
        np.testing.assert_allclose(DGeod, 0, err_msg="should be 0 in vacuum")

        def test(name, rho_range=default_range, rtol=default_rtol, atol=default_atol):
            rho, vmec = get_vmec_data(name, "DGeod")
            eq = get_desc_eq(name)
            grid = get_grid(eq, rho)
            obj = MercierGeod(eq=eq, grid=grid)
            DGeod = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.c_l, eq.Psi)
            assert np.all(
                DGeod[np.isfinite(DGeod)] <= 0
            ), "DGeod should always have a destabilizing effect."
            all_close(DGeod, vmec, rho, rho_range, rtol, atol)

        test("DSHAPE", (0.15, 0.975))
        test("HELIOTRON", (0.15, 0.825), rtol=77e-3)
        test("HELIOTRON", (0.825, 1), atol=12e-2)
