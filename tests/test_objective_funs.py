import unittest
import numpy as np
from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    GenericObjective,
    Energy,
    Volume,
    AspectRatio,
    ToroidalCurrent,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
    MercierStability,
    MagneticWell,
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
        V_compute_scalar = obj.compute_scalar(eq.R_lmn, eq.Z_lmn)
        np.testing.assert_allclose(V_compute_scalar, 10)

    def test_aspect_ratio(self):
        eq = Equilibrium()
        obj = AspectRatio(target=5, weight=2, eq=eq)
        AR = obj.compute(eq.R_lmn, eq.Z_lmn)
        np.testing.assert_allclose(AR, 10)

    def test_energy(self):
        eq = Equilibrium(node_pattern="quad")
        obj = Energy(target=0, weight=(4 * np.pi * 1e-7), eq=eq)
        W = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.c_l, eq.Psi)
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
        np.testing.assert_equal(len(DMerc), obj.grid.num_rho)
        np.testing.assert_allclose(DMerc, 0)

    def test_magnetic_well(self):
        eq = Equilibrium()
        obj = MagneticWell(eq=eq)
        magnetic_well = obj.compute(
            eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.c_l, eq.Psi
        )
        np.testing.assert_equal(len(magnetic_well), obj.grid.num_rho)
        np.testing.assert_allclose(magnetic_well, 0, atol=1e-15)


def test_derivative_modes():
    eq = Equilibrium(M=2, N=1, L=2)
    obj1 = ObjectiveFunction(MagneticWell(), deriv_mode="batched", use_jit=False)
    obj2 = ObjectiveFunction(MagneticWell(), deriv_mode="blocked", use_jit=False)

    obj1.build(eq)
    obj2.build(eq)
    x = obj1.x(eq)
    g1 = obj1.grad(x)
    g2 = obj2.grad(x)
    np.testing.assert_allclose(g1, g2, atol=1e-10)
    J1 = obj1.jac(x)
    J2 = obj2.jac(x)
    np.testing.assert_allclose(J1, J2, atol=1e-10)
    H1 = obj1.hess(x)
    H2 = obj2.hess(x)
    np.testing.assert_allclose(np.diag(H1), np.diag(H2), atol=1e-10)
