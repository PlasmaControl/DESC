import unittest
import numpy as np
from desc.equilibrium import Equilibrium
from desc.objectives import (
    GenericObjective,
    Energy,
    Volume,
    AspectRatio,
    ToroidalCurrent,
    MagneticWell,
    RadialCurrentDensity,
    PoloidalCurrentDensity,
    ToroidalCurrentDensity,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
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
        W = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.p_l, eq.Psi)
        np.testing.assert_allclose(W, 10)

    def test_toroidal_current(self):
        eq = Equilibrium()
        obj = ToroidalCurrent(target=1, weight=2, eq=eq)
        I = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(I, -2)

    def test_radial_current_density(self):
        eq = Equilibrium()
        obj = RadialCurrentDensity(eq=eq)
        Jr = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(np.sum(Jr), 0, atol=2e-8)

    def test_poloidal_current_density(self):
        eq = Equilibrium()
        obj = PoloidalCurrentDensity(eq=eq)
        Jt = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(np.sum(Jt), 0, atol=2e-8)

    def test_toroidal_current_density(self):
        eq = Equilibrium()
        obj = ToroidalCurrentDensity(target=1, weight=2, eq=eq)
        Jt = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(Jt, -2)

    def test_qs_boozer(self):
        eq = Equilibrium()
        obj = QuasisymmetryBoozer(eq=eq)
        fb = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(fb, 0)

    def test_qs_twoterm(self):
        eq = Equilibrium()
        obj = QuasisymmetryTwoTerm(eq=eq)
        fc = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(fc, 0)

    def test_qs_tp(self):
        eq = Equilibrium()
        obj = QuasisymmetryTripleProduct(eq=eq)
        ft = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(ft, 0)

    def test_magnetic_well(self):
        eq = Equilibrium()
        obj = MagneticWell(eq=eq)
        w = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.Psi)
        np.testing.assert_allclose(w, 0)  # should fail this
