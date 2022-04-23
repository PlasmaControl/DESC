import unittest
import numpy as np
from desc.equilibrium import Equilibrium
from desc.objectives import (
    Energy,
    Volume,
    AspectRatio,
    ToroidalCurrent,
    ToroidalCurrentDensity,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
)


class TestObjectiveFunction(unittest.TestCase):
    """Test ObjectiveFunction class."""

    def test_volume(self):
        eq = Equilibrium()
        obj = Volume(target=10 * np.pi ** 2, weight=1 / np.pi ** 2)
        obj.build(eq)
        V = obj.compute(eq.R_lmn, eq.Z_lmn)
        np.testing.assert_allclose(V, 10)

    def test_aspect_ratio(self):
        eq = Equilibrium()
        obj = AspectRatio(target=5, weight=2)
        obj.build(eq)
        AR = obj.compute(eq.R_lmn, eq.Z_lmn)
        np.testing.assert_allclose(AR, 10)

    def test_energy(self):
        eq = Equilibrium()
        obj = Energy(target=0, weight=(4 * np.pi * 1e-7))
        obj.build(eq)
        W = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.p_l, eq.Psi)
        np.testing.assert_allclose(W, 10)

    def test_toroidal_current(self):
        eq = Equilibrium()
        obj = ToroidalCurrent(target=1, weight=2)
        obj.build(eq)
        I = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(I, -2)

    def test_toroidal_current_density(self):
        eq = Equilibrium()
        obj = ToroidalCurrentDensity(target=1, weight=2)
        obj.build(eq)
        Jt = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(Jt, -2)

    def test_qs_boozer(self):
        eq = Equilibrium()
        obj = QuasisymmetryBoozer()
        obj.build(eq)
        fb = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(fb, 0)

    def test_qs_twoterm(self):
        eq = Equilibrium()
        obj = QuasisymmetryTwoTerm()
        obj.build(eq)
        fc = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(fc, 0)

    def test_qs_tp(self):
        eq = Equilibrium()
        obj = QuasisymmetryTripleProduct()
        obj.build(eq)
        ft = obj.compute(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.i_l, eq.Psi)
        np.testing.assert_allclose(ft, 0)
