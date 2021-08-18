import numpy as np
import pytest
import unittest
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.geometry import (
    FourierRZCurve,
    FourierRZToroidalSurface,
    ZernikeRZToroidalSection,
)


class TestConstructor(unittest.TestCase):
    def test_defaults(self):

        eq = Equilibrium()

        self.assertEqual(eq.spectral_indexing, "ansi")
        self.assertEqual(eq.NFP, 1)
        self.assertEqual(eq.L, 1)
        self.assertEqual(eq.M, 1)
        self.assertEqual(eq.N, 0)
        self.assertEqual(eq.sym, False)
        self.assertTrue(eq.surface.eq(FourierRZToroidalSurface()))
        self.assertIsInstance(eq.pressure, PowerSeriesProfile)
        np.testing.assert_allclose(eq.p_l, [0])
        self.assertIsInstance(eq.iota, PowerSeriesProfile)
        np.testing.assert_allclose(eq.i_l, [0])

    def test_supplied_objects(self):

        pressure = SplineProfile([1, 2, 3])
        iota = SplineProfile([2, 3, 4])
        surface = ZernikeRZToroidalSection(spectral_indexing="ansi")
        axis = FourierRZCurve([-1, 10, 1], [1, 0, -1], NFP=2)

        eq = Equilibrium(pressure=pressure, iota=iota, surface=surface, axis=axis)

        self.assertTrue(eq.pressure.eq(pressure))
        self.assertTrue(eq.iota.eq(iota))
        self.assertTrue(eq.surface.eq(surface))
        self.assertTrue(eq.axis.eq(axis))
        self.assertEqual(eq.spectral_indexing, "ansi")
        self.assertEqual(eq.NFP, 2)

        surface2 = FourierRZToroidalSurface(NFP=3)
        eq2 = Equilibrium(surface=surface2)
        self.assertEqual(eq2.NFP, 3)
        self.assertEqual(eq2.axis.NFP, 3)

        eq3 = Equilibrium(surface=surface, axis=None)
        np.testing.assert_allclose(eq3.axis.R_n, [10])

    def test_dict(self):

        inputs = {
            "L": 4,
            "M": 2,
            "N": 2,
            "NFP": 3,
            "sym": False,
            "spectral_indexing": "ansi",
            "surface": np.array(
                [[0, 0, 0, 10, 0], [0, 1, 0, 1, 1], [0, -1, 1, 0.1, 0.1]]
            ),
            "axis": np.array([[0, 10, 0]]),
            "pressure": np.array([[0, 10], [2, 5]]),
            "iota": np.array([[0, 1], [2, 3]]),
        }
        eq = Equilibrium(**inputs)

        self.assertEqual(eq.L, 4)
        self.assertEqual(eq.M, 2)
        self.assertEqual(eq.N, 2)
        self.assertEqual(eq.NFP, 3)
        self.assertEqual(eq.spectral_indexing, "ansi")
        np.testing.assert_allclose(eq.p_l, [10, 0, 5])
        np.testing.assert_allclose(eq.i_l, [1, 0, 3])
        self.assertIsInstance(eq.surface, FourierRZToroidalSurface)
        np.testing.assert_allclose(eq.Rb_lmn, [0, 0, 0, 0, 10, 1, 0.1, 0, 0])
        np.testing.assert_allclose(eq.Zb_lmn, [0, 0, 0, 0, 0, 1, 0.1, 0, 0])

        inputs["surface"] = np.array([[0, 0, 0, 10, 0], [1, 1, 0, 1, 1]])
        eq = Equilibrium(**inputs)
        self.assertEqual(eq.bdry_mode, "poincare")
        np.testing.assert_allclose(eq.Rb_lmn, [10, 0, 1])

    def test_asserts(self):

        with pytest.raises(AssertionError):
            eq = Equilibrium(L=3.4)
        with pytest.raises(AssertionError):
            eq = Equilibrium(M=3.4)
        with pytest.raises(AssertionError):
            eq = Equilibrium(N=3.4)
        with pytest.raises(AssertionError):
            eq = Equilibrium(NFP=3.4j)
        with pytest.raises(ValueError):
            eq = Equilibrium(surface=np.array([[1, 1, 1, 10, 2]]))
        with pytest.raises(TypeError):
            eq = Equilibrium(surface=FourierRZCurve())
        with pytest.raises(TypeError):
            eq = Equilibrium(axis=2)
        with pytest.raises(ValueError):
            eq = Equilibrium(surface=FourierRZToroidalSurface(NFP=1), NFP=2)
        with pytest.raises(TypeError):
            eq = Equilibrium(pressure="abc")
        with pytest.raises(TypeError):
            eq = Equilibrium(iota="def")

    def test_supplied_coeffs(self):

        R_lmn = np.random.random(3)
        Z_lmn = np.random.random(3)
        L_lmn = np.random.random(3)
        eq = Equilibrium(R_lmn=R_lmn, Z_lmn=Z_lmn, L_lmn=L_lmn)
        np.testing.assert_allclose(R_lmn, eq.R_lmn)
        np.testing.assert_allclose(Z_lmn, eq.Z_lmn)
        np.testing.assert_allclose(L_lmn, eq.L_lmn)

        with pytest.raises(ValueError):
            eq = Equilibrium(L=4, R_lmn=R_lmn)


class TestInitialGuess(unittest.TestCase):
    def test_errors(self):

        eq = Equilibrium()
        with pytest.raises(ValueError):
            eq.set_initial_guess(1, "a", 4)
        with pytest.raises(ValueError):
            eq.set_initial_guess(1, 2)
        with pytest.raises(ValueError):
            eq.set_initial_guess(eq, eq.surface)
        del eq._surface
        with pytest.raises(ValueError):
            eq.set_initial_guess()

        with pytest.raises(ValueError):
            eq.set_initial_guess("path", 3)
        with pytest.raises(ValueError):
            eq.set_initial_guess("path", "hdf5")

    def test_guess_from_other(self):

        eq1 = Equilibrium(L=4, M=2)
        eq2 = Equilibrium(L=2, M=1)
        eq2.set_initial_guess(eq1)

        eq2.change_resolution(L=4, M=2)
        np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn)
        np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn)

    def test_guess_from_file(self):

        eq1 = Equilibrium(L=24, M=12, sym=True, spectral_indexing="fringe")
        path = "./tests/inputs/SOLOVEV_output.h5"
        eq1.set_initial_guess(path)
        eq2 = EquilibriaFamily.load(path)

        np.testing.assert_allclose(eq1.R_lmn, eq2[-1].R_lmn)
        np.testing.assert_allclose(eq1.Z_lmn, eq2[-1].Z_lmn)

    def test_guess_from_surface(self):

        eq = Equilibrium()
        surface = FourierRZToroidalSurface()
        # turn the circular cross section into an elipse w AR=2
        surface.set_coeffs(m=-1, n=0, R=None, Z=2)
        # move z axis up to 0.5 for no good reason
        axis = FourierRZCurve([0, 10, 0], [0, 0.5, 0])
        eq.set_initial_guess(surface, axis)

        np.testing.assert_allclose(eq.compute_volume(), 2 * 10 * np.pi * np.pi * 2 * 1)
