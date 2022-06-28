import numpy as np
import pytest
import unittest
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.grid import ConcentricGrid
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
        np.testing.assert_allclose(eq.p_l, [10, 5])
        np.testing.assert_allclose(eq.i_l, [1, 3])
        self.assertIsInstance(eq.surface, FourierRZToroidalSurface)
        np.testing.assert_allclose(
            eq.Rb_lmn,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                10.0,
                1.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        np.testing.assert_allclose(
            eq.Zb_lmn,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )

        inputs["surface"] = np.array([[0, 0, 0, 10, 0], [1, 1, 0, 1, 1]])
        eq = Equilibrium(**inputs)
        self.assertEqual(eq.bdry_mode, "poincare")
        np.testing.assert_allclose(
            eq.Rb_lmn, [10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

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
    def test_default_set(self):
        eq = Equilibrium()
        eq.set_initial_guess()
        np.testing.assert_allclose(eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 1 * 1)
        del eq._axis
        eq.set_initial_guess()
        np.testing.assert_allclose(eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 1 * 1)

    def test_errors(self):

        eq = Equilibrium()
        with pytest.raises(ValueError):
            eq.set_initial_guess(1, "a", 4, 5, 6)
        with pytest.raises(ValueError):
            eq.set_initial_guess(1, 2)
        with pytest.raises(ValueError):
            eq.set_initial_guess(eq, eq.surface)
        with pytest.raises(TypeError):
            eq.set_initial_guess(eq.surface, [1, 2, 3])
        del eq._surface
        with pytest.raises(ValueError):
            eq.set_initial_guess()

        with pytest.raises(ValueError):
            eq.set_initial_guess("path", 3)
        with pytest.raises(ValueError):
            eq.set_initial_guess("path", "hdf5")
        with pytest.raises(ValueError):
            eq.surface = eq.get_surface_at(rho=1)
            eq.change_resolution(2, 2, 2)
            eq._initial_guess_surface(eq.R_basis, eq.R_lmn, eq.R_basis)
        with pytest.raises(ValueError):
            eq._initial_guess_surface(
                eq.R_basis, eq.surface.R_lmn, eq.surface.R_basis, mode="foo"
            )

    def test_guess_from_other(self):

        eq1 = Equilibrium(L=4, M=2)
        eq2 = Equilibrium(L=2, M=1)
        eq2.set_initial_guess(eq1)

        eq2.change_resolution(L=4, M=2)
        np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn)
        np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn)

    def test_guess_from_surface(self):

        eq = Equilibrium()
        surface = FourierRZToroidalSurface()
        # turn the circular cross section into an elipse w AR=2
        surface.set_coeffs(m=-1, n=0, R=None, Z=2)
        # move z axis up to 0.5 for no good reason
        axis = FourierRZCurve([0, 10, 0], [0, 0.5, 0])
        eq.set_initial_guess(surface, axis)
        np.testing.assert_allclose(eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 2 * 1)

    def test_guess_from_surface2(self):

        eq = Equilibrium()
        # specify an interior flux surface
        surface = FourierRZToroidalSurface(rho=0.5)
        eq.set_initial_guess(surface)
        np.testing.assert_allclose(
            eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 2 ** 2
        )

    def test_guess_from_points(self):
        eq = Equilibrium(L=3, M=3, N=1)
        # these are just the default circular tokamak with a random normal
        # perturbation with std=0.03, fixed for repeatability
        eq.R_lmn = np.array(
            [
                3.94803875e-02,
                7.27321367e-03,
                -8.88095373e-03,
                1.47523628e-02,
                1.18518478e-02,
                -2.61657165e-02,
                -1.27473081e-02,
                3.26441003e-02,
                4.47427817e-03,
                1.24734770e-02,
                9.99231496e00,
                -2.74400311e-03,
                1.00447777e00,
                3.22285107e-02,
                1.16571026e-02,
                -3.15868165e-03,
                -6.77657739e-04,
                -1.97894171e-02,
                2.13535622e-02,
                -2.19703593e-02,
                5.15586341e-02,
                3.39651128e-02,
                -1.66077603e-02,
                -2.20514583e-02,
                -3.13335598e-02,
                7.16090760e-02,
                -1.30064709e-03,
                -4.00687024e-02,
                5.25583677e-02,
                4.04325991e-03,
            ]
        )
        eq.Z_lmn = np.array(
            [
                2.58179465e-02,
                -6.58108612e-03,
                3.67459870e-02,
                9.32236734e-04,
                -2.07982449e-03,
                -1.67700140e-02,
                2.56951390e-02,
                -4.49230035e-04,
                9.93325894e-02,
                4.28162330e-03,
                9.39812383e-03,
                9.95829268e-01,
                4.14468984e-02,
                -3.10725101e-02,
                -1.42026152e-02,
                -2.20423483e-02,
                -1.37389716e-02,
                -1.31592276e-02,
                -3.13922472e-02,
                1.88145630e-03,
                2.72255620e-02,
                -9.42746650e-03,
                2.15264372e-02,
                2.43549268e-02,
                5.33383228e-02,
                1.65948808e-02,
                1.45908076e-03,
                1.85101895e-02,
                1.25967662e-02,
                -2.07374046e-02,
            ]
        )
        grid = ConcentricGrid(L=6, M=6, N=2, node_pattern="ocs")
        coords = eq.compute("R", grid)
        coords = eq.compute("lambda", grid, data=coords)
        eq2 = Equilibrium(L=3, M=3, N=1)
        eq2.set_initial_guess(grid.nodes, coords["R"], coords["Z"], coords["lambda"])
        np.testing.assert_allclose(eq.R_lmn, eq2.R_lmn, atol=1e-8)
        np.testing.assert_allclose(eq.Z_lmn, eq2.Z_lmn, atol=1e-8)
        np.testing.assert_allclose(eq.L_lmn, eq2.L_lmn, atol=1e-8)


def test_guess_from_file(SOLOVEV):

    path = SOLOVEV["desc_h5_path"]
    eq1 = Equilibrium(M=12, sym=True)
    eq1.set_initial_guess(path)
    eq2 = EquilibriaFamily.load(path)[-1]

    np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn)
    np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn)


class TestSurfaces(unittest.TestCase):
    def test_get_rho_surface(self):
        eq = Equilibrium()
        surf = eq.get_surface_at(rho=0.5)
        np.testing.assert_allclose(
            surf.compute_surface_area(), 4 * np.pi ** 2 * 10 * 0.5
        )
        assert surf.rho == 0.5

    def test_get_zeta_surface(self):
        eq = Equilibrium()
        surf = eq.get_surface_at(zeta=np.pi)
        np.testing.assert_allclose(surf.compute_surface_area(), np.pi * (1.0) ** 2)
        assert surf.zeta == np.pi

    def test_get_theta_surface(self):
        eq = Equilibrium()
        with pytest.raises(NotImplementedError):
            surf = eq.get_surface_at(theta=np.pi)

    def test_asserts(self):
        eq = Equilibrium()
        with pytest.raises(ValueError):
            surf = eq.get_surface_at(rho=1, zeta=2)
        with pytest.raises(AssertionError):
            surf = eq.get_surface_at(rho=1.2)
