import numpy as np
import unittest
import pytest

from desc.geometry import FourierRZToroidalSurface, ZernikeRZToroidalSection
from desc.grid import Grid, LinearGrid


class TestFourierRZToroidalSurface(unittest.TestCase):
    def test_area(self):
        s = FourierRZToroidalSurface()
        grid = LinearGrid(L=1, M=50, N=50)
        s.grid = grid
        np.testing.assert_allclose(s.compute_surface_area(), 4 * np.pi ** 2 * 10)
        np.testing.assert_allclose(s.compute_surface_area(grid=20), 4 * np.pi ** 2 * 10)
        np.testing.assert_allclose(
            s.compute_surface_area(grid=(20, 30)), 4 * np.pi ** 2 * 10
        )

    def test_normal(self):
        s = FourierRZToroidalSurface()
        grid = LinearGrid(L=1, theta=np.pi / 2, zeta=np.pi)
        s.grid = grid
        N = s.compute_normal()
        # note default surface is left handed
        np.testing.assert_allclose(N[0], [0, 0, -1], atol=1e-14)
        grid = LinearGrid(L=1, theta=0, zeta=0)
        s.grid = grid
        N = s.compute_normal(basis="xyz")
        np.testing.assert_allclose(N[0], [-1, 0, 0], atol=1e-12)

    def test_misc(self):
        c = FourierRZToroidalSurface()
        grid = LinearGrid(L=1, M=4, N=4)
        c.grid = grid
        assert grid.eq(c.grid)

        R, Z = c.get_coeffs(0, 0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 0, 5, 0)
        np.testing.assert_allclose(
            c.R_lmn,
            [
                5,
                1,
            ],
        )
        np.testing.assert_allclose(
            c.Z_lmn,
            [
                1,
            ],
        )

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5, 5)
        with pytest.raises(ValueError):
            c.R_lmn = s.R_lmn
        with pytest.raises(ValueError):
            c.Z_lmn = s.Z_lmn


class TestZernikeRZToroidalSection(unittest.TestCase):
    def test_area(self):
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=20, M=20, N=1)
        s.grid = grid
        np.testing.assert_allclose(s.compute_surface_area(), np.pi * 1 ** 2)
        np.testing.assert_allclose(s.compute_surface_area(grid=30), np.pi * 1 ** 2)
        np.testing.assert_allclose(
            s.compute_surface_area(grid=(10, 10)), np.pi * 1 ** 2
        )

    def test_normal(self):
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=20, M=20, N=1)
        s.grid = grid
        N = s.compute_normal(basis="xyz")
        np.testing.assert_allclose(N, np.broadcast_to([0, 1, 0], N.shape), atol=1e-12)

    def test_misc(self):
        c = ZernikeRZToroidalSection()
        grid = LinearGrid(L=4, M=4, N=1)
        c.grid = grid
        assert grid.eq(c.grid)

        R, Z = c.get_coeffs(0, 0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 0, 5, 0)
        np.testing.assert_allclose(
            c.R_lmn,
            [
                5,
                1,
            ],
        )
        np.testing.assert_allclose(
            c.Z_lmn,
            [
                1,
            ],
        )

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5, 5)
        with pytest.raises(ValueError):
            c.R_lmn = s.R_lmn
        with pytest.raises(ValueError):
            c.Z_lmn = s.Z_lmn
