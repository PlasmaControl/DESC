import numpy as np
import unittest
import pytest

from desc.geometry import FourierRZToroidalSurface, ZernikeRZToroidalSection
from desc.grid import LinearGrid
import desc.examples


class TestFourierRZToroidalSurface(unittest.TestCase):
    def test_area(self):
        s = FourierRZToroidalSurface()
        grid = LinearGrid(M=24, N=24)
        s.grid = grid

        area = 4 * np.pi ** 2 * 10
        np.testing.assert_allclose(s.compute_surface_area(), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=10), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=(10, 15)), area)

    def test_normal(self):
        s = FourierRZToroidalSurface()
        grid = LinearGrid(theta=np.pi / 2, zeta=np.pi)
        s.grid = grid
        N = s.compute_normal()
        # note default surface is left handed
        np.testing.assert_allclose(N[0], [0, 0, -1], atol=1e-14)
        grid = LinearGrid(theta=0.0, zeta=0.0)
        s.grid = grid
        N = s.compute_normal(basis="xyz")
        np.testing.assert_allclose(N[0], [-1, 0, 0], atol=1e-12)

    def test_misc(self):
        c = FourierRZToroidalSurface()
        grid = LinearGrid(L=0, M=2, N=2)
        c.grid = grid
        assert grid.eq(c.grid)

        R, Z = c.get_coeffs(0, 0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 0, 5, None)
        c.set_coeffs(-1, 0, None, 2)
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
                2,
            ],
        )

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(0, 5, 5)
        with pytest.raises(ValueError):
            c.R_lmn = s.R_lmn
        with pytest.raises(ValueError):
            c.Z_lmn = s.Z_lmn

        c.name = "my curve"
        assert "my" in c.name
        assert c.name in str(c)
        assert "FourierRZToroidalSurface" in str(c)

        c.NFP = 3
        assert c.NFP == 3
        assert c.R_basis.NFP == 3
        assert c.Z_basis.NFP == 3

    def test_from_input_file(self):

        vmec_path = ".//tests//inputs//input.DSHAPE"
        desc_path = ".//tests//inputs//DSHAPE"
        vmec_surf = FourierRZToroidalSurface.from_input_file(vmec_path)
        desc_surf = FourierRZToroidalSurface.from_input_file(desc_path)
        true_surf = desc.examples.get("DSHAPE", "boundary")

        vmec_surf.change_resolution(M=6, N=0)
        desc_surf.change_resolution(M=6, N=0)
        true_surf.change_resolution(M=6, N=0)

        np.testing.assert_allclose(
            true_surf.R_lmn, vmec_surf.R_lmn, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            true_surf.Z_lmn, vmec_surf.Z_lmn, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            true_surf.R_lmn, desc_surf.R_lmn, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            true_surf.Z_lmn, desc_surf.Z_lmn, atol=1e-10, rtol=1e-10
        )


class TestZernikeRZToroidalSection(unittest.TestCase):
    def test_area(self):
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=10, M=10)
        s.grid = grid

        area = np.pi * 1 ** 2
        np.testing.assert_allclose(s.compute_surface_area(), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=15), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=(5, 5)), area)

    def test_normal(self):
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=8, M=4, N=0)
        s.grid = grid
        N = s.compute_normal(basis="xyz")
        np.testing.assert_allclose(N, np.broadcast_to([0, 1, 0], N.shape), atol=1e-12)

    def test_misc(self):
        c = ZernikeRZToroidalSection()
        grid = LinearGrid(L=2, M=2, N=0)
        c.grid = grid
        assert grid.eq(c.grid)

        R, Z = c.get_coeffs(0, 0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 0, 5, None)
        c.set_coeffs(1, -1, None, 2)
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
                2,
            ],
        )
        with pytest.raises(ValueError):
            c.set_coeffs(0, 0, None, 2)
        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5, 5, 0)
        with pytest.raises(ValueError):
            c.R_lmn = s.R_lmn
        with pytest.raises(ValueError):
            c.Z_lmn = s.Z_lmn

        assert c.sym
