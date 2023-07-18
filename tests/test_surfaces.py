"""Tests for Surface classes."""

import numpy as np
import pytest

import desc.examples
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface, ZernikeRZToroidalSection
from desc.grid import LinearGrid


class TestFourierRZToroidalSurface:
    """Tests for FourierRZToroidalSurface class."""

    @pytest.mark.unit
    def test_area(self):
        """Test calculation of surface area."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(M=24, N=24)
        s.grid = grid

        area = 4 * np.pi**2 * 10
        np.testing.assert_allclose(s.compute_surface_area(), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=10), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=(10, 15)), area)

    @pytest.mark.unit
    def test_normal(self):
        """Test calculation of surface normal vector."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(theta=np.pi / 2, zeta=np.pi)
        s.grid = grid
        N = s.compute_normal()
        np.testing.assert_allclose(N[0], [0, 0, -1], atol=1e-14)
        grid = LinearGrid(theta=0.0, zeta=0.0)
        s.grid = grid
        N = s.compute_normal(basis="xyz")
        np.testing.assert_allclose(N[0], [1, 0, 0], atol=1e-12)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting attributes of surface."""
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
        assert c.grid.NFP == 3

    @pytest.mark.unit
    def test_from_input_file(self):
        """Test reading a surface from a vmec or desc input file."""
        vmec_path = ".//tests//inputs//input.DSHAPE"
        desc_path = ".//tests//inputs//DSHAPE"
        with pytest.warns(UserWarning):
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

    @pytest.mark.unit
    def test_from_near_axis(self):
        """Test constructing approximate QI surface from near axis parameters."""
        surf = FourierRZToroidalSurface.from_near_axis(10, 4, 0.3, 0.2)
        np.testing.assert_allclose(
            surf.R_lmn,
            np.array([0.075, 0, 1, 0.125, 0, 0.0150853, -0.2, 0.075]),
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            surf.Z_lmn,
            np.array([0.2, -0.075, 0, 0, -0.125, -0.00377133, 0.075]),
            rtol=1e-4,
            atol=1e-6,
        )

    @pytest.mark.unit
    def test_curvature(self):
        """Tests for gaussian, mean, principle curvatures."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(theta=np.pi / 2, zeta=np.pi)
        s.grid = grid
        K, H, k1, k2 = s.compute_curvature()
        np.testing.assert_allclose(K, 0)
        np.testing.assert_allclose(H, -1 / 2)
        np.testing.assert_allclose(k1, 0)
        np.testing.assert_allclose(k2, -1)


class TestZernikeRZToroidalSection:
    """Tests for ZerinkeRZTorioidalSection class."""

    @pytest.mark.unit
    def test_area(self):
        """Test calculation of surface area."""
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=10, M=10)
        s.grid = grid

        area = np.pi * 1**2
        np.testing.assert_allclose(s.compute_surface_area(), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=15), area)
        np.testing.assert_allclose(s.compute_surface_area(grid=(5, 5)), area)

    @pytest.mark.unit
    def test_normal(self):
        """Test calculation of surface normal vector."""
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=8, M=4, N=0)
        s.grid = grid
        N = s.compute_normal(basis="xyz")
        np.testing.assert_allclose(N, np.broadcast_to([0, 1, 0], N.shape), atol=1e-12)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting surface attributes."""
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

    @pytest.mark.unit
    def test_curvature(self):
        """Tests for gaussian, mean, principle curvatures.

        (kind of pointless since it's a flat surface so its always 0)
        """
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(theta=np.pi / 2, rho=0.5)
        s.grid = grid
        K, H, k1, k2 = s.compute_curvature()
        np.testing.assert_allclose(K, 0)
        np.testing.assert_allclose(H, 0)
        np.testing.assert_allclose(k1, 0)
        np.testing.assert_allclose(k2, 0)


@pytest.mark.unit
def test_surface_orientation():
    """Tests for computing the orientation of a surface in weird edge cases."""
    # this has the axis outside the surface, and negative orientation
    Rb = np.array([3.41, 0.8, 0.706, -0.3])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, -0.16, 1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == -1
    eq = Equilibrium(M=surf.M, N=surf.N, surface=surf)
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == -1

    # same surface but flipped to have positive orientation
    Rb = np.array([3.41, 0.8, 0.706, -0.3])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, 0.16, -1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == 1
    eq = Equilibrium(M=surf.M, N=surf.N, surface=surf)
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == 1

    # this has theta=0 on inboard side and positive orientation
    Rb = np.array([3.51, -1.3, -0.506, 0.1])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, -0.16, 1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == 1
    eq = Equilibrium(M=surf.M, N=surf.N, surface=surf)
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == 1

    # same but with negative orientation
    Rb = np.array([3.51, -1.3, -0.506, 0.1])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, 0.16, -1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == -1
    eq = Equilibrium(M=surf.M, N=surf.N, surface=surf)
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == -1
