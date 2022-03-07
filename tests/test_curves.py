import numpy as np
import unittest
import pytest

from desc.geometry import FourierRZCurve, FourierXYZCurve, FourierPlanarCurve
from desc.grid import LinearGrid


class TestRZCurve(unittest.TestCase):
    def test_length(self):
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 10 * 2 * np.pi)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_length(grid=20), 10 * 2 * np.pi)

    def test_curvature(self):
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 10)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 10)

    def test_torsion(self):
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)

    def test_frenet(self):
        c = FourierRZCurve()
        T, N, B = c.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="rpz")
        np.testing.assert_allclose(T, np.array([[0, 1, 0]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 0, 1]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        T, N, B = c.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz")
        np.testing.assert_allclose(T, np.array([[0, 1, 0]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 0, 1]]), atol=1e-12)

    def test_coords(self):
        c = FourierRZCurve()
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 10)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 1)
        np.testing.assert_allclose(y, -9)
        np.testing.assert_allclose(z, 1)

    def test_misc(self):
        c = FourierRZCurve()
        grid = LinearGrid(L=1, M=4, N=4)
        c.grid = grid
        assert grid.eq(c.grid)

        R, Z = c.get_coeffs(0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 5, None)
        np.testing.assert_allclose(
            c.R_n,
            [
                5,
            ],
        )
        np.testing.assert_allclose(c.Z_n, [])

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5)
        c.set_coeffs(-1, None, 2)
        np.testing.assert_allclose(
            c.R_n,
            [5, 0, 0, 0, 0, 0],
        )
        np.testing.assert_allclose(c.Z_n, [0, 0, 0, 0, 2])

        with pytest.raises(ValueError):
            c.R_n = s.R_n
        with pytest.raises(ValueError):
            c.Z_n = s.Z_n


class TestXYZCurve(unittest.TestCase):
    def test_length(self):
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)

    def test_curvature(self):
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    def test_torsion(self):
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    def test_frenet(self):
        c = FourierXYZCurve()
        T, N, B = c.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="rpz")
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        T, N, B = c.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz")
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)

    def test_coords(self):
        c = FourierXYZCurve()
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 12)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 1)
        np.testing.assert_allclose(y, -11)
        np.testing.assert_allclose(z, 1)

    def test_misc(self):
        c = FourierXYZCurve()
        grid = LinearGrid(L=1, M=4, N=4)
        c.grid = grid
        assert grid.eq(c.grid)

        X, Y, Z = c.get_coeffs(0)
        np.testing.assert_allclose(X, 10)
        np.testing.assert_allclose(Y, 0)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 5, 2, 3)
        np.testing.assert_allclose(c.X_n, [0, 5, 2])
        np.testing.assert_allclose(c.Y_n, [0, 2, 0])
        np.testing.assert_allclose(c.Z_n, [-2, 3, 0])

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5)
        with pytest.raises(ValueError):
            c.X_n = s.X_n
        with pytest.raises(ValueError):
            c.Y_n = s.Y_n
        with pytest.raises(ValueError):
            c.Z_n = s.Z_n


class TestPlanarCurve(unittest.TestCase):
    def test_length(self):
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)

    def test_curvature(self):
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    def test_torsion(self):
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)

    def test_frenet(self):
        c = FourierPlanarCurve()
        T, N, B = c.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        T, N, B = c.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz")
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)

    def test_coords(self):
        c = FourierPlanarCurve()
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 12)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 1)
        np.testing.assert_allclose(y, -11)
        np.testing.assert_allclose(z, 1)

    def test_misc(self):
        c = FourierPlanarCurve()
        grid = LinearGrid(L=1, M=4, N=4)
        c.grid = grid
        assert grid.eq(c.grid)

        r = c.get_coeffs(0)
        np.testing.assert_allclose(r, 2)
        c.set_coeffs(0, 3)
        np.testing.assert_allclose(
            c.r_n,
            [
                3,
            ],
        )

        c.normal = [1, 2, 3]
        c.center = [3, 2, 1]
        np.testing.assert_allclose(np.linalg.norm(c.normal), 1)
        np.testing.assert_allclose(c.normal * np.linalg.norm(c.center), c.center[::-1])

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5)
        with pytest.raises(ValueError):
            c.r_n = s.r_n
