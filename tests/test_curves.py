import numpy as np
import unittest
import pytest

from desc.geometry import FourierRZCurve, FourierXYZCurve, FourierPlanarCurve


class TestRZCurve(unittest.TestCase):
    def test_length(self):
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 10 * 2 * np.pi)

    def test_curvature(self):
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 10)

    def test_torsion(self):
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)

    def test_frenet(self):
        c = FourierRZCurve()
        T, N, B = c.compute_frenet_frame(grid=np.array([[0, 0, 0]]))
        np.testing.assert_allclose(T, np.array([[0, 1, 0]]))
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]))
        np.testing.assert_allclose(B, np.array([[0, 0, 1]]))


class TestXYZCurve(unittest.TestCase):
    def test_length(self):
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)

    def test_curvature(self):
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    def test_torsion(self):
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)

    def test_frenet(self):
        c = FourierXYZCurve()
        T, N, B = c.compute_frenet_frame(grid=np.array([[0, 0, 0]]))
        np.testing.assert_allclose(T, np.array([[0, 0, 1]]))
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]))
        np.testing.assert_allclose(B, np.array([[0, -1, 0]]))


class TestPlanarCurve(unittest.TestCase):
    def test_length(self):
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)

    def test_curvature(self):
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    def test_torsion(self):
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)

    def test_frenet(self):
        c = FourierPlanarCurve()
        T, N, B = c.compute_frenet_frame(grid=np.array([[0, 0, 0]]))
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]))
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]))
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]))
