import numpy as np
import unittest
import pytest

from desc.coils import Coil, CoilSet
from desc.geometry import FourierRZCurve, FourierXYZCurve, FourierPlanarCurve


class TestCoil(unittest.TestCase):
    def test_biot_savart(self):
        R = 2
        y = 1
        I = 1
        By_true = 1e-7 * 2 * np.pi * R ** 2 * I / (y ** 2 + R ** 2) ** (3 / 2)
        B_true = np.array([0, By_true, 0])
        coil = Coil(FourierXYZCurve(), I)
        coil.grid = 100
        assert coil.grid.num_nodes == 100
        B_approx = coil.compute_magnetic_field([10, y, 0], basis="xyz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    def test_properties(self):

        curve = FourierPlanarCurve()
        current = 4.34
        coil = Coil(curve, current)
        assert coil.curve is curve
        assert coil.current == current
        new_current = 3.5
        coil.current = new_current
        assert coil.current == new_current


class TestCoilSet(unittest.TestCase):
    def test_linspaced_linear(self):
        """field from straight solenoid"""
        R = 10
        z = np.linspace(0, 10, 10)
        I = 1
        Bz_true = np.sum(1e-7 * 2 * np.pi * R ** 2 * I / (z ** 2 + R ** 2) ** (3 / 2))
        B_true = np.array([0, 0, Bz_true])
        curve = FourierRZCurve()
        coils = CoilSet.linspaced_linear(curve, I, [0, 0, 10], 10, True)
        coils.grid = 100
        assert coils.grid.num_nodes == 100
        B_approx = coils.compute_magnetic_field([0, 0, z[-1]], basis="xyz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    def test_linspaced_angular(self):
        """field from uniform toroidal solenoid"""
        R = 10
        N = 50
        I = 1
        Bp_true = np.sum(1e-7 * 4 * np.pi * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        curve = FourierPlanarCurve()
        coils = CoilSet.linspaced_angular(curve, I, n=N)
        coils.grid = 100
        assert all([coil.grid.num_nodes == 100 for coil in coils])
        B_approx = coils.compute_magnetic_field([10, 0, 0], basis="rpz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    def test_from_symmetry(self):
        """same as above, but different construction"""
        R = 10
        N = 48
        I = 1
        Bp_true = np.sum(1e-7 * 4 * np.pi * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        curve = FourierPlanarCurve()
        coils = CoilSet.linspaced_angular(curve, I, angle=np.pi / 2, n=N // 4)
        coils = CoilSet.from_symmetry(coils, NFP=4)
        coils.grid = 100
        assert all([coil.grid.num_nodes == 100 for coil in coils])
        B_approx = coils.compute_magnetic_field([10, 0, 0], basis="rpz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

        # with stellarator symmetry
        NFP = 4
        curve = FourierXYZCurve()
        curve.rotate(angle=np.pi / N)
        coils = CoilSet.linspaced_angular(
            curve, I, [0, 0, 1], np.pi / NFP, N // NFP // 2
        )
        coils.grid = 100
        assert coils.grid.num_nodes == 100
        coils2 = CoilSet.from_symmetry(coils, NFP, True)
        B_approx = coils2.compute_magnetic_field([10, 0, 0], basis="rpz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)
