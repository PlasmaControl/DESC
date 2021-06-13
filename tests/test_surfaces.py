import numpy as np
import unittest
import pytest

from desc.geometry import FourierRZToroidalSurface, ZernikeToroidalSection
from desc.grid import Grid, LinearGrid


class TestFourierRZToroidalSurface(unittest.TestCase):
    def test_area(self):
        s = FourierRZToroidalSurface()
        grid = LinearGrid(L=1, M=50, N=50)
        s.grid = grid
        np.testing.assert_allclose(s.compute_surface_area(), 4 * np.pi ** 2 * 10)

    def test_normal(self):
        s = FourierRZToroidalSurface()
        grid = LinearGrid(L=1, theta=np.pi / 2, zeta=np.pi)
        s.grid = grid
        N = s.compute_normal()
        # note default surface is left handed
        np.testing.assert_allclose(N[0], [0, 0, -1], atol=1e-14)
        grid = LinearGrid(L=1, theta=0, zeta=0)
        s.grid = grid
        N = s.compute_normal()
        np.testing.assert_allclose(N[0], [-1, 0, 0])


class TestZernikeToroidalSection(unittest.TestCase):
    def test_area(self):
        s = ZernikeToroidalSection()
        grid = LinearGrid(L=20, M=20, N=1)
        s.grid = grid
        np.testing.assert_allclose(s.compute_surface_area(), np.pi * 1 ** 2)

    def test_normal(self):
        s = ZernikeToroidalSection()
        grid = LinearGrid(L=20, M=20, N=1)
        s.grid = grid
        N = s.compute_normal()
        np.testing.assert_allclose(N, np.broadcast_to([0, 1, 0], N.shape))
