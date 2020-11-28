import unittest
import numpy as np

from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import PowerSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.transform import Transform


class TestTransform(unittest.TestCase):
    """Tests Transform classes"""

    def test_transform_order_error(self):
        """Tests error handling with transform method
        """
        grid = LinearGrid(L=11, endpoint=True)
        basis = PowerSeries(L=2)
        transf = Transform(grid, basis, order=0)

        # invalid derivative orders
        with self.assertRaises(ValueError):
            c = np.array([1, 2, 3])
            transf.transform(c, 0, 0, 1)

        # incompatible number of coefficients
        with self.assertRaises(ValueError):
            c = np.array([1, 2])
            transf.transform(c, 0, 0, 0)

    def test_profile(self):
        """Tests transform of power series on a radial profile
        """
        grid = LinearGrid(L=11, endpoint=True)
        basis = PowerSeries(L=2)
        transf = Transform(grid, basis, order=1)

        x = grid.nodes[0, :]
        c = np.array([-1, 2, 1])

        values = transf.transform(c, 0, 0, 0)
        derivs = transf.transform(c, 1, 0, 0)

        correct_vals = c[0] + c[1]*x + c[2]*x**2
        correct_ders = c[1] + c[2]*2*x

        np.testing.assert_almost_equal(correct_vals, values)
        np.testing.assert_almost_equal(correct_ders, derivs)

    def test_surface(self):
        """Tests transform of double Fourier series on a flux surface
        """
        grid = LinearGrid(M=4, N=4)
        basis = DoubleFourierSeries(M=1, N=1)
        transf = Transform(grid, basis, order=1)

        t = grid.nodes[1, :]    # theta coordinates
        z = grid.nodes[2, :]    # zeta coordinates

        correct_d0 = np.sin(t-z) + 2*np.cos(t-z)
        correct_dt = np.cos(t-z) - 2*np.sin(t-z)
        correct_dz = -np.cos(t-z) + 2*np.sin(t-z)
        correct_dtz = np.sin(t-z) + 2*np.cos(t-z)

        sin_idx_1 = np.where(np.all(np.array([np.array(basis.modes[:,1] == -1),
                                  np.array(basis.modes[:,2] == 1)]), axis=0))
        sin_idx_2 = np.where(np.all(np.array([np.array(basis.modes[:,1] == 1),
                                  np.array(basis.modes[:,2] == -1)]), axis=0))
        cos_idx_1 = np.where(np.all(np.array([np.array(basis.modes[:,1] == -1),
                                  np.array(basis.modes[:,2] == -1)]), axis=0))
        cos_idx_2 = np.where(np.all(np.array([np.array(basis.modes[:,1] == 1),
                                  np.array(basis.modes[:,2] == 1)]), axis=0))

        c = np.zeros((basis.modes.shape[0],))
        c[sin_idx_1] = 1
        c[sin_idx_2] = -1
        c[cos_idx_1] = 2
        c[cos_idx_2] = 2

        d0 = transf.transform(c, 0, 0, 0)   # original transform
        dt = transf.transform(c, 0, 1, 0)   # theta derivative
        dz = transf.transform(c, 0, 0, 1)   # zeta derivative
        dtz = transf.transform(c, 0, 1, 1)  # mixed derivative

        np.testing.assert_almost_equal(correct_d0, d0)
        np.testing.assert_almost_equal(correct_dt, dt)
        np.testing.assert_almost_equal(correct_dz, dz)
        np.testing.assert_almost_equal(correct_dtz, dtz)

    def test_volume(self):
        """Tests transform of Fourier-Zernike basis in a toroidal volume
        """
        grid = ConcentricGrid(M=2, N=2)
        basis = FourierZernikeBasis(M=1, N=1)
        transf = Transform(grid, basis)

        r = grid.nodes[0, :]    # rho coordiantes
        t = grid.nodes[1, :]    # theta coordinates
        z = grid.nodes[2, :]    # zeta coordinates

        correct_vals = 2*r*np.sin(t)*np.cos(z) - 0.5*r*np.cos(t) + np.sin(z)

        idx_0 = np.where(np.all(np.array([np.array(basis.modes[:,0] == 1),
                                          np.array(basis.modes[:,1] == -1),
                                          np.array(basis.modes[:,2] == 1)]), axis=0))
        idx_1 = np.where(np.all(np.array([np.array(basis.modes[:,0] == 1),
                                          np.array(basis.modes[:,1] == 1),
                                          np.array(basis.modes[:,2] == 0)]), axis=0))
        idx_2 = np.where(np.all(np.array([np.array(basis.modes[:,0] == 0),
                                          np.array(basis.modes[:,1] == 0),
                                          np.array(basis.modes[:,2] == -1)]), axis=0))

        c = np.zeros((basis.modes.shape[0],))
        c[idx_0] = 2
        c[idx_1] = -0.5
        c[idx_2] = 1

        values = transf.transform(c, 0, 0, 0)

        np.testing.assert_almost_equal(correct_vals, values)
