import unittest
import numpy as np

from desc.basis import jacobi, fourier


class TestBasis(unittest.TestCase):
    """Tests Basis classes"""

    def test_jacobi(self):
        """Tests jacobi function
        """
        l = np.array([3, 4, 6])
        m = np.array([1, 2, 2])
        r = np.linspace(0, 1, 20)  # rho coordinates

        # correct value functions
        def Z3_1(x): return 3*x**3 - 2*x
        def Z4_2(x): return 4*x**4 - 3*x**2
        def Z6_2(x): return 15*x**6 - 20*x**4 + 6*x**2

        # correct derivative functions
        def dZ3_1(x): return 9*x**2 - 2
        def dZ4_2(x): return 16*x**3 - 6*x
        def dZ6_2(x): return 90*x**5 - 80*x**3 + 12*x

        correct_vals = np.array([Z3_1(r), Z4_2(r), Z6_2(r)]).T
        correct_ders = np.array([dZ3_1(r), dZ4_2(r), dZ6_2(r)]).T

        values = jacobi(r, l, m, 0)
        derivs = jacobi(r, l, m, 1)

        np.testing.assert_almost_equal(correct_vals, values)
        np.testing.assert_almost_equal(correct_ders, derivs)

    def test_fourier(self):
        """Tests fourier function
        """
        m = np.array([-1, 0, 1])
        t = np.array([0, np.pi/2, np.pi, 3*np.pi/2])  # theta coordinates

        correct_vals = np.array([[0, 1, 1],
                                 [1, 1, 0],
                                 [0, 1, -1],
                                 [-1, 1, 0]])
        correct_ders = np.array([[1, 0, 0],
                                 [0, 0, -1],
                                 [-1, 0, 0],
                                 [0, 0, 1]])

        values = fourier(t, m, dt=0)
        derivs = fourier(t, m, dt=1)

        np.testing.assert_almost_equal(correct_vals, values)
        np.testing.assert_almost_equal(correct_ders, derivs)
