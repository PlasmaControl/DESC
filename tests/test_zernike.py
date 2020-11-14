import unittest
import numpy as np
from desc.zernike import get_zern_basis_idx_dense, ZernikeTransform
from desc.zernike import zern, zern_radial, zern_azimuthal, four_toroidal
from desc.backend import get_needed_derivatives
from desc.nodes import get_nodes_pattern


class TestZernikeTransform(unittest.TestCase):
    """tests for zernike transform class"""

    def test_zern_radial(self):

        l = np.array([3, 4, 6])
        m = np.array([1, 2, 2])
        def Z3_1(x): return 3*x**3 - 2*x
        def Z4_2(x): return 4*x**4 - 3*x**2
        def Z6_2(x): return 15*x**6 - 20*x**4 + 6*x**2

        def dZ3_1(x): return 9*x**2 - 2
        def dZ4_2(x): return 16*x**3 - 6*x
        def dZ6_2(x): return 90*x**5 - 80*x**3 + 12*x

        r = np.linspace(0, 1, 20)

        correct = np.array([Z3_1(r), Z4_2(r), Z6_2(r)]).T
        ours = zern_radial(r, l, m, 0)
        np.testing.assert_almost_equal(correct, ours)

        correct_d = np.array([dZ3_1(r), dZ4_2(r), dZ6_2(r)]).T
        ours_d = zern_radial(r, l, m, 1)
        np.testing.assert_almost_equal(correct_d, ours_d)

    def test_zern_azimuthal(self):
                     # s c c
        m = np.array([-1, 0, 1])
        theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

        correct = np.array([[0, 1, 1],
                            [1, 1, 0],
                            [0, 1, -1],
                            [-1, 1, 0]])
        ours = zern_azimuthal(theta, m, 0)
        np.testing.assert_almost_equal(correct, ours)

        correct_d = np.array([[1, 0, 0],
                              [0, 0, -1],
                              [-1, 0, 0],
                              [0, 0, 1]])
        ours_d = zern_azimuthal(theta, m, 1)
        np.testing.assert_almost_equal(correct_d, ours_d)

    def test_four_toroidal(self):
                     # s c c
        n = np.array([-1, 0, 1])
        zeta = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
        NFP = 2

        correct = np.array([[0, 1, 1],
                            [1, 1, 0],
                            [0, 1, -1],
                            [-1, 1, 0]])
        ours = four_toroidal(zeta, n, NFP, 0)
        np.testing.assert_almost_equal(correct, ours)

        correct_d = np.array([[2, 0, 0],
                              [0, 0, -2],
                              [-2, 0, 0],
                              [0, 0, 2]])
        ours_d = four_toroidal(zeta, n, NFP, 1)
        np.testing.assert_almost_equal(correct_d, ours_d)

    def test_direct_fft_equal(self):

        M = 3
        N = 2
        Mnodes = 4
        Nnodes = 3
        NFP = 4

        zern_idx = get_zern_basis_idx_dense(M, N)
        nodes, volumes = get_nodes_pattern(Mnodes, Nnodes, NFP, surfs='cheb1')
        derivatives = get_needed_derivatives('all')

        direct = ZernikeTransform(
            nodes, zern_idx, NFP, derivatives, method='direct')
        fft = ZernikeTransform(nodes, zern_idx, NFP, derivatives, method='fft')

        ys = []

        for i, d in enumerate(derivatives):
            dr = d[0]
            dv = d[1]
            dz = d[2]
            x = np.random.random(len(zern_idx))
            y1 = direct.transform(x, dr, dv, dz)
            y2 = fft.transform(x, dr, dv, dz)
            ys.append(np.allclose(y1, y2,))
        assert np.all(ys)

        M += 1
        N += 1
        Mnodes += 1
        Nnodes += 1

        zern_idx = get_zern_basis_idx_dense(M, N)
        nodes, volumes = get_nodes_pattern(Mnodes, Nnodes, NFP, surfs='cheb1')

        fft.expand_nodes(nodes, volumes)
        direct.expand_nodes(nodes, volumes)
        fft.expand_spectral_resolution(zern_idx)
        direct.expand_spectral_resolution(zern_idx)

        ys = []

        for i, d in enumerate(derivatives):
            dr = d[0]
            dv = d[1]
            dz = d[2]
            x = np.random.random(len(zern_idx))
            y1 = direct.transform(x, dr, dv, dz)
            y2 = fft.transform(x, dr, dv, dz)
            ys.append(np.allclose(y1, y2,))
        assert np.all(ys)
