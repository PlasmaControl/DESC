import unittest
import numpy as np
from desc.zernike import get_zern_basis_idx_dense, ZernikeTransform
from desc.backend import get_needed_derivatives
from desc.nodes import get_nodes_pattern


class TestZernikeTransform(unittest.TestCase):
    """tests for zernike transform class"""

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
