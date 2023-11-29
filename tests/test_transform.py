"""Tests for transforming from spectral coefficients to real space values."""

import numpy as np
import pytest

from desc.basis import ChebyshevZernikeBasis
from desc.grid import ConcentricGrid
from desc.transform import Transform


class TestTransform:

    @pytest.mark.mirror_unit
    def test_volume_chebyshev_zernike(self):
        """Tests transform of Chebyshev-Zernike basis in a toroidal volume."""
        grid = ConcentricGrid(L=2, M=2, N=2)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=2, sym=None)
        transf = Transform(grid, basis, method = "direct1")


        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates

        z_shift = z/np.pi - 1
        correct_vals = (
            2*r  * np.cos(t) * z_shift
            - 0.5*r * np.sin(t) *(2*z_shift**2 - 1)
            + 1
        ) 

        idx_0 = np.where((basis.modes == [1, 1, 1]).all(axis=1))[0]#Zernike 1, -1: r sin()
        idx_1 = np.where((basis.modes == [1, -1, 2]).all(axis=1))[0]#Zernike 1, 1: r cos()
        idx_2 = np.where((basis.modes == [0, 0, 0]).all(axis=1))[0]#Zernike 0,0: 1
        c = np.zeros((basis.modes.shape[0],))

        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        c[idx_1] = -0.5
        c[idx_2] = 1

        values = transf.transform(c, 0, 0, 0)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
