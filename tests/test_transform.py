"""Tests for transforming from spectral coefficients to real space values."""

import numpy as np
import pytest

import desc.examples
from desc.backend import jit
from desc.basis import (
    ChebyshevDoubleFourierBasis,
    DoubleFourierSeries,
    FourierSeries,
    FourierZernikeBasis,
    PowerSeries,
    ZernikePolynomial,
    ChebyshevZernikeBasis
)
from desc.compute import get_transforms
from desc.grid import ConcentricGrid, Grid, LinearGrid
from desc.transform import Transform


class TestTransform:

    @pytest.mark.mirror_unit
    def test_volume_chebyshev_zernike(self):
        """Tests transform of Chebyshev-Zernike basis in a toroidal volume."""
        grid = ConcentricGrid(L=2, M=2, N=6)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=2, sym=None)
        transf = Transform(grid, basis, method = "direct1")


        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates

        z_shift = z/np.pi - 1
        correct_vals = (
            r  * np.cos(t) * z_shift#(2*z_shift**2 - 1)
            #- 0.5 * r * np.sin(t) * z_shift
            #+ 1
        ) 
        # 1, -1, 1 is x * np.sin(t) * np.cos(z) 
        # 1, 1, 0 is x * np.sin(t) * 1
        idx_0 = np.where((basis.modes == [1, 1, 1]).all(axis=1))[0]#Zernike 1, -1: r sin()
        #idx_1 = np.where((basis.modes == [1, -1, 1]).all(axis=1))[0]#Zernike 1, 1: r cos()
        #idx_2 = np.where((basis.modes == [0, 0, 0]).all(axis=1))[0]#Zernike 0,0: 1
        c = np.zeros((basis.modes.shape[0],))

        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 1
        #c[idx_1] = -0.5
        #c[idx_2] = 1

        values = transf.transform(c, 0, 0, 0)
        #print(np.array(values))
        #print("Correct values")
        #print(list(correct_vals))
        # print(correct_vals-values)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
