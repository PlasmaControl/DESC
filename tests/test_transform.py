import unittest
import numpy as np
import pytest
from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    FourierZernikeBasis,
)
from desc.transform import Transform


class TestTransform(unittest.TestCase):
    """Tests Transform classes"""

    def test_eq(self):
        """Tests equals operator overload method"""
        grid_1 = LinearGrid(L=11, N=3)
        grid_2 = LinearGrid(M=5, N=5)
        grid_3 = ConcentricGrid(L=4, M=2, N=2)

        basis_1 = DoubleFourierSeries(M=1, N=1)
        basis_2 = FourierZernikeBasis(L=-1, M=1, N=1)

        transf_11 = Transform(grid_1, basis_1)
        transf_21 = Transform(grid_2, basis_1)
        transf_31 = Transform(grid_3, basis_1)
        transf_32 = Transform(grid_3, basis_2)
        transf_32b = Transform(grid_3, basis_2)

        self.assertFalse(transf_11.eq(transf_21))
        self.assertFalse(transf_31.eq(transf_32))
        self.assertTrue(transf_32.eq(transf_32b))

    def test_transform_order_error(self):
        """Tests error handling with transform method"""
        grid = LinearGrid(L=11, endpoint=True)
        basis = PowerSeries(L=2)
        transf = Transform(grid, basis, derivs=0)

        # invalid derivative orders
        with self.assertRaises(ValueError):
            c = np.array([1, 2, 3])
            transf.transform(c, 1, 1, 1)

        # incompatible number of coefficients
        with self.assertRaises(ValueError):
            c = np.array([1, 2])
            transf.transform(c, 0, 0, 0)

    def test_profile(self):
        """Tests transform of power series on a radial profile"""
        grid = LinearGrid(L=11, endpoint=True)
        basis = PowerSeries(L=2)
        transf = Transform(grid, basis, derivs=1)

        x = grid.nodes[:, 0]
        c = np.array([-1, 2, 1])

        values = transf.transform(c, 0, 0, 0)
        derivs = transf.transform(c, 1, 0, 0)

        correct_vals = c[0] + c[1] * x + c[2] * x ** 2
        correct_ders = c[1] + c[2] * 2 * x

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    def test_surface(self):
        """Tests transform of double Fourier series on a flux surface"""
        grid = LinearGrid(M=5, N=5, sym=True)
        basis = DoubleFourierSeries(M=1, N=1)
        transf = Transform(grid, basis, derivs=1)

        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates

        correct_d0 = np.sin(t - z) + 2 * np.cos(t - z)
        correct_dt = np.cos(t - z) - 2 * np.sin(t - z)
        correct_dz = -np.cos(t - z) + 2 * np.sin(t - z)
        correct_dtz = np.sin(t - z) + 2 * np.cos(t - z)

        sin_idx_1 = np.where((basis.modes[:, 1:] == [-1, 1]).all(axis=1))[0]
        sin_idx_2 = np.where((basis.modes[:, 1:] == [1, -1]).all(axis=1))[0]
        cos_idx_1 = np.where((basis.modes[:, 1:] == [-1, -1]).all(axis=1))[0]
        cos_idx_2 = np.where((basis.modes[:, 1:] == [1, 1]).all(axis=1))[0]

        c = np.zeros((basis.modes.shape[0],))
        c[sin_idx_1] = 1
        c[sin_idx_2] = -1
        c[cos_idx_1] = 2
        c[cos_idx_2] = 2

        d0 = transf.transform(c, 0, 0, 0)  # original transform
        dt = transf.transform(c, 0, 1, 0)  # theta derivative
        dz = transf.transform(c, 0, 0, 1)  # zeta derivative
        dtz = transf.transform(c, 0, 1, 1)  # mixed derivative

        np.testing.assert_allclose(d0, correct_d0, atol=1e-8)
        np.testing.assert_allclose(dt, correct_dt, atol=1e-8)
        np.testing.assert_allclose(dz, correct_dz, atol=1e-8)
        np.testing.assert_allclose(dtz, correct_dtz, atol=1e-8)

    def test_volume(self):
        """Tests transform of Fourier-Zernike basis in a toroidal volume"""
        grid = ConcentricGrid(L=4, M=2, N=2)
        basis = FourierZernikeBasis(L=-1, M=1, N=1, sym="sin")
        transf = Transform(grid, basis)

        r = grid.nodes[:, 0]  # rho coordiantes
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates

        correct_vals = (
            2 * r * np.sin(t) * np.cos(z) - 0.5 * r * np.cos(t) * np.sin(z) + np.sin(z)
        )

        idx_0 = np.where((basis.modes == [1, -1, 1]).all(axis=1))[0]
        idx_1 = np.where((basis.modes == [1, 1, -1]).all(axis=1))[0]
        idx_2 = np.where((basis.modes == [0, 0, -1]).all(axis=1))[0]

        c = np.zeros((basis.modes.shape[0],))
        c[idx_0] = 2
        c[idx_1] = -0.5
        c[idx_2] = 1

        values = transf.transform(c, 0, 0, 0)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    def test_set_grid(self):
        """Tests the grid setter method"""
        basis = FourierZernikeBasis(L=-1, M=1, N=1)

        grid_1 = LinearGrid(L=1, M=1, N=1)
        grid_3 = LinearGrid(L=3, M=1, N=1)
        grid_5 = LinearGrid(L=5, M=1, N=1)

        with pytest.warns(UserWarning):
            transf_1 = Transform(grid_1, basis, method="fft")
            transf_3 = Transform(grid_3, basis, method="fft")
            transf_5 = Transform(grid_5, basis, method="fft")

        transf_3.grid = grid_5
        self.assertTrue(transf_3.eq(transf_5))

        transf_3.grid = grid_1
        self.assertTrue(transf_3.eq(transf_1))

    def test_set_basis(self):
        """Tests the basis setter method"""
        grid = ConcentricGrid(L=4, M=2, N=1)

        basis_20 = FourierZernikeBasis(L=-1, M=2, N=0)
        basis_21 = FourierZernikeBasis(L=-1, M=2, N=1)
        basis_31 = FourierZernikeBasis(L=-1, M=3, N=1)

        transf_20 = Transform(grid, basis_20, method="fft")
        transf_21 = Transform(grid, basis_21, method="fft")
        transf_31 = Transform(grid, basis_31, method="fft")

        transf_21.basis = basis_31
        self.assertTrue(transf_21.eq(transf_31))

        transf_21.basis = basis_20
        self.assertTrue(transf_21.eq(transf_20))

    def test_direct_fft_equal(self):
        """tests that the direct and fft method produce the same results"""

        L = 4
        M = 3
        N = 2
        Lnodes = 8
        Mnodes = 4
        Nnodes = 3
        NFP = 4

        grid = ConcentricGrid(Lnodes, Mnodes, Nnodes, NFP)
        basis1 = FourierZernikeBasis(L, M, N, NFP)
        basis2 = FourierSeries(N, NFP)
        basis3 = DoubleFourierSeries(M, N, NFP)

        t1f = Transform(grid, basis1, method="fft")
        t2f = Transform(grid, basis2, method="fft")
        t3f = Transform(grid, basis3, method="fft")

        t1d1 = Transform(grid, basis1, method="direct1")
        t2d1 = Transform(grid, basis2, method="direct1")
        t3d1 = Transform(grid, basis3, method="direct1")

        t1d2 = Transform(grid, basis1, method="direct2")
        t2d2 = Transform(grid, basis2, method="direct2")
        t3d2 = Transform(grid, basis3, method="direct2")

        for d in t1f.derivatives:
            dr = d[0]
            dv = d[1]
            dz = d[2]
            x = np.random.random(basis1.num_modes)
            y1 = t1f.transform(x, dr, dv, dz)
            y2 = t1d1.transform(x, dr, dv, dz)
            y3 = t1d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                y1, y2, atol=1e-12, err_msg="failed on zernike, d={}".format(d)
            )
            np.testing.assert_allclose(
                y3, y2, atol=1e-12, err_msg="failed on zernike, d={}".format(d)
            )
            x = np.random.random(basis2.num_modes)
            y1 = t2f.transform(x, dr, dv, dz)
            y2 = t2d1.transform(x, dr, dv, dz)
            y3 = t2d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                y1, y2, atol=1e-12, err_msg="failed on fourier, d={}".format(d)
            )
            np.testing.assert_allclose(
                y3, y2, atol=1e-12, err_msg="failed on fourier, d={}".format(d)
            )
            x = np.random.random(basis3.num_modes)
            y1 = t3f.transform(x, dr, dv, dz)
            y2 = t3d1.transform(x, dr, dv, dz)
            y3 = t3d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                y1, y2, atol=1e-12, err_msg="failed on double fourier, d={}".format(d)
            )
            np.testing.assert_allclose(
                y3, y2, atol=1e-12, err_msg="failed on double fourier, d={}".format(d)
            )

        M += 1
        N += 1
        Mnodes += 1
        Nnodes += 1

        grid = ConcentricGrid(Lnodes, Mnodes, Nnodes, NFP, sym=True)
        basis1 = FourierZernikeBasis(L, M, N, NFP, sym="cos")
        basis2 = FourierSeries(N, NFP, sym="sin")
        basis3 = DoubleFourierSeries(M, N, NFP, sym="sin")

        t1f.change_resolution(grid, basis1)
        t2f.change_resolution(grid, basis2)
        t3f.change_resolution(grid, basis3)
        t1d1.change_resolution(grid, basis1)
        t2d1.change_resolution(grid, basis2)
        t3d1.change_resolution(grid, basis3)
        t1d2.change_resolution(grid, basis1)
        t2d2.change_resolution(grid, basis2)
        t3d2.change_resolution(grid, basis3)

        for d in t1f.derivatives:
            dr = d[0]
            dv = d[1]
            dz = d[2]
            x = np.random.random(basis1.num_modes)
            y1 = t1f.transform(x, dr, dv, dz)
            y2 = t1d1.transform(x, dr, dv, dz)
            y3 = t1d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                y1,
                y2,
                atol=1e-12,
                err_msg="failed on zernike after change, d={}".format(d),
            )
            np.testing.assert_allclose(
                y3,
                y2,
                atol=1e-12,
                err_msg="failed on zernike after change, d={}".format(d),
            )
            x = np.random.random(basis2.num_modes)
            y1 = t2f.transform(x, dr, dv, dz)
            y2 = t2d1.transform(x, dr, dv, dz)
            y3 = t2d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                y1,
                y2,
                atol=1e-12,
                err_msg="failed on fourier after change, d={}".format(d),
            )
            np.testing.assert_allclose(
                y3,
                y2,
                atol=1e-12,
                err_msg="failed on fourier after change, d={}".format(d),
            )
            x = np.random.random(basis3.num_modes)
            y1 = t3f.transform(x, dr, dv, dz)
            y2 = t3d1.transform(x, dr, dv, dz)
            y3 = t3d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                y1,
                y2,
                atol=1e-12,
                err_msg="failed on double fourier after change, d={}".format(d),
            )
            np.testing.assert_allclose(
                y3,
                y2,
                atol=1e-12,
                err_msg="failed on double fourier after change, d={}".format(d),
            )

    def test_project(self):
        """tests projection method"""

        basis = FourierZernikeBasis(L=-1, M=5, N=3)
        grid = ConcentricGrid(L=4, M=2, N=5)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        transform.build()
        dtransform1.build()
        dtransform2.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))

        basis = FourierZernikeBasis(L=-1, M=5, N=3, sym="cos")
        grid = ConcentricGrid(L=4, M=2, N=5)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        transform.build()
        dtransform1.build()
        dtransform2.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))

        basis = FourierZernikeBasis(L=-1, M=5, N=0, sym="sin")
        grid = ConcentricGrid(L=4, M=2, N=5, sym=True)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        transform.build()
        dtransform1.build()
        dtransform2.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))
