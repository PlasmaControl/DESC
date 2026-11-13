"""Tests for transforming from spectral coefficients to real space values."""

import numpy as np
import pytest

import desc.examples
from desc.backend import jit, jnp
from desc.basis import (
    ChebyshevDoubleFourierBasis,
    DoubleFourierSeries,
    FourierSeries,
    FourierZernikeBasis,
    PowerSeries,
    ZernikePolynomial,
)
from desc.compute import get_transforms
from desc.grid import ConcentricGrid, Grid, LinearGrid
from desc.transform import Transform


class TestTransform:
    """Tests Transform classes."""

    @pytest.mark.unit
    def test_eq(self):
        """Tests equals operator overload method."""
        grid_1 = LinearGrid(L=10, N=1)
        grid_2 = LinearGrid(M=2, N=2)
        grid_3 = ConcentricGrid(L=4, M=2, N=2)

        basis_1 = DoubleFourierSeries(M=1, N=1)
        basis_2 = FourierZernikeBasis(L=1, M=1, N=1)

        transf_11 = Transform(grid_1, basis_1)
        transf_21 = Transform(grid_2, basis_1)
        transf_31 = Transform(grid_3, basis_1)
        transf_32 = Transform(grid_3, basis_2)
        transf_32b = Transform(grid_3, basis_2)

        assert not transf_11.equiv(transf_21)
        assert not transf_31.equiv(transf_32)
        assert transf_32.equiv(transf_32b)

    @pytest.mark.unit
    def test_transform_order_error(self):
        """Tests error handling with transform method."""
        grid = LinearGrid(L=10)
        basis = PowerSeries(L=2, sym=False)
        for method in ("auto", "pest"):
            transf = Transform(grid, basis, derivs=0, method=method)

            # invalid derivative orders
            with pytest.raises(ValueError):
                c = np.array([1, 2, 3])
                transf.transform(c, 1, 1, 1)

            # incompatible number of coefficients
            with pytest.raises(ValueError):
                c = np.array([1, 2])
                transf.transform(c, 0, 0, 0)

    @pytest.mark.unit
    def test_profile(self):
        """Tests transform of power series on a radial profile."""
        grid = LinearGrid(L=10)
        basis = PowerSeries(L=2, sym=False)
        transf = Transform(grid, basis, derivs=1)

        x = grid.nodes[:, 0]
        c = np.array([-1, 2, 1])

        values = transf.transform(c, 0, 0, 0)
        derivs = transf.transform(c, 1, 0, 0)

        correct_vals = c[0] + c[1] * x + c[2] * x**2
        correct_ders = c[1] + c[2] * 2 * x

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    @pytest.mark.unit
    def test_surface(self):
        """Tests transform of double Fourier series on a flux surface."""
        grid = LinearGrid(M=2, N=2, sym=True)
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

    @pytest.mark.unit
    def test_volume_chebyshev(self):
        """Tests transform of Chebyshev-Fourier basis in a toroidal volume."""
        grid = ConcentricGrid(L=4, M=2, N=2)
        basis = ChebyshevDoubleFourierBasis(L=1, M=1, N=1, sym="sin")
        transf = Transform(grid, basis)

        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates

        x = 2 * r - 1
        correct_vals = (
            2 * x * np.sin(t) * np.cos(z) - 0.5 * x * np.cos(t) * np.sin(z) + np.sin(z)
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

    @pytest.mark.unit
    def test_volume_zernike(self):
        """Tests transform of Fourier-Zernike basis in a toroidal volume."""
        grid = ConcentricGrid(L=4, M=2, N=2)
        basis = FourierZernikeBasis(L=1, M=1, N=1, sym="sin")
        transf = Transform(grid, basis)

        r = grid.nodes[:, 0]  # rho coordinates
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

    @pytest.mark.unit
    def test_set_grid(self):
        """Tests the grid setter method."""
        basis = FourierZernikeBasis(L=1, M=1, N=1)

        grid_1 = LinearGrid(L=0)
        grid_3 = LinearGrid(L=2)
        grid_5 = LinearGrid(L=4)

        with pytest.warns(UserWarning):
            transf_1 = Transform(grid_1, basis, method="fft")
            transf_3 = Transform(grid_3, basis, method="fft")
            transf_5 = Transform(grid_5, basis, method="fft")

        transf_3.grid = grid_5
        assert transf_3.equiv(transf_5)

        transf_3.grid = grid_1
        assert transf_3.equiv(transf_1)

        np.testing.assert_allclose(transf_3.nodes, grid_1.nodes)

        pest_3 = Transform(grid_3, basis, method="pest")
        pest_5 = Transform(grid_5, basis, method="pest")
        pest_3.grid = grid_5
        assert pest_3.method == "pest"
        assert pest_3.equiv(pest_5)

    @pytest.mark.unit
    def test_set_basis(self):
        """Tests the basis setter method."""
        grid = ConcentricGrid(L=4, M=2, N=1)

        basis_20 = FourierZernikeBasis(L=1, M=2, N=0)
        basis_21 = FourierZernikeBasis(L=1, M=2, N=1)
        basis_31 = FourierZernikeBasis(L=1, M=3, N=1)

        transf_20 = Transform(grid, basis_20, method="fft")
        transf_21 = Transform(grid, basis_21, method="fft")
        transf_31 = Transform(grid, basis_31, method="fft")

        transf_21.basis = basis_31
        assert transf_21.equiv(transf_31)

        transf_21.basis = basis_20
        assert transf_21.equiv(transf_20)

        np.testing.assert_allclose(transf_21.modes, basis_20.modes)

        pest_21 = Transform(grid, basis_21, method="pest")
        pest_31 = Transform(grid, basis_31, method="pest")
        pest_21.basis = basis_31
        assert pest_21.method == "pest"
        assert pest_21.equiv(pest_31)

    @pytest.mark.unit
    def test_fft(self):
        """Tests Fast Fourier Transform method."""
        grid = LinearGrid(N=16)
        zeta = grid.nodes[:, 2]

        sin_coeffs = np.array([0.5, -1, 2])
        cos_coeffs = np.array([3, -1, 1.5, -0.5])
        for_coeffs = np.hstack((sin_coeffs, cos_coeffs))

        sin_basis = FourierSeries(N=3, sym="sin")
        cos_basis = FourierSeries(N=3, sym="cos")
        for_basis = FourierSeries(N=3)

        sin_tform = Transform(grid, sin_basis, derivs=1, method="fft")
        cos_tform = Transform(grid, cos_basis, derivs=1, method="fft")
        for_tform = Transform(grid, for_basis, derivs=1, method="fft")

        correct_s0 = 0.5 * np.sin(3 * zeta) - np.sin(2 * zeta) + 2 * np.sin(zeta)
        correct_s1 = 1.5 * np.cos(3 * zeta) - 2 * np.cos(2 * zeta) + 2 * np.cos(zeta)
        correct_c0 = 3 - np.cos(zeta) + 1.5 * np.cos(2 * zeta) - 0.5 * np.cos(3 * zeta)
        correct_c1 = np.sin(zeta) - 3 * np.sin(2 * zeta) + 1.5 * np.sin(3 * zeta)
        correct_f0 = correct_s0 + correct_c0
        correct_f1 = correct_s1 + correct_c1

        s0 = sin_tform.transform(sin_coeffs, 0, 0, 0)
        s1 = sin_tform.transform(sin_coeffs, 0, 0, 1)
        c0 = cos_tform.transform(cos_coeffs, 0, 0, 0)
        c1 = cos_tform.transform(cos_coeffs, 0, 0, 1)
        f0 = for_tform.transform(for_coeffs, 0, 0, 0)
        f1 = for_tform.transform(for_coeffs, 0, 0, 1)

        np.testing.assert_allclose(s0, correct_s0, atol=1e-8)
        np.testing.assert_allclose(s1, correct_s1, atol=1e-8)
        np.testing.assert_allclose(c0, correct_c0, atol=1e-8)
        np.testing.assert_allclose(c1, correct_c1, atol=1e-8)
        np.testing.assert_allclose(f0, correct_f0, atol=1e-8)
        np.testing.assert_allclose(f1, correct_f1, atol=1e-8)

        for basis, coeffs, correct in (
            (sin_basis, sin_coeffs, (correct_s0, correct_s1)),
            (cos_basis, cos_coeffs, (correct_c0, correct_c1)),
            (for_basis, for_coeffs, (correct_f0, correct_f1)),
        ):
            tform = Transform(grid, basis, derivs=1, method="pest")
            assert tform.method == "pest"
            for dz in (0, 1):
                np.testing.assert_allclose(
                    tform.transform(coeffs, 0, 0, dz), correct[dz], atol=1e-8
                )

    @pytest.mark.slow
    @pytest.mark.unit
    def test_direct_fft_equal(self):
        """Tests that the direct and fft method produce the same results."""
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

        t1p = Transform(grid, basis1, method="pest")
        t2p = Transform(grid, basis2, method="pest")
        t3p = Transform(grid, basis3, method="pest")

        for d in t1f.derivatives:
            dr = d[0]
            dv = d[1]
            dz = d[2]
            x = np.random.random(basis1.num_modes)
            y1 = t1f.transform(x, dr, dv, dz)
            y2 = t1d1.transform(x, dr, dv, dz)
            y3 = t1d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                t1p.transform(x, dr, dv, dz),
                y2,
                atol=1e-12,
                err_msg=f"pest, basis1, d={d}",
            )
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
                t2p.transform(x, dr, dv, dz),
                y2,
                atol=1e-12,
                err_msg=f"pest, basis2, d={d}",
            )
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
                t3p.transform(x, dr, dv, dz),
                y2,
                atol=1e-12,
                err_msg=f"pest, basis3, d={d}",
            )
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

        # should pass the methods, otherwise default might change
        t1f.change_resolution(grid, basis1, method="fft")
        t2f.change_resolution(grid, basis2, method="fft")
        t3f.change_resolution(grid, basis3, method="fft")
        t1d1.change_resolution(grid, basis1, method="direct1")
        t2d1.change_resolution(grid, basis2, method="direct1")
        t3d1.change_resolution(grid, basis3, method="direct1")
        t1d2.change_resolution(grid, basis1, method="direct2")
        t2d2.change_resolution(grid, basis2, method="direct2")
        t3d2.change_resolution(grid, basis3, method="direct2")
        t1p.change_resolution(grid, basis1, method="pest")
        t2p.change_resolution(grid, basis2, method="pest")
        t3p.change_resolution(grid, basis3, method="pest")

        for d in t1f.derivatives:
            dr = d[0]
            dv = d[1]
            dz = d[2]
            x = np.random.random(basis1.num_modes)
            y1 = t1f.transform(x, dr, dv, dz)
            y2 = t1d1.transform(x, dr, dv, dz)
            y3 = t1d2.transform(x, dr, dv, dz)
            np.testing.assert_allclose(
                t1p.transform(x, dr, dv, dz),
                y2,
                atol=1e-12,
                err_msg=f"pest, basis1, d={d}",
            )
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
                t2p.transform(x, dr, dv, dz),
                y2,
                atol=1e-12,
                err_msg=f"pest, basis2, d={d}",
            )
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
                t3p.transform(x, dr, dv, dz),
                y2,
                atol=1e-12,
                err_msg=f"pest, basis3, d={d}",
            )
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

    @pytest.mark.unit
    def test_project(self):
        """Tests projection method for Galerkin method."""
        basis = FourierZernikeBasis(L=1, M=5, N=3)
        grid = ConcentricGrid(L=4, M=2, N=5)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        ptransform = Transform(grid, basis, method="pest")
        transform.build()
        dtransform1.build()
        dtransform2.build()
        ptransform.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))
        np.testing.assert_allclose(ptransform.project(y), dtransform1.project(y))

        basis = FourierZernikeBasis(L=1, M=5, N=3, sym="cos")
        grid = ConcentricGrid(L=4, M=2, N=5)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        ptransform = Transform(grid, basis, method="pest")
        transform.build()
        dtransform1.build()
        dtransform2.build()
        ptransform.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))
        np.testing.assert_allclose(ptransform.project(y), dtransform1.project(y))

        basis = FourierZernikeBasis(L=1, M=5, N=0, sym="sin")
        grid = ConcentricGrid(L=4, M=2, N=5, sym=True)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        ptransform = Transform(grid, basis, method="pest")
        transform.build()
        dtransform1.build()
        dtransform2.build()
        ptransform.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))
        np.testing.assert_allclose(ptransform.project(y), dtransform1.project(y))

    @pytest.mark.unit
    def test_fft_warnings(self):
        """Test that warnings are thrown when trying to use fft where it won't work."""
        g = Grid(np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]]))
        b = ZernikePolynomial(L=2, M=2)
        with pytest.warns(UserWarning, match="compatible grid"):
            t = Transform(g, b, method="fft")
        assert t.method == "direct1"

        g = LinearGrid(rho=2, M=2, N=2, NFP=2)
        b = DoubleFourierSeries(M=2, N=2)
        # this actually will emit 2 warnings, one for the NFP for
        # basis and grid not matching, and one for nodes completing 1 full period
        # we will catch the UserWarning generically then check each message
        with pytest.warns(
            UserWarning
        ) as record:  # , match="nodes complete 1 full field period"):
            t = Transform(g, b, method="fft")
        assert t.method == "direct2"
        NFP_grid_basis_warning_exists = False
        nodes_warning_exists = False
        for r in record:
            if "Unequal number of field periods" in str(r.message):
                NFP_grid_basis_warning_exists = True
            if "grid and basis to have the same NFP" in str(r.message):
                nodes_warning_exists = True
        assert NFP_grid_basis_warning_exists and nodes_warning_exists

        g = LinearGrid(rho=2, M=2, N=2)
        b = DoubleFourierSeries(M=1, N=3)
        with pytest.warns(UserWarning, match="can not undersample in zeta"):
            t = Transform(g, b, method="fft")
        assert t.method == "direct2"

        b._fft_toroidal = False
        g = LinearGrid(2, 3, 4)
        with pytest.warns(UserWarning, match="compatible basis"):
            t = Transform(g, b, method="fft")
        assert t.method == "direct1"

    @pytest.mark.unit
    def test_direct2_warnings(self):
        """Test that warnings are thrown when trying to use direct2 if it won't work."""
        g = Grid(np.array([[0, 0, -1], [1, 1, 0], [1, 1, 1]]))
        b = ZernikePolynomial(L=2, M=2)
        with pytest.warns(UserWarning, match="requires compatible grid"):
            t = Transform(g, b, method="direct2")
        assert t.method == "direct1"

        b._fft_toroidal = False
        g = LinearGrid(2, 3, 4)
        with pytest.warns(UserWarning, match="compatible basis"):
            t = Transform(g, b, method="direct2")
        assert t.method == "direct1"

    @pytest.mark.unit
    def test_pest_warnings(self):
        """The pest method falls back to direct1 for an incompatible basis."""
        g = Grid(np.array([[0, 0, -1], [1, 1, 0], [1, 1, 1]]))
        b = ZernikePolynomial(L=2, M=2)
        t = Transform(g, b, method="pest")
        assert t.method == "pest"

        b._fft_toroidal = False
        g = LinearGrid(2, 3, 4)
        with pytest.warns(UserWarning, match="compatible basis"):
            t = Transform(g, b, method="pest")
        assert t.method == "direct1"

    @pytest.mark.unit
    def test_fit_direct1_and_jitable(self):
        """Test fitting with direct1 and jitable method."""
        basis = FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi")
        grid = ConcentricGrid(3, 3, 2, node_pattern="ocs")
        transform = Transform(grid, basis, method="direct1", build_pinv=True)
        np.random.seed(0)
        c = (0.5 - np.random.random(basis.num_modes)) * abs(basis.modes).sum(axis=-1)
        x = transform.transform(c)
        c1 = transform.fit(x)
        np.testing.assert_allclose(c, c1, atol=1e-12)
        # also test jitable which is the same as direct1
        transform = Transform(grid, basis, method="jitable", build_pinv=True)
        x = transform.transform(c)
        c1 = transform.fit(x)
        np.testing.assert_allclose(c, c1, atol=1e-12)

    @pytest.mark.unit
    def test_fit_direct2(self):
        """Test fitting with direct2 method."""
        basis = FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi")
        grid = ConcentricGrid(4, 4, 3, node_pattern="jacobi")
        transform = Transform(grid, basis, method="direct2", build_pinv=True)
        np.random.seed(1)
        c = (0.5 - np.random.random(basis.num_modes)) * abs(basis.modes).sum(axis=-1)
        x = transform.transform(c)
        c1 = transform.fit(x)
        np.testing.assert_allclose(c, c1, atol=1e-12)

    @pytest.mark.unit
    def test_fit_pest(self):
        """Test fitting with pest method (7 zeta planes > 5 toroidal modes)."""
        basis = FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi")
        grid = ConcentricGrid(4, 4, 3, node_pattern="jacobi")
        transform = Transform(grid, basis, method="pest", build_pinv=True)
        assert transform.method == "pest"
        np.random.seed(1)
        c = (0.5 - np.random.random(basis.num_modes)) * abs(basis.modes).sum(axis=-1)
        x = transform.transform(c)
        c1 = transform.fit(x)
        np.testing.assert_allclose(c, c1, atol=1e-12)

    @pytest.mark.unit
    def test_fit_fft(self):
        """Test fitting with fft method."""
        basis = FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi")
        grid = LinearGrid(4, 4, 3)
        transform = Transform(grid, basis, method="fft", build_pinv=True)
        np.random.seed(2)
        c = (0.5 - np.random.random(basis.num_modes)) * abs(basis.modes).sum(axis=-1)
        x = transform.transform(c)
        c1 = transform.fit(x)
        np.testing.assert_allclose(c, c1, atol=1e-12)

    @pytest.mark.unit
    def test_empty_grid(self):
        """Make sure we can build transforms with empty grids."""
        grid = Grid(nodes=np.empty((0, 3)))
        basis = FourierZernikeBasis(6, 0, 0)
        _ = Transform(grid, basis)

        basis = FourierZernikeBasis(6, 6, 6)
        _ = Transform(grid, basis)
        _ = Transform(grid, basis, method="pest")

    @pytest.mark.unit
    def test_Z_projection(self):
        """Make sure we always have the 0,0,0 derivative for projections."""
        eq = desc.examples.get("DSHAPE")
        data_keys = ["F_rho", "|grad(rho)|", "sqrt(g)", "F_helical", "|e^helical|"]
        grid = ConcentricGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            axis=False,
        )
        tr = get_transforms(data_keys, eq, grid)
        f = np.ones(grid.num_nodes)

        assert tr["Z"].matrices["direct1"][0][0][0].shape == (
            grid.num_nodes,
            eq.Z_basis.num_modes,
        )
        _ = tr["Z"].project(f)

    @pytest.mark.unit
    def test_fft_even_grid(self):
        """Test fft method with even number of grid points."""
        for sym in ["cos", "sin", False]:
            basis = FourierZernikeBasis(2, 2, 4, sym=sym)
            c = np.random.random(basis.num_modes)
            for N in range(9, 16):
                grid = LinearGrid(L=2, M=2, zeta=N)
                t1 = Transform(grid, basis, method="direct1", build_pinv=True)
                t2 = Transform(grid, basis, method="fft", build_pinv=True)
                x1 = t1.transform(c)
                x2 = t2.transform(c)
                np.testing.assert_allclose(
                    x1, x2, atol=1e-10, err_msg=f"N={N} sym={sym}"
                )
                c1 = t1.fit(x1)
                c2 = t2.fit(x2)
                np.testing.assert_allclose(
                    c1, c2, atol=1e-10, err_msg=f"N={N} sym={sym}"
                )
                y1 = t1.project(x1)
                y2 = t2.project(x2)
                np.testing.assert_allclose(
                    y1, y2, atol=1e-10, err_msg=f"N={N} sym={sym}"
                )
                t3 = Transform(grid, basis, method="pest", build_pinv=True)
                x3 = t3.transform(c)
                np.testing.assert_allclose(
                    x1, x3, atol=1e-10, err_msg=f"pest N={N} sym={sym}"
                )
                np.testing.assert_allclose(
                    c1, t3.fit(x3), atol=1e-10, err_msg=f"pest N={N} sym={sym}"
                )
                np.testing.assert_allclose(
                    y1, t3.project(x3), atol=1e-10, err_msg=f"pest N={N} sym={sym}"
                )


@pytest.mark.unit
def test_transform_pytree():
    """Ensure that Transforms are valid pytree/JAX types."""
    grid = LinearGrid(5, 6, 7)
    basis = FourierZernikeBasis(4, 5, 6)
    transform = Transform(grid, basis, build=True)

    import jax

    leaves, treedef = jax.tree_util.tree_flatten(transform)
    transform = jax.tree_util.tree_unflatten(treedef, leaves)

    @jit
    def foo(x, tr):
        # this one we pass in transform as a pytree
        return tr.transform(x)

    @jit
    def bar(x):
        # this one we close over it
        return transform.transform(x)

    x = np.random.random(basis.num_modes)
    np.testing.assert_allclose(foo(x, transform), transform.transform(x))
    np.testing.assert_allclose(bar(x), transform.transform(x))

    pest = Transform(grid, basis, build=True, method="pest")
    leaves, treedef = jax.tree_util.tree_flatten(pest)
    pest = jax.tree_util.tree_unflatten(treedef, leaves)
    assert pest.method == "pest"
    np.testing.assert_allclose(foo(x, pest), transform.transform(x), atol=1e-12)


@pytest.mark.unit
def test_NFP_warning():
    """Make sure we only warn about basis/grid NFP in cases where it matters."""
    rho = np.linspace(0, 1, 20)
    g01 = LinearGrid(rho=rho, M=5, N=0, NFP=1)
    g02 = LinearGrid(rho=rho, M=5, N=0, NFP=2)
    g21 = LinearGrid(rho=rho, M=5, N=5, NFP=1)
    g22 = LinearGrid(rho=rho, M=5, N=5, NFP=2)
    b01 = FourierZernikeBasis(L=2, M=2, N=0, NFP=1)
    b02 = FourierZernikeBasis(L=2, M=2, N=0, NFP=2)
    b21 = FourierZernikeBasis(L=2, M=2, N=2, NFP=1)
    b22 = FourierZernikeBasis(L=2, M=2, N=2, NFP=2)

    # No toroidal nodes, shouldn't warn
    _ = Transform(g01, b01)
    _ = Transform(g01, b02)
    _ = Transform(g01, b21)
    _ = Transform(g01, b22)

    # No toroidal nodes, shouldn't warn
    _ = Transform(g02, b01)
    _ = Transform(g02, b02)
    _ = Transform(g02, b21)
    _ = Transform(g02, b22)

    # toroidal nodes but no toroidal modes, no warning
    _ = Transform(g21, b01)
    # toroidal nodes but no toroidal modes, no warning
    _ = Transform(g21, b02)
    # toroidal nodes and modes, but equal nfp, no warning
    _ = Transform(g21, b21)
    # toroidal modes and nodes and unequal NFP -> warning
    with pytest.warns(UserWarning):
        _ = Transform(g21, b22)

    # no toroidal modes, no warning
    _ = Transform(g22, b01)
    # no toroidal modes, no warning
    _ = Transform(g22, b02)
    # toroidal modes and nodes and unequal NFP -> warning
    with pytest.warns(UserWarning):
        _ = Transform(g22, b21)
    # toroidal nodes and modes, but equal nfp, no warning
    _ = Transform(g22, b22)


class TestPestTransform:
    """``method="pest"``: zeta is a tensor axis of the grid, (rho, theta) is not."""

    NFP = 3
    RHO = (0.2, 0.45, 0.7, 0.95)

    # f(rho, theta, zeta) (analytic function used for testing)
    TERMS = {
        (0, 0, 0): (1.5, lambda r: (1 + 0 * r, 0 * r, 0 * r)),
        (1, 1, 0): (2.0, lambda r: (r, 1 + 0 * r, 0 * r)),
        (2, -2, 1): (-0.7, lambda r: (r**2, 2 * r, 2 + 0 * r)),
        (2, 0, -1): (0.4, lambda r: (2 * r**2 - 1, 4 * r, 4 + 0 * r)),
        (3, 1, 2): (0.3, lambda r: (3 * r**3 - 2 * r, 9 * r**2 - 2, 18 * r)),
        (3, -3, -1): (-0.25, lambda r: (r**3, 3 * r**2, 6 * r)),
    }

    @staticmethod
    def _trig(k, x, d):
        """d-th derivative of cos(k x) (k >= 0) or sin(|k| x) (k < 0)."""
        shift = d * np.pi / 2
        a = abs(k)
        return a**d * (np.cos(a * x + shift) if k >= 0 else np.sin(a * x + shift))

    def _f(self, nodes, d=(0, 0, 0), terms=None):
        """The test function, or its (dr, dt, dz) derivative, at ``nodes``."""
        r, t, z = nodes.T
        out = np.zeros(r.size)
        for (_, m, n), (coef, R) in (terms or self.TERMS).items():
            out += (
                coef
                * R(r)[d[0]]
                * self._trig(m, t, d[1])
                * self.NFP ** d[2]
                * self._trig(n, self.NFP * z, d[2])
            )
        return out

    def _coeffs(self, basis, terms=None):
        """The coefficients of ``f`` in ``basis``: the TERMS, zero elsewhere."""
        c = np.zeros(basis.num_modes)
        for lmn, (coef, _) in (terms or self.TERMS).items():
            c[basis.get_idx(*lmn)] = coef
        return c

    def _grid(self, n_zeta, n_theta=8, seed=0, ragged=False):
        """Uniform zeta planes; each (rho, zeta) pair has its own sorted random thetas.

        ``ragged``: different zeta planes have different numbers of nodes
        """
        rng = np.random.default_rng(seed)
        zeta = 2 * np.pi * np.arange(n_zeta) / (n_zeta * self.NFP)
        rho = np.asarray(self.RHO)
        nt = n_theta + (np.arange(n_zeta) if ragged else np.zeros(n_zeta, dtype=int))
        # (zeta, rho, theta) slots; a plane's unused slots (ragged) sort to the end as
        # inf and are dropped by `keep`.
        keep = np.broadcast_to(
            np.arange(nt.max()) < nt[:, None, None], (n_zeta, rho.size, nt.max())
        )
        theta = np.sort(
            np.where(keep, rng.uniform(0, 2 * np.pi, keep.shape), np.inf), axis=-1
        )
        z, r, t = np.broadcast_arrays(zeta[:, None, None], rho[None, :, None], theta)
        nodes = np.stack([r[keep], t[keep], z[keep]], axis=-1)  # zeta, rho, theta order
        return Grid(nodes, NFP=self.NFP, sort=False)

    def _basis(self):
        return FourierZernikeBasis(L=3, M=3, N=2, NFP=self.NFP, sym=False)

    @pytest.mark.unit
    def test_pest_fit_recovers_coefficients(self):
        """fit(f) gives f's known coefficients; 5 zeta planes, 5 toroidal modes."""
        basis = self._basis()
        grid = self._grid(5)
        assert grid.num_zeta == 5 and basis.N == 2
        transform = Transform(grid, basis, method="pest", build_pinv=True)
        assert transform.method == "pest"
        c = transform.fit(self._f(grid.nodes))
        np.testing.assert_allclose(c, self._coeffs(basis), atol=1e-12)

    @pytest.mark.unit
    def test_pest_fit_square(self):
        """num_nodes == num_modes: the fit recreates f's coefficients."""
        basis = self._basis()
        nodes = np.array(
            ConcentricGrid(
                L=basis.L, M=basis.M, N=basis.N, NFP=self.NFP, node_pattern="jacobi"
            ).nodes
        )
        # one random rotation per (rho, zeta) ring
        _, ring = np.unique(nodes[:, [0, 2]], axis=0, return_inverse=True)
        shift = np.random.default_rng(6).uniform(0, 2 * np.pi, ring.max() + 1)
        nodes[:, 1] = (nodes[:, 1] + shift[ring.ravel()]) % (2 * np.pi)
        grid = Grid(nodes, NFP=self.NFP, sort=False)
        assert grid.num_nodes == basis.num_modes == 50
        transform = Transform(grid, basis, method="pest", build_pinv=True)
        assert transform.method == "pest"
        c = transform.fit(self._f(grid.nodes))
        np.testing.assert_allclose(c, self._coeffs(basis), atol=1e-12)

    @pytest.mark.unit
    def test_pest_derivatives_match_analytic(self):
        """transform(c, dr, dt, dz) equals the analytic derivative of f."""
        basis = self._basis()
        grid = self._grid(5)
        derivs = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        )
        transform = Transform(grid, basis, derivs=derivs, method="pest", build=True)
        c = self._coeffs(basis)
        for d in derivs:
            np.testing.assert_allclose(
                transform.transform(c, *d),
                self._f(grid.nodes, d),
                atol=1e-11,
                err_msg=f"derivative {d}",
            )

    @pytest.mark.unit
    def test_pest_fit_oversampled_zeta(self):
        """More zeta planes than toroidal modes."""
        basis = self._basis()
        grid = self._grid(9)  # 9 planes > 5 toroidal modes
        transform = Transform(grid, basis, method="pest", build_pinv=True)
        assert transform.method == "pest"
        f = self._f(grid.nodes)
        np.testing.assert_allclose(transform.fit(f), self._coeffs(basis), atol=1e-12)
        # data outside the span: the fit must satisfy the normal equations
        x = f + 0.1 * np.random.default_rng(1).standard_normal(f.size)
        c = transform.fit(x)
        V = jnp.asarray(basis.evaluate(grid))
        np.testing.assert_allclose(
            V.T @ (V @ c - x), 0, atol=1e-10 * float(jnp.linalg.norm(V.T @ x))
        )

    @pytest.mark.unit
    def test_pest_fit_undersampled_zeta(self):
        """Toroidal modes n and n +- 3 alias, so f coefficients aren't unique."""
        basis = self._basis()
        grid = self._grid(3)
        transform = Transform(grid, basis, method="pest", build_pinv=True)
        f = self._f(grid.nodes)
        c = transform.fit(f)
        V = jnp.asarray(basis.evaluate(grid))
        np.testing.assert_allclose(V @ c, f, atol=1e-11)
        # minimum norm <=> c lies in the row space of V
        lam = jnp.linalg.lstsq(V.T, c)[0]
        np.testing.assert_allclose(V.T @ lam, c, atol=1e-11)

    @pytest.mark.unit
    def test_pest_ragged_planes(self):
        """Different node counts per plane: fit, transform and project all work."""
        basis = self._basis()
        grid = self._grid(5, ragged=True)
        transform = Transform(grid, basis, method="pest", build=True, build_pinv=True)
        assert transform.method == "pest"
        f = self._f(grid.nodes)
        np.testing.assert_allclose(transform.fit(f), self._coeffs(basis), atol=1e-12)
        np.testing.assert_allclose(
            transform.transform(self._coeffs(basis)), f, atol=1e-12
        )
        y = np.random.default_rng(2).standard_normal(grid.num_nodes)
        V = jnp.asarray(basis.evaluate(grid))
        np.testing.assert_allclose(transform.project(y), V.T @ y, atol=1e-11)

    @pytest.mark.unit
    def test_pest_project_is_adjoint(self):
        """project(y) = V^T y with V the basis evaluated at the nodes."""
        basis = self._basis()
        grid = self._grid(5)
        transform = Transform(grid, basis, method="pest", build=True)
        y = np.random.default_rng(3).standard_normal(grid.num_nodes)
        V = jnp.asarray(basis.evaluate(grid))
        np.testing.assert_allclose(transform.project(y), V.T @ y, atol=1e-11)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n_zeta, ragged", [(5, False), (5, True), (9, False), (3, False)]
    )
    def test_pest_project_matches_direct1(self, n_zeta, ragged):
        """Projection with pest is the same as with direct1."""
        basis = self._basis()
        grid = self._grid(n_zeta, ragged=ragged)
        pest = Transform(grid, basis, method="pest", build=True)
        direct1 = Transform(grid, basis, method="direct1", build=True)
        assert pest.method == "pest" and direct1.method == "direct1"
        y = np.random.default_rng(5).standard_normal(grid.num_nodes)
        np.testing.assert_allclose(pest.project(y), direct1.project(y), atol=1e-12)

    @pytest.mark.unit
    def test_pest_without_zeta_indices_falls_back(self):
        """A grid that does not carry zeta indices falls back to direct1, warning."""
        nodes = self._grid(5).nodes
        grid = Grid(nodes, NFP=self.NFP, sort=False, jitable=True)
        with pytest.warns(UserWarning, match="falling back to direct1"):
            transform = Transform(grid, self._basis(), method="pest")
        assert transform.method == "direct1"

    @pytest.mark.unit
    def test_pest_under_jit(self):
        """Built and applied under jit."""
        basis = self._basis()
        ref = self._grid(5)
        n = ref.num_nodes
        idx = dict(
            _unique_rho_idx=np.asarray(ref.unique_rho_idx),
            _inverse_rho_idx=np.asarray(ref.inverse_rho_idx),
            _unique_poloidal_idx=np.arange(n),
            _inverse_poloidal_idx=np.arange(n),
            _unique_zeta_idx=np.asarray(ref.unique_zeta_idx),
            _inverse_zeta_idx=np.asarray(ref.inverse_zeta_idx),
        )
        c = self._coeffs(basis)
        f = self._f(ref.nodes)
        y = np.random.default_rng(4).standard_normal(n)
        V = jnp.asarray(basis.evaluate(ref))

        def build(nodes):
            grid = Grid(nodes, NFP=self.NFP, sort=False, jitable=True, **idx)
            return Transform(
                grid,
                basis,
                derivs=np.array([[0, 1, 0]]),
                method="pest",
                build=True,
                build_pinv=True,
            )

        @jit
        def inside(nodes):
            transform = build(nodes)
            assert transform.method == "pest"
            assert transform.pest_plane_idx is None  # traced: no per-plane grouping
            return (
                transform.transform(c, 0, 1, 0),
                transform.fit(f),
                transform.project(y),
            )

        for dt_f, c_fit, proj in (
            inside(ref.nodes),  # built under jit
            (  # built outside, applied under jit
                jit(lambda c: build(ref.nodes).transform(c, 0, 1, 0))(c),
                jit(build(ref.nodes).fit)(f),
                jit(build(ref.nodes).project)(y),
            ),
        ):
            np.testing.assert_allclose(dt_f, self._f(ref.nodes, (0, 1, 0)), atol=1e-11)
            np.testing.assert_allclose(c_fit, c, atol=1e-12)
            np.testing.assert_allclose(proj, V.T @ y, atol=1e-11)

    @pytest.mark.unit
    def test_get_transforms_honors_pest(self):
        """An explicit method="pest" reaches the transforms."""
        from desc.equilibrium import Equilibrium

        eq = Equilibrium(L=3, M=3, N=2, NFP=self.NFP)
        grid = self._grid(5)
        transforms = get_transforms(["R", "Z"], eq, grid, method="pest")
        assert transforms["R"].method == "pest"
        assert transforms["Z"].method == "pest"
        # unchanged: no method -> auto's choice; jitable -> jitable
        assert get_transforms(["R"], eq, grid)["R"].method != "pest"
        assert get_transforms(["R"], eq, grid, jitable=True)["R"].method == "jitable"
