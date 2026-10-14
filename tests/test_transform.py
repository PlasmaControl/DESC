"""Tests for transforming from spectral coefficients to real space values."""

import numpy as np
import pytest

import desc.examples
from desc.backend import jit
from desc.basis import (
    ChebyshevDoubleFourierBasis,
    DoubleChebyshevFourierBasis,
    DoubleFourierSeries,
    FourierSeries,
    FourierZernikeBasis,
    PowerSeries,
    ZernikePolynomial,
)
from desc.compute import get_transforms
from desc.grid import (
    ConcentricGridFlux,
    CustomGridCylindrical,
    CustomGridFlux,
    LinearGridFlux,
    QuadratureGridCylindrical,
)
from desc.transform import MeshgridTransform, Transform


class TestTransform:
    """Tests Transform classes."""

    @pytest.mark.unit
    def test_eq(self):
        """Tests equals operator overload method."""
        grid_1 = LinearGridFlux(L=10, N=1)
        grid_2 = LinearGridFlux(M=2, N=2)
        grid_3 = ConcentricGridFlux(L=4, M=2, N=2)

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
        grid = LinearGridFlux(L=10)
        basis = PowerSeries(L=2, sym=False)
        for T in [MeshgridTransform, Transform]:
            transf = T(grid, basis, derivs=0)

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
        grid = LinearGridFlux(L=10)
        basis = PowerSeries(L=2, sym=False)

        for T in [MeshgridTransform, Transform]:
            transf = T(grid, basis, derivs=1)

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
        for T in [Transform, MeshgridTransform]:
            grid = LinearGridFlux(M=2, N=2, sym=True)
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
        grid = ConcentricGridFlux(L=4, M=2, N=2)
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
        grid = ConcentricGridFlux(L=4, M=2, N=2)
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

        grid_1 = LinearGridFlux(L=0)
        grid_3 = LinearGridFlux(L=2)
        grid_5 = LinearGridFlux(L=4)

        with pytest.warns(UserWarning):
            transf_1 = Transform(grid_1, basis, method="fft")
            transf_3 = Transform(grid_3, basis, method="fft")
            transf_5 = Transform(grid_5, basis, method="fft")

        transf_3.grid = grid_5
        assert transf_3.equiv(transf_5)

        transf_3.grid = grid_1
        assert transf_3.equiv(transf_1)

        np.testing.assert_allclose(transf_3.nodes, grid_1.nodes)

    @pytest.mark.unit
    def test_set_basis(self):
        """Tests the basis setter method."""
        grid = ConcentricGridFlux(L=4, M=2, N=1)

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

    @pytest.mark.unit
    def test_fft(self):
        """Tests Fast Fourier Transform method."""
        grid = LinearGridFlux(N=16)
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

        grid = ConcentricGridFlux(Lnodes, Mnodes, Nnodes, NFP)
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

        grid = ConcentricGridFlux(Lnodes, Mnodes, Nnodes, NFP, sym=True)
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

    @pytest.mark.unit
    def test_project(self):
        """Tests projection method for Galerkin method."""
        basis = FourierZernikeBasis(L=1, M=5, N=3)
        grid = ConcentricGridFlux(L=4, M=2, N=5)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        transform.build()
        dtransform1.build()
        dtransform2.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))

        basis = FourierZernikeBasis(L=1, M=5, N=3, sym="cos")
        grid = ConcentricGridFlux(L=4, M=2, N=5)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        transform.build()
        dtransform1.build()
        dtransform2.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))

        basis = FourierZernikeBasis(L=1, M=5, N=0, sym="sin")
        grid = ConcentricGridFlux(L=4, M=2, N=5, sym=True)
        transform = Transform(grid, basis, method="fft")
        dtransform1 = Transform(grid, basis, method="direct1")
        dtransform2 = Transform(grid, basis, method="direct2")
        transform.build()
        dtransform1.build()
        dtransform2.build()

        y = np.random.random(grid.num_nodes)

        np.testing.assert_allclose(transform.project(y), dtransform1.project(y))
        np.testing.assert_allclose(transform.project(y), dtransform2.project(y))

    @pytest.mark.unit
    def test_fft_warnings(self):
        """Test that warnings are thrown when trying to use fft where it won't work."""
        g = CustomGridFlux(np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]]))
        b = ZernikePolynomial(L=2, M=2)
        with pytest.warns(UserWarning, match="compatible grid"):
            t = Transform(g, b, method="fft")
        assert t.method == "direct1"

        g = LinearGridFlux(rho=2, M=2, N=2, NFP=2)
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

        g = LinearGridFlux(rho=2, M=2, N=2)
        b = DoubleFourierSeries(M=1, N=3)
        with pytest.warns(UserWarning, match="can not undersample in x2"):
            t = Transform(g, b, method="fft")
        assert t.method == "direct2"

        b._fft[2] = False
        g = LinearGridFlux(2, 3, 4)
        with pytest.warns(UserWarning, match="compatible basis"):
            t = Transform(g, b, method="fft")
        assert t.method == "direct1"

    @pytest.mark.unit
    def test_direct2_warnings(self):
        """Test that warnings are thrown when trying to use direct2 if it won't work."""
        g = CustomGridFlux(np.array([[0, 0, -1], [1, 1, 0], [1, 1, 1]]))
        b = ZernikePolynomial(L=2, M=2)
        with pytest.warns(UserWarning, match="requires compatible grid"):
            t = Transform(g, b, method="direct2")
        assert t.method == "direct1"

        b._fft[2] = False
        g = LinearGridFlux(2, 3, 4)
        with pytest.warns(UserWarning, match="compatible basis"):
            t = Transform(g, b, method="direct2")
        assert t.method == "direct1"

    @pytest.mark.unit
    def test_fit_direct1_and_jitable(self):
        """Test fitting with direct1 and jitable method."""
        basis = FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi")
        grid = ConcentricGridFlux(3, 3, 2, node_pattern="ocs")
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
        grid = ConcentricGridFlux(4, 4, 3, node_pattern="jacobi")
        transform = Transform(grid, basis, method="direct2", build_pinv=True)
        np.random.seed(1)
        c = (0.5 - np.random.random(basis.num_modes)) * abs(basis.modes).sum(axis=-1)
        x = transform.transform(c)
        c1 = transform.fit(x)
        np.testing.assert_allclose(c, c1, atol=1e-12)

    @pytest.mark.unit
    def test_fit_fft(self):
        """Test fitting with fft method."""
        basis = FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi")
        grid = LinearGridFlux(4, 4, 3)
        transform = Transform(grid, basis, method="fft", build_pinv=True)
        np.random.seed(2)
        c = (0.5 - np.random.random(basis.num_modes)) * abs(basis.modes).sum(axis=-1)
        x = transform.transform(c)
        c1 = transform.fit(x)
        np.testing.assert_allclose(c, c1, atol=1e-12)

    @pytest.mark.unit
    def test_empty_grid(self):
        """Make sure we can build transforms with empty grids."""
        grid = CustomGridFlux(nodes=np.empty((0, 3)))
        basis = FourierZernikeBasis(6, 0, 0)
        _ = Transform(grid, basis)

        basis = FourierZernikeBasis(6, 6, 6)
        _ = Transform(grid, basis)

    @pytest.mark.unit
    def test_Z_projection(self):
        """Make sure we always have the 0,0,0 derivative for projections."""
        eq = desc.examples.get("DSHAPE")
        data_keys = ["F_rho", "|grad(rho)|", "sqrt(g)", "F_helical", "|e^helical|"]
        grid = ConcentricGridFlux(
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
                grid = LinearGridFlux(L=2, M=2, zeta=N)
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


class TestMeshgridTransform:
    """Tests MeshgridTransform classes."""

    @pytest.mark.unit
    def test_eq(self):
        """Tests equals operator overload method for MeshgridTransform."""
        grid_1 = LinearGridFlux(L=10, M=3, N=1)
        grid_2 = LinearGridFlux(L=2, M=2, N=2)
        grid_3 = QuadratureGridCylindrical(L=4, M=2, N=2)

        basis_1 = ChebyshevDoubleFourierBasis(L=1, M=1, N=1)
        basis_2 = DoubleChebyshevFourierBasis(L=1, M=1, N=1)

        transf_11 = MeshgridTransform(grid_1, basis_1)
        transf_21 = MeshgridTransform(grid_2, basis_1)
        transf_31 = MeshgridTransform(grid_3, basis_1)
        transf_32 = MeshgridTransform(grid_3, basis_2)
        transf_32b = MeshgridTransform(grid_3, basis_2)

        assert not transf_11.equiv(transf_21)
        assert not transf_31.equiv(transf_32)
        assert transf_32.equiv(transf_32b)

    @pytest.mark.unit
    def test_volume_chebyshev(self):
        """Tests MeshgridTransform of Chebyshev-Fourier basis in a toroidal volume."""
        grid = LinearGridFlux(L=4, M=2, N=2)
        basis = ChebyshevDoubleFourierBasis(L=1, M=1, N=1, sym=False)
        transf = MeshgridTransform(grid, basis)

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
    def test_set_grid(self):
        """Tests the grid setter method for MeshgridTransform."""
        basis = ChebyshevDoubleFourierBasis(L=1, M=1, N=1)

        grid_1 = LinearGridFlux(L=0, M=2)
        grid_3 = LinearGridFlux(L=2, M=2)
        grid_5 = LinearGridFlux(L=6, M=2)

        transf_1 = MeshgridTransform(grid_1, basis)
        transf_3 = MeshgridTransform(grid_3, basis)
        transf_5 = MeshgridTransform(grid_5, basis)

        transf_3.grid = grid_5
        assert transf_3.equiv(transf_5)

        transf_3.grid = grid_1
        assert transf_3.equiv(transf_1)

        np.testing.assert_allclose(transf_3.nodes, grid_1.nodes)

    @pytest.mark.unit
    def test_set_basis(self):
        """Tests the basis setter method for MeshgridTransform."""
        grid = LinearGridFlux(L=4, M=2, N=1)

        basis_20 = ChebyshevDoubleFourierBasis(L=1, M=2, N=0)
        basis_21 = ChebyshevDoubleFourierBasis(L=1, M=2, N=1)
        basis_31 = ChebyshevDoubleFourierBasis(L=1, M=3, N=1)

        transf_20 = MeshgridTransform(grid, basis_20)
        transf_21 = MeshgridTransform(grid, basis_21)
        transf_31 = MeshgridTransform(grid, basis_31)

        transf_21.basis = basis_31
        assert transf_21.equiv(transf_31)

        transf_21.basis = basis_20
        assert transf_21.equiv(transf_20)

        np.testing.assert_allclose(transf_21.modes, basis_20.modes)

    @pytest.mark.unit
    def test_rpz_transform(self):
        """Testing MeshgridTransform with R, phi, Z grid."""
        basis = DoubleChebyshevFourierBasis(L=12, M=4, N=5, NFP=2)
        grid = QuadratureGridCylindrical(L=basis.L, M=4, N=basis.N, NFP=basis.NFP)
        with pytest.warns(UserWarning) as record:
            transform = MeshgridTransform(
                grid,
                basis,
                derivs=2,
                build=True,
                build_pinv=True,
                method=["fft", "auto", "auto"],
            )
        for r in record:
            if "fft method along dimension x0 requires compatible grid" in str(
                r.message
            ):
                fft_warning_exists = True
        assert fft_warning_exists
        assert transform.method == ["dct", "fft", "dct"]
        # fit
        x = (
            (2 * grid.nodes[:, 0] - 1)
            * np.cos(basis.NFP * grid.nodes[:, 1])
            * (2 * (2 * grid.nodes[:, 2] - 1) ** 2 - 1)
        )
        c = transform.fit(x)
        np.testing.assert_allclose(basis.modes[np.abs(c) > 1e-10], [[1, 1, 2]])
        np.testing.assert_allclose(c[np.abs(c) > 1e-10], 1)

        d = (1, 2, 0)

        # transform
        x_reconstructed = transform.transform(c, dx0=d[0], dx1=d[1], dx2=d[2])
        x_prime = (
            2
            * (-basis.NFP**2 * np.cos(basis.NFP * grid.nodes[:, 1]))
            * (2 * (2 * grid.nodes[:, 2] - 1) ** 2 - 1)
        )
        np.testing.assert_allclose(x_prime, x_reconstructed, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_rtz_transform(self):
        """Testing MeshgridTransform with rho, theta, zeta grid."""
        basis = ChebyshevDoubleFourierBasis(6, 3, 4, NFP=3)
        grid = LinearGridFlux(6, 3, 4, NFP=basis.NFP)
        transform = MeshgridTransform(
            grid,
            basis,
            derivs=np.array([[2, 3, 4]]),
            build=True,
            build_pinv=True,
            method=["auto", "auto", "auto"],
        )
        assert transform.method == ["direct", "fft", "fft"]
        # fit
        x = (
            (
                8 * (2 * grid.nodes[:, 0] - 1) ** 4
                - 8 * (2 * grid.nodes[:, 0] - 1) ** 2
                + 1
            )
            * np.cos(grid.nodes[:, 1])
            * np.sin(3 * basis.NFP * grid.nodes[:, 2])
        )
        c = transform.fit(x)
        np.testing.assert_allclose(basis.modes[np.abs(c) > 1e-10], [[4, 1, -3]])
        np.testing.assert_allclose(c[np.abs(c) > 1e-10], 1)

        d = (2, 3, 5)

        # transform
        x_reconstructed = transform.transform(c, dx0=d[0], dx1=d[1], dx2=d[2])
        x_prime = (
            (384 * (2 * grid.nodes[:, 0] - 1) ** 2 - 64)
            * np.sin(grid.nodes[:, 1])
            * ((3 * basis.NFP) ** 5)
            * np.cos(3 * basis.NFP * grid.nodes[:, 2])
        )

        # check
        np.testing.assert_allclose(x_prime, x_reconstructed, rtol=1e-10, atol=1e-7)

    @pytest.mark.unit
    def test_direct_transform(self):
        """Test MeshgridTransform with [direct, direct, direct]."""
        # Custom grid is incompatible with fft and dct
        basis = DoubleChebyshevFourierBasis(L=5, M=4, N=5, NFP=1)
        grid = CustomGridCylindrical.create_meshgrid(
            nodes=[
                np.linspace(0, 1, 8),
                np.linspace(0, 2 * np.pi / basis.NFP, 6, endpoint=False),
                np.linspace(0, 1, 7),
            ],
            NFP=basis.NFP,
        )
        transform = MeshgridTransform(
            grid,
            basis,
            derivs=2,
            build=True,
            build_pinv=True,
            method=["auto", "auto", "auto"],
        )
        assert transform.method == ["direct", "direct", "direct"]
        # fit
        x = (
            (2 * grid.nodes[:, 0] - 1)
            * np.cos(basis.NFP * grid.nodes[:, 1])
            * (2 * (2 * grid.nodes[:, 2] - 1) ** 2 - 1)
        )
        c = transform.fit(x)
        np.testing.assert_allclose(basis.modes[np.abs(c) > 1e-10], [[1, 1, 2]])
        np.testing.assert_allclose(c[np.abs(c) > 1e-10], 1)

        d = (1, 2, 0)

        # transform
        x_reconstructed = transform.transform(c, dx0=d[0], dx1=d[1], dx2=d[2])
        x_prime = (
            2
            * (-basis.NFP**2 * np.cos(basis.NFP * grid.nodes[:, 1]))
            * (2 * (2 * grid.nodes[:, 2] - 1) ** 2 - 1)
        )
        np.testing.assert_allclose(x_prime, x_reconstructed, rtol=1e-10, atol=1e-10)

        # PowerSeries basis is incompatible with fft and dct
        basis = PowerSeries(5, sym=False)
        grid = QuadratureGridCylindrical(5, 2, 2)
        transform = MeshgridTransform(
            grid,
            basis,
            derivs=2,
            build=True,
            build_pinv=True,
            method=["auto", "auto", "auto"],
        )
        assert transform.method == ["direct", "direct", "direct"]

        # fit
        x = grid.nodes[:, 0] ** 4
        c = transform.fit(x)
        np.testing.assert_allclose(basis.modes[np.abs(c) > 1e-10], [[4, 0, 0]])
        np.testing.assert_allclose(c[np.abs(c) > 1e-10], 1)

        d = (2, 0, 0)

        # transform
        x_reconstructed = transform.transform(c, dx0=d[0], dx1=d[1], dx2=d[2])
        x_prime = 12 * grid.nodes[:, 0] ** 2
        np.testing.assert_allclose(x_prime, x_reconstructed, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_meshgrid_errors(self):
        """Test error handling for MeshgridTransform initialization."""
        # raise an error when method list is not length 3
        grid = LinearGridFlux(5, 5, 5)
        basis = ChebyshevDoubleFourierBasis(5, 5, 5)
        with pytest.raises(
            ValueError,
            match="Method must be a list of length 3,",
        ):
            MeshgridTransform(grid, basis, method=["dct", "fft"])

        # raise an error when grid is symmetric
        grid = LinearGridFlux(5, 5, 5, NFP=1, sym=True)
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=1)
        with pytest.raises(
            NotImplementedError,
            match="MeshgridTransform for symmetric grids has not been implemented",
        ):
            MeshgridTransform(grid, basis)

        # raise an error when basis is symmetric
        grid = LinearGridFlux(5, 5, 5, NFP=1)
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=1, sym="cos")
        with pytest.raises(
            ValueError,
            match="MeshgridTransform requires a tensor product basis",
        ):
            MeshgridTransform(grid, basis)

        # raise an error when grid is not a meshgrid
        grid = ConcentricGridFlux(5, 5, 5, NFP=1)
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=1)
        with pytest.raises(
            ValueError, match="MeshgridTransform requires a meshgrid grid"
        ):
            MeshgridTransform(grid, basis)

        # raise an error when basis is not a tensor product basis
        grid = LinearGridFlux(5, 5, 5, NFP=1)
        basis = FourierZernikeBasis(L=5, M=5, N=5, NFP=1)
        with pytest.raises(
            ValueError, match="MeshgridTransform requires a tensor product basis"
        ):
            MeshgridTransform(grid, basis)

        # raise an error when NFP does not match
        grid = LinearGridFlux(5, 5, 5, NFP=1)
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=2)
        with pytest.raises(
            ValueError, match="Unequal number of field periods for grid 1 and basis 2"
        ):
            MeshgridTransform(grid, basis)

        # raise an error when toroidal coordinates differ
        grid = LinearGridFlux(5, 5, 5, NFP=2)
        basis = DoubleChebyshevFourierBasis(L=5, M=5, N=5, NFP=2)
        with pytest.raises(
            ValueError,
            match="Basis and grid have different toroidal coordinates: basis=1, grid=2",
        ):
            MeshgridTransform(grid, basis)

        # raise a warning when grid is undersampled in any direction
        grid = LinearGridFlux(5, 5, 5, NFP=2)
        basis = ChebyshevDoubleFourierBasis(L=6, M=5, N=5, NFP=2)
        with pytest.warns(
            UserWarning, match="Grid is undersampled along dimensions x0"
        ):
            MeshgridTransform(grid, basis)

    @pytest.mark.unit
    def test_fft_errors(self):
        """Test error handling for FFT method in MeshgridTransform."""
        # warnings when the grid is not compatible with the fft method
        grid = CustomGridCylindrical.create_meshgrid(
            nodes=[np.linspace(0, 1, 10), np.array([0, 0.3, 1]), np.linspace(0, 1, 10)]
        )
        basis = DoubleChebyshevFourierBasis(L=grid.L, M=grid.M, N=grid.N, NFP=1)

        with pytest.warns(UserWarning) as record:
            MeshgridTransform(grid, basis, method=["auto", "fft", "auto"])

        fft_warning_exists = False
        dct_warning_exists = False
        for r in record:
            if "fft method along dimension x1 requires compatible grid" in str(
                r.message
            ):
                fft_warning_exists = True
            if "dct method along dimension x1 requires compatible grid" in str(
                r.message
            ):
                dct_warning_exists = True
        assert fft_warning_exists and dct_warning_exists

        # warnings when the basis is not compatible with the fft method
        grid = LinearGridFlux(3, 3, 3, NFP=1)
        basis = DoubleChebyshevFourierBasis(L=grid.L, M=grid.M, N=grid.N, NFP=1)

        with pytest.warns(UserWarning) as record:
            MeshgridTransform(grid, basis, method=["auto", "auto", "fft"])

        fft_warning_exists = False
        dct_warning_exists = False
        for r in record:
            if "fft method along dimension x2 requires compatible basis" in str(
                r.message
            ):
                fft_warning_exists = True
            if "dct method along dimension x2 requires compatible grid" in str(
                r.message
            ):
                dct_warning_exists = True
        assert fft_warning_exists and dct_warning_exists

    @pytest.mark.unit
    def test_dct_errors(self):
        """Test error handling for DCT method in MeshgridTransform."""
        # warnings when dct is used with incompatible grid
        basis = ChebyshevDoubleFourierBasis(6, 3, 4, NFP=3)
        grid = LinearGridFlux(6, 3, 4, NFP=basis.NFP)
        with pytest.warns(
            UserWarning, match="dct method along dimension x1 requires compatible grid"
        ):
            transform = MeshgridTransform(grid, basis, method=["auto", "dct", "auto"])
        assert transform.method == ["direct", "direct", "fft"]

        # warnings when dct is used with incompatible basis
        basis = PowerSeries(6, sym=False)
        grid = QuadratureGridCylindrical(6, 0, 0)
        with pytest.warns(
            UserWarning, match="dct method along dimension x0 requires compatible basis"
        ):
            transform = MeshgridTransform(grid, basis, method=["dct", "auto", "auto"])
        assert transform.method == ["direct", "direct", "direct"]

        # warnings when dct is used for a grid and basis with different resolution
        basis = DoubleChebyshevFourierBasis(6, 3, 4, NFP=3)
        grid = QuadratureGridCylindrical(6, 3, 6, NFP=3)
        match = (
            "dct method along dimension x2 requires grid"
            + " and basis to have the same resolution"
        )
        with pytest.warns(UserWarning, match=match):
            transform = MeshgridTransform(grid, basis, method=["auto", "auto", "dct"])
        assert transform.method == ["dct", "fft", "direct"]

    @pytest.mark.unit
    def test_change_resolution(self):
        """Test that changing resolution works as expected."""
        # automatically change method when changing resolution
        grid = LinearGridFlux(5, 5, 5, NFP=2)
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=2)
        transform = MeshgridTransform(grid, basis)
        assert transform.method == ["direct", "fft", "fft"]
        grid = CustomGridFlux.create_meshgrid(
            nodes=[
                np.linspace(0, 1, 6),
                np.linspace(0, 2 * np.pi / 2, 11, endpoint=False),
                np.linspace(0, 2 * np.pi / basis.NFP, 11),
            ]
        )
        transform.change_resolution(grid=grid)
        assert transform.method == ["direct", "direct", "direct"]
        assert transform.grid.equiv(grid)

        # check that changing the basis actually works
        basis = ChebyshevDoubleFourierBasis(L=4, M=4, N=5, NFP=2)
        transform.change_resolution(basis=basis)
        assert transform.basis.equiv(basis)

        grid = LinearGridFlux(5, 5, 5, NFP=2)
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=2)
        transform = MeshgridTransform(grid, basis)

        # raise an error when method is not of length 3
        with pytest.raises(
            ValueError,
            match="Method must be a list of length 3",
        ):
            transform.change_resolution(method=["fft", "dct"])

        # raise an error when grid is symmetric
        grid = LinearGridFlux(5, 5, 5, NFP=1, sym=True)
        with pytest.raises(
            NotImplementedError,
            match="MeshgridTransform for symmetric grids has not been implemented",
        ):
            transform.change_resolution(grid=grid)

        # raise an error when basis is symmetric so is not a tensor product
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=1, sym="cos")
        with pytest.raises(
            ValueError,
            match="MeshgridTransform requires a tensor product basis",
        ):
            transform.change_resolution(basis=basis)

        # raise an error when grid is not a meshgrid
        grid = ConcentricGridFlux(5, 5, 5, NFP=2)
        with pytest.raises(
            ValueError,
            match="MeshgridTransform requires a meshgrid grid, got ConcentricGridFlux",
        ):
            transform.change_resolution(grid=grid)

        # raise an error when basis is not a tensor product basis
        basis = FourierZernikeBasis(L=5, M=5, N=5, NFP=1)
        with pytest.raises(
            ValueError, match="MeshgridTransform requires a tensor product basis"
        ):
            transform.change_resolution(basis=basis)

        # raise an error when NFP does not match
        basis = ChebyshevDoubleFourierBasis(L=5, M=5, N=5, NFP=3)
        with pytest.raises(
            ValueError, match="Unequal number of field periods for grid 2 and basis 3"
        ):
            transform.change_resolution(basis=basis)

        # raise an error when toroidal coordinates differ
        basis = DoubleChebyshevFourierBasis(L=5, M=5, N=5, NFP=2)
        match = "Basis and grid have different toroidal coordinates: basis=1, grid=2."
        with pytest.raises(
            ValueError,
            match=match,
        ):
            transform.change_resolution(basis=basis)


@pytest.mark.unit
def test_transform_pytree():
    """Ensure that Transforms are valid pytree/JAX types."""
    grid = LinearGridFlux(5, 6, 7)
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


@pytest.mark.unit
def test_NFP_warning():
    """Make sure we only warn about basis/grid NFP in cases where it matters."""
    rho = np.linspace(0, 1, 20)
    g01 = LinearGridFlux(rho=rho, M=5, N=0, NFP=1)
    g02 = LinearGridFlux(rho=rho, M=5, N=0, NFP=2)
    g21 = LinearGridFlux(rho=rho, M=5, N=5, NFP=1)
    g22 = LinearGridFlux(rho=rho, M=5, N=5, NFP=2)
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
