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
)
from desc.compute import get_transforms
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
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
        transf = Transform(grid, basis, derivs=0)

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

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["direct2", "fft"])
    def test_transform_methods_equal(self, method):
        """Tests that different methods for transform gives the same results."""
        L = 4
        M = 3
        N = 2
        Lnodes = 8
        Mnodes = 4
        Nnodes = 3
        NFP = 4

        grids = {
            "QuadratureGrid": QuadratureGrid(Lnodes, Mnodes, Nnodes, NFP),
            "LinearGrid_sym": LinearGrid(Lnodes, Mnodes, Nnodes, NFP, sym=True),
        }
        bases = {
            "FourierSeries": FourierSeries(N, NFP),
            "DoubleFourierSeries": DoubleFourierSeries(M, N, NFP),
            "DoubleFourierSeries_sin": DoubleFourierSeries(M, N, NFP, sym="sin"),
            "DoubleFourierSeries_cos": DoubleFourierSeries(M, N, NFP, sym="cos"),
            "ChebyshevDoubleFourierBasis": ChebyshevDoubleFourierBasis(L, M, N, NFP),
            "ZernikePolynomial": ZernikePolynomial(L, M),
            "FourierZernikeBasis": FourierZernikeBasis(L, M, N, NFP),
            "FourierZernikeBasis_cos": FourierZernikeBasis(L, M, N, NFP, sym="cos"),
            "FourierZernikeBasis_sin": FourierZernikeBasis(L, M, N, NFP, sym="sin"),
        }

        t1s = {
            gridname: {
                basisname: Transform(
                    grid,
                    basis,
                    derivs=0 if isinstance(basis, ChebyshevDoubleFourierBasis) else 3,
                    method="direct1",
                )
                for basisname, basis in bases.items()
            }
            for gridname, grid in grids.items()
        }
        t2s = {
            gridname: {
                basisname: Transform(
                    grid,
                    basis,
                    derivs=0 if isinstance(basis, ChebyshevDoubleFourierBasis) else 3,
                    method=method,
                )
                for basisname, basis in bases.items()
            }
            for gridname, grid in grids.items()
        }

        for gridname in grids.keys():
            for basisname in bases.keys():
                t1 = t1s[gridname][basisname]
                t2 = t2s[gridname][basisname]
                for d in t1.derivatives:
                    dr = d[0]
                    dv = d[1]
                    dz = d[2]
                    x = np.random.random(t1.basis.num_modes)
                    y1 = t1.transform(x, dr, dv, dz)
                    y2 = t2.transform(x, dr, dv, dz)
                    np.testing.assert_allclose(
                        y1,
                        y2,
                        atol=1e-10,
                        rtol=1e-10,
                        err_msg=f"failed on {gridname}, {basisname}, d={d}",
                    )

        M += 1
        N += 1
        Mnodes += 1
        Nnodes += 1
        grids = {
            "QuadratureGrid": QuadratureGrid(Lnodes, Mnodes, Nnodes, NFP),
            "LinearGrid_sym": LinearGrid(Lnodes, Mnodes, Nnodes, NFP, sym=True),
        }
        bases = {
            "FourierSeries": FourierSeries(N, NFP),
            "DoubleFourierSeries": DoubleFourierSeries(M, N, NFP),
            "DoubleFourierSeries_sin": DoubleFourierSeries(M, N, NFP, sym="sin"),
            "DoubleFourierSeries_cos": DoubleFourierSeries(M, N, NFP, sym="cos"),
            "ChebyshevDoubleFourierBasis": ChebyshevDoubleFourierBasis(L, M, N, NFP),
            "ZernikePolynomial": ZernikePolynomial(L, M),
            "FourierZernikeBasis": FourierZernikeBasis(L, M, N, NFP),
            "FourierZernikeBasis_cos": FourierZernikeBasis(L, M, N, NFP, sym="cos"),
            "FourierZernikeBasis_sin": FourierZernikeBasis(L, M, N, NFP, sym="sin"),
        }

        for gridname, grid in grids.items():
            for basisname, basis in bases.items():
                t1s[gridname][basisname].change_resolution(grid, basis)
                t2s[gridname][basisname].change_resolution(grid, basis)

        for gridname in grids.keys():
            for basisname in bases.keys():
                t1 = t1s[gridname][basisname]
                t2 = t2s[gridname][basisname]
                for d in t1.derivatives:
                    dr = d[0]
                    dv = d[1]
                    dz = d[2]
                    x = np.random.random(t1.basis.num_modes)
                    y1 = t1.transform(x, dr, dv, dz)
                    y2 = t2.transform(x, dr, dv, dz)
                    np.testing.assert_allclose(
                        y1,
                        y2,
                        atol=1e-10,
                        rtol=1e-10,
                        err_msg=f"failed on {gridname}, {basisname}, d={d}",
                    )

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["direct2", "fft"])
    def test_project(self, method):
        """Tests projection method for Galerkin method."""
        grids = [
            ConcentricGrid(L=4, M=2, N=5),
            ConcentricGrid(L=4, M=2, N=5),
            ConcentricGrid(L=4, M=2, N=5, sym=True),
        ]
        bases = [
            FourierZernikeBasis(L=1, M=5, N=3),
            FourierZernikeBasis(L=1, M=5, N=3, sym="cos"),
            FourierZernikeBasis(L=1, M=5, N=0, sym="sin"),
        ]

        for grid, basis in zip(grids, bases):
            t1 = Transform(grid, basis, method="direct1")
            t2 = Transform(grid, basis, method=method)
            t1.build()
            t2.build()
            y = np.random.random(grid.num_nodes)
            np.testing.assert_allclose(t1.project(y), t2.project(y))

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
    @pytest.mark.parametrize("method", ["direct1", "direct2", "fft"])
    def test_fit(self, method):
        """Test fitting with different methods."""
        grids = [
            ConcentricGrid(3, 3, 2, node_pattern="ocs"),
            ConcentricGrid(3, 3, 2, node_pattern="jacobi"),
            QuadratureGrid(3, 3, 2),
        ]
        bases = [
            FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi"),
            FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi", sym="cos"),
            FourierZernikeBasis(3, 3, 2, spectral_indexing="ansi", sym="sin"),
        ]

        rng = np.random.default_rng(123)

        for grid in grids:
            for basis in bases:
                transform = Transform(grid, basis, method=method, build_pinv=True)
                c = (0.5 - rng.random(basis.num_modes)) * abs(basis.modes).sum(axis=-1)
                x = transform.transform(c)
                c1 = transform.fit(x)
                np.testing.assert_allclose(
                    c, c1, atol=1e-12, err_msg=f"{grid}, {basis}, {method}"
                )

    @pytest.mark.unit
    def test_empty_grid(self):
        """Make sure we can build transforms with empty grids."""
        grid = Grid(nodes=np.empty((0, 3)))
        basis = FourierZernikeBasis(6, 0, 0)
        _ = Transform(grid, basis)

        basis = FourierZernikeBasis(6, 6, 6)
        _ = Transform(grid, basis)

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
