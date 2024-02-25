"""Tests compute utilities."""

import jax
import numpy as np
import pytest

from desc.backend import fori_loop, jnp, put, root_scalar
from desc.basis import FourierZernikeBasis
from desc.compute.geom_utils import rotation_matrix
from desc.compute.utils import (
    _get_grid_surface,
    cubic_poly_roots,
    field_line_to_desc_coords,
    line_integrals,
    polyder,
    polyint,
    polyval,
    surface_averages,
    surface_integrals,
    surface_integrals_transform,
    surface_max,
    surface_min,
    surface_variance,
)
from desc.examples import get
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
from desc.transform import Transform

# arbitrary choice
L = 5
M = 5
N = 2
NFP = 3


class TestComputeUtils:
    """Tests for compute utilities related to surface averaging, etc."""

    @staticmethod
    def surface_integrals(grid, q=np.array([1.0]), surface_label="rho"):
        """Compute a surface integral for each surface in the grid.

        Notes
        -----
            It is assumed that the integration surface has area 4π² when the
            surface label is rho and area 2π when the surface label is theta or
            zeta. You may want to multiply q by the surface area Jacobian.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the nodes to evaluate at.
        q : ndarray
            Quantity to integrate.
            The first dimension of the array should have size ``grid.num_nodes``.

            When ``q`` is 1-dimensional, the intention is to integrate,
            over the domain parameterized by rho, theta, and zeta,
            a scalar function over the previously mentioned domain.

            When ``q`` is 2-dimensional, the intention is to integrate,
            over the domain parameterized by rho, theta, and zeta,
            a vector-valued function over the previously mentioned domain.

            When ``q`` is 3-dimensional, the intention is to integrate,
            over the domain parameterized by rho, theta, and zeta,
            a matrix-valued function over the previously mentioned domain.
        surface_label : str
            The surface label of rho, theta, or zeta to compute the integration over.

        Returns
        -------
        integrals : ndarray
            Surface integral of the input over each surface in the grid.

        """
        _, _, spacing, has_endpoint_dupe = _get_grid_surface(grid, surface_label)
        weights = (spacing.prod(axis=1) * np.nan_to_num(q).T).T

        surfaces = {}
        nodes = grid.nodes[:, {"rho": 0, "theta": 1, "zeta": 2}[surface_label]]
        # collect node indices for each surface_label surface
        for grid_row_idx, surface_label_value in enumerate(nodes):
            surfaces.setdefault(surface_label_value, []).append(grid_row_idx)
        # integration over non-contiguous elements
        integrals = [weights[surfaces[key]].sum(axis=0) for key in sorted(surfaces)]
        if has_endpoint_dupe:
            integrals[0] = integrals[-1] = integrals[0] + integrals[-1]
        return np.asarray(integrals)

    @pytest.mark.unit
    def test_surface_integrals(self):
        """Test surface_integrals against a more intuitive implementation.

        This test should ensure that the algorithm in implementation is correct
        on different types of grids (e.g. LinearGrid, ConcentricGrid). Each test
        should also be done on grids with duplicate nodes (e.g. endpoint=True).
        """

        def test_b_theta(surface_label, grid, eq):
            q = eq.compute("B_theta", grid=grid)["B_theta"]
            integrals = surface_integrals(grid, q, surface_label, expand_out=False)
            unique_size = {
                "rho": grid.num_rho,
                "theta": grid.num_theta,
                "zeta": grid.num_zeta,
            }[surface_label]
            assert integrals.shape == (unique_size,), surface_label

            desired = self.surface_integrals(grid, q, surface_label)
            np.testing.assert_allclose(
                integrals, desired, atol=1e-16, err_msg=surface_label
            )

        eq = get("W7-X")
        lg = LinearGrid(L=L, M=M, N=N, NFP=eq.NFP, endpoint=False)
        lg_endpoint = LinearGrid(L=L, M=M, N=N, NFP=eq.NFP, endpoint=True)
        cg_sym = ConcentricGrid(L=L, M=M, N=N, NFP=eq.NFP, sym=True)
        for label in ("rho", "theta", "zeta"):
            test_b_theta(label, lg, eq)
            test_b_theta(label, lg_endpoint, eq)
            if label != "theta":
                # theta integrals are poorly defined on concentric grids
                test_b_theta(label, cg_sym, eq)

    @pytest.mark.unit
    def test_surface_integrals_transform(self):
        """Test surface integral of a kernel function."""

        def test(surface_label, grid):
            ints = np.arange(grid.num_nodes)
            # better to test when all elements have the same sign
            q = np.abs(np.outer(np.cos(ints), np.sin(ints)))
            # This q represents the kernel function
            # K_{u_1} = |cos(x(u_1, u_2, u_3)) * sin(x(u_4, u_5, u_6))|
            # The first dimension of q varies the domain u_1, u_2, and u_3
            # and the second dimension varies the codomain u_4, u_5, u_6.
            integrals = surface_integrals_transform(grid, surface_label)(q)
            unique_size = {
                "rho": grid.num_rho,
                "theta": grid.num_theta,
                "zeta": grid.num_zeta,
            }[surface_label]
            assert integrals.shape == (unique_size, grid.num_nodes), surface_label

            desired = self.surface_integrals(grid, q, surface_label)
            np.testing.assert_allclose(integrals, desired, err_msg=surface_label)

        cg = ConcentricGrid(L=L, M=M, N=N, sym=True, NFP=NFP)
        lg = LinearGrid(L=L, M=M, N=N, sym=True, NFP=NFP, endpoint=True)
        test("rho", cg)
        test("theta", lg)
        test("zeta", cg)

    @pytest.mark.unit
    def test_surface_averages_vector_functions(self):
        """Test surface averages of vector-valued, function-valued integrands."""

        def test(surface_label, grid):
            g_size = grid.num_nodes  # not a choice; required
            f_size = g_size // 10 + (g_size < 10)
            # arbitrary choice, but f_size != v_size != g_size is better to test
            v_size = g_size // 20 + (g_size < 20)
            g = np.cos(np.arange(g_size))
            fv = np.sin(np.arange(f_size * v_size).reshape(f_size, v_size))
            # better to test when all elements have the same sign
            q = np.abs(np.einsum("g,fv->gfv", g, fv))
            sqrt_g = np.arange(g_size).astype(float)

            averages = surface_averages(grid, q, sqrt_g, surface_label)
            assert averages.shape == q.shape == (g_size, f_size, v_size), surface_label

            desired = (
                self.surface_integrals(grid, (sqrt_g * q.T).T, surface_label).T
                / self.surface_integrals(grid, sqrt_g, surface_label)
            ).T
            np.testing.assert_allclose(
                grid.compress(averages, surface_label), desired, err_msg=surface_label
            )

        cg = ConcentricGrid(L=L, M=M, N=N, sym=True, NFP=NFP)
        lg = LinearGrid(L=L, M=M, N=N, sym=True, NFP=NFP, endpoint=True)
        test("rho", cg)
        test("theta", lg)
        test("zeta", cg)

    @pytest.mark.unit
    def test_surface_area(self):
        """Test that surface_integrals(ds) is 4π² for rho, 2pi for theta, zeta.

        This test should ensure that surfaces have the correct area on grids
        constructed by specifying L, M, N and by specifying an array of nodes.
        Each test should also be done on grids with duplicate nodes
        (e.g. endpoint=True) and grids with symmetry.
        """

        def test(surface_label, grid):
            areas = surface_integrals(
                grid, surface_label=surface_label, expand_out=False
            )
            correct_area = 4 * np.pi**2 if surface_label == "rho" else 2 * np.pi
            np.testing.assert_allclose(areas, correct_area, err_msg=surface_label)

        lg = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=False, endpoint=False)
        lg_sym = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=True, endpoint=False)
        lg_endpoint = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=False, endpoint=True)
        lg_sym_endpoint = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=True, endpoint=True)
        rho = np.linspace(1, 0, L)[::-1]
        theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
        theta_endpoint = np.linspace(0, 2 * np.pi, M, endpoint=True)
        zeta = np.linspace(0, 2 * np.pi / NFP, N, endpoint=False)
        zeta_endpoint = np.linspace(0, 2 * np.pi / NFP, N, endpoint=True)
        lg_2 = LinearGrid(
            rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=False, endpoint=False
        )
        lg_2_sym = LinearGrid(
            rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=True, endpoint=False
        )
        lg_2_endpoint = LinearGrid(
            rho=rho,
            theta=theta_endpoint,
            zeta=zeta_endpoint,
            NFP=NFP,
            sym=False,
            endpoint=True,
        )
        lg_2_sym_endpoint = LinearGrid(
            rho=rho,
            theta=theta_endpoint,
            zeta=zeta_endpoint,
            NFP=NFP,
            sym=True,
            endpoint=True,
        )
        cg = ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=False)
        cg_sym = ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=True)

        for label in ("rho", "theta", "zeta"):
            test(label, lg)
            test(label, lg_sym)
            test(label, lg_endpoint)
            test(label, lg_sym_endpoint)
            test(label, lg_2)
            test(label, lg_2_sym)
            test(label, lg_2_endpoint)
            test(label, lg_2_sym_endpoint)
            if label != "theta":
                # theta integrals are poorly defined on concentric grids
                test(label, cg)
                test(label, cg_sym)

    @pytest.mark.unit
    def test_line_length(self):
        """Test that line_integrals(dl) is 1 for rho, 2π for theta, zeta.

        This test should ensure that lines have the correct length on grids
        constructed by specifying L, M, N and by specifying an array of nodes.
        """

        def test(grid):
            if not isinstance(grid, ConcentricGrid):
                for theta_val in grid.nodes[grid.unique_theta_idx, 1]:
                    result = line_integrals(
                        grid,
                        line_label="rho",
                        fix_surface=("theta", theta_val),
                        expand_out=False,
                    )
                    np.testing.assert_allclose(result, 1)
                for rho_val in grid.nodes[grid.unique_rho_idx, 0]:
                    result = line_integrals(
                        grid,
                        line_label="zeta",
                        fix_surface=("rho", rho_val),
                        expand_out=False,
                    )
                    np.testing.assert_allclose(result, 2 * np.pi)
            for zeta_val in grid.nodes[grid.unique_zeta_idx, 2]:
                result = line_integrals(
                    grid,
                    line_label="theta",
                    fix_surface=("zeta", zeta_val),
                    expand_out=False,
                )
                np.testing.assert_allclose(result, 2 * np.pi)

        lg = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=False)
        lg_sym = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=True)
        rho = np.linspace(1, 0, L)[::-1]
        theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / NFP, N, endpoint=False)
        lg_2 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=False)
        lg_2_sym = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=True)
        cg = ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=False)
        cg_sym = ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=True)

        test(lg)
        test(lg_sym)
        test(lg_2)
        test(lg_2_sym)
        test(cg)
        test(cg_sym)

    @pytest.mark.unit
    def test_surface_averages_identity_op(self):
        """Test flux surface averages of surface functions are identity operations."""
        eq = get("W7-X")
        grid = ConcentricGrid(L=L, M=M, N=N, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["p", "sqrt(g)"], grid=grid)
        pressure_average = surface_averages(grid, data["p"], data["sqrt(g)"])
        np.testing.assert_allclose(data["p"], pressure_average)

    @pytest.mark.unit
    def test_surface_averages_homomorphism(self):
        """Test flux surface averages of surface functions are additive homomorphisms.

        Meaning average(a + b) = average(a) + average(b).
        """
        eq = get("W7-X")
        grid = ConcentricGrid(L=L, M=M, N=N, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["|B|", "|B|_t", "sqrt(g)"], grid=grid)
        a = surface_averages(grid, data["|B|"], data["sqrt(g)"])
        b = surface_averages(grid, data["|B|_t"], data["sqrt(g)"])
        a_plus_b = surface_averages(grid, data["|B|"] + data["|B|_t"], data["sqrt(g)"])
        np.testing.assert_allclose(a_plus_b, a + b)

    @pytest.mark.unit
    def test_surface_integrals_against_shortcut(self):
        """Test integration against less general methods."""
        grid = ConcentricGrid(L=L, M=M, N=N, NFP=NFP)
        ds = grid.spacing[:, :2].prod(axis=-1)
        # something arbitrary that will give different sum across surfaces
        q = np.arange(grid.num_nodes) ** 2
        # The predefined grids sort nodes in zeta surface chunks.
        # To compute a quantity local to a surface, we can reshape it into zeta
        # surface chunks and compute across the chunks.
        result = grid.expand(
            (ds * q).reshape((grid.num_zeta, -1)).sum(axis=-1),
            surface_label="zeta",
        )
        np.testing.assert_allclose(
            surface_integrals(grid, q, surface_label="zeta"),
            desired=result,
        )

    @pytest.mark.unit
    def test_surface_averages_against_shortcut(self):
        """Test averaging against less general methods."""
        # test on zeta surfaces
        grid = LinearGrid(L=L, M=M, N=N, NFP=NFP)
        # something arbitrary that will give different average across surfaces
        q = np.arange(grid.num_nodes) ** 2
        # The predefined grids sort nodes in zeta surface chunks.
        # To compute a quantity local to a surface, we can reshape it into zeta
        # surface chunks and compute across the chunks.
        mean = grid.expand(
            q.reshape((grid.num_zeta, -1)).mean(axis=-1),
            surface_label="zeta",
        )
        # number of nodes per surface
        n = grid.num_rho * grid.num_theta
        np.testing.assert_allclose(np.bincount(grid.inverse_zeta_idx), desired=n)
        ds = grid.spacing[:, :2].prod(axis=-1)
        np.testing.assert_allclose(
            surface_integrals(grid, q / ds, surface_label="zeta") / n,
            desired=mean,
        )
        np.testing.assert_allclose(
            surface_averages(grid, q, surface_label="zeta"),
            desired=mean,
        )

        # test on grids with a single rho surface
        eq = get("W7-X")
        rho = np.array((1 - 1e-4) * np.random.default_rng().random() + 1e-4)
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["|B|", "sqrt(g)"], grid=grid)
        np.testing.assert_allclose(
            surface_averages(grid, data["|B|"], data["sqrt(g)"]),
            np.mean(data["sqrt(g)"] * data["|B|"]) / np.mean(data["sqrt(g)"]),
            err_msg="average with sqrt(g) fail",
        )
        np.testing.assert_allclose(
            surface_averages(grid, data["|B|"]),
            np.mean(data["|B|"]),
            err_msg="average without sqrt(g) fail",
        )

    @pytest.mark.unit
    def test_symmetry_surface_average_1(self):
        """Test surface average of a symmetric function."""

        def test(grid):
            r = grid.nodes[:, 0]
            t = grid.nodes[:, 1]
            z = grid.nodes[:, 2] * grid.NFP
            true_surface_avg = 5
            function_of_rho = 1 / (r + 0.35)
            f = (
                true_surface_avg
                + np.cos(t)
                - 0.5 * np.cos(z)
                + 3 * np.cos(t) * np.cos(z) ** 2
                - 2 * np.sin(z) * np.sin(t)
            ) * function_of_rho
            np.testing.assert_allclose(
                surface_averages(grid, f),
                true_surface_avg * function_of_rho,
                rtol=1e-15,
                err_msg=type(grid),
            )

        # these tests should be run on relatively low resolution grids,
        # or at least low enough so that the asymmetric spacing test fails
        L = [3, 3, 5, 3]
        M = [3, 6, 5, 7]
        N = [2, 2, 2, 2]
        NFP = [5, 3, 5, 3]
        sym = np.array([True, True, False, False])
        # to test code not tested on grids made with M=.
        even_number = 4
        n_theta = even_number - sym

        # asymmetric spacing
        with pytest.raises(AssertionError):
            theta = 2 * np.pi * np.array([t**2 for t in np.linspace(0, 1, max(M))])
            test(LinearGrid(L=max(L), theta=theta, N=max(N), sym=False))

        for i in range(len(L)):
            test(LinearGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            test(LinearGrid(L=L[i], theta=n_theta[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, 2 * np.pi, n_theta[i]),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, 2 * np.pi, n_theta[i] + 1),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(QuadratureGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i]))
            test(ConcentricGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            # nonuniform spacing when sym is False, but spacing is still symmetric
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, np.pi, n_theta[i]),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, np.pi, n_theta[i] + 1),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )

    @pytest.mark.unit
    def test_symmetry_surface_average_2(self):
        """Tests that surface averages are correct using specified basis."""

        def test(grid, basis, true_avg=1):
            transform = Transform(grid, basis)

            # random data with specified average on each surface
            coeffs = np.random.rand(basis.num_modes)
            coeffs[np.all(basis.modes[:, 1:] == [0, 0], axis=1)] = 0
            coeffs[np.all(basis.modes == [0, 0, 0], axis=1)] = true_avg

            # compute average for each surface in grid
            values = transform.transform(coeffs)
            numerical_avg = surface_averages(grid, values, expand_out=False)
            np.testing.assert_allclose(
                # values closest to axis are never accurate enough
                numerical_avg[isinstance(grid, ConcentricGrid) :],
                true_avg,
                err_msg=str(type(grid)) + " " + str(grid.sym),
            )

        M = 5
        M_grid = 13
        test(
            QuadratureGrid(L=M_grid, M=M_grid, N=0), FourierZernikeBasis(L=M, M=M, N=0)
        )
        test(
            LinearGrid(L=M_grid, M=M_grid, N=0, sym=True),
            FourierZernikeBasis(L=M, M=M, N=0, sym="cos"),
        )
        test(
            ConcentricGrid(L=M_grid, M=M_grid, N=0), FourierZernikeBasis(L=M, M=M, N=0)
        )
        test(
            ConcentricGrid(L=M_grid, M=M_grid, N=0, sym=True),
            FourierZernikeBasis(L=M, M=M, N=0, sym="cos"),
        )

    @pytest.mark.unit
    def test_surface_variance(self):
        """Test correctness of variance against less general methods."""
        grid = LinearGrid(L=L, M=M, N=N, NFP=NFP)
        # something arbitrary that will give different variance across surfaces
        q = np.arange(grid.num_nodes) ** 2

        # Test weighted sample variance with different weights.
        # positive weights to prevent cancellations that may hide implementation error
        weights = np.cos(q) * np.sin(q) + 5
        biased = surface_variance(
            grid, q, weights, bias=True, surface_label="zeta", expand_out=False
        )
        unbiased = surface_variance(
            grid, q, weights, surface_label="zeta", expand_out=False
        )
        # The predefined grids sort nodes in zeta surface chunks.
        # To compute a quantity local to a surface, we can reshape it into zeta
        # surface chunks and compute across the chunks.
        chunks = q.reshape((grid.num_zeta, -1))
        # The ds weights are built into the surface variance function.
        # So weights for np.cov should be ds * weights. Since ds is constant on
        # LinearGrid, we need to get the same result if we don't multiply by ds.
        weights = weights.reshape((grid.num_zeta, -1))
        for i in range(grid.num_zeta):
            np.testing.assert_allclose(
                biased[i],
                desired=np.cov(chunks[i], bias=True, aweights=weights[i]),
            )
            np.testing.assert_allclose(
                unbiased[i],
                desired=np.cov(chunks[i], aweights=weights[i]),
            )

        # Test weighted sample variance converges to unweighted sample variance
        # when all weights are equal.
        chunks = grid.expand(chunks, surface_label="zeta")
        np.testing.assert_allclose(
            surface_variance(grid, q, np.e, bias=True, surface_label="zeta"),
            desired=chunks.var(axis=-1),
        )
        np.testing.assert_allclose(
            surface_variance(grid, q, np.e, surface_label="zeta"),
            desired=chunks.var(axis=-1, ddof=1),
        )

    @pytest.mark.unit
    def test_surface_min_max(self):
        """Test the surface_min and surface_max functions."""
        for grid_type in [LinearGrid, QuadratureGrid, ConcentricGrid]:
            grid = grid_type(L=L, M=M, N=N, NFP=NFP)
            rho = grid.nodes[:, 0]
            theta = grid.nodes[:, 1]
            zeta = grid.nodes[:, 2]
            # Make up an arbitrary function of the coordinates:
            B = (
                1.7
                + 0.4 * rho * np.cos(theta)
                + 0.8 * rho * rho * np.cos(2 * theta - 3 * zeta)
            )
            Bmax_alt = np.zeros(grid.num_rho)
            Bmin_alt = np.zeros(grid.num_rho)
            for j in range(grid.num_rho):
                mask = grid.inverse_rho_idx == j
                Bmax_alt[j] = np.max(B[mask])
                Bmin_alt[j] = np.min(B[mask])
            np.testing.assert_allclose(Bmax_alt, grid.compress(surface_max(grid, B)))
            np.testing.assert_allclose(Bmin_alt, grid.compress(surface_min(grid, B)))

    @pytest.mark.unit
    def test_rotation_matrix(self):
        """Test that rotation_matrix works with fwd & rev AD for axis=[0, 0, 0]."""
        dfdx_fwd = jax.jacfwd(rotation_matrix)
        dfdx_rev = jax.jacrev(rotation_matrix)
        x0 = jnp.array([0.0, 0.0, 0.0])

        np.testing.assert_allclose(rotation_matrix(x0), np.eye(3))
        np.testing.assert_allclose(dfdx_fwd(x0), np.zeros((3, 3, 3)))
        np.testing.assert_allclose(dfdx_rev(x0), np.zeros((3, 3, 3)))

    @pytest.mark.unit
    def test_cubic_poly_roots(self):
        """Test vectorized computation of cubic polynomial exact roots."""
        cubic = 4
        poly = np.arange(-60, 60).reshape(cubic, 6, -1)
        poly[0] = np.where(poly[0] == 0, np.ones_like(poly[0]), poly[0])
        poly = poly * np.e * np.pi
        assert np.unique(poly.shape).size == poly.ndim
        constant = np.arange(10)
        assert np.unique(poly.shape + constant.shape).size == poly.ndim + constant.ndim
        roots = cubic_poly_roots(poly, constant, sort=True)
        for j in range(poly.shape[1]):
            for k in range(poly.shape[2]):
                for s in range(constant.size):
                    a, b, c, d = poly[:, j, k]
                    d = d - constant[s]
                    np.testing.assert_allclose(
                        roots[s, j, k],
                        np.sort_complex(np.roots([a, b, c, d])),
                    )

    @pytest.mark.unit
    def test_polyint(self):
        """Test vectorized computation of polynomial primitive."""
        quintic = 6
        poly = np.arange(-90, 90).reshape(quintic, 3, -1) * np.e * np.pi
        assert np.unique(poly.shape).size == poly.ndim
        out = polyint(poly)
        for j in range(poly.shape[1]):
            for k in range(poly.shape[2]):
                np.testing.assert_allclose(out[:, j, k], np.polyint(poly[:, j, k])[:-1])

    @pytest.mark.unit
    def test_polyder(self):
        """Test vectorized computation of polynomial derivative."""
        quintic = 6
        poly = np.arange(-90, 90).reshape(quintic, 3, -1) * np.e * np.pi
        assert np.unique(poly.shape).size == poly.ndim
        out = polyder(poly)
        for j in range(poly.shape[1]):
            for k in range(poly.shape[2]):
                np.testing.assert_allclose(out[:, j, k], np.polyder(poly[:, j, k]))

    @pytest.mark.unit
    def test_polyval(self):
        """Test vectorized computation of polynomial evaluation."""
        quintic = 6
        poly = np.arange(-90, 90).reshape(quintic, 3, -1) * np.e * np.pi
        assert np.unique(poly.shape).size == poly.ndim
        x = np.linspace(0, 20, poly.shape[1] * poly.shape[2]).reshape(
            poly.shape[1], poly.shape[2]
        )
        x = np.stack([x, x * 2], axis=-1)
        x = np.stack([x, x * 2, x * 3, x * 4], axis=-1)
        assert np.unique(x.shape).size == x.ndim
        assert poly.shape[1:] == x.shape[: poly.ndim - 1]
        assert np.unique((poly.shape[0],) + x.shape[poly.ndim - 1 :]).size == x.ndim - 1
        val = polyval(x, poly)
        for j in range(poly.shape[1]):
            for k in range(poly.shape[2]):
                np.testing.assert_allclose(val[j, k], np.poly1d(poly[:, j, k])(x[j, k]))

    # TODO: FIXME
    def bounce_point(
        self, eq, lambdas, rho, alpha, max_bounce_points=20, max_field_line=10 * np.pi
    ):
        """Find bounce points."""
        # TODO:
        #     1. make another version of desc.backend.root_scalar
        #        to avoid separate root finding routines in residual and jac
        #        and use previous desc coords as initial guess for next iteration
        #     2. write docstrings and use transforms in api instead of eq
        def residual(zeta, i):
            grid, data = field_line_to_desc_coords(rho, alpha, zeta, eq)
            data = eq.compute(["|B|"], grid=grid, data=data)
            return data["|B|"] - lambdas[i]

        def jac(zeta):
            grid, data = field_line_to_desc_coords(rho, alpha, zeta, eq)
            data = eq.compute(["|B|_z constant rho alpha"], grid=grid, data=data)
            return data["|B|_z constant rho alpha"]

        # Compute |B| - lambda on a dense grid.
        # For every field line, find the roots of this linear spline.
        # These estimates for the true roots will serve as an initial guess, and
        # let us form a boundary mesh around root estimates to limit search domain
        # of the root finding algorithms.
        zeta = np.linspace(0, max_field_line, 3 * max_bounce_points)
        grid, data = field_line_to_desc_coords(rho, alpha, zeta, eq)
        data = eq.compute(["|B|"], grid=grid, data=data)
        B_norm = data["|B|"].reshape(
            alpha.size, rho.size, -1
        )  # constant field line chunks

        boundary_lt = np.zeros((lambdas.size, max_bounce_points, alpha.size, rho.size))
        boundary_rt = np.zeros((lambdas.size, max_bounce_points, alpha.size, rho.size))
        guess = np.zeros((lambdas.size, max_bounce_points, alpha.size, rho.size))
        # todo: scan over this
        for i in range(lambdas.size):
            for j in range(alpha.size):
                for k in range(rho.size):
                    # indices of zeta values observed prior to sign change
                    idx = np.nonzero(np.diff(np.sign(B_norm[j, k] - lambdas[i])))[0]
                    guess[i, :, j, k] = grid.nodes[idx, 2]
                    boundary_lt[i, :, j, k] = np.append(zeta[0], guess[:-1])
                    boundary_rt[i, :, j, k] = np.append(guess[1:], zeta[-1])
        guess = guess.reshape(lambdas.size, max_bounce_points, alpha.size * rho.size)
        boundary_lt = boundary_lt.reshape(
            lambdas.size, max_bounce_points, alpha.size * rho.size
        )
        boundary_rt = boundary_rt.reshape(
            lambdas.size, max_bounce_points, alpha.size * rho.size
        )

        def body_lambdas(i, out):
            def body_roots(j, out_i):
                def fixup(z):
                    return np.clip(z, boundary_lt[i, j], boundary_rt[i, j])

                # todo: call vmap to vectorize on guess[i, j] so that we solve
                #  guess[i, j].size independent root finding problems
                root = root_scalar(residual, guess[i, j], jac=jac, args=i, fixup=fixup)
                out_i = put(out_i, j, root)
                return out_i

            out = put(out, i, fori_loop(0, max_bounce_points, body_roots, out[i]))
            return out

        bounce_points = np.zeros(
            shape=(lambdas.size, alpha.size, rho.size, max_bounce_points)
        )
        bounce_points = fori_loop(0, lambdas.size, body_lambdas, bounce_points)
        return bounce_points
