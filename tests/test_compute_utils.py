"""Tests for surface averaging etc."""

import numpy as np
import pytest

from desc.compute.utils import (
    _get_grid_surface,
    compress,
    expand,
    line_integrals,
    surface_averages,
    surface_integrals,
    surface_integrals_transform,
    surface_max,
    surface_min,
)
from desc.examples import get
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid


def benchmark_surface_integrals(grid, q=np.array([1.0]), surface_label="rho"):
    """Compute a surface integral for each surface in the grid.

    Notes
    -----
        It is assumed that the integration surface has area 4π^2 when the
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
    integrals = []
    for _, surface_idx in sorted(surfaces.items()):
        integrals += [weights[surface_idx].sum(axis=0)]
    if has_endpoint_dupe:
        integrals[0] += integrals[-1]
        integrals[-1] = integrals[0]
    return np.asarray(integrals)


# arbitrary choice
L = 6
M = 6
N = 3
NFP = 5


class TestComputeUtils:
    """Tests for grid operations, surface averages etc."""

    @pytest.mark.unit
    def test_compress_expand_inverse_op(self):
        """Test that compress & expand are inverse operations for surface functions.

        Each test should be done on different types of grids
        (e.g. LinearGrid, ConcentricGrid) and grids with duplicate nodes
        (e.g. endpoint=True).
        """

        def test(surface_label, grid):
            r = np.random.random_sample(
                size={
                    "rho": grid.num_rho,
                    "theta": grid.num_theta,
                    "zeta": grid.num_zeta,
                }[surface_label]
            )
            expanded = expand(grid, r, surface_label)
            assert expanded.size == grid.num_nodes
            s = compress(grid, expanded, surface_label)
            np.testing.assert_allclose(r, s, err_msg=surface_label)

        lg_endpoint = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=True, endpoint=True)
        cg_sym = ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=True)

        test("rho", lg_endpoint)
        test("theta", lg_endpoint)
        test("zeta", lg_endpoint)
        test("rho", cg_sym)
        test("theta", cg_sym)
        test("zeta", cg_sym)

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
            assert (
                integrals.size
                == {
                    "rho": grid.num_rho,
                    "theta": grid.num_theta,
                    "zeta": grid.num_zeta,
                }[surface_label]
            )
            desired = benchmark_surface_integrals(grid, q, surface_label)
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
            assert integrals.shape == (
                {
                    "rho": grid.num_rho,
                    "theta": grid.num_theta,
                    "zeta": grid.num_zeta,
                }[surface_label],
                grid.num_nodes,
            ), surface_label

            desired = benchmark_surface_integrals(grid, q, surface_label)
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
                benchmark_surface_integrals(grid, (sqrt_g * q.T).T, surface_label).T
                / benchmark_surface_integrals(grid, sqrt_g, surface_label)
            ).T
            np.testing.assert_allclose(
                compress(grid, averages, surface_label), desired, err_msg=surface_label
            )

        cg = ConcentricGrid(L=L, M=M, N=N, sym=True, NFP=NFP)
        lg = LinearGrid(L=L, M=M, N=N, sym=True, NFP=NFP, endpoint=True)
        test("rho", cg)
        test("theta", lg)
        test("zeta", cg)

    @pytest.mark.unit
    def test_surface_area(self):
        """Test that surface_integrals(ds) is 4pi^2 for rho, 2pi for theta, zeta.

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
        """Test that line_integrals(dl) is 1 for rho, 2pi for theta, zeta.

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
        """Test that surface averages of flux functions are identity operations."""
        eq = get("W7-X")
        grid = ConcentricGrid(L=L, M=M, N=N, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["p", "sqrt(g)"], grid=grid)
        pressure_average = surface_averages(grid, data["p"], data["sqrt(g)"])
        np.testing.assert_allclose(data["p"], pressure_average)

    @pytest.mark.unit
    def test_surface_averages_homomorphism(self):
        """Test that surface averages of flux functions are additive homomorphisms.

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
    def test_surface_averages_shortcut(self):
        """Test that surface_averages on single rho surface matches mean() shortcut."""
        eq = get("W7-X")
        rho = np.array((1 - 1e-4) * np.random.default_rng().random() + 1e-4)
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
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
    def test_min_max(self):
        """Test the surface_min and surface_max functions."""
        for grid_type in [LinearGrid, QuadratureGrid, ConcentricGrid]:
            grid = grid_type(L=3, M=4, N=5, NFP=3)
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
                Bmax_alt[j] = np.max(B[grid.inverse_rho_idx == j])
                Bmin_alt[j] = np.min(B[grid.inverse_rho_idx == j])
            np.testing.assert_allclose(Bmax_alt, compress(grid, surface_max(grid, B)))
            np.testing.assert_allclose(Bmin_alt, compress(grid, surface_min(grid, B)))
