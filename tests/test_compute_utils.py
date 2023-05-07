"""Tests for surface averaging etc."""

import numpy as np
import pytest

from desc.compute.utils import (
    _get_grid_surface,
    compress,
    expand,
    surface_averages,
    surface_integrals,
    surface_max,
    surface_min,
)
from desc.examples import get
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid


def benchmark_surface_integrals(grid, q=np.array([1.0]), surface_label="rho"):
    """Intuitive implementation of surface_integrals function in compute.utils.

    Compute the surface integral of a quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
    surface_label : str
        The surface label of rho, theta, or zeta to compute the integration over.

    Returns
    -------
    integrals : ndarray
        Surface integrals of q over each surface in grid.

    """
    nodes, _, _, ds, has_endpoint_dupe = _get_grid_surface(grid, surface_label)
    weights = (ds * np.nan_to_num(q).T).T

    surfaces = {}
    # collect node indices for each surface_label surface
    for grid_column_idx, surface_label_value in enumerate(nodes):
        surfaces.setdefault(surface_label_value, []).append(grid_column_idx)
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
            # enforce r is a surface function via compression
            r = compress(
                grid, np.random.random_sample(size=grid.num_nodes), surface_label
            )
            s = compress(grid, expand(grid, r, surface_label), surface_label)
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
            integrals = surface_integrals(grid, q, surface_label)
            assert integrals.shape == q.shape, surface_label
            desired = benchmark_surface_integrals(grid, q, surface_label)
            np.testing.assert_allclose(
                compress(grid, integrals, surface_label),
                desired,
                atol=1e-16,
                err_msg=surface_label,
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
    def test_surface_averages_vectors(self):
        """Test surface averages of vector-valued integrands.

        In particular, tests surface averages of a function-valued integrand.
        """

        def test(surface_label, grid):
            g_size = grid.num_nodes  # not a choice; required
            # arbitrary choice, but 1 != f_size != g_size is better to test
            f_size = g_size // 3
            g = np.cos(np.arange(g_size)) ** 2
            f = np.sin(np.arange(f_size)) ** 2
            q = np.outer(g, f)
            sqrt_g = np.arange(g_size).astype(float)

            averages = surface_averages(grid, q, sqrt_g, surface_label)
            assert averages.shape == q.shape == (g_size, f_size), surface_label

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
    def test_surface_averages_vector_functions(self):
        """Test surface averages of vector-valued, function-valued integrands."""

        def test(surface_label, grid):
            g_size = grid.num_nodes  # not a choice; required
            # arbitrary choice, but v_size != f_size != g_size is better to test
            f_size = g_size // 3
            # arbitrary choice, but f_size != v_size != g_size is better to test
            v_size = g_size // 8
            g = np.cos(np.arange(g_size)) ** 2
            fv = np.sin(np.arange(f_size * v_size).reshape(f_size, v_size)) ** 2
            q = np.einsum("g,fv->gfv", g, fv)
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
    def test_surface_area_unweighted(self):
        """Test that surface_integrals(ds) is 4pi^2 for rho, 2pi for theta, zeta.

        This test should ensure that surfaces have the correct area on grids
        constructed by specifying L, M, N and by specifying an array of nodes.
        Each test should also be done on grids with duplicate nodes
        (e.g. endpoint=True) and grids with symmetry.
        """

        def test(surface_label, grid):
            areas = surface_integrals(grid, surface_label=surface_label)
            correct_area = 4 * np.pi**2 if surface_label == "rho" else 2 * np.pi
            np.testing.assert_allclose(areas, correct_area, err_msg=surface_label)

        rho = np.linspace(1, 0, L)[::-1]
        theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
        theta_endpoint = np.linspace(0, 2 * np.pi, M, endpoint=True)
        zeta = np.linspace(0, 2 * np.pi / NFP, N, endpoint=False)
        zeta_endpoint = np.linspace(0, 2 * np.pi / NFP, N, endpoint=True)

        lg = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=False, endpoint=False)
        lg_sym = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=True, endpoint=False)
        lg_endpoint = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=False, endpoint=True)
        lg_sym_endpoint = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=True, endpoint=True)

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
    def test_surface_area_weighted(self):
        """Test that rho surface integral(dt*dz*sqrt(g)) are monotonic wrt rho."""
        eq = get("W7-X")
        grid = ConcentricGrid(L=L, M=M, N=N, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute("sqrt(g)", grid=grid)
        areas = compress(grid, surface_integrals(grid, data["sqrt(g)"]))
        np.testing.assert_allclose(areas, np.sort(areas))

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
        data = eq.compute("|B|_t", grid=grid)
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
        data = eq.compute("|B|", grid=grid)

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
            np.testing.assert_allclose(Bmax_alt, surface_max(grid, B))
            np.testing.assert_allclose(Bmin_alt, surface_min(grid, B))
