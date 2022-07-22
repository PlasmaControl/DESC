import pytest
import numpy as np
from desc.compute.utils import surface_averages, surface_integrals, _get_proper_surface
from desc.grid import ConcentricGrid, LinearGrid
import desc.io


def random_grid():
    """
    Returns
    -------
    ConcentricGrid
        Randomized.
    """
    rng = np.random.default_rng()
    L = rng.integers(low=1, high=30)
    M = rng.integers(low=1, high=30)
    N = rng.integers(low=1, high=30)
    NFP = rng.integers(low=1, high=30)
    sym = True if rng.integers(2) > 0 else False
    return ConcentricGrid(L=L, N=N, M=M, NFP=NFP, sym=sym)


def benchmark(grid, integrand, surface_label="rho"):
    """
    More intuitive implementation with loops of bulk surface integral function in compute.utils.

    Computes the surface integral of the specified quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    integrand : ndarray
        Quantity to integrate.
        Should not include the surface differential element ds (dtheta * dzeta for rho surface).
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
        Defaults to the flux surface label rho.

    Returns
    -------
    :rtype: (ndarray, dict)
    integrals : ndarray
        Surface integrals of integrand over each surface in grid.
    surfaces : dict
        Keys are unique surface label values.
        Values are list of indices the key appears at in grid.nodes column.
    """

    surface_label_nodes, unique_indices, upper_bound, ds = _get_proper_surface(
        grid, surface_label
    )
    integrals = np.empty(len(unique_indices))

    surfaces = dict()
    # collect collocation node indices for each rho surface
    for index_in_grid_column, surface_label_value in enumerate(surface_label_nodes):
        surfaces.setdefault(surface_label_value, list()).append(index_in_grid_column)
    # integration over non-contiguous elements
    for i, surface_indices in enumerate(surfaces.values()):
        integrals[i] = (ds * integrand)[surface_indices].sum()
    return integrals, surfaces


class TestComputeUtils:
    def test_surface_integrals(self):
        """Test the bulk surface averaging against the more intuitive implementation with loops."""
        grid = random_grid()
        integrand = np.random.random_sample(size=len(grid.nodes))

        def test(surface_label):
            integrals_1 = benchmark(grid, integrand, surface_label)[0]
            integrals_2 = surface_integrals(
                grid, integrand, surface_label, match_grid=False
            )
            assert np.allclose(integrals_1, integrals_2), surface_label + " fail"

        test("rho")
        test("zeta")

    def test_expand(self):
        """Test the expand function. Relies on correctness of the surface_integrals function."""
        grid = random_grid()
        integrand = np.random.random_sample(size=len(grid.nodes))

        def test(surface_label):
            integrals = surface_integrals(
                grid, integrand, surface_label, match_grid=False
            )
            integrals_match_grid = surface_integrals(
                grid, integrand, surface_label, match_grid=True
            )
            surface_indices = benchmark(grid, integrand, surface_label)[1].values()
            for i, indices in enumerate(surface_indices):
                for index in indices:
                    assert np.allclose(integrals[i], integrals_match_grid[index]), (
                        surface_label + " fail"
                    )

        test("rho")
        test("zeta")

    def test_surface_averages_identity_op(self):
        """
        Test that all surface averages of a flux surface function are identity operations.
        Relies on correctness of the surface_integrals and _expand functions.
        """
        try:
            eq = desc.io.load("examples/DESC/" + "HELIOTRON" + "_output.h5")[-1]
        except FileNotFoundError:
            assert False, "Could not locate equilibrium output file."

        grid = ConcentricGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            node_pattern=eq.node_pattern,
        )
        pressure = eq.compute("p", grid=grid)["p"]
        sqrtg = eq.compute("sqrt(g)", grid=grid)["sqrt(g)"]
        pressure_average = surface_averages(grid, pressure, sqrtg, match_grid=True)
        assert np.allclose(pressure, pressure_average)

    def test_surface_averages_shortcut(self):
        """
        Test that surface average on LinearGrid with single rho surface matches mean() shortcut.
        Relies on correctness of surface_integrals.
        """
        try:
            eq = desc.io.load("examples/DESC/" + "HELIOTRON" + "_output.h5")[-1]
        except FileNotFoundError:
            assert False, "Could not locate equilibrium output file."

        rho = (1 - 1e-3) * np.random.default_rng().random() + 1e-3  # uniform [1e-3, 1)
        grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            rho=rho,
        )
        B = eq.compute("|B|", grid=grid)["|B|"]
        sqrtg = eq.compute("sqrt(g)", grid=grid)["sqrt(g)"]
        B_average = surface_averages(grid, B, sqrtg, match_grid=False)
        # grid has 1 rho value only
        # match_grid=True returns an array of len(grid.nodes) with all elements same value.
        # match_grid=False returns that single value.
        assert np.allclose(B_average, np.mean(sqrtg * B) / np.mean(sqrtg))
