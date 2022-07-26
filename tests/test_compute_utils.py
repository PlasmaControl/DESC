import pytest
import numpy as np
from desc.compute.utils import (
    compress,
    surface_averages,
    surface_integrals,
    _get_proper_surface,
)
from desc.grid import ConcentricGrid, LinearGrid
import desc.io


def random_grid(NFP=None, sym=None):
    """
    NFP: int
        Number of field periods. Random is not specified.
    sym: bool
        Stellarator symmetry. Random is not specified.

    Returns
    -------
    ConcentricGrid
        Randomized grid.
    """
    rng = np.random.default_rng()
    L = rng.integers(low=1, high=30)
    M = rng.integers(low=1, high=30)
    N = rng.integers(low=1, high=30)
    if NFP is None:
        NFP = rng.integers(low=1, high=30)
    if sym is None:
        sym = True if rng.integers(2) > 0 else False
    return ConcentricGrid(L=L, N=N, M=M, NFP=NFP, sym=sym)


def benchmark(grid, integrands=1, surface_label="rho"):
    """
    More intuitive implementation with loops of bulk surface integral function in compute.utils.

    Computes the surface integral of the specified quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    integrands : ndarray
        Quantity to integrate.
        Should not include the differential elements (dtheta * dzeta for rho surface).
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
    # collect collocation node indices for each surface_label surface
    for index_in_grid_column, surface_label_value in enumerate(surface_label_nodes):
        surfaces.setdefault(surface_label_value, list()).append(index_in_grid_column)
    # integration over non-contiguous elements
    for i, surface_indices in enumerate(surfaces.values()):
        integrals[i] = (ds * integrands)[surface_indices].sum()
    return integrals, surfaces


class TestComputeUtils:
    def test_surface_integrals(self):
        """Test the bulk surface averaging against the more intuitive implementation with loops."""
        grid = random_grid()
        integrands = np.random.random_sample(size=len(grid.nodes))

        def test(surface_label):
            integrals_1 = benchmark(grid, integrands, surface_label)[0]
            integrals_2 = surface_integrals(grid, integrands, surface_label)
            assert np.allclose(integrals_1, integrals_2), surface_label + ", fail"

        test("rho")
        test("zeta")

    def test_surface_area(self):
        """Test the surface area is computed correctly."""
        grid = random_grid(NFP=1, sym=False)  # for the grid bugs

        def test(surface_label, max_area):
            areas_1 = benchmark(grid, surface_label=surface_label)[0]
            areas_2 = surface_integrals(grid, surface_label=surface_label)
            assert np.allclose(areas_1, areas_2), surface_label + ", fail"
            assert np.allclose(areas_2, np.sort(areas_2)), (
                surface_label + ", area not increasing"
            )
            assert np.isclose(areas_2[-1], max_area), surface_label + ", max area fail"

        test("rho", 4 * np.pi ** 2)
        test("zeta", 2 * np.pi)

    def test_expand(self):
        """Test the expand function."""
        grid = random_grid()
        integrands = np.random.random_sample(size=len(grid.nodes))

        def test(surface_label):
            integrals = surface_integrals(grid, integrands, surface_label)
            integrals_match_grid = surface_integrals(
                grid, integrands, surface_label, match_grid=True
            )
            assert np.allclose(
                integrals, compress(grid, integrals_match_grid, surface_label)
            ), (surface_label + ", fail")

            surface_indices = benchmark(grid, integrands, surface_label)[1].values()
            for i, indices in enumerate(surface_indices):
                for index in indices:
                    assert np.allclose(integrals[i], integrals_match_grid[index]), (
                        surface_label + ", fail"
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

        rho = (1 - 1e-4) * np.random.default_rng().random() + 1e-4  # uniform [1e-4, 1)
        grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            rho=np.atleast_1d(rho),
        )
        B = eq.compute("|B|", grid=grid)["|B|"]
        sqrtg = eq.compute("sqrt(g)", grid=grid)["sqrt(g)"]

        assert np.allclose(
            surface_averages(grid, B, sqrtg), np.mean(sqrtg * B) / np.mean(sqrtg)
        ), "sqrt(g) fail"
        assert np.allclose(surface_averages(grid, B), np.mean(B)), "no sqrt(g) fail"
