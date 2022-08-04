import pytest
import matplotlib.pyplot as plt
import numpy as np
from desc.compute.utils import (
    compress,
    surface_averages,
    surface_integrals,
    _get_proper_surface,
)
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
import desc.io


def plot_compare(title, x, y, label_1, x2, y2, label_2):
    """Displays log plot of two quantities on same axis."""
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label_1)
    ax.scatter(x, y)
    ax.plot(x2, y2, label=label_2)
    ax.scatter(x2, y2, s=10)
    ax.set(
        yscale="log",
        xlabel=r"$\rho$",
        title=title,
        facecolor="white",
    )
    ax.grid()
    fig.legend()
    plt.savefig(title)


def load_eq(name):
    """
    Parameters
    ----------
    name : str
        Name of the equilibrium.

    Returns
    -------
    eq : Equilibrium
        The equilibrium.
    """
    try:
        eq = desc.io.load("examples/DESC/" + name + "_output.h5")[-1]
    except FileNotFoundError:
        assert False, "Could not locate equilibrium output file."
    return eq


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
    integrands = np.asarray(integrands)  # to use fancy indexing

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

    def test_surface_area_unweighted(self):
        """Test the surface area without the sqrt(g) factor is computed correctly."""
        grid = random_grid(NFP=1, sym=False)  # for the grid bugs

        def test(surface_label, area):
            areas_1 = benchmark(grid, surface_label=surface_label)[0]
            areas_2 = surface_integrals(grid, surface_label=surface_label)
            assert np.allclose(areas_1, areas_2), surface_label + ", fail"
            assert np.allclose(areas_2, area), surface_label + ", unweighted area fail"

        test("rho", 4 * np.pi ** 2)
        test("zeta", 2 * np.pi)

    def test_surface_area_weighted(self):
        """Test the rho surface area with the sqrt(g) factor is monotonic."""
        eq = load_eq("HELIOTRON")
        grid = ConcentricGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            node_pattern=eq.node_pattern,
        )
        sqrtg = np.abs(eq.compute("sqrt(g)", grid=grid)["sqrt(g)"])

        areas_1 = benchmark(grid, sqrtg)[0]
        areas_2 = surface_integrals(grid, sqrtg)
        assert np.allclose(areas_1, areas_2)
        assert np.allclose(areas_2, np.sort(areas_2)), "weighted area not monotonic"

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
        eq = load_eq("HELIOTRON")
        grid = ConcentricGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            node_pattern=eq.node_pattern,
        )
        data = eq.compute("p", grid=grid)
        data = eq.compute("sqrt(g)", grid=grid, data=data)
        pressure_average = surface_averages(
            grid, data["p"], data["sqrt(g)"], match_grid=True
        )
        assert np.allclose(data["p"], pressure_average)

    def test_surface_averages_homomorphism(self):
        """
        Test that all surface averages of a flux surface function are additive homomorphisms.
        Meaning average(a + b) = average(a) + average(b).
        """
        eq = load_eq("HELIOTRON")
        grid = ConcentricGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            node_pattern=eq.node_pattern,
        )
        data = eq.compute("|B|_t", grid=grid)
        a = surface_averages(grid, data["|B|"], data["sqrt(g)"])
        b = surface_averages(grid, data["|B|_t"], data["sqrt(g)"])
        a_plus_b = surface_averages(grid, data["|B|"] + data["|B|_t"], data["sqrt(g)"])
        assert np.allclose(a_plus_b, a + b)

    def test_surface_averages_shortcut(self):
        """
        Test that surface average on LinearGrid with single rho surface matches mean() shortcut.
        Relies on correctness of surface_integrals.
        """
        eq = load_eq("HELIOTRON")
        rho = (1 - 1e-4) * np.random.default_rng().random() + 1e-4  # uniform [1e-4, 1)
        grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            rho=np.atleast_1d(rho),
        )
        data = eq.compute("|B|", grid=grid)

        assert np.allclose(
            surface_averages(grid, data["|B|"], data["sqrt(g)"]),
            np.mean(data["sqrt(g)"] * data["|B|"]) / np.mean(data["sqrt(g)"]),
        ), "sqrt(g) fail"
        assert np.allclose(
            surface_averages(grid, data["|B|"]), np.mean(data["|B|"])
        ), "no sqrt(g) fail"

    def test_grid_diffs(self):
        """
        Test that surface average on Concentric grids and Quadrature grids are similar.
        Displays plot.
        """

        def iota_zero_current(eq, grid):
            data = eq.compute("|B|", grid=grid)  # puts most quantities in data
            num = surface_averages(
                grid,
                data["psi_r"]
                / data["sqrt(g)"]
                * (
                    data["g_tt"] * data["lambda_z"]
                    - data["g_tz"] * (1 + data["lambda_t"])
                ),
            )
            den = surface_averages(grid, data["psi_r"] / data["sqrt(g)"] * data["g_tt"])
            return np.abs(num / den)

        eq = load_eq("HELIOTRON")
        conc_grid = ConcentricGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=False,
            node_pattern=eq.node_pattern,
        )
        quad_grid = QuadratureGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
        )
        plot_compare(
            "magnitude of iota zero current",
            conc_grid.nodes[conc_grid.unique_rho_indices, 0],
            iota_zero_current(eq, conc_grid),
            "conc",
            quad_grid.nodes[quad_grid.unique_rho_indices, 0],
            iota_zero_current(eq, quad_grid),
            "quad",
        )
