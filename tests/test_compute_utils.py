import pytest
import numpy as np
from desc.compute.utils import (
    compress,
    enclosed_volumes,
    surface_averages,
    surface_integrals,
    _get_proper_surface,
)
from desc.grid import ConcentricGrid, LinearGrid
from desc.equilibrium import Equilibrium
import desc.io


def get_desc_eq(name):
    """
    Parameters
    ----------
    name : str
        Name of the equilibrium.

    Returns
    -------
    Equilibrium
        DESC equilibrium.
    """
    return desc.io.load("examples/DESC/" + name + "_output.h5")[-1]


def random_grid(NFP=None, sym=None):
    """
    NFP: int
        Number of field periods.
    sym: bool
        Stellarator symmetry.

    Returns
    -------
    ConcentricGrid
        Randomized grid.
    """
    rng = np.random.default_rng()
    L = rng.integers(low=1, high=20)
    M = rng.integers(low=1, high=20)
    N = rng.integers(low=1, high=20)
    if NFP is None:
        NFP = rng.integers(low=1, high=20)
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
    integrands = np.asarray(integrands)  # to use fancy indexing

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
            np.testing.assert_allclose(
                integrals_1, integrals_2, err_msg=surface_label + ", fail"
            )

        test("rho")
        test("zeta")

    def test_surface_area_unweighted(self):
        """Test the surface area without the sqrt(g) factor is computed correctly."""
        grid = random_grid(NFP=1, sym=False)

        def test(surface_label, area):
            areas_1 = benchmark(grid, surface_label=surface_label)[0]
            areas_2 = surface_integrals(grid, surface_label=surface_label)
            np.testing.assert_allclose(
                areas_1, areas_2, err_msg=surface_label + ", fail"
            )
            np.testing.assert_allclose(
                areas_2, area, err_msg=surface_label + ", unweighted area fail"
            )

        test("rho", 4 * np.pi ** 2)
        test("zeta", 2 * np.pi)

    def test_surface_area_weighted(self):
        """Test the rho surface area with the sqrt(g) factor is monotonic."""
        grid = random_grid()
        eq = get_desc_eq("HELIOTRON")
        sqrtg = np.abs(eq.compute("sqrt(g)", grid=grid)["sqrt(g)"])

        areas_1 = benchmark(grid, sqrtg)[0]
        areas_2 = surface_integrals(grid, sqrtg)
        np.testing.assert_allclose(areas_1, areas_2)
        np.testing.assert_allclose(
            areas_2, np.sort(areas_2), err_msg="weighted area not monotonic"
        )

    def test_expand(self):
        """Test the expand function."""
        grid = random_grid()
        integrands = np.random.random_sample(size=len(grid.nodes))

        def test(surface_label):
            integrals = surface_integrals(grid, integrands, surface_label)
            integrals_match_grid = surface_integrals(
                grid, integrands, surface_label, match_grid=True
            )
            np.testing.assert_allclose(
                integrals,
                compress(grid, integrals_match_grid, surface_label),
                err_msg=surface_label + ", fail",
            )
            surface_indices = benchmark(grid, integrands, surface_label)[1].values()
            for i, indices in enumerate(surface_indices):
                for index in indices:
                    np.testing.assert_allclose(
                        integrals[i],
                        integrals_match_grid[index],
                        err_msg=surface_label + ", fail",
                    )

        test("rho")
        test("zeta")

    def test_surface_averages_identity_op(self):
        """
        Test that all surface averages of a flux surface function are identity operations.
        Relies on correctness of the surface_integrals and _expand functions.
        """
        grid = random_grid()
        eq = get_desc_eq("HELIOTRON")
        data = eq.compute("p", grid=grid)
        data = eq.compute("sqrt(g)", grid=grid, data=data)
        pressure_average = surface_averages(
            grid, data["p"], data["sqrt(g)"], match_grid=True
        )
        np.testing.assert_allclose(data["p"], pressure_average)

    def test_surface_averages_homomorphism(self):
        """
        Test that all surface averages of a flux surface function are additive homomorphisms.
        Meaning average(a + b) = average(a) + average(b).
        """
        grid = random_grid()
        eq = get_desc_eq("HELIOTRON")
        data = eq.compute("|B|_t", grid=grid)
        a = surface_averages(grid, data["|B|"], data["sqrt(g)"])
        b = surface_averages(grid, data["|B|_t"], data["sqrt(g)"])
        a_plus_b = surface_averages(grid, data["|B|"] + data["|B|_t"], data["sqrt(g)"])
        np.testing.assert_allclose(a_plus_b, a + b)

    def test_surface_averages_shortcut(self):
        """
        Test that surface average on LinearGrid with single rho surface matches mean() shortcut.
        Relies on correctness of surface_integrals.
        """
        eq = get_desc_eq("HELIOTRON")
        rho = (1 - 1e-4) * np.random.default_rng().random() + 1e-4  # uniform [1e-4, 1)
        grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
            rho=np.atleast_1d(rho),
        )
        data = eq.compute("|B|", grid=grid)

        np.testing.assert_allclose(
            surface_averages(grid, data["|B|"], data["sqrt(g)"]),
            np.mean(data["sqrt(g)"] * data["|B|"]) / np.mean(data["sqrt(g)"]),
            err_msg="sqrt(g) fail",
        )
        np.testing.assert_allclose(
            surface_averages(grid, data["|B|"]),
            np.mean(data["|B|"]),
            err_msg="no sqrt(g) fail",
        )

    def test_enclosed_volumes(self):
        """Test that the volume enclosed by flux surfaces matches the known analytic formula."""
        torus = Equilibrium()
        rho = np.linspace(1 / 128, 1, 128)
        grid = LinearGrid(
            M=torus.M_grid,
            N=torus.N_grid,
            NFP=1,
            sym=False,
            rho=rho,
        )
        data = torus.compute("sqrt(g)_rr", grid=grid)
        volume = enclosed_volumes(grid, data)
        volume_r = enclosed_volumes(grid, data, dr=1)
        volume_rr = enclosed_volumes(grid, data, dr=2)

        np.testing.assert_allclose(volume, 20 * (np.pi * rho) ** 2, rtol=1e-2)
        np.testing.assert_allclose(volume_r, 40 * np.pi ** 2 * rho, rtol=1e-2)
        np.testing.assert_allclose(volume_rr, 40 * np.pi ** 2, rtol=1e-2)

    def test_total_volume(self):
        """Test that the volume enclosed by flux surfaces matches the device volume."""

        def test(eq):
            # TODO: Good test for the sym/NFP bug in grid.py.
            #   If the grid is over multiple rho values, sym and NFP must be forced to False, 1.
            #   I have fixed the bug for single surface grids to be able to use any sym / NFP.
            #   Either of these grids will pass this test.
            # grid = LinearGrid(
            #     M=eq.M_grid,
            #     N=eq.N_grid,
            #     NFP=eq.NFP,
            #     sym=eq.sym,
            #     rho=np.atleast_1d(1),
            # )
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=1,
                sym=False,
            )
            data = eq.compute("e_theta", grid=grid)
            data = eq.compute("e_zeta", grid=grid, data=data)
            total_volume = enclosed_volumes(grid, data)[-1]
            np.testing.assert_allclose(
                total_volume, eq.compute("V", grid=grid)["V"], rtol=1e-2
            )

        test(Equilibrium())
        test(get_desc_eq("DSHAPE"))
        test(get_desc_eq("HELIOTRON"))
