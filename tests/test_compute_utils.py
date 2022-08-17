import numpy as np

import desc.io
from desc.grid import ConcentricGrid, LinearGrid
from desc.compute.utils import (
    _get_grid_surface,
    compress,
    surface_averages,
    surface_integrals,
)


def random_grid(linear=False):
    rng = np.random.default_rng()
    L = rng.integers(low=1, high=4)
    M = rng.integers(low=7, high=20)
    N = rng.integers(low=1, high=20)
    NFP = rng.integers(low=1, high=20)
    sym = True if rng.integers(2) > 0 else False
    return (
        LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=sym)
        if linear
        else ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=sym)
    )


def benchmark_integrals(grid, q=1, surface_label="rho"):
    """Intuitive implementation of bulk surface integral function in compute.utils.

    Computes surface integrals of the specified quantity for all surfaces in the grid.

    Notes
    -----
        See D'haeseleer flux coordinates eq. 4.9.11 for more details.
        LinearGrid will have better accuracy than QuadratureGrid for a theta surface.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.

    Returns
    -------
    integrals : ndarray
        Surface integrals of q over each surface in grid.
    """
    q = np.asarray(q)
    nodes, _, ds = _get_grid_surface(grid, surface_label)

    surfaces = dict()
    # collect collocation node indices for each surface_label surface
    for grid_column_idx, surface_label_value in enumerate(nodes):
        surfaces.setdefault(surface_label_value, list()).append(grid_column_idx)
    # integration over non-contiguous elements
    integrals = list()
    for _, surface_idx in sorted(surfaces.items()):
        integrals.append((ds * q)[surface_idx].sum())
    return np.asarray(integrals)


class TestComputeUtils:
    def test_surface_integrals(self):
        """Test the bulk surface averaging against a more intuitive implementation."""

        def test(surface_label):
            grid = random_grid()
            q = np.random.random_sample(size=grid.num_nodes)
            integrals_1 = benchmark_integrals(grid, q, surface_label)
            integrals_2 = surface_integrals(grid, q, surface_label)
            np.testing.assert_allclose(integrals_1, integrals_2)

        test("rho")
        test("theta")
        test("zeta")

    def test_expand(self):
        """Test the expand function."""

        def test(surface_label):
            grid = random_grid()
            q = np.random.random_sample(size=grid.num_nodes)
            integrals = surface_integrals(grid, q, surface_label)
            integrals_expand = surface_integrals(
                grid, q, surface_label, match_grid=True
            )

            nodes, _, _ = _get_grid_surface(grid, surface_label)
            _, inverse = np.unique(nodes, return_inverse=True)
            np.testing.assert_allclose(integrals[inverse], integrals_expand)

        test("rho")
        test("theta")
        test("zeta")

    def test_compress(self):
        """Test the compress function."""

        def test(surface_label):
            grid = random_grid()
            q = np.random.random_sample(size=grid.num_nodes)
            integrals = surface_integrals(grid, q, surface_label)
            integrals_expand = surface_integrals(
                grid, q, surface_label, match_grid=True
            )
            np.testing.assert_allclose(
                integrals, compress(grid, integrals_expand, surface_label)
            )

        test("rho")
        test("theta")
        test("zeta")

    def test_surface_area_unweighted(self):
        """Test the surface integral(ds) is 4pi^2, 2pi for rho, zeta surfaces."""

        def test(surface_label, area, linear=False):
            grid = random_grid(linear)
            areas = surface_integrals(grid, surface_label=surface_label)
            np.testing.assert_allclose(areas, area)

        test("rho", 4 * np.pi ** 2)
        test("theta", 2 * np.pi, True)
        test("zeta", 2 * np.pi)

    def test_surface_area_weighted(self, DSHAPE, HELIOTRON):
        """Test that rho surface integral(dt*dz*sqrt(g)) are monotonic wrt rho."""

        def test(stellarator):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            grid = random_grid()
            sqrtg = np.abs(eq.compute("sqrt(g)", grid=grid)["sqrt(g)"])
            areas = surface_integrals(grid, sqrtg)
            np.testing.assert_allclose(areas, np.sort(areas))

        test(DSHAPE)
        test(HELIOTRON)

    def test_total_volume(self, DSHAPE, HELIOTRON):
        """Test that the volume enclosed by the LCFS is equal to the total volume."""

        def test(stellarator):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
            )
            lcfs_volume = eq.compute("V enclosed", grid=grid)["V enclosed"][-1]
            total_volume = eq.compute("V", grid=grid)["V"]
            np.testing.assert_allclose(lcfs_volume, total_volume, rtol=1e-2)

        test(DSHAPE)
        test(HELIOTRON)

    def test_surface_averages_identity_op(self, DSHAPE, HELIOTRON):
        """Test that surface averages of flux functions are identity operations."""

        def test(stellarator):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            grid = random_grid()
            data = eq.compute("p", grid=grid)
            data = eq.compute("sqrt(g)", grid=grid, data=data)
            pressure_average = surface_averages(
                grid, data["p"], data["sqrt(g)"], match_grid=True
            )
            np.testing.assert_allclose(data["p"], pressure_average)

        test(DSHAPE)
        test(HELIOTRON)

    def test_surface_averages_homomorphism(self, DSHAPE, HELIOTRON):
        """
        Test that surface averages of flux surface functions are additive homomorphisms.
        Meaning average(a + b) = average(a) + average(b).
        """

        def test(stellarator):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            grid = random_grid()
            data = eq.compute("|B|_t", grid=grid)
            a = surface_averages(grid, data["|B|"], data["sqrt(g)"])
            b = surface_averages(grid, data["|B|_t"], data["sqrt(g)"])
            a_plus_b = surface_averages(
                grid, data["|B|"] + data["|B|_t"], data["sqrt(g)"]
            )
            np.testing.assert_allclose(a_plus_b, a + b)

        test(DSHAPE)
        test(HELIOTRON)

    def test_surface_averages_shortcut(self, DSHAPE, HELIOTRON):
        """Test that surface_averages on single rho surface matches mean() shortcut."""

        def test(stellarator):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            rho = (1 - 1e-4) * np.random.default_rng().random() + 1e-4
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
                err_msg="average with sqrt(g) fail",
            )
            np.testing.assert_allclose(
                surface_averages(grid, data["|B|"]),
                np.mean(data["|B|"]),
                err_msg="average without sqrt(g) fail",
            )

        test(DSHAPE)
        test(HELIOTRON)
