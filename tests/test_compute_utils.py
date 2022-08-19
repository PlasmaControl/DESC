import numpy as np

import desc.io
from desc.grid import ConcentricGrid, LinearGrid
from desc.compute.utils import (
    _get_grid_surface,
    compress,
    expand,
    surface_averages,
    surface_integrals,
)


def test_grid(linear=False):
    # Good to have enough resolution in L, M, N, so that this
    # grid's features are not unique to low resolution.
    L, M, N, NFP, sym = 5, 11, 9, 13, True
    return (
        LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=sym)
        if linear
        else ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=sym)
    )


def benchmark_surface_integrals(grid, q=np.array([1]), surface_label="rho"):
    """Intuitive implementation of bulk surface integral function in compute.utils.

    Compute the surface integral of a quantity for all surfaces in the grid.

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
    nodes, _, ds, _ = _get_grid_surface(grid, surface_label)
    weights = np.asarray(ds * q)

    surfaces = dict()
    # collect node indices for each surface_label surface
    for grid_column_idx, surface_label_value in enumerate(nodes):
        surfaces.setdefault(surface_label_value, list()).append(grid_column_idx)
    # integration over non-contiguous elements
    integrals = list()
    for _, surface_idx in sorted(surfaces.items()):
        integrals.append(weights[surface_idx].sum())
    return np.asarray(integrals)


class TestComputeUtils:
    def test_compress_expand_inverse_op(self):
        """Test that compress() and expand() are inverse operations for surface functions."""

        def test(surface_label):
            grid = test_grid()
            # enforce r is a surface function via compression
            r = compress(
                grid, np.random.random_sample(size=grid.num_nodes), surface_label
            )
            s = compress(grid, expand(grid, r, surface_label), surface_label)
            np.testing.assert_allclose(r, s)

        test("rho")
        test("theta")
        test("zeta")

    def test_surface_integrals(self):
        """Test the bulk surface averaging against a more intuitive implementation."""

        def test(surface_label):
            grid = test_grid()
            q = np.random.random_sample(size=grid.num_nodes)
            integrals_1 = benchmark_surface_integrals(grid, q, surface_label)
            integrals_2 = compress(
                grid, surface_integrals(grid, q, surface_label), surface_label
            )
            np.testing.assert_allclose(integrals_1, integrals_2)

        test("rho")
        test("theta")
        test("zeta")

    def test_surface_area_unweighted(self):
        """Test the surface integral(ds) is 4pi^2, 2pi for rho, zeta surfaces."""

        def test(surface_label):
            grid = test_grid(linear=surface_label == "theta")
            areas = surface_integrals(grid, surface_label=surface_label)
            correct_area = 4 * np.pi ** 2 if surface_label == "rho" else 2 * np.pi
            np.testing.assert_allclose(areas, correct_area)

        test("rho")
        test("theta")
        test("zeta")

    def test_surface_area_weighted(self, DSHAPE, HELIOTRON):
        """Test that rho surface integral(dt*dz*sqrt(g)) are monotonic wrt rho."""

        def test(stellarator):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            grid = test_grid()
            sqrt_g = np.abs(eq.compute("sqrt(g)", grid=grid)["sqrt(g)"])
            areas = compress(grid, surface_integrals(grid, sqrt_g))
            np.testing.assert_allclose(areas, np.sort(areas))

        test(DSHAPE)
        test(HELIOTRON)

    def test_surface_averages_identity_op(self, DSHAPE, HELIOTRON):
        """Test that surface averages of flux functions are identity operations."""

        def test(stellarator):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            grid = test_grid()
            data = eq.compute("p", grid=grid)
            data = eq.compute("sqrt(g)", grid=grid, data=data)
            pressure_average = surface_averages(grid, data["p"], data["sqrt(g)"])
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
            grid = test_grid()
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

        test(DSHAPE)
        test(HELIOTRON)
