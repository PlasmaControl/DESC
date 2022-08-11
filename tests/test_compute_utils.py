import pytest
import numpy as np
from desc.compute.utils import (
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
    return desc.io.load("desc/examples/" + name + "_output.h5")[-1]


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


def benchmark(grid, q=1, surface_label="rho"):
    """
    More intuitive implementation with loops of bulk surface integral function in compute.utils.

    Computes the surface integral of the specified quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
        Should not include the differential elements (dtheta * dzeta for rho surface).
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
        Defaults to the flux surface label rho.

    Returns
    -------
    integrals : ndarray
        Surface integrals of q over each surface in grid.
    """
    surface_label_nodes, unique_indices, ds = _get_proper_surface(grid, surface_label)
    integrals = np.empty(len(unique_indices))
    q = np.asarray(q)

    surfaces = dict()
    # collect collocation node indices for each surface_label surface
    for index_in_grid_column, surface_label_value in enumerate(surface_label_nodes):
        surfaces.setdefault(surface_label_value, list()).append(index_in_grid_column)
    # integration over non-contiguous elements
    for i, e in enumerate(sorted(surfaces.items())):
        _, surface_indices = e
        integrals[i] = (ds * q)[surface_indices].sum()
    return integrals


class TestComputeUtils:
    def test_surface_integrals(self):
        """Test the bulk surface averaging against a more intuitive implementation."""

        def test(surface_label):
            grid = random_grid()
            q = np.random.random_sample(size=grid.num_nodes)
            integrals_1 = benchmark(grid, q, surface_label)
            integrals_2 = surface_integrals(grid, q, surface_label)
            np.testing.assert_allclose(integrals_1, integrals_2)

        test("rho")
        test("theta")
        test("zeta")

    def test_expand(self):
        """Test the expand function."""

        def test(surface_label, linear=False):
            grid = random_grid(linear)
            q = np.random.random_sample(size=grid.num_nodes)
            integrals = surface_integrals(grid, q, surface_label)
            integrals_expand = surface_integrals(
                grid, q, surface_label, match_grid=True
            )

            surface_label_nodes, _, _ = _get_proper_surface(grid, surface_label)
            _, inverse = np.unique(surface_label_nodes, return_inverse=True)
            np.testing.assert_allclose(integrals[inverse], integrals_expand)

        test("rho")
        test("theta", linear=True)
        test("zeta")

    def test_surface_area_unweighted(self):
        """Test the surface integral(ds) is 4pi^2, 2pi for rho, zeta surfaces."""

        def test(surface_label, area, linear=False):
            grid = random_grid(linear)
            areas = surface_integrals(grid, surface_label=surface_label)
            np.testing.assert_allclose(areas, area)

        test("rho", 4 * np.pi ** 2)
        test("theta", 2 * np.pi, linear=True)
        test("zeta", 2 * np.pi)

    def test_surface_area_weighted(self):
        """Test that rho surface integral(dt * dz * sqrt(g)) is monotonic increasing with rho."""

        def test(eq, err_msg):
            grid = random_grid()
            sqrtg = np.abs(eq.compute("sqrt(g)", grid=grid)["sqrt(g)"])
            areas = surface_integrals(grid, sqrtg)
            np.testing.assert_allclose(areas, np.sort(areas), err_msg=err_msg)

        test(get_desc_eq("DSHAPE"), "DSHAPE")
        test(get_desc_eq("HELIOTRON"), "HELIOTRON")

    def test_enclosed_volumes(self):
        """Test that the volume enclosed by flux surfaces matches known analytic formulas."""
        torus = Equilibrium()
        rho = np.linspace(1 / 128, 1, 128)
        grid = LinearGrid(
            M=torus.M_grid,
            N=torus.N_grid,
            NFP=torus.NFP,
            sym=torus.sym,
            rho=rho,
        )
        data = torus.compute("sqrt(g)_rr", grid=grid)
        volume = 20 * (np.pi * rho) ** 2
        volume_r = 40 * np.pi ** 2 * rho
        volume_rr = 40 * np.pi ** 2
        np.testing.assert_allclose(
            enclosed_volumes(grid, data), volume, rtol=1e-2, err_msg="dr=0"
        )
        np.testing.assert_allclose(
            enclosed_volumes(grid, data, dr=1), volume_r, rtol=1e-2, err_msg="dr=1"
        )
        np.testing.assert_allclose(
            enclosed_volumes(grid, data, dr=2), volume_rr, rtol=1e-2, err_msg="dr=2"
        )

    def test_total_volume(self):
        """Test that the volume enclosed by the last enclosed flux surfaces matches the device volume."""

        def test(eq, err_msg):
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
            )
            data = eq.compute("e_theta", grid=grid)
            data = eq.compute("e_zeta", grid=grid, data=data)
            last_flux_surface_volume = enclosed_volumes(grid, data)[-1]
            device_volume = eq.compute("V", grid=grid)["V"]
            np.testing.assert_allclose(
                last_flux_surface_volume, device_volume, rtol=1e-2, err_msg=err_msg
            )

        test(Equilibrium(), "torus")
        test(get_desc_eq("DSHAPE"), "DSHAPE")
        test(get_desc_eq("HELIOTRON"), "HELIOTRON")

    def test_surface_averages_identity_op(self):
        """Test that all surface averages of a flux surface function are identity operations."""

        def test(eq, err_msg):
            grid = random_grid()
            data = eq.compute("p", grid=grid)
            data = eq.compute("sqrt(g)", grid=grid, data=data)
            pressure_average = surface_averages(
                grid, data["p"], data["sqrt(g)"], match_grid=True
            )
            np.testing.assert_allclose(data["p"], pressure_average, err_msg=err_msg)

        test(get_desc_eq("DSHAPE"), "DSHAPE")
        test(get_desc_eq("HELIOTRON"), "HELIOTRON")

    def test_surface_averages_homomorphism(self):
        """
        Test that all surface averages of a flux surface function are additive homomorphisms.
        Meaning average(a + b) = average(a) + average(b).
        """

        def test(eq, err_msg):
            grid = random_grid()
            data = eq.compute("|B|_t", grid=grid)
            a = surface_averages(grid, data["|B|"], data["sqrt(g)"])
            b = surface_averages(grid, data["|B|_t"], data["sqrt(g)"])
            a_plus_b = surface_averages(
                grid, data["|B|"] + data["|B|_t"], data["sqrt(g)"]
            )
            np.testing.assert_allclose(a_plus_b, a + b, err_msg=err_msg)

        test(get_desc_eq("DSHAPE"), "DSHAPE")
        test(get_desc_eq("HELIOTRON"), "HELIOTRON")

    def test_surface_averages_shortcut(self):
        """Test that surface_averages on single rho surface matches mean() shortcut."""

        def test(eq, err_msg):
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
                err_msg=err_msg + " average with sqrt(g) fail",
            )
            np.testing.assert_allclose(
                surface_averages(grid, data["|B|"]),
                np.mean(data["|B|"]),
                err_msg=err_msg + " average without sqrt(g) fail",
            )

        test(get_desc_eq("DSHAPE"), "DSHAPE")
        test(get_desc_eq("HELIOTRON"), "HELIOTRON")
