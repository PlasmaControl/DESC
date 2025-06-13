"""Tests Grid classes."""

import numpy as np
import pytest
from scipy import special

from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import (
    ConcentricGrid,
    Grid,
    LinearGrid,
    QuadratureGrid,
    dec_to_cf,
    find_least_rational_surfaces,
    find_most_rational_surfaces,
    most_rational,
)
from desc.profiles import PowerSeriesProfile


class TestGrid:
    """Test for Grid classes."""

    @pytest.mark.unit
    def test_linear_grid(self):
        """Test node placement in a LinearGrid."""
        L, M, N, NFP, axis, endpoint = 8, 5, 3, 2, True, False
        g = LinearGrid(L, M, N, NFP, sym=False, axis=axis, endpoint=endpoint)

        np.testing.assert_equal(g.num_rho, L + 1)
        np.testing.assert_equal(g.num_theta, 2 * M + 1)
        np.testing.assert_equal(g.num_zeta, 2 * N + 1)
        assert g.endpoint == endpoint

        nodes = np.stack(
            [
                np.tile(
                    np.repeat(np.linspace(1, 0, g.num_rho, axis)[::-1], g.num_theta),
                    g.num_zeta,
                ),
                np.tile(
                    np.linspace(0, 2 * np.pi, g.num_theta, endpoint),
                    g.num_rho * g.num_zeta,
                ),
                np.repeat(
                    np.linspace(0, 2 * np.pi / NFP, g.num_zeta, endpoint),
                    g.num_rho * g.num_theta,
                ),
            ]
        ).T

        np.testing.assert_allclose(g.nodes, nodes)
        np.testing.assert_allclose(g.weights.sum(), (2 * np.pi) ** 2)
        # spacing.prod != weights for grid with duplicates
        if not endpoint:
            np.testing.assert_allclose(g.spacing.prod(axis=1), g.weights)

    @pytest.mark.unit
    def test_grid_LMN(self):
        """Make sure grid attributes LMN are set correctly."""
        theta = 5
        g1 = LinearGrid(theta=theta, zeta=3, rho=2, sym=True)
        assert g1.num_theta == (theta + 1) // 2
        g2 = LinearGrid(M=g1.M, N=g1.N, L=g1.L, sym=True)
        assert (g1.L, g1.M, g1.N) == (g2.L, g2.M, g2.N)
        g1 = LinearGrid(theta=theta, zeta=3, rho=2, sym=False)
        assert g1.num_theta == theta
        g2 = LinearGrid(M=g1.M, N=g1.N, L=g1.L, sym=False)
        assert (g1.L, g1.M, g1.N) == (g2.L, g2.M, g2.N)

    @pytest.mark.unit
    def test_linear_grid_spacing_consistency(self):
        """Test consistency between alternate construction methods."""

        def test(sym, endpoint, axis, ntheta_is_odd):
            nrho = 1
            ntheta = 6 + ntheta_is_odd
            nzeta = 7
            NFP = 3
            # We need to construct the theta coordinates to match
            # whatever is done in grid.py for construction via theta=integer input.
            # First round up ntheta + endpoint to next even integer.
            theta = np.linspace(
                0,
                2 * np.pi,
                (
                    (ntheta + endpoint + 1) // 2 * 2 - endpoint
                    if (sym and ntheta > 1)
                    else ntheta
                ),
                endpoint=endpoint,
            )
            if sym and ntheta > 1:
                # Second, change the array to match the effect done in grid.py,
                # which sets dtheta to a constant. This requires shifting nodes
                # off the axis.
                theta += theta[1] / 2
                # Third, manually delete last node because the addition
                # above pushed it to the (2pi, pi) range which = (0, pi)
                # range, meaning it wouldn't have been deleted by
                # enforce_symmetry which only removes nodes from (pi, 2pi).
                theta = theta[: theta.size - endpoint]
            lg_1 = LinearGrid(
                rho=nrho,
                theta=ntheta,
                zeta=nzeta,
                NFP=NFP,
                sym=sym,
                axis=axis,
                endpoint=endpoint,
            )
            lg_2 = LinearGrid(
                rho=np.linspace(1, 0, nrho, endpoint=axis)[::-1],
                theta=theta,
                zeta=np.linspace(0, 2 * np.pi / NFP, nzeta, endpoint=endpoint),
                NFP=NFP,
                sym=sym,
                axis=axis,
                endpoint=endpoint,
            )
            np.testing.assert_allclose(lg_1.nodes, lg_2.nodes)
            np.testing.assert_allclose(lg_1.spacing, lg_2.spacing)
            np.testing.assert_allclose(lg_1.weights, lg_2.weights)

        # ntheta is odd
        test(sym=False, endpoint=False, axis=False, ntheta_is_odd=True)
        test(sym=False, endpoint=False, axis=True, ntheta_is_odd=True)
        test(sym=False, endpoint=True, axis=False, ntheta_is_odd=True)
        test(sym=False, endpoint=True, axis=True, ntheta_is_odd=True)
        test(sym=True, endpoint=False, axis=False, ntheta_is_odd=True)
        test(sym=True, endpoint=False, axis=True, ntheta_is_odd=True)
        test(sym=True, endpoint=True, axis=False, ntheta_is_odd=True)
        test(sym=True, endpoint=True, axis=True, ntheta_is_odd=True)
        # ntheta is even
        test(sym=False, endpoint=False, axis=False, ntheta_is_odd=False)
        test(sym=False, endpoint=False, axis=True, ntheta_is_odd=False)
        test(sym=False, endpoint=True, axis=False, ntheta_is_odd=False)
        test(sym=False, endpoint=True, axis=True, ntheta_is_odd=False)
        test(sym=True, endpoint=False, axis=False, ntheta_is_odd=False)
        test(sym=True, endpoint=False, axis=True, ntheta_is_odd=False)
        test(sym=True, endpoint=True, axis=False, ntheta_is_odd=False)
        test(sym=True, endpoint=True, axis=True, ntheta_is_odd=False)

    @pytest.mark.unit
    def test_linear_grid_symmetric_nodes_consistency(self):
        """Test that specifying theta nodes from [0, pi] is sufficient."""
        # uniform spacing
        theta = np.linspace(0, 2 * np.pi, 10)
        lg_1 = LinearGrid(L=5, theta=theta, N=5, sym=True)
        lg_2 = LinearGrid(L=5, theta=theta[theta <= np.pi], N=5, sym=True)
        np.testing.assert_allclose(lg_1.nodes, lg_2.nodes)
        np.testing.assert_allclose(lg_1.spacing, lg_2.spacing)
        np.testing.assert_allclose(lg_1.weights, lg_2.weights)

        # non-uniform spacing
        pts = np.linspace(0, 1, 10)
        rho = np.array([r**3 for r in pts])
        theta = 2 * np.pi * np.array([t**1.85 for t in pts])
        zeta = 2 * np.pi * np.array([z**4 for z in pts])
        lg_1 = LinearGrid(rho=rho, theta=theta, zeta=zeta, sym=True)
        lg_2 = LinearGrid(rho=rho, theta=theta[theta <= np.pi], zeta=zeta, sym=True)
        np.testing.assert_allclose(lg_1.nodes, lg_2.nodes)
        np.testing.assert_allclose(lg_1.spacing, lg_2.spacing)
        np.testing.assert_allclose(lg_1.weights, lg_2.weights)

    @pytest.mark.unit
    def test_linear_grid_spacing_two_nodes(self):
        """Test that 2 node grids assign equal spacing to nodes."""
        node_count = 2
        NFP = 7  # any integer > 1 is good candidate for test

        def test(endpoint):
            lg = LinearGrid(
                theta=np.linspace(0, 2 * np.pi, node_count, endpoint=endpoint),
                zeta=np.linspace(0, 2 * np.pi / NFP, node_count, endpoint=endpoint),
                NFP=NFP,
                sym=False,
                endpoint=endpoint,
            )
            # When endpoint is true the rho nodes spacing should be scaled down
            # so that theta and zeta surface integrals weigh the duplicate
            # nodes less.
            spacing = np.tile(
                [1 / 2 if endpoint else 1, np.pi, np.pi], (node_count**2, 1)
            )
            np.testing.assert_allclose(lg.spacing, spacing)

        test(endpoint=False)
        test(endpoint=True)

    @pytest.mark.unit
    def test_spacing_when_duplicate_node_is_removed(self):
        """Test grid spacing when the duplicate node is removed due to symmetry."""
        sym = True
        endpoint = True
        spacing = np.tile([1, np.pi, 2 * np.pi], (2, 1))

        lg = LinearGrid(L=0, M=1, N=0, sym=sym, endpoint=endpoint)
        np.testing.assert_allclose(lg.spacing, spacing)

        lg_2 = LinearGrid(
            rho=np.linspace(1, 0, num=1)[::-1],
            theta=np.linspace(0, 2 * np.pi, num=3, endpoint=endpoint),
            zeta=np.linspace(0, 2 * np.pi, num=1, endpoint=endpoint),
            sym=sym,
            endpoint=endpoint,
        )
        np.testing.assert_allclose(lg_2.spacing, spacing)
        np.testing.assert_allclose(lg_2.weights, spacing.prod(axis=1))

    @pytest.mark.unit
    def test_node_spacing_non_sym(self):
        """Test surface spacing on grids with sym=False."""
        self._test_node_spacing_non_sym(False, 8, 13, 3)
        self._test_node_spacing_non_sym(True, 8, 13, 3)
        self._test_node_spacing_non_sym(False, 1, 1, 3)

    @staticmethod
    def _test_node_spacing_non_sym(
        endpoint, unique_theta_count, unique_zeta_count, NFP
    ):
        """Test surface spacing on grids with sym=False."""
        nrho = 1
        sym = False

        def test(grid):
            if not endpoint:
                # if no duplicates
                np.testing.assert_allclose(grid.weights, grid.spacing.prod(axis=1))
            is_theta_dupe = (endpoint and unique_theta_count > 0 and not sym) & (
                grid.nodes[:, 1] % (2 * np.pi) == 0
            )
            is_zeta_dupe = (endpoint and unique_zeta_count > 0) & (
                grid.nodes[:, 2] % (2 * np.pi / NFP) == 0
            )

            def test_surface(surface_label, actual_ds, desired_ds):
                for index, ds in enumerate(actual_ds):
                    if is_theta_dupe[index] and is_zeta_dupe[index]:
                        # the grid has 4 of these nodes
                        np.testing.assert_allclose(
                            ds, desired_ds / 4, err_msg=surface_label
                        )
                    elif is_theta_dupe[index] or is_zeta_dupe[index]:
                        # the grid has 2 of these nodes
                        np.testing.assert_allclose(
                            ds, desired_ds / 2, err_msg=surface_label
                        )
                    else:
                        # unique node
                        np.testing.assert_allclose(
                            ds, desired_ds, err_msg=surface_label
                        )

            test_surface(
                "rho",
                grid.spacing[:, 1:].prod(axis=1),
                (2 * np.pi / max(unique_theta_count, 1))
                * (2 * np.pi / max(unique_zeta_count, 1)),
            )
            test_surface(
                "theta",
                grid.spacing[:, [0, 2]].prod(axis=1),
                (1 / nrho) * (2 * np.pi / max(unique_zeta_count, 1)),
            )
            test_surface(
                "zeta",
                grid.spacing[:, :2].prod(axis=1),
                (1 / nrho) * (2 * np.pi / max(unique_theta_count, 1)),
            )

        lg_1 = LinearGrid(
            rho=nrho,
            theta=unique_theta_count + endpoint,
            zeta=unique_zeta_count + endpoint,
            NFP=NFP,
            sym=sym,
            endpoint=endpoint,
        )
        rho = np.linspace(1, 0, nrho)[::-1]
        theta = np.linspace(
            0, 2 * np.pi, unique_theta_count + endpoint, endpoint=endpoint
        )
        zeta = np.linspace(
            0, 2 * np.pi / NFP, unique_zeta_count + endpoint, endpoint=endpoint
        )
        lg_2 = LinearGrid(
            rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=sym, endpoint=endpoint
        )
        lg_3 = LinearGrid(
            rho=rho,
            theta=theta,
            zeta=zeta,
            NFP=NFP,
            sym=sym,
            endpoint=not endpoint,  # incorrect marker should have no effect
        )
        assert lg_3.endpoint == endpoint
        test(lg_1)
        # The test below might fail for theta and zeta surfaces only if nrho > 1.
        # This is unrelated to how duplicate node spacing is handled.
        # The cause is because grid construction does not always
        # compute drho as constant (even when the rho nodes are linearly spaced),
        # and this test assumes drho to be a constant for grids without duplicates.
        test(lg_2)
        test(lg_3)

    @pytest.mark.unit
    def test_symmetry_spacing_basic(self):
        """Test symmetry effect on spacing in a basic case."""
        nrho = 2
        ntheta = 3
        nzeta = 2
        lg = LinearGrid(rho=nrho, theta=ntheta, zeta=nzeta, sym=False)
        lg_sym = LinearGrid(rho=nrho, theta=ntheta, zeta=nzeta, sym=True)
        np.testing.assert_allclose(
            lg.spacing,
            np.tile(
                [1 / nrho, 2 * np.pi / ntheta, 2 * np.pi / nzeta],
                (nrho * ntheta * nzeta, 1),
            ),
        )
        np.testing.assert_allclose(lg.weights, lg.spacing.prod(axis=1))
        np.testing.assert_allclose(
            lg_sym.spacing,
            np.tile(
                [1 / nrho, 2 * np.pi / (ntheta - 1), 2 * np.pi / nzeta],
                (nrho * (ntheta - 1) * nzeta, 1),
            ),
        )
        np.testing.assert_allclose(lg_sym.weights, lg_sym.spacing.prod(axis=1))

    @pytest.mark.unit
    def test_node_spacing_sym(self):
        """Test surface spacing on grids with sym=True."""
        # It's useful to test with ntheta even and odd.
        # These tests should pass for any combination of parameters.
        self._test_node_spacing_sym(False, 8, 13, 3)
        self._test_node_spacing_sym(False, 7, 13, 3)
        self._test_node_spacing_sym(True, 8, 13, 3)
        self._test_node_spacing_sym(True, 7, 13, 3)

    @staticmethod
    def _test_node_spacing_sym(endpoint, ntheta, unique_zeta_count, NFP):
        """Test surface spacing on grids with sym=True."""
        nrho = 1
        sym = True

        # unique_theta_and_reflection count is twice
        # the number of nodes at unique theta coordinates (0, pi)
        # on a given theta curve.
        def test(grid, unique_theta_and_reflection_count):
            is_zeta_dupe = (endpoint and unique_zeta_count > 0) & (
                grid.nodes[:, 2] % (2 * np.pi / NFP) == 0
            )

            def test_surface(label, actual_ds, desired_ds):
                for index, ds in enumerate(actual_ds):
                    if label != "theta" and sym and grid.nodes[index, 1] % np.pi != 0:
                        # these nodes should have double weight to account for
                        # reflection across symmetry line
                        if is_zeta_dupe[index]:
                            # the grid has 2 of these nodes,
                            # so each should have half weight
                            np.testing.assert_allclose(
                                ds, (desired_ds / 2) * 2, err_msg=label
                            )
                        else:
                            np.testing.assert_allclose(
                                ds, desired_ds * 2, err_msg=label
                            )
                    elif is_zeta_dupe[index]:
                        # the grid has 2 of these nodes,
                        # so each should have half weight
                        np.testing.assert_allclose(ds, desired_ds / 2, err_msg=label)
                    else:
                        # unique node
                        np.testing.assert_allclose(ds, desired_ds, err_msg=label)

            test_surface(
                "rho",
                grid.spacing[:, 1:].prod(axis=1),
                (2 * np.pi / max(unique_theta_and_reflection_count, 1))
                * (2 * np.pi / max(unique_zeta_count, 1)),
            )
            test_surface(
                "theta",
                grid.spacing[:, [0, 2]].prod(axis=1),
                (1 / nrho) * (2 * np.pi / max(unique_zeta_count, 1)),
            )
            test_surface(
                "zeta",
                grid.spacing[:, :2].prod(axis=1),
                (1 / nrho) * (2 * np.pi / max(unique_theta_and_reflection_count, 1)),
            )

        lg_1_sym = LinearGrid(
            rho=nrho,
            theta=ntheta,
            zeta=unique_zeta_count + endpoint,
            NFP=NFP,
            sym=sym,
            endpoint=endpoint,
        )
        # Recall that LinearGrid created with integers forces ntheta + endpoint
        # to be even and shifts nodes counterclockwise. The result is that
        # unique_theta_and_reflection_count is ntheta rounded up (down) to the
        # closest even integer when endpoint is false (true).
        test(
            lg_1_sym, unique_theta_and_reflection_count=(ntheta - endpoint + 1) // 2 * 2
        )

        rho = np.linspace(1, 0, nrho)[::-1]
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=endpoint)
        zeta = np.linspace(
            0, 2 * np.pi / NFP, unique_zeta_count + endpoint, endpoint=endpoint
        )
        lg_2_sym = LinearGrid(
            rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=sym, endpoint=endpoint
        )
        lg_3_sym = LinearGrid(
            rho=rho,
            theta=theta,
            zeta=zeta,
            NFP=NFP,
            sym=sym,
            endpoint=not endpoint,  # incorrect marker should have no effect
        )
        # endpoints were deleted
        assert not lg_1_sym.endpoint
        assert not lg_2_sym.endpoint
        assert not lg_3_sym.endpoint
        # The test below might fail for theta and zeta surfaces only if nrho > 1.
        # This is unrelated to how duplicate node spacing is handled.
        # The cause is because grid construction does not always
        # compute drho as constant (even when the rho nodes are linearly spaced),
        # and this test assumes drho to be a constant for grids without duplicates.
        test(lg_2_sym, unique_theta_and_reflection_count=ntheta - endpoint)
        test(lg_3_sym, unique_theta_and_reflection_count=ntheta - endpoint)

    @pytest.mark.unit
    def test_concentric_grid(self):
        """Test node placement in ConcentricGrid."""
        M = 2
        N = 0
        NFP = 1
        grid_ansi = ConcentricGrid(
            M, M, N, NFP, sym=False, axis=True, node_pattern="linear"
        )
        grid_fringe = ConcentricGrid(
            2 * M, M, N, NFP, sym=True, axis=True, node_pattern="linear"
        )
        ansi_nodes = np.stack(
            [
                np.array([0, 1, 1, 1, 1, 1]),
                np.array(
                    [0, 0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5]
                ),
                np.zeros((int((M + 1) * (M + 2) / 2),)),
            ]
        ).T
        fringe_nodes = np.stack(
            [
                np.array([0, 0.5, 0.5, 1, 1, 1]),
                np.array(
                    [
                        np.pi / 2,
                        np.pi / 4,
                        3 * np.pi / 4,
                        np.pi / 6,
                        np.pi / 2,
                        5 * np.pi / 6,
                    ]
                ),
                np.zeros((6,)),
            ]
        ).T
        np.testing.assert_allclose(grid_ansi.nodes, ansi_nodes, err_msg="ansi")
        np.testing.assert_allclose(grid_fringe.nodes, fringe_nodes, err_msg="fringe")
        np.testing.assert_allclose(grid_ansi.weights.sum(), (2 * np.pi) ** 2)

    @pytest.mark.unit
    def test_concentric_grid_high_res(self):
        """Test that we can create high resolution grids without crashing.

        Test for GH issue #207.
        """
        _ = ConcentricGrid(L=32, M=28, N=30)

    @pytest.mark.unit
    def test_quadrature_grid(self):
        """Test node placement in QuadratureGrid."""
        L = 2
        M = 2
        N = 0
        NFP = 1
        grid_quad = QuadratureGrid(L, M, N, NFP)
        roots, weights = special.js_roots(2, 2, 2)
        quadrature_nodes = np.stack(
            [
                np.array(
                    [roots[0]] * grid_quad.num_theta + [roots[1]] * grid_quad.num_theta
                ),
                np.array(
                    [0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5] * 2
                ),
                np.zeros(10),
            ]
        ).T
        np.testing.assert_allclose(grid_quad.spacing.prod(axis=1), grid_quad.weights)
        np.testing.assert_allclose(grid_quad.nodes, quadrature_nodes)

    @pytest.mark.unit
    def test_quad_grid_volume_integration(self):
        """Test that quadrature grid gives correct volume integrals."""
        r = 1
        R = 10
        vol = 2 * (np.pi**2) * (r**2) * R

        inputs = {
            "L": 1,
            "M": 1,
            "N": 0,
            "NFP": 1,
            "Psi": 1.0,
            "pressure": np.array([[0, 0]]),
            "iota": np.array([[0, 0]]),
            "surface": np.array([[0, 0, 0, R, 0], [0, 1, 0, r, 0], [0, -1, 0, 0, -r]]),
            "spectral_indexing": "ansi",
        }

        eq = Equilibrium(**inputs)
        grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)
        g = eq.compute("sqrt(g)", grid=grid)
        vol_quad = np.sum(np.abs(g["sqrt(g)"]) * grid.weights)

        np.testing.assert_allclose(vol, vol_quad)

    @pytest.mark.unit
    def test_repr(self):
        """Test string representations of grid objects."""
        qg = ConcentricGrid(2, 3, 4)
        s = str(qg)
        assert "ConcentricGrid" in s
        assert "jacobi" in s
        assert "L=2" in s
        assert "M=3" in s
        assert "N=4" in s

    @pytest.mark.unit
    def test_change_resolution(self):
        """Test changing grid resolution."""

        def test(grid, *desired_resolution):
            assert (grid.L, grid.M, grid.N, grid.NFP) == desired_resolution
            assert grid.num_rho == grid.unique_rho_idx.size
            assert grid.num_theta == grid.unique_theta_idx.size
            assert grid.num_zeta == grid.unique_zeta_idx.size
            np.testing.assert_equal(
                (grid.unique_rho_idx, grid.inverse_rho_idx),
                np.unique(grid.nodes[:, 0], return_index=True, return_inverse=True)[1:],
            )
            np.testing.assert_equal(
                (grid.unique_theta_idx, grid.inverse_theta_idx),
                np.unique(grid.nodes[:, 1], return_index=True, return_inverse=True)[1:],
            )
            np.testing.assert_equal(
                (grid.unique_zeta_idx, grid.inverse_zeta_idx),
                np.unique(grid.nodes[:, 2], return_index=True, return_inverse=True)[1:],
            )
            np.testing.assert_array_equal(
                grid.axis, np.nonzero(grid.nodes[:, 0] == 0)[0]
            )
            # test that changing NFP updated the nodes
            assert np.isclose(
                grid.nodes[grid.unique_zeta_idx[-1], 2],
                (grid.num_zeta - 1) / grid.num_zeta * 2 * np.pi / grid.NFP,
            )

        lg = LinearGrid(1, 2, 3)
        lg.change_resolution(2, 3, 4, 5)
        test(lg, 2, 3, 4, 5)
        qg = QuadratureGrid(1, 2, 3)
        qg.change_resolution(2, 3, 4, 5)
        test(qg, 2, 3, 4, 5)
        cg = ConcentricGrid(2, 3, 4)
        cg.change_resolution(3, 4, 5, 2)
        test(cg, 3, 4, 5, 2)
        cg = ConcentricGrid(2, 3, 4)
        cg.change_resolution(cg.L, cg.M, cg.N, NFP=5)
        test(cg, cg.L, cg.M, cg.N, 5)

    @pytest.mark.unit
    def test_enforce_symmetry(self):
        """Test correctness of enforce_symmetry() for uniformly spaced nodes.

        Unlike enforce_symmetry(), the algorithm used in LinearGrid for
        symmetry also works if the nodes are not uniformly spaced. This test
        compares the two methods when the grid is uniformly spaced in theta,
        as a means to ensure enforce_symmetry() is correct.
        """
        ntheta = 6
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        lg_1 = LinearGrid(L=5, theta=theta, N=4, NFP=4, sym=True)
        lg_2 = LinearGrid(L=5, theta=theta, N=4, NFP=4, sym=False)
        # precondition for the following tests to work
        np.testing.assert_allclose(lg_2.spacing[:, 1], 2 * np.pi / ntheta)

        lg_2._sym = True
        lg_2._enforce_symmetry()
        np.testing.assert_allclose(lg_1.nodes, lg_2.nodes)
        np.testing.assert_allclose(lg_1.spacing, lg_2.spacing)
        lg_2._weights = lg_2._scale_weights()
        np.testing.assert_allclose(lg_1.spacing, lg_2.spacing)
        np.testing.assert_allclose(lg_1.weights, lg_2.weights)

    @pytest.mark.unit
    def test_symmetry_volume_integral(self):
        """Test volume integral of a symmetric function."""
        L = [3, 3, 5, 3]
        M = [3, 6, 5, 7]
        N = [2, 2, 2, 2]
        NFP = [5, 3, 5, 3]
        sym = np.array([True, True, False, False])
        # to test code not tested on grids made with M=.
        even_number = 4
        n_theta = even_number - sym

        # Currently, midpoint rule is false for LinearGrid made with L=number.
        def test(grid, midpoint_rule=False):
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

            if midpoint_rule:
                r_unique = r[grid.unique_rho_idx]
                dr = np.empty_like(r_unique)
                dr[0] = (r_unique[0] + r_unique[1]) / 2
                dr[1:-1] = (r_unique[2:] - r_unique[:-2]) / 2
                dr[-1] = 1 - (r_unique[-2] + r_unique[-1]) / 2
            else:
                dr = 1 / grid.num_rho
            expected_integral = np.sum(dr * function_of_rho[grid.unique_rho_idx])
            true_integral = np.log(1.35 / 0.35)
            midpoint_rule_error_bound = np.max(dr) ** 2 / 24 * (2 / 0.35**3)
            right_riemann_error_bound = dr * (1 / 0.35 - 1 / 1.35)
            np.testing.assert_allclose(
                expected_integral,
                true_integral,
                rtol=0,
                atol=(
                    midpoint_rule_error_bound / 4
                    if midpoint_rule
                    else right_riemann_error_bound / 3
                ),
                err_msg=type(grid),
            )
            np.testing.assert_allclose(
                np.sum(grid.weights * f) / (4 * np.pi**2 * true_surface_avg),
                expected_integral,
                rtol=1e-15,
                err_msg=type(grid),
            )

        for i in range(len(L)):
            test(LinearGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            test(LinearGrid(L=L[i], theta=n_theta[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, 2 * np.pi, n_theta[i], endpoint=False),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, 2 * np.pi, n_theta[i] + 1, endpoint=False),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(
                ConcentricGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i], sym=sym[i]),
                midpoint_rule=True,
            )
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
            expanded = grid.expand(r, surface_label)
            assert expanded.size == grid.num_nodes
            s = grid.compress(expanded, surface_label)
            np.testing.assert_allclose(r, s, err_msg=surface_label)

        L, M, N, NFP = 6, 6, 3, 5
        lg_endpoint = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=True, endpoint=True)
        cg_sym = ConcentricGrid(L=L, M=M, N=N, NFP=NFP, sym=True)
        test("rho", lg_endpoint)
        test("theta", lg_endpoint)
        test("zeta", lg_endpoint)
        test("rho", cg_sym)
        test("theta", cg_sym)
        test("zeta", cg_sym)

    @pytest.mark.unit
    def test_meshgrid(self):
        """Test meshgrid constructor."""
        R = np.linspace(0, 1, 4)
        A = np.linspace(0, 2 * np.pi, 2)
        Z = np.linspace(0, 2 * np.pi, 3)
        grid = Grid.create_meshgrid(
            [R, A, Z], coordinates="raz", period=(np.inf, 2 * np.pi, 2 * np.pi)
        )
        # treating theta == alpha just for grid construction
        grid1 = LinearGrid(rho=R, theta=A, zeta=Z)
        # atol=1e-12 bc Grid by default shifts points away from the axis a tiny bit
        np.testing.assert_allclose(grid1.nodes, grid.nodes, atol=1e-12)
        # want radial/poloidal/toroidal nodes sorted in the same order for both
        np.testing.assert_allclose(grid1.unique_rho_idx, grid.unique_rho_idx)
        np.testing.assert_allclose(grid1.unique_theta_idx, grid.unique_alpha_idx)
        np.testing.assert_allclose(grid1.unique_zeta_idx, grid.unique_zeta_idx)
        np.testing.assert_allclose(grid1.inverse_rho_idx, grid.inverse_rho_idx)
        np.testing.assert_allclose(grid1.inverse_theta_idx, grid.inverse_alpha_idx)
        np.testing.assert_allclose(grid1.inverse_zeta_idx, grid.inverse_zeta_idx)

    @pytest.mark.unit
    def test_meshgrid_reshape(self):
        """Test that reshaping meshgrids works correctly."""
        grid = LinearGrid(2, 3, 4)

        r = grid.nodes[grid.unique_rho_idx, 0]
        t = grid.nodes[grid.unique_theta_idx, 1]
        z = grid.nodes[grid.unique_zeta_idx, 2]

        # user regular allclose for broadcasting to work correctly
        # reshaping rtz should have rho along first axis
        assert np.allclose(
            grid.meshgrid_reshape(grid.nodes[:, 0], "rtz"), r[:, None, None]
        )
        # reshaping rzt should have theta along last axis
        assert np.allclose(
            grid.meshgrid_reshape(grid.nodes[:, 1], "rzt"), t[None, None, :]
        )
        # reshaping tzr should have zeta along 2nd axis
        assert np.allclose(
            grid.meshgrid_reshape(grid.nodes, "tzr")[:, :, :, 2], z[None, :, None]
        )

        # coordinates are rtz, not raz
        with pytest.raises(ValueError):
            grid.meshgrid_reshape(grid.nodes[:, 0], "raz")

        # not a meshgrid
        grid = ConcentricGrid(2, 3, 4)
        with pytest.raises(ValueError):
            grid.meshgrid_reshape(grid.nodes[:, 0], "rtz")

        rho = np.linspace(0, 1, 3)
        alpha = np.linspace(0, 2 * np.pi, 4)
        zeta = np.linspace(0, 6 * np.pi, 5)
        grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
        r, a, z = grid.nodes.T
        # functions of zeta should separate along first two axes
        # since those are contiguous, this should work
        f = grid.meshgrid_reshape(z, "raz").reshape(-1, zeta.size)
        for i in range(1, f.shape[0]):
            np.testing.assert_allclose(f[i - 1], f[i])
        # likewise for rho
        f = grid.meshgrid_reshape(r, "raz").reshape(rho.size, -1)
        for i in range(1, f.shape[-1]):
            np.testing.assert_allclose(f[:, i - 1], f[:, i])
        # test reshaping result won't mix data
        f = grid.meshgrid_reshape(a**2 + z, "raz")
        for i in range(1, f.shape[0]):
            np.testing.assert_allclose(f[i - 1], f[i])
        f = grid.meshgrid_reshape(r**2 + z, "raz")
        for i in range(1, f.shape[1]):
            np.testing.assert_allclose(f[:, i - 1], f[:, i])
        f = grid.meshgrid_reshape(r**2 + a, "raz")
        for i in range(1, f.shape[-1]):
            np.testing.assert_allclose(f[..., i - 1], f[..., i])

    @pytest.mark.unit
    def test_meshgrid_flatten(self):
        """Test that meshgrid_flatten is the inverse of meshgrid_reshape."""
        grid = LinearGrid(2, 3, 4)
        orders = ["rtz", "trz", "zrt", "rzt", "tzr", "ztr"]
        rng = np.random.default_rng(123)
        x = rng.random(grid.num_nodes)
        for order in orders:
            y = grid.meshgrid_reshape(x, order)
            z = grid.meshgrid_flatten(y, order)
            np.testing.assert_allclose(x, z)

        x = rng.random((grid.num_nodes, 3))
        for order in orders:
            y = grid.meshgrid_reshape(x, order)
            z = grid.meshgrid_flatten(y, order)
            np.testing.assert_allclose(x, z)


@pytest.mark.unit
def test_find_most_rational_surfaces():
    """Test finding the most rational surfaces and their locations."""
    # simple test, linear iota going from 1 to 3
    iota = PowerSeriesProfile([1, 2])
    rho, io = find_most_rational_surfaces(iota, 5)
    np.testing.assert_allclose(rho, np.linspace(0, 1, 5), atol=1e-14, rtol=0)
    np.testing.assert_allclose(io, np.linspace(1, 3, 5), atol=1e-14, rtol=0)

    # simple test, linear iota going from -1 to -3
    iota = PowerSeriesProfile([-1, -2])
    rho, io = find_most_rational_surfaces(iota, 5)
    np.testing.assert_allclose(rho, np.linspace(0, 1, 5), atol=1e-14, rtol=0)
    np.testing.assert_allclose(io, np.linspace(-1, -3, 5), atol=1e-14, rtol=0)

    # invalid (a > b) ranges and negative ranges are swapped and made positive
    all_same = [
        most_rational(1, 2),
        most_rational(2, 1),
        most_rational(-1, -2),
        most_rational(-2, -1),
    ]

    assert len(set(map(abs, all_same))) == 1

    # if 0 in range, return 0
    has_zero = [
        most_rational(0, 1),
        most_rational(0, -1),
        most_rational(0, 0),
        most_rational(-1, 0),
        most_rational(1, 0),
    ]

    assert all(result == 0 for result in has_zero)


@pytest.mark.unit
def test_find_least_rational_surfaces():
    """Test finding the least rational surfaces and their locations."""
    # simple test, linear iota going from 1 to 3
    iota = PowerSeriesProfile([1, 2])
    rhor, ior = find_most_rational_surfaces(iota, 10)
    rho, io = find_least_rational_surfaces(iota, 10)
    # to compare how rational/irrational things are, we use the length of the
    # continued fraction. Not a great test due to rounding errors, but seems to work
    lio = [len(dec_to_cf(i)) for i in io]
    lior = [len(dec_to_cf(i)) for i in ior]
    max_rational = max(lior)

    assert np.all(np.array(lio) > max_rational)


@pytest.mark.unit
def test_custom_jitable_grid_indexing():
    """Test that unique/inverse indices are set correctly when jitable=True."""
    eq = get("NCSX")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(3, 3, 3, 6, 6, 6)
    # field lines on two surfaces
    rho = np.concatenate([0.5 * np.ones(10), 0.7 * np.ones(10)])
    theta = np.concatenate([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]) * 2 * np.pi
    zeta = np.concatenate([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]) * 2 * np.pi
    grid1 = Grid(np.array([rho, theta, zeta]).T, jitable=False)
    grid2 = Grid(np.array([rho, theta, zeta]).T, jitable=True)
    grid3 = Grid(np.array([rho, theta, zeta]).T, jitable=True, _unique_rho_idx=[0, 10])
    np.testing.assert_allclose(grid1.nodes, grid2.nodes)

    x = np.random.random(grid1.num_nodes)

    # these shouldn't error
    _ = grid1.unique_rho_idx
    _ = grid1.unique_theta_idx
    _ = grid1.unique_zeta_idx
    _ = grid1.inverse_rho_idx
    _ = grid1.inverse_theta_idx
    _ = grid1.inverse_zeta_idx
    _ = grid3.unique_rho_idx

    with pytest.raises(AttributeError):
        _ = grid2.unique_rho_idx
    with pytest.raises(AttributeError):
        _ = grid2.unique_theta_idx
    with pytest.raises(AttributeError):
        _ = grid2.unique_zeta_idx
    with pytest.raises(AttributeError):
        _ = grid2.inverse_rho_idx
    with pytest.raises(AttributeError):
        _ = grid2.inverse_theta_idx
    with pytest.raises(AttributeError):
        _ = grid2.inverse_zeta_idx

    assert not hasattr(grid1, "weights")
    assert not hasattr(grid1, "spacing")
    assert not hasattr(grid1, "source_grid")
    assert not hasattr(grid2, "num_rho")
    assert not hasattr(grid2, "num_theta")
    assert not hasattr(grid2, "num_zeta")

    y1 = grid1.copy_data_from_other(x, grid2, "rho")
    y2 = grid2.copy_data_from_other(x, grid1, "rho")
    y3 = grid3.copy_data_from_other(x, grid1, "rho")

    np.testing.assert_allclose(y1, y2)
    np.testing.assert_allclose(y1, y3)

    # make sure compress/expand done when override_grid=True works as expected
    b1 = eq.compute(["|B|"], grid=grid1, override_grid=True)["|B|"]
    with pytest.raises(AttributeError):
        _ = eq.compute(["|B|"], grid=grid2, override_grid=True)["|B|"]
    b3 = eq.compute(["|B|"], grid=grid3, override_grid=True)["|B|"]
    np.testing.assert_allclose(b1, b3)
