"""Tests for Grid classes."""

import numpy as np
import pytest
from scipy import special

from desc.basis import FourierZernikeBasis
from desc.compute.utils import compress, surface_averages, surface_integrals
from desc.equilibrium import Equilibrium
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.transform import Transform


class TestGrid:
    """Test Grid classes."""

    @pytest.mark.unit
    def test_custom_grid(self):
        """Test creating a grid with custom set of nodes."""
        nodes = np.array(
            [
                [0, 0, 0],
                [0.25, 0, 0],
                [0.5, np.pi / 2, np.pi / 3],
                [0.5, np.pi / 2, np.pi / 3],
                [0.75, np.pi, np.pi],
                [1, 2 * np.pi, 3 * np.pi / 2],
            ]
        )
        grid = Grid(nodes)
        weights = grid.weights

        w = 4 * np.pi**2 / (grid.num_nodes - 1)
        weights_ref = np.array([w, w, w / 2, w / 2, w, w])

        np.testing.assert_allclose(weights, weights_ref)
        np.testing.assert_allclose(np.sum(grid.weights), (2 * np.pi) ** 2)

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
        # spacing.prod == weights for linear grids
        # this is not true for concentric or any grid with duplicates
        if not endpoint:
            np.testing.assert_allclose(g.spacing.prod(axis=1), g.weights)

    @pytest.mark.unit
    def test_linear_grid_spacing(self):
        """Test linear grid spacing is consistent."""

        def test(sym, endpoint, axis, ntheta_is_odd):
            nrho = 1
            # more edge cases are caught when ntheta is odd when endpoint=True
            # and ntheta is even when endpoint is False
            ntheta = 6 + ntheta_is_odd
            nzeta = 7
            NFP = 3
            theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=endpoint)
            if sym:
                # When users supply nodes and set symmetry to true, they need
                # to ensure their nodes are placed symmetrically.
                # Here we change the array to match the effect done in grid.py,
                # so the nodes are actually symmetric.
                theta += theta[1] / 2
                if endpoint:
                    # We have to manually delete last node because the addition
                    # above pushed it to the (2pi, pi) range which = (0, pi)
                    # range, meaning it wouldn't have been deleted by
                    # enforce_symmetry which only removes nodes from (pi, 2pi).
                    theta = theta[:-1]
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

    @pytest.mark.unit
    def test_duplicate_node_spacing(self):
        """Test surface spacing on all types of grids with endpoint=True."""
        nrho = 1
        ntheta = 8  # unique theta count
        nzeta = 13  # unique zeta count
        NFP = 3
        axis = True
        endpoint = True

        def test(grid):
            is_theta_dupe = (endpoint and (ntheta > 0) and not grid.sym) & (
                grid.nodes[:, 1] % (2 * np.pi) == 0
            )
            is_zeta_dupe = (endpoint and nzeta > 0) & (
                grid.nodes[:, 2] % (2 * np.pi / NFP) == 0
            )

            def test_surface(label, actual_ds, desired_ds):
                for index, ds in enumerate(actual_ds):
                    if is_theta_dupe[index] and is_zeta_dupe[index]:
                        # the grid has 4 of these nodes
                        np.testing.assert_allclose(ds, desired_ds / 4, err_msg=label)
                    elif is_theta_dupe[index] or is_zeta_dupe[index]:
                        # the grid has 2 of these nodes
                        np.testing.assert_allclose(ds, desired_ds / 2, err_msg=label)
                    else:
                        # unique node
                        np.testing.assert_allclose(ds, desired_ds, err_msg=label)

            if grid.sym:
                dtheta_scale = max(ntheta, 1) / grid.num_theta
            else:
                dtheta_scale = 1
            test_surface(
                "rho",
                grid.spacing[:, 1:].prod(axis=1),
                (2 * np.pi / max(ntheta, 1))
                * (2 * np.pi / max(nzeta, 1))
                * dtheta_scale,
            )
            test_surface(
                "theta",
                grid.spacing[:, [0, 2]].prod(axis=1),
                (1 / nrho) * (2 * np.pi / max(nzeta, 1)),
            )
            test_surface(
                "zeta",
                grid.spacing[:, :2].prod(axis=1),
                (1 / nrho) * (2 * np.pi / max(ntheta, 1)) * dtheta_scale,
            )

        lg_1 = LinearGrid(
            rho=nrho,
            theta=ntheta + endpoint,
            zeta=nzeta + endpoint,
            NFP=NFP,
            sym=False,
            axis=axis,
            endpoint=endpoint,
        )
        lg_1_sym = LinearGrid(
            rho=nrho,
            theta=ntheta + endpoint,
            zeta=nzeta + endpoint,
            NFP=NFP,
            sym=True,
            axis=axis,
            endpoint=endpoint,
        )
        lg_2 = LinearGrid(
            rho=np.linspace(1, 0, nrho, endpoint=axis)[::-1],
            theta=np.linspace(0, 2 * np.pi, ntheta + endpoint, endpoint=endpoint),
            zeta=np.linspace(0, 2 * np.pi / NFP, nzeta + endpoint, endpoint=endpoint),
            NFP=NFP,
            sym=False,
            axis=axis,
            endpoint=endpoint,
        )
        lg_2_sym = LinearGrid(
            rho=np.linspace(1, 0, nrho, endpoint=axis)[::-1],
            theta=np.linspace(0, 2 * np.pi, ntheta + endpoint, endpoint=endpoint),
            zeta=np.linspace(0, 2 * np.pi / NFP, nzeta + endpoint, endpoint=endpoint),
            NFP=NFP,
            sym=True,
            axis=axis,
            endpoint=endpoint,
        )
        lg_3 = LinearGrid(
            rho=np.linspace(1, 0, nrho, endpoint=axis)[::-1],
            theta=np.linspace(0, 2 * np.pi, ntheta + endpoint, endpoint=endpoint),
            zeta=np.linspace(0, 2 * np.pi / NFP, nzeta + endpoint, endpoint=endpoint),
            NFP=NFP,
            sym=False,
            axis=axis,
            endpoint=not endpoint,  # incorrect marker should have no effect
        )
        assert lg_3.endpoint == endpoint
        test(lg_1)
        test(lg_1_sym)
        # The test below might fail for theta and zeta surfaces only if nrho > 1.
        # This is unrelated to how duplicate node spacing is handled.
        # The cause is because grid construction does not always
        # compute drho as constant (even when the rho nodes are linearly spaced),
        # and this test assumes drho to be a constant for grids without duplicates.
        test(lg_2)
        test(lg_2_sym)
        test(lg_3)

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
                    [
                        0,
                        0,
                        2 * np.pi / 5,
                        4 * np.pi / 5,
                        6 * np.pi / 5,
                        8 * np.pi / 5,
                    ]
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
    def test_quadrature_grid(self):
        """Test node placement in QuadratureGrid."""
        L = 2
        M = 2
        N = 0
        NFP = 1

        grid_quad = QuadratureGrid(L, M, N, NFP)

        roots, weights = special.js_roots(3, 2, 2)

        quadrature_nodes = np.stack(
            [
                np.array([roots[0]] * 5 + [roots[1]] * 5 + [roots[2]] * 5),
                np.array(
                    [0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5] * 3
                ),
                np.zeros(15),
            ]
        ).T

        # spacing.prod == weights for quad grids (not true for concentric)
        np.testing.assert_allclose(grid_quad.spacing.prod(axis=1), grid_quad.weights)
        np.testing.assert_allclose(grid_quad.nodes, quadrature_nodes)

    @pytest.mark.unit
    def test_concentric_grid_high_res(self):
        """Test that we can create high resolution grids without crashing.

        Verifies solution to GH issue #207.
        """
        _ = ConcentricGrid(L=32, M=28, N=30)

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
            "surface": np.array([[0, 0, 0, R, 0], [0, 1, 0, r, 0], [0, -1, 0, 0, r]]),
            "spectral_indexing": "ansi",
            "bdry_mode": "lcfs",
            "node_pattern": "quad",
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
        lg = LinearGrid(1, 2, 3)
        lg.change_resolution(2, 3, 4)
        assert lg.L == 2
        assert lg.M == 3
        assert lg.N == 4

        qg = QuadratureGrid(1, 2, 3)
        qg.change_resolution(2, 3, 4)
        assert qg.L == 2
        assert qg.M == 3
        assert qg.N == 4

        cg = ConcentricGrid(2, 3, 4)
        cg.change_resolution(3, 4, 5)
        assert cg.L == 3
        assert cg.M == 4
        assert cg.N == 5

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
        np.testing.assert_allclose(
            lg_sym.spacing,
            np.tile(
                [1 / nrho, 2 * np.pi / (ntheta - 1), 2 * np.pi / nzeta],
                (nrho * (ntheta - 1) * nzeta, 1),
            ),
        )

    @pytest.mark.unit
    def test_enforce_symmetry(self):
        """Test that enforce_symmetry spaces theta nodes correctly.

        Relies on correctness of surface_integrals.
        """

        def test(grid):
            # check if theta nodes cover the circumference of the theta curve
            dtheta_sums = surface_integrals(grid, q=1 / grid.spacing[:, 2])
            np.testing.assert_allclose(dtheta_sums, 2 * np.pi * grid.num_zeta)

        # Before enforcing symmetry,
        # this grid has 2 surfaces near axis lacking theta > pi nodes.
        # These edge cases should be handled correctly.
        # Otherwise, a dimension mismatch or broadcast error should be raised.
        test(ConcentricGrid(L=20, M=3, N=2, sym=True))
        test(LinearGrid(L=20, M=3, N=2, sym=True))

    @pytest.mark.unit
    def test_symmetry_1(self):
        """Test surface averages of a smooth function."""

        def test(grid, err_msg):
            t = grid.nodes[:, 1]
            z = grid.nodes[:, 2] * grid.NFP
            true_avg = 5
            f = (
                true_avg
                + np.cos(t)
                - 0.5 * np.cos(z)
                + 3 * np.cos(t) * np.cos(z)
                - 2 * np.sin(z) * np.sin(t)
            )
            numerical_avg = surface_averages(grid, f)
            np.testing.assert_allclose(numerical_avg, true_avg, err_msg=err_msg)

        L, M, N, NFP, sym = 6, 6, 3, 5, True
        test(LinearGrid(L, M, N, NFP, sym), "LinearGrid")
        test(QuadratureGrid(L, M, N, NFP), "QuadratureGrid")
        test(ConcentricGrid(L, M, N, NFP, sym), "ConcentricGrid")

        L, M, N, NFP, sym = 3, 6, 3, 3, True
        test(LinearGrid(L, M, N, NFP, sym), "LinearGrid")
        test(QuadratureGrid(L, M, N, NFP), "QuadratureGrid")
        test(ConcentricGrid(L, M, N, NFP, sym), "ConcentricGrid")

        L, M, N, NFP, sym = 5, 5, 3, 5, False
        test(LinearGrid(L, M, N, NFP, sym), "LinearGrid")
        test(QuadratureGrid(L, M, N, NFP), "QuadratureGrid")
        test(ConcentricGrid(L, M, N, NFP, sym), "ConcentricGrid")

        L, M, N, NFP, sym = 3, 7, 3, 3, False
        test(LinearGrid(L, M, N, NFP, sym), "LinearGrid")
        test(QuadratureGrid(L, M, N, NFP), "QuadratureGrid")
        test(ConcentricGrid(L, M, N, NFP, sym), "ConcentricGrid")

    @pytest.mark.unit
    def test_symmetry_2(self):
        """Tests that surface averages are correct using specified basis."""

        def test(grid, basis, err_msg, true_avg=1):
            transform = Transform(grid, basis)

            # random data with specified average on each surface
            coeffs = np.random.rand(basis.num_modes)
            coeffs[np.where((basis.modes[:, 1:] == [0, 0]).all(axis=1))[0]] = 0
            coeffs[np.where((basis.modes == [0, 0, 0]).all(axis=1))[0]] = true_avg

            # compute average for each surface in grid
            values = transform.transform(coeffs)
            numerical_avg = compress(grid, surface_averages(grid, values))
            if isinstance(grid, ConcentricGrid):
                # values closest to axis are never accurate enough
                numerical_avg = numerical_avg[1:]
            np.testing.assert_allclose(numerical_avg, true_avg, err_msg=err_msg)

        M = 10
        M_grid = 23
        test(
            QuadratureGrid(L=M_grid, M=M_grid, N=0),
            FourierZernikeBasis(L=M, M=M, N=0),
            "QuadratureGrid",
        )
        test(
            LinearGrid(L=M_grid, M=M_grid, N=0, sym=True),
            FourierZernikeBasis(L=M, M=M, N=0, sym="cos"),
            "LinearGrid with symmetry",
        )
        test(
            ConcentricGrid(L=M_grid, M=M_grid, N=0),
            FourierZernikeBasis(L=M, M=M, N=0),
            "ConcentricGrid without symmetry",
        )
        test(
            ConcentricGrid(L=M_grid, M=M_grid, N=0, sym=True),
            FourierZernikeBasis(L=M, M=M, N=0, sym="cos"),
            "ConcentricGrid with symmetry",
        )
