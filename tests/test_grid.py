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
        # spacing.prod == weights for linear grids (not true for concentric)
        np.testing.assert_allclose(g.spacing.prod(axis=1), g.weights)
        np.testing.assert_allclose(g.weights.sum(), (2 * np.pi) ** 2)

    @pytest.mark.unit
    def test_linear_grid_spacing(self):
        """Test linear grid spacing is consistent."""

        def test(endpoint=False, axis=True):
            nrho = 1
            ntheta = 5
            nzeta = 7
            NFP = 3
            grid1 = LinearGrid(
                rho=nrho,
                theta=ntheta,
                zeta=nzeta,
                NFP=NFP,
                axis=axis,
                endpoint=endpoint,
            )
            grid2 = LinearGrid(
                rho=np.linspace(1, 0, nrho, endpoint=axis)[::-1],
                theta=np.linspace(0, 2 * np.pi, ntheta, endpoint=endpoint),
                zeta=np.linspace(0, 2 * np.pi / NFP, nzeta, endpoint=endpoint),
                NFP=NFP,
                axis=axis,
                endpoint=endpoint,
            )
            np.testing.assert_allclose(grid1.nodes, grid2.nodes)
            np.testing.assert_allclose(grid1.spacing, grid2.spacing)

        test(endpoint=False)
        test(axis=False)
        test(axis=True)

    @pytest.mark.unit
    def test_linear_grid_spacing_two_nodes(self):
        """Test that 2 node grids assign equal spacing to nodes."""
        node_count = 2
        NFP = 7  # any integer > 1 is good candidate for test
        endpoint = False  # TODO: fix endpoint = True issue later
        lg = LinearGrid(
            theta=np.linspace(0, 2 * np.pi, node_count, endpoint=endpoint),
            zeta=np.linspace(0, 2 * np.pi / NFP, node_count, endpoint=endpoint),
            NFP=NFP,
            endpoint=endpoint,
        )
        spacing = np.tile([1, np.pi, np.pi], (node_count * node_count, 1))
        np.testing.assert_allclose(lg.spacing, spacing)

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
        g = eq.compute("sqrt(g)", grid)
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
    def test_enforce_symmetry(self):
        """Test that enforce_symmetry spaces theta nodes correctly."""

        def test(grid):
            # check if theta nodes cover the circumference of the theta curve
            dtheta_sums = surface_integrals(grid, q=1 / grid.spacing[:, 2])
            np.testing.assert_allclose(dtheta_sums, 2 * np.pi * grid.num_zeta)

        # Before enforcing symmetry,
        # this grid has 2 surfaces near axis lacking theta > pi nodes.
        # These edge cases should be handled correctly.
        # Otherwise, a dimension mismatch / broadcast error should be raised.
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
