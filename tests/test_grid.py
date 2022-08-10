import unittest
import pytest
import numpy as np
from scipy import special
from desc.compute.utils import surface_averages, surface_integrals
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.equilibrium import Equilibrium


class TestGrid(unittest.TestCase):
    """Test Grid classes."""

    def test_custom_grid(self):

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

        w = 4 * np.pi ** 2 / (grid.num_nodes - 1)
        weights_ref = np.array([w, w, w / 2, w / 2, w, w])

        np.testing.assert_allclose(weights, weights_ref, atol=1e-12)
        self.assertAlmostEqual(np.sum(grid.weights), (2 * np.pi) ** 2)

    def test_linear_grid(self):

        L = 2
        M = 1
        N = 1
        NFP = 1

        grid = LinearGrid(L=L, M=M, N=N, NFP=NFP, sym=False, axis=True, endpoint=False)

        nodes = np.stack(
            [
                np.array(
                    [
                        0,
                        0,
                        0,
                        0.5,
                        0.5,
                        0.5,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0.5,
                        0.5,
                        0.5,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0.5,
                        0.5,
                        0.5,
                        1,
                        1,
                        1,
                    ]
                ),
                np.array(
                    [
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                    ]
                ),
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                        4 * np.pi / 3,
                    ]
                ),
            ]
        ).T

        np.testing.assert_allclose(grid.nodes, nodes, atol=1e-8)

        self.assertAlmostEqual(np.sum(grid.weights), (2 * np.pi) ** 2 / NFP)

    def test_concentric_grid(self):

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
                np.array(
                    [
                        0,
                        0.5,
                        0.5,
                        1,
                        1,
                        1,
                    ]
                ),
                np.array(
                    [
                        2 / 3 * np.pi,
                        2 / 3 * np.pi / 3,
                        2 * np.pi / 3 + 2 / 3 * np.pi / 3,
                        2 / 3 * np.pi / 5,
                        2 * np.pi / 5 + 2 / 3 * np.pi / 5,
                        4 * np.pi / 5 + 2 / 3 * np.pi / 5,
                    ]
                ),
                np.zeros((6,)),
            ]
        ).T

        np.testing.assert_allclose(grid_ansi.nodes, ansi_nodes, atol=1e-8)
        np.testing.assert_allclose(grid_fringe.nodes, fringe_nodes, atol=1e-8)

        self.assertAlmostEqual(np.sum(grid_ansi.weights), (2 * np.pi) ** 2 / NFP)

    def test_quadrature_grid(self):

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

        np.testing.assert_allclose(grid_quad.nodes, quadrature_nodes, atol=1e-8)

    def test_concentric_grid_high_res(self):
        # need to make sure this builds without crashing, as in GH issue #207
        grid = ConcentricGrid(L=32, M=28, N=30)

    def test_quad_grid_volume_integration(self):

        r = 1
        R = 10
        vol = 2 * (np.pi ** 2) * (r ** 2) * R

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

        self.assertAlmostEqual(vol, vol_quad)

    def test_repr(self):

        qg = ConcentricGrid(2, 3, 4)
        s = str(qg)
        assert "ConcentricGrid" in s
        assert "jacobi" in s
        assert "L=2" in s
        assert "M=3" in s
        assert "N=4" in s

    def test_change_resolution(self):

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

    def test_rotation(self):
        M = 2
        N = 0
        NFP = 1
        cos_grid = ConcentricGrid(
            M, M, N, NFP, sym=False, axis=True, rotation="cos", node_pattern="linear"
        )
        sin_grid = ConcentricGrid(
            M, M, N, NFP, sym=False, axis=True, rotation="sin", node_pattern="linear"
        )
        cos_nodes = np.stack(
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
        sin_nodes = np.stack(
            [
                np.array([0, 1, 1, 1, 1, 1]),
                np.array(
                    [
                        0 + np.pi / 2,
                        0 + np.pi / 10,
                        2 * np.pi / 5 + np.pi / 10,
                        4 * np.pi / 5 + np.pi / 10,
                        6 * np.pi / 5 + np.pi / 10,
                        8 * np.pi / 5 + np.pi / 10,
                    ]
                ),
                np.zeros((int((M + 1) * (M + 2) / 2),)),
            ]
        ).T
        np.testing.assert_allclose(cos_grid.nodes, cos_nodes, atol=1e-8)
        np.testing.assert_allclose(sin_grid.nodes, sin_nodes, atol=1e-8)
        with pytest.raises(ValueError):
            tan_grid = ConcentricGrid(
                M,
                M,
                N,
                NFP,
                sym=False,
                axis=True,
                rotation="tan",
                node_pattern="linear",
            )

    def test_symmetry(self):
        """
        Test if symmetric grids alter dtheta spacing correctly.
        This is a necessary but not sufficient test.
        Should make sure that all test_compute_utils functions must also pass.
        """

        def test(grid, err_msg):
            area = surface_integrals(grid)
            np.testing.assert_allclose(area, 4 * np.pi ** 2, err_msg=err_msg)
            t = grid.nodes[:, 1]
            z = grid.nodes[:, 2] * grid.NFP
            f = (
                5
                + np.cos(t)
                - 0.5 * np.cos(z)
                + 3 * np.cos(t) * np.cos(z)
                - 2 * np.sin(z) * np.sin(t)
            )
            avg = surface_averages(grid, f)
            np.testing.assert_allclose(avg, 5, err_msg=err_msg)

        g1 = LinearGrid(
            L=3,
            M=6,
            N=3,
            NFP=3,
            sym=True,
        )
        g2 = QuadratureGrid(
            L=3,
            M=6,
            N=3,
            NFP=3,
        )
        g3 = ConcentricGrid(
            L=3,
            M=6,
            N=3,
            NFP=3,
            sym=True,
            rotation="cos",
        )
        grids = {g1: "LinearGrid", g2: "QuadratureGrid", g3: "ConcentricGrid"}
        for g, msg in grids.items():
            test(g, msg)
