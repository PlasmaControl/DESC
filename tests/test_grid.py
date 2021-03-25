import unittest
import numpy as np
from scipy import special
from desc.grid import LinearGrid, ConcentricGrid, QuadratureGrid
from desc.equilibrium import Equilibrium


class TestGrid(unittest.TestCase):
    """Tests Grid classes"""

    def test_linear_grid(self):

        L = 3
        M = 3
        N = 3
        NFP = 1

        grid = LinearGrid(L, M, N, NFP, sym=False, axis=True, endpoint=False)

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
            M,
            N,
            NFP,
            sym=False,
            axis=True,
            spectral_indexing="ansi",
            node_pattern="linear",
        )
        grid_fringe = ConcentricGrid(
            M,
            N,
            NFP,
            sym=False,
            axis=True,
            spectral_indexing="fringe",
            node_pattern="linear",
        )

        ansi_nodes = np.stack(
            [
                np.array([0, 0.5, 0.5, 1, 1, 1]),
                np.array([0, 0, np.pi, 0, 2 * np.pi / 3, 4 * np.pi / 3]),
                np.zeros((int((M + 1) * (M + 2) / 2),)),
            ]
        ).T
        fringe_nodes = np.stack(
            [
                np.array([0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]),
                np.array(
                    [
                        0,
                        0,
                        2 * np.pi / 3,
                        4 * np.pi / 3,
                        0,
                        2 * np.pi / 5,
                        4 * np.pi / 5,
                        6 * np.pi / 5,
                        8 * np.pi / 5,
                    ]
                ),
                np.zeros((int((M + 1) ** 2),)),
            ]
        ).T

        np.testing.assert_allclose(grid_ansi.nodes, ansi_nodes, atol=1e-8)
        np.testing.assert_allclose(grid_fringe.nodes, fringe_nodes, atol=1e-8)

        self.assertAlmostEqual(np.sum(grid_ansi.weights), (2 * np.pi) ** 2 / NFP)
        self.assertAlmostEqual(np.sum(grid_fringe.weights), (2 * np.pi) ** 2 / NFP)

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
            "profiles": np.array([[0, 0, 0]]),
            "boundary": np.array([[0, 0, 0, R, 0], [0, 1, 0, r, 0], [0, -1, 0, 0, r]]),
            "spectral_indexing": "ansi",
            "bdry_mode": "lcfs",
            "node_pattern": "quad",
        }

        eq = Equilibrium(inputs)
        g = eq.compute_jacobian(eq.grid)
        vol_quad = np.sum(np.abs(g["g"]) * eq.grid.weights)

        self.assertAlmostEqual(vol, vol_quad)
