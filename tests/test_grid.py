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
        M = 4
        N = 1
        NFP = 1

        grid_quad = QuadratureGrid(L, M, N, NFP, sym=False)

        roots, weights = special.js_roots(L, 2, 2)

        quadrature_nodes = np.stack(
            [
                np.array([roots[0]] * 4 + [roots[1]] * 4),
                np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2] * 2),
                np.zeros((2 * 2 * (L),)),
            ]
        ).T

        np.testing.assert_allclose(grid_quad.nodes, quadrature_nodes, atol=1e-8)
    def test_quad_grid_volume_integration(self):

        inputs = {
            "L": 4,
            "M": 2,
            "N": 0,
            "NFP": 1,
            "Psi": 1.0,
            "profiles": np.array([[0, 0, 0.23]]),
            "boundary": np.array([[0, 0, 0, 10, 0], [0, 1, 0, 1, 0]]),
            "index": "fringe",
        }
        
        r = 1
        R = 10
        
        eq1 = Equilibrium(inputs)
        eq1.R_lmn = np.array([R, 1, 0, 0, 0, 0, 0, 0, 0])
        eq1.Z_lmn = np.array([0, 0, -1, 0, 0, 0, 0, 0, 0])
        
        vol = 2*(np.pi**2)*(r**2)*R
        
        L = eq1.M + 1
        M = 2*eq1.M + 2
        N = 1
        NFP = 1
        
        grid_quad = QuadratureGrid(L, M, N, NFP, sym=False)
        
        g = eq1.compute_jacobian(grid_quad)
        
        vol_quad = np.sum(np.abs(g['g']) * grid_quad.weights)
        
        self.assertAlmostEqual(vol, vol_quad)
