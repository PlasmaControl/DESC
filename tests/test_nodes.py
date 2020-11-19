import unittest
import numpy as np
from desc.grid import get_nodes_pattern


class TestNodes(unittest.TestCase):
    """tests for nodes functions"""

    def test_get_nodes_pattern(self):

        M = 2
        N = 0
        NFP = 1

        nodes_ansi, vols_ansi = get_nodes_pattern(M, N, NFP, surfs='linear', index='ansi', axis=True)
        nodes_fringe, vols_fringe = get_nodes_pattern(M, N, NFP, surfs='linear', index='fringe', axis=True)

        ansi_nodes = np.stack([np.array([0, 0.5, 0.5, 1, 1, 1]),
                               np.array([0, 0, np.pi, 0, 2*np.pi/3, 4*np.pi/3]),
                               np.zeros((int((M+1)*(M+2)/2),))])
        fringe_nodes = np.stack([np.array([0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]),
                                 np.array([0, 0, 2*np.pi/3, 4*np.pi/3, 0, 2*np.pi/5, 4*np.pi/5, 6*np.pi/5, 8*np.pi/5]),
                                 np.zeros((int((M+1)**2),))])

        np.testing.assert_array_almost_equal(nodes_ansi, ansi_nodes)
        np.testing.assert_array_almost_equal(nodes_fringe, fringe_nodes)
        self.assertAlmostEqual(np.sum(vols_ansi[0]*vols_ansi[1]*vols_ansi[2]), (2*np.pi)**2/NFP)
        self.assertAlmostEqual(np.sum(vols_fringe[0]*vols_fringe[1]*vols_fringe[2]), (2*np.pi)**2/NFP)
