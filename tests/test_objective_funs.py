import unittest
import numpy as np
from desc.objective_funs import is_nested, curve_self_intersects
from desc.transform import get_zern_basis_idx_dense


class TestIsNested(unittest.TestCase):
    """tests for  functions"""

    def test_is_nested(self):
        zidx = get_zern_basis_idx_dense(2, 0)
        cR1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
        cZ1 = np.array([0, 0, -1, 0, 0, 0, 0, 0, 0])
        cR2 = np.array([0, 1, 0, 0, 0, 0, 5, 0, 0])
        cZ2 = np.array([0, 0, -1, 0, 0, 4, 0, 0, 0])
        self.assertTrue(is_nested(cR1, cZ1, zidx, 1))
        self.assertFalse(is_nested(cR2, cZ2, zidx, 1))

    def test_self_intersection(self):

        # elipse: not self intersected

        a = 1
        b = 1
        d = np.pi/2
        t = np.linspace(0, 2*np.pi, 361)
        x = np.sin(a*t+d)
        y = np.sin(b*t)
        self.assertFalse(curve_self_intersects(x, y))

        # lissajois: is self intersected
        a = 1
        b = 2
        d = np.pi/2
        t = np.linspace(0, 2*np.pi, 361)
        x = np.sin(a*t+d)
        y = np.sin(b*t)
        self.assertTrue(curve_self_intersects(x, y))
