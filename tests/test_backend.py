import unittest
import numpy as np

from desc.backend import factorial, put
from desc.utils import sign


class TestBackend(unittest.TestCase):
    """tests for backend functions"""

    def test_factorial(self):
        self.assertAlmostEqual(factorial(10), 3628800)
        self.assertAlmostEqual(factorial(0), 1)

    def test_put(self):
        a = np.array([0, 0, 0])
        b = np.array([1, 2, 3])

        a = put(a, [0, 1, 2], [1, 2, 3])

        np.testing.assert_array_almost_equal(a, b)

    def test_sign(self):

        self.assertEqual(sign(4), 1)
        self.assertEqual(sign(0), 1)
        self.assertEqual(sign(-10.3), -1)
