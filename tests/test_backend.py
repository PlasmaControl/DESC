import unittest
import numpy as np

from desc.backend import put
from desc.utils import sign


class TestBackend(unittest.TestCase):
    """tests for backend functions"""

    def test_put(self):
        a = np.array([0, 0, 0])
        b = np.array([1, 2, 3])

        a = put(a, np.array([0, 1, 2]), np.array([1, 2, 3]))

        np.testing.assert_array_almost_equal(a, b)

    def test_sign(self):

        self.assertEqual(sign(4), 1)
        self.assertEqual(sign(0), 1)
        self.assertEqual(sign(-10.3), -1)
