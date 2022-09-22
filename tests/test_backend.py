import pytest
import numpy as np

from desc.backend import put, sign


@pytest.mark.fast
class TestBackend:
    """tests for backend functions"""

    def test_put(self):
        a = np.array([0, 0, 0])
        b = np.array([1, 2, 3])

        a = put(a, np.array([0, 1, 2]), np.array([1, 2, 3]))

        np.testing.assert_array_almost_equal(a, b)

    def test_sign(self):
        assert sign(4) == 1
        assert sign(0) == 1
        assert sign(-10.3) == -1
