"""Tests for utility functions."""

import numpy as np
import pytest

from desc.grid import LinearGrid
from desc.utils import isalmostequal, islinspaced


@pytest.mark.unit
def test_isalmostequal():
    """Test that isalmostequal function works on constants, 1D and larger arrays."""
    grid_small = LinearGrid(rho=1, M=1, N=10)
    zeta_cts = grid_small.num_zeta
    assert isalmostequal(
        grid_small.nodes[:, :2].T.reshape((2, zeta_cts, -1), order="F")
    )

    grid_large = LinearGrid(rho=1, M=1, N=100)
    zeta_cts = grid_large.num_zeta
    assert isalmostequal(
        grid_large.nodes[:, :2].T.reshape((2, zeta_cts, -1), order="F")
    )

    # along axis other than -1
    arr = np.array([[1, 2], [3, 4]])
    newarr = np.dstack([arr] * 2)
    newarr[:, 0, :] = newarr[:, 1, :]
    assert isalmostequal(newarr, axis=1)
    assert not isalmostequal(newarr, axis=0)

    # 1D arrays
    assert isalmostequal(np.zeros(5))
    # 0D arrays will return True
    assert isalmostequal(np.array(0))


@pytest.mark.unit
def test_islinspaced():
    """Test that islinspaced function works on large arrays."""
    grid_small = LinearGrid(rho=1, M=1, N=10)
    zeta_vals = grid_small.nodes[grid_small.unique_zeta_idx, 2]
    assert islinspaced(zeta_vals)

    grid_large = LinearGrid(rho=1, M=1, N=100)
    zeta_vals = grid_large.nodes[grid_large.unique_zeta_idx, 2]
    assert islinspaced(zeta_vals)

    # on a 2D array
    zz = np.tile(zeta_vals, (2, 1))
    zz[1, :] *= 2
    assert islinspaced(zz)

    # 0D arrays will return True
    assert islinspaced(np.array(0))
