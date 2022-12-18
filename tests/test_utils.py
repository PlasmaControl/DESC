"""Tests for utility functions."""

import numpy as np
import pytest

from desc.grid import LinearGrid
from desc.utils import isalmostequal, islinspaced


@pytest.mark.unit
def test_isalmostequal():
    """Test that isalmostequal function works on constants, 1D and larger arrays."""
    grid_small = LinearGrid(rho=1, M=1, N=10)
    _, zeta_cts = np.unique(grid_small.nodes[:, 2], return_counts=True)
    assert isalmostequal(
        grid_small.nodes[:, :2].T.reshape((2, zeta_cts[0], -1), order="F")
    )

    grid_large = LinearGrid(rho=1, M=1, N=100)
    _, zeta_cts = np.unique(grid_large.nodes[:, 2], return_counts=True)
    assert isalmostequal(
        grid_large.nodes[:, :2].T.reshape((2, zeta_cts[0], -1), order="F")
    )

    # 1D arrays
    assert isalmostequal(np.zeros(5))
    # 0D arrays will return True
    assert isalmostequal(np.array(0))


@pytest.mark.unit
def test_islinspaced():
    """Test that islinspaced function works on large arrays."""
    grid_small = LinearGrid(rho=1, M=1, N=10)
    zeta_vals = np.unique(grid_small.nodes[:, 2])
    assert islinspaced(zeta_vals)

    grid_large = LinearGrid(rho=1, M=1, N=100)
    zeta_vals = np.unique(grid_large.nodes[:, 2])
    assert islinspaced(zeta_vals)
