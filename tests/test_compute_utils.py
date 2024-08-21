"""Tests compute utilities."""

from functools import partial

import jax
import numpy as np
import pytest

from desc.backend import flatnonzero, jnp
from desc.compute.geom_utils import rotation_matrix
from desc.utils import take_mask


@pytest.mark.unit
def test_rotation_matrix():
    """Test that rotation_matrix works with fwd & rev AD for axis=[0, 0, 0]."""
    dfdx_fwd = jax.jacfwd(rotation_matrix)
    dfdx_rev = jax.jacrev(rotation_matrix)
    x0 = jnp.array([0.0, 0.0, 0.0])

    np.testing.assert_allclose(rotation_matrix(x0), np.eye(3))
    np.testing.assert_allclose(dfdx_fwd(x0), np.zeros((3, 3, 3)))
    np.testing.assert_allclose(dfdx_rev(x0), np.zeros((3, 3, 3)))


@partial(jnp.vectorize, signature="(m)->()")
def _last_value(a):
    """Return the last non-nan value in ``a``."""
    a = a[::-1]
    idx = np.squeeze(flatnonzero(~np.isnan(a), size=1, fill_value=0))
    return a[idx]


@pytest.mark.unit
def test_mask_operations():
    """Test custom masked array operation."""
    rows = 5
    cols = 7
    a = np.random.rand(rows, cols)
    nan_idx = np.random.choice(rows * cols, size=(rows * cols) // 2, replace=False)
    a.ravel()[nan_idx] = np.nan
    taken = take_mask(a, ~np.isnan(a))
    last = _last_value(taken)
    for i in range(rows):
        desired = a[i, ~np.isnan(a[i])]
        assert np.array_equal(
            taken[i],
            np.pad(desired, (0, cols - desired.size), constant_values=np.nan),
            equal_nan=True,
        )
        assert np.array_equal(
            last[i],
            desired[-1] if desired.size else np.nan,
            equal_nan=True,
        )
