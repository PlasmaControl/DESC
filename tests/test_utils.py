"""Tests for utility functions."""

from functools import partial

import numpy as np
import pytest

from desc.backend import flatnonzero, jnp, tree_leaves, tree_structure
from desc.grid import LinearGrid
from desc.utils import broadcast_tree, isalmostequal, islinspaced, take_mask


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


@pytest.mark.unit
def test_broadcast_tree():
    """Test that broadcast_tree works on various pytree structures."""
    tree_out = [
        {"a": np.arange(1), "b": np.arange(2), "c": np.arange(3)},
        [
            {"a": np.arange(2)},
            [{"a": np.arange(1), "d": np.arange(3)}, {"a": np.arange(2)}],
        ],
    ]

    # tree with tuples, not lists
    tree_in = [{}, ({}, [{}, {}])]
    with pytest.raises(ValueError):
        _ = broadcast_tree(tree_in, tree_out)

    # tree_in is deeper than tree_out
    tree_in = [
        [{"a": np.arange(1)}, {"b": np.arange(2), "c": np.arange(3)}],
        [{}, [{}, {"a": np.arange(2)}]],
    ]
    with pytest.raises(ValueError):
        _ = broadcast_tree(tree_in, tree_out)

    # tree_in has different number of branches as tree_out
    tree_in = [{}, [{}, [{}]]]
    with pytest.raises(ValueError):
        _ = broadcast_tree(tree_in, tree_out)

    # tree with incorrect keys
    tree_in = [{"a": np.arange(1), "b": np.arange(2)}, {"d": np.arange(2)}]
    with pytest.raises(ValueError):
        _ = broadcast_tree(tree_in, tree_out)

    # tree with incorrect values
    tree_in = [{"a": np.arange(1), "b": np.arange(2)}, {"a": np.arange(2)}]
    with pytest.raises(ValueError):
        _ = broadcast_tree(tree_in, tree_out)

    # tree with proper structure already does not change
    tree_in = tree_out.copy()
    tree = broadcast_tree(tree_in, tree_out)
    assert tree_structure(tree) == tree_structure(tree_out)
    for leaf, leaf_correct in zip(tree_leaves(tree), tree_leaves(tree_out)):
        np.testing.assert_allclose(leaf, leaf_correct)

    # broadcast single leaf to full tree
    tree_in = {"a": np.arange(1)}
    tree = broadcast_tree(tree_in, tree_out)
    assert tree_structure(tree) == tree_structure(tree_out)
    tree_correct = [
        {"a": np.arange(1), "b": np.array([], dtype=int), "c": np.array([], dtype=int)},
        [
            {"a": np.arange(1)},
            [{"a": np.arange(1), "d": np.array([], dtype=int)}, {"a": np.arange(1)}],
        ],
    ]
    for leaf, leaf_correct in zip(tree_leaves(tree), tree_leaves(tree_correct)):
        np.testing.assert_allclose(leaf, leaf_correct)

    # broadcast from only major branches
    tree_in = [{"b": np.arange(2), "c": np.arange(1, 3)}, {"a": np.arange(1)}]
    tree = broadcast_tree(tree_in, tree_out)
    assert tree_structure(tree) == tree_structure(tree_out)
    tree_correct = [
        {"a": np.array([], dtype=int), "b": np.arange(2), "c": np.arange(1, 3)},
        [
            {"a": np.arange(1)},
            [{"a": np.arange(1), "d": np.array([], dtype=int)}, {"a": np.arange(1)}],
        ],
    ]
    for leaf, leaf_correct in zip(tree_leaves(tree), tree_leaves(tree_correct)):
        np.testing.assert_allclose(leaf, leaf_correct)

    # broadcast from minor branches
    tree_in = [
        {"b": np.arange(2), "c": np.arange(1, 3)},
        [{"a": np.arange(2)}, {"a": np.arange(1)}],
    ]
    tree = broadcast_tree(tree_in, tree_out)
    assert tree_structure(tree) == tree_structure(tree_out)
    tree_correct = [
        {"a": np.array([], dtype=int), "b": np.arange(2), "c": np.arange(1, 3)},
        [
            {"a": np.arange(2)},
            [{"a": np.arange(1), "d": np.array([], dtype=int)}, {"a": np.arange(1)}],
        ],
    ]
    for leaf, leaf_correct in zip(tree_leaves(tree), tree_leaves(tree_correct)):
        np.testing.assert_allclose(leaf, leaf_correct)

    # tree_in with empty dicts and arrays
    tree_in = [
        {},
        [
            {"a": np.array([], dtype=int)},
            [{"a": np.arange(1), "d": np.array([0, 2], dtype=int)}, {}],
        ],
    ]
    tree = broadcast_tree(tree_in, tree_out)
    assert tree_structure(tree) == tree_structure(tree_out)
    tree_correct = [
        {
            "a": np.array([], dtype=int),
            "b": np.array([], dtype=int),
            "c": np.array([], dtype=int),
        },
        [
            {"a": np.array([], dtype=int)},
            [
                {"a": np.arange(1), "d": np.array([0, 2], dtype=int)},
                {"a": np.array([], dtype=int)},
            ],
        ],
    ]
    for leaf, leaf_correct in zip(tree_leaves(tree), tree_leaves(tree_correct)):
        np.testing.assert_allclose(leaf, leaf_correct)

    # tree_in with bool values
    tree_in = [
        {"a": False, "b": True, "c": np.array([0, 2], dtype=int)},
        [
            {"a": True},
            [{"a": False, "d": np.arange(2)}, {"a": True}],
        ],
    ]
    tree = broadcast_tree(tree_in, tree_out)
    assert tree_structure(tree) == tree_structure(tree_out)
    tree_correct = [
        {
            "a": np.array([], dtype=int),
            "b": np.arange(2),
            "c": np.array([0, 2], dtype=int),
        },
        [
            {"a": np.arange(2)},
            [{"a": np.array([], dtype=int), "d": np.arange(2)}, {"a": np.arange(2)}],
        ],
    ]
    for leaf, leaf_correct in zip(tree_leaves(tree), tree_leaves(tree_correct)):
        np.testing.assert_allclose(leaf, leaf_correct)


@partial(jnp.vectorize, signature="(m)->()")
def _last_value(a):
    """Return the last non-nan value in ``a``."""
    a = a[::-1]
    idx = jnp.squeeze(flatnonzero(~jnp.isnan(a), size=1, fill_value=0))
    return a[idx]


@pytest.mark.unit
def test_take_mask():
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
