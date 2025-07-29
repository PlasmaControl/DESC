"""Tests for geometry util functions for converting coordinates."""

import numpy as np
import pytest

from desc.utils import rotation_matrix, rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec


@pytest.mark.unit
def test_rotation_matrix():
    """Test calculation of rotation matrices."""
    A = rotation_matrix([0, 0, np.pi / 2])
    At = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(A, At, atol=1e-10)


@pytest.mark.unit
def test_xyz2rpz():
    """Test converting between cartesian and polar coordinates."""
    xyz = np.array([1, 1, 1])
    rpz = xyz2rpz(xyz)
    np.testing.assert_allclose(rpz, [np.sqrt(2), np.pi / 4, 1], atol=1e-10)

    xyz = np.array([0, 1, 1])
    rpz = xyz2rpz_vec(xyz, x=0, y=1)
    np.testing.assert_allclose(rpz, np.array([1, 0, 1]), atol=1e-10)


@pytest.mark.unit
def test_rpz2xyz():
    """Test converting between polar and cartesian coordinates."""
    rpz = np.array([np.sqrt(2), np.pi / 4, 1])
    xyz = rpz2xyz(rpz)
    np.testing.assert_allclose(xyz, [1, 1, 1], atol=1e-10)

    rpz = np.array([[1, 0, 1]])
    xyz = rpz2xyz_vec(rpz, x=0, y=1)
    np.testing.assert_allclose(xyz, np.array([[0, 1, 1]]), atol=1e-10)
