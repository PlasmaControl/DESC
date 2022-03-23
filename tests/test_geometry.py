import numpy as np
from desc.geometry.utils import (
    rotation_matrix,
    xyz2rpz,
    xyz2rpz_vec,
    rpz2xyz,
    rpz2xyz_vec,
)


def test_rotation_matrix():
    A = rotation_matrix([0, 0, np.pi / 2])
    At = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(A, At, atol=1e-10)


def test_xyz2rpz():
    xyz = np.array([1, 1, 1])
    rpz = xyz2rpz(xyz)
    np.testing.assert_allclose(rpz, [np.sqrt(2), np.pi / 4, 1], atol=1e-10)

    xyz = np.array([0, 1, 1])
    rpz = xyz2rpz_vec(xyz, x=0, y=1)
    np.testing.assert_allclose(rpz, np.array([[1, 0, 1]]), atol=1e-10)


def test_rpz2xyz():
    rpz = np.array([np.sqrt(2), np.pi / 4, 1])
    xyz = rpz2xyz(rpz)
    np.testing.assert_allclose(xyz, [1, 1, 1], atol=1e-10)

    rpz = np.array([[1, 0, 1]])
    xyz = rpz2xyz_vec(rpz, x=0, y=1)
    np.testing.assert_allclose(xyz, np.array([[0, 1, 1]]), atol=1e-10)
