"""Tests for geometry util functions for converting coordinates."""

import numpy as np
import pytest

from desc.compute.geom_utils import (
    rotation_matrix,
    rotation_matrix_vector_vector,
    rpz2xyz,
    rpz2xyz_vec,
    xyz2rpz,
    xyz2rpz_vec,
)


@pytest.mark.unit
def test_rotation_matrix():
    """Test calculation of rotation matrices."""

    def test1(axis, expected=None):

        A = rotation_matrix(axis)
        At = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(A, At, atol=1e-10)
        # ensure is an orthonormal matrix
        np.testing.assert_allclose(np.linalg.det(A), 1.0, atol=1e-10)
        np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-10)
        if expected is not None:
            np.testing.assert_allclose(A, expected, atol=1e-10)

    def test2(a, b, expected=None):
        A = rotation_matrix_vector_vector(a, b)
        # ensure is an orthonormal matrix
        np.testing.assert_allclose(np.linalg.det(A), 1.0, atol=1e-12)
        np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-12)
        if expected is not None:
            np.testing.assert_allclose(A, expected, atol=1e-12)
        return A

    At = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    test1([0, 0, np.pi / 2], At)
    test2([1, 0, 0], [0, 1, 0], At)

    # for parallel vectors, should return identity
    test2([1, 0, 0], [1, 0, 0], np.eye(3))
    # for antiparallel vectors, should return np.diag([1,-1,-1])
    A = test2([1, 0, 0], [-1, 0, 0], np.diag(np.array([1, -1, -1])))
    np.testing.assert_allclose(A @ np.array([0, 1, 0]), np.array([0, -1, 0]), atol=1e-8)

    # check for nearly-parallel vectors
    a = np.array([1, 0, 0])
    b = np.array([1, 1e-4, 0])
    A = test2(a, b)
    c = np.array([1, 1, 1])
    angle = np.arccos(np.dot(a, b) / np.linalg.norm(b))
    norm_xy = np.linalg.norm(c[:2])
    # rotation is just of the component in the xy-plane,
    # about the Z-axis by angle. can check this with simple trig
    c_rot = np.array(
        [norm_xy * np.cos(angle + np.pi / 4), norm_xy * np.sin(angle + np.pi / 4), 1]
    )
    np.testing.assert_allclose(A @ c, c_rot, atol=1e-8)
    np.testing.assert_allclose(A @ a, b / np.linalg.norm(b), atol=1e-8)
    # check for nearly-anti-parallel vectors
    a = np.array([1, 0, 0])
    b = np.array([-1, 1e-4, 0])
    A = test2(a, b)
    c = np.array([1, 1, 1])
    angle = np.arccos(np.dot(a, b) / np.linalg.norm(b))
    norm_xy = np.linalg.norm(c[:2])
    # rotation is just of the component in the xy-plane,
    # about the Z-axis by angle. can check this with simple trig
    c_rot = np.array(
        [norm_xy * np.cos(angle + np.pi / 4), norm_xy * np.sin(angle + np.pi / 4), 1]
    )
    np.testing.assert_allclose(A @ c, c_rot, atol=1e-8)
    np.testing.assert_allclose(A @ a, b / np.linalg.norm(b), atol=1e-8)


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
