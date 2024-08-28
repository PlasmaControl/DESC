"""Test interpolation utilities."""

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyvander

from desc.integrals.interp_utils import poly_root, polyder_vec, polyval_vec


class TestPolyUtils:
    """Test polynomial stuff used for local spline interpolation."""

    @pytest.mark.unit
    def test_poly_root(self):
        """Test vectorized computation of cubic polynomial exact roots."""
        cubic = 4
        c = np.arange(-24, 24).reshape(cubic, 6, -1) * np.pi
        # make sure broadcasting won't hide error in implementation
        assert np.unique(c.shape).size == c.ndim
        constant = np.broadcast_to(np.arange(c.shape[-1]), c.shape[1:])
        constant = np.stack([constant, constant])
        root = poly_root(c, constant, sort=True)

        for i in range(constant.shape[0]):
            for j in range(c.shape[1]):
                for k in range(c.shape[2]):
                    d = c[-1, j, k] - constant[i, j, k]
                    np.testing.assert_allclose(
                        actual=root[i, j, k],
                        desired=np.sort(np.roots([*c[:-1, j, k], d])),
                    )

        c = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, -1, -8, 12],
                [1, -6, 11, -6],
                [0, -6, 11, -2],
            ]
        )
        root = poly_root(c.T, sort=True, distinct=True)
        for j in range(c.shape[0]):
            unique_roots = np.unique(np.roots(c[j]))
            np.testing.assert_allclose(
                actual=root[j][~np.isnan(root[j])], desired=unique_roots, err_msg=str(j)
            )
        c = np.array([0, 1, -1, -8, 12])
        root = poly_root(c, sort=True, distinct=True)
        root = root[~np.isnan(root)]
        unique_root = np.unique(np.roots(c))
        assert root.size == unique_root.size
        np.testing.assert_allclose(root, unique_root)

    @pytest.mark.unit
    def test_polyder_vec(self):
        """Test vectorized computation of polynomial derivative."""
        quintic = 6
        c = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
        # make sure broadcasting won't hide error in implementation
        assert np.unique(c.shape).size == c.ndim
        derivative = polyder_vec(c)
        desired = np.vectorize(np.polyder, signature="(m)->(n)")(c.T).T
        np.testing.assert_allclose(derivative, desired)

    @pytest.mark.unit
    def test_polyval_vec(self):
        """Test vectorized computation of polynomial evaluation."""

        def test(x, c):
            np.testing.assert_allclose(
                polyval_vec(x=x, c=c),
                np.sum(
                    polyvander(x, c.shape[0] - 1) * np.moveaxis(np.flipud(c), 0, -1),
                    axis=-1,
                ),
            )

        quartic = 5
        c = np.arange(-60, 60).reshape(quartic, 3, -1) * np.pi
        # make sure broadcasting won't hide error in implementation
        assert np.unique(c.shape).size == c.ndim
        x = np.linspace(0, 20, c.shape[1] * c.shape[2]).reshape(c.shape[1], c.shape[2])
        test(x, c)

        x = np.stack([x, x * 2], axis=0)
        x = np.stack([x, x * 2, x * 3, x * 4], axis=0)
        # make sure broadcasting won't hide error in implementation
        assert np.unique(x.shape).size == x.ndim
        assert c.shape[1:] == x.shape[x.ndim - (c.ndim - 1) :]
        assert np.unique((c.shape[0],) + x.shape[c.ndim - 1 :]).size == x.ndim - 1
        test(x, c)
