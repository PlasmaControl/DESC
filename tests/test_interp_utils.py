"""Test interpolation utilities."""

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyvander

from desc.integrals.interp_utils import polyder_vec, polyroot_vec, polyval_vec


class TestPolyUtils:
    """Test polynomial utilities used for local spline interpolation in integrals."""

    @pytest.mark.unit
    def test_polyroot_vec(self):
        """Test vectorized computation of cubic polynomial exact roots."""
        c = np.arange(-24, 24).reshape(4, 6, -1).transpose(-1, 1, 0)
        # Ensure broadcasting won't hide error in implementation.
        assert np.unique(c.shape).size == c.ndim

        k = np.broadcast_to(np.arange(c.shape[-2]), c.shape[:-1])
        # Now increase dimension so that shapes still broadcast, but stuff like
        # ``c[...,-1]-=k`` is not allowed because it grows the dimension of ``c``.
        # This is needed functionality in ``polyroot_vec`` that requires an awkward
        # loop to obtain if using jnp.vectorize.
        k = np.stack([k, k * 2 + 1])
        r = polyroot_vec(c, k, sort=True)

        for i in range(k.shape[0]):
            d = c.copy()
            d[..., -1] -= k[i]
            # np.roots cannot be vectorized because it strips leading zeros and
            # output shape is therefore dynamic.
            for idx in np.ndindex(d.shape[:-1]):
                np.testing.assert_allclose(
                    r[(i, *idx)],
                    np.sort(np.roots(d[idx])),
                    err_msg=f"Eigenvalue branch of polyroot_vec failed at {i, *idx}.",
                )

        # Now test analytic formula branch, Ensure it filters distinct roots,
        # and ensure zero coefficients don't bust computation due to singularities
        # in analytic formulae which are not present in iterative eigenvalue scheme.
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
        r = polyroot_vec(c, sort=True, distinct=True)
        for j in range(c.shape[0]):
            root = r[j][~np.isnan(r[j])]
            unique_root = np.unique(np.roots(c[j]))
            assert root.size == unique_root.size
            np.testing.assert_allclose(
                root,
                unique_root,
                err_msg=f"Analytic branch of polyroot_vec failed at {j}.",
            )
        c = np.array([0, 1, -1, -8, 12])
        r = polyroot_vec(c, sort=True, distinct=True)
        r = r[~np.isnan(r)]
        unique_r = np.unique(np.roots(c))
        assert r.size == unique_r.size
        np.testing.assert_allclose(r, unique_r)

    @pytest.mark.unit
    def test_polyder_vec(self):
        """Test vectorized computation of polynomial derivative."""
        c = np.arange(-18, 18).reshape(3, -1, 6)
        # Ensure broadcasting won't hide error in implementation.
        assert np.unique(c.shape).size == c.ndim
        np.testing.assert_allclose(
            polyder_vec(c),
            np.vectorize(np.polyder, signature="(m)->(n)")(c),
        )

    @pytest.mark.unit
    def test_polyval_vec(self):
        """Test vectorized computation of polynomial evaluation."""

        def test(x, c):
            # Ensure broadcasting won't hide error in implementation.
            assert np.unique(x.shape).size == x.ndim
            assert np.unique(c.shape).size == c.ndim
            np.testing.assert_allclose(
                polyval_vec(x=x, c=c),
                np.sum(polyvander(x, c.shape[-1] - 1) * c[..., ::-1], axis=-1),
            )

        c = np.arange(-60, 60).reshape(-1, 5, 3)
        x = np.linspace(0, 20, np.prod(c.shape[:-1])).reshape(c.shape[:-1])
        test(x, c)

        x = np.stack([x, x * 2], axis=0)
        x = np.stack([x, x * 2, x * 3, x * 4], axis=0)
        assert c.shape[:-1] == x.shape[x.ndim - (c.ndim - 1) :]
        assert np.unique((c.shape[-1],) + x.shape[c.ndim - 1 :]).size == x.ndim - 1
        test(x, c)
