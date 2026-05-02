"""Test interpolation utilities."""

from functools import partial

import numpy as np
import pytest
from jax import grad
from matplotlib.colors import LogNorm
from numpy.polynomial.polynomial import polyvander
from tests.test_plotting import tol_2d

from desc.backend import jnp, rfft, rfft2
from desc.examples import get
from desc.integrals import Bounce2D
from desc.integrals._interp_utils import nufft1d2r, nufft2d2r, poly_val, polyroot_vec


def _c_1d(x):
    """Test function for 1D FFT."""
    return jnp.cos(7 * x) + jnp.sin(x) - 33.2


def _c_1d_nyquist_freq():
    return 7


def _c_2d(x, y):
    """Test function for 2D FFT."""
    x_freq, y_freq = 3, 5
    return (
        # something that's not separable
        jnp.cos(x_freq * x) * jnp.sin(2 * x + y)
        + jnp.sin(y_freq * y) * jnp.cos(x + 3 * y)
        - 33.2
        + jnp.cos(x)
        + jnp.cos(y)
    )


def _c_2d_nyquist_freq():
    x_freq, y_freq = 3, 5
    x_freq_nyquist = x_freq + 2
    y_freq_nyquist = y_freq + 3
    return x_freq_nyquist, y_freq_nyquist


def _f_non_periodic(z):
    return np.sin(np.sqrt(2) * z) * np.cos(1 / (2 + z)) * np.cos(z**2) * z


def _f_algebraic(z):
    return z**3 - 10 * z**6 - z - np.e + z**4


_test_inputs_1D = [
    (_c_1d, 2 * _c_1d_nyquist_freq() + 1, (0, 2 * jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq(), (0, 2 * jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq() + 1, (-jnp.pi, jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq(), (-jnp.pi, jnp.pi)),
    (lambda x: jnp.cos(7 * x), 2, (-jnp.pi / 7, jnp.pi / 7)),
    (lambda x: jnp.sin(7 * x), 3, (-jnp.pi / 7, jnp.pi / 7)),
]

_test_inputs_2D = [
    (
        _c_2d,
        2 * _c_2d_nyquist_freq()[0] + 1,
        2 * _c_2d_nyquist_freq()[1] + 1,
        (0, 2 * jnp.pi),
        (0, 2 * jnp.pi),
    ),
    (
        _c_2d,
        2 * _c_2d_nyquist_freq()[0] + 1,
        2 * _c_2d_nyquist_freq()[1] + 1,
        (-jnp.pi / 3, 5 * jnp.pi / 3),
        (jnp.pi, 3 * jnp.pi),
    ),
    (
        lambda x, y: jnp.cos(30 * x) + jnp.sin(y) ** 2 + 1,
        2 * 30 // 30 + 1,
        2 * 2 + 1,
        (0, 2 * jnp.pi / 30),
        (jnp.pi, 3 * jnp.pi),
    ),
]


class TestFastInterp:
    """Test fast (non-uniform) FFT and partial sum interpolation."""

    @pytest.mark.unit
    @pytest.mark.parametrize("func, n, domain", _test_inputs_1D)
    def test_non_uniform_real_FFT(self, func, n, domain):
        """Test non-uniform real FFT interpolation."""
        x = jnp.linspace(domain[0], domain[1], n, endpoint=False)
        c = func(x)
        xq = jnp.array([7.34, 1.10134, 2.28])

        f = 2 * rfft(c, norm="forward")
        f = f.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)
        np.testing.assert_allclose(nufft1d2r(xq, f, domain), func(xq))

        @grad
        def g(xq):
            return nufft1d2r(xq, f, domain, eps=1e-7).sum()

        @grad
        def true_g(xq):
            return func(xq).sum()

        np.testing.assert_allclose(g(xq), true_g(xq))

    @pytest.mark.unit
    @pytest.mark.parametrize("func, m, n, domain_x, domain_y", _test_inputs_2D)
    def test_non_uniform_real_FFT_2D(self, func, m, n, domain_x, domain_y):
        """Test non-uniform real FFT 2D interpolation."""
        x = jnp.linspace(domain_x[0], domain_x[1], m, endpoint=False)
        y = jnp.linspace(domain_y[0], domain_y[1], n, endpoint=False)
        x, y = jnp.meshgrid(x, y, indexing="ij")
        c = func(x, y)

        xq = jnp.array([7.34, 1.10134, 2.28, 1e3 * np.e])
        yq = jnp.array([1.1, 3.78432, 8.542, 0])

        f1 = 2 * rfft2(c, norm="forward")
        f1 = f1.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)
        f2 = jnp.fft.fft2(c, norm="forward")

        v = func(xq, yq)
        np.testing.assert_allclose(
            nufft2d2r(xq, yq, f1, domain_x, domain_y), v, rtol=1e-6
        )
        np.testing.assert_allclose(
            nufft2d2r(xq, yq, f2, domain_x, domain_y, None), v, rtol=1e-6
        )

        @partial(grad, argnums=(0, 1))
        def g1(xq, yq):
            return nufft2d2r(xq, yq, f1, domain_x, domain_y, eps=1e-8).sum()

        @partial(grad, argnums=(0, 1))
        def g2(xq, yq):
            return nufft2d2r(xq, yq, f2, domain_x, domain_y, None, eps=1e-8).sum()

        @partial(grad, argnums=(0, 1))
        def true_g(xq, yq):
            return func(xq, yq).sum()

        g = true_g(xq, yq)
        np.testing.assert_allclose(g1(xq, yq), g, atol=1e-10)
        np.testing.assert_allclose(g2(xq, yq), g, atol=1e-10)

    @pytest.mark.unit
    def test_nufft2_vec(self):
        """Test vectorized JAX-finufft vectorized interpolation."""
        func_1, n, domain = _test_inputs_1D[0]
        func_2 = lambda x: -77 * np.sin(7 * x) + 18 * np.cos(x) + 100  # noqa: E731
        x = np.linspace(domain[0], domain[1], n, endpoint=False)
        c = np.stack([func_2(x), func_2(x)])

        f = 2 * rfft(c, norm="forward")
        f = f.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)

        # multiple (2) fourier series evaluated at the same (3) points
        xq = np.array([7.34, 1.10134, 2.28])
        np.testing.assert_allclose(
            nufft1d2r(xq, f, domain),
            np.stack([func_2(xq), func_2(xq)]),
        )

        # batch with shape (1, 4)
        xq = np.stack([xq, xq**2, xq**3, xq**4])[np.newaxis]
        f = np.stack([f, -f, 2 * f, 3 * f])[np.newaxis]

        # vectorized over batch with shape (1, 4),
        # multiple (2) fourier series evaluated at the same (3) points
        np.testing.assert_allclose(
            nufft1d2r(xq, f, domain, vec=True),
            np.vectorize(
                partial(nufft1d2r, domain=domain, vec=False),
                signature="(x),(b,f)->(b,x)",
            )(xq, f),
        )


class TestStreams:
    """Test convergence of inverse stream maps."""

    tol = 1e-8
    norm = LogNorm(1e-7, 1e0)
    X = 48
    Y = 48
    rho = 0.6

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ["W7-X", "NCSX", "HELIOTRON"])
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    @staticmethod
    def test_delta_fourier_chebyshev(name):
        """Plot Fourier-Chebyshev spectrum of δ(α, ζ)."""
        eq = get(name)
        X = TestStreams.X
        Y = TestStreams.Y
        angle = Bounce2D.angle(eq, X, Y, TestStreams.rho, tol=TestStreams.tol)
        return Bounce2D.plot_angle_spectrum(angle, 0, norm=TestStreams.norm)

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ["W7-X", "NCSX", "HELIOTRON"])
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    @staticmethod
    def test_lambda_fourier_vartheta_zeta(name):
        """Plot Fourier spectrum of Λ(ϑ, ζ)."""
        eq = get(name)
        X = TestStreams.X
        Y = TestStreams.Y
        angle = Bounce2D.angle(
            eq,
            X,
            Y,
            TestStreams.rho,
            tol=TestStreams.tol,
            name="lambda",
            ignore_lambda_guard=True,
        )
        return Bounce2D.plot_angle_spectrum(
            angle, 0, name="lambda", norm=TestStreams.norm
        )


# TODO(#1388)
class TestPolyUtils:
    """Test polynomial utilities used for local spline interpolation."""

    @pytest.mark.unit
    def test_polyroot_vec_general(self):
        """Test vectorized computation of cubic polynomial exact roots."""
        c = np.arange(-24, 24).reshape(4, 6, -1).transpose(-1, 1, 0)
        # Ensure broadcasting won't hide error in implementation.
        assert np.unique(c.shape).size == c.ndim

        k = np.broadcast_to(np.arange(c.shape[-2]), c.shape[:-1])
        # Now increase dimension so that shapes still broadcast, but stuff like
        # c[...,-1]-=k is not allowed because it grows the dimension of c.
        # This is needed functionality in polyroot_vec that requires an awkward
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

    @pytest.mark.unit
    @pytest.mark.parametrize("companion", [True, False])
    def test_polyroot_vec_distinct(self, companion):
        """Test vectorized computation of cubic polynomial exact roots."""
        a_min = -jnp.inf
        a_max = +jnp.inf

        # coefficient of poly x^3, .... x^0.
        c = np.array(
            [
                [1, 0, 0, 0],  # triple root (0, 0, 0)
                [0, 1, 0, 0],  # double root (0, 0)
                [0, 0, 1, 0],  # single root 0
                [0, 0, 0, 1],  # no roots
                [1, -1, -8, 12],  # (-3, 2, 2)
                [1, -6, 11, -6],  # (1, 2, 3)
                [0, -6, 11, -2],  # 2 distinct irrational
            ]
        )
        if companion:
            c = np.concat([np.zeros((c.shape[0], 1)), c], axis=-1)
            assert c.shape == (7, 5)

        # acceptable outputs for multiple roots are those which are
        # consistent with continuity invariant. That is if user requests
        # distinct roots only, then acceptable output is
        #   1) If sign(derivative) == 0, should return only one of the roots.
        #   2) If sign(derivative) != 0 and changes across roots, then you can
        #      return both since even if they were very close and supposed to
        #      be the same root in infinite precision, they are distinct roots
        #      in finite precision, so invariants are preserved by returning both.a_max
        #   3) If sign(derivative) != 0 and same across roots. This is not possible
        #      for continuous functions and it means that numerical error has split
        #      single root into multiple. If the true root is a single root,
        #      we should return just one of the roots. If the true root has multiplicity
        #      greater than one, we should not discard the pairs with same sign.
        r = polyroot_vec(c, a_min=a_min, a_max=a_max, sort=True, distinct=True)
        for j in range(c.shape[0]):
            root = r[j][~np.isnan(r[j])]
            unique_root = np.unique(np.roots(c[j]).real)
            try:
                np.testing.assert_allclose(
                    root,
                    unique_root,
                    err_msg=f"Companion={companion} branch failed at {j}.",
                )
                assert root.size == unique_root.size
            except AssertionError as e:
                if j == 0 or j == 1:
                    assert root.size == 0 or np.allclose(root, 0)
                elif j == 4:
                    if root.size == 3:
                        # ok to keep both double roots here because they are
                        # one red and one green since they lie on opposite
                        # sides of the true double root.
                        assert np.allclose(root, (-3, 2, 2))
                    else:
                        # also okay to remove both
                        assert root.size == 1 and np.allclose(root, -3)
                else:
                    assert False, e

        c = np.array([0, 1, -1, -8, 12])
        r = polyroot_vec(c, a_min=a_min, a_max=a_max, sort=True, distinct=True)
        r = r[~np.isnan(r)]
        unique_r = np.unique(np.roots(c))
        assert r.size == unique_r.size
        np.testing.assert_allclose(r, unique_r)

    @pytest.mark.unit
    @pytest.mark.parametrize("der", [0, 1])
    def test_cubic_val(self, der):
        """Test vectorized computation of polynomial evaluation."""

        def test(x, c):
            # Ensure broadcasting won't hide error in implementation.
            assert np.unique(x.shape).size == x.ndim
            assert np.unique(c.shape).size == c.ndim
            dc = np.vectorize(np.polyder, signature="(m)->(n)")(c) if der else c
            np.testing.assert_allclose(
                poly_val(x=x, c=c, der=der),
                np.sum(polyvander(x, dc.shape[-1] - 1) * dc[..., ::-1], axis=-1),
            )

        c = np.arange(-60, 60).reshape(-1, 5, 4)
        x = np.linspace(0, 20, np.prod(c.shape[:-1])).reshape(c.shape[:-1])
        test(x, c)

        x = np.stack([x, x * 2], axis=0)
        x = np.stack([x, x * 2, x * 3, x * 4], axis=0)
        assert c.shape[:-1] == x.shape[x.ndim - (c.ndim - 1) :]
        assert np.unique((c.shape[-1],) + x.shape[c.ndim - 1 :]).size == x.ndim - 1
        test(x, c)
