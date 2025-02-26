"""Test interpolation utilities."""

import numpy as np
import pytest
from numpy.polynomial.chebyshev import (
    cheb2poly,
    chebinterpolate,
    chebpts1,
    chebpts2,
    chebval,
)
from numpy.polynomial.polynomial import polyvander
from scipy.fft import dct as sdct
from scipy.fft import idct as sidct

from desc.backend import dct, idct, rfft
from desc.integrals._interp_utils import (
    cheb_from_dct,
    cheb_pts,
    fourier_pts,
    interp_dct,
    interp_rfft,
    interp_rfft2,
    polyder_vec,
    polyroot_vec,
    polyval_vec,
    rfft_to_trig,
    trig_vander,
)
from desc.integrals.basis import FourierChebyshevSeries
from desc.integrals.quad_utils import bijection_to_disc


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


def _f_1d(x):
    """Test function for 1D FFT."""
    return np.cos(7 * x) + np.sin(x) - 33.2


def _f_1d_nyquist_freq():
    return 7


def _f_2d(x, y):
    """Test function for 2D FFT."""
    x_freq, y_freq = 3, 5
    return (
        # something that's not separable
        np.cos(x_freq * x) * np.sin(2 * x + y)
        + np.sin(y_freq * y) * np.cos(x + 3 * y)
        # DC terms
        - 33.2
        + np.cos(x)
        + np.cos(y)
    )


def _f_2d_nyquist_freq():
    # can just sum frequencies multiplied above thanks to fourier
    x_freq, y_freq = 3, 5
    x_freq_nyquist = x_freq + 2
    y_freq_nyquist = y_freq + 3
    return x_freq_nyquist, y_freq_nyquist


def _identity(x):
    return x


def _f_non_periodic(z):
    return np.sin(np.sqrt(2) * z) * np.cos(1 / (2 + z)) * np.cos(z**2) * z


def _f_algebraic(z):
    return z**3 - 10 * z**6 - z - np.e + z**4


class TestFastInterp:
    """Test fast interpolation."""

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 6, 7])
    def test_cheb_pts(self, N):
        """Test we use Chebyshev points compatible with standard definition of DCT."""
        np.testing.assert_allclose(cheb_pts(N), chebpts1(N)[::-1], atol=1e-15)
        np.testing.assert_allclose(
            cheb_pts(N, domain=(-np.pi, np.pi), lobatto=True),
            np.pi * chebpts2(N)[::-1],
            atol=1e-15,
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("M", [1, 8, 9])
    def test_rfftfreq(self, M):
        """Make sure numpy uses Nyquist interpolant frequencies."""
        np.testing.assert_allclose(np.fft.rfftfreq(M, d=1 / M), np.arange(M // 2 + 1))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "func, n, domain",
        [
            # Test cases chosen with purpose, don't remove any.
            (_f_1d, 2 * _f_1d_nyquist_freq() + 1, (0, 2 * np.pi)),
            (_f_1d, 2 * _f_1d_nyquist_freq(), (0, 2 * np.pi)),
            (_f_1d, 2 * _f_1d_nyquist_freq() + 1, (-np.pi, np.pi)),
            (_f_1d, 2 * _f_1d_nyquist_freq(), (-np.pi, np.pi)),
            (lambda x: np.cos(7 * x), 2, (-np.pi / 7, np.pi / 7)),
            (lambda x: np.sin(7 * x), 3, (-np.pi / 7, np.pi / 7)),
        ],
    )
    def test_interp_rfft(self, func, n, domain):
        """Test non-uniform FFT interpolation."""
        x = np.linspace(domain[0], domain[1], n, endpoint=False)
        f = func(x)
        xq = np.array([7.34, 1.10134, 2.28])
        fq = func(xq)
        np.testing.assert_allclose(interp_rfft(xq, f, domain), fq)
        M = f.shape[-1]
        coef = rfft_to_trig(rfft(f, norm="forward"), M)
        vander = trig_vander(xq, M, domain)
        np.testing.assert_allclose((vander * coef).sum(axis=-1), fq)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "func, m, n, domain0, domain1",
        [
            # Test cases chosen with purpose, don't remove any.
            (
                _f_2d,
                2 * _f_2d_nyquist_freq()[0] + 1,
                2 * _f_2d_nyquist_freq()[1] + 1,
                (0, 2 * np.pi),
                (0, 2 * np.pi),
            ),
            (
                _f_2d,
                2 * _f_2d_nyquist_freq()[0] + 1,
                2 * _f_2d_nyquist_freq()[1] + 1,
                (-np.pi / 3, 5 * np.pi / 3),
                (np.pi, 3 * np.pi),
            ),
            (
                lambda x, y: np.cos(30 * x) + np.sin(y) ** 2 + 1,
                2 * 30 // 30 + 1,
                2 * 2 + 1,
                (0, 2 * np.pi / 30),
                (np.pi, 3 * np.pi),
            ),
        ],
    )
    def test_interp_rfft2(self, func, m, n, domain0, domain1):
        """Test non-uniform FFT interpolation."""
        theta = np.array([7.34, 1.10134, 2.28, 1e3 * np.e])
        zeta = np.array([1.1, 3.78432, 8.542, 0])
        x = np.linspace(domain0[0], domain0[1], m, endpoint=False)
        y = np.linspace(domain1[0], domain1[1], n, endpoint=False)
        x, y = map(np.ravel, list(np.meshgrid(x, y, indexing="ij")))
        truth = func(theta, zeta)
        f = func(x, y).reshape(m, n)
        np.testing.assert_allclose(
            interp_rfft2(theta, zeta, f, domain0, domain1, axes=(-2, -1)),
            truth,
        )
        np.testing.assert_allclose(
            interp_rfft2(theta, zeta, f, domain0, domain1, axes=(-1, -2)),
            truth,
        )
        np.testing.assert_allclose(
            interp_rfft2(zeta, theta, f.T, domain1, domain0, axes=(-2, -1)),
            truth,
        )
        np.testing.assert_allclose(
            interp_rfft2(zeta, theta, f.T, domain1, domain0, axes=(-1, -2)),
            truth,
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "f, M, lobatto",
        [
            # Identity map known for bad Gibbs; if discrete Chebyshev transform
            # implemented correctly then won't see Gibbs.
            (_identity, 2, False),
            (_identity, 3, False),
            (_identity, 3, True),
            (_identity, 4, True),
        ],
    )
    def test_dct(self, f, M, lobatto):
        """Test discrete cosine transform interpolation.

        Parameters
        ----------
        f : callable
            Function to test.
        M : int
            Fourier spectral resolution.
        lobatto : bool
            Whether ``f`` should be sampled on the Gauss-Lobatto (extrema-plus-endpoint)
            or interior roots grid for Chebyshev points.

        """
        # Need to test fft used in Fourier Chebyshev interpolation due to issues like
        # https://github.com/scipy/scipy/issues/15033
        # https://github.com/scipy/scipy/issues/21198
        # https://github.com/google/jax/issues/22466.
        domain = (0, 2 * np.pi)
        m = cheb_pts(M, domain, lobatto)
        n = cheb_pts(m.size * 10, domain, lobatto)
        norm = (n.size - lobatto) / (m.size - lobatto)

        dct_type = 2 - lobatto
        fq_1 = np.sqrt(norm) * sidct(
            sdct(f(m), type=dct_type, norm="ortho", orthogonalize=False),
            type=dct_type,
            n=n.size,
            norm="ortho",
            orthogonalize=False,
        )
        if lobatto:
            # JAX has yet to implement type 1 DCT.
            fq_2 = norm * sidct(sdct(f(m), type=dct_type), n=n.size, type=dct_type)
        else:
            fq_2 = norm * idct(dct(f(m), type=dct_type), n=n.size, type=dct_type)
        np.testing.assert_allclose(fq_1, f(n), atol=1e-14)
        # Resolved by https://github.com/google/jax/issues/23895.
        np.testing.assert_allclose(fq_2, f(n), atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "f, M",
        [(_f_non_periodic, 5), (_f_non_periodic, 6), (_f_algebraic, 7)],
    )
    def test_interp_dct(self, f, M):
        """Test non-uniform DCT interpolation."""
        c0 = chebinterpolate(f, M - 1)
        assert not np.allclose(
            c0,
            cheb_from_dct(dct(f(chebpts1(M)), 2)) / M,
        ), (
            "Interpolation should fail because cosine basis is in wrong domain, "
            "yet the supplied test function was interpolated fine using this wrong "
            "domain. Pick a better test function."
        )
        # test interpolation
        z = cheb_pts(M)
        fz = f(z)
        np.testing.assert_allclose(c0, cheb_from_dct(dct(fz, 2) / M), atol=1e-13)
        if np.allclose(_f_algebraic(z), fz):  # Should reconstruct exactly.
            np.testing.assert_allclose(
                cheb2poly(c0),
                np.array([-np.e, -1, 0, 1, 1, 0, -10]),
                atol=1e-13,
            )
        # test evaluation
        xq = np.arange(10 * 3 * 2).reshape(10, 3, 2)
        xq = bijection_to_disc(xq, 0, xq.size)
        fq = chebval(xq, c0, tensor=False)
        np.testing.assert_allclose(fq, interp_dct(xq, fz), atol=1e-13)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "func, m, n",
        [
            (
                _f_2d,
                2 * _f_2d_nyquist_freq()[0] + 1,
                2 * _f_2d_nyquist_freq()[1] + 1,
            )
        ],
    )
    def test_fourier_chebyshev(self, func, m, n):
        """Tests for coverage of FourierChebyshev series."""
        x = fourier_pts(m)
        y = cheb_pts(n)
        x, y = map(np.ravel, list(np.meshgrid(x, y, indexing="ij")))
        f = func(x, y).reshape(m, n)
        fc = FourierChebyshevSeries(f)
        np.testing.assert_allclose(fc.evaluate(m, n), f, rtol=1e-6)
