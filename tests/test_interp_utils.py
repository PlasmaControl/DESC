"""Test interpolation utilities."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import (
    cheb2poly,
    chebinterpolate,
    chebpts1,
    chebpts2,
    chebval,
)
from scipy.fft import dct as sdct
from scipy.fft import idct as sidct

from desc.backend import dct, idct, rfft
from desc.integrals.interp_utils import (
    cheb_from_dct,
    cheb_pts,
    harmonic,
    harmonic_vander,
    interp_dct,
    interp_rfft,
    interp_rfft2,
    poly_root,
    polyder_vec,
    polyval_vec,
)
from desc.integrals.quad_utils import bijection_to_disc


@pytest.mark.unit
def test_poly_root():
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
def test_polyder_vec():
    """Test vectorized computation of polynomial derivative."""
    quintic = 6
    c = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    derivative = polyder_vec(c)
    desired = np.vectorize(np.polyder, signature="(m)->(n)")(c.T).T
    np.testing.assert_allclose(derivative, desired)


@pytest.mark.unit
def test_polyval_vec():
    """Test vectorized computation of polynomial evaluation."""

    def test(x, c):
        val = polyval_vec(x=x, c=c)
        c = np.moveaxis(c, 0, -1)
        x = x[..., np.newaxis]
        np.testing.assert_allclose(
            val,
            np.vectorize(np.polyval, signature="(m),(n)->(n)")(c, x).squeeze(axis=-1),
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
    x_freq_nyquist = 3 + 2
    y_freq_nyquist = 5 + 3
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
            cheb_pts(N, lobatto=True, domain=(-np.pi, np.pi)),
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
        "func, n",
        [
            (_f_1d, 2 * _f_1d_nyquist_freq() + 1),
            (_f_1d, 2 * _f_1d_nyquist_freq()),
        ],
    )
    def test_interp_rfft(self, func, n):
        """Test non-uniform FFT interpolation."""
        xq = np.array([7.34, 1.10134, 2.28])
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        assert not np.any(np.isclose(xq[..., np.newaxis], x))
        f, fq = func(x), func(xq)
        np.testing.assert_allclose(interp_rfft(xq, f), fq)
        M = f.shape[-1]
        np.testing.assert_allclose(
            np.sum(
                harmonic_vander(xq, M) * harmonic(rfft(f, norm="forward"), M), axis=-1
            ),
            fq,
        )

    @pytest.mark.xfail(
        reason="Numpy, jax, and scipy need to fix bug with 2D FFT (fft2)."
    )
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "func, m, n",
        [
            (_f_2d, 2 * _f_2d_nyquist_freq()[0] + 1, 2 * _f_2d_nyquist_freq()[1] + 1),
            (_f_2d, 2 * _f_2d_nyquist_freq()[0], 2 * _f_2d_nyquist_freq()[1]),
        ],
    )
    def test_interp_rfft2(self, func, m, n):
        """Test non-uniform FFT interpolation."""
        xq = np.array([[7.34, 1.10134, 2.28], [1.1, 3.78432, 8.542]]).T
        x = np.linspace(0, 2 * np.pi, m, endpoint=False)
        y = np.linspace(0, 2 * np.pi, n, endpoint=False)
        assert not np.any(np.isclose(xq[..., 0, np.newaxis], x))
        assert not np.any(np.isclose(xq[..., 1, np.newaxis], y))
        x, y = map(np.ravel, list(np.meshgrid(x, y, indexing="ij")))
        truth = func(xq[..., 0], xq[..., 1])
        np.testing.assert_allclose(
            interp_rfft2(xq, func(x, y).reshape(m, n), axes=(-2, -1)),
            truth,
        )
        np.testing.assert_allclose(
            interp_rfft2(xq, func(x, y).reshape(m, n), axes=(-1, -2)),
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
        # Want to unit test external code used in Fourier Chebyshev interpolation
        # due to issues like
        # https://github.com/scipy/scipy/issues/15033
        # https://github.com/scipy/scipy/issues/21198
        # https://github.com/google/jax/issues/22466,
        domain = (0, 2 * np.pi)
        m = cheb_pts(M, lobatto, domain)
        n = cheb_pts(m.size * 10, lobatto, domain)
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
        # JAX is much less accurate than scipy.
        np.testing.assert_allclose(fq_2, f(n), atol=1e-6)

        fig, ax = plt.subplots()
        ax.scatter(m, f(m))
        ax.plot(n, fq_1)
        ax.plot(n, fq_2)
        return fig

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "f, M",
        [(_f_non_periodic, 5), (_f_non_periodic, 6), (_f_algebraic, 7)],
    )
    def test_interp_dct(self, f, M):
        """Test non-uniform DCT interpolation."""
        c0 = chebinterpolate(f, M - 1)
        assert not np.allclose(c0, cheb_from_dct(dct(f(chebpts1(M)), 2) / M)), (
            "Interpolation should fail because cosine basis is in different domain. "
            "Use better test function."
        )
        # test interpolation
        z = cheb_pts(M)
        fz = f(z)
        np.testing.assert_allclose(c0, cheb_from_dct(dct(fz, 2) / M), atol=1e-13)
        if np.allclose(_f_algebraic(z), fz):
            np.testing.assert_allclose(
                cheb2poly(c0), np.array([-np.e, -1, 0, 1, 1, 0, -10]), atol=1e-13
            )
        # test evaluation
        xq = np.arange(10 * 3 * 2).reshape(10, 3, 2)
        xq = bijection_to_disc(xq, 0, xq.size)
        fq = chebval(xq, c0, tensor=False)
        np.testing.assert_allclose(fq, interp_dct(xq, fz), atol=1e-13)
