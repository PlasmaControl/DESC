"""Test interpolation utilities."""

from functools import partial

import numpy as np
import pytest
from jax import grad
from numpy.polynomial.chebyshev import (
    cheb2poly,
    chebinterpolate,
    chebpts1,
    chebpts2,
    chebval,
)
from numpy.polynomial.polynomial import polyvander

from desc.backend import dct, idct, jnp, rfft, rfft2
from desc.integrals._interp_utils import (
    cheb_from_dct,
    cheb_pts,
    fourier_pts,
    ifft_mmt,
    interp_dct,
    interp_rfft,
    interp_rfft2,
    irfft_mmt,
    nufft1d2r,
    nufft2d2r,
    polyder_vec,
    polyroot_vec,
    polyval_vec,
    rfft_to_trig,
    trig_vander,
)
from desc.integrals.basis import FourierChebyshevSeries
from desc.integrals.quad_utils import bijection_to_disc
from desc.utils import identity


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
    @pytest.mark.parametrize("M", [1, 8, 9])
    def test_fft_shift(self, M):
        """Test frequency shifting."""
        a = np.fft.rfftfreq(M, 1 / M)
        np.testing.assert_allclose(a, np.arange(M // 2 + 1))
        b = np.fft.fftfreq(a.size, 1 / a.size) + a.size // 2
        np.testing.assert_allclose(np.fft.ifftshift(a), b)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "func, n, domain, imag_undersampled",
        [
            (*_test_inputs_1D[0], False),
            (*_test_inputs_1D[1], True),
            (*_test_inputs_1D[2], False),
            (*_test_inputs_1D[3], True),
            (*_test_inputs_1D[4], True),
            (*_test_inputs_1D[5], False),
        ],
    )
    def test_non_uniform_FFT(self, func, n, domain, imag_undersampled):
        """Test non-uniform FFT interpolation."""
        x = np.linspace(domain[0], domain[1], n, endpoint=False)
        c = func(x)
        xq = np.array([7.34, 1.10134, 2.28])

        f = np.fft.fft(c, norm="forward")
        np.testing.assert_allclose(f[0].imag, 0, atol=1e-12)
        if n % 2 == 0:
            np.testing.assert_allclose(f[n // 2].imag, 0, atol=1e-12)

        r = ifft_mmt(xq, f, domain)
        np.testing.assert_allclose(r.real if imag_undersampled else r, func(xq))

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
        x, y = map(jnp.ravel, tuple(jnp.meshgrid(x, y, indexing="ij")))
        c = func(x, y).reshape(m, n)

        xq = jnp.array([7.34, 1.10134, 2.28, 1e3 * np.e])
        yq = jnp.array([1.1, 3.78432, 8.542, 0])

        f1 = 2 * rfft2(c, norm="forward")
        f1 = f1.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)
        f2 = jnp.fft.fft2(c, norm="forward")

        v = func(xq, yq)
        np.testing.assert_allclose(nufft2d2r(xq, yq, f1, domain_x, domain_y), v)
        np.testing.assert_allclose(nufft2d2r(xq, yq, f2, domain_x, domain_y, None), v)

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
        np.testing.assert_allclose(g1(xq, yq), g, atol=1e-11)
        np.testing.assert_allclose(g2(xq, yq), g, atol=1e-11)

    @pytest.mark.unit
    @pytest.mark.parametrize("func, n, domain", _test_inputs_1D)
    def test_non_uniform_real_MMT(self, func, n, domain):
        """Test non-uniform real MMT interpolation."""
        x = np.linspace(domain[0], domain[1], n, endpoint=False)
        c = func(x)
        xq = np.array([7.34, 1.10134, 2.28])

        np.testing.assert_allclose(interp_rfft(xq, c, domain), func(xq))
        vand = trig_vander(xq, c.shape[-1], domain)
        coef = rfft_to_trig(rfft(c, norm="forward"), c.shape[-1])
        np.testing.assert_allclose((vand * coef).sum(-1), func(xq))

    @pytest.mark.unit
    @pytest.mark.parametrize("func, m, n, domain_x, domain_y", _test_inputs_2D)
    def test_non_uniform_real_MMT_2D(self, func, m, n, domain_x, domain_y):
        """Test non-uniform real MMT 2D interpolation."""
        x = np.linspace(domain_x[0], domain_x[1], m, endpoint=False)
        y = np.linspace(domain_y[0], domain_y[1], n, endpoint=False)
        x, y = map(np.ravel, tuple(np.meshgrid(x, y, indexing="ij")))
        c = func(x, y).reshape(m, n)
        xq = np.array([7.34, 1.10134, 2.28, 1e3 * np.e])
        yq = np.array([1.1, 3.78432, 8.542, 0])

        v = func(xq, yq)
        np.testing.assert_allclose(
            interp_rfft2(xq, yq, c, domain_x, domain_y, (-2, -1)), v
        )
        np.testing.assert_allclose(
            interp_rfft2(xq, yq, c, domain_x, domain_y, (-1, -2)), v
        )
        np.testing.assert_allclose(
            interp_rfft2(yq, xq, c.T, domain_y, domain_x, (-2, -1)), v
        )
        np.testing.assert_allclose(
            interp_rfft2(yq, xq, c.T, domain_y, domain_x, (-1, -2)), v
        )
        np.testing.assert_allclose(
            irfft_mmt(
                yq,
                ifft_mmt(xq[:, None], rfft2(c, norm="forward"), domain_x, -2),
                n,
                domain_y,
            ),
            v,
        )

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

    @pytest.mark.unit
    @pytest.mark.parametrize("N", [2, 6, 7])
    def test_cheb_pts(self, N):
        """Test we use Chebyshev points compatible with DCT."""
        np.testing.assert_allclose(cheb_pts(N), chebpts1(N)[::-1], atol=1e-15)
        np.testing.assert_allclose(
            cheb_pts(N, domain=(-np.pi, np.pi), lobatto=True),
            np.pi * chebpts2(N)[::-1],
            atol=1e-15,
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "f, M, lobatto",
        [
            # Identity map known for bad Gibbs; if discrete Chebyshev transform
            # implemented correctly then won't see Gibbs.
            (identity, 2, False),
            (identity, 3, False),
            (identity, 3, True),
            (identity, 4, True),
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
        # Need to test interpolation due to issues like
        # https://github.com/scipy/scipy/issues/15033
        # https://github.com/scipy/scipy/issues/21198
        # https://github.com/google/jax/issues/22466
        # https://github.com/google/jax/issues/23895.
        from scipy.fft import dct as sdct
        from scipy.fft import idct as sidct

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
        np.testing.assert_allclose(fq_2, f(n), atol=1e-14)

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

        z = cheb_pts(M)
        fz = f(z)
        np.testing.assert_allclose(c0, cheb_from_dct(dct(fz, 2) / M), atol=1e-13)
        if np.allclose(_f_algebraic(z), fz):  # Should reconstruct exactly.
            np.testing.assert_allclose(
                cheb2poly(c0),
                np.array([-np.e, -1, 0, 1, 1, 0, -10]),
                atol=1e-13,
            )

        xq = np.arange(10 * 3 * 2).reshape(10, 3, 2)
        xq = bijection_to_disc(xq, 0, xq.size)
        fq = chebval(xq, c0, tensor=False)
        np.testing.assert_allclose(fq, interp_dct(xq, fz), atol=1e-13)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "func, m, n",
        [(_c_2d, 2 * _c_2d_nyquist_freq()[0] + 1, 2 * _c_2d_nyquist_freq()[1] + 1)],
    )
    def test_fourier_chebyshev(self, func, m, n):
        """Tests for coverage of FourierChebyshev series."""
        x = fourier_pts(m)
        y = cheb_pts(n)
        x, y = map(np.ravel, list(np.meshgrid(x, y, indexing="ij")))
        f = func(x, y).reshape(m, n)
        fc = FourierChebyshevSeries(f)
        np.testing.assert_allclose(fc.evaluate(m, n), f)


# TODO(#1388)
class TestPolyUtils:
    """Test polynomial utilities used for local spline interpolation."""

    @pytest.mark.unit
    def test_polyroot_vec(self):
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
