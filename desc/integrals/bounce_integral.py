"""Methods for computing bounce integrals (singular or otherwise)."""

import numpy as np
from interpax import CubicHermiteSpline
from matplotlib import pyplot as plt
from orthax.legendre import leggauss

from desc.backend import dct, idct, irfft, jnp, rfft
from desc.integrals.bounce_utils import (
    _add2legend,
    _check_bounce_points,
    _interp_to_argmin_B_soft,
    _plot_intersect,
    bounce_points,
    bounce_quadrature,
    chebroots_vec,
    epigraph_and,
    flatten_matrix,
    get_alpha,
    plot_ppoly,
    subtract,
)
from desc.integrals.interp_utils import (
    _filter_distinct,
    cheb_from_dct,
    cheb_pts,
    fourier_pts,
    harmonic,
    idct_non_uniform,
    interp_rfft2,
    irfft2_non_uniform,
    irfft_non_uniform,
    polyder_vec,
    transform_to_desc,
)
from desc.integrals.quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    bijection_to_disc,
    get_quadrature,
    grad_automorphism_sin,
)
from desc.utils import (
    atleast_2d_end,
    atleast_3d_mid,
    atleast_nd,
    errorif,
    isposint,
    setdefault,
    take_mask,
    warnif,
)


class FourierChebyshevBasis:
    """Fourier-Chebyshev series.

    f(x, y) = ∑ₘₙ aₘₙ ψₘ(x) Tₙ(y)
    where ψₘ are trigonometric polynomials on [0, 2π]
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ].

    Notes
    -----
    Performance may improve significantly
    if the spectral resolutions ``M`` and ``N`` are powers of two.

    Attributes
    ----------
    M : int
        Fourier spectral resolution.
    N : int
        Chebyshev spectral resolution.
    lobatto : bool
        Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots grid for Chebyshev points.
    domain : (float, float)
        Domain for y coordinates.

    """

    def __init__(self, f, domain=(-1, 1), lobatto=False):
        """Interpolate Fourier-Chebyshev basis to ``f``.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (..., M, N).
            Samples of real function on the ``FourierChebyshevBasis.nodes`` grid.
        domain : (float, float)
            Domain for y coordinates. Default is [-1, 1].
        lobatto : bool
            Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
            instead of the interior roots grid for Chebyshev points.

        """
        self.M = f.shape[-2]
        self.N = f.shape[-1]
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        self.domain = tuple(domain)
        errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
        self.lobatto = bool(lobatto)
        self._c = FourierChebyshevBasis._fast_transform(f, self.lobatto)

    @staticmethod
    def _fast_transform(f, lobatto):
        M = f.shape[-2]
        N = f.shape[-1]
        return rfft(dct(f, type=2 - lobatto, axis=-1), axis=-2) / (M * (N - lobatto))

    @staticmethod
    def nodes(M, N, L=None, domain=(-1, 1), lobatto=False):
        """Tensor product grid of optimal collocation nodes for this basis.

        Parameters
        ----------
        M : int
            Grid resolution in x direction. Preferably power of 2.
        N : int
            Grid resolution in y direction. Preferably power of 2.
        L : int or jnp.ndarray
            Optional, resolution in radial direction of domain [0, 1].
            May also be an array of coordinates values. If given, then the
            returned ``coords`` is a 3D tensor-product with shape (L * M * N, 3).
        domain : (float, float)
            Domain for y coordinates. Default is [-1, 1].
        lobatto : bool
            Whether to use the Gauss-Lobatto (Extrema-plus-Endpoint)
            instead of the interior roots grid for Chebyshev points.

        Returns
        -------
        coords : jnp.ndarray
            Shape (M * N, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        x = fourier_pts(M)
        y = cheb_pts(N, lobatto, domain)
        if L is not None:
            if isposint(L):
                L = jnp.flipud(jnp.linspace(1, 0, L, endpoint=False))
            coords = (L, x, y)
        else:
            coords = (x, y)
        coords = list(map(jnp.ravel, jnp.meshgrid(*coords, indexing="ij")))
        coords = jnp.column_stack(coords)
        return coords

    def evaluate(self, M, N):
        """Evaluate Fourier-Chebyshev series.

        Parameters
        ----------
        M : int
            Grid resolution in x direction. Preferably power of 2.
        N : int
            Grid resolution in y direction. Preferably power of 2.

        Returns
        -------
        fq : jnp.ndarray
            Shape (..., M, N)
            Fourier-Chebyshev series evaluated at ``FourierChebyshevBasis.nodes(M, N)``.

        """
        fq = idct(irfft(self._c, n=M, axis=-2), type=2 - self.lobatto, n=N, axis=-1) * (
            M * (N - self.lobatto)
        )
        return fq

    def harmonics(self):
        """Spectral coefficients aₘₙ of the interpolating polynomial.

        Transform Fourier interpolant harmonics to Nyquist trigonometric
        interpolant harmonics so that the coefficients are all real.

        Returns
        -------
        a_mn : jnp.ndarray
            Shape (..., M, N).
            Real valued spectral coefficients for Fourier-Chebyshev basis.

        """
        a_mn = harmonic(cheb_from_dct(self._c, axis=-1), self.M, axis=-2)
        assert a_mn.shape[-2:] == (self.M, self.N)
        return a_mn

    def compute_cheb(self, x):
        """Evaluate Fourier basis at ``x`` to obtain set of 1D Chebyshev coefficients.

        Parameters
        ----------
        x : jnp.ndarray
            Points to evaluate Fourier basis.

        Returns
        -------
        cheb : ChebyshevBasisSet
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        # Always add new axis to broadcast against Chebyshev coefficients.
        x = jnp.atleast_1d(x)[..., jnp.newaxis]
        cheb = cheb_from_dct(irfft_non_uniform(x, self._c, self.M, axis=-2), axis=-1)
        assert cheb.shape[-2:] == (x.shape[-2], self.N)
        return ChebyshevBasisSet(cheb, self.domain)


class ChebyshevBasisSet:
    """Chebyshev series.

    { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ]

    Attributes
    ----------
    cheb : jnp.ndarray
        Shape (..., M, N).
        Chebyshev coefficients αₙ(x) for fₓ(y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).
    M : int
        Number of function in this basis set.
    N : int
        Chebyshev spectral resolution.
    domain : (float, float)
        Domain for y coordinates.

    """

    _eps = min(jnp.finfo(jnp.array(1.0).dtype).eps * 1e2, 1e-10)

    def __init__(self, cheb, domain=(-1, 1)):
        """Make Chebyshev series basis from given coefficients.

        Parameters
        ----------
        cheb : jnp.ndarray
            Shape (..., M, N).
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).
        domain : (float, float)
            Domain for y coordinates. Default is [-1, 1].

        """
        self.cheb = jnp.atleast_2d(cheb)
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        self.domain = tuple(domain)

    @property
    def M(self):
        """Number of function in this basis set."""
        return self.cheb.shape[-2]

    @property
    def N(self):
        """Chebyshev spectral resolution."""
        return self.cheb.shape[-1]

    @staticmethod
    def _chebcast(cheb, arr):
        # Input should not have rightmost dimension of cheb that iterates coefficients,
        # but may have additional leftmost dimension for batch operation.
        errorif(
            jnp.ndim(arr) > cheb.ndim,
            NotImplementedError,
            msg=f"Only one additional axis for batch dimension is allowed. "
            f"Got {jnp.ndim(arr) - cheb.ndim + 1} additional axes.",
        )
        return cheb if jnp.ndim(arr) < cheb.ndim else cheb[jnp.newaxis]

    def intersect2d(self, k=0.0, eps=_eps):
        """Coordinates yᵢ such that f(x, yᵢ) = k(x).

        Parameters
        ----------
        k : jnp.ndarray
            Shape must broadcast with (..., *cheb.shape[:-1]).
            Specify to find solutions yᵢ to f(x, yᵢ) = k(x). Default 0.
        eps : float
            Absolute tolerance with which to consider value as zero.

        Returns
        -------
        y : jnp.ndarray
            Shape (..., *cheb.shape[:-1], N - 1).
            Solutions yᵢ of f(x, yᵢ) = k(x), in ascending order.
        is_intersect : jnp.ndarray
            Shape y.shape.
            Boolean array into ``y`` indicating whether element is an intersect.
        df_dy_sign : jnp.ndarray
            Shape y.shape.
            Sign of ∂f/∂y (x, yᵢ).

        """
        c = subtract(ChebyshevBasisSet._chebcast(self.cheb, k), k)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = chebroots_vec(c)
        assert y.shape == (*c.shape[:-1], self.N - 1)

        # Intersects must satisfy y ∈ [-1, 1].
        # Pick sentinel such that only distinct roots are considered intersects.
        y = _filter_distinct(y, sentinel=-2.0, eps=eps)
        is_intersect = (jnp.abs(y.imag) <= eps) & (jnp.abs(y.real) <= 1.0)
        y = jnp.where(is_intersect, y.real, 1.0)  # ensure y is in domain of arcos

        # TODO: Multipoint evaluation with FFT.
        #   Chapter 10, https://doi.org/10.1017/CBO9781139856065.
        n = jnp.arange(self.N)
        #      ∂f/∂y =      ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n Uₙ₋₁(y)
        # sign ∂f/∂y = sign ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n sin(n arcos y)
        df_dy_sign = jnp.sign(
            jnp.linalg.vecdot(
                n * jnp.sin(n * jnp.arccos(y)[..., jnp.newaxis]),
                self.cheb[..., jnp.newaxis, :],
            )
        )
        y = bijection_from_disc(y, self.domain[0], self.domain[-1])
        return y, is_intersect, df_dy_sign

    def intersect1d(self, k=0.0, num_intersect=None, pad_value=0.0):
        """Coordinates z(x, yᵢ) such that fₓ(yᵢ) = k for every x.

        Parameters
        ----------
        k : jnp.ndarray
            Shape must broadcast with (..., *cheb.shape[:-2]).
            Specify to find solutions yᵢ to fₓ(yᵢ) = k. Default 0.
        num_intersect : int or None
            If not specified, then all intersects are returned in an array whose
            last axis has size ``self.M*(self.N-1)``. If there were less than that many
            intersects detected, then the last axis of the returned arrays is padded
            with ``pad_value``. Specify to return the first ``num_intersect`` pairs
            of intersects. This is useful if ``num_intersect`` tightly bounds the
            actual number.
        pad_value : float
            Value with which to pad array. Default 0.

        Returns
        -------
        z1, z2 : (jnp.ndarray, jnp.ndarray)
            Shape broadcasts with (..., *self.cheb.shape[:-2], num_intersect).
            ``z1``, ``z2`` holds intersects satisfying ∂f/∂y <= 0, ∂f/∂y >= 0,
            respectively.

        """
        errorif(
            self.N < 2,
            NotImplementedError,
            "This method requires the Chebyshev spectral resolution of at "
            f"least 2, but got N={self.N}.",
        )

        # Add axis to use same k over all Chebyshev series of the piecewise object.
        y, is_intersect, df_dy_sign = self.intersect2d(
            jnp.atleast_1d(k)[..., jnp.newaxis]
        )
        # Flatten so that last axis enumerates intersects along the piecewise object.
        y, is_intersect, df_dy_sign = map(
            flatten_matrix, (self.isomorphism_to_C1(y), is_intersect, df_dy_sign)
        )

        # Note for bounce point applications:
        # We ignore the degenerate edge case where the boundary shared by adjacent
        # polynomials is a left intersect point i.e. ``is_z1`` because the subset of
        # pitch values that generate this edge case has zero measure. Note that
        # the technique to account for this would be to disqualify intersects
        # within ``_eps`` from ``domain[-1]``.
        is_z1 = (df_dy_sign <= 0) & is_intersect
        is_z2 = (df_dy_sign >= 0) & epigraph_and(is_intersect, df_dy_sign)

        sentinel = self.domain[0] - 1.0
        z1 = take_mask(y, is_z1, size=num_intersect, fill_value=sentinel)
        z2 = take_mask(y, is_z2, size=num_intersect, fill_value=sentinel)

        mask = (z1 > sentinel) & (z2 > sentinel)
        # Set outside mask to same value so integration is over set of measure zero.
        z1 = jnp.where(mask, z1, pad_value)
        z2 = jnp.where(mask, z2, pad_value)
        return z1, z2

    def eval1d(self, z, cheb=None):
        """Evaluate piecewise Chebyshev spline at coordinates z.

        Parameters
        ----------
        z : jnp.ndarray
            Shape (..., *cheb.shape[:-2], z.shape[-1]).
            Coordinates in [sef.domain[0], ∞).
            The coordinates z ∈ ℝ are assumed isomorphic to (x, y) ∈ ℝ² where
            ``z // domain`` yields the index into the proper Chebyshev series
            along the second to last axis of ``cheb`` and ``z % domain`` is
            the coordinate value on the domain of that Chebyshev series.
        cheb : jnp.ndarray
            Shape (..., M, N).
            Chebyshev coefficients to use. If not given, uses ``self.cheb``.

        Returns
        -------
        f : jnp.ndarray
            Shape z.shape.
            Chebyshev basis evaluated at z.

        """
        cheb = self._chebcast(setdefault(cheb, self.cheb), z)
        N = cheb.shape[-1]
        x_idx, y = self.isomorphism_to_C2(z)
        y = bijection_to_disc(y, self.domain[0], self.domain[1])
        # Chebyshev coefficients αₙ for f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z])
        # are held in cheb with shape (..., num cheb series, N).
        cheb = jnp.take_along_axis(cheb, x_idx[..., jnp.newaxis], axis=-2)
        f = idct_non_uniform(y, cheb, N)
        assert f.shape == z.shape
        return f

    def isomorphism_to_C1(self, y):
        """Return coordinates z ∈ ℂ isomorphic to (x, y) ∈ ℂ².

        Maps row x of y to z = y + f(x) where f(x) = x * |domain|.

        Parameters
        ----------
        y : jnp.ndarray
            Shape (..., y.shape[-2], y.shape[-1]).
            Second to last axis iterates the rows.

        Returns
        -------
        z : jnp.ndarray
            Shape y.shape.
            Isomorphic coordinates.

        """
        assert y.ndim >= 2
        z_shift = jnp.arange(y.shape[-2]) * (self.domain[-1] - self.domain[0])
        z = y + z_shift[:, jnp.newaxis]
        return z

    def isomorphism_to_C2(self, z):
        """Return coordinates (x, y) ∈ ℂ² isomorphic to z ∈ ℂ.

        Returns index x and value y such that z = f(x) + y where f(x) = x * |domain|.

        Parameters
        ----------
        z : jnp.ndarray
            Shape z.shape.

        Returns
        -------
        x_idx, y_val : (jnp.ndarray, jnp.ndarray)
            Shape z.shape.
            Isomorphic coordinates.

        """
        x_idx, y_val = jnp.divmod(z - self.domain[0], self.domain[-1] - self.domain[0])
        x_idx = x_idx.astype(int)
        y_val += self.domain[0]
        return x_idx, y_val

    def _check_shape(self, z1, z2, k):
        """Return shapes that broadcast with (k.shape[0], *self.cheb.shape[:-2], W)."""
        # Ensure pitch batch dim exists and add back dim to broadcast with wells.
        k = atleast_nd(self.cheb.ndim - 1, k)[..., jnp.newaxis]
        # Same but back dim already exists.
        z1, z2 = atleast_nd(self.cheb.ndim, z1, z2)
        # Cheb has shape    (..., M, N) and others
        #     have shape (K, ..., W)
        errorif(not (z1.ndim == z2.ndim == k.ndim == self.cheb.ndim))
        return z1, z2, k

    def check_intersect1d(self, z1, z2, k, plot=True, **kwargs):
        """Check that intersects are computed correctly.

        Parameters
        ----------
        z1, z2 : jnp.ndarray
            Shape must broadcast with (k, *self.cheb.shape[:-2], W).
            ``z1``, ``z2`` holds intersects satisfying ∂f/∂y <= 0, ∂f/∂y >= 0,
            respectively.
        k : jnp.ndarray
            Shape must broadcast with (k.shape[0], *self.cheb.shape[:-2]).
            k such that fₓ(yᵢ) = k.
        plot : bool
            Whether to plot stuff. Default is true.
        kwargs : dict
            Keyword arguments into ``self.plot``.

        """
        assert z1.shape == z2.shape
        mask = (z1 - z2) != 0.0
        z1 = jnp.where(mask, z1, jnp.nan)
        z2 = jnp.where(mask, z2, jnp.nan)
        z1, z2, k = self._check_shape(z1, z2, k)

        err_1 = jnp.any(z1 > z2, axis=-1)
        err_2 = jnp.any(z1[..., 1:] < z2[..., :-1], axis=-1)
        f_m = self.eval1d((z1 + z2) / 2)
        assert f_m.shape == z1.shape
        err_3 = jnp.any(f_m > k + self._eps, axis=-1)
        if not (plot or jnp.any(err_1 | err_2 | err_3)):
            return

        # Ensure l axis exists for iteration in below loop.
        cheb = atleast_nd(3, self.cheb)
        mask, z1, z2, f_m = atleast_3d_mid(mask, z1, z2, f_m)
        err_1, err_2, err_3 = atleast_2d_end(err_1, err_2, err_3)

        for l in np.ndindex(cheb.shape[:-2]):
            for p in range(k.shape[0]):
                idx = (p, *l)
                if not (err_1[idx] or err_2[idx] or err_3[idx]):
                    continue
                _z1 = z1[idx][mask[idx]]
                _z2 = z2[idx][mask[idx]]
                if plot:
                    self.plot1d(
                        cheb=cheb[l],
                        z1=_z1,
                        z2=_z2,
                        k=k[idx],
                        **kwargs,
                    )
                print("      z1    |    z2")
                print(jnp.column_stack([_z1, _z2]))
                assert not err_1[idx], "Intersects have an inversion.\n"
                assert not err_2[idx], "Detected discontinuity.\n"
                assert not err_3[idx], (
                    "Detected f > k in well. Increase Chebyshev resolution.\n"
                    f"{f_m[idx][mask[idx]]} > {k[idx] + self._eps}"
                )
            idx = (slice(None), *l)
            if plot:
                self.plot1d(
                    cheb=cheb[l],
                    z1=z1[idx],
                    z2=z2[idx],
                    k=k[idx],
                    **kwargs,
                )

    def plot1d(
        self,
        cheb,
        num=1000,
        z1=None,
        z2=None,
        k=None,
        k_transparency=0.5,
        klabel=r"$k$",
        title=r"Intersects $z$ in epigraph of $f(z) = k$",
        hlabel=r"$z$",
        vlabel=r"$f(z)$",
        show=True,
    ):
        """Plot the piecewise Chebyshev series.

        Parameters
        ----------
        cheb : jnp.ndarray
            Shape (M, N).
            Piecewise Chebyshev series f.
        num : int
            Number of points to evaluate ``cheb`` for plot.
        z1 : jnp.ndarray
            Shape (k.shape[0], W).
            Optional, intersects with ∂f/∂y <= 0.
        z2 : jnp.ndarray
            Shape (k.shape[0], W).
            Optional, intersects with ∂f/∂y >= 0.
        k : jnp.ndarray
            Shape (k.shape[0], ).
            Optional, k such that fₓ(yᵢ) = k.
        k_transparency : float
            Transparency of pitch lines.
        klabel : float
            Label of intersect lines.
        title : str
            Plot title.
        hlabel : str
            Horizontal axis label.
        vlabel : str
            Vertical axis label.
        show : bool
            Whether to show the plot. Default is true.

        Returns
        -------
        fig, ax : matplotlib figure and axes

        """
        fig, ax = plt.subplots()
        legend = {}
        z = jnp.linspace(
            start=self.domain[0],
            stop=self.domain[0] + (self.domain[1] - self.domain[0]) * self.M,
            num=num,
        )
        _add2legend(legend, ax.plot(z, self.eval1d(z, cheb), label=vlabel))
        _plot_intersect(
            ax=ax,
            legend=legend,
            z1=z1,
            z2=z2,
            k=k,
            k_transparency=k_transparency,
            klabel=klabel,
        )
        ax.set_xlabel(hlabel)
        ax.set_ylabel(vlabel)
        ax.legend(legend.values(), legend.keys())
        ax.set_title(title)
        plt.tight_layout()
        if show:
            plt.show()
            plt.close()
        return fig, ax


def _transform_to_clebsch(grid, desc_from_clebsch, M, N, B):
    """Transform to Clebsch spectral domain.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes in
        (2π × 2π) poloidal and toroidal coordinates.
        Note that below shape notation defines
        L = ``grid.num_rho``, m = ``grid.num_theta``, and n = ``grid.num_zeta``.
    desc_from_clebsch : jnp.ndarray
        Shape (L * M * N, 3).
        DESC coordinates (ρ, θ, ζ) sourced from the Clebsch coordinates
        ``FourierChebyshevBasis.nodes(M,N,domain=FourierBounce.domain)``.
    M : int
        Grid resolution in poloidal direction for Clebsch coordinate grid.
        Preferably power of 2. A good choice is ``m``. If the poloidal stream
        function condenses the Fourier spectrum of |B| significantly, then a
        larger number may be beneficial.
    N : int
        Grid resolution in toroidal direction for Clebsch coordinate grid.
        Preferably power of 2.
    B : jnp.ndarray
        |B| evaluated on ``grid``.

    Returns
    -------
    T, B : (FourierChebyshevBasis, FourierChebyshevBasis)

    """
    T = FourierChebyshevBasis(
        # θ is computed on the optimal nodes in Clebsch space,
        # which is a tensor product node set in Clebsch space.
        f=desc_from_clebsch[:, 1].reshape(grid.num_rho, M, N),
        domain=Bounce2D.domain,
    )
    B = FourierChebyshevBasis(
        f=interp_rfft2(
            # Interpolate to optimal nodes in Clebsch space,
            # which is not a tensor product node set in DESC space.
            xq=desc_from_clebsch[:, 1:].reshape(grid.num_rho, -1, 2),
            f=grid.meshgrid_reshape(B, order="rtz")[:, jnp.newaxis],
            axes=(-1, -2),
        ).reshape(grid.num_rho, M, N),
        domain=Bounce2D.domain,
    )
    return T, B


# TODO:
#  After GitHub issue #1034 is resolved, we can also pass in the previous
#  θ(α) coordinates as an initial guess for the next coordinate mapping.
#  Perhaps tell the optimizer to perturb the coefficients of the
#  |B|(α, ζ) directly? Maybe auto diff to see change on |B|(θ, ζ)
#  and hence stream functions. just guessing. not sure if feasible / useful.
# TODO: Allow multiple starting labels for near-rational surfaces.
#  can just concatenate along second to last axis of cheb.


class Bounce2D:
    """Computes bounce integrals using two-dimensional pseudo-spectral methods.

    The bounce integral is defined as ∫ f(ℓ) dℓ, where
        dℓ parameterizes the distance along the field line in meters,
        f(ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points ζ₁, ζ₂ s.t. λ|B|(ζᵢ) = 1,
        where λ is a constant proportional to the magnetic moment over energy
        and |B| is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Notes
    -----
    Brief motivation and description of algorithm for developers.

    For applications which reduce to computing a nonlinear function of distance
    along field lines between bounce points, it is required to identify these
    points with field-line-following coordinates. In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as this operation becomes a surface integral,
    which is invariant to the order of summation.

    The DESC coordinate system is related to field-line-following coordinate
    systems by a relation whose solution is best found with Newton iteration.
    There is a unique real solution to this equation, so Newton iteration is a
    globally convergent root-finding algorithm here. For the task of finding
    bounce points, even if the inverse map: θ(α, ζ) was known, Newton iteration
    is not a globally convergent algorithm to find the real roots of
    f : ζ ↦ |B|(ζ) − 1/λ where ζ is a field-line-following  coordinate.
    For this, function approximation of |B| is necessary.

    Therefore, to compute bounce points {(ζ₁, ζ₂)}, we approximate |B| by a
    series expansion of basis functions in (α, ζ) coordinates restricting the
    class of basis functions to low order (e.g. N = 2ᵏ where k is small)
    algebraic or trigonometric polynomial with integer frequencies. These are
    the two classes useful for function approximation and for which there exists
    globally convergent root-finding algorithms. We require low order because
    the computation expenses grow with the number of potential roots, and the
    theorem of algebra states that number is N (2N) for algebraic
    (trigonometric) polynomials of degree N.

    The frequency transform of a map under the chosen basis must be concentrated
    at low frequencies for the series to converge to the true function fast.
    For periodic (non-periodic) maps, the best basis is a Fourier (Chebyshev)
    series. Both converge exponentially, but the larger region of convergence in
    the complex plane of Fourier series make it preferable in practice to choose
    coordinate systems such that the function to approximate is periodic. The
    Chebyshev series is preferred to other orthogonal polynomial series since
    fast discrete polynomial transforms (DPT) are implemented via fast transform
    to Chebyshev then DCT. Although nothing prohibits a direct DPT, we want to
    rely on existing, optimized libraries. There are other reasons to prefer
    Chebyshev series not discussed here.

    Therefore, |B| is interpolated to a Fourier-Chebyshev series in (α, ζ).
    The roots of f are computed as the eigenvalues of the Chebyshev companion
    matrix. This will later be replaced with Boyd's method:
    Computing real roots of a polynomial in Chebyshev series form through
    subdivision. https://doi.org/10.1016/j.apnum.2005.09.007.

    Computing accurate series expansions in (α, ζ) coordinates demands
    particular interpolation points in that coordinate system. Newton iteration
    is used to compute θ at these interpolation points. Note that interpolation
    is necessary because there is no transformation that converts series
    coefficients in periodic coordinates, e.g. (ϑ, ϕ), to a low order
    polynomial basis in non-periodic coordinates. For example, one can obtain
    series coefficients in (α, ϕ) coordinates from those in (ϑ, ϕ) as follows
        g : ϑ, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mϑ + nϕ])

        g : α, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mα + (m ι + n)ϕ])
    However, the basis for the latter are trigonometric functions with
    irrational frequencies since the rotational transform is irrational.
    Globally convergent root-finding schemes for that basis (at fixed α) are
    not known. The denominator of a close rational could be absorbed into the
    coordinate ϕ, but this balloons the frequency, and hence the degree of the
    series. Although since Fourier series may converge faster than Chebyshev,
    an alternate strategy that should work is to interpolate |B| to a double
    Fourier series in (ϑ, ϕ), then apply bisection methods to find roots of f
    with mesh size inversely proportional to the max frequency along the field
    line: M ι + N. ``Bounce2D`` does not use this approach because the
    root-finding scheme is inferior.

    After obtaining the bounce points, the supplied quadrature is performed.
    By default, Gauss quadrature is performed after removing the singularity.
    Fast fourier transforms interpolate functions in the integrand to the
    quadrature nodes.

    Fast transforms are used where possible, though fast multipoint methods
    are not yet implemented. For non-uniform interpolation, Vandermode MMT with
    the linear algebra libraries of JAX are used. It should be worthwhile to use
    the inverse non-uniform fast transforms. Fast multipoint methods are
    preferable because they are exact, but this requires more development work.
    Future work may implement these techniques, along with empirical testing of
    a few change of variables for the Chebyshev interpolation that may allow
    earlier truncation of the series without loss of accuracy.

    See Also
    --------
    Bounce1D
        Uses one-dimensional local spline methods for the same task.
        An advantage of ``Bounce2D`` over ``Bounce1D`` is that the coordinates on
        which the root-finding must be done to map from DESC to Clebsch coords is
        fixed to ``M*N``, independent of the number of toroidal transits.

    Warnings
    --------
    It is assumed that ζ = ϕ.

    Attributes
    ----------
    _B : ChebyshevBasisSet
        Set of 1D Chebyshev spectral coefficients of |B| along field line.
        {|B|_α : ζ ↦ |B|(α, ζ) | α ∈ A } where A = (α₀, α₁, …, αₘ₋₁) is the
        sequence of poloidal coordinates that specify the field line.
    _T : ChebyshevBasisSet
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {θ_α : ζ ↦ θ(α, ζ) | α ∈ A } where A = (α₀, α₁, …, αₘ₋₁) is the
        sequence of poloidal coordinates that specify the field line.

    """

    domain = (0, 2 * jnp.pi)

    def __init__(
        self,
        grid,
        data,
        desc_from_clebsch,
        M,
        N,
        alpha_0=0.0,
        num_transit=50,
        quad=leggauss(32),
        automorphism=(automorphism_sin, grad_automorphism_sin),
        B_ref=1.0,
        L_ref=1.0,
        check=False,
        **kwargs,
    ):
        """Returns an object to compute bounce integrals.

        Notes
        -----
        Performance may improve significantly
        if the spectral resolutions ``M`` and ``N`` are powers of two.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes in
            (2π × 2π) poloidal and toroidal coordinates.
            Note that below shape notation defines
            L = ``grid.num_rho``, m = ``grid.num_theta``, and n = ``grid.num_zeta``.
        data : dict[str, jnp.ndarray]
            Data evaluated on ``grid``. Must include ``FourierBounce.required_names()``.
        desc_from_clebsch : jnp.ndarray
            Shape (L * M * N, 3).
            DESC coordinates (ρ, θ, ζ) sourced from the Clebsch coordinates
            ``FourierChebyshevBasis.nodes(M,N,domain=FourierBounce.domain)``.
        M : int
            Grid resolution in poloidal direction for Clebsch coordinate grid.
            Preferably power of 2. A good choice is ``m``. If the poloidal stream
            function condenses the Fourier spectrum of |B| significantly, then a
            larger number may be beneficial.
        N : int
            Grid resolution in toroidal direction for Clebsch coordinate grid.
            Preferably power of 2.
        alpha_0 : float
            Starting field line poloidal label.
        num_transit : int
            Number of toroidal transits to follow field line.
        quad : (jnp.ndarray, jnp.ndarray)
            Quadrature points xₖ and weights wₖ for the approximate evaluation of an
            integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 32 points.
        automorphism : (Callable, Callable) or None
            The first callable should be an automorphism of the real interval [-1, 1].
            The second callable should be the derivative of the first. This map defines
            a change of variable for the bounce integral. The choice made for the
            automorphism will affect the performance of the quadrature method.
        B_ref : float
            Optional. Reference magnetic field strength for normalization.
        L_ref : float
            Optional. Reference length scale for normalization.
        check : bool
            Flag for debugging. Must be false for JAX transformations.

        """
        errorif(grid.sym, NotImplementedError, msg="Need grid that works with FFTs.")
        # Strictly increasing zeta knots enforces dζ > 0.
        # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require
        # B^ζ = B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ or [∂ℓ/∂ζ]|ρ,a.
        # Recall dζ = ∇ζ⋅dR, implying 1 = ∇ζ⋅(e_ζ|ρ,a). Hence, a sign change in ∇ζ
        # requires the same sign change in e_ζ|ρ,a to retain the metric identity.
        warnif(
            check and kwargs.pop("warn", True) and jnp.any(data["B^zeta"] <= 0),
            msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Enforcing positive B^ζ.",
        )
        self._m = grid.num_theta
        self._n = grid.num_zeta
        self._b_sup_z = jnp.expand_dims(
            transform_to_desc(grid, jnp.abs(data["B^zeta"]) / data["|B|"] * L_ref),
            axis=1,
        )
        self._x, self._w = get_quadrature(quad, automorphism)

        # Compute global splines.
        T, B = _transform_to_clebsch(grid, desc_from_clebsch, M, N, data["|B|"] / B_ref)
        # peel off field lines
        alphas = get_alpha(
            alpha_0,
            grid.compress(data["iota"]),
            num_transit,
            period=Bounce2D.domain[-1],
        )
        self._B = B.compute_cheb(alphas)
        # Evaluating set of Chebyshev series more efficient than evaluating
        # Fourier Chebyshev series, so we project θ to Chebyshev series as well.
        self._T = T.compute_cheb(alphas)
        assert self._B.M == self._T.M == num_transit
        assert self._B.N == self._T.N == N
        assert (
            self._B.cheb.shape == self._T.cheb.shape == (grid.num_rho, num_transit, N)
        )

    @staticmethod
    def desc_from_clebsch(eq, L, M, N, clebsch=None, **kwargs):
        """Return DESC coordinates of optimal Fourier Chebyshev basis nodes.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to use defining the coordinate mapping.
        L : int or jnp.ndarray
            Number of flux surfaces uniformly in [0, 1] on which to compute.
            May also be an array of non-uniform coordinates.
        M : int
            Grid resolution in poloidal direction for Clebsch coordinate grid.
            Preferably power of 2. A good choice is ``m``. If the poloidal stream
            function condenses the Fourier spectrum of |B| significantly, then a
            larger number may be beneficial.
        N : int
            Grid resolution in toroidal direction for Clebsch coordinate grid.
            Preferably power of 2.
        clebsch : jnp.ndarray
            Optional, Clebsch coordinate tensor-product grid (ρ, α, ζ).
            If given, ``L``, ``M``, and ``N`` are ignored.
        kwargs : dict
            Additional parameters to supply to the coordinate mapping function.
            See ``desc.equilibrium.Equilibrium.map_coordinates``.

        Returns
        -------
        desc_coords : jnp.ndarray
            Shape (L * M * N, 3).
            DESC coordinate grid (ρ, θ, ζ) sourced from the Clebsch coordinate
            tensor-product grid (ρ, α, ζ).

        """
        if clebsch is None:
            clebsch = FourierChebyshevBasis.nodes(M, N, L, Bounce2D.domain)
        desc_coords = eq.map_coordinates(
            coords=clebsch,
            inbasis=("rho", "alpha", "zeta"),
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
            **kwargs,
        )
        return desc_coords

    @staticmethod
    def required_names():
        """Return names in ``data_index`` required to compute bounce integrals."""
        return ["B^zeta", "|B|", "iota"]

    @staticmethod
    def reshape_data(grid, *data):
        """Reshape``data`` given by ``names`` for input to ``self.integrate``.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ).
        data : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : list[jnp.ndarray]
            List of reshaped data which may be given to ``self.integrate``.

        """
        return [grid.meshgrid_reshape(d, "rtz")[:, jnp.newaxis] for d in data]

    @property
    def _L(self):
        """int: Number of flux surfaces to compute on."""
        return self._B.cheb.shape[0]

    def bounce_points(self, pitch, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch : jnp.ndarray
            Shape (P, L).
            λ values to evaluate the bounce integral at each field line. λ(ρ) is
            specified by ``pitch[...,ρ]`` where in the latter the labels ρ are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number of wells. As a reference, there are typically
            at most 5 wells per toroidal transit for a given pitch.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.

        Returns
        -------
        bp1, bp2 : (jnp.ndarray, jnp.ndarray)
            Shape (P, L, num_well).
            The field line-following coordinates of bounce points.
            The pairs ``bp1`` and ``bp2`` form left and right integration boundaries,
            respectively, for the bounce integrals.

        """
        return self._B.intersect1d(1 / jnp.atleast_2d(pitch), num_well)

    def check_bounce_points(self, bp1, bp2, pitch, plot=True, **kwargs):
        """Check that bounce points are computed correctly and plot them."""
        kwargs.setdefault(
            "title", r"Intersects $\zeta$ for $\vertB(\zeta)\vert = 1/\lambda$"
        )
        kwargs.setdefault("hlabel", r"$\zeta$")
        kwargs.setdefault("vlabel", r"$\vertB\vert(\zeta)$")
        self._B.check_intersect1d(bp1, bp2, 1 / pitch, plot, **kwargs)

    def integrate(self, pitch, integrand, f, weight=None, num_well=None):
        """Bounce integrate ∫ f(ℓ) dℓ.

        Computes the bounce integral ∫ f(ℓ) dℓ for every specified field line
        for every λ value in ``pitch``.

        Parameters
        ----------
        pitch : jnp.ndarray
            Shape (P, L).
            λ values to evaluate the bounce integral at each field line. λ(ρ) is
            specified by ``pitch[...,ρ]`` where in the latter the labels ρ are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        integrand : callable
            The composition operator on the set of functions in ``f`` that maps the
            functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ. It should accept the
            arrays in ``f`` as arguments as well as the additional keyword arguments:
            ``B`` and ``pitch``. A quadrature will be performed to approximate the
            bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
        f : list[jnp.ndarray]
            Shape (L, 1, m, n).
            Real scalar-valued (2π × 2π) periodic in (θ, ζ) functions evaluated
            on the ``grid`` supplied to construct this object. These functions
            should be arguments to the callable ``integrand``. Use the method
            ``self.reshape_data`` to reshape the data into the expected shape.
        weight : jnp.ndarray
            Shape (L, 1, m, n).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in the magnetic well. Use the method
            ``self.reshape_data`` to reshape the data into the expected shape.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number of wells. As a reference, there are typically
            at most 5 wells per toroidal transit for a given pitch.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.

        Returns
        -------
        result : jnp.ndarray
            Shape (P, L, num_well).
            First axis enumerates pitch values. Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        pitch = jnp.atleast_2d(pitch)
        bp1, bp2 = self.bounce_points(pitch, num_well)
        result = self._integrate(bp1, bp2, pitch, integrand, f)
        errorif(weight is not None, NotImplementedError)
        return result

    def _integrate(self, bp1, bp2, pitch, integrand, f):
        assert bp1.ndim == 3
        assert bp1.shape == bp2.shape
        assert pitch.ndim == 2
        W = bp1.shape[-1]  # number of wells
        shape = (pitch.shape[0], self._L, W, self._x.size)

        # quadrature points parameterized by ζ for each pitch and flux surface
        Q_zeta = flatten_matrix(
            bijection_from_disc(self._x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis])
        )
        # quadrature points in (θ, ζ) coordinates
        Q = jnp.stack([self._T.eval1d(Q_zeta), Q_zeta], axis=-1)

        # interpolate and integrate
        f = [interp_rfft2(Q, f_i, axes=(-1, -2)).reshape(shape) for f_i in f]
        result = jnp.dot(
            integrand(
                *f,
                B=self._B.eval1d(Q_zeta).reshape(shape),
                pitch=pitch[..., jnp.newaxis, jnp.newaxis],
            )
            / irfft2_non_uniform(
                Q, self._b_sup_z, self._m, self._n, axes=(-1, -2)
            ).reshape(shape),
            self._w,
        )
        assert result.shape == (pitch.shape[0], self._L, W)
        return result


class Bounce1D:
    """Computes bounce integrals using one-dimensional local spline methods.

    The bounce integral is defined as ∫ f(ℓ) dℓ, where
        dℓ parameterizes the distance along the field line in meters,
        f(ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points ζ₁, ζ₂ s.t. λ|B|(ζᵢ) = 1,
        where λ is a constant proportional to the magnetic moment over energy
        and |B| is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Notes
    -----
    Brief description of algorithm for developers.

    For applications which reduce to computing a nonlinear function of distance
    along field lines between bounce points, it is required to identify these
    points with field-line-following coordinates. In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as this operation becomes a surface integral,
    which is invariant to the order of summation.

    The DESC coordinate system is related to field-line-following coordinate
    systems by a relation whose solution is best found with Newton iteration.
    There is a unique real solution to this equation, so Newton iteration is a
    globally convergent root-finding algorithm here. For the task of finding
    bounce points, even if the inverse map: θ(α, ζ) was known, Newton iteration
    is not a globally convergent algorithm to find the real roots of
    f : ζ ↦ |B|(ζ) − 1/λ where ζ is a field-line-following coordinate.
    For this, function approximation of |B| is necessary.

    The function approximation in ``Bounce1D`` is ignorant that the objects to
    approximate are defined on a bounded subset of ℝ². Instead, the domain is
    projected to ℝ, where information sampled about the function at infinity
    cannot support reconstruction of the function near the origin. As the
    functions of interest do not vanish at infinity, pseudo-spectral techniques
    are not used. Instead, function approximation is done with local splines.
    This is useful if one can efficiently obtain data along field lines.

    After obtaining the bounce points, the supplied quadrature is performed.
    By default, Gauss quadrature is performed after removing the singularity.
    Local splines interpolate functions in the integrand to the quadrature nodes.

    See Also
    --------
    Bounce2D : Uses two-dimensional pseudo-spectral techniques for the same task.

    Warnings
    --------
    The supplied data must be from a Clebsch coordinate (ρ, α, ζ) tensor-product grid.
    The field-line-following  coordinate ζ must be strictly increasing.
    The ζ coordinate is preferably uniformly spaced, although this is not required.
    These are used as knots to construct splines.
    A reference density is 100 knots per toroidal transit.

    Examples
    --------
    See ``tests/test_integrals.py::TestBounce1D::test_integrate_checks``.

    Attributes
    ----------
    B : jnp.ndarray
        Shape (4, L * M, N - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.

    """

    plot_ppoly = staticmethod(plot_ppoly)

    def __init__(
        self,
        grid,
        data,
        quad=leggauss(32),
        automorphism=(automorphism_sin, grad_automorphism_sin),
        Bref=1.0,
        Lref=1.0,
        check=False,
        **kwargs,
    ):
        """Returns an object to compute bounce integrals.

        Parameters
        ----------
        grid : Grid
            Clebsch coordinate (ρ, α, ζ) tensor-product grid.
            Note that below shape notation defines
            L = ``grid.num_rho``, M = ``grid.num_alpha``, and N = ``grid.num_zeta``.
        data : dict[str, jnp.ndarray]
            Data evaluated on ``grid``.
            Must include names in ``Bounce1D.required_names()``.
        quad : (jnp.ndarray, jnp.ndarray)
            Quadrature points xₖ and weights wₖ for the approximate evaluation of an
            integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 32 points.
        automorphism : (Callable, Callable) or None
            The first callable should be an automorphism of the real interval [-1, 1].
            The second callable should be the derivative of the first. This map defines
            a change of variable for the bounce integral. The choice made for the
            automorphism will affect the performance of the quadrature method.
        Bref : float
            Optional. Reference magnetic field strength for normalization.
        Lref : float
            Optional. Reference length scale for normalization.
        check : bool
            Flag for debugging. Must be false for JAX transformations.

        """
        # Strictly increasing zeta knots enforces dζ > 0.
        # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require
        # B^ζ = B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ or [∂ℓ/∂ζ]|ρ,a.
        # Recall dζ = ∇ζ⋅dR, implying 1 = ∇ζ⋅(e_ζ|ρ,a). Hence, a sign change in ∇ζ
        # requires the same sign change in e_ζ|ρ,a to retain the metric identity.
        warnif(
            check and kwargs.pop("warn", True) and jnp.any(data["B^zeta"] <= 0),
            msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Enforcing positive B^ζ.",
        )
        data = {
            "B^zeta": jnp.abs(data["B^zeta"]) * Lref / Bref,
            "B^zeta_z|r,a": data["B^zeta_z|r,a"]
            * jnp.sign(data["B^zeta"])
            * Lref
            / Bref,
            "|B|": data["|B|"] / Bref,
            "|B|_z|r,a": data["|B|_z|r,a"] / Bref,  # This is already the correct sign.
        }
        self._data = {
            key: grid.meshgrid_reshape(val, "raz").reshape(-1, grid.num_zeta)
            for key, val in data.items()
        }
        self._x, self._w = get_quadrature(quad, automorphism)

        # Compute local splines.
        self._zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
        self.B = jnp.moveaxis(
            CubicHermiteSpline(
                x=self._zeta,
                y=self._data["|B|"],
                dydx=self._data["|B|_z|r,a"],
                axis=-1,
                check=check,
            ).c,
            source=1,
            destination=-1,
        )
        self._dB_dz = polyder_vec(self.B)
        degree = 3
        assert self.B.shape[0] == degree + 1
        assert self._dB_dz.shape[0] == degree
        assert self.B.shape[-1] == self._dB_dz.shape[-1] == grid.num_zeta - 1

    @staticmethod
    def required_names():
        """Return names in ``data_index`` required to compute bounce integrals."""
        return ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]

    @staticmethod
    def reshape_data(grid, *data):
        """Reshape ``data`` given by ``names`` for input to ``self.integrate``.

        Parameters
        ----------
        grid : Grid
            Clebsch coordinate (ρ, α, ζ) tensor-product grid.
        data : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : list[jnp.ndarray]
            List of reshaped data which may be given to ``self.integrate``.

        """
        return [
            grid.meshgrid_reshape(d, "raz").reshape(-1, grid.num_zeta) for d in data
        ]

    def bounce_points(self, pitch, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch : jnp.ndarray
            Shape must broadcast with (P, L * M).
            λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
            specified by ``pitch[...,ρ]`` where in the latter the labels (ρ,α) are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number of wells. As a reference, there are typically
            at most 5 wells per toroidal transit for a given pitch.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.

        Returns
        -------
        bp1, bp2 : (jnp.ndarray, jnp.ndarray)
            Shape (P, L * M, num_well).
            The field line-following coordinates of bounce points.
            The pairs ``bp1`` and ``bp2`` form left and right integration boundaries,
            respectively, for the bounce integrals.

            If there were less than ``num_wells`` wells detected along a field line,
            then the last axis, which enumerates bounce points for  a particular field
            line and pitch, is padded with zero.

        """
        return bounce_points(
            pitch=pitch,
            knots=self._zeta,
            B=self.B,
            dB_dz=self._dB_dz,
            num_well=num_well,
        )

    def check_bounce_points(self, bp1, bp2, pitch, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        bp1, bp2 : (jnp.ndarray, jnp.ndarray)
            Shape (P, L * M, num_well).
            The field line-following coordinates of bounce points.
            The pairs ``bp1`` and ``bp2`` form left and right integration boundaries,
            respectively, for the bounce integrals.
        pitch : jnp.ndarray
            Shape must broadcast with (P, L * M).
            λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
            specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        plot : bool
            Whether to plot stuff.
        kwargs : dict
            Keyword arguments into ``self.plot_ppoly``.

        """
        _check_bounce_points(
            bp1=bp1,
            bp2=bp2,
            pitch=jnp.atleast_2d(pitch),
            knots=self._zeta,
            B=self.B,
            plot=plot,
            **kwargs,
        )

    def integrate(
        self,
        pitch,
        integrand,
        f,
        weight=None,
        num_well=None,
        method="cubic",
        batch=True,
        check=False,
    ):
        """Bounce integrate ∫ f(ℓ) dℓ.

        Computes the bounce integral ∫ f(ℓ) dℓ for every specified field line
        for every λ value in ``pitch``.

        Parameters
        ----------
        pitch : jnp.ndarray
            Shape must broadcast with (P, L * M).
            λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
            specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        integrand : callable
            The composition operator on the set of functions in ``f`` that maps the
            functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ. It should accept the
            arrays in ``f`` as arguments as well as the additional keyword arguments:
            ``B`` and ``pitch``. A quadrature will be performed to approximate the
            bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
        f : list[jnp.ndarray]
            Shape (L * M, N).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. These functions should be arguments to the callable
            ``integrand``. Use the method ``self.reshape_data`` to reshape the data
            into the expected shape.
        weight : jnp.ndarray
            Shape (L * M, N).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in the magnetic well. Use the method
            ``self.reshape_data`` to reshape the data into the expected shape.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number of wells. As a reference, there are typically
            at most 5 wells per toroidal transit for a given pitch.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.
        method : str
            Method of interpolation for functions contained in ``f``.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
            Default is cubic C1 local spline.
        batch : bool
            Whether to perform computation in a batched manner. Default is true.
        check : bool
            Flag for debugging. Must be false for JAX transformations.

        Returns
        -------
        result : jnp.ndarray
            Shape (P, L*M, num_well).
            First axis enumerates pitch values. Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        pitch = jnp.atleast_2d(pitch)
        bp1, bp2 = self.bounce_points(pitch, num_well)
        result = bounce_quadrature(
            x=self._x,
            w=self._w,
            bp1=bp1,
            bp2=bp2,
            pitch=pitch,
            integrand=integrand,
            f=f,
            data=self._data,
            knots=self._zeta,
            method=method,
            batch=batch,
            check=check,
        )
        if weight is not None:
            result *= _interp_to_argmin_B_soft(
                g=weight,
                bp1=bp1,
                bp2=bp2,
                knots=self._zeta,
                B=self.B,
                dB_dz=self._dB_dz,
                method=method,
            )
        assert result.shape[-1] == setdefault(num_well, (self._zeta.size - 1) * 3)
        return result
