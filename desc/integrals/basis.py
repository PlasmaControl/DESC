"""Fast transformable basis."""

from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from desc.backend import dct, flatnonzero, idct, irfft, jnp, put, rfft
from desc.integrals.interp_utils import (
    _filter_distinct,
    cheb_from_dct,
    cheb_pts,
    chebroots_vec,
    fourier_pts,
    harmonic,
    idct_non_uniform,
    irfft_non_uniform,
)
from desc.integrals.quad_utils import bijection_from_disc, bijection_to_disc
from desc.utils import (
    atleast_2d_end,
    atleast_3d_mid,
    atleast_nd,
    errorif,
    flatten_matrix,
    isposint,
    setdefault,
    take_mask,
)


def _subtract(c, k):
    """Subtract ``k`` from first index of last axis of ``c``.

    Semantically same as ``return c.copy().at[...,0].add(-k)``,
    but allows dimension to increase.
    """
    c_0 = c[..., 0] - k
    c = jnp.concatenate(
        [
            c_0[..., jnp.newaxis],
            jnp.broadcast_to(c[..., 1:], (*c_0.shape, c.shape[-1] - 1)),
        ],
        axis=-1,
    )
    return c


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def _in_epigraph_and(is_intersect, df_dy_sign):
    """Set and epigraph of function f with the given set of points.

    Used to return only intersects where the straight line path between
    adjacent intersects resides in the epigraph of a continuous map ``f``.

    Warnings
    --------
    Does not support keyword arguments.

    Parameters
    ----------
    is_intersect : jnp.ndarray
        Boolean array indicating whether index corresponds to an intersect.
    df_dy_sign : jnp.ndarray
        Shape ``is_intersect.shape``.
        Sign of ∂f/∂y (yᵢ) for f(yᵢ) = 0.

    Returns
    -------
    is_intersect : jnp.ndarray
        Boolean array indicating whether element is an intersect
        and satisfies the stated condition.

    """
    # The pairs ``y1`` and ``y2`` are boundaries of an integral only if ``y1 <= y2``.
    # For the integrals to be over wells, it is required that the first intersect
    # has a non-positive derivative. Now, by continuity,
    # ``df_dy_sign[...,k]<=0`` implies ``df_dy_sign[...,k+1]>=0``,
    # so there can be at most one inversion, and if it exists, the inversion
    # must be at the first pair. To correct the inversion, it suffices to disqualify the
    # first intersect as a right boundary, except under an edge case of a series of
    # inflection points.
    idx = flatnonzero(is_intersect, size=2, fill_value=-1)  # idx of first 2 intersects
    edge_case = (
        (df_dy_sign[idx[0]] == 0)
        & (df_dy_sign[idx[1]] < 0)
        & is_intersect[idx[0]]
        & is_intersect[idx[1]]
        # In theory, we need to keep propagating this edge case, e.g.
        # (df_dy_sign[..., 1] < 0) | (
        #     (df_dy_sign[..., 1] == 0) & (df_dy_sign[..., 2] < 0)...
        # ).
        # At each step, the likelihood that an intersection has already been lost
        # due to floating point errors grows, so the real solution is to pick a less
        # degenerate pitch value - one that does not ride the global extrema of f.
    )
    return put(is_intersect, idx[0], edge_case)


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
        errorif(lobatto, NotImplementedError, "JAX hasn't implemented type 1 DCT.")
        self.lobatto = bool(lobatto)
        self._c = FourierChebyshevBasis._fast_transform(f, self.lobatto)

    @staticmethod
    def _fast_transform(f, lobatto):
        N = f.shape[-1]
        c = rfft(
            dct(f, type=2 - lobatto, axis=-1) / (N - lobatto),
            axis=-2,
            norm="forward",
        )
        return c

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
            coords = (jnp.atleast_1d(L), x, y)
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
            Fourier-Chebyshev series evaluated at
            ``FourierChebyshevBasis.nodes(M,N,L,self.domain,self.lobatto)``.

        """
        fq = idct(
            irfft(self._c, n=M, axis=-2, norm="forward"),
            type=2 - self.lobatto,
            n=N,
            axis=-1,
        ) * (N - self.lobatto)
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
        Number of functions in this basis set.
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
        """Number of functions in this basis set."""
        return self.cheb.shape[-2]

    @property
    def N(self):
        """Chebyshev spectral resolution."""
        return self.cheb.shape[-1]

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

        Returns index x and minimum value y such that
        z = f(x) + y where f(x) = x * |domain|.

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

    def eval1d(self, z, cheb=None):
        """Evaluate piecewise Chebyshev series at coordinates z.

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
        cheb = _chebcast(setdefault(cheb, self.cheb), z)
        N = cheb.shape[-1]
        x_idx, y = self.isomorphism_to_C2(z)
        y = bijection_to_disc(y, self.domain[0], self.domain[1])
        # Chebyshev coefficients αₙ for f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z])
        # are held in cheb with shape (..., num cheb series, N).
        cheb = jnp.take_along_axis(cheb, x_idx[..., jnp.newaxis], axis=-2)
        f = idct_non_uniform(y, cheb, N)
        assert f.shape == z.shape
        return f

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
        c = _subtract(_chebcast(self.cheb, k), k)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = chebroots_vec(c)
        assert y.shape == (*c.shape[:-1], self.N - 1)

        # Intersects must satisfy y ∈ [-1, 1].
        # Pick sentinel such that only distinct roots are considered intersects.
        y = _filter_distinct(y, sentinel=-2.0, eps=eps)
        is_intersect = (jnp.abs(y.imag) <= eps) & (jnp.abs(y.real) <= 1.0)
        # Ensure y is in domain of arcos; choose 1 because kernel probably cheaper.
        y = jnp.where(is_intersect, y.real, 1.0)

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
            Specify to return the first ``num_intersect`` intersects.
            This is useful if ``num_intersect`` tightly bounds the actual number.

            If not specified, then all intersects are returned. If there were fewer
            intersects detected than the size of the last axis of the returned arrays,
            then that axis is padded with ``pad_value``.
        pad_value : float
            Value with which to pad array. Default 0.

        Returns
        -------
        z1, z2 : (jnp.ndarray, jnp.ndarray)
            Shape broadcasts with (..., *self.cheb.shape[:-2], num_intersect).
            ``z1`` and ``z2`` are intersects satisfying ∂f/∂y <= 0 and ∂f/∂y >= 0,
            respectively. The points are grouped and ordered such that the straight
            line path between ``z1`` and ``z2`` resides in the epigraph of f.

        """
        errorif(
            self.N < 2,
            NotImplementedError,
            "This method requires a Chebyshev spectral resolution of N > 1, "
            f"but got N = {self.N}.",
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
        # polynomials is a left intersection i.e. ``is_z1`` because the subset of
        # pitch values that generate this edge case has zero measure. By ignoring
        # this, for those subset of pitch values the integrations will be done in
        # the hypograph of |B|, which will yield zero. If in far future decide to
        # not ignore this, note the solution is to disqualify intersects within
        # ``_eps`` from ``domain[-1]``.
        is_z1 = (df_dy_sign <= 0) & is_intersect
        is_z2 = (df_dy_sign >= 0) & _in_epigraph_and(is_intersect, df_dy_sign)

        sentinel = self.domain[0] - 1.0
        z1 = take_mask(y, is_z1, size=num_intersect, fill_value=sentinel)
        z2 = take_mask(y, is_z2, size=num_intersect, fill_value=sentinel)

        mask = (z1 > sentinel) & (z2 > sentinel)
        # Set outside mask to same value so integration is over set of measure zero.
        z1 = jnp.where(mask, z1, pad_value)
        z2 = jnp.where(mask, z2, pad_value)
        return z1, z2

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
            Shape must broadcast with (*self.cheb.shape[:-2], W).
            ``z1`` and ``z2`` are intersects satisfying ∂f/∂y <= 0 and ∂f/∂y >= 0,
            respectively. The points are grouped and ordered such that the straight
            line path between ``z1`` and ``z2`` resides in the epigraph of f.
        k : jnp.ndarray
            Shape must broadcast with *self.cheb.shape[:-2].
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
        f_midpoint = self.eval1d((z1 + z2) / 2)
        assert f_midpoint.shape == z1.shape
        err_3 = jnp.any(f_midpoint > k + self._eps, axis=-1)
        if not (plot or jnp.any(err_1 | err_2 | err_3)):
            return

        # Ensure l axis exists for iteration in below loop.
        cheb = atleast_nd(3, self.cheb)
        mask, z1, z2, f_midpoint = atleast_3d_mid(mask, z1, z2, f_midpoint)
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
                    "Detected f > k in well, implying the straight line path between "
                    "z1 and z2 is in hypograph(f). Increase spectral resolution.\n"
                    f"{f_midpoint[idx][mask[idx]]} > {k[idx] + self._eps}"
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
        title=r"Intersects $z$ in epigraph($f$) s.t. $f(z) = k$",
        hlabel=r"$z$",
        vlabel=r"$f$",
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
        ax.legend(legend.values(), legend.keys(), loc="lower right")
        ax.set_title(title)
        plt.tight_layout()
        if show:
            plt.show()
            plt.close()
        return fig, ax


def _add2legend(legend, lines):
    """Add lines to legend if it's not already in it."""
    for line in setdefault(lines, [lines], hasattr(lines, "__iter__")):
        label = line.get_label()
        if label not in legend:
            legend[label] = line


def _plot_intersect(ax, legend, z1, z2, k, k_transparency, klabel):
    """Plot intersects on ``ax``."""
    if k is None:
        return

    k = jnp.atleast_1d(jnp.squeeze(k))
    assert k.ndim == 1
    z1, z2 = jnp.atleast_2d(z1, z2)
    assert z1.ndim == z2.ndim >= 2
    assert k.shape[0] == z1.shape[0] == z2.shape[0]
    for p in k:
        _add2legend(
            legend,
            ax.axhline(p, color="tab:purple", alpha=k_transparency, label=klabel),
        )
    for i in range(k.size):
        _z1, _z2 = z1[i], z2[i]
        if _z1.size == _z2.size:
            mask = (z1 - z2) != 0.0
            _z1 = z1[mask]
            _z2 = z2[mask]
        _add2legend(
            legend,
            ax.scatter(
                _z1,
                jnp.full_like(_z1, k[i]),
                marker="v",
                color="tab:red",
                label=r"$z_1$",
            ),
        )
        _add2legend(
            legend,
            ax.scatter(
                _z2,
                jnp.full_like(_z2, k[i]),
                marker="^",
                color="tab:green",
                label=r"$z_2$",
            ),
        )
