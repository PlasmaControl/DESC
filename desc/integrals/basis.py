"""Fast transformable series."""

from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from orthax.chebyshev import chebroots

from desc.backend import dct, flatnonzero, idct, irfft, jnp, rfft
from desc.integrals._interp_utils import (
    _eps,
    _filter_distinct,
    _subtract_first,
    cheb_from_dct,
    cheb_pts,
    dct_from_cheb,
    fourier_pts,
    idct_mmt,
    irfft_mmt,
    rfft_to_trig,
)
from desc.integrals.quad_utils import bijection_from_disc, bijection_to_disc
from desc.io import IOAble
from desc.utils import (
    atleast_2d_end,
    atleast_3d_mid,
    atleast_nd,
    errorif,
    flatten_mat,
    isposint,
    setdefault,
    take_mask,
    warnif,
)

_chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def _in_epigraph_and(is_intersect, df_dy, /):
    """Set and epigraph of function f with the given set of points.

    Used to return only intersects where the straight line path between
    adjacent intersects resides in the epigraph of a continuous map ``f``.

    Parameters
    ----------
    is_intersect : jnp.ndarray
        Boolean array indicating whether index corresponds to an intersect.
    df_dy : jnp.ndarray
        Shape ``is_intersect.shape``.
        Sign of ∂f/∂y (yᵢ) for f(yᵢ) = 0.

    Returns
    -------
    is_intersect : jnp.ndarray
        Boolean array indicating whether element is an intersect
        and satisfies the stated condition.

    Examples
    --------
    See ``desc/integrals/_bounce_utils.py::bounce_points``.
    This is used there to ensure the domains of integration are magnetic wells.

    """
    # The pairs y1 and y2 are boundaries of an integral only if y1 <= y2. For the
    # to be over wells, it is required that the first intersect has a non-positive
    # derivative. Now, by continuity, df_dy[...,k]<=0 implies df_dy[...,k+1]>=0, so
    # there can be at most one inversion, and if it exists, it must be at the first
    # pair. To correct the inversion, it suffices to disqualify the first intersect
    # as a right boundary, except under an edge case of a series of inflection points.
    idx = flatnonzero(is_intersect, size=2, fill_value=-1)
    edge_case = (
        (df_dy[idx[0]] == 0)
        & (df_dy[idx[1]] < 0)
        & is_intersect[idx[0]]
        & is_intersect[idx[1]]
        # In theory, we need to keep propagating this edge case, e.g.
        # (df_dy[..., 1] < 0) | (
        #     (df_dy[..., 1] == 0) & (df_dy[..., 2] < 0)...
        # ).
        # At each step, the likelihood that an intersection has already been lost
        # due to floating point errors grows, so the real solution is to pick a less
        # degenerate pitch value - one that does not ride the global extrema of f.
    )
    return is_intersect.at[idx[0]].set(edge_case)


# TODO: Move interpolation part to interpax. This is the only
#       existing differentiable FourierChebyshev FFT interpolation
#       in Python.
class FourierChebyshevSeries(IOAble):
    """Real-valued Fourier-Chebyshev series.

    f(x, y) = ∑ₘₙ aₘₙ ψₘ(x) Tₙ(y)
    where ψₘ are trigonometric polynomials on [0, 2π]
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ].

    Examples
    --------
    Let the magnetic field be B = ∇ρ × ∇x. This basis will then parameterize
    maps in Clebsch coordinates. Passing in a sequence of x values tracking
    the field line (see ``get_fieldline``) to the ``compute_cheb`` method will
    generate a 1D parameterization of f along the field line.

    This is useful to interpolate f ≝ θ and use the map x, ζ ↦ θ(x, ζ) to
    compute quantities along field lines via evaluating Fourier series
    parameterized in DESC computational coordinates θ, ζ, where the Fourier
    transform is more condensed, especially when NFP > 1.

    Notes
    -----
    Performance may improve if ``X`` and ``Y`` are powers of two.


    Parameters
    ----------
    f : jnp.ndarray
        Shape (..., X, Y).
        Samples of real function on the ``FourierChebyshevSeries.nodes`` grid.
    domain : tuple[float]
        Domain for y coordinates. Default is [-1, 1].
    lobatto : bool
        Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots grid for Chebyshev points.

    Attributes
    ----------
    X : int
        Fourier spectral resolution.
    Y : int
        Chebyshev spectral resolution.

    """

    def __init__(self, f, domain=(-1, 1), lobatto=False):
        """Interpolate Fourier-Chebyshev series to ``f``."""
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
        self.X = f.shape[-2]
        self.Y = f.shape[-1]
        self.domain = domain
        self.lobatto = lobatto
        self._c = rfft(
            dct(f, type=2 - lobatto, axis=-1) / (self.Y - lobatto),
            axis=-2,
            norm="forward",
        )

    @staticmethod
    def nodes(X, Y, L=None, domain=(-1, 1), lobatto=False):
        """Tensor product grid of optimal collocation nodes for this basis.

        Parameters
        ----------
        X : int
            Grid resolution in x direction. Preferably power of 2.
        Y : int
            Grid resolution in y direction. Preferably power of 2.
        L : int or jnp.ndarray
            Optional, resolution in radial direction of domain [0, 1].
            May also be an array of coordinates values. If given, then the
            returned ``coords`` is a 3D tensor-product with shape (L * X * Y, 3).
        domain : tuple[float]
            Domain for y coordinates. Default is [-1, 1].
        lobatto : bool
            Whether to use the Gauss-Lobatto (Extrema-plus-Endpoint)
            instead of the interior roots grid for Chebyshev points.

        Returns
        -------
        coords : jnp.ndarray
            Shape (X * Y, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        x = fourier_pts(X)
        y = cheb_pts(Y, domain, lobatto)
        if L is None:
            coords = (x, y)
        else:
            if isposint(L):
                L = jnp.flipud(jnp.linspace(1, 0, L, endpoint=False))
            coords = (jnp.atleast_1d(L), x, y)
        coords = tuple(map(jnp.ravel, jnp.meshgrid(*coords, indexing="ij")))
        return jnp.column_stack(coords)

    def evaluate(self, X, Y):
        """Evaluate Fourier-Chebyshev series on tensor-product grid.

        Parameters
        ----------
        X : int
            Grid resolution in x direction. Preferably power of 2.
        Y : int
            Grid resolution in y direction. Preferably power of 2.

        Returns
        -------
        fq : jnp.ndarray
            Shape (..., X, Y)
            Fourier-Chebyshev series evaluated at
            ``FourierChebyshevSeries.nodes(X,Y,L,self.domain,self.lobatto)``.

        """
        warnif(
            X < self.X,
            msg="Frequency spectrum of FFT interpolation will be truncated because "
            "the grid resolution is less than the Fourier resolution.\n"
            f"Got X = {X} < {self.X} = self.X.",
        )
        warnif(
            Y < self.Y,
            msg="Frequency spectrum of DCT interpolation will be truncated because "
            "the grid resolution is less than the Chebyshev resolution.\n"
            f"Got Y = {Y} < {self.Y} = self.Y.",
        )
        return idct(
            irfft(self._c, n=X, axis=-2, norm="forward"),
            type=2 - self.lobatto,
            n=Y,
            axis=-1,
        ) * (Y - self.lobatto)

    def harmonics(self):
        """Real spectral coefficients aₘₙ of the interpolating polynomial.

        The order of the returned coefficient array
        matches the Vandermonde matrix formed by an outer
        product of Fourier and Chebyshev matrices with order
        [sin(k𝐱), ..., sin(𝐱), 1, cos(𝐱), ..., cos(k𝐱)]
        ⊗ [T₀(𝐲), T₁(𝐲), ..., Tₙ(𝐲)]

        When ``self.X`` is even the sin(k𝐱) coefficient is zero and is excluded.

        Returns
        -------
        a_mn : jnp.ndarray
            Shape (..., X, Y).
            Real valued spectral coefficients for Fourier-Chebyshev series.

        """
        return rfft_to_trig(cheb_from_dct(self._c), self.X, axis=-2)

    def compute_cheb(self, x):
        """Evaluate Fourier series at ``x`` to obtain set of 1D Chebyshev coefficients.

        Parameters
        ----------
        x : jnp.ndarray
            Points to evaluate Fourier series.

        Returns
        -------
        cheb : jnp.ndarray
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        # Add axis to broadcast against Chebyshev coefficients.
        x = jnp.atleast_1d(x)[..., None]
        # Add axis to broadcast against multiple x values.
        cheb = cheb_from_dct(irfft_mmt(x, self._c[..., None, :, :], self.X, axis=-2))
        assert cheb.shape[-2:] == (x.shape[-2], self.Y)
        return cheb


def _add_lead_axis(cheb, arr):
    """Add leading axis for batching ``cheb`` depending on ``arr.ndim``.

    Input ``arr`` should not have rightmost dimension of cheb that iterates
    coefficients, but may have one leading axis for batching.
    """
    errorif(
        jnp.ndim(arr) > cheb.ndim,
        NotImplementedError,
        msg=f"Only one leading axis for batching is allowed. "
        f"Got {jnp.ndim(arr) - cheb.ndim + 1} leading axes.",
    )
    return cheb if jnp.ndim(arr) < cheb.ndim else cheb[None]


class PiecewiseChebyshevSeries(IOAble):
    """Chebyshev series.

    { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ]

    Parameters
    ----------
    cheb : jnp.ndarray
        Shape (..., X, Y).
        Chebyshev coefficients αₙ(x) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).
    domain : tuple[float]
        Domain for y coordinates. Default is [-1, 1].

    """

    def __init__(self, cheb, domain=(-1, 1)):
        """Make piecewise series from given Chebyshev coefficients."""
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        self.cheb = jnp.atleast_2d(cheb)
        self.domain = domain

    @property
    def X(self):
        """Number of cuts."""
        return self.cheb.shape[-2]

    @property
    def Y(self):
        """Chebyshev spectral resolution."""
        return self.cheb.shape[-1]

    def stitch(self):
        """Enforce continuity of the piecewise series."""
        # evaluate at left boundary
        f_0 = self.cheb[..., ::2].sum(-1) - self.cheb[..., 1::2].sum(-1)
        # evaluate at right boundary
        f_1 = self.cheb.sum(-1)
        dfx = f_1[..., :-1] - f_0[..., 1:]  # Δf = f(xᵢ, y₁) - f(xᵢ₊₁, y₀)
        self.cheb = self.cheb.at[..., 1:, 0].add(dfx.cumsum(-1))

    def evaluate(self, Y):
        """Evaluate Chebyshev series at Y Chebyshev points.

        Evaluate each function in this set
        { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
        at y points given by the Y Chebyshev points.

        Parameters
        ----------
        Y : int
            Grid resolution in y direction. Preferably power of 2.

        Returns
        -------
        fq : jnp.ndarray
            Shape (..., X, Y)
            Chebyshev series evaluated at Y Chebyshev points.

        """
        warnif(
            Y < self.Y,
            msg="Frequency spectrum of DCT interpolation will be truncated because "
            "the grid resolution is less than the Chebyshev resolution.\n"
            f"Got Y = {Y} < {self.Y} = self.Y.",
        )
        return idct(dct_from_cheb(self.cheb), type=2, n=Y, axis=-1) * Y

    def _isomorphism_to_C1(self, y):
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
        return y + z_shift[:, None]

    def _isomorphism_to_C2(self, z):
        """Return coordinates (x, y) ∈ ℂ² isomorphic to z ∈ ℂ.

        Returns index x and minimum value y such that
        z = f(x) + y where f(x) = x * |domain|.

        Parameters
        ----------
        z : jnp.ndarray
            Shape z.shape.

        Returns
        -------
        x_idx, y : tuple[jnp.ndarray]
            Shape z.shape.
            Isomorphic coordinates.

        """
        x_idx, y = jnp.divmod(z - self.domain[0], self.domain[-1] - self.domain[0])
        x_idx = x_idx.astype(int)
        y += self.domain[0]
        return x_idx, y

    def eval1d(self, z, cheb=None):
        """Evaluate piecewise Chebyshev series at coordinates z.

        Parameters
        ----------
        z : jnp.ndarray
            Shape (..., *cheb.shape[:-2], z.shape[-1]).
            Coordinates in [self.domain[0], ∞).
            The coordinates z ∈ ℝ are assumed isomorphic to (x, y) ∈ ℝ² where
            ``z//domain`` yields the index into the proper Chebyshev series
            along the second to last axis of ``cheb`` and ``z%domain`` is
            the coordinate value on the domain of that Chebyshev series.
        cheb : jnp.ndarray
            Shape (..., X, Y).
            Chebyshev coefficients to use. If not given, uses ``self.cheb``.

        Returns
        -------
        f : jnp.ndarray
            Chebyshev series evaluated at z.

        """
        cheb = _add_lead_axis(setdefault(cheb, self.cheb), z)
        x_idx, y = self._isomorphism_to_C2(z)
        y = bijection_to_disc(y, self.domain[0], self.domain[-1])
        # Chebyshev coefficients αₙ for f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z])
        # are in cheb array whose shape is (..., num cheb series, spectral resolution).
        cheb = jnp.take_along_axis(cheb, x_idx[..., None], axis=-2)
        return idct_mmt(y, cheb)

    #  The following changes would make root finding here more efficient.
    #  1. Boyd's method 𝒪(n²) instead of Chebyshev companion matrix 𝒪(n³).
    #  John P. Boyd, Computing real roots of a polynomial in Chebyshev series
    #  form through subdivision. https://doi.org/10.1016/j.apnum.2005.09.007.
    #  Use that once to find extrema of |B| if Y_B > 64.
    #  2. Then to find roots of bounce points use the closed formula in Boyd's
    #  spectral methods section 19.6. Can isolate interval to search for root by
    #  observing whether |B|-1/λ changes sign at extrema. Only need to do
    #  evaluate Chebyshev series at quadrature points once, and can use that to
    #  compute the integral for every λ. The integral will converge rapidly
    #  since a low order polynomial approximates |B| well in between adjacent
    #  extrema. 2 is a larger improvement than 1.

    def intersect2d(self, k=0.0, *, eps=_eps):
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
            Shape (..., *cheb.shape[:-1], Y - 1).
            Solutions yᵢ of f(x, yᵢ) = k(x), in ascending order.
        mask : jnp.ndarray
            Shape y.shape.
            Boolean array into ``y`` indicating whether element is an intersect.
        df_dy : jnp.ndarray
            Shape y.shape.
            Sign of ∂f/∂y (x, yᵢ).

        """
        c = _subtract_first(_add_lead_axis(self.cheb, k), k)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = _chebroots_vec(c)
        assert y.shape == (*c.shape[:-1], self.Y - 1)

        # Intersects must satisfy y ∈ [-1, 1].
        # Pick sentinel such that only distinct roots are considered intersects.
        y = _filter_distinct(y, sentinel=-2.0, eps=eps)
        mask = (jnp.abs(y.imag) <= eps) & (jnp.abs(y.real) < 1.0)
        # Ensure y ∈ (-1, 1), i.e. where arccos is differentiable.
        y = jnp.where(mask, y.real, 0.0)

        n = jnp.arange(self.Y)
        #      ∂f/∂y =      ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n Uₙ₋₁(y)
        # sign ∂f/∂y = sign ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n sin(n arcos y)
        # Fast multipoint method would be better. Reduces to FFT. Cost is
        # 𝒪([F+Q] log²[F + Q]) where F is spectral resolution and Q is number of points.
        # See Chapter 10, https://doi.org/10.1017/CBO9781139856065.
        df_dy = jnp.sign(
            jnp.linalg.vecdot(
                n * jnp.sin(n * jnp.arccos(y)[..., None]),
                self.cheb[..., None, :],
            )
        )
        y = bijection_from_disc(y, self.domain[0], self.domain[-1])
        return y, mask, df_dy

    def intersect1d(self, k=0.0, num_intersect=None):
        """Coordinates z(x, yᵢ) such that fₓ(yᵢ) = k for every x.

        Examples
        --------
        In ``Bounce2D.points``, the labels x, y, z, f, k are
          * z = ζ = (ϑ−α)/ι̅-ω ∈ ℝ
          * y = ζ mod (2π) ∈ [0, 2π]
          * x = α mod (2π) ∈ [0, 2π]
          * f = ‖B‖
          * k = 1/λ

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
            then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape broadcasts with (..., *self.cheb.shape[:-2], num intersect).
            Tuple of length two (z1, z2) of coordinates of intersects.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of f.

        """
        errorif(
            self.Y < 2,
            NotImplementedError,
            "This method requires a Chebyshev spectral resolution of Y > 1, "
            f"but got Y = {self.Y}.",
        )

        # Add axis to use same k over all Chebyshev series of the piecewise spline.
        y, mask, df_dy = self.intersect2d(jnp.atleast_1d(k)[..., None])
        # Flatten so that last axis enumerates intersects along the piecewise spline.
        y = flatten_mat(self._isomorphism_to_C1(y))
        mask = flatten_mat(mask)
        df_dy = flatten_mat(df_dy)

        # Note for bounce point applications:
        # We ignore the degenerate edge case where the boundary shared by adjacent
        # polynomials is a left intersection because the subset of pitch values
        # that generate this edge case has zero measure. By ignoring this, for those
        # subset of pitch values the integrations will be done in the hypograph of
        # |B|, which will yield zero. If in future decide to not ignore this, note
        # the solution is to
        # 1. disqualify intersects within ``_eps`` from ``domain[-1]``
        # 2. Evaluate sign in ``intersect2d`` at boundary of Chebyshev polynomial
        #    using Chebyshev identities rather than arccos(-1) or arccos(1) which
        #    are not differentiable.
        z1 = (df_dy <= 0) & mask
        z2 = (df_dy >= 0) & _in_epigraph_and(mask, df_dy)

        sentinel = self.domain[0] - 1.0
        z1 = take_mask(y, z1, size=num_intersect, fill_value=sentinel)
        z2 = take_mask(y, z2, size=num_intersect, fill_value=sentinel)

        mask = (z1 > sentinel) & (z2 > sentinel)
        # Set to zero so integration is over set of measure zero
        # and basis functions are faster to evaluate in downstream routines.
        z1 = jnp.where(mask, z1, 0.0)
        z2 = jnp.where(mask, z2, 0.0)
        return z1, z2

    def _check_shape(self, z1, z2, k):
        """Return shapes that broadcast with (k.shape[0], *self.cheb.shape[:-2], W)."""
        assert z1.shape == z2.shape
        # Ensure pitch batch dim exists and add back dim to broadcast with wells.
        k = atleast_nd(self.cheb.ndim - 1, k)[..., None]
        # Same but back dim already exists.
        z1 = atleast_nd(self.cheb.ndim, z1)
        z2 = atleast_nd(self.cheb.ndim, z2)
        # Cheb has shape    (..., X, Y) and others
        #     have shape (K, ..., W)
        assert z1.ndim == z2.ndim == k.ndim == self.cheb.ndim
        return z1, z2, k

    def check_intersect1d(self, z1, z2, k, plot=True, **kwargs):
        """Check that intersects are computed correctly.

        Parameters
        ----------
        z1, z2 : jnp.ndarray
            Shape must broadcast with (*self.cheb.shape[:-2], W).
            Coordinates of intersects.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of f.
        k : jnp.ndarray
            Shape must broadcast with self.cheb.shape[:-2].
            k such that fₓ(yᵢ) = k.
        plot : bool
            Whether to plot the piecewise spline and intersects for the given ``k``.
            For the plotting labels of ρ(l), α(m), it is assumed that the axis that
            enumerates the index l preceds the axis that enumerates the index m.
        kwargs : dict
            Keyword arguments into ``self.plot``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        kwargs.setdefault("title", r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$")
        title = kwargs.pop("title")
        plots = []

        z1, z2, k = self._check_shape(z1, z2, k)
        mask = (z1 - z2) != 0.0
        z1 = jnp.where(mask, z1, jnp.nan)
        z2 = jnp.where(mask, z2, jnp.nan)

        err_1 = jnp.any(z1 > z2, axis=-1)
        err_2 = jnp.any(z1[..., 1:] < z2[..., :-1], axis=-1)
        f_midpoint = self.eval1d((z1 + z2) / 2)
        eps = kwargs.pop("eps", jnp.finfo(jnp.array(1.0).dtype).eps * 10)
        err_3 = jnp.any(f_midpoint > k + eps, axis=-1)
        if not (plot or jnp.any(err_1 | err_2 | err_3)):
            return plots

        cheb = atleast_nd(3, self.cheb)
        mask, z1, z2, f_midpoint = map(atleast_3d_mid, (mask, z1, z2, f_midpoint))
        err_1, err_2, err_3 = map(atleast_2d_end, (err_1, err_2, err_3))

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
                        title=title
                        + rf" on field line $\rho(l)$, $\alpha(m)$, $(l,m)=${l}",
                        **kwargs,
                    )
                print("      z1    |    z2")
                print(jnp.column_stack([_z1, _z2]))
                assert not err_1[idx], "Intersects have an inversion.\n"
                assert not err_2[idx], "Detected discontinuity.\n"
                assert not err_3[idx], (
                    f"Detected f = {f_midpoint[idx][mask[idx]]} > {k[idx] + _eps} = k"
                    "in well, implying the straight line path between z1 and z2 is in"
                    "hypograph(f). Increase spectral resolution.\n"
                )
            idx = (slice(None), *l)
            if plot:
                plots.append(
                    self.plot1d(
                        cheb=cheb[l],
                        z1=z1[idx],
                        z2=z2[idx],
                        k=k[idx],
                        title=title
                        + rf" on field line $\rho(l)$, $\alpha(m)$, $(l,m)=${l}",
                        **kwargs,
                    )
                )
        return plots

    def plot1d(
        self,
        cheb,
        num=5000,
        z1=None,
        z2=None,
        k=None,
        k_transparency=0.5,
        klabel=r"$k$",
        title=r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$",
        hlabel=r"$z$",
        vlabel=r"$f$",
        show=True,
        include_legend=True,
    ):
        """Plot the piecewise Chebyshev series ``cheb``.

        Parameters
        ----------
        cheb : jnp.ndarray
            Shape (X, Y).
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
        include_legend : bool
            Whether to include the legend in the plot. Default is true.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        fig, ax = plt.subplots()
        legend = {}
        z = jnp.linspace(
            start=self.domain[0],
            stop=self.domain[0] + (self.domain[-1] - self.domain[0]) * self.X,
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
        if include_legend:
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
            mask = (_z1 - _z2) != 0.0
            _z1 = _z1[mask]
            _z2 = _z2[mask]
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
