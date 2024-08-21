"""Methods for computing Fast Fourier Chebyshev transforms and bounce integrals."""

import numpy as np
from matplotlib import pyplot as plt
from orthax.chebyshev import chebroots
from orthax.legendre import leggauss

from desc.backend import dct, idct, irfft, jnp, rfft, rfft2
from desc.integrals._interp_utils import (
    _filter_distinct,
    cheb_from_dct,
    cheb_pts,
    fourier_pts,
    harmonic,
    idct_non_uniform,
    interp_rfft2,
    irfft2_non_uniform,
    irfft_non_uniform,
)
from desc.integrals._quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    bijection_to_disc,
    grad_automorphism_sin,
)
from desc.integrals.bounce_integral import _fix_inversion, filter_bounce_points
from desc.utils import (
    atleast_2d_end,
    atleast_3d_mid,
    atleast_nd,
    errorif,
    setdefault,
    take_mask,
    warnif,
)

_chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


def _flatten_matrix(y):
    # Flatten batch of matrix to batch of vector.
    return y.reshape(*y.shape[:-2], -1)


def get_alphas(alpha_0, iota, num_transit, period):
    """Get sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) of field line.

    Parameters
    ----------
    alpha_0 : float
        Starting field line poloidal label.
    iota : jnp.ndarray
        Shape (iota.size, ).
        Rotational transform normalized by 2π.
    num_transit : float
        Number of ``period``s to follow field line.
    period : float
        Toroidal period after which to update label.

    Returns
    -------
    alphas : jnp.ndarray
        Shape (iota.size, num_transit).
        Sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) that specify field line.

    """
    # Δϕ (∂α/∂ϕ) = Δϕ ι̅ = Δϕ ι/2π = Δϕ data["iota"]
    alphas = alpha_0 + period * iota[:, jnp.newaxis] * jnp.arange(num_transit)
    return alphas


class FourierChebyshevBasis:
    """Fourier-Chebyshev series.

    f(x, y) = ∑ₘₙ aₘₙ ψₘ(x) Tₙ(y)
    where ψₘ are trigonometric polynomials on [0, 2π]
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ].

    Attributes
    ----------
    M : int
        Fourier spectral resolution.
    N : int
        Chebyshev spectral resolution.
    lobatto : bool
        Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        or interior roots grid for Chebyshev points.
    domain : (float, float)
        Domain for y coordinates.

    """

    def __init__(self, f, domain, lobatto=False):
        """Interpolate Fourier-Chebyshev basis to ``f``.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (..., M, N).
            Samples of real function on the ``FourierChebyshevBasis.nodes`` grid.
            M, N preferably power of 2.
        domain : (float, float)
            Domain for y coordinates.
        lobatto : bool
            Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
            or interior roots grid for Chebyshev points.

        """
        self.M = f.shape[-2]
        self.N = f.shape[-1]
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        self.domain = domain
        errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
        self.lobatto = bool(lobatto)
        self._c = self._fast_transform(f, self.lobatto)

    @staticmethod
    def nodes(M, N, domain, lobatto=False, **kwargs):
        """Tensor product grid of optimal collocation nodes for this basis.

        Parameters
        ----------
        M : int
            Grid resolution in x direction. Preferably power of 2.
        N : int
            Grid resolution in y direction. Preferably power of 2.
        domain : (float, float)
            Domain for y coordinates.
        lobatto : bool
            Whether to use the Gauss-Lobatto (Extrema-plus-Endpoint)
            or interior roots grid for Chebyshev points.

        Returns
        -------
        coord : jnp.ndarray
            Shape (M * N, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        x = fourier_pts(M)
        y = cheb_pts(N, lobatto, domain)
        coord = [jnp.atleast_1d(kwargs.pop("rho")), x, y] if "rho" in kwargs else [x, y]
        coord = list(map(jnp.ravel, jnp.meshgrid(*coord, indexing="ij")))
        coord = jnp.column_stack(coord)
        return coord

    @staticmethod
    def _fast_transform(f, lobatto):
        M = f.shape[-2]
        N = f.shape[-1]
        return rfft(dct(f, type=2 - lobatto, axis=-1), axis=-2) / (M * (N - lobatto))

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
        cheb : PiecewiseChebyshevBasis
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        # Always add new axis to broadcast against Chebyshev coefficients.
        x = jnp.atleast_1d(x)[..., jnp.newaxis]
        cheb = cheb_from_dct(irfft_non_uniform(x, self._c, self.M, axis=-2), axis=-1)
        assert cheb.shape[-2:] == (x.shape[-2], self.N)
        return PiecewiseChebyshevBasis(cheb, self.domain)


def _subtract(c, k):
    # subtract k from last axis of c, obeying numpy broadcasting
    c_0 = c[..., 0] - k
    c = jnp.concatenate(
        [
            c_0[..., jnp.newaxis],
            jnp.broadcast_to(c[..., 1:], (*c_0.shape, c.shape[-1] - 1)),
        ],
        axis=-1,
    )
    return c


class PiecewiseChebyshevBasis:
    """Chebyshev series.

    { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ].

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

    def __init__(self, cheb, domain):
        """Make Chebyshev series basis from given coefficients.

        Parameters
        ----------
        cheb : jnp.ndarray
            Shape (..., M, N).
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        self.domain = domain
        self.cheb = jnp.atleast_2d(cheb)

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

    def intersect(self, k, eps=_eps):
        """Coordinates yᵢ such that f(x, yᵢ) = k(x).

        Parameters
        ----------
        k : jnp.ndarray
            Shape cheb.shape[:-1] or (k.shape[0], *cheb.shape[:-1]).
            Specify to find solutions yᵢ to f(x, yᵢ) = k(x). Default 0.
        eps : float
            Absolute tolerance with which to consider value as zero.

        Returns
        -------
        y : jnp.ndarray
            Shape (..., *cheb.shape[:-1], N - 1).
            Solutions yᵢ of f(x, yᵢ) = k(x), in ascending order.
        is_decreasing : jnp.ndarray
            Shape y.shape.
            Whether ∂f/∂y (x, yᵢ) is decreasing.
        is_increasing : jnp.ndarray
            Shape y.shape.
            Whether ∂f/∂y (x, yᵢ) is increasing.
        is_intersect : jnp.ndarray
            Shape y.shape.
            Boolean array into ``y`` indicating whether element is an intersect.

        """
        c = _subtract(self._chebcast(self.cheb, k), k)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = _chebroots_vec(c)
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
        s = jnp.linalg.vecdot(
            n * jnp.sin(n * jnp.arccos(y)[..., jnp.newaxis]),
            self.cheb[..., jnp.newaxis, :],
        )
        is_decreasing = s <= 0
        is_increasing = s >= 0

        y = bijection_from_disc(y, *self.domain)
        return y, is_decreasing, is_increasing, is_intersect

    def bounce_points(self, pitch, num_well=None):
        """Compute bounce points given intersections.

        Parameters
        ----------
        pitch : jnp.ndarray
            Shape must broadcast with (P, *self.cheb.shape[:-2]).
            λ values to evaluate the bounce integral.
        num_well : int or None
            If not specified, then all bounce points are returned in an array whose
            last axis has size ``self.M*(self.N-1)``. If there were less than that many
            wells detected along a field line, then the last axis of the returned
            arrays, which enumerates bounce points for a particular field line and
            pitch, is padded with zero.

            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number of wells. As a reference, there are
            typically <= 5 wells per toroidal transit.

        Returns
        -------
        bp1, bp2 : jnp.ndarray
            Shape broadcasts with (P, *self.cheb.shape[:-2], num_well).
            The field line-following coordinates of bounce points.
            The pairs ``bp1`` and ``bp2`` form left and right integration boundaries,
            respectively, for the bounce integrals.

        """
        # _fix_inversion assumes N > 1.
        errorif(self.N < 2, NotImplementedError, f"Got self.N = {self.N} < 2.")
        y, is_decreasing, is_increasing, is_intersect = self.intersect(
            # Add axis to use same pitch over all cuts of field line.
            1
            / jnp.atleast_1d(pitch)[..., jnp.newaxis]
        )
        # Flatten so that last axis enumerates intersects of a pitch along a field line.
        y = _flatten_matrix(self._isomorphism_to_C1(y))
        is_decreasing = _flatten_matrix(is_decreasing)
        is_increasing = _flatten_matrix(is_increasing)
        is_intersect = _flatten_matrix(is_intersect)
        # We ignore the degenerate edge case where the boundary shared by adjacent
        # polynomials is a left bounce point i.e. ``is_bp1`` because the subset of
        # pitch values that generate this edge case has zero measure. Note that
        # the technique to account for this would be to disqualify intersects
        # within ``_eps`` from ``domain[-1]``.
        is_bp1 = is_decreasing & is_intersect
        is_bp2 = is_increasing & _fix_inversion(is_intersect, is_increasing)

        sentinel = self.domain[0] - 1.0
        bp1 = take_mask(y, is_bp1, size=num_well, fill_value=sentinel)
        bp2 = take_mask(y, is_bp2, size=num_well, fill_value=sentinel)

        mask = (bp1 > sentinel) & (bp2 > sentinel)
        # Set outside mask to same value so integration is over set of measure zero.
        bp1 = jnp.where(mask, bp1, 0.0)
        bp2 = jnp.where(mask, bp2, 0.0)
        return bp1, bp2

    def eval1d(self, z, cheb=None):
        """Evaluate piecewise Chebyshev spline at coordinates z.

        The coordinates z ∈ ℝ are assumed isomorphic to (x, y) ∈ ℝ²
        where z integer division domain yields index into the proper
        Chebyshev series of the spline and z mod domain is the coordinate
        value along the domain of that Chebyshev series.

        Parameters
        ----------
        z : jnp.ndarray
            Shape (..., *cheb.shape[:-2], z.shape[-1]).
            Isomorphic coordinates along field line [0, ∞).
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
        x_idx, y = self._isomorphism_to_C2(z)
        y = bijection_to_disc(y, self.domain[0], self.domain[1])
        # Chebyshev coefficients αₙ for f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z])
        # are held in cheb with shape (..., num cheb series, N).
        cheb = jnp.take_along_axis(cheb, x_idx[..., jnp.newaxis], axis=-2)
        f = idct_non_uniform(y, cheb, N)
        assert f.shape == z.shape
        return f

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
        return y + z_shift[:, jnp.newaxis]

    def _isomorphism_to_C2(self, z):
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
        return x_idx.astype(int), y_val + self.domain[0]

    def _check_shape(self, bp1, bp2, pitch):
        """Return shapes that broadcast with (P, *self.cheb.shape[:-2], W)."""
        # Ensure pitch batch dim exists and add back dim to broadcast with wells.
        pitch = atleast_nd(self.cheb.ndim - 1, pitch)[..., jnp.newaxis]
        # Same but back dim already exists.
        bp1, bp2 = atleast_nd(self.cheb.ndim, bp1, bp2)
        # Cheb has shape    (..., M, N) and others
        #     have shape (P, ..., W)
        errorif(not (bp1.ndim == bp2.ndim == pitch.ndim == self.cheb.ndim))
        return bp1, bp2, pitch

    def check_bounce_points(self, bp1, bp2, pitch, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        bp1, bp2 : jnp.ndarray
            Shape must broadcast with (P, *self.cheb.shape[:-2], W).
            The field line-following coordinates of bounce points.
            The pairs ``bp1`` and ``bp2`` form left and right integration boundaries,
            respectively, for the bounce integrals.
        pitch : jnp.ndarray
            Shape must broadcast with (P, *self.cheb.shape[:-2]).
            λ values to evaluate the bounce integral.
        plot : bool
            Whether to plot stuff. Default is true.
        kwargs : dict
            Keyword arguments into ``plot_field_line``.

        """
        assert bp1.shape == bp2.shape
        mask = (bp1 - bp2) != 0.0
        bp1 = jnp.where(mask, bp1, jnp.nan)
        bp2 = jnp.where(mask, bp2, jnp.nan)
        bp1, bp2, pitch = self._check_shape(bp1, bp2, pitch)

        err_1 = jnp.any(bp1 > bp2, axis=-1)
        err_2 = jnp.any(bp1[..., 1:] < bp2[..., :-1], axis=-1)
        B_m = self.eval1d((bp1 + bp2) / 2)
        assert B_m.shape == bp1.shape
        err_3 = jnp.any(B_m > 1 / pitch + self._eps, axis=-1)
        if not (plot or jnp.any(err_1 | err_2 | err_3)):
            return

        # Ensure l axis exists for iteration in below loop.
        cheb = atleast_nd(3, self.cheb)
        mask, bp1, bp2, B_m = atleast_3d_mid(mask, bp1, bp2, B_m)
        err_1, err_2, err_3 = atleast_2d_end(err_1, err_2, err_3)

        for l in np.ndindex(cheb.shape[:-2]):
            for p in range(pitch.shape[0]):
                if not (err_1[p, l] or err_2[p, l] or err_3[p, l]):
                    continue
                _bp1 = bp1[p, l][mask[p, l]]
                _bp2 = bp2[p, l][mask[p, l]]
                if plot:
                    self.plot_field_line(
                        cheb[l],
                        pitch=pitch[p, l],
                        bp1=_bp1,
                        bp2=_bp2,
                        title_id=f"{p},{l}",
                        **kwargs,
                    )
                print("      bp1    |    bp2")
                print(jnp.column_stack([_bp1, _bp2]))
                assert not err_1[p, l], "Bounce points have an inversion.\n"
                assert not err_2[p, l], "Detected discontinuity.\n"
                assert not err_3[p, l], (
                    "Detected |B| > 1/λ in well. Increase Chebyshev resolution.\n"
                    f"{B_m[p, l][mask[p, l]]} > {1 / pitch[p, l] + self._eps}"
                )
            if plot:
                self.plot_field_line(
                    cheb[l],
                    pitch=pitch[:, l],
                    bp1=bp1[:, l],
                    bp2=bp2[:, l],
                    title_id=str(l),
                    **kwargs,
                )

    def plot_field_line(
        self,
        cheb,
        bp1=jnp.array([[]]),
        bp2=jnp.array([[]]),
        pitch=jnp.array([]),
        num=1000,
        title=r"Computed bounce points for $\vert B \vert$ and pitch $\lambda$",
        title_id=None,
        transparency_pitch=0.5,
        show=True,
    ):
        """Plot the field line given spline of |B|.

        Parameters
        ----------
        cheb : jnp.ndarray
            Piecewise Chebyshev coefficients of |B| along the field line.
        num : int
            Number of ζ points to plot. Pick a big number.
        bp1 : jnp.ndarray
            Bounce points with (∂|B|/∂ζ)|ρ,α <= 0.
        bp2 : jnp.ndarray
            Bounce points with (∂|B|/∂ζ)|ρ,α >= 0.
        pitch : jnp.ndarray
            λ value.
        title : str
            Plot title.
        title_id : str
            Identifier string to append to plot title.
        transparency_pitch : float
            Transparency of pitch lines.
        show : bool
            Whether to show the plot. Default is true.

        Returns
        -------
        fig, ax : matplotlib figure and axes.

        """
        legend = {}

        def add(lines):
            for line in setdefault(lines, [lines], hasattr(lines, "__iter__")):
                label = line.get_label()
                if label not in legend:
                    legend[label] = line

        fig, ax = plt.subplots()
        z = jnp.linspace(
            start=self.domain[0],
            stop=self.domain[0] + (self.domain[1] - self.domain[0]) * self.M,
            num=num,
        )
        add(ax.plot(z, self.eval1d(z, cheb), label=r"$\vert B \vert (\zeta)$"))

        if pitch is not None:
            b = 1 / jnp.atleast_1d(pitch)
            for val in b:
                add(
                    ax.axhline(
                        val,
                        color="tab:purple",
                        alpha=transparency_pitch,
                        label=r"$1 / \lambda$",
                    )
                )
            bp1, bp2 = jnp.atleast_2d(bp1, bp2)
            for i in range(bp1.shape[0]):
                if bp1.shape == bp2.shape:
                    _bp1, _bp2 = filter_bounce_points(bp1[i], bp2[i])
                else:
                    _bp1, _bp2 = bp1[i], bp2[i]
                add(
                    ax.scatter(
                        _bp1,
                        jnp.full_like(_bp1, b[i]),
                        marker="v",
                        color="tab:red",
                        label="bp1",
                    )
                )
                add(
                    ax.scatter(
                        _bp2,
                        jnp.full_like(_bp2, b[i]),
                        marker="^",
                        color="tab:green",
                        label="bp2",
                    )
                )

        ax.set_xlabel(r"Field line $\zeta$")
        ax.set_ylabel(r"$\vert B \vert \sim 1 / \lambda$")
        ax.legend(legend.values(), legend.keys(), loc="lower right")
        if title_id is not None:
            title = f"{title}. ID={title_id}."
        ax.set_title(title)
        plt.tight_layout()
        if show:
            plt.show()
            plt.close()
        return fig, ax


def _bounce_quadrature(bp1, bp2, x, w, m, n, integrand, f, b_sup_z, B, T, pitch):
    """Bounce integrate ∫ f(ℓ) dℓ.

    Parameters
    ----------
    bp1 : jnp.ndarray
        Shape (P, L, num_well).
        The field line-following coordinates of bounce points for a given pitch
        along a field line. The pairs ``bp1`` and ``bp2`` form left and right
        integration boundaries, respectively, for the bounce integrals.
    bp2 : jnp.ndarray
        Shape (P, L, num_well).
        The field line-following coordinates of bounce points for a given pitch
        along a field line. The pairs ``bp1`` and ``bp2`` form left and right
        integration boundaries, respectively, for the bounce integrals.
    x : jnp.ndarray
        Shape (w.size, ).
        Quadrature points in [-1, 1].
    w : jnp.ndarray
        Shape (w.size, ).
        Quadrature weights.
    m : int
        Poloidal periodic DESC coordinate resolution on which the given
         ``f`` and ``b_sup_z`` were evaluated.
    n : int
        Toroidal periodic DESC coordinate resolution on which the given
        ``f`` and ``b_sup_z`` were evaluated.
    integrand : callable
        The composition operator on the set of functions in ``f`` that maps the
        functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ. It should accept the
        arrays in ``f`` as arguments as well as the additional keyword arguments:
        ``B`` and ``pitch``. A quadrature will be performed to approximate the
        bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
    f : list of jnp.ndarray
        Shape (L * m * n, ).
        Arguments to the callable ``integrand``. These should be real scalar-valued
        functions in the bounce integrand evaluated on the periodic DESC coordinate
        (ρ, θ, ζ) tensor-product grid.
    b_sup_z : jnp.ndarray
        Shape (L, 1, m, n).
        Set of 2D Fourier spectral coefficients of B^ζ/|B|.
    B : PiecewiseChebyshevBasis
        Set of 1D Chebyshev spectral coefficients of |B| along field line.
        {|B|_α : ζ ↦ |B|(α, ζ) | α ∈ A }.
    T : PiecewiseChebyshevBasis
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {θ_α : ζ ↦ θ(α, ζ) | α ∈ A }.
    pitch : jnp.ndarray
        Shape (P, L).
        λ values to evaluate the bounce integral at each field line.

    Returns
    -------
    result : jnp.ndarray
        Shape (P, L, num_well).
        First axis enumerates pitch values. Second axis enumerates the field lines.
        Last axis enumerates the bounce integrals.

    """
    assert bp1.ndim == 3
    assert bp1.shape == bp2.shape
    assert x.ndim == 1
    assert x.shape == w.shape
    assert B.cheb.ndim == 3
    assert B.cheb.shape == T.cheb.shape
    assert pitch.ndim == 2

    P, L, num_well = bp1.shape
    shape = (P, L, num_well, x.size)
    # Quadrature points parameterized by ζ, for each pitch and flux surface.
    Q_zeta = _flatten_matrix(
        bijection_from_disc(
            x,
            bp1[..., jnp.newaxis],
            bp2[..., jnp.newaxis],
        )
    )
    # Quadrature points in (θ, ζ) coordinates.
    Q_desc = jnp.stack([T.eval1d(Q_zeta), Q_zeta], axis=-1)
    f = [interp_rfft2(Q_desc, f_i.reshape(L, 1, m, n)).reshape(shape) for f_i in f]
    result = jnp.dot(
        integrand(
            *f,
            B=B.eval1d(Q_zeta).reshape(shape),
            pitch=pitch[..., jnp.newaxis, jnp.newaxis],
        )
        / irfft2_non_uniform(Q_desc, b_sup_z, m, n).reshape(shape),
        w,
    )
    assert result.shape == (P, L, num_well)
    return result


def required_names():
    """Return names in ``data_index`` required to compute bounce integrals."""
    return ["B^zeta", "|B|", "iota"]


# TODO: Assumes zeta = phi (alpha sequence)
def bounce_integral(
    grid,
    data,
    M,
    N,
    desc_from_clebsch,
    alpha_0=0.0,
    num_transit=50,
    quad=leggauss(21),
    automorphism=(automorphism_sin, grad_automorphism_sin),
    B_ref=1.0,
    L_ref=1.0,
    check=False,
    plot=False,
    **kwargs,
):
    """Returns a method to compute bounce integrals.

    The bounce integral is defined as ∫ f(ℓ) dℓ, where
        dℓ parameterizes the distance along the field line in meters,
        λ is a constant proportional to the magnetic moment over energy,
        |B| is the norm of the magnetic field,
        f(ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points ζ₁, ζ₂ s.t. λ|B|(ζᵢ) = 1.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Parameters
    ----------
    grid : Grid
        Periodic tensor-product grid in (ρ, θ, ζ).
        Note that below shape notation defines
        L = ``grid.num_rho``, m = ``grid.num_theta``, and n = ``grid.num_zeta``.
    data : dict of jnp.ndarray
        Data evaluated on grid.
    M : int
        Grid resolution in poloidal direction for Clebsch coordinates.
        Preferably power of 2. A good choice is ``m``. If the poloidal stream
        function condenses the Fourier spectrum of |B| significantly, then a
        larger number may be beneficial.
    N : int
        Grid resolution in toroidal direction for Clebsch coordinates.
        Preferably power of 2.
    desc_from_clebsch : jnp.ndarray
        Shape (L * M * N, 3).
        DESC coordinate grid (ρ, θ, ζ) sourced from the Clebsch coordinate
         tensor-product grid (ρ, α, ζ) returned by
        ``FourierChebyshevBasis.nodes(M,N,domain=(0,2π))``.
    alpha_0 : float
        Starting field line poloidal label.
        TODO: Allow multiple starting labels for near-rational surfaces.
              Concatenate along second to last axis of cheb.
    num_transit : int
        Number of toroidal transits to follow field line.
    quad : (jnp.ndarray, jnp.ndarray)
        Quadrature points xₖ and weights wₖ for the approximate evaluation of an
        integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 21 points.
    automorphism : (Callable, Callable) or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines a
        change of variable for the bounce integral. The choice made for the automorphism
        will affect the performance of the quadrature method.
    B_ref : float
        Optional. Reference magnetic field strength for normalization.
        Has no effect on computation, but may be useful for analysis.
    L_ref : float
        Optional. Reference length scale for normalization.
        Has no effect on computation, but may be useful for analysis.
    check : bool
        Flag for debugging. Must be false for jax transformations.
    plot : bool
        Whether to plot stuff if ``check`` is true. Default is false.

    Returns
    -------
    bounce_integrate : callable
        This callable method computes the bounce integral ∫ f(ℓ) dℓ for every
        specified field line for every λ value in ``pitch``.
    spline : tuple(ndarray, PiecewiseChebyshevBasis, PiecewiseChebyshevBasis)
        alphas : jnp.ndarray
            Poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) that specify field line.
        B : PiecewiseChebyshevBasis
            Set of 1D Chebyshev spectral coefficients of |B| along field line.
            {|B|_α : ζ ↦ |B|(α, ζ) | α ∈ A }.
        T : PiecewiseChebyshevBasis
            Set of 1D Chebyshev spectral coefficients of θ along field line.
            {θ_α : ζ ↦ θ(α, ζ) | α ∈ A }.

    """
    # Strictly increasing zeta knots enforces dζ > 0.
    # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require B^ζ = B⋅∇ζ > 0.
    # This is equivalent to changing the sign of ∇ζ (or [∂ℓ/∂ζ]|ρ,a).
    warnif(
        check and kwargs.pop("warn", True) and jnp.any(data["B^zeta"] <= 0),
        msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Enforcing positive B^ζ.",
    )

    # Resolution of periodic DESC coordinate tensor-product grid.
    L, m, n = grid.num_rho, grid.num_theta, grid.num_zeta
    # Transform to DESC spectral domain.
    b_sup_z = rfft2(  # B^ζ(θ,ζ)
        (jnp.abs(data["B^zeta"]) / data["|B|"] * L_ref).reshape(L, 1, m, n),
        norm="forward",
    )
    domain = (0, 2 * jnp.pi)
    # Transform to Clebsch spectral domain.
    # We compute θ(α,ζ) to avoid nonlinear root finding later, and |B|(α,ζ)
    # so that roots are computable without inferior local search algorithms.
    T = FourierChebyshevBasis(desc_from_clebsch[:, 1].reshape(L, M, N), domain)
    B = FourierChebyshevBasis(
        interp_rfft2(
            xq=desc_from_clebsch[:, 1:].reshape(L, -1, 2),
            f=data["|B|"].reshape(L, 1, m, n) / B_ref,
        ).reshape(L, M, N),
        domain,
    )
    # Peel off field lines.
    alphas = get_alphas(alpha_0, grid.compress(data["iota"]), num_transit, domain[-1])
    T = T.compute_cheb(alphas)
    B = B.compute_cheb(alphas)
    assert T.cheb.shape == B.cheb.shape == (L, num_transit, N)
    # Evaluation of a set of Chebyshev series is always more efficient than evaluating
    # single Fourier Chebyshev series, so we also get Chebyshev series for θ.

    x, w = quad
    assert x.ndim == w.ndim == 1
    if automorphism is not None:
        auto, grad_auto = automorphism
        w = w * grad_auto(x)
        # Recall bijection_from_disc(auto(x), ζ_b₁, ζ_b₂) = ζ.
        x = auto(x)

    def bounce_integrate(integrand, f, pitch, weight=None, num_well=None):
        """Bounce integrate ∫ f(ℓ) dℓ.

        Parameters
        ----------
        integrand : callable
            The composition operator on the set of functions in ``f`` that maps the
            functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ. It should accept the
            arrays in ``f`` as arguments as well as the additional keyword arguments:
            ``B`` and ``pitch``. A quadrature will be performed to approximate the
            bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
        f : list of jnp.ndarray
            Shape (L * m * n, ) or (L, m, n).
            Arguments to the callable ``integrand``. These should be real scalar-valued
            functions in the bounce integrand evaluated on ``grid``.
        pitch : jnp.ndarray
            Shape (P, L).
            λ values to evaluate the bounce integral at each field line. λ(ρ) is
            specified by ``pitch[...,ρ]`` where in the latter the labels ρ are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        weight : jnp.ndarray
            Shape (L * m * n, ) or (L, m, n).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(ℓ) dℓ, where w(j) is ``weight``
            evaluated at the deepest point in the magnetic well.
        num_well : int or None
            If not specified, then all bounce integrals are returned in an array whose
            last axis has size ``num_transit*(N-1)``. If there were less than that many
            wells detected along a field line, then the last axis of the returned array,
            which enumerates bounce integrals for a particular field line and
            pitch, is padded with zero.

            Specify to return the bounce integrals between the first ``num_well``
            wells for each pitch along each field line. This is useful if ``num_well``
            tightly bounds the actual number of wells. To obtain a good
            choice for ``num_well``, plot the field line with all the bounce points
            identified. This will be done automatically if the ``bounce_integral``
            function is called with ``check=True`` and ``plot=True``. As a reference,
            there are typically <= 5 wells per toroidal transit.

        Returns
        -------
        result : jnp.ndarray
            Shape (P, L, num_well).
            First axis enumerates pitch values. Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.cd

        """
        errorif(weight is not None, NotImplementedError)
        pitch = jnp.atleast_2d(pitch)
        bp1, bp2 = B.bounce_points(pitch, num_well)
        if check:
            B.check_bounce_points(bp1, bp2, pitch, plot)
        result = _bounce_quadrature(
            bp1, bp2, x, w, m, n, integrand, f, b_sup_z, B, T, pitch
        )
        assert result.shape == (pitch.shape[0], L, setdefault(num_well, N - 1))
        return result

    return bounce_integrate, (alphas, B, T)
