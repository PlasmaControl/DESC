"""Methods for computing Fourier Chebyshev FFTs and bounce integrals."""

import numpy as np
from matplotlib import pyplot as plt
from orthax.chebyshev import chebroots, chebvander
from orthax.legendre import leggauss

from desc.backend import dct, idct, irfft, jnp, rfft, rfft2
from desc.compute._interp_utils import (
    _filter_distinct,
    cheb_from_dct,
    cheb_pts,
    fourier_pts,
    harmonic,
    interp_rfft2,
    irfft2_non_uniform,
    irfft_non_uniform,
)
from desc.compute._quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    bijection_to_disc,
    grad_automorphism_sin,
)
from desc.compute.bounce_integral import _filter_nonzero_measure, _fix_inversion
from desc.compute.utils import take_mask
from desc.utils import errorif, warnif

# TODO: There are better techniques to find eigenvalues of Chebyshev colleague matrix.
_chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


def _flatten_matrix(y):
    # Flatten batch of matrix to batch of vector.
    return y.reshape(*y.shape[:-2], -1)


def _alpha_sequence(alpha_0, iota, num_period, period=2 * jnp.pi):
    """Get sequence of poloidal coordinates (α₀, α₁, …, αₘ₋₁) of field line.

    Parameters
    ----------
    alpha_0 : float
        Starting field line poloidal label.
    iota : jnp.ndarray
        Shape (iota.size, ).
        Rotational transform normalized by 2π.
    num_period : float
        Number of periods to follow field line.
    period : float
        Toroidal period after which to update label.

    Returns
    -------
    alpha : jnp.ndarray
        Shape (iota.size, m).
        Sequence of poloidal coordinates (α₀, α₁, …, αₘ₋₁) that specify field line.

    """
    # Δϕ (∂α/∂ϕ) = Δϕ ι̅ = Δϕ ι/2π = Δϕ data["iota"]
    return (alpha_0 + period * iota[:, jnp.newaxis] * jnp.arange(num_period)) % (
        2 * jnp.pi
    )


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

    def __init__(self, f, lobatto=False, domain=(0, 2 * jnp.pi)):
        """Interpolate Fourier-Chebyshev basis to ``f``.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (..., M, N).
            Samples of real function on the ``FourierChebyshevBasis.nodes`` grid.
            M, N preferably power of 2.
        lobatto : bool
            Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
            or interior roots grid for Chebyshev points.
        domain : (float, float)
            Domain for y coordinates. Default is [0, 2π].

        """
        errorif(domain[0] > domain[-1], msg="Got inverted y coordinate domain.")
        errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
        self.M = f.shape[-2]
        self.N = f.shape[-1]
        self.lobatto = bool(lobatto)
        self.domain = domain
        self._c = (
            rfft(
                dct(f, type=2 - self.lobatto, axis=-1) / (self.N - self.lobatto),
                axis=-2,
            )
            / self.M
        )

    @staticmethod
    def nodes(M, N, lobatto=False, domain=(0, 2 * jnp.pi), **kwargs):
        """Tensor product grid of optimal collocation nodes for this basis.

        Parameters
        ----------
        M : int
            Grid resolution in x direction. Preferably power of 2.
        N : int
            Grid resolution in y direction. Preferably power of 2.
        lobatto : bool
            Whether to use the Gauss-Lobatto (Extrema-plus-Endpoint)
            or interior roots grid for Chebyshev points.
        domain : (float, float)
            Domain for y coordinates. Default is [0, 2π].

        Returns
        -------
        coords : jnp.ndarray
            Shape (M * N, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        x = fourier_pts(M)
        y = cheb_pts(N, lobatto, domain)
        coords = [kwargs.pop("rho"), x, y] if "rho" in kwargs else [x, y]
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
        fq = idct(
            irfft(self._c, n=M, axis=-2) * M,
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
            Shape (..., x.shape[-1]).
            Evaluation points. If 1d assumes batch dimension over L is implicit
            (i.e. standard numpy broadcasting rules).

        Returns
        -------
        cheb : _PiecewiseChebyshevBasis
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        # Always add new axis to broadcast against Chebyshev coefficients.
        x = jnp.atleast_1d(x)[..., jnp.newaxis]
        cheb = cheb_from_dct(irfft_non_uniform(x, self._c, self.M, axis=-2), axis=-1)
        assert cheb.shape[-2:] == (x.shape[-1], self.N)
        return _PiecewiseChebyshevBasis(cheb, self.domain)


class _PiecewiseChebyshevBasis:
    """Chebyshev series.

    { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ].

    Attributes
    ----------
    cheb : jnp.ndarray
        Shape (..., N).
        Chebyshev coefficients αₙ(x) for fₓ(y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).
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
            Shape (..., N).
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        self.cheb = cheb
        self.N = cheb.shape[-1]
        self.domain = domain

    def intersect(self, k=0, eps=_eps):
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
        errorif(
            k.ndim > self.cheb.ndim,
            NotImplementedError,
            msg=f"Got k.ndim {k.ndim} > cheb.ndim {self.cheb.ndim}.",
        )
        c = self.cheb if k.ndim < self.cheb.ndim else self.cheb[jnp.newaxis]
        c = c.copy().at[..., 0].add(-k)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = _chebroots_vec(c)
        assert y.shape == (*c.shape[:-1], self.N - 1)

        y = _filter_distinct(y, sentinel=-2, eps=eps)
        # Pick sentinel above such that only distinct roots are considered intersects.
        is_intersect = (jnp.abs(jnp.imag(y)) <= eps) & (jnp.abs(jnp.real(y)) <= 1)
        y = jnp.where(is_intersect, jnp.real(y), 0)  # ensure y is in domain of arcos
        #      ∂f/∂y =      ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n Uₙ₋₁(y)
        # sign ∂f/∂y = sign ∑ₙ₌₁ᴺ⁻¹ aₙ(x) sin(n arcos y)
        s = jnp.linalg.vecdot(
            # TODO: Multipoint evaluation with FFT.
            #   Chapter 10, https://doi.org/10.1017/CBO9781139856065.
            self.cheb[..., jnp.newaxis, :],
            jnp.sin(jnp.arange(self.N) * jnp.arccos(y)[..., jnp.newaxis]),
        )
        is_decreasing = s <= 0
        is_increasing = s >= 0

        y = bijection_from_disc(y, self.domain[0], self.domain[-1])
        return y, is_decreasing, is_increasing, is_intersect

    def bounce_points(
        self, y, is_decreasing, is_increasing, is_intersect, num_well=None
    ):
        """Compute bounce points given intersections.

        Parameters
        ----------
        y : jnp.ndarray
            Shape (..., *y.shape[-2:]).
            Solutions yᵢ of f(x, yᵢ) = k(x), in ascending order.
            Assumes the -2nd axis enumerates over poloidal coordinates
            all belonging to a single field line. See ``alpha_sequence``.
        is_decreasing : jnp.ndarray
            Shape y.shape.
            Whether ∂f/∂y (x, yᵢ) is decreasing.
        is_increasing : jnp.ndarray
            Shape y.shape.
            Whether ∂f/∂y (x, yᵢ) is increasing.
        is_intersect : jnp.ndarray
            Shape y.shape.
            Boolean array into ``y`` indicating whether element is an intersect.
        num_well : int or None
            If not specified, then all bounce points are returned in an array whose
            last axis has size ``y.shape[-1]*y.shape[-2]``. If there
            were less than that many wells detected along a field line, then the last
            axis of the returned arrays, which enumerates bounce points for a particular
            field line and pitch, is padded with zero.

            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number of wells. As a reference, there are
            typically <= 5 wells per toroidal transit.

        Returns
        -------
        bp1, bp2 : (jnp.ndarray, jnp.ndarray)
            Shape (*y.shape[:-2], num_well).
            The field line-following coordinates of bounce points for a given pitch
            along a field line. The pairs ``bp1`` and ``bp2`` form left and right
            integration boundaries, respectively, for the bounce integrals.

        """
        # Flatten so that last axis enumerates intersects of a pitch along a field line.
        y = _flatten_matrix(self._isomorphism_1d(y))
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

        sentinel = self.domain[0] - 1
        bp1 = take_mask(y, is_bp1, size=num_well, fill_value=sentinel)
        bp2 = take_mask(y, is_bp2, size=num_well, fill_value=sentinel)

        mask = (bp1 > sentinel) & (bp2 > sentinel)
        # Set outside mask to same value so integration is over set of measure zero.
        bp1 = jnp.where(mask, bp1, 0)
        bp2 = jnp.where(mask, bp2, 0)
        return bp1, bp2

    def plot_field_line(
        self,
        start,
        stop,
        num=1000,
        bp1=np.array([]),
        bp2=np.array([]),
        pitch=np.array([]),
        title=r"Computed bounce points for $\vert B \vert$ and pitch $\lambda$",
        title_id=None,
        transparency_pitch=0.3,
        show=True,
    ):
        """Plot the field line given spline of |B|.

        Parameters
        ----------
        start : float
            Minimum ζ on plot.
        stop : float
            Maximum ζ on plot.
        num : int
            Number of ζ points to plot. Pick a big number.
        bp1 : np.ndarray
            Bounce points with (∂|B|/∂ζ)|ρ,α <= 0.
        bp2 : np.ndarray
            Bounce points with (∂|B|/∂ζ)|ρ,α >= 0.
        pitch : np.ndarray
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
        errorif(start is None or stop is None)
        legend = {}

        def add(lines):
            if not hasattr(lines, "__iter__"):
                lines = [lines]
            for line in lines:
                label = line.get_label()
                if label not in legend:
                    legend[label] = line

        fig, ax = plt.subplots()
        z = np.linspace(start=start, stop=stop, num=num)
        add(ax.plot(z, self.eval1d(z), label=r"$\vert B \vert (\zeta)$"))

        if pitch is not None:
            b = 1 / np.atleast_1d(pitch)
            for val in b:
                add(
                    ax.axhline(
                        val,
                        color="tab:purple",
                        alpha=transparency_pitch,
                        label=r"$1 / \lambda$",
                    )
                )
            bp1, bp2 = np.atleast_2d(bp1, bp2)
            for i in range(bp1.shape[0]):
                bp1_i, bp2_i = _filter_nonzero_measure(bp1[i], bp2[i])
                add(
                    ax.scatter(
                        bp1_i,
                        np.full_like(bp1_i, b[i]),
                        marker="v",
                        color="tab:red",
                        label="bp1",
                    )
                )
                add(
                    ax.scatter(
                        bp2_i,
                        np.full_like(bp2_i, b[i]),
                        marker="^",
                        color="tab:green",
                        label="bp2",
                    )
                )

        ax.set_xlabel(r"Field line $\zeta$")
        ax.set_ylabel(r"$\vert B \vert \sim 1 / \lambda$")
        ax.legend(legend.values(), legend.keys(), loc="lower right")
        if title_id is not None:
            title = f"{title}. id = {title_id}."
        ax.set_title(title)
        plt.tight_layout()
        if show:
            plt.show()
            plt.close()
        return fig, ax

    def check_bounce_points(
        self, bp1, bp2, pitch, plot=True, start=None, stop=None, **kwargs
    ):
        """Check that bounce points are computed correctly."""
        pitch = jnp.atleast_3d(pitch)
        errorif(not (pitch.ndim == bp1.ndim == bp2.ndim == 3), NotImplementedError)
        errorif(bp1.shape != bp2.shape)

        P, L, num_wells = bp1.shape
        msg_1 = "Bounce points have an inversion."
        err_1 = jnp.any(bp1 > bp2, axis=-1)
        msg_2 = "Discontinuity detected."
        err_2 = jnp.any(bp1[..., 1:] < bp2[..., :-1], axis=-1)

        for l in range(L):
            for p in range(P):
                B_mid = self.eval1d((bp1[p, l] + bp2[p, l]) / 2)
                err_3 = jnp.any(B_mid > 1 / pitch[p, l] + self._eps)
                if err_1[p, l] or err_2[p, l] or err_3:
                    bp1_p, bp2_p = _filter_nonzero_measure(bp1[p, l], bp2[p, l])
                    B_mid = B_mid[(bp1[p, l] - bp2[p, l]) != 0]
                    if plot:
                        self.plot_field_line(
                            start=start,
                            stop=stop,
                            pitch=pitch[p, l],
                            bp1=bp1_p,
                            bp2=bp2_p,
                            title_id=f"{p},{l}",
                            **kwargs,
                        )
                    print("bp1:", bp1_p)
                    print("bp2:", bp2_p)
                    assert not err_1[p, l], msg_1
                    assert not err_2[p, l], msg_2
                    msg_3 = (
                        f"Detected B midpoint = {B_mid}>{1 / pitch[p, l] + self._eps} ="
                        " 1/pitch. You need to use more knots."
                    )
                    assert not err_3, msg_3
            if plot:
                self.plot_field_line(
                    start=start,
                    stop=stop,
                    pitch=pitch[:, l],
                    bp1=bp1[:, l],
                    bp2=bp2[:, l],
                    title_id=str(l),
                    **kwargs,
                )

    def eval1d(self, z):
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

        Returns
        -------
        f : jnp.ndarray
            Shape z.shape.
            Chebyshev basis evaluated at z.

        """
        x_idx, y = self._isomorphism_2d(z)
        y = bijection_to_disc(y, self.domain[0], self.domain[1])
        # Chebyshev coefficients αₙ for f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z])
        # are held in self.cheb with shape (..., num cheb series, N).
        cheb = jnp.moveaxis(self.cheb, source=-1, destination=0)
        cheb = jnp.take_along_axis(cheb, x_idx, axis=-1)
        # TODO: Multipoint evaluation with FFT.
        #   Chapter 10, https://doi.org/10.1017/CBO9781139856065.
        f = jnp.linalg.vecdot(chebvander(y, self.N - 1), cheb)
        return f

    def _isomorphism_1d(self, y):
        """Return coordinates z ∈ ℂ isomorphic to (x, y) ∈ ℂ².

        Maps row x of y to z = α(x) + y where α(x) = x * |domain|.

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
        period = self.domain[-1] - self.domain[0]
        zeta_shift = period * jnp.arange(y.shape[-2])
        z = zeta_shift[:, jnp.newaxis] + y
        return z

    def _isomorphism_2d(self, z):
        """Return coordinates (x, y) ∈ ℂ² isomorphic to z ∈ ℂ.

        Returns index x and value y such that z = α(x) + y where α(x) = x * |domain|.

        Parameters
        ----------
        z : jnp.ndarray
            Shape z.shape.

        Returns
        -------
        x_index, y_value : (jnp.ndarray, jnp.ndarray)
            Shape z.shape.
            Isomorphic coordinates.

        """
        period = self.domain[-1] - self.domain[0]
        x_index = z // period
        y_value = z % period
        return x_index, y_value


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
        Shape (L * m * n, ) or (L, m, n) or (L, 1, m, n).
        Arguments to the callable ``integrand``. These should be real scalar-valued
        functions in the bounce integrand evaluated on the periodic DESC coordinate
        (ρ, θ, ζ) tensor-product grid.
    b_sup_z : jnp.ndarray
        Shape (L, 1, m, n).
        Set of 2D Fourier spectral coefficients of B^ζ/|B|.
    B : jnp.ndarray
        Set of 1D Chebyshev spectral coefficients of |B| along field line.
    T : jnp.ndarray
        Set of 1D Chebyshev spectral coefficients of θ along field line.
    pitch : jnp.ndarray
        Shape (P, L, 1).
        λ values to evaluate the bounce integral at each field line.

    Returns
    -------
    result : jnp.ndarray
        Shape (P, S, num_well).
        First axis enumerates pitch values. Second axis enumerates the field lines.
        Last axis enumerates the bounce integrals.

    """
    errorif(bp1.ndim != 3 or bp1.shape != bp2.shape)
    errorif(pitch.ndim != 3)
    errorif(x.ndim != 1 or x.shape != w.shape)
    errorif(
        B.cheb.shape != T.cheb.shape
        or B.cheb.ndim != 3
        or B.cheb.shape[0] != bp1.shape[1]
    )

    P, L, num_well = bp1.shape
    shape = (P, L, num_well, x.size)
    # Quadrature points parameterized by ζ, for each pitch and flux surface.
    Q_zeta = _flatten_matrix(
        bijection_from_disc(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis])
    )
    # Quadrature points in DESC (θ, ζ) coordinates.
    Q_desc = jnp.stack([T.eval1d(Q_zeta), Q_zeta], axis=-1)
    f = [interp_rfft2(Q_desc, f_i.reshape(L, 1, m, n)).reshape(shape) for f_i in f]
    result = jnp.dot(
        integrand(*f, B=B.eval1d(Q_zeta).reshape(shape), pitch=pitch[..., jnp.newaxis])
        / irfft2_non_uniform(Q_desc, b_sup_z, m, n).reshape(shape),
        w,
    )
    assert result.shape == (P, L, num_well)


# TODO: Assumes zeta = phi
# input is
# that clebsch = FourierChebyshevBasis.nodes(M, N, rho=grid.compress(data["rho"]))
# then get desc_from_clebsch = map_coordinates(clebsch)
def bounce_integral(
    grid,
    data,
    M,
    N,
    desc_from_clebsch,
    alpha_0,
    num_transit,
    quad=leggauss(21),
    automorphism=(automorphism_sin, grad_automorphism_sin),
    B_ref=1.0,
    L_ref=1.0,
    check=False,
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
        Note that below shape notation uses ``L=grid.num_rho``, ``m=grid.num_theta``,
        and ``n=grid.num_zeta``.
    data : dict of jnp.ndarray
        Data evaluated on grid.
    M : int
        Grid resolution in poloidal direction for Clebsch coordinates.
        Preferably power of 2. A good choice is ``grid.num_theta``.
    N : int
        Grid resolution in toroidal direction for Clebsch coordinates.
        Preferably power of 2.
    desc_from_clebsch : jnp.ndarray
        Shape (L * M * N, 3).
        DESC coordinate grid (ρ, θ, ζ) sourced from the Clebsch coordinate
        tensor-product grid (ρ, α, ζ) returned by ``FourierChebyshevBasis.nodes(M, N)``.
    alpha_0 : float
        Starting field line poloidal label.
        TODO: Allow multiple starting labels for near-rational surfaces.
    num_transit : int
        Number of toroidal transits to follow field line.
    quad : (jnp.ndarray, jnp.ndarray)
        Quadrature points xₖ and weights wₖ for the approximate evaluation of an
        integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 21 points.
    automorphism : (Callable, Callable) or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines a
        change of variable for the bounce integral. The choice made for the automorphism
        can affect the performance of the quadrature method.
    B_ref : float
        Optional. Reference magnetic field strength for normalization.
    L_ref : float
        Optional. Reference length scale for normalization.
    check : bool
        Flag for debugging. Must be false for jax transformations.

    Returns
    -------
    bounce_integrate : callable
        This callable method computes the bounce integral ∫ f(ℓ) dℓ for every
        specified field line for every λ value in ``pitch``.

    """
    # Resolution of periodic DESC coordinate tensor-product grid.
    L, m, n = grid.num_rho, grid.num_theta, grid.num_zeta
    # Strictly increasing zeta knots enforces dζ > 0.
    # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require B^ζ = B⋅∇ζ > 0.
    # This is equivalent to changing the sign of ∇ζ.
    warnif(
        check and kwargs.pop("warn", True) and jnp.any(data["B^zeta"] <= 0),
        msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Enforcing positive B^ζ.",
    )

    # Transform to periodic DESC spectral domain.
    b_sup_z = rfft2(
        (jnp.abs(data["B^zeta"]) / data["|B|"] * L_ref).reshape(L, 1, m, n),
        norm="forward",
    )
    # Transform to non-periodic Clebsch spectral domain.
    T = FourierChebyshevBasis(desc_from_clebsch[:, 1].reshape(L, M, N))  # θ(α, ζ)
    B = FourierChebyshevBasis(  # |B|(α, ζ)
        interp_rfft2(
            xq=desc_from_clebsch[:, 1:].reshape(L, -1, 2),
            f=data["|B|"].reshape(L, m, n) / B_ref,
        ).reshape(L, M, N),
    )
    # Peel off field lines.
    alpha = _alpha_sequence(alpha_0, grid.compress(data["iota"]), num_transit)
    T = T.compute_cheb(alpha)
    B = B.compute_cheb(alpha)
    assert T.cheb.shape == B.cheb.shape == (L, num_transit, N)

    x, w = quad
    assert x.ndim == w.ndim == 1
    if automorphism is not None:
        auto, grad_auto = automorphism
        w = w * grad_auto(x)
        # Recall affine_bijection(auto(x), ζ_b₁, ζ_b₂) = ζ.
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
            last axis has size ``(N-1)*num_transit``. If there
            were less than that many wells detected along a field line, then the last
            axis of the returned array, which enumerates bounce integrals for a
            particular field line and pitch, is padded with zero.

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
            Last axis enumerates the bounce integrals.

        """
        errorif(weight is not None, NotImplementedError)
        # Compute bounce points.
        pitch = jnp.atleast_3d(pitch)
        P = pitch.shape[0]
        assert pitch.shape[1:] == B.cheb.shape[:-1]
        bp1, bp2 = B.bounce_points(*B.intersect(1 / pitch), num_well)
        num_well = bp1.shape[-1]
        assert bp1.shape == bp2.shape == (P, L, num_well)

        result = _bounce_quadrature(
            bp1, bp2, x, w, m, n, integrand, f, b_sup_z, B, T, pitch
        )
        return result

    return bounce_integrate
