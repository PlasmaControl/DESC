"""Methods for constructing f(α, ζ) splines and bounce integrals."""

from orthax.chebyshev import chebpts1, chebpts2, chebval

from desc.backend import dct, idct, irfft, jnp, put, rfft
from desc.compute._interp_utils import _filter_distinct, irfft_non_uniform
from desc.compute._quadrature_utils import affine_bijection as map_domain
from desc.compute._quadrature_utils import (
    affine_bijection_to_disc as map_domain_to_disc,
)
from desc.compute.bounce_integral import _fix_inversion
from desc.compute.utils import take_mask
from desc.utils import Index, errorif

# Vectorized versions of numpy functions. Need root finding to be as efficient as
# possible, so vectorize to solve stack of matrices. Also skip the slow input
# massaging because we don't allow duck typed lists.


def _chebcompanion(c):
    # Adapted from
    # numpy.org/doc/stable/reference/generated/
    # numpy.polynomial.chebyshev.chebcompanion.html.
    # github.com/f0uriest/orthax/blob/main/orthax/chebyshev.py.
    errorif(c.shape[-1] < 2, msg="Series must have maximum degree of at least 1.")
    if c.shape[-1] == 2:
        return jnp.array([[-c[..., 0] / c[..., 1]]])

    n = c.shape[-1] - 1
    scl = jnp.hstack([1.0, jnp.full(n - 1, jnp.sqrt(0.5))])
    mat = jnp.zeros((*c.shape[:-1], n, n), dtype=c.dtype)
    mat = put(mat, Index[..., 0, 0], jnp.sqrt(0.5))
    mat = put(mat, Index[..., 0, 1:], 0.5)
    mat = put(mat, Index[..., -1, :], mat[..., 0, :])
    mat = put(
        mat,
        Index[..., -1],
        mat[..., -1] - c[..., :-1] / c[..., -1] * scl / scl[-1] * 0.5,
    )
    return mat


def _chebroots(c):
    # Adapted from
    # numpy.org/doc/stable/reference/generated/
    # numpy.polynomial.chebyshev.chebroots.html.
    # github.com/f0uriest/orthax/blob/main/orthax/chebyshev.py,
    if c.shape[-1] < 2:
        return jnp.reshape([], (0,) * c.ndim)
    if c.shape[-1] == 2:
        return jnp.array([-c[..., 0] / c[..., 1]])

    # rotated companion matrix reduces error
    m = _chebcompanion(c)[..., ::-1, ::-1]
    # Low priority:
    # there are better techniques to find eigenvalues of Chebyshev colleague matrix.
    r = jnp.sort(jnp.linalg.eigvals(m))
    return r


def _cheb_from_dct(c):
    # Return Chebshev polynomial coefficients given forward dct type 2.
    return c.at[..., 0].divide(2.0) * 2


def _flatten_matrix(y):
    # Flatten batch of matrix to batch of vector.
    return y.reshape(*y.shape[:-2], -1)


def alpha_sequence(alpha_0, m, iota, period):
    """Get sequence of poloidal coordinates (α₀, α₁, …, αₘ₋₁) of field line.

    Parameters
    ----------
    alpha_0 : float
        Starting field line poloidal label.
    m : float
        Number of periods to follow field line.
    iota : jnp.ndarray
        Shape (iota.size, ).
        Rotational transform normalized by 2π.
    period : float
        Toroidal period after which to update label.

    Returns
    -------
    alpha : jnp.ndarray
        Shape (iota.size, m).
        Sequence of poloidal coordinates (α₀, α₁, …, αₘ₋₁) that specify field line.

    """
    # Δz (∂α/∂ζ) = Δz ι̅ = Δz ι/2π = Δz data["iota"]
    return (alpha_0 + period * iota[:, jnp.newaxis] * jnp.arange(m)) % (2 * jnp.pi)


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

    _eps = min(jnp.finfo(jnp.array(1.0).dtype).eps * 1e2, 1e-10)

    def __init__(self, f, lobatto=False, domain=(-1, 1)):
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
            Domain for y coordinates. Default is [-1, 1].

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
    def _fourier_pts(M):
        return -jnp.pi + 2 * jnp.pi * jnp.arange(1, M + 1) / M

    # Y = [a, b] evaluate on grid -> y = [-1, 1] chebyshev points -> y = cos(z)
    # evenly spaced z.
    # So I find coefficients to chebyshev series T_n(y) = cos(n arcos(y)) = cos(n z).
    # So evaluating my chebyshev series in y is same as evaluting cosine series in
    # z = arcos(y).
    # for y = inversemap[a, b].
    # Open questions is finding roots y using chebroots better or is finding roots z
    # of trig poly.
    # answer: research shows doesn't really matter.
    @staticmethod
    def _chebyshev_pts(N, lobatto, domain=(-1, 1)):
        y = chebpts2(N) if lobatto else chebpts1(N)
        return map_domain(y, domain[0], domain[-1])

    @staticmethod
    def nodes(M, N, lobatto=False, domain=(-1, 1), **kwargs):
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
            Domain for y coordinates. Default is [-1, 1].

        Returns
        -------
        coords : jnp.ndarray
            Shape (M * N, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        x = FourierChebyshevBasis._fourier_pts(M)
        y = FourierChebyshevBasis._chebyshev_pts(N, lobatto, domain)
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
        f : jnp.ndarray
            Shape (..., M, N)
            Fourier-Chebyshev series evaluated at ``FourierChebyshevBasis.nodes(M, N)``.

        """
        f = idct(
            irfft(self._c, n=M, axis=-2) * M,
            type=2 - self.lobatto,
            n=N,
            axis=-1,
        ) * (N - self.lobatto)
        return f

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
        c = _cheb_from_dct(self._c)
        # Convert rfft to Nyquist trigonometric harmonics.
        is_even = (self.M % 2) == 0
        # ∂ₓ = 0 coefficients
        a0 = jnp.real(c[..., 0, :])[..., jnp.newaxis, :]
        # cos(mx) Tₙ(y) coefficients
        an = jnp.real(c[..., 1:, :].at[..., -1, :].divide(1.0 + is_even)) * 2
        # sin(mx) Tₙ(y) coefficients
        bn = jnp.imag(c[..., 1 : c.shape[-2] - is_even, :]) * (-2)

        a_mn = jnp.concatenate([a0, an, bn], axis=-2)
        assert a_mn.shape[-2:] == (self.M, self.N)
        return a_mn

    def compute_cheb(self, x):
        """Evaluate Fourier basis at ``x`` to obtain set of 1d Chebyshev coefficients.

        Parameters
        ----------
        x : jnp.ndarray
            Shape (..., x.shape[-1]).
            Evaluation points. If 1d assumes batch dimension over L is implicit
            (i.e. standard numpy broadcasting rules).

        Returns
        -------
        cheb : jnp.ndarray
            Shape (..., x.shape[-1], N).
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        x = jnp.array(x, ndmin=self._c.ndim)
        errorif(x.ndim != self._c.ndim, NotImplementedError)
        cheb = _cheb_from_dct(
            irfft_non_uniform(x, jnp.swapaxes(self._c, -1, -2), self.M)
        )
        cheb = jnp.swapaxes(cheb, -1, -2)
        assert cheb.shape == (*self._c.shape[:-2], x.shape[-1], self.N)
        return cheb

    def y_intersect(self, cheb, k=0, eps=_eps):
        """Coordinates yᵢ such that f(x, yᵢ) = k(x).

        Parameters
        ----------
        cheb : jnp.ndarray
            Shape (..., N).
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).
        k : jnp.ndarray
            Shape (..., *cheb.shape).
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
        assert cheb.shape[-1] == self.N
        c = cheb[jnp.newaxis] if k.ndim > cheb.ndim else cheb
        c = c.at[..., 0].add(-k)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = _chebroots(c)
        assert y.shape == (*c.shape[:-1], self.N - 1)

        y = _filter_distinct(y, sentinel=-2, eps=eps)
        # Pick sentinel above such that only distinct roots are considered intersects.
        is_intersect = (jnp.abs(jnp.imag(y)) <= eps) & (jnp.abs(jnp.real(y)) <= 1)
        y = jnp.where(is_intersect, jnp.real(y), 0)  # ensure y is in domain of arcos
        #      ∂f/∂y =      ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n Uₙ₋₁(y)
        # sign ∂f/∂y = sign ∑ₙ₌₁ᴺ⁻¹ aₙ(x) sin(n arcos y)
        s = jnp.einsum(
            # TODO: Multipoint evaluation with FFT.
            #   Chapter 10, https://doi.org/10.1017/CBO9781139856065.
            "...n,...yn",
            cheb,
            jnp.sin(jnp.arange(self.N) * jnp.arccos(y)[..., jnp.newaxis]),
        )
        is_decreasing = s <= 0
        is_increasing = s >= 0

        y = map_domain(y, self.domain[0], self.domain[-1])
        return y, is_decreasing, is_increasing, is_intersect

    def bounce_points(
        self, y, is_decreasing, is_increasing, is_intersect, num_wells=None
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
        num_wells : int
            If not specified, then all bounce points are returned in an array whose
            last axis has size ``y.shape[-1] * y.shape[-2]``. If there
            were less than that many wells detected along a field line, then the last
            axis of the returned arrays, which enumerates bounce points for a particular
            field line and pitch, is padded with zero.

            Specify to return the first ``num_wells`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_wells`` tightly
            bounds the actual number of wells. As a reference, there are
            typically <= 5 wells per toroidal transit.

        Returns
        -------
        bp1, bp2 : (jnp.ndarray, jnp.ndarray)
            Shape (*y.shape[:-2], num_wells).
            The field line-following coordinates of bounce points for a given pitch
            along a field line. The pairs ``bp1`` and ``bp2`` form left and right
            integration boundaries, respectively, for the bounce integrals.

        """
        # Flatten so that last axis enumerates intersects of a pitch along a field line.
        y = _flatten_matrix(self._isomorphism_1d(y))
        is_decreasing = _flatten_matrix(is_decreasing)
        is_increasing = _flatten_matrix(is_increasing)
        is_intersect = _flatten_matrix(is_intersect)
        is_bp1 = is_decreasing & is_intersect
        is_bp2 = is_increasing & _fix_inversion(is_intersect, is_increasing)

        sentinel = self.domain[0] - 1
        bp1 = take_mask(y, is_bp1, size=num_wells, fill_value=sentinel)
        bp2 = take_mask(y, is_bp2, size=num_wells, fill_value=sentinel)

        mask = (bp1 > sentinel) & (bp2 > sentinel)
        # Set outside mask to same value so integration is over set of measure zero.
        bp1 = jnp.where(mask, bp1, 0)
        bp2 = jnp.where(mask, bp2, 0)
        # These typically have shape (num pitch, num rho, num wells).
        return bp1, bp2

    def interp_cheb_spline(self, z, cheb):
        """Evaluate piecewise Chebyshev spline at coordinates z.

        The coordinates z ∈ ℝ are assumed isomorphic to (x, y) ∈ ℝ²
        where z integer division domain yields index into the proper
        Chebyshev series of the spline and z mod domain is the coordinate
        value along the domain of that Chebyshev series.

        Parameters
        ----------
        z : jnp.ndarray
            Shape (*cheb.shape[:-2], num wells, num quadrature points).
            Isomorphic coordinates along field line [0, inf].
        cheb: jnp.ndarray
            Shape (..., num cheb series, N).
            Chebyshev coefficients αₙ for f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z]).

        Returns
        -------
        f : jnp.ndarray
            Shape z.shape.
            Chebyshev basis evaluated at z.

        """
        # TODO: Multipoint evaluation with FFT.
        #   Chapter 10, https://doi.org/10.1017/CBO9781139856065.
        x_idx, y = map(_flatten_matrix, self._isomorphism_2d(z))
        y = map_domain_to_disc(y, self.domain[0], self.domain[1])
        cheb = jnp.moveaxis(cheb, source=-1, destination=0)
        cheb = jnp.take_along_axis(cheb, x_idx, axis=-1, mode="promise_in_bounds")
        f = chebval(y, cheb, tensor=False).reshape(z.shape)
        # TODO: Add below as unit test.
        # n = jnp.arange(self.N) # noqa: E800
        # T = jnp.cos(n * jnp.arccos(y)[..., jnp.newaxis]) # noqa: E800
        # f = jnp.einsum("...n,n...", T, cheb).reshape(z.shape) # noqa: E800
        return f

    def _isomorphism_1d(self, y):
        """Return coordinates z ∈ ℂ isomorphic to (x, y) ∈ ℂ².

        Maps row x of y to z = α(x) + y where α(x) = x * |domain|.

        Parameters
        ----------
        y : jnp.ndarray
            Shape (..., *y.shape[-2:]).
            Second to last axis iterates the rows.

        Returns
        -------
        z : jnp.ndarray
            Shape y.shape.
            Isomorphic coordinates.

        """
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
        x_index : jnp.ndarray
            Shape y.shape.
            Isomorphic coordinates.

        """
        period = self.domain[-1] - self.domain[0]
        x_index = z // period
        y_value = z % period
        return x_index, y_value


def bounce_integral(data, M, N, rho):
    """WIP."""
    cheb_nodes = FourierChebyshevBasis.nodes(M, N, domain=(0, 2 * jnp.pi), rho=rho)
    return cheb_nodes
