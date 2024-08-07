"""Methods for computing bounce integrals."""

from orthax.chebyshev import chebpts1, chebpts2

from desc.backend import dct, idct, irfft, jnp, put, rfft
from desc.compute.bounce_integral import _filter_distinct, _fix_inversion, _take_mask
from desc.compute.bounce_integral import affine_bijection as map_domain
from desc.compute.bounce_integral import affine_bijection_to_disc as map_domain_to_disc
from desc.grid import Grid
from desc.utils import Index, errorif

_eps = min(jnp.finfo(jnp.array(1.0).dtype).eps * 1e2, 1e-10)


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
# TODO: could try boyd. eq. 16.46 pg 336
def _chebyshev_pts(N, lobatto, domain=(-1, 1)):
    y = chebpts2(N) if lobatto else chebpts1(N)
    return map_domain(y, domain[0], domain[-1])


# Vectorized versions of numpy functions. Need root finding to be as efficient as
# possible, so manually vectorize to solve stack of matrices with single LAPACK call.
# Also skip the slow input massaging because we don't allow duck typed lists.


def _chebcompanion(c):
    # Adapted from
    # https://numpy.org/doc/stable/reference/generated/
    # numpy.polynomial.chebyshev.chebcompanion.html.
    # https://github.com/f0uriest/orthax/blob/main/orthax/chebyshev.py.
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
    # https://numpy.org/doc/stable/reference/generated/
    # numpy.polynomial.chebyshev.chebroots.html.
    # https://github.com/f0uriest/orthax/blob/main/orthax/chebyshev.py,
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


class FourierChebyshevBasis:
    """Fourier-Chebyshev series.

    f(x, y) = ∑ₘₙ aₘₙ ψₘ(x) Tₙ(y)
    where ψₘ are trigonometric functions and Tₙ are Chebyshev polynomials
    on domain [−yₘᵢₙ, yₘₐₓ].

    Attributes
    ----------
    L : int
        Batch dimension size.
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

    def __init__(self, f, M, N, lobatto=False, domain=(-1, 1)):
        """Interpolate Fourier-Chebyshev basis to ``f``.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (..., M, N).
            Samples of function on the ``FourierChebyshevBasis.nodes`` grid.
        M : int
            Grid resolution in x direction. Preferably power of 2.
        N : int
            Grid resolution in y direction. Preferably power of 2.
        lobatto : bool
            Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
            or interior roots grid for Chebyshev points.
        domain : (float, float)
            Domain for y coordinates. Default is [-1, 1].

        """
        errorif(domain[0] > domain[-1], msg="Got inverted y coordinate domain.")
        errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
        self.domain = domain
        self.lobatto = bool(lobatto)
        self._c = rfft(
            dct(f.reshape(-1, M, N), type=2 - self.lobatto, axis=-1),
            axis=-2,
        )
        self.N = N
        self.M = M
        self.L = self._c.shape[0]
        self._a_n = None

    @classmethod
    def nodes(cls, M, N, lobatto=False, domain=(-1, 1), **kwargs):
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
        grid : jnp.ndarray
            Shape (M * N, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        x = _fourier_pts(M)
        y = _chebyshev_pts(N, lobatto, domain)
        if "rho" in kwargs:
            # then user wants a 3D DESC grid
            grid = Grid.create_meshgrid([kwargs.pop("rho"), x, y], **kwargs)
        else:
            xx, yy = map(jnp.ravel, jnp.meshgrid(x, y, indexing="ij"))
            grid = jnp.column_stack([xx, yy])
        return grid

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
            Shape (L, M, N)
            Fourier-Chebyshev series evaluated at ``FourierChebyshevBasis.nodes(M, N)``.

        """
        f = (
            idct(
                irfft(self._c, n=M, axis=-2) * M / self.M,
                type=2 - self.lobatto,
                n=N,
                axis=-1,
            )
            * (N - self.lobatto)
            / (self.N - self.lobatto)
        )
        return f

    def _harmonics(self):
        """Spectral coefficients aₘₙ of the interpolating polynomial.

        Transform Fourier interpolant harmonics to Nyquist trigonometric
        interpolant harmonics so that the coefficients are all real.

        Returns
        -------
        a_mn : jnp.ndarray
            Shape (L, μ, N) where μ ∈ {M, M+1}.
            Real valued spectral coefficients for Fourier-Chebyshev basis.

        """
        # ∂ₓ = 0 coefficients
        a0 = jnp.real(self._c[:, 0])[:, jnp.newaxis]
        # cos(mx) Tₙ(y) coefficients
        an = jnp.real(self._c[:, 1:]) * 2
        # sin(mx) Tₙ(y) coefficients
        bn = jnp.imag(self._c[:, 1:]) * (-2)
        # Note 2*(M//2)+1 <= M+1 and bM = 0 if equality.
        a_mn = jnp.hstack([a0, an, bn])
        assert a_mn.shape[-2] in (self.M, self.M + 1) and a_mn.shape[-1] == self.N
        return a_mn

    def _evaluate_fourier_basis(self, x):
        """Evaluate Fourier basis at points ``x`` and cache the coefficients.

        Parameters
        ----------
        x : jnp.ndarray
            Shape (L, x.shape[-1]) or (x.shape[-1], ).
            Evaluation points. If 1d assumes batch dimension over L is implicit.

        Returns
        -------
        a_n : jnp.ndarray
            Shape (L, N, x.shape[-1])

        """
        # TODO: do in desc.basis too for potentially significant performance boost.
        # Partial summation technique; see Boyd p. 185, eq. 10.2.
        x = jnp.atleast_2d(x)[:, jnp.newaxis]
        m = jnp.arange(1, self.M // 2 + 1)[:, jnp.newaxis]
        psi = jnp.dstack([jnp.ones(x.shape), jnp.cos(m * x), jnp.sin(m * x)])
        # batch matrix product (L, N, μ) @ (L, μ, x) = (L, N, x)
        self._a_n = jnp.swapaxes(self._harmonics(), -1, -2) @ psi
        assert self._a_n.shape == (self.L, self.N, x.shape[-1])
        return self._a_n

    def y_intersect(self, x, k=0, eps=_eps):
        """Coordinates yᵢ such that f(x, yᵢ) = k(x).

        Parameters
        ----------
        x : jnp.ndarray
            Shape (L, x.shape[-1]) or broadcastable of lower dimension.
            Evaluation points. If 1d assumes batch dimension over L is implicit
            (i.e. standard numpy broadcasting rules).
        k : jnp.ndarray
            Shape (P, L, x.shape[-1]) or broadcastable of lower dimension.
            Specify to find solutions yᵢ to f(x, yᵢ) = k(x). Default 0.
        eps : float
            Absolute tolerance with which to consider value as zero.

        Returns
        -------
        y : jnp.ndarray
            Shape (P, L, x.shape[-1], N - 1) or (L, x.shape[-1], N - 1).
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
        a_n = self._evaluate_fourier_basis(x)
        if k.ndim == 3:
            a_n = a_n[jnp.newaxis]
        a_n = put(a_n, Index[..., 0, :], a_n[..., 0, :] - k)
        a_n = jnp.swapaxes(a_n, -1, -2)  # shape is (P, L, x, N)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = _chebroots(a_n)
        assert y.shape[-3:] == (self.L, x.shape[-1], self.N - 1)

        # Pick sentinel such that only distinct roots are considered intersects.
        y = _filter_distinct(y, sentinel=-2, eps=eps)
        is_intersect = (jnp.abs(jnp.imag(y)) <= eps) & (jnp.abs(jnp.real(y)) <= 1)
        y = jnp.where(is_intersect, jnp.real(y), 0)  # ensure y is in domain of arcos
        #      ∂f/∂y =      ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n Uₙ₋₁(y)
        # sign ∂f/∂y = sign ∑ₙ₌₁ᴺ⁻¹ aₙ(x) sin(n arcos y)
        s = jnp.inner(
            a_n, jnp.sin(jnp.arange(self.N) * jnp.arccos(y)[..., jnp.newaxis])
        )
        is_decreasing = s <= 0
        is_increasing = s >= 0

        y = map_domain(y, self.domain[0], self.domain[-1])
        return y, is_decreasing, is_increasing, is_intersect

    def _isomorphism_1d(self, y):
        """Return coordinates z ∈ ℂ isomorphic to (x, y) ∈ ℂ².

        Maps row x of y to z = α(x) + y where α(x) = x * |domain|.

        Parameters
        ----------
        y : jnp.ndarray
            Shape (..., *y.shape[-2:]).
            Second to last axis iterates the rows.
            Leading axes are considered batch axes in usual numpy broadcasting.

        Returns
        -------
        z : jnp.ndarray
            Shape (..., y.shape[-2] * y.shape[-1]).
            Isomorphic coordinates.

        """
        alpha = (self.domain[-1] - self.domain[0]) * jnp.arange(y.shape[-2])
        z = _flatten_matrix(alpha[:, jnp.newaxis] + y)
        return z

    def _isomorphism_2d(self, z):
        """Return coordinates (x, y) ∈ ℂ² isomorphic to z ∈ ℂ.

        Returns index x and value y such that z = α(x) + y where α(x) = x * |domain|.

        Parameters
        ----------
        z : jnp.ndarray

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

    def bounce_points(self, y, is_decreasing, is_increasing, is_intersect):
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

        Returns
        -------
        bp1, bp2 : (jnp.ndarray, jnp.ndarray)
        Shape (*y.shape[:-2], y.shape[-1] * y.shape[-2]).
        The field line-following coordinates of bounce points for a given pitch along
        a field line. The pairs ``bp1`` and ``bp2`` form left and right integration
        boundaries, respectively, for the bounce integrals.

        If there were less than ``y.shape[-1] * y.shape[-2]`` bounce points detected
        along a field line, then the last axis, which enumerates the bounce points for
        a particular field line, is padded with zero.

        """
        # Last axis enumerates intersects of a pitch along a field line.
        y = self._isomorphism_1d(y)
        is_decreasing = _flatten_matrix(is_decreasing)
        is_increasing = _flatten_matrix(is_increasing)
        is_intersect = _flatten_matrix(is_intersect)
        is_bp1 = is_decreasing & is_intersect
        is_bp2 = is_increasing & _fix_inversion(is_intersect, is_increasing)

        sentinel = self.domain[0] - 1
        bp1 = _take_mask(y, is_bp1, fill_value=sentinel)
        bp2 = _take_mask(y, is_bp2, fill_value=sentinel)

        mask = (bp1 > sentinel) & (bp2 > sentinel)
        # Set outside mask to same value so integration is over set of measure zero.
        bp1 = jnp.where(mask, bp1, 0)
        bp2 = jnp.where(mask, bp2, 0)
        return bp1, bp2

    def _interp1d(
        self, z
    ):  # assumes z is on x points from a_n generated after evaluate fourier
        """Evaluate basis at coordinates z ∈ ℝ isomorphic to (x, y) ∈ ℝ².

        Parameters
        ----------
        z : jnp.ndarray
            Shape (P, L, B, Q).
            Isomorphic coordinates.
            Pitch, radial, bounce points, quad points.

        Returns
        -------
        f : jnp.ndarray
            Shape z.shape.
            This basis evaluated at z.

        """
        # Will have shape (P, L, BQ)
        x_index, y_values = map(_flatten_matrix, self._isomorphism_2d(z))
        y_values = map_domain_to_disc(y_values, self.domain[0], self.domain[1])
        a_n = jnp.swapaxes(self._a_n, -1, -2)  # changes to shape (L, x, N)
        n = jnp.arange(self.N)
        T = jnp.cos(n * jnp.arccos(y_values)[..., jnp.newaxis])
        # f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z])
        f = jnp.inner(a_n[x_index], T).reshape(z.shape)
        return f


def alpha_sequence(alpha_0, m, iota, period):
    """Get sequence of poloidal coordinates (α₀, α₁, …, αₘ₋₁) of field line.

    Parameters
    ----------
    alpha_0 : float
        Starting field line poloidal label.
    m : float
        Number of periods to follow field line.
    iota : jnp.ndarray
        Shape (L, )
        Rotational transform normalized by 2π.
    period : float
        Toroidal period after which to update label.

    Returns
    -------
    alpha : jnp.ndarray
        Shape (L, m)
        Sequence of poloidal coordinates (α₀, α₁, …, αₘ₋₁) that specify field line.

    """
    # Δz (∂α/∂ζ) = Δz ι̅ = Δz ι/2π = Δz data["iota"]
    return (alpha_0 + period * iota[:, jnp.newaxis] * jnp.arange(m)) % (2 * jnp.pi)


def _flatten_matrix(y):
    return y.reshape(*y.shape[:-2], -1)
