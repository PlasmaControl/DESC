"""Methods for computing bounce integrals (singular or otherwise)."""

from interpax import CubicHermiteSpline, PPoly
from orthax.legendre import leggauss

from desc.backend import jnp, rfft2
from desc.integrals.basis import FourierChebyshevBasis
from desc.integrals.bounce_utils import (
    _bounce_quadrature,
    _check_bounce_points,
    _set_default_plot_kwargs,
    bounce_points,
    get_alpha,
    get_pitch_inv,
    interp_to_argmin,
    plot_ppoly,
)
from desc.integrals.interp_utils import interp_rfft2, irfft2_non_uniform, polyder_vec
from desc.integrals.quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    get_quadrature,
    grad_automorphism_sin,
)
from desc.io import IOAble
from desc.utils import errorif, flatten_matrix, setdefault, warnif


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
            # Real fft over poloidal since usually num theta > num zeta.
            axes=(-1, -2),
        ).reshape(grid.num_rho, M, N),
        domain=Bounce2D.domain,
    )
    return T, B


def _transform_to_desc(grid, f):
    """Transform to DESC spectral domain.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (θ, ζ) with uniformly spaced nodes in
        (2π × 2π) poloidal and toroidal coordinates.
    f : jnp.ndarray
        Function evaluated on ``grid``.

    Returns
    -------
    a : jnp.ndarray
        Shape (grid.num_rho, grid.num_theta // 2 + 1, grid.num_zeta)
        Complex coefficients of 2D real FFT.

    """
    f = grid.meshgrid_reshape(f, order="rtz")
    a = rfft2(f, axes=(-1, -2), norm="forward")
    assert a.shape == (grid.num_rho, grid.num_theta // 2 + 1, grid.num_zeta)
    return a


# TODO:
#  After GitHub issue #1034 is resolved, we should pass in the previous
#  θ(α) coordinates as an initial guess for the next coordinate mapping.
#  Perhaps tell the optimizer to perturb the coefficients of the
#  |B|(α, ζ) directly? Maybe auto diff to see change on |B|(θ, ζ)
#  and hence stream functions. Not sure how feasible...

# TODO: Allow multiple starting labels for near-rational surfaces.
#  can just concatenate along second to last axis of cheb, but will
#  do in later pull request since it's not urgent.


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
    points with field-line-following coordinates. (In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as this operation reduces to a surface integral,
    which is invariant to the order of summation).

    The DESC coordinate system is related to field-line-following coordinate
    systems by a relation whose solution is best found with Newton iteration.
    There is a unique real solution to this equation, so Newton iteration is a
    globally convergent root-finding algorithm here. For the task of finding
    bounce points, even if the inverse map: θ(α, ζ) was known, Newton iteration
    is not a globally convergent algorithm to find the real roots of
    f : ζ ↦ |B|(ζ) − 1/λ where ζ is a field-line-following coordinate.
    For this, function approximation of |B| is necessary.

    Therefore, to compute bounce points {(ζ₁, ζ₂)}, we approximate |B| by a
    series expansion of basis functions in (α, ζ) coordinates, restricting the
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
    irrational frequencies, courtesy of the irrational rotational transform.
    Globally convergent root-finding schemes for that basis (at fixed α) are
    not known. The denominator of a close rational could be absorbed into the
    coordinate ϕ, but this balloons the frequency, and hence the degree of the
    series. Although, because Fourier series may converge faster than Chebyshev,
    an alternate strategy that should work is to interpolate |B| to a double
    Fourier series in (ϑ, ϕ), then apply bisection methods to find roots of f
    with mesh size inversely proportional to the max frequency along the field
    line: M ι + N. ``Bounce2D`` does not use that approach because that
    root-finding scheme is inferior.

    After obtaining the bounce points, the supplied quadrature is performed.
    By default, this is a Gauss quadrature after removing the singularity.
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
        fixed to ``L*M*N``, independent of the number of toroidal transits.

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
        num_transit=32,
        quad=leggauss(32),
        automorphism=(automorphism_sin, grad_automorphism_sin),
        Bref=1.0,
        Lref=1.0,
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
            ``FourierChebyshevBasis.nodes(M,N,L,domain=FourierBounce.domain)``.
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
        Bref : float
            Optional. Reference magnetic field strength for normalization.
        Lref : float
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
        self._x, self._w = get_quadrature(quad, automorphism)

        # Compute global splines.
        self._b_sup_z = _transform_to_desc(
            grid,
            jnp.abs(data["B^zeta"]) / data["|B|"] * Lref,
        )[:, jnp.newaxis]
        T, B = _transform_to_clebsch(
            grid,
            desc_from_clebsch,
            M,
            N,
            data["|B|"] / Bref,
        )
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
            ``FourierChebyshevBasis.nodes(M,N,L,domain=FourierBounce.domain)``.
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
        """Reshape ``data`` arrays for acceptable input to ``integrate``.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ).
        data : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : list[jnp.ndarray]
            List of reshaped arrays which may be given to ``integrate``.

        """
        f = [grid.meshgrid_reshape(d, "rtz")[:, jnp.newaxis] for d in data]
        return f

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
            bounds the actual number. As a reference, there are typically at most 5
            wells per toroidal transit for a given pitch.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : (jnp.ndarray, jnp.ndarray)
            Shape (P, L, num_well).
            ζ coordinates of bounce points. The points are grouped and ordered such
            that the straight line path between ``z1`` and ``z2`` resides in the
            epigraph of |B|.

        """
        return self._B.intersect1d(1 / jnp.atleast_2d(pitch), num_well)

    def check_bounce_points(self, z1, z2, pitch, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        z1, z2 : (jnp.ndarray, jnp.ndarray)
            Shape (P, L, num_well).
            ζ coordinates of bounce points. The points are grouped and ordered such
            that the straight line path between ``z1`` and ``z2`` resides in the
            epigraph of |B|.
        pitch : jnp.ndarray
            Shape (P, L).
            λ values to evaluate the bounce integral at each field line. λ(ρ) is
            specified by ``pitch[...,ρ]`` where in the latter the labels ρ are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        plot : bool
            Whether to plot stuff.
        kwargs : dict
            Keyword arguments into ``ChebyshevBasisSet.plot1d``.

        """
        kwargs.setdefault(
            "title",
            r"Intersects $\zeta$ in epigraph($\vert B \vert$) s.t. "
            r"$\vert B \vert(\zeta) = 1/\lambda$",
        )
        kwargs.setdefault("klabel", r"$1/\lambda$")
        kwargs.setdefault("hlabel", r"$\zeta$")
        kwargs.setdefault("vlabel", r"$\vert B \vert$")
        self._B.check_intersect1d(z1, z2, 1 / pitch, plot, **kwargs)

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
            bounds the actual number. As a reference, there are typically at most 5
            wells per toroidal transit for a given pitch.

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
        z1, z2 = self.bounce_points(pitch, num_well)
        result = self._integrate(z1, z2, pitch, integrand, f)
        errorif(weight is not None, NotImplementedError)
        return result

    def _integrate(self, z1, z2, pitch, integrand, f):
        assert z1.ndim == 3
        assert z1.shape == z2.shape
        assert pitch.ndim == 2
        W = z1.shape[-1]  # number of wells
        shape = (pitch.shape[0], self._L, W, self._x.size)

        # quadrature points parameterized by ζ for each pitch and flux surface
        Q_zeta = flatten_matrix(
            bijection_from_disc(self._x, z1[..., jnp.newaxis], z2[..., jnp.newaxis])
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
                xq=Q, a=self._b_sup_z, M=self._n, N=self._m, axes=(-1, -2)
            ).reshape(shape),
            self._w,
        )
        assert result.shape == (pitch.shape[0], self._L, W)
        return result


class Bounce1D(IOAble):
    """Computes bounce integrals using one-dimensional local spline methods.

    The bounce integral is defined as ∫ f(λ, ℓ) dℓ, where
        dℓ parameterizes the distance along the field line in meters,
        f(λ, ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points ℓ₁, ℓ₂ s.t. λ|B|(ℓᵢ) = 1,
        where λ is a constant defining the integral proportional to the magnetic moment
        over energy and |B| is the norm of the magnetic field.

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
    points with field-line-following coordinates. (In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as this operation reduces to a surface integral,
    which is invariant to the order of summation).

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
    This is useful if one can efficiently obtain data along field lines and
    most efficient if the number of toroidal transits to follow a field line is
    not too large.

    After computing the bounce points, the supplied quadrature is performed.
    By default, this is a Gauss quadrature after removing the singularity.
    Local splines interpolate functions in the integrand to the quadrature nodes.

    See Also
    --------
    Bounce2D : Uses two-dimensional pseudo-spectral techniques for the same task.

    Examples
    --------
    See ``tests/test_integrals.py::TestBounce1D::test_bounce1d_checks``.

    Attributes
    ----------
    required_names : list
        Names in ``data_index`` required to compute bounce integrals.
    B : jnp.ndarray
        Shape (M, L, N - 1, B.shape[-1]).
        Polynomial coefficients of the spline of |B| in local power basis.
        Last axis enumerates the coefficients of power series. For a polynomial
        given by ∑ᵢⁿ cᵢ xⁱ, coefficient cᵢ is stored at ``B[...,n-i]``.
        Third axis enumerates the polynomials that compose a particular spline.
        Second axis enumerates flux surfaces.
        First axis enumerates field lines of a particular flux surface.

    """

    required_names = ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]
    get_pitch_inv = staticmethod(get_pitch_inv)

    def __init__(
        self,
        grid,
        data,
        quad=leggauss(32),
        automorphism=(automorphism_sin, grad_automorphism_sin),
        Bref=1.0,
        Lref=1.0,
        *,
        is_reshaped=False,
        check=False,
        **kwargs,
    ):
        """Returns an object to compute bounce integrals.

        Parameters
        ----------
        grid : Grid
            Clebsch coordinate (ρ, α, ζ) tensor-product grid.
            The ζ coordinates (the unique values prior to taking the tensor-product)
            must be strictly increasing and preferably uniformly spaced. These are used
            as knots to construct splines. A reference knot density is 100 knots per
            toroidal transit. Note that below shape notation defines
            L = ``grid.num_rho``, M = ``grid.num_alpha``, and N = ``grid.num_zeta``.
        data : dict[str, jnp.ndarray]
            Data evaluated on ``grid``.
            Must include names in ``Bounce1D.required_names``.
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
        is_reshaped : bool
            Whether the arrays in ``data`` are already reshaped to the expected form of
            shape (..., N) or (..., L, N) or (M, L, N). This option can be used to
            iteratively compute bounce integrals one field line or one flux surface
            at a time, respectively, potentially reducing memory usage. To do so,
            set to true and provide only those axes of the reshaped data.
            Default is false.
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
        self._data = (
            data
            if is_reshaped
            else dict(zip(data.keys(), Bounce1D.reshape_data(grid, *data.values())))
        )
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
            source=(0, 1),
            destination=(-1, -2),
        )
        self._dB_dz = polyder_vec(self.B)

        # Add axis here instead of in ``_bounce_quadrature``.
        for name in self._data:
            self._data[name] = self._data[name][..., jnp.newaxis, :]

    @staticmethod
    def reshape_data(grid, *arys):
        """Reshape arrays for acceptable input to ``integrate``.

        Parameters
        ----------
        grid : Grid
            Clebsch coordinate (ρ, α, ζ) tensor-product grid.
        arys : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : jnp.ndarray
            Shape (M, L, N).
            Reshaped data which may be given to ``integrate``.

        """
        f = [grid.meshgrid_reshape(d, "arz") for d in arys]
        return f if len(f) > 1 else f[0]

    def points(self, pitch_inv, *, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (M, L, P).
            1/λ values to compute the bounce points at each field line. 1/λ(α,ρ) is
            specified by ``pitch_inv[α,ρ]`` where in the latter the labels
            are interpreted as the indices that correspond to that field line.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number. As a reference, there are typically 20 wells
            per toroidal transit for a given pitch. You can check this by plotting
            the field lines with the ``check_points`` method.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : (jnp.ndarray, jnp.ndarray)
            Shape (M, L, P, num_well).
            ζ coordinates of bounce points. The points are ordered and grouped such
            that the straight line path between ``z1`` and ``z2`` resides in the
            epigraph of |B|.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        return bounce_points(pitch_inv, self._zeta, self.B, self._dB_dz, num_well)

    def check_points(self, z1, z2, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        z1, z2 : (jnp.ndarray, jnp.ndarray)
            Shape (M, L, P, num_well).
            ζ coordinates of bounce points. The points are ordered and grouped such
            that the straight line path between ``z1`` and ``z2`` resides in the
            epigraph of |B|.
        pitch_inv : jnp.ndarray
            Shape (M, L, P).
            1/λ values to compute the bounce points at each field line. 1/λ(α,ρ) is
            specified by ``pitch_inv[α,ρ]`` where in the latter the labels
            are interpreted as the indices that correspond to that field line.
        plot : bool
            Whether to plot the field lines and bounce points of the given pitch angles.
        kwargs
            Keyword arguments into ``desc/integrals/bounce_utils.py::plot_ppoly``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        return _check_bounce_points(
            z1=z1,
            z2=z2,
            pitch_inv=pitch_inv,
            knots=self._zeta,
            B=self.B,
            plot=plot,
            **kwargs,
        )

    def integrate(
        self,
        integrand,
        pitch_inv,
        f=None,
        weight=None,
        *,
        num_well=None,
        method="cubic",
        batch=True,
        check=False,
        plot=False,
    ):
        """Bounce integrate ∫ f(λ, ℓ) dℓ.

        Computes the bounce integral ∫ f(λ, ℓ) dℓ for every field line and pitch.

        Parameters
        ----------
        integrand : callable
            The composition operator on the set of functions in ``f`` that maps the
            functions in ``f`` to the integrand f(λ, ℓ) in ∫ f(λ, ℓ) dℓ. It should
            accept the arrays in ``f`` as arguments as well as the additional keyword
            arguments: ``B`` and ``pitch``. A quadrature will be performed to
            approximate the bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
        pitch_inv : jnp.ndarray
            Shape (M, L, P).
            1/λ values to compute the bounce integrals. 1/λ(α,ρ) is specified by
            ``pitch_inv[α,ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        f : list[jnp.ndarray] or jnp.ndarray
            Shape (M, L, N).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. These functions should be arguments to the callable
            ``integrand``. Use the method ``self.reshape_data`` to reshape the data
            into the expected shape.
        weight : jnp.ndarray
            Shape (M, L, N).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(λ, ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in that magnetic well. Use the method
            ``self.reshape_data`` to reshape the data into the expected shape.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number. As a reference, there are typically 20 wells
            per toroidal transit for a given pitch. You can check this by plotting
            the field lines with the ``check_points`` method.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.
        method : str
            Method of interpolation.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
            Default is cubic C1 local spline.
        batch : bool
            Whether to perform computation in a batched manner. Default is true.
        check : bool
            Flag for debugging. Must be false for JAX transformations.
        plot : bool
            Whether to plot the quantities in the integrand interpolated to the
            quadrature points of each integral. Ignored if ``check`` is false.

        Returns
        -------
        result : jnp.ndarray
            Shape (M, L, P, num_well).
            Last axis enumerates the bounce integrals for a given field line,
            flux surface, and pitch value.

        """
        z1, z2 = self.points(pitch_inv, num_well=num_well)
        result = _bounce_quadrature(
            x=self._x,
            w=self._w,
            z1=z1,
            z2=z2,
            integrand=integrand,
            pitch_inv=pitch_inv,
            f=setdefault(f, []),
            data=self._data,
            knots=self._zeta,
            method=method,
            batch=batch,
            check=check,
            plot=plot,
        )
        if weight is not None:
            result *= interp_to_argmin(
                weight,
                z1,
                z2,
                self._zeta,
                self.B,
                self._dB_dz,
                method,
            )
        assert result.shape == z1.shape
        return result

    def plot(self, m, l, pitch_inv=None, /, **kwargs):
        """Plot the field line and bounce points of the given pitch angles.

        Parameters
        ----------
        m, l : int, int
            Indices into the nodes of the grid supplied to make this object.
            ``alpha,rho=grid.meshgrid_reshape(grid.nodes[:,:2],"arz")[m,l,0]``.
        pitch_inv : jnp.ndarray
            Shape (P, ).
            Optional, 1/λ values whose corresponding bounce points on the field line
            specified by Clebsch coordinate α(m), ρ(l) will be plotted.
        kwargs
            Keyword arguments into ``desc/integrals/bounce_utils.py::plot_ppoly``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        B, dB_dz = self.B, self._dB_dz
        if B.ndim == 4:
            B = B[m]
            dB_dz = dB_dz[m]
        if B.ndim == 3:
            B = B[l]
            dB_dz = dB_dz[l]
        if pitch_inv is not None:
            errorif(
                pitch_inv.ndim > 1,
                msg=f"Got pitch_inv.ndim={pitch_inv.ndim}, but expected 1.",
            )
            z1, z2 = bounce_points(pitch_inv, self._zeta, B, dB_dz)
            kwargs["z1"] = z1
            kwargs["z2"] = z2
            kwargs["k"] = pitch_inv
        fig, ax = plot_ppoly(PPoly(B.T, self._zeta), **_set_default_plot_kwargs(kwargs))
        return fig, ax
