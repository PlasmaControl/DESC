"""Methods for computing bounce integrals (singular or otherwise)."""

from interpax import CubicHermiteSpline
from orthax.legendre import leggauss

from desc.backend import jnp
from desc.integrals.bounce_utils import (
    _check_bounce_points,
    bounce_points,
    bounce_quadrature,
    get_pitch,
    interp_to_argmin,
    plot_ppoly,
)
from desc.integrals.interp_utils import polyder_vec
from desc.integrals.quad_utils import (
    automorphism_sin,
    get_quadrature,
    grad_automorphism_sin,
)
from desc.utils import setdefault, warnif


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
    This is useful if one can efficiently obtain data along field lines.

    After obtaining the bounce points, the supplied quadrature is performed.
    By default, this is a Gauss quadrature after removing the singularity.
    Local splines interpolate functions in the integrand to the quadrature nodes.

    See Also
    --------
    Bounce2D : Uses two-dimensional pseudo-spectral techniques for the same task.

    Warnings
    --------
    The supplied data must be from a Clebsch coordinate (ρ, α, ζ) tensor-product grid.
    The ζ coordinates (the unique values prior to taking the tensor-product) must be
    strictly increasing and preferably uniformly spaced. These are used as knots to
    construct splines; a reference knot density is 100 knots per toroidal transit.

    Also note the argument ``pitch`` in the below method is defined as
    1/λ ~ E/μ = energy / magnetic moment.

    Examples
    --------
    See ``tests/test_integrals.py::TestBounce1D::test_integrate_checks``.

    Attributes
    ----------
    _B : jnp.ndarray
        TODO: Make this (4, M, L, N-1) now that tensor product in rho and alpha
          required as well after GitHub PR #1214.
        Shape (4, L * M, N - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.

    """

    plot_ppoly = staticmethod(plot_ppoly)
    get_pitch = staticmethod(get_pitch)

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
        self._B = jnp.moveaxis(
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
        self._dB_dz = polyder_vec(self._B)
        degree = 3
        assert self._B.shape[0] == degree + 1
        assert self._dB_dz.shape[0] == degree
        assert self._B.shape[-1] == self._dB_dz.shape[-1] == grid.num_zeta - 1

    @staticmethod
    def required_names():
        """Return names in ``data_index`` required to compute bounce integrals."""
        return ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]

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
        f : list[jnp.ndarray]
            List of reshaped data which may be given to ``integrate``.

        """
        f = [grid.meshgrid_reshape(d, "raz").reshape(-1, grid.num_zeta) for d in arys]
        return f

    def points(self, pitch, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch : jnp.ndarray
            Shape must broadcast with (P, L * M).
            1/λ values to evaluate the bounce integral at each field line. 1/λ(ρ,α) is
            specified by ``pitch[...,ρ]`` where in the latter the labels (ρ,α) are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
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
            Shape (P, L * M, num_well).
            ζ coordinates of bounce points. The points are ordered and grouped such
            that the straight line path between ``z1`` and ``z2`` resides in the
            epigraph of |B|.

            If there were less than ``num_wells`` wells detected along a field line,
            then the last axis, which enumerates bounce points for  a particular field
            line and pitch, is padded with zero.

        """
        return bounce_points(pitch, self._zeta, self._B, self._dB_dz, num_well)

    def check_points(self, z1, z2, pitch, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        z1, z2 : (jnp.ndarray, jnp.ndarray)
            Shape (P, L * M, num_well).
            ζ coordinates of bounce points. The points are ordered and grouped such
            that the straight line path between ``z1`` and ``z2`` resides in the
            epigraph of |B|.
        pitch : jnp.ndarray
            Shape must broadcast with (P, L * M).
            1/λ values to evaluate the bounce integral at each field line. 1/λ(ρ,α) is
            specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        plot : bool
            Whether to plot stuff.
        kwargs
            Keyword arguments into ``self.plot_ppoly``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        return _check_bounce_points(
            z1=z1,
            z2=z2,
            pitch=jnp.atleast_2d(pitch),
            knots=self._zeta,
            B=self._B,
            plot=plot,
            **kwargs,
        )

    def integrate(
        self,
        pitch,
        integrand,
        f=None,
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
            1/λ values to evaluate the bounce integral at each field line. 1/λ(ρ,α) is
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
            Shape must broadcast with (L * M, N).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in the magnetic well. Use the method
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

        Returns
        -------
        result : jnp.ndarray
            Shape (P, L*M, num_well).
            First axis enumerates pitch values. Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        pitch = jnp.atleast_2d(pitch)
        z1, z2 = self.points(pitch, num_well)
        result = bounce_quadrature(
            x=self._x,
            w=self._w,
            z1=z1,
            z2=z2,
            pitch=pitch,
            integrand=integrand,
            f=setdefault(f, []),
            data=self._data,
            knots=self._zeta,
            method=method,
            batch=batch,
            check=check,
        )
        if weight is not None:
            result *= interp_to_argmin(
                weight,
                z1,
                z2,
                self._zeta,
                self._B,
                self._dB_dz,
                method,
            )
        assert result.shape[-1] == setdefault(num_well, (self._zeta.size - 1) * 3)
        return result
