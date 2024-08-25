"""Utilities for bounce integrals."""

from functools import partial

from interpax import PPoly
from matplotlib import pyplot as plt
from orthax.chebyshev import chebroots

from desc.backend import flatnonzero, imap, jnp, put, softmax
from desc.integrals.interp_utils import (
    interp1d_vec,
    interp1d_vec_with_df,
    poly_root,
    polyval_vec,
)
from desc.integrals.quad_utils import (
    bijection_from_disc,
    composite_linspace,
    grad_bijection_from_disc,
)
from desc.utils import atleast_3d_mid, errorif, setdefault, take_mask

# TODO: Boyd's method 𝒪(N²) instead of Chebyshev companion matrix 𝒪(N³).
#  John P. Boyd, Computing real roots of a polynomial in Chebyshev series
#  form through subdivision. https://doi.org/10.1016/j.apnum.2005.09.007.
chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


def flatten_matrix(y):
    """Flatten matrix to vector."""
    return y.reshape(*y.shape[:-2], -1)


def subtract(c, k):
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


def get_pitch(min_B, max_B, num, relative_shift=1e-6):
    """Return uniformly spaced values between ``1/max_B`` and ``1/min_B``.

    Parameters
    ----------
    min_B : jnp.ndarray
        Minimum |B| value.
    max_B : jnp.ndarray
        Maximum |B| value.
    num : int
        Number of values, not including endpoints.
    relative_shift : float
        Relative amount to shift maxima down and minima up to avoid floating point
        errors in downstream routines.

    Returns
    -------
    pitch : jnp.ndarray
        Shape (num + 2, *min_B.shape).

    """
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + relative_shift) * min_B
    max_B = (1 - relative_shift) * max_B
    pitch = composite_linspace(1 / jnp.stack([max_B, min_B]), num)
    assert pitch.shape == (num + 2, *min_B.shape)
    return pitch


# TODO: Generalize this beyond ζ = ϕ or just map to Clebsch with ϕ.
def get_alpha(alpha_0, iota, num_transit, period):
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
    alpha : jnp.ndarray
        Shape (iota.size, num_transit).
        Sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) that specify field line.

    """
    # Δϕ (∂α/∂ϕ) = Δϕ ι̅ = Δϕ ι/2π = Δϕ data["iota"]
    alpha = alpha_0 + period * iota[:, jnp.newaxis] * jnp.arange(num_transit)
    return alpha


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def epigraph_and(is_intersect, df_dy_sign):
    """Set and  epigraph of f with ``is_intersect``.

    Remove intersects for which there does not exist a connected path between
    adjacent intersects in the epigraph of a continuous map ``f``.

    Parameters
    ----------
    is_intersect : jnp.ndarray
        Boolean array indicating whether element is an intersect.
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
        # degenerate pitch value - one that does not ride the global extrema of |B|.
    )
    return put(is_intersect, idx[0], edge_case)


def _check_spline_shape(knots, B, dB_dz, pitch=None):
    """Ensure inputs have compatible shape, and return them with full dimension.

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line-following ζ coordinates of spline knots. Must be strictly increasing.
    B : jnp.ndarray
        Shape (B.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    dB_dz : jnp.ndarray
        Shape (B.shape[0] - 1, *B.shape[1:]).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|ρ,α in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    pitch : jnp.ndarray
        Shape must broadcast with (P, S).
        λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
        specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
        interpreted as the index into the last axis that corresponds to that field
        line. If two-dimensional, the first axis is the batch axis.

    """
    errorif(knots.ndim != 1, msg=f"knots should be 1d; got shape {knots.shape}.")
    errorif(
        B.shape[-1] != (knots.size - 1),
        msg=(
            "Last axis does not enumerate polynomials of spline. "
            f"B.shape={B.shape}. knots.shape={knots.shape}."
        ),
    )
    errorif(
        B.ndim > 3
        or dB_dz.ndim > 3
        or (B.shape[0] - 1) != dB_dz.shape[0]
        or B.shape[1:] != dB_dz.shape[1:],
        msg=f"Invalid shape for spline. B.shape={B.shape}. dB_dz.shape={dB_dz.shape}.",
    )
    # Add axis which enumerates field lines if necessary.
    B, dB_dz = atleast_3d_mid(B, dB_dz)
    if pitch is not None:
        pitch = jnp.atleast_2d(pitch)
        errorif(
            pitch.ndim != 2
            or not (pitch.shape[-1] == 1 or pitch.shape[-1] == B.shape[1]),
            msg=f"Invalid shape {pitch.shape} for pitch angles.",
        )
    return B, dB_dz, pitch


def bounce_points(
    pitch, knots, B, dB_dz, num_well=None, check=False, plot=True, **kwargs
):
    """Compute the bounce points given spline of |B| and pitch λ.

    Parameters
    ----------
    pitch : jnp.ndarray
        Shape must broadcast with (P, S).
        λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
        specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
        interpreted as the index into the last axis that corresponds to that field
        line. If two-dimensional, the first axis is the batch axis.
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line-following ζ coordinates of spline knots. Must be strictly increasing.
    B : jnp.ndarray
        Shape (B.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    dB_dz : jnp.ndarray
        Shape (B.shape[0] - 1, *B.shape[1:]).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|ρ,α in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    num_well : int or None
        Specify to return the first ``num_well`` pairs of bounce points for each
        pitch along each field line. This is useful if ``num_well`` tightly
        bounds the actual number. As a reference, there are typically at most 5
        wells per toroidal transit for a given pitch.

        If not specified, then all bounce points are returned. If there were fewer
        wells detected along a field line than the size of the last axis of the
        returned arrays, then that axis is padded with zero.
    check : bool
        Flag for debugging. Must be false for JAX transformations.
    plot : bool
        Whether to plot some things if check is true. Default is true.
    kwargs : dict
        Keyword arguments into ``plot_ppoly``.

    Returns
    -------
    bp1, bp2 : (jnp.ndarray, jnp.ndarray)
        Shape (P, S, num_well).
        The field line-following coordinates of bounce points.
        The pairs ``bp1`` and ``bp2`` form left and right integration boundaries,
        respectively, for the bounce integrals.

        If there were less than ``num_wells`` wells detected along a field line,
        then the last axis, which enumerates bounce points for  a particular field
        line and pitch, is padded with zero.

    """
    B, dB_dz, pitch = _check_spline_shape(knots, B, dB_dz, pitch)
    P, S, degree = pitch.shape[0], B.shape[1], B.shape[0] - 1
    # Intersection points in local power basis.
    intersect = poly_root(
        c=B,
        k=(1 / pitch)[..., jnp.newaxis],
        a_min=jnp.array([0.0]),
        a_max=jnp.diff(knots),
        sort=True,
        sentinel=-1.0,
        distinct=True,
    )
    assert intersect.shape == (P, S, knots.size - 1, degree)

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    dB_dz_sign = jnp.sign(
        polyval_vec(x=intersect, c=dB_dz[..., jnp.newaxis]).reshape(P, S, -1)
    )
    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = intersect.reshape(P, S, -1) >= 0
    # Following discussion on page 3 and 5 of https://doi.org/10.1063/1.873749,
    # we ignore the bounce points of particles only assigned to a class that are
    # trapped outside this snapshot of the field line.
    is_bp1 = (dB_dz_sign <= 0) & is_intersect
    is_bp2 = (dB_dz_sign >= 0) & epigraph_and(is_intersect, dB_dz_sign)

    # Transform out of local power basis expansion.
    intersect = (intersect + knots[:-1, jnp.newaxis]).reshape(P, S, -1)
    # New versions of JAX only like static sentinels.
    sentinel = -10000000.0  # instead of knots[0] - 1
    bp1 = take_mask(intersect, is_bp1, size=num_well, fill_value=sentinel)
    bp2 = take_mask(intersect, is_bp2, size=num_well, fill_value=sentinel)

    mask = (bp1 > sentinel) & (bp2 > sentinel)
    # Set outside mask to same value so integration is over set of measure zero.
    bp1 = jnp.where(mask, bp1, 0.0)
    bp2 = jnp.where(mask, bp2, 0.0)

    if check:
        _check_bounce_points(bp1, bp2, pitch, knots, B, plot, **kwargs)

    return bp1, bp2


def _check_bounce_points(bp1, bp2, pitch, knots, B, plot=True, **kwargs):
    """Check that bounce points are computed correctly."""
    eps = jnp.finfo(jnp.array(1.0).dtype).eps * 10
    title = kwargs.pop(
        "title",
        r"Intersects $\zeta$ in epigraph of $\vert B \vert(\zeta) = 1/\lambda$",
    )
    klabel = kwargs.pop("klabel", r"$1/\lambda$")
    hlabel = kwargs.pop("hlabel", r"$\zeta$")
    vlabel = kwargs.pop("vlabel", r"$\vert B \vert(\zeta)$")

    assert bp1.shape == bp2.shape
    mask = (bp1 - bp2) != 0.0
    bp1 = jnp.where(mask, bp1, jnp.nan)
    bp2 = jnp.where(mask, bp2, jnp.nan)

    err_1 = jnp.any(bp1 > bp2, axis=-1)
    err_2 = jnp.any(bp1[..., 1:] < bp2[..., :-1], axis=-1)

    P, S, _ = bp1.shape
    for s in range(S):
        Bs = PPoly(B[:, s], knots)
        for p in range(P):
            Bs_midpoint = Bs((bp1[p, s] + bp2[p, s]) / 2)
            err_3 = jnp.any(Bs_midpoint > 1 / pitch[p, s] + eps)
            if not (err_1[p, s] or err_2[p, s] or err_3):
                continue
            _bp1 = bp1[p, s][mask[p, s]]
            _bp2 = bp2[p, s][mask[p, s]]
            if plot:
                plot_ppoly(
                    ppoly=Bs,
                    z1=_bp1,
                    z2=_bp2,
                    k=1 / pitch[p, s],
                    klabel=klabel,
                    title=title,
                    hlabel=hlabel,
                    vlabel=vlabel,
                    **kwargs,
                )
            print("      bp1    |    bp2")
            print(jnp.column_stack([_bp1, _bp2]))
            assert not err_1[p, s], "Intersects have an inversion.\n"
            assert not err_2[p, s], "Detected discontinuity.\n"
            assert not err_3, (
                f"Detected |B| = {Bs_midpoint[mask[p, s]]} > {1 / pitch[p, s] + eps} "
                f"= 1/λ in well. Use more knots.\n"
            )
        if plot:
            plot_ppoly(
                ppoly=Bs,
                z1=bp1[:, s],
                z2=bp2[:, s],
                k=1 / pitch[:, s],
                klabel=klabel,
                title=title,
                hlabel=hlabel,
                vlabel=vlabel,
                **kwargs,
            )


def bounce_quadrature(
    x,
    w,
    bp1,
    bp2,
    pitch,
    integrand,
    f,
    data,
    knots,
    method="cubic",
    batch=True,
    check=False,
    plot=False,
):
    """Bounce integrate ∫ f(ℓ) dℓ.

    Parameters
    ----------
    x : jnp.ndarray
        Shape (w.size, ).
        Quadrature points in [-1, 1].
    w : jnp.ndarray
        Shape (w.size, ).
        Quadrature weights.
    bp1, bp2 : jnp.ndarray
        Shape (P, S, num_well).
        The field line-following coordinates of bounce points.
        The pairs ``bp1`` and ``bp2`` form left and right integration boundaries,
        respectively, for the bounce integrals.
    pitch : jnp.ndarray
        Shape must broadcast with (P, S).
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
        Shape (S, knots.size).
        Real scalar-valued functions evaluated on the ``knots``.
        These functions should be arguments to the callable ``integrand``.
    data : dict[str, jnp.ndarray]
        Data evaluated on ``grid`` and reshaped with ``Bounce1D.reshape_data``.
        Must include names in ``Bounce1D.required_names()``.
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line-following sorted, unique ζ coordinates where the arrays in
        ``data`` and ``f`` were evaluated.
    method : str
        Method of interpolation for functions contained in ``f``.
        See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
        Default is cubic C1 local spline.
    batch : bool
        Whether to perform computation in a batched manner. Default is true.
    check : bool
        Flag for debugging. Must be false for JAX transformations.
        Ignored if ``batch`` is false.
    plot : bool
        Whether to plot stuff if ``check`` is true. Default is false.
        Only developers doing debugging want to see these plots.

    Returns
    -------
    result : jnp.ndarray
        Shape (P, S, num_well).
        Quadrature for every pitch along every field line.
        First axis enumerates pitch values. Second axis enumerates the field lines.
        Last axis enumerates the bounce integrals.

    """
    errorif(bp1.ndim != 3 or bp1.shape != bp2.shape)
    errorif(x.ndim != 1 or x.shape != w.shape)
    pitch = jnp.atleast_2d(pitch)
    if not isinstance(f, (list, tuple)):
        f = [f]

    # Integrate and complete the change of variable.
    if batch:
        result = _interpolate_and_integrate(
            w=w,
            Q=bijection_from_disc(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis]),
            pitch=pitch,
            integrand=integrand,
            f=f,
            data=data,
            knots=knots,
            method=method,
            check=check,
            plot=plot,
        )
    else:
        f = list(f)

        # TODO: Use batched vmap.
        def loop(bp):
            bp1, bp2 = bp
            # Need to return tuple because input was tuple; artifact of JAX map.
            return None, _interpolate_and_integrate(
                w=w,
                Q=bijection_from_disc(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis]),
                pitch=pitch,
                integrand=integrand,
                f=f,
                data=data,
                knots=knots,
                method=method,
                check=False,
                plot=False,
            )

        result = jnp.moveaxis(
            imap(loop, (jnp.moveaxis(bp1, -1, 0), jnp.moveaxis(bp2, -1, 0)))[1],
            source=0,
            destination=-1,
        )

    result = result * grad_bijection_from_disc(bp1, bp2)
    assert result.shape == (pitch.shape[0], data["|B|"].shape[0], bp1.shape[-1])
    return result


def _interpolate_and_integrate(
    w,
    Q,
    pitch,
    integrand,
    f,
    data,
    knots,
    method,
    check=False,
    plot=False,
):
    """Interpolate given functions to points ``Q`` and perform quadrature.

    Parameters
    ----------
    w : jnp.ndarray
        Shape (w.size, ).
        Quadrature weights.
    Q : jnp.ndarray
        Shape (P, S, Q.shape[2], w.size).
        Quadrature points at field line-following ζ coordinates.
    data : dict[str, jnp.ndarray]
        Data evaluated on ``grid`` and reshaped with ``Bounce1D.reshape_data``.
        Must include names in ``Bounce1D.required_names()``.

    Returns
    -------
    result : jnp.ndarray
        Shape Q.shape[:-1].
        Quadrature for every pitch along every field line.

    """
    assert pitch.ndim == 2
    assert w.ndim == knots.ndim == 1
    assert 3 <= Q.ndim <= 4 and Q.shape[:2] == (pitch.shape[0], data["|B|"].shape[0])
    assert Q.shape[-1] == w.size
    assert knots.size == data["|B|"].shape[-1]
    assert (
        data["B^zeta"].shape
        == data["B^zeta_z|r,a"].shape
        == data["|B|"].shape
        == data["|B|_z|r,a"].shape
    )

    pitch = jnp.expand_dims(pitch, axis=(2, 3) if (Q.ndim == 4) else 2)
    shape = Q.shape
    Q = Q.reshape(Q.shape[0], Q.shape[1], -1)
    b_sup_z = interp1d_vec_with_df(
        Q,
        knots,
        data["B^zeta"] / data["|B|"],
        data["B^zeta_z|r,a"] / data["|B|"]
        - data["B^zeta"] * data["|B|_z|r,a"] / data["|B|"] ** 2,
    ).reshape(shape)
    B = interp1d_vec_with_df(Q, knots, data["|B|"], data["|B|_z|r,a"]).reshape(shape)
    # Spline the integrand so that we can evaluate it at quadrature points without
    # expensive coordinate mappings and root finding. Spline each function separately so
    # that the singularity near the bounce points can be captured more accurately than
    # can be by any polynomial.
    f = [interp1d_vec(Q, knots, f_i, method=method).reshape(shape) for f_i in f]
    result = jnp.dot(integrand(*f, B=B, pitch=pitch) / b_sup_z, w)

    if check:
        _check_interp(Q.reshape(shape), f, b_sup_z, B, data["|B|_z|r,a"], result, plot)

    return result


def _check_interp(Z, f, b_sup_z, B, B_z_ra, result, plot):
    """Check for floating point errors.

    Parameters
    ----------
    Z : jnp.ndarray
        Quadrature points at field line-following ζ coordinates.
    f : list of jnp.ndarray
        Arguments to the integrand interpolated to Z.
    b_sup_z : jnp.ndarray
        Contravariant field-line following toroidal component of magnetic field,
        interpolated to Z.
    B : jnp.ndarray
        Norm of magnetic field, interpolated to Z.
    B_z_ra : jnp.ndarray
        Norm of magnetic field, derivative with respect to field-line following
        coordinate.
    result : jnp.ndarray
        Output of ``_interpolate_and_integrate``.
    plot : bool
        Whether to plot stuff.

    """
    assert jnp.isfinite(Z).all(), "NaN interpolation point."
    # Integrals that we should be computing.
    marked = jnp.any(Z != 0, axis=-1)
    goal = jnp.sum(marked)

    msg = "Interpolation failed."
    assert jnp.isfinite(B_z_ra).all(), msg
    assert goal == jnp.sum(marked & jnp.isfinite(jnp.sum(b_sup_z, axis=-1))), msg
    assert goal == jnp.sum(marked & jnp.isfinite(jnp.sum(B, axis=-1))), msg
    for f_i in f:
        assert goal == jnp.sum(marked & jnp.isfinite(jnp.sum(f_i, axis=-1))), msg

    msg = "|B| has vanished, violating the hairy ball theorem."
    assert not jnp.isclose(B, 0).any(), msg
    assert not jnp.isclose(b_sup_z, 0).any(), msg

    # Number of those integrals that were computed.
    actual = jnp.sum(marked & jnp.isfinite(result))
    assert goal == actual, (
        f"Lost {goal - actual} integrals from NaN generation in the integrand. This "
        "can be caused by floating point error or a poor choice of quadrature nodes."
    )
    if plot:
        _plot_check_interp(Z, B, name=r"$\vert B \vert$")
        _plot_check_interp(Z, b_sup_z, name=r"$ (B / \vert B \vert) \cdot e^{\zeta}$")


def _plot_check_interp(Z, V, name=""):
    """Plot V[λ, (ρ, α), (ζ₁, ζ₂)](Z)."""
    for p in range(Z.shape[0]):
        for s in range(Z.shape[1]):
            marked = jnp.nonzero(jnp.any(Z != 0, axis=-1))[0]
            if marked.size == 0:
                continue
            fig, ax = plt.subplots()
            ax.set_xlabel(r"Field line $\zeta$")
            ax.set_ylabel(name)
            ax.set_title(
                f"Interpolation of {name} to quadrature points. Index {p},{s}."
            )
            for i in marked:
                ax.plot(Z[p, s, i], V[p, s, i], marker="o")
            fig.text(
                0.01,
                0.01,
                f"Each color specifies the set of points and values (ζ, {name}(ζ)) "
                "used to evaluate an integral.",
            )
            plt.tight_layout()
            plt.show()


def _get_extrema(knots, B, dB_dz, sentinel=jnp.nan):
    """Return extrema (ζ*, |B|(ζ*)) along field line.

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line-following ζ coordinates of spline knots. Must be strictly increasing.
    B : jnp.ndarray
        Shape (B.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    dB_dz : jnp.ndarray
        Shape (B.shape[0] - 1, *B.shape[1:]).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|ρ,α in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    sentinel : float
        Value with which to pad array to return fixed shape.

    Returns
    -------
    extrema, B_extrema : jnp.ndarray
        Shape (S, (knots.size - 1) * (degree - 1)).
        First array enumerates ζ*. Second array enumerates |B|(ζ*)
        Sorted order of ζ* is not promised.

    """
    B, dB_dz, _ = _check_spline_shape(knots, B, dB_dz)
    S, degree = B.shape[1], B.shape[0] - 1
    extrema = poly_root(
        c=dB_dz, a_min=jnp.array([0.0]), a_max=jnp.diff(knots), sentinel=sentinel
    )
    assert extrema.shape == (S, knots.size - 1, degree - 1)
    B_extrema = polyval_vec(x=extrema, c=B[..., jnp.newaxis]).reshape(S, -1)
    # Transform out of local power basis expansion.
    extrema = (extrema + knots[:-1, jnp.newaxis]).reshape(S, -1)
    return extrema, B_extrema


def interp_to_argmin_B_soft(g, bp1, bp2, knots, B, dB_dz, method="cubic", beta=-50):
    """Interpolate ``g`` to the deepest point in the magnetic well.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A = argmin_E |B|(ζ). Returns mean_A g(ζ).

    Parameters
    ----------
    g : jnp.ndarray
        Shape must broadcast with (S, knots.size).
        Values evaluated on ``knots`` to interpolate.
    beta : float
        More negative gives exponentially better approximation at the
        expense of noisier gradients - noisier in the physics sense (unrelated
        to the automatic differentiation).

    """
    ext, B = _get_extrema(knots, B, dB_dz, sentinel=0)
    assert ext.shape[0] == B.shape[0] == bp1.shape[1] == bp2.shape[1]
    argmin = softmax(
        beta
        * jnp.where(
            (bp1[..., jnp.newaxis] < ext[:, jnp.newaxis])
            & (ext[:, jnp.newaxis] < bp2[..., jnp.newaxis]),
            jnp.expand_dims(B / jnp.mean(B, axis=-1, keepdims=True), axis=1),
            1e2,  # >> max(|B|) / mean(|B|)
        ),
        axis=-1,
    )
    g = jnp.linalg.vecdot(
        argmin,
        interp1d_vec(ext, knots, jnp.atleast_2d(g), method=method)[:, jnp.newaxis],
    )
    assert g.shape == bp1.shape == bp2.shape
    return g


# Less efficient than soft if P >> 1.
def interp_to_argmin_B_hard(g, bp1, bp2, knots, B, dB_dz, method="cubic"):
    """Interpolate ``g`` to the deepest point in the magnetic well.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E |B|(ζ). Returns g(A).

    Parameters
    ----------
    g : jnp.ndarray
        Shape must broadcast with (S, knots.size).
        Values evaluated on ``knots`` to interpolate.

    """
    ext, B = _get_extrema(knots, B, dB_dz, sentinel=0)
    assert ext.shape[0] == B.shape[0] == bp1.shape[1] == bp2.shape[1]
    argmin = jnp.argmin(
        jnp.where(
            (bp1[..., jnp.newaxis] < ext[:, jnp.newaxis])
            & (ext[:, jnp.newaxis] < bp2[..., jnp.newaxis]),
            B[:, jnp.newaxis],
            1e2 + jnp.max(B),
        ),
        axis=-1,
    )
    A = jnp.take_along_axis(ext[jnp.newaxis], argmin, axis=-1)
    g = interp1d_vec(A, knots, jnp.atleast_2d(g), method=method)
    assert g.shape == bp1.shape == bp2.shape
    return g


def plot_ppoly(
    ppoly,
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
    start=None,
    stop=None,
    include_knots=False,
    knot_transparency=0.1,
):
    """Plot the piecewise polynomial ``ppoly``.

    Parameters
    ----------
    ppoly : PPoly
        Piecewise polynomial f.
    num : int
        Number of points to evaluate for plot.
    z1 : jnp.ndarray
        Shape (k.shape[0], W).
        Optional, intersects with ∂f/∂ζ <= 0.
    z2 : jnp.ndarray
        Shape (k.shape[0], W).
        Optional, intersects with ∂f/∂ζ >= 0.
    k : jnp.ndarray
        Shape (k.shape[0], ).
        Optional, k such that f(ζ) = k.
    k_transparency : float
        Transparency of intersect lines.
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
    start : float
        Minimum ζ on plot.
    stop : float
        Maximum ζ on plot.
    include_knots : bool
        Whether to plot vertical lines at the knots.
    knot_transparency : float
        Transparency of knot lines.

    Returns
    -------
    fig, ax : matplotlib figure and axes

    """
    fig, ax = plt.subplots()
    legend = {}
    if include_knots:
        for knot in ppoly.x:
            _add2legend(
                legend,
                ax.axvline(
                    x=knot, color="tab:blue", alpha=knot_transparency, label="knot"
                ),
            )

    z = jnp.linspace(
        start=setdefault(start, ppoly.x[0]),
        stop=setdefault(stop, ppoly.x[-1]),
        num=num,
    )
    _add2legend(legend, ax.plot(z, ppoly(z), label=vlabel))
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
        ax.scatter(_z1, jnp.full_like(_z1, k[i]), marker="v", color="tab:red")
        ax.scatter(_z2, jnp.full_like(_z2, k[i]), marker="^", color="tab:green")
