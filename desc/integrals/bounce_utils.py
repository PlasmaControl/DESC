"""Utilities and functional programming interface for bounce integrals."""

from interpax import PPoly
from matplotlib import pyplot as plt

from desc.backend import imap, jnp
from desc.backend import softmax as softargmax
from desc.integrals.basis import _add2legend, _in_epigraph_and, _plot_intersect
from desc.integrals.interp_utils import (
    interp1d_Hermite_vec,
    interp1d_vec,
    poly_root,
    polyval_vec,
)
from desc.integrals.quad_utils import (
    bijection_from_disc,
    composite_linspace,
    grad_bijection_from_disc,
)
from desc.utils import atleast_3d_mid, errorif, setdefault, take_mask


def get_pitch(min_B, max_B, num, relative_shift=1e-6):
    """Return 1/λ values uniformly spaced between ``min_B`` and ``max_B``.

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
        1/λ values. Note ``pitch`` = 1/λ ~ E/μ = energy / magnetic moment.

    """
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + relative_shift) * min_B
    max_B = (1 - relative_shift) * max_B
    # Samples should be uniformly spaced in |B| and not λ (GitHub issue #1228).
    pitch = composite_linspace(jnp.stack([min_B, max_B]), num)
    assert pitch.shape == (num + 2, *min_B.shape)
    return pitch


def _check_spline_shape(knots, g, dg_dz, pitch=None):
    """Ensure inputs have compatible shape, and return them with full dimension.

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (knots.size, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (g.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of g in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (g.shape[0] - 1, *g.shape[1:]).
        Polynomial coefficients of the spline of ∂g/∂ζ in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    pitch : jnp.ndarray
        Shape must broadcast with (P, S).
        1/λ values to evaluate the bounce integral at each field line. 1/λ(ρ,α) is
        specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
        interpreted as the index into the last axis that corresponds to that field
        line. If two-dimensional, the first axis is the batch axis.

    """
    errorif(knots.ndim != 1, msg=f"knots should be 1d; got shape {knots.shape}.")
    errorif(
        g.shape[-1] != (knots.size - 1),
        msg=(
            "Last axis does not enumerate polynomials of spline. "
            f"Spline shape {g.shape}. Knots shape {knots.shape}."
        ),
    )
    errorif(
        g.ndim > 3
        or dg_dz.ndim > 3
        or (g.shape[0] - 1) != dg_dz.shape[0]
        or g.shape[1:] != dg_dz.shape[1:],
        msg=f"Invalid shape {g.shape} for spline and derivative {dg_dz.shape}.",
    )
    # Add axis which enumerates field lines if necessary.
    g, dg_dz = atleast_3d_mid(g, dg_dz)
    if pitch is not None:
        pitch = jnp.atleast_2d(pitch)
        errorif(
            pitch.ndim != 2
            or not (pitch.shape[-1] == 1 or pitch.shape[-1] == g.shape[1]),
            msg=f"Invalid shape {pitch.shape} for pitch angles.",
        )
    return g, dg_dz, pitch


def bounce_points(
    pitch, knots, B, dB_dz, num_well=None, check=False, plot=True, **kwargs
):
    """Compute the bounce points given spline of |B| and pitch λ.

    Parameters
    ----------
    pitch : jnp.ndarray
        Shape must broadcast with (P, S).
        1/λ values to evaluate the bounce integral at each field line. 1/λ(ρ,α) is
        specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
        interpreted as the index into the last axis that corresponds to that field
        line. If two-dimensional, the first axis is the batch axis.
    knots : jnp.ndarray
        Shape (knots.size, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    B : jnp.ndarray
        Shape (B.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    dB_dz : jnp.ndarray
        Shape (B.shape[0] - 1, *B.shape[1:]).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|(ρ,α) in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    num_well : int or None
        Specify to return the first ``num_well`` pairs of bounce points for each
        pitch along each field line. This is useful if ``num_well`` tightly
        bounds the actual number. As a reference, there are typically 20 wells
        per toroidal transit for a given pitch. You can check this by plotting
        the field lines with the ``_check_bounce_points`` method.

        If not specified, then all bounce points are returned. If there were fewer
        wells detected along a field line than the size of the last axis of the
        returned arrays, then that axis is padded with zero.
    check : bool
        Flag for debugging. Must be false for JAX transformations.
    plot : bool
        Whether to plot some things if check is true. Default is true.
    kwargs
        Keyword arguments into ``plot_ppoly``.

    Returns
    -------
    z1, z2 : (jnp.ndarray, jnp.ndarray)
        Shape (P, S, num_well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of |B|.

        If there were less than ``num_wells`` wells detected along a field line,
        then the last axis, which enumerates bounce points for a particular field
        line and pitch, is padded with zero.

    """
    B, dB_dz, pitch = _check_spline_shape(knots, B, dB_dz, pitch)
    P, S, degree = pitch.shape[0], B.shape[1], B.shape[0] - 1
    # Intersection points in local power basis.
    intersect = poly_root(
        c=B,
        k=pitch[..., jnp.newaxis],
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
    is_z1 = (dB_dz_sign <= 0) & is_intersect
    is_z2 = (dB_dz_sign >= 0) & _in_epigraph_and(is_intersect, dB_dz_sign)

    # Transform out of local power basis expansion.
    intersect = (intersect + knots[:-1, jnp.newaxis]).reshape(P, S, -1)
    # New versions of JAX only like static sentinels.
    sentinel = -10000000.0  # instead of knots[0] - 1
    z1 = take_mask(intersect, is_z1, size=num_well, fill_value=sentinel)
    z2 = take_mask(intersect, is_z2, size=num_well, fill_value=sentinel)

    mask = (z1 > sentinel) & (z2 > sentinel)
    # Set outside mask to same value so integration is over set of measure zero.
    z1 = jnp.where(mask, z1, 0.0)
    z2 = jnp.where(mask, z2, 0.0)

    if check:
        _check_bounce_points(z1, z2, pitch, knots, B, plot, **kwargs)

    return z1, z2


def _check_bounce_points(z1, z2, pitch, knots, B, plot=True, **kwargs):
    """Check that bounce points are computed correctly."""
    eps = kwargs.pop("eps", jnp.finfo(jnp.array(1.0).dtype).eps * 10)
    kwargs.setdefault(
        "title",
        r"Intersects $\zeta$ in epigraph($\vert B \vert$) s.t. "
        r"$\vert B \vert(\zeta) = 1/\lambda$",
    )
    kwargs.setdefault("klabel", r"$1/\lambda$")
    kwargs.setdefault("hlabel", r"$\zeta$")
    kwargs.setdefault("vlabel", r"$\vert B \vert$")
    plots = []

    assert z1.shape == z2.shape
    mask = (z1 - z2) != 0.0
    z1 = jnp.where(mask, z1, jnp.nan)
    z2 = jnp.where(mask, z2, jnp.nan)

    err_1 = jnp.any(z1 > z2, axis=-1)
    err_2 = jnp.any(z1[..., 1:] < z2[..., :-1], axis=-1)

    P, S, _ = z1.shape
    for s in range(S):
        Bs = PPoly(B[:, s], knots)
        for p in range(P):
            Bs_midpoint = Bs((z1[p, s] + z2[p, s]) / 2)
            err_3 = jnp.any(Bs_midpoint > pitch[p, s] + eps)
            if not (err_1[p, s] or err_2[p, s] or err_3):
                continue
            _z1 = z1[p, s][mask[p, s]]
            _z2 = z2[p, s][mask[p, s]]
            if plot:
                plot_ppoly(
                    ppoly=Bs,
                    z1=_z1,
                    z2=_z2,
                    k=pitch[p, s],
                    **kwargs,
                )

            print("      z1    |    z2")
            print(jnp.column_stack([_z1, _z2]))
            assert not err_1[p, s], "Intersects have an inversion.\n"
            assert not err_2[p, s], "Detected discontinuity.\n"
            assert not err_3, (
                f"Detected |B| = {Bs_midpoint[mask[p, s]]} > {pitch[p, s] + eps} "
                "= 1/λ in well, implying the straight line path between "
                "bounce points is in hypograph(|B|). Use more knots.\n"
            )
        if plot:
            plots.append(
                plot_ppoly(
                    ppoly=Bs,
                    z1=z1[:, s],
                    z2=z2[:, s],
                    k=pitch[:, s],
                    **kwargs,
                )
            )
    return plots


def bounce_quadrature(
    x,
    w,
    z1,
    z2,
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
    z1, z2 : jnp.ndarray
        Shape (P, S, num_well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of |B|.
    pitch : jnp.ndarray
        Shape must broadcast with (P, S).
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
        Shape (S, knots.size).
        Real scalar-valued functions evaluated on the ``knots``.
        These functions should be arguments to the callable ``integrand``.
    data : dict[str, jnp.ndarray]
        Data evaluated on ``grid`` and reshaped with ``Bounce1D.reshape_data``.
        Must include names in ``Bounce1D.required_names()``.
    knots : jnp.ndarray
        Shape (knots.size, ).
        Unique ζ coordinates where the arrays in ``data`` and ``f`` were evaluated.
    method : str
        Method of interpolation.
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
        Quadrature for every pitch.
        First axis enumerates pitch values. Second axis enumerates the field lines.
        Last axis enumerates the bounce integrals.

    """
    errorif(z1.ndim != 3 or z1.shape != z2.shape)
    errorif(x.ndim != 1 or x.shape != w.shape)
    pitch = jnp.atleast_2d(pitch)
    if not isinstance(f, (list, tuple)):
        f = [f]

    # Integrate and complete the change of variable.
    if batch:
        result = _interpolate_and_integrate(
            w=w,
            Q=bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis]),
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
        def loop(z):
            z1, z2 = z
            # Need to return tuple because input was tuple; artifact of JAX map.
            return None, _interpolate_and_integrate(
                w=w,
                Q=bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis]),
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
            imap(loop, (jnp.moveaxis(z1, -1, 0), jnp.moveaxis(z2, -1, 0)))[1],
            source=0,
            destination=-1,
        )

    result = result * grad_bijection_from_disc(z1, z2)
    assert result.shape == (pitch.shape[0], data["|B|"].shape[0], z1.shape[-1])
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
        Quadrature points in ζ coordinates.
    data : dict[str, jnp.ndarray]
        Data evaluated on ``grid`` and reshaped with ``Bounce1D.reshape_data``.
        Must include names in ``Bounce1D.required_names()``.

    Returns
    -------
    result : jnp.ndarray
        Shape Q.shape[:-1].
        Quadrature for every pitch.

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
    b_sup_z = interp1d_Hermite_vec(
        Q,
        knots,
        data["B^zeta"] / data["|B|"],
        data["B^zeta_z|r,a"] / data["|B|"]
        - data["B^zeta"] * data["|B|_z|r,a"] / data["|B|"] ** 2,
    ).reshape(shape)
    B = interp1d_Hermite_vec(Q, knots, data["|B|"], data["|B|_z|r,a"]).reshape(shape)
    # Spline each function separately so that operations in the integrand
    # that do not preserve smoothness can be captured.
    f = [interp1d_vec(Q, knots, f_i, method=method).reshape(shape) for f_i in f]
    result = jnp.dot(integrand(*f, B=B, pitch=pitch) / b_sup_z, w)

    if check:
        _check_interp(Q.reshape(shape), f, b_sup_z, B, data["|B|_z|r,a"], result, plot)

    return result


def _check_interp(Q, f, b_sup_z, B, B_z_ra, result, plot):
    """Check for floating point errors.

    Parameters
    ----------
    Q : jnp.ndarray
        Quadrature points in ζ coordinates.
    f : list of jnp.ndarray
        Arguments to the integrand, interpolated to Q.
    b_sup_z : jnp.ndarray
        Contravariant toroidal component of magnetic field, interpolated to Q.
    B : jnp.ndarray
        Norm of magnetic field, interpolated to Q.
    B_z_ra : jnp.ndarray
        Norm of magnetic field derivative, (∂|B|/∂ζ)|(ρ,α).
    result : jnp.ndarray
        Output of ``_interpolate_and_integrate``.
    plot : bool
        Whether to plot stuff.

    """
    assert jnp.isfinite(Q).all(), "NaN interpolation point."
    # Integrals that we should be computing.
    marked = jnp.any(Q != 0.0, axis=-1)
    goal = marked.sum()

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
        "is caused by floating point error, usually due to a poor quadrature choice."
    )
    if plot:
        _plot_check_interp(Q, B, name=r"$\vert B \vert$")
        _plot_check_interp(Q, b_sup_z, name=r"$ (B / \vert B \vert) \cdot e^{\zeta}$")


def _plot_check_interp(Q, V, name=""):
    """Plot V[λ, (ρ, α), (ζ₁, ζ₂)](Q)."""
    for p in range(Q.shape[0]):
        for s in range(Q.shape[1]):
            marked = jnp.nonzero(jnp.any(Q != 0.0, axis=-1))[0]
            if marked.size == 0:
                continue
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\zeta$")
            ax.set_ylabel(name)
            ax.set_title(
                f"Interpolation of {name} to quadrature points. Index {p},{s}."
            )
            for i in marked:
                ax.plot(Q[p, s, i], V[p, s, i], marker="o")
            fig.text(
                0.01,
                0.01,
                f"Each color specifies the set of points and values (ζ, {name}(ζ)) "
                "used to evaluate an integral.",
            )
            plt.tight_layout()
            plt.show()


def _get_extrema(knots, g, dg_dz, sentinel=jnp.nan):
    """Return extrema (ζ*, g(ζ*)).

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (knots.size, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (g.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of g in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (g.shape[0] - 1, *g.shape[1:]).
        Polynomial coefficients of the spline of ∂g/∂ζ in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    sentinel : float
        Value with which to pad array to return fixed shape.

    Returns
    -------
    ext, g_ext : jnp.ndarray
        Shape (S, (knots.size - 1) * (degree - 1)).
        First array enumerates ζ*. Second array enumerates g(ζ*)
        Sorting order of extrema is arbitrary.

    """
    g, dg_dz, _ = _check_spline_shape(knots, g, dg_dz)
    S, degree = g.shape[1], g.shape[0] - 1
    ext = poly_root(
        c=dg_dz, a_min=jnp.array([0.0]), a_max=jnp.diff(knots), sentinel=sentinel
    )
    assert ext.shape == (S, knots.size - 1, degree - 1)
    g_ext = polyval_vec(x=ext, c=g[..., jnp.newaxis]).reshape(S, -1)
    # Transform out of local power basis expansion.
    ext = (ext + knots[:-1, jnp.newaxis]).reshape(S, -1)
    return ext, g_ext


def _where_for_argmin(z1, z2, ext, g_ext, upper_sentinel):
    assert z1.shape[1] == z2.shape[1] == ext.shape[0] == g_ext.shape[0]
    return jnp.where(
        (z1[..., jnp.newaxis] < ext[:, jnp.newaxis])
        & (ext[:, jnp.newaxis] < z2[..., jnp.newaxis]),
        g_ext[:, jnp.newaxis],
        upper_sentinel,
    )


def interp_to_argmin(
    h, z1, z2, knots, g, dg_dz, method="cubic", beta=-100, upper_sentinel=1e2
):
    """Interpolate ``h`` to the deepest point of ``g`` between ``z1`` and ``z2``.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A = argmin_E g(ζ). Returns mean_A h(ζ).

    Parameters
    ----------
    h : jnp.ndarray
        Shape must broadcast with (S, knots.size).
        Values evaluated on ``knots`` to interpolate.
    z1, z2 : jnp.ndarray
        Shape (P, S, num_well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of g.
    knots : jnp.ndarray
        Shape (knots.size, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (g.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of g in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (g.shape[0] - 1, *g.shape[1:]).
        Polynomial coefficients of the spline of ∂g/∂ζ in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    method : str
        Method of interpolation.
        See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
        Default is cubic C1 local spline.
    beta : float
        More negative gives exponentially better approximation at the
        expense of noisier gradients - noisier in the physics sense (unrelated
        to the automatic differentiation).
    upper_sentinel : float
        Something larger than g. Choose value such that
        exp(max(g)) << exp(``upper_sentinel``). Don't make too large or numerical
        resolution is lost.

    Warnings
    --------
    Recall that if g is small then the effect of β is reduced.
    If the intention is to use this function as argmax, be sure to supply
    a lower sentinel for ``upper_sentinel``.

    Returns
    -------
    h : jnp.ndarray
        Shape (P, S, num_well).
        mean_A h(ζ)

    """
    ext, g = _get_extrema(knots, g, dg_dz, sentinel=0)
    # JAX softmax(x) does the proper shift to compute softmax(x - max(x)), but it's
    # still not a good idea to compute over a large length scale, so we warn in
    # docstring to choose upper sentinel properly.
    argmin = softargmax(
        beta * _where_for_argmin(z1, z2, ext, g, upper_sentinel), axis=-1
    )
    h = jnp.linalg.vecdot(
        argmin,
        interp1d_vec(ext, knots, jnp.atleast_2d(h), method=method)[:, jnp.newaxis],
    )
    assert h.shape == z1.shape
    return h


def interp_to_argmin_hard(h, z1, z2, knots, g, dg_dz, method="cubic"):
    """Interpolate ``h`` to the deepest point of ``g`` between ``z1`` and ``z2``.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E g(ζ). Returns h(A).

    See Also
    --------
    interp_to_argmin
        Accomplishes the same task, but handles the case of non-unique global minima
        more correctly. It is also more efficient if P >> 1.

    Parameters
    ----------
    h : jnp.ndarray
        Shape must broadcast with (S, knots.size).
        Values evaluated on ``knots`` to interpolate.
    z1, z2 : jnp.ndarray
        Shape (P, S, num_well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of g.
    knots : jnp.ndarray
        Shape (knots.size, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (g.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of g in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (g.shape[0] - 1, *g.shape[1:]).
        Polynomial coefficients of the spline of ∂g/∂ζ in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines. Last axis enumerates the polynomials that
        compose a particular spline.
    method : str
        Method of interpolation.
        See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
        Default is cubic C1 local spline.

    Returns
    -------
    h : jnp.ndarray
        Shape (P, S, num_well).
        h(A)

    """
    ext, g = _get_extrema(knots, g, dg_dz, sentinel=0)
    # We can use the non-differentiable max because we actually want the gradients
    # to accumulate through only the minimum since we are differentiating how our
    # physics objective changes wrt equilibrium perturbations not wrt which of the
    # extrema get interpolated to.
    argmin = jnp.argmin(_where_for_argmin(z1, z2, ext, g, jnp.max(g) + 1), axis=-1)
    A = jnp.take_along_axis(ext[jnp.newaxis], argmin, axis=-1)
    h = interp1d_vec(A, knots, jnp.atleast_2d(h), method=method)
    assert h.shape == z1.shape
    return h


def plot_ppoly(
    ppoly,
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
        Optional, intersects with ∂f/∂z <= 0.
    z2 : jnp.ndarray
        Shape (k.shape[0], W).
        Optional, intersects with ∂f/∂z >= 0.
    k : jnp.ndarray
        Shape (k.shape[0], ).
        Optional, k such that f(z) = k.
    k_transparency : float
        Transparency of intersect lines.
    klabel : str
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
        Minimum z on plot.
    stop : float
        Maximum z on plot.
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
    ax.legend(legend.values(), legend.keys(), loc="lower right")
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close()
    return fig, ax
