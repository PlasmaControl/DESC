"""Utilities and functional programming interface for bounce integrals."""

import numpy as np
from interpax import PPoly
from matplotlib import pyplot as plt

from desc.backend import imap, jnp, softargmax
from desc.integrals.basis import _add2legend, _in_epigraph_and, _plot_intersect
from desc.integrals.interp_utils import (
    interp1d_Hermite_vec,
    interp1d_vec,
    polyroot_vec,
    polyval_vec,
)
from desc.integrals.quad_utils import (
    bijection_from_disc,
    composite_linspace,
    grad_bijection_from_disc,
)
from desc.utils import (
    atleast_nd,
    errorif,
    flatten_matrix,
    is_broadcastable,
    setdefault,
    take_mask,
)


def get_pitch_inv(min_B, max_B, num, relative_shift=1e-6):
    """Return 1/λ values for quadrature between ``min_B`` and ``max_B``.

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
    pitch_inv : jnp.ndarray
        Shape (*min_B.shape, num + 2).
        1/λ values.

    """
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + relative_shift) * min_B
    max_B = (1 - relative_shift) * max_B
    # Samples should be uniformly spaced in |B| and not λ (GitHub issue #1228).
    pitch_inv = jnp.moveaxis(composite_linspace(jnp.stack([min_B, max_B]), num), 0, -1)
    assert pitch_inv.shape == (*min_B.shape, num + 2)
    return pitch_inv


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


def _check_spline_shape(knots, g, dg_dz, pitch_inv=None):
    """Ensure inputs have compatible shape.

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (N, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (..., N - 1, g.shape[-1]).
        Polynomial coefficients of the spline of g in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (..., N - 1, g.shape[-1] - 1).
        Polynomial coefficients of the spline of ∂g/∂ζ in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    pitch_inv : jnp.ndarray
        Shape (..., P).
        1/λ values. 1/λ(α,ρ) is specified by ``pitch_inv[α,ρ]`` where in
        the latter the labels are interpreted as the indices that correspond
        to that field line.

    """
    errorif(knots.ndim != 1, msg=f"knots should be 1d; got shape {knots.shape}.")
    errorif(
        g.shape[-2] != (knots.size - 1),
        msg=(
            "Second to last axis does not enumerate polynomials of spline. "
            f"Spline shape {g.shape}. Knots shape {knots.shape}."
        ),
    )
    errorif(
        not (g.ndim == dg_dz.ndim < 5)
        or g.shape != (*dg_dz.shape[:-1], dg_dz.shape[-1] + 1),
        msg=f"Invalid shape {g.shape} for spline and derivative {dg_dz.shape}.",
    )
    g, dg_dz = jnp.atleast_2d(g, dg_dz)
    if pitch_inv is not None:
        pitch_inv = jnp.atleast_1d(pitch_inv)
        errorif(
            pitch_inv.ndim > 3
            or not is_broadcastable(pitch_inv.shape[:-1], g.shape[:-2]),
            msg=f"Invalid shape {pitch_inv.shape} for pitch angles.",
        )
    return g, dg_dz, pitch_inv


def bounce_points(
    pitch_inv, knots, B, dB_dz, num_well=None, check=False, plot=True, **kwargs
):
    """Compute the bounce points given spline of |B| and pitch λ.

    Parameters
    ----------
    pitch_inv : jnp.ndarray
        Shape (..., P).
        1/λ values to compute the bounce points.
    knots : jnp.ndarray
        Shape (N, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    B : jnp.ndarray
        Shape (..., N - 1, B.shape[-1]).
        Polynomial coefficients of the spline of |B| in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dB_dz : jnp.ndarray
        Shape (..., N - 1, B.shape[-1] - 1).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|(ρ,α) in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
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
        Shape (..., P, num_well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of |B|.

        If there were less than ``num_well`` wells detected along a field line,
        then the last axis, which enumerates bounce points for a particular field
        line and pitch, is padded with zero.

    """
    B, dB_dz, pitch_inv = _check_spline_shape(knots, B, dB_dz, pitch_inv)
    intersect = polyroot_vec(
        c=B[..., jnp.newaxis, :, :],  # Add P axis
        k=pitch_inv[..., jnp.newaxis],  # Add N axis
        a_min=jnp.array([0.0]),
        a_max=jnp.diff(knots),
        sort=True,
        sentinel=-1.0,
        distinct=True,
    )
    assert intersect.shape[-3:] == (
        pitch_inv.shape[-1],
        knots.size - 1,
        B.shape[-1] - 1,
    )

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    dB_sign = flatten_matrix(
        jnp.sign(polyval_vec(x=intersect, c=dB_dz[..., jnp.newaxis, :, jnp.newaxis, :]))
    )
    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = flatten_matrix(intersect) >= 0
    # Following discussion on page 3 and 5 of https://doi.org/10.1063/1.873749,
    # we ignore the bounce points of particles only assigned to a class that are
    # trapped outside this snapshot of the field line.
    is_z1 = (dB_sign <= 0) & is_intersect
    is_z2 = (dB_sign >= 0) & _in_epigraph_and(is_intersect, dB_sign)

    # Transform out of local power basis expansion.
    intersect = flatten_matrix(intersect + knots[:-1, jnp.newaxis])
    # New versions of JAX only like static sentinels.
    sentinel = -10000000.0  # instead of knots[0] - 1
    z1 = take_mask(intersect, is_z1, size=num_well, fill_value=sentinel)
    z2 = take_mask(intersect, is_z2, size=num_well, fill_value=sentinel)

    mask = (z1 > sentinel) & (z2 > sentinel)
    # Set outside mask to same value so integration is over set of measure zero.
    z1 = jnp.where(mask, z1, 0.0)
    z2 = jnp.where(mask, z2, 0.0)

    if check:
        _check_bounce_points(z1, z2, pitch_inv, knots, B, plot, **kwargs)

    return z1, z2


def _set_default_plot_kwargs(kwargs):
    kwargs.setdefault(
        "title",
        r"Intersects $\zeta$ in epigraph($\vert B \vert$) s.t. "
        r"$\vert B \vert(\zeta) = 1/\lambda$",
    )
    kwargs.setdefault("klabel", r"$1/\lambda$")
    kwargs.setdefault("hlabel", r"$\zeta$")
    kwargs.setdefault("vlabel", r"$\vert B \vert$")
    return kwargs


def _check_bounce_points(z1, z2, pitch_inv, knots, B, plot=True, **kwargs):
    """Check that bounce points are computed correctly."""
    z1 = atleast_nd(4, z1)
    z2 = atleast_nd(4, z2)
    pitch_inv = atleast_nd(3, pitch_inv)
    B = atleast_nd(4, B)

    kwargs = _set_default_plot_kwargs(kwargs)
    plots = []

    assert z1.shape == z2.shape
    mask = (z1 - z2) != 0.0
    z1 = jnp.where(mask, z1, jnp.nan)
    z2 = jnp.where(mask, z2, jnp.nan)

    err_1 = jnp.any(z1 > z2, axis=-1)
    err_2 = jnp.any(z1[..., 1:] < z2[..., :-1], axis=-1)

    eps = kwargs.pop("eps", jnp.finfo(jnp.array(1.0).dtype).eps * 10)
    for ml in np.ndindex(B.shape[:-2]):
        ppoly = PPoly(B[ml].T, knots)
        for p in range(pitch_inv.shape[-1]):
            idx = (*ml, p)
            B_midpoint = ppoly((z1[idx] + z2[idx]) / 2)
            err_3 = jnp.any(B_midpoint > pitch_inv[idx] + eps)
            if not (err_1[idx] or err_2[idx] or err_3):
                continue
            _z1 = z1[idx][mask[idx]]
            _z2 = z2[idx][mask[idx]]
            if plot:
                plot_ppoly(
                    ppoly=ppoly,
                    z1=_z1,
                    z2=_z2,
                    k=pitch_inv[idx],
                    title=kwargs.pop("title") + f", (m,l,p)={idx}",
                    **kwargs,
                )

            print("      z1    |    z2")
            print(jnp.column_stack([_z1, _z2]))
            assert not err_1[idx], "Intersects have an inversion.\n"
            assert not err_2[idx], "Detected discontinuity.\n"
            assert not err_3, (
                f"Detected |B| = {B_midpoint[mask[idx]]} > {pitch_inv[idx] + eps} "
                "= 1/λ in well, implying the straight line path between "
                "bounce points is in hypograph(|B|). Use more knots.\n"
            )
        if plot:
            plots.append(
                plot_ppoly(
                    ppoly=ppoly,
                    z1=z1[ml],
                    z2=z2[ml],
                    k=pitch_inv[ml],
                    **kwargs,
                )
            )
    return plots


def _bounce_quadrature(
    x,
    w,
    z1,
    z2,
    integrand,
    pitch_inv,
    f,
    data,
    knots,
    method="cubic",
    batch=True,
    check=False,
    plot=False,
):
    """Bounce integrate ∫ f(λ, ℓ) dℓ.

    Parameters
    ----------
    x : jnp.ndarray
        Shape (w.size, ).
        Quadrature points in [-1, 1].
    w : jnp.ndarray
        Shape (w.size, ).
        Quadrature weights.
    z1, z2 : jnp.ndarray
        Shape (..., P, num_well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of |B|.
    integrand : callable
        The composition operator on the set of functions in ``f`` that maps the
        functions in ``f`` to the integrand f(λ, ℓ) in ∫ f(λ, ℓ) dℓ. It should
        accept the arrays in ``f`` as arguments as well as the additional keyword
        arguments: ``B`` and ``pitch``. A quadrature will be performed to
        approximate the bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
    pitch_inv : jnp.ndarray
        Shape (..., P).
        1/λ values to compute the bounce integrals.
    f : list[jnp.ndarray]
        Shape (..., N).
        Real scalar-valued functions evaluated on the ``knots``.
        These functions should be arguments to the callable ``integrand``.
    data : dict[str, jnp.ndarray]
        Shape (..., 1, N).
        Required data evaluated on ``grid`` and reshaped with ``Bounce1D.reshape_data``.
        Must include names in ``Bounce1D.required_names``.
    knots : jnp.ndarray
        Shape (N, ).
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
        Whether to plot the quantities in the integrand interpolated to the
        quadrature points of each integral. Ignored if ``check`` is false.

    Returns
    -------
    result : jnp.ndarray
        Shape (..., P, num_well).
        Last axis enumerates the bounce integrals for a field line,
        flux surface, and pitch.

    """
    errorif(x.ndim != 1 or x.shape != w.shape)
    errorif(z1.ndim < 2 or z1.shape != z2.shape)
    pitch_inv = jnp.atleast_1d(pitch_inv)
    if not isinstance(f, (list, tuple)):
        f = [f] if isinstance(f, (jnp.ndarray, np.ndarray)) else list(f)

    # Integrate and complete the change of variable.
    if batch:
        result = _interpolate_and_integrate(
            w=w,
            Q=bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis]),
            pitch_inv=pitch_inv,
            integrand=integrand,
            f=f,
            data=data,
            knots=knots,
            method=method,
            check=check,
            plot=plot,
        )
    else:
        # TODO: Use batched vmap.
        def loop(z):  # over num well axis
            z1, z2 = z
            # Need to return tuple because input was tuple; artifact of JAX map.
            return None, _interpolate_and_integrate(
                w=w,
                Q=bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis]),
                pitch_inv=pitch_inv,
                integrand=integrand,
                f=f,
                data=data,
                knots=knots,
                method=method,
                check=False,
                plot=False,
                batch=True,
            )

        result = jnp.moveaxis(
            imap(loop, (jnp.moveaxis(z1, -1, 0), jnp.moveaxis(z2, -1, 0)))[1],
            source=0,
            destination=-1,
        )

    return result * grad_bijection_from_disc(z1, z2)


def _interpolate_and_integrate(
    w,
    Q,
    pitch_inv,
    integrand,
    f,
    data,
    knots,
    method,
    check,
    plot,
    batch=False,
):
    """Interpolate given functions to points ``Q`` and perform quadrature.

    Parameters
    ----------
    w : jnp.ndarray
        Shape (w.size, ).
        Quadrature weights.
    Q : jnp.ndarray
        Shape (..., P, Q.shape[-2], w.size).
        Quadrature points in ζ coordinates.

    Returns
    -------
    result : jnp.ndarray
        Shape Q.shape[:-1].
        Quadrature result.

    """
    assert w.ndim == 1 and Q.shape[-1] == w.size
    assert Q.shape[-3 + batch] == pitch_inv.shape[-1]
    assert data["|B|"].shape[-1] == knots.size

    shape = Q.shape
    if not batch:
        Q = flatten_matrix(Q)
    b_sup_z = interp1d_Hermite_vec(
        Q,
        knots,
        data["B^zeta"] / data["|B|"],
        data["B^zeta_z|r,a"] / data["|B|"]
        - data["B^zeta"] * data["|B|_z|r,a"] / data["|B|"] ** 2,
    )
    B = interp1d_Hermite_vec(Q, knots, data["|B|"], data["|B|_z|r,a"])
    # Spline each function separately so that operations in the integrand
    # that do not preserve smoothness can be captured.
    f = [interp1d_vec(Q, knots, f_i[..., jnp.newaxis, :], method=method) for f_i in f]
    result = (
        (integrand(*f, B=B, pitch=1 / pitch_inv[..., jnp.newaxis]) / b_sup_z)
        .reshape(shape)
        .dot(w)
    )
    if check:
        _check_interp(shape, Q, f, b_sup_z, B, result, plot)

    return result


def _check_interp(shape, Q, f, b_sup_z, B, result, plot):
    """Check for interpolation failures and floating point issues.

    Parameters
    ----------
    shape : tuple
        (..., P, Q.shape[-2], w.size).
    Q : jnp.ndarray
        Quadrature points in ζ coordinates.
    f : list[jnp.ndarray]
        Arguments to the integrand, interpolated to Q.
    b_sup_z : jnp.ndarray
        Contravariant toroidal component of magnetic field, interpolated to Q.
    B : jnp.ndarray
        Norm of magnetic field, interpolated to Q.
    result : jnp.ndarray
        Output of ``_interpolate_and_integrate``.
    plot : bool
        Whether to plot stuff.

    """
    assert jnp.isfinite(Q).all(), "NaN interpolation point."
    assert not (
        jnp.isclose(B, 0).any() or jnp.isclose(b_sup_z, 0).any()
    ), "|B| has vanished, violating the hairy ball theorem."

    # Integrals that we should be computing.
    marked = jnp.any(Q.reshape(shape) != 0.0, axis=-1)
    goal = marked.sum()

    assert goal == (marked & jnp.isfinite(b_sup_z).reshape(shape).all(axis=-1)).sum()
    assert goal == (marked & jnp.isfinite(B).reshape(shape).all(axis=-1)).sum()
    for f_i in f:
        assert goal == (marked & jnp.isfinite(f_i).reshape(shape).all(axis=-1)).sum()

    # Number of those integrals that were computed.
    actual = (marked & jnp.isfinite(result)).sum()
    assert goal == actual, (
        f"Lost {goal - actual} integrals from NaN generation in the integrand. This "
        "is caused by floating point error, usually due to a poor quadrature choice."
    )
    if plot:
        Q = Q.reshape(shape)
        _plot_check_interp(Q, B.reshape(shape), name=r"$\vert B \vert$")
        _plot_check_interp(
            Q, b_sup_z.reshape(shape), name=r"$(B / \vert B \vert) \cdot e^{\zeta}$"
        )


def _plot_check_interp(Q, V, name=""):
    """Plot V[..., λ, (ζ₁, ζ₂)](Q)."""
    for idx in np.ndindex(Q.shape[:3]):
        marked = jnp.nonzero(jnp.any(Q[idx] != 0.0, axis=-1))[0]
        if marked.size == 0:
            continue
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\zeta$")
        ax.set_ylabel(name)
        ax.set_title(f"Interpolation of {name} to quadrature points, (m,l,p)={idx}")
        for i in marked:
            ax.plot(Q[(*idx, i)], V[(*idx, i)], marker="o")
        fig.text(0.01, 0.01, "Each color specifies a particular integral.")
        plt.tight_layout()
        plt.show()


def _get_extrema(knots, g, dg_dz, sentinel=jnp.nan):
    """Return extrema (z*, g(z*)).

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (N, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (..., N - 1, g.shape[-1]).
        Polynomial coefficients of the spline of g in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (..., N - 1, g.shape[-1] - 1).
        Polynomial coefficients of the spline of ∂g/∂z in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    sentinel : float
        Value with which to pad array to return fixed shape.

    Returns
    -------
    ext, g_ext : jnp.ndarray
        Shape (..., (N - 1) * (g.shape[-1] - 2)).
        First array enumerates z*. Second array enumerates g(z*)
        Sorting order of extrema is arbitrary.

    """
    g, dg_dz, _ = _check_spline_shape(knots, g, dg_dz)
    ext = polyroot_vec(
        c=dg_dz, a_min=jnp.array([0.0]), a_max=jnp.diff(knots), sentinel=sentinel
    )
    g_ext = flatten_matrix(polyval_vec(x=ext, c=g[..., jnp.newaxis, :]))
    # Transform out of local power basis expansion.
    ext = flatten_matrix(ext + knots[:-1, jnp.newaxis])
    assert ext.shape == g_ext.shape and ext.shape[-1] == g.shape[-2] * (g.shape[-1] - 2)
    return ext, g_ext


def _where_for_argmin(z1, z2, ext, g_ext, upper_sentinel):
    return jnp.where(
        (z1[..., jnp.newaxis] < ext[..., jnp.newaxis, jnp.newaxis, :])
        & (ext[..., jnp.newaxis, jnp.newaxis, :] < z2[..., jnp.newaxis]),
        g_ext[..., jnp.newaxis, jnp.newaxis, :],
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
        Shape (..., N).
        Values evaluated on ``knots`` to interpolate.
    z1, z2 : jnp.ndarray
        Shape (..., P, W).
        Boundaries to detect argmin between.
    knots : jnp.ndarray
        Shape (N, ).
        z coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (..., N - 1, g.shape[-1]).
        Polynomial coefficients of the spline of g in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (..., N - 1, g.shape[-1] - 1).
        Polynomial coefficients of the spline of ∂g/∂z in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
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
        Shape (..., P, W).

    """
    assert z1.ndim == z2.ndim >= 2 and z1.shape == z2.shape
    ext, g_ext = _get_extrema(knots, g, dg_dz, sentinel=0)
    # Our softargmax(x) does the proper shift to compute softargmax(x - max(x)),
    # but it's still not a good idea to compute over a large length scale, so we
    # warn in docstring to choose upper sentinel properly.
    argmin = softargmax(
        beta * _where_for_argmin(z1, z2, ext, g_ext, upper_sentinel),
        axis=-1,
    )
    h = jnp.linalg.vecdot(
        argmin,
        interp1d_vec(ext, knots, h, method=method)[..., jnp.newaxis, jnp.newaxis, :],
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
        Shape (..., N).
        Values evaluated on ``knots`` to interpolate.
    z1, z2 : jnp.ndarray
        Shape (..., P, W).
        Boundaries to detect argmin between.
    knots : jnp.ndarray
        Shape (N, ).
        z coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (..., N - 1, g.shape[-1]).
        Polynomial coefficients of the spline of g in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (..., N - 1, g.shape[-1] - 1).
        Polynomial coefficients of the spline of ∂g/∂z in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    method : str
        Method of interpolation.
        See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
        Default is cubic C1 local spline.

    Returns
    -------
    h : jnp.ndarray
        Shape (..., P, W).

    """
    assert z1.ndim == z2.ndim >= 2 and z1.shape == z2.shape
    ext, g_ext = _get_extrema(knots, g, dg_dz, sentinel=0)
    # We can use the non-differentiable max because we actually want the gradients
    # to accumulate through only the minimum since we are differentiating how our
    # physics objective changes wrt equilibrium perturbations not wrt which of the
    # extrema get interpolated to.
    argmin = jnp.argmin(
        _where_for_argmin(z1, z2, ext, g_ext, jnp.max(g_ext) + 1),
        axis=-1,
    )
    h = interp1d_vec(
        jnp.take_along_axis(ext[jnp.newaxis], argmin, axis=-1),
        knots,
        h[..., jnp.newaxis, :],
        method=method,
    )
    assert h.shape == z1.shape, h.shape
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
    knot_transparency=0.2,
    include_legend=True,
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
    include_legend : bool
        Whether to include the legend in the plot. Default is true.

    Returns
    -------
    fig, ax
        Matplotlib (fig, ax) tuple.

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
    if include_legend:
        ax.legend(legend.values(), legend.keys(), loc="lower right")
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close()
    return fig, ax
