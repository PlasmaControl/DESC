"""Utilities for bounce integrals."""

import numpy as np
from interpax import CubicSpline, PPoly
from matplotlib import pyplot as plt

from desc.backend import dct, imap, jnp
from desc.integrals._basis import (
    FourierChebyshevSeries,
    PiecewiseChebyshevSeries,
    _add2legend,
    _in_epigraph_and,
    _plot_intersect,
)
from desc.integrals._interp_utils import (
    cheb_from_dct,
    cheb_pts,
    idct_non_uniform,
    interp1d_Hermite_vec,
    interp1d_vec,
    interp_rfft2,
    irfft2_non_uniform,
    polyroot_vec,
    polyval_vec,
)
from desc.integrals.quad_utils import (
    bijection_from_disc,
    grad_bijection_from_disc,
    simpson2,
    uniform,
)
from desc.utils import (
    atleast_nd,
    errorif,
    flatten_matrix,
    is_broadcastable,
    setdefault,
    take_mask,
)


def get_pitch_inv_quad(min_B, max_B, num_pitch, simp=False):
    """Return 1/λ values and weights for quadrature between ``min_B`` and ``max_B``.

    Parameters
    ----------
    min_B : jnp.ndarray
        Minimum |B| value.
    max_B : jnp.ndarray
        Maximum |B| value.
    num_pitch : int
        Number of values.
    simp : bool
        Whether to use an open Simpson rule instead of uniform weights.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Shape (*min_B.shape, num pitch).
        1/λ values and weights.

    """
    errorif(
        num_pitch > 1e5,
        msg="Floating point error impedes detection of bounce points "
        f"near global extrema. Choose {num_pitch} < 1e5.",
    )
    # Samples should be uniformly spaced in |B| and not λ.
    # Important to do an open quadrature since the bounce integrals at the
    # global maxima of |B| are not computable even ignoring precision issues.
    x, w = simpson2(num_pitch) if simp else uniform(num_pitch)
    x = bijection_from_disc(x, min_B[..., jnp.newaxis], max_B[..., jnp.newaxis])
    w = w * grad_bijection_from_disc(min_B, max_B)[..., jnp.newaxis]
    return x, w


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
        Shape (..., num pitch).
        1/λ values. 1/λ(α,ρ) is specified by ``pitch_inv[α,ρ]`` where in
        the latter the labels are interpreted as the indices that correspond
        to that field line.

    """
    errorif(knots.ndim != 1, msg=f"knots should be 1d, got shape {knots.shape}.")
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
        Shape (..., num pitch).
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
        pitch and field line. Default is ``None``, which will detect all wells,
        but due to current limitations in JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``,``B`` are the poloidal and
        toroidal Fourier resolution of |B|, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.

        If there were fewer wells detected along a field line than the size of the
        last axis of the returned arrays, then that axis is padded with zero.
    check : bool
        Flag for debugging. Must be false for JAX transformations.
    plot : bool
        Whether to plot some things if check is true. Default is true.
    kwargs
        Keyword arguments into ``plot_ppoly``.

    Returns
    -------
    z1, z2 : tuple[jnp.ndarray]
        Shape (..., num pitch, num well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of |B|.

        If there were less than ``num_well`` wells detected along a field line,
        then the last axis, which enumerates bounce points for a particular field
        line and pitch, is padded with zero.

    """
    B, dB_dz, pitch_inv = _check_spline_shape(knots, B, dB_dz, pitch_inv)
    intersect = polyroot_vec(
        c=B[..., jnp.newaxis, :, :],  # add num pitch axis
        k=pitch_inv[..., jnp.newaxis],  # add polynomial axis
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
    vlabel = r"$\vert B \vert$"
    kwargs.setdefault(
        "title",
        r"Intersects $\zeta$ in epigraph("
        + vlabel
        + ") s.t. "
        + vlabel
        + r"$(\zeta) = 1/\lambda$",
    )
    kwargs.setdefault("klabel", r"$1/\lambda$")
    kwargs.setdefault("hlabel", r"$\zeta$")
    kwargs.setdefault("vlabel", vlabel)
    return kwargs


def _check_bounce_points(z1, z2, pitch_inv, knots, B, plot=True, **kwargs):
    """Check that bounce points are computed correctly."""
    kwargs = _set_default_plot_kwargs(kwargs)
    plots = []

    assert z1.shape == z2.shape
    z1 = atleast_nd(4, z1)
    z2 = atleast_nd(4, z2)
    pitch_inv = atleast_nd(3, pitch_inv)
    B = atleast_nd(4, B)
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
    knots,
    integrand,
    pitch_inv,
    data,
    names,
    points,
    *,
    method="cubic",
    batch=True,
    check=False,
    plot=False,
):
    """Bounce integrate ∫ f(ρ,α,λ,ℓ) dℓ.

    Parameters
    ----------
    x : jnp.ndarray
        Shape (num quad, ).
        Quadrature points in [-1, 1].
    w : jnp.ndarray
        Shape (num quad, ).
        Quadrature weights.
    knots : jnp.ndarray
        Shape (N, ).
        Unique ζ coordinates where the arrays in ``data`` and ``f`` were evaluated.
    integrand : callable or list[callable]
        The composition operator on the set of functions in ``data`` that
        maps that determines ``f`` in ∫ f(ρ,α,λ,ℓ) dℓ. It should accept a dictionary
        which stores the interpolated data and the keyword argument ``pitch``.
    pitch_inv : jnp.ndarray
        Shape (num alpha, num rho, num pitch).
        1/λ values to compute the bounce integrals. 1/λ(α,ρ) is specified by
        ``pitch_inv[α,ρ]`` where in the latter the labels are interpreted
        as the indices that correspond to that field line.
    data : dict[str, jnp.ndarray]
        Shape (num alpha, num rho, num zeta).
        Real scalar-valued periodic functions in (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP)
        evaluated on the ``grid`` supplied to construct this object.
        Use the method ``Bounce1D.reshape_data`` to reshape the data into the
        expected shape.
    names : str or list[str]
        Names in ``data`` to interpolate. Default is all keys in ``data``.
    points : tuple[jnp.ndarray]
        Shape (num alpha, num rho, num pitch, num well).
        Optional, output of method ``self.points``.
        Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
        The points are ordered and grouped such that the straight line path
        between ``z1`` and ``z2`` resides in the epigraph of |B|.
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
        Shape (..., num pitch, num well).
        Last axis enumerates the bounce integrals for a field line,
        flux surface, and pitch.

    """
    errorif(x.ndim != 1 or x.shape != w.shape)
    z1, z2 = points
    errorif(z1.ndim < 2 or z1.shape != z2.shape)
    pitch_inv = jnp.atleast_1d(pitch_inv)

    # Integrate and complete the change of variable.
    if batch:
        result = _interpolate_and_integrate(
            batch=True,
            x=x,
            w=w,
            knots=knots,
            integrand=integrand,
            pitch_inv=pitch_inv,
            data=data,
            names=names,
            points=points,
            method=method,
            check=check,
            plot=plot,
        )
    else:

        def loop(points):  # over num well axis
            # Need to return tuple because input was tuple; artifact of JAX map.
            return None, _interpolate_and_integrate(
                batch=False,
                x=x,
                w=w,
                knots=knots,
                integrand=integrand,
                pitch_inv=pitch_inv,
                data=data,
                names=names,
                points=points,
                method=method,
                check=False,
                plot=False,
            )

        # TODO (#1386): Use batch_size arg of imap after
        #  increasing JAX version requirement.
        result = imap(loop, (jnp.moveaxis(z1, -1, 0), jnp.moveaxis(z2, -1, 0)))[1]
        result = [jnp.moveaxis(r, source=0, destination=-1) for r in result]

    return result


def _interpolate_and_integrate(
    batch,
    x,
    w,
    knots,
    integrand,
    pitch_inv,
    data,
    names,
    points,
    *,
    method="cubic",
    check=False,
    plot=False,
):
    z1, z2 = points
    # shape (..., num pitch, num well, num quad)
    Q = bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis])
    assert w.ndim == 1 and Q.shape[-1] == w.size
    assert Q.shape[-3 + (not batch)] == pitch_inv.shape[-1]
    assert data["|B|"].shape[-1] == knots.size
    shape = Q.shape
    if batch:
        Q = flatten_matrix(Q)

    b_sup_z = interp1d_Hermite_vec(
        Q,
        knots,
        (data["|B^zeta|"] / data["|B|"])[..., jnp.newaxis, :],
        (
            data["|B^zeta|_z|r,a"] / data["|B|"]
            - data["|B^zeta|"] * data["|B|_z|r,a"] / data["|B|"] ** 2
        )[..., jnp.newaxis, :],
    )
    B = interp1d_Hermite_vec(
        Q,
        knots,
        data["|B|"][..., jnp.newaxis, :],
        data["|B|_z|r,a"][..., jnp.newaxis, :],
    )
    # Spline each function separately so that operations in the integrand
    # that do not preserve smoothness can be captured.
    data = {
        name: interp1d_vec(Q, knots, data[name][..., jnp.newaxis, :], method=method)
        for name in names
    }
    pitch = 1 / pitch_inv[..., jnp.newaxis]
    cov = grad_bijection_from_disc(z1, z2)
    result = [
        (f(data, B, pitch) / b_sup_z).reshape(shape).dot(w) * cov for f in integrand
    ]

    if check:
        _check_interp(
            shape,
            Q,
            b_sup_z,
            B,
            result[0],
            [data[name] for name in names],
            plot=plot,
        )

    return result


def _check_interp(shape, Q, b_sup_z, B, result, f=None, plot=True):
    """Check for interpolation failures and floating point issues.

    Parameters
    ----------
    shape : tuple
        (..., num pitch, num well, num quad).
    Q : jnp.ndarray
        Quadrature points in ζ coordinates.
    b_sup_z : jnp.ndarray
        Contravariant toroidal component of magnetic field, interpolated to Q.
    B : jnp.ndarray
        Norm of magnetic field, interpolated to Q.
    result : jnp.ndarray
        Output of ``_interpolate_and_integrate``.
    f : list[jnp.ndarray]
        Arguments to the integrand, interpolated to Q.
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
    f = setdefault(f, [])
    for f_i in f:
        assert goal == (marked & jnp.isfinite(f_i).reshape(shape).all(axis=-1)).sum()

    if plot:
        Q = Q.reshape(shape)
        _plot_check_interp(Q, B.reshape(shape), name=r"$\vert B \vert$")
        _plot_check_interp(
            Q, b_sup_z.reshape(shape), name=r"$(B / \vert B \vert) \cdot e^{\zeta}$"
        )
        for i, f_i in enumerate(f):
            _plot_check_interp(Q, f_i.reshape(shape), name=f"f_{i}")

    # Number of those integrals that were computed.
    actual = (marked & jnp.isfinite(result)).sum()
    assert goal == actual, (
        f"Lost {goal - actual} integrals from NaN generation in the integrand. This "
        "is caused by floating point error, usually due to a poor quadrature choice."
    )


def _plot_check_interp(Q, V, name=""):
    """Plot V[..., λ, (ζ₁, ζ₂)](Q)."""
    if Q.shape[-2] == 1:
        # Just one well along the field line, so plot
        # interpolations for every pitch simultaneously.
        Q = Q.squeeze(axis=-2)
        V = V.squeeze(axis=-2)
        label = "(m,l)"
        shape = Q.shape[:2]
    else:
        label = "(m,l,p)"
        shape = Q.shape[:3]
    for idx in np.ndindex(shape):
        marked = jnp.nonzero(jnp.any(Q[idx] != 0.0, axis=-1))[0]
        if marked.size == 0:
            continue
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\zeta$")
        ax.set_ylabel(name)
        ax.set_title(f"Interpolation of {name} to quadrature points, {label}={idx}")
        for i in marked:
            ax.plot(Q[(*idx, i)], V[(*idx, i)], marker="o")
        fig.text(0.01, 0.01, "Each color specifies a bounce integral.")
        plt.tight_layout()
        plt.show()


def plot_ppoly(
    ppoly,
    num=5000,
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
    assert ext.shape == g_ext.shape
    assert ext.shape[-1] == g.shape[-2] * (g.shape[-1] - 2)
    return ext, g_ext


# We can use the non-differentiable min because we actually want the gradients
# to accumulate through only the minimum since we are differentiating how our
# physics objective changes wrt equilibrium perturbations not wrt which of the
# extrema get interpolated to.


def _argmin(z1, z2, ext, g_ext):
    assert z1.ndim > 1 and z2.ndim > 1
    # Given
    #      z1 and z2 with shape (..., num pitch, num well)
    # and ext, g_ext with shape (..., num extrema),
    # add dims to broadcast
    #      z1 and z2 with shape (..., num pitch, num well, 1).
    # and ext, g_ext with shape (...,         1,        1, num extrema).
    where = jnp.where(
        (z1[..., jnp.newaxis] < ext[..., jnp.newaxis, jnp.newaxis, :])
        & (ext[..., jnp.newaxis, jnp.newaxis, :] < z2[..., jnp.newaxis]),
        g_ext[..., jnp.newaxis, jnp.newaxis, :],
        jnp.inf,
    )
    # shape is (..., num pitch, num well, 1)
    return jnp.argmin(where, axis=-1, keepdims=True)


def _fft_argmin(z1, z2, ext, g_ext):
    assert z1.ndim >= 1 and z2.ndim >= 1
    # Given
    #      z1 and z2 with shape (..., num well)
    # and ext, g_ext with shape (..., num extrema),
    # add dims to broadcast
    #      z1 and z2 with shape (..., num well, 1).
    # and ext, g_ext with shape (...,        1, num extrema).
    where = jnp.where(
        (z1[..., jnp.newaxis] < ext[..., jnp.newaxis, :])
        & (ext[..., jnp.newaxis, :] < z2[..., jnp.newaxis]),
        g_ext[..., jnp.newaxis, :],
        jnp.inf,
    )
    # shape is (..., num well, 1)
    return jnp.argmin(where, axis=-1, keepdims=True)


def interp_to_argmin(h, points, knots, g, dg_dz, method="cubic"):
    """Interpolate ``h`` to the deepest point of ``g`` between ``z1`` and ``z2``.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E g(ζ). Returns h(A).

    Parameters
    ----------
    h : jnp.ndarray
        Shape (..., knots.size).
        Values evaluated on ``knots`` to interpolate.
    points : jnp.ndarray
        Shape (..., num pitch, num well).
        Boundaries to detect argmin between.
        First (second) element stores left (right) boundaries.
    knots : jnp.ndarray
        Shape (knots.size, ).
        z coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (..., knots.size - 1, g.shape[-1]).
        Polynomial coefficients of the spline of g in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (..., knots.size - 1, g.shape[-1] - 1).
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
        Shape (..., num pitch, num well).

    """
    z1, z2 = points
    ext, g_ext = _get_extrema(knots, g, dg_dz, sentinel=0)
    return jnp.take_along_axis(
        # adding axes to broadcast with num pitch and num well axes
        interp1d_vec(ext, knots, h, method=method)[..., jnp.newaxis, jnp.newaxis, :],
        _argmin(z1, z2, ext, g_ext),
        axis=-1,
    ).squeeze(axis=-1)


def interp_fft_to_argmin(
    NFP, T, h, points, knots, g, dg_dz, is_fourier=False, M=None, N=None
):
    """Interpolate ``h`` to the deepest point of ``g`` between ``z1`` and ``z2``.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E g(ζ). Returns h(A).

    Parameters
    ----------
    NFP : int
        Number of field periods.
    T : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {θ_α : ζ ↦ θ(α, ζ) | α ∈ A} where A = (α₀, α₁, …, αₘ₋₁) is the same
        field line as ``alpha``. Each Chebyshev series approximates θ over
        one toroidal transit.
    h : jnp.ndarray
        Shape (..., grid.num_theta, grid.num_zeta)
        Periodic function evaluated on tensor-product grid in (ρ, θ, ζ) with
        uniformly spaced nodes (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
        Preferably power of 2 for ``grid.num_theta`` and ``grid.num_zeta``.
    points : jnp.ndarray
        Shape (..., num well).
        Boundaries to detect argmin between.
        First (second) element stores left (right) boundaries.
    knots : jnp.ndarray
        Shape (knots.size, ).
        z coordinates of spline knots. Must be strictly increasing.
    g : jnp.ndarray
        Shape (..., knots.size - 1, g.shape[-1]).
        Polynomial coefficients of the spline of g in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dg_dz : jnp.ndarray
        Shape (..., knots.size - 1, g.shape[-1] - 1).
        Polynomial coefficients of the spline of ∂g/∂z in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    is_fourier : bool
        If true, then it is assumed that ``h`` is the Fourier
        transform as returned by ``Bounce2D.fourier``.
    M, N : int
        Fourier resolution.

    Returns
    -------
    h : jnp.ndarray
        Shape (..., num well).

    """
    ext, g_ext = _get_extrema(knots, g, dg_dz, sentinel=0)
    theta = T.eval1d(ext)
    if is_fourier:
        h = irfft2_non_uniform(
            theta,
            ext,
            h[..., jnp.newaxis, :, :],
            M,
            N,
            domain1=(0, 2 * jnp.pi / NFP),
            axes=(-1, -2),
        )
    else:
        h = interp_rfft2(
            theta,
            ext,
            h[..., jnp.newaxis, :, :],
            domain1=(0, 2 * jnp.pi / NFP),
            axes=(-1, -2),
        )
    h = h[..., jnp.newaxis, :]  # to broadcast with num well axis
    z1, z2 = points
    if z1[0].ndim == h.ndim - 1:
        h = h[jnp.newaxis]  # to broadcast with num pitch axis
    return jnp.take_along_axis(h, _fft_argmin(z1, z2, ext, g_ext), axis=-1).squeeze(
        axis=-1
    )


# TODO (#568): Generalize this beyond ζ = ϕ or just map to Clebsch with ϕ.
def get_fieldline(alpha_0, iota, num_transit, period):
    """Get sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) of field line.

    Parameters
    ----------
    alpha_0 : float
        Starting field line poloidal label.
    iota : jnp.ndarray
        Shape (num rho, ).
        Rotational transform normalized by 2π.
    num_transit : float
        Number of ``period``s to follow field line.
    period : float
        Toroidal period after which to update label.

    Returns
    -------
    fieldline : jnp.ndarray
        Shape (num rho, num transit) or broadcastable.
        Sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) that specify field line.

    """
    # Δϕ (∂α/∂ϕ) = Δϕ ι̅ = Δϕ ι/2π = Δϕ data["iota"]
    return alpha_0 + period * (
        iota if iota.size == 1 else iota[:, jnp.newaxis]
    ) * jnp.arange(num_transit)


def fourier_chebyshev(theta, iota, alpha, num_transit):
    """Parameterize θ along field lines ``alpha``.

    Parameters
    ----------
    theta : jnp.ndarray
        Shape (num rho, X, Y) or (X, Y).
        DESC coordinates θ sourced from the Clebsch coordinates
        ``FourierChebyshevSeries.nodes(M,N,domain=(0,2*jnp.pi))``.
        Use the ``Bounce2D.compute_theta`` method to obtain this.
    iota : jnp.ndarray
        Shape (num rho, ).
        Rotational transform normalized by 2π.
    alpha : float
        Starting field line poloidal label.
    num_transit : int
        Number of toroidal transits to follow field line.

    Returns
    -------
    T : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {θ_α : ζ ↦ θ(α, ζ) | α ∈ A} where A = (α₀, α₁, …, αₘ₋₁) is the same
        field line as ``alpha``. Each Chebyshev series approximates θ over
        one toroidal transit.

    Notes
    -----
    The field line label α changes discontinuously, so the approximation
    g defined with basis function in (α, ζ) coordinates to some continuous
    function f does not guarantee continuity between cuts of the field line
    until sufficient convergence of g to f.

    Note if g were defined with basis functions in straight field line
    coordinates, then continuity between cuts of the field line, as
    determined by the straight field line coordinates (ϑ, ζ), is
    guaranteed even with incomplete convergence (because the
    parameters (ϑ, ζ) change continuously along the field line).

    Do not interpret this as superior function approximation.
    Indeed, if g is defined with basis functions in (α, ζ) coordinates, then
    g(α=α₀, ζ) will sample the approximation to f(α=α₀, ζ) for the full domain in ζ.
    This holds even with incomplete convergence of g to f.
    However, if g is defined with basis functions in (ϑ, ζ) coordinates, then
    g(ϑ(α=α₀,ζ), ζ) will sample the approximation to f(α=α₀ ± ε, ζ) with ε → 0 as
    g converges to f.

    (Visually, the small discontinuity apparent in g(α, ζ) at cuts of the field
    line will not be visible in g(ϑ, ζ) because when moving along the field line
    with g(ϑ, ζ) one is continuously flowing away from the starting field line,
    (whereas g(α, ζ) has to "decide" at the cut what the next field line is).
    (If full convergence is difficult to achieve, then in the context of surface
    averaging bounce integrals, function approximation in (α, ζ) coordinates
    might be preferable because most of the bounce integrals do not stretch
    across toroidal transits).)

    Note that if g is an unbounded function, as all coordinates are, then
    it is impossible to approximate it with a finite number of periodic
    basis functions, so we are forced to use a Fourier Chebyshev series to
    interpolate θ anyway.

    We explicitly enforce continuity of our approximation of θ between
    cuts to short-circuit the convergence of the Fourier series for θ.
    This works to remove the small discontinuity between cuts of the field line
    because the first cut is on α=0, which is a knot of the Fourier series, and
    the Chebyshev points include a knot near endpoints, so θ at the next cut of
    the field line is known with precision. This map is infinitely differentiable
    on ζ ∈ ℝ.

    """
    # peeling off field lines
    fieldline = get_fieldline(alpha, iota, num_transit, 2 * jnp.pi)
    # Evaluating set of single variable maps is more efficient than evaluating
    # multivariable map, so we project θ to a set of Chebyshev series.
    T = FourierChebyshevSeries(f=theta, domain=(0, 2 * jnp.pi)).compute_cheb(fieldline)
    T.stitch()
    assert T.X == num_transit
    assert T.Y == theta.shape[-1]
    return T


def chebyshev(n0, n1, NFP, T, f, Y):
    """Chebyshev along field lines.

    Parameters
    ----------
    NFP : int
        Number of field periods.
    T : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {θ_α : ζ ↦ θ(α, ζ) | α ∈ A} where A = (α₀, α₁, …, αₘ₋₁) is the same
        field line as ``alpha``. Each Chebyshev series approximates θ over
        one toroidal transit.
    f : jnp.ndarray
        Fourier transform of f(θ, ζ) as returned by ``_fourier``.
        ``n0=grid.num_theta``, ``n1=grid.num_zeta``, ``NFP=grid.NFP``.
    Y : int
        Chebyshev spectral resolution for ``f``. Preferably power of 2.

    Returns
    -------
    f : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {f_α : ζ ↦ f(α, ζ) | α ∈ A} where A = (α₀, α₁, …, αₘ₋₁) is the same
        field line. Each Chebyshev series approximates f over one toroidal transit.

    """
    # When f = |B|, it is expected that Y > T.Y so the code immediately below
    # is then up-sampling the Chebyshev resolution, which is good since the
    # spectrum of |B| is wider than θ.

    # θ at Chebyshev points, reshaped to (num rho, num transit * num points)
    theta = T.evaluate(Y).reshape(*T.cheb.shape[:-2], T.X * Y)
    zeta = jnp.broadcast_to(cheb_pts(Y, domain=T.domain), (T.X, Y)).ravel()

    # f at Chebyshev points
    f = irfft2_non_uniform(
        theta,
        zeta,
        f[..., jnp.newaxis, :, :],
        n0=n0,
        n1=n1,
        domain1=(0, 2 * jnp.pi / NFP),
        axes=(-1, -2),
    ).reshape(*T.cheb.shape[:-1], Y)
    f = PiecewiseChebyshevSeries(cheb_from_dct(dct(f, type=2, axis=-1)) / Y, T.domain)
    return f


def cubic_spline(n0, n1, NFP, T, f, Y, check=False):
    """Cubic spline along field lines.

    Parameters
    ----------
    NFP : int
        Number of field periods.
    T : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {θ_α : ζ ↦ θ(α, ζ) | α ∈ A} where A = (α₀, α₁, …, αₘ₋₁) is the same
        field line as ``alpha``. Each Chebyshev series approximates θ over
        one toroidal transit.
    f : jnp.ndarray
        Fourier transform of f(θ, ζ) as returned by ``_fourier``.
        ``n0=grid.num_theta``, ``n1=grid.num_zeta``, ``NFP=grid.NFP``.
    Y : int
        Number of knots per transit to interpolate ``f``.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    Returns
    -------
    f : jnp.ndarray
        Shape (num rho, num transit * (Y - 1), 4).
        Polynomial coefficients of the spline of f in local power basis.
        Last axis enumerates the coefficients of power series. For a polynomial
        given by ∑ᵢⁿ cᵢ xⁱ, coefficient cᵢ is stored at ``f[...,n-i]``.
        Second axis enumerates the polynomials that compose a particular spline.
        First axis enumerates field lines of a particular flux surface.
    knots : jnp.ndarray
        Shape (num transit * (Y - 1)).
        Knots of spline ``f``.

    """
    knots = jnp.linspace(-1, 1, Y, endpoint=False)
    # θ at uniformly spaced points along field line
    theta = idct_non_uniform(knots, T.cheb[..., jnp.newaxis, :], T.Y).reshape(
        *T.cheb.shape[:-2], T.X * Y  # num rho, num transit * num points
    )
    knots = (
        bijection_from_disc(knots, T.domain[0], T.domain[-1])
        + (T.domain[-1] - T.domain[0]) * jnp.arange(T.X)[:, jnp.newaxis]
    ).ravel()

    f = jnp.moveaxis(
        CubicSpline(
            x=knots,
            y=irfft2_non_uniform(
                theta,
                knots,
                f[..., jnp.newaxis, :, :],
                n0=n0,
                n1=n1,
                domain1=(0, 2 * jnp.pi / NFP),
                axes=(-1, -2),
            ),
            axis=-1,
            check=check,
        ).c,
        source=(0, 1),
        destination=(-1, -2),
    )
    assert f.shape[-2:] == (T.X * Y - 1, 4)
    return f, knots
