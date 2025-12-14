"""Utilities for bounce integrals.

Note that since the filename is preceded by an underscore,
these utilities are private, and although it is unlikely,
their API may change without warning.
"""

from functools import partial

import numpy as np
from interpax import CubicSpline, PPoly
from interpax_fft import (
    FourierChebyshevSeries,
    cheb_from_dct,
    cheb_pts,
    dct_from_cheb,
    idct_mmt,
    ifft_mmt,
    irfft_mmt_pos,
)
from matplotlib import pyplot as plt
from orthax.chebyshev import chebroots, chebvander

from desc.backend import dct, flatnonzero, fori_loop, idct, ifft, jnp
from desc.integrals._interp_utils import (
    _eps,
    _filter_distinct,
    _subtract_first,
    nufft1d2r,
    polyroot_vec,
    polyval_vec,
)
from desc.integrals.quad_utils import bijection_from_disc, bijection_to_disc
from desc.io import IOAble
from desc.utils import (
    atleast_2d_end,
    atleast_3d_mid,
    atleast_nd,
    errorif,
    flatten_mat,
    setdefault,
    take_mask,
    warnif,
)

_sentinel = -1e5
_chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def _in_epigraph_and(is_intersect, df_dy, /):
    """Set and epigraph of function f with the given set of points.

    Used to return only intersects where the straight line path between
    adjacent intersects resides in the epigraph of a continuous map ``f``.

    Parameters
    ----------
    is_intersect : jnp.ndarray
        Boolean array indicating whether index corresponds to an intersect.
    df_dy : jnp.ndarray
        Shape ``is_intersect.shape``.
        Sign of ∂f/∂y (yᵢ) for f(yᵢ) = 0.

    Returns
    -------
    is_intersect : jnp.ndarray
        Boolean array indicating whether element is an intersect
        and satisfies the stated condition.

    Examples
    --------
    See ``desc/integrals/_bounce_utils.py::bounce_points``.
    This is used there to ensure the domains of integration are magnetic wells.

    """
    # The pairs y1 and y2 are boundaries of an integral only if y1 <= y2. For the
    # to be over wells, it is required that the first intersect has a non-positive
    # derivative. Now, by continuity, df_dy[...,k]<=0 implies df_dy[...,k+1]>=0, so
    # there can be at most one inversion, and if it exists, it must be at the first
    # pair. To correct the inversion, it suffices to disqualify the first intersect
    # as a right boundary, except under an edge case of a series of inflection points.
    idx = flatnonzero(is_intersect, size=2, fill_value=-1)
    edge_case = (
        (df_dy[idx[0]] == 0)
        & (df_dy[idx[1]] < 0)
        & is_intersect[idx[0]]
        & is_intersect[idx[1]]
        # In theory, we need to keep propagating this edge case, e.g.
        # (df_dy[..., 1] < 0) | (
        #     (df_dy[..., 1] == 0) & (df_dy[..., 2] < 0)...
        # ).
        # At each step, the likelihood that an intersection has already been lost
        # due to floating point errors grows, so the real solution is to pick a less
        # degenerate pitch value - one that does not ride the global extrema of f.
    )
    return is_intersect.at[idx[0]].set(edge_case)


def bounce_points(pitch_inv, knots, B, dB_dz, num_well=None):
    """Compute the bounce points given 1D spline of B and pitch λ.

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
        Polynomial coefficients of the spline of B in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    dB_dz : jnp.ndarray
        Shape (..., N - 1, B.shape[-1] - 1).
        Polynomial coefficients of the spline of (∂B/∂ζ)|(ρ,α) in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    num_well : int or None
        Specify to return the first ``num_well`` pairs of bounce points for each
        pitch and field line. Choosing ``-1`` will detect all wells, but due
        to current limitations in JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+C`` where ``A``, ``C`` are the poloidal and
        toroidal Fourier resolution of B, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+C)*num_transit`` is preferable.

        If there were fewer wells detected along a field line than the size of the
        last axis of the returned arrays, then that axis is padded with zero.

    Returns
    -------
    z1, z2 : tuple[jnp.ndarray]
        Shape (..., num pitch, num well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of B.

        If there were less than ``num_well`` wells detected along a field line,
        then the last axis, which enumerates bounce points for a particular field
        line and pitch, is padded with zero.

    """
    intersect = polyroot_vec(
        c=B[..., None, :, :],
        k=jnp.atleast_1d(pitch_inv)[..., None],
        a_min=jnp.array([0.0]),
        a_max=jnp.diff(knots),
        sort=True,
        sentinel=-1.0,
        distinct=True,
    )
    assert intersect.shape[-2:] == (knots.size - 1, B.shape[-1] - 1)

    dB_dz = flatten_mat(
        jnp.sign(polyval_vec(x=intersect, c=dB_dz[..., None, :, None, :]))
    )
    # Only consider intersect if it is within knots that bound that polynomial.
    mask = flatten_mat(intersect >= 0)
    z1 = (dB_dz <= 0) & mask
    z2 = (dB_dz >= 0) & _in_epigraph_and(mask, dB_dz)

    # Transform out of local power basis expansion.
    intersect = flatten_mat(intersect + knots[:-1, None])
    z1 = take_mask(intersect, z1, size=num_well, fill_value=_sentinel)
    z2 = take_mask(intersect, z2, size=num_well, fill_value=_sentinel)

    mask = (z1 > _sentinel) & (z2 > _sentinel)
    # Set to zero so integration is over set of measure zero
    # and basis functions are faster to evaluate in downstream routines.
    z1 = jnp.where(mask, z1, 0.0)
    z2 = jnp.where(mask, z2, 0.0)
    return z1, z2


def set_default_plot_kwargs(kwargs, l=None, m=None):
    """Sets some plot kwargs to defaults."""
    vlabel = r"$\vert B \vert$"
    default_title = (
        rf"Intersects $\zeta$ in epigraph$(${vlabel}$)$ "
        + rf"s.t. $\lambda${vlabel}$(\zeta) = 1$"
    )
    if l is not None and m is not None:
        default_title += rf" on field line $(\rho_{{l={l}}}, \alpha_{{m={m}}})$"
    kwargs.setdefault("title", default_title)
    kwargs.setdefault("klabel", r"$1/\lambda$")
    kwargs.setdefault("hlabel", r"$\zeta$")
    kwargs.setdefault("vlabel", vlabel)
    return kwargs


def check_bounce_points(z1, z2, pitch_inv, knots, B, plot=True, **kwargs):
    """Check that bounce points are computed correctly.

    For the plotting labels of ρ(l), α(m), it is assumed that the axis that
    enumerates the index l precedes the axis that enumerates the index m.
    """
    kwargs = set_default_plot_kwargs(kwargs)
    title = kwargs.pop("title")
    plots = []

    assert z1.shape == z2.shape
    assert knots.ndim == 1, f"knots should have ndim 1, got shape {knots.shape}."
    assert B.shape[-2] == (knots.size - 1), (
        "Second to last axis does not enumerate polynomials of spline. "
        f"Spline shape {B.shape}. Knots shape {knots.shape}."
    )
    assert knots[0] > _sentinel, "Reduce sentinel in desc/integrals/_bounce_utils.py."

    z1 = atleast_nd(4, z1)
    z2 = atleast_nd(4, z2)

    # do not need to broadcast to full size because
    # https://jax.readthedocs.io/en/latest/notebooks/
    # Common_Gotchas_in_JAX.html#out-of-bounds-indexing
    pitch_inv = atleast_nd(3, broadcast_for_bounce(pitch_inv))
    B = atleast_nd(4, B)

    mask = (z1 - z2) != 0.0
    z1 = jnp.where(mask, z1, jnp.nan)
    z2 = jnp.where(mask, z2, jnp.nan)

    err_1 = jnp.any(z1 > z2, axis=-1)
    err_2 = jnp.any(z1[..., 1:] < z2[..., :-1], axis=-1)

    eps = kwargs.pop("eps", jnp.finfo(jnp.array(1.0).dtype).eps * 10)
    for lm in np.ndindex(B.shape[:-2]):
        ppoly = PPoly(B[lm].T, knots)
        for p in range(pitch_inv.shape[-1]):
            idx = (*lm, p)
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
                    title=title
                    + rf" on field line $(\rho_{{l={lm[0]}}}, \alpha_{{m={lm[1]}}})$",
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
                    z1=z1[lm],
                    z2=z2[lm],
                    k=pitch_inv[lm],
                    title=title,
                    **kwargs,
                )
            )
    return plots


def check_interp(zeta, b_sup_z, B, f, result, plot=True):
    """Check for interpolation failures and floating point issues.

    Parameters
    ----------
    zeta : jnp.ndarray
        Quadrature points in ζ coordinates.
    b_sup_z : jnp.ndarray
        Contravariant toroidal component of magnetic field, interpolated to ``zeta``.
    B : jnp.ndarray
        Norm of magnetic field, interpolated to ``zeta``.
    f : list[jnp.ndarray]
        Arguments to the integrand, interpolated to ``zeta``.
    result : list[jnp.ndarray]
        Computed integrals.
    plot : bool
        Whether to plot stuff.

    """
    assert isinstance(result, list)
    assert jnp.isfinite(zeta).all(), "NaN interpolation point."
    assert not (
        jnp.isclose(B, 0).any() or jnp.isclose(b_sup_z, 0).any()
    ), "|B| has vanished, violating the hairy ball theorem."

    # Integrals that we should be computing.
    marked = jnp.any(zeta != 0.0, axis=-1)
    goal = marked.sum()

    assert goal == jnp.sum(marked & jnp.isfinite(b_sup_z).all(axis=-1))
    assert goal == jnp.sum(marked & jnp.isfinite(B).all(axis=-1))
    for f_i in f:
        assert goal == jnp.sum(marked & jnp.isfinite(f_i).all(axis=-1))

    if plot:
        _plot_check_interp(zeta, B, name=r"$\vert B \vert$")
        _plot_check_interp(
            zeta, b_sup_z, name=r"$B / \vert B \vert \cdot \nabla \zeta$"
        )
        for i, f_i in enumerate(f):
            _plot_check_interp(zeta, f_i, name=f"f_{i}")

    for res in result:
        # Number of those integrals that were computed.
        actual = jnp.sum(marked & jnp.isfinite(res))
        assert goal == actual, (
            f"Lost {goal - actual} integrals from NaN generation in the integrand."
            f" This is caused by floating point error due to a poor quadrature choice."
        )


def _plot_check_interp(zeta, V, name=""):
    """Plot V[..., λ, (ζ₁, ζ₂)](ζ)."""
    if zeta.shape[-2] == 1:
        # Just one well along the field line, so plot
        # interpolations for every pitch simultaneously.
        zeta = zeta.squeeze(-2)
        V = V.squeeze(-2)
        shape = zeta.shape[:2]
    else:
        shape = zeta.shape[:3]
    for idx in np.ndindex(shape):
        marked = jnp.nonzero(jnp.any(zeta[idx] != 0.0, axis=-1))[0]
        if marked.size == 0:
            continue
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\zeta$")
        ax.set_ylabel(name)
        ax.set_title(
            f"Interpolation of {name} to quadrature points"
            + rf" on field line $(\rho_{{l={idx[0]}}}, \alpha_{{m={idx[1]}}})$"
        )
        for i in marked:
            ax.plot(zeta[(*idx, i)], V[(*idx, i)], marker="o")
        fig.text(0.01, 0.01, "Each color specifies a bounce integral.")
        plt.show()


def plot_ppoly(
    ppoly,
    num=5000,
    z1=None,
    z2=None,
    k=None,
    *,
    k_transparency=0.5,
    klabel=r"$k$",
    title=r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$",
    hlabel=r"$z$",
    vlabel=r"$f$",
    show=True,
    start=None,
    stop=None,
    include_knots=False,
    knot_transparency=0.4,
    include_legend=True,
    return_legend=False,
    legend_kwargs=None,
    **kwargs,
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
        Whether to plot the legend. Default is true.

    Returns
    -------
    fig, ax
        Matplotlib (fig, ax) tuple.

    """
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))

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
    _add2legend(legend, ax.plot(z, ppoly(z), label=vlabel, **kwargs))
    _plot_intersect(
        ax=ax,
        legend=legend,
        z1=z1,
        z2=z2,
        k=k,
        k_transparency=k_transparency,
        klabel=klabel,
        hlabel=hlabel,
        **kwargs,
    )
    ax.set_xlabel(hlabel)
    ax.set_ylabel(vlabel)
    ax.set_title(title)

    if include_legend:
        if legend_kwargs is None:
            legend_kwargs = dict(loc="lower right")
        ax.legend(legend.values(), legend.keys(), **legend_kwargs)

    if show:
        plt.show()
        plt.close()
    return (fig, ax, legend) if return_legend else (fig, ax)


def _add2legend(legend, lines):
    """Add lines to legend if it's not already in it."""
    for line in setdefault(lines, [lines], hasattr(lines, "__iter__")):
        label = line.get_label()
        if label not in legend:
            legend[label] = line


def _plot_intersect(
    ax,
    legend,
    z1,
    z2,
    k,
    k_transparency,
    klabel,
    hlabel,
    markersize=plt.rcParams["lines.markersize"] * 3,
    **kwargs,
):
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
            ax.axhline(
                p,
                color="tab:purple",
                alpha=k_transparency,
                label=klabel,
                linestyle="--",
            ),
        )
    for i in range(k.size):
        _z1, _z2 = z1[i], z2[i]
        if _z1.size == _z2.size:
            mask = (_z1 - _z2) != 0.0
            _z1 = _z1[mask]
            _z2 = _z2[mask]
        _add2legend(
            legend,
            ax.scatter(
                _z1,
                jnp.full_like(_z1, k[i]),
                marker="v",
                color="tab:red",
                label=hlabel + r"$_1(w)$",
                s=markersize,
            ),
        )
        _add2legend(
            legend,
            ax.scatter(
                _z2,
                jnp.full_like(_z2, k[i]),
                marker="^",
                color="tab:green",
                label=hlabel + r"$_2(w)$",
                s=markersize,
            ),
        )


def get_extrema(knots, g, dg_dz, sentinel=jnp.nan):
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
    ext = polyroot_vec(
        c=dg_dz, a_min=jnp.array([0.0]), a_max=jnp.diff(knots), sentinel=sentinel
    )
    g_ext = flatten_mat(polyval_vec(x=ext, c=g[..., None, :]))
    # Transform out of local power basis expansion.
    ext = flatten_mat(ext + knots[:-1, None])
    assert ext.shape == g_ext.shape
    assert ext.shape[-1] == g.shape[-2] * (g.shape[-1] - 2)
    return ext, g_ext


def argmin(z1, z2, f, ext, g_ext):
    """Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E g(ζ). Returns f(A).

    Parameters
    ----------
    z1, z2 : jnp.ndarray
        Shape (..., num pitch, num well).
        Boundaries to detect argmin between.
        ``z1`` (``z2``) stores left (right) boundaries.
    f : jnp.ndarray
        Function interpolated to ``ext``.
        Shape (..., num extrema).

    Returns
    -------
    f : jnp.ndarray
        Shape (..., num pitch, num well).
        ``f`` at the minimum extrema of ``g`` between ``z1`` and ``z2``.

    """
    assert z1.ndim > 1 and z2.ndim > 1
    assert f.shape == ext.shape == g_ext.shape
    # We can use the non-differentiable argmin because we actually want the gradients
    # to accumulate through only the minimum since we are differentiating how our
    # physics objective changes wrt equilibrium perturbations not wrt which of the
    # extrema get interpolated to.
    where = jnp.argmin(
        jnp.where(
            (z1[..., None] < ext[..., None, None, :])
            & (ext[..., None, None, :] < z2[..., None]),
            g_ext[..., None, None, :],
            jnp.inf,
        ),
        axis=-1,
        keepdims=True,
    )
    return jnp.take_along_axis(f[..., None, None, :], where, axis=-1).squeeze(-1)


def get_alphas(alpha, iota, num_transit, NFP):
    """Get set of field line poloidal coordinates {Aᵢ | Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)}.

    Parameters
    ----------
    alpha : jnp.ndarray
        Shape (num α, ).
        Starting field line poloidal labels {αᵢ₀}.
    iota : jnp.ndarray
        Shape (num ρ, ).
        Rotational transform normalized by 2π.
    num_transit : int
        Number of toroidal transits to follow field line.
    NFP: int
        Number of field periods.

    Returns
    -------
    alphas : jnp.ndarray
        Shape (num α, num ρ, num transit * NFP).
        Set of field line poloidal coordinates {Aᵢ | Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)}.

    """
    alpha = alpha[:, None, None]
    iota = iota[:, None]
    return alpha + iota * (2 * jnp.pi / NFP) * jnp.arange(num_transit * NFP)


def theta_on_fieldlines(angle, iota, alpha, num_transit, NFP):
    """Parameterize θ on field lines α.

    Parameters
    ----------
    angle : jnp.ndarray
        Shape (num ρ, X, Y).
        Angle returned by ``Bounce2D.angle``.
    iota : jnp.ndarray
        Shape (num ρ, ).
        Rotational transform normalized by 2π.
    alpha : jnp.ndarray
        Shape (num α, ).
        Starting field line poloidal labels {αᵢ₀}.
    num_transit : int
        Number of toroidal transits to follow field line.
    NFP : int
        Number of field periods.

    Returns
    -------
    theta : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line ``α[i]``. Each Chebyshev series approximates
        θ over one toroidal transit. ``theta.cheb`` broadcasts with
        shape (num ρ, num α, num transit * NFP, max(1,7Y//8)).

    Notes
    -----
    To accelerate convergence, we introduced the stream variable δ such that
    θ = α + δ. This stream map δ : α, ζ ↦ δ(α, ζ) is linear in θ.
    Hence, it may be interpolated directly from discrete solutions θ* to

    θ* - (δ−ιζ)(θ*, ζ) = α + ιζ.

    This feature avoids expensive off-grid re-interpolation in optimization.

    Note the field line label α changes discontinuously along a magnetic field line.
    So an approximation f defined with basis functions in (α, NFP ζ) coordinates to
    some map F which is continuous along the magnetic field line does not guarantee
    continuity between branch cuts of (α, NFP ζ) ∈ [0, 2π)² until sufficient convergence
    of f to F. If f is instead defined with basis functions in flux coordinates such as
    (ϑ, NFP ζ), then continuity between branch cuts of (α, NFP ζ) ∈ [0, 2π)² is
    guaranteed even with incomplete convergence because the parameters (ϑ, ζ) change
    continuously along the magnetic field line.

    This does not imply a parameterization without branch cuts is superior for
    approximation; convergence is determined by the properties of the basis and the
    domain size moreso than whether the parameters have branch cuts on the domain.
    For example, if f is defined with basis functions in (α, NFP ζ) coordinates, then
    f(α=α₀, ζ) will sample the approximation to F(α=α₀, ζ) for ζ ∈ [0, NFP 2π) even
    with incomplete convergence. However, if f is defined with basis functions in
    (ϑ, NFP ζ) coordinates, then f(ϑ(α=α₀, ζ), ζ) will sample the approximation to
    F(α=α₀ ± ε, ζ) with ε → 0 as f converges to F.

    This property was mentioned because parameterizing the stream map in (α, ζ) enables
    partial summation. However, the small discontinuity due to discretization error
    between branch cuts is undesirable as it can give significant error to the singular
    integrals whose integration boundary is near a branch cut. If we were using splines
    instead of pseudo-spectral methods to interpolate then we would have to account
    for this.

    """
    num_alpha = alpha.size
    # peeling off field lines
    alpha = get_alphas(alpha, iota, num_transit, NFP)
    if angle.ndim == 2:
        alpha = alpha.squeeze(1)

    # Mod early for speed and conditioning
    # (since this avoids modding on more points later and keeps θ bounded).
    alpha %= 2 * jnp.pi

    domain = (0, 2 * jnp.pi / NFP)
    Y = truncate_rule(angle.shape[-1])
    delta = (
        FourierChebyshevSeries(angle, domain, truncate=Y)
        .compute_cheb(alpha)
        .swapaxes(0, -3)
    )
    alpha = alpha.swapaxes(0, -2)
    delta = delta.at[..., 0].add(alpha)
    assert delta.shape == (*angle.shape[:-2], num_alpha, num_transit * NFP, Y)
    return PiecewiseChebyshevSeries(delta, domain)


def fast_chebyshev(theta, f, Y, num_θ, modes_θ, modes_z, *, vander=None):
    """Compute Chebyshev approximation of ``f`` on field lines using fast transforms.

    Parameters
    ----------
    theta : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line αᵢ. Each Chebyshev series approximates
        θ over one toroidal transit. ``theta.cheb`` should broadcast with
        shape (num ρ, num α, num transit * NFP, theta.Y).
    f : jnp.ndarray
        Shape broadcasts with (num ρ, 1, modes_z.size, modes_θ.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Chebyshev spectral resolution for ``f`` over a field period.
        Preferably power of 2.
    num_θ : int
        Fourier resolution in poloidal direction.
    modes_θ : jnp.ndarray
        Real FFT Fourier modes in poloidal direction.
    modes_z : jnp.ndarray
        FFT Fourier modes in toroidal direction.
    vander : jnp.ndarray
        Precomputed transform matrix.

    Returns
    -------
    f : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of ``f`` on field lines.
        {f_αᵢⱼ : ζ ↦ f(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line αᵢ. Each Chebyshev series approximates
        ``f`` over one toroidal transit. ``f.cheb`` broadcasts with
        shape (num ρ, num α, num transit * NFP, Y).

    """
    # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
    # compute a set of 2D Fourier series each on non-uniform tensor product grids
    # of size |𝛉|×|𝛇| where |𝛉| = num α × num transit × NFP and |𝛇| = Y.
    # Partial summation is more efficient than direct evaluation when
    # mn|𝛉||𝛇| > mn|𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > n + |𝛉|.

    f = ifft_mmt(
        cheb_pts(Y, theta.domain)[:, None] if vander is None else None,
        f,
        theta.domain,
        axis=-2,
        modes=modes_z,
        vander=vander,
    )[..., None, None, :, :]
    f = irfft_mmt_pos(theta.evaluate(Y), f, num_θ, modes=modes_θ)
    f = cheb_from_dct(dct(f, type=2, axis=-1) / Y)
    f = PiecewiseChebyshevSeries(f, theta.domain)
    assert f.cheb.shape == (*theta.cheb.shape[:-1], Y)
    return f


def fast_cubic_spline(
    theta,
    f,
    Y,
    num_θ,
    modes_θ,
    modes_z,
    NFP=1,
    nufft_eps=1e-6,
    *,
    vander_θ=None,
    vander_z=None,
    check=False,
):
    """Compute cubic spline of ``f`` on field lines using fast transforms.

    Parameters
    ----------
    theta : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line αᵢ. Each Chebyshev series approximates
        θ over one toroidal transit. ``theta.cheb`` should broadcast with
        shape (num ρ, num α, num transit * NFP, theta.Y).
    f : jnp.ndarray
        Shape broadcasts with (num ρ, 1, modes_z.size, modes_θ.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Number of knots per toroidal transit to interpolate ``f``.
        This number will be rounded up to an integer multiple of ``NFP``.
    num_θ : int
        Fourier resolution in poloidal direction.
    modes_θ : jnp.ndarray
        Real FFT Fourier modes in poloidal direction.
    modes_z : jnp.ndarray
        FFT Fourier modes in toroidal direction.
    NFP : int
        Number of field periods.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
    vander_θ : jnp.ndarray
        Precomputed transform matrix.
    vander_z : jnp.ndarray
        Precomputed transform matrix.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    Returns
    -------
    f : jnp.ndarray
        Shape broadcasts with (num ρ, num α, num transit * Y - 1, 4).
        Polynomial coefficients of the spline of f in local power basis.
        Last axis enumerates the coefficients of power series. For a polynomial
        given by ∑ᵢⁿ cᵢ xⁱ, coefficient cᵢ is stored at ``f[...,n-i]``.
        Second to last axis enumerates the polynomials that compose a particular
        spline.
    knots : jnp.ndarray
        Shape (num transit * Y).
        Knots of spline ``f``.

    """
    assert theta.domain == (0, 2 * jnp.pi / NFP)

    lines = theta.cheb.shape[:-2]
    num_transit = theta.X // NFP

    axisymmetric = f.shape[-2] == 1
    Y, num_z = round_up_rule(Y, NFP, axisymmetric)
    x = jnp.linspace(-1, 1, (Y // NFP) if axisymmetric else num_z, endpoint=False)
    z = bijection_from_disc(x, *theta.domain)

    # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
    # compute a set of 2D Fourier series each on uniform (non-uniform) in ζ (θ)
    # tensor product grids of size
    #   |𝛉|×|𝛇| where |𝛉| = num α × num transit × NFP and |𝛇| = Y/NFP.
    # Partial summation via FFT is more efficient than direct evaluation when
    # mn|𝛉||𝛇| > m log(|𝛇|) |𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > log|𝛇| + |𝛉|.

    if num_z >= f.shape[-2]:
        f = f.squeeze(-3)
        p = num_z - f.shape[-2]
        p = (p // 2, p - p // 2)
        pad = [(0, 0)] * f.ndim
        pad[-2] = p if (f.shape[-2] % 2 == 0) else p[::-1]
        f = jnp.fft.ifftshift(jnp.pad(jnp.fft.fftshift(f, -2), pad), -2)
        f = ifft(f, axis=-2, norm="forward")
    else:
        f = ifft_mmt(
            z[:, None],
            f,
            theta.domain,
            axis=-2,
            modes=modes_z,
            vander=vander_z,
        )

    # θ at uniform ζ on field lines
    θ = idct_mmt(
        x,
        theta.cheb.reshape(*lines, num_transit, NFP, 1, theta.Y),
        vander=vander_θ,
    )
    if axisymmetric:
        θ = θ.reshape(*lines, num_transit, -1, 1)

    if nufft_eps < 1e-14 or f.shape[-1] < 14:
        # second condition for GPU
        f = f[..., None, None, None, :, :]
        f = irfft_mmt_pos(θ, f, num_θ, modes=modes_θ)
    else:
        if len(lines) > 1:
            θ = θ.transpose(0, 4, 1, 2, 3).reshape(lines[0], num_z, -1)
        else:
            θ = θ.transpose(3, 0, 1, 2).reshape(num_z, -1)
        f = nufft1d2r(θ, f, eps=nufft_eps).mT
    f = f.reshape(*lines, -1)

    z = jnp.ravel(
        z + (theta.domain[1] - theta.domain[0]) * jnp.arange(theta.X)[:, None]
    )
    f = CubicSpline(x=z, y=f, axis=-1, check=check).c
    f = jnp.moveaxis(f, (0, 1), (-1, -2))
    assert f.shape == (*lines, num_transit * Y - 1, 4)
    return f, z


def move(f, out=True):
    """Use to move between the following shapes.

    The LHS shape enables the simplest broadcasting so it is used internally,
    while the RHS shape is the returned shape which enables simplest to use API
    for computing various quantities.

    When out is True, goes from left to right. Goes other way when False.

    (num pitch, num ρ, num α, -1) -> (num ρ, num α, num pitch, -1)
    (num pitch,        num α, -1) -> (       num α, num pitch, -1)
    (num pitch,               -1) -> (              num pitch, -1)
    """
    assert f.ndim <= 4
    s, d = (0, -2) if out else (-2, 0)
    return jnp.moveaxis(f, s, d)


def mmt_for_bounce(z, t, c):
    """Matrix multiplication transform.

    Parameters
    ----------
    z : jnp.ndarray
        Shape (num ρ, num α, num pitch, num well, num quad, num ζ modes).
        Vandermonde array.
    t : jnp.ndarray
        Shape (num ρ, num α, num pitch, num well, num quad, num θ modes).
        Vandermonde array.
    c : jnp.ndarray
        Shape (num ρ, 1, num ζ modes, num θ modes).
        Fourier coefficients.

    """
    # Reduce over ζ first since the derivative graph is deeper in θ, and
    # because num ζ modes ~= 2 num θ modes since real fft done over θ.
    return (t * jnp.einsum("...pwqz, ...zt -> ...pwqt", z, c)).real.sum(-1)


def broadcast_for_bounce(pitch_inv):
    """Add axis if necessary.

    Parameters
    ----------
    pitch_inv : jnp.ndarray
        Shape broadcasts with (num ρ, num pitch).

    Returns
    -------
    pitch_inv : jnp.ndarray
        Shape broadcasts with (num ρ, num α, num pitch).

    """
    if jnp.ndim(pitch_inv) == 2:
        pitch_inv = pitch_inv[:, None]
    return pitch_inv


def truncate_rule(Y):
    """Truncation of Chebyshev series to reduce spectral aliasing."""
    return max(1, 7 * Y // 8)


def round_up_rule(Y, NFP, axisymmetric=False):
    """Round Y up to NFP multiple.

    Returns
    -------
    Y : int
        Number of points per toroidal transit.
    num_z : int
        Number of points per field period.

    """
    if axisymmetric:
        assert Y % NFP == 0, "Should set NFP = 1."
        NFP = Y
    num_z = (Y + NFP - 1) // NFP
    return num_z * NFP, num_z


def fieldline_quad_rule(Y):
    """Ensure field line quadrature has reasonable resolution.

    Parameters
    ----------
    Y : int
        Resolution of Chebyshev spectrum of angle over one field period.

    Returns
    -------
    Y : int
        Resolution for Gauss-Legendre quadrature over one field period.

    """
    return max(Y, 8)


def Y_B_rule(Y, NFP, spline=True):
    """Guess Y_B from resolution of Chebyshev spectrum of angle."""
    return (2 * Y * int(np.sqrt(NFP))) if spline else Y


def num_well_rule(num_transit, NFP, Y_B=None):
    """Guess upper bound for number of wells based on spectrum.

    This should be loose enough that it is equivalent to ``num_well=None``,
    but more performant.
    """
    num_well = num_transit * (20 + NFP)
    return num_well if Y_B is None else min(num_well, num_transit * Y_B)


def get_vander(grid, Y, Y_B, NFP):
    """Builds Vandermonde matrices for objectives."""
    assert isinstance(Y, int)
    assert isinstance(Y_B, int)
    assert isinstance(NFP, int)

    Y_trunc = truncate_rule(Y)
    Y_B, num_z = round_up_rule(Y_B, NFP, grid.num_zeta == 1)
    x = jnp.linspace(
        -1, 1, (Y_B // NFP) if (grid.num_zeta == 1) else num_z, endpoint=False
    )
    return {"dct spline": chebvander(x, Y_trunc - 1)}


def _add_lead_axis(cheb, arr):
    """Add leading axis for batching ``cheb`` depending on ``arr.ndim``.

    Input ``arr`` should not have rightmost dimension of cheb that iterates
    coefficients, but may have one leading axis for batching.
    """
    errorif(
        jnp.ndim(arr) > cheb.ndim,
        NotImplementedError,
        msg=f"Only one leading axis for batching is allowed. "
        f"Got {jnp.ndim(arr) - cheb.ndim + 1} leading axes.",
    )
    return cheb if jnp.ndim(arr) < cheb.ndim else cheb[None]


class PiecewiseChebyshevSeries(IOAble):
    """Chebyshev series.

    { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ]

    Parameters
    ----------
    cheb : jnp.ndarray
        Shape (..., X, Y).
        Chebyshev coefficients aₙ(x) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y).
    domain : tuple[float]
        Domain for y coordinates. Default is [-1, 1].

    """

    def __init__(self, cheb, domain=(-1, 1)):
        """Make piecewise series from given Chebyshev coefficients."""
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        self.cheb = jnp.atleast_2d(cheb)
        self.domain = domain

    @property
    def X(self):
        """Number of cuts."""
        return self.cheb.shape[-2]

    @property
    def Y(self):
        """Chebyshev spectral resolution."""
        return self.cheb.shape[-1]

    def stitch(self):
        """Enforce continuity of the piecewise series."""
        # evaluate at left boundary
        f_0 = self.cheb[..., ::2].sum(-1) - self.cheb[..., 1::2].sum(-1)
        # evaluate at right boundary
        f_1 = self.cheb.sum(-1)
        dfx = f_1[..., :-1] - f_0[..., 1:]  # Δf = f(xᵢ, y₁) - f(xᵢ₊₁, y₀)
        self.cheb = self.cheb.at[..., 1:, 0].add(dfx.cumsum(-1))

    def evaluate(self, Y):
        """Evaluate Chebyshev series at Y Chebyshev points.

        Evaluate each function in this set
        { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
        at y points given by the Y Chebyshev points.

        Parameters
        ----------
        Y : int
            Grid resolution in y direction. Preferably power of 2.

        Returns
        -------
        fq : jnp.ndarray
            Shape (..., X, Y)
            Chebyshev series evaluated at Y Chebyshev points.

        """
        warnif(
            Y < self.Y,
            msg="Frequency spectrum of DCT interpolation will be truncated because "
            "the grid resolution is less than the Chebyshev resolution.\n"
            f"Got Y = {Y} < {self.Y} = self.Y.",
        )
        return idct(dct_from_cheb(self.cheb), type=2, n=Y, axis=-1) * Y

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
        return y + z_shift[:, None]

    def _isomorphism_to_C2(self, z):
        """Return coordinates (x, y) ∈ ℂ² isomorphic to z ∈ ℂ.

        Returns index x and minimum value y such that
        z = f(x) + y where f(x) = x * |domain|.

        Parameters
        ----------
        z : jnp.ndarray
            Shape z.shape.

        Returns
        -------
        x_idx, y : tuple[jnp.ndarray]
            Shape z.shape.
            Isomorphic coordinates.

        """
        x_idx, y = jnp.divmod(z - self.domain[0], self.domain[-1] - self.domain[0])
        x_idx = x_idx.astype(int)
        y += self.domain[0]
        return x_idx, y

    def eval1d(self, z, cheb=None, loop=False):
        """Evaluate piecewise Chebyshev series at coordinates z.

        Parameters
        ----------
        z : jnp.ndarray
            Shape (..., *cheb.shape[:-2], z.shape[-1]).
            Coordinates in [self.domain[0], ∞).
            The coordinates z ∈ ℝ are assumed isomorphic to (x, y) ∈ ℝ² where
            ``z//domain`` yields the index into the proper Chebyshev series
            along the second to last axis of ``cheb`` and ``z%domain`` is
            the coordinate value on the domain of that Chebyshev series.
        cheb : jnp.ndarray
            Shape (..., X, Y).
            Chebyshev coefficients to use. If not given, uses ``self.cheb``.
        loop : bool
            Whether to use Clenshaw recursion.
            This is slower on CPU, but it reduces memory of the Jacobian.

        Returns
        -------
        f : jnp.ndarray
            Chebyshev series evaluated at z.

        """
        cheb = _add_lead_axis(setdefault(cheb, self.cheb), z)
        x_idx, y = self._isomorphism_to_C2(z)
        y = bijection_to_disc(y, self.domain[0], self.domain[-1])

        # Recall that the Chebyshev coefficients αₙ for f(z) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x[z]) Tₙ(y[z])
        # are in cheb array whose shape is (..., num cheb series, spectral resolution).

        if not loop or self.Y < 3:
            cheb = jnp.take_along_axis(cheb, x_idx[..., None], axis=-2)
            return idct_mmt(y, cheb)

        def body(i, val):
            c0, c1 = val
            return jnp.take_along_axis(cheb[..., -i], x_idx, axis=-1) - c1, c0 + c1 * y2

        y2 = 2 * y
        c0 = jnp.take_along_axis(cheb[..., -2], x_idx, axis=-1)
        c1 = jnp.take_along_axis(cheb[..., -1], x_idx, axis=-1)
        c0, c1 = fori_loop(3, self.Y + 1, body, (c0, c1))
        return c0 + c1 * y

    def intersect2d(self, k=0.0, *, eps=_eps):
        """Coordinates yᵢ such that f(x, yᵢ) = k(x).

        Parameters
        ----------
        k : jnp.ndarray
            Shape must broadcast with (..., *cheb.shape[:-1]).
            Specify to find solutions yᵢ to f(x, yᵢ) = k(x). Default 0.
        eps : float
            Absolute tolerance with which to consider value as zero.

        Returns
        -------
        y : jnp.ndarray
            Shape (..., *cheb.shape[:-1], Y - 1).
            Solutions yᵢ of f(x, yᵢ) = k(x), in ascending order.
        mask : jnp.ndarray
            Shape y.shape.
            Boolean array into ``y`` indicating whether element is an intersect.
        df_dy : jnp.ndarray
            Shape y.shape.
            Sign of ∂f/∂y (x, yᵢ).

        """
        c = _subtract_first(_add_lead_axis(self.cheb, k), k)
        # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
        y = _chebroots_vec(c)
        assert y.shape == (*c.shape[:-1], self.Y - 1)

        # Intersects must satisfy y ∈ [-1, 1].
        # Pick sentinel such that only distinct roots are considered intersects.
        y = _filter_distinct(y, sentinel=-2.0, eps=eps)
        mask = (jnp.abs(y.imag) <= eps) & (jnp.abs(y.real) < 1.0)
        # Ensure y ∈ (-1, 1), i.e. where arccos is differentiable.
        y = jnp.where(mask, y.real, 0.0)

        n = jnp.arange(self.Y)
        #      ∂f/∂y =      ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n Uₙ₋₁(y)
        # sign ∂f/∂y = sign ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n sin(n arcos y)
        df_dy = jnp.sign(
            jnp.einsum(
                "...yn, ...n",
                n * jnp.sin(n * jnp.arccos(y)[..., None]),
                self.cheb,
            )
        )
        y = bijection_from_disc(y, self.domain[0], self.domain[-1])
        return y, mask, df_dy

    def intersect1d(self, k=0.0, num_intersect=None):
        """Coordinates z(x, yᵢ) such that fₓ(yᵢ) = k for every x.

        Examples
        --------
        In ``Bounce2D.points``, the labels x, y, z, f, k are
          * z = ζ = (ϑ−α)/ι̅-ω ∈ ℝ
          * y = ζ mod (2π/NFP)
          * x = α mod (2π)
          * f = ‖B‖
          * k = 1/λ

        Parameters
        ----------
        k : jnp.ndarray
            Shape must broadcast with (..., *cheb.shape[:-2]).
            Specify to find solutions yᵢ to fₓ(yᵢ) = k. Default 0.
        num_intersect : int or None
            Specify to return the first ``num_intersect`` intersects.
            This is useful if ``num_intersect`` tightly bounds the actual number.

            If not specified, then all intersects are returned. If there were fewer
            intersects detected than the size of the last axis of the returned arrays,
            then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape broadcasts with (..., *self.cheb.shape[:-2], num intersect).
            Tuple of length two (z1, z2) of coordinates of intersects.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of f.

        """
        errorif(
            self.Y < 2,
            NotImplementedError,
            "This method requires a Chebyshev spectral resolution of Y > 1, "
            f"but got Y = {self.Y}.",
        )

        # Add axis to use same k over all Chebyshev series of the piecewise spline.
        y, mask, df_dy = self.intersect2d(jnp.atleast_1d(k)[..., None])
        # Flatten so that last axis enumerates intersects along the piecewise spline.
        y = flatten_mat(self._isomorphism_to_C1(y))
        mask = flatten_mat(mask)
        df_dy = flatten_mat(df_dy)

        # Note for bounce point applications:
        # We ignore the degenerate edge case where the boundary shared by adjacent
        # polynomials is a left intersection because the subset of pitch values
        # that generate this edge case has zero measure. By ignoring this, for those
        # subset of pitch values the integrations will be done in the hypograph of
        # |B|, which will yield zero. If in future decide to not ignore this, note
        # the solution is to
        # 1. disqualify intersects within ``_eps`` from ``domain[-1]``
        # 2. Evaluate sign in ``intersect2d`` at boundary of Chebyshev polynomial
        #    using Chebyshev identities rather than arccos(-1) or arccos(1) which
        #    are not differentiable.
        z1 = (df_dy <= 0) & mask
        z2 = (df_dy >= 0) & _in_epigraph_and(mask, df_dy)

        sentinel = self.domain[0] - 1.0
        z1 = take_mask(y, z1, size=num_intersect, fill_value=sentinel)
        z2 = take_mask(y, z2, size=num_intersect, fill_value=sentinel)

        mask = (z1 > sentinel) & (z2 > sentinel)
        # Set to zero so integration is over set of measure zero
        # and basis functions are faster to evaluate in downstream routines.
        z1 = jnp.where(mask, z1, 0.0)
        z2 = jnp.where(mask, z2, 0.0)
        return z1, z2

    def _check_shape(self, z1, z2, k):
        """Return shapes that broadcast with (k.shape[0], *self.cheb.shape[:-2], W)."""
        assert z1.shape == z2.shape
        # Ensure pitch batch dim exists and add back dim to broadcast with wells.
        k = atleast_nd(self.cheb.ndim - 1, k)[..., None]
        # Same but back dim already exists.
        z1 = atleast_nd(self.cheb.ndim, z1)
        z2 = atleast_nd(self.cheb.ndim, z2)
        # Cheb has shape    (..., X, Y) and others
        #     have shape (K, ..., W)
        assert z1.ndim == z2.ndim == k.ndim == self.cheb.ndim
        return z1, z2, k

    def check_intersect1d(self, z1, z2, k, plot=True, **kwargs):
        """Check that intersects are computed correctly.

        Parameters
        ----------
        z1, z2 : jnp.ndarray
            Shape must broadcast with (*self.cheb.shape[:-2], W).
            Coordinates of intersects.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of f.
        k : jnp.ndarray
            Shape must broadcast with self.cheb.shape[:-2].
            k such that fₓ(yᵢ) = k.
        plot : bool
            Whether to plot the piecewise spline and intersects for the given ``k``.
            For the plotting labels of ρ(l), α(m), it is assumed that the axis that
            enumerates the index l preceds the axis that enumerates the index m.
        kwargs : dict
            Keyword arguments into ``self.plot``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        kwargs.setdefault("title", r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$")
        title = kwargs.pop("title")
        plots = []

        z1, z2, k = self._check_shape(z1, z2, k)
        mask = (z1 - z2) != 0.0
        z1 = jnp.where(mask, z1, jnp.nan)
        z2 = jnp.where(mask, z2, jnp.nan)

        err_1 = jnp.any(z1 > z2, axis=-1)
        err_2 = jnp.any(z1[..., 1:] < z2[..., :-1], axis=-1)
        f_midpoint = self.eval1d((z1 + z2) / 2)
        eps = kwargs.pop("eps", jnp.finfo(jnp.array(1.0).dtype).eps * 10)
        err_3 = jnp.any(f_midpoint > k + eps, axis=-1)
        if not (plot or jnp.any(err_1 | err_2 | err_3)):
            return plots

        cheb = atleast_nd(3, self.cheb)
        mask, z1, z2, f_midpoint = map(atleast_3d_mid, (mask, z1, z2, f_midpoint))
        err_1, err_2, err_3 = map(atleast_2d_end, (err_1, err_2, err_3))

        for l in np.ndindex(cheb.shape[:-2]):
            for p in range(k.shape[0]):
                idx = (p, *l)
                if not (err_1[idx] or err_2[idx] or err_3[idx]):
                    continue
                _z1 = z1[idx][mask[idx]]
                _z2 = z2[idx][mask[idx]]
                if plot:
                    self.plot1d(
                        cheb=cheb[l],
                        z1=_z1,
                        z2=_z2,
                        k=k[idx],
                        title=title
                        + rf" on field line $\rho(l)$, $\alpha(m)$, $(l,m)=${l}",
                        **kwargs,
                    )
                print("      z1    |    z2")
                print(jnp.column_stack([_z1, _z2]))
                assert not err_1[idx], "Intersects have an inversion.\n"
                assert not err_2[idx], "Detected discontinuity.\n"
                assert not err_3[idx], (
                    f"Detected f = {f_midpoint[idx][mask[idx]]} > {k[idx] + _eps} = k"
                    "in well, implying the straight line path between z1 and z2 is in"
                    "hypograph(f). Increase spectral resolution.\n"
                )
            idx = (slice(None), *l)
            if plot:
                plots.append(
                    self.plot1d(
                        cheb=cheb[l],
                        z1=z1[idx],
                        z2=z2[idx],
                        k=k[idx],
                        title=title,
                        **kwargs,
                    )
                )
        return plots

    def plot1d(
        self,
        cheb,
        num=5000,
        z1=None,
        z2=None,
        k=None,
        *,
        k_transparency=0.5,
        klabel=r"$k$",
        title=r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$",
        hlabel=r"$z$",
        vlabel=r"$f$",
        show=True,
        include_legend=True,
        return_legend=False,
        legend_kwargs=None,
        **kwargs,
    ):
        """Plot the piecewise Chebyshev series ``cheb``.

        Parameters
        ----------
        cheb : jnp.ndarray
            Shape (X, Y).
            Piecewise Chebyshev series f.
        num : int
            Number of points to evaluate ``cheb`` for plot.
        z1 : jnp.ndarray
            Shape (k.shape[0], W).
            Optional, intersects with ∂f/∂y <= 0.
        z2 : jnp.ndarray
            Shape (k.shape[0], W).
            Optional, intersects with ∂f/∂y >= 0.
        k : jnp.ndarray
            Shape (k.shape[0], ).
            Optional, k such that fₓ(yᵢ) = k.
        k_transparency : float
            Transparency of pitch lines.
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
        include_legend : bool
            Whether to plot the legend. Default is true.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))

        legend = {}
        z = jnp.linspace(
            start=self.domain[0],
            stop=self.domain[0] + (self.domain[-1] - self.domain[0]) * self.X,
            num=num,
        )
        _add2legend(legend, ax.plot(z, self.eval1d(z, cheb), label=vlabel, **kwargs))
        _plot_intersect(
            ax=ax,
            legend=legend,
            z1=z1,
            z2=z2,
            k=k,
            k_transparency=k_transparency,
            klabel=klabel,
            hlabel=hlabel,
            **kwargs,
        )
        ax.set_xlabel(hlabel)
        ax.set_ylabel(vlabel)
        ax.set_title(title)

        if include_legend:
            if legend_kwargs is None:
                legend_kwargs = dict(loc="lower right")
            ax.legend(legend.values(), legend.keys(), **legend_kwargs)

        if show:
            plt.show()
            plt.close()
        return (fig, ax, legend) if return_legend else (fig, ax)
