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
    PiecewiseChebyshevSeries,
    cheb_from_dct,
    cheb_pts,
    epigraph_and,
    idct_mmt,
    ifft_mmt,
    irfft_mmt_pos,
    take_mask,
)
from interpax_fft._series import _add2legend, _plot_intersect
from matplotlib import pyplot as plt

from desc.backend import dct, ifft, jax, jnp
from desc.integrals._interp_utils import (
    _JF_BUG,
    _root_eps,
    chebder,
    nufft1d2r,
    nufft2d2r,
    poly_val,
    polyroot_vec,
)
from desc.integrals.quad_utils import bijection_from_disc
from desc.utils import atleast_nd, flatten_mat, setdefault


def _bounce_points(
    pitch_inv, knots, B, num_well=-1, *, sentinel=-1.0, return_mask=False
):
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
        Shape (..., N - 1, 4).
        Polynomial coefficients of the spline of B in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    num_well : int
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
    sentinel : float
        Sentinel value which should be less ζ coordinate of all bounce points,
        which can be guaranteed by choosing branch cut for α appropriately.
        Default is -1.
    return_mask : bool
        Whether to return the mask ``z1<z2``. Default is ``False``.

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
    if num_well < 0 or num_well > B.shape[-2]:
        # The number of interior minima for C¹ continuous cubic spline must be < N,
        # and every minima must be a simple root.
        num_well = B.shape[-2]

    B = B[..., None, :, :]
    intersect = polyroot_vec(
        c=B,
        k=jnp.atleast_1d(pitch_inv)[..., None],
        a_min=jnp.array([0.0]),
        a_max=jnp.diff(knots),
        sort=True,
        sentinel=-1.0,
        distinct=True,
    )
    assert intersect.shape[-2:] == (knots.size - 1, B.shape[-1] - 1)

    dB_dz = flatten_mat(jnp.sign(poly_val(x=intersect, c=B[..., None, :], der=True)))
    # Only consider intersect if it is within knots that bound that polynomial.
    mask = flatten_mat(intersect >= 0)
    z1 = (dB_dz <= 0) & mask
    z2 = (dB_dz >= 0) & epigraph_and(mask, dB_dz)
    del dB_dz

    # Transform out of local power basis expansion.
    intersect = flatten_mat(intersect + knots[:-1, None])
    z1 = take_mask(intersect, z1, size=num_well, fill_value=sentinel)
    z2 = take_mask(intersect, z2, size=num_well, fill_value=sentinel)
    del intersect

    mask = (z1 > sentinel) & (z2 > sentinel)
    # Set to zero so integration is over set of measure zero
    # and basis functions are faster to evaluate in downstream routines.
    z1 = jnp.where(mask, z1, 0.0)
    z2 = jnp.where(mask, z2, 0.0)
    return (z1, z2, mask) if return_mask else (z1, z2)


def _halley(o, pitch_inv, z, mask, nufft_eps, diagnostic=0):
    """Solve for the bounce points using the maps used in quadrature.

    Halley (Schröder second kind) irrational step.

    The bounce parameters Y_B and Y should be high enough that
    initial guess is in basin of attraction for Halley step.

    Parameters
    ----------
    o : Bounce2D
        Object instance.
    pitch_inv : jnp.ndarray
        Shape broadcasts with (num ρ, num α, num pitch).
    z: jnp.ndarray
        Shape (2, num ρ ?, num α, num pitch, num well).
        Bounce points.
    mask : jnp.ndarray
        Shape (num ρ ?, num α, num pitch, num well).
        Subset of points to refine.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.

        Should satisfy ε < εᵢₙ² where εᵢₙ is the error of the input points.
    diagnostic : int
        Positive integer denoting iteration step to print.

    Returns
    -------
    z : jnp.ndarray
        Shape (2, num ρ ?, num α, num pitch, num well).

    """
    t, B = _halley_coefficients(o)

    # calling this method with shapes    (1,  b=2, ρ?,α,   λ, w)
    #                                    (3,    1, ρ?,α,   1, X, Y).
    t, dt, dt2 = o._theta.eval1d(z[None], t[:, None, ..., None, :, :])
    # shapes match z, i.e. (b, ρ ?, α, λ, w)

    if no_nufft(nufft_eps):
        # Halley step is free compared to recomputing the basis.
        z_eff = z if o._num_z > 1 else jnp.zeros((1,) * z.ndim)
        dB_dz, dB_dt, B, dB_dz2, dB_dzdt, dB_dt2 = jnp.einsum(
            "...czt, b...apwz, b...apwt -> cb...apw",
            B,
            jnp.exp(1j * o._modes_z * z_eff[..., None]),
            jnp.exp(1j * o._modes_t * t[..., None]),
            optimize=[(0, 1), (0, 1)],
        ).real
    else:
        dB_dz, dB_dt, B, dB_dz2, dB_dzdt, dB_dt2 = _acrobatics(
            z, t, B, o._NFP, nufft_eps, mask
        )

    f = B - pitch_inv[..., None]
    df = dB_dz + dB_dt * dt
    df2 = dB_dz2 + 2 * dB_dzdt * dt + dB_dt2 * dt**2 + dB_dt * dt2
    df2 = df**2 - 2 * f * df2
    update = 2 * f / (df + jnp.sign(df) * jnp.sqrt(jnp.where(df2 > 0, df2, df**2)))

    del f, df, df2, dB_dz, dB_dt, B, dB_dz2, dB_dzdt, dB_dt2, t, dt, dt2

    if diagnostic:
        jax.debug.print(
            "After {iteration:1d} iteration(s) | "
            "ζ₁₂(w) error mean = {:5.0e} | std. dev. = {:5.0e} | max = {:5.0e}",
            jnp.abs(update).mean(where=mask),
            jnp.abs(update).std(where=mask),
            jnp.abs(update).max(where=mask, initial=-jnp.inf),
            iteration=diagnostic - 1,
            ordered=True,
        )

    return _safe_update(mask, z, update, FENCE=2 * jnp.pi / o._NFP)


def _safe_update(mask, old, update, FENCE):
    """Returns old - update where intervals are preserved and update < FENCE."""
    new = old - update
    return jnp.where(
        mask & (new[0] < new[1]) & (jnp.abs(update) < FENCE),
        new,
        old,
    )


def _halley_coefficients(o, jvp=False):
    """Returns coefficient arrays for the nonlinear solve.

    Parameters
    ----------
    o : Bounce2D
        Object instance.
    jvp : bool
        Whether to return only the coefficients needed for the jvp.

    Returns
    -------
    t, B : tuple[jnp.ndarray]
        Coefficient arrays.
        t, dt, dt2
        dB_dz, dB_dt, B, dB_dz2, dB_dzdt, dB_dt2

    """
    t = [
        o._theta.cheb,
        chebder(o._theta.cheb, scl=o._NFP / jnp.pi, axis=-1, keepdims=True),
    ]
    B = [
        o._c["|B|"] * (1j * o._modes_z)[:, None],
        o._c["|B|"] * (1j * o._modes_t),
    ]

    if not jvp:
        B += [
            o._c["|B|"],
            o._c["|B|"] * (-o._modes_z**2)[:, None],
            o._c["|B|"] * (-o._modes_z[:, None] * o._modes_t),
            o._c["|B|"] * (-o._modes_t**2),
        ]
        t.append(chebder(t[1], scl=o._NFP / jnp.pi, axis=-1, keepdims=True))

    # shape is (# of funs e.g. 2 or 3, ρ ?, α, X, Y)
    t = jnp.stack(t)
    # shape is (ρ ?, # of funs, z modes, t modes)
    B = jnp.concatenate(B, -3)
    return t, B


def _acrobatics(z, t, c, NFP, eps, mask):
    # Some reshape acrobatics required due to frankenstein vectorization of jax-finufft.
    t = t.swapaxes(0, -4)
    swapped_shape = t.shape
    t = flatten_mat(t, 4)  # shape is (ρ, points per ρ surface)
    #                         or just (   points per ρ surface)
    return (
        nufft2d2r(
            flatten_mat(z.swapaxes(0, -4), 4),
            t,
            c,
            (0, 2 * jnp.pi / NFP),
            vec=True,
            eps=eps,
            mask=(
                None
                if _JF_BUG
                else flatten_mat(
                    jnp.broadcast_to(mask[None], (2,) + mask.shape).swapaxes(0, -4), 4
                )
            ),
        )
        .swapaxes(0, -2)  # so that first axis splits into e.g. dB_dz, dB_dt, B
        .reshape((-1,) + swapped_shape)  # then shape is (-1, ρ ?, b, α, λ, w)
        .swapaxes(1, -4)  # recover shape (-1, b, ρ ?, α, λ, w)
    )


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def bounce_points(o, pitch_inv, num_well):
    """Bounce points then iterative solve, with regularized ift jvp.

    Parameters
    ----------
    o : Bounce2D
        Object instance.
    pitch_inv : jnp.ndarray
        Shape broadcasts with (num ρ, num α, num pitch).
    num_well : int
        The usual suspect.

    """
    *z, mask = _bounce_points(
        pitch_inv, o._c["knots"], o._B, num_well, return_mask=True
    )
    z = jnp.stack(z)
    return _halley(
        o, pitch_inv, z, mask, nufft_eps=min(o._nufft_eps, 1e-10), diagnostic=0
    )


@bounce_points.defjvp
def bounce_points_jvp(num_well, primals, tangents):
    """Implicit function theorem with regularization.

    Regularization used to smooth the discretized system so that it recognizes
    any non-differentiable sample it has observed actually has zero measure in
    the continuous system.

    References
    ----------
    See supplementary information in publications/unalmis2025.

    """
    # Cannot mix primals and tangents; see https://github.com/jax-ml/jax/issues/36319.

    o, p = primals
    do, dp = tangents

    z = bounce_points(o, p, num_well)

    mask = z[0] < z[1]
    nufft_eps = min(o._nufft_eps, 1e-10)
    t, dB_dz = _halley_coefficients(o, jvp=True)

    # calling this method with shapes    (1,  b=2, ρ?,α,   λ, w)
    #                                    (2,    1, ρ?,α,   1, X, Y).
    t, dt_dz = o._theta.eval1d(z[None], t[:, None, ..., None, :, :])
    dt_do = o._theta.eval1d(z, do._theta.cheb[None, ..., None, :, :])
    # shapes match z, i.e. (b, ρ ?, α, λ, w)

    if no_nufft(nufft_eps):
        z_eff = jnp.exp(
            1j
            * o._modes_z
            * (z if o._num_z > 1 else jnp.zeros((1,) * z.ndim))[..., None]
        )
        t = jnp.exp(1j * o._modes_t * t[..., None])

        dB_dz, dB_dt = jnp.einsum(
            "...czt, b...apwz, b...apwt -> cb...apw",
            dB_dz,
            z_eff,
            t,
            optimize=[(0, 1), (0, 1)],
        ).real
        dB_do = jnp.einsum(
            "...zt, b...apwz, b...apwt -> b...apw",
            do._c["|B|"].squeeze(-3),
            z_eff,
            t,
            optimize=[(0, 1), (0, 1)],
        ).real

        del z_eff, t
    else:
        dB_dz, dB_dt = _acrobatics(z, t, dB_dz, o._NFP, nufft_eps, mask)
        (dB_do,) = _acrobatics(z, t, do._c["|B|"], o._NFP, nufft_eps, mask)

    # chain rule to move from (∂/∂ζ)|ρ,θ to (∂/∂ζ)|ρ,a
    dB_dz += dB_dt * dt_dz
    dB_do += dB_dt * dt_do

    regularization = _root_eps()
    dB_dz = jnp.where(
        jnp.abs(dB_dz) > regularization,
        dB_dz,
        dB_dz + jnp.copysign(regularization, dB_dz.real),
    )
    dz = jnp.where(mask, (dp[..., None] - dB_do) / dB_dz, 0.0)

    return z, dz


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


def get_mins(knots, B, num_mins=-1, fill_value=0.0):
    """Return minima of (z*, B(z*)) within interval defined by knots.

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (N, ).
        ζ coordinates of spline knots. Must be strictly increasing.
    B : jnp.ndarray
        Shape (..., N - 1, 4).
        Polynomial coefficients of the spline of B in local power basis.
        Last axis enumerates the coefficients of power series. Second to
        last axis enumerates the polynomials that compose a particular spline.
    num_mins : int
        Number of minima to return. Otherwise returns maximum possible.
    fill_value : float
        If there were less than ``num_mins`` minima detected, then the result
        is padded with ``fill_value``.

    Returns
    -------
    mins, B_mins : jnp.ndarray
        Shape (..., num mins).
        First array enumerates z*. Second array enumerates B(z*).
        Sorting order of extrema is arbitrary.

    """
    if num_mins < 0 or num_mins > B.shape[-2]:
        # The number of interior minima for C¹ continuous cubic spline must be < N,
        # and every minima must be a simple root.
        num_mins = B.shape[-2]

    b = B[..., :-1] * jnp.arange(B.shape[-1] - 1, 0, -1)
    mins = polyroot_vec(
        c=b,
        a_min=jnp.array([0.0]),
        a_max=jnp.diff(knots),
        sentinel=0.0,
        distinct=False,
    )
    b = flatten_mat((poly_val(x=mins, c=b[..., None, :], der=True) > 0) & (mins > 0))
    mins = flatten_mat(
        jnp.stack(
            [
                # Transform out of local power basis expansion.
                mins + knots[:-1, None],
                poly_val(x=mins, c=B[..., None, :]),
            ]
        )
    )
    mins, b = take_mask(mins, b, size=num_mins, fill_value=fill_value)
    assert mins.shape[-1] == num_mins
    return mins, b


def argmin(z1, z2, f, mins, B_mins):
    """Returns f at argmin of B between ``z1`` and ``z2``.

    Let E(w) = {ζ ∣ ζ₁(w) < ζ < ζ₂(w)} and A(w) ∈ argmin_E(w) B.
    Given the minima of B and f interpolated to those minima,
    returns {f ∘ A(w)}.

    Parameters
    ----------
    z1, z2 : jnp.ndarray
        Shape (..., num pitch, num well).
        Boundaries to detect argmin between.
        ``z1`` (``z2``) stores left (right) boundaries.
    f : jnp.ndarray
        Function interpolated to ``mins``.
        Shape (..., num mins).
    mins : jnp.ndarray
        Minima of B.
        Shape ``f.shape``.
    B_mins : jnp.ndarray
        B interpolated to ``mins``.
        Shape ``f.shape``.

    Returns
    -------
    f : jnp.ndarray
        Shape (..., num pitch, num well).
        Returns f at argmin of B between ``z1`` and ``z2``.

    """
    assert z1.ndim > 1 and z2.ndim > 1
    assert f.shape == mins.shape == B_mins.shape
    # We can use the non-differentiable argmin because we actually want the gradients
    # to accumulate through only the minimum since we are differentiating how our
    # physics objective changes wrt equilibrium perturbations not wrt which of the
    # extrema get interpolated to.
    where = jnp.argmin(
        jnp.where(
            (z1[..., None] < mins[..., None, None, :])
            & (mins[..., None, None, :] < z2[..., None]),
            B_mins[..., None, None, :],
            jnp.inf,
        ),
        axis=-1,
        keepdims=True,
    )
    return jnp.take_along_axis(f[..., None, None, :], where, axis=-1).squeeze(-1)


def get_alphas(alpha, iota, num_field_periods, NFP):
    """Get set of field line poloidal coordinates {Aᵢ | Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)}.

    Parameters
    ----------
    alpha : jnp.ndarray
        Shape (num α, ).
        Starting field line poloidal labels {αᵢ₀}.
    iota : jnp.ndarray
        Shape (num ρ, ).
        Rotational transform normalized by 2π.
    num_field_periods : int
        Number of field periods to follow field line.
    NFP: int
        Number of field periods per toroidal transit.

    Returns
    -------
    alphas : jnp.ndarray
        Shape (num α, num ρ, num field periods).
        Set of field line poloidal coordinates {Aᵢ | Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)}.

    """
    alpha = alpha[:, None, None]
    iota = iota[:, None]
    return alpha + iota * (2 * jnp.pi / NFP) * jnp.arange(num_field_periods)


def theta_on_fieldlines(angle, iota, alpha, num_field_periods, NFP, *, X_min=24):
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
    num_field_periods : int
        Number of field periods to follow field line.
    NFP : int
        Number of field periods per toroidal transit.
    X_min : int
        See notes section. This parameter should never be changed.
        It is included in the function signature for code optics only.
        It is the number below which we short-circuit convergence to enforce
        continuity by removing a discontinuity which is near machine precision
        due to exponential convergence. This is not a hack; it has rigorous
        mathematical justification regardless of the size of the removed
        discontinuity, and does not bias the output beyond that of more
        floating point operations in finite-precision.

    Returns
    -------
    theta : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line ``α[i]``. Each Chebyshev series approximates
        θ over one field period. ``theta.cheb`` broadcasts with
        shape (num ρ, num α, num field periods, max(1,7Y//8)).

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

    """
    X = angle.shape[-2]
    Y = truncate_rule(angle.shape[-1])
    num_alpha = alpha.size
    domain = (0, 2 * jnp.pi / NFP)

    # peeling off field lines
    alpha = get_alphas(alpha, iota, num_field_periods, NFP)
    if angle.ndim == 2:
        alpha = alpha.squeeze(1)

    # Mod early for speed and conditioning
    # (since this avoids modding on more points later and keeps θ bounded).
    alpha %= 2 * jnp.pi

    delta = (
        FourierChebyshevSeries(angle, domain, truncate=Y)
        .compute_cheb(alpha)
        .swapaxes(0, -3)
    )
    alpha = alpha.swapaxes(0, -2)
    delta = delta.at[..., 0].add(alpha)  # This is now θ = α + δ.
    assert delta.shape == (*angle.shape[:-2], num_alpha, num_field_periods, Y)

    if X < X_min:
        # This is needed as our algorithm assumes continuity of |B| along field
        # lines when gathering bounce points. This is always true physically.
        delta = PiecewiseChebyshevSeries.stitch(delta)

    return PiecewiseChebyshevSeries(delta, domain)


def fast_chebyshev(theta, f, Y, modes_t, modes_z, *, vander=None):
    """Compute Chebyshev approximation of ``f`` on field lines using fast transforms.

    Parameters
    ----------
    theta : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line αᵢ. Each Chebyshev series approximates
        θ over one field period. ``theta.cheb`` should broadcast with
        shape (num ρ, num α, num field periods, theta.Y).
    f : jnp.ndarray
        Shape broadcasts with (num ρ, 1, modes_z.size, modes_t.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Chebyshev spectral resolution for ``f`` over a field period.
        Preferably power of 2.
    modes_t : jnp.ndarray
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
        ``f`` over one field period. ``f.cheb`` broadcasts with
        shape (num ρ, num α, num field periods, Y).

    """
    if f.shape[-2] == 1:  # axisymmetric
        vander = None
        z_eff = jnp.zeros((1, 1))
    elif vander is None:
        z_eff = cheb_pts(Y, theta.domain)[:, None]
    else:
        z_eff = None

    # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
    # compute a set of 2D Fourier series on non-uniform tensor product grids of size
    # |𝛉|×|𝛇| where |𝛉| = num α × num field periods × Y/z_eff and |𝛇| = z_eff.
    # Partial summation is more efficient than direct evaluation since
    # mn|𝛉||𝛇| > mn|𝛇| + m|𝛉||𝛇| i.e. when n|𝛉| > n + |𝛉|.

    f = ifft_mmt(
        z_eff,
        f,
        theta.domain,
        axis=-2,
        modes=modes_z,
        vander=vander,
    )[..., None, None, :, :]
    f = irfft_mmt_pos(theta.evaluate(Y), f, n=jnp.nan, modes=modes_t)
    f = cheb_from_dct(dct(f, type=2, axis=-1) / Y)
    f = PiecewiseChebyshevSeries(f, theta.domain)
    assert f.cheb.shape == (*theta.cheb.shape[:-1], Y)
    return f


def fast_cubic_spline(
    theta,
    f,
    Y,
    modes_t,
    modes_z,
    nufft_eps=1e-6,
    *,
    vander_t=None,
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
        θ over one field period. ``theta.cheb`` should broadcast with
        shape (num ρ, num α, num field periods, theta.Y).
    f : jnp.ndarray
        Shape broadcasts with (num ρ, 1, modes_z.size, modes_t.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Number of knots per field period to interpolate ``f``.
    modes_t : jnp.ndarray
        Real FFT Fourier modes in poloidal direction.
    modes_z : jnp.ndarray
        FFT Fourier modes in toroidal direction.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
    vander_t : jnp.ndarray
        Precomputed transform matrix.
    vander_z : jnp.ndarray
        Precomputed transform matrix.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    Returns
    -------
    f : jnp.ndarray
        Shape broadcasts with (num ρ, num α, num field periods * Y - 1, 4).
        Polynomial coefficients of the spline of f in local power basis.
        Last axis enumerates the coefficients of power series. For a polynomial
        given by ∑ᵢⁿ cᵢ xⁱ, coefficient cᵢ is stored at ``f[...,n-i]``.
        Second to last axis enumerates the polynomials that compose a particular
        spline.
    knots : jnp.ndarray
        Shape (num field periods * Y).
        Knots of spline ``f``.

    """
    x = jnp.linspace(-1, 1, Y, endpoint=False)
    z = bijection_from_disc(x, *theta.domain)
    axisymmetric = f.shape[-2] == 1
    z_eff = 1 if axisymmetric else Y

    # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
    # compute a set of 2D Fourier series on uniform (non-uniform) in ζ (θ)
    # tensor product grids of size
    # |𝛉|×|𝛇| where |𝛉| = num α × num field periods × Y/z_eff and |𝛇| = z_eff.
    # Partial summation via FFT is more efficient than direct evaluation since
    # mn|𝛉||𝛇| > m log(|𝛇|) |𝛇| + m|𝛉||𝛇| i.e. when n|𝛉| > log|𝛇| + |𝛉|.

    if z_eff >= f.shape[-2]:
        f = f.squeeze(-3)
        p = z_eff - f.shape[-2]
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
    # f shape is (..., z_eff, modes_t.size)

    lines = theta.cheb.shape[:-2]  # (..., num α)
    num_field_periods = theta.X

    # θ at uniform ζ on field lines
    t = idct_mmt(x, theta.cheb[..., None, :], vander=vander_t)
    assert t.shape == (*lines, num_field_periods, Y)

    if no_nufft(nufft_eps) or f.shape[-1] <= 16:
        f = f[..., None, None, :, :]
        f = irfft_mmt_pos(t, f, n=jnp.nan, modes=modes_t)
        assert f.shape == t.shape
    else:
        if axisymmetric:
            t = t.reshape(*lines, -1, z_eff)
        if len(lines) > 1:  # then lines is (num ρ, num α)
            t = t.transpose(0, 3, 1, 2).reshape(lines[0], z_eff, -1)
        else:
            t = t.transpose(2, 0, 1).reshape(z_eff, -1)
        # t shape is (..., z_eff, num α × num field periods × Y/z_eff)
        f = nufft1d2r(t, f, eps=nufft_eps).mT
        # f shape is (..., num α × num field periods × Y/z_eff, z_eff)
    f = f.reshape(*lines, -1)

    z = jnp.ravel(
        z + (theta.domain[1] - theta.domain[0]) * jnp.arange(num_field_periods)[:, None]
    )
    f = CubicSpline(x=z, y=f, axis=-1, check=check).c
    f = jnp.moveaxis(f, (0, 1), (-1, -2))
    assert f.shape == (*lines, num_field_periods * Y - 1, 4)
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
    pitch_inv = jnp.atleast_1d(pitch_inv)
    if pitch_inv.ndim == 2:
        pitch_inv = pitch_inv[:, None]
    return pitch_inv


def truncate_rule(Y):
    """Truncation of Chebyshev series to reduce spectral aliasing.

    The truncation will remove aliasing error at the shortest wavelengths
    where the signal to noise ratio is lowest.
    We truncate with a 7/8 rule in the toroidal direction so that the Lebesgue
    constant is ~ (4/π²) log(Y) when the data is corrupted by ≤ 10⁻⁸ noise from
    the inexact Newton solve. The Lebesgue constant discussed here is the one in
    L Mason, J.C. & Handscomb, David C. 2002 Chebyshev Polynomials, chapter 5.
    This is useful since we evaluate the series on a much denser grid than the
    Newton solve grid; and therefore, prefer all discretization error is from
    the error of the projection rather than the interpolation.

    """
    return max(1, 7 * Y // 8)


def no_nufft(nufft_eps):
    """True if nuffts should not be used."""
    return nufft_eps < 1e-14
