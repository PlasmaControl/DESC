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
from orthax.chebyshev import chebvander

from desc.backend import dct, ifft, jax, jnp
from desc.integrals._interp_utils import (
    _eps,
    chebder,
    nufft1d2r,
    nufft2d2r,
    poly_val,
    polyroot_vec,
)
from desc.integrals.quad_utils import bijection_from_disc
from desc.utils import atleast_nd, flatten_mat, setdefault

_sentinel = -1e5


def bounce_points(pitch_inv, knots, B, num_well=-1):
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
    z1, z2, mask : tuple[jnp.ndarray]
        Shape (..., num pitch, num well).
        ζ coordinates of bounce points. The points are ordered and grouped such
        that the straight line path between ``z1`` and ``z2`` resides in the
        epigraph of B.

        If there were less than ``num_well`` wells detected along a field line,
        then the last axis, which enumerates bounce points for a particular field
        line and pitch, is padded with zero.

    """
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

    # Transform out of local power basis expansion.
    intersect = flatten_mat(intersect + knots[:-1, None])
    z1 = take_mask(intersect, z1, size=num_well, fill_value=_sentinel)
    z2 = take_mask(intersect, z2, size=num_well, fill_value=_sentinel)

    mask = (z1 > _sentinel) & (z2 > _sentinel)
    # Set to zero so integration is over set of measure zero
    # and basis functions are faster to evaluate in downstream routines.
    z1 = jnp.where(mask, z1, 0.0)
    z2 = jnp.where(mask, z2, 0.0)
    return z1, z2, mask


def _newton(o, pitch_inv, z1, z2, mask, nufft_eps=1e-10):
    """Newton step using maps used in the quadrature.

    An error of ε in a bounce point manifests
      * 𝒪(ε¹ᐧ⁵) error in bounce integrals with (v_∥)¹.
      * 𝒪(ε⁰ᐧ⁵) error in bounce integrals with (v_∥)⁻¹.

    Parameters
    ----------
    o : Bounce2D
        Object instance.
    pitch_inv : jnp.ndarray
        Shape broadcasts with (num ρ, num α, num pitch).
    z1, z2 : tuple[jnp.ndarray]
        Shape (num ρ, num α, num pitch, num well).
    mask : jnp.ndarray
        Shape (num ρ, num α, num pitch, num well).
        Subset of points to refine.
    nufft_eps : float
        Desired error ε of the bounce points.

    Returns
    -------
    z1, z2 : tuple[jnp.ndarray]
        Shape (num ρ, num α, num pitch, num well).

    """
    shape = (*z1.shape[:-2], 2, *z1.shape[-2:])

    z = flatten_mat(jnp.stack((z1, z2), axis=-3), 3)
    t, dt_dz = o._theta.eval1d(
        z[None],
        jnp.stack(
            [
                o._theta.cheb,
                chebder(o._theta.cheb, scl=o._NFP / jnp.pi, axis=-1, keepdims=True),
            ]
        ),
    )
    dt_dz = dt_dz.reshape(shape)
    t = flatten_mat(t)
    z = flatten_mat(z)

    B = nufft2d2r(
        z,
        t,
        jnp.concatenate(
            [
                o._c["|B|"],
                o._c["|B|"] * (1j * o._modes_z)[:, None],
                o._c["|B|"] * (1j * o._modes_t),
            ],
            -3,
        ),
        (0, 2 * jnp.pi / o._NFP),
        vec=True,
        eps=nufft_eps,
        mask=flatten_mat(jnp.broadcast_to(mask[..., None, :, :], shape), 4),
    )
    B, dB_dz, dB_dt = (
        B.reshape(3, *shape)
        if B.ndim == 2
        # reshape before swap to avoid memory copy
        else B.reshape(shape[0], 3, *shape[1:]).swapaxes(0, 1)
    )
    z = z.reshape(shape)

    dz = (B - pitch_inv[..., None, :, None]) / (dB_dz + dB_dt * dt_dz)
    Z = z - dz
    mask = mask & (Z[..., 0, :, :] < Z[..., 1, :, :])  # Deny interval inversion.
    mask = mask[..., None, :, :] & (jnp.abs(dz) < 1e-1)  # Deny large updates.
    z = jnp.where(mask, Z, z)

    return z[..., 0, :, :], z[..., 1, :, :]


@partial(jax.custom_jvp, nondiff_argnames=("num_well", "nufft_eps"))
def regular_points(o, pitch_inv, num_well, nufft_eps):
    """Bounce points then newton, with regularized jvp."""
    return _newton(
        o,
        pitch_inv,
        *bounce_points(pitch_inv, o._c["knots"], o._c["B(z)"], num_well),
        nufft_eps,
    )


@regular_points.defjvp
def regular_points_jvp(num_well, nufft_eps, primals, tangents):
    """Implicit function theorem with regularization.

    Regularization used to smooth the discretized system so that it recognizes
    any non-differentiable sample it has observed actually has zero measure in
    the continuous system.

    References
    ----------
    Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
    and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    """
    # Cannot mix primals and tangents; see https://github.com/jax-ml/jax/issues/36319.

    o, p = primals
    do, dp = tangents

    z1, z2 = regular_points(o, p, num_well, nufft_eps)

    shape = (*z1.shape[:-2], 2, *z1.shape[-2:])

    z = flatten_mat(jnp.stack((z1, z2), axis=-3), 3)
    t, dt_dz = o._theta.eval1d(
        z[None],
        jnp.stack(
            [
                o._theta.cheb,
                chebder(o._theta.cheb, scl=o._NFP / jnp.pi, axis=-1, keepdims=True),
            ]
        ),
    )
    dt_do = o._theta.eval1d(z, do._theta.cheb).reshape(shape)
    dt_dz = dt_dz.reshape(shape)
    t = flatten_mat(t)
    z = flatten_mat(z)

    mask = (z1 < z2)[..., None, :, :]

    dB_dz = nufft2d2r(
        z,
        t,
        jnp.concatenate(
            [o._c["|B|"] * (1j * o._modes_z)[:, None], o._c["|B|"] * (1j * o._modes_t)],
            -3,
        ),
        (0, 2 * jnp.pi / o._NFP),
        vec=True,
        eps=nufft_eps,
        mask=flatten_mat(jnp.broadcast_to(mask, shape), 4),
    )
    dB_do = nufft2d2r(
        z,
        t,
        do._c["|B|"].squeeze(-3),
        (0, 2 * jnp.pi / o._NFP),
        eps=nufft_eps,
        mask=flatten_mat(jnp.broadcast_to(mask, shape), 4),
    ).reshape(shape)

    dB_dz, dB_dt = (
        dB_dz.reshape(2, *shape)
        if dB_dz.ndim == 2
        # reshape before swap to avoid memory copy
        else dB_dz.reshape(shape[0], 2, *shape[1:]).swapaxes(0, 1)
    )

    # chain rule to move from (∂/∂ζ)|ρ,θ to (∂/∂ζ)|ρ,a
    dB_dz += dB_dt * dt_dz
    dB_do += dB_dt * dt_do

    dB_dz = jnp.where(
        jnp.abs(dB_dz) > _eps,
        dB_dz,
        dB_dz + jnp.copysign(_eps, dB_dz.real),
    )
    dz12 = jnp.where(mask, (dp[..., None, :, None] - dB_do) / dB_dz, 0.0)

    return (z1, z2), (dz12[..., 0, :, :], dz12[..., 1, :, :])


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
    # TODO: should eventually be a passable argument, we should not expect users to
    # potentially modify source code to use the bounce integrals if this assert ever
    # gets triggered
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
                this_title = (
                    title
                    + rf" on field line $(\rho_{{l={lm[0]}}}, \alpha_{{m={lm[1]}}})$"
                )
                plot_ppoly(
                    ppoly=ppoly,
                    z1=_z1,
                    z2=_z2,
                    k=pitch_inv[idx],
                    title=this_title,
                    **kwargs,
                )

            print("      z1    |    z2")
            print(jnp.column_stack([_z1, _z2]))
            assert not err_1[idx], "Intersects have an inversion.\n"
            assert not err_2[idx], "Detected discontinuity.\n"
            assert not err_3, (
                f"Detected |B| = {B_midpoint[mask[idx]]} > {pitch_inv[idx] + eps} "
                "= 1/λ in well, implying the straight line path between "
                "bounce points is in hypograph(|B|). Use more knots (Y argument to"
                "Bounce2D).\n"
            )
        if plot:
            this_title = (
                title + rf" on field line $(\rho_{{l={lm[0]}}}, \alpha_{{m={lm[1]}}})$"
            )
            plots.append(
                plot_ppoly(
                    ppoly=ppoly,
                    z1=z1[lm],
                    z2=z2[lm],
                    k=pitch_inv[lm],
                    title=this_title,
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
    """Return minima of (z*, B(z*)) within open interval defined by knots.

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
    num_mins : jnp.ndarray
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
    """Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E B(ζ). Returns f(A).

    Parameters
    ----------
    z1, z2 : jnp.ndarray
        Shape (..., num pitch, num well).
        Boundaries to detect argmin between.
        ``z1`` (``z2``) stores left (right) boundaries.
    f : jnp.ndarray
        Function interpolated to ``mins``.
        Shape (..., num mins).

    Returns
    -------
    f : jnp.ndarray
        Shape (..., num pitch, num well).
        ``f`` at the minimum of ``B`` between ``z1`` and ``z2``.

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
    # now the variable delta is actually referring to theta. Rename to
    # avoid using more memory
    delta = delta.at[..., 0].add(alpha)
    assert delta.shape == (*angle.shape[:-2], num_alpha, num_transit * NFP, Y)
    return PiecewiseChebyshevSeries(delta, domain)


def fast_chebyshev(theta, f, Y, num_t, modes_t, modes_z, *, vander=None):
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
        Shape broadcasts with (num ρ, 1, modes_z.size, modes_t.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Chebyshev spectral resolution for ``f`` over a field period.
        Preferably power of 2.
    num_t : int
        Fourier resolution in poloidal direction.
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
        ``f`` over one toroidal transit. ``f.cheb`` broadcasts with
        shape (num ρ, num α, num transit * NFP, Y).

    """
    # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
    # compute a set of 2D Fourier series each on non-uniform tensor product grids
    # of size |𝛉|×|𝛇| where |𝛉| = num α × num transit × NFP and |𝛇| = Y.
    # Partial summation is more efficient than direct evaluation when
    # mn|𝛉||𝛇| > mn|𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > n + |𝛉|.

    # compute the toroidal part first
    f = ifft_mmt(
        cheb_pts(Y, theta.domain)[:, None] if vander is None else None,
        f,
        theta.domain,
        axis=-2,
        modes=modes_z,
        vander=vander,
    )[..., None, None, :, :]
    # then compute the poloidal part of f
    f = irfft_mmt_pos(theta.evaluate(Y), f, num_t, modes=modes_t)
    f = cheb_from_dct(dct(f, type=2, axis=-1) / Y)
    f = PiecewiseChebyshevSeries(f, theta.domain)
    assert f.cheb.shape == (*theta.cheb.shape[:-1], Y)
    return f


def fast_cubic_spline(
    theta,
    f,
    Y,
    num_t,
    modes_t,
    modes_z,
    NFP=1,
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
        θ over one toroidal transit. ``theta.cheb`` should broadcast with
        shape (num ρ, num α, num transit * NFP, theta.Y).
    f : jnp.ndarray
        Shape broadcasts with (num ρ, 1, modes_z.size, modes_t.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Number of knots per toroidal transit to interpolate ``f``.
        This number will be rounded up to an integer multiple of ``NFP``.
    num_t : int
        Fourier resolution in poloidal direction.
    modes_t : jnp.ndarray
        Real FFT Fourier modes in poloidal direction.
    modes_z : jnp.ndarray
        FFT Fourier modes in toroidal direction.
    NFP : int
        Number of field periods.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
    vander_t : jnp.ndarray
        Precomputed transform matrix for poloidal coordinate.
    vander_z : jnp.ndarray
        Precomputed transform matrix for toroidal coordinate.
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
    t = idct_mmt(
        x,
        theta.cheb.reshape(*lines, num_transit, NFP, 1, theta.Y),
        vander=vander_t,
    )
    if axisymmetric:
        t = t.reshape(*lines, num_transit, -1, 1)

    if nufft_eps < 1e-14 or f.shape[-1] < 14:
        # second condition for GPU
        f = f[..., None, None, None, :, :]
        f = irfft_mmt_pos(t, f, num_t, modes=modes_t)
    else:
        if len(lines) > 1:
            t = t.transpose(0, 4, 1, 2, 3).reshape(lines[0], num_z, -1)
        else:
            t = t.transpose(3, 0, 1, 2).reshape(num_z, -1)
        f = nufft1d2r(t, f, eps=nufft_eps).mT
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
    pitch_inv = jnp.atleast_1d(pitch_inv)
    if pitch_inv.ndim == 2:
        pitch_inv = pitch_inv[:, None]
    return pitch_inv


def truncate_rule(Y):
    """Truncation of Chebyshev series to reduce spectral aliasing."""
    return max(1, 7 * Y // 8)


def round_up_rule(Y, NFP, axisymmetric):
    """Round Y up to NFP multiple.

    Returns
    -------
    Y : int
        Number of points per toroidal transit.
    num_z : int
        Number of points per field period.
    axisymmetric : bool
        Whether there toroidal smmetry.

    """
    if axisymmetric:
        assert Y % NFP == 0, "Should set NFP = 1."
        NFP = Y
    num_z = (Y + NFP - 1) // NFP
    return num_z * NFP, num_z


def Y_B_rule(grid, spline):
    """Guess Y_B from grid resolution.

    Parameters
    ----------
    grid : Grid
        Grid.
    spline : bool
        Whether |B| will be approximated with cubic spline or Chebyshev.

    Returns
    -------
    Y_B : int
        Desired resolution for algorithm to compute bounce points.

    """
    Y_B = (grid.num_theta + grid.num_zeta) // 2
    # Due to backwards compatibility reasons Y_B is expected to indicate
    # resolution over full transit (a single field period) when spline is
    # true (false).
    return (Y_B * grid.NFP) if spline else Y_B


def num_well_rule(num_transit, NFP, Y_B=None):
    """Guess upper bound for number of wells based on spectrum.

    This should be loose enough that it is equivalent to ``num_well=None``,
    but more performant.
    """
    num_well = num_transit * (20 + NFP)
    return num_well if Y_B is None else min(num_well, num_transit * Y_B)


def get_vander(grid, Y, Y_B, NFP):
    """Builds Vandermonde matrices for objectives."""
    Y_trunc = truncate_rule(Y)
    Y_B, num_z = round_up_rule(Y_B, NFP, grid.num_zeta == 1)
    x = jnp.linspace(
        -1, 1, (Y_B // NFP) if (grid.num_zeta == 1) else num_z, endpoint=False
    )
    return {"dct spline": chebvander(x, Y_trunc - 1)}
