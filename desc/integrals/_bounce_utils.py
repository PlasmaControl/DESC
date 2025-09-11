"""Utilities for bounce integrals."""

import numpy as np
from interpax import CubicSpline, PPoly
from matplotlib import pyplot as plt

from desc.backend import dct, ifft, jnp
from desc.integrals._interp_utils import (
    cheb_from_dct,
    cheb_pts,
    idct_mmt,
    ifft_mmt,
    irfft_mmt,
    nufft1d2r,
    polyroot_vec,
    polyval_vec,
)
from desc.integrals.basis import (
    FourierChebyshevSeries,
    PiecewiseChebyshevSeries,
    _add2legend,
    _in_epigraph_and,
    _plot_intersect,
)
from desc.integrals.quad_utils import bijection_from_disc
from desc.utils import atleast_nd, flatten_mat, setdefault, take_mask

_sentinel = -1e5


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
        pitch and field line. Default is ``None``, which will detect all wells,
        but due to current limitations in JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``, ``B`` are the poloidal and
        toroidal Fourier resolution of B, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.

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
    # We ignore the bounce points of particles only assigned to a class that are
    # trapped outside this snapshot of the field line.
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


def _set_default_plot_kwargs(kwargs, l=None, m=None):
    vlabel = r"$\vert B \vert$"
    default_title = (
        rf"Intersects $\zeta$ in epigraph$(${vlabel}$)$ "
        + rf"s.t. {vlabel}$(\zeta) = 1/\lambda$"
    )
    if l is not None and m is not None:
        default_title += rf" on field line $\rho(l={l})$, $\alpha(m={m})$"
    kwargs.setdefault("title", default_title)
    kwargs.setdefault("klabel", r"$1/\lambda$")
    kwargs.setdefault("hlabel", r"$\zeta$")
    kwargs.setdefault("vlabel", vlabel)
    return kwargs


def _check_bounce_points(z1, z2, pitch_inv, knots, B, plot=True, **kwargs):
    """Check that bounce points are computed correctly.

    For the plotting labels of ρ(l), α(m), it is assumed that the axis that
    enumerates the index l preceds the axis that enumerates the index m.
    """
    kwargs = _set_default_plot_kwargs(kwargs)
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
    pitch_inv = atleast_nd(3, _broadcast_for_bounce(pitch_inv))
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
                    + rf" on field line $\rho(l={lm[0]})$, $\alpha(m={lm[1]})$",
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
                    title=title
                    + rf" on field line $\rho(l={lm[0]})$, $\alpha(m={lm[1]})$",
                    **kwargs,
                )
            )
    return plots


def _check_interp(zeta, b_sup_z, B, f, result, plot=True):
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
        Output of ``_integrate``.
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
            + rf" on field line $\rho(l={idx[0]})$, $\alpha(m={idx[1]})$"
        )
        for i in marked:
            ax.plot(zeta[(*idx, i)], V[(*idx, i)], marker="o")
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
    title=r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$",
    hlabel=r"$z$",
    vlabel=r"$f$",
    show=True,
    start=None,
    stop=None,
    include_knots=False,
    knot_transparency=0.4,
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
        # With the new axes, the shapes are:
        #     z1, z2 (..., num pitch, num well, 1)
        # ext, g_ext (...,         1,        1, num extrema)
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


def get_fieldline(alpha, iota, num_transit):
    """Get set of field line poloidal coordinates {Aᵢ | Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)}.

    Parameters
    ----------
    alpha : jnp.ndarray
        Shape (num alpha, ).
        Starting field line poloidal labels {αᵢ₀}.
    iota : jnp.ndarray
        Shape (num rho, ).
        Rotational transform normalized by 2π.
    num_transit : int
        Number of toroidal transits to follow field line.

    Returns
    -------
    fieldline : jnp.ndarray
        Shape (num alpha, num rho, num transit).
        Set of field line poloidal coordinates {Aᵢ | Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)}.

    """
    iota = jnp.atleast_1d(iota)[:, None]
    alpha = jnp.atleast_1d(alpha)[:, None, None]
    # Select the next branch such that ϑ is continuous.
    #      αᵢ = ϑ − ιϕᵢ
    #    αᵢ₊₁ = ϑ − ιϕᵢ₊₁
    # αᵢ₊₁−αᵢ = ι(ϕᵢ-ϕᵢ₊₁) = ι(ζᵢ-ζᵢ₊₁) = ι 2π
    return alpha + iota * (2 * jnp.pi) * jnp.arange(num_transit)


def fourier_chebyshev(theta, iota, alpha, num_transit):
    """Parameterize θ on field lines ``alpha``.

    Parameters
    ----------
    theta : jnp.ndarray
        Shape (num rho, X, Y).
        DESC coordinates θ from ``Bounce2D.compute_theta``.
        ``X`` and ``Y`` are preferably rounded down to powers of two.
    iota : jnp.ndarray
        Shape (num rho, ).
        Rotational transform normalized by 2π.
    alpha : jnp.ndarray
        Shape (num alpha, ).
        Starting field line poloidal labels {αᵢ₀}.
    num_transit : int
        Number of toroidal transits to follow field line.

    Returns
    -------
    T : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line ``alpha[i]``. Each Chebyshev series approximates
        θ over one toroidal transit. ``T.cheb`` broadcasts with
        shape (num rho, num alpha, num transit, Y).

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
    whereas g(α, ζ) has to "decide" at the cut what the next field line is).

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
    fieldline = get_fieldline(alpha, iota, num_transit)
    if theta.ndim == 2:
        fieldline = fieldline.squeeze(1)

    # Reduce θ to a set of Chebyshev series. This is a partial summation technique.
    domain = (0, 2 * jnp.pi)
    T = FourierChebyshevSeries(theta, domain).compute_cheb(fieldline).swapaxes(0, -3)
    T = PiecewiseChebyshevSeries(T, domain)
    T.stitch()
    assert T.cheb.shape == (
        *theta.shape[:-2],
        jnp.size(alpha),
        num_transit,
        theta.shape[-1],
    )
    return T


def fast_chebyshev(T, f, Y, num_theta, m_modes, n_modes, NFP=1, *, vander=None):
    """Compute Chebyshev approximation of ``f`` on field lines using fast transforms.

    Parameters
    ----------
    T : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line ``alpha[i]``. Each Chebyshev series approximates
        θ over one toroidal transit. ``T.cheb`` should broadcast with
        shape (num rho, num alpha, num transit, T.Y).
    f : jnp.ndarray
        Shape broadcasts with (num rho, 1, n_modes.size, m_modes.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Chebyshev spectral resolution for ``f``. Preferably power of 2.
        Usually the spectrum of ``f`` is wider than θ, so one can upsample
        to about double the resolution of θ. (This is function dependent).
    num_theta : int
        Fourier resolution in poloidal direction.
    m_modes : jnp.ndarray
        Real FFT Fourier modes in poloidal direction.
    n_modes : jnp.ndarray
        FFT Fourier modes in toroidal direction.
    NFP : int
        Number of field periods.
    vander : jnp.ndarray
        Precomputed transform matrix.

    Returns
    -------
    f : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of ``f`` on field lines.
        {f_αᵢⱼ : ζ ↦ f(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line ``alpha[i]``. Each Chebyshev series approximates
        ``f`` over one toroidal transit. ``f.cheb`` broadcasts with
        shape (num rho, num alpha, num transit, Y).

    """
    # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
    # compute a set of 2D Fourier series each on non-uniform tensor product grids
    # of size |𝛉|×|𝛇| where |𝛉| = num alpha × num transit and |𝛇| = Y.
    # Partial summation is more efficient than direct evaluation when
    # mn|𝛉||𝛇| > mn|𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > n + |𝛉|.

    f = ifft_mmt(
        cheb_pts(Y, T.domain)[:, None] if vander is None else None,
        f,
        (0, 2 * jnp.pi / NFP),
        axis=-2,
        modes=n_modes,
        vander=vander,
    )[..., None, None, :, :]
    f = irfft_mmt(T.evaluate(Y), f, num_theta, _modes=m_modes)
    f = cheb_from_dct(dct(f, type=2, axis=-1) / Y)
    f = PiecewiseChebyshevSeries(f, T.domain)
    assert f.cheb.shape == (*T.cheb.shape[:-1], Y)
    return f


def fast_cubic_spline(
    T,
    f,
    Y,
    num_theta,
    m_modes,
    n_modes,
    NFP=1,
    nufft_eps=1e-6,
    *,
    vander_zeta=None,
    vander_theta=None,
    check=False,
):
    """Compute cubic spline of ``f`` on field lines using fast transforms.

    Parameters
    ----------
    T : PiecewiseChebyshevSeries
        Set of 1D Chebyshev spectral coefficients of θ on field lines.
        {θ_αᵢⱼ : ζ ↦ θ(αᵢⱼ, ζ) | αᵢⱼ ∈ Aᵢ} where Aᵢ = (αᵢ₀, αᵢ₁, ..., αᵢ₍ₘ₋₁₎)
        enumerates field line ``alpha[i]``. Each Chebyshev series approximates
        θ over one toroidal transit. ``T.cheb`` should broadcast with
        shape (num rho, num alpha, num transit, T.Y).
    f : jnp.ndarray
        Shape broadcasts with (num rho, 1, n_modes.size, m_modes.size).
        Fourier transform of f(θ, ζ) as returned by ``Bounce2D.fourier``.
    Y : int
        Number of knots per toroidal transit to interpolate ``f``.
        This number will be rounded up to an integer multiple of ``NFP``.
    num_theta : int
        Fourier resolution in poloidal direction.
    m_modes : jnp.ndarray
        Real FFT Fourier modes in poloidal direction.
    n_modes : jnp.ndarray
        FFT Fourier modes in toroidal direction.
    NFP : int
        Number of field periods.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
    vander_zeta : jnp.ndarray
        Precomputed transform matrix.
    vander_theta : jnp.ndarray
        Precomputed transform matrix.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    Returns
    -------
    f : jnp.ndarray
        Shape broadcasts with (num rho, num alpha, num transit * Y - 1, 4).
        Polynomial coefficients of the spline of f in local power basis.
        Last axis enumerates the coefficients of power series. For a polynomial
        given by ∑ᵢⁿ cᵢ xⁱ, coefficient cᵢ is stored at ``f[...,n-i]``.
        Second to last axis enumerates the polynomials that compose a particular
        spline.
    knots : jnp.ndarray
        Shape (num transit * Y).
        Knots of spline ``f``.

    """
    assert T.cheb.ndim >= 3
    lines = T.cheb.shape[:-2]

    num_zeta = (Y + NFP - 1) // NFP
    Y = num_zeta * NFP
    x = jnp.linspace(-1, 1, Y, endpoint=False)
    zeta = bijection_from_disc(x, T.domain[0], T.domain[1])

    # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
    # compute a set of 2D Fourier series each on uniform (non-uniform) in ζ (θ)
    # tensor product grids of size
    #   |𝛉|×|𝛇| where |𝛉| = num alpha × num transit × NFP and |𝛇| = Y/NFP.
    # Partial summation via FFT is more efficient than direct evaluation when
    # mn|𝛉||𝛇| > m log(|𝛇|) |𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > log|𝛇| + |𝛉|.

    if num_zeta >= f.shape[-2]:
        f = f.squeeze(-3)
        p = num_zeta - f.shape[-2]
        p = (p // 2, p - p // 2)
        pad = [(0, 0)] * f.ndim
        pad[-2] = p if (f.shape[-2] % 2 == 0) else p[::-1]
        f = jnp.fft.ifftshift(jnp.pad(jnp.fft.fftshift(f, -2), pad), -2)
        f = ifft(f, axis=-2, norm="forward")
    else:
        f = ifft_mmt(
            zeta[:num_zeta, None],
            f,
            (0, 2 * jnp.pi / NFP),
            axis=-2,
            modes=n_modes,
            vander=vander_zeta,
        )

    # θ at uniform ζ on field lines
    theta = idct_mmt(x, T.cheb[..., None, :], vander=vander_theta).reshape(
        *lines, T.X, NFP, num_zeta
    )

    if nufft_eps < 1e-14:
        f = irfft_mmt(theta, f[..., None, None, None, :, :], num_theta, _modes=m_modes)
    else:
        if len(lines) > 1:
            theta = theta.transpose(0, 4, 1, 2, 3).reshape(lines[0], num_zeta, -1)
        else:
            theta = theta.transpose(3, 0, 1, 2).reshape(num_zeta, -1)
        f = nufft1d2r(theta, f, eps=nufft_eps).mT
    f = f.reshape(*lines, -1)

    zeta = jnp.ravel(zeta + (T.domain[1] - T.domain[0]) * jnp.arange(T.X)[:, None])
    f = CubicSpline(x=zeta, y=f, axis=-1, check=check).c
    f = jnp.moveaxis(f, (0, 1), (-1, -2))
    assert f.shape == (*lines, T.X * Y - 1, 4)
    return f, zeta


def _move(f, out=True):
    """Use to move between the following shapes.

    The LHS shape enables the simplest broadcasting so it is used internally,
    while the RHS shape is the returned shape which enables simplest to use API
    for computing various quantities.

    When out is True, goes from left to right. Goes other way when False.

    (num pitch, num rho, num alpha, -1) ->  (num rho, num alpha, num pitch, -1)
    (num pitch,          num alpha, -1) ->  (         num alpha, num pitch, -1)
    (num pitch,                     -1) ->  (                    num pitch, -1)
    """
    assert f.ndim <= 4
    s, d = (0, -2) if out else (-2, 0)
    return jnp.moveaxis(f, s, d)


def _mmt_for_bounce(v, c):
    """Matrix multiplication transform.

    Parameters
    ----------
    v : jnp.ndarray
        Shape (num rho, num alpha, num pitch, num well, num quad,
                num zeta modes, num theta modes).
        Vandermonde array.
    c : jnp.ndarray
        Shape (num rho, 1, num zeta modes, num theta modes).
        Fourier coefficients.

    """
    return (v * c[..., None, None, None, :, :]).real.sum((-2, -1))


def _broadcast_for_bounce(pitch_inv):
    """Add axis if necessary.

    Parameters
    ----------
    pitch_inv : jnp.ndarray
        Shape broadcasts with (num rho, num pitch).

    Returns
    -------
    pitch_inv : jnp.ndarray
        Shape broadcasts with (num rho, num alpha, num pitch).

    """
    if jnp.ndim(pitch_inv) == 2:
        pitch_inv = pitch_inv[:, None]
    return pitch_inv
