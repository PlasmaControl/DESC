"""Available-energy plotting functions."""

from dataclasses import dataclass

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from interpax import PPoly
from interpax_fft import PiecewiseChebyshevSeries
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

from desc.backend import jnp
from desc.grid import LinearGrid
from desc.utils import apply, errorif, safediv, setdefault

from .bounce_integral import Bounce2D, Options


@dataclass(frozen=True)
class _AvailableEnergyWellData:
    """Available-energy data on one flux surface and one field line.

    Attributes
    ----------
    rho : float
        Flux-surface label.
    alpha : float
        Field-line label.
    zeta : ndarray
        Toroidal field-line coordinate used for plotting ``B``.
    B : ndarray
        Magnetic-field strength on ``zeta``.
    pitch_inv : ndarray
        Pitch quadrature nodes.
    points : tuple of ndarray
        Lower and upper bounce points. Each array has shape
        ``(num_pitch, num_well)``.
    valid : ndarray
        Boolean mask for wells with ordered bounce points, with shape
        ``(num_pitch, num_well)``.
    omega_alpha, omega_psi, ae_per_pitch_well : ndarray
        Per-pitch, per-well bounce quantities with shape
        ``(num_pitch, num_well)``.
    """

    rho: float
    alpha: float
    zeta: np.ndarray
    B: np.ndarray
    pitch_inv: np.ndarray
    points: tuple[np.ndarray, np.ndarray]
    valid: np.ndarray
    omega_alpha: np.ndarray
    omega_psi: np.ndarray
    ae_per_pitch_well: np.ndarray


def _surface_quantity(grid, template, value):
    if value is None:
        return None
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value * jnp.ones_like(template)
    if value.size == grid.num_rho:
        return grid.expand(value)
    return value * jnp.ones_like(template)


def _prepare_ae_data(
    grid,
    data,
    angle,
    bounce_names,
    radial_scale,
    binormal_scale,
    density_gradient,
    temperature_gradient,
):
    fun_data = apply(data, subset=bounce_names)
    for name in Bounce2D.required_names:
        fun_data[name] = data[name]
    fun_data.pop("iota", None)

    for name in fun_data:
        fun_data[name] = Bounce2D.fourier(Bounce2D.reshape(grid, fun_data[name]))

    density_gradient = _surface_quantity(grid, data["rho"], density_gradient)
    if density_gradient is None:
        density_gradient = safediv(data["ne_r"], data["ne"])
    temperature_gradient = _surface_quantity(grid, data["rho"], temperature_gradient)
    if temperature_gradient is None:
        temperature_gradient = safediv(data["Te_r"], data["Te"])
    errorif(
        not np.isfinite(density_gradient).all(),
        msg=("Density gradient was not set for this equilibrium."),
    )
    errorif(
        not np.isfinite(temperature_gradient).all(),
        msg=("Temperature gradient was not set for this equilibrium."),
    )

    surface_data = {
        "ae grad(density)": radial_scale * density_gradient,
        "ae psi width": radial_scale * data["psi_r"],
        "ae alpha width": binormal_scale * safediv(1.0, data["rho"]),
        "ae grad(temperature)": radial_scale * temperature_gradient,
    }
    fun_data.update(apply(surface_data, grid.compress))
    fun_data["iota"] = grid.compress(data["iota"])
    fun_data["min_tz |B|"] = grid.compress(data["min_tz |B|"])
    fun_data["max_tz |B|"] = grid.compress(data["max_tz |B|"])
    fun_data["angle"] = angle
    return fun_data


def _fieldline_B(bounce, num_zeta, rho_idx=0):
    def evaluate_ppoly(B):
        P = PPoly(B.T, bounce._c["knots"])
        return P(zeta)

    if isinstance(bounce._B, PPoly):
        zeta = np.linspace(bounce._B.x[0], bounce._B.x[-1], num_zeta)
        return zeta, bounce._B(zeta)[None, :]

    if isinstance(bounce._B, PiecewiseChebyshevSeries):
        domain = bounce._B.domain
        B = bounce._B.cheb
        if B.ndim == 4:
            B = B[rho_idx]
        zeta = np.linspace(domain[0], domain[-1], num_zeta)
        if B.ndim == 2:
            B = B[None, ...]
        return zeta, np.stack(
            [
                np.ravel(PiecewiseChebyshevSeries(B_alpha, domain).eval1d(zeta))
                for B_alpha in B
            ]
        )

    B = bounce._B
    if B.ndim == 4:
        B = B[rho_idx]
    zeta = np.linspace(bounce._c["knots"][0], bounce._c["knots"][-1], num_zeta)
    if B.ndim == 2:
        B = B[None, ...]
    return zeta, np.stack([evaluate_ppoly(B_alpha) for B_alpha in B])


def _ae_well_data(
    eq,
    rho=None,
    alpha=0.0,
    grid=None,
    data=None,
    angle=None,
    radial_scale=1.0,
    binormal_scale=1.0,
    density_gradient=None,
    temperature_gradient=None,
    num_energy=16,
    num_zeta=2000,
    **kwargs,
):
    """Compute per-pitch, per-well available-energy data for plotting.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to evaluate.
    rho : float, optional
        Flux-surface label. Exactly one value is supported. If omitted with a
        supplied ``grid``, the single rho value from ``grid`` is used.
    alpha : float, optional
        Field-line label. Exactly one value is supported.
    grid : Grid, optional
        Single-rho grid used for equilibrium quantities. If omitted, a
        ``LinearGrid`` is constructed.
    data : dict, optional
        Precomputed equilibrium data on ``grid``.
    angle : ndarray, optional
        Bounce2D angle map. If omitted, it is computed from ``eq``.
    radial_scale, binormal_scale : float, optional
        Correlation-length scale factors.
    density_gradient, temperature_gradient : float or ndarray, optional
        Values replacing ``ne_r / ne`` and ``Te_r / Te`` before multiplication
        by ``radial_scale``. If omitted, the equilibrium profiles are used.
    num_energy : int, optional
        Number of generalized Gauss-Laguerre nodes for the energy integral.
    num_zeta : int, optional
        Number of points used to plot ``|B|`` along the field line.
    **kwargs
        Additional options forwarded to ``Options.guess`` and ``Bounce2D``.

    Returns
    -------
    _AvailableEnergyWellData
        Per-pitch, per-well available-energy data on one flux surface and one
        field line.
    """
    from desc.compute._drift import _binormal_drift, _radial_drift, _sqrt_G_hat
    from desc.compute._turbulence import _ae, _energy_quad

    bounce_names = (
        "cvdrift (periodic)",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "|grad(psi)|*kappa_g",
    )
    surface_names = (
        "min_tz |B|",
        "max_tz |B|",
        "psi_r",
        "rho",
        "ne",
        "ne_r",
        "Te",
        "Te_r",
    )

    rho = None if rho is None else np.asarray(rho, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()
    errorif(
        rho is not None and rho.size != 1,
        msg="available-energy well plots require exactly one rho value.",
    )
    errorif(
        alpha.size != 1,
        msg="available-energy well plots require exactly one alpha value.",
    )

    X = kwargs.pop("X", 32)
    Y = kwargs.pop("Y", 32)
    if grid is None:
        if rho is None:
            rho = np.asarray([0.5])
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    else:
        errorif(
            grid.num_rho != 1,
            msg="available-energy well plots require a single-rho grid.",
        )
        grid_rho = grid.compress(grid.nodes[:, 0])
        if rho is None:
            rho = grid_rho
        else:
            errorif(
                not np.allclose(rho, grid_rho),
                msg="rho must match the single rho value in grid.",
            )
    if angle is None:
        angle = Bounce2D.angle(eq, X=X, Y=Y, rho=rho)

    opts = Options.guess(-1, grid, alpha=alpha, **kwargs)
    compute_names = bounce_names + surface_names + tuple(Bounce2D.required_names)
    if data is None:
        data = eq.compute(list(dict.fromkeys(compute_names)), grid=grid)

    fun_data = _prepare_ae_data(
        grid,
        data,
        angle,
        bounce_names,
        radial_scale,
        binormal_scale,
        density_gradient,
        temperature_gradient,
    )
    bounce = Bounce2D(grid, fun_data, angle, **opts)

    pitch_inv, _ = Bounce2D.pitch_quad(
        fun_data["min_tz |B|"], fun_data["max_tz |B|"], opts.pitch_quad
    )
    points = bounce.points(pitch_inv, opts.num_well)
    G, G_ω_α, G_ω_ψ = bounce.integrate(
        [_sqrt_G_hat, _binormal_drift, _radial_drift],
        pitch_inv,
        fun_data,
        bounce_names,
        points,
        loop=opts.loop,
    )

    energy, energy_weight = _energy_quad(num_energy)
    ae_per_pitch_well = jnp.sum(
        _ae(G, G_ω_α, G_ω_ψ, fun_data, energy) * energy_weight[..., None],
        axis=-2,
    )

    shape = (-1,) + (1,) * (G.ndim - 1)
    G_ω_α = G_ω_α * fun_data["ae psi width"].reshape(shape)
    G_ω_ψ = G_ω_ψ * fun_data["ae alpha width"].reshape(shape)
    ω_α = safediv(G_ω_α, G)
    ω_ψ = safediv(G_ω_ψ, G)

    zeta, B = _fieldline_B(bounce, num_zeta)
    valid = points[0] < points[1]
    if B.ndim > 1:
        B = B[0]

    return _AvailableEnergyWellData(
        rho=rho[0],
        alpha=alpha[0],
        zeta=zeta,
        B=B,
        pitch_inv=pitch_inv[0],
        points=(points[0][0, 0], points[1][0, 0]),
        valid=valid[0, 0],
        omega_alpha=ω_α[0, 0],
        omega_psi=ω_ψ[0, 0],
        ae_per_pitch_well=ae_per_pitch_well[0, 0],
    )


def _well_segments(well_data):
    zeta1, zeta2 = well_data.points
    segments = []
    values = []

    for pitch_idx, y in enumerate(well_data.pitch_inv):
        for well_idx in range(zeta1.shape[1]):
            if not well_data.valid[pitch_idx, well_idx]:
                continue
            segments.append(
                [
                    (zeta1[pitch_idx, well_idx], y),
                    (zeta2[pitch_idx, well_idx], y),
                ]
            )
            values.append(well_data.ae_per_pitch_well[pitch_idx, well_idx])

    return segments, np.asarray(values)


def _color_norm(norm, values, vmin, vmax):
    if norm is None or norm == "linear":
        return colors.Normalize(vmin=vmin, vmax=vmax)
    if norm == "log":
        positive = values[np.isfinite(values) & (values > 0)]
        errorif(
            not positive.size,
            msg="LogNorm requires at least one positive available-energy value.",
        )
        vmin = 1e-5 if vmax > 1e-5 else np.nanmin(positive)
        return colors.LogNorm(vmin=vmin, vmax=vmax)
    return norm


def _pitch_spacing_linewidths(fig, ax, pitch_inv, segment_pitch, fallback=1.0):
    pitch_inv = np.asarray(pitch_inv, dtype=float)
    pitch_inv = np.unique(pitch_inv[np.isfinite(pitch_inv)])
    if pitch_inv.size < 2:
        return fallback

    display_y = ax.transData.transform(
        np.column_stack((np.zeros_like(pitch_inv), pitch_inv))
    )[:, 1]
    spacing = np.diff(np.sort(display_y))
    if not np.any(spacing > 0):
        return fallback

    spacing = np.concatenate(
        ([spacing[0]], np.minimum(spacing[:-1], spacing[1:]), [spacing[-1]])
    )
    # Adjacent strokes that exactly touch can still leave subpixel white seams
    # after antialiasing. Overscan slightly in display space to make filled
    # pitch bands visually continuous.
    LINEWIDTH_OVERSCAN_PIXELS = 1.0
    max_linewidth = (spacing + LINEWIDTH_OVERSCAN_PIXELS) * 72 / fig.dpi
    idx = np.searchsorted(pitch_inv, segment_pitch)
    idx = np.clip(idx, 0, pitch_inv.size - 1)
    prev_idx = np.clip(idx - 1, 0, pitch_inv.size - 1)
    idx = np.where(
        np.abs(pitch_inv[prev_idx] - segment_pitch)
        < np.abs(pitch_inv[idx] - segment_pitch),
        prev_idx,
        idx,
    )
    return max_linewidth[idx]


def _add_well_segments(
    fig,
    ax,
    segments,
    values,
    pitch_inv,
    cmap,
    normalize_ae,
    norm,
    linewidth,
):
    if not values.size:
        return None
    auto_linewidth = linewidth is None or not np.isfinite(linewidth)
    initial_linewidth = 1.0 if auto_linewidth else linewidth

    max_value = np.nanmax(values)
    if normalize_ae and max_value > 0:
        values = values / max_value
        cbar_label = r"$\widehat{A}_\lambda / \widehat{A}_{\lambda,\max}$"
        vmin, vmax = 0.0, 1.0
    else:
        cbar_label = r"$\widehat{A}_\lambda$"
        vmin, vmax = 0.0, max_value

    collection = LineCollection(
        segments,
        array=values,
        cmap=cmap,
        norm=_color_norm(norm, values, vmin, vmax),
        linewidths=initial_linewidth,
        alpha=0.95,
        zorder=1,
    )
    ax.add_collection(collection)
    cbar = fig.colorbar(collection, ax=ax, pad=0.08, fraction=0.055)
    cbar.set_label(cbar_label)
    if not isinstance(collection.norm, colors.LogNorm):
        cbar.ax.yaxis.set_major_locator(MaxNLocator(5))
    if auto_linewidth:
        fig.canvas.draw()
        segment_pitch = np.asarray(segments, dtype=float)[:, 0, 1]
        collection.set_linewidths(
            _pitch_spacing_linewidths(fig, ax, pitch_inv, segment_pitch)
        )
    return collection


def _drift_samples(well_data):
    zeta1, zeta2 = well_data.points
    valid = well_data.valid
    if not np.any(valid):
        empty = np.empty(0)
        return empty, empty, empty

    roots = np.column_stack((zeta1[valid], zeta2[valid])).ravel()
    omega_alpha = np.repeat(well_data.omega_alpha[valid], 2)
    omega_psi = np.repeat(well_data.omega_psi[valid], 2)
    return roots, omega_alpha, omega_psi


def _add_drifts(ax, well_data, omega_psi_color, omega_alpha_color):
    drift_ax = ax.twinx()
    roots, omega_alpha, omega_psi = _drift_samples(well_data)
    if roots.size:
        drift_ax.scatter(
            roots,
            omega_psi,
            color=omega_psi_color,
            marker=".",
            s=9,
            alpha=0.9,
            label=r"$\widehat{\omega}_\psi$",
        )
        drift_ax.scatter(
            roots,
            omega_alpha,
            color=omega_alpha_color,
            marker=".",
            s=9,
            alpha=0.9,
            label=r"$\widehat{\omega}_\alpha$",
        )
        drift_ax.axhline(0.0, color="0.15", linestyle="dotted", linewidth=1.0)
        legend = drift_ax.legend(
            loc="lower right",
            facecolor="white",
            edgecolor="0.25",
            framealpha=0.95,
            markerscale=2.5,
            scatterpoints=3,
        )
        legend.get_frame().set_linewidth(0.8)
    drift_ax.set_ylabel(r"$\widehat{\omega}_\alpha,\ \widehat{\omega}_\psi$")
    drift_ax.yaxis.set_major_locator(MaxNLocator(5))
    return drift_ax


def _style_well_axes(ax, zeta, B):
    zeta_min = np.nanmin(zeta)
    zeta_max = np.nanmax(zeta)
    y_min = np.nanmin(B)
    y_max = np.nanmax(B)
    y_range = y_max - y_min
    y_pad = 0.04 * y_range if y_range > 0 else max(0.04 * abs(y_max), 1.0)

    ax.set_xlim(zeta_min, zeta_max)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_ylabel(r"$|B|$")
    ax.set_xlabel(r"$\zeta$")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_available_energy(
    eq,
    ax=None,
    cmap=None,
    normalize_ae=True,
    norm="log",
    include_drifts=True,
    linewidth=None,
    omega_psi_color=None,
    omega_alpha_color=None,
    return_data=False,
    **kwargs,
):
    """Plot an AEpy-style available-energy map over bounce wells.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to evaluate.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If omitted, a new figure and axes are created.
    cmap : str or Colormap, optional
        Colormap for the available-energy well segments. If omitted, ``"turbo"``
        is used for log normalization and ``"plasma"`` is used otherwise.
    normalize_ae : bool, optional
        If True, normalize segment colors by the maximum plotted
        available-energy value.
    norm : {"log", "linear"} or matplotlib.colors.Normalize, optional
        Color normalization for available-energy well segments. ``"log"``
        uses ``matplotlib.colors.LogNorm`` with cutoff at ``1e-5``.
    include_drifts : bool, optional
        If True, overlay drift-frequency samples on a secondary y-axis.
    linewidth : float or None, optional
        Width of the available-energy well segments. If ``None`` or ``np.inf``,
        fill the pitch-grid spacing.
    omega_psi_color, omega_alpha_color : color, optional
        Colors for the overlaid drift-frequency markers.
    return_data : bool, optional
        If True, return ``well_data`` along with the figure and axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Primary axes containing ``|B|`` and the well segments.
    well_data : _AvailableEnergyWellData, optional
        Returned only when ``return_data`` is True.

    """
    well_data = _ae_well_data(eq, **kwargs)

    fig, ax = (
        plt.subplots(figsize=kwargs.pop("figsize", (7, 4.5)), constrained_layout=True)
        if ax is None
        else (ax.figure, ax)
    )

    zeta = well_data.zeta
    ax.plot(zeta, well_data.B, color="black", linewidth=2, zorder=3)
    _style_well_axes(ax, zeta, well_data.B)

    segments, values = _well_segments(well_data)
    _add_well_segments(
        fig,
        ax,
        segments,
        values,
        well_data.pitch_inv,
        setdefault(
            cmap,
            (
                "turbo"
                if (norm == "log" or isinstance(norm, colors.LogNorm))
                else "plasma"
            ),
        ),
        normalize_ae,
        norm,
        linewidth,
    )

    is_log = norm == "log" or isinstance(norm, colors.LogNorm)
    omega_psi_color = setdefault(
        omega_psi_color, "tab:purple" if is_log else "springgreen"
    )
    omega_alpha_color = setdefault(omega_alpha_color, "tab:gray")

    if include_drifts:
        _add_drifts(ax, well_data, omega_psi_color, omega_alpha_color)

    return (fig, ax, well_data) if return_data else (fig, ax)
