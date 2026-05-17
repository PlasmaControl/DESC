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
from desc.utils import apply, errorif, safediv

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
    from desc.compute._fast_ion import _radial_drift
    from desc.compute._turbulence import (
        _ae,
        _binormal_drift_wb_inverse,
        _energy_quad,
        _G_hat_half,
    )

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
        [_G_hat_half, _binormal_drift_wb_inverse, _radial_drift],
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


def _add_well_segments(fig, ax, segments, values, cmap, normalize_ae, linewidth):
    if not values.size:
        return None

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
        norm=colors.Normalize(vmin=vmin, vmax=vmax),
        linewidths=linewidth,
        alpha=0.95,
        zorder=1,
    )
    ax.add_collection(collection)
    cbar = fig.colorbar(collection, ax=ax, pad=0.08, fraction=0.055)
    cbar.set_label(cbar_label)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(5))
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


def _add_drifts(ax, well_data):
    drift_ax = ax.twinx()
    roots, omega_alpha, omega_psi = _drift_samples(well_data)
    if roots.size:
        drift_ax.scatter(
            roots,
            omega_psi,
            color="tab:orange",
            marker=".",
            s=9,
            alpha=0.9,
            label=r"$\widehat{\omega}_\psi$",
        )
        drift_ax.scatter(
            roots,
            omega_alpha,
            color="tab:olive",
            marker=".",
            s=9,
            alpha=0.9,
            label=r"$\widehat{\omega}_\alpha$",
        )
        drift_ax.axhline(0.0, color="0.15", linestyle="dotted", linewidth=0.9)
        legend = drift_ax.legend(
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="0.25",
            framealpha=0.95,
            fancybox=False,
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
    cmap="turbo",
    normalize_ae=True,
    include_drifts=True,
    linewidth=1.0,
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
        Colormap for the available-energy well segments.
    normalize_ae : bool, optional
        If True, normalize segment colors by the maximum plotted
        available-energy value.
    include_drifts : bool, optional
        If True, overlay drift-frequency samples on a secondary y-axis.
    linewidth : float, optional
        Width of the available-energy well segments.
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
    segments, values = _well_segments(well_data)
    _add_well_segments(fig, ax, segments, values, cmap, normalize_ae, linewidth)

    ax.plot(zeta, well_data.B, color="black", linewidth=1.8, zorder=3)
    _style_well_axes(ax, zeta, well_data.B)

    if include_drifts:
        _add_drifts(ax, well_data)

    return (fig, ax, well_data) if return_data else (fig, ax)
