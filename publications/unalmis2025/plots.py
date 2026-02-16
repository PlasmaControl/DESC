"""Scripts to generate plots for the article."""

import os
import pickle
import warnings
from fractions import Fraction
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pytest
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import plotting
from desc.backend import jnp
from desc.compat import flip_theta
from desc.compute import data_index
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.external.neo import NeoIO
from desc.grid import Grid, LinearGrid
from desc.integrals import Bounce2D
from desc.integrals._bounce_utils import truncate_rule
from desc.plotting import plot_boozer_surface
from plotting import plot_2d, plot_3d, plot_comparison, plot_section

plotting._AXIS_LABELS_RPZ = [
    r"$R ~(\mathrm{meters})$",
    r"$\phi$",
    r"$Z ~(\mathrm{meters})$",
]


def pi_formatter(ax, NFP=1):

    def pi_form(x, pos):
        multiple = np.round(NFP * x / np.pi)
        if multiple == 0:
            return "0"
        elif multiple == 1:
            return r"$\pi$"
        elif multiple == -1:
            return r"$-\pi$"
        else:
            return rf"${int(multiple)}\pi$"

    ax.set_major_locator(mticker.MultipleLocator(base=np.pi / NFP))
    ax.set_major_formatter(mticker.FuncFormatter(pi_form))
    return ax


def pi_half_formatter(ax, NFP=1):

    def pi_half_form(x, pos):
        multiple = np.round(NFP * x / (np.pi / 2))
        if multiple == 0:
            return "0"

        frac = Fraction(int(multiple), 2).limit_denominator()
        num, den = frac.numerator, frac.denominator

        if num == 1:
            num_str = r"\pi"
        elif num == -1:
            num_str = r"-\pi"
        else:
            num_str = rf"{num}\pi"

        return rf"${num_str}$" if (den == 1) else rf"${num_str}/{den}$"

    ax.set_major_locator(mticker.MultipleLocator(base=np.pi / 2 / NFP))
    ax.set_major_formatter(mticker.FuncFormatter(pi_half_form))
    return ax


def dual_pi_formatter(ax_axis, iota_NFP, NFP=1):
    def major_pi_form(x, pos):
        if np.isclose(x, np.pi / 2):
            return ""  # too close to iota tick for NFP

        multiple = np.round(NFP * x / (np.pi / 2))
        if multiple == 0:
            return "0"

        frac = Fraction(int(multiple), 2 * NFP).limit_denominator()
        num, den = frac.numerator, frac.denominator

        num_str = r"\pi" if num == 1 else (r"-\pi" if num == -1 else rf"{num}\pi")
        return rf"${num_str}$" if (den == 1) else rf"${num_str}/{den}$"

    def minor_iota_form(x, pos):
        iota_step = iota_NFP * (2 * np.pi)
        multiple_iota = np.round(x / iota_step, 5)

        if np.isclose(x, 0):
            return ""

        if multiple_iota % 1 == 0:
            return rf"$({int(multiple_iota * 2)} \pi / N_{{\text{{FP}}}}) \iota$"
        return ""

    def get_locs(base):
        vmin, vmax = ax_axis.get_view_interval()
        return np.arange(
            np.floor(vmin / base) * base, np.ceil(vmax / base) * base + base, base
        )

    ax_axis.set_major_locator(mticker.FixedLocator(get_locs(np.pi / 2)))
    ax_axis.set_major_formatter(mticker.FuncFormatter(major_pi_form))

    ax_axis.set_minor_locator(mticker.FixedLocator(get_locs(iota_NFP * (2 * np.pi))))
    ax_axis.set_minor_formatter(mticker.FuncFormatter(minor_iota_form))

    ax = ax_axis.axes

    ax.tick_params(axis="y", which="minor", length=3, labelsize="small")
    ax.tick_params(axis="y", which="major")

    return ax_axis


def test_plot_bounce_point(name="W7-X", X=64, Y=64, Y_B=500, num_pitch=20):
    """High resolution plot for the paper."""
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 12

    eq = get(name)
    rho = 1.0
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data = eq.compute(Bounce2D.required_names + ["min_tz |B|", "max_tz |B|"], grid=grid)
    angle = Bounce2D.angle(eq, X, Y, rho=rho)
    bounce = Bounce2D(grid, data, angle, Y_B, num_transit=2)
    pitch_inv, _ = Bounce2D.get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
        simp=False,
    )
    fig, ax, legend = bounce.plot(
        0,
        0,
        pitch_inv[0],
        klabel=r"$\varrho$",
        k_transparency=0.25,
        show=False,
        include_legend=False,
        return_legend=True,
        figsize=(8.5, 4),
        title="",
        markersize=plt.rcParams["lines.markersize"] * 1.5,
    )
    fig.tight_layout()
    ax.xaxis = pi_half_formatter(ax.xaxis)
    fig.subplots_adjust(right=0.75)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.75)

    left, bottom, width, height = [0.775, 0.6, 0.1625, 0.35]
    axins = fig.add_axes([left, bottom, width, height])
    line = ax.lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    axins.plot(x_data, y_data)

    for collection in ax.collections:
        offsets = collection.get_offsets()
        x_scatter, y_scatter = offsets[:, 0], offsets[:, 1]
        color = collection.get_facecolor()
        marker = collection.get_paths()[0]
        size = collection.get_sizes()[0]
        axins.scatter(x_scatter, y_scatter, c=color, marker=marker, s=size)

    x1, x2 = 6, 7.2
    y1, y2 = 2.6, 2.8
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.spines["top"].set_visible(True)
    axins.spines["right"].set_visible(True)
    axins.spines["left"].set_visible(True)
    axins.spines["bottom"].set_visible(True)
    for spine in axins.spines.values():
        spine.set_linewidth(1.75)

    axins.tick_params(
        left=False, right=True, labelleft=False, labelright=True, labelsize=11
    )
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.5, linestyle=":")
    fig.legend(
        legend.values(),
        legend.keys(),
        loc="lower right",
        bbox_to_anchor=(0.925, 0.175),
        labelspacing=0.1,
        frameon=False,
    )
    plt.savefig("bounce_point_w7x.pdf")


def test_plot2d_alphas(name="NCSX"):
    """Plot alpha 2d plots."""
    plt.rcParams["figure.constrained_layout.use"] = True
    fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharex=True)

    eq = get(name)
    M = max(eq.M_grid, 50)
    N = max(eq.N_grid, 50)
    grid = LinearGrid(rho=1, M=M, N=N, NFP=eq.NFP)
    iota = eq.compute("iota", grid=grid)["iota"].mean()

    kwargs = dict(
        cbar_format="%.2f",
        cbar_ax_tick_label_size=10,
        ax_tick_params_label_size=12,
        title_fontsize=14,
        xlabel_fontsize=13,
        ylabel_fontsize=13,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unequal number of field periods")
        warnings.filterwarnings(
            "ignore",
            "Poloidal grid resolution is higher than necessary for coordinate",
        )
        _, ax = plot_2d(
            eq,
            "|B|",
            grid=grid,
            ax=axs[1],
            cmap="viridis",
            label=r"$\vert B \vert$",
            **kwargs,
        )
        ax.yaxis = pi_half_formatter(ax.yaxis)
        ax.xaxis = pi_half_formatter(ax.xaxis, NFP=eq.NFP)
        ax.set_xlabel(r"$N_{\text{FP}} \zeta$")
        ax.set_ylabel(r"$\theta$", labelpad=-5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.75)
        custom_name = "alpha, zeta to theta - alpha"
        fig, ax, t_dat = plot_2d(
            eq,
            custom_name,
            grid=grid,
            ax=axs[0],
            cmap="twilight",
            label=r"$\theta - \alpha$",
            **kwargs,
            return_data=True,
        )
        ax.yaxis = dual_pi_formatter(ax.yaxis, np.abs(iota) / eq.NFP)
        ax.set_xlabel(r"$N_{\text{FP}} \zeta$")
        alphs = t_dat[r"$\alpha$".strip("$").strip("\\")]
        theta = t_dat["alpha, zeta to theta - alpha"] + alphs
        zets = t_dat[r"$\zeta$".strip("$").strip("\\")]

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.75)

    print(alphs.shape, theta.shape)
    mid = alphs.shape[0] // 2
    tmid = (theta[mid, :] + theta[mid + 1, :]) / 2
    axs[1].plot(zets[mid], tmid, color="white", linewidth=2.5)

    plt.savefig(f"plot_2d_alphas_{name}.pdf")


def test_plot_theta_mod(X=64, Y=64, tol=1e-7):
    """θ mod (2π)."""
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 15})
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15

    eq = get("NCSX")
    rho = 1.0
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data = eq.compute(Bounce2D.required_names, grid=grid)
    angle = Bounce2D.angle(eq, X, Y, rho, tol=tol)
    bounce = Bounce2D(grid, data, angle, 1, np.array([0.0, np.pi / 2, np.pi]), 4)

    lw = 1.5
    kwargs = dict(title="", show=False, include_legend=False, lw=lw)
    fig, ax = bounce.plot_theta(0, 0, **kwargs)
    fig2, ax2 = bounce.plot_theta(0, 1, **kwargs)
    fig3, ax3 = bounce.plot_theta(0, 2, **kwargs)
    for line in ax2.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        label = line.get_label()
        ax.plot(x_data, y_data, label=label, lw=lw)
    for line in ax3.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        label = line.get_label()
        ax.plot(x_data, y_data, label=label, lw=lw)
    plt.close(fig2)
    plt.close(fig3)

    ax.lines[0].set_label(r"$\alpha = 0$")
    ax.lines[1].set_label(r"$\alpha = \pi/2$")
    ax.lines[2].set_label(r"$\alpha = \pi$")

    for line in ax.lines:
        y_data = line.get_ydata()
        diffs = np.diff(y_data)
        jump_indices = np.where(np.abs(diffs) > 1.9 * np.pi)[0]
        y_new = y_data.copy()
        y_new[jump_indices] = np.nan
        line.set_ydata(y_new)

    ax.yaxis = pi_half_formatter(ax.yaxis)
    ax.xaxis = pi_formatter(ax.xaxis)
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3)
    ax.legend(labelspacing=0.001, framealpha=1, borderpad=0.25, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.savefig("theta_plot_ncsx.pdf")


@pytest.mark.parametrize("name, X, Y", [("NCSX", 72, 59), ("HELIOTRON", 50, 30)])
def test_plot_angle_spectrum(name, X, Y, tol=1e-7):
    """Magnitude of the spectral coefficients of α, ζ → θ − α."""
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16

    angle = Bounce2D.angle(get(name), X, Y, rho=1.0, tol=tol)
    fig = Bounce2D.plot_angle_spectrum(angle, 0, title="", truncate=truncate_rule(Y))
    fig.savefig(f"angle_spectrum_{name}.pdf")


def test_plot_resolution_scan(name="W7-X", mode=1, top=0.0011):
    """Mode 0 for compute, 1 for plot."""
    assert mode == 0 or mode == 1

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.figure(figsize=(7, 5))

    res_table = [
        [16, 24, 12, 25],
        [16, 24, 16, 50],
        [24, 32, 32, 100],
        # very high resolution... let's assume this one is truth to machine precision
        [64, 64, 64, 200],
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(res_table)]
    colors[3] = "purple"
    linestyles = ["-", "--", ":", "-."]

    if mode == 0:
        eq = get(name)
        rho = jnp.linspace(0, 1, 40)
        alpha = jnp.linspace(0, 2 * jnp.pi, 5, endpoint=False)
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        num_transit = 20
        pick_data = {"rho": rho, "eps_32": []}
    else:
        with open(f"res_scan_{name}.pkl", "rb") as file:
            pick_data = pickle.load(file)

    for i, res in enumerate(res_table):
        if mode == 0:
            data = eq.compute(
                "effective ripple 3/2",
                grid=grid,
                angle=Bounce2D.angle(eq, X=res[0], Y=res[1], rho=rho),
                alpha=alpha,
                Y_B=200,  # something super high to focus on others
                num_transit=num_transit,
                num_well=20 * num_transit,
                num_quad=res[2],
                num_pitch=res[3],
            )
            eps_32 = grid.compress(data["effective ripple 3/2"])
            pick_data["eps_32"].append(eps_32)
            continue

        plt.plot(
            pick_data["rho"],
            pick_data["eps_32"][i],
            linewidth=2,
            label=rf"${res[0], res[1], res[3], res[2]}$",
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i],
        )

    if mode == 0:
        with open(f"res_scan_{name}.pkl", "wb") as file:
            pickle.dump(pick_data, file)
        return

    plt.ylim(top=top)
    plt.xlabel(r"$\rho$", fontsize=24)
    plt.ylabel(r"$\epsilon_{\text{eff}}^{3/2}$", fontsize=24)
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.legend(fontsize=23)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)

    plt.savefig(f"res_scan_{name}.pdf")


def test_plot_neo_compare(
    pick_filename="res_scan_W7-X.pkl",
    neo_filename="../../tests/inputs/neo_out.W7-X",
    top=0.0011,
):
    """Plot comparison to NEO."""
    with open(pick_filename, "rb") as file:
        data = pickle.load(file)

    neo_rho, neo_eps_32 = NeoIO.read(neo_filename)

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.figure(figsize=(7, 5))

    plt.plot(neo_rho, neo_eps_32, "--", linewidth=3, label="NEO", color="tab:blue")
    plt.plot(
        data["rho"], data["eps_32"][-1], "-", linewidth=3, label="DESC", color="purple"
    )
    plt.ylim(top=top)
    plt.xlabel(r"$\rho$", fontsize=24)
    plt.ylabel(r"$\epsilon_{\text{eff}}^{3/2}$", fontsize=24)
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.legend(fontsize=20)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)

    plt.savefig("NEO_vs_DESC.pdf")


def test_plot_optimized_ripple(mode=1):
    """Mode 0 for compute, 1 for plot."""
    assert mode == 0 or mode == 1

    def compute(eq):
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        num_transit = 20
        data = eq.compute(
            "effective ripple 3/2",
            grid=grid,
            angle=Bounce2D.angle(eq, X=32, Y=32, rho=rho),
            alpha=jnp.linspace(0, 2 * jnp.pi, 3, endpoint=False),
            Y_B=200,
            num_transit=num_transit,
            num_well=20 * num_transit,
            num_quad=32,
            num_pitch=100,
        )
        return grid.compress(data["effective ripple 3/2"])

    if mode == 0:
        eq0 = Equilibrium.load("eq_initial.h5")
        eq1 = Equilibrium.load("eq_optimized.h5")
        rho = jnp.linspace(0, 1, 20)
        data = {"rho": rho, "eps_32_init": compute(eq0), "eps_32_opt": compute(eq1)}
        with open("data_opt.pkl", "wb") as file:
            pickle.dump(data, file)
        return

    with open("data_opt.pkl", "rb") as file:
        data = pickle.load(file)

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 22})
    plt.rcParams["axes.labelsize"] = 29
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 23
    plt.figure(figsize=(7, 6))

    plt.plot(
        data["rho"],
        data["eps_32_init"],
        "-",
        linewidth=4,
        label="initial",
        color="tab:red",
    )
    plt.plot(
        data["rho"],
        data["eps_32_opt"],
        "-",
        linewidth=4,
        label="optimized",
        color="tab:blue",
    )
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    plt.legend(fontsize=26)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)

    plt.savefig("ripple_comparison.pdf")


def test_plot_optimized_boozer():
    eq0 = Equilibrium.load("eq_initial.h5")
    eq1 = Equilibrium.load("eq_optimized.h5")
    eq1 = flip_theta(eq1)
    assert eq0.NFP == eq1.NFP

    plt.rcParams["figure.constrained_layout.use"] = True

    rho0 = 1.0
    fig, ax, Boozer_data0 = plot_boozer_surface(eq0, rho=rho0, return_data=True)
    plt.close()

    fig, ax, Boozer_data1 = plot_boozer_surface(eq1, rho=rho0, return_data=True)
    plt.close()

    for i, Boozer_data in enumerate([Boozer_data0, Boozer_data1]):
        theta_B0 = Boozer_data["theta_B"]
        zeta_B0 = Boozer_data["zeta_B"]
        B0 = Boozer_data["|B|"]

        fig, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contour(
            zeta_B0,
            theta_B0,
            B0,
            levels=np.linspace(np.min(B0), np.max(B0), 30),
            cmap="turbo",
        )

        cbar = fig.colorbar(contour, ax=ax, orientation="vertical", format="%.2f")
        cbar.ax.tick_params(labelsize=18)
        ax.xaxis = pi_half_formatter(ax.xaxis, NFP=eq1.NFP)
        ax.yaxis = pi_half_formatter(ax.yaxis)
        ax.set_xlabel(r"$N_{\text{FP}} \zeta_{\mathrm{Boozer}}$", fontsize=24)
        ax.set_ylabel(r"$\theta_{\mathrm{Boozer}}$", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=20)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

        plt.savefig(f"Boozer_contour_plot_{i}.pdf")
        plt.close()


def plot_bavg_drift(
    eq, rho=1.0, alphas=None, num_pitch=None, ax=None, mode=1, **kwargs
):
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    levels = kwargs.pop("levels", 30)

    plt.rcParams["figure.constrained_layout.use"] = True
    nufft_eps = kwargs.pop("nufft_eps", 1e-9)
    X = kwargs.pop("X", 32)
    Y = kwargs.pop("Y", 32)
    Y_B = kwargs.pop("Y_B", 64)
    num_quad = kwargs.pop("num_quad", 32)
    num_transit = kwargs.pop("num_transit", 1 if (eq.NFP > 1) else 2)

    from desc.integrals.bounce_integral import Bounce2D

    figsize = kwargs.pop("figsize", (3.5, 3))
    cmap = kwargs.pop("cmap", "turbo")
    cbar_format = kwargs.pop("cbar_format", "%.2f")
    cbar_ax_tick_label_size = kwargs.pop("cbar_ax_tick_label_size", 11)
    ax_tick_params_label_size = kwargs.pop("ax_tick_params_label_size", 12)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", 13)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", 13)

    if alphas is None:
        alphas = np.linspace(0, 2 * np.pi, 50)
    if num_pitch is None:
        num_pitch = 50

    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    data0 = eq.compute(
        "gamma_c",
        grid=grid,
        angle=Bounce2D.angle(eq, X, Y, rho, tol=1e-9),
        Y_B=Y_B,
        num_transit=num_transit,
        num_quad=num_quad,
        num_pitch=num_pitch,
        alpha=alphas,
        nufft_eps=nufft_eps,
    )

    gamma_c = data0["gamma_c"][0].T
    if mode == 0:
        return gamma_c.min(), gamma_c.max()

    minB = data0["min_tz |B|"][0]
    maxB = data0["max_tz |B|"][0]
    pitch, _ = Bounce2D.get_pitch_inv_quad(minB, maxB, num_pitch)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    xx, yy = np.meshgrid(alphas, pitch)

    im = ax.contourf(xx, yy, gamma_c, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, format=cbar_format)
    cbar.ax.tick_params(labelsize=cbar_ax_tick_label_size)
    cbar.update_ticks()

    ax.xaxis = pi_half_formatter(ax.xaxis)
    ax.set_xlabel(r"$\alpha$", fontsize=xlabel_fontsize)
    ax.set_ylabel(r"$\varrho$", fontsize=ylabel_fontsize)
    ax.tick_params(labelsize=ax_tick_params_label_size)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.75)

    return fig, ax


def test_plot_bavg_drift():
    eq0 = Equilibrium.load("eq_final_OH.h5")
    eq1 = Equilibrium.load("eq_optimized_final2.h5")
    eq1 = flip_theta(eq1)
    assert eq0.NFP == eq1.NFP

    rho0 = 1.0
    alphas = np.linspace(0, 2 * np.pi, 100)
    num_pitch = 150

    ming, maxg = plot_bavg_drift(
        eq0,
        rho=rho0,
        alphas=alphas,
        num_pitch=num_pitch,
        X=80,
        Y=80,
        Y_B=400,
        num_quad=150,
        mode=0,
    )
    ming1, maxg1 = plot_bavg_drift(
        eq1,
        rho=rho0,
        alphas=alphas,
        num_pitch=num_pitch,
        X=80,
        Y=80,
        Y_B=400,
        num_quad=150,
        mode=0,
    )

    vmin = min(ming, ming1)
    vmax = max(maxg, maxg1)
    levels = np.linspace(vmin, vmax, 30)

    fig, ax = plot_bavg_drift(
        eq0,
        rho=rho0,
        alphas=alphas,
        num_pitch=num_pitch,
        X=80,
        Y=80,
        Y_B=400,
        num_quad=150,
        mode=1,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
    )
    plt.savefig("gamma_contour_plot_1.pdf")
    plt.close()

    fig, ax = plot_bavg_drift(
        eq1,
        rho=rho0,
        alphas=alphas,
        num_pitch=num_pitch,
        X=80,
        Y=80,
        Y_B=400,
        num_quad=150,
        vmin=vmin,
        mode=1,
        vmax=vmax,
        levels=levels,
    )
    plt.savefig("gamma_contour_plot_2.pdf")
    plt.close()


def test_plot_X_section():
    """Plots cross-sections of initial and optimized equilibrium."""
    eq0 = Equilibrium.load("eq_initial.h5")
    eq1 = Equilibrium.load("eq_optimized.h5")

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 24})

    fig, ax = plot_comparison(
        [eq0, eq1],
        lw=np.array([2, 1.5]),
        phi=4,
        xlabel_fontsize=22,
        ylabel_fontsize=22,
        title_fontsize=22,
        color=["tab:red", "tab:blue"],
        labels=["initial", "optimized"],
        rows=1,
        figw=12,
        legend_kw=dict(loc="lower right"),
    )
    plt.savefig("Xsection_comparison.pdf")


def test_plot_3d():
    """Plot initial and optimized boundary in 3D."""
    eq0 = Equilibrium.load("eq_initial.h5")
    eq1 = Equilibrium.load("eq_optimized.h5")
    plt.rcParams["figure.constrained_layout.use"] = True

    legend_list = ["initial", "optimized"]
    eq_list = [eq0, eq1]
    scale_list = [2, 4]

    for eq, legend, scale in zip(eq_list, legend_list, scale_list):
        plt.figure()
        theta_grid = np.linspace(0, 2 * np.pi, 300)
        zeta_grid = np.linspace(0, 2 * np.pi, 300)
        grid = LinearGrid(rho=1.0, theta=theta_grid, zeta=zeta_grid)
        # May want to turn off title in source code.
        fig = plot_3d(
            eq,
            name="|B|",
            grid=grid,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showaxislabels=False,
            update_traces=dict(colorbar=dict(tickfont=dict(size=75))),
        )

        config = {
            "toImageButtonOptions": {
                "filename": f"modB_3d_{legend}",
                "format": "svg",
                "scale": scale,
            }
        }
        save_path_html = os.getcwd() + f"/modB_3d_{legend}.html"
        fig.write_html(
            save_path_html, config=config, include_plotlyjs=True, full_html=True
        )
        plt.close()


def test_plot_binormal_drift():
    """Move this code into the relevant test to get the plots."""
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 17})
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 17
    plt.rcParams["ytick.labelsize"] = 17
    fig = bounce.check_points(  # noqa: F821
        points,  # noqa: F821
        pitch_inv,  # noqa: F821
        plot=True,
        klabel=r"$1/(\lambda B_0)$",
        k_transparency=0.25,
        show=False,
        vlabel=r"$\vert B \vert / B_0$",
        legend_kwargs=dict(
            loc="lower right", labelspacing=0.1, framealpha=1, borderpad=0.3
        ),
        markersize=(plt.rcParams["lines.markersize"] ** 2) * 1.5,
        title="",
        linewidth=2.5,
    )
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    plt.savefig("bavg_drift_field.pdf")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(
        pitch_inv,  # noqa: F821
        drift_analytic,  # noqa: F821
        label="model",
        color="black",
        lw=4,
    )

    ax.plot(
        pitch_inv,  # noqa: F821
        drift_numerical,  # noqa: F821
        label="computation",
        color="tab:orange",
        lw=2,
        linestyle="--",
    )
    ax.set_xlabel(r"$1/(\lambda B_0)$")
    ax.set_ylabel(r"1/seconds")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    plt.legend()
    plt.savefig("bavg_drift.pdf")
    plt.close()


keys = [
    "kappa_g",
    "|grad(rho)|",
    "|e_alpha|r,p|",
    "|B|",
    "|B|_r|v,p",
    "B^phi_r|v,p",
    "gbdrift",
]


def _err_PEST(data_PEST, data):
    return {k: np.abs(data[k] - data_PEST[k]).max() for k in keys}


def _err_append(err_PEST_1, err_PEST_2):
    if err_PEST_1 is None:
        return err_PEST_2
    return {k: np.append(err_PEST_1[k], err_PEST_2[k]) for k in keys}


@pytest.mark.parametrize(
    "name, upscale, tol",
    product(["W7-X", "NCSX"], np.array([1, 2, 3, 4]), [1e-10, 5e-7]),
)
def test_PEST_convergence_run(name, upscale, tol, maxiter=30):
    """Generate data for PEST basis convergence."""
    eq = get(name)
    eq_PEST = eq.to_sfl(
        L=eq.L * upscale,
        M=eq.M * upscale,
        N=eq.N * upscale,
        copy=True,
        tol=tol,
        maxiter=maxiter,
    )
    eq.change_resolution(
        L_grid=eq_PEST.L_grid, M_grid=eq_PEST.M_grid, N_grid=eq_PEST.N_grid
    )

    grid_PEST = LinearGrid(
        rho=np.linspace(0.1, 1, 20), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
    )
    grid_DESC = Grid(
        eq.map_coordinates(
            grid_PEST.nodes, ("rho", "theta_PEST", "zeta"), tol=tol, maxiter=maxiter
        )
    )
    data_PEST = eq_PEST.compute(keys, grid_PEST)
    data = eq.compute(keys, grid_DESC)
    err = _err_PEST(data_PEST, data)

    grid_PEST = LinearGrid(rho=0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    grid_DESC = Grid(
        eq.map_coordinates(
            grid_PEST.nodes, ("rho", "theta_PEST", "zeta"), tol=tol, maxiter=maxiter
        )
    )
    data_PEST = eq_PEST.compute(keys, grid_PEST)
    data = eq.compute(keys, grid_DESC)
    axis_err = _err_PEST(data_PEST, data)

    with open(f"{name}_{upscale}_{tol}.pkl", "wb") as file:
        pickle.dump(
            {"eq": eq, "eq_PEST": eq_PEST, "err": err, "axis_err": axis_err},
            file,
        )


def _plot_PEST_convergence(plot_data, keys, data_index, filename):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    lines, labels = [], []

    for i, ax in enumerate(axs):
        data = plot_data[i]
        err = data["err"]

        for k in keys:
            label_text = data_index["desc.equilibrium.equilibrium.Equilibrium"][k][
                "label"
            ]
            x_vals = np.arange(1, err[k].size + 1) ** 3
            line = ax.semilogy(x_vals, err[k], "--", marker="D")[0]
            ax.set_xticks(x_vals)

            if i == 0:
                lines.append(line)
                labels.append(rf"${label_text}$")

        ax.set_title(data["title"])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.75)

    fig.supxlabel(
        "Spectral resolution ratio "
        r"$(L M N)_{\vartheta, \phi} / (L M N)_{\theta, \zeta}$"
        r" of $(R, Z, \Lambda, \omega)$."
    )
    fig.supylabel("Absolute error")
    fig.legend(handles=lines, labels=labels, loc="center right", frameon=False)

    fig.tight_layout(rect=[0, -0.05, 0.8, 1])
    plt.savefig(filename)
    plt.close(fig)


@pytest.mark.parametrize("tol", [1e-10, 5e-7])
def test_plot_PEST_convergence(tol, plot_axis=False):
    """Saves PEST basis conversion plots for W7-X and NCSX."""
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "axes.grid": False,
        }
    )
    p = "desc.equilibrium.equilibrium.Equilibrium"
    data_index[p]["gbdrift"]["label"] = "(\\nabla \\vert B \\vert)_{\\mathrm{drift}}"

    configs = ["W7-X", "NCSX"]
    all_data = {}

    for name in configs:
        err, axis_err = None, None
        for i in range(1, 5):
            with open(f"{name}_{i}_{tol}.pkl", "rb") as file:
                pick = pickle.load(file)
            err = _err_append(err, pick["err"])
            axis_err = _err_append(axis_err, pick["axis_err"])
        all_data[name] = {"err": err, "axis_err": axis_err, "eq": get(name)}

    weq = all_data["W7-X"]["eq"]
    neq = all_data["NCSX"]["eq"]

    a1 = {
        "err": all_data["W7-X"]["err"],
        "eq": weq,
        "title": rf"W7-X $(L M N)_{{\theta, \zeta}} = ({weq.L},{weq.M},{weq.N})$",
    }
    a2 = {
        "err": all_data["NCSX"]["err"],
        "eq": neq,
        "title": rf"NCSX $(L M N)_{{\theta, \zeta}} = ({neq.L},{neq.M},{neq.N})$",
    }
    _plot_PEST_convergence(
        [a1, a2], keys, data_index, f"plot_PEST_convergence_{tol}.pdf"
    )

    if plot_axis:
        a1 = {
            "err": all_data["W7-X"]["axis_err"],
            "eq": weq,
            "title": rf"W7-X $(L,M,N)_{{\theta, \zeta}} = ({weq.L},{weq.M},{weq.N})$",
        }
        a2 = {
            "err": all_data["NCSX"]["axis_err"],
            "eq": neq,
            "title": rf"NCSX $(L,M,N)_{{\theta, \zeta}} = ({neq.L},{neq.M},{neq.N})$",
        }
        _plot_PEST_convergence(
            [a1, a2], keys, data_index, f"plot_PEST_convergence_{tol}_axis.pdf"
        )
