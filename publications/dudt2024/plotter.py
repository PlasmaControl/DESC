"""Script for creating plots in dudt2024."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import desc.io
from desc.basis import DoubleFourierSeries
from desc.grid import LinearGrid
from desc.magnetic_fields import OmnigenousField
from desc.transform import Transform

save = True
model = True
fields = True
confinement = True

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 14
green = "#1b9e77"
orange = "#d95f02"
purple = "#7570b3"
pink = "#e7298a"
colormap = "plasma"

eq_pol = desc.io.load("publications/dudt2024/poloidal.h5")[-1]
eq_tor = desc.io.load("publications/dudt2024/toroidal.h5")[-1]
eq_hel = desc.io.load("publications/dudt2024/helical.h5")[-1]
eq_pol_qs = desc.io.load("publications/dudt2024/poloidal_qs.h5")[-1]
eq_tor_qs = desc.io.load("publications/dudt2024/toroidal_qs.h5")[-1]
eq_hel_qs = desc.io.load("publications/dudt2024/helical_qs.h5")[-1]


def interp_helper(y, idx=np.array([], dtype=int)):
    """Interpolate NaNs or values at idx."""
    x = lambda z: z.nonzero()[0]
    nans = np.isnan(y)
    nans[idx] = True
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


# model ===============================================================================

if model:
    iota = 0.25
    field = OmnigenousField(
        L_B=0,
        M_B=4,
        L_x=0,
        M_x=1,
        N_x=1,
        NFP=1,
        helicity=(0, 1),
        B_lm=np.array([0.8, 0.9, 1.1, 1.2]),
        x_lmn=np.array([0, -np.pi / 8, 0, np.pi / 8, 0, np.pi / 4]),
    )
    z = np.linspace(0, np.pi / 2, num=field.M_B)
    grid = LinearGrid(theta=100, zeta=101, endpoint=True)
    data = field.compute(["|B|", "theta_B"], grid=grid, iota=iota)
    eta = data["eta"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
    theta = (
        data["theta_B"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
    )
    zeta = data["zeta_B"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
    B = data["|B|"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
    fig, (ax0, ax1, ax2) = plt.subplots(
        ncols=3, figsize=(18, 6), sharex=False, sharey=False
    )
    # plot a
    ax0.plot(
        eta[:, 0],
        B[:, 0],
        color=purple,
        linestyle="-",
        lw=6,
        label=r"spline interpolation",
    )
    ax0.plot(
        z,
        field.B_lm,
        color=orange,
        linestyle="",
        marker="o",
        ms=12,
        label=r"$B(\eta)$ parameters",
    )
    ax0.set_xlim([-np.pi / 2, np.pi / 2])
    ax0.set_xticks(
        [-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2]
    )
    ax0.set_xticklabels(
        [
            r"$-\pi/2$",
            r"$-\pi/3$",
            r"$-\pi/6$",
            r"$0$",
            r"$\pi/6$",
            r"$\pi/3$",
            r"$\pi/2$",
        ]
    )
    ax0.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax0.set_xlabel(r"$\eta$")
    ax0.set_ylabel(r"$|\mathbf{B}|$")
    ax0.legend(loc="upper center")
    # plot b
    ax1.plot(zeta[:, 0], B[:, 0], color=green, lw=6, label=r"$\alpha=0$")
    ax1.plot(zeta[:, 25], B[:, 25], color=orange, lw=6, label=r"$\alpha=\pi/2$")
    ax1.plot(zeta[:, 50], B[:, 50], color=purple, lw=6, label=r"$\alpha=\pi$")
    ax1.plot(zeta[:, 75], B[:, 75], color=pink, lw=6, label=r"$\alpha=3\pi/2$")
    ax1.set_xlim([0, 2 * np.pi])
    ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax1.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax1.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax1.set_xlabel(r"$h$")
    ax1.set_ylabel(r"$|\mathbf{B}|$")
    ax1.legend(loc="upper center")
    # plot c
    theta3 = np.vstack((theta - 2 * np.pi, theta, theta + 2 * np.pi))
    zeta3 = np.tile(zeta, (3, 1))
    B3 = np.tile(B, (3, 1))
    div = make_axes_locatable(ax2)
    im2 = ax2.contour(zeta3, theta3, B3, norm=Normalize(), levels=20, cmap=colormap)
    cax = div.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax, ticks=[0.8, 0.9, 1.0, 1.1, 1.2])
    cbar.update_ticks()
    arr = np.linspace(0, 2 * np.pi)
    ax2.plot(arr, iota * arr + 0, color=green, linestyle=":", lw=6, label=r"$\alpha=0$")
    ax2.plot(
        arr,
        iota * arr + np.pi / 2,
        color=orange,
        linestyle=":",
        lw=6,
        label=r"$\alpha=\pi/2$",
    )
    ax2.plot(
        arr,
        iota * arr + np.pi,
        color=purple,
        linestyle=":",
        lw=6,
        label=r"$\alpha=\pi$",
    )
    ax2.plot(
        arr,
        iota * arr + 3 * np.pi / 2,
        color=pink,
        linestyle=":",
        lw=6,
        label=r"$\alpha=3\pi/2$",
    )
    ax2.set_xlim([0, 2 * np.pi])
    ax2.set_ylim([0, 2 * np.pi])
    ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax2.set_xlabel(r"$\zeta_B$")
    ax2.set_ylabel(r"$\theta_B$")
    fig.tight_layout()
    if save:
        plt.savefig("publications/dudt2024/model.png")
        plt.savefig("publications/dudt2024/model.eps")
    else:
        plt.show()

# Boozer surfaces =====================================================================

if fields:
    fig, ax = plt.subplots(
        nrows=3, ncols=2, figsize=(18, 24), sharex=False, sharey=True
    )
    cax_kwargs = {"size": "5%", "pad": 0.05}
    props = dict(boxstyle="round", facecolor="w", alpha=1.0)
    # poloidal
    NFP = 2
    grid = LinearGrid(M=32, N=32, NFP=NFP, sym=False, rho=1.0)
    grid_plot = LinearGrid(
        theta=101, zeta=101, NFP=NFP, sym=False, endpoint=True, rho=1.0
    )
    zz = (
        grid_plot.nodes[:, 2]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    tt = (
        grid_plot.nodes[:, 1]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    basis = DoubleFourierSeries(M=16, N=16, sym="cos", NFP=NFP)
    B_transform = Transform(grid_plot, basis)
    # omnigenous
    data = eq_pol.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
    iota = grid.compress(data["iota"])
    BB = B_transform.transform(data["|B|_mn"]).reshape(
        (grid_plot.num_theta, grid_plot.num_zeta), order="F"
    )
    zeta = np.linspace(0, 2 * np.pi, 101)
    theta = iota * zeta
    contour_kwargs = {}
    contour_kwargs["norm"] = Normalize()
    contour_kwargs["levels"] = np.linspace(np.nanmin(BB), np.nanmax(BB), 20)
    contour_kwargs["cmap"] = colormap
    contour_kwargs["extend"] = "both"
    div = make_axes_locatable(ax[0, 0])
    im = ax[0, 0].contour(zz, tt, BB, **contour_kwargs)
    cax = div.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax, format="%.2f")
    cbar.update_ticks()
    ax[0, 0].plot(zeta, theta, color="k", ls="--", lw=2)
    ax[0, 0].set_title(r"Omnigenity")
    ax[0, 0].set_ylabel(r"$\theta_{Boozer}$")
    ax[0, 0].set_xlim([0, 2 * np.pi / NFP])
    ax[0, 0].set_ylim([0, 2 * np.pi])
    ax[0, 0].set_xticks([0, np.pi / NFP, 2 * np.pi / NFP])
    ax[0, 0].set_yticks([0, np.pi, 2 * np.pi])
    ax[0, 0].set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    ax[0, 0].set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax[0, 0].text(
        0.05,
        0.95,
        r"OP",
        transform=ax[0, 0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    # quasi-symmetric
    NFP = 1
    grid = LinearGrid(M=32, N=32, NFP=NFP, sym=False, rho=1.0)
    grid_plot = LinearGrid(
        theta=101, zeta=101, NFP=NFP, sym=False, endpoint=True, rho=1.0
    )
    zz = (
        grid_plot.nodes[:, 2]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    tt = (
        grid_plot.nodes[:, 1]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    basis = DoubleFourierSeries(M=16, N=16, sym="cos", NFP=NFP)
    B_transform = Transform(grid_plot, basis)
    data = eq_pol_qs.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
    iota = grid.compress(data["iota"])
    BB = B_transform.transform(data["|B|_mn"]).reshape(
        (grid_plot.num_theta, grid_plot.num_zeta), order="F"
    )
    zeta = np.linspace(0, 2 * np.pi, 101)
    theta = iota * zeta
    contour_kwargs = {}
    contour_kwargs["norm"] = Normalize()
    contour_kwargs["levels"] = np.linspace(np.nanmin(BB), np.nanmax(BB), 20)
    contour_kwargs["cmap"] = colormap
    contour_kwargs["extend"] = "both"
    div = make_axes_locatable(ax[0, 1])
    im = ax[0, 1].contour(zz, tt, BB, **contour_kwargs)
    cax = div.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax, format="%.2f")
    cbar.update_ticks()
    ax[0, 1].plot(zeta, theta, color="k", ls="--", lw=2)
    ax[0, 1].set_title(r"Quasi-Symmetry")
    ax[0, 1].set_xlim([0, 2 * np.pi / NFP])
    ax[0, 1].set_ylim([0, 2 * np.pi])
    ax[0, 1].set_xticks([0, np.pi / NFP, 2 * np.pi / NFP])
    ax[0, 1].set_yticks([0, np.pi, 2 * np.pi])
    ax[0, 1].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax[0, 1].text(
        0.05,
        0.95,
        r"QP",
        transform=ax[0, 1].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    # helical
    NFP = 5
    grid = LinearGrid(M=32, N=32, NFP=NFP, sym=False, rho=1.0)
    grid_plot = LinearGrid(
        theta=101, zeta=101, NFP=NFP, sym=False, endpoint=True, rho=1.0
    )
    zz = (
        grid_plot.nodes[:, 2]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    tt = (
        grid_plot.nodes[:, 1]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    basis = DoubleFourierSeries(M=16, N=16, sym="cos", NFP=NFP)
    B_transform = Transform(grid_plot, basis)
    # omnigenous
    data = eq_hel.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
    iota = grid.compress(data["iota"])
    BB = B_transform.transform(data["|B|_mn"]).reshape(
        (grid_plot.num_theta, grid_plot.num_zeta), order="F"
    )
    zeta = np.linspace(0, 2 * np.pi, 101)
    theta = iota * zeta
    contour_kwargs = {}
    contour_kwargs["norm"] = Normalize()
    contour_kwargs["levels"] = np.linspace(np.nanmin(BB), np.nanmax(BB), 20)
    contour_kwargs["cmap"] = colormap
    contour_kwargs["extend"] = "both"
    div = make_axes_locatable(ax[1, 0])
    im = ax[1, 0].contour(zz, tt, BB, **contour_kwargs)
    cax = div.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax, format="%.2f")
    cbar.update_ticks()
    ax[1, 0].plot(zeta, theta, color="k", ls="--", lw=2)
    ax[1, 0].set_ylabel(r"$\theta_{Boozer}$")
    ax[1, 0].set_xlim([0, 2 * np.pi / NFP])
    ax[1, 0].set_ylim([0, 2 * np.pi])
    ax[1, 0].set_xticks([0, np.pi / NFP, 2 * np.pi / NFP])
    ax[1, 0].set_yticks([0, np.pi, 2 * np.pi])
    ax[1, 0].set_xticklabels([r"$0$", r"$\pi/5$", r"$2\pi/5$"])
    ax[1, 0].text(
        0.05,
        0.95,
        r"OH",
        transform=ax[1, 0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    # quasi-symmetric
    data = eq_hel_qs.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
    iota = grid.compress(data["iota"])
    BB = B_transform.transform(data["|B|_mn"]).reshape(
        (grid_plot.num_theta, grid_plot.num_zeta), order="F"
    )
    zeta = np.linspace(0, 2 * np.pi, 101)
    theta = iota * zeta
    contour_kwargs = {}
    contour_kwargs["norm"] = Normalize()
    contour_kwargs["levels"] = np.linspace(np.nanmin(BB), np.nanmax(BB), 20)
    contour_kwargs["cmap"] = colormap
    contour_kwargs["extend"] = "both"
    div = make_axes_locatable(ax[1, 1])
    im = ax[1, 1].contour(zz, tt, BB, **contour_kwargs)
    cax = div.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax, format="%.2f")
    cbar.update_ticks()
    ax[1, 1].plot(zeta, theta, color="k", ls="--", lw=2)
    ax[1, 1].set_xlim([0, 2 * np.pi / NFP])
    ax[1, 1].set_ylim([0, 2 * np.pi])
    ax[1, 1].set_xticks([0, np.pi / NFP, 2 * np.pi / NFP])
    ax[1, 1].set_yticks([0, np.pi, 2 * np.pi])
    ax[1, 1].set_xticklabels([r"$0$", r"$\pi/5$", r"$2\pi/5$"])
    ax[1, 1].text(
        0.05,
        0.95,
        r"QH",
        transform=ax[1, 1].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    # toroidal
    NFP = 1
    grid = LinearGrid(M=32, N=32, NFP=NFP, sym=False, rho=1.0)
    grid_plot = LinearGrid(
        theta=101, zeta=101, NFP=NFP, sym=False, endpoint=True, rho=1.0
    )
    zz = (
        grid_plot.nodes[:, 2]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    tt = (
        grid_plot.nodes[:, 1]
        .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
        .squeeze()
    )
    basis = DoubleFourierSeries(M=16, N=16, sym="cos", NFP=NFP)
    B_transform = Transform(grid_plot, basis)
    # omnigenous
    data = eq_tor.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
    iota = grid.compress(data["iota"])
    BB = B_transform.transform(data["|B|_mn"]).reshape(
        (grid_plot.num_theta, grid_plot.num_zeta), order="F"
    )
    zeta = np.linspace(0, 2 * np.pi, 101)
    theta = iota * zeta
    contour_kwargs = {}
    contour_kwargs["norm"] = Normalize()
    contour_kwargs["levels"] = np.linspace(np.nanmin(BB), np.nanmax(BB), 20)
    contour_kwargs["cmap"] = colormap
    contour_kwargs["extend"] = "both"
    div = make_axes_locatable(ax[2, 0])
    im = ax[2, 0].contour(zz, tt, BB, **contour_kwargs)
    cax = div.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax, format="%.2f")
    cbar.update_ticks()
    ax[2, 0].plot(zeta, theta, color="k", ls="--", lw=2)
    ax[2, 0].set_xlabel(r"$\zeta_{Boozer}$")
    ax[2, 0].set_ylabel(r"$\theta_{Boozer}$")
    ax[2, 0].set_xlim([0, 2 * np.pi / NFP])
    ax[2, 0].set_ylim([0, 2 * np.pi])
    ax[2, 0].set_xticks([0, np.pi / NFP, 2 * np.pi / NFP])
    ax[2, 0].set_yticks([0, np.pi, 2 * np.pi])
    ax[2, 0].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax[2, 0].text(
        0.05,
        0.95,
        r"OT",
        transform=ax[2, 0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    # quasi-symmetric
    data = eq_tor_qs.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
    iota = grid.compress(data["iota"])
    BB = B_transform.transform(data["|B|_mn"]).reshape(
        (grid_plot.num_theta, grid_plot.num_zeta), order="F"
    )
    zeta = np.linspace(0, 2 * np.pi, 101)
    theta = iota * zeta
    contour_kwargs = {}
    contour_kwargs["norm"] = Normalize()
    contour_kwargs["levels"] = np.linspace(np.nanmin(BB), np.nanmax(BB), 20)
    contour_kwargs["cmap"] = colormap
    contour_kwargs["extend"] = "both"
    div = make_axes_locatable(ax[2, 1])
    im = ax[2, 1].contour(zz, tt, BB, **contour_kwargs)
    cax = div.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax, format="%.2f")
    cbar.update_ticks()
    ax[2, 1].plot(zeta, theta, color="k", ls="--", lw=2)
    ax[2, 1].set_xlabel(r"$\zeta_{Boozer}$")
    ax[2, 1].set_xlim([0, 2 * np.pi / NFP])
    ax[2, 1].set_ylim([0, 2 * np.pi])
    ax[2, 1].set_xticks([0, np.pi / NFP, 2 * np.pi / NFP])
    ax[2, 1].set_yticks([0, np.pi, 2 * np.pi])
    ax[2, 1].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax[2, 1].text(
        0.05,
        0.95,
        r"QA",
        transform=ax[2, 1].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    fig.tight_layout()
    if save:
        plt.savefig("publications/dudt2024/fields.png")
        plt.savefig("publications/dudt2024/fields.eps")
    else:
        plt.show()

# confinement ====================================================================

if confinement:
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(8, 14))
    # effective ripple
    eps_pol = interp_helper(np.loadtxt("publications/dudt2024/neo_out.poloidal")[:, 1])
    eps_hel = interp_helper(np.loadtxt("publications/dudt2024/neo_out.helical")[:, 1])
    eps_tor = interp_helper(np.loadtxt("publications/dudt2024/neo_out.toroidal")[:, 1])
    eps_pol_qs = interp_helper(
        np.loadtxt("publications/dudt2024/neo_out.poloidal_qs")[:, 1], idx=248
    )
    eps_hel_qs = interp_helper(
        np.loadtxt("publications/dudt2024/neo_out.helical_qs")[:, 1]
    )
    eps_tor_qs = interp_helper(
        np.loadtxt("publications/dudt2024/neo_out.toroidal_qs")[:, 1]
    )
    eps_w7x = interp_helper(np.loadtxt("publications/dudt2024/neo_out.w7x")[:, 1])
    s = np.linspace(0, 1, eps_pol.size + 1)[1:]
    ax0.semilogy(s, eps_pol, color=purple, linestyle="-", lw=4, label="OP")
    ax0.semilogy(s, eps_hel, color=orange, linestyle="-", lw=4, label="OH")
    ax0.semilogy(s, eps_tor, color=green, linestyle="-", lw=4, label="OT")
    ax0.semilogy(s, eps_pol_qs, color=purple, linestyle=":", lw=4, label="QP")
    ax0.semilogy(s, eps_hel_qs, color=orange, linestyle=":", lw=4, label="QH")
    ax0.semilogy(s, eps_tor_qs, color=green, linestyle=":", lw=4, label="QA")
    s = np.linspace(0, 1, eps_w7x.size + 1)[1:]
    ax0.semilogy(s, eps_w7x, color="k", linestyle="--", lw=4, label="W7-X")
    handles, labels = ax0.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2, 5, 6]
    ax0.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc=(0.12, 0.25),
        ncol=4,
    )
    ax0.set_xlim([0, 1])
    ax0.set_ylim([1e-8, 1e-2])
    ax0.set_xlabel(r"Normalized toroidal flux = $\rho^2$")
    ax0.set_ylabel(r"$\epsilon_{eff}^{3/2}$")
    # particle losses
    lost_pol = 1 - interp_helper(
        np.sum(
            np.loadtxt("publications/dudt2024/confined_fraction_poloidal.dat")[:, 1:3],
            axis=1,
        )
    )
    lost_hel = 1 - interp_helper(
        np.sum(
            np.loadtxt("publications/dudt2024/confined_fraction_helical.dat")[:, 1:3],
            axis=1,
        )
    )
    lost_tor = 1 - interp_helper(
        np.sum(
            np.loadtxt("publications/dudt2024/confined_fraction_toroidal.dat")[:, 1:3],
            axis=1,
        )
    )
    lost_pol_qs = 1 - interp_helper(
        np.sum(
            np.loadtxt("publications/dudt2024/confined_fraction_poloidal_qs.dat")[
                :, 1:3
            ],
            axis=1,
        )
    )
    lost_hel_qs = 1 - interp_helper(
        np.sum(
            np.loadtxt("publications/dudt2024/confined_fraction_helical_qs.dat")[
                :, 1:3
            ],
            axis=1,
        )
    )
    lost_tor_qs = 1 - interp_helper(
        np.sum(
            np.loadtxt("publications/dudt2024/confined_fraction_toroidal_qs.dat")[
                :, 1:3
            ],
            axis=1,
        )
    )
    lost_w7x = 1 - interp_helper(
        np.sum(
            np.loadtxt("publications/dudt2024/confined_fraction_w7x.dat")[:, 1:3],
            axis=1,
        )
    )
    t = np.loadtxt("publications/dudt2024/confined_fraction_poloidal.dat")[:, 0]
    ax1.loglog(t, lost_pol, color=purple, linestyle="-", lw=4, label="OP")
    ax1.loglog(t, lost_pol_qs, color=purple, linestyle=":", lw=4, label="QP")
    ax1.loglog(t, lost_hel, color=orange, linestyle="-", lw=4, label="OH")
    ax1.loglog(t, lost_hel_qs, color=orange, linestyle=":", lw=4, label="QH")
    ax1.loglog(t, lost_tor, color=green, linestyle="-", lw=4, label="OT")
    ax1.loglog(t, lost_tor_qs, color=green, linestyle=":", lw=4, label="QA")
    t = np.loadtxt("publications/dudt2024/confined_fraction_w7x.dat")[:, 0]
    ax1.loglog(t, lost_w7x, color="k", linestyle="--", lw=4, label="W7-X")
    ax1.legend(loc=(0.12, 0.85), ncol=4)
    ax1.set_xlim([1e-4, 2e-1])
    ax1.set_ylim([1e-3, 1e0])
    ax1.set_xlabel(r"Time (s)")
    ax1.set_ylabel(r"Fraction of alpha particles lost")
    ax1.text(
        0.55,
        0.2,
        r"No losses for QH and QA",
        transform=ax1.transAxes,
        verticalalignment="top",
    )
    fig.tight_layout()
    if save:
        plt.savefig("publications/dudt2024/confinement.png")
        plt.savefig("publications/dudt2024/confinement.eps")
    else:
        plt.show()

# =====================================================================================

print("Done!")
