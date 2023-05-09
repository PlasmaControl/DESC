"""Script for creating plots in dudt2023."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import desc.io
from desc.basis import DoubleFourierSeries
from desc.compute.utils import compress
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.transform import Transform


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 14
green = "#1b9e77"
orange = "#d95f02"
purple = "#7570b3"
colormap = "plasma"

labels = True
save = False

eq_pol = desc.io.load("publications/dudt2023/poloidal.h5")
eq_tor = desc.io.load("publications/dudt2023/toroidal.h5")
eq_hel = desc.io.load("publications/dudt2023/helical.h5")
eq_pol_qs = desc.io.load("publications/dudt2023/poloidal_qs.h5")
eq_tor_qs = desc.io.load("publications/dudt2023/toroidal_qs.h5")
eq_hel_qs = desc.io.load("publications/dudt2023/helical_qs.h5")

#### boundaries ####

colors = [purple, orange]
styles = ["-", "-"]
labels = ["Omnigenous", "Quasi-Symmetric"]
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 7), sharex=True, sharey=True)
# poloidal
grid_ax = LinearGrid(theta=1, zeta=6, NFP=1, rho=0.0, endpoint=True)
grid = LinearGrid(theta=100, zeta=6, NFP=1, rho=1.0, endpoint=True)
for i, eq in enumerate((eq_pol, eq_pol_qs)):
    coords_ax = eq.compute(["R", "Z"], grid=grid_ax)
    coords = eq.compute(["R", "Z"], grid=grid)
    R0 = coords_ax["R"]
    Z0 = coords_ax["Z"]
    R = coords["R"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    Z = coords["Z"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    ax[0].plot(R0, Z0, "ko", ms=6)
    for j in range(grid.num_zeta - 1):
        (line,) = ax[0].plot(
            R[:, -1, j],
            Z[:, -1, j],
            color=colors[i],
            linestyle=styles[i],
            lw=3,
        )
ax[0].set_xlabel(r"$R$ (m)")
ax[0].set_ylabel(r"$Z$ (m)")
ax[0].set_title(r"$M=0,~N=1$")
# helical
grid_ax = LinearGrid(theta=1, zeta=6, NFP=5, rho=0.0, endpoint=True)
grid = LinearGrid(theta=100, zeta=6, NFP=5, rho=1.0, endpoint=True)
for i, eq in enumerate((eq_hel, eq_hel_qs)):
    coords_ax = eq.compute(["R", "Z"], grid=grid_ax)
    coords = eq.compute(["R", "Z"], grid=grid)
    R0 = coords_ax["R"]
    Z0 = coords_ax["Z"]
    R = coords["R"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    Z = coords["Z"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    for j in range(grid.num_zeta - 1):
        (line,) = ax[1].plot(
            R[:, -1, j],
            Z[:, -1, j],
            color=colors[i],
            linestyle=styles[i],
            lw=3,
        )
        if j == 0:
            line.set_label(labels[i])
    if i == 1:
        ax[1].plot(R0, Z0, "ko", ms=6, label="Magnetic Axis")
ax[1].legend(loc="upper left", ncol=1)
ax[1].set_xlabel(r"$R$ (m)")
ax[1].set_title(r"$M=1,~N=5$")
# toroidal
grid_ax = LinearGrid(theta=1, zeta=6, NFP=1, rho=0.0, endpoint=True)
grid = LinearGrid(theta=100, zeta=6, NFP=1, rho=1.0, endpoint=True)
for i, eq in enumerate((eq_tor, eq_tor_qs)):
    coords_ax = eq.compute(["R", "Z"], grid=grid_ax)
    coords = eq.compute(["R", "Z"], grid=grid)
    R0 = coords_ax["R"]
    Z0 = coords_ax["Z"]
    R = coords["R"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    Z = coords["Z"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    ax[2].plot(R0, Z0, "ko", ms=6)
    for j in range(grid.num_zeta - 1):
        (line,) = ax[2].plot(
            R[:, -1, j],
            Z[:, -1, j],
            color=colors[i],
            linestyle=styles[i],
            lw=3,
        )
ax[2].set_xlabel(r"$R$ (m)")
ax[2].set_title(r"$M=1,~N=0$")
fig.tight_layout()
plt.show()
if save:
    plt.savefig("publications/dudt2023/boundaries.png")
    plt.savefig("publications/dudt2023/boundaries.eps")

#### Boozer surfaces ####

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=False, sharey=True)
cax_kwargs = {"size": "5%", "pad": 0.05}
props = dict(boxstyle="round", facecolor="w", alpha=1.0)
# poloidal
grid = LinearGrid(M=32, N=32, NFP=1, sym=False, rho=1.0)
grid_plot = LinearGrid(theta=101, zeta=101, NFP=1, sym=False, endpoint=True, rho=1.0)
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
basis = DoubleFourierSeries(M=16, N=16, sym="cos", NFP=1)
B_transform = Transform(grid_plot, basis)
# quasi-symmetric
data = eq_pol_qs.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
iota = compress(grid, data["iota"])
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
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax[0, 0].plot(zeta, theta, color="k", ls="--", lw=2)
ax[0, 0].set_ylabel(r"$\theta_{Boozer}$")
ax[0, 0].set_title(r"$M=0,~N=1$")
ax[0, 0].set_xlim([0, 2 * np.pi])
ax[0, 0].set_ylim([0, 2 * np.pi])
ax[0, 0].text(
   0.05,
   0.95,
   r"QP",
   transform=ax[0, 0].transAxes,
   fontsize=14,
   verticalalignment="top",
   bbox=props,
)
# omnigenous
data = eq_pol.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
iota = compress(grid, data["iota"])
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
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax[1, 0].plot(zeta, theta, color="k", ls="--", lw=2)
ax[1, 0].set_xlabel(r"$\zeta_{Boozer}$")
ax[1, 0].set_ylabel(r"$\theta_{Boozer}$")
ax[1, 0].set_xlim([0, 2 * np.pi])
ax[1, 0].set_ylim([0, 2 * np.pi])
ax[1, 0].text(
   0.05,
   0.95,
   r"QI",
   transform=ax[1, 0].transAxes,
   fontsize=14,
   verticalalignment="top",
   bbox=props,
)
# helical
grid = LinearGrid(M=32, N=32, NFP=5, sym=False, rho=1.0)
grid_plot = LinearGrid(theta=101, zeta=101, NFP=5, sym=False, endpoint=True, rho=1.0)
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
basis = DoubleFourierSeries(M=16, N=16, sym="cos", NFP=5)
B_transform = Transform(grid_plot, basis)
# quasi-symmetric
data = eq_hel_qs.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
iota = compress(grid, data["iota"])
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
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax[0, 1].plot(zeta, theta, color="k", ls="--", lw=2)
ax[0, 1].set_title(r"$M=1,~N=5$")
ax[0, 1].set_xlim([0, 2 * np.pi / 5])
ax[0, 1].set_ylim([0, 2 * np.pi])
ax[0, 1].text(
   0.05,
   0.95,
   r"QH",
   transform=ax[0, 1].transAxes,
   fontsize=14,
   verticalalignment="top",
   bbox=props,
)
# omnigenous
data = eq_hel.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
iota = compress(grid, data["iota"])
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
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax[1, 1].plot(zeta, theta, color="k", ls="--", lw=2)
ax[1, 1].set_xlabel(r"$\zeta_{Boozer}$")
ax[1, 1].set_xlim([0, 2 * np.pi / 5])
ax[1, 1].set_ylim([0, 2 * np.pi])
ax[1, 1].text(
   0.05,
   0.95,
   r"OH",
   transform=ax[1, 1].transAxes,
   fontsize=14,
   verticalalignment="top",
   bbox=props,
)
# toroidal
grid = LinearGrid(M=32, N=32, NFP=1, sym=False, rho=1.0)
grid_plot = LinearGrid(theta=101, zeta=101, NFP=1, sym=False, endpoint=True, rho=1.0)
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
basis = DoubleFourierSeries(M=16, N=16, sym="cos", NFP=1)
B_transform = Transform(grid_plot, basis)
# quasi-symmetric
data = eq_tor_qs.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
iota = compress(grid, data["iota"])
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
div = make_axes_locatable(ax[0, 2])
im = ax[0, 2].contour(zz, tt, BB, **contour_kwargs)
cax = div.append_axes("right", **cax_kwargs)
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax[0, 2].plot(zeta, theta, color="k", ls="--", lw=2)
ax[0, 2].set_title(r"$M=1,~N=0$")
ax[0, 2].set_xlim([0, 2 * np.pi])
ax[0, 2].set_ylim([0, 2 * np.pi])
ax[0, 2].text(
   0.05,
   0.95,
   r"QA",
   transform=ax[0, 2].transAxes,
   fontsize=14,
   verticalalignment="top",
   bbox=props,
)
# omnigenous
data = eq_tor.compute("|B|_mn", M_booz=16, N_booz=16, grid=grid)
iota = compress(grid, data["iota"])
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
div = make_axes_locatable(ax[1, 2])
im = ax[1, 2].contour(zz, tt, BB, **contour_kwargs)
cax = div.append_axes("right", **cax_kwargs)
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax[1, 2].plot(zeta, theta, color="k", ls="--", lw=2)
ax[1, 2].set_xlabel(r"$\zeta_{Boozer}$")
ax[1, 2].set_xlim([0, 2 * np.pi])
ax[1, 2].set_ylim([0, 2 * np.pi])
ax[1, 2].text(
   0.05,
   0.95,
   r"OT",
   transform=ax[1, 2].transAxes,
   fontsize=14,
   verticalalignment="top",
   bbox=props,
)
fig.tight_layout()
plt.show()
if save:
    plt.savefig("publications/dudt2023/fields.png")
    plt.savefig("publications/dudt2023/fields.eps")

#### effective ripple ####

fig, ax = plt.subplots(figsize=(7, 7))
s = np.linspace(0, 1, 256)[1:]
eps_pol = np.load("publications/dudt2023/poloidal.npy")
eps_hel = np.load("publications/dudt2023/helical.npy")
eps_tor = np.load("publications/dudt2023/toroidal.npy")
eps_pol_qs = np.load("publications/dudt2023/poloidal_qs.npy")
eps_hel_qs = np.load("publications/dudt2023/helical_qs.npy")
eps_tor_qs = np.load("publications/dudt2023/toroidal_qs.npy")
ax.semilogy(s, eps_pol, color=purple, linestyle="-", lw=4, label="QI")
ax.semilogy(s, eps_pol_qs, color=purple, linestyle=":", lw=4, label="QP")
ax.semilogy(s, eps_hel, color=orange, linestyle="-", lw=4, label="OH")
ax.semilogy(s, eps_hel_qs, color=orange, linestyle=":", lw=4, label="QH")
ax.semilogy(s, eps_tor, color=green, linestyle="-", lw=4, label="OT")
ax.semilogy(s, eps_tor_qs, color=green, linestyle=":", lw=4, label="QA")
ax.legend(loc=(0.2, 0.7), ncol=3)
ax.set_xlim([0, 1])
ax.set_ylim([1e-7, 1e-2])
ax.set_xlabel(r"Normalized toroidal flux = $\rho^2$")
ax.set_ylabel(r"$\epsilon_{eff}^{3/2}$")
fig.tight_layout()
plt.show()
if save:
    plt.savefig("publications/dudt2023/ripple.png")
    plt.savefig("publications/dudt2023/ripple.eps")
"""
#### model ####

helicity = (0, 1)
iota = 0.25
well_l = np.array([0.8, 0.9, 1.1, 1.2])
omni_lmn = np.array([0, -np.pi / 8, 0, np.pi / 8, 0, np.pi / 4])  # 0, sin, 0, const, 0, cos

eq = Equilibrium(iota=np.array([iota]), well_l=well_l, omni_lmn=omni_lmn)
z = np.linspace(0, np.pi / 2, num=well_l.size)
grid = LinearGrid(theta=100, zeta=101, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(eta,alpha)"], grid=grid, helicity=helicity)

eta = data["eta"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
theta = data["theta_B(eta,alpha)"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
zeta = data["zeta_B(eta,alpha)"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
B = data["|B|_omni"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()

fig, (ax0, ax1, ax2) = plt.subplots(
    ncols=3, figsize=(18, 6), sharex=False, sharey=False
)
# plot a
ax0.plot(
    eta[:, 0],
    B[:, 0],
    color=color3,
    linestyle="-",
    lw=6,
    label=r"spline interpolation",
)
ax0.plot(z, well_l, color=color1, linestyle="", marker="o", ms=12, label=r"$x_l$")
ax0.set_xlim([-np.pi / 2, np.pi / 2])
ax0.set_xticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2])
ax0.set_xticklabels(
    [r"$-\pi/2$", r"$-\pi/3$", r"$-\pi/6$", r"$0$", r"$\pi/6$", r"$\pi/3$", r"$\pi/2$"]
)
ax0.set_xlabel(r"$\eta$")
ax0.set_ylabel(r"$|\mathbf{B}|$")
ax0.legend(loc="upper center")
# plot b
ax1.plot(zeta[:, 0], B[:, 0], color=color0, lw=6, label=r"$\alpha=0$")
ax1.plot(zeta[:, 25], B[:, 25], color=color1, lw=6, label=r"$\alpha=\pi/2$")
ax1.plot(zeta[:, 50], B[:, 50], color=color2, lw=6, label=r"$\alpha=\pi$")
ax1.plot(zeta[:, 75], B[:, 75], color=color3, lw=6, label=r"$\alpha=3\pi/2$")
ax1.set_xlim([0, 2 * np.pi])
ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax1.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax1.set_xlabel(r"$\zeta_B$")
ax1.set_ylabel(r"$|\mathbf{B}|$")
ax1.legend(loc="upper center")
# plot c
theta3 = np.vstack((theta - 2 * np.pi, theta, theta + 2 * np.pi))
zeta3 = np.tile(zeta, (3, 1))
B3 = np.tile(B, (3, 1))
div = make_axes_locatable(ax2)
im2 = ax2.contour(zeta3, theta3, B3, norm=Normalize(), levels=20, cmap=colormap)
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im2, cax=cax, ticks=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
cbar.update_ticks()
arr = np.linspace(0, 2 * np.pi)
ax2.plot(arr, iota * arr + 0, color=color0, linestyle=":", lw=6, label=r"$\alpha=0$")
ax2.plot(
    arr,
    iota * arr + np.pi / 2,
    color=color1,
    linestyle=":",
    lw=6,
    label=r"$\alpha=\pi/2$",
)
ax2.plot(
    arr,
    iota * arr + np.pi,
    color=color2,
    linestyle=":",
    lw=6,
    label=r"$\alpha=\pi$",
)
ax2.plot(
    arr,
    iota * arr + 3 * np.pi / 2,
    color=color3,
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
plt.show()
if save:
    plt.savefig("model.png")
    plt.savefig("model.eps")

#### examples ####

well_l = np.array([0.8, 0.9, 1.1, 1.2])
omni_lmn = np.array([0, 0, 0, np.pi / 6, np.pi / 7, np.pi / 8])

# general field
iota = 1 / (1 + np.sqrt(5))
basis = DoubleFourierSeries(M=4, N=4, NFP=1, sym=False)
grid = LinearGrid(theta=101, zeta=100, endpoint=True)
transform = Transform(grid, basis)
B_mn = np.zeros((basis.num_modes,))
B_mn[np.nonzero((basis.modes == [0, -2, 0]).all(axis=1))[0]] = -np.pi / 4
B_mn[np.nonzero((basis.modes == [0, 0, 4]).all(axis=1))[0]] = np.pi / 6
B_mn[np.nonzero((basis.modes == [0, 1, 1]).all(axis=1))[0]] = np.pi / 3
B_mn[np.nonzero((basis.modes == [0, 2, -3]).all(axis=1))[0]] = -np.pi / 5
B = (
    transform.transform(B_mn)
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
theta = grid.nodes[:, 1].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
zeta = grid.nodes[:, 2].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap=colormap)
arr = np.linspace(0, 2 * np.pi)
ax.plot(arr, iota * arr, color="k", linestyle="--", lw=2)
ax.set_xlim([0, 2 * np.pi])
ax.set_ylim([0, 2 * np.pi])
if labels:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_title(r"General Magnetic Field")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
if save:
    plt.savefig("general.png")
    plt.savefig("general.eps")

# omnigenity poloidal (QI)
helicity = (0, 1)
iota = 1 / (1 + np.sqrt(5))
eq = Equilibrium(iota=np.array([iota]), well_l=well_l, omni_lmn=omni_lmn)
grid = LinearGrid(theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(alpha,eta)"], grid=grid, helicity=helicity)
theta = (
    data["theta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
zeta = (
    data["zeta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
B = data["|B|_omni"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
theta = np.vstack((theta - 2 * np.pi, theta, theta + 2 * np.pi))
zeta = np.tile(zeta, (3, 1))
B = np.tile(B, (3, 1))
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap=colormap)
arr = np.linspace(0, 2 * np.pi)
ax.plot(arr, iota * arr, color="k", linestyle="--", lw=2)
ax.set_xlim([0, 2 * np.pi])
ax.set_ylim([0, 2 * np.pi])
if labels:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_title(r"Omnigeneous with $M=0$, $N=1$ (QI)")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
if save:
    plt.savefig("omnigenity_M0_N1.png")
    plt.savefig("omnigenity_M0_N1.eps")

# omnigenity helical
well_l = np.array([0.8, 0.9, 1.1, 1.2])
omni_lmn = np.array([0, 0, 0, np.pi / 6, np.pi / 7, np.pi / 8])
NFP = 5
helicity = (1, NFP)
iota = 1.2  # -1 / (1 + np.sqrt(5))
eq = Equilibrium(NFP=NFP, iota=np.array([iota]), well_l=well_l, omni_lmn=omni_lmn)
grid = LinearGrid(NFP=NFP, theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(eta,alpha)"], grid=grid, helicity=helicity)
theta = (
    data["theta_B(eta,alpha)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
zeta = (
    data["zeta_B(eta,alpha)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
B = data["|B|_omni"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
dtheta = 4 * np.pi * iota / (np.sqrt(2) * (1 - iota))
dzeta = 4 * np.pi / (np.sqrt(2) * (1 - iota))
theta = np.block(
    [
        [theta, theta - dtheta],
        [theta + 2 * np.pi / np.sqrt(2), theta + 2 * np.pi / np.sqrt(2) - dtheta],
    ]
)
zeta = np.block(
    [
        [zeta, zeta - dzeta],
        [zeta + 2 * np.pi / np.sqrt(2), zeta + 2 * np.pi / np.sqrt(2) - dzeta],
    ]
)
B = np.tile(B, (2, 2))
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap=colormap)
arr = np.linspace(0, 2 * np.pi)
ax.plot(arr, iota * arr + 2 * np.pi, color="k", linestyle="--", lw=2)
ax.set_xlim([0, 2 * np.pi])
ax.set_ylim([0, 2 * np.pi])
if labels:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_title(r"Omnigeneous with $M=1$, $N=1$")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
if save:
    plt.savefig("omnigenity_M1_N1.png")
    plt.savefig("omnigenity_M1_N1.eps")

# omnigenity toroidal
helicity = (1, 0)
iota = 1 + np.sqrt(5)
eq = Equilibrium(iota=np.array([iota]), well_l=well_l, omni_lmn=omni_lmn)
grid = LinearGrid(theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(alpha,eta)"], grid=grid, helicity=helicity)
theta = (
    data["theta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
zeta = (
    data["zeta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
B = data["|B|_omni"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
theta = np.tile(theta + 2 * np.pi, (1, 2))
zeta = np.hstack((zeta, zeta + 2 * np.pi))
B = np.tile(B, (1, 2))
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap=colormap)
arr = np.linspace(0, 2 * np.pi)
ax.plot(arr, iota * arr, color="k", linestyle="--", lw=2)
ax.set_xlim([0, 2 * np.pi])
ax.set_ylim([0, 2 * np.pi])
if labels:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_title(r"Omnigeneous with $M=1$, $N=0$")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
if save:
    plt.savefig("omnigenity_M1_N0.png")
    plt.savefig("omnigenity_M1_N0.eps")

well_l = np.array([0.8, 0.9, 1.1, 1.2])
omni_lmn = np.array([0, 0, 0, 0, 0, -np.pi / 5])  # 0, sin, 0, const, 0, cos

# quasi-symmetry poloidal (QP)
helicity = (0, 1)
iota = 1 / (1 + np.sqrt(5))
eq = Equilibrium(iota=np.array([iota]), well_l=well_l, omni_lmn=omni_lmn)
grid = LinearGrid(theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(alpha,eta)"], grid=grid, helicity=helicity)
theta = (
    data["theta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
zeta = (
    data["zeta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
B = data["|B|_omni"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
theta3 = np.vstack((theta - 2 * np.pi, theta, theta + 2 * np.pi))
zeta3 = np.tile(zeta, (3, 1))
B3 = np.tile(B, (3, 1))
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.contour(zeta3, theta3, B3, norm=Normalize(), levels=20, cmap=colormap)
arr = np.linspace(0, 2 * np.pi)
ax.plot(arr, iota * arr, color="k", linestyle="--", lw=2)
ax.set_xlim([0, 2 * np.pi])
ax.set_ylim([0, 2 * np.pi])
if labels:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_title(r"Quasi-Symmetric with $M=0$, $N=1$ (QP)")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
if save:
    plt.savefig("quasisymmetry_M0_N1.png")
    plt.savefig("quasisymmetry_M0_N1.eps")

# quasi-symmetry helical (QH)
NFP = 5
helicity = (1, NFP)
iota = 1.2  # -1 / (1 + np.sqrt(5))
eq = Equilibrium(NFP=NFP, iota=np.array([iota]), well_l=well_l, omni_lmn=omni_lmn)
grid = LinearGrid(NFP=NFP, theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(eta,alpha)"], grid=grid, helicity=helicity)
theta = (
    data["theta_B(eta,alpha)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
zeta = (
    data["zeta_B(eta,alpha)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
B = data["|B|_omni"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
dtheta = 2 * np.pi
dzeta = 2 * np.pi * helicity[0] / helicity[1]
theta = np.block([[theta, theta], [theta - dtheta, theta - dtheta]])
zeta = np.block([[zeta - dzeta, zeta], [zeta - 2 * dzeta, zeta - dzeta]])
B = np.tile(B, (2, 2))
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap=colormap)
arr = np.linspace(0, 2 * np.pi)
ax.plot(arr, iota * arr, color="k", linestyle="--", lw=2)
ax.plot(arr, iota * arr + 4, color="k", linestyle="--", lw=2)
ax.set_xlim([0, 2 * np.pi / NFP])
ax.set_ylim([0, 2 * np.pi])
if labels:
    ax.set_xticks([0, np.pi / (2 * NFP), np.pi / NFP, 3 * np.pi / (2 * NFP), 2 * np.pi / NFP])
    ax.set_xticklabels([r"$0$", r"$\pi/2N_{FP}$", r"$\pi/N_{FP}$", r"$3\pi/2N_{FP}$", r"$2\pi/N_{FP}$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_title(r"Quasi-Symmetric with $M=1$, $N=1$ (QH)")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
if save:
    plt.savefig("quasisymmetry_M1_N1.png")
    plt.savefig("quasisymmetry_M1_N1.eps")

# quasi-symmetry toroidal (QA)
helicity = (1, 0)
iota = 1 + np.sqrt(5)
eq = Equilibrium(iota=np.array([iota]), well_l=well_l, omni_lmn=omni_lmn)
grid = LinearGrid(theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(alpha,eta)"], grid=grid, helicity=helicity)
theta = (
    data["theta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
zeta = (
    data["zeta_B(alpha,eta)"]
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
B = data["|B|_omni"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
theta = np.tile(theta + 2 * np.pi, (1, 2))
zeta = np.hstack((zeta, zeta + 2 * np.pi))
B = np.tile(B, (1, 2))
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap=colormap)
arr = np.linspace(0, 2 * np.pi)
ax.plot(arr, iota * arr, color="k", linestyle="--", lw=2)
ax.set_xlim([0, 2 * np.pi])
ax.set_ylim([0, 2 * np.pi])
if labels:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_title(r"Quasi-Symmetric with $M=1$, $N=0$ (QA)")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
if save:
    plt.savefig("quasisymmetry_M1_N0.png")
    plt.savefig("quasisymmetry_M1_N0.eps")
"""
print("Done!")
