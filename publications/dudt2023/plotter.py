"""Script for creating plots in dudt2023."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import desc.io
from desc.basis import DoubleFourierSeries
from desc.compute.utils import compress
from desc.grid import LinearGrid
from desc.transform import Transform


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 14
green = "#1b9e77"
orange = "#d95f02"
purple = "#7570b3"
colormap = "plasma"

save = False

eq_pol = desc.io.load("publications/dudt2023/poloidal.h5")
eq_tor = desc.io.load("publications/dudt2023/toroidal.h5")
eq_hel = desc.io.load("publications/dudt2023/helical.h5")
eq_pol_qs = desc.io.load("publications/dudt2023/poloidal_qs.h5")
eq_tor_qs = desc.io.load("publications/dudt2023/toroidal_qs.h5")
eq_hel_qs = desc.io.load("publications/dudt2023/helical_qs.h5")

# boundaries ==========================================================================

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

# Boozer surfaces =====================================================================

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

# effective ripple ====================================================================

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

print("Done!")
