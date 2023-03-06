"""Script for creating plots in dudt2023."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

import desc.io
from desc.basis import DoubleFourierSeries
from desc.compute.utils import compress
from desc.grid import LinearGrid
from desc.transform import Transform


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 14
color0 = "#d7191c"  # red
color1 = "#fdae61"  # orange
color2 = "#abd9e9"  # cyan
color3 = "#2c7bb6"  # blue

eq_Good = desc.io.load("Goodman_QI_nfp1_vacuum_M7_N7_output.h5")[-1]
eq_init = desc.io.load("Goodman_QI_nfp1_vacuum_M2_N2_output.h5")[-1]
eq_DESC = desc.io.load("Goodman_QI_nfp1_vacuum_DESC.h5")

with open("QI_params.pkl", "rb") as handle:
    QI_params = pickle.load(handle)

# boundaries
colors = [color0, color1, color3]
styles = ["-", "-", "-"]
labels = ["Initial", "Goodman et al.", "DESC"]
fig, ax = plt.subplots(figsize=(7, 7), sharex=True, sharey=True)
grid = LinearGrid(theta=100, zeta=6, NFP=eq_init.NFP, endpoint=True)
for i, eq in enumerate((eq_init, eq_Good, eq_DESC)):
    coords = eq.compute(["R", "Z"], grid=grid)
    R = coords["R"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    Z = coords["Z"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    for j in range(grid.num_zeta - 1):
        (line,) = ax.plot(
            R[:, -1, j],
            Z[:, -1, j],
            color=colors[i],
            linestyle=styles[i],
            lw=4,
        )
        if j == 0:
            line.set_label(labels[i])
ax.legend(loc="upper left")
ax.set_xlabel(r"$R$ (m)")
ax.set_ylabel(r"$Z$ (m)")
fig.tight_layout()
plt.show()
plt.savefig("Goodman_boundaries.png")
plt.savefig("Goodman_boundaries.eps")

grid0_compute = LinearGrid(
    M=6 * eq_DESC.M + 1, N=6 * eq_DESC.N + 1, NFP=eq_DESC.NFP, sym=False, rho=1.0
)
grid1_compute = LinearGrid(
    M=6 * eq_DESC.M + 1, N=6 * eq_DESC.N + 1, NFP=eq_DESC.NFP, sym=False, rho=0.2
)

# Boozer surfaces
fig, (ax0, ax1, ax2) = plt.subplots(
    nrows=1, ncols=3, figsize=(25, 10), sharex=True, sharey=True
)
grid_plot = LinearGrid(
    theta=101, zeta=101, NFP=eq_DESC.NFP, sym=False, endpoint=True, rho=1.0
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
basis = DoubleFourierSeries(
    M=2 * eq_DESC.M, N=2 * eq_DESC.N, sym=eq_DESC.R_basis.sym, NFP=eq_DESC.NFP
)
B_transform = Transform(grid_plot, basis)
# initial
data0_init = eq_init.compute("|B|_mn", grid=grid0_compute)
data1_init = eq_init.compute("|B|_mn", grid=grid1_compute)
iota0_init = compress(grid0_compute, data0_init["iota"])
iota1_init = compress(grid1_compute, data1_init["iota"])
BB_init = B_transform.transform(data0_init["|B|_mn"])
BB_init = BB_init.reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
# DESC
data0_DESC = eq_DESC.compute("|B|_mn", grid=grid0_compute)
data1_DESC = eq_DESC.compute("|B|_mn", grid=grid1_compute)
iota0_DESC = compress(grid0_compute, data0_DESC["iota"])
iota1_DESC = compress(grid1_compute, data1_DESC["iota"])
BB_DESC = B_transform.transform(data0_DESC["|B|_mn"])
BB_DESC = BB_DESC.reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
# Goodman
data0_Good = eq_Good.compute("|B|_mn", grid=grid0_compute)
data1_Good = eq_Good.compute("|B|_mn", grid=grid1_compute)
iota0_Good = compress(grid0_compute, data0_Good["iota"])
iota1_Good = compress(grid1_compute, data1_Good["iota"])
BB_Good = B_transform.transform(data0_Good["|B|_mn"])
BB_Good = BB_Good.reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
# contours
amin = min(np.nanmin(BB_init), np.nanmin(BB_DESC), np.nanmin(BB_Good))
amax = max(np.nanmax(BB_init), np.nanmax(BB_DESC), np.nanmax(BB_Good))
contour_kwargs = {}
contour_kwargs["norm"] = Normalize()
contour_kwargs["levels"] = np.linspace(amin, amax, 30)
contour_kwargs["cmap"] = "plasma"
contour_kwargs["extend"] = "both"
cax_kwargs = {"size": "5%", "pad": 0.05}
div = make_axes_locatable(ax2)
im_init = ax0.contour(zz, tt, BB_init, **contour_kwargs)
im_DESC = ax1.contour(zz, tt, BB_DESC, **contour_kwargs)
im_Good = ax2.contour(zz, tt, BB_Good, **contour_kwargs)
cax = div.append_axes("right", **cax_kwargs)
cbar = fig.colorbar(im_Good, cax=cax)
cbar.update_ticks()
ax0.set_ylabel(r"$\theta_{Boozer}$")
# fieldlines
theta0 = np.atleast_2d(np.array([np.pi, 2 * np.pi]))
zeta = np.atleast_2d(np.linspace(0, 2 * np.pi / grid_plot.NFP, 100)).T
alpha_init = theta0 + iota0_init * zeta
alpha_DESC = theta0 + iota0_DESC * zeta
alpha_Good = theta0 + iota0_Good * zeta
ax0.plot(zeta, alpha_init, color="k", ls="-", lw=2)
ax1.plot(zeta, alpha_DESC, color="k", ls="-", lw=2)
ax2.plot(zeta, alpha_Good, color="k", ls="-", lw=2)
ax0.set_xlim([0, 2 * np.pi])
ax0.set_ylim([0, 2 * np.pi])
ax1.set_xlim([0, 2 * np.pi])
ax1.set_ylim([0, 2 * np.pi])
ax2.set_xlim([0, 2 * np.pi])
ax2.set_ylim([0, 2 * np.pi])
props = dict(boxstyle="round", facecolor="w", alpha=0.7)
ax0.text(
    0.78,
    0.95,
    r"$\iota = {:2.3f}$".format(iota0_init[0]),
    transform=ax0.transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=props,
)
ax1.text(
    0.78,
    0.95,
    r"$\iota = {:2.3f}$".format(iota0_DESC[0]),
    transform=ax1.transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=props,
)
ax2.text(
    0.78,
    0.95,
    r"$\iota = {:2.3f}$".format(iota0_Good[0]),
    transform=ax2.transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=props,
)
ax0.set_xlabel(r"$\zeta_{Boozer}$")
ax1.set_xlabel(r"$\zeta_{Boozer}$")
ax2.set_xlabel(r"$\zeta_{Boozer}$")
ax0.set_title(r"Initial $|\mathbf{B}|~(T)$")
ax1.set_title(r"DESC $|\mathbf{B}|~(T)$")
ax2.set_title(r"Goodman et al. $|\mathbf{B}|~(T)$")
fig.tight_layout()
plt.show()
plt.savefig("Goodman_Boozer.png")
plt.savefig("Goodman_Boozer.eps")

# magnetic wells
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
grid0 = LinearGrid(
    theta=np.array([0, np.pi]),
    zeta=101,
    NFP=eq_DESC.NFP,
    sym=False,
    endpoint=True,
    rho=1.0,
)
grid1 = LinearGrid(
    theta=np.array([0, np.pi]),
    zeta=101,
    NFP=eq_DESC.NFP,
    sym=False,
    endpoint=True,
    rho=0.6,
)
data0 = eq_Good.compute(
    ["|B|_QI", "zeta_QI"],
    grid=grid0,
    M_QI=6,
    N_QI=6,
    QI_l=QI_params["1.000"]["QI_l"],
    QI_mn=QI_params["1.000"]["QI_mn"],
)
data1 = eq_Good.compute(
    ["|B|_QI", "zeta_QI"],
    grid=grid1,
    M_QI=6,
    N_QI=6,
    QI_l=QI_params["0.600"]["QI_l"],
    QI_mn=QI_params["0.600"]["QI_mn"],
)
zeta0 = data0["zeta_QI"].reshape((2, -1), order="F")
zeta1 = data1["zeta_QI"].reshape((2, -1), order="F")
# initial
nodes0_init = np.array(
    [data0["rho"], data0["theta"] + iota0_init * data0["zeta_QI"], data0["zeta_QI"]]
).T
nodes1_init = np.array(
    [data1["rho"], data1["theta"] + iota1_init * data1["zeta_QI"], data1["zeta_QI"]]
).T
B0_init = np.matmul(basis.evaluate(nodes0_init), data0_init["|B|_mn"]).reshape(
    (2, -1), order="F"
)
B1_init = np.matmul(basis.evaluate(nodes1_init), data1_init["|B|_mn"]).reshape(
    (2, -1), order="F"
)
# DESC
nodes0_DESC = np.array(
    [data0["rho"], data0["theta"] + iota0_DESC * data0["zeta_QI"], data0["zeta_QI"]]
).T
nodes1_DESC = np.array(
    [data1["rho"], data1["theta"] + iota1_DESC * data1["zeta_QI"], data1["zeta_QI"]]
).T
B0_DESC = np.matmul(basis.evaluate(nodes0_DESC), data0_DESC["|B|_mn"]).reshape(
    (2, -1), order="F"
)
B1_DESC = np.matmul(basis.evaluate(nodes1_DESC), data1_DESC["|B|_mn"]).reshape(
    (2, -1), order="F"
)
# Goodman
nodes0_Good = np.array(
    [data0["rho"], data0["theta"] + iota0_Good * data0["zeta_QI"], data0["zeta_QI"]]
).T
nodes1_Good = np.array(
    [data1["rho"], data1["theta"] + iota1_Good * data1["zeta_QI"], data1["zeta_QI"]]
).T
B0_Good = np.matmul(basis.evaluate(nodes0_Good), data0_Good["|B|_mn"]).reshape(
    (2, -1), order="F"
)
B1_Good = np.matmul(basis.evaluate(nodes1_Good), data1_Good["|B|_mn"]).reshape(
    (2, -1), order="F"
)
# QI
B0_QI = data0["|B|_QI"].reshape((2, -1), order="F")
B1_QI = data1["|B|_QI"].reshape((2, -1), order="F")
# rho = 1, alpha = 0
ax[0, 0].set_title(r"$\rho = 1.0,~\alpha = 0$")
ax[0, 0].plot(
    zeta0[0, :], B0_init[0, :], color=color3, linestyle="-", lw=8, label="Initial"
)
ax[0, 0].plot(
    zeta0[0, :],
    B0_Good[0, :],
    color=color2,
    linestyle="-",
    lw=8,
    label="Goodman et al.",
)
ax[0, 0].plot(
    zeta0[0, :], B0_QI[0, :], color=color1, linestyle="--", lw=8, label="QI Target"
)
ax[0, 0].plot(
    zeta0[0, :], B0_DESC[0, :], color=color0, linestyle=":", lw=8, label="DESC"
)
ax[0, 0].legend(loc="upper center")
ax[0, 0].set_xlim([0, 2 * np.pi])
ax[0, 0].set_xlim([0, 2 * np.pi])
ax[0, 0].set_ylabel(r"$|\mathbf{B}|~(T)$")
# rho = 1, alpha = pi
ax[0, 1].set_title(r"$\rho = 1.0,~\alpha = \pi$")
ax[0, 1].plot(
    zeta1[1, :], B0_init[1, :], color=color3, linestyle="-", lw=8, label="Initial"
)
ax[0, 1].plot(
    zeta1[1, :],
    B0_Good[1, :],
    color=color2,
    linestyle="-",
    lw=8,
    label="Goodman et al.",
)
ax[0, 1].plot(
    zeta1[1, :], B0_QI[1, :], color=color1, linestyle="--", lw=8, label="QI Target"
)
ax[0, 1].plot(
    zeta1[1, :], B0_DESC[1, :], color=color0, linestyle=":", lw=8, label="DESC"
)
ax[0, 1].set_xlim([0, 2 * np.pi])
ax[0, 1].set_xlim([0, 2 * np.pi])
# rho = 0.2, alpha = 0
ax[1, 0].set_title(r"$\rho = 0.2,~\alpha = 0$")
ax[1, 0].plot(
    zeta0[0, :], B1_init[0, :], color=color3, linestyle="-", lw=8, label="Initial"
)
ax[1, 0].plot(
    zeta0[0, :],
    B1_Good[0, :],
    color=color2,
    linestyle="-",
    lw=8,
    label="Goodman et al.",
)
ax[1, 0].plot(
    zeta0[0, :], B1_QI[0, :], color=color1, linestyle="--", lw=8, label="QI Target"
)
ax[1, 0].plot(
    zeta0[0, :], B1_DESC[0, :], color=color0, linestyle=":", lw=8, label="DESC"
)
ax[1, 0].set_xlim([0, 2 * np.pi])
ax[1, 0].set_xlim([0, 2 * np.pi])
ax[1, 0].set_xlabel(r"$\zeta_{Boozer}$")
ax[1, 0].set_ylabel(r"$|\mathbf{B}|~(T)$")
# rho = 0.2, alpha = pi
ax[1, 1].set_title(r"$\rho = 0.2,~\alpha = \pi$")
ax[1, 1].plot(
    zeta1[1, :], B1_init[1, :], color=color3, linestyle="-", lw=8, label="Initial"
)
ax[1, 1].plot(
    zeta1[1, :],
    B1_Good[1, :],
    color=color2,
    linestyle="-",
    lw=8,
    label="Goodman et al.",
)
ax[1, 1].plot(
    zeta1[1, :], B1_QI[1, :], color=color1, linestyle="--", lw=8, label="QI Target"
)
ax[1, 1].plot(
    zeta1[1, :], B1_DESC[1, :], color=color0, linestyle=":", lw=8, label="DESC"
)
ax[1, 1].set_xlim([0, 2 * np.pi])
ax[1, 1].set_xlim([0, 2 * np.pi])
ax[1, 1].set_xlabel(r"$\zeta_{Boozer}$")
fig.tight_layout()
plt.show()
plt.savefig("Goodman_wells.png")
plt.savefig("Goodman_wells.eps")

print("Done!")
