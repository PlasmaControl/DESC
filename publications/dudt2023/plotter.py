"""Script for creating plots in dudt2023."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from desc.basis import DoubleFourierSeries
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.transform import Transform


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 14
color0 = "#d7191c"  # red
color1 = "#fdae61"  # orange
color2 = "#abd9e9"  # cyan
color3 = "#2c7bb6"  # blue

labels = True
helicity = (0, 1)
iota = 0.25
omni_l = np.array([1, 1.4, 1.8, 2])
omni_mn = np.array([np.pi / 8, -np.pi / 8, np.pi / 4])

eq = Equilibrium(iota=np.array([iota]), omni_l=omni_l, omni_mn=omni_mn)
z = np.linspace(0, np.pi / 2, num=omni_l.size)
grid = LinearGrid(theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_omni", "|B|(alpha,eta)"], grid=grid, helicity=helicity)

eta = (
    ((data["zeta"] * data["NFP"] - np.pi) / 2)
    .reshape((grid.num_theta, grid.num_zeta), order="F")
    .squeeze()
)
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

fig, (ax0, ax1, ax2) = plt.subplots(
    ncols=3, figsize=(18, 6), sharex=False, sharey=False
)
# plot a
ax0.plot(
    eta[0, :],
    B[0, :],
    color=color3,
    linestyle="-",
    lw=6,
    label=r"spline interpolation",
)
ax0.plot(z, omni_l, color=color1, linestyle="", marker="o", ms=12, label=r"$x_l$")
ax0.set_xlim([-np.pi / 2, np.pi / 2])
ax0.set_xticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2])
ax0.set_xticklabels(
    [r"$-\pi/2$", r"$-\pi/3$", r"$-\pi/6$", r"$0$", r"$\pi/6$", r"$\pi/3$", r"$\pi/2$"]
)
ax0.set_xlabel(r"$\eta$")
ax0.set_ylabel(r"$|\mathbf{B}|$")
ax0.legend(loc="upper center")
# plot b
ax1.plot(zeta[0, :], B[0, :], color=color0, lw=6, label=r"$\alpha=0$")
ax1.plot(zeta[25, :], B[25, :], color=color1, lw=6, label=r"$\alpha=\pi/2$")
ax1.plot(zeta[50, :], B[50, :], color=color2, lw=6, label=r"$\alpha=\pi$")
ax1.plot(zeta[75, :], B[75, :], color=color3, lw=6, label=r"$\alpha=3\pi/2$")
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
im2 = ax2.contour(zeta3, theta3, B3, norm=Normalize(), levels=20, cmap="plasma")
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
plt.savefig("model.png")
plt.savefig("model.eps")

omni_mn = np.array([-np.pi / 6, -np.pi / 7, -np.pi / 8])

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
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap="plasma")
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
plt.savefig("general.png")
# plt.savefig("general.eps")

# omnigenity poloidal (QI)
helicity = (0, 1)
iota = 2 / (1 + np.sqrt(5))
eq = Equilibrium(iota=np.array([iota]), omni_l=omni_l, omni_mn=omni_mn)
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
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap="plasma")
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
plt.savefig("omnigenity_M0_N1.png")
# plt.savefig("omnigenity_M0_N1.eps")

# omnigenity helical
helicity = (1, 1)
iota = -2 / (1 + np.sqrt(5))
eq = Equilibrium(iota=np.array([iota]), omni_l=omni_l, omni_mn=omni_mn)
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
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap="plasma")
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
plt.savefig("omnigenity_M1_N1.png")
# plt.savefig("omnigenity_M1_N1.eps")

# omnigenity toroidal
helicity = (1, 0)
iota = (1 + np.sqrt(5)) / 2
eq = Equilibrium(iota=np.array([iota]), omni_l=omni_l, omni_mn=omni_mn)
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
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap="plasma")
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
plt.savefig("omnigenity_M1_N0.png")
# plt.savefig("omnigenity_M1_N0.eps")

omni_mn = np.array([0, omni_mn[1], 0])

# quasi-symmetry poloidal (QP)
helicity = (0, 1)
iota = 2 / (1 + np.sqrt(5))
eq = Equilibrium(iota=np.array([iota]), omni_l=omni_l, omni_mn=omni_mn)
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
im = ax.contour(zeta3, theta3, B3, norm=Normalize(), levels=20, cmap="plasma")
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
plt.savefig("quasisymmetry_M0_N1.png")
# plt.savefig("quasisymmetry_M0_N1.eps")

# quasi-symmetry helical (QH)
helicity = (1, 1)
iota = -2 / (1 + np.sqrt(5))
eq = Equilibrium(iota=np.array([iota]), omni_l=omni_l, omni_mn=omni_mn)
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
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap="plasma")
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
    ax.set_title(r"Quasi-Symmetric with $M=1$, $N=1$ (QH)")
else:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
plt.savefig("quasisymmetry_M1_N1.png")
# plt.savefig("quasisymmetry_M1_N1.eps")

# quasi-symmetry toroidal (QA)
helicity = (1, 0)
iota = (1 + np.sqrt(5)) / 2
eq = Equilibrium(iota=np.array([iota]), omni_l=omni_l, omni_mn=omni_mn)
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
im = ax.contour(zeta, theta, B, norm=Normalize(), levels=20, cmap="plasma")
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
plt.savefig("quasisymmetry_M1_N0.png")
# plt.savefig("quasisymmetry_M1_N0.eps")

print("Done!")
