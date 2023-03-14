"""Script for creating plots in dudt2023."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 14
color0 = "#d7191c"  # red
color1 = "#fdae61"  # orange
color2 = "#abd9e9"  # cyan
color3 = "#2c7bb6"  # blue

QI_l = np.array([1, 1.3, 1.8, 2])
QI_mn = np.array([np.pi / 16, -np.pi / 10, np.pi / 8])

eq = Equilibrium()
z = np.linspace(0, np.pi / 2, num=QI_l.size)
grid = LinearGrid(theta=101, zeta=100, endpoint=True)
data = eq.compute(
    ["|B|_QI", "zeta_QI"], grid=grid, M_QI=1, N_QI=1, QI_l=QI_l, QI_mn=QI_mn
)

alph = grid.nodes[:, 1].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
zbar = data["zeta-bar_QI"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
zqi = data["zeta_QI"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
B = data["|B|_QI"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()

fig, (ax0, ax1, ax2) = plt.subplots(
    ncols=3, figsize=(18, 6), sharex=False, sharey=False
)
# plot a
ax0.plot(
    zbar[0, :],
    B[0, :],
    color=color3,
    linestyle="-",
    lw=6,
    label=r"spline interpolation",
)
ax0.plot(z, QI_l, color=color1, linestyle="", marker="o", ms=12, label=r"$x_l$")
ax0.set_xticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2])
ax0.set_xticklabels(
    [r"$-\pi/2$", r"$-\pi/3$", r"$-\pi/6$", r"$0$", r"$\pi/6$", r"$\pi/3$", r"$\pi/2$"]
)
ax0.legend(loc="upper center")
ax0.set_xlabel(r"$\bar{\zeta}$")
ax0.set_ylabel(r"$|\mathbf{B}|$")
# plot b
ax1.plot(zqi[0, :], B[0, :], color=color0, lw=6, label=r"$\alpha=0$")
ax1.plot(zqi[25, :], B[25, :], color=color1, lw=6, label=r"$\alpha=\pi/2$")
ax1.plot(zqi[50, :], B[50, :], color=color2, lw=6, label=r"$\alpha=\pi$")
ax1.plot(zqi[75, :], B[75, :], color=color3, lw=6, label=r"$\alpha=3\pi/2$")
ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax1.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax1.legend(loc="lower right")
ax1.set_xlabel(r"$\zeta_B$")
ax1.set_ylabel(r"$|\mathbf{B}|$")
# plot c
div = make_axes_locatable(ax2)
im2 = ax2.contour(zqi, alph, B, norm=Normalize(), levels=20, cmap="plasma")
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im2, cax=cax, ticks=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
cbar.update_ticks()
arr = np.array([1, 1])
ax2.plot(
    [0, 2 * np.pi], 0 * arr, color=color0, linestyle=":", lw=6, label=r"$\alpha=0$"
)
ax2.plot(
    [0, 2 * np.pi],
    np.pi / 2 * arr,
    color=color1,
    linestyle=":",
    lw=6,
    label=r"$\alpha=\pi/2$",
)
ax2.plot(
    [0, 2 * np.pi],
    np.pi * arr,
    color=color2,
    linestyle=":",
    lw=6,
    label=r"$\alpha=\pi$",
)
ax2.plot(
    [0, 2 * np.pi],
    3 * np.pi / 2 * arr,
    color=color3,
    linestyle=":",
    lw=6,
    label=r"$\alpha=3\pi/2$",
)
ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax2.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax2.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax2.set_xlabel(r"$\zeta_B$")
ax2.set_ylabel(r"$\alpha$")
fig.tight_layout()
plt.show()
plt.savefig("QI.png")
plt.savefig("QI.eps")

print("Done!")
