"""Script for creating plots in dudt2022."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid


plt.rcParams["font.size"] = 14
dim = 11

eq = Equilibrium()
eq.QI_l = np.array([1, 1.3, 1.8, 2])
eq.QI_mn = np.array([np.pi / 16, -np.pi / 10, np.pi / 8])
z = np.linspace(0, np.pi / 2, num=eq.QI_l.size)

grid = LinearGrid(theta=101, zeta=100, endpoint=True)
data = eq.compute(["|B|_QI", "zeta_QI"], grid=grid)

alph = grid.nodes[:, 1].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
zbar = data["zeta-bar_QI"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
zqi = data["zeta_QI"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()
B = data["|B|_QI"].reshape((grid.num_theta, grid.num_zeta), order="F").squeeze()

fig, (ax0, ax1, ax2) = plt.subplots(
    ncols=3, figsize=(18, 6), sharex=False, sharey=False
)
# plot a
ax0.plot(zbar[0, :], B[0, :], "b-", lw=6, label=r"spline interpolation")
ax0.plot(z, eq.QI_l, "ro", ms=12, label=r"$x_l$")
ax0.legend(loc="upper center")
ax0.set_xlabel(r"$\bar{\zeta}$")
ax0.set_ylabel(r"$|\mathbf{B}|$")
# plot b
ax1.plot(zqi[0, :], B[0, :], "k-", lw=6, label=r"$\alpha=0$")
ax1.plot(zqi[25, :], B[25, :], "g-", lw=6, label=r"$\alpha=\pi/2$")
ax1.plot(zqi[50, :], B[50, :], "b-", lw=6, label=r"$\alpha=\pi$")
ax1.plot(zqi[75, :], B[75, :], "r-", lw=6, label=r"$\alpha=3\pi/2$")
ax1.legend(loc="lower right")
ax1.set_xlabel(r"$\zeta_{Boozer}$")
ax1.set_ylabel(r"$|\mathbf{B}|$")
# plot c
div = make_axes_locatable(ax2)
im2 = ax2.contour(zqi, alph, B, norm=Normalize(), levels=20, cmap="jet")
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im2, cax=cax)
cbar.update_ticks()
arr = np.array([1, 1])
ax2.plot([0, 2 * np.pi], 0 * arr, "k:", lw=6, label=r"$\alpha=0$")
ax2.plot([0, 2 * np.pi], np.pi / 2 * arr, "g:", lw=6, label=r"$\alpha=\pi/2$")
ax2.plot([0, 2 * np.pi], np.pi * arr, "b:", lw=6, label=r"$\alpha=\pi$")
ax2.plot([0, 2 * np.pi], 3 * np.pi / 2 * arr, "r:", lw=6, label=r"$\alpha=3\pi/2$")
ax2.set_xlabel(r"$\zeta_{Boozer}$")
ax2.set_ylabel(r"$\alpha$")
fig.tight_layout()
