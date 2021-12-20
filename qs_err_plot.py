import numpy as np
import matplotlib.pyplot as plt

from desc.grid import LinearGrid
from desc.equilibrium import EquilibriaFamily
from desc.plotting import _format_ax


fam_4 = EquilibriaFamily.load("/home/ddudt/DESC/examples/DESC/SIMSOPT_QA.h5")
fam_3 = EquilibriaFamily.load("/home/ddudt/DESC/examples/DESC/SIMSOPT_QA_n3.h5")
fam_2 = EquilibriaFamily.load("/home/ddudt/DESC/examples/DESC/SIMSOPT_QA_n2.h5")
fam_1 = EquilibriaFamily.load("/home/ddudt/DESC/examples/DESC/SIMSOPT_QA_n1.h5")

eq_4 = fam_4[-1]
eq_3 = fam_3[-1]
eq_2 = fam_2[-1]
eq_1 = fam_1[-1]

rho = np.linspace(1, 0, num=10, endpoint=False)
fig, ax = _format_ax(None)

f_4 = np.array([])
f_3 = np.array([])
f_2 = np.array([])
f_1 = np.array([])
for i, r in enumerate(rho):
    grid = LinearGrid(M=2 * eq_4.M_grid + 1, N=2 * eq_4.N_grid + 1, NFP=eq_4.NFP, rho=r)
    data_4 = eq_4.compute("|B|_mn", grid)
    data_3 = eq_3.compute("|B|_mn", grid)
    data_2 = eq_2.compute("|B|_mn", grid)
    data_1 = eq_1.compute("|B|_mn", grid)
    modes = data_4["Boozer modes"]
    idx = np.where((modes[2, :] != 0))[0]
    f_4 = np.append(
        f_4,
        np.sqrt(np.sum(data_4["|B|_mn"][idx] ** 2)),
        # / np.sqrt(np.sum(data_4["|B|_mn"] ** 2))
        # np.mean(np.abs(data_4["f_C"]) * data_4["sqrt(g)"]) / np.mean(data_4["sqrt(g)"]),
    )
    f_3 = np.append(
        f_3,
        np.sqrt(np.sum(data_3["|B|_mn"][idx] ** 2)),
        # / np.sqrt(np.sum(data_3["|B|_mn"] ** 2))
        # np.mean(np.abs(data_3["f_C"]) * data_3["sqrt(g)"]) / np.mean(data_3["sqrt(g)"]),
    )
    f_2 = np.append(
        f_2,
        np.sqrt(np.sum(data_2["|B|_mn"][idx] ** 2)),
        # / np.sqrt(np.sum(data_2["|B|_mn"] ** 2))
        # np.mean(np.abs(data_2["f_C"]) * data_2["sqrt(g)"]) / np.mean(data_2["sqrt(g)"]),
    )
    f_1 = np.append(
        f_1,
        np.sqrt(np.sum(data_1["|B|_mn"][idx] ** 2)),
        # / np.sqrt(np.sum(data_1["|B|_mn"] ** 2))
        # np.mean(np.abs(data_1["f_C"]) * data_1["sqrt(g)"]) / np.mean(data_1["sqrt(g)"]),
    )

ax.semilogy(rho, f_4, "ko-", label=r"$M,N \leq 4$")
ax.semilogy(rho, f_3, "go-", label=r"$M,N \leq 3$")
ax.semilogy(rho, f_2, "bo-", label=r"$M,N \leq 2$")
ax.semilogy(rho, f_1, "ro-", label=r"$M,N \leq 1$")

ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$f_B$")
fig.legend(loc="center right")
fig.set_tight_layout(True)

plt.show()
plt.savefig("/home/ddudt/DESC/fB_errors.png")
