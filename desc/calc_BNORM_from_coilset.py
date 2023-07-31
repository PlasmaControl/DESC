import os

# %matplotlib inline
import sys

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as nnp
from jax import grad, jit, vmap
from scipy.interpolate import interp1d

from desc import set_device
from desc.coils import Coil, CoilSet, FourierPlanarCoil, XYZCoil

# set_device("gpu")
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import (
    MagneticField,
    SumMagneticField,
    ToroidalMagneticField,
    field_line_integrate,
)
from desc.plotting import plot_1d, plot_2d, plot_comparison, plot_surfaces


def calc_BNORM_from_coilset(coils, eqname, alpha, step, B0=None):
    eq = load(eqname)
    if B0:
        R0_ves = 0.7035
        if not isinstance(B0, MagneticField):
            assert float(B0) == B0, "B0 must be a float or a MagneticField!"
            TF_field = ToroidalMagneticField(R0=R0_ves, B0=B0)

        coils = SumMagneticField(coils, TF_field)

    dirname = f"{eqname.split('/')[-1].strip('.h5')}"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    grid = LinearGrid(rho=1, M=40, N=40, axis=False, NFP=eq.NFP, sym=False)
    grid_ax = LinearGrid(
        rho=np.array(1e-6), theta=np.array(0.0), N=10, axis=False, NFP=eq.NFP, sym=False
    )

    data_desc = eq.compute(
        ["n_rho", "|B|", "B_R", "B_phi", "B_Z", "R", "phi", "Z"], grid=grid
    )
    data_ax = eq.compute(["|B|", "B_R", "B_phi", "B_Z", "R", "phi", "Z"], grid=grid_ax)

    cords = np.vstack((data_desc["R"], data_desc["phi"], data_desc["Z"])).T
    cords_ax = np.vstack((data_ax["R"], data_ax["phi"], data_ax["Z"])).T

    B = coils.compute_magnetic_field(cords)
    B_ax = coils.compute_magnetic_field(cords_ax)

    Bnorm = (
        B[:, 0] * data_desc["n_rho"][:, 0]
        + B[:, 1] * data_desc["n_rho"][:, 1]
        + B[:, 2] * data_desc["n_rho"][:, 2]
    )

    print(f"Maximum |Bnormal| on surface: {np.max(np.abs(Bnorm))}")
    print(f"Minimum |Bnormal| on surface: {np.min(np.abs(Bnorm))}")
    print(f"average |Bnormal| on surface: {np.mean(np.abs(Bnorm))}")
    print(f"average |B| on surface: {np.mean(np.abs(np.linalg.norm(B,axis=-1)))}\n")
    print(f"average |B| on axis: {np.mean(np.abs(np.linalg.norm(B_ax,axis=-1)))}\n")
    print(f"eq average |B| on surface: {np.mean(np.abs(data_desc['|B|']))}\n")
    print(f"eq average |B| on axis: {np.mean(np.abs(data_ax['|B|']))}\n")
    print(
        f"|B| on axis eq / |B| on axis coil : {np.mean(np.abs(data_ax['|B|'])) / np.mean(np.abs(np.linalg.norm(B_ax,axis=-1)))}\n"
    )

    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(8, 8))
    plt.contourf(
        grid.nodes[grid.unique_theta_idx, 1],
        grid.nodes[grid.unique_zeta_idx, 2],
        Bnorm.reshape(grid.num_theta, grid.num_zeta),
    )
    plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\zeta$")

    plt.title(f"Bnormal with N={len(coils)} coils")

    plt.savefig(
        f"{dirname}/Bnormal_ncoils_{len(coils)}_alpha_{alpha:1.4e}_step_{step}_{dirname}.png"
    )

    with open(
        f"{dirname}/Bnormal_info_ncoils_{len(coils)}_alpha_{alpha:1.4e}_step_{step}_{dirname}.txt",
        "w+",
    ) as f:
        f.write(f"Maximum |Bnormal| on surface: {np.max(np.abs(Bnorm))}\n")
        f.write(f"Minimum |Bnormal| on surface: {np.min(np.abs(Bnorm))}\n")
        f.write(f"average |Bnormal| on surface: {np.mean(np.abs(Bnorm))}\n")
        f.write(
            f"average |B| on surface: {np.mean(np.abs(np.linalg.norm(B,axis=-1)))}\n"
        )
        f.write(
            f"average |B| on axis: {np.mean(np.abs(np.linalg.norm(B_ax,axis=-1)))}\n"
        )
        f.write(f"eq average |B| on surface: {np.mean(np.abs(data_desc['|B|']))}\n")
        f.write(f"eq average |B| on axis: {np.mean(np.abs(data_ax['|B|']))}\n")
        f.write(
            f"|B| on axis eq / |B| on axis coil : {np.mean(np.abs(data_ax['|B|'])) / np.mean(np.abs(np.linalg.norm(B_ax,axis=-1)))}\n"
        )
    return None
