"""Find BNORMAL on surfac given a coilset and plot it."""
import os

import jax.numpy as np
import matplotlib.pyplot as plt

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import MagneticField, SumMagneticField, ToroidalMagneticField


def calc_BNORM_from_coilset(coils, eqname, alpha, step, B0=None):
    """Find BNORMAL on surfac given a coilset and plot it, and save to a txt file.

    Parameters
    ----------
    coils : MagneticField
        A DESC MagneticField object (which can be a CoilSet) to find the Bnormal from
    eqname : str or Equilibrium
        The DESC equilibrum the surface current potential was found for
        If str, assumes it is the name of the equilibrium .h5 output and will
        load it
    alpha : float
        regularization parameter used in run_regcoil
        #TODO: can remove this and replace with something like
        basename to be used for every saved figure
    step : int, optional
        Amount of points to step when saving the coil geometry
        by default 2, meaning that every other point will be saved
        if higher, less points will be saved
        #TODO: can remove this and replace with something like
        basename to be used for every saved figure
    B0 : MagneticField, optional
        Magnetic field external to the coils given, to
        include when field line tracing.
        for example, a simple TF field
        should be same field as that used when calling
        run_regcoil to find the surface current that was
        discretized into the coilset

    """
    if isinstance(eqname, str):
        eq = load(eqname)
    elif isinstance(eqname, EquilibriaFamily) or isinstance(eqname, Equilibrium):
        eq = eqname
    if hasattr(eq, "__len__"):
        eq = eq[-1]
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

    data_desc = eq.compute(["|B|", "R", "phi", "Z"], grid=grid)
    data_ax = eq.compute(["|B|", "R", "phi", "Z"], grid=grid_ax)

    cords = np.vstack((data_desc["R"], data_desc["phi"], data_desc["Z"])).T
    cords_ax = np.vstack((data_ax["R"], data_ax["phi"], data_ax["Z"])).T

    B = coils.compute_magnetic_field(cords, basis="rpz")
    B_ax = coils.compute_magnetic_field(cords_ax, basis="rpz")

    Bnorm, _ = coils.compute_Bnormal(eq.surface, grid)

    print(f"Maximum |Bnormal| on surface: {np.max(np.abs(Bnorm))}")
    print(f"Minimum |Bnormal| on surface: {np.min(np.abs(Bnorm))}")
    print(f"average |Bnormal| on surface: {np.mean(np.abs(Bnorm))}")
    print(f"average |B| on surface: {np.mean(np.abs(np.linalg.norm(B,axis=-1)))}\n")
    print(f"average |B| on axis: {np.mean(np.abs(np.linalg.norm(B_ax,axis=-1)))}\n")
    print(f"eq average |B| on surface: {np.mean(np.abs(data_desc['|B|']))}\n")
    print(f"eq average |B| on axis: {np.mean(np.abs(data_ax['|B|']))}\n")
    print(
        "|B| on axis eq / |B| on axis coil :"
        f"{np.mean(np.abs(data_ax['|B|'])) / np.mean(np.linalg.norm(B_ax,axis=-1))}\n"
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
        f"{dirname}/Bnormal_ncoils_{len(coils)}"
        f"_alpha_{alpha:1.4e}_step_{step}_{dirname}.png"
    )

    with open(
        f"{dirname}/Bnormal_info_ncoils_{len(coils)}"
        f"_alpha_{alpha:1.4e}_step_{step}_{dirname}.txt",
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

    return None
