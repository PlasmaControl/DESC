# git checkout dp/coils for this script to work

from desc import set_device

set_device("gpu")
import os

# %matplotlib inline
import sys
import time

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as nnp
from jax import grad, jit, vmap
from scipy.interpolate import interp1d

from desc.coils import Coil, CoilSet, FourierPlanarCoil, XYZCoil
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import (
    SumMagneticField,
    ToroidalMagneticField,
    field_line_integrate,
)
from desc.plotting import plot_1d, plot_2d, plot_comparison, plot_surfaces
from desc.field_line_tracing_DESC_with_current_potential_python_regcoil import (
    compare_surfs_DESC_field_line_trace,
)

################## Inputs ################
# coil .txt file in MAKEGRID format
# vacuum equilibrium to compare against
################# outputs #################
# poincare plot at phi=0 of the surfaces
# plot of Bnormal on the LCFS from the coils


# Field line tracing


# first arg : coil file
def field_trace_from_coilset(coils, eqname, ntransit=100, dirname=None):
    R0 = 703.5 / 1000
    Z0 = 0
    r = 36.5 / 1000

    if isinstance(coils, str):
        coils = CoilSet.from_makegrid_coilfile(coils)
        dirname = f"{coils.strip('.txt')}"
    else:
        assert isinstance(coils, Coil)
        dirname = dirname if dirname else eqname

    t0 = time.time()
    phis = np.arange(0, ntransit * 2 * np.pi, 2 * np.pi)
    npts = 15
    rrr = np.linspace(
        R0 - 0.7 * r, R0 + 0.7 * r, npts
    )  # initial R positions of field-lines to trace
    # set initial Z positions to zero
    print("Beginning Field Line Integration")
    # integrate field lines
    field_R, field_Z = field_line_integrate(rrr, np.zeros_like(rrr), phis, coils)

    t_elapse = time.time() - t0

    print(
        f"{dirname} field line tracing done, took {t_elapse} seconds which is {t_elapse/60} mins or  {t_elapse/3600} hours"
    )

    R_list = []
    # save data

    import os

    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    for i in range(field_R.shape[1]):
        nnp.savetxt(
            f"{dirname}/trace_{ntransit}_transits_R_{i}.txt", nnp.asarray(field_R[:, i])
        )
        nnp.savetxt(
            f"{dirname}/trace_{ntransit}_transits_Z_{i}.txt", nnp.asarray(field_Z[:, i])
        )

    # read data again and plot

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(npts):
        field_R = nnp.genfromtxt(f"{dirname}/trace_{ntransit}_transits_R_{i}.txt")
        field_Z = nnp.genfromtxt(f"{dirname}/trace_{ntransit}_transits_Z_{i}.txt")
        R_list.append(field_R[0])

        if nnp.max(nnp.abs(field_R)) < 4:
            plt.scatter(field_R, field_Z, s=1)
        else:
            plt.scatter(field_R, field_Z, s=1)

    plt.ylabel("Z")
    plt.xlabel("R")
    R0_ves = 0.7035
    a_ves = 0.0365
    theta = np.linspace(0, 2 * np.pi, 100)
    R_ves = R0_ves + a_ves * np.cos(theta)
    Z_ves = a_ves * np.sin(theta)
    plt.plot(R_ves, Z_ves, "k", label="Vacuum vessel")
    a_lim = 0.0365 - 8 / 1000
    theta = np.linspace(0, 2 * np.pi, 100)
    R_lim = R0_ves + a_lim * np.cos(theta)
    Z_lim = a_lim * np.sin(theta)
    plt.plot(R_lim, Z_lim, "r--", label="8mm from vessel")
    plt.legend()
    plt.ylabel("Z")
    plt.xlabel("R")
    plt.xlim([0.66, 0.74])
    plt.ylim([-0.04, 0.04])

    ax = compare_surfs_DESC_field_line_trace(eqname, ax, R_list)

    plt.savefig(f"{dirname}/trace{ntransit}_transits.png")
