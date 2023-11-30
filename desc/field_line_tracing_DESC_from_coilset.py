"""Field line trace from coilset."""
import os
import time

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as nnp

from desc.coils import CoilSet
from desc.field_line_tracing_DESC_with_current_potential_python_regcoil import (
    compare_surfs_DESC_field_line_trace,
)
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import SumMagneticField, _MagneticField, field_line_integrate


def field_trace_from_coilset(
    coils,
    eqname,
    ntransit=100,
    dirname=None,
    Rs=None,
    Zs=None,
    show_surface=True,
    xlim=[0.66, 0.74],
    ylim=[-0.04, 0.04],
    save_files=True,
    only_return_data=False,
    savename=None,
    external_TF=None,
):
    """Field line trace from coilset.

    Parameters
    ----------
    coils : CoilSet or str
        CoilSet object to field line trace with
        if str, is assumed to be a MAKEGRID-formatted coils file and will
        load the coils from that
    eqname : Equilibrium or str
        Equilibrium against whose surfaces to plot the field line tracing
    ntransit : int, optional
        number of toroidal transits to follow, by default 100
    dirname : str, optional
        directory name to save files to, by default None
    Rs : ndarray, optional
        starting seed R points at zeta = 0 for the tracing, by default None
    Zs : ndarray, optional
        starting seed R points at zeta = 0 for the tracing, by default None
    show_surface : bool, optional
        _description_, by default True
    xlim : tuple or list, optional
        x limits for the plot, by default [0.66, 0.74]
    ylim : list, optional
        y limits for the plot, by default [-0.04, 0.04]
    save_files : bool, optional
        whether to save files or not, by default True
    only_return_data : bool, optional
        whether to only return the field line tracing data
        and not attempt to plot or save anything, by default False
    savename : str, optional
        Name of .png to save figure to (inside of dirname directory)
        by default None
    external_TF : _MagneticField, optional
        external magnetic field to include to trace with, in addition to
        the coilset's magnetic field, by default None
    Returns
    -------
    field_R  : ndarray, size [ntransits, Rs.size]
        R locations each transit for each field line traced
    field_Z : ndarray, size [ntransits, Rs.size]
        Z locations each transit for each field line traced

    """
    R0 = 703.5 / 1000
    r = 36.5 / 1000

    if isinstance(coils, str):
        dirname = dirname if dirname else f"{coils.strip('.txt')}"
        coils = CoilSet.from_makegrid_coilfile(coils)

    else:
        assert isinstance(coils, _MagneticField)
        dirname = dirname if dirname else eqname
    if hasattr(coils[0], "X"):
        N_grid = coils[0].X.size * 2 + 5
    else:
        N_grid = coils[0].N * 2 + 5
    if external_TF:
        coils = SumMagneticField(coils, external_TF)
    else:
        coils = coils

    t0 = time.time()
    phis = np.arange(0, ntransit * 2 * np.pi, 2 * np.pi)
    npts = 15
    rrr = (
        np.linspace(R0 - 0.7 * r, R0 + 0.7 * r, npts) if Rs is None else Rs
    )  # initial R positions of field-lines to trace
    zzz = np.zeros_like(rrr) if Zs is None else Zs
    print("Beginning Field Line Integration")

    grid = LinearGrid(N=N_grid, NFP=1, endpoint=True)

    # integrate field lines
    field_R, field_Z = field_line_integrate(rrr, zzz, phis, coils, grid=grid)

    t_elapse = time.time() - t0

    field_R_full = field_R.copy()
    field_Z_full = field_Z.copy()

    print(
        f"Field line tracing done, took {t_elapse}"
        f" seconds which is {t_elapse/60} mins or  {t_elapse/3600} hours"
    )

    if only_return_data:
        return field_R, field_Z

    R_list = []
    # save data

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

    for i in range(rrr.size):
        field_R = nnp.genfromtxt(f"{dirname}/trace_{ntransit}_transits_R_{i}.txt")
        field_Z = nnp.genfromtxt(f"{dirname}/trace_{ntransit}_transits_Z_{i}.txt")
        R_list.append(field_R[0])

        if nnp.max(nnp.abs(field_R)) < 1e3:
            plt.scatter(field_R, field_Z, s=1)
        else:
            plt.scatter(field_R, field_Z, s=1)

    plt.ylabel("Z")
    plt.xlabel("R")
    if show_surface:
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
    plt.xlim(xlim)
    plt.ylim(ylim)

    ax = compare_surfs_DESC_field_line_trace(load(eqname), ax, R_list)
    if not savename:
        plt.savefig(f"{dirname}/trace{ntransit}_transits_{dirname}.png")
    else:
        plt.savefig(f"{dirname}/{savename}.png")

    return field_R_full, field_Z_full
