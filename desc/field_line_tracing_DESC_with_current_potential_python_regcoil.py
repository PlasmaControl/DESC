"""Field line tracing from a current potential."""
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import desc.examples
import desc.io
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid
from desc.magnetic_fields import (
    CurrentPotentialField,
    FourierCurrentPotentialField,
    SumMagneticField,
    field_line_integrate,
)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


##################################


# TODO: add option to first make a spline field then use that for integration
def trace_from_curr_pot(  # noqa: C901 - FIXME: simplify this
    current_potential_field,
    eqname,
    M=30,
    N=30,
    alpha=0,
    ntransit=100,
    external_TF=None,
    savename=None,
    Rs=None,
    phi0=0,
    xlim=[0.66, 0.74],
    ylim=[-0.04, 0.04],
    **kwargs,
):
    """Field line trace from current potential.

    Parameters
    ----------
    current_potential_field : CurrentPotentialField or FourierCurrentPotentialField
        CurrentPotentialField or FourierCurrentPotentialField object from which to
        calculate the magnetic field to field line trace with.
    eqname : str or Equilibrium
        The DESC equilibrum the surface current potential was found for
        If str, assumes it is the name of the equilibrium .h5 output and will
        load it
    M : int, optional
        Poloidal resolution of source grid, by default 30
    N : int, optional
        Toroidal resolution of source grid, by default 30
    alpha : float
        regularization parameter used in run_regcoil
        #TODO: can remove this and replace with something like
        basename to be used for every saved figure
    ntransit : int, optional
        number of toroidal transits to trace, by default 100
    external_TF : MagneticField, optional
        external magnetic field to include to trace with, in addition to
        the current potential's magnetic field, by default None
    savename : str, optional
        Name of .png to save figure to (inside of dirname directory)
        by default None
    Rs : ndarray, optional
        starting seed R points at zeta = 0 for the tracing, by default None
    phi0 : int, optional
        phi plane to create poincare plot at, by default 0
    xlim : tuple or list, optional
        x limits for the plot, by default [0.66, 0.74]
    ylim : list, optional
        y limits for the plot, by default [-0.04, 0.04]

    Returns
    -------
    field_R  : ndarray, size [ntransits, Rs.size]
        R locations each transit for each field line traced
    field_Z : ndarray, size [ntransits, Rs.size]
        Z locations each transit for each field line traced

    """
    if kwargs.get("use_agg_backend", False):
        # this may be needed if matplotlib complains
        # about not having a gui backend available
        matplotlib.use("agg")
    if isinstance(eqname, str):
        eq = desc.io.load(eqname)
    elif isinstance(eqname, EquilibriaFamily) or isinstance(eqname, Equilibrium):
        eq = eqname
    elif eqname is None:
        eq = None
    if hasattr(eq, "__len__"):
        eq = eq[-1]
    assert isinstance(
        current_potential_field, FourierCurrentPotentialField
    ) or isinstance(current_potential_field, CurrentPotentialField), (
        "current_potential_field must be one of CurrentPotentialField "
        "or FourierCurrentPotentialField"
    )

    current_potential_field.surface_grid = LinearGrid(
        M=M, N=N, rho=np.array(1.0), NFP=current_potential_field.NFP
    )

    # Field line tracing

    R0 = 703.5 / 1000
    r = 36.5 / 1000

    Bfield_currpot = current_potential_field

    if external_TF:
        Bfield = SumMagneticField(Bfield_currpot, external_TF)
    else:
        Bfield = Bfield_currpot

    t0 = time.time()
    phis = np.arange(0, ntransit * 2 * np.pi, 2 * np.pi) + phi0
    n_R_points = 45
    rrr = (
        np.linspace(R0 - 0.9 * r, R0 + 0.9 * r, n_R_points) if Rs is None else Rs
    )  # initial R positions of field-lines to trace
    # set initial Z positions are zero
    n_R_points = rrr.size

    # integrate field lines
    field_R, field_Z = field_line_integrate(
        rrr, np.zeros_like(rrr), phis, Bfield, grid=current_potential_field.surface_grid
    )

    t_elapse = time.time() - t0
    print(
        f" field line tracing done, took {t_elapse} seconds which"
        f" is {t_elapse/60} mins or  {t_elapse/3600} hours"
    )

    R_list = []
    # save data
    if isinstance(eqname, str):
        dirname = f"trace_M_{M}_N_{N}_alpha_{alpha}_{str(os.path.basename(eqname)).strip('.h5')}"  # noqa
    else:
        dirname = "field_trace_from_potential"
    print(f"Saving to {dirname}")

    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    for i in range(field_R.shape[1]):
        np.savetxt(
            f"{dirname}/trace_{ntransit}_transits_R_{i}.txt", np.asarray(field_R[:, i])
        )
        np.savetxt(
            f"{dirname}/trace_{ntransit}_transits_Z_{i}.txt", np.asarray(field_Z[:, i])
        )

    # read data again and plot

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(n_R_points):
        R = np.genfromtxt(f"{dirname}/trace_{ntransit}_transits_R_{i}.txt")
        Z = np.genfromtxt(f"{dirname}/trace_{ntransit}_transits_Z_{i}.txt")
        R_list.append(R[0])
        if (
            np.max(np.abs(R)) < np.max(Rs) * 1.5
            and np.min(np.abs(R)) > np.min(Rs) * 0.5
        ):
            plt.scatter(R, Z, s=1)
        else:
            plt.scatter(R[0], Z[0], marker="x", s=5)

    plt.ylabel("Z")
    plt.xlabel("R")

    data = current_potential_field.compute(
        "x", basis="rpz", grid=LinearGrid(rho=1, M=20)
    )["x"]

    R_ves = data[:, 0]
    Z_ves = data[:, 2]
    plt.plot(R_ves, Z_ves, "k", label="Vacuum vessel")
    plt.legend()
    plt.ylabel("Z")
    plt.xlabel("R")

    plt.xlim(xlim)
    plt.ylim(ylim)

    ax = compare_surfs_DESC_field_line_trace(eq, ax, R_list)
    if not savename:
        plt.savefig(f"{dirname}/trace{ntransit}_transits_{dirname}.png")
    else:
        plt.savefig(f"{dirname}/{savename}.png")
    return field_R, field_Z


def compare_surfs_DESC_field_line_trace(
    eq_name, ax, R_list, savename_poin=None, phi0=0
):
    """Compare surfaces of an equilibrium to traced flux surfaces."""
    # given initial ax object which already has stuff plotted,
    # and a list of R values of different surfaces
    # plot the flux surfaces of a DESC equiilibrium at the rhos corr. to those surfaces
    # those are values should be at theta=zeta=0... which should corr.
    # to the outboard plane

    if eq_name is None:
        return  # dont do anything if no eq passed in

    if isinstance(eq_name, str):
        eq = desc.io.load(eq_name)
    elif isinstance(eq_name, EquilibriaFamily) or isinstance(eq_name, Equilibrium):
        eq = eq_name
    if hasattr(eq, "__len__"):
        eq = eq[-1]

    Rgrid = LinearGrid(theta=np.array([np.pi]), zeta=np.array([phi0]), L=100)
    Rcoords = eq.compute("R", grid=Rgrid)
    rho_of_R = interp1d(x=Rcoords["R"], y=Rgrid.nodes[:, 0], fill_value="extrapolate")

    rhos_to_plot = rho_of_R(np.asarray(R_list))
    rhos_to_plot = rhos_to_plot[
        np.where(np.logical_and(rhos_to_plot > 0, np.abs(rhos_to_plot) <= 1))
    ]
    rhos_to_plot = np.append(rhos_to_plot, 1.0)
    print("plotting surfaces at these rhos: ", rhos_to_plot)
    labelled = False
    for r in np.sort(rhos_to_plot):
        dat = eq.compute(
            ["R", "Z"],
            grid=LinearGrid(
                rho=r, theta=np.linspace(0, np.pi, 100), zeta=np.array([0])
            ),
        )
        if labelled:
            plt.plot(dat["R"], dat["Z"], "c--", lw=4)
        else:
            plt.plot(dat["R"], dat["Z"], "c--", lw=4, label="DESC Vacuum Equilibrium")
            labelled = True
        if np.isclose(r, 1.0):
            plt.plot(
                dat["R"], dat["Z"], "m--", lw=4, label="DESC Vacuum Equilibrium Bdry"
            )

    return ax
