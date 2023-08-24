"""Field line tracing from a current potential."""
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import desc.examples
import desc.io
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz_vec
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.magnetic_fields import (
    SumMagneticField,
    ToroidalMagneticField,
    field_line_integrate,
)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


##################################


# TODO: add option to first make a spline field then use that for integration
def trace_from_curr_pot(  # noqa: C901 - FIXME: simplify this
    phi_mn_desc_basis,
    curr_pot_trans,
    eqname,
    net_toroidal_current_Amperes,
    net_poloidal_current_Amperes,
    M=30,
    N=30,
    alpha=0,
    ntransit=100,
    external_TF=None,
    savename=None,
    Rs=None,
    phi0=0,
    surface=None,
):
    """Field line trace from current potential.

    Parameters
    ----------
    phi_mn_desc_basis : _type_
        _description_
    curr_pot_trans : _type_
        _description_
    eqname : _type_
        _description_
    net_toroidal_current_Amperes : _type_
        _description_
    net_poloidal_current_Amperes : _type_
        _description_
    M : int, optional
        _description_, by default 30
    N : int, optional
        _description_, by default 30
    alpha : int, optional
        _description_, by default 0
    ntransit : int, optional
        _description_, by default 100
    external_TF : _type_, optional
        _description_, by default None
    savename : _type_, optional
        _description_, by default None
    Rs : _type_, optional
        _description_, by default None
    phi0 : int, optional
        _description_, by default 0
    surface

    Returns
    -------
    field_R
        _description_
    field_Z
        _description_
    field_phis
        _description_
    """
    if isinstance(eqname, str):
        eq = desc.io.load(eqname)
    elif isinstance(eqname, EquilibriaFamily) or isinstance(eqname, Equilibrium):
        eq = eqname
    if hasattr(eq, "__len__"):
        eq = eq[-1]

    R0_ves = 0.7035  # m
    a_ves = 0.0365  # m

    if surface is None:
        winding_surf = FourierRZToroidalSurface(
            R_lmn=np.array([R0_ves, -a_ves]),  # boundary coefficients in m
            Z_lmn=np.array([-a_ves]),
            modes_R=np.array([[0, 0], [1, 0]]),  # [M, N] boundary Fourier modes
            modes_Z=np.array([[-1, 0]]),
            NFP=1,  # number of (toroidal) field periods
        )
    else:
        winding_surf = surface

    curr_pot_trans.change_resolution(grid=LinearGrid(M=M, N=N))

    @jax.jit
    def biot_loop(re, rs, J, dV):
        """Biot Savart loop.

        Parameters
        ----------
        re : ndarray, shape(n_eval_pts, 3)
            evaluation points
        rs : ndarray, shape(n_src_pts, 3)
            source points
        J : ndarray, shape(n_src_pts, 3)
            current density vector at source points
        dV : ndarray, shape(n_src_pts)
            volume element at source points
        """
        re, rs, J, dV = map(jnp.asarray, (re, rs, J, dV))
        assert J.shape == rs.shape
        JdV = J * dV[:, None]
        B = jnp.zeros_like(re)

        def body(i, B):
            r = re - rs[i, :]
            num = jnp.cross(JdV[i, :], r, axis=-1)
            den = jnp.linalg.norm(r, axis=-1) ** 3
            B = B + jnp.where(den[:, None] == 0, 0, num / den[:, None])
            return B

        return 1e-7 * jax.lax.fori_loop(0, J.shape[0], body, B)

    def get_B_function_from_regcoil_current_potential(phi_mn_desc_basis, M=30, N=30):
        """Given current potential, return fxn that calculates B."""
        # M is grid M for source grid
        # N is grid N for source grid

        # calc surface geometric quantities needed for integration
        sgrid = (
            curr_pot_trans.grid
        )  # LinearGrid(M=M + 1, N=N * eq.NFP + 1, NFP=1)  # source (wind surf)
        # calc quantities on winding surface (source)
        data = winding_surf.compute(
            ["x", "e_theta", "e_zeta"], grid=sgrid
        )  # data on winding surface
        rs = data["x"]
        rs_t = data["e_theta"]
        rs_z = data["e_zeta"]

        # define functions that calc B

        ns_mag = np.linalg.norm(np.cross(rs_t, rs_z), axis=1)

        phi_t = curr_pot_trans.transform(
            phi_mn_desc_basis, dt=1
        ) + net_toroidal_current_Amperes / (2 * np.pi)
        phi_z = curr_pot_trans.transform(
            phi_mn_desc_basis, dz=1
        ) + net_poloidal_current_Amperes / (2 * np.pi)
        ns_mag = np.linalg.norm(np.cross(rs_t, rs_z), axis=1)
        # changed signs here
        K = -(phi_t * (1 / ns_mag) * rs_z.T).T + (phi_z * (1 / ns_mag) * rs_t.T).T

        def B_from_K_trace(re, params=None, basis="rpz", grid=None):
            dV = sgrid.weights * jnp.linalg.norm(
                jnp.cross(rs_t, rs_z, axis=-1), axis=-1
            )
            B = biot_loop(
                rpz2xyz(re), rpz2xyz(rs), rpz2xyz_vec(K, phi=sgrid.nodes[:, 2]), dV
            )
            return xyz2rpz_vec(B, phi=re[:, 1])

        return B_from_K_trace

    # Field line tracing

    R0 = 703.5 / 1000
    r = 36.5 / 1000

    Bfield_currpot = ToroidalMagneticField(B0=1, R0=1)
    Bfield_currpot.compute_magnetic_field = (
        get_B_function_from_regcoil_current_potential(phi_mn_desc_basis, M, N)
    )

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
    field_R, field_Z = field_line_integrate(rrr, np.zeros_like(rrr), phis, Bfield)

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

    data = winding_surf.compute("x", basis="rpz", grid=LinearGrid(rho=1, M=20))["x"]

    R_ves = data[:, 0]
    Z_ves = data[:, 2]
    plt.plot(R_ves, Z_ves, "k", label="Vacuum vessel")
    plt.legend()
    plt.ylabel("Z")
    plt.xlabel("R")

    ax = compare_surfs_DESC_field_line_trace(eq, ax, R_list)
    if not savename:
        plt.savefig(f"{dirname}/trace{ntransit}_transits_{dirname}.png")
    else:
        plt.savefig(f"{dirname}/{savename}.png")
    return field_R, field_Z


def compare_surfs_DESC_field_line_trace(eqname, ax, R_list, savename_poin=None, phi0=0):
    """Compare surfaces of an equilibrium to traced flux surfaces."""
    # given initial ax object which already has stuff plotted,
    #  and a list of R values of different surfaces
    # plot the flux surfaces of a DESC equiilibrium at the rhos corr. to those surfaces
    # those are values should be at theta=zeta=0... which should corr.
    #  to the outboard plane

    if isinstance(eqname, str):
        eq = desc.io.load(eqname)
    elif isinstance(eqname, EquilibriaFamily) or isinstance(eqname, Equilibrium):
        eq = eqname
    if hasattr(eq, "__len__"):
        eq = eq[-1]

    Rgrid = LinearGrid(theta=np.array([np.pi]), zeta=np.array([phi0]), L=100)
    Rcoords_no_isl = eq.compute("R", grid=Rgrid)
    rho_of_R_no_isl = interp1d(
        x=Rcoords_no_isl["R"], y=Rgrid.nodes[:, 0], fill_value="extrapolate"
    )

    rhos_to_plot = rho_of_R_no_isl(np.asarray(R_list))
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
