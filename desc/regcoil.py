"""Python implementation of REGCOIL algorithm."""
import time

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

from desc.backend import jit
from desc.basis import DoubleFourierSeries
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.geometry.utils import rpz2xyz, rpz2xyz_vec
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import ToroidalMagneticField
from desc.transform import Transform


######################### Define functions #######################
@jax.jit
def biot_loop(re, rs, J, dV):
    """Generic biot savart law.

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


######################################################################


def run_regcoil(  # noqa: C901 fxn too complex
    eqname,
    helicity_ratio=0,
    alpha=0,
    basis_M=16,
    basis_N=16,
    source_grid_M=15,
    source_grid_N=15,
    eval_grid_M=15,
    eval_grid_N=15,
    winding_surf=None,
    scan=False,
    nscan=30,
    scan_lower=-30,
    scan_upper=-1,
    external_TF_fraction=0,
    external_TF_scan=False,
    external_TF_scan_upper=1.0,
    external_TF_scan_lower=0,
    external_TF_scan_n=10,
    jac=None,
    return_A=False,
    show_plots=False,
    verbose=1,
    dirname=".",
):
    """Python regcoil to find single-valued current potential.

    Parameters
    ----------
    eqname: str, name of DESC eq to calculate the current on the winding surface which
     makes Bnormal=0 on LCFS
    helicity_ratio: int, used to determine if coils are modular (0) or helical (!=0)
    alpha: float, regularization parameter, >0, regularizes minimization of Bn
        with minimization of K on winding surface
        i.e. larger alpha, simpler coilset and smaller currents, but worse Bn
    basis_M: int, poloidal resolution of Single valued partof current potential
    basis_N: int, Toroidal resolution of Single valued partof current potential
    source_grid_M: int, poloidal resolution of source grid, defaults to basis_M*2
    source_grid_N: int, Toroidal resolution of source grid, defaults to basis_N*NFP*2
    NOTE: this grid resolution should be at least basis_N*NFP*2 in order to properly
          resolve the current potential on the winding surface
    eval_grid_M: int, poloidal resolution of evaluation grid on plasma surface,
                defaults to basis_M*3
    Eval_grid_N: int, Toroidal resolution of evaluation grid, defaults to basis_N*3
    NOTE: this grid resolution may need to be much higher than the basis resolution
        used in order to accurately resolve the Bnormal distribution,
        think for example of a uniform continuous poloidal current density which
        creates a simple 1/R Tf field. That field's Bnormal distribution will
        require many Fourier modes to describe on the surface
        of a stellarator like W7-X.

    winding_surf: FourierRZToroidalSurface, surface to find
        current potential on. If None, defaults to NT_tao circular torus
    scan: bool, whether to scan over alpha values starting from 0 and ending
     at the given alpha and return a plot of the chiB vs alpha
    nscan: int, number of alpha values to scan over
    scan_lower: int, default -30, power of 10 (i.e. 10^(-30)) that is the
        lower bound of the alpha values to scan over
    scan_upper: int, default -1, power of 10 (i.e. 10^(-1)) that is the
        upper bound of the alpha values to scan over
    external_TF_fraction: float, default 0
        how much TF is provided by coils external to the
        winding surface being considered.
    external_TF_fraction_scan: bool, default False
        whether to scan over TF fraction
    external_TF_fraction_scan_upper: float, default 1.0
        upper limit of TF fraction scan
    external_TF_fraction_scan_upper: float, default 0.0
        lower limit of TF fraction scan
    external_TF_fraction_scann: int, default 10
        number of steps in TF fraction scan
    jac: jacobian to use (must agree in size with eq basis, grid and basis res)
    return_A: bool, default False, whether to return the jacobian matrix A
        jacobian of the Bnormal on the plasma surface wrt the phi_mn coeffs
    show_plots: bool, default false
        whether to show plots or not
    verbose: int, level of verbosity
    dirname: where to save figures, defaults to current directory
            should not include the trailing '/'

    Returns
    -------
    phi_mn_opt: array, the double fourier series coefficients for the
         single-valued part of the current potential
         if scan=True, this is a list of length n_scan containing
         the phi_mn corresponding to each scanned value of alpha
         if external_TF_scan=True, is a dict of the TF fractions
         and the entries are the arrays of phi_mn for different alpha
    curr_pot_trans: Transform, transform for the basis used for the phi_mn_opt,
         can find value of phi_SV with curr_pot_trans.transform(phi_mn_opt)
    I: float, net toroidal current linking the plasma and coils,
         determined by helicity ratio and G
    G: float, net poloidal current linking the plasma and coils,
         determined by the equilibrium toroidal field
         note: this value is the value after subtraction of the
         external linking poloidal current if external_TF_fraction > 0
    phi_total_function: fxn, accepts a LinearGrid object (or any grid),
         and returns the total current potential on that grid. Convenience function.
    TF_B: ToroidalMagneticField, the TF provided by external TF coils.
    lowest_idx_without_saddles: int, the lowest index of the
        phi_mn_opt array that has contours without saddle coils.
        only returned if scan=True
    """
    # TODO: add defaults for grid values, as stated in docstring
    ##### Load in DESC equilbrium #####
    if isinstance(eqname, str):
        eqfv = load(eqname)
        eq = eqfv
    elif isinstance(eqname, Equilibrium):
        eq = eqname
    else:
        raise TypeError(f"not a valid input for eqname, got {type(eqname)}")
    if hasattr(eq, "__len__"):
        eq = eq[-1]
    ########### calculate quantities on DESC  plasma surface #############

    sgrid = LinearGrid(M=source_grid_M, N=source_grid_N)  # source grid must have NFP=1
    egrid = LinearGrid(M=eval_grid_M, N=eval_grid_N, NFP=eq.NFP)
    edata = eq.compute(["e^rho", "R", "Z", "phi", "e_theta", "e_zeta"], egrid)

    ne = jnp.cross(
        edata["e_theta"], edata["e_zeta"], axis=-1
    )  # surface normal on evaluation surface (ie plasma bdry)
    ne = rpz2xyz_vec(ne, phi=egrid.nodes[:, 2])
    ne_mag = jnp.linalg.norm(ne, axis=-1)
    ne = ne / ne_mag[:, None]
    re = jnp.array(
        [edata["R"], egrid.nodes[:, 2], edata["Z"]]
    ).T  # evaluation points on plasma bdry

    if winding_surf is None:
        # use nt tao as default
        R0_ves = 0.7035  # m
        a_ves = 0.0365  # m

        winding_surf = FourierRZToroidalSurface(
            R_lmn=np.array([R0_ves, -a_ves]),  # boundary coefficients in m
            Z_lmn=np.array([a_ves]),
            modes_R=np.array([[0, 0], [1, 0]]),  # [M, N] boundary Fourier modes
            modes_Z=np.array([[-1, 0]]),
            NFP=1,  # number of (toroidal) field periods
        )
    # make basis for current potential double fourier series
    curr_pot_basis = DoubleFourierSeries(M=basis_M, N=basis_N, NFP=eq.NFP)
    curr_pot_trans = Transform(sgrid, curr_pot_basis, derivs=1, build=True)

    # calc quantities on winding surface (source)
    rs = winding_surf.compute_coordinates(
        grid=sgrid
    )  # surface normal on winding surface
    rs_t = winding_surf.compute_coordinates(grid=sgrid, dt=1)
    rs_z = winding_surf.compute_coordinates(grid=sgrid, dz=1)

    # calculate net enclosed poloidal and toroidal currents
    G_tot = eq.compute("G", grid=sgrid)["G"][0] / mu_0 * 2 * np.pi  # poloidal
    # 2pi factor is present in regcoil code
    #  https://github.com/landreman/regcoil/blob
    # /99f9abf8b0b0c6ec7bb6e7975dbee5e438808162/regcoil_init_plasma_mod.f90#L500

    # define fxns to calculate Bnormal from SV part of phi and from secular part
    def B_from_K_SV(phi_mn, I, G, re, rs, rs_t, rs_z, ne):
        """B from single value part of K from REGCOIL eqn 4."""
        phi_t = curr_pot_trans.transform(phi_mn, dt=1)
        phi_z = curr_pot_trans.transform(phi_mn, dz=1)
        ns_mag = jnp.linalg.norm(jnp.cross(rs_t, rs_z), axis=1)
        K = -(phi_t * (1 / ns_mag) * rs_z.T).T + (phi_z * (1 / ns_mag) * rs_t.T).T
        dV = sgrid.weights * jnp.linalg.norm(jnp.cross(rs_t, rs_z, axis=-1), axis=-1)
        B = biot_loop(
            rpz2xyz(re), rpz2xyz(rs), rpz2xyz_vec(K, phi=sgrid.nodes[:, 2]), dV
        )
        return jnp.sum(B * ne, axis=-1)

    def B_from_K_secular(I, G, re, rs, rs_t, rs_z, ne):
        """B from secular part of K, i.e. B^GI_{normal} from REGCOIL eqn 4."""
        phi_t = I / (2 * jnp.pi)
        phi_z = G / (2 * jnp.pi)
        ns_mag = jnp.linalg.norm(jnp.cross(rs_t, rs_z), axis=1)
        K = -(phi_t * (1 / ns_mag) * rs_z.T).T + (phi_z * (1 / ns_mag) * rs_t.T).T
        dV = sgrid.weights * jnp.linalg.norm(jnp.cross(rs_t, rs_z, axis=-1), axis=-1)
        B = biot_loop(
            rpz2xyz(re), rpz2xyz(rs), rpz2xyz_vec(K, phi=sgrid.nodes[:, 2]), dV
        )
        return jnp.sum(B * ne, axis=-1)

    # $B$ is linear in $K$ as long as the geometry is fixed
    # so just need to evaluate the jacobian

    if jac is None:
        if external_TF_scan:
            A_fun = jit(jax.jacfwd(B_from_K_SV))
        else:
            A_fun = jax.jacfwd(B_from_K_SV)
    else:
        A = jac
        print("Using passed-in Jacobian")

    if not external_TF_scan:
        external_TFs = [external_TF_fraction]
    else:
        external_TFs = np.linspace(
            external_TF_scan_lower,
            external_TF_scan_upper,
            external_TF_scan_n,
            endpoint=True,
        )

    all_phi_mns = {}

    for i, external_TF_fraction in enumerate(external_TFs):
        assert (
            external_TF_fraction >= 0 and external_TF_fraction <= 1
        ), "external_TF_fraction must be a float between 0 and 1!"

        G_ext = external_TF_fraction * G_tot

        G = G_tot - G_ext

        if helicity_ratio == 0:  # modular coils
            I = 0  # toroidal
        else:
            I = G / helicity_ratio / eq.NFP  # toroidal
        # initialize phi_mn
        phi_mn = jnp.zeros(curr_pot_basis.num_modes)
        # calculate jacobian A at this external TF fraction
        t_start = time.time()
        print(f"Starting Jacobian Calculation {i}/{len(external_TFs)}")
        A = A_fun(phi_mn, I, G, re, rs, rs_t, rs_z, ne)
        print(f"Jacobian Calculation finished, took {time.time()-t_start} s")

        B_GI_normal = B_from_K_secular(I, G, re, rs, rs_t, rs_z, ne)
        Bn = np.zeros_like(B_GI_normal)
        if external_TF_fraction == 0:
            Bn_ext = np.zeros_like(B_GI_normal)
            TF_B = ToroidalMagneticField(B0=0, R0=1)
        else:
            TF_B = ToroidalMagneticField(B0=mu_0 * G_ext / 2 / jnp.pi, R0=R0_ves)
            Bn_ext = B_from_K_secular(0, G_ext, re, rs, rs_t, rs_z, ne)
            # TODO: check that this is the same as calculating B from TF_B...

        rhs = -(Bn + Bn_ext + B_GI_normal).T @ A

        # alpha is regularization param, if >0,
        # makes simpler coils (less current density), but worse Bn
        alphas = (
            [alpha]
            if not scan
            else jnp.concatenate(
                (jnp.array([0.0]), jnp.logspace(scan_lower, scan_upper, nscan))
            )
        )
        chi2Bs = []
        phi_mns = []

        for alpha in alphas:
            printstring = f"Calculating Phi_SV for alpha = {alpha:1.5e}"
            if verbose > 0:
                print(
                    "#" * len(printstring)
                    + "\n"
                    + printstring
                    + "\n"
                    + "#" * len(printstring)
                )

            # calculate phi
            phi_mn_opt = jnp.linalg.pinv(A.T @ A + alpha * jnp.eye(A.shape[1])) @ rhs

            phi_mns.append(phi_mn_opt)

            Bn_SV = A @ phi_mn_opt
            Bn_tot = Bn_SV + Bn + B_GI_normal + Bn_ext
            chi_B = np.sum(Bn_tot * Bn_tot * ne_mag * egrid.weights)
            chi2Bs.append(chi_B)
            if verbose > 1:
                printstring = f"chi^2 B = {chi_B:1.5e}"
                print(printstring)
                printstring = f"min Bnormal = {np.min(Bn_tot):1.5e}"
                print(printstring)
                printstring = f"Max Bnormal = {np.max(Bn_tot):1.5e}"
                print(printstring)
                printstring = f"Avg Bnormal = {np.mean(Bn_tot):1.5e}"
                print(printstring)
        all_phi_mns[external_TF_fraction] = phi_mns
        lowest_idx_without_saddles = -1
        saddles_exists_bools = []
        ncontours = 20

        if scan and show_plots:
            plt.figure(figsize=(10, 8))
            plt.rcParams.update({"font.size": 24})
            plt.scatter(alphas, chi2Bs, label="python regcoil")
            plt.xlabel("alpha (regularization parameter)")
            plt.ylabel(r"$\chi^2_B = \int \int B_{normal}^2 dA$ ")
            plt.yscale("log")
            plt.xscale("log")

            nlambda = len(chi2Bs)
            max_nlambda_for_contour_plots = 16
            numPlots = min(nlambda, max_nlambda_for_contour_plots)
            ilambda_to_plot = np.sort(
                list(set(map(int, np.linspace(1, nlambda, numPlots))))
            )
            numPlots = len(ilambda_to_plot)

            numCols = int(np.ceil(np.sqrt(numPlots)))
            numRows = int(np.ceil(numPlots * 1.0 / numCols))

            mpl.rc("xtick", labelsize=7)
            mpl.rc("ytick", labelsize=7)
            plt.title(f"External TF fraction = {external_TF_fraction}")

            ########################################################
            # Plot total current potential
            ########################################################

            plt.figure(figsize=(15, 8))

            for whichPlot in range(numPlots):
                plt.subplot(numRows, numCols, whichPlot + 1)
                phi_mn_opt = phi_mns[ilambda_to_plot[whichPlot] - 1]
                phi = curr_pot_trans.transform(phi_mn_opt)

                phi_tot = (
                    phi
                    + G / 2 / np.pi * curr_pot_trans.grid.nodes[:, 2]
                    + I / 2 / np.pi * curr_pot_trans.grid.nodes[:, 1]
                )
                plt.rcParams.update({"font.size": 18})

                cdata = plt.contour(
                    sgrid.nodes[sgrid.unique_zeta_idx, 2],
                    sgrid.nodes[sgrid.unique_theta_idx, 1],
                    (phi_tot).reshape(sgrid.num_theta, sgrid.num_zeta, order="F"),
                    levels=ncontours,
                )
                plt.ylabel("theta")
                plt.xlabel("zeta")
                plt.title(
                    f"lambda= {alphas[ilambda_to_plot[whichPlot] - 1]:1.5e}"
                    + f" index = {ilambda_to_plot[whichPlot] - 1}",
                    fontsize="x-small",
                )
                plt.colorbar()
                plt.xlim([0, 2 * np.pi / eq.NFP])
                saddles_exist_in_potential = False
                for j in range(ncontours):
                    try:
                        p = cdata.collections[j].get_paths()[0]
                    except Exception:
                        continue
                    v = p.vertices
                    temp_zeta = v[:, 0]
                    if np.abs(temp_zeta[-1] - temp_zeta[0]) < 1e-2:
                        saddles_exist_in_potential = True
                        break
                saddles_exists_bools.append(saddles_exist_in_potential)

            plt.tight_layout()
            plt.figtext(
                0.5,
                0.995,
                "Total current potential"
                + f" at External TF fraction = {external_TF_fraction}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize="small",
            )
            if not np.any(saddles_exists_bools):
                lowest_idx_without_saddles = 0
                pass
            elif np.any(saddles_exists_bools):
                lowest_idx_without_saddles = np.where(np.asarray(saddles_exists_bools))[
                    0
                ][0]
                print(
                    "Lowest alpha value without saddle coil contours in potential"
                    + f" = {alphas[lowest_idx_without_saddles]:1.5e}"
                )
            else:
                lowest_idx_without_saddles = -1
                print(
                    "No alpha value yielded a current potential without"
                    + " saddle coil contours or badly behaved contours!!"
                )
            plt.savefig(f"{dirname}/Scan_ext_TF{external_TF_fraction}.png")

        if show_plots:

            plt.figure(figsize=(10, 10))
            plt.rcParams.update({"font.size": 26})
            plt.figure(figsize=(8, 8))
            plt.contourf(
                egrid.nodes[egrid.unique_zeta_idx, 2],
                egrid.nodes[egrid.unique_theta_idx, 1],
                (Bn_tot).reshape(egrid.num_theta, egrid.num_zeta, order="F"),
            )
            plt.ylabel("theta")
            plt.xlabel("zeta")
            plt.title("Bnormal on plasma surface")
            plt.colorbar()
            plt.xlim([0, 2 * np.pi / eq.NFP])

        phi = curr_pot_trans.transform(phi_mn_opt)

        phi_tot = (
            phi
            + G / 2 / np.pi * curr_pot_trans.grid.nodes[:, 2]
            + I / 2 / np.pi * curr_pot_trans.grid.nodes[:, 1]
        )

        if show_plots and not scan:
            plt.figure(figsize=(10, 10))
            plt.rcParams.update({"font.size": 18})
            plt.figure(figsize=(8, 8))
            plt.contourf(
                sgrid.nodes[sgrid.unique_zeta_idx, 2],
                sgrid.nodes[sgrid.unique_theta_idx, 1],
                (phi_tot).reshape(sgrid.num_theta, sgrid.num_zeta, order="F"),
                levels=ncontours,
            )
            plt.colorbar()
            plt.contour(
                sgrid.nodes[sgrid.unique_zeta_idx, 2],
                sgrid.nodes[sgrid.unique_theta_idx, 1],
                (phi_tot).reshape(sgrid.num_theta, sgrid.num_zeta, order="F"),
                levels=ncontours,
            )
            plt.ylabel("theta")
            plt.xlabel("zeta")
            plt.title("Total Current Potential on winding surface")

            plt.xlim([0, 2 * np.pi / eq.NFP])
        if show_plots and external_TF_scan:
            plt.figure()

            plt.contour(
                egrid.nodes[egrid.unique_zeta_idx, 2],
                egrid.nodes[egrid.unique_theta_idx, 1],
                (Bn_ext).reshape(egrid.num_theta, egrid.num_zeta, order="F"),
                levels=ncontours,
            )
            plt.ylabel("theta")
            plt.xlabel("zeta")
            plt.title("external coil current B normal on plasma surface")

            plt.xlim([0, 2 * np.pi / eq.NFP])

            plt.figure()

            plt.contour(
                egrid.nodes[egrid.unique_zeta_idx, 2],
                egrid.nodes[egrid.unique_theta_idx, 1],
                (B_GI_normal).reshape(egrid.num_theta, egrid.num_zeta, order="F"),
                levels=ncontours,
            )
            plt.ylabel("theta")
            plt.xlabel("zeta")
            plt.title("G and I B normal on plasma surface")

            plt.xlim([0, 2 * np.pi / eq.NFP])

    def phi_total_function(grid):
        """Helper fxn to calculate the total phi given a LinearGrid."""
        trans = Transform(grid, curr_pot_basis)
        phi = trans.transform(phi_mn_opt)
        return (
            phi
            + G / 2 / np.pi * trans.grid.nodes[:, 2]
            + I / 2 / np.pi * trans.grid.nodes[:, 1]
        )

    if not external_TF_scan:
        all_phi_mns = all_phi_mns[list(all_phi_mns.keys())[0]]
    if not scan and not external_TF_scan:
        all_phi_mns = all_phi_mns[0]

    if scan:
        return (
            all_phi_mns,
            alphas,
            curr_pot_trans,
            I,
            G,
            phi_total_function,
            TF_B,
            chi_B,
            lowest_idx_without_saddles,
        )

    return (
        all_phi_mns,
        curr_pot_trans,
        I,
        G,
        phi_total_function,
        TF_B,
        chi_B,
        lowest_idx_without_saddles,
    )
