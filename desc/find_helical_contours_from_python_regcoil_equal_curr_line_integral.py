"""Find helical coils from surface current potential."""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import desc.io
from desc.backend import jnp
from desc.coils import CoilSet
from desc.compute.utils import cross, dot
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid
from desc.plotting import plot_3d
from desc.transform import Transform

# make this a fxn that takes in the phi_MN and does the usual thing
# does theta convention matter for this?


def find_helical_coils(  # noqa: C901 - FIXME: simplify this
    phi_mn_desc_basis,
    basis,
    eqname,
    net_toroidal_current,
    net_poloidal_current,
    alpha,
    desirednumcoils=10,
    step=2,
    ntheta=128,
    winding_surf=None,
    coilsFilename="coils.txt",
    maxiter=25,
    dirname=".",
    method="Nelder-Mead",
    equal_current=True,
    initial_guess=None,
    save_figs=True,
):
    """Find helical coils from a surface current potential.

    Parameters
    ----------
    phi_mn_desc_basis : ndarray
        The DoubleFourierSeries coefficients for the surface current potential.
    basis : DoubleFourierSeries
        Basis corresponding to the phi_mn_desc_basis
    eqname : str or Equilibrium
        The DESC equilibrum the surface current potential was found for
        If str, assumes it is the name of the equilibrium .h5 output and will
        load it
    net_toroidal_current : float
        Net current linking the plasma and the coils toroidally
        Denoted I in the algorithm
        An output of the run_regcoil function
        If nonzero, helical coils are sought
        If 0, then modular coils are sought, and this function is not
        appropriate for that, and will raise an error
    net_poloidal_current : float
        Net current linking the plasma and the coils poloidally
        Denoted G in the algorithm
        an output of the run_regcoil function
    alpha : float
        regularization parameter used in run_regcoil
        #TODO: can remove this and replace with something like
        basename to be used for every saved figure
    desirednumcoils : int, optional
        number of coils to discretize the surface current with, by default 10
    step : int, optional
        Amount of points to step when saving the coil geometry
        by default 2, meaning that every other point will be saved
        if higher, less points will be saved
    ntheta : int, optional
        number of theta points to use in the integration to find the current
        pertaining to each coil, by default 128
    winding_surf : _type_, optional
        Winding surface on which the surface current lies, if None will default
        to a circular torus of R0=0.7035 and a = 0.0365

    coilsFilename : str, optional
        name of txt file to save coils to, by default "coils.txt"
    maxiter : int, optional
        max iterations to use in equal current optimization, by default 25
    dirname : str, optional
        directory to save files to, by default "."
    method : str, optional
        scipy minimization method to use for equal current algorithm,
        by default "Nelder-Mead"
    equal_current : bool, optional
        Whether to optimize the contour positions to find
        coils which have equal currents, by default True
    initial_guess : array-like, optional
        initial guess to use for the contour values in the minimization
        by default None
    save_figs : bool, optional
        whether to save figures, by default True

    Returns
    -------
    coils : CoilSet
        DESC CoilSet object that is a discretization of the input
        surface current on the given winding surface
    """
    if isinstance(eqname, str):
        eq = desc.io.load(eqname)
    elif isinstance(eqname, EquilibriaFamily) or isinstance(eqname, Equilibrium):
        eq = eqname
    if hasattr(eq, "__len__"):
        eq = eq[-1]

    nfp = eq.NFP
    theta_coil = np.linspace(0, 2 * np.pi, ntheta)
    zeta_coil = np.linspace(0, 2 * np.pi / nfp, ntheta)
    dz = zeta_coil[1] - zeta_coil[0]
    zetal_coil = np.arange(0, 2 * np.pi, dz)

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    ##################################################################################
    ################# Make winding surface DESC object #######################
    ##################################################################################

    # get
    # rs: source pts R,phi,Z (just from the winding surface)

    if winding_surf is None:
        R0_ves = 0.7035  # m
        a_ves = 0.0365  # m

        winding_surf = FourierRZToroidalSurface(
            R_lmn=np.array([R0_ves, -a_ves]),  # boundary coefficients in m
            Z_lmn=np.array([a_ves]),
            modes_R=np.array([[0, 0], [1, 0]]),  # [M, N] boundary Fourier modes
            modes_Z=np.array([[-1, 0]]),
            NFP=1,  # number of (toroidal) field periods
        )

    ##################################################################################
    ################# get function to calc phi_tot(theta,zeta) #######################
    ##################################################################################
    # fxn will accept phi_mn and its basis

    # TODO:just use fxns imported from regcoil.py

    def phi_tot_fun_vec(theta, zeta):
        nodes = np.vstack(
            (
                np.zeros_like(theta.flatten(order="F")),
                theta.flatten(order="F"),
                zeta.flatten(order="F"),
            )
        ).T
        trans_temp = Transform(Grid(nodes, sort=False), basis)
        return (
            np.asarray(trans_temp.transform(phi_mn_desc_basis))
            + net_poloidal_current * zeta.flatten(order="F") / 2 / np.pi
            + net_toroidal_current * theta.flatten(order="F") / 2 / np.pi
        )

    def phi_tot_fun_theta_deriv_vec(theta, zeta):
        nodes = np.vstack(
            (
                np.zeros_like(theta.flatten(order="F")),
                theta.flatten(order="F"),
                zeta.flatten(order="F"),
            )
        ).T
        trans_temp = Transform(
            Grid(nodes, sort=False), basis, derivs=np.array([[0, 1, 0]])
        )
        return (
            np.asarray(trans_temp.transform(phi_mn_desc_basis, dt=1))
            + net_toroidal_current / 2 / np.pi
        )

    def surf_current_vec_contravariant_zeta_times_R_times_g_tt(sgrid):
        data = winding_surf.compute(["x", "e_theta", "e_zeta"], grid=sgrid, basis="rpz")
        Rs = data["x"][:, 0]
        rs_t = data["e_theta"]
        rs_z = data["e_zeta"]
        ns_mag = np.linalg.norm(cross(rs_t, rs_z), axis=1)

        phi_t = phi_tot_fun_theta_deriv_vec(sgrid.nodes[:, 1], sgrid.nodes[:, 2])
        g_tt = dot(rs_t, rs_t)

        # "vector" is the K vector in terms of grad(theta) and grad(zeta)
        K_sup_zeta = -phi_t * (1 / ns_mag)

        return K_sup_zeta * Rs * jnp.sqrt(g_tt)

    def phi_tot_fun(theta, zeta):
        if np.shape(theta):
            theta = theta[0]
        trans_temp = Transform(Grid(np.array([[1, theta, zeta]])), basis)
        return (
            np.asarray(trans_temp.transform(phi_mn_desc_basis))
            + net_poloidal_current * zeta / 2 / np.pi
            + net_toroidal_current * theta / 2 / np.pi
        )

    ##### find helicity naively ######################################################

    if np.isclose(net_toroidal_current, 0):
        raise ValueError(
            "this function only works for helical coils, not modular or window pane!"
        )
    # we know that I = -(G - G_ext) / (helicity * NFP)

    helicity = -net_poloidal_current / net_toroidal_current / eq.NFP
    phi_slope = np.sign(helicity)
    ###############################################

    def get_integration_points_and_line_elems(
        contour_theta_halfway,
        contours_were_sorted,
        nthetas=50,
    ):
        # simply gets the theta pts btwn the halfway contours for each coil contour
        # and the dtheta for each given the ntheta
        N_trial_contours = len(contour_theta_halfway)
        coil_nodes = []  # will be a list of lists
        coil_dthetas = []  # will be a list of lists of dtheta for each point
        for i in range(0, N_trial_contours):
            phis = np.zeros(
                (nthetas,)
            )  # assumes we have contours which are intially at phi=0
            if i != 0:  # do as normal
                thetas = np.linspace(
                    contour_theta_halfway[i - 1],
                    contour_theta_halfway[i],
                    nthetas,
                    endpoint=False,
                )
            elif (
                not contours_were_sorted
            ):  # if i==0, then must use the last and the first halfway
                # contour periodicity to find the integration domain
                thetas = jnp.linspace(
                    contour_theta_halfway[i],
                    contour_theta_halfway[-1] + 2 * np.pi * np.sign(phi_slope),
                    nthetas,
                    endpoint=False,
                )
            elif (
                contours_were_sorted
            ):  # if i==0, then must use the last and the first halfway contour
                # periodicity to find the integration domain
                # here I assume this contour is on the bottom...
                # when it could be on top too
                thetas = np.linspace(
                    contour_theta_halfway[i - 1],
                    contour_theta_halfway[i],
                    nthetas,
                    endpoint=False,
                )
            dthetas = np.abs(thetas[1] - thetas[0]) * np.ones_like(phis)
            coil_nodes.append((np.asarray(thetas), np.asarray(phis)))
            coil_dthetas.append(dthetas)
        return coil_nodes, coil_dthetas

    ################################################################
    # find contours of constant phi
    ################################################################
    # Phi is linear in theta at zeta=0 (bc is a sin series for phi SV), so can
    # have theta be the optimization varibale
    # and theta halfway is literally the theta halfway btwn,
    #  so no need to call matploltib

    # grids of values to use to approximately find the thetas
    # corresponding to the contour Phi value halfway between two contours
    theta_lookup_grid = np.linspace(
        -2 * np.pi / eq.NFP, 2 * np.pi + 2 * np.pi / eq.NFP, 10000
    )
    contour_vals_on_theta_lookup_grid = phi_tot_fun_vec(
        theta_lookup_grid, np.zeros_like(theta_lookup_grid)
    )
    theta_of_contour_val = interp1d(
        x=contour_vals_on_theta_lookup_grid, y=theta_lookup_grid
    )

    def find_contours_and_current_variance(
        contours, return_full_info=False, show_plots=False, nthetas=200
    ):
        """Accepts a list of current potential contour values of length Ncoils+1.

        returns variance of current in the coils

        """
        N_trial_contours = len(contours) - 1
        # thetas is the decision variable

        if not contours[1] - contours[0] > 1:
            contours = np.sort(contours)
            contours_were_sorted = True
        else:
            contours_were_sorted = False

        theta_halfways = []
        contour_theta = []

        for i in range(N_trial_contours):
            halfway_contour_val = (contours[i] + contours[i + 1]) / 2
            theta_halfways.append(theta_of_contour_val(halfway_contour_val))
            contour_theta.append(theta_of_contour_val(contours[i]))

        ################################################################
        # Find contours that are halfway between each coil contour,
        # to set the bounds of integration of surface current for
        # each coil
        ################################################################

        # number of theta points at each phi point for integration

        coil_nodes_line, coil_line_elements = get_integration_points_and_line_elems(
            theta_halfways,
            contours_were_sorted,
            nthetas=nthetas,
        )

        if show_plots:
            plt.figure(figsize=(10, 10))

            for i in range(N_trial_contours):  # np.flip(np.arange(desirednumcoils)):
                plt.plot(
                    coil_nodes_line[i][1],
                    coil_nodes_line[i][0],
                    label="line integration bound",
                )
                plt.plot(
                    0,
                    contour_theta[i],
                    "c.",
                    label="Theta0 of Coil to integrate current for",
                )
                if i != 0:
                    plt.plot(
                        0, theta_halfways[i], "k"
                    )  # , label="bounding contours for integration")
                    plt.plot(0, theta_halfways[i - 1], "k")

                elif not contours_were_sorted and i == 0:
                    plt.plot(
                        0,
                        theta_halfways[-1] + 2 * np.pi * np.sign(phi_slope),
                        "k+",
                        label="bounds for integration",
                    )
                    plt.plot(0, theta_halfways[i], "k-")
                    plt.legend()
                elif contours_were_sorted and i == 0:
                    plt.plot(
                        0,
                        theta_halfways[0] + 2 * np.pi * np.sign(phi_slope),
                        "m.",
                        label="bounding contours for integration",
                    )
                    plt.plot(0, theta_halfways[i], "k.")
                    plt.legend()
            plt.xlabel("zeta")
            plt.ylabel("theta")

        ################################################################
        # integrate surface current through the integration bounds for each coil
        # to find the current for each coil
        ################################################################

        coil_currents_line = []  # list of the coil currents
        for icoi, (coords, line_elems) in enumerate(
            zip(coil_nodes_line, coil_line_elements)
        ):
            nodes_surf = np.vstack(
                (
                    jnp.zeros_like(coords[0].flatten(order="F")),
                    coords[0].flatten(order="F"),
                    coords[1].flatten(order="F"),
                )
            ).T
            sgrid = Grid(nodes_surf, sort=False)

            K_sup_z_time_R = surf_current_vec_contravariant_zeta_times_R_times_g_tt(
                sgrid
            )
            current = jnp.sum(K_sup_z_time_R * line_elems)
            coil_currents_line.append(current)

        variance = np.var(coil_currents_line)
        if not show_plots:
            plt.close("all")

        ####### debug #######################
        # should just check that the net toroidal current is the same
        # as what we get after integration...?
        # no it wont be quite that though

        if not return_full_info:
            return variance
        else:  # return all info: contour_theta and coil_currents
            return contour_theta, coil_currents_line

    # TODO: don't need this for the whole zeta/theta domain,
    #  should be able to only use like 2 or 3 repetitions
    # to find the contour then repeat given the discrete
    #  periodicity in zeta, by just rotating it by
    # an angle phi/NFP, just need
    # wide enough in zeta and tall enough inzeta
    # to capture the contour entering at zeta=0
    # and exiting at zeta = XX
    # then rotate it by repeating it 2pi/(2pi-XX)
    # times over the angle 2pi-XX/something
    def find_full_coil_contours(contours, show_plots=True, ax=None, label=None, ls="-"):
        """Accepts a list of current potential contour values of length Ncoils+1.

        returns the theta,zeta points of each contour

        """
        theta_full = theta_coil * np.sign(phi_slope)
        for inn in range(1, int(2 * nfp)):
            theta_full = np.concatenate(
                (theta_full, (theta_coil + 2 * np.pi * inn) * np.sign(phi_slope))
            )
        theta_full = np.append(
            theta_full, theta_full[-1] + theta_full[1]
        )  # add the last point
        zeta_full = np.append(zetal_coil, 2 * np.pi)
        theta_full_2D, zeta_full_2D = np.meshgrid(theta_full, zeta_full, indexing="ij")
        my_tot_full = phi_tot_fun_vec(theta_full_2D, zeta_full_2D).reshape(
            theta_full.size, zeta_full.size, order="F"
        )

        N_trial_contours = len(contours) - 1
        contour_zeta = []
        contour_theta = []
        if not contours[1] - contours[0] > 1:
            contours = np.sort(contours)
        plt.figure(figsize=(18, 10))
        cdata = plt.contour(
            zeta_full_2D.T, theta_full_2D.T, np.transpose(my_tot_full), contours
        )

        contour_zeta = []
        contour_theta = []

        numCoils = 0
        for j in range(N_trial_contours):
            try:
                p = cdata.collections[j].get_paths()[0]
            except Exception:
                print("no path found for given contour")
                continue
            v = p.vertices
            temp_zeta = v[:, 0]
            temp_theta = v[:, 1]
            contour_zeta.append(temp_zeta)
            contour_theta.append(temp_theta)

            numCoils += 1
            if show_plots:
                plt.plot(contour_zeta[-1], contour_theta[-1], "-r", linewidth=1)
                plt.plot(contour_zeta[-1][-1], contour_theta[-1][-1], "sk")
        plt.xlim([0, 2 * np.pi / nfp])
        plt.ylim([0, 2 * np.pi + 2 * np.pi / nfp / 4])

        return contour_theta, contour_zeta

    N_trial_contours = desirednumcoils
    # flip is so the contour levels are increasing
    #  (this may not be necessary dep. on contour direction)
    # adding extra contour above and below the ones we care about,
    #  so we can then find halfway point btwn them

    theta0s = np.flip(
        np.linspace(
            0,
            (2 * np.pi + 2 * np.pi / N_trial_contours) * np.sign(phi_slope),
            N_trial_contours + 1,
            endpoint=False,
        )
    )

    contours = []
    for t in theta0s:
        contours.append(float(phi_tot_fun(t, 0.0)[0]))  # contour values
    import time

    if initial_guess:
        assert len(initial_guess) == len(contours)
        contours = initial_guess

    xs = [contours]
    fun_vals = [find_contours_and_current_variance(contours)]

    contour_theta_initial, contour_zeta_initial = find_full_coil_contours(contours)

    if equal_current:
        t_start = time.time()

        def callback(x):
            curr_val = find_contours_and_current_variance(x)
            print(f"Iteration: {len(xs)} Time elapsed = {time.time()-t_start} s")
            print(f"Current function value: {curr_val:1.3e}")
            print(f"Current x: {x}")

            xs.append(x)
            fun_vals.append(curr_val)
            return False

        result = minimize(
            find_contours_and_current_variance,
            contours,
            options={"maxiter": maxiter, "disp": True},
            callback=callback,
            method=method,
        )
        t_end = time.time()
        print(f"Optimization for coils took {t_end-t_start} s")
        contours = result.x

    final_coil_theta0s, coil_currents = find_contours_and_current_variance(
        contours, return_full_info=True, show_plots=True
    )
    print("initial coil thetas:", theta0s)
    print("final coil thetas:", final_coil_theta0s)

    contour_theta, contour_zeta = find_full_coil_contours(contours)

    ################################################################
    # Find the XYZ points in real space of the coil contours
    ################################################################
    if save_figs:
        fig, ax = plot_3d(eq, "|B|", figsize=(12, 12))
    else:
        ax = None

    def find_XYZ_points(
        theta_pts, zeta_pts, surface, find_min_dist=None, ax=None, ls="-", label=None
    ):
        contour_X = []
        contour_Y = []
        contour_Z = []

        for thetas, zetas in zip(theta_pts, zeta_pts):
            coords = surface.compute(
                "x",
                grid=Grid(
                    jnp.vstack((jnp.zeros_like(thetas), thetas, zetas)).T, sort=False
                ),
                basis="xyz",
            )["x"]
            contour_X.append(coords[:, 0])
            contour_Y.append(coords[:, 1])
            contour_Z.append(coords[:, 2])
        if save_figs:
            if ax is None:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(projection="3d")
            for j in range(len(contour_X)):
                if j == 0:
                    ax.plot(contour_X[j], contour_Y[j], contour_Z[j], ls, label=label)
                else:
                    ax.plot(contour_X[j], contour_Y[j], contour_Z[j], ls)

        # # Find the point of minimum separation
        # if find_min_dist:
        #     minSeparation2 = 1.0e20
        #     for whichCoil1 in range(desirednumcoils):
        #         for whichCoil2 in range(whichCoil1):
        #             for whichPoint in range(len(contour_X[whichCoil1])):
        #                 dx = contour_X[whichCoil1][whichPoint] - contour_X[whichCoil2]
        #                 dy = contour_Y[whichCoil1][whichPoint] - contour_Y[whichCoil2]
        #                 dz = contour_Z[whichCoil1][whichPoint] - contour_Z[whichCoil2]
        #                 separation2 = dx * dx + dy * dy + dz * dz
        #                 this_minSeparation2 = np.min(separation2)
        #                 if this_minSeparation2 < minSeparation2:
        #                     minSeparation2 = this_minSeparation2

        #     print(
        #         f"Minimum coil-coil separation: {np.sqrt(minSeparation2)*1000:3.2f} mm"
        #     )
        return contour_X, contour_Y, contour_Z, ax

    contour_X, contour_Y, contour_Z, ax = find_XYZ_points(
        contour_theta,
        contour_zeta,
        winding_surf,
        find_min_dist=True,
        label="Final",
        ax=ax,
    )
    _, _, _, ax = find_XYZ_points(
        contour_theta_initial,
        contour_zeta_initial,
        winding_surf,
        find_min_dist=False,
        label="Initial",
        ls="--",
        ax=ax,
    )

    if save_figs:
        ax.legend()

    figfilename = f"coil_3d_ncoil_{desirednumcoils}_alpha_{alpha:1.4e}_{dirname}.png"
    if save_figs:
        plt.savefig(f"{dirname}/{figfilename}")

    ################

    # Write coils file
    write_coil = True
    if write_coil:
        with open(coilsFilename, "w") as f:
            f.write("periods " + str(nfp) + "\n")
            f.write("begin filament\n")
            f.write("mirror NIL\n")

            for j in range(desirednumcoils):
                N = len(contour_X[j])
                thisCurrent = (
                    np.mean(coil_currents) if equal_current else coil_currents[j]
                )
                for k in range(0, N, step):
                    f.write(
                        "{:14.22e} {:14.22e} {:14.22e} {:14.22e}\n".format(
                            contour_X[j][k],
                            contour_Y[j][k],
                            contour_Z[j][k],
                            thisCurrent,
                        )
                    )
                # Close the loop
                k = 0
                f.write(
                    "{:14.22e} {:14.22e} {:14.22e} {:14.22e} 1 Modular\n".format(
                        contour_X[j][k], contour_Y[j][k], contour_Z[j][k], 0
                    )
                )
            f.write("end\n")
    final_coilset = CoilSet.from_makegrid_coilfile(coilsFilename)
    ###################
    print(f"Coil current average is {np.mean(coil_currents):1.4e} A")
    print(f"Coil current variance is {np.var(coil_currents):1.4e} A")

    return final_coilset
