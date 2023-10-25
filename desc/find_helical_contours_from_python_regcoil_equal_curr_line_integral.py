"""Find helical coils from surface current potential."""
import os
import warnings

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import desc.io
from desc.backend import jit, jnp
from desc.coils import MixedCoilSet
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid
from desc.plotting import plot_3d, plot_coils

# make this a fxn that takes in the phi_MN and does the usual thing
# does theta convention matter for this?


def find_helical_coils(  # noqa: C901 - FIXME: simplify this
    surface_current_field,
    eqname,
    alpha,
    desirednumcoils=10,
    step=2,
    ntheta=200,
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
    surface_current_field : FourierCurrentPotentialField or CurrentPotentialField
        CurrentPotentialField or FourierCurrentPotentialField object to
        discretize into coils.
    eqname : str or Equilibrium
        The DESC equilibrum the surface current potential was found for
        If str, assumes it is the name of the equilibrium .h5 output and will
        load it
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
    if save_figs:
        warnings.warn(
            "Not implemented with plotly yet, so 3D figs will show but not save",
            UserWarning,
        )

    nfp = eq.NFP
    theta_coil = jnp.linspace(0, 2 * jnp.pi, ntheta)
    zeta_coil = jnp.linspace(0, 2 * jnp.pi / nfp, ntheta)
    dz = zeta_coil[1] - zeta_coil[0]
    zetal_coil = jnp.arange(0, 2 * jnp.pi, dz)
    net_toroidal_current = surface_current_field.I
    net_poloidal_current = surface_current_field.G

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    ##################################################################################
    ################# Make winding surface DESC object #######################
    ##################################################################################

    # get
    # rs: source pts R,phi,Z (just from the winding surface)

    winding_surf = surface_current_field

    ##################################################################################
    ################# get function to calc phi_tot(theta,zeta) #######################
    ##################################################################################
    # fxn will accept phi_mn and its basis

    def phi_tot_fun_vec(theta, zeta):
        nodes = jnp.vstack(
            (
                jnp.zeros_like(theta.flatten(order="F")),
                theta.flatten(order="F"),
                zeta.flatten(order="F"),
            )
        ).T
        grid = Grid(nodes, sort=False)
        return surface_current_field.compute("Phi", grid=grid)["Phi"]

    def phi_tot_fun_theta_deriv_vec(theta, zeta):
        nodes = jnp.vstack(
            (
                jnp.zeros_like(theta.flatten(order="F")),
                theta.flatten(order="F"),
                zeta.flatten(order="F"),
            )
        ).T
        grid = Grid(nodes, sort=False)
        return surface_current_field.compute("Phi_t", grid=grid)["Phi_t"]

    ##### find helicity naively ######################################################

    if jnp.isclose(net_toroidal_current, 0):
        raise ValueError(
            "this function only works for helical coils, not modular or window pane!"
        )
    # we know that I = -(G - G_ext) / (helicity * NFP)

    helicity = -net_poloidal_current / net_toroidal_current / eq.NFP
    phi_slope = jnp.sign(helicity)
    ###############################################

    def get_integration_points_and_line_elems(
        contour_theta_halfway,
        contours_were_sorted,
        nthetas=50,
    ):
        # simply gets the theta pts btwn the halfway contours for each coil contour
        # and the dtheta for each given the ntheta
        # assumes phi=0 for all contour initial points
        N_trial_contours = len(contour_theta_halfway)
        coil_nodes = []  # will be a list of lists
        coil_dthetas = []  # will be a list of lists of dtheta for each point
        for i in range(0, N_trial_contours):
            if i != 0:  # do as normal
                thetas = jnp.linspace(
                    contour_theta_halfway[i - 1],
                    contour_theta_halfway[i],
                    nthetas,
                    endpoint=False,
                )
            elif (
                contours_were_sorted
            ):  # if i==0, then must use the last and the first halfway
                # contour periodicity to find the integration domain
                thetas = jnp.linspace(
                    contour_theta_halfway[i],
                    contour_theta_halfway[-1] + 2 * jnp.pi * jnp.sign(phi_slope),
                    nthetas,
                    endpoint=False,
                )
            elif (
                not contours_were_sorted
            ):  # if i==0, then must use the last and the first halfway contour
                # periodicity to find the integration domain
                # here I assume this contour is on the bottom...
                # when it could be on top too
                # TODO:
                # this current method does not seem to work when helicity=1
                # (so first contour is on top)
                thetas = jnp.linspace(
                    contour_theta_halfway[-1] - 2 * jnp.pi,
                    contour_theta_halfway[i],
                    nthetas,
                    endpoint=False,
                )
            dthetas = jnp.abs(thetas[1] - thetas[0]) * jnp.ones_like(thetas)
            coil_nodes.append((thetas, jnp.zeros_like(thetas)))
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
    theta_lookup_grid = jnp.linspace(
        -2 * jnp.pi / eq.NFP, 2 * jnp.pi + 2 * jnp.pi / eq.NFP, 10000
    )
    contour_vals_on_theta_lookup_grid = phi_tot_fun_vec(
        theta_lookup_grid, jnp.zeros_like(theta_lookup_grid)
    )
    conts_sorted_inds = jnp.argsort(contour_vals_on_theta_lookup_grid)

    theta_of_contour_val_fun = lambda xx: jnp.interp(
        xx,
        xp=contour_vals_on_theta_lookup_grid[conts_sorted_inds],
        fp=theta_lookup_grid[conts_sorted_inds],
    )
    theta_of_contour_val = jit(theta_of_contour_val_fun)

    # TODO: also precompute and interpolate Phi_t, so no need
    # to create grid objects inside the variance function

    # TODO: constrain contours to lie between 0 and 2pi

    def find_contours_and_current_variance(
        contours, return_full_info=False, show_plots=False, nthetas=200
    ):
        """Accepts a list of current potential contour values of length Ncoils+1.

        returns variance of current in the coils

        """
        N_trial_contours = len(contours) - 1
        # thetas is the decision variable

        if not contours[1] - contours[0] > 1:
            contours = jnp.sort(jnp.asarray(contours))
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

            for i in range(N_trial_contours):  # jnp.flip(np.arange(desirednumcoils)):
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
                        theta_halfways[-1] + 2 * jnp.pi * jnp.sign(phi_slope),
                        "k+",
                        label="bounds for integration",
                    )
                    plt.plot(0, theta_halfways[i], "k-")
                    plt.legend()
                elif contours_were_sorted and i == 0:
                    plt.plot(
                        0,
                        theta_halfways[0] + 2 * jnp.pi * jnp.sign(phi_slope),
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

        # how to do so in a way that is not appending to a list?
        elem_sum = 0
        coil_currents_line = jnp.zeros(N_trial_contours)  # list of the coil currents
        for icoi, (coords, line_elems) in enumerate(
            zip(coil_nodes_line, coil_line_elements)
        ):
            # only need to integrate Phi_t to find net toroidal current
            # through a given dtheta
            Phi_t_vals = phi_tot_fun_theta_deriv_vec(
                coords[0].flatten(), coords[1].flatten()
            )

            current = jnp.sum(Phi_t_vals * line_elems)
            coil_currents_line = coil_currents_line.at[icoi].set(current)
            elem_sum += jnp.sum(line_elems)
        if not jnp.isclose(jnp.sum(coil_currents_line), net_toroidal_current):
            print(
                "net toroidal current of coils does not match I! something went wrong"
            )
            print(jnp.sum(coil_currents_line))
            print(net_toroidal_current)
            print(f"{elem_sum=}")

        variance = jnp.var(coil_currents_line)

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
        theta_full = theta_coil * jnp.sign(phi_slope)
        for inn in range(1, int(2 * nfp)):
            theta_full = jnp.concatenate(
                (theta_full, (theta_coil + 2 * jnp.pi * inn) * jnp.sign(phi_slope))
            )
        theta_full = jnp.append(
            theta_full, theta_full[-1] + theta_full[1]
        )  # add the last point
        zeta_full = jnp.append(zetal_coil, 2 * jnp.pi)
        theta_full_2D, zeta_full_2D = jnp.meshgrid(theta_full, zeta_full, indexing="ij")
        my_tot_full = phi_tot_fun_vec(theta_full_2D, zeta_full_2D).reshape(
            theta_full.size, zeta_full.size, order="F"
        )

        N_trial_contours = len(contours) - 1
        contour_zeta = []
        contour_theta = []
        plt.figure(figsize=(18, 10))
        cdata = plt.contour(
            zeta_full_2D.T, theta_full_2D.T, jnp.transpose(my_tot_full), contours
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
        plt.xlim([0, 2 * jnp.pi / nfp])
        plt.ylim([0, 2 * jnp.pi + 2 * jnp.pi / nfp / 4])

        return contour_theta, contour_zeta

    N_trial_contours = desirednumcoils
    # flip is so the contour levels are increasing
    #  (this may not be necessary dep. on contour direction)
    # adding extra contour above and below the ones we care about,
    #  so we can then find halfway point btwn them

    theta0s = jnp.linspace(
        0,
        (2 * jnp.pi + 2 * jnp.pi / N_trial_contours) * jnp.sign(phi_slope),
        N_trial_contours + 1,
        endpoint=False,
    )

    contours = []
    for t in theta0s:
        contours.append(
            float(phi_tot_fun_vec(jnp.array([t]), jnp.array([0.0]))[0])
        )  # contour values
    import time

    if initial_guess:
        assert len(initial_guess) == len(contours)
        contours = initial_guess
    contours = jnp.sort(jnp.asarray(contours))

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
    contours = jnp.sort(jnp.asarray(contours))

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
        fig = plot_3d(eq, "|B|", figsize=(12, 12))
    else:
        ax = None

    def find_XYZ_points(
        theta_pts, zeta_pts, surface, find_min_dist=None, ax=None, ls="-", label=None
    ):
        contour_X = []
        contour_Y = []
        contour_Z = []
        coil_coords = []

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
            coil_coords.append(jnp.vstack((coords[:, 0], coords[:, 1], coords[:, 2])).T)

        # Find the point of minimum separation
        if find_min_dist:
            minSeparation2 = 1.0e20
            for whichCoil1 in range(desirednumcoils):
                coords1 = coil_coords[whichCoil1]
                for whichCoil2 in range(whichCoil1):
                    coords2 = coil_coords[whichCoil2]

                    d = jnp.linalg.norm(
                        coords1[:, None, :] - coords2[None, :, :],
                        axis=-1,
                    )
                    # for whichPoint in range(len(contour_X[whichCoil1])):
                    this_minSeparation2 = jnp.min(d)
                    if this_minSeparation2 < minSeparation2:
                        minSeparation2 = this_minSeparation2

            print(f"Minimum coil-coil separation: {minSeparation2*1000:3.2f} mm")
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
                    jnp.mean(coil_currents) if equal_current else coil_currents[j]
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
    final_coilset = MixedCoilSet.from_makegrid_coilfile(coilsFilename)

    if save_figs:
        fig = plot_coils(final_coilset, fig=fig)
    ###################
    print(f"Coil current average is {jnp.mean(coil_currents):1.4e} A")
    print(f"Coil current variance is {jnp.var(coil_currents):1.4e} A")

    return final_coilset
