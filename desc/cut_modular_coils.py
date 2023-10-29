"""Find helical coils from surface current potential."""
import os
import warnings

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize

import desc.io
from desc.backend import jnp
from desc.coils import MixedCoilSet
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid
from desc.plotting import plot_3d, plot_coils


def find_modular_coils(  # noqa: C901 - FIXME: simplify this
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
    verbose=0,
):
    """Find modular coils from a surface current potential.

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
    # add a buffer around zeta=0 and zeta=2pi to catch contours that may go past
    # those lines
    zetal_coil = jnp.arange(-jnp.pi / nfp, (2 + 1 / nfp) * jnp.pi, dz)
    net_toroidal_current = surface_current_field.I
    net_poloidal_current = surface_current_field.G

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    ##################################################################################
    ################# Make winding surface DESC object #######################
    ##################################################################################

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

    if not jnp.isclose(net_toroidal_current, 0):
        raise ValueError("this function only works for modular coils! Got nonzero I")

    ################################################################
    # find contours of constant phi
    ################################################################

    # TODO: also precompute and interpolate Phi, so no need
    # to create grid objects inside the variance function
    # instead just need to interpolate Phi(theta)

    # TODO: make a jittable compute_Phi
    # though since the theta points we eval at are changing
    # each time, need to bypass the Grid/Transform
    # and directly implement the (fast)fourier transform?

    # TODO: only calc for one Field period, should be easy
    # with modular coils

    def get_currents(zetas_halfway, return_variance=True):
        # TODO: if G is negative... then larger zeta is more negative potential right?
        # so that matters...
        # let zetas_halfway be starting at the leftmost contour
        # and G_i = Phi(zetas[i+1])-Phi(zetas[i])
        # with i going from 0 to desiredNumCoils
        # we are missing the Phi value at the contour halfway above our last contour
        # but that is just the Phi[0] + G
        # since it is Phi(zeta+2pi) and Phi_sv is periodic and
        #  2nd term in zeta is linear in zeta i.e. is G * zeta / 2pi

        # let zetas_halfway be starting at the leftmost contour
        # and G_i = Phi(zetas[i+1])-Phi(zetas[i])
        # with i going from 0 to desiredNumCoils
        # TODO: check sign (should it maybe be negative?)
        # sign should be entirely determined by G and the contours
        # if G is pos, grad(Phi) pts right, K points down
        # and so if sign(contour_theta[-1] - contour_theta[0]) < 0
        # then current is pos
        # else if > 0, current is neg
        # so current sign should be =-sign(G)(sign(contour_theta[-1]-contour_theta[0]))
        # and current = sign * abs(current) (to be robust against sorting in zeta?)
        # CHECK THIS
        Phis = surface_current_field.compute(
            "Phi",
            grid=Grid(
                jnp.vstack(
                    (
                        jnp.zeros_like(zetas_halfway),
                        jnp.zeros_like(zetas_halfway),
                        zetas_halfway,
                    )
                ).T,
                jitable=True,
                sort=False,
            ),
        )["Phi"]
        Phis = jnp.append(Phis, Phis[0] + net_poloidal_current)
        currents = Phis[1:] - Phis[:-1]
        if return_variance:
            return jnp.var(currents)
        return currents

    # TODO: implement the analytic deriv of above

    # TODO: don't need this for the whole zeta domain, just do 1 FP

    # TODO: change this so that  this fxn only accepts Ncoils length array
    def find_full_coil_contours(contours, show_plots=True, ax=None, label=None, ls="-"):
        """Accepts a list of current potential contour values of length Ncoils+1.

        returns the theta,zeta points of each contour

        """
        theta_full = theta_coil
        theta_full = jnp.append(theta_full, 2 * jnp.pi)  # add the last point

        # TODO: this shold go to just 2pi/NFP

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
            if not jnp.isclose(jnp.abs(temp_theta[-1] - temp_theta[0]), 2 * jnp.pi):
                print(f"contour {j} is not closed in theta!")
                print(temp_theta[-1] - temp_theta[0])
            contour_zeta.append(temp_zeta)
            contour_theta.append(temp_theta)

            numCoils += 1
            if show_plots:
                plt.plot(contour_zeta[-1], contour_theta[-1], "-r", linewidth=1)
                plt.plot(contour_zeta[-1][-1], contour_theta[-1][-1], "sk")
        plt.xlim([-jnp.pi / nfp, 2 * jnp.pi / nfp])
        plt.ylim([0, 2 * jnp.pi])

        return contour_theta, contour_zeta

    # TODO: this shold go to just 2pi/NFP
    zeta0s = jnp.linspace(
        0,
        (2 * jnp.pi),
        desirednumcoils,
        endpoint=False,
    )

    # theta positions halfway between contour start points
    zeta0s_halfway = zeta0s - zeta0s[1] / 2

    import time

    contours = phi_tot_fun_vec(jnp.zeros_like(zeta0s), zeta0s)

    if initial_guess:
        assert len(initial_guess) == len(contours)
        contours = initial_guess
    contours = jnp.sort(jnp.asarray(contours))

    # TODO: this call to find full contours results in things of length(Ncoils-1)?
    contour_theta_initial, contour_zeta_initial = find_full_coil_contours(contours)

    if equal_current:
        xs = [zeta0s_halfway]
        fun_vals = [get_currents(zeta0s_halfway)]
        t_start = time.time()

        def callback(x):
            curr_val = get_currents(x)
            if verbose > 1:
                print(f"Iteration: {len(xs)} Time elapsed = {time.time()-t_start} s")
                print(f"Current function value: {curr_val:1.3e}")
                print(f"Current x: {x}")

            xs.append(x)
            fun_vals.append(curr_val)
            return False

        result = minimize(
            get_currents,
            zeta0s_halfway,
            options={"maxiter": maxiter, "disp": False},
            callback=callback,
            method=method,
        )
        t_end = time.time()
        print(f"Optimization for coils took {t_end-t_start} s")
        zetas_halfway = result.x
    else:
        zetas_halfway = zeta0s_halfway
    # TODO: remove the need to add another point at end by
    # modifying find full contours fxn
    coil_currents = get_currents(zetas_halfway, return_variance=False)
    # TODO: this shold go to just 2pi/NFP

    zetas_halfway = jnp.append(zetas_halfway, zetas_halfway[0] + 2 * jnp.pi)
    final_coil_zeta0s = (zetas_halfway[0:-1] + zetas_halfway[1:]) / 2
    final_coil_zeta0s = jnp.append(
        final_coil_zeta0s, final_coil_zeta0s[-1] + final_coil_zeta0s[1]
    )

    zeta0s = jnp.append(zeta0s, zeta0s[-1] + zeta0s[1])
    contours = phi_tot_fun_vec(jnp.zeros_like(final_coil_zeta0s), final_coil_zeta0s)
    contours = jnp.sort(jnp.asarray(contours))

    print("initial coil zetas:", zeta0s)
    print("final coil zetas:", final_coil_zeta0s)

    contour_theta, contour_zeta = find_full_coil_contours(contours)
    sign_of_theta_contours = jnp.sign(contour_theta[0][-1] - contour_theta[0][0])

    ################################################################
    # Find the XYZ points in real space of the coil contours
    ################################################################
    if save_figs:
        fig = plot_3d(eq, "|B|", figsize=(12, 12))
    else:
        fig = None

    def find_XYZ_points(
        theta_pts,
        zeta_pts,
        surface,
        find_min_dist=None,
        fig=None,
        ls="solid",
        label=None,
        color=None,
    ):
        if fig is None:
            fig = go.Figure()
        contour_X = []
        contour_Y = []
        contour_Z = []
        coil_coords = []

        for k, (thetas, zetas) in enumerate(zip(theta_pts, zeta_pts)):
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
            d_endpoint = jnp.linalg.norm(coords[0, :] - coords[-1, :])
            if d_endpoint > 0.05:  # arbitrary threshold length
                warnings.warn(
                    f"Coil {k} might not be closed! {d_endpoint}"
                    f"m btwn start point and endpoint",
                    RuntimeWarning,
                )

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
                trace = go.Scatter3d(
                    x=coords1[:, 0],
                    y=coords1[:, 1],
                    z=coords1[:, 2],
                    marker=dict(
                        size=8,
                    ),
                    line=dict(
                        color=color,
                        width=8,
                        dash=ls,
                    ),
                    showlegend=False,
                    name=label,
                    hovertext=label,
                )
                fig.add_trace(trace)

            print(f"Minimum coil-coil separation: {minSeparation2*1000:3.2f} mm")
        return contour_X, contour_Y, contour_Z, fig

    plt.figure()
    plt.plot(contour_zeta[0], contour_theta[0])
    plt.scatter(contour_zeta[0][0], contour_theta[0][0], c="r", label="start")
    plt.scatter(contour_zeta[0][-1], contour_theta[0][-1], c="b", label="end")
    plt.legend()

    plt.xlabel("zeta")
    plt.ylabel("theta")
    plt.title("first coil theta,zeta curve")
    index = 1
    plt.figure()
    plt.plot(contour_zeta[index], contour_theta[index])
    plt.scatter(contour_zeta[index][0], contour_theta[index][0], c="r", label="start")
    plt.scatter(contour_zeta[index][-1], contour_theta[index][-1], c="b", label="end")
    plt.legend()

    plt.xlabel("zeta")
    plt.ylabel("theta")
    plt.title("2nd coil theta,zeta curve")

    contour_X, contour_Y, contour_Z, fig = find_XYZ_points(
        contour_theta,
        contour_zeta,
        winding_surf,
        find_min_dist=True,
        label="Final",
        color="black",
        fig=fig,
    )
    _, _, _, fig = find_XYZ_points(
        contour_theta_initial,
        contour_zeta_initial,
        winding_surf,
        find_min_dist=False,
        label="Initial",
        color="tomato",
        ls="dash",
        fig=fig,
    )
    if fig and save_figs:
        fig.show()
    fig2 = plot_3d(eq, "|B|", figsize=(12, 12))
    _, _, _, fig2 = find_XYZ_points(
        contour_theta_initial,
        contour_zeta_initial,
        winding_surf,
        find_min_dist=False,
        label="Initial",
        color="tomato",
        ls="dashdot",
        fig=fig2,
    )
    fig2.show()
    # TODO: update deps to include kaleido so can save 3D figs
    # needs kaleido:fig.write_image("3D_coils" + coilsFilename.strip(".txt") + ".png")

    ################

    # TODO: when alg is changed to only go over one NFP
    # make this part rotate the coils found
    # using Coilset.from_symmetry(NFP=eq.NFP)
    # and passing in the coils found

    # Write coils file
    write_coil = True
    if write_coil:
        with open(coilsFilename, "w") as f:
            f.write("periods " + str(nfp) + "\n")
            f.write("begin filament\n")
            f.write("mirror NIL\n")

            for j in range(desirednumcoils):
                N = len(contour_X[j])
                # TODO: here is where we mult by the sign
                current_sign = -sign_of_theta_contours * jnp.sign(net_poloidal_current)
                thisCurrent = (
                    net_poloidal_current / desirednumcoils
                    if equal_current
                    else coil_currents[j]
                )
                thisCurrent = jnp.abs(thisCurrent) * current_sign
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
    if equal_current:
        print(
            "Coil currents set to be equal to total net pol.current / coils:",
            f" {net_poloidal_current / desirednumcoils:1.4e}",
        )
        avg_abs_diff = jnp.mean(
            jnp.abs(net_poloidal_current / desirednumcoils - jnp.asarray(coil_currents))
        )
        print(
            "Avg difference between optimized coil currents and the set current:"
            f"{avg_abs_diff:1.4e} A"
        )

    print("sum of coil currents", jnp.sum(coil_currents))
    print("should equal net G :", net_poloidal_current)

    return final_coilset
