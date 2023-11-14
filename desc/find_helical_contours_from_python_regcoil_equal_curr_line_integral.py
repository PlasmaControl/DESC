"""Find helical coils from surface current potential."""
import os
import warnings

import matplotlib.pyplot as plt

import desc.io
from desc.backend import jnp
from desc.coils import MixedCoilSet
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid
from desc.plotting import plot_3d, plot_coils

# TODO: move plot stuff from modular coils to here as well
# esp. the before/after optimization of the coils


def find_helical_coils(  # noqa: C901 - FIXME: simplify this
    surface_current_field,
    eqname,
    desirednumcoils=10,
    step=2,
    coilsFilename="coils.txt",
    dirname=".",
    save_figs=True,
    **kwargs,
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
    dirname : str, optional
        directory to save files to, by default "."
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

    ##### find helicity naively ######################################################

    if jnp.isclose(net_toroidal_current, 0):
        raise ValueError(
            "this function only works for helical coils, not modular or window pane!"
        )
    # we know that I = -(G - G_ext) / (helicity * NFP)

    helicity = -net_poloidal_current / net_toroidal_current / eq.NFP
    phi_slope = jnp.sign(helicity)

    theta_coil = jnp.linspace(0, 2 * jnp.pi, 128)
    zeta_coil = jnp.linspace(0, 2 * jnp.pi / nfp, round(128 * jnp.abs(helicity)))
    dz = zeta_coil[1] - zeta_coil[0]
    zetal_coil = zeta_coil

    theta_coil = theta_coil  # * phi_slope
    theta_coil = jnp.sort(theta_coil)

    ################################################################
    # find contours of constant phi
    ################################################################

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
    # TODO: change this so that  this fxn only accepts Ncoils length array
    def find_full_coil_contours(contours, show_plots=True, ax=None, label=None, ls="-"):
        """Accepts a list of current potential contour values of length Ncoils+1.

        returns the theta,zeta points of each contour

        """
        theta_full = theta_coil
        for inn in range(1, int(2 * nfp) + nfp * int(jnp.abs(helicity))):
            theta_full = jnp.concatenate(
                (
                    theta_full,
                    (theta_coil + 2 * jnp.pi * inn * jnp.sign(phi_slope)),
                )
            )
        theta_full = jnp.append(
            theta_full, theta_full[-1] + theta_full[1]
        )  # add the last point
        theta_full = jnp.sort(theta_full)
        print(theta_full)
        print(theta_coil)
        zeta_full = jnp.append(zetal_coil, 2 * jnp.pi / nfp)
        theta_full_2D, zeta_full_2D = jnp.meshgrid(theta_full, zeta_full, indexing="ij")
        my_tot_full = phi_tot_fun_vec(theta_full_2D, zeta_full_2D).reshape(
            theta_full.size, zeta_full.size, order="F"
        )
        # print(notavar)

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
            print(
                "residual when created",
                (contour_theta[j][-1] - contour_theta[j][0]) % (2 * jnp.pi),
            )

            numCoils += 1
            if show_plots:
                plt.plot(contour_zeta[-1], contour_theta[-1], "-r", linewidth=1)
                plt.plot(contour_zeta[-1][-1], contour_theta[-1][-1], "sk")

        plt.xlabel(r"$\zeta$")
        plt.ylabel(r"$\theta$")

        # before returning, right now these are only over 1 FP
        # should tile them s.t. they are full coils, by repeating them
        #  with a pi/NFP shift in zeta, I think
        print("first 10 theta", contour_theta[0][0:10])
        print("first 10 zeta", contour_zeta[0][0:10])
        print("last 10 theta", contour_theta[0][-10:])
        print("last 10 zeta", contour_zeta[0][-10:])
        for i_contour in range(len(contour_theta)):
            inds = jnp.argsort(contour_zeta[i_contour])
            orig_theta = contour_theta[i_contour][inds]
            orig_endpoint_theta = orig_theta[-1]

            print("residual", (orig_theta[-1] - orig_theta[0]) % (2 * jnp.pi))

            orig_theta = jnp.atleast_1d(orig_theta[:-1])  # dont need last point
            orig_zeta = contour_zeta[i_contour][inds]
            orig_zeta = jnp.atleast_1d(orig_zeta[:-1])  # dont need last point

            contour_theta[i_contour] = jnp.atleast_1d(orig_theta)
            contour_zeta[i_contour] = jnp.atleast_1d(orig_zeta)

            theta_shift = orig_endpoint_theta - orig_theta[0]

            zeta_shift = 2 * jnp.pi / nfp - orig_zeta[0]

            for i in range(1, nfp):
                contour_theta[i_contour] = jnp.concatenate(
                    [contour_theta[i_contour], orig_theta + theta_shift * i]
                )
                contour_zeta[i_contour] = jnp.concatenate(
                    [contour_zeta[i_contour], orig_zeta + zeta_shift * i]
                )
            contour_theta[i_contour] = jnp.append(
                contour_theta[i_contour],
                nfp * (orig_endpoint_theta - contour_theta[i_contour][0])
                + contour_theta[i_contour][0],
            )
            contour_zeta[i_contour] = jnp.append(contour_zeta[i_contour], 2 * jnp.pi)

        for j in range(N_trial_contours):
            # plt.plot(contour_zeta[j], contour_theta[j], "--b", linewidth=1)
            print("#" * 10 + f" {j} " + "#" * 10)
            print("first 10 theta", contour_theta[j][0:10])
            print("first 10 zeta", contour_zeta[j][0:10])
            print("last 10 theta", contour_theta[j][-10:])
            print("last 10 zeta", contour_zeta[j][-10:])
            print(
                "residual", (contour_theta[j][-1] - contour_theta[j][0]) % (2 * jnp.pi)
            )
        plt.xlim([0, 3 * jnp.pi / nfp])
        # plt.xlim([1.5, 1.7])
        plt.ylim([0, 10])
        # plt.ylim(
        #     [
        #         0,
        #         jnp.sign(phi_slope)*
        #         # * (2 + jnp.abs(helicity))
        #         * (2 * jnp.pi + 2 * jnp.pi / nfp / 4),
        #     ]
        # )
        # plt.ylim([-5, -7])
        return contour_theta, contour_zeta

    # make linspace contour
    contours = jnp.linspace(
        0, jnp.abs(net_toroidal_current), desirednumcoils + 1, endpoint=True
    )
    contours = jnp.sort(jnp.sign(phi_slope) * jnp.asarray(contours))

    contour_theta, contour_zeta = find_full_coil_contours(contours)

    ################################################################
    # Find the XYZ points in real space of the coil contours
    ################################################################

    def find_XYZ_points(
        theta_pts, zeta_pts, surface, find_min_dist=None, ls="-", label=None
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
        return contour_X, contour_Y, contour_Z

    if save_figs:
        fig = plot_3d(eq, "|B|", figsize=(12, 12))
    contour_X, contour_Y, contour_Z = find_XYZ_points(
        contour_theta,
        contour_zeta,
        winding_surf,
        find_min_dist=True,
        label="Final",
    )
    ################

    print("first 10 X", contour_X[0][0:10])
    print("first 10 Y", contour_Z[0][0:10])
    print("first 10 Z", contour_Y[0][0:10])
    print("last 10 X", contour_X[0][-10:])
    print("last 10 Y", contour_Z[0][-10:])
    print("last 10 Z", contour_Y[0][-10:])

    # Write coils file
    write_coil = True
    if write_coil:
        with open(coilsFilename, "w") as f:
            f.write("periods " + str(nfp) + "\n")
            f.write("begin filament\n")
            f.write("mirror NIL\n")

            for j in range(desirednumcoils):
                N = len(contour_X[j])
                # vector along direction of contour
                contour_vector = jnp.array(
                    [
                        contour_X[j][1] - contour_X[j][0],
                        contour_Y[j][1] - contour_Y[j][0],
                        contour_Z[j][1] - contour_Z[j][0],
                    ]
                )
                K = surface_current_field.compute(
                    "K",
                    grid=Grid(
                        jnp.array([[0, contour_theta[j][0], contour_zeta[j][0]]])
                    ),
                    basis="xyz",
                )["K"]
                current_sign = jnp.sign(jnp.dot(contour_vector, K[0, :]))
                thisCurrent = current_sign * net_toroidal_current / desirednumcoils
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
        fig.show()
    ###################
    print(f"Current per coil is {thisCurrent:1.4e}" f" A for {desirednumcoils} coils")

    return final_coilset
