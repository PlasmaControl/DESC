"""Find helical coils from surface current potential."""
import os
import warnings

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import desc.io
from desc.backend import jnp
from desc.coils import MixedCoilSet
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid
from desc.plotting import plot_3d, plot_coils


def find_modular_coils(  # noqa: C901 - FIXME: simplify this
    surface_current_field,
    eqname,
    desirednumcoils=10,
    step=2,
    coilsFilename="coils.txt",
    dirname=".",
    save_figs=True,
    **kwargs,
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
    desirednumcoils : int, optional
        number of coils to discretize the surface current with, by default 10
    step : int, optional
        Amount of points to step when saving the coil geometry
        by default 2, meaning that every other point will be saved
        if higher, less points will be saved
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

    # TODO: make this go only to 1 FP
    nfp = eq.NFP
    theta_coil = jnp.linspace(0, 2 * jnp.pi, 128)
    zeta_coil = jnp.linspace(0, 2 * jnp.pi / nfp, 128)
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

    if not jnp.isclose(net_toroidal_current, 0):
        raise ValueError("this function only works for modular coils! Got nonzero I")

    ################################################################
    # find contours of constant phi
    ################################################################

    # TODO: only calc for one Field period, should be easy
    # with modular coils

    # TODO: don't need this for the whole zeta domain, just do 1 FP
    # TODO: if stell sym, only need 1/2 FP

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

    # make linspace contour
    contours = jnp.linspace(
        0, jnp.abs(net_poloidal_current), desirednumcoils + 1, endpoint=True
    ) * jnp.sign(net_poloidal_current)
    contours = jnp.sort(jnp.asarray(contours))

    coil_current = net_poloidal_current / desirednumcoils

    contour_theta, contour_zeta = find_full_coil_contours(contours)
    # TODO: make sure sign of current is correct
    sign_of_theta_contours = jnp.sign(contour_theta[0][-1] - contour_theta[0][0])

    ################################################################
    # Find the XYZ points in real space of the coil contours
    ################################################################

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
            trace = go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                marker=dict(size=0, opacity=0),
                line=dict(color=color, width=6, dash=ls),
                showlegend=False,
                name=label,
                hovertext=label,
            )
            fig.add_trace(trace)

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
        return contour_X, contour_Y, contour_Z, fig

    if save_figs:
        fig = plot_3d(eq, "|B|", figsize=(12, 12))
    else:
        fig = None

    contour_X, contour_Y, contour_Z, fig = find_XYZ_points(
        contour_theta,
        contour_zeta,
        winding_surf,
        find_min_dist=True,
        label="Final",
        color="black",
        fig=fig,
    )
    if fig and save_figs:
        fig.show()

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
                thisCurrent = net_poloidal_current / desirednumcoils
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
    print(
        f"Current per coil is {coil_current*current_sign:1.4e}"
        f" A for {desirednumcoils} coils"
    )

    return final_coilset
