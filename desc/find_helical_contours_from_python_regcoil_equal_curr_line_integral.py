"""Find helical coils from surface current potential."""

import matplotlib.pyplot as plt

from desc.backend import jnp
from desc.coils import MixedCoilSet, SplineXYZCoil
from desc.compute.utils import safediv
from desc.grid import Grid

# TODO: move plot stuff from modular coils to here as well
# esp. the before/after optimization of the coils


def cut_surface_current_into_coils(  # noqa: C901 - FIXME: simplify this
    surface_current_field,
    desirednumcoils=10,  # TODO: make this coils_per_NFP for modular...
    step=2,
    spline_method="cubic",
    show_plots=False,
):
    """Find helical or modular coils from a surface current potential.

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField or CurrentPotentialField
        CurrentPotentialField or FourierCurrentPotentialField object to
        discretize into coils.
    desirednumcoils : int, optional
        number of coils to discretize the surface current with, by default 10
    step : int, optional
        Amount of points to skip by when saving the coil geometry spline
        by default 2, meaning that every other point will be saved
        if higher, less points will be saved e.g. 3 saves every 3rd point
    spline_method : str, optional
        method of fitting to use for the spline, by default ``"cubic"``
        see ``SplineXYZCoil`` for more info
    show_plots : bool, optional,
        whether to show plots of the contours chosen for coils, by default False

    Returns
    -------
    coils : CoilSet
        DESC CoilSet object that is a discretization of the input
        surface current on the given winding surface
    """
    nfp = surface_current_field.Phi_basis.NFP

    net_toroidal_current = surface_current_field.I
    net_poloidal_current = surface_current_field.G
    assert not jnp.isclose(net_toroidal_current, 0) or not jnp.isclose(
        net_poloidal_current, 0
    ), (
        "Detected both net toroidal and poloidal current are both zero, "
        "this function cannot find windowpane coils"
    )

    winding_surf = surface_current_field

    ################################################################
    # find current helicity
    ################################################################
    # we know that I = -(G - G_ext) / (helicity * NFP)
    # if net_toroidal_current is zero, then we have modular coils,
    # and just make helicity zero
    helicity = safediv(
        -net_poloidal_current, net_toroidal_current * nfp, threshold=1e-8
    )
    phi_slope = jnp.sign(helicity)

    theta_coil = jnp.linspace(0, 2 * jnp.pi, 128)
    if not jnp.isclose(helicity, 0):
        # helical coils, we need finer toroidal resolution
        zeta_coil = jnp.linspace(0, 2 * jnp.pi / nfp, round(128 * jnp.abs(helicity)))
        zetal_coil = zeta_coil
    else:
        # modular coils
        theta_coil = jnp.linspace(0, 2 * jnp.pi, 128)
        zeta_coil = jnp.linspace(0, 2 * jnp.pi / nfp, 128)
        dz = zeta_coil[1] - zeta_coil[0]
        zetal_coil = jnp.arange(-jnp.pi / nfp, (2 + 1 / nfp) * jnp.pi, dz)

    ################################################################
    # find contours of constant phi
    ################################################################
    # make linspace contours
    if not jnp.isclose(helicity, 0):
        # helical coils
        # we start them on zeta=0 plane, so we will find contours
        # going up from 0 to I (corresponding to zeta=0, and theta increasing)
        contours = jnp.linspace(
            0, jnp.abs(net_toroidal_current), desirednumcoils + 1, endpoint=True
        )
        contours = jnp.sort(jnp.sign(phi_slope) * jnp.asarray(contours))
        coil_current = jnp.abs(net_toroidal_current) / desirednumcoils

    else:
        # modular coils
        # go from zero to G
        contours = jnp.linspace(
            0, jnp.abs(net_poloidal_current), desirednumcoils + 1, endpoint=True
        ) * jnp.sign(net_poloidal_current)
        contours = jnp.sort(jnp.asarray(contours))
        coil_current = net_poloidal_current / desirednumcoils

    # TODO: change this so that  this we only need Ncoils length array

    theta_full = theta_coil
    if not jnp.isclose(helicity, 0):
        # if helical coils, need theta in larger range than 0 to 2pi
        # to completely capture contours that start at zeta=0
        # a contour at (theta,zeta)=(0,0) will end after one NFP
        # at (theta,zeta)=(2pi*helicity,2pi/NFP)
        # TODO: why do I have more than that here?
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
    else:
        theta_full = jnp.append(theta_full, 2 * jnp.pi)
    if not jnp.isclose(helicity, 0):
        zeta_full = jnp.append(zetal_coil, 2 * jnp.pi / nfp)
    else:
        # TODO: make this also go to only 2pi/NFP
        zeta_full = jnp.append(zetal_coil, 2 * jnp.pi)
    theta_full_2D, zeta_full_2D = jnp.meshgrid(theta_full, zeta_full, indexing="ij")
    grid = Grid(
        jnp.vstack(
            (
                jnp.zeros_like(theta_full_2D.flatten(order="F")),
                theta_full_2D.flatten(order="F"),
                zeta_full_2D.flatten(order="F"),
            )
        ).T,
        sort=False,
    )
    phi_total_full = surface_current_field.compute("Phi", grid=grid)["Phi"].reshape(
        theta_full.size, zeta_full.size, order="F"
    )

    N_trial_contours = len(contours) - 1
    contour_zeta = []
    contour_theta = []
    plt.figure(figsize=(18, 10))
    cdata = plt.contour(
        zeta_full_2D.T, theta_full_2D.T, jnp.transpose(phi_total_full), contours
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

    plt.xlabel(r"$\zeta$")
    plt.ylabel(r"$\theta$")

    if not jnp.isclose(helicity, 0):
        # before returning, right now these are only over 1 FP
        # so must tile them s.t. they are full coils, by repeating them
        #  with a pi/NFP shift in zeta

        for i_contour in range(len(contour_theta)):
            inds = jnp.argsort(contour_zeta[i_contour])
            orig_theta = contour_theta[i_contour][inds]
            orig_endpoint_theta = orig_theta[-1]

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

        plt.xlim([0, 3 * jnp.pi / nfp])
        plt.ylim([0, 10])
    else:
        # TODO: this should be able to easily be used to
        # find only N contours in one FP then rotate them
        # to get the full coilset.
        pass

    # for modular coils, easiest way to check contour direction is to see
    # direction of the contour thetas
    sign_of_theta_contours = jnp.sign(contour_theta[0][-1] - contour_theta[0][0])

    ################################################################
    # Find the XYZ points in real space of the coil contours
    ################################################################
    def find_XYZ_points(
        theta_pts,
        zeta_pts,
        surface,
        find_min_dist=None,
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

    contour_X, contour_Y, contour_Z = find_XYZ_points(
        contour_theta,
        contour_zeta,
        winding_surf,
        find_min_dist=True,
    )
    ################################################################
    # Create CoilSet object
    ################################################################
    coils = []
    for j in range(len(contour_X)):
        if not jnp.isclose(helicity, 0):
            # helical coils
            # make sure that the sign of the coil current is correct
            # by dotting K with the vector along the contour
            contour_vector = jnp.array(
                [
                    contour_X[j][1] - contour_X[j][0],
                    contour_Y[j][1] - contour_Y[j][0],
                    contour_Z[j][1] - contour_Z[j][0],
                ]
            )
            K = surface_current_field.compute(
                "K",
                grid=Grid(jnp.array([[0, contour_theta[j][0], contour_zeta[j][0]]])),
                basis="xyz",
            )["K"]
            current_sign = jnp.sign(jnp.dot(contour_vector, K[0, :]))
            thisCurrent = current_sign * coil_current
        else:
            # modular coils
            # make sure that the sign of the coil current is correct
            # don't need to dot with K here because we know the direction
            # based off the direction of the theta contour and sign of G
            # (extra negative sign because a positive G -> negative toroidal B
            # but we always have a right-handed coord system, and so current flowing
            # in positive poloidal direction creates a positive toroidal B)
            current_sign = -sign_of_theta_contours * jnp.sign(net_poloidal_current)
            thisCurrent = jnp.abs(coil_current) * current_sign
        coils.append(
            SplineXYZCoil(
                thisCurrent,
                jnp.append(contour_X[j][0::step], contour_X[j][0]),
                jnp.append(contour_Y[j][0::step], contour_Y[j][0]),
                jnp.append(contour_Z[j][0::step], contour_Z[j][0]),
                method=spline_method,
            )
        )

    final_coilset = MixedCoilSet(*coils)

    return final_coilset
