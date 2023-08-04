import os
import sys

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import desc.io
from desc.backend import jnp
from desc.coils import CoilSet
from desc.compute.utils import cross, dot
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.transform import Transform

# make this a fxn that takes in the phi_MN and does the usual thing
# does theta convention matter for this?

# TODO: if the original file had TF coils external, this still works right? yes I think so, bc we subtract the
# external current from the net poloidal current before it is saved in regcoil, so this script will work even
# for cases where I ran regcoil with external coils

##################### INPUTS #####################
# regcoil_out .nc file
# number of helical coils
# which lambda to us from the regcoil file
# how many pts to output for the coil geometry (integer i, to use only every i'th point from the contour)
##################### OUTPUTS #####################
# coilfile .txt file, contains the helical coil geometry and the currents in them


def find_helical_coils(
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
):
    eq = desc.io.load(eqname)
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
            Z_lmn=np.array([-a_ves]),
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

    def phi_tot_fun_theta_deriv_vec_jit(sgrid, trans_temp):

        return (
            trans_temp.transform(phi_mn_desc_basis, dt=1)
            + net_toroidal_current / 2 / jnp.pi
        )

    def phi_tot_fun_zeta_deriv_vec(theta, zeta):
        nodes = np.vstack(
            (
                np.zeros_like(theta.flatten(order="F")),
                theta.flatten(order="F"),
                zeta.flatten(order="F"),
            )
        ).T
        trans_temp = Transform(
            Grid(nodes, sort=False), basis, derivs=np.array([[0, 0, 1]])
        )
        return (
            np.asarray(trans_temp.transform(phi_mn_desc_basis, dz=1))
            + net_poloidal_current / 2 / np.pi
        )

    def surf_current_vec_contravariant_zeta_times_R_times_g_tt(sgrid):

        Rs = winding_surf.compute_coordinates(grid=sgrid, basis="rpz")[:, 0]
        rs_t = winding_surf.compute_coordinates(grid=sgrid, dt=1)
        rs_z = winding_surf.compute_coordinates(grid=sgrid, dz=1)
        ns_mag = np.linalg.norm(cross(rs_t, rs_z), axis=1)

        phi_t = phi_tot_fun_theta_deriv_vec(sgrid.nodes[:, 0], sgrid.nodes[:, 1])
        g_tt = dot(rs_t, rs_t)

        # "vector" is the K vector in terms of grad(theta) and grad(zeta)
        K_sup_zeta = -phi_t * (1 / ns_mag)

        return K_sup_zeta * Rs * jnp.sqrt(g_tt)

    def surf_g_tt(theta, zeta):
        # the theta, zeta here are in REGCOIL
        # the theta used for DESC to evaluate from the surface must be -theta
        theta_DESC_surf = theta
        nodes_surf = np.vstack(
            (
                np.zeros_like(theta_DESC_surf.flatten(order="F")),
                theta_DESC_surf.flatten(order="F"),
                zeta.flatten(order="F"),
            )
        ).T
        sgrid = Grid(nodes_surf, sort=False)
        rs_t = winding_surf.compute_coordinates(grid=sgrid, dt=1)
        g_tt = dot(rs_t, rs_t)

        return np.sqrt(g_tt)

    def phi_tot_fun(theta, zeta):
        if np.shape(theta):
            theta = theta[0]
        trans_temp = Transform(Grid(np.array([[1, theta, zeta]])), basis)
        return (
            np.asarray(trans_temp.transform(phi_mn_desc_basis))
            + net_poloidal_current * zeta / 2 / np.pi
            + net_toroidal_current * theta / 2 / np.pi
        )

    # this tiling only works if we have our contours increasing to the up and right
    # otherwise, we do not capture full contouts
    # do trial run with this and see slope, if slope positive keep, if slope negative then make theta decrease

    ##### find helicity naively ######################################################
    t_2D, z_2D = np.meshgrid(theta_coil, zeta_coil, indexing="ij")

    total_phi_trial = phi_tot_fun_vec(t_2D, z_2D).reshape(
        theta_coil.size, zeta_coil.size, order="F"
    )
    plt.figure(figsize=(18, 10))
    cdata = plt.contour(t_2D.T, z_2D.T, np.transpose(total_phi_trial))
    numCoilsFound = len(cdata.collections)
    contour_zeta = []
    contour_theta = []

    for j in range(numCoilsFound):
        try:
            p = cdata.collections[j].get_paths()[0]
        except:
            print("no path found for given contour")
            continue
        v = p.vertices
        temp_zeta = v[:, 0]
        if len(temp_zeta) < 5:
            continue
        temp_theta = v[:, 1]
        phi_slope = (temp_theta[-1] - temp_theta[0]) / (temp_zeta[-1] - temp_zeta[0])
        # print("sign of slope", np.sign(phi_slope))
        # if slope positive, we can tile as theta=(0 -> NFP*2pi),if negative we tile from (-2*NFP*2pi -> 0)
        break
    ##################################################################

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

    def get_integration_points_and_line_elems(
        contour_theta_halfway,
        contours_were_sorted,
        nthetas=50,
    ):
        # simply gets the theta pts btwn the halfway contours for each coil contour
        # and the dtheta for each given the ntheta
        N_trial_contours = len(contour_theta_halfway)
        coil_nodes = []  # will be a list of lists
        coil_dthetas = []  # will be a list of lists of dphi*dtheta for each point
        for i in range(0, N_trial_contours):
            phis = np.zeros(
                (nthetas,)
            )  # assumes we have contours which are intially at phi=0
            if i != 0:  # do as normal
                thetas = np.linspace(
                    contour_theta_halfway[i - 1][0],
                    contour_theta_halfway[i][0],
                    nthetas,
                    endpoint=False,
                )
            elif (
                not contours_were_sorted
            ):  # if i==0, then must use the last and the first halfway contour periodicity to find the area
                thetas = jnp.linspace(
                    contour_theta_halfway[i][0],
                    contour_theta_halfway[-1][j] + 2 * np.pi * np.sign(phi_slope),
                    nthetas,
                    endpoint=False,
                )
            elif (
                contours_were_sorted
            ):  # if i==0, then must use the last and the first halfway contour periodicity to find the area
                # here I assume this contour is on the bottom... when it could be on top too
                thetas = np.linspace(
                    contour_theta_halfway[i - 1][0],
                    contour_theta_halfway[i][0],
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

    def find_contours_and_current_variance(
        contours, return_full_info=False, show_plots=False, nthetas=200
    ):
        """Accepts a list of current potential contour values of length Ncoils+1,
        returns variance of current in the coils

        """
        N_trial_contours = len(contours) - 1
        # contours is the decision variable
        contour_zeta = []
        contour_theta = []
        if not contours[1] - contours[0] > 1:
            contours = np.sort(contours)
            contours_were_sorted = True
        else:
            contours_were_sorted = False
        contours_halfway = []
        for i in range(N_trial_contours):
            halfway_below = (contours[i] + contours[i + 1]) / 2
            contours_halfway.append(halfway_below)

        plt.figure(figsize=(18, 10))
        cdata = plt.contour(
            zeta_full_2D.T, theta_full_2D.T, np.transpose(my_tot_full), contours
        )
        cdata_half = plt.contour(
            zeta_full_2D.T,
            theta_full_2D.T,
            np.transpose(my_tot_full),
            contours_halfway,
            colors="w",
        )

        contour_zeta = []
        contour_theta = []
        contour_zeta_halfway = []
        contour_theta_halfway = []
        numCoils = 0
        for j in range(N_trial_contours):
            try:
                p = cdata.collections[j].get_paths()[0]
            except:
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
            if numCoils == N_trial_contours:
                plt.plot(
                    contour_zeta[0][50:],
                    np.asarray(contour_theta[0][50:]) - 2 * np.pi,
                    "--k",
                )
                break
        for k in range(len(cdata_half.collections)):
            try:
                p = cdata_half.collections[k].get_paths()[0]
            except:
                print("no path found for given contour")
                continue
            v = p.vertices
            contour_zeta_halfway.append(v[:, 0])
            contour_theta_halfway.append(v[:, 1])

            numCoils += 1
            if show_plots:
                plt.plot(
                    contour_zeta_halfway[-1],
                    contour_theta_halfway[-1],
                    "--g",
                    linewidth=1,
                )
                plt.plot(
                    contour_zeta_halfway[-1][-1], contour_theta_halfway[-1][-1], "--k"
                )
        plt.close()
        del cdata
        del cdata_half
        ################################################################
        # Find contours that are halfway between each coil contour,
        # to set the bounds of integration of surface current for
        # each coil
        ################################################################

        # number of theta points at each phi point for integration

        coil_nodes_line, coil_line_elements = get_integration_points_and_line_elems(
            contour_theta_halfway,
            contour_zeta_halfway,
            contours_were_sorted,
            node_skip=2,
            nthetas=nthetas,
        )

        if show_plots:
            plt.figure(figsize=(10, 10))

            for i in range(N_trial_contours):  # np.flip(np.arange(desirednumcoils)):
                plt.scatter(
                    coil_nodes_line[i][1],
                    coil_nodes_line[i][0],
                    label="line integration nodes",
                    s=50,
                    marker="x",
                )
                plt.plot(
                    contour_zeta[i],
                    contour_theta[i],
                    "c",
                    label="Coil to integrate current for",
                )
                if i != 0:
                    plt.plot(
                        contour_zeta_halfway[i], contour_theta_halfway[i], "k"
                    )  # , label="bounding contours for integration")
                    plt.plot(
                        contour_zeta_halfway[i - 1], contour_theta_halfway[i - 1], "k"
                    )

                elif not contours_were_sorted and i == 0:
                    plt.plot(
                        contour_zeta_halfway[-1],
                        contour_theta_halfway[-1] + 2 * np.pi * np.sign(phi_slope),
                        "k-",
                        label="bounding contours for integration",
                    )
                    plt.plot(contour_zeta_halfway[i], contour_theta_halfway[i], "k-")
                    plt.legend()
                elif contours_were_sorted and i == 0:
                    plt.plot(
                        contour_zeta_halfway[0],
                        contour_theta_halfway[0] + 2 * np.pi * np.sign(phi_slope),
                        "m--",
                        label="bounding contours for integration",
                    )
                    plt.plot(contour_zeta_halfway[i], contour_theta_halfway[i], "k--")
                    plt.legend()
            plt.xlabel("zeta")
            plt.ylabel("theta")

        ################################################################
        # integrate surface current over the areas for each coil
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

        numCoils = len(contour_theta)

        variance = np.var(coil_currents_line)
        if not show_plots:
            plt.close("all")

        ####### debug #######################
        # # test that the length is what we expect
        true_length = 2 * np.pi * a_ves
        approx_length = 0
        len_elem_sum = 0
        for coords, line_elems in zip(coil_nodes_line, coil_line_elements):
            gtt_sqrt = np.abs(surf_g_tt(coords[0], coords[1]))
            approx_length += np.sum(gtt_sqrt * line_elems)
            len_elem_sum += np.sum(line_elems)
        length_diff = jnp.abs(true_length - approx_length)
        print(f" Difference in length approx and true length: {length_diff}")
        area_elem_diff = np.abs(2 * np.pi - len_elem_sum)
        if area_elem_diff > 0.1:
            print(f"line elem sum diff is large! {area_elem_diff:1.4e}")

        if not return_full_info:
            return variance
        else:  # return all info: contour_theta and contour_zeta, coil_currents, etc
            return contour_theta, contour_zeta, coil_currents_line

    def find_full_coil_contours(contours, show_plots=True):
        """Accepts a list of current potential contour values of length Ncoils+1,
        returns variance of current in the coils

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
            except:
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
        return contour_theta, contour_zeta

    N_trial_contours = desirednumcoils
    # flip is so the contour levels are increasing (this may not be necessary dep. on contour direction)
    # adding extra contour above and below the ones we care about, so we can then find halfway point...

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

    xs = [contours]
    fun_vals = [find_contours_and_current_variance(contours)]

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

    contour_theta, contour_zeta, coil_currents = find_contours_and_current_variance(
        contours, return_full_info=True, show_plots=True
    )
    contour_theta, contour_zeta = find_full_coil_contours(contours)

    ################################################################
    # Find the XYZ points in real space of the coil contours
    ################################################################
    contour_X = []
    contour_Y = []
    contour_Z = []

    for thetas, zetas in zip(contour_theta, contour_zeta):
        coords = winding_surf.compute_coordinates(
            grid=Grid(
                jnp.vstack((jnp.zeros_like(thetas), thetas, zetas)).T, sort=False
            ),
            basis="xyz",
        )
        contour_X.append(coords[:, 0])
        contour_Y.append(coords[:, 1])
        contour_Z.append(coords[:, 2])
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    for j in range(len(contour_X)):
        ax.plot(contour_X[j], contour_Y[j], contour_Z[j], "-")

    # Find the point of minimum separation
    minSeparation2 = 1.0e20
    for whichCoil1 in range(desirednumcoils):
        for whichCoil2 in range(whichCoil1):
            for whichPoint in range(len(contour_X[whichCoil1])):
                dx = contour_X[whichCoil1][whichPoint] - contour_X[whichCoil2]
                dy = contour_Y[whichCoil1][whichPoint] - contour_Y[whichCoil2]
                dz = contour_Z[whichCoil1][whichPoint] - contour_Z[whichCoil2]
                separation2 = dx * dx + dy * dy + dz * dz
                this_minSeparation2 = np.min(separation2)
                if this_minSeparation2 < minSeparation2:
                    minSeparation2 = this_minSeparation2

    print(f"Minimum coil-coil separation: {np.sqrt(minSeparation2)*1000:3.2f} mm")

    figfilename = f"coil_3d_ncoil_{desirednumcoils}_alpha_{alpha:1.4e}_{dirname}.png"
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
