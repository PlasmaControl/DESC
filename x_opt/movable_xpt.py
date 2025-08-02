import os
import sys
import contextlib

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

from desc import set_device

set_device("gpu")

from desc.backend import print_backend_info

import jax.numpy as jnp
import numpy as np

from desc.io import load
from desc.grid import LinearGrid, Grid
from desc.coils import CoilSet, FourierPlanarCoil, FourierRZCoil
from desc.integrals import compute_B_plasma
from desc.magnetic_fields import SumMagneticField
from desc.objectives import (
    QuadraticFlux,
    CoilLength,
    FixCoilCurrent,
    CoilSetMinDistance,
    PlasmaCoilSetDistanceBound,
    FixParameters,
    ObjectiveFunction,
    MinCoilSetPointDistance,
    Bxdl,
    XPointDistanceBound,
)

from desc.optimize import Optimizer
from desc.compute import rpz2xyz, xyz2rpz
from desc.compute.geom_utils import reflection_matrix
import networkx as nx
from networkx.algorithms import approximation as approx
from desc.coils import FourierPlanarFiniteBuildCoil, CoilSet, FourierXYFiniteBuildCoil
from desc.objectives import (
    ObjectiveFunction,
    CoilSetMaxB,
)

rad = 0.6
offset = 0.3
max_current = 6e6


def compute_intersections(centers, clearance):
    centers = centers[
        :, jnp.newaxis, :
    ]  # add dummy dimension for vectorized operations
    sym = True
    NFP = 2

    # if stellarator symmetric, add reflected coils from the other half field period
    if sym:
        normal = jnp.array([-jnp.sin(jnp.pi / NFP), jnp.cos(jnp.pi / NFP), 0])
        xyz_sym = centers @ reflection_matrix(normal).T @ reflection_matrix([0, 0, 1]).T
        centers = jnp.vstack((centers, xyz_sym))

    # field period rotation is easiest in [R,phi,Z] coordinates
    rpz = xyz2rpz(centers)

    # if field period symmetry, add rotated coils from other field periods
    if NFP > 1:
        rpz0 = rpz
        for k in range(1, NFP):
            rpz = jnp.vstack((rpz, rpz0 + jnp.array([0, 2 * jnp.pi * k / NFP, 0])))

    # ensure phi in [0, 2pi)
    rpz = rpz.at[:, :, 1].set(jnp.mod(rpz[:, :, 1], 2 * jnp.pi))

    x = rpz2xyz(rpz)

    x = x[:, 0, :]  # just return one center for each coil

    n_sym_coils = x.shape[0]
    num_unique_coils = n_sym_coils // 4  # 4 copies per coil

    distances = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=-1)

    np.fill_diagonal(distances, 10 * clearance)  # Ensure self-intersection not flagged

    intersections_symmetric = (distances < clearance).astype(int)
    intersections_symmetric = np.array(intersections_symmetric)

    intersections_reshaped_orig = intersections_symmetric.reshape(
        4, num_unique_coils, 4, num_unique_coils
    )
    # Max over copy indices (axes 0 and 2)
    final_intersections = intersections_reshaped_orig.max(axis=(0, 2))

    self_clash_indices = np.where(np.diag(final_intersections) == 1)[0]

    if self_clash_indices.size > 0:
        final_intersections[self_clash_indices, :] = 1
        final_intersections[:, self_clash_indices] = 1

    return final_intersections


def add_coils(
    candidate_coil_centers,
    candidate_coil_normals,
    candidate_coil_currents,
    min_separation,
    shaping_coils,
):

    coil_centers = candidate_coil_centers

    # filter out coil centers too close to existing centers
    if shaping_coils is not None:
        existing_centers = np.array([coil.center for coil in shaping_coils])

        coil_centers = np.concatenate((existing_centers, coil_centers), axis=0)

    intersections = compute_intersections(coil_centers, min_separation)

    if shaping_coils is not None:
        # remove new points which are too close to existing coils
        current_coil_count = len(shaping_coils)

        current_candidate_intersections = intersections[
            :current_coil_count, current_coil_count:
        ]

        filtered_candidate_indices = np.where(
            current_candidate_intersections.sum(axis=0) == 0
        )[0]

        filtered_candidate_indices_offset = (
            filtered_candidate_indices + current_coil_count
        )  # adjust indices to account for existing coils

        intersections = intersections[filtered_candidate_indices_offset, :][
            :, filtered_candidate_indices_offset
        ]
    else:
        filtered_candidate_indices = np.arange(len(candidate_coil_centers))

    graph = nx.from_numpy_array(intersections)
    mis_nodes = approx.maximum_independent_set(graph)

    mis_nodes = np.array(list(mis_nodes))

    if len(mis_nodes) == 0:
        print(
            "No independent set found, all candidate coils are too close to existing coils."
        )
        return shaping_coils
    else:
        selected_coil_indices = filtered_candidate_indices[mis_nodes]
        final_centers = candidate_coil_centers[selected_coil_indices]
        final_normals = candidate_coil_normals[selected_coil_indices]
        final_currents = candidate_coil_currents[selected_coil_indices]

    for i in range(len(final_centers)):
        coil = FourierPlanarCoil(
            current=final_currents[i],
            center=final_centers[i],
            normal=final_normals[i],
            r_n=rad,
            basis="xyz",
        )
        if shaping_coils is None:
            shaping_coils = CoilSet(coil, NFP=2, sym=True)
        else:
            shaping_coils.append(coil)

    print(f"Added {len(final_centers)} new coils.")
    return shaping_coils


def pack_remaining_coils(shaping_coils, shaping_sheet):
    # pack the remaining coils densely
    hfp_grid = LinearGrid(M=20, N=40, NFP=2, sym=True)
    surface_data = shaping_sheet.compute(
        ["Phi", "n_rho", "X", "Y", "Z"], grid=hfp_grid, basis="xyz"
    )
    sheet_pos = np.array([surface_data["X"], surface_data["Y"], surface_data["Z"]]).T
    sheet_normal = jnp.vstack(
        [
            surface_data["n_rho"][:, 0],
            surface_data["n_rho"][:, 1],
            surface_data["n_rho"][:, 2],
        ]
    ).T
    sheet_current = -surface_data["Phi"]

    min_separation = 2 * rad + offset

    shaping_coils = add_coils(
        sheet_pos,
        sheet_normal,
        sheet_current,
        min_separation,
        shaping_coils,
    )

    return shaping_coils


def optimize_coils_xpoint(
    shaping_coils, encircling, eq, eval_grid_N=360, curve_N=24, **weights
):
    N = 40

    encircling_points = encircling._compute_position(grid=2 * N, basis="xyz")
    # collapse first two dimensions
    encircling_points = jnp.reshape(encircling_points, (-1, 3))

    # Define a starting curve based on a curve below the plasma boundary
    offset = 0.02
    grid = LinearGrid(rho=[1.0], theta=[3 * np.pi / 2], N=24, NFP=eq.NFP)
    l = eq.compute(["R", "phi", "Z"], grid=grid)
    coords = jnp.stack([l["R"], l["phi"], l["Z"] - offset]).T

    # Define c as a FourierRZCoil (this code doesn't work otherwise!!)
    curve = FourierRZCoil.from_values(
        current=0, coords=coords, N=curve_N, basis="rpz", NFP=eq.NFP, name="x-point"
    )
    obj = ObjectiveFunction(
        (
            QuadraticFlux(
                eq,
                [shaping_coils, encircling],
                field_grid=[LinearGrid(N=N), LinearGrid(N=2 * N)],
                bs_chunk_size=1,
                B_plasma_chunk_size=1,
                weight=weights.get("b_dot_n", 10),
                loss_function="mean",
            ),
            # small coils
            CoilSetMinDistance(
                shaping_coils,
                bounds=(offset, np.inf),
                use_softmin=True,
                softmin_alpha=100,
                dist_chunk_size=1,
                grid=LinearGrid(N=N),
                normal_project_dist=0.5,
                name="small coil-coil dist",
                weight=weights.get("coil_coil", 1),
            ),
            PlasmaCoilSetDistanceBound(
                eq,
                shaping_coils,
                bounds=(1.2, 2.0),
                use_softmin=True,
                softmin_alpha=100,
                eq_fixed=True,
                dist_chunk_size=1,
                coil_grid=LinearGrid(N=5),
                plasma_grid=LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP),
                name="small plasma-coil dist",
                weight=weights.get("plasma_coil", 1),
            ),
            MinCoilSetPointDistance(
                coil=shaping_coils,
                points=encircling_points,
                bounds=(0.5, np.inf),
                coil_grid=LinearGrid(N=N),
                use_softmin=True,
                softmin_alpha=100,
                dist_chunk_size=1,
                name="small coil-encircling dist",
                weight=weights.get("coil_encircling", 1),
            ),
            MinCoilSetPointDistance(
                coil=shaping_coils,
                points=coords,
                bounds=(1.2, np.inf),
                coil_grid=LinearGrid(N=N),
                dist_chunk_size=1,
                name="small coil-xpoint dist",
                weight=weights.get("coil_xpt", 1),
            ),
            Bxdl(
                curve=curve,
                eq=eq,
                field=[shaping_coils, encircling],
                target=0,
                field_grid=[LinearGrid(N=N), LinearGrid(N=2 * N)],
                eval_grid=LinearGrid(N=eval_grid_N),
                bs_chunk_size=1,
                name="x point Bxdl",
                field_fixed=False,
                curve_fixed=False,
                eq_kwargs={"method": "biot-savart"},
                weight=weights.get("bxdl", 1e2),
            ),
            FixCoilCurrent(
                shaping_coils,
                bounds=(-max_current, max_current),
                name="small coil current",
                weight=weights.get("small_current", 1),
            ),
            XPointDistanceBound(
                eq,
                curve,
                N_grid=36,
                M_grid=36,
                eq_fixed=True,
                target=0,
                weight=weights.get("xpt_dist", 1),
                name="x-point distance bound",
            ),
        ),
        deriv_mode="batched",
        jac_chunk_size=1024,
    )

    constraints = (
        FixParameters(shaping_coils, {"r_n": True}),
        FixParameters(encircling),
        FixCoilCurrent(curve),
    )

    obj.build()

    opt = Optimizer("lsq-exact")
    (shaping_coils, curve, _), info = opt.optimize(
        (shaping_coils, curve, encircling),
        objective=obj,
        constraints=constraints,
        verbose=3,
        maxiter=250,
        ftol=1e-6,
        copy=True,
    )

    grid_Bn = LinearGrid(
        rho=np.array([1.0]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False
    )
    data = eq.compute("|e_theta x e_zeta|", grid=grid_Bn)
    total_field = SumMagneticField([shaping_coils, encircling])
    Bn, surf_coords = total_field.compute_Bnormal(eq, eval_grid=grid_Bn)
    B_norm_vec = total_field.compute_magnetic_field(surf_coords)
    B_norm_vec += compute_B_plasma(eq, eval_grid=grid_Bn)
    B_norm = np.linalg.norm(B_norm_vec, axis=1)
    Bn_avg = np.sum(np.abs(Bn) * data["|e_theta x e_zeta|"] * grid_Bn.weights) / np.sum(
        B_norm * data["|e_theta x e_zeta|"] * grid_Bn.weights
    )
    print("\n================")
    print("avg(B*n) without shaping sheet = {:.2f}%".format(Bn_avg * 1e2))
    print("================")

    return shaping_coils, curve


def postproc_finite_build(shaping_coils, encircling):
    # convert coils to finite build version
    shaping_coils_fb_list = [
        FourierPlanarFiniteBuildCoil.from_FourierPlanarCoil(
            coil, cross_section_dims=[0.2, 0.2]
        )
        for coil in shaping_coils
    ]
    shaping_coils_fb = CoilSet(
        shaping_coils_fb_list,
        NFP=shaping_coils.NFP,
        sym=shaping_coils.sym,
        check_intersection=False,
    )

    encircling_flatten = CoilSet.from_symmetry(
        encircling, NFP=encircling.NFP, sym=True, check_intersection=False
    )
    encircling_currents = np.array([coil.current for coil in encircling_flatten.coils])
    encircling_a = np.repeat(0.4, len(encircling_currents))
    current_density = 7e6 / (
        0.2 * 0.4 * 1.4
    )  # A/m^2, scaled to A008 new cross section for large encircling coils
    encircling_b = np.abs(encircling_currents / (current_density * encircling_a))

    encircling_fb_list = [
        FourierXYFiniteBuildCoil.from_FourierXYCoil(
            coil, cross_section_dims=[encircling_a[i], encircling_b[i]]
        )
        for i, coil in enumerate(encircling_flatten)
    ]
    encircling_fb = CoilSet(
        encircling_fb_list,
        NFP=encircling_flatten.NFP,
        sym=encircling_flatten.sym,
        check_intersection=False,
    )

    print("Shaping coils finite build:")
    objective = ObjectiveFunction(
        (
            CoilSetMaxB(
                coil=shaping_coils_fb,
                field=encircling_fb,
                component="mag",
                bounds=(0, 20),
                xsection_grid=LinearGrid(L=2, M=1, endpoint=True),
                centerline_grid=LinearGrid(N=64),
                field_grid=LinearGrid(N=128),
                use_softmax=False,
                bs_chunk_size=512,
            ),
        ),
        deriv_mode="batched",
    )

    objective.build()

    x = objective.x(shaping_coils_fb)
    shaping_fields = np.array(objective.compute_unscaled(x))
    objective.print_value(x)

    print("Encircling coils finite build:")

    objective_large = ObjectiveFunction(
        (
            CoilSetMaxB(
                coil=encircling_fb,
                field=shaping_coils_fb,
                component="mag",
                bounds=(0, 20),
                xsection_grid=LinearGrid(L=6, M=3, endpoint=True),
                centerline_grid=LinearGrid(N=64),
                field_grid=LinearGrid(N=64),
                use_softmax=False,
                bs_chunk_size=512,
            ),
        ),
        deriv_mode="batched",
    )

    objective_large.build()
    x_large = objective_large.x(encircling_fb)
    encircling_fields = np.array(objective_large.compute_unscaled(x_large))
    objective_large.print_value(x_large)

    shaping_lengths = shaping_coils.compute(
        ["length"], grid=LinearGrid(N=64), basis="xyz"
    )
    shaping_lengths = np.array([coil_data["length"] for coil_data in shaping_lengths])
    shaping_currents = np.array([coil.current for coil in shaping_coils.coils])
    shaping_currents = np.abs(shaping_currents)

    encircling_lengths = encircling_flatten.compute(
        ["length"], grid=LinearGrid(N=64), basis="xyz"
    )
    encircling_lengths = np.array(
        [coil_data["length"] for coil_data in encircling_lengths]
    )
    encircling_currents = np.array([coil.current for coil in encircling_flatten.coils])
    encircling_currents = np.abs(encircling_currents)

    shaping_gmat = shaping_lengths * shaping_currents * shaping_fields
    encircling_gmat = encircling_lengths * encircling_currents * encircling_fields

    total_shaping_gmat = np.sum(shaping_gmat) * 4
    total_encircling_gmat = np.sum(encircling_gmat)

    print("Total shaping coil GmAT:", total_shaping_gmat * 1e-9)
    print("Total encircling coil GmAT:", total_encircling_gmat * 1e-9)

    total_gmat = np.sum(shaping_gmat) * 4 + np.sum(encircling_gmat)

    print("Total GmAT:", total_gmat * 1e-9)

    shaping_gmat_capped = (
        shaping_lengths * shaping_currents * np.clip(shaping_fields, 0, 20)
    )
    total_shaping_gmat_capped = np.sum(shaping_gmat_capped) * 4
    print("Total shaping coil GmAT capped:", total_shaping_gmat_capped * 1e-9)
    encircling_gmat_capped = (
        encircling_lengths * encircling_currents * np.clip(encircling_fields, 0, 20)
    )
    total_encircling_gmat_capped = np.sum(encircling_gmat_capped)
    print("Total encircling coil GmAT capped:", total_encircling_gmat_capped * 1e-9)
    total_gmat_capped = total_shaping_gmat_capped + total_encircling_gmat_capped
    print("Total GmAT capped:", total_gmat_capped * 1e-9)

    total_tape = (total_gmat_capped * 1e-9) / 4.03
    print("Total tape length with capping (Mm):", total_tape)


def optimize_coils(dir, **weights):
    out_tag = (
        f"shaping_div_98_{int(rad * 100)}cm_{int(max_current / 1e6)}MA_"
        + "_".join([f"{key}{value}" for key, value in weights.items()])
    )

    log_filename = f"{out_tag}.log"
    with open(log_filename, "w") as log_file, contextlib.redirect_stdout(log_file):

        print_backend_info()

        eq = load(dir + "equil_div-opt_98_DESC_fixed.h5")

        shaping_sheet = load(dir + "shaping_sheet_div_98.h5")
        shaping_sheet.change_resolution(M=12, N=24)
        encircling = load(dir + "encircling_div_98.h5")

        print("Dense packing remaining coils")
        shaping_coils = pack_remaining_coils(None, shaping_sheet)

        # Final optimization
        print("Final optimization of coils")
        shaping_coils, curve = optimize_coils_xpoint(
            shaping_coils, encircling, eq, **weights
        )
        print("Final optimization done.")

        shaping_coils.save(out_tag + "_shaping_coils_final.h5")

        postproc_finite_build(shaping_coils, encircling)

        all_fields = SumMagneticField([encircling, shaping_coils])

        all_fields.save_mgrid(
            dir + "mgrid_" + out_tag + "-128x128x128_vecpot_larger.nc",
            Rmin=5,
            Rmax=11,
            Zmin=-5,
            Zmax=5,
            nR=128,
            nZ=128,
            nphi=128,
            save_vector_potential=True,
            chunk_size=300,
        )

        curve.save(dir + out_tag + "_xpt_final.h5")
