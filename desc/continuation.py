import numpy as np
import scipy.optimize
import copy
from termcolor import colored

from desc.backend import use_jax, jit
from desc.utils import Timer
from desc.grid import ConcentricGrid
from desc.transform import Transform
from desc.configuration import EquilibriaFamily
from desc.objective_funs import ObjectiveFunctionFactory
from desc.perturbations import perturb_continuation_params
from desc.vmec import VMECIO


def solve_eq_continuation(inputs, file_name=None, vmec_path=None, device=None):
    """Solves for an equilibrium by continuation method

    Follows this procedure to solve the equilibrium:
        1. Creates an initial guess from the given inputs
        2. Optimizes the equilibrium's flux surfaces by minimizing
            the given objective function.
        3. Step up to higher resolution and perturb the previous solution
        4. Repeat 2 and 3 until at desired resolution

    Parameters
    ----------
    inputs : dict
        dictionary with input parameters defining problem setup and solver options
    file_name : str or path-like
        file to save checkpoint data (Default value = None)
    vmec_path : str or path-like
        VMEC netCDF file to load initial guess from (Default value = None)
    device : jax.device or None
        device handle to JIT compile to (Default value = None)

    Returns
    -------
    equil_fam : EquilibriaFamily
        Container object that contains a list of the intermediate solutions,
            as well as the final solution, stored as Equilibrium objects
    timer : Timer
        Timer object containing timing data for individual iterations

    """
    timer = Timer()
    timer.start("Total time")

    sym = inputs["sym"]
    NFP = inputs["NFP"]
    Psi = inputs["Psi"]
    L = inputs["L"]  # arr
    M = inputs["M"]  # arr
    N = inputs["N"]  # arr
    M_grid = inputs["M_grid"]  # arr
    N_grid = inputs["N_grid"]  # arr
    bdry_ratio = inputs["bdry_ratio"]  # arr
    pres_ratio = inputs["pres_ratio"]  # arr
    zeta_ratio = inputs["zeta_ratio"]  # arr
    pert_order = inputs["pert_order"]  # arr
    ftol = inputs["ftol"]  # arr
    xtol = inputs["xtol"]  # arr
    gtol = inputs["gtol"]  # arr
    nfev = inputs["nfev"]  # arr
    optim_method = inputs["optim_method"]
    errr_mode = inputs["errr_mode"]
    bdry_mode = inputs["bdry_mode"]
    zern_mode = inputs["zern_mode"]
    node_mode = inputs["node_mode"]
    profiles = inputs["profiles"]
    boundary = inputs["boundary"]
    axis = inputs["axis"]
    verbose = inputs["verbose"]

    if file_name is not None:
        checkpoint = True
    else:
        checkpoint = False

    arr_len = M.size
    for ii in range(arr_len):

        if verbose > 0:
            print("================")
            print("Step {}/{}".format(ii + 1, arr_len))
            print("================")
            print("Spectral resolution (L,M,N)=({},{},{})".format(L[ii], M[ii], N[ii]))
            print("Node resolution (M,N)=({},{})".format(M_grid[ii], N_grid[ii]))
            print("Boundary ratio = {}".format(bdry_ratio[ii]))
            print("Pressure ratio = {}".format(pres_ratio[ii]))
            print("Zeta ratio = {}".format(zeta_ratio[ii]))
            print("Perturbation Order = {}".format(pert_order[ii]))
            print("Function tolerance = {}".format(ftol[ii]))
            print("Gradient tolerance = {}".format(gtol[ii]))
            print("State vector tolerance = {}".format(xtol[ii]))
            print("Max function evaluations = {}".format(nfev[ii]))
            print("================")

        # initial solution
        # at initial soln, must: create bases, create grids, create transforms
        if ii == 0:
            timer.start("Iteration {} total".format(ii + 1))

            inputs_ii = {
                "sym": sym,
                "Psi": Psi,
                "NFP": NFP,
                "L": L[ii],
                "M": M[ii],
                "N": N[ii],
                "index": zern_mode,
                "bdry_mode": bdry_mode,
                "zeta_ratio": zeta_ratio[ii],
                "profiles": profiles,
                "boundary": boundary,
                "axis": axis,
            }
            # apply pressure ratio
            inputs_ii["profiles"][:, 1] *= pres_ratio[ii]
            # apply boundary ratio
            bdry_factor = np.where(inputs_ii["boundary"][:, 1] != 0, bdry_ratio[ii], 1)
            inputs_ii["boundary"][:, 2] *= bdry_factor
            inputs_ii["boundary"][:, 3] *= bdry_factor

            equil_fam = EquilibriaFamily(inputs=inputs_ii)
            if vmec_path is not None:
                equil_fam[ii] = VMECIO.load(
                    vmec_path, L=L[ii], M=M[ii], N=N[ii], index=zern_mode
                )
            equil = equil_fam[ii]

            timer.start("Transform precomputation")
            if verbose > 0:
                print("Precomputing Transforms")

            # bases (extracted from Equilibrium)
            R0_basis = equil.R0_basis
            Z0_basis = equil.Z0_basis
            r_basis = equil.r_basis
            l_basis = equil.l_basis
            p_basis = equil.p_basis
            i_basis = equil.i_basis
            R1_basis = equil.R1_basis
            Z1_basis = equil.Z1_basis

            # grid
            RZ_grid = ConcentricGrid(
                M_grid[ii],
                N_grid[ii],
                NFP=NFP,
                sym=sym,
                axis=False,
                index=zern_mode,
                surfs=node_mode,
            )

            # transforms
            R0_transform = Transform(RZ_grid, R0_basis, derivs=2)
            Z0_transform = Transform(RZ_grid, Z0_basis, derivs=2)
            R1_transform = Transform(RZ_grid, R1_basis, derivs="force")
            Z1_transform = Transform(RZ_grid, Z1_basis, derivs="force")
            r_transform = Transform(RZ_grid, r_basis, derivs="force")
            l_transform = Transform(RZ_grid, l_basis, derivs="force")
            p_transform = Transform(RZ_grid, p_basis, derivs=1)
            i_transform = Transform(RZ_grid, i_basis, derivs=1)

            timer.stop("Transform precomputation")
            if verbose > 1:
                timer.disp("Transform precomputation")

        # continuing from previous solution
        else:
            equil_fam.append(copy.deepcopy(equil))
            equil = equil_fam[ii]
            equil.x0 = equil.x  # new initial guess is previous solution
            equil.solved = False

            timer.start("Iteration {} changing resolution".format(ii + 1))
            # change grids
            if M_grid[ii] != M_grid[ii - 1] or N_grid[ii] != N_grid[ii - 1]:
                RZ_grid = ConcentricGrid(
                    M_grid[ii],
                    N_grid[ii],
                    NFP=NFP,
                    sym=sym,
                    axis=False,
                    index=zern_mode,
                    surfs=node_mode,
                )
                if verbose:
                    print(
                        "Changing node resolution from (M_grid,N_grid) = ({},{}) to ({},{})".format(
                            M_grid[ii - 1], N_grid[ii - 1], M_grid[ii], N_grid[ii]
                        )
                    )

            # change bases
            if M[ii] != M[ii - 1] or N[ii] != N[ii - 1] or L[ii] != L[ii - 1]:
                # update equilibrium bases to the new resolutions
                if verbose > 0:
                    print(
                        "Changing spectral resolution from (L,M,N) = ({},{},{}) to ({},{},{})".format(
                            L[ii - 1], M[ii - 1], N[ii - 1], L[ii], M[ii], N[ii]
                        )
                    )

                equil.change_resolution(L=L[ii], M=M[ii], N=N[ii])
                R0_basis = equil.R0_basis
                Z0_basis = equil.Z0_basis
                r_basis = equil.r_basis
                l_basis = equil.l_basis
                p_basis = equil.p_basis
                i_basis = equil.i_basis
                R1_basis = equil.R1_basis
                Z1_basis = equil.Z1_basis
            # change transform matrices

            R0_transform.change_resolution(grid=RZ_grid, basis=R0_basis)
            Z0_transform.change_resolution(grid=RZ_grid, basis=Z0_basis)
            r_transform.change_resolution(grid=RZ_grid, basis=r_basis)
            l_transform.change_resolution(grid=RZ_grid, basis=l_basis)
            p_transform.change_resolution(grid=RZ_grid)
            i_transform.change_resolution(grid=RZ_grid)
            R1_transform.change_resolution(grid=RZ_grid, basis=R1_basis)
            Z1_transform.change_resolution(grid=RZ_grid, basis=Z1_basis)
            timer.stop("Iteration {} changing resolution".format(ii + 1))
            if verbose > 1:
                timer.disp("Iteration {} changing resolution".format(ii + 1))

            # continuation parameters
            delta_bdry = bdry_ratio[ii] - bdry_ratio[ii - 1]
            delta_pres = pres_ratio[ii] - pres_ratio[ii - 1]
            delta_zeta = zeta_ratio[ii] - zeta_ratio[ii - 1]
            deltas = np.array([delta_bdry, delta_pres, delta_zeta])

            # need a non-scalar objective function to do the perturbations
            obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
                errr_mode,
                R0_transform=R0_transform,
                Z0_transform=Z0_transform,
                r_transform=r_transform,
                l_transform=l_transform,
                R1_transform=R1_transform,
                Z1_transform=Z1_transform,
                p_transform=p_transform,
                i_transform=i_transform,
            )
            equil_obj = obj_fun.compute
            callback = obj_fun.callback
            args = (
                equil.R1_mn,
                equil.Z1_mn,
                equil.p_l,
                equil.i_l,
                equil.Psi,
                zeta_ratio[ii - 1],
            )

            # TODO: should probably perturb before expanding resolution
            # perturbations
            if np.any(deltas):
                if verbose > 1:
                    print("Perturbing equilibrium")
                x, timer = perturb_continuation_params(
                    equil.x, equil_obj, deltas, args, pert_order[ii], verbose, timer
                )
                equil.x = x

        # equilibrium objective function
        if optim_method in ["bfgs"]:
            scalar = True
        else:
            scalar = False

        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
            errr_mode,
            R0_transform=R0_transform,
            Z0_transform=Z0_transform,
            r_transform=r_transform,
            l_transform=l_transform,
            R1_transform=R1_transform,
            Z1_transform=Z1_transform,
            p_transform=p_transform,
            i_transform=i_transform,
        )

        equil.objective = obj_fun
        equil.optimizer = "scipy-trf"
        equil.solve(
            ftol=ftol[ii],
            xtol=xtol[ii],
            gtol=gtol[ii],
            verbose=verbose,
            maxiter=nfev[ii],
        )

        if checkpoint:
            if verbose > 0:
                print("Saving latest iteration")
            equil_fam.save(file_name)

    timer.stop("Total time")
    print("====================")
    print("Done")
    if verbose > 1:
        timer.disp("Total time")
    if file_name is not None:
        print("Output written to {}".format(file_name))
    print("====================")

    return equil_fam, timer
