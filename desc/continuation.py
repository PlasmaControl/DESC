import numpy as np
import scipy.optimize
import warnings
import copy

from desc.backend import TextColors, use_jax, jit
from desc.utils import Timer
from desc.grid import LinearGrid, ConcentricGrid
from desc.transform import Transform
from desc.configuration import EquilibriaFamily
from desc.objective_funs import is_nested, ObjectiveFunctionFactory
from desc.perturbations import perturb_continuation_params


def solve_eq_continuation(inputs, file_name=None, device=None):
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

    stell_sym = inputs['stell_sym']
    NFP = inputs['NFP']
    Psi_lcfs = inputs['Psi_lcfs']
    M = inputs['Mpol']                  # arr
    N = inputs['Ntor']                  # arr
    delta_lm = inputs['delta_lm']       # arr
    Mnodes = inputs['Mnodes']           # arr
    Nnodes = inputs['Nnodes']           # arr
    bdry_ratio = inputs['bdry_ratio']   # arr
    pres_ratio = inputs['pres_ratio']   # arr
    zeta_ratio = inputs['zeta_ratio']   # arr
    errr_ratio = inputs['errr_ratio']   # arr
    pert_order = inputs['pert_order']   # arr
    ftol = inputs['ftol']               # arr
    xtol = inputs['xtol']               # arr
    gtol = inputs['gtol']               # arr
    nfev = inputs['nfev']               # arr
    optim_method = inputs['optim_method']
    errr_mode = inputs['errr_mode']
    bdry_mode = inputs['bdry_mode']
    zern_mode = inputs['zern_mode']
    node_mode = inputs['node_mode']
    cP = inputs['cP']
    cI = inputs['cI']
    axis = inputs['axis']
    bdry = inputs['bdry']
    verbose = inputs['verbose']

    if file_name is not None:
        checkpoint = True
    else:
        checkpoint = False

    arr_len = M.size
    for ii in range(arr_len):

        if verbose > 0:
            print("================")
            print("Step {}/{}".format(ii+1, arr_len))
            print("================")
            print("Spectral resolution (M,N,delta_lm)=({},{},{})".format(
                M[ii], N[ii], delta_lm[ii]))
            print("Node resolution (M,N)=({},{})".format(
                Mnodes[ii], Nnodes[ii]))
            print("Boundary ratio = {}".format(bdry_ratio[ii]))
            print("Pressure ratio = {}".format(pres_ratio[ii]))
            print("Zeta ratio = {}".format(zeta_ratio[ii]))
            print("Error ratio = {}".format(errr_ratio[ii]))
            print("Perturbation Order = {}".format(pert_order[ii]))
            print("Function tolerance = {}".format(ftol[ii]))
            print("Gradient tolerance = {}".format(gtol[ii]))
            print("State vector tolerance = {}".format(xtol[ii]))
            print("Max function evaluations = {}".format(nfev[ii]))
            print("================")

        # initial solution
        # at initial soln, must: create bases, create grids, create transforms
        if ii == 0:
            timer.start("Iteration {} total".format(ii+1))

            inputs_ii = {
                'L': delta_lm[ii],
                'M': M[ii],
                'N': N[ii],
                'cP': cP*pres_ratio[ii],
                'cI': cI,
                'Psi': Psi_lcfs,
                'NFP': NFP,
                'bdry': bdry,
                'sym': stell_sym,
                'index': zern_mode,
                'bdry_mode': bdry_mode,
                'bdry_ratio': bdry_ratio[ii],
                'axis': axis,
            }
            equil_fam = EquilibriaFamily(inputs=inputs_ii)
            equil = equil_fam[ii]

            timer.start("Transform precomputation")
            if verbose > 0:
                print("Precomputing Transforms")

            # bases (extracted from Equilibrium)
            R_basis, Z_basis, L_basis, P_basis, I_basis = equil.R_basis, \
                equil.Z_basis, \
                equil.L_basis, \
                equil.P_basis, \
                equil.I_basis

            # grids
            RZ_grid = ConcentricGrid(Mnodes[ii], Nnodes[ii], NFP=NFP, sym=stell_sym,
                                     axis=False, index=zern_mode, surfs=node_mode)
            L_grid = LinearGrid(
                M=Mnodes[ii], N=2*Nnodes[ii]+1, NFP=NFP, sym=stell_sym)

            # transforms
            R_transform = Transform(RZ_grid, R_basis, derivs=3)
            Z_transform = Transform(RZ_grid, Z_basis, derivs=3)
            R1_transform = Transform(L_grid, R_basis)
            Z1_transform = Transform(L_grid, Z_basis)
            L_transform = Transform(L_grid,  L_basis, derivs=0)
            P_transform = Transform(RZ_grid, P_basis, derivs=1)
            I_transform = Transform(RZ_grid, I_basis, derivs=1)

            timer.stop("Transform precomputation")
            if verbose > 1:
                timer.disp("Transform precomputation")

        # continuing from previous solution
        else:
            equil_fam.append(copy.deepcopy(equil))
            equil = equil_fam[ii]
            equil.x0 = equil.x  # new initial guess is previous solution
            equil.solved = False

            # change grids
            if Mnodes[ii] != Mnodes[ii-1] or Nnodes[ii] != Nnodes[ii-1]:
                RZ_grid = ConcentricGrid(Mnodes[ii], Nnodes[ii], NFP=NFP, sym=stell_sym,
                                         axis=False, index=zern_mode, surfs=node_mode)
                L_grid = LinearGrid(
                    M=Mnodes[ii], N=2*Nnodes[ii]+1, NFP=NFP, sym=stell_sym)

            # change bases
            if M[ii] != M[ii-1] or N[ii] != N[ii-1] or delta_lm[ii] != delta_lm[ii-1]:
                # update equilibrium bases to the new resolutions
                equil.change_resolution(L=delta_lm[ii], M=M[ii], N=N[ii])
                R_basis, Z_basis, L_basis = equil.R_basis, equil.Z_basis, equil.L_basis

            # change transform matrices
            timer.start(
                "Iteration {} changing resolution".format(ii+1))
            if verbose > 0:
                print("Changing node resolution from (Mnodes,Nnodes) = ({},{}) to ({},{})".format(
                    Mnodes[ii-1], Nnodes[ii-1], Mnodes[ii], Nnodes[ii]))
                print("Changing spectral resolution from (L,M,N) = ({},{},{}) to ({},{},{})".format(
                    delta_lm[ii-1], M[ii-1], N[ii-1], delta_lm[ii], M[ii], N[ii]))

            R_transform.change_resolution(grid=RZ_grid, basis=R_basis)
            Z_transform.change_resolution(grid=RZ_grid, basis=Z_basis)
            R1_transform.change_resolution(grid=L_grid, basis=R_basis)
            Z1_transform.change_resolution(grid=L_grid, basis=Z_basis)
            L_transform.change_resolution(grid=L_grid, basis=L_basis)
            P_transform.change_resolution(grid=RZ_grid)
            I_transform.change_resolution(grid=RZ_grid)
            timer.stop(
                "Iteration {} changing resolution".format(ii+1))
            if verbose > 1:
                timer.disp(
                    "Iteration {} changing resolution".format(ii+1))

            # continuation parameters
            delta_bdry = bdry_ratio[ii] - bdry_ratio[ii-1]
            delta_pres = pres_ratio[ii] - pres_ratio[ii-1]
            delta_zeta = zeta_ratio[ii] - zeta_ratio[ii-1]
            deltas = np.array([delta_bdry, delta_pres, delta_zeta])

            # need a non-scalar objective function to do the perturbations
            obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
                errr_mode,
                R_transform=R_transform, Z_transform=Z_transform,
                R1_transform=R1_transform, Z1_transform=Z1_transform,
                L_transform=L_transform, P_transform=P_transform,
                I_transform=I_transform)
            equil_obj = obj_fun.compute
            callback = obj_fun.callback
            args = (equil.cRb, equil.cZb, equil.cP, equil.cI, equil.Psi,
                    bdry_ratio[ii-1], pres_ratio[ii-1], zeta_ratio[ii-1], errr_ratio[ii-1])

            # TODO: should probably perturb before expanding resolution
            # perturbations
            if np.any(deltas):
                if verbose > 1:
                    print("Perturbing equilibrium")
                x, timer = perturb_continuation_params(equil.x, equil_obj, deltas, args,
                                                       pert_order[ii], verbose, timer)
                equil.x = x

        # equilibrium objective function
        if optim_method in ['bfgs']:
            scalar = True
        else:
            scalar = False
        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
            errr_mode,
            R_transform=R_transform, Z_transform=Z_transform,
            R1_transform=R1_transform, Z1_transform=Z1_transform,
            L_transform=L_transform, P_transform=P_transform,
            I_transform=I_transform)
        equil_obj = obj_fun.compute
        equil_obj_scalar = obj_fun.compute_scalar
        grad = obj_fun.grad
        hess = obj_fun.hess
        jac = obj_fun.jac
        callback = obj_fun.callback
        args = (equil.cRb, equil.cZb, equil.cP, equil.cI, equil.Psi,
                bdry_ratio[ii-1], pres_ratio[ii-1], zeta_ratio[ii-1], errr_ratio[ii-1])

        if use_jax:
            if verbose > 0:
                print("Compiling objective function")
            if device is None:
                import jax
                device = jax.devices()[0]
            timer.start("Iteration {} compilation".format(ii+1))

            if optim_method in ['bfgs']:
                equil_obj_jit = jit(
                    equil_obj_scalar, static_argnums=(), device=device)
                grad_obj_jit = jit(grad, device=device)
                f0 = equil_obj_jit(equil.x, *args)
                g0 = grad_obj_jit(equil.x, *args)
            else:
                equil_obj_jit = jit(
                    equil_obj, static_argnums=(), device=device)
                jac_obj_jit = jit(jac, device=device)
                f0 = equil_obj_jit(equil.x, *args)
                j0 = jac_obj_jit(equil.x, *args)

            timer.stop("Iteration {} compilation".format(ii+1))
            if verbose > 1:
                timer.disp("Iteration {} compilation".format(ii+1))
        else:
            equil_obj_jit = equil_obj
            jac_obj_jit = '2-point'
        if verbose > 0:
            print("Starting optimization")

        x_init = equil.x
        timer.start("Iteration {} solution".format(ii+1))
        if optim_method in ['bfgs']:
            out = scipy.optimize.minimize(equil_obj_jit,
                                          x0=x_init,
                                          args=args,
                                          method=optim_method,
                                          jac=grad_obj_jit,
                                          tol=gtol[ii],
                                          options={'maxiter': nfev[ii],
                                                   'disp': verbose})

        elif optim_method in ['trf', 'lm', 'dogbox']:
            out = scipy.optimize.least_squares(equil_obj_jit,
                                               x0=x_init,
                                               args=args,
                                               jac=jac_obj_jit,
                                               method=optim_method,
                                               x_scale='jac',
                                               ftol=ftol[ii],
                                               xtol=xtol[ii],
                                               gtol=gtol[ii],
                                               max_nfev=nfev[ii],
                                               verbose=verbose)
        else:
            raise NotImplementedError(
                TextColors.FAIL + "optim_method must be one of 'bfgs', 'trf', 'lm', 'dogbox'" + TextColors.ENDC)

        timer.stop("Iteration {} solution".format(ii+1))

        equil.x = out['x']
        equil.solved = True

        if verbose > 1:
            timer.disp("Iteration {} solution".format(ii+1))
            timer.pretty_print("Iteration {} avg time per step".format(ii+1),
                               timer["Iteration {} solution".format(ii+1)]/out['nfev'])
        if verbose > 0:
            print("Start of Step {}:".format(ii+1))
            callback(x_init, *args)
            print("End of Step {}:".format(ii+1))
            callback(equil.x, *args)

        if checkpoint:
            if verbose > 0:
                print('Saving latest iteration')
            equil_fam.save(file_name)

        if not is_nested(equil.cR, equil.cZ, equil.R_basis, equil.Z_basis):
            warnings.warn(TextColors.WARNING + 'WARNING: Flux surfaces are no longer nested, exiting early.'
                          + 'Consider increasing errr_ratio or taking smaller perturbation steps' + TextColors.ENDC)
            break

    timer.stop("Total time")
    print('====================')
    print('Done')
    if verbose > 1:
        timer.disp("Total time")
    if file_name is not None:
        print('Output written to {}'.format(file_name))
    print('====================')

    return equil_fam, timer
