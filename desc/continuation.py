import numpy as np
import scipy.optimize
import warnings
import copy

from desc.backend import jit, use_jax, Timer, TextColors, Tristate
from desc.backend import jacfwd, grad
from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import PowerSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.transform import Transform
from desc.configuration import Equilibrium, EquilibriaFamily
from desc.objective_funs import is_nested, ObjectiveFunctionFactory
from desc.equilibrium_io import Checkpoint
from desc.perturbations import perturb_continuation_params


def solve_eq_continuation(inputs, checkpoint_filename=None, device=None):
    """Solves for an equilibrium by continuation method

    Steps up resolution, perturbs pressure, 3d bdry etc.

    Parameters
    ----------
    inputs : dict
        dictionary with input parameters defining problem setup and solver options
    checkpoint_filename : str or path-like
        file to save checkpoint data (Default value = None)
    device : jax.device or None
        device handle to JIT compile to (Default value = None)

    Returns
    -------
    iterations : dict
        dictionary of intermediate solutions
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

    if checkpoint_filename is not None:
        checkpoint = True
        checkpoint_file = Checkpoint(checkpoint_filename, write_ascii=True)
    else:
        checkpoint = False

    if not use_jax:
        pert_order *= 0

    if stell_sym:
        R_sym = Tristate(True)
        Z_sym = Tristate(False)
        L_sym = Tristate(False)
    else:
        R_sym = Tristate(None)
        Z_sym = Tristate(None)
        L_sym = Tristate(None)

    equil_fam = EquilibriaFamily(inputs=inputs)

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
                'axis': axis
            }
            equil = Equilibrium(inputs=inputs_ii)
            x = equil.x

            timer.start("Transform precomputation")
            if verbose > 0:
                print("Precomputing Transforms")
            # grids
            RZ_grid = ConcentricGrid(Mnodes[ii], Nnodes[ii], NFP=NFP, sym=stell_sym,
                                     axis=False, index=zern_mode, surfs=node_mode)
            L_grid = LinearGrid(M=Mnodes[ii], N=2*Nnodes[ii]+1, NFP=NFP, sym=stell_sym)
            # bases
            R_basis = FourierZernikeBasis(L=delta_lm[ii], M=M[ii], N=N[ii],
                                          NFP=NFP, sym=R_sym, index=zern_mode)
            Z_basis = FourierZernikeBasis(L=delta_lm[ii], M=M[ii], N=N[ii],
                                          NFP=NFP, sym=Z_sym, index=zern_mode)
            L_basis = DoubleFourierSeries(M=M[ii], N=N[ii], NFP=NFP, sym=L_sym)
            P_basis = PowerSeries(L=cP.size-1)
            I_basis = PowerSeries(L=cI.size-1)
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

#            if checkpoint:
#                checkpoint_file.write_iteration(equil, 'init', inputs)

        # continuing from previous solution
        else:
            # change grids
            if Mnodes[ii] != Mnodes[ii-1] or Nnodes[ii] != Nnodes[ii-1]:
                RZ_grid = ConcentricGrid(Mnodes[ii], Nnodes[ii], NFP=NFP, sym=stell_sym,
                                         axis=False, index=zern_mode, surfs=node_mode)
                L_grid = LinearGrid(M=Mnodes[ii], N=2*Nnodes[ii]+1, NFP=NFP, sym=stell_sym)

            # change bases
            if M[ii] != M[ii-1] or N[ii] != N[ii-1] or delta_lm[ii] != delta_lm[ii-1]:
                R_basis = FourierZernikeBasis(L=delta_lm[ii], M=M[ii], N=N[ii],
                                              NFP=NFP, sym=R_sym, index=zern_mode)
                Z_basis = FourierZernikeBasis(L=delta_lm[ii], M=M[ii], N=N[ii],
                                              NFP=NFP, sym=Z_sym, index=zern_mode)
                L_basis = DoubleFourierSeries(M=M[ii], N=N[ii], NFP=NFP, sym=L_sym)

                equil.change_resolution(L=delta_lm[ii], M=M[ii], N=N[ii])
                x = equil.x

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
                    errr_mode, scalar=False,
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
                x, timer = perturb_continuation_params(x, equil_obj, deltas, args,
                                                       pert_order[ii], verbose, timer)

        # equilibrium objective function
        if optim_method in ['bfgs']:
            scalar = True
        else:
            scalar = False
        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
                    errr_mode, scalar=scalar,
                    R_transform=R_transform, Z_transform=Z_transform,
                    R1_transform=R1_transform, Z1_transform=Z1_transform,
                    L_transform=L_transform, P_transform=P_transform,
                    I_transform=I_transform)
        equil_obj = obj_fun.compute
        callback = obj_fun.callback
        args = (equil.cRb, equil.cZb, equil.cP, equil.cI, equil.Psi,
                bdry_ratio[ii-1], pres_ratio[ii-1], zeta_ratio[ii-1], errr_ratio[ii-1])

        if use_jax:
            if optim_method in ['bfgs']:
                jac = grad(equil_obj, argnums=0)
            else:
                jac = jacfwd(equil_obj, argnums=0)
            if verbose > 0:
                print("Compiling objective function")
            if device is None:
                import jax
                device = jax.devices()[0]
            equil_obj_jit = jit(equil_obj, static_argnums=(), device=device)
            jac_obj_jit = jit(jac, device=device)
            timer.start("Iteration {} compilation".format(ii+1))
            f0 = equil_obj_jit(x, *args)
            J0 = jac_obj_jit(x, *args)
            timer.stop("Iteration {} compilation".format(ii+1))
            if verbose > 1:
                timer.disp("Iteration {} compilation".format(ii+1))
        else:
            equil_obj_jit = equil_obj
            jac_obj_jit = '2-point'
        if verbose > 0:
            print("Starting optimization")

        x_init = x
        timer.start("Iteration {} solution".format(ii+1))
        if optim_method in ['bfgs']:
            out = scipy.optimize.minimize(equil_obj_jit,
                                          x0=x_init,
                                          args=args,
                                          method=optim_method,
                                          jac=jac_obj_jit,
                                          tol=gtol[ii],
                                          options={'maxiter': nfev[ii],
                                                   'disp': verbose})

        elif optim_method in ['trf', 'lm', 'dogleg']:
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
                TextColors.FAIL + "optim_method must be one of 'bfgs', 'trf', 'lm', 'dogleg'" + TextColors.ENDC)

        timer.stop("Iteration {} solution".format(ii+1))

        equil.x = out['x']
        equil_fam.append(copy.deepcopy(equil))#insert(ii, copy.deepcopy(equil))

        if verbose > 1:
            timer.disp("Iteration {} solution".format(ii+1))
            timer.pretty_print("Iteration {} avg time per step".format(ii+1),
                               timer["Iteration {} solution".format(ii+1)]/out['nfev'])
        if verbose > 0:
            print("Start of Step {}:".format(ii+1))
            callback(x_init, *args)
            print("End of Step {}:".format(ii+1))
            callback(x, *args)

#        if checkpoint:
#            if verbose > 0:
#                print('Saving latest iteration')
#            checkpoint_file.write_iteration(equil, ii+1, inputs)

        if not is_nested(equil.cR, equil.cZ, equil.R_basis, equil.Z_basis):
            warnings.warn(TextColors.WARNING + 'WARNING: Flux surfaces are no longer nested, exiting early.'
                          + 'Consider increasing errr_ratio or taking smaller perturbation steps' + TextColors.ENDC)
            break

#    if checkpoint:
#        checkpoint_file.close()

    timer.stop("Total time")
    print('====================')
    print('Done')
    if verbose > 1:
        timer.disp("Total time")
    if checkpoint_filename is not None:
        print('Output written to {}'.format(checkpoint_filename))
    print('====================')

    return equil_fam, timer
