import numpy as np
import scipy.optimize
import warnings

from desc.backend import jnp, jit, use_jax, Timer, TextColors
from desc.backend import jacfwd, grad
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry
from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import PowerSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.transform import Transform
from desc.configuration import unpack_state, symmetry_matrix, change_resolution
from desc.objective_funs import is_nested, ObjectiveFunctionFactory
from desc.input_output import Checkpoint
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
    iterations = {}

    if not use_jax:
        pert_order *= 0

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
            # interpolator
            timer.start("Iteration {} total".format(ii+1))
            timer.start("Transform precomputation")
            if verbose > 0:
                print("Precomputing Transforms")
            RZ_grid = ConcentricGrid(Mnodes[ii], Nnodes[ii], NFP=NFP, sym=stell_sym,
                                     axis=True, index=zern_mode, surfs=node_mode)
            # FIXME: hard-coded non-symmetric L_grid until symmetry is implemented in Basis
            L_grid = LinearGrid(M=Mnodes[ii], N=2*Nnodes[ii]+1, NFP=NFP, sym=False)
            RZ_basis = FourierZernikeBasis(L=delta_lm[ii], M=M[ii], N=N[ii],
                                           NFP=NFP, index=zern_mode)
            L_basis = DoubleFourierSeries(M=M[ii], N=N[ii], NFP=NFP)
            pres_basis = PowerSeries(L=cP.size-1)
            iota_basis = PowerSeries(L=cI.size-1)
            RZ_transform = Transform(RZ_grid, RZ_basis, derivs=3)
            RZb_transform = Transform(L_grid, RZ_basis)
            L_transform = Transform(L_grid, L_basis, derivs=0)
            pres_transform = Transform(RZ_grid, pres_basis, derivs=1)
            iota_transform = Transform(RZ_grid, iota_basis, derivs=1)
            timer.stop("Transform precomputation")
            if verbose > 1:
                timer.disp("Transform precomputation")
            # format boundary shape
            cRb, cZb = format_bdry(bdry, L_basis, bdry_mode)

            # initial guess
            timer.start("Initial guess computation")
            if verbose > 0:
                print("Computing initial guess")
            cR, cZ = get_initial_guess_scale_bdry(axis, bdry, bdry_ratio[ii], RZ_basis)
            cL = np.zeros((L_basis.num_modes,))
            x = jnp.concatenate([cR, cZ, cL])
            sym_mat = symmetry_matrix(RZ_basis.modes, L_basis.modes, sym=stell_sym)
            x = jnp.matmul(sym_mat.T, x)
            timer.stop("Initial guess computation")
            if verbose > 1:
                timer.disp("Initial guess computation")
            equil_init = {
                'M': M[ii],
                'N': N[ii],
                'cR': cR,
                'cZ': cZ,
                'cL': cL,
                'bdryR': bdry[:, 2]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
                'bdryZ': bdry[:, 3]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
                'cP': cP*pres_ratio[ii],
                'cI': cI,
                'Psi_lcfs': Psi_lcfs,
                'NFP': NFP,
                'RZ_basis': RZ_basis,
                'L_basis': L_basis,
                'bdry_idx': bdry[:, :2]
            }
            iterations['init'] = equil_init
            if checkpoint:
                checkpoint_file.write_iteration(equil_init, 'init', inputs)

            # equilibrium objective function
            obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(errr_mode,
                RZ_transform=RZ_transform, RZb_transform=RZb_transform,
                L_transform=L_transform, pres_transform=pres_transform,
                iota_transform=iota_transform, stell_sym=stell_sym, scalar=False)
            equil_obj = obj_fun.compute
            callback = obj_fun.callback
            args = [cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio[ii],
                    pres_ratio[ii], zeta_ratio[ii], errr_ratio[ii]]

        # continuing from previous solution
        else:
            # change grids
            if Mnodes[ii] != Mnodes[ii-1] or Nnodes[ii] != Nnodes[ii-1]:
                RZ_grid = ConcentricGrid(Mnodes[ii], Nnodes[ii], NFP=NFP, sym=stell_sym,
                                         axis=True, index=zern_mode, surfs=node_mode)
                # FIXME: hard-coded non-symmetric L_grid until symmetry is implemented in Basis
                L_grid = LinearGrid(M=Mnodes[ii], N=2*Nnodes[ii]+1, NFP=NFP, sym=False)

            # change bases
            if M[ii] != M[ii-1] or N[ii] != N[ii-1] or delta_lm[ii] != delta_lm[ii-1]:
                RZ_basis_old = RZ_basis
                L_basis_old = L_basis
                RZ_basis = FourierZernikeBasis(L=delta_lm[ii], M=M[ii], N=N[ii],
                                               NFP=NFP, index=zern_mode)
                L_basis = DoubleFourierSeries(M=M[ii], N=N[ii], NFP=NFP)

                # re-format boundary shape
                cRb, cZb = format_bdry(bdry, L_basis, bdry_mode)
                # update state vector
                sym_mat = symmetry_matrix(RZ_basis.modes, L_basis.modes, sym=stell_sym)
                x = change_resolution(x, stell_sym, RZ_basis_old, RZ_basis, L_basis_old, L_basis)

            # change transform matrices
            timer.start(
                "Iteration {} changing resolution".format(ii+1))
            if verbose > 0:
                print("Changing node resolution from (Mnodes,Nnodes) = ({},{}) to ({},{})".format(
                    Mnodes[ii-1], Nnodes[ii-1], Mnodes[ii], Nnodes[ii]))
                print("Changing spectral resolution from (L,M,N) = ({},{},{}) to ({},{},{})".format(
                        delta_lm[ii-1], M[ii-1], N[ii-1], delta_lm[ii], M[ii], N[ii]))
            RZ_transform.change_resolution(grid=RZ_grid, basis=RZ_basis)
            RZb_transform.change_resolution(grid=L_grid, basis=RZ_basis)
            L_transform.change_resolution(grid=L_grid, basis=L_basis)
            pres_transform.change_resolution(grid=RZ_grid)
            iota_transform.change_resolution(grid=RZ_grid)
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

            # equilibrium objective function
            obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(errr_mode,
                RZ_transform=RZ_transform, RZb_transform=RZb_transform,
                L_transform=L_transform, pres_transform=pres_transform,
                iota_transform=iota_transform, stell_sym=stell_sym, scalar=False)
            equil_obj = obj_fun.compute
            callback = obj_fun.callback
            args = [cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio[ii-1],
                    pres_ratio[ii-1], zeta_ratio[ii-1], errr_ratio[ii-1]]

            # perturbations
            if np.any(deltas):
                if verbose > 1:
                    print("Perturbing equilibrium")
                x, timer = perturb_continuation_params(x, equil_obj, deltas, args,
                                                       pert_order[ii], verbose, timer)

        # equilibrium objective function
        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(errr_mode,
                RZ_transform=RZ_transform, RZb_transform=RZb_transform,
                L_transform=L_transform, pres_transform=pres_transform,
                iota_transform=iota_transform, stell_sym=stell_sym, scalar=False)
        equil_obj = obj_fun.compute
        callback = obj_fun.callback
        args = (cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio[ii],
                pres_ratio[ii], zeta_ratio[ii], errr_ratio[ii])

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
        x = out['x']

        if verbose > 1:
            timer.disp("Iteration {} solution".format(ii+1))
            timer.pretty_print("Iteration {} avg time per step".format(ii+1),
                               timer["Iteration {} solution".format(ii+1)]/out['nfev'])
        if verbose > 0:
            print("Start of Step {}:".format(ii+1))
            callback(x_init, *args)
            print("End of Step {}:".format(ii+1))
            callback(x, *args)

        cR, cZ, cL = unpack_state(jnp.matmul(sym_mat, x), RZ_transform.num_modes)
        equil = {
            'M': M[ii],
            'N': N[ii],
            'cR': cR,
            'cZ': cZ,
            'cL': cL,
            'bdryR': bdry[:, 2]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
            'bdryZ': bdry[:, 3]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
            'cP': cP*pres_ratio[ii],
            'cI': cI,
            'Psi_lcfs': Psi_lcfs,
            'NFP': NFP,
            'RZ_basis': RZ_basis,
            'L_basis': L_basis,
            'bdry_idx': bdry[:, :2]
        }

        iterations[ii] = equil
        iterations['final'] = equil
        if checkpoint:
            if verbose > 0:
                print('Saving latest iteration')
            checkpoint_file.write_iteration(equil, ii+1, inputs)

        if not is_nested(cR, cZ, RZ_basis):
            warnings.warn(TextColors.WARNING + 'WARNING: Flux surfaces are no longer nested, exiting early.'
                          + 'Consider increasing errr_ratio or taking smaller perturbation steps' + TextColors.ENDC)
            break

    if checkpoint:
        checkpoint_file.close()

    timer.stop("Total time")
    print('====================')
    print('Done')
    if verbose > 1:
        timer.disp("Total time")
    if checkpoint_filename is not None:
        print('Output written to {}'.format(checkpoint_filename))
    print('====================')

    return iterations, timer
