import numpy as np
import scipy.optimize
import warnings
from desc.backend import jnp, jit, use_jax, Timer, TextColors
from desc.backend import get_needed_derivatives, unpack_x, jacfwd, grad
from desc.zernike import ZernikeTransform, get_zern_basis_idx_dense
from desc.zernike import get_double_four_basis_idx_dense, symmetric_x
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry
from desc.objective_funs import  is_nested, obj_fun_factory
from desc.nodes import get_nodes_pattern, get_nodes_surf
from desc.input_output import Checkpoint
from desc.perturbations import perturb_continuation_params


def expand_resolution(x, zernike_transform, bdry_zernike_transform, zern_idx_old, zern_idx_new,
                      lambda_idx_old, lambda_idx_new):
    """Expands solution to a higher resolution by filling with zeros
    Also modifies zernike transform object to work at higher resolution

    Parameters
    ----------
    x : ndarray
        solution at initial resolution
    zernike_transform : ZernikeTransform
        zernike transform object corresponding to initial x
    bdry_zernike_transform : ZernikeTransform
        zernike transform object corresponding to initial x at bdry
    zern_idx_old : ndarray of int, shape(nRZ_old,3)
        mode indices corresponding to initial R,Z
    zern_idx_new : ndarray of int, shape(nRZ_new,3)
        mode indices corresponding to new R,Z
    lambda_idx_old : ndarray of int, shape(nL_old,2)
        mode indices corresponding to initial lambda
    lambda_idx_new : ndarray of int, shape(nL_new,2)
        mode indices corresponding to new lambda

    Returns
    -------
    x_new : ndarray
        solution expanded to new resolution
    zernike_transform : ZernikeTransform
        zernike transform object corresponding to expanded x
    bdry_zernike_transform : ZernikeTransform
        zernike transform object corresponding to expanded x at bdry

    """

    cR, cZ, cL = unpack_x(x, len(zern_idx_old))
    cR_new = np.zeros(len(zern_idx_new))
    cZ_new = np.zeros(len(zern_idx_new))
    cL_new = np.zeros(len(lambda_idx_new))
    old_in_new = np.where((zern_idx_new[:, None] == zern_idx_old).all(-1))[0]
    cR_new[old_in_new] = cR
    cZ_new[old_in_new] = cZ
    old_in_new = np.where(
        (lambda_idx_new[:, None] == lambda_idx_old).all(-1))[0]
    cL_new[old_in_new] = cL
    x_new = np.concatenate([cR_new, cZ_new, cL_new])

    zernike_transform.expand_spectral_resolution(zern_idx_new)
    bdry_zernike_transform.expand_spectral_resolution(zern_idx_new)

    return x_new, zernike_transform, bdry_zernike_transform


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
            timer.start("Fourier-Zernike precomputation")
            if verbose > 0:
                print("Precomputing Fourier-Zernike basis")
            nodes, volumes = get_nodes_pattern(
                Mnodes[ii], Nnodes[ii], NFP, index=zern_mode, surfs=node_mode, sym=stell_sym, axis=False)
            derivatives = get_needed_derivatives('all')
            zern_idx = get_zern_basis_idx_dense(
                M[ii], N[ii], delta_lm[ii], zern_mode)
            lambda_idx = get_double_four_basis_idx_dense(M[ii], N[ii])
            zernike_transform = ZernikeTransform(
                nodes, zern_idx, NFP, derivatives, volumes, method='fft')
            # bdry interpolator
            bdry_nodes, _ = get_nodes_surf(
                Mnodes[ii], Nnodes[ii], NFP, surf=1.0, sym=stell_sym)
            bdry_zernike_transform = ZernikeTransform(bdry_nodes, zern_idx, NFP, [
                0, 0, 0], method='direct')
            timer.stop("Fourier-Zernike precomputation")
            if verbose > 1:
                timer.disp("Fourier-Zernike precomputation")
            # format boundary shape
            bdry_pol, bdry_tor, bdryR, bdryZ = format_bdry(
                M[ii], N[ii], NFP, bdry, bdry_mode, bdry_mode)

            # stellarator symmetry
            if stell_sym:
                sym_mat = symmetric_x(zern_idx, lambda_idx)
            else:
                sym_mat = np.eye(2*zern_idx.shape[0] + lambda_idx.shape[0])

            # initial guess
            timer.start("Initial guess computation")
            if verbose > 0:
                print("Computing initial guess")
            cR_init, cZ_init = get_initial_guess_scale_bdry(
                axis, bdry, bdry_ratio[ii], zern_idx, NFP, mode=bdry_mode, rcond=1e-6)
            cL_init = np.zeros(len(lambda_idx))
            x_init = jnp.concatenate([cR_init, cZ_init, cL_init])
            x_init = jnp.matmul(sym_mat.T, x_init)
            x = x_init
            timer.stop("Initial guess computation")
            if verbose > 1:
                timer.disp("Initial guess computation")
            equil_init = {
                'cR': cR_init,
                'cZ': cZ_init,
                'cL': cL_init,
                'bdryR': bdry[:, 2]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
                'bdryZ': bdry[:, 3]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
                'cP': cP*pres_ratio[ii],
                'cI': cI,
                'Psi_lcfs': Psi_lcfs,
                'NFP': NFP,
                'zern_idx': zern_idx,
                'lambda_idx': lambda_idx,
                'bdry_idx': bdry[:, :2]
            }
            iterations['init'] = equil_init
            if checkpoint:
                checkpoint_file.write_iteration(equil_init, 'init', inputs)

            # equilibrium objective function
            obj_fun = obj_fun_factory.get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
                                                    NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                                    bdry_pol, bdry_tor)
            equil_obj = obj_fun.compute
            callback = obj_fun.callback
            args = [bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii],
                    pres_ratio[ii], zeta_ratio[ii], errr_ratio[ii]]

        # continuing from prev soln
        else:
            # collocation nodes
            if Mnodes[ii] != Mnodes[ii-1] or Nnodes[ii] != Nnodes[ii-1]:
                timer.start(
                    "Iteration {} changing node resolution".format(ii+1))
                if verbose > 0:
                    print("Changing node resolution from (Mnodes,Nnodes) = ({},{}) to ({},{})".format(
                        Mnodes[ii-1], Nnodes[ii-1], Mnodes[ii], Nnodes[ii]))
                nodes, volumes = get_nodes_pattern(
                    Mnodes[ii], Nnodes[ii], NFP, index=zern_mode, surfs=node_mode, sym=stell_sym, axis=False)
                bdry_nodes, _ = get_nodes_surf(
                    Mnodes[ii], Nnodes[ii], NFP, surf=1.0, sym=stell_sym)
                zernike_transform.expand_nodes(nodes, volumes)
                bdry_zernike_transform.expand_nodes(bdry_nodes)
                timer.stop(
                    "Iteration {} changing node resolution".format(ii+1))
                if verbose > 1:
                    timer.disp(
                        "Iteration {} changing node resolution".format(ii+1))

            # spectral resolution
            if M[ii] != M[ii-1] or N[ii] != N[ii-1] or delta_lm[ii] != delta_lm[ii-1]:
                timer.start(
                    "Iteration {} changing spectral resolution".format(ii+1))
                if verbose > 0:
                    print("Changing spectral resolution from (M,N,delta_lm) = ({},{},{}) to ({},{},{})".format(
                        M[ii-1], N[ii-1], delta_lm[ii-1], M[ii], N[ii], delta_lm[ii]))
                zern_idx_old = zern_idx
                lambda_idx_old = lambda_idx
                zern_idx = get_zern_basis_idx_dense(
                    M[ii], N[ii], delta_lm[ii], zern_mode)
                lambda_idx = get_double_four_basis_idx_dense(M[ii], N[ii])
                bdry_pol, bdry_tor, bdryR, bdryZ = format_bdry(
                    M[ii], N[ii], NFP, bdry, bdry_mode, bdry_mode)

                x, zernike_transform, bdry_zernike_transform = expand_resolution(jnp.matmul(sym_mat, x), zernike_transform, bdry_zernike_transform,
                                                                                 zern_idx_old, zern_idx, lambda_idx_old, lambda_idx)
                # stellarator symmetry
                if stell_sym:
                    sym_mat = symmetric_x(zern_idx, lambda_idx)
                else:
                    sym_mat = np.eye(2*zern_idx.shape[0] + lambda_idx.shape[0])
                x = jnp.matmul(sym_mat.T, x)
                timer.stop(
                    "Iteration {} changing spectral resolution".format(ii+1))
                if verbose > 1:
                    timer.disp(
                        "Iteration {} changing spectral resolution".format(ii+1))

            # continuation parameters
            delta_bdry = bdry_ratio[ii] - bdry_ratio[ii-1]
            delta_pres = pres_ratio[ii] - pres_ratio[ii-1]
            delta_zeta = zeta_ratio[ii] - zeta_ratio[ii-1]
            deltas = np.array([delta_bdry, delta_pres, delta_zeta])

            # equilibrium objective function
            obj_fun = obj_fun_factory.get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
                                                    NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                                    bdry_pol, bdry_tor)
            equil_obj = obj_fun.compute
            callback = obj_fun.callback
            args = [bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii-1],
                    pres_ratio[ii-1], zeta_ratio[ii-1], errr_ratio[ii-1]]

            if np.any(deltas):
                if verbose > 1:
                    print("Perturbing equilibrium")
                x, timer = perturb_continuation_params(x, equil_obj, deltas, args,
                                                       pert_order[ii], verbose, timer)

        args = (bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii],
                pres_ratio[ii], zeta_ratio[ii], errr_ratio[ii])

        if optim_method in ['bfgs']:
            # equil_obj, callback = get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
            #                                         NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
            #                                         bdry_pol, bdry_tor, scalar=True)
            obj_fun = obj_fun_factory.get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
                                                    NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                                    bdry_pol, bdry_tor,scalar=True)
            equil_obj = obj_fun.compute
            callback = obj_fun.callback
            jac = grad(equil_obj, argnums=0)
        else:
            jac = jacfwd(equil_obj, argnums=0)

        if use_jax:
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

        cR, cZ, cL = unpack_x(np.matmul(sym_mat, x), len(zern_idx))
        equil = {
            'cR': cR,
            'cZ': cZ,
            'cL': cL,
            'bdryR': bdry[:, 2]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
            'bdryZ': bdry[:, 3]*np.where((bdry[:, 1] == 0), bdry_ratio[ii], 1),
            'cP': cP*pres_ratio[ii],
            'cI': cI,
            'Psi_lcfs': Psi_lcfs,
            'NFP': NFP,
            'zern_idx': zern_idx,
            'lambda_idx': lambda_idx,
            'bdry_idx': bdry[:, :2]
        }

        iterations[ii] = equil
        iterations['final'] = equil
        if checkpoint:
            if verbose > 0:
                print('Saving latest iteration')
            checkpoint_file.write_iteration(equil, ii+1, inputs)

        if not is_nested(cR, cZ, zern_idx, NFP):
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
