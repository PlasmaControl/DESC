import numpy as np
import scipy.optimize
import time

from desc.backend import jnp, jit, use_jax
from desc.backend import get_needed_derivatives, unpack_x, jacfwd, grad
from desc.zernike import ZernikeTransform, get_zern_basis_idx_dense
from desc.zernike import get_double_four_basis_idx_dense, symmetric_x
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry
from desc.objective_funs import get_equil_obj_fun, is_nested
from desc.nodes import get_nodes_pattern, get_nodes_surf
from desc.input_output import Checkpoint
from desc.perturbations import perturb_continuation_params


def expand_resolution(x, zernt, bdry_zernt, zern_idx_old, zern_idx_new,
                      lambda_idx_old, lambda_idx_new):
    """Expands solution to a higher resolution by filling with zeros
    Also modifies zernike transform object to work at higher resolution

    Args:
        x (ndarray): solution at initial resolution
        zernt (ZernikeTransform): zernike transform object corresponding to initial x
        bdry_zernt (ZernikeTransform): zernike transform object corresponding to initial x at bdry
        zern_idx_old (ndarray of int, size(nRZ_old,3)): mode indices corresponding to initial R,Z
        zern_idx_new (ndarray of int, size(nRZ_new,3)): mode indices corresponding to new R,Z
        lambda_idx_old (ndarray of int, size(nL_old,2)): mode indices corresponding to initial lambda
        lambda_idx_new (ndarray of int, size(nL_new,2)): mode indices corresponding to new lambda
    Returns:
        x_new (ndarray): solution expanded to new resolution
        zernt (ZernikeTransform): zernike transform object corresponding to expanded x
        bdry_zernt (ZernikeTransform): zernike transform object corresponding to expanded x at bdry
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

    zernt.expand_spectral_resolution(zern_idx_new)
    bdry_zernt.expand_spectral_resolution(zern_idx_new)

    return x_new, zernt, bdry_zernt


def solve_eq_continuation(inputs, checkpoint_filename=None, device=None):
    """Solves for an equilibrium by continuation method

    Steps up resolution, perturbs pressure, 3d bdry etc.

    Args:
        inputs (dict): dictionary with input parameters defining problem setup and solver options
        checkpoint_filename (str or path-like): file to save checkpoint data
        device (JAX device or None): device handle to JIT compile to

    Returns:
        iterations (dict): dictionary of intermediate solutions
    """
    t_start = time.perf_counter()

    stell_sym = inputs['stell_sym']
    NFP = inputs['NFP']
    Psi_lcfs = inputs['Psi_lcfs']
    M = inputs['Mpol']                  # arr
    N = inputs['Ntor']                  # arr
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
            print("Spectral resolution (M,N)=({},{})".format(M[ii], N[ii]))
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
            t0 = time.perf_counter()
            if verbose > 0:
                print("Precomputing Fourier-Zernike basis")
            nodes, volumes = get_nodes_pattern(
                Mnodes[ii], Nnodes[ii], NFP, index=zern_mode, surfs=node_mode, sym=stell_sym, axis=False)
            derivatives = get_needed_derivatives('all')
            zern_idx = get_zern_basis_idx_dense(M[ii], N[ii], zern_mode)
            lambda_idx = get_double_four_basis_idx_dense(M[ii], N[ii])
            zernt = ZernikeTransform(
                nodes, zern_idx, NFP, derivatives, volumes, method='fft')
            # bdry interpolator
            bdry_nodes, _ = get_nodes_surf(
                Mnodes[ii], Nnodes[ii], NFP, surf=1.0, sym=stell_sym)
            bdry_zernt = ZernikeTransform(bdry_nodes, zern_idx, NFP, [
                                          0, 0, 0], method='direct')
            t1 = time.perf_counter()
            if verbose > 1:
                print("Precomputation time = {} s".format(t1-t0))
            # format boundary shape
            bdry_pol, bdry_tor, bdryR, bdryZ = format_bdry(
                M[ii], N[ii], NFP, bdry, bdry_mode, bdry_mode)

            # stellarator symmetry
            if stell_sym:
                sym_mat = symmetric_x(zern_idx, lambda_idx)
            else:
                sym_mat = np.eye(2*zern_idx.shape[0] + lambda_idx.shape[0])

            # initial guess
            t0 = time.perf_counter()
            if verbose > 0:
                print("Computing initial guess")
            cR_init, cZ_init = get_initial_guess_scale_bdry(
                axis, bdry, bdry_ratio[ii], zern_idx, NFP, mode=bdry_mode, rcond=1e-6)
            cL_init = np.zeros(len(lambda_idx))
            x_init = jnp.concatenate([cR_init, cZ_init, cL_init])
            x_init = jnp.matmul(sym_mat.T, x_init)
            x = x_init
            t1 = time.perf_counter()
            if verbose > 1:
                print("Initial guess time = {} s".format(t1-t0))
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
            equil_obj, callback = get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
                                                    NFP, zernt, bdry_zernt, zern_idx, lambda_idx, bdry_pol, bdry_tor)
            args = [bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii],
                    pres_ratio[ii], zeta_ratio[ii], errr_ratio[ii]]

        # continuing from prev soln
        else:
            # collocation nodes
            if Mnodes[ii] != Mnodes[ii-1] or Nnodes[ii] != Nnodes[ii-1]:
                t0 = time.perf_counter()
                if verbose > 0:
                    print("Changing node resolution from (Mnodes,Nnodes) = ({},{}) to ({},{})".format(
                        Mnodes[ii-1], Nnodes[ii-1], Mnodes[ii], Nnodes[ii]))
                nodes, volumes = get_nodes_pattern(
                    Mnodes[ii], Nnodes[ii], NFP, index=zern_mode, surfs=node_mode, sym=stell_sym, axis=False)
                bdry_nodes, _ = get_nodes_surf(
                    Mnodes[ii], Nnodes[ii], NFP, surf=1.0, sym=stell_sym)
                zernt.expand_nodes(nodes, volumes)
                bdry_zernt.expand_nodes(bdry_nodes)
                t1 = time.perf_counter()
                if verbose > 1:
                    print("Changing node resolution time = {} s".format(t1-t0))

            # spectral resolution
            if M[ii] != M[ii-1] or N[ii] != N[ii-1]:
                t0 = time.perf_counter()
                if verbose > 0:
                    print("Changing spectral resolution from (M,N) = ({},{}) to ({},{})".format(
                        M[ii-1], N[ii-1], M[ii], N[ii]))
                zern_idx_old = zern_idx
                lambda_idx_old = lambda_idx
                zern_idx = get_zern_basis_idx_dense(M[ii], N[ii], zern_mode)
                lambda_idx = get_double_four_basis_idx_dense(M[ii], N[ii])
                bdry_pol, bdry_tor, bdryR, bdryZ = format_bdry(
                    M[ii], N[ii], NFP, bdry, bdry_mode, bdry_mode)

                x, zernt, bdry_zernt = expand_resolution(jnp.matmul(sym_mat, x), zernt, bdry_zernt,
                                                         zern_idx_old, zern_idx, lambda_idx_old, lambda_idx)
                # stellarator symmetry
                if stell_sym:
                    sym_mat = symmetric_x(zern_idx, lambda_idx)
                else:
                    sym_mat = np.eye(2*zern_idx.shape[0] + lambda_idx.shape[0])
                x = jnp.matmul(sym_mat.T, x)
                t1 = time.perf_counter()
                if verbose > 1:
                    print("Changing spectral resolution time = {} s".format(t1-t0))

            # continuation parameters
            delta_bdry = bdry_ratio[ii] - bdry_ratio[ii-1]
            delta_pres = pres_ratio[ii] - pres_ratio[ii-1]
            delta_zeta = zeta_ratio[ii] - zeta_ratio[ii-1]
            deltas = np.array([delta_bdry, delta_pres, delta_zeta])

            # equilibrium objective function
            equil_obj, callback = get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
                                                    NFP, zernt, bdry_zernt, zern_idx, lambda_idx, bdry_pol, bdry_tor)
            args = [bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii-1],
                    pres_ratio[ii-1], zeta_ratio[ii-1], errr_ratio[ii-1]]

            if np.any(deltas):
                if verbose > 1:
                    print("Perturbing equilibrium")
                x = perturb_continuation_params(x, equil_obj, deltas, args,
                                                pert_order[ii], verbose)

        args = (bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii],
                pres_ratio[ii], zeta_ratio[ii], errr_ratio[ii])

        if optim_method in ['bfgs']:
            equil_obj, callback = get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
                                                    NFP, zernt, bdry_zernt, zern_idx, lambda_idx, bdry_pol, bdry_tor, scalar=True)
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
            t0 = time.perf_counter()
            f0 = equil_obj_jit(x, *args)
            J0 = jac_obj_jit(x, *args)
            t1 = time.perf_counter()
            if verbose > 1:
                print("Objective function compiled, time = {} s".format(t1-t0))
        else:
            equil_obj_jit = equil_obj
            jac_obj_jit = '2-point'
        if verbose > 0:
            print("Starting optimization")

        x_init = x
        t0 = time.perf_counter()
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
            raise NotImplementedError

        t1 = time.perf_counter()
        x = out['x']

        if verbose > 1:
            print('Step {} time = {} s'.format(ii+1, t1-t0))
            print("Avg time per step: {} s".format((t1-t0)/out['nfev']))
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
            print('WARNING: Flux surfaces are no longer nested, exiting early.'
                  + 'Consider increasing errr_ratio or taking smaller perturbation steps')
            break

    if checkpoint:
        checkpoint_file.close()

    t_end = time.perf_counter()
    print('====================')
    print('Done')
    if verbose > 1:
        print('total time = {} s'.format(t_end-t_start))
    if checkpoint_filename is not None:
        print('Output written to {}'.format(checkpoint_filename))
    print('====================')

    return iterations
