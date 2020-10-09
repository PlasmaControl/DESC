import numpy as np
import scipy.optimize
import time
from functools import reduce

from desc.backend import jnp, jit, use_jax
from desc.backend import get_needed_derivatives, unpack_x, jacfwd
from desc.zernike import ZernikeTransform, get_zern_basis_idx_dense
from desc.zernike import get_double_four_basis_idx_dense, symmetric_x
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry
from desc.objective_funs import get_equil_obj_fun
from desc.nodes import get_nodes_pattern, get_nodes_surf
from desc.input_output import Checkpoint


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


def perturb(x, equil_obj, deltas, args, pert_order, verbose):
    """perturbs an equilibrium"""

    delta_strings = ['boundary', 'pressure', 'zeta', 'error']
    dimF = len(equil_obj(x, *args))
    dimX = len(x)
    dimC = 1

    # x
    t00 = time.perf_counter()
    if pert_order >= 1:
        t0 = time.perf_counter()
        obj_jac_x = jacfwd(equil_obj, argnums=0)
        Jx = obj_jac_x(x, *args).reshape((dimF, dimX))
        t1 = time.perf_counter()
        LHS = Jx
        RHS = equil_obj(x, *args)
        if verbose > 1:
            print("dF/dx computation time: {} s".format(t1-t0))
    if pert_order >= 2:
        t0 = time.perf_counter()
        obj_jac_xx = jacfwd(jacfwd(equil_obj, argnums=0), argnums=0)
        Jxx = obj_jac_xx(x, *args).reshape((dimF, dimX, dimX))
        t1 = time.perf_counter()
        if verbose > 1:
            print("dF/dxx computation time: {} s".format(t1-t0))

    # continuation parameters
    for i in range(deltas.size):
        if deltas[i] != 0:
            if verbose > 1:
                print("Perturbing {}".format(delta_strings[i]))
            if pert_order >= 1:
                t0 = time.perf_counter()
                obj_jac_c = jacfwd(equil_obj, argnums=6+i)
                Jc = obj_jac_c(x, *args).reshape((dimF, dimC))
                t1 = time.perf_counter()
                RHS += np.tensordot(Jc, np.atleast_1d(deltas[i]), axes=1)
                if verbose > 1:
                    print("dF/dc computation time: {} s".format(t1-t0))
            if pert_order >= 2:
                t0 = time.perf_counter()
                obj_jac_cc = jacfwd(
                    jacfwd(equil_obj, argnums=6+i), argnums=6+i)
                Jcc = obj_jac_cc(x, *args).reshape((dimF, dimC, dimC))
                t1 = time.perf_counter()
                if verbose > 1:
                    print("dF/dcc computation time: {} s".format(t1-t0))
                obj_jac_xc = jacfwd(jacfwd(equil_obj, argnums=0), argnums=6+i)
                Jxc = obj_jac_xc(x, *args).reshape((dimF, dimX, dimC))
                t2 = time.perf_counter()
                LHS += np.tensordot(Jxc, np.atleast_1d(deltas[i]), axes=1)
                RHS += 0.5 * np.tensordot(Jcc, np.tensordot(np.atleast_1d(deltas[i]),
                                                            np.atleast_1d(deltas[i]), axes=0), axes=2)
                if verbose > 1:
                    print("dF/dxc computation time: {} s".format(t2-t1))

    # perturbation
    if pert_order >= 2:
        LHSi = np.linalg.pinv(LHS, rcond=1e-6)
        RHS += 0.5 * np.tensordot(Jxx, np.tensordot(np.tensordot(LHSi, RHS, axes=1),
                                                    np.tensordot(RHS.T, LHSi.T, axes=1), axes=0), axes=2)
    if pert_order > 0:
        dx = -np.linalg.lstsq(LHS, RHS, rcond=1e-6)[0]
    else:
        dx = np.zeros_like(x)
    t1 = time.perf_counter()
    if verbose > 1:
        print("Total perturbation time: {} s".format(t1-t00))
    return x + dx


def solve_eq_continuation(inputs, checkpoint_filename=None):
    """Solves for an equilibrium by continuation method

    Steps up resolution, perturbs pressure, 3d bdry etc.

    Args:
        inputs (dict): dictionary with input parameters defining problem setup and solver options
        checkpoint_filename (str or path-like): file to save checkpoint data

    Returns:
        equil (dict): dictionary of solution values
        iterations (dict): dictionary of intermediate solutions
    """

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
                Mnodes[ii], Nnodes[ii], NFP, surfs=node_mode)
            derivatives = get_needed_derivatives('all')
            zern_idx = get_zern_basis_idx_dense(M[ii], N[ii], zern_mode)
            lambda_idx = get_double_four_basis_idx_dense(M[ii], N[ii])
            zernt = ZernikeTransform(
                nodes, zern_idx, NFP, derivatives, volumes, method='fft')
            # bdry interpolator
            bdry_nodes, _ = get_nodes_surf(
                Mnodes[ii], Nnodes[ii], NFP, surf=1.0)
            bdry_zernt = ZernikeTransform(bdry_nodes, zern_idx, NFP, [
                                          0, 0, 0], method='direct')
            t1 = time.perf_counter()
            if verbose > 0:
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
                axis, bdry, zern_idx, NFP, mode=bdry_mode, rcond=1e-6)
            cL_init = np.zeros(len(lambda_idx))
            x_init = jnp.concatenate([cR_init, cZ_init, cL_init])
            x_init = jnp.matmul(sym_mat.T, x_init)
            x = x_init
            t1 = time.perf_counter()
            if verbose > 0:
                print("Initial guess time = {} s".format(t1-t0))
            equil_init = {
                'cR': cR_init,
                'cZ': cZ_init,
                'cL': cL_init,
                'bdryR': bdry[:, 2]*np.clip(bdry_ratio[ii]+(bdry[:, 1] == 0), 0, 1),
                'bdryZ': bdry[:, 3]*np.clip(bdry_ratio[ii]+(bdry[:, 1] == 0), 0, 1),
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
                    print("Expanding node resolution from (Mnodes,Nnodes) = ({},{}) to ({},{})".format(
                        Mnodes[ii-1], Nnodes[ii-1], Mnodes[ii], Nnodes[ii]))
                nodes, volumes = get_nodes_pattern(
                    Mnodes[ii], Nnodes[ii], NFP, surfs=node_mode)
                bdry_nodes, _ = get_nodes_surf(
                    Mnodes[ii], Nnodes[ii], NFP, surf=1.0)
                zernt.expand_nodes(nodes, volumes)
                bdry_zernt.expand_nodes(bdry_nodes)
                t1 = time.perf_counter()
                if verbose > 0:
                    print("Expanding node resolution time = {} s".format(t1-t0))

            # spectral resolution
            if M[ii] != M[ii-1] or N[ii] != N[ii-1]:
                t0 = time.perf_counter()
                if verbose > 0:
                    print("Expanding spectral resolution from (M,N) = ({},{}) to ({},{})".format(
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
                if verbose > 0:
                    print("Expanding spectral resolution time = {} s".format(t1-t0))

            # continuation parameters
            delta_bdry = bdry_ratio[ii] - bdry_ratio[ii-1]
            delta_pres = pres_ratio[ii] - pres_ratio[ii-1]
            delta_zeta = zeta_ratio[ii] - zeta_ratio[ii-1]
            delta_errr = errr_ratio[ii] - errr_ratio[ii-1]
            deltas = np.array([delta_bdry, delta_pres, delta_zeta, delta_errr])

            # equilibrium objective function
            equil_obj, callback = get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M[ii], N[ii],
                                                    NFP, zernt, bdry_zernt, zern_idx, lambda_idx, bdry_pol, bdry_tor)
            args = [bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii-1],
                    pres_ratio[ii-1], zeta_ratio[ii-1], errr_ratio[ii-1]]

            if np.any(deltas):
                if verbose > 1:
                    print("Perturbing equilibrium")
                x = perturb(x, equil_obj, deltas, args,
                            pert_order[ii], verbose)

        args = [bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio[ii],
                pres_ratio[ii], zeta_ratio[ii], errr_ratio[ii]]
        if use_jax:
            if verbose > 0:
                print("Compiling objective function")
            equil_obj_jit = jit(equil_obj, static_argnums=())
            t0 = time.perf_counter()
            foo = equil_obj_jit(x, *args)
            t1 = time.perf_counter()
            if verbose > 0:
                print("Objective function compiled, time= {} s".format(t1-t0))
        else:
            equil_obj_jit = equil_obj
        if verbose > 0:
            print("Starting optimization")

        x_init = x
        t0 = time.perf_counter()
        out = scipy.optimize.least_squares(equil_obj_jit,
                                           x0=x_init,
                                           args=args,
                                           jac='2-point',
                                           method='trf',
                                           x_scale='jac',
                                           ftol=ftol[ii],
                                           xtol=xtol[ii],
                                           gtol=gtol[ii],
                                           max_nfev=nfev[ii],
                                           verbose=verbose)
        t1 = time.perf_counter()
        x = out['x']

        if verbose:
            print("Avg time per step: {} s".format((t1-t0)/out['nfev']))
            print("Start of Step {}:".format(ii+1))
            callback(x_init, *args)
            print("End of Step {}:".format(ii+1))
            callback(x, *args)

        cR, cZ, cL = unpack_x(np.matmul(sym_mat, x), len(zern_idx))
        equil = {
            'cR': cR,
            'cZ': cZ,
            'cL': cL,
            'bdryR': bdry[:, 2]*np.clip(bdry_ratio[ii]+(bdry[:, 1] == 0), 0, 1),
            'bdryZ': bdry[:, 3]*np.clip(bdry_ratio[ii]+(bdry[:, 1] == 0), 0, 1),
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
            checkpoint_file.write_iteration(equil, ii, inputs)

    if checkpoint:
        checkpoint_file.close()

    print('====================')
    print('Done')
    print('====================')

    return iterations
