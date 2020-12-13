import numpy as np
import time

from desc.backend import use_jax, Timer
from desc.jacobian import AutoDiffJacobian, FiniteDiffJacobian

if use_jax:
    jac = AutoDiffJacobian
else:
    jac = FiniteDiffJacobian


def perturb_continuation_params(x, equil_obj, deltas, args, pert_order=1, verbose=False, timer=None):
    """perturbs an equilibrium wrt the continuation parameters

    Parameters
    ----------
    x : ndarray
        state vector
    equil_obj : function
        equilibrium objective function
    deltas : ndarray
        changes in the continuation parameters
    args : tuple
        additional arguments passed to equil_obj
    pert_order : int
         order of perturbation (1=linear, 2=quadratic) (Default value = 1)
    verbose : int or bool
         level of output to display (Default value = False)
    timer : Timer
         Timer object (Default value = None)

    Returns
    -------
    x : ndarray
        perturbed state vector
    timer : Timer
        Timer object with timing data

    """

    delta_strings = ['boundary', 'pressure', 'zeta'] if len(deltas) == 3 else [
        None]*len(deltas)
    f = equil_obj(x, *args)
    dimF = len(f)
    dimX = len(x)
    dimC = 1
    if timer is None:
        timer = Timer()
    timer.start('Total perturbation')

    # 1st order
    if pert_order >= 1:

        # partial derivatives wrt x
        timer.start('df/dx computation')
        obj_jac_x = jac(equil_obj, argnum=0).compute
        Jx = obj_jac_x(x, *args).reshape((dimF, dimX))
        timer.stop('df/dx computation')
        RHS = f
        if verbose > 1:
            timer.disp('df/dx computation')

        # partial derivatives wrt c
        for i in range(deltas.size):
            if deltas[i] != 0:
                if verbose > 1:
                    print("Perturbing {}".format(delta_strings[i]))
                timer.start('df/dc computation ({})'.format(delta_strings[i]))
                obj_jac_c = jac(equil_obj, argnum=6+i).compute
                Jc = obj_jac_c(x, *args).reshape((dimF, dimC))
                timer.stop("df/dc computation ({})".format(delta_strings[i]))
                RHS += np.tensordot(Jc, np.atleast_1d(deltas[i]), axes=1)
                if verbose > 1:
                    timer.disp(
                        "df/dc computation ({})".format(delta_strings[i]))

    # 2nd order
    if pert_order >= 2:

        # partial derivatives wrt x
        Jxi = np.linalg.pinv(Jx, rcond=1e-6)
        timer.start("df/dxx computation")
        obj_jac_xx = jac(jac(equil_obj, argnum=0).compute, argnum=0).compute
        Jxx = obj_jac_xx(x, *args).reshape((dimF, dimX, dimX))
        timer.stop("df/dxx computation")
        RHS += 0.5 * np.tensordot(Jxx, np.tensordot(np.tensordot(Jxi, RHS, axes=1),
                                                    np.tensordot(RHS.T, Jxi.T, axes=1), axes=0), axes=2)
        if verbose > 1:
            timer.disp("df/dxx computation")

        # partial derivatives wrt c
        for i in range(deltas.size):
            if deltas[i] != 0:
                if verbose > 1:
                    print("Perturbing {}".format(delta_strings[i]))
                timer.start("df/dcc computation ({})".format(delta_strings[i]))
                obj_jac_cc = jac(
                    jac(equil_obj, argnum=6+i).compute, argnum=6+i).compute
                Jcc = obj_jac_cc(x, *args).reshape((dimF, dimC, dimC))
                timer.stop("df/dcc computation ({})".format(delta_strings[i]))
                RHS += 0.5 * np.tensordot(Jcc, np.tensordot(np.atleast_1d(deltas[i]),
                                                            np.atleast_1d(deltas[i]), axes=0), axes=2)
                if verbose > 1:
                    timer.disp(
                        "df/dcc computation ({})".format(delta_strings[i]))

                timer.start("df/dxc computation ({})".format(delta_strings[i]))
                obj_jac_xc = jac(
                    jac(equil_obj, argnum=0).compute, argnum=6+i).compute
                Jxc = obj_jac_xc(x, *args).reshape((dimF, dimX, dimC))
                timer.stop("df/dxc computation ({})".format(delta_strings[i]))
                RHS -= np.tensordot(Jxc, np.tensordot(Jxi, np.tensordot(RHS, np.atleast_1d(deltas[i]),
                                                                        axes=0), axes=1), axes=2)
                if verbose > 1:
                    timer.disp(
                        "df/dxc computation ({})".format(delta_strings[i]))

    # perturbation
    if pert_order > 0:
        dx = -np.linalg.lstsq(Jx, RHS, rcond=1e-6)[0]
    else:
        dx = np.zeros_like(x)
    timer.stop('Total perturbation')
    if verbose > 1:
        timer.disp('Total perturbation')
    return x + dx, timer


def get_system_derivatives(equil_obj, args, arg_dict, pert_order=1, verbose=False):
    """computes Jacobian and Hessian arrays

    Parameters
    ----------
    equil_obj : function
        objective function to calculate jacobian and hessian of
    args : tuple
        additional arguments passed to equil_obj
    arg_dict : dict
        dictionary of variable names and arguments to calculate derivatives with
        respect to.
    pert_order : int
         order of perturbation (1=linear, jacobian. 2=quadratic, hessian) (Default value = 1)
    verbose : int or bool
         level of text output (Default value = False)

    Returns
    -------
    Jx : ndarray
        jacobian wrt to state vector
    Jc : ndarray
        jacobian wrt to other parameters specified in arg_dict
    Jxx : ndarray
        hessian wrt to state vector.
        Only calculated if pert_order > 1
    Jcc : ndarray
        hessian wrt to other parameters specified in arg_dict.
        Only calculated if pert_order > 1
    Jxc : ndarray
        hessian wrt to state vector and other parameters.
        Only calculated if pert_order > 1

    """

    Jx = None
    Jc = None

    arg_idx = list(arg_dict.keys())

    f = equil_obj(*args)
    dimF = len(f)
    dimX = len(args[0])

    t00 = time.perf_counter()

    # 1st order
    if pert_order >= 1:

        # partial derivatives wrt x
        t0 = time.perf_counter()
        obj_jac_x = jac(equil_obj, argnum=0).compute
        Jx = obj_jac_x(*args).reshape((dimF, dimX))
        t1 = time.perf_counter()
        if verbose > 1:
            print("df/dx computation time: {} s".format(t1-t0))

        # partial derivatives wrt c
        flag = True
        for i in arg_idx:
            dimC = args[i].size
            t0 = time.perf_counter()
            obj_jac_c = jac(equil_obj, argnum=i).compute
            Jc_i = obj_jac_c(*args).reshape((dimF, dimC))
            Jc_i = Jc_i[:, arg_dict[i]]
            t1 = time.perf_counter()
            if verbose > 1:
                print("df/dc computation time: {} s".format(t1-t0))
            if flag:
                Jc = Jc_i
                flag = False
            else:
                Jc = np.concatenate((Jc, Jc_i), axis=1)

    # 2nd order
    if pert_order >= 2:

        # partial derivatives wrt x
        t0 = time.perf_counter()
        obj_jac_xx = jac(jac(equil_obj, argnum=0).compute, argnum=0).compute
        Jxx = obj_jac_xx(*args).reshape((dimF, dimX, dimX))
        t1 = time.perf_counter()
        if verbose > 1:
            print("df/dxx computation time: {} s".format(t1-t0))

        # partial derivatives wrt c
        flag = True
        for i in arg_idx:
            dimC = args[i].size
            t0 = time.perf_counter()
            obj_jac_cc = jac(jac(equil_obj, argnum=i).compute, argnum=i).compute
            Jcc_i = obj_jac_cc(*args).reshape((dimF, dimC, dimC))
            Jcc_i = Jcc_i[:, arg_dict[i], arg_dict[i]]
            t1 = time.perf_counter()
            if verbose > 1:
                print("df/dcc computation time: {} s".format(t1-t0))
            obj_jac_xc = jac(jac(equil_obj, argnum=0).compute, argnum=i).compute
            Jxc_i = obj_jac_xc(*args).reshape((dimF, dimX, dimC))
            Jxc_i = Jxc_i[:, :, arg_dict[i]]
            t2 = time.perf_counter()
            if verbose > 1:
                print("df/dxc computation time: {} s".format(t2-t1))
            if flag:
                Jcc = Jcc_i
                Jxc = Jxc_i
                flag = False
            else:
                Jcc = np.concatenate((Jcc, Jcc_i), axis=2)
                Jxc = np.concatenate((Jxc, Jxc_i), axis=2)

    t1 = time.perf_counter()
    if verbose > 1:
        print("Total perturbation time: {} s".format(t1-t00))

    if pert_order == 1:
        return Jx, Jc
    elif pert_order == 2:
        return Jx, Jc, Jxx, Jcc, Jxc
    else:
        return None
