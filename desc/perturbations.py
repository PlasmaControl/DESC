import numpy as np
import time

from desc.backend import jacfwd


def perturb_continuation_params(x, equil_obj, deltas, args, pert_order=1, verbose=False):
    """perturbs an equilibrium wrt the continuation parameters"""

    delta_strings = ['boundary', 'pressure', 'zeta']
    f = equil_obj(x, *args)
    dimF = len(f)
    dimX = len(x)
    dimC = 1

    t00 = time.perf_counter()

    # 1st order
    if pert_order >= 1:

        # partial derivatives wrt x
        t0 = time.perf_counter()
        obj_jac_x = jacfwd(equil_obj, argnums=0)
        Jx = obj_jac_x(x, *args).reshape((dimF, dimX))
        t1 = time.perf_counter()
        RHS = f
        if verbose > 1:
            print("df/dx computation time: {} s".format(t1-t0))

        # partial derivatives wrt continuation parameters
        for i in range(deltas.size):
            if deltas[i] != 0:
                if verbose > 1:
                    print("Perturbing {}".format(delta_strings[i]))
                t0 = time.perf_counter()
                obj_jac_c = jacfwd(equil_obj, argnums=6+i)
                Jc = obj_jac_c(x, *args).reshape((dimF, dimC))
                t1 = time.perf_counter()
                RHS += np.tensordot(Jc, np.atleast_1d(deltas[i]), axes=1)
                if verbose > 1:
                    print("df/dc computation time: {} s".format(t1-t0))

    # 2nd order
    if pert_order >= 2:

        # partial derivatives wrt x
        Jxi = np.linalg.pinv(Jx, rcond=1e-6)
        t0 = time.perf_counter()
        obj_jac_xx = jacfwd(jacfwd(equil_obj, argnums=0), argnums=0)
        Jxx = obj_jac_xx(x, *args).reshape((dimF, dimX, dimX))
        t1 = time.perf_counter()
        RHS += 0.5 * np.tensordot(Jxx, np.tensordot(np.tensordot(Jxi, RHS, axes=1),
                                                    np.tensordot(RHS.T, Jxi.T, axes=1), axes=0), axes=2)
        if verbose > 1:
            print("df/dxx computation time: {} s".format(t1-t0))

        # partial derivatives wrt continuation parameters
        for i in range(deltas.size):
            if deltas[i] != 0:
                if verbose > 1:
                    print("Perturbing {}".format(delta_strings[i]))
                t0 = time.perf_counter()
                obj_jac_cc = jacfwd(
                    jacfwd(equil_obj, argnums=6+i), argnums=6+i)
                Jcc = obj_jac_cc(x, *args).reshape((dimF, dimC, dimC))
                t1 = time.perf_counter()
                RHS += 0.5 * np.tensordot(Jcc, np.tensordot(np.atleast_1d(deltas[i]),
                                                            np.atleast_1d(deltas[i]), axes=0), axes=2)
                if verbose > 1:
                    print("df/dcc computation time: {} s".format(t1-t0))
                obj_jac_xc = jacfwd(jacfwd(equil_obj, argnums=0), argnums=6+i)
                Jxc = obj_jac_xc(x, *args).reshape((dimF, dimX, dimC))
                t2 = time.perf_counter()
                RHS -= np.tensordot(Jxc, np.tensordot(Jxi, np.tensordot(RHS, np.atleast_1d(deltas[i]),
                                                                        axes=0), axes=1), axes=2)
                if verbose > 1:
                    print("df/dxc computation time: {} s".format(t2-t1))

    # perturbation
    if pert_order > 0:
        dx = -np.linalg.lstsq(Jx, RHS, rcond=1e-6)[0]
    else:
        dx = np.zeros_like(x)
    t1 = time.perf_counter()
    if verbose > 1:
        print("Total perturbation time: {} s".format(t1-t00))
    return x + dx


def get_system_derivatives(equil_obj, args, arg_dict, pert_order=1, verbose=False):
    """computes Jacobian and Hessian arrays"""

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
        obj_jac_x = jacfwd(equil_obj, argnums=0)
        Jx = obj_jac_x(*args).reshape((dimF, dimX))
        t1 = time.perf_counter()
        if verbose > 1:
            print("df/dx computation time: {} s".format(t1-t0))

        # partial derivatives wrt c
        flag = True
        for i in arg_idx:
            dimC = args[i].size
            t0 = time.perf_counter()
            obj_jac_c = jacfwd(equil_obj, argnums=i)
            Jc_i = obj_jac_c(*args).reshape((dimF, dimC))
            Jc_i = Jc_i[:,arg_dict[i]]
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
        obj_jac_xx = jacfwd(jacfwd(equil_obj, argnums=0), argnums=0)
        Jxx = obj_jac_xx(*args).reshape((dimF, dimX, dimX))
        t1 = time.perf_counter()
        if verbose > 1:
            print("df/dxx computation time: {} s".format(t1-t0))

        # partial derivatives wrt c
        flag = True
        for i in arg_idx:
            dimC = args[i].size
            t0 = time.perf_counter()
            obj_jac_cc = jacfwd(jacfwd(equil_obj, argnums=i), argnums=i)
            Jcc_i = obj_jac_cc(*args).reshape((dimF, dimC, dimC))
            Jcc_i = Jcc_i[:,arg_dict[i],arg_dict[i]]
            t1 = time.perf_counter()
            if verbose > 1:
                print("df/dcc computation time: {} s".format(t1-t0))
            obj_jac_xc = jacfwd(jacfwd(equil_obj, argnums=0), argnums=i)
            Jxc_i = obj_jac_xc(*args).reshape((dimF, dimX, dimC))
            Jxc_i = Jxc_i[:,:,arg_dict[i]]
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
