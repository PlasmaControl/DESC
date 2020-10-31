import numpy as np

from desc.backend import jacfwd, Timer


def perturb_continuation_params(x, equil_obj, deltas, args, pert_order=1, verbose=False, timer=None):
    """perturbs an equilibrium wrt the continuation parameters"""

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
        obj_jac_x = jacfwd(equil_obj, argnums=0)
        Jx = obj_jac_x(x, *args).reshape((dimF, dimX))
        timer.stop('df/dx computation')
        RHS = f
        if verbose > 1:
            timer.disp('df/dx computation')

        # partial derivatives wrt continuation parameters
        for i in range(deltas.size):
            if deltas[i] != 0:
                if verbose > 1:
                    print("Perturbing {}".format(delta_strings[i]))
                timer.start('df/dc computation ({})'.format(delta_strings[i]))
                obj_jac_c = jacfwd(equil_obj, argnums=6+i)
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
        obj_jac_xx = jacfwd(jacfwd(equil_obj, argnums=0), argnums=0)
        Jxx = obj_jac_xx(x, *args).reshape((dimF, dimX, dimX))
        timer.stop("df/dxx computation")
        RHS += 0.5 * np.tensordot(Jxx, np.tensordot(np.tensordot(Jxi, RHS, axes=1),
                                                    np.tensordot(RHS.T, Jxi.T, axes=1), axes=0), axes=2)
        if verbose > 1:
            timer.disp("df/dxx computation")

        # partial derivatives wrt continuation parameters
        for i in range(deltas.size):
            if deltas[i] != 0:
                if verbose > 1:
                    print("Perturbing {}".format(delta_strings[i]))
                timer.start("df/dcc computation ({})".format(delta_strings[i]))
                obj_jac_cc = jacfwd(
                    jacfwd(equil_obj, argnums=6+i), argnums=6+i)
                Jcc = obj_jac_cc(x, *args).reshape((dimF, dimC, dimC))
                timer.stop("df/dcc computation ({})".format(delta_strings[i]))
                RHS += 0.5 * np.tensordot(Jcc, np.tensordot(np.atleast_1d(deltas[i]),
                                                            np.atleast_1d(deltas[i]), axes=0), axes=2)
                if verbose > 1:
                    timer.disp(
                        "df/dcc computation ({})".format(delta_strings[i]))

                timer.start("df/dxc computation ({})".format(delta_strings[i]))
                obj_jac_xc = jacfwd(jacfwd(equil_obj, argnums=0), argnums=6+i)
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


# TODO: finish this function
def perturb_coefficients(x, equil_obj, delta_bdryR, delta_bdryZ, delta_cP, delta_cI, delta_Psi_lcfs, args, pert_order=1, verbose=False):
    """perturbs an equilibrium wrt the input coefficients"""

    delta_strings = ['R boundary', 'Z boundary', 'pressure',
                     'rotational transform', 'max toroidal flux']
    f = equil_obj(x, *args)
    dimF = len(f)
    dimX = len(x)
    dimC = np.array([delta_bdryR.size, delta_bdryZ.size,
                     delta_cP.size, delta_cI.size, delta_Psi_lcfs.size])

    return Null
