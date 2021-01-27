import numpy as np
import time

from desc.utils import Timer
from desc.boundary_conditions import BoundaryConstraint

__all__ = ["perturb"]


def perturb(
    eq, deltas, order=0, Jx=None, verbose=1, copy=True,
):
    """Perturbs an Equilibrium wrt input parameters.

    Parameters
    ----------
    eq : Equilibrium
        equilibrium to perturb
    deltas : dict
        dictionary of ndarray of objective function parameters to perturb.
        Allowed keys are: ``'Rb_mn'``, ``'Zb_mn'``, ``'p_l'``, ``'i_l'``,
        ``'Psi'``, ``'zeta_ratio'``
    order : int, optional
        order of perturbation (0=none, 1=linear, 2=quadratic)
    Jx : ndarray, optional
        jacobian matrix df/dx
    verbose : int
        level of output to display
    copy : bool
        whether to perturb the input equilibrium or make a copy. Defaults to True

    Returns
    -------
    eq_new : Equilibrium
        perturbed equilibrium

    """
    timer = Timer()
    timer.start("Total perturbation")

    arg_idx = {"Rb_mn": 1, "Zb_mn": 2, "p_l": 3, "i_l": 4, "Psi": 5, "zeta_ratio": 6}
    if not eq.built:
        eq.build(verbose)
    y = eq.objective.BC_constraint.project(eq.x)
    args = (y, eq.Rb_mn, eq.Zb_mn, eq.p_l, eq.i_l, eq.Psi, eq.zeta_ratio)

    # 1st order
    if order > 0:

        # partial derivatives wrt state vector (x)
        if Jx is None:
            timer.start("df/dx computation")
            Jx = eq.objective.jac_x(*args)
            timer.stop("df/dx computation")
            RHS = eq.objective.compute(*args)

            if verbose > 1:
                timer.disp("df/dx computation")

        # partial derivatives wrt input parameters (c)
        for key, dc in deltas.items():
            if verbose > 0:
                print("Perturbing {}".format(key))
            timer.start("df/dc computation ({})".format(key))
            RHS += eq.objective.jvp(arg_idx[key], dc, *args)
            timer.stop("df/dc computation ({})".format(key))
            if verbose > 1:
                timer.disp("df/dc computation ({})".format(key))

    # 2nd order
    if order > 1:

        RHS1 = RHS

        # partial derivatives wrt state vector (x)
        Jxi = np.linalg.pinv(Jx, rcond=1e-6)
        timer.start("df/dxx computation")
        Jxx = eq.objective.derivative((0, 0), *args)
        timer.stop("df/dxx computation")
        RHS += 0.5 * np.tensordot(
            Jxx,
            np.tensordot(
                np.tensordot(Jxi, RHS1, axes=1),
                np.tensordot(RHS1.T, Jxi.T, axes=1),
                axes=0,
            ),
            axes=2,
        )
        if verbose > 1:
            timer.disp("df/dxx computation")

        # partial derivatives wrt input parameters (c)
        for key, dc in deltas.items():

            if verbose > 0:
                print("Perturbing {}".format(key))
            timer.start("df/dcc computation ({})".format(key))
            Jcc = eq.objective.derivative((arg_idx[key], arg_idx[key]), *args)
            timer.stop("df/dcc computation ({})".format(key))
            RHS += 0.5 * np.tensordot(Jcc, np.tensordot(dc, dc, axes=0), axes=2,)
            if verbose > 1:
                timer.disp("df/dcc computation ({})".format(key))

            timer.start("df/dxc computation ({})".format(key))
            Jxc = eq.objective.derivative((0, arg_idx[key]), *args)
            timer.stop("df/dxc computation ({})".format(key))
            RHS -= np.tensordot(
                Jxc, np.tensordot(Jxi, np.tensordot(RHS1, dc, axes=0), axes=1), axes=2,
            )
            if verbose > 1:
                timer.disp("df/dxc computation ({})".format(key))

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    # update input parameters
    for key, dc in deltas.items():
        setattr(eq_new, key, getattr(eq_new, key) + dc)

    # update boundary constraint
    if "Rb_mn" in deltas or "Zb_mn" in deltas:
        eq_new.objective.BC_constraint = BoundaryConstraint(
            eq_new.R_basis,
            eq_new.Z_basis,
            eq_new.L_basis,
            eq_new.Rb_basis,
            eq_new.Zb_basis,
            eq_new.Rb_mn,
            eq_new.Zb_mn,
        )

    # perturbation
    if order > 0:
        dy = -np.linalg.lstsq(Jx, RHS, rcond=1e-6)[0]
        eq_new.x = eq_new.objective.BC_constraint.recover(y + dy)

    timer.stop("Total perturbation")
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new


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
        obj_jac_x = Derivative(equil_obj, argnum=0).compute
        Jx = obj_jac_x(*args).reshape((dimF, dimX))
        t1 = time.perf_counter()
        if verbose > 1:
            print("df/dx computation time: {} s".format(t1 - t0))

        # partial derivatives wrt c
        flag = True
        for i in arg_idx:
            dimC = args[i].size
            t0 = time.perf_counter()
            obj_jac_c = Derivative(equil_obj, argnum=i).compute
            Jc_i = obj_jac_c(*args).reshape((dimF, dimC))
            Jc_i = Jc_i[:, arg_dict[i]]
            t1 = time.perf_counter()
            if verbose > 1:
                print("df/dc computation time: {} s".format(t1 - t0))
            if flag:
                Jc = Jc_i
                flag = False
            else:
                Jc = np.concatenate((Jc, Jc_i), axis=1)

    # 2nd order
    if pert_order >= 2:

        # partial derivatives wrt x
        t0 = time.perf_counter()
        obj_jac_xx = Derivative(
            Derivative(equil_obj, argnum=0).compute, argnum=0
        ).compute
        Jxx = obj_jac_xx(*args).reshape((dimF, dimX, dimX))
        t1 = time.perf_counter()
        if verbose > 1:
            print("df/dxx computation time: {} s".format(t1 - t0))

        # partial derivatives wrt c
        flag = True
        for i in arg_idx:
            dimC = args[i].size
            t0 = time.perf_counter()
            obj_jac_cc = Derivative(
                Derivative(equil_obj, argnum=i).compute, argnum=i
            ).compute
            Jcc_i = obj_jac_cc(*args).reshape((dimF, dimC, dimC))
            Jcc_i = Jcc_i[:, arg_dict[i], arg_dict[i]]
            t1 = time.perf_counter()
            if verbose > 1:
                print("df/dcc computation time: {} s".format(t1 - t0))
            obj_jac_xc = Derivative(
                Derivative(equil_obj, argnum=0).compute, argnum=i
            ).compute
            Jxc_i = obj_jac_xc(*args).reshape((dimF, dimX, dimC))
            Jxc_i = Jxc_i[:, :, arg_dict[i]]
            t2 = time.perf_counter()
            if verbose > 1:
                print("df/dxc computation time: {} s".format(t2 - t1))
            if flag:
                Jcc = Jcc_i
                Jxc = Jxc_i
                flag = False
            else:
                Jcc = np.concatenate((Jcc, Jcc_i), axis=2)
                Jxc = np.concatenate((Jxc, Jxc_i), axis=2)

    t1 = time.perf_counter()
    if verbose > 1:
        print("Total perturbation time: {} s".format(t1 - t00))

    if pert_order == 1:
        return Jx, Jc
    elif pert_order == 2:
        return Jx, Jc, Jxx, Jcc, Jxc
    else:
        return None
