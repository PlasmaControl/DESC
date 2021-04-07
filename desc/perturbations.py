import numpy as np
import warnings
from termcolor import colored

from desc.utils import Timer
from desc.backend import use_jax
from desc.boundary_conditions import get_boundary_condition
from desc.optimize.tr_subproblems import trust_region_step_exact

__all__ = ["perturb", "optimal_perturb"]


def perturb(
    eq,
    dRb=None,
    dZb=None,
    dp=None,
    di=None,
    dPsi=None,
    dzeta_ratio=None,
    order=2,
    tr_ratio=0.1,
    cutoff=1e-6,
    Jx=None,
    verbose=1,
    copy=True,
):
    """Perturb an Equilibrium with respect to input parameters.

    Parameters
    ----------
    eq : Equilibrium
        equilibrium to perturb
    dRb, dZb, dp, di, dPsi, dzeta_ratio : ndarray or float
        deltas for perturbations of R_boundary, Z_boundary, pressure, iota,
        toroidal flux, and zeta ratio.
        Setting to None or zero ignores that term in the expansion.
    order : {0,1,2,3}
        order of perturbation (0=none, 1=linear, 2=quadratic, etc)
    tr_ratio : float or array of float
        radius of the trust region, as a fraction of ||x||.
        enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||
        if a scalar uses same ratio for all steps, if an array uses the first element
        for the first step and so on
    cutoff : float
        relative cutoff for small singular values in pseudoinverse
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
    if not use_jax:
        warnings.warn(
            colored(
                "Computing perturbations with finite differences can be "
                + "highly innacurate. Consider using JAX for exact derivatives.",
                "yellow",
            )
        )
    if not eq.objective:
        raise AttributeError(
            "Equilibrium must have objective defined before perturbing."
        )
    if eq.objective.scalar:
        raise AttributeError(
            "Cannot perturb with a scalar objective: {}".format(eq.objective)
        )
    if np.isscalar(tr_ratio):
        tr_ratio = tr_ratio * np.ones(order)
    elif len(tr_ratio) < order:
        raise ValueError(
            "Got only {} tr_ratios for order {} perturbations".format(
                len(tr_ratio), order
            )
        )

    deltas = {}
    if dRb is not None and np.any(dRb):
        deltas["Rb_lmn"] = dRb
    if dZb is not None and np.any(dZb):
        deltas["Zb_lmn"] = dZb
    if dp is not None and np.any(dp):
        deltas["p_l"] = dp
    if di is not None and np.any(di):
        deltas["i_l"] = di
    if dPsi is not None and np.any(dPsi):
        deltas["Psi"] = dPsi
    if dzeta_ratio is not None and np.any(dzeta_ratio):
        deltas["zeta_ratio"] = dzeta_ratio

    keys = ", ".join(deltas.keys())
    if verbose > 0:
        print("Perturbing {}".format(keys))

    timer = Timer()
    timer.start("Total perturbation")

    arg_idx = {"Rb_lmn": 1, "Zb_lmn": 2, "p_l": 3, "i_l": 4, "Psi": 5, "zeta_ratio": 6}
    if not eq.built:
        eq.build(verbose)
    y = eq.objective.BC_constraint.project(eq.x)
    args = (y, eq.Rb_lmn, eq.Zb_lmn, eq.p_l, eq.i_l, eq.Psi, eq.zeta_ratio)
    dx1 = 0
    dx2 = 0
    dx3 = 0

    # 1st order
    if order > 0:

        # 1st partial derivatives wrt state vector (x)
        if Jx is None:
            if verbose > 0:
                print("Computing df")
            timer.start("df/dx computation")
            Jx = eq.objective.jac_x(*args)
            timer.stop("df/dx computation")
            if verbose > 1:
                timer.disp("df/dx computation")

        u, s, vt = np.linalg.svd(Jx, full_matrices=False)
        m, n = Jx.shape
        RHS1 = eq.objective.compute(*args)

        # partial derivatives wrt input parameters (c)
        inds = tuple([arg_idx[key] for key in deltas])
        dc = tuple([val for val in deltas.values()])
        timer.start("df/dc computation ({})".format(keys))
        RHS1 += eq.objective.jvp(inds, dc, *args)
        timer.stop("df/dc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("df/dc computation ({})".format(keys))

        dx1, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS1,
            u,
            s,
            vt.T,
            tr_ratio[0] * np.linalg.norm(y),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # 2nd order
    if order > 1:

        # 2nd partial derivatives wrt state vector (x)
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        inds = tuple([arg_idx[key] for key in deltas])
        tangents = tuple([val for val in deltas.values()])
        inds = (0, *inds)
        tangents = (dx1, *tangents)
        RHS2 = 0.5 * eq.objective.jvp2(inds, inds, tangents, tangents, *args)

        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        dx2, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS2,
            u,
            s,
            vt.T,
            tr_ratio[1] * np.linalg.norm(dx1),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # 3rd order
    if order > 2:

        # 3rd partial derivatives wrt state vector (x)
        if verbose > 0:
            print("Computing d^3f")
        timer.start("d^3f computation")
        inds = tuple([arg_idx[key] for key in deltas])
        tangents = tuple([val for val in deltas.values()])
        inds = (0, *inds)
        tangents = (dx1, *tangents)
        RHS3 = (
            1
            / 6
            * eq.objective.jvp3(inds, inds, inds, tangents, tangents, tangents, *args)
        )
        RHS3 += eq.objective.jvp2(0, inds, dx2, tangents, *args)
        timer.stop("d^3f computation")
        if verbose > 1:
            timer.disp("d^3f computation")

        dx3, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS3,
            u,
            s,
            vt.T,
            tr_ratio[2] * np.linalg.norm(dx2),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    # update input parameters
    for key, dc in deltas.items():
        setattr(eq_new, key, getattr(eq_new, key) + dc)

    # update boundary constraint
    if "Rb_lmn" in deltas or "Zb_lmn" in deltas:
        eq_new.objective.BC_constraint = get_boundary_condition(
            eq.objective.BC_constraint.name,
            eq_new.R_basis,
            eq_new.Z_basis,
            eq_new.L_basis,
            eq_new.Rb_basis,
            eq_new.Zb_basis,
            eq_new.Rb_lmn,
            eq_new.Zb_lmn,
        )

    # update state vector
    dy = dx1 + dx2 + dx3
    eq_new.x = eq_new.objective.BC_constraint.recover(y + dy)

    timer.stop("Total perturbation")
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new


def optimal_perturb(
    eq,
    objective,
    dRb=False,
    dZb=False,
    dp=False,
    di=False,
    dPsi=False,
    dzeta_ratio=False,
    order=1,
    tr_ratio=0.1,
    cutoff=1e-6,
    Jx=None,
    verbose=1,
    copy=True,
):
    """Perturb an Equilibrium with respect to input parameters to optimize an objective.

    Parameters
    ----------
    eq : Equilibrium
        equilibrium to optimize
    objective : ObjectiveFunction
        objective to optimize
    dRb, dZb, dp, di, dPsi, dzeta_ratio : ndarray or bool
        array of indicies of modes to include in the perturbations of
        R_boundary, Z_boundary, pressure, iota, toroidal flux, and zeta ratio.
        Setting to True (False) includes (excludes) all modes.
    order : {0,1,2,3}
        order of perturbation (0=none, 1=linear, 2=quadratic, etc)
    tr_ratio : float or array of float
        radius of the trust region, as a fraction of ||x||.
        enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||
        if a scalar uses same ratio for all steps, if an array uses the first element
        for the first step and so on
    cutoff : float
        relative cutoff for small singular values in pseudoinverse
    Jx : ndarray, optional
        jacobian matrix df/dx
    verbose : int
        level of output to display
    copy : bool
        whether to perturb the input equilibrium or make a copy. Defaults to True

    Returns
    -------
    eq_new : Equilibrium
        optimized equilibrium

    """
    if not use_jax:
        warnings.warn(
            colored(
                "Computing perturbations with finite differences can be "
                + "highly innacurate. Consider using JAX for exact derivatives.",
                "yellow",
            )
        )
    if not eq.objective:
        raise AttributeError(
            "Equilibrium must have objective defined before perturbing."
        )
    if eq.objective.scalar:
        raise AttributeError(
            "Cannot perturb with a scalar objective: {}".format(eq.objective)
        )
    if np.isscalar(tr_ratio):
        tr_ratio = tr_ratio * np.ones(order)
    elif len(tr_ratio) < order:
        raise ValueError(
            "Got only {} tr_ratios for order {} perturbations".format(
                len(tr_ratio), order
            )
        )

    deltas = {}
    if type(dRb) is bool or dRb is None:
        if dRb is True:
            deltas["Rb_lmn"] = np.ones((eq.Rb_basis.num_modes,), dtype=bool)
    elif np.any(dRb):
        deltas["Rb_lmn"] = dRb
    if type(dZb) is bool or dZb is None:
        if dZb is True:
            deltas["Zb_lmn"] = np.ones((eq.Zb_basis.num_modes,), dtype=bool)
    elif np.any(dZb):
        deltas["Zb_lmn"] = dZb
    if type(dp) is bool or dp is None:
        if dp is True:
            deltas["p_l"] = np.ones((eq.p_basis.num_modes,), dtype=bool)
    elif np.any(dp):
        deltas["p_l"] = dp
    if type(di) is bool or di is None:
        if di is True:
            deltas["i_l"] = np.ones((eq.i_basis.num_modes,), dtype=bool)
    elif np.any(di):
        deltas["i_l"] = di
    if type(dPsi) is bool or dPsi is None:
        if dPsi is True:
            deltas["Psi"] = np.array([True])
    if type(dzeta_ratio) is bool or dzeta_ratio is None:
        if dzeta_ratio is True:
            deltas["zeta_ratio"] = np.array([True])
    if not len(deltas):
        raise ValueError("At least one input must be a free variable for optimization.")

    keys = ", ".join(deltas.keys())
    if verbose > 0:
        print("Perturbing {}".format(keys))

    timer = Timer()
    timer.start("Total perturbation")

    arg_idx = {"Rb_lmn": 1, "Zb_lmn": 2, "p_l": 3, "i_l": 4, "Psi": 5, "zeta_ratio": 6}
    if not eq.built:
        eq.build(verbose)
    y = eq.objective.BC_constraint.project(eq.x)
    c = np.array([])
    for key, dc in deltas.items():
        c = np.concatenate((c, getattr(eq, key)))
    args = (y, eq.Rb_lmn, eq.Zb_lmn, eq.p_l, eq.i_l, eq.Psi, eq.zeta_ratio)
    dc1 = 0
    dc2 = 0
    dc3 = 0
    dx1 = 0
    dx2 = 0
    dx3 = 0

    # 1st order
    if order > 0:

        f = eq.objective.compute(*args)  # primary objective residual
        g = objective.compute(*args)  # secondary objective residual

        # Jacobian of primary objective (f) wrt state vector (x)
        if verbose > 0:
            print("Computing df")
        if Jx is None:
            timer.start("df/dx computation")
            Jx = eq.objective.jac_x(*args)
            timer.stop("df/dx computation")
            if verbose > 1:
                timer.disp("df/dx computation")
        Jx_inv = np.linalg.pinv(Jx, rcond=cutoff)

        # Jacobian of primary objective (f) wrt input parameters (c)
        Jc = np.array([])
        timer.start("df/dc computation ({})".format(keys))
        for key, idx in deltas.items():
            Jc_i = eq.objective.derivative(arg_idx[key], *args)[:, idx]
            Jc = np.hstack((Jc, Jc_i)) if Jc.size else Jc_i
        timer.stop("df/dc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("df/dc computation ({})".format(keys))

        LHS1 = np.matmul(Jx_inv, Jc)
        RHS1 = np.matmul(Jx_inv, f)

        # Jacobian of secondary objective (g) wrt state vector (x)
        if verbose > 0:
            print("Computing dg")
        timer.start("dg/dx computation")
        Gx = objective.jac_x(*args)
        timer.stop("dg/dx computation")
        if verbose > 1:
            timer.disp("dg/dx computation")

        # Jacobian of secondary objective (g) wrt input parameters (c)
        Gc = np.array([])
        timer.start("dg/dc computation ({})".format(keys))
        for key, idx in deltas.items():
            Gc_i = objective.derivative(arg_idx[key], *args)[:, idx]
            Gc = np.hstack((Gc, Gc_i)) if Gc.size else Gc_i
        timer.stop("dg/dc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("dg/dc computation ({})".format(keys))

        LHS1 = np.matmul(Gx, LHS1) - Gc
        RHS1 = np.matmul(Gx, RHS1) - g

        u, s, vt = np.linalg.svd(LHS1, full_matrices=False)
        m, n = LHS1.shape

        # find optimal perturbation
        dc1, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS1,
            u,
            s,
            vt.T,
            tr_ratio[0] * np.linalg.norm(c),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

        RHS1 = f + np.matmul(Jc, dc1)

        u, s, vt = np.linalg.svd(Jx, full_matrices=False)
        m, n = Jx.shape

        # apply optimal perturbation
        dx1, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS1,
            u,
            s,
            vt.T,
            tr_ratio[0] * np.linalg.norm(y),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    if order > 1:
        raise ValueError(
            "Higher-order optimizations not yet implemented: {}".format(order)
        )

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    dc = dc1 + dc2 + dc3
    dy = dx1 + dx2 + dx3
    if verbose > 1:
        print("||dc||/||c|| = {}".format(np.linalg.norm(dc) / np.linalg.norm(c)))
        print("||dx||/||x|| = {}".format(np.linalg.norm(dy) / np.linalg.norm(y)))

    # update input parameters
    idx0 = 0
    for key, idx in deltas.items():
        dc_i = np.zeros((getattr(eq_new, key).size,))
        dc_i[idx] = dc[idx0 : idx0 + np.sum(idx)]
        setattr(eq_new, key, getattr(eq_new, key) + dc_i)
        idx0 += np.sum(idx)

    # update boundary constraint
    if "Rb_lmn" in deltas or "Zb_lmn" in deltas:
        eq_new.objective.BC_constraint = get_boundary_condition(
            eq.objective.BC_constraint.name,
            eq_new.R_basis,
            eq_new.Z_basis,
            eq_new.L_basis,
            eq_new.Rb_basis,
            eq_new.Zb_basis,
            eq_new.Rb_lmn,
            eq_new.Zb_lmn,
        )

    # update state vector
    eq_new.x = eq_new.objective.BC_constraint.recover(y + dy)

    timer.stop("Total perturbation")
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new
