import numpy as np
import warnings
from termcolor import colored
from desc.utils import Timer
from desc.backend import use_jax, jnp
from desc.optimize.tr_subproblems import trust_region_step_exact_svd

__all__ = ["perturb", "optimal_perturb"]


def perturb(
    eq,
    dRb=None,
    dZb=None,
    dp=None,
    di=None,
    dPsi=None,
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
    dRb, dZb, dp, di, dPsi : ndarray or float
        deltas for perturbations of R_boundary, Z_boundary, pressure, iota, and
        toroidal flux.
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
    eq.objective.build(verbose=verbose)
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

    keys = ", ".join(deltas.keys())
    if verbose > 0:
        print("Perturbing {}".format(keys))

    timer = Timer()
    timer.start("Total perturbation")

    arg_idx = {"Rb_lmn": 1, "Zb_lmn": 2, "p_l": 3, "i_l": 4, "Psi": 5}
    if not eq.built:
        eq.build(verbose)
    y = eq.objective.BC_constraint.project(eq.x)
    args = (y, eq.Rb_lmn, eq.Zb_lmn, eq.p_l, eq.i_l, eq.Psi)
    dx1 = 0
    dx2 = 0
    dx3 = 0

    # 1st order
    if order > 0:

        RHS1 = eq.objective.compute(*args)

        # 1st partial derivatives wrt state vector (x)
        if Jx is None:
            if verbose > 0:
                print("Computing df")
            timer.start("df/dx computation")
            Jx = eq.objective.jac_x(*args)
            timer.stop("df/dx computation")
            if verbose > 1:
                timer.disp("df/dx computation")

        if verbose > 0:
            print("Factoring df")
        timer.start("df/dx factorization")
        m, n = Jx.shape
        u, s, vt = jnp.linalg.svd(Jx, full_matrices=False)
        timer.stop("df/dx factorization")
        if verbose > 1:
            timer.disp("df/dx factorization")
        # once we have the SVD we don't need Jx anymore so can save some memory
        del Jx

        # 1st partial derivatives wrt input parameters (c)
        inds = tuple([arg_idx[key] for key in deltas])
        dc = tuple([val for val in deltas.values()])
        timer.start("df/dc computation ({})".format(keys))
        RHS1 += eq.objective.jvp(inds, dc, *args)
        timer.stop("df/dc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("df/dc computation ({})".format(keys))

        dx1, hit, alpha = trust_region_step_exact_svd(
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

        inds = tuple([arg_idx[key] for key in deltas])
        tangents = tuple([val for val in deltas.values()])
        inds = (0, *inds)
        tangents = (dx1, *tangents)

        # 2nd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        RHS2 = 0.5 * eq.objective.jvp2(inds, inds, tangents, tangents, *args)
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        dx2, hit, alpha = trust_region_step_exact_svd(
            RHS2,
            u,
            s,
            vt.T,
            tr_ratio[1] * np.linalg.norm(dx1),
            initial_alpha=alpha / tr_ratio[1],
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # 3rd order
    if order > 2:

        # 3rd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^3f")
        timer.start("d^3f computation")
        RHS3 = (
            1
            / 6
            * eq.objective.jvp3(inds, inds, inds, tangents, tangents, tangents, *args)
        )
        RHS3 += eq.objective.jvp2(0, inds, dx2, tangents, *args)
        timer.stop("d^3f computation")
        if verbose > 1:
            timer.disp("d^3f computation")

        dx3, hit, alpha = trust_region_step_exact_svd(
            RHS3,
            u,
            s,
            vt.T,
            tr_ratio[2] * np.linalg.norm(dx2),
            initial_alpha=alpha / tr_ratio[2],
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
        eq_new.objective.BC_constraint = eq.surface.get_constraint(
            eq_new.R_basis,
            eq_new.Z_basis,
            eq_new.L_basis,
        )

    # update state vector
    dy = dx1 + dx2 + dx3
    eq_new.x = np.copy(eq_new.objective.BC_constraint.recover(y + dy))
    timer.stop("Total perturbation")
    if verbose > 1:
        timer.disp("Total perturbation")
        print("||dx||/||x|| = {}".format(np.linalg.norm(dy) / np.linalg.norm(y)))

    return eq_new


def optimal_perturb(
    eq,
    objective,
    dRb=False,
    dZb=False,
    dp=False,
    di=False,
    dPsi=False,
    order=2,
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
    dRb, dZb, dp, di, dPsi : ndarray or bool
        array of indicies of modes to include in the perturbations of
        R_boundary, Z_boundary, pressure, iota, and toroidal flux.
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
            deltas["p_l"] = np.ones_like(eq.p_l, dtype=bool)
    elif np.any(dp):
        deltas["p_l"] = dp
    if type(di) is bool or di is None:
        if di is True:
            deltas["i_l"] = np.ones_like(eq.i_l, dtype=bool)
    elif np.any(di):
        deltas["i_l"] = di
    if type(dPsi) is bool or dPsi is None:
        if dPsi is True:
            deltas["Psi"] = np.array([True])
    if not len(deltas):
        raise ValueError("At least one input must be a free variable for optimization.")

    keys = ", ".join(deltas.keys())
    if verbose > 0:
        print("Perturbing {}".format(keys))

    timer = Timer()
    timer.start("Total perturbation")

    Fx = Jx
    arg_idx = {"Rb_lmn": 1, "Zb_lmn": 2, "p_l": 3, "i_l": 4, "Psi": 5}
    if not eq.built:
        eq.build(verbose)
    y = eq.objective.BC_constraint.project(eq.x)
    c = np.array([])
    c_idx = np.array([], dtype=bool)
    for key, idx in deltas.items():
        c = np.concatenate((c, getattr(eq, key)))
        c_idx = np.concatenate((c_idx, idx))
    args = (y, eq.Rb_lmn, eq.Zb_lmn, eq.p_l, eq.i_l, eq.Psi)
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
        if Fx is None:
            timer.start("df/dx computation")
            Fx = eq.objective.jac_x(*args)
            timer.stop("df/dx computation")
            if verbose > 1:
                timer.disp("df/dx computation")
        Fx_inv = np.linalg.pinv(Fx, rcond=cutoff)

        # Jacobian of primary objective (f) wrt input parameters (c)
        Fc = np.array([])
        timer.start("df/dc computation ({})".format(keys))
        for key in deltas.keys():
            Fc_i = eq.objective.derivative(arg_idx[key], *args)
            Fc = np.hstack((Fc, Fc_i)) if Fc.size else Fc_i
        timer.stop("df/dc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("df/dc computation ({})".format(keys))

        LHS = np.matmul(Fx_inv, Fc)
        RHS_1g = np.matmul(Fx_inv, f)

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
        for key in deltas.keys():
            Gc_i = objective.derivative(arg_idx[key], *args)
            Gc = np.hstack((Gc, Gc_i)) if Gc.size else Gc_i
        timer.stop("dg/dc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("dg/dc computation ({})".format(keys))

        LHS = np.matmul(Gx, LHS) - Gc
        RHS_1g = np.matmul(Gx, RHS_1g) - g

        LHS = LHS[:, c_idx]  # restrict optimization space
        uA, sA, vtA = np.linalg.svd(LHS, full_matrices=False)

        # find optimal perturbation
        dc1_opt, hit, alpha_c = trust_region_step_exact_svd(
            RHS_1g,
            uA,
            sA,
            vtA.T,
            tr_ratio[0] * np.linalg.norm(c[c_idx]),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

        dc1 = np.zeros_like(c)
        dc1[c_idx] = dc1_opt
        inputs = {}
        idx0 = 0
        for key, idx in deltas.items():
            inputs[key] = dc1[idx0 : idx0 + len(idx)]
            idx0 += len(idx)

        RHS_1f = f + np.matmul(Fc, dc1)

        uJ, sJ, vtJ = np.linalg.svd(Fx, full_matrices=False)

        # apply optimal perturbation
        dx1, hit, alpha_x = trust_region_step_exact_svd(
            RHS_1f,
            uJ,
            sJ,
            vtJ.T,
            tr_ratio[0] * np.linalg.norm(y),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # second order
    if order > 1:

        inds = tuple([arg_idx[key] for key in inputs])
        tangents = tuple([val for val in inputs.values()])
        inds = (0, *inds)
        tangents = (dx1, *tangents)

        # Hessian of primary objective (f) wrt both state vector and input parameters
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        RHS_2f = 0.5 * eq.objective.jvp2(inds, inds, tangents, tangents, *args)
        RHS_2g = np.matmul(Gx, np.matmul(Fx_inv, RHS_2f))
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        # Hessian of secondary objective (g) wrt both state vector and input parameters
        if verbose > 0:
            print("Computing d^2g")
        timer.start("d^2g computation")
        RHS_2g += -0.5 * objective.jvp2(inds, inds, tangents, tangents, *args)
        timer.stop("d^2g computation")
        if verbose > 1:
            timer.disp("d^2g computation")

        # find optimal perturbation
        dc2_opt, hit, alpha_c = trust_region_step_exact_svd(
            RHS_2g,
            uA,
            sA,
            vtA.T,
            tr_ratio[0] * np.linalg.norm(dc1_opt),
            initial_alpha=alpha_c / tr_ratio[1],
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

        dc2 = np.zeros_like(c)
        dc2[c_idx] = dc2_opt
        idx0 = 0
        for val in inputs.values():
            val += dc2[idx0 : idx0 + len(val)]
            idx0 += len(val)

        RHS_2f += np.matmul(Fc, dc2)

        # apply optimal perturbation
        dx2, hit, alpha_x = trust_region_step_exact_svd(
            RHS_2f,
            uJ,
            sJ,
            vtJ.T,
            tr_ratio[0] * np.linalg.norm(y),
            initial_alpha=alpha_x / tr_ratio[1],
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    if order > 2:
        raise ValueError(
            "Higher-order optimizations not yet implemented: {}".format(order)
        )

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    # update input parameters
    for key, dc in inputs.items():
        setattr(eq_new, key, getattr(eq_new, key) + dc)

    # update boundary constraint
    if "Rb_lmn" in inputs or "Zb_lmn" in inputs:
        eq_new.objective.BC_constraint = eq.surface.get_constraint(
            eq_new.R_basis,
            eq_new.Z_basis,
            eq_new.L_basis,
        )

    # update state vector
    dc = dc1 + dc2 + dc3
    dy = dx1 + dx2 + dx3
    eq_new.x = np.copy(eq_new.objective.BC_constraint.recover(y + dy))
    timer.stop("Total perturbation")
    if verbose > 1:
        timer.disp("Total perturbation")
        print("||dc||/||c|| = {}".format(np.linalg.norm(dc) / np.linalg.norm(c[c_idx])))
        print("||dx||/||x|| = {}".format(np.linalg.norm(dy) / np.linalg.norm(y)))

    return eq_new
