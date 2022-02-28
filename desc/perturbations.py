import numpy as np
import warnings
from termcolor import colored

from desc.backend import use_jax, put
from desc.utils import Timer
from desc.compute import arg_order
from desc.optimize.utils import evaluate_quadratic
from desc.optimize.tr_subproblems import trust_region_step_exact_svd

__all__ = ["perturb", "optimal_perturb"]


# TODO: add `weight` input option to scale Jacobian
def perturb(
    eq,
    objective,
    dR=None,
    dZ=None,
    dL=None,
    dRb=None,
    dZb=None,
    dp=None,
    di=None,
    dPsi=None,
    order=2,
    tr_ratio=[0.1, 0.25, 0.5],
    verbose=1,
    copy=True,
):
    """Perturb an Equilibrium with respect to input parameters.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to perturb.
    objective : ObjectiveFunction
        Objective function to satisfy.
    dR, dZ, dL, dRb, dZb, dp, di, dPsi : ndarray or float
        Deltas for perturbations of R, Z, lambda, R_boundary, Z_boundary, pressure,
        rotational transform, and total toroidal magnetic flux.
        Setting to None or zero ignores that term in the expansion.
    order : {0,1,2,3}
        Order of perturbation (0=none, 1=linear, 2=quadratic, etc.)
    tr_ratio : float or array of float
        Radius of the trust region, as a fraction of ||x||.
        Enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||.
        If a scalar, uses the same ratio for all steps. If an array, uses the first
        element for the first step and so on.
    verbose : int
        Level of output.
    copy : bool
        Whether to perturb the input equilibrium (False) or make a copy (True, Default).

    Returns
    -------
    eq_new : Equilibrium
        Perturbed equilibrium.

    """
    if not use_jax:
        warnings.warn(
            colored(
                "Computing perturbations with finite differences can be "
                + "highly innacurate. Consider using JAX for exact derivatives.",
                "yellow",
            )
        )
    if np.isscalar(tr_ratio):
        tr_ratio = tr_ratio * np.ones(order)
    elif len(tr_ratio) < order:
        raise ValueError(
            "Got only {} tr_ratios for order {} perturbations.".format(
                len(tr_ratio), order
            )
        )

    if not objective.built:
        objective.build(eq, verbose=verbose)
    if objective.scalar:  # FIXME: change to num objectives >= num parameters
        raise AttributeError(
            "Cannot perturb with a scalar objective: {}.".format(objective)
        )

    deltas = {}
    if dR is not None and np.any(dR):
        deltas["R_lmn"] = dR
    if dZ is not None and np.any(dZ):
        deltas["Z_lmn"] = dZ
    if dL is not None and np.any(dL):
        deltas["L_lmn"] = dL
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

    # perturbation deltas
    dc = np.concatenate([deltas[arg] for arg in arg_order if arg in deltas.keys()])

    if verbose > 0:
        print("Perturbing {}".format(", ".join(deltas.keys())))

    timer = Timer()
    timer.start("Total perturbation")

    # state vectors
    x = objective.x(eq)
    y = objective.y(eq)

    # perturbation vectors
    dx1 = 0
    dx2 = 0
    dx3 = 0

    # dy/dc*dc
    if objective.dim_c:
        b_idx = np.concatenate([objective.b_idx[arg] for arg in deltas.keys()])
        b_idx.sort(kind="mergesort")
        tangents = np.dot(objective.Ainv, np.dot(np.eye(objective.dim_c)[:, b_idx], dc))
        # FIXME: this can only perturb arguments that are used in the constraints
    else:
        y_idx = np.concatenate([objective.y_idx[arg] for arg in deltas.keys()])
        y_idx.sort(kind="mergesort")
        tangents = np.dot(np.eye(objective.dim_y)[:, y_idx], dc)

    # 1st order
    if order > 0:

        f = objective.compute(y)

        # 1st partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Jy = objective.jac(y)
        Jx = np.dot(Jy, objective.Z)
        RHS1 = f + np.dot(Jy, tangents)
        timer.stop("df computation")
        if verbose > 1:
            timer.disp("df computation")

        if verbose > 0:
            print("Factoring df")
        timer.start("df/dx factorization")
        u, s, vt = np.linalg.svd(Jx, full_matrices=False)
        timer.stop("df/dx factorization")
        if verbose > 1:
            timer.disp("df/dx factorization")

        dx1, hit, alpha = trust_region_step_exact_svd(
            RHS1,
            u,
            s,
            vt.T,
            tr_ratio[0] * np.linalg.norm(x),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # 2nd order
    if order > 1:

        # 2nd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        tangents += np.dot(objective.Z, dx1)
        RHS2 = 0.5 * objective.jvp((tangents, tangents), y)
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
        RHS3 = (1 / 6) * objective.jvp((tangents, tangents, tangents), y)
        RHS3 += objective.jvp((np.dot(objective.Z, dx2), tangents), y)
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

    if order > 3:
        raise ValueError(
            "Higher-order perturbations not yet implemented: {}".format(order)
        )

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    # update perturbation attributes
    for key, value in deltas.items():
        setattr(eq_new, key, getattr(eq_new, key) + value)
    objective.rebuild_constraints(eq_new)

    # update other attributes
    dx = dx1 + dx2 + dx3
    args = objective.unpack_state(x + dx)
    for key, value in args.items():
        if key not in deltas:
            value = put(  # parameter values below threshold are set to 0
                value, np.where(np.abs(value) < 10 * np.finfo(value.dtype).eps)[0], 0
            )
            setattr(eq_new, key, value)

    timer.stop("Total perturbation")
    if verbose > 0:
        print("||dx||/||x|| = {:10.3e}".format(np.linalg.norm(dx) / np.linalg.norm(x)))
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new


def optimal_perturb(
    eq,
    objective_f,
    objective_g,
    dR=False,
    dZ=False,
    dL=False,
    dRb=False,
    dZb=False,
    dp=False,
    di=False,
    dPsi=False,
    opt_subspace=None,
    order=2,
    tr_ratio=[0.1, 0.25],
    cutoff=1e-6,
    verbose=1,
    copy=True,
):
    """Perturb an Equilibrium with respect to input parameters to optimize an objective.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to perturb.
    objective_f : ObjectiveFunction
        Objective function to satisfy.
    objective_g : ObjectiveFunction
        Objective function to optimize.
    dR, dZ, dL, dRb, dZb, dp, di, dPsi : ndarray or bool, optional
        Array of indicies of modes to include in the perturbations of R, Z, lambda,
        R_boundary, Z_boundary, pressure, rotational transform, and total magnetic flux.
        Setting to True (False) includes (excludes) all modes.
    opt_subspace : ndarray, optional
        # TODO: explain!
    order : {0,1,2,3}
        Order of perturbation (0=none, 1=linear, 2=quadratic, etc.)
    tr_ratio : float or array of float
        Radius of the trust region, as a fraction of ||x||.
        Enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||.
        If a scalar, uses the same ratio for all steps. If an array, uses the first
        element for the first step and so on.
    cutoff : float
        Relative cutoff for small singular values in pseudo-inverse.
    verbose : int
        Level of output.
    copy : bool
        Whether to perturb the input equilibrium (False) or make a copy (True, Default).

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
    if np.isscalar(tr_ratio):
        tr_ratio = tr_ratio * np.ones(order)
    elif len(tr_ratio) < order:
        raise ValueError(
            "Got only {} tr_ratios for order {} perturbations.".format(
                len(tr_ratio), order
            )
        )

    if not objective_f.built:
        objective_f.build(eq, verbose=verbose)
    if not objective_g.built:
        objective_g.build(eq, verbose=verbose)

    deltas = {}
    if type(dR) is bool or dR is None:
        if dR is True:
            deltas["R_lmn"] = np.ones((objective_f.dimensions["R_lmn"],), dtype=bool)
    elif np.any(dR):
        deltas["R_lmn"] = dR
    if type(dZ) is bool or dZ is None:
        if dZ is True:
            deltas["Z_lmn"] = np.ones((objective_f.dimensions["Z_lmn"],), dtype=bool)
    elif np.any(dZ):
        deltas["Z_lmn"] = dZ
    if type(dRb) is bool or dRb is None:
        if dRb is True:
            deltas["Rb_lmn"] = np.ones((objective_f.dimensions["Rb_lmn"],), dtype=bool)
    elif np.any(dRb):
        deltas["Rb_lmn"] = dRb
    if type(dZb) is bool or dZb is None:
        if dZb is True:
            deltas["Zb_lmn"] = np.ones((objective_f.dimensions["Zb_lmn"],), dtype=bool)
    elif np.any(dZb):
        deltas["Zb_lmn"] = dZb
    if type(dp) is bool or dp is None:
        if dp is True:
            deltas["p_l"] = np.ones((objective_f.dimensions["p_l"],), dtype=bool)
    elif np.any(dp):
        deltas["p_l"] = dp
    if type(di) is bool or di is None:
        if di is True:
            deltas["i_l"] = np.ones((objective_f.dimensions["i_l"],), dtype=bool)
    elif np.any(di):
        deltas["i_l"] = di
    if type(dPsi) is bool or dPsi is None:
        if dPsi is True:
            deltas["Psi"] = np.ones((objective_f.dimensions["Psi"],), dtype=bool)

    if not len(deltas):
        raise ValueError("At least one input must be a free variable for optimization.")

    if verbose > 0:
        print("Perturbing {}".format(", ".join(deltas.keys())))

    timer = Timer()
    timer.start("Total perturbation")

    # state vectors
    x = objective_f.x(eq)
    y = objective_f.y(eq)

    # parameter vector
    c = np.array([])
    c_idx = np.array([], dtype=bool)
    for key, value in deltas.items():
        c_idx = np.append(c_idx, np.where(value)[0] + c.size)
        c = np.concatenate((c, getattr(eq, key)))

    # optimization subspace matrix
    if opt_subspace is None:
        opt_subspace = np.eye(c.size)[:, c_idx]
    dim_c, dim_opt = opt_subspace.shape

    if dim_c != c.size:
        raise ValueError(
            "Invalid dimension: opt_subspace must have {} rows.".format(c.size)
        )
    if objective_g.dim_f < dim_opt:
        raise ValueError(
            "Cannot perturb {} parameters with {} objectives.".format(
                dim_opt, objective_g.dim_f
            )
        )

    # vector norms
    x_norm = np.linalg.norm(x)
    c_norm = np.linalg.norm(c)
    c_opt = np.dot(c, opt_subspace)
    c_opt_norm = np.linalg.norm(c_opt)

    # perturbation vectors
    dc1 = np.zeros_like(c)
    dc2 = np.zeros_like(c)
    dx1 = np.zeros_like(x)
    dx2 = np.zeros_like(x)

    # dy/dc for f objective
    if objective_f.dim_c:
        b_idx_f = np.concatenate([objective_f.b_idx[arg] for arg in deltas.keys()])
        b_idx_f.sort(kind="mergesort")
        dydc_f = np.dot(objective_f.Ainv, np.eye(objective_f.dim_c)[:, b_idx_f])
        # FIXME: this can only perturb arguments that are used in the constraints
    else:
        y_idx_f = np.concatenate([objective_f.y_idx[arg] for arg in deltas.keys()])
        y_idx_f.sort(kind="mergesort")
        dydc_f = np.eye(objective_f.dim_y)[:, y_idx_f]

    # dy/dc for g objective
    if objective_g.dim_c:
        b_idx_g = np.concatenate([objective_g.b_idx[arg] for arg in deltas.keys()])
        b_idx_g.sort(kind="mergesort")
        dydc_g = np.dot(objective_g.Ainv, np.eye(objective_g.dim_c)[:, b_idx_g])
        # FIXME: this can only perturb arguments that are used in the constraints
    else:
        y_idx_g = np.concatenate([objective_g.y_idx[arg] for arg in deltas.keys()])
        y_idx_g.sort(kind="mergesort")
        dydc_g = np.eye(objective_g.dim_y)[:, y_idx_g]

    # 1st order
    if order > 0:

        f = objective_f.compute(y)
        g = objective_g.compute(y)

        # 1st partial derivatives of f objective wrt both x and c
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Fy = objective_f.jac(y)
        Fx = np.dot(Fy, objective_f.Z)
        Fc = np.dot(Fy, dydc_f)
        Fx_inv = np.linalg.pinv(Fx, rcond=cutoff)
        timer.stop("df computation")
        if verbose > 1:
            timer.disp("df computation")

        # 1st partial derivatives of g objective wrt both x and c
        if verbose > 0:
            print("Computing dg")
        timer.start("dg computation")
        Gy = objective_g.jac(y)
        Gx = np.dot(Gy, objective_g.Z)
        Gc = np.dot(Gy, dydc_g)
        timer.stop("dg computation")
        if verbose > 1:
            timer.disp("dg computation")

        GxFx = np.dot(Gx, Fx_inv)
        LHS = np.dot(GxFx, Fc) - Gc
        RHS_1g = g - np.dot(GxFx, f)

        # restrict to optimization subspace
        LHS_opt = np.dot(LHS, opt_subspace)

        if verbose > 0:
            print("Factoring LHS")
        timer.start("LHS factorization")
        ug, sg, vtg = np.linalg.svd(LHS_opt, full_matrices=False)
        timer.stop("LHS factorization")
        if verbose > 1:
            timer.disp("LHS factorization")

        dc1_opt, bound_hit, alpha = trust_region_step_exact_svd(
            -RHS_1g,
            ug,
            sg,
            vtg.T,
            tr_ratio[0] * c_norm,
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

        dc1 = np.dot(dc1_opt, opt_subspace.T)
        RHS_1f = -f - np.dot(Fc, dc1)
        uf, sf, vtf = np.linalg.svd(Fx, full_matrices=False)

        dx1, _, _ = trust_region_step_exact_svd(
            -RHS_1f,
            uf,
            sf,
            vtf.T,
            tr_ratio[0] * x_norm,
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # 2nd order
    if order > 1:

        # 2nd partial derivatives of f objective wrt both x and c
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        tangents_f = np.dot(objective_f.Z, dx1) + np.dot(dydc_f, dc1)
        RHS_2f = -0.5 * objective_f.jvp((tangents_f, tangents_f), y)
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        # 2nd partial derivatives of g objective wrt both x and c
        if verbose > 0:
            print("Computing d^2g")
        timer.start("d^2g computation")
        tangents_g = np.dot(objective_g.Z, dx1) + np.dot(dydc_g, dc1)
        RHS_2g = 0.5 * objective_g.jvp((tangents_g, tangents_g), y) + np.dot(
            GxFx, RHS_2f
        )
        timer.stop("d^2g computation")
        if verbose > 1:
            timer.disp("d^2g computation")

        dc2_opt, _, _ = trust_region_step_exact_svd(
            -RHS_2g,
            ug,
            sg,
            vtg.T,
            tr_ratio[1] * np.linalg.norm(dc1_opt),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

        dc2 = np.dot(dc2_opt, opt_subspace.T)
        RHS_2f += -np.dot(Fc, dc2)

        dx2, _, _ = trust_region_step_exact_svd(
            -RHS_2f,
            uf,
            sf,
            vtf.T,
            tr_ratio[1] * np.linalg.norm(dx1),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    if order > 2:
        raise ValueError(
            "Higher-order perturbations not yet implemented: {}".format(order)
        )

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    dc = dc1 + dc2
    dc_opt = np.dot(dc, opt_subspace)

    # update perturbation attributes
    idx0 = 0
    for key, value in deltas.items():
        setattr(eq_new, key, getattr(eq_new, key) + dc[idx0 : idx0 + len(value)])
        idx0 += len(value)
    objective_f.rebuild_constraints(eq_new)
    objective_g.rebuild_constraints(eq_new)

    # update other attributes
    dx = dx1 + dx2
    args = objective_f.unpack_state(x + dx)
    for key, value in args.items():
        if key not in deltas:
            value = put(  # parameter values below threshold are set to 0
                value, np.where(np.abs(value) < 10 * np.finfo(value.dtype).eps)[0], 0
            )
            setattr(eq_new, key, value)

    predicted_reduction = -evaluate_quadratic(LHS, -np.dot(RHS_1g.T, LHS), dc)

    timer.stop("Total perturbation")
    if verbose > 0:
        print("||dc||/||c|| = {:10.3e}".format(np.linalg.norm(dc) / c_norm))
        print("||dx||/||x|| = {:10.3e}".format(np.linalg.norm(dx) / x_norm))
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new, predicted_reduction, dc_opt, dc, c_norm, bound_hit
