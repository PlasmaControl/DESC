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
    dp=None,
    di=None,
    dPsi=None,
    dRb=None,
    dZb=None,
    order=2,
    tr_ratio=0.1,
    weight="auto",
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
    dR, dZ, dL, dp, di, dPsi, dRb, dZb : ndarray or float
        Deltas for perturbations of R, Z, lambda, pressure, rotational transform,
        total toroidal magnetic flux, R_boundary, and Z_boundary.
        Setting to None or zero ignores that term in the expansion.
    order : {0,1,2,3}
        Order of perturbation (0=none, 1=linear, 2=quadratic, etc.)
    tr_ratio : float or array of float
        Radius of the trust region, as a fraction of ||x||.
        Enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||.
        If a scalar, uses the same ratio for all steps. If an array, uses the first
        element for the first step and so on.
    weight : ndarray, "auto", or None, optional
        1d or 2d array for weighted least squares. 1d arrays are turned into diagonal
        matrices. Default is to weight by (mode number)**2. None applies no weighting.
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
    if dp is not None and np.any(dp):
        deltas["p_l"] = dp
    if di is not None and np.any(di):
        deltas["i_l"] = di
    if dPsi is not None and np.any(dPsi):
        deltas["Psi"] = dPsi
    if dRb is not None and np.any(dRb):
        deltas["Rb_lmn"] = dRb
    if dZb is not None and np.any(dZb):
        deltas["Zb_lmn"] = dZb

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

    # tangent vectors
    tangents = np.zeros((objective.dim_y,))
    if "Rb_lmn" in deltas.keys():
        dc = deltas["Rb_lmn"]
        tangents += (
            np.eye(objective.dim_y)[:, objective.y_idx["R_lmn"]]
            @ objective.Ainv["R_lmn"]
            @ dc
        )
    if "Zb_lmn" in deltas.keys():
        dc = deltas["Zb_lmn"]
        tangents += (
            np.eye(objective.dim_y)[:, objective.y_idx["Z_lmn"]]
            @ objective.Ainv["Z_lmn"]
            @ dc
        )
    # all other perturbations besides the boundary
    if len([arg for arg in arg_order if arg in deltas.keys()]):
        dc = np.concatenate([deltas[arg] for arg in arg_order if arg in deltas.keys()])
        y_idx = np.concatenate(
            [objective.y_idx[arg] for arg in arg_order if arg in deltas.keys()]
        )
        y_idx.sort(kind="mergesort")
        tangents += np.eye(objective.dim_y)[:, y_idx] @ dc

    # 1st order
    if order > 0:

        if weight == "auto" and (("p_l" in deltas) or ("i_l" in deltas)):
            weight = (
                np.concatenate(
                    [
                        abs(eq.R_basis.modes[:, :2]).sum(axis=1),
                        abs(eq.Z_basis.modes[:, :2]).sum(axis=1),
                        abs(eq.L_basis.modes[:, :2]).sum(axis=1),
                    ]
                )
                + 1
            )
        elif (weight is None) or (weight == "auto"):
            weight = np.ones(
                eq.R_basis.num_modes + eq.Z_basis.num_modes + eq.L_basis.num_modes
            )

        weight = np.pad(weight, len(y), constant_values=1)
        weight = np.atleast_1d(weight)
        weight = weight[objective._unfixed_idx]
        if weight.ndim == 1:
            weight = np.diag(weight)
        W = objective.Z.T @ weight @ objective.Z
        scale_inv = W
        scale = np.linalg.inv(scale_inv)

        f = objective.compute(y)

        # 1st partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Jy = objective.jac(y)
        Jx = np.dot(Jy[:, objective._unfixed_idx], objective.Z) @ scale
        RHS1 = f + objective.jvp(tangents, y)
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

        dx1_h, hit, alpha = trust_region_step_exact_svd(
            RHS1,
            u,
            s,
            vt.T,
            tr_ratio[0] * np.linalg.norm(scale_inv @ x),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )
        dx1 = scale @ dx1_h
        dy1 = objective.recover(dx1) - objective.y0

    # 2nd order
    if order > 1:

        # 2nd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        tangents += dy1
        RHS2 = 0.5 * objective.jvp((tangents, tangents), y)
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        dx2_h, hit, alpha = trust_region_step_exact_svd(
            RHS2,
            u,
            s,
            vt.T,
            tr_ratio[1] * np.linalg.norm(dx1_h),
            initial_alpha=alpha / tr_ratio[1],
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )
        dx2 = scale @ dx2_h
        dy2 = objective.recover(dx2) - objective.y0

    # 3rd order
    if order > 2:

        # 3rd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^3f")
        timer.start("d^3f computation")
        RHS3 = (1 / 6) * objective.jvp((tangents, tangents, tangents), y)
        RHS3 += objective.jvp((dy2, tangents), y)
        timer.stop("d^3f computation")
        if verbose > 1:
            timer.disp("d^3f computation")

        dx3_h, hit, alpha = trust_region_step_exact_svd(
            RHS3,
            u,
            s,
            vt.T,
            tr_ratio[2] * np.linalg.norm(dx2_h),
            initial_alpha=alpha / tr_ratio[2],
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )
        dx3 = scale @ dx3_h
        dy3 = objective.recover(dx3) - objective.y0

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
    dp=False,
    di=False,
    dPsi=False,
    dRb=False,
    dZb=False,
    subspace=None,
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
    dR, dZ, dL, dp, di, dPsi, dRb, dZb : ndarray or bool, optional
        Array of indicies of modes to include in the perturbations of R, Z, lambda,
        pressure, rotational transform, total magnetic flux, R_boundary, and Z_boundary.
        Setting to True (False) includes (excludes) all modes.
    subspace : ndarray, optional
        Transform matrix to give a subspace from the full parameter space.
        Can be used to enforce custom optimization constraints.
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

    if not np.atleast_1d(objective_f.y(eq) == objective_g.y(eq)).all():
        raise ValueError("Objectives must have the same state vectors (y).")

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
    if type(dL) is bool or dL is None:
        if dL is True:
            deltas["L_lmn"] = np.ones((objective_f.dimensions["L_lmn"],), dtype=bool)
    elif np.any(dL):
        deltas["L_lmn"] = dL
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
    if subspace is None:
        subspace = np.eye(c.size)[:, c_idx]
    dim_c, dim_opt = subspace.shape

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

    # perturbation vectors
    dc1 = np.zeros_like(c)
    dc2 = np.zeros_like(c)
    dx1 = np.zeros_like(x)
    dx2 = np.zeros_like(x)

    # dy/dx
    dydx = np.dot(np.eye(objective_f.dim_y)[:, objective_f._unfixed_idx], objective_f.Z)

    # dy/dc
    dydc = np.zeros((objective_f.dim_y, 0))
    if len([arg for arg in arg_order if arg in deltas.keys()]):
        y_idx = np.concatenate(
            [objective_f.y_idx[arg] for arg in arg_order if arg in deltas.keys()]
        )
        y_idx.sort(kind="mergesort")
        dydc = np.eye(objective_f.dim_y)[:, y_idx]
    if "Rb_lmn" in deltas.keys():
        dydRb = (
            np.eye(objective_f.dim_y)[:, objective_f.y_idx["R_lmn"]]
            @ objective_f.Ainv["R_lmn"]
        )
        dydc = np.hstack((dydc, dydRb))
    if "Zb_lmn" in deltas.keys():
        dydZb = (
            np.eye(objective_f.dim_y)[:, objective_f.y_idx["Z_lmn"]]
            @ objective_f.Ainv["Z_lmn"]
        )
        dydc = np.hstack((dydc, dydZb))

    # 1st order
    if order > 0:

        f = objective_f.compute(y)
        g = objective_g.compute(y)

        # 1st partial derivatives of f objective wrt both x and c
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Fy = objective_f.jac(y)
        Fx = np.dot(Fy[:, objective_f._unfixed_idx], objective_f.Z)
        Fc = np.dot(Fy, dydc)
        Fx_inv = np.linalg.pinv(Fx, rcond=cutoff)
        timer.stop("df computation")
        if verbose > 1:
            timer.disp("df computation")

        # 1st partial derivatives of g objective wrt both x and c
        if verbose > 0:
            print("Computing dg")
        timer.start("dg computation")
        Gy = objective_g.jac(y)
        Gx = np.dot(Gy[:, objective_g._unfixed_idx], objective_g.Z)
        Gc = np.dot(Gy, dydc)
        timer.stop("dg computation")
        if verbose > 1:
            timer.disp("dg computation")

        GxFx = np.dot(Gx, Fx_inv)
        LHS = np.dot(GxFx, Fc) - Gc
        RHS_1g = g - np.dot(GxFx, f)

        # restrict to optimization subspace
        LHS_opt = np.dot(LHS, subspace)

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

        dc1 = np.dot(dc1_opt, subspace.T)
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
        tangents_f = np.dot(dydx, dx1) + np.dot(dydc, dc1)
        RHS_2f = -0.5 * objective_f.jvp((tangents_f, tangents_f), y)
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        # 2nd partial derivatives of g objective wrt both x and c
        if verbose > 0:
            print("Computing d^2g")
        timer.start("d^2g computation")
        tangents_g = np.dot(dydx, dx1) + np.dot(dydc, dc1)
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

        dc2 = np.dot(dc2_opt, subspace.T)
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
    dc_opt = np.dot(dc, subspace)

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
