"""Functions for perturbing equilibria."""

import warnings

import numpy as np
from termcolor import colored

from desc.backend import put, use_jax
from desc.compute import arg_order
from desc.objectives.utils import align_arguments, factorize_linear_constraints
from desc.optimize.tr_subproblems import trust_region_step_exact_svd
from desc.utils import Timer

__all__ = ["get_deltas", "perturb", "optimal_perturb"]


def get_deltas(things1, things2):
    """Compute differences between parameters for perturbations.

    Parameters
    ----------
    things1, things2 : dict
        should be dictionary with keys "surface", "iota", "pressure", etc.
        Values should be objects of the appropriate type (Surface, Profile).
        Finds deltas for a perturbation going from things1 to things2.
        Should have same keys in both dictionaries.

    Returns
    -------
    deltas : dict of ndarray
        deltas to pass in to perturb

    """
    deltas = {}
    assert things1.keys() == things2.keys(), "Must have same keys in both dictionaries"

    if "surface" in things1:
        s1 = things1.pop("surface")
        s2 = things2.pop("surface")
        if s1 is not None and s2 is not None:
            s1 = s1.copy()
            s2 = s2.copy()
            s1.change_resolution(s2.L, s2.M, s2.N)
            if not np.allclose(s2.R_lmn, s1.R_lmn):
                deltas["Rb_lmn"] = s2.R_lmn - s1.R_lmn
            if not np.allclose(s2.Z_lmn, s1.Z_lmn):
                deltas["Zb_lmn"] = s2.Z_lmn - s1.Z_lmn

    for key in ["iota", "pressure", "current"]:
        if key in things1:
            t1 = things1.pop(key)
            t2 = things2.pop(key)
            if t1 is not None and t2 is not None:
                t1 = t1.copy()
                t2 = t2.copy()
                if hasattr(t1, "change_resolution") and hasattr(t2, "basis"):
                    t1.change_resolution(t2.basis.L)
                if not np.allclose(t2.params, t1.params):
                    deltas[key[0] + "_l"] = t2.params - t1.params

    if "Psi" in things1:
        psi1 = things1.pop("Psi")
        psi2 = things2.pop("Psi")
        if psi1 is not None and not np.allclose(psi2, psi1):
            deltas["Psi"] = psi2 - psi1

    assert len(things1) == 0, "get_deltas got an unexpected key: {}".format(
        things1.keys()
    )
    return deltas


def perturb(  # noqa: C901 - FIXME: break this up into simpler pieces
    eq,
    objective,
    constraints,
    deltas,
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
    constraints : tuple of Objective, optional
        List of objectives to be used as constraints during perturbation.
    deltas : dict of ndarray
        Deltas for perturbations. Keys should names of Equilibrium attributes ("p_l",
        "Rb_lmn", "L_lmn" etc.) and values of arrays of desired change in the attribute.
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
                + "highly inaccurate. Consider using JAX for exact derivatives.",
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
    # remove deltas that are zero
    deltas = {key: val for key, val in deltas.items() if np.any(val)}

    if not objective.built:
        objective.build(eq, verbose=verbose)
    for constraint in constraints:
        if not constraint.built:
            constraint.build(eq, verbose=verbose)

    if objective.scalar:  # FIXME: change to num objectives >= num parameters
        raise AttributeError(
            "Cannot perturb with a scalar objective: {}.".format(objective)
        )

    if verbose > 0:
        print("Perturbing {}".format(", ".join(deltas.keys())))

    timer = Timer()
    timer.start("Total perturbation")

    if verbose > 0:
        print("Factorizing linear constraints")
    timer.start("linear constraint factorize")
    xp, A, Ainv, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints, objective.args
    )
    timer.stop("linear constraint factorize")
    if verbose > 1:
        timer.disp("linear constraint factorize")

    # state vector
    x = objective.x(eq)
    x_reduced = project(x)
    x_norm = np.linalg.norm(x_reduced)

    # perturbation vectors
    dx1_reduced = np.zeros_like(x_reduced)
    dx2_reduced = np.zeros_like(x_reduced)
    dx3_reduced = np.zeros_like(x_reduced)

    # tangent vectors
    tangents = np.zeros((objective.dim_x,))
    if "Rb_lmn" in deltas.keys():
        dc = deltas["Rb_lmn"]
        tangents += (
            np.eye(objective.dim_x)[:, objective.x_idx["R_lmn"]] @ Ainv["R_lmn"] @ dc
        )
    if "Zb_lmn" in deltas.keys():
        dc = deltas["Zb_lmn"]
        tangents += (
            np.eye(objective.dim_x)[:, objective.x_idx["Z_lmn"]] @ Ainv["Z_lmn"] @ dc
        )
    # all other perturbations besides the boundary
    other_args = [arg for arg in arg_order if arg not in ["Rb_lmn", "Zb_lmn"]]
    if len([arg for arg in other_args if arg in deltas.keys()]):
        dc = np.concatenate(
            [
                deltas[arg]
                for arg in other_args
                if arg in deltas.keys() and arg in objective.args
            ]
        )
        x_idx = np.concatenate(
            [
                objective.x_idx[arg]
                for arg in other_args
                if arg in deltas.keys() and arg in objective.args
            ]
        )
        x_idx.sort(kind="mergesort")
        tangents += np.eye(objective.dim_x)[:, x_idx] @ dc

    # 1st order
    if order > 0:

        if (weight is None) or (weight == "auto"):
            w = np.ones((objective.dim_x,))
            if weight == "auto" and (("p_l" in deltas) or ("i_l" in deltas)):
                w[objective.x_idx["R_lmn"]] = (
                    abs(eq.R_basis.modes[:, :2]).sum(axis=1) + 1
                )
                w[objective.x_idx["Z_lmn"]] = (
                    abs(eq.Z_basis.modes[:, :2]).sum(axis=1) + 1
                )
                w[objective.x_idx["L_lmn"]] = (
                    abs(eq.L_basis.modes[:, :2]).sum(axis=1) + 1
                )
            weight = w
        weight = np.atleast_1d(weight)
        assert (
            len(weight) == objective.dim_x
        ), "Size of weight supplied to perturbation does not match objective.dim_x."
        if weight.ndim == 1:
            weight = weight[unfixed_idx]
            weight = np.diag(weight)
        else:
            weight = weight[unfixed_idx, unfixed_idx]
        W = Z.T @ weight @ Z
        scale_inv = W
        scale = np.linalg.inv(scale_inv)

        f = objective.compute(x)

        # 1st partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Jx = objective.jac(x)
        Jx_reduced = Jx[:, unfixed_idx] @ Z @ scale
        RHS1 = f + objective.jvp(tangents, x)
        timer.stop("df computation")
        if verbose > 1:
            timer.disp("df computation")

        if verbose > 0:
            print("Factoring df")
        timer.start("df/dx factorization")
        u, s, vt = np.linalg.svd(Jx_reduced, full_matrices=False)
        timer.stop("df/dx factorization")
        if verbose > 1:
            timer.disp("df/dx factorization")

        dx1_h, hit, alpha = trust_region_step_exact_svd(
            RHS1,
            u,
            s,
            vt.T,
            tr_ratio[0] * np.linalg.norm(scale_inv @ x_reduced),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )
        dx1_reduced = scale @ dx1_h
        dx1 = recover(dx1_reduced) - xp

    # 2nd order
    if order > 1:

        # 2nd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        tangents += dx1
        RHS2 = 0.5 * objective.jvp((tangents, tangents), x)
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
        dx2_reduced = scale @ dx2_h
        dx2 = recover(dx2_reduced) - xp

    # 3rd order
    if order > 2:

        # 3rd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^3f")
        timer.start("d^3f computation")
        RHS3 = (1 / 6) * objective.jvp((tangents, tangents, tangents), x)
        RHS3 += objective.jvp((dx2, tangents), x)
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
        dx3_reduced = scale @ dx3_h

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
    for constraint in constraints:
        constraint.update_target(eq_new)
    xp, A, Ainv, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints, objective.args
    )

    # update other attributes
    dx_reduced = dx1_reduced + dx2_reduced + dx3_reduced
    x_new = recover(x_reduced + dx_reduced)
    args = objective.unpack_state(x_new)
    for key, value in args.items():
        if key not in deltas:
            value = put(  # parameter values below threshold are set to 0
                value, np.where(np.abs(value) < 10 * np.finfo(value.dtype).eps)[0], 0
            )
            # don't set nonexistent profile (values are empty ndarrays)
            if not (key == "c_l" or key == "i_l") or value.size:
                setattr(eq_new, key, value)

    timer.stop("Total perturbation")
    if verbose > 0:
        print("||dx||/||x|| = {:10.3e}".format(np.linalg.norm(dx_reduced) / x_norm))
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new


def optimal_perturb(  # noqa: C901 - FIXME: break this up into simpler pieces
    eq,
    objective,
    constraints,
    order=2,
    tr_ratio=[0.1, 0.25],
    cutoff=None,
    verbose=1,
    copy=True,
):
    """Perturb an Equilibrium with respect to input parameters to optimize an objective.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to perturb.
    objective : ObjectiveFunction
        Objective function to optimize.
    constraints : tuple of Objective, optional
        List of objectives to be used as constraints during optimization.
    order : {0,1,2}
        Order of perturbation (0=none, 1=linear, 2=quadratic, etc.)
    tr_ratio : float or array of float
        Radius of the trust region, as a fraction of ||x||.
        Enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||.
        If a scalar, uses the same ratio for all steps. If an array, uses the first
        element for the first step and so on.
    cutoff : float
        Relative cutoff for small singular values in pseudo-inverse.
        Default is np.finfo(A.dtype).eps*max(A.shape) where A is the Jacobian matrix.
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
                + "highly inaccurate. Consider using JAX for exact derivatives.",
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
    for constraint in constraints:
        if not constraint.built:
            constraint.build(eq, verbose=verbose)

    _, _, _, _, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints, objective.args + ["Rb_lmn", "Zb_lmn"]  # FIXME: generalize
    )

    if False:
        raise ValueError("At least one input must be a free variable for optimization.")

    if verbose > 0:
        print("Perturbing which variables?")  # FIXME: fix

    timer = Timer()
    timer.start("Total perturbation")

    # perturbation vectors
    dx1_reduced = 0
    dx2_reduced = 0

    if verbose > 0:
        print("Number of parameters: {}".format(Z.shape[-1]))
        print("Number of objectives: {}".format(objective.dim_f))

    # 1st order
    if order > 0:

        x_obj = objective.x(eq)
        f = objective.compute(x_obj)

        # Jacobian matrix df/dx
        if verbose > 0:
            print("Computing df/dx")
        timer.start("df/dx computation")
        Fx = objective.jac(x_obj)
        timer.stop("df/dx computation")
        if verbose > 1:
            timer.disp("df/dx computation")

        # FIXME: generalize arguments
        Fx, x_idx = align_arguments(Fx, objective, ["Rb_lmn", "Zb_lmn"])
        Fx_reduced = np.dot(Fx[:, unfixed_idx], Z)

        x = np.zeros((Fx.shape[-1],))
        for key, value in x_idx.items():
            x[value] = getattr(eq, key)
        x_reduced = project(x)
        x_reduced_norm = np.linalg.norm(x_reduced)

        if verbose > 0:
            print("Factoring LHS")
        timer.start("LHS factorization")
        u, s, vt = np.linalg.svd(Fx_reduced, full_matrices=False)
        timer.stop("LHS factorization")
        if verbose > 1:
            timer.disp("LHS factorization")

        # TODO: add scaling, see optimizer example
        dx1_reduced, bound_hit, alpha = trust_region_step_exact_svd(
            f,
            u,
            s,
            vt.T,
            tr_ratio[0] * x_reduced_norm,
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # 2nd order
    if order > 1:
        pass

    if order > 2:
        raise ValueError(
            "Higher-order perturbations not yet implemented: {}".format(order)
        )

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    # update equilibrium attributes
    dx_reduced = dx1_reduced + dx2_reduced
    x_new = recover(x_reduced + dx_reduced)
    for key, value in x_idx.items():
        if len(value):
            setattr(eq_new, key, x_new[value])

    timer.stop("Total perturbation")
    if verbose > 0:
        print("||dx||/||x|| = {:10.3e}".format(
            np.linalg.norm(dx_reduced) / x_reduced_norm))
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new
