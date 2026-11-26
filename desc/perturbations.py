"""Functions for perturbing equilibria."""

import warnings

import numpy as np
from termcolor import colored

from desc.backend import jnp, put, use_jax
from desc.compute import profile_names
from desc.objectives import (
    ObjectiveFunction,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.optimize import LinearConstraintProjection
from desc.optimize.tr_subproblems import trust_region_step_exact_svd
from desc.optimize.utils import (
    compute_jac_scale,
    evaluate_quadratic_form_jac,
    scatter_rows,
)
from desc.utils import Timer, warnif

__all__ = ["get_deltas", "perturb", "optimal_perturb"]


def get_deltas(things1, things2):  # noqa: C901
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
    msg = (
        " has lower resolution than what is being perturbed."
        " This might clip non-zero coefficients"
        " and cause dimension issues later. "
    )
    if "surface" in things1:
        s1 = things1.pop("surface")
        s2 = things2.pop("surface")
        if s1 is not None and s2 is not None:
            s1 = s1.copy()
            s2 = s2.copy()
            warnif(s1.M > s2.M or s1.N > s2.N, msg="The target surface" + msg)
            s1.change_resolution(s2.L, s2.M, s2.N)
            if not jnp.allclose(s2.R_lmn, s1.R_lmn):
                deltas["Rb_lmn"] = s2.R_lmn - s1.R_lmn
            if not jnp.allclose(s2.Z_lmn, s1.Z_lmn):
                deltas["Zb_lmn"] = s2.Z_lmn - s1.Z_lmn

    if "axis" in things1:
        a1 = things1.pop("axis")
        a2 = things2.pop("axis")
        if a1 is not None and a2 is not None:
            a1 = a1.copy()
            a2 = a2.copy()
            warnif(a1.N > a2.N, msg="The target axis" + msg)
            a1.change_resolution(a2.N)
            if not jnp.allclose(a2.R_n, a1.R_n):
                deltas["Ra_n"] = a2.R_n - a1.R_n
            if not jnp.allclose(a2.Z_n, a1.Z_n):
                deltas["Za_n"] = a2.Z_n - a1.Z_n

    for key, val in profile_names.items():
        if key in things1:
            t1 = things1.pop(key)
            t2 = things2.pop(key)
            if t1 is not None and t2 is not None:
                t1 = t1.copy()
                t2 = t2.copy()
                if hasattr(t1, "change_resolution") and hasattr(t2, "basis"):
                    warnif(
                        t1.basis.L > t2.basis.L, msg=f"The target {key} profile" + msg
                    )
                    t1.change_resolution(t2.basis.L)
                if not jnp.allclose(t2.params, t1.params):
                    deltas[val] = t2.params - t1.params

    if "Psi" in things1:
        psi1 = things1.pop("Psi")
        psi2 = things2.pop("Psi")
        if psi1 is not None and not jnp.allclose(psi2, psi1):
            deltas["Psi"] = psi2 - psi1

    assert len(things1) == 0, "get_deltas got an unexpected key: {}".format(
        things1.keys()
    )
    return deltas


def perturb(  # noqa: C901
    eq,
    objective,
    constraints,
    deltas,
    order=2,
    tr_ratio=0.1,
    weight="auto",
    include_f=True,
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
    include_f : bool, optional
        Whether to include the 0th order objective residual in the perturbation
        equation. Including this term can improve force balance if the perturbation
        step is large, but can result in too large a step if the perturbation is small.
    verbose : int
        Level of output.
    copy : bool
        Whether to perturb the input equilibrium (False) or make a copy (True, Default).

    Returns
    -------
    eq_new : Equilibrium
        Perturbed equilibrium.

    """
    is_linear_proj = isinstance(objective, LinearConstraintProjection)
    if not use_jax:
        warnings.warn(
            colored(
                "Computing perturbations with finite differences can be "
                + "highly inaccurate. Consider using JAX for exact derivatives.",
                "yellow",
            )
        )
    if jnp.isscalar(tr_ratio):
        tr_ratio = tr_ratio * jnp.ones(order)
    elif len(tr_ratio) < order:
        raise ValueError(
            "Got only {} tr_ratios for order {} perturbations.".format(
                len(tr_ratio), order
            )
        )

    if is_linear_proj and constraints is not None:
        raise ValueError(
            "If a LinearConstraintProjection is passed, "
            "no constraints should be passed. Passed constraints:"
            f"{constraints}."
        )
    # remove deltas that are zero
    deltas = {key: val for key, val in deltas.items() if jnp.any(val)}

    # make sure things are at least 1D for jnp.concatenate later
    # in case only a single delta is being passed
    for key in deltas.keys():
        deltas[key] = jnp.atleast_1d(deltas[key])

    if not objective.built:
        objective.build(verbose=verbose)

    if is_linear_proj:
        obj = objective
        objective = obj._objective
        constraints = obj._constraint.objectives
        constraint = obj._constraint
        dim_f = obj.dim_f
    else:
        constraints = maybe_add_self_consistency(eq, constraints)
        constraint = ObjectiveFunction(constraints)
        constraint.build(verbose=verbose)
        dim_f = objective.dim_f
    warnif(
        dim_f < (objective.dim_x - constraint.dim_f),
        UserWarning,
        "Perturbing an underdetermined system may give bad results",
    )

    if verbose > 0:
        print("Perturbing {}".format(", ".join(deltas.keys())))

    timer = Timer()
    timer.start("Total perturbation")

    if verbose > 0:
        print("Factorizing linear constraints")
    timer.start("linear constraint factorize")
    if is_linear_proj:
        xp, D, unfixed_idx, project, recover = (
            obj._xp,
            obj._D,
            obj._unfixed_idx,
            obj._project,
            obj._recover,
        )
    else:
        xp, _, _, _, D, unfixed_idx, project, recover, *_ = (
            factorize_linear_constraints(objective, constraint)
        )
    # dx/dx_reduced, ie D @ Z scattered into the full state vector
    dxdx_reduced = recover.feasible_tangents
    timer.stop("linear constraint factorize")
    if verbose > 1:
        timer.disp("linear constraint factorize")

    # state vector
    x = objective.x(eq)
    x_reduced = project(x)

    x_norm = jnp.linalg.norm(x_reduced)

    # perturbation vectors
    dx1_reduced = jnp.zeros_like(x_reduced)
    dx2_reduced = jnp.zeros_like(x_reduced)
    dx3_reduced = jnp.zeros_like(x_reduced)

    # tangent vector: the change of the full state vector made by the perturbation at
    # fixed reduced coordinates. The perturbed state is formed below as
    # recover(x_reduced + dx_reduced) with the particular solution of the linear
    # constraints for the new inputs, which is the minimum norm solution of all of them
    # together (eg. the axis self-consistency ties R_lmn at rho=0 to Ra_n), so the
    # derivatives must be taken along the change of that particular solution. A tangent
    # from the pseudo-inverse of one constraint alone (eg. BoundaryRSelfConsistency for
    # Rb_lmn) differs from it by an amount linear in the perturbation, which leaves a
    # first order force error that no order of the perturbation removes.
    eq_tmp = eq.copy()
    for key, value in deltas.items():
        setattr(eq_tmp, key, getattr(eq_tmp, key) + value)
    if is_linear_proj:
        obj.update_constraint_target(eq_tmp)
        tangents = obj._recover(x_reduced) - x
        obj.update_constraint_target(eq)
        # the update replaces (and frees) the arrays taken from obj above
        D, dxdx_reduced = obj._D, obj._recover.feasible_tangents
    else:
        for con in constraints:
            if hasattr(con, "update_target"):
                con.update_target(eq_tmp)
        constraint_tmp = ObjectiveFunction(constraints)
        constraint_tmp.build(verbose=0)
        _, _, _, _, _, _, _, recover_tmp, *_ = factorize_linear_constraints(
            objective, constraint_tmp
        )
        tangents = recover_tmp(x_reduced) - x
        for con in constraints:
            if hasattr(con, "update_target"):
                con.update_target(eq)
    # the perturbed parameters change by exactly deltas, also any that no constraint
    # fixes (those are set directly below, not by recover)
    for key, value in deltas.items():
        if key in eq.optimizable_params:
            tangents = put(tangents, eq.x_idx[key], value)

    # 1st order
    if order > 0:
        if weight is None or (isinstance(weight, str) and weight == "auto"):
            w = jnp.ones((eq.dim_x,))
            if weight == "auto" and (("p_l" in deltas) or ("i_l" in deltas)):
                w = put(
                    w,
                    eq.x_idx["R_lmn"],
                    (abs(eq.R_basis.modes[:, :2]).sum(axis=1) + 1),
                )
                w = put(
                    w,
                    eq.x_idx["Z_lmn"],
                    (abs(eq.Z_basis.modes[:, :2]).sum(axis=1) + 1),
                )
                w = put(
                    w,
                    eq.x_idx["L_lmn"],
                    (abs(eq.L_basis.modes[:, :2]).sum(axis=1) + 1),
                )
            weight = w
        weight = jnp.atleast_1d(jnp.asarray(weight))
        assert (
            len(weight) == objective.dim_x
        ), "Size of weight supplied to perturbation does not match objective.dim_x."
        if weight.ndim == 1:
            weight = weight[unfixed_idx]
            weight = jnp.diag(weight)
        else:
            weight = weight[unfixed_idx][:, unfixed_idx]
        Z = (1 / D)[unfixed_idx, None] * dxdx_reduced[unfixed_idx]
        W = Z.T @ weight @ Z
        scale_inv = W
        scale = jnp.linalg.inv(scale_inv)

        # 1st partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Jx = objective.jac_scaled_error(x)
        Jx_reduced = Jx @ dxdx_reduced @ scale
        RHS1 = objective.jvp_scaled(tangents, x)
        if include_f:
            f = objective.compute_scaled_error(x)
            RHS1 += f
        timer.stop("df computation")
        if verbose > 1:
            timer.disp("df computation")

        if verbose > 0:
            print("Factoring df")
        timer.start("df/dx factorization")
        u, s, vt = jnp.linalg.svd(Jx_reduced, full_matrices=False)
        timer.stop("df/dx factorization")
        if verbose > 1:
            timer.disp("df/dx factorization")

        dx1_h, _, alpha = trust_region_step_exact_svd(
            RHS1,
            u,
            s,
            vt.T,
            tr_ratio[0] * jnp.linalg.norm(scale_inv @ x_reduced),
            initial_alpha=0.0,
            rtol=0.01,
            max_iter=10,
        )
        dx1_reduced = scale @ dx1_h
        # change in the full state vector in unscaled units
        dx1 = dxdx_reduced @ dx1_reduced

    # 2nd order
    if order > 1:
        # 2nd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        tangents += dx1
        RHS2 = 0.5 * objective.jvp_scaled((tangents, tangents), x)
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        dx2_h, _, alpha = trust_region_step_exact_svd(
            RHS2,
            u,
            s,
            vt.T,
            tr_ratio[1] * jnp.linalg.norm(dx1_h),
            initial_alpha=alpha / tr_ratio[1],
            rtol=0.01,
            max_iter=10,
        )
        dx2_reduced = scale @ dx2_h
        dx2 = dxdx_reduced @ dx2_reduced

    # 3rd order
    if order > 2:
        # 3rd partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing d^3f")
        timer.start("d^3f computation")
        RHS3 = (1 / 6) * objective.jvp_scaled((tangents, tangents, tangents), x)
        RHS3 += objective.jvp_scaled((dx2, tangents), x)
        timer.stop("d^3f computation")
        if verbose > 1:
            timer.disp("d^3f computation")

        dx3_h, _, alpha = trust_region_step_exact_svd(
            RHS3,
            u,
            s,
            vt.T,
            tr_ratio[2] * jnp.linalg.norm(dx2_h),
            initial_alpha=alpha / tr_ratio[2],
            rtol=0.01,
            max_iter=10,
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

    if is_linear_proj:
        obj.update_constraint_target(eq_new)
        recover = obj._recover
    else:
        for con in constraints:
            if hasattr(con, "update_target"):
                con.update_target(eq_new)
        constraint = ObjectiveFunction(constraints)
        constraint.build(verbose=verbose)
        _, _, _, _, _, _, _, recover, *_ = factorize_linear_constraints(
            objective, constraint
        )

    # update other attributes
    dx_reduced = dx1_reduced + dx2_reduced + dx3_reduced
    x_new = recover(x_reduced + dx_reduced)
    params = objective.unpack_state(x_new, False)[0]
    for key, value in params.items():
        if key not in deltas:
            value = put(  # parameter values below threshold are set to 0
                value, jnp.where(jnp.abs(value) < 10 * jnp.finfo(value.dtype).eps)[0], 0
            )
            # don't set nonexistent profile (values are empty ndarrays)
            if value.size:
                setattr(eq_new, key, value)

    timer.stop("Total perturbation")
    if verbose > 0:
        print("||dx||/||x|| = {:10.3e}".format(jnp.linalg.norm(dx_reduced) / x_norm))
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new


def optimal_perturb(  # noqa: C901
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
    cutoff=None,
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
        Array of indices of modes to include in the perturbations of R, Z, lambda,
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
        element for the first step and so on. Note that ||X|| uses a scaled norm,
        weighted by the jacobian.
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
    if jnp.isscalar(tr_ratio):
        tr_ratio = tr_ratio * jnp.ones(order)
    elif len(tr_ratio) < order:
        raise ValueError(
            "Got only {} tr_ratios for order {} perturbations.".format(
                len(tr_ratio), order
            )
        )

    if not objective_f.built:
        objective_f.build(verbose=verbose)
    if not objective_g.built:
        objective_g.build(verbose=verbose)

    argmap = {
        "R_lmn": dR,
        "Z_lmn": dZ,
        "L_lmn": dL,
        "p_l": dp,
        "i_l": di,
        "Psi": dPsi,
        "Rb_lmn": dRb,
        "Zb_lmn": dZb,
    }
    deltas = {}
    for name, arg in argmap.items():
        if type(arg) is bool or arg is None:
            if arg is True:
                deltas[name] = jnp.ones((eq.dimensions[name],), dtype=bool)
        elif jnp.any(arg):
            deltas[name] = arg

    if not len(deltas):
        raise ValueError("At least one input must be a free variable for optimization.")

    if verbose > 0:
        print("Perturbing {}".format(", ".join(deltas.keys())))

    timer = Timer()
    timer.start("Total perturbation")

    # parameter vector
    c = jnp.array([])
    c_idx = jnp.array([], dtype=bool)
    for key, value in deltas.items():
        c_idx = jnp.append(c_idx, jnp.where(value)[0] + c.size)
        c = jnp.concatenate((c, getattr(eq, key)))

    # optimization subspace matrix
    if subspace is None:
        subspace = jnp.eye(c.size)[:, c_idx]
    dim_c, dim_opt = subspace.shape

    if dim_c != c.size:
        raise ValueError(
            "Invalid dimension: opt_subspace must have {} rows.".format(c.size)
        )
    if verbose > 0:
        print("Number of parameters: {}".format(dim_opt))
        print("Number of objectives: {}".format(objective_g.dim_f))

    constraints = get_fixed_boundary_constraints(eq=eq)
    constraints = maybe_add_self_consistency(eq, constraints)
    constraint = ObjectiveFunction(constraints)
    constraint.build(verbose=verbose)

    _, _, _, _, D, unfixed_idx, project, recover, *_ = factorize_linear_constraints(
        objective_f, constraint
    )

    # state vector
    xf = objective_f.x(eq)
    xg = objective_g.x(eq)

    x_reduced = project(xf)

    # perturbation vectors
    dc1 = 0
    dc2 = 0
    dx1_reduced = 0
    dx2_reduced = 0

    # dx/dx_reduced, ie D @ Z scattered into the full state vector
    dxdx_reduced = recover.feasible_tangents

    # dx/dc: the change of the full state vector per unit change of c at fixed reduced
    # coordinates, as in the tangents of perturb. The state is updated at the end with
    # recover(x_reduced + dx_reduced), whose particular solution of the linear
    # constraints is taken for the new c, so the derivatives must be taken along the
    # change of that particular solution. For a parameter fixed by the constraints (eg.
    # Rb_lmn) that is the minimum norm change of the unfixed ones that keeps all the
    # constraints satisfied together, eg. R_lmn and Ra_n through both the boundary and
    # the axis self consistency. Unfixed parameters in c (eg. R_lmn) only change by c.
    c_x_idx = np.concatenate([np.asarray(eq.x_idx[key]) for key in deltas])
    c_fixed = ~np.isin(c_x_idx, unfixed_idx)
    dxdc = scatter_rows(eq.dim_x, c_x_idx, jnp.eye(c.size))
    if np.any(c_fixed):
        # a feasible change: c changes and the other parameters solve the constraints
        # that involve them, which have targets that do not depend on c. The ones on c
        # alone are the constraints that fix c, whose targets change with c.
        A = np.asarray(constraint.jac_unscaled(jnp.zeros(constraint.dim_x)))
        other_idx = np.setdiff1d(np.arange(eq.dim_x), c_x_idx[c_fixed])
        rows = np.any(A[:, other_idx] != 0, axis=1)
        dxdc_other = -jnp.linalg.pinv(A[rows][:, other_idx]) @ (
            A[rows][:, c_x_idx] * c_fixed
        )
        dxdc += scatter_rows(eq.dim_x, other_idx, dxdc_other)
        # the particular solution is the feasible change with no component along the
        # null space of the constraints (in the variables scaled by D)
        dxdc -= c_fixed * (dxdx_reduced @ (dxdx_reduced.T @ (dxdc / D[:, None] ** 2)))

    # 1st order
    if order > 0:
        f = objective_f.compute_scaled_error(xf)
        g = objective_g.compute_scaled_error(xg)

        # 1st partial derivatives of f objective wrt x
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Fx = objective_f.jac_scaled_error(xf)
        timer.stop("df computation")
        if verbose > 1:
            timer.disp("df computation")

        # 1st partial derivatives of g objective wrt x
        if verbose > 0:
            print("Computing dg")
        timer.start("dg computation")
        Gx = objective_g.jac_scaled_error(xg)
        timer.stop("dg computation")
        if verbose > 1:
            timer.disp("dg computation")

        # projections onto optimization space
        Fx_reduced = Fx @ dxdx_reduced
        Gx_reduced = Gx @ dxdx_reduced
        Fc = Fx @ dxdc
        Gc = Gx @ dxdc

        # some scaling to improve conditioning and rescale trust region
        wf, _ = compute_jac_scale(Fx_reduced)
        wg, _ = compute_jac_scale(Gx_reduced)
        wx = wf + wg
        Fxh = Fx_reduced * wx
        Gxh = Gx_reduced * wx
        if cutoff is None:
            cutoff = jnp.finfo(Fx_reduced.dtype).eps * max(Fx_reduced.shape)
        uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
        sf += sf[-1]  # add a tiny bit of regularization
        sfi = jnp.where(sf < cutoff * sf[0], 0, 1 / sf)
        Fxh_inv = vtf.T @ (sfi[..., jnp.newaxis] * uf.T)

        GxFx = Gxh @ Fxh_inv
        LHS = GxFx @ Fc - Gc
        RHS_1g = g - GxFx @ f

        # scaling for c
        # approx dc/dx at const f
        dcdx_f = jnp.linalg.lstsq(Fc, Fx_reduced, rcond=None)[0]
        wc, _ = compute_jac_scale(dcdx_f.T)
        LHSh = LHS * wc
        # restrict to optimization subspace
        LHS_opt = LHSh @ subspace

        if verbose > 0:
            print("Factoring LHS")
        timer.start("LHS factorization")
        ug, sg, vtg = jnp.linalg.svd(LHS_opt, full_matrices=False)
        timer.stop("LHS factorization")
        if verbose > 1:
            timer.disp("LHS factorization")

        c_norm = jnp.linalg.norm(dcdx_f @ x_reduced)
        x_norm = jnp.linalg.norm(wx * x_reduced)

        dc1h_opt, bound_hit, _ = trust_region_step_exact_svd(
            -RHS_1g,
            ug,
            sg,
            vtg.T,
            tr_ratio[0] * c_norm,
            initial_alpha=0.0,
            rtol=0.01,
            max_iter=10,
        )

        dc1h = dc1h_opt @ subspace.T
        dc1 = dc1h * wc
        RHS_1f = -f - Fc @ dc1

        dx1h_reduced, _, _ = trust_region_step_exact_svd(
            -RHS_1f,
            uf,
            sf,
            vtf.T,
            tr_ratio[0] * x_norm,
            initial_alpha=0.0,
            rtol=0.01,
            max_iter=10,
        )
        dx1_reduced = dx1h_reduced * wx
    # 2nd order
    if order > 1:
        # 2nd partial derivatives of f objective wrt both x and c
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        tangents_f = dxdx_reduced @ dx1_reduced + dxdc @ dc1
        RHS_2f = -0.5 * objective_f.jvp_scaled((tangents_f, tangents_f), xf)
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        # 2nd partial derivatives of g objective wrt both x and c
        if verbose > 0:
            print("Computing d^2g")
        timer.start("d^2g computation")
        tangents_g = dxdx_reduced @ dx1_reduced + dxdc @ dc1
        RHS_2g = (
            0.5 * objective_g.jvp_scaled((tangents_g, tangents_g), xg) + GxFx @ RHS_2f
        )
        timer.stop("d^2g computation")
        if verbose > 1:
            timer.disp("d^2g computation")

        dc2h_opt, _, _ = trust_region_step_exact_svd(
            -RHS_2g,
            ug,
            sg,
            vtg.T,
            tr_ratio[1] * jnp.linalg.norm(dc1h_opt),
            initial_alpha=0.0,
            rtol=0.01,
            max_iter=10,
        )

        dc2h = dc2h_opt @ subspace.T
        dc2 = dc2h * wc
        RHS_2f += -Fc @ dc2

        dx2h_reduced, _, _ = trust_region_step_exact_svd(
            -RHS_2f,
            uf,
            sf,
            vtf.T,
            tr_ratio[1] * jnp.linalg.norm(dx1h_reduced),
            initial_alpha=0.0,
            rtol=0.01,
            max_iter=10,
        )
        dx2_reduced = dx2h_reduced * wx
    if order > 2:
        raise ValueError(
            "Higher-order perturbations not yet implemented: {}".format(order)
        )

    if copy:
        eq_new = eq.copy()
    else:
        eq_new = eq

    dc = dc1 + dc2
    dc_opt = dc @ subspace

    # update perturbation attributes
    idx0 = 0
    for key, value in deltas.items():
        setattr(eq_new, key, getattr(eq_new, key) + dc[idx0 : idx0 + len(value)])
        idx0 += len(value)
    for con in constraints:
        if hasattr(con, "update_target"):
            con.update_target(eq_new)
    constraint = ObjectiveFunction(constraints)
    constraint.build(verbose=verbose)
    _, _, _, _, _, _, _, recover, *_ = factorize_linear_constraints(
        objective_f, constraint
    )

    # update other attributes
    dx_reduced = dx1_reduced + dx2_reduced
    x_new = recover(x_reduced + dx_reduced)
    params = objective_f.unpack_state(x_new, False)[0]
    for key, value in params.items():
        if key not in deltas:
            value = put(  # parameter values below threshold are set to 0
                value, jnp.where(jnp.abs(value) < 10 * jnp.finfo(value.dtype).eps)[0], 0
            )
            # don't set nonexistent profile (values are empty ndarrays)
            if value.size:
                setattr(eq_new, key, value)

    predicted_reduction = -evaluate_quadratic_form_jac(LHS, -RHS_1g.T @ LHS, dc)

    timer.stop("Total perturbation")
    if verbose > 0:
        print(
            "||dc||/||c|| = {:10.3e}".format(jnp.linalg.norm(dc) / jnp.linalg.norm(c))
        )
        print(
            "||dx||/||x|| = {:10.3e}".format(
                jnp.linalg.norm(dx_reduced) / jnp.linalg.norm(x_reduced)
            )
        )
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new, predicted_reduction, dc_opt, dc, jnp.linalg.norm(c), bound_hit
