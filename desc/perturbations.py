"""Functions for perturbing equilibria."""

import warnings

import numpy as np
from termcolor import colored

from desc.backend import put, use_jax
from desc.compute import arg_order
from desc.objectives import get_fixed_boundary_constraints
from desc.objectives.utils import align_jacobian, factorize_linear_constraints
from desc.optimize.tr_subproblems import trust_region_step_exact_svd
from desc.optimize.utils import compute_jac_scale, evaluate_quadratic_form_jac
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
                deltas["dRb"] = s2.R_lmn - s1.R_lmn
            if not np.allclose(s2.Z_lmn, s1.Z_lmn):
                deltas["dZb"] = s2.Z_lmn - s1.Z_lmn

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
                    deltas["d" + key[0]] = t2.params - t1.params

    if "Psi" in things1:
        psi1 = things1.pop("Psi")
        psi2 = things2.pop("Psi")
        if psi1 is not None and not np.allclose(psi2, psi1):
            deltas["dPsi"] = psi2 - psi1

    assert len(things1) == 0, "get_deltas got an unexpected key: {}".format(
        things1.keys()
    )
    return deltas


def perturb(  # noqa: C901 - FIXME: break this up into simpler pieces
    eq,
    objective,
    constraints=(),
    dR=None,
    dZ=None,
    dL=None,
    dp=None,
    di=None,
    dc=None,
    dPsi=None,
    dRb=None,
    dZb=None,
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
    dR, dZ, dL, dp, di, dc, dPsi, dRb, dZb : ndarray or float
        Deltas for perturbations of R, Z, lambda, pressure, rotational transform,
        toroidal current, total toroidal magnetic flux, R_boundary, and Z_boundary.
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
    if dc is not None and np.any(dc):
        deltas["c_l"] = dc
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
        dc = np.concatenate([deltas[arg] for arg in other_args if arg in deltas.keys()])
        x_idx = np.concatenate(
            [objective.x_idx[arg] for arg in other_args if arg in deltas.keys()]
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

        # 1st partial derivatives wrt both state vector (x) and input parameters (c)
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Jx = objective.jac(x)
        Jx_reduced = Jx[:, unfixed_idx] @ Z @ scale
        RHS1 = objective.jvp(tangents, x)
        if include_f:
            f = objective.compute(x)
            RHS1 += f
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
    if verbose > 0:
        print("Number of parameters: {}".format(dim_opt))
        print("Number of objectives: {}".format(objective_g.dim_f))

    # FIXME: generalize to other constraints
    constraints = get_fixed_boundary_constraints(iota=eq.iota is not None)
    for constraint in constraints:
        if not constraint.built:
            constraint.build(eq, verbose=verbose)
    (
        xp,
        A,
        Ainv,
        b,
        Z,
        unfixed_idx,
        project,
        recover,
    ) = factorize_linear_constraints(constraints, objective_f.args)

    # state vector
    xf = objective_f.x(eq)
    xg = objective_g.x(eq)

    x_reduced = project(xf)

    # perturbation vectors
    dc1 = 0
    dc2 = 0
    dx1_reduced = 0
    dx2_reduced = 0

    # dx/dx_reduced
    dxdx_reduced = np.eye(objective_f.dim_x)[:, unfixed_idx] @ Z

    # dx/dc
    dxdc = np.zeros((objective_f.dim_x, 0))
    if len(
        [
            arg
            for arg in ("R_lmn", "Z_lmn", "L_lmn", "p_l", "i_l", "Psi")
            if arg in deltas.keys()
        ]
    ):
        x_idx = np.concatenate(
            [objective_f.x_idx[arg] for arg in arg_order if arg in deltas.keys()]
        )
        x_idx.sort(kind="mergesort")
        dxdc = np.eye(objective_f.dim_x)[:, x_idx]
    if "Rb_lmn" in deltas.keys():
        dxdRb = np.eye(objective_f.dim_x)[:, objective_f.x_idx["R_lmn"]] @ Ainv["R_lmn"]
        dxdc = np.hstack((dxdc, dxdRb))
    if "Zb_lmn" in deltas.keys():
        dxdZb = np.eye(objective_f.dim_x)[:, objective_f.x_idx["Z_lmn"]] @ Ainv["Z_lmn"]
        dxdc = np.hstack((dxdc, dxdZb))

    # 1st order
    if order > 0:

        f = objective_f.compute(xf)
        g = objective_g.compute(xg)

        # 1st partial derivatives of f objective wrt x
        if verbose > 0:
            print("Computing df")
        timer.start("df computation")
        Fx = objective_f.jac(xf)
        Fx = align_jacobian(Fx, objective_f, objective_g)
        timer.stop("df computation")
        if verbose > 1:
            timer.disp("df computation")

        # 1st partial derivatives of g objective wrt x
        if verbose > 0:
            print("Computing dg")
        timer.start("dg computation")
        Gx = objective_g.jac(xg)
        Gx = align_jacobian(Gx, objective_g, objective_f)
        timer.stop("dg computation")
        if verbose > 1:
            timer.disp("dg computation")

        # projections onto optimization space
        Fx_reduced = Fx[:, unfixed_idx] @ Z
        Gx_reduced = Gx[:, unfixed_idx] @ Z
        Fc = Fx @ dxdc
        Gc = Gx @ dxdc

        # some scaling to improve conditioning and rescale trust region
        wf, _ = compute_jac_scale(Fx_reduced)
        wg, _ = compute_jac_scale(Gx_reduced)
        wx = wf + wg
        Fxh = Fx_reduced * wx
        Gxh = Gx_reduced * wx
        if cutoff is None:
            cutoff = np.finfo(Fx_reduced.dtype).eps * np.max(Fx_reduced.shape)
        uf, sf, vtf = np.linalg.svd(Fxh, full_matrices=False)
        sf += sf[-1]  # add a tiny bit of regularization
        sfi = np.where(sf < cutoff * sf[0], 0, 1 / sf)
        Fxh_inv = vtf.T @ (sfi[..., np.newaxis] * uf.T)

        GxFx = Gxh @ Fxh_inv
        LHS = GxFx @ Fc - Gc
        RHS_1g = g - GxFx @ f

        # scaling for c
        # approx dc/dx at const f
        dcdx_f = np.linalg.lstsq(Fc, Fx_reduced, rcond=None)[0]
        wc, _ = compute_jac_scale(dcdx_f.T)
        LHSh = LHS * wc
        # restrict to optimization subspace
        LHS_opt = LHSh @ subspace

        if verbose > 0:
            print("Factoring LHS")
        timer.start("LHS factorization")
        ug, sg, vtg = np.linalg.svd(LHS_opt, full_matrices=False)
        timer.stop("LHS factorization")
        if verbose > 1:
            timer.disp("LHS factorization")

        c_norm = np.linalg.norm(dcdx_f @ x_reduced)
        x_norm = np.linalg.norm(wx * x_reduced)

        dc1h_opt, bound_hit, _ = trust_region_step_exact_svd(
            -RHS_1g,
            ug,
            sg,
            vtg.T,
            tr_ratio[0] * c_norm,
            initial_alpha=None,
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
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
        )
        dx1_reduced = dx1h_reduced * wx
    # 2nd order
    if order > 1:

        idx = np.array([], dtype=int)
        for arg in objective_f.args:
            if arg not in objective_g.args:
                idx = np.concatenate((idx, objective_f.x_idx[arg]))
        dxf_dxg = np.delete(np.eye(objective_f.dim_x), idx, 1)

        # 2nd partial derivatives of f objective wrt both x and c
        if verbose > 0:
            print("Computing d^2f")
        timer.start("d^2f computation")
        tangents_f = dxdx_reduced @ dx1_reduced + dxdc @ dc1
        RHS_2f = -0.5 * objective_f.jvp((tangents_f, tangents_f), xf)
        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        # 2nd partial derivatives of g objective wrt both x and c
        if verbose > 0:
            print("Computing d^2g")
        timer.start("d^2g computation")
        tangents_g = (dxdx_reduced @ dx1_reduced + dxdc @ dc1) @ dxf_dxg
        RHS_2g = 0.5 * objective_g.jvp((tangents_g, tangents_g), xg) + GxFx @ RHS_2f
        timer.stop("d^2g computation")
        if verbose > 1:
            timer.disp("d^2g computation")

        dc2h_opt, _, _ = trust_region_step_exact_svd(
            -RHS_2g,
            ug,
            sg,
            vtg.T,
            tr_ratio[1] * np.linalg.norm(dc1h_opt),
            initial_alpha=None,
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
            tr_ratio[1] * np.linalg.norm(dx1h_reduced),
            initial_alpha=None,
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
    for constraint in constraints:
        constraint.update_target(eq_new)
    xp, A, Ainv, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints, objective_f.args
    )

    # update other attributes
    dx_reduced = dx1_reduced + dx2_reduced
    x_new = recover(x_reduced + dx_reduced)
    args = objective_f.unpack_state(x_new)
    for key, value in args.items():
        if key not in deltas:
            value = put(  # parameter values below threshold are set to 0
                value, np.where(np.abs(value) < 10 * np.finfo(value.dtype).eps)[0], 0
            )
            # don't set nonexistent profile (values are empty ndarrays)
            if not (key == "c_l" or key == "i_l") or value.size:
                setattr(eq_new, key, value)

    predicted_reduction = -evaluate_quadratic_form_jac(LHS, -RHS_1g.T @ LHS, dc)

    timer.stop("Total perturbation")
    if verbose > 0:
        print("||dc||/||c|| = {:10.3e}".format(np.linalg.norm(dc) / np.linalg.norm(c)))
        print(
            "||dx||/||x|| = {:10.3e}".format(
                np.linalg.norm(dx_reduced) / np.linalg.norm(x_reduced)
            )
        )
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new, predicted_reduction, dc_opt, dc, np.linalg.norm(c), bound_hit
