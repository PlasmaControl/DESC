import numpy as np
import time
import warnings
from termcolor import colored

from desc.utils import Timer
from desc.backend import use_jax
from desc.boundary_conditions import get_boundary_condition
from desc.optimize.tr_subproblems import trust_region_step_exact

__all__ = ["perturb"]


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
        toroidal flux, and zeta ratio.. Setting to None or zero ignores that term
        in the expansion.
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
                + "highly innacurate. Consider using JAX for exact derivatives",
                "yellow",
            )
        )
    if not eq.objective:
        raise AttributeError(
            "Equilibrium must have objective defined before perturbing"
        )
    if eq.objective.scalar:
        raise AttributeError(
            "Cannot perturb with a scalar objective {}".format(eq.objective)
        )

    deltas = {}
    if dRb is not None and not np.all(dRb == 0):
        deltas["Rb_lmn"] = dRb
    if dZb is not None and not np.all(dZb == 0):
        deltas["Zb_lmn"] = dZb
    if dp is not None and not np.all(dp == 0):
        deltas["p_l"] = dp
    if di is not None and not np.all(di == 0):
        deltas["i_l"] = di
    if dPsi is not None and not np.all(dPsi == 0):
        deltas["Psi"] = dPsi
    if dzeta_ratio is not None and not np.all(dzeta_ratio == 0):
        deltas["zeta_ratio"] = dzeta_ratio

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

    if np.isscalar(tr_ratio):
        tr_ratio = tr_ratio * np.ones(order)
    elif len(tr_ratio) < order:
        raise ValueError(
            "Got only {} tr_ratios for order {} perturbations".format(
                len(tr_ratio), order
            )
        )
    keys = ", ".join(deltas.keys())
    if verbose > 0:
        print("Perturbing {}".format(keys))

    # 1st order
    if order > 0:

        # partial derivatives wrt state vector (x)
        if Jx is None:
            if verbose > 0:
                print("Computing df")
            timer.start("df computation")
            Jx = eq.objective.jac_x(*args)
            timer.stop("df computation")
            if verbose > 1:
                timer.disp("df computation")
        u, s, vt = np.linalg.svd(Jx, full_matrices=False)
        m, n = Jx.shape
        #  cutoff = np.finfo(s.dtype).eps * m * np.amax(s, axis=-1, keepdims=True)
        cutoff = cutoff * s[0]
        large = s > cutoff
        s_inv = np.divide(1, s, where=large)
        s_inv[~large] = 0
        RHS = eq.objective.compute(*args)

        # partial derivatives wrt input parameters (c)

        inds = tuple([arg_idx[key] for key in deltas])
        dc = tuple([val for val in deltas.values()])
        timer.start("df/dc computation ({})".format(keys))
        temp = eq.objective.jvp(inds, dc, *args)
        RHS += temp
        timer.stop("df/dc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("df/dc computation ({})".format(keys))
        dx1, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS,
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
        temp = 0.5 * eq.objective.jvp2(inds, inds, tangents, tangents, *args)
        RHS = temp

        timer.stop("d^2f computation")
        if verbose > 1:
            timer.disp("d^2f computation")

        dx2, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS,
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

        # 3nd partial derivatives wrt state vector (x)
        if verbose > 0:
            print("Computing d^3f")
        timer.start("d^3f computation")
        inds = tuple([arg_idx[key] for key in deltas])
        tangents = tuple([val for val in deltas.values()])
        inds = (0, *inds)
        tangents = (dx1, *tangents)
        RHS = (
            1
            / 6
            * eq.objective.jvp3(inds, inds, inds, tangents, tangents, tangents, *args)
        )
        RHS += eq.objective.jvp2(0, inds, dx2, tangents, *args)
        timer.stop("d^3f computation")
        if verbose > 1:
            timer.disp("d^3f computation")

        dx2, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS,
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

    dy = dx1 + dx2 + dx3
    eq_new.x = eq_new.objective.BC_constraint.recover(y + dy)

    timer.stop("Total perturbation")
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new
