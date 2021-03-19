import numpy as np
import time
import warnings
from termcolor import colored

from desc.utils import Timer
from desc.backend import use_jax
from desc.boundary_conditions import BoundaryCondition
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
    order : int, optional
        order of perturbation (0=none, 1=linear, 2=quadratic)
    tr_ratio : float or array of float
        radius of the trust region, as a fraction of ||x||.
        enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||
        if a scalar uses same ratio for both steps, if an array uses the first element
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

    try:
        tr_ratio1, tr_ratio2 = tr_ratio
    except TypeError:
        tr_ratio1 = tr_ratio
        tr_ratio2 = tr_ratio

    # 1st order
    if order > 0:

        # partial derivatives wrt state vector (x)
        if Jx is None:
            if verbose > 0:
                print("Computing df/dx")
            timer.start("df/dx computation")
            Jx = eq.objective.jac_x(*args)
            timer.stop("df/dx computation")
            if verbose > 1:
                timer.disp("df/dx computation")
        u, s, vt = np.linalg.svd(Jx, full_matrices=False)
        m, n = Jx.shape
        #  cutoff = np.finfo(s.dtype).eps * m * np.amax(s, axis=-1, keepdims=True)
        cutoff = cutoff * s[0]
        large = s > cutoff
        s_inv = np.divide(1, s, where=large)
        s_inv[~large] = 0
        RHS = eq.objective.compute(*args)

        # partial derivatives wrt input parameters (c)
        keys = ", ".join(deltas.keys())
        inds = tuple([arg_idx[key] for key in deltas])
        dc = tuple([val for val in deltas.values()])
        if verbose > 0:
            print("Perturbing {}".format(keys))
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
            tr_ratio * np.linalg.norm(y),
            initial_alpha=None,
            rtol=0.01,
            max_iter=10,
            threshold=1e-6,
        )

    # 2nd order
    if order > 1:

        # 2nd partial derivatives wrt state vector (x)
        if verbose > 0:
            print("Computing d^2f/dx^2")
        timer.start("d^2f/dx^2 computation")
        temp = 0.5 * eq.objective.jvp2(0, 0, dx1, dx1, *args)
        RHS = temp

        timer.stop("d^2f/dx^2 computation")
        if verbose > 1:
            timer.disp("d^2f/dx^2 computation")

        keys = ", ".join(deltas.keys())
        inds = tuple([arg_idx[key] for key in deltas])
        dc = tuple([val for val in deltas.values()])
        timer.start("d^2f/dc^2 computation ({})".format(keys))
        temp = 0.5 * eq.objective.jvp2(inds, inds, dc, dc, *args)
        RHS += temp

        timer.stop("d^2f/dc^2 computation ({})".format(keys))
        if verbose > 1:
            timer.disp("d^2f/dc^2 computation ({})".format(keys))

        # mixed partials wrt to x, c
        keys = ", ".join(deltas.keys())
        inds = tuple([arg_idx[key] for key in deltas])
        dc = tuple([val for val in deltas.values()])
        timer.start("d^2f/dxdc computation ({})".format(keys))
        temp = eq.objective.jvp2(0, inds, dx1, dc, *args)
        RHS += temp

        timer.stop("d^2f/dxdc computation ({})".format(keys))
        if verbose > 1:
            timer.disp("d^2f/dxdc computation ({})".format(keys))

        dx2, hit, alpha = trust_region_step_exact(
            n,
            m,
            RHS,
            u,
            s,
            vt.T,
            tr_ratio * np.linalg.norm(dx1),
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
    if "dRb" in deltas or "dZb" in deltas:
        eq_new.objective.BC_constraint = BoundaryCondition(
            eq_new.R_basis,
            eq_new.Z_basis,
            eq_new.L_basis,
            eq_new.Rb_basis,
            eq_new.Zb_basis,
            eq_new.Rb_lmn,
            eq_new.Zb_lmn,
        )

    # perturbation
    if order > 0:
        dy = dx1 + dx2
        eq_new.x = eq_new.objective.BC_constraint.recover(y + dy)

    timer.stop("Total perturbation")
    if verbose > 1:
        timer.disp("Total perturbation")

    return eq_new
