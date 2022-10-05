import numpy as np
import warnings
from termcolor import colored

from desc.equilibrium import Equilibrium
from desc.utils import Timer


def solve_continuation(
    surface,
    pressure,
    iota,
    L,
    M,
    N,
    NFP=1,
    Psi=1.0,
    sym=None,
    L_grid=None,
    M_grid=None,
    N_grid=None,
    node_pattern="jacobi",
    spectral_indexing="ansi",
    objective="force",
    optimizer="lsq-exact",
    pert_order=2,
    ftol=1e-2,
    xtol=1e-4,
    gtol=1e-6,
    nfev=100,
    verbose=1,
    checkpoint_path=None,
):
    """Solve for an equilibrium using an automatic continuation method.

    By default, the method first solves for a vacuum tokamak, then a finite beta
    tokamak, then a finite beta stellarator. Currently hard coded to take a fixed
    number of perturbation steps based on conservative estimates and testing. In the
    future, continuation stepping will be done adaptively.

    Parameters
    ----------
    surface : Surface
        desired surface of the final equilibrium
    pressure, iota : Profile
        desired profiles of the final equilibrium
    L, M, N : int
        desired spectral resolution of the final equilibrium
    NFP : int
        number of field periods
    Psi : float (optional)
        total toroidal flux (in Webers) within LCFS. Default 1.0
    sym : bool (optional)
        Whether to enforce stellarator symmetry. Default surface.sym or False.
    L_grid, M_grid, N_grid : int
        desired real space resolution of the final equilibrium
    node_pattern : str (optional)
        pattern of nodes in real space. Default is ``'jacobi'``
    spectral_indexing : str (optional)
        Type of Zernike indexing scheme to use. Default ``'ansi'``
    objective : str or ObjectiveFunction (optional)
        function to solve for equilibrium solution
    optimizer : str or Optimzer (optional)
        optimizer to use
    pert_order : int
        order of perturbations to use.
    ftol, xtol, gtol : float
        stopping tolerances for subproblem at each step.
    nfev : int
        maximum number of function evaluations in each equilibrium subproblem.

    Returns
    -------
    eqf : EquilibriaFamily
        family of equilibria for the intermediate steps, where the last member is the
        final desired configuration,

    """

    timer = Timer()
    timer.start("Total time")

    surf_i = surface.copy()
    pres_i = pressure.copy()
    iota_i = iota.copy()

    L_grid = L_grid or 2 * L
    M_grid = M_grid or 2 * M
    N_grid = N_grid or 2 * N

    res_step = 6
    pres_step = 1 / 2
    bdry_step = 1 / 4
    Mi = min(M // 2, res_step)
    Li = 2 * Mi if spectral_indexing == "fringe" else Mi
    Ni = 0
    L_gridi = L_grid / L * Li
    M_gridi = M_grid / M * Mi
    N_gridi = N_grid / N * Ni

    # first we solve vacuum until we reach full L,M
    # then pressure
    # then 3d shaping
    res_steps = max(M // res_step, 1)
    pres_steps = (
        0
        if (pressure(np.linspace(0, 1, 20)) == 0).all()
        else int(np.ceil(1 / pres_step))
    )
    bdry_steps = 0 if N == 0 else int(np.ceil(1 / bdry_step))

    eqfam = []

    bdry_ratio = pres_ratio = curr_ratio = 0

    surf_i.change_resolution(Li, Mi, Ni)
    pres_i.change_resolution(Li)
    iota_i.change_resolution(Li)

    # start with zero pressure
    pres_i.params *= 0

    eq_init = Equilibrium(
        Psi,
        NFP,
        Li,
        Mi,
        Ni,
        L_gridi,
        M_gridi,
        N_gridi,
        node_pattern,
        pres_i,
        iota_i,
        surf_i,
        None,
        sym,
        spectral_indexing,
    )

    eq_init.solve(objective, optimizer, ftol, xtol, gtol, nfev)
    ii = 0
    nn = res_steps + pres_steps + bdry_steps
    for ii in range(nn):
        # TODO : print iteration summary
        eqi = eqfam[-1].copy()
        if ii < res_steps:
            # increase resolution of vacuum soln
            Mi = min(Mi + res_step, M)
            Li = 2 * Mi if spectral_indexing == "fringe" else Mi
            L_gridi = np.ceil(L_grid / L * Li).astype(int)
            M_gridi = np.ceil(M_grid / M * Mi).astype(int)
            N_gridi = np.ceil(N_grid / N * Ni).astype(int)

            surf_i2 = surface.copy()
            surf_i2.change_resolution(Li, Mi, Ni)
            iota_i2 = iota.copy()
            iota_i2.change_resolution(Li)
            deltas = _get_deltas(
                {"surface": [surf_i, surf_i2], "iota": [iota_i, iota_i2]}
            )
            eqi.change_resolution(Li, Mi, Ni, L_gridi, M_gridi, N_gridi)
            surf_i = surf_i2
            iota_i = iota_i2
        if ii > res_steps and ii < res_steps + pres_steps:
            # increase pressure
            deltas = _get_deltas({"pressure": [eqfam[res_steps].pressure, pressure]})
            deltas["p_l"] *= pres_step
            pres_ratio += pres_step

        if ii > res_steps + pres_steps:
            # boundary perturbations
            # TODO: do we want to jump to full N? or maybe step that up too?
            eqi.change_resolution(L, M, N, L_grid, M_grid, N_grid)
            deltas = _get_deltas(
                {"surface": [eqfam[res_steps + pres_steps].surface, surface]}
            )
            deltas["Rb_lmn"] *= bdry_step
            deltas["Zb_lmn"] *= bdry_step
            bdry_ratio += bdry_step

        _print_iteration_summary(
            ii,
            nn,
            eqi,
            bdry_ratio,
            pres_ratio,
            curr_ratio,
            pert_order,
            objective,
            optimizer,
            ftol,
            gtol,
            xtol,
            nfev,
        )
        eqfam.append(eqi)
        eqfam[-1].perturb(**deltas, order=pert_order)
        if not eqi.is_nested(msg="auto"):
            break

        eqfam[-1].solve(objective, optimizer, ftol, xtol, gtol, nfev)
        if not eqi.is_nested(msg="auto"):
            break
        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)

    timer.stop("Total time")
    if verbose > 0:
        print("====================")
        print("Done")
    if verbose > 1:
        timer.disp("Total time")
    if checkpoint_path is not None:
        if verbose > 0:
            print("Output written to {}".format(checkpoint_path))
        eqfam.save(checkpoint_path)
    if verbose:
        print("====================")

    return eqfam


def _get_deltas(things):

    deltas = {}
    if "surface" in things:
        s1 = things["surface"][0].copy()
        s2 = things["surface"][1].copy()

        s1.change_resolution(s2.L, s2.M, s2.N)
        deltas["Rb_lmn"] = s2.R_lmn - s1.R_lmn
        deltas["Zb_lmn"] = s2.Z_lmn - s1.Z_lmn
    if "iota" in things:
        i1 = things["iota"][0].copy()
        i2 = things["iota"][1].copy()

        i1.change_resolution(i2.L)
        deltas["i_l"] = i2.params - i1.params
    if "current" in things:
        c1 = things["current"][0].copy()
        c2 = things["current"][1].copy()

        c1.change_resolution(c2.L)
        deltas["c_l"] = c2.params - c1.params
    if "pressure" in things:
        p1 = things["pressure"][0].copy()
        p2 = things["pressure"][1].copy()

        p1.change_resolution(p2.L)
        deltas["p_l"] = p2.params - p1.params

    return deltas


def _print_iteration_summary(
    ii,
    nn,
    eq,
    bdry_ratio,
    pres_ratio,
    curr_ratio,
    pert_order,
    objective,
    optimizer,
    ftol,
    gtol,
    xtol,
    nfev,
    **kwargs
):
    print("================")
    print("Step {}/{}".format(ii + 1, nn))
    print("================")
    eq.resolution_summary()
    print("Boundary ratio = {}".format(bdry_ratio))
    print("Pressure ratio = {}".format(pres_ratio))
    if eq.current is not None:
        print("Current ratio = {}".format(curr_ratio))
    print("Perturbation Order = {}".format(pert_order))
    print("Objective: {}".format(objective))
    print("Optimizer: {}".format(optimizer))
    print("Function tolerance = {}".format(ftol))
    print("Gradient tolerance = {}".format(gtol))
    print("State vector tolerance = {}".format(xtol))
    print("Max function evaluations = {}".format(nfev))
    print("================")
