import numpy as np
from desc.equilibrium import Equilibrium


def solve_continuation(
    surface,
    pressure,
    iota,
    L,
    M,
    N,
    Psi=1.0,
    sym=None,
    L_grid=None,
    M_grid=None,
    N_grid=None,
    node_pattern="jacobi",
    spectral_indexing="ansi",
    objective="force",
    optimizer="lsq-exact",
    pres_ratio=None,
    bdry_ratio=None,
    ftol=1e-2,
    xtol=1e-4,
    gtol=1e-6,
    maxiter=100,
    maxsteps=10,
):
    """Solve for an equilibrium using an automatic continuation method.

    By default, the method first solves for a vacuum tokamak, then a finite beta
    tokamak, then a finite beta stellarator. Perturbations are done adaptively, or they
    can be specified manually by passing arrays for pres_ratio and bdry_ratio.

    Parameters
    ----------
    surface : Surface
        desired surface of the final equilibrium
    pressure, iota : Profile
        desired profiles of the final equilibrium
    L, M, N : int or array-like of int
        desired spectral resolution of the final equilibrium, or the resolution at each
        continuation step
    Psi : float (optional)
        total toroidal flux (in Webers) within LCFS. Default 1.0
    sym : bool (optional)
        Whether to enforce stellarator symmetry. Default surface.sym or False.
    L_grid, M_grid, N_grid : int or array-like of int
        desired real space resolution of the final equilibrium, or the resolution at each
        continuation step
    node_pattern : str (optional)
        pattern of nodes in real space. Default is ``'jacobi'``
    spectral_indexing : str (optional)
        Type of Zernike indexing scheme to use. Default ``'ansi'``
    objective : str or ObjectiveFunction (optional)
        function to solve for equilibrium solution
    optimizer : str or Optimzer (optional)
        optimizer to use
    pres_ratio, bdry_ratio : array-like of float (optional)
        manually specify perturbation steps
    ftol, xtol, gtol : float or array-like of float
        stopping tolerances for subproblem at each step. Can be passed as arrays to
        give different tolerances at each step.
    maxiter : int or array-like of int
        maximum number of iterations of the equilibrium subproblem. Can be passed as array
        to give different number for each step.
    maxsteps : int
        maximum number of continuation steps. Method will fail if final configuration
        is not achieved in this number of steps.

    Returns
    -------
    eqf : EquilibriaFamily
        family of equilibria for the intermediate steps, where the last member is the
        final desired configuration,

    """

    # TODO: broadcast arrays against each other

    # TODO: how to combine automatic w/ user specified stuff?

    surf_i = surface.copy()
    pres_i = pressure.copy()
    iota_i = iota.copy()

    res_step = 6
    pres_step = 1 / 2
    bdry_step = 1 / 2
    Mi = min(M // 2, res_step)
    Li = 2 * Mi if spectral_indexing == "fringe" else Mi
    Ni = 0

    # first we solve vacuum until we reach full L,M
    # then pressure
    # then 3d shaping
    res_steps = max(M // res_step, 1)
    pres_steps = (
        0
        if (pressure(np.linspace(0, 1, 20)) == 0).all()
        else int(np.ceil(1 / pres_step))
    )
    bdry_steps = 0 if surface.N == 0 else int(np.ceil(1 / bdry_step))

    eqfam = []

    surf_i.change_resolution(Li, Mi, Ni)
    pres_i.change_resolution(Li, Mi, Ni)
    iota_i.change_resolution(Li, Mi, Ni)

    eq_init = Equilibrium(
        Psi,
        surf_i.NFP,
        Li,
        Mi,
        Ni,
        node_pattern,
        pres_i,
        iota_i,
        surf_i,
        None,
        sym,
        spectral_indexing,
    )

    eq_init.solve(objective, optimizer, ftol, xtol, gtol, maxiter)
    ii = 0
    while ii < (res_steps + pres_steps + bdry_steps):
        if ii > maxsteps:
            print("exceeded max number of continuation steps")
            break
        # TODO : print iteration summary
        eqi = eqfam[-1].copy()
        if ii < res_steps:
            # increase resolution of vacuum soln
            Mi = min(Mi + res_step, M)
            Li = 2 * Mi if spectral_indexing == "fringe" else Mi
            surf_i2 = surface.copy()
            surf_i2.change_resolution(Li, Mi, Ni)
            pres_i.change_resolution(Li, Mi, Ni)
            deltas = get_deltas(surf_i2, surf_i, iota2, iota1)
            eqi.change_resolution(Li, Mi, Ni)
            eqi.perturb(**deltas)
            eqi.solve(objective, optimizer)
            if not eqi.isnested():
                print("step failed, trying smaller step")
                res_step = max(res_step - 2, 1)
                res_steps = max(M // res_step, 1)
            else:
                eqfam.append(eqi)
                ii += 1
        if ii > res_steps and ii < res_steps + pres_steps:
            # increase pressure
            deltas = get_deltas(eqfam[res_steps].pressure, pressure) * pres_step
            eqi.perturb(**deltas)
            eqi.solve(objective, optimizer)
            if not eqi.isnested():
                print("step failed, trying smaller step")
                pres_step = pres_step / 1.5
                pres_steps = (
                    0
                    if (pressure(np.linspace(0, 1, 20)) == 0).all()
                    else int(np.ceil(1 / pres_step))
                )
            else:
                eqfam.append(eqi)
                ii += 1
        if ii > res_steps + pres_steps:
            # boundary perturbations
            # TODO: do we want to jump to full N? or maybe step that up too?
            eqi.change_resolution(L, M, N)
            deltas = (
                get_deltas(eqfam[res_steps + pres_steps].surface, surface) * bdry_step
            )
            eqi.perturb(**deltas)
            eqi.solve(objective, optimizer)
            if not eqi.isnested():
                print("step failed, trying smaller step")
                bdry_step = bdry_step / 1.5
                bdry_steps = 0 if surface.N == 0 else int(np.ceil(1 / bdry_step))
            else:
                eqfam.append(eqi)
                ii += 1

    return eqfam


def get_deltas(thing1, thing2):

    # TODO: stuff for making resolutions the same etc

    return thing2.params - thing1.params
