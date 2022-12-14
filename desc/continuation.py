"""Functions for solving for equilibria with multigrid continuation method."""

import copy

import numpy as np

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.objectives import get_equilibrium_objective, get_fixed_boundary_constraints
from desc.optimize import Optimizer
from desc.perturbations import get_deltas
from desc.utils import Timer


def solve_continuation_automatic(  # noqa: C901
    eq,
    objective="force",
    optimizer="lsq-exact",
    pert_order=2,
    ftol=None,
    xtol=None,
    gtol=None,
    nfev=100,
    verbose=1,
    checkpoint_path=None,
    **kwargs,
):
    """Solve for an equilibrium using an automatic continuation method.

    By default, the method first solves for a no pressure tokamak, then a finite beta
    tokamak, then a finite beta stellarator. Currently hard coded to take a fixed
    number of perturbation steps based on conservative estimates and testing. In the
    future, continuation stepping will be done adaptively.

    Parameters
    ----------
    eq : Equilibrium
        Unsolved Equilibrium with the final desired boundary, profiles, resolution.
    objective : {"force", "energy", "vacuum"}
        function to solve for equilibrium solution
    optimizer : str or Optimzer (optional)
        optimizer to use
    pert_order : int
        order of perturbations to use.
    ftol, xtol, gtol : float
        stopping tolerances for subproblem at each step. `None` will use defaults
        for given optimizer.
    nfev : int
        maximum number of function evaluations in each equilibrium subproblem.
    verbose : integer
        * 0: no output
        * 1: summary of each iteration
        * 2: as above plus timing information
        * 3: as above plus detailed solver output
    checkpoint_path : str or path-like
        file to save checkpoint data (Default value = None)
    **kwargs : control continuation step sizes

        Valid keyword arguments are:

        mres_step: int, the amount to increase Mpol by at each continuation step
        pres_step: float, 0<=pres_step<=1, the amount to increase pres_ratio by
                          at each continuation step
        bdry_step: float, 0<=bdry_step<=1, the amount to increase pres_ratio by
                          at each continuation step
    Returns
    -------
    eqfam : EquilibriaFamily
        family of equilibria for the intermediate steps, where the last member is the
        final desired configuration,

    """
    timer = Timer()
    timer.start("Total time")

    surface = eq.surface
    pressure = eq.pressure
    L, M, N, L_grid, M_grid, N_grid = eq.L, eq.M, eq.N, eq.L_grid, eq.M_grid, eq.N_grid
    spectral_indexing = eq.spectral_indexing

    mres_step = kwargs.pop("mres_step", 6)
    pres_step = kwargs.pop("pres_step", 1 / 2)
    bdry_step = kwargs.pop("bdry_step", 1 / 4)
    assert len(kwargs) == 0, "Got an unexpected kwarg {}".format(kwargs.keys())

    Mi = min(M // 2, mres_step) if mres_step > 0 else M
    Li = int(np.ceil(L / M) * Mi)
    Ni = 0 if bdry_step > 0 else N
    L_gridi = np.ceil(L_grid / L * Li).astype(int)
    M_gridi = np.ceil(M_grid / M * Mi).astype(int)
    N_gridi = np.ceil(N_grid / max(N, 1) * Ni).astype(int)

    # first we solve vacuum until we reach full L,M
    # then pressure
    # then 3d shaping
    mres_steps = int(max(np.ceil(M / mres_step), 1)) if mres_step > 0 else 0
    pres_steps = (
        0
        if (abs(pressure(np.linspace(0, 1, 20))) < 1e-14).all() or pres_step == 0
        else int(np.ceil(1 / pres_step))
    )
    bdry_steps = 0 if N == 0 or bdry_step == 0 else int(np.ceil(1 / bdry_step))
    pres_ratio = 0 if pres_steps else 1
    bdry_ratio = 0 if N else 1
    curr_ratio = 1
    deltas = {}

    surf_axisym = surface.copy()
    pres_vac = pressure.copy()
    surf_axisym.change_resolution(L, M, Ni)
    # start with zero pressure
    pres_vac.params *= 0 if pres_step else 1

    eqi = Equilibrium(
        eq.Psi,
        eq.NFP,
        Li,
        Mi,
        Ni,
        L_gridi,
        M_gridi,
        N_gridi,
        eq.node_pattern,
        pres_vac.copy(),
        copy.copy(eq.iota),  # have to use copy.copy here since may be None
        copy.copy(eq.current),
        surf_axisym.copy(),
        None,
        eq.sym,
        spectral_indexing,
    )

    if not isinstance(optimizer, Optimizer):
        optimizer = Optimizer(optimizer)
    constraints_i = get_fixed_boundary_constraints(
        iota=objective != "vacuum" and eq.iota is not None
    )
    objective_i = get_equilibrium_objective(objective)

    eqfam = EquilibriaFamily()

    ii = 0
    nn = mres_steps + pres_steps + bdry_steps
    stop = False
    while ii < nn and not stop:
        timer.start("Iteration {} total".format(ii + 1))

        if ii > 0:
            eqi = eqfam[-1].copy()

        if ii < mres_steps and ii > 0:
            # increase resolution of vacuum soln
            Mi = min(Mi + mres_step, M)
            Li = int(np.ceil(L / M) * Mi)
            L_gridi = np.ceil(L_grid / L * Li).astype(int)
            M_gridi = np.ceil(M_grid / M * Mi).astype(int)
            N_gridi = np.ceil(N_grid / max(N, 1) * Ni).astype(int)
            eqi.change_resolution(Li, Mi, Ni, L_gridi, M_gridi, N_gridi)

            surf_i = eqi.surface
            surf_i2 = surface.copy()
            surf_i2.change_resolution(Li, Mi, Ni)
            deltas = get_deltas({"surface": surf_i}, {"surface": surf_i2})
            surf_i = surf_i2

        if ii >= mres_steps and ii < mres_steps + pres_steps:
            # make sure its at full radial/poloidal resolution
            eqi.change_resolution(L=L, M=M, L_grid=L_grid, M_grid=M_grid)
            # increase pressure
            deltas = get_deltas(
                {"pressure": eqfam[mres_steps - 1].pressure}, {"pressure": pressure}
            )
            deltas["dp"] *= pres_step
            pres_ratio += pres_step

        elif ii >= mres_steps + pres_steps:
            # otherwise do boundary perturbations to get 3d shape from tokamak
            eqi.change_resolution(L, M, N, L_grid, M_grid, N_grid)
            surf_axisym.change_resolution(L, M, N)
            deltas = get_deltas({"surface": surf_axisym}, {"surface": surface})
            if "dRb" in deltas:
                deltas["dRb"] *= bdry_step
            if "dZb" in deltas:
                deltas["dZb"] *= bdry_step
            bdry_ratio += bdry_step

        if verbose:
            _print_iteration_summary(
                ii,
                nn,
                eqi,
                bdry_ratio,
                pres_ratio,
                curr_ratio,
                pert_order,
                objective_i,
                optimizer,
            )

        if len(eqfam) == 0 or (eqfam[-1].resolution != eqi.resolution):
            constraints_i = get_fixed_boundary_constraints(
                iota=objective != "vacuum" and eq.iota is not None
            )
            objective_i = get_equilibrium_objective(objective)
        if len(deltas) > 0:
            if verbose > 0:
                print("Perturbing equilibrium")
            eqi.perturb(
                objective=objective_i,
                constraints=constraints_i,
                **deltas,
                order=pert_order,
                verbose=verbose,
                copy=False,
            )
            deltas = {}

        if not eqi.is_nested(msg="auto"):
            stop = True
        if not stop:
            eqi.solve(
                optimizer=optimizer,
                objective=objective_i,
                constraints=constraints_i,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=verbose,
                maxiter=nfev,
            )
        if not eqi.is_nested(msg="auto"):
            stop = True
        eqfam.append(eqi)

        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)
        timer.stop("Iteration {} total".format(ii + 1))
        if verbose > 1:
            timer.disp("Iteration {} total".format(ii + 1))
        ii += 1

    eq.R_lmn = eqi.R_lmn
    eq.Z_lmn = eqi.Z_lmn
    eq.L_lmn = eqi.L_lmn
    eqfam[-1] = eq
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


def solve_continuation(  # noqa: C901
    eqfam,
    objective="force",
    optimizer="lsq-exact",
    pert_order=2,
    ftol=None,
    xtol=None,
    gtol=None,
    nfev=100,
    verbose=1,
    checkpoint_path=None,
):
    """Solve for an equilibrium by continuation method.

    Steps through an EquilibriaFamily, solving each equilibrium, and uses pertubations
    to step between different profiles/boundaries.

    Uses the previous step as an initial guess for each solution.

    Parameters
    ----------
    eqfam : EquilibriaFamily or list of Equilibria
        Equilibria to solve for at each step.
    objective : {"force", "energy", "vacuum"}
        function to solve for equilibrium solution
    optimizer : str or Optimzer (optional)
        optimizer to use
    pert_order : int or array of int
        order of perturbations to use. If array-like, should be same length as eqfam
        to specify different values for each step.
    ftol, xtol, gtol : float or array-like of float
        stopping tolerances for subproblem at each step. `None` will use defaults
        for given optimizer.
    nfev : int or array-like of int
        maximum number of function evaluations in each equilibrium subproblem.
    verbose : integer
        * 0: no output
        * 1: summary of each iteration
        * 2: as above plus timing information
        * 3: as above plus detailed solver output
    checkpoint_path : str or path-like
        file to save checkpoint data (Default value = None)

    Returns
    -------
    eqfam : EquilibriaFamily
        family of equilibria for the intermediate steps, where the last member is the
        final desired configuration,

    """
    timer = Timer()
    timer.start("Total time")
    pert_order, ftol, xtol, gtol, nfev, _ = np.broadcast_arrays(
        pert_order, ftol, xtol, gtol, nfev, eqfam
    )
    if isinstance(eqfam, (list, tuple)):
        eqfam = EquilibriaFamily(*eqfam)

    if not isinstance(optimizer, Optimizer):
        optimizer = Optimizer(optimizer)
    objective_i = get_equilibrium_objective(objective)
    constraints_i = get_fixed_boundary_constraints(
        iota=objective != "vacuum" and eqfam[0].iota is not None
    )

    ii = 0
    nn = len(eqfam)
    stop = False
    while ii < nn and not stop:
        timer.start("Iteration {} total".format(ii + 1))
        eqi = eqfam[ii]
        if verbose:
            _print_iteration_summary(
                ii,
                nn,
                eqi,
                _get_ratio(eqi.surface, eqfam[-1].surface),
                _get_ratio(eqi.pressure, eqfam[-1].pressure),
                _get_ratio(eqi.current, eqfam[-1].current),
                pert_order[ii],
                objective_i,
                optimizer,
            )
            deltas = {}

        if ii > 0:
            eqi.set_initial_guess(eqfam[ii - 1])
            # figure out if we need perturbations
            things1 = {
                "surface": eqfam[ii - 1].surface,
                "iota": eqfam[ii - 1].iota,
                "current": eqfam[ii - 1].current,
                "pressure": eqfam[ii - 1].pressure,
                "Psi": eqfam[ii - 1].Psi,
            }
            things2 = {
                "surface": eqi.surface,
                "iota": eqi.iota,
                "current": eqi.current,
                "pressure": eqi.pressure,
                "Psi": eqi.Psi,
            }
            deltas = get_deltas(things1, things2)

            # maybe rebuild objective if resolution changed.
            if eqfam[ii - 1].resolution != eqi.resolution:
                objective_i = get_equilibrium_objective(objective)
                constraints_i = get_fixed_boundary_constraints(
                    iota=objective != "vacuum" and eqfam[ii].iota is not None
                )

        if len(deltas) > 0:
            if verbose > 0:
                print("Perturbing equilibrium")
            # TODO: pass Jx if available
            eqp = eqfam[ii - 1].copy()
            eqp.change_resolution(**eqi.resolution)
            eqp.perturb(
                objective=objective_i,
                constraints=constraints_i,
                **deltas,
                order=pert_order[ii],
                verbose=verbose,
                copy=False,
            )
            eqi.R_lmn = eqp.R_lmn
            eqi.Z_lmn = eqp.Z_lmn
            eqi.L_lmn = eqp.L_lmn
            deltas = {}
            del eqp

        if not eqi.is_nested(msg="manual"):
            stop = True

        if not stop:
            eqi.solve(
                optimizer=optimizer,
                objective=objective_i,
                constraints=constraints_i,
                ftol=ftol[ii],
                xtol=xtol[ii],
                gtol=gtol[ii],
                verbose=verbose,
                maxiter=nfev[ii],
            )

        if not eqi.is_nested(msg="manual"):
            stop = True

        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)
        timer.stop("Iteration {} total".format(ii + 1))
        if verbose > 1:
            timer.disp("Iteration {} total".format(ii + 1))
        ii += 1

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


def _get_ratio(thing1, thing2):
    """Figure out bdry_ratio, pres_ratio etc from objects."""
    if thing1 is None or thing2 is None:
        return None
    if hasattr(thing1, "R_lmn"):  # treat it as surface
        R1 = thing1.R_lmn[thing1.R_basis.modes[:, 2] != 0]
        R2 = thing2.R_lmn[thing2.R_basis.modes[:, 2] != 0]
        Z1 = thing1.Z_lmn[thing1.Z_basis.modes[:, 2] != 0]
        Z2 = thing2.Z_lmn[thing2.Z_basis.modes[:, 2] != 0]
        num = np.linalg.norm([R1, Z1])
        den = np.linalg.norm([R2, Z2])
    else:
        num = np.linalg.norm(thing1.params)
        den = np.linalg.norm(thing2.params)
    if num == 0 and den == 0:
        return 1
    return num / den


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
    **kwargs,
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
    print(
        "Objective: {}".format(
            objective if isinstance(objective, str) else objective.objectives[0].name
        )
    )
    print(
        "Optimizer: {}".format(
            optimizer if isinstance(optimizer, str) else optimizer.method
        )
    )
    print("================")
