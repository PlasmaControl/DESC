"""Functions for solving for equilibria with multigrid continuation method."""

import copy
import warnings

import numpy as np
from termcolor import colored

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.objectives import get_equilibrium_objective, get_fixed_boundary_constraints
from desc.optimize import Optimizer
from desc.perturbations import get_deltas
from desc.utils import Timer

MIN_MRES_STEP = 1
MIN_PRES_STEP = 0.1
MIN_BDRY_STEP = 0.05


def _solve_axisym(
    eq,
    mres_step,
    objective="force",
    optimizer="lsq-exact",
    pert_order=2,
    ftol=None,
    xtol=None,
    gtol=None,
    maxiter=100,
    verbose=1,
    checkpoint_path=None,
):
    """Solve initial axisymmetric case with adaptive step sizing."""
    timer = Timer()

    surface = eq.surface
    pressure = eq.pressure
    L, M, N, L_grid, M_grid, N_grid = eq.L, eq.M, eq.N, eq.L_grid, eq.M_grid, eq.N_grid
    spectral_indexing = eq.spectral_indexing

    Mi = min(M // 2, mres_step) if mres_step > 0 else M
    Li = int(np.ceil(L / M) * Mi)
    Ni = 0
    L_gridi = np.ceil(L_grid / L * Li).astype(int)
    M_gridi = np.ceil(M_grid / M * Mi).astype(int)
    N_gridi = np.ceil(N_grid / max(N, 1) * Ni).astype(int)

    # first we solve vacuum until we reach full L,M
    mres_steps = int(max(np.ceil(M / mres_step), 1)) if mres_step > 0 else 0
    deltas = {}

    surf_axisym = surface.copy()
    pres_vac = pressure.copy()
    surf_axisym.change_resolution(L, M, Ni)
    # start with zero pressure
    pres_vac.params *= 0

    eqi = Equilibrium(
        Psi=eq.Psi,
        NFP=eq.NFP,
        L=Li,
        M=Mi,
        N=Ni,
        L_grid=L_gridi,
        M_grid=M_gridi,
        N_grid=N_gridi,
        node_pattern=eq.node_pattern,
        pressure=pres_vac.copy(),
        iota=copy.copy(eq.iota),  # have to use copy.copy here since may be None
        current=copy.copy(eq.current),
        surface=surf_axisym.copy(),
        sym=eq.sym,
        spectral_indexing=spectral_indexing,
    )

    if not isinstance(optimizer, Optimizer):
        optimizer = Optimizer(optimizer)
    constraints_i = get_fixed_boundary_constraints(
        eq=eqi,
        iota=objective != "vacuum" and eq.iota is not None,
        kinetic=eq.electron_temperature is not None,
    )
    objective_i = get_equilibrium_objective(eq=eqi, mode=objective)

    eqfam = EquilibriaFamily()

    ii = 0
    stop = False
    while ii < mres_steps and not stop:
        timer.start("Iteration {} total".format(ii + 1))

        if ii > 0:
            eqi = eqfam[-1].copy()
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

        if verbose:
            _print_iteration_summary(
                ii,
                None,
                eqi,
                0,
                0,
                1,
                pert_order,
                objective_i,
                optimizer,
            )

        constraints_i = get_fixed_boundary_constraints(
            eq=eqi,
            iota=objective != "vacuum" and eq.iota is not None,
            kinetic=eq.electron_temperature is not None,
        )
        objective_i = get_equilibrium_objective(eq=eqi, mode=objective)
        if len(deltas) > 0:
            if verbose > 0:
                print("Perturbing equilibrium")
            eqi.perturb(
                objective=objective_i,
                constraints=constraints_i,
                deltas=deltas,
                order=pert_order,
                verbose=verbose,
                copy=False,
            )
            deltas = {}

        stop = not eqi.is_nested()

        if not stop:
            eqi.solve(
                optimizer=optimizer,
                objective=objective_i,
                constraints=constraints_i,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=verbose,
                maxiter=maxiter,
            )
        stop = stop or not eqi.is_nested()
        eqfam.append(eqi)

        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)
        timer.stop("Iteration {} total".format(ii + 1))
        if verbose > 1:
            timer.disp("Iteration {} total".format(ii + 1))
        ii += 1

    if stop:
        if mres_step == MIN_MRES_STEP:
            raise RuntimeError(
                "Automatic continuation failed with mres_step=1, "
                + "something is probaby very wrong with your desired equilibrium."
            )
        else:
            warnings.warn(
                colored(
                    "WARNING: Automatic continuation failed with "
                    + f"mres_step={mres_step}, retrying with mres_step={mres_step//2}",
                    "yellow",
                )
            )
            return _solve_axisym(
                eq,
                mres_step // 2,
                objective,
                optimizer,
                pert_order,
                ftol,
                xtol,
                gtol,
                maxiter,
                verbose,
                checkpoint_path,
            )

    return eqfam


def _add_pressure(
    eq,
    eqfam,
    pres_step,
    objective="force",
    optimizer="lsq-exact",
    pert_order=2,
    ftol=None,
    xtol=None,
    gtol=None,
    maxiter=100,
    verbose=1,
    checkpoint_path=None,
):
    """Add pressure with adaptive step sizing."""
    timer = Timer()

    eqi = eqfam[-1].copy()
    eqfam_temp = eqfam.copy()
    # make sure its at full radial/poloidal resolution
    eqi.change_resolution(L=eq.L, M=eq.M, L_grid=eq.L_grid, M_grid=eq.M_grid)

    constraints_i = get_fixed_boundary_constraints(
        eq=eqi,
        iota=objective != "vacuum" and eq.iota is not None,
        kinetic=eq.electron_temperature is not None,
    )
    objective_i = get_equilibrium_objective(eq=eqi, mode=objective)

    pres_steps = (
        0
        if (abs(eq.pressure(np.linspace(0, 1, 20))) < 1e-14).all() or pres_step == 0
        else int(np.ceil(1 / pres_step))
    )
    pres_ratio = 0 if pres_steps else 1

    ii = len(eqfam_temp)
    stop = False
    while ii - len(eqfam_temp) < pres_steps and not stop:
        timer.start("Iteration {} total".format(ii + 1))
        # increase pressure
        deltas = get_deltas(
            {"pressure": eqfam_temp[-1].pressure}, {"pressure": eq.pressure}
        )
        deltas["p_l"] *= pres_step
        pres_ratio += pres_step

        if verbose:
            _print_iteration_summary(
                ii,
                None,
                eqi,
                0,
                pres_ratio,
                1,
                pert_order,
                objective_i,
                optimizer,
            )

        if len(deltas) > 0:
            if verbose > 0:
                print("Perturbing equilibrium")
            eqi.perturb(
                objective=objective_i,
                constraints=constraints_i,
                deltas=deltas,
                order=pert_order,
                verbose=verbose,
                copy=False,
            )
            deltas = {}

        stop = not eqi.is_nested()

        if not stop:
            eqi.solve(
                optimizer=optimizer,
                objective=objective_i,
                constraints=constraints_i,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=verbose,
                maxiter=maxiter,
            )
        stop = stop or not eqi.is_nested()
        eqfam.append(eqi)

        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)
        timer.stop("Iteration {} total".format(ii + 1))
        if verbose > 1:
            timer.disp("Iteration {} total".format(ii + 1))
        ii += 1

    if stop:
        if pres_step <= MIN_PRES_STEP:
            raise RuntimeError(
                "Automatic continuation failed with "
                + f"pres_step={pres_step}, something is probaby very wrong with your "
                + "desired equilibrium."
            )
        else:
            warnings.warn(
                colored(
                    "WARNING: Automatic continuation failed with "
                    + f"pres_step={pres_step}, retrying with pres_step={pres_step/2}",
                    "yellow",
                )
            )
            return _add_pressure(
                eq,
                eqfam_temp,
                pres_step / 2,
                objective,
                optimizer,
                pert_order,
                ftol,
                xtol,
                gtol,
                maxiter,
                verbose,
                checkpoint_path,
            )

    return eqfam


def _add_shaping(
    eq,
    eqfam,
    bdry_step,
    objective="force",
    optimizer="lsq-exact",
    pert_order=2,
    ftol=None,
    xtol=None,
    gtol=None,
    maxiter=100,
    verbose=1,
    checkpoint_path=None,
):
    """Add 3D shaping with adaptive step sizing."""
    timer = Timer()

    eqi = eqfam[-1].copy()
    eqfam_temp = eqfam.copy()
    # make sure its at full resolution
    eqi.change_resolution(eq.L, eq.M, eq.N, eq.L_grid, eq.M_grid, eq.N_grid)

    constraints_i = get_fixed_boundary_constraints(
        eq=eqi,
        iota=objective != "vacuum" and eq.iota is not None,
        kinetic=eq.electron_temperature is not None,
    )
    objective_i = get_equilibrium_objective(eq=eqi, mode=objective)

    bdry_steps = 0 if eq.N == 0 or bdry_step == 0 else int(np.ceil(1 / bdry_step))
    bdry_ratio = 0 if eq.N else 1

    surf_axisym = eq.surface.copy()
    surf_axisym.change_resolution(eq.L, eq.M, 0)
    surf_axisym.change_resolution(eq.L, eq.M, eq.N)

    ii = len(eqfam_temp)
    stop = False
    while ii - len(eqfam_temp) < bdry_steps and not stop:
        timer.start("Iteration {} total".format(ii + 1))
        # increase shaping
        deltas = get_deltas({"surface": surf_axisym}, {"surface": eq.surface})
        if "Rb_lmn" in deltas:
            deltas["Rb_lmn"] *= bdry_step
        if "Zb_lmn" in deltas:
            deltas["Zb_lmn"] *= bdry_step
        bdry_ratio += bdry_step

        if verbose:
            _print_iteration_summary(
                ii,
                None,
                eqi,
                bdry_ratio,
                1,
                1,
                pert_order,
                objective_i,
                optimizer,
            )

        if len(deltas) > 0:
            if verbose > 0:
                print("Perturbing equilibrium")
            eqi.perturb(
                objective=objective_i,
                constraints=constraints_i,
                deltas=deltas,
                order=pert_order,
                verbose=verbose,
                copy=False,
            )
            deltas = {}

        stop = not eqi.is_nested()

        if not stop:
            eqi.solve(
                optimizer=optimizer,
                objective=objective_i,
                constraints=constraints_i,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=verbose,
                maxiter=maxiter,
            )
        stop = stop or not eqi.is_nested()
        eqfam.append(eqi)

        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)
        timer.stop("Iteration {} total".format(ii + 1))
        if verbose > 1:
            timer.disp("Iteration {} total".format(ii + 1))
        ii += 1

    if stop:
        if bdry_step <= MIN_BDRY_STEP:
            raise RuntimeError(
                "Automatic continuation failed with "
                + f"bdry_step={bdry_step}, something is probaby very wrong with your "
                + "desired equilibrium."
            )
        else:
            warnings.warn(
                colored(
                    "WARNING: Automatic continuation failed with "
                    + f"bdry_step={bdry_step}, retrying with bdry_step={bdry_step/2}",
                    "yellow",
                )
            )
            return _add_shaping(
                eq,
                eqfam_temp,
                bdry_step / 2,
                objective,
                optimizer,
                pert_order,
                ftol,
                xtol,
                gtol,
                maxiter,
                verbose,
                checkpoint_path,
            )

    return eqfam


def solve_continuation_automatic(  # noqa: C901
    eq,
    objective="force",
    optimizer="lsq-exact",
    pert_order=2,
    ftol=None,
    xtol=None,
    gtol=None,
    maxiter=100,
    verbose=1,
    checkpoint_path=None,
    **kwargs,
):
    """Solve for an equilibrium using an automatic continuation method.

    By default, the method first solves for a no pressure tokamak, then a finite beta
    tokamak, then a finite beta stellarator. Steps in resolution, pressure, and 3D
    shaping are determined adaptively, and the method may backtrack to use smaller steps
    if the initial steps are too large.

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
    maxiter : int
        maximum number of iterations in each equilibrium subproblem.
    verbose : integer
        * 0: no output
        * 1: summary of each iteration
        * 2: as above plus timing information
        * 3: as above plus detailed solver output
    checkpoint_path : str or path-like
        file to save checkpoint data (Default value = None)
    **kwargs : dict, optional
        * ``mres_step``: int, default 6. The amount to increase Mpol by at each
          continuation step
        * ``pres_step``: float, ``0<=pres_step<=1``, default 0.5. The amount to
          increase pres_ratio by at each continuation step
        * ``bdry_step``: float, ``0<=bdry_step<=1``, default 0.25. The amount to
          increase bdry_ratio by at each continuation step

    Returns
    -------
    eqfam : EquilibriaFamily
        family of equilibria for the intermediate steps, where the last member is the
        final desired configuration,

    """
    if eq.electron_temperature is not None:
        raise NotImplementedError(
            "Continuation method with kinetic profiles is not currently supported"
        )

    timer = Timer()
    timer.start("Total time")

    mres_step = kwargs.pop("mres_step", 6)
    pres_step = kwargs.pop("pres_step", 1 / 2)
    bdry_step = kwargs.pop("bdry_step", 1 / 4)
    assert len(kwargs) == 0, "Got an unexpected kwarg {}".format(kwargs.keys())
    if not isinstance(optimizer, Optimizer):
        optimizer = Optimizer(optimizer)

    eqfam = _solve_axisym(
        eq,
        mres_step,
        objective,
        optimizer,
        pert_order,
        ftol,
        xtol,
        gtol,
        maxiter,
        verbose,
        checkpoint_path,
    )

    eqfam = _add_pressure(
        eq,
        eqfam,
        pres_step,
        objective,
        optimizer,
        pert_order,
        ftol,
        xtol,
        gtol,
        maxiter,
        verbose,
        checkpoint_path,
    )

    eqfam = _add_shaping(
        eq,
        eqfam,
        bdry_step,
        objective,
        optimizer,
        pert_order,
        ftol,
        xtol,
        gtol,
        maxiter,
        verbose,
        checkpoint_path,
    )

    eq.R_lmn = eqfam[-1].R_lmn
    eq.Z_lmn = eqfam[-1].Z_lmn
    eq.L_lmn = eqfam[-1].L_lmn
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
    maxiter=100,
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
    maxiter : int or array-like of int
        maximum number of iterations in each equilibrium subproblem.
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
    if not all([eq.electron_temperature is None for eq in eqfam]):
        raise NotImplementedError(
            "Continuation method with kinetic profiles is not currently supported"
        )

    timer = Timer()
    timer.start("Total time")
    pert_order, ftol, xtol, gtol, maxiter, _ = np.broadcast_arrays(
        pert_order, ftol, xtol, gtol, maxiter, eqfam
    )
    if isinstance(eqfam, (list, tuple)):
        eqfam = EquilibriaFamily(*eqfam)

    if not isinstance(optimizer, Optimizer):
        optimizer = Optimizer(optimizer)
    objective_i = get_equilibrium_objective(eq=eqfam[-1], mode=objective)
    constraints_i = get_fixed_boundary_constraints(
        eq=eqfam[-1],
        iota=objective != "vacuum" and eqfam[0].iota is not None,
        kinetic=eqfam[0].electron_temperature is not None,
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
                objective_i = get_equilibrium_objective(eq=eqfam[ii], mode=objective)
                constraints_i = get_fixed_boundary_constraints(
                    eq=eqfam[ii],
                    iota=objective != "vacuum" and eqfam[ii].iota is not None,
                    kinetic=eqfam[ii].electron_temperature is not None,
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
                deltas=deltas,
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
                maxiter=maxiter[ii],
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
    print(f"Step {ii+1}" + ("" if nn is None else f"/{nn}"))
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
