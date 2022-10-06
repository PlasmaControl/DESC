import numpy as np

from desc.backend import jnp
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.utils import Timer
from desc.optimize import Optimizer
from desc.objectives import (
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)


def solve_continuation_automatic(
    L,
    M,
    N,
    surface,
    pressure,
    iota=None,
    current=None,
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
    eqfam=None,
):
    """Solve for an equilibrium using an automatic continuation method.

    By default, the method first solves for a vacuum tokamak, then a finite beta
    tokamak, then a finite beta stellarator. Currently hard coded to take a fixed
    number of perturbation steps based on conservative estimates and testing. In the
    future, continuation stepping will be done adaptively.

    Parameters
    ----------
    L, M, N : int
        desired spectral resolution of the final equilibrium
    surface : Surface
        desired surface of the final equilibrium
    pressure, iota, current : Profile
        desired profiles of the final equilibrium. Must specify pressure and either
        iota or current
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
    verbose : integer
        * 0: no output
        * 1: summary of each iteration
        * 2: as above plus timing information
        * 3: as above plus detailed solver output
    checkpoint_path : str or path-like
        file to save checkpoint data (Default value = None)
    eqfam : EquilibriaFamily
        Family object to store results in. If None, a new one is created.

    Returns
    -------
    eqfam : EquilibriaFamily
        family of equilibria for the intermediate steps, where the last member is the
        final desired configuration,

    """

    assert (
        iota is None or current is None
    ), "Can only specify iota or current, not both."
    timer = Timer()
    timer.start("Total time")

    surface.change_resolution(L, M, N)
    surf_axisym = surface.copy()
    pres_vac = pressure.copy()

    L_grid = L_grid or 2 * L
    M_grid = M_grid or 2 * M
    N_grid = N_grid or 2 * N

    res_step = 6
    pres_step = 1 / 2
    bdry_step = 1 / 4
    Mi = min(M // 2, res_step)
    Li = 2 * Mi if spectral_indexing == "fringe" else Mi
    Ni = 0
    L_gridi = np.ceil(L_grid / L * Li).astype(int)
    M_gridi = np.ceil(M_grid / M * Mi).astype(int)
    N_gridi = np.ceil(N_grid / N * Ni).astype(int)

    # first we solve vacuum until we reach full L,M
    # then pressure
    # then 3d shaping
    res_steps = max(M // res_step, 1)
    pres_steps = (
        0
        if (abs(pressure(np.linspace(0, 1, 20))) < 1e-14).all()
        else int(np.ceil(1 / pres_step))
    )
    bdry_steps = 0 if N == 0 else int(np.ceil(1 / bdry_step))
    bdry_ratio = pres_ratio = curr_ratio = 0
    deltas = {}
    nn = res_steps + pres_steps + bdry_steps

    surf_axisym.change_resolution(L, M, 0)

    # start with zero pressure
    pres_vac.params *= 0

    eqi = Equilibrium(
        Psi,
        NFP,
        Li,
        Mi,
        Ni,
        L_gridi,
        M_gridi,
        N_gridi,
        node_pattern,
        pres_vac.copy(),
        iota,
        current,
        surf_axisym.copy(),
        None,
        sym,
        spectral_indexing,
    )
    optimizer = Optimizer(optimizer)

    eqfam = EquilibriaFamily() if eqfam is None else eqfam

    for ii in range(nn):
        timer.start("Iteration {} total".format(ii + 1))

        if ii < res_steps and ii > 0:
            # increase resolution of vacuum soln
            Mi = min(Mi + res_step, M)
            Li = 2 * Mi if spectral_indexing == "fringe" else Mi
            L_gridi = np.ceil(L_grid / L * Li).astype(int)
            M_gridi = np.ceil(M_grid / M * Mi).astype(int)
            N_gridi = np.ceil(N_grid / N * Ni).astype(int)
            eqi.change_resolution(Li, Mi, Ni, L_gridi, M_gridi, N_gridi)

            surf_i = eqi.surface
            surf_i2 = surface.copy()
            surf_i2.change_resolution(Li, Mi, Ni)
            deltas = _get_deltas({"surface": [surf_i, surf_i2]})
            surf_i = surf_i2

        if ii >= res_steps and ii < res_steps + pres_steps:
            # make sure its at full radial/poloidal resolution
            eqi.change_resolution(L=L, M=M, L_grid=L_grid, M_grid=M_grid)
            # increase pressure
            deltas = _get_deltas(
                {"pressure": [eqfam[res_steps - 1].pressure, pressure]}
            )
            deltas["dp"] *= pres_step
            pres_ratio += pres_step

        if ii >= res_steps + pres_steps:
            # boundary perturbations
            # TODO: do we want to jump to full N? or maybe step that up too?
            eqi.change_resolution(L, M, N, L_grid, M_grid, N_grid)
            surf_axisym.change_resolution(L, M, N)
            deltas = _get_deltas({"surface": [surf_axisym, surface]})
            if "dRb" in deltas:
                deltas["dRb"] *= bdry_step
            if "dZb" in deltas:
                deltas["dZb"] *= bdry_step
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
        objective_ = get_equilibrium_objective(objective)
        constraints_ = get_fixed_boundary_constraints(
            iota=objective != "vacuum" and iota is not None
        )

        if len(deltas) > 0:
            if verbose > 0:
                print("Perturbing equilibrium")
            eqi.perturb(
                objective=objective_,
                constraints=constraints_,
                **deltas,
                order=pert_order,
                verbose=verbose,
                copy=False,
            )
            deltas = {}

        if not eqi.is_nested(msg="auto"):
            break

        eqi.solve(objective_, constraints_, optimizer, ftol, xtol, gtol, nfev)
        if not eqi.is_nested(msg="auto"):
            break

        eqfam.append(eqi)
        eqi = eqfam[-1].copy()
        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)
        timer.stop("Iteration {} total".format(ii + 1))
        if verbose > 1:
            timer.disp("Iteration {} total".format(ii + 1))

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


def solve_continuation_manual(inputs, verbose=None, checkpoint_path=None, eqfam=None):
    """Solve for an equilibrium by continuation method.

        1. Creates an initial guess from the given inputs
        2. Find equilibrium flux surfaces by minimizing the objective function.
        3. Step up to higher resolution and perturb the previous solution
        4. Repeat 2 and 3 until at desired resolution

    Parameters
    ----------
    inputs : list of dict
        dictionaries with input parameters for each continuation step, eg, from InputReader
    verbose : integer
        * 0: no output
        * 1: summary of each iteration
        * 2: as above plus timing information
        * 3: as above plus detailed solver output
    checkpoint_path : str or path-like
        file to save checkpoint data (Default value = None)
    eqfam : EquilibriaFamily
        Family object to store results in. If None, a new one is created.

    Returns
    -------
    eqfam : EquilibriaFamily
        family of equilibria for the intermediate steps, where the last member is the
        final desired configuration,

    """
    timer = Timer()
    timer.start("Total time")
    eqfam = EquilibriaFamily(inputs) if eqfam is None else eqfam
    verbose = inputs[0]["verbose"] if verbose is None else verbose

    for ii in range(len(inputs)):
        timer.start("Iteration {} total".format(ii + 1))

        # TODO: make this more efficient (minimize re-building)
        optimizer = Optimizer(inputs[ii]["optimizer"])
        objective = get_equilibrium_objective(inputs[ii]["objective"])
        constraints = get_fixed_boundary_constraints(
            iota=inputs[ii]["objective"] != "vacuum" and "iota" in inputs[ii]
        )

        if ii == 0:
            equil = eqfam[ii]
            if verbose > 0:
                _print_iteration_summary(ii, len(inputs), equil, **inputs[ii])

        else:
            equil = eqfam[ii - 1].copy()
            eqfam.insert(ii, equil)

            equil.change_resolution(
                L=inputs[ii]["L"],
                M=inputs[ii]["M"],
                N=inputs[ii]["N"],
                L_grid=inputs[ii]["L_grid"],
                M_grid=inputs[ii]["M_grid"],
                N_grid=inputs[ii]["N_grid"],
            )

            if verbose > 0:
                _print_iteration_summary(ii, len(inputs), equil, **inputs[ii])

            # figure out if we need perturbations
            things = {
                "surface": (equil.surface, inputs[ii].get("surface")),
                "iota": (equil.iota, inputs[ii].get("iota")),
                "current": (equil.current, inputs[ii].get("current")),
                "pressure": (equil.pressure, inputs[ii].get("pressure")),
                "Psi": (equil.Psi, inputs[ii].get("Psi")),
            }
            deltas = _get_deltas(things)

            if len(deltas) > 0:
                if verbose > 0:
                    print("Perturbing equilibrium")
                # TODO: pass Jx if available
                equil.perturb(
                    objective=objective,
                    constraints=constraints,
                    **deltas,
                    order=inputs[ii]["pert_order"],
                    verbose=verbose,
                    copy=False,
                )

        if not equil.is_nested(msg="manual"):
            break

        equil.solve(
            optimizer=optimizer,
            objective=objective,
            constraints=constraints,
            ftol=inputs[ii]["ftol"],
            xtol=inputs[ii]["xtol"],
            gtol=inputs[ii]["gtol"],
            verbose=verbose,
            maxiter=inputs[ii]["nfev"],
        )

        if checkpoint_path is not None:
            if verbose > 0:
                print("Saving latest iteration")
            eqfam.save(checkpoint_path)
        timer.stop("Iteration {} total".format(ii + 1))
        if verbose > 1:
            timer.disp("Iteration {} total".format(ii + 1))

        if not equil.is_nested(msg="manual"):
            break

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
    """Compute differences between parameters.

    Parameters
    ----------
    things: dict
        should be dictionary with keys "surface", "iota", "current", "pressure"
        values should be a tuple of (current_thing, target_thing) where the _things are
        either objects of the appropriate type (Surface, Profile, etc) or ndarray.
        Finds deltas for a perturbation going from current to target.

    Returns
    -------
    deltas : dict of ndarray
        deltas to pass in to perturb

    """
    deltas = {}
    if "surface" in things and things["surface"][0] is not None:
        s1 = things["surface"][0].copy()
        s2 = things["surface"][1].copy()
        s2 = _arr2obj(s2, s1)
        s2.change_resolution(s1.L, s1.M, s1.N)
        if not np.allclose(s2.R_lmn, s1.R_lmn):
            deltas["dRb"] = s2.R_lmn - s1.R_lmn
        if not np.allclose(s2.Z_lmn, s1.Z_lmn):
            deltas["dZb"] = s2.Z_lmn - s1.Z_lmn

    if "iota" in things and things["iota"][0] is not None:
        i1 = things["iota"][0].copy()
        i2 = things["iota"][1].copy()
        i2 = _arr2obj(i2, i1)
        if hasattr(i2, "change_resolution") and hasattr(i1, "basis"):
            i2.change_resolution(i1.basis.L)
        if not np.allclose(i2.params, i1.params):
            deltas["di"] = i2.params - i1.params

    if "current" in things and things["current"][0] is not None:
        c1 = things["current"][0].copy()
        c2 = things["current"][1].copy()
        c2 = _arr2obj(c2, c1)
        if hasattr(c2, "change_resolution") and hasattr(c1, "basis"):
            c2.change_resolution(c1.basis.L)
        if not np.allclose(c2.params, c1.params):
            deltas["dc"] = c2.params - c1.params

    if "pressure" in things and things["pressure"][0] is not None:
        p1 = things["pressure"][0].copy()
        p2 = things["pressure"][1].copy()
        p2 = _arr2obj(p2, p1)
        if hasattr(p2, "change_resolution") and hasattr(p1, "basis"):
            p2.change_resolution(p1.basis.L)
        if not np.allclose(p2.params, p1.params):
            deltas["dp"] = p2.params - p1.params

    if "Psi" in things and things["Psi"][0] is not None:
        psi1 = things["Psi"][0]
        psi2 = things["Psi"][1]
        if not np.allclose(psi2, psi1):
            deltas["dPsi"] = psi2 - psi1

    return deltas


def _arr2obj(arr, obj):
    """Convert array of parameters to object of the appropriate type (surface, profile etc)."""
    if not isinstance(arr, (np.ndarray, jnp.ndarray)):
        return arr
    cls = obj.__class__
    if arr.shape[1] == 2:
        # treat it as a profile
        return cls(arr[:, 1], arr[:, 0])
    # treat it as a surface
    return cls(
        arr[:, 3],
        arr[:, 4],
        arr[:, 1:3].astype(int),
        arr[:, 1:3].astype(int),
        obj.NFP,
        obj.sym,
    )


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
    print("Function tolerance = {}".format(ftol))
    print("Gradient tolerance = {}".format(gtol))
    print("State vector tolerance = {}".format(xtol))
    print("Max function evaluations = {}".format(nfev))
    print("================")
