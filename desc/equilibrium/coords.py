"""Functions for mapping between flux, sfl, and real space coordinates."""

import functools

import numpy as np

from desc.backend import jit, jnp, root, root_scalar, vmap
from desc.batching import batch_map
from desc.compute import compute as compute_fun
from desc.compute import data_index, get_data_deps, get_profiles, get_transforms
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.transform import Transform
from desc.utils import check_posint, errorif, safenorm, setdefault, warnif


def _periodic(x, period):
    return jnp.where(jnp.isfinite(period), x % period, x)


def _fixup_residual(r, period):
    r = _periodic(r, period)
    # r should be between -period/2 and period/2
    return jnp.where((r > period / 2) & jnp.isfinite(period), -period + r, r)


def map_coordinates(  # noqa: C901
    eq,
    coords,
    inbasis,
    outbasis=("rho", "theta", "zeta"),
    guess=None,
    params=None,
    period=None,
    tol=1e-6,
    maxiter=30,
    full_output=False,
    **kwargs,
):
    """Transform coordinates given in ``inbasis`` to ``outbasis``.

    Solves for the computational coordinates that correspond to ``inbasis``,
    then evaluates ``outbasis`` at those locations.

    Performance can often improve significantly given a reasonable initial guess.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use.
    coords : ndarray
        Shape (k, 3).
        2D array of input coordinates. Each row is a different point in space.
    inbasis, outbasis : tuple of str
        Labels for input and output coordinates, e.g. ("R", "phi", "Z") or
        ("rho", "alpha", "zeta") or any combination thereof. Labels should be the
        same as the compute function data key.
    guess : jnp.ndarray
        Shape (k, 3).
        Initial guess for the computational coordinates ['rho', 'theta', 'zeta']
        corresponding to ``coords`` in ``inbasis``. If not given, then heuristics
        based on ``inbasis`` or a nearest neighbor search on a grid may be used.
        In general, this must be given to be compatible with JIT.
    params : dict
        Values of equilibrium parameters to use, e.g. ``eq.params_dict``.
    period : tuple of float
        Assumed periodicity for each quantity in ``inbasis``.
        Use ``np.inf`` to denote no periodicity.
    tol : float
        Stopping tolerance.
    maxiter : int
        Maximum number of Newton iterations.
    full_output : bool, optional
        If True, also return a tuple where the first element is the residual from
        the root finding and the second is the number of iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to ``root`` such as ``maxiter_ls``,
        ``alpha``.

    Returns
    -------
    out : jnp.ndarray
        Shape (k, 3).
        Coordinates mapped from ``inbasis`` to ``outbasis``. Values of NaN will be
        returned for coordinates where root finding did not succeed, possibly
        because the coordinate is not in the plasma volume.
    info : tuple
        2 element tuple containing residuals and number of iterations
        for each point. Only returned if ``full_output`` is True.

    """
    check_posint(maxiter, allow_none=False)
    errorif(
        not np.isfinite(tol) or tol <= 0,
        ValueError,
        f"tol must be a positive float, got {tol}",
    )
    inbasis = tuple(inbasis)
    outbasis = tuple(outbasis)
    params = setdefault(params, eq.params_dict)

    basis_derivs = tuple(f"{X}_{d}" for X in inbasis for d in ("r", "t", "z"))
    for key in basis_derivs:
        errorif(
            key not in data_index["desc.equilibrium.equilibrium.Equilibrium"],
            NotImplementedError,
            f"don't have recipe to compute partial derivative {key}",
        )

    profiles = (
        kwargs["profiles"]
        if "profiles" in kwargs
        else get_profiles(inbasis + basis_derivs, eq)
    )

    # TODO (#1382): make this work for permutations of in/out basis
    if outbasis == ("rho", "theta", "zeta"):
        if inbasis == ("rho", "alpha", "zeta"):
            if "iota" in kwargs:
                iota = kwargs.pop("iota")
            else:
                if profiles["iota"] is None:
                    profiles["iota"] = eq.get_profile(
                        ["iota", "iota_r"], params=params, **kwargs
                    )
                iota = profiles["iota"].compute(Grid(coords, sort=False, jitable=True))
            return _map_clebsch_coordinates(
                coords=coords,
                iota=iota,
                L_lmn=params["L_lmn"],
                L_basis=eq.L_basis,
                guess=guess[:, 1] if guess is not None else None,
                period=period[1] if period is not None else np.inf,
                tol=tol,
                maxiter=maxiter,
                full_output=full_output,
                **kwargs,
            )
        if inbasis == ("rho", "theta_PEST", "zeta"):
            return _map_PEST_coordinates(
                coords=coords,
                L_lmn=params["L_lmn"],
                L_basis=eq.L_basis,
                guess=guess[:, 1] if guess is not None else None,
                period=period[1] if period is not None else np.inf,
                tol=tol,
                maxiter=maxiter,
                full_output=full_output,
                **kwargs,
            )

    # do surface average to get iota once
    if "iota" in profiles and profiles["iota"] is None:
        profiles["iota"] = eq.get_profile(["iota", "iota_r"], params=params, **kwargs)
        params["i_l"] = profiles["iota"].params

    rhomin = kwargs.pop("rhomin", tol / 10)
    period = np.asarray(setdefault(period, (np.inf, np.inf, np.inf)))
    coords = _periodic(coords, period)

    p = "desc.equilibrium.equilibrium.Equilibrium"
    names = inbasis + basis_derivs + outbasis
    deps = list(set(get_data_deps(names, obj=p) + list(names)))

    @functools.partial(jit, static_argnums=1)
    def compute(y, basis):
        grid = Grid(y, sort=False, jitable=True)
        data = {}
        if "iota" in deps:
            data["iota"] = profiles["iota"].compute(grid, params=params["i_l"])
        if "iota_r" in deps:
            data["iota_r"] = profiles["iota"].compute(grid, dr=1, params=params["i_l"])
        if "iota_rr" in deps:
            data["iota_rr"] = profiles["iota"].compute(grid, dr=2, params=params["i_l"])
        transforms = get_transforms(basis, eq, grid, jitable=True)
        data = compute_fun(eq, basis, params, transforms, profiles, data)
        x = jnp.array([data[k] for k in basis]).T
        return x

    @jit
    def residual(y, coords):
        xk = compute(y, inbasis)
        return _fixup_residual(xk - coords, period)

    @jit
    def jac(y, coords):
        J = compute(y, basis_derivs)
        J = J.reshape((3, 3))
        return J

    @jit
    def fixup(y, *args):
        r, t, z = y.T
        # negative rho -> flip theta
        t = jnp.where(r < 0, (t + np.pi), t)
        r = jnp.abs(r)
        r = jnp.clip(r, rhomin, 1)
        y = jnp.array([r, t, z]).T
        return y

    yk = guess
    if yk is None:
        yk = _initial_guess_heuristic(yk, coords, inbasis, eq, profiles)
    if yk is None:
        yk = _initial_guess_nn_search(coords, inbasis, eq, period, compute)

    yk = fixup(yk)

    vecroot = jit(
        vmap(
            lambda x0, *p: root(
                residual,
                x0,
                jac=jac,
                args=p,
                fixup=fixup,
                tol=tol,
                maxiter=maxiter,
                full_output=full_output,
                **kwargs,
            )
        )
    )
    # See description here
    # https://github.com/PlasmaControl/DESC/pull/504#discussion_r1194172532
    # except we make sure properly handle periodic coordinates.
    if full_output:
        yk, (res, niter) = vecroot(yk, coords)
    else:
        yk = vecroot(yk, coords)

    out = compute(yk, outbasis)
    if full_output:
        return out, (res, niter)
    return out


def _initial_guess_heuristic(yk, coords, inbasis, eq, profiles):
    # some qtys have obvious initial guess based on coords
    # commonly, the desired coordinates are something like (radial, poloidal, toroidal)
    radialish = {"rho", "psi"}
    poloidalish = {"theta", "theta_PEST", "alpha"}
    toroidalish = {"zeta", "phi"}

    radial = list(set(inbasis).intersection(radialish))
    poloidal = list(set(inbasis).intersection(poloidalish))
    toroidal = list(set(inbasis).intersection(toroidalish))
    if len(radial) != 1 or len(poloidal) != 1 or len(toroidal) != 1:
        # no heuristics for other cases
        return yk

    rho = theta = zeta = None

    radial = radial[0]
    poloidal = poloidal[0]
    toroidal = toroidal[0]
    if radial == "rho":
        rho = coords[:, inbasis.index("rho")]
    elif radial == "psi":
        rho = jnp.sqrt(coords[:, inbasis.index("psi")] / eq.Psi)

    # omega usually small (zero for now)
    zeta = coords[:, inbasis.index(toroidal)]

    if poloidal == "theta" or poloidal == "theta_PEST":  # lambda usually small
        theta = coords[:, inbasis.index(poloidal)]
    elif poloidal == "alpha":
        alpha = coords[:, inbasis.index("alpha")]
        rho = jnp.atleast_1d(rho)
        zero = jnp.zeros_like(rho)
        grid = Grid(nodes=jnp.column_stack([rho, zero, zero]), sort=False, jitable=True)
        iota = profiles["iota"].compute(grid)
        theta = alpha + iota * zeta

    yk = jnp.column_stack([rho, theta, zeta])
    return yk


def _initial_guess_nn_search(coords, inbasis, eq, period, compute):
    # nearest neighbor search on dense grid
    yg = ConcentricGrid(eq.L_grid, eq.M_grid, max(eq.N_grid, eq.M_grid)).nodes
    xg = compute(yg, inbasis)
    coords = jnp.asarray(coords)

    def _distance_body(coords):
        distance = safenorm(_fixup_residual(coords - xg, period), axis=-1)
        return jnp.argmin(distance, axis=-1)

    idx = batch_map(_distance_body, coords[..., jnp.newaxis, :], 20)
    return yg[idx]


# TODO(#568): decide later whether to assume given phi instead of zeta.
def _map_PEST_coordinates(
    coords,
    L_lmn,
    L_basis,
    guess,
    period=np.inf,
    tol=1e-6,
    maxiter=30,
    full_output=False,
    **kwargs,
):
    """Find θ (theta_DESC) for given straight field line ϑ (theta_PEST).

    Parameters
    ----------
    coords : ndarray
        Shape (k, 3).
        Straight field line PEST coordinates [ρ, ϑ, ϕ]. Assumes ζ = ϕ.
        Each row is a different point in space.
    L_lmn : jnp.ndarray
        Spectral coefficients for lambda.
    L_basis : Basis
        Spectral basis for lambda.
    guess : jnp.ndarray
        Shape (k, ).
        Optional initial guess for the computational coordinates.
    period : float
        Assumed periodicity for ϑ.
        Use ``np.inf`` to denote no periodicity.
    tol : float
        Stopping tolerance.
    maxiter : int
        Maximum number of Newton iterations.
    full_output : bool, optional
        If True, also return a tuple where the first element is the residual from
        the root finding and the second is the number of iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to ``root_scalar`` such as ``maxiter_ls``,
        ``alpha``.

    Returns
    -------
    out : ndarray
        Shape (k, 3).
        DESC computational coordinates [ρ, θ, ζ].
    info : tuple
        2 element tuple containing residuals and number of iterations for each point.
        Only returned if ``full_output`` is True.

    """
    # noqa: D202

    # Root finding for θₖ such that r(θₖ) = ϑₖ(ρ, θₖ, ζ) − ϑ = 0.
    def rootfun(theta, theta_PEST, rho, zeta):
        nodes = jnp.array([rho.squeeze(), theta.squeeze(), zeta.squeeze()], ndmin=2)
        A = L_basis.evaluate(nodes)
        lmbda = A @ L_lmn
        theta_PEST_k = theta + lmbda
        return _fixup_residual(theta_PEST_k - theta_PEST, period).squeeze()

    def jacfun(theta, theta_PEST, rho, zeta):
        # Valid everywhere except θ such that θ+λ = k period where k ∈ ℤ.
        nodes = jnp.array([rho.squeeze(), theta.squeeze(), zeta.squeeze()], ndmin=2)
        A1 = L_basis.evaluate(nodes, (0, 1, 0))
        lmbda_t = jnp.dot(A1, L_lmn)
        return 1 + lmbda_t.squeeze()

    def fixup(x, *args):
        return _periodic(x, period)

    vecroot = jit(
        vmap(
            lambda x0, *p: root_scalar(
                rootfun,
                x0,
                jac=jacfun,
                args=p,
                fixup=fixup,
                tol=tol,
                maxiter=maxiter,
                full_output=full_output,
                **kwargs,
            )
        )
    )
    rho, theta_PEST, zeta = coords.T
    if full_output:
        theta, (res, niter) = vecroot(
            # Assume λ=0 for default initial guess.
            setdefault(guess, theta_PEST),
            theta_PEST,
            rho,
            zeta,
        )
    else:
        theta = vecroot(
            # Assume λ=0 for default initial guess.
            setdefault(guess, theta_PEST),
            theta_PEST,
            rho,
            zeta,
        )
    out = jnp.column_stack([rho, jnp.atleast_1d(theta.squeeze()), zeta])
    if full_output:
        return out, (res, niter)
    return out


# TODO(#568): decide later whether to assume given phi instead of zeta.
def _map_clebsch_coordinates(
    coords,
    iota,
    L_lmn,
    L_basis,
    guess=None,
    period=np.inf,
    tol=1e-6,
    maxiter=30,
    full_output=False,
    **kwargs,
):
    """Find θ for given Clebsch field line poloidal label α.

    Parameters
    ----------
    coords : ndarray
        Shape (k, 3).
        Clebsch field line coordinates [ρ, α, ζ]. Assumes ζ = ϕ.
        Each row is a different point in space.
    iota : ndarray
        Shape (k, ).
        Rotational transform on each node.
    L_lmn : jnp.ndarray
        Spectral coefficients for lambda.
    L_basis : Basis
        Spectral basis for lambda.
    guess : jnp.ndarray
        Shape (k, ).
        Optional initial guess for the computational coordinates.
    period : float
        Assumed periodicity for α.
        Use ``np.inf`` to denote no periodicity.
    tol : float
        Stopping tolerance.
    maxiter : int
        Maximum number of Newton iterations.
    full_output : bool, optional
        If True, also return a tuple where the first element is the residual from
        the root finding and the second is the number of iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to ``root_scalar`` such as ``maxiter_ls``,
        ``alpha``.

    Returns
    -------
    out : ndarray
        Shape (k, 3).
        DESC computational coordinates [ρ, θ, ζ].
    info : tuple
        2 element tuple containing residuals and number of iterations for each point.
        Only returned if ``full_output`` is True.

    """
    # noqa: D202

    # Root finding for θₖ such that r(θₖ) = αₖ(ρ, θₖ, ζ) − α = 0.
    def rootfun(theta, alpha, rho, zeta, iota):
        nodes = jnp.array([rho.squeeze(), theta.squeeze(), zeta.squeeze()], ndmin=2)
        A = L_basis.evaluate(nodes)
        lmbda = A @ L_lmn
        alpha_k = theta + lmbda - iota * zeta
        return _fixup_residual(alpha_k - alpha, period).squeeze()

    def jacfun(theta, alpha, rho, zeta, iota):
        # Valid everywhere except θ such that θ+λ = k period where k ∈ ℤ.
        nodes = jnp.array([rho.squeeze(), theta.squeeze(), zeta.squeeze()], ndmin=2)
        A1 = L_basis.evaluate(nodes, (0, 1, 0))
        lmbda_t = jnp.dot(A1, L_lmn)
        return 1 + lmbda_t.squeeze()

    def fixup(x, *args):
        return _periodic(x, period)

    vecroot = jit(
        vmap(
            lambda x0, *p: root_scalar(
                rootfun,
                x0,
                jac=jacfun,
                args=p,
                fixup=fixup,
                tol=tol,
                maxiter=maxiter,
                full_output=full_output,
                **kwargs,
            )
        )
    )
    rho, alpha, zeta = coords.T
    if guess is None:
        # Assume λ=0 for default initial guess.
        guess = alpha + iota * zeta
    if full_output:
        theta, (res, niter) = vecroot(guess, alpha, rho, zeta, iota)
    else:
        theta = vecroot(guess, alpha, rho, zeta, iota)

    out = jnp.column_stack([rho, jnp.atleast_1d(theta.squeeze()), zeta])
    if full_output:
        return out, (res, niter)
    return out


def is_nested(eq, grid=None, R_lmn=None, Z_lmn=None, L_lmn=None, msg=None):
    """Check that an equilibrium has properly nested flux surfaces in a plane.

    Does so by checking coordinate Jacobian (sqrt(g)) sign.
    If coordinate Jacobian switches sign somewhere in the volume, this
    indicates that it is zero at some point, meaning surfaces are touching and
    the equilibrium is not nested.

    NOTE: If grid resolution used is too low, or the solution is just barely
    unnested, this function may fail to return the correct answer.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use
    grid  :  Grid, optional
        Grid on which to evaluate the coordinate Jacobian and check for the sign.
        (Default to QuadratureGrid with eq's current grid resolutions)
    R_lmn, Z_lmn, L_lmn : ndarray, optional
        spectral coefficients for R, Z, lambda. Defaults to eq.R_lmn, eq.Z_lmn
    msg : {None, "auto", "manual"}
        Warning to throw if unnested.

    Returns
    -------
    is_nested : bool
        whether the surfaces are nested

    """
    if R_lmn is None:
        R_lmn = eq.R_lmn
    if Z_lmn is None:
        Z_lmn = eq.Z_lmn
    if L_lmn is None:
        L_lmn = eq.L_lmn
    if grid is None:
        grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)

    transforms = get_transforms("sqrt(g)_PEST", obj=eq, grid=grid)
    data = compute_fun(
        "desc.equilibrium.equilibrium.Equilibrium",
        "sqrt(g)_PEST",
        params={"R_lmn": R_lmn, "Z_lmn": Z_lmn, "L_lmn": L_lmn},
        transforms=transforms,
        profiles={},  # no profiles needed
    )

    nested = jnp.all(
        jnp.sign(data["sqrt(g)_PEST"][0]) == jnp.sign(data["sqrt(g)_PEST"])
    )
    warnif(
        not nested and msg is not None,
        RuntimeWarning,
        "Flux surfaces are no longer nested, exiting early. "
        + {
            "auto": "Automatic continuation method failed, consider specifying "
            "continuation steps manually.",
            "manual": "Consider taking smaller perturbation/resolution steps "
            "or reducing trust radius.",
            None: "",
        }[msg],
    )
    return nested


def to_sfl(
    eq,
    L=None,
    M=None,
    N=None,
    L_grid=None,
    M_grid=None,
    N_grid=None,
    rcond=None,
    copy=False,
):
    """Transform this equilibrium to use straight field line coordinates.

    Uses a least squares fit to find FourierZernike coefficients of R, Z, Rb, Zb
    with respect to the straight field line coordinates, rather than the boundary
    coordinates. The new lambda value will be zero.

    NOTE: Though the converted equilibrium will have the same flux surfaces,
    the force balance error will likely be higher than the original equilibrium.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use
    L : int, optional
        radial resolution to use for SFL equilibrium. Default = 1.5*eq.L
    M : int, optional
        poloidal resolution to use for SFL equilibrium. Default = 1.5*eq.M
    N : int, optional
        toroidal resolution to use for SFL equilibrium. Default = 1.5*eq.N
    L_grid : int, optional
        radial spatial resolution to use for fit to new basis. Default = 2*L
    M_grid : int, optional
        poloidal spatial resolution to use for fit to new basis. Default = 2*M
    N_grid : int, optional
        toroidal spatial resolution to use for fit to new basis. Default = 2*N
    rcond : float, optional
        cutoff for small singular values in the least squares fit.
    copy : bool, optional
        Whether to update the existing equilibrium or make a copy (Default).

    Returns
    -------
    eq_sfl : Equilibrium
        Equilibrium transformed to a straight field line coordinate representation.

    """
    L = L or int(1.5 * eq.L)
    M = M or int(1.5 * eq.M)
    N = N or int(1.5 * eq.N)
    L_grid = L_grid or int(2 * L)
    M_grid = M_grid or int(2 * M)
    N_grid = N_grid or int(2 * N)

    grid = ConcentricGrid(L_grid, M_grid, N_grid, node_pattern="ocs", NFP=eq.NFP)
    bdry_grid = LinearGrid(M=M, N=N, rho=1.0, NFP=eq.NFP)

    toroidal_coords = eq.compute(["R", "Z", "lambda"], grid=grid)
    theta = grid.nodes[:, 1]
    vartheta = theta + toroidal_coords["lambda"]
    sfl_grid = Grid(np.array([grid.nodes[:, 0], vartheta, grid.nodes[:, 2]]).T)

    bdry_coords = eq.compute(["R", "Z", "lambda"], grid=bdry_grid)
    bdry_theta = bdry_grid.nodes[:, 1]
    bdry_vartheta = bdry_theta + bdry_coords["lambda"]
    bdry_sfl_grid = Grid(
        np.array([bdry_grid.nodes[:, 0], bdry_vartheta, bdry_grid.nodes[:, 2]]).T
    )

    if copy:
        eq_sfl = eq.copy()
    else:
        eq_sfl = eq
    eq_sfl.change_resolution(L, M, N)

    R_sfl_transform = Transform(
        sfl_grid, eq_sfl.R_basis, build=False, build_pinv=True, rcond=rcond
    )
    R_lmn_sfl = R_sfl_transform.fit(toroidal_coords["R"])
    del R_sfl_transform  # these can take up a lot of memory so delete when done.

    Z_sfl_transform = Transform(
        sfl_grid, eq_sfl.Z_basis, build=False, build_pinv=True, rcond=rcond
    )
    Z_lmn_sfl = Z_sfl_transform.fit(toroidal_coords["Z"])
    del Z_sfl_transform
    L_lmn_sfl = np.zeros_like(eq_sfl.L_lmn)

    R_sfl_bdry_transform = Transform(
        bdry_sfl_grid,
        eq_sfl.surface.R_basis,
        build=False,
        build_pinv=True,
        rcond=rcond,
    )
    Rb_lmn_sfl = R_sfl_bdry_transform.fit(bdry_coords["R"])
    del R_sfl_bdry_transform

    Z_sfl_bdry_transform = Transform(
        bdry_sfl_grid,
        eq_sfl.surface.Z_basis,
        build=False,
        build_pinv=True,
        rcond=rcond,
    )
    Zb_lmn_sfl = Z_sfl_bdry_transform.fit(bdry_coords["Z"])
    del Z_sfl_bdry_transform

    eq_sfl.Rb_lmn = Rb_lmn_sfl
    eq_sfl.Zb_lmn = Zb_lmn_sfl
    eq_sfl.R_lmn = R_lmn_sfl
    eq_sfl.Z_lmn = Z_lmn_sfl
    eq_sfl.L_lmn = L_lmn_sfl

    return eq_sfl


def get_rtz_grid(
    eq,
    radial,
    poloidal,
    toroidal,
    coordinates,
    period=(np.inf, np.inf, np.inf),
    jitable=True,
    **kwargs,
):
    """Return DESC grid in (rho, theta, zeta) coordinates from given coordinates.

    Create a tensor-product grid from the given coordinates, and return the same
    grid in DESC coordinates.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium on which to perform coordinate mapping.
    radial : ndarray
        Sorted unique radial coordinates.
    poloidal : ndarray
        Sorted unique poloidal coordinates.
    toroidal : ndarray
        Sorted unique toroidal coordinates.
    coordinates : str
        Input coordinates that are specified by the arguments, respectively.
        raz : rho, alpha, zeta
        rvp : rho, theta_PEST, phi
        rtz : rho, theta, zeta
    period : tuple of float
        Assumed periodicity of the given coordinates.
        Use ``np.inf`` to denote no periodicity.
    jitable : bool, optional
        If false the returned grid has additional attributes.
        Required to be false to retain nodes at magnetic axis.
    kwargs
        Additional parameters to supply to the coordinate mapping function.
        See ``desc.equilibrium.coords.map_coordinates``.

    Returns
    -------
    desc_grid : Grid
        DESC coordinate grid for the given coordinates.

    """
    grid = Grid.create_meshgrid(
        [radial, poloidal, toroidal],
        coordinates=coordinates,
        period=period,
        jitable=jitable,
    )
    if "iota" in kwargs:
        kwargs["iota"] = grid.expand(jnp.atleast_1d(kwargs["iota"]))
    inbasis = {
        "r": "rho",
        "t": "theta",
        "v": "theta_PEST",
        "a": "alpha",
        "z": "zeta",
        "p": "phi",
    }
    rtz_nodes = map_coordinates(
        eq,
        grid.nodes,
        inbasis=[inbasis[char] for char in coordinates],
        outbasis=("rho", "theta", "zeta"),
        period=period,
        **kwargs,
    )
    idx = {}
    if inbasis[coordinates[0]] == "rho":
        # Should work as long as inbasis radial coordinate is
        # single variable, monotonic increasing function of rho.
        idx["_unique_rho_idx"] = grid.unique_rho_idx
        idx["_inverse_rho_idx"] = grid.inverse_rho_idx
    desc_grid = Grid(
        nodes=rtz_nodes,
        coordinates="rtz",
        source_grid=grid,
        sort=False,
        jitable=jitable,
        **idx,
    )
    return desc_grid


# TODO(#1383): deprecated, remove eventually
def compute_theta_coords(
    eq, flux_coords, L_lmn=None, tol=1e-6, maxiter=20, full_output=False, **kwargs
):
    """Find θ (theta_DESC) for given straight field line ϑ (theta_PEST).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use.
    flux_coords : ndarray
        Shape (k, 3).
        Straight field line PEST coordinates [ρ, ϑ, ϕ]. Assumes ζ = ϕ.
        Each row is a different point in space.
    L_lmn : ndarray
        Spectral coefficients for lambda. Defaults to ``eq.L_lmn``.
    tol : float
        Stopping tolerance.
    maxiter : int
        Maximum number of Newton iterations.
    full_output : bool, optional
        If True, also return a tuple where the first element is the residual from
        the root finding and the second is the number of iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to ``root_scalar`` such as
        ``maxiter_ls``, ``alpha``.

    Returns
    -------
    coords : ndarray
        Shape (k, 3).
        DESC computational coordinates [ρ, θ, ζ].
    info : tuple
        2 element tuple containing residuals and number of iterations for each
        point. Only returned if ``full_output`` is True.

    """
    return eq.compute_theta_coords(
        flux_coords, L_lmn, tol, maxiter, full_output, **kwargs
    )
