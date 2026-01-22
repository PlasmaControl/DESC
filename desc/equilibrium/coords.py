"""Functions for mapping between flux, sfl, and real space coordinates."""

import functools
from functools import partial

import numpy as np

from desc.backend import jit, jnp, rfft, root, root_scalar, vmap
from desc.batching import batch_map
from desc.compute import compute as compute_fun
from desc.compute import data_index, get_data_deps, get_profiles, get_transforms
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.transform import Transform
from desc.utils import (
    ResolutionWarning,
    check_posint,
    errorif,
    safenorm,
    setdefault,
    warnif,
)


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
    period=(np.inf, np.inf, np.inf),
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
        Default assumes no periodicity.
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
            errorif(
                np.isfinite(period[1]),
                msg=f"Period must be ∞ for inbasis={inbasis}, but got {period[1]}.",
            )
            if "iota" in kwargs:
                iota = kwargs.pop("iota")
            elif "profiles" in kwargs:
                iota = eq._compute_iota_under_jit(coords, params, **kwargs)
            else:
                iota = eq._compute_iota_under_jit(coords, params, profiles, **kwargs)
            rho, alpha, zeta = coords.T
            omega = 0  # TODO(#568)
            coords = jnp.column_stack([rho, alpha + iota * (zeta + omega), zeta])
            inbasis = ("rho", "theta_PEST", "zeta")
        if inbasis == ("rho", "theta_PEST", "zeta"):
            return _map_PEST_coordinates(
                coords=coords,
                L_lmn=params["L_lmn"],
                L_basis=eq.L_basis,
                guess=guess[:, 1] if guess is not None else None,
                period=period[1],
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
    period = np.asarray(period)
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
    errorif(
        np.isfinite(period) and period != (2 * jnp.pi),
        msg=f"Period must be ∞ or 2π, but got {period}.",
    )

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
    theta = vecroot(
        # Assume λ=0 for default initial guess.
        setdefault(guess, theta_PEST),
        theta_PEST,
        rho,
        zeta,
    )
    if full_output:
        theta, (res, niter) = theta
    out = jnp.column_stack([rho, jnp.atleast_1d(theta.squeeze()), zeta])
    if full_output:
        return out, (res, niter)
    return out


def _partial_sum(lmbda, L_lmn, omega, W_lmn, iota):
    """Convert FourierZernikeBasis to set of Fourier series.

    TODO(#1243) Do proper partial summation once the DESC
    basis are improved to store the padded tensor product modes.
    https://github.com/PlasmaControl/DESC/issues/1243#issuecomment-3131182128.
    The partial summation implemented here has a totally unnecessary FourierZernike
    spectral to real transform and unnecessary N^2 FFT's of size N. Still the
    performance improvement is significant. To avoid the transform and FFTs,
    I suggest padding the FourierZernike basis modes to make the partial summation
    trivial. Then this computation will likely take microseconds.

    Parameters
    ----------
    lmbda : Transform
        FourierZernikeBasis
    L_lmn : jnp.ndarray
        FourierZernikeBasis basis coefficients for λ.
    omega : Transform
        FourierZernikeBasis
    W_lmn : jnp.ndarray
        FourierZernikeBasis basis coefficients for ω.
    iota : jnp.ndarray
        Shape (lmbda.grid.num_rho, )

    Returns
    -------
    lmbda_minus_iota_omega, modes
        Spectral coefficients and modes.
        Shape (num rho, num zeta, num modes).

    """
    grid = lmbda.grid
    errorif(not grid.fft_poloidal, NotImplementedError, msg="See note in docstring.")
    # TODO(#1243): assert grid.sym==eq.sym once basis is padded for partial sum
    # TODO: (#568)
    warnif(
        grid.M > lmbda.basis.M,
        ResolutionWarning,
        msg="Poloidal grid resolution is higher than necessary for coordinate mapping.",
    )
    warnif(
        grid.M < lmbda.basis.M,
        ResolutionWarning,
        msg="High frequency lambda modes will be truncated in coordinate mapping.",
    )
    lmbda_minus_iota_omega = lmbda.transform(L_lmn)
    lmbda_minus_iota_omega = (
        rfft(grid.meshgrid_reshape(lmbda_minus_iota_omega, "rzt"), norm="forward")
        .at[..., (0, -1) if ((grid.num_theta % 2) == 0) else 0]
        .divide(2)
        * 2
    )
    return lmbda_minus_iota_omega, jnp.fft.rfftfreq(grid.num_theta, 1 / grid.num_theta)


@partial(jit, static_argnames=["tol", "maxiter"])
def _map_clebsch_coordinates(
    iota,
    alpha,
    zeta,
    L_lmn,
    lmbda,
    guess=None,
    *,
    tol=1e-6,
    maxiter=30,
    **kwargs,
):
    """Find θ for given Clebsch field line poloidal label α.

    # TODO: input (rho, alpha, zeta) coordinates may be an arbitrary point cloud
    #       and the partial summation will work without modification.
    #       Clean up input parameter API to support this.

    Parameters
    ----------
    iota : ndarray
        Shape (num iota, ).
        Rotational transform.
    alpha : ndarray
        Shape (num alpha, ).
        Field line labels.
    zeta : ndarray
        Shape (num zeta, ).
        DESC toroidal angle.
    L_lmn : jnp.ndarray
        Spectral coefficients for λ.
    lmbda : Transform
        Transform for λ built on DESC coordinates [ρ, θ, ζ].
    guess : jnp.ndarray
        Shape (num iota, num alpha, num zeta).
        Optional initial guess for the DESC computational coordinate θ solution.
    tol : float
        Stopping tolerance.
    maxiter : int
        Maximum number of Newton iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to ``root_scalar`` such as ``maxiter_ls``,
        ``alpha``.

    Returns
    -------
    theta : ndarray
        Shape (num iota, num alpha, num zeta).
        DESC computational coordinates θ at given input meshgrid.

    """
    # noqa: D202

    def rootfun(theta, target, c_m):
        c = (jnp.exp(1j * modes * theta) * c_m).real.sum()
        target_k = theta + c
        return target_k - target

    def jacfun(theta, target, c_m):
        dc_dt = ((1j * jnp.exp(1j * modes * theta) * c_m).real * modes).sum()
        return 1 + dc_dt

    @partial(jnp.vectorize, signature="(),(),(m)->()")
    def vecroot(guess, target, c_m):
        return root_scalar(
            rootfun,
            guess,
            jac=jacfun,
            args=(target, c_m),
            tol=tol,
            maxiter=maxiter,
            full_output=False,
            **kwargs,
        )

    c_m, modes = _partial_sum(lmbda, L_lmn, None, None, iota)
    c_m = c_m[:, jnp.newaxis]
    target = alpha[:, jnp.newaxis] + iota[:, jnp.newaxis, jnp.newaxis] * zeta
    # Assume λ − ι ω = 0 for default initial guess.
    return vecroot(setdefault(guess, target), target, c_m)


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
    tol=1e-9,
):
    """Transform this equilibrium to use straight field line PEST coordinates.

    Uses a least squares fit to find FourierZernike coefficients of R, Z, Rb, Zb
    with respect to the straight field line coordinates, rather than the boundary
    coordinates. The new lambda value will be zero.

    The flux surfaces of the returned equilibrium usually differ from the original
    by 1% when the default resolution parameters are used.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use
    L : int, optional
        Radial resolution to use for SFL equilibrium.
        Default is ``3*eq.L``.
    M : int, optional
        Poloidal resolution to use for SFL equilibrium.
        Default is ``4*eq.M``.
    N : int, optional
        toroidal resolution to use for SFL equilibrium.
        Default is ``3*eq.N``.
    L_grid : int, optional
        Radial grid resolution to use for fit to Zernike series.
        Default is ``1.5*L``.
    M_grid : int, optional
        Poloidal grid resolution to use for fit to Zernike series.
        Default is ``1.5*M``.
    N_grid : int, optional
        Toroidal grid resolution to use for fit to Fourier series.
        Default is ``N``.
    rcond : float, optional
        Cutoff for small singular values in the least squares fit.
    copy : bool, optional
        Whether to update the existing equilibrium or make a copy (Default).
    tol : float
        Tolerance for coordinate mapping.
        Default is ``1e-9``.

    Returns
    -------
    eq_PEST : Equilibrium
        Equilibrium transformed to a straight field line coordinate representation.

    """
    L = L or int(3 * eq.L)
    M = M or int(4 * eq.M)
    N = N or int(3 * eq.N)
    L_grid = L_grid or int(1.5 * L)
    M_grid = M_grid or int(1.5 * M)
    N_grid = N_grid or int(N)

    grid_PEST = ConcentricGrid(L_grid, M_grid, N_grid, node_pattern="ocs", NFP=eq.NFP)
    grid_PEST_bdry = LinearGrid(M=M, N=N, rho=1.0, NFP=eq.NFP)
    data = eq.compute(
        ["R", "Z", "lambda"],
        Grid(
            eq.map_coordinates(grid_PEST.nodes, ("rho", "theta_PEST", "zeta"), tol=tol)
        ),
    )
    data_bdry = eq.compute(
        ["R", "Z", "lambda"],
        Grid(
            eq.map_coordinates(
                grid_PEST_bdry.nodes, ("rho", "theta_PEST", "zeta"), tol=tol
            )
        ),
    )

    eq_PEST = eq.copy() if copy else eq
    eq_PEST.change_resolution(
        L,
        M,
        N,
        L_grid=max(eq_PEST.L_grid, L),
        M_grid=max(eq_PEST.M_grid, M),
        N_grid=max(eq_PEST.N_grid, N),
    )

    eq_PEST.R_lmn = Transform(
        grid_PEST,
        eq_PEST.R_basis,
        build=False,
        build_pinv=True,
        rcond=rcond,
    ).fit(data["R"])

    eq_PEST.Z_lmn = Transform(
        grid_PEST,
        eq_PEST.Z_basis,
        build=False,
        build_pinv=True,
        rcond=rcond,
    ).fit(data["Z"])

    eq_PEST.L_lmn = np.zeros_like(eq_PEST.L_lmn)

    eq_PEST.Rb_lmn = Transform(
        grid_PEST_bdry,
        eq_PEST.surface.R_basis,
        build=False,
        build_pinv=True,
        rcond=rcond,
    ).fit(data_bdry["R"])

    eq_PEST.Zb_lmn = Transform(
        grid_PEST_bdry,
        eq_PEST.surface.Z_basis,
        build=False,
        build_pinv=True,
        rcond=rcond,
    ).fit(data_bdry["Z"])

    return eq_PEST


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
