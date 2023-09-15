"""Functions for mapping between flux, sfl, and real space coordinates."""

import functools
import warnings

import numpy as np
from termcolor import colored

from desc.backend import fori_loop, jit, jnp, put, while_loop
from desc.compute import compute as compute_fun
from desc.compute import data_index, get_params, get_profiles, get_transforms
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.transform import Transform
from desc.utils import Index


def map_coordinates(  # noqa: C901
    eq,
    coords,
    inbasis,
    outbasis=("rho", "theta", "zeta"),
    guess=None,
    period=(np.inf, np.inf, np.inf),
    tol=1e-6,
    maxiter=30,
    **kwargs,
):
    """Given coordinates in inbasis, compute corresponding coordinates in outbasis.

    First solves for the computational coordinates that correspond to inbasis, then
    evaluates outbasis at those locations.

    Speed can often be significantly improved by providing a reasonable initial guess.
    The default is a nearest neighbor search on a coarse grid.

    NOTE: this function cannot be JIT compiled or differentiated with AD.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use
    coords : ndarray, shape(k,3)
        2D array of input coordinates. Each row is a different
        point in space.
    inbasis, outbasis : tuple of str
        Labels for input and output coordinates, eg ("R", "phi", "Z") or
        ("rho", "alpha", "zeta") or any combination thereof. Labels should be the
        same as the compute function data key
    guess : None or ndarray, shape(k,3)
        Initial guess for the computational coordinates ['rho', 'theta', 'zeta']
        corresponding to coords in inbasis. If None, heuristics are used based on
        in basis and a nearest neighbor search on a coarse grid.
    period : tuple of float
        Assumed periodicity for each quantity in inbasis.
        Use np.inf to denote no periodicity.
    tol : float
        Stopping tolerance.
    maxiter : int > 0
        Maximum number of Newton iterations

    Returns
    -------
    coords : ndarray, shape(k,3)
        Coordinates mapped from inbasis to outbasis. Values of NaN will be returned
        for coordinates where root finding did not succeed, possibly because the
        coordinate is not in the plasma volume.

    """
    inbasis = tuple(inbasis)
    outbasis = tuple(outbasis)
    assert (
        np.isfinite(maxiter) and maxiter > 0
    ), f"maxiter must be a positive integer, got {maxiter}"
    assert np.isfinite(tol) and tol > 0, f"tol must be a positive float, got {tol}"

    basis_derivs = tuple([f"{X}_{d}" for X in inbasis for d in ("r", "t", "z")])
    for key in basis_derivs:
        assert (
            key in data_index["desc.equilibrium.equilibrium.Equilibrium"]
        ), f"don't have recipe to compute partial derivative {key}"

    rhomin = kwargs.pop("rhomin", tol / 10)
    alpha = kwargs.pop("backtrack_frac", 0.1)
    maxls = kwargs.pop("max_backtrack", 10)
    assert len(kwargs) == 0, f"map_coordinates got unexpected kwargs: {kwargs.keys()}"
    period = np.asarray(period)
    coords = coords % period

    params = get_params(inbasis + basis_derivs, eq)

    @functools.partial(jit, static_argnums=1)
    def compute(y, basis):
        grid = Grid(y, sort=False, jitable=True)
        profiles = get_profiles(inbasis + basis_derivs, eq, grid, jitable=True)
        # do surface average to get iota once
        if "current" in profiles and profiles["current"] is not None:
            profiles["iota"] = eq.get_profile("iota")
        transforms = get_transforms(inbasis + basis_derivs, eq, grid, jitable=True)
        data = compute_fun(eq, basis, params, transforms, profiles)
        x = jnp.stack([data[k] for k in basis], axis=-1)
        return x

    @jit
    def residual(y):
        xk = compute(y, inbasis)
        r = xk % period - coords % period
        return jnp.where(r > period / 2, -period + r, r)

    @jit
    def jac(y):
        J = compute(y, basis_derivs)
        J = J.reshape((-1, 3, 3))
        return J

    @jit
    def fixup(y):
        r, t, z = y.T
        # negative rho -> flip theta
        t = jnp.where(r < 0, (t + np.pi), t)
        r = jnp.abs(r)
        r = jnp.clip(r, rhomin, 1)
        y = jnp.array([r, t, z]).T
        return y

    # use least squares to avoid singular behavior, cheap for 3x3 systems
    lstsq = jit(
        jnp.vectorize(
            lambda A, b: jnp.linalg.lstsq(A, b)[0], signature="(n,n),(n)->(n)"
        )
    )

    yk = guess
    if yk is None:
        # nearest neighbor search on coarse grid for initial guess
        yg = ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP).nodes
        xg = compute(yg, inbasis)
        idx = jnp.zeros(len(coords)).astype(int)
        coords = jnp.asarray(coords)

        def _distance_body(i, idx):
            d = (coords[i] % period) - (xg % period)
            d = jnp.where(d > period / 2, period - d, d)
            distance = jnp.linalg.norm(d, axis=-1)
            k = jnp.argmin(distance)
            idx = put(idx, i, k)
            return idx

        idx = fori_loop(0, len(coords), _distance_body, idx)
        yk = yg[idx]

        # apply some heuristics based on common patterns
        if "rho" in inbasis:
            yk = put(yk.T, 0, coords[:, inbasis.index("rho")]).T
        if "theta" in inbasis:
            yk = put(yk.T, 1, coords[:, inbasis.index("theta")]).T
        elif "theta_PEST" in inbasis:  # lambda is usually small
            yk = put(yk.T, 1, coords[:, inbasis.index("theta_PEST")]).T
        if "zeta" in inbasis:
            yk = put(yk.T, 2, coords[:, inbasis.index("zeta")]).T
        elif "phi" in inbasis:
            yk = put(yk.T, 2, coords[:, inbasis.index("phi")]).T

    yk = fixup(yk)
    resk = residual(yk)

    for i in range(maxiter):
        if np.max(np.abs(resk)) < tol:
            break
        J = jac(yk)
        d = lstsq(J, resk)
        alphak = jnp.ones(yk.shape[0])
        for j in range(maxls):
            # backtracking line search
            yt = fixup(yk - alphak[:, None] * d)
            res = residual(yt)
            if np.max(np.abs(res)) < np.max(np.abs(resk)):
                yk = yt
                resk = res
                break
            else:
                # only reduce step size where needed
                alphak = jnp.where(
                    np.max(np.abs(res), axis=-1) > np.max(np.abs(resk), axis=-1),
                    alpha * alphak,
                    alphak,
                )
        else:
            # if we haven't found a good step, try taking a bad one and
            # hope the next one is better
            yk = yt
            resk = res

    r, t, z = yk.T
    res = jnp.max(jnp.abs(residual(yk)), axis=-1)
    r = jnp.where(res > tol, jnp.nan, r)
    t = jnp.where(res > tol, jnp.nan, t)
    z = jnp.where(res > tol, jnp.nan, z)

    yk = jnp.vstack([r, t, z]).T

    out = compute(yk, outbasis)

    return out


def compute_theta_coords(eq, flux_coords, L_lmn=None, tol=1e-6, maxiter=20):
    """Find geometric theta for given straight field line theta.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use
    flux_coords : ndarray, shape(k,3)
        2d array of flux coordinates [rho,theta*,zeta]. Each row is a different
        point in space.
    L_lmn : ndarray
        spectral coefficients for lambda. Defaults to eq.L_lmn
    tol : float
        Stopping tolerance.
    maxiter : int > 0
        maximum number of Newton iterations

    Returns
    -------
    coords : ndarray, shape(k,3)
        coordinates [rho,theta,zeta]. If Newton method doesn't converge for
        a given coordinate nan will be returned for those values

    """
    if L_lmn is None:
        L_lmn = eq.L_lmn
    rho, theta_star, zeta = flux_coords.T
    if maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got{maxiter}")
    if jnp.any(rho < 0):
        raise ValueError("rho values must be positive")

    # Note: theta* (also known as vartheta) is the poloidal straight field-line
    # angle in PEST-like flux coordinates

    nodes = flux_coords.copy()
    A0 = eq.L_basis.evaluate(nodes, (0, 0, 0))

    # theta* = theta + lambda
    lmbda = jnp.dot(A0, L_lmn)
    k = 0

    def cond_fun(nodes_k_lmbda):
        nodes, k, lmbda = nodes_k_lmbda
        theta_star_k = nodes[:, 1] + lmbda
        err = theta_star - theta_star_k
        return jnp.any(jnp.abs(err) > tol) & (k < maxiter)

    # Newton method for root finding
    def body_fun(nodes_k_lmbda):
        nodes, k, lmbda = nodes_k_lmbda
        A1 = eq.L_basis.evaluate(nodes, (0, 1, 0))
        lmbda_t = jnp.dot(A1, L_lmn)
        f = theta_star - nodes[:, 1] - lmbda
        df = -1 - lmbda_t
        nodes = put(nodes, Index[:, 1], nodes[:, 1] - f / df)
        A0 = eq.L_basis.evaluate(nodes, (0, 0, 0))
        lmbda = jnp.dot(A0, L_lmn)
        k += 1
        return (nodes, k, lmbda)

    nodes, k, lmbda = jit(while_loop, static_argnums=(0, 1))(
        cond_fun, body_fun, (nodes, k, lmbda)
    )
    theta_star_k = nodes[:, 1] + lmbda
    err = theta_star - theta_star_k
    noconverge = jnp.abs(err) > tol
    nodes = jnp.where(noconverge[:, np.newaxis], jnp.nan, nodes)

    return nodes


def compute_flux_coords(
    eq, real_coords, R_lmn=None, Z_lmn=None, tol=1e-6, maxiter=20, rhomin=1e-6
):
    """Find the (rho, theta, zeta) that correspond to given (R, phi, Z).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use
    real_coords : ndarray, shape(k,3)
        2D array of real space coordinates [R,phi,Z]. Each row is a different
        point in space.
    R_lmn, Z_lmn : ndarray
        spectral coefficients for R and Z. Defaults to eq.R_lmn, eq.Z_lmn
    tol : float
        Stopping tolerance. Iterations stop when sqrt((R-Ri)**2 + (Z-Zi)**2) < tol
    maxiter : int > 0
        maximum number of Newton iterations
    rhomin : float
        minimum allowable value of rho (to avoid singularity at rho=0)

    Returns
    -------
    flux_coords : ndarray, shape(k,3)
        flux coordinates [rho,theta,zeta]. If Newton method doesn't converge for
        a given coordinate (often because it is outside the plasma boundary),
        nan will be returned for those values

    """
    if maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got{maxiter}")
    if R_lmn is None:
        R_lmn = eq.R_lmn
    if Z_lmn is None:
        Z_lmn = eq.Z_lmn

    R, phi, Z = real_coords.T
    R = jnp.abs(R)

    # nearest neighbor search on coarse grid for initial guess
    nodes = ConcentricGrid(L=20, M=10, N=0).nodes
    AR = eq.R_basis.evaluate(nodes)
    AZ = eq.Z_basis.evaluate(nodes)
    Rg = jnp.dot(AR, R_lmn)
    Zg = jnp.dot(AZ, Z_lmn)
    distance = (R[:, np.newaxis] - Rg) ** 2 + (Z[:, np.newaxis] - Zg) ** 2
    idx = jnp.argmin(distance, axis=1)

    rhok = nodes[idx, 0]
    thetak = nodes[idx, 1]
    Rk = Rg[idx]
    Zk = Zg[idx]
    k = 0

    def cond_fun(k_rhok_thetak_Rk_Zk):
        k, rhok, thetak, Rk, Zk = k_rhok_thetak_Rk_Zk
        return jnp.any(((R - Rk) ** 2 + (Z - Zk) ** 2) > tol**2) & (k < maxiter)

    def body_fun(k_rhok_thetak_Rk_Zk):
        k, rhok, thetak, Rk, Zk = k_rhok_thetak_Rk_Zk
        nodes = jnp.array([rhok, thetak, phi]).T
        ARr = eq.R_basis.evaluate(nodes, (1, 0, 0))
        Rr = jnp.dot(ARr, R_lmn)
        AZr = eq.Z_basis.evaluate(nodes, (1, 0, 0))
        Zr = jnp.dot(AZr, Z_lmn)
        ARt = eq.R_basis.evaluate(nodes, (0, 1, 0))
        Rt = jnp.dot(ARt, R_lmn)
        AZt = eq.Z_basis.evaluate(nodes, (0, 1, 0))
        Zt = jnp.dot(AZt, Z_lmn)

        tau = Rt * Zr - Rr * Zt
        eR = R - Rk
        eZ = Z - Zk
        thetak += (Zr * eR - Rr * eZ) / tau
        rhok += (Rt * eZ - Zt * eR) / tau
        # negative rho -> rotate theta instead
        thetak = jnp.where(
            rhok < 0, (thetak + np.pi) % (2 * np.pi), thetak % (2 * np.pi)
        )
        rhok = jnp.abs(rhok)
        rhok = jnp.clip(rhok, rhomin, 1)
        nodes = jnp.array([rhok, thetak, phi]).T

        AR = eq.R_basis.evaluate(nodes, (0, 0, 0))
        Rk = jnp.dot(AR, R_lmn)
        AZ = eq.Z_basis.evaluate(nodes, (0, 0, 0))
        Zk = jnp.dot(AZ, Z_lmn)
        k += 1
        return (k, rhok, thetak, Rk, Zk)

    k, rhok, thetak, Rk, Zk = while_loop(cond_fun, body_fun, (k, rhok, thetak, Rk, Zk))

    noconverge = (R - Rk) ** 2 + (Z - Zk) ** 2 > tol**2
    rho = jnp.where(noconverge, jnp.nan, rhok)
    theta = jnp.where(noconverge, jnp.nan, thetak)
    phi = jnp.where(noconverge, jnp.nan, phi)

    return jnp.vstack([rho, theta, phi]).T


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
        params={
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
            "L_lmn": L_lmn,
        },
        transforms=transforms,
        profiles={},  # no profiles needed
    )

    nested = jnp.all(
        jnp.sign(data["sqrt(g)_PEST"][0]) == jnp.sign(data["sqrt(g)_PEST"])
    )
    if not nested:
        if msg == "auto":
            warnings.warn(
                colored(
                    "WARNING: Flux surfaces are no longer nested, exiting early. "
                    + "Automatic continuation method failed, consider specifying "
                    + "continuation steps manually",
                    "yellow",
                )
            )
        elif msg == "manual":
            warnings.warn(
                colored(
                    "WARNING: Flux surfaces are no longer nested, exiting early."
                    + "Consider taking smaller perturbation/resolution steps "
                    + "or reducing trust radius",
                    "yellow",
                )
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

    grid = ConcentricGrid(L_grid, M_grid, N_grid, node_pattern="ocs")
    bdry_grid = LinearGrid(M=M, N=N, rho=1.0)

    toroidal_coords = eq.compute(["R", "Z", "lambda"], grid=grid)
    theta = grid.nodes[:, 1]
    vartheta = theta + toroidal_coords["lambda"]
    sfl_grid = grid
    sfl_grid.nodes[:, 1] = vartheta

    bdry_coords = eq.compute(["R", "Z", "lambda"], grid=bdry_grid)
    bdry_theta = bdry_grid.nodes[:, 1]
    bdry_vartheta = bdry_theta + bdry_coords["lambda"]
    bdry_sfl_grid = bdry_grid
    bdry_sfl_grid.nodes[:, 1] = bdry_vartheta

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
