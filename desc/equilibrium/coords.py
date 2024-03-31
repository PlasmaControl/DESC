"""Functions for mapping between flux, sfl, and real space coordinates."""

import functools
import warnings

import numpy as np
from termcolor import colored

from desc.backend import fori_loop, jit, jnp, put, root, root_scalar, vmap
from desc.compute import compute as compute_fun
from desc.compute import data_index, get_profiles, get_transforms
from desc.grid import (
    ConcentricGrid,
    Grid,
    LinearGrid,
    QuadratureGrid,
    meshgrid_expand,
    meshgrid_inverse_idx,
    meshgrid_unique_idx,
)
from desc.transform import Transform
from desc.utils import setdefault


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
    """Given coordinates in inbasis, compute corresponding coordinates in outbasis.

    First solves for the computational coordinates that correspond to inbasis, then
    evaluates outbasis at those locations.

    Speed can often be significantly improved by providing a reasonable initial guess.
    The default is a nearest neighbor search on a coarse grid.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to use
    coords : ndarray, shape(k,3)
        2D array of input coordinates. Each row is a different point in space.
    inbasis, outbasis : tuple of str
        Labels for input and output coordinates, eg ("R", "phi", "Z") or
        ("rho", "alpha", "zeta") or any combination thereof. Labels should be the
        same as the compute function data key
    guess : None or ndarray, shape(k,3)
        Initial guess for the computational coordinates ['rho', 'theta', 'zeta']
        corresponding to coords in inbasis. If None, heuristics are used based on
        in basis and a nearest neighbor search on a coarse grid.
    params : dict
        Values of equilibrium parameters to use, eg eq.params_dict
    period : tuple of float
        Assumed periodicity for each quantity in inbasis.
        Use np.inf to denote no periodicity.
    tol : float
        Stopping tolerance.
    maxiter : int > 0
        Maximum number of Newton iterations
    full_output : bool, optional
        If True, also return a tuple where the first element is the residual from
        the root finding and the second is the number of iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to ``root`` such as ``maxiter_ls``,
        ``alpha``.

    Returns
    -------
    coords : ndarray, shape(k,3)
        Coordinates mapped from inbasis to outbasis. Values of NaN will be returned
        for coordinates where root finding did not succeed, possibly because the
        coordinate is not in the plasma volume.
    info : tuple
        2 element tuple containing residuals and number of iterations
        for each point. Only returned if ``full_output`` is True

    Notes
    -----
    ``guess`` must be given for this function to be compatible with ``jit``.

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
    kwargs.setdefault("tol", tol)
    kwargs.setdefault("maxiter", maxiter)
    period = np.asarray(period)
    coords = coords % period

    params = setdefault(params, eq.params_dict)

    profiles = get_profiles(inbasis + basis_derivs, eq, None)
    # do surface average to get iota once
    if "iota" in profiles and profiles["iota"] is None:
        profiles["iota"] = eq.get_profile("iota")

    @functools.partial(jit, static_argnums=1)
    def compute(y, basis):
        grid = Grid(y, sort=False, jitable=True)
        transforms = get_transforms(basis, eq, grid, jitable=True)
        data = compute_fun(eq, basis, params, transforms, profiles)
        x = jnp.array([data[k] for k in basis]).T
        return x

    @jit
    def residual(y, coords):
        xk = compute(y, inbasis)
        r = xk % period - coords % period
        return jnp.where(r > period / 2, -period + r, r)

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

    vecroot = jit(
        vmap(lambda x0, *p: root(residual, x0, jac=jac, args=p, fixup=fixup, **kwargs))
    )
    yk, (res, niter) = vecroot(yk, coords)

    out = compute(yk, outbasis)

    if full_output:
        return out, (res, niter)
    return out


def compute_theta_coords(
    eq, flux_coords, L_lmn=None, tol=1e-6, maxiter=20, full_output=False, **kwargs
):
    """Find theta_DESC for given straight field line theta_PEST.

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
    full_output : bool, optional
        If True, also return a tuple where the first element is the residual from
        the root finding and the second is the number of iterations.
    kwargs : dict, optional
        Additional keyword arguments to pass to ``root_scalar`` such as ``maxiter_ls``,
        ``alpha``.

    Returns
    -------
    coords : ndarray, shape(k,3)
        coordinates [rho,theta,zeta].
    info : tuple
        2 element tuple containing residuals and number of iterations
        for each point. Only returned if ``full_output`` is True
    """
    kwargs.setdefault("maxiter", maxiter)
    kwargs.setdefault("tol", tol)

    if L_lmn is None:
        L_lmn = eq.L_lmn
    rho, theta_star, zeta = flux_coords.T

    def rootfun(theta_DESC, theta_PEST, rho, zeta):
        nodes = jnp.atleast_2d(
            jnp.array([rho.squeeze(), theta_DESC.squeeze(), zeta.squeeze()])
        )
        A = eq.L_basis.evaluate(nodes)
        lmbda = A @ L_lmn
        theta_PESTk = theta_DESC + lmbda
        r = (theta_PESTk % (2 * np.pi)) - (theta_PEST % (2 * np.pi))
        # r should be between -pi and pi
        r = jnp.where(r > np.pi, r - 2 * np.pi, r)
        r = jnp.where(r < -np.pi, r + 2 * np.pi, r)
        return r.squeeze()

    def jacfun(theta_DESC, theta_PEST, rho, zeta):
        nodes = jnp.atleast_2d(
            jnp.array([rho.squeeze(), theta_DESC.squeeze(), zeta.squeeze()])
        )
        A1 = eq.L_basis.evaluate(nodes, (0, 1, 0))
        lmbda_t = jnp.dot(A1, L_lmn)
        return 1 + lmbda_t.squeeze()

    def fixup(x, *args):
        return x % (2 * np.pi)

    vecroot = jit(
        vmap(
            lambda x0, *p: root_scalar(
                rootfun, x0, jac=jacfun, args=p, fixup=fixup, **kwargs
            )
        )
    )
    theta_DESC, (res, niter) = vecroot(theta_star, theta_star, rho, zeta)

    nodes = jnp.array([rho, theta_DESC.squeeze(), zeta]).T

    out = nodes
    if full_output:
        return out, (res, niter)
    return out


def desc_grid_from_field_line_coords(eq, rho, alpha, zeta):
    """Return DESC coordinate grid from given Clebsch-Type field-line coordinates.

    Create a meshgrid from the given field line coordinates,
    and transform this to a meshgrid in DESC coordinates.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium on which to perform coordinate mapping.
    rho : ndarray
        Unique flux surface label coordinates.
    alpha : ndarray
        Unique field line label coordinates over a constant rho surface.
    zeta : ndarray
        Unique Field line-following ζ coordinates.

    Returns
    -------
    grid_desc : Grid
        DESC coordinate grid.
    data_desc : dict
        Some flux surface quantities that may be more accurate than what
        can be computed on the returned grid.

    """
    # The rotational transform can be computed apriori to the coordinate
    # transformation because it is a single variable function of the flux surface
    # label rho, and the coordinate mapping does not change rho. Once it is known,
    # we can compute the straight field-line poloidal angle theta_PEST from the
    # field-line label alpha.
    # Then we transform from straight field-line coordinates to DESC coordinates
    # with the root-finding method ``compute_theta_coords``.
    # This is preferable to transforming from field-line coordinates to DESC
    # coordinates directly with the more general root-finding method
    # ``map_coordinates``. That method requires an initial guess, and generating
    # a reasonable initial guess requires computing the rotational transform to
    # approximate theta_PEST and the poloidal stream function anyway.

    # Choose nodes such that even spacing will yield correct flux surface integrals.
    t = jnp.linspace(0, 2 * jnp.pi, 2 * eq.M_grid + 1, endpoint=False)
    z = jnp.linspace(0, 2 * jnp.pi / eq.NFP, 2 * eq.N_grid + 1, endpoint=False)
    label = ["rho", "theta", "zeta"]
    unique_idx = {
        f"_unique_{label[i]}_idx": idx
        for i, idx in enumerate(meshgrid_unique_idx(rho.size, t.size, z.size))
    }
    inverse_idx = {
        f"_inverse_{label[i]}_idx": idx
        for i, idx in enumerate(meshgrid_inverse_idx(rho.size, t.size, z.size))
    }
    grid = Grid(
        nodes=jnp.column_stack(
            tuple(map(jnp.ravel, jnp.meshgrid(rho, t, z, indexing="ij")))
        ),
        spacing=jnp.array([1 / rho.size, 2 * jnp.pi / t.size, 2 * jnp.pi / z.size])[
            jnp.newaxis
        ],
        sort=False,
        jitable=True,
        **unique_idx,
        **inverse_idx,
    )
    # We only need to compute the rotational transform to transform to straight
    # field-line coordinates. However, it is a good idea to compute other flux
    # surface quantities on this grid because the DESC coordinates corresponding
    # to the given field line coordinates may not be uniformly distributed over
    # flux surfaces. This would make quadratures performed over flux surfaces
    # on the returned DESC grid inaccurate.
    data = eq.compute(names=["iota", "iota_r"], grid=grid)
    data_desc = {
        d: meshgrid_expand(grid.compress(data[d]), rho.size, alpha.size, zeta.size)
        for d in data
        if data_index["desc.equilibrium.equilibrium.Equilibrium"][d]["coordinates"]
        == "r"
    }
    # meshgrid of Clebsch-Type field-line coordinates
    r, a, z_fl = map(jnp.ravel, jnp.meshgrid(rho, alpha, zeta, indexing="ij"))
    coords_sfl = jnp.column_stack([r, a + data_desc["iota"] * z_fl, z_fl])
    coords_desc = eq.compute_theta_coords(coords_sfl)
    unique_idx = {
        f"_unique_{label[i]}_idx": idx
        for i, idx in enumerate(meshgrid_unique_idx(rho.size, alpha.size, zeta.size))
    }
    inverse_idx = {
        f"_inverse_{label[i]}_idx": idx
        for i, idx in enumerate(meshgrid_inverse_idx(rho.size, alpha.size, zeta.size))
    }
    grid_desc = Grid(
        nodes=coords_desc, sort=False, jitable=True, **unique_idx, **inverse_idx
    )
    return grid_desc, data_desc


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
