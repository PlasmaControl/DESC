"""Compute functions for multiply connected geometry Laplace solver.

References
----------
    [1] Unalmis et al. New high-order accurate free surface stellarator
        equilibria optimization and boundary integral methods in DESC.

"""

from interpax_fft import rfft_interp2d

from desc.backend import jnp
from desc.integrals.singularities import (
    _kernel_dipole,
    _kernel_dipole_plus_half,
    _kernel_monopole,
    _nonsingular_part,
    _prune_data,
    get_interpolator,
    singular_integral,
)
from desc.utils import apply

from .data_index import register_compute_fun

_doc = {
    "Phi_0": """jnp.ndarray :
        Initial guess for iteration.
        """,
    "xtol": """float :
        Stopping tolerance for fixed point method. Default is ``1e-7``.
        """,
    "maxiter": """int :
        Maximum number of iterations for fixed point method.
        If non-positive then the linear operator will be inverted instead.
        If positive, then performs that many fixed point iterations until ``maxiter``
        or an error tolerance of ``xtol`` is reached. For reference, ``20`` yields an
        error of ``1e-5`` as illustrated in [1]. Default is ``25``.
        """,
    "full_output": """bool
        Whether to compute the maximum error ``Phi error`` and store the number of
        iterations ``num iter`` used for the fixed point method. Default is ``False``.
        """,
    "chunk_size": """int or None :
        Size to split integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.  Default is ``None``.
        Recommend to verify computation with ``chunk_size`` set to a
        small number due to bugs in JAX or XLA.
        """,
    "_midpoint_quad": """bool :
        Set to ``True`` to perform double layer potential quadrature with a midpoint
        rule. Default is ``False``. This is intended for developer use.
        """,
    "_D_quad": """bool
        Set to ``True`` to perform double layer potential quadrature without removing
        singularities. Default is ``False``. This is intended for developer use.
        """,
}


def _D_plus_half(
    eval_data,
    source_data,
    interpolator,
    basis=None,
    chunk_size=None,
    prune_data=True,
    pest_coords=False,
    _midpoint_quad=False,
    _D_quad=False,
):
    """Compute (D[Φ] + Φ/2)(x).

    D[Φ](x) = ∫_y Φ(y)〈∇_x G(x−y),ds(y)〉.

    Parameters
    ----------
    basis : DoubleFourierSeries
        If not supplied, then computes (D[Φ] + Φ/2)(x).
        If supplied, then constructs the operator which
        acts on the spectral coefficients of Φ in the supplied + secular basis.
    prune_data : bool
        Whether the data should be pruned. Default is True.
    _midpoint_quad : bool
        Set to ``True`` to perform double layer potential quadrature with a midpoint
        rule. Default is ``False``. This is intended for developer use.
    _D_quad : bool
        Set to ``True`` to perform double layer potential quadrature without removing
        singularities. Default is ``False``. This is intended for developer use.
    pest_coords : bool
        Set to True to do the entire computation in PEST coordinates. Interpolator
        grid must be in PEST coordinates, and fft2able.
    """
    if basis is None:
        ndim = 1
        known_map = None
    else:
        ndim = basis.num_modes
        known_map = ("Phi (periodic)", basis.evaluate)

    kernel = _kernel_dipole if _D_quad else _kernel_dipole_plus_half

    if _midpoint_quad:
        if prune_data:
            eval_data, source_data = prune_data(
                eval_data,
                interpolator.eval_grid,
                source_data,
                interpolator.source_grid,
                _kernel_dipole_plus_half,
            )
        result = _nonsingular_part(
            eval_data,
            None,
            source_data,
            interpolator.source_grid,
            st=jnp.nan,
            sz=jnp.nan,
            kernel=kernel,
            ndim=ndim,
            chunk_size=chunk_size,
        )
    else:
        result = singular_integral(
            eval_data,
            source_data,
            interpolator,
            kernel,
            known_map=known_map,
            ndim=ndim,
            chunk_size=chunk_size,
            _prune_data=prune_data,
        )
    if ndim == 1:
        result = result.squeeze(-1)

    if _D_quad:
        result += eval_data["Phi(x) (periodic)"] / 2

    return result


def _compute_single_layer_matrix(
    eval_data,
    source_data,
    interpolator,
    chunk_size=None,
    ndim=None,
):
    """Compute the single-layer operator as a matrix M_S where S[B0*n] = M_S @ B_n.

    Parameters
    ----------
    eval_data : dict
        Data at evaluation points (R, phi, Z and any eval_keys needed by the kernel).
    source_data : dict
        Data at source points (must include |e_theta x e_zeta| and geometry).
        Does not need to include B0*n.
        In _lsmr_compute_phi_matrix, eval_data and source_data are hard-coded
        to be the same.
    interpolator : _BIESTInterpolator
    chunk_size : int or None
        Chunk size for eval-point batching *inside* each ``singular_integral`` call.

    Returns
    -------
    M_S : jnp.ndarray, shape (N_eval, N_modes)
        Matrix satisfying S[B0*n] = M_S @ B_n.
    """
    source_grid = interpolator.source_grid

    eval_data_p, source_data_p = _prune_data(
        eval_data, interpolator.eval_grid, source_data, source_grid, _kernel_monopole
    )
    # B0*n is not in source_data_p (no specific B_n to prune); supply it per-column.
    spectral_matrix = singular_integral(
        eval_data_p,
        source_data_p,
        interpolator,
        _kernel_monopole,
        chunk_size=chunk_size,
        _prune_data=False,
        ndim=ndim,
    )
    return spectral_matrix


def _lsmr_compute_phi_matrix(
    potential_data,
    source_data,
    interpolator,
    phi_transform,
    problem,
    chunk_size=None,
    pest_coords=False,
    _midpoint_quad=False,
    _D_quad=False,
):
    """Compute matrix A where Phi (periodic) = A @ B_n.

    Constructs the double-layer operator D and single-layer matrix M_S, then
    solves D @ A_mn = M_S in one batch, returning E @ A_mn where E is the
    Vandermonde matrix (so the output maps directly to nodal potential values).

    Parameters
    ----------
    potential_data : dict
        Data at potential grid points (geometry, not B0*n).
    source_data : dict
        Data at source grid points (geometry including |e_theta x e_zeta|).
    interpolator : _BIESTInterpolator
    basis : DoubleFourierSeries
        Spectral basis for the periodic part of the potential.
    problem : str
        One of {"interior Neumann", "exterior Neumann", "interior Dirichlet"}.
    chunk_size : int or None
        Inner chunk size for eval-point batching inside each ``singular_integral`` call.
    _midpoint_quad : bool
    _D_quad : bool

    Returns
    -------
    A : jnp.ndarray, shape (N_potential, N_source)
        Matrix satisfying Phi (periodic) = A @ B_n.
    """
    assert problem in {"interior Neumann", "exterior Neumann", "interior Dirichlet"}

    # hard-code that Phi & Bn are on same grid (necessary for ext. mode stability)
    potential_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    basis = phi_transform.basis
    if pest_coords:
        assert (
            source_grid.can_fft2
        ), f"pest_grid must have can_fft2=True, got {source_grid}"
        assert (
            potential_grid.can_fft2
        ), f"potential pest_grid must have can_fft2=True, got {potential_grid}"
    assert basis.M <= potential_grid.M
    assert basis.N <= potential_grid.N

    # Build double-layer operator D: shape (N_potential, N_modes).
    # Prune into a separate copy so original dicts are available for M_S below.

    # phi_transform.matrices["direct1"][0][0][0] is just basis.evaluate(grid)
    Phi = phi_transform.matrices["direct1"][0][0][0]
    potential_data_d, source_data_d = _prune_data(
        potential_data,
        potential_grid,
        source_data,
        source_grid,
        _kernel_dipole_plus_half,
    )

    potential_data_d["Phi(x) (periodic)"] = Phi
    source_data_d["Phi (periodic)"] = (
        Phi if (potential_grid == source_grid) else basis.evaluate(source_grid)
    )

    pinv = phi_transform.matrices["pinv"]

    source_data["B0*n"] = Phi  # use same basis for B0*n as for phi

    print("source data computed")

    D = _D_plus_half(
        potential_data_d,
        source_data_d,
        interpolator,
        basis,
        chunk_size,
        prune_data=False,
        _midpoint_quad=_midpoint_quad,
        _D_quad=_D_quad,
    )
    print("D + half computed")
    assert D.shape == (potential_grid.num_nodes, basis.num_modes)
    if problem == "exterior Neumann" or problem == "interior Dirichlet":
        D -= Phi

    # Build single-layer matrix M_S: shape (N_potential, N_modes).
    # Uses the original (unpruned) data so that |e_theta x e_zeta| is available.
    M_S = _compute_single_layer_matrix(
        potential_data, source_data, interpolator, chunk_size, ndim=basis.num_modes
    )
    print("single layer matrix computed")
    # Solve D @ A_mn = M_S for all N_source right-hand sides simultaneously.
    # A_mn has shape (N_modes, N_source).
    if potential_grid.num_nodes == basis.num_modes:
        A_mn = jnp.linalg.solve(D, M_S)
        print(A_mn.shape)
        print(pinv.shape)
        print(D.shape)
        print(M_S.shape)
        A_mn = A_mn @ pinv
    else:
        A_mn = jnp.linalg.lstsq(D, M_S)[0]
        print(A_mn.shape)
        print(pinv.shape)
        print(D.shape)
        print(M_S.shape)
        A_mn = A_mn @ pinv
    print("linear system solved")
    # Phi (periodic) = Phi_E @ A_mn @ B_n, shape (N_potential, N_source).
    return A_mn, -Phi @ A_mn  # sign convention that makes B dot n the outward normal


@register_compute_fun(
    name="interpolator",
    label="",
    units="",
    units_long="",
    description="Interpolator for singular integrals.",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=["|e_theta x e_zeta|", "e_theta", "e_zeta"],
    parameterization=[
        "desc.geometry.surface.FourierRZToroidalSurface",
        "desc.equilibrium.equilibrium.Equilibrium",
    ],
    q="int : Order of quadrature in polar domain.",
    potential_grid="""LinearGrid :
        Grid to evaluate potential on boundary.
        If not given, default is to interpolate to source grid.
        """,
    warn_fft="""bool :
        Whether to warn if the interpolation will be lossy. Default is ``True``.
        """,
)
def _interpolator(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    grid = transforms["grid"]
    potential_grid = kwargs.get("potential_grid", grid)
    data["interpolator"] = get_interpolator(potential_grid, grid, data, **kwargs)

    if potential_grid == grid:
        data["potential data"] = apply(data, subset=("R", "phi", "Z"))
    else:
        dt = 2 * jnp.pi / grid.num_theta
        dz = 2 * jnp.pi / grid.num_zeta / grid.NFP

        # TODO: just interpolate Rb_mn, Zb_mn, and omegab_mn onto potential grid
        #       to avoid interpolation on oversampled grid
        def fun(x):
            return rfft_interp2d(
                grid.meshgrid_reshape(x, "rtz")[0],
                potential_grid.num_theta,
                potential_grid.num_zeta,
                dx=dt,
                dy=dz,
            ).ravel(order="F")

        data["potential data"] = apply(data, fun, ("R", "omega", "Z"))
        zeta = potential_grid.nodes[:, 2]
        data["potential data"]["phi"] = zeta + data["potential data"]["omega"]

    return data


@register_compute_fun(
    name="interpolator_pest",
    label="",
    units="",
    units_long="",
    description="Interpolator for singular integrals in PEST coordinates.",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=["|e_theta_PEST x e_phi|r,v|", "e_theta_PEST", "e_phi|r,v"],
    parameterization="desc.equilibrium.equilibrium.Equilibrium",
    q="int : Order of quadrature in polar domain.",
    pest_grid="""Grid :
        Grid in PEST (rvp) coordinates with ``can_fft2=True`` to use as the
        source grid for the singular integral interpolator.
        Must have the same number of poloidal and toroidal nodes as the
        main equilibrium grid, with nodes at the same physical surface points.
        """,
    potential_grid="""LinearGrid :
        Grid to evaluate potential on boundary.
        If not given, defaults to ``pest_grid``.
        """,
    warn_fft="""bool :
        Whether to warn if the interpolation will be lossy. Default is ``True``.
        """,
)
def _interpolator_pest(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    pest_grid = kwargs["pest_grid"]
    potential_grid = kwargs.get("potential_grid", pest_grid)
    # Relabel PEST basis vectors to standard names expected by get_interpolator
    # (_best_ratio uses e_theta, e_zeta, |e_theta x e_zeta|).
    source_data = dict(data)
    source_data["e_theta"] = data["e_theta_PEST"]
    source_data["e_zeta"] = data["e_phi|r,v"]
    source_data["|e_theta x e_zeta|"] = data["|e_theta_PEST x e_phi|r,v|"]
    data["interpolator_pest"] = get_interpolator(
        potential_grid, pest_grid, source_data, **kwargs
    )

    if potential_grid == pest_grid:
        data["potential data"] = apply(data, subset=("R", "phi", "Z"))
    else:
        dt = 2 * jnp.pi / pest_grid.num_theta
        dz = 2 * jnp.pi / pest_grid.num_zeta / pest_grid.NFP

        def fun(x):
            return rfft_interp2d(
                pest_grid.meshgrid_reshape(x, "rtz")[
                    0
                ],  # rtz_grid.meshgrid_reshape(x, "rtz")[0],
                potential_grid.num_theta,
                potential_grid.num_zeta,
                dx=dt,
                dy=dz,
            ).ravel(order="F")

        data["potential data"] = apply(data, fun, ("R", "omega", "Z"))
        zeta = potential_grid.nodes[:, 2]
        data["potential data"]["phi"] = zeta + data["potential data"]["omega"]

    return data


@register_compute_fun(
    name="potential data",
    label="potential data",
    units="~",
    units_long="not applicable",
    description="RpZ position on the potential grid",
    dim=1,
    coordinates="rtz",
    params=[],
    transforms={},
    profiles=[],
    data=["interpolator"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    public=False,
)
def _potential_grid_position(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    return data


@register_compute_fun(
    name="phi_matrix",
    label="A",
    units="T m^2",
    units_long="Tesla meter squared",
    description="Matrix A mapping B·n on the boundary to the periodic "
    "scalar potential, Phi (periodic) = A @ B_n. More efficient than "
    "solving per unit vector when the geometry is fixed and B_n varies.",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=list(
        (set(_kernel_dipole_plus_half.keys) - {"Phi (periodic)"})
        | (set(_kernel_monopole.keys) - {"B0*n"})
    )
    + ["interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    public=False,
    problem='str : Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    chunk_size=_doc["chunk_size"],
    _midpoint_quad=_doc["_midpoint_quad"],
    _D_quad=_doc["_D_quad"],
)
def _phi_matrix_compute(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    data["A_mn"], data["phi_matrix"] = _lsmr_compute_phi_matrix(
        data.get("potential data", data),
        data,
        data["interpolator"],
        transforms["Phi"],
        problem=kwargs["problem"],
        chunk_size=kwargs.get("chunk_size", None),
        _midpoint_quad=kwargs.get("_midpoint_quad", False),
        _D_quad=kwargs.get("_D_quad", False),
    )
    return data


@register_compute_fun(
    name="phi_matrix_pest",
    label="A_{PEST}",
    units="T m^2",
    units_long="Tesla meter squared",
    description="Matrix A mapping B·n on the boundary to the periodic scalar potential "
    "in PEST (straight field line) coordinates. "
    "Phi (periodic) = A @ B_n.",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]], "Phi_PEST": [[0, 0, 0]]},
    profiles=[],
    data=list(
        (set(_kernel_dipole_plus_half.keys) - {"Phi (periodic)"})
        | (set(_kernel_monopole.keys) - {"B0*n"})
        - {"e_theta x e_zeta", "|e_theta x e_zeta|"}
    )
    + [
        "e_theta_PEST x e_phi|r,v",
        "|e_theta_PEST x e_phi|r,v|",
        "interpolator_pest",
    ],
    resolution_requirement="tz",
    parameterization="desc.equilibrium.equilibrium.Equilibrium",
    public=False,
    problem='str : Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    pest_grid="""Grid :
        Grid in PEST (rvp) coordinates with ``can_fft2=True``.
        Passed through to ``interpolator_pest``.
        """,
    chunk_size=_doc["chunk_size"],
    _midpoint_quad=_doc["_midpoint_quad"],
    _D_quad=_doc["_D_quad"],
)
# MOVE PEST GRID TO TRANSFORMS
# THEN COMPUTE PHI ON THE PEST GRID
# THEN USE THAT INSTEAD IN THE KERNELS TO COMPUTE D AND M_S
def _phi_matrix_pest_compute(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    # Relabel PEST basis vectors to the standard key names expected by the
    # BIEST kernels (_kernel_monopole, _kernel_dipole_plus_half).
    # The same relabeling was applied in _interpolator_pest, so the data
    # passed here is consistent with what the interpolator was built with.
    data["e_theta x e_zeta"] = data["e_theta_PEST x e_phi|r,v"]
    data["|e_theta x e_zeta|"] = data["|e_theta_PEST x e_phi|r,v|"]
    print(transforms["Phi"].basis.modes)
    print(transforms["Phi_PEST"].basis.modes)

    data["A_mn"], data["phi_matrix_pest"] = _lsmr_compute_phi_matrix(
        data.get("potential data", data),
        data,
        data["interpolator_pest"],
        transforms["Phi_PEST"],
        problem=kwargs["problem"],
        chunk_size=kwargs.get("chunk_size", None),
        pest_coords=True,
        _midpoint_quad=kwargs.get("_midpoint_quad", False),
        _D_quad=kwargs.get("_D_quad", False),
    )
    return data


@register_compute_fun(
    name="A_mn",
    label="A_{mn}",
    units="T m^2",
    units_long="Tesla meter squared",
    description="Spectral matrix mapping B·n on the boundary to the periodic scalar "
    "potential coefficients. Phi (periodic) = Phi_E @ A_mn @ B_n, "
    "shape (N_modes, N_source).",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=["phi_matrix"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    public=False,
    problem='str : Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    chunk_size=_doc["chunk_size"],
    _midpoint_quad=_doc["_midpoint_quad"],
    _D_quad=_doc["_D_quad"],
)
def _A_mn_compute(params, transforms, profiles, data, **kwargs):
    return data  # noqa: unused dependency


@register_compute_fun(
    name="A_mn_pest",
    label="A_{mn}",
    units="T m^2",
    units_long="Tesla meter squared",
    description="Spectral matrix mapping B·n on the boundary to the periodic scalar "
    "potential coefficients in PEST coordinates. Phi (periodic) = Phi_E @ A_mn @ B_n, "
    "shape (N_modes, N_source).",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=["phi_matrix_pest"],
    parameterization="desc.equilibrium.equilibrium.Equilibrium",
    public=False,
    problem='str : Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    pest_grid="""Grid :
        Grid in PEST (rvp) coordinates with ``can_fft2=True``.
        Passed through to ``interpolator_pest``.
        """,
    chunk_size=_doc["chunk_size"],
    _midpoint_quad=_doc["_midpoint_quad"],
    _D_quad=_doc["_D_quad"],
)
def _A_mn_pest_compute(params, transforms, profiles, data, **kwargs):
    return data  # noqa: unused dependency
