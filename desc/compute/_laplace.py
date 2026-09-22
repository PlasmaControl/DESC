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

    The integral runs over the N_source quadrature points (``source_data``,
    ``interpolator.source_grid``) and is evaluated at the N_eval points
    (``eval_data``, ``interpolator.eval_grid``). The two need not match.

    Parameters
    ----------
    eval_data : dict
        Data at the N_eval evaluation points. With ``basis`` supplied,
        ``eval_data["Phi(x) (periodic)"]`` is the eval-grid Vandermonde,
        shape (N_eval, N_modes).
    source_data : dict
        Data at the N_source quadrature points. With ``basis`` supplied,
        ``source_data["Phi (periodic)"]`` is the source-grid Vandermonde,
        shape (N_source, N_modes).
    basis : DoubleFourierSeries
        If not supplied, then computes (D[Φ] + Φ/2)(x), shape (N_eval,).
        If supplied, then constructs the operator which
        acts on the spectral coefficients of Φ in the supplied + secular basis,
        shape (N_eval, N_modes).
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
        Data at the N_eval evaluation points (R, phi, Z and any eval_keys needed
        by the kernel).
    source_data : dict
        Data at the N_source quadrature points (must include |e_theta x e_zeta|
        and geometry). Does not need to include B0*n; the caller supplies it
        per-column. N_source need not equal N_eval.
    interpolator : _BIESTInterpolator
    chunk_size : int or None
        Chunk size for eval-point batching *inside* each ``singular_integral`` call.
    ndim : int or None
        Number of columns carried through the integral, i.e. N_modes when
        ``source_data["B0*n"]`` is a Vandermonde matrix.

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
    eval_data,
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

    Two grids, and they need not have the same size:

    * the SOURCE grid (``interpolator.source_grid``) holds the N_source
      quadrature points that discretize the singular integrals;
    * the EVAL grid (``interpolator.eval_grid``) holds the N_eval points where
      those integrals are evaluated, i.e. where B_n is supplied and Phi is
      wanted.

    Refining only the source grid resolves the singular integral -- which is
    what makes the discrete operator self-adjoint in the surface measure --
    without changing the size of the returned matrix or the number of modes.

    Parameters
    ----------
    eval_data : dict
        Data at the N_eval evaluation points (geometry, not B0*n).
    source_data : dict
        Data at the N_source quadrature points (geometry including
        |e_theta x e_zeta|).
    interpolator : _BIESTInterpolator
        Carries both grids; nothing in it requires N_source == N_eval.
    phi_transform : Transform
        Built on the SOURCE grid (see ``get_transforms``: "Phi_PEST" uses
        ``kwargs["pest_grid"]``), so its Vandermonde and pseudoinverse are
        source-grid quantities. The eval-grid counterparts are built here.
    problem : str
        One of {"interior Neumann", "exterior Neumann", "interior Dirichlet"}.
    chunk_size : int or None
        Inner chunk size for eval-point batching inside each ``singular_integral`` call.
    _midpoint_quad : bool
    _D_quad : bool

    Returns
    -------
    A_mn : jnp.ndarray, shape (N_modes, N_eval)
        Nodal B_n on the eval grid -> spectral coefficients of Phi (periodic).
    A : jnp.ndarray, shape (N_eval, N_eval)
        Matrix satisfying Phi (periodic) = A @ B_n, both nodal on the eval grid.
    """
    assert problem in {"interior Neumann", "exterior Neumann", "interior Dirichlet"}

    eval_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    basis = phi_transform.basis
    if pest_coords:
        assert (
            source_grid.can_fft2
        ), f"pest_grid must have can_fft2=True, got {source_grid}"
        assert (
            eval_grid.can_fft2
        ), f"potential pest_grid must have can_fft2=True, got {eval_grid}"
    # The mode count is bounded by the EVAL grid, never by the source grid, so
    # refining the quadrature alone never changes the number of unknowns.
    assert basis.M <= eval_grid.M
    assert basis.N <= eval_grid.N

    same_grid = eval_grid == source_grid

    # Vandermonde matrices. `phi_transform.matrices["direct1"][0][0][0]` is just
    # basis.evaluate(source_grid), and its "pinv" the matching pseudoinverse.
    # Reused verbatim when the grids coincide so that case stays bit-identical.
    Phi_src = phi_transform.matrices["direct1"][0][0][0]  # (N_source, N_modes)
    Phi_eval = Phi_src if same_grid else basis.evaluate(eval_grid)  # (N_eval, N_modes)
    # B_n is supplied nodally ON THE EVAL GRID, so the fit to spectral
    # coefficients has to be the eval-grid pseudoinverse.
    pinv_eval = (  # (N_modes, N_eval)
        phi_transform.matrices["pinv"] if same_grid else jnp.linalg.pinv(Phi_eval)
    )

    # Double-layer operator D, shape (N_eval, N_modes). Prune into a separate
    # copy so the original dicts are still available for M_S below.
    eval_data_d, source_data_d = _prune_data(
        eval_data,
        eval_grid,
        source_data,
        source_grid,
        _kernel_dipole_plus_half,
    )
    eval_data_d["Phi(x) (periodic)"] = Phi_eval  # (N_eval, N_modes)
    source_data_d["Phi (periodic)"] = Phi_src  # (N_source, N_modes)

    # Expand B0*n in the same basis as Phi. This lives on the SOURCE grid: it is
    # what the quadrature integrates against.
    source_data["B0*n"] = Phi_src  # (N_source, N_modes)

    D = _D_plus_half(
        eval_data_d,
        source_data_d,
        interpolator,
        basis,
        chunk_size,
        prune_data=False,
        _midpoint_quad=_midpoint_quad,
        _D_quad=_D_quad,
    )
    assert D.shape == (eval_grid.num_nodes, basis.num_modes)
    if problem == "exterior Neumann" or problem == "interior Dirichlet":
        D -= Phi_eval

    # Single-layer matrix M_S, shape (N_eval, N_modes). Uses the original
    # (unpruned) data so that |e_theta x e_zeta| is available.
    M_S = _compute_single_layer_matrix(
        eval_data, source_data, interpolator, chunk_size, ndim=basis.num_modes
    )
    # Solve D @ A_mn = M_S for all N_modes right-hand sides simultaneously.
    # A_mn is (N_modes, N_modes) here; @ pinv_eval makes it (N_modes, N_eval).
    if eval_grid.num_nodes == basis.num_modes:
        A_mn = jnp.linalg.solve(D, M_S)
    else:
        A_mn = jnp.linalg.lstsq(D, M_S)[0]
    A_mn = A_mn @ pinv_eval  # (N_modes, N_eval)

    # Phi (periodic) = Phi_eval @ A_mn @ B_n, shape (N_eval, N_eval).
    # Sign convention makes B dot n the outward normal.
    return A_mn, -Phi_eval @ A_mn


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
    st="int : Support size of the local singular grid in theta. If not given "
    "(along with sz and q), a heuristic based on the source geometry chooses "
    "all three.",
    sz="int : Support size of the local singular grid in zeta. See st.",
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
    st="int : Support size of the local singular grid in theta. If not given "
    "(along with sz and q), a heuristic based on the source geometry chooses "
    "all three.",
    sz="int : Support size of the local singular grid in zeta. See st.",
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
    Phi_basis="DoubleFourierSeries, optional: override the equilibrium's own "
    "Phi_basis (eq.surface.Phi_basis) for this compute. Useful when the grid "
    "being evaluated on cannot resolve the equilibrium's own (fixed, "
    "file-level) Phi_basis resolution -- see FinitenStability._phi_matrix.",
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
    Phi_basis="DoubleFourierSeries, optional: override the equilibrium's own "
    "Phi_basis (eq.surface.Phi_basis) for this compute. Useful when the grid "
    "being evaluated on cannot resolve the equilibrium's own (fixed, "
    "file-level) Phi_basis resolution -- see FinitenStability._phi_matrix.",
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
