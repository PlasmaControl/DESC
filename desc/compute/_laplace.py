"""Compute functions for multiply connected geometry Laplace solver.

References
----------
    [1] Unalmis et al. New high-order accurate free surface stellarator
        equilibria optimization and boundary integral methods in DESC.

"""

from functools import partial

import numpy as np

from desc.backend import fixed_point, jit, jnp
from desc.integrals.singularities import (
    _kernel_BS_plus_grad_S,
    _kernel_dipole,
    _kernel_dipole_plus_half,
    _kernel_monopole,
    _nonsingular_part,
    _prune_data,
    get_interpolator,
    singular_integral,
)
from desc.utils import cross, dot, errorif

from .data_index import register_compute_fun

_doc = {
    "Phi_0": """jnp.ndarray :
        Initial guess for iteration.
        Initial guess must be periodic to converge to true solution.
        """,
    "maxiter": """int :
        Maximum number of iterations for fixed point method.
        Default is zero, which means that matrix inversion will be used.
        If nonzero, then performs fixed point iterations until maximum
        iterations or error tolerance of ``1e-7`` is reached.
        """,
    "chunk_size": """int or None :
        Size to split integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.  Default is ``None``.
        Recommend to verify computation with ``chunk_size`` set to a
        small number due to bugs in JAX or XLA.
        """,
    "_midpoint_quad": """bool :
        Set to ``True`` to perform double layer potential quadrature
        with midpoint rule. Default is ``False``.
        """,
    "_D_quad": """bool
        Set to ``True`` to perform double layer potential quadrature
        without removing singularities. Default is ``False``.
        """,
    "_full_output": """bool
        Whether to return the maximum relative error and
        number of iterations for the fixed point method.
        """,
}


def _D_plus_half(
    eval_data,
    source_data,
    interpolator,
    basis=None,
    chunk_size=None,
    prune_data=True,
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
        Set to ``True`` to perform double layer potential quadrature
        with midpoint rule. Default is ``False``.
    _D_quad : bool
        Set to ``True`` to perform double layer potential quadrature
        without removing singularities. Default is ``False``.

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
        result = result.squeeze(axis=-1)

    if _D_quad:
        result += eval_data["Phi(x) (periodic)"] / 2

    return result


def _lsmr_compute_potential(
    boundary_condition,
    potential_data,
    source_data,
    interpolator,
    basis,
    problem,
    same_grid=True,
    chunk_size=None,
    _midpoint_quad=False,
    _D_quad=False,
    **kwargs,
):
    assert problem in {"interior Neumann", "exterior Neumann", "interior Dirichlet"}

    potential_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    assert basis.M <= potential_grid.M
    assert basis.N <= potential_grid.N

    potential_data, source_data = _prune_data(
        potential_data,
        potential_grid,
        source_data,
        source_grid,
        _kernel_dipole_plus_half,
    )
    Phi = basis.evaluate(potential_grid)
    potential_data["Phi(x) (periodic)"] = Phi
    source_data["Phi (periodic)"] = Phi if same_grid else basis.evaluate(source_grid)

    D = _D_plus_half(
        potential_data,
        source_data,
        interpolator,
        basis,
        chunk_size,
        prune_data=False,
        _midpoint_quad=_midpoint_quad,
        _D_quad=_D_quad,
    )
    assert D.shape == (potential_grid.num_nodes, basis.num_modes)
    if problem == "exterior Neumann" or problem == "interior Dirichlet":
        D -= Phi

    # TODO: https://github.com/patrick-kidger/lineax/pull/86
    return (
        jnp.linalg.solve(D, boundary_condition)
        if (potential_grid.num_nodes == basis.num_modes)
        else jnp.linalg.lstsq(D, boundary_condition)[0]
    )


def _iteration_operator(
    Phi,
    gamma,
    potential_data,
    source_data,
    interpolator,
    chunk_size,
    xi=2 / 3,
):
    """Equation 3.16 in [1]."""
    potential_data["Phi(x) (periodic)"] = Phi
    source_data["Phi (periodic)"] = Phi
    return (
        _D_plus_half(
            potential_data,
            source_data,
            interpolator,
            chunk_size=chunk_size,
            prune_data=False,
        )
        + (xi - 1) * Phi
        - gamma
    ) / xi


@partial(jit, static_argnames=["xtol", "maxiter", "chunk_size", "_full_output"])
def _fixed_point_potential(
    boundary_condition,
    potential_data,
    source_data,
    interpolator,
    Phi_0=None,
    *,
    xtol=1e-7,
    maxiter=20,
    chunk_size=None,
    _full_output=False,
):
    potential_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    potential_data, source_data = _prune_data(
        potential_data,
        potential_grid,
        source_data,
        source_grid,
        _kernel_dipole_plus_half,
    )
    if Phi_0 is None:
        Phi_0 = jnp.ones(potential_grid.num_nodes)
    return fixed_point(
        _iteration_operator,
        Phi_0,
        (boundary_condition, potential_data, source_data, interpolator, chunk_size),
        xtol,
        maxiter,
        method="simple",
        scalar=True,
        full_output=_full_output,
    )


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
    parameterization=["desc.geometry.surface.FourierRZToroidalSurface"],
    public=False,
)
def _interpolator(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    # Grids with resolution less than source grid yield poor convergence
    # due to FFT frequency spectrum truncation.
    grid = transforms["grid"]
    data["interpolator"] = get_interpolator(grid, grid, data)
    return data


@register_compute_fun(
    name="S[B0*n]",
    label="S[B_0 \\cdot n_{\\rho}]",
    units="T m",
    units_long="Tesla meter",
    description="Single layer potential of monopole density B0*n",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=_kernel_monopole.keys + ["interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    chunk_size=_doc["chunk_size"],
    public=False,
)
def _S_B0_n(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    data["S[B0*n]"] = singular_integral(
        data,
        data,
        data["interpolator"],
        _kernel_monopole,
        chunk_size=kwargs.get("chunk_size", None),
    ).squeeze(axis=-1)
    return data


@register_compute_fun(
    name="Phi_mn",
    label="\\Phi_{m n}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of periodic part of potential",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=list(set(_kernel_dipole_plus_half.keys) - {"Phi (periodic)"})
    + ["S[B0*n]", "interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    problem='str :Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    **_doc,
)
def _scalar_potential_mn_Neumann(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency

    if kwargs.get("maxiter", -1) > 0:
        errorif(
            "interior" in kwargs["problem"],
            msg="maxiter cannot be positive for interior Neumann problem.",
        )
        data["Phi (periodic)"] = _fixed_point_potential(
            data["S[B0*n]"],
            data,
            data,
            data["interpolator"],
            **{key: kwargs[key] for key in _doc if key in kwargs},
        )
        if kwargs.get("_full_output", False):
            data["Phi (periodic)"], (err, data["num iter"]) = data["Phi (periodic)"]
            data["Phi error"] = jnp.abs(err).max()

        transforms["Phi"].build_pinv()
        data["Phi_mn"] = transforms["Phi"].fit(data["Phi (periodic)"])
    else:
        data["Phi_mn"] = _lsmr_compute_potential(
            data["S[B0*n]"],
            data,
            data,
            data["interpolator"],
            transforms["Phi"].basis,
            **kwargs,
        )
    return data


@register_compute_fun(
    name="Phi (periodic)",
    label="\\Phi",
    units="T m",
    units_long="Tesla meter",
    description="Periodic part of magnetic scalar potential",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=["Phi_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _Phi_periodic_potential(params, transforms, profiles, data, **kwargs):
    assert data["Phi_mn"].size == transforms["Phi"].basis.num_modes
    data["Phi (periodic)"] = transforms["Phi"].transform(data["Phi_mn"])
    return data


@register_compute_fun(
    name="Phi_t (periodic)",
    label="\\partial_{\\theta} \\Phi_{\\text{periodic}}",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential, poloidal derivative",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 1, 0]]},
    profiles=[],
    data=["Phi_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _pot_Phi_t_periodic(params, transforms, profiles, data, **kwargs):
    assert data["Phi_mn"].size == transforms["Phi"].basis.num_modes
    data["Phi_t (periodic)"] = transforms["Phi"].transform(data["Phi_mn"], dt=1)
    return data


@register_compute_fun(
    name="Phi_z (periodic)",
    label="\\partial_{\\zeta} \\Phi_{\\text{periodic}}",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential, toroidal derivative",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 1]]},
    profiles=[],
    data=["Phi_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _pot_Phi_z_periodic(params, transforms, profiles, data, **kwargs):
    assert data["Phi_mn"].size == transforms["Phi"].basis.num_modes
    data["Phi_z (periodic)"] = transforms["Phi"].transform(data["Phi_mn"], dz=1)
    return data


@register_compute_fun(
    name="K_vc (periodic)",
    label="-n \\times \\nabla \\Phi_{\\text{periodic}}",
    units="T",
    units_long="Tesla",
    description="Virtual surface current due to potential",
    dim=3,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=[
        "n_rho x grad(theta)",
        "n_rho x grad(zeta)",
        "Phi_t (periodic)",
        "Phi_z (periodic)",
    ],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _virtual_surface_current_periodic(params, transforms, profiles, data, **kwargs):
    data["K_vc (periodic)"] = -(
        data["Phi_t (periodic)"][:, jnp.newaxis] * data["n_rho x grad(theta)"]
        + data["Phi_z (periodic)"][:, jnp.newaxis] * data["n_rho x grad(zeta)"]
    )
    return data


@register_compute_fun(
    name="Phi",
    label="\\Phi",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential",
    dim=1,
    coordinates="tz",
    params=["I", "Y"],
    transforms={},
    profiles=[],
    data=["Phi (periodic)", "theta", "zeta"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _Phi_scalar_potential(params, transforms, profiles, data, **kwargs):
    data["Phi"] = (
        data["Phi (periodic)"]
        + params["I"] * data["theta"]
        + params["Y"] * data["zeta"]
    )
    return data


@register_compute_fun(
    name="Phi_t",
    label="\\partial_{\\theta} \\Phi",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential, poloidal derivative",
    dim=1,
    coordinates="tz",
    params=["I"],
    transforms={},
    profiles=[],
    data=["Phi_t (periodic)"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _pot_Phi_t(params, transforms, profiles, data, **kwargs):
    data["Phi_t"] = data["Phi_t (periodic)"] + params["I"]
    return data


@register_compute_fun(
    name="Phi_z",
    label="\\partial_{\\zeta} \\Phi",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential, toroidal derivative",
    dim=1,
    coordinates="tz",
    params=["Y"],
    transforms={},
    profiles=[],
    data=["Phi_z (periodic)"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _pot_Phi_z(params, transforms, profiles, data, **kwargs):
    data["Phi_z"] = data["Phi_z (periodic)"] + params["Y"]
    return data


@register_compute_fun(
    name="K_vc",
    label="-n \\times \\nabla \\Phi",
    units="T",
    units_long="Tesla",
    description="Virtual surface current due to potential",
    dim=3,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["n_rho x grad(theta)", "n_rho x grad(zeta)", "Phi_t", "Phi_z"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _virtual_surface_current(params, transforms, profiles, data, **kwargs):
    data["K_vc"] = -(
        data["Phi_t"][:, jnp.newaxis] * data["n_rho x grad(theta)"]
        + data["Phi_z"][:, jnp.newaxis] * data["n_rho x grad(zeta)"]
    )
    return data


@register_compute_fun(
    name="|K_vc|^2",
    label="\\vert K_{\\text{vc}}) \\vert^2",
    units="T^2",
    units_long="Tesla squared",
    description="Squared norm of virtual surface current",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["K_vc"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _K_vc_squared(params, transforms, profiles, data, **kwargs):
    data["|K_vc|^2"] = dot(data["K_vc"], data["K_vc"])
    return data


@register_compute_fun(
    name="∇φ",
    label="\\nabla \\varphi",
    units="T",
    units_long="Tesla",
    description="Magnetic field due to potential which solves the"
    " boundary value problem.",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={},
    profiles=[],
    data=_kernel_BS_plus_grad_S.keys,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    chunk_size=_doc["chunk_size"],
    eval_interpolator="""_BIESTInterpolator :
        Interpolator from source grid to evaluation grid on boundary.
        If not given, default is to interpolate to source grid.
        """,
    problem='str :Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    on_boundary="bool : Whether RpZcoords are on boundary surface.",
)
def _grad_potential(params, transforms, profiles, data, RpZ_data, **kwargs):
    # noqa: unused dependency
    chunk_size = kwargs.get("chunk_size", None)
    interpolator = kwargs.get("eval_interpolator", data.get("interpolator", None))
    sign = 1 - 2 * int("exterior" in kwargs.get("problem", ""))

    if kwargs["on_boundary"]:
        RpZ_data["∇φ"] = (
            sign
            * 2
            * singular_integral(
                RpZ_data,
                data,
                interpolator,
                _kernel_BS_plus_grad_S,
                chunk_size=chunk_size,
            )
        )
    else:
        grid = transforms["grid"]
        eval_data, source_data = _prune_data(
            RpZ_data, None, data, grid, _kernel_BS_plus_grad_S
        )
        RpZ_data["∇φ"] = sign * _nonsingular_part(
            eval_data,
            None,
            source_data,
            grid,
            st=jnp.nan,
            sz=jnp.nan,
            kernel=_kernel_BS_plus_grad_S,
            chunk_size=chunk_size,
        )
    return RpZ_data


@register_compute_fun(
    name="B",
    label="B",
    units="T",
    units_long="Tesla",
    description="Magnetic field",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={},
    profiles=[],
    data=["∇φ", "B0"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _total_B(params, transforms, profiles, data, RpZ_data, **kwargs):
    RpZ_data["B"] = RpZ_data["∇φ"] + RpZ_data["B0"]
    return RpZ_data


@register_compute_fun(
    name="B0*n",
    label="B_0 \\cdot n_{\\rho}",
    units="T",
    units_long="Tesla",
    description="Auxillary field dotted into flux surface normal",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=[],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    B0="_MagneticField : Field object to compute with.",
    surface="Surface : Surface to compute on.",
    chunk_size=_doc["chunk_size"],
)
def _B0_dot_n(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    data["B0*n"] = kwargs["B0"].compute_Bnormal(
        kwargs["surface"],
        eval_grid=grid,
        source_grid=grid,
        vc_source_grid=grid,
        chunk_size=kwargs.get("chunk_size", None),
    )[0]
    return data


@register_compute_fun(
    name="B0",
    label="B0",
    units="T",
    units_long="Tesla",
    description="Auxillary field",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=[],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    chunk_size=_doc["chunk_size"],
    B0="_MagneticField : Field object to compute with.",
    public=False,
)
def _B0_field(params, transforms, profiles, data, RpZ_data, **kwargs):
    coords = jnp.column_stack([RpZ_data["R"], RpZ_data["phi"], RpZ_data["Z"]])
    RpZ_data["B0"] = kwargs["B0"].compute_magnetic_field(
        coords=coords,
        source_grid=transforms["grid"],
        chunk_size=kwargs.get("chunk_size", None),
    )
    return RpZ_data


@register_compute_fun(
    name="B_coil",
    label="B_{\\text{coil}}",
    units="T",
    units_long="Tesla",
    description="Magnetic field due to coils",
    dim=3,
    coordinates="rtz",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=["x"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    chunk_size=_doc["chunk_size"],
    B_coil="_MagneticField : Field object to compute with.",
)
def _B_coil_field(params, transforms, profiles, data, **kwargs):
    data["B_coil"] = kwargs["B_coil"].compute_magnetic_field(
        coords=data["x"],
        source_grid=transforms["grid"],
        chunk_size=kwargs.get("chunk_size", None),
    )
    return data


@register_compute_fun(
    name="Y_coil",
    label="Y_{\\text{coil}}",
    units="T m",
    units_long="Tesla meter",
    description="Net poloidal current produced by magnetic coils",
    dim=0,
    coordinates="",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=["e_zeta", "B_coil"],
    chunk_size=_doc["chunk_size"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Y_coil(params, transforms, profiles, data, **kwargs):
    assert transforms["grid"].num_rho == 1
    assert np.isclose(transforms["grid"].nodes[0, 0], 1)
    # Equation B.2 in [1].
    data["Y_coil"] = dot(data["B_coil"], data["e_zeta"]).mean()
    return data


@register_compute_fun(
    name="n_rho x B_coil",
    label="n_{\\rho} \\times B_{\\text{coil}}",
    units="T",
    units_long="Tesla",
    description="Flux surface normal cross magnetic field due to coils",
    dim=1,
    coordinates="rtz",
    params=[],
    transforms={},
    profiles=[],
    data=["n_rho", "B_coil"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _n_rho_x_B_coil(params, transforms, profiles, data, **kwargs):
    data["n_rho x B_coil"] = cross(data["n_rho"], data["B_coil"])
    return data


# TODO: compute this from vector or scalar potential
@register_compute_fun(
    name="Phi_coil_mn",
    label="\\Phi_{\\text{coil}, mn}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of periodic part of coil scalar potential",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi_coil": [[0, 0, 0]]},
    profiles=[],
    data=["n_rho x B_coil", "n_rho x grad(theta)", "n_rho x grad(zeta)", "Y_coil"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Phi_mn_coil(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    assert grid.num_rho == 1
    assert np.isclose(grid.nodes[0, 0], 1)

    basis = transforms["Phi_coil"].basis
    _t = basis.evaluate(grid, [0, 1, 0])[:, jnp.newaxis]
    _z = basis.evaluate(grid, [0, 0, 1])[:, jnp.newaxis]

    mat = (
        _t * data["n_rho x grad(theta)"][..., jnp.newaxis]
        + _z * data["n_rho x grad(zeta)"][..., jnp.newaxis]
    ).reshape(grid.num_nodes * 3, basis.num_modes)

    # Equation 5.16 in [1].
    data["Phi_coil_mn"] = jnp.linalg.lstsq(
        mat,
        (data["n_rho x B_coil"] - data["Y_coil"] * data["n_rho x grad(zeta)"]).ravel(),
    )[0]
    return data


@register_compute_fun(
    name="Phi_coil (periodic)",
    label="(n \\times \\nabla)^{-1} (n \\times B_{\\text{coil}})",
    units="T m",
    units_long="Tesla meter",
    description="Periodic part of magnetic scalar potential of coil field",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi_coil": [[0, 0, 0]]},
    profiles=[],
    data=["Phi_coil_mn"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Phi_coil_periodic(params, transforms, profiles, data, **kwargs):
    data["Phi_coil (periodic)"] = transforms["Phi_coil"].transform(data["Phi_coil_mn"])
    return data


@register_compute_fun(
    name="Phi_coil (secular)",
    label="(n \\times \\nabla)^{-1} (n \\times B_{\\text{coil}})",
    units="T m",
    units_long="Tesla meter",
    description="Secular part of magnetic scalar potential of coil field",
    dim=1,
    coordinates="z",
    params=[],
    transforms={},
    profiles=[],
    data=["zeta", "Y_coil"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Phi_coil_secular(params, transforms, profiles, data, **kwargs):
    data["Phi_coil (secular)"] = data["Y_coil"] * data["zeta"]
    return data


@register_compute_fun(
    name="Phi_coil",
    label="(n \\times \\nabla)^{-1} (n \\times B_{\\text{coil}})",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential of coil field",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["Phi_coil (periodic)", "Phi_coil (secular)"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Phi_coil(params, transforms, profiles, data, **kwargs):
    data["Phi_coil"] = data["Phi_coil (periodic)"] + data["Phi_coil (secular)"]
    return data


@register_compute_fun(
    name="Phi_mn",
    label="\\Phi_{m n}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of periodic part of potential",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=list(set(_kernel_dipole_plus_half.keys) - {"Phi (periodic)"})
    + ["Phi_coil (periodic)", "S[B0*n]", "interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
    **_doc,
)
def _scalar_potential_mn_free_surface(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency

    boundary_condition = data["S[B0*n]"] - data["Phi_coil (periodic)"]

    if kwargs.get("maxiter", -1) > 0:
        data["Phi (periodic)"] = _fixed_point_potential(
            boundary_condition,
            data,
            data,
            data["interpolator"],
            **{key: kwargs[key] for key in _doc if key in kwargs},
        )
        if kwargs.get("_full_output", False):
            data["Phi (periodic)"], (err, data["num iter"]) = data["Phi (periodic)"]
            data["Phi error"] = jnp.abs(err).max()

        transforms["Phi"].build_pinv()
        data["Phi_mn"] = transforms["Phi"].fit(data["Phi (periodic)"])
    else:
        data["Phi_mn"] = _lsmr_compute_potential(
            boundary_condition,
            data,
            data,
            data["interpolator"],
            transforms["Phi"].basis,
            problem="interior Dirichlet",
            **kwargs,
        )
    return data


@register_compute_fun(
    name="γ potential",
    label="\\gamma",
    units="T m",
    units_long="Tesla meter",
    description="Double layer potential with dipole density -Φ",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=_kernel_dipole_plus_half.keys + ["interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
    chunk_size=_doc["chunk_size"],
    public=False,
)
def _gamma_potential(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    data["Phi(x) (periodic)"] = data["Phi (periodic)"]
    # Left hand side of equation 5.15 in [1] computed by evaluating
    # the right hand side. This is used for testing.
    data["γ potential"] = data["Phi (periodic)"] - _D_plus_half(
        data,
        data,
        data["interpolator"],
        chunk_size=kwargs.get("chunk_size", None),
    )
    return data
