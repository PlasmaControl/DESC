"""Compute functions for Laplace solver."""

from functools import partial

import numpy as np

from desc.backend import fixed_point, jit, jnp
from desc.integrals.singularities import (
    _kernel_biot_savart_coulomb,
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
    "chunk_size": """int or None :
        Size to split integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.  Default is ``None``.
        Recommend to verify computation with ``chunk_size`` set to a
        small number due to bugs in JAX or XLA.
        """,
    "Phi_0": """jnp.ndarray :
        Initial guess for iteration.
        """,
    "maxiter": """int :
        Maximum number of iterations for fixed point method.
        Default is zero, which means that matrix inversion will be used.
        """,
}


def _D_plus_half(
    eval_data, source_data, interpolator, basis=None, chunk_size=None, _prune_data=True
):
    """Compute (D[Φ] + Φ/2)(x).

    D[Φ](x) = ∫_y Φ(y)〈∇_x G(x−y),ds(y)〉.
    If ``basis`` is not supplied, then computes (D[Φ] + Φ/2)(x).
    If ``basis`` is supplied, then constructs the operator which
    acts on the spectral coefficients of Φ in the supplied + secular basis.

    """
    kwargs = {}
    if basis is not None:
        kwargs["known_map"] = ("Phi", partial(basis.evaluate, secular=True))
        kwargs["ndim"] = basis.num_modes + 2
    # TODO: This integral is not singular. See prescription in shared paper.
    return singular_integral(
        eval_data,
        source_data,
        interpolator,
        _kernel_dipole_plus_half,
        chunk_size=chunk_size,
        _prune_data=_prune_data,
        **kwargs,
    )


def _lsmr_compute_potential(
    boundary_condition,
    I,  # noqa: E741
    Y,
    potential_data,
    source_data,
    interpolator,
    basis,
    problem,
    same_grid,
    chunk_size=None,
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
    Phi = basis.evaluate(potential_grid, secular=True)
    potential_data["Phi(x)"] = Phi
    source_data["Phi"] = Phi if same_grid else basis.evaluate(source_grid, secular=True)

    D = _D_plus_half(
        potential_data, source_data, interpolator, basis, chunk_size, _prune_data=False
    )
    if problem == "exterior Neumann" or problem == "interior Dirichlet":
        D -= Phi
    assert D.shape == (potential_grid.num_nodes, basis.num_modes + 2)

    # TODO: Test that boundary condition is periodic.
    #       For free boundary this is equivalent to the
    #       statement that equation 5.12 vanishes on the boundary
    #       if nu_1 = I theta. That is if D[theta] = 1/2,
    #       that is if (D + 1/2)[theta] = (1 + theta)/2
    #       Thus we can test this by checking D array last column.
    boundary_condition -= I * D[:, -2] + Y * D[:, -1]
    D = D[:, :-2]

    # Solving overdetermined system useful to reduce size of D while
    # retaining FFT interpolation accuracy in the singular integrals.
    # TODO: https://github.com/patrick-kidger/lineax/pull/86
    return (
        jnp.linalg.solve(D, boundary_condition)
        if (potential_grid.num_nodes == basis.num_modes)
        else jnp.linalg.lstsq(D, boundary_condition)[0]
    )


def _iteration_operator(
    Phi, gamma, potential_data, source_data, interpolator, chunk_size
):
    # TODO: does this have to be pure?
    potential_data["Phi(x)"] = Phi
    source_data["Phi"] = Phi
    return (
        _D_plus_half(
            potential_data,
            source_data,
            interpolator,
            chunk_size=chunk_size,
            _prune_data=False,
        )
        - gamma
    )


def _fixed_point_potential(
    boundary_condition,
    I,
    Y,
    potential_data,
    source_data,
    interpolator,
    same_grid,
    chunk_size=None,
    Phi_0=None,
    xtol=1e-6,
    maxiter=20,
    method="simple",
    **kwargs,
):
    # TODO: Appendix B for secular part.
    errorif(not same_grid, NotImplementedError)

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
        method,
        scalar=True,
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
    grid_requirement={"can_fft2": True},
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
    name="Phi_mn",
    label="\\Phi_{m n}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of periodic part of potential",
    dim=1,
    coordinates="tz",
    params=["I", "Y"],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=list(
        set(_kernel_dipole_plus_half.keys + _kernel_monopole.keys + ["interpolator"])
        - {"Phi"}
    ),
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    problem='str :Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    **_doc,
)
@partial(jit, static_argnames=["problem", "chunk_size", "maxiter"])
def _scalar_potential_mn_Neumann(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    problem = kwargs["problem"]
    chunk_size = kwargs.get("chunk_size", None)

    boundary_condition = singular_integral(
        data,
        data,
        data["interpolator"],
        _kernel_monopole,
        chunk_size=chunk_size,
    ).squeeze(axis=-1)

    if kwargs.get("maxiter", -1) > 0:
        errorif(
            "interior" in problem,
            msg="maxiter cannot be positive for interior Neumann problem.",
        )
        data["Phi"] = _fixed_point_potential(
            boundary_condition,
            params["I"],
            params["Y"],
            data,
            data,
            data["interpolator"],
            same_grid=True,
            **kwargs,
        )
        data["Phi_mn"] = transforms["Phi"].fit(data["Phi"])
    else:
        data["Phi_mn"] = _lsmr_compute_potential(
            boundary_condition,
            params["I"],
            params["Y"],
            data,
            data,
            data["interpolator"],
            transforms["Phi"].basis,
            problem,
            same_grid=True,
            chunk_size=chunk_size,
        )
    return data


@register_compute_fun(
    name="Phi_mn",
    label="\\Phi_{m n}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of periodic part of potential",
    dim=1,
    coordinates="tz",
    params=["I", "Y"],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=list(set(_kernel_dipole_plus_half.keys) - {"Phi"})
    + ["interpolator", "Phi_coil"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
    **_doc,
)
@partial(jit, static_argnames=["chunk_size", "maxiter"])
def _scalar_potential_mn_free_surface(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    if kwargs.get("maxiter", -1) > 0:
        data["Phi"] = _fixed_point_potential(
            data["Phi_coil"],
            params["I"],
            params["Y"],
            data,
            data,
            data["interpolator"],
            same_grid=True,
            **kwargs,
        )
        data["Phi_mn"] = transforms["Phi"].fit(data["Phi"])
    else:
        data["Phi_mn"] = _lsmr_compute_potential(
            data["Phi_coil"],
            params["I"],
            params["Y"],
            data,
            data,
            data["interpolator"],
            transforms["Phi"].basis,
            problem="interior Dirichlet",
            same_grid=True,
            chunk_size=kwargs.get("chunk_size", None),
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
    transforms={"Phi": [[0, 1, 0]]},
    profiles=[],
    data=["Phi_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _pot_Phi_t(params, transforms, profiles, data, **kwargs):
    assert data["Phi_mn"].size == transforms["Phi"].basis.num_modes
    data["Phi_t"] = transforms["Phi"].transform(data["Phi_mn"], dt=1) + params["I"]
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
    transforms={"Phi": [[0, 0, 1]]},
    profiles=[],
    data=["Phi_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _pot_Phi_z(params, transforms, profiles, data, **kwargs):
    assert data["Phi_mn"].size == transforms["Phi"].basis.num_modes
    data["Phi_z"] = transforms["Phi"].transform(data["Phi_mn"], dz=1) + params["Y"]
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
    name="grad(Phi)",
    label="\\nabla \\Phi",
    units="T",
    units_long="Tesla",
    description="Magnetic field due to potential",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={},
    profiles=[],
    data=_kernel_biot_savart_coulomb.keys,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    problem='str :Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    eval_interpolator="""_BIESTInterpolator :
        Interpolator from source grid to evaluation grid on boundary.
        If not given, default is to interpolate to source grid.
        """,
    on_boundary="bool : Whether coords are on boundary surface.",
)
def _grad_potential(params, transforms, profiles, data, RpZ_data, **kwargs):
    # noqa: unused dependency
    chunk_size = kwargs.get("chunk_size", None)
    interpolator = kwargs.get("eval_interpolator", data.get("interpolator", None))
    sign = 1 - 2 * int("exterior" in kwargs.get("problem", ""))

    # TODO: avoid near singular integral by removing singularity
    if kwargs["on_boundary"]:
        RpZ_data["grad(Phi)"] = (
            sign
            * 2
            * singular_integral(
                RpZ_data,
                data,
                interpolator,
                _kernel_biot_savart_coulomb,
                chunk_size=chunk_size,
            )
        )
    else:
        grid = transforms["grid"]
        eval_data, source_data = _prune_data(
            RpZ_data, None, data, grid, _kernel_biot_savart_coulomb
        )
        RpZ_data["grad(Phi)"] = sign * _nonsingular_part(
            eval_data,
            None,
            source_data,
            grid,
            st=jnp.nan,
            sz=jnp.nan,
            kernel=_kernel_biot_savart_coulomb,
            chunk_size=chunk_size,
        )
    return RpZ_data


@register_compute_fun(
    name="B0*n",
    label="B_0 \\cdot n_{\\rho}",
    units="T",
    units_long="Tesla",
    description="Magnetic field due to volume current "
    "where the potential is defined and due to net currents elsewhere, "
    "dotted into flux surface normal",
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
    description="Auxillary field.",
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
    data=["grad(Phi)", "B0"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _total_B(params, transforms, profiles, data, RpZ_data, **kwargs):
    RpZ_data["B"] = RpZ_data["grad(Phi)"] + RpZ_data["B0"]
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
    # Equation B.2 averaged over all χ_θ for increased accuracy
    # since we only have discrete interpolation to true B_coil.
    # (L2 error of Fourier series better than max pointwise).
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


@register_compute_fun(
    name="Phi_coil_mn",
    label="(n \\times \\nabla)^{-1} (n \\times B_{\\text{coil}})",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of periodic part of coil scalar potential",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=["n_rho x B_coil", "n_rho x grad(theta)", "n_rho x grad(zeta)", "Y_coil"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Phi_mn_coil(params, transforms, profiles, data, **kwargs):
    assert transforms["grid"].num_rho == 1
    assert np.isclose(transforms["grid"].nodes[0, 0], 1)

    basis = transforms["Phi"].basis
    grid = transforms["Phi"].grid

    _t = basis.evaluate(grid, [0, 1, 0])[:, jnp.newaxis]
    _z = basis.evaluate(grid, [0, 0, 1])[:, jnp.newaxis]
    # n × ∇ in Fourier basis
    mat = (
        _t * data["n_rho x grad(theta)"][..., jnp.newaxis]
        + _z * data["n_rho x grad(zeta)"][..., jnp.newaxis]
    ).reshape(grid.num_nodes * 3, basis.num_modes)

    # TODO: compute this vector or scalar potential
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
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    data=["Phi_coil_mn"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Phi_coil_periodic(params, transforms, profiles, data, **kwargs):
    data["Phi_coil (periodic)"] = transforms["Phi"].transform(data["Phi_coil_mn"])
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
