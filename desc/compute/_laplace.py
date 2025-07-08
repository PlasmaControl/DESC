"""Compute functions for magnetic fields which do not assume nested surfaces."""

from functools import partial

import numpy as np

from desc.backend import jnp
from desc.integrals.singularities import (
    _kernel_biot_savart_coulomb,
    _kernel_dipole,
    _kernel_dipole_smooth,
    _kernel_monopole,
    _nonsingular_part,
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
    "maxiter": """int :
        Maximum number of iterations for fixed point method.
        Default is zero, which means that matrix inversion will be used.
        """,
    "method": """{'del2', 'simple'} :
        Method of finding fixed-point. Default is ``simple``.
        """,
    "tol": """float :
        Stopping tolerance for fixed point method.""",
    "guess": """jnp.ndarray :
        Initial guess for iteration.
        """,
}
RpZ_coords = """dict[str, jnp.ndarray] :
        (R, ϕ, Z) coordinates at which to evaluate quantities.
        Should store the three entries ``"R"``, ``"phi"``, and ``"Z"``.
        If not given, the entries for these keys in ``data`` are used.
        """


def _D(eval_data, source_data, interpolator, basis=None, chunk_size=None):
    """Compute D[Φ](x) = ∫_y Φ(y)〈∇_x G(x−y),ds(y)〉.

    If ``basis`` is not supplied, then computes D[Φ].
    If ``basis`` is supplied, then constructs the operator which
    acts on the spectral coefficients of Φ in the supplied + secular basis.

    """
    kwargs = {}
    if basis is not None:
        kwargs["known_map"] = ("Phi", partial(basis.evaluate, secular=True))
        kwargs["ndim"] = basis.num_modes + 2
    return singular_integral(
        eval_data=eval_data,
        source_data=source_data,
        interpolator=interpolator,
        kernel=_kernel_dipole,
        chunk_size=chunk_size,
        **kwargs,
    )


def _D_plus_half(eval_data, source_data, interpolator, basis=None, chunk_size=None):
    """Compute (D[Φ] + Φ/2)(x).

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
        eval_data=eval_data,
        source_data=source_data,
        interpolator=interpolator,
        kernel=_kernel_dipole_smooth,
        chunk_size=chunk_size,
        **kwargs,
    )


def _lsmr_compute_potential(
    problem,
    boundary_condition,
    I,  # noqa: E741
    Y,
    basis,
    interpolator,
    potential_data,
    source_data,
    same_grid,
    chunk_size=None,
):
    assert problem in {"interior Neumann", "exterior Neumann", "interior Dirichlet"}

    potential_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    assert basis.M <= potential_grid.M
    assert basis.N <= potential_grid.N

    Phi = basis.evaluate(potential_grid, secular=True)
    potential_data["Phi(x)"] = Phi
    source_data["Phi"] = Phi if same_grid else basis.evaluate(source_grid, secular=True)

    D = _D_plus_half(potential_data, source_data, interpolator, basis, chunk_size)
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
    data["interpolator"] = get_interpolator(grid, grid, data, **kwargs)
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
        set(_kernel_dipole_smooth.keys + _kernel_monopole.keys + ["interpolator"])
        - {"Phi"}
    ),
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    problem='str :Problem to solve in {"interior Neumann", "exterior Neumann"}.',
    **_doc,
)
def _scalar_potential_mn_Neumann(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    problem = kwargs["problem"]
    maxiter = kwargs.get("maxiter", 0)

    # TODO: Fixed point method.
    errorif(
        ("interior" in problem) and (maxiter > 0),
        msg="Fixed point method not possible for the interior Neumann.",
    )

    copy_data = data.copy()
    interpolator = copy_data.pop("interpolator")
    bc = singular_integral(
        eval_data=copy_data,
        source_data=copy_data,
        interpolator=interpolator,
        kernel=_kernel_monopole,
        chunk_size=kwargs.get("chunk_size", None),
    ).squeeze(axis=-1)

    data["Phi_mn"] = _lsmr_compute_potential(
        problem=problem,
        boundary_condition=bc,
        I=params["I"],
        Y=params["Y"],
        basis=transforms["Phi"].basis,
        interpolator=interpolator,
        potential_data=copy_data,
        source_data=copy_data,
        same_grid=True,
        chunk_size=kwargs.get("chunk_size", None),
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
    data=list(set(_kernel_dipole_smooth.keys) - {"Phi"}) + ["interpolator", "Phi_coil"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
    **_doc,
)
def _scalar_potential_mn_free_surface(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    # TODO: Fixed point method.
    copy_data = data.copy()
    data["Phi_mn"] = _lsmr_compute_potential(
        problem="interior Dirichlet",
        boundary_condition=copy_data["Phi_coil"],
        I=params["I"],
        Y=params["Y"],
        basis=transforms["Phi"].basis,
        interpolator=copy_data.pop("interpolator"),
        potential_data=copy_data,
        source_data=copy_data,
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
    RpZ_coords=RpZ_coords,
    on_boundary="bool : Whether coords are on boundary surface.",
)
def _grad_potential(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    # TODO: avoid near singular integral by removing singularity
    chunk_size = kwargs.get("chunk_size", None)
    copy = data.copy()
    RpZ_coords = kwargs.get("RpZ_coords", copy)
    interpolator = kwargs.get("eval_interpolator", copy.pop("interpolator", None))
    sign = 1 - 2 * int("exterior" in kwargs.get("problem", ""))

    if kwargs["on_boundary"]:
        data["grad(Phi)"] = (
            sign
            * 2
            * singular_integral(
                eval_data=RpZ_coords,
                source_data=copy,
                interpolator=interpolator,
                kernel=_kernel_biot_savart_coulomb,
                chunk_size=chunk_size,
            )
        )
    else:
        data["grad(Phi)"] = sign * _nonsingular_part(
            eval_data=RpZ_coords,
            eval_grid=None,
            source_data=copy,
            source_grid=transforms["grid"],
            st=jnp.nan,
            sz=jnp.nan,
            kernel=_kernel_biot_savart_coulomb,
            chunk_size=chunk_size,
        )

    return data


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
    description="Magnetic field due to volume current "
    "where the potential is defined and due to net currents elsewhere.",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=[],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    RpZ_coords=RpZ_coords,
    chunk_size=_doc["chunk_size"],
    B0="_MagneticField : Field object to compute with.",
)
def _B0_field(params, transforms, profiles, data, **kwargs):
    RpZ_coords = kwargs.get("RpZ_coords", data)
    coords = jnp.column_stack([RpZ_coords["R"], RpZ_coords["phi"], RpZ_coords["Z"]])
    data["B0"] = kwargs["B0"].compute_magnetic_field(
        coords=coords,
        source_grid=transforms["grid"],
        chunk_size=kwargs.get("chunk_size", None),
    )
    return data


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
def _total_B(params, transforms, profiles, data, **kwargs):
    data["B"] = data["grad(Phi)"] + data["B0"]
    return data


@register_compute_fun(
    name="B*n",
    label="B \\cdot n_{\\rho}",
    units="T",
    units_long="Tesla",
    description="Magnetic field",
    dim=1,
    coordinates="RpZ",
    params=[],
    transforms={},
    profiles=[],
    data=["B", "n_rho"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _B_dot_n_laplace(params, transforms, profiles, data, **kwargs):
    data["B*n"] = dot(data["B"], data["n_rho"])
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
    data=["e_zeta", "x"],
    # TODO: make this vector potential
    B_coil="""_MagneticField : Magnetic field due to coils.""",
    chunk_size=_doc["chunk_size"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Y_coil(params, transforms, profiles, data, **kwargs):
    assert transforms["grid"].num_rho == 1
    assert np.isclose(transforms["grid"].nodes[0, 0], 1)
    B_coil = kwargs["B_coil"].compute_magnetic_field(
        coords=data["x"],
        source_grid=transforms["grid"],
        chunk_size=kwargs.get("chunk_size", None),
    )
    # Equation B.2 averaged over all χ_θ for increased accuracy
    # since we only have discrete interpolation to true B_coil.
    # (L2 error of Fourier series better than max pointwise).
    data["Y_coil"] = dot(B_coil, data["e_zeta"]).mean()
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
    data=["n_rho"],
    # TODO: make this vector potential
    B_coil="_MagneticField : Magnetic field due to coils.",
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _n_rho_x_B_coil(params, transforms, profiles, data, **kwargs):
    data["n_rho x B_coil"] = cross(data["n_rho"], kwargs["B_coil"])
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
    basis = transforms["Phi"].basis
    grid = transforms["Phi"].grid

    _t = basis.evaluate(grid, [0, 1, 0])[:, jnp.newaxis]
    _z = basis.evaluate(grid, [0, 0, 1])[:, jnp.newaxis]
    # n × ∇ in Fourier basis
    mat = (
        _t * data["n_rho x grad(theta)"][..., jnp.newaxis]
        + _z * data["n_rho x grad(zeta)"][..., jnp.newaxis]
    ).reshape(grid.num_nodes * 3, basis.num_modes)

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
