"""Compute functions for quantities with obvious geometric meaning.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from quadax import simpson

from desc.backend import jnp

from ..grid import QuadratureGrid
from ..integrals.surface_integral import line_integrals, surface_integrals
from ..utils import cross, dot, safenorm
from .data_index import register_compute_fun


@register_compute_fun(
    name="V",
    label="V",
    units="m^{3}",
    units_long="cubic meters",
    description="Volume scaled by max(ρ)⁻² to extrapolate to last closed flux surface",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["sqrt(g)", "V(r)", "rho"],
    resolution_requirement="rtz",
)
def _V(params, transforms, profiles, data, **kwargs):
    if isinstance(transforms["grid"], QuadratureGrid):
        data["V"] = jnp.sum(data["sqrt(g)"] * transforms["grid"].weights)
    else:
        # To approximate volume at ρ ~ 1, we scale by ρ⁻², assuming the integrand
        # varies little from max ρ to ρ = 1 and a roughly circular cross-section.
        data["V"] = jnp.max(data["V(r)"]) / jnp.max(data["rho"]) ** 2
    return data


@register_compute_fun(
    name="V",
    label="V",
    units="m^{3}",
    units_long="cubic meters",
    description="Volume enclosed by surface",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["V(r)"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
)
def _V_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    data["V"] = jnp.max(data["V(r)"])
    return data


@register_compute_fun(
    name="V(r)",
    label="V(\\rho)",
    units="m^{3}",
    units_long="cubic meters",
    description="Volume enclosed by flux surfaces",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["e_theta", "e_zeta", "Z"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    resolution_requirement="tz",
)
def _V_of_r(params, transforms, profiles, data, **kwargs):
    # divergence theorem: integral(dV div [0, 0, Z]) = integral(dS dot [0, 0, Z])
    data["V(r)"] = jnp.abs(
        surface_integrals(
            transforms["grid"],
            cross(data["e_theta"], data["e_zeta"])[:, 2] * data["Z"],
        )
    )
    return data


@register_compute_fun(
    name="V_r(r)",
    label="\\partial_{\\rho} V(\\rho)",
    units="m^{3}",
    units_long="cubic meters",
    description="Volume enclosed by flux surfaces, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)"],
    resolution_requirement="tz",
)
def _V_r_of_r(params, transforms, profiles, data, **kwargs):
    # eq. 4.9.10 in W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.
    data["V_r(r)"] = surface_integrals(transforms["grid"], data["sqrt(g)"])
    return data


@register_compute_fun(
    name="V_rr(r)",
    label="\\partial_{\\rho\\rho} V(\\rho)",
    units="m^{3}",
    units_long="cubic meters",
    description="Volume enclosed by flux surfaces, second derivative wrt radial "
    + "coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)_r"],
    resolution_requirement="tz",
)
def _V_rr_of_r(params, transforms, profiles, data, **kwargs):
    # The sign of sqrt(g) is enforced to be non-negative.
    data["V_rr(r)"] = surface_integrals(transforms["grid"], data["sqrt(g)_r"])
    return data


@register_compute_fun(
    name="V_rrr(r)",
    label="\\partial_{\\rho\\rho\\rho} V(\\rho)",
    units="m^{3}",
    units_long="cubic meters",
    description="Volume enclosed by flux surfaces, third derivative wrt radial "
    + "coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)_rr"],
    resolution_requirement="tz",
)
def _V_rrr_of_r(params, transforms, profiles, data, **kwargs):
    # The sign of sqrt(g) is enforced to be non-negative.
    data["V_rrr(r)"] = surface_integrals(transforms["grid"], data["sqrt(g)_rr"])
    return data


def _compute_A_of_z(grid, data, rho):
    # TODO: For nonzero omega we need to integrate over theta at constant phi.
    #  Add source_grid_requirement={"coordinates": "rtp", "is_meshgrid": True}
    # TODO: In compute_utils recognize when omega = 0 and ignore all source
    #  grid requirements if the given grid satisfies them with phi replaced by zeta.
    if isinstance(grid, QuadratureGrid):
        return surface_integrals(
            grid,
            # TODO: generalize for nonzero omega.
            data["|e_rho x e_theta|"],
            surface_label="zeta",
            expand_out=False,
        )
    # Denote any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ by v = [v¹, v², v³] where R̂, ϕ̂, Ẑ
    # are the normalized basis vectors of the cylindrical coordinates R, ϕ, Z.
    # We use a 2D divergence theorem over constant ϕ toroidal surface (i.e. R, Z plane).
    # In this geometry, the divergence operator in this coordinate system is
    # div = ([∂_R, ∂_ϕ, ∂_Z] ⊗ [1, 0, 1]) dot .
    # ∫ dA div v = ∫ dℓ n dot v
    # where n is the unit normal such that n dot e_θ|ρ,ϕ = 0 and n dot e_ϕ|R,Z = 0,
    # and the labels following | denote those coordinates are fixed.
    # Now choose v = [0, 0, Z], and n in the direction (e_θ|ρ,ζ × e_ζ|ρ,θ) ⊗ [1, 0, 1].
    n = data["n_rho"]
    n = n.at[:, 1].set(0)
    n = n / jnp.linalg.norm(n, axis=-1)[:, jnp.newaxis]
    return jnp.abs(
        line_integrals(
            grid,
            data["Z"] * n[:, 2] * safenorm(data["e_theta|r,p"], axis=-1),
            line_label="theta",
            fix_surface=("rho", rho),
            expand_out=False,
        )
    )


@register_compute_fun(
    name="A(z)",
    label="A(\\zeta)",
    units="m^{2}",
    units_long="square meters",
    description="Area of enclosed cross-section (enclosed constant phi surface), "
    "scaled by max(ρ)⁻² to extrapolate to last closed flux surface",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=["Z", "n_rho", "e_theta|r,p", "rho", "|e_rho x e_theta|"],
    parameterization=["desc.equilibrium.equilibrium.Equilibrium"],
    resolution_requirement="t",
    grid_requirement={"sym": False},
)
def _A_of_z(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    # To approximate area at ρ ~ 1, we scale by ρ⁻², assuming the integrand
    # varies little from max ρ to ρ = 1 and a roughly circular cross-section.
    max_rho = jnp.max(data["rho"])
    data["A(z)"] = transforms["grid"].expand(
        _compute_A_of_z(transforms["grid"], data, max_rho) / max_rho**2,
        surface_label="zeta",
    )
    return data


@register_compute_fun(
    name="A(z)",
    label="A(\\zeta)",
    units="m^{2}",
    units_long="square meters",
    description="Area of enclosed cross-section (enclosed constant phi surface)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=["Z", "n_rho", "e_theta|r,p", "rho", "|e_rho x e_theta|"],
    parameterization=[
        "desc.geometry.surface.ZernikeRZToroidalSection",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    resolution_requirement="t",
    grid_requirement={"sym": False},
)
def _A_of_z_surface(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    data["A(z)"] = transforms["grid"].expand(
        _compute_A_of_z(transforms["grid"], data, jnp.max(data["rho"])),
        surface_label="zeta",
    )
    return data


@register_compute_fun(
    name="A",
    label="A",
    units="m^{2}",
    units_long="square meters",
    description="Average enclosed cross-sectional area, scaled by max(ρ)⁻²"
    " to extrapolate to last closed flux surface",
    # Simple toroidal average A₀ = ∫ A(ζ) dζ / (2π) matches the convention for the
    # average major radius R₀ = ∫ R(ρ=0) dζ / (2π).
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["Z", "n_rho", "e_theta|r,p", "rho", "|e_rho x e_theta|"],
    parameterization=["desc.equilibrium.equilibrium.Equilibrium"],
    resolution_requirement="tz",
)
def _A(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    # To approximate area at ρ ~ 1, we scale by ρ⁻², assuming the integrand
    # varies little from max ρ to ρ = 1 and a roughly circular cross-section.
    max_rho = jnp.max(data["rho"])
    data["A"] = (
        jnp.mean(_compute_A_of_z(transforms["grid"], data, max_rho)) / max_rho**2
    )
    return data


@register_compute_fun(
    name="A",
    label="A",
    units="m^{2}",
    units_long="square meters",
    description="Average enclosed cross-sectional area",
    # Simple toroidal average A₀ = ∫ A(ζ) dζ / (2π) matches the convention for the
    # average major radius R₀ = ∫ R(ρ=0) dζ / (2π).
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["Z", "n_rho", "e_theta|r,p", "rho", "|e_rho x e_theta|"],
    parameterization=["desc.geometry.core.Surface"],
    resolution_requirement="tz",
)
def _A_surface(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    data["A"] = jnp.mean(
        _compute_A_of_z(transforms["grid"], data, jnp.max(data["rho"]))
    )
    return data


@register_compute_fun(
    name="A(r)",
    label="A(\\rho)",
    units="m^{2}",
    units_long="square meters",
    description="Average cross-sectional area enclosed by flux surfaces",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["R0", "V(r)"],
)
def _A_of_r(params, transforms, profiles, data, **kwargs):
    data["A(r)"] = data["V(r)"] / (2 * jnp.pi * data["R0"])
    return data


@register_compute_fun(
    name="S",
    label="S",
    units="m^{2}",
    units_long="square meters",
    description="Surface area of outermost flux surface, scaled by max(ρ)⁻¹"
    " to extrapolate to last closed flux surface",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["S(r)", "rho"],
    parameterization=["desc.equilibrium.equilibrium.Equilibrium"],
)
def _S(params, transforms, profiles, data, **kwargs):
    # To approximate surface are at ρ ~ 1, we scale by ρ⁻¹, assuming the integrand
    # varies little from max ρ to ρ = 1.
    data["S"] = jnp.max(data["S(r)"]) / jnp.max(data["rho"])
    return data


@register_compute_fun(
    name="S",
    label="S",
    units="m^{2}",
    units_long="square meters",
    description="Surface area",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["S(r)"],
    parameterization=["desc.geometry.surface.FourierRZToroidalSurface"],
)
def _S_fourier_rz_surface(params, transforms, profiles, data, **kwargs):
    data["S"] = jnp.max(data["S(r)"])
    return data


@register_compute_fun(
    name="S(r)",
    label="S(\\rho)",
    units="m^{2}",
    units_long="square meters",
    description="Surface area of flux surfaces",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|e_theta x e_zeta|"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    resolution_requirement="tz",
)
def _S_of_r(params, transforms, profiles, data, **kwargs):
    data["S(r)"] = surface_integrals(transforms["grid"], data["|e_theta x e_zeta|"])
    return data


@register_compute_fun(
    name="S_r(r)",
    label="\\partial_{\\rho} S(\\rho)",
    units="m^{2}",
    units_long="square meters",
    description="Surface area of flux surfaces, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|e_theta x e_zeta|_r"],
    resolution_requirement="tz",
)
def _S_r_of_r(params, transforms, profiles, data, **kwargs):
    data["S_r(r)"] = surface_integrals(transforms["grid"], data["|e_theta x e_zeta|_r"])
    return data


@register_compute_fun(
    name="S_rr(r)",
    label="\\partial_{\\rho\\rho} S(\\rho)",
    units="m^{2}",
    units_long="square meters",
    description="Surface area of flux surfaces, second derivative wrt radial"
    " coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|e_theta x e_zeta|_rr"],
    resolution_requirement="tz",
)
def _S_rr_of_r(params, transforms, profiles, data, **kwargs):
    data["S_rr(r)"] = surface_integrals(
        transforms["grid"], data["|e_theta x e_zeta|_rr"]
    )
    return data


@register_compute_fun(
    name="R0",
    label="R_{0} = V / (2\\pi A) = \\int R(\\rho=0) d\\zeta / (2\\pi)",
    units="m",
    units_long="meters",
    description="Average major radius",
    # This differs from the average value of R on the magnetic axis.
    # R₀ ≠ 〈 R(ρ=0) 〉 =  ∫ (R ‖e_ζ‖)(ρ=0) dζ / ∫ ‖e_ζ‖(ρ=0) dζ.
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["V", "A"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R0(params, transforms, profiles, data, **kwargs):
    data["R0"] = data["V"] / (2 * jnp.pi * data["A"])
    return data


@register_compute_fun(
    name="a",
    label="a",
    units="m",
    units_long="meters",
    description="Average minor radius",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["A"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _a(params, transforms, profiles, data, **kwargs):
    data["a"] = jnp.sqrt(data["A"] / jnp.pi)
    return data


@register_compute_fun(
    name="R0/a",
    label="R_{0} / a",
    units="~",
    units_long="None",
    description="Aspect ratio",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["R0", "a"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R0_over_a(params, transforms, profiles, data, **kwargs):
    data["R0/a"] = data["R0"] / data["a"]
    return data


@register_compute_fun(
    name="perimeter(z)",
    label="P(\\zeta)",
    units="m",
    units_long="meters",
    description="Perimeter of enclosed cross-section (enclosed constant phi surface), "
    "scaled by max(ρ)⁻¹ to extrapolate to last closed flux surface.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=["rho", "e_theta|r,p"],
    parameterization=["desc.equilibrium.equilibrium.Equilibrium"],
    resolution_requirement="t",
    grid_requirement={"sym": False},
    # TODO: For nonzero omega we need to integrate over theta at constant phi.
    #  Add source_grid_requirement={"coordinates": "rtp", "is_meshgrid": True}
    # TODO: Recognize when omega = 0 and ignore all source grid requirements
    #  if the given grid satisfies them with phi replaced by zeta.
)
def _perimeter_of_z(params, transforms, profiles, data, **kwargs):
    max_rho = jnp.max(data["rho"])
    data["perimeter(z)"] = (
        line_integrals(
            transforms["grid"],
            safenorm(data["e_theta|r,p"], axis=-1),
            line_label="theta",
            fix_surface=("rho", max_rho),
            expand_out=True,
        )
        # To approximate perimeter at ρ ~ 1, we scale by ρ⁻¹, assuming the integrand
        # varies little from max ρ to ρ = 1.
        / max_rho
    )
    return data


@register_compute_fun(
    name="perimeter(z)",
    label="P(\\zeta)",
    units="m",
    units_long="meters",
    description="Perimeter of enclosed cross-section (enclosed constant phi surface)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=["rho", "e_theta|r,p"],
    parameterization=["desc.geometry.core.Surface"],
    resolution_requirement="t",
    grid_requirement={"sym": False},
)
def _perimeter_of_z_surface(params, transforms, profiles, data, **kwargs):
    data["perimeter(z)"] = line_integrals(
        transforms["grid"],
        safenorm(data["e_theta|r,p"], axis=-1),
        line_label="theta",
        fix_surface=("rho", jnp.max(data["rho"])),
        expand_out=True,
    )
    return data


def _ramanujan(A, P):
    # derived from Ramanujan approximation for the perimeter of an ellipse
    a = (  # semi-major radius
        jnp.sqrt(3)
        * (
            jnp.sqrt(8 * jnp.pi * A + P**2)
            + jnp.sqrt(
                jnp.abs(
                    2 * jnp.sqrt(3) * P * jnp.sqrt(8 * jnp.pi * A + P**2)
                    - 40 * jnp.pi * A
                    + 4 * P**2
                )
            )
        )
        + 3 * P
    ) / (12 * jnp.pi)
    b = A / (jnp.pi * a)  # semi-minor radius
    return a / b


@register_compute_fun(
    name="a_major/a_minor",
    label="a_{\\mathrm{major}} / a_{\\mathrm{minor}}",
    units="~",
    units_long="None",
    description="Elongation at a toroidal cross-section",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=["A(z)", "perimeter(z)"],
    parameterization=["desc.equilibrium.equilibrium.Equilibrium"],
    aliases=["a_major/a_minor LCFS"],
)
def _a_major_over_a_minor(params, transforms, profiles, data, **kwargs):
    A = transforms["grid"].compress(data["A(z)"], surface_label="zeta")
    P = transforms["grid"].compress(data["perimeter(z)"], surface_label="zeta")
    data["a_major/a_minor"] = transforms["grid"].expand(
        _ramanujan(A, P), surface_label="zeta"
    )
    return data


@register_compute_fun(
    name="a_major/a_minor",
    label="a_{\\mathrm{major}} / a_{\\mathrm{minor}}",
    units="~",
    units_long="None",
    description="Elongation at a toroidal cross-section",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=["A(z)", "perimeter(z)"],
    parameterization=["desc.geometry.core.Surface"],
)
def _a_major_over_a_minor_surface(params, transforms, profiles, data, **kwargs):
    A = transforms["grid"].compress(data["A(z)"], surface_label="zeta")
    P = transforms["grid"].compress(data["perimeter(z)"], surface_label="zeta")
    data["a_major/a_minor"] = transforms["grid"].expand(
        _ramanujan(A, P), surface_label="zeta"
    )
    return data


@register_compute_fun(
    name="a_major/a_minor LCFS",
    label="a_{\\mathrm{major}} / a_{\\mathrm{minor}}",
    units="~",
    units_long="None",
    description="Elongation at a toroidal cross-section, "
    "extrapolated to last closed flux surface.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=["A(z)", "perimeter(z)", "rho"],
    parameterization=["desc.geometry.core.Surface"],
)
def _a_major_over_a_minor_surface_lcfs(params, transforms, profiles, data, **kwargs):
    max_rho = jnp.max(data["rho"])
    A = transforms["grid"].compress(data["A(z)"], surface_label="zeta") / max_rho**2
    P = (
        transforms["grid"].compress(data["perimeter(z)"], surface_label="zeta")
        / max_rho
    )
    data["a_major/a_minor LCFS"] = transforms["grid"].expand(
        _ramanujan(A, P), surface_label="zeta"
    )
    return data


@register_compute_fun(
    name="L_sff_rho",
    label="L_{\\mathrm{SFF},\\rho}",
    units="m",
    units_long="meters",
    description="L coefficient of second fundamental form of constant rho surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_rho", "e_theta_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _L_sff_rho(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["L_sff_rho"] = dot(data["e_theta_t"], data["n_rho"])
    return data


@register_compute_fun(
    name="M_sff_rho",
    label="M_{\\mathrm{SFF},\\rho}",
    units="m",
    units_long="meters",
    description="M coefficient of second fundamental form of constant rho surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_rho", "e_theta_z"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _M_sff_rho(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["M_sff_rho"] = dot(data["e_theta_z"], data["n_rho"])
    return data


@register_compute_fun(
    name="N_sff_rho",
    label="N_{\\mathrm{SFF},\\rho}",
    units="m",
    units_long="meters",
    description="N coefficient of second fundamental form of constant rho surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_rho", "e_zeta_z"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _N_sff_rho(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["N_sff_rho"] = dot(data["e_zeta_z"], data["n_rho"])
    return data


@register_compute_fun(
    name="curvature_k1_rho",
    label="k_{1,\\rho}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="First principle curvature of constant rho surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g_tt", "g_tz", "g_zz", "L_sff_rho", "M_sff_rho", "N_sff_rho"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _curvature_k1_rho(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    E = data["g_tt"]
    F = data["g_tz"]
    G = data["g_zz"]
    L = data["L_sff_rho"]
    M = data["M_sff_rho"]
    N = data["N_sff_rho"]
    a = E * G - F**2
    b = 2 * F * M - L * G - E * N
    c = L * N - M**2
    r1 = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    r2 = (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    # In the axis limit, the matrix of the first fundamental form is singular.
    # The diagonal of the shape operator becomes unbounded,
    # so the eigenvalues do not exist.
    data["curvature_k1_rho"] = jnp.maximum(r1, r2)
    data["curvature_k2_rho"] = jnp.minimum(r1, r2)
    return data


@register_compute_fun(
    name="curvature_k2_rho",
    label="k_{2,\\rho}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Second principle curvature of constant rho surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_rho"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _curvature_k2_rho(params, transforms, profiles, data, **kwargs):
    return data  # noqa: unused dependency


@register_compute_fun(
    name="curvature_K_rho",
    label="K_{\\rho}",
    units="m^2",
    units_long="meters squared",
    description="Gaussian curvature of constant rho surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_rho", "curvature_k2_rho"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _curvature_K_rho(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["curvature_K_rho"] = data["curvature_k1_rho"] * data["curvature_k2_rho"]
    return data


@register_compute_fun(
    name="curvature_H_rho",
    label="H_{\\rho}",
    units="m",
    units_long="meters",
    description="Mean curvature of constant rho surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_rho", "curvature_k2_rho"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _curvature_H_rho(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["curvature_H_rho"] = (data["curvature_k1_rho"] + data["curvature_k2_rho"]) / 2
    return data


@register_compute_fun(
    name="L_sff_theta",
    label="L_{\\mathrm{SFF},\\theta}",
    units="m",
    units_long="meters",
    description="L coefficient of second fundamental form of constant theta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_theta", "e_zeta_z"],
)
def _L_sff_theta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["L_sff_theta"] = dot(data["e_zeta_z"], data["n_theta"])
    return data


@register_compute_fun(
    name="M_sff_theta",
    label="M_{\\mathrm{SFF},\\theta}",
    units="m",
    units_long="meters",
    description="M coefficient of second fundamental form of constant theta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_theta", "e_zeta_r"],
)
def _M_sff_theta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["M_sff_theta"] = dot(data["e_zeta_r"], data["n_theta"])
    return data


@register_compute_fun(
    name="N_sff_theta",
    label="N_{\\mathrm{SFF},\\theta}",
    units="m",
    units_long="meters",
    description="N coefficient of second fundamental form of constant theta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_theta", "e_rho_r"],
)
def _N_sff_theta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["N_sff_theta"] = dot(data["e_rho_r"], data["n_theta"])
    return data


@register_compute_fun(
    name="curvature_k1_theta",
    label="k_{1,\\theta}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="First principle curvature of constant theta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g_rr", "g_rz", "g_zz", "L_sff_theta", "M_sff_theta", "N_sff_theta"],
)
def _curvature_k1_theta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    E = data["g_zz"]
    F = data["g_rz"]
    G = data["g_rr"]
    L = data["L_sff_theta"]
    M = data["M_sff_theta"]
    N = data["N_sff_theta"]
    a = E * G - F**2
    b = 2 * F * M - L * G - E * N
    c = L * N - M**2
    r1 = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    r2 = (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    data["curvature_k1_theta"] = jnp.maximum(r1, r2)
    data["curvature_k2_theta"] = jnp.minimum(r1, r2)
    return data


@register_compute_fun(
    name="curvature_k2_theta",
    label="k_{2,\\theta}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Second principle curvature of constant theta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_theta"],
)
def _curvature_k2_theta(params, transforms, profiles, data, **kwargs):
    return data  # noqa: unused dependency


@register_compute_fun(
    name="curvature_K_theta",
    label="K_{\\theta}",
    units="m^2",
    units_long="meters squared",
    description="Gaussian curvature of constant theta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_theta", "curvature_k2_theta"],
)
def _curvature_K_theta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["curvature_K_theta"] = data["curvature_k1_theta"] * data["curvature_k2_theta"]
    return data


@register_compute_fun(
    name="curvature_H_theta",
    label="H_{\\theta}",
    units="m",
    units_long="meters",
    description="Mean curvature of constant theta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_theta", "curvature_k2_theta"],
)
def _curvature_H_theta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["curvature_H_theta"] = (
        data["curvature_k1_theta"] + data["curvature_k2_theta"]
    ) / 2
    return data


@register_compute_fun(
    name="L_sff_zeta",
    label="L_{\\mathrm{SFF},\\zeta}",
    units="m",
    units_long="meters",
    description="L coefficient of second fundamental form of constant zeta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_zeta", "e_rho_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _L_sff_zeta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["L_sff_zeta"] = dot(data["e_rho_r"], data["n_zeta"])
    return data


@register_compute_fun(
    name="M_sff_zeta",
    label="M_{\\mathrm{SFF},\\zeta}",
    units="m",
    units_long="meters",
    description="M coefficient of second fundamental form of constant zeta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_zeta", "e_rho_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _M_sff_zeta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["M_sff_zeta"] = dot(data["e_rho_t"], data["n_zeta"])
    return data


@register_compute_fun(
    name="N_sff_zeta",
    label="N_{\\mathrm{SFF},\\zeta}",
    units="m",
    units_long="meters",
    description="N coefficient of second fundamental form of constant zeta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_zeta", "e_theta_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _N_sff_zeta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["N_sff_zeta"] = dot(data["e_theta_t"], data["n_zeta"])
    return data


@register_compute_fun(
    name="curvature_k1_zeta",
    label="k_{1,\\zeta}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="First principle curvature of constant zeta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g_rr", "g_rt", "g_tt", "L_sff_zeta", "M_sff_zeta", "N_sff_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _curvature_k1_zeta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    E = data["g_rr"]
    F = data["g_rt"]
    G = data["g_tt"]
    L = data["L_sff_zeta"]
    M = data["M_sff_zeta"]
    N = data["N_sff_zeta"]
    a = E * G - F**2
    b = 2 * F * M - L * G - E * N
    c = L * N - M**2
    r1 = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    r2 = (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    # In the axis limit, the matrix of the first fundamental form is singular.
    # The diagonal of the shape operator becomes unbounded,
    # so the eigenvalues do not exist.
    data["curvature_k1_zeta"] = jnp.maximum(r1, r2)
    data["curvature_k2_zeta"] = jnp.minimum(r1, r2)
    return data


@register_compute_fun(
    name="curvature_k2_zeta",
    label="k_{2,\\zeta}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Second principle curvature of constant zeta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _curvature_k2_zeta(params, transforms, profiles, data, **kwargs):
    return data  # noqa: unused dependency


@register_compute_fun(
    name="curvature_K_zeta",
    label="K_{\\zeta}",
    units="m^2",
    units_long="meters squared",
    description="Gaussian curvature of constant zeta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_zeta", "curvature_k2_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _curvature_K_zeta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["curvature_K_zeta"] = data["curvature_k1_zeta"] * data["curvature_k2_zeta"]
    return data


@register_compute_fun(
    name="curvature_H_zeta",
    label="H_{\\zeta}",
    units="m",
    units_long="meters",
    description="Mean curvature of constant zeta surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1_zeta", "curvature_k2_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _curvature_H_zeta(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface
    data["curvature_H_zeta"] = (
        data["curvature_k1_zeta"] + data["curvature_k2_zeta"]
    ) / 2
    return data


@register_compute_fun(
    name="fieldline length",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}}"
    " \\frac{d\\zeta}{|B^{\\zeta}|}",
    units="m / T",
    units_long="Meter / tesla",
    description="(Mean) proper length of field line(s)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta"],
    resolution_requirement="z",
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _fieldline_length(data, transforms, profiles, **kwargs):
    grid = transforms["grid"].source_grid
    data["fieldline length"] = grid.expand(
        jnp.abs(
            simpson(
                y=grid.meshgrid_reshape(1 / data["B^zeta"], "raz"),
                x=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
                axis=-1,
            ).mean(axis=-1)
        )
    )
    return data


@register_compute_fun(
    name="fieldline length/volume",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}}"
    " \\frac{d\\zeta}{|B^{\\zeta} \\sqrt g|}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="(Mean) proper length over volume of field line(s)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta", "sqrt(g)"],
    resolution_requirement="z",
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _fieldline_length_over_volume(data, transforms, profiles, **kwargs):
    grid = transforms["grid"].source_grid
    data["fieldline length/volume"] = grid.expand(
        jnp.abs(
            simpson(
                y=grid.meshgrid_reshape(1 / (data["B^zeta"] * data["sqrt(g)"]), "raz"),
                x=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
                axis=-1,
            ).mean(axis=-1)
        )
    )
    return data
