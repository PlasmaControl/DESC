from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross, dot, line_integrals, surface_integrals


@register_compute_fun(
    name="V",
    label="V",
    units="m^{3}",
    units_long="cubic meters",
    description="Volume",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["sqrt(g)"],
)
def _V(params, transforms, profiles, data, **kwargs):
    data["V"] = jnp.sum(data["sqrt(g)"] * transforms["grid"].weights)
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
    data=["sqrt(g)_r", "sqrt(g)"],
)
def _V_rr_of_r(params, transforms, profiles, data, **kwargs):
    data["V_rr(r)"] = surface_integrals(
        transforms["grid"], data["sqrt(g)_r"] * jnp.sign(data["sqrt(g)"])
    )
    return data


@register_compute_fun(
    name="A",
    label="A",
    units="m^{2}",
    units_long="square meters",
    description="Average cross-sectional area",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["sqrt(g)", "R"],
)
def _A(params, transforms, profiles, data, **kwargs):
    data["A"] = jnp.mean(
        surface_integrals(
            transforms["grid"],
            jnp.abs(data["sqrt(g)"] / data["R"]),
            surface_label="zeta",
            expand_out=False,
        )
    )
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
)
def _S_of_r(params, transforms, profiles, data, **kwargs):
    data["S(r)"] = surface_integrals(transforms["grid"], data["|e_theta x e_zeta|"])
    return data


@register_compute_fun(
    name="R0",
    label="R_{0}",
    units="m",
    units_long="meters",
    description="Average major radius",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["V", "A"],
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
)
def _R0_over_a(params, transforms, profiles, data, **kwargs):
    data["R0/a"] = data["R0"] / data["a"]
    return data


@register_compute_fun(
    name="a_major/a_minor",
    label="a_{major} / a_{minor}",
    units="~",
    units_long="None",
    description="Maximum elongation",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["sqrt(g)", "g_tt"],
)
def _a_major_over_a_minor(params, transforms, profiles, data, **kwargs):
    max_rho = transforms["grid"].nodes[transforms["grid"].unique_rho_idx[-1], 0]
    P = (  # perimeter
        line_integrals(
            transforms["grid"],
            jnp.sqrt(data["g_tt"]),
            line_label="theta",
            fix_surface=("rho", max_rho),
            expand_out=False,
        )
        / max_rho
    )
    # surface area
    A = surface_integrals(
        transforms["grid"],
        jnp.abs(data["sqrt(g)"] / data["R"]),
        surface_label="zeta",
        expand_out=False,
    )
    # derived from Ramanujan approximation for the perimeter of an ellipse
    a = (  # semi-major radius
        jnp.sqrt(3)
        * (
            jnp.sqrt(8 * jnp.pi * A + P**2)
            + jnp.sqrt(
                2 * jnp.sqrt(3) * P * jnp.sqrt(8 * jnp.pi * A + P**2)
                - 40 * jnp.pi * A
                + 4 * P**2
            )
        )
        + 3 * P
    ) / (12 * jnp.pi)
    b = A / (jnp.pi * a)  # semi-minor radius
    data["a_major/a_minor"] = jnp.max(a / b)
    return data


@register_compute_fun(
    name="L_sff",
    label="L_{sff}",
    units="m",
    units_long="meters",
    description="L coefficient of second fundamental form",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_rho", "e_theta_t"],
)
def _L_sff(params, transforms, profiles, data, **kwargs):
    data["L_sff"] = dot(data["e_theta_t"], data["n_rho"])
    return data


@register_compute_fun(
    name="M_sff",
    label="M_{sff}",
    units="m",
    units_long="meters",
    description="M coefficient of second fundamental form",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_rho", "e_theta_z"],
)
def _M_sff(params, transforms, profiles, data, **kwargs):
    data["M_sff"] = dot(data["e_theta_z"], data["n_rho"])
    return data


@register_compute_fun(
    name="N_sff",
    label="N_{sff}",
    units="m",
    units_long="meters",
    description="N coefficient of second fundamental form",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["n_rho", "e_zeta_z"],
)
def _N_sff(params, transforms, profiles, data, **kwargs):
    data["N_sff"] = dot(data["e_zeta_z"], data["n_rho"])
    return data


@register_compute_fun(
    name="curvature_k1",
    label="k_{1}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="First principle curvature of flux surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g_tt", "g_tz", "g_zz", "L_sff", "M_sff", "N_sff"],
)
def _curvature_k1(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface#Curvature
    E = data["g_tt"]
    F = data["g_tz"]
    G = data["g_zz"]
    L = data["L_sff"]
    M = data["M_sff"]
    N = data["N_sff"]
    a = E * G - F**2
    b = F * M - L * G - E * N
    c = L * N - M**2
    r1 = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    r2 = (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    data["curvature_k1"] = jnp.maximum(r1, r2)
    data["curvature_k2"] = jnp.minimum(r1, r2)
    return data


@register_compute_fun(
    name="curvature_k2",
    label="k_{2}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Second principle curvature of flux surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g_tt", "g_tz", "g_zz", "L_sff", "M_sff", "N_sff"],
)
def _curvature_k2(params, transforms, profiles, data, **kwargs):
    # following notation from
    # https://en.wikipedia.org/wiki/Parametric_surface#Curvature
    E = data["g_tt"]
    F = data["g_tz"]
    G = data["g_zz"]
    L = data["L_sff"]
    M = data["M_sff"]
    N = data["N_sff"]
    a = E * G - F**2
    b = F * M - L * G - E * N
    c = L * N - M**2
    r1 = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    r2 = (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
    data["curvature_k1"] = jnp.maximum(r1, r2)
    data["curvature_k2"] = jnp.minimum(r1, r2)
    return data


@register_compute_fun(
    name="curvature_K",
    label="K",
    units="m^2",
    units_long="meters squared",
    description="Gaussian curvature of flux surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1", "curvature_k2"],
)
def _curvature_K(params, transforms, profiles, data, **kwargs):
    data["curvature_K"] = data["curvature_k1"] * data["curvature_k2"]
    return data


@register_compute_fun(
    name="curvature_H",
    label="H",
    units="m",
    units_long="meters",
    description="Mean curvature of flux surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curvature_k1", "curvature_k2"],
)
def _curvature_H(params, transforms, profiles, data, **kwargs):
    data["curvature_H"] = (data["curvature_k1"] + data["curvature_k2"]) / 2
    return data
