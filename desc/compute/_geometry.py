from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross, surface_integrals


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
    data=["sqrt(g)"],
)
def _V(params, transforms, profiles, data, **kwargs):
    data["V"] = jnp.sum(jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights)
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
    data=["sqrt(g)"],
)
def _V_r_of_r(params, transforms, profiles, data, **kwargs):
    # eq. 4.9.10 in W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.
    data["V_r(r)"] = surface_integrals(transforms["grid"], jnp.abs(data["sqrt(g)"]))
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
    data=["sqrt(g)", "R"],
)
def _A(params, transforms, profiles, data, **kwargs):
    data["A"] = jnp.mean(
        surface_integrals(
            transforms["grid"],
            jnp.abs(data["sqrt(g)"] / data["R"]),
            surface_label="zeta",
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
    data=["R0", "a"],
)
def _R0_over_a(params, transforms, profiles, data, **kwargs):
    data["R0/a"] = data["R0"] / data["a"]
    return data
