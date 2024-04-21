"""Compute functions for stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from orthax import legendre
from scipy.constants import mu_0

from desc.backend import jnp, trapezoid

from .bounce_integral import (
    affine_bijection_reverse,
    bounce_integral_map,
    grad_affine_bijection_reverse,
    pitch_of_extrema,
    pitch_trapz,
)
from .data_index import register_compute_fun
from .utils import dot, surface_integrals_map


@register_compute_fun(
    name="D_shear",
    label="D_{\\mathrm{shear}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic shear term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["iota_psi"],
)
def _D_shear(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.16 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_shear"] = data["iota_psi"] ** 2 / (16 * jnp.pi**2)
    return data


@register_compute_fun(
    name="D_current",
    label="D_{\\mathrm{current}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion toroidal current term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "psi_r",
        "iota_psi",
        "B",
        "J",
        "G",
        "I_r",
        "|grad(psi)|",
        "|e_theta x e_zeta|",
    ],
)
def _D_current(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.17 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    Xi = mu_0 * data["J"] - (data["I_r"] / data["psi_r"] * data["B"].T).T
    integrate = surface_integrals_map(transforms["grid"])
    data["D_current"] = (
        -jnp.sign(data["G"])
        / (2 * jnp.pi) ** 4
        * data["iota_psi"]
        * transforms["grid"].replace_at_axis(
            integrate(
                data["|e_theta x e_zeta|"]
                / data["|grad(psi)|"] ** 3
                * dot(Xi, data["B"])
            ),
            # Todo: implement equivalent of equation 4.3 in desc coordinates
            jnp.nan,
        )
    )
    return data


@register_compute_fun(
    name="D_well",
    label="D_{\\mathrm{well}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic well term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "p_r",
        "psi",
        "psi_r",
        "psi_rr",
        "V_rr(r)",
        "V_r(r)",
        "|B|^2",
        "|grad(psi)|",
        "|e_theta x e_zeta|",
    ],
)
def _D_well(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.18 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    integrate = surface_integrals_map(transforms["grid"])
    dp_dpsi = mu_0 * data["p_r"] / data["psi_r"]
    d2V_dpsi2 = (
        data["V_rr(r)"] * data["psi_r"] - data["V_r(r)"] * data["psi_rr"]
    ) / data["psi_r"] ** 3
    data["D_well"] = (
        dp_dpsi
        * (
            jnp.sign(data["psi"]) * d2V_dpsi2
            - dp_dpsi
            * integrate(
                data["|e_theta x e_zeta|"] / (data["|B|^2"] * data["|grad(psi)|"])
            )
        )
        * integrate(
            data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3
        )
        / (2 * jnp.pi) ** 6
    )
    # Axis limit does not exist as ∂ᵨ ψ and ‖∇ ψ‖ terms dominate so that D_well
    # is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="D_geodesic",
    label="D_{\\mathrm{geodesic}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion geodesic curvature term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|grad(psi)|", "J*B", "|B|^2", "|e_theta x e_zeta|"],
)
def _D_geodesic(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.19 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    integrate = surface_integrals_map(transforms["grid"])
    data["D_geodesic"] = transforms["grid"].replace_at_axis(
        (
            integrate(
                data["|e_theta x e_zeta|"]
                * mu_0
                * data["J*B"]
                / data["|grad(psi)|"] ** 3
            )
            ** 2
            - integrate(
                data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3
            )
            * integrate(
                data["|e_theta x e_zeta|"]
                * mu_0**2
                * data["J*B"] ** 2
                / (data["|B|^2"] * data["|grad(psi)|"] ** 3),
            )
        )
        / (2 * jnp.pi) ** 6,
        jnp.nan,  # enforce manually because our integration replaces nan with 0
    )
    # Axis limit does not exist as ‖∇ ψ‖ terms dominate so that D_geodesic
    # is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="D_Mercier",
    label="D_{\\mathrm{Mercier}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["D_shear", "D_current", "D_well", "D_geodesic"],
)
def _D_Mercier(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.20 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_Mercier"] = (
        data["D_shear"] + data["D_current"] + data["D_well"] + data["D_geodesic"]
    )
    # The axis limit does not exist as D_Mercier is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="magnetic well",
    label="\\mathrm{Magnetic~Well}",
    units="~",
    units_long="None",
    description="Magnetic well proxy for MHD stability (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["V(r)", "V_r(r)", "p_r", "<|B|^2>", "<|B|^2>_r"],
)
def _magnetic_well(params, transforms, profiles, data, **kwargs):
    # Implements equation 3.2 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    # pressure = thermal + magnetic = 2 mu_0 p + |B|^2
    # The surface average operation is an additive homomorphism.
    # Thermal pressure is constant over a rho surface.
    # surface average(pressure) = thermal + surface average(magnetic)
    # The sign of sqrt(g) is enforced to be non-negative.
    data["magnetic well"] = transforms["grid"].replace_at_axis(
        data["V(r)"]
        * (2 * mu_0 * data["p_r"] + data["<|B|^2>_r"])
        / (data["V_r(r)"] * data["<|B|^2>"]),
        0,  # coefficient of limit is V_r / V_rr, rest is finite
    )
    return data


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\text{eff}}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["rho", "|grad(psi)|", "kappa_g", "sqrt(g)", "psi_r", "S(r)", "R"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    # V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    # Evaluation of 1/ν neoclassical transport in stellarators.
    # Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    # https://doi.org/10.1063/1.873749.
    raise NotImplementedError("Effective ripple requires GitHub PR #854.")

    kwargs.setdefault(
        "knots",
        jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, transforms["grid"].num_zeta),
    )
    alpha_max = (2 - transforms["grid"].sym) * jnp.pi
    alpha, alpha_weight = legendre.leggauss(
        kwargs.get("num_alpha", transforms["grid"].num_theta)
    )
    alpha = affine_bijection_reverse(alpha, 0, alpha_max)
    alpha_weight = alpha_weight * grad_affine_bijection_reverse(0, alpha_max)
    rho = transforms["grid"].compress(data["rho"])
    bounce_integral, items = bounce_integral_map(rho, alpha, kwargs["knots"])

    def integrand_H(grad_psi_norm, kappa_g, B, pitch, Z):
        return (
            pitch
            * jnp.sqrt(1 / pitch - B)
            * (4 / B - pitch)
            * grad_psi_norm
            * kappa_g
            / B
        )

    def integrand_I(B, pitch, Z):
        return jnp.sqrt(1 - pitch * B) / B

    pitch = pitch_of_extrema(kwargs["knots"], items["B.c"], items["B_z_ra.c"])
    pitch = pitch_trapz(pitch, kwargs.get("pitch quad resolution", 20))
    H = bounce_integral(integrand_H, [data["|grad(psi)|"], data["kappa_g"]], pitch)
    I = bounce_integral(integrand_I, [], pitch)
    H = H.reshape(H.shape[0], rho.size, alpha.size, -1)
    I = I.reshape(I.shape[0], rho.size, alpha.size, -1)
    pitch = pitch.reshape(-1, rho.size, alpha.size)

    # E = ∑ⱼ Hⱼ² / Iⱼ except the sum over j is split across the axis enumerating alpha.
    E = jnp.nansum(H**2 / I, axis=-1)
    # E = ∫ db ∑ⱼ Hⱼ² / Iⱼ
    E = jnp.dot(trapezoid(E, 1 / pitch, axis=0), alpha_weight)
    surface_integrate = surface_integrals_map(transforms["grid"], expand_out=False)
    s1 = surface_integrate(data["sqrt(g)"] / data["psi_r"])
    s2 = transforms["grid"].compress(data["S(r)"])

    epsilon = (jnp.pi * data["R"] ** 2 / (8 * 2**0.5) * s1 / s2**2 * E) ** (2 / 3)
    data["effective ripple"] = transforms["grid"].expand(epsilon)
    return data
