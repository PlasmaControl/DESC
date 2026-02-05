"""Compute functions for ITG turbulence proxies and related quantities.

This module provides GX geometric coefficients and ITG turbulence proxies
for gyrokinetic turbulence prediction. The implementation follows the
conventions established in Landreman et al. 2025 (arXiv:2502.11657).

Notes
-----
Reference quantities use the following normalizations:
- B_reference = 2|ψ_b|/a² where ψ_b is the boundary toroidal flux
- L_reference = a (minor radius)

Some quantities may have singularities at the magnetic axis (ρ=0).
Objectives using these quantities should use ρ > 0.
"""

from desc.backend import jnp

from ..utils import dot
from .data_index import register_compute_fun


# =============================================================================
# Reference Quantities
# =============================================================================


@register_compute_fun(
    name="gx_B_reference",
    label="B_{\\mathrm{ref}}",
    units="T",
    units_long="Tesla",
    description="GX reference magnetic field: B_ref = 2|ψ_b|/a²",
    dim=0,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="",
    data=["a"],
)
def _gx_B_reference(params, transforms, profiles, data, **kwargs):
    psi_b = jnp.abs(params["Psi"]) / (2 * jnp.pi)
    data["gx_B_reference"] = 2 * psi_b / data["a"] ** 2
    return data


@register_compute_fun(
    name="gx_L_reference",
    label="L_{\\mathrm{ref}}",
    units="m",
    units_long="meters",
    description="GX reference length: L_ref = a (minor radius)",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["a"],
)
def _gx_L_reference(params, transforms, profiles, data, **kwargs):
    data["gx_L_reference"] = data["a"]
    return data


# =============================================================================
# Normalized Magnetic Field
# =============================================================================


@register_compute_fun(
    name="gx_bmag",
    label="|\\mathbf{B}|/B_{\\mathrm{ref}}",
    units="~",
    units_long="dimensionless",
    description="Normalized magnetic field magnitude |B|/B_ref for GX",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "a"],
)
def _gx_bmag(params, transforms, profiles, data, **kwargs):
    psi_b = jnp.abs(params["Psi"]) / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    data["gx_bmag"] = data["|B|"] / B_ref
    return data


# =============================================================================
# Gradient Coefficients
# =============================================================================


@register_compute_fun(
    name="gx_gds2",
    label="|\\nabla \\alpha|^2 L_{\\mathrm{ref}}^2 s",
    units="~",
    units_long="dimensionless",
    description="GX gds2: |grad(alpha)|^2 * L_ref^2 * s, where s = rho^2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "rho", "grad(alpha)"],
)
def _gx_gds2(params, transforms, profiles, data, **kwargs):
    L_ref = data["a"]
    s = data["rho"] ** 2
    grad_alpha_sq = dot(data["grad(alpha)"], data["grad(alpha)"])
    data["gx_gds2"] = grad_alpha_sq * L_ref**2 * s
    return data


@register_compute_fun(
    name="gx_gds21_over_shat",
    label="\\mathrm{gds21} / \\hat{s}",
    units="~",
    units_long="dimensionless",
    description="GX gds21/shat: grad(alpha).grad(psi) * sigma_Bxy / B_ref",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "grad(alpha)", "grad(psi)"],
)
def _gx_gds21_over_shat(params, transforms, profiles, data, **kwargs):
    sigma_Bxy = -1  # GX sign convention
    psi_b = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    grad_alpha_dot_grad_psi = dot(data["grad(alpha)"], data["grad(psi)"])
    data["gx_gds21_over_shat"] = sigma_Bxy * grad_alpha_dot_grad_psi / B_ref
    return data


@register_compute_fun(
    name="gx_gds22_over_shat_squared",
    label="\\mathrm{gds22} / \\hat{s}^2",
    units="~",
    units_long="dimensionless",
    description="GX gds22/shat^2: |grad(psi)|^2 / (L_ref^2 * B_ref^2 * s)",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "rho", "|grad(psi)|^2"],
)
def _gx_gds22_over_shat_squared(params, transforms, profiles, data, **kwargs):
    psi_b = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    L_ref = data["a"]
    s = data["rho"] ** 2
    data["gx_gds22_over_shat_squared"] = (
        data["|grad(psi)|^2"] / (L_ref**2 * B_ref**2 * s)
    )
    return data
