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
    # ψ_b = Psi/(2π) is the boundary toroidal flux
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
    # B_ref = 2|ψ_b|/a² where ψ_b = Psi/(2π)
    psi_b = jnp.abs(params["Psi"]) / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    data["gx_bmag"] = data["|B|"] / B_ref
    return data
