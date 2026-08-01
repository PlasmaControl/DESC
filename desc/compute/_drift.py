"""Geometric parts of particle drifts.

Orbit models remove the secular part of the drifts.
Flux tube models retain the secular part of the drifts.

When the magnetic shear is nonzero, flux tube models are
not well-defined until a ballooning convention is made.
We choose the ballooning angle as ζ₀ = 0.

Flux tube models have meaningless ergodic limits; and therefore,
require specifying a fixed length scale window apriori.
For tokamaks, the only justified choice is a single poloidal transit
between two global maxima of the potential.

"""

from desc.backend import jnp
from desc.utils import identity, safediv


def register_drift(**_):
    """Decorator for labeling functions with metadata."""
    return identity


@register_drift(
    name="binormal drift",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Local binormal drift for flux tube model.",
)
def _binormal_drift(data, B, pitch):
    # TODO (#465), multiply by (omega + zeta) instead of zeta
    gbdrift_secular = data["gbdrift (secular)/phi"] * data["zeta"]
    cvdrift = data["cvdrift (periodic)"] + gbdrift_secular
    gbdrift = data["gbdrift (periodic)"] + gbdrift_secular
    g = jnp.sqrt(jnp.abs(1 - pitch * B))
    return (cvdrift - 0.5 * gbdrift) * g + safediv(0.5 * gbdrift, g)


@register_drift(
    name="sqrt(G hat)",
    units="~",
    units_long="None",
)
def _sqrt_G_hat(data, B, pitch):
    return safediv(1.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


@register_drift(
    name="v tau",
    units="~",
    units_long="None",
    description="Local bounce time.",
)
def _v_tau(data, B, pitch):
    # v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
    # τ is the bounce time, and I is defined in Nemov et al. eq. 36.
    return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


@register_drift(
    name="radial drift",
    units="~",
    units_long="None",
    description="Local radial drift.",
)
def _radial_drift(data, B, pitch):
    return (
        safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
        * data["|grad(psi)|*kappa_g"]
        / B
    )


@register_drift(
    name="radial drift",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Local radial drift.",
)
def _radial_drift_wb_inverse(data, B, pitch):
    return safediv(
        data["cvdrift0"] * (1 - 0.5 * pitch * B),
        jnp.sqrt(jnp.abs(1 - pitch * B)),
    )


@register_drift(
    name="vartheta drift",
    units="~",
    units_long="None",
    description="Poloidal drift for trapped particle orbit model.",
)
def _vartheta_drift(data, B, pitch):
    g = jnp.sqrt(jnp.abs(1 - pitch * B))
    return (safediv(1 - 0.5 * pitch * B, g) * data["|B|_r|v,p"] + g * data["K"]) / B


@register_drift(
    name="alpha drift",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Poloidal drift for flux tube model.",
)
def _alpha_drift_wb_inverse(data, B, pitch):
    # TODO (#465), multiply by (omega + zeta) instead of zeta
    return safediv(
        (data["gbdrift (periodic)"] + data["gbdrift (secular)/phi"] * data["zeta"])
        * (1 - 0.5 * pitch * B),
        jnp.sqrt(jnp.abs(1 - pitch * B)),
    )


@register_drift(
    name="I_1",
    units="m^{-2} T^{-1}",
    units_long="Inverse square meters per tesla",
    description="Integrand of equation 2.9 in [2]_  in neoclassical file "
    "with |∂ψ/∂ρ| removed.",
)
def _I_1(data, B, pitch):
    return (
        jnp.sqrt(jnp.abs(1 - pitch * B))
        * (4 / (pitch * B) - 1)
        * data["|grad(rho)|*kappa_g"]
        / B
    )


@register_drift(
    name="I_2",
    units="T^{-1}",
    units_long="Inverse tesla",
    description="Integrand of equation 2.10 in [2]_ in neoclassical file.",
)
def _I_2(data, B, pitch):
    return jnp.sqrt(jnp.abs(1 - pitch * B)) / B
