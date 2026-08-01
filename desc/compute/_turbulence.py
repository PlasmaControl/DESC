"""Compute functions for turbulent transport.

References
----------
.. [1] J. H. E. Proll et al., "TEM turbulence optimisation in stellarators,"
       Plasma Phys. Control. Fusion 58, 014006 (2016).
       https://doi.org/10.1088/0741-3335/58/1/014006.
.. [2] R. J. J. Mackenbach et al., J. Plasma Phys. 89, 905890513 (2023).
.. [3] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
       bounce-averaging algorithm and its applications,"
       J. Plasma Physics. https://doi:10.1017/S0022377826101652.
.. [4] R. J. J. Mackenbach, P. Helander, M. Landreman, S. Brunner, and
       J. H. E. Proll, "On the curvature-driven ion-temperature-gradient
       instability and its available energy," J. Plasma Phys. 91, E144 (2025).
       https://doi.org/10.1017/S0022377825100846.
.. [5] E. Rodríguez and R. J. J. Mackenbach, "Trapped-particle precession and
       modes in quasisymmetric stellarators and tokamaks: a near-axis perspective,"
       J. Plasma Phys. 89, 905890521 (2023).

"""

from functools import partial

import numpy as np
from jax.lax import stop_gradient
from orthax import orthgauss
from orthax.recurrence import GeneralizedLaguerre

from desc.backend import jit, jnp

from ..integrals.bounce_integral import Bounce2D, Options
from ..utils import safediv
from ._drift import (
    _energy_normalized_binormal_drift,
    _energy_normalized_radial_drift,
    _sqrt_G_hat,
)
from .data_index import register_compute_fun


def _ae(G, G_ω_α, G_ω_ψ, data, energy):
    shape = (-1,) + (1,) * G.ndim

    G = G[..., None, :]  # This is sqrt G hat.
    # scale by conjugate widths
    G_ω_α = G_ω_α[..., None, :] * data["ae psi width"].reshape(shape)
    G_ω_ψ = G_ω_ψ[..., None, :] * data["ae alpha width"].reshape(shape)
    η_n = data["ae grad(density)"].reshape(shape)
    η_T = data["ae grad(temperature)"].reshape(shape)
    C = η_n - 1.5 * η_T
    energy = energy[..., None]

    drift = jnp.hypot(G_ω_α, G_ω_ψ)
    drive = jnp.hypot(G * (η_T + safediv(C, energy)) - G_ω_α, G_ω_ψ)

    return G_ω_α * C + (G_ω_α * η_T + safediv(drift * (drive - drift), G)) * energy


def _energy_quad(deg):
    # The energy integral has weight E^(5/2) exp(-E), but
    # ω_* = η_T + C / E makes AE(E) ~ C/E for E near zero.
    return stop_gradient(orthgauss(deg, GeneralizedLaguerre(np.array([1.5]))))


@register_compute_fun(
    name="available energy",
    label="\\widehat{A}",
    units="~",
    units_long="None",
    description="Dimensionless available energy of trapped electrons",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "psi_r",
        "rho",
        "ne",
        "ne_r",
        "Te",
        "Te_r",
        "cvdrift (periodic)",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "|grad(psi)|*kappa_g",
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    radial_scale=(
        "float : Radial correlation-length coefficient Cᵣ in Δr_A = Cᵣρₗ. "
        "After factoring out ρ★² in the definition of Â, this scales both "
        "Δψ_A/ρ★ = Cᵣ ∂ψ/∂ρ and the radial profile gradients."
    ),
    binormal_scale=(
        "float : Binormal correlation-length coefficient Cₛ in Δs_A = Cₛρₗ."
    ),
    fieldline_normalization=(
        "float or ndarray : Field-line factor 𝒩ₗ = Vψ/(2π𝓛), where "
        "𝓛 is the sum of ∫dℓ/B over the retained complete field-line domain. "
        "The default NFP/num_field_periods is the long-field-line estimate. "
        "For k complete axisymmetric poloidal transits, pass |ι|/k."
    ),
    energy_quad="tuple : Optional nodes and weights for fixed energy quadrature.",
    **Options._doc,
)
@partial(
    jit,
    static_argnames=Options._static_argnames,
)
def _available_energy(params, transforms, profiles, data, **kwargs):
    """Dimensionless available energy of trapped electrons [2]_.

    Parameters
    ----------
    radial_scale : float
        Radial correlation-length coefficient Cᵣ in Δr_A = Cᵣρₗ.
        After factoring out ρ★² in the definition of Â, this scales both
        Δψ_A/ρ★ = Cᵣ ∂ψ/∂ρ and the radial profile gradients.
        Default is 1.0.
    binormal_scale : float
        Binormal correlation-length coefficient Cₛ in Δs_A = Cₛρₗ.
        Default is 1.0.
    fieldline_normalization : float or ndarray, optional
        Field-line factor 𝒩ₗ = Vψ/(2π𝓛), where 𝓛 is the sum of ∫dℓ/B over
        the retained complete field-line domain. The default
        ``NFP / num_field_periods`` is the long-field-line estimate. For k
        complete axisymmetric poloidal transits, pass |ι|/k.

    Notes
    -----
    Let ρ★ = ρₗ/a and r = aρ. Equations (2.47) and (2.49) of [2]_
    define Δr_A = Cᵣρₗ and factor ρ★² out of the available energy.
    Consequently, the widths used here are
    Δψ_A/ρ★ = Cᵣ ∂ψ/∂ρ and Δα_A/ρ★ = Cₛ/ρ. The parameters
    ``radial_scale`` and ``binormal_scale`` are therefore Cᵣ and Cₛ,
    respectively, rather than the normalized coordinate width Δρ_A = Cᵣρ★.

    DESC uses ψ = Ψρ²/(2π) = ψₑρ², so ∂ψ/∂ρ = 2ψₑρ. Therefore,
    Δψ_A/ρ★ already contains the factor of ρ in Eq. (4.7) of [5]_.

    Before energy normalization, the bounce-integral ratios satisfy
    G_ω/G = qω/(mv²). Equations (2.35) and (2.38) of [2]_ instead use
    qω/ε₀ with ε₀ = mv²/2. The AE-specific drift integrands apply this factor
    of two before bounce integration.

    All complete wells in the traced interval are summed, as in Eq. (2.45) of [2]_.
    The compute function does not infer a special axisymmetric domain. To use k
    complete poloidal transits between global maxima of |B|, choose ``alpha`` and
    ``num_field_periods`` so the traced interval contains those transits and pass
    ``fieldline_normalization=|ι|/k``.

    The result uses the 3nT/2 thermal-energy normalization in Eqs. (2.44) and
    (2.49) of [2]_. It is therefore ⅔ of an otherwise identical convention
    normalized by nT, such as Eq. (4.2) of [5]_.

    """
    # noqa: unused dependency
    radial_scale = kwargs.get("radial_scale", 1.0)
    binormal_scale = kwargs.get("binormal_scale", 1.0)
    fieldline_normalization = kwargs.get("fieldline_normalization", None)
    energy_quad = kwargs.get("energy_quad", None)
    if energy_quad is None:
        energy_quad = _energy_quad(32)

    grid = transforms["grid"]
    opts = Options.guess(-1, grid, **kwargs)

    def foreach_surface(data):
        pitch_inv, weight = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        weight /= pitch_inv**2
        ae_data = Bounce2D(grid, data, data["angle"], **opts).integrate(
            [
                _sqrt_G_hat,
                _energy_normalized_binormal_drift,
                _energy_normalized_radial_drift,
            ],
            pitch_inv,
            data,
            names,
            num_well=opts.num_well,
            loop=opts.loop,
        )

        return jnp.sum(
            _ae(*ae_data, data, energy_quad[0]).sum(-1).mean(-3).dot(energy_quad[1])
            * weight,
            axis=-1,
        )

    names = (
        "cvdrift (periodic)",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "|grad(psi)|*kappa_g",
    )
    out = Bounce2D.batch(
        foreach_surface,
        data,
        grid,
        angle=kwargs["angle"],
        names=names,
        flux_data={
            "ae grad(density)": safediv(radial_scale * data["ne_r"], data["ne"]),
            # After factoring out ρ★, Δψ_A/ρ★ = Cᵣ∂ψ/∂ρ.
            # Since ψ = ψₑρ², ∂ψ/∂ρ already contains the surface-label ρ.
            "ae psi width": radial_scale * data["psi_r"],
            "ae alpha width": safediv(binormal_scale, data["rho"]),
            "ae grad(temperature)": safediv(radial_scale * data["Te_r"], data["Te"]),
        },
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1

    if fieldline_normalization is None:
        # Long-field-line limit: 𝓛 → num_field_periods Vψ/(2π NFP).
        fieldline_normalization = grid.NFP / opts.num_field_periods
    scalar = jnp.sqrt(jnp.pi) * jnp.asarray(fieldline_normalization) / 3
    data["available energy"] = grid.expand(scalar * out) / data["V_psi"]
    return data
