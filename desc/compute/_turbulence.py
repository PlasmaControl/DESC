"""Compute functions for turbulent transport."""

from functools import partial

import numpy as np
from jax.lax import stop_gradient
from orthax import orthgauss
from orthax.recurrence import GeneralizedLaguerre

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D, Options
from ..utils import safediv
from ._fast_ion import _radial_drift
from .data_index import register_compute_fun


def _G_hat_half(data, B, pitch):
    return safediv(1.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


def _binormal_drift_wb_inverse(data, B, pitch):
    # TODO (#465), multiply by (omega + zeta) instead of zeta
    gbdrift_secular = data["gbdrift (secular)/phi"] * data["zeta"]
    cvdrift = data["cvdrift (periodic)"] + gbdrift_secular
    gbdrift = data["gbdrift (periodic)"] + gbdrift_secular
    g = jnp.sqrt(jnp.abs(1 - pitch * B))
    return (cvdrift - 0.5 * gbdrift) * g + safediv(0.5 * gbdrift, g)


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
    drive = jnp.hypot((G * η_T - G_ω_α) + G * C / energy, G_ω_ψ)

    return G_ω_α * C + (G_ω_α * η_T + safediv(drift * (drive - drift), G)) * energy


def _energy_quad(num_energy):
    # The energy integral has weight E^(5/2) exp(-E), but
    # ω_* = η_T + C / E makes AE(E) ~ C/E for E near zero.
    return stop_gradient(orthgauss(num_energy, GeneralizedLaguerre(np.array([1.5]))))


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
    radial_scale="float : Multiplier for the radial correlation length.",
    binormal_scale="float : Multiplier for the binormal correlation length.",
    num_energy="int : Resolution for generalized Laguerre quadrature over energy.",
    energy_quad="tuple : Nodes and weights for the energy quadrature.",
    **Options._doc,
)
@partial(
    jit,
    static_argnames=Options._static_argnames + ("num_energy",),
)
def _available_energy(params, transforms, profiles, data, **kwargs):
    """Dimensionless available energy of trapped electrons.

    References
    ----------
    .. [1] R. J. J. Mackenbach et al., J. Plasma Phys. 89, 905890513 (2023).
    .. [2] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
           bounce-averaging algorithm and its applications,"
           J. Plasma Physics. https://doi:10.1017/S0022377826101652.

    Parameters
    ----------
    radial_scale, binormal_scale : float
        Correlation-length multipliers. Default is 1.
    num_energy : int
        Resolution for generalized Gauss-Laguerre quadrature over energy.

    """
    # noqa: unused dependency
    radial_scale = kwargs.get("radial_scale", 1.0)
    binormal_scale = kwargs.get("binormal_scale", 1.0)
    energy_quad = kwargs.get("energy_quad", None)
    if energy_quad is None:
        energy_quad = _energy_quad(kwargs.get("num_energy", 16))

    grid = transforms["grid"]
    opts = Options.guess(-1, grid, **kwargs)

    def foreach_surface(data):
        bounce = Bounce2D(grid, data, data["angle"], **opts)

        def foreach(pitch_inv):
            return (
                _ae(
                    *bounce.integrate(
                        [_G_hat_half, _binormal_drift_wb_inverse, _radial_drift],
                        pitch_inv,
                        data,
                        names,
                        num_well=opts.num_well,
                        loop=opts.loop,
                    ),
                    data,
                    energy_quad[0],
                )
                .sum(-1)
                .mean(-3)
                .dot(energy_quad[1])
            )

        pitch_inv, weight = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        return jnp.sum(
            batch_map(foreach, pitch_inv, opts.pitch_batch_size)
            * weight
            / pitch_inv**2,
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
        rho_data={
            "ae grad(density)": safediv(radial_scale * data["ne_r"], data["ne"]),
            "ae psi width": radial_scale * data["psi_r"],
            "ae alpha width": safediv(binormal_scale, data["rho"]),
            "ae grad(temperature)": safediv(radial_scale * data["Te_r"], data["Te"]),
        },
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1

    B0 = data["max_tz |B|"]
    scalar = 2 * jnp.pi * grid.NFP / opts.num_field_periods
    data["available energy"] = (scalar * B0 / data["V_psi"]) * grid.expand(out)
    return data
