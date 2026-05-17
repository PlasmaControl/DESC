"""Compute functions for available energy of trapped electrons."""

from functools import partial

from jax.lax import stop_gradient
from orthax.laguerre import laggauss

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D, Options
from ..utils import apply, parse_argname_change, safediv
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


def _ae(G, G_ω_α, G_ω_ψ, data, e):
    shape = (-1,) + (1,) * G.ndim

    G = G[..., None, :]  # Thiis is sqrt G hat.
    # scale by conjugate widths
    G_ω_α = G_ω_α[..., None, :] * data["ae psi width"].reshape(shape)
    G_ω_ψ = G_ω_ψ[..., None, :] * data["ae alpha width"].reshape(shape)
    G_ω_star = G * (
        (
            data["ae grad(density)"].reshape(shape)
            + data["ae grad(temperature)"].reshape(shape) * (e - 1.5)[..., None]
        )
        / e[..., None]
    )
    drift = jnp.hypot(G_ω_α, G_ω_ψ)
    drive = jnp.hypot(G_ω_star - G_ω_α, G_ω_ψ)

    return safediv(G_ω_α * G_ω_star + drift * (drive - drift), G)


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
    num_energy="int : Resolution for Gauss-Laguerre quadrature over energy.",
    **Options._doc,
)
@partial(
    jit,
    static_argnames=Options._static_argnames + ("num_energy",),
)
def _available_energy(params, transforms, profiles, data, **kwargs):
    """Dimensionless available energy of trapped electrons.

    Refrences
    ---------
    [1] Mackenbach et al., J. Plasma Phys. 89, 905890513 (2023).
    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
        and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    Parameters
    ----------
    radial_scale, binormal_scale : float
        Correlation-length multipliers. Default is 1.
    num_energy : int
        Resolution for Gauss-Laguerre quadrature over energy.

    """
    # noqa: unused dependency
    num_energy = kwargs.get("num_energy", 16)
    radial_scale = kwargs.get("radial_scale", 1.0)
    binormal_scale = kwargs.get("binormal_scale", 1.0)
    e, e_weight = stop_gradient(laggauss(num_energy))

    angle = parse_argname_change(
        kwargs.get("angle", kwargs.get("theta", None)), kwargs, "theta", "angle"
    )
    grid = transforms["grid"]
    opts = Options.guess(-1, grid, **kwargs)

    def available_energy(data):
        bounce = Bounce2D(grid, data, data["angle"], **opts)

        def fun(pitch_inv):
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
                    e,
                )
                .sum(-1)
                .mean(-3)
                .dot(e**2.5 * e_weight)
            )

        pitch_inv, weight = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        return jnp.sum(
            batch_map(fun, pitch_inv, opts.pitch_batch_size) * weight / pitch_inv**2,
            axis=-1,
        )

    names = (
        "cvdrift (periodic)",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "|grad(psi)|*kappa_g",
    )
    out = Bounce2D.batch(
        available_energy,
        apply(data, subset=names),
        data,
        angle,
        grid,
        opts.surf_batch_size,
        surface_data={
            "ae grad(density)": radial_scale * safediv(data["ne_r"], data["ne"]),
            "ae psi width": radial_scale * data["psi_r"],
            "ae alpha width": binormal_scale * safediv(1.0, data["rho"]),
            "ae grad(temperature)": radial_scale * safediv(data["Te_r"], data["Te"]),
        },
    )
    assert out.ndim == 1

    B0 = data["max_tz |B|"]
    scalar = 2 * jnp.pi * grid.NFP / opts.num_field_periods
    data["available energy"] = (scalar * B0 / data["V_psi"]) * grid.expand(out)
    return data
