"""Compute functions for neoclassical transport."""

from functools import partial

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D, Options
from ..integrals.surface_integral import surface_integrals
from ..utils import parse_argname_change, safediv
from .data_index import register_compute_fun


@register_compute_fun(
    name="V_psi",
    label="\\int \\vert B^{\\zeta} \\vert^{-1} \\mathrm{d}\\alpha \\mathrm{d}\\zeta",
    units="m^{3} / Wb",
    units_long="cubic meters per Weber",
    description="Surface integrated volume Jacobian determinant of "
    " Clebsch field line coordinate system (ψ,α,ζ)"
    " where ζ is the DESC toroidal coordinate.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    resolution_requirement="tz",
    data=["psi_r/sqrt(g)"],
)
def _field_line_weight(params, transforms, profiles, data, **kwargs):
    """∬_Ω abs(𝐁⋅∇ζ)⁻¹ dα dζ where (α,ζ) ∈ Ω = [0, 2π)²."""
    data["V_psi"] = surface_integrals(
        transforms["grid"], jnp.abs(jnp.reciprocal(data["psi_r/sqrt(g)"]))
    )
    return data


def _dI_1(data, B, pitch):
    """Integrand of Unalmis et al. eqaution 2.9 with |∂ψ/∂ρ| removed."""
    return (
        jnp.sqrt(jnp.abs(1 - pitch * B))
        * (4 / (pitch * B) - 1)
        * data["|grad(rho)|*kappa_g"]
        / B
    )


def _dI_2(data, B, pitch):
    """Integrand of Unalmis et al. equation 2.10."""
    return jnp.sqrt(jnp.abs(1 - pitch * B)) / B


@register_compute_fun(
    name="effective ripple 3/2",
    label="\\epsilon_{\\mathrm{eff}}^{3/2}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude to 3/2 power",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "kappa_g",
        "R0",
        "|grad(rho)|",
        "<|grad(rho)|>",
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **Options._doc,
)
@partial(jit, static_argnames=Options._static_argnames)
def _epsilon_32(params, transforms, profiles, data, **kwargs):
    """Effective ripple modulation amplitude to 3/2 power.

    [1] Evaluation of 1/ν neoclassical transport in stellarators.
        V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
        Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
        https://doi.org/10.1063/1.873749.

    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
        and its applications. Kaya Unalmis et al. Journal of Plasma Physics.
        Equation 2.12.

    """
    # noqa: unused dependency
    angle = parse_argname_change(
        kwargs.get("angle", kwargs.get("theta", None)), kwargs, "theta", "angle"
    )
    grid = transforms["grid"]
    opts = Options.guess(1, grid, **kwargs)

    def eps_32(data):
        pitch_inv, weight = Bounce2D.get_pitch_inv_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        bounce = Bounce2D(grid, data, data["angle"], **opts, is_fourier=True)

        def fun(pitch_inv):
            I_1, I_2 = bounce.integrate(
                [_dI_1, _dI_2],
                pitch_inv,
                data,
                ["|grad(rho)|*kappa_g"],
                num_well=opts.num_well,
                nufft_eps=opts.nufft_eps,
                is_fourier=True,
            )
            return safediv(I_1**2, I_2).sum(-1).mean(-2)

        # B₀ has units of λ⁻¹.
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        return jnp.sum(
            batch_map(fun, pitch_inv, opts.pitch_batch_size) * weight / pitch_inv**3,
            axis=-1,
        )

    B0 = data["max_tz |B|"]
    scalar = (jnp.pi * data["R0"]) ** 2 / (opts.num_transit * 4 * 2**0.5)
    out = Bounce2D.batch(
        eps_32,
        {"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]},
        data,
        angle,
        grid,
        opts.surf_batch_size,
    )
    data["effective ripple 3/2"] = scalar * (
        (B0 / data["<|grad(rho)|>"]) ** 2 * grid.expand(out) / data["V_psi"]
    )
    return data


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Neoclassical transport in the banana regime",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["effective ripple 3/2"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """Proxy for neoclassical transport in the banana regime.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. In the banana regime, neoclassical (thermal)
    transport from ripple wells can become the dominant transport channel.
    The effective ripple (ε) proxy estimates the neoclassical transport
    coefficients in the banana regime.
    """
    data["effective ripple"] = data["effective ripple 3/2"] ** (2 / 3)
    return data
