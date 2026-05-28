"""Compute functions for neoclassical transport."""

from functools import partial

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D, Options
from ..integrals.surface_integral import surface_integrals
from ..utils import safediv
from ._drift import _I_1, _I_2
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
        "|grad(rho)|*kappa_g",
        "R0",
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

    References
    ----------
    .. [1] V. V. Nemov, S. V. Kasilov, W. Kernbichler, and M. F. Heyn,
           "Evaluation of 1/ν neoclassical transport in stellarators,"
           Phys. Plasmas 6, 4622-4632 (1999).
           https://doi.org/10.1063/1.873749.
    .. [2] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
           bounce-averaging algorithm and its applications,"
           J. Plasma Physics. https://doi:10.1017/S0022377826101652.

    """
    # noqa: unused dependency
    # TODO: in future don't close over grid so that sharding works
    grid = transforms["grid"]
    opts = Options.guess(1, grid, **kwargs)

    def foreach_surface(data):

        def foreach(pitch_inv):
            I_1, I_2 = bounce.integrate(
                [_I_1, _I_2],
                pitch_inv,
                data,
                ["|grad(rho)|*kappa_g"],
                num_well=opts.num_well,
            )
            return safediv(I_1**2, I_2).sum(-1).mean(-2)

        pitch_inv, weight = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        bounce = Bounce2D(grid, data, data["angle"], **opts)
        # B₀ has units of λ⁻¹.
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        return jnp.sum(
            batch_map(foreach, pitch_inv, opts.pitch_batch_size)
            * (weight / pitch_inv**3),
            axis=-1,
        )

    B0 = data["max_tz |B|"]
    scalar = (jnp.pi * data["R0"]) ** 2 / (
        opts.num_field_periods / grid.NFP * 4 * 2**0.5
    )
    out = Bounce2D.batch(
        foreach_surface,
        data,
        grid,
        angle=kwargs["angle"],
        names=("|grad(rho)|*kappa_g",),
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1
    data["effective ripple 3/2"] = (
        (B0 / data["<|grad(rho)|>"]) ** 2 * grid.expand(out * scalar) / data["V_psi"]
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
