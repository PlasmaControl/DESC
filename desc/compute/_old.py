"""Old compute functions.

These do not appear in the public documentation under the list of variables.
They are kept for verification and correctness testing.
"""

from functools import partial

from desc.backend import jit, jnp

from ..integrals.bounce_integral import Bounce1D, Options
from ..utils import safediv
from ._fast_ion import (
    _gamma_c_data,
    _poloidal_drift_periodic,
    _radial_drift_1,
    _radial_drift_2,
    _v_tau,
)
from ._neoclassical import _dI_1, _dI_2
from .data_index import register_compute_fun

_bounce1D_doc = {
    "num_well": Options._doc["num_well"],
    "num_quad": Options._doc["num_quad"],
    "num_pitch": Options._doc["num_pitch"],
    "surf_batch_size": Options._doc["surf_batch_size"],
    "quad": Options._doc["quad"],
}


@register_compute_fun(
    name="old effective ripple 3/2",
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
        "fieldline length",
    ]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
@partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"])
def _epsilon_32_1D(params, transforms, profiles, data, **kwargs):
    """Effective ripple modulation amplitude to 3/2 power.

    [1] Evaluation of 1/ν neoclassical transport in stellarators.
        V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
        Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
        https://doi.org/10.1063/1.873749.

    """
    # noqa: unused dependency
    grid = transforms["grid"].source_grid
    opts = Options.guess(eta=1, grid=grid, Y_B=grid.num_zeta, **kwargs)
    num_well = kwargs.get("num_well", -1)

    def eps_32(data):
        pitch_inv, weight = Bounce1D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        I_1, I_2 = Bounce1D(grid, data, opts.quad).integrate(
            [_dI_1, _dI_2], pitch_inv, data, ["|grad(rho)|*kappa_g"], num_well=num_well
        )
        return jnp.sum(
            safediv(I_1**2, I_2).sum(-1).mean(-2) * weight / pitch_inv**3,
            axis=-1,
        )

    B0 = data["max_tz |B|"]
    scalar = jnp.pi / (8 * 2**0.5) * data["R0"] ** 2
    out = Bounce1D.batch(
        eps_32,
        {"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]},
        data,
        grid,
        opts.surf_batch_size,
    )
    data["old effective ripple 3/2"] = (
        (B0 / data["<|grad(rho)|>"]) ** 2
        * scalar
        * grid.expand(out)
        / data["fieldline length"]
    )
    return data


@register_compute_fun(
    name="old effective ripple",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Neoclassical transport in the banana regime",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["old effective ripple 3/2"],
    public=False,
)
def _effective_ripple_1D(params, transforms, profiles, data, **kwargs):
    """Proxy for neoclassical transport in the banana regime.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. In the banana regime, neoclassical (thermal)
    transport from ripple wells can become the dominant transport channel.
    The effective ripple (ε) proxy estimates the neoclassical transport
    coefficients in the banana regime.
    """
    data["old effective ripple"] = data["old effective ripple 3/2"] ** (2 / 3)
    return data


@register_compute_fun(
    name="old Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^phi",
        "B^phi_r|v,p",
        "|B|_r|v,p",
        "b",
        "grad(phi)",
        "grad(psi)",
        "|grad(psi)|",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "iota_r",
        "fieldline length",
    ]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
@partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"])
def _Gamma_c_1D(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Nemov et al.

    [1] Poloidal motion of trapped particle orbits in real-space coordinates.
        V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
        Phys. Plasmas 1 May 2008; 15 (5): 052501.
        https://doi.org/10.1063/1.2912456.
        Equation 61.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. The energetic particle confinement
    metric γ_c quantifies whether the contours of the second adiabatic invariant
    close on the flux surfaces. In the limit where the poloidal drift velocity
    majorizes the radial drift velocity, the contours lie parallel to flux
    surfaces. The optimization metric Γ_c averages γ_c² over the distribution
    of trapped particles on each flux surface.

    The radial electric field has a negligible effect, since fast particles
    have high energy with collisionless orbits, so it is assumed to be zero.
    """
    # noqa: unused dependency
    grid = transforms["grid"].source_grid
    opts = Options.guess(eta=-2, grid=grid, Y_B=grid.num_zeta, **kwargs)
    num_well = kwargs.get("num_well", -1)

    def Gamma_c(data):
        pitch_inv, weight = Bounce1D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        bounce = Bounce1D(grid, data, opts.quad)
        points = bounce.points(pitch_inv, num_well)
        v_tau, radial_drift, poloidal_drift = bounce.integrate(
            [_v_tau, _radial_drift_1, _poloidal_drift_periodic],
            pitch_inv,
            data,
            ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
            points,
        )
        # This is γ_c π/2.
        gamma_c = jnp.arctan(
            safediv(
                radial_drift,
                poloidal_drift
                * bounce.interp_to_argmin(data["|grad(rho)|*|e_alpha|r,p|"], points),
            )
        )
        return jnp.sum(
            (v_tau * gamma_c**2).sum(-1).mean(-2) * weight / pitch_inv**2,
            axis=-1,
        )

    out = Bounce1D.batch(Gamma_c, _gamma_c_data(data), data, grid, opts.surf_batch_size)
    data["old Gamma_c"] = (
        grid.expand(out) / data["fieldline length"] / (2**1.5 * jnp.pi)
    )
    return data


@register_compute_fun(
    name="old Gamma_c Velasco",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy "
    "as defined by Velasco et al. (doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "cvdrift0", "gbdrift", "fieldline length"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
@partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"])
def _Gamma_c_Velasco_1D(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Velasco et al.

    [1] A model for the fast evaluation of prompt losses of energetic ions in
        stellarators. Equation 16.
        J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
        https://doi.org/10.1088/1741-4326/ac2994.

    """
    # noqa: unused dependency
    grid = transforms["grid"].source_grid
    opts = Options.guess(eta=-1, grid=grid, Y_B=grid.num_zeta, **kwargs)
    num_well = kwargs.get("num_well", -1)

    def _poloidal_drift_secular(data, B, pitch):
        return safediv(
            data["gbdrift"] * (1 - 0.5 * pitch * B),
            jnp.sqrt(jnp.abs(1 - pitch * B)),
        )

    def Gamma_c(data):
        pitch_inv, weight = Bounce1D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        v_tau, radial_drift, poloidal_drift = Bounce1D(grid, data, opts.quad).integrate(
            [_v_tau, _radial_drift_2, _poloidal_drift_secular],
            pitch_inv,
            data,
            ["cvdrift0", "gbdrift"],
            num_well=num_well,
        )
        # This is γ_c π/2.
        gamma_c = jnp.arctan(safediv(radial_drift, poloidal_drift))
        return jnp.sum(
            (v_tau * gamma_c**2).sum(-1).mean(-2) * weight / pitch_inv**2,
            axis=-1,
        )

    out = Bounce1D.batch(
        Gamma_c,
        {"cvdrift0": data["cvdrift0"], "gbdrift": data["gbdrift"]},
        data,
        grid,
        opts.surf_batch_size,
    )
    data["old Gamma_c Velasco"] = (
        grid.expand(out) / data["fieldline length"] / (2**1.5 * jnp.pi)
    )
    return data
