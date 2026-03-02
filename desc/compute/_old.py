"""Old compute functions.

These do not appear in the public documentation under the list of variables.
"""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce1D
from ..integrals.quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from ..utils import cross, dot, safediv
from ._fast_ion import _drift1, _drift2, _radial_drift, _v_tau
from ._neoclassical import _bounce_doc, _dH_ripple, _dI_ripple
from .data_index import register_compute_fun

_bounce1D_doc = {
    "num_well": _bounce_doc["num_well"],
    "num_quad": _bounce_doc["num_quad"],
    "num_pitch": _bounce_doc["num_pitch"],
    "surf_batch_size": _bounce_doc["surf_batch_size"],
    "quad": _bounce_doc["quad"],
}


def _compute(fun, fun_data, data, grid, num_pitch, surf_batch_size=1, simp=False):
    """Compute Bounce1D integral quantity with ``fun``.

    Parameters
    ----------
    fun : callable
        Function to compute.
    fun_data : dict[str, jnp.ndarray]
        Data to provide to ``fun``. This dict will be modified.
    data : dict[str, jnp.ndarray]
        DESC data dict.
    grid : Grid
        Grid that can expand and compress.
    num_pitch : int
        Resolution for quadrature over velocity coordinate.
    surf_batch_size : int
        Number of flux surfaces with which to compute simultaneously.
        Default is ``1``.
    simp : bool
        Whether to use an open Simpson rule instead of uniform weights.

    """
    for name in Bounce1D.required_names:
        fun_data[name] = data[name]
    for name in fun_data:
        fun_data[name] = Bounce1D.reshape(grid, fun_data[name])
    fun_data["pitch_inv"], fun_data["pitch_inv weight"] = Bounce1D.get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
        simp=simp,
    )
    out = batch_map(fun, fun_data, surf_batch_size)
    assert out.ndim == 1
    return grid.expand(out)


@register_compute_fun(
    name="old effective ripple 3/2",
    label=(
        # ε¹ᐧ⁵ = π/(8√2) R₀²〈|∇ψ|〉⁻² B₀⁻¹ ∫ dλ λ⁻² 〈 ∑ⱼ Hⱼ²/Iⱼ 〉
        "\\epsilon_{\\mathrm{eff}}^{3/2} = \\frac{\\pi}{8 \\sqrt{2}} "
        "R_0^2 \\langle \\vert\\nabla \\psi\\vert \\rangle^{-2} "
        "B_0^{-1} \\int d\\lambda \\lambda^{-2} "
        "\\langle \\sum_j H_j^2 / I_j \\rangle"
    ),
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

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    https://doi.org/10.1063/1.873749.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    # noqa: unused dependency
    num_well = kwargs.get("num_well", None)
    num_pitch = kwargs.get("num_pitch", 51)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    quad = (
        kwargs["quad"] if "quad" in kwargs else chebgauss2(kwargs.get("num_quad", 32))
    )

    def eps_32(data):
        """(∂ψ/∂ρ)⁻² B₀⁻³ ∫ dλ λ⁻² ∑ⱼ Hⱼ²/Iⱼ."""
        # B₀ has units of λ⁻¹.
        # Nemov's ∑ⱼ Hⱼ²/Iⱼ = (∂ψ/∂ρ)² (λB₀)³ (H² / I).sum(-1).
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        H, I = bounce.integrate(
            [_dH_ripple, _dI_ripple],
            data["pitch_inv"],
            data,
            "|grad(rho)|*kappa_g",
            num_well=num_well,
        )
        return jnp.sum(
            safediv(H**2, I).sum(-1).mean(-2)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 3,
            axis=-1,
        )

    grid = transforms["grid"].source_grid
    B0 = data["max_tz |B|"]
    data["old effective ripple 3/2"] = (
        _compute(
            eps_32,
            {"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]},
            data,
            grid,
            num_pitch,
            surf_batch_size,
            simp=True,
        )
        / data["fieldline length"]
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * (jnp.pi / (8 * 2**0.5))
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

    Poloidal motion of trapped particle orbits in real-space coordinates.
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
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )

    def Gamma_c(data):
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well)
        v_tau, drift1, drift2 = bounce.integrate(
            [_v_tau, _drift1, _drift2],
            data["pitch_inv"],
            data,
            ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
            points,
        )
        # This is γ_c π/2.
        gamma_c = jnp.arctan(
            safediv(
                drift1,
                drift2
                * bounce.interp_to_argmin(data["|grad(rho)|*|e_alpha|r,p|"], points),
            )
        )
        return jnp.sum(
            (v_tau * gamma_c**2).sum(-1).mean(-2)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / (2**1.5 * jnp.pi)

    fun_data = {
        "|grad(psi)|*kappa_g": data["|grad(psi)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_r|v,p": data["|B|_r|v,p"],
        "K": data["iota_r"]
        * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]),
    }
    grid = transforms["grid"].source_grid
    data["old Gamma_c"] = (
        _compute(Gamma_c, fun_data, data, grid, num_pitch, surf_batch_size)
        / data["fieldline length"]
    )
    return data


def _poloidal_drift(data, B, pitch):
    return safediv(
        data["gbdrift"] * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
    )


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

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.

    This expression has a secular term that drives the result to zero as the number
    of toroidal transits increases if the secular term is not averaged out from the
    singular integrals. It is observed that this implementation does not average
    out the secular term. Currently, an optimization using this metric may need
    to be evaluated by measuring decrease in Γ_c at a fixed number of toroidal
    transits.
    """
    # noqa: unused dependency
    num_well = kwargs.get("num_well", None)
    num_pitch = kwargs.get("num_pitch", 64)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )

    def Gamma_c(data):
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        v_tau, radial_drift, poloidal_drift = bounce.integrate(
            [_v_tau, _radial_drift, _poloidal_drift],
            data["pitch_inv"],
            data,
            ["cvdrift0", "gbdrift"],
            num_well=num_well,
        )
        # This is γ_c π/2.
        gamma_c = jnp.arctan(safediv(radial_drift, poloidal_drift))
        return jnp.sum(
            (v_tau * gamma_c**2).sum(-1).mean(-2)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / (2**1.5 * jnp.pi)

    grid = transforms["grid"].source_grid
    data["old Gamma_c Velasco"] = (
        _compute(
            Gamma_c,
            {"cvdrift0": data["cvdrift0"], "gbdrift": data["gbdrift"]},
            data,
            grid,
            num_pitch,
            surf_batch_size,
        )
        / data["fieldline length"]
    )
    return data
