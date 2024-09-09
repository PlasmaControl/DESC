"""Compute functions for neoclassical transport.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import pdb
from functools import partial

from orthax.legendre import leggauss
from quadax import romberg, simpson

from desc.backend import imap, jax, jit, jnp, trapezoid

from .bounce_integral import bounce_integral, get_pitch
from .data_index import register_compute_fun
from .utils import cross, dot, safediv

import matplotlib.pyplot as plt
import numpy as np


def _vec_quadax(quad, **kwargs):
    """Vectorize an adaptive quadrature method from quadax.

    Parameters
    ----------
    quad : callable
        Adaptive quadrature method matching API from quadax.

    Returns
    -------
    vec_quad : callable
        Vectorized adaptive quadrature method.

    """

    def vec_quad(fun, interval, B_sup_z, B, B_z_ra, arg1, arg2):
        return quad(fun, interval, args=(B_sup_z, B, B_z_ra, arg1, arg2), **kwargs)[0]

    vec_quad = jnp.vectorize(
        vec_quad, signature="(2),(m),(m),(m),(m),(m)->()", excluded={0}
    )
    return vec_quad


def _get_pitch(grid, min_B, max_B, num, for_adaptive=False):
    """Get points for quadrature over velocity coordinate.

    Parameters
    ----------
    grid : Grid
        The grid on which data is computed.
    min_B : jnp.ndarray
        Minimum |B| value.
    max_B : jnp.ndarray
        Maximum |B| value.
    num : int
        Number of values to uniformly space in between.
    for_adaptive : bool
        Whether to return just the points useful for an adaptive quadrature.

    Returns
    -------
    pitch : Array
        Pitch values in the desired shape to use in compute methods.

    """
    min_B = grid.compress(min_B)
    max_B = grid.compress(max_B)
    if for_adaptive:
        pitch = jnp.reciprocal(jnp.stack([max_B, min_B], axis=-1))[:, jnp.newaxis]
        assert pitch.shape == (grid.num_rho, 1, 2)
    else:
        pitch = get_pitch(min_B, max_B, num)
        pitch = jnp.broadcast_to(
            pitch[..., jnp.newaxis], (pitch.shape[0], grid.num_rho, grid.num_alpha)
        ).reshape(pitch.shape[0], grid.num_rho * grid.num_alpha)
    return pitch


def _poloidal_mean(grid, f):
    """Integrate f over poloidal angle and divide by 2π."""
    assert f.shape[-1] == grid.num_poloidal
    if grid.num_poloidal == 1:
        return jnp.squeeze(f, axis=-1)
    if not hasattr(grid, "spacing"):
        return jnp.mean(f, axis=-1)
    assert grid.is_meshgrid
    dp = grid.compress(grid.spacing[:, 1], surface_label="poloidal")
    return f @ dp / jnp.sum(dp)


@register_compute_fun(
    name="<L|r,a>",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}}"
    " \\frac{d\\zeta}{|B^{\\zeta}|}",
    units="m / T",
    units_long="Meter / tesla",
    description="(Mean) proper length of field line(s)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta"],
    resolution_requirement="z",  # and poloidal if near rational surfaces
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _L_ra_fsa(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    L_ra = simpson(
        y=jnp.reciprocal(data["B^zeta"]).reshape(shape),
        x=jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    )
    data["<L|r,a>"] = g.expand(jnp.abs(_poloidal_mean(g, L_ra)))
    return data


@register_compute_fun(
    name="<G|r,a>",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}}"
    " \\frac{d\\zeta}{|B^{\\zeta} \\sqrt g|}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="(Mean) proper length over volume of field line(s)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta", "sqrt(g)"],
    resolution_requirement="z",  # and poloidal if near rational surfaces
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _G_ra_fsa(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    G_ra = simpson(
        y=jnp.reciprocal(data["B^zeta"] * data["sqrt(g)"]).reshape(shape),
        x=jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    )
    data["<G|r,a>"] = g.expand(jnp.abs(_poloidal_mean(g, G_ra)))
    return data


@register_compute_fun(
    name="effective ripple",  # this is ε¹ᐧ⁵
    label=(
        # ε¹ᐧ⁵ = π/(8√2) (R₀/〈|∇ψ|〉)² ∫dλ λ⁻²B₀⁻¹ 〈 ∑ⱼ Hⱼ²/Iⱼ 〉
        "\\epsilon^{3/2} = \\frac{\\pi}{8 \\sqrt{2}} "
        "(R_0 / \\langle \\vert\\nabla \\psi\\vert \\rangle)^2 "
        "\\int d\\lambda \\lambda^{-2} B_0^{-1} "
        "\\langle \\sum_j H_j^2 / I_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^zeta",
        "|B|",
        "|B|_z|r,a",
        "|grad(rho)|",
        "kappa_g",
        "<L|r,a>",
        "R0",
        "<|grad(rho)|>",
    ],
    resolution_requirement="z",  # and poloidal if near rational surfaces
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    num_quad="int : Resolution for quadrature of bounce integrals. Default is 31.",
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate, preferably odd. "
        "Default is 125. Effective ripple will look smoother at high values. "
        "(If computed on many flux surfaces and micro oscillation is seen "
        "between neighboring surfaces, increasing num_pitch will smooth the profile)."
    ),
    num_wells=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance. "
        "As a reference, there are typically <= 5 wells per toroidal transit."
    ),
    batch="bool : Whether to vectorize part of the computation. Default is true.",
    # Some notes on choosing the resolution hyperparameters:
    # The default settings above were chosen such that the effective ripple profile on
    # the W7-X stellarator looks similar to the profile computed at higher resolution,
    # indicating convergence. The final resolution parameter to keep in mind is that
    # the supplied grid should sufficiently cover the flux surfaces. At/above the
    # num_quad and num_pitch parameters chosen above, the grid coverage should be the
    # parameter that has the strongest effect on the profile.
    # As a reference for W7-X, when computing the effective ripple by tracing a single
    # field line on each flux surface, a density of 100 knots per toroidal transit
    # accurately reconstructs the ripples along the field line. Truncating the field
    # line to [0, 20π] offers good convergence (after [0, 30π] the returns diminish).
    # Note that when further truncating the field line to [0, 10π], a dip/cusp appears
    # between the rho=0.7 and rho=0.8 surfaces, indicating that more coverage is
    # required to resolve the effective ripple in this region.
    # TODO: Improve performance... related to GitHub issue #1045.
    #  The difficulty is computing the magnetic field is expensive:
    #  the ripples along field lines are fine compared to the length of the field line
    #  required for sufficient coverage of the surface. This requires many knots to
    #  for the spline of the magnetic field to capture fine ripples in a large interval.
)
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_wells", "batch"])
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """https://doi.org/10.1063/1.873749.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    batch = kwargs.get("batch", True)
    num_wells = kwargs.get("size", None)
    g = transforms["grid"].source_grid
    bounce_integrate, _ = bounce_integral(
        data["B^zeta"],
        data["|B|"],
        data["|B|_z|r,a"],
        g.compress(g.nodes[:, 2], surface_label="zeta"),
        leggauss(kwargs.get("num_quad", 31)),
    )

    def dH(grad_rho_norm_kappa_g, B, pitch):
        # Removed |∂ψ/∂ρ| (λB₀)¹ᐧ⁵ from integrand of Nemov eq. 30. Reintroduced later.
        return (
            jnp.sqrt(jnp.abs(1 - pitch * B))
            * (4 / (pitch * B) - 1)
            * grad_rho_norm_kappa_g
            / B
        )

    def dI(B, pitch):  # Integrand of Nemov eq. 31.
        return jnp.sqrt(jnp.abs(1 - pitch * B)) / B

    def d_ripple(pitch):
        # Return (∂ψ/∂ρ)⁻² λ⁻²B₀⁻³ ∑ⱼ Hⱼ²/Iⱼ evaluated at λ = pitch.
        # Note (λB₀)³ db = (λB₀)³ λ⁻²B₀⁻¹ (-dλ) = λB₀² (-dλ) where B₀ has units of λ⁻¹.
        H = bounce_integrate(
            dH,
            # Interpolate |∇ρ| κ_g since it is smoother than κ_g alone.
            data["|grad(rho)|"] * data["kappa_g"],
            pitch,
            batch=batch,
            num_wells=num_wells,
        )
        I = bounce_integrate(dI, [], pitch, batch=batch, num_wells=num_wells)
        return pitch * jnp.sum(safediv(H**2, I), axis=-1)

    # The integrand is continuous and likely poorly approximated by a polynomial.
    # Composite quadrature should perform better than higher order methods.
    pitch = _get_pitch(
        g, data["min_tz |B|"], data["max_tz |B|"], kwargs.get("num_pitch", 125)
    )
    ripple = simpson(y=imap(d_ripple, pitch).squeeze(axis=1), x=pitch, axis=0)
    data["effective ripple"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (data["max_tz |B|"] * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * g.expand(_poloidal_mean(g, ripple.reshape(g.num_rho, g.num_alpha)))
        / data["<L|r,a>"]
    )
    return data


@register_compute_fun(
    name="Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^zeta",
        "|B|",
        "|B|_z|r,a",
        "cvdrift0",
        "gbdrift",
        "<L|r,a>",
    ],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    num_quad="int : Resolution for quadrature of bounce integrals. Default is 31.",
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate. Default is 125."
    ),
    num_wells=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance. "
        "As a reference, there are typically <= 5 wells per toroidal transit."
    ),
    batch="bool : Whether to vectorize part of the computation. Default is true.",
    adaptive=(
        "bool : Whether to adaptively integrate over the velocity coordinate. "
        "If true, then num_pitch specifies an upper bound on the maximum number "
        "of function evaluations."
    ),
)
@partial(
    jit, static_argnames=["num_quad", "num_pitch", "num_wells", "batch", "adaptive"]
)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Velasco et al.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61, using Velasco's γ_c from equation 15 of the above paper.
    """
    batch = kwargs.get("batch", True)
    num_wells = kwargs.get("num_wells", None)
    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    quad = leggauss(kwargs.get("num_quad", 31))
    num_pitch = kwargs.get("num_pitch", 125)
    adaptive = kwargs.get("adaptive", False)
    pitch = _get_pitch(g, data["min_tz |B|"], data["max_tz |B|"], num_pitch, adaptive)

    def d_v_tau(B, pitch):
        return safediv(2, jnp.sqrt(jnp.abs(1 - pitch * B)))

    def d_gamma_c(f, B, pitch):
        return safediv(f * (1 - pitch * B / 2), jnp.sqrt(jnp.abs(1 - pitch * B)))

    if not adaptive:
        bounce_integrate, _ = bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots, quad
        )

        def d_Gamma_c(pitch):
            # Return ∑ⱼ [v τ γ_c²]ⱼ evaluated at λ = pitch.
            # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
            # τ is the bounce time, and I is defined in Nemov eq. 36.
            v_tau = bounce_integrate(
                d_v_tau, [], pitch, batch=batch, num_wells=num_wells
            )
            gamma_c = (
                2
                / jnp.pi
                * jnp.arctan(
                    safediv(
                        bounce_integrate(
                            d_gamma_c,
                            data["cvdrift0"],
                            pitch,
                            batch=batch,
                            num_wells=num_wells,
                        ),
                        bounce_integrate(
                            d_gamma_c,
                            data["gbdrift"],
                            pitch,
                            batch=batch,
                            num_wells=num_wells,
                        ),
                    )
                )
            )

            return jnp.sum(v_tau * gamma_c**2, axis=-1)

        # The integrand is piecewise continuous and likely poorly approximated by a
        # polynomial. Composite quadrature should perform better than higher order
        # methods.
        Gamma_c = trapezoid(y=imap(d_Gamma_c, pitch).squeeze(axis=1), x=pitch, axis=0)
    else:

        def d_Gamma_c(pitch, B_sup_z, B, B_z_ra, cvdrift0, gbdrift):
            bounce_integrate, _ = bounce_integral(B_sup_z, B, B_z_ra, knots, quad)
            v_tau = bounce_integrate(
                d_v_tau, [], pitch, batch=batch, num_wells=num_wells
            )
            gamma_c = (
                2
                / jnp.pi
                * jnp.arctan(
                    safediv(
                        bounce_integrate(
                            d_gamma_c, cvdrift0, pitch, batch=batch, num_wells=num_wells
                        ),
                        bounce_integrate(
                            d_gamma_c, gbdrift, pitch, batch=batch, num_wells=num_wells
                        ),
                    )
                )
            )
            return jnp.squeeze(jnp.sum(v_tau * gamma_c**2, axis=-1))
       
        args = [
            f.reshape(g.num_rho, g.num_alpha, g.num_zeta)
            for f in [
                data["B^zeta"],
                data["|B|"],
                data["|B|_z|r,a"],
                data["cvdrift0"],
                data["gbdrift"],
            ]
        ]
        Gamma_c = _vec_quadax(romberg, divmax=jnp.log2(num_pitch + 1))(
            d_Gamma_c, pitch, *args
        )

    data["Gamma_c"] = (
        jnp.pi
        / (8 * 2**0.5)
        * g.expand(_poloidal_mean(g, Gamma_c.reshape(g.num_rho, g.num_alpha)))
        / data["<L|r,a>"]
    )
    return data


@register_compute_fun(
    name="Gamma_c Nemov",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy, Nemov et al.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^zeta",
        "B^phi",
        "B^phi_r|v,p",
        "b",
        "|B|",
        "|B|_z|r,a",
        "|B|_r|v,p",
        "<L|r,a>",
        "iota_r",
        "grad(phi)",
        "e^rho",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "psi_r",
    ],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    num_quad="int : Resolution for quadrature of bounce integrals. Default is 31.",
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate. Default is 125."
    ),
    num_wells=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance. "
        "As a reference, there are typically <= 5 wells per toroidal transit."
    ),
    batch="bool : Whether to vectorize part of the computation. Default is true.",
)
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_wells", "batch"])
def _Gamma_c_Nemov(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Nemov et al.

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61.

    The radial electric field has a negligible effect on alpha particle confinement,
    so it is assumed to be zero.
    """
    batch = kwargs.get("batch", True)
    num_wells = kwargs.get("num_wells", None)
    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    quad = leggauss(kwargs.get("num_quad", 31))
    num_pitch = kwargs.get("num_pitch", 125)
    pitch = _get_pitch(g, data["min_tz |B|"], data["max_tz |B|"], num_pitch)

    # The derivative (∂/∂ψ)|ϑ,ϕ belongs to flux coordinates which satisfy
    # α = ϑ − χ(ψ) ϕ where α is the poloidal label of ψ,α Clebsch coordinates.
    # Choosing χ = ι implies ϑ, ϕ are PEST angles.
    # ∂G/∂((λB₀)⁻¹) =     λ²B₀  ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) ∂|B|/∂ψ / |B|
    # ∂V/∂((λB₀)⁻¹) = 3/2 λ²B₀  ∫ dℓ √(1 − λ|B|) K / |B|
    # ∂g/∂((λB₀)⁻¹) =     λ²B₀² ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
    # tan(π/2 γ_c) =
    #              ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ρ| κ_g / |B|
    #              ----------------------------------------------
    # (|∇ρ| ‖e_α|ρ,ϕ‖)ᵢ ∫ dℓ √(1 − λ|B|) [ (1 − λ|B|/2)/(1 − λ|B|) ∂|B|/∂ψ + K ] / |B|

    def d_v_tau(B, pitch):
        return safediv(2, jnp.sqrt(jnp.abs(1 - pitch * B)))

    def num(grad_rho_norm_kappa_g, B, pitch):
        return (
            safediv(1 - pitch * B / 2, jnp.sqrt(jnp.abs(1 - pitch * B)))
            * grad_rho_norm_kappa_g
            / B
        )

    def den(dB_dpsi, K, B, pitch):
        return (
            jnp.sqrt(jnp.abs(1 - pitch * B))
            * (safediv(1 - pitch * B / 2, 1 - pitch * B) * dB_dpsi + K)
            / B
        )

    bounce_integrate, _ = bounce_integral(
        data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots, quad
    )

    def d_Gamma_c(pitch):
        # Return ∑ⱼ [v τ γ_c²]ⱼ evaluated at λ = pitch.
        # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
        # τ is the bounce time, and I is defined in Nemov eq. 36.
        v_tau = bounce_integrate(d_v_tau, [], pitch, batch=batch, num_wells=num_wells)
        gamma_c = (
            2
            / jnp.pi
            * jnp.arctan(
                safediv(
                    bounce_integrate(
                        num,
                        grad_rho_norm_kappa_g,
                        pitch,
                        batch=batch,
                        num_wells=num_wells,
                    ),
                    bounce_integrate(
                        den,
                        [dB_dpsi, K],
                        pitch,
                        batch=batch,
                        num_wells=num_wells,
                        weight=weight,
                    ),
                )
            )
        )

        return jnp.sum(v_tau * gamma_c**2, axis=-1)

    grad_rho_norm_kappa_g = data["|grad(rho)|"] * data["kappa_g"]
    dB_dpsi = data["|B|_r|v,p"] / data["psi_r"]
    weight = data["|grad(rho)|"] * data["|e_alpha|r,p|"]
    K = (
        # TODO: Confirm if K is smoother than individual components.
        #  If not, should spline separately.
        data["iota_r"] * dot(cross(data["e^rho"], data["b"]), data["grad(phi)"])
        # Behaves as log derivative if one ignores the issue of an argument with units.
        # Smoothness guaranteed by + lower bound of argument ∂log(|B|²/B^ϕ)/∂ψ |B|.
        # Note that Nemov assumes B^ϕ > 0; this is not true in DESC, but we account
        # for that in this computation.
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"])
        / data["psi_r"]
    )

    # The integrand is piecewise continuous and likely poorly approximated by a
    # polynomial. Composite quadrature should perform better than higher order
    # methods.
    Gamma_c = trapezoid(y=imap(d_Gamma_c, pitch).squeeze(axis=1), x=pitch, axis=0)
    data["Gamma_c Nemov"] = (
        jnp.pi
        / (8 * 2**0.5)
        * g.expand(_poloidal_mean(g, Gamma_c.reshape(g.num_rho, g.num_alpha)))
        / data["<L|r,a>"]
    )
    return data


@register_compute_fun(
    name="Gamma_a",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_a = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^zeta",
        "|B|",
        "|B|_z|r,a",
        "cvdrift0",
        "gbdrift",
        "<L|r,a>",
    ],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    num_quad="int : Resolution for quadrature of bounce integrals. Default is 31.",
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate. Default is 125."
    ),
    num_wells=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance. "
        "As a reference, there are typically <= 5 wells per toroidal transit."
    ),
    batch="bool : Whether to vectorize part of the computation. Default is true.",
    adaptive=(
        "bool : Whether to adaptively integrate over the velocity coordinate. "
        "If true, then num_pitch specifies an upper bound on the maximum number "
        "of function evaluations."
    ),
)
@partial(
    jit, static_argnames=["num_quad", "num_pitch", "num_wells", "batch", "adaptive"]
)
def _Gamma_a(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Velasco et al.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61, using Velasco's γ_c from equation 15 of the above paper.
    """
    batch = kwargs.get("batch", True)
    num_wells = kwargs.get("num_wells", None)
    g = transforms["grid"].source_grid
    # knots in zeta
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")

    # Quadrature resolution/bounce integral
    quad = leggauss(kwargs.get("num_quad", 31))

    adaptive = kwargs.get("adaptive", False)

    num_pitch = kwargs.get("num_pitch", 25)
    # Select pitch angles between min(|B|) and max(|B|)
    pitch = _get_pitch(g, data["min_tz |B|"], data["max_tz |B|"], num_pitch, adaptive)

    num_rho = g.num_rho
    num_alpha = g.num_alpha
    num_zeta = g.num_zeta

    # Repeat the calculation for each alpha
    alphas = g.nodes[::num_zeta, 1][::num_rho]

    def d_v_tau(B, pitch):
        return safediv(2, jnp.sqrt(jnp.abs(1 - pitch * B)))

    def d_gamma_a(f, B, pitch):
        return safediv(f * (1 - pitch * B / 2), jnp.sqrt(jnp.abs(1 - pitch * B)))

    if not adaptive:
        bounce_integrate, _ = bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots, quad
        )

        def d_Gamma_a(pitch, thresh=0.2):
            # Return ∑ⱼ [v τ γ_c²]ⱼ evaluated at λ = pitch.
            # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
            # τ is the bounce time, and I is defined in Nemov eq. 36.
            v_tau = bounce_integrate(
                d_v_tau, [], pitch, batch=batch, num_wells=num_wells
            )
            gamma_a = (
                2
                / jnp.pi
                * jnp.arctan(
                    safediv(
                        bounce_integrate(
                            d_gamma_a,
                            data["cvdrift0"],
                            pitch,
                            batch=batch,
                            num_wells=num_wells,
                        ),
                        bounce_integrate(
                            d_gamma_a,
                            data["gbdrift"],
                            pitch,
                            batch=batch,
                            num_wells=num_wells,
                        ),
                    )
                )
            )

            bavg_v_dot_grad_alpha = safediv(
                bounce_integrate(
                    d_gamma_a,
                    data["gbdrift"],
                    pitch,
                    batch=batch,
                    num_wells=num_wells,
                ),
                1,
            )

            # summing bounce integrals over all wells for a given pitch
            gamma_a = jnp.sum(v_tau * gamma_a, axis=-1)

            gamma_a_reshaped = jnp.reshape(gamma_a, (num_rho, num_alpha))

            bavg_v_dot_grad_alpha = jnp.sum(bavg_v_dot_grad_alpha, axis=-1)
            bavg_v_dot_grad_alpha_reshaped = jnp.reshape(
                bavg_v_dot_grad_alpha, (num_rho, num_alpha)
            )

            # --no-verify alpha_out=jnp.where(gamma_a_reshaped<-thresh,
            # --no-verify       alphas,-jnp.inf).max()
            # --no-verify alpha_in=jnp.where(gamma_a_reshaped>thresh,
            # --no-verify       alphas,jnp.inf).min()

            # --no-verify diff_out = alpha_out - alphas
            # --no-verify diff_in = alphas - alpha_in

            # --no-verify pdb.set_trace()
            # --no-verify # Compute H1 and H2
            # --no-verify H1 = jax.nn.relu(diff_out * bavg_v_dot_grad_alpha_reshaped)
            # --no-verify H2 = jax.nn.relu(diff_in * bavg_v_dot_grad_alpha_reshaped)

            return jnp.reshape(
                gamma_a_reshaped * bavg_v_dot_grad_alpha_reshaped,
                (1, num_rho * num_alpha),
            )

        # The integrand is piecewise continuous and likely poorly approximated by a
        # polynomial. Composite quadrature should perform better than higher order
        # methods.
        Gamma_a = trapezoid(y=imap(d_Gamma_a, pitch).squeeze(axis=1), x=pitch, axis=0)
    else:

        def d_Gamma_a(pitch, B_sup_z, B, B_z_ra, cvdrift0, gbdrift, thresh=0.5):
            bounce_integrate, _ = bounce_integral(B_sup_z, B, B_z_ra, knots, quad)
            v_tau = bounce_integrate(
                d_v_tau, [], pitch, batch=batch, num_wells=num_wells
            )
            gamma_a = (
                2
                / jnp.pi
                * jnp.arctan(
                    safediv(
                        bounce_integrate(
                            d_gamma_a, cvdrift0, pitch, batch=batch, num_wells=num_wells
                        ),
                        bounce_integrate(
                            d_gamma_a, gbdrift, pitch, batch=batch, num_wells=num_wells
                        ),
                    )
                )
            )

            bavg_v_dot_grad_alpha = bounce_integrate(
                d_gamma_a,
                data["gbdrift"],
                pitch,
                batch=batch,
                num_wells=num_wells,
            )

            idx_in = jnp.where(gamma_a < -thresh)[0]
            idx_out = jnp.where(gamma_a > thresh)[0]
            H1 = jax.nn.relu((idx_out - alphas) * bavg_v_dot_grad_alpha)
            H2 = jax.nn.relu((alphas - idx_in) * bavg_v_dot_grad_alpha)

            return v_tau * H1 * H2

        args = [
            f.reshape(g.num_rho, g.num_alpha, g.num_zeta)
            for f in [
                data["B^zeta"],
                data["|B|"],
                data["|B|_z|r,a"],
                data["cvdrift0"],
                data["gbdrift"],
            ]
        ]
        Gamma_a = _vec_quadax(romberg, divmax=jnp.log2(num_pitch + 1))(
            d_Gamma_a, pitch, *args
        )

    pdb.set_trace()

    data["Gamma_a"] = (
        jnp.pi
        / (8 * 2**0.5)
        * g.expand(_poloidal_mean(g, Gamma_a.reshape(g.num_rho, g.num_alpha)))
        / data["<L|r,a>"]
    )
    return data


@register_compute_fun(
    name="Gamma_d",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_a = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^zeta",
        "|B|",
        "|B|_z|r,a",
        "cvdrift0",
        "gbdrift",
        "<L|r,a>",
    ],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    num_quad="int : Resolution for quadrature of bounce integrals. Default is 31.",
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate. Default is 125."
    ),
    num_wells=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance. "
        "As a reference, there are typically <= 5 wells per toroidal transit."
    ),
    batch="bool : Whether to vectorize part of the computation. Default is true.",
    adaptive=(
        "bool : Whether to adaptively integrate over the velocity coordinate. "
        "If true, then num_pitch specifies an upper bound on the maximum number "
        "of function evaluations."
    ),
)
@partial(
    jit, static_argnames=["num_quad", "num_pitch", "num_wells", "batch", "adaptive"]
)
def _Gamma_d(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Velasco et al.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61, using Velasco's γ_c from equation 15 of the above paper.
    """
    batch = kwargs.get("batch", True)
    num_wells = kwargs.get("num_wells", None)
    g = transforms["grid"].source_grid
    # knots in zeta
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")

    # Quadrature resolution/bounce integral
    quad = leggauss(kwargs.get("num_quad", 31))

    adaptive = kwargs.get("adaptive", False)

    num_pitch = kwargs.get("num_pitch", 100)
    pitch = np.linspace(0.35,0.47,100) #this is the range used in Velasco demo 

    num_rho = g.num_rho
    num_alpha = g.num_alpha
    num_zeta = g.num_zeta
    
    alphas = g.nodes[g.unique_poloidal_idx,1]

    def d_v_tau(B, pitch):
        return safediv(2, jnp.sqrt(jnp.abs(1 - pitch * B)))

    def d_gamma_d(f, B, pitch):
        return safediv(f * (1 - pitch * B / 2), jnp.sqrt(jnp.abs(1 - pitch * B)))

    if not adaptive:
        bounce_integrate, _ = bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots, quad
        )
        
        gamma_d_evals =[] # store gamma_d evaluations of the pitch 

        for p in pitch:
            v_tau = bounce_integrate(
                d_v_tau, [], p, batch=batch, num_wells=num_wells
            )
            gamma_d = (
                2
                / jnp.pi
                * jnp.arctan(
                    safediv(
                        bounce_integrate(
                            d_gamma_d,
                            data["cvdrift0"],
                            p,
                            batch=batch,
                            num_wells=num_wells,
                        ),
                        bounce_integrate(
                            d_gamma_d,
                            data["gbdrift"],
                            p,
                            batch=batch,
                            num_wells=num_wells,
                        ),
                    )
                )
            )

            # summing bounce integrals over all wells for a given pitch
            gamma_d = jnp.sum(v_tau * gamma_d, axis=-1)
            gamma_d_reshaped = jnp.reshape(gamma_d, (num_rho, num_alpha)).squeeze()

            gamma_d_evals.append(gamma_d)  # Append results to list

        # Convert to a numpy array after the loop finishes
        gamma_d_evals = np.array(gamma_d_evals)
        
        print(gamma_d_evals.size())
        
        plt.contourf(pitch, alphas ,gamma_d_evals.T,cmap='JET')
        plt.colorbar(label="Gamma_d")
        plt.xlabel("r'%\lamda$")
        plt.ylabel("r'$\frac{\alpha}{2}$")
        plt.save()
 
        H = jax.nn.relu(gamma_d_reshaped - thresh)

        return jnp.reshape(H, (1, num_rho * num_alpha))
    
    

        # The integrand is piecewise continuous and likely poorly approximated by a
        # polynomial. Composite quadrature should perform better than higher order
        # methods.
        Gamma_d = trapezoid(y=imap(d_Gamma_d, pitch).squeeze(axis=1), x=pitch, axis=0)
        
    data["Gamma_d"] = (
        jnp.pi
        / (8 * 2**0.5)
        * g.expand(_poloidal_mean(g, Gamma_d.reshape(g.num_rho, g.num_alpha)))
        / data["<L|r,a>"]
    )
    return data
