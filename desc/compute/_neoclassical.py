"""Compute functions for neoclassical transport.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from functools import partial

import orthax
import quadax
from termcolor import colored

from desc.backend import jit, jnp, trapezoid

from ..utils import warnif
from .bounce_integral import (
    affine_bijection,
    bounce_integral,
    composite_linspace,
    grad_affine_bijection,
)
from .data_index import register_compute_fun


def _is_Newton_Cotes(quad):
    return quad in [trapezoid, quadax.trapezoid, quadax.simpson]


def vec_quadax(quad):
    """Vectorize an adaptive quadrature method from quadax to compute ripple.

    Parameters
    ----------
    quad : callable
        Adaptive quadrature method matching API from quadax.

    Returns
    -------
    vec_quad : callable
        Vectorized adaptive quadrature method.

    """
    if _is_Newton_Cotes(quad):
        return quad

    def vec_quad(fun, interval, B_sup_z, B, B_z_ra, arg1, arg2):
        return quad(fun, interval, args=(B_sup_z, B, B_z_ra, arg1, arg2))[0]

    vec_quad = jnp.vectorize(
        vec_quad, signature="(2),(m),(m),(m),(m),(m)->()", excluded={0}
    )
    return vec_quad


def poloidal_leggauss(deg, a_min=0, a_max=2 * jnp.pi):
    """Gauss-Legendre quadrature.

    Returns quadrature points αₖ and weights wₖ for the approximate evaluation
    of the integral ∫ f(α) dα ≈ ∑ₖ wₖ f(αₖ).

    Set resolution > 1 to see if a long field line integral can be approximated
    by flux surface average of shorter field line integrals with finite transits.

    Assuming the rotational transform is irrational, the limit where the
    parameterization of the field line length tends to infinity of an average
    along the field line will converge to a flux surface average.
    In theory, we can compute such quantities with averages over finite lengths
    of the field line, e.g. one toroidal transit, for many values of the poloidal
    field line label and then average this over the poloidal domain.

    This should also work for integrands which are bounce integrals;
    Since everything is continuous, as the number of nodes tend to infinity both
    approaches should converge to the same result, assuming irrational surface.
    However, the order at which all the bounce integrals detected over the surface
    are summed differs, so the convergence rate will differ.

    Parameters
    ----------
    deg: int
        Number of quadrature points.
    a_min: float
        Min α value.
    a_max: float
        Max α value.

    Returns
    -------
    alpha : Array
        Quadrature points.
    w : Array
        Quadrature weights.

    """
    x, w = orthax.legendre.leggauss(deg)
    w = w * grad_affine_bijection(a_min, a_max)
    alpha = affine_bijection(x, a_min, a_max)
    return alpha, w


def _poloidal_average(grid, f, name=""):
    assert f.shape[-1] == grid.num_poloidal
    if grid.poloidal_weight is None:
        warnif(
            grid.num_poloidal != 1,
            msg=colored(f"{name} reduced via uniform poloidal mean.", "yellow"),
        )
        avg = jnp.mean(f, axis=-1)
    else:
        avg = f @ grid.poloidal_weight / jnp.sum(grid.poloidal_weight)
    return avg


@register_compute_fun(
    name="L|r,a",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}"
    " \\frac{d\\zeta}{B^{\\zeta} \\vert B \\vert}",
    units="m",
    units_long="meters",
    description="Length along field line",
    dim=2,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="ra",
    data=["B^zeta", "|B|"],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
)
def _L_ra(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    data["L|r,a"] = quadax.simpson(
        jnp.reshape(1 / (data["B^zeta"] * data["|B|"]), shape),
        jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    )
    return data


@register_compute_fun(
    name="G|r,a",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}"
    " \\frac{d\\zeta}{B^{\\zeta} \\vert B \\vert \\sqrt g}",
    units="m^{-2}",
    units_long="inverse meters squared",
    description="Length over volume along field line",
    dim=2,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="ra",
    data=["B^zeta", "|B|", "sqrt(g)"],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
)
def _G_ra(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    data["G|r,a"] = quadax.simpson(
        jnp.reshape(1 / (data["B^zeta"] * data["|B|"] * data["sqrt(g)"]), shape),
        jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    ) / (4 * jnp.pi**2)
    return data


@register_compute_fun(
    name="effective ripple raw",
    label="∫ dλ λ⁻² \\langle ∑ⱼ Hⱼ²/Iⱼ \\rangle",
    units="T^2",
    units_long="Tesla squared",
    description="Effective ripple modulation amplitude, not dimensionless",
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
        "|grad(psi)|",
        "kappa_g",
        "L|r,a",
    ],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
    bounce_integral=(
        "callable : Method to compute bounce integrals. "
        "(You may want to wrap desc.compute.bounce_integral.bounce_integral "
        "to change optional parameters such as quadrature resolution, etc.)."
    ),
    batch="bool : Whether to perform computation in a batched manner.",
    quad=(
        "callable : Quadrature method over velocity coordinate. "
        "Adaptive quadrature method from quadax must be wrapped with vec_quadax."
    ),
    quad_res="int : Resolution for quadrature over velocity coordinate.",
)
@partial(jit, static_argnames=["bounce_integral", "batch", "quad", "quad_res"])
def _effective_ripple_raw(params, transforms, profiles, data, **kwargs):
    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    _bounce_integral = kwargs.get("bounce_integral", bounce_integral)
    batch = kwargs.get("batch", True)
    quad = kwargs.get("quad", quadax.simpson)
    quad_res = kwargs.get("quad_res", 100)
    # Get endpoints of integral over pitch for each field line.
    min_B, max_B = map(g.compress, (data["min_tz |B|"], data["max_tz |B|"]))
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + 1e-6) * min_B
    max_B = (1 - 1e-6) * max_B
    pitch_endpoint = 1 / jnp.stack([max_B, min_B])

    def dH(grad_psi_norm, kappa_g, B, pitch, Z):
        # absorbed 1/λ into H
        return jnp.sqrt(1 / pitch - B) * (4 / B - pitch) * grad_psi_norm * kappa_g / B

    def dI(B, pitch, Z):
        return jnp.sqrt(1 - pitch * B) / B

    if _is_Newton_Cotes(quad):
        bounce_integrate, _ = _bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots
        )

        def d_ripple(pitch):
            """Return λ⁻² ∑ⱼ Hⱼ²/Iⱼ evaluated at λ = pitch.

            Parameters
            ----------
            pitch : Array, shape(*pitch.shape[:-1], g.num_rho * g.num_alpha)
                Pitch angle.

            Returns
            -------
            d_ripple : Array, shape(pitch.shape)
                λ⁻² ∑ⱼ Hⱼ²/Iⱼ

            """
            H = bounce_integrate(
                dH, [data["|grad(psi)|"], data["kappa_g"]], pitch, batch=batch
            )
            I = bounce_integrate(dI, [], pitch, batch=batch)
            return jnp.nansum(H**2 / I, axis=-1)

        pitch = composite_linspace(pitch_endpoint, quad_res)
        pitch = jnp.broadcast_to(
            pitch[..., jnp.newaxis], (pitch.shape[0], g.num_rho, g.num_alpha)
        ).reshape(pitch.shape[0], g.num_rho * g.num_alpha)
        ripple = quad(d_ripple(pitch), pitch, axis=0)
    else:
        # Use adaptive quadrature.

        def d_ripple(pitch, B_sup_z, B, B_z_ra, grad_psi_norm, kappa_g):
            bounce_integrate, _ = _bounce_integral(B_sup_z, B, B_z_ra, knots)
            H = bounce_integrate(dH, [grad_psi_norm, kappa_g], pitch, batch=batch)
            I = bounce_integrate(dI, [], pitch, batch=batch)
            return jnp.squeeze(jnp.nansum(H**2 / I, axis=-1))

        pitch_endpoint = pitch_endpoint.T[:, jnp.newaxis]
        assert pitch_endpoint.shape == (g.num_rho, 1, 2)
        args = [
            f.reshape(g.num_rho, g.num_alpha, g.num_zeta)
            for f in [
                data["B^zeta"],
                data["|B|"],
                data["|B|_z|r,a"],
                data["|grad(psi)|"],
                data["kappa_g"],
            ]
        ]
        ripple = quad(d_ripple, pitch_endpoint, *args)

    ripple = _poloidal_average(
        g, ripple.reshape(g.num_rho, g.num_alpha) / data["L|r,a"]
    )
    data["effective ripple raw"] = g.expand(ripple)
    return data


@register_compute_fun(
    name="effective ripple",  # this is ε¹ᐧ⁵
    label="π/(8√2) (R₀(∂_ψ V)/S)² ∫ dλ λ⁻² \\langle ∑ⱼ Hⱼ²/Iⱼ \\rangle",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["R0", "V_r(r)", "psi_r", "S(r)", "effective ripple raw"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """Evaluation of 1/ν neoclassical transport in stellarators.

    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    https://doi.org/10.1063/1.873749.
    """
    data["effective ripple"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (data["R0"] * data["V_r(r)"] / data["psi_r"] / data["S(r)"]) ** 2
        * data["effective ripple raw"]
    )
    return data


@register_compute_fun(
    name="Gamma_c",
    # When comparing Velasco's Γ_c: https://doi.org/10.1088/1741-4326/ac2994
    #           with Nemov's   Γ_c: https://doi.org/10.1063/1.2912456,
    # note that
    #     dλ v τ_b = 8 (∂I/∂b) db = -8 ∂I/∂(λ⁻¹) dλ λ⁻²
    # and
    #     4π² ∂Ψₜ/∂V = lim{L → ∞} ( [∫₀ᴸ ds/(B √g)] / [∫₀ᴸ ds/B] )
    # where the integrals are along an irrational field line with
    #     ds given by dζ / B^ζ
    #     √g the (Ψ, θ, ζ)-coordinate Jacobian
    # There is also the dimensionless difference between Nemov's and Velascos's γ_c,
    # mentioned in Velasco's footnote 4. If this difference is γ_c ignored, and the
    # (missing?) √g factor is pushed into the integral over alpha in Velasco eq. 18
    # then we have that
    #     Velasco Γ_c = Nemov Γ_c * lim{L → ∞} ∫₀ᴸ ds/(B √g) / (2π).
    # In particular, Velasco Γ_c grows with the length of the field line
    # which isn't what we want.
    # TODO:
    #  We currently implement Nemov Γ_c with Velasco γ_c.
    #  Switch to Nemov γ_c too.
    label="π/(2√2) ∫ dλ λ⁻² \\langle ∑ⱼ [γ_c² ∂I/∂(λ⁻¹)]ⱼ \\rangle",
    units="~",
    units_long="None",
    description="Nemov's energetic ion confinement proxy",
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
        "L|r,a",
    ],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
    bounce_integral=(
        "callable : Method to compute bounce integrals. "
        "(You may want to wrap desc.compute.bounce_integral.bounce_integral "
        "to change optional parameters such as quadrature resolution, etc.)."
    ),
    batch="bool : Whether to perform computation in a batched manner.",
    quad=(
        "callable : Quadrature method over velocity coordinate. "
        "Adaptive quadrature method from quadax must be wrapped with vec_quadax."
    ),
    quad_res="int : Resolution for quadrature over velocity coordinate.",
)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Poloidal motion of trapped particle orbits in real-space coordinates.

    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    """
    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    _bounce_integral = kwargs.get("bounce_integral", bounce_integral)
    batch = kwargs.get("batch", True)
    quad = kwargs.get("quad", quadax.simpson)
    quad_res = kwargs.get("quad_res", 100)
    # Get endpoints of integral over pitch for each field line.
    min_B, max_B = map(g.compress, (data["min_tz |B|"], data["max_tz |B|"]))
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + 1e-6) * min_B
    max_B = (1 - 1e-6) * max_B
    pitch_endpoint = 1 / jnp.stack([max_B, min_B])

    def d_gamma_c(f, B, pitch, Z):
        return f * (1 - pitch * B / 2) / jnp.sqrt(1 - pitch * B)

    def dK(B, pitch, Z):
        return 0.5 / jnp.sqrt(1 - pitch * B)

    if _is_Newton_Cotes(quad):
        bounce_integrate, _ = _bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots
        )

        def d_Gamma_c(pitch):
            """Return ∑ⱼ [γ_c² ∂I/∂(λ⁻¹)]ⱼ evaluated at λ = pitch.

            Parameters
            ----------
            pitch : Array, shape(*pitch.shape[:-1], g.num_rho * g.num_alpha)
                Pitch angle.

            Returns
            -------
            d_Gamma_c : Array, shape(pitch.shape)
                ∑ⱼ [γ_c² ∂I/∂(λ⁻¹)]ⱼ

            """
            # TODO: Currently we have implemented Velasco's Gamma_c.
            #       If we add a 1/|grad(psi)| into the arctan of little
            #       gamma_c, we implement Nemov's Gamma_c. (Check this again).
            #       This will affect the gamma_c profile since |grad(psi)|
            #       depends on alpha.
            gamma_c = (
                2
                / jnp.pi
                * jnp.arctan(
                    bounce_integrate(d_gamma_c, data["cvdrift0"], pitch, batch=batch)
                    / bounce_integrate(d_gamma_c, data["gbdrift"], pitch, batch=batch)
                )
            )
            K = bounce_integrate(dK, [], pitch, batch=batch)  # λ⁻² ∂I/∂(λ⁻¹)
            return jnp.nansum(gamma_c**2 * K, axis=-1)

        pitch = composite_linspace(pitch_endpoint, quad_res)
        pitch = jnp.broadcast_to(
            pitch[..., jnp.newaxis], (pitch.shape[0], g.num_rho, g.num_alpha)
        ).reshape(pitch.shape[0], g.num_rho * g.num_alpha)
        Gamma_c = quad(d_Gamma_c(pitch), pitch, axis=0)
    else:
        # Use adaptive quadrature.

        def d_Gamma_c(pitch, B_sup_z, B, B_z_ra, cvdrift0, gbdrift):
            bounce_integrate, _ = _bounce_integral(B_sup_z, B, B_z_ra, knots)
            gamma_c = (
                2
                / jnp.pi
                * jnp.arctan(
                    bounce_integrate(d_gamma_c, cvdrift0, pitch, batch=batch)
                    / bounce_integrate(d_gamma_c, gbdrift, pitch, batch=batch)
                )
            )
            K = bounce_integrate(dK, [], pitch, batch=batch)
            return jnp.squeeze(jnp.nansum(gamma_c**2 * K, axis=-1))

        pitch_endpoint = pitch_endpoint.T[:, jnp.newaxis]
        assert pitch_endpoint.shape == (g.num_rho, 1, 2)
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
        Gamma_c = quad(d_Gamma_c, pitch_endpoint, *args)

    Gamma_c = _poloidal_average(
        g, Gamma_c.reshape(g.num_rho, g.num_alpha) / data["L|r,a"]
    )
    data["Gamma_c"] = g.expand(jnp.pi / 2**1.5 * Gamma_c)
    return data
