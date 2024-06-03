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

import quadax
from termcolor import colored

from desc.backend import jit, jnp, trapezoid

from ..utils import warnif
from .bounce_integral import bounce_integral, get_pitch
from .data_index import register_compute_fun


def _is_adaptive(quad):
    if hasattr(quad, "is_adaptive"):
        return quad.is_adaptive
    else:
        return quad not in [trapezoid, quadax.trapezoid, quadax.simpson]


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
    if not _is_adaptive(quad):
        return quad

    def vec_quad(fun, interval, B_sup_z, B, B_z_ra, arg1, arg2):
        return quad(fun, interval, args=(B_sup_z, B, B_z_ra, arg1, arg2))[0]

    vec_quad = jnp.vectorize(
        vec_quad, signature="(2),(m),(m),(m),(m),(m)->()", excluded={0}
    )
    return vec_quad


def _poloidal_mean(grid, f):
    assert f.shape[-1] == grid.num_poloidal
    if grid.spacing is None:
        warnif(
            grid.num_poloidal != 1,
            msg=colored("Reduced via uniform poloidal mean.", "yellow"),
        )
        return jnp.mean(f, axis=-1)
    else:
        assert grid.is_meshgrid
        dp = grid.compress(grid.spacing[:, 1], surface_label="poloidal")
        return f @ dp / jnp.sum(dp)


def _get_pitch(grid, data, quad, num=75):
    # Get endpoints of integral over pitch for each flux surface.
    # with num values uniformly spaced in between.
    min_B = grid.compress(data["min_tz |B|"])
    max_B = grid.compress(data["max_tz |B|"])
    if not _is_adaptive(quad):
        pitch = get_pitch(min_B, max_B, num)
        pitch = jnp.broadcast_to(
            pitch[..., jnp.newaxis], (pitch.shape[0], grid.num_rho, grid.num_alpha)
        ).reshape(pitch.shape[0], grid.num_rho * grid.num_alpha)
    else:
        pitch = 1 / jnp.stack([max_B, min_B], axis=-1)[:, jnp.newaxis]
        assert pitch.shape == (grid.num_rho, 1, 2)
    return pitch


@register_compute_fun(
    name="L|r,a",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}"
    " \\frac{d\\zeta}{B^{\\zeta}}",
    units="m / T",
    units_long="Meter / tesla",
    description="Length along field line",
    dim=2,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="ra",
    data=["B^zeta"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _L_ra(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    data["L|r,a"] = quadax.simpson(
        jnp.reshape(1 / data["B^zeta"], shape),
        jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    )
    return data


@register_compute_fun(
    name="G|r,a",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}"
    " \\frac{d\\zeta}{B^{\\zeta} \\sqrt g}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Length over volume along field line",
    dim=2,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="ra",
    data=["B^zeta", "sqrt(g)"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _G_ra(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    data["G|r,a"] = quadax.simpson(
        jnp.reshape(1 / (data["B^zeta"] * data["sqrt(g)"]), shape),
        jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    )
    return data


@register_compute_fun(
    name="effective ripple raw",
    label="∫dλ λ⁻²B₀⁻¹ \\langle ∑ⱼ Hⱼ²/Iⱼ \\rangle",
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
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    bounce_integral=(
        "callable : Method to compute bounce integrals. "
        "(You may want to wrap desc.compute.bounce_integral.bounce_integral "
        "to change optional parameters such as quadrature resolution, etc.)."
    ),
    batch="bool : Whether to perform computation in a batched manner.",
    # Composite quadrature should perform better than higher order methods.
    quad=(
        "callable : Quadrature method over velocity coordinate. "
        "Default is composite Simpson's rule. "
        "Accepts any callable with signature matching quad(f(λ), λ, ..., axis). "
        "Accepts adaptive quadrature methods from quadax wrapped with vec_quadax. "
        "If using an adaptive method, it is highly recommended to set batch=True."
    ),
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate. "
        "Default is 75. This setting is ignored for adaptive quadrature."
    ),
)
# temporary
@partial(jit, static_argnames=["bounce_integral", "batch", "quad", "num_pitch"])
def _effective_ripple_raw(params, transforms, profiles, data, **kwargs):
    bounce = kwargs.get("bounce_integral", bounce_integral)
    batch = kwargs.get("batch", False)
    quad = kwargs.get("quad", quadax.simpson)
    num_pitch = kwargs.get("num_pitch", 75)

    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    pitch = _get_pitch(g, data, quad, num_pitch)

    def dH(grad_psi_norm, kappa_g, B, pitch):
        # Pulled out dimensionless factor of (λB₀)¹ᐧ⁵ from integrand of
        # Nemov equation 30. Multiplied back in at end.
        return (
            jnp.sqrt(1 - pitch * B)
            * (4 / (pitch * B) - 1)
            * grad_psi_norm
            * kappa_g
            / B
        )

    def dI(B, pitch):
        # Integrand of Nemov equation 31.
        return jnp.sqrt(1 - pitch * B) / B

    if not _is_adaptive(quad):
        bounce_integrate, _ = bounce(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots
        )

        def d_ripple(pitch):
            """Return λ⁻²B₀⁻¹ ∑ⱼ Hⱼ²/Iⱼ evaluated at λ = pitch.

            Parameters
            ----------
            pitch : Array, shape(*pitch.shape[:-1], g.num_rho * g.num_alpha)
                Pitch angle.

            Returns
            -------
            d_ripple : Array, shape(pitch.shape)
                λ⁻²B₀⁻¹ ∑ⱼ Hⱼ²/Iⱼ

            """
            H = bounce_integrate(
                dH, [data["|grad(psi)|"], data["kappa_g"]], pitch, batch=batch
            )
            I = bounce_integrate(dI, [], pitch, batch=batch)
            # (λB₀)³ db = (λB₀)³ B₀⁻¹λ⁻² (-dλ) = B₀²λ (-dλ)
            # We chose B₀ = 1 (inverse units of λ).
            # TODO: Think Neo chooses B₀ = "max_tz |B|".
            # The minus sign is accounted for with the integration order.
            return pitch * jnp.nansum(H**2 / I, axis=-1)

        # This has units of tesla meters.
        ripple = quad(d_ripple(pitch), pitch, axis=0)
    else:
        # Use adaptive quadrature.

        def d_ripple(pitch, B_sup_z, B, B_z_ra, grad_psi_norm, kappa_g):
            bounce_integrate, _ = bounce(B_sup_z, B, B_z_ra, knots)
            H = bounce_integrate(dH, [grad_psi_norm, kappa_g], pitch, batch=batch)
            I = bounce_integrate(dI, [], pitch, batch=batch)
            return jnp.squeeze(pitch * jnp.nansum(H**2 / I, axis=-1))

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
        ripple = quad(d_ripple, pitch, *args)

    ripple = _poloidal_mean(g, ripple.reshape(g.num_rho, g.num_alpha) / data["L|r,a"])
    data["effective ripple raw"] = g.expand(ripple)
    return data


@register_compute_fun(
    name="effective ripple",  # this is ε¹ᐧ⁵
    label="π/(8√2) (R₀(∂V/∂ψ)/S)² ∫dλ λ⁻²B₀⁻¹ \\langle ∑ⱼ Hⱼ²/Iⱼ \\rangle",
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
    label="π/(2√2) ∫dλ λ⁻²B₀⁻¹ \\langle ∑ⱼ [γ_c² ∂I/∂((λB₀)⁻¹)]ⱼ \\rangle",
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
    # Composite quadrature should perform better than higher order methods.
    # TODO: Two layers of pitch for quadrature, linearly spaced minB to maxB
    #  and then second layer for extrema in fixed [0, zeta*].
    quad=(
        "callable : Quadrature method over velocity coordinate. "
        "Default is composite Simpson's rule. "
        "Accepts any callable with signature matching quad(f(λ), λ, ..., axis). "
        "Accepts adaptive quadrature methods from quadax wrapped with vec_quadax. "
        "If using an adaptive method, it is highly recommended to set batch=True."
    ),
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate. "
        "Default is 75. This setting is ignored for adaptive quadrature."
    ),
)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16. (Not equation 18).

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61, using Velasco's γ_c from equation 15 of the above paper.

    Besides the difference in γ_c mentioned above, Nemov's Γ_c and Velasco Γ_c
    as defined in equation 16 are identical, although Nemov's expression is
    precise while Velasco's requires a flexible interpretation for the expression
    to be well-defined.

    Also, Velasco Γ_c as defined in equation 18 does not seem to match
    equation 16. Note that
        dλ v τ_b = - 4 (∂I/∂b) db = 4 ∂I/∂((λB₀)⁻¹) B₀⁻¹λ⁻² dλ
        4π² ∂Ψₜ/∂V = lim{L → ∞} ( [∫₀ᴸ ds/(B √g)] / [∫₀ᴸ ds/B] )
    where the integrals are along an irrational field line with
        ds / B given by dζ / B^ζ
        √g the (Ψ, α, ζ)-coordinate Jacobian
    If the (missing?) √g factor in Velasco Γ_c equation 18 is pushed into the
    integral over alpha in Velasco equation 18 then we have that
        eq. 18 Velasco Γ_c ∼ eq. 16 Velasco Γ_c * lim{L → ∞} ∫₀ᴸ ds/(B √g).

    """
    bounce = kwargs.get("bounce_integral", bounce_integral)
    batch = kwargs.get("batch", False)
    quad = kwargs.get("quad", quadax.simpson)
    num_pitch = kwargs.get("num_pitch", 75)

    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    pitch = _get_pitch(g, data, quad, num_pitch)

    def d_gamma_c(f, B, pitch):
        return f * (1 - pitch * B / 2) / jnp.sqrt(1 - pitch * B)

    def dK(B, pitch):
        return 1 / jnp.sqrt(1 - pitch * B)

    if not _is_adaptive(quad):
        bounce_integrate, _ = bounce(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots
        )

        def d_Gamma_c(pitch):
            """Return 2λ⁻²B₀⁻¹ ∑ⱼ [γ_c² ∂I/∂((λB₀)⁻¹)]ⱼ evaluated at λ = pitch.

            Parameters
            ----------
            pitch : Array, shape(*pitch.shape[:-1], g.num_rho * g.num_alpha)
                Pitch angle.

            Returns
            -------
            d_Gamma_c : Array, shape(pitch.shape)
                2λ⁻²B₀⁻¹ ∑ⱼ [γ_c² ∂I/∂((λB₀)⁻¹)]ⱼ

            """
            gamma_c = (
                2
                / jnp.pi
                * jnp.arctan(
                    bounce_integrate(d_gamma_c, data["cvdrift0"], pitch, batch=batch)
                    / bounce_integrate(d_gamma_c, data["gbdrift"], pitch, batch=batch)
                )
            )
            # K = 2λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where I is given in Nemov equation 36.
            # The factor of B₀ cancels, making this quantity independent of the
            # chosen reference magnetic field strength.
            K = bounce_integrate(dK, [], pitch, batch=batch)
            return jnp.nansum(gamma_c**2 * K, axis=-1)

        # This has units of meters / tesla.
        Gamma_c = quad(d_Gamma_c(pitch), pitch, axis=0)
    else:
        # Use adaptive quadrature.

        def d_Gamma_c(pitch, B_sup_z, B, B_z_ra, cvdrift0, gbdrift):
            bounce_integrate, _ = bounce(B_sup_z, B, B_z_ra, knots)
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
        Gamma_c = quad(d_Gamma_c, pitch, *args)

    Gamma_c = (
        jnp.pi
        / (4 * 2**0.5)
        * _poloidal_mean(g, Gamma_c.reshape(g.num_rho, g.num_alpha) / data["L|r,a"])
    )
    data["Gamma_c"] = g.expand(Gamma_c)
    return data
