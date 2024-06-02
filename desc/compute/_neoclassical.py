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
    get_pitch,
    grad_affine_bijection,
)
from .data_index import register_compute_fun


def _is_Newton_Cotes(quad):
    if hasattr(quad, "is_Newton_Cotes"):
        return quad.is_Newton_Cotes
    else:
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


def _get_pitch(grid, data, quad, num=75):
    # Get endpoints of integral over pitch for each flux surface.
    # with num values uniformly spaced in between.
    min_B = grid.compress(data["min_tz |B|"])
    max_B = grid.compress(data["max_tz |B|"])
    if _is_Newton_Cotes(quad):
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

    if _is_Newton_Cotes(quad):
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

    ripple = _poloidal_average(
        g, ripple.reshape(g.num_rho, g.num_alpha) / data["L|r,a"]
    )
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
