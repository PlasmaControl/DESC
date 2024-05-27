"""Compute functions for neoclassical transport.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import orthax
import quadax
from termcolor import colored

from desc.backend import jit, jnp, trapezoid

from ..utils import warnif
from .bounce_integral import (
    affine_bijection,
    bounce_integral,
    composite_linspace,
    get_extrema,
    grad_affine_bijection,
)
from .data_index import register_compute_fun


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

    def vec_quad(fun, interval, B_sup_z, B, B_z_ra, grad_psi, cvdrift0):
        return quad(fun, interval, args=(B_sup_z, B, B_z_ra, grad_psi, cvdrift0))[0]

    vec_quad = jnp.vectorize(
        vec_quad, signature="(2),(m),(m),(m),(m),(m)->()", excluded={0}
    )
    return vec_quad


def alpha_leggauss(resolution, a_min=0, a_max=2 * jnp.pi):
    """Gauss-Legendre quadrature.

    Returns quadrature points αₖ and weights wₖ for the approximate evaluation
    of the integral ∫ f(α) dα ≈ ∑ₖ wₖ f(αₖ).

    For use with computing effective ripple, set resolution > 1 to see if a long
    field line integral can be approximated by flux surface average of field line
    integrals over finite transits.

    Assuming the rotational transform is irrational, the limit where the
    parameterization of the field line length tends to infinity, the
    integrals in (29), (30), (31) will converge to a flux surface average.
    In theory, we can compute the effective ripple via flux surface
    averages. It is not tested whether such a method will converge to the
    limit faster than extending the length of the field line chunk.

    Parameters
    ----------
    resolution: int
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
    x, w = orthax.legendre.leggauss(resolution)
    w = w * grad_affine_bijection(a_min, a_max)
    alpha = affine_bijection(x, a_min, a_max)
    return alpha, w


def _dH(grad_psi_norm, cvdrift0, B, pitch, Z):
    # Used to compute "ripple".
    return (
        pitch * jnp.sqrt(1 / pitch - B) * (4 / B - pitch) * grad_psi_norm * cvdrift0 / B
    )


def _dI(B, pitch, Z):
    # Used to compute "ripple".
    return jnp.sqrt(1 - pitch * B) / B


@register_compute_fun(
    name="ripple",
    label="∫ db ∑ⱼ Hⱼ² / Iⱼ",
    units="~",
    units_long="None",
    description="Ripple sum integral",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^zeta",
        "|B|_z|r,a",
        "|B|",
        "|grad(psi)|",
        "cvdrift0",
    ],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
    bounce_integral=(
        "callable : Method to compute bounce integrals."
        " (You may want to wrap desc.compute.bounce_integral.bounce_integral"
        " to change optional parameters such as quadrature resolution, etc.)."
    ),
    batched="bool : Whether to perform computation in a batched manner.",
    quad="callable : Quadrature method over velocity space (i.e. pitch integral)."
    "Adaptive quadrature method from quadax must be wrapped with vec_quadax.",
    quad_res="int : Resolution for quadrature over velocity space.",
    # if doing a flux surface average
    alpha_weight="Array : Quadrature weight over alpha.",
)
def _ripple(params, transforms, profiles, data, **kwargs):
    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")

    _bounce_integral = kwargs.pop("bounce_integral", bounce_integral)
    batched = kwargs.pop("batched", True)
    quad = kwargs.pop("quad", trapezoid)
    quad_res = kwargs.pop("quad_res", 21)
    alpha_weight = jnp.atleast_1d(kwargs.pop("alpha_weight", 1 / g.num_alpha))

    # Get boundary of integral over pitch for each field line.
    min_B, max_B = map(g.compress, (data["min_tz |B|"], data["max_tz |B|"]))
    min_B = (1 - 1e-6) * min_B
    max_B = (1 - 1e-6) * max_B
    min_B = jnp.broadcast_to(min_B[:, jnp.newaxis], (g.num_rho, g.num_alpha))
    max_B = jnp.broadcast_to(max_B[:, jnp.newaxis], (g.num_rho, g.num_alpha))
    boundary = jnp.stack([min_B, max_B])

    is_Newton_Cotes = quad in [trapezoid, quadax.trapezoid, quadax.simpson]
    if is_Newton_Cotes:
        bounce_integrate, spline = _bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots
        )

        @jit
        def rs(b):
            """Return ∑ⱼ Hⱼ² / Iⱼ evaluated at b.

            Parameters
            ----------
            b : Array, shape(*b.shape[:-1], g.num_rho * g.num_alpha)
                Multiplicative inverse of pitch angle.

            Returns
            -------
            rs : Array, shape(b.shape)
                ∑ⱼ Hⱼ² / Iⱼ except the sum over j is split across alpha.

            """
            pitch = 1 / b
            H = bounce_integrate(
                _dH, [data["|grad(psi)|"], data["cvdrift0"]], pitch, batched=batched
            )
            I = bounce_integrate(_dI, [], pitch, batched=batched)
            return jnp.nansum(H**2 / I, axis=-1)

        # For ε ∼ ∫ db ∑ⱼ Hⱼ² / Iⱼ, the contribution of ∑ⱼ Hⱼ² / Iⱼ may be largest in
        # the intervals such that b ∈ [|B|(ζ*) - db, |B|(ζ*)] where ζ* is a maxima. To
        # see this, observe that Iⱼ ∼ √(1 − λ |B|), so Hⱼ² / Iⱼ ∼ Hⱼ² / √(1 − λ |B|).
        # For λ = 1 / |B|(ζ*), near |B|(ζ*), the quantity 1 / √(1 − λ |B|) is singular
        # with strength ~ 1 / |∂|B|/∂_ζ|. Therefore, a quadrature for ε should evaluate
        # the integrand near b = 1 / λ = |B|(ζ*) to capture the fat banana orbits.
        breaks = get_extrema(**spline, sort=False).reshape(-1, g.num_rho, g.num_alpha)
        # Need to remove nan and include max_B regardless of whether there are any nan.
        breaks = jnp.where(jnp.isnan(breaks), max_B, breaks)
        # Gather quadrature points at breaks and linearly spaced within boundary.
        b = jnp.vstack([composite_linspace(boundary, quad_res), breaks])
        b = jnp.sort(b, axis=0).reshape(-1, g.num_rho * g.num_alpha)
        ripple = quad(rs(b), b, axis=0)
    else:
        # Use adaptive quadrature.
        @jit
        def rs(b, B_sup_z, B, B_z_ra, grad_psi, cvdrift0):
            # Quadax requires scalar integration interval, so we need scalar integrand.
            bounce_integrate, _ = _bounce_integral(B_sup_z, B, B_z_ra, knots)
            pitch = 1 / b
            H = bounce_integrate(_dH, [grad_psi, cvdrift0], pitch, batched=batched)
            I = bounce_integrate(_dI, [], pitch, batched=batched)
            return jnp.squeeze(jnp.nansum(H**2 / I, axis=-1))

        boundary = boundary.reshape(2, -1).T
        args = list(
            map(
                lambda f: f.reshape(g.num_rho * g.num_alpha, g.num_zeta),
                (
                    data["B^zeta"],
                    data["|B|"],
                    data["|B|_z|r,a"],
                    data["|grad(psi)|"],
                    data["cvdrift0"],
                ),
            )
        )
        ripple = quad(rs, boundary, *args)

    # Integrate over flux surface.
    ripple = ripple.reshape(g.num_rho, g.num_alpha) @ alpha_weight
    data["ripple"] = g.expand(ripple)
    return data


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\text{eff}}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta", "|B|", "|grad(psi)|", "ripple", "R0", "V_r(r)", "psi_r", "S(r)"],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
    fsa="bool : Whether to surface average to approximate the limit.",
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """Evaluation of 1/ν neoclassical transport in stellarators.

    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    https://doi.org/10.1063/1.873749
    """
    g = transforms["grid"].source_grid
    if kwargs.get("fsa", False):
        l = g.nodes[g.unique_zeta_idx[-1], 2] - g.nodes[g.unique_zeta_idx[0], 2]
        V = data["V_r(r)"] / data["psi_r"] * l
        S = data["S(r)"] * l
    else:
        shape = (g.num_rho, g.num_alpha, g.num_zeta)
        z = jnp.reshape(g.nodes[:, 2], shape)
        v = jnp.reshape(1 / (data["B^zeta"] * data["|B|"]), shape)
        V = jnp.mean(quadax.simpson(v, z, axis=-1), axis=1)
        S = jnp.mean(
            quadax.simpson(v * data["|grad(psi)|"].reshape(shape), z, axis=-1),
            axis=1,
        )
        V, S = map(g.expand, (V, S))
        warnif(g.num_alpha != 1, msg=colored("Reduced via mean over alpha.", "yellow"))

    data["effective ripple"] = (
        jnp.pi * data["R0"] ** 2 / (8 * 2**0.5) * V / S**2 * data["ripple"]
    ) ** (2 / 3)
    return data
