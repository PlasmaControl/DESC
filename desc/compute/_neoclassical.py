"""Compute functions for neoclassical transport objectives.

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

from desc.backend import jnp, trapezoid

from ..utils import errorif, warnif
from .bounce_integral import (
    affine_bijection_reverse,
    bounce_integral,
    composite_linspace,
    get_extrema,
    grad_affine_bijection_reverse,
)
from .data_index import register_compute_fun


def _dH(grad_psi_norm, cvdrift0, B, pitch, Z):
    return (
        pitch * jnp.sqrt(1 / pitch - B) * (4 / B - pitch) * grad_psi_norm * cvdrift0 / B
    )


def _dI(B, pitch, Z):
    return jnp.sqrt(1 - pitch * B) / B


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
    w = w * grad_affine_bijection_reverse(a_min, a_max)
    alpha = affine_bijection_reverse(x, a_min, a_max)
    return alpha, w


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
    data=["B^zeta", "|B|_z|r,a", "|B|", "max_tz |B|", "|grad(psi)|", "cvdrift0"],
    grid_fl="Grid : Field line grid.",
    alpha_weight="Array : Quadrature weight over alpha.",
    b_quad="callable : Quadrature method to integrate over dB.",
    b_quad_res="int : Resolution for quadrature over dB.",
    quad="callable : Quadrature method to compute bounce integrals.",
    automorphism="(callable, callable) : Change of variables for bounce integral.",
    check="bool : Flag for debugging.",
    plot="bool : Whether to plot some things if check is true.",
)
def _ripple(params, transforms, profiles, data, **kwargs):
    grid_fl = kwargs.pop("grid_fl")
    num_rho = grid_fl.num_rho
    errorif(num_rho != transforms["grid"].num_rho)
    errorif(
        # TODO: Add grid labels to compute quantities so this doesn't occur.
        grid_fl.num_nodes != transforms["grid"].num_nodes,
        msg="Set override_grid=False.",
    )
    alpha = grid_fl.compress(grid_fl.nodes[:, 1], surface_label="theta")
    alpha_weight = jnp.atleast_1d(kwargs.pop("alpha_weight", 1 / alpha.size))
    knots = grid_fl.compress(grid_fl.nodes[:, 2], surface_label="zeta")
    b_quad = kwargs.pop("b_quad", trapezoid)
    b_quad_res = kwargs.pop("b_quad_res", 5)
    bounce_integrate, spline = bounce_integral(
        data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots, **kwargs
    )

    def ripple_sum(b):
        """Return the ripple sum ∑ⱼ Hⱼ² / Iⱼ evaluated at b.

        Parameters
        ----------
        b : Array, shape(..., num_rho * alpha.size)
            Multiplicative inverse of pitch angle.

        Returns
        -------
        ripple_sum : Array, shape(..., num_rho * alpha.size)
            ∑ⱼ Hⱼ² / Iⱼ except the sum over j is split across alpha.

        """
        pitch = 1 / b
        H = bounce_integrate(_dH, [data["|grad(psi)|"], data["cvdrift0"]], pitch)
        I = bounce_integrate(_dI, [], pitch)
        return jnp.nansum(H**2 / I, axis=-1)

    # For ε ∼ ∫ db ∑ⱼ Hⱼ² / Iⱼ, the contribution of ∑ⱼ Hⱼ² / Iⱼ is largest in the
    # intervals such that b ∈ [|B|(ζ*) - db, |B|(ζ*)] where ζ* is a maxima. To
    # see this, observe that Iⱼ ∼ √(1 − λ |B|), so Hⱼ² / Iⱼ ∼ Hⱼ² / √(1 − λ |B|).
    # For λ = 1 / |B|(ζ*), near |B|(ζ*), the quantity 1 / √(1 − λ |B|) is singular
    # with strength ~ 1 / |∂|B|/∂_ζ|. Therefore, a quadrature for ε should evaluate
    # the integrand near b = 1 / λ = |B|(ζ*) to capture the fat banana orbits.
    # Breakpoints where the quadrature should take more care.
    # For simple schemes this means to include a quadrature point here.
    breaks = get_extrema(**spline, sort=False).reshape(-1, num_rho, alpha.size)
    # Need to remove nan and include (soft)max_B regardless of whether any nan.
    max_B = (1 - 1e-6) * transforms["grid"].compress(data["max_tz |B|"])
    max_B = max_B[:, jnp.newaxis]
    breaks = jnp.where(jnp.isnan(breaks), max_B, breaks)
    breaks = jnp.vstack(
        [breaks, jnp.broadcast_to(max_B, (num_rho, alpha.size))[jnp.newaxis]]
    )
    breaks = jnp.sort(breaks, axis=0).reshape(breaks.shape[0], -1)

    try:
        is_Newton_Cotes = b_quad in [trapezoid, quadax.trapezoid, quadax.simpson]
        if is_Newton_Cotes:
            b = composite_linspace(breaks, b_quad_res)
            rip = b_quad(ripple_sum(b), b, axis=0)
        else:
            # use adaptive quadrature from quadax
            rip = b_quad(ripple_sum, breaks)
    except TypeError as e:
        raise NotImplementedError from e

    # Integrate over flux surface.
    ripple = rip.reshape(num_rho, -1) @ alpha_weight
    data["ripple"] = transforms["grid"].expand(ripple)
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
    fsa="bool : Whether to surface average to approximate the limit.",
    grid_fl="Grid : Field line grid.",
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """Evaluation of 1/ν neoclassical transport in stellarators.

    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    https://doi.org/10.1063/1.873749
    """
    g = kwargs["grid_fl"]
    if kwargs.get("fsa", False):
        l = g.nodes[g.unique_zeta_idx[-1], 2] - g.nodes[g.unique_zeta_idx[0], 2]
        V = data["V_r(r)"] / data["psi_r"] * l
        S = data["S(r)"] * l
    else:
        num_alpha = g.num_theta
        shape = (g.num_rho, num_alpha, g.num_zeta)
        z = jnp.reshape(g.nodes[:, 2], shape)
        v = jnp.reshape(1 / (data["B^zeta"] * data["|B|"]), shape)
        V = jnp.mean(quadax.simpson(v, z, axis=-1), axis=1)
        S = jnp.mean(
            quadax.simpson(v * data["|grad(psi)|"].reshape(shape), z, axis=-1),
            axis=1,
        )
        V, S = map(transforms["grid"].expand, (V, S))
        warnif(num_alpha != 1, msg=colored("Reduced via mean over alpha.", "yellow"))

    data["effective ripple"] = (
        jnp.pi * data["R0"] ** 2 / (8 * 2**0.5) * V / S**2 * data["ripple"]
    ) ** (2 / 3)
    return data
