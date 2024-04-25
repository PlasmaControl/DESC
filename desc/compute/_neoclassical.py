"""Compute functions for neoclassical transport objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from orthax import legendre

from desc.backend import jnp, trapezoid

from .bounce_integral import (
    affine_bijection_reverse,
    bounce_integral,
    composite_linspace,
    grad_affine_bijection_reverse,
    pitch_of_extrema,
)
from .data_index import register_compute_fun


def _dH(grad_psi_norm, cvdrift0, B, pitch, Z):
    return (
        pitch * jnp.sqrt(1 / pitch - B) * (4 / B - pitch) * grad_psi_norm * cvdrift0 / B
    )


def _dI(B, pitch, Z):
    return jnp.sqrt(1 - pitch * B) / B


def _alpha_leggauss(resolution, a_min=0, a_max=2 * jnp.pi):
    # Set resolution > 1 to see if long field line integral can be approximated
    # by flux surface average of field line integrals over finite transits.
    x, w = legendre.leggauss(resolution)
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
)
def _ripple(params, transforms, profiles, data, **kwargs):
    ripple_quad = kwargs.pop("ripple quad", trapezoid)
    ripple_quad_res = kwargs.pop("ripple quad resolution", 19)
    relative_shift = kwargs.pop("relative shift", 1e-6)

    grid_desc = transforms["grid"]
    grid_fl = kwargs.pop("grid_fl")
    num_rho = grid_fl.num_rho
    alpha = grid_fl.compress(grid_fl.nodes[:, 1], surface_label="theta")
    knots = grid_fl.compress(grid_fl.nodes[:, 2], surface_label="zeta")
    alpha_weight = kwargs.pop("alpha weight", 2 * jnp.pi / alpha.size)
    bounce_integrate, spline = bounce_integral(
        data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots, **kwargs
    )

    def ripple_sum(b):
        """Return the ripple sum ∑ⱼ Hⱼ² / Iⱼ evaluated at b.

        Parameters
        ----------
        b : Array, shape(..., rho.size * alpha.size)
            Multiplicative inverse of pitch angle.

        Returns
        -------
        ripple_sum : Array, shape(..., rho.size * alpha.size)
            ∑ⱼ Hⱼ² / Iⱼ except the sum over j is split across alpha.

        """
        pitch = 1 / b
        H = bounce_integrate(_dH, [data["|grad(psi)|"], data["cvdrift0"]], pitch)
        I = bounce_integrate(_dI, [], pitch)
        return jnp.nansum(H**2 / I, axis=-1)

    # Breakpoints where the quadrature should take more care.
    # For simple schemes this means to include a quadrature point here.
    breaks = 1 / pitch_of_extrema(
        spline["knots"], spline["B.c"], spline["B_z_ra.c"], relative_shift
    ).reshape(-1, num_rho, alpha.size)
    max_tz_B = (1 - relative_shift) * grid_desc.compress(data["max_tz |B|"])
    max_tz_B = jnp.broadcast_to(max_tz_B[:, jnp.newaxis], (num_rho, alpha.size))
    breaks = jnp.vstack([breaks, max_tz_B[jnp.newaxis]])
    breaks = jnp.sort(breaks, axis=0).reshape(breaks.shape[0], -1)

    if ripple_quad == trapezoid:  # or quadax.simpson
        b = composite_linspace(breaks, ripple_quad_res)
        rip = ripple_quad(ripple_sum(b), b, axis=0)
    else:
        # want to use Clenshaw-Curtis with quadax.quadcc(ripple_sum, breaks)
        raise NotImplementedError("Dependency conflict with quadax.")

    # Integrate over flux surface.
    ripple = jnp.dot(rip.reshape(num_rho, -1), alpha_weight)
    data["ripple"] = grid_desc.expand(ripple)
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
    data=["ripple", "psi_r", "S(r)", "V_r(r)", "R"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    # V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    # Evaluation of 1/ν neoclassical transport in stellarators.
    # Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    # https://doi.org/10.1063/1.873749.
    data["effective ripple"] = (
        jnp.pi
        * data["R"] ** 2
        / (8 * 2**0.5)
        * (data["V_r(r)"] / data["psi_r"])
        / data["S(r)"] ** 2
        * data["ripple"]
    ) ** (2 / 3)
    return data
