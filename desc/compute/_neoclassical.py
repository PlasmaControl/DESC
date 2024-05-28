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

from desc.backend import jnp, trapezoid

from ..utils import warnif
from .bounce_integral import (
    affine_bijection,
    bounce_integral,
    composite_linspace,
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

    def vec_quad(fun, interval, B_sup_z, B, B_z_ra, arg1, arg2):
        return quad(fun, interval, args=(B_sup_z, B, B_z_ra, arg1, arg2))[0]

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
    parameterization of the field line length tends to infinity of an average
    along the field line will converge to a flux surface average.
    In theory, we can compute such quantities with averages over finite lengths
    of the field line, e.g. one toroidal transit, for many values of the poloidal
    field line label and then average this over the poloidal domain.

    This should also work for integrands which are bounce integrals;
    Since everything is continuous, as the number of nodes tend to infinity both
    approaches should converge to the same result, assuming irrational surface.
    However, the order at which all the bounce integrals detected over the surface
    are summed differs, so the convergence rate will differ. It is not tested whether
    such a method will converge to the limit faster than extending the length of the
    field line chunk.

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


@register_compute_fun(
    name="V_psi(r)*range(z)",
    label="\\int d \\ell / \\vert B \\vert",
    units="m^3 / Wb",
    units_long="Cubic meters per Weber",
    description=(
        "Volume enclosed by flux surfaces, derivative with respect to toroidal flux, "
        "computed along field line, scaled by dimensionless length of field line"
    ),
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta", "|B|"],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
)
def _V_psi_range_z(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    z = jnp.reshape(g.nodes[:, 2], shape)
    V = jnp.reshape(1 / (data["B^zeta"] * data["|B|"]), shape)
    V = jnp.mean(quadax.simpson(V, z, axis=-1), axis=1)
    warnif(g.num_alpha != 1, msg=colored("Reduced via mean over alpha.", "yellow"))
    data["V_psi(r)*range(z)"] = g.expand(V)
    return data


@register_compute_fun(
    name="S(r)*range(z)",
    label="\\int d \\ell \\vert \\nabla \\psi \\vert / \\vert B \\vert",
    units="m^2",
    units_long="Square meters",
    description=(
        "Surface area of flux surfaces, computed along field line, "
        "scaled by dimensionless length of field line"
    ),
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta", "|B|", "|grad(psi)|"],
    grid_requirement=[
        "source_grid",
        lambda grid: grid.source_grid.coordinates == "raz"
        and grid.source_grid.is_meshgrid,
    ],
)
def _S_range_z(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    z = jnp.reshape(g.nodes[:, 2], shape)
    S = jnp.reshape(data["|grad(psi)|"] / (data["B^zeta"] * data["|B|"]), shape)
    S = jnp.mean(quadax.simpson(S, z, axis=-1), axis=1)
    warnif(g.num_alpha != 1, msg=colored("Reduced via mean over alpha.", "yellow"))
    data["S(r)*range(z)"] = g.expand(S)
    return data


@register_compute_fun(
    name="effective ripple raw",
    label="∫ db ∑ⱼ Hⱼ² / Iⱼ",
    units="Wb / m",
    units_long="Webers per meter",
    description="Effective ripple modulation amplitude, not normalized",
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
        "callable : Method to compute bounce integrals. "
        "(You may want to wrap desc.compute.bounce_integral.bounce_integral "
        "to change optional parameters such as quadrature resolution, etc.)."
    ),
    batch="bool : Whether to perform computation in a batched manner.",
    quad=(
        "callable : Quadrature method over velocity space. "
        "Adaptive quadrature method from quadax must be wrapped with vec_quadax."
    ),
    quad_res="int : Resolution for quadrature over velocity space.",
    alpha_weight="Array : Quadrature weight over alpha.",
)
def _effective_ripple_raw(params, transforms, profiles, data, **kwargs):
    def dH(grad_psi_norm, cvdrift0, B, pitch, Z):
        return (
            pitch
            * jnp.sqrt(1 / pitch - B)
            * (4 / B - pitch)
            * grad_psi_norm
            * cvdrift0
            / B
        )

    def dI(B, pitch, Z):
        return jnp.sqrt(1 - pitch * B) / B

    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    _bounce_integral = kwargs.pop("bounce_integral", bounce_integral)
    batch = kwargs.pop("batch", True)
    quad = kwargs.pop("quad", quadax.simpson)
    quad_res = kwargs.pop("quad_res", 50)
    alpha_weight = jnp.atleast_1d(kwargs.pop("alpha_weight", 1 / g.num_alpha))
    # Get boundary of integral over pitch for each field line.
    min_B, max_B = map(g.compress, (data["min_tz |B|"], data["max_tz |B|"]))
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + 1e-6) * min_B
    max_B = (1 - 1e-6) * max_B
    boundary = jnp.stack([min_B, max_B])

    is_Newton_Cotes = quad in [trapezoid, quadax.trapezoid, quadax.simpson]
    if is_Newton_Cotes:
        bounce_integrate, _ = _bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots
        )

        def d_ripple(b):
            """Return ∑ⱼ Hⱼ² / Iⱼ evaluated at b.

            Parameters
            ----------
            b : Array, shape(*b.shape[:-1], g.num_rho * g.num_alpha)
                Multiplicative inverse of pitch angle.

            Returns
            -------
            rs : Array, shape(b.shape)
                ∑ⱼ Hⱼ² / Iⱼ

            """
            pitch = 1 / b
            H = bounce_integrate(
                dH, [data["|grad(psi)|"], data["cvdrift0"]], pitch, batch=batch
            )
            I = bounce_integrate(dI, [], pitch, batch=batch)
            return jnp.nansum(H**2 / I, axis=-1)

        b = composite_linspace(boundary, quad_res)
        b = jnp.broadcast_to(
            b[..., jnp.newaxis], (b.shape[0], g.num_rho, g.num_alpha)
        ).reshape(b.shape[0], g.num_rho * g.num_alpha)
        ripple = quad(d_ripple(b), b, axis=0)
    else:
        # Use adaptive quadrature.

        def d_ripple(b, B_sup_z, B, B_z_ra, grad_psi, cvdrift0):
            # Quadax requires scalar integration interval, so we need to return scalar.
            bounce_integrate, _ = _bounce_integral(B_sup_z, B, B_z_ra, knots)
            pitch = 1 / b
            H = bounce_integrate(dH, [grad_psi, cvdrift0], pitch, batch=batch)
            I = bounce_integrate(dI, [], pitch, batch=batch)
            return jnp.squeeze(jnp.nansum(H**2 / I, axis=-1))

        boundary = boundary.T[:, jnp.newaxis]
        assert boundary.shape == (g.num_rho, 1, 2)
        args = [
            f.reshape(g.num_rho, g.num_alpha, g.num_zeta)
            for f in [
                data["B^zeta"],
                data["|B|"],
                data["|B|_z|r,a"],
                data["|grad(psi)|"],
                data["cvdrift0"],
            ]
        ]
        ripple = quad(d_ripple, boundary, *args)

    # Integrate over flux surface.
    ripple = jnp.reshape(ripple, (g.num_rho, g.num_alpha)) @ alpha_weight
    data["effective ripple raw"] = g.expand(ripple)
    return data


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\text{eff}}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["effective ripple raw", "R0", "V_psi(r)*range(z)", "S(r)*range(z)"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """Evaluation of 1/ν neoclassical transport in stellarators.

    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    https://doi.org/10.1063/1.873749.
    """
    data["effective ripple"] = (
        jnp.pi
        * data["R0"] ** 2
        / (8 * 2**0.5)
        * data["V_psi(r)*range(z)"]
        / data["S(r)*range(z)"] ** 2
        * data["effective ripple raw"]
    ) ** (2 / 3)
    return data


@register_compute_fun(
    name="Gamma_c raw",
    label="∫ db ∑ⱼ (γ_c² * ∂I/∂b)ⱼ",
    units="m^3 / Wb",
    units_long="Cubic meters per Weber",
    description="Energetic ion confinement, not normalized",
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
        "cvdrift0",
        "gbdrift",
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
        "callable : Quadrature method over velocity space. "
        "Adaptive quadrature method from quadax must be wrapped with vec_quadax."
    ),
    quad_res="int : Resolution for quadrature over velocity space.",
    alpha_weight="Array : Quadrature weight over alpha.",
)
def _Gamma_c_raw(params, transforms, profiles, data, **kwargs):
    def d_gamma_c(f, B, pitch, Z):
        return f * (1 - pitch * B / 2) / jnp.sqrt(1 - pitch * B)

    def d_dI_db(B, pitch, Z):
        return pitch**2 / (2 * jnp.sqrt(1 - pitch * B))

    g = transforms["grid"].source_grid
    knots = g.compress(g.nodes[:, 2], surface_label="zeta")
    _bounce_integral = kwargs.pop("bounce_integral", bounce_integral)
    batch = kwargs.pop("batch", True)
    quad = kwargs.pop("quad", quadax.simpson)
    quad_res = kwargs.pop("quad_res", 50)
    alpha_weight = jnp.atleast_1d(kwargs.pop("alpha_weight", 1 / g.num_alpha))
    # Get boundary of integral over pitch for each field line.
    min_B, max_B = map(g.compress, (data["min_tz |B|"], data["max_tz |B|"]))
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + 1e-6) * min_B
    max_B = (1 - 1e-6) * max_B
    boundary = jnp.stack([min_B, max_B])

    is_Newton_Cotes = quad in [trapezoid, quadax.trapezoid, quadax.simpson]
    if is_Newton_Cotes:
        bounce_integrate, _ = _bounce_integral(
            data["B^zeta"], data["|B|"], data["|B|_z|r,a"], knots
        )

        def d_Gamma_c_raw(b):
            """Return ∑ⱼ (γ_c² * ∂I/∂b)ⱼ evaluated at b.

            Parameters
            ----------
            b : Array, shape(*b.shape[:-1], g.num_rho * g.num_alpha)
                Multiplicative inverse of pitch angle.

            Returns
            -------
            rs : Array, shape(b.shape)
                ∑ⱼ (γ_c² * ∂I/∂b)ⱼ

            """
            pitch = 1 / b
            # todo: add Nemov's |grad(psi)|
            gamma_c = (
                2
                / jnp.pi
                * jnp.arctan(
                    bounce_integrate(d_gamma_c, data["cvdrift0"], pitch, batch=batch)
                    / bounce_integrate(d_gamma_c, data["gbdrift"], pitch, batch=batch)
                )
            )
            dI_db = bounce_integrate(d_dI_db, [], pitch, batch=batch)
            return jnp.nansum(gamma_c**2 * dI_db, axis=-1)

        b = composite_linspace(boundary, quad_res)
        b = jnp.broadcast_to(
            b[..., jnp.newaxis], (b.shape[0], g.num_rho, g.num_alpha)
        ).reshape(b.shape[0], g.num_rho * g.num_alpha)
        Gamma_c_raw = quad(d_Gamma_c_raw(b), b, axis=0)
    else:
        # Use adaptive quadrature.

        def d_Gamma_c_raw(b, B_sup_z, B, B_z_ra, cvdrift0, gbdrift):
            # Quadax requires scalar integration interval, so we need to return scalar.
            bounce_integrate, _ = _bounce_integral(B_sup_z, B, B_z_ra, knots)
            pitch = 1 / b
            # todo: add Nemov's |grad(psi)|
            gamma_c = (
                2
                / jnp.pi
                * jnp.arctan(
                    bounce_integrate(d_gamma_c, cvdrift0, pitch, batch=batch)
                    / bounce_integrate(d_gamma_c, gbdrift, pitch, batch=batch)
                )
            )
            dI_db = bounce_integrate(d_dI_db, [], pitch, batch=batch)
            return jnp.squeeze(jnp.nansum(gamma_c**2 * dI_db, axis=-1))

        boundary = boundary.T[:, jnp.newaxis]
        assert boundary.shape == (g.num_rho, 1, 2)
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
        Gamma_c_raw = quad(d_Gamma_c_raw, boundary, *args)

    # Integrate over flux surface.
    Gamma_c_raw = jnp.reshape(Gamma_c_raw, (g.num_rho, g.num_alpha)) @ alpha_weight
    data["Gamma_c raw"] = g.expand(Gamma_c_raw)
    return data


@register_compute_fun(
    name="Gamma_c",
    label="\\Gamma_{c}",
    units="~",
    units_long="None",
    description="Energetic ion confinement",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["Gamma_c raw", "V_psi(r)*range(z)"],
)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Poloidal motion of trapped particle orbits in real-space coordinates.

    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    """
    data["Gamma_c"] = (
        jnp.pi / (2**1.5) * data["Gamma_c raw"] / data["V_psi(r)*range(z)"]
    )
    return data
