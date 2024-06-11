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
from orthax.legendre import leggauss
from termcolor import colored

from desc.backend import imap, jit, jnp, trapezoid

from ..utils import warnif
from .bounce_integral import bounce_integral, get_pitch
from .data_index import register_compute_fun


def _is_adaptive(quad):
    if hasattr(quad, "is_adaptive"):
        return quad.is_adaptive
    else:
        return quad not in [trapezoid, quadax.trapezoid, quadax.simpson]


def vec_quadax(quad):
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
    if not _is_adaptive(quad):
        return quad

    def vec_quad(fun, interval, B_sup_z, B, B_z_ra, arg1, arg2):
        return quad(fun, interval, args=(B_sup_z, B, B_z_ra, arg1, arg2))[0]

    vec_quad = jnp.vectorize(
        vec_quad, signature="(2),(m),(m),(m),(m),(m)->()", excluded={0}
    )
    return vec_quad


def _poloidal_mean(grid, f):
    """Integrate f over poloidal angle and divide by 2π."""
    assert f.shape[-1] == grid.num_poloidal
    if grid.num_poloidal == 1:
        return jnp.squeeze(f, axis=-1)
    if not hasattr(grid, "spacing"):
        warnif(True, msg=colored("Reduced via uniform poloidal mean.", "yellow"))
        return jnp.mean(f, axis=-1)
    assert grid.is_meshgrid
    dp = grid.compress(grid.spacing[:, 1], surface_label="poloidal")
    return f @ dp / jnp.sum(dp)


def _get_pitch(grid, data, quad, num):
    # Get endpoints of integral over pitch for each flux surface.
    # with num values uniformly spaced in between.
    min_B = grid.compress(data["min_tz |B|"])
    max_B = grid.compress(data["max_tz |B|"])
    if _is_adaptive(quad):
        pitch = 1 / jnp.stack([max_B, min_B], axis=-1)[:, jnp.newaxis]
        assert pitch.shape == (grid.num_rho, 1, 2)
    else:
        pitch = get_pitch(min_B, max_B, num)
        pitch = jnp.broadcast_to(
            pitch[..., jnp.newaxis], (pitch.shape[0], grid.num_rho, grid.num_alpha)
        ).reshape(pitch.shape[0], grid.num_rho * grid.num_alpha)
    return pitch


@register_compute_fun(
    name="<L|r,a>",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}"
    " \\frac{d\\zeta}{B^{\\zeta}}",
    units="m / T",
    units_long="Meter / tesla",
    description="(Mean) length along field line(s)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _L_ra_fsa(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    L_ra = quadax.simpson(
        jnp.reshape(1 / data["B^zeta"], shape),
        jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    )
    data["<L|r,a>"] = g.expand(_poloidal_mean(g, L_ra))
    return data


@register_compute_fun(
    name="<G|r,a>",
    label="\\int_{\\zeta_{\\mathrm{min}}}^{\\zeta_{\\mathrm{max}}"
    " \\frac{d\\zeta}{B^{\\zeta} \\sqrt g}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="(Mean) length over volume along field line(s)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B^zeta", "sqrt(g)"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _G_ra_fsa(data, transforms, profiles, **kwargs):
    g = transforms["grid"].source_grid
    shape = (g.num_rho, g.num_alpha, g.num_zeta)
    G_ra = quadax.simpson(
        jnp.reshape(1 / (data["B^zeta"] * data["sqrt(g)"]), shape),
        jnp.reshape(g.nodes[:, 2], shape),
        axis=-1,
    )
    data["<G|r,a>"] = g.expand(_poloidal_mean(g, G_ra))
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
        "<L|r,a>",
    ],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    num_quad=(
        "int : Resolution for quadrature of bounce integrals. Default is 31, "
        "which gets sufficient convergence, so higher values are likely unnecessary."
    ),
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate, preferably odd. "
        "Default is 125. Effective ripple will look smoother at high values. "
        "(If computed on many flux surfaces and micro oscillation is seen "
        "between neighboring surfaces, increasing num_pitch will smooth the profile)."
    ),
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
@partial(jit, static_argnames=["num_quad", "num_pitch"])
def _effective_ripple_raw(params, transforms, profiles, data, **kwargs):
    g = transforms["grid"].source_grid
    bounce_integrate, _ = bounce_integral(
        data["B^zeta"],
        data["|B|"],
        data["|B|_z|r,a"],
        knots=g.compress(g.nodes[:, 2], surface_label="zeta"),
        quad=leggauss(kwargs.get("num_quad", 31)),
    )

    def dH(grad_psi_norm, kappa_g, B, pitch):
        # Removed dimensionless (λB₀)¹ᐧ⁵ from integrand of Nemov equation 30.
        # Reintroduced later.
        return (
            jnp.sqrt(1 - pitch * B)
            * (4 / (pitch * B) - 1)
            * grad_psi_norm
            * kappa_g
            / B
        )

    def dI(B, pitch):  # Integrand of Nemov equation 31.
        return jnp.sqrt(1 - pitch * B) / B

    def d_ripple(pitch):
        """Return λ⁻²B₀⁻³ ∑ⱼ Hⱼ²/Iⱼ evaluated at λ = pitch.

        Parameters
        ----------
        pitch : Array, shape(*pitch.shape[:-1], g.num_rho * g.num_alpha)
            Pitch angle.

        Returns
        -------
        d_ripple : Array, shape(pitch.shape)
            λ⁻²B₀⁻³ ∑ⱼ Hⱼ²/Iⱼ

        """
        H = bounce_integrate(dH, [data["|grad(psi)|"], data["kappa_g"]], pitch)
        I = bounce_integrate(dI, [], pitch)
        return pitch * jnp.nansum(H**2 / I, axis=-1)
        # (λB₀)³ db = (λB₀)³ λ⁻² B₀⁻¹ (-dλ) = λ B₀² (-dλ) where B₀ has units of λ⁻¹.

    # The integrand is continuous and likely poorly approximated by a polynomial,
    # so composite quadrature should perform better than higher order methods.
    pitch = _get_pitch(g, data, quadax.simpson, kwargs.get("num_pitch", 125))
    ripple = quadax.simpson(jnp.squeeze(imap(d_ripple, pitch), axis=1), pitch, axis=0)
    data["effective ripple raw"] = (
        g.expand(_poloidal_mean(g, ripple.reshape(g.num_rho, g.num_alpha)))
        * data["max_tz |B|"] ** 2
        / data["<L|r,a>"]
    )
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
