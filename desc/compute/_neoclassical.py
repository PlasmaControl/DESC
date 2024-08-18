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

from orthax.legendre import leggauss
from quadax import simpson

from desc.backend import imap, jit, jnp

from .bounce_integral import bounce_integral, get_pitch
from .data_index import register_compute_fun
from .utils import safediv


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
        # Interpolate |∇ρ| κ_g since it is smoother than κ_g alone.
        H = bounce_integrate(
            dH,
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
