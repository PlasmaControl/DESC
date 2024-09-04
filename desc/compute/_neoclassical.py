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

from quadax import simpson

from desc.backend import jit, jnp

from ..integrals.bounce_integral import Bounce1D
from ..integrals.quad_utils import leggauss_lob
from ..utils import map2, safediv
from .data_index import register_compute_fun
from .utils import _get_pitch_inv, _poloidal_mean


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
    resolution_requirement="z",
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _L_ra_fsa(data, transforms, profiles, **kwargs):
    grid = transforms["grid"].source_grid
    L_ra = simpson(
        y=grid.meshgrid_reshape(1 / data["B^zeta"], "arz"),
        x=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
        axis=-1,
    )
    data["<L|r,a>"] = grid.expand(jnp.abs(_poloidal_mean(grid, L_ra)))
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
    resolution_requirement="z",
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _G_ra_fsa(data, transforms, profiles, **kwargs):
    grid = transforms["grid"].source_grid
    G_ra = simpson(
        y=grid.meshgrid_reshape(1 / (data["B^zeta"] * data["sqrt(g)"]), "arz"),
        x=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
        axis=-1,
    )
    data["<G|r,a>"] = grid.expand(jnp.abs(_poloidal_mean(grid, G_ra)))
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
        "|grad(rho)|",
        "kappa_g",
        "<L|r,a>",
        "R0",
        "<|grad(rho)|>",
    ]
    + Bounce1D.required_names,
    resolution_requirement="z",
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    num_quad="int : Resolution for quadrature of bounce integrals. Default is 32.",
    num_pitch=(
        "int : Resolution for quadrature over velocity coordinate, preferably odd. "
        "Default is 125. Profile will look smoother at high values. "
        "(If computed on many flux surfaces and small oscillations is seen "
        "between neighboring surfaces, increasing this will smooth the profile)."
    ),
    num_well=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance."
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
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_well", "batch"])
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """https://doi.org/10.1063/1.873749.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    quad = leggauss_lob(kwargs.get("num_quad", 32))
    num_pitch = kwargs.get("num_pitch", 125)
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    grid = transforms["grid"].source_grid

    def dH(grad_rho_norm_kappa_g, B, pitch):
        # Integrand of Nemov eq. 30 with |∂ψ/∂ρ| (λB₀)¹ᐧ⁵ removed.
        return (
            jnp.sqrt(jnp.abs(1 - pitch * B))
            * (4 / (pitch * B) - 1)
            * grad_rho_norm_kappa_g
            / B
        )

    def dI(B, pitch):
        # Integrand of Nemov eq. 31.
        return jnp.sqrt(jnp.abs(1 - pitch * B)) / B

    def compute(data):
        """(∂ψ/∂ρ)⁻² B₀⁻² ∫ dλ ∑ⱼ Hⱼ²/Iⱼ."""
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        # Interpolate |∇ρ| κ_g since it is smoother than κ_g alone.
        H = bounce.integrate(
            dH,
            data["pitch_inv"],
            data["|grad(rho)|*kappa_g"],
            num_well=num_well,
            batch=batch,
        )
        I = bounce.integrate(dI, data["pitch_inv"], num_well=num_well, batch=batch)
        # Note B₀ has units of λ⁻¹.
        # Nemov's ∑ⱼ Hⱼ²/Iⱼ = (∂ψ/∂ρ)² (λB₀)³ ``(H**2 / I).sum(axis=-1)``.
        # (λB₀)³ db = λ³B₀² d(λ⁻¹) = λB₀² (-dλ).
        y = data["pitch_inv"] ** (-3) * safediv(H**2, I).sum(axis=-1)
        return simpson(y=y, x=data["pitch_inv"])
        # TODO: Try Gauss-Chebyshev quadrature after automorphism arcsin to
        #   make nodes more evenly spaced.

    _data = {  # noqa: unused dependency
        name: Bounce1D.reshape_data(grid, data[name])
        for name in Bounce1D.required_names
    }
    _data["|grad(rho)|*kappa_g"] = Bounce1D.reshape_data(
        grid, data["|grad(rho)|"] * data["kappa_g"]
    )
    _data["pitch_inv"] = _get_pitch_inv(grid, data, num_pitch)
    out = _poloidal_mean(grid, map2(compute, _data))
    data["effective ripple"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (data["max_tz |B|"] * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * grid.expand(out)
        / data["<L|r,a>"]
    )
    return data
