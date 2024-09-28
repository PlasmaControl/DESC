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

from desc.backend import imap, jit, jnp

from ..integrals.bounce_integral import Bounce1D
from ..integrals.bounce_utils import get_pitch_inv_quad
from ..integrals.quad_utils import chebgauss2
from ..utils import safediv
from .data_index import register_compute_fun


def _alpha_mean(f):
    """Simple mean over field lines.

    Simple mean rather than integrating over α and dividing by 2π
    (i.e. f.T.dot(dα) / dα.sum()), because when the toroidal angle extends
    beyond one transit we need to weight all field lines uniformly, regardless
    of their spacing wrt α.
    """
    return f.mean(axis=0)


def _get_pitch_inv_quad(grid, data, num_pitch, _data):
    p, w = get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]), grid.compress(data["max_tz |B|"]), num_pitch
    )
    _data["pitch_inv"] = jnp.broadcast_to(
        p[jnp.newaxis], (grid.num_alpha, grid.num_rho, num_pitch)
    )
    _data["pitch_inv weight"] = jnp.broadcast_to(
        w[jnp.newaxis], (grid.num_alpha, grid.num_rho, num_pitch)
    )
    return _data


def map2(fun, xs, *, batch_size=None):
    """Map over leading two axes iteratively."""
    # Can't pass in batch_size to imap yet because only new version jax allow that.
    return imap(lambda x: imap(fun, x), xs)


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
    data["<L|r,a>"] = grid.expand(jnp.abs(_alpha_mean(L_ra)))
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
    data["<G|r,a>"] = grid.expand(jnp.abs(_alpha_mean(G_ra)))
    return data


@register_compute_fun(
    name="effective ripple 3/2",
    label=(
        # ε¹ᐧ⁵ = π/(8√2) R₀²〈|∇ψ|〉⁻² B₀⁻¹ ∫dλ λ⁻² 〈 ∑ⱼ Hⱼ²/Iⱼ 〉
        "\\epsilon_{\\mathrm{eff}}^{3/2} = \\frac{\\pi}{8 \\sqrt{2}} "
        "R_0^2 \\langle \\vert\\nabla \\psi\\vert \\rangle^{-2} "
        "B_0^{-1} \\int d\\lambda \\lambda^{-2} "
        "\\langle \\sum_j H_j^2 / I_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude to 3/2 power",
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
    quad="jnp.ndarray : Optional, quadrature points and weights for bounce integrals.",
    num_pitch="int : Resolution for quadrature over velocity coordinate. Default 50.",
    num_well=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance."
    ),
    batch="bool : Whether to vectorize part of the computation. Default is true.",
    # Some notes on choosing the resolution hyperparameters:
    # The default settings were chosen such that the effective ripple profile on
    # the W7-X stellarator looks similar to the profile computed at higher resolution,
    # indicating convergence. The parameters ``num_transit`` and ``knots_per_transit``
    # have a stronger effect on the result. As a reference for W7-X, when computing the
    # effective ripple by tracing a single field line on each flux surface, a density of
    # 100 knots per toroidal transit accurately reconstructs the ripples along the field
    # line. After 10 toroidal transits convergence is apparent (after 15 the returns
    # diminish). Dips in the resulting profile indicates insufficient ``num_transit``.
    # Unreasonably high values indicates insufficient ``knots_per_transit``.
    # One can plot the field line with ``Bounce1D.plot`` to see if the number of knots
    # was sufficient to reconstruct the field line.
    # TODO: Improve performance... see GitHub issue #1045.
    #  Need more efficient function approximation of |B|(α, ζ).
)
@partial(jit, static_argnames=["num_pitch", "num_well", "batch"])
def _epsilon_32(params, transforms, profiles, data, **kwargs):
    """https://doi.org/10.1063/1.873749.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    quad = kwargs["quad"] if "quad" in kwargs else chebgauss2(32)
    num_pitch = kwargs.get("num_pitch", 50)
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
        """Return (∂ψ/∂ρ)⁻² B₀⁻² ∫ dλ λ⁻² ∑ⱼ Hⱼ²/Iⱼ."""
        # B₀ has units of λ⁻¹.
        # Nemov's ∑ⱼ Hⱼ²/Iⱼ = (∂ψ/∂ρ)² (λB₀)³ ``(H**2 / I).sum(axis=-1)``.
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        bounce = Bounce1D(grid, data, quad, automorphism=None, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        H = bounce.integrate(
            dH,
            data["pitch_inv"],
            data["|grad(rho)|*kappa_g"],
            points=points,
            batch=batch,
        )
        I = bounce.integrate(dI, data["pitch_inv"], points=points, batch=batch)
        return (
            safediv(H**2, I).sum(axis=-1)
            * data["pitch_inv"] ** (-3)
            * data["pitch_inv weight"]
        ).sum(axis=-1)

    _data = {  # noqa: unused dependency
        name: Bounce1D.reshape_data(grid, data[name])
        for name in Bounce1D.required_names
    }
    # Interpolate |∇ρ| κ_g since it is smoother than κ_g alone.
    _data["|grad(rho)|*kappa_g"] = Bounce1D.reshape_data(
        grid, data["|grad(rho)|"] * data["kappa_g"]
    )
    _data = _get_pitch_inv_quad(grid, data, num_pitch, _data)
    B0 = data["max_tz |B|"]
    data["effective ripple 3/2"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * grid.expand(_alpha_mean(map2(compute, _data)))
        / data["<L|r,a>"]
    )
    return data


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["effective ripple 3/2"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    data["effective ripple"] = data["effective ripple 3/2"] ** (2 / 3)
    return data
