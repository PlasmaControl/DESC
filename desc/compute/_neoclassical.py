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

from ..integrals.bounce_integral import Bounce1D
from ..integrals.bounce_utils import get_pitch_inv_quad, interp_to_argmin
from ..integrals.quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from ..utils import cross, dot, safediv
from .data_index import register_compute_fun

_bounce_doc = {
    "quad": (
        "tuple[jnp.ndarray] : Quadrature points and weights for bounce integrals. "
        "Default option is well tested."
    ),
    "num_quad": (
        "int : Resolution for quadrature of bounce integrals. "
        "Default is 32. This option is ignored if given ``quad``."
    ),
    "num_pitch": "int : Resolution for quadrature over velocity coordinate.",
    "num_well": (
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance."
    ),
    "batch": "bool : Whether to vectorize part of the computation. Default is true.",
}


def _alpha_mean(f):
    """Simple mean over field lines.

    Simple mean rather than integrating over α and dividing by 2π
    (i.e. f.T.dot(dα) / dα.sum()), because when the toroidal angle extends
    beyond one transit we need to weight all field lines uniformly, regardless
    of their spacing wrt α.
    """
    return f.mean(axis=0)


def _compute(fun, interp_data, data, grid, num_pitch, reduce=True):
    """Compute ``fun`` for each α and ρ value iteratively to reduce memory usage.

    Parameters
    ----------
    fun : callable
        Function to compute.
    interp_data : dict[str, jnp.ndarray]
        Data to provide to ``fun``.
        Names in ``Bounce1D.required_names`` will be overridden.
        Reshaped automatically.
    data : dict[str, jnp.ndarray]
        DESC data dict.
    reduce : bool
        Whether to compute mean over α and expand to grid.
        Default is true.

    """
    pitch_inv, pitch_inv_weight = get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
    )

    def for_each_rho(x):
        # using same λ values for every field line α on flux surface ρ
        x["pitch_inv"] = pitch_inv
        x["pitch_inv weight"] = pitch_inv_weight
        return imap(fun, x)

    for name in Bounce1D.required_names:
        interp_data[name] = data[name]
    interp_data = dict(
        zip(interp_data.keys(), Bounce1D.reshape_data(grid, *interp_data.values()))
    )
    out = imap(for_each_rho, interp_data)
    return grid.expand(_alpha_mean(out)) if reduce else out


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
    **_bounce_doc,
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
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_well", "batch"])
def _epsilon_32(params, transforms, profiles, data, **kwargs):
    """https://doi.org/10.1063/1.873749.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    # noqa: unused dependency
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = chebgauss2(kwargs.get("num_quad", 32))
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

    def eps_32(data):
        """(∂ψ/∂ρ)⁻² B₀⁻² ∫ dλ λ⁻² ∑ⱼ Hⱼ²/Iⱼ."""
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

    # Interpolate |∇ρ| κ_g since it is smoother than κ_g alone.
    interp_data = {"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]}
    B0 = data["max_tz |B|"]
    data["effective ripple 3/2"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * _compute(eps_32, interp_data, data, grid, kwargs.get("num_pitch", 50))
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
    transforms={},
    profiles=[],
    coordinates="r",
    data=["effective ripple 3/2"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    data["effective ripple"] = data["effective ripple 3/2"] ** (2 / 3)
    return data


@register_compute_fun(
    name="Gamma_c Velasco",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "cvdrift0", "gbdrift", "<L|r,a>"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    **_bounce_doc,
)
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_well", "batch"])
def _Gamma_c_Velasco(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Velasco et al.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.
    """
    # noqa: unused dependency
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    grid = transforms["grid"].source_grid

    def d_v_tau(B, pitch):
        return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))

    def drift(f, B, pitch):
        return safediv(f * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B)))

    def Gamma_c_Velasco(data):
        """∫ dλ ∑ⱼ [v τ γ_c²]ⱼ."""
        bounce = Bounce1D(grid, data, quad, automorphism=None, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau = bounce.integrate(d_v_tau, data["pitch_inv"], points=points, batch=batch)
        gamma_c = jnp.arctan(
            safediv(
                bounce.integrate(
                    drift,
                    data["pitch_inv"],
                    data["cvdrift0"],
                    points=points,
                    batch=batch,
                ),
                bounce.integrate(
                    drift,
                    data["pitch_inv"],
                    data["gbdrift"],
                    points=points,
                    batch=batch,
                ),
            )
        )
        return (4 / jnp.pi**2) * (
            (v_tau * gamma_c**2).sum(axis=-1)
            * data["pitch_inv"] ** (-2)
            * data["pitch_inv weight"]
        ).sum(axis=-1)

    interp_data = {"cvdrift0": data["cvdrift0"], "gbdrift": data["gbdrift"]}
    data["Gamma_c Velasco"] = (
        jnp.pi
        / (8 * 2**0.5)
        * _compute(
            Gamma_c_Velasco, interp_data, data, grid, kwargs.get("num_pitch", 64)
        )
        / data["<L|r,a>"]
    )
    return data


@register_compute_fun(
    name="Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy, Nemov et al.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^phi",
        "B^phi_r|v,p",
        "b",
        "|B|_r|v,p",
        "<L|r,a>",
        "iota_r",
        "grad(phi)",
        "e^rho",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "psi_r",
    ]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    **_bounce_doc,
    quad2="Same as ``quad`` for the weak singular integrals in particular.",
)
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_well", "batch"])
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Nemov et al.

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61.

    The radial electric field has a negligible effect on alpha particle confinement,
    so it is assumed to be zero.
    """
    # noqa: unused dependency
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    quad2 = kwargs["quad2"] if "quad2" in kwargs else chebgauss2(quad[0].size)
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    grid = transforms["grid"].source_grid

    # The derivative (∂/∂ψ)|ϑ,ϕ belongs to flux coordinates which satisfy
    # α = ϑ − χ(ψ) ϕ where α is the poloidal label of ψ,α Clebsch coordinates.
    # Choosing χ = ι implies ϑ, ϕ are PEST angles.
    # ∂G/∂((λB₀)⁻¹) =     λ²B₀  ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) ∂|B|/∂ψ / |B|
    # ∂V/∂((λB₀)⁻¹) = 3/2 λ²B₀  ∫ dℓ √(1 − λ|B|) K / |B|
    # ∂g/∂((λB₀)⁻¹) =     λ²B₀² ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
    # tan(π/2 γ_c) =
    #              ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ρ| κ_g / |B|
    #              ----------------------------------------------
    # (|∇ρ| ‖e_α|ρ,ϕ‖)ᵢ ∫ dℓ [ (1 − λ|B|/2)/√(1 − λ|B|) ∂|B|/∂ψ + √(1 − λ|B|) K ] / |B|

    # Note that we rewrite equivalents of Nemov et al.'s expression's using
    # single valued maps of a physical coordinates. This avoids the computational
    # issues of multivalued maps. It further enables use of more efficient methods,
    # such as fast transforms and fixed computational grids throughout optimization,
    # which are used in the ``Bounce2D`` operator on a developer branch.

    def d_v_tau(B, pitch):
        return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))

    def drift1(grad_rho_norm_kappa_g, B, pitch):
        return (
            safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
            * grad_rho_norm_kappa_g
            / B
        )

    def drift2(B_psi, B, pitch):
        return (
            safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B))) * B_psi / B
        )

    def drift3(K, B, pitch):
        return jnp.sqrt(jnp.abs(1 - pitch * B)) * K / B

    def Gamma_c(data):
        """∫ dλ ∑ⱼ [v τ γ_c²]ⱼ."""
        # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
        # τ is the bounce time, and I is defined in Nemov eq. 36.
        bounce = Bounce1D(grid, data, quad, automorphism=None, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau = bounce.integrate(d_v_tau, data["pitch_inv"], points=points, batch=batch)
        gamma_c = jnp.arctan(
            safediv(
                bounce.integrate(
                    drift1,
                    data["pitch_inv"],
                    data["|grad(rho)|*kappa_g"],
                    points=points,
                    batch=batch,
                ),
                (
                    bounce.integrate(
                        drift2,
                        data["pitch_inv"],
                        data["|B|_psi|v,p"],
                        points=points,
                        batch=batch,
                    )
                    + bounce.integrate(
                        drift3,
                        data["pitch_inv"],
                        data["K"],
                        points=points,
                        batch=batch,
                        quad=quad2,
                    )
                )
                * interp_to_argmin(
                    data["|grad(rho)|*|e_alpha|r,p|"],
                    points,
                    bounce.zeta,
                    bounce.B,
                    bounce.dB_dz,
                ),
            )
        )
        return (4 / jnp.pi**2) * (
            (v_tau * gamma_c**2).sum(axis=-1)
            * data["pitch_inv"] ** (-2)
            * data["pitch_inv weight"]
        ).sum(axis=-1)

    interp_data = {
        "|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_psi|v,p": data["|B|_r|v,p"] / data["psi_r"],
        # TODO: Confirm if K is smoother than individual components.
        #  If not, should spline separately.
        "K": data["iota_r"] * dot(cross(data["e^rho"], data["b"]), data["grad(phi)"])
        # Behaves as log derivative if one ignores the issue of an argument with units.
        # Smoothness determined by + lower bound of argument ∂log(|B|²/B^ϕ)/∂ψ |B|.
        # Note that Nemov assumes B^ϕ > 0; this is not true in DESC, but we account
        # for that in this computation.
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"])
        / data["psi_r"],
    }
    data["Gamma_c"] = (
        jnp.pi
        / (8 * 2**0.5)
        * _compute(Gamma_c, interp_data, data, grid, kwargs.get("num_pitch", 64))
        / data["<L|r,a>"]
    )
    return data
