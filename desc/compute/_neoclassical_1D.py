"""Deprecated compute functions for neoclassical transport."""

from functools import partial

from orthax.legendre import leggauss
from quadax import simpson

from desc.backend import imap, jit, jnp

from ..integrals._bounce_utils import interp_to_argmin
from ..integrals._quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from ..integrals.bounce_integral import Bounce1D
from ..utils import cross, dot, safediv
from ._neoclassical import _Bounce2D_doc
from .data_index import register_compute_fun

_Bounce1D_doc = {
    "quad": _Bounce2D_doc["quad"],
    "num_quad": _Bounce2D_doc["num_quad"],
    "num_pitch": _Bounce2D_doc["num_pitch"],
    "num_well": _Bounce2D_doc["num_well"],
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


def _compute(fun, fun_data, data, grid, num_pitch, reduce=True):
    """Compute ``fun`` for each α and ρ value iteratively to reduce memory usage.

    Parameters
    ----------
    fun : callable
        Function to compute.
    fun_data : dict[str, jnp.ndarray]
        Data to provide to ``fun``.
        Names in ``Bounce1D.required_names`` will be overridden.
        Reshaped automatically.
    data : dict[str, jnp.ndarray]
        DESC data dict.
    reduce : bool
        Whether to compute mean over α and expand to grid.
        Default is true.

    """
    pitch_inv, pitch_inv_weight = Bounce1D.get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
    )

    def foreach_rho(x):
        # using same λ values for every field line α on flux surface ρ
        x["pitch_inv"] = pitch_inv
        x["pitch_inv weight"] = pitch_inv_weight
        return imap(fun, x)

    for name in Bounce1D.required_names:
        fun_data[name] = data[name]
    fun_data = dict(
        zip(fun_data.keys(), Bounce1D.reshape_data(grid, *fun_data.values()))
    )
    out = imap(foreach_rho, fun_data)
    return grid.expand(_alpha_mean(out)) if reduce else out


@register_compute_fun(
    name="fieldline length",
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
def _fieldline_length(data, transforms, profiles, **kwargs):
    grid = transforms["grid"].source_grid
    L_ra = simpson(
        y=grid.meshgrid_reshape(1 / data["B^zeta"], "arz"),
        x=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
        axis=-1,
    )
    data["fieldline length"] = grid.expand(jnp.abs(_alpha_mean(L_ra)))
    return data


@register_compute_fun(
    name="fieldline length/volume",
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
def _fieldline_length_over_volume(data, transforms, profiles, **kwargs):
    grid = transforms["grid"].source_grid
    G_ra = simpson(
        y=grid.meshgrid_reshape(1 / (data["B^zeta"] * data["sqrt(g)"]), "arz"),
        x=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
        axis=-1,
    )
    data["fieldline length/volume"] = grid.expand(jnp.abs(_alpha_mean(G_ra)))
    return data


def _dH(grad_rho_norm_kappa_g, B, pitch):
    """Integrand of Nemov eq. 30 with |∂ψ/∂ρ| (λB₀)¹ᐧ⁵ removed."""
    return (
        jnp.sqrt(jnp.abs(1 - pitch * B))
        * (4 / (pitch * B) - 1)
        * grad_rho_norm_kappa_g
        / B
    )


def _dI(B, pitch):
    """Integrand of Nemov eq. 31."""
    return jnp.sqrt(jnp.abs(1 - pitch * B)) / B


@register_compute_fun(
    name="deprecated(effective ripple 3/2)",
    label=(
        # ε¹ᐧ⁵ = π/(8√2) R₀²〈|∇ψ|〉⁻² B₀⁻¹ ∫dλ λ⁻² 〈 ∑ⱼ Hⱼ²/Iⱼ 〉
        "\\epsilon_{\\mathrm{eff}}^{3/2} = \\frac{\\pi}{8 \\sqrt{2}} "
        "R_0^2 \\langle \\vert\\nabla \\psi\\vert \\rangle^{-2} "
        "B_0^{-1} \\int d\\lambda \\lambda^{-2} "
        "\\langle \\sum_j H_j^2 / I_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude to 3/2 power. "
    "Uses numerical methods of the Bounce1D class.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "kappa_g",
        "R0",
        "|grad(rho)|",
        "<|grad(rho)|>",
        "fieldline length",
    ]
    + Bounce1D.required_names,
    resolution_requirement="z",
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    **_Bounce1D_doc,
)
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_well", "batch"])
def _epsilon_32_1D(params, transforms, profiles, data, **kwargs):
    """https://doi.org/10.1063/1.873749.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    # noqa: unused dependency
    num_pitch = kwargs.get("num_pitch", 50)
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = chebgauss2(kwargs.get("num_quad", 32))

    def eps_32(data):
        """(∂ψ/∂ρ)⁻² B₀⁻² ∫ dλ λ⁻² ∑ⱼ Hⱼ²/Iⱼ."""
        # B₀ has units of λ⁻¹.
        # Nemov's ∑ⱼ Hⱼ²/Iⱼ = (∂ψ/∂ρ)² (λB₀)³ ``(H**2 / I).sum(axis=-1)``.
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        bounce = Bounce1D(grid, data, quad, automorphism=None, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        H = bounce.integrate(
            _dH,
            data["pitch_inv"],
            data["|grad(rho)|*kappa_g"],
            points=points,
            batch=batch,
        )
        I = bounce.integrate(_dI, data["pitch_inv"], points=points, batch=batch)
        return (
            safediv(H**2, I).sum(axis=-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 3
        ).sum(axis=-1)

    grid = transforms["grid"].source_grid
    B0 = data["max_tz |B|"]
    data["deprecated(effective ripple 3/2)"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * _compute(
            eps_32,
            fun_data={"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]},
            data=data,
            grid=grid,
            num_pitch=num_pitch,
        )
        / data["fieldline length"]
    )
    return data


@register_compute_fun(
    name="deprecated(effective ripple)",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude. "
    "Uses numerical methods of the Bounce1D class.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["deprecated(effective ripple 3/2)"],
)
def _effective_ripple_1D(params, transforms, profiles, data, **kwargs):
    data["deprecated(effective ripple)"] = data["deprecated(effective ripple 3/2)"] ** (
        2 / 3
    )
    return data


def _v_tau(B, pitch):
    return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


def _f1(grad_psi_norm_kappa_g, B, pitch):
    return (
        safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
        * grad_psi_norm_kappa_g
        / B
    )


def _f2(B_r, B, pitch):
    return safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B))) * B_r / B


def _f3(K, B, pitch):
    return jnp.sqrt(jnp.abs(1 - pitch * B)) * K / B


@register_compute_fun(
    name="deprecated(Gamma_c)",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy, Nemov et al. "
    "Uses the numerical methods of the Bounce1D class.",
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
        "|B|_r|v,p",
        "b",
        "grad(phi)",
        "grad(psi)",
        "|grad(psi)|",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "iota_r",
        "fieldline length",
    ]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    **_Bounce1D_doc,
    quad2="Same as ``quad`` for the weak singular integrals in particular.",
)
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_well", "batch"])
def _Gamma_c_1D(params, transforms, profiles, data, **kwargs):
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
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    quad2 = kwargs["quad2"] if "quad2" in kwargs else chebgauss2(quad[0].size)

    def Gamma_c(data):
        """∫ dλ ∑ⱼ [v τ γ_c²]ⱼ."""
        bounce = Bounce1D(grid, data, quad, automorphism=None, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau = bounce.integrate(_v_tau, data["pitch_inv"], points=points, batch=batch)
        gamma_c = jnp.arctan(
            safediv(
                bounce.integrate(
                    _f1,
                    data["pitch_inv"],
                    data["|grad(psi)|*kappa_g"],
                    points=points,
                    batch=batch,
                ),
                (
                    bounce.integrate(
                        _f2,
                        data["pitch_inv"],
                        data["|B|_r|v,p"],
                        points=points,
                        batch=batch,
                    )
                    + bounce.integrate(
                        _f3,
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
        return (
            (v_tau * gamma_c**2).sum(axis=-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2
        ).sum(axis=-1)

    grid = transforms["grid"].source_grid
    fun_data = {
        "|grad(psi)|*kappa_g": data["|grad(psi)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_r|v,p": data["|B|_r|v,p"],
        "K": data["iota_r"]
        * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]),
    }
    data["deprecated(Gamma_c)"] = (
        _compute(Gamma_c, fun_data, data, grid, num_pitch)
        / data["fieldline length"]
        / (2**1.5 * jnp.pi)
    )
    return data


def _drift(f, B, pitch):
    return safediv(f * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B)))


@register_compute_fun(
    name="Gamma_c Velasco",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy. "
    "Uses the numerical methods of the Bounce1D class.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "cvdrift0", "gbdrift", "fieldline length"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    **_Bounce1D_doc,
)
@partial(jit, static_argnames=["num_quad", "num_pitch", "num_well", "batch"])
def _Gamma_c_Velasco_1D(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Velasco et al.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.
    """
    # noqa: unused dependency
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )

    def Gamma_c(data):
        """∫ dλ ∑ⱼ [v τ γ_c²]ⱼ."""
        bounce = Bounce1D(grid, data, quad, automorphism=None, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau = bounce.integrate(_v_tau, data["pitch_inv"], points=points, batch=batch)
        gamma_c = jnp.arctan(
            safediv(
                bounce.integrate(
                    _drift,
                    data["pitch_inv"],
                    data["cvdrift0"],
                    points=points,
                    batch=batch,
                ),
                bounce.integrate(
                    _drift,
                    data["pitch_inv"],
                    data["gbdrift"],
                    points=points,
                    batch=batch,
                ),
            )
        )
        return (
            (v_tau * gamma_c**2).sum(axis=-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2
        ).sum(axis=-1)

    grid = transforms["grid"].source_grid
    data["Gamma_c Velasco"] = (
        _compute(
            Gamma_c,
            fun_data={"cvdrift0": data["cvdrift0"], "gbdrift": data["gbdrift"]},
            data=data,
            grid=grid,
            num_pitch=num_pitch,
        )
        / data["fieldline length"]
        / (2**1.5 * jnp.pi)
    )
    return data
