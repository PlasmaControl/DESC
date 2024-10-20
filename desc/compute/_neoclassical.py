"""Compute functions for neoclassical transport.

Performance will improve significantly by resolving these GitHub issues.

* ``1154`` Improve coordinate mapping performance
* ``1294`` Nonuniform fast transforms
* ``1303`` Patch for differentiable code with dynamic shapes
* ``1206`` Upsample data above midplane to full grid assuming stellarator symmetry
* ``1034`` Optimizers/objectives with auxilary output

If memory is still an issue, consider computing one pitch at a time. This
can be done by copy-pasting the code given at
https://github.com/PlasmaControl/DESC/pull/1003#discussion_r1780459450.
Note that imap supports computing in batches, so that can also be used.
Make sure to benchmark whether this reduces memory in an optimization.
"""

from functools import partial

from orthax.legendre import leggauss
from quadax import simpson

from desc.backend import imap, jit, jnp

from ..integrals.bounce_integral import Bounce1D, Bounce2D
from ..integrals.bounce_utils import (
    get_pitch_inv_quad,
    interp_fft_to_argmin,
    interp_to_argmin,
)
from ..integrals.interp_utils import polyder_vec
from ..integrals.quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from ..utils import cross, dot, errorif, safediv
from .data_index import register_compute_fun

_Bounce1D_doc = {
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
_Bounce2D_doc = {
    "spline": "bool : Whether to use cubic splines to compute bounce points.",
    "theta": "jnp.ndarray : DESC coordinates θ of (α,ζ) Fourier Chebyshev basis nodes.",
    "Y_B": (
        "int : Desired resolution for |B| along field lines to compute bounce points. "
        "Default is to double the resolution of ``theta``."
    ),
    "num_transit": "int : Number of toroidal transits to follow field line.",
    "fieldline_quad": (
        "tuple[jnp.ndarray] : Quadrature points xₖ and weights wₖ for the "
        "approximate evaluation of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ). "
        "Used to compute the proper length of the field line ∫ dℓ / |B|. "
        "Default is Gauss-Legendre quadrature."
    ),
    "quad": _Bounce1D_doc["quad"],
    "num_quad": _Bounce1D_doc["num_quad"],
    "num_pitch": _Bounce1D_doc["num_pitch"],
    "num_well": _Bounce1D_doc["num_well"],
}


def _alpha_mean(f):
    """Simple mean over field lines.

    Simple mean rather than integrating over α and dividing by 2π
    (i.e. f.T.dot(dα) / dα.sum()), because when the toroidal angle extends
    beyond one transit we need to weight all field lines uniformly, regardless
    of their spacing wrt α.
    """
    return f.mean(axis=0)


def _compute_1D(fun, fun_data, data, grid, num_pitch, reduce=True):
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
        fun_data[name] = data[name]
    fun_data = dict(
        zip(fun_data.keys(), Bounce1D.reshape_data(grid, *fun_data.values()))
    )
    out = imap(for_each_rho, fun_data)
    return grid.expand(_alpha_mean(out)) if reduce else out


def _compute_2D(fun, fun_data, data, theta, grid, num_pitch):
    """Compute ``fun`` for each ρ value iteratively to reduce memory usage.

    Parameters
    ----------
    fun : callable
        Function to compute.
    fun_data : dict[str, jnp.ndarray]
        Data to provide to ``fun``.
        Names in ``Bounce2D.required_names`` will be overridden.
        Reshaped automatically.
    data : dict[str, jnp.ndarray]
        DESC data dict.
    theta : jnp.ndarray
        Shape (num rho, X, Y).
        DESC coordinates θ sourced from the Clebsch coordinates
        ``FourierChebyshevSeries.nodes(X,Y,rho,domain=(0,2*jnp.pi))``.

    """
    for name in Bounce2D.required_names:
        fun_data[name] = data[name]
    fun_data = dict(
        zip(fun_data.keys(), Bounce2D.reshape_data(grid, *fun_data.values()))
    )
    # These already have expected shape with num rho along first axis.
    fun_data["pitch_inv"], fun_data["pitch_inv weight"] = get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
    )
    fun_data["iota"] = grid.compress(data["iota"])
    fun_data["theta"] = theta
    return grid.expand(imap(fun, fun_data))


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


@register_compute_fun(
    name="effective ripple 3/2*",
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
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 3
        ).sum(axis=-1)

    # Interpolate |∇ρ| κ_g since it is smoother than κ_g alone.
    fun_data = {"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]}
    B0 = data["max_tz |B|"]
    data["effective ripple 3/2*"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * _compute_1D(eps_32, fun_data, data, grid, kwargs.get("num_pitch", 50))
        / data["fieldline length"]
    )
    return data


@register_compute_fun(
    name="effective ripple*",
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
    data=["effective ripple 3/2*"],
)
def _effective_ripple_1D(params, transforms, profiles, data, **kwargs):
    data["effective ripple*"] = data["effective ripple 3/2*"] ** (2 / 3)
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
    description="Effective ripple modulation amplitude to 3/2 power. "
    "Uses numerical methods of the Bounce2D class.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "kappa_g", "R0", "|grad(rho)|", "<|grad(rho)|>"]
    + Bounce2D.required_names,
    resolution_requirement="z",  # FIXME: GitHub issue #1312. Should be "tz".
    # TODO: Uniformly spaced points on (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP) are required.
    grid_requirement={"coordinates": "rtz", "is_meshgrid": True, "sym": False},
    **_Bounce2D_doc,
)
@partial(
    jit,
    static_argnames=[
        "spline",
        "Y_B",
        "num_transit",
        "num_quad",
        "num_pitch",
        "num_well",
    ],
)
def _epsilon_32_2D(params, transforms, profiles, data, **kwargs):
    """https://doi.org/10.1063/1.873749.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    spline = kwargs.get("spline", True)
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    num_transit = kwargs.get("num_transit", 20)
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = chebgauss2(kwargs.get("num_quad", 32))
    if "fieldline_quad" in kwargs:
        fieldline_quad = kwargs["fieldline_quad"]
    else:
        fieldline_quad = leggauss(Y_B // 2)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    grid = transforms["grid"]

    def dH(grad_rho_norm_kappa_g, B, pitch, zeta):
        # Integrand of Nemov eq. 30 with |∂ψ/∂ρ| (λB₀)¹ᐧ⁵ removed.
        return (
            jnp.sqrt(jnp.abs(1 - pitch * B))
            * (4 / (pitch * B) - 1)
            * grad_rho_norm_kappa_g
            / B
        )

    def dI(B, pitch, zeta):
        # Integrand of Nemov eq. 31.
        return jnp.sqrt(jnp.abs(1 - pitch * B)) / B

    def eps_32(data):
        """(∂ψ/∂ρ)⁻² B₀⁻² ∫ dλ λ⁻² ∑ⱼ Hⱼ²/Iⱼ."""
        # B₀ has units of λ⁻¹.
        # Nemov's ∑ⱼ Hⱼ²/Iⱼ = (∂ψ/∂ρ)² (λB₀)³ ``(H**2 / I).sum(axis=-1)``.
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        bounce = Bounce2D(
            grid,
            data,
            data["iota"],
            data["theta"],
            Y_B,
            num_transit,
            quad=quad,
            automorphism=None,
            is_reshaped=True,
            spline=spline,
        )
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        H = bounce.integrate(
            dH,
            data["pitch_inv"],
            data["|grad(rho)|*kappa_g"],
            points=points,
        )
        I = bounce.integrate(dI, data["pitch_inv"], points=points)
        return (
            safediv(H**2, I).sum(axis=-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 3
        ).sum(axis=-1) / bounce.compute_fieldline_length(fieldline_quad)

    # Interpolate |∇ρ| κ_g since it is smoother than κ_g alone.
    fun_data = {"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]}
    B0 = data["max_tz |B|"]
    data["effective ripple 3/2"] = (
        jnp.pi
        / (8 * 2**0.5)
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * _compute_2D(eps_32, fun_data, data, theta, grid, kwargs.get("num_pitch", 50))
    )
    return data


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude. "
    "Uses numerical methods of the Bounce2D class.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["effective ripple 3/2"],
)
def _effective_ripple_2D(params, transforms, profiles, data, **kwargs):
    data["effective ripple"] = data["effective ripple 3/2"] ** (2 / 3)
    return data


@register_compute_fun(
    name="Gamma_c Velasco*",
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

    def Gamma_c(data):
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
        return (
            (v_tau * gamma_c**2).sum(axis=-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2
        ).sum(axis=-1)

    fun_data = {"cvdrift0": data["cvdrift0"], "gbdrift": data["gbdrift"]}
    data["Gamma_c Velasco*"] = (
        _compute_1D(Gamma_c, fun_data, data, grid, kwargs.get("num_pitch", 64))
        / data["fieldline length"]
        / (2**1.5 * jnp.pi)
    )
    return data


@register_compute_fun(
    name="Gamma_c*",
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
        "b",
        "|B|_r|v,p",
        "iota_r",
        "grad(phi)",
        "e^rho",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "psi_r",
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
        return (
            (v_tau * gamma_c**2).sum(axis=-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2
        ).sum(axis=-1)

    # We rewrite equivalents of Nemov et al.'s expression's using single-valued
    # maps of a physical coordinates. This avoids the computational issues of
    # multivalued maps. It further enables use of more efficient methods, such as
    # fast transforms and fixed computational grids throughout optimization, which
    # are used in the numerical methods of the ``Bounce2D`` class. Also, Nemov
    # assumes B^ϕ > 0 in some comments; this is not true in DESC, but the
    # computations done here are invariant to the sign.

    # It is assumed the grid is sufficiently dense to reconstruct |B|,
    # so anything smoother than |B| may be captured accurately as a single
    # spline rather than splining each component.
    fun_data = {
        "|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_psi|v,p": data["|B|_r|v,p"] / data["psi_r"],
        "K": data["iota_r"] * dot(cross(data["e^rho"], data["b"]), data["grad(phi)"])
        # Behaves as ∂log(|B|²/B^ϕ)/∂ψ |B| if one ignores the issue of a log argument
        # with units. Smoothness determined by positive lower bound of log argument,
        # and hence behaves as ∂log(|B|)/∂ψ |B| = ∂|B|/∂ψ.
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"])
        / data["psi_r"],
    }
    data["Gamma_c*"] = (
        _compute_1D(Gamma_c, fun_data, data, grid, kwargs.get("num_pitch", 64))
        / data["fieldline length"]
        / (2**1.5 * jnp.pi)
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
    description="Energetic ion confinement proxy, Nemov et al. "
    "Uses the numerical methods of the Bounce2D class.",
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
        "iota_r",
        "grad(phi)",
        "e^rho",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "psi_r",
    ]
    + Bounce2D.required_names,
    resolution_requirement="z",  # FIXME: GitHub issue #1312. Should be "tz".
    # TODO: Uniformly spaced points on (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP) are required.
    grid_requirement={"coordinates": "rtz", "is_meshgrid": True, "sym": False},
    **_Bounce2D_doc,
    quad2="Same as ``quad`` for the weak singular integrals in particular.",
)
@partial(
    jit,
    static_argnames=[
        "spline",
        "Y_B",
        "num_transit",
        "num_quad",
        "num_pitch",
        "num_well",
    ],
)
def _Gamma_c_2D(params, transforms, profiles, data, **kwargs):
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
    theta = kwargs["theta"]
    spline = kwargs.get("spline", True)
    errorif(not spline, NotImplementedError)
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    num_transit = kwargs.get("num_transit", 20)
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = chebgauss2(kwargs.get("num_quad", 32))
    quad2 = kwargs["quad2"] if "quad2" in kwargs else chebgauss2(quad[0].size)
    if "fieldline_quad" in kwargs:
        fieldline_quad = kwargs["fieldline_quad"]
    else:
        fieldline_quad = leggauss(Y_B // 2)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    grid = transforms["grid"]

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

    def d_v_tau(B, pitch, zeta):
        return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))

    def drift1(grad_rho_norm_kappa_g, B, pitch, zeta):
        return (
            safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
            * grad_rho_norm_kappa_g
            / B
        )

    def drift2(B_psi, B, pitch, zeta):
        return (
            safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B))) * B_psi / B
        )

    def drift3(K, B, pitch, zeta):
        return jnp.sqrt(jnp.abs(1 - pitch * B)) * K / B

    def Gamma_c(data):
        """∫ dλ ∑ⱼ [v τ γ_c²]ⱼ."""
        bounce = Bounce2D(
            grid,
            data,
            data["iota"],
            data["theta"],
            Y_B,
            num_transit,
            quad=quad,
            automorphism=None,
            is_reshaped=True,
            spline=spline,
        )
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau = bounce.integrate(d_v_tau, data["pitch_inv"], points=points)
        gamma_c = jnp.arctan(
            safediv(
                bounce.integrate(
                    drift1,
                    data["pitch_inv"],
                    data["|grad(rho)|*kappa_g"],
                    points=points,
                ),
                (
                    bounce.integrate(
                        drift2, data["pitch_inv"], data["|B|_psi|v,p"], points=points
                    )
                    + bounce.integrate(
                        drift3, data["pitch_inv"], data["K"], points=points, quad=quad2
                    )
                )
                * interp_fft_to_argmin(
                    bounce._NFP,
                    bounce._c["T(z)"],
                    data["|grad(rho)|*|e_alpha|r,p|"],
                    points,
                    bounce._c["knots"],
                    bounce._c["B(z)"],
                    polyder_vec(bounce._c["B(z)"]),
                ),
            )
        )
        return (
            (v_tau * gamma_c**2).sum(axis=-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2
        ).sum(axis=-1) / bounce.compute_fieldline_length(fieldline_quad)

    # We rewrite equivalents of Nemov et al.'s expression's using single-valued
    # maps of a physical coordinates. This avoids the computational issues of
    # multivalued maps. It further enables use of more efficient methods, such as
    # fast transforms and fixed computational grids throughout optimization, which
    # are used in the numerical methods of the ``Bounce2D`` class. Also, Nemov
    # assumes B^ϕ > 0 in some comments; this is not true in DESC, but the
    # computations done here are invariant to the sign.

    # It is assumed the grid is sufficiently dense to reconstruct |B|,
    # so anything smoother than |B| may be captured accurately as a single
    # Fourier series rather than transforming each component.
    fun_data = {
        "|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_psi|v,p": data["|B|_r|v,p"] / data["psi_r"],
        "K": data["iota_r"] * dot(cross(data["e^rho"], data["b"]), data["grad(phi)"])
        # Behaves as ∂log(|B|²/B^ϕ)/∂ψ |B| if one ignores the issue of a log argument
        # with units. Smoothness determined by positive lower bound of log argument,
        # and hence behaves as ∂log(|B|)/∂ψ |B| = ∂|B|/∂ψ.
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"])
        / data["psi_r"],
    }
    data["Gamma_c"] = _compute_2D(
        Gamma_c, fun_data, data, theta, grid, kwargs.get("num_pitch", 64)
    ) / (2**1.5 * jnp.pi)
    return data
