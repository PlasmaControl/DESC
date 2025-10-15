"""Compute functions for fast ion confinement."""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D
from ..integrals.quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from ..utils import cross, dot, safediv
from ._neoclassical import _bounce_doc, _compute
from .data_index import register_compute_fun

# We rewrite equivalents of Nemov et al.'s expressions (21, 22) to resolve
# the indeterminate form of the limit and use single-valued maps of physical
# coordinates. This avoids the computational issues of multivalued maps.
# The derivative (∂/∂ψ)|ϑ,ϕ belongs to flux coordinates which satisfy
# α = ϑ − χ(ψ) ϕ where α is the poloidal label of ψ,α Clebsch coordinates.
# Choosing χ = ι implies ϑ, ϕ are PEST angles.
# ∂G/∂((λB₀)⁻¹) =     λ²B₀  ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) ∂|B|/∂ψ / |B|
# ∂V/∂((λB₀)⁻¹) = 3/2 λ²B₀  ∫ dℓ √(1 − λ|B|) R / |B|
# ∂g/∂((λB₀)⁻¹) =     λ²B₀² ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
# K ≝ R dψ/dρ
# tan(π/2 γ_c) =
#              ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
#              ----------------------------------------------
# (|∇ρ| ‖e_α|ρ,ϕ‖)ᵢ ∫ dℓ [ (1 − λ|B|/2)/√(1 − λ|B|) ∂|B|/∂ρ + √(1 − λ|B|) K ] / |B|


def _v_tau(data, B, pitch):
    # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
    # τ is the bounce time, and I is defined in Nemov et al. eq. 36.
    return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


def _drift1(data, B, pitch):
    return (
        safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
        * data["|grad(psi)|*kappa_g"]
        / B
    )


def _drift2(data, B, pitch):
    return (
        safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
        * data["|B|_r|v,p"]
        + jnp.sqrt(jnp.abs(1 - pitch * B)) * data["K"]
    ) / B


@register_compute_fun(
    name="Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy (scalar)",
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
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_well",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Nemov et al.

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. The energetic particle confinement
    metric γ_c quantifies whether the contours of the second adiabatic invariant
    close on the flux surfaces. In the limit where the poloidal drift velocity
    majorizes the radial drift velocity, the contours lie parallel to flux
    surfaces. The optimization metric Γ_c averages γ_c² over the distribution
    of trapped particles on each flux surface.

    The radial electric field has a negligible effect, since fast particles
    have high energy with collisionless orbits, so it is assumed to be zero.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."
    fl_quad = (
        kwargs["fieldline_quad"] if "fieldline_quad" in kwargs else leggauss(Y_B // 2)
    )
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-7)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def Gamma_c(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            points = bounce.points(pitch_inv, num_well)
            v_tau, drift1, drift2 = bounce.integrate(
                [_v_tau, _drift1, _drift2],
                pitch_inv,
                data,
                ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
                points,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            # This is γ_c π/2.
            gamma_c = jnp.arctan(
                safediv(
                    drift1,
                    drift2
                    * bounce.interp_to_argmin(
                        data["|grad(rho)|*|e_alpha|r,p|"],
                        points,
                        nufft_eps=nufft_eps,
                        is_fourier=True,
                    ),
                )
            )
            return (v_tau * gamma_c**2).sum(-1).mean(-2)

        return jnp.sum(
            batch_map(fun, data["pitch_inv"], pitch_batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / (bounce.compute_fieldline_length(fl_quad, vander) * 2**1.5 * jnp.pi)

    # It is assumed the grid is sufficiently dense to reconstruct |B|,
    # so anything smoother than |B| may be captured accurately as a single
    # Fourier series rather than transforming each component. Last term in K
    # behaves as ∂log(|B|²/(R₀B₀B^ϕ))/∂ρ |B| where R₀B₀ is a constant with
    # units Tesla meters. Smoothness is determined by positive lower bound of
    # log argument, and hence behaves as ∂log(|B|/B₀)/∂ρ |B| = ∂|B|/∂ρ.
    fun_data = {
        "|grad(psi)|*kappa_g": data["|grad(psi)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_r|v,p": data["|B|_r|v,p"],
        "K": data["iota_r"]
        * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]),
    }
    grid = transforms["grid"]
    data["Gamma_c"] = _compute(
        Gamma_c, fun_data, data, theta, grid, num_pitch, surf_batch_size
    )
    return data


def _adiabatic_J_num(data, B, pitch):
    """Numerator of the second adiabatic invariant J||."""
    # v_∥/ (√2E/m)
    return jnp.sqrt(jnp.abs(1 - pitch * B))


def _radial_drift(data, B, pitch):
    return safediv(
        data["cvdrift0"] * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
    )


def _poloidal_drift(data, B, pitch):
    return safediv(
        (data["gbdrift (periodic)"] + data["gbdrift (secular)/phi"] * data["zeta"])
        * (1 - 0.5 * pitch * B),
        jnp.sqrt(jnp.abs(1 - pitch * B)),
    )


def _binormal_drift(data, B, pitch):
    return safediv(
        (data["gbdrift (periodic)"] + data["gbdrift (secular)/phi"] * data["zeta"])
        * (1 - 0.5 * pitch * B)
        + (
            data["cvdrift (periodic)"] - data["gbdrift (periodic)"]
        )  # pressure gradient term
        * (1 - pitch * B),
        jnp.sqrt(jnp.abs(1 - pitch * B)),
    )


@register_compute_fun(
    name="gamma_c",
    label="\\sum_{w} \\gamma_c(\\rho, \\alpha, \\lambda, w)",
    units="~",
    units_long="None",
    description="Fast ion confinement proxy",
    dim=2,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
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
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_well",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _little_gamma_c_Nemov(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Nemov et al.

    Returns
    -------
    ∑_w γ_c(ρ, α, λ, w) where w indexes a well.
        Shape (num rho, num alpha, num pitch).

    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-7)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def gamma_c0(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            points = bounce.points(pitch_inv, num_well)
            drift1, drift2 = bounce.integrate(
                [_drift1, _drift2],
                pitch_inv,
                data,
                ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
                points,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            return (2 / jnp.pi) * jnp.arctan(
                safediv(
                    drift1,
                    drift2
                    * bounce.interp_to_argmin(
                        data["|grad(rho)|*|e_alpha|r,p|"],
                        points,
                        nufft_eps=nufft_eps,
                        is_fourier=True,
                    ),
                )
            ).sum(-1)

        return batch_map(fun, data["pitch_inv"], pitch_batch_size)

    fun_data = {
        "|grad(psi)|*kappa_g": data["|grad(psi)|"] * data["kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_r|v,p": data["|B|_r|v,p"],
        "K": data["iota_r"]
        * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]),
    }
    grid = transforms["grid"]
    data["gamma_c"] = _compute(
        gamma_c0,
        fun_data,
        data,
        theta,
        grid,
        num_pitch,
        surf_batch_size,
        expand_out=False,
    )
    return data


@register_compute_fun(
    name="Gamma_c Velasco",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy (scalar) "
    "as defined by Velasco et al. (doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "cvdrift0",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_well",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _Gamma_c_Velasco(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Velasco et al.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.

    This expression has a secular term that drives the result to zero as the number
    of toroidal transits increases if the secular term is not averaged out from the
    singular integrals. It is observed that this implementation does not average
    out the secular term. Currently, an optimization using this metric may need
    to be evaluated by measuring decrease in Γ_c at a fixed number of toroidal
    transits.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."
    fl_quad = (
        kwargs["fieldline_quad"] if "fieldline_quad" in kwargs else leggauss(Y_B // 2)
    )

    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-7)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def Gamma_c(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            v_tau, radial_drift, poloidal_drift = bounce.integrate(
                [_v_tau, _radial_drift, _poloidal_drift],
                pitch_inv,
                data,
                ["cvdrift0", "gbdrift (periodic)", "gbdrift (secular)/phi"],
                num_well=num_well,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            # This is γ_c π/2.
            gamma_c = jnp.arctan(safediv(radial_drift, poloidal_drift))
            return (v_tau * gamma_c**2).sum(-1).mean(-2)

        return jnp.sum(
            batch_map(fun, data["pitch_inv"], pitch_batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / (bounce.compute_fieldline_length(fl_quad, vander) * 2**1.5 * jnp.pi)

    grid = transforms["grid"]

    data["Gamma_c Velasco"] = _compute(
        Gamma_c,
        {
            "cvdrift0": data["cvdrift0"],
            "gbdrift (periodic)": data["gbdrift (periodic)"],
            "gbdrift (secular)/phi": data["gbdrift (secular)/phi"],
        },
        data,
        theta,
        grid,
        num_pitch,
        surf_batch_size,
    )
    return data


@register_compute_fun(
    name="adiabatic J",
    label=(  # J_∥ = ∫ dl v_∥/ (√2E/m) )/∫ dl
        "\\J_{\\parallel} = \\integrate v_{\\parallel} dl/\\integrate dl/B"
    ),
    units="~",
    units_long="~",
    description="Normalized second adiabatic invariant of motion.",
    coordinates="r",
    dim=1,
    profiles=[],
    params=[],
    transforms={"grid": []},
    data=["min_tz |B|", "max_tz |B|", "R0"] + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_well",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _adiabatic_J(params, transforms, profiles, data, **kwargs):
    """Second adiabatic invariant of particle motion.

    The normalization requires a length for which we have used the fieldline
    length ∫ dl.
    Typically calculated as a function of (rho, alpha, lambda)
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."

    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            chebgauss2(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-6)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def adiabatic_J0(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            split_by_NFP=False,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            return bounce.integrate(
                [_adiabatic_J_num],
                pitch_inv,
                data,
                [],
                num_well=num_well,
                nufft_eps=nufft_eps,
                is_fourier=True,
            ).sum(-1)

        return batch_map(fun, data["pitch_inv"], pitch_batch_size)

    grid = transforms["grid"]
    data["adiabatic J"] = _compute(
        adiabatic_J0,
        {},
        data,
        theta,
        grid,
        num_pitch,
        surf_batch_size,
    ) / (2 * jnp.pi * num_transit * data["R0"])
    return data


@register_compute_fun(
    name="<v_dot_grads>",
    label=(  # <v⋅∇s> = ∮ dl/|v_∥| (v_d ⋅ ∇s), s=ρ²
        "\\langle v \\cdot \\nabla s \\rangle"
    ),
    units="~",
    units_long="m^{-2}",
    description="Bounce integrated radial drift.",
    coordinates="r",
    dim=1,
    profiles=[],
    transforms={"grid": []},
    params=[],
    data=["min_tz |B|", "max_tz |B|", "cvdrift0", "R0"] + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_well",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _bounceavg_v_dot_grads(params, transforms, profiles, data, **kwargs):
    """Direct measure of omnigenity.

    Exactly equivalent to the bounce-averaged radial drift.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."
    fl_quad = (
        kwargs["fieldline_quad"] if "fieldline_quad" in kwargs else leggauss(Y_B // 2)
    )

    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-7)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def v_dot_grads0(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            v_tau, radial_drift = bounce.integrate(
                [_v_tau, _radial_drift],
                pitch_inv,
                data,
                ["cvdrift0"],
                num_well=num_well,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            # Take sum over wells, then divide
            v_dot_grads = safediv(radial_drift.sum(-1), v_tau.sum(-1))

            # Now take max in alpha (max radial excursion)
            # Negative or positive radial excursion is both departure
            # from omnigenity, hence the abs
            return v_dot_grads

        return (
            batch_map(fun, data["pitch_inv"], pitch_batch_size)
            / bounce.compute_fieldline_length(fl_quad, vander)[:, None, None]
        )

    grid = transforms["grid"]
    data["<v_dot_grads>"] = _compute(
        v_dot_grads0,
        {"cvdrift0": data["cvdrift0"]},
        data,
        theta,
        grid,
        num_pitch,
        surf_batch_size,
    )
    # )--no-verify / (2 * jnp.pi * num_transit * data["R0"])
    return data


@register_compute_fun(
    name="J_alpha",
    label=(  # ∂_α J_∥ /∫dl = ∮ dl/|v_∥| (v_d ⋅ ∇s) /∫dl, s=ρ²
        "\\partial_{\\alpha} \\J_{\\parallel}"
    ),
    units="~",
    units_long="m^{-2}",
    description="Bounce-averaged radial drift.",
    coordinates="r",
    dim=1,
    profiles=[],
    transforms={"grid": []},
    params=[],
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "cvdrift0",
        "R0",
        "<v_dot_grads>",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _dJ_dalpha(params, transforms, profiles, data, **kwargs):
    """Direct measure of omnigenity.

    Exactly equivalent to the bounce-averaged radial drift.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."
    fl_quad = (
        kwargs["fieldline_quad"] if "fieldline_quad" in kwargs else leggauss(Y_B // 2)
    )

    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-7)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def dJ_dalpha0(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        # Find the most "leaky"/"lossy" fieldline and pick that,
        # then integrate over the pitch angle
        return jnp.sum(
            jnp.abs(data["radial_drift"]).max(-1)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / bounce.compute_fieldline_length(fl_quad, vander)

    grid = transforms["grid"]

    fourier_transformed_data = {}
    data["J_alpha"] = _compute(
        dJ_dalpha0,
        fourier_transformed_data,
        data,
        theta,
        grid,
        num_pitch,
        surf_batch_size,
        radial_drift=grid.compress(data["<v_dot_grads>"]),
    )
    # )--no-verify / (2 * jnp.pi * num_transit * data["R0"])
    return data


@register_compute_fun(
    name="J_s",
    label=(
        # ∂ₛJ_∥ = - ∫ dl/|v_∥| (v_d ⋅ ∇α)
        "\\partial_{\\s} \\J_{\\parallel}/\\oint dl"
    ),
    units="~",
    units_long="m-1",
    description="max-J term, bounce-integrated binormal drift",
    coordinates="r",
    dim=1,
    profiles=[],
    transforms={"grid": []},
    params=[],
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "cvdrift (periodic)",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_well",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _dJ_ds(params, transforms, profiles, data, **kwargs):
    """The max-J term.

    Bounce-averaged binormal drift.
    Normalization has been chosen to eliminate dependence on
    num_transits.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."

    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-7)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def dJ_ds0(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def fun(pitch_inv):
            poloidal_drift = bounce.integrate(
                [_poloidal_drift],
                pitch_inv,
                data,
                ["cvdrift (periodic)", "gbdrift (periodic)", "gbdrift (secular)/phi"],
                num_well=num_well,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )

            # Take sum over wells
            dJ_ds = jnp.sum(poloidal_drift, axis=-1)

            # max drift < 0 provides TEM(trapped electron mode)
            # stability for all rhos and pitches
            return dJ_ds

        # Output dimension (rho, alpha, lambda)
        return batch_map(fun, data["pitch_inv"], pitch_batch_size)

    grid = transforms["grid"]
    data["J_s"] = -1 * _compute(
        dJ_ds0,
        {
            "cvdrift (periodic)": data["cvdrift (periodic)"],
            "gbdrift (periodic)": data["gbdrift (periodic)"],
            "gbdrift (secular)/phi": data["gbdrift (secular)/phi"],
        },
        data,
        theta,
        grid,
        num_pitch,
        surf_batch_size,
    )

    return data
