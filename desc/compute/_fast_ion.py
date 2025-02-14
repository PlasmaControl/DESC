"""Compute functions for fast ion confinement."""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D
from ..integrals.quad_utils import (
    automorphism_sin,
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
    name="gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "small gamma_c"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy",
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
        "spline",
    ],
)
def _gamma_c(params, transforms, profiles, data, **kwargs):
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
    spline = kwargs.get("spline", True)
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )

    def gamma_c0(data):
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
        )

        def fun(pitch_inv):
            points = bounce.points(pitch_inv, num_well)
            v_tau, drift1, drift2 = bounce.integrate(
                [_v_tau, _drift1, _drift2],
                pitch_inv,
                data,
                ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
                points,
                is_fourier=True,
            )

            # This is γ_c π/2.
            gamma_c = jnp.arctan(
                safediv(
                    drift1,
                    drift2
                    * bounce.interp_to_argmin(
                        data["|grad(rho)|*|e_alpha|r,p|"], points, is_fourier=True
                    ),
                )
            )

            # TODO: return v_tau as well?
            # --no-verify return jnp.sum(v_tau * gamma_c**2, axis=-1).mean(axis=-2)
            # --no-verify return gamma_c, v_tau
            return jnp.sum(gamma_c, axis=-1)

        return batch_map(fun, data["pitch_inv"], pitch_batch_size)

    # It is assumed the grid is sufficiently dense to reconstruct |B|,
    # so anything smoother than |B| may be captured accurately as a single
    # Fourier series rather than transforming each component.
    # Last term in K behaves as ∂log(|B|²/B^ϕ)/∂ρ |B| if one ignores the issue
    # of a log argument with units. Smoothness determined by positive lower bound
    # of log argument, and hence behaves as ∂log(|B|)/∂ρ |B| = ∂|B|/∂ρ.
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
        gamma_c0, fun_data, data, theta, grid, num_pitch, surf_batch_size
    )

    return data


@register_compute_fun(
    name="Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy",
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
    spline = kwargs.get("spline", True)
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

    def Gamma_c(data):
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
        )

        def fun(pitch_inv):
            points = bounce.points(pitch_inv, num_well)
            v_tau, drift1, drift2 = bounce.integrate(
                [_v_tau, _drift1, _drift2],
                pitch_inv,
                data,
                ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
                points,
                is_fourier=True,
            )
            # This is γ_c π/2.
            gamma_c = jnp.arctan(
                safediv(
                    drift1,
                    drift2
                    * bounce.interp_to_argmin(
                        data["|grad(rho)|*|e_alpha|r,p|"], points, is_fourier=True
                    ),
                )
            )
            return jnp.sum(v_tau * gamma_c**2, axis=-1).mean(axis=-2)

        return jnp.sum(
            batch_map(fun, data["pitch_inv"], pitch_batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / (bounce.compute_fieldline_length(fl_quad) * 2**1.5 * jnp.pi)

    # It is assumed the grid is sufficiently dense to reconstruct |B|,
    # so anything smoother than |B| may be captured accurately as a single
    # Fourier series rather than transforming each component.
    # Last term in K behaves as ∂log(|B|²/B^ϕ)/∂ρ |B| if one ignores the issue
    # of a log argument with units. Smoothness determined by positive lower bound
    # of log argument, and hence behaves as ∂log(|B|)/∂ρ |B| = ∂|B|/∂ρ.
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


@register_compute_fun(
    name="Gamma_c Velasco",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy "
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
    spline = kwargs.get("spline", True)
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

    def Gamma_c(data):
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
        )

        def fun(pitch_inv):
            v_tau, radial_drift, poloidal_drift = bounce.integrate(
                [_v_tau, _radial_drift, _poloidal_drift],
                pitch_inv,
                data,
                ["cvdrift0", "gbdrift (periodic)", "gbdrift (secular)/phi"],
                bounce.points(pitch_inv, num_well),
                is_fourier=True,
            )
            # This is γ_c π/2.
            gamma_c = jnp.arctan(safediv(radial_drift, poloidal_drift))
            return jnp.sum(v_tau * gamma_c**2, axis=-1).mean(axis=-2)

        return jnp.sum(
            batch_map(fun, data["pitch_inv"], pitch_batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / (bounce.compute_fieldline_length(fl_quad) * 2**1.5 * jnp.pi)

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
