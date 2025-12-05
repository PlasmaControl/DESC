"""Compute functions for fast ion confinement."""

from functools import partial

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D
from ..utils import cross, dot, safediv
from ._neoclassical import _bounce_doc
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
        "V_r/psi_r",
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

    [1] Poloidal motion of trapped particle orbits in real-space coordinates.
        V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
        Phys. Plasmas 1 May 2008; 15 (5): 052501.
        https://doi.org/10.1063/1.2912456.
        Equation 61.

    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging
        algorithm and its applications.
        Kaya E. Unalmis, Rahul Gaur, Rory Conlin, Dario Panici, Egemen Kolemen.
        https://arxiv.org/abs/2412.01724.

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
    grid = transforms["grid"]
    (
        angle,
        Y_B,
        alpha,
        num_transit,
        num_well,
        num_pitch,
        pitch_batch_size,
        surf_batch_size,
        quad,
        nufft_eps,
        spline,
        vander,
    ) = Bounce2D._default_kwargs("weak", grid.NFP, **kwargs)

    def Gamma_c(data):
        bounce = Bounce2D(
            grid,
            data,
            data["angle"],
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
        )

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
    data["Gamma_c"] = grid.expand(
        Bounce2D.batch(
            Gamma_c,
            fun_data,
            data,
            angle,
            grid,
            num_pitch,
            surf_batch_size,
        )
        / data["V_r/psi_r"]
        / (num_transit * 2**0.5)
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
    grid = transforms["grid"]
    (
        angle,
        Y_B,
        alpha,
        num_transit,
        num_well,
        num_pitch,
        pitch_batch_size,
        surf_batch_size,
        quad,
        nufft_eps,
        spline,
        vander,
    ) = Bounce2D._default_kwargs("weak", grid.NFP, **kwargs)

    def gamma_c0(data):
        bounce = Bounce2D(
            grid,
            data,
            data["angle"],
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
    data["gamma_c"] = Bounce2D.batch(
        gamma_c0, fun_data, data, angle, grid, num_pitch, surf_batch_size
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
        "V_r/psi_r",
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

    [1] A model for the fast evaluation of prompt losses of energetic ions in
        stellarators. Equation 16.
        J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
        https://doi.org/10.1088/1741-4326/ac2994.

    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging
        algorithm and its applications.
        Kaya E. Unalmis, Rahul Gaur, Rory Conlin, Dario Panici, Egemen Kolemen.
        https://arxiv.org/abs/2412.01724.

    This expression has a secular term that drives the result to zero as the number
    of toroidal transits increases if the secular term is not averaged out from the
    singular integrals. It is observed that this implementation does not average
    out the secular term. Currently, an optimization using this metric may need
    to be evaluated by measuring decrease in Γ_c at a fixed number of toroidal
    transits.
    """
    # noqa: unused dependency
    grid = transforms["grid"]
    (
        angle,
        Y_B,
        alpha,
        num_transit,
        num_well,
        num_pitch,
        pitch_batch_size,
        surf_batch_size,
        quad,
        nufft_eps,
        spline,
        vander,
    ) = Bounce2D._default_kwargs("weak", grid.NFP, **kwargs)

    def Gamma_c(data):
        bounce = Bounce2D(
            grid,
            data,
            data["angle"],
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
        )

    data["Gamma_c Velasco"] = grid.expand(
        Bounce2D.batch(
            Gamma_c,
            {
                "cvdrift0": data["cvdrift0"],
                "gbdrift (periodic)": data["gbdrift (periodic)"],
                "gbdrift (secular)/phi": data["gbdrift (secular)/phi"],
            },
            data,
            angle,
            grid,
            num_pitch,
            surf_batch_size,
        )
        / data["V_r/psi_r"]
        / (num_transit * 2**0.5)
    )
    return data
