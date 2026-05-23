"""Compute functions for fast ion confinement.

References
----------
.. [1] V. V. Nemov, S. V. Kasilov, W. Kernbichler, and G. O. Leitold,
       "Poloidal motion of trapped particle orbits in real-space coordinates,"
       Phys. Plasmas 15, 052501 (2008). https://doi.org/10.1063/1.2912456.
.. [2] J. L. Velasco, I. Calvo, S. Mulas, E. Sanchez, F. I. Parra, A. Cappa,
       and the W7-X Team, "A model for the fast evaluation of prompt losses of
       energetic ions in stellarators," Nucl. Fusion 61, 116059 (2021).
       https://doi.org/10.1088/1741-4326/ac2994.
.. [3] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
       bounce-averaging algorithm and its applications,"
       J. Plasma Physics. https://doi:10.1017/S0022377826101652.

"""

from functools import partial

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D, Options
from ..integrals.quad_utils import _LossCone, _periodic_voronoi_widths
from ..utils import cross, dot, safediv
from ._drift import (
    _alpha_drift_wb_inverse,
    _radial_drift,
    _radial_drift_wb_inverse,
    _v_tau,
    _vartheta_drift,
)
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


def _gamma_c_data(data):
    # The grid has to be dense enough to avoid aliasing error on |B| anyway,
    # so we might as well interplate anything smoother than |B| with one
    # Fourier series rather than transforming each term. Last term in K
    # behaves as ∂log(|B|²/(R₀B₀B^ϕ))/∂ρ |B| where R₀B₀ is a constant with
    # units Tesla meters. Smoothness is determined by positive lower bound of
    # log argument, and hence behaves as ∂log(|B|/B₀)/∂ρ |B| = ∂|B|/∂ρ.
    return {
        "|grad(psi)|*kappa_g": data["|grad(psi)|*kappa_g"],
        "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
        "|B|_r|v,p": data["|B|_r|v,p"],
        "K": data["iota_r"]
        * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
        - (2 * data["|B|_r|v,p"] - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]),
    }


def _gamma_c(radial_drift, poloidal_drift, weight=1.0):
    return (2 / jnp.pi) * jnp.arctan(safediv(radial_drift, poloidal_drift * weight))


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
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "|grad(psi)|*kappa_g",
        "iota_r",
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **Options._doc,
)
@partial(jit, static_argnames=Options._static_argnames)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Equation 61 of [1]_.

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
    opts = Options.guess(-2, grid, **kwargs)

    def foreach_surface(data):

        def foreach(pitch_inv):
            points = bounce.points(pitch_inv, opts.num_well)
            v_tau, radial, poloidal = bounce.integrate(
                [_v_tau, _radial_drift, _vartheta_drift],
                pitch_inv,
                data,
                ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
                points,
            )
            return _reduction_gamma_c(
                v_tau,
                radial,
                poloidal
                * bounce.interp_to_argmin(data["|grad(rho)|*|e_alpha|r,p|"], points),
            )

        pitch_inv, weight = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        bounce = Bounce2D(grid, data, data["angle"], **opts)
        return jnp.sum(
            batch_map(foreach, pitch_inv, opts.pitch_batch_size)
            * (weight / pitch_inv**2),
            axis=-1,
        )

    out = Bounce2D.batch(
        foreach_surface,
        data,
        grid,
        angle=kwargs["angle"],
        custom_data=_gamma_c_data(data),
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1
    scalar = jnp.pi**2 / 2**2.5 * grid.NFP / opts.num_field_periods
    data["Gamma_c"] = grid.expand(out * scalar) / data["V_psi"]
    return data


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
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "|grad(psi)|*kappa_g",
        "iota_r",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **Options._doc,
)
@partial(jit, static_argnames=Options._static_argnames)
def _little_gamma_c_Nemov(params, transforms, profiles, data, **kwargs):
    """Equation 50 of [1]_.

    Returns
    -------
    ∑_w γ_c(ρ, α, λ, w) where w indexes a well.
        Shape (num rho, num alpha, num pitch).

    """
    # noqa: unused dependency
    grid = transforms["grid"]
    opts = Options.guess(-2, grid, loop=True, **kwargs)

    def foreach_surface(data):
        pitch_inv, _ = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        bounce = Bounce2D(grid, data, data["angle"], **opts)
        points = bounce.points(pitch_inv, opts.num_well)
        return _gamma_c(
            *bounce.integrate(
                [_radial_drift, _vartheta_drift],
                pitch_inv,
                data,
                ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
                points,
                loop=opts.loop,
            ),
            bounce.interp_to_argmin(data["|grad(rho)|*|e_alpha|r,p|"], points),
        ).sum(-1)

    data["gamma_c"] = Bounce2D.batch(
        foreach_surface,
        data,
        grid,
        angle=kwargs["angle"],
        custom_data=_gamma_c_data(data),
        batch_size=1,
        sparse=False,  # don't know of any applications that differentiate anyway
    )
    return data


def _reduction_gamma_c(v_tau, radial, poloidal, opts=None):
    return (v_tau * _gamma_c(radial, poloidal) ** 2).sum(-1).mean(-2)


def _reshape_iota(iota, suffix_ndim):
    """Reshape iota to broadcast over prefixed surface data."""
    iota = jnp.asarray(iota)
    prefix = () if iota.size == 1 else iota.shape
    return iota.reshape(prefix + (1,) * suffix_ndim)


def _well_field_period(points, NFP):
    """Locate each long-field-line well by its midpoint field period."""
    z1, z2 = points
    midpoint = 0.5 * (z1 + z2)
    return ((NFP * midpoint) // (2 * jnp.pi)).astype(jnp.int32)


def _local_well_rank(field_period, valid):
    """Rank wells within each single-field-period segment."""
    well = jnp.arange(field_period.shape[-1])
    previous = well[None, :] < well[:, None]
    return (
        (field_period[..., None] == field_period[..., None, :])
        & valid[..., None, :]
        & previous
    ).sum(-1)


def _fold_wells_to_alpha(values, points, opts, iota, NFP):
    """Treat long-field-line wells as denser alpha samples over one field period."""
    z1, z2 = points
    valid_well = z1 < z2
    well_field_period = _well_field_period(points, NFP)
    local_well_index = _local_well_rank(well_field_period, valid_well)
    num_alpha, num_pitch, num_well = z1.shape[-3:]
    prefix = z1.shape[:-3]

    field_period_index = jnp.arange(opts.num_field_periods).reshape(
        (1, opts.num_field_periods, 1, 1, 1)
    )
    local_well_slot = jnp.arange(num_well).reshape((1, 1, 1, 1, num_well))
    # Axes: prefix, base alpha, field period, pitch, original well, local well.
    well_to_alpha = (
        valid_well[..., :, None, :, :, None]
        & (well_field_period[..., :, None, :, :, None] == field_period_index)
        & (local_well_index[..., :, None, :, :, None] == local_well_slot)
    )

    def fold(value):
        value = jnp.where(well_to_alpha, value[..., :, None, :, :, None], 0.0).sum(-2)
        return value.reshape(
            prefix + (num_alpha * opts.num_field_periods, num_pitch, num_well)
        )

    alpha = opts.alpha.reshape((1,) * len(prefix) + (num_alpha, 1))
    alpha = alpha + _reshape_iota(iota, 2) * (2 * jnp.pi / NFP) * jnp.arange(
        opts.num_field_periods
    )
    alpha = (alpha % (2 * jnp.pi)).reshape(
        prefix + (num_alpha * opts.num_field_periods, 1, 1)
    )
    mask = well_to_alpha.any(-2).reshape(
        prefix + (num_alpha * opts.num_field_periods, num_pitch, num_well)
    )
    return (*[fold(value) for value in values], alpha, mask)


def _alpha_weights(alpha, valid, period=2 * jnp.pi):
    """Periodic Voronoi cell widths for a possibly nonuniform alpha grid."""
    alpha = alpha.swapaxes(-3, -1)
    valid = valid.swapaxes(-3, -1)
    count = valid.sum(-1, keepdims=True)
    _, _, width = _periodic_voronoi_widths(alpha, valid, period)
    weight = jnp.where(count == 1, 1.0, width / period)
    return jnp.where(valid, weight, 0.0).swapaxes(-3, -1)


def _reduction_gamma_delta(v_tau, radial, poloidal, opts, alpha, mask):
    v_tau = (v_tau * _alpha_weights(alpha, mask)).sum(-3)
    outward_superbanana = (radial > opts.thresh * jnp.abs(poloidal)).any(-3)
    return (v_tau * outward_superbanana).sum(-1)


def _reduction_gamma_alpha(v_tau, radial, poloidal, opts, alpha, mask, order=1):
    thresh = opts.thresh * jnp.abs(poloidal)
    outward_score = radial - thresh  # alpha out candidate
    inward_score = -radial - thresh  # alpha in candidate

    loss_cone = jnp.where(
        poloidal >= 0,
        _LossCone.indicator_nonuniform(
            inward_score, outward_score, alpha, mask, order=order
        ),
        _LossCone.indicator_nonuniform(
            outward_score, inward_score, alpha, mask, order=order
        ),
    )
    has_alpha_out = (outward_score > 0).any(-3, keepdims=True)
    has_alpha_in = (inward_score > 0).any(-3, keepdims=True)
    loss_cone = (has_alpha_out & has_alpha_in) * loss_cone + (
        has_alpha_out & ~has_alpha_in
    )
    return (v_tau * loss_cone * _alpha_weights(alpha, mask)).sum((-3, -1))


@register_compute_fun(
    name="Gamma_c Velasco",
    label=(
        "\\check{\\Gamma}_c = \\frac{1}{2} "
        "\\left\\langle \\int d\\lambda \\frac{B}{\\sqrt{1 - \\lambda B}} "
        "\\gamma_c^2"
        "\\right\\rangle"
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
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **Options._doc,
)
@partial(jit, static_argnames=Options._static_argnames)
def _Gamma_c_Velasco(params, transforms, profiles, data, **kwargs):
    """Equation 20 of [2]_."""
    # noqa: unused dependency
    data["Gamma_c Velasco"] = _Gamma(
        _reduction_gamma_c, params, transforms, profiles, data, **kwargs
    )
    return data


@register_compute_fun(
    name="Gamma_delta",
    label=(
        "\\Gamma_\\delta = \\frac{1}{2} "
        "\\left\\langle \\int d\\lambda \\frac{B}{\\sqrt{1 - \\lambda B}} "
        "H\\left(\\max_\\alpha \\gamma_c^* - \\gamma_\\mathrm{th}\\right) "
        "\\right\\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion superbanana proxy (scalar) "
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
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **Options._doc,
)
@partial(jit, static_argnames=Options._static_argnames)
def _Gamma_delta(params, transforms, profiles, data, **kwargs):
    """Equation 22 of [2]_."""
    # noqa: unused dependency
    data["Gamma_delta"] = _Gamma(
        _reduction_gamma_delta,
        params,
        transforms,
        profiles,
        data,
        fold_alpha=True,
        **kwargs,
    )
    return data


@register_compute_fun(
    name="Gamma_alpha",
    label=(
        "\\Gamma_\\alpha = \\frac{1}{2} "
        "\\left\\langle \\int d\\lambda \\frac{B}{\\sqrt{1 - \\lambda B}} "
        "H\\left((\\alpha_\\mathrm{out} - \\alpha)"
        "\\mathbf{v}_M \\cdot \\nabla \\alpha\\right) "
        "H\\left((\\alpha - \\alpha_\\mathrm{in})"
        "\\mathbf{v}_M \\cdot \\nabla \\alpha\\right) "
        "\\right\\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion superbanana proxy (scalar) "
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
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    **Options._doc,
)
@partial(jit, static_argnames=Options._static_argnames)
def _Gamma_alpha(params, transforms, profiles, data, **kwargs):
    """Equation 25 of [2]_."""
    # noqa: unused dependency
    data["Gamma_alpha"] = _Gamma(
        _reduction_gamma_alpha,
        params,
        transforms,
        profiles,
        data,
        fold_alpha=True,
        **kwargs,
    )
    return data


def _Gamma(reduction, params, transforms, profiles, data, fold_alpha=False, **kwargs):
    grid = transforms["grid"]
    opts = Options.guess(-1, grid, **kwargs)

    def foreach_surface(data):

        def foreach(pitch_inv):
            points = bounce.points(pitch_inv, opts.num_well) if fold_alpha else None
            integrals = bounce.integrate(
                [
                    _v_tau,
                    _radial_drift_wb_inverse,
                    _alpha_drift_wb_inverse,
                ],
                pitch_inv,
                data,
                names,
                points,
                num_well=opts.num_well,
            )
            if fold_alpha:
                *integrals, alpha, mask = _fold_wells_to_alpha(
                    integrals, points, opts, data["iota"], grid.NFP
                )
                return reduction(*integrals, opts, alpha=alpha, mask=mask)
            return reduction(*integrals, opts)

        pitch_inv, weight = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        bounce = Bounce2D(grid, data, data["angle"], **opts)
        return jnp.sum(
            batch_map(foreach, pitch_inv, opts.pitch_batch_size)
            * (weight / pitch_inv**2),
            axis=-1,
        )

    names = ("cvdrift0", "gbdrift (periodic)", "gbdrift (secular)/phi")
    out = Bounce2D.batch(
        foreach_surface,
        data,
        grid,
        angle=kwargs["angle"],
        names=names,
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1
    scalar = jnp.pi**3 / 16 * grid.NFP / opts.num_field_periods
    if fold_alpha:
        scalar *= opts.num_field_periods
    return grid.expand(out * scalar) / data["V_psi"]
