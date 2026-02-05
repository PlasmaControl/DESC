"""Compute functions for ITG turbulence proxies and related quantities.

This module provides GX geometric coefficients and ITG turbulence proxies
for gyrokinetic turbulence prediction. The implementation follows the
conventions established in Landreman et al. 2025 (arXiv:2502.11657).

Notes
-----
Reference quantities use the following normalizations:
- B_reference = 2|ψ_b|/a² where ψ_b is the boundary toroidal flux
- L_reference = a (minor radius)

Some quantities may have singularities at the magnetic axis (ρ=0).
Objectives using these quantities should use ρ > 0.

Flux tube utilities use the θ_PEST (straight-field-line) parameterization
for arclength computation, which matches the GX training data conventions.
"""

import jax

from scipy.constants import mu_0

from desc.backend import jnp

from ..utils import cross, dot
from .data_index import register_compute_fun

# Slope for smooth Heaviside (sigmoid) used in ITG proxy
_HEAVISIDE_SMOOTH_K = 10.0


# =============================================================================
# Reference Quantities
# =============================================================================


@register_compute_fun(
    name="gx_B_reference",
    label="B_{\\mathrm{ref}}",
    units="T",
    units_long="Tesla",
    description="GX reference magnetic field: B_ref = 2|ψ_b|/a²",
    dim=0,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="",
    data=["a"],
)
def _gx_B_reference(params, transforms, profiles, data, **kwargs):
    psi_b = jnp.abs(params["Psi"]) / (2 * jnp.pi)
    data["gx_B_reference"] = 2 * psi_b / data["a"] ** 2
    return data


@register_compute_fun(
    name="gx_L_reference",
    label="L_{\\mathrm{ref}}",
    units="m",
    units_long="meters",
    description="GX reference length: L_ref = a (minor radius)",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["a"],
)
def _gx_L_reference(params, transforms, profiles, data, **kwargs):
    data["gx_L_reference"] = data["a"]
    return data


# =============================================================================
# Normalized Magnetic Field
# =============================================================================


@register_compute_fun(
    name="gx_bmag",
    label="|\\mathbf{B}|/B_{\\mathrm{ref}}",
    units="~",
    units_long="dimensionless",
    description="Normalized magnetic field magnitude |B|/B_ref for GX",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "a"],
)
def _gx_bmag(params, transforms, profiles, data, **kwargs):
    psi_b = jnp.abs(params["Psi"]) / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    data["gx_bmag"] = data["|B|"] / B_ref
    return data


# =============================================================================
# Gradient Coefficients
# =============================================================================


@register_compute_fun(
    name="gx_gds2",
    label="|\\nabla \\alpha|^2 L_{\\mathrm{ref}}^2 s",
    units="~",
    units_long="dimensionless",
    description="GX gds2: |grad(alpha)|^2 * L_ref^2 * s, where s = rho^2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "rho", "grad(alpha)"],
)
def _gx_gds2(params, transforms, profiles, data, **kwargs):
    L_ref = data["a"]
    s = data["rho"] ** 2
    grad_alpha_sq = dot(data["grad(alpha)"], data["grad(alpha)"])
    data["gx_gds2"] = grad_alpha_sq * L_ref**2 * s
    return data


@register_compute_fun(
    name="gx_gds21_over_shat",
    label="\\mathrm{gds21} / \\hat{s}",
    units="~",
    units_long="dimensionless",
    description="GX gds21/shat: grad(alpha).grad(psi) * sigma_Bxy / B_ref",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "grad(alpha)", "grad(psi)"],
)
def _gx_gds21_over_shat(params, transforms, profiles, data, **kwargs):
    sigma_Bxy = -1  # GX sign convention
    psi_b = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    grad_alpha_dot_grad_psi = dot(data["grad(alpha)"], data["grad(psi)"])
    data["gx_gds21_over_shat"] = sigma_Bxy * grad_alpha_dot_grad_psi / B_ref
    return data


@register_compute_fun(
    name="gx_gds22_over_shat_squared",
    label="\\mathrm{gds22} / \\hat{s}^2",
    units="~",
    units_long="dimensionless",
    description="GX gds22/shat^2: |grad(psi)|^2 / (L_ref^2 * B_ref^2 * s)",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "rho", "|grad(psi)|^2"],
)
def _gx_gds22_over_shat_squared(params, transforms, profiles, data, **kwargs):
    psi_b = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    L_ref = data["a"]
    s = data["rho"] ** 2
    data["gx_gds22_over_shat_squared"] = (
        data["|grad(psi)|^2"] / (L_ref**2 * B_ref**2 * s)
    )
    return data


# =============================================================================
# Drift Coefficients
# =============================================================================


@register_compute_fun(
    name="gx_gbdrift",
    label="\\mathrm{gbdrift}",
    units="~",
    units_long="dimensionless",
    description="GX gbdrift: 2*B_ref*L_ref²*√s*(Bx∇|B|)·∇α / |B|³",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "rho", "|B|", "grad(|B|)", "B", "grad(alpha)"],
)
def _gx_gbdrift(params, transforms, profiles, data, **kwargs):
    sigma_Bxy = -1  # GX sign convention
    psi_sign = jnp.sign(params["Psi"])
    psi_b = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    L_ref = data["a"]
    sqrt_s = data["rho"]
    B_cross_grad_B = cross(data["B"], data["grad(|B|)"])
    B_cross_grad_B_dot_grad_alpha = dot(B_cross_grad_B, data["grad(alpha)"])
    data["gx_gbdrift"] = (
        2
        * sigma_Bxy
        * psi_sign
        * B_ref
        * L_ref**2
        * sqrt_s
        * B_cross_grad_B_dot_grad_alpha
        / data["|B|"] ** 3
    )
    return data


@register_compute_fun(
    name="gx_cvdrift",
    label="\\mathrm{cvdrift}",
    units="~",
    units_long="dimensionless",
    description="GX cvdrift: gbdrift + pressure term",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "rho", "|B|", "gx_gbdrift", "p_r"],
)
def _gx_cvdrift(params, transforms, profiles, data, **kwargs):
    sigma_Bxy = -1  # GX sign convention
    psi_sign = jnp.sign(params["Psi"])
    psi_b = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["a"] ** 2
    L_ref = data["a"]
    sqrt_s = data["rho"]
    d_pressure_d_s = data["p_r"] / (2 * sqrt_s)
    pressure_term = (
        2
        * mu_0
        * sigma_Bxy
        * psi_sign
        * B_ref
        * L_ref**2
        * sqrt_s
        * d_pressure_d_s
        / (psi_b * data["|B|"] ** 2)
    )
    data["gx_cvdrift"] = data["gx_gbdrift"] + pressure_term
    return data


@register_compute_fun(
    name="gx_gbdrift0_over_shat",
    label="\\mathrm{gbdrift0} / \\hat{s}",
    units="~",
    units_long="dimensionless",
    description="GX gbdrift0/shat: 2*sign(Psi)*(Bx∇|B|)·∇ψ / (|B|³*√s)",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["rho", "|B|", "grad(|B|)", "B", "grad(psi)"],
)
def _gx_gbdrift0_over_shat(params, transforms, profiles, data, **kwargs):
    psi_sign = jnp.sign(params["Psi"])
    sqrt_s = data["rho"]
    B_cross_grad_B = cross(data["B"], data["grad(|B|)"])
    B_cross_grad_B_dot_grad_psi = dot(B_cross_grad_B, data["grad(psi)"])
    data["gx_gbdrift0_over_shat"] = (
        2 * psi_sign * B_cross_grad_B_dot_grad_psi / (data["|B|"] ** 3 * sqrt_s)
    )
    return data


# =============================================================================
# Parallel Gradient
# =============================================================================


@register_compute_fun(
    name="gx_gradpar",
    label="L_{\\mathrm{ref}} \\mathbf{b} \\cdot \\nabla",
    units="~",
    units_long="dimensionless",
    description="GX gradpar: L_ref * (B^θ*(1+λ_θ) + B^ζ*λ_ζ) / |B|",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "|B|", "B^theta", "B^zeta", "lambda_t", "lambda_z"],
)
def _gx_gradpar(params, transforms, profiles, data, **kwargs):
    L_ref = data["a"]
    data["gx_gradpar"] = (
        L_ref
        * (data["B^theta"] * (1 + data["lambda_t"]) + data["B^zeta"] * data["lambda_z"])
        / data["|B|"]
    )
    return data


# =============================================================================
# ITG Proxy (Landreman et al. 2025)
# =============================================================================


@register_compute_fun(
    name="ITG proxy integrand",
    label="f_Q \\mathrm{integrand}",
    units="~",
    units_long="dimensionless",
    description="Integrand for ITG proxy: (sigmoid(k*cvdrift) + 0.2) * |grad_x|^3 / B",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gx_cvdrift", "gx_bmag", "gx_gds22_over_shat_squared"],
)
def _itg_proxy_integrand(params, transforms, profiles, data, **kwargs):
    """Compute the integrand for ITG proxy from Landreman et al. 2025.

    Uses a smooth sigmoid in place of Heaviside for differentiability.
    The sign of cvdrift is negated to match GX convention where
    positive cvdrift indicates bad curvature regions.
    """
    cvdrift_gx_convention = -data["gx_cvdrift"]
    grad_x_cubed = data["gx_gds22_over_shat_squared"] ** 1.5
    smooth_step = jax.nn.sigmoid(_HEAVISIDE_SMOOTH_K * cvdrift_gx_convention)
    data["ITG proxy integrand"] = (smooth_step + 0.2) * grad_x_cubed / data["gx_bmag"]
    return data


@register_compute_fun(
    name="ITG proxy",
    label="f_Q",
    units="~",
    units_long="dimensionless",
    description="ITG turbulence proxy: mean of integrand over field line",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["ITG proxy integrand"],
)
def _itg_proxy(params, transforms, profiles, data, **kwargs):
    """Compute ITG proxy from Landreman et al. 2025.

    f_Q = mean((sigmoid(cvdrift) + 0.2) * |grad_x|^3 / B)
    """
    data["ITG proxy"] = jnp.mean(data["ITG proxy integrand"])
    return data


# =============================================================================
# Flux Tube Utilities
# =============================================================================


def compute_arclength_via_gradpar(gradpar, theta_pest):
    """Compute cumulative arclength along a field line using θ_PEST parameterization.

    This function integrates 1/|gradpar| along the field line to obtain the
    cumulative arclength. This is the recommended method for NNITGProxy as it
    matches the GX training data conventions.

    Parameters
    ----------
    gradpar : ndarray, shape (npoints,) or (npoints, num_alpha)
        GX gradpar coefficient along the field line. This already includes the
        L_ref factor: gradpar = L_ref x b·∇θ_PEST.
    theta_pest : ndarray, shape (npoints,)
        Straight-field-line poloidal angles (assumed uniformly spaced).

    Returns
    -------
    arclength : ndarray, same shape as gradpar
        Cumulative arclength s(θ_PEST) in units of L_ref. Starts at 0 for the
        first point.

    Notes
    -----
    The arclength element is:
        dl/dθ_PEST = L_ref / |gradpar|

    Since gx_gradpar already includes L_ref, we have:
        dl/dθ_PEST = L_ref² / |gx_gradpar|

    For normalized arclength (setting L_ref=1):
        dl/dθ_PEST = 1 / |gradpar|

    The integration uses the trapezoidal rule for accuracy.

    See Also
    --------
    resample_to_uniform_arclength : Resample data to uniform arclength grid.
    """
    # dl/dθ_PEST = 1 / |gradpar| (for normalized arclength)
    dl_dtheta = 1.0 / jnp.abs(gradpar)

    # Trapezoidal integration
    dtheta = theta_pest[1] - theta_pest[0]

    # Handle both 1D and 2D cases
    if dl_dtheta.ndim == 1:
        integrand_half = 0.5 * (dl_dtheta[1:] + dl_dtheta[:-1])
        arclength = dtheta * jnp.concatenate(
            [jnp.array([0.0]), jnp.cumsum(integrand_half)]
        )
    else:
        # Shape (npoints, num_alpha) - integrate along axis 0
        integrand_half = 0.5 * (dl_dtheta[1:] + dl_dtheta[:-1])
        arclength = dtheta * jnp.concatenate(
            [jnp.zeros((1, dl_dtheta.shape[1])), jnp.cumsum(integrand_half, axis=0)],
            axis=0,
        )

    return arclength


def resample_to_uniform_arclength(arclength, data, npoints_out):
    """Resample field line data from non-uniform to uniform arclength spacing.

    This function takes data computed on a theta_PEST grid (with non-uniform
    arclength spacing) and resamples it to a uniform arclength grid in [-pi, pi).
    This matches the GX geometry conventions used for CNN training data.

    Parameters
    ----------
    arclength : ndarray, shape (npoints_in,) or (npoints_in, num_alpha)
        Cumulative arclength at each input point, as computed by
        ``compute_arclength_via_gradpar``.
    data : ndarray, shape (nfeatures, npoints_in) or (nfeatures, npoints_in, num_alpha)
        Data to resample. First axis is features (e.g., 7 GX coefficients).
    npoints_out : int
        Number of output points. Output z will be uniformly spaced in [-pi, pi)
        with the periodic endpoint excluded.

    Returns
    -------
    z_uniform : ndarray, shape (npoints_out,)
        Uniform arclength coordinates in [-pi, pi).
    data_uniform : ndarray, same shape as data with npoints_in -> npoints_out
        Resampled data on uniform arclength grid.

    Notes
    -----
    The transformation maps the cumulative arclength [0, L] to z in [-pi, pi).
    This normalization ensures the CNN sees features at evenly-spaced physical
    distances along the field line, regardless of the underlying magnetic
    geometry.

    Requires ``interpax`` package for cubic interpolation.

    See Also
    --------
    compute_arclength_via_gradpar : Compute cumulative arclength from gradpar.

    Examples
    --------
    >>> # Compute arclength from gradpar
    >>> arclength = compute_arclength_via_gradpar(gradpar, theta_pest)
    >>> # Stack 7 GX features
    >>> signals = jnp.stack([bmag, gbdrift, cvdrift, ...])  # shape (7, npoints)
    >>> # Resample to 96 points (CNN input resolution)
    >>> z, signals_uniform = resample_to_uniform_arclength(arclength, signals, 96)
    """
    from interpax import interp1d

    # Handle 1D and 2D cases
    if arclength.ndim == 1:
        # Single field line: arclength shape (npoints,), data shape (nfeatures, npoints)
        L = arclength[-1]
        z_orig = arclength * (2 * jnp.pi / L) - jnp.pi
        z_uniform = jnp.linspace(-jnp.pi, jnp.pi, npoints_out + 1)[:-1]

        nfeatures = data.shape[0]
        data_uniform = jnp.stack(
            [
                interp1d(z_uniform, z_orig, data[i], method="cubic")
                for i in range(nfeatures)
            ]
        )
    else:
        # Multiple field lines: arclength shape (npoints, num_alpha)
        # data shape (nfeatures, npoints, num_alpha)
        num_alpha = arclength.shape[1]
        z_uniform = jnp.linspace(-jnp.pi, jnp.pi, npoints_out + 1)[:-1]

        def resample_one_alpha(arc_1d, data_2d):
            """Resample a single field line."""
            L = arc_1d[-1]
            z_orig = arc_1d * (2 * jnp.pi / L) - jnp.pi
            return jnp.stack(
                [
                    interp1d(z_uniform, z_orig, data_2d[i], method="cubic")
                    for i in range(data_2d.shape[0])
                ]
            )

        # Process each alpha
        data_uniform = jnp.stack(
            [
                resample_one_alpha(arclength[:, a], data[:, :, a])
                for a in range(num_alpha)
            ],
            axis=2,
        )

    return z_uniform, data_uniform
