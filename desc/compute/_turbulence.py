"""Compute functions for ITG turbulence proxies and related quantities.

Notes
-----
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

# GX sign convention for Bxy direction (Landreman et al. 2025)
_SIGMA_BXY = -1


# =============================================================================
# Reference Quantities
# =============================================================================


@register_compute_fun(
    name="gx_B_reference",
    label="B_{\\mathrm{ref}}",
    units="T",
    units_long="Tesla",
    description="GX reference magnetic field strength",
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
    description="GX reference length (minor radius)",
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
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "gx_B_reference"],
)
def _gx_bmag(params, transforms, profiles, data, **kwargs):
    data["gx_bmag"] = data["|B|"] / data["gx_B_reference"]
    return data


# =============================================================================
# Gradient Coefficients
# =============================================================================


@register_compute_fun(
    name="gx_gds2",
    label="|\\nabla \\alpha|^2 L_{\\mathrm{ref}}^2 s",
    units="~",
    units_long="dimensionless",
    description="GX gds2: perpendicular wavenumber squared in y direction",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gx_L_reference", "rho", "grad(alpha)"],
)
def _gx_gds2(params, transforms, profiles, data, **kwargs):
    L_ref = data["gx_L_reference"]
    s = data["rho"] ** 2
    grad_alpha_sq = dot(data["grad(alpha)"], data["grad(alpha)"])
    data["gx_gds2"] = grad_alpha_sq * L_ref**2 * s
    return data


@register_compute_fun(
    name="gx_gds21_over_shat",
    label="\\mathrm{gds21} / \\hat{s}",
    units="~",
    units_long="dimensionless",
    description="GX gds21/shat: cross term between x and y gradients",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gx_L_reference", "grad(alpha)", "grad(psi)"],
)
def _gx_gds21_over_shat(params, transforms, profiles, data, **kwargs):
    # Note: needs signed psi_b for physics, cannot use gx_B_reference
    psi_b = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * psi_b / data["gx_L_reference"] ** 2
    grad_alpha_dot_grad_psi = dot(data["grad(alpha)"], data["grad(psi)"])
    data["gx_gds21_over_shat"] = _SIGMA_BXY * grad_alpha_dot_grad_psi / B_ref
    return data


@register_compute_fun(
    name="gx_gds22_over_shat_squared",
    label="\\mathrm{gds22} / \\hat{s}^2",
    units="~",
    units_long="dimensionless",
    description="GX gds22/shat^2: perpendicular wavenumber squared in x direction",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gx_L_reference", "gx_B_reference", "rho", "|grad(psi)|^2"],
)
def _gx_gds22_over_shat_squared(params, transforms, profiles, data, **kwargs):
    L_ref = data["gx_L_reference"]
    B_ref = data["gx_B_reference"]
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
    description="GX gbdrift: grad-B drift dotted with grad y",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gx_L_reference", "gx_B_reference", "rho", "|B|", "grad(|B|)", "B", "grad(alpha)"],
)
def _gx_gbdrift(params, transforms, profiles, data, **kwargs):
    psi_sign = jnp.sign(params["Psi"])
    B_ref = data["gx_B_reference"]
    L_ref = data["gx_L_reference"]
    sqrt_s = data["rho"]
    B_cross_grad_B = cross(data["B"], data["grad(|B|)"])
    B_cross_grad_B_dot_grad_alpha = dot(B_cross_grad_B, data["grad(alpha)"])
    data["gx_gbdrift"] = (
        2
        * _SIGMA_BXY
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
    description="GX cvdrift: curvature drift dotted with grad y",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gx_L_reference", "gx_B_reference", "rho", "|B|", "gx_gbdrift", "p_r"],
)
def _gx_cvdrift(params, transforms, profiles, data, **kwargs):
    psi_sign = jnp.sign(params["Psi"])
    psi_b = jnp.abs(params["Psi"]) / (2 * jnp.pi)  # Still needed for pressure term
    B_ref = data["gx_B_reference"]
    L_ref = data["gx_L_reference"]
    sqrt_s = data["rho"]
    d_pressure_d_s = data["p_r"] / (2 * sqrt_s)
    pressure_term = (
        2
        * mu_0
        * _SIGMA_BXY
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
    description="GX gbdrift0/shat: grad-B drift dotted with grad x",
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
    description="GX gradpar: parallel gradient operator coefficient",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gx_L_reference", "|B|", "B^theta", "B^zeta", "lambda_t", "lambda_z"],
)
def _gx_gradpar(params, transforms, profiles, data, **kwargs):
    L_ref = data["gx_L_reference"]
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
    description="Integrand for ITG turbulence proxy",
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
    description="ITG turbulence proxy scalar",
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


def solve_poloidal_turns_for_length(length_fn, target_length, x0_guess=1.0):
    """Solve for poloidal turns to achieve a target flux tube length.

    Uses Brent's method to find the number of poloidal turns that produces
    a flux tube of the specified length.

    Parameters
    ----------
    length_fn : callable
        Function that takes poloidal_turns (float) and returns the flux tube
        length. This typically involves evaluating gradpar along a field line
        and integrating.
    target_length : float
        Target flux tube length in units of L_ref.
    x0_guess : float, optional
        Initial guess for poloidal turns. Default 1.0. The solver uses a
        bracket of [x0_guess/5, x0_guess*5] clamped to [0.05, 10.0].

    Returns
    -------
    poloidal_turns : float
        Number of poloidal turns that achieves the target length.

    Notes
    -----
    The target length is typically around 75.4 (as used in Landreman et al. 2025).
    The required poloidal turns depends on the equilibrium geometry:
    - For tokamaks: often 2-4 poloidal turns
    - For stellarators: depends on aspect ratio and rotational transform

    A good initial guess can be estimated from equilibrium geometry:
        x0_guess ~ target_length / (4 * pi * a * R0/a)
    where a is the minor radius and R0/a is the aspect ratio.

    Raises
    ------
    ValueError
        If the solver fails to converge (target length not achievable in bracket).

    See Also
    --------
    compute_arclength_via_gradpar : Compute cumulative arclength from gradpar.

    Examples
    --------
    >>> def length_fn(poloidal_turns):
    ...     # Compute gradpar for this many poloidal turns
    ...     theta_pest = np.linspace(-np.pi * poloidal_turns, np.pi * poloidal_turns, 1001)
    ...     gradpar = compute_gradpar_for_field_line(eq, rho, alpha, theta_pest)
    ...     return np.abs(np.trapezoid(1.0 / gradpar, theta_pest))
    >>> poloidal_turns = solve_poloidal_turns_for_length(length_fn, target_length=75.4)
    """
    from scipy.optimize import brentq

    def residual(poloidal_turns):
        return length_fn(poloidal_turns) - target_length

    # Create bracket centered on guess, clamped to reasonable range
    bracket_lo = max(0.05, x0_guess / 5)
    bracket_hi = min(10.0, x0_guess * 5)

    try:
        poloidal_turns = brentq(residual, bracket_lo, bracket_hi)
    except ValueError as e:
        # Check if target is outside the bracket range
        length_lo = length_fn(bracket_lo)
        length_hi = length_fn(bracket_hi)
        raise ValueError(
            f"Could not find poloidal_turns for target_length={target_length}. "
            f"Bracket [{bracket_lo:.2f}, {bracket_hi:.2f}] gives lengths "
            f"[{length_lo:.2f}, {length_hi:.2f}]. Try adjusting x0_guess."
        ) from e

    return poloidal_turns


# =============================================================================
# CNN Layer Primitives for NNITGProxy
# =============================================================================
#
# These functions implement the basic building blocks for the neural network
# forward pass in JAX. The architecture is CyclicInvariantNet from Landreman
# et al. 2025 (ensemble models without BatchNorm).
#
# The circular padding ensures translation invariance along the field line,
# which is physically motivated since heat flux should be invariant to
# the starting point along a periodic flux tube.


def _circular_pad_1d(x, pad_left, pad_right):
    """Apply circular padding to a 1D array along the last axis.

    This wraps values from the end of the array to the beginning and vice versa,
    implementing periodic boundary conditions.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., length)
    pad_left : int
        Number of elements to pad on the left (from end of array)
    pad_right : int
        Number of elements to pad on the right (from start of array)

    Returns
    -------
    padded : jax.Array
        Padded array of shape (..., length + pad_left + pad_right)

    Examples
    --------
    >>> x = jnp.array([1, 2, 3, 4, 5])
    >>> _circular_pad_1d(x, 2, 2)
    Array([4, 5, 1, 2, 3, 4, 5, 1, 2], dtype=int32)
    """
    left_pad = x[..., -pad_left:] if pad_left > 0 else x[..., :0]
    right_pad = x[..., :pad_right] if pad_right > 0 else x[..., :0]
    return jnp.concatenate([left_pad, x, right_pad], axis=-1)


def _conv1d_circular(x, weight, bias, kernel_size):
    """1D convolution with circular padding.

    This implements the same behavior as PyTorch's Conv1d with
    padding_mode='circular' and padding='same'.

    Parameters
    ----------
    x : jax.Array
        Input of shape (batch, in_channels, length)
    weight : jax.Array
        Convolution weights of shape (out_channels, in_channels, kernel_size)
    bias : jax.Array
        Bias of shape (out_channels,)
    kernel_size : int
        Size of the convolution kernel

    Returns
    -------
    out : jax.Array
        Output of shape (batch, out_channels, length)

    Notes
    -----
    The circular padding ensures that conv1d output has the same length as
    input (equivalent to PyTorch's padding='same' with padding_mode='circular').
    This maintains the periodic structure of the flux tube geometry.

    The padding is asymmetric for even kernel sizes to match PyTorch exactly:
    - For kernel_size=3: pad_left=1, pad_right=1 (symmetric)
    - For kernel_size=4: pad_left=1, pad_right=2 (asymmetric)
    """
    total_pad = kernel_size - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    x_padded = _circular_pad_1d(x, pad_left, pad_right)

    out = jax.lax.conv_general_dilated(
        x_padded,
        weight,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    return out + bias[None, :, None]


def _max_pool_1d(x, pool_size=2, stride=2):
    """1D max pooling along the spatial dimension.

    Parameters
    ----------
    x : jax.Array
        Input of shape (batch, channels, length)
    pool_size : int, optional
        Size of the pooling window. Default 2.
    stride : int, optional
        Stride of the pooling operation. Default 2.

    Returns
    -------
    out : jax.Array
        Pooled output of shape (batch, channels, length // stride)

    Notes
    -----
    After 5 pooling layers with stride 2, a length-96 input becomes length 3:
    96 → 48 → 24 → 12 → 6 → 3
    """
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 1, pool_size),
        window_strides=(1, 1, stride),
        padding="VALID",
    )


def _global_avg_pool_1d(x):
    """Global average pooling over the spatial dimension.

    Reduces a (batch, channels, length) tensor to (batch, channels) by
    averaging over all spatial positions. This creates a fixed-size
    representation regardless of input length.

    Parameters
    ----------
    x : jax.Array
        Input of shape (batch, channels, length)

    Returns
    -------
    out : jax.Array
        Output of shape (batch, channels)
    """
    return jnp.mean(x, axis=-1)


def _batch_norm_1d(x, gamma, beta, running_mean, running_var, eps=1e-5):
    """Apply BatchNorm1d in inference mode (using running statistics).

    Uses pre-computed running mean and variance rather than batch statistics,
    making the output deterministic and independent of other samples in the batch.

    Parameters
    ----------
    x : jax.Array
        Input of shape (batch, channels, length).
    gamma : jax.Array
        Scale parameter (weight) of shape (channels,).
    beta : jax.Array
        Shift parameter (bias) of shape (channels,).
    running_mean : jax.Array
        Running mean of shape (channels,).
    running_var : jax.Array
        Running variance of shape (channels,).
    eps : float, optional
        Small constant for numerical stability. Default 1e-5.

    Returns
    -------
    out : jax.Array
        Normalized output of shape (batch, channels, length).
    """
    # Reshape params for broadcasting: (channels,) -> (1, channels, 1)
    mean = running_mean[None, :, None]
    var = running_var[None, :, None]
    scale = gamma[None, :, None]
    shift = beta[None, :, None]

    return scale * (x - mean) / jnp.sqrt(var + eps) + shift


# =============================================================================
# CNN Forward Pass
# =============================================================================
#
# Architecture overview (CyclicInvariantNet without BatchNorm):
#   Input: (batch, 7, 96) - 7 GX features at 96 arclength points
#   → 5 conv blocks: Conv1D(circular) → ReLU → MaxPool1D(2)
#   → Global average pooling
#   → Concatenate scalar inputs (a/LT, a/Ln)
#   → 2 FC layers with ReLU
#   → Output: log(Q)


def _has_batch_norm(weights):
    """Check if model weights include BatchNorm parameters.

    Supports two naming conventions from different model versions:
    - 'conv_layers.0.bn.weight' (older integrated format)
    - 'conv_batch_norms.0.weight' (current separate format)
    """
    return "conv_layers.0.bn.weight" in weights or "conv_batch_norms.0.weight" in weights


def _apply_conv_bn(x, weights, layer_idx, use_bn):
    """Apply conv block: Conv1D(circular) -> [BatchNorm] -> ReLU -> MaxPool."""
    conv_w = weights[f"conv_layers.{layer_idx}.weight"]
    conv_b = weights[f"conv_layers.{layer_idx}.bias"]
    kernel_size = conv_w.shape[2]

    x = _conv1d_circular(x, conv_w, conv_b, kernel_size)

    if use_bn:
        prefix = f"conv_batch_norms.{layer_idx}"
        x = _batch_norm_1d(
            x,
            weights[f"{prefix}.weight"],
            weights[f"{prefix}.bias"],
            weights[f"{prefix}.running_mean"],
            weights[f"{prefix}.running_var"],
        )

    x = jax.nn.relu(x)
    return _max_pool_1d(x)


def _apply_fc_bn(x, weights, layer_idx, use_bn):
    """Apply FC block: Linear -> [BatchNorm] -> ReLU."""
    x = jnp.dot(x, weights[f"fc_layers.{layer_idx}.weight"].T)
    x = x + weights[f"fc_layers.{layer_idx}.bias"]

    if use_bn:
        # BatchNorm expects 3D input; reshape (batch, features) -> (batch, features, 1)
        prefix = f"fc_batch_norms.{layer_idx}"
        x = _batch_norm_1d(
            x[:, :, None],
            weights[f"{prefix}.weight"],
            weights[f"{prefix}.bias"],
            weights[f"{prefix}.running_mean"],
            weights[f"{prefix}.running_var"],
        )[:, :, 0]

    return jax.nn.relu(x)


def _cyclic_invariant_forward(signals, scalars, weights):
    """Forward pass through CyclicInvariantNet (with or without BatchNorm).

    Automatically detects whether the model uses BatchNorm based on weight keys.

    Parameters
    ----------
    signals : jax.Array
        Input signals of shape (batch, n_features, length) - GX geometric features.
    scalars : jax.Array
        Scalar inputs of shape (batch, n_scalars) - e.g. [a/LT, a/Ln].
    weights : dict
        Model weights (state_dict). Layer counts are inferred from weight keys.
        Supports models with any number of conv and FC layers, with or without
        BatchNorm. Expected key patterns:

        - 'conv_layers.{i}.weight/bias' for i in 0..n_conv-1
        - 'fc_layers.{i}.weight/bias' for i in 0..n_fc-1
        - 'output_layer.weight/bias'
        - Optional: 'conv_batch_norms.{i}.*', 'fc_batch_norms.{i}.*'

    Returns
    -------
    output : jax.Array
        Predicted log(Q) of shape (batch, 1).
    """
    use_bn = _has_batch_norm(weights)

    # Infer layer counts from weight keys
    n_conv = sum(
        1 for k in weights if k.startswith("conv_layers.") and k.endswith(".weight")
    )
    n_fc = sum(
        1 for k in weights if k.startswith("fc_layers.") and k.endswith(".weight")
    )

    # Convolutional blocks: Conv -> [BN] -> ReLU -> MaxPool
    x = signals
    for i in range(n_conv):
        x = _apply_conv_bn(x, weights, i, use_bn)

    # Global average pooling -> (batch, channels)
    x = _global_avg_pool_1d(x)

    # Concatenate with scalar inputs
    x = jnp.concatenate([x, scalars], axis=-1)

    # Fully connected blocks: Linear -> [BN] -> ReLU
    for i in range(n_fc):
        x = _apply_fc_bn(x, weights, i, use_bn)

    # Output layer: predict log(Q)
    x = jnp.dot(x, weights["output_layer.weight"].T) + weights["output_layer.bias"]

    return x


def _ensemble_forward(signals, scalars, weights_list, return_std=False):
    """Ensemble forward pass averaging predictions from multiple models.

    Parameters
    ----------
    signals : jax.Array
        Input signals of shape (batch, 7, length)
    scalars : jax.Array
        Scalar inputs of shape (batch, 2) - [a/LT, a/Ln]
    weights_list : list of dict
        List of weight dictionaries, one per ensemble member
    return_std : bool, optional
        If True, also return std of predictions in log-space. Default False.

    Returns
    -------
    log_Q : jax.Array
        Mean predicted log(Q) of shape (batch, 1)
    std_log_Q : jax.Array, optional
        Std of log(Q) predictions of shape (batch, 1). Only returned if
        return_std=True. This represents ensemble uncertainty in log-space.

    Notes
    -----
    Uses a Python loop over models since ensemble members may have different
    hyperparameters (different channel sizes, kernel sizes, etc.). The averaging
    is done in log-space before returning.
    """
    predictions = []
    for weights in weights_list:
        log_Q = _cyclic_invariant_forward(signals, scalars, weights)
        predictions.append(log_Q)
    stacked = jnp.stack(predictions)  # (n_ensemble, batch, 1)
    mean_log_Q = jnp.mean(stacked, axis=0)
    if return_std:
        std_log_Q = jnp.std(stacked, axis=0)
        return mean_log_Q, std_log_Q
    return mean_log_Q


def _make_jit_forward(weights):
    """Create JIT-compiled forward function with weights captured as constants.

    Parameters
    ----------
    weights : dict
        Model weights dictionary (JAX arrays).

    Returns
    -------
    jit_forward : callable
        JIT-compiled function with signature (signals, scalars) -> log_Q
    """
    @jax.jit
    def jit_forward(signals, scalars):
        return _cyclic_invariant_forward(signals, scalars, weights)
    return jit_forward


# =============================================================================
# Model Loading and Caching
# =============================================================================

# Module-level caches to avoid reloading weights from disk
_NN_WEIGHTS_CACHE = {}
_ENSEMBLE_WEIGHTS_CACHE = {}


def _convert_pytorch_weights(state_dict):
    """Convert PyTorch state dict to JAX-compatible weight dictionary.

    Parameters
    ----------
    state_dict : dict
        PyTorch state dictionary (from torch.load)

    Returns
    -------
    weights : dict
        Dictionary with same keys but JAX arrays as values
    """
    return {k: jnp.array(v.numpy()) for k, v in state_dict.items()}


def _load_nn_weights(model_path):
    """Load PyTorch model weights and create JIT-compiled forward function.

    Uses module-level cache to avoid reloading the same model multiple times.

    Parameters
    ----------
    model_path : str
        Path to the PyTorch checkpoint file (.pt or .pth)

    Returns
    -------
    weights : dict
        Dictionary of JAX arrays containing the model weights
    jit_forward : callable
        JIT-compiled forward function with signature (signals, scalars) -> log_Q

    Raises
    ------
    FileNotFoundError
        If the model file does not exist
    ImportError
        If torch is not installed (required for loading .pt files)
    """
    if model_path in _NN_WEIGHTS_CACHE:
        return _NN_WEIGHTS_CACHE[model_path]

    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Handle both checkpoint format and direct state_dict format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    weights = _convert_pytorch_weights(state_dict)
    jit_forward = _make_jit_forward(weights)
    result = (weights, jit_forward)
    _NN_WEIGHTS_CACHE[model_path] = result
    return result


def _load_ensemble_weights(model_dir, csv_path, top_k=10, pre_method=0, verbose=False):
    """Load top-k ensemble models and create JIT-compiled forward functions.

    Parameters
    ----------
    model_dir : str
        Path to directory containing model .pth files
    csv_path : str
        Path to results.csv with DeepHyper hyperparameter search results
    top_k : int, optional
        Number of top-performing models to load. Default 10.
    pre_method : int, optional
        Preprocessing method to filter models by (0=log transform). Default 0.
    verbose : bool, optional
        If True, print progress during loading. Default False.

    Returns
    -------
    weights_list : list of dict
        List of weight dictionaries, sorted by validation loss (best first)
    jit_forwards : list of callable
        List of JIT-compiled forward functions, one per model

    Raises
    ------
    FileNotFoundError
        If model_dir or csv_path don't exist
    ImportError
        If pandas or torch are not installed

    Notes
    -----
    Supports both BatchNorm and non-BatchNorm models. The JAX forward pass
    auto-detects the model type from weight keys. Architecture is inferred
    from weight shapes rather than requiring hyperparameters.
    """
    import os

    import pandas as pd
    import torch

    df = pd.read_csv(csv_path)

    # Filter out failed runs and convert objective to float
    df_valid = df[df["objective"] != "F"].copy()
    df_valid["objective"] = df_valid["objective"].astype(float)

    # Filter by pre_method if column exists
    if "p:pre_method" in df_valid.columns:
        df_valid = df_valid[df_valid["p:pre_method"] == pre_method]

    # Sort by objective (higher = better = lower validation loss)
    df_sorted = df_valid.sort_values("objective", ascending=False).head(top_k)

    if verbose:
        print(f"Loading {len(df_sorted)} ensemble models from {model_dir}...")

    weights_list = []
    jit_forwards = []
    for idx, task_id in enumerate(df_sorted["m:task_id"]):
        model_path = os.path.join(model_dir, f"model_{task_id}.pth")
        if not os.path.exists(model_path):
            continue
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        weights = _convert_pytorch_weights(state_dict)
        weights_list.append(weights)
        jit_forwards.append(_make_jit_forward(weights))
        if verbose and (idx + 1) % 10 == 0:
            print(f"  Loaded {idx + 1}/{len(df_sorted)} models...")

    if verbose:
        print(f"  Successfully loaded {len(weights_list)}/{len(df_sorted)} models")

    return weights_list, jit_forwards


def _load_ensemble_weights_cached(
    model_dir, csv_path, top_k=10, pre_method=0, verbose=False
):
    """Cached version of _load_ensemble_weights.

    Parameters are the same as _load_ensemble_weights. Uses module-level
    cache keyed by (model_dir, csv_path, top_k, pre_method).
    Verbose is only used on cache miss.

    Returns
    -------
    weights_list : list of dict
        List of weight dictionaries
    jit_forwards : list of callable
        List of JIT-compiled forward functions
    """
    cache_key = (model_dir, csv_path, top_k, pre_method)
    if cache_key not in _ENSEMBLE_WEIGHTS_CACHE:
        _ENSEMBLE_WEIGHTS_CACHE[cache_key] = _load_ensemble_weights(
            model_dir, csv_path, top_k, pre_method, verbose=verbose
        )
    return _ENSEMBLE_WEIGHTS_CACHE[cache_key]


def clear_nn_cache():
    """Clear the neural network weights cache.

    Call this if you need to reload models after they've been modified on disk.
    """
    _NN_WEIGHTS_CACHE.clear()
    _ENSEMBLE_WEIGHTS_CACHE.clear()
