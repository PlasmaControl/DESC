"""Compute functions for trapped energetic particle resonance."""

from quadax import simpson

from desc.backend import jnp
from desc.grid import Grid

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce1D
from ..utils import safediv
from ._neoclassical import _bounce_doc
from .data_index import register_compute_fun


def _alpha_mean(f):
    """Simple mean over field lines.

    Simple mean rather than integrating over α and dividing by 2π
    (i.e. f.T.dot(dα) / dα.sum()), because when the toroidal angle extends
    beyond one transit we need to weight all field lines uniformly, regardless
    of their spacing wrt α.
    """
    return f.mean(axis=0)


def _v_tau(data, B, pitch):
    # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
    # τ is the bounce time, and I is defined in Nemov et al. eq. 36.
    return safediv(2.0, jnp.sqrt(jnp.maximum(jnp.abs(1 - pitch * B), 1e-30)))


def _jnpmean_nz(x, axis=0, fill=jnp.nan):
    """Mean over an axis, ignoring zero and fill-value entries."""
    mask = (x != 0.0) & _is_valid_value(x, fill)
    count = jnp.sum(mask, axis=axis)
    return safediv(jnp.sum(jnp.where(mask, x, 0.0), axis=axis), count, fill=fill)


def _is_valid_value(x, fill_value):
    """Validity mask compatible with finite sentinel fill values."""
    return x != fill_value


def _masked_sum(x, mask, axis=None):
    """Sum x over axis, excluding entries where mask is False."""
    return jnp.sum(jnp.where(mask, x, jnp.zeros_like(x)), axis=axis)


def _build_eta_grid(eq, rhos, alpha_per_rho, zeta, iotas, params):
    """Build a DESC grid with per-rho alpha values derived from uniform eta."""
    from desc.equilibrium.coords import map_coordinates

    num_rho = len(rhos)
    num_eta = alpha_per_rho.shape[1]
    num_zeta = len(zeta)

    # Build raz nodes in meshgrid order: alpha fastest, rho middle, zeta slowest.
    _, rr, zz = jnp.meshgrid(jnp.arange(num_eta), rhos, zeta, indexing="ij")
    alpha_arr = jnp.broadcast_to(
        alpha_per_rho.T[:, :, jnp.newaxis], (num_eta, num_rho, num_zeta)
    )
    raz_nodes = jnp.column_stack(
        [
            rr.flatten(order="F"),
            alpha_arr.flatten(order="F"),
            zz.flatten(order="F"),
        ]
    )

    unique_rho_idx = jnp.arange(num_rho) * num_eta
    unique_poloidal_idx = jnp.arange(num_eta)
    unique_zeta_idx = jnp.arange(num_zeta) * num_rho * num_eta
    inverse_rho_idx = jnp.tile(jnp.repeat(jnp.arange(num_rho), num_eta), num_zeta)
    inverse_poloidal_idx = jnp.tile(jnp.arange(num_eta), num_rho * num_zeta)
    inverse_zeta_idx = jnp.repeat(jnp.arange(num_zeta), num_rho * num_eta)

    raz_grid = Grid(
        nodes=raz_nodes,
        coordinates="raz",
        period=(jnp.inf, jnp.inf, jnp.inf),
        sort=False,
        is_meshgrid=True,
        jitable=True,
        _unique_rho_idx=unique_rho_idx,
        _unique_poloidal_idx=unique_poloidal_idx,
        _unique_zeta_idx=unique_zeta_idx,
        _inverse_rho_idx=inverse_rho_idx,
        _inverse_poloidal_idx=inverse_poloidal_idx,
        _inverse_zeta_idx=inverse_zeta_idx,
    )

    iota_expanded = raz_grid.expand(jnp.atleast_1d(jnp.asarray(iotas)))
    rtz_nodes = map_coordinates(
        eq,
        raz_grid.nodes,
        inbasis=["rho", "alpha", "zeta"],
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, jnp.inf, jnp.inf),
        iota=iota_expanded,
        params=params,
    )

    return Grid(
        nodes=rtz_nodes,
        coordinates="rtz",
        source_grid=raz_grid,
        sort=False,
        jitable=True,
        _unique_rho_idx=unique_rho_idx,
        _inverse_rho_idx=inverse_rho_idx,
    )


def _compute1D(
    fun,
    fun_data,
    data,
    grid,
    num_pitch,
    surf_batch_size=1,
    simp=True,
    pitch_invs=None,
    pitch_inv_weight=None,
):
    """Compute Bounce1D integral quantity with ``fun``.

    Parameters
    ----------
    fun : callable
        Function to compute.
    fun_data : dict[str, jnp.ndarray]
        Data to provide to ``fun``. This dict will be modified.
    data : dict[str, jnp.ndarray]
        DESC data dict.
    grid : Grid
        Grid that can expand and compress.
    num_pitch : int
        Resolution for quadrature over velocity coordinate.
    surf_batch_size : int
        Number of flux surfaces with which to compute simultaneously.
        Default is ``1``.
    simp : bool
        Whether to use an open Simpson rule instead of uniform weights.
    pitch_invs : jnp.ndarray
        If specified, use the given pitch_invs values rather than using num_pitch.
    pitch_inv_weight : jnp.ndarray
        Quadrature weight paired with ``pitch_invs``. If ``pitch_invs`` is given
        without a matching weight, falls back to uniform weighting.

    """
    for name in Bounce1D.required_names:
        fun_data[name] = data[name]
    for name in fun_data:
        fun_data[name] = Bounce1D.reshape(grid, fun_data[name])
    if pitch_invs is None:
        fun_data["pitch_inv"], fun_data["pitch_inv weight"] = Bounce1D.get_pitch_inv_quad(
            grid.compress(data["min_tz |B|"]),
            grid.compress(data["max_tz |B|"]),
            num_pitch,
            simp=simp
        )
    else: # Caller-supplied pitch_invs with matching quadrature weight.
        n = len(pitch_invs)
        if pitch_inv_weight is None:
            pitch_inv_weight = jnp.ones(n) / n
        fun_data["pitch_inv"] = jnp.broadcast_to(pitch_invs, (grid.num_rho, n))
        fun_data["pitch_inv weight"] = jnp.broadcast_to(pitch_inv_weight, (grid.num_rho, n))

    out = batch_map(fun, fun_data, surf_batch_size)

    return out


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


def _s_drift_integrand(data, B, pitch):
    """radial drift integrand for bounce integration in _trapped_EP_resonance objective"""
    return safediv(
        2 * data["cvdrift0"] * (2 - pitch * B), jnp.sqrt(jnp.maximum(jnp.abs(1 - pitch * B), 1e-30))
    )

def _alpha_drift_integrand(data, B, pitch):
    """cross-field-line drift integrand for bounce integration in _trapped_EP_resonance objective"""
    return safediv(
        2 * (data["gbdrift (periodic)"] * pitch * B + 2 * (1 - pitch * B) * data["cvdrift (periodic)"]), jnp.sqrt(jnp.maximum(jnp.abs(1 - pitch * B), 1e-30))
    )

_bounce1D_doc = {
    "num_well": _bounce_doc["num_well"],
    "num_quad": _bounce_doc["num_quad"],
    "num_pitch": _bounce_doc["num_pitch"],
    "surf_batch_size": _bounce_doc["surf_batch_size"],
    "quad": _bounce_doc["quad"],
}


def _phase_space_average(
    vtau_out,
    f_res,
    pitch_inv,
    pitch_inv_weight,
    fl_length,
    num_alpha=None,
    fill_value=jnp.nan,
):
    """Phase-space average of f_res.

    Computes <f_res> = Σ_w ∫dα ∫dλ v·τ_b · f / (2 ∫dα ∫dl/B).
    Pitch quadrature uses Gauss-Legendre weights from
    ``Bounce1D.get_pitch_inv_quad``. 

    Parameters
    ----------
    vtau_out : jnp.ndarray, shape (rho, alpha, Bcrit, well)
        Bounce integral of v·τ.
    f_res : jnp.ndarray, shape (rho, Bcrit, well)
        Objective function per (rho, pitch, well).
    pitch_inv : jnp.ndarray, shape (rho, Bcrit)
        Pitch inverse values.
    pitch_inv_weight : jnp.ndarray, shape (rho, Bcrit)
        Quadrature weights for pitch integration.
    fl_length : jnp.ndarray, shape (rho,)
        Mean-alpha fieldline length, i.e. mean_α ∫ dl/B.
    num_alpha : int or None, optional
        If ``None``, number of field lines considered is consistent with bounce integration
        in _trapped_EP_resonance. If not ``None``, specifies number of total field lines to consider.
        Defaults to ``None``.
    fill_value : float, optional
        Value to set bounce integration outputs to if no well is found. Cannot use jnp.nan to retain optimization abilities.
        Cannot use 0 for confusion with other quantities and averages.
        Defaults to 11.0.

    Returns
    -------
    f_res_avg : jnp.ndarray, shape (rho,)
    """
    if num_alpha is None:
        num_alpha = vtau_out.shape[1]
    # Zero out BT-filtered (fill_value sentinel) f_res before weighting by vtau.
    f_res_clean = jnp.where(
        _is_valid_value(f_res, fill_value), f_res, jnp.zeros_like(f_res)
    )
    integrand = vtau_out * f_res_clean[:, jnp.newaxis, :, :]
    # 1. Integrate over pitch (per α, per well): ∫dλ g(λ) = ∫dp g(1/p)/p²
    pitch_inv_4d = pitch_inv[:, jnp.newaxis, :, jnp.newaxis]
    pitch_mask = _is_valid_value(pitch_inv_4d, fill_value)
    integrand_mask = _is_valid_value(vtau_out, fill_value) & pitch_mask
    safe_pitch_inv = jnp.where(pitch_mask, pitch_inv_4d, jnp.ones_like(pitch_inv_4d))
    pitch_integrated = _masked_sum(
        integrand
        * pitch_inv_weight[:, jnp.newaxis, :, jnp.newaxis]
        / safe_pitch_inv ** 2,
        mask=integrand_mask,
        axis=2,
    )  # (rho, alpha, well)
    # 2. Sum over α (discrete ∫dα)
    alpha_summed = pitch_integrated.sum(axis=1)  # (rho, well)
    # 3. Sum over wells
    numerator = _masked_sum(
        alpha_summed,
        mask=_is_valid_value(alpha_summed, fill_value),
        axis=-1,
    )  # (rho,)
    # Denominator: 2 · Σ_α ∫dl/B = 2 · N_α · mean_α(∫dl/B)
    return safediv(numerator, 2 * num_alpha * fl_length)


def _resonance_physics(
    alpha_drift_out, s_drift_out, vtau_out,
    iotas, rhos, rho_res, KE_frac, nfp, M, N,
    res_arr, q_arr, eta_vals, eta_res,
    weight_method, Delta_Omega, wd_blur, fill_value, stab_sacrifice,
    cropping_DOmega=False,
):
    """Compute resonance frequencies, weights, island widths, and f_res.

    Takes bounce-averaged drifts and converts them to physical frequencies,
    computes the normalised precession frequency Omega and its radial
    derivative Omega'(s), assigns resonance weights, evaluates Fourier
    coefficients of the radial drift, and finally computes island widths.

    Parameters
    ----------
    alpha_drift_out : jnp.ndarray, shape (rho, alpha, Bcrit, well)
        Bounce-averaged poloidal drift (dimensionless, before energy scaling).
    s_drift_out : jnp.ndarray, shape (rho, alpha, Bcrit, well)
        Bounce-averaged radial drift (dimensionless, before energy scaling).
    vtau_out : jnp.ndarray, shape (rho, alpha, Bcrit, well)
        Bounce integral of v·τ.
    iotas : jnp.ndarray, shape (rho,)
        Rotational transform per surface.
    rhos : jnp.ndarray, shape (rho,)
        Flux surface labels.
    rho_res : float
        Radial grid spacing.
    KE_frac : float
        Fraction of 3.5 MeV kinetic energy.
    nfp : int
        Number of field periods.
    M, N : int
        Poloidal and toroidal mode numbers for resonance condition.
    res_arr : jnp.ndarray, shape (res,)
        Resonance frequency ratios p/q.
    q_arr : jnp.ndarray, shape (res,)
        Toroidal mode numbers of resonances.
    eta_vals : jnp.ndarray, shape (num_eta,)
        Uniform eta grid on [0, 2π).
    eta_res : float
        Grid spacing for eta.
    weight_method : str
        ``"linear"`` or ``"bump"`` resonance weighting.
    Delta_Omega : float or None
        Half-width for bump weighting.
    wd_blur : float
        Multiplicative blur factor used to compute bump half-width from
        adjacent-surface Omega spacing when ``Delta_Omega`` is not provided.
    stab_sacrifice : bool
        Whether to sacrifice accuracy for stability in island widths.
    cropping_DOmega : bool
        If ``True``, Delta_Omega calculation is clipped by 0.01 * max(Omega_eta) < Delta_Omega < 0.10 * max(Omega_eta).
        This must be when using the ``bump`` weighting method and Delta_Omega = None case.
        Otherwise this quantity is ignored.
        Defaults to ``False``.

    Returns
    -------
    result : dict
        Dictionary with keys: ``f_res``, ``Omega``, ``omega_bounce_avg``,
        ``eta_drift_avg``, ``omega_bounce``, ``eta_drift``,
        ``Omega_prime_s``, ``res_weight``, ``f_q_c``, ``f_q_s``,
        ``f_q_abs``, ``Delta_s``,  ``s_res``,
        ``f_res``.
    """
    m_alpha = 6.6446573450e-27
    e_charge = 1.602e-19
    Z = 2
    KE = KE_frac * 5.6076e-13
    v2 = 2 * KE / m_alpha

    # Bounce-averaged drifts → physical frequencies
    alpha_drift = alpha_drift_out * KE / (Z * e_charge)
    ado_shape = alpha_drift.shape
    iotas_omega = jnp.broadcast_to(
        iotas[..., None, None, None],
        (iotas.shape[0], ado_shape[1], ado_shape[2], ado_shape[3]),
    )
    eta_drift = safediv(
        nfp * alpha_drift, N * nfp - iotas_omega * M, fill=fill_value
    )

    s_drift = s_drift_out * KE / (Z * e_charge)
    tau_bounce = vtau_out / jnp.sqrt(v2)
    omega_bounce = safediv(2 * jnp.pi, tau_bounce, fill=fill_value)

    # Require particle to be trapped at all alpha/eta values for a given
    # (rho, pitch, well). 
    all_alpha_valid = (
        _is_valid_value(omega_bounce, fill_value)).all(axis=1)  # (rho, pitch, well)

    # Alpha-averaged frequencies → normalized precession Omega
    omega_bounce_avg = _jnpmean_nz(omega_bounce, axis=1, fill=fill_value)
    eta_drift_avg = _jnpmean_nz(eta_drift, axis=1, fill=fill_value)
    Omega = safediv(eta_drift_avg, omega_bounce_avg, fill=fill_value)
    valid = (
        _is_valid_value(eta_drift_avg, fill_value)
        & _is_valid_value(omega_bounce_avg, fill_value)
        & all_alpha_valid
    )
    Omega = jnp.where(valid, Omega, fill_value)

    # Omega'(s) via finite differences
    # Use "double-where" to replace invalid Omega with 0 before arithmetic,
    # preventing NaN gradients from flowing through unused jnp.where branches.
    Omega_safe = jnp.where(valid, Omega, 0.0)
    valid_prev = jnp.concatenate(
        [jnp.zeros((1,) + valid.shape[1:], dtype=bool), valid[:-1]], axis=0
    )
    valid_next = jnp.concatenate(
        [valid[1:], jnp.zeros((1,) + valid.shape[1:], dtype=bool)], axis=0
    )
    Omega_prev_safe = jnp.concatenate(
        [jnp.zeros((1,) + Omega.shape[1:]), Omega_safe[:-1]], axis=0
    )
    Omega_next_safe = jnp.concatenate(
        [Omega_safe[1:], jnp.zeros((1,) + Omega.shape[1:])], axis=0
    )
    grad_central = (Omega_next_safe - Omega_prev_safe) / (2 * rho_res)
    grad_forward = (Omega_next_safe - Omega_safe) / rho_res
    grad_backward = (Omega_safe - Omega_prev_safe) / rho_res
    dOmega_drho = jnp.where(
        valid & valid_prev & valid_next,
        grad_central,
        jnp.where(
            valid & valid_next & ~valid_prev,
            grad_forward,
            jnp.where(
                valid & valid_prev & ~valid_next,
                grad_backward,
                fill_value,
            ),
        ),
    )
    Omega_prime_s = safediv(
        dOmega_drho, 2 * rhos[:, None, None], fill=fill_value
    )
    Omega_prime_s = jnp.where(
        _is_valid_value(dOmega_drho, fill_value), Omega_prime_s, fill_value
    )

    # Resonance weights
    Omega_broad = Omega[..., None]
    res_broad = res_arr[None, None, None, :]

    # Indices are valid if Omega_prime_s is valid and Omega is valid. 
    valid_prime = valid & _is_valid_value(Omega_prime_s, fill_value)

    if weight_method == "bump":
        if Delta_Omega is None:
            Omega_safe_bump = jnp.where(valid, Omega, 0.0)
            Omega_prev_b = Omega_safe_bump[:-1, :, :]
            Omega_next_b = Omega_safe_bump[1:, :, :]
            valid_pair = jnp.logical_and(valid[:-1, :, :], valid[1:, :, :])
            domega_arr = jnp.where(
                valid_pair,
                jnp.abs(Omega_next_b - Omega_prev_b),
                0.0,
            )  # (rho-1, pitch, well)
            from desc.objectives.utils import softmax as _softmax
            Delta_Omega_val = (
                wd_blur * _softmax(domega_arr, alpha=50, axis=0) / 2.0
            )[None, :, :, None]
            if cropping_DOmega:
                # Delta_Omega_val needs to be cropped if resolution is too low or Omega_eta shear is too high
                Omega_max = _softmax(Omega_safe_bump, alpha=50, axis=0)[None, :, :, None]
                Delta_Omega_val_max = 0.1 * Omega_max  # DeltaOmega < 10% of maximum Omega
                Delta_Omega_val_min = 0.01 * Omega_max  # DeltaOmega > 1% of maximum Omega
                Delta_Omega_val = jnp.where(Delta_Omega_val > Delta_Omega_val_max, Delta_Omega_val_max, Delta_Omega_val)
                Delta_Omega_val = jnp.where(Delta_Omega_val < Delta_Omega_val_min, Delta_Omega_val_min, Delta_Omega_val)
        else:
            Delta_Omega_val = Delta_Omega
        a = res_broad + Delta_Omega_val
        b = res_broad - Delta_Omega_val
        in_interval = (Omega_broad >= b) & (Omega_broad <= a)
        denom = (Omega_broad - b) * (Omega_broad - a)
        exp_arg = safediv(
            (2.0 * Delta_Omega_val) ** 2, denom, fill=-1e10
        )
        C_norm = safediv(71.12518788738504, Delta_Omega_val, fill=0.0)
        w_raw = rho_res * C_norm * jnp.abs(dOmega_drho[..., None]) * jnp.exp(exp_arg)
        # Weight is non-zero only if in interval and valid 
        res_weight = jnp.where(in_interval & valid_prime[..., None], w_raw, 0)
        # Normalize res_weight to sum to 1 
        # res_weight = safediv(res_weight, res_weight.sum(axis=0), fill=0.0)
    else:
        # Double-where: use Omega_safe (0 at invalid entries) so that
        # safediv never sees fill_value operands, preventing NaN gradients.
        Omega_safe_lin = jnp.where(valid, Omega, 0.0)
        Omega_prev_lin = jnp.concatenate(
            [jnp.zeros_like(Omega_safe_lin[:1]), Omega_safe_lin[:-1]], axis=0
        )[..., None]
        Omega_next_lin = jnp.concatenate(
            [Omega_safe_lin[1:], jnp.zeros_like(Omega_safe_lin[:1])], axis=0
        )[..., None]
        Omega_broad_safe = Omega_safe_lin[..., None]
        valid_next_lin = jnp.concatenate(
            [valid_prime[1:], jnp.zeros_like(valid_prime[:1])], axis=0
        )[..., None]
        valid_prev_lin = jnp.concatenate(
            [jnp.zeros_like(valid_prime[:1]), valid_prime[:-1]], axis=0
        )[..., None]
        between_next = valid_next_lin & (
            ((Omega_next_lin >= res_broad) & (res_broad >= Omega_broad_safe))
            | ((Omega_next_lin <= res_broad) & (res_broad <= Omega_broad_safe))
        )
        w_next = safediv(
            Omega_next_lin - res_broad, Omega_next_lin - Omega_broad_safe, fill=0.0
        )
        between_prev = valid_prev_lin & (
            ((Omega_prev_lin >= res_broad) & (res_broad >= Omega_broad_safe))
            | ((Omega_prev_lin <= res_broad) & (res_broad <= Omega_broad_safe))
        )
        w_prev = safediv(
            Omega_prev_lin - res_broad, Omega_prev_lin - Omega_broad_safe, fill=0.0
        )
        res_weight = jnp.where(
            between_next, w_next, jnp.where(between_prev, w_prev, 0.0)
        )

    # Set weight to zero for invalid points. 
    res_weight = jnp.where(valid_prime[..., None], res_weight, 0)

    # Fourier analysis of radial drift.
    # Only perform FT if all eta points are valid.
    # Mask out fill_value entries in s_drift (set by bt/rt filters) so they
    # don't contaminate the alpha sum.
    s_drift_valid = jnp.where(_is_valid_value(s_drift_out, fill_value), s_drift, 0.0)
    ft_integrand = s_drift_valid * tau_bounce

    phase = q_arr[None, :] * eta_vals[:, None]
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    ft_cos = ft_integrand[..., None] * cos_phase[None, :, None, None, :]
    ft_sin = ft_integrand[..., None] * sin_phase[None, :, None, None, :]
    ft_prefactor = eta_res / jnp.pi
    f_q_c = ft_prefactor * jnp.sum(ft_cos, axis=1)
    f_q_s = ft_prefactor * jnp.sum(ft_sin, axis=1)
    f_q_abs = 0.5 * jnp.sqrt(jnp.maximum(f_q_c**2 + f_q_s**2, 1e-30))
    f_q_abs_sq = 0.5**2 * (f_q_c**2 + f_q_s**2)

    # Filter FT results to valid points. 
    f_q_abs = jnp.where(valid_prime[..., None], f_q_abs, 0.0)

    # Island widths
    q_iw = q_arr[None, None, None, :]
    denom = jnp.pi * q_iw * jnp.abs(Omega_prime_s[..., None])
    Delta_s_profile = 4 * jnp.sqrt(
        jnp.maximum(safediv(f_q_abs, denom, fill=0.0), 1e-30)
    )
    Delta_s_4_profile = 4**4 * safediv(f_q_abs_sq, denom**2, fill=0.0)
    Delta_s_4_sum = (Delta_s_4_profile * res_weight).sum(axis=-1)

    if stab_sacrifice:
        f_res = Delta_s_4_sum * Omega_prime_s**2 
    else:
        f_res = Delta_s_4_sum

    # Sum over radius to get weighted island width and resonance location. 
    Delta_s = (Delta_s_profile * res_weight).sum(axis=0)
    s_vals = rhos**2
    s_res = (res_weight * s_vals[:, None, None, None]).sum(axis=0)

    return {
        'f_res': f_res,  # (rho, pitch, well)
        'Omega': Omega,  # (rho, pitch, well)
        'omega_bounce_avg': omega_bounce_avg,  # (rho, pitch, well)
        'eta_drift_avg': eta_drift_avg,  # (rho, pitch, well)
        'omega_bounce': omega_bounce,  # (rho, alpha, pitch, well)
        'eta_drift': eta_drift,  # (rho, alpha, pitch, well)
        'Omega_prime_s': Omega_prime_s,  # (rho, pitch, well)
        'res_weight': res_weight,  # (rho, pitch, well, res)
        'f_q_abs': f_q_abs,  # (rho, pitch, well, res)
        'Delta_s': Delta_s,  # (pitch, well, res), rho-weighted diagnostic
        'Delta_s_prof': Delta_s_profile,  # (rho, pitch, well, res)
        's_res': s_res,  # (pitch, well, res), rho-weighted resonance location
        'valid_prime': valid_prime,  # (rho, pitch, well)
    }


@register_compute_fun(
    name="trapped EP resonance",
    label=(
        "Trapped Energetic Particle Resonance Objective Function"
    ),
    units="s^-2",
    units_long="seconds squared",
    description="Trapped Energetic Particle Resonance Minimizer",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["iota", "min_tz |B|", "max_tz |B|", "Psi"],
    source_grid_requirement={},
    public=False,
    **_bounce1D_doc,
)
def _trapped_EP_resonance(params, transforms, profiles, data, **kwargs):
    """Trapped particle resonance penalty.

    Three stages:
      0. Construct 3D eta and PSA grids; evaluate field data on them.
      1. Bounce integrals  (per-surface, via ``_compute1D`` / ``batch_map``)
      2. Resonance physics  (cross-surface, via ``_resonance_physics``)
      3. Phase-space average (via ``_phase_space_average``)
    """
    from .utils import _compute as compute_fun
    from .utils import get_transforms, get_profiles, _parse_parameterization
    from quadax import simpson

    num_pitch = kwargs.get("num_pitch", None)
    num_well = 1
    M = kwargs.get("M", 1)
    N = kwargs.get("N", 1)
    nfp = kwargs.get("nfp", None)
    KE_frac = kwargs.get("KE_frac", None)
    pitch_invs = kwargs.get("pitch_invs", None)
    rho_res = kwargs.get("rho_res", None)
    eta_res = kwargs.get("eta_res", None)
    res_arr = kwargs.get("res_arr", None)
    p_arr = kwargs.get("p_arr", None)
    q_arr = kwargs.get("q_arr", None)
    quad = kwargs.get("quad", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    pitch_batch_size = kwargs.get("pitch_batch_size", 1)
    num_eta = kwargs.get("num_eta", None)
    weight_method = kwargs.get("weight_method", "linear")
    Delta_Omega = kwargs.get("Delta_Omega", None)
    wd_blur = kwargs.get("wd_blur", 1.25)
    fill_value = kwargs.get("fill_value", 11)
    eq = kwargs.get("eq", None)
    zeta = kwargs.get("zeta", None)
    stab_sacrifice = kwargs.get("stab_sacrifice", False)
    bt_filter_flag = kwargs.get("bt_filter_flag", False)
    cropping_DOmega = kwargs.get("cropping_DOmega", False)

    # --- 0. Build 3D grids and evaluate field data on them ---
    base_grid = transforms["grid"]
    iotas = base_grid.compress(data["iota"])
    rhos = base_grid.compress(base_grid.nodes[:, 0])

    eta_vals = jnp.linspace(0, 2 * jnp.pi, num_eta, endpoint=False)
    ft_denom = N * nfp - iotas * M
    alpha_per_rho = eta_vals[None, :] * ft_denom[:, None] / nfp

    eta_desc_grid = _build_eta_grid(eq, rhos, alpha_per_rho, zeta, iotas, params)
    eta_grid = eta_desc_grid.source_grid

    alpha_psa = jnp.linspace(0, 2 * jnp.pi, num_eta, endpoint=False)
    psa_desc_grid = eq._get_rtz_grid(
        rhos, alpha_psa, zeta,
        coordinates="raz",
        iota=iotas,
        params=params,
    )
    psa_grid = psa_desc_grid.source_grid

    eta_data_keys = (
        list(Bounce1D.required_names)
        + ["cvdrift0", "gbdrift (periodic)", "cvdrift (periodic)",
           "iota", "min_tz |B|", "max_tz |B|"]
    )
    psa_bounce_keys = (
        list(Bounce1D.required_names)
        + ["min_tz |B|", "max_tz |B|", "|B|"]
    )
    all_needed_keys = list(set(eta_data_keys + psa_bounce_keys))

    # Pre-compute all transitive dependencies on the base grid (which has
    # spacing for surface integrals).  This gives us 1D intermediates like
    # iota_den, iota_num, Psi, etc. that the 3D grids cannot compute.
    internal_profiles = get_profiles(all_needed_keys, eq)
    base_data = compute_fun(
        eq, all_needed_keys, params,
        get_transforms(all_needed_keys, eq, base_grid, jitable=True),
        internal_profiles,
        data=data,
    )

    # Seed only per-surface (coordinates="r") quantities onto the 3D grids.
    # 3D quantities will be recomputed with proper angular resolution.
    from .data_index import data_index as _data_index
    _p = _parse_parameterization(eq)
    seed_1d = {}
    for key, val in base_data.items():
        entry = _data_index.get(_p, {}).get(key, None)
        if entry is not None and entry.get("coordinates", "") == "r":
            seed_1d[key] = val

    eta_seed = {
        key: eta_desc_grid.copy_data_from_other(val, base_grid)
        for key, val in seed_1d.items()
    }
    data_eta = compute_fun(
        eq, eta_data_keys, params,
        get_transforms(eta_data_keys, eq, eta_desc_grid, jitable=True),
        internal_profiles,
        data=eta_seed,
    )

    psa_seed = {
        key: psa_desc_grid.copy_data_from_other(val, base_grid)
        for key, val in seed_1d.items()
    }
    data_psa = compute_fun(
        eq, psa_bounce_keys, params,
        get_transforms(psa_bounce_keys, eq, psa_desc_grid, jitable=True),
        internal_profiles,
        data=psa_seed,
    )

    # --- 1. Bounce integrals on the eta grid ---
    # Build a global pitch grid from the base grid's min/max |B|, which is
    # computed from the equilibrium's full Fourier resolution and is independent
    # of num_transit.  This avoids the pitch grid shifting with num_transit
    # when the eta-grid field lines sample more of the surface at higher nt.
    # Use Bounce1D.get_pitch_inv_quad with simp=True (genuine Simpson quadrature)
    # so the returned weight is paired correctly with the pitch values, rather
    # than discarding the quadrature weight and re-deriving a uniform one.
    # Note simpson2 rounds num_pitch to an odd count and adds 2 boundary points,
    # so the actual pitch count may differ slightly from num_pitch.
    if pitch_invs is None:
        B_min_base = base_grid.compress(base_data["min_tz |B|"])  # (num_rho,)
        B_max_base = base_grid.compress(base_data["max_tz |B|"])  # (num_rho,)
        B_min_scalar = jnp.min(B_min_base)
        B_max_scalar = jnp.max(B_max_base)
        p_global, w_global = Bounce1D.get_pitch_inv_quad(
            jnp.array([B_min_scalar]), jnp.array([B_max_scalar]), num_pitch, simp=True
        )
        pitch_invs_use = p_global[0]
        pitch_inv_weight_use = w_global[0]
    else:
        pitch_invs_use = pitch_invs
        pitch_inv_weight_use = None

    def drifts(data_in):
        bounce = Bounce1D(eta_grid, data_in, quad, is_reshaped=True)
        points = bounce.points(data_in["pitch_inv"], num_well=num_well)
        v_tau, _alpha_drift, _s_drift = bounce.integrate(
            [_v_tau, _alpha_drift_integrand, _s_drift_integrand],
            data_in["pitch_inv"],
            data_in,
            ["cvdrift0", "gbdrift (periodic)", "cvdrift (periodic)"],
            num_well=num_well,
        )
        _alpha_drift = safediv(_alpha_drift, v_tau)
        _s_drift = safediv(_s_drift, v_tau)
        return _alpha_drift, _s_drift, points, v_tau, data_in

    alpha_drift_out, s_drift_out, points, vtau_out, _data = _compute1D(
        drifts,
        {
            "cvdrift0": data_eta["cvdrift0"],
            "gbdrift (periodic)": data_eta["gbdrift (periodic)"],
            "cvdrift (periodic)": data_eta["cvdrift (periodic)"],
        },
        data_eta,
        eta_grid,
        num_pitch,
        surf_batch_size,
        pitch_invs=pitch_invs_use,
        pitch_inv_weight=pitch_inv_weight_use,
    )

    # --- 1b. Barely-trapped filter ---
    # Zero out wells whose poloidal bounce width delta_chi > 2.5*pi.
    if bt_filter_flag:
        points_0 = points[0]  # zeta at bounce start, shape (rho, alpha, pitch, well)
        points_1 = points[1]  # zeta at bounce end
        iotas_bc = jnp.broadcast_to(
            iotas[..., None, None, None],
            (iotas.shape[0], points_0.shape[1], points_0.shape[2], points_0.shape[3]),
        )
        delta_chi = jnp.abs(jnp.abs(points_0 - points_1) * (M * iotas_bc - N * nfp))
        s_drift_out = jnp.where(delta_chi < float(2 * jnp.pi), s_drift_out, fill_value)

    # --- 2. Resonance physics (cross-surface) ---
    res = _resonance_physics(
        alpha_drift_out, s_drift_out, vtau_out,
        iotas, rhos, rho_res, KE_frac, nfp, M, N,
        res_arr, q_arr, eta_vals, eta_res,
        weight_method, Delta_Omega, wd_blur, fill_value,
        stab_sacrifice, cropping_DOmega,
        )

    # --- 3. Phase-space average on the PSA grid (uniform in alpha) ---
    # Skip PSA when custom pitch_invs are provided.
    if pitch_invs is None:
        num_alpha_psa = psa_grid.num_poloidal

        def drifts_vtau(data_local):
            bounce = Bounce1D(psa_grid, data_local, quad, is_reshaped=True)
            v_tau = bounce.integrate(
                [_v_tau],
                data_local["pitch_inv"],
                data_local,
                [],
                num_well=num_well,
            )[0]
            return v_tau, data_local

        vtau_psa, _data_psa = _compute1D(
            drifts_vtau,
            {},
            data_psa,
            psa_grid,
            num_pitch,
            surf_batch_size,
            pitch_invs=pitch_invs_use,
            pitch_inv_weight=pitch_inv_weight_use,
        )
        num_rho_psa = psa_grid.num_rho
        if vtau_psa.ndim == 3 and vtau_psa.shape[0] == num_rho_psa * num_alpha_psa:
            vtau_psa = vtau_psa.reshape(
                num_rho_psa, num_alpha_psa, vtau_psa.shape[1], vtau_psa.shape[2]
            )

        # Field-line length ∫dl/B = ∫dζ/B^ζ, matching both the
        # registered "fieldline length" quantity (_geometry.py) and the dl/dζ
        # Jacobian Bounce1D.integrate applies internally. Integrate over the
        # full multi-transit ζ domain and mean over α
        Bzeta_psa = Bounce1D.reshape(psa_grid, data_psa["B^zeta"])
        num_transit = kwargs.get("num_transit", 1)
        n_1t = len(zeta) // num_transit
        # Normalize by length of single toroidal transit. 
        fl_length = jnp.abs(simpson(1 / Bzeta_psa[..., :n_1t], x=zeta[:n_1t], axis=-1).mean(axis=1))

        f_res_avg = _phase_space_average(
            vtau_psa,
            res['f_res'],
            _data_psa["pitch_inv"],
            _data_psa["pitch_inv weight"],
            fl_length,
            num_alpha=num_alpha_psa,
            fill_value=fill_value,
        )
        data["trapped EP resonance"] = base_grid.expand(f_res_avg)
    else: # Custom pitch_invs specified: skip phase-space average,
          # just return the raw resonance physics results
        data["trapped EP resonance"] = {
            **res,
            'pitch_inv': _data['pitch_inv'],
            'res_arr': res_arr,
            'p_arr': p_arr,
            'q_arr': q_arr,
            'rhos': rhos,
        }

    return data
