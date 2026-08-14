"""Compute functions for trapped energetic particle resonance."""

from quadax import simpson

from desc.backend import jax, jnp
from desc.grid import Grid

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce1D, Bounce2D
from ..utils import safediv
from ._fast_ion import _radial_drift
from ._neoclassical import _bounce_doc
from .data_index import register_compute_fun


def _v_tau(data, B, pitch):
    # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
    # τ is the bounce time, and I is defined in Nemov et al. eq. 36.
    return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


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


def _build_eta_source_grid(rhos, alpha_per_rho, zeta):
    """Build the (rho, alpha, zeta) grid with alpha derived from uniform eta.

    This is the field line following grid that ``Bounce1D`` integrates along.
    It is built from array arithmetic alone, unlike the (rho, theta, zeta)
    grid of ``_build_eta_grid``, which needs a coordinate map.
    """
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

    return Grid(
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


def _build_eta_grid(eq, rhos, alpha_per_rho, zeta, iotas, params):
    """Build a DESC grid with per-rho alpha values derived from uniform eta."""
    from desc.equilibrium.coords import map_coordinates

    raz_grid = _build_eta_source_grid(rhos, alpha_per_rho, zeta)

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
        _unique_rho_idx=raz_grid.unique_rho_idx,
        _inverse_rho_idx=raz_grid.inverse_rho_idx,
    )


def _compute2D(
    fun,
    fun_data,
    data,
    grid,
    angle,
    alpha_per_rho,
    num_pitch,
    surf_batch_size=1,
    simp=True,
    pitch_invs=None,
    pitch_inv_weight=None,
):
    """Compute Bounce2D integral quantity with ``fun``.

    The Bounce2D analogue of ``_compute1D``. ``fun_data`` and ``data`` are
    given on a tensor-product (θ, ζ) grid rather than on a field line grid,
    and are Fourier transformed here for ``Bounce2D`` to interpolate onto
    field lines internally. This is what avoids materializing the large
    field-line grids that ``Bounce1D`` requires.

    Parameters
    ----------
    fun : callable
        Function to compute. Receives the batched data dictionary, whose
        arrays hold Fourier transforms (pass ``is_fourier=True`` to
        ``Bounce2D``).
    fun_data : dict[str, jnp.ndarray]
        Data to Fourier transform and pass to ``fun``. Modified in place.
    data : dict[str, jnp.ndarray]
        DESC data dict evaluated on ``grid``.
    grid : Grid
        Tensor-product (ρ, θ, ζ) grid satisfying ``can_fft2``.
    angle : jnp.ndarray
        Shape (num ρ, X, Y). Angle returned by ``Bounce2D.angle``.
    alpha_per_rho : jnp.ndarray
        Shape (num ρ, num α). Field line labels, which differ between flux
        surfaces because they derive from the omnigenity angle η. Stored
        with ρ leading so that ``batch_map`` slices it consistently with the
        rest of the data; ``fun`` transposes it back to (num α, num ρ).
    num_pitch : int
        Resolution for quadrature over velocity coordinate.
    surf_batch_size : int
        Number of flux surfaces with which to compute simultaneously.
    simp : bool
        Whether to use an open Simpson rule instead of uniform weights.
    pitch_invs : jnp.ndarray
        If specified, use these pitch_inv values rather than ``num_pitch``.
    pitch_inv_weight : jnp.ndarray
        Quadrature weight paired with ``pitch_invs``.

    """
    for name in Bounce2D.required_names:
        fun_data[name] = data[name]
    # iota is per-surface, not a (θ, ζ) field, so it must stay out of the
    # transform below and come back compressed instead.
    fun_data.pop("iota")
    for name in fun_data:
        fun_data[name] = Bounce2D.fourier(Bounce2D.reshape(grid, fun_data[name]))
    fun_data["iota"] = grid.compress(data["iota"])
    fun_data["angle"] = angle
    fun_data["alpha_per_rho"] = alpha_per_rho
    if pitch_invs is None:
        # A single B_crit set is shared across flux surfaces, so broadcast the
        # global extrema rather than using each surface's own.
        num_rho = grid.num_rho
        B_min = jnp.min(grid.compress(data["min_tz |B|"]))
        B_max = jnp.max(grid.compress(data["max_tz |B|"]))
        (
            fun_data["pitch_inv"],
            fun_data["pitch_inv weight"],
        ) = Bounce2D.get_pitch_inv_quad(
            jnp.full(num_rho, B_min), jnp.full(num_rho, B_max), num_pitch, simp=simp
        )
    else:
        n = len(pitch_invs)
        fun_data["pitch_inv"] = jnp.broadcast_to(pitch_invs, (grid.num_rho, n))
        fun_data["pitch_inv weight"] = jnp.broadcast_to(
            (
                jnp.ones(n) * (2 * jnp.pi / n)
                if pitch_inv_weight is None
                else pitch_inv_weight
            ),
            (grid.num_rho, n),
        )
    return batch_map(fun, fun_data, surf_batch_size)


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
        (
            fun_data["pitch_inv"],
            fun_data["pitch_inv weight"],
        ) = Bounce1D.get_pitch_inv_quad(
            grid.compress(data["min_tz |B|"]),
            grid.compress(data["max_tz |B|"]),
            num_pitch,
            simp=simp,
        )
    else:  # Caller-supplied pitch_invs with matching quadrature weight.
        n = len(pitch_invs)
        if pitch_inv_weight is None:
            pitch_inv_weight = jnp.ones(n) / n
        fun_data["pitch_inv"] = jnp.broadcast_to(pitch_invs, (grid.num_rho, n))
        fun_data["pitch_inv weight"] = jnp.broadcast_to(
            pitch_inv_weight, (grid.num_rho, n)
        )

    out = batch_map(fun, fun_data, surf_batch_size)

    return out


def _alpha_drift_integrand(data, B, pitch):
    """Cross-field-line drift integrand for bounce integration.

    Used in ``_trapped_EP_resonance``.
    """
    return safediv(
        2
        * (
            data["gbdrift (periodic)"] * pitch * B
            + 2 * (1 - pitch * B) * data["cvdrift (periodic)"]
        ),
        jnp.sqrt(jnp.abs(1 - pitch * B)),
    )


# Alpha particle (⁴He nucleus) constants, used to turn bounce-averaged
# drifts into physical frequencies.
_M_ALPHA = 6.6446573450e-27  # mass, kg
_E_CHARGE = 1.602e-19  # elementary charge, C
_Z_ALPHA = 2  # charge number
# 3.5 MeV, the birth energy of alpha particles from D-T fusion.
_E_BIRTH = 5.6076e-13  # J

_BOUNCE_INTEGRAND_KEYS = ("cvdrift0", "gbdrift (periodic)", "cvdrift (periodic)")
_ETA_BOUNCE_KEYS = tuple(Bounce1D.required_names) + _BOUNCE_INTEGRAND_KEYS
_FFT_BOUNCE_KEYS = tuple(Bounce2D.required_names) + _BOUNCE_INTEGRAND_KEYS


def _global_pitch_quad(base_grid, data, num_pitch, pitch_invs):
    """Pitch grid shared by every flux surface.

    Built from the base grid's min/max |B|, which comes from the equilibrium's
    full Fourier resolution and so does not move with ``num_transit``.

    Returns
    -------
    pitch_inv, pitch_inv_weight : tuple[jnp.ndarray]
        ``pitch_inv_weight`` is ``None`` when ``pitch_invs`` was supplied, in
        which case ``_compute1D`` falls back to uniform weights.

    """
    if pitch_invs is not None:
        return pitch_invs, None
    B_min = jnp.min(base_grid.compress(data["min_tz |B|"]))
    B_max = jnp.max(base_grid.compress(data["max_tz |B|"]))
    pitch_inv, weight = Bounce1D.get_pitch_inv_quad(
        jnp.array([B_min]), jnp.array([B_max]), num_pitch, simp=True
    )
    return pitch_inv[0], weight[0]


def _barely_trapped_filter(s_drift, points, iotas, M, N, nfp, fill_value):
    """Zero out wells whose poloidal bounce width exceeds 2π.

    Those particles are barely trapped: they sample the whole poloidal angle
    within one bounce, so the resonance analysis does not describe them.

    Parameters
    ----------
    s_drift : jnp.ndarray
        Shape (rho, alpha, Bcrit, well). Bounce-averaged radial drift.
    points : tuple[jnp.ndarray]
        ζ at the start and end of each bounce.

    Returns
    -------
    s_drift : jnp.ndarray
        ``s_drift`` with the filtered wells set to ``fill_value``.

    """
    z1, z2 = points
    delta_chi = jnp.abs(jnp.abs(z1 - z2) * (M * iotas[:, None, None, None] - N * nfp))
    return jnp.where(delta_chi < 2 * jnp.pi, s_drift, fill_value)


def _frequencies(
    alpha_drift_out, s_drift_out, vtau_out, iotas, KE_frac, nfp, M, N, fill_value
):
    """Turn bounce-averaged drifts into frequencies and precession Omega.

    Parameters
    ----------
    alpha_drift_out, s_drift_out : jnp.ndarray
        Shape (rho, alpha, Bcrit, well).
        Bounce-averaged poloidal and radial drift, before energy scaling.
    vtau_out : jnp.ndarray
        Shape (rho, alpha, Bcrit, well).
        Bounce integral of v·τ.
    iotas : jnp.ndarray, shape (rho,)
        Rotational transform per surface.
    KE_frac : float
        Fraction of the 3.5 MeV D-T fusion alpha-particle birth energy to use
        for the energetic particle kinetic energy.
    nfp : int
        Number of field periods.
    M, N : int
        Generalized omnigenous helicity.
    fill_value : float
        Value bounce integration outputs take when no well is found.

    Returns
    -------
    dict
        ``Omega`` plus the intermediates it is built from. ``valid`` is
        ``True`` where a trapped particle exists at every alpha and ``Omega``
        is defined.

    """
    KE = KE_frac * _E_BIRTH
    v2 = 2 * KE / _M_ALPHA

    # Bounce-averaged drifts → physical frequencies
    alpha_drift = alpha_drift_out * KE / (_Z_ALPHA * _E_CHARGE)
    eta_drift = safediv(
        nfp * alpha_drift, N * nfp - iotas[:, None, None, None] * M, fill=fill_value
    )

    s_drift = s_drift_out * KE / (_Z_ALPHA * _E_CHARGE)
    tau_bounce = vtau_out / jnp.sqrt(v2)
    omega_bounce = safediv(2 * jnp.pi, tau_bounce, fill=fill_value)

    # Require particle to be trapped at all alpha/eta values for a given
    # (rho, pitch, well).
    all_alpha_valid = (_is_valid_value(omega_bounce, fill_value)).all(
        axis=1
    )  # (rho, pitch, well)

    # Alpha-averaged frequencies → normalized precession Omega
    omega_bounce_avg = _jnpmean_nz(omega_bounce, axis=1, fill=fill_value)
    eta_drift_avg = _jnpmean_nz(eta_drift, axis=1, fill=fill_value)
    Omega = safediv(eta_drift_avg, omega_bounce_avg, fill=fill_value)
    valid = (
        _is_valid_value(eta_drift_avg, fill_value)
        & _is_valid_value(omega_bounce_avg, fill_value)
        & all_alpha_valid
    )
    return {
        "Omega": jnp.where(valid, Omega, fill_value),
        "valid": valid,
        "omega_bounce_avg": omega_bounce_avg,
        "eta_drift_avg": eta_drift_avg,
        "omega_bounce": omega_bounce,
        "eta_drift": eta_drift,
        "tau_bounce": tau_bounce,
        "s_drift": s_drift,
    }


_bounce1D_doc = {
    "num_quad": _bounce_doc["num_quad"],
    "num_pitch": _bounce_doc["num_pitch"],
    "surf_batch_size": _bounce_doc["surf_batch_size"],
    "quad": _bounce_doc["quad"],
}

_resonance_doc = {
    "M": """int :
        Generalized omnigenous helicity. Each B contour closes on itself after
        traversing the torus M times toroidally and N times poloidally.
        """,
    "N": """int :
        Generalized omnigenous helicity. Each B contour closes on itself after
        traversing the torus M times toroidally and N times poloidally.
        """,
    "nfp": """int :
        Number of field periods.
        """,
    "KE_frac": """jnp.ndarray :
        Fraction of the 3.5 MeV D-T fusion alpha-particle birth energy to use
        for the energetic particle kinetic energy.
        """,
    "pitch_invs": """jnp.ndarray or None :
        If not ``None``, sets pitch_invs (Bcrits) to the specified value, and
        causes this function to skip the phase-space average and return the
        raw per-(rho, pitch, well) resonance-physics dictionary instead of the
        phase-space-averaged objective. If ``None``, uses a linspace of
        num_pitch between Bmin and Bmax of each flux surface.
        """,
    "rho_res": """float :
        Radial grid spacing.
        """,
    "eta_res": """float :
        Grid spacing for eta.
        """,
    "res_arr": """jnp.ndarray :
        Resonance frequency ratios p/q to check Omega_eta against, for all
        combinations of p/q up to p_max/q_max within
        [res_range_min, res_range_max].
        """,
    "p_arr": """jnp.ndarray :
        Numerators of the resonance ratios in ``res_arr``.
        """,
    "q_arr": """jnp.ndarray :
        Denominators (toroidal mode numbers) of the resonance ratios in
        ``res_arr``.
        """,
    "weight_method": """str :
        ``"linear"`` or ``"bump"`` resonance weighting.
        """,
    "Delta_Omega": """float or None :
        Half-width of the resonance interval for ``weight_method="bump"``.
        If ``None``, defaults to wd_blur × the max |Ω[i+1]-Ω[i]| spacing.
        Ignored when ``weight_method="linear"``.
        """,
    "wd_blur": """float :
        Multiplicative blur factor used to compute bump half-width from
        adjacent-surface Omega spacing when ``Delta_Omega`` is not provided.
        """,
    "fill_value": """float :
        Value to set bounce integration outputs to if no well is found.
        Cannot use ``jnp.nan`` to retain optimization abilities. Cannot use 0
        for confusion with other quantities and averages.
        """,
    "stab_sacrifice": """bool :
        If ``True``, multiply the island-width term by ``Omega_prime_s**2``
        in the objective. If ``False``, omit that factor to preserve
        numerical stability.
        """,
    "bt_filter_flag": """bool :
        If ``True``, zero out wells whose poloidal bounce width exceeds 2π
        (barely-trapped filter) before the resonance physics calculation.
        """,
    "cropping_DOmega": """bool :
        If ``True``, Delta_Omega calculation is clipped by
        ``0.01 * max(Omega_eta) < Delta_Omega < 0.10 * max(Omega_eta)``.
        Only used with the ``bump`` weighting method and
        ``Delta_Omega = None``. Otherwise this quantity is ignored.
        """,
    "num_transit": """int :
        Number of toroidal transits spanned by ``zeta``. Used to normalize
        the field-line length by the length of a single transit.
        """,
    "num_eta": """int :
        Number of uniformly spaced eta points in [0, 2π). Alpha values are
        derived per rho surface via ``alpha = eta * (N*nfp - iota*M) / nfp``.
        """,
    "zeta": """jnp.ndarray :
        Toroidal angle values spanning ``num_transit`` toroidal transits,
        used for field-line integration and to compute the field-line length.
        """,
    "_eta_grid": """Grid :
        Field-line-following grid with per-rho alpha values derived from
        uniform eta, built by ``TrappedResonance.compute``. This private
        parameter is intended to be used only by developers for objectives.
        """,
    "_psa_grid": """Grid :
        Field-line-following grid uniform in alpha, used for the phase-space
        average, built by ``TrappedResonance.compute``. This private
        parameter is intended to be used only by developers for objectives.
        """,
    "_data_eta": """dict[str, jnp.ndarray] :
        Field data evaluated on ``_eta_grid``, built by
        ``TrappedResonance.compute``. This private parameter is intended to
        be used only by developers for objectives.
        """,
    "_data_fft_r": """dict[str, jnp.ndarray] :
        Radial derivatives, at fixed theta and zeta, of the field data in
        ``_data_fft``. The Bounce2D counterpart of ``_data_eta_r``. Required
        on the Bounce2D path.
        """,
    "_angle_r": """jnp.ndarray :
        Radial derivative of ``_angle``, required alongside ``_data_fft_r``
        because the poloidal angle map moves with rho through both iota and
        lambda.
        """,
    "_data_eta_r": """dict[str, jnp.ndarray] :
        Radial derivatives, at fixed eta and zeta, of the field data in
        ``_data_eta``, built by ``TrappedResonance.compute``. Only the keys in
        ``_ETA_BOUNCE_KEYS`` are read. Required on the Bounce1D path: Ω'(s) is
        obtained by differentiating the bounce integrals through them. This
        private parameter is intended to be used only by developers for
        objectives.
        """,
    "_data_psa": """dict[str, jnp.ndarray] :
        Field data evaluated on ``_psa_grid``, built by
        ``TrappedResonance.compute``. This private parameter is intended to
        be used only by developers for objectives.
        """,
}


def _field_line_length(
    use_bounce1d, psa_grid, data_psa, zeta, num_transit, base_grid, data
):
    """Alpha-averaged field line length of a single toroidal transit, ∫dl/B.

    ∫dl/B = ∫dζ/B^ζ, integrated along the field lines of the phase-space
    average grid for Bounce1D. Bounce2D does not materialize those field
    lines, so the identity V_psi = ∬ |𝐁⋅∇ζ|⁻¹ dα dζ over (α, ζ) ∈ [0, 2π)²
    is used instead, making the alpha-averaged single transit length
    V_psi / 2π.

    Returns
    -------
    fl_length : jnp.ndarray, shape (rho, )

    """
    if not use_bounce1d:
        return base_grid.compress(data["V_psi"]) / (2 * jnp.pi)
    Bzeta_psa = Bounce1D.reshape(psa_grid, data_psa["B^zeta"])
    n_1t = len(zeta) // num_transit
    return jnp.abs(
        simpson(1 / Bzeta_psa[..., :n_1t], x=zeta[:n_1t], axis=-1).mean(axis=1)
    )


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
        If ``None``, number of field lines considered is consistent with bounce
        integration in ``_trapped_EP_resonance``. If not ``None``, specifies number
        of total field lines to consider.
        Defaults to ``None``.
    fill_value : float, optional
        Value to set bounce integration outputs to if no well is found. Cannot use
        ``jnp.nan`` to retain optimization abilities. Cannot use 0 for confusion
        with other quantities and averages.
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
        / safe_pitch_inv**2,
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
    alpha_drift_out,
    s_drift_out,
    vtau_out,
    iotas,
    rhos,
    rho_res,
    KE_frac,
    nfp,
    M,
    N,
    res_arr,
    q_arr,
    eta_vals,
    eta_res,
    weight_method,
    Delta_Omega,
    wd_blur,
    fill_value,
    stab_sacrifice,
    dOmega_drho,
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
        Fraction of the 3.5 MeV D-T fusion alpha-particle birth energy to use
        for the energetic particle kinetic energy.
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
    dOmega_drho : jnp.ndarray
        Shape (rho, pitch, well).
        ∂Ω/∂ρ at fixed λ and η, obtained by differentiating the bounce
        integrals.
    cropping_DOmega : bool
        If ``True``, Delta_Omega calculation is clipped by
        ``0.01 * max(Omega_eta) < Delta_Omega < 0.10 * max(Omega_eta)``.
        This must be when using the ``bump`` weighting method and
        ``Delta_Omega = None`` case. Otherwise this quantity is ignored.
        Defaults to ``False``.

    Returns
    -------
    result : dict
        Dictionary containing:

        f_res : jnp.ndarray, shape (rho, pitch, well)
            Per-(rho, pitch, well) resonance objective contribution: island
            width squared, summed over resonances in ``res_arr`` and weighted
            by ``res_weight``, optionally scaled by ``Omega_prime_s`` if
            ``stab_sacrifice``. Phase-space averaged elsewhere to form the
            "trapped EP resonance" objective. The least-squares objective
            built on top of this squares its residual again, so the net
            penalty is (island width)^4, optionally scaled by
            ``Omega_prime_s**2``, matching the pre-existing scaling without
            squaring it twice.
        Omega : jnp.ndarray, shape (rho, pitch, well)
            Normalized precession frequency, i.e. Omega_eta,
            ``eta_drift_avg / omega_bounce_avg``, compared against the
            rational ratios in ``res_arr`` to locate resonances.
        omega_bounce_avg : jnp.ndarray, shape (rho, pitch, well)
            Alpha-averaged bounce frequency ``2π / tau_bounce``.
        eta_drift_avg : jnp.ndarray, shape (rho, pitch, well)
            Alpha-averaged eta precession frequency ω_η (the numerator of
            ``Omega``/Omega_eta).
        omega_bounce : jnp.ndarray, shape (rho, alpha, pitch, well)
            Bounce frequency per field line, before alpha-averaging.
        eta_drift : jnp.ndarray, shape (rho, alpha, pitch, well)
            Eta precession frequency ω_η per field line, before
            alpha-averaging.
        Omega_prime_s : jnp.ndarray, shape (rho, pitch, well)
            Radial derivative dOmega/ds, where s = rho², from ``dOmega_drho``.
        res_weight : jnp.ndarray, shape (rho, pitch, well, res)
            Weight assigning each (rho, pitch, well) to each resonance in
            ``res_arr``, via 2-point linear interpolation or a smooth bump
            function, depending on ``weight_method``.
        f_q_abs : jnp.ndarray, shape (rho, pitch, well, res)
            Magnitude of the q-th eta-Fourier harmonic of the bounce-averaged
            radial (s) drift.
        Delta_s : jnp.ndarray, shape (pitch, well, res)
            Resonance-weighted island width (s = rho² units), summed over
            rho; a diagnostic quantity.
        Delta_s_prof : jnp.ndarray, shape (rho, pitch, well, res)
            Per-surface island width (s = rho² units) at each resonance.
        s_res : jnp.ndarray, shape (pitch, well, res)
            Resonance-weighted mean s = rho² location of each resonance.
        valid : jnp.ndarray, shape (rho, pitch, well)
            Boolean mask, ``True`` where a trapped particle exists at all
            alpha/eta and both ``Omega`` and ``Omega_prime_s`` are defined.
    """
    freq = _frequencies(
        alpha_drift_out, s_drift_out, vtau_out, iotas, KE_frac, nfp, M, N, fill_value
    )
    Omega = freq["Omega"]
    valid = freq["valid"]
    omega_bounce_avg = freq["omega_bounce_avg"]
    eta_drift_avg = freq["eta_drift_avg"]
    omega_bounce = freq["omega_bounce"]
    eta_drift = freq["eta_drift"]
    tau_bounce = freq["tau_bounce"]
    s_drift = freq["s_drift"]

    dOmega_drho = jnp.where(valid, dOmega_drho, fill_value)
    Omega_prime_s = jnp.where(
        valid, dOmega_drho / (2 * rhos[:, None, None]), fill_value
    )

    # Resonance weights
    Omega_broad = Omega[..., None]
    res_broad = res_arr[None, None, None, :]

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

            Delta_Omega_val = (wd_blur * _softmax(domega_arr, alpha=50, axis=0) / 2.0)[
                None, :, :, None
            ]
            if cropping_DOmega:
                # Delta_Omega_val needs to be cropped if resolution is too low
                # or Omega_eta shear is too high
                Omega_max = _softmax(Omega_safe_bump, alpha=50, axis=0)[
                    None, :, :, None
                ]
                Delta_Omega_val_max = (
                    0.1 * Omega_max
                )  # DeltaOmega < 10% of maximum Omega
                Delta_Omega_val_min = (
                    0.01 * Omega_max
                )  # DeltaOmega > 1% of maximum Omega
                Delta_Omega_val = jnp.where(
                    Delta_Omega_val > Delta_Omega_val_max,
                    Delta_Omega_val_max,
                    Delta_Omega_val,
                )
                Delta_Omega_val = jnp.where(
                    Delta_Omega_val < Delta_Omega_val_min,
                    Delta_Omega_val_min,
                    Delta_Omega_val,
                )
        else:
            Delta_Omega_val = Delta_Omega
        a = res_broad + Delta_Omega_val
        b = res_broad - Delta_Omega_val
        in_interval = (Omega_broad >= b) & (Omega_broad <= a)
        denom = (Omega_broad - b) * (Omega_broad - a)
        exp_arg = safediv((2.0 * Delta_Omega_val) ** 2, denom, fill=-1e10)
        C_norm = safediv(71.12518788738504, Delta_Omega_val, fill=0.0)
        w_raw = rho_res * C_norm * jnp.abs(dOmega_drho[..., None]) * jnp.exp(exp_arg)
        # Weight is non-zero only if in interval and valid
        res_weight = jnp.where(in_interval & valid[..., None], w_raw, 0)
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
            [valid[1:], jnp.zeros_like(valid[:1])], axis=0
        )[..., None]
        valid_prev_lin = jnp.concatenate(
            [jnp.zeros_like(valid[:1]), valid[:-1]], axis=0
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
    res_weight = jnp.where(valid[..., None], res_weight, 0)

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

    f_q_r2 = f_q_c**2 + f_q_s**2
    is_zero = f_q_r2 == 0
    f_q_abs = 0.5 * jnp.sqrt(jnp.where(is_zero, 1.0, f_q_r2))
    f_q_abs = jnp.where(is_zero, 0.0, f_q_abs)

    # Filter FT results to valid points.
    f_q_abs = jnp.where(valid[..., None], f_q_abs, 0.0)

    # Island widths
    q_iw = q_arr[None, None, None, :]
    denom = jnp.pi * q_iw * jnp.abs(Omega_prime_s[..., None])
    Delta_s_profile = 4 * jnp.sqrt(safediv(f_q_abs, denom, fill=0.0))
    Delta_s_sq_profile = 16 * safediv(f_q_abs, denom, fill=0.0)
    Delta_s_sq_sum = (Delta_s_sq_profile * res_weight).sum(axis=-1)

    if stab_sacrifice:
        f_res = Delta_s_sq_sum * Omega_prime_s
    else:
        f_res = Delta_s_sq_sum

    # Sum over radius to get weighted island width and resonance location.
    Delta_s = (Delta_s_profile * res_weight).sum(axis=0)
    s_vals = rhos**2
    s_res = (res_weight * s_vals[:, None, None, None]).sum(axis=0)

    return {
        "f_res": f_res,  # (rho, pitch, well)
        "Omega": Omega,  # (rho, pitch, well)
        "omega_bounce_avg": omega_bounce_avg,  # (rho, pitch, well)
        "eta_drift_avg": eta_drift_avg,  # (rho, pitch, well)
        "omega_bounce": omega_bounce,  # (rho, alpha, pitch, well)
        "eta_drift": eta_drift,  # (rho, alpha, pitch, well)
        "Omega_prime_s": Omega_prime_s,  # (rho, pitch, well)
        "res_weight": res_weight,  # (rho, pitch, well, res)
        "f_q_abs": f_q_abs,  # (rho, pitch, well, res)
        "Delta_s": Delta_s,  # (pitch, well, res), rho-weighted diagnostic
        "Delta_s_prof": Delta_s_profile,  # (rho, pitch, well, res)
        "s_res": s_res,  # (pitch, well, res), rho-weighted resonance location
        "valid": valid,  # (rho, pitch, well)
    }


@register_compute_fun(
    name="trapped EP resonance",
    label=("Trapped Energetic Particle Resonance Objective Function"),
    units="s^-2",
    units_long="seconds squared",
    description="Trapped Energetic Particle Resonance Minimizer",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["iota", "iota_r", "min_tz |B|", "max_tz |B|", "Psi", "V_psi"],
    grid_requirement={"is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
    **_resonance_doc,
)
def _trapped_EP_resonance(params, transforms, profiles, data, **kwargs):
    """Trapped particle resonance penalty.

    Three stages:
      1. Bounce integrals  (per-surface, via ``_compute1D`` / ``batch_map``)
      2. Resonance physics  (cross-surface, via ``_resonance_physics``)
      3. Phase-space average (via ``_phase_space_average``)

    The eta/PSA grids and the field data evaluated on them (``_eta_grid``,
    ``_psa_grid``, ``_data_eta``, ``_data_psa``) are built by the caller (see
    ``TrappedResonance.compute``) rather than here, since building them
    requires the full ``Equilibrium`` object, which compute functions must
    stay pure with respect to (only ``params``/``transforms``/``profiles``/
    ``data``) to remain properly differentiable and dispatchable for any
    parameterization.
    """
    num_pitch = kwargs.get("num_pitch")
    num_well = 1
    M = kwargs.get("M", 1)
    N = kwargs.get("N", 1)
    nfp = kwargs.get("nfp")
    KE_frac = kwargs.get("KE_frac")
    pitch_invs = kwargs.get("pitch_invs")
    rho_res = kwargs.get("rho_res")
    eta_res = kwargs.get("eta_res")
    res_arr = kwargs.get("res_arr")
    p_arr = kwargs.get("p_arr")
    q_arr = kwargs.get("q_arr")
    quad = kwargs.get("quad")
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    num_eta = kwargs.get("num_eta")
    weight_method = kwargs.get("weight_method", "linear")
    Delta_Omega = kwargs.get("Delta_Omega")
    wd_blur = kwargs.get("wd_blur", 1.25)
    fill_value = kwargs.get("fill_value", 11)
    zeta = kwargs.get("zeta")
    stab_sacrifice = kwargs.get("stab_sacrifice", False)
    bt_filter_flag = kwargs.get("bt_filter_flag", False)
    cropping_DOmega = kwargs.get("cropping_DOmega", False)
    eta_grid = kwargs.get("_eta_grid")
    psa_grid = kwargs.get("_psa_grid")
    data_eta = kwargs.get("_data_eta")
    data_psa = kwargs.get("_data_psa")
    num_transit = kwargs.get("num_transit", 1)
    use_bounce1d = kwargs.get("use_bounce1d", False)
    # Bounce2D path only.
    angle = kwargs.get("_angle")
    fft_grid = kwargs.get("_fft_grid")
    data_fft = kwargs.get("_data_fft")
    Y_B = kwargs.get("Y_B")
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander")

    nufft_eps = kwargs.get("nufft_eps", 1e-10)

    base_grid = transforms["grid"]
    iotas = base_grid.compress(data["iota"])
    iotas_r = base_grid.compress(data["iota_r"])
    rhos = base_grid.compress(base_grid.nodes[:, 0])
    eta_vals = jnp.linspace(0, 2 * jnp.pi, num_eta, endpoint=False)

    # --- 1. Bounce integrals on the eta grid ---
    pitch_invs_use, pitch_inv_weight_use = _global_pitch_quad(
        base_grid, data, num_pitch, pitch_invs
    )

    drift_names = list(_BOUNCE_INTEGRAND_KEYS)

    if use_bounce1d:

        def drifts(data_in):
            bounce = Bounce1D(eta_grid, data_in, quad, is_reshaped=True)
            points = bounce.points(data_in["pitch_inv"], num_well=num_well)
            v_tau, _alpha_drift, _s_drift = bounce.integrate(
                [_v_tau, _alpha_drift_integrand, _radial_drift],
                data_in["pitch_inv"],
                data_in,
                drift_names,
                num_well=num_well,
            )
            _alpha_drift = safediv(_alpha_drift, v_tau)
            _s_drift = 4 * safediv(_s_drift, v_tau)
            return _alpha_drift, _s_drift, points, v_tau, data_in["pitch_inv"]

        def bounce_and_omega(data_in, iotas_in):
            """Bounce integrals and Omega, as a function of the field line data.

            Kept as its own function so that pushing the radial derivatives of
            the field line data through it in forward mode gives the exact
            ∂Ω/∂ρ. The pitch grid is closed over rather than passed in, so that
            derivative is taken at fixed λ, as the resonance condition requires.
            """
            _alpha_drift, _s_drift, _points, _v_tau, _pitch_inv = _compute1D(
                drifts,
                {name: data_in[name] for name in drift_names},
                data_in,
                eta_grid,
                num_pitch,
                surf_batch_size,
                pitch_invs=pitch_invs_use,
                pitch_inv_weight=pitch_inv_weight_use,
            )
            _Omega = _frequencies(
                _alpha_drift,
                _s_drift,
                _v_tau,
                iotas_in,
                KE_frac,
                nfp,
                M,
                N,
                fill_value,
            )["Omega"]
            return (
                _alpha_drift,
                _s_drift,
                _points[0],
                _points[1],
                _v_tau,
                _pitch_inv,
                _Omega,
            )

        data_eta_r = kwargs["_data_eta_r"]
        eta_bounce_data = {name: data_eta[name] for name in _ETA_BOUNCE_KEYS}
        # ∂Ω/∂ρ at fixed λ and η, from the radial derivatives of the field line
        # data.
        bounce_out, bounce_dot = jax.jvp(
            bounce_and_omega,
            (eta_bounce_data, iotas),
            ({name: data_eta_r[name] for name in _ETA_BOUNCE_KEYS}, iotas_r),
        )
        dOmega_drho = bounce_dot[-1]
        alpha_drift_out, s_drift_out, z1, z2, vtau_out, pitch_inv_out = bounce_out[:-1]
        points = (z1, z2)
    else:

        def drifts(data_in):
            bounce = Bounce2D(
                fft_grid,
                data_in,
                data_in["angle"],
                Y_B,
                data_in["alpha_per_rho"].T,
                num_transit,
                quad,
                nufft_eps=nufft_eps,
                is_fourier=True,
                spline=spline,
                vander=vander,
            )
            points = bounce.points(data_in["pitch_inv"], num_well=num_well)
            v_tau, _alpha_drift, _s_drift = bounce.integrate(
                [_v_tau, _alpha_drift_integrand, _radial_drift],
                data_in["pitch_inv"],
                data_in,
                drift_names,
                num_well=num_well,
                nufft_eps=nufft_eps,
                is_fourier=True,
                low_ram=True,
            )
            _alpha_drift = safediv(_alpha_drift, v_tau)
            _s_drift = 4 * safediv(_s_drift, v_tau)
            return _alpha_drift, _s_drift, points, v_tau, data_in["pitch_inv"]

        def bounce_and_omega(data_in, angle_in, iotas_in):
            """Bounce integrals and Omega, as a function of the field data.

            The Bounce2D counterpart of the Bounce1D closure above, and
            differentiated the same way. ``fft_grid`` enters only through
            reshapes and compressions, so it stays fixed while the data it
            indexes carries the radial tangents.
            """
            # eta -> alpha, which differs between flux surfaces because it
            # depends on iota, and so moves with rho. Carried with rho leading
            # so ``batch_map`` slices it with the rest of the data;
            # ``Bounce2D`` wants (num alpha, num rho).
            _alpha_eta = eta_vals[None, :] * (N * nfp - iotas_in[:, None] * M) / nfp
            _alpha_drift, _s_drift, _points, _v_tau, _pitch_inv = _compute2D(
                drifts,
                {name: data_in[name] for name in drift_names},
                data_in,
                fft_grid,
                angle_in,
                _alpha_eta,
                num_pitch,
                surf_batch_size,
                pitch_invs=pitch_invs_use,
                pitch_inv_weight=pitch_inv_weight_use,
            )
            _Omega = _frequencies(
                _alpha_drift,
                _s_drift,
                _v_tau,
                iotas_in,
                KE_frac,
                nfp,
                M,
                N,
                fill_value,
            )["Omega"]
            return (
                _alpha_drift,
                _s_drift,
                _points[0],
                _points[1],
                _v_tau,
                _pitch_inv,
                _Omega,
            )

        data_fft_r, angle_r = kwargs["_data_fft_r"], kwargs["_angle_r"]
        fft_bounce_data = {name: data_fft[name] for name in _FFT_BOUNCE_KEYS}
        bounce_out, bounce_dot = jax.jvp(
            bounce_and_omega,
            (fft_bounce_data, angle, iotas),
            (
                {name: data_fft_r[name] for name in _FFT_BOUNCE_KEYS},
                angle_r,
                iotas_r,
            ),
        )
        dOmega_drho = bounce_dot[-1]
        alpha_drift_out, s_drift_out, z1, z2, vtau_out, pitch_inv_out = bounce_out[:-1]
        points = (z1, z2)

    # --- 1b. Barely-trapped filter ---
    if bt_filter_flag:
        s_drift_out = _barely_trapped_filter(
            s_drift_out, points, iotas, M, N, nfp, fill_value
        )

    # --- 2. Resonance physics (cross-surface) ---
    res = _resonance_physics(
        alpha_drift_out,
        s_drift_out,
        vtau_out,
        iotas,
        rhos,
        rho_res,
        KE_frac,
        nfp,
        M,
        N,
        res_arr,
        q_arr,
        eta_vals,
        eta_res,
        weight_method,
        Delta_Omega,
        wd_blur,
        fill_value,
        stab_sacrifice,
        dOmega_drho,
        cropping_DOmega,
    )

    # --- 3. Phase-space average on the PSA grid (uniform in alpha) ---
    # Skip PSA when custom pitch_invs are provided.
    if pitch_invs is None:
        if use_bounce1d:
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
        else:
            # The phase-space average is taken over field lines uniform in
            # alpha, unlike the eta grid above, so the labels are the same on
            # every surface. Reuses the same (θ, ζ) data and angle.
            num_alpha_psa = num_eta
            alpha_psa = jnp.broadcast_to(
                jnp.linspace(0, 2 * jnp.pi, num_alpha_psa, endpoint=False),
                (rhos.size, num_alpha_psa),
            )

            def drifts_vtau(data_local):
                bounce = Bounce2D(
                    fft_grid,
                    data_local,
                    data_local["angle"],
                    Y_B,
                    data_local["alpha_per_rho"].T,
                    num_transit,
                    quad,
                    nufft_eps=nufft_eps,
                    is_fourier=True,
                    spline=spline,
                    vander=vander,
                )
                v_tau = bounce.integrate(
                    [_v_tau],
                    data_local["pitch_inv"],
                    data_local,
                    [],
                    num_well=num_well,
                    nufft_eps=nufft_eps,
                    is_fourier=True,
                )[0]
                return v_tau, data_local

            vtau_psa, _data_psa = _compute2D(
                drifts_vtau,
                {},
                data_fft,
                fft_grid,
                angle,
                alpha_psa,
                num_pitch,
                surf_batch_size,
                pitch_invs=pitch_invs_use,
                pitch_inv_weight=pitch_inv_weight_use,
            )
            num_rho_psa = rhos.size
            if vtau_psa.ndim == 3 and vtau_psa.shape[0] == num_rho_psa * num_alpha_psa:
                vtau_psa = vtau_psa.reshape(
                    num_rho_psa, num_alpha_psa, vtau_psa.shape[1], vtau_psa.shape[2]
                )

        fl_length = _field_line_length(
            use_bounce1d, psa_grid, data_psa, zeta, num_transit, base_grid, data
        )

        f_res_avg = _phase_space_average(
            vtau_psa,
            res["f_res"],
            _data_psa["pitch_inv"],
            _data_psa["pitch_inv weight"],
            fl_length,
            num_alpha=num_alpha_psa,
            fill_value=fill_value,
        )
        data["trapped EP resonance"] = base_grid.expand(f_res_avg)
    else:  # Custom pitch_invs specified: skip phase-space average,
        # just return the raw resonance physics results
        data["trapped EP resonance"] = {
            **res,
            "pitch_inv": pitch_inv_out,
            "res_arr": res_arr,
            "p_arr": p_arr,
            "q_arr": q_arr,
            "rhos": rhos,
        }

    return data
