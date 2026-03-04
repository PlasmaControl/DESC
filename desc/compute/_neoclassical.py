"""Compute functions for neoclassical transport."""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import imap, jax, jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce1D, Bounce2D
from ..utils import safediv
from .data_index import register_compute_fun

from ..integrals.quad_utils import chebgauss2
from ..integrals._bounce_utils import get_pitch_inv_quad
from quadax import simpson

_bounce_doc = {
    "theta": """jnp.ndarray :
        Shape (num rho, X, Y).
        DESC coordinates θ from ``Bounce2D.compute_theta``.
        ``X`` and ``Y`` are preferably rounded down to powers of two.
        """,
    "Y_B": """int :
        Desired resolution for algorithm to compute bounce points.
        A reference value is 100. Default is double ``Y``.
        """,
    "alpha": """jnp.ndarray :
        Shape (num alpha, ).
        Starting field line poloidal labels.
        Default is single field line. To compute a surface average
        on a rational surface, it is necessary to average over multiple
        field lines until the surface is covered sufficiently.
        """,
    "num_transit": """int :
        Number of toroidal transits to follow field line.
        In an axisymmetric device, field line integration over a single poloidal
        transit is sufficient to capture a surface average. For a 3D
        configuration, more transits will approximate surface averages on an
        irrational magnetic surface better, with diminishing returns.
        """,
    "num_well": """int :
        Maximum number of wells to detect for each pitch and field line.
        Giving ``None`` will detect all wells but due to current limitations in
        JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``, ``B`` are the poloidal and
        toroidal Fourier resolution of B, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
        The ``check_points`` or ``plot`` methods in ``desc.integrals.Bounce2D``
        are useful to select a reasonable value.
        """,
    "num_quad": """int :
        Resolution for quadrature of bounce integrals.
        Default is 32. This parameter is ignored if given ``quad``.
        """,
    "num_pitch": """int :
        Resolution for quadrature over velocity coordinate.
        """,
    "pitch_batch_size": """int :
        Number of pitch values with which to compute simultaneously.
        If given ``None``, then ``pitch_batch_size`` is ``num_pitch``.
        Default is ``num_pitch``.
        """,
    "surf_batch_size": """int :
        Number of flux surfaces with which to compute simultaneously.
        If given ``None``, then ``surf_batch_size`` is ``grid.num_rho``.
        Default is ``1``. Only consider increasing if ``pitch_batch_size`` is ``None``.
        """,
    "fieldline_quad": """tuple[jnp.ndarray] :
        Used to compute the proper length of the field line ∫ dℓ / B.
        Quadrature points xₖ and weights wₖ for the
        approximate evaluation of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).
        Default is Gauss-Legendre quadrature at resolution ``Y_B//2``
        on each toroidal transit.
        """,
    "quad": """tuple[jnp.ndarray] :
        Used to compute bounce integrals.
        Quadrature points xₖ and weights wₖ for the
        approximate evaluation of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).
        """,
    "nufft_eps": """float :
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
        """,
    "spline": """bool :
        Whether to use cubic splines to compute bounce points.
        """,
    "_vander": """dict[str,jnp.ndarray] :
        Precomputed transform matrices "dct spline", "dct cfl", "dft cfl".
        This parameter is intended to be used by objectives only.
        """,
}

def _v_tau(data, B, pitch):
    # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
    # τ is the bounce time, and I is defined in Nemov et al. eq. 36.
    return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))

def _alpha_mean(f):
    """Simple mean over field lines.

    Simple mean rather than integrating over α and dividing by 2π
    (i.e. f.T.dot(dα) / dα.sum()), because when the toroidal angle extends
    beyond one transit we need to weight all field lines uniformly, regardless
    of their spacing wrt α.
    """
    return f.mean(axis=0)

def _jnpmean_nz(x, axis=0, fill=jnp.nan):
    """Mean over an axis, ignoring zero entries."""
    mask = x != 0.0
    count = jnp.sum(mask, axis=axis)
    return safediv(jnp.sum(x, axis=axis), count, fill=fill)

def _get_pitch_inv_quad(grid, data, num_pitch, _data):
    p, w = get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]), grid.compress(data["max_tz |B|"]), num_pitch
    )
    _data["pitch_inv"] = jnp.broadcast_to(
        p[jnp.newaxis], (grid.num_alpha, grid.num_rho, num_pitch)
    )
    _data["pitch_inv weight"] = jnp.broadcast_to(
        w[jnp.newaxis], (grid.num_alpha, grid.num_rho, num_pitch)
    )
    return _data

def map2(fun, xs, *, batch_size=None):
    """Map over leading two axes iteratively."""
    # Can't pass in batch_size to imap yet because only new version jax allow that.
    return imap(lambda x: imap(fun, x), xs)

def _compute2D(
    fun,
    fun_data,
    data,
    theta,
    grid,
    num_pitch,
    surf_batch_size=1,
    simp=False,
    expand_out=True,
):
    """Compute Bounce2D integral quantity with ``fun``.

    Parameters
    ----------
    fun : callable
        Function to compute.
    fun_data : dict[str, jnp.ndarray]
        Data to provide to ``fun``. This dict will be modified.
    data : dict[str, jnp.ndarray]
        DESC data dict.
    theta : jnp.ndarray
        Shape (num rho, X, Y).
        DESC coordinates θ from ``Bounce2D.compute_theta``.
        ``X`` and ``Y`` are preferably rounded down to powers of two.
    grid : Grid
        Grid that can expand and compress.
    num_pitch : int
        Resolution for quadrature over velocity coordinate.
    surf_batch_size : int
        Number of flux surfaces with which to compute simultaneously.
        Default is ``1``.
    simp : bool
        Whether to use an open Simpson rule instead of uniform weights.
    expand_out : bool
        Whether to expand output to full grid so that the first dimension
        has size ``grid.num_nodes`` instead of ``grid.num_rho``.
        Default is True.

    """
    for name in Bounce2D.required_names:
        fun_data[name] = data[name]
    fun_data.pop("iota", None)
    for name in fun_data:
        fun_data[name] = Bounce2D.fourier(Bounce2D.reshape(grid, fun_data[name]))
    fun_data["iota"] = grid.compress(data["iota"])
    fun_data["theta"] = theta
    fun_data["pitch_inv"], fun_data["pitch_inv weight"] = Bounce2D.get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
        simp=simp,
    )
    out = batch_map(fun, fun_data, surf_batch_size)
    if expand_out:
        assert out.ndim == 1, "Are you sure you want to expand to full grid?"
        return grid.expand(out)
    return out


def _compute(fun, fun_data, data, grid, num_pitch, surf_batch_size=1, simp=False, pitch_method=1, pitch_invs=None):
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

    """
    for name in Bounce1D.required_names:
        fun_data[name] = data[name]
    for name in fun_data:
        fun_data[name] = Bounce1D.reshape(grid, fun_data[name])
    if pitch_method==0:
        fun_data["pitch_inv"], fun_data["pitch_inv weight"] = Bounce1D.get_pitch_inv_quad(
            grid.compress(data["min_tz |B|"]),
            grid.compress(data["max_tz |B|"]),
            num_pitch,
            simp=simp
        )
    if pitch_method==1:
        Bmin = grid.compress(data["min_tz |B|"]) # := (rho)
        Bmax = grid.compress(data["max_tz |B|"])# := (rho)
        fun_data["pitch_inv"] = jnp.transpose( jnp.linspace(Bmin,Bmax,num_pitch), (1,0) ) # := (rho,Bcrit)
        fun_data["Bcrit_res"] = fun_data["pitch_inv"][:,1]-fun_data["pitch_inv"][:,0] # := (rho)
        fun_data["pitch_inv weight"] = jnp.broadcast_to(
            fun_data["Bcrit_res"][:, jnp.newaxis], fun_data["pitch_inv"].shape
        )
    elif pitch_method==2:
        fun_data["pitch_inv"] = jnp.broadcast_to(pitch_invs, (grid.num_rho,len(pitch_invs) )) # needs to be shape (rho,Bcrit)
        fun_data["Bcrit_res"] = fun_data["pitch_inv"][:,1]-fun_data["pitch_inv"][:,0] # := (rho)
        fun_data["pitch_inv weight"] = jnp.broadcast_to(
            fun_data["Bcrit_res"][:, jnp.newaxis], fun_data["pitch_inv"].shape
        )
    out = batch_map(fun, fun_data, surf_batch_size)
    # assert out.ndim == 1
    # return grid.expand(out)
    return out


# def _compute2D(
#     fun,
#     fun_data,
#     data,
#     theta,
#     grid,
#     num_pitch,
#     surf_batch_size=1,
#     simp=False,
#     expand_out=True,
# ):
#     """Compute Bounce2D integral quantity with ``fun``.

#     Parameters
#     ----------
#     fun : callable
#         Function to compute.
#     fun_data : dict[str, jnp.ndarray]
#         Data to provide to ``fun``. This dict will be modified.
#     data : dict[str, jnp.ndarray]
#         DESC data dict.
#     theta : jnp.ndarray
#         Shape (num rho, X, Y).
#         DESC coordinates θ from ``Bounce2D.compute_theta``.
#         ``X`` and ``Y`` are preferably rounded down to powers of two.
#     grid : Grid
#         Grid that can expand and compress.
#     num_pitch : int
#         Resolution for quadrature over velocity coordinate.
#     surf_batch_size : int
#         Number of flux surfaces with which to compute simultaneously.
#         Default is ``1``.
#     simp : bool
#         Whether to use an open Simpson rule instead of uniform weights.
#     expand_out : bool
#         Whether to expand output to full grid so that the first dimension
#         has size ``grid.num_nodes`` instead of ``grid.num_rho``.
#         Default is True.

#     """
#     for name in Bounce2D.required_names:
#         fun_data[name] = data[name]
#     fun_data.pop("iota", None)
#     for name in fun_data:
#         fun_data[name] = Bounce2D.fourier(Bounce2D.reshape(grid, fun_data[name]))
#     fun_data["iota"] = grid.compress(data["iota"])
#     fun_data["theta"] = theta
#     fun_data["pitch_inv"], fun_data["pitch_inv weight"] = Bounce2D.get_pitch_inv_quad(
#         grid.compress(data["min_tz |B|"]),
#         grid.compress(data["max_tz |B|"]),
#         num_pitch,
#         simp=simp,
#     )
#     out = batch_map(fun, fun_data, surf_batch_size)
#     if expand_out:
#         assert out.ndim == 1, "Are you sure you want to expand to full grid?"
#         return grid.expand(out)
#     return out



def _dH_ripple(data, B, pitch):
    """Integrand of Nemov eq. 30 with |∂ψ/∂ρ| (λB₀)¹ᐧ⁵ removed."""
    return (
        jnp.sqrt(jnp.abs(1 - pitch * B))
        * (4 / (pitch * B) - 1)
        * data["|grad(rho)|*kappa_g"]
        / B
    )


def _dI_ripple(data, B, pitch):
    """Integrand of Nemov eq. 31."""
    return jnp.sqrt(jnp.abs(1 - pitch * B)) / B


@register_compute_fun(
    name="effective ripple 3/2",
    label=(
        # ε¹ᐧ⁵ = π/(8√2) R₀²〈|∇ψ|〉⁻² B₀⁻¹ ∫ dλ λ⁻² 〈 ∑ⱼ Hⱼ²/Iⱼ 〉
        "\\epsilon_{\\mathrm{eff}}^{3/2} = \\frac{\\pi}{8 \\sqrt{2}} "
        "R_0^2 \\langle \\vert\\nabla \\psi\\vert \\rangle^{-2} "
        "B_0^{-1} \\int d\\lambda \\lambda^{-2} "
        "\\langle \\sum_j H_j^2 / I_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude to 3/2 power",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "kappa_g", "R0", "|grad(rho)|", "<|grad(rho)|>"]
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
def _epsilon_32(params, transforms, profiles, data, **kwargs):
    """Effective ripple modulation amplitude to 3/2 power.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    https://doi.org/10.1063/1.873749.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 51)
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
        kwargs["quad"] if "quad" in kwargs else chebgauss2(kwargs.get("num_quad", 32))
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-6)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def eps_32(data):
        """(∂ψ/∂ρ)⁻² B₀⁻³ ∫ dλ λ⁻² 〈 ∑ⱼ Hⱼ²/Iⱼ 〉."""
        # B₀ has units of λ⁻¹.
        # Nemov's ∑ⱼ Hⱼ²/Iⱼ = (∂ψ/∂ρ)² (λB₀)³ (H² / I).sum(-1).
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        bounce = Bounce2D(
            grid,
            data,
            Bounce2D.reshape(grid, grid.nodes[:, 1]),
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
            H, I = bounce.integrate(
                [_dH_ripple, _dI_ripple],
                pitch_inv,
                data,
                "|grad(rho)|*kappa_g",
                num_well=num_well,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            return safediv(H**2, I).sum(-1).mean(-2)

        return jnp.sum(
            batch_map(fun, data["pitch_inv"], pitch_batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 3,
            axis=-1,
        ) / bounce.compute_fieldline_length(fl_quad, vander)

    grid = transforms["grid"]
    B0 = data["max_tz |B|"]
    data["effective ripple 3/2"] = (
        _compute(
            eps_32,
            {"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]},
            data,
            theta,
            grid,
            num_pitch,
            surf_batch_size,
            simp=True,
        )
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * jnp.pi
        / (8 * 2**0.5)
    )
    return data


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Neoclassical transport in the banana regime",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["effective ripple 3/2"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    """Proxy for neoclassical transport in the banana regime.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. In the banana regime, neoclassical (thermal)
    transport from ripple wells can become the dominant transport channel.
    The effective ripple (ε) proxy estimates the neoclassical transport
    coefficients in the banana regime.
    """
    data["effective ripple"] = data["effective ripple 3/2"] ** (2 / 3)
    return data


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

def _s_drift_integrand(data, B, pitch):
    return safediv(
        2 * data["cvdrift0"] * (2 - pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
    )

def _alpha_drift_integrand(data, B, pitch):
    return safediv(
        2 * (data["gbdrift (periodic)"] * pitch * B + 2 * (1 - pitch * B) * data["cvdrift (periodic)"]), jnp.sqrt(jnp.abs(1 - pitch * B))
    )

_bounce1D_doc = {
    "num_well": _bounce_doc["num_well"],
    "num_quad": _bounce_doc["num_quad"],
    "num_pitch": _bounce_doc["num_pitch"],
    "surf_batch_size": _bounce_doc["surf_batch_size"],
    "quad": _bounce_doc["quad"],
}


def _phase_space_average(vtau_out, f_res, pitch_inv, pitch_inv_weight, fl_length):
    """Phase-space average of f_res.

    Computes <f_res> = Σ_w ∫dα ∫dλ v·τ_b · f / (2 ∫dα ∫dl/B).
    f_res is α-independent, so it is pulled out of the α integral.
    Pitch quadrature uses Gauss-Legendre weights from
    ``Bounce1D.get_pitch_inv_quad``, matching eps_eff / Gamma_c.

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

    Returns
    -------
    f_res_avg : jnp.ndarray, shape (rho,)
    """
    num_alpha = vtau_out.shape[1]
    integrand = vtau_out * f_res[:, jnp.newaxis, :, :]
    # 1. Integrate over pitch (per α, per well): ∫dλ g(λ) = ∫dp g(1/p)/p²
    pitch_integrated = jnp.nansum(
        integrand
        * pitch_inv_weight[:, jnp.newaxis, :, jnp.newaxis]
        / pitch_inv[:, jnp.newaxis, :, jnp.newaxis] ** 2,
        axis=2,
    )  # (rho, alpha, well)
    # 2. Sum over α (discrete ∫dα)
    alpha_summed = pitch_integrated.sum(axis=1)  # (rho, well)
    # 3. Sum over wells
    numerator = jnp.nansum(alpha_summed, axis=-1)  # (rho,)
    # Denominator: 2 · Σ_α ∫dl/B = 2 · N_α · mean_α(∫dl/B)
    return safediv(numerator, 2 * num_alpha * fl_length)


def _resonance_physics(
    alpha_drift_out, s_drift_out, vtau_out,
    iotas, rhos, rho_res, KE_frac, nfp, M, N,
    res_arr, q_arr, eta_vals,
    f_q_conservative, weight_method, Delta_Omega,
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
    f_q_conservative : bool
        Whether to use conservative Fourier coefficient estimate.
    weight_method : str
        ``"linear"`` or ``"bump"`` resonance weighting.
    Delta_Omega : float or None
        Half-width for bump weighting.

    Returns
    -------
    result : dict
        Dictionary with keys: ``f_res``, ``Omega``, ``omega_bounce_avg``,
        ``eta_drift_avg``, ``omega_bounce``, ``eta_drift``,
        ``Omega_prime_s``, ``res_weight``, ``f_q_c``, ``f_q_s``,
        ``f_q_abs``, ``Delta_s``, ``f_q_conservative``, ``s_res``,
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
    eta_drift = nfp * alpha_drift / (N * nfp - iotas_omega * M)

    s_drift = s_drift_out * KE / (Z * e_charge)
    tau_bounce = vtau_out / jnp.sqrt(v2)
    omega_bounce = 2 * jnp.pi / tau_bounce

    # Alpha-averaged frequencies → normalised precession Omega
    omega_bounce_avg = _jnpmean_nz(omega_bounce, axis=1, fill=jnp.nan)
    eta_drift_avg = _jnpmean_nz(eta_drift, axis=1, fill=jnp.nan)
    Omega = safediv(eta_drift_avg, omega_bounce_avg, fill=jnp.nan)
    valid = jnp.isfinite(Omega)

    # Omega'(s) via finite differences
    Omega_prev_g = jnp.concatenate(
        [jnp.full((1,) + Omega.shape[1:], jnp.nan), Omega[:-1]], axis=0
    )
    Omega_next_g = jnp.concatenate(
        [Omega[1:], jnp.full((1,) + Omega.shape[1:], jnp.nan)], axis=0
    )
    valid_prev = jnp.concatenate(
        [jnp.zeros((1,) + valid.shape[1:], dtype=bool), valid[:-1]], axis=0
    )
    valid_next = jnp.concatenate(
        [valid[1:], jnp.zeros((1,) + valid.shape[1:], dtype=bool)], axis=0
    )
    grad_central = (Omega_next_g - Omega_prev_g) / (2 * rho_res)
    grad_forward = (Omega_next_g - Omega) / rho_res
    grad_backward = (Omega - Omega_prev_g) / rho_res
    dOmega_drho = jnp.where(
        valid & valid_prev & valid_next,
        grad_central,
        jnp.where(
            valid & valid_next & ~valid_prev,
            grad_forward,
            jnp.where(
                valid & valid_prev & ~valid_next,
                grad_backward,
                jnp.nan,
            ),
        ),
    )
    Omega_prime_s = safediv(
        dOmega_drho, 2 * rhos[:, None, None], fill=jnp.nan
    )

    # Resonance weights
    Omega_broad = Omega[..., None]
    res_broad = res_arr[None, None, None, :]

    if weight_method == "bump":
        dOmega_spacing = jnp.nanmean(
            jnp.where(valid, jnp.abs(Omega_next_g - Omega), jnp.nan)
        )
        dOmega_spacing = jnp.where(
            jnp.isfinite(dOmega_spacing), dOmega_spacing, 0.1
        )
        Delta_Omega_val = (
            Delta_Omega if Delta_Omega is not None else 2.0 * dOmega_spacing
        )
        a = res_broad + Delta_Omega_val
        b = res_broad - Delta_Omega_val
        in_interval = (Omega_broad >= b) & (Omega_broad <= a)
        denom = (Omega_broad - b) * (Omega_broad - a)
        exp_arg = safediv(
            (2.0 * Delta_Omega_val) ** 2, denom, fill=-1e10
        )
        C_norm = 71.12518788738504 / Delta_Omega_val
        w_raw = C_norm * jnp.abs(Omega_prime_s[..., None]) * jnp.exp(exp_arg)
        w_raw = jnp.where(in_interval, w_raw, jnp.nan)
        w_sum = jnp.nansum(jnp.where(valid[..., None], w_raw, jnp.nan), axis=0)
        res_weight = safediv(w_raw, w_sum[None, :, :, :], fill=jnp.nan)
    else:
        Omega_prev = jnp.concatenate(
            [jnp.full_like(Omega[:1], jnp.nan), Omega[:-1]], axis=0
        )[..., None]
        Omega_next = jnp.concatenate(
            [Omega[1:], jnp.full_like(Omega[:1], jnp.nan)], axis=0
        )[..., None]
        between_next = (
            ((Omega_next >= res_broad) & (res_broad >= Omega_broad))
            | ((Omega_next <= res_broad) & (res_broad <= Omega_broad))
        )
        w_next = safediv(
            Omega_next - res_broad, Omega_next - Omega_broad, fill=jnp.nan
        )
        between_prev = (
            ((Omega_prev >= res_broad) & (res_broad >= Omega_broad))
            | ((Omega_prev <= res_broad) & (res_broad <= Omega_broad))
        )
        w_prev = safediv(
            Omega_prev - res_broad, Omega_prev - Omega_broad, fill=jnp.nan
        )
        res_weight = jnp.where(
            between_next, w_next, jnp.where(between_prev, w_prev, jnp.nan)
        )

    res_weight = jnp.where(valid[..., None], res_weight, jnp.nan)

    # Fourier analysis of radial drift
    ft_integrand = s_drift * tau_bounce
    ft_integrand = jnp.where(jnp.isfinite(ft_integrand), ft_integrand, jnp.nan)
    ft_integrand = jnp.where(valid[:, None, :, :], ft_integrand, jnp.nan)

    Delta_eta = eta_vals[1] - eta_vals[0] if eta_vals.shape[0] > 1 else 2 * jnp.pi

    if f_q_conservative:
        f_q_abs_sq = (Delta_eta / (4 * jnp.pi)) * jnp.sum(
            ft_integrand**2, axis=1
        )
        f_q_abs = jnp.sqrt(f_q_abs_sq)
        n_res = res_arr.shape[0]
        f_q_abs = jnp.broadcast_to(
            f_q_abs[..., None], (*f_q_abs.shape, n_res)
        )
        f_q_c = None
        f_q_s = None
    else:
        phase = q_arr[None, :] * eta_vals[:, None]
        cos_phase = jnp.cos(phase)
        sin_phase = jnp.sin(phase)
        f_q_c = jnp.sum(
            ft_integrand[..., None] * cos_phase[None, :, None, None, :], axis=1
        )
        f_q_s = jnp.sum(
            ft_integrand[..., None] * sin_phase[None, :, None, None, :], axis=1
        )
        ft_prefactor = Delta_eta / jnp.pi
        f_q_c = ft_prefactor * f_q_c
        f_q_s = ft_prefactor * f_q_s
        f_q_abs = 0.5 * jnp.sqrt(f_q_c**2 + f_q_s**2)

    # Island widths
    q_iw = q_arr[None, None, None, :]
    denom = jnp.pi * q_iw * jnp.abs(Omega_prime_s[..., None])
    Delta_s_profile = 4 * jnp.sqrt(
        safediv(f_q_abs, denom, fill=jnp.nan)
    )
    Delta_s_sum = jnp.nansum(Delta_s_profile, axis=-1)

    # f_res = Delta_s_sum**4
    f_res = jnp.ones_like(Delta_s_sum)

    # Diagnostics
    Delta_s = jnp.nansum(res_weight * Delta_s_profile, axis=0)
    s_vals = rhos**2
    w_sum = jnp.nansum(res_weight, axis=0)
    s_res = safediv(
        jnp.nansum(
            res_weight * s_vals[:, None, None, None],
            axis=0,
        ),
        w_sum,
        fill=jnp.nan,
    )

    return {
        'f_res': f_res,  # (rho, pitch, well)
        'Omega': Omega,  # (rho, pitch, well)
        'omega_bounce_avg': omega_bounce_avg,  # (rho, pitch, well)
        'eta_drift_avg': eta_drift_avg,  # (rho, pitch, well)
        'omega_bounce': omega_bounce,  # (rho, alpha, pitch, well)
        'eta_drift': eta_drift,  # (rho, alpha, pitch, well)
        'Omega_prime_s': Omega_prime_s,  # (rho, pitch, well)
        'res_weight': res_weight,  # (rho, pitch, well, res)
        'f_q_c': f_q_c,  # (rho, pitch, well, res) or None if conservative
        'f_q_s': f_q_s,  # (rho, pitch, well, res) or None if conservative
        'f_q_abs': f_q_abs,  # (rho, pitch, well, res)
        'Delta_s': Delta_s,  # (pitch, well, res), rho-weighted diagnostic
        'Delta_s_prof': Delta_s_profile,  # (rho, pitch, well, res)
        'f_q_conservative': f_q_conservative,  # scalar bool
        's_res': s_res,  # (pitch, well, res), rho-weighted resonance location
    }


@register_compute_fun(
    name="f_tr2",
    label=(
        "Eq. (50) in https://www.overleaf.com/project/68fabdc5d13eac7e69a15467"
    ),
    units="s^-2", # may want to add in units in "objectives/_neoclassical.py" compute() statement for normalization
    units_long="seconds squared",
    description="Trapped Particle Resonance Minimizer"
    "as defined by Eq. (50) in https://www.overleaf.com/project/68fabdc5d13eac7e69a15467",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["iota", "min_tz |B|", "max_tz |B|", "cvdrift0", "gbdrift", "fieldline length",
          "gbdrift (periodic)", "cvdrift (periodic)"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
# @partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"])
def f_tr2(params, transforms, profiles, data, **kwargs):
    """Trapped particle resonance penalty.

    Three stages:
      1. Bounce integrals  (per-surface, via ``_compute`` / ``batch_map``)
      2. Resonance physics  (cross-surface, via ``_resonance_physics``)
      3. Phase-space average (via ``_phase_space_average``)
    """
    # Parse kwargs
    num_pitch = kwargs.get("num_pitch", None)
    num_well = kwargs.get("num_well", None)
    grid = transforms["grid"].source_grid
    M = kwargs.get("M", 1)
    N = kwargs.get("N", 1)
    nfp = kwargs.get("nfp", None)
    KE_frac = kwargs.get("KE_frac", None)
    pitch_invs = kwargs.get("pitch_invs", None)
    rho_res = kwargs.get("rho_res", None)
    res_arr = kwargs.get("res_arr", None)
    p_arr = kwargs.get("p_arr", None)
    q_arr = kwargs.get("q_arr", None)
    pitch_method = kwargs.get("pitch_method", 0)
    quad = kwargs.get("quad", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    eta_vals = kwargs.get("eta_vals", None)
    DEBUG = kwargs.get("DEBUG", False)
    f_q_conservative = kwargs.get("f_q_conservative", False)
    weight_method = kwargs.get("weight_method", "linear")
    Delta_Omega = kwargs.get("Delta_Omega", None)

    iotas = grid.compress(data['iota'])
    rhos = grid.compress(grid.nodes[:, 0])

    # --- 1. Bounce integrals (per-surface, analogous to eps_eff/Gamma_c inner fn) ---
    def drifts(data):
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau, _alpha_drift, _s_drift = bounce.integrate(
            [_v_tau, _alpha_drift_integrand, _s_drift_integrand],
            data["pitch_inv"],
            data,
            ["cvdrift0", "gbdrift (periodic)", "cvdrift (periodic)"],
            num_well=num_well,
        )
        _alpha_drift = safediv(_alpha_drift, v_tau)
        _s_drift = safediv(_s_drift, v_tau)
        return _alpha_drift, _s_drift, points, v_tau, data

    alpha_drift_out, s_drift_out, points, vtau_out, _data = _compute(
        drifts,
        {
            "cvdrift0": data["cvdrift0"],
            "gbdrift (periodic)": data["gbdrift (periodic)"],
            "cvdrift (periodic)": data["cvdrift (periodic)"],
        },
        data,
        grid,
        num_pitch,
        surf_batch_size,
        pitch_invs=pitch_invs,
        pitch_method=pitch_method,
    )

    # --- 2. Resonance physics (cross-surface) ---
    res = _resonance_physics(
        alpha_drift_out, s_drift_out, vtau_out,
        iotas, rhos, rho_res, KE_frac, nfp, M, N,
        res_arr, q_arr, eta_vals,
        f_q_conservative, weight_method, Delta_Omega,
    )

    # --- 3. Phase-space average (normalized by full velocity space) ---
    fl_length = grid.compress(data["fieldline length"])
    f_res_avg = _phase_space_average(
        vtau_out, res['f_res'], _data["pitch_inv"], _data["pitch_inv weight"], fl_length,
    )

    # --- 4. Output ---
    if DEBUG:
        # DEBUG payload dimensions:
        # pitch_inv: (rho, pitch)
        # res_arr/p_arr/q_arr: (res,)
        # f_res_avg: (rho,)
        # plus keys returned by _resonance_physics (see inline comments there).
        data["f_tr2"] = {
            **res,
            'pitch_inv': _data['pitch_inv'],
            'res_arr': res_arr,
            'p_arr': p_arr,
            'q_arr': q_arr,
            'f_res_avg': f_res_avg,
        }
    else:
        data["f_tr2"] = grid.expand(f_res_avg)

    return data