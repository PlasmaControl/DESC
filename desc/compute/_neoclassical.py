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
    if pitch_method==0: # not sure what to do for Bcrit_res in this one for integration
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
    elif pitch_method==2:
        fun_data["pitch_inv"] = jnp.broadcast_to(pitch_invs, (grid.num_rho,len(pitch_invs) )) # needs to be shape (rho,Bcrit)
        fun_data["Bcrit_res"] = fun_data["pitch_inv"][:,1]-fun_data["pitch_inv"][:,0] # := (rho)
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

def _theta_drift_integrand(data, B, pitch):
    return safediv(
        2 * (data["gbdrift (periodic)"] * pitch * B + 2 * (1 - pitch * B) * data["cvdrift (periodic)"]), jnp.sqrt(jnp.abs(1 - pitch * B))
        # 2 * (data["gbdrift_theta"] * pitch + 2 * (1 - pitch * B) * data["cvdrift_theta"]), jnp.sqrt(jnp.abs(1 - pitch * B))
    )

def _phi_drift_integrand(data, B, pitch):
    return safediv(
        2 * (data["gbdrift (periodic)"] * pitch * B + 2 * (1 - pitch * B) * data["cvdrift (periodic)"]), jnp.sqrt(jnp.abs(1 - pitch * B))
        # 2 * (data["gbdrift_phi"] * pitch + 2 * (1 - pitch * B) * data["cvdrift_phi"]), jnp.sqrt(jnp.abs(1 - pitch * B))
    )

_bounce1D_doc = {
    "num_well": _bounce_doc["num_well"],
    "num_quad": _bounce_doc["num_quad"],
    "num_pitch": _bounce_doc["num_pitch"],
    "surf_batch_size": _bounce_doc["surf_batch_size"],
    "quad": _bounce_doc["quad"],
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
          "gbdrift_theta", "gbdrift_phi", "cvdrift_theta", "cvdrift_phi",
          "gbdrift (periodic)", "cvdrift (periodic)"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
# @partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"]) # uncomment after debugging, maybe not
def f_tr2(params, transforms, profiles, data, **kwargs):
    
    # Import kwargs
    num_pitch = kwargs.get("num_pitch",None)
    num_well = kwargs.get("num_well", None)
    grid = transforms["grid"].source_grid # use initial raz-specified grid
    M = kwargs.get("M",1) # default is QA, M=1
    nfp = kwargs.get("nfp",None)
    KE_frac = kwargs.get("KE_frac",None)
    pitch_invs = kwargs.get("pitch_invs",None)
    alpha_res = kwargs.get("alpha_res",None)
    rho_res = kwargs.get("rho_res",None)
    Bcrit_res = kwargs.get("Bcrit_res",None)
    wd_blur = kwargs.get("wd_blur",1.25)
    res_arr = kwargs.get("res_arr",None)
    q_arr = kwargs.get("q_arr",None)
    pitch_method = kwargs.get("pitch_method",1)
    psi_a = data["Psi"][-1] # Total toroidal flux

    # Bounce integral parameters
    quad = kwargs.get("quad",None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)

    # Flags
    DEBUG = kwargs.get("DEBUG",False)

    # Setup energies
    m_alpha = 6.6446573450*10**(-27) # kg, mass of alpha particle
    e = 1.602*10**(-19) # C
    Z=2 # fully ionized alpha particle
    KE = KE_frac * 5.6076*10**(-13) # J, 3.5 MeV if KE_frac=1
    v2 = 2*KE/m_alpha # m/s

    iotas = grid.compress(data['iota'])

    # Start with evaluation of bounce integrals (rho,alpha,Bcrit,well)
    # Select drift integrands based on M:
    #   M != 0: use eta = nfp * phi (toroidal) components
    #   M  = 0: use eta = theta (poloidal) components
    # The combined integrand computes: λ*gbdrift_η + 2(1−λB)*cvdrift_η
    # The "psi" (radial) drift is evaluated separately via cvdrift0.
    if M == 0:
        _eta_integrand_integrand = _theta_drift_integrand
        _gb_key = "gbdrift_theta"
        _cv_key = "cvdrift_theta"
    else:
        _eta_integrand_integrand = _phi_drift_integrand
        _gb_key = "gbdrift_phi"
        _cv_key = "cvdrift_phi"

    def drifts(data):
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau, _eta_drift, _s_drift = bounce.integrate(
            [_v_tau, _eta_integrand_integrand, _s_drift_integrand],
            data["pitch_inv"],
            data,
            [_gb_key, _cv_key, "cvdrift0", "gbdrift (periodic)", "cvdrift (periodic)"],
            num_well=num_well,
        )
        # eta drift: bounce-averaged v_d · ∇η (η = θ or φ)
        _eta_drift = safediv(_eta_drift, v_tau)
        # psi (radial) drift evaluated separately
        _s_drift = safediv(_s_drift, v_tau)

        return _eta_drift, _s_drift, points, v_tau, data

    eta_drift_out, s_drift_out, points, vtau_out, _data = ( # *_drift_out := (rho,alpha,Bcrit,wells). Energy will be added in at some other time
        _compute(
            drifts,
            {_gb_key: data[_gb_key],
             _cv_key: data[_cv_key],
             "cvdrift0": data["cvdrift0"],
             "gbdrift (periodic)": data["gbdrift (periodic)"],
             "cvdrift (periodic)": data["cvdrift (periodic)"]},
            data,
            grid,
            num_pitch,
            surf_batch_size, # avoid jax's vectorizing if set to 1 in the rho dimension
            pitch_invs=pitch_invs,
            pitch_method=pitch_method # 1 for uniform pitch inverses across each surface
        )
    )

    # Bounce-averaged drifts
    eta_drift = eta_drift_out * KE / (Z * e)
    if M != 0:
        eta_drift *= nfp # eta = nfp * zeta
        
    ado_shape = eta_drift.shape
    iotas_omega = jnp.broadcast_to(iotas[...,None,None,None],(iotas.shape[0], ado_shape[1], ado_shape[2], ado_shape[3]))
    def tb_QS(nfp,N,iotas_omega):
        return safediv(nfp , ((N*nfp)-iotas_omega))
    def fb_QS(nfp,N,iotas_omega):
        return jnp.ones(iotas_omega.shape)
    QS_factor = jax.lax.cond(M==0,tb_QS,fb_QS,nfp,1,iotas_omega)
    eta_drift *= QS_factor

    s_drift = s_drift_out * KE / (Z * e)
    tau_bounce = vtau_out / jnp.sqrt(v2)
    omega_bounce = 2 * jnp.pi / tau_bounce
    # Normalized frequency
    Omega = eta_drift / omega_bounce

    pitch_invs = _data['pitch_inv']
    
    # Setup mean and standard deviation functions
    def jnpmean_nz(x,axis=0,fill=0): # if all values in x along axis = 0, this outputs zero. This is okay for f_b, well=0 points where no bounce occurs will be taken care of by psi_drift_out
        mask = x!=0.0
        count = jnp.sum(mask,axis) # how many wells that are not 0
        return safediv(jnp.sum(x,axis=axis) , count, fill=fill)

    # Omega eta calculation (currently for one energy only), average over alphas
    Omega_avg = jnpmean_nz(Omega,axis=1,fill=11) # :=(rho,Bcrit,well), will return 11.0 only if there was not a single field line for this (rho,Bc,well) combination that had a non-0.0 value (extremely unlikely for something close to the zero resonance but will be true if not trapped at this combination)

    f = s_drift / omega_bounce  

    """
    # Setting wd (wd=DeltaOmega)
    # wd takes a different value for each (Bcrit,well) combination
    dOmega_arr = jnp.abs(Omega_avg[1:,:,:] - Omega_avg[:-1,:,:]) # := (rho-1,Bcrit,well)

    # Setting up resonance arrays
    res_broad = res_arr[None,None,None,:] # := (rho,Bcrit,well,res)
    res_broad = jnp.broadcast_to(res_broad, (ado_shape[0], ado_shape[2], ado_shape[3], res_arr.shape[0])) # := (rho,Bcrit,well,res)
    wd = jnp.broadcast_to(wd[...,None],(wd.shape[0],wd.shape[1],wd.shape[2],res_arr.shape[0])) # := (rho,Bcrit,well,res)
    omega_broad = jnp.broadcast_to(omega_arr[...,None], (omega_arr.shape[0],omega_arr.shape[1],omega_arr.shape[2],res_arr.shape[0])) # := (rho,Bcrit,well,res)

  

    ##### ISLAND WIDTH TERM #####
    # Sum psi_drift_out term over alpha
    psi_drift_out = alpha_res * jnp.sum(psi_drift_out**2,axis=1) # := (rho,Bcrit,well)
    psi_drift_out = jnp.broadcast_to(psi_drift_out[...,None],(omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], q_arr.shape[0])) # := (rho,Bcrit,well,res)

    # Create n array for island width - make 4D array with n values on axis=3
    q_broad = jnp.broadcast_to(q_arr[None,None,None,:], (omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], q_arr.shape[0])) # := (rho,Bcrit,well,res)

    # Calculate island width or modified island width
    if STAB_SACRIFICE:
        Deltarho_4 = safediv(psi_drift_out , q_broad**2) # := (rho,Bcrit,well,res)
    else:
        omega_prime = jnp.where(omega_arr == 11.0, 0 , jnp.gradient(omega_arr,rho_res,axis=0))# := (rho,Bcrit,well), omega_arr is :=(rho,Bcrit,well)
        omega_prime = jnp.broadcast_to(omega_prime[...,None],(omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], q_arr.shape[0])) # := (rho,Bcrit,well,res)
        Deltarho_4 = safediv(psi_drift_out , omega_prime*(q_broad**2)) # := (rho,Bcrit,well,res)
    rhos_broad = jnp.broadcast_to(rhos[...,None,None,None],(omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], q_arr.shape[0])) # := (rho,Bcrit,well,res)
    Deltarho_4 = safediv(Deltarho_4,(psi_a**4)*(rhos_broad**3)) # one rhos_broad factor cancels with Jacobian

    ##### WEIGHTING BASED ON PLASMA EDGE VICINITY #####
    if LOSS_FRAC_WEIGHT:
        rho_max = rhos_broad**2
    else:
        rho_max=1 # no weighting


    ##### PHASE-SPACE AVERAGING #####
    f = jnp.sum( rho_max * f_b * Deltarho_4 ,axis=-1) # := (rho,Bcrit,well)

    # Sum over Bcrit
    f_tr2_out = jnp.broadcast_to(f[...,None],(ado_shape[0],ado_shape[2],ado_shape[3],ado_shape[1])) # := (rho,Bcrit,well,alpha)
    f_tr2_out = jnp.transpose(f_tr2_out,(0,3,1,2)) # := (rho,alpha,Bcrit,well)
    pitch_invs = jnp.broadcast_to(pitch_invs[...,None,None],(ado_shape[0],ado_shape[2],ado_shape[1],ado_shape[3])) # := (rho,Bcrit,alpha,well)
    pitch_invs = jnp.transpose(pitch_invs,(0,2,1,3)) # := (rho,alpha,Bcrit,well)
    Bcrit_res = jnp.broadcast_to(Bcrit_res[:,None,None,None],(ado_shape[0],ado_shape[1],ado_shape[2],ado_shape[3])) # := (rho,alpha,Bcrit,well))
    f_tr2_out = jnp.sum( safediv(Bcrit_res * f_tr2_out * tau_arr , pitch_invs**2) , axis=2 ) # := (rho,alpha,well)

    # Second sum over alpha
    f_tr2_out = alpha_res * jnp.sum(f_tr2_out, axis=1) # := (rho,well)

    # Sum over rho
    f_tr2_out = rho_res * jnp.sum(f_tr2_out, axis=0) # := (well)

    # Sum over wells
    f_tr2_out = jnp.sum(f_tr2_out,axis=0) # scalar
    """
    if DEBUG:
        data["f_tr2"] = { # for plotting/debugging
            # 'omega_arr':omega_arr,
            # 'f_b': f_b,
            # 'f_tr2_out':f_tr2_out,
            # 'res_arr': res_arr,
            'Omega': Omega,
            'omega_bounce': omega_bounce,
            'pitch_inv': _data['pitch_inv'],
            'Omega_avg': Omega_avg,
            'eta_drift': eta_drift,
            # 'p_res': data['Bcrit_res'],
            # 'Deltarho_4': Deltarho_4,
            # 'Omega_prime': omega_prime,
            # 'wd': wd,
            }
    else:
        data["f_tr2"] = Omega # full output
        
    return data