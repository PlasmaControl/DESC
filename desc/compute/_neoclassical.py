"""Compute functions for neoclassical transport."""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import imap, jax, jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce1D, Bounce2D
# from ..integrals.quad_utils import chebgauss2
from ..utils import safediv
from .data_index import register_compute_fun

from desc.objectives.utils import softmin, softmax

from ..integrals.quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
    simpson2
)
from ..integrals._bounce_utils import get_pitch_inv_quad
from quadax import simpson

# from ._fast_ion import _v_tau


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
def _L_ra_fsa(data, transforms, profiles, **kwargs):
    grid = transforms["grid"].source_grid
    L_ra = simpson(
        y=grid.meshgrid_reshape(1 / data["B^zeta"], "arz"),
        x=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
        axis=-1,
    )
    data["<L|r,a>"] = grid.expand(jnp.abs(_alpha_mean(L_ra)))
    return data


def _poloidal_drift(data, B, pitch):
    return safediv(
        data["gbdrift"] * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
    )

def _radial_drift(data, B, pitch):
    return safediv(
        data["cvdrift0"] * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
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
    data=["min_tz |B|", "max_tz |B|", "cvdrift0", "gbdrift", "fieldline length"]
    # data=["min_tz |B|", "max_tz |B|", "gbdrift", "fieldline length"]
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
    grid_og = transforms["grid"] # rtz grid
    grid = transforms["grid"].source_grid # use initial raz-specified grid
    M = kwargs.get("M",1) # default is QA, M=1
    N = kwargs.get("N",0) # default is QA, N=0
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

    # Bounce integral parameters
    quad = kwargs.get("quad",None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    # pitch_batch_size = kwargs.get("pitch_batch_size", 1)
    # pitch_batch_size = None
    # nufft_eps = kwargs.get("nufft_eps", 1e-6)
    # spline = kwargs.get("spline", True)
    # vander = kwargs.get("_vander", None)
    # num_transit = kwargs.get("num_transit",None)

    # Flags
    bt_filter_flag = kwargs.get("bt_filter_flag",False)
    rt_filter_flag = kwargs.get("rt_filter_flag",True)
    STAB_SACRIFICE = kwargs.get("STAB_SACRIFICE",True)
    QS_flag = kwargs.get("QS_flag",False) # True for QS (QA, QH configurations, not QI)
    LOSS_FRAC_WEIGHT = kwargs.get("LOSS_FRAC_WEIGHT",True)


    # Setup energies
    m_alpha = 6.6446573450*10**(-27) # kg, mass of alpha particle
    e = 1.602*10**(-19) # C
    Z=2 # fully ionized alpha particle
    KE = KE_frac * 5.6076*10**(-13) # J, 3.5 MeV if KE_frac=1
    v2 = 2*KE/m_alpha # m/s

    # Setup other grid-related parameters
    rhos = grid.nodes[grid.unique_rho_idx, 0]
    iotas = grid.compress(data['iota']) # only look at the iotas on the surfaces specified by user\
    # alphas = grid.nodes[grid.unique_alpha_idx, 1]
    # thetas = grid_og.nodes[grid_og.unique_theta_idx, 1]

    # Calculate critical magnetic field values on each rho surface

    # Start with evaluation of bounce integrals (rho,alpha,Bcrit,well)
    def drifts(data):
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau, _alpha_drift, _psi_drift = bounce.integrate(
            [_v_tau, _poloidal_drift, _radial_drift],
            data["pitch_inv"],
            data,
            ["gbdrift","cvdrift0"],
            num_well=num_well,
        )
        _alpha_drift = safediv(2.0 * _alpha_drift , v_tau) # safediv will take out NaNs
        _psi_drift = safediv(2.0 * _psi_drift , v_tau) # safediv will take out NaNs

        return _alpha_drift, _psi_drift, points, v_tau, data
    alpha_drift_out, psi_drift_out, points, vtau_out, data = ( # *_drift_out := (rho,alpha,Bcrit,wells). Energy will be added in at some other time
        _compute(
            drifts, 
            {"gbdrift": data["gbdrift"],"cvdrift0": data["cvdrift0"]},
            data,
            grid,
            num_pitch,
            surf_batch_size, # avoid jax's vectorizing if set to 1 in the rho dimension
            pitch_invs=pitch_invs,
            pitch_method=pitch_method # 1 for uniform pitch inverses across each surface
        )
    )

    # Use Bounce2D to evaluate bounce integrals (rho,alpha,Bcrit,well)
    # is the grid theta grid? how do I get it to be alpha? grid.compress() afterwards?
    # how do I deal with pitch batch size?
    # Y_B=None
    # def drifts(data):
    #     bounce = Bounce2D(
    #             grid,
    #             data,
    #             data["theta"],
    #             Y_B,
    #             alphas,
    #             num_transit,
    #             quad,
    #             # nufft_eps=nufft_eps,
    #             is_fourier=True,
    #             # spline=spline,
    #             # vander=vander,
    #         )
    #     points = bounce.points(data["pitch_inv"], num_well=num_well)
    #     def fun(pitch_inv):
    #         v_tau, alpha_drift_out, psi_drift_out = bounce.integrate(
    #             [_v_tau, _poloidal_drift, _radial_drift],
    #             pitch_inv,
    #             data,
    #             ["gbdrift","cvdrift0"],
    #             num_well=num_well,
    #             nufft_eps=nufft_eps,
    #             is_fourier=True,
    #         )
    #         _alpha_drift = safediv(2.0 * _alpha_drift , v_tau) # safediv will take out NaNs
    #         _psi_drift = safediv(2.0 * _psi_drift , v_tau) # safediv will take out NaNs
    #         return v_tau, alpha_drift_out, psi_drift_out, points

    #     return batch_map(fun, data["pitch_inv"], pitch_batch_size) # batch for efficiency (avoid jax vectorized maps)
    # vtau_out, alpha_drift_out, psi_drift_out, points = (
    #     _compute2D(
    #         drifts,
    #         {"gbdrift": data["gbdrift"],"cvdrift0": data["cvdrift0"]},
    #         data,
    #         thetas,
    #         grid,
    #         num_pitch,
    #         surf_batch_size,
    #         simp=True,
    #     )
    # )

    assert alpha_drift_out.shape[:-1] == (grid.num_rho,grid.num_alpha,num_pitch) # don't know well number yet, default is None, and assert is useable in optimization
    ado_shape = jnp.shape(alpha_drift_out)

    Bcrit_res = data['Bcrit_res']
    pitch_invs = data['pitch_inv']

    
    # Setup array allocations
    '''
    num_rho = ado_shape[0]
    num_fieldlines = ado_shape[1]
    num_Bcrit = ado_shape[2]
    num_wells = ado_shape[3]
    num_KE = len(KE_frac)
    '''
    
    # Setup mean and standard deviation functions
    def jnpmean_nz(x,axis=0,fill=0): # if all values in x along axis = 0, this outputs zero. This is okay for f_b, well=0 points where no bounce occurs will be taken care of by psi_drift_out
        mask = x!=0.0
        count = jnp.sum(mask,axis) # how many wells that are not 0
        return safediv(jnp.sum(x,axis=axis) , count, fill=fill)
    def jnpstd_nz(x,axis=0): # compute population standard deviation of an array while ignoring "0" elements in JAX numpy
        # x is an array with size: (num_rho,num_alpha,num_pitch,num_fieldlines)
        xbar = jnpmean_nz(x,axis=axis)
        xbar = jnp.broadcast_to(xbar[..., None], (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        # xbar = jnp.transpose(xbar, (2,0,1)) # rearrange to match initial x
        sq_diff = (x - xbar) ** 2
        return jnp.sqrt(jnpmean_nz(sq_diff,axis=axis))


    ##### PARTICLE FILTERING #####

    # Barely trapped filtering (if bt_filter_flag==True)
    def tb_btfilter(iotas,points,N,nfp,psi_drift_out,M): # Perform barely trapped filter
        points_0 = points[0][:][:][:][:] # zeta0 (cylindrical)
        points_1 = points[1][:][:][:][:] # zeta1 (cylindrical)
        iotas_tb = jnp.broadcast_to(iotas[...,None,None,None],(iotas.shape[0],points_0.shape[1],points_0.shape[2],points_0.shape[3]))
        delta_chi = jnp.abs(jnp.abs(points_0 - points_1) * (M*iotas_tb - N*nfp)) # zeta->chi assuming delta(alpha)=0
        return jnp.where(delta_chi < float(2.5*jnp.pi),psi_drift_out,0.0) # set barely-trapped particles to 0
    def fb_btfilter(iotas,points,N,nfp,psi_drift_out,M): # Do nothing
        return psi_drift_out
    # Only need to filter on psi_drift_out to filter all of f
    psi_drift_out = jax.lax.cond(bt_filter_flag,tb_btfilter,fb_btfilter,iotas,points,N,nfp,psi_drift_out,M)
   
    # Ripple-trapped filtering (if rt_filter_flag==True)
    tau_arr = vtau_out / jnp.sqrt(v2) # := (rho,alpha,Bcrit,well), vtau->tau
    tau_well_avg = jnpmean_nz(tau_arr,axis=3) # :=(rho,alpha,pitch), does not include zero wells in averaging
    tau_well_std = jnpstd_nz(tau_arr,axis=3) # :=(rho,alpha,pitch)
    tau_well_avg = jnp.broadcast_to(tau_well_avg[..., None], (ado_shape[0], ado_shape[1], ado_shape[2], ado_shape[3])) # :=(rho,alpha,Bcrit,well)
    tau_well_std = jnp.broadcast_to(tau_well_std[..., None], (ado_shape[0], ado_shape[1], ado_shape[2], ado_shape[3])) # :=(rho,alpha,Bcrit,well)
    def tb_rtfilter(psi_drift_out,tau_arr,tau_well_avg,tau_well_std): # Perform ripple-trapped filter
        return jnp.where(jnp.abs(tau_arr - tau_well_avg) < 2*tau_well_std, psi_drift_out, 0.0) # this is true if the value of interest is greater than 2 standard deviations away from the mean
    def fb_rtfilter(psi_drift_out,tau_arr,tau_well_avg,tau_well_std): # Do nothing
        return psi_drift_out
    # Only need to filter on psi_drift_out to filter all of f
    psi_drift_out = jax.lax.cond(rt_filter_flag,tb_rtfilter,fb_rtfilter,psi_drift_out,tau_arr,tau_well_avg,tau_well_std)


    ##### BUMP FUNCTION TERM #####

    # Omega eta calculation (currently for one energy only), average over alphas
    iotas_omega = jnp.broadcast_to(iotas[...,None,None,None],(iotas.shape[0], ado_shape[1], ado_shape[2], ado_shape[3]))
    def tb_QS(nfp,N,iotas_omega):
        return safediv(nfp , ((N*nfp)-iotas_omega))
    def fb_QS(nfp,N,iotas_omega):
        return jnp.ones(iotas_omega.shape)
    QS_factor = jax.lax.cond(QS_flag,tb_QS,fb_QS,nfp,N,iotas_omega)
    omega_arr_test = QS_factor * tau_arr * (m_alpha/(Z*e)) * alpha_drift_out * v2[0] / (2*jnp.pi) # :=(rho,alpha,Bcrit,well)
    # omega_arr = alpha_res * jnp.sum(omega_arr_test,axis=1) / (2*jnp.pi) # :=(rho,Bcrit,well), concern in this line about if there are values that don't have an Omega_eta in omega_arr, they will skew the results
    omega_arr = jnpmean_nz(omega_arr_test,axis=1,fill=11.0) # :=(rho,Bcrit,well), will return 11.0 only if there was not a single field line for this (rho,Bc,well) combination that had a non-0.0 value (extremely unlikely for something close to the zero resonance but will be true if not trapped at this combination)

    # Setting wd (wd=DeltaOmega)
    # wd takes a different value for each (Bcrit,well) combination
    domega_arr = jnp.abs(omega_arr[1:,:,:] - omega_arr[:-1,:,:]) # := (rho-1,Bcrit,well)
    wd = wd_blur * softmax(domega_arr,alpha=50,axis=0) / 2 # := (Bcrit,well), wd really specifies the half-width of the bump function

    # Check if wd needs to be cropped if resolution or shear issues
    wd_max = softmax(omega_arr,alpha=50,axis=0) # := (Bcrit,well)
    wd_max = jnp.ones(wd_max.shape) * 0.05 * wd_max # if all elements needed to be cropped
    wd_min = softmin(omega_arr,alpha=50,axis=0) # := (Bcrit,well)
    wd_min = jnp.ones(wd_min.shape) * 0.01 * wd_max # if all elements needed to be cropped
    wd = jnp.where(wd > wd_max,wd_max,wd) # limit max size of wd based on 10% of wd_max
    wd = jnp.where(wd < wd_min,wd_min,wd) # limit min size of wd based on 1% of wd_max

    # Could include these warnings by checking if wd is the same before and after the jnp.where lines
    # if verbose:
    #     jax.debug.print("WARNING: Delta_Omega may be too small to capture all rational crossings! Try increasing wd_blur.")
    # if verbose:
    #     jax.debug.print("WARNING: Delta_Omega may be too big and not capturing individual rational crossings well! Try decreasing wd_blur or increasing rho resolution.")

    wd = jnp.broadcast_to(wd[...,None],(wd.shape[0],wd.shape[1],ado_shape[0])) # := (Bcrit,well,rho)
    wd = jnp.transpose(wd,(2,0,1)) # := (rho,Bcrit,well)

    # Setting up resonance arrays
    res_broad = res_arr[None,None,None,:] # := (rho,Bcrit,well,res)
    res_broad = jnp.broadcast_to(res_broad, (ado_shape[0], ado_shape[2], ado_shape[3], res_arr.shape[0])) # := (rho,Bcrit,well,res)
    wd = jnp.broadcast_to(wd[...,None],(wd.shape[0],wd.shape[1],wd.shape[2],res_arr.shape[0])) # := (rho,Bcrit,well,res)
    omega_broad = jnp.broadcast_to(omega_arr[...,None], (omega_arr.shape[0],omega_arr.shape[1],omega_arr.shape[2],res_arr.shape[0])) # := (rho,Bcrit,well,res)

    # Conditional for non-zero bump
    a = res_broad + wd
    b = res_broad - wd
    y = omega_broad - res_broad
    condition = jnp.logical_and( jnp.logical_and(abs(y) < wd, res_broad!=jnp.pi) , omega_broad!=11.0 ) # check that corresponding Omega_eta value is less than wd away from the resonance and not res_broad==jnp.pi (no resonance considered at this array element) and omega_broad!=11.0 so a trapping condition exists somewhere in this (rho,Bc,well) combination

    # Calculate bump function (f_b)
    w=1 # width and amplitude parameter, recommended to keep =1
    f_b = jnp.where(
        condition,
        jnp.exp(  jnp.clip( safediv(w * ((a-b)**2) , ( (omega_broad-b) * (omega_broad-a)) ) ,-500,500)  ), # clip to avoid overflow warning in jnp.exp()
        0
        ) # := (rho,Bcrit,well,res)
        

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
        omega_prime = jnp.broadcast_to(omega_prime[...,:],(omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], q_arr.shape[0])) # := (rho,Bcrit,well,res)
        Deltarho_4 = safediv(psi_drift_out , omega_prime*(q_broad**2)) # := (rho,Bcrit,well,res)


    ##### WEIGHTING BASED ON PLASMA EDGE VICINITY #####
    # rhos = grid.nodes[:,0]
    if LOSS_FRAC_WEIGHT:
        rho_max = jnp.broadcast_to(rhos[...,None,None,None],(omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], q_arr.shape[0])) # := (rho,Bcrit,well,res)
    else:
        rho_max=1 # no weighting


    ##### PHASE-SPACE AVERAGING #####
    f = jnp.sum( rho_max * f_b * Deltarho_4 ,axis=-1) # := (rho,Bcrit,well)

    # First sum over rho
    # iotas_sum = jnp.broadcast_to(iotas[...,None,None],(iotas.shape[0], ado_shape[2], ado_shape[3])) # := (rho,Bcrit,well)
    # f_tr2_out = rho_res * jnp.sum( safediv( f , iotas_sum) , axis=0 )  # := (Bcrit,well)

    # Sum over Bcrit
    f_tr2_out = jnp.broadcast_to(f[...,None],(ado_shape[0],ado_shape[2],ado_shape[3],ado_shape[1])) # := (rho,Bcrit,well,alpha)
    f_tr2_out = jnp.transpose(f_tr2_out,(0,3,1,2)) # := (rho,alpha,Bcrit,well)
    pitch_invs = jnp.broadcast_to(pitch_invs[...,None,None],(ado_shape[0],ado_shape[2],ado_shape[1],ado_shape[3])) # := (rho,Bcrit,alpha,well)
    pitch_invs = jnp.transpose(pitch_invs,(0,2,1,3)) # := (rho,alpha,Bcrit,well)
    Bcrit_res = jnp.broadcast_to(Bcrit_res[:,None,None,None],(ado_shape[0],ado_shape[1],ado_shape[2],ado_shape[3])) # := (rho,alpha,Bcrit,well))
    f_tr2_out = jnp.sum(Bcrit_res * f_tr2_out * tau_arr * pitch_invs**(-2) , axis=2 ) # := (rho,alpha,well)

    # Second sum over alpha
    f_tr2_out = alpha_res * jnp.sum(f_tr2_out, axis=1) # := (rho,well)

    # Second sum over rho
    rhos = jnp.broadcast_to(rhos[...,None],(ado_shape[0],ado_shape[3])) # := (rho,well)
    f_tr2_out = rho_res * jnp.sum(rhos * f_tr2_out, axis=0) # := (well)

    # Sum over wells
    f_tr2_out = jnp.sum(f_tr2_out,axis=0) # scalar


    data["f_tr2"] = f_tr2_out # full output
    # data["f_tr2"] = { # for plotting/debugging
    #     'omega_arr':omega_arr,
    #     'f_b': f_b,
    #     'f_tr2_out':f_tr2_out,
    #     'rhos': rhos,
    #     'res_arr': res_arr,
    #     'pitch_inv': data['pitch_inv'],
    #     'p_res': data['Bcrit_res']
    #     }
    return data