"""Compute functions for neoclassical transport."""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import imap, jax, jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce1D, Bounce2D
# from ..integrals.quad_utils import chebgauss2
from ..utils import safediv
from .data_index import register_compute_fun

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
'''
def _compute(
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
'''

def _compute(fun, fun_data, data, grid, num_pitch, surf_batch_size=1, simp=False, pitch_invs=None, num_rho=None):
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
    fun_data["pitch_inv"], fun_data["pitch_inv weight"] = Bounce1D.get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
        simp=simp
    )
    if pitch_invs is not None:
        fun_data["pitch_inv"] = jnp.broadcast_to(pitch_invs, (grid.num_rho,len(pitch_invs) )) # needs to be shape (rho,pitch)
    out = batch_map(fun, fun_data, surf_batch_size)
    # assert out.ndim == 1
    # return grid.expand(out)
    return out


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
            data["theta"],
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
# @partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"]) # uncomment after debugging
def f_tr2(params, transforms, profiles, data, **kwargs):
    
    # Import kwargs
    num_pitch = kwargs.get("num_pitch",None)
    num_well = kwargs.get("num_well", None)
    grid = transforms["grid"].source_grid
    N = kwargs.get("N",0) # default is QA, N=0
    nfp = kwargs.get("nfp",None)
    KE_frac = kwargs.get("KE_frac",None)
    p_max = kwargs.get("p_max",5)
    q_max = kwargs.get("q_max",5)
    res_range_min = kwargs.get("res_range_min",-4)
    res_range_max = kwargs.get("res_range_max",4)
    bt_filter_flag = kwargs.get("bt_filter_flag",True)
    rt_filter_flag = kwargs.get("rt_filter_flag",True)
    include_zero_res = kwargs.get("include_zero_res",True)
    pitch_invs = kwargs.get("pitch_invs",None)
    alpha_res = kwargs.get("alpha_res",None)
    rho_res = kwargs.get("rho_res",None)
    Bcrit_res = kwargs.get("Bcrit_res",None)
    Psi = kwargs.get("Psi",None)
    wd_blur = kwargs.get('wd_blur',3)
    QS_flag = kwargs.get('QS_flag',False) # True for QS

    # Setup energies
    m_alpha = 6.6446573450*10**(-27) # kg, mass of alpha particle
    e = 1.602*10**(-19) # C
    Z=2 # fully ionized alpha particle
    KE = KE_frac * 5.6076*10**(-13) # J, 3.5 MeV if KE_frac=1
    v2 = 2*KE/m_alpha # m/s

    # Setup iotas
    iotas = grid.compress(data['iota']) # only look at the iotas on the surfaces specified by user

    # Setup resonances
    def tb_zr(res_arr_set,res_arr,q_arr):
        res_arr = res_arr.at[0].set(0)
        q_arr = q_arr.at[0].set(1)
        return res_arr_set+1,res_arr,q_arr
    def fb_zr(res_arr_set,res_arr,q_arr):
        return res_arr_set, res_arr, q_arr
    def false_branch_res_setup(res_arr_set,res_arr,p,q,q_arr): # do nothing
        return res_arr_set, res_arr, q_arr
    def true_branch_res_setup(res_arr_set,res_arr,p,q,q_arr): # add both positive and negative resonance to res_arr
        res_arr = res_arr.at[res_arr_set].set(p/q)
        res_arr = res_arr.at[res_arr_set+1].set(-p/q)
        q_arr = q_arr.at[res_arr_set].set(q)
        q_arr = q_arr.at[res_arr_set+1].set(q)
        return res_arr_set+2, res_arr, q_arr
    res_arr = jnp.full(2*p_max*q_max + 1, jnp.pi) # maximum possible size of array of resonances, including the zero resonance and negative resonances
    q_arr = jnp.full(2*p_max*q_max + 1, 1)
    res_arr_set = 0
    res_arr_set, res_arr, q_arr = jax.lax.cond(include_zero_res,tb_zr,fb_zr,res_arr_set,res_arr,q_arr)
    # p_max and q_max are Python integers so these loops remain differentiable with jax and jit
    for p in range(1,p_max+1): # include the zero resonance
        for q in range(1,q_max+1):
            condition = jnp.logical_and(
                ~jnp.isin(p/q, res_arr),
                jnp.logical_and(p/q >= res_range_min, p/q <= res_range_max)
                )
            res_arr_set, res_arr, q_arr = jax.lax.cond( condition, true_branch_res_setup, false_branch_res_setup, res_arr_set,res_arr,p,q,q_arr )

    # Setup quadratures
    # noqa: unused dependency
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )

    # Setup grid and resolutions
    grid = transforms["grid"].source_grid

    # Start with evaluation of bounce integrals (rho,alpha,Bcrit,well)
    def alpha_drift(data):
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau, _alpha_drift = bounce.integrate(
            [_v_tau, _poloidal_drift],
            data["pitch_inv"],
            data,
            ["gbdrift"],
            num_well=num_well,
        )
        _alpha_drift = safediv(2.0 * _alpha_drift , v_tau) # jnp.countnonzero, safediv will take out NaNs
        count_nz = jnp.count_nonzero(v_tau,axis=-1) # array size [rho][alpha][pitch]
        count_nz = jnp.sum(count_nz,axis=1) # array size [rho][pitch]

        return _alpha_drift, points, v_tau, count_nz, data["pitch_inv"]
    alpha_drift_out, points, vtau_out, count_nz, pitch_inv = ( # alpha_drift_out := (rho,alpha,Bcrit,wells). Energy will be added in later
        _compute(
            alpha_drift, 
            {"gbdrift": data["gbdrift"]},
            data,
            grid,
            num_pitch,
            surf_batch_size,
            pitch_invs=pitch_invs
        )
    )
    # count_nz will be 3D [rho,alpha,pitch]
    assert alpha_drift_out.shape[:-1] == (grid.num_rho,grid.num_alpha,num_pitch) # don't know well number yet, default is None, and assert is useable in optimization
    ado_shape = jnp.shape(alpha_drift_out)

    def psi_drift(data):
        bounce = Bounce1D(grid, data, quad, is_reshaped=True)
        v_tau, _psi_drift = bounce.integrate(
            [_v_tau, _radial_drift],
            data["pitch_inv"],
            data,
            ["cvdrift0"],
            num_well=num_well,
        )
        _psi_drift = safediv(2.0 * _psi_drift , v_tau) # jnp.countnonzero, safediv will take out NaNs

        return _psi_drift

    psi_drift_out = ( # psi_drift_out := (rho,alpha,Bcrit,wells) in order. Energy will be added in later
        _compute(
            psi_drift, 
            {"cvdrift0": data["cvdrift0"]},
            data,
            grid,
            num_pitch,
            surf_batch_size,
            pitch_invs=pitch_invs
        )
    ) 

    
    # Setup array allocations
    '''
    num_rho = ado_shape[0]
    num_fieldlines = ado_shape[1]
    num_Bcrit = ado_shape[2]
    num_wells = ado_shape[3]
    num_KE = len(KE_frac)
    '''
    
    # Setup mean and standard deviation functions
    def jnpmean_nz(x,axis=0):
        mask = x!=0.0
        count = jnp.sum(mask,axis) # how many wells that are not 0
        return jnp.sum(x,axis=axis) / count
    def jnpstd_nz(x,axis=0): # compute population standard deviation of an array while ignoring "0" elements in JAX numpy
        # x is an array with size: (num_rho,num_alpha,num_pitch,num_fieldlines)
        xbar = jnpmean_nz(x,axis=axis)
        xbar = jnp.broadcast_to(xbar[..., None], (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        # xbar = jnp.transpose(xbar, (2,0,1)) # rearrange to match initial x
        sq_diff = (x - xbar) ** 2
        return jnp.sqrt(jnpmean_nz(sq_diff,axis=axis))


    # Barely trapped filtering (if bt_filter_flag==True)
    def tb_btfilter(iotas,points,N,nfp,alpha_drift_out,psi_drift_out): # Perform barely trapped filter
        points_0 = points[0][:][:][:][:]
        points_1 = points[1][:][:][:][:]
        iotas_tb = jnp.broadcast_to(iotas[...,None,None,None],(iotas.shape[0],points_0.shape[1],points_0.shape[2],points_0.shape[3]))
        delta_chi = jnp.abs(jnp.abs(jnp.abs(points_0) - jnp.abs(points_1)) * (iotas_tb - N*nfp)) # zeta->chi assuming delta(alpha)=0
        return jnp.where(delta_chi < float(2*jnp.pi),alpha_drift_out,0.0),jnp.where(delta_chi < float(2*jnp.pi),psi_drift_out,0.0) # set barely-trapped particles to 0
    def fb_btfilter(iotas,points,N,nfp,alpha_drift_out,psi_drift_out): # Do nothing
        return alpha_drift_out, psi_drift_out
    alpha_drift_out,psi_drift_out = jax.lax.cond(bt_filter_flag,tb_btfilter,fb_btfilter,iotas,points,N,nfp,alpha_drift_out,psi_drift_out)

    # Ripple-trapped filtering (if rt_filter_flag==True)
    # Average and standard deviation per-surface and pitch inverse
    alpha_drift_avg = jnpmean_nz(alpha_drift_out,axis=3) # :=(rho,alpha,pitch), does not include zero wells in averaging
    alpha_drift_std = jnpstd_nz(alpha_drift_out,axis=3) # :=(rho,alpha,pitch)
    alpha_drift_avg = jnp.broadcast_to(alpha_drift_avg[..., None], (ado_shape[0], ado_shape[1], ado_shape[2], ado_shape[3])) # :=(rho,alpha,Bcrit,well)
    alpha_drift_std = jnp.broadcast_to(alpha_drift_std[..., None], (ado_shape[0], ado_shape[1], ado_shape[2], ado_shape[3])) # :=(rho,alpha,Bcrit,well)

    def tb_rtfilter(alpha_drift_out,psi_drift_out,alpha_drift_avg,alpha_drift_std): # Perform ripple-trapped filter
        return jnp.where(jnp.abs(alpha_drift_out - alpha_drift_avg) < 2*alpha_drift_std, alpha_drift_out, 0.0),jnp.where(jnp.abs(alpha_drift_out - alpha_drift_avg) < 2*alpha_drift_std, psi_drift_out, 0.0) # this is true if the value of interest is greater than 2 standard deviations away from the mean
    def fb_rtfilter(alpha_drift_out,psi_drift_out,alpha_drift_avg,alpha_drift_std): # Do nothing
        return alpha_drift_out, psi_drift_out
    alpha_drift_out, psi_drift_out = jax.lax.cond(rt_filter_flag,tb_rtfilter,fb_rtfilter,alpha_drift_out,psi_drift_out,alpha_drift_avg,alpha_drift_std)


    # Sum psi_drift_out term over alpha
    psi_drift_out = alpha_res * jnp.sum(psi_drift_out**2,axis=1) # := (rho,Bcrit,well)


    # # Omega eta calculation (currently for one energy only), average over alphas
    # tau_arr = vtau_out / jnp.sqrt(v2) # := (rho,alpha,Bcrit,well), vtau->tau
    # iotas_omega = jnp.broadcast_to(iotas[...,None,None,None],(iotas.shape[0], ado_shape[1], ado_shape[2], ado_shape[3]))
    # # omega_arr_test = (tau_arr*nfp / (2*jnp.pi * (N*nfp-iotas_omega))) * (m_alpha/(Z*e)) * alpha_drift_out * v2[0] # :=(rho,alpha,Bcrit,well) ONLY CONSIDERING ONE ENERGY
    # # omega_arr_test1 = (tau_arr*nfp / (2*jnp.pi * ((N*nfp)-iotas_omega))) * (m_alpha/(Z*e)) * alpha_drift_out
    # omega_arr_test = (tau_arr*nfp / (2*jnp.pi * ((N*nfp)-iotas_omega))) * (m_alpha/(Z*e)) * alpha_drift_out * v2[0] # :=(rho,alpha,Bcrit,well) ONLY CONSIDERING ONE ENERGY
    # # omega_arr = jnp.broadcast_to(omega_arr[...,None],(omega_arr.shape[0],omega_arr.shape[1],omega_arr.shape[2],omega_arr.shape[3],len(KE_frac))) * v2 # :=(rho,alpha,Bcrit,well,energy)
    # omega_arr = alpha_res * jnp.sum(omega_arr_test,axis=1) / (2*jnp.pi) # :=(rho,Bcrit,well)
        # Omega eta calculation (currently for one energy only), average over alphas
    tau_arr = vtau_out / jnp.sqrt(v2) # := (rho,alpha,Bcrit,well), vtau->tau
    iotas_omega = jnp.broadcast_to(iotas[...,None,None,None],(iotas.shape[0], ado_shape[1], ado_shape[2], ado_shape[3]))
    def tb_QS(nfp,N,iotas_omega):
        return (nfp / ((N*nfp)-iotas_omega))
    def fb_QS(nfp,N,iotas_omega):
        return jnp.ones(iotas_omega.shape)
    QS_factor = jax.lax.cond(QS_flag,tb_QS,fb_QS,nfp,N,iotas_omega)
    omega_arr_test = QS_factor * tau_arr * (m_alpha/(Z*e)) * alpha_drift_out * v2[0] / (2*jnp.pi) # :=(rho,alpha,Bcrit,well) ONLY CONSIDERING ONE ENERGY
    # omega_arr = jnp.broadcast_to(omega_arr[...,None],(omega_arr.shape[0],omega_arr.shape[1],omega_arr.shape[2],omega_arr.shape[3],len(KE_frac))) * v2 # :=(rho,alpha,Bcrit,well,energy)
    omega_arr = alpha_res * jnp.sum(omega_arr_test,axis=1) / (2*jnp.pi) # :=(rho,Bcrit,well)


    # Bump function calculation #

    # Misc parameters
    w=1 # width and amplitude parameter, recommended to keep =1

    # Setting wd
    def arr_max_1d(arr): # find the maximum of an array (jax/jit differentiable)
        def body_fun(i, carry):
            max_val, max_idx = carry
            new_val = arr[i]
            cond = new_val > max_val

            # jnp.where chooses elementwise between two values
            max_val = jnp.where(cond, new_val, max_val)
            max_idx = jnp.where(cond, i, max_idx)
            return (max_val, max_idx)
        
        carry_init = (arr[0], 0)
        max_val, max_idx = jax.lax.fori_loop(1, arr.shape[0], body_fun, carry_init)
        return {'max_i': max_idx, 'max_num': max_val}
    # wd takes a different value for each (Bcrit,well) combination
    max_rhospace = jax.vmap(jax.vmap(arr_max_1d,in_axes=1,out_axes=0),in_axes=2,out_axes=1) # apply arr_max_1d to each (Bcrit,well) combo
    wd = max_rhospace(omega_arr) # := (Bcrit,well)
    wd = wd_blur*wd['max_num']
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
    condition = jnp.logical_and(abs(y) < wd, res_broad!=jnp.pi) # check that corresponding omega value is less than wd away from the resonance and not jnp.pi (unset)

    # Create q array for division into bump function
    q_broad = q_arr[None,None,None,:] # make 4D array with q values on axis=3
    q_broad = jnp.broadcast_to(q_broad, (omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], q_arr.shape[0])) # := (rho,Bcrit,well,res)

    # Calculate bump function (f_b) and sum over resonances
    f_b_res = jnp.where(
        condition,
        safediv(jnp.exp(  jnp.clip( w * ((a-b)**2) / ( (omega_broad-b) * (omega_broad-a) ) ,-500,500)  ), q_broad), # clip to avoid overflow warning in jnp.exp()
        0
        ) # := (rho,Bcrit,well,res)
    f_b = jnp.sum(f_b_res,axis=-1) # := (rho,Bcrit,well)

    # First sum over rho
    iotas_rho1_sum = jnp.broadcast_to(iotas[...,None,None],(iotas.shape[0], ado_shape[2], ado_shape[3])) # := (rho,Bcrit,well)
    f_tr2_out = rho_res * jnp.sum( f_b * psi_drift_out / iotas_rho1_sum , axis=0 )  # := (Bcrit,well)

    # Sum over Bcrit
    f_tr2_out = jnp.broadcast_to(f_tr2_out[...,None,None],(ado_shape[2],ado_shape[3],ado_shape[0],ado_shape[1])) # := (Bcrit,well,rho,alpha)
    f_tr2_out = jnp.transpose(f_tr2_out,(2,3,0,1)) # := (rho,alpha,Bcrit,well)
    pitch_invs = jnp.broadcast_to(pitch_invs[...,None,None,None],(ado_shape[2],ado_shape[0],ado_shape[1],ado_shape[3])) # := (Bcrit,rho,alpha,well)
    pitch_invs = jnp.transpose(pitch_invs,(1,2,0,3)) # := (rho,alpha,Bcrit,well)
    f_tr2_out = Bcrit_res * jnp.sum(f_tr2_out * tau_arr * pitch_invs**(-2) , axis=2 ) # := (rho,alpha,well)

    # Second sum over alpha
    f_tr2_out = alpha_res * jnp.sum(f_tr2_out, axis=1) # := (rho,well)

    # Second sum over rho
    Psi_sqrt = jnp.sqrt(Psi) # I don't think Psi=0 ever with discretation of the rho grid in DESC
    Psi_sqrt = jnp.broadcast_to(Psi_sqrt[...,None],(ado_shape[0],ado_shape[3])) # := (rho,well)
    f_tr2_out = rho_res * jnp.sqrt(Psi[-1]) * jnp.sum(Psi_sqrt * f_tr2_out, axis=0) # := (well)

    # Sum over wells
    f_tr2_out = jnp.sum(f_tr2_out,axis=0) # scalar


    data["f_tr2"] = f_tr2_out # full output
    # data["f_tr2"] = { # for plotting/debugging
    #     'omega_arr':omega_arr_test,
    #     'psi_drift_out':psi_drift_out,
    #     'iotas_rho1_sum': iotas_rho1_sum,
    #     'f_b': f_b,
    #     'tau_arr': tau_arr,
    #     'nfp': nfp,
    #     'alpha_drift_out':alpha_drift_out,
    #     'pitch_inv':pitch_inv,
    #     'f_tr2_out':f_tr2_out,
    #     'iotas': iotas
    #     }
    return data