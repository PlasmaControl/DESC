"""Compute functions for neoclassical transport."""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import imap, jax, jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce1D, Bounce2D
from ..integrals.quad_utils import chebgauss2
from ..utils import safediv
from .data_index import register_compute_fun

from ..integrals.quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
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

def _compute(fun, fun_data, data, grid, num_pitch, surf_batch_size=1, simp=False):
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
        simp=simp,
    )
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


@register_compute_fun(
    name="f_tr",
    label=(
        "\\f_tr = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c_denom^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Trapped particle resonance objective function",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "cvdrift0", "gbdrift", "<L|r,a>"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    quad="jnp.ndarray : Optional, quadrature points and weights for bounce integrals.",
    num_pitch="int : Resolution for quadrature over velocity coordinate. Default 64.",
    num_well=(
        "int : Maximum number of wells to detect for each pitch and field line. "
        "Default is to detect all wells, but due to limitations in JAX this option "
        "may consume more memory. Specifying a number that tightly upper bounds "
        "the number of wells will increase performance. "
    ),
    batch="bool : Whether to vectorize part of the computation. Default is true.",
)
@partial(jit, static_argnames=["num_pitch", "num_well", "batch"])
def f_tr(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Velasco et al.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    """
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(leggauss(32), (automorphism_sin, grad_automorphism_sin))
    )
    num_pitch = kwargs.get("num_pitch",1)
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    grid = transforms["grid"].source_grid
    # N = kwargs.get("N",-1) # ?????
    N=1
    nfp = kwargs.get("nfp",jnp.nan)
    KE_frac = kwargs.get("KE_frac",0.00000001)
    p_max = kwargs.get("p_max",2)
    q_max = kwargs.get("q_max",2)
    res_range_min = kwargs.get("res_range_min",0)
    res_range_max = kwargs.get("res_range_max",4)
    bt_filter_flag = kwargs.get("bt_filter_flag",True)
    rt_filter_flag = kwargs.get("rt_filter_flag",True)

    def d_v_tau(data, B, pitch):
        return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B))) # abs to avoid divide by zero error

    def drift(data, B, pitch):
        return safediv(
            data["gbdrift"] * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
        )

    # def drift(data, B, pitch):
    #     f = 
    #     return safediv(f * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B)))

    def compute_ad(data):
        bounce = Bounce1D(grid, data, quad, automorphism=None, is_reshaped=True)
        points = bounce.points(data["pitch_inv"], num_well=num_well)
        v_tau = bounce.integrate(d_v_tau, data["pitch_inv"], points=points, batch=batch)
        _alpha_drift = bounce.integrate(
            drift,
            pitch_inv=data["pitch_inv"],
            # f=data["gbdrift"], # f
            points=points,
            batch=batch,
        )
        _alpha_drift = 2.0 * _alpha_drift / v_tau

        return _alpha_drift, points, v_tau

    _data = {  # noqa: unused dependency
        name: Bounce1D.reshape_data(grid, data[name])
        for name in Bounce1D.required_names + ["cvdrift0", "gbdrift"]
    }
    _data = _get_pitch_inv_quad(grid, data, num_pitch, _data)
    #data["alpha_drift"] = grid.expand(_alpha_mean(map2(compute, _data))) # takes mean over all field lines
    data["alpha_drift"], points, vtau_out = map2(compute_ad,_data) # alpha_drift has indices [alpha,rho,pitch,well num]
    
    
    alpha_drift_out = data["alpha_drift"]
    ado_shape = jnp.shape(alpha_drift_out)
    vtau_out = jnp.where(vtau_out == 0, jnp.nan, vtau_out) # in case filters are turned off

    # specify energy
    m_alpha = 6.6446573450*10**(-27) # kg, mass of alpha particle
    e = 1.602*10**(-19) # C
    Z=2 # fully ionized alpha particle
    KE = KE_frac * 5.6076*10**(-13) # J, 3.5 MeV if KE_frac=1
    v2 = 2*KE/m_alpha # m/s

    iotas = data['iota'] # ????? need to align to surfaces better
    rhos = jnp.zeros((grid.num_rho))
    rho_loop=0
    for idx in grid.unique_rho_idx:
        rhos = rhos.at[rho_loop].set(grid.nodes[idx,0])
        rho_loop+=1
    tau_arr = jnp.zeros((len(rhos),ado_shape[2])) # axis=0 is rhos, axis=1 is pitch inv
    omega_arr = jnp.zeros((len(rhos),ado_shape[2])) # axis=0 is rhos, axis=1 is pitch invs


    def false_branch_res_setup(indict): # do nothing
        return indict['res_arr_set'], indict['res_arr']
    def true_branch_res_setup(indict): # add to res_arr
        res_arr = indict['res_arr']
        res_arr = res_arr.at[indict['res_arr_set']].set(indict['p']/indict['q'])
        return indict['res_arr_set']+1, res_arr

    def jnpnanstd(x,axis): # compute standard deviation of an array while ignoring "nan" elements in JAX numpy
        # x is an array with size: (num_wells*num_fieldlines,num_rho,num_pitch)
        xbar = jnp.nanmean(x,axis=0)
        mask = ~jnp.isnan(x)
        count = jnp.sum(mask) # how many elements are in the array that are not "nan"s
        xbar = jnp.broadcast_to(xbar[..., None], (xbar.shape[0], xbar.shape[1], num_wells*num_fieldlines))
        xbar = jnp.transpose(xbar, (2,0,1)) # rearrange to match alpha_drift_avg above
        sq_diff = jnp.where(mask, (x - xbar) ** 2, 0.0) # replace array elements that are "nan" with "0"
        return jnp.sqrt(jnp.sum(sq_diff,axis=0) / count) # returning an array with size: (num_rho,num_pitch)

    # Set up calculation for resonance objective
    # Set which resonances to consider and create function for each considered resonance
    # consider the N lowest-order resonances over the range with minimum res_range_max and maximum res_range_max
    res_arr = jnp.full(p_max*q_max + 1, jnp.nan) # maximum possible size of array of resonances, including the zero resonance
    obj_out = omega_arr * 0
    # sigma=jnp.ones((jnp.shape(omega_arr))) # for Gaussian
    # sigma = sigma*10 # can set to vary with order of resonance if desired
    res_arr_set = 0
    # p_max and q_max are Python integers so these loops remain differentiable with jax and jit
    for p in range(0,p_max+1): # include the zero resonance
        for q in range(1,q_max+1):
            condition = jnp.logical_and(
                ~jnp.isin(p/q, res_arr),
                jnp.logical_and(p/q >= res_range_min, p/q <= res_range_max)
                )
            input_res_setup = {
                'res_arr_set': res_arr_set,
                'res_arr': res_arr,
                'p': p,
                'q': q,
            }
            res_arr_set, res_arr = jax.lax.cond( condition, true_branch_res_setup, false_branch_res_setup, operand=input_res_setup )
    res_arr_nozero = jnp.where(res_arr==0,jnp.nan,res_arr)
    res_arr = jnp.reshape(jnp.array([res_arr, res_arr_nozero*-1]),len(res_arr)*2)
    
    # for perfect equilibria, using the [0,0,0,0] value is acceptable. But for non-perfect, need to average over all non-zero wells and different field lines and filter
    num_wells = ado_shape[3]
    num_fieldlines = ado_shape[0]

    # Check and confirm nan entries
    points_0 = jnp.where(~jnp.isnan(alpha_drift_out), points[0][:][:][:][:], jnp.nan)
    points_1 = jnp.where(~jnp.isnan(alpha_drift_out), points[1][:][:][:][:], jnp.nan)

    # For non-nan entries, perform barely trapped filter
    delta_chi = jnp.abs(jnp.abs(jnp.abs(points_0) - jnp.abs(points_1)) * (iotas[0] - N*nfp)) # zeta->chi assuming delta(alpha)=0
    # ????? NEED WAY TO SET IOTA TO CORRECT SURFACE IN LINE ABOVE
    alpha_drift_out = jnp.where(delta_chi < float(2*jnp.pi),alpha_drift_out,jnp.nan) # set barely-trapped particles to nan
    
    # Average and standard deviation per-surface and pitch inverse
    alpha_drift_out = jnp.transpose(alpha_drift_out, (0,3,1,2)) # rearrange for flattening
    alpha_drift_avg = jnp.nanmean(
        jnp.reshape(alpha_drift_out,(num_wells*num_fieldlines,ado_shape[1],ado_shape[2])),
        axis=0) # flatten array to average simultaneously over wells and field lines
    alpha_drift_std = jnpnanstd(
        jnp.reshape(alpha_drift_out,(num_wells*num_fieldlines,ado_shape[1],ado_shape[2])),
        axis=0) # flatten array to find standard deviation simultaneously over wells and field lines
    
    # resize alpha_drift_avg and alpha_drift_std to make computation possible
    alpha_drift_avg = jnp.broadcast_to(alpha_drift_avg[..., None], (alpha_drift_avg.shape[0], alpha_drift_avg.shape[1], num_wells*num_fieldlines))
    alpha_drift_avg = jnp.transpose(alpha_drift_avg, (2,0,1)) # rearrange to match alpha_drift_avg above
    alpha_drift_std = jnp.broadcast_to(alpha_drift_std[..., None], (alpha_drift_std.shape[0], alpha_drift_std.shape[1], num_wells*num_fieldlines)) 
    alpha_drift_std = jnp.transpose(alpha_drift_std, (2,0,1)) # rearrange to match alpha_drift_std above

    # Ripple-trapped filter
    alpha_drift_out = jnp.reshape(alpha_drift_out,(num_wells*num_fieldlines,ado_shape[1],ado_shape[2]))
    alpha_drift_out = jnp.where(jnp.abs(alpha_drift_out - alpha_drift_avg) < 2*alpha_drift_std, alpha_drift_out, jnp.nan) # this is true if the value of interest is greater than 2 standard deviations away from the mean
    
    # Average per-surface and pitch inverse
    alpha_drift_avg_1 = jnp.nanmean(
        alpha_drift_out,
        axis=0) # average simultaneously over wells and field lines

    # Calculate tau
    vtau_out = jnp.transpose(vtau_out, (0,3,1,2))
    vtau_out = jnp.reshape(vtau_out,(num_wells*num_fieldlines,ado_shape[1],ado_shape[2]))
    vtau_out = jnp.where(~jnp.isnan(alpha_drift_out), vtau_out, jnp.nan)
    tau_arr = jnp.nanmean(vtau_out,axis=0) / jnp.sqrt(v2) # vtau->tau

    # Calculate omega (per surface per pitch angle)
    omega_arr = (tau_arr*nfp / (2*jnp.pi * (N*nfp-iotas[0]))) * (m_alpha*v2/(Z*e)) * alpha_drift_avg_1
    # ????? NEED WAY TO SET IOTA TO CORRECT SURFACE IN LINE ABOVE

    # Calculate objective function (per surface per pitch angle)
    res_broad = res_arr[None,None,:] # make 3D array with res values on axis=2
    res_broad = jnp.broadcast_to(res_broad, (omega_arr.shape[0], omega_arr.shape[1], res_arr.shape[0]))
    omega_broad = jnp.broadcast_to(omega_arr[...,None], (omega_arr.shape[0],omega_arr.shape[1],res_arr.shape[0]))

    y = omega_broad - res_broad
    condition = jnp.logical_and( # check that res_broad value is not nan and corresponding omega value is less than 0.5 away from the resonance
                ~jnp.isnan(res_broad),
                abs(y) < 0.5
                )
    # Set weights
    w = 1
    t = -1
    A = jnp.ones((jnp.shape(omega_broad))) * 100 # can set to vary with order of resonance if desired
    obj_out = jnp.where(
        condition,
        A * jnp.exp(-w * (( -((y+0.5)**2) + (y+0.5) )**t)),
        0
        ) # need to broadcast res_arr to 3D to match each res with each 2D matrix of omega_arr and then do this subtraction and jnp.where operation
    obj_out = jnp.sum(obj_out,axis=2) # outputs array with size (rho,pitch)

    # Gaussian method
    # def true_branch_res(indict):
    #     res_arr = indict['res_arr']
    #     gaus_out = indict['obj_out']
    #     res_arr = res_arr.at[indict['res_arr_set']].set(indict['p']/indict['q'])

    #     p_arr = jnp.ones(jnp.shape(indict['A'])) * indict['p']
    #     q_arr = jnp.ones(jnp.shape(indict['A'])) * indict['q']

    #     gaus_out += indict['A']*jnp.exp( -( (indict['omega_arr']-(p_arr/q_arr))**2 ) / (2*indict['sigma']**2) ) # Gaussian evaluated at distance from resonance, one element per surface
    #     return indict['res_arr_set']+1, res_arr, gaus_out
    # def false_branch_res(indict):
    #     return indict['res_arr_set'], indict['res_arr'], indict['gaus_out']

    # return obj_out, which is a 1D array (each element represents a surface)
    # data["f_tr"] = jnp.reshape(obj_out,num_pitch*grid.num_rho)
    data["f_tr"] = omega_arr
    return data




def _poloidal_drift(data, B, pitch):
    return safediv(
        data["gbdrift"] * (1 - 0.5 * pitch * B), jnp.sqrt(jnp.abs(1 - pitch * B))
    )

_bounce1D_doc = {
    "num_well": _bounce_doc["num_well"],
    "num_quad": _bounce_doc["num_quad"],
    "num_pitch": _bounce_doc["num_pitch"],
    "surf_batch_size": _bounce_doc["surf_batch_size"],
    "quad": _bounce_doc["quad"],
}

@register_compute_fun(
    name="f_tr1",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy "
    "as defined by Velasco et al. (doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "gbdrift", "fieldline length"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
@partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"]) # uncomment after debugging
def f_tr1(params, transforms, profiles, data, **kwargs):
    # debug_out = jnp.array([17])
    num_pitch = kwargs.get("num_pitch",1)
    num_well = kwargs.get("num_well", None)
    batch = kwargs.get("batch", True)
    grid = transforms["grid"].source_grid
    # N = kwargs.get("N",-1) # ?????
    N=1
    nfp = kwargs.get("nfp",jnp.nan)
    KE_frac = kwargs.get("KE_frac",0.00000001)
    p_max = kwargs.get("p_max",2)
    q_max = kwargs.get("q_max",2)
    res_range_min = kwargs.get("res_range_min",0)
    res_range_max = kwargs.get("res_range_max",4)
    bt_filter_flag = kwargs.get("bt_filter_flag",True)
    rt_filter_flag = kwargs.get("rt_filter_flag",True)

    # noqa: unused dependency
    # num_well = kwargs.get("num_well", None)
    # num_pitch = kwargs.get("num_pitch", 64)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    quad = (
        kwargs["quad"]
        if "quad" in kwargs
        else get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    )

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

        return _alpha_drift, points, v_tau, count_nz


    grid = transforms["grid"].source_grid
    alpha_drift_out, points, vtau_out, count_nz = (
        _compute(
            alpha_drift, 
            {"gbdrift": data["gbdrift"]},
            data,
            grid,
            num_pitch,
            surf_batch_size,
        )
    ) # [rho][alpha][pitch][wells]
    # count_nz will be 3D [rho][alpha][pitch]
    assert alpha_drift_out.shape[:-1] == (grid.num_rho,grid.num_alpha,num_pitch) # don't know well number yet, default is None, and assert is useable in optimization
    ado_shape = jnp.shape(alpha_drift_out)
    
    debug_out = alpha_drift_out[0,0,0,0]

    ''# specify energy
    m_alpha = 6.6446573450*10**(-27) # kg, mass of alpha particle
    e = 1.602*10**(-19) # C
    Z=2 # fully ionized alpha particle
    KE = KE_frac * 5.6076*10**(-13) # J, 3.5 MeV if KE_frac=1
    v2 = 2*KE/m_alpha # m/s

    iotas = data['iota'] # ????? need to align to surfaces better
    tau_arr = jnp.zeros((ado_shape[0],ado_shape[2])) # axis=0 is rhos, axis=2 is pitch inv
    omega_arr = jnp.zeros((ado_shape[0],ado_shape[2])) # axis=0 is rhos, axis=2 is pitch invs


    # Set up calculation for resonance objective
    # Set which resonances to consider and create function for each considered resonance
    # consider the N lowest-order resonances over the range with minimum res_range_max and maximum res_range_max
    res_arr1 = jnp.full(p_max*q_max + 1, 0) # maximum possible size of array of resonances, including the zero resonance
    obj_out = omega_arr * 0
    # sigma=jnp.ones((jnp.shape(omega_arr))) # for Gaussian
    # sigma = sigma*10 # can set to vary with order of resonance if desired

    # def jnpmeanzeros(f,count_nz,axis=0):
    #     return jnp.sum(f,axis=axis) / (f.shape[axis] - count_nz)

    '''def false_branch_res_setup(indict): # do nothing
        return indict['res_arr_set'], indict['res_arr']
    def true_branch_res_setup(indict): # add to res_arr
        res_arr = indict['res_arr']
        res_arr = res_arr.at[indict['res_arr_set']].set(indict['p']/indict['q'])
        return indict['res_arr_set']+1, res_arr'''
    
    def false_branch_res_setup(res_arr_set,res_arr,p,q): # do nothing
        return res_arr_set, res_arr
    def true_branch_res_setup(res_arr_set,res_arr,p,q): # add to res_arr
        res_arr = res_arr.at[res_arr_set].set(p/q)
        return res_arr_set+1, res_arr

    res_arr = jnp.full(p_max*q_max + 1, 0.0) # maximum possible size of array of resonances, including the zero resonance
    res_arr_set = 0
    # p_max and q_max are Python integers so these loops remain differentiable with jax and jit
    for p in range(0,p_max+1): # include the zero resonance
        for q in range(1,q_max+1):
            condition = jnp.logical_and(
                ~jnp.isin(p/q, res_arr),
                jnp.logical_and(p/q >= res_range_min, p/q <= res_range_max)
                )
            '''input_res_setup = {
                'res_arr_set': res_arr_set,
                'res_arr': res_arr,
                'p': p,
                'q': q,
            }'''
            '''res_arr_set, res_arr = jax.lax.cond( condition, true_branch_res_setup, false_branch_res_setup, operand=input_res_setup )'''
            res_arr_set, res_arr = jax.lax.cond( condition, true_branch_res_setup, false_branch_res_setup, res_arr_set,res_arr,p,q )
    # make size of res_arr such that it is the size of how many unique resonances there are
    '''def true_branch_res(indict):
        return indict['res_arr'].at[indict['count']].set(indict['res']), indict['count']+1
    def false_branch_res(indict):
        return indict['res_arr'], indict['count']
    res_arr1 = jnp.full(res_arr_set+1, 0.0)
    count=0
    for res in res_arr:
        res_arr1, count = jax.lax.cond( ~jnp.isin(res, res_arr1), true_branch_res, false_branch_res, operand={'res_arr': res_arr1,'res': res, 'count': count} )
    '''

    def true_branch_res(res_arr,count,res):   
        return res_arr.at[count].set(res), count+1
    def false_branch_res(res_arr,count,res):
        return res_arr, count
    res_arr1 = jnp.full(res_arr_set+1, 0.0)
    count=0
    for res in res_arr:
        res_arr1, count = jax.lax.cond( ~jnp.isin(res, res_arr1), true_branch_res, false_branch_res, res_arr, count, res )
 

    # for perfect equilibria, using the [0,0,0,0] value is acceptable. But for non-perfect, need to average over all non-zero wells and different field lines and filter
    num_wells = ado_shape[3]
    num_fieldlines = ado_shape[1]

    points_0 = points[0][:][:][:][:]
    points_1 = points[1][:][:][:][:]

    # For non-nan entries, perform barely trapped filter
    delta_chi = jnp.abs(jnp.abs(jnp.abs(points_0) - jnp.abs(points_1)) * (iotas[0] - N*nfp)) # zeta->chi assuming delta(alpha)=0
    # ????? NEED WAY TO SET IOTA TO CORRECT SURFACE IN LINE ABOVE
    alpha_drift_out = jnp.where(delta_chi < float(2*jnp.pi),alpha_drift_out,0.0) # set barely-trapped particles to 0

    def jnpmean_nz(x,axis=0):
        mask = x!=0
        count = jnp.sum(mask,axis) # how many wells that are not 0
        return jnp.sum(x,axis=axis) / count

    def jnpstd_nz(x,axis=0): # compute population standard deviation of an array while ignoring "nan" elements in JAX numpy
        # x is an array with size: (num_wells*num_fieldlines,num_rho,num_pitch)
        xbar = jnpmean_nz(x,axis=axis)
        xbar = jnp.broadcast_to(xbar[..., None], (xbar.shape[0], xbar.shape[1], num_wells*num_fieldlines))
        xbar = jnp.transpose(xbar, (2,0,1)) # rearrange to match initial x
        sq_diff = (x - xbar) ** 2
        return jnp.sqrt(jnpmean_nz(sq_diff,axis=0)) # returning an array with size: (num_rho,num_pitch)

    # Average and standard deviation per-surface and pitch inverse
    alpha_drift_out = jnp.transpose(alpha_drift_out, (1,3,0,2)) # rearrange for flattening
    alpha_drift_out = jnp.reshape(alpha_drift_out,(num_wells*num_fieldlines,ado_shape[0],ado_shape[2])) # flatten array to average simultaneously over wells and field lines
    alpha_drift_avg = jnpmean_nz(alpha_drift_out,axis=0) # does not include zero wells in averaging, should be size [rho][pitch] / size [rho][pitch]
    alpha_drift_std = jnpstd_nz(alpha_drift_out,axis=0)
    
    # resize alpha_drift_avg and alpha_drift_std to make computation possible
    alpha_drift_avg = jnp.broadcast_to(alpha_drift_avg[..., None], (ado_shape[0], ado_shape[2], num_wells*num_fieldlines))
    alpha_drift_avg = jnp.transpose(alpha_drift_avg, (2,0,1)) # rearrange to match alpha_drift_avg above
    alpha_drift_std = jnp.broadcast_to(alpha_drift_std[..., None], (ado_shape[0], ado_shape[2], num_wells*num_fieldlines)) 
    alpha_drift_std = jnp.transpose(alpha_drift_std, (2,0,1)) # rearrange to match alpha_drift_std above

    # Ripple-trapped filter
    alpha_drift_out = jnp.where(jnp.abs(alpha_drift_out - alpha_drift_avg) < 2*alpha_drift_std, alpha_drift_out, 0.0) # this is true if the value of interest is greater than 2 standard deviations away from the mean
    
    # Average per-surface and pitch inverse
    alpha_drift_avg_1 = jnpmean_nz(alpha_drift_out,axis=0) # average simultaneously over wells and field lines

    # Calculate tau
    vtau_out = jnp.transpose(vtau_out, (1,3,0,2))
    vtau_out = jnp.reshape(vtau_out,(num_wells*num_fieldlines,ado_shape[0],ado_shape[2]))
    tau_arr = jnpmean_nz(vtau_out,axis=0) / jnp.sqrt(v2) # vtau->tau

    # Calculate omega (per surface per pitch angle)
    omega_arr = (tau_arr*nfp / (2*jnp.pi * (N*nfp-iotas[0]))) * (m_alpha*v2/(Z*e)) * alpha_drift_avg_1 # shape [rho][pitch]
    # ????? NEED WAY TO SET IOTA TO CORRECT SURFACE IN LINE ABOVE


    # Calculate objective function (per surface per pitch angle)
    res_broad = res_arr[None,None,:] # make 3D array with res values on axis=2
    res_broad = jnp.broadcast_to(res_broad, (omega_arr.shape[0], omega_arr.shape[1], res_arr.shape[0]))
    omega_broad = jnp.broadcast_to(omega_arr[...,None], (omega_arr.shape[0],omega_arr.shape[1],res_arr.shape[0]))

    y = omega_broad - res_broad
    condition = abs(y) < 0.5 # check that corresponding omega value is less than 0.5 away from the resonance
    # Set weights
    w = 1
    t = -1
    A = jnp.ones((jnp.shape(omega_broad))) * 100 # can set to vary with order of resonance if desired
    obj_out = jnp.where(
        condition,
        A * jnp.exp(-w * (( -((y+0.5)**2) + (y+0.5) )**t)),
        0
        ) # need to broadcast res_arr to 3D to match each res with each 2D matrix of omega_arr and then do this subtraction and jnp.where operation
    obj_out = jnp.sum(obj_out,axis=2) # outputs array with size (rho,pitch)

    # Gaussian method
    # def true_branch_res(indict):
    #     res_arr = indict['res_arr']
    #     gaus_out = indict['obj_out']
    #     res_arr = res_arr.at[indict['res_arr_set']].set(indict['p']/indict['q'])

    #     p_arr = jnp.ones(jnp.shape(indict['A'])) * indict['p']
    #     q_arr = jnp.ones(jnp.shape(indict['A'])) * indict['q']

    #     gaus_out += indict['A']*jnp.exp( -( (indict['omega_arr']-(p_arr/q_arr))**2 ) / (2*indict['sigma']**2) ) # Gaussian evaluated at distance from resonance, one element per surface
    #     return indict['res_arr_set']+1, res_arr, gaus_out
    # def false_branch_res(indict):
    #     return indict['res_arr_set'], indict['res_arr'], indict['gaus_out']

    # return obj_out, which is a 1D array (each element represents a surface and pitch combination)
    data["f_tr1"] = jnp.reshape(obj_out,num_pitch*grid.num_rho)
    # data["f_tr1"] = debug_out
    return data