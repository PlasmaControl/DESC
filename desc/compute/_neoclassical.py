"""Compute functions for neoclassical transport."""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D
from ..integrals.quad_utils import chebgauss2
from ..utils import safediv
from .data_index import register_compute_fun

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
        * (jnp.pi / (8 * 2**0.5))
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
