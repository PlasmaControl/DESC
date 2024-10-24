"""Compute functions for neoclassical transport.

Performance will improve significantly by resolving these GitHub issues.
  * ``1154`` Improve coordinate mapping performance
  * ``1294`` Nonuniform fast transforms
  * ``1303`` Patch for differentiable code with dynamic shapes
  * ``1206`` Upsample data above midplane to full grid assuming stellarator symmetry
  * ``1034`` Optimizers/objectives with auxiliary output
"""

from functools import partial

from orthax.legendre import leggauss

from desc.backend import imap, jit, jnp

from ..integrals._bounce_utils import interp_fft_to_argmin
from ..integrals._interp_utils import polyder_vec
from ..integrals._quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from ..integrals.bounce_integral import Bounce2D
from ..utils import cross, dot, safediv
from .data_index import register_compute_fun

_bounce_doc = {
    "theta": """jnp.ndarray :
        DESC coordinates θ sourced from the Clebsch coordinates
        ``FourierChebyshevSeries.nodes(X,Y,rho,domain=(0,2*jnp.pi))``.
        Use the ``Bounce2D.compute_theta`` method to obtain this.
        """,
    "Y_B": """int :
        Desired resolution for |B| along field lines to compute bounce points.
        Default is double the resolution of ``theta``.
        """,
    "num_transit": """int :
        Number of toroidal transits to follow field line.
        For axisymmetric devices, one poloidal transit is sufficient. Otherwise,
        assuming the surface is not near rational, more transits will
        approximate surface averages better, with diminishing returns.
        """,
    "num_quad": """int :
        Resolution for quadrature of bounce integrals.
        Default is 32. This parameter is ignored if given ``quad``.
        """,
    "num_pitch": "int : Resolution for quadrature over velocity coordinate.",
    "num_well": """int :
        Maximum number of wells to detect for each pitch and field line.
        Giving ``None`` will detect all wells but due to current limitations in
        JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``,``B`` are the poloidal and
        toroidal Fourier resolution of |B|, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
        The ``check_points`` or ``plot`` methods in ``desc.integrals.Bounce2D``
        are useful to select a reasonable value.
        """,
    "batch_size": """int :
        Number of pitch values with which to compute simultaneously.
        If given ``None``, then ``batch_size`` defaults to ``num_pitch``.
        """,
    "spline": "bool : Whether to use cubic splines to compute bounce points.",
    "fieldline_quad": """tuple[jnp.ndarray] :
        Used to compute the proper length of the field line ∫ dℓ / |B|.
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
}


def _compute(fun, fun_data, data, theta, grid, num_pitch):
    """Compute ``fun`` for each ρ value iteratively to reduce memory usage.

    Parameters
    ----------
    fun : callable
        Function to compute.
    fun_data : dict[str, jnp.ndarray]
        Data to provide to ``fun``.
        Names in ``Bounce2D.required_names`` will be overridden.
        Reshaped automatically.
    data : dict[str, jnp.ndarray]
        DESC data dict.
    theta : jnp.ndarray
        Shape (num rho, X, Y).
        DESC coordinates θ sourced from the Clebsch coordinates
        ``FourierChebyshevSeries.nodes(X,Y,rho,domain=(0,2*jnp.pi))``.

    """
    for name in Bounce2D.required_names:
        fun_data[name] = data[name]
    fun_data = dict(
        zip(fun_data.keys(), Bounce2D.reshape_data(grid, *fun_data.values()))
    )
    # These already have expected shape with num rho along first axis.
    fun_data["pitch_inv"], fun_data["pitch_inv weight"] = Bounce2D.get_pitch_inv_quad(
        grid.compress(data["min_tz |B|"]),
        grid.compress(data["max_tz |B|"]),
        num_pitch,
    )
    fun_data["iota"] = grid.compress(data["iota"])
    fun_data["theta"] = theta
    return grid.expand(imap(fun, fun_data))


def _foreach_pitch(fun, pitch_inv, batch_size):
    """Compute ``fun`` for pitch values iteratively to reduce memory usage.

    Parameters
    ----------
    fun : callable
        Function to compute.
    pitch_inv : jnp.ndarray
        Shape (num_pitch, ).
        1/λ values to compute the bounce integrals.
    batch_size : int or None
        Number of pitch values with which to compute simultaneously.
        If given ``None``, then computes everything simultaneously.

    """
    # FIXME: Make this work with older JAX versions.
    #  We don't need to rely on JAX to iteratively vectorize since
    #  ``fun``` natively supports vectorization.
    return (
        fun(pitch_inv)
        if (batch_size is None or batch_size >= pitch_inv.size)
        else imap(fun, pitch_inv, batch_size=batch_size).squeeze(axis=-1)
    )


@register_compute_fun(
    name="effective ripple",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude. "
    "Uses numerical methods of the Bounce2D class.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["effective ripple 3/2"],
)
def _effective_ripple(params, transforms, profiles, data, **kwargs):
    data["effective ripple"] = data["effective ripple 3/2"] ** (2 / 3)
    return data


def _dH(grad_rho_norm_kappa_g, B, pitch, zeta):
    """Integrand of Nemov eq. 30 with |∂ψ/∂ρ| (λB₀)¹ᐧ⁵ removed."""
    return (
        jnp.sqrt(jnp.abs(1 - pitch * B))
        * (4 / (pitch * B) - 1)
        * grad_rho_norm_kappa_g
        / B
    )


def _dI(B, pitch, zeta):
    """Integrand of Nemov eq. 31."""
    return jnp.sqrt(jnp.abs(1 - pitch * B)) / B


@register_compute_fun(
    name="effective ripple 3/2",
    label=(
        # ε¹ᐧ⁵ = π/(8√2) R₀²〈|∇ψ|〉⁻² B₀⁻¹ ∫dλ λ⁻² 〈 ∑ⱼ Hⱼ²/Iⱼ 〉
        "\\epsilon_{\\mathrm{eff}}^{3/2} = \\frac{\\pi}{8 \\sqrt{2}} "
        "R_0^2 \\langle \\vert\\nabla \\psi\\vert \\rangle^{-2} "
        "B_0^{-1} \\int d\\lambda \\lambda^{-2} "
        "\\langle \\sum_j H_j^2 / I_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude to 3/2 power.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "kappa_g", "R0", "|grad(rho)|", "<|grad(rho)|>"]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft": True},
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_quad",
        "num_pitch",
        "num_well",
        "batch_size",
        "spline",
    ],
)
def _epsilon_32(params, transforms, profiles, data, **kwargs):
    """https://doi.org/10.1063/1.873749.

    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
    """
    # noqa: unused dependency
    grid = transforms["grid"]

    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 50)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    batch_size = kwargs.get("batch_size", None)
    spline = kwargs.get("spline", True)
    if "fieldline_quad" in kwargs:
        fieldline_quad = kwargs["fieldline_quad"]
    else:
        fieldline_quad = leggauss(Y_B // 2)
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = chebgauss2(kwargs.get("num_quad", 32))

    def eps_32(data):
        """(∂ψ/∂ρ)⁻² B₀⁻² ∫ dλ λ⁻² ∑ⱼ Hⱼ²/Iⱼ."""
        # B₀ has units of λ⁻¹.
        # Nemov's ∑ⱼ Hⱼ²/Iⱼ = (∂ψ/∂ρ)² (λB₀)³ ``(H**2 / I).sum(axis=-1)``.
        # (λB₀)³ d(λB₀)⁻¹ = B₀² λ³ d(λ⁻¹) = -B₀² λ dλ.
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            num_transit,
            quad=quad,
            automorphism=None,
            is_reshaped=True,
            spline=spline,
        )
        data["|grad(rho)|*kappa_g"] = Bounce2D.fourier(data["|grad(rho)|*kappa_g"])

        def fun(pitch_inv):
            points = bounce.points(pitch_inv, num_well=num_well)
            H = bounce.integrate(
                _dH,
                pitch_inv,
                data["|grad(rho)|*kappa_g"],
                points=points,
                is_fourier=True,
            )
            I = bounce.integrate(_dI, pitch_inv, points=points)
            return safediv(H**2, I).sum(axis=-1)

        return jnp.sum(
            _foreach_pitch(fun, data["pitch_inv"], batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 3,
            axis=-1,
        ) / bounce.compute_fieldline_length(fieldline_quad)

    B0 = data["max_tz |B|"]
    data["effective ripple 3/2"] = (
        _compute(
            eps_32,
            fun_data={"|grad(rho)|*kappa_g": data["|grad(rho)|"] * data["kappa_g"]},
            data=data,
            theta=theta,
            grid=grid,
            num_pitch=num_pitch,
        )
        * (B0 * data["R0"] / data["<|grad(rho)|>"]) ** 2
        * jnp.pi
        / (8 * 2**0.5)
    )
    return data


# We rewrite equivalents of Nemov et al.'s expressions (21, 22) to resolve
# the indeterminate form of the limit and using single-valued maps of a
# physical coordinates. This avoids the computational issues of multivalued
# maps.
# The derivative (∂/∂ψ)|ϑ,ϕ belongs to flux coordinates which satisfy
# α = ϑ − χ(ψ) ϕ where α is the poloidal label of ψ,α Clebsch coordinates.
# Choosing χ = ι implies ϑ, ϕ are PEST angles.
# ∂G/∂((λB₀)⁻¹) =     λ²B₀  ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) ∂|B|/∂ψ / |B|
# ∂V/∂((λB₀)⁻¹) = 3/2 λ²B₀  ∫ dℓ √(1 − λ|B|) R / |B|
# ∂g/∂((λB₀)⁻¹) =     λ²B₀² ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
# K ≝ R dψ/dρ
# tan(π/2 γ_c) =
#              ∫ dℓ (1 − λ|B|/2) / √(1 − λ|B|) |∇ψ| κ_g / |B|
#              ----------------------------------------------
# (|∇ρ| ‖e_α|ρ,ϕ‖)ᵢ ∫ dℓ [ (1 − λ|B|/2)/√(1 − λ|B|) ∂|B|/∂ρ + √(1 − λ|B|) K ] / |B|


def _v_tau(B, pitch, zeta):
    # Note v τ = 4λ⁻²B₀⁻¹ ∂I/∂((λB₀)⁻¹) where v is the particle velocity,
    # τ is the bounce time, and I is defined in Nemov eq. 36.
    return safediv(2.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


def _f1(grad_psi_norm_kappa_g, B, pitch, zeta):
    return (
        safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B)))
        * grad_psi_norm_kappa_g
        / B
    )


def _f2(B_r, B, pitch, zeta):
    return safediv(1 - 0.5 * pitch * B, jnp.sqrt(jnp.abs(1 - pitch * B))) * B_r / B


def _f3(K, B, pitch, zeta):
    return jnp.sqrt(jnp.abs(1 - pitch * B)) * K / B


@register_compute_fun(
    name="Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Energetic ion confinement proxy, Nemov et al.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^phi",
        "B^phi_r|v,p",
        "|B|_r|v,p",
        "b",
        "grad(phi)",
        "grad(psi)",
        "|grad(psi)|",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "kappa_g",
        "iota_r",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft": True},
    **_bounce_doc,
    quad2="Same as ``quad`` for the weak singular integrals in particular.",
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_quad",
        "num_pitch",
        "num_well",
        "batch_size",
        "spline",
    ],
)
def _Gamma_c(params, transforms, profiles, data, **kwargs):
    """Energetic ion confinement proxy as defined by Nemov et al.

    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61.

    The radial electric field has a negligible effect on alpha particle confinement,
    so it is assumed to be zero.
    """
    # noqa: unused dependency
    grid = transforms["grid"]

    theta = kwargs["theta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    num_transit = kwargs.get("num_transit", 20)
    num_pitch = kwargs.get("num_pitch", 64)
    num_well = kwargs.get("num_well", Y_B * num_transit)
    batch_size = kwargs.get("batch_size", None)
    spline = kwargs.get("spline", True)
    if "fieldline_quad" in kwargs:
        fieldline_quad = kwargs["fieldline_quad"]
    else:
        fieldline_quad = leggauss(Y_B // 2)
    if "quad" in kwargs:
        quad = kwargs["quad"]
    else:
        quad = get_quadrature(
            leggauss(kwargs.get("num_quad", 32)),
            (automorphism_sin, grad_automorphism_sin),
        )
    quad2 = kwargs["quad2"] if "quad2" in kwargs else chebgauss2(quad[0].size)

    def Gamma_c(data):
        """∫ dλ ∑ⱼ [v τ γ_c²]ⱼ."""
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            num_transit,
            quad=quad,
            automorphism=None,
            is_reshaped=True,
            spline=spline,
        )
        data["|grad(psi)|*kappa_g"] = Bounce2D.fourier(data["|grad(psi)|*kappa_g"])
        data["|B|_r|v,p"] = Bounce2D.fourier(data["|B|_r|v,p"])
        data["K"] = Bounce2D.fourier(data["K"])
        data["|grad(rho)|*|e_alpha|r,p|"] = Bounce2D.fourier(
            data["|grad(rho)|*|e_alpha|r,p|"]
        )

        def fun(pitch_inv):
            points = bounce.points(pitch_inv, num_well=num_well)
            v_tau = bounce.integrate(_v_tau, pitch_inv, points=points)
            gamma_c = jnp.arctan(
                safediv(
                    bounce.integrate(
                        _f1,
                        pitch_inv,
                        data["|grad(psi)|*kappa_g"],
                        points=points,
                        is_fourier=True,
                    ),
                    (
                        bounce.integrate(
                            _f2,
                            pitch_inv,
                            data["|B|_r|v,p"],
                            points=points,
                            is_fourier=True,
                        )
                        + bounce.integrate(
                            _f3,
                            pitch_inv,
                            data["K"],
                            points=points,
                            quad=quad2,
                            is_fourier=True,
                        )
                    )
                    * interp_fft_to_argmin(
                        grid.NFP,
                        bounce._c["T(z)"],
                        data["|grad(rho)|*|e_alpha|r,p|"],
                        points,
                        bounce._c["knots"],
                        bounce._c["B(z)"],
                        polyder_vec(bounce._c["B(z)"]),
                        is_fourier=True,
                        M=grid.num_theta,
                        N=grid.num_zeta,
                    ),
                )
            )
            return jnp.sum(v_tau * gamma_c**2, axis=-1)

        return jnp.sum(
            _foreach_pitch(fun, data["pitch_inv"], batch_size)
            * data["pitch_inv weight"]
            / data["pitch_inv"] ** 2,
            axis=-1,
        ) / bounce.compute_fieldline_length(fieldline_quad)

    # It is assumed the grid is sufficiently dense to reconstruct |B|,
    # so anything smoother than |B| may be captured accurately as a single
    # Fourier series rather than transforming each component.
    # Last term in K behaves as ∂log(|B|²/B^ϕ)/∂ρ |B| if one ignores the issue
    # of a log argument with units. Smoothness determined by positive lower bound
    # of log argument, and hence behaves as ∂log(|B|)/∂ρ |B| = ∂|B|/∂ρ.
    data["Gamma_c"] = _compute(
        Gamma_c,
        fun_data={
            "|grad(psi)|*kappa_g": data["|grad(psi)|"] * data["kappa_g"],
            "|grad(rho)|*|e_alpha|r,p|": data["|grad(rho)|"] * data["|e_alpha|r,p|"],
            "|B|_r|v,p": data["|B|_r|v,p"],
            "K": data["iota_r"]
            * dot(cross(data["grad(psi)"], data["b"]), data["grad(phi)"])
            - (
                2 * data["|B|_r|v,p"]
                - data["|B|"] * data["B^phi_r|v,p"] / data["B^phi"]
            ),
        },
        data=data,
        theta=theta,
        grid=grid,
        num_pitch=num_pitch,
    ) / (2**1.5 * jnp.pi)
    return data
