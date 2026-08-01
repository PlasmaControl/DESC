"""Objectives for turbulence proxies."""

import jax
from packaging import version

from desc.backend import jnp
from desc.compute._turbulence import _energy_quad
from desc.compute.utils import _compute as compute_fun
from desc.integrals._interp_utils import check_nufft
from desc.integrals.bounce_integral import Options
from desc.utils import errorif

from .objective_funs import _Objective, collect_docs, doc_bounce


class AvailableEnergy(_Objective):
    """Available energy of trapped electrons.

    The available-energy metric estimates the dimensionless free energy available
    to trapped electrons from density and temperature profile gradients.

    The objective is presented in [1]_, and the computation is presented in [2]_.
    This objective computes the particle drifts using a flux tube model;
    and therefore, has a meaningless ergodic limit. In 3-D, an optimization should
    be evaluated by measuring improvement over a fixed number of field-period
    transits. In axisymmetry, use one poloidal transit between global maxima of |B|.

    Notes
    -----
    Let ρ★ = ρₗ/a and r = aρ. Equations (2.47) and (2.49) of [1]_
    define Δr_A = Cᵣρₗ and factor ρ★² out of the available energy.
    Consequently, the widths used internally are
    Δψ_A/ρ★ = Cᵣ ∂ψ/∂ρ and Δα_A/ρ★ = Cₛ/ρ. The parameters
    ``radial_scale`` and ``binormal_scale`` are Cᵣ and Cₛ, not the normalized
    coordinate width Δρ_A = Cᵣρ★.

    DESC uses ψ = Ψρ²/(2π) = ψₑρ², so ∂ψ/∂ρ = 2ψₑρ. Thus,
    Δψ_A/ρ★ already contains the factor of ρ in Eq. (4.7) of [3]_.

    Before energy normalization, the bounce-integral ratios satisfy
    G_ω/G = qω/(mv²). They are converted to the qω/ε₀ convention, with
    ε₀ = mv²/2, by the AE-specific drift integrands before bounce integration,
    as required by Eqs. (2.35) and (2.38) of [1]_.

    Every complete well in the traced interval is summed. The registered compute
    function does not infer a special axisymmetric domain. For k complete
    axisymmetric poloidal transits between global maxima of |B|, choose ``alpha``
    and ``num_field_periods`` accordingly and pass
    ``fieldline_normalization=|ι|/k``.

    The result uses the 3nT/2 thermal-energy normalization in Eqs. (2.44) and
    (2.49) of [1]_. It is therefore ⅔ of an otherwise identical convention
    normalized by nT, such as Eq. (4.2) of [3]_.

    References
    ----------
    .. [1] R. J. J. Mackenbach et al., J. Plasma Phys. 89, 905890513 (2023).
    .. [2] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
           bounce-averaging algorithm and its applications,"
           J. Plasma Physics. 2026;92(3):E72. https://arxiv.org/pdf/2412.01724.
    .. [3] E. Rodríguez and R. J. J. Mackenbach, "Trapped-particle precession and
           modes in quasisymmetric stellarators and tokamaks: a near-axis
           perspective," J. Plasma Phys. 89, 905890521 (2023).

    Warnings
    --------
    By default, an adaptive quadrature in the energy integral will be used.
    The current implementation to compute the derivative relevant for optimisation
    of the adaptive quadrature can be made significantly more effecient.
    See https://github.com/f0uriest/quadax/issues/111 if you would like to contribute.
    For faster performance, albeit at the expense of accuracy, set ``quad_atol=0.0`` to
    use a generalized Laguerre quadrature with a resolution of 32 points.

    """

    __doc__ = (
        __doc__.rstrip()
        + doc_bounce
        + """
    radial_scale : float
        Radial correlation-length coefficient Cᵣ in Δr_A = Cᵣρₗ.
        After factoring out ρ★² in the definition of Â, this scales both
        Δψ_A/ρ★ = Cᵣ ∂ψ/∂ρ and the radial profile gradients.
        Default is 1.0.
    binormal_scale : float
        Binormal correlation-length coefficient Cₛ in Δs_A = Cₛρₗ.
        Default is 1.0.
    fieldline_normalization : float or ndarray, optional
        Field-line factor 𝒩ₗ = Vψ/(2π𝓛), where 𝓛 is the sum of ∫dℓ/B over
        the retained complete field-line domain. The default
        ``NFP / num_field_periods`` is the long-field-line estimate. For k
        complete axisymmetric poloidal transits, pass |ι|/k.
    quad_atol : float
        Absolute tolerance for adaptive energy quadrature.
        If ``quad_atol=0.0``, then this is interpreted as a flag to use a fixed
        quadrature, which is faster, but less accurate.
        Default is 1e-6.
    quad_rtol : float
        Relative tolerance for adaptive energy quadrature.
        Default is 1e-6.
        """.rstrip()
        + collect_docs(
            target_default="``target=0``.",
            bounds_default="``target=0``.",
            normalize_detail=" Note: Has no effect for this objective.",
            normalize_target_detail=" Note: Has no effect for this objective.",
            jac_chunk_size=False,
        )
    )

    _static_attrs = _Objective._static_attrs + ["_hyperparam"]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Available energy: "
    _compute_fun = staticmethod(compute_fun)

    def __init__(
        self,
        eq,
        *,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="Available energy",
        grid=None,
        X=32,
        Y=32,
        Y_B=None,
        alpha=None,
        num_field_periods=20,
        num_well=None,
        num_quad=32,
        num_pitch=65,
        pitch_batch_size=None,
        surf_batch_size=1,
        nufft_eps=1e-7,
        spline=True,
        radial_scale=1.0,
        binormal_scale=1.0,
        fieldline_normalization=None,
        quad_atol=1e-6,
        quad_rtol=1e-6,
    ):
        errorif(
            deriv_mode == "fwd"
            and (version.parse(jax.__version__) < version.parse("0.11.0")),
            ValueError,
            "JAX version >= 0.11.0 required for fwd deriv mode for objective: "
            "AvailableEnergy.",
        )
        nufft_eps = check_nufft(nufft_eps)

        if target is None and bounds is None:
            target = 0.0

        self._grid = grid
        if alpha is None:
            alpha = jnp.zeros(1)
        self._constants = {"quad_weights": 1.0, "alpha": alpha}
        self._hyperparam = {
            "X": X,
            "Y": Y,
            "Y_B": Y_B,
            "num_field_periods": num_field_periods,
            "num_well": num_well,
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "pitch_batch_size": pitch_batch_size,
            "surf_batch_size": surf_batch_size,
            "nufft_eps": nufft_eps,
            "spline": spline,
            "radial_scale": radial_scale,
            "binormal_scale": binormal_scale,
            "fieldline_normalization": fieldline_normalization,
            "quad_atol": float(quad_atol),
            "quad_rtol": float(quad_rtol),
        }

        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=None,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        Options._build_objective(self, "available energy", eta=-1)
        if not self._hyperparam["quad_atol"]:
            self._constants["energy_quad"] = _energy_quad(32)
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the available energy.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``.

        Returns
        -------
        available_energy : ndarray
            Available energy as a function of the flux surface label.

        """
        return Options._compute_objective(self, params, constants, "available energy")
