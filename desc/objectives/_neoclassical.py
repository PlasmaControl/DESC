"""Objectives for neoclassical transport."""

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.integrals._interp_utils import check_nufft
from desc.integrals.bounce_integral import Options

from .objective_funs import _Objective, collect_docs, doc_bounce
from .utils import errorif


class EffectiveRipple(_Objective):
    """Proxy for neoclassical transport in the banana regime.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. In the banana regime, neoclassical (thermal)
    transport from ripple wells can become the dominant transport channel.
    The effective ripple (ε) proxy estimates the neoclassical transport
    coefficients in the banana regime. To ensure low neoclassical transport,
    a stellarator is typically optimized so that ε < 10⁻².

    The objective is presented in [1]_, [2]_, and the computation is presented in [2]_.

    References
    ----------
    .. [1] V. V. Nemov, S. V. Kasilov, W. Kernbichler, and M. F. Heyn,
           "Evaluation of 1/ν neoclassical transport in stellarators,"
           Phys. Plasmas 6, 4622-4632 (1999).
           https://doi.org/10.1063/1.873749.
    .. [2] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
           bounce-averaging algorithm and its applications,"
           J. Plasma Physics. https://doi:10.1017/S0022377826101652.

    """

    __doc__ = (
        __doc__.rstrip()
        + doc_bounce
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
    _print_value_fmt = "Effective ripple ε: "
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
        name="Effective ripple",
        grid=None,
        X=32,
        Y=32,
        Y_B=None,
        alpha=None,
        num_field_periods=20,
        num_well=None,
        num_quad=32,
        num_pitch=51,
        pitch_batch_size=None,
        surf_batch_size=1,
        nufft_eps=1e-6,
        spline=True,
    ):
        errorif(
            deriv_mode == "fwd",
            ValueError,
            "Reverse mode should be used for the objective: EffectiveRipple.",
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
        Options._build_objective(self, "effective ripple", eta=1)
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the effective ripple.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``. (Deprecated)

        Returns
        -------
        epsilon : ndarray
            Effective ripple as a function of the flux surface label.

        """
        return Options._compute_objective(self, params, constants, "effective ripple")
