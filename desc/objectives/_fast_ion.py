"""Objectives for fast ion confinement."""

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.integrals._interp_utils import check_nufft
from desc.integrals.bounce_integral import Options

from .objective_funs import _Objective, collect_docs, doc_bounce
from .utils import errorif


class GammaC(_Objective):
    """Proxy for fast ion confinement.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. The energetic particle confinement
    metric γ_c quantifies whether the contours of the second adiabatic invariant
    close on the flux surfaces. In the limit where the poloidal drift velocity
    majorizes the radial drift velocity, the contours lie parallel to flux
    surfaces. The optimization metric Γ_c averages γ_c² over the distribution
    of trapped particles on each flux surface.

    The radial electric field has a negligible effect, since fast particles
    have high energy with collisionless orbits, so it is assumed to be zero.

    References
    ----------
    [1] Poloidal motion of trapped particle orbits in real-space coordinates.
        V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
        Phys. Plasmas 1 May 2008; 15 (5): 052501.
        https://doi.org/10.1063/1.2912456.
        Equation 61.

    [2] A model for the fast evaluation of prompt losses of energetic ions in
        stellarators. Equation 16.
        J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
        https://doi.org/10.1088/1741-4326/ac2994.

    [3] Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
        and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    """

    __doc__ = (
        __doc__.rstrip()
        + doc_bounce
        + """
    Nemov : bool
        Whether to use the Γ_c as defined by Nemov et al. or Velasco et al.
        Default is Nemov. Set to ``False`` to use Velasco's.

        Nemov's Γ_c converges to a finite nonzero value in the infinity limit
        of the number of toroidal transits. Velasco's expression has a secular
        term that drives the result to zero as the number of toroidal transits
        increases if the secular term is not averaged out from the singular
        integrals. At finite resolution, an optimization using Velasco's metric
        may need to be evaluated by measuring decrease in Γ_c at a fixed number
        of toroidal transits.
        """.rstrip()
        + collect_docs(
            target_default="``target=0``.",
            bounds_default="``target=0``.",
            normalize_detail=" Note: Has no effect for this objective.",
            normalize_target_detail=" Note: Has no effect for this objective.",
            jac_chunk_size=False,
        )
    )

    _static_attrs = _Objective._static_attrs + ["_hyperparam", "_key"]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Γ_c: "
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
        name="Gamma_c",
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
        Nemov=True,
    ):
        errorif(
            deriv_mode == "fwd",
            ValueError,
            "Reverse mode should be used for the objective: GammaC.",
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
        self._key = "Gamma_c" if Nemov else "Gamma_c Velasco"

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
        Options._build_objective(
            self, self._key, eta={"Gamma_c": -2, "Gamma_c Velasco": -1}[self._key]
        )
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Γ_c.

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
        Gamma_c : ndarray
            Γ_c as a function of the flux surface label.

        """
        return Options._compute_objective(self, params, constants, self._key)


class GammaLoss(_Objective):
    """Fast ion prompt-loss proxy based on superbanana classification.

    The ``kind`` argument selects model I, ``"delta"``, which classifies a pitch
    value as lost if there exists an outward superbanana somewhere on the flux
    surface, or model II, ``"alpha"``, which classifies a subset of alpha values
    as lost between consecutive inward and outward superbanana branches.
    If not specified, the default field-line labels are 16 points uniformly spaced
    on ``[0, (1 + eq.sym) * pi)`` and ``num_field_periods`` is ``eq.NFP + 2``.

    References
    ----------
    [1] A model for the fast evaluation of prompt losses of energetic ions in
        stellarators. Equations 22 and 25.
        J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
        https://doi.org/10.1088/1741-4326/ac2994.

    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
        and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    """

    __doc__ = (
        __doc__.rstrip()
        + doc_bounce.split("Parameters\n----------\n", maxsplit=1)[0]
        + """
    Parameters
    ----------
    kind : {"delta", "alpha"}
        Select ``"delta"`` for Γ_δ or ``"alpha"`` for Γ_α.
        """.rstrip()
        + "\n"
        + doc_bounce.split("Parameters\n----------\n", maxsplit=1)[1]
        + "\n"
        + """
    gamma_threshold : float
        Threshold for superbanana classification. Must be in ``(0,1)``.
        Default is 0.2.
        """.rstrip()
        + collect_docs(
            target_default="``target=0``.",
            bounds_default="``target=0``.",
            normalize_detail=" Note: Has no effect for this objective.",
            normalize_target_detail=" Note: Has no effect for this objective.",
            jac_chunk_size=False,
        )
    )

    _static_attrs = _Objective._static_attrs + ["_hyperparam", "_key"]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Γ: "
    _compute_fun = staticmethod(compute_fun)

    @staticmethod
    def _default_alpha(eq, num_alpha=16):
        """Return default field-line labels for prompt-loss objectives."""
        return jnp.linspace(0, (1 + int(eq.sym)) * jnp.pi, num_alpha, endpoint=False)

    def __init__(
        self,
        kind,
        eq,
        *,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name=None,
        grid=None,
        X=32,
        Y=32,
        Y_B=None,
        alpha=None,
        num_field_periods=None,
        num_well=None,
        num_quad=32,
        num_pitch=65,
        gamma_threshold=0.2,
        pitch_batch_size=None,
        surf_batch_size=1,
        nufft_eps=1e-7,
        spline=True,
    ):
        errorif(
            kind not in ("delta", "alpha"),
            ValueError,
            f"Expected kind to be 'delta' or 'alpha', got {kind}.",
        )
        errorif(
            deriv_mode == "fwd",
            ValueError,
            "Reverse mode should be used for the objective: GammaLoss.",
        )
        nufft_eps = check_nufft(nufft_eps)

        if target is None and bounds is None:
            target = 0.0

        self._grid = grid
        if alpha is None:
            alpha = GammaLoss._default_alpha(eq)
        if num_field_periods is None:
            num_field_periods = eq.NFP + 2
        self._constants = {"quad_weights": 1.0, "alpha": alpha}
        self._hyperparam = {
            "X": X,
            "Y": Y,
            "Y_B": Y_B,
            "num_field_periods": num_field_periods,
            "num_well": num_well,
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "gamma_threshold": gamma_threshold,
            "pitch_batch_size": pitch_batch_size,
            "surf_batch_size": surf_batch_size,
            "nufft_eps": nufft_eps,
            "spline": spline,
        }
        self._key = {"delta": "Gamma_delta", "alpha": "Gamma_alpha"}[kind]
        self._print_value_fmt = {"delta": "Γ_δ: ", "alpha": "Γ_α: "}[kind]
        if name is None:
            name = self._key

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
        Options._build_objective(self, self._key, eta=-1)
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the selected fast ion prompt-loss proxy.

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
        Gamma_loss : ndarray
            Γ_δ or Γ_α as a function of the flux surface label.

        """
        return Options._compute_objective(self, params, constants, self._key)
