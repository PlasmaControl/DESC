"""Objectives for fast ion confinement."""

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.integrals.bounce_integral import Options
from desc.utils import warnif

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
        alpha=jnp.array([0.0]),
        num_transit=20,
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
        try:
            import jax_finufft  # noqa: F401
        except:  # noqa: E722
            warnif(
                nufft_eps >= 1e-14,
                msg="\njax-finufft is not installed properly.\n"
                "Setting parameter nufft_eps to zero.\n"
                "Performance may be somewhat slower.\n",
            )
            nufft_eps = 0.0
        nufft_eps = float(nufft_eps)

        if target is None and bounds is None:
            target = 0.0

        self._grid = grid
        self._constants = {"quad_weights": 1.0, "alpha": alpha}
        self._hyperparam = {
            "X": X,
            "Y": Y,
            "Y_B": Y_B,
            "num_transit": num_transit,
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
        if constants is None:
            constants = self.constants
        eq = self.things[0]

        data = compute_fun(
            eq, "iota", params, constants["transforms"], constants["profiles"]
        )
        delta = eq._map_poloidal_coordinates(
            constants["transforms"]["grid"].compress(data["iota"]),
            constants["x"],
            constants["y"],
            params["L_lmn"],
            constants["lambda"],
            outbasis="delta",
            # TODO (#1034): Use old theta values as initial guess.
            tol=1e-8,
        )[..., ::-1]

        data = compute_fun(
            eq,
            self._key,
            params,
            constants["transforms"],
            constants["profiles"],
            data,
            angle=delta,
            alpha=constants["alpha"],
            quad=constants["quad"],
            _vander=constants["_vander"],
            **self._hyperparam,
        )
        return constants["transforms"]["grid"].compress(data[self._key])
