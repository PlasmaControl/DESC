"""Objectives for neoclassical transport."""

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.integrals._bounce_utils import Y_B_rule, num_well_rule
from desc.integrals.bounce_integral import Bounce2D
from desc.utils import setdefault, warnif

from ..integrals.quad_utils import chebgauss2
from .objective_funs import _Objective, collect_docs, doc_bounce
from .utils import _parse_callable_target_bounds


class EffectiveRipple(_Objective):
    """Proxy for neoclassical transport in the banana regime.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. In the banana regime, neoclassical (thermal)
    transport from ripple wells can become the dominant transport channel.
    The effective ripple (ε) proxy estimates the neoclassical transport
    coefficients in the banana regime. To ensure low neoclassical transport,
    a stellarator is typically optimized so that ε < 0.02.

    References
    ----------
    [1] Evaluation of 1/ν neoclassical transport in stellarators.
        V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
        Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.
        https://doi.org/10.1063/1.873749.

    [2] Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
        and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    """

    __doc__ = (
        __doc__.rstrip()
        + doc_bounce.rstrip()
        + collect_docs(
            target_default="``target=0``.",
            bounds_default="``target=0``.",
            normalize_detail=" Note: Has no effect for this objective.",
            normalize_target_detail=" Note: Has no effect for this objective.",
        )
    )

    _static_attrs = _Objective._static_attrs + [
        "_hyperparam",
        "_keys_1dr",
        "_use_bounce1d",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Effective ripple ε: "

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
        jac_chunk_size=None,
        name="Effective ripple",
        grid=None,
        X=32,
        Y=32,
        Y_B=None,
        alpha=jnp.array([0.0]),
        num_transit=20,
        num_well=None,
        num_quad=32,
        num_pitch=51,
        pitch_batch_size=None,
        surf_batch_size=1,
        nufft_eps=1e-6,
        spline=True,
        use_bounce1d=False,
        **kwargs,
    ):
        try:
            import jax_finufft  # noqa: F401
        except:  # noqa: E722
            warnif(
                nufft_eps >= 1e-14,
                msg="\njax-finufft is not installed properly.\n"
                "Setting parameter nufft_eps to zero.\n"
                "Performance will deteriorate significantly.\n",
            )
            nufft_eps = 0.0

        if target is None and bounds is None:
            target = 0.0

        self._use_bounce1d = use_bounce1d
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
        if use_bounce1d:
            self._hyperparam.pop("X")
            self._hyperparam.pop("Y")
            self._hyperparam.pop("pitch_batch_size")
            self._hyperparam.pop("nufft_eps")
            self._hyperparam.pop("spline")

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
            jac_chunk_size=jac_chunk_size,
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
        if self._use_bounce1d:
            return self._build_bounce1d(use_jit, verbose)

        Bounce2D._build(self, names="effective ripple", eta=1)
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
            Defaults to ``self.constants``.

        Returns
        -------
        epsilon : ndarray
            Effective ripple as a function of the flux surface label.

        """
        if self._use_bounce1d:
            return self._compute_bounce1d(params, constants)

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
            "effective ripple",
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
        return constants["transforms"]["grid"].compress(data["effective ripple"])

    def _build_bounce1d(self, use_jit=True, verbose=1):
        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        assert self._grid.is_meshgrid and eq.sym == self._grid.sym

        Y_B = self._hyperparam.pop("Y_B")
        Y_B = setdefault(Y_B, Y_B_rule(self._grid, spline=True))

        num_transit = self._hyperparam.pop("num_transit")

        if self._hyperparam["num_well"] is None:
            self._hyperparam["num_well"] = num_well_rule(num_transit, eq.NFP, Y_B)

        num_quad = self._hyperparam.pop("num_quad")

        self._constants["zeta"] = jnp.linspace(
            0, 2 * jnp.pi * num_transit, Y_B * num_transit
        )

        self._keys_1dr = [
            "iota",
            "iota_r",
            "<|grad(rho)|>",
            "min_tz |B|",
            "max_tz |B|",
            "R0",
        ]

        rho = self._grid.compress(self._grid.nodes[:, 0])
        self._constants["rho"] = rho
        self._constants["quad"] = chebgauss2(num_quad)
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + ["old effective ripple"], eq, self._grid
        )
        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid
        )

        self._dim_f = self._grid.num_rho
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, rho
        )
        super().build(use_jit=use_jit, verbose=verbose)

    def _compute_bounce1d(self, params, constants=None):
        if constants is None:
            constants = self.constants
        eq = self.things[0]

        data = compute_fun(
            eq,
            self._keys_1dr,
            params,
            constants["transforms_1dr"],
            constants["profiles"],
        )
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=self._grid.compress(data["iota"]),
            params=params,
        )
        data = {
            key: (
                grid.copy_data_from_other(data[key], self._grid)
                if key != "R0"
                else data[key]
            )
            for key in self._keys_1dr
        }
        data = compute_fun(
            eq,
            "old effective ripple",
            params,
            transforms=get_transforms("old effective ripple", eq, grid, jitable=True),
            profiles=constants["profiles"],
            data=data,
            quad=constants["quad"],
            **self._hyperparam,
        )
        return grid.compress(data["old effective ripple"])
