"""Objectives for turbulence proxies."""

from desc.backend import jnp
from desc.compute._turbulence import _energy_quad
from desc.compute.utils import _compute as compute_fun
from desc.integrals._interp_utils import check_nufft
from desc.integrals.bounce_integral import Options

from .objective_funs import _Objective, collect_docs, doc_bounce
from .utils import errorif


class AvailableEnergy(_Objective):
    """Available energy of trapped electrons.

    The available-energy metric estimates the dimensionless free energy available
    to trapped electrons from density and temperature profile gradients.

    The objective is presented in [1]_, and the computation is presented in [2]_.
    This objective computes the particle drifts using a flux tube model;
    and therefore, has a meaningless ergodic limit. An optimization should be
    evaluated by measuring improvement over a fixed number of field period
    transits.

    References
    ----------
    .. [1] R. J. J. Mackenbach et al., J. Plasma Phys. 89, 905890513 (2023).
    .. [2] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
           bounce-averaging algorithm and its applications,"
           J. Plasma Physics. https://doi:10.1017/S0022377826101652.

    """

    __doc__ = (
        __doc__.rstrip()
        + doc_bounce
        + """
    num_energy : int
        Resolution for generalized Laguerre quadrature over energy.
        Default is 16.
    radial_scale : float
        Multiplier for the radial correlation length.
        Default is 1.0.
    binormal_scale : float
        Multiplier for the binormal correlation length.
        Default is 1.0.
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
        num_energy=16,
        radial_scale=1.0,
        binormal_scale=1.0,
    ):
        errorif(
            deriv_mode == "fwd",
            ValueError,
            "Reverse mode should be used for the objective: AvailableEnergy.",
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
            "num_energy": num_energy,
            "radial_scale": radial_scale,
            "binormal_scale": binormal_scale,
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
        self._constants["energy_quad"] = _energy_quad(
            self._hyperparam.pop("num_energy")
        )
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
