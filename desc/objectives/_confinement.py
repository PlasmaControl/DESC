"""Objectives for targeting energy confinement."""

from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import QuadratureGrid
from desc.utils import Timer, errorif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class HeatingPower(_Objective):
    """Heating power required by the ISS04 energy confinement time scaling.

    tau_E^ISS04 = 0.134 a^2.28 R^0.64 P^-0.61 n_e^0.54 B^0.84 iota^0.41 (s)

    References
    ----------
    https://doi.org/10.1088/0029-5515/45/12/024.
    Characterization of energy confinement in net-current free plasmas using the
    extended International Stellarator Database.
    H. Yamada et al. Nucl. Fusion 29 November 2005; 45 (12): 1684–1693.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=0``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Note: Has no effect for this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: Has no effect for this objective.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    f_ren : float, optional
        ISS04 renormalization factor. Default = 1.
    H_ISS04 : float, optional
        ISS04 confinement enhancement factor. Default = 1.
    gamma : float, optional
        Adiabatic (compressional) index. Default = 0.
    grid : Grid, optional
        Collocation grid used to compute the intermediate quantities.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)``.
    name : str, optional
        Name of the objective function.

    """

    _scalar = True
    _units = "(W)"
    _print_value_fmt = "Heating power: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        f_ren=1,
        H_ISS04=1,
        gamma=0,
        grid=None,
        name="heating power",
    ):
        if target is None and bounds is None:
            target = 0
        self._f_ren = f_ren
        self._H_ISS04 = H_ISS04
        self._gamma = gamma
        self._grid = grid
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
        eq = self.things[0]
        errorif(
            eq.electron_density is None,
            ValueError,
            "Equilibrium must have an electron density profile.",
        )
        if self._grid is None:
            self._grid = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        self._dim_f = 1
        self._data_keys = ["W_p", "a", "R0", "<ne>_rho", "<|B|>_axis", "iota_23"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants = {
            "profiles": get_profiles(self._data_keys, obj=eq, grid=self._grid),
            "transforms": get_transforms(self._data_keys, obj=eq, grid=self._grid),
            "f_ren": self._f_ren,
            "H_ISS04": self._H_ISS04,
            "gamma": self._gamma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["W"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute heating power.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom, eg
            Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        P : float
            Heating power required by the ISS04 energy confinement time scaling (W).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            gamma=constants["gamma"],
        )

        P = (
            (data["W_p"] / -1e6)  # MJ
            / (
                0.134
                * data["a"] ** 2.28  # m
                * data["R0"] ** 0.64  # m
                * (data["<ne>_rho"] / 1e19) ** 0.54  # 1e19/m^3
                * data["<|B|>_axis"] ** 0.84  # T
                * data["iota_23"] ** 0.41
                * constants["f_ren"]
                * constants["H_ISS04"]
            )
        ) ** (1 / 0.39)
        return P * 1e6  # W

    @property
    def f_ren(self):
        """float: Confinement enhancement factor."""
        return self._f_ren

    @f_ren.setter
    def f_ren(self, new):
        self._f_ren = new

    @property
    def gamma(self):
        """float: Adiabatic (compressional) index."""
        return self._gamma

    @gamma.setter
    def gamma(self, new):
        self._gamma = new
