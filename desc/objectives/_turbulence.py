"""Objectives to target turbulent transport."""

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_params, get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer

from .objective_funs import _Objective


class EffectiveRadius(_Objective):
    """The effective radius is a proxy for turbulent transport.

    TODO Finish descirption

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to Objective.dim_f.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
    normalize : bool, optional
        This quantity is already normalized so this parameter is ignored.
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is ``True`` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    rho : ndarray
        Unique coordinate values specifying flux surfaces to compute on.
        defaults to 0.5 surface
    alpha : ndarray
        Unique coordinate values specifying field line labels to compute on.
        defaults to 0
    n_pol : int
        Number of poloidal transits to follow field line.
    knots_per_transit : int
        Number of points per poloidal transit at which to sample data along field
        line. Default is 200.
    n_wells : Number of wells to fit
        Defaults to 10
    target_type : {"max", "mean", "all"}
        Whether to target only the largest value of R_eff,
        the average of the values along the field line,
        or all the values at the same time.
        Defaults to max
    name : str, optional
        Name of the objective function.
    """

    _scalar = False
    _units = "~"
    _print_value_fmt = "Effective radius of curvature R_eff: "

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
        rho=0.5,  # TODO make more outside
        alpha=0.0,
        n_pol=10,
        knots_per_transit=200,
        n_wells=10,
        target_type="all",
        name="Effective radius",
    ):
        if target is None and bounds is None:
            target = 0.0
        self._rho = rho
        self._alpha = alpha
        self._n_pol = n_pol
        self._knots_per_transit = knots_per_transit
        self._n_wells = n_wells
        self._target_type = target_type
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

    def build(self, eq=None, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]

        # we need a uniform grid to get correct surface averages for iota
        iota_grid = LinearGrid(
            rho=self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
        )
        self._iota_keys = ["iota", "iota_r", "shear"]
        iota_profiles = get_profiles(self._iota_keys, obj=eq, grid=iota_grid)
        iota_transforms = get_transforms(self._iota_keys, obj=eq, grid=iota_grid)

        # Separate grid to calculate the right length scale for normalization
        len_grid = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        self._len_keys = ["a"]
        len_profiles = get_profiles(self._len_keys, obj=eq, grid=len_grid)
        len_transforms = get_transforms(self._len_keys, obj=eq, grid=len_grid)

        # Optimize on all ("all") wells or optimize on a single well ("max" or "mean")
        if self._target_type == "all":
            self._dim_f = self._n_wells
        else:
            self._dim_f = 1
        self._data_keys = ["R_eff"]
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._args = get_params(
            self._iota_keys + self._len_keys + self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        self._hyperparameters = {
            "n_wells": self._n_wells,
        }

        self._constants = {
            "iota_transforms": iota_transforms,
            "iota_profiles": iota_profiles,
            "len_transforms": len_transforms,
            "len_profiles": len_profiles,
            "rho": self._rho,
            "alpha": self._alpha,
            "n_wells": self._n_wells,
            "quad_weights": 1.0,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute effective radius.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        R_eff : float/ndarray
            if target_mode is max/mean, maximum or average effective radius
            along the field line.
            if target_mode is all, array or effective radii along the field line
        """
        eq = self.things[0]

        if constants is None:
            constants = self.constants

        # we first compute iota on a uniform grid to get correct averaging etc.
        iota_data = compute_fun(
            eq,
            self._iota_keys,
            params=params,
            transforms=constants["iota_transforms"],
            profiles=constants["iota_profiles"],
        )

        len_data = compute_fun(
            eq,
            self._len_keys,
            params=params,
            transforms=constants["len_transforms"],
            profiles=constants["len_profiles"],
        )

        rho, alpha = constants["rho"], constants["alpha"]
        # we prime the data dict with the correct iota values so we don't recompute them
        # using the wrong grid
        data = {
            "iota": iota_data["iota"][0],
            "iota_r": iota_data["iota_r"][0],
            "shear": iota_data["shear"][0],
            "a": len_data["a"],
        }
        n_tor = self._n_pol / (data["iota"] * eq.NFP)
        zeta = jnp.linspace(
            0, 2 * jnp.pi * n_tor, self._n_pol * self._knots_per_transit
        )

        grid = eq.get_rtz_grid(
            rho,
            alpha,
            zeta,
            coordinates="raz",
            iota=data["iota"],
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        )

        data = compute_fun(
            eq,
            self._data_keys,
            params=params,
            transforms=get_transforms(self._data_keys, eq, grid, jitable=True),
            profiles=get_profiles(self._data_keys, eq, grid),
            data=data,
            **self._hyperparameters,
        )

        if self._target_type == "max":
            return jnp.max(data["R_eff"])
        elif self._target_type == "mean":
            return jnp.mean(data["R_eff"])
        elif self._target_type == "all":
            return data["R_eff"]


class ParallelConnectionLength(_Objective):
    """The parallel connection length is a proxy for turbulent transport.

    TODO Finish descirption

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to Objective.dim_f.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
    normalize : bool, optional
        This quantity is already normalized so this parameter is ignored.
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is ``True`` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    rho : ndarray
        Unique coordinate values specifying flux surfaces to compute on.
        defaults to 0.5 surface
    alpha : ndarray
        Unique coordinate values specifying field line labels to compute on.
        defaults to 0
    n_pol : int
        Number of poloidal transits to follow field line.
    knots_per_transit : int
        Number of points per poloidal transit at which to sample data along field
        line. Default is 200.
    target_type : {"max", "mean", "all"}
        Whether to target only the largest value of R_eff,
        the average of the values along the field line,
        or all the values at the same time TODO see if all is possible
        Defaults to max
    name : str, optional
        Name of the objective function.
    """

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
        rho=0.5,  # TODO make more outside
        alpha=0.0,
        n_pol=10,
        knots_per_transit=200,
        n_wells=10,
        target_type="all",
        name="Parallel connection length",
    ):
        if target is None and bounds is None:
            target = 0.0
        self._rho = rho
        self._alpha = alpha
        self._n_pol = n_pol
        self._knots_per_transit = knots_per_transit
        self._n_wells = n_wells
        self._target_type = target_type
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

    def build(self, eq=None, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]

        # we need a uniform grid to get correct surface averages for iota
        iota_grid = LinearGrid(
            rho=self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
        )
        self._iota_keys = ["iota", "iota_r", "shear"]
        iota_profiles = get_profiles(self._iota_keys, obj=eq, grid=iota_grid)
        iota_transforms = get_transforms(self._iota_keys, obj=eq, grid=iota_grid)

        # Separate grid to calculate the right length scale for normalization
        len_grid = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        self._len_keys = ["a"]
        len_profiles = get_profiles(self._len_keys, obj=eq, grid=len_grid)
        len_transforms = get_transforms(self._len_keys, obj=eq, grid=len_grid)

        if self._target_type == "all":
            self._dim_f = self._n_wells
        else:
            self._dim_f = 1

        self._data_keys = ["L_par"]
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._args = get_params(
            self._iota_keys + self._len_keys + self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        self._hyperparameters = {
            "n_wells": self._n_wells,
        }

        self._constants = {
            "iota_transforms": iota_transforms,
            "iota_profiles": iota_profiles,
            "len_transforms": len_transforms,
            "len_profiles": len_profiles,
            "rho": self._rho,
            "alpha": self._alpha,
            "n_wells": self._n_wells,
            "quad_weights": 1.0,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute parallel connection length.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        L_par : float/ndarray
            if target_mode is max/mean, maximum or average parallel connection length
            along the field line.
            if target_mode is all, array of L_par along the field line
        """
        eq = self.things[0]

        if constants is None:
            constants = self.constants

        # we first compute iota on a uniform grid to get correct averaging etc.
        iota_data = compute_fun(
            eq,
            self._iota_keys,
            params=params,
            transforms=constants["iota_transforms"],
            profiles=constants["iota_profiles"],
        )

        len_data = compute_fun(
            eq,
            self._len_keys,
            params=params,
            transforms=constants["len_transforms"],
            profiles=constants["len_profiles"],
        )

        rho, alpha = constants["rho"], constants["alpha"]
        # we prime the data dict with the correct iota values so we don't recompute them
        # using the wrong grid
        data = {
            "iota": iota_data["iota"][0],
            "iota_r": iota_data["iota_r"][0],
            "shear": iota_data["shear"][0],
            "a": len_data["a"],
        }
        n_tor = self._n_pol / (data["iota"] * eq.NFP)
        zeta = jnp.linspace(
            0, 2 * jnp.pi * n_tor, self._n_pol * self._knots_per_transit
        )

        grid = eq.get_rtz_grid(
            rho,
            alpha,
            zeta,
            coordinates="raz",
            iota=data["iota"],
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        )

        data = compute_fun(
            eq,
            self._data_keys,
            params=params,
            transforms=get_transforms(self._data_keys, eq, grid, jitable=True),
            profiles=get_profiles(self._data_keys, eq, grid),
            data=data,
            **self._hyperparameters,
        )

        if self._target_type == "max":
            return jnp.max(data["L_par"])
        elif self._target_type == "mean":
            return jnp.mean(data["L_par"])
        elif self._target_type == "all":
            return data["L_par"]
