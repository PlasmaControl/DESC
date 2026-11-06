"""Objectives to target turbulent transport."""

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_params, get_profiles, get_transforms
from desc.equilibrium.coords import get_rtz_grid
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
        n_tor = jnp.abs(self._n_pol / (data["iota"]))
        zeta = jnp.linspace(
            -2 * jnp.pi / jnp.abs(data["iota"]),
            2 * jnp.pi * n_tor,
            (self._n_pol + 1) * self._knots_per_transit,
        )

        grid = get_rtz_grid(
            eq,
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
        Whether to target only the largest value of L_par,
        the average of the values along the field line,
        or all the values at the same time
        Defaults to all
    n_wells : int
        Number of wells to target if target_type is set to "mean" or "all"
    curvature : {"good", "bad"}
        Whether to target regions of bad curvature, making them smaller or
        regions of good curvature to make them larger.
        Define two objectives to target both
    name : str, optional
        Name of the objective function.
    """

    _scalar = False
    _units = "~"
    _print_value_fmt = "Parallel connection length L_par: "

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
        target_type="all",
        n_wells=5,
        curvature="bad",
        name="Parallel connection length",
    ):
        if target is None and bounds is None:
            target = 0
        self._rho = rho
        self._alpha = alpha
        self._n_pol = n_pol
        self._knots_per_transit = knots_per_transit
        self._n_wells = n_wells
        self._target_type = target_type
        self._curvature = curvature
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
            "curvature": self._curvature,
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
        n_tor = jnp.abs(self._n_pol / (data["iota"]))
        zeta = jnp.linspace(
            -2 * jnp.pi / jnp.abs(data["iota"]),
            2 * jnp.pi * n_tor,
            (self._n_pol + 1) * self._knots_per_transit,
        )

        grid = get_rtz_grid(
            eq,
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
            if self._curvature == "good":
                L_par = jnp.min(data["L_par"])
            else:
                L_par = jnp.max(data["L_par"])
        elif self._target_type == "mean":
            L_par = jnp.mean(data["L_par"])
        elif self._target_type == "all":
            L_par = data["L_par"]

        if self._hyperparameters["curvature"] == "good":
            return 1 / L_par
        else:
            return L_par


class GradRho(_Objective):
    """Grad Rho is a proxy for turbulent transport.

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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(rho=0.5, M=50, N=50,NFP=eq.NFP)``.
    rho : ndarray
        Unique coordinate values specifying flux surfaces to compute on.
        defaults to 0.5 surface
    """

    _scalar = True
    _units = "~"
    _print_value_fmt = "Grad rho: "

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
        grid=None,
        rho=jnp.array([0.5]),
        name="GradRho",
    ):
        if target is None and bounds is None:
            target = 0.0
        self._grid = grid
        self._rho = rho
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
        if self._grid is None:
            grid = LinearGrid(rho=self._rho, M=50, N=50, NFP=eq.NFP, endpoint=True)
        else:
            grid = self._grid

        self._dim_f = len(self._rho)
        self._data_keys = ["xi"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute grad s.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        integral : float/ndarray
            TODO finish
        """
        eq = self.things[0]

        if constants is None:
            constants = self.constants

        grid = constants["transforms"]["grid"]

        data = compute_fun(
            eq,
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )

        xi_flat = jnp.array(data["xi"].flatten())
        xi_95 = jnp.percentile(xi_flat, 95)
        mask = data["xi"] < xi_95
        integrand = grid.spacing[:, 1] * grid.spacing[:, 2] * mask * data["xi"]
        integral = integrand.sum()

        return integral
