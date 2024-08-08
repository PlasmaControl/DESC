"""Objectives for targeting MHD stability."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_params, get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid
from desc.utils import Timer, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective
from .utils import _parse_callable_target_bounds


class MercierStability(_Objective):
    """The Mercier criterion is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with D_Mercier > 0 are favorable for stability.

    See equation 4.16 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If a callable, should take a
        single argument `rho` and return the desired value of the profile at those
        locations. Defaults to ``bounds=(0, np.inf)``
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
        If a callable, each should take a single argument `rho` and return the
        desired bound (lower or upper) of the profile at those locations.
        Defaults to ``bounds=(0, np.inf)``
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "r"
    _units = "(Wb^-2)"
    _print_value_fmt = "Mercier Stability: {:10.3e} "

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
        name="Mercier Stability",
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
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
        if self._grid is None:
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            RuntimeWarning,
            "MercierStability objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "MercierStability objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["D_Mercier"]

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

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = 1 / scales["Psi"] ** 2

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the Mercier stability criterion.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        D_Mercier : ndarray
            Mercier stability criterion.

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return constants["transforms"]["grid"].compress(data["D_Mercier"])


class MagneticWell(_Objective):
    """The magnetic well is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with magnetic well > 0 are favorable for stability.

    This objective uses the magnetic well parameter defined in equation 3.2 of
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If a callable, should take a
        single argument `rho` and return the desired value of the profile at those
        locations. Defaults to ``bounds=(0, np.inf)``
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
        If a callable, each should take a single argument `rho` and return the
        desired bound (lower or upper) of the profile at those locations.
        Defaults to ``bounds=(0, np.inf)``
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Note: Has no effect for this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "r"
    _units = "(dimensionless)"
    _print_value_fmt = "Magnetic Well: {:10.3e} "

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
        name="Magnetic Well",
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
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
        if self._grid is None:
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            RuntimeWarning,
            "MagneticWell objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "MagneticWell objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["magnetic well"]

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
        """Compute a magnetic well parameter.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        magnetic_well : ndarray
            Magnetic well parameter.

        """
        if constants is None:
            constants = self.constants

        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return constants["transforms"]["grid"].compress(data["magnetic well"])


class BallooningStability(_Objective):
    """A type of ideal MHD instability.

    Infinite-n ideal MHD ballooning modes are of significant interest.
    These instabilities are also related to smaller-scale kinetic instabilities.
    With this class, we optimize MHD equilibria against the ideal ballooning mode.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If a callable, should take a
        single argument `rho` and return the desired value of the profile at those
        locations.
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
        If a callable, each should take a single argument `rho` and return the
        desired bound (lower or upper) of the profile at those locations.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "r"
    _units = "(normalized)"
    _print_value_fmt = "Ideal-ballooning Stability: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        deriv_mode="rev",
        loss_function=None,
        rho=0.5,
        alpha=jnp.array([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0], [1.2], [1.4]]),
        zetamax=3 * jnp.pi,
        nzeta=200,
        name="ideal-ball gamma",
    ):
        if target is None and bounds is None:
            target = 0

        self.rho = rho
        self.alpha = alpha
        self.zetamax = zetamax
        self.nzeta = nzeta

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
        iota_grid = LinearGrid(rho=self.rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        self._iota_keys = ["iota", "iota_r", "iota_rr"]  # might not need all of these
        iota_profiles = get_profiles(self._iota_keys, obj=eq, grid=iota_grid)
        iota_transforms = get_transforms(self._iota_keys, obj=eq, grid=iota_grid)

        ## Separate grid to calculate the right length scale for normalization
        len_grid = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        self._len_keys = ["a"]
        len_profiles = get_profiles(self._len_keys, obj=eq, grid=len_grid)
        len_transforms = get_transforms(self._len_keys, obj=eq, grid=len_grid)

        # make a set of nodes along a single fieldline
        zeta = np.linspace(-self.zetamax, self.zetamax, self.nzeta)
        rho, alpha, zeta = np.reshape(
            np.broadcast_arrays(self.rho, self.alpha, zeta), (3, -1)
        )
        fieldline_nodes = np.array([rho, alpha, zeta]).T

        self._dim_f = 1
        self._data_keys = ["ideal_ball_gamma2"]

        self._args = get_params(
            self._iota_keys + self._len_keys + self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        self._constants = {
            "iota_transforms": iota_transforms,
            "iota_profiles": iota_profiles,
            "len_transforms": len_transforms,
            "len_profiles": len_profiles,
            "fieldline_nodes": fieldline_nodes,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """
        Compute the ballooning stability growth rate.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        gamma : ndarray
            ideal ballooning growth rate.

        """
        from desc.equilibrium.coords import compute_theta_coords

        eq = self.things[0]

        if constants is None:
            constants = self.constants
        # we first compute iota on a uniform grid to get correct averaging etc.
        iota_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._iota_keys,
            params=params,
            transforms=constants["iota_transforms"],
            profiles=constants["iota_profiles"],
        )

        len_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._len_keys,
            params=params,
            transforms=constants["len_transforms"],
            profiles=constants["len_profiles"],
        )

        # Now we compute theta_DESC for given theta_PEST
        iota = iota_data["iota"][0]
        rho, alpha, zeta = constants["fieldline_nodes"].T
        theta_PEST = alpha + iota * zeta
        theta_coords = jnp.array([rho, theta_PEST, zeta]).T
        desc_coords = compute_theta_coords(
            eq, theta_coords, L_lmn=params["L_lmn"], tol=1e-8, maxiter=40
        )

        sfl_grid = Grid(desc_coords, sort=False, jitable=True)
        transforms = get_transforms(
            self._data_keys, obj=eq, grid=sfl_grid, jitable=True
        )
        profiles = get_profiles(self._data_keys, obj=eq, grid=sfl_grid)

        # we prime the data dict with the correct iota values so we don't recompute them
        # using the wrong grid
        data = {
            "iota": iota_data["iota"][0] * jnp.ones_like(zeta),
            "iota_r": iota_data["iota_r"][0] * jnp.ones_like(zeta),
            "iota_rr": iota_data["iota_rr"][0] * jnp.ones_like(zeta),
            "a": len_data["a"],
        }

        # now compute ballooning stuff
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
        )

        return data["ideal_ball_gamma2"]
