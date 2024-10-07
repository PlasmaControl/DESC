'''Objectives to target turbulent transport'''
import numpy as np

from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.backend import jnp
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective
from ..equilibrium.coords import get_rtz_grid

class EffectiveRadius(_Objective):
    '''The effective radius of curvature is a proxy for turbulent transport
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
    grid : Collocation grid containing the nodes used to compute the drift curvature and fit the drift wells
        Defaults to TODO
    target_type : {"max", "mean", "all"}
        Whether to target only the largest value of R_eff, the average of the values along the field line
        or all the values at the same time TODO see if all is possible
        Defaults to max
    name : str, optional
        Name of the objective function.
    '''

    _coordinates = ""   # Effective radius is a scalar parameter of each drift well 
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
            rho=0.5,    # TODO make more outside 
            alpha=0.0,
            n_pol=10,
            knots_per_transit=200,
            grid = None,
            target_type = "max",
            name="Effective radius",
        ):
            if target is None and bounds is None:
                target = 0.0
            self._rho = rho
            self._alpha = alpha
            self._n_pol = n_pol
            self._knots_per_transit = knots_per_transit
            self._grid = grid
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

        if self._grid is None:
            # Get value of iota on the chosen surface
            iota_grid = get_rtz_grid(
                eq,
                np.array(self._rho),
                np.array(self._alpha),
                np.array(0),
                coordinates="rtz",
                period=(np.inf,2*np.pi,np.inf),
            )
            iota = eq.compute("iota",grid=iota_grid)["iota"]
            n_tor = self._n_pol/(iota*eq.NFP)
            grid = get_rtz_grid(
                eq,
                np.array(self._rho),
                np.array(self._alpha),
                np.linspace(0,2*n_tor*np.pi,self._n_pol*self._knots_per_transit),
                coordinates="raz",
                period=(np.inf,2*np.pi,np.inf),
            )
        else :
            grid = self._grid
        self._dim_f = 1
        self._data_keys = ["R_eff"]
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
        """Compute effective radius

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
            if target_mode is max/mean, maximum or average effective radius along the field line
            if target_mode is all, array or effective radii along the field line            
        """
        eq = self.things[0]

        if constants is None:
            constants = self.constants
        data = compute_fun(
            eq,
            self._data_keys,
            params=params,
            data=data,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        R_eff = data["R_eff"]
        if self._target_type == "max":
            R_eff = jnp.max(R_eff)
        elif self._target_type == "mean":
            R_eff = jnp.mean("R_eff")
        return R_eff
