"""Objectives to target turbulent transport."""

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer

from .objective_funs import _Objective


class GradRho(_Objective):
    """Grad Rho is a proxy for turbulent transport.

    https://arxiv.org/html/2405.19860v1.

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
            Value of integral from https://arxiv.org/html/2405.19860v1
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
