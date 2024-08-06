"""Objectives for targeting energy confinement."""

import numpy as np

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import LinearGrid, QuadratureGrid
from desc.utils import Timer, errorif

from .objective_funs import _Objective


class HeatingPower(_Objective):
    """Heating power required by the ISS04 energy confinement time scaling.

    tau_E^ISS04 = 0.134 a^2.28 R^0.64 P^-0.61 n_e^0.54 B^0.84 iota^0.41 (s)

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
        Renormalization or confinement enhancement factor. Defaults = 1.
    gamma : float, optional
        Adiabatic (compressional) index. Default = 0.
    grid_geometry : Grid, optional
        Collocation grid used to compute the major and minor radii.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)``.
    grid_density : Grid, optional
        Collocation grid used to compute the line-averaged density.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)``.
    grid_B : Grid, optional
        Collocation grid used to compute the magnetic field strength.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)``.
    grid_iota : Grid, optional
        Collocation grid used to compute the rotational transform.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)``.
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
        gamma=0,
        grid_geometry=None,
        grid_density=None,
        grid_B=None,
        grid_iota=None,
        name="heating power",
    ):
        if target is None and bounds is None:
            target = 2
        self._f_ren = f_ren
        self._gamma = gamma
        self._grid_geometry = grid_geometry
        self._grid_density = grid_density
        self._grid_B = grid_B
        self._grid_iota = grid_iota
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

        if self._grid_geometry is None:
            self._grid_geometry = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        if self._grid_density is None:
            self._grid_density = LinearGrid(L=eq.L_grid, M=0, N=0)
        if self._grid_B is None:
            self._grid_B = LinearGrid(rho=np.array([0]), N=eq.N_grid, NFP=eq.NFP)
        if self._grid_iota is None:
            self._grid_iota = LinearGrid(
                rho=np.array([2 / 3.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )

        self._dim_f = 1
        self._data_keys_geometry = ["a", "R0", "W_p"]
        self._data_keys_density = ["ne"]
        self._data_keys_B = ["|B|"]
        self._data_keys_iota = ["iota"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants = {
            "profiles_geometry": get_profiles(
                self._data_keys_geometry, obj=eq, grid=self._grid_geometry
            ),
            "transforms_geometry": get_transforms(
                self._data_keys_geometry, obj=eq, grid=self._grid_geometry
            ),
            "profiles_density": get_profiles(
                self._data_keys_density, obj=eq, grid=self._grid_density
            ),
            "transforms_density": get_transforms(
                self._data_keys_density, obj=eq, grid=self._grid_density
            ),
            "profiles_B": get_profiles(self._data_keys_B, obj=eq, grid=self._grid_B),
            "transforms_B": get_transforms(
                self._data_keys_B, obj=eq, grid=self._grid_B
            ),
            "profiles_iota": get_profiles(
                self._data_keys_iota, obj=eq, grid=self._grid_iota
            ),
            "transforms_iota": get_transforms(
                self._data_keys_iota, obj=eq, grid=self._grid_iota
            ),
            "f_ren": self._f_ren,
            "gamma": self._gamma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

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
        data_geometry = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys_geometry,
            params=params,
            transforms=constants["transforms_geometry"],
            profiles=constants["profiles_geometry"],
            gamma=constants["gamma"],
        )
        data_density = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys_density,
            params=params,
            transforms=constants["transforms_density"],
            profiles=constants["profiles_density"],
        )
        data_B = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys_B,
            params=params,
            transforms=constants["transforms_B"],
            profiles=constants["profiles_B"],
        )
        data_iota = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys_iota,
            params=params,
            transforms=constants["transforms_iota"],
            profiles=constants["profiles_iota"],
        )

        P = (data_geometry["W_p"] / 1e6) / (  # MJ
            0.134
            * data_geometry["a"] ** 2.28  # m
            * data_geometry["R0"] ** 0.64  # m
            * (jnp.mean(data_density["n_e"]) ** 0.54 / 1e19)  # 1e19/m^3
            * jnp.mean(data_B["|B|"]) ** 0.84  # T
            * jnp.mean(data_iota["iota"]) ** 0.41
            * constants["f_ren"]
        ) ** -0.39
        return P

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
