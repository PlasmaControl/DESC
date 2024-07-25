"""Objectives for targeting neoclassical transport."""

import numpy as np

from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer

from ..backend import jnp
from ..profiles import SplineProfile
from .objective_funs import _Objective
from .utils import _parse_callable_target_bounds


class EffectiveRipple(_Objective):
    """The effective ripple is a proxy for neoclassical transport.

    The 3D geometry of the magnetic field in stellarators produces local magnetic
    wells that lead to bad confinement properties with enhanced radial drift,
    especially for trapped particles. Neoclassical (thermal) transport can become the
    dominant transport channel in stellarators which are not optimized to reduce it.
    The effective ripple is a proxy, measuring the effective modulation amplitude of the
    magnetic field averaged along a magnetic surface, which can be used to optimize for
    stellarators with improved confinement.

    References
    ----------
    https://doi.org/10.1063/1.873749.
    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If a callable, should take a
        single argument ``rho`` and return the desired value of the profile at those
        locations. Defaults to 0.
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to Objective.dim_f
        If a callable, each should take a single argument ``rho`` and return the
        desired bound (lower or upper) of the profile at those locations.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
    normalize : bool, optional
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
        Collocation grid to evaluate flux surface averages.
        Should have poloidal and toroidal resolution.
    alpha : ndarray
        Unique coordinate values for field line poloidal angle label alpha.
    zeta : ndarray
        Unique coordinate values for field line following coordinate zeta. Must be
        strictly increasing. A good reference density is 100 knots per toroidal transit.
        For axisymmetric devices only one toroidal transit is necessary. Otherwise,
        more toroidal transits will give more accurate result, with diminishing returns.
    num_quad : int
        Resolution for quadrature of bounce integrals. Default is 31.
    num_pitch : int
        Resolution for quadrature over velocity coordinate, preferably odd.
        Default is 99. Effective ripple will look smoother at high values.
        (If computed on many flux surfaces and micro oscillation is seen
        between neighboring surfaces, increasing num_pitch will smooth the profile).
    batch : bool
        Whether to vectorize part of the computation. Default is true.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Effective ripple ε¹ᐧ⁵: {:10.3e} "

    def __init__(
        self,
        eq,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        alpha=np.array([0]),
        zeta=np.linspace(0, 2 * np.pi, 100),
        num_quad=31,
        num_pitch=99,
        batch=True,
        name="Effective ripple",
    ):
        if bounds is not None:
            target = None

        # Assign in build.
        self._grid_1dr = grid
        self._constants = {"quad_weights": 1}
        self._dim_f = 1
        self._rho = np.array([1.0])
        # Assign here.
        self._alpha = alpha
        self._zeta = zeta
        # R0 should be evaluated on Quadrature grid, but it's just a constant
        # factor, so it's not worth the memory of building the transforms.
        self._keys_1dr = [
            "iota",
            "iota_r",
            "<|grad(rho)|>",
            "min_tz |B|",
            "max_tz |B|",
            "R0",
        ]
        self._keys = ["effective ripple"]
        self._hyperparameters = {
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "batch": batch,
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
        if self._grid_1dr is None:
            self._rho = np.linspace(0.1, 1, 5)
            self._grid_1dr = LinearGrid(
                rho=self._rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
            )
        else:
            self._rho = self._grid_1dr.compress(self._grid_1dr.nodes[:, 0])
        self._dim_f = self._rho.size
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, self._rho
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid_1dr
        )
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + self._keys, eq, self._grid_1dr
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the effective ripple.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        effective_ripple : ndarray

        """
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
        iota = self._grid_1dr.compress(data["iota"])
        iota_r = self._grid_1dr.compress(data["iota_r"])
        grid = eq.get_rtz_grid(
            self._rho,
            self._alpha,
            self._zeta,
            coordinates="raz",
            period=(np.inf, 2 * np.pi, np.inf),
            iota=SplineProfile(iota, df=iota_r, knots=self._rho, name="iota", jnp=jnp),
            params=params,
        )
        data = {
            key: (
                grid.copy_data_from_other(data[key], self._grid_1dr)
                if key != "R0"
                else data[key]
            )
            for key in self._keys_1dr
        }
        data = compute_fun(
            eq,
            self._keys,
            params,
            get_transforms(self._keys, eq, grid, jitable=True),
            constants["profiles"],
            data=data,
            **self._hyperparameters,
        )
        return grid.compress(data["effective ripple"])


class GammaC(_Objective):
    """Γ_c is a proxy for measuring energetic ion confinement.

    References
    ----------
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.
    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If a callable, should take a
        single argument ``rho`` and return the desired value of the profile at those
        locations. Defaults to 0.
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to Objective.dim_f
        If a callable, each should take a single argument ``rho`` and return the
        desired bound (lower or upper) of the profile at those locations.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
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
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid to evaluate flux surface averages.
        Should have poloidal and toroidal resolution.
    alpha : ndarray
        Unique coordinate values for field line poloidal angle label alpha.
    zeta : ndarray
        Unique coordinate values for field line following coordinate zeta. Must be
        strictly increasing. A good reference density is 100 knots per toroidal transit.
        For axisymmetric devices only one toroidal transit is necessary. Otherwise,
        more toroidal transits will give more accurate result, with diminishing returns.
    num_quad : int
        Resolution for quadrature of bounce integrals. Default is 31.
        which gets sufficient convergence, so higher values are likely unnecessary.
    num_pitch : int
        Resolution for quadrature over velocity coordinate. Default is 99.
    batch : bool
        Whether to vectorize part of the computation. Default is true.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Γ_c: {:10.3e} "

    def __init__(
        self,
        eq,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        alpha=np.array([0]),
        zeta=np.linspace(0, 2 * np.pi, 100),
        num_quad=31,
        num_pitch=99,
        batch=True,
        name="Gamma_c",
    ):
        if bounds is not None:
            target = None

        # Assign in build.
        self._grid_1dr = grid
        self._constants = {"quad_weights": 1}
        self._dim_f = 1
        self._rho = np.array([1.0])
        # Assign here.
        self._alpha = alpha
        self._zeta = zeta
        self._keys_1dr = ["iota", "iota_r", "min_tz |B|", "max_tz |B|"]
        self._keys = ["Gamma_c"]
        self._hyperparameters = {
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "batch": batch,
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
        if self._grid_1dr is None:
            self._rho = np.linspace(0.1, 1, 5)
            self._grid_1dr = LinearGrid(
                rho=self._rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
            )
        else:
            self._rho = self._grid_1dr.compress(self._grid_1dr.nodes[:, 0])
        self._dim_f = self._rho.size
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, self._rho
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid_1dr
        )
        self._constants["profiles_1dr"] = get_profiles(
            self._keys_1dr, eq, self._grid_1dr
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Γ_c.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        Gamma_c : ndarray

        """
        if constants is None:
            constants = self.constants
        eq = self.things[0]
        data = compute_fun(
            eq,
            self._keys_1dr,
            params,
            constants["transforms_1dr"],
            constants["profiles_1dr"],
        )
        iota = self._grid_1dr.compress(data["iota"])
        iota_r = self._grid_1dr.compress(data["iota_r"])
        grid = eq.get_rtz_grid(
            self._rho,
            self._alpha,
            self._zeta,
            coordinates="raz",
            period=(np.inf, 2 * np.pi, np.inf),
            iota=SplineProfile(iota, df=iota_r, knots=self._rho, name="iota", jnp=jnp),
            params=params,
        )
        data = {
            key: grid.copy_data_from_other(data[key], self._grid_1dr)
            for key in self._keys_1dr
        }
        data = compute_fun(
            eq,
            self._keys,
            params,
            get_transforms(self._keys, eq, grid, jitable=True),
            get_profiles(self._keys, eq, grid),
            data=data,
            **self._hyperparameters,
        )
        return grid.compress(data["Gamma_c"])
