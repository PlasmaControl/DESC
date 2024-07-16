"""Objectives for targeting geometrical quantities."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms, rpz2xyz
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import safenorm
from desc.grid import LinearGrid, QuadratureGrid
from desc.utils import Timer, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective
from .utils import softmin


class AspectRatio(_Objective):
    """Aspect ratio = major radius / minor radius.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium or FourierRZToroidalSurface that
        will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=2``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=2``.
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.
    name : str, optional
        Name of the objective function.

    """

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Aspect ratio: {:10.3e} "

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
        name="aspect ratio",
    ):
        if target is None and bounds is None:
            target = 2
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
            if hasattr(eq, "L_grid"):
                grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )
            else:
                # if not an Equilibrium, is a Surface,
                # has no radial resolution so just need
                # the surface points
                grid = LinearGrid(
                    rho=1.0,
                    M=eq.M * 2,
                    N=eq.N * 2,
                    NFP=eq.NFP,
                )
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["R0/a"]

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
        """Compute aspect ratio.

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
        AR : float
            Aspect ratio, dimensionless.

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self.things[0],
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return data["R0/a"]


class Elongation(_Objective):
    """Elongation = semi-major radius / semi-minor radius.

    Elongation is a function of the toroidal angle.
    Default ``loss_function="max"`` returns the maximum of all toroidal angles.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium or FourierRZToroidalSurface that
        will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=1``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=1``.
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.
    name : str, optional
        Name of the objective function.

    """

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Elongation: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function="max",
        deriv_mode="auto",
        grid=None,
        name="elongation",
    ):
        if target is None and bounds is None:
            target = 1
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
            if hasattr(eq, "L_grid"):
                grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )
            else:
                # if not an Equilibrium, is a Surface,
                # has no radial resolution so just need
                # the surface points
                grid = LinearGrid(
                    rho=1.0,
                    M=eq.M * 2,
                    N=eq.N * 2,
                    NFP=eq.NFP,
                )
        else:
            grid = self._grid

        self._dim_f = grid.num_zeta
        self._data_keys = ["a_major/a_minor"]

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
        """Compute elongation.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom,
            eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        elongation : float
            Elongation, dimensionless.

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self.things[0],
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return self._constants["transforms"]["grid"].compress(
            data["a_major/a_minor"], surface_label="zeta"
        )


class Volume(_Objective):
    """Plasma volume.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium or FourierRZToroidalSurface that
        will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=1``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=1``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
        be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: Has no effect for this objective.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.
    name : str, optional
        Name of the objective function.

    """

    _scalar = True
    _units = "(m^3)"
    _print_value_fmt = "Plasma volume: {:10.3e} "

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
        name="volume",
    ):
        if target is None and bounds is None:
            target = 1
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
            if hasattr(eq, "L_grid"):
                grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )
            else:
                # if not an Equilibrium, is a Surface,
                # has no radial resolution so just need
                # the surface points
                grid = LinearGrid(
                    rho=1.0,
                    M=eq.M * 2,
                    N=eq.N * 2,
                    NFP=eq.NFP,
                )
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["V"]

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
            self._normalization = scales["V"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute plasma volume.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom,
            eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        V : float
            Plasma volume (m^3).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self.things[0],
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return data["V"]


class PlasmaVesselDistance(_Objective):
    """Target the distance between the plasma and a surrounding surface.

    Computes the minimum distance from each point on the surface grid to a point on the
    plasma grid. For dense grids, this will approximate the global min, but in general
    will only be an upper bound on the minimum separation between the plasma and the
    surrounding surface.

    NOTE: By default, assumes the surface is not fixed and its coordinates are computed
    at every iteration, for example if the winding surface you compare to is part of the
    optimization and thus changing.
    If the bounding surface is fixed, set surface_fixed=True to precompute the surface
    coordinates and improve the efficiency of the calculation.

    NOTE: for best results, use this objective in combination with either MeanCurvature
    or PrincipalCurvature, to penalize the tendency for the optimizer to only move the
    points on surface corresponding to the grid that the plasma-vessel distance
    is evaluated at, which can cause cusps or regions of very large curvature.

    NOTE: When use_softmin=True, ensures that alpha*values passed in is
    at least >1, otherwise the softmin will return inaccurate approximations
    of the minimum. Will automatically multiply array values by 2 / min_val if the min
    of alpha*array is <1. This is to avoid inaccuracies that arise when values <1
    are present in the softmin, which can cause inaccurate mins or even incorrect
    signs of the softmin versus the actual min.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    surface : Surface
        Bounding surface to penalize distance to.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``bounds=(1,np.inf)``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``bounds=(1,np.inf)``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    surface_grid : Grid, optional
        Collocation grid containing the nodes to evaluate surface geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    plasma_grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    use_softmin: bool, optional
        Use softmin or hard min.
    surface_fixed: bool, optional
        Whether the surface the distance from the plasma is computed to
        is fixed or not. If True, the surface is fixed and its coordinates are
        precomputed, which saves on computation time during optimization, and
        self.things = [eq] only.
        If False, the surface coordinates are computed at every iteration.
        False by default, so that self.things = [eq, surface]
    alpha: float, optional
        Parameter used for softmin. The larger alpha, the closer the softmin
        approximates the hardmin. softmin -> hardmin as alpha -> infinity.
        if alpha*array < 1, the underlying softmin will automatically multiply
        the array by 2/min_val to ensure that alpha*array>1. Making alpha larger
        than this minimum value will make the softmin a more accurate approximation
        of the true min.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "rtz"
    _units = "(m)"
    _print_value_fmt = "Plasma-vessel distance: {:10.3e} "

    def __init__(
        self,
        eq,
        surface,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        surface_grid=None,
        plasma_grid=None,
        use_softmin=False,
        surface_fixed=False,
        alpha=1.0,
        name="plasma-vessel distance",
    ):
        if target is None and bounds is None:
            bounds = (1, np.inf)
        self._surface = surface
        self._surface_grid = surface_grid
        self._plasma_grid = plasma_grid
        self._use_softmin = use_softmin
        self._surface_fixed = surface_fixed
        self._alpha = alpha
        super().__init__(
            things=[eq, self._surface] if not surface_fixed else [eq],
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
        surface = self._surface if self._surface_fixed else self.things[1]
        # if things[1] is different than self._surface, update self._surface
        if surface != self._surface:
            self._surface = surface
        if self._surface_grid is None:
            surface_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            surface_grid = self._surface_grid
        if self._plasma_grid is None:
            plasma_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            plasma_grid = self._plasma_grid
        warnif(
            not np.allclose(surface_grid.nodes[:, 0], 1),
            UserWarning,
            "Surface grid includes off-surface pts, should be rho=1.",
        )
        warnif(
            not np.allclose(plasma_grid.nodes[:, 0], 1),
            UserWarning,
            "Plasma grid includes interior points, should be rho=1.",
        )

        self._dim_f = surface_grid.num_nodes
        self._equil_data_keys = ["R", "phi", "Z"]
        self._surface_data_keys = ["x"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        equil_profiles = get_profiles(
            self._equil_data_keys,
            obj=eq,
            grid=plasma_grid,
            has_axis=plasma_grid.axis.size,
        )
        equil_transforms = get_transforms(
            self._equil_data_keys,
            obj=eq,
            grid=plasma_grid,
            has_axis=plasma_grid.axis.size,
        )
        surface_transforms = get_transforms(
            self._surface_data_keys,
            obj=surface,
            grid=surface_grid,
            has_axis=surface_grid.axis.size,
        )

        # compute returns points on the grid of the surface
        # (dim_f = surface_grid.num_nodes)
        # so set quad_weights to the surface grid
        # to avoid it being incorrectly set in the super build
        w = surface_grid.weights
        w *= jnp.sqrt(surface_grid.num_nodes)

        self._constants = {
            "equil_transforms": equil_transforms,
            "equil_profiles": equil_profiles,
            "surface_transforms": surface_transforms,
            "quad_weights": w,
        }

        if self._surface_fixed:
            # precompute the surface coordinates
            # as the surface is fixed during the optimization
            surface_coords = compute_fun(
                self._surface,
                self._surface_data_keys,
                params=self._surface.params_dict,
                transforms=surface_transforms,
                profiles={},
            )["x"]
            surface_coords = rpz2xyz(surface_coords)
            self._constants["surface_coords"] = surface_coords

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, equil_params, surface_params=None, constants=None):
        """Compute plasma-surface distance.

        Parameters
        ----------
        equil_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        surface_params : dict
            Dictionary of surface degrees of freedom, eg Surface.params_dict
            Only needed if self._surface_fixed = False
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        d : ndarray, shape(surface_grid.num_nodes,)
            For each point in the surface grid, approximate distance to plasma.

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._equil_data_keys,
            params=equil_params,
            transforms=constants["equil_transforms"],
            profiles=constants["equil_profiles"],
        )
        plasma_coords = rpz2xyz(jnp.array([data["R"], data["phi"], data["Z"]]).T)
        if self._surface_fixed:
            surface_coords = constants["surface_coords"]
        else:
            surface_coords = compute_fun(
                self._surface,
                self._surface_data_keys,
                params=surface_params,
                transforms=constants["surface_transforms"],
                profiles={},
            )["x"]
            surface_coords = rpz2xyz(surface_coords)
        d = safenorm(plasma_coords[:, None, :] - surface_coords[None, :, :], axis=-1)

        if self._use_softmin:  # do softmin
            return jnp.apply_along_axis(softmin, 0, d, self._alpha)
        else:  # do hardmin
            return d.min(axis=0)


class MeanCurvature(_Objective):
    """Target a particular value for the mean curvature.

    The mean curvature H of a surface is an extrinsic measure of curvature that locally
    describes the curvature of an embedded surface in Euclidean space.

    Positive mean curvature generally corresponds to "concave" regions of the plasma
    boundary which may be difficult to create with coils or magnets.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium or FourierRZToroidalSurface that
        will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``bounds=(-np.inf, 0)``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``bounds=(-np.inf, 0)``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
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
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(m^-1)"
    _print_value_fmt = "Mean curvature: {:10.3e} "

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
        name="mean curvature",
    ):
        if target is None and bounds is None:
            bounds = (-np.inf, 0)
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
            grid = LinearGrid(  # getattr statements in case a surface is passed in
                M=getattr(eq, "M_grid", eq.M * 2),
                N=getattr(eq, "N_grid", eq.N * 2),
                NFP=eq.NFP,
                sym=eq.sym,
            )
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["curvature_H_rho"]

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
            self._normalization = 1 / scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute mean curvature.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom,
            eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        H : ndarray
            Mean curvature at each point (m^-1).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self.things[0],
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return data["curvature_H_rho"]


class PrincipalCurvature(_Objective):
    """Target a particular value for the (unsigned) principal curvature.

    The two principal curvatures at a given point of a surface are the maximum and
    minimum values of the curvature as expressed by the eigenvalues of the shape
    operator at that point. They measure how the surface bends by different amounts in
    different directions at that point.

    This objective targets the maximum absolute value of the two principal curvatures.
    Principal curvature with large absolute value indicates a tight radius of curvature
    which may be difficult to obtain with coils or magnets.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium or FourierRZToroidalSurface that
        will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=1``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=1``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
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
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(m^-1)"
    _print_value_fmt = "Principal curvature: {:10.3e} "

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
        name="principal-curvature",
    ):
        if target is None and bounds is None:
            target = 1
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
            grid = LinearGrid(  # getattr statements in case a surface is passed in
                M=getattr(eq, "M_grid", eq.M * 2),
                N=getattr(eq, "N_grid", eq.N * 2),
                NFP=eq.NFP,
                sym=eq.sym,
            )
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["curvature_k1_rho", "curvature_k2_rho"]

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
            self._normalization = 1 / scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute max absolute principal curvature.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom,
            eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        k : ndarray
            Max absolute principal curvature at each point (m^-1).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self.things[0],
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return jnp.maximum(
            jnp.abs(data["curvature_k1_rho"]), jnp.abs(data["curvature_k2_rho"])
        )


class BScaleLength(_Objective):
    """Target a particular value for the magnetic field scale length.

    The magnetic field scale length, defined as √2 ||B|| / ||∇ 𝐁||, is a length scale
    over which the magnetic field varies. It can be a useful proxy for coil complexity,
    as short length scales require complex coils that are close to the plasma surface.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``bounds=(1,np.inf)``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``bounds=(1,np.inf)``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
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
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(m)"
    _print_value_fmt = "Magnetic field scale length: {:10.3e} "

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
        name="B-scale-length",
    ):
        if target is None and bounds is None:
            bounds = (1, np.inf)
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
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["L_grad(B)"]

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
            self._normalization = scales["R0"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute magnetic field scale length.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        L : ndarray
            Magnetic field scale length at each point (m).

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
        return data["L_grad(B)"]


class GoodCoordinates(_Objective):
    """Target "good" coordinates, meaning non self-intersecting curves.

    Uses a method by Z. Tecchiolli et al, minimizing

    1/ρ² ||√g||² + σ ||𝐞ᵨ||²

    where √g is the jacobian of the coordinate system and 𝐞ᵨ is the covariant radial
    basis vector.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    sigma : float
        Relative weight between the Jacobian and radial terms.
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
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coordinate goodness : {:10.3e} "

    def __init__(
        self,
        eq,
        sigma=1,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coordinate goodness",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._sigma = sigma
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
            grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = 2 * grid.num_nodes
        self._data_keys = ["sqrt(g)", "g_rr", "rho"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "quad_weights": np.sqrt(np.concatenate([grid.weights, grid.weights])),
            "sigma": self._sigma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["V"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coordinate goodness error.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        err : ndarray
            coordinate goodness error, (m^6)

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

        g = jnp.where(data["rho"] == 0, 0, data["sqrt(g)"] ** 2 / data["rho"] ** 2)
        f = data["g_rr"]

        return jnp.concatenate([g, constants["sigma"] * f])
