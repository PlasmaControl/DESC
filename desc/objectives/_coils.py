import numbers

import numpy as np

from desc.backend import jnp, tree_flatten, tree_leaves, tree_map, tree_unflatten
from desc.compute import compute as compute_fun
from desc.compute import get_profiles, get_transforms
from desc.grid import LinearGrid, _Grid
from desc.singularities import compute_B_plasma
from desc.utils import Timer, errorif, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class _CoilObjective(_Objective):
    """Base class for calculating coil objectives.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil for which the data keys will be optimized.
    data_keys : list of str
        data keys that will be optimized when this class is inherited.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of float, ndarray, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : float, ndarray, optional
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
        normalization. Operates over all coils, not each individual coil.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, list, optional
        Collocation grid containing the nodes to evaluate at. If list, has to adhere to
        Objective.dim_f
    name : str, optional
        Name of the objective function.

    """

    def __init__(
        self,
        coil,
        data_keys,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name=None,
    ):
        self._grid = grid
        self._data_keys = data_keys
        self._normalize = normalize
        super().__init__(
            things=[coil],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):  # noqa:C901
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        # local import to avoid circular import
        from desc.coils import CoilSet, MixedCoilSet

        self._dim_f = 0
        self._quad_weights = jnp.array([])

        def to_list(coilset):
            """Turn a MixedCoilSet container into a list of what it's containing."""
            if isinstance(coilset, list):
                return [to_list(x) for x in coilset]
            elif isinstance(coilset, MixedCoilSet):
                return [to_list(x) for x in coilset]
            elif isinstance(coilset, CoilSet):
                # use the same grid/transform for CoilSet
                return to_list(coilset.coils[0])
            else:
                return [coilset]

        # gives structure of coils, e.g. MixedCoilSet(coils, coils) would give a
        # a structure of [[*, *], [*, *]] if n = 2 coils
        coil_leaves, coil_structure = tree_flatten(
            self.things[0], is_leaf=lambda x: not hasattr(x, "__len__")
        )
        self._num_coils = len(coil_leaves)

        # check type
        if isinstance(self._grid, numbers.Integral):
            self._grid = LinearGrid(N=self._grid, endpoint=False)
        # all of these cases return a container MixedCoilSet that contains
        # LinearGrids. i.e. MixedCoilSet.coils = list of LinearGrid
        leaves = tree_leaves(
            self.things[0], is_leaf=lambda x: not hasattr(x, "__len__")
        )
        print(leaves)
        if self._grid is None:
            # map default grid to structure of inputted coils
            self._grid = tree_map(
                lambda x: LinearGrid(
                    N=2 * x.N + 5, NFP=getattr(x, "NFP", 1), endpoint=False
                ),
                self.things[0],
                is_leaf=lambda x: not hasattr(x, "__len__"),
            )
        elif isinstance(self._grid, _Grid):
            # map inputted single LinearGrid to structure of inputted coils
            self._grid = [self._grid] * self._num_coils
            self._grid = tree_unflatten(coil_structure, self._grid)
        else:
            # this case covers an inputted list of grids that matches the size
            # of the inputted coils. Can be a 1D list or nested list.
            flattened_grid = tree_leaves(
                self._grid, is_leaf=lambda x: isinstance(x, _Grid)
            )
            self._grid = tree_unflatten(coil_structure, flattened_grid)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        transforms = tree_map(
            lambda x, y: get_transforms(self._data_keys, obj=x, grid=y),
            self.things[0],
            self._grid,
            is_leaf=lambda x: not hasattr(x, "__len__"),
        )

        grids = tree_leaves(self._grid, is_leaf=lambda x: hasattr(x, "num_nodes"))
        self._dim_f = np.sum([grid.num_nodes for grid in grids])
        self._quad_weights = np.concatenate([grid.spacing[:, 2] for grid in grids])

        # get only needed grids (1 per CoilSet) and flatten that list
        self._grid = tree_leaves(
            to_list(self._grid), is_leaf=lambda x: isinstance(x, _Grid)
        )
        transforms = tree_leaves(
            to_list(transforms), is_leaf=lambda x: isinstance(x, dict)
        )

        errorif(
            np.any([grid.num_rho > 1 or grid.num_theta > 1 for grid in self._grid]),
            ValueError,
            "Only use toroidal resolution for coil grids.",
        )

        # CoilSet and _Coil have one grid/transform
        if not isinstance(self.things[0], MixedCoilSet):
            self._grid = self._grid[0]
            transforms = transforms[0]

        self._constants = {
            "transforms": transforms,
            "quad_weights": self._quad_weights,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            self._scales = [compute_scaling_factors(coil) for coil in coil_leaves]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute data of coil for given data key.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants.

        Returns
        -------
        f : float or array of floats
            Coil length.

        """
        if constants is None:
            constants = self._constants

        coils = self.things[0]
        data = coils.compute(
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            grid=self._grid,
        )

        return data


class CoilLength(_CoilObjective):
    """Coil length.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If array, it has to
        be flattened according to the number of inputs. Defaults to ``target=2*np.pi``.
    bounds : tuple of float, ndarray, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=2*np.pi``.
    weight : float, ndarray, optional
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
        normalization. Operates over all coils, not each individual coil.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``
    name : str, optional
        Name of the objective function.

    """

    _scalar = False  # Not always a scalar, if a coilset is passed in
    _units = "(m)"
    _print_value_fmt = "Coil length: {:10.3e} "

    def __init__(
        self,
        coils,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil length",
    ):
        self._coils = coils
        if target is None and bounds is None:
            target = 2 * np.pi

        super().__init__(
            coils,
            ["length"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._dim_f = self._num_coils
        self._constants["quad_weights"] = 1

        if self._normalize:
            self._normalization = np.mean([scale["a"] for scale in self._scales])

    def compute(self, params, constants=None):
        """Compute coil length.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants.

        Returns
        -------
        f : array of floats
            Coil length.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.array([dat["length"] for dat in data])
        return out


class CoilCurvature(_CoilObjective):
    """Coil curvature.

    Targets the local curvature value per grid node for each coil. A smaller curvature
    value indicates straighter coils. All curvature values are positive.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If array, it has to
        be flattened according to the number of inputs. Defaults to ``bounds=(0,1)``.
    bounds : tuple of float, ndarray, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``bounds=(0,1)``.
    weight : float, ndarray, optional
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
        normalization. Operates over all coils, not each individual coil.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _units = "(m^-1)"
    _print_value_fmt = "Coil curvature: {:10.3e} "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil curvature",
    ):
        if target is None and bounds is None:
            bounds = (0, 1)

        super().__init__(
            coil,
            ["curvature"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        if self._normalize:
            self._normalization = 1 / np.mean([scale["a"] for scale in self._scales])

    def compute(self, params, constants=None):
        """Compute coil curvature.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants.

        Returns
        -------
        f : array of floats
            1D array of coil curvature values.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.concatenate([dat["curvature"] for dat in data])
        return out


class CoilTorsion(_CoilObjective):
    """Coil torsion.

    Targets the local torsion value per grid node for each coil. Indicative
    of how much the coil goes out of the poloidal plane. e.g. a torsion
    value of 0 means the coil is completely planar.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If array, it has to
        be flattened according to the number of inputs. Defaults to ``target=0``.
    bounds : tuple of float, ndarray, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : float, ndarray, optional
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
        normalization. Operates over all coils, not each individual coil.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _units = "(m^-1)"
    _print_value_fmt = "Coil torsion: {:10.3e} "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil torsion",
    ):
        if target is None and bounds is None:
            target = 0

        super().__init__(
            coil,
            ["torsion"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        if self._normalize:
            self._normalization = 1 / np.mean([scale["a"] for scale in self._scales])

    def compute(self, params, constants=None):
        """Compute coil torsion.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants.

        Returns
        -------
        f : array of floats
            Coil torsion.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.concatenate([dat["torsion"] for dat in data])
        return out


class CoilCurrentLength(CoilLength):
    """Coil current length.

    Targets the coil current length, i.e. current * length for each coil.
    Useful for approximating HTS cost.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If array, it has to
        be flattened according to the number of inputs. Defaults to ``target=0``.
    bounds : tuple of float, ndarray, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : float, ndarray, optional
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
        normalization. Operates over all coils, not each individual coil.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _units = "(A*m)"
    _print_value_fmt = "Coil current length: {:10.3e} "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil current length",
    ):
        if target is None and bounds is None:
            target = 0

        super().__init__(
            coil,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._dim_f = self._num_coils
        self._constants["quad_weights"] = 1

        if self._normalize:
            mean_length = np.mean([scale["a"] for scale in self._scales])
            params = tree_leaves(
                self.things[0].params_dict, is_leaf=lambda x: isinstance(x, dict)
            )
            mean_current = np.mean([np.abs(param["current"]) for param in params])
            mean_current = np.max((mean_current, 1))
            self._normalization = mean_current * mean_length

    def compute(self, params, constants=None):
        """Compute coil current length (current * length).

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants.

        Returns
        -------
        f : array of floats

        """
        lengths = super().compute(params, constants=constants)
        params = tree_leaves(params, is_leaf=lambda x: isinstance(x, dict))
        currents = [param["current"] for param in params]
        out = jnp.asarray(lengths) * jnp.asarray(currents)
        return out


class QuadraticFlux(_Objective):
    """Target B*n = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n. The equilibrium is kept fixed while the
    field is unfixed.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error will be minimized.
        The equilibrium is kept fixed during the optimization with this objective.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f.
        Default target is zero.
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    source_grid : Grid, optional
        Collocation grid containing the nodes for plasma source terms.
        Default grid is detailed in the docs for ``compute_B_plasma``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the plasma surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid,
            NFP=int(eq.NFP), sym=False)
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    vacuum : bool
        If true, B_plasma (the contribution to the normal field on the boundary from the
        plasma currents) is set to zero.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field error: {:10.3e} "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        source_grid=None,
        eval_grid=None,
        field_grid=None,
        vacuum=False,
        name="Quadratic flux",
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid
        self._vacuum = vacuum
        things = [field]
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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
        eq = self._eq

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=eq.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )

        # pre-compute B_plasma because we are assuming eq is fixed
        if self._vacuum:
            Bplasma = jnp.zeros(eval_grid.num_nodes)

        else:
            Bplasma = compute_B_plasma(
                eq, eval_grid, self._source_grid, normal_only=True
            )

        self._constants = {
            "field": self._field,
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "B_plasma": Bplasma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, field_params, constants=None):
        """Compute boundary force error.

        Parameters
        ----------
        field_params : dict
            Dictionary of the external field's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Bnorm from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants

        # B_plasma from equilibrium precomputed
        eval_data = constants["eval_data"]
        B_plasma = constants["B_plasma"]

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B_ext is not pre-computed because field is not fixed
        B_ext = constants["field"].compute_magnetic_field(
            x, source_grid=constants["field_grid"], basis="rpz", params=field_params
        )
        B_ext = jnp.sum(B_ext * eval_data["n_rho"], axis=-1)
        f = (B_ext + B_plasma) * eval_data["|e_theta x e_zeta|"]
        return f


class ToroidalFlux(_Objective):
    """Target the toroidal flux in an equilibrium from a magnetic field.

    This objective is needed when performing stage-two coil optimization on
    a vacuum equilibrium, to avoid the trivial solution of minimizing Bn
    by making the coil currents zero. Instead, this objective ensures
    the coils create the necessary toroidal flux for the equilibrium field.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium for which the toroidal flux will be calculated.
        The Equilibrium is assumed to be held fixed when using this
        objective.
    field : MagneticField
        MagneticField object, the parameters of this will be optimized
        to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Defaults to eq.Psi. Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
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
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField). Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at. Defaults to a LinearGrid(L=eq.L_grid, M=eq.M_grid,
        zeta=jnp.array(0.0), NFP=eq.NFP).
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(Wb)"
    _print_value_fmt = "Toroidal Flux: {:10.3e} "

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        field_grid=None,
        eval_grid=None,
        name="toroidal-flux",
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._field = field
        self._field_grid = field_grid
        self._eval_grid = eval_grid
        self._eq = eq

        super().__init__(
            things=[field],
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
        eq = self._eq
        if self._eval_grid is None:
            eval_grid = LinearGrid(
                L=eq.L_grid, M=eq.M_grid, zeta=jnp.array(0.0), NFP=eq.NFP
            )
            self._eval_grid = eval_grid
        eval_grid = self._eval_grid

        errorif(
            not np.allclose(eval_grid.nodes[:, 2], eval_grid.nodes[0, 2]),
            ValueError,
            "Evaluation grid should be at constant zeta",
        )
        if self._normalize:
            self._normalization = eq.Psi

        # ensure vacuum eq, as is unneeded for finite beta
        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure appears to be non-zero (max {pres} Pa), "
            + "this objective is unneeded at finite beta.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current appears to be non-zero (max {curr} A), "
            + "this objective is unneeded at finite beta.",
        )

        # eval_grid.num_nodes for quad flux cost,
        self._dim_f = 1
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        data = eq.compute(
            ["R", "phi", "Z", "|e_rho x e_theta|", "n_zeta"], grid=eval_grid
        )

        plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T

        self._constants = {
            "plasma_coords": plasma_coords,
            "equil_data": data,
            "quad_weights": 1.0,
            "field": self._field,
            "field_grid": self._field_grid,
            "eval_grid": eval_grid,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, field_params=None, constants=None):
        """Compute toroidal flux.

        Parameters
        ----------
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotential.params_dict or CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : float
            Toroidal flux from coils and external field

        """
        if constants is None:
            constants = self.constants

        data = constants["equil_data"]
        plasma_coords = constants["plasma_coords"]

        B = constants["field"].compute_magnetic_field(
            plasma_coords,
            basis="rpz",
            source_grid=constants["field_grid"],
            params=field_params,
        )
        grid = constants["eval_grid"]

        B_dot_n_zeta = jnp.sum(B * data["n_zeta"], axis=1)

        Psi = jnp.sum(
            grid.spacing[:, 0]
            * grid.spacing[:, 1]
            * data["|e_rho x e_theta|"]
            * B_dot_n_zeta
        )

        return Psi
