import numbers

import numpy as np

from desc.backend import (
    jnp,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import compute as compute_fun
from desc.grid import LinearGrid, _Grid
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
        normalization. Operates over all coils, not each individial coil.
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
        from desc.coils import CoilSet, MixedCoilSet, _Coil

        self._dim_f = 0
        self._quad_weights = jnp.array([])

        def get_dim_f_and_weights(coilset):
            """Get dim_f and quad_weights from grid."""
            if isinstance(coilset, list):
                [get_dim_f_and_weights(x) for x in coilset]
            elif isinstance(coilset, MixedCoilSet):
                [get_dim_f_and_weights(x) for x in coilset]
            elif isinstance(coilset, CoilSet):
                get_dim_f_and_weights(coilset.coils)
            elif isinstance(coilset, _Grid):
                self._dim_f += coilset.num_zeta
                self._quad_weights = jnp.concatenate(
                    (self._quad_weights, coilset.spacing[:, 2])
                )

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

        is_single_coil = lambda x: isinstance(x, _Coil) and not isinstance(x, CoilSet)
        # gives structure of coils, e.g. MixedCoilSet(coils, coils) would give a
        # a structure of [[*, *], [*, *]] if n = 2 coils
        coil_structure = tree_structure(
            self.things[0],
            is_leaf=lambda x: is_single_coil(x),
        )
        coil_leaves = tree_leaves(self.things[0], is_leaf=lambda x: is_single_coil(x))

        # check type
        if isinstance(self._grid, numbers.Integral):
            self._grid = LinearGrid(N=self._grid, endpoint=False)
        # all of these cases return a container MixedCoilSet that contains
        # LinearGrids. i.e. MixedCoilSet.coils = list of LinearGrid
        if self._grid is None:
            # map default grid to structure of inputted coils
            self._grid = tree_map(
                lambda x: LinearGrid(
                    N=2 * x.N + 5, NFP=getattr(x, "NFP", 1), endpoint=False
                ),
                self.things[0],
                is_leaf=lambda x: is_single_coil(x),
            )
        elif isinstance(self._grid, _Grid):
            # map inputted single LinearGrid to structure of inputted coils
            self._grid = [self._grid] * len(coil_leaves)
            self._grid = tree_unflatten(coil_structure, self._grid)
        else:
            # this case covers an inputted list of grids that matches the size
            # of the inputted coils. Can be a 1D list or nested list.
            flattened_grid = tree_flatten(
                self._grid, is_leaf=lambda x: isinstance(x, _Grid)
            )[0]
            self._grid = tree_unflatten(coil_structure, flattened_grid)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        transforms = tree_map(
            lambda x, y: get_transforms(self._data_keys, obj=x, grid=y),
            self.things[0],
            self._grid,
            is_leaf=lambda x: is_single_coil(x),
        )

        get_dim_f_and_weights(self._grid)
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
            self._scales = compute_scaling_factors(coil_leaves[0])

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
        be flattened according to the number of inputs.
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
        normalization. Operates over all coils, not each individial coil.
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
        from desc.coils import CoilSet, _Coil

        super().build(use_jit=use_jit, verbose=verbose)

        if self._normalize:
            self._normalization = self._scales["a"]

        # TODO: repeated code but maybe it's fine
        flattened_coils = tree_flatten(
            self._coils,
            is_leaf=lambda x: isinstance(x, _Coil) and not isinstance(x, CoilSet),
        )[0]
        flattened_coils = (
            [flattened_coils[0]]
            if not isinstance(self._coils, CoilSet)
            else flattened_coils
        )
        self._dim_f = len(flattened_coils)
        self._constants["quad_weights"] = 1

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
        f : float or array of floats
            Coil length.
        """
        data = super().compute(params, constants=constants)
        data = tree_flatten(data, is_leaf=lambda x: isinstance(x, dict))[0]
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
        be flattened according to the number of inputs.
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
        normalization. Operates over all coils, not each individial coil.
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
            self._normalization = 1 / self._scales["a"]

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
        data = tree_flatten(data, is_leaf=lambda x: isinstance(x, dict))[0]
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
        be flattened according to the number of inputs.
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
        normalization. Operates over all coils, not each individial coil.
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
            self._normalization = 1 / self._scales["a"]

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
        f : float or array of floats
            Coil torsion.
        """
        data = super().compute(params, constants=constants)
        data = tree_flatten(data, is_leaf=lambda x: isinstance(x, dict))[0]
        out = jnp.concatenate([dat["torsion"] for dat in data])
        return out


class ToroidalFlux(_Objective):
    """Target the toroidal flux in an equilibrium from a magnetic field.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
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
        Note: has no effect on this objective.
        FIXME: add normalization for the B part of this objective
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField). Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at. Defaults to a LinearGrid(L=eq.L_grid, M=eq.M_grid,
        zeta=jnp.array(0.0), NFP=eq.NFP).
    external_field : MagneticField, optional
        MagneticField object containing the external field to consider when
        minimizing the Bn errors. If None, the external field is assumed to be zero.
        e.g. this could be a 1/R field representing external TF coils, or
        it could be set of discrete TF coils so that coil ripple is considered during
        the optimization of the ``field`` object.
    external_field_source_grid : Grid, optional
        Grid object used to discretize the external field source. Defaults to the
        default for the given external_field, see the docstring of the field object
        for the specific default.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "rtz"
    _units = "T"
    _print_value_fmt = "Quadratic Flux: {:10.3e} "

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
        source_grid=None,
        eval_grid=None,
        external_field=None,
        external_field_source_grid=None,
        name="toroidal-flux",
        eq_fixed=False,
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._field = field
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq_fixed = eq_fixed
        self._eq = eq if eq_fixed else None
        self._external_field = external_field
        self._external_field_source_grid = external_field_source_grid

        super().__init__(
            things=[field, eq] if not eq_fixed else [field],
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
        eq = self._eq if self._eq_fixed else self.things[1]
        if self._eval_grid is None:
            eval_grid = LinearGrid(
                L=eq.L_grid, M=eq.M_grid, zeta=jnp.array(0.0), NFP=eq.NFP
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        warnif(
            not np.allclose(eval_grid.nodes[:, 2], eval_grid.nodes[0, 2]),
            UserWarning,
            "Evaluation grid should be at constant zeta",
        )

        # ensure vacuum eq, as we don't yet support finite beta
        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure is non-zero (max {pres} Pa), "
            + "finite beta not supported yet.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current is non-zero (max {curr} A), "
            + "finite plasma currents not supported yet.",
        )

        # eval_grid.num_nodes for quad flux cost,
        self._dim_f = 1
        self._equil_data_keys = ["|e_rho x e_theta|", "R", "phi", "Z"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        equil_profiles = get_profiles(
            self._equil_data_keys,
            obj=eq,
            grid=eval_grid,
            has_axis=eval_grid.axis.size,
        )
        equil_transforms = get_transforms(
            self._equil_data_keys,
            obj=eq,
            grid=eval_grid,
            has_axis=eval_grid.axis.size,
        )

        if self._eq_fixed:
            data = eq.compute(["R", "phi", "Z", "|e_rho x e_theta|"], grid=eval_grid)

            plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T

        if not self._eq_fixed:
            self._constants = {
                "equil_transforms": equil_transforms,
                "equil_profiles": equil_profiles,
                "quad_weights": 1.0,
            }
        else:
            self._constants = {
                "equil_transforms": equil_transforms,
                "equil_profiles": equil_profiles,
                "plasma_coords": plasma_coords,
                "equil_data": data,
                "quad_weights": 1.0,
            }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1=None, params_2=None, constants=None):
        """Compute toroidal flux.

        Parameters
        ----------
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotential.params_dict
        equil_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
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
        if self._eq_fixed:
            field_params = params_1
        else:
            field_params = params_2
            equil_params = params_1
        if not self._eq_fixed:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._equil_data_keys,
                params=equil_params,
                transforms=constants["equil_transforms"],
                profiles=constants["equil_profiles"],
            )
            plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T

        else:
            data = constants["equil_data"]
            plasma_coords = constants["plasma_coords"]

        B = self._field.compute_magnetic_field(
            plasma_coords,
            basis="rpz",
            source_grid=self._source_grid,
            params=field_params,
        )
        grid = self._eval_grid
        Psi = jnp.sum(
            grid.spacing[:, 0]
            * grid.spacing[:, 1]
            * data["|e_rho x e_theta|"]
            * B[:, 1]
        )

        if self._external_field is not None:
            B_ext = self._external_field.compute_magnetic_field(
                plasma_coords,
                grid=self._external_field_source_grid,
                basis="rpz",
            )
            Psi += jnp.sum(
                grid.spacing[:, 0]
                * grid.spacing[:, 1]
                * data["|e_rho x e_theta|"]
                * B_ext[:, 1]
            )

        return Psi
