import numbers

import numpy as np

from desc.backend import (
    fori_loop,
    jnp,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_unflatten,
)
from desc.compute import get_profiles, get_transforms, rpz2xyz
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid, _Grid
from desc.integrals import compute_B_plasma
from desc.utils import Timer, errorif, safenorm, warnif

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
        Both bounds must be broadcastable to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
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
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, list, optional
        Collocation grid containing the nodes to evaluate at.
        If a list, must have the same structure as coil.
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

        def _is_single_coil(c):
            return isinstance(c, _Coil) and not isinstance(c, CoilSet)

        def _prune_coilset_tree(coilset):
            """Remove extra members from CoilSets (but not MixedCoilSets)."""
            if isinstance(coilset, list) or isinstance(coilset, MixedCoilSet):
                return [_prune_coilset_tree(c) for c in coilset]
            elif isinstance(coilset, CoilSet):
                # CoilSet only uses a single grid/transform for all coils
                return _prune_coilset_tree(coilset.coils[0])
            else:
                return coilset  # single coil

        coil = self.things[0]
        grid = self._grid

        # get individual coils from coilset
        coils, structure = tree_flatten(coil, is_leaf=_is_single_coil)
        for c in coils:
            errorif(
                not isinstance(c, _Coil),
                TypeError,
                f"Expected object of type Coil, got {type(c)}",
            )
        self._num_coils = len(coils)

        # map grid to list of length coils
        if grid is None:
            grid = []
            for c in coils:
                grid.append(LinearGrid(N=2 * c.N * getattr(c, "NFP", 1) + 5))
        if isinstance(grid, numbers.Integral):
            grid = LinearGrid(N=self._grid)
        if isinstance(grid, _Grid):
            grid = [grid] * self._num_coils
        if isinstance(grid, list):
            grid = tree_leaves(grid, is_leaf=lambda g: isinstance(g, _Grid))

        errorif(
            len(grid) != len(coils),
            ValueError,
            "grid input must be broadcastable to the coil structure.",
        )
        errorif(
            np.any([g.num_rho > 1 or g.num_theta > 1 for g in grid]),
            ValueError,
            "Only use toroidal resolution for coil grids.",
        )

        self._dim_f = np.sum([g.num_nodes for g in grid])
        quad_weights = np.concatenate([g.spacing[:, 2] for g in grid])

        # map grid to the same structure as coil and then remove unnecessary members
        grid = tree_unflatten(structure, grid)
        grid = _prune_coilset_tree(grid)
        coil = _prune_coilset_tree(coil)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        transforms = tree_map(
            lambda c, g: get_transforms(self._data_keys, obj=c, grid=g),
            coil,
            grid,
            is_leaf=lambda x: _is_single_coil(x) or isinstance(x, _Grid),
        )

        self._grid = grid
        self._constants = {"transforms": transforms, "quad_weights": quad_weights}

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            self._scales = [compute_scaling_factors(coil) for coil in coils]

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

        coil = self.things[0]
        data = coil.compute(
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
        Both bounds must be broadcastable to Objective.dim_f.
        Defaults to ``target=2*np.pi``.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
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
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
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
    _print_value_fmt = "Coil length: "

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
        name="coil length",
    ):
        if target is None and bounds is None:
            target = 2 * np.pi

        super().__init__(
            coil,
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

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

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

    Targets the local curvature at each grid node for each coil.
    Positive curvature corresponds to "convex" curves (a circle has positive curvature),
    while negative curvature corresponds to "concave" curves.
    Curvature values closer to 0 indicate straighter sections of coils.

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
        Both bounds must be broadcastable to Objective.dim_f.
        Defaults to ``bounds=(0,1)``.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
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
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
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
    _print_value_fmt = "Coil curvature: "

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

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

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

    Targets the local torsion value at each grid node for each coil. Indicative of how
    non-planar the coil is (a torsion value of 0 means the coil is perfectly planar).

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
        Both bounds must be broadcastable to Objective.dim_f.
        Defaults to ``target=0``.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
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
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
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
    _print_value_fmt = "Coil torsion: "

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

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

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
        Both bounds must be broadcastable to Objective.dim_f.
        Defaults to ``target=0``.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
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
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
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
    _print_value_fmt = "Coil current length: "

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

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

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
        currents = jnp.concatenate([param["current"] for param in params])
        out = jnp.atleast_1d(lengths * currents)
        return out


class CoilSetMinDistance(_Objective):
    """Target the minimum distance between coils in a coilset.

    Will yield one value per coil in the coilset, which is the minimum distance to
    another coil in that coilset.

    Parameters
    ----------
    coil : CoilSet
        Coil(s) that are to be optimized.
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
    grid : Grid, list, optional
        Collocation grid used to discretize each coil. Defaults to the default grid
        for the given coil-type, see ``coils.py`` and ``curve.py`` for more details.
        If a list, must have the same structure as coils.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _units = "(m)"
    _print_value_fmt = "Minimum coil-coil distance: "

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
        name="coil-coil minimum distance",
    ):
        from desc.coils import CoilSet

        if target is None and bounds is None:
            bounds = (1, np.inf)
        self._grid = grid
        errorif(
            not isinstance(coil, CoilSet),
            ValueError,
            "coil must be of type CoilSet, not an individual Coil",
        )
        super().__init__(
            things=coil,
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
        coilset = self.things[0]
        grid = self._grid or None

        self._dim_f = coilset.num_coils
        self._constants = {"coilset": coilset, "grid": grid, "quad_weights": 1.0}

        if self._normalize:
            coils = tree_leaves(coilset, is_leaf=lambda x: not hasattr(x, "__len__"))
            scales = [compute_scaling_factors(coil)["a"] for coil in coils]
            self._normalization = np.mean(scales)  # mean length of coils

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute minimum distances between coils.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants.

        Returns
        -------
        f : array of floats
            Minimum distance to another coil for each coil in the coilset.

        """
        if constants is None:
            constants = self.constants
        pts = constants["coilset"]._compute_position(
            params=params, grid=constants["grid"], basis="xyz"
        )

        def body(k):
            # dist btwn all pts; shape(ncoils,num_nodes,num_nodes)
            # dist[i,j,n] is the distance from the jth point on the kth coil
            # to the nth point on the ith coil
            dist = safenorm(pts[k][None, :, None] - pts[:, None, :], axis=-1)
            # exclude distances between points on the same coil
            mask = jnp.ones(self.dim_f).at[k].set(0)[:, None, None]
            return jnp.min(dist, where=mask, initial=jnp.inf)

        min_dist_per_coil = fori_loop(
            0,
            self.dim_f,
            lambda k, min_dist: min_dist.at[k].set(body(k)),
            jnp.zeros(self.dim_f),
        )
        return min_dist_per_coil


class PlasmaCoilSetMinDistance(_Objective):
    """Target the minimum distance between the plasma and coilset.

    Will yield one value per coil in the coilset, which is the minimum distance from
    that coil to the plasma boundary surface.

    NOTE: By default, assumes the plasma boundary is not fixed and its coordinates are
    computed at every iteration, for example if the equilibrium is changing in a
    single-stage optimization.
    If the plasma boundary is fixed, set eq_fixed=True to precompute the last closed
    flux surface coordinates and improve the efficiency of the calculation.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium (or FourierRZToroidalSurface) that will be optimized
        to satisfy the Objective.
    coil : CoilSet
        Coil(s) that are to be optimized.
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
    plasma_grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    coil_grid : Grid, list, optional
        Collocation grid containing the nodes to evaluate coilset geometry at.
        Defaults to the default grid for the given coil-type, see ``coils.py``
        and ``curve.py`` for more details.
        If a list, must have the same structure as coils.
    eq_fixed: bool, optional
        Whether the equilibrium is fixed or not. If True, the last closed flux surface
        is fixed and its coordinates are precomputed, which saves on computation time
        during optimization, and self.things = [coil] only.
        If False, the surface coordinates are computed at every iteration.
        False by default, so that self.things = [coil, eq].
    coils_fixed: bool, optional
        Whether the coils are fixed or not. If True, the coils
        are fixed and their coordinates are precomputed, which saves on computation time
        during optimization, and self.things = [eq] only.
        If False, the coil coordinates are computed at every iteration.
        False by default, so that self.things = [coil, eq].
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _units = "(m)"
    _print_value_fmt = "Minimum plasma-coil distance: "

    def __init__(
        self,
        eq,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        plasma_grid=None,
        coil_grid=None,
        eq_fixed=False,
        coils_fixed=False,
        name="plasma-coil minimum distance",
    ):
        if target is None and bounds is None:
            bounds = (1, np.inf)
        self._eq = eq
        self._coil = coil
        self._plasma_grid = plasma_grid
        self._coil_grid = coil_grid
        self._eq_fixed = eq_fixed
        self._coils_fixed = coils_fixed
        errorif(eq_fixed and coils_fixed, ValueError, "Cannot fix both eq and coil")
        things = []
        if not eq_fixed:
            things.append(eq)
        if not coils_fixed:
            things.append(coil)
        super().__init__(
            things=things,
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
        if self._eq_fixed:
            eq = self._eq
            coil = self.things[0]
        elif self._coils_fixed:
            eq = self.things[0]
            coil = self._coil
        else:
            eq = self.things[0]
            coil = self.things[1]
        plasma_grid = self._plasma_grid or LinearGrid(M=eq.M_grid, N=eq.N_grid)
        coil_grid = self._coil_grid or None
        warnif(
            not np.allclose(plasma_grid.nodes[:, 0], 1),
            UserWarning,
            "Plasma/Surface grid includes interior points, should be rho=1.",
        )

        self._dim_f = coil.num_coils
        self._eq_data_keys = ["R", "phi", "Z"]

        eq_profiles = get_profiles(self._eq_data_keys, obj=eq, grid=plasma_grid)
        eq_transforms = get_transforms(self._eq_data_keys, obj=eq, grid=plasma_grid)

        self._constants = {
            "eq": eq,
            "coil": coil,
            "coil_grid": coil_grid,
            "eq_profiles": eq_profiles,
            "eq_transforms": eq_transforms,
            "quad_weights": 1.0,
        }

        if self._eq_fixed:
            # precompute the equilibrium surface coordinates
            data = compute_fun(
                eq,
                self._eq_data_keys,
                params=eq.params_dict,
                transforms=eq_transforms,
                profiles=eq_profiles,
            )
            plasma_pts = rpz2xyz(jnp.array([data["R"], data["phi"], data["Z"]]).T)
            self._constants["plasma_coords"] = plasma_pts
        if self._coils_fixed:
            coils_pts = coil._compute_position(params=coil.params_dict, grid=coil_grid)
            self._constants["coil_coords"] = coils_pts

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute minimum distance between coils and the plasma/surface.

        Parameters
        ----------
        params_1 : dict
            Dictionary of coilset degrees of freedom, eg ``CoilSet.params_dict`` if
            self._coils_fixed is False, else is the equilibrium or surface degrees of
            freedom
        params_2 : dict
            Dictionary of equilibrium or surface degrees of freedom,
            eg ``Equilibrium.params_dict``
            Only required if ``self._eq_fixed = False``.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants.

        Returns
        -------
        f : array of floats
            Minimum distance from coil to surface for each coil in the coilset.

        """
        if constants is None:
            constants = self.constants
        if self._eq_fixed:
            coils_params = params_1
        elif self._coils_fixed:
            eq_params = params_1
        else:
            eq_params = params_1
            coils_params = params_2

        # coil pts; shape(ncoils,coils_grid.num_nodes,3)
        if self._coils_fixed:
            coils_pts = constants["coil_coords"]
        else:
            coils_pts = constants["coil"]._compute_position(
                params=coils_params, grid=constants["coil_grid"]
            )

        # plasma pts; shape(plasma_grid.num_nodes,3)
        if self._eq_fixed:
            plasma_pts = constants["plasma_coords"]
        else:
            data = compute_fun(
                constants["eq"],
                self._eq_data_keys,
                params=eq_params,
                transforms=constants["eq_transforms"],
                profiles=constants["eq_profiles"],
            )
            plasma_pts = rpz2xyz(jnp.array([data["R"], data["phi"], data["Z"]]).T)

        def body(k):
            # dist btwn all pts; shape(ncoils,plasma_grid.num_nodes,coil_grid.num_nodes)
            dist = safenorm(coils_pts[k][None, :, :] - plasma_pts[:, None, :], axis=-1)
            return jnp.min(dist, initial=jnp.inf)

        min_dist_per_coil = fori_loop(
            0,
            self.dim_f,
            lambda k, min_dist: min_dist.at[k].set(body(k)),
            jnp.zeros(self.dim_f),
        )
        return min_dist_per_coil


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
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid,
        NFP=eq.NFP, sym=False)``
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
    _print_value_fmt = "Boundary normal field error: "
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
                rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
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
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=field_params,
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

    Will try to use the vector potential method to calculate the toroidal flux
    (Φ = ∮ 𝐀 ⋅ 𝐝𝐥 over the perimeter of a constant zeta plane)
    instead of the brute force method using the magnetic field
    (Φ = ∯ 𝐁 ⋅ 𝐝𝐒 over a constant zeta XS). The vector potential method
    is much more efficient, however not every ``MagneticField`` object
    has a vector potential available to compute, so in those cases
    the magnetic field method is used.

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
        Both bounds must be broadcastable to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
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
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
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
    _print_value_fmt = "Toroidal Flux: "

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
        # TODO: add eq_fixed option so this can be used in single stage

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
        self._use_vector_potential = True
        try:
            self._field.compute_magnetic_vector_potential([0, 0, 0])
        except (NotImplementedError, ValueError):
            self._use_vector_potential = False
        if self._eval_grid is None:
            eval_grid = LinearGrid(
                L=eq.L_grid if not self._use_vector_potential else 0,
                M=eq.M_grid,
                zeta=jnp.array(0.0),
                NFP=eq.NFP,
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
        data_keys = ["R", "phi", "Z"]
        if self._use_vector_potential:
            data_keys += ["e_theta"]
        else:
            data_keys += ["|e_rho x e_theta|", "n_zeta"]
        data = eq.compute(data_keys, grid=eval_grid)

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
        grid = constants["eval_grid"]

        if self._use_vector_potential:
            A = constants["field"].compute_magnetic_vector_potential(
                plasma_coords,
                basis="rpz",
                source_grid=constants["field_grid"],
                params=field_params,
            )

            A_dot_e_theta = jnp.sum(A * data["e_theta"], axis=1)
            Psi = jnp.sum(grid.spacing[:, 1] * A_dot_e_theta)
        else:
            B = constants["field"].compute_magnetic_field(
                plasma_coords,
                basis="rpz",
                source_grid=constants["field_grid"],
                params=field_params,
            )

            B_dot_n_zeta = jnp.sum(B * data["n_zeta"], axis=1)
            Psi = jnp.sum(
                grid.spacing[:, 0]
                * grid.spacing[:, 1]
                * data["|e_rho x e_theta|"]
                * B_dot_n_zeta
            )

        return Psi
