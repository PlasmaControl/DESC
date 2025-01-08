import numbers
import warnings

import numpy as np
from scipy.constants import mu_0

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
from desc.utils import Timer, broadcast_tree, errorif, safenorm, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


class _CoilObjective(_Objective):
    """Base class for calculating coil objectives.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil for which the data keys will be optimized.
    data_keys : list of str
        data keys that will be optimized when this class is inherited.
    grid : Grid, list, optional
        Collocation grid containing the nodes to evaluate at.
        If a list, must have the same structure as coil.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        coil=True,
    )

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
        jac_chunk_size=None,
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
            jac_chunk_size=jac_chunk_size,
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
            Coil objective value(s).

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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=2*np.pi``.",
        bounds_default="``target=2*np.pi``.",
        coil=True,
    )

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
        jac_chunk_size=None,
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
            jac_chunk_size=jac_chunk_size,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(0,1).``",
        bounds_default="``bounds=(0,1).``",
        coil=True,
    )

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
        jac_chunk_size=None,
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
            jac_chunk_size=jac_chunk_size,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

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
        jac_chunk_size=None,
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
            jac_chunk_size=jac_chunk_size,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

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
        jac_chunk_size=None,
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
            jac_chunk_size=jac_chunk_size,
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
    grid : Grid, list, optional
        Collocation grid used to discretize each coil. Defaults to the default grid
        for the given coil-type, see ``coils.py`` and ``curve.py`` for more details.
        If a list, must have the same structure as coils.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(1,np.inf)``.",
        bounds_default="``bounds=(1,np.inf)``.",
        coil=True,
    )

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
        jac_chunk_size=None,
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
            jac_chunk_size=jac_chunk_size,
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

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(1,np.inf)``.",
        bounds_default="``bounds=(1,np.inf)``.",
        coil=True,
    )

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
        jac_chunk_size=None,
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
            jac_chunk_size=jac_chunk_size,
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


class CoilArclengthVariance(_CoilObjective):
    """Variance of ||dx/ds|| along the curve.

    This objective is meant to combat any issues corresponding to non-uniqueness of
    the representation of a curve, in that the same physical curve can be represented
    by different parametrizations by changing the curve parameter [1]_.
    Note that this objective has no effect for ``FourierRZCoil``, ``FourierPlanarCoil``,
    and ``FourierXYCoil`` which have a single unique parameterization
    (the objective will always return 0 for these types).

    References
    ----------
    .. [1] Wechsung, et al. "Precise stellarator quasi-symmetry can be achieved
       with electromagnetic coils." PNAS (2022)

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

    _scalar = False  # Not always a scalar, if a coilset is passed in
    _units = "(m^2)"
    _print_value_fmt = "Coil Arclength Variance: "

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
        name="coil arclength variance",
    ):
        if target is None and bounds is None:
            target = 0

        super().__init__(
            coils,
            ["x_s"],
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

        coilset = self.things[0]
        # local import to avoid circular import
        from desc.coils import CoilSet, FourierXYZCoil, SplineXYZCoil, _Coil

        def _is_single_coil(c):
            return isinstance(c, _Coil) and not isinstance(c, CoilSet)

        coils = tree_leaves(coilset, is_leaf=_is_single_coil)
        self._constants["mask"] = np.array(
            [int(isinstance(coil, (FourierXYZCoil, SplineXYZCoil))) for coil in coils]
        )

        if self._normalize:
            self._normalization = np.mean([scale["a"] ** 2 for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil arclength variance.

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
            Coil arclength variance.
        """
        if constants is None:
            constants = self.constants
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.array([jnp.var(jnp.linalg.norm(dat["x_s"], axis=1)) for dat in data])
        return out * constants["mask"]


class QuadraticFlux(_Objective):
    """Target B*n = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n. The equilibrium is kept fixed while the
    field is unfixed.

    Note: This objective is intended for coil optimization. For finding the surface
    that minimizes the normal field error, use the SurfaceQuadraticFlux objective.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error
        will be minimized. The equilibrium is kept fixed during the optimization
        with this objective.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
    source_grid : Grid, optional
        Collocation grid containing the nodes for plasma source terms.
        Default grid is detailed in the docs for ``compute_B_plasma``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
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

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

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
        jac_chunk_size=None,
    ):
        from desc.geometry import FourierRZToroidalSurface

        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid
        self._vacuum = vacuum
        errorif(
            isinstance(eq, FourierRZToroidalSurface),
            TypeError,
            "Detected FourierRZToroidalSurface object "
            "if attempting to find a QFM surface, please use "
            "SurfaceQuadraticFlux objective instead.",
        )
        things = [field]
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
            jac_chunk_size=jac_chunk_size,
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
                NFP=eq.NFP,
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
            eq,
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
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
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
        """Compute normal field error on boundary.

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
        f = (B_ext + B_plasma) * jnp.sqrt(eval_data["|e_theta x e_zeta|"])
        return f


class SurfaceQuadraticFlux(_Objective):
    """Target B*n = 0 on a surface.

    Used to find a quadratic-flux-minimizing (QFM) surface, so a
    `FourierRZToroidalSurface` should be passed to the objective.
    Should always be used along with a ``ToroidalFlux`` or ``Volume`` objective to
    ensure that the resulting QFM surface has the desired amount of
    flux enclosed and avoid trivial solutions.

    Note: This objective can be used with ``field_fixed=True`` to find the QFM surface
    by fixing the coils, however the surface is always free to change. For coil
    optimization, use the ``QuadraticFlux`` objective.

    Parameters
    ----------
    surface : FourierRZToroidalSurface
        QFM surface upon which the normal field error will be minimized.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided QFM surface. May be fixed
        by passing in ``field_fixed=True``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=surface.M_grid,``
        ``N=surface.N_grid, NFP=surface.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    field_fixed : bool
        Whether or not to fix the magnetic field's DOFs during the optimization.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _scalar = False
    _linear = False
    _print_value_fmt = "QFM surface normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        surface,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        eval_grid=None,
        field_grid=None,
        name="Surface Quadratic Flux",
        field_fixed=False,
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._eval_grid = eval_grid
        self._surface = surface
        self._field = field
        self._field_grid = field_grid
        self._field_fixed = field_fixed

        things = [surface]
        if not field_fixed:
            things += [field]
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
            jac_chunk_size=jac_chunk_size,
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
        surface = self._surface

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=2 * surface.M,
                N=2 * surface.N,
                NFP=surface.NFP,
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

        eval_profiles = get_profiles(self._data_keys, obj=surface, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=surface, grid=eval_grid)
        eval_data = compute_fun(
            surface,
            self._data_keys,
            params=surface.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )

        self._constants = {
            "field": self._field,
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(surface)
            Bscale = 1.0  # surface has no inherent B scale
            self._normalization = Bscale * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute normal field on surface.

        Parameters
        ----------
        params_1 : dict
            Dictionary of the surface's degrees of freedom.
        params_2 : dict
            Dictionary of the external field's degrees of freedom, only provided if
            if field_fixed=False.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Bnorm on the QFM surface from the external field

        """
        if constants is None:
            constants = self.constants
        field_params = params_2 if not self._field_fixed else None
        surf_params = params_1

        eval_data = compute_fun(
            self._surface,
            self._data_keys,
            surf_params,
            constants["eval_transforms"],
            constants["eval_profiles"],
        )
        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T
        if field_params is None:
            field_params = constants["field"].params_dict
        B_ext = constants["field"].compute_magnetic_field(
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=field_params,
        )
        B_ext = jnp.sum(B_ext * eval_data["n_rho"], axis=-1)
        f = (B_ext) * jnp.sqrt(eval_data["|e_theta x e_zeta|"])
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
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium (or QFM surface) for which the toroidal flux will be calculated.
    field : MagneticField
        MagneticField object, the parameters of this will be optimized
        to minimize the objective.
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField). Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at. Defaults to a LinearGrid(L=eq.L_grid, M=eq.M_grid,
        zeta=jnp.array(0.0), NFP=eq.NFP).
    field_fixed : bool
        Whether or not to fix the field's DOFs during the optimization.
    eq_fixed : bool
        Whether or not to fix the equilibrium (or QFM surface) DOFs
        during the optimization.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default=(
            "``target=eq.Psi`` if an Equilibrium is passed,"
            + " or ``target=1.0`` if a surface."
        ),
        bounds_default=(
            "``target=eq.Psi`` if an Equilibrium is passed,"
            + " or ``target=1.0`` if a surface."
        ),
        loss_detail=" Note: has no effect for this objective.",
    )

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
        field_fixed=False,
        eq_fixed=False,
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 1.0 if not hasattr(eq, "Psi") else eq.Psi
        self._field = field
        self._field_grid = field_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field_fixed = field_fixed
        self._eq_fixed = eq_fixed
        errorif(
            eq_fixed and field_fixed,
            ValueError,
            "Cannot have both `field_fixed=True` and `eq_fixed=True`",
        )
        things = []
        if not eq_fixed:
            things += [eq]
        if not field_fixed:
            things += [field]
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
            jac_chunk_size=jac_chunk_size,
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
        from desc.geometry import FourierRZToroidalSurface

        eq = self._eq
        self._use_vector_potential = True
        try:
            self._field.compute_magnetic_vector_potential([0, 0, 0])
        except (NotImplementedError, ValueError) as e:
            self._use_vector_potential = False
            errorif(
                isinstance(eq, FourierRZToroidalSurface)
                and not self._use_vector_potential,
                ValueError,
                "Targeting a QFM surface requires the vector potential to be "
                "calculated from the field, however the field cannot calculate "
                f"the vector potential, encountered error {e}",
            )
        if self._eval_grid is None:
            eval_grid = LinearGrid(
                L=eq.L_grid if not self._use_vector_potential else 0,
                M=eq.M_grid if hasattr(eq, "M_grid") else 3 * eq.M,
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
            self._normalization = 1.0 if not hasattr(eq, "Psi") else eq.Psi
        if not isinstance(eq, FourierRZToroidalSurface):
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

        # eval_grid.num_nodes for quad flux cost
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
        self._data_keys = data_keys
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)
        data = compute_fun(
            eq,
            self._data_keys,
            params=eq.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )

        plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T

        self._constants = {
            "plasma_coords": plasma_coords,
            "equil_data": data,
            "quad_weights": 1.0,
            "field": self._field,
            "field_grid": self._field_grid,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "eval_grid": eval_grid,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute toroidal flux.

        Parameters
        ----------
        params_1 : dict
            Dictionary of the external field's degrees of freedom, or the surface's
            degrees of freedom if qfm_surface=True.
        params_2 : dict
            Dictionary of the external field's degrees of freedom, if qfm_surface=True.
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
        field_params = params_2 if not self._eq_fixed else params_1
        field_params = (
            constants["field"].params_dict if self._field_fixed else field_params
        )
        surf_params = params_1 if not self._eq_fixed else None

        if not self._eq_fixed:
            data = compute_fun(
                self._eq,
                self._data_keys,
                surf_params,
                constants["eval_transforms"],
                constants["eval_profiles"],
            )
            plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T
        else:
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


class LinkingCurrentConsistency(_Objective):
    """Target the self-consistent poloidal linking current between the plasma and coils.

    A self-consistent coil + plasma configuration must have the sum of the signed
    currents in the coils that poloidally link the plasma equal to the total poloidal
    current required to be linked by the plasma according to the loop integral of its
    toroidal magnetic field, given by `G(rho=1)`. This objective computes the difference
    between these two quantities, such that a value of zero means the coils create the
    correct net poloidal current.

    Assumes the coil topology does not change (ie the linking number with the plasma
    is fixed).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    coil : CoilSet
        Coil(s) that are to be optimized.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma current at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)``.
    eq_fixed : bool
        Whether the equilibrium is assumed fixed (should be true for stage 2, false
        for single stage).
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _scalar = True
    _units = "(A)"
    _print_value_fmt = "Linking current error: "

    def __init__(
        self,
        eq,
        coil,
        *,
        grid=None,
        eq_fixed=False,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="linking current",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._eq_fixed = eq_fixed
        self._linear = eq_fixed
        self._eq = eq
        self._coil = coil

        super().__init__(
            things=[coil] if eq_fixed else [coil, eq],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            jac_chunk_size=jac_chunk_size,
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
        coil = self._coil
        grid = self._grid or LinearGrid(
            M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
        )
        warnif(
            not np.allclose(grid.nodes[:, 0], 1),
            UserWarning,
            "grid includes interior points, should be rho=1.",
        )

        self._dim_f = 1
        self._data_keys = ["G"]

        all_params = tree_map(lambda dim: np.arange(dim), coil.dimensions)
        current_params = tree_map(lambda idx: {"current": idx}, True)
        # indices of coil currents
        self._indices = tree_leaves(broadcast_tree(current_params, all_params))
        self._num_coils = coil.num_coils

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        # compute linking number of coils with plasma. To do this we add a fake "coil"
        # along the magnetic axis and compute the linking number of that coilset
        from desc.coils import FourierRZCoil, MixedCoilSet

        axis_coil = FourierRZCoil(
            1.0,
            eq.axis.R_n,
            eq.axis.Z_n,
            eq.axis.R_basis.modes[:, 2],
            eq.axis.Z_basis.modes[:, 2],
            eq.axis.NFP,
        )
        dummy_coilset = MixedCoilSet(axis_coil, coil, check_intersection=False)
        # linking number for coils with axis
        link = np.round(dummy_coilset._compute_linking_number())[0, 1:]

        self._constants = {
            "quad_weights": 1.0,
            "link": link,
        }

        if self._eq_fixed:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._data_keys,
                params=eq.params_dict,
                transforms=transforms,
                profiles=profiles,
            )
            eq_linking_current = 2 * jnp.pi * data["G"][0] / mu_0
            self._constants["eq_linking_current"] = eq_linking_current
        else:
            self._constants["profiles"] = profiles
            self._constants["transforms"] = transforms

        if self._normalize:
            params = tree_leaves(
                coil.params_dict, is_leaf=lambda x: isinstance(x, dict)
            )
            self._normalization = np.sum([np.abs(param["current"]) for param in params])

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, coil_params, eq_params=None, constants=None):
        """Compute linking current error.

        Parameters
        ----------
        coil_params : dict
            Dictionary of coilset degrees of freedom, eg ``CoilSet.params_dict``
        eq_params : dict
            Dictionary of equilibrium degrees of freedom, eg ``Equilibrium.params_dict``
            Only required if eq_fixed=False.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants.

        Returns
        -------
        f : array of floats
            Linking current error.

        """
        if constants is None:
            constants = self.constants
        if self._eq_fixed:
            eq_linking_current = constants["eq_linking_current"]
        else:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._data_keys,
                params=eq_params,
                transforms=constants["transforms"],
                profiles=constants["profiles"],
            )
            eq_linking_current = 2 * jnp.pi * data["G"][0] / mu_0

        coil_currents = jnp.concatenate(
            [
                jnp.atleast_1d(param[idx])
                for param, idx in zip(tree_leaves(coil_params), self._indices)
            ]
        )
        coil_currents = self.things[0]._all_currents(coil_currents)
        coil_linking_current = jnp.sum(constants["link"] * coil_currents)
        return eq_linking_current - coil_linking_current


class CoilSetLinkingNumber(_Objective):
    """Prevents coils from becoming interlinked.

    The linking number of 2 curves is (approximately) 0 if they are not linked, and
    (approximately) +/-1 if they are (with the sign indicating the helicity of the
    linking).

    This objective returns a single value for each coil in the coilset, with that number
    being the sum of the absolute value of the linking numbers of that coil with every
    other coil in the coilset, approximating the number of other coils that are linked

    Parameters
    ----------
    coil : CoilSet
        Coil(s) that are to be optimized.
    grid : Grid, list, optional
        Collocation grid used to discretize each coil. Defaults to
        ``LinearGrid(N=50)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coil linking number: "

    def __init__(
        self,
        coil,
        grid=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="coil-coil linking number",
    ):
        from desc.coils import CoilSet

        if target is None and bounds is None:
            target = 0
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
            jac_chunk_size=jac_chunk_size,
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
        grid = self._grid or LinearGrid(N=50)

        self._dim_f = coilset.num_coils
        self._constants = {"coilset": coilset, "grid": grid, "quad_weights": 1.0}

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute linking numbers between coils.

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
            For each coil, the sum of the absolute value of the linking numbers between
            that coil and every other coil in the coilset, which approximates the
            number of coils linked with that coil.

        """
        if constants is None:
            constants = self.constants
        link = constants["coilset"]._compute_linking_number(
            params=params, grid=constants["grid"]
        )

        return jnp.abs(link).sum(axis=0)


class SurfaceCurrentRegularization(_Objective):
    """Target the surface current magnitude.

    compute::

        w * ||K|| * sqrt(||e_theta x e_zeta||)

    where K is the winding surface current density, w is the
    regularization parameter (the weight on this objective),
    and ||e_theta x e_zeta|| is the magnitude of the surface normal i.e. the
    surface jacobian ||e_theta x e_zeta||

    This is intended to be used with a surface current::

        K = n x ∇ Φ

    i.e. a CurrentPotentialField

    Intended to be used with a QuadraticFlux objective, to form
    a problem similar to the REGCOIL algorithm described in [1]_ (if used with a
    ``FourierCurrentPotentialField``, is equivalent to the ``simple``
    regularization of the ``solve_regularized_surface_current`` method).

    References
    ----------
    .. [1] Landreman, Matt. "An improved current potential method for fast computation
      of stellarator coil shapes." Nuclear Fusion (2017).

    Parameters
    ----------
    surface_current_field : CurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with its ``field_grid`` matching this ``source_grid``, this replicates the
        REGCOIL algorithm described in [1]_ .

    """

    weight_str = (
        "weight : {float, ndarray}, optional"
        "\n\tWeighting to apply to the Objective, relative to other Objectives."
        "\n\tMust be broadcastable to to ``Objective.dim_f``"
        "\n\tWhen used with QuadraticFlux objective, this acts as the regularization"
        "\n\tparameter (with w^2 = lambda), with 0 corresponding to no regularization."
        "\n\tThe larger this parameter is, the less complex the surface current will "
        "be,\n\tbut the worse the normal field."
    )
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        overwrite={"weight": weight_str},
    )

    _coordinates = "tz"
    _units = "A/m"
    _print_value_fmt = "Surface Current Regularization: "

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        source_grid=None,
        name="surface-current-regularization",
    ):
        from desc.magnetic_fields import (
            CurrentPotentialField,
            FourierCurrentPotentialField,
        )

        if target is None and bounds is None:
            target = 0
        assert isinstance(
            surface_current_field, (CurrentPotentialField, FourierCurrentPotentialField)
        ), (
            "surface_current_field must be a CurrentPotentialField or "
            + f"FourierCurrentPotentialField, instead got {type(surface_current_field)}"
        )
        self._surface_current_field = surface_current_field
        self._source_grid = source_grid

        super().__init__(
            things=[surface_current_field],
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
        from desc.magnetic_fields import FourierCurrentPotentialField

        surface_current_field = self.things[0]
        if isinstance(surface_current_field, FourierCurrentPotentialField):
            M_Phi = surface_current_field._M_Phi
            N_Phi = surface_current_field._N_Phi
        else:
            M_Phi = surface_current_field.M
            N_Phi = surface_current_field.N

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=3 * M_Phi + 1,
                N=3 * N_Phi + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["K", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        surface_transforms = get_transforms(
            self._surface_data_keys,
            obj=surface_current_field,
            grid=source_grid,
            has_axis=source_grid.axis.size,
        )
        if self._normalize:
            if isinstance(surface_current_field, FourierCurrentPotentialField):
                self._normalization = np.max(
                    [abs(surface_current_field.I) + abs(surface_current_field.G), 1]
                )
            else:  # it does not have I,G bc is CurrentPotentialField
                Phi = surface_current_field.compute("Phi", grid=source_grid)["Phi"]
                self._normalization = np.max(
                    [
                        np.mean(np.abs(Phi)),
                        1,
                    ]
                )

        self._constants = {
            "surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, surface_params=None, constants=None):
        """Compute surface current regularization.

        Parameters
        ----------
        surface_params : dict
            Dictionary of surface degrees of freedom,
            eg FourierCurrentPotential.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            The surface current density magnitude on the source surface.

        """
        if constants is None:
            constants = self.constants

        surface_data = compute_fun(
            self._surface_current_field,
            self._surface_data_keys,
            params=surface_params,
            transforms=constants["surface_transforms"],
            profiles={},
        )

        K_mag = safenorm(surface_data["K"], axis=-1)
        return K_mag * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
