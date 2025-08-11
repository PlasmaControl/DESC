import numbers
import warnings

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
from .objective_funs import _Objective, collect_docs

import jax
from desc.transform import Transform
from desc.basis import DoubleFourierSeries

from desc.derivatives import Derivative

from desc.compute._isothermal import (first_derivative_t, first_derivative_z, 
                                    first_derivative_t2, first_derivative_z2,
                                    )

from desc.compute import xyz2rpz, rpz2xyz_vec#rpz2xyz,  xyz2rpz_vec
#from desc.magnetic_fields._core import biot_savart_general
#import biot_savart_general
#from desc.fns_simp import _compute_magnetic_field_from_Current2

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
    by different parametrizations by changing the curve parameter [1]_. Note that this
    objective has no effect for ``FourierRZCoil`` and ``FourierPlanarCoil`` which have
    a single unique parameterization (the objective will always return 0 for these
    types).

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

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error will be minimized.
        The equilibrium is kept fixed during the optimization with this objective.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
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
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

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
        sqrt_area_weighting=False,
        name="Quadratic flux",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._sqrt_area_weighting = sqrt_area_weighting
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
        eval_data["area_weighting"] = (
            jnp.sqrt(eval_data["|e_theta x e_zeta|"])
            if self._sqrt_area_weighting
            else eval_data["|e_theta x e_zeta|"]
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
        f = (B_ext + B_plasma) * eval_data["area_weighting"]
        return f

class QuadraticFlux_fd(_Objective):
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
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

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
        scalar,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        source_grid=None,
        eval_grid=None,
        field_grid=None,
        vacuum=False,
        sqrt_area_weighting=False,
        name="Quadratic flux",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._sqrt_area_weighting = sqrt_area_weighting
        things = [field, scalar]
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
        
        eval_data["area_weighting"] = (
            jnp.sqrt(eval_data["|e_theta x e_zeta|"])
            if self._sqrt_area_weighting
            else eval_data["|e_theta x e_zeta|"]
        )

        self._surface_data_keys = ["theta","zeta",
                                   "x",
                                   "e_theta","e_zeta",
                                   "n_rho", "phi", "|e_theta x e_zeta|"]
        
        surface_transforms = get_transforms(self._surface_data_keys, obj=self._field, grid=self._source_grid)
        
        surface_data = compute_fun(
            self._field,
            self._surface_data_keys,
            params= self._field.params_dict,
            transforms=surface_transforms,#constants["surface_transforms"],
            profiles={},
            basis="rpz",
        )
        
        self._constants = {
            #"field": self._field,
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "source_grid": self._source_grid,
            "surface_transforms": surface_transforms,
            "surface_data":surface_data,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                params1,
                params2 = None,
                constants=None):
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
        #B_plasma = constants["B_plasma"]

        #Bdata = eq.compute(["R","phi","Z","n_rho"], grid = Bgrid)
        #coords = np.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T

        #surface_transforms = get_transforms(self._surface_data_keys, obj=self.things[0], grid=self._source_grid)
        
        #surface_data = compute_fun(
        #    self._field,
        #    self._surface_data_keys,
        #    params=params1,
        #    transforms=constants["surface_transforms"],
        #    profiles={},
        #    basis="rpz",
        #)

        surface_data = constants["surface_data"]
        B_ext = _compute_magnetic_field_from_Current(params2["a_n"], 
                                                     surface_data, 
                                                     constants["source_grid"],
                                                     #coords,
                                                     eval_data, # Data of the equilibrium
                                                     #eq,
                                                     #Bgrid,
                                                     basis="rpz")
        
        B_ext = jnp.sum(B_ext * eval_data["n_rho"], axis=-1)
        f = (B_ext + 0) * eval_data["area_weighting"]
        return f

class b_fd(_Objective):
    """Target b^2 = 0 on winding surface.

    Parameters
    ----------
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
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
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

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
        field,
        scalar,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        #self._eval_grid = eval_grid
        #self._eq = eq
        self._field = field
        #self._field_grid = field_grid
        #self._vacuum = vacuum
        #self._sqrt_area_weighting = sqrt_area_weighting
        things = [field, scalar]
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
            #jac_chunk_size=jac_chunk_size,
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

        self._data_keys = ["theta","zeta",
                           "R", "Z", "n_rho", "phi", 
                           "e_theta","e_zeta",
                           "e_theta_t","e_theta_z",
                           "e_zeta_t","e_zeta_z",
                           "|e_theta x e_zeta|",
                          "|e_theta x e_zeta|_t","|e_theta x e_zeta|_z"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        surface_grid = self._source_grid
        self._dim_f = surface_grid.num_nodes

        w = surface_grid.weights
        w *= jnp.sqrt(surface_grid.num_nodes)
        surface_profiles = get_profiles(self._data_keys, obj=self.things[0], grid=surface_grid)
        surface_transforms = get_transforms(self._data_keys, obj=self.things[0], grid=surface_grid)
        
        surface_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=self._field.params_dict,
            transforms=surface_transforms,
            profiles=surface_profiles,
        )
        
        surface_data["area_weighting"] = (
            jnp.sqrt(surface_data["|e_theta x e_zeta|"])
            #if self._sqrt_area_weighting
            #else surface_data["|e_theta x e_zeta|"]
        )

        # source_grid.num_nodes for the regularization cost
        self._dim_f = surface_grid.num_nodes

        self._constants = {
            #"field": self._field,
            #"field_grid": self._field_grid,
            "quad_weights": w,
            #"eval_data": eval_data,
            "source_grid": self._source_grid,
            "surface_transforms": surface_transforms,
            "surface_data": surface_data
            #"B_plasma": Bplasma,
        }
        
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                params1,
                params2 = None, 
                constants=None):
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
        
        #surface_data = compute_fun(
        #    self._field,
        #    self._data_keys,
        #    params=params1,
        #    transforms=constants["surface_transforms"],
        #    profiles={},
        #    basis="rpz",
        #)
        surface_data = constants["surface_data"]
        #b = find_b(surface_data,x,y,z, m_size,n_size)
        #return find_b(surface_data, constants["source_grid"], params2["a_n"],) * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
        return find_b(surface_data , constants["source_grid"], params2["a_n"],) * jnp.sqrt(surface_data["|e_theta x e_zeta|"])


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
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField). Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at. Defaults to a LinearGrid(L=eq.L_grid, M=eq.M_grid,
        zeta=jnp.array(0.0), NFP=eq.NFP).

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=eq.Psi``.",
        bounds_default="``target=eq.Psi``.",
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
        jac_chunk_size=None,
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

        w * ||K|| * ||e_theta x e_zeta||

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
        "\n\tMust be broadcastable to to Objective.dim_f"
        "\n\tWhen used with QuadraticFlux objective, this acts as the regularization"
        "\n\tparameter (with w^2 = lambda), with 0 corresponding to no regularization."
        "\n\tThe larger this parameter is, the less complex the surface current will "
        "be,\n\tbut the worse the normal field."
    )
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        loss_detail=" Note: has no effect for this objective.",
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
            M_Phi = surface_current_field.M + 1
            N_Phi = surface_current_field.N + 1

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
    
####################################################################################################################    
## Adding objective for sigma variation
#################################################################################################################### 
class SigmaVariation_fourier(_Objective):
    """Target the thickness of the single coil not to surpass a current density limit.

    compute::

        w * ||K|| * e^{b_max - b}

    where K is the winding surface current density, w is the
    regularization parameter (the weight on this objective)

    This is intended to be used with a surface current::

        K = n x ∇ Φ

    i.e. a CurrentPotentialField

    Intended to be used with a QuadraticFlux objective.

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
        "\n\tMust be broadcastable to to Objective.dim_f"
        "\n\tWhen used with QuadraticFlux objective, this acts as the regularization"
        "\n\tparameter (with w^2 = lambda), with 0 corresponding to no regularization."
        "\n\tThe larger this parameter is, the less complex the surface current will "
        "be,\n\tbut the worse the normal field."
    )
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        loss_detail=" Note: has no effect for this objective.",
        overwrite={"weight": weight_str},
    )

    _coordinates = "tz"
    _units = "A/m"
    _print_value_fmt = "Coil Thickness Regularization: "

    def __init__(
        self,
        surface_current_field,
        M_b = 0,
        N_b = 0,
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
        #self._b_field = b_field
        self._M_b = M_b
        self._N_b = N_b
        self._source_grid = source_grid

        super().__init__(
            things=[surface_current_field,
                    #b_field
                   ],
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
            M_Phi = surface_current_field.M + 1
            N_Phi = surface_current_field.N + 1

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
        self._surface_data_keys = ["K", "|e_theta x e_zeta|",
                                   "theta","zeta",
                                   "x_s","y_s","z_s",
                                  ]

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
        
        # Generate a basis for the single value component of the non-dimensional thickness
        b_sv_basis = DoubleFourierSeries(M = self._M_b, 
                                         N = self._N_b, 
                                         NFP = surface_current_field.NFP, #eq.NFP, 
                                         #sym = "sin",
                                        )
        
        b_sv_trans = Transform(source_grid, 
                                           b_sv_basis, 
                                           derivs=1, 
                                           rcond='auto', 
                                           build=True, 
                                           build_pinv=False, 
                                           method='auto')
        
        #######################################################
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
            #"surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
            "b_sv_basis": b_sv_basis,
            "b_sv_trans": b_sv_trans,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                surface_params,#=None, 
                #b_params,
                constants=None):
        """Compute surface current regularization.

        Parameters
        ----------
        surface_params : dict
            Dictionary of surface degrees of freedom,
            eg FourierCurrentPotential.params_dict
        b_params : dict
            Dictionary of surface degrees of freedom
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

        surface_transforms = get_transforms(self._surface_data_keys, 
                                            obj=self._surface_current_field, 
                                            grid=self._source_grid, 
                                            jitable = True,)
        
        surface_data = compute_fun(
            self._surface_current_field,
            self._surface_data_keys,
            params=surface_params,
            #transforms=constants["surface_transforms"],
            transforms=surface_transforms,
            profiles={},
        )
        
        #b_soln = b_eval(constants["b_sv_basis"], constants["b_sv_trans"], surface_data)
        
        fs_t, fs_z = db_eval(constants["b_sv_basis"], constants["b_sv_trans"], surface_data)
        
        #K_mag = safenorm(surface_data["K"], axis=-1)
        #return K_mag * jnp.exp(jnp.max(b_soln) - b_soln) #* jnp.sqrt(surface_data["|e_theta x e_zeta|"])
        
        #return b_soln * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
        return jnp.sqrt( ( fs_t**2 + fs_z**2 ) * surface_data["|e_theta x e_zeta|"] )

def b_eval(basis,trans,data):
    
    # Initial guess for modes
    x = jnp.ones(basis.num_modes + 0)
    fun_wrapped = lambda x : a_matrix(basis, trans, x, data["x_s"], data["y_s"])
    
    # Implement chunk method to reduce memory costs
    A = Derivative(fun_wrapped,deriv_mode="looped").compute(x)
    b_soln = jnp.linalg.pinv(A) @ data["z_s"]
    
    return (#b_soln[basis.num_modes] * data["theta"]
            #+ b_soln[basis.num_modes + 1] * data["zeta"]
            trans.transform(b_soln[0 : basis.num_modes], dt=0, dz=0)
           )

def db_eval(basis,trans,data):
    
    # Initial guess for modes
    x = jnp.ones(basis.num_modes + 2)
    fun_wrapped = lambda x : a_matrix(basis, trans, x, data["x_s"], data["y_s"])
    
    # Implement chunk method to reduce memory costs
    #A = jax.jacfwd(fun_wrapped)(x)
    #A = Derivative(fun_wrapped,deriv_mode="looped").compute(x)
    
    # Invert the matrix and find the solution
    b_soln_ = jnp.linalg.pinv( Derivative(fun_wrapped,deriv_mode="looped").compute(x)
                             ) @ data["z_s"]
    
    fst = b_soln_[basis.num_modes] + trans.transform(b_soln_[0 : basis.num_modes], dr=0, dt=1, dz=0)
    fsz = b_soln_[basis.num_modes + 1] + trans.transform(b_soln_[0 : basis.num_modes], dr=0, dt=0, dz=1)
    
    return fst, fsz

def a_matrix(basis, trans, xu, x_, y_,):
    
    x_mn = xu[0 : basis.num_modes]

    fs_ = {"bf_t": trans.transform(x_mn, dt = 1),
           "bf_z": trans.transform(x_mn, dz = 1),
          }
    
    return (x_*(fs_["bf_t"] 
                #+ xu[basis.num_modes] 
               )
            - y_*(fs_["bf_z"]  
                  #+ xu[basis.num_modes + 1]
                 )
           )

class SigmaVariation_fd(_Objective):
    """Minimize error between σ ∇ V and K_s

    compute:

        w*|σ ∇ V - K_s|^2

    where w is the regularization parameter (the weight on this objective).
    b is defined through the equation:
    
        x*b_t - y*b_z = z
        
    where:
    
        σ = e^b
        z = K_t . e_zeta + K . e_zeta_t - (K_z . e_theta + K . e_theta_z)
    
    where K is the winding surface current density, 
    and K_t, K_z are the toroidal and poloidal derivatives of K

    This is intended to be used with a surface current:

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField

    Intended to be used with a QuadraticFlux objective, to form
    the REGCOIL algorithm described in [1]_.

    [1] Landreman, An improved current potential method for fast computation
        of stellarator coil shapes, Nuclear Fusion (2017)

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
        When used with QuadraticFlux objective, this acts as the regularization
        parameter, with 0 corresponding to no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with the same ``source_grid``, this replicates the REGCOIL algorithm described
        in [1]_.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #soff_min,
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["K_eng",#"b_s","b_t","b_z",
                                   "|e_theta x e_zeta|",
                                  ]

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
            basis="rpz",
        )
        
        return safenorm( surface_data["K_eng"] - surface_data["K"], axis=-1 ) * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
    
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
class xVariation(_Objective):
    """Minimize error between σ ∇ V and K_s

    compute:

        w*|σ ∇ V - K_s|^2

    where w is the regularization parameter (the weight on this objective).
    b is defined through the equation:
    
        x*b_t - y*b_z = z
        
    where:
    
        σ = e^b
        z = K_t . e_zeta + K . e_zeta_t - (K_z . e_theta + K . e_theta_z)
    
    where K is the winding surface current density, 
    and K_t, K_z are the toroidal and poloidal derivatives of K

    This is intended to be used with a surface current:

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField

    Intended to be used with a QuadraticFlux objective, to form
    the REGCOIL algorithm described in [1]_.

    [1] Landreman, An improved current potential method for fast computation
        of stellarator coil shapes, Nuclear Fusion (2017)

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
        When used with QuadraticFlux objective, this acts as the regularization
        parameter, with 0 corresponding to no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with the same ``source_grid``, this replicates the REGCOIL algorithm described
        in [1]_.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #soff_min,
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["x_s",
                                   "|e_theta x e_zeta|",
                                  ]

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
            basis="rpz",
        )
        
        #return jnp.sqrt( ( surface_data["x_s"] ** 2 + surface_data["y_s"] ** 2 ) * surface_data["|e_theta x e_zeta|"] )
        return surface_data["x_s"] * jnp.sqrt( surface_data["|e_theta x e_zeta|"] )
    
class yVariation(_Objective):
    """Minimize error between σ ∇ V and K_s

    compute:

        w*|σ ∇ V - K_s|^2

    where w is the regularization parameter (the weight on this objective).
    b is defined through the equation:
    
        x*b_t - y*b_z = z
        
    where:
    
        σ = e^b
        z = K_t . e_zeta + K . e_zeta_t - (K_z . e_theta + K . e_theta_z)
    
    where K is the winding surface current density, 
    and K_t, K_z are the toroidal and poloidal derivatives of K

    This is intended to be used with a surface current:

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField

    Intended to be used with a QuadraticFlux objective, to form
    the REGCOIL algorithm described in [1]_.

    [1] Landreman, An improved current potential method for fast computation
        of stellarator coil shapes, Nuclear Fusion (2017)

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
        When used with QuadraticFlux objective, this acts as the regularization
        parameter, with 0 corresponding to no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with the same ``source_grid``, this replicates the REGCOIL algorithm described
        in [1]_.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #soff_min,
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["y_s",
                                   "|e_theta x e_zeta|",
                                  ]

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
            basis="rpz",
        )
        
        return surface_data["y_s"] * jnp.sqrt( surface_data["|e_theta x e_zeta|"] )

class zVariation(_Objective):
    """Minimize error between σ ∇ V and K_s

    compute:

        w*|σ ∇ V - K_s|^2

    where w is the regularization parameter (the weight on this objective).
    b is defined through the equation:
    
        x*b_t - y*b_z = z
        
    where:
    
        σ = e^b
        z = K_t . e_zeta + K . e_zeta_t - (K_z . e_theta + K . e_theta_z)
    
    where K is the winding surface current density, 
    and K_t, K_z are the toroidal and poloidal derivatives of K

    This is intended to be used with a surface current:

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField

    Intended to be used with a QuadraticFlux objective, to form
    the REGCOIL algorithm described in [1]_.

    [1] Landreman, An improved current potential method for fast computation
        of stellarator coil shapes, Nuclear Fusion (2017)

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
        When used with QuadraticFlux objective, this acts as the regularization
        parameter, with 0 corresponding to no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with the same ``source_grid``, this replicates the REGCOIL algorithm described
        in [1]_.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #soff_min,
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["z_s",
                                   "|e_theta x e_zeta|",
                                  ]

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
            basis="rpz",
        )
        
        return surface_data["z_s"] * jnp.sqrt( surface_data["|e_theta x e_zeta|"] )
####################################################################################################################    
## Adding objective for sigma variation
#################################################################################################################### 
class CoilThicknessRegularization(_Objective):
    """Target the thickness of the single coil not to surpass a current density limit.

    compute::

        w * ||K|| * e^{b_max - b}

    where K is the winding surface current density, w is the
    regularization parameter (the weight on this objective)

    This is intended to be used with a surface current::

        K = n x ∇ Φ

    i.e. a CurrentPotentialField

    Intended to be used with a QuadraticFlux objective.

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
        "\n\tMust be broadcastable to to Objective.dim_f"
        "\n\tWhen used with QuadraticFlux objective, this acts as the regularization"
        "\n\tparameter (with w^2 = lambda), with 0 corresponding to no regularization."
        "\n\tThe larger this parameter is, the less complex the surface current will "
        "be,\n\tbut the worse the normal field."
    )
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        loss_detail=" Note: has no effect for this objective.",
        overwrite={"weight": weight_str},
    )

    _coordinates = "tz"
    _units = "A/m"
    _print_value_fmt = "Coil Thickness Regularization: "

    def __init__(
        self,
        surface_current_field,
        M_b = 0,
        N_b = 0,
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
        #self._b_field = b_field
        self._M_b = M_b
        self._N_b = N_b
        self._source_grid = source_grid

        super().__init__(
            things=[surface_current_field,
                    #b_field
                   ],
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
            M_Phi = surface_current_field.M + 1
            N_Phi = surface_current_field.N + 1

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
        self._surface_data_keys = ["|e_theta x e_zeta|",
                                   "theta","zeta",
                                   "x_s","y_s","z_s",
                                  ]

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
        
        # Generate a basis for the single value component of the non-dimensional thickness
        b_sv_basis = DoubleFourierSeries(M = self._M_b, 
                                         N = self._N_b, 
                                         NFP = surface_current_field.NFP, #eq.NFP, 
                                         #sym = "sin",
                                        )
        
        b_sv_trans = Transform(source_grid, 
                                           b_sv_basis, 
                                           derivs=1, 
                                           rcond='auto', 
                                           build=True, 
                                           build_pinv=False, 
                                           method='auto')
        
        #######################################################
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
            #"surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
            "b_sv_basis": b_sv_basis,
            "b_sv_trans": b_sv_trans,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                surface_params,#=None, 
                #b_params,
                constants=None):
        """Compute surface current regularization.

        Parameters
        ----------
        surface_params : dict
            Dictionary of surface degrees of freedom,
            eg FourierCurrentPotential.params_dict
        b_params : dict
            Dictionary of surface degrees of freedom
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

        surface_transforms = get_transforms(self._surface_data_keys, 
                                            obj=self._surface_current_field, 
                                            grid=self._source_grid, 
                                            jitable = True,)
        
        surface_data = compute_fun(
            self._surface_current_field,
            self._surface_data_keys,
            params=surface_params,
            #transforms=constants["surface_transforms"],
            transforms=surface_transforms,
            profiles={},
        )
        
        #b_soln = b_eval(constants["b_sv_basis"], constants["b_sv_trans"], surface_data)
        
        fs_t, fs_z = db_eval(constants["b_sv_basis"], constants["b_sv_trans"], surface_data)
        #K_error = safenorm(surface_data["K"], axis=-1)
        #return K_mag * jnp.exp(jnp.max(b_soln) - b_soln) #* jnp.sqrt(surface_data["|e_theta x e_zeta|"])
        
        #return b_soln * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
        return jnp.sqrt( ( fs_t**2 + fs_z**2 ) * surface_data["|e_theta x e_zeta|"] )

class bRegularization_fd(_Objective):
    """Minimize variation of conductivity distribution

    compute:

        w*(b_t^2 + b_z^2) or w*b^2

    where w is the regularization parameter (the weight on this objective).
    b is defined through the equation:
    
        x*b_t - y*b_z = z
        
    where:
    
        z = K_t . e_zeta + K . e_zeta_t - (K_z . e_theta + K . e_theta_z)
    
    where K is the winding surface current density, 
    and K_t, K_z are the toroidal and poloidal derivatives of K

    This is intended to be used with a surface current:

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField

    Intended to be used with a QuadraticFlux objective, to form
    the REGCOIL algorithm described in [1]_.

    [1] Landreman, An improved current potential method for fast computation
        of stellarator coil shapes, Nuclear Fusion (2017)

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
        When used with QuadraticFlux objective, this acts as the regularization
        parameter, with 0 corresponding to no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with the same ``source_grid``, this replicates the REGCOIL algorithm described
        in [1]_.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["b_s",
                                   #"b_t","b_z",
                                   "|e_theta x e_zeta|",
                                   #"K",
                                  ]

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
            #basis="rpz",
            #basis="rpz",
        )
        
        #b_soln = surface_data["b_s"]
        #K_mag = safenorm(surface_data["K"], axis=-1)
        #return K_mag * jnp.exp(jnp.max(b_soln) - b_soln) * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
        return surface_data["b_s"] * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
        #return jnp.sqrt(surface_data["b_t"]**2 + surface_data["b_z"]**2) * jnp.sqrt(surface_data["|e_theta x e_zeta|"])

class bRegularization_fd2(_Objective):

    """Minimize variation of conductivity distribution

    compute:

        w*(b_t^2 + b_z^2) or w*b^2

    where w is the regularization parameter (the weight on this objective).
    b is defined through the equation:
    
        x*b_t - y*b_z = z
        
    where:
    
        z = K_t . e_zeta + K . e_zeta_t - (K_z . e_theta + K . e_theta_z)
    
    where K is the winding surface current density, 
    and K_t, K_z are the toroidal and poloidal derivatives of K

    This is intended to be used with a surface current:

        K = n x ∇ Φ
        Φ(θ,ζ) = Gζ/2π + Φ_b

    Φ_b is a potential defined on grid points on the winding surface

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
        When used with QuadraticFlux objective, this acts as the regularization
        parameter, with 0 corresponding to no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with the same ``source_grid``, this replicates the REGCOIL algorithm described
        in [1]_.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization with double FD:"

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = [#"zeta",
                                   #"e_theta","e_zeta",
                                   #"e_theta_t", "e_theta_z",
                                   #"e_zeta_t", "e_zeta_z"
                                   "|e_theta x e_zeta|", 
                                   # "|e_theta x e_zeta|_t","|e_theta x e_zeta|_z",
                                   "b_t","b_z",
                                  ]

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
        
        #b_soln = surface_data["b_s"]
        #K_mag = safenorm(surface_data["K"], axis=-1)
        return jnp.sqrt(surface_data["b_t"] ** 2 + surface_data["b_z"] ** 2) * jnp.sqrt(surface_data["|e_theta x e_zeta|"])

class JRegularization_fd(_Objective):
    """Minimize variation of conductivity distribution

    compute:

        w*(b_t^2 + b_z^2) or w*b^2

    where w is the regularization parameter (the weight on this objective).
    b is defined through the equation:
    
        x*b_t - y*b_z = z
        
    where:
    
        z = K_t . e_zeta + K . e_zeta_t - (K_z . e_theta + K . e_theta_z)
    
    where K is the winding surface current density, 
    and K_t, K_z are the toroidal and poloidal derivatives of K

    This is intended to be used with a surface current:

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["b_s",
                                   "b_t","b_z",
                                   "|e_theta x e_zeta|",
                                   "K",
                                  ]

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
            #basis="rpz",
            #basis="rpz",
        )
        
        return jnp.sqrt( surface_data["b_s"] ** 2 * surface_data["|e_theta x e_zeta|"] )
        
    
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

# Let's generate an objective for the error between Biot-Savart's Law and the Harmonic field for a given plasma surface
class HarmonicError(_Objective):
    """Target | ∫ K x a dS - H | = 0 on LCFS.

    Uses Biot-Savart's Law to generate a vacuum magnetic field, whose B at the surface is generated by
    harmonic vectors of said surface. This objective penalizes the difference between the magnetic 
    field generated by a surface current potential with a fixed net poloidal current and a harmmonic 
    magnetic field at rho = 1. 
    The fixed net poloidal current makes the magnetic field non trivial. 
    The shape of the surface is changed to minimize the error.
    field is unfixed.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error will be minimized.
        The equilibrium is kept fixed during the optimization with this objective.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
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
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

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
        coeffs, # HarmonicCoefficients object
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        #source_grid=None,
        eval_grid=None,
        field_grid=None,
        vacuum=False,
        sqrt_area_weighting=False,
        name="Harmonic Error",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        #self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        
        self._coeffs = coeffs
        
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._sqrt_area_weighting = sqrt_area_weighting
        things = [eq, field, coeffs]
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
                rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|",
                           "H_1", "H_2", # Compute additional data for harmonic vectors
                          ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        # pre-compute B_plasma because we are assuming eq is fixed
        if self._vacuum:
            Bplasma = jnp.zeros(eval_grid.num_nodes)
        else:
            Bplasma = compute_B_plasma(
                eq, eval_grid, self._source_grid, normal_only=True
            )

        self._constants = {
            #"field": self._field,
            "field_grid": self._field_grid,
            "eval_grid": self._eval_grid,
            "quad_weights": w,
            # This one changes in each iteration given that we are changing the shape of the surface
            #"eval_data": eval_data,
            "B_plasma": Bplasma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                params1,
                params2 = None,
                params3 = None,
                #field_params, 
                constants=None):
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
            B from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants

        eq = self._eq # Plasma Boundary treated as a fake equilibrium
        field = self._field # Surface Current Potential
        coeffs = self._coeffs # Harmonic coefficients
        
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=constants["eval_grid"])
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=constants["eval_grid"], jitable = True,)
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params= params1,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )
        eval_data["area_weighting"] = (
            jnp.sqrt(eval_data["|e_theta x e_zeta|"])
            if self._sqrt_area_weighting
            else eval_data["|e_theta x e_zeta|"]
        )

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B_ext is not pre-computed because field is not fixed
        B_coil = field.compute_magnetic_field(
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=params2,
        )
        
        B_harm = params3["a_n"]*eval_data["H_1"] + params3["b_n"]*eval_data["H_2"]
        
        dB = (B_coil - B_harm)
        
        #f = jnp.sum(dB * dB, axis=-1) * eval_data["area_weighting"]
        f = safenorm(dB, axis=-1) * eval_data["area_weighting"]
        return f
    
# Let's generate an objective for the error between Biot-Savart's Law and the Harmonic field for a given plasma surface
class HarmonicError2(_Objective):
    """Target | ∫ K x a dS - H | = 0 on LCFS.

    Uses Biot-Savart's Law to generate a vacuum magnetic field, whose B at the surface is generated by
    harmonic vectors of said surface. This objective penalizes the difference between the magnetic 
    field generated by a surface current potential with a fixed net poloidal current and a harmmonic 
    magnetic field at rho = 1. 
    The fixed net poloidal current makes the magnetic field non trivial. 
    The shape of the surface is changed to minimize the error.
    field is unfixed.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error will be minimized.
        The equilibrium is kept fixed during the optimization with this objective.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
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
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _scalar = False
    _linear = False
    _print_value_fmt = "Harmonic field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
        coeffs, # HarmonicCoefficients object
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        #source_grid=None,
        eval_grid=None,
        field_grid=None,
        vacuum=False,
        sqrt_area_weighting=False,
        name="Harmonic Error",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        #self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        
        self._coeffs = coeffs
        
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._sqrt_area_weighting = sqrt_area_weighting
        things = [eq, field, coeffs]
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
                rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|",
                           "H_1", "H_2", # Compute additional data for harmonic vectors
                          ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        # pre-compute B_plasma because we are assuming eq is fixed
        if self._vacuum:
            Bplasma = jnp.zeros(eval_grid.num_nodes)
        else:
            Bplasma = compute_B_plasma(
                eq, eval_grid, self._source_grid, normal_only=True
            )

        self._constants = {
            #"field": self._field,
            "field_grid": self._field_grid,
            "eval_grid": self._eval_grid,
            "quad_weights": w,
            # This one changes in each iteration given that we are changing the shape of the surface
            #"eval_data": eval_data,
            #"B_plasma": Bplasma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                params1,
                params2 = None,
                params3 = None,
                #field_params, 
                constants=None):
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
            B from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants

        eq = self._eq # Plasma Boundary treated as a fake equilibrium
        field = self._field # Surface Current Potential
        coeffs = self._coeffs # Harmonic coefficients
        
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=constants["eval_grid"])
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=constants["eval_grid"], jitable = True,)
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params= params1,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )
        eval_data["area_weighting"] = (
            jnp.sqrt(eval_data["|e_theta x e_zeta|"])
            if self._sqrt_area_weighting
            else eval_data["|e_theta x e_zeta|"]
        )

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B_ext is not pre-computed because field is not fixed
        B_coil = field.compute_magnetic_field(
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=params2,
        )
        
        # params3["b_n"] is not in use in this objective
        # Find the b_n coefficient that cancels the current enclosed by the LCFS
        b_n = (pol_cont_int(eval_data["theta"][0:constants["eval_grid"].M*2+1],
                            jnp.sum(eval_data["H_1"]*eval_data["e_theta"],
                                    axis = -1)[0:constants["eval_grid"].M*2+1],
                           ) / pol_cont_int(eval_data["theta"][0:constants["eval_grid"].M*2+1],
                                            jnp.sum(eval_data["H_2"]*eval_data["e_theta"],
                                                    axis = -1)[0:constants["eval_grid"].M*2+1],
                                           )
              )
        #a_n is merely used to adjust the norm of the magnetic field
        B_harm = params3["a_n"]*(eval_data["H_1"] - b_n*eval_data["H_2"])
        
        dB = (B_coil - B_harm)
        
        #f = jnp.sum(dB * dB, axis=-1) * eval_data["area_weighting"]
        f = safenorm(dB, axis=-1) * eval_data["area_weighting"]
        return f

# Let's generate an objective for the error between Biot-Savart's Law and the Harmonic field for a given plasma surface
class HarmonicError3(_Objective):
    """Target the dot product between e_uv and normal unit vector to a surface.

    Generate a surface whose isothermal coordinates coincides with curvature 
    This objective penalizes the difference between the magnetic 
    field generated by a surface current potential with a fixed net poloidal current and a harmmonic 
    magnetic field at rho = 1. 
    The fixed net poloidal current makes the magnetic field non trivial. 
    The shape of the surface is changed to minimize the error.
    field is unfixed.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error will be minimized.
        The equilibrium is kept fixed during the optimization with this objective.
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the plasma surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid,
        NFP=eq.NFP, sym=False)``
    vacuum : bool
        If true, B_plasma (the contribution to the normal field on the boundary from the
        plasma currents) is set to zero.
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

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
        #field,
        coeffs, # HarmonicCoefficients object
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        #source_grid=None,
        eval_grid=None,
        #field_grid=None,
        #vacuum=False,
        sqrt_area_weighting=False,
        name="Harmonic Error",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        #self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        #self._field = field
        
        self._coeffs = coeffs
        
        #self._field_grid = field_grid
        #self._vacuum = vacuum
        self._sqrt_area_weighting = sqrt_area_weighting
        things = [eq,
                  coeffs,
                 ]
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
        #coeffs = self._coeffs

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        self._data_keys = ["|e_theta x e_zeta|",
                           "e_theta","e_zeta",
                           "e_theta_t","e_theta_z",
                           "e_zeta_t","e_zeta_z",
                           "u_iso_t","u_iso_z",
                           "u_iso_t","u_iso_tz","u_iso_zz",
                           "v_iso_t","v_iso_z",
                           "v_iso_tt","v_iso_tz","v_iso_zz",
                           "n_rho","H_1", "H_2", 
                           #"R", "Z", "phi",
                          ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        self._constants = {
            "eval_grid": self._eval_grid,
            "quad_weights": w,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                params1,
                params2 = None,
                constants=None):
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
            B from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants

        eq = self._eq # Plasma Boundary treated as a fake equilibrium
        coeffs = self._coeffs
        
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=constants["eval_grid"])
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=constants["eval_grid"], jitable = True,)
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params= params1,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )
        eval_data["area_weighting"] = (
            jnp.sqrt(eval_data["|e_theta x e_zeta|"])
            if self._sqrt_area_weighting
            else eval_data["|e_theta x e_zeta|"]
        )
        
        # Define the basis vectors through isothermal coordinates
        a = 1
        b = params2["b_n"]
        c = - b
        d = a #jnp.where(params2["b_n"] == 0, 0, - (a/b)*c ) 

        u_t = a*eval_data["u_iso_t"] + b*eval_data["v_iso_t"]
        u_z = a*eval_data["u_iso_z"] + b*eval_data["v_iso_z"]
        v_t = c*eval_data["u_iso_t"] + d*eval_data["v_iso_t"]
        v_z = c*eval_data["u_iso_z"] + d*eval_data["v_iso_z"]
        
        u_tt = a*eval_data["u_iso_tt"] + b*eval_data["v_iso_tt"]
        u_tz = a*eval_data["u_iso_tz"] + b*eval_data["v_iso_tz"]
        u_zz = a*eval_data["u_iso_zz"] + b*eval_data["v_iso_zz"]
        
        v_tt = c*eval_data["u_iso_tt"] + d*eval_data["v_iso_tt"]
        v_tz = c*eval_data["u_iso_tz"] + d*eval_data["v_iso_tz"]
        v_zz = c*eval_data["u_iso_zz"] + d*eval_data["v_iso_zz"]

        l = u_t*v_z - u_z*v_t

        #e_u = ( (1/l) * ( vz * edata["e_theta"].T - vt * edata["e_zeta"].T ) ).T
        #e_v = ( (1/l) * ( - uz * edata["e_theta"].T + ut * edata["e_zeta"].T ) ).T
        
        # Define the derivatives of the basis vectors
        #e1 = eval_data["e_theta_t"] - ( utt*e_u.T + vtt*e_v.T ).T
        #e2 = eval_data["e_theta_z"] - ( utz*e_u.T + vtz*e_v.T ).T
        #e3 = eval_data["e_zeta_z"] - ( uzz*e_u.T + vzz*e_v.T ).T
        
        # Modified vectors to avoid divisions by control variables
        e_u = ( (1/1) * ( v_z * eval_data["e_theta"].T - v_t * eval_data["e_zeta"].T ) ).T
        e_v = ( (1/1) * ( - u_z * eval_data["e_theta"].T + u_t * eval_data["e_zeta"].T ) ).T
        
        # Define the derivatives of the basis vectors
        e1 = ( l * eval_data["e_theta_t"].T -  ( u_tt * e_u.T + v_tt * e_v.T ) ).T
        e2 = ( l * eval_data["e_theta_z"].T - ( u_tz * e_u.T + v_tz * e_v.T ) ).T
        e3 = ( l * eval_data["e_zeta_z"].T - ( u_zz * e_u.T + v_zz * e_v.T ) ).T

        # Cofactors of the I matrix (inverse of determinant of I is ignored)
        a1 = - 2 * (u_t*v_t*v_z**2 - u_z*v_z*v_t**2)
        a2 = u_t**2*v_z**2 - u_z**2*v_t**2
        a3 = - 2 * (u_t**2*u_z*v_z - u_z**2*u_t*v_t)
        
        e_uv = ( a1*e1.T + a2*e2.T + a3*e3.T ).T
        
        num = jnp.sum( eval_data["n_rho"] * e_uv, axis = -1 )
        #den = jnp.sum( e_uv * e_uv, axis = -1 )
        den = safenorm( e_uv , axis = -1 )
        f = jnp.where(den == 0, 0, num / den) * eval_data["area_weighting"]
        
        return f    
    
# Function for poloidal integration
def pol_cont_int(x_vals,f):
    
    test = 0
    # For loop for trapezoidal rule as an approximation to the line integral
    for i in range(0,len(f)-1):
        test = test + (x_vals[i+1] - x_vals[i])*1/2*(f[i+1] + f[i])
    return test    

################################################
class QuadraticFluxMod(_Objective):
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
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

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
        sqrt_area_weighting=False,
        name="Quadratic flux Modified",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._sqrt_area_weighting = sqrt_area_weighting
        things = [eq,field]
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
        field = self._field

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

        #eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        #eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)
        #eval_data = compute_fun(
        #    "desc.equilibrium.equilibrium.Equilibrium",
        #    self._data_keys,
        #    params=eq.params_dict,
        #    transforms=eval_transforms,
        #    profiles=eval_profiles,
        #)
        #eval_data["area_weighting"] = (
        #    jnp.sqrt(eval_data["|e_theta x e_zeta|"])
        #    if self._sqrt_area_weighting
        #    else eval_data["|e_theta x e_zeta|"]
        #)

        # pre-compute B_plasma because we are assuming eq is fixed
        if self._vacuum:
            Bplasma = jnp.zeros(eval_grid.num_nodes)

        self._constants = {
            #"field": self._field,
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_grid": self._eval_grid,
            #"eval_data": eval_data,
            #"B_plasma": Bplasma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                params, 
                field_params = None, 
                constants=None):
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
        eq = self._eq # Plasma Boundary
        field = self._field # Surface Current Potential
        
        if constants is None:
            constants = self.constants

        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=constants["eval_grid"])
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=constants["eval_grid"], jitable = True,)
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=eq.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )
        eval_data["area_weighting"] = (
            jnp.sqrt(eval_data["|e_theta x e_zeta|"])
            if self._sqrt_area_weighting
            else eval_data["|e_theta x e_zeta|"]
        )

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B_ext is not pre-computed because field is not fixed
        B_ext = field.compute_magnetic_field(
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=field_params,
        )
        B_ext = jnp.sum(B_ext * eval_data["n_rho"], axis=-1)
        f = (B_ext) * eval_data["area_weighting"]
        return f

class SurfaceDivergence(_Objective):
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
    sqrt_area_weighting : bool
        Whether or not to square the local area weighting on the objective, i.e.
        whether to return Bn * ||e_theta x e_zeta|| or Bn * sqrt(||e_theta x e_zeta||).
        If True, when combined with SurfaceCurrentRegularization, the resulting problem
        is equivalent to the REGCOIL algorithm.

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
        sqrt_area_weighting=False,
        name="Surface Divergence",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._sqrt_area_weighting = sqrt_area_weighting
        things = [eq, field]
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
                rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|", 
                           "e_theta", "e_zeta", 
                           "e^theta_s", "e^zeta_s"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        #eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        #eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)
        #eval_data = compute_fun(
        #    "desc.equilibrium.equilibrium.Equilibrium",
        #    self._data_keys,
        #    params=eq.params_dict,
        #    transforms=eval_transforms,
        #    profiles=eval_profiles,
        #)
        #eval_data["area_weighting"] = (
        #    jnp.sqrt(eval_data["|e_theta x e_zeta|"])
        #    if self._sqrt_area_weighting
        #    else eval_data["|e_theta x e_zeta|"]
        #)

        # pre-compute B_plasma because we are assuming eq is fixed
        if self._vacuum:
            Bplasma = jnp.zeros(eval_grid.num_nodes)

        self._constants = {
            #"field": self._field,
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_grid": self._eval_grid,
            #"eval_data": eval_data,
            #"B_plasma": Bplasma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                params1,
                params2 = None,
                constants = None):
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
            B from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants

        eq = self._eq # Plasma Boundary treated as a fake equilibrium
        field = self._field # Surface Current Potential
        #coeffs = self._coeffs # Harmonic coefficients
        
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=constants["eval_grid"])
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=constants["eval_grid"], jitable = True,)
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params= params1,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )
        eval_data["area_weighting"] = (
            jnp.sqrt(eval_data["|e_theta x e_zeta|"])
            if self._sqrt_area_weighting
            else eval_data["|e_theta x e_zeta|"]
        )

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B_ext is not pre-computed because field is not fixed
        surf_div_B_coil = field.compute_surface_divergence(
            x,
            eval_data["e_theta"],
            eval_data["e_zeta"],
            eval_data["e^theta_s"],
            eval_data["e^zeta_s"],
            eval_data["phi"],
            source_grid=constants["field_grid"],
            basis="rpz",
            params = params2,
        )
        
        f = surf_div_B_coil * eval_data["area_weighting"]
        return f

################################################################
# Define own functions to calculate magnetic field
def _compute_magnetic_field_from_Current(phi_scalar,
                                         surf_data, 
                                         Kgrid,
                                         eval_data,
                                         #coords, eq, Bgrid,
                                         basis="rpz"):
    
    """Compute magnetic field at a set of points.

    Parameters
    ----------
    K_at_grid : ndarray, shape (num_nodes,3)
        Surface current evaluated at points on a grid, which you want to calculate
        B from, should be in cartesian ("xyz") or cylindrical ("rpz") specifiec
        by "basis" argument
    surface : FourierRZToroidalSurface
        surface object upon which the surface current K_at_grid lies
    coords : array-like shape(N,3) or Grid
        cylindrical or cartesian coordinates to evlauate B at
    grid : Grid,
        source grid upon which to evaluate the surface current density K
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field

    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field at specified points

    """

    #Bdata = eq.compute(["R","phi","Z","n_rho"], grid = Bgrid)
    coords = jnp.vstack([eval_data["R"],eval_data["phi"],eval_data["Z"]]).T
    
    phi_t = first_derivative_t2(phi_scalar, surf_data,
                                2 * (Kgrid.M) + 1 , 2 * (Kgrid.N) + 1 )
    phi_z = first_derivative_z2(phi_scalar, surf_data,
                                2 * (Kgrid.M) + 1 , 2 * (Kgrid.N) + 1 )
    
    K_at_grid = ( ( - phi_z * (surf_data["|e_theta x e_zeta|"]) **(-1) ) * surf_data["e_theta"].T 
                 + ( phi_t * (surf_data["|e_theta x e_zeta|"]) **(-1) ) * surf_data["e_zeta"].T ) .T

    
    from desc.compute import xyz2rpz, xyz2rpz_vec, rpz2xyz_vec
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])
    
    surface_grid = Kgrid

    # compute and store grid quantities
    # needed for integration
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    #data = surface.compute(["x", "|e_theta x e_zeta|"], grid=Kgrid, basis="xyz")

    _rs = xyz2rpz(surf_data["x"])
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    _dV = Kgrid.weights * surf_data["|e_theta x e_zeta|"] / Kgrid.NFP

    from desc.magnetic_fields._core import biot_savart_general
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (Kgrid.nodes[:, 2] + j * 2 * jnp.pi / Kgrid.NFP) % (
            2 * jnp.pi
        )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general( coords, rs, K, _dV,
        )
        f += fj
        return f
    
    B = fori_loop(0, Kgrid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

# Invert the matrix and find b
def find_b(data,grid, phi_sca,):
    
    x_,y_,z_ = calc_data(phi_sca,grid,data)
    
    # Find the matrix
    f = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda f: b_residual(f,data,x_,y_,grid.M*2+1, grid.N*2+1)
    A_ = Derivative(fun_wrapped,deriv_mode="looped").compute(f)
    #A_ = jax.jacfwd(fun_wrapped)(x)
    
    return jnp.linalg.pinv(A_)@z_

# Function to find build a matrix to find the scalar b
def b_residual(g,data,_x,_y,m_size,n_size):
    
    #g_t = first_derivative_t(g, data,m_size,n_size)
    #g_z = first_derivative_z(g, data,m_size,n_size)
    return _x * first_derivative_t(g, data,m_size,n_size) - _y * first_derivative_z(g, data,m_size,n_size)

def calc_data(sca,grid,data):
    # Phi is not periodic
    phi_t = first_derivative_t2(sca, data,
                                    2 * (grid.M) + 1 , 2 * (grid.N) + 1 )
    phi_z = first_derivative_z2(sca, data,
                                    2 * (grid.M) + 1 , 2 * (grid.N) + 1 )
        
    # First derivative are periodic.
    phi_tt = first_derivative_t(phi_t, data,
                                    2 * (grid.M) + 1 , 2 * (grid.N) + 1 )
    phi_tz = first_derivative_z2(phi_t, data,
                                    2 * (grid.M) + 1 , 2 * (grid.N) + 1 )
    phi_zz = first_derivative_z(phi_z, data,
                                    2 * (grid.M) + 1 , 2 * (grid.N) + 1 )

    K_sup_theta = - phi_z * (data["|e_theta x e_zeta|"]) **(-1)
    K_sup_zeta = phi_t * (data["|e_theta x e_zeta|"]) **(-1)
     
    K_sup_theta_t = - ( phi_tz * (data["|e_theta x e_zeta|"]) **(-1) 
                           + phi_z * - (data["|e_theta x e_zeta|"]) ** (-2) * data["|e_theta x e_zeta|_t"] ) 
    
    K_sup_theta_z = - ( phi_zz * (data["|e_theta x e_zeta|"]) **(-1) 
                           + phi_z * - (data["|e_theta x e_zeta|"]) ** (-2) * data["|e_theta x e_zeta|_z"] ) 
    
    K_sup_zeta_t = ( phi_tt * (data["|e_theta x e_zeta|"]) **(-1) 
                        + phi_t * - (data["|e_theta x e_zeta|"]) **(-2) * data["|e_theta x e_zeta|_t"] )
    
    K_sup_zeta_z = ( phi_tz * (data["|e_theta x e_zeta|"]) **(-1) 
                        + phi_t * - (data["|e_theta x e_zeta|"]) **(-2) * data["|e_theta x e_zeta|_z"] )
    
    x = ( K_sup_theta * jnp.sum( data["e_theta"] * data["e_zeta"], axis=-1) 
             + K_sup_zeta * jnp.sum( data["e_zeta"] * data["e_zeta"], axis=-1) )
    
    y = ( K_sup_theta * jnp.sum( data["e_theta"] * data["e_theta"], axis=-1) 
             + K_sup_zeta * jnp.sum( data["e_theta"] * data["e_zeta"], axis=-1) )
    
    x_t = ( K_sup_theta_t * jnp.sum( data["e_theta"] * data["e_zeta"], axis=-1) 
               + K_sup_theta * jnp.sum( data["e_theta_t"] * data["e_zeta"] 
                                       + data["e_theta"] * data["e_zeta_t"], axis=-1) 
               + K_sup_zeta_t * jnp.sum( data["e_zeta"] * data["e_zeta"], axis=-1) 
               + K_sup_zeta * jnp.sum( 2 * data["e_zeta_t"] * data["e_zeta"], axis=-1) 
              )
    
    y_z = ( K_sup_theta_z * jnp.sum( data["e_theta"] * data["e_theta"], axis=-1) 
           + K_sup_theta * jnp.sum( 2 * data["e_theta"] * data["e_theta_z"], axis=-1) 
           + K_sup_zeta_z * jnp.sum( data["e_theta"] * data["e_zeta"], axis=-1) 
           + K_sup_zeta * jnp.sum( data["e_theta_z"] * data["e_zeta"] 
                                  + data["e_theta"] * data["e_zeta_z"], axis=-1) 
           )
    
    z = x_t - y_z

    return x, y, z

def K_calc(sca,grid,data):
    
    # Phi is not periodic
    _phi_t = first_derivative_t2(sca, data,
                                    2 * (grid.M) + 1 , 2 * (grid.N) + 1 )
    _phi_z = first_derivative_z2(sca, data,
                                    2 * (grid.M) + 1 , 2 * (grid.N) + 1 )
    
    return  ( data["|e_theta x e_zeta|"] ** (-1) * ( ( - _phi_z * data["e_theta"].T) 
                                                    + (_phi_t * data["e_zeta"].T) 
                                                   )
             ).T

#################################################################################################
class OmegaBoozer(_Objective):

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        surface_current_field,
        coeffs,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
        self._surface_current_field = surface_current_field
        self._source_grid = source_grid
        self.coeffs = coeffs
        
        super().__init__(
            things=[surface_current_field, coeffs],
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
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["|e_theta x e_zeta|",
                                   "e_theta","e_zeta",
                                   "Phi_t","Phi_z",
                                   "theta","zeta",
                                  ]

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

        self._constants = {
            "surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, surface_params, params2=None, constants=None):
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
            #basis="rpz",
            #basis="rpz",
        )
        
        iota = params2["a_n"]

        e_t = (( 1 + iota * surface_data["Phi_t"] ) * surface_data["e_theta"].T 
               + ( iota * surface_data["Phi_t"] ) * surface_data["e_zeta"].T).T

        e_z = ( ( iota * surface_data["Phi_z"] ) * surface_data["e_theta"].T 
               + ( 1 + iota * surface_data["Phi_z"] ) * surface_data["e_zeta"].T).T
        
        e_tt = jnp.sum(e_t * e_t,axis = 1)
        e_tz = jnp.sum(e_t * e_z,axis = 1)

        #error_1 = (e_tz + iota * e_tt) ** 2
        #error_2 = (surface_data["theta"] - )
        #K_mag = safenorm(surface_data["K"], axis=-1)
        return (1 * e_tz + iota * e_tt) * jnp.sqrt(surface_data["|e_theta x e_zeta|"])

#################################################################################################
#from desc.fns_simp import surf_int
class OmegaBoozer2(_Objective):

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        lambda_field,
        nu_field,
        coeffs,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            lambda_field, "Phi_mn"
        ), "lambda_field must be a FourierCurrentPotentialField"
        assert hasattr(
            nu_field, "Phi_mn"
        ), "nu_current_field must be a FourierCurrentPotentialField"
        
        self._lambda_field = lambda_field
        self._nu_field = nu_field
        self._source_grid = source_grid
        self.coeffs = coeffs
        
        super().__init__(
            things=[lambda_field, nu_field, coeffs],
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
        lambda_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=lambda_field._M_Phi * 3 + 1,
                N=lambda_field._N_Phi * 3 + 1,
                NFP=lambda_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["|e_theta x e_zeta|",
                                   "|e_theta x e_zeta|_t","|e_theta x e_zeta|_z",
                                   "e_theta","e_zeta",
                                   "e_theta_t","e_theta_z",
                                   "e_zeta_t","e_zeta_z",
                                   "Phi_t","Phi_z",
                                   "Phi_tt","Phi_tz","Phi_zz",
                                   "e^theta_s","e^zeta_s",
                                   "grad_s(Phi)",
                                   "Laplace_Beltrami(Phi)",
                                   "nabla_s^2_theta","nabla_s^2_zeta",
                                  ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        surface_transforms = get_transforms(
            self._surface_data_keys,
            obj=lambda_field,
            grid=source_grid,
            has_axis=source_grid.axis.size,
        )

        self._constants = {
            "surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                lambda_params, 
                nu_params = None,
                params2=None, constants=None):
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

        lambda_data = compute_fun(
            self._lambda_field,
            self._surface_data_keys,
            params=lambda_params,
            transforms=constants["surface_transforms"],
            profiles={},
            #basis="rpz",
            #basis="rpz",
        )

        nu_data = compute_fun(
            self._nu_field,
            self._surface_data_keys,
            params=nu_params,
            transforms=constants["surface_transforms"],
            profiles={},
        )
        
        iota = params2["a_n"]
        
        f = ( ( 1 + lambda_data["Phi_t"] + iota * nu_data["Phi_t"] ) * ( 1 + nu_data["Phi_z" ])
             - ( lambda_data["Phi_z"] + iota * nu_data["Phi_z"] ) * nu_data["Phi_t"]
            )

        f_t = ( ( 0 + lambda_data["Phi_tt"] + iota * nu_data["Phi_tt"] ) * ( 1 + nu_data["Phi_z"])
               + ( 1 + lambda_data["Phi_t"] + iota * nu_data["Phi_t"] ) * ( 0 + nu_data["Phi_tz"])
               - ( lambda_data["Phi_tz"] + iota * nu_data["Phi_tz"] ) * nu_data["Phi_t"]
               - ( lambda_data["Phi_z"] + iota * nu_data["Phi_z"] ) * nu_data["Phi_tt"]
               )

        f_z = ( ( 0 + lambda_data["Phi_tz"] + iota * nu_data["Phi_tz"] ) * ( 1 + nu_data["Phi_z"])
               + ( 1 + lambda_data["Phi_t"] + iota * nu_data["Phi_t"] ) * ( 0 + nu_data["Phi_zz"])
               - ( lambda_data["Phi_zz"] + iota * nu_data["Phi_zz"] ) * nu_data["Phi_t"]
               - ( lambda_data["Phi_z"] + iota * nu_data["Phi_z"] ) * nu_data["Phi_tz"]
               )

        # This is not the actual expression of e_t_Boozer since it is missing part of the denominator
        e_t_B = ( ( 1 + nu_data["Phi_z"] ) * lambda_data["e_theta"].T - nu_data["Phi_t"] * lambda_data["e_zeta"].T 
                 ).T

        e_t_B_t = ( ( 0 + nu_data["Phi_tz"] ) * lambda_data["e_theta"].T
                   + ( 1 + nu_data["Phi_z"] ) * lambda_data["e_theta_t"].T
                   - nu_data["Phi_tt"] * lambda_data["e_zeta"].T
                   - nu_data["Phi_t"] * lambda_data["e_zeta_t"].T
                   ).T

        e_t_B_z = ( ( 0 + nu_data["Phi_zz"] ) * lambda_data["e_theta"].T 
                   + ( 1 + nu_data["Phi_z"] ) * lambda_data["e_theta_z"].T 
                   - nu_data["Phi_tz"] * lambda_data["e_zeta"].T 
                   - nu_data["Phi_t"] * lambda_data["e_zeta_z"].T 
                 ).T
        
        # This is not the actual expression of e_t_Boozer since it is missing part of the denominator
        e_z_B = ( - ( lambda_data["Phi_z"] + iota * nu_data["Phi_z"]) * lambda_data["e_theta"].T 
                   + ( 1 + lambda_data["Phi_t"] + iota * nu_data["Phi_t"] ) * lambda_data["e_zeta"].T 
                   ).T

        e_z_B_t = ( - ( lambda_data["Phi_tz"] + iota * nu_data["Phi_tz"]) * lambda_data["e_theta"].T 
                     - ( lambda_data["Phi_z"] + iota * nu_data["Phi_z"]) * lambda_data["e_theta_t"].T  
                     + ( 0 + lambda_data["Phi_tt"] + iota * nu_data["Phi_tt"] ) * lambda_data["e_zeta"].T
                     + ( 1 + lambda_data["Phi_t"] + iota * nu_data["Phi_t"] ) * lambda_data["e_zeta_t"].T 
                   ).T

        e_z_B_z = ( - ( lambda_data["Phi_zz"] + iota * nu_data["Phi_zz"]) * lambda_data["e_theta"].T 
                     - ( lambda_data["Phi_z"] + iota * nu_data["Phi_z"]) * lambda_data["e_theta_z"].T  
                     + ( 0 + lambda_data["Phi_tz"] + iota * nu_data["Phi_tz"] ) * lambda_data["e_zeta"].T
                     + ( 1 + lambda_data["Phi_t"] + iota * nu_data["Phi_t"] ) * lambda_data["e_zeta_z"].T
                   ).T
        
        e_t_B_dot_e_t_B = jnp.sum(e_t_B * e_t_B,axis = 1)
        e_t_B_dot_e_z_B = jnp.sum(e_t_B * e_z_B,axis = 1)
        e_z_B_dot_e_z_B = jnp.sum(e_t_B * e_z_B,axis = 1)
        
        p3 = e_z_B_dot_e_z_B - iota * e_t_B_dot_e_t_B

        p3_t = 2 * ( jnp.sum(e_z_B * e_z_B_t,axis = 1) 
                    - iota * jnp.sum(e_t_B * e_t_B_t,axis = 1) )
        
        p3_z = 2 * ( jnp.sum(e_z_B * e_z_B_z,axis = 1) 
                    - iota * jnp.sum(e_t_B * e_t_B_z,axis = 1) )
        
        error_1 = e_t_B_dot_e_z_B + iota * e_t_B_dot_e_t_B

        # |e^psi|
        # |e^psi|_t
        #e_sub_psi_t = ( p3 ** (-1) * (f_t * lambda_data["|e_theta x e_zeta|"] + f * lambda_data["|e_theta x e_zeta|_t"])
        #               - f * lambda_data["|e_theta x e_zeta|"] * p3 ** (-2) * p3_t
        #              )

        e_sub_psi_t = ( p3 * (f_t * lambda_data["|e_theta x e_zeta|"] + f * lambda_data["|e_theta x e_zeta|_t"])
                       - f * lambda_data["|e_theta x e_zeta|"] * p3_t
                      )
        # |e^psi|_z
        #e_sub_psi_z = ( p3 ** (-1) * (f_z * lambda_data["|e_theta x e_zeta|"] + f * lambda_data["|e_theta x e_zeta|_z"])
        #               - f * lambda_data["|e_theta x e_zeta|"] * p3 ** (-2) * p3_z
        #              )

        e_sub_psi_z = ( p3 * ( f_z * lambda_data["|e_theta x e_zeta|"] + f * lambda_data["|e_theta x e_zeta|_z"] )
                       - f * lambda_data["|e_theta x e_zeta|"] * p3_z
                      )
        
        nabla_s_e_sub_psi = ( e_sub_psi_t * lambda_data["e^theta_s"].T 
                             + e_sub_psi_z * lambda_data["e^zeta_s"].T ).T
        
        error_2 = ( p3 ** (2) *  ( lambda_data["Laplace_Beltrami(Phi)"] 
                                  + lambda_data["nabla_s^2_theta"] 
                                  - iota * lambda_data["nabla_s^2_zeta"] )
                   + jnp.sum( nabla_s_e_sub_psi * ( lambda_data["grad_s(Phi)"] 
                                                   + lambda_data["e^theta_s"] 
                                                  - iota * lambda_data["e^zeta_s"]), 
                             axis = 1)
                  )
        
        error_3 = (iota 
                   - ( - surf_int(e_t_B_dot_e_t_B * f **(-1), 
                                  lambda_data,
                                  grid = self._source_grid) / surf_int(e_t_B_dot_e_z_B * f **(-1),
                                                                       lambda_data,
                                                                       grid = self._source_grid)
                     )
                  )
        
        return jnp.sqrt( ( error_1 ** 2 + error_2 ** 2 + error_3 ** 2 ) * lambda_data["|e_theta x e_zeta|"])

def surf_int(f,data,grid):

    Q = f*data["|e_theta x e_zeta|"]
    
    integrand = grid.spacing[:, 1] * grid.spacing[:, 2] * Q
    #desired_rho_surface = 1.0
    #indices = jnp.where(grid.nodes[:, 0] == desired_rho_surface)[0]
    #integrand = integrand[indices]
    
    return integrand.sum()

#
#################################################################################################
class Surf_Div(_Objective):

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        lambda_field,
        nu_field,
        coeffs,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            lambda_field, "Phi_mn"
        ), "lambda_field must be a FourierCurrentPotentialField"
        assert hasattr(
            nu_field, "Phi_mn"
        ), "nu_current_field must be a FourierCurrentPotentialField"
        
        self._lambda_field = lambda_field
        self._nu_field = nu_field
        self._source_grid = source_grid
        self.coeffs = coeffs
        
        super().__init__(
            things=[lambda_field, nu_field, coeffs],
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
        lambda_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=lambda_field._M_Phi * 3 + 1,
                N=lambda_field._N_Phi * 3 + 1,
                NFP=lambda_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._lambda_data_keys = ["|e_theta x e_zeta|",
                                  # "|e_theta x e_zeta|_t","|e_theta x e_zeta|_z",
                                  # "e_theta","e_zeta",
                                  # "e_theta_t","e_theta_z",
                                  # "e_zeta_t","e_zeta_z",
                                   "Phi_t","Phi_z",
                                   #"Phi_tt","Phi_tz","Phi_zz",
                                   "e^theta_s","e^zeta_s",
                                  # "grad_s(Phi)",
                                   "Laplace_Beltrami(Phi)",
                                  # "nabla_s^2_theta","nabla_s^2_zeta",
                                  "lambda_ratio","b_iso",
                                  "phi_iso_t","Psi_iso_t",
                                  "phi_iso_z","Psi_iso_z",
                                  ]

        self._nu_data_keys = ["Phi","grad_s(Phi)",
                              ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        #surface_transforms = get_transforms(
        #    self._surface_data_keys,
        #    obj=lambda_field,
        #    grid=source_grid,
        #    has_axis=source_grid.axis.size,
        #)

        self._constants = {
            #"surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
            "eval_grid": source_grid,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                lambda_params, 
                nu_params = None,
                params2=None, 
                constants=None):
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

        #eq = self._eq # Plasma Boundary treated as a fake equilibrium
        #field = self._field # Surface Current Potential
        #coeffs = self._coeffs # Harmonic coefficients
        
        lambda_profiles = get_profiles(self._lambda_data_keys, obj= self._lambda_field, grid=constants["eval_grid"])
        lambda_transforms = get_transforms(self._lambda_data_keys, obj=self._lambda_field, grid=constants["eval_grid"], jitable = True,)
        
        nu_profiles = get_profiles(self._lambda_data_keys, obj=self._nu_field, grid=constants["eval_grid"])
        nu_transforms = get_transforms(self._lambda_data_keys, obj=self._nu_field, grid=constants["eval_grid"], jitable = True,)
        
        lambda_data = compute_fun(
            self._lambda_field,
            self._lambda_data_keys,
            params=lambda_params,
            transforms=lambda_transforms,
            profiles={},
            basis="rpz",
            #basis="rpz",
        )

        nu_data = compute_fun(
            self._nu_field,
            self._nu_data_keys,
            params=nu_params,
            transforms=nu_transforms,
            profiles={},
            basis="rpz",
        )
        
        iota = jnp.exp(params2["a_n"])
        eta = params2["b_n"]
        r = lambda_data["lambda_ratio"]
        b = lambda_data["b_iso"]
        
        error_1 = (lambda_data["Phi_t"] * ( (eta + r) * (1 - lambda_data["phi_iso_z"]) - r * b * lambda_data["Psi_iso_z"])
                   - eta * (1 - lambda_data["phi_iso_z"]) 
                   - ( lambda_data["Phi_z"] - iota ) * ( (eta + r) * lambda_data["phi_iso_t"] 
                                                        - r * b * ( 1 - lambda_data["Psi_iso_t"] )
                                                       )
                   - r * ( ( 1 - lambda_data["phi_iso_z"] ) - b * lambda_data["Psi_iso_z"] )
                  )
        
        return jnp.sqrt( error_1 ** 2 * lambda_data["|e_theta x e_zeta|"])

#################################################################################################
class J_perp(_Objective):

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Sigma Regularization:"

    def __init__(
        self,
        lambda_field,
        nu_field,
        coeffs,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        #sorf_min
        deriv_mode="auto",
        source_grid=None,
        name="Variable-Conductivity-Regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            lambda_field, "Phi_mn"
        ), "lambda_field must be a FourierCurrentPotentialField"
        assert hasattr(
            nu_field, "Phi_mn"
        ), "nu_current_field must be a FourierCurrentPotentialField"
        
        self._lambda_field = lambda_field
        self._nu_field = nu_field
        self._source_grid = source_grid
        self.coeffs = coeffs
        
        super().__init__(
            things=[lambda_field, nu_field, coeffs],
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
        lambda_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=lambda_field._M_Phi * 3 + 1,
                N=lambda_field._N_Phi * 3 + 1,
                NFP=lambda_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._lambda_data_keys = ["|e_theta x e_zeta|",
                                  # "|e_theta x e_zeta|_t","|e_theta x e_zeta|_z",
                                  # "e_theta","e_zeta",
                                  # "e_theta_t","e_theta_z",
                                  # "e_zeta_t","e_zeta_z",
                                  # "Phi_t","Phi_z",
                                  # "Phi_tt","Phi_tz","Phi_zz",
                                   "e^theta_s","e^zeta_s",
                                   "grad_s(Phi)",
                                   "Laplace_Beltrami(Phi)",
                                   "nabla_s^2_theta","nabla_s^2_zeta",
                                  "lambda_ratio","b_iso",
                                  #"phi_iso_t","Psi_iso_t",
                                  #"phi_iso_z","Psi_iso_z",
                                  ]

        self._nu_data_keys = ["Phi","grad_s(Phi)",
                              ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        #surface_transforms = get_transforms(
        #    self._surface_data_keys,
        #    obj=lambda_field,
        #    grid=source_grid,
        #    has_axis=source_grid.axis.size,
        #)

        self._constants = {
            #"surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
            "eval_grid": source_grid,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, 
                lambda_params, 
                nu_params = None,
                params2=None, 
                constants=None):
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

        #eq = self._eq # Plasma Boundary treated as a fake equilibrium
        #field = self._field # Surface Current Potential
        #coeffs = self._coeffs # Harmonic coefficients
        
        lambda_profiles = get_profiles(self._lambda_data_keys, obj= self._lambda_field, grid=constants["eval_grid"])
        lambda_transforms = get_transforms(self._lambda_data_keys, obj=self._lambda_field, grid=constants["eval_grid"], jitable = True,)
        
        nu_profiles = get_profiles(self._lambda_data_keys, obj=self._nu_field, grid=constants["eval_grid"])
        nu_transforms = get_transforms(self._lambda_data_keys, obj=self._nu_field, grid=constants["eval_grid"], jitable = True,)
        
        lambda_data = compute_fun(
            self._lambda_field,
            self._lambda_data_keys,
            params=lambda_params,
            transforms=lambda_transforms,
            profiles={},
            basis="rpz",
            #basis="rpz",
        )

        nu_data = compute_fun(
            self._nu_field,
            self._nu_data_keys,
            params=nu_params,
            transforms=nu_transforms,
            profiles={},
            basis="rpz",
        )
        
        iota = jnp.exp(params2["a_n"])
        #eta = params2["b_n"]
        r = lambda_data["lambda_ratio"]
        b = lambda_data["b_iso"]
        
        error_2 = ( ( lambda_data["Laplace_Beltrami(Phi)"] 
                                  + lambda_data["nabla_s^2_theta"] 
                                  - iota * lambda_data["nabla_s^2_zeta"] )
                   + jnp.sum( ( jnp.exp( - nu_data["Phi"] ) * nu_data["grad_s(Phi)"].T ).T * ( lambda_data["grad_s(Phi)"]
                                                                                             + lambda_data["e^theta_s"] 
                                                                                             - iota * lambda_data["e^zeta_s"]), 
                             axis = 1)
                  )
        
        return jnp.sqrt( error_2 ** 2 * lambda_data["|e_theta x e_zeta|"])