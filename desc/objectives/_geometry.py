"""Objectives for targeting geometrical quantities."""

import numpy as np

from desc.backend import jnp, vmap
from desc.compute import get_profiles, get_transforms, rpz2xyz
from desc.compute.geom_utils import copy_rpz_periods
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid, QuadratureGrid
from desc.utils import Timer, errorif, parse_argname_change, safenorm, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs
from .utils import check_if_points_are_inside_perimeter, softmin


class AspectRatio(_Objective):
    """Aspect ratio = major radius / minor radius.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium or FourierRZToroidalSurface that
        will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=2``.",
        bounds_default="``target=2``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
        loss_detail=" Note: Has no effect for this objective.",
    )

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Aspect ratio: "

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
        jac_chunk_size=None,
        device_id=0,
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=1``.",
        bounds_default="``target=1``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Elongation: "

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
        jac_chunk_size=None,
        device_id=0,
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=1``.",
        bounds_default="``target=1``.",
        loss_detail=" Note: Has no effect for this objective.",
    )

    _scalar = True
    _units = "(m^3)"
    _print_value_fmt = "Plasma volume: "

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
        jac_chunk_size=None,
        device_id=0,
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    surface : Surface
        Bounding surface to penalize distance to.
    surface_grid : Grid, optional
        Collocation grid containing the nodes to evaluate surface geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    plasma_grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    use_softmin: bool, optional
        Use softmin or hard min.
    use_signed_distance: bool, optional
        Whether to use absolute value of distance or a signed distance, with d
        being positive if the plasma is inside of the bounding surface, and
        negative if outside of the bounding surface.
        NOTE: ``plasma_grid`` and ``surface_grid`` must have the same
        toroidal angle values for signed distance to be used.
    eq_fixed, surface_fixed: bool, optional
        Whether the eq/surface is fixed or not. If True, the eq/surface is fixed
        and its coordinates are precomputed, which saves on computation time during
        optimization, and self.things = [surface]/[eq] only.
        If False, the eq/surface coordinates are computed at every iteration.
        False by default, so that self.things = [eq, surface]
        Both cannot be True.
    softmin_alpha: float, optional
        Parameter used for softmin. The larger ``softmin_alpha``, the closer the
        softmin approximates the hardmin. softmin -> hardmin as
        ``softmin_alpha`` -> infinity.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(1,np.inf)``.",
        bounds_default="``bounds=(1,np.inf)``.",
    )

    _coordinates = "rtz"
    _units = "(m)"
    _print_value_fmt = "Plasma-vessel distance: "

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
        eq_fixed=False,
        surface_fixed=False,
        softmin_alpha=1.0,
        name="plasma-vessel distance",
        use_signed_distance=False,
        jac_chunk_size=None,
        device_id=0,
        **kwargs,
    ):
        if target is None and bounds is None:
            bounds = (1, np.inf)
        self._surface = surface
        self._surface_grid = surface_grid
        self._plasma_grid = plasma_grid
        self._use_softmin = use_softmin
        self._use_signed_distance = use_signed_distance
        self._surface_fixed = surface_fixed
        self._eq_fixed = eq_fixed
        self._eq = eq
        errorif(
            eq_fixed and surface_fixed, ValueError, "Cannot fix both eq and surface"
        )

        self._softmin_alpha = parse_argname_change(
            softmin_alpha, kwargs, "alpha", "softmin_alpha"
        )
        errorif(
            len(kwargs) != 0,
            AssertionError,
            f"PlasmaVesselDistance got unexpected keyword argument: {kwargs.keys()}",
        )
        things = []
        if not eq_fixed:
            things.append(eq)
        if not surface_fixed:
            things.append(surface)
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
            device_id=device_id,
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
            surface = self.things[0]
        elif self._surface_fixed:
            eq = self.things[0]
            surface = self._surface
        else:
            eq = self.things[0]
            surface = self.things[1]
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

        # TODO(#568): How to use with generalized toroidal angle?
        # first check that the number of zeta nodes are the same, which
        # is a prerequisite to the zeta nodes themselves being the same
        errorif(
            self._use_signed_distance
            and not np.allclose(
                plasma_grid.num_zeta,
                surface_grid.num_zeta,
            ),
            ValueError,
            "Plasma grid and surface grid must contain points only at the "
            "same zeta values in order to use signed distance",
        )
        errorif(
            self._use_signed_distance
            and not np.allclose(
                plasma_grid.nodes[plasma_grid.unique_zeta_idx, 2],
                surface_grid.nodes[surface_grid.unique_zeta_idx, 2],
            ),
            ValueError,
            "Plasma grid and surface grid must contain points only at the "
            "same zeta values in order to use signed distance",
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
            self._constants["surface_coords"] = surface_coords
        elif self._eq_fixed:
            data_eq = compute_fun(
                self._eq,
                self._equil_data_keys,
                params=self._eq.params_dict,
                transforms=equil_transforms,
                profiles=equil_profiles,
            )
            self._constants["data_equil"] = data_eq
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute plasma-surface distance.

        Parameters
        ----------
        params_1 : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict,
            if eq_fixed is False, else the surface degrees of freedom
        params_2 : dict
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
        if self._eq_fixed:
            surface_params = params_1
        elif self._surface_fixed:
            equil_params = params_1
        else:
            equil_params = params_1
            surface_params = params_2
        if not self._eq_fixed:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._equil_data_keys,
                params=equil_params,
                transforms=constants["equil_transforms"],
                profiles=constants["equil_profiles"],
            )
        else:
            data = constants["data_equil"]
        plasma_coords_rpz = jnp.array([data["R"], data["phi"], data["Z"]]).T
        # we only copy the plasma data to the full torus, so that we still only
        # consider a single period of the surface.
        plasma_coords_nfp = copy_rpz_periods(
            plasma_coords_rpz, constants["equil_transforms"]["grid"].NFP
        )
        plasma_coords = rpz2xyz(plasma_coords_nfp)
        if self._surface_fixed:
            surface_coords_rpz = constants["surface_coords"]
        else:
            surface_coords_rpz = compute_fun(
                self._surface,
                self._surface_data_keys,
                params=surface_params,
                transforms=constants["surface_transforms"],
                profiles={},
            )["x"]
        surface_coords = rpz2xyz(surface_coords_rpz)

        diff_vec = plasma_coords[:, None, :] - surface_coords[None, :, :]
        d = safenorm(diff_vec, axis=-1)

        point_signs = jnp.ones(surface_coords.shape[0])
        if self._use_signed_distance:
            # for sign, we ignore other periods since the sign will be the same in each
            plasma_coords_rpz = plasma_coords_rpz.reshape(
                constants["equil_transforms"]["grid"].num_zeta,
                constants["equil_transforms"]["grid"].num_theta,
                3,
            )
            surface_coords_rpz = surface_coords_rpz.reshape(
                constants["surface_transforms"]["grid"].num_zeta,
                constants["surface_transforms"]["grid"].num_theta,
                3,
            )

            # loop over zeta planes
            def fun(plasma_pts_at_zeta_plane, surface_pts_at_zeta_plane):
                plasma_pts_at_zeta_plane = jnp.vstack(
                    (plasma_pts_at_zeta_plane, plasma_pts_at_zeta_plane[0, :])
                )

                pt_sign = check_if_points_are_inside_perimeter(
                    plasma_pts_at_zeta_plane[:, 0],
                    plasma_pts_at_zeta_plane[:, 2],
                    surface_pts_at_zeta_plane[:, 0],
                    surface_pts_at_zeta_plane[:, 2],
                )

                return pt_sign

            point_signs = vmap(fun, in_axes=0)(
                plasma_coords_rpz, surface_coords_rpz
            ).flatten()
            # at end here, point_signs is either +/- 1  with
            # positive meaning the surface pt
            # is outside the plasma and -1 if the surface pt is
            # inside the plasma

        if self._use_softmin:  # do softmin
            return (
                jnp.apply_along_axis(softmin, 0, d, self._softmin_alpha) * point_signs
            )
        else:  # do hardmin
            return d.min(axis=0) * point_signs


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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(-np.inf,0)``.",
        bounds_default="``bounds=(-np.inf,0)``.",
    )

    _coordinates = "rtz"
    _units = "(m^-1)"
    _print_value_fmt = "Mean curvature: "

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
        jac_chunk_size=None,
        device_id=0,
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(M=2*eq.M, N=2*eq.N)`` for ``FourierRZToroidalSurface``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=1``.",
        bounds_default="``target=1``.",
    )

    _coordinates = "rtz"
    _units = "(m^-1)"
    _print_value_fmt = "Principal curvature: "

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
        jac_chunk_size=None,
        device_id=0,
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(1,np.inf)``.",
        bounds_default="``bounds=(1,np.inf)``.",
    )

    _coordinates = "rtz"
    _units = "(m)"
    _print_value_fmt = "Magnetic field scale length: "

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
        jac_chunk_size=None,
        device_id=0,
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coordinate goodness : "

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
        jac_chunk_size=None,
        device_id=0,
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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


class MirrorRatio(_Objective):
    """Target a particular value mirror ratio.

    The mirror ratio is defined as:

    (Bₘₐₓ - Bₘᵢₙ) / (Bₘₐₓ + Bₘᵢₙ)

    Where Bₘₐₓ and Bₘᵢₙ are the maximum and minimum values of ||B|| on a given surface.
    Returns one value for each surface in ``grid``.

    Parameters
    ----------
    eq : Equilibrium or OmnigenousField
        Equilibrium or OmnigenousField that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)`` for ``Equilibrium``
        or ``LinearGrid(theta=2*eq.M_B, N=2*eq.N_x)`` for ``OmnigenousField``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0.2``.",
        bounds_default="``target=0.2``.",
    )

    _coordinates = "r"
    _units = "(dimensionless)"
    _print_value_fmt = "Mirror ratio: "

    def __init__(
        self,
        eq,
        *,
        grid=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="mirror ratio",
        jac_chunk_size=None,
        device_id=0,
    ):
        if target is None and bounds is None:
            target = 0.2
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
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
        from desc.equilibrium import Equilibrium
        from desc.magnetic_fields import OmnigenousField

        if self._grid is None and isinstance(eq, Equilibrium):
            grid = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
            )
        elif self._grid is None and isinstance(eq, OmnigenousField):
            grid = LinearGrid(
                theta=2 * eq.M_B,
                N=2 * eq.N_x,
                NFP=eq.NFP,
            )
        else:
            grid = self._grid

        self._dim_f = grid.num_rho
        self._data_keys = ["mirror ratio"]

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
        """Compute mirror ratio.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or field degrees of freedom,
            eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        M : ndarray
            Mirror ratio on each surface.

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
        return constants["transforms"]["grid"].compress(data["mirror ratio"])
