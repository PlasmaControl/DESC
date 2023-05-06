"""Objectives for targeting geometrical quantities."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.geometry.utils import rpz2xyz
from desc.grid import LinearGrid, QuadratureGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class AspectRatio(_Objective):
    """Aspect ratio = major radius / minor radius.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: Has no effect for this objective.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Note: Has no effect for this objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Aspect ratio: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=2,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="aspect ratio",
    ):

        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["R0/a"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute aspect ratio.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        AR : float
            Aspect ratio, dimensionless.

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        return data["R0/a"]


class Elongation(_Objective):
    """Elongation = semi-major radius / semi-minor radius. Max of all toroidal angles.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: Has no effect for this objective.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Note: Has no effect for this objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Elongation: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=1,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="elongation",
    ):

        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["a_major/a_minor"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute elongation.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        elongation : float
            Elongation, dimensionless.

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        return data["a_major/a_minor"]


class FluxGradient(_Objective):
    """Penalizes |grad(psi)| as a proxy for elongation.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
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
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(Wb/m)"
    _print_value_fmt = "Flux gradient: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="flux gradient",
    ):

        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.
        """
        if self._grid is None:
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["|grad(psi)|"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["Psi"] / scales["a"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute flux gradient.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        |grad(psi)| : float
            Toroidal flux gradient (normalized by 2pi), Webers per meter.

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        return data["|grad(psi)|"]

    def compute_scaled(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        return super().compute_scaled(*args, **kwargs) * jnp.sqrt(
            self._transforms["grid"].weights
        )


class Volume(_Objective):
    """Plasma volume.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
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
        be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "(m^3)"
    _print_value_fmt = "Plasma volume: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=1,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="volume",
    ):

        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["V"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["V"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute plasma volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        V : float
            Plasma volume (m^3).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        return data["V"]


class PlasmaVesselDistance(_Objective):
    """Target the distance between the plasma and a surounding surface.

    Computes the minimum distance from each point on the surface grid to a point on the
    plasma grid. For dense grids, this will approximate the global min, but in general
    will only be an upper bound on the minimum separation between the plasma and the
    surrounding surface.

    Parameters
    ----------
    surface : Surface
        Bounding surface to penalize distance to.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    surface_grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate surface geometry at.
    plasma_grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate plasma geometry at.
    name : str
        Name of the objective function.
    """

    _scalar = False
    _linear = False
    _units = "(m)"
    _print_value_fmt = "Plasma-vessel distance: {:10.3e} "

    def __init__(
        self,
        surface,
        eq=None,
        target=None,
        bounds=(1, np.inf),
        weight=1,
        normalize=True,
        normalize_target=True,
        surface_grid=None,
        plasma_grid=None,
        name="plasma vessel distance",
    ):
        self._surface = surface
        self._surface_grid = surface_grid
        self._plasma_grid = plasma_grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._surface_grid is None:
            surface_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            surface_grid = self._surface_grid
        if self._plasma_grid is None:
            plasma_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            plasma_grid = self._plasma_grid
        if not np.allclose(surface_grid.nodes[:, 0], 1):
            warnings.warn("Surface grid includes off-surface pts, should be rho=1")
        if not np.allclose(plasma_grid.nodes[:, 0], 1):
            warnings.warn("Plasma grid includes interior points, should be rho=1")

        self._dim_f = surface_grid.num_nodes
        self._data_keys = ["R", "phi", "Z"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._surface_coords = self._surface.compute_coordinates(
            grid=surface_grid, basis="xyz"
        )
        self._profiles = get_profiles(self._data_keys, eq=eq, grid=plasma_grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=plasma_grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute plasma-surface distance.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        d : ndarray, shape(surface_grid.num_nodes,)
            For each point in the surface grid, approximate distance to plasma.

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        plasma_coords = rpz2xyz(jnp.array([data["R"], data["phi"], data["Z"]]).T)
        d = jnp.linalg.norm(
            plasma_coords[:, None, :] - self._surface_coords[None, :, :], axis=-1
        )
        return d.min(axis=0)

    def compute_scaled(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        return super().compute_scaled(*args, **kwargs) * jnp.sqrt(
            self._transforms["grid"].weights
        )


class MeanCurvature(_Objective):
    """Target a particular value for the mean curvature.

    The mean curvature H of a surface is an extrinsic measure of curvature that locally
    describes the curvature of an embedded surface in Euclidean space.

    Positive mean curvature generally corresponds to "concave" regions of the plasma
    boundary which may be difficult to create with coils or magnets.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(m^-1)"
    _print_value_fmt = "Mean curvature: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=(-np.inf, 0),
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="mean-curvature",
    ):

        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["curvature_H"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = 1 / scales["a"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute mean curvature.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        H : ndarray
            Mean curvature at each point (m^-1).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        return data["curvature_H"]

    def compute_scaled(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        return super().compute_scaled(*args, **kwargs) * jnp.sqrt(self.grid.weights)


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
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(m^-1)"
    _print_value_fmt = "Principal curvature: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=1,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="principal-curvature",
    ):

        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["curvature_k1", "curvature_k2"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = 1 / scales["a"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute max absolute principal curvature.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        k : ndarray
            Max absolute principal curvature at each point (m^-1).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        return jnp.maximum(jnp.abs(data["curvature_k1"]), jnp.abs(data["curvature_k2"]))

    def compute_scaled(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        return super().compute_scaled(*args, **kwargs) * jnp.sqrt(self.grid.weights)
