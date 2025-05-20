"""Objectives for targeting MHD stability."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_params, get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid, QuadratureGrid
from desc.utils import Timer, errorif, setdefault, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs
from .utils import _parse_callable_target_bounds

overwrite_stability = {
    "target": """
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. If a callable, should take a
        single argument ``rho`` and return the desired value of the profile at those
        locations. Defaults to ``bounds=(0, np.inf)``
    """,
    "bounds": """
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``
        If a callable, each should take a single argument ``rho`` and return the
        desired bound (lower or upper) of the profile at those locations.
        Defaults to ``bounds=(0, np.inf)``
    """,
}


class MercierStability(_Objective):
    """The Mercier criterion is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with D_Mercier > 0 are favorable for stability.

    See equation 4.16 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.

    """

    __doc__ = __doc__.rstrip() + collect_docs(overwrite=overwrite_stability)

    _coordinates = "r"
    _units = "(Wb^-2)"
    _print_value_fmt = "Mercier Stability: "

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
        name="Mercier Stability",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
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
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            RuntimeWarning,
            "MercierStability objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "MercierStability objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["D_Mercier"]

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
            self._normalization = 1 / scales["Psi"] ** 2

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the Mercier stability criterion.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        D_Mercier : ndarray
            Mercier stability criterion.

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
        return constants["transforms"]["grid"].compress(data["D_Mercier"])


class MagneticWell(_Objective):
    """The magnetic well is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with magnetic well > 0 are favorable for stability.

    This objective uses the magnetic well parameter defined in equation 3.2 of
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite=overwrite_stability,
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = "r"
    _units = "(dimensionless)"
    _print_value_fmt = "Magnetic Well: "

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
        name="Magnetic Well",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
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
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            RuntimeWarning,
            "MagneticWell objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "MagneticWell objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["magnetic well"]

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
        """Compute a magnetic well parameter.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        magnetic_well : ndarray
            Magnetic well parameter.

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
        return constants["transforms"]["grid"].compress(data["magnetic well"])


class BallooningStability(_Objective):
    """A type of ideal MHD instability.

    Infinite-n ideal MHD ballooning modes are of significant interest.
    These instabilities are also related to smaller-scale kinetic instabilities.
    With this class, we optimize MHD equilibria against the ideal ballooning mode.

    Targets the following metric:

    f = w₀ sum(ReLU(λ-λ₀)) + w₁ max(ReLU(λ-λ₀))

    where λ is the negative squared growth rate for each field line (such that λ>0 is
    unstable), λ₀ is a cutoff, and w₀ and w₁ are weights.

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    rho : float
        Flux surface to optimize on. To optimize over multiple surfaces, use multiple
        objectives each with a single rho value.
    alpha : float, ndarray
        Field line labels to optimize. Values should be in [0, 2π). Default is
        ``alpha=0`` for axisymmetric equilibria, or 8 field lines linearly spaced
        in [0, π] for non-axisymmetric cases.
    nturns : int
        Number of toroidal transits of a field line to consider. Field line
        will run from -π*``nturns`` to π*``nturns``. Default 3.
    nzetaperturn : int
        Number of points along the field line per toroidal transit. Total number of
        points is ``nturns*nzetaperturn``. Default 100.
    zeta0 : array-like
        Points of vanishing integrated local shear to scan over.
        Default 15 points in [-π/2,π/2]
    lambda0 : float
        Threshold for penalizing growth rates in metric above.
    w0, w1 : float
        Weights for sum and max terms in metric above.
    name : str, optional
        Name of the objective function.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = ""  # not vectorized over rho, always a scalar
    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Ideal ballooning lambda: "

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
        rho=0.5,
        alpha=None,
        nturns=3,
        nzetaperturn=200,
        zeta0=None,
        lambda0=0.0,
        w0=1.0,
        w1=10.0,
        name="ideal ballooning lambda",
    ):
        if target is None and bounds is None:
            target = 0

        self._rho = rho
        self._alpha = alpha
        self._nturns = nturns
        self._nzetaperturn = nzetaperturn
        self._zeta0 = zeta0
        self._lambda0 = lambda0
        self._w0 = w0
        self._w1 = w1

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

        errorif(
            np.asarray(self._rho).size > 1,
            ValueError,
            "BallooningStability objective only works on a single surface. "
            "To optimize multiple surfaces, use multiple instances of the objective.",
        )

    def build(self, eq=None, use_jit=True, verbose=1):
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
        eq = self.things[0]

        # we need a uniform grid to get correct surface averages for iota
        iota_grid = LinearGrid(
            rho=self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
        )
        self._iota_keys = ["iota", "iota_r", "shear"]
        iota_profiles = get_profiles(self._iota_keys, obj=eq, grid=iota_grid)
        iota_transforms = get_transforms(self._iota_keys, obj=eq, grid=iota_grid)

        # TODO: Generalize balloning stabilty funs to multiple flux surfaces,
        #       include last closed flux surface requirement, and remove quadrature
        #       transforms.
        # Separate grid to calculate the right length scale for normalization
        len_grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        self._len_keys = ["a"]
        len_profiles = get_profiles(self._len_keys, obj=eq, grid=len_grid)
        len_transforms = get_transforms(self._len_keys, obj=eq, grid=len_grid)

        # make a set of nodes along a single fieldline
        zeta = jnp.linspace(
            -jnp.pi * self._nturns,
            jnp.pi * self._nturns,
            self._nturns * self._nzetaperturn,
        )

        # set alpha/zeta0 grids
        self._alpha = setdefault(
            self._alpha,
            (
                jnp.linspace(0, jnp.pi, 8)
                if eq.N != 0 and eq.sym is True
                else (
                    jnp.linspace(0, 2 * np.pi, 16)
                    if eq.N != 0 and eq.sym is False
                    else jnp.array(0.0)
                )
            ),
        )

        self._zeta0 = setdefault(
            self._zeta0, jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15)
        )
        self._dim_f = 1
        self._data_keys = ["ideal ballooning lambda"]

        self._args = get_params(
            self._iota_keys + self._len_keys + self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        self._constants = {
            "iota_transforms": iota_transforms,
            "iota_profiles": iota_profiles,
            "len_transforms": len_transforms,
            "len_profiles": len_profiles,
            "rho": self._rho,
            "alpha": self._alpha,
            "zeta": zeta,
            "zeta0": self._zeta0,
            "lambda0": self._lambda0,
            "w0": self._w0,
            "w1": self._w1,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """
        Compute the ballooning stability growth rate.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        lam : ndarray
            ideal ballooning growth rate.

        """
        eq = self.things[0]

        if constants is None:
            constants = self.constants
        # we first compute iota on a uniform grid to get correct averaging etc.
        iota_data = compute_fun(
            eq,
            self._iota_keys,
            params=params,
            transforms=constants["iota_transforms"],
            profiles=constants["iota_profiles"],
        )

        len_data = compute_fun(
            eq,
            self._len_keys,
            params=params,
            transforms=constants["len_transforms"],
            profiles=constants["len_profiles"],
        )

        # Now we compute theta_DESC for given theta_PEST
        rho, alpha, zeta = constants["rho"], constants["alpha"], constants["zeta"]

        # we prime the data dict with the correct iota values so we don't recompute them
        # using the wrong grid
        # RG: This would have to be modified for multiple rho values
        data = {
            "iota": iota_data["iota"][0],
            "iota_r": iota_data["iota_r"][0],
            "shear": iota_data["shear"][0],
            "a": len_data["a"],
        }

        grid = eq._get_rtz_grid(
            rho,
            alpha,
            zeta,
            coordinates="raz",
            period=(np.inf, 2 * np.pi, np.inf),
            params=params,
            iota=data["iota"],
        )

        lam = compute_fun(
            eq,
            self._data_keys,
            params,
            get_transforms(self._data_keys, eq, grid, jitable=True),
            profiles=get_profiles(self._data_keys, eq, grid),
            data=data,
            zeta0=constants["zeta0"],
        )["ideal ballooning lambda"]

        lambda0, w0, w1 = constants["lambda0"], constants["w0"], constants["w1"]

        # Shifted ReLU operation
        data = (lam - lambda0) * (lam >= lambda0)

        results = w0 * jnp.sum(data) + w1 * jnp.max(data)

        return results
