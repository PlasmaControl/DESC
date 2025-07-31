"""Objectives for targeting MHD stability."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import ResolutionWarning, Timer, setdefault, warnif

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
            ResolutionWarning,
            "MercierStability objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            ResolutionWarning,
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
            ResolutionWarning,
            "MagneticWell objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            ResolutionWarning,
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
        Flux surface to optimize on. Instabilities often peak near the middle.
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
        Default 15 points in [-π/2,π/2].
        The values ``zeta0`` correspond to values of ι ζ₀ and not ζ₀.
    Neigvals : int
        Number of top eigenvalues to select.
        Default is 1.
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

    _static_attrs = _Objective._static_attrs + [
        "_iota_keys",
        "_Neigvals",
        "_nturns",
        "_nzetaperturn",
        "_add_lcfs",
    ]

    _coordinates = "r"
    _units = "~"
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
        rho=np.array([0.5]),
        alpha=None,
        nturns=3,
        nzetaperturn=200,
        zeta0=None,
        Neigvals=1,
        lambda0=0.0,
        w0=1.0,
        w1=10.0,
        name="ideal ballooning lambda",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        self._nturns = nturns
        self._nzetaperturn = nzetaperturn
        self._Neigvals = Neigvals
        self._lambda0 = lambda0
        self._w0 = w0
        self._w1 = w1
        self._rho = np.atleast_1d(rho)
        self._add_lcfs = np.all(self._rho < 0.97)
        self._alpha = setdefault(
            alpha,
            (
                jnp.linspace(0, (1 + eq.sym) * jnp.pi, (1 + eq.sym) * 8)
                if eq.N
                else jnp.array([0])
            ),
        )
        self._zeta0 = setdefault(zeta0, jnp.linspace(-0.5 * np.pi, 0.5 * np.pi, 15))

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
        self._iota_keys = ["iota", "iota_r", "shear", "a"]

        eq = self.things[0]
        iota_grid = LinearGrid(
            # to compute length scale quantities correctly
            rho=np.append(self._rho, 1) if self._add_lcfs else self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
        )
        assert not iota_grid.axis.size
        self._dim_f = iota_grid.num_rho - self._add_lcfs
        transforms = get_transforms(self._iota_keys, eq, iota_grid)
        profiles = get_profiles(
            self._iota_keys + ["ideal ballooning lambda"], eq, iota_grid
        )
        self._constants = {
            "lambda0": self._lambda0,
            "w0": self._w0,
            "w1": self._w1,
            "rho": self._rho,
            "alpha": self._alpha,
            "zeta": jnp.linspace(
                -self._nturns * jnp.pi,
                +self._nturns * jnp.pi,
                +self._nturns * self._nzetaperturn,
            ),
            "zeta0": self._zeta0,
            "iota_transforms": transforms,
            "profiles": profiles,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the ballooning stability growth rate.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``.

        Returns
        -------
        lam : ndarray
            Ideal ballooning growth rate.

        """
        if constants is None:
            constants = self.constants
        eq = self.things[0]
        iota_data = compute_fun(
            eq,
            self._iota_keys,
            params,
            constants["iota_transforms"],
            constants["profiles"],
        )
        iota_grid = constants["iota_transforms"]["grid"]

        def get(key):
            x = iota_grid.compress(iota_data[key])
            return x[:-1] if self._add_lcfs else x

        iota = get("iota")
        # TODO(#1243): Upgrade this to use _map_clebsch_coordinates once
        #  the note in _L_partial_sum method is resolved.
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=iota,
            params=params,
        )
        data = {
            key: grid.expand(get(key))
            for key in self._iota_keys
            if (key != "iota" and key != "a")
        }
        data["iota"] = grid.expand(iota)
        data["a"] = iota_data["a"]
        data = compute_fun(
            eq,
            ["ideal ballooning lambda"],
            params,
            transforms=get_transforms(
                ["ideal ballooning lambda"], eq, grid, jitable=True
            ),
            profiles=constants["profiles"],
            data=data,
            zeta0=constants["zeta0"],
            Neigvals=self._Neigvals,
        )
        lam = data["ideal ballooning lambda"]
        lambda0, w0, w1 = constants["lambda0"], constants["w0"], constants["w1"]
        # shifted ReLU
        lam = (lam - lambda0) * (lam >= lambda0)
        lam = w0 * lam.sum(axis=(-1, -2, -3)) + w1 * lam.max(axis=(-1, -2, -3))
        return lam
