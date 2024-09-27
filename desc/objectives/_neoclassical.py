"""Objectives for targeting neoclassical transport."""

import numpy as np

from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer

from ..integrals.quad_utils import chebgauss2
from .objective_funs import _Objective
from .utils import _parse_callable_target_bounds


class EffectiveRipple(_Objective):
    """The effective ripple is a proxy for neoclassical transport.

    The 3D geometry of the magnetic field in stellarators produces local magnetic
    wells that lead to bad confinement properties with enhanced radial drift,
    especially for trapped particles. Neoclassical (thermal) transport can become the
    dominant transport channel in stellarators which are not optimized to reduce it.
    The effective ripple is a proxy, measuring the effective modulation amplitude of the
    magnetic field averaged along a magnetic surface, which can be used to optimize for
    stellarators with improved confinement. It is targeted as a flux surface function.

    References
    ----------
    https://doi.org/10.1063/1.873749.
    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. If a callable, should take a
        single argument ``rho`` and return the desired value of the profile at those
        locations. Defaults to 0.
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to Objective.dim_f.
        If a callable, each should take a single argument ``rho`` and return the
        desired bound (lower or upper) of the profile at those locations.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to Objective.dim_f
    normalize : bool, optional
        This quantity is already normalized so this parameter is ignored.
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is ``True`` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    rho : ndarray
        Unique coordinate values specifying flux surfaces to compute on.
    alpha : ndarray
        Unique coordinate values specifying field line labels to compute on.
    knots_per_transit : int
        Number of points per toroidal transit at which to sample data along field
        line. Default is 100.
    num_transit : int
        Number of toroidal transits to follow field line.
        For axisymmetric devices, one poloidal transit is sufficient. Otherwise,
        more transits will give more accurate result, with diminishing returns.
    num_quad : int
        Resolution for quadrature of bounce integrals. Default is 32.
    num_pitch : int
        Resolution for quadrature over velocity coordinate. Default 50.
    batch : bool
        Whether to vectorize part of the computation. Default is true.
    num_well : int
        Maximum number of wells to detect for each pitch and field line.
        Default is to detect all wells, but due to limitations in JAX this option
        may consume more memory. Specifying a number that tightly upper bounds
        the number of wells will increase performance.
    name : str, optional
        Name of the objective function.
    jac_chunk_size : int , optional
        Will calculate the Jacobian for this objective ``jac_chunk_size``
        columns at a time, instead of all at once. The memory usage of the
        Jacobian calculation is roughly ``memory usage = m0 + m1*jac_chunk_size``:
        the smaller the chunk size, the less memory the Jacobian calculation
        will require (with some baseline memory usage). The time to compute the
        Jacobian is roughly ``t=t0 +t1/jac_chunk_size``, so the larger the
        ``jac_chunk_size``, the faster the calculation takes, at the cost of
        requiring more memory. A ``jac_chunk_size`` of 1 corresponds to the least
        memory intensive, but slowest method of calculating the Jacobian.
        If None, it will use the largest size i.e ``obj.dim_x``.

    """

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Effective ripple ε: "

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
        rho=np.linspace(0.5, 1, 3),
        alpha=np.array([0]),
        *,
        knots_per_transit=100,
        num_transit=10,
        num_quad=32,
        num_pitch=50,
        batch=True,
        num_well=None,
        name="Effective ripple",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0.0

        rho, alpha = np.atleast_1d(rho, alpha)
        self._dim_f = rho.size
        self._keys_1dr = [
            "iota",
            "iota_r",
            "<|grad(rho)|>",
            "min_tz |B|",
            "max_tz |B|",
            "R0",  # TODO: GitHub PR #1094
        ]
        self._constants = {
            "quad_weights": 1,
            "rho": rho,
            "alpha": alpha,
            "zeta": np.linspace(
                0, 2 * np.pi * num_transit, knots_per_transit * num_transit
            ),
            "quad": chebgauss2(num_quad),
        }
        self._hyperparameters = {
            "num_pitch": num_pitch,
            "batch": batch,
            "num_well": num_well,
        }

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
        self._grid_1dr = LinearGrid(
            rho=self._constants["rho"], M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
        )
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, self._constants["rho"]
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid_1dr
        )
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + ["effective ripple"], eq, self._grid_1dr
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the effective ripple.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``.

        Returns
        -------
        result : ndarray
            Effective ripple as a function of the flux surface label.

        """
        if constants is None:
            constants = self.constants
        eq = self.things[0]
        # TODO: compute all deps of effective ripple here
        data = compute_fun(
            eq,
            self._keys_1dr,
            params,
            constants["transforms_1dr"],
            constants["profiles"],
        )
        # TODO: interpolate all deps to this grid with fft utilities from fourier bounce
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=self._grid_1dr.compress(data["iota"]),
            params=params,
        )
        data = {
            key: (
                grid.copy_data_from_other(data[key], self._grid_1dr)
                if key != "R0"
                else data[key]
            )
            for key in self._keys_1dr
        }
        data = compute_fun(
            eq,
            "effective ripple",
            params,
            get_transforms("effective ripple", eq, grid, jitable=True),
            constants["profiles"],
            data=data,
            quad=constants["quad"],
            **self._hyperparameters,
        )
        return grid.compress(data["effective ripple"])
