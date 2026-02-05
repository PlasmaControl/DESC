"""Objectives for ITG turbulence optimization."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute._turbulence import (
    _cyclic_invariant_forward,
    _ensemble_forward,
    _load_ensemble_weights_cached,
    _load_nn_weights,
    compute_arclength_via_gradpar,
    resample_to_uniform_arclength,
    solve_poloidal_turns_for_length,
)
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid
from desc.utils import setdefault

from .objective_funs import _Objective, collect_docs


class ITGProxy(_Objective):
    """ITG turbulence proxy from Landreman et al. 2025.

    This objective computes the analytical proxy for Ion Temperature Gradient
    (ITG) turbulence heat flux. The proxy is defined as:

    f_Q = mean((sigmoid(cvdrift) + 0.2) * |grad_x|^3 / B)

    Lower values indicate configurations with reduced ITG turbulence.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to be optimized.
    rho : float or array-like
        Flux surface(s) to evaluate on.
    alpha : float or array-like, optional
        Field line labels to evaluate. Default is 8 field lines in [0, pi]
        for non-axisymmetric cases, or alpha=0 for axisymmetric.
    nturns : int, optional
        Number of toroidal transits per field line. Default is 3.
    nzetaperturn : int, optional
        Number of points per toroidal transit. Default is 100.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_iota_keys",
        "_nturns",
        "_nzetaperturn",
        "_add_lcfs",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "ITG Proxy f_Q: "

    def __init__(
        self,
        eq,
        rho,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        alpha=None,
        nturns=3,
        nzetaperturn=100,
        name="ITG Proxy",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        self._nturns = nturns
        self._nzetaperturn = nzetaperturn
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
            self._iota_keys + ["ITG proxy integrand"], eq, iota_grid
        )
        self._constants = {
            "rho": self._rho,
            "alpha": self._alpha,
            "zeta": jnp.linspace(
                -self._nturns * jnp.pi,
                +self._nturns * jnp.pi,
                +self._nturns * self._nzetaperturn,
            ),
            "iota_transforms": transforms,
            "profiles": profiles,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the ITG turbulence proxy.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom.
        constants : dict
            Dictionary of constant data. Defaults to self.constants.

        Returns
        -------
        f_Q : ndarray
            ITG proxy values for each flux surface.

        """
        if constants is None:
            constants = self.constants
        eq = self.things[0]

        # Compute iota on the iota grid
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

        # Create field-aligned grid
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=iota,
            params=params,
        )

        # Prepare data for the compute function
        data = {
            key: grid.expand(get(key))
            for key in self._iota_keys
            if key not in ("iota", "a")
        }
        data["iota"] = grid.expand(iota)
        data["a"] = iota_data["a"]

        # Compute the ITG proxy integrand
        data = compute_fun(
            eq,
            ["ITG proxy integrand"],
            params,
            transforms=get_transforms(["ITG proxy integrand"], eq, grid, jitable=True),
            profiles=constants["profiles"],
            data=data,
        )

        # Average over field line to get per-flux-surface values
        integrand = data["ITG proxy integrand"]
        num_rho = len(constants["rho"])
        num_alpha = len(constants["alpha"])
        num_zeta = len(constants["zeta"])

        integrand_reshaped = integrand.reshape(num_zeta, num_rho, num_alpha)
        f_Q = jnp.mean(integrand_reshaped, axis=(0, 2))

        return f_Q


class NNITGProxy(_Objective):
    """Neural network ITG turbulence proxy from Landreman et al. 2025.

    Uses a trained CNN to predict ITG-driven heat flux based on the 7 GX
    geometric coefficients along a flux tube. The CNN was trained on over
    200,000 GX gyrokinetic simulations.

    The model takes as input:
    - 7 geometric features along the field line (bmag, gbdrift, cvdrift, etc.)
    - Temperature gradient scale length (a/LT)
    - Density gradient scale length (a/Ln)

    And predicts the gyro-Bohm normalized heat flux Q.

    Two modes are supported:
    - **Single model mode**: Using `model_path` for a single .pt/.pth file
    - **Ensemble mode**: Using `ensemble_dir` and `ensemble_csv` for multiple models

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    rho : float or array-like
        Flux surface(s) to evaluate on. Must be specified (no default).
    alpha : float or array-like, optional
        Field line labels to evaluate. Values should be in [0, 2pi).
        Default is alpha=0.
    npoints : int, optional
        Number of points along the field line for CNN input. Grid is periodic
        with endpoint excluded (i.e., uniform in [-π, π)). Default is 96.
    nz_internal : int, optional
        Number of internal grid points for accurate arclength computation.
        Default is 1001.
    target_flux_tube_length : float, optional
        Target flux tube length. Default is 75.4 (= 4*pi*6).
    a_over_LT : float, optional
        Normalized inverse temperature gradient scale length. Default 3.0.
    a_over_Ln : float, optional
        Normalized inverse density gradient scale length. Default 1.0.
    model_path : str, optional
        Path to the trained model weights (single-model mode).
    ensemble_dir : str, optional
        Path to directory containing ensemble model files (.pth).
    ensemble_csv : str, optional
        Path to results.csv with DeepHyper scores for model selection.
        Required if ensemble_dir is specified.
    n_ensemble : int, optional
        Number of top-performing models to use for ensemble. Default 10.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_gx_keys",
        "_iota_keys",
        "_npoints",
        "_nz_internal",
        "_target_flux_tube_length",
        "_a_over_LT",
        "_a_over_Ln",
        "_model_path",
        "_ensemble_dir",
        "_ensemble_csv",
        "_n_ensemble",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "NN ITG Q: "

    def __init__(
        self,
        eq,
        rho,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        alpha=None,
        npoints=96,
        nz_internal=1001,
        target_flux_tube_length=75.4,
        a_over_LT=3.0,
        a_over_Ln=1.0,
        model_path=None,
        ensemble_dir=None,
        ensemble_csv=None,
        n_ensemble=10,
        name="NN ITG Proxy",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        # Validate ensemble parameters
        if ensemble_dir is not None and ensemble_csv is None:
            raise ValueError("ensemble_csv must be specified when using ensemble_dir")

        self._npoints = npoints
        self._nz_internal = nz_internal
        self._target_flux_tube_length = target_flux_tube_length
        self._a_over_LT = a_over_LT
        self._a_over_Ln = a_over_Ln
        self._model_path = model_path
        self._ensemble_dir = ensemble_dir
        self._ensemble_csv = ensemble_csv
        self._n_ensemble = n_ensemble
        self._rho = np.atleast_1d(rho)
        self._alpha = np.atleast_1d(setdefault(alpha, jnp.array([0.0])))

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
        # GX coefficient keys for CNN input
        self._gx_keys = [
            "gx_bmag",
            "gx_gbdrift",
            "gx_cvdrift",
            "gx_gbdrift0_over_shat",
            "gx_gds2",
            "gx_gds21_over_shat",
            "gx_gds22_over_shat_squared",
            "gx_gradpar",
        ]
        # Iota and related quantities needed for coordinate mapping
        self._iota_keys = ["iota", "iota_r", "shear", "a", "p_r"]

        eq = self.things[0]
        iota_grid = LinearGrid(
            rho=self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
        )
        self._dim_f = len(self._rho)
        transforms = get_transforms(self._iota_keys, eq, iota_grid)
        profiles = get_profiles(self._iota_keys + self._gx_keys, eq, iota_grid)

        # Compute target flux tube length
        eq_data = eq.compute(["a"])
        minor_radius = float(eq_data["a"])

        target_length = self._target_flux_tube_length

        # Compute iota at reference rho for poloidal_turns computation
        iota_data = compute_fun(
            eq, self._iota_keys, eq.params_dict, transforms, profiles
        )
        iota = iota_grid.compress(iota_data["iota"])

        # Use first rho/alpha as reference for solving poloidal_turns
        rho_ref = float(self._rho[0])
        alpha_ref = float(self._alpha[0])
        iota_ref = float(iota[0])
        iota_r_ref = float(iota_grid.compress(iota_data["iota_r"])[0])

        def length_fn(poloidal_turns):
            """Compute flux tube length for given poloidal_turns."""
            # Uniform θ_PEST grid
            theta_pest_offset = jnp.linspace(
                -jnp.pi * poloidal_turns, jnp.pi * poloidal_turns, self._nz_internal
            )
            theta_pest = alpha_ref + theta_pest_offset
            zeta = theta_pest_offset / iota_ref

            # Map PEST → DESC coordinates
            rho_arr = jnp.full(self._nz_internal, rho_ref)
            coords_pest = jnp.column_stack([rho_arr, theta_pest, zeta])
            desc_coords = eq.map_coordinates(
                coords_pest,
                inbasis=("rho", "theta_PEST", "zeta"),
                outbasis=("rho", "theta", "zeta"),
                params=eq.params_dict,
                tol=1e-10,
                maxiter=50,
            )
            theta_desc = desc_coords[:, 1]

            # Fix theta wrapping for continuity
            theta_desc = theta_desc + 2 * jnp.pi * jnp.round(
                (theta_pest - theta_desc) / (2 * jnp.pi)
            )

            # Create grid and compute gradpar
            grid = Grid(
                jnp.column_stack([rho_arr, theta_desc, zeta]), sort=False, jitable=True
            )
            data = compute_fun(
                eq,
                ["gx_gradpar"],
                eq.params_dict,
                get_transforms(["gx_gradpar"], eq, grid, jitable=True),
                get_profiles(["gx_gradpar"], eq, grid),
                data={
                    "a": minor_radius,
                    "iota": jnp.full(self._nz_internal, iota_ref),
                    "iota_r": jnp.full(self._nz_internal, iota_r_ref),
                    "rho": rho_arr,
                },
            )

            # L = ∫ dθ_PEST / |gradpar|
            return jnp.abs(jnp.trapezoid(1.0 / data["gx_gradpar"], theta_pest))

        poloidal_turns = solve_poloidal_turns_for_length(
            length_fn, target_length, x0_guess=2.0 * abs(iota_ref)
        )

        if verbose > 0:
            print(
                f"NNITGProxy: target L={target_length:.2f}, "
                f"poloidal_turns={poloidal_turns:.4f}, "
                f"achieved L={length_fn(poloidal_turns):.2f}"
            )

        # Load model weights (ensemble or single model)
        nn_weights, ensemble_weights = self._load_model_weights(verbose)

        self._constants = {
            "rho": self._rho,
            "alpha": self._alpha,
            "poloidal_turns": poloidal_turns,
            "target_length": target_length,
            "a": minor_radius,
            "iota_transforms": transforms,
            "profiles": profiles,
            "a_over_LT": self._a_over_LT,
            "a_over_Ln": self._a_over_Ln,
            "nn_weights": nn_weights,
            "ensemble_weights": ensemble_weights,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def _load_model_weights(self, verbose):
        """Load neural network weights (ensemble or single model).

        Parameters
        ----------
        verbose : int
            Verbosity level for printing.

        Returns
        -------
        nn_weights : dict or None
            Single model weights, or None if using ensemble.
        ensemble_weights : list or None
            List of ensemble model weights, or None if using single model.

        Raises
        ------
        ValueError
            If neither model_path nor ensemble_dir is specified.

        """
        if self._ensemble_dir is not None:
            ensemble_weights = _load_ensemble_weights_cached(
                self._ensemble_dir, self._ensemble_csv, self._n_ensemble
            )
            if verbose > 0:
                print(f"NNITGProxy: Loaded {len(ensemble_weights)} ensemble models")
            return None, ensemble_weights

        if self._model_path is not None:
            nn_weights = _load_nn_weights(self._model_path)
            return nn_weights, None

        raise ValueError("Either model_path or ensemble_dir must be specified")

    def compute(self, params, constants=None):
        """Compute the NN ITG heat flux prediction.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom.
        constants : dict
            Dictionary of constant data. Defaults to self.constants.

        Returns
        -------
        Q : ndarray
            Predicted heat flux for each flux surface, averaged over field lines.

        """
        if constants is None:
            constants = self.constants

        num_rho = len(constants["rho"])
        return jnp.ones(num_rho)
