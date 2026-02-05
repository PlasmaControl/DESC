"""Objectives for ITG turbulence optimization."""

import numpy as np

from desc.backend import jnp, root_scalar
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
from desc.utils import Timer, setdefault

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

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

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

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

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

    Note: Requires PyTorch for loading model weights (``pip install torch``).

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
    solve_length_at_compute : bool, optional
        If True, re-solve for exact poloidal_turns per rho at compute time to
        maintain the target flux tube length as iota changes during optimization.
        More accurate but slower (loses rho-batching). Default False.

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
        "_solve_length_at_compute",
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
        solve_length_at_compute=False,
        name="NN ITG Proxy",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        # Check for torch dependency (required for loading .pth model weights)
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "NNITGProxy requires PyTorch for loading model weights. "
                "Install with: pip install torch"
            )

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
        self._solve_length_at_compute = solve_length_at_compute
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

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

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

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

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

        if verbose > 0:
            print("Solving for poloidal turns")
        timer.start("Solving poloidal turns")

        poloidal_turns = solve_poloidal_turns_for_length(
            length_fn, target_length, x0_guess=2.0 * abs(iota_ref)
        )

        timer.stop("Solving poloidal turns")
        if verbose > 1:
            timer.disp("Solving poloidal turns")

        # Convert to toroidal_turns for dynamic adaptation during optimization.
        # As iota changes, poloidal_turns = toroidal_turns * |iota| adapts to
        # maintain approximately the same physical flux tube length.
        toroidal_turns = poloidal_turns / abs(iota_ref)

        if verbose > 0:
            print(
                f"NNITGProxy: target L={target_length:.2f}, "
                f"toroidal_turns={toroidal_turns:.4f}, "
                f"achieved L={length_fn(poloidal_turns):.2f}"
            )

        if verbose > 0:
            print("Loading neural network weights")
        timer.start("Loading NN weights")

        # Load model weights and JIT-compiled forward functions
        nn_weights, jit_forward, ensemble_weights, jit_forwards = (
            self._load_model_weights(verbose)
        )

        timer.stop("Loading NN weights")
        if verbose > 1:
            timer.disp("Loading NN weights")

        self._constants = {
            "rho": self._rho,
            "alpha": self._alpha,
            "toroidal_turns": toroidal_turns,
            "target_length": target_length,
            "a": minor_radius,
            "iota_transforms": transforms,
            "profiles": profiles,
            "a_over_LT": self._a_over_LT,
            "a_over_Ln": self._a_over_Ln,
            "nn_weights": nn_weights,
            "jit_forward": jit_forward,
            "ensemble_weights": ensemble_weights,
            "jit_forwards": jit_forwards,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def _load_model_weights(self, verbose):
        """Load neural network weights and JIT-compiled forward functions.

        Parameters
        ----------
        verbose : int
            Verbosity level for printing.

        Returns
        -------
        nn_weights : dict or None
            Single model weights, or None if using ensemble.
        jit_forward : callable or None
            JIT-compiled forward function for single model.
        ensemble_weights : list or None
            List of ensemble model weights, or None if using single model.
        jit_forwards : list or None
            List of JIT-compiled forward functions for ensemble.

        Raises
        ------
        ValueError
            If neither model_path nor ensemble_dir is specified.

        Notes
        -----
        For ensemble loading, auto-detects the most common pre_method among
        the top-k models from the CSV file.
        """
        if self._ensemble_dir is not None:
            # Auto-detect pre_method from CSV (most common among top-k)
            pre_method = 0  # Default
            try:
                import pandas as pd

                df = pd.read_csv(self._ensemble_csv)
                if "p:pre_method" in df.columns:
                    df_valid = df[df["objective"] != "F"].copy()
                    if len(df_valid) > 0:
                        df_valid["objective"] = df_valid["objective"].astype(float)
                        df_sorted = df_valid.sort_values(
                            "objective", ascending=False
                        ).head(self._n_ensemble)
                        pre_method = int(df_sorted["p:pre_method"].mode().iloc[0])
                        if verbose > 0:
                            print(f"NNITGProxy: Auto-detected pre_method={pre_method}")
            except Exception:
                pass  # Use default

            ensemble_weights, jit_forwards = _load_ensemble_weights_cached(
                self._ensemble_dir,
                self._ensemble_csv,
                self._n_ensemble,
                pre_method=pre_method,
                verbose=(verbose > 0),
            )
            if verbose > 0:
                print(f"NNITGProxy: Loaded {len(ensemble_weights)} ensemble models")
            return None, None, ensemble_weights, jit_forwards

        if self._model_path is not None:
            nn_weights, jit_forward = _load_nn_weights(self._model_path)
            return nn_weights, jit_forward, None, None

        raise ValueError("Either model_path or ensemble_dir must be specified")

    def _map_pest_to_desc_coords(self, eq, params, rho_flat, theta_pest_flat, zeta_flat):
        """Map (rho, theta_PEST, zeta) to DESC theta coordinates.

        Returns
        -------
        theta_desc_flat : ndarray
            DESC theta coordinates with wrapping fixed for continuity.
        """
        coords_pest = jnp.column_stack([rho_flat, theta_pest_flat, zeta_flat])
        desc_coords = eq.map_coordinates(
            coords_pest,
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            params=params,
            tol=1e-10,
            maxiter=50,
        )
        theta_desc_flat = desc_coords[:, 1]
        # Fix theta wrapping for continuity
        theta_desc_flat = theta_desc_flat + 2 * jnp.pi * jnp.round(
            (theta_pest_flat - theta_desc_flat) / (2 * jnp.pi)
        )
        return theta_desc_flat

    def _compute_gx_features_on_grid(
        self, eq, params, rho_flat, theta_desc_flat, zeta_flat,
        iota_flat, iota_r_flat, minor_radius, gx_keys
    ):
        """Compute GX features on a field-line grid.

        Returns
        -------
        gx_data : dict
            Dictionary with GX features and gradpar.
        """
        grid = Grid(
            jnp.column_stack([rho_flat, theta_desc_flat, zeta_flat]),
            sort=False, jitable=True,
        )
        transforms = get_transforms(gx_keys + ["gx_gradpar"], eq, grid, jitable=True)
        profiles = get_profiles(gx_keys + ["gx_gradpar"], eq, grid)

        return compute_fun(
            eq,
            gx_keys + ["gx_gradpar"],
            params,
            transforms,
            profiles,
            data={
                "a": minor_radius,
                "iota": iota_flat,
                "iota_r": iota_r_flat,
                "rho": rho_flat,
            },
        )

    def _run_cnn_inference_for_rho(
        self, gradpar_2d, signals_2d, theta_pest_offset,
        a_over_LT, a_over_Ln, nn_weights, ensemble_weights, num_alpha,
        jit_forward=None, jit_forwards=None,
        return_std=False,
    ):
        """Run CNN inference for a single rho slice.

        Parameters
        ----------
        gradpar_2d : ndarray, shape (nz, num_alpha)
        signals_2d : ndarray, shape (7, nz, num_alpha)
        theta_pest_offset : ndarray, shape (nz,)
        jit_forward : callable or None
            JIT-compiled forward function for single model.
        jit_forwards : list of callable or None
            List of JIT-compiled forward functions for ensemble.
        return_std : bool, optional
            If True and using ensemble, return std of predictions.

        Returns
        -------
        Q_per_alpha : ndarray, shape (num_alpha,)
            Heat flux for each field line (alpha).
        signals_batch : ndarray, shape (num_alpha, 7, npoints)
            Resampled signals for return_signals mode.
        Q_std_per_alpha : ndarray, shape (num_alpha,), optional
            Std of heat flux. Only returned if return_std=True and using ensemble.
        """
        arclength_2d = compute_arclength_via_gradpar(gradpar_2d, theta_pest_offset)
        _, signals_resampled = resample_to_uniform_arclength(
            arclength_2d, signals_2d, self._npoints
        )

        # Transpose to (num_alpha, 7, npoints) for batched CNN
        signals_batch = jnp.transpose(signals_resampled, (2, 0, 1)).astype(jnp.float32)

        scalars_batch = jnp.broadcast_to(
            jnp.array([[a_over_LT, a_over_Ln]], dtype=jnp.float32),
            (num_alpha, 2),
        )

        # Use JIT-compiled functions when available
        if jit_forwards is not None:
            # Ensemble with JIT-compiled functions
            predictions = [f(signals_batch, scalars_batch) for f in jit_forwards]
            stacked = jnp.stack(predictions)
            log_Q = jnp.mean(stacked, axis=0)
            if return_std:
                std_log_Q = jnp.std(stacked, axis=0)
                Q_per_alpha = jnp.exp(log_Q[:, 0])
                Q_std_per_alpha = Q_per_alpha * std_log_Q[:, 0]
                return Q_per_alpha, signals_batch, Q_std_per_alpha
        elif jit_forward is not None:
            # Single model with JIT
            log_Q = jit_forward(signals_batch, scalars_batch)
        elif ensemble_weights is not None:
            # Fallback: ensemble without JIT
            if return_std:
                log_Q, std_log_Q = _ensemble_forward(
                    signals_batch, scalars_batch, ensemble_weights, return_std=True
                )
                Q_per_alpha = jnp.exp(log_Q[:, 0])
                Q_std_per_alpha = Q_per_alpha * std_log_Q[:, 0]
                return Q_per_alpha, signals_batch, Q_std_per_alpha
            log_Q = _ensemble_forward(signals_batch, scalars_batch, ensemble_weights)
        else:
            # Fallback: single model without JIT
            log_Q = _cyclic_invariant_forward(signals_batch, scalars_batch, nn_weights)

        Q_per_alpha = jnp.exp(log_Q[:, 0])
        if return_std:
            # Single model has no ensemble uncertainty
            Q_std_per_alpha = jnp.zeros_like(Q_per_alpha)
            return Q_per_alpha, signals_batch, Q_std_per_alpha
        return Q_per_alpha, signals_batch

    def _build_return_value(
        self, Q_all_per_alpha, all_signals, feature_names, return_signals,
        return_per_alpha, Q_std_all_per_alpha=None,
    ):
        """Build final return value with optional signals info.

        Parameters
        ----------
        Q_all_per_alpha : list of ndarray
            List of Q arrays per rho, each shape (num_alpha,).
        all_signals : list or None
            List of signal arrays if return_signals, else None.
        feature_names : list
            Names of the 7 GX features.
        return_signals : bool
            Whether to include signals in return.
        return_per_alpha : bool
            If True, return Q per field line. If False, average over alpha.
        Q_std_all_per_alpha : list of ndarray or None
            List of std arrays per rho if return_std, else None.

        Returns
        -------
        Q : ndarray
            Shape (num_rho,) if return_per_alpha=False.
            Shape (num_rho, num_alpha) if return_per_alpha=True.
        Q_std : ndarray, optional
            Same shape as Q. Only returned if Q_std_all_per_alpha provided.
        signals_info : dict, optional
            Only if return_signals=True.
        """
        Q_stacked = jnp.stack(Q_all_per_alpha)  # (num_rho, num_alpha)

        if return_per_alpha:
            Q_result = Q_stacked  # (num_rho, num_alpha)
        else:
            Q_result = jnp.mean(Q_stacked, axis=-1)  # (num_rho,)

        # Handle std similarly
        Q_std_result = None
        if Q_std_all_per_alpha is not None:
            Q_std_stacked = jnp.stack(Q_std_all_per_alpha)  # (num_rho, num_alpha)
            if return_per_alpha:
                Q_std_result = Q_std_stacked
            else:
                # When averaging Q over alpha, propagate std via quadrature
                # std(mean) = sqrt(sum(std_i^2)) / N
                Q_std_result = jnp.sqrt(jnp.mean(Q_std_stacked**2, axis=-1))

        # Build return tuple
        result = [Q_result]
        if Q_std_result is not None:
            result.append(Q_std_result)
        if return_signals:
            z_uniform = jnp.linspace(-jnp.pi, jnp.pi, self._npoints + 1)[:-1]
            signals_info = {
                "z": z_uniform,
                "signals": jnp.stack(all_signals),
                "feature_names": feature_names,
            }
            result.append(signals_info)

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def compute(
        self, params, constants=None, return_signals=False, return_per_alpha=False,
        return_std=False,
    ):
        """Compute the NN ITG heat flux prediction.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom.
        constants : dict
            Dictionary of constant data. Defaults to self.constants.
        return_signals : bool, optional
            If True, also return the 7 GX input features used for CNN inference.
            Useful for debugging and validation. Default False.
        return_per_alpha : bool, optional
            If True, return Q values for each field line (alpha) separately,
            with shape (num_rho, num_alpha). If False (default), return the
            mean over alpha with shape (num_rho,).
        return_std : bool, optional
            If True, also return ensemble std of predictions. Requires ensemble
            model (not single model). The std represents model uncertainty.
            Default False.

        Returns
        -------
        Q : ndarray
            Predicted heat flux for each flux surface.
            Shape (num_rho,) if return_per_alpha=False (default).
            Shape (num_rho, num_alpha) if return_per_alpha=True.
        Q_std : ndarray, optional
            Only returned if return_std=True. Ensemble std of predictions,
            same shape as Q.
        signals_info : dict, optional
            Only returned if return_signals=True. Contains:
            - 'z': uniform arclength coordinates, shape (npoints,)
            - 'signals': GX features, shape (num_rho, num_alpha, 7, npoints)
            - 'feature_names': list of 7 feature names

        """
        if constants is None:
            constants = self.constants

        eq = self.things[0]
        rho_arr = jnp.asarray(constants["rho"])
        alpha_arr = jnp.asarray(constants["alpha"])
        toroidal_turns = constants["toroidal_turns"]
        minor_radius = constants["a"]
        a_over_LT = constants["a_over_LT"]
        a_over_Ln = constants["a_over_Ln"]
        nn_weights = constants["nn_weights"]
        jit_forward = constants.get("jit_forward")
        ensemble_weights = constants["ensemble_weights"]
        jit_forwards = constants.get("jit_forwards")

        num_rho = len(rho_arr)
        num_alpha = len(alpha_arr)
        nz = self._nz_internal

        # Compute iota at each rho (compress from grid to unique rho values)
        iota_grid = constants["iota_transforms"]["grid"]
        iota_data = compute_fun(
            eq,
            ["iota", "iota_r"],
            params,
            constants["iota_transforms"],
            constants["profiles"],
        )
        iota_arr = iota_grid.compress(iota_data["iota"])
        iota_r_arr = iota_grid.compress(iota_data["iota_r"])

        # GX features to compute (order matches CNN input)
        gx_keys = [
            "gx_bmag",
            "gx_gbdrift",
            "gx_cvdrift",
            "gx_gbdrift0_over_shat",
            "gx_gds2",
            "gx_gds21_over_shat",
            "gx_gds22_over_shat_squared",
        ]
        feature_names = [
            "bmag",
            "gbdrift",
            "cvdrift",
            "gbdrift0_over_shat",
            "gds2",
            "gds21_over_shat",
            "gds22_over_shat_squared",
        ]

        # Branch based on solve_length_at_compute option
        if self._solve_length_at_compute:
            return self._compute_with_length_solving(
                params, eq, rho_arr, alpha_arr, iota_arr, iota_r_arr,
                minor_radius, a_over_LT, a_over_Ln, nn_weights, jit_forward,
                ensemble_weights, jit_forwards, gx_keys, feature_names, constants,
                return_signals, return_per_alpha, return_std,
            )

        # =====================================================================
        # BATCHED GRID CONSTRUCTION (default fast path)
        # Create all (nz x num_rho x num_alpha) points in one grid
        # Using zeta (toroidal angle) parameterization for dynamic iota adaptation
        # =====================================================================

        # Uniform zeta offset (same for all rho/alpha)
        # This keeps toroidal extent fixed; poloidal extent adapts with iota
        zeta_offset = jnp.linspace(
            -jnp.pi * toroidal_turns, jnp.pi * toroidal_turns, nz
        )

        # Create meshgrid: shapes will be (nz, num_rho, num_alpha)
        rho_mesh = jnp.broadcast_to(rho_arr[None, :, None], (nz, num_rho, num_alpha))
        alpha_mesh = jnp.broadcast_to(alpha_arr[None, None, :], (nz, num_rho, num_alpha))
        zeta_mesh = jnp.broadcast_to(zeta_offset[:, None, None], (nz, num_rho, num_alpha))

        # iota varies by rho: iota_mesh[i,j,k] = iota_arr[j]
        iota_mesh = jnp.broadcast_to(iota_arr[None, :, None], (nz, num_rho, num_alpha))
        iota_r_mesh = jnp.broadcast_to(iota_r_arr[None, :, None], (nz, num_rho, num_alpha))

        # Compute theta_PEST from field line equation: theta_PEST = alpha + iota * zeta
        # This varies by rho (different iota) while keeping toroidal extent fixed
        theta_pest_mesh = alpha_mesh + iota_mesh * zeta_mesh

        # Flatten for coordinate mapping: total points = nz x num_rho x num_alpha
        rho_flat = rho_mesh.flatten()
        theta_pest_flat = theta_pest_mesh.flatten()
        zeta_flat = zeta_mesh.flatten()
        iota_flat = iota_mesh.flatten()
        iota_r_flat = iota_r_mesh.flatten()

        # Map PEST to DESC coordinates and compute GX features
        theta_desc_flat = self._map_pest_to_desc_coords(
            eq, params, rho_flat, theta_pest_flat, zeta_flat
        )
        gx_data = self._compute_gx_features_on_grid(
            eq, params, rho_flat, theta_desc_flat, zeta_flat,
            iota_flat, iota_r_flat, minor_radius, gx_keys
        )

        # Reshape GX data to (nz, num_rho, num_alpha)
        def reshape_gx(arr):
            return arr.reshape(nz, num_rho, num_alpha)

        gx_features = {k: reshape_gx(gx_data[k]) for k in gx_keys}
        gradpar_3d = reshape_gx(gx_data["gx_gradpar"])

        # =====================================================================
        # RESAMPLING AND CNN INFERENCE
        # Loop over rho (arclength varies per rho due to different iota)
        # =====================================================================

        Q_all_per_alpha = []
        Q_std_all_per_alpha = [] if return_std else None
        all_signals = [] if return_signals else None

        for i_rho in range(num_rho):
            gradpar_2d = gradpar_3d[:, i_rho, :]
            signals_2d = jnp.stack([gx_features[k][:, i_rho, :] for k in gx_keys])
            theta_pest_offset_rho = iota_arr[i_rho] * zeta_offset

            result = self._run_cnn_inference_for_rho(
                gradpar_2d, signals_2d, theta_pest_offset_rho,
                a_over_LT, a_over_Ln, nn_weights, ensemble_weights, num_alpha,
                jit_forward=jit_forward, jit_forwards=jit_forwards,
                return_std=return_std,
            )
            if return_std:
                Q_per_alpha, signals_batch, Q_std_per_alpha = result
                Q_std_all_per_alpha.append(Q_std_per_alpha)
            else:
                Q_per_alpha, signals_batch = result
            Q_all_per_alpha.append(Q_per_alpha)
            if return_signals:
                all_signals.append(signals_batch)

        return self._build_return_value(
            Q_all_per_alpha, all_signals, feature_names, return_signals,
            return_per_alpha, Q_std_all_per_alpha,
        )

    def _compute_with_length_solving(
        self, params, eq, rho_arr, alpha_arr, iota_arr, iota_r_arr,
        minor_radius, a_over_LT, a_over_Ln, nn_weights, jit_forward,
        ensemble_weights, jit_forwards, gx_keys, feature_names, constants,
        return_signals, return_per_alpha, return_std=False,
    ):
        """Compute with exact flux tube length solving per rho.

        This is the accurate but slower code path used when solve_length_at_compute=True.
        For each rho, we solve for the exact poloidal_turns that achieves target_length,
        then compute GX features and run CNN inference.
        """
        num_rho = len(rho_arr)
        num_alpha = len(alpha_arr)
        nz = self._nz_internal
        target_length = constants["target_length"]
        toroidal_turns = constants["toroidal_turns"]

        Q_all_per_alpha = []
        Q_std_all_per_alpha = [] if return_std else None
        all_signals = [] if return_signals else None

        for i_rho in range(num_rho):
            rho = rho_arr[i_rho]
            iota = iota_arr[i_rho]
            iota_r = iota_r_arr[i_rho]

            # Solve for exact poloidal_turns that achieves target_length
            x0 = toroidal_turns * jnp.abs(iota)

            def length_residual(poloidal_turns):
                """Compute length - target_length for root finding."""
                theta_pest_offset = jnp.linspace(
                    -jnp.pi * poloidal_turns, jnp.pi * poloidal_turns, nz
                )
                theta_pest = theta_pest_offset  # alpha=0 reference
                zeta = theta_pest_offset / iota
                rho_grid = jnp.full(nz, rho)

                theta_desc = self._map_pest_to_desc_coords(
                    eq, params, rho_grid, theta_pest, zeta
                )
                grid = Grid(
                    jnp.column_stack([rho_grid, theta_desc, zeta]),
                    sort=False, jitable=True
                )
                data = compute_fun(
                    eq, ["gx_gradpar"], params,
                    get_transforms(["gx_gradpar"], eq, grid, jitable=True),
                    get_profiles(["gx_gradpar"], eq, grid),
                    data={
                        "a": minor_radius,
                        "iota": jnp.full(nz, iota),
                        "iota_r": jnp.full(nz, iota_r),
                        "rho": rho_grid,
                    },
                )
                length = jnp.abs(jnp.trapezoid(1.0 / data["gx_gradpar"], theta_pest))
                return length - target_length

            poloidal_turns = root_scalar(length_residual, x0, tol=1e-4, maxiter=10)

            # Compute GX features for all alphas at this rho
            theta_pest_offset = jnp.linspace(
                -jnp.pi * poloidal_turns, jnp.pi * poloidal_turns, nz
            )

            # Build grid for all alphas
            alpha_mesh = jnp.broadcast_to(alpha_arr[None, :], (nz, num_alpha))
            theta_offset_mesh = jnp.broadcast_to(
                theta_pest_offset[:, None], (nz, num_alpha)
            )
            theta_pest_mesh = alpha_mesh + theta_offset_mesh
            zeta_mesh = theta_offset_mesh / iota

            rho_flat = jnp.full(nz * num_alpha, rho)
            theta_pest_flat = theta_pest_mesh.flatten()
            zeta_flat = zeta_mesh.flatten()
            iota_flat = jnp.full(nz * num_alpha, iota)
            iota_r_flat = jnp.full(nz * num_alpha, iota_r)

            theta_desc_flat = self._map_pest_to_desc_coords(
                eq, params, rho_flat, theta_pest_flat, zeta_flat
            )
            gx_data = self._compute_gx_features_on_grid(
                eq, params, rho_flat, theta_desc_flat, zeta_flat,
                iota_flat, iota_r_flat, minor_radius, gx_keys
            )

            # Reshape to (nz, num_alpha) and run CNN
            gradpar_2d = gx_data["gx_gradpar"].reshape(nz, num_alpha)
            signals_2d = jnp.stack(
                [gx_data[k].reshape(nz, num_alpha) for k in gx_keys]
            )

            result = self._run_cnn_inference_for_rho(
                gradpar_2d, signals_2d, theta_pest_offset,
                a_over_LT, a_over_Ln, nn_weights, ensemble_weights, num_alpha,
                jit_forward=jit_forward, jit_forwards=jit_forwards,
                return_std=return_std,
            )
            if return_std:
                Q_per_alpha, signals_batch, Q_std_per_alpha = result
                Q_std_all_per_alpha.append(Q_std_per_alpha)
            else:
                Q_per_alpha, signals_batch = result
            Q_all_per_alpha.append(Q_per_alpha)
            if return_signals:
                all_signals.append(signals_batch)

        return self._build_return_value(
            Q_all_per_alpha, all_signals, feature_names, return_signals,
            return_per_alpha, Q_std_all_per_alpha,
        )
