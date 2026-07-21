"""Objectives for trapped energetic particle resonance."""

import numpy as np
from orthax.legendre import leggauss

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute._trapped_resonance import _build_eta_grid
from desc.compute.data_index import data_index
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import _parse_parameterization
from desc.grid import LinearGrid
from desc.integrals.bounce_integral import Bounce1D
from desc.utils import Timer

from ..integrals.quad_utils import (
    automorphism_sin,
    get_quadrature,
    grad_automorphism_sin,
)
from .objective_funs import _Objective
from .utils import _parse_callable_target_bounds


# New resonance objective from John Anthony Labbate
class TrappedResonance(_Objective):
    """Trapped energetic particle resonance penalty.

    Penalizes rational crossings of Omega_eta (the ratio between precessional
    motion and bounce frequency) to minimize trapped energetic particle radial
    motion due to resonances with magnetic field perturbations from omnigenity.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    num_rho : int, optional
        Number of flux surfaces.  Constructed as
        ``np.linspace(0, 1, num_rho + 1)[1:]``, giving ``num_rho``
        uniformly spaced surfaces from ``1/num_rho`` to ``1`` with
        spacing ``1/num_rho``.  Default is 10.
    num_eta : int, optional
        Number of uniformly spaced eta points in [0, 2*pi).
        Alpha values are derived per rho surface via
        ``alpha = eta * (N*nfp - iota*M) / nfp``.
        Default is 10.
    weight_method : {"linear", "bump"}, optional
        How to weight surfaces near resonance. ``"linear"`` uses 2-point linear
        interpolation between bracketing surfaces. ``"bump"`` uses a smooth
        normalized bump function. Default is ``"linear"``.
    Delta_Omega : float, optional
        Half-width of the resonance interval for ``weight_method="bump"``.
        If ``None``, defaults to wd_blur × the max |Ω[i+1]-Ω[i]| spacing.
        Ignored when ``weight_method="linear"``.
    wd_blur : float, optional
        Factor multiplying Delta_Omega in case where Delta_Omega = ``None``
        (see Delta_Omega). Otherwise is ignored.
        Defaults to 1.25.
    num_transit : float, optional
        2π * num_transits sets the extent of zeta for bounce integration.
        Defaults to 5.
    num_quad : int, optional
        Number of quadrature points utilized for any integration in this objective.
        Defaults to 32.
    num_pitch : int, optional
        Number of trapped particle pitches/Bcrit to consider, calculated in
        evenly-spaced intervals between Bmin,Bmax on each flux surface.
        Defaults to 16.
    KE_frac : array, optional
        Fraction of 3.5 MeV to use for the energetic particle kinetic energy.
        Defaults to np.array([1]).
    knots_per_transit : int, optional
        knots_per_transit * num_transits gives how many points to use in zeta grid.
        Defaults to 100.
    batch : bool, optional
        Whether or not to calculate multiple trapped particles simultaneously,
        especially for bounce integration.
        Defaults to True.
    num_well : int or None, optional
        Specify to return the first ``num_well`` pairs of bounce points for each
        pitch and field line. Default is ``None``, which will detect all wells,
        but due to current limitations in JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``, ``B`` are the poloidal and
        toroidal Fourier resolution of B, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
        The ``check_points`` or ``plot`` method is useful to select a reasonable
        value.

        If there were fewer wells detected along a field line than the size of the
        last axis of the returned arrays, then that axis is padded with zero.

        Defaults to None
    pitch_invs : array or None, optional
        If not None, sets pitch_invs (Bcrits) to specified value. If None, let's
        compute specify a linspace of num_pitch between Bmin and Bmax of each
        flux surface. Also causes ``compute`` to skip the phase-space average and
        return the raw per-(rho, pitch, well) resonance-physics dictionary instead
        of the phase-space-averaged objective.
        Defaults to None.
    N : int, optional
        Generalized omnigenous helicity. Each B contour closes on itself after
        traversing the torus M times toroidally and N times poloidally.
        Defaults to 0, which is a quasi-axisymmetric configuration.
    M : int, optional
        Generalized omnigenous helicity. Each B contour closes on itself after
        traversing the torus M times toroidally and N times poloidally.
        Defaults to 1, which is a quasi-axisymmetric configuration.
    p_max : int, optional
        Maximum numerator of rational Omega_eta considered. Rational Omega_eta
        will be considered for all combinations of p/q up to p_max/q_max.
        Defaults to 10.
    q_max : int, optional
        Maximum denominator of rational Omega_eta considered. Rational Omega_eta
        will be considered for all combinations of p/q up to p_max/q_max.
        Defaults to 10.
    res_range_min : float, optional
        Minimum value of rational Omega_eta to consider regardless of p and q.
        Defaults to -4.
    res_range_max : float, optional
        Maximum value of rational Omega_eta to consider regardless of p and q.
        Defaults to 4.
    fill_value : float, optional
        Value to set bounce integration outputs to if no well is found. Cannot
        use ``jnp.nan`` to retain optimization abilities. Cannot use 0 for
        confusion with other quantities and averages.
        Defaults to 11.0.
    stab_sacrifice : bool, optional
        If ``True``, multiply the island-width term by ``Omega_prime_s**2`` in the
        objective. If ``False``, omit that factor to preserve numerical stability.
        Defaults to ``False``.
    cropping_DOmega : bool, optional
        If ``True``, Delta_Omega calculation is clipped by
        ``0.01 * max(Omega_eta) < Delta_Omega < 0.10 * max(Omega_eta)``.
        This must be when using the ``bump`` weighting method and
        ``Delta_Omega = None`` case. Otherwise this quantity is ignored.
        Defaults to ``False``.
    bt_filter_flag : bool, optional
        If ``True``, zero out wells whose poloidal bounce width exceeds 2π
        (barely-trapped filter) before the resonance physics calculation.
        Defaults to ``False``.
    """

    _scalar = False
    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Trapped EP Resonance Penalty: "

    _static_attrs = _Objective._static_attrs + ["_hyperparameters", "_keys_1dr", "_key"]

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="TrappedResonance",
        jac_chunk_size=None,
        verbose=False,
        pitch_batch_size=1,
        surf_batch_size=1,
        num_rho=10,
        num_eta=10,
        weight_method="linear",
        Delta_Omega=None,
        wd_blur=1.25,
        num_transit=5,
        num_quad=32,
        num_pitch=16,
        KE_frac=np.array([1]),
        knots_per_transit=100,
        batch=True,
        num_well=None,
        pitch_invs=None,
        N=0,
        M=1,
        p_max=10,
        q_max=10,
        res_range_min=-4,
        res_range_max=4,
        fill_value=11,
        stab_sacrifice=False,
        cropping_DOmega=False,
        bt_filter_flag=False,
    ):
        if target is None and bounds is None:
            target = 1e-8
        self._num_rho = int(num_rho)
        self._num_eta = int(num_eta)
        if self._num_eta < 2:
            raise ValueError(f"num_eta must be >= 2, got {self._num_eta}.")
        if self._num_rho < 2:
            raise ValueError(f"num_rho must be >= 2, got {self._num_rho}.")

        self._constants = {"quad_weights": 1}
        self._constants["zeta"] = np.linspace(
            0, 2 * np.pi * num_transit, knots_per_transit * num_transit
        )

        self._hyperparameters = {
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "num_eta": self._num_eta,
            "batch": batch,
            "KE_frac": KE_frac,
            "pitch_invs": pitch_invs,
            "N": N,
            "M": M,
            "p_max": p_max,
            "q_max": q_max,
            "res_range_min": res_range_min,
            "res_range_max": res_range_max,
            "verbose": verbose,
            "pitch_batch_size": pitch_batch_size,
            "surf_batch_size": surf_batch_size,
            "num_transit": num_transit,
            "weight_method": weight_method,
            "Delta_Omega": Delta_Omega,
            "fill_value": fill_value,
            "wd_blur": wd_blur,
            "stab_sacrifice": stab_sacrifice,
            "cropping_DOmega": cropping_DOmega,
            "bt_filter_flag": bt_filter_flag,
        }
        self._keys_1dr = ["iota", "iota_r", "min_tz |B|", "max_tz |B|", "Psi"]
        self._key = "trapped EP resonance"

        super().__init__(
            things=[eq],
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
        eq = self.things[0]

        rho = np.linspace(0, 1, self._num_rho + 1)[1:]
        self._constants["rho"] = rho
        self._dim_f = rho.size

        self._grid_1dr = LinearGrid(
            rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
        )
        self._constants["quad"] = get_quadrature(
            leggauss(self._hyperparameters["num_quad"]),
            (automorphism_sin, grad_automorphism_sin),
        )
        rho_res = 1.0 / self._num_rho
        eta_res = 2 * np.pi / self._num_eta
        self._params2 = {
            "rho_res": rho_res,
            "eta_res": eta_res,
        }
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, rho
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid_1dr
        )
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + [self._key], eq, self._grid_1dr
        )

        # Setup rational array
        p_max = self._hyperparameters["p_max"]
        q_max = self._hyperparameters["q_max"]
        res_range_min = self._hyperparameters["res_range_min"]
        res_range_max = self._hyperparameters["res_range_max"]

        # Preallocate: max resonances = n_max (m=0) + 2*m_max*n_max (m>0)
        n_res_max = q_max + 2 * p_max * q_max
        res_arr = np.full(n_res_max, np.nan)
        q_arr = np.zeros(n_res_max, dtype=int)
        p_arr = np.zeros(n_res_max, dtype=int)
        res_arr_set = 0

        for p in range(0, p_max + 1):
            for q in range(1, q_max + 1):
                condition = np.logical_and(
                    p / q >= res_range_min, p / q <= res_range_max
                )
                if condition:
                    res_arr[res_arr_set] = p / q
                    q_arr[res_arr_set] = q
                    p_arr[res_arr_set] = p
                    res_arr_set += 1
                    if p != 0:
                        res_arr[res_arr_set] = -p / q
                        q_arr[res_arr_set] = q
                        p_arr[res_arr_set] = -p
                        res_arr_set += 1

        res_arr = res_arr[:res_arr_set]
        q_arr = q_arr[:res_arr_set]
        p_arr = p_arr[:res_arr_set]

        self._hyperparameters["q_arr"] = q_arr
        self._hyperparameters["res_arr"] = res_arr
        self._hyperparameters["p_arr"] = p_arr
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute TrappedResonance objective.

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
        f_res_avg : ndarray
            Phase-space-averaged trapped resonance penalty as a function
            of the flux surface label.

        """
        if constants is None:
            constants = self._constants
        eq = self.things[0]

        data = compute_fun(
            eq,
            self._keys_1dr,
            params,
            constants["transforms_1dr"],
            constants["profiles"],
        )
        quad2 = {}
        if "quad2" in constants:
            quad2["quad2"] = constants["quad2"]

        # Build the eta/PSA grids and evaluate field data on them here (rather
        # than inside the "trapped EP resonance" compute function), since doing
        # so needs the full Equilibrium object and compute functions must stay
        # pure w.r.t. params/transforms/profiles/data. Must be rebuilt every
        # call since the grids depend on iota, itself a function of params.
        base_grid = self._grid_1dr
        iotas = base_grid.compress(data["iota"])
        rhos = base_grid.compress(base_grid.nodes[:, 0])
        M = self._hyperparameters["M"]
        N = self._hyperparameters["N"]
        nfp = eq.NFP
        zeta = constants.get("zeta", None)
        # Use the static hyperparameters copy, not self._num_eta directly:
        # self._num_eta isn't in _static_attrs, so under jit-tracing (e.g.
        # during grad) it becomes a traced leaf, and jnp.linspace requires a
        # concrete `num`.
        num_eta = self._hyperparameters["num_eta"]

        eta_vals = jnp.linspace(0, 2 * jnp.pi, num_eta, endpoint=False)
        ft_denom = N * nfp - iotas * M
        alpha_per_rho = eta_vals[None, :] * ft_denom[:, None] / nfp

        eta_desc_grid = _build_eta_grid(eq, rhos, alpha_per_rho, zeta, iotas, params)
        eta_grid = eta_desc_grid.source_grid

        alpha_psa = jnp.linspace(0, 2 * jnp.pi, num_eta, endpoint=False)
        psa_desc_grid = eq._get_rtz_grid(
            rhos,
            alpha_psa,
            zeta,
            coordinates="raz",
            iota=iotas,
            params=params,
        )
        psa_grid = psa_desc_grid.source_grid

        eta_data_keys = list(Bounce1D.required_names) + [
            "cvdrift0",
            "gbdrift (periodic)",
            "cvdrift (periodic)",
            "iota",
            "min_tz |B|",
            "max_tz |B|",
        ]
        psa_bounce_keys = list(Bounce1D.required_names) + [
            "min_tz |B|",
            "max_tz |B|",
            "|B|",
        ]
        all_needed_keys = list(set(eta_data_keys + psa_bounce_keys))

        # Pre-compute all transitive dependencies on the base grid (which has
        # spacing for surface integrals).  This gives us 1D intermediates like
        # iota_den, iota_num, Psi, etc. that the 3D grids cannot compute.
        internal_profiles = get_profiles(all_needed_keys, eq)
        base_data = compute_fun(
            eq,
            all_needed_keys,
            params,
            get_transforms(all_needed_keys, eq, base_grid, jitable=True),
            internal_profiles,
            data=data,
        )

        # Seed only per-surface (coordinates="r") quantities onto the 3D grids.
        # 3D quantities will be recomputed with proper angular resolution.
        _p = _parse_parameterization(eq)
        seed_1d = {}
        for key, val in base_data.items():
            entry = data_index.get(_p, {}).get(key, None)
            if entry is not None and entry.get("coordinates", "") == "r":
                seed_1d[key] = val

        eta_seed = {
            key: eta_desc_grid.copy_data_from_other(val, base_grid)
            for key, val in seed_1d.items()
        }
        data_eta = compute_fun(
            eq,
            eta_data_keys,
            params,
            get_transforms(eta_data_keys, eq, eta_desc_grid, jitable=True),
            internal_profiles,
            data=eta_seed,
        )

        psa_seed = {
            key: psa_desc_grid.copy_data_from_other(val, base_grid)
            for key, val in seed_1d.items()
        }
        data_psa = compute_fun(
            eq,
            psa_bounce_keys,
            params,
            get_transforms(psa_bounce_keys, eq, psa_desc_grid, jitable=True),
            internal_profiles,
            data=psa_seed,
        )

        data = compute_fun(
            eq,
            self._key,
            params,
            get_transforms(self._key, eq, self._grid_1dr, jitable=True),
            constants["profiles"],
            data=data,
            quad=constants["quad"],
            nfp=eq.NFP,
            zeta=zeta,
            eta_grid=eta_grid,
            psa_grid=psa_grid,
            data_eta=data_eta,
            data_psa=data_psa,
            **quad2,
            **self._hyperparameters,
            **self._params2,
        )
        if self._hyperparameters.get("pitch_invs", None) is not None:
            return data[self._key]
        return self._grid_1dr.compress(data[self._key])
