"""Objectives for trapped energetic particle resonance."""

import numpy as np
from orthax.legendre import leggauss

from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer

from .objective_funs import _Objective
from .utils import _parse_callable_target_bounds

from ..integrals.quad_utils import automorphism_sin, get_quadrature, grad_automorphism_sin


# New resonance objective from John Anthony Labbate
class TrappedResonance(_Objective):
    """Trapped energetic particle resonance penalty.

    Creates bump function about a specified number of lowest order resonances
    (m/n) for trapped energetic particle motion. Vicinity to these rational
    values is penalized.

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
        If ``None``, defaults to 2× the mean |Ω[i+1]-Ω[i]| spacing.
        Ignored when ``weight_method="linear"``.

    """

    _scalar = False
    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Trapped EP Resonance Penalty: "

    _static_attrs = _Objective._static_attrs + [
        "_hyperparameters", "_keys_1dr", "_key",
    ]

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
        num_rho=10,
        num_eta=10,
        KE_frac=np.array([1]),
        *,
        num_transit=5,
        knots_per_transit=100,
        num_quad=32,
        num_pitch=16,
        batch=True,
        name="TrappedResonance",
        jac_chunk_size=None,
        pitch_invs=None,
        N=0,
        M=1,
        p_max=10,
        q_max=10,
        res_range_min=-4,
        res_range_max=4,
        verbose=False,
        pitch_batch_size=1,
        surf_batch_size=1,
        weight_method="linear",
        Delta_Omega=None,
        fill_value=11,
        wd_blur=1.25,
        stab_sacrifice=False,
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
            "bt_filter_flag": bt_filter_flag,
        }
        self._keys_1dr = ["iota", "iota_r", "min_tz |B|", "max_tz |B|", "Psi"]
        self._key = "f_tr2"

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
            leggauss(self._hyperparameters.pop("num_quad")),
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

        for p in range(0,p_max+1):
            for q in range(1,q_max+1):
                condition = np.logical_and(p/q >= res_range_min, p/q <= res_range_max)
                if condition:
                    res_arr[res_arr_set] = p/q
                    q_arr[res_arr_set] = q
                    p_arr[res_arr_set] = p
                    res_arr_set+=1
                    if p != 0:
                        res_arr[res_arr_set] = -p/q
                        q_arr[res_arr_set] = q
                        p_arr[res_arr_set] = -p
                        res_arr_set += 1

        res_arr = res_arr[:res_arr_set]
        q_arr = q_arr[:res_arr_set]
        p_arr = p_arr[:res_arr_set]

        self._hyperparameters['q_arr'] = q_arr
        self._hyperparameters['res_arr'] = res_arr
        self._hyperparameters['p_arr'] = p_arr
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

        data = compute_fun(
            eq,
            self._key,
            params,
            get_transforms(self._key, eq, self._grid_1dr, jitable=True),
            constants["profiles"],
            data=data,
            quad=constants["quad"],
            nfp=eq.NFP,
            eq=eq,
            zeta=constants.get("zeta", None),
            **quad2,
            **self._hyperparameters,
            **self._params2,
        )
        if self._hyperparameters.get("pitch_invs", None) is not None:
            return data[self._key]
        return self._grid_1dr.compress(data[self._key])
