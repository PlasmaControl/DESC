"""Objectives for trapped energetic particle resonance."""

import numpy as np
from interpax_fft import cheb_pts, fourier_pts
from orthax.legendre import leggauss
from scipy.constants import elementary_charge

from desc.backend import jax, jnp
from desc.compute import get_profiles, get_transforms
from desc.compute._trapped_resonance import (
    _FFT_BOUNCE_KEYS,
    _build_eta_grid,
    _build_eta_source_grid,
)
from desc.compute.data_index import data_index
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import _parse_parameterization, get_data_deps
from desc.grid import Grid, LinearGrid
from desc.integrals._bounce_utils import Y_B_rule, get_vander_spline
from desc.integrals.bounce_integral import Bounce1D, Bounce2D
from desc.utils import Timer, errorif

from ..integrals.quad_utils import (
    automorphism_sin,
    get_quadrature,
    grad_automorphism_sin,
)
from .objective_funs import _Objective
from .utils import _parse_callable_target_bounds


def _shift_grid_rho(grid, drho):
    """``grid`` with rho displaced by ``drho``, traceable in ``drho``.

    A ``LinearGrid`` is built from numpy and so cannot carry a tangent in rho.
    This rebuilds it as a jitable ``Grid`` over the same nodes, which lets the
    quantities computed on it be differentiated with respect to rho.

    Everything copied over is either untouched by a radial shift or describes
    the (theta, zeta) structure, which a radial shift leaves alone:
    ``spacing`` and ``weights`` because surface integrals are over theta and
    zeta; ``fft_poloidal`` and ``fft_toroidal`` because they record uniformity
    in those angles, and ``_partial_sum`` refuses a grid without the former.

    ``M`` and ``N`` must be copied rather than left to the ``Grid`` defaults,
    which report 0 instead of the source ``LinearGrid``'s resolution. Dropping
    them zeroes the Fourier resolution silently rather than raising.

    Parameters
    ----------
    grid : Grid
    drho : float or jnp.ndarray
        Radial displacement.

    Returns
    -------
    Grid

    """
    rho, theta, zeta = grid.nodes.T
    out = Grid(
        nodes=jnp.column_stack([rho + drho, theta, zeta]),
        spacing=grid.spacing,
        weights=grid.weights,
        coordinates=grid.coordinates,
        period=grid.period,
        NFP=grid.NFP,
        sort=False,
        is_meshgrid=grid.is_meshgrid,
        jitable=True,
        _unique_rho_idx=grid.unique_rho_idx,
        _unique_poloidal_idx=grid.unique_poloidal_idx,
        _unique_zeta_idx=grid.unique_zeta_idx,
        _inverse_rho_idx=grid.inverse_rho_idx,
        _inverse_poloidal_idx=grid.inverse_poloidal_idx,
        _inverse_zeta_idx=grid.inverse_zeta_idx,
    )
    out._fft_poloidal = grid.fft_poloidal
    out._fft_toroidal = grid.fft_toroidal
    out._M = grid.M
    out._N = grid.N
    return out


def _seed_tangent_key(name):
    """Name of the compute quantity holding d(``name``)/dρ."""
    return name + "r" if name.endswith("_r") or name.endswith("_rr") else name + "_r"


def _seeded_keys(keys, parameterization):
    """Per-surface keys seeded onto a field line grid, and those it reads.

    ``_compute`` skips any quantity already present in ``data``, so a seeded
    quantity shadows its own dependencies: only the seeded keys that some
    recomputed quantity depends on directly can influence the result. Those are
    the ones whose radial derivative has to be supplied for an analytic Ω'(s).

    Parameters
    ----------
    keys : list[str]
        Quantities requested on the field line grid.
    parameterization : str
        Parameterization of the thing being computed, e.g.
        ``"desc.equilibrium.equilibrium.Equilibrium"``.

    Returns
    -------
    seeded, consumed : tuple[set[str]]

    """
    index = data_index[parameterization]
    closure = set(get_data_deps(keys, obj=parameterization)) | set(keys)
    seeded = {k for k in closure if index.get(k, {}).get("coordinates", "") == "r"}

    consumed, visited, stack = set(), set(), list(keys)
    while stack:
        key = stack.pop()
        if key in visited:
            continue
        visited.add(key)
        if key in seeded and key not in keys:
            continue  # seeded, so it is not recomputed and its deps are unused
        for dep in index.get(key, {}).get("dependencies", {}).get("data", []):
            if dep in seeded:
                consumed.add(dep)
            stack.append(dep)
    return seeded, consumed


def _tangent_keys(parameterization, keys):
    """Map each seeded per-surface key to the key holding its radial derivative.

    Raises if a key the field line compute actually reads has no radial
    derivative registered in ``data_index``, so that a missing tangent can
    never quietly turn into a wrong Ω'(s).
    """
    seeded, consumed = _seeded_keys(keys, parameterization)
    index = data_index[parameterization]
    tangent = {
        key: t_key for key in seeded if (t_key := _seed_tangent_key(key)) in index
    }
    # "rho" differentiates to 1; "p_r" is covered by _p_rr.
    missing = sorted(consumed - set(tangent) - {"rho", "p_r"})
    if missing:
        raise NotImplementedError(
            f"TrappedResonance needs the radial derivative of {missing} to "
            "differentiate the bounce integrals, and DESC has no compute "
            "quantity for it."
        )
    return tangent


def _p_rr(params, profiles, grid, data):
    """d²p/dρ², which has no compute quantity of its own to read it from.

    Mirrors how ``p_r`` itself is computed, one derivative higher.
    """
    if profiles.get("pressure") is not None:
        return profiles["pressure"].compute(grid, params["p_l"], dr=2)
    return elementary_charge * (
        data["ne_rr"] * data["Te"]
        + 2 * data["ne_r"] * data["Te_r"]
        + data["ne"] * data["Te_rr"]
        + data["ni_rr"] * data["Ti"]
        + 2 * data["ni_r"] * data["Ti_r"]
        + data["ni"] * data["Ti_rr"]
    )


def _seed_tangents(params, profiles, grid, data, seed_1d, tangent_keys):
    """d/dρ of each per-surface quantity seeded onto a field line grid."""
    seed_dot = {}
    for key, val in seed_1d.items():
        if key == "rho":
            seed_dot[key] = jnp.ones_like(val)
        elif key == "p_r":
            seed_dot[key] = _p_rr(params, profiles, grid, data)
        elif key in tangent_keys:
            seed_dot[key] = data[tangent_keys[key]]
        else:
            # Nothing on the field line grid reads this, as _tangent_keys
            # checked. min_tz |B| and max_tz |B| land here on purpose: the
            # pitch grid must stay put so the derivative is taken at fixed λ.
            seed_dot[key] = jnp.zeros_like(val)
    return seed_dot


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
    rho : int or ndarray, optional
        Flux surfaces on which to evaluate the objective. If an int, the
        surfaces are constructed as ``np.linspace(0, 1, rho + 1)[1:]``, giving
        ``rho`` uniformly spaced surfaces from ``1/rho`` to ``1`` with spacing
        ``1/rho``. If an array, it must be increasing, linearly spaced, and
        must not include the magnetic axis (rho=0); e.g. pass an array ending
        before rho=1 for equilibria whose pressure profile is not well-defined
        at the edge. Default is 10.
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

    Notes
    -----
    Ω'(s), the radial derivative of the normalized precession frequency that
    sets the island width, is obtained by differentiating the bounce integrals
    with respect to rho at fixed λ and η. That is exact for any radial grid and
    defined on every surface, and costs roughly double the work of evaluating
    the field data and the bounce integrals.

    It replaced a finite difference of Ω across neighbouring surfaces, which
    was removed: on a stellarator dΩ/dρ swings over orders of magnitude between
    adjacent surfaces, so that estimate does not converge at any practical
    number of surfaces, and it is biased low because a secant cannot resolve
    |Ω'| below the scale set by the radial grid spacing. The two agree exactly
    when ``stab_sacrifice=True``, where Ω' cancels out of the objective.
    """

    _scalar = False
    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Trapped EP Resonance Penalty: "

    _static_attrs = _Objective._static_attrs + [
        "_hyperparameters",
        "_keys_1dr",
        "_key",
        # Selects which branch ``compute`` takes, so it must stay concrete
        # under jit rather than becoming a traced leaf.
        "_use_bounce1d",
        "_X",
        "_Y",
    ]

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
        rho=10,
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
        use_bounce1d=False,
        X=32,
        Y=32,
        Y_B=None,
        spline=True,
        nufft_eps=1e-10,
    ):
        if target is None and bounds is None:
            target = 1e-8
        self._use_bounce1d = bool(use_bounce1d)
        self._rho = int(rho) if np.isscalar(rho) else np.atleast_1d(np.asarray(rho))
        self._num_eta = int(num_eta)
        if self._num_eta < 2:
            raise ValueError(f"num_eta must be >= 2, got {self._num_eta}.")

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
            "use_bounce1d": self._use_bounce1d,
        }
        if not self._use_bounce1d:
            self._hyperparameters["Y_B"] = Y_B
            self._hyperparameters["spline"] = spline
            self._hyperparameters["nufft_eps"] = nufft_eps
            self._X = int(X)
            self._Y = int(Y)
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

        rho = (
            np.linspace(0, 1, self._rho + 1)[1:]
            if isinstance(self._rho, int)
            else self._rho
        )
        errorif(
            rho.size < 2,
            ValueError,
            msg=f"rho must have >= 2 surfaces, got {rho.size}.",
        )
        errorif(
            rho[1] <= rho[0] or not np.allclose(np.diff(rho), rho[1] - rho[0]),
            ValueError,
            msg="rho array must be increasing and linearly spaced!",
        )
        errorif(
            np.any(np.isclose(rho, 0.0)),
            ValueError,
            msg="rho array must not include the axis!",
        )
        self._constants["rho"] = rho
        self._dim_f = rho.size

        self._grid_1dr = LinearGrid(
            rho=rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym if self._use_bounce1d else False,
        )
        self._constants["quad"] = get_quadrature(
            leggauss(self._hyperparameters["num_quad"]),
            (automorphism_sin, grad_automorphism_sin),
        )
        if not self._use_bounce1d:
            assert self._grid_1dr.can_fft2
            # Nodes at which the poloidal angle map is interpolated, and the
            # transform used to solve for it. Mirrors ``Bounce2D._build``.
            self._constants["x"] = fourier_pts(self._X)
            self._constants["y"] = cheb_pts(self._Y, (0, 2 * np.pi / eq.NFP))[::-1]
            self._constants["lambda"] = get_transforms(
                "lambda",
                eq,
                grid=LinearGrid(
                    rho=rho,
                    M=eq.L_basis.M,
                    zeta=self._constants["y"],
                    NFP=eq.NFP,
                ),
            )["L"]
            spline = self._hyperparameters["spline"]
            Y_B = self._hyperparameters["Y_B"]
            if Y_B is None:
                Y_B = Y_B_rule(self._grid_1dr, spline)
                self._hyperparameters["Y_B"] = Y_B
            self._constants["_vander"] = (
                get_vander_spline(self._grid_1dr, self._Y, Y_B, eq.NFP)
                if spline
                else {}
            )
        rho_res = rho[1] - rho[0]
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

        base_grid = self._grid_1dr
        iotas = base_grid.compress(data["iota"])
        iotas_r = base_grid.compress(data["iota_r"])
        rhos = base_grid.compress(base_grid.nodes[:, 0])
        M = self._hyperparameters["M"]
        N = self._hyperparameters["N"]
        nfp = eq.NFP
        zeta = constants.get("zeta")
        num_eta = self._hyperparameters["num_eta"]

        eta_vals = jnp.linspace(0, 2 * jnp.pi, num_eta, endpoint=False)
        ft_denom = N * nfp - iotas * M
        alpha_per_rho = eta_vals[None, :] * ft_denom[:, None] / nfp

        if not self._use_bounce1d:
            fft_keys = list(Bounce2D.required_names) + [
                "cvdrift0",
                "gbdrift (periodic)",
                "cvdrift (periodic)",
                "min_tz |B|",
                "max_tz |B|",
            ]
            fft_profiles = get_profiles(fft_keys, eq)
            lambda_grid = constants["lambda"].grid

            def _fft_stage(t):
                """Field data and angle map on surfaces displaced by ``t``.

                Written as a function of a radial displacement so that its
                forward mode derivative at ``t = 0`` is d/dρ at fixed theta
                and zeta, which is what Omega'(s) needs. Unlike the Bounce1D
                path there is nothing to seed: the grid carries the whole
                radial dependence, so every quantity moves with it.
                """
                grid_t = _shift_grid_rho(base_grid, t)
                data_t = compute_fun(
                    eq,
                    self._keys_1dr,
                    params,
                    get_transforms(self._keys_1dr, eq, grid_t, jitable=True),
                    constants["profiles"],
                )
                data_fft_t = compute_fun(
                    eq,
                    fft_keys,
                    params,
                    get_transforms(fft_keys, eq, grid_t, jitable=True),
                    fft_profiles,
                    data=data_t,
                )
                # Poloidal angle map, rebuilt every call since it moves with
                # rho through both iota and lambda.
                angle_t = eq._map_poloidal_coordinates(
                    grid_t.compress(data_t["iota"]),
                    constants["x"],
                    constants["y"],
                    params["L_lmn"],
                    get_transforms(
                        "lambda",
                        eq,
                        grid=_shift_grid_rho(lambda_grid, t),
                        jitable=True,
                    )["L"],
                    outbasis="delta",
                    tol=1e-8,
                )[..., ::-1]
                return {k: data_fft_t[k] for k in _FFT_BOUNCE_KEYS}, angle_t

            (data_fft, angle), (data_fft_r, angle_r) = jax.jvp(
                _fft_stage, (0.0,), (1.0,)
            )

            data = compute_fun(
                eq,
                self._key,
                params,
                get_transforms(self._key, eq, base_grid, jitable=True),
                constants["profiles"],
                data=data,
                quad=constants["quad"],
                nfp=nfp,
                zeta=zeta,
                _angle=angle,
                _angle_r=angle_r,
                _fft_grid=base_grid,
                _data_fft=data_fft,
                _data_fft_r=data_fft_r,
                _vander=constants["_vander"],
                **quad2,
                **self._hyperparameters,
                **self._params2,
            )
            if self._hyperparameters.get("pitch_invs") is not None:
                return data[self._key]
            return base_grid.compress(data[self._key])

        # The field line following grid the bounce integrals run along. It
        # needs no coordinate map, so it is built here rather than pulled off
        # the (rho, theta, zeta) grid, which _eta_data rebuilds at shifted rho.
        eta_grid = _build_eta_source_grid(rhos, alpha_per_rho, zeta)

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

        # An analytic Omega'(s) needs the radial derivative of every
        # per-surface quantity seeded onto the eta grid; without it the seed
        # would look constant in rho and the derivative would come out wrong.
        _p = _parse_parameterization(eq)
        tangent_keys = _tangent_keys(_p, eta_data_keys)
        extra = list(tangent_keys.values())
        if eq.pressure is None:
            # _p_rr then builds d²p/dρ² out of the kinetic profiles.
            extra += ["ne_rr", "ni_rr", "Te_rr", "Ti_rr"]
        all_needed_keys = list(set(all_needed_keys + extra))

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
        seed_1d = {}
        for key, val in base_data.items():
            entry = data_index.get(_p, {}).get(key)
            if entry is not None and entry.get("coordinates", "") == "r":
                seed_1d[key] = val

        seed_dot = _seed_tangents(
            params, internal_profiles, base_grid, base_data, seed_1d, tangent_keys
        )

        def _eta_data(t):
            """Field line data on the eta grid, with rho displaced by ``t``.

            Written as a function of a radial displacement so that its forward
            mode derivative at ``t = 0`` is d/dρ at fixed eta and zeta. It is
            eta, not alpha, that is held fixed: alpha follows rho through iota,
            exactly as it does when Omega is instead finite differenced across
            neighbouring surfaces.
            """
            iotas_t = iotas + t * iotas_r
            alpha_t = eta_vals[None, :] * (N * nfp - iotas_t[:, None] * M) / nfp
            grid_t = _build_eta_grid(eq, rhos + t, alpha_t, zeta, iotas_t, params)
            seed_t = {key: val + t * seed_dot[key] for key, val in seed_1d.items()}
            eta_seed = {
                key: grid_t.copy_data_from_other(val, base_grid)
                for key, val in seed_t.items()
            }
            return compute_fun(
                eq,
                eta_data_keys,
                params,
                get_transforms(eta_data_keys, eq, grid_t, jitable=True),
                internal_profiles,
                data=eta_seed,
            )

        data_eta, data_eta_r = jax.jvp(_eta_data, (0.0,), (1.0,))

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
            _eta_grid=eta_grid,
            _psa_grid=psa_grid,
            _data_eta=data_eta,
            _data_eta_r=data_eta_r,
            _data_psa=data_psa,
            **quad2,
            **self._hyperparameters,
            **self._params2,
        )
        if self._hyperparameters.get("pitch_invs") is not None:
            return data[self._key]
        return self._grid_1dr.compress(data[self._key])
