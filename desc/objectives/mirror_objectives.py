import warnings
from abc import ABC

import numpy as np
from termcolor import colored

from desc.backend import jnp
from desc.basis import (
    ChebyshevZernikeBasis,
    chebyshev_z,
    zernike_radial,
    zernike_radial_coeffs,
)

from .linear_objectives import _FixedObjective
from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class FixEndCapR(_FixedObjective):
    """Fixes Zernike R coefficients at one zeta coordinate (endcap).

    Parameters
    ----------
    zeta : EndCap zeta location: should be 0 or 2*np.pi
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    target_idx: {int, ndarray(dim_f,2)}, optional
        Corresponding L, M indexes for the targeted modes on end cap
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix sum of.
        len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "R_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-R sum modes error"

    def __init__(
        self,
        zeta,
        eq=None,
        target=None,
        target_idx=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="FixR at Zeta",
    ):
        self._zeta0 = zeta
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixSumModesR! got {modes}"
            )
        self._target_from_user = target
        self._target_idx_from_user = target_idx
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def _parse_modes_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _set_target_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _build_idx_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def build(self, eq=None, use_jit=False, verbose=1):
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
        eq = eq or self.things[0]
        if self._modes is True:  # all modes
            modes = eq.R_basis.modes
            idx = np.arange(eq.R_basis.num_modes)
            self._idx = idx
        else:  # specified modes
            raise NotImplementedError("Specifying Modes not implemented")
            # modes = np.atleast_2d(self._modes)
            # dtype = {
            #     "names": ["f{}".format(i) for i in range(3)],
            #     "formats": 3 * [modes.dtype],
            # }
            # _, idx, modes_idx = np.intersect1d(
            #     eq.R_basis.modes.astype(modes.dtype).view(dtype),
            #     modes.view(dtype),
            #     return_indices=True,
            # )
            # self._idx = idx
            # # rearrange modes and weights to match order of eq.R_basis.modes
            # # and eq.R_lmn,
            # # necessary so that the A matrix rows match up with the target b
            # modes = np.atleast_2d(eq.R_basis.modes[idx, :])
            # if idx.size < modes.shape[0]:
            #     warnings.warn(
            #         colored(
            #             "Some of the given modes are not in the basis, "
            #             + "these modes will not be fixed.",
            #             "yellow",
            #         )
            #     )

        # N modes and weight
        Nmodes = np.unique(modes[:, 2])
        self._Nmodes = Nmodes
        num_Nmodes = Nmodes.shape[0]
        # if not isinstance(eq.R_basis, ChebyshevZernikeBasis):
        #     raise NotImplementedError("End Cap BC not implemented for this basis")
        N_weights = chebyshev_z(self._zeta0, Nmodes)
        # build index for N
        idx_N = {}
        for idx_temp, N in enumerate(Nmodes):
            idx_N[N] = idx_temp
        # LM modes
        LMmodes = np.unique(modes[:, :2], axis=0)
        self._LMmodes = LMmodes
        num_LMmodes = LMmodes.shape[0]
        self._dim_f = num_LMmodes
        # build index for LM
        idx_LM = {}
        for idx_temp, (L, M) in enumerate(LMmodes):
            if L not in idx_LM:
                idx_LM[L] = {}
            idx_LM[L][M] = idx_temp

        self._A = np.zeros((self._dim_f, eq.R_basis.num_modes))
        for i, (l, m, n) in enumerate(modes):
            j = eq.R_basis.get_idx(L=l, M=m, N=n)
            k = idx_LM[l][m]
            self._A[k, j] = N_weights[idx_N[n]]

        # use current sum as target if needed
        if self._target_from_user is None:
            self.target = self._A @ eq.R_lmn[self._idx]
        else:
            self.target = np.zeros(self._dim_f)
            for i, (l, m) in enumerate(self._target_idx_from_user):
                self.target[idx_LM[l][m]] = self._target_from_user[i]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode R errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) .

        Returns
        -------
        f : ndarray
            Fixed sum mode R errors.

        """
        f = jnp.dot(self._A, params["R_lmn"])
        return f


class FixEndCapZ(_FixedObjective):
    """Fixes Zernike Z coefficients at one zeta coordinate (endcap).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    sum_weight : float, ndarray, optional
        Weights on the coefficients in the sum, should be same length as modes.
        Defaults to 1 i.e. target = 1*Z_111 + 1*Z_222...
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix sum of.
        len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Z_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-Z sum modes error"

    def __init__(
        self,
        zeta,
        eq=None,
        target=None,
        target_idx=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="FixZ at Zeta",
    ):
        self._zeta0 = zeta
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixSumModesZ! got {modes}"
            )
        self._target_from_user = target
        self._target_idx_from_user = target_idx
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def _parse_modes_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _set_target_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _build_idx_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def build(self, eq=None, use_jit=False, verbose=1):
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
        eq = eq or self.things[0]
        if self._modes is True:  # all modes
            modes = eq.Z_basis.modes
            idx = np.arange(eq.Z_basis.num_modes)
            self._idx = idx
        else:  # specified modes
            raise NotImplementedError("Specifying Modes not implemented")
            # modes = np.atleast_2d(self._modes)
            # dtype = {
            #     "names": ["f{}".format(i) for i in range(3)],
            #     "formats": 3 * [modes.dtype],
            # }
            # _, idx, modes_idx = np.intersect1d(
            #     eq.Z_basis.modes.astype(modes.dtype).view(dtype),
            #     modes.view(dtype),
            #     return_indices=True,
            # )
            # self._idx = idx
            # # rearrange modes and weights to match order of eq.Z_basis.modes
            # # and eq.Z_lmn,
            # # necessary so that the A matrix rows match up with the target b
            # modes = np.atleast_2d(eq.Z_basis.modes[idx, :])
            # if self._sum_weights is not None:
            #     self._sum_weights = np.atleast_1d(self._sum_weights)
            #     self._sum_weights = self._sum_weights[modes_idx]

            # if idx.size < modes.shape[0]:
            #     warnings.warn(
            #         colored(
            #             "Some of the given modes are not in the basis, "
            #             + "these modes will not be fixed.",
            #             "yellow",
            #         )
            #     )

        Nmodes = np.unique(modes[:, 2])
        self._Nmodes = Nmodes
        num_Nmodes = Nmodes.shape[0]
        # if not isinstance(eq.R_basis, ChebyshevZernikeBasis):
        #     raise NotImplementedError("End Cap BC not implemented for this basis")
        N_weights = chebyshev_z(self._zeta0, Nmodes)
        # build index for N
        idx_N = {}
        for idx_temp, N in enumerate(Nmodes):
            idx_N[N] = idx_temp
        # LM modes
        LMmodes = np.unique(modes[:, :2], axis=0)
        self._LMmodes = LMmodes
        num_LMmodes = LMmodes.shape[0]
        self._dim_f = num_LMmodes
        # build index for LM
        idx_LM = {}
        for idx_temp, (L, M) in enumerate(LMmodes):
            if L not in idx_LM:
                idx_LM[L] = {}
            idx_LM[L][M] = idx_temp

        self._A = np.zeros((self._dim_f, eq.Z_basis.num_modes))
        for i, (l, m, n) in enumerate(modes):
            j = eq.Z_basis.get_idx(L=l, M=m, N=n)
            k = idx_LM[l][m]
            self._A[k, j] = N_weights[idx_N[n]]

        # use current sum as target if needed
        if self._target_from_user is None:
            self.target = self._A @ eq.Z_lmn[self._idx]
        else:
            self.target = np.zeros(self._dim_f)
            for i, (l, m) in enumerate(self._target_idx_from_user):
                self.target[idx_LM[l][m]] = self._target_from_user[i]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode Z errors.

        Parameters
        ----------
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) .

        Returns
        -------
        f : ndarray
            Fixed sum mode Z errors.

        """
        f = jnp.dot(self._A, params["Z_lmn"])
        return f


class FixEndCapLambda(_FixedObjective):
    """Fixes Zernike Lambda coefficients at one zeta coordinate (endcap).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    sum_weight : float, ndarray, optional
        Weights on the coefficients in the sum, should be same length as modes.
        Defaults to 1 i.e. target = 1*Z_111 + 1*Z_222...
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix sum of.
        len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "L_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-L sum modes error"

    def __init__(
        self,
        zeta,
        eq=None,
        target=None,
        target_idx=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="FixL at Zeta",
    ):
        self._zeta0 = zeta
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixSumModesL! got {modes}"
            )
        self._target_from_user = target
        self._target_idx_from_user = target_idx
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def _parse_modes_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _set_target_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _build_idx_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def build(self, eq=None, use_jit=False, verbose=1):
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
        eq = eq or self.things[0]
        if self._modes is True:  # all modes
            modes = eq.L_basis.modes
            idx = np.arange(eq.L_basis.num_modes)
            self._idx = idx
        else:  # specified modes
            raise NotImplementedError("Specifying Modes not implemented")
            #--no-verify modes = np.atleast_2d(self._modes)
            #--no-verify dtype = {
            #--no-verify     "names": ["f{}".format(i) for i in range(3)],
            #--no-verify     "formats": 3 * [modes.dtype],
            #--no-verify }
            #--no-verify _, idx, modes_idx = np.intersect1d(
            #--no-verify     eq.Z_basis.modes.astype(modes.dtype).view(dtype),
            #--no-verify     modes.view(dtype),
            #--no-verify     return_indices=True,
            #--no-verify )
            #--no-verify self._idx = idx
            #--no-verify # rearrange modes and weights to match order of eq.Z_basis.modes
            #--no-verify # and eq.Z_lmn,
            #--no-verify # necessary so that the A matrix rows match up with the target b
            #--no-verify modes = np.atleast_2d(eq.Z_basis.modes[idx, :])
            #--no-verify if self._sum_weights is not None:
            #--no-verify     self._sum_weights = np.atleast_1d(self._sum_weights)
            #--no-verify     self._sum_weights = self._sum_weights[modes_idx]

            #--no-verify if idx.size < modes.shape[0]:
            #--no-verify     warnings.warn(
            #--no-verify         colored(
            #--no-verify             "Some of the given modes are not in the basis, "
            #--no-verify             + "these modes will not be fixed.",
            #--no-verify             "yellow",
            #--no-verify         )
            #--no-verify     )

        Nmodes = np.unique(modes[:, 2])
        self._Nmodes = Nmodes
        num_Nmodes = Nmodes.shape[0]
        # if not isinstance(eq.R_basis, ChebyshevZernikeBasis):
        #     raise NotImplementedError("End Cap BC not implemented for this basis")
        N_weights = chebyshev_z(self._zeta0, Nmodes)
        # build index for N
        idx_N = {}
        for idx_temp, N in enumerate(Nmodes):
            idx_N[N] = idx_temp
        # LM modes
        LMmodes = np.unique(modes[:, :2], axis=0)
        self._LMmodes = LMmodes
        num_LMmodes = LMmodes.shape[0]
        self._dim_f = num_LMmodes
        # build index for LM
        idx_LM = {}
        for idx_temp, (L, M) in enumerate(LMmodes):
            if L not in idx_LM:
                idx_LM[L] = {}
            idx_LM[L][M] = idx_temp

        self._A = np.zeros((self._dim_f, eq.L_basis.num_modes))
        for i, (l, m, n) in enumerate(modes):
            j = eq.L_basis.get_idx(L=l, M=m, N=n)
            k = idx_LM[l][m]
            self._A[k, j] = N_weights[idx_N[n]]

        # use current sum as target if needed
        if self._target_from_user is None:
            self.target = self._A @ eq.L_lmn[self._idx]
        else:
            self.target = np.zeros(self._dim_f)
            for i, (l, m) in enumerate(self._target_idx_from_user):
                self.target[idx_LM[l][m]] = self._target_from_user[i]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode L errors.

        Parameters
        ----------
        L_lmn : ndarray
            Spectral coefficients of L(rho,theta,zeta) .

        Returns
        -------
        f : ndarray
            Fixed sum mode L errors.

        """
        f = jnp.dot(self._A, params["L_lmn"])
        return f


class MatchEndCapR(_FixedObjective):
    """Match Zernike R coefficients at 0 and 2pi.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    target_idx: {int, ndarray(dim_f,2)}, optional
        Corresponding L, M indexes for the targeted modes on end cap
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix sum of.
        len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "R_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-R sum modes error"

    def __init__(
        self,
        eq=None,
        target=None,
        target_idx=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="MatchR at Endcaps",
    ):
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixSumModesR! got {modes}"
            )
        self._target_from_user = target
        self._target_idx_from_user = target_idx
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def _parse_modes_mirror(self):
        """Utility function for some codes in self.build."""
        pass

    def _set_target_mirror(self):
        """Utility function for some codes in self.build."""
        pass

    def _build_idx_mirror(self):
        """Utility function for some codes in self.build."""
        pass

    def build(self, eq=None, use_jit=False, verbose=1):
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
        # RG: The goal is to match R at z = 0 with z = 2pi

        # In spectral form, the constraint should enforce the following equation

        # T₀(0) ∑ₗₘ Rₗₘ₀ Zₗₘ(ρ, θ) + T₁(0) ∑ₗₘ  Rₗₘ₁ Zₗₘ(ρ, θ) + … = T₀(2π) ∑ₗₘ Rₗₘ₀ Zₗₘ(ρ, θ) + T₁(2π) ∑ₗₘ  Rₗₘ₁ Zₗₘ(ρ, θ) + …

        # where T are the Chebyshev polynomials and R_lmn are the
        # Chebyshev-Zernike spectral coefficients.

        # Using the orthogonality relation of Zernike polynomials, we can
        # separate out each l, m mode and we will get as many equations as
        # the number of lm modes
        #  ∑ₙ (Tₙ(0)−Tₙ(2π)) Rₗₘₙ
        eq = eq or self.things[0]
        modes = eq.R_basis.modes

        Nmodes = np.unique(modes[:, 2])

        # Chebyshev polynomial eval at 0 and 2π
        N_weights_0 = chebyshev_z(0.0, Nmodes)
        N_weights_2pi = chebyshev_z(2 * np.pi, Nmodes)

        idx_N = {N: i for i, N in enumerate(Nmodes)}

        # Find all the Zernike lm modes
        #--no-verify LMmodes = np.unique(modes[:, :2], axis=0)
        #--no-verify idx_LM = {tuple(lm): i for i, lm in enumerate(LMmodes)}

        LM_pairs, row_idx = np.unique(modes[:, :2], axis=0, return_inverse=True)

        n_rows = LM_pairs.shape[0]
        self._dim_f = n_rows
        n_cols = modes.shape[0]

        # Number of equations is same as the number of unique modes
        self._A = np.zeros((n_rows, n_cols))

        for col, (l, m, n) in enumerate(modes):
            w = N_weights_0[idx_N[n]] - N_weights_2pi[idx_N[n]]
            self._A[row_idx[col], col] = w

        self.target = np.zeros(n_rows)
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode R errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) .

        Returns
        -------
        f : ndarray
            Fixed sum mode R errors.

        """
        f = jnp.dot(self._A, params["R_lmn"])
        return f


class MatchEndCapZ(_FixedObjective):
    """matches Zernike Z coefficients at 0 and 2pi).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    sum_weight : float, ndarray, optional
        Weights on the coefficients in the sum, should be same length as modes.
        Defaults to 1 i.e. target = 1*Z_111 + 1*Z_222...
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix sum of.
        len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Z_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-Z sum modes error"

    def __init__(
        self,
        eq=None,
        target=None,
        target_idx=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="matchZ at Endcaps",
    ):
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixSumModesZ! got {modes}"
            )
        self._target_from_user = target
        self._target_idx_from_user = target_idx
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def _parse_modes_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _set_target_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _build_idx_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def build(self, eq=None, use_jit=False, verbose=1):
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
        eq = eq or self.things[0]
        modes = eq.Z_basis.modes

        Nmodes = np.unique(modes[:, 2])

        # Chebyshev polynomial eval at 0 and 2π
        N_weights_0 = chebyshev_z(0.0, Nmodes)
        N_weights_2pi = chebyshev_z(2 * np.pi, Nmodes)

        idx_N = {N: i for i, N in enumerate(Nmodes)}

        # Find all the Zernike lm modes
        #--no-verify LMmodes = np.unique(modes[:, :2], axis=0)
        #--no-verify idx_LM = {tuple(lm): i for i, lm in enumerate(LMmodes)}

        LM_pairs, row_idx = np.unique(modes[:, :2], axis=0, return_inverse=True)

        n_rows = LM_pairs.shape[0]  # same as dim_f
        self._dim_f = n_rows
        n_cols = modes.shape[0]

        self._A = np.zeros((n_rows, n_cols))

        for col, (l, m, n) in enumerate(modes):
            w = N_weights_0[idx_N[n]] - N_weights_2pi[idx_N[n]]
            if w != 0.0:
                self._A[row_idx[col], col] = w

        self.target = np.zeros(n_rows)
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode Z errors.

        Parameters
        ----------
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) .

        Returns
        -------
        f : ndarray
            Fixed sum mode Z errors.

        """
        f = jnp.dot(self._A, params["Z_lmn"])
        return f


class MatchEndCapLambda(_FixedObjective):
    """Matches Zernike Lambda coefficients at 0 and 2pi.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    sum_weight : float, ndarray, optional
        Weights on the coefficients in the sum, should be same length as modes.
        Defaults to 1 i.e. target = 1*Z_111 + 1*Z_222...
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix sum of.
        len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "L_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-L sum modes error:"

    def __init__(
        self,
        eq=None,
        target=None,
        target_idx=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="MatchL at Endcaps",
    ):
        self._zeta0 = 0.0
        self._zeta1 = 2 * np.pi
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixSumModesL! got {modes}"
            )
        self._target_from_user = target
        self._target_idx_from_user = target_idx
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def _parse_modes_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _set_target_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def _build_idx_mirror(self):
        """utility function for some codes in self.build"""
        pass

    def build(self, eq=None, use_jit=False, verbose=1):
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
        eq = eq or self.things[0]
        modes = eq.L_basis.modes

        Nmodes = np.unique(modes[:, 2])

        # Chebyshev polynomial eval at 0 and 2π
        N_weights_0 = chebyshev_z(0.0, Nmodes)
        N_weights_2pi = chebyshev_z(2 * np.pi, Nmodes)

        idx_N = {N: i for i, N in enumerate(Nmodes)}

        # Find all the Zernike lm modes
        #--no-verify LMmodes = np.unique(modes[:, :2], axis=0)
        #--no-verify idx_LM = {tuple(lm): i for i, lm in enumerate(LMmodes)}

        LM_pairs, row_idx = np.unique(modes[:, :2], axis=0, return_inverse=True)

        n_rows = LM_pairs.shape[0]  # same as dim_f
        self._dim_f = n_rows
        n_cols = modes.shape[0]

        self._A = np.zeros((n_rows, n_cols))

        for col, (l, m, n) in enumerate(modes):
            w = N_weights_0[idx_N[n]] - N_weights_2pi[idx_N[n]]
            if w != 0.0:
                self._A[row_idx[col], col] = w

        self.target = np.zeros(self._dim_f)
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode L errors.

        Parameters
        ----------
        L_lmn : ndarray
            Spectral coefficients of L(rho,theta,zeta) .

        Returns
        -------
        f : ndarray
            Fixed sum mode L errors.

        """
        f = jnp.dot(self._A, params["L_lmn"])
        return f
