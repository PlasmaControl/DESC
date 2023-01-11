"""Classes for linear optimization constraints.

Linear objective functions must be of the form `A*x-b`, where:
    - `A` is a constant matrix that can be pre-computed
    - `x` is a vector of one or more arguments included in `compute.arg_order`
    - `b` is the desired vector set by `objective.target`
"""

import warnings
from abc import ABC

import numpy as np
from termcolor import colored

from desc.backend import jnp
from desc.basis import zernike_radial, zernike_radial_coeffs

from .normalization import compute_scaling_factors
from .objective_funs import _Objective

# TODO: need dim_x attribute


class FixBoundaryR(_Objective):
    """Boundary condition on the R boundary parameters.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Boundary surface coefficients to fix. If None, uses surface coefficients.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    fixed_boundary : bool, optional
        True to enforce the boundary condition on flux surfaces,
        or False to fix the boundary surface coefficients (default).
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the profile modes.
    surface_label : float
        Surface to enforce boundary conditions on. Defaults to Equilibrium.surface.rho
    name : str
        Name of the objective function.


    Notes
    -----
    If specifying particular modes to fix, the rows of the resulting constraint `A`
    matrix and `target` vector will be re-sorted according to the ordering of
    `basis.modes` which may be different from the order that was passed in.
    """

    _scalar = False
    _linear = True
    _fixed = False  # TODO: can we dynamically detect this instead?
    _units = "(m)"
    _print_value_fmt = "R boundary error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        fixed_boundary=False,
        modes=True,
        surface_label=None,
        name="lcfs R",
    ):

        self._fixed_boundary = fixed_boundary
        self._modes = modes
        self._surface_label = surface_label
        self._args = ["R_lmn"] if self._fixed_boundary else ["Rb_lmn"]
        super().__init__(
            eq=eq,
            target=target,
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
        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            idx = np.array([], dtype=int)
        elif self._modes is True:  # all modes
            modes = eq.surface.R_basis.modes
            idx = np.arange(eq.surface.R_basis.num_modes)
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.surface.R_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            # rearrange modes to match order of eq.surface.R_basis.modes
            # and eq.surface.R_lmn,
            # necessary so that the A matrix rows match up with the target b
            modes = np.atleast_2d(eq.surface.R_basis.modes[idx, :])

            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the surface, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = idx.size
        if self.target is not None:  # rearrange given target to match modes order
            if self._modes is True or self._modes is False:
                raise RuntimeError(
                    "Attempting to provide target for R boundary modes without "
                    + "providing modes array!"
                    + "You must pass in the modes corresponding to the"
                    + "provided target"
                )
            self.target = self.target[modes_idx]

        if self._fixed_boundary:  # R_lmn -> Rb_lmn boundary condition
            self._A = np.zeros((self._dim_f, eq.R_basis.num_modes))
            for i, (l, m, n) in enumerate(eq.R_basis.modes):
                if eq.bdry_mode == "lcfs":
                    j = np.argwhere((modes[:, 1:] == [m, n]).all(axis=1))
                    surf = (
                        eq.surface.rho
                        if self._surface_label is None
                        else self._surface_label
                    )
                    self._A[j, i] = zernike_radial(surf, l, m)

        else:  # Rb_lmn -> Rb optimization space
            self._A = np.eye(eq.surface.R_basis.num_modes)[idx, :]

        # use surface parameters as target if needed
        if self.target is None:
            self.target = eq.surface.R_lmn[idx]

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute deviation from desired boundary."""
        if len(args):
            x = kwargs.get(self.args[0], args[0])
        else:
            x = kwargs.get(self.args[0])
        Rb = jnp.dot(self._A, x)
        return self._shift_scale(Rb)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Rb_lmn"


class FixBoundaryZ(_Objective):
    """Boundary condition on the Z boundary parameters.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Boundary surface coefficients to fix. If None, uses surface coefficients.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    fixed_boundary : bool, optional
        True to enforce the boundary condition on flux surfaces,
        or False to fix the boundary surface coefficients (default).
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the profile modes.
    surface_label : float
        Surface to enforce boundary conditions on. Defaults to Equilibrium.surface.rho
    name : str
        Name of the objective function.


    Notes
    -----
    If specifying particular modes to fix, the rows of the resulting constraint `A`
    matrix and `target` vector will be re-sorted according to the ordering of
    `basis.modes` which may be different from the order that was passed in.
    """

    _scalar = False
    _linear = True
    _fixed = False
    _units = "(m)"
    _print_value_fmt = "Z boundary error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        fixed_boundary=False,
        modes=True,
        surface_label=None,
        name="lcfs Z",
    ):

        self._fixed_boundary = fixed_boundary
        self._modes = modes
        self._surface_label = surface_label
        self._args = ["Z_lmn"] if self._fixed_boundary else ["Zb_lmn"]
        super().__init__(
            eq=eq,
            target=target,
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
        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            idx = np.array([], dtype=int)
        elif self._modes is True:  # all modes
            modes = eq.surface.Z_basis.modes
            idx = np.arange(eq.surface.Z_basis.num_modes)
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.surface.Z_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            # rearrange modes to match order of eq.surface.Z_basis.modes
            # and eq.surface.Z_lmn,
            # necessary so that the A matrix rows match up with the target b
            modes = np.atleast_2d(eq.surface.Z_basis.modes[idx, :])

            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the surface, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = idx.size
        if self.target is not None:  # rearrange given target to match modes order
            if self._modes is True or self._modes is False:
                raise RuntimeError(
                    "Attempting to provide target for Z boundary modes without "
                    + "providing modes array!"
                    + "You must pass in the modes corresponding to the"
                    + "provided target"
                )
            self.target = self.target[modes_idx]

        if self._fixed_boundary:  # Z_lmn -> Zb_lmn boundary condition
            self._A = np.zeros((self._dim_f, eq.Z_basis.num_modes))
            for i, (l, m, n) in enumerate(eq.Z_basis.modes):
                if eq.bdry_mode == "lcfs":
                    j = np.argwhere((modes[:, 1:] == [m, n]).all(axis=1))
                    surf = (
                        eq.surface.rho
                        if self._surface_label is None
                        else self._surface_label
                    )
                    self._A[j, i] = zernike_radial(surf, l, m)
        else:  # Zb_lmn -> Zb optimization space
            self._A = np.eye(eq.surface.Z_basis.num_modes)[idx, :]

        # use surface parameters as target if needed
        if self.target is None:
            self.target = eq.surface.Z_lmn[idx]

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute deviation from desired boundary."""
        if len(args):
            x = kwargs.get(self.args[0], args[0])
        else:
            x = kwargs.get(self.args[0])
        Zb = jnp.dot(self._A, x)
        return self._shift_scale(Zb)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Zb_lmn"


class FixLambdaGauge(_Objective):
    """Fixes gauge freedom for lambda: lambda(rho=0)=0 and lambda(theta=0,zeta=0)=0.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Value to fix lambda to at rho=0 and (theta=0,zeta=0)
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False
    _units = "(radians)"
    _print_value_fmt = "lambda gauge error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        name="lambda gauge",
    ):

        super().__init__(
            eq=eq,
            target=target,
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
        L_basis = eq.L_basis

        if L_basis.sym:
            # l(0,t,z) = 0
            # any zernike mode that has m != 0 (i.e., has any theta dependence)
            # contains radial dependence at least as rho^m
            # therefore if m!=0, no constraint is needed to make the mode go to
            # zero at rho=0

            # for the other modes with m=0, at rho =0 the basis reduces
            # to just a linear combination of sin(n*zeta), cos(n*zeta), and 1
            # since these are all linearly independent, to make lambda -> 0 at rho=0,
            # each coefficient on these terms must individually go to zero
            # i.e. if at rho=0 the lambda basis is given by
            # Lambda(rho=0) = (L_{00-1} - L_{20-1})sin(zeta) + (L_{001}
            #                   - L_{201})cos(zeta) + L_{000} - L_{200}
            # Lambda(rho=0) = 0 constraint being enforced means that each
            # coefficient goes to zero:
            # L_{00-1} - L_{20-1} = 0
            # L_{001} - L_{201} = 0
            # L_{000} - L_{200} = 0
            self._A = np.zeros((L_basis.N, L_basis.num_modes))
            ns = np.arange(-L_basis.N, 1)
            for i, (l, m, n) in enumerate(L_basis.modes):
                if m != 0:
                    continue
                if (
                    l // 2
                ) % 2 == 0:  # this basis mode radial polynomial is +1 at rho=0
                    j = np.argwhere(n == ns)
                    self._A[j, i] = 1
                else:  # this basis mode radial polynomial is -1 at rho=0
                    j = np.argwhere(n == ns)
                    self._A[j, i] = -1
        else:
            # l(0,t,z) = 0

            ns = np.arange(-L_basis.N, L_basis.N + 1)
            self._A = np.zeros((len(ns), L_basis.num_modes))
            for i, (l, m, n) in enumerate(L_basis.modes):
                if m != 0:
                    continue
                if (l // 2) % 2 == 0:
                    j = np.argwhere(n == ns)
                    self._A[j, i] = 1
                else:
                    j = np.argwhere(n == ns)
                    self._A[j, i] = -1
            # l(rho,0,0) = 0
            # at theta=zeta=0, basis for lamba reduces to just a polynomial in rho
            # what this constraint does is make all the coefficients of each power
            # of rho equal to zero
            # i.e. if lambda = (L_200 + 2*L_310) rho**2 + (L_100 + 2*L_210)*rho
            # this constraint will make
            # L_200 + 2*L_310 = 0
            # L_100 + 2*L_210 = 0
            L_modes = L_basis.modes
            mnpos = np.where((L_modes[:, 1:] >= [0, 0]).all(axis=1))[0]
            l_lmn = L_modes[mnpos, :]
            if len(l_lmn) > 0:
                c = zernike_radial_coeffs(l_lmn[:, 0], l_lmn[:, 1])
            else:
                c = np.zeros((0, 0))

            A = np.zeros((c.shape[1], L_basis.num_modes))
            A[:, mnpos] = c.T
            self._A = np.vstack((self._A, A))

        self._dim_f = self._A.shape[0]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, L_lmn, **kwargs):
        """Compute lambda gauge symmetry errors.

        Parameters
        ----------
        L_lmn : ndarray
            Spectral coefficients of L(rho,theta,zeta) -- poloidal stream function.

        Returns
        -------
        f : ndarray
            Lambda gauge symmetry errors.

        """
        f = jnp.dot(self._A, L_lmn)
        return self._shift_scale(f)


class _FixProfile(_Objective, ABC):
    """Fixes profile coefficients (or values, for SplineProfile).

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : tuple, float, ndarray, optional
        Target value(s) of the objective.
        len(target) = len(weight) = len(modes). If None, uses Profile.params.
        e.g. for PowerSeriesProfile these are profile coefficients, and for
        SplineProfile they are values at knots.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(target) = len(weight) = len(modes)
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or Bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = True

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="",
    ):

        self._profile = profile
        self._indices = indices
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )
        self._print_value_fmt = None

    def build(self, eq, profile=None, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.
        profile : Profile, optional
            profile to fix
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._profile is None or self._profile.params.size != eq.L + 1:
            self._profile = profile

        # find indices to fix
        if self._indices is False or self._indices is None:  # no indices to fix
            self._idx = np.array([], dtype=int)
        elif self._indices is True:  # all indices of Profile.params
            self._idx = np.arange(np.size(self._profile.params))
        else:  # specified indices
            self._idx = np.atleast_1d(self._indices)

        self._dim_f = self._idx.size
        # use profile parameters as target if needed
        if self.target is None:
            self.target = self._profile.params[self._idx]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)


class FixPressure(_FixProfile):
    """Fixes pressure coefficients.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : tuple, float, ndarray, optional
        Target value(s) of the objective.
        len(target) = len(weight) = len(modes). If None, uses profile coefficients.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(target) = len(weight) = len(modes)
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(Pa)"
    _print_value_fmt = "Fixed-pressure profile error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="fixed-pressure",
    ):

        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        profile = eq.pressure
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["p"]
        super().build(eq, profile, use_jit, verbose)

    def compute(self, p_l, **kwargs):
        """Compute fixed pressure profile errors.

        Parameters
        ----------
        p_l : ndarray
            parameters of the pressure profile.

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        fixed_params = p_l[self._idx]
        return self._shift_scale(fixed_params)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "p_l"


class FixIota(_FixProfile):
    """Fixes rotational transform coefficients.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : tuple, float, ndarray, optional
        Target value(s) of the objective.
        len(target) = len(weight) = len(modes). If None, uses profile coefficients.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(target) = len(weight) = len(modes)
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices.
        corresponding to knots for a SplineProfile).
        Must len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(dimensionless)"
    _print_value_fmt = "Fixed-iota profile error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        profile=None,
        indices=True,
        name="fixed-iota",
    ):

        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if eq.iota is None:
            raise RuntimeError(
                "Attempt to fix rotational transform on an equilibrium with no "
                + "rotational transform profile assigned"
            )
        profile = eq.iota
        super().build(eq, profile, use_jit, verbose)

    def compute(self, i_l, **kwargs):
        """Compute fixed iota errors.

        Parameters
        ----------
        i_l : ndarray
            parameters of the iota profile.

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        fixed_params = i_l[self._idx]
        return self._shift_scale(fixed_params)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "i_l"


class FixCurrent(_FixProfile):
    """Fixes toroidal current profile coefficients.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : tuple, float, ndarray, optional
        Target value(s) of the objective.
        len(target) = len(weight) = len(modes). If None, uses profile coefficients.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(target) = len(weight) = len(modes)
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(A)"
    _print_value_fmt = "Fixed-current profile error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="fixed-current",
    ):

        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if eq.current is None:
            raise RuntimeError(
                "Attempting to fix toroidal current on an equilibrium with no "
                + "current profile assigned"
            )
        profile = eq.current
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["I"]
        super().build(eq, profile, use_jit, verbose)

    def compute(self, c_l, **kwargs):
        """Compute fixed current errors.

        Parameters
        ----------
        c_l : ndarray
            parameters of the current profile.

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        fixed_params = c_l[self._idx]
        return self._shift_scale(fixed_params)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "c_l"


class FixPsi(_Objective):
    """Fixes total toroidal magnetic flux within the last closed flux surface.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, optional
        Target value(s) of the objective. If None, uses Equilibrium value.
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    weight : float, optional
        Weighting to apply to the Objective, relative to other Objectives.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = True
    _fixed = True
    _units = "(Wb)"
    _print_value_fmt = "Fixed-Psi error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed-Psi",
    ):

        super().__init__(
            eq=eq,
            target=target,
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
        self._dim_f = 1

        if self.target is None:
            self.target = eq.Psi

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["Psi"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, Psi, **kwargs):
        """Compute fixed-Psi error.

        Parameters
        ----------
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Total toroidal magnetic flux error (Wb).

        """
        return self._shift_scale(Psi)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Psi"
