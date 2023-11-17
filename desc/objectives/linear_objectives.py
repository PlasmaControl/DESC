"""Classes for linear optimization constraints.

Linear objective functions must be of the form `A*x-b`, where:
    - `A` is a constant matrix that can be pre-computed
    - `x` is an array of a single parameter
    - `b` is the desired vector set by `objective.target`
"""

import warnings
from abc import ABC

import numpy as np
from termcolor import colored

from desc.backend import jnp
from desc.basis import zernike_radial, zernike_radial_coeffs
from desc.utils import errorif, setdefault

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class _FixedObjective(_Objective):
    _fixed = True
    _linear = True
    _scalar = False

    def update_target(self, eq):
        """Update target values using an Equilibrium.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.

        """
        self.target = np.atleast_1d(getattr(eq, self._target_arg, self.target))
        if self._use_jit:
            self.jit()

    def _parse_target_from_user(
        self, target_from_user, default_target, default_bounds, idx
    ):
        if target_from_user is None:
            target = default_target
            bounds = default_bounds
        elif isinstance(target_from_user, tuple) and (
            len(target_from_user) == 2
        ):  # treat as bounds
            target = None
            bounds = (
                np.broadcast_to(target_from_user[0], self._dim_f).copy()[idx],
                np.broadcast_to(target_from_user[1], self._dim_f).copy()[idx],
            )
        else:
            target = np.broadcast_to(target_from_user, self._dim_f).copy()[idx]
            bounds = None
        return target, bounds


# TODO: make this work with above, but for multiple target args?
class FixParameter(_FixedObjective):
    """Fix specific degrees of freedom associated with a given Optimizable object.

    Parameters
    ----------
    thing : Optimizable
        Object whose degrees of freedom are being fixed.
    params : str or list of str
        Names of parameters to fix. Defaults to all parameters.
    index : array-like or list of array-like
        Indices to fix for each parameter in params. Use True to fix all indices.
    target : dict of {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Should have the same tree structure as thing.params. Defaults to things.params.
    bounds : tuple of dict {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Should have the same tree structure as thing.params.
    weight : dict of {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Should be a scalar or have the same tree structure as thing.params.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Has no effect for this objective.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(~)"
    _print_value_fmt = "Fixed parameter error: {:10.3e} "

    def __init__(
        self,
        thing,
        params=None,
        indices=True,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        name="Fixed parameter",
    ):
        self._target_from_user = target
        self._params = params
        self._indices = indices
        super().__init__(
            things=thing,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        thing = self.things[0]
        params = setdefault(self._params, thing.optimizable_params)

        if not isinstance(params, (list, tuple)):
            params = [params]
        for par in params:
            errorif(
                par not in thing.optimizable_params,
                ValueError,
                f"parameter {par} not found in optimizable_parameters: "
                + f"{thing.optimizable_params}",
            )
        self._params = params

        # replace indices=True with actual indices
        if isinstance(self._indices, bool) and self._indices:
            self._indices = [np.arange(thing.dimensions[par]) for par in self._params]
        # make sure its iterable if only a scalar was passed in
        if not isinstance(self._indices, (list, tuple)):
            self._indices = [self._indices]
        # replace idx=True with array of all indices, throwing an error if the length
        # of indices is different from number of params
        indices = {}
        errorif(
            len(self._params) != len(self._indices),
            ValueError,
            f"not enough indices ({len(self._indices)}) "
            + f"for params ({len(self._params)})",
        )
        for idx, par in zip(self._indices, self._params):
            if isinstance(idx, bool) and idx:
                idx = np.arange(thing.dimensions[par])
            indices[par] = np.atleast_1d(idx)
        self._indices = indices
        self._dim_f = sum(t.size for t in self._indices.values())

        default_target = {
            par: thing.params_dict[par][self._indices[par]] for par in params
        }
        default_bounds = None
        target, bounds = self._parse_target_from_user(
            self._target_from_user, default_target, default_bounds, indices
        )
        if target:
            self.target = jnp.concatenate([target[par] for par in params])
            self.bounds = None
        else:
            self.target = None
            self.bounds = (
                jnp.concatenate([bounds[0][par] for par in params]),
                jnp.concatenate([bounds[1][par] for par in params]),
            )
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed degree of freedom errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed degree of freedom errors.

        """
        return jnp.concatenate(
            [params[par][self._indices[par]] for par in self._params]
        )


class BoundaryRSelfConsistency(_Objective):
    """Ensure that the boundary and interior surfaces are self-consistent.

    Note: this constraint is automatically applied when needed, and does not need to be
    included by the user.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    surface_label : float, optional
        Surface to enforce boundary conditions on. Defaults to Equilibrium.surface.rho
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False
    _units = "(m)"
    _print_value_fmt = "R boundary self consistency error: {:10.3e} "

    def __init__(
        self,
        eq,
        surface_label=None,
        name="self_consistency R",
    ):
        self._surface_label = surface_label
        super().__init__(
            things=eq,
            target=0,
            bounds=None,
            weight=1,
            normalize=False,
            normalize_target=False,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        modes = eq.surface.R_basis.modes
        idx = np.arange(eq.surface.R_basis.num_modes)

        self._dim_f = idx.size
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
            else:
                raise NotImplementedError(
                    "bdry_mode is not lcfs, yell at Dario to finish poincare stuff"
                )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary R self-consistency errors.

        IE, the mismatch between the Fourier-Zernike basis evaluated at rho=1 and the
        double Fourier series defining the equilibrium LCFS

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            boundary R self-consistency errors.

        """
        return jnp.dot(self._A, params["R_lmn"]) - params["Rb_lmn"]


class BoundaryZSelfConsistency(_Objective):
    """Ensure that the boundary and interior surfaces are self consistent.

    Note: this constraint is automatically applied when needed, and does not need to be
    included by the user.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    surface_label : float, optional
        Surface to enforce boundary conditions on. Defaults to Equilibrium.surface.rho
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False
    _units = "(m)"
    _print_value_fmt = "Z boundary self consistency error: {:10.3e} "

    def __init__(
        self,
        eq,
        surface_label=None,
        name="self_consistency Z",
    ):
        self._surface_label = surface_label
        super().__init__(
            things=eq,
            target=0,
            bounds=None,
            weight=1,
            normalize=False,
            normalize_target=False,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
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
        modes = eq.surface.Z_basis.modes
        idx = np.arange(eq.surface.Z_basis.num_modes)

        self._dim_f = idx.size
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
            else:
                raise NotImplementedError(
                    "bdry_mode is not lcfs, yell at Dario to finish poincare stuff"
                )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary Z self-consistency errors.

        IE, the mismatch between the Fourier-Zernike basis evaluated at rho=1 and the
        double Fourier series defining the equilibrium LCFS

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            boundary Z self-consistency errors.

        """
        return jnp.dot(self._A, params["Z_lmn"]) - params["Zb_lmn"]


class AxisRSelfConsistency(_Objective):
    """Ensure consistency between Zernike and Fourier coefficients on axis.

    Note: this constraint is automatically applied when needed, and does not need to be
    included by the user.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False
    _print_value_fmt = "R axis self consistency error: {:10.3e} (m)"

    def __init__(
        self,
        eq,
        name="axis R self consistency",
    ):

        super().__init__(
            things=eq,
            target=0,
            weight=1,
            name=name,
            normalize=False,
            normalize_target=False,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        ns = eq.axis.R_basis.modes[:, 2]
        self._dim_f = ns.size
        self._A = np.zeros((self._dim_f, eq.R_basis.num_modes))

        for i, (l, m, n) in enumerate(eq.R_basis.modes):
            if m != 0:
                continue
            if (l // 2) % 2 == 0:
                j = np.argwhere(n == ns)
                self._A[j, i] = 1
            else:
                j = np.argwhere(n == ns)
                self._A[j, i] = -1

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute axis R self-consistency errors.

        IE, the mismatch between the Fourier-Zernike basis evaluated at rho=0 and the
        Fourier series defining the equilibrium axis position

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            axis R self-consistency errors.

        """
        f = jnp.dot(self._A, params["R_lmn"]) - params["Ra_n"]
        return f


class AxisZSelfConsistency(_Objective):
    """Ensure consistency between Zernike and Fourier coefficients on axis.

    Note: this constraint is automatically applied when needed, and does not need to be
    included by the user.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False
    _print_value_fmt = "Z axis self consistency error: {:10.3e} (m)"

    def __init__(
        self,
        eq,
        name="axis Z self consistency",
    ):

        super().__init__(
            things=eq,
            target=0,
            weight=1,
            name=name,
            normalize=False,
            normalize_target=False,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        ns = eq.axis.Z_basis.modes[:, 2]
        self._dim_f = ns.size
        self._A = np.zeros((self._dim_f, eq.Z_basis.num_modes))

        for i, (l, m, n) in enumerate(eq.Z_basis.modes):
            if m != 0:
                continue
            if (l // 2) % 2 == 0:
                j = np.argwhere(n == ns)
                self._A[j, i] = 1
            else:
                j = np.argwhere(n == ns)
                self._A[j, i] = -1

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute axis Z self-consistency errors.

        IE, the mismatch between the Fourier-Zernike basis evaluated at rho=0 and the
        Fourier series defining the equilibrium axis position

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            axis Z self-consistency errors.

        """
        f = jnp.dot(self._A, params["Z_lmn"]) - params["Za_n"]
        return f


class FixBoundaryR(_FixedObjective):
    """Boundary condition on the R boundary parameters.

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
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the profile modes.
    surface_label : float, optional
        Surface to enforce boundary conditions on. Defaults to Equilibrium.surface.rho
    name : str, optional
        Name of the objective function.


    Notes
    -----
    If specifying particular modes to fix, the rows of the resulting constraint `A`
    matrix and `target` vector will be re-sorted according to the ordering of
    `basis.modes` which may be different from the order that was passed in.
    """

    _target_arg = "Rb_lmn"
    _units = "(m)"
    _print_value_fmt = "R boundary error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        surface_label=None,
        name="lcfs R",
    ):
        self._modes = modes
        self._target_from_user = setdefault(bounds, target)
        self._surface_label = surface_label
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            idx = np.array([], dtype=int)
            modes_idx = idx
        elif self._modes is True:  # all modes
            modes = eq.surface.R_basis.modes
            idx = np.arange(eq.surface.R_basis.num_modes)
            modes_idx = idx
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
        # Rb_lmn -> Rb optimization space
        self._A = np.eye(eq.surface.R_basis.num_modes)[idx, :]

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.surface.R_lmn[idx], None, modes_idx
        )

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary R errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            boundary R errors.

        """
        return jnp.dot(self._A, params["Rb_lmn"])


class FixBoundaryZ(_FixedObjective):
    """Boundary condition on the Z boundary parameters.

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
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the surface modes.
    surface_label : float, optional
        Surface to enforce boundary conditions on. Defaults to Equilibrium.surface.rho
    name : str, optional
        Name of the objective function.


    Notes
    -----
    If specifying particular modes to fix, the rows of the resulting constraint `A`
    matrix and `target` vector will be re-sorted according to the ordering of
    `basis.modes` which may be different from the order that was passed in.
    """

    _target_arg = "Zb_lmn"
    _units = "(m)"
    _print_value_fmt = "Z boundary error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        surface_label=None,
        name="lcfs Z",
    ):
        self._modes = modes
        self._target_from_user = setdefault(bounds, target)
        self._surface_label = surface_label
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            idx = np.array([], dtype=int)
            modes_idx = idx
        elif self._modes is True:  # all modes
            modes = eq.surface.Z_basis.modes
            idx = np.arange(eq.surface.Z_basis.num_modes)
            modes_idx = idx
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
        # Zb_lmn -> Zb optimization space
        self._A = np.eye(eq.surface.Z_basis.num_modes)[idx, :]

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.surface.Z_lmn[idx], None, modes_idx
        )

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary Z errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            boundary Z errors.

        """
        return jnp.dot(self._A, params["Zb_lmn"])


class FixLambdaGauge(_Objective):
    """Fixes gauge freedom for lambda: lambda(theta=0,zeta=0)=0.

    Note: this constraint is automatically applied when needed, and does not need to be
    included by the user.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False
    _units = "(radians)"
    _print_value_fmt = "lambda gauge error: {:10.3e} "

    def __init__(
        self,
        eq,
        name="lambda gauge",
    ):
        super().__init__(
            things=eq,
            target=0,
            bounds=None,
            weight=1,
            normalize=False,
            normalize_target=False,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        L_basis = eq.L_basis

        if L_basis.sym:
            self._A = np.zeros((0, L_basis.num_modes))
        else:
            # l(rho,0,0) = 0
            # at theta=zeta=0, basis for lambda reduces to just a polynomial in rho
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
            self._A = A

        self._dim_f = self._A.shape[0]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute lambda gauge freedom errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            gauge freedom errors.

        """
        return jnp.dot(self._A, params["L_lmn"])


class FixThetaSFL(_Objective):
    """Fixes lambda=0 so that poloidal angle is the SFL poloidal angle.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(radians)"
    _print_value_fmt = "Theta - Theta SFL error: {:10.3e} "

    def __init__(self, eq, name="Theta SFL"):

        super().__init__(things=eq, target=0, weight=1, name=name)

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        idx = np.arange(eq.L_basis.num_modes)
        modes_idx = idx
        self._idx = idx

        self._dim_f = modes_idx.size

        self.target = np.zeros_like(modes_idx)

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Theta SFL errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Theta - Theta SFL errors.

        """
        fixed_params = params["L_lmn"][self._idx]
        return fixed_params


class FixAxisR(_FixedObjective):
    """Fixes magnetic axis R coefficients.

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
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of axis modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the axis modes.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Ra_n"
    _units = "(m)"
    _print_value_fmt = "R axis error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="axis R",
    ):
        self._modes = modes
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]

        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            idx = np.array([], dtype=int)
            modes_idx = idx
        elif self._modes is True:  # all modes
            modes = eq.axis.R_basis.modes
            idx = np.arange(eq.axis.R_basis.num_modes)
            modes_idx = idx
        else:  # specified modes
            modes = np.atleast_1d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.axis.R_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            # rearrange modes to match order of eq.axis.R_basis.modes
            # and eq.axis.R_n,
            # necessary so that the A matrix rows match up with the target b
            modes = np.atleast_2d(eq.axis.R_basis.modes[idx, :])

            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the axis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = idx.size
        # Ra_lmn -> Ra optimization space
        self._A = np.eye(eq.axis.R_basis.num_modes)[idx, :]

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.axis.R_n[idx], None, modes_idx
        )

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute axis R errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Axis R errors.

        """
        f = jnp.dot(self._A, params["Ra_n"])
        return f


class FixAxisZ(_FixedObjective):
    """Fixes magnetic axis Z coefficients.

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
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of axis modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the axis modes.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Za_n"
    _units = "(m)"
    _print_value_fmt = "Z axis error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="axis Z",
    ):
        self._modes = modes
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]

        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            idx = np.array([], dtype=int)
            modes_idx = idx
        elif self._modes is True:  # all modes
            modes = eq.axis.Z_basis.modes
            idx = np.arange(eq.axis.Z_basis.num_modes)
            modes_idx = idx
        else:  # specified modes
            modes = np.atleast_1d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.axis.Z_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            # rearrange modes to match order of eq.axis.Z_basis.modes
            # and eq.axis.Z_n,
            # necessary so that the A matrix rows match up with the target b
            modes = np.atleast_2d(eq.axis.Z_basis.modes[idx, :])

            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the axis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = idx.size
        # Za_lmn -> Za optimization space
        self._A = np.eye(eq.axis.Z_basis.num_modes)[idx, :]

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.axis.Z_n[idx], None, modes_idx
        )

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute axis Z errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Axis Z errors.

        """
        f = jnp.dot(self._A, params["Za_n"])
        return f


class FixModeR(_FixedObjective):
    """Fixes Fourier-Zernike R coefficients.

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
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix.
        len(target) = len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "R_lmn"
    _units = "(m)"
    _print_value_fmt = "Fixed-R modes error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="Fix Mode R",
    ):
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixModeR got {modes}"
            )
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is True:  # all modes
            modes = eq.R_basis.modes
            idx = np.arange(eq.R_basis.num_modes)
            modes_idx = idx
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.R_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            self._idx = idx
            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the basis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = modes_idx.size

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.R_lmn[idx], None, modes_idx
        )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Fixed mode R errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed mode R errors.

        """
        fixed_params = params["R_lmn"][self._idx]
        return fixed_params


class FixModeZ(_FixedObjective):
    """Fixes Fourier-Zernike Z coefficients.

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
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix.
        len(target) = len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Z_lmn"
    _units = "(m)"
    _print_value_fmt = "Fixed-Z modes error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="Fix Mode Z",
    ):
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                f"modes kwarg must be specified or True with FixModeZ got {modes}"
            )
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is True:  # all modes
            modes = eq.Z_basis.modes
            idx = np.arange(eq.Z_basis.num_modes)
            modes_idx = idx
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.Z_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            self._idx = idx
            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the basis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = modes_idx.size

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.Z_lmn[idx], None, modes_idx
        )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Fixed mode Z errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed mode Z errors.

        """
        fixed_params = params["Z_lmn"][self._idx]
        return fixed_params


class FixModeLambda(_FixedObjective):
    """Fixes Fourier-Zernike lambda coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Fourier-Zernike lambda coefficient target values. If None,
         uses Equilibrium's lambda coefficients.
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix.
        len(target) = len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    name : str
        Name of the objective function.

    """

    _target_arg = "L_lmn"
    _units = "(dimensionless)"
    _print_value_fmt = "Fixed-lambda modes error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="Fix Mode lambda",
    ):
        self._modes = modes
        if modes is None or modes is False:
            raise ValueError(
                "modes kwarg must be specified"
                + f" or True with FixModeLambda got {modes}"
            )
        self._target_from_user = target
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is True:  # all modes
            modes = eq.L_basis.modes
            idx = np.arange(eq.L_basis.num_modes)
            modes_idx = idx
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.L_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            self._idx = idx
            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the basis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = modes_idx.size

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.L_lmn[idx], None, modes_idx
        )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Fixed mode lambda errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed mode lambda errors.

        """
        fixed_params = params["L_lmn"][self._idx]
        return fixed_params


class FixSumModesR(_FixedObjective):
    """Fixes a linear sum of Fourier-Zernike R coefficients.

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
        Defaults to 1 i.e. target = 1*R_111 + 1*R_222...
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
    _print_value_fmt = "Fixed-R sum modes error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        sum_weights=None,
        modes=True,
        name="Fix Sum Modes R",
    ):

        errorif(
            modes is None or modes is False,
            ValueError,
            f"modes kwarg must be specified or True with FixSumModesR got {modes}",
        )
        errorif(
            target is not None and np.asarray(target).size > 1,
            ValueError,
            "FixSumModesR only accepts 1 target value, please use multiple"
            + " FixSumModesR objectives if you wish to have multiple"
            + " sets of constrained mode sums",
        )
        errorif(
            bounds is not None and np.asarray(bounds)[0].size > 1,
            ValueError,
            "FixSumModesR only accepts 1 target value, please use multiple"
            + " FixSumModesR objectives if you wish to have multiple"
            + " sets of constrained mode sums",
        )
        self._modes = modes
        self._sum_weights = sum_weights
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is True:  # all modes
            modes = eq.R_basis.modes
            idx = np.arange(eq.R_basis.num_modes)
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.R_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            self._idx = idx
            # rearrange modes and weights to match order of eq.R_basis.modes
            # and eq.R_lmn,
            # necessary so that the A matrix rows match up with the target b
            modes = np.atleast_2d(eq.R_basis.modes[idx, :])
            if self._sum_weights is not None:
                self._sum_weights = np.atleast_1d(self._sum_weights)
                self._sum_weights = self._sum_weights[modes_idx]
            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the basis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )
        if self._sum_weights is None:
            sum_weights = np.ones(modes.shape[0])
        else:
            sum_weights = np.atleast_1d(self._sum_weights)
        self._dim_f = 1

        self._A = np.zeros((1, eq.R_basis.num_modes))
        for i, (l, m, n) in enumerate(modes):
            j = eq.R_basis.get_idx(L=l, M=m, N=n)
            self._A[0, j] = sum_weights[i]

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user,
            np.dot(sum_weights.T, eq.R_lmn[self._idx]),
            None,
            np.array([0]),
        )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode R errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed sum mode R errors.

        """
        f = jnp.dot(self._A, params["R_lmn"])
        return f


class FixSumModesZ(_FixedObjective):
    """Fixes a linear sum of Fourier-Zernike Z coefficients.

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
    _print_value_fmt = "Fixed-Z sum modes error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        sum_weights=None,
        modes=True,
        name="Fix Sum Modes Z",
    ):
        errorif(
            modes is None or modes is False,
            ValueError,
            f"modes kwarg must be specified or True with FixSumModesZ got {modes}",
        )
        errorif(
            target is not None and np.asarray(target).size > 1,
            ValueError,
            "FixSumModesZ only accepts 1 target value, please use multiple"
            + " FixSumModesZ objectives if you wish to have multiple"
            + " sets of constrained mode sums",
        )
        errorif(
            bounds is not None and np.asarray(bounds)[0].size > 1,
            ValueError,
            "FixSumModesZ only accepts 1 target value, please use multiple"
            + " FixSumModesZ objectives if you wish to have multiple"
            + " sets of constrained mode sums",
        )
        self._modes = modes
        self._sum_weights = sum_weights
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is True:  # all modes
            modes = eq.Z_basis.modes
            idx = np.arange(eq.Z_basis.num_modes)
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.Z_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            self._idx = idx
            # rearrange modes and weights to match order of eq.Z_basis.modes
            # and eq.Z_lmn,
            # necessary so that the A matrix rows match up with the target b
            modes = np.atleast_2d(eq.Z_basis.modes[idx, :])
            if self._sum_weights is not None:
                self._sum_weights = np.atleast_1d(self._sum_weights)
                self._sum_weights = self._sum_weights[modes_idx]

            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the basis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )
        if self._sum_weights is None:
            sum_weights = np.ones(modes.shape[0])
        else:
            sum_weights = np.atleast_1d(self._sum_weights)
        self._dim_f = 1

        self._A = np.zeros((1, eq.Z_basis.num_modes))
        for i, (l, m, n) in enumerate(modes):
            j = eq.Z_basis.get_idx(L=l, M=m, N=n)
            self._A[0, j] = sum_weights[i]

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user,
            np.dot(sum_weights.T, eq.Z_lmn[self._idx]),
            None,
            np.array([0]),
        )
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode Z errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed sum mode Z errors.

        """
        f = jnp.dot(self._A, params["Z_lmn"])
        return f


class FixSumModesLambda(_FixedObjective):
    """Fixes a linear sum of Fourier-Zernike lambda coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Fourier-Zernike Lambda coefficient target sum. If None,
        uses current sum of Equilibrium's lambda coefficients.
        len(target)=1
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    sum_weight : float, ndarray, optional
        Weights on the coefficients in the sum, should be same length as modes.
        Defaults to 1 i.e. target = 1*L_111 + 1*L_222...
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix sum of.
        len(weight) = len(modes).
        If True uses all of the Equilibrium's modes.
        Must be either True or specified as an array
    surface_label : float
        Surface to enforce boundary conditions on. Defaults to Equilibrium.surface.rho
    name : str
        Name of the objective function.

    """

    _fixed = False  # not "diagonal", since its fixing a sum
    _target_arg = "L_lmn"
    _units = "(dimensionless)"
    _print_value_fmt = "Fixed-lambda sum modes error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        sum_weights=None,
        modes=True,
        name="Fix Sum Modes lambda",
    ):
        errorif(
            modes is None or modes is False,
            ValueError,
            f"modes kwarg must be specified or True with FixSumModesLambda got {modes}",
        )
        errorif(
            target is not None and np.asarray(target).size > 1,
            ValueError,
            "FixSumModesLambda only accepts 1 target value, please use multiple"
            + " FixSumModesLambda objectives if you wish to have multiple"
            + " sets of constrained mode sums",
        )
        errorif(
            bounds is not None and np.asarray(bounds)[0].size > 1,
            ValueError,
            "FixSumModesLambda only accepts 1 target value, please use multiple"
            + " FixSumModesLambda objectives if you wish to have multiple"
            + " sets of constrained mode sums",
        )
        self._modes = modes
        self._sum_weights = sum_weights
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            name=name,
            normalize=normalize,
            normalize_target=normalize_target,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._modes is True:  # all modes
            modes = eq.L_basis.modes
            idx = np.arange(eq.L_basis.num_modes)
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, idx, modes_idx = np.intersect1d(
                eq.L_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            self._idx = idx
            # rearrange modes and weights to match order of eq.L_basis.modes
            # and eq.L_lmn,
            # necessary so that the A matrix rows match up with the target b
            modes = np.atleast_2d(eq.L_basis.modes[idx, :])
            if self._sum_weights is not None:
                self._sum_weights = np.atleast_1d(self._sum_weights)
                self._sum_weights = self._sum_weights[modes_idx]

            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the basis, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )
        if self._sum_weights is None:
            sum_weights = np.ones(modes.shape[0])
        else:
            sum_weights = np.atleast_1d(self._sum_weights)
        self._dim_f = 1

        self._A = np.zeros((1, eq.L_basis.num_modes))
        for i, (l, m, n) in enumerate(modes):
            j = eq.L_basis.get_idx(L=l, M=m, N=n)
            self._A[0, j] = sum_weights[i]

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user,
            np.dot(sum_weights.T, eq.L_lmn[self._idx]),
            None,
            np.array([0]),
        )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Sum mode lambda errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed sum mode lambda errors.

        """
        f = jnp.dot(self._A, params["L_lmn"])
        return f


class _FixProfile(_FixedObjective, ABC):
    """Fixes profile coefficients (or values, for SplineProfile).

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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or Bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _print_value_fmt = "Fix-profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="",
    ):
        self._profile = profile
        self._indices = indices
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, profile, use_jit=False, verbose=1):
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
        eq = self.things[0]
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

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, self._profile.params[self._idx], None, self._idx
        )

        super().build(use_jit=use_jit, verbose=verbose)


class FixPressure(_FixProfile):
    """Fixes pressure coefficients.

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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "p_l"
    _units = "(Pa)"
    _print_value_fmt = "Fixed-pressure profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
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
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if eq.pressure is None:
            raise RuntimeError(
                "Attempting to fix pressure on an equilibrium with no "
                + "pressure profile assigned"
            )
        profile = eq.pressure
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["p"]
        super().build(eq, profile, use_jit, verbose)

    def compute(self, params, constants=None):
        """Compute fixed pressure profile errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["p_l"][self._idx]


class FixAnisotropy(_FixProfile):
    """Fixes anisotropic pressure coefficients.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : tuple, float, ndarray, optional
        Target value(s) of the objective.
        len(target) = len(weight) = len(modes). If None, uses profile coefficients.
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(target) = len(weight) = len(modes)
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
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

    _target_arg = "a_lmn"
    _units = "(dimensionless)"
    _print_value_fmt = "Fixed-anisotropy profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="fixed-anisotropy",
    ):
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
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
        if eq.anisotropy is None:
            raise RuntimeError(
                "Attempting to fix anisotropy on an equilibrium with no "
                + "anisotropy profile assigned"
            )
        profile = eq.anisotropy
        super().build(eq, profile, use_jit, verbose)

    def compute(self, params, constants=None):
        """Compute fixed pressure profile errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["a_lmn"][self._idx]


class FixIota(_FixProfile):
    """Fixes rotational transform coefficients.

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
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Has no effect for this objective.
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices.
        corresponding to knots for a SplineProfile).
        Must len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "i_l"
    _units = "(dimensionless)"
    _print_value_fmt = "Fixed-iota profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
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
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if eq.iota is None:
            raise RuntimeError(
                "Attempt to fix rotational transform on an equilibrium with no "
                + "rotational transform profile assigned"
            )
        profile = eq.iota
        super().build(eq, profile, use_jit, verbose)

    def compute(self, params, constants=None):
        """Compute fixed iota errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["i_l"][self._idx]


class FixCurrent(_FixProfile):
    """Fixes toroidal current profile coefficients.

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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "c_l"
    _units = "(A)"
    _print_value_fmt = "Fixed-current profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
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
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
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

    def compute(self, params, constants=None):
        """Compute fixed current errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["c_l"][self._idx]


class FixElectronTemperature(_FixProfile):
    """Fixes electron temperature profile coefficients.

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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Te_l"
    _units = "(eV)"
    _print_value_fmt = "Fixed-electron-temperature profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="fixed-electron-temperature",
    ):
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
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
        if eq.electron_temperature is None:
            raise RuntimeError(
                "Attempting to fix electron temperature on an equilibrium with no "
                + "electron temperature profile assigned"
            )
        profile = eq.electron_temperature
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["T"]
        super().build(eq, profile, use_jit, verbose)

    def compute(self, params, constants=None):
        """Compute fixed electron temperature errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["Te_l"][self._idx]


class FixElectronDensity(_FixProfile):
    """Fixes electron density profile coefficients.

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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "ne_l"
    _units = "(m^-3)"
    _print_value_fmt = "Fixed-electron-density profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="fixed-electron-density",
    ):
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
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
        if eq.electron_density is None:
            raise RuntimeError(
                "Attempting to fix electron density on an equilibrium with no "
                + "electron density profile assigned"
            )
        profile = eq.electron_density
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["n"]
        super().build(eq, profile, use_jit, verbose)

    def compute(self, params, constants=None):
        """Compute fixed electron density errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["ne_l"][self._idx]


class FixIonTemperature(_FixProfile):
    """Fixes ion temperature profile coefficients.

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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Ti_l"
    _units = "(eV)"
    _print_value_fmt = "Fixed-ion-temperature profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        profile=None,
        indices=True,
        name="fixed-ion-temperature",
    ):
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
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
        if eq.ion_temperature is None:
            raise RuntimeError(
                "Attempting to fix ion temperature on an equilibrium with no "
                + "ion temperature profile assigned"
            )
        profile = eq.ion_temperature
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["T"]
        super().build(eq, profile, use_jit, verbose)

    def compute(self, params, constants=None):
        """Compute fixed ion temperature errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["Ti_l"][self._idx]


class FixAtomicNumber(_FixProfile):
    """Fixes effective atomic number profile coefficients.

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
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Has no effect for this objective.
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(modes).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Zeff_l"
    _units = "(dimensionless)"
    _print_value_fmt = "Fixed-atomic-number profile error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        profile=None,
        indices=True,
        name="fixed-atomic-number",
    ):
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            profile=profile,
            indices=indices,
            name=name,
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
        if eq.atomic_number is None:
            raise RuntimeError(
                "Attempting to fix atomic number on an equilibrium with no "
                + "atomic number profile assigned"
            )
        profile = eq.atomic_number
        super().build(eq, profile, use_jit, verbose)

    def compute(self, params, constants=None):
        """Compute fixed atomic number errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed profile errors.

        """
        return params["Zeff_l"][self._idx]


class FixPsi(_FixedObjective):
    """Fixes total toroidal magnetic flux within the last closed flux surface.

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
    name : str, optional
        Name of the objective function.

    """

    _target_arg = "Psi"
    _units = "(Wb)"
    _print_value_fmt = "Fixed-Psi error: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed-Psi",
    ):
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        self._dim_f = 1

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, eq.Psi, None, np.array([0])
        )

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["Psi"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed-Psi error.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Total toroidal magnetic flux error (Wb).

        """
        return params["Psi"]
