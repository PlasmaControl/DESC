"""Classes for linear optimization constraints.

Linear objective functions must be of the form `A*x-b`, where:
    - `A` is a constant matrix that can be pre-computed
    - `x` is an array of a single parameter
    - `b` is the desired vector set by `objective.target`
"""

import warnings

import numpy as np
from termcolor import colored

from desc.backend import (
    execute_on_cpu,
    jnp,
    tree_leaves,
    tree_map,
    tree_map_with_path,
    tree_structure,
)
from desc.basis import zernike_radial
from desc.geometry import FourierRZCurve
from desc.utils import broadcast_tree, errorif, setdefault

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


# TODO (#1391): get rid of this class and inherit from FixParameters instead?
class _FixedObjective(_Objective):
    _fixed = True
    _linear = True
    _scalar = False

    def update_target(self, thing):
        """Update target values using an Optimizable object.

        Parameters
        ----------
        thing : Optimizable
            Optimizable object that will be optimized to satisfy the Objective.

        """
        new_target = self.compute(thing.params_dict)
        assert len(new_target) == len(self.target)
        self.target = new_target
        self._target_from_user = self.target  # in case the Objective is re-built
        if not self._use_jit:
            self._unjit()

    def _parse_target_from_user(
        self, target_from_user, default_target, default_bounds, idx
    ):
        # TODO (#1391): add logic here to deal with `target_from_user` as a pytree?
        # TODO (#1391: does this actually need idx?
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


class FixParameters(_Objective):
    """Fix specific degrees of freedom associated with a given Optimizable thing.

    Parameters
    ----------
    thing : Optimizable
        Object whose degrees of freedom are being fixed.
    params : nested list of dicts
        Dict keys are the names of parameters to fix (str), and dict values are the
        indices to fix for each corresponding parameter (int array).
        Use True (False) instead of an int array to fix all (none) of the indices
        for that parameter.
        Must have the same pytree structure as thing.params_dict.
        The default is to fix all indices of all parameters.
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
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Has no effect for this objective.
    name : str, optional
        Name of the objective function.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from desc.coils import (
            CoilSet, FourierPlanarCoil, FourierRZCoil, FourierXYZCoil, MixedCoilSet
        )
        from desc.objectives import FixParameters

        # toroidal field coil set with 4 coils
        tf_coil = FourierPlanarCoil(
            current=3, center=[2, 0, 0], normal=[0, 1, 0], r_n=[1]
        )
        tf_coilset = CoilSet.linspaced_angular(tf_coil, n=4)
        # vertical field coil set with 3 coils
        vf_coil = FourierRZCoil(current=-1, R_n=3, Z_n=-1)
        vf_coilset = CoilSet.linspaced_linear(
            vf_coil, displacement=[0, 0, 2], n=3, endpoint=True
        )
        # another single coil
        xyz_coil = FourierXYZCoil(current=2)
        # full coil set with TF coils, VF coils, and other single coil
        full_coilset = MixedCoilSet((tf_coilset, vf_coilset, xyz_coil))

        params = [
            [
                {"current": True},  # fix "current" of the 1st TF coil
                # fix "center" and one component of "normal" for the 2nd TF coil
                {"center": True, "normal": np.array([1])},
                {"r_n": True},  # fix radius of the 3rd TF coil
                {},  # fix nothing in the 4th TF coil
            ],
            {"shift": True, "rotmat": True},  # fix "shift" & "rotmat" for all VF coils
            # fix specified indices of "X_n" and "Z_n", but not "Y_n", for other coil
            {"X_n": np.array([1, 2]), "Y_n": False, "Z_n": np.array([0])},
        ]
        obj = FixParameters(full_coilset, params)

    """

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(~)"
    _print_value_fmt = "Fixed parameters error: "

    def __init__(
        self,
        thing,
        params=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed parameters",
    ):
        self._params = params
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

        # default params
        default_params = tree_map(lambda dim: np.arange(dim), thing.dimensions)
        self._params = setdefault(self._params, default_params)
        self._params = broadcast_tree(self._params, default_params)
        self._indices = tree_leaves(self._params)
        assert tree_structure(self._params) == tree_structure(default_params)

        self._dim_f = sum(idx.size for idx in self._indices)

        # default target
        if self.target is None and self.bounds is None:
            self.target = np.concatenate(
                [
                    np.atleast_1d(param[idx])
                    for param, idx in zip(tree_leaves(thing.params_dict), self._indices)
                ]
            )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed degree of freedom errors.

        Parameters
        ----------
        params : list of dict
            List of dictionaries of degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed degree of freedom errors.

        """
        return jnp.concatenate(
            [
                jnp.atleast_1d(param[idx])
                for param, idx in zip(tree_leaves(params), self._indices)
            ]
        )

    def update_target(self, thing):
        """Update target values using an Optimizable object.

        Parameters
        ----------
        thing : Optimizable
            Optimizable object that will be optimized to satisfy the Objective.

        """
        self.target = self.compute(thing.params_dict)
        if not self._use_jit:
            self._unjit()


class ShareParameters(_Objective):
    """Fix specific degrees of freedom to be the same between Optimizable things.

    Parameters
    ----------
    things : list of Optimizable
        List of objects whose degrees of freedom are being fixed to
        each other's values. Must be at least length 2, but may be of arbitrary length.
        Every object must be of the same type, and have the same size array for the
        desired parameter to be fixed (e.g. same geometric resolution if fixing
        ``R_lmn``, or same pressure profile resolution if fixing ``p_l``)
    params : dict
        Dict keys are the names of parameters to fix (str), and dict values are the
        indices to fix for each corresponding parameter (int array).
        Use True (False) instead of an int array to fix all (none) of the indices
        for that parameter.
        Must have the same pytree structure as things[0].params_dict.
        The default is to fix all indices of all parameters.
    name : str, optional
        Name of the objective function.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite={
            "target": "",
            "bounds": "",
            "normalize": "",
            "normalize_target": "",
            "weight": "",
        }
    )
    __doc__ += """
    Examples
    --------
    .. code-block:: python

        import numpy as np
        from desc.coils import (
            CoilSet, FourierPlanarCoil, FourierRZCoil, FourierXYZCoil, MixedCoilSet
        )
        from desc.objectives import ShareParameters

        # toroidal field coil set with 4 coils
        tf_coil = FourierPlanarCoil(
            current=3, center=[2, 0, 0], normal=[0, 1, 0], r_n=[1]
        )
        tf_coilset = CoilSet.linspaced_angular(tf_coil, n=4)
        # vertical field coil set with 3 coils
        vf_coil = FourierRZCoil(current=-1, R_n=3, Z_n=-1)
        vf_coilset = CoilSet.linspaced_linear(
            vf_coil, displacement=[0, 0, 2], n=3, endpoint=True
        )
        # another single coil
        xyz_coil = FourierXYZCoil(current=2)
        # full coil set with TF coils, VF coils, and other single coil
        full_coilset = MixedCoilSet((tf_coilset, vf_coilset, xyz_coil))
        coilset2 = full_coilset.copy()

        # between the two coilsets...
        params = [
            [ # share the "current" of the 1st TF coil and 1st center component
                {"current": True, "center":np.array([0])},
                # share "center" and  "normal" for the 2nd TF coil
                {"center": True, "normal": True},
                {"r_n": True},  # share radius of the 3rd TF coil
                {},  # share nothing in the 4th TF coil
            ],
            # share "shift" & "rotmat" for all VF coils
            {"shift": True, "rotmat": True},
            # share specified indices of "X_n" and "Z_n",
            # but not "Y_n", for other coil
            {"X_n": np.array([1, 2]), "Y_n": False, "Z_n": np.array([0])},
        ]
        obj=ShareParameters([full_coilset, coilset2], params=params)

    """
    _scalar = False
    _linear = True
    _fixed = False
    _units = "(~)"
    _print_value_fmt = "Shared parameters error: "

    def __init__(
        self,
        things,
        params=None,
        name="shared parameters",
    ):
        self._params = params
        assert len(things) > 1, "only makes sense for >1 thing"
        assert np.all(
            [isinstance(things[0], type(t)) for t in things[1:]]
        ), f"expected same type for all things, got types {[type(t) for t in things]}"

        super().__init__(
            things=things,
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
        thing = self.things[0]

        # default params
        default_params = tree_map(lambda dim: np.arange(dim), thing.dimensions)
        self._params = setdefault(self._params, default_params)
        self._params = broadcast_tree(self._params, default_params)
        self._indices = tree_leaves(self._params)
        assert tree_structure(self._params) == tree_structure(default_params)

        # check here that the things being shared have the same dimensions
        def look(kp, d1, d2, p):
            # p.size is 0 if user didn't pass them in as params to share,
            # they shouldn't cause failure
            if p.size == 0:
                return True
            else:
                assert d1 == d2, (
                    "At least one parameter that is being shared does not match"
                    " dimensions between 2 or more of the passed things!\n"
                    f"Differing parameter dimensions: {d1} versus {d2}\n"
                    f"Parameter is at pytree key path {kp}\n"
                    "Check that this parameter's dimension is the same across all"
                    " things passed to ShareParameters.\n"
                    "\nSee below link if confused on what a key path is:\n"
                    "https://docs.jax.dev/en/latest/pytrees.html#explicit-key-paths"
                )
                return d1 == d2

        for t2 in self.things[1:]:
            tree_map_with_path(look, thing.dimensions, t2.dimensions, self._params)

        self._dim_f = sum(idx.size for idx in self._indices) * (len(self.things) - 1)

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, *params, constants=None):
        """Compute shared degree of freedom errors.

        Parameters
        ----------
        params : dict
            2 or more dictionaries of params to fix parameters between.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants.

        Returns
        -------
        f : ndarray
            Shared degree of freedom errors.

        """
        # basically, just subtract the first things' params
        # and every subsequent thing (adding on rows to dim_f if
        # more than 2 total things) so that the Jacobian of this
        # ends up being just rows with 1 in the first object's params
        # indices and -1 in the second objects params indices,
        # repeated vertically for each additional object
        # i.e. for a size-1 param being shared among 4 objects, the
        # Jacobian looks like
        #  [ 1 -1  0  0]
        #  [ 1 0  -1  0]
        #  [ 1 0   0 -1]
        params_1 = params[0]
        reference_params_array = jnp.concatenate(
            [
                jnp.atleast_1d(param[idx])
                for param, idx in zip(tree_leaves(params_1), self._indices)
            ]
        )
        return jnp.concatenate(
            [
                reference_params_array
                - jnp.concatenate(
                    [
                        jnp.atleast_1d(param[idx])
                        for param, idx in zip(tree_leaves(this_params), self._indices)
                    ]
                )
                for this_params in params[1:]
            ]
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

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite={
            "target": "",
            "bounds": "",
            "normalize": "",
            "normalize_target": "",
            "weight": "",
        }
    )
    _scalar = False
    _linear = True
    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(m)"
    _print_value_fmt = "R boundary self consistency error: "

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

    @execute_on_cpu
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
        Js = []
        surf = eq.surface.rho if self._surface_label is None else self._surface_label
        for i, (l, m, n) in enumerate(eq.R_basis.modes):
            if eq.bdry_mode == "lcfs":
                j = np.argwhere((modes[:, 1:] == [m, n]).all(axis=1))
                Js.append(j.flatten())
            else:
                raise NotImplementedError(
                    "bdry_mode is not lcfs, yell at Dario to finish poincare stuff"
                )
        Js = np.array(Js)
        # Broadcasting at once is faster. We need to use np.arange to avoid
        # setting the value to the whole row.
        self._A[Js[:, 0], np.arange(eq.R_basis.num_modes)] = zernike_radial(
            surf, eq.R_basis.modes[:, 0], eq.R_basis.modes[:, 1]
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

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite={
            "target": "",
            "bounds": "",
            "normalize": "",
            "normalize_target": "",
            "weight": "",
        }
    )
    _scalar = False
    _linear = True
    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(m)"
    _print_value_fmt = "Z boundary self consistency error: "

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

    @execute_on_cpu
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
        Js = []
        surf = eq.surface.rho if self._surface_label is None else self._surface_label
        for i, (l, m, n) in enumerate(eq.Z_basis.modes):
            if eq.bdry_mode == "lcfs":
                j = np.argwhere((modes[:, 1:] == [m, n]).all(axis=1))
                Js.append(j.flatten())
            else:
                raise NotImplementedError(
                    "bdry_mode is not lcfs, yell at Dario to finish poincare stuff"
                )
        Js = np.array(Js)
        # Broadcasting at once is faster. We need to use np.arange to avoid
        # setting the value to the whole row.
        self._A[Js[:, 0], np.arange(eq.Z_basis.num_modes)] = zernike_radial(
            surf, eq.Z_basis.modes[:, 0], eq.Z_basis.modes[:, 1]
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

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite={
            "target": "",
            "bounds": "",
            "normalize": "",
            "normalize_target": "",
            "weight": "",
        }
    )
    _scalar = False
    _linear = True
    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(m)"
    _print_value_fmt = "R axis self consistency error: "

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

    @execute_on_cpu
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

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite={
            "target": "",
            "bounds": "",
            "normalize": "",
            "normalize_target": "",
            "weight": "",
        }
    )
    _scalar = False
    _linear = True
    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(m)"
    _print_value_fmt = "Z axis self consistency error: "

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

    @execute_on_cpu
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


class FixBoundaryR(FixParameters):
    """Boundary condition on the R boundary parameters.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Rb_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Rb_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
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
    name : str, optional
        Name of the objective function.

    """

    _units = "(m)"
    _print_value_fmt = "R boundary error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="lcfs R",
    ):
        if isinstance(modes, bool):
            indices = modes
        else:
            indices = np.array([], dtype=int)
            for mode in np.atleast_2d(modes):
                indices = np.append(indices, eq.surface.R_basis.get_idx(*mode))
        super().__init__(
            thing=eq,
            params={"Rb_lmn": indices},
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
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixBoundaryZ(FixParameters):
    """Boundary condition on the Z boundary parameters.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Zb_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Zb_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
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
    name : str, optional
        Name of the objective function.

    """

    _units = "(m)"
    _print_value_fmt = "Z boundary error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="lcfs Z",
    ):
        if isinstance(modes, bool):
            indices = modes
        else:
            indices = np.array([], dtype=int)
            for mode in np.atleast_2d(modes):
                indices = np.append(indices, eq.surface.Z_basis.get_idx(*mode))
        super().__init__(
            thing=eq,
            params={"Zb_lmn": indices},
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
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixLambdaGauge(FixParameters):
    """Fixes gauge freedom for lambda, which sets the flux surface avg of lambda to 0.

    Note: this constraint is automatically applied when needed, and does not need to be
    included by the user.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    name : str, optional
        Name of the objective function.

    """

    _units = "(rad)"
    _print_value_fmt = "lambda gauge error: "

    def __init__(
        self,
        eq,
        normalize=True,
        normalize_target=True,
        name="lambda gauge",
    ):
        if eq.sym:
            indices = False
        else:
            indices = np.where(
                np.logical_and(eq.L_basis.modes[:, 1] == 0, eq.L_basis.modes[:, 2] == 0)
            )[0]
        super().__init__(
            thing=eq,
            params={"L_lmn": indices},
            target=0,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixThetaSFL(FixParameters):
    """Fixes lambda=0 so that poloidal angle is the SFL poloidal angle.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    name : str, optional
        Name of the objective function.

    """

    _units = "(rad)"
    _print_value_fmt = "theta - theta SFL error: "

    def __init__(
        self,
        eq,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="theta SFL",
    ):
        super().__init__(
            thing=eq,
            params={"L_lmn": True},
            target=0,
            bounds=None,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixAxisR(FixParameters):
    """Fixes magnetic axis R coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Ra_n``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Ra_n``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
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

    _units = "(m)"
    _print_value_fmt = "R axis error: "

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
        if isinstance(modes, bool):
            indices = modes
        else:
            indices = np.array([], dtype=int)
            for mode in np.atleast_2d(modes):
                indices = np.append(indices, eq.axis.R_basis.get_idx(*mode))
        super().__init__(
            thing=eq,
            params={"Ra_n": indices},
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
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixAxisZ(FixParameters):
    """Fixes magnetic axis Z coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Za_n``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Za_n``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
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

    _units = "(m)"
    _print_value_fmt = "Z axis error: "

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
        if isinstance(modes, bool):
            indices = modes
        else:
            indices = np.array([], dtype=int)
            for mode in np.atleast_2d(modes):
                indices = np.append(indices, eq.axis.Z_basis.get_idx(*mode))
        super().__init__(
            thing=eq,
            params={"Za_n": indices},
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
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixModeR(FixParameters):
    """Fixes Fourier-Zernike R coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.R_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.R_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the basis modes.
    name : str, optional
        Name of the objective function.

    """

    _units = "(m)"
    _print_value_fmt = "Fixed-R modes error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="fix mode R",
    ):
        if isinstance(modes, bool):
            indices = modes
        else:
            indices = np.array([], dtype=int)
            for mode in np.atleast_2d(modes):
                indices = np.append(indices, eq.R_basis.get_idx(*mode))
        super().__init__(
            thing=eq,
            params={"R_lmn": indices},
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
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixModeZ(FixParameters):
    """Fixes Fourier-Zernike Z coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Z_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Z_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the basis modes.
    name : str, optional
        Name of the objective function.

    """

    _units = "(m)"
    _print_value_fmt = "Fixed-Z modes error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="fix mode Z",
    ):
        if isinstance(modes, bool):
            indices = modes
        else:
            indices = np.array([], dtype=int)
            for mode in np.atleast_2d(modes):
                indices = np.append(indices, eq.Z_basis.get_idx(*mode))
        super().__init__(
            thing=eq,
            params={"Z_lmn": indices},
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
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixModeLambda(FixParameters):
    """Fixes Fourier-Zernike lambda coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Fourier-Zernike lambda coefficient target values.
        Must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.L_lmn``.
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.L_lmn``.
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``.
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of Fourier-Zernike modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the basis modes.
    name : str
        Name of the objective function.

    """

    _units = "(rad)"
    _print_value_fmt = "Fixed lambda modes error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        modes=True,
        name="fix mode lambda",
    ):
        if isinstance(modes, bool):
            indices = modes
        else:
            indices = np.array([], dtype=int)
            for mode in np.atleast_2d(modes):
                indices = np.append(indices, eq.L_basis.get_idx(*mode))
        super().__init__(
            thing=eq,
            params={"L_lmn": indices},
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixSumModesR(_FixedObjective):
    """Fixes a linear sum of Fourier-Zernike R coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.R_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.R_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    sum_weights : float, ndarray, optional
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

    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-R sum modes error: "

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
            # rearrange modes & weights to match order of eq.R_basis.modes and eq.R_lmn,
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
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Z_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Z_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    sum_weights : float, ndarray, optional
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

    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-Z sum modes error: "

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

    @execute_on_cpu
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
            # rearrange modes & weights to match order of eq.Z_basis.modes and eq.Z_lmn,
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
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.L_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.L_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``.
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
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

    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(rad)"
    _print_value_fmt = "Fixed-lambda sum modes error: "

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
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    @execute_on_cpu
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


class FixPressure(FixParameters):
    """Fixes pressure coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.p_l``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.p_l``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _units = "(Pa)"
    _print_value_fmt = "Fixed pressure profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed pressure",
    ):
        super().__init__(
            thing=eq,
            params={"p_l": indices},
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
        if eq.pressure is None:
            raise RuntimeError(
                "Attempting to fix pressure on an Equilibrium with no "
                + "pressure profile assigned."
            )
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["p"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixAnisotropy(FixParameters):
    """Fixes anisotropic pressure coefficients.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.a_lmn``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.a_lmn``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str
        Name of the objective function.

    """

    _units = "(dimensionless)"
    _print_value_fmt = "Fixed anisotropy profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed anisotropy",
    ):
        super().__init__(
            thing=eq,
            params={"a_lmn": indices},
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
        if eq.anisotropy is None:
            raise RuntimeError(
                "Attempting to fix anisotropy on an Equilibrium with no "
                + "anisotropy profile assigned."
            )
        super().build(use_jit=use_jit, verbose=verbose)


class FixIota(FixParameters):
    """Fixes rotational transform coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.i_l``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.i_l``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices.
        corresponding to knots for a SplineProfile).
        Must len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _units = "(dimensionless)"
    _print_value_fmt = "Fixed iota profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed iota",
    ):
        super().__init__(
            thing=eq,
            params={"i_l": indices},
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
        if eq.iota is None:
            raise RuntimeError(
                "Attempting to fix iota on an Equilibrium with no "
                + "iota profile assigned."
            )
        super().build(use_jit=use_jit, verbose=verbose)


class FixCurrent(FixParameters):
    """Fixes toroidal current profile coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.c_l``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.c_l``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _units = "(A)"
    _print_value_fmt = "Fixed current profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed current",
    ):
        super().__init__(
            thing=eq,
            params={"c_l": indices},
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
        if eq.current is None:
            raise RuntimeError(
                "Attempting to fix current on an Equilibrium with no "
                + "current profile assigned."
            )
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["I"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixElectronTemperature(FixParameters):
    """Fixes electron temperature profile coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Te_l``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Te_l``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _units = "(eV)"
    _print_value_fmt = "Fixed electron temperature profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed electron temperature",
    ):
        super().__init__(
            thing=eq,
            params={"Te_l": indices},
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
        if eq.electron_temperature is None:
            raise RuntimeError(
                "Attempting to fix electron temperature on an Equilibrium with no "
                + "electron temperature profile assigned."
            )
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["T"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixElectronDensity(FixParameters):
    """Fixes electron density profile coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.ne_l``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.ne_l``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
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
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _units = "(m^-3)"
    _print_value_fmt = "Fixed electron density profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed electron density",
    ):
        super().__init__(
            thing=eq,
            params={"ne_l": indices},
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
        if eq.electron_density is None:
            raise RuntimeError(
                "Attempting to fix electron density on an Equilibrium with no "
                + "electron density profile assigned."
            )
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["n"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixIonTemperature(FixParameters):
    """Fixes ion temperature profile coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Ti_l``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Ti_l``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _units = "(eV)"
    _print_value_fmt = "Fixed ion temperature profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed ion temperature",
    ):
        super().__init__(
            thing=eq,
            params={"Ti_l": indices},
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
        if eq.ion_temperature is None:
            raise RuntimeError(
                "Attempting to fix ion temperature on an Equilibrium with no "
                + "ion temperature profile assigned."
            )
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["T"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixAtomicNumber(FixParameters):
    """Fixes effective atomic number profile coefficients.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Defaults to ``target=eq.Zeff_l``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Defaults to ``target=eq.Zeff_l``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    indices : ndarray or bool, optional
        indices of the Profile.params array to fix.
        (e.g. indices corresponding to modes for a PowerSeriesProfile or indices
        corresponding to knots for a SplineProfile).
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the Profile.params indices.
    name : str, optional
        Name of the objective function.

    """

    _units = "(dimensionless)"
    _print_value_fmt = "Fixed atomic number profile error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed atomic number",
    ):
        super().__init__(
            thing=eq,
            params={"Zeff_l": indices},
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
        if eq.atomic_number is None:
            raise RuntimeError(
                "Attempting to fix atomic number on an Equilibrium with no "
                + "atomic_number profile assigned."
            )
        super().build(use_jit=use_jit, verbose=verbose)


class FixPsi(FixParameters):
    """Fixes total toroidal magnetic flux within the last closed flux surface.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. Default is ``target=eq.Psi``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Default is ``target=eq.Psi``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    name : str, optional
        Name of the objective function.

    """

    _units = "(Wb)"
    _print_value_fmt = "Fixed Psi error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed Psi",
    ):
        super().__init__(
            thing=eq,
            params={"Psi": True},
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
        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["Psi"]
        super().build(use_jit=use_jit, verbose=verbose)


class FixCurveShift(FixParameters):
    """Fixes Curve.shift attribute, which is redundant with other Curve params.

    Parameters
    ----------
    curve : Curve
        Curve that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    name : str, optional
        Name of the objective function.

    """

    _units = "(m)"
    _print_value_fmt = "Fixed shift error: "

    def __init__(
        self,
        curve,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed shift",
    ):
        super().__init__(
            thing=curve,
            params={"shift": True},
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixCurveRotation(FixParameters):
    """Fixes Curve.rotmat attribute, which is redundant with other Curve params.

    Parameters
    ----------
    curve : Curve
        Curve that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    name : str, optional
        Name of the objective function.

    """

    _units = "(rad)"
    _print_value_fmt = "Fixed rotation error: "

    def __init__(
        self,
        curve,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed rotation",
    ):
        super().__init__(
            thing=curve,
            params={"rotmat": True},
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixCoilCurrent(FixParameters):
    """Fixes current(s) in a Coil or CoilSet.

    Parameters
    ----------
    coil : Coil
        Coil(s) that will be optimized to satisfy the Objective.
    target : dict of {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Should have the same tree structure as coil.params.
        Default is ``target=coil.current``.
    bounds : tuple of dict {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Should have the same tree structure as coil.params.
        Default is ``target=coil.current``.
    weight : dict of {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Should be a scalar or have the same tree structure as coil.params.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    indices : nested list of bool, optional
        Pytree of bool specifying which coil currents to fix.
        See the example for how to use this on a mixed coil set.
        If True/False fixes all/none of the coil currents.
    name : str, optional
        Name of the objective function.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from desc.coils import (
            CoilSet, FourierPlanarCoil, FourierRZCoil, FourierXYZCoil, MixedCoilSet
        )
        from desc.objectives import FixCoilCurrent

        # toroidal field coil set with 4 coils
        tf_coil = FourierPlanarCoil(
            current=3, center=[2, 0, 0], normal=[0, 1, 0], r_n=[1]
        )
        tf_coilset = CoilSet.linspaced_angular(tf_coil, n=4)
        # vertical field coil set with 3 coils
        vf_coil = FourierRZCoil(current=-1, R_n=3, Z_n=-1)
        vf_coilset = CoilSet.linspaced_linear(
            vf_coil, displacement=[0, 0, 2], n=3, endpoint=True
        )
        # another single coil
        xyz_coil = FourierXYZCoil(current=2)
        # full coil set with TF coils, VF coils, and other single coil
        full_coilset = MixedCoilSet((tf_coilset, vf_coilset, xyz_coil))

        # fix the current of the 1st & 3rd TF coil
        # fix none of the currents in the VF coil set
        # fix the current of the other coil
        obj = FixCoilCurrent(
            full_coilset, indices=[[True, False, True, False], False, True]
        )

    """

    _units = "(A)"
    _print_value_fmt = "Fixed coil current error: "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed coil current",
    ):
        indices = tree_map(lambda idx: {"current": idx}, indices)
        super().__init__(
            thing=coil,
            params=indices,
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
        coil = self.things[0]
        if self._normalize:
            params = tree_leaves(
                coil.params_dict, is_leaf=lambda x: isinstance(x, dict)
            )
            mean_current = np.mean([np.abs(param["current"]) for param in params])
            self._normalization = np.max((mean_current, 1))
        super().build(use_jit=use_jit, verbose=verbose)


class FixSumCoilCurrent(FixCoilCurrent):
    """Fixes the sum of coil current(s) in a Coil or CoilSet.

    NOTE: When using this objective, take care in knowing the signs of the current in
    the coils and the orientations of the coils. It is possible for coils with the same
    signs of their current to have currents flowing in differing directions in physical
    space due to the orientation of the coils.

    Parameters
    ----------
    coil : Coil
        Coil(s) that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``.
        Default is the objective value for the coil.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Default is to use the target instead.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    indices : nested list of bool, optional
        Pytree of bool specifying which coil currents to sum together.
        See the example for how to use this on a mixed coil set.
        If True/False sums all/none of the coil currents.
    name : str, optional
        Name of the objective function.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from desc.coils import (
            CoilSet, FourierPlanarCoil, FourierRZCoil, FourierXYZCoil, MixedCoilSet
        )
        from desc.objectives import FixSumCoilCurrent

        # toroidal field coil set with 4 coils
        tf_coil = FourierPlanarCoil(
            current=3, center=[2, 0, 0], normal=[0, 1, 0], r_n=[1]
        )
        tf_coilset = CoilSet.linspaced_angular(tf_coil, n=4)
        # vertical field coil set with 3 coils
        vf_coil = FourierRZCoil(current=-1, R_n=3, Z_n=-1)
        vf_coilset = CoilSet.linspaced_linear(
            vf_coil, displacement=[0, 0, 2], n=3, endpoint=True
        )
        # another single coil
        xyz_coil = FourierXYZCoil(current=2)
        # full coil set with TF coils, VF coils, and other single coil
        full_coilset = MixedCoilSet((tf_coilset, vf_coilset, xyz_coil))

        # equilibrium G(rho=1) determines the necessary net poloidal current through
        # the coils (as dictated by Ampere's law)
        # the sign convention is positive poloidal current flows up through the torus
        # hole
        grid_at_surf = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid)
        G_tot = 2*jnp.pi*eq.compute("G", grid=grid_at_surf)["G"][0] / mu_0

        # to use this objective to satisfy Ampere's law for the targeted equilibrium,
        # only coils that link the equilibrium poloidally should be included in the sum,
        # which is the TF coil set and the FourierXYZ coil, but not the VF coil set
        obj = FixSumCoilCurrent(full_coilset, indices=[True, False, True], target=G_tot)

    """

    _scalar = True
    _linear = True
    _fixed = False
    _units = "(A)"
    _print_value_fmt = "Summed coil current error: "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="summed coil current",
    ):
        self._default_target = False
        if target is None and bounds is None:
            self._default_target = True
        super().__init__(
            coil=coil,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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
        super().build(use_jit=use_jit, verbose=verbose)
        self._dim_f = 1
        if self._default_target:
            self.update_target(thing=self.things[0])

    def compute(self, params, constants=None):
        """Compute sum of coil currents.

        Parameters
        ----------
        params : list of dict
            List of dictionaries of degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Sum of coil currents.

        """
        return jnp.sum(
            jnp.concatenate(
                [
                    jnp.atleast_1d(param[idx])
                    for param, idx in zip(tree_leaves(params), self._indices)
                ]
            )
        )


class FixOmniWell(FixParameters):
    """Fixes OmnigenousField.B_lm coefficients.

    Parameters
    ----------
    field : OmnigenousField
        Field that will be optimized to satisfy the Objective.
    target : float, optional
        Target value(s) of the objective. If None, uses field value.
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
    weight : float, optional
        Weighting to apply to the Objective, relative to other Objectives.
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    indices : ndarray or bool, optional
        indices of the field.B_lm array to fix.
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the field.B_lm indices.
    name : str
        Name of the objective function.

    """

    _units = "(T)"
    _print_value_fmt = "Fixed omnigenity well error: "

    def __init__(
        self,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed omnigenity well",
    ):
        super().__init__(
            thing=field,
            params={"B_lm": indices},
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixOmniMap(FixParameters):
    """Fixes OmnigenousField.x_lmn coefficients.

    Parameters
    ----------
    field : OmnigenousField
        Field that will be optimized to satisfy the Objective.
    target : float, optional
        Target value(s) of the objective. If None, uses field value.
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
    weight : float, optional
        Weighting to apply to the Objective, relative to other Objectives.
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    indices : ndarray or bool, optional
        indices of the field.x_lmn array to fix.
        Must have len(target) = len(weight) = len(indices).
        If True/False uses all/none of the field.x_lmn indices.
    name : str
        Name of the objective function.

    """

    _units = "(rad)"
    _print_value_fmt = "Fixed omnigenity map error: "

    def __init__(
        self,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        indices=True,
        name="fixed omnigenity map",
    ):
        super().__init__(
            thing=field,
            params={"x_lmn": indices},
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixOmniBmax(_FixedObjective):
    """Ensures the B_max contour is straight in Boozer coordinates.

    Parameters
    ----------
    field : OmnigenousField
        Field that will be optimized to satisfy the Objective.
    target : float, optional
        Target value(s) of the objective. If None, uses field value.
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
    weight : float, optional
        Weighting to apply to the Objective, relative to other Objectives.
    normalize : bool, optional
        Has no effect for this objective.
    normalize_target : bool, optional
        Has no effect for this objective.
    name : str
        Name of the objective function.

    """

    _fixed = False  # not "diagonal", since it is fixing a sum
    _units = "(rad)"
    _print_value_fmt = "Fixed omnigenity B_max error: "

    def __init__(
        self,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed omnigenity B_max",
    ):
        self._target_from_user = setdefault(bounds, target)
        super().__init__(
            things=field,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    @execute_on_cpu
    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        field = self.things[0]

        basis = field.x_basis
        self._dim_f = int(basis.num_modes / (basis.M + 1))

        self._A = np.zeros((self._dim_f, basis.num_modes))
        m0_modes = basis.modes[np.nonzero(basis.modes[:, 1] == 0)[0]]
        for i, (l, m, n) in enumerate(m0_modes):
            idx_0 = np.nonzero((basis.modes == [l, m, n]).all(axis=1))[0]
            idx_m = np.nonzero(
                np.logical_and(
                    (basis.modes[:, (0, 2)] == [l, n]).all(axis=1),
                    np.logical_and((basis.modes[:, 1] % 2 == 0), basis.modes[:, 1] > 0),
                )
            )[0]
            mm = basis.modes[idx_m, 1]
            self._A[i, idx_0] = 1
            self._A[i, idx_m] = (mm % 2 - 1) * (mm % 4 - 1)

        self.target, self.bounds = self._parse_target_from_user(
            self._target_from_user, 0, None, np.array([0])
        )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed omnigenity B_max error.

        Parameters
        ----------
        params : dict
            Dictionary of field degrees of freedom, eg OmnigenousField.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed omnigenity B_max error.

        """
        f = jnp.dot(self._A, params["x_lmn"])
        return f


class FixSheetCurrent(FixParameters):
    """Fixes the sheet current parameters of a free-boundary equilibrium.

    Note: this constraint is automatically applied when needed, and does not need to be
    included by the user.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``.
        Defaults to the equilibrium sheet current parameters.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``.
        Default is to use target.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    name : str, optional
        Name of the objective function.

    """

    _units = "(~)"
    _print_value_fmt = "Fixed sheet current error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed sheet current",
    ):
        super().__init__(
            thing=eq,
            params={"I": True, "G": True, "Phi_mn": True},
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class FixNearAxisR(_FixedObjective):
    """Fixes an equilibrium's near-axis behavior in R to specified order.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    order : {0,1,2}
        order (in rho) of near-axis behavior to constrain
    N : int
        max toroidal resolution to constrain.
        If `None`, defaults to equilibrium's toroidal resolution
    target : Qsc, optional
        pyQSC Qsc object describing the NAE solution to fix the equilibrium's
        near-axis behavior to. If None, will fix the equilibrium's current near
        axis behavior.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Unused by this objective
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
        Unused by this objective
    name : str, optional
        Name of the objective function.

    """

    _static_attrs = _Objective._static_attrs + ["_nae_eq", "_order"]
    _target_arg = "R_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-Near-Axis R Behavior error: "

    def __init__(
        self,
        eq,
        order=1,
        N=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="Fix Near Axis R Behavior",
    ):
        self._nae_eq = target
        self._eq = eq
        self._order = order
        self._N = N
        super().__init__(
            things=eq,
            target=None,
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
        from .getters import _get_NAE_constraints

        eq = self.things[0]
        cons = _get_NAE_constraints(
            eq, self._nae_eq, order=self._order, N=self._N, fix_lambda=False
        )
        for con in cons:
            if isinstance(con, FixSumModesR) or isinstance(con, AxisRSelfConsistency):
                con.build(use_jit=use_jit, verbose=0)
        cons_that_fix_near_axis = [con for con in cons if isinstance(con, FixSumModesR)]
        self._A = (
            np.stack([con._A for con in cons_that_fix_near_axis]).squeeze()
            if cons_that_fix_near_axis
            else None
        )
        # add the axis constraint last if it is in cons
        axis_con = None
        for con in cons:
            if isinstance(con, AxisRSelfConsistency):
                self._A = (
                    np.vstack([self._A, con._A]).squeeze()
                    if np.any(self._A)
                    else con._A
                )
                axis_con = con

        self._target = (
            np.concatenate([con.target for con in cons_that_fix_near_axis])
            if cons_that_fix_near_axis
            else None
        )
        if axis_con:
            if self._nae_eq is not None:
                # use NAE axis as target
                axis = FourierRZCurve(
                    R_n=np.concatenate(
                        (np.flipud(self._nae_eq.rs[1:]), self._nae_eq.rc)
                    ),
                    Z_n=np.concatenate(
                        (np.flipud(self._nae_eq.zs[1:]), self._nae_eq.zc)
                    ),
                    NFP=self._nae_eq.nfp,
                    sym=eq.sym,
                )
                axis.change_resolution(N=self._eq.N, sym=eq.sym)
                axis_target = axis.R_n
            else:  # else use eq axis a target
                axis_target = self._eq.Ra_n
            self._target = (
                np.append(self._target, axis_target).squeeze()
                if np.any(self._target)
                else axis_target
            )
        self._dim_f = self.target.size
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed near axis R behavior errors.

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
            Fixed near axis behavior errors.

        """
        f = jnp.dot(self._A, params["R_lmn"]).squeeze()

        return f


class FixNearAxisZ(_FixedObjective):
    """Fixes an equilibrium's near-axis behavior in Z to specified order.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    order : {0,1,2}
        order (in rho) of near-axis behavior to constrain
    N : int
        max toroidal resolution to constrain.
        If `None`, defaults to equilibrium's toroidal resolution
    target : Qsc, optional
        pyQSC Qsc object describing the NAE solution to fix the equilibrium's
        near-axis behavior to. If None, will fix the equilibrium's current near
        axis behavior.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Unused by this objective
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
        Unused by this objective
    name : str, optional
        Name of the objective function.

    """

    _static_attrs = _Objective._static_attrs + ["_nae_eq", "_order"]
    _target_arg = "Z_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(m)"
    _print_value_fmt = "Fixed-Near-Axis Z Behavior error: "

    def __init__(
        self,
        eq,
        order=1,
        N=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="Fix Near Axis Z Behavior",
    ):
        self._nae_eq = target
        self._eq = eq
        self._order = order
        self._N = N
        super().__init__(
            things=eq,
            target=None,
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
        from .getters import _get_NAE_constraints

        eq = self.things[0]
        cons = _get_NAE_constraints(
            eq, self._nae_eq, order=self._order, N=self._N, fix_lambda=False
        )
        for con in cons:
            if isinstance(con, FixSumModesZ) or isinstance(con, AxisZSelfConsistency):
                con.build(use_jit=use_jit, verbose=0)
        cons_that_fix_near_axis = [con for con in cons if isinstance(con, FixSumModesZ)]
        self._A = (
            np.stack([con._A for con in cons_that_fix_near_axis]).squeeze()
            if cons_that_fix_near_axis
            else None
        )
        # add the axis constraint last if it is in cons
        axis_con = None
        for con in cons:
            if isinstance(con, AxisZSelfConsistency):
                self._A = (
                    np.vstack([self._A, con._A]).squeeze()
                    if np.any(self._A)
                    else con._A
                )
                axis_con = con
        self._target = (
            np.concatenate([con.target for con in cons_that_fix_near_axis])
            if cons_that_fix_near_axis
            else None
        )
        if axis_con:
            if self._nae_eq is not None:
                # use NAE axis as target
                axis = FourierRZCurve(
                    R_n=np.concatenate(
                        (np.flipud(self._nae_eq.rs[1:]), self._nae_eq.rc)
                    ),
                    Z_n=np.concatenate(
                        (np.flipud(self._nae_eq.zs[1:]), self._nae_eq.zc)
                    ),
                    NFP=self._nae_eq.nfp,
                    sym=eq.sym,
                )
                axis.change_resolution(N=self._eq.N, sym=eq.sym)
                axis_target = axis.Z_n
            else:  # else use eq axis a target
                axis_target = self._eq.Za_n
            self._target = (
                np.append(self._target, axis_target).squeeze()
                if np.any(self._target)
                else axis_target
            )

        self._dim_f = self.target.size

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed near axis Z behavior errors.

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
            Fixed near axis behavior errors.

        """
        f = jnp.dot(self._A, params["Z_lmn"]).squeeze()

        return f


class FixNearAxisLambda(_FixedObjective):
    """Fixes an equilibrium's near-axis behavior in lambda to specified order.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    order : int
        order (in rho) of near-axis lambda behavior to constrain
    N : int
        max toroidal resolution to constrain.
        If `None`, defaults to equilibrium's toroidal resolution
    target : Qsc, optional
        pyQSC Qsc object describing the NAE solution to fix the equilibrium's
        near-axis behavior to. If None, will fix the equilibrium's current near
        axis behavior.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``
        Unused for this objective, as target will be automatically
        set according to the ``nae_eq``
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Unused by this objective
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
        Unused by this objective
    name : str, optional
        Name of the objective function.

    """

    _static_attrs = _Objective._static_attrs + ["_nae_eq", "_order"]
    _target_arg = "L_lmn"
    _fixed = False  # not "diagonal", since its fixing a sum
    _units = "(dimensionless)"
    _print_value_fmt = "Fixed-Near-Axis Lambda Behavior error: "

    def __init__(
        self,
        eq,
        order=1,
        N=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="Fix Near Axis Lambda Behavior",
    ):
        self._nae_eq = target
        self._order = order
        self._N = N
        self._target_from_user = setdefault(bounds, None)
        super().__init__(
            things=eq,
            target=None,
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
        from .getters import _get_NAE_constraints

        eq = self.things[0]
        cons = _get_NAE_constraints(
            eq,
            self._nae_eq,
            order=self._order,
            N=self._N,
            fix_lambda=self._order,
        )
        for con in cons:
            if isinstance(con, FixSumModesLambda):
                con.build(use_jit=use_jit, verbose=0)
        self._A = np.vstack(
            [con._A for con in cons if isinstance(con, FixSumModesLambda)]
        )
        self._target = np.concatenate(
            [con.target for con in cons if isinstance(con, FixSumModesLambda)]
        )

        self._dim_f = self.target.size

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed near axis Lambda behavior errors.

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
            Fixed near axis Lambda behavior errors.

        """
        f = jnp.dot(self._A, params["L_lmn"]).squeeze()
        return f
