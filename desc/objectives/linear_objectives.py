import numpy as np
from termcolor import colored
import warnings

from desc.backend import jnp
from desc.utils import Timer
from desc.grid import LinearGrid
from desc.basis import (
    zernike_radial_coeffs,
)
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.compute import compute_rotational_transform
from .objective_funs import _Objective


"""Linear objective functions must be of the form `A*x-b`, where:
    - `A` is a constant matrix that can be pre-computed
    - `x` is a vector of one or more arguments included in `compute.arg_order`
    - `b` is the desired vector set by `objective.target`
"""

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
    fixed_boundary : bool, optional
        True to enforce the boundary condition on flux surfaces,
        or Falseto fix the boundary surface coefficients (defualt).
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the profile modes.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False  # TODO: can we dynamically detect this instead?

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        fixed_boundary=False,
        modes=True,
        name="lcfs R",
    ):

        self._fixed_boundary = fixed_boundary
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "R boundary error: {:10.3e} (m)"

        if self._fixed_boundary:
            self.compute = self._compute_R
        else:
            self.compute = self._compute_Rb

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
            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the surface, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = idx.size

        if self._fixed_boundary:  # R_lmn -> Rb_lmn boundary condition
            self._A = np.zeros((self._dim_f, eq.R_basis.num_modes))
            for i, (l, m, n) in enumerate(eq.R_basis.modes):
                if eq.bdry_mode == "lcfs":
                    j = np.argwhere((modes[:, 1:] == [m, n]).all(axis=1))
                self._A[j, i] = 1
        else:  # Rb_lmn -> Rb optimization space
            self._A = np.eye(eq.surface.R_basis.num_modes)[idx, :]

        # use given targets and weights if specified
        if self.target.size == modes.shape[0]:
            self.target = self._target[modes_idx]
        if self.weight.size == modes.shape[0]:
            self.weight = self._weight[modes_idx]

        # use surface parameters as target if needed
        if None in self.target or self.target.size != self.dim_f:
            self.target = eq.surface.R_lmn[idx]

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, *args, **kwargs):
        pass

    def _compute_R(self, R_lmn, **kwargs):
        Rb = jnp.dot(self._A, R_lmn)
        return self._shift_scale(Rb)

    def _compute_Rb(self, Rb_lmn, **kwargs):
        Rb = jnp.dot(self._A, Rb_lmn)
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
    fixed_boundary : bool, optional
        True to enforce the boundary condition on flux surfaces,
        or Falseto fix the boundary surface coefficients (defualt).
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the profile modes.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        fixed_boundary=False,
        modes=True,
        name="lcfs Z",
    ):

        self._fixed_boundary = fixed_boundary
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Z boundary error: {:10.3e} (m)"

        if self._fixed_boundary:
            self.compute = self._compute_Z
        else:
            self.compute = self._compute_Zb

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
            if idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the surface, "
                        + "these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = idx.size

        if self._fixed_boundary:  # Z_lmn -> Zb_lmn boundary condition
            self._A = np.zeros((self._dim_f, eq.Z_basis.num_modes))
            for i, (l, m, n) in enumerate(eq.Z_basis.modes):
                if eq.bdry_mode == "lcfs":
                    j = np.argwhere((modes[:, 1:] == [m, n]).all(axis=1))
                self._A[j, i] = 1
        else:  # Zb_lmn -> Zb optimization space
            self._A = np.eye(eq.surface.Z_basis.num_modes)[idx, :]

        # use given targets and weights if specified
        if self.target.size == modes.shape[0]:
            self.target = self._target[modes_idx]
        if self.weight.size == modes.shape[0]:
            self.weight = self._weight[modes_idx]

        # use surface parameters as target if needed
        if None in self.target or self.target.size != self.dim_f:
            self.target = eq.surface.Z_lmn[idx]

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, *args, **kwargs):
        pass

    def _compute_Z(self, Z_lmn, **kwargs):
        Zb = jnp.dot(self._A, Z_lmn)
        return self._shift_scale(Zb)

    def _compute_Zb(self, Zb_lmn, **kwargs):
        Zb = jnp.dot(self._A, Zb_lmn)
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
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = False

    def __init__(self, eq=None, target=0, weight=1, name="lambda gauge"):

        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "lambda gauge error: {:10.3e} (m)"

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
            # therefore if m!=0, no constraint is needed to make the mode go to zero at rho=0

            # for the other modes with m=0, at rho =0 the basis reduces
            # to just a linear combination of sin(n*zeta), cos(n*zeta), and 1
            # since these are all linearly independent, to make lambda -> 0 at rho=0,
            # each coefficient on these terms must individually go to zero
            # i.e. if at rho=0 the lambda basis is given by
            # Lambda(rho=0) = (L_{00-1} - L_{20-1})sin(zeta) + (L_{001} - L_{201})cos(zeta) + L_{000} - L_{200}
            # Lambda(rho=0) = 0 constraint being enforced means that each coefficient goes to zero:
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
            # what this constraint does is make all of the coefficients of each power of rho
            # equal to zero
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

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

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


class FixPressure(_Objective):
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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the profile modes.
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
        profile=None,
        modes=True,
        name="fixed-pressure",
    ):

        self._profile = profile
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Fixed-pressure profile error: {:10.3e} (Pa)"

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
        if self._profile is None or self._profile.params.size != eq.L + 1:
            self._profile = eq.pressure

        if isinstance(self._profile, PowerSeriesProfile):
            # find indices of pressure modes to fix
            if self._modes is False or self._modes is None:  # no modes
                modes = np.array([[]], dtype=int)
                self._idx = np.array([], dtype=int)
                idx = self._idx
            elif self._modes is True:  # all modes in profile
                modes = self._profile.basis.modes
                self._idx = np.arange(self._profile.basis.num_modes)
                idx = self._idx
            else:  # specified modes
                modes = np.atleast_2d(self._modes)
                dtype = {
                    "names": ["f{}".format(i) for i in range(3)],
                    "formats": 3 * [modes.dtype],
                }
                _, self._idx, idx = np.intersect1d(
                    self._profile.basis.modes.astype(modes.dtype).view(dtype),
                    modes.view(dtype),
                    return_indices=True,
                )
                if self._idx.size < modes.shape[0]:
                    warnings.warn(
                        colored(
                            "Some of the given modes are not in the pressure profile, "
                            + "these modes will not be fixed.",
                            "yellow",
                        )
                    )

            self._dim_f = self._idx.size
            # use given targets and weights if specified
            if self.target.size == modes.shape[0]:
                self.target = self._target[idx]
            if self.weight.size == modes.shape[0]:
                self.weight = self._weight[idx]
            # use profile parameters as target if needed
            if None in self.target or self.target.size != self.dim_f:
                self.target = self._profile.params[self._idx]

        elif isinstance(self._profile, SplineProfile):
            # for spline profile, params = values of the profile at the knot locations
            # and the passed-in modes are actually the desired values at the knot locations

            # find indices of pressure values to fix
            if self._modes is False or self._modes is None:  # no values
                values = np.array([[]], dtype=int)
                self._idx = np.array([], dtype=int)
                idx = self._idx
            elif self._modes is True:  # all values/knot locations in profile
                values = self._profile.params
                self._idx = np.arange(len(self._profile.params))
                idx = self._idx
            else:  # specified values
                # FIXME: not tested and also not sure of what we want to do here
                # we want to be able to I assume fix profile values at certain knot
                # locations but not others (like keep core fixed vary edge etc)
                # in this setup, _modes would be the knot locations we want fixed?
                # I think it would just be that but I am not completely sure
                # might only make sense if target is also supplied?
                raise NotImplementedError(
                    f"Specifying specific values for SplineProfile is not implemented yet."
                )
                knots = np.atleast_2d(self._modes)
                dtype = {
                    "names": ["f{}".format(i) for i in range(3)],
                    "formats": 3 * [values.dtype],
                }
                _, self._idx, idx = np.intersect1d(
                    self._profile._knots.astype(knots.dtype).view(dtype),
                    values.view(dtype),
                    return_indices=True,
                )
                if self._idx.size < knots.shape[0]:
                    warnings.warn(
                        colored(
                            "Some of the given knots are not in the pressure profile, "
                            + "these modes will not be fixed.",
                            "yellow",
                        )
                    )

            self._dim_f = self._idx.size
            # use given targets and weights if specified
            if self.target.size == values.shape[0]:
                self.target = self._target[idx]
            if self.weight.size == values.shape[0]:
                self.weight = self._weight[idx]
            # use profile parameters as target if needed
            if None in self.target or self.target.size != self.dim_f:
                self.target = self._profile.params[self._idx]
        else:
            raise NotImplementedError(
                f"Given pressure profile type for {self._profile} is not implemented yet."
            )
        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, p_l, **kwargs):
        """Compute fixed-pressure profile errors.

        Parameters
        ----------
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.

        Returns
        -------
        f : ndarray
            Pressure profile errors (Pa).

        """
        p = p_l[self._idx]
        return self._shift_scale(p)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "p_l"


class FixIota(_Objective):
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
    profile : Profile, optional
        Profile containing the radial modes to evaluate at.
    modes : ndarray, optional
        Basis modes numbers [l,m,n] of boundary modes to fix.
        len(target) = len(weight) = len(modes).
        If True/False uses all/none of the profile modes.
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
        profile=None,
        modes=True,
        name="fixed-iota",
    ):

        self._profile = profile
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Fixed-iota profile error: {:10.3e}"

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
        if self._profile is None or self._profile.params.size != eq.L + 1:
            self._profile = eq.iota

        if isinstance(self._profile, PowerSeriesProfile):
            # find inidies of iota modes to fix
            if self._modes is False or self._modes is None:  # no modes
                modes = np.array([[]], dtype=int)
                self._idx = np.array([], dtype=int)
                idx = self._idx
            elif self._modes is True:  # all modes in profile
                modes = self._profile.basis.modes
                self._idx = np.arange(self._profile.basis.num_modes)
                idx = self._idx
            else:  # specified modes
                modes = np.atleast_2d(self._modes)
                dtype = {
                    "names": ["f{}".format(i) for i in range(3)],
                    "formats": 3 * [modes.dtype],
                }
                _, self._idx, idx = np.intersect1d(
                    self._profile.basis.modes.astype(modes.dtype).view(dtype),
                    modes.view(dtype),
                    return_indices=True,
                )
                if self._idx.size < modes.shape[0]:
                    warnings.warn(
                        colored(
                            "Some of the given modes are not in the iota profile, "
                            + "these modes will not be fixed.",
                            "yellow",
                        )
                    )

            self._dim_f = self._idx.size

            # use given targets and weights if specified
            if self.target.size == modes.shape[0]:
                self.target = self._target[idx]
            if self.weight.size == modes.shape[0]:
                self.weight = self._weight[idx]
            # use profile parameters as target if needed
            if None in self.target or self.target.size != self.dim_f:
                self.target = self._profile.params[self._idx]

        elif isinstance(self._profile, SplineProfile):
            # FIXME: same things as in FixedPressure

            # find indices of pressure values to fix
            if self._modes is False or self._modes is None:  # no values
                values = np.array([[]], dtype=int)
                self._idx = np.array([], dtype=int)
                idx = self._idx
            elif self._modes is True:  # all values in profile
                values = self._profile.params
                self._idx = np.arange(len(self._profile.params))
                idx = self._idx
            else:  # specified values
                # FIXME: not tested and also not sure of what we want to do here
                # we want to be able to I assume fix profile values at certain knot
                # locations but not others (like keep core fixed vary edge etc)
                # in this setup, _modes would be the knot locations we want fixed?
                # I think it would just be that but I am not completely sure
                # might only make sense if target is also supplied?
                raise NotImplementedError(
                    f"Specifying specific values for SplineProfile is not implemented yet."
                )
                knots = np.atleast_2d(self._modes)
                dtype = {
                    "names": ["f{}".format(i) for i in range(3)],
                    "formats": 3 * [values.dtype],
                }
                _, self._idx, idx = np.intersect1d(
                    self._profile._knots.astype(knots.dtype).view(dtype),
                    values.view(dtype),
                    return_indices=True,
                )
                if self._idx.size < knots.shape[0]:
                    warnings.warn(
                        colored(
                            "Some of the given knots are not in the pressure profile, "
                            + "these modes will not be fixed.",
                            "yellow",
                        )
                    )

            self._dim_f = self._idx.size
            # use given targets and weights if specified
            if self.target.size == values.shape[0]:
                self.target = self._target[idx]
            if self.weight.size == values.shape[0]:
                self.weight = self._weight[idx]
            # use profile parameters as target if needed
            if None in self.target or self.target.size != self.dim_f:
                self.target = self._profile.params[self._idx]
        else:
            raise NotImplementedError(
                f"Given pressure profile type for {self._profile} is not implemented yet."
            )
        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, i_l, **kwargs):
        """Compute fixed-iota profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        Returns
        -------
        f : ndarray
            Rotational transform profile errors.

        """
        i = i_l[self._idx]
        return self._shift_scale(i)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "i_l"


class FixPsi(_Objective):
    """Fixes total toroidal magnetic flux within the last closed flux surface.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, optional
        Target value(s) of the objective. If None, uses Equilibrium value.
    weight : float, optional
        Weighting to apply to the Objective, relative to other Objectives.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = True
    _fixed = True

    def __init__(self, eq=None, target=None, weight=1, name="fixed-Psi"):

        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Fixed-Psi error: {:10.3e} (Wb)"

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

        if None in self.target:
            self.target = eq.Psi

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

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
