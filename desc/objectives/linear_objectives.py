import numpy as np
from termcolor import colored
import warnings

from desc.backend import jnp
from desc.utils import Timer
from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile
from desc.compute import compute_rotational_transform
from .objective_funs import _Objective


"""Linear objective functions must be of the form `A*x-b`, where:
    - `A` is a constant matrix that can be pre-computed
    - `x` is a vector of one or more arguments included in `compute.arg_order`
    - `b` is the desired vector set by `objective.target`
"""


class FixedBoundaryR(_Objective):
    """Fixes boundary R coefficients."""

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        surface=None,
        modes=True,
    ):
        """Initialize a FixedBoundaryR Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective. len(target) = len(weight) = len(modes).
            If None, uses surface coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(target) = len(weight) = len(modes)
        surface : Surface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        modes : ndarray, optional
            Basis modes numbers [l,m,n] of boundary modes to fix.
            len(target) = len(weight) = len(modes).
            If True/False uses all/none of the surface modes.

        """
        self._surface = surface
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight)

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
        if (
            self._surface is None
            or (
                self._surface.L,
                self._surface.M,
                self._surface.N,
            )
            != (eq.surface.L, eq.surface.M, eq.surface.N)
        ):
            self._surface = eq.surface

        # find indicies of R boundary modes to fix
        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            self._idx = np.array([], dtype=int)
            idx = self._idx
        elif self._modes is True:  # all modes in surface
            modes = self._surface.R_basis.modes
            self._idx = np.arange(self._surface.R_basis.num_modes)
            idx = self._idx
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, self._idx, idx = np.intersect1d(
                self._surface.R_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            if self._idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the boundary surface, "
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
        # use surface coefficients as target if needed
        if None in self.target or self.target.size != self.dim_f:
            self.target = self._surface.R_lmn[self._idx]

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, Rb_lmn):
        Rb = Rb_lmn[self._idx] - self.target
        return Rb

    def compute(self, Rb_lmn, **kwargs):
        """Compute fixed-boundary R errors.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate (m).

        Returns
        -------
        f : ndarray
            Boundary surface errors (m).

        """
        Rb = self._compute(Rb_lmn)
        return Rb * self.weight

    def callback(self, Rb_lmn, **kwargs):
        """Print fixed-boundary R errors.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate (m).

        """
        Rb = self._compute(Rb_lmn)
        print("R fixed-boundary error: {:10.3e} (m)".format(jnp.linalg.norm(Rb)))
        return None

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Rb_lmn"

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "fixed-boundary R"


class FixedBoundaryZ(_Objective):
    """Fixes boundary Z coefficients."""

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        surface=None,
        modes=True,
    ):
        """Initialize a FixedBoundaryZ Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective. len(target) = len(weight) = len(modes).
            If None, uses surface coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(target) = len(weight) = len(modes)
        surface : Surface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        modes : ndarray, optional
            Basis modes numbers [l,m,n] of boundary modes to fix.
            len(target) = len(weight) = len(modes).
            If True/False uses all/none of the surface modes.

        """
        self._surface = surface
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight)

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
        if (
            self._surface is None
            or (
                self._surface.L,
                self._surface.M,
                self._surface.N,
            )
            != (eq.surface.L, eq.surface.M, eq.surface.N)
        ):
            self._surface = eq.surface

        # find indicies of Z boundary modes to fix
        if self._modes is False or self._modes is None:  # no modes
            modes = np.array([[]], dtype=int)
            self._idx = np.array([], dtype=int)
            idx = self._idx
        elif self._modes is True:  # all modes in surface
            modes = self._surface.Z_basis.modes
            self._idx = np.arange(self._surface.Z_basis.num_modes)
            idx = self._idx
        else:  # specified modes
            modes = np.atleast_2d(self._modes)
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [modes.dtype],
            }
            _, self._idx, idx = np.intersect1d(
                self._surface.Z_basis.modes.astype(modes.dtype).view(dtype),
                modes.view(dtype),
                return_indices=True,
            )
            if self._idx.size < modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the boundary surface, "
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
        # use surface coefficients as target if needed
        if None in self.target or self.target.size != self.dim_f:
            self.target = self._surface.Z_lmn[self._idx]

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, Zb_lmn):
        Zb = Zb_lmn[self._idx] - self.target
        return Zb

    def compute(self, Zb_lmn, **kwargs):
        """Compute fixed-boundary Z errors.

        Parameters
        ----------
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordinate (m).

        Returns
        -------
        f : ndarray
            Boundary surface errors (m).

        """
        Zb = self._compute(Zb_lmn)
        return Zb * self.weight

    def callback(self, Zb_lmn, **kwargs):
        """Print fixed-boundary Z errors.

        Parameters
        ----------
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordinate (m).

        """
        Zb = self._compute(Zb_lmn)
        print("Z fixed-boundary error: {:10.3e} (m)".format(jnp.linalg.norm(Zb)))
        return None

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Zb_lmn"

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "fixed-boundary Z"


class FixedPressure(_Objective):
    """Fixes pressure coefficients."""

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        profile=None,
        modes=True,
    ):
        """Initialize a FixedPressure Objective.

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

        """
        self._profile = profile
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight)

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
        if not isinstance(self._profile, PowerSeriesProfile):
            raise NotImplementedError("profile must be of type `PowerSeriesProfile`")
            # TODO: add implementation for SplineProfile & MTanhProfile

        # find inidies of pressure modes to fix
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

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, p_l):
        return p_l[self._idx] - self.target

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
        return self._compute(p_l) * self.weight

    def callback(self, p_l, **kwargs):
        """Print fixed-pressure profile errors.

        Parameters
        ----------
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.

        """
        f = self._compute(p_l)
        print("Fixed-pressure profile error: {:10.3e} (Pa)".format(jnp.linalg.norm(f)))
        return None

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "p_l"

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "fixed-pressure"


class FixedIota(_Objective):
    """Fixes rotational transform coefficients."""

    def __init__(
        self,
        eq=None,
        target=None,
        weight=1,
        profile=None,
        modes=True,
    ):
        """Initialize a FixedIota Objective.

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

        """
        self._profile = profile
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight)

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
        if not isinstance(self._profile, PowerSeriesProfile):
            raise NotImplementedError("profile must be of type `PowerSeriesProfile`")
            # TODO: add implementation for SplineProfile & MTanhProfile

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

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, i_l):
        return i_l[self._idx] - self.target

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
        return self._compute(i_l) * self.weight

    def callback(self, i_l, **kwargs):
        """Print fixed-iota profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        """
        f = self._compute(i_l)
        print("Fixed-iota profile error: {:10.3e}".format(jnp.linalg.norm(f)))
        return None

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "i_l"

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "fixed-iota"


class FixedPsi(_Objective):
    """Fixes total toroidal magnetic flux within the last closed flux surface."""

    def __init__(self, eq=None, target=None, weight=1):
        """Initialize a FixedIota Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, optional
            Target value(s) of the objective. If None, uses Equilibrium value.
        weight : float, optional
            Weighting to apply to the Objective, relative to other Objectives.

        """
        super().__init__(eq=eq, target=target, weight=weight)

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

    def _compute(self, Psi):
        return Psi - self.target

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
        return self._compute(Psi) * self.weight

    def callback(self, Psi, **kwargs):
        """Print fixed-Psi error.

        Parameters
        ----------
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(Psi)
        print("Fixed-Psi error: {:10.3e} (Wb)".format(f[0]))
        return None

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Psi"

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "fixed-Psi"


class LCFSBoundary(_Objective):
    """Boundary condition on the last closed flux surface."""

    def __init__(self, eq=None, target=0, weight=1, surface=None):
        """Initialize a LCFSBoundary Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            This target always gets overridden to be 0.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        surface : FourierRZToroidalSurface, optional
            Toroidal surface containing the Fourier modes to evaluate at.

        """
        target = 0
        self._surface = surface
        super().__init__(eq=eq, target=target, weight=weight)

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
        if (
            self._surface is None
            or (
                self._surface.L,
                self._surface.M,
                self._surface.N,
            )
            != (eq.surface.L, eq.surface.M, eq.surface.N)
        ):
            self._surface = eq.surface

        R_modes = eq.R_basis.modes
        Z_modes = eq.Z_basis.modes
        Rb_modes = self._surface.R_basis.modes
        Zb_modes = self._surface.Z_basis.modes

        dim_R = eq.R_basis.num_modes
        dim_Z = eq.Z_basis.num_modes
        dim_Rb = self._surface.R_basis.num_modes
        dim_Zb = self._surface.Z_basis.num_modes
        self._dim_f = dim_Rb + dim_Zb

        self._A_R = np.zeros((dim_Rb, dim_R))
        self._A_Z = np.zeros((dim_Zb, dim_Z))

        for i, (l, m, n) in enumerate(R_modes):
            j = np.argwhere(np.logical_and(Rb_modes[:, 1] == m, Rb_modes[:, 2] == n))
            self._A_R[j, i] = 1

        for i, (l, m, n) in enumerate(Z_modes):
            j = np.argwhere(np.logical_and(Zb_modes[:, 1] == m, Zb_modes[:, 2] == n))
            self._A_Z[j, i] = 1

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn):
        Rb = jnp.dot(self._A_R, R_lmn) - Rb_lmn
        Zb = jnp.dot(self._A_Z, Z_lmn) - Zb_lmn
        return Rb, Zb

    def compute(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn, **kwargs):
        """Compute last closed flux surface boundary errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate (m).
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordinate (m).

        Returns
        -------
        f : ndarray
            Boundary surface errors (m).

        """
        Rb, Zb = self._compute(R_lmn, Z_lmn, Rb_lmn, Zb_lmn)
        return jnp.concatenate((Rb, Zb)) * self.weight

    def callback(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn, **kwargs):
        """Print last closed flux surface boundary errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate (m).
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordinate (m).

        """
        Rb, Zb = self._compute(R_lmn, Z_lmn, Rb_lmn, Zb_lmn)
        print("R boundary error: {:10.3e} (m)".format(jnp.linalg.norm(Rb)))
        print("Z boundary error: {:10.3e} (m)".format(jnp.linalg.norm(Zb)))
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "lcfs"


class TargetIota(_Objective):
    """Targets a rotational transform profile."""

    def __init__(self, eq=None, target=0, weight=1, profile=None, grid=None):
        """Initialize a TargetIota Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : tuple, float, ndarray, optional
            Target value(s) of the objective.
            len(target) = len(weight) = len(modes).
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(target) = len(weight) = len(modes)
        profile : Profile, optional
            Profile containing the radial modes to evaluate at.
        grid : Grid, optional
            Collocation grid containing the nodes to evaluate at.

        """
        self._profile = profile
        self._grid = grid
        super().__init__(eq=eq, target=target, weight=weight)

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
            self._profile = eq.iota.copy()
        if self._grid is None:
            self._grid = LinearGrid(L=2, NFP=eq.NFP, axis=True, rho=[0, 1])

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profile.grid = self._grid

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, i_l):
        data = compute_rotational_transform(i_l, self._profile)
        return data["iota"] - self.target

    def compute(self, i_l, **kwargs):
        """Compute rotational transform profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        Returns
        -------
        f : ndarray
            Rotational transform profile errors.

        """
        return self._compute(i_l) * self.weight

    def callback(self, i_l, **kwargs):
        """Print rotational transform profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        """
        f = self._compute(i_l)
        print("Target-iota profile error: {:10.3e}".format(jnp.linalg.norm(f)))
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "target-iota"


# XXX: this is a hack
class VMECBoundaryConstraint(_Objective):
    """Constraint to fix a boundary mode in the VMEC double-Fourier basis."""

    def __init__(self, eq=None, target=0, weight=1, mode=(0, 0)):
        """Initialize a VMECBoundaryConstraint Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            target = [RBC, ZBS], RBC*cos(m*t - n*NFP*z), ZBS*sin(m*t - n*NFP*z)
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        mode : tuple of int
            mode = (m, n)

        """
        self._m = mode[0]
        self._n = mode[1]
        super().__init__(eq=eq, target=target, weight=weight)

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
        Rb_modes = eq.surface.R_basis.modes
        Zb_modes = eq.surface.Z_basis.modes

        dim_Rb = eq.surface.R_basis.num_modes
        dim_Zb = eq.surface.Z_basis.num_modes

        self._dim_f = 2

        self._A_R = np.zeros((1, dim_Rb))
        j = np.argwhere(
            np.logical_and(
                Rb_modes[:, 1] == np.abs(self._m), Rb_modes[:, 2] == np.abs(self._n)
            )
        )
        self._A_R[0, j] = 0.5
        j = np.argwhere(
            np.logical_and(
                Rb_modes[:, 1] == -np.abs(self._m), Rb_modes[:, 2] == -np.abs(self._n)
            )
        )
        self._A_R[0, j] = -0.5

        self._A_Z = np.zeros((1, dim_Zb))
        j = np.argwhere(
            np.logical_and(
                Zb_modes[:, 1] == -np.abs(self._m), Zb_modes[:, 2] == np.abs(self._n)
            )
        )
        self._A_Z[0, j] = 0.5
        j = np.argwhere(
            np.logical_and(
                Zb_modes[:, 1] == np.abs(self._m), Zb_modes[:, 2] == -np.abs(self._n)
            )
        )
        self._A_Z[0, j] = 0.5

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, Rb_lmn, Zb_lmn):
        Rb = jnp.dot(self._A_R, Rb_lmn)
        Zb = jnp.dot(self._A_Z, Zb_lmn)
        return jnp.concatenate((Rb, Zb)) - self.target

    def compute(self, Rb_lmn, Zb_lmn, **kwargs):
        """Compute VMEC boundary constraint errors.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate (m).
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordinate (m).

        Returns
        -------
        f : ndarray
            Boundary surface errors (m).

        """
        return self._compute(Rb_lmn, Zb_lmn) * self.weight

    def callback(self, Rb_lmn, Zb_lmn, **kwargs):
        """Print last closed flux surface boundary errors.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate (m).
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordinate (m).

        """
        f = self._compute(Rb_lmn, Zb_lmn)
        print("R constraint error: {:10.3e} (m)".format(f[0]))
        print("Z constraint error: {:10.3e} (m)".format(f[1]))
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "VMEC"
