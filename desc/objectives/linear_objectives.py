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


class LCFSBoundaryR(_Objective):
    """Boundary condition on the last closed flux surface."""

    _scalar = False
    _linear = True
    _fixed = False  # TODO: can we dynamically detect this instead?

    def __init__(self, eq=None, target=None, weight=1, surface=None, name="lcfs R"):
        """Initialize a LCFSBoundary Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Boundary surface coefficients to fix. If None, uses surface coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        surface : FourierRZToroidalSurface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        name : str
            Name of the objective function.

        """
        self._surface = surface
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "R boundary error: {:10.3e} (m)"

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
        if self._surface is None:
            self._surface = eq.surface

        R_modes = eq.R_basis.modes
        Rb_modes = self._surface.R_basis.modes

        dim_R = eq.R_basis.num_modes
        self._dim_f = self._surface.R_basis.num_modes

        self._A = np.zeros((self._dim_f, dim_R))
        for i, (l, m, n) in enumerate(R_modes):
            j = np.argwhere(np.logical_and(Rb_modes[:, 1] == m, Rb_modes[:, 2] == n))
            self._A[j, i] = 1

        if None in self.target:
            self.target = self._surface.R_lmn

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, **kwargs):
        """Compute last closed flux surface boundary errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).

        Returns
        -------
        f : ndarray
            Boundary surface errors (m).

        """
        Rb_lmn = jnp.dot(self._A, R_lmn)
        return self._shift_scale(Rb_lmn)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Rb_lmn"


class LCFSBoundaryZ(_Objective):
    """Boundary condition on the last closed flux surface."""

    _scalar = False
    _linear = True
    _fixed = False

    def __init__(self, eq=None, target=None, weight=1, surface=None, name="lcfs Z"):
        """Initialize a LCFSBoundary Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Boundary surface coefficients to fix. If None, uses surface coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        surface : FourierRZToroidalSurface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        name : str
            Name of the objective function.

        """
        self._surface = surface
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Z boundary error: {:10.3e} (m)"

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
        if self._surface is None:
            self._surface = eq.surface

        Z_modes = eq.Z_basis.modes
        Zb_modes = self._surface.Z_basis.modes

        dim_Z = eq.Z_basis.num_modes
        self._dim_f = self._surface.Z_basis.num_modes

        self._A = np.zeros((self._dim_f, dim_Z))
        for i, (l, m, n) in enumerate(Z_modes):
            j = np.argwhere(np.logical_and(Zb_modes[:, 1] == m, Zb_modes[:, 2] == n))
            self._A[j, i] = 1

        if None in self.target:
            self.target = self._surface.Z_lmn

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, Z_lmn, **kwargs):
        """Compute last closed flux surface boundary errors.

        Parameters
        ----------
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        f : ndarray
            Boundary surface errors (m).

        """
        Zb_lmn = jnp.dot(self._A, Z_lmn)
        return self._shift_scale(Zb_lmn)

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return "Zb_lmn"


class LambdaGauge(_Objective):
    """Fixes gauge freedom for lambda: lambda(rho=0)=0 and lambda(theta=0,zeta=0)=0."""

    _scalar = False
    _linear = True
    _fixed = False

    def __init__(self, eq=None, target=0, weight=1, name="lambda gauge"):
        """Initialize a LambdaGauge Objective.

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
            self._A = np.zeros((L_basis.N, L_basis.num_modes))
            ns = np.arange(-L_basis.N, 1)
            for i, (l, m, n) in enumerate(L_basis.modes):
                if m != 0:
                    continue
                if (l // 2) % 2 == 0:
                    j = np.argwhere(n == ns)
                    self._A[j, i] = 1
                else:
                    j = np.argwhere(n == ns)
                    self._A[j, i] = -1
        else:
            raise NotImplementedError("Lambda gauge freedom not implemented yet.")

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


class FixedPressure(_Objective):
    """Fixes pressure coefficients."""

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
        name : str
            Name of the objective function.

        """
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


class FixedIota(_Objective):
    """Fixes rotational transform coefficients."""

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
        name : str
            Name of the objective function.

        """
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


class FixedPsi(_Objective):
    """Fixes total toroidal magnetic flux within the last closed flux surface."""

    _scalar = True
    _linear = True
    _fixed = True

    def __init__(self, eq=None, target=None, weight=1, name="fixed-Psi"):
        """Initialize a FixedIota Objective.

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


class TargetIota(_Objective):
    """Targets a rotational transform profile."""

    _scalar = False
    _linear = True
    _fixed = False

    def __init__(
        self, eq=None, target=0, weight=1, profile=None, grid=None, name="target-iota"
    ):
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
        name : str
            Name of the objective function.

        """
        self._profile = profile
        self._grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Target-iota profile error: {:10.3e}"

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
        data = compute_rotational_transform(i_l, self._profile)
        return self._shift_scale(data["iota"])
