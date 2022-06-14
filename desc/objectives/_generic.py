from scipy.constants import mu_0
from inspect import signature

from desc.backend import jnp
from desc.utils import Timer
from desc.grid import QuadratureGrid, ConcentricGrid, LinearGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
import desc.compute as compute_funs
from desc.compute import (
    arg_order,
    data_index,
    compute_covariant_metric_coefficients,
    compute_magnetic_field_magnitude,
    compute_contravariant_current_density,
    compute_quasisymmetry_error,
)
from .objective_funs import _Objective


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`.

    Parameters
    ----------
    f : str
        Name of the quatity to compute.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False

    def __init__(self, f, eq=None, target=0, weight=1, grid=None, name="generic"):

        self.f = f
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Residual: {:10.3e} (" + data_index[self.f]["units"] + ")"

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
        if self.grid is None:
            self.grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)

        args = []
        self._dim_f = self.grid.num_nodes

        self.fun = getattr(compute_funs, data_index[self.f]["fun"])
        self.sig = signature(self.fun)
        self.inputs = {"data": None}

        for arg in self.sig.parameters.keys():
            if arg in arg_order:
                args.append(arg)
            elif arg == "R_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.R_basis,
                    derivs=data_index[self.f]["R_derivs"],
                    build=True,
                )
            elif arg == "Z_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.Z_basis,
                    derivs=data_index[self.f]["R_derivs"],
                    build=True,
                )
            elif arg == "L_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.L_basis,
                    derivs=data_index[self.f]["L_derivs"],
                    build=True,
                )
            elif arg == "B_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    DoubleFourierSeries(
                        M=2 * eq.M, N=2 * eq.N, sym=eq.R_basis.sym, NFP=eq.NFP
                    ),
                    derivs=0,
                    build_pinv=True,
                )
            elif arg == "w_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    DoubleFourierSeries(
                        M=2 * eq.M, N=2 * eq.N, sym=eq.Z_basis.sym, NFP=eq.NFP
                    ),
                    derivs=1,
                )
            elif arg == "pressure":
                self.inputs[arg] = eq.pressure.copy()
                self.inputs[arg].grid = self.grid
            elif arg == "iota":
                self.inputs[arg] = eq.iota.copy()
                self.inputs[arg].grid = self.grid

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._args = args
        self._built = True

    def compute(self, **kwargs):
        """Compute the quantity.

        Parameters
        ----------
        args : list of ndarray
            Any of the arguments given in `arg_order`.

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        data = self.fun(**kwargs, **self.inputs)
        f = data[self.f]
        return self._shift_scale(f)


class ToroidalCurrent(_Objective):
    """Toroidal current encolsed by a surface.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="toroidal current"):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Toroidal current: {:10.3e} (A)"

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
        if self.grid is None:
            self.grid = LinearGrid(
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=1,
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["I"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["I"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["I"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute toroidal current.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        I : float
            Toroidal current (A).

        """
        data = compute_quasisymmetry_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        I = 2 * jnp.pi / mu_0 * data["I"]
        return self._shift_scale(jnp.atleast_1d(I))
