import numpy as np
from termcolor import colored
import warnings
from collections import MutableSequence
from desc.backend import use_jax
from desc.utils import Timer, isalmostequal
from desc.configuration import _Configuration, format_boundary, format_profiles
from desc.io import IOAble
from desc.boundary_conditions import (
    get_boundary_condition,
    BoundaryCondition,
    LCFSConstraint,
    PoincareConstraint,
)
from desc.objective_funs import (
    get_objective_function,
    ObjectiveFunction,
    ForceErrorNodes,
    ForceErrorGalerkin,
    EnergyVolIntegral,
)
from desc.optimize import Optimizer
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.transform import Transform
from desc.perturbations import perturb, optimal_perturb


class Equilibrium(_Configuration, IOAble):
    """Equilibrium is an object that represents a plasma equilibrium.

    It contains information about a plasma state, including the shapes of flux surfaces
    and profile inputs. It can compute additional information, such as the magnetic
    field and plasma currents, as well as "solving" itself by finding the equilibrium
    fields, and perturbing those fields to find nearby equilibria.

    Parameters
    ----------
    inputs : dict
        Dictionary of inputs with the following required keys:

        * ``'Psi'`` : float, total toroidal flux (in Webers) within LCFS
        * ``'NFP'`` : int, number of field periods
        * ``'L'`` : int, radial resolution
        * ``'M'`` : int, poloidal resolution
        * ``'N'`` : int, toroidal resolution
        * ``'profiles'`` : ndarray, array of profile coeffs [l, p_l, i_l]
        * ``'boundary'`` : ndarray, array of boundary coeffs [l, m, n, Rb_lmn, Zb_lmn]

        And the following optional keys:

        * ``'sym'`` : bool, is the problem stellarator symmetric or not, default is False
        * ``'spectral_indexing'`` : str, type of Zernike indexing scheme to use, default is ``'ansi'``
        * ``'bdry_mode'`` : str, how to calculate error at bdry, default is ``'spectral'``
        * ``'zeta_ratio'`` : float, Multiplier on the toroidal derivatives. Default = 1.0.
        * ``'axis'`` : ndarray, array of magnetic axis coeffs [n, R0_n, Z0_n]
        * ``'x'`` : ndarray, state vector [R_lmn, Z_lmn, L_lmn]
        * ``'R_lmn'`` : ndarray, spectral coefficients of R
        * ``'Z_lmn'`` : ndarray, spectral coefficients of Z
        * ``'L_lmn'`` : ndarray, spectral coefficients of lambda
        * ``'M_grid'`` : int, resolution of real space nodes in poloidal/radial direction
        * ``'N_grid'`` : int, resolution of real space nodes in toroidal direction
        * ``'node_pattern'`` : str, node pattern, default is "cheb1"
        * ``'objective'`` : str, mode for equilibrium solution
        * ``'optimizer'`` : str, optimizer to use
    """

    # TODO: make this ^ format correctly with sphinx, dont show it as init method

    _io_attrs_ = _Configuration._io_attrs_ + [
        "_solved",
        "_x0",
        "_M_grid",
        "_N_grid",
        "_grid",
        "_node_pattern",
        "_transforms",
        "_constraint",
        "_objective",
        "optimizer_results",
        "_optimizer",
    ]
    _object_lib_ = _Configuration._object_lib_
    _object_lib_.update(
        {
            "_Configuration": _Configuration,
            "Grid": Grid,
            "LinearGrid": LinearGrid,
            "ConcentricGrid": ConcentricGrid,
            "QuadratureGrid": QuadratureGrid,
            "Transform": Transform,
            "Optimizer": Optimizer,
            "ForceErrorNodes": ForceErrorNodes,
            "ForceErrorGalerkin": ForceErrorGalerkin,
            "EnergyVolIntegral": EnergyVolIntegral,
            "BoundaryCondition": BoundaryCondition,
            "LCFSConstraint": LCFSConstraint,
            "PoincareConstraint": PoincareConstraint,
        }
    )

    def __init__(self, inputs):

        super().__init__(inputs=inputs)
        self._x0 = self._x
        self._M_grid = inputs.get("M_grid", self._M)
        self._N_grid = inputs.get("N_grid", self._N)
        self._spectral_indexing = inputs.get("spectral_indexing", "fringe")
        self._node_pattern = inputs.get("node_pattern", "quad")
        self._solved = False
        self._constraint = None
        self._objective = None
        self._optimizer = None
        self._set_grid()
        self._transforms = {}
        self._set_transforms()
        self.constraint = inputs.get("bdry_mode", None)
        self.objective = inputs.get("objective", None)
        self.optimizer = inputs.get("optimizer", None)
        self.optimizer_results = {}

    @property
    def x0(self):
        """Return initial optimization vector before solution (ndarray)."""
        if not hasattr(self, "_x0"):
            self._x0 = None
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = x0

    @property
    def M_grid(self):
        """Poloidal resolution of grid in real space (int)."""
        if not hasattr(self, "_M_grid"):
            self._M_grid = 1
        return self._M_grid

    @M_grid.setter
    def M_grid(self, new):
        if self.M_grid != new:
            self._M_grid = new
            self._set_grid()
            self._set_transforms()

    @property
    def N_grid(self):
        """Toroidal resolution of grid in real space (int)."""
        if not hasattr(self, "_N_grid"):
            self._N_grid = 0
        return self._N_grid

    @N_grid.setter
    def N_grid(self, new):
        if self.N_grid != new:
            self._N_grid = new
            self._set_grid()
            self._set_transforms()

    @property
    def node_pattern(self):
        """Pattern for placement of nodes in curvilinear coordinates (str)."""
        if not hasattr(self, "_node_pattern"):
            self._node_pattern = None
        return self._node_pattern

    @property
    def transforms(self):
        if not hasattr(self, "_transforms"):
            self._transforms = {}
            self._set_transforms()
        return self._transforms

    def _set_grid(self):
        if self.node_pattern in ["cheb1", "cheb2", "jacobi"]:
            self._grid = ConcentricGrid(
                M=self.M_grid,
                N=self.N_grid,
                NFP=self.NFP,
                sym=self.sym,
                axis=False,
                spectral_indexing=self.spectral_indexing,
                node_pattern=self.node_pattern,
            )
        elif self.node_pattern in ["linear", "uniform"]:
            self._grid = LinearGrid(
                L=2 * self.M_grid + 1,
                M=2 * self.M_grid + 1,
                N=2 * self.N_grid + 1,
                NFP=self.NFP,
                sym=self.sym,
                axis=False,
            )
        elif self.node_pattern in ["quad"]:
            L_grid = (
                self.M_grid if self.spectral_indexing == "ansi" else 2 * self.M_grid
            )
            self._grid = QuadratureGrid(
                L=L_grid,
                M=self.M_grid,
                N=self.N_grid,
                NFP=self.NFP,
            )
        else:
            raise ValueError(
                colored("unknown grid type {}".format(self.node_pattern), "red")
            )

    def _set_transforms(self):

        if len(self.transforms) == 0:
            self._transforms["R"] = Transform(
                self.grid, self.R_basis, derivs=0, build=False
            )
            self._transforms["Z"] = Transform(
                self.grid, self.Z_basis, derivs=0, build=False
            )
            self._transforms["L"] = Transform(
                self.grid, self.L_basis, derivs=0, build=False
            )
            self._transforms["Rb"] = Transform(
                self.grid, self.Rb_basis, derivs=0, build=False
            )
            self._transforms["Zb"] = Transform(
                self.grid, self.Zb_basis, derivs=0, build=False
            )
            self._transforms["p"] = Transform(
                self.grid, self.p_basis, derivs=1, build=False
            )
            self._transforms["i"] = Transform(
                self.grid, self.i_basis, derivs=1, build=False
            )

        else:
            self.transforms["R"].change_resolution(self.grid, self.R_basis, build=False)
            self.transforms["Z"].change_resolution(self.grid, self.Z_basis, build=False)
            self.transforms["L"].change_resolution(self.grid, self.L_basis, build=False)
            self.transforms["Rb"].change_resolution(
                self.grid, self.Rb_basis, build=False
            )
            self.transforms["Zb"].change_resolution(
                self.grid, self.Zb_basis, build=False
            )
            self.transforms["p"].change_resolution(self.grid, self.p_basis, build=False)
            self.transforms["i"].change_resolution(self.grid, self.i_basis, build=False)
        if self.objective is not None:
            derivs = self.objective.derivatives
            self.transforms["R"].change_derivatives(derivs, build=False)
            self.transforms["Z"].change_derivatives(derivs, build=False)
            self.transforms["L"].change_derivatives(derivs, build=False)

    def build(self, verbose=1):
        """Build transform matrices and factorizes boundary constraint.

        Parameters
        ----------
        verbose : int
            level of output

        """
        timer = Timer()
        timer.start("Transform computation")
        if verbose > 0:
            print("Precomputing Transforms")
        self._set_transforms()
        for tr in self.transforms.values():
            tr.build()

        timer.stop("Transform computation")
        if verbose > 1:
            timer.disp("Transform computation")

        timer.start("Boundary constraint factorization")
        if verbose > 0:
            print("Factorizing boundary constraint")
        if self.objective is not None and self.objective.BC_constraint is not None:
            self.objective.BC_constraint.build()
        timer.stop("Boundary constraint factorization")
        if verbose > 1:
            timer.disp("Boundary constraint factorization")

    def change_resolution(self, L=None, M=None, N=None, M_grid=None, N_grid=None):
        """Set the spectral and real space resolution.

        Parameters
        ----------
        L : int
            maximum radial zernike mode number
        M : int
            maximum poloidal fourier mode number
        N : int
            maximum toroidal fourier mode number
        M_grid : int
            poloidal/radial real space resolution
        N_grid : int
            toroidal real space resolution

        """
        L_change = M_change = N_change = False
        if L is not None and L != self.L:
            L_change = True
        if M is not None and M != self.M:
            M_change = True
        if N is not None and N != self.N:
            N_change = True

        if any([L_change, M_change, N_change]):
            super().change_resolution(L, M, N)

        M_grid_change = N_grid_change = False
        if M_grid is not None and M_grid != self.M_grid:
            self._M_grid = M_grid
            M_grid_change = True
        if N_grid is not None and N_grid != self.N_grid:
            self._N_grid = N_grid
            N_grid_change = True
        if any([M_grid_change, N_grid_change]):
            self._set_grid()
        self._set_transforms()
        if (
            any([L_change, M_change, N_change, M_grid_change, N_grid_change])
            and self.objective is not None
        ):
            self.constraint = self.constraint.name
            self.objective = self.objective.name

    @property
    def built(self):
        """Whether the equilibrium is ready to solve (bool)."""
        tr = np.all([tr.built for tr in self.transforms.values()])
        if self.objective is not None and self.objective.BC_constraint is not None:
            bc = self.objective.BC_constraint.built
        else:
            bc = True
        return tr and bc

    @property
    def grid(self):
        """Grid of real space collocation nodes (Grid)."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, Grid):
            raise ValueError("grid attribute must be of type 'Grid' or a subclass")
        self._grid = grid
        self._set_transforms()

    @property
    def solved(self):
        """Whether the equilibrium has been solved (bool)."""
        return self._solved

    @solved.setter
    def solved(self, solved):
        self._solved = solved

    @property
    def constraint(self):
        """Boundary condition currently assigned (BoundaryCondition)."""
        if not hasattr(self, "_constraint"):
            self._constraint = None
        return self._constraint

    @constraint.setter
    def constraint(self, constraint):
        if constraint is None:
            self._constraint = constraint
        elif isinstance(constraint, BoundaryCondition) and constraint.eq(
            self.constraint
        ):
            return
        elif isinstance(constraint, BoundaryCondition) and not constraint.eq(
            self.constraint
        ):
            self._constraint = constraint
        elif isinstance(constraint, str):
            self._constraint = get_boundary_condition(
                constraint,
                R_basis=self.R_basis,
                Z_basis=self.Z_basis,
                L_basis=self.L_basis,
                Rb_basis=self.Rb_basis,
                Zb_basis=self.Zb_basis,
                Rb_lmn=self.Rb_lmn,
                Zb_lmn=self.Zb_lmn,
                build=False,
            )
        else:
            raise ValueError(
                "objective should be of type 'BoundaryCondition' or string, "
                + "got {}".format(constraint)
            )

    @property
    def objective(self):
        """Objective function currently assigned (ObjectiveFunction)."""
        if not hasattr(self, "_objective"):
            self._objective = None
        return self._objective

    @objective.setter
    def objective(self, objective):
        if objective is None:
            self._objective = objective
        elif isinstance(objective, ObjectiveFunction) and objective.eq(self.objective):
            return
        elif isinstance(objective, ObjectiveFunction) and not objective.eq(
            self.objective
        ):
            self._objective = objective
        elif isinstance(objective, str):
            self._set_transforms()
            self._objective = get_objective_function(
                objective,
                R_transform=self.transforms["R"],
                Z_transform=self.transforms["Z"],
                L_transform=self.transforms["L"],
                Rb_transform=self.transforms["Rb"],
                Zb_transform=self.transforms["Zb"],
                p_transform=self.transforms["p"],
                i_transform=self.transforms["i"],
                BC_constraint=self.constraint,
            )
        else:
            raise ValueError(
                "objective should be of type 'ObjectiveFunction' or string, "
                + "got {}".format(objective)
            )
        self.solved = False
        self.optimizer_results = {}

    @property
    def optimizer(self):
        """Optimizer currently assigned (Optimizer)."""
        if not hasattr(self, "_optimizer"):
            self._optimizer = None
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if optimizer is None:
            self._optimizer = optimizer
        elif isinstance(optimizer, Optimizer) and optimizer == self.optimizer:
            return
        elif isinstance(optimizer, Optimizer) and optimizer != self.optimizer:
            self._optimizer = optimizer
        elif optimizer in Optimizer._all_methods:
            self._optimizer = Optimizer(optimizer)
        else:
            raise ValueError(
                "optimizer should be of type Optimizer or str, got  {}".format(
                    optimizer
                )
            )

    @property
    def initial(self):
        """Return initial equilibrium state from which it was solved (Equilibrium)."""
        p_modes = np.array(
            [self.p_basis.modes[:, 0], self.p_l, np.zeros_like(self.p_l)]
        ).T
        i_modes = np.array(
            [self.i_basis.modes[:, 0], np.zeros_like(self.i_l), self.i_l]
        ).T
        Rb_lmn = self.Rb_lmn.reshape((-1, 1))
        Zb_lmn = self.Zb_lmn.reshape((-1, 1))
        Rb_modes = np.hstack([self.Rb_basis.modes, Rb_lmn, np.zeros_like(Rb_lmn)])
        Zb_modes = np.hstack([self.Zb_basis.modes, np.zeros_like(Zb_lmn), Zb_lmn])
        inputs = {
            "sym": self.sym,
            "NFP": self.NFP,
            "Psi": self.Psi,
            "L": self.L,
            "M": self.M,
            "N": self.N,
            "spectral_indexing": self.spectral_indexing,
            "bdry_mode": self.bdry_mode,
            "zeta_ratio": self.zeta_ratio,
            "profiles": np.vstack((p_modes, i_modes)),
            "boundary": np.vstack((Rb_modes, Zb_modes)),
            "x": self.x0,
            "objective": self.objective.name,
            "optimizer": self.optimizer.method,
        }
        return Equilibrium(inputs=inputs)

    def evaluate(self, jac=False):
        """Evaluate the objective function.

        Parameters
        ----------
        jac : bool
            whether to compute and return the jacobian df/dx as well

        Returns
        -------
        f : ndarray or float
            function value
        jac : ndarray
            derivative df/dx

        """
        if self.objective is None:
            raise AttributeError(
                "Equilibrium must have objective defined before evaluating."
            )

        y = self.objective.BC_constraint.project(self.x)
        f = self.objective.compute(
            y, self.Rb_lmn, self.Zb_lmn, self.p_l, self.i_l, self.Psi
        )
        if jac:
            jac = self.objective.jac_x(
                y, self.Rb_lmn, self.Zb_lmn, self.p_l, self.i_l, self.Psi
            )
            return f, jac
        else:
            return f

    def resolution_summary(self):
        """Print a summary of the spectral and real space resolution."""
        print("Spectral indexing: {}".format(self.spectral_indexing))
        print("Spectral resolution (L,M,N)=({},{},{})".format(self.L, self.M, self.N))
        print("Node pattern: {}".format(self.node_pattern))
        print("Node resolution (M,N)=({},{})".format(self.M_grid, self.N_grid))

    def solve(
        self,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=1,
        x_scale="auto",
        maxiter=None,
        options={},
    ):
        """Solve to find the equilibrium configuration.

        Parameters
        ----------
        ftol : float
            relative stopping tolerance on objective function value
        xtol : float
            stopping tolerance on step size
        gtol : float
            stopping tolerance on norm of gradient
        verbose : int
            level of output
        maxiter : int
            maximum number of solver steps
        options : dict
            dictionary of additional options to pass to optimizer

        """
        if self.optimizer is None:
            raise AttributeError(
                "Equilibrium must have objective and optimizer defined before solving."
            )

        args = (self.Rb_lmn, self.Zb_lmn, self.p_l, self.i_l, self.Psi, self.zeta_ratio)

        self.x0 = self.x
        x_init = self.objective.BC_constraint.project(self.x)

        result = self.optimizer.optimize(
            self.objective,
            x_init=x_init,
            args=args,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            verbose=verbose,
            maxiter=maxiter,
            options=options,
        )

        if verbose > 0:
            print("Start of solver")
            self.objective.callback(x_init, *args)
            print("End of solver")
            self.objective.callback(result["x"], *args)

        self.optimizer_results = {
            key: val if isinstance(val, str) else np.asarray(val)
            for key, val in result.items()
        }
        self.x = self.objective.BC_constraint.recover(result["x"])
        self.solved = result["success"]
        return result

    def perturb(
        self,
        objective=None,
        dRb=None,
        dZb=None,
        dp=None,
        di=None,
        dPsi=None,
        dzeta_ratio=None,
        order=2,
        tr_ratio=0.1,
        cutoff=1e-6,
        Jx=None,
        verbose=1,
        copy=True,
    ):
        """Perturb the configuration while mainting equilibrium.

        Parameters
        ----------
        objective : ObjectiveFunction
            objective to optimize during the perturbation (optional)
        dRb, dZb, dp, di, dPsi, dzeta_ratio : ndarray or float
            If objective not given: deltas for perturbations of
            R_boundary, Z_boundary, pressure, iota, toroidal flux, and zeta ratio.
            Setting to None or zero ignores that term in the expansion.
            If objective is given: indicies of modes to include in the perturbations of
            R_boundary, Z_boundary, pressure, iota, toroidal flux, and zeta ratio.
            Setting to True (False/None) includes (excludes) all modes.
        order : int, optional
            order of perturbation (0=none, 1=linear, 2=quadratic)
        Jx : ndarray, optional
            jacobian matrix df/dx
        verbose : int
            level of output to display
        copy : bool
            True to return a modified copy of the current equilibrium, False to perturb
            the current equilibrium in place

        Returns
        -------
        eq_new : Equilibrium
            perturbed equilibrum, only returned if copy=True

        """
        if objective is None:
            # perturb with the given input parameter deltas
            equil = perturb(
                self,
                dRb,
                dZb,
                dp,
                di,
                dPsi,
                dzeta_ratio,
                order=order,
                tr_ratio=tr_ratio,
                cutoff=cutoff,
                Jx=Jx,
                verbose=verbose,
                copy=copy,
            )
        else:
            equil = optimal_perturb(
                # find the deltas that optimize the objective, then perturb
                self,
                objective,
                dRb,
                dZb,
                dp,
                di,
                dPsi,
                dzeta_ratio,
                order=order,
                tr_ratio=tr_ratio,
                cutoff=cutoff,
                Jx=Jx,
                verbose=verbose,
                copy=copy,
            )

        equil.solved = False
        equil.optimizer_results = {}

        if copy:
            return equil
        else:
            return None

    def optimize(self):
        """Optimize an equilibrium for a physics or engineering objective."""
        raise NotImplementedError("Optimizing equilibria has not yet been implemented.")


class EquilibriaFamily(IOAble, MutableSequence):
    """EquilibriaFamily stores a list of Equilibria.

    Has methods for solving complex equilibria using a multi-grid continuation method.

    Parameters
    ----------
    inputs : dict or list
        either a dictionary of inputs or list of dictionaries. For more information
        see inputs required by ``'Equilibrium'``.
        If solving using continuation method, a list should be given.

    """

    _io_attrs_ = ["equilibria"]
    _object_lib_ = Equilibrium._object_lib_
    _object_lib_.update({"Equilibrium": Equilibrium})

    def __init__(self, inputs):
        # did we get 1 set of inputs or several?
        if isinstance(inputs, (list, tuple)):
            self.equilibria = [Equilibrium(inputs[0])]
        else:
            self.equilibria = [Equilibrium(inputs=inputs)]
        self.inputs = inputs

    @staticmethod
    def _format_deltas(inputs, inputs_prev, equil):
        """Format the changes in continuation parameters.

        Parameters
        ----------
        inputs, inputs_prev : dict
            dictionaries of continuation parameters for current and  previous step
        equil : Equilibrium
            equilibrium being perturbed

        Returns
        -------
        deltas : dict
            dictionary of delta values to be passed to equil.perturb

        """
        deltas = {}
        Rb_lmn, Zb_lmn = format_boundary(
            inputs["boundary"], equil.Rb_basis, equil.Zb_basis, equil.bdry_mode
        )
        p_l, i_l = format_profiles(inputs["profiles"], equil.p_basis, equil.i_basis)
        if not np.allclose(Rb_lmn, equil.Rb_lmn):
            deltas["dRb"] = Rb_lmn - equil.Rb_lmn
        if not np.allclose(Zb_lmn, equil.Zb_lmn):
            deltas["dZb"] = Zb_lmn - equil.Zb_lmn
        if not np.allclose(p_l, equil.p_l):
            deltas["dp"] = p_l - equil.p_l
        if not np.allclose(i_l, equil.i_l):
            deltas["di"] = i_l - equil.i_l
        if not np.allclose(inputs["Psi"], inputs_prev["Psi"]):
            deltas["dPsi"] = inputs["Psi"] - inputs_prev["Psi"]
        if not np.allclose(inputs["zeta_ratio"], inputs_prev["zeta_ratio"]):
            deltas["dzeta_ratio"] = inputs["zeta_ratio"] - inputs_prev["zeta_ratio"]
        return deltas

    def _print_iteration(self, ii, equil):
        print("================")
        print("Step {}/{}".format(ii + 1, len(self.inputs)))
        print("================")
        equil.resolution_summary()
        print("Boundary ratio = {}".format(self.inputs[ii]["bdry_ratio"]))
        print("Pressure ratio = {}".format(self.inputs[ii]["pres_ratio"]))
        print("Zeta ratio = {}".format(self.inputs[ii]["zeta_ratio"]))
        print("Perturbation Order = {}".format(self.inputs[ii]["pert_order"]))
        print("Constraint: {}".format(equil.constraint.name))
        print("Objective: {}".format(equil.objective.name))
        print("Optimizer: {}".format(equil.optimizer.method))
        print("Function tolerance = {}".format(self.inputs[ii]["ftol"]))
        print("Gradient tolerance = {}".format(self.inputs[ii]["gtol"]))
        print("State vector tolerance = {}".format(self.inputs[ii]["xtol"]))
        print("Max function evaluations = {}".format(self.inputs[ii]["nfev"]))
        print("================")

    def solve_continuation(self, start_from=0, verbose=None, checkpoint_path=None):
        """Solve for an equilibrium by continuation method.

            1. Creates an initial guess from the given inputs
            2. Find equilibrium flux surfaces by minimizing the objective function.
            3. Step up to higher resolution and perturb the previous solution
            4. Repeat 2 and 3 until at desired resolution

        Parameters
        ----------
        start_from : integer
            start solution from the given index
        verbose : integer
            * 0: no output
            * 1: summary of each iteration
            * 2: as above plus timing information
            * 3: as above plus detailed solver output
        checkpoint_path : str or path-like
            file to save checkpoint data (Default value = None)

        """
        timer = Timer()
        if verbose is None:
            verbose = self.inputs[0]["verbose"]
        timer.start("Total time")

        if (
            not (
                isalmostequal([inp["bdry_ratio"] for inp in self.inputs])
                and isalmostequal([inp["pres_ratio"] for inp in self.inputs])
                and isalmostequal([inp["zeta_ratio"] for inp in self.inputs])
            )
            and not use_jax
        ):
            warnings.warn(
                colored(
                    "Computing perturbations with finite differences can be "
                    + "highly innacurate, consider using JAX or setting all "
                    + "perturbation rations to 1",
                    "yellow",
                )
            )

        for ii in range(start_from, len(self.inputs)):
            timer.start("Iteration {} total".format(ii + 1))
            if ii == start_from:
                equil = self[ii]
                if verbose > 0:
                    self._print_iteration(ii, equil)

            else:
                equil = self[ii - 1].copy()
                self.insert(ii, equil)
                # this is basically free if nothings actually changing, so we can call
                # it on each iteration
                equil.change_resolution(
                    L=self.inputs[ii]["L"],
                    M=self.inputs[ii]["M"],
                    N=self.inputs[ii]["N"],
                    M_grid=self.inputs[ii]["M_grid"],
                    N_grid=self.inputs[ii]["N_grid"],
                )
                if verbose > 0:
                    self._print_iteration(ii, equil)

                # figure out if we we need perturbations
                deltas = self._format_deltas(
                    self.inputs[ii], self.inputs[ii - 1], equil
                )

                if len(deltas) > 0:
                    equil.build(verbose)
                    if verbose > 0:
                        print("Perturbing equilibrium")

                    equil.perturb(
                        **deltas,
                        order=self.inputs[ii]["pert_order"],
                        verbose=verbose,
                        copy=False,
                    )
            if not equil.is_nested():
                warnings.warn(
                    colored(
                        "WARNING: Flux surfaces are no longer nested, exiting early."
                        + "Consider taking smaller perturbation/resolution steps "
                        + "or reducing trust radius",
                        "yellow",
                    )
                )
                break

            # boundary condition
            constraint = get_boundary_condition(
                self.inputs[ii]["bdry_mode"],
                R_basis=equil.R_basis,
                Z_basis=equil.Z_basis,
                L_basis=equil.L_basis,
                Rb_basis=equil.Rb_basis,
                Zb_basis=equil.Zb_basis,
                Rb_lmn=equil.Rb_lmn,
                Zb_lmn=equil.Zb_lmn,
            )
            equil.constraint = constraint

            # objective function
            objective = get_objective_function(
                self.inputs[ii]["objective"],
                R_transform=equil.transforms["R"],
                Z_transform=equil.transforms["Z"],
                L_transform=equil.transforms["L"],
                Rb_transform=equil.transforms["Rb"],
                Zb_transform=equil.transforms["Zb"],
                p_transform=equil.transforms["p"],
                i_transform=equil.transforms["i"],
                BC_constraint=equil.constraint,
                use_jit=True,
            )
            # reuse old objective if possible to avoid recompiling
            if objective.eq(self[ii - 1].objective):
                equil.objective = self[ii - 1].objective
            else:
                equil.objective = objective

            # optimization algorithm
            optimizer = Optimizer(self.inputs[ii]["optimizer"])
            equil.optimizer = optimizer

            equil.build(verbose)
            equil.solve(
                ftol=self.inputs[ii]["ftol"],
                xtol=self.inputs[ii]["xtol"],
                gtol=self.inputs[ii]["gtol"],
                verbose=verbose,
                maxiter=self.inputs[ii]["nfev"],
            )

            if checkpoint_path is not None:
                if verbose > 0:
                    print("Saving latest iteration")
                self.save(checkpoint_path)
            timer.stop("Iteration {} total".format(ii + 1))
            if verbose > 1:
                timer.disp("Iteration {} total".format(ii + 1))

            if not equil.is_nested():
                warnings.warn(
                    colored(
                        "WARNING: Flux surfaces are no longer nested, exiting early."
                        + "Consider taking smaller perturbation/resolution steps "
                        + "or reducing trust radius",
                        "yellow",
                    )
                )
                break

        timer.stop("Total time")
        print("====================")
        print("Done")
        if verbose > 1:
            timer.disp("Total time")
        if checkpoint_path is not None:
            print("Output written to {}".format(checkpoint_path))
        print("====================")

    @property
    def equilibria(self):
        """List of equilibria contained in the family (list)."""
        return self._equilibria

    @equilibria.setter
    def equilibria(self, equil):
        if isinstance(equil, tuple):
            equil = list(equil)
        elif isinstance(equil, np.ndarray):
            equil = equil.tolist()
        elif not isinstance(equil, list):
            equil = [equil]
        if not np.all([isinstance(eq, Equilibrium) for eq in equil]):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria = list(equil)

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self._equilibria[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria[i] = new_item

    def __delitem__(self, i):
        del self._equilibria[i]

    def __len__(self):
        return len(self._equilibria)

    def insert(self, i, new_item):
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria.insert(i, new_item)

    def __slice__(self, idx):
        if idx is None:
            theslice = slice(None, None)
        elif isinstance(idx, int):
            theslice = idx
        elif isinstance(idx, list):
            try:
                theslice = slice(idx[0], idx[1], idx[2])
            except IndexError:
                theslice = slice(idx[0], idx[1])
        else:
            raise TypeError("index is not a valid type.")
        return theslice
