import numpy as np
from termcolor import colored
import warnings
from collections import MutableSequence
from desc.utils import Timer
from desc.configuration import _Configuration, format_boundary, format_profiles
from desc.io import IOAble
from desc.boundary_conditions import BoundaryConstraint
from desc.objective_funs import ObjectiveFunction, get_objective_function
from desc.optimize import Optimizer
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.transform import Transform
from desc.perturbations import perturb


class Equilibrium(_Configuration, IOAble):
    """Equilibrium is an object that represents a plasma equilibrium. It
    contains information about a plasma state, including the
    shapes of flux surfaces and profile inputs. It can compute additional
    information, such as the magnetic field and plasma currents, as well as
    "solving" itself by finding the equilibrium fields, and perturbing those fields
    to find nearby equilibria.


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
        * ``'boundary'`` : ndarray, array of boundary coeffs [m, n, Rb_mn, Zb_mn]

        And the following optional keys:

        * ``'sym'`` : bool, is the problem stellarator symmetric or not, default is False
        * ``'spectral_indexing'`` : str, type of Zernike indexing scheme to use, default is 'ansi'
        * ``'bdry_mode'`` : str, how to calculate error at bdry, default is 'spectral'
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

    load_from : str file path OR file instance
        file to initialize from
    file_format : str
        file format of file initializing from. Default is 'hdf5'


    """

    # TODO: make this ^ format correctly with sphinx, dont show it as init method

    # TODO: add optimizer, objective, transform to io_attrs
    # and figure out why it wont save
    _io_attrs_ = _Configuration._io_attrs_ + [
        "_solved",
        "_x0",
        "_M_grid",
        "_N_grid",
        "_grid",
        "_node_pattern",
        # "_transforms", # TODO: figure out why we can't save dictionaries
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
        }
    )

    def __init__(
        self,
        inputs=None,
        load_from=None,
        file_format="hdf5",
        obj_lib=None,
    ):

        super().__init__(
            inputs=inputs, load_from=load_from, file_format=file_format, obj_lib=obj_lib
        )

    def _init_from_inputs_(self, inputs=None):
        super()._init_from_inputs_(inputs=inputs)
        self._x0 = self._x
        self._M_grid = inputs.get("M_grid", self._M)
        self._N_grid = inputs.get("N_grid", self._N)
        self._spectral_indexing = inputs.get("spectral_indexing", "fringe")
        self._node_pattern = inputs.get("node_pattern", "quad")
        self.optimizer_results = {}
        self._solved = False
        self._transforms = {}
        self._objective = None
        self._optimizer = None
        self.timer = Timer()
        self._set_grid()
        self._set_transforms()
        self.objective = inputs.get("objective", None)
        self.optimizer = inputs.get("optimizer", None)

    @property
    def x0(self):
        """ndarray : initial optimization vector (before solution)"""
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = x0

    @property
    def M_grid(self):
        """int : poloidal/radial resolution in real space"""
        return self._M_grid

    @M_grid.setter
    def M_grid(self, new: Grid):
        if self._M_grid != new:
            self._M_grid = new
            self._set_grid()
            self._set_transforms()

    @property
    def N_grid(self):
        """int : toroidal resolution in real space"""
        return self._N_grid

    @N_grid.setter
    def N_grid(self, new: Grid):
        if self._N_grid != new:
            self._N_grid = new
            self._set_grid()
            self._set_transforms()

    def _set_grid(self):
        if self._node_pattern in ["cheb1", "cheb2", "jacobi"]:
            self._grid = ConcentricGrid(
                M=self._M_grid,
                N=self._N_grid,
                NFP=self._NFP,
                sym=self._sym,
                axis=False,
                spectral_indexing=self._spectral_indexing,
                node_pattern=self._node_pattern,
            )
        elif self._node_pattern in ["linear", "uniform"]:
            self._grid = LinearGrid(
                L=2 * self._M_grid + 1,
                M=2 * self._M_grid + 1,
                N=2 * self._N_grid + 1,
                NFP=self._NFP,
                sym=self._sym,
                axis=False,
            )
        elif self._node_pattern in ["quad"]:
            self._grid = QuadratureGrid(
                L=np.ceil((self._L + 1) / 2),
                M=2 * self._M_grid + 1,
                N=2 * self._N_grid + 1,
                NFP=self._NFP,
                sym=self._sym,
            )
        else:
            raise ValueError(
                colored("unknown grid type {}".format(self._node_pattern), "red")
            )

    def _set_transforms(self):

        if len(self._transforms) == 0:
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
            self._transforms["R"].change_resolution(
                self.grid, self.R_basis, build=False
            )
            self._transforms["Z"].change_resolution(
                self.grid, self.Z_basis, build=False
            )
            self._transforms["L"].change_resolution(
                self.grid, self.L_basis, build=False
            )
            self._transforms["Rb"].change_resolution(
                self.grid, self.Rb_basis, build=False
            )
            self._transforms["Zb"].change_resolution(
                self.grid, self.Zb_basis, build=False
            )
            self._transforms["p"].change_resolution(
                self.grid, self.p_basis, build=False
            )
            self._transforms["i"].change_resolution(
                self.grid, self.i_basis, build=False
            )
        if self.objective is not None:
            derivs = self.objective.derivatives
            self._transforms["R"].change_derivatives(derivs, build=False)
            self._transforms["Z"].change_derivatives(derivs, build=False)
            self._transforms["L"].change_derivatives(derivs, build=False)

    def build(self, verbose=1):
        """Builds transform matrices and factorizes boundary constraint

        Parameters
        ----------
        verbose : int
            level of output
        """

        self.timer.start("Transform computation")
        if verbose > 0:
            print("Precomputing Transforms")
        self._set_transforms()
        for tr in self._transforms.values():
            tr.build()

        self.timer.stop("Transform computation")
        if verbose > 1:
            self.timer.disp("Transform computation")

        self.timer.start("Boundary constraint factorization")
        if verbose > 0:
            print("Factorizing boundary constraint")
        if self.objective is not None and self.objective.BC_constraint is not None:
            self.objective.BC_constraint.build()
        self.timer.stop("Boundary constraint factorization")
        if verbose > 1:
            self.timer.disp("Boundary constraint factorization")

    def change_resolution(self, L=None, M=None, N=None, M_grid=None, N_grid=None):
        """Set the spectral and real space resolution

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
        if L is not None and L != self._L:
            L_change = True
        if M is not None and M != self._M:
            M_change = True
        if N is not None and N != self._N:
            N_change = True

        if any([L_change, M_change, N_change]):
            super().change_resolution(L, M, N)

        M_grid_change = N_grid_change = False
        if M_grid is not None and M_grid != self._M_grid:
            self._M_grid = M_grid
            M_grid_change = True
        if N_grid is not None and N_grid != self._N_grid:
            self._N_grid = N_grid
            N_grid_change = True
        if any([M_grid_change, N_grid_change]):
            self._set_grid()
        self._set_transforms()
        if (
            any([L_change, M_change, N_change, M_grid_change, N_grid_change])
            and self.objective is not None
        ):
            self.objective = self.objective.name

    @property
    def built(self):
        """bool : whether the equilibrium is ready to solve"""
        tr = np.all([tr.built for tr in self._transforms.values()])
        if self.objective is not None and self.objective.BC_constraint is not None:
            bc = self.objective.BC_constraint.built
        else:
            bc = True
        return tr and bc

    @property
    def grid(self) -> Grid:
        """Grid : the grid of real space collocation nodes"""
        return self._grid

    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, Grid):
            raise ValueError("grid attribute must be of type 'Grid' or a subclass")
        self._grid = grid
        self._set_transforms()

    @property
    def solved(self):
        """bool : whether the equilibrium has been solved"""
        return self._solved

    @solved.setter
    def solved(self, solved):
        self._solved = solved

    @property
    def objective(self):
        """ObjectiveFunction : the objective function currently assigned"""
        return self._objective

    @objective.setter
    def objective(self, objective):
        if isinstance(objective, ObjectiveFunction) or objective is None:
            self._objective = objective
        elif isinstance(objective, str):
            self._objective = get_objective_function(
                objective,
                R_transform=self._transforms["R"],
                Z_transform=self._transforms["Z"],
                L_transform=self._transforms["L"],
                Rb_transform=self._transforms["Rb"],
                Zb_transform=self._transforms["Zb"],
                p_transform=self._transforms["p"],
                i_transform=self._transforms["i"],
                BC_constraint=BoundaryConstraint(
                    self.R_basis,
                    self.Z_basis,
                    self.L_basis,
                    self.Rb_basis,
                    self.Zb_basis,
                    self.Rb_mn,
                    self.Zb_mn,
                    build=False,
                ),
            )
        else:
            raise ValueError(
                "objective should be of type 'ObjectiveFunction' or string"
            )
        self.solved = False
        self.optimizer_results = {}

    @property
    def optimizer(self):
        """Optimizer : the optimizer currently assigned"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if isinstance(optimizer, Optimizer) or optimizer is None:
            self._optimizer = optimizer
        elif optimizer in Optimizer._all_methods:
            self._optimizer = Optimizer(optimizer, self.objective)
        else:
            raise ValueError("Invalid optimizer {}".format(optimizer))

    @property
    def initial(self):
        """Equilibrium : initial Equilibrium from which the Equilibrium was solved"""

        p_modes = np.array(
            [self._p_basis.modes[:, 0], self._p_l, np.zeros_like(self._p_l)]
        ).T
        i_modes = np.array(
            [self._i_basis.modes[:, 0], np.zeros_like(self._i_l), self._i_l]
        ).T
        Rb_mn = self._Rb_mn.reshape((-1, 1))
        Zb_mn = self._Zb_mn.reshape((-1, 1))
        Rb_modes = np.hstack([self._Rb_basis.modes[:, 1:], Rb_mn, np.zeros_like(Rb_mn)])
        Zb_modes = np.hstack([self._Zb_basis.modes[:, 1:], np.zeros_like(Zb_mn), Zb_mn])
        inputs = {
            "sym": self._sym,
            "NFP": self._NFP,
            "Psi": self._Psi,
            "L": self._L,
            "M": self._M,
            "N": self._N,
            "spectral_indexing": self._spectral_indexing,
            "bdry_mode": self._bdry_mode,
            "zeta_ratio": self._zeta_ratio,
            "profiles": np.vstack((p_modes, i_modes)),
            "boundary": np.vstack((Rb_modes, Zb_modes)),
            "x": self._x0,
        }
        return Equilibrium(inputs=inputs)

    def evaluate(self):
        """Evaluates the objective function.

        Returns
        -------
        f : ndarray or float
            function value
        jac : ndarray
            derivative df/dx

        """
        y = self.objective.BC_constraint.project(self.x)
        f = self._objective.compute(
            y,
            self.Rb_mn,
            self.Zb_mn,
            self.p_l,
            self.i_l,
            self.Psi,
        )
        jac = self._objective.jac_x(
            y,
            self.Rb_mn,
            self.Zb_mn,
            self.p_l,
            self.i_l,
            self.Psi,
        )
        return f, jac

    def resolution_summary(self):
        """Prints a summary of the spectral and real space resolution"""

        print("Spectral indexing: {}".format(self._spectral_indexing))
        print("Spectral resolution (L,M,N)=({},{},{})".format(self.L, self.M, self.N))
        print("Node pattern: {}".format(self._node_pattern))
        print("Node resolution (M,N)=({},{})".format(self._M_grid, self._N_grid))

    def solve(
        self,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=1,
        x_scale="auto",
        maxiter=100,
        options={},
    ):
        """Solve to find the equilibrium configuration

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

        if self._optimizer is None:
            raise AttributeError(
                "Equilibrium must have objective and optimizer defined before solving."
            )

        args = (
            self.Rb_mn,
            self.Zb_mn,
            self.p_l,
            self.i_l,
            self.Psi,
            self._zeta_ratio,
        )
        if verbose > 0:
            print("Starting optimization")

        self.x0 = self.x
        x_init = self._optimizer.objective.BC_constraint.project(self.x)
        self.timer.start("Solution time")

        result = self._optimizer.optimize(
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
        self.timer.stop("Solution time")

        if verbose > 1:
            self.timer.disp("Solution time")
            self.timer.pretty_print(
                "Avg time per step",
                self.timer["Solution time"] / result["nfev"],
            )
        if verbose > 0:
            print("Start of solver")
            self._objective.callback(x_init, *args)
            print("End of solver")
            self._objective.callback(result["x"], *args)

        self.optimizer_results = {key: val for key, val in result.items()}
        self.x = self._optimizer.objective.BC_constraint.recover(result["x"])
        self.solved = result["success"]
        return result

    def perturb(
        self, deltas, order, tr_ratio=0.1, cutoff=1e-6, Jx=None, verbose=1, copy=True
    ):
        """Perturb the equilibrium while maintaining equilibrium

        Parameters
        ----------
        deltas : dict
            dictionary of ndarray of objective function parameters to perturb.
            Allowed keys are: 'Rb_mn', 'Zb_mn', 'p_l', 'i_l', 'Psi', 'zeta_ratio'
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
        equil = perturb(
            self,
            deltas,
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
        """Optimize an equilibrium for a physics or engineering objective"""
        raise NotImplementedError("optimizing equilibria has not yet been implemented")


class EquilibriaFamily(IOAble, MutableSequence):
    """EquilibriaFamily stores a list of Equilibria and has methods for solving
    for complex equilibria using a multi-grid continuation method

    Parameters
    ----------
    inputs : dict or list
        either a dictionary of inputs or list of dictionaries. For more information
        see inputs required by ``'Equilibrium'``.
        If solving using continuation method, a list should be given.
    load_from : str file path OR file instance
        file to initialize from
    file_format : str
        file format of file initializing from. Default is 'hdf5'

    """

    _io_attrs_ = ["equilibria"]
    _object_lib_ = Equilibrium._object_lib_
    _object_lib_.update({"Equilibrium": Equilibrium})

    def __init__(self, inputs=None, load_from=None, file_format="hdf5", obj_lib=None):
        self.timer = Timer()
        self._file_format_ = file_format
        if load_from is None:
            self._init_from_inputs_(inputs=inputs)
        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )

    def _init_from_inputs_(self, inputs=None):
        # did we get 1 set of inputs or several?
        if isinstance(inputs, (list, tuple)):
            self._equilibria = [Equilibrium(inputs[0])]
        else:
            self._equilibria = [Equilibrium(inputs=inputs)]
        self.inputs = inputs

    @staticmethod
    def _format_deltas(inputs, inputs_prev, equil):
        """Figures out the changes in continuation parameters

        Parameters
        ----------
        inputs, inputs_prev : dict
            dictionaries of continuation parameters for current and  previous step
        equil : Equilibrium
            equilibrium being perturbed

        Returns
        -------
        deltas : dict
            dictionary of delta values to be passed to equil.perturb"""
        deltas = {}
        Rb_mn, Zb_mn = format_boundary(
            inputs["boundary"],
            equil.Rb_basis,
            equil.Zb_basis,
            equil.bdry_mode,
        )
        if not np.allclose(Rb_mn, equil.Rb_mn):
            deltas["Rb_mn"] = Rb_mn - equil.Rb_mn
        if not np.allclose(Zb_mn, equil.Zb_mn):
            deltas["Zb_mn"] = Zb_mn - equil.Zb_mn
        p_l, i_l = format_profiles(inputs["profiles"], equil.p_basis, equil.i_basis)
        if not np.allclose(p_l, equil.p_l):
            deltas["p_l"] = p_l - equil.p_l
        if not np.allclose(i_l, equil.i_l):
            deltas["i_l"] = i_l - equil.i_l
        if not np.allclose(inputs["zeta_ratio"], inputs_prev["zeta_ratio"]):
            deltas["zeta_ratio"] = inputs["zeta_ratio"] - inputs_prev["zeta_ratio"]
        if not np.allclose(inputs["Psi"], inputs_prev["Psi"]):
            deltas["Psi"] = inputs["Psi"] - inputs_prev["Psi"]
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
        print("Objective: {}".format(equil.objective.name))
        print("Optimizer: {}".format(equil.optimizer.method))
        print("Function tolerance = {}".format(self.inputs[ii]["ftol"]))
        print("Gradient tolerance = {}".format(self.inputs[ii]["gtol"]))
        print("State vector tolerance = {}".format(self.inputs[ii]["xtol"]))
        print("Max function evaluations = {}".format(self.inputs[ii]["nfev"]))
        print("================")

    def solve_continuation(
        self, start_from=0, verbose=None, checkpoint_path=None, device=None
    ):
        """Solves for an equilibrium by continuation method

            1. Creates an initial guess from the given inputs
            2. Find equilibrium flux surfaces by minimizing the given objective function.
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
        device : jax.device or None
            device handle to JIT compile to (Default value = None)

        """
        if verbose is None:
            verbose = self.inputs[0]["verbose"]
        self.timer.start("Total time")

        for ii in range(start_from, len(self.inputs)):
            self.timer.start("Iteration {} total".format(ii + 1))
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
                # TODO: updating transforms instead of recomputing
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
                        deltas,
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

            objective = get_objective_function(
                self.inputs[ii]["objective"],
                R_transform=equil._transforms["R"],
                Z_transform=equil._transforms["Z"],
                L_transform=equil._transforms["L"],
                Rb_transform=equil._transforms["Rb"],
                Zb_transform=equil._transforms["Zb"],
                p_transform=equil._transforms["p"],
                i_transform=equil._transforms["i"],
                BC_constraint=BoundaryConstraint(
                    equil.R_basis,
                    equil.Z_basis,
                    equil.L_basis,
                    equil.Rb_basis,
                    equil.Zb_basis,
                    equil.Rb_mn,
                    equil.Zb_mn,
                    build=False,
                ),
                use_jit=True,
                devices=device,
            )

            equil.objective = objective
            equil.optimizer = self.inputs[ii]["optimizer"]
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
            self.timer.stop("Iteration {} total".format(ii + 1))
            if verbose > 1:
                self.timer.disp("Iteration {} total".format(ii + 1))

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

        self.timer.stop("Total time")
        print("====================")
        print("Done")
        if verbose > 1:
            self.timer.disp("Total time")
        if checkpoint_path is not None:
            print("Output written to {}".format(checkpoint_path))
        print("====================")

    @property
    def equilibria(self):
        """list : list of equilibria contained in the family"""
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
                "Members of EquilibriaFamily should be of type Equilibrium or a subclass"
            )
        self._equilibria = list(equil)

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self._equilibria[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or a subclass"
            )
        self._equilibria[i] = new_item

    def __delitem__(self, i):
        del self._equilibria[i]

    def __len__(self):
        return len(self._equilibria)

    def insert(self, i, new_item):
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or a subclass"
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
