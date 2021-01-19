import numpy as np
from collections import MutableSequence
from desc.utils import Timer, expand_state
from desc.configuration import _Configuration
from desc.io import IOAble
from desc.boundary_conditions import BoundaryConstraint
from desc.objective_funs import ObjectiveFunction, get_objective_function
from desc.optimize import Optimizer
from desc.grid import ConcentricGrid, Grid
from desc.transform import Transform
from desc.perturbations import perturb


class Equilibrium(_Configuration, IOAble):
    """Equilibrium is a decorator design pattern on top of Configuration.
    It adds information about how the equilibrium configuration was solved.
    """

    # TODO: add optimizer, objective, grid, transform to io_attrs
    # and figure out why it wont save
    _io_attrs_ = _Configuration._io_attrs_ + ["_solved"]
    _object_lib_ = _Configuration._object_lib_
    _object_lib_.update(
        {"_Configuration": _Configuration, "ObjectiveFunction": ObjectiveFunction}
    )

    def __init__(
        self,
        inputs: dict = None,
        load_from=None,
        file_format: str = "hdf5",
        obj_lib=None,
    ) -> None:
        super().__init__(
            inputs=inputs, load_from=load_from, file_format=file_format, obj_lib=obj_lib
        )

    def _init_from_inputs_(self, inputs: dict = None) -> None:
        super()._init_from_inputs_(inputs=inputs)
        self._x0 = self._x
        self._M_grid = inputs.get("M_grid", self._M)
        self._N_grid = inputs.get("N_grid", self._N)
        self._zern_mode = inputs.get("zern_mode", "ansi")
        self._node_mode = inputs.get("node_mode", "cheb1")
        self.optimizer_results = {}
        self._solved = False
        self._transforms = {}
        self._objective = None
        self._optimizer = None
        self.timer = Timer()
        self._set_grid()
        self._set_transforms()
        self.objective = inputs.get("errr_mode", None)
        self.optimizer = inputs.get("optim_method", None)

    @classmethod
    def from_configuration(cls, configuration):
        """Create an Equilibrium from a Configuration"""
        return cls(configuration.inputs)

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0) -> None:
        self._x0 = x0

    @property
    def M_grid(self):
        return self._M_grid

    @M_grid.setter
    def M_grid(self, new):
        self._M_grid = new
        self._set_grid()
        self._set_transforms()

    @property
    def N_grid(self):
        return self._N_grid

    @N_grid.setter
    def N_grid(self, new):
        self._N_grid = new
        self._set_grid()
        self._set_transforms()

    def _set_grid(self):
        self._grid = ConcentricGrid(
            M=self.M_grid,
            N=self.N_grid,
            NFP=self.NFP,
            sym=self.sym,
            axis=False,
            index=self._zern_mode,
            surfs=self._node_mode,
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
        """Builds transform matrices"""

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
        super().change_resolution(L, M, N)
        if M_grid is not None:
            self.M_grid = M_grid
        if N_grid is not None:
            self.N_grid = N_grid
        self._set_transforms()

    @property
    def built(self):
        tr = np.all([tr.built for tr in self._transforms.values()])
        if self.objective is not None and self.objective.BC_constraint is not None:
            bc = self.objective.BC_constraint.built
        else:
            bc = True
        return tr and bc

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, Grid):
            raise ValueError("grid attribute must be of type 'Grid' or a subclass")
        self._grid = grid
        self._set_transforms()

    @property
    def solved(self) -> bool:
        return self._solved

    @solved.setter
    def solved(self, solved) -> None:
        self._solved = solved

    @property
    def objective(self):
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

    @property
    def optimizer(self):
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
        """
        Initial Equilibrium from which the Equilibrium was solved

        Returns
        -------
        Equilibrium

        """
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
            "index": self._index,
            "bdry_mode": self._bdry_mode,
            "zeta_ratio": self._zeta_ratio,
            "profiles": np.vstack((p_modes, i_modes)),
            "boundary": np.vstack((Rb_modes, Zb_modes)),
            "x": self._x0,
        }
        return Equilibrium(inputs=inputs)

    def compute(self):
        y = self._objective.BC_constraint.project(self._x)
        args = (
            y,
            self._Rb_mn,
            self._Zb_mn,
            self._p_l,
            self._i_l,
            self._Psi,
            self._zeta_ratio,
        )
        return self._objective.compute(*args)

    def solve(
        self,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=1,
        maxiter=None,
        x_scale="auto",
        options={},
    ):
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

        x_init = self._optimizer.objective.BC_constraint.project(self.x)
        self.timer.start("Solution time")

        result = self._optimizer.optimize(
            x_init=x_init,
            args=args,
            x_scale=x_scale,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=verbose,
            maxiter=maxiter,
            options=options,
        )
        self.timer.stop("Solution time")

        if verbose > 1:
            self.timer.disp("Solution time")
            self.timer.pretty_print(
                "Avg time per step", self.timer["Solution time"] / result["nfev"],
            )
        if verbose > 0:
            print("Start of solver")
            self._objective.callback(x_init, *args)
            print("End of solver")
            self._objective.callback(result["x"], *args)

        self.optimizer_results = result
        self.x = self._optimizer.objective.BC_constraint.recover(result["x"])
        self.solved = result["success"]
        return result

    def perturb(self, deltas, **kwargs):
        equil = perturb(self, deltas, **kwargs)
        equil._parent = self
        self._children.append(equil)
        return equil

    def optimize(self):
        raise NotImplementedError("optimizing equilibria has not yet been implemented")


class EquilibriaFamily(IOAble, MutableSequence):
    """EquilibriaFamily stores a list of Equilibria"""

    _io_attrs_ = ["equilibria"]
    _object_lib_ = Equilibrium._object_lib_
    _object_lib_.update({"Equilibrium": Equilibrium})

    def __init__(
        self, inputs=None, load_from=None, file_format="hdf5", obj_lib=None
    ) -> None:
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
            self._equilibria = [Equilibrium(inp) for inp in inputs]
        else:
            self._equilibria = [Equilibrium(inputs=inputs)]
        self.inputs = inputs

    def solve_continuation(
        self, start_from=0, verbose=None, checkpoint_path=None, device=None
    ):
        """Solves for an equilibrium by continuation method

        Follows this procedure to solve the equilibrium:
            1. Creates an initial guess from the given inputs
            2. Optimizes the equilibrium's flux surfaces by minimizing
                the given objective function.
            3. Step up to higher resolution and perturb the previous solution
            4. Repeat 2 and 3 until at desired resolution

        Parameters
        ----------
        start_from : integer
            start solution from the given index
        verbose : integer
            how much progress information to display
                0: no output
                1: summary of each iteration
                2: as above plus timing information
                3: as above plus detailed solver output
        checkpoint_path : str or path-like
            file to save checkpoint data (Default value = None)
        device : jax.device or None
            device handle to JIT compile to (Default value = None)
        """
        if verbose is None:
            verbose = self._inputs[0]["verbose"]
        self.timer = Timer()
        self.timer.start("Total time")

        for ii in range(start_from, len(self)):
            self.timer.start("Iteration {} total".format(ii + 1))
            equil = self[ii]
            if verbose > 0:
                print("================")
                print("Step {}/{}".format(ii + 1, len(self)))
                print("================")
                print(
                    "Spectral resolution (L,M,N)=({},{},{})".format(
                        equil.L, equil.M, equil.N
                    )
                )
                print(
                    "Node resolution (M,N)=({},{})".format(equil.M_grid, equil.N_grid)
                )
                print("Boundary ratio = {}".format(equil.inputs["bdry_ratio"]))
                print("Pressure ratio = {}".format(equil.inputs["pres_ratio"]))
                print("Zeta ratio = {}".format(equil.inputs["zeta_ratio"]))
                print("Perturbation Order = {}".format(equil.inputs["pert_order"]))
                print("Function tolerance = {}".format(equil.inputs["ftol"]))
                print("Gradient tolerance = {}".format(equil.inputs["gtol"]))
                print("State vector tolerance = {}".format(equil.inputs["xtol"]))
                print("Max function evaluations = {}".format(equil.inputs["nfev"]))
                print("================")

            if ii > start_from:
                # new initial guess is previous solution
                new_x = expand_state(
                    self[ii - 1].x,
                    self[ii - 1].R_basis.modes,
                    equil.R_basis.modes,
                    self[ii - 1].Z_basis.modes,
                    equil.Z_basis.modes,
                    self[ii - 1].L_basis.modes,
                    equil.L_basis.modes,
                )

                equil.x0 = equil.x = new_x
                equil._parent = self[ii - 1]
                self[ii - 1]._children.append(equil)
            # TODO: updating transforms instead of recomputing
            # TODO: check if params changed and do perturbations
            equil.build(verbose)
            if equil.objective is None:
                equil.objective = self.objective
            if equil.optimizer is None:
                equil.optimizer = self.objective

            equil.solve(
                ftol=equil.inputs["ftol"],
                xtol=equil.inputs["xtol"],
                gtol=equil.inputs["gtol"],
                verbose=verbose,
                maxiter=equil.inputs["nfev"],
            )

            if checkpoint_path is not None:
                if verbose > 0:
                    print("Saving latest iteration")
                self.save(checkpoint_path)
            self.timer.stop("Iteration {} total".format(ii + 1))
            if verbose > 1:
                self.timer.disp("Iteration {} total".format(ii + 1))

        self.timer.stop("Total time")
        print("====================")
        print("Done")
        if verbose > 1:
            self.timer.disp("Total time")
        if checkpoint_path is not None:
            print("Output written to {}".format(checkpoint_path))
        print("====================")

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver
        for eq in self._equilibria:
            eq.optimizer = solver

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        self._objective = objective
        for eq in self._equilibria:
            eq.objective = objective

    @property
    def equilibria(self):
        return self._equilibria

    @equilibria.setter
    def equilibria(self, equil):
        if not isinstance(equil, (list, tuple, np.ndarray)):
            equil = list(equil)
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
        elif type(idx) is int:
            theslice = idx
        elif type(idx) is list:
            try:
                theslice = slice(idx[0], idx[1], idx[2])
            except IndexError:
                theslice = slice(idx[0], idx[1])
        else:
            raise TypeError("index is not a valid type.")
        return theslice
