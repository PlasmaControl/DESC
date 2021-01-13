import numpy as np
from collections import MutableSequence

from desc.utils import Timer
from desc.configuration import Configuration
from desc.io import IOAble
from desc.objective_funs import ObjectiveFunction, ObjectiveFunctionFactory
from desc.optimize import Optimizer



class Equilibrium(Configuration, IOAble):
    """Equilibrium is a decorator design pattern on top of Configuration.
    It adds information about how the equilibrium configuration was solved.
    """
    # TODO: add optimizer and objective to io_attrs and figure out why it wont save
    _io_attrs_ = Configuration._io_attrs_ + ["_solved"]
    _object_lib_ = Configuration._object_lib_
    _object_lib_.update(
        {"Configuration": Configuration, "ObjectiveFunction": ObjectiveFunction}
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
        self.objective = inputs.get("objective", None)
        self.optimizer = inputs.get("optimizer", None)
        self.optimizer_results = {}
        self._solved = False
        self.timer = Timer()

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0) -> None:
        self._x0 = x0

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
            # TODO: have equilibrium know about transforms etc.
            self._objective = ObjectiveFunctionFactory(objective)
        else:
            raise ValueError("Invalid objective {}".format(objective))
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
    def initial(self) -> Configuration:
        """
        Initial Configuration from which the Equilibrium was solved

        Returns
        -------
        Configuration

        """
        p_modes = np.array(
            [self._p_basis.modes[:, 0], self._p_l, np.zeros_like(self._p_l)]
        ).T
        i_modes = np.array(
            [self._i_basis.modes[:, 0], np.zeros_like(self._i_l), self._i_l]
        ).T
        R0_modes = np.array(
            [self._R0_basis.modes[:, 2], self._R0_n, np.zeros_like(self._R0_n)]
        ).T
        Z0_modes = np.array(
            [self._Z0_basis.modes[:, 2], np.zeros_like(self._R0_n), self._Z0_n]
        ).T
        R1_mn = self._R1_mn.reshape((-1, 1))
        Z1_mn = self._Z1_mn.reshape((-1, 1))
        R1_modes = np.hstack([self._R1_basis.modes[:, 1:], R1_mn, np.zeros_like(R1_mn)])
        Z1_modes = np.hstack([self._Z1_basis.modes[:, 1:], np.zeros_like(Z1_mn), Z1_mn])
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
            "axis": np.vstack((R0_modes, Z0_modes)),
            "boundary": np.vstack((R1_modes, Z1_modes)),
            "x": self._x0,
        }
        return Configuration(inputs=inputs)

    def optimize(self):
        pass

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
            self.R1_mn,
            self.Z1_mn,
            self.p_l,
            self.i_l,
            self.Psi,
            self._zeta_ratio,
        )
        if verbose > 0:
            print("Starting optimization")

        x_init = self._optimizer.objective.bc_constraint.project(self.x)
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
                "Avg time per step",
                self.timer["Solution time"] / result["nfev"],
            )
        if verbose > 0:
            print("Start of solver")
            self._objective.callback(x_init, *args)
            print("End of solver")
            self._objective.callback(result["x"], *args)

        self.optimizer_results = result
        self.x = self._optimizer.objective.bc_constraint.recover(result["x"])
        self.solved = result["success"]
        return result


# XXX: Should this (also) inherit from Equilibrium?
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
        self._equilibria = []
        self.append(Equilibrium(inputs=inputs))
        return None

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self._equilibria[i]

    def __setitem__(self, i, new_item):
        if isinstance(new_item, Configuration):
            self._equilibria[i] = new_item
        else:
            raise ValueError(
                "Members of EquilibriaFamily should be of type Configuration or a subclass"
            )

    def __delitem__(self, i):
        del self._equilibria[i]

    def __len__(self):
        return len(self._equilibria)

    def insert(self, i, new_item):
        self._equilibria.insert(i, new_item)

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    @property
    def equilibria(self):
        return self._equilibria

    @equilibria.setter
    def equilibria(self, eq):
        self._equilibria = eq

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


# TODO: overwrite all Equilibrium methods and default to self._equilibria[-1]

