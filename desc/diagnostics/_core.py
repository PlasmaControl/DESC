"""Core classes for computing diagnostic measurements."""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import numpy as np

from desc.backend import jnp
from desc.coils import flatten_list
from desc.io import IOAble
from desc.optimizable import Optimizable, OptimizableCollection


class AbstractDiagnostic(IOAble, Optimizable, ABC):
    """Base class for virtual diagnostics.

    The init of this class should accept the necessary geometric and grids
    for computing the diagnostic (e.g. the geometry of a rogowski coil
    and the grid used to discretize the integration of B over it, and
    the grid to be used to discretize the external coilset and equilibrium
    virtual casing principle)

    Subclasses of this class must implement:

    a `compute` method which accepts an Equilibrium and MagneticField object
    and computes and returns the diagnostic measurement.

    a `_compute` method, which should
    compute the same values as the compute method, but _compute should accept
    the total A or B field at the necessary evaluation points as its only input,
    and so should not compute B. _compute can also accept a dictionary of some
    auxiliary data.
    Note that `_compute` can only be called if the diagnostic is built first.

    Subclasses must also define a ``self._all_eval_x_rpz`` attribute in their build
    method, which contains every R,phi,Z point at which the total A or B must be
    evaluated at. These are the points at which _compute should expect the A or B
    passed in to be computed at.
    ``self._dim_f`` should also be
    defined in their build method, which is the number of values returned when
    evaluating the diagnostic (i.e. one for a single flux loop,
    n for n point B measurements)

    """

    def __init__(self, *, name=""):
        self._name = name

    def build(self):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        assert hasattr(self, "_all_eval_x_rpz"), (
            "Subclasses of AbstractDiagnostic must define self._all_eval_x_rpz in"
            "  their build method!"
        )
        assert hasattr(self, "_dim_f"), (
            "Subclasses of AbstractDiagnostic must define self._dim_f in"
            "  their build method!"
        )

    @abstractmethod
    def compute_normalization(self, scales):
        """Compute normalization for diagnostic from input scales dictionary."""

    @abstractmethod
    def _compute_data(self, eq_params, field_params, diag_params=None, constants=None):
        """Compute necessary auxiliary data for diagnostic from params."""

    @abstractmethod
    def _compute(self, A_or_B, data):
        """Compute magnetic diagnostic signal using passed-in total field and data."""

    @property
    def name(self):
        """Name of diagnostic (str)."""
        return self.__dict__.setdefault("_name", "")


class DiagnosticSet(OptimizableCollection, AbstractDiagnostic, MutableSequence):
    """A collection of diagnostics.

    Parameters
    ----------
    diagnostics : list of AbstractDiagnostic
        List of diagnostic objects to include in the set. May include other
        DiagnosticSet objects.
    name : str, optional
        Name of the diagnostic set.

    """

    _static_attrs = OptimizableCollection._static_attrs + ["_name"]

    def __init__(self, *diagnostics, name="DiagnosticSet"):
        diagnostics = flatten_list(diagnostics, flatten_tuple=True)
        assert all([isinstance(diag, AbstractDiagnostic) for diag in diagnostics])
        self._diagnostics = list(diagnostics)
        super().__init__(name=name)

    @property
    def diagnostics(self):
        """list: diagnostics in the diagnostic set."""
        return self._diagnostics

    def _make_arraylike(self, x):
        if isinstance(x, dict):
            x = [x] * len(self)
        try:
            len(x)
        except TypeError:
            x = [x] * len(self)
        assert len(x) == len(self)
        return x

    def build(self, verbose=1):
        """Build all diagnostics in the set.

        Parameters
        ----------
        verbose : int, optional
            Level of output.

        """
        for diag in self._diagnostics:
            diag.build(verbose=verbose)
        # Now that all diag are built, all should have their _all_eval_x_rpz
        # set. Collect those now
        self._all_eval_x_rpz = np.vstack(
            [diag._all_eval_x_rpz for diag in self._diagnostics]
        )
        # make the indices of the overall eval_x_rpz array corresponding
        # to each sub-objective
        self._eval_x_idxs = [np.arange(self._diagnostics[0]._all_eval_x_rpz.shape[0])]
        for i in range(1, len(self._diagnostics)):
            self._eval_x_idxs.append(
                np.arange(self._diagnostics[i]._all_eval_x_rpz.shape[0])
                + self._eval_x_idxs[i - 1][-1]
                + 1
            )
        # dim_f is number of signals we have across all diags
        self._dim_f = np.sum([diag._dim_f for diag in self._diagnostics])

    def compute_normalization(self, scales):
        """Compute normalization for diagnostic from input scales dictionary."""
        return np.hstack(
            [
                diag.compute_normalization(scales) * np.ones(diag._dim_f)
                for diag in self._diagnostics
            ]
        )

    def _compute_data(self, eq_params, field_params, diag_params=None, constants=None):
        """Compute necessary auxiliary data for diagnostic from params.

        returns a pytree of data dictionaries, corresponding to the structure of
        this DiagnosticSet.

        """
        if constants is None:
            constants = self.constants
        if diag_params is None:
            diag_params = self.params_dict
        data = [
            diag._compute_data(
                eq_params, field_params, diag_params=diag_params[i], constants=constants
            )
            for i, diag in enumerate(self._diagnostics)
        ]

        return data

    def _compute(self, B, data=None):
        """Compute all diagnostics in the set using passed-in total field and data list.

        B should be the total magnetic field in cylindrical basis evaluated at
        self._all_eval_x_rpz cylindrical points in real space.

        data list should be a list of data dictionaries, corresponding to the pytree
        structure of this DiagnosticSet.

        """
        results = [
            diag._compute(B[idx, :], d)
            for diag, d, idx in zip(self._diagnostics, data, self._eval_x_idxs)
        ]
        return jnp.concatenate(results)

    def __add__(self, other):
        if isinstance(other, (DiagnosticSet)):
            return DiagnosticSet(*self._diagnostics, *other._diagnostics)
        if isinstance(other, (list, tuple)):
            return DiagnosticSet(*self._diagnostics, *other)
        else:
            return NotImplemented

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self._diagnostics[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, AbstractDiagnostic):
            raise TypeError(
                "Members of DiagnosticSet must be of type AbstractDiagnostic."
            )
        # TODO: if making like CoilSet, need _check_type(new_item, self[0])
        self._diagnostics[i] = new_item

    def __delitem__(self, i):
        del self._diagnostics[i]

    def __len__(self):
        return len(self._diagnostics)

    def insert(self, i, new_item):
        """Insert a new diagnostic into the diagnostic set at position i."""
        if not isinstance(new_item, AbstractDiagnostic):
            raise TypeError(
                "Members of DiagnosticSet must be of type AbstractDiagnostic."
            )
        self._diagnostics.insert(i, new_item)

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, with {} submembers)".format(self.name, len(self))
        )
