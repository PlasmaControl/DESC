"""Base classes for optimizable objects."""

import inspect
import warnings
from abc import ABC

import numpy as np

from desc.backend import jnp


class Optimizable(ABC):
    """Base class for all objects in DESC that can be optimized.

    Sub-classes should decorate optimizable attributes with the
    ``optimizable_parameter`` decorator.
    """

    _static_attrs = ["_optimizable_params"]

    @property
    def optimizable_params(self):
        """list: string names of parameters that have been declared optimizable."""
        if not hasattr(self, "_optimizable_params"):
            p = []
            for methodname in dir(self):
                if methodname.startswith("__"):
                    continue
                # to avoid executing code and causing recursion
                method = inspect.getattr_static(self, methodname)
                if isinstance(method, property):
                    method = method.fget  # we want the property itself, not the value
                if hasattr(method, "optimizable"):
                    p.append(methodname)
            self._optimizable_params = self._sort_args(p)
            if not len(p):
                warnings.warn(
                    f"Object {self} was subclassed from Optimizable but no "
                    + "optimizable parameters were declared"
                )
        return self._optimizable_params

    @property
    def params_dict(self):
        """dict: dictionary of arrays of optimizable parameters."""
        return {
            key: jnp.atleast_1d(jnp.asarray(getattr(self, key))).copy()
            for key in self.optimizable_params
        }

    @params_dict.setter
    def params_dict(self, d):
        for key, val in d.items():
            if jnp.asarray(val).size:
                setattr(self, key, val)

    @property
    def dimensions(self):
        """dict: dictionary of integers of sizes of each optimizable parameter."""
        return {
            key: jnp.asarray(getattr(self, key)).size for key in self.optimizable_params
        }

    @property
    def x_idx(self):
        """dict: arrays of indices for each parameter in concatenated array."""
        dimensions = self.dimensions
        idx = {}
        dim_x = 0
        for arg in self.optimizable_params:
            idx[arg] = jnp.arange(dim_x, dim_x + dimensions[arg])
            dim_x += dimensions[arg]
        return idx

    @property
    def dim_x(self):
        """int: total number of optimizable parameters."""
        return sum(self.dimensions.values())

    def pack_params(self, p):
        """Convert a dictionary of parameters into a single array.

        Parameters
        ----------
        p : dict
            Dictionary of ndarray of optimizable parameters.

        Returns
        -------
        x : ndarray
            optimizable parameters concatenated into a single array, with indices
            given by ``x_idx``

        """
        return jnp.concatenate(
            [jnp.atleast_1d(jnp.asarray(p[key])) for key in self.optimizable_params]
        )

    def unpack_params(self, x):
        """Convert a single array of concatenated parameters into a dictionary.

        Parameters
        ----------
        x : ndarray
            optimizable parameters concatenated into a single array, with indices
            given by ``x_idx``

        Returns
        -------
        p : dict
            Dictionary of ndarray of optimizable parameters.

        """
        x_idx = self.x_idx
        params = {}
        for arg in self.optimizable_params:
            params[arg] = jnp.atleast_1d(jnp.asarray(x[x_idx[arg]]))
        return params

    def _sort_args(self, args):
        """Put arguments in a canonical order. Returns unique sorted elements.

        Actual order doesn't really matter as long as its consistent, though subclasses
        may override this method to enforce a specific ordering.

        """
        return sorted(set(list(args)))


class OptimizableCollection(Optimizable):
    """Base class for collections of multiple optimizable objects (coilsets, etc).

    Subclasses should be iterable, where each member is itself Optimizable.
    """

    @property
    def optimizable_params(self):
        """list: string names of parameters that have been declared optimizable."""
        return [s.optimizable_params for s in self]

    @property
    def params_dict(self):
        """list: list of dictionary of arrays of optimizable parameters."""
        return [s.params_dict for s in self]

    @params_dict.setter
    def params_dict(self, d):
        for s, p in zip(self, d):
            s.params_dict = p

    @property
    def dimensions(self):
        """list: list of dictionary of integers of sizes of each parameter."""
        return [s.dimensions for s in self]

    @property
    def x_idx(self):
        """list: list of dict of arrays of idx for each param in concatenated array."""
        x_idx = [s.x_idx for s in self]
        offset = jnp.concatenate(
            [jnp.array([0]), jnp.cumsum(jnp.array([s.dim_x for s in self]))[:-1]]
        )
        for d, idx in zip(offset, x_idx):
            # offset subsequent indices by length of priors
            for key in idx:
                idx[key] += d
        return x_idx

    @property
    def dim_x(self):
        """int: total number of optimizable parameters."""
        return sum(s.dim_x for s in self)

    def pack_params(self, params):
        """Convert a list of dictionary of parameters into a single array.

        Parameters
        ----------
        params : list of dict
            list of dictionary of ndarray of optimizable parameters.

        Returns
        -------
        x : ndarray
            optimizable parameters concatenated into a single array, with indices
            given by ``x_idx``

        """
        return jnp.concatenate([s.pack_params(p) for s, p in zip(self, params)])

    def unpack_params(self, x):
        """Convert a single array of concatenated parameters into a dictionary.

        Parameters
        ----------
        x : ndarray
            optimizable parameters concatenated into a single array, with indices
            given by ``x_idx``

        Returns
        -------
        p : list dict
            list of dictionary of ndarray of optimizable parameters.

        """
        split_idx = np.cumsum([s.dim_x for s in self])  # must be np not jnp
        xs = jnp.split(x, split_idx)
        params = [s.unpack_params(xi) for s, xi in zip(self, xs)]
        return params


def optimizable_parameter(f):
    """Decorator to declare an attribute or property as optimizable.

    The attribute should be a scalar or ndarray of floats.

    Examples
    --------
    .. code-block:: python

        class MyClass(Optimizable):

            def __init__(self, x, y):
                self.x = x
                self.y = optimizable_parameter(y)

            @optimizable_parameter
            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, new):
                assert len(x) == 10
                self._x = x

    """
    if isinstance(f, property):
        f.fget.optimizable = True
    else:
        f.optimizable = True
    return f
