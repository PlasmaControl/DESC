"""Base classes for optimizeable objects."""

import inspect

from desc.backend import jnp


class Optimizeable:
    """Base class for all objects in DESC that can be optimized.

    Sub-classes should decocate optimizeable attributes with the
    ``optimizeable_parameter`` decorator.
    """

    def __init__(self):
        self._optimizeable_params = []
        p = []
        for methodname in dir(self):
            if methodname.startswith("__"):
                continue
            method = inspect.getattr_static(self, methodname)
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, "optimizeable"):
                p.append(methodname)
        self._optimizeable_params = p

    @property
    def optimizeable_params(self):
        """list: names of parameters that have been declared optimizeable."""
        return self._optimizeable_params

    @property
    def params_dict(self):
        """dict: dictionary of arrays of optimizeable parameters."""
        return {key: getattr(self, key) for key in self.optimizeable_params}

    @params_dict.setter
    def params_dict(self, d):
        for key, val in d.items():
            setattr(self, key, val)

    @property
    def dimensions(self):
        """dict: dictionary of integers of sizes of each optimizeable parameter."""
        return {
            key: jnp.asarray(getattr(self, key)).size
            for key in self.optimizeable_params
        }

    @property
    def x_idx(self):
        """dict: dictionary of arrays of indices into array for each parameter."""
        dimensions = self.dimensions
        idx = {}
        dim_x = 0
        for arg in sorted(self.optimizeable_params):
            idx[arg] = jnp.arange(dim_x, dim_x + dimensions[arg])
            dim_x += dimensions[arg]
        return idx

    def pack_params(self, p):
        """Convert a dictionary of parameters into a single array.

        Parameters
        ----------
        p : dict
            Dictionary of ndarray of optimizeable parameters.

        Returns
        -------
        x : ndarray
            optimizeable parameters concatenated into a single array, with indices
            given by ``x_idx``
        """
        return jnp.concatenate(
            [jnp.atleast_1d(p[key]) for key in sorted(self.optimizeable_params)]
        )

    def unpack_params(self, x):
        """Convert a single array of concatenated parameters into a dictionary.

        Parameters
        ----------
        x : ndarray
            optimizeable parameters concatenated into a single array, with indices
            given by ``x_idx``

        Returns
        -------
        p : dict
            Dictionary of ndarray of optimizeable parameters.
        """
        x_idx = self.x_idx
        params = {}
        for arg in self.optimizeable_params:
            params[arg] = jnp.atleast_1d(x[x_idx[arg]])
        return params


def optimizeable_parameter(f):
    """Decorator to declare an attribute as optimizeable."""
    if isinstance(f, property):
        f.fget.optimizeable = True
    else:
        f.optimizeable = True
    return f
