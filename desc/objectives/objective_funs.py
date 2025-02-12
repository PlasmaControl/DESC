"""Base classes for objectives."""

import functools
from abc import ABC, abstractmethod

import numpy as np

from desc.backend import (
    desc_config,
    execute_on_cpu,
    jit,
    jnp,
    tree_flatten,
    tree_map,
    tree_unflatten,
    use_jax,
)
from desc.batching import batched_vectorize
from desc.derivatives import Derivative
from desc.io import IOAble
from desc.optimizable import Optimizable
from desc.utils import (
    PRINT_WIDTH,
    Timer,
    ensure_tuple,
    errorif,
    flatten_list,
    is_broadcastable,
    isposint,
    setdefault,
    unique_list,
    warnif,
)

doc_target = """
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if ``bounds`` is ``None``.
        Must be broadcastable to ``Objective.dim_f``.
"""
doc_bounds = """
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides ``target``.
        Both bounds must be broadcastable to ``Objective.dim_f``.
"""
doc_weight = """
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``.
"""
doc_normalize = """
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
"""
doc_normalize_target = """
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If ``normalize`` is ``True`` and the target is in physical units,
        this should also be set to ``True``.
"""
doc_loss_function = """
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
"""
doc_deriv_mode = """
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        ``auto`` selects forward or reverse mode based on the size of the input and
        output of the objective. Has no effect on ``self.grad`` or ``self.hess`` which
        always use reverse mode and forward over reverse mode respectively.
"""
doc_name = """
    name : str, optional
        Name of the objective.
"""
doc_jac_chunk_size = """
    jac_chunk_size : int or ``auto``, optional
        Will calculate the Jacobian
        ``jac_chunk_size`` columns at a time, instead of all at once.
        The memory usage of the Jacobian calculation is roughly
        ``memory usage = m0+m1*jac_chunk_size``: the smaller the chunk size,
        the less memory the Jacobian calculation will require (with some baseline
        memory usage). The time it takes to compute the Jacobian is roughly
        ``t = t0+t1/jac_chunk_size`` so the larger the ``jac_chunk_size``, the faster
        the calculation takes, at the cost of requiring more memory.
        If ``None``, it will use the largest size i.e ``obj.dim_x``.
        Can also help with Hessian computation memory, as Hessian is essentially
        ``jacfwd(jacrev(f))``, and each of these operations may be chunked.
        Defaults to ``chunk_size=None``.
        Note: When running on a CPU (not a GPU) on a HPC cluster, DESC is unable to
        accurately estimate the available device memory, so the ``auto`` chunk_size
        option will yield a larger chunk size than may be needed. It is recommended
        to manually choose a chunk_size if an OOM error is experienced in this case.
"""
docs = {
    "target": doc_target,
    "bounds": doc_bounds,
    "weight": doc_weight,
    "normalize": doc_normalize,
    "normalize_target": doc_normalize_target,
    "loss_function": doc_loss_function,
    "deriv_mode": doc_deriv_mode,
    "name": doc_name,
    "jac_chunk_size": doc_jac_chunk_size,
}


def collect_docs(
    overwrite=None,
    target_default="",
    bounds_default="",
    normalize_detail=None,
    normalize_target_detail=None,
    loss_detail=None,
    coil=False,
):
    """Collect default parameters for the docstring of Objective.

    Parameters
    ----------
    overwrite : dict, optional
        Dict of strings to overwrite from the ``_Objective``'s docstring. If None,
        all default parameters are included as they are. Use this argument if
        you want to specify a special docstring for a specific parameter in
        your objective definition.
    target_default : str, optional
        Default value for the ``target`` parameter.
    bounds_default : str, optional
        Default value for the ``bounds`` parameter.
    normalize_detail : str, optional
        Additional information about the ``normalize`` parameter.
    normalize_target_detail : str, optional
        Additional information about the ``normalize_target`` parameter.
    loss_detail : str, optional
        Additional information about the ``loss`` function.
    coil : bool, optional
        Whether the objective is a coil objective. If ``True``, adds extra docs
        to ``target`` and ``loss_function``.

    Returns
    -------
    doc_params : str
        String of default parameters for the docstring.

    """
    doc_params = ""
    for key in docs.keys():
        if overwrite is not None and key in overwrite.keys():
            doc_params += overwrite[key].rstrip()
        else:
            if key == "target":
                target = ""
                if coil:
                    target += (
                        "If array, it has to be flattened according to the "
                        + "number of inputs."
                    )
                if target_default != "":
                    target = target + " Defaults to " + target_default
                doc_params += docs[key].rstrip() + target
            elif key == "bounds" and bounds_default != "":
                doc_params = (
                    doc_params + docs[key].rstrip() + " Defaults to " + bounds_default
                )
            elif key == "loss_function":
                loss = ""
                if coil:
                    loss = " Operates over all coils, not each individual coil."
                if loss_detail is not None:
                    loss += loss_detail
                doc_params += docs[key].rstrip() + loss
            elif key == "normalize":
                norm = ""
                if normalize_detail is not None:
                    norm += normalize_detail
                doc_params += docs[key].rstrip() + norm
            elif key == "normalize_target":
                norm_target = ""
                if normalize_target_detail is not None:
                    norm_target = normalize_target_detail
                doc_params += docs[key].rstrip() + norm_target
            else:
                doc_params += docs[key].rstrip()

    return doc_params


class ObjectiveFunction(IOAble):
    """Objective function comprised of one or more Objectives.

    Parameters
    ----------
    objectives : tuple of Objective
        List of objectives to be minimized.
    use_jit : bool, optional
        Whether to just-in-time compile the objectives and derivatives.
    deriv_mode : {"auto", "batched", "blocked"}
        Method for computing Jacobian matrices. ``batched`` uses forward mode, applied
        to the entire objective at once, and is generally the fastest for vector
        valued objectives. Its memory intensity vs. speed may be traded off through
        the ``jac_chunk_size`` keyword argument. "blocked" builds the Jacobian for
        each objective separately, using each objective's preferred AD mode (and
        each objective's `jac_chunk_size`). Generally the most efficient option when
        mixing scalar and vector valued objectives.
        ``auto`` defaults to ``batched`` if all sub-objectives are set to ``fwd``,
        otherwise ``blocked``.
    name : str
        Name of the objective function.
    jac_chunk_size : int or ``auto``, optional
         If ``batched`` deriv_mode is used, will calculate the Jacobian
        ``jac_chunk_size`` columns at a time, instead of all at once.
        The memory usage of the Jacobian calculation is roughly
        ``memory usage = m0+m1*jac_chunk_size``: the smaller the chunk size,
        the less memory the Jacobian calculation will require (with some baseline
        memory usage). The time it takes to compute the Jacobian is roughly
        ``t = t0+t1/jac_chunk_size`` so the larger the ``jac_chunk_size``, the faster
        the calculation takes, at the cost of requiring more memory.
        If ``None``, it will use the largest size i.e ``obj.dim_x``.
        Can also help with Hessian computation memory, as Hessian is essentially
        ``jacfwd(jacrev(f))``, and each of these operations may be chunked.
        Defaults to ``chunk_size=None``.
        Note: When running on a CPU (not a GPU) on a HPC cluster, DESC is unable to
        accurately estimate the available device memory, so the "auto" chunk_size
        option will yield a larger chunk size than may be needed. It is recommended
        to manually choose a chunk_size if an OOM error is experienced in this case.

    """

    _io_attrs_ = ["_objectives"]

    def __init__(
        self,
        objectives,
        use_jit=True,
        deriv_mode="auto",
        name="ObjectiveFunction",
        jac_chunk_size="auto",
    ):
        if not isinstance(objectives, (tuple, list)):
            objectives = (objectives,)
        assert all(
            isinstance(obj, _Objective) for obj in objectives
        ), "members of ObjectiveFunction should be instances of _Objective"
        assert use_jit in {True, False}
        if deriv_mode == "looped":
            # overwrite the user inputs if deprecated "looped" was given
            warnif(
                True,
                DeprecationWarning,
                '``deriv_mode="looped"`` is deprecated in favor of'
                ' ``deriv_mode="batched"`` with ``jac_chunk_size=1``.',
            )
            deriv_mode = "batched"
            jac_chunk_size = 1
        assert deriv_mode in {"auto", "batched", "blocked"}
        assert jac_chunk_size in ["auto", None] or isposint(jac_chunk_size)

        self._jac_chunk_size = jac_chunk_size
        self._objectives = objectives
        self._use_jit = use_jit
        self._deriv_mode = deriv_mode
        self._built = False
        self._compiled = False
        self._name = name

    def _unjit(self):
        """Remove jit compiled methods."""
        methods = [
            "compute_scaled",
            "compute_scaled_error",
            "compute_unscaled",
            "compute_scalar",
            "jac_scaled",
            "jac_scaled_error",
            "jac_unscaled",
            "hess",
            "grad",
            "jvp_scaled",
            "jvp_scaled_error",
            "jvp_unscaled",
            "vjp_scaled",
            "vjp_scaled_error",
            "vjp_unscaled",
        ]
        for method in methods:
            try:
                setattr(
                    self, method, functools.partial(getattr(self, method)._fun, self)
                )
            except AttributeError:
                pass

    @execute_on_cpu
    def build(self, use_jit=None, verbose=1):
        """Build the objective.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if use_jit is not None:
            self._use_jit = use_jit
        timer = Timer()
        timer.start("Objective build")

        # build objectives
        self._dim_f = 0
        for objective in self.objectives:
            if not objective.built:
                if verbose > 0:
                    print("Building objective: " + objective.name)
                objective.build(use_jit=self.use_jit, verbose=verbose)
            self._dim_f += objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        self._set_things()

        # setting derivative mode and chunking.
        errorif(
            isposint(self._jac_chunk_size) and self._deriv_mode in ["auto", "blocked"],
            ValueError,
            "'jac_chunk_size' was passed into ObjectiveFunction, but the "
            "ObjectiveFunction is not using 'batched' deriv_mode",
        )
        sub_obj_jac_chunk_sizes_are_ints = [
            isposint(obj._jac_chunk_size) for obj in self.objectives
        ]
        errorif(
            any(sub_obj_jac_chunk_sizes_are_ints) and self._deriv_mode == "batched",
            ValueError,
            "'jac_chunk_size' was passed into one or more sub-objectives, but the"
            " ObjectiveFunction is using 'batched' deriv_mode, so sub-objective "
            "'jac_chunk_size' will be ignored in favor of the ObjectiveFunction's "
            f"'jac_chunk_size' of {self._jac_chunk_size}."
            " Specify 'blocked' deriv_mode if each sub-objective is desired to have a "
            "different 'jac_chunk_size' for its Jacobian computation.",
        )

        if self._deriv_mode == "auto":
            if all((obj._deriv_mode == "fwd") for obj in self.objectives):
                self._deriv_mode = "batched"
            else:
                self._deriv_mode = "blocked"

        if self._jac_chunk_size == "auto":
            # Heuristic estimates of fwd mode Jacobian memory usage,
            # slightly conservative, based on using ForceBalance as the objective
            estimated_memory_usage = 2.4e-7 * self.dim_f * self.dim_x + 1  # in GB
            max_chunk_size = round(
                (desc_config.get("avail_mem") / estimated_memory_usage - 0.22)
                / 0.85
                * self.dim_x
            )
            self._jac_chunk_size = max([1, max_chunk_size])
            if self._deriv_mode == "blocked":
                for obj in self.objectives:
                    if obj._jac_chunk_size is None:
                        obj._jac_chunk_size = self._jac_chunk_size

        if not self.use_jit:
            self._unjit()

        self._built = True

        timer.stop("Objective build")
        if verbose > 1:
            timer.disp("Objective build")

    def _set_things(self, things=None):
        """Tell the ObjectiveFunction what things it is optimizing.

        Parameters
        ----------
        things : list, tuple, or nested list, tuple of Optimizable
            Collection of things used by this objective. Defaults to all things from
            all sub-objectives.

        Notes
        -----
        Sets ``self._flatten`` as a function to return unique flattened list of things
        and ``self._unflatten`` to recreate full nested list of things
        from unique flattened version.

        """
        # This is a unique list of the things the ObjectiveFunction knows about.
        # By default it is only the things that each sub-Objective needs,
        # but it can be set to include extra things from other objectives.
        self._things = setdefault(
            things,
            unique_list(flatten_list([obj.things for obj in self.objectives]))[0],
        )
        things_per_objective = [self._things for _ in self.objectives]

        flat_, treedef_ = tree_flatten(
            things_per_objective, is_leaf=lambda x: isinstance(x, Optimizable)
        )
        unique_, inds_ = unique_list(flat_)

        # this is needed to know which "thing" goes with which sub-objective,
        # ie objectives[i].things == [things[k] for k in things_per_objective_idx[i]]
        self._things_per_objective_idx = []
        for obj in self.objectives:
            self._things_per_objective_idx.append(
                [i for i, t in enumerate(unique_) if t in obj.things]
            )

        self._unflatten = _ThingUnflattener(len(unique_), inds_, treedef_)
        self._flatten = _ThingFlattener(len(flat_), treedef_)

    @jit
    def compute_unscaled(self, x, constants=None):
        """Compute the raw value of the objective function.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        params = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        assert len(params) == len(constants) == len(self.objectives)
        f = jnp.concatenate(
            [
                obj.compute_unscaled(*par, constants=const)
                for par, obj, const in zip(params, self.objectives, constants)
            ]
        )
        return f

    @jit
    def compute_scaled(self, x, constants=None):
        """Compute the objective function and apply weighting and normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        params = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        assert len(params) == len(constants) == len(self.objectives)
        f = jnp.concatenate(
            [
                obj.compute_scaled(*par, constants=const)
                for par, obj, const in zip(params, self.objectives, constants)
            ]
        )
        return f

    @jit
    def compute_scaled_error(self, x, constants=None):
        """Compute and apply the target/bounds, weighting, and normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        params = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        assert len(params) == len(constants) == len(self.objectives)
        f = jnp.concatenate(
            [
                obj.compute_scaled_error(*par, constants=const)
                for par, obj, const in zip(params, self.objectives, constants)
            ]
        )
        return f

    @jit
    def compute_scalar(self, x, constants=None):
        """Compute the sum of squares error.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : float
            Objective function scalar value.

        """
        f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
        return f

    def print_value(self, x, x0=None, constants=None):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            State vector.
        x0 : ndarray, optional
            Initial state vector before optimization.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        values: dict
            Dictionary mapping objective titles/names to residual values.
        """
        out = {}
        if constants is None:
            constants = self.constants
        if self.compiled and self._compile_mode in {"scalar", "all"}:
            f = self.compute_scalar(x, constants=constants)
            if x0 is not None:
                f0 = self.compute_scalar(x0, constants=constants)
        else:
            f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
            if x0 is not None:
                f0 = (
                    jnp.sum(self.compute_scaled_error(x0, constants=constants) ** 2) / 2
                )
        if x0 is not None:
            print(
                f"{'Total (sum of squares): ':<{PRINT_WIDTH}}"
                + "{:10.3e}  -->  {:10.3e}, ".format(f0, f)
            )
            temp_out = {"f": f, "f0": f0}
        else:
            print(
                f"{'Total (sum of squares): ':<{PRINT_WIDTH}}" + "{:10.3e}, ".format(f)
            )
            temp_out = {"f": f}
        out["Total (sum of squares)"] = temp_out
        params = self.unpack_state(x)
        assert len(params) == len(constants) == len(self.objectives)
        if x0 is not None:
            params0 = self.unpack_state(x0)
            assert len(params0) == len(constants) == len(self.objectives)
            for par, par0, obj, const in zip(
                params, params0, self.objectives, constants
            ):
                outi = obj.print_value(par, par0, constants=const)
                if obj._print_value_fmt in out:
                    out[obj._print_value_fmt].append(outi)
                else:
                    out[obj._print_value_fmt] = [outi]
        else:
            for par, obj, const in zip(params, self.objectives, constants):
                outi = obj.print_value(par, constants=const)
                if obj._print_value_fmt in out:
                    out[obj._print_value_fmt].append(outi)
                else:
                    out[obj._print_value_fmt] = [outi]
        return out

    def unpack_state(self, x, per_objective=True):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            State vector.
        per_objective : bool
            Whether to return param dicts for each objective (default) or for each
            unique optimizable thing.

        Returns
        -------
        params : pytree of dict
            if per_objective is True, this is a nested list of parameters for each
            sub-Objective, such that self.objectives[i] has parameters params[i].
            Otherwise, it is a list of parameters tied to each optimizable thing
            such that params[i] = self.things[i].params_dict
        """
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")

        x = jnp.atleast_1d(jnp.asarray(x))
        if x.size != self.dim_x:
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self.dim_x} got {x.size}."
            )

        xs_splits = np.cumsum([t.dim_x for t in self.things])
        xs = jnp.split(x, xs_splits)
        xs = xs[: len(self.things)]  # jnp.split returns an empty array at the end
        assert len(xs) == len(self.things)
        params = [t.unpack_params(xi) for t, xi in zip(self.things, xs)]
        if per_objective:
            # params is a list of lists of dicts, for each thing and for each objective
            params = self._unflatten(params)
            # this filters out the params of things that are unused by each objective
            assert len(params) == len(self._things_per_objective_idx)
            params = [
                [param[i] for i in idx]
                for param, idx in zip(params, self._things_per_objective_idx)
            ]
        return params

    def x(self, *things):
        """Return the full state vector from the Optimizable objects things."""
        # TODO (#1392): also check resolution of the things etc?
        things = things or self.things
        errorif(
            len(things) != len(self.things),
            ValueError,
            "Got the wrong number of things, "
            f"expected {len(self.things)} got {len(things)}",
        )
        for t1, t2 in zip(things, self.things):
            errorif(
                not isinstance(t1, type(t2)),
                TypeError,
                f"got incompatible types between things {type(t1)} "
                f"and self.things {type(t2)}",
            )
        xs = [t.pack_params(t.params_dict) for t in things]
        return jnp.concatenate(xs)

    @jit
    def grad(self, x, constants=None):
        """Compute gradient vector of self.compute_scalar wrt x."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_1d(
            Derivative(self.compute_scalar, mode="grad")(x, constants).squeeze()
        )

    @jit
    def hess(self, x, constants=None):
        """Compute Hessian matrix of self.compute_scalar wrt x."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_2d(
            Derivative(self.compute_scalar, mode="hess")(x, constants).squeeze()
        )

    @jit
    def jac_scaled(self, x, constants=None):
        """Compute Jacobian matrix of self.compute_scaled wrt x."""
        v = jnp.eye(x.shape[0])
        return self.jvp_scaled(v, x, constants).T

    @jit
    def jac_scaled_error(self, x, constants=None):
        """Compute Jacobian matrix of self.compute_scaled_error wrt x."""
        v = jnp.eye(x.shape[0])
        return self.jvp_scaled_error(v, x, constants).T

    @jit
    def jac_unscaled(self, x, constants=None):
        """Compute Jacobian matrix of self.compute_unscaled wrt x."""
        v = jnp.eye(x.shape[0])
        return self.jvp_unscaled(v, x, constants).T

    def _jvp_blocked(self, v, x, constants=None, op="scaled"):
        v = ensure_tuple(v)
        if len(v) > 1:
            # using blocked for higher order derivatives is a pain, and only really
            # is needed for perturbations. Just pass that to jvp_batched for now
            return self._jvp_batched(v, x, constants, op)

        if constants is None:
            constants = self.constants
        xs_splits = np.cumsum([t.dim_x for t in self.things])
        xs = jnp.split(x, xs_splits)
        vs = jnp.split(v[0], xs_splits, axis=-1)
        J = []
        assert len(self.objectives) == len(self.constants)
        # basic idea is we compute the jacobian of each objective wrt each thing
        # one by one, and assemble into big block matrix
        # if objective doesn't depend on a given thing, that part is set to 0.
        for k, (obj, const) in enumerate(zip(self.objectives, constants)):
            # get the xs that go to that objective
            thing_idx = self._things_per_objective_idx[k]
            xi = [xs[i] for i in thing_idx]
            vi = [vs[i] for i in thing_idx]
            Ji_ = getattr(obj, "jvp_" + op)(vi, xi, constants=const)
            J += [Ji_]
        # this is the transpose of the jvp when v is a matrix, for consistency with
        # jvp_batched
        J = jnp.hstack(J)
        return J

    def _jvp_batched(self, v, x, constants=None, op="scaled"):
        v = ensure_tuple(v)

        fun = lambda x: getattr(self, "compute_" + op)(x, constants)
        if len(v) == 1:
            jvpfun = lambda dx: Derivative.compute_jvp(fun, 0, dx, x)
            return batched_vectorize(
                jvpfun, signature="(n)->(k)", chunk_size=self._jac_chunk_size
            )(v[0])
        elif len(v) == 2:
            jvpfun = lambda dx1, dx2: Derivative.compute_jvp2(fun, 0, 0, dx1, dx2, x)
            return batched_vectorize(
                jvpfun, signature="(n),(n)->(k)", chunk_size=self._jac_chunk_size
            )(v[0], v[1])
        elif len(v) == 3:
            jvpfun = lambda dx1, dx2, dx3: Derivative.compute_jvp3(
                fun, 0, 0, 0, dx1, dx2, dx3, x
            )
            return batched_vectorize(
                jvpfun,
                signature="(n),(n),(n)->(k)",
                chunk_size=self._jac_chunk_size,
            )(v[0], v[1], v[2])
        else:
            raise NotImplementedError("Cannot compute JVP higher than 3rd order.")

    @jit
    def jvp_scaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            The number of vectors given determines the order of derivative taken.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        if self._deriv_mode == "batched":
            J = self._jvp_batched(v, x, constants, "scaled")
        if self._deriv_mode == "blocked":
            J = self._jvp_blocked(v, x, constants, "scaled")
        return J

    @jit
    def jvp_scaled_error(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            The number of vectors given determines the order of derivative taken.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        if self._deriv_mode == "batched":
            J = self._jvp_batched(v, x, constants, "scaled_error")
        if self._deriv_mode == "blocked":
            J = self._jvp_blocked(v, x, constants, "scaled_error")
        return J

    @jit
    def jvp_unscaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            The number of vectors given determines the order of derivative taken.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        if self._deriv_mode == "batched":
            J = self._jvp_batched(v, x, constants, "unscaled")
        if self._deriv_mode == "blocked":
            J = self._jvp_blocked(v, x, constants, "unscaled")
        return J

    def _vjp(self, v, x, constants=None, op="scaled"):
        fun = lambda x: getattr(self, "compute_" + op)(x, constants)
        return Derivative.compute_vjp(fun, 0, v, x)

    @jit
    def vjp_scaled(self, v, x, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x, constants, "scaled")

    @jit
    def vjp_scaled_error(self, v, x, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled_error.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x, constants, "scaled_error")

    @jit
    def vjp_unscaled(self, v, x, constants=None):
        """Compute vector-Jacobian product of self.compute_unscaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x, constants, "unscaled")

    def compile(self, mode="auto", verbose=1):
        """Call the necessary functions to ensure the function is compiled.

        Parameters
        ----------
        mode : {"auto", "lsq", "scalar", "bfgs", "all"}
            Whether to compile for least squares optimization or scalar optimization.
            "auto" compiles based on the type of objective, either scalar or lsq
            "bfgs" compiles only scalar objective and gradient,
            "all" compiles all derivatives.
        verbose : int, optional
            Level of output.

        """
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        if not use_jax:
            self._compiled = True
            return

        timer = Timer()
        if mode == "auto" and self.scalar:
            mode = "scalar"
        elif mode == "auto":
            mode = "lsq"
        self._compile_mode = mode
        x = self.x()

        if verbose > 0:
            msg = "Compiling objective function and derivatives: "
            print(msg + f"{[obj.name for obj in self.objectives]}")
        timer.start("Total compilation time")

        if mode in ["scalar", "bfgs", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute_scalar(x, self.constants).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")

            timer.start("Gradient compilation time")
            _ = self.grad(x, self.constants).block_until_ready()
            timer.stop("Gradient compilation time")
            if verbose > 1:
                timer.disp("Gradient compilation time")
        if mode in ["scalar", "all"]:
            timer.start("Hessian compilation time")
            _ = self.hess(x, self.constants).block_until_ready()
            timer.stop("Hessian compilation time")
            if verbose > 1:
                timer.disp("Hessian compilation time")
        if mode in ["lsq", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute_scaled_error(x, self.constants).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")

            timer.start("Jacobian compilation time")
            _ = self.jac_scaled_error(x, self.constants).block_until_ready()
            timer.stop("Jacobian compilation time")
            if verbose > 1:
                timer.disp("Jacobian compilation time")

        timer.stop("Total compilation time")
        if verbose > 1:
            timer.disp("Total compilation time")
        self._compiled = True

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        return [obj.constants for obj in self.objectives]

    @property
    def objectives(self):
        """list: List of objectives."""
        return self._objectives

    @property
    def use_jit(self):
        """bool: Whether to just-in-time compile the objective and derivatives."""
        return self._use_jit

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar or vector."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._scalar

    @property
    def built(self):
        """bool: Whether the objectives have been built or not."""
        return self._built

    @property
    def compiled(self):
        """bool: Whether the functions have been compiled or not."""
        return self._compiled

    @property
    def dim_x(self):
        """int: Dimensional of the state vector."""
        return sum(t.dim_x for t in self.things)

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        if not hasattr(self, "_dim_f"):
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_f

    @property
    def name(self):
        """Name of objective function (str)."""
        return self.__dict__.setdefault("_name", "")

    @property
    def target_scaled(self):
        """ndarray: target vector."""
        target = []
        for obj in self.objectives:
            if obj.target is not None:
                target_i = jnp.ones(obj.dim_f) * obj.target
            else:
                # need to return something, so use midpoint of bounds as approx target
                target_i = jnp.ones(obj.dim_f) * (obj.bounds[0] + obj.bounds[1]) / 2
            target_i = obj._scale(target_i)
            if not obj._normalize_target:
                target_i *= obj.normalization
            target += [target_i]
        return jnp.concatenate(target)

    @property
    def bounds_scaled(self):
        """tuple: lower and upper bounds for residual vector."""
        lb, ub = [], []
        for obj in self.objectives:
            if obj.bounds is not None:
                lb_i = jnp.ones(obj.dim_f) * obj.bounds[0]
                ub_i = jnp.ones(obj.dim_f) * obj.bounds[1]
            else:
                lb_i = jnp.ones(obj.dim_f) * obj.target
                ub_i = jnp.ones(obj.dim_f) * obj.target
            lb_i = obj._scale(lb_i)
            ub_i = obj._scale(ub_i)
            if not obj._normalize_target:
                lb_i *= obj.normalization
                ub_i *= obj.normalization
            lb += [lb_i]
            ub += [ub_i]
        return (jnp.concatenate(lb), jnp.concatenate(ub))

    @property
    def weights(self):
        """ndarray: weight vector."""
        return jnp.concatenate(
            [jnp.ones(obj.dim_f) * obj.weight for obj in self.objectives]
        )

    @property
    def things(self):
        """list: Unique list of optimizable things that this objective is tied to."""
        return self._things


class _Objective(IOAble, ABC):
    """Objective (or constraint) used in the optimization of an Equilibrium.

    Parameters
    ----------
    things : Optimizable or tuple/list of Optimizable
        Objects that will be optimized to satisfy the Objective."""  # noqa: D208, D209

    _scalar = False
    _linear = False
    _coordinates = ""
    _units = "(Unknown)"
    _equilibrium = False
    _io_attrs_ = [
        "_target",
        "_bounds",
        "_weight",
        "_name",
        "_normalize",
        "_normalize_target",
        "_normalization",
        "_deriv_mode",
    ]

    def __init__(
        self,
        things=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name=None,
        jac_chunk_size=None,
    ):
        if self._scalar:
            assert self._coordinates == ""
        assert np.all(np.asarray(weight) > 0)
        assert normalize in {True, False}
        assert normalize_target in {True, False}
        assert (bounds is None) or (isinstance(bounds, tuple) and len(bounds) == 2)
        assert (bounds is None) or (target is None), "Cannot use both bounds and target"
        assert loss_function in [None, "mean", "min", "max"]
        assert deriv_mode in {"auto", "fwd", "rev"}
        assert jac_chunk_size is None or isposint(jac_chunk_size)

        self._jac_chunk_size = jac_chunk_size

        self._target = target
        self._bounds = bounds
        self._weight = weight
        self._normalize = normalize
        self._normalize_target = normalize_target
        self._normalization = 1
        self._deriv_mode = deriv_mode
        self._name = name
        self._use_jit = True
        self._built = False
        self._loss_function = {
            "mean": jnp.mean,
            "max": jnp.max,
            "min": jnp.min,
            None: None,
        }[loss_function]

        self._things = flatten_list([things], True)

    def _set_derivatives(self):
        """Choose derivative mode based on size of inputs/outputs."""
        if self._deriv_mode == "auto":
            # choose based on shape of jacobian. dim_x is usually an overestimate of
            # the true number of DOFs because linear constraints remove some. Also
            # fwd mode is more memory efficient so we prefer that unless the jacobian
            # is really wide
            self._deriv_mode = (
                "fwd"
                if self.dim_f >= 0.2 * sum(t.dim_x for t in self.things)
                else "rev"
            )

    def _unjit(self):
        """Remove jit compiled methods."""
        methods = [
            "compute_scaled",
            "compute_scaled_error",
            "compute_unscaled",
            "compute_scalar",
            "jac_scaled",
            "jac_scaled_error",
            "jac_unscaled",
            "jvp_scaled",
            "jvp_scaled_error",
            "jvp_unscaled",
            "hess",
            "grad",
        ]
        for method in methods:
            try:
                setattr(
                    self, method, functools.partial(getattr(self, method)._fun, self)
                )
            except AttributeError:
                pass

    def _check_dimensions(self):
        """Check that len(target) = len(bounds) = len(weight) = dim_f."""
        if self.bounds is not None:  # must be a tuple of length 2
            self._bounds = tuple([np.asarray(bound) for bound in self._bounds])
            for bound in self.bounds:
                if not is_broadcastable((self.dim_f,), bound.shape) or (
                    self.dim_f == 1 and bound.size != 1
                ):
                    raise ValueError("len(bounds) != dim_f")
            if np.any(self.bounds[1] < self.bounds[0]):
                raise ValueError("bounds must be: (lower bound, upper bound)")
        else:  # target only gets used if bounds is None
            self._target = np.asarray(self._target)
            if not is_broadcastable((self.dim_f,), self.target.shape) or (
                self.dim_f == 1 and self.target.size != 1
            ):
                raise ValueError("len(target) != dim_f")

        self._weight = np.asarray(self._weight)
        if not is_broadcastable((self.dim_f,), self.weight.shape) or (
            self.dim_f == 1 and self.weight.size != 1
        ):
            raise ValueError("len(weight) != dim_f")

    @abstractmethod
    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        self._check_dimensions()
        self._set_derivatives()

        # set quadrature weights if they haven't been
        if hasattr(self, "_constants") and ("quad_weights" not in self._constants):
            grid = self._constants["transforms"]["grid"]
            if self._coordinates == "rtz":
                w = grid.weights
                w *= jnp.sqrt(grid.num_nodes)
            elif self._coordinates == "r":
                w = grid.compress(grid.spacing[:, 0], surface_label="rho")
                w = jnp.sqrt(w)
            else:
                w = jnp.ones((self.dim_f,))
            if w.size:
                w = jnp.tile(w, self.dim_f // w.size)
            self._constants["quad_weights"] = w

        if self._loss_function is not None:
            self._dim_f = 1
            if hasattr(self, "_constants"):
                self._constants["quad_weights"] = 1.0

        if use_jit is not None:
            self._use_jit = use_jit
        if not self._use_jit:
            self._unjit()

        self._built = True

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the objective function."""

    def _maybe_array_to_params(self, *args):
        argsout = tuple()
        assert len(args) == len(self.things)
        for arg, thing in zip(args, self.things):
            if isinstance(arg, (np.ndarray, jnp.ndarray)):
                argsout += (thing.unpack_params(arg),)
            else:
                argsout += (arg,)
        return argsout

    @jit
    def compute_unscaled(self, *args, **kwargs):
        """Compute the raw value of the objective."""
        args = self._maybe_array_to_params(*args)
        f = self.compute(*args, **kwargs)
        if self._loss_function is not None:
            f = self._loss_function(f)
        return jnp.atleast_1d(f)

    @jit
    def compute_scaled(self, *args, **kwargs):
        """Compute and apply weighting and normalization."""
        args = self._maybe_array_to_params(*args)
        f = self.compute(*args, **kwargs)
        if self._loss_function is not None:
            f = self._loss_function(f)
        return jnp.atleast_1d(self._scale(f, **kwargs))

    @jit
    def compute_scaled_error(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        args = self._maybe_array_to_params(*args)
        f = self.compute(*args, **kwargs)
        if self._loss_function is not None:
            f = self._loss_function(f)
        return jnp.atleast_1d(self._scale(self._shift(f), **kwargs))

    def _shift(self, f):
        """Subtract target or clamp to bounds."""
        if self.bounds is not None:  # using lower/upper bounds instead of target
            if self._normalize_target:
                bounds = self.bounds
            else:
                bounds = tuple([bound * self.normalization for bound in self.bounds])
            f_target = jnp.where(  # where f is within target bounds, return 0 error
                jnp.logical_and(f >= bounds[0], f <= bounds[1]),
                jnp.zeros_like(f),
                jnp.where(  # otherwise return error = f - bound
                    jnp.abs(f - bounds[0]) < jnp.abs(f - bounds[1]),
                    f - bounds[0],  # errors below lower bound are negative
                    f - bounds[1],  # errors above upper bound are positive
                ),
            )
        else:  # using target instead of lower/upper bounds
            if self._normalize_target:
                target = self.target
            else:
                target = self.target * self.normalization
            f_target = f - target
        return f_target

    def _scale(self, f, *args, **kwargs):
        """Apply weighting, normalization etc."""
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]
        f_norm = jnp.atleast_1d(f) / self.normalization  # normalization
        return f_norm * w * self.weight

    @jit
    def compute_scalar(self, *args, **kwargs):
        """Compute the scalar form of the objective."""
        if self.scalar:
            f = self.compute_scaled_error(*args, **kwargs)
        else:
            f = jnp.sum(self.compute_scaled_error(*args, **kwargs) ** 2) / 2
        return f.squeeze()

    @jit
    def grad(self, *args, **kwargs):
        """Compute gradient vector of self.compute_scalar wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(self.compute_scalar, argnums, mode="grad")(*args, **kwargs)

    @jit
    def hess(self, *args, **kwargs):
        """Compute Hessian matrix of self.compute_scalar wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(self.compute_scalar, argnums, mode="hess")(*args, **kwargs)

    @jit
    def jac_scaled(self, *args, **kwargs):
        """Compute Jacobian matrix of self.compute_scaled wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(
            self.compute_scaled,
            argnums,
            mode=self._deriv_mode,
            chunk_size=self._jac_chunk_size,
        )(*args, **kwargs)

    @jit
    def jac_scaled_error(self, *args, **kwargs):
        """Compute Jacobian matrix of self.compute_scaled_error wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(
            self.compute_scaled_error,
            argnums,
            mode=self._deriv_mode,
            chunk_size=self._jac_chunk_size,
        )(*args, **kwargs)

    @jit
    def jac_unscaled(self, *args, **kwargs):
        """Compute Jacobian matrix of self.compute_unscaled wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(
            self.compute_unscaled,
            argnums,
            mode=self._deriv_mode,
            chunk_size=self._jac_chunk_size,
        )(*args, **kwargs)

    def _jvp(self, v, x, constants=None, op="scaled"):
        v = ensure_tuple(v)
        x = ensure_tuple(x)
        assert len(x) == len(v)

        if self._deriv_mode == "fwd":
            fun = lambda *x: getattr(self, "compute_" + op)(*x, constants=constants)
            jvpfun = lambda *dx: Derivative.compute_jvp(
                fun, tuple(range(len(x))), dx, *x
            )
            sig = ",".join(f"(n{i})" for i in range(len(x))) + "->(k)"
            return batched_vectorize(
                jvpfun, signature=sig, chunk_size=self._jac_chunk_size
            )(*v)
        else:  # rev mode. We compute full jacobian and manually do mv. In this case
            # the jacobian should be wide so this isn't very expensive.
            jac = getattr(self, "jac_" + op)(*x, constants=constants)
            # jac is a tuple, 1 array for each thing. Transposes here and below make it
            # equivalent to fwd mode above, which batches over the first axis
            Jv = tree_map(lambda a, b: jnp.dot(a, b.T), jac, v)
            # sum over different things.
            return jnp.sum(jnp.asarray(Jv), axis=0).T

    @jit
    def jvp_scaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x : tuple of ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x, constants, "scaled")

    @jit
    def jvp_scaled_error(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x : tuple of ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x, constants, "scaled_error")

    @jit
    def jvp_unscaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x : tuple of ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x, constants, "unscaled")

    def print_value(self, args, args0=None, **kwargs):  # noqa: C901
        """Print the value of the objective and return a dict of values."""
        # compute_unscaled is jitted so better to use than than bare compute
        out = {}
        if args0 is not None:
            f = self.compute_unscaled(*args, **kwargs)
            f0 = self.compute_unscaled(*args0, **kwargs)
            print_value_fmt = (
                f"{self._print_value_fmt:<{PRINT_WIDTH}}" + "{:10.3e}  -->  {:10.3e} "
            )
        else:
            f = self.compute_unscaled(*args, **kwargs)
            f0 = f
            # In this case, print_value_fmt only has 1 value,
            # but the format string is still used with 2 arguments given.
            # This is a bit of a hack, but it works. the format() only replaces
            # the first value in the {} string, so the second one is unused.
            # That is why we set f0 to f.
            print_value_fmt = f"{self._print_value_fmt:<{PRINT_WIDTH}}" + "{:10.3e} "

        if self.linear:
            # probably a Fixed* thing, just need to know norm
            f = jnp.linalg.norm(self._shift(f))
            f0 = jnp.linalg.norm(self._shift(f0))
            print(print_value_fmt.format(f0, f) + self._units)
            out["f"] = f
            if args0 is not None:
                out["f0"] = f0

        elif self.scalar:
            # dont need min/max/mean of a scalar
            fs = f.squeeze()
            f0s = f0.squeeze()
            print(print_value_fmt.format(f0s, fs) + self._units)
            out["f"] = fs
            if args0 is not None:
                out["f0"] = f0s
            if self._normalize and self._units != "(dimensionless)":
                fs_norm = self._scale(self._shift(f)).squeeze()
                f0s_norm = self._scale(self._shift(f0)).squeeze()
                print(print_value_fmt.format(f0s_norm, fs_norm) + "(normalized error)")
                out["f_norm"] = fs_norm
                if args0 is not None:
                    out["f0_norm"] = f0s_norm

        else:
            # try to do weighted mean if possible
            constants = kwargs.get("constants", self.constants)
            if constants is None:
                w = jnp.ones_like(f)
            else:
                w = constants["quad_weights"]

            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            abserr = jnp.all(self.target == 0)
            f = jnp.abs(f) if abserr else f
            fmax = jnp.max(f)
            fmin = jnp.min(f)
            fmean = jnp.mean(f * w) / jnp.mean(w)

            f0 = jnp.abs(f0) if abserr else f0
            f0max = jnp.max(f0)
            f0min = jnp.min(f0)
            f0mean = jnp.mean(f0 * w) / jnp.mean(w)

            pre_width = len("Maximum absolute ") if abserr else len("Maximum ")
            if args0 is not None:
                print_value_fmt = (
                    f"{self._print_value_fmt:<{PRINT_WIDTH-pre_width}}"
                    + "{:10.3e}  -->  {:10.3e} "
                )
            else:
                print_value_fmt = (
                    f"{self._print_value_fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
                )
            print(
                "Maximum "
                + ("absolute " if abserr else "")
                + print_value_fmt.format(f0max, fmax)
                + self._units
            )
            out["f_max"] = fmax
            if args0 is not None:
                out["f0_max"] = f0max
            print(
                "Minimum "
                + ("absolute " if abserr else "")
                + print_value_fmt.format(f0min, fmin)
                + self._units
            )
            out["f_min"] = fmin
            if args0 is not None:
                out["f0_min"] = f0min
            print(
                "Average "
                + ("absolute " if abserr else "")
                + print_value_fmt.format(f0mean, fmean)
                + self._units
            )
            out["f_mean"] = fmean
            if args0 is not None:
                out["f0_mean"] = f0mean

            if self._normalize and self._units != "(dimensionless)":
                fmax_norm = fmax / jnp.mean(self.normalization)
                fmin_norm = fmin / jnp.mean(self.normalization)
                fmean_norm = fmean / jnp.mean(self.normalization)

                f0max_norm = f0max / jnp.mean(self.normalization)
                f0min_norm = f0min / jnp.mean(self.normalization)
                f0mean_norm = f0mean / jnp.mean(self.normalization)

                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + print_value_fmt.format(f0max_norm, fmax_norm)
                    + "(normalized)"
                )
                out["f_max_norm"] = fmax_norm
                if args0 is not None:
                    out["f0_max_norm"] = f0max_norm
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + print_value_fmt.format(f0min_norm, fmin_norm)
                    + "(normalized)"
                )
                out["f_min_norm"] = fmin_norm
                if args0 is not None:
                    out["f0_min_norm"] = f0min_norm
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + print_value_fmt.format(f0mean_norm, fmean_norm)
                    + "(normalized)"
                )
                out["f_mean_norm"] = fmean_norm
                if args0 is not None:
                    out["f0_mean_norm"] = f0mean_norm
        return out

    def xs(self, *things):
        """Return a tuple of args required by this objective from optimizable things."""
        things = things or self.things
        errorif(
            len(things) != len(self.things),
            ValueError,
            "Got the wrong number of things, "
            f"expected {len(self.things)} got {len(things)}",
        )
        for t1, t2 in zip(things, self.things):
            errorif(
                not isinstance(t1, type(t2)),
                TypeError,
                f"got incompatible types between things {type(t1)} "
                f"and self.things {type(t2)}",
            )
        return tuple([t.params_dict for t in things])

    @property
    def constants(self):
        """dict: Constant parameters such as transforms and profiles."""
        if hasattr(self, "_constants"):
            return self._constants
        return None

    @property
    def target(self):
        """float: Target value(s) of the objective."""
        return self._target

    @target.setter
    def target(self, target):
        self._target = np.atleast_1d(target) if target is not None else target
        self._check_dimensions()

    @property
    def bounds(self):
        """tuple: Lower and upper bounds of the objective."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        assert (bounds is None) or (isinstance(bounds, tuple) and len(bounds) == 2)
        self._bounds = bounds
        self._check_dimensions()

    @property
    def weight(self):
        """float: Weighting to apply to the Objective, relative to other Objectives."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        assert np.all(np.asarray(weight) > 0)
        self._weight = np.atleast_1d(weight)
        self._check_dimensions()

    @property
    def normalization(self):
        """float: normalizing scale factor."""
        if self._normalize and not self.built:
            raise ValueError("Objective must be built first")
        return self._normalization

    @property
    def built(self):
        """bool: Whether the transforms have been precomputed (or not)."""
        return self._built

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        return self._dim_f

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar or vector."""
        return self._scalar

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return self._linear

    @property
    def fixed(self):
        """bool: Whether the objective fixes individual parameters (or linear combo)."""
        if self.linear:
            return self._fixed
        else:
            return False

    @property
    def name(self):
        """Name of objective (str)."""
        return self.__dict__.setdefault("_name", "")

    @property
    def things(self):
        """list: Optimizable things that this objective is tied to."""
        if not hasattr(self, "_things"):
            self._things = []
        return list(self._things)

    @things.setter
    def things(self, new):
        if not isinstance(new, (tuple, list)):
            new = [new]
        assert all(isinstance(x, Optimizable) for x in new)
        assert len(new) == len(self.things)
        assert all(type(a) is type(b) for a, b in zip(new, self.things))
        self._things = list(new)
        # can maybe improve this later to not rebuild if resolution is the same
        self._built = False


_Objective.__doc__ += "".join(value.rstrip("\n") for value in docs.values())

# local functions assigned as attributes aren't hashable so they cause stuff to
# recompile, so instead we define a hashable class to do the same thing.


class _ThingUnflattener(IOAble):

    _static_attrs = ["length", "inds", "treedef"]

    def __init__(self, length, inds, treedef):
        self.length = length
        self.inds = inds
        self.treedef = treedef

    def __call__(self, unique):
        assert len(unique) == self.length
        flat = [unique[i] for i in self.inds]
        return tree_unflatten(self.treedef, flat)


class _ThingFlattener(IOAble):

    _static_attrs = ["length", "treedef"]

    def __init__(self, length, treedef):
        self.length = length
        self.treedef = treedef

    def __call__(self, things):
        flat, treedef = tree_flatten(
            things, is_leaf=lambda x: isinstance(x, Optimizable)
        )
        assert treedef == self.treedef
        assert len(flat) == self.length
        unique, _ = unique_list(flat)
        return unique
