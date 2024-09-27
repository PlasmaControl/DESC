"""Utility functions for the ``batched_vectorize`` function."""

import functools
from functools import partial
from typing import Any, Callable, Optional, Sequence

import numpy as np
from jax._src import core, dispatch, dtypes
from jax._src.api_util import (
    _ensure_index,
    argnums_partial,
    check_callable,
    flatten_fun_nokwargs,
    flatten_fun_nokwargs2,
    shaped_abstractify,
)
from jax._src.interpreters import ad
from jax._src.lax import lax as lax_internal
from jax._src.tree_util import (
    Partial,
    tree_flatten,
    tree_map,
    tree_structure,
    tree_transpose,
    tree_unflatten,
)
from jax._src.util import safe_map, wraps

from desc.backend import jax, jnp

if jax.__version_info__ >= (0, 4, 16):
    from jax.extend import linear_util as lu
else:
    from jax import linear_util as lu

from jax._src.numpy.vectorize import (
    _apply_excluded,
    _check_output_dims,
    _parse_gufunc_signature,
    _parse_input_dimensions,
)

_dtype = partial(dtypes.dtype, canonicalize=True)

# The following section of this code is derived from the NetKet project
# https://github.com/netket/netket/blob/9881c9fb217a2ac4dc9274a054bf6e6a2993c519/
# netket/jax/_chunk_utils.py
#
# The original copyright notice is as follows
# Copyright 2021 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


def _treeify(f):
    def _f(x, *args, **kwargs):
        return jax.tree_util.tree_map(lambda y: f(y, *args, **kwargs), x)

    return _f


@_treeify
def _unchunk(x):
    return x.reshape((-1,) + x.shape[2:])


@_treeify
def _chunk(x, chunk_size=None):
    # chunk_size=None -> add just a dummy chunk dimension,
    #  same as np.expand_dims(x, 0)
    if x.ndim == 0:
        raise ValueError("x cannot be chunked as it has 0 dimensions.")
    n = x.shape[0]
    if chunk_size is None:
        chunk_size = n

    n_chunks, residual = divmod(n, chunk_size)
    if residual != 0:
        raise ValueError(
            "The first dimension of x must be divisible by chunk_size."
            + f"\n        Got x.shape={x.shape} but chunk_size={chunk_size}."
        )
    return x.reshape((n_chunks, chunk_size) + x.shape[1:])


####

# The following section of this code is derived from the NetKet project
# https://github.com/netket/netket/blob/9881c9fb217a2ac4dc9274a054bf6e6a2993c519/
# netket/jax/_scanmap.py


def scan_append(f, x):
    """Evaluate f element by element in x while appending the results.

    Parameters
    ----------
    f: a function that takes elements of the leading dimension of x
    x: a pytree where each leaf array has the same leading dimension

    Returns
    -------
    a (pytree of) array(s) with leading dimension same as x,
    containing the evaluation of f at each element in x
    """
    carry_init = True

    def f_(carry, x):
        return False, f(x)

    _, res_append = jax.lax.scan(f_, carry_init, x, unroll=1)
    return res_append


# TODO in_axes a la vmap?
def _scanmap(fun, scan_fun, argnums=0):
    """A helper function to wrap f with a scan_fun."""

    def f_(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = jax.api_util.argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_


# The following section of this code is derived from the NetKet project
# https://github.com/netket/netket/blob/9881c9fb217a2ac4dc9274a054bf6e6a2993c519/
# netket/jax/_vmap_chunked.py


def _eval_fun_in_chunks(vmapped_fun, chunk_size, argnums, *args, **kwargs):
    n_elements = jax.tree_util.tree_leaves(args[argnums[0]])[0].shape[0]
    n_chunks, n_rest = divmod(n_elements, chunk_size)

    if n_chunks == 0 or chunk_size >= n_elements:
        y = vmapped_fun(*args, **kwargs)
    else:
        # split inputs
        def _get_chunks(x):
            x_chunks = jax.tree_util.tree_map(
                lambda x_: x_[: n_elements - n_rest, ...], x
            )
            x_chunks = _chunk(x_chunks, chunk_size)
            return x_chunks

        def _get_rest(x):
            x_rest = jax.tree_util.tree_map(
                lambda x_: x_[n_elements - n_rest :, ...], x
            )
            return x_rest

        args_chunks = [
            _get_chunks(a) if i in argnums else a for i, a in enumerate(args)
        ]
        args_rest = [_get_rest(a) if i in argnums else a for i, a in enumerate(args)]

        y_chunks = _unchunk(
            _scanmap(vmapped_fun, scan_append, argnums)(*args_chunks, **kwargs)
        )

        if n_rest == 0:
            y = y_chunks
        else:
            y_rest = vmapped_fun(*args_rest, **kwargs)
            y = jax.tree_util.tree_map(
                lambda y1, y2: jnp.concatenate((y1, y2)), y_chunks, y_rest
            )
    return y


def _chunk_vmapped_function(
    vmapped_fun: Callable,
    chunk_size: Optional[int],
    argnums=0,
) -> Callable:
    """Takes a vmapped function and computes it in chunks."""
    if chunk_size is None:
        return vmapped_fun

    if isinstance(argnums, int):
        argnums = (argnums,)
    return functools.partial(_eval_fun_in_chunks, vmapped_fun, chunk_size, argnums)


def _parse_in_axes(in_axes):
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(
        map(lambda ix: ix[0], filter(lambda ix: ix[1] is not None, enumerate(in_axes)))
    )
    return in_axes, argnums


def vmap_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: Optional[int],
) -> Callable:
    """Behaves like jax.vmap but uses scan to chunk the computations in smaller chunks.

    Parameters
    ----------
    f: The function to be vectorised.
    in_axes: The axes that should be scanned along. Only supports `0` or `None`
    chunk_size: The maximum size of the chunks to be used. If it is `None`,
        chunking is disabled


    Returns
    -------
    f: A vectorised and chunked function
    """
    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums)


def batched_vectorize(pyfunc, *, excluded=frozenset(), signature=None, chunk_size=None):
    """Define a vectorized function with broadcasting and batching.

    :func:`vectorize` is a convenience wrapper for defining vectorized
    functions with broadcasting, in the style of NumPy's
    `generalized universal functions
    <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_.
    It allows for defining functions that are automatically repeated across
    any leading dimensions, without the implementation of the function needing to
    be concerned about how to handle higher dimensional inputs.

    :func:`jax.numpy.vectorize` has the same interface as
    :class:`numpy.vectorize`, but it is syntactic sugar for an auto-batching
    transformation (:func:`vmap`) rather than a Python loop. This should be
    considerably more efficient, but the implementation must be written in terms
    of functions that act on JAX arrays.

    Parameters
    ----------
    pyfunc: callable,function to vectorize.
    excluded: optional set of integers representing positional arguments for
    which the function will not be vectorized. These will be passed directly
    to ``pyfunc`` unmodified.
    signature: optional generalized universal function signature, e.g.,
    ``(m,n),(n)->(m)`` for vectorized matrix-vector multiplication. If
    provided, ``pyfunc`` will be called with (and expected to return) arrays
    with shapes given by the size of corresponding core dimensions. By
    default, pyfunc is assumed to take scalars arrays as input and output.
    chunk_size: the size of the batches to pass to vmap. If None, defaults to
    the largest possible chunk_size (like the default behavior of ``vectorize11)

    Returns
    -------
    Batch-vectorized version of the given function.

    """
    if any(not isinstance(exclude, (str, int)) for exclude in excluded):
        raise TypeError(
            "jax.numpy.vectorize can only exclude integer or string arguments, "
            "but excluded={!r}".format(excluded)
        )
    if any(isinstance(e, int) and e < 0 for e in excluded):
        raise ValueError(f"excluded={excluded!r} contains negative numbers")

    @functools.wraps(pyfunc)
    def wrapped(*args, **kwargs):
        error_context = (
            "on vectorized function with excluded={!r} and "
            "signature={!r}".format(excluded, signature)
        )
        excluded_func, args, kwargs = _apply_excluded(pyfunc, excluded, args, kwargs)

        if signature is not None:
            input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
        else:
            input_core_dims = [()] * len(args)
            output_core_dims = None

        none_args = {i for i, arg in enumerate(args) if arg is None}
        if any(none_args):
            if any(input_core_dims[i] != () for i in none_args):
                raise ValueError(
                    f"Cannot pass None at locations {none_args} with {signature=}"
                )
            excluded_func, args, _ = _apply_excluded(excluded_func, none_args, args, {})
            input_core_dims = [
                dim for i, dim in enumerate(input_core_dims) if i not in none_args
            ]

        args = tuple(map(jnp.asarray, args))

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims, error_context
        )

        checked_func = _check_output_dims(
            excluded_func, dim_sizes, output_core_dims, error_context
        )

        # Rather than broadcasting all arguments to full broadcast shapes, prefer
        # expanding dimensions using vmap. By pushing broadcasting
        # into vmap, we can make use of more efficient batching rules for
        # primitives where only some arguments are batched (e.g., for
        # lax_linalg.triangular_solve), and avoid instantiating large broadcasted
        # arrays.

        squeezed_args = []
        rev_filled_shapes = []

        for arg, core_dims in zip(args, input_core_dims):
            noncore_shape = arg.shape[: arg.ndim - len(core_dims)]

            pad_ndim = len(broadcast_shape) - len(noncore_shape)
            filled_shape = pad_ndim * (1,) + noncore_shape
            rev_filled_shapes.append(filled_shape[::-1])

            squeeze_indices = tuple(
                i for i, size in enumerate(noncore_shape) if size == 1
            )
            squeezed_arg = jnp.squeeze(arg, axis=squeeze_indices)
            squeezed_args.append(squeezed_arg)

        vectorized_func = checked_func
        dims_to_expand = []
        for negdim, axis_sizes in enumerate(zip(*rev_filled_shapes)):
            in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
            if all(axis is None for axis in in_axes):
                dims_to_expand.append(len(broadcast_shape) - 1 - negdim)
            else:
                # change the vmap here to chunked_vmap
                vectorized_func = vmap_chunked(
                    vectorized_func, in_axes, chunk_size=chunk_size
                )
        result = vectorized_func(*squeezed_args)

        if not dims_to_expand:
            return result
        elif isinstance(result, tuple):
            return tuple(jnp.expand_dims(r, axis=dims_to_expand) for r in result)
        else:
            return jnp.expand_dims(result, axis=dims_to_expand)

    return wrapped


# The following section of this code is derived from JAX
# https://github.com/jax-ml/jax/blob/ff0a98a2aef958df156ca149809cf532efbbcaf4/
# jax/_src/api.py
#
# The original copyright notice is as follows
# Copyright 2018 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License");


def jacfwd_chunked(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    *,
    chunk_size=None,
) -> Callable:
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    Parameters
    ----------
    fun: callable
        Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers.
        Specifies which positional argument(s) to differentiate with respect to
        (default ``0``).
    has_aux: Optional, bool.
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data. Default False.
    chunk_size: int
        The size of the batches to pass to vmap. If None, defaults to the largest
        possible chunk_size.

    Returns
    -------
    jac: callable
        A function with the same arguments as ``fun``, that evaluates the Jacobian of
        ``fun`` using forward-mode automatic differentiation. If ``has_aux`` is True
        then a pair of (jacobian, auxiliary_data) is returned.

    """
    check_callable(fun)
    argnums = _ensure_index(argnums)

    docstr = (
        "Jacobian of {fun} with respect to positional argument(s) "
        "{argnums}. Takes the same arguments as {fun} but returns the "
        "jacobian of the output with respect to the arguments at "
        "positions {argnums}."
    )

    @wraps(fun, docstr=docstr, argnums=argnums)
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        tree_map(_check_input_dtype_jacfwd, dyn_args)
        if not has_aux:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap_chunked(pushfwd, chunk_size=chunk_size)(_std_basis(dyn_args))
            y = tree_map(lambda x: x[0], y)
            jac = tree_map(lambda x: jnp.moveaxis(x, 0, -1), jac)
        else:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap_chunked(pushfwd, chunk_size=chunk_size)(
                _std_basis(dyn_args)
            )
            y = tree_map(lambda x: x[0], y)
            jac = tree_map(lambda x: jnp.moveaxis(x, 0, -1), jac)
            aux = tree_map(lambda x: x[0], aux)

        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if not has_aux:
            return jac_tree
        else:
            return jac_tree, aux

    return jacfun


def jacrev_chunked(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    *,
    chunk_size=None,
) -> Callable:
    """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    Parameters
    ----------
    fun: callable
        Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers.
        Specifies which positional argument(s) to differentiate with respect to
        (default ``0``).
    has_aux: Optional, bool.
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data. Default False.
    chunk_size: int
        The size of the batches to pass to vmap. If None, defaults to the largest
        possible chunk_size.

    Returns
    -------
    jac: callable
        A function with the same arguments as ``fun``, that evaluates the Jacobian of
        ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
        then a pair of (jacobian, auxiliary_data) is returned.

    """
    check_callable(fun)

    docstr = (
        "Jacobian of {fun} with respect to positional argument(s) "
        "{argnums}. Takes the same arguments as {fun} but returns the "
        "jacobian of the output with respect to the arguments at "
        "positions {argnums}."
    )

    @wraps(fun, docstr=docstr, argnums=argnums)
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        tree_map(_check_input_dtype_jacrev, dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(_check_output_dtype_jacrev, y)
        jac = vmap_chunked(pullback, chunk_size=chunk_size)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
        jac_tree = tree_transpose(
            tree_structure(example_args), tree_structure(y), jac_tree
        )
        if not has_aux:
            return jac_tree
        else:
            return jac_tree, aux

    return jacfun


def _check_input_dtype_jacrev(x):
    dispatch.check_arg(x)
    aval = core.get_aval(x)
    if (
        dtypes.issubdtype(aval.dtype, dtypes.extended)
        or dtypes.issubdtype(aval.dtype, np.integer)
        or dtypes.issubdtype(aval.dtype, np.bool_)
    ):
        raise TypeError(
            f"jacrev_chunked requires real- or complex-valued inputs (input dtype "
            f"that is a sub-dtype of np.inexact), but got {aval.dtype.name}. "
            "If you want to use Boolean- or integer-valued inputs, use vjp "
            "or set allow_int to True."
        )
    elif not dtypes.issubdtype(aval.dtype, np.inexact):
        raise TypeError(
            f"jacrev_chunked requires numerical-valued inputs (input dtype that is a "
            f"sub-dtype of np.bool_ or np.number), but got {aval.dtype.name}."
        )


def _check_output_dtype_jacrev(x):
    aval = core.get_aval(x)
    if dtypes.issubdtype(aval.dtype, dtypes.extended):
        raise TypeError(f"jacrev_chunked with output element type {aval.dtype.name}")
    elif dtypes.issubdtype(aval.dtype, np.complexfloating):
        raise TypeError(
            f"jacrev_chunked requires real-valued outputs (output dtype that is "
            f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
            "For holomorphic differentiation, pass holomorphic=True. "
            "For differentiation of non-holomorphic functions involving complex "
            "outputs, use jax.vjp directly."
        )
    elif not dtypes.issubdtype(aval.dtype, np.floating):
        raise TypeError(
            f"jacrev_chunked requires real-valued outputs (output dtype that is "
            f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
            "For differentiation of functions with integer outputs, use "
            "jax.vjp directly."
        )


def _check_input_dtype_jacfwd(x: Any) -> None:
    dispatch.check_arg(x)
    aval = core.get_aval(x)
    if dtypes.issubdtype(aval.dtype, dtypes.extended):
        raise TypeError(f"jacfwd with input element type {aval.dtype.name}")
    elif not dtypes.issubdtype(aval.dtype, np.floating):
        raise TypeError(
            "jacfwd requires real-valued inputs (input dtype that is "
            f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
            "For holomorphic differentiation, pass holomorphic=True. "
            "For differentiation of non-holomorphic functions involving "
            "complex inputs or integer inputs, use jax.jvp directly."
        )


def _jacfwd_unravel(input_pytree, output_pytree_leaf, arr):
    return _unravel_array_into_pytree(input_pytree, -1, output_pytree_leaf, arr)


def _jacrev_unravel(output_pytree, input_pytree_leaf, arr):
    return _unravel_array_into_pytree(output_pytree, 0, input_pytree_leaf, arr)


def _possible_downcast(x, example):
    if dtypes.issubdtype(x.dtype, np.complexfloating) and not dtypes.issubdtype(
        _dtype(example), np.complexfloating
    ):
        x = x.real
    dtype = None if example is None else _dtype(example)
    weak_type = None if example is None else dtypes.is_weakly_typed(example)
    return lax_internal._convert_element_type(x, dtype, weak_type)


def _std_basis(pytree):
    leaves, _ = tree_flatten(pytree)
    ndim = sum(safe_map(np.size, leaves))
    dtype = dtypes.result_type(*leaves)
    flat_basis = jnp.eye(ndim, dtype=dtype)
    return _unravel_array_into_pytree(pytree, 1, None, flat_basis)


def _unravel_array_into_pytree(pytree, axis, example, arr):
    """Unravel an array into a PyTree with a given structure.

    Parameters
    ----------
        pytree: The pytree that provides the structure.
        axis: The parameter axis is either -1, 0, or 1.  It controls the
          resulting shapes.
        example: If specified, cast the components to the matching dtype/weak_type,
          or else use the pytree leaf type if example is None.
        arr: The array to be unraveled.
    """
    leaves, treedef = tree_flatten(pytree)
    axis = axis % arr.ndim
    shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis + 1 :] for l in leaves]
    parts = _split(arr, np.cumsum(safe_map(np.size, leaves[:-1])), axis)
    reshaped_parts = [
        _possible_downcast(np.reshape(x, shape), leaf if example is None else example)
        for x, shape, leaf in zip(parts, shapes, leaves)
    ]
    return tree_unflatten(treedef, reshaped_parts)


def _split(x, indices, axis):
    if isinstance(x, np.ndarray):
        return np.split(x, indices, axis)
    else:
        return x._split(indices, axis)


def _jvp(fun: lu.WrappedFun, primals, tangents, has_aux=False):
    """Variant of jvp() that takes an lu.WrappedFun."""
    if not isinstance(primals, (tuple, list)) or not isinstance(
        tangents, (tuple, list)
    ):
        raise TypeError(
            "primal and tangent arguments to jax.jvp must be tuples or lists; "
            f"found {type(primals).__name__} and {type(tangents).__name__}."
        )

    ps_flat, tree_def = tree_flatten(primals)
    ts_flat, tree_def_2 = tree_flatten(tangents)
    if tree_def != tree_def_2:
        raise TypeError(
            "primal and tangent arguments to jax.jvp must have the same tree "
            f"structure; primals have tree structure {tree_def} whereas tangents have "
            f"tree structure {tree_def_2}."
        )
    for p, t in zip(ps_flat, ts_flat):
        if core.primal_dtype_to_tangent_dtype(_dtype(p)) != _dtype(t):
            raise TypeError(
                "primal and tangent arguments to jax.jvp do not match; "
                "dtypes must be equal, or in case of int/bool primal dtype "
                "the tangent dtype must be float0."
                f"Got primal dtype {_dtype(p)} and so expected tangent dtype "
                f"{core.primal_dtype_to_tangent_dtype(_dtype(p))}, but got "
                f"tangent dtype {_dtype(t)} instead."
            )
        if np.shape(p) != np.shape(t):
            raise ValueError(
                "jvp called with different primal and tangent shapes;"
                f"Got primal shape {np.shape(p)} and tangent shape as {np.shape(t)}"
            )

    if not has_aux:
        flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)
        out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
        out_tree = out_tree()
        return (
            tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents),
        )
    else:
        flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, tree_def)
        jvp_fun, aux = ad.jvp(flat_fun, has_aux=True)
        out_primals, out_tangents = jvp_fun.call_wrapped(ps_flat, ts_flat)
        out_tree, aux_tree = out_aux_trees()
        return (
            tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents),
            tree_unflatten(aux_tree, aux()),
        )


def _vjp(fun: lu.WrappedFun, *primals, has_aux=False):
    """Variant of vjp() that takes an lu.WrappedFun."""
    primals_flat, in_tree = tree_flatten(primals)
    for arg in primals_flat:
        dispatch.check_arg(arg)
    if not has_aux:
        flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
        out_primals, vjp = ad.vjp(flat_fun, primals_flat)
        out_tree = out_tree()
    else:
        flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, in_tree)
        out_primals, vjp, aux = ad.vjp(flat_fun, primals_flat, has_aux=True)
        out_tree, aux_tree = out_aux_trees()
    out_primal_avals = map(shaped_abstractify, out_primals)
    out_primal_py = tree_unflatten(out_tree, out_primals)
    vjp_py = Partial(
        partial(
            _vjp_pullback_wrapper, fun.__name__, out_primal_avals, (out_tree, in_tree)
        ),
        vjp,
    )
    if not has_aux:
        return out_primal_py, vjp_py
    else:
        return out_primal_py, vjp_py, tree_unflatten(aux_tree, aux)


def _vjp_pullback_wrapper(name, out_primal_avals, io_tree, fun, *py_args_):
    (py_args,) = py_args_
    in_tree_expected, out_tree = io_tree
    args, in_tree = tree_flatten(py_args)
    if in_tree != in_tree_expected:
        raise ValueError(
            f"unexpected tree structure of argument to vjp function: "
            f"got {in_tree}, but expected to match {in_tree_expected}"
        )
    for arg, aval in zip(args, out_primal_avals):
        ct_aval = shaped_abstractify(arg)
        try:
            ct_aval_expected = aval.to_tangent_type()
        except AttributeError:
            # https://github.com/jax-ml/jax/commit/018189491bde26fe9c7ade1213c5cbbad8bca1c6
            ct_aval_expected = aval.at_least_vspace()
        if not core.typecompat(
            ct_aval, ct_aval_expected
        ) and not _temporary_dtype_exception(ct_aval, ct_aval_expected):
            raise ValueError(
                "unexpected JAX type (e.g. shape/dtype) for argument to vjp function: "
                f"got {ct_aval.str_short()}, but expected "
                f"{ct_aval_expected.str_short()} because the corresponding output "
                f"of the function {name} had JAX type {aval.str_short()}"
            )
    ans = fun(*args)
    return tree_unflatten(out_tree, ans)


def _temporary_dtype_exception(a, a_) -> bool:
    if isinstance(a, core.ShapedArray) and isinstance(a_, core.ShapedArray):
        return a.shape == a_.shape and a_.dtype == dtypes.float0
    return False
