"""Utility functions for the ``batched_vectorize`` function."""

import functools
from functools import partial
from typing import Callable, Optional

from jax._src.api import (
    _check_input_dtype_jacfwd,
    _check_input_dtype_jacrev,
    _check_output_dtype_jacfwd,
    _check_output_dtype_jacrev,
    _jacfwd_unravel,
    _jacrev_unravel,
    _jvp,
    _std_basis,
    _vjp,
)
from jax._src.api_util import _ensure_index, argnums_partial, check_callable
from jax._src.tree_util import tree_map, tree_structure, tree_transpose
from jax._src.util import wraps

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
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    *,
    chunk_size=None,
):
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
    holomorphic: Optional, bool.
        Indicates whether ``fun`` is promised to be holomorphic. Default False.
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
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
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
        tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if not has_aux:
            return jac_tree
        else:
            return jac_tree, aux

    return jacfun


def jacrev_chunked(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    allow_int=False,
    *,
    chunk_size=None,
):
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
    holomorphic: Optional, bool.
        Indicates whether ``fun`` is promised to be holomorphic. Default False.
    allow_int: Optional, bool.
        Whether to allow differentiating with respect to integer valued inputs. The
        gradient of an integer input will have a trivial vector-space dtype (float0).
        Default False.
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
        tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
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
