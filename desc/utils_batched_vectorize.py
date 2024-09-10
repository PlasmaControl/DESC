"""Utility functions for the ``batched_vectorize`` function."""

from typing import Callable, Optional

from desc.backend import functools, jax, jnp

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
def _chunk(x, jac_chunk_size=None):
    # jac_chunk_size=None -> add just a dummy chunk dimension,
    #  same as np.expand_dims(x, 0)
    if x.ndim == 0:
        raise ValueError("x cannot be chunked as it has 0 dimensions.")
    n = x.shape[0]
    if jac_chunk_size is None:
        jac_chunk_size = n

    n_chunks, residual = divmod(n, jac_chunk_size)
    if residual != 0:
        raise ValueError(
            "The first dimension of x must be divisible by jac_chunk_size."
            + f"\n        Got x.shape={x.shape} but jac_chunk_size={jac_chunk_size}."
        )
    return x.reshape((n_chunks, jac_chunk_size) + x.shape[1:])


def _jac_chunk_size(x):
    b = set(map(lambda x: x.shape[:2], jax.tree_util.tree_leaves(x)))
    if len(b) != 1:
        raise ValueError(
            "The arrays in x have inconsistent jac_chunk_size or number of chunks"
        )
    return b.pop()[1]


def unchunk(x_chunked):
    """Merge the first two axes of an array (or a pytree of arrays).

    Parameters
    ----------
    x_chunked: an array (or pytree of arrays) of at least 2 dimensions

    Returns
    -------
    (x, chunk_fn) : tuple
        where x is x_chunked reshaped to (-1,)+x.shape[2:]
        and chunk_fn is a function which restores x given x_chunked

    """
    return _unchunk(x_chunked), functools.partial(
        _chunk, jac_chunk_size=_jac_chunk_size(x_chunked)
    )


def chunk(x, jac_chunk_size=None):
    """Split an array (or a pytree of arrays) into chunks along the first axis.

    Parameters
    ----------
        x: an array (or pytree of arrays)
        jac_chunk_size: an integer or None (default)
            The first axis in x must be a multiple of jac_chunk_size

    Returns
    -------
    (x_chunked, unchunk_fn): tuple
        - x_chunked is x reshaped to (-1, jac_chunk_size)+x.shape[1:]
          if jac_chunk_size is None then it defaults to x.shape[0], i.e. just one chunk
        - unchunk_fn is a function which restores x given x_chunked
    """
    return _chunk(x, jac_chunk_size), _unchunk


####

# The following section of this code is derived from the NetKet project
# https://github.com/netket/netket/blob/9881c9fb217a2ac4dc9274a054bf6e6a2993c519/
# netket/jax/_scanmap.py
#
# The original copyright notice is as follows
# Copyright 2021 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

_tree_add = functools.partial(jax.tree_util.tree_map, jax.lax.add)
_tree_zeros_like = functools.partial(
    jax.tree_util.tree_map, lambda x: jnp.zeros(x.shape, dtype=x.dtype)
)


# TODO put it somewhere
def _multimap(f, *args):
    try:
        print("one")
        return tuple(map(lambda a: f(*a), zip(*args)))
    except TypeError:
        print("two")
        return f(*args)


def scan_append(f, x):
    """Evaluate f element by element in x while appending and/or reducing the results.

    Parameters
    ----------
        f: a function that takes elements of the leading dimension of x
        x: a pytree where each leaf array has the same leading dimension
        append_cond: a bool (if f returns just one result) or a tuple of
                     bools (if f returns multiple values)
                     which indicates whether the individual result should
                     be appended or reduced
        op: a function to (pairwise) reduce the specified results. Defaults to a sum.
        zero_fun: a function which prepares the zero element of op for a given input
                  shape/dtype tree. Defaults to zeros.

    Returns
    -------
        The (tuple of) results corresponding to the output of f
        where each result is given by:
        if append_cond is True:
            a (pytree of) array(s) with leading dimension same as x,
            containing the evaluation of f at each element in x
        else (append_cond is False):
            a (pytree of) array(s) with the same shape as the corresponding
            output of f, containing the reduction over op of f evaluated at each x


    Example:

        import jax.numpy as jnp
        from netket.jax import scan_append_reduce

        def f(x):
             y = jnp.sin(x)
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)

        y, s, s2 = scan_append_reduce(f, x, (True, False, False))
        mean = s/N
        var = s2/N - mean**2
    """
    # TODO: different op for each result

    x0 = jax.tree_util.tree_map(lambda x: x[0], x)

    # special code path if there is only one element
    # to avoid having to rely on xla/llvm to optimize the overhead away
    if jax.tree_util.tree_leaves(x)[0].shape[0] == 1:
        return _multimap(lambda c, x: jnp.expand_dims(x, 0) if c else x, True, f(x0))

    # the original idea was to use pytrees,
    # however for now just operate on the return value tuple
    _get_append_part = functools.partial(_multimap, lambda c, x: x if c else None, True)

    carry_init = True

    def f_(carry, x):
        y = f(x)
        y_append = _get_append_part(y)
        return False, y_append

    _, res_append = jax.lax.scan(f_, carry_init, x, unroll=1)
    # reconstruct the result from the reduced and appended parts in the two trees
    return res_append  # _tree_select(res_append, res_op)


# TODO in_axes a la vmap?
def _scanmap(fun, scan_fun, argnums=0):
    """A helper function to wrap f with a scan_fun.

    Example
    -------
        import jax.numpy as jnp
        from functools import partial

        from desc.utils import _scanmap, scan_append_reduce

        scan_fun = partial(scan_append_reduce, append_cond=(True, False, False))

        @partial(_scanmap, scan_fun=scan_fun, argnums=1)
        def f(c, x):
             y = jnp.sin(x) + c
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)
        c = 1.


        y, s, s2 = f(c, x)
        mean = s/N
        var = s2/N - mean**2
    """

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
#
# The original copyright notice is as follows
# Copyright 2021 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


def _eval_fun_in_chunks(vmapped_fun, jac_chunk_size, argnums, *args, **kwargs):
    n_elements = jax.tree_util.tree_leaves(args[argnums[0]])[0].shape[0]
    n_chunks, n_rest = divmod(n_elements, jac_chunk_size)

    if n_chunks == 0 or jac_chunk_size >= n_elements:
        y = vmapped_fun(*args, **kwargs)
    else:
        # split inputs
        def _get_chunks(x):
            x_chunks = jax.tree_util.tree_map(
                lambda x_: x_[: n_elements - n_rest, ...], x
            )
            x_chunks = _chunk(x_chunks, jac_chunk_size)
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
    jac_chunk_size: Optional[int],
    argnums=0,
) -> Callable:
    """Takes a vmapped function and computes it in chunks."""
    if jac_chunk_size is None:
        return vmapped_fun

    if isinstance(argnums, int):
        argnums = (argnums,)
    return functools.partial(_eval_fun_in_chunks, vmapped_fun, jac_chunk_size, argnums)


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
    jac_chunk_size: Optional[int],
) -> Callable:
    """Behaves like jax.vmap but uses scan to chunk the computations in smaller chunks.

    Parameters
    ----------
        f: The function to be vectorised.
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        jac_chunk_size: The maximum size of the chunks to be used. If it is `None`,
            chunking is disabled


    Returns
    -------
        f: A vectorised and chunked function
    """
    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, jac_chunk_size, argnums)


def batched_vectorize(
    pyfunc, *, excluded=frozenset(), signature=None, jac_chunk_size=None
):
    """Define a vectorized function with broadcasting and batching.

    below is taken from JAX
    FIXME: change restof docstring
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
        pyfunc: function to vectorize.
        excluded: optional set of integers representing positional arguments for
        which the function will not be vectorized. These will be passed directly
        to ``pyfunc`` unmodified.
        signature: optional generalized universal function signature, e.g.,
        ``(m,n),(n)->(m)`` for vectorized matrix-vector multiplication. If
        provided, ``pyfunc`` will be called with (and expected to return) arrays
        with shapes given by the size of corresponding core dimensions. By
        default, pyfunc is assumed to take scalars arrays as input and output.
        jac_chunk_size: the size of the batches to pass to vmap. if 1, will only

    Returns
    -------
        Vectorized version of the given function.

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
                    vectorized_func, in_axes, jac_chunk_size=jac_chunk_size
                )
        result = vectorized_func(*squeezed_args)

        if not dims_to_expand:
            return result
        elif isinstance(result, tuple):
            return tuple(jnp.expand_dims(r, axis=dims_to_expand) for r in result)
        else:
            return jnp.expand_dims(result, axis=dims_to_expand)

    return wrapped
