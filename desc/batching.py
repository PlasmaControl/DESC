"""Batched operations."""

from functools import partial

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
from jax._src.numpy.vectorize import (
    _apply_excluded,
    _check_output_dims,
    _parse_gufunc_signature,
    _parse_input_dimensions,
)
from jax._src.util import wraps
from jax.tree_util import (
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_transpose,
)

from desc.backend import jax, jnp, scan, vmap
from desc.utils import errorif, identity

try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu


try:
    from jax._src.lax.control_flow.loops import _batch_and_remainder

except ImportError:
    # The old version of JAX doesn't have the required functions and will throw
    # an ImportError. We use a simpler version of _batch_and_remainder from an older JAX
    # version.
    def _batch_and_remainder(x, batch_size: int):
        """Taken from JAX 0.5.0.

        Function is the same down to JAX 0.4.31.
        """
        leaves, treedef = tree_flatten(x)

        scan_leaves = []
        remainder_leaves = []

        for leaf in leaves:
            num_batches = leaf.shape[0] // batch_size
            total_batch_elems = num_batches * batch_size
            scan_leaves.append(
                leaf[:total_batch_elems].reshape(
                    num_batches, batch_size, *leaf.shape[1:]
                )
            )
            remainder_leaves.append(leaf[total_batch_elems:])

        scan_tree = treedef.unflatten(scan_leaves)
        remainder_tree = treedef.unflatten(remainder_leaves)
        return scan_tree, remainder_tree


_unchunk = partial(tree_map, lambda y: y.reshape(-1, *y.shape[2:]))
_concat = partial(tree_map, lambda y1, y2: jnp.concatenate((y1, y2)))
_get_first_chunk = partial(tree_map, lambda x: x[0])


def _scan_append(f, x, reduction=None, carry_init_fun=None):
    """Evaluate f element-wise in x while appending the results."""

    def body(carry, x):
        return (), f(x)

    _, result = scan(body, (), x)
    return result


def _scan_reduce(
    f,
    x,
    reduction=None,
    carry_init_fun=partial(tree_map, lambda x: jnp.zeros_like(x)),
):
    """Evaluate f element-wise in x while reducing the results."""

    def body(carry, x):
        return reduction(carry, f(x)), None

    carry_init = carry_init_fun(jax.eval_shape(f, _get_first_chunk(x)))
    result, _ = scan(body, carry_init, x)
    return result


def _scanmap(fun, argnums=0, reduction=None, chunk_reduction=identity):
    """A helper function to wrap f with a scan_fun.

    Refrences
    ---------
    Adapted from the NetKet project.
    https://github.com/netket/netket/blob/master/netket/jax/_scanmap.py.

    The original copyright notice is as follows
    Copyright 2021 The NetKet Authors - All rights reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    """
    scan_fun = _scan_append if reduction is None else _scan_reduce

    def f_(*args, **kwargs):
        f_partial, dyn_args = argnums_partial(
            lu.wrap_init(fun, kwargs),
            argnums,
            args,
            require_static_args_hashable=False,
        )
        return scan_fun(
            lambda x: chunk_reduction(f_partial.call_wrapped(*x)),
            dyn_args,
            reduction,
        )

    return f_


def _evaluate_in_chunks(
    vmapped_fun,
    chunk_size,
    argnums,
    reduction=None,
    chunk_reduction=identity,
    *args,
    **kwargs,
):
    n_elements = tree_leaves(args[argnums[0]])[0].shape[0]
    if n_elements <= chunk_size:
        return chunk_reduction(vmapped_fun(*args, **kwargs))

    scan_x, remain_x = zip(
        *[
            _batch_and_remainder(a, chunk_size) if i in argnums else (a, a)
            for i, a in enumerate(args)
        ]
    )
    # Note that num_batches in _batch_and_remainder is always positive.
    scan_y = _scanmap(vmapped_fun, argnums, reduction, chunk_reduction)(
        *scan_x, **kwargs
    )
    if reduction is None:
        scan_y = _unchunk(scan_y)

    if n_elements % chunk_size == 0:
        return scan_y

    remain_y = chunk_reduction(vmapped_fun(*remain_x, **kwargs))
    if reduction is None:
        return _concat(scan_y, remain_y)

    return reduction(scan_y, remain_y)


def _parse_in_axes(in_axes):
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    errorif(
        not set(in_axes).issubset((0, None)),
        NotImplementedError,
        f"Only in_axes 0/None are currently supported, but got {in_axes}.",
    )
    argnums = tuple(i for i, axis in enumerate(in_axes) if axis is not None)
    return in_axes, argnums


def vmap_chunked(
    f,
    /,
    in_axes=0,
    *,
    chunk_size=None,
    reduction=None,
    chunk_reduction=identity,
):
    """Behaves like ``vmap`` but uses scan to chunk the computations in smaller chunks.

    Parameters
    ----------
    f : callable
        The function to be vectorised.
    in_axes : int or None
        The axes that should be scanned along. Only supports ``0`` or ``None``.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    reduction : callable or None
        Binary reduction operation.
        Should take two arguments and return one output, e.g. ``jnp.add``.
    chunk_reduction : callable
        Chunk-wise reduction operation.
        Should apply ``reduction`` along the mapped axis, e.g. ``jnp.add.reduce``.

    Returns
    -------
    f : callable
        A vectorised and chunked function.

    """
    in_axes, argnums = _parse_in_axes(in_axes)
    if isinstance(argnums, int):
        argnums = (argnums,)

    f = vmap(f, in_axes=in_axes)
    if chunk_size is None:
        return lambda *args, **kwargs: chunk_reduction(f(*args, **kwargs))
    return partial(
        _evaluate_in_chunks, f, chunk_size, argnums, reduction, chunk_reduction
    )


def batch_map(
    fun, fun_input, /, batch_size=None, *, reduction=None, chunk_reduction=identity
):
    """Compute ``chunk_reduction(fun(fun_input))`` in batches.

    This utility is like ``vmap_chunked`` except that ``fun`` is assumed to be
    vectorized natively. No JAX vectorization such as ``vmap`` is applied to the
    supplied function. This makes compilation faster and avoids the weaknesses of
    applying JAX vectorization, such as executing all branches of code conditioned on
    dynamic values. For example, this function would be useful for GitHub issue #1303

    Parameters
    ----------
    fun : callable
        Natively vectorized function.
    fun_input : pytree
        Data to split into batches to feed to ``fun``.
    batch_size : int or None
        Size of batches. If no batching should be done or the batch size is the
        full input then supply ``None``.
    reduction : callable or None
        Binary reduction operation.
        Should take two arguments and return one output, e.g. ``jnp.add``.
    chunk_reduction : callable
        Chunk-wise reduction operation.
        Should typically apply ``reduction`` along the mapped axis,
        e.g. ``jnp.add.reduce``.

    Returns
    -------
    fun_output
        Returns ``chunk_reduction(fun(fun_input))``.

    """
    return (
        chunk_reduction(fun(fun_input))
        if batch_size is None
        else _evaluate_in_chunks(
            fun, batch_size, (0,), reduction, chunk_reduction, fun_input
        )
    )


def batched_vectorize(pyfunc, *, excluded=frozenset(), signature=None, chunk_size=None):
    """Define a vectorized function with broadcasting and batching.

    Refrences
    ---------
    The original copyright notice is as follows
    Copyright 2018 The JAX Authors.
    Licensed under the Apache License, Version 2.0 (the "License");
    https://github.com/jax-ml/jax/blob/main/jax/_src/api.py.

    Notes
    -----
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
    errorif(
        any(not isinstance(exclude, (str, int)) for exclude in excluded),
        TypeError,
        "jax.numpy.vectorize can only exclude integer or string arguments, "
        "but excluded={!r}".format(excluded),
    )
    errorif(
        any(isinstance(e, int) and e < 0 for e in excluded),
        msg=f"excluded={excluded!r} contains negative numbers",
    )

    @wraps(pyfunc)
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
            errorif(
                any(input_core_dims[i] != () for i in none_args),
                msg=f"Cannot pass None at locations {none_args} with {signature=}",
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


def jacfwd_chunked(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    *,
    chunk_size=None,
):
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    Refrences
    ---------
    The original copyright notice is as follows
    Copyright 2018 The JAX Authors.
    Licensed under the Apache License, Version 2.0 (the "License");
    https://github.com/jax-ml/jax/blob/main/jax/_src/api.py.

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
            pushfwd = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap_chunked(pushfwd, chunk_size=chunk_size)(_std_basis(dyn_args))
            y = tree_map(lambda x: x[0], y)
            jac = tree_map(lambda x: jnp.moveaxis(x, 0, -1), jac)
        else:
            pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
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

    Refrences
    ---------
    The original copyright notice is as follows
    Copyright 2018 The JAX Authors.
    Licensed under the Apache License, Version 2.0 (the "License");
    https://github.com/jax-ml/jax/blob/main/jax/_src/api.py.

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
