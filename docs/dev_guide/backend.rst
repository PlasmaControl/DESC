=======
Backend
=======


DESC uses JAX for faster compile times, automatic differentiation, and other scientific computing tools.
The purpose of ``backend.py`` is to determine whether DESC may take advantage of JAX and GPUs or default to standard ``numpy`` and CPUs.

JAX provides a ``numpy`` style API for array operations.
In many cases, to take advantage of JAX, one only needs to replace calls to ``numpy`` with calls to ``jax.numpy``.
A convenient way to do this is with the import statement ``import jax.numpy as jnp``.

Of course if such an import statement is used in DESC, and DESC is run on a machine where JAX is not installed, then a runtime error is thrown.
We would prefer if DESC still works on machines where JAX is not installed.
With that goal, in functions which can benefit from JAX, we use the following import statement: ``from desc.backend import jnp``.
``desc.backend.jnp`` is an alias to ``jax.numpy`` if JAX is installed and ``numpy`` otherwise.

While ``jax.numpy`` attempts to serve as a drop in replacement for ``numpy``, it imposes some constraints on how the code is written.
For example, ``jax.numpy`` arrays are immutable.
This means in-place updates to elements in arrays is not possible.
To update elements in ``jax.numpy`` arrays, memory needs to be allocated to create a new array with the updated element.
Similarly, JAX's JIT compilation requires code flow structures such as loops and conditionals to be written in a specific way.

The utility functions in ``desc.backend`` provide a simple interface to perform these operations.
