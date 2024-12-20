================
Performance Tips
================

Caching the Compiled (Jitted) Code
----------------------------------
Although the compiled code is fast, it still takes time to compile. If you are running the same optimization, or some similar optimization, multiple times, you can save time by caching the compiled code. This automatically happens for a single session (for example, until you restart your kernel in Jupyter Notebook) but once you start using another session, the code will need to be recompiled. Fortunately, there is a way to bypass this. First create a cache directory, and put the following code at the beginning of your script:

.. code-block:: python
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_compilation_cache_dir", "../jax-caches")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

This will create a cache directory called ``jax-caches`` in the parent directory of the script. The ``jax_persistent_cache_min_entry_size_bytes`` and ``jax_persistent_cache_min_compile_time_secs`` parameters are set to -1 and 0, respectively, to ensure that all compiled code is cached. For more details on caching, refer to official JAX documentation https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html#persistent-compilation-cache.

Note: Updating JAX version might re-compile some previously cached code, and thi might increase the cache size. Every once in a while, you might need to clear your cache directory.


Reducing Memory Size of Objective Jacobian Calculation
------------------------------------------------------

During optimization, one of the most memory-intensive steps is the calculation of the Jacobian
of the cost function. This memory cost comes from attempting to calculate the entire Jacobian
matrix in one vectorized operation. However, this can be tuned between high memory usage but quick (default)
and low memory usage but slower with the ``jac_chunk_size`` keyword argument. By default, where this matters
is when creating the overall ``ObjectiveFunction`` to be used in the optimization (where by default ``deriv_mode="batched"``). The Jacobian is a
matrix of shape [``obj.dim_f`` x ``obj.dim_x``], and the calculation of the Jacobian is vectorized over
the columns (the ``obj.dim_x`` dimension), where ``obj`` is the ``ObjectiveFunction`` object. Passing in the ``jac_chunk_size`` attribute allows one to split up
the vectorized computation into chunks of ``jac_chunk_size`` columns at a time, allowing one to compute the Jacobian
in a slightly slower, but more memory-efficient manner. The memory usage of the Jacobian calculation is
``memory usage = m0 + m1*jac_chunk_size``: the smaller the chunk size, the less memory the Jacobian calculation
will require (with some baseline memory usage). The time to compute the Jacobian is roughly ``t=t0 +t1/jac_chunk_size``
with some baseline time, so the larger the ``jac_chunk_size``, the faster the calculation takes,
at the cost of requiring more memory. A ``jac_chunk_size`` of 1 corresponds to the least memory intensive,
but slowest method of calculating the Jacobian. If ``jac_chunk_size="auto"``, it will default to a size
that should make the calculation fit in memory based on a heuristic estimate of the Jacobian memory usage.

If ``deriv_mode="blocked"`` is specified when the ``ObjectiveFunction`` is created, then the Jacobian will
be calculated individually for each of the sub-objectives inside of the ``ObjectiveFunction``, and in that case
the ``jac_chunk_size`` of the individual ``_Objective`` objects inside of the ``ObjectiveFunction`` will be used.
For example, if ``obj1 = QuasisymmetryTripleProduct(eq, jac_chunk_size=100)``, ``obj2 = MeanCurvature(eq, jac_chunk_size=2000)``
and ``obj = ObjectiveFunction((obj1, obj2), deriv_mode="blocked")``, then the Jacobian will be calculated with a
``jac_chunk_size=100`` for the quasisymmetry part and a ``jac_chunk_size=2000`` for the curvature part, then the full Jacobian
will be formed as a blocked matrix with the individual Jacobians of these two objectives.
