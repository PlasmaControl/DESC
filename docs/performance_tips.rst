================
Performance Tips
================

Caching the Compiled (Jitted) Code
----------------------------------
Although the compiled code is fast, it still takes time to compile. If you are running the same optimization, or some similar optimization, multiple times, you can save time by caching the compiled code. This automatically happens for a single session (for example, until you restart your kernel in Jupyter Notebook) but once you start using another session, the code will need to be recompiled. Fortunately, there is a way to bypass this. First create a cache directory (i.e. ``jax-caches``), and put the following code at the beginning of your script:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_compilation_cache_dir", "../jax-caches")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

This will use a directory called ``jax-caches`` in the parent directory of the script to store the compiled code. The ``jax_persistent_cache_min_entry_size_bytes`` and ``jax_persistent_cache_min_compile_time_secs`` parameters are set to -1 and 0, respectively, to ensure that all compiled code is cached. For more details on caching, refer to official JAX documentation `here <https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html#persistent-compilation-cache>`__.

Note: Updating JAX version might re-compile some previously cached code, and this might increase the cache size. Every once in a while, you might need to clear your cache directory.


GPU Memory Allocation
---------------------
By default, JAX will pre-allocate only 75% of the available GPU memory and hence, XLA compiler will do its optimizations based on that. This also allows other JAX/XLA processes to run on the GPU at the same time. However, if you are running a single process and want to pre-allocate all of the available memory (or some other amount of it), you can set the following environment variable:

.. code-block:: python

    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

We have seen that increasing the fraction can resolve Out of Memory (OOM) errors for many cases.

Alternatively, if you don't want to preallocate memory (for debugging or memory profiling, see `profiler <https://github.com/PlasmaControl/DESC/blob/master/tests/benchmarks/memory_benchmark_cpu.py>`__ and `script <https://github.com/PlasmaControl/DESC/blob/master/tests/benchmarks/memory_funcs.py>`__ files for example memory profiling), you can set the following environment variable to allow JAX to allocate memory as needed:

.. code-block:: python

    import os
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

Note that each environment variable has their pros and cons for XLA compiler, and can prevent different types of OOM errors. For example, the second option presented above is documented to have a slight performance hit due to allocation and de-allocation of GPU memory, hence we recommend it for only memory profiling. For more details on the memory allocation in JAX, we strongly recommend refering to the official JAX documentation `here <https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html>`__.


Reducing Memory Usage of Objective Jacobian Calculation
-------------------------------------------------------

During optimization, one of the most memory-intensive steps is the calculation of the Jacobian
of the cost function. This memory cost comes from attempting to calculate the entire Jacobian
matrix in one vectorized operation. However, this can be tuned between high memory usage but quick (default)
and low memory usage but slower with the ``jac_chunk_size`` keyword argument. By default, where this matters
is when creating the overall ``ObjectiveFunction`` to be used in the optimization. The Jacobian is a
matrix of shape [``obj.dim_f`` x ``x_opt.size``] (to see the size of ``x_opt``, one can check the printed ``Number of parameters:`` line, before optimization), and the calculation of the Jacobian is vectorized over
the columns (the ``x_opt.size`` dimension), where ``obj`` is the ``ObjectiveFunction`` object. Passing in the ``jac_chunk_size`` attribute allows one to split up
the vectorized computation into chunks of ``jac_chunk_size`` columns at a time, allowing one to compute the Jacobian in a slightly slower, but more memory-efficient manner. The syntax for creating an ``ObjectiveFunction`` with ``batched`` mode is:

.. code-block:: python

    obj1 = QuasisymmetryTripleProduct(eq)
    obj2 = MeanCurvature(eq)

    obj = ObjectiveFunction((obj1, obj2), deriv_mode="batched", jac_chunk_size=100)

The below Jacobian represents a ``batched`` execution with ``jac_chunk_size=2``, each color is computed sequantially.

.. math::

    \begin{equation}
        \begin{bmatrix}
            \color{red} \cfrac{\partial f_1}{\partial x_1} & \color{red}\cfrac{\partial f_1}{\partial x_2} & \cfrac{\partial f_1}{\partial x_3} & \cfrac{\partial f_1}{\partial x_4} & \color{blue}\cfrac{\partial f_1}{\partial x_5} & \color{blue}\cfrac{\partial f_1}{\partial x_6} & ... & \cfrac{\partial f_1}{\partial x_n}\\
            \color{red}\cfrac{\partial f_2}{\partial x_1} & \color{red}\cfrac{\partial f_2}{\partial x_2} & \cfrac{\partial f_2}{\partial x_3} & \cfrac{\partial f_2}{\partial x_4} & \color{blue}\cfrac{\partial f_2}{\partial x_5} & \color{blue}\cfrac{\partial f_2}{\partial x_6}& ... & \cfrac{\partial f_2}{\partial x_n}\\
            . & . & .& & .\\
            . & . & .& & .\\
            \color{red}\cfrac{\partial f_m}{\partial x_1} & \color{red}\cfrac{\partial f_m}{\partial x_2} & \cfrac{\partial f_m}{\partial x_3} & \cfrac{\partial f_m}{\partial x_4} & \color{blue}\cfrac{\partial f_m}{\partial x_5} & \color{blue}\cfrac{\partial f_m}{\partial x_6}& ... & \cfrac{\partial f_m}{\partial x_n}\\
        \end{bmatrix}
    \end{equation}


The memory usage of the Jacobian calculation is
``memory usage = m0 + m1*jac_chunk_size``: the smaller the chunk size, the less memory the Jacobian calculation
will require (with some baseline memory usage). The time to compute the Jacobian is roughly ``t=t0 +t1/jac_chunk_size``
with some baseline time, so the larger the ``jac_chunk_size``, the faster the calculation takes,
at the cost of requiring more memory. A ``jac_chunk_size`` of 1 corresponds to the least memory intensive,
but slowest method of calculating the Jacobian. If ``jac_chunk_size="auto"``, it will default to a size
that should make the calculation fit in memory based on a heuristic estimate of the Jacobian memory usage.

If ``deriv_mode="blocked"`` is specified when the ``ObjectiveFunction`` is created, then the Jacobian will
be calculated individually for each of the sub-objectives inside of the ``ObjectiveFunction``, and in that case
the ``jac_chunk_size`` of the individual ``_Objective`` objects inside of the ``ObjectiveFunction`` will be used. This can be thought as dividing the Jacobian into blocks as shown below, and then using the column chunking for each block.

.. math::

    \begin{equation}
        \begin{bmatrix}
        \color{red}\dfrac{\partial f_1}{\partial x_1} & \color{red}\dfrac{\partial f_1}{\partial x_2} & \color{red}\cdots & \color{red}\dfrac{\partial f_1}{\partial x_n}\\
        \color{red}\dfrac{\partial f_2}{\partial x_1} & \color{red}\dfrac{\partial f_2}{\partial x_2} & \color{red}\cdots & \color{red}\dfrac{\partial f_2}{\partial x_n}\\
        \dfrac{\partial f_3}{\partial x_1} & \dfrac{\partial f_3}{\partial x_2} & \cdots & \dfrac{\partial f_3}{\partial x_n}\\
        \dfrac{\partial f_4}{\partial x_1} & \dfrac{\partial f_4}{\partial x_2} & \cdots & \dfrac{\partial f_4}{\partial x_n}\\
        \dfrac{\partial f_5}{\partial x_1} & \dfrac{\partial f_5}{\partial x_2} & \cdots & \dfrac{\partial f_5}{\partial x_n}\\
        \color{blue}\dfrac{\partial f_6}{\partial x_1} & \color{blue}\dfrac{\partial f_6}{\partial x_2} & \color{blue}\cdots & \color{blue}\dfrac{\partial f_6}{\partial x_n}
        \end{bmatrix}
    \end{equation}


The syntax for this is,

.. code-block:: python

    obj1 = QuasisymmetryTripleProduct(eq, jac_chunk_size=100)
    obj2 = MeanCurvature(eq, jac_chunk_size=2000)

    # deriv_mode="blocked" will be chosen automatically if any of the sub-objectives has a jac_chunk_size
    obj = ObjectiveFunction((obj1, obj2), deriv_mode="blocked")

The Jacobian will be calculated with a ``jac_chunk_size=100`` for the quasisymmetry part and a ``jac_chunk_size=2000`` for the curvature part, then the full Jacobian
will be formed as a blocked matrix with the individual Jacobians of these two objectives.


.. attention:: How to choose the ``jac_chunk_size``?

    A good starting point for ``jac_chunk_size`` is to set it to a value that allows the Jacobian to fit in memory, and then adjust it based on the performance of the optimization. One can choose it by looking at the printed output of the optimization, which will show the Jacobian size by these 2 lines,

    .. code-block:: bash

        Number of parameters: 250
        Number of objectives: 3000

    The Jacobian size is 3000 x 250, where 250 is the optimization variable (aka. reduced state vector) and 3000 is the number of objectives (for example, for a grid with 1500 nodes, ``ForceBalance`` objective will have 2x1500=3000 rows). **When considering the ``jac_chunk_size``, one should use a value smaller than ``Number of parameters``, otherwise chunking will do nothing!**


.. tip::

    Several other functions in DESC also have ``chunk_size`` or similar keywords arguments, which can be used to reduce memory usage.
