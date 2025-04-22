Optimizers Supported
####################

The table below contains a list of different optimzers that DESC is interfaced to.
New optimizers can be added easily, see :ref:`adding-optimizers`.

  * **Name** : Name of the optimizer method. Pass this string to `desc.optimze.Optimizer` to use the method.
  * **Scalar** : Whether the method assumes a scalar residual, or a vector of residuals for least squares.
  * **Equality constraints** : Whether the method handles equality constraints.
  * **Inequality constraints** : Whether the method handles inequality constraints.
  * **Stochastic** : Whether the method can handle noisy objectives.
  * **Hessian** : Whether the method requires calculation of the full hessian matrix.
  * **GPU** : Whether the method supports running on GPU
  * **Description** : Short description of the optimizer method.

In addition to the listed optimizers, DESC also includes ``proximal-`` prefix to them (i.e. ``proximal-lsq-exact``), which allows for proximal optimization. This is useful for optimization problems where ``ForceBalance`` is in the constraints. Any optimizer with ``proximal-`` prefix solves the equilibrium for each optimization step to preserve the MHD force balance. Iterations of these optimizers will take longer than the original optimizers, but they will be more robust to large perturbations in the initial guess. For example, after an optimization with ``lsq-auglag`` optimizer, the final result will not be good force balance, hence we suggest solving equilibrium again (see `Advanced QS Optimization <https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/advanced_optimization.html#Constrained-Optimization>`__) The proximal optimizers are not listed in the table below, but they can be used by passing ``proximal-OptimizerName``, or this conversion will happen automatically if the ``ForceBalance`` constraint is used.

.. csv-table:: List of Optimizers
   :file: optimizers.csv
   :widths: 15, 10, 10, 10, 10, 10, 10, 60
   :header-rows: 1
   :class: longtable
