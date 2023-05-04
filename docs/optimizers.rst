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


.. csv-table:: List of Optimizers
   :file: optimizers.csv
   :widths: 15, 10, 10, 10, 10, 10, 10, 60
   :header-rows: 1
   :class: longtable
