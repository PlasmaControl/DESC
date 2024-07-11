Adding new physics quantities
-----------------------------

.. role:: console(code)
   :language: console

All calculation of physics quantities takes place in ``desc.compute``

As an example, we'll walk through the calculation of the contravariant radial
component of the plasma current density :math:`J^\rho`

The full code is below:
::

    from desc.data_index import register_compute_fun

    @register_compute_fun(
        name="J^rho",
        label="J^{\\rho}",
        units="A \\cdot m^{-3}",
        units_long="Amperes / cubic meter",
        description="Contravariant radial component of plasma current density",
        dim=1,
        params=[],
        transforms={},
        profiles=[],
        coordinates="rtz",
        data=["sqrt(g)", "B_zeta_t", "B_theta_z"],
        axis_limit_data=["sqrt(g)_r", "B_zeta_rt", "B_theta_rz"],
        parameterization="desc.equilibrium.equilibrium.Equilibrium",
    )
    def _J_sup_rho(params, transforms, profiles, data, **kwargs):
        # At the magnetic axis,
        # ∂_θ (𝐁 ⋅ 𝐞_ζ) - ∂_ζ (𝐁 ⋅ 𝐞_θ) = 𝐁 ⋅ (∂_θ 𝐞_ζ - ∂_ζ 𝐞_θ) = 0
        # because the partial derivatives commute. So 𝐉^ρ is of the indeterminate
        # form 0/0 and we may compute the limit as follows.
        data["J^rho"] = (
            transforms["grid"].replace_at_axis(
                (data["B_zeta_t"] - data["B_theta_z"]) / data["sqrt(g)"],
                lambda: (data["B_zeta_rt"] - data["B_theta_rz"]) / data["sqrt(g)_r"],
            )
        ) / mu_0
        return data

The decorator ``register_compute_fun`` tells DESC that the quantity exists and contains
metadata about the quantity. The necessary fields are detailed below:


* ``name``: A short, meaningful name that is used elsewhere in the code to refer to the
  quantity. This name will appear in the ``data`` dictionary returned by ``Equilibrium.compute``,
  and is also the argument passed to ``compute`` to calculate the quantity. IE,
  ``Equilibrium.compute("J^rho")`` will return a dictionary containing ``J^rho`` as well
  as all the intermediate quantities needed to compute it. General conventions are that
  covariant components of a vector are called ``X_rho`` etc, contravariant components
  ``X^rho`` etc, and derivatives by a single character subscript, ``X_r`` etc for :math:`\partial_{\rho} X`
* ``label``: a LaTeX style label used for plotting.
* ``units``: SI units of the quantity in LaTeX format
* ``units_long``: SI units of the quantity, spelled out
* ``description``: A short description of the quantity
* ``dim``: Dimensionality of the quantity. Vectors (such as magnetic field) have ``dim=3``,
  local scalar quantities (such as vector components, pressure, volume element, etc)
  have ``dim=1``, global scalars (such as total volume, aspect ratio, etc) have ``dim=0``
* ``params``: list of strings of ``Equilibrium`` parameters needed to compute the quantity
  such as ``R_lmn``, ``Z_lmn`` etc. These will be passed into the compute function as a
  dictionary in the ``params`` argument. Note that this only includes direct dependencies
  (things that are used in this function). For most quantities, this will be an empty list.
  For example, if the function relies on ``R_t``, this dependency should be specified as a
  data dependency (see below), while the function to compute ``R_t`` itself will depend on
  ``R_lmn``
* ``transforms``: a dictionary of what ``transform`` objects are needed, with keys being the
  quantity being transformed (``R``, ``p``, etc) and the values are a list of derivative
  orders needed in ``rho``, ``theta``, ``zeta``. IE, if the quantity requires
  :math:`R_{\rho}{\zeta}{\zeta}`, then ``transforms`` should be ``{"R": [[1, 0, 2]]}``
  indicating a first derivative in ``rho`` and a second derivative in ``zeta``. Note that
  this only includes direct dependencies (things that are used in this function). For most
  quantities this will be an empty dictionary.
* ``profiles``: List of string of ``Profile`` quantities needed, such as ``pressure`` or
  ``iota``. Note that this only includes direct dependencies (things that are used in
  this function). For most quantities this will be an empty list.
* ``coordinates``: String denoting which coordinate the quantity depends on. Most will be
  ``"rtz"`` indicating it is a function of :math:`\rho, \theta, \zeta`. Profiles and flux surface
  quantities would have ``coordinates="r"`` indicating it only depends on :math:`\rho`
* ``data``: What other physics quantities are needed to compute this quantity. In our
  example, we need the poloidal (theta) derivative of the covariant toroidal (zeta) component
  of the magnetic field ``B_zeta_t``, the toroidal derivative of the covariant poloidal
  component of the magnetic field ``B_theta_z``, and the 3-D volume Jacobian determinant
  ``sqrt(g)``. These dependencies will be passed to the compute function as a dictionary
  in the ``data`` argument. Note that this only includes direct dependencies (things that
  are used in this function). For example, we need ``sqrt(g)``, which itself depends on
  ``e_rho``, etc. But we don't need to specify those sub-dependencies here.
* ``axis_limit_data``: Some quantities require additional work to compute at the
  magnetic axis. A Python lambda function is used to lazily compute the magnetic
  axis limits of these quantities. These lambda functions are evaluated only when
  the computational grid has a node on the magnetic axis to avoid potentially
  expensive computations. In our example, we need the radial derivatives of each
  the quantities mentioned above to evaluate the magnetic axis limit. These dependencies
  are specified in ``axis_limit_data``. The dependencies specified in this list are
  marked to be computed only when there is a node at the magnetic axis.
* ``parameterization``: what sorts of DESC objects is this function for. Most functions
  will just be for ``Equilibrium``, but some methods may also be for ``desc.geometry.core.Curve``,
  or specific types eg ``desc.geometry.curve.FourierRZCurve``. If a quantity is computed differently
  for a subclass versus a superclass, then one may define a compute function for the superclass
  (e.g. for ``desc.geometry.Curve``) which will be used for that class and any of its subclasses,
  and then if a specific subclass requires a different method, one may define a second compute function for
  the same quantity, with a parameterization for that subclass (e.g. ``desc.geometry.curve.SplineXYZCurve``).
  See the compute definitions for the ``length`` quantity in ``compute/_curve.py`` for an example of this,
  which is similar to the inheritance structure of Python classes.
* ``kwargs``: If the compute function requires any additional arguments they should
  be specified like ``kwarg="description"`` where ``kwarg`` is replaced by the actual
  keyword argument, and ``"description"`` is a string describing what it is.
  Most quantities do not take kwargs.


The function itself should have a signature of the form
::

    _foo(params, transforms, profiles, data, **kwargs)

Our convention is to start the function name with an underscore and have the it be
something like the ``name`` attribute, but name of the function doesn't actually matter
as long as it is registered.
``params``, ``transforms``, ``profiles``, and ``data`` are dictionaries containing the needed
dependencies specified by the decorator. The function itself should do any calculation
needed using these dependencies and then add the output to the ``data`` dictionary and
return it. The key in the ``data`` dictionary should match the ``name`` of the quantity.

Once a new quantity is added to the ``desc.compute`` module, there are two final steps involving the testing suite which must be checked.
The first step is implementing the correct axis limit, or marking it as not finite or not implemented.
We can check whether the axis limit currently evaluates as finite by computing the quantity on a grid with nodes at the axis.
::

    from desc.examples import get
    from desc.grid import LinearGrid
    import numpy as np

    eq = get("HELIOTRON")
    grid = LinearGrid(rho=np.array([0.0]), M=4, N=8, axis=True)
    new_quantity = eq.compute(name="new_quantity_name", grid=grid)["new_quantity_name"]
    print(np.isfinite(new_quantity).all())

if ``False`` is printed, then the limit of the quantity does not evaluate as finite which can be due to 3 reasons:


* The limit is actually not finite, in which case please add the new quantity to the ``not_finite_limits`` set in ``tests/test_axis_limits.py``.
* The new quantity has an indeterminate expression at the magnetic axis, in which case you should try to implement the correct limit as done in the example for ``J^rho`` above.
  If you wish to skip implementing the limit at the magnetic axis, please add the new quantity to the ``not_implemented_limits`` set in ``tests/test_axis_limits.py``.
* The new quantity includes a dependency whose limit at the magnetic axis has not been implemented.
  The tests automatically detect this, so no further action is needed from developers in this case.


The second step is to run the ``test_compute_everything`` test located in the ``tests/test_compute_everything.py`` file.
This can be done with the command :console:`pytest tests/test_compute_everything.py`.
This test is a regression test to ensure that compute quantities in each new update of DESC do not differ significantly
from previous versions of DESC.
Since the new quantity did not exist in previous versions of DESC, one must run this test
and commit the outputted ``tests/inputs/master_compute_data.pkl`` file which is updated automatically when a new quantity is detected.
