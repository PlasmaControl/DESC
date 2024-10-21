=============================
Adding new physics quantities
=============================


All calculation of physics quantities takes place in ``desc.compute``

As an example, we'll walk through the calculation of the radial component of the MHD
force :math:`F_\rho`

The full code is below:
::

    from desc.data_index import register_compute_fun

    @register_compute_fun(
        name="F_rho",
        label="F_{\\rho}",
        units="N \\cdot m^{-2}",
        units_long="Newtons / square meter",
        description="Covariant radial component of force balance error",
        dim=1,
        params=[],
        transforms={},
        profiles=[],
        coordinates="rtz",
        data=["p_r", "sqrt(g)", "B^theta", "B^zeta", "J^theta", "J^zeta"],
    )
    def _F_rho(params, transforms, profiles, data, **kwargs):
        data["F_rho"] = -data["p_r"] + data["sqrt(g)"] * (
            data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"]
        )
        return data

The decorator ``register_compute_fun`` tells DESC that the quantity exists and contains
metadata about the quantity. The necessary fields are detailed below:


* ``name``: A short, meaningful name that is used elsewhere in the code to refer to the
  quantity. This name will appear in the ``data`` dictionary returned by ``Equilibrium.compute``,
  and is also the argument passed to ``compute`` to calculate the quantity. IE,
  ``Equilibrium.compute("F_rho")`` will return a dictionary containing ``F_rho`` as well
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
  quantities would have ``coordinates="r"`` indicating it only depends on `:math:\rho`
* ``data``: What other physics quantities are needed to compute this quantity. In our
  example, we need the radial derivative of pressure ``p_r``, the Jacobian determinant
  ``sqrt(g)``, and contravariant components of current and magnetic field. These dependencies
  will be passed to the compute function as a dictionary in the ``data`` argument. Note
  that this only includes direct dependencies (things that are used in this function).
  For example, we need ``sqrt(g)``, which itself depends on ``e_rho``, etc. But we don't
  need to specify ``e_rho`` here, that dependency is determined automatically at runtime.
* ``kwargs``: If the compute function requires any additional arguments they should
  be specified like ``kwarg="thing"`` where the value is the name of the keyword argument
  that will be passed to the compute function. Most quantities do not take kwargs.


The function itself should have a signature of the form
::

    _foo(params, transforms, profiles, data, **kwargs)

Our convention is to start the function name with an underscore and have it be
something like the ``name`` attribute, but the name of the function doesn't actually matter
as long as it is registered.
``params``, ``transforms``, ``profiles``, and ``data`` are dictionaries containing the needed
dependencies specified by the decorator. The function itself should do any calculation
needed using these dependencies and then add the output to the ``data`` dictionary and
return it. The key in the ``data`` dictionary should match the ``name`` of the quantity.
