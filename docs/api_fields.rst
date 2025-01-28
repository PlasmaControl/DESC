=========================
Magnetic Fields and Coils
=========================

There are a number of ways for representing arbitrary magnetic fields in the lab
(:math:`R, \phi, Z`) frame. Their common characteristic is a ``compute_magnetic_field``
method allowing evaluation of :math:`\mathbf{B}` at points in space.

Magnetic Fields
***************
The ``desc.magnetic_fields`` module contains classes representing a number of standard
field configurations, as well as classes for combining and scaling these fields.
For interfacing with MAKEGRID type files, the ``SplineMagneticField`` class has a
``from_mgrid`` option allowing you to load an mgrid file and interpolate where needed.
All `MagneticField` objects also have a `save_mgrid` method to save field data so that
it can be read by VMEC and STELLOPT. They also have a `compute_Bnormal` method which accepts
a surface and computes the normal field strength on that surface.

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.SplineMagneticField
    desc.magnetic_fields.DommaschkPotentialField
    desc.magnetic_fields.MagneticFieldFromUser
    desc.magnetic_fields.ScalarPotentialField
    desc.magnetic_fields.ToroidalMagneticField
    desc.magnetic_fields.VerticalMagneticField
    desc.magnetic_fields.PoloidalMagneticField
    desc.magnetic_fields.ScaledMagneticField
    desc.magnetic_fields.SumMagneticField

There are also classes for representing a current potential on a winding surface:

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.CurrentPotentialField
    desc.magnetic_fields.FourierCurrentPotentialField

``desc.magnetic_fields`` also has a function which can solve a regularized least-squares problem
in order to find the optimal surface current distribution which minimizes the normal field on a
given plasma boundary, ``desc.magnetic_fields.solve_regularized_surface_current``, which can
be used in a similar way as the REGCOIL code:

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.solve_regularized_surface_current

There is also a class for representing omnigenous magnetic fields:

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.OmnigenousField

For analyzing the structure of magnetic fields, it is often useful to find the trajectories
of magnetic field lines, which can be done via ``desc.magnetic_fields.field_line_integrate``.

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.field_line_integrate

``desc.magnetic_fields`` also contains a utility function for reading output files from
the BNORM code:

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.read_BNORM_file


Coils
*****
``Coil`` objects in ``desc.coils`` are themselves subclasses of ``MagneticField``, allowing
them to be used anywhere that expects a magnetic field type. There are a number of parameterizations
based on the ``Curve`` classes defined in ``desc.geometry`` (which, since they are based on ``Curve``
classes, can also use the same ``Curve`` conversion methods to convert between coil representations):

.. autosummary::
    :toctree: _api/coils/
    :recursive:
    :template: class.rst

    desc.coils.FourierRZCoil
    desc.coils.FourierXYZCoil
    desc.coils.FourierPlanarCoil
    desc.coils.FourierXYCoil
    desc.coils.SplineXYZCoil

There are also objects for holding a collection of coils with efficient methods for
evaluating the combined field. A ``CoilSet`` must consist of members with the same
parameterization, while a ``MixedCoilSet`` can contain arbitrary types (including
another ``CoilSet``).

.. autosummary::
    :toctree: _api/coils/
    :recursive:
    :template: class.rst

    desc.coils.CoilSet
    desc.coils.MixedCoilSet

DESC ``CoilSet`` or ``MixedCoilSet`` objects can also be created from MAKEGRID-formatted coil text files via
the `from_makegrid_coilfile` method. They can also be saved in a MAKEGRID-formatted text file with
the `save_in_makegrid_format` method.

There are also utility functions for getting an initial guess for coil optimization using modular or
saddle coils:

.. autosummary::
    :toctree: _api/coils/
    :recursive:
    :template: class.rst

    desc.coils.initialize_modular_coils
    desc.coils.initialize_saddle_coils
