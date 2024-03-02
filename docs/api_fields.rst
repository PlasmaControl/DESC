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
it can be read by VMEC and STELLOPT.

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.SplineMagneticField
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
based on the ``Curve`` classes defined in ``desc.geometry``:

.. autosummary::
    :toctree: _api/coils/
    :recursive:
    :template: class.rst

    desc.coils.FourierRZCoil
    desc.coils.FourierXYZCoil
    desc.coils.FourierPlanarCoil
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
