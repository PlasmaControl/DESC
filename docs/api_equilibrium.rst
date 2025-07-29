==========================================
Equilibrium, Surfaces, and Profile classes
==========================================


Equilibrium
***********
The ``Equilibrium`` is the core class representing an MHD equilibrium configuration.
An ``EquilibriaFamily`` is a ``list`` like object for storing multiple equilibria.

.. autosummary::
    :toctree: _api/equilibrium
    :recursive:
    :template: class.rst

    desc.equilibrium.Equilibrium
    desc.equilibrium.EquilibriaFamily

The ``Equilibrium`` class may be instantiated in a couple of ways in addition to providing inputs to its constructor.
- from an existing DESC or VMEC input file with its ``from_input_file`` method
- from a ``pyQSC`` ``Qsc``  or ``pyQIC`` ``Qic`` near-axis equilibrium with the ``Equilibrium``'s ``from_near_axis`` method


Geometry
********
The ``desc.geometry`` module contains important classes such as ``FourierRZToroidalSurface``
for representing the shape of the plasma boundary, as well as classes for representing
the magnetic axis, cross section, and various space curves.

.. autosummary::
   :toctree: _api/geometry/
   :recursive:
   :template: class.rst

    desc.geometry.FourierRZToroidalSurface
    desc.geometry.FourierRZCurve
    desc.geometry.FourierXYZCurve
    desc.geometry.FourierPlanarCurve
    desc.geometry.FourierXYCurve
    desc.geometry.SplineXYZCurve
    desc.geometry.ZernikeRZToroidalSection

The ``FourierRZToroidalSurface`` and the ``FourierRZCurve`` classes may be instantiated from an existing DESC or VMEC input file with their ``from_input_file`` method.


Profiles
********
``desc.profiles`` contains objects representing 1-D flux functions such as pressure,
current, rotational transform, temperature, or density. It is also possible to combine
profiles together by addition, multiplication, or scaling.

.. autosummary::
    :toctree: _api/profiles
    :recursive:
    :template: class.rst

    desc.profiles.PowerSeriesProfile
    desc.profiles.SplineProfile
    desc.profiles.MTanhProfile
    desc.profiles.TwoPowerProfile
    desc.profiles.HermiteSplineProfile
    desc.profiles.ScaledProfile
    desc.profiles.ProductProfile
    desc.profiles.SumProfile
    desc.profiles.PowerProfile


Utilities
*********
``desc.compat`` has utility functions for enforcing sign conventions or rescaling
equilibria to a given size and/or field strength.

.. autosummary::
    :toctree: _api/compat
    :recursive:

    desc.compat.ensure_positive_jacobian
    desc.compat.flip_helicity
    desc.compat.flip_theta
    desc.compat.rescale
    desc.compat.rotate_zeta
