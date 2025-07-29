==================
Saving and Loading
==================


IO
***
Nearly all objects in DESC have an ``object.save`` method to save the object as either
an HDF5 (``.h5``) or Pickle ``.pkl`` format. The function ``desc.io.load`` can then
be used to load a saved object and all of its data.

.. autosummary::
    :toctree: _api/io/
    :recursive:
    :template: class.rst

    desc.io.load


Examples
********
The ``desc.examples`` module contains a number of pre-computed equilibrium solutions to
well known configurations or benchmark problems.

.. autosummary::
    :toctree: _api/examples
    :recursive:

    desc.examples.get
    desc.examples.listall


VMEC
****
The ``desc.vmec.VMECIO`` class has a number of methods for interacting with VMEC
equilibria, such as loading a VMEC solution and converting to a DESC ``Equilibrium``,
saving a DESC solution in the VMEC ``wout_*.nc`` format, and plotting comparisons
between solutions.

.. autosummary::
    :toctree: _api/vmec/
    :recursive:
    :template: class.rst

    desc.vmec.VMECIO
