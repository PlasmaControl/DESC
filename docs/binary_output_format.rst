By default, DESC saves to the hdf5 self-describing binary format [1]_, [2]_
(".h5" file extension). The file contains all information necessary to reconstruct
the python object that was saved. The file structure will depend slightly upon which
object was saved (all objects in DESC have a ``save`` method), but generally all
objects will contain the following fields:

- ``__class__`` : name of the python class of the object
- ``__version__`` : which version of DESC created the file

Other fields in the hdf5 file will depend on the type of object, with each attribute
of the python object being stored as a data field in the hdf5 file (specifically,
all attributes listed in a classes ``_io_attrs_`` property). These may be nested
objects, such as an EquilibriaFamily that has the attribute ``_equilibria`` which is
a list containing the individual equilibria in the family, and is then indexed by
number, for example ``../_equilibria/0``, or an Equilibrium which contains objects
for the pressure and iota profiles and spectral bases for R, Z and :math:`\lambda`.
In general, all names in the hdf5 file will mirror the python attributes, but with a
leading underscore. For example, ``eq.R_lmn`` will be stored as ``../_R_lmn``.

Below are some examples of common data items and where to find them within a saved Equilibrium:

* Spectral coefficients for R, Z, :math:`\lambda`:

  - ``/_R_lmn``
  - ``/_Z_lmn``
  - ``/_L_lmn``

* Mode numbers corresponding to spectral coefficients:

  - ``/_R_basis/_modes``
  - ``/_Z_basis/_modes``
  - ``/_L_basis/_modes``

* Profile coefficients:

  - ``/_pressure/_params``
  - ``/_iota/_params``

* Number of field periods:

  - ``/_NFP``

* Total toroidal flux:

  - ``/_Psi``

* Boundary Fourier coefficients:

  - ``/_surface/_R_lmn``
  - ``/_surface/_Z_lmn``

* Boundary mode numbers:

  - ``/_surface/_R_basis/_modes``
  - ``/_surface/_Z_basis/_modes``

A saved hdf5 file can be loaded with ``desc.io.load``, and it will return a
reconstruction of the object(s) saved within. Some data may not be saved (fields not
in ``_io_attrs_``), generally things that require large amounts of memory but are
trivially recomputable (i.e., transform matrices).

DESC also has the option of saving to python's standard binary format, known as
``pickle`` [3]_ (".pkl" file extension). The internal structure of this is somewhat
more complicated than hdf5 and is meant only for saving and loading data between
python environments.

The developers strive to maintain backwards and forwards compatibility with saved data,
so that equilibria computed in an older version of the code can be loaded in a newer
version and vice versa, but we make no guarantees at this time.

.. [1] https://portal.hdfgroup.org/display/HDF5/HDF5
.. [2] https://docs.h5py.org/en/stable/index.html
.. [3] https://docs.python.org/3/library/pickle.html
