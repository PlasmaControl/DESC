===================================
Configuration.py and Equilibrium.py
===================================

To construct an equilibrium, the relevant parameters that decide the plasma state need to created and passed into the constructor of the ``Equilibrium`` class.
See ``Initializing an Equilibrium`` for a walk-through of this process.

These parameters are then automatically passed into the ``Configuration`` class, which is the abstract base class for equilibrium objects.
Almost all the work to initialize an equilibrium object is done in ``configuration.py``, while ``equilibrium.py`` serves as a wrapper class with methods that call routines to solve and optimize the equilibrium.

The attributes of a ``Configuration`` object can be organized into three groups.

The first group has parameters relavant to generating the basis functions based on the specified device parameters like the number of field periods and grid parameters that determine resolution and type of spectral indexing of ``ansi`` or ``fringe``.

The second group of attributes are related to device geometry.
The ``surface`` of an equilbrium specifies the plasma boundary condition in terms of either parameterizing the last closed flux surface with a ``FourierRZToroidalSurface`` or a poincare cross-section of the toroidal coordinate with a ``ZernikeRZToroidalSection``.
An initial guess for the magnetic axis can help the equilibrium solver find better equilbria quicker.
This can be specified with a ``Curve`` object as the parameter for the ``axis`` field of the equilibrium constructor.
If the magenetic axis is not specified, then the center of the surface is used as the initial guess.

The third group of attributes are profile quantities such as pressure, rotational transform, toroidal current, and kinetic profiles.
The pressure profile and rotational transform are required to specify the plasma state.
If the pressure profile is not known kinetic profiles can be given instead.
Similarly, if the rotational transform is not known, the toroidal current profile can be given instead.

The purpose of the ``__init__`` method is to assign these attributes while ensuring that the equilibrium is nether underdetermined (missing parameters) nor overdetermined (too many conflicting parameters, e.g. pressure and kinetic profiles).

Once an equilbrium is initialized, it can be solved with ``equilibrium.solve()`` and later optimized with ``equilbrium.optimize()``.
Each of these methods starts an optimization routine to either minimize the force balance residual errors or a some other specified objective function.
T``Configuration`` class also contains the methods to compute quantities on the equilbrium.

Once an equilibrium is optimized, we can compute quantities on this equilbrium with ``equilibrium.compute(names=names, grid=grid)`` where ``names`` is a list of strings that denote the names of the quantities as discussed in ``Adding new physics quantities``.
This method calls the ``compute`` method in ``Configuration.py``.

Some quantities require certain grids to ensure they computed accurately.
In particular, quantities which rely on surface averages operations should use grids that span the entire surface evenly.
Many profiles are functions of the flux surface label and likely to rely on a surface average operation.
Similarly, volume averages are global quantities that should be computed on a quadrature grid to exactly integrate the Fourier-Zernike basis functions.
Hence, regardless of the grid specified by the user, if a flux surface function or a global quantity is a dependency of the specified parameter to be computed, these dependencies are first computed on ``LinearGrid`` and ``QuadratureGrid``, respectively.
The arrays which store these quantities are then manipulated to be broadcastable with quantities computed on the grid specified by the user and passed in as dependencies to the compute functions.
