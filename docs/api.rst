=================
API Documentation
=================



Basis
*****

.. autosummary::
   :toctree: api/basis/
   :recursive:
	     
    desc.basis.PowerSeries
    desc.basis.FourierSeries
    desc.basis.DoubleFourierSeries
    desc.basis.FourierZernikeBasis

Derivatives
***********
Note that the ``derivative`` module also exposes the ``Derivative`` class, which is an alias for ``AutoDiffDerivative`` if JAX is installed, or ``FiniteDiffDerivative`` if not.

.. autosummary::
    :toctree: api/derivatives
    :recursive:

    desc.derivatives.AutoDiffDerivative
    desc.derivatives.FiniteDiffDerivative


Equilibrium
***********

.. autosummary:: 
    :toctree: api/equilibrium
    :recursive:

    desc.equilibrium.Equilibrium
    desc.equilibrium.EquilibriaFamily

    
Grid
****

.. autosummary::
   :toctree: api/grid/
   :recursive:

    desc.grid.Grid
    desc.grid.LinearGrid
    desc.grid.ConcentricGrid


IO
***

.. autosummary::
   :toctree: api/io/
   :recursive:
      
   desc.io.InputReader
    
Objective Functions
*******************

.. autosummary::
    :toctree: api/objective_funs
    :recursive:

    desc.objective_funs.get_objective_function
    desc.objective_funs.ForceErrorNodes
    desc.objective_funs.EnergyVolIntegral

Optimize
********

.. autosummary:: 
   :toctree: api/optimize
   :recursive:

   desc.optimize.Optimizer
   desc.optimize.fmintr
   desc.optimize.lsqtr


Perturbations
*************

.. autosummary:: 
    :toctree: api/perturbations
    :recursive:

    desc.perturbations.perturb

    
Plotting
********

.. autosummary:: 
    :toctree: api/plotting
    :recursive:

    desc.plotting.plot_1d
    desc.plotting.plot_2d    
    desc.plotting.plot_3d
    desc.plotting.plot_surfaces
    desc.plotting.plot_section   
   

Transform
*********

.. autosummary::
   :toctree: api/transform/
   :recursive:
      
   desc.transform.Transform

   
Boundary Conditions
*******************

.. automodule:: desc.boundary_conditions
    :members:
    :undoc-members:
    :inherited-members:    




VMEC
****

.. automodule:: desc.vmec
    :members:
    :undoc-members:
