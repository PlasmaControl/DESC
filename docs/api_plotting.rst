========
Plotting
========

There are a number of functions for plotting data from an ``Equilibrium`` or other DESC
object in the ``desc.plotting`` module. All functions apart from 3-D methods are based on
Matplotlib, and can plot to specified ``matplotlib.axes.Axes`` objects, or generate their own.
``plot_3d`` and ``plot_coils`` use the Plotly backend which can rotate/pan/zoom to see more detail.

1-D Line Plots
--------------
.. autosummary::
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_1d
    desc.plotting.plot_fsa


Plotting Flux Surfaces
----------------------
.. autosummary::
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_surfaces
    desc.plotting.plot_comparison
    desc.plotting.plot_boundary
    desc.plotting.plot_boundaries
    desc.plotting.poincare_plot


Contour Plots of 2-D data
-------------------------
.. autosummary::
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_2d
    desc.plotting.plot_section


3-D Plotting
------------
.. autosummary::
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_3d
    desc.plotting.plot_coils
    desc.plotting.plot_field_lines
    desc.plotting.plot_particle_trajectories


Specialized Plots for QS Metrics
--------------------------------
.. autosummary::
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_boozer_surface
    desc.plotting.plot_boozer_modes
    desc.plotting.plot_qs_error


Plotting energetic particle proxies
-----------------------------------
.. autosummary::
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_gammac


Misc Plotting Utilities
-----------------------
.. autosummary::
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_coefficients
    desc.plotting.plot_basis
    desc.plotting.plot_grid
    desc.plotting.plot_logo


Exporting Paraview Files
------------------------
Note that these utility functions are not regularly tested and maybe incompatible with newer version
of Paraview. Please see the README of the `desc.external` for more details.

.. autosummary::
    :toctree: _api/external
    :recursive:

    desc.external.paraview.export_coils_to_paraview
    desc.external.paraview.export_surface_to_paraview
    desc.external.paraview.export_volume_to_paraview
