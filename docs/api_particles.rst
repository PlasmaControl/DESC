==================
Particles
==================


Trajectory Models
*****************
In DESC, particle tracing can be performed using various trajectory models. These models
are designed to simulate the motion of charged particles in magnetic fields either from
``Equilibrium`` or ``_MagneticField`` classes. The available trajectory models include:

.. autosummary::
    :toctree: _api/particles/
    :recursive:
    :template: class.rst

    desc.particles.VacuumGuidingCenterTrajectory


Particle Initializers
*********************
Particle initializers are used to define the initial conditions for particle tracing.
One can manually initialize particles or use random particle generation on given geometry.

.. autosummary::
    :toctree: _api/particles/
    :recursive:
    :template: class.rst

    desc.particles.CurveParticleInitializer
    desc.particles.ManualParticleInitializerFlux
    desc.particles.ManualParticleInitializerLab
    desc.particles.SurfaceParticleInitializer


Particle Tracing
****************
The main function to use for particle tracing is ``desc.particles.trace_particles``.
This function takes a field/equilibrium object, as well as particle initializer and trajectory model, and returns the particle trajectories.

.. autosummary::
    :toctree: _api/particles/
    :recursive:

    desc.particles.trace_particles
