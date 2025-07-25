"""Tests for particle tracing utilities."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.magnetic_fields import VerticalMagneticField
from desc.particles import (
    ManualParticleInitializerLab,
    VacuumGuidingCenterTrajectory,
    trace_particles,
)


@pytest.mark.unit
def test_vertical_field_cases():
    """Test particle tracing with constant vertical magnetic field."""
    R0 = jnp.array([1.2, 1.25])
    ts = jnp.linspace(0, 1e-4, 10)

    # Particles with no perpendicular velocity
    field = VerticalMagneticField(B0=1.0)
    model = VacuumGuidingCenterTrajectory(frame="lab")
    particles = ManualParticleInitializerLab(
        R0=R0,
        phi0=jnp.zeros_like(R0),
        Z0=jnp.zeros_like(R0),
        xi0=jnp.ones_like(R0),
        E=1e10,
        m=4.0,
        q=1.0,
    )

    x0, args = particles.init_particles(model=model, field=field)
    ms, qs, mus = args[:3]
    rpz, _ = trace_particles(field, x0, ms, qs, mus, model=model, ts=ts)
    p1r = rpz[0, :, 0]
    p1z = rpz[0, :, 2]
    p2r = rpz[1, :, 0]
    p2z = rpz[1, :, 2]
    exact_z = particles.v0[0] * ts

    assert np.allclose(p1r, R0[0], atol=1e-12)
    assert np.allclose(p2r, R0[1], atol=1e-12)
    assert np.allclose(p1z, exact_z, atol=1e-8)
    assert np.allclose(p2z, exact_z, atol=1e-8)

    # Particles with no parallel velocity
    particles = ManualParticleInitializerLab(
        R0=R0,
        phi0=jnp.zeros_like(R0),
        Z0=jnp.zeros_like(R0),
        xi0=jnp.zeros_like(R0),
        E=1e10,
        m=4.0,
        q=1.0,
    )

    x0, args = particles.init_particles(model=model, field=field)
    ms, qs, mus = args[:3]
    rpz, _ = trace_particles(field, x0, ms, qs, mus, model=model, ts=ts)
    p1r = rpz[0, :, 0]
    p1z = rpz[0, :, 2]
    p2r = rpz[1, :, 0]
    p2z = rpz[1, :, 2]

    assert np.allclose(p1r, R0[0], atol=1e-12)
    assert np.allclose(p2r, R0[1], atol=1e-12)
    assert np.allclose(p1z, 0.0, atol=1e-12)
    assert np.allclose(p2z, 0.0, atol=1e-12)
