"""Tests for particle tracing utilities."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.magnetic_fields import (
    MagneticFieldFromUser,
    ToroidalMagneticField,
    VerticalMagneticField,
)
from desc.particles import (
    ManualParticleInitializerLab,
    VacuumGuidingCenterTrajectory,
    trace_particles,
)
from desc.utils import rpz2xyz, xyz2rpz_vec


@pytest.mark.unit
def test_constant_field_cases():
    """Test particle tracing with constant magnetic field."""
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


@pytest.mark.unit
def test_mirror_force_drift():
    """Test particle tracing with a magnetic field that has a mirror force."""

    def custom_B(coords, params):
        """Custom magnetic field function.

        B_Z = B0 + dB * Z
        B_R = B_phi = 0
        """
        xyz = rpz2xyz(coords)
        _, _, Z = xyz.T
        B0, dB = params
        B = jnp.zeros_like(coords)
        B = B.at[:, 2].set(B0 + dB * Z)
        B = xyz2rpz_vec(B, phi=coords[:, 1])
        return B

    B0 = 1.0  # Constant magnetic field strength
    dB = 0.1  # Gradient of the magnetic field
    E = 1e8  # Particle energy
    xi0 = 0.9  # Initial pitch angle
    R0 = np.array([1.0])

    field = MagneticFieldFromUser(fun=custom_B, params=(B0, dB))
    particles = ManualParticleInitializerLab(R0=R0, phi0=0, Z0=0, xi0=xi0, E=E)
    model = VacuumGuidingCenterTrajectory(frame="lab")
    ts = np.linspace(0, 1e-6, 10)
    x0, args = particles.init_particles(model=model, field=field)
    ms, qs, mus = args[:3]
    rpz, vpar = trace_particles(
        field=field,
        y0=x0,
        ms=ms,
        qs=qs,
        mus=mus,
        model=model,
        ts=ts,
    )
    # The time derivative of drift velocity due to the mirror force is:
    # vpardot = - (mu / m) (𝐛 ⋅ ∇B)   # noqa : E800
    # For the given field, this is -dB * mu / m. So, the exact Z position
    # should be: z(t) = z0 + v0 * t - (dB * mu / m) * t^2 / 2
    # where v0 is the initial parallel velocity and z0 is the initial Z position.
    z = x0[0, 2] + x0[0, 3] * ts - dB * mus[0] / ms[0] * ts**2 / 2
    # The parallel velocity should be vpar(t) = v0 - (dB * mu / m) * t
    vpar_exact = x0[0, 3] - dB * mus[0] / ms[0] * ts

    assert np.allclose(rpz[0, :, 2], z, atol=1e-5)
    assert np.allclose(vpar[0, :, 0], vpar_exact, atol=1e-5)


@pytest.mark.unit
def test_traceing_pure_toroidal_magnetic_field():
    """Test particle tracing within a purely toroidal magnetic field."""
    B0 = 1.0  # Constant magnetic field strength
    R0t = 3.0  # Major radius of the toroidal field
    R0 = np.array([4.0])  # Initial radial position of the particle
    ts = np.linspace(0, 1e-6, 100)
    # B_phi = B0 * R0t / r  # noqa : E800
    field = ToroidalMagneticField(B0=B0, R0=R0t)
    particles = ManualParticleInitializerLab(R0=R0, phi0=0, Z0=0, xi0=0.9, E=1e8)
    model = VacuumGuidingCenterTrajectory(frame="lab")
    x0, args = particles.init_particles(model=model, field=field)
    ms, qs, mus = args[:3]
    rpz, vpar = trace_particles(
        field=field,
        y0=x0,
        ms=ms,
        qs=qs,
        mus=mus,
        model=model,
        ts=ts,
    )
    # Total drif velocity is given by
    # vd = m / (qB^2) (vpar^2 + vperp^2/2) (𝐛 x ∇B)
    # For purely toroidal field,
    # 𝐛 = [0, 1, 0].T   # noqa : E800
    # ∇B = [-B0*R0t/r^2, 0, 0].T   # noqa : E800
    # B = B0*R0t/r   # noqa : E800
    # So, the drift velocity in Z direction is:
    # vd = m / (qB0*R0t) * (vpar^2 + vperp^2/2)   # noqa : E800
    vd = (
        ms[0]
        / (qs[0] * B0 * R0t)
        * (particles.vpar0[0] ** 2 + particles.v0[0] ** 2)
        / 2
    )
    z_exact = vd * ts

    assert np.allclose(rpz[0, :, 2], z_exact, atol=1e-12)
    assert np.allclose(rpz[0, :, 0], R0[0], atol=1e-12)
    assert np.allclose(vpar[0, :, 0], particles.vpar0, atol=1e-12)
