"""Tests for particle tracing utilities."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
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
        B0 = params[0]
        dB = params[1]
        B = jnp.zeros_like(coords)
        B = B.at[:, 2].set(B0 + dB * Z)
        B = xyz2rpz_vec(B, phi=coords[:, 1])
        return B.squeeze()

    B0 = 1.0  # Constant magnetic field strength
    dB = 0.1  # Gradient of the magnetic field
    E = 1e8  # Particle energy
    xi0 = 0.9  # Initial pitch angle
    # TODO: test when R0 is not 1
    # (to make sure the vphi -> phidot conversions are right)
    R0 = np.array([1.0])

    field = MagneticFieldFromUser(fun=custom_B, params=(B0, dB))
    print(field.params_dict)
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
    z_exact = x0[0, 2] + x0[0, 3] * ts - dB * mus[0] / ms[0] * ts**2 / 2
    # The parallel velocity should be vpar(t) = v0 - (dB * mu / m) * t
    vpar_exact = x0[0, 3] - dB * mus[0] / ms[0] * ts

    assert np.allclose(rpz[0, :, 2], z_exact, atol=1e-10)
    assert np.allclose(vpar[0, :, 0], vpar_exact, atol=1e-10)


@pytest.mark.unit
def test_tracing_purely_toroidal_magnetic_field():
    """Test particle tracing within a purely toroidal magnetic field."""
    B0 = 1.0  # Constant magnetic field strength
    Rmajor = 3.0  # Major radius of the toroidal field
    R0 = 4.0  # Initial radial position of the particle
    ts = np.linspace(0, 1e-6, 100)
    # B_phi = B0 * Rmajor / r  # noqa : E800
    field = ToroidalMagneticField(B0=B0, R0=Rmajor)
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
    # ∇B = [-B0*Rmajor/r^2, 0, 0].T   # noqa : E800
    # B = B0*Rmajor/r   # noqa : E800
    # So, the drift velocity in Z direction is:
    # vd = m / (qB0*Rmajor) * (vpar^2 + vperp^2/2)   # noqa : E800
    vd = (
        ms[0]
        / (qs[0] * B0 * Rmajor)
        * (particles.vpar0[0] ** 2 + particles.v0[0] ** 2)
        / 2
    )
    z_exact = vd * ts
    # Angular velocity is constant and given by vpar0 / R0 in radians per second
    # So, the exact phi position is given by phi(t) = vpar0 / R0 * t
    # where vpar0 is the initial parallel velocity and R0 is the major radius.
    phi_exact = particles.vpar0[0] / R0 * ts

    np.testing.assert_allclose(rpz[0, :, 2], z_exact, atol=1e-12)
    # There is no radial drift, R should remain constant
    np.testing.assert_allclose(rpz[0, :, 0], R0, atol=1e-12)
    # There is no mirror force, parallel velocity should be the same
    np.testing.assert_allclose(
        vpar[0, :, 0], particles.vpar0 * np.ones_like(vpar[0, :, 0]), atol=1e-12
    )
    # The phi position should be given by the angular velocity
    np.testing.assert_allclose(rpz[0, :, 1], phi_exact, atol=1e-12)


@pytest.mark.unit
def test_tracing_vacuum_tokamak():
    """Test particle tracing in a vacuum tokamak."""
    rmajor = 4.0
    rminor = 1.0
    ts = np.linspace(0, 1e-7, 100)
    R0 = rmajor + rminor / 2

    surf = FourierRZToroidalSurface(
        R_lmn=np.array([rmajor, rminor]),
        modes_R=np.array([[0, 0], [1, 0]]),
        Z_lmn=np.array([0, -1]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
    )
    eq = Equilibrium(surface=surf, L=8, M=8, N=0, Psi=3)
    eq.solve(verbose=1)

    particles = ManualParticleInitializerLab(
        R0=R0, phi0=0, Z0=0.0, xi0=0.9, E=1e7, eq=eq
    )
    model = VacuumGuidingCenterTrajectory(frame="flux")

    # Particle tracing compute the field on individual points as grid which
    # is not enough to compute iota profile. Instead find the iota profile before
    # and assign it to the equilibrium as a hack. For this test, not very
    # necessary since iota is 0.
    with pytest.warns(UserWarning, match="Setting rotational transform profile"):
        eq.iota = eq.get_profile("iota")

    # Initialize particles
    with pytest.warns(UserWarning, match="The input coordinates are in lab"):
        x0, args = particles.init_particles(model=model, field=eq)
    ms, qs, mus = args[:3]

    # Ensure particles stay within the surface by bounds_R (not actually
    # needed here since the tracing time is chosen accordingly, but this
    # is the intended use case).
    rtz, vpar = trace_particles(
        y0=x0,
        field=eq,
        model=model,
        ms=ms,
        qs=qs,
        mus=mus,
        ts=ts,
        bounds_R=(0, 1),
    )
    rpz = eq.map_coordinates(
        coords=np.array([rtz[0, :, 0], rtz[0, :, 1], rtz[0, :, 2]]).T,
        inbasis=("rho", "theta", "zeta"),
        outbasis=("R", "phi", "Z"),
    )
    # We will find the B0*r00/R field representation of the vacuum tokamak
    # First, find the magnetic field at a random R position (equation doesn't
    # depend on R as llong as B0 and r00 are consistent)
    # Then, the exact solution is the same as given in the
    # test_tracing_purely_toroidal_magnetic_field above
    grid = LinearGrid(rho=0.5, M=eq.M_grid, N=eq.N_grid)
    data = eq.compute(["|B|", "x"], grid=grid)
    B0 = grid.compress(data["|B|"])[0]
    r00 = grid.compress(data["x"])[0, 0]
    vd = (
        ms[0]
        / (qs[0] * B0 * r00)
        * (particles.vpar0[0] ** 2 + particles.v0[0] ** 2)
        / 2
    )
    z_exact = vd * ts
    # Angular velocity is constant and given by vpar0 / R0 in radians per second
    # So, the exact phi position is given by phi(t) = vpar0 / R0 * t
    # where vpar0 is the initial parallel velocity and R0 is the major radius.
    phi_exact = particles.vpar0[0] / R0 * ts

    assert np.allclose(rpz[:, 2], z_exact, atol=1e-12)
    # There is no radial drift, R should remain constant
    assert np.allclose(rpz[:, 0], R0, atol=1e-12)
    # There is no mirror force, parallel velocity should be the same
    assert np.allclose(vpar[0, :, 0], particles.vpar0, atol=1e-12)
    # The phi position should be given by the angular velocity
    assert np.allclose(rpz[:, 1], phi_exact, atol=1e-12)
