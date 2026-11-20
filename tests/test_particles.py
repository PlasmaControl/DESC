"""Tests for particle tracing utilities."""

import numpy as np
import pytest

from desc.backend import jit, jnp
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZCurve, FourierRZToroidalSurface
from desc.grid import (
    CustomGridFlux,
    LinearGridCurve,
    LinearGridFlux,
    LinearGridToroidalSurface,
)
from desc.magnetic_fields import (
    MagneticFieldFromUser,
    ToroidalMagneticField,
    VerticalMagneticField,
)
from desc.particles import (
    CurveParticleInitializer,
    ManualParticleInitializerFlux,
    ManualParticleInitializerLab,
    SurfaceParticleInitializer,
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

    rpz, _ = trace_particles(field, initializer=particles, model=model, ts=ts)
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

    rpz, _ = trace_particles(field, initializer=particles, model=model, ts=ts)
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
    # choose R0 != 1 to make sure the vphi -> phidot conversions are right
    R0 = np.array([2.0])

    field = MagneticFieldFromUser(fun=custom_B, params=(B0, dB))
    particles = ManualParticleInitializerLab(R0=R0, phi0=0, Z0=0, xi0=xi0, E=E)
    model = VacuumGuidingCenterTrajectory(frame="lab")
    x0, args = particles.init_particles(model=model, field=field)
    ts = np.linspace(0, 1e-6, 10)
    rpz, vpar = trace_particles(
        field=field,
        initializer=particles,
        model=model,
        ts=ts,
    )
    m, _, mu = args[0, :]
    # The time derivative of drift velocity due to the mirror force is:
    # vpardot = - (mu / m) (𝐛 ⋅ ∇B)   # noqa : E800
    # For the given field, this is -dB * mu / m. So, the exact Z position
    # should be: z(t) = z0 + v0 * t - (dB * mu / m) * t^2 / 2
    # where v0 is the initial parallel velocity and z0 is the initial Z position.
    z_exact = x0[0, 2] + x0[0, 3] * ts - dB * mu / m * ts**2 / 2
    # The parallel velocity should be vpar(t) = v0 - (dB * mu / m) * t
    vpar_exact = x0[0, 3] - dB * mu / m * ts

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
    m, q, _ = args[0, :]
    rpz, vpar = trace_particles(
        field=field,
        initializer=particles,
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
    vd = m / (q * B0 * Rmajor) * (particles.vpar0[0] ** 2 + particles.v0[0] ** 2) / 2
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
    ts = np.linspace(0, 1e-6, 100)
    R0 = rmajor + rminor / 2

    # Create a vacuum tokamak equilibrium with a FourierRZToroidalSurface
    surf = FourierRZToroidalSurface(
        R_lmn=np.array([rmajor, rminor]),
        modes_R=np.array([[0, 0], [1, 0]]),
        Z_lmn=np.array([0, -1]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
    )
    eq = Equilibrium(surface=surf, L=8, M=8, N=0, Psi=3)
    eq.solve(verbose=1)

    particles = ManualParticleInitializerLab(R0=R0, phi0=0, Z0=0.0, xi0=0.9, E=1e6)
    model = VacuumGuidingCenterTrajectory(frame="flux")

    # Particle tracing compute the field on individual points as grid which
    # is not enough to compute iota profile. Instead find the iota profile before
    # and assign it to the equilibrium as a hack. For this test, not very
    # necessary since iota is 0.
    with pytest.warns(UserWarning, match="Setting rotational transform profile"):
        eq.iota = eq.get_profile("iota")

    # Initialize particles
    x0, args = particles.init_particles(model=model, field=eq)
    m, q, _ = args[0, :]
    rtz, vpar = trace_particles(
        field=eq,
        initializer=particles,
        model=model,
        ts=ts,
    )
    grid = CustomGridFlux(rtz[0, :, :], jitable=True)
    rpz = eq.compute("x", grid=grid)["x"]
    # We will find the B0*r00/R field representation of the vacuum tokamak
    # First, find the magnetic field at a random R position (equation doesn't
    # depend on R as long as B0 and r00 are consistent)
    # Then, the exact solution is the same as given in the
    # test_tracing_purely_toroidal_magnetic_field above
    grid = LinearGridFlux(rho=0.5, M=eq.M_grid, N=eq.N_grid)
    data = eq.compute(["|B|", "x"], grid=grid)
    B0 = grid.compress(data["|B|"])[0]
    r00 = grid.compress(data["x"])[0, 0]
    vd = m / (q * B0 * r00) * (particles.vpar0[0] ** 2 + particles.v0[0] ** 2) / 2
    z_exact = vd * ts
    # Angular velocity is constant and given by vpar0 / R0 in radians per second
    # So, the exact phi position is given by phi(t) = vpar0 / R0 * t
    # where vpar0 is the initial parallel velocity and R0 is the initial R.
    phi_exact = particles.vpar0[0] / R0 * ts

    assert np.allclose(rpz[:, 2], z_exact, atol=1e-12)
    # There is no radial drift, R should remain constant
    assert np.allclose(rpz[:, 0], R0, atol=1e-12)
    # There is no mirror force, parallel velocity should be the same
    assert np.allclose(vpar[0, :, 0], particles.vpar0, atol=1e-12)
    # The phi position should be given by the angular velocity
    assert np.allclose(rpz[:, 1], phi_exact, atol=1e-12)


@pytest.mark.unit
def test_init_manual_lab():
    """Test ManualParticleInitializerLab class."""
    # some dummy field
    field = ToroidalMagneticField(1.0, 3.0)
    model = VacuumGuidingCenterTrajectory(frame="lab")

    def test(R0):
        particles = ManualParticleInitializerLab(R0=R0, phi0=0, Z0=0, xi0=0.9, E=1e6)
        x0, args = particles.init_particles(model, field)
        assert x0.shape == (1, 4)
        assert args.shape == (1, 3)

    # test different data formats initialize properly
    test(1.0)
    test(np.array([1.0]))
    test([1.0])

    # test multiple particle initialization
    R0 = np.linspace(1, 3, 100)
    particles = ManualParticleInitializerLab(R0=R0, phi0=0, Z0=0, xi0=0.9, E=1e6)
    x0, args = particles.init_particles(model, field)
    assert x0.shape == (100, 4)
    assert args.shape == (100, 3)

    # test that intitializer is jitable
    _, _ = jit(particles.init_particles)(model, field)

    # test using lab class to initialize in flux coordinates
    eq = Equilibrium(M=4, N=0)
    model = VacuumGuidingCenterTrajectory(frame="lab")
    with pytest.raises(NotImplementedError):
        x0, args = particles.init_particles(model, eq)
    model = VacuumGuidingCenterTrajectory(frame="flux")
    R0 = [10.0, 10.1]
    particles = ManualParticleInitializerLab(R0=R0, phi0=0, Z0=0, xi0=0.9, E=1e6)
    x0, args = particles.init_particles(model, eq)
    assert x0.shape == (2, 4)
    assert args.shape == (2, 3)

    # test that intitializer is jitable with coordinate mapping
    _, _ = jit(particles.init_particles)(model, eq)


@pytest.mark.unit
def test_init_manual_flux():
    """Test ManualParticleInitializerFlux class."""
    # some dummy equilibrium
    eq = Equilibrium(M=4, N=0)
    model = VacuumGuidingCenterTrajectory(frame="flux")

    def test(R0):
        particles = ManualParticleInitializerFlux(
            rho0=R0, theta0=0, zeta0=0, xi0=0.9, E=1e6
        )
        x0, args = particles.init_particles(model, eq)
        ms, qs, mus = args.T
        assert x0.shape == (1, 4)
        assert ms.shape == (1,)
        assert qs.shape == (1,)
        assert mus.shape == (1,)

    # test different data formats initialize properly
    test(1.0)
    test(np.array([1.0]))
    test([0.8])
    with pytest.raises(ValueError):
        # rho is defined between 0 and 1
        test([1.5])

    # test multiple particle initialization
    R0 = np.linspace(0.1, 1.0, 100)
    particles = ManualParticleInitializerFlux(
        rho0=R0, theta0=0, zeta0=0, xi0=0.9, E=1e6
    )
    x0, args = particles.init_particles(model, eq)
    ms, qs, mus = args.T
    assert x0.shape == (100, 4)
    assert ms.shape == (100,)
    assert qs.shape == (100,)
    assert mus.shape == (100,)

    # test that intitializer is jitable
    _, _ = jit(particles.init_particles)(model, eq)

    # test using flux class to initialize in lab coordinates
    eq = Equilibrium(M=4, N=0)
    model = VacuumGuidingCenterTrajectory(frame="lab")
    # this is computational expensive, requires 2coordinate mapping
    # user shouldn't use it
    with pytest.raises(NotImplementedError):
        x0, args = particles.init_particles(model, eq)

    # we don't know how to go from flux input to lab frame with MagneticField
    field = ToroidalMagneticField(1.0, 3.0)
    with pytest.raises(NotImplementedError):
        x0, args = particles.init_particles(model, field)

    # we can pass eq to allow mapping from flux to lab
    x0, args = particles.init_particles(model, field, eq=eq)
    # test that intitializer is jitable
    _, _ = jit(particles.init_particles)(model, field, eq=eq)


@pytest.mark.unit
def test_init_surface_particles():
    """Test SurfaceParticleInitializer class."""
    # some dummy equilibrium with R=10 and a=1
    eq = Equilibrium(M=4, N=0)
    # some dummy equilibrium with R=10 and a=2
    surf = FourierRZToroidalSurface(
        R_lmn=np.array([10, 2]),
        modes_R=np.array([[0, 0], [1, 0]]),
        Z_lmn=np.array([0, -2]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
    )
    eq_large = Equilibrium(M=4, N=0, surface=surf)
    model = VacuumGuidingCenterTrajectory(frame="flux")

    rho = 0.8
    surf = eq.get_surface_at(rho=rho)
    # rho=0.8 surface is not inside eq but inside eq_large
    surf_large = eq_large.get_surface_at(rho=rho)
    grid = LinearGridToroidalSurface(M=eq.M_grid, N=eq.N_grid)

    N = 100
    particles = SurfaceParticleInitializer(
        surface=surf, N=N, xi_min=0.1, xi_max=0.9, grid=grid, seed=42
    )
    particles_large = SurfaceParticleInitializer(
        surface=surf_large, N=N, xi_min=0.1, xi_max=0.9, grid=grid, seed=42
    )
    # surface and equilibrium are consistent in rho since we generated the surface
    # from the equilibrium
    x0, args = particles.init_particles(model, eq)

    assert x0.shape == (N, 4)
    assert args.shape == (N, 3)
    np.testing.assert_allclose(x0[:, 0], surf.rho)

    # larger surface is out of small equilibrium, so it should fail
    with pytest.raises(match="Mapping from lab to flux coordinates failed"):
        _, _ = particles_large.init_particles(model, eq)

    # surface and equilibrium are consistent in rho since we generated the surface
    # from the equilibrium
    x0, args = particles_large.init_particles(model, eq_large)

    assert x0.shape == (N, 4)
    assert args.shape == (N, 3)
    np.testing.assert_allclose(x0[:, 0], surf.rho)

    # surface lies inside the equilibrium, but their rho shouldn't match
    x0, args = particles.init_particles(model, eq_large)

    assert x0.shape == (N, 4)
    assert args.shape == (N, 3)
    assert (x0[:, 0] != surf.rho).all()

    # test that intitializer is jitable
    _, _ = jit(particles.init_particles)(model, eq_large)


@pytest.mark.unit
def test_init_curve_particles():
    """Test CurveParticleInitializer class."""
    # some dummy equilibrium with R=10 and a=1
    eq = Equilibrium(M=4, N=0)
    # some dummy equilibrium with R=20 and a=1
    surface = FourierRZToroidalSurface(
        R_lmn=np.array([20, 1]),
        modes_R=np.array([[0, 0], [1, 0]]),
        Z_lmn=np.array([0, -1]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
    )
    eq_large = Equilibrium(M=4, N=0, surface=surface)
    model = VacuumGuidingCenterTrajectory(frame="flux")

    curve = eq.get_axis()
    # curve that passesthrough the LCFS of small equilibrium
    curve_mid = FourierRZCurve(R_n=11.0)
    curve_large = eq_large.get_axis()
    grid = LinearGridCurve(N=eq.N_grid)

    N = 100
    particles = CurveParticleInitializer(
        curve=curve, N=N, xi_min=0.1, xi_max=0.9, grid=grid, seed=42
    )
    particles_mid = CurveParticleInitializer(
        curve=curve_mid, N=N, xi_min=0.1, xi_max=0.9, grid=grid, seed=42
    )
    particles_large = CurveParticleInitializer(
        curve=curve_large, N=N, xi_min=0.1, xi_max=0.9, grid=grid, seed=42
    )
    x0, args = particles.init_particles(model, eq)

    assert x0.shape == (N, 4)
    assert args.shape == (N, 3)
    np.testing.assert_allclose(x0[:, 0], 0.0, atol=1e-8)

    # mid curve should lie on the LCFS of small equilibrium
    x0, args = particles_mid.init_particles(model, eq)

    assert x0.shape == (N, 4)
    assert args.shape == (N, 3)
    np.testing.assert_allclose(x0[:, 0], 1.0, atol=1e-8)

    # test that intitializer is jitable
    _, _ = jit(particles_mid.init_particles)(model, eq)

    # larger curve is out of smaller equilibrium, so it should fail
    with pytest.raises(match="Mapping from lab to flux coordinates failed"):
        _, _ = particles_large.init_particles(model, eq)

    x0, args = particles_large.init_particles(model, eq_large)

    assert x0.shape == (N, 4)
    assert args.shape == (N, 3)
    np.testing.assert_allclose(x0[:, 0], 0.0, atol=1e-8)

    # also check that particle positions are correct in lab frame
    grid_t = CustomGridFlux(x0[:, :3], jitable=True)
    rpz = eq_large.compute("x", grid=grid_t)["x"]

    np.testing.assert_allclose(rpz[:, 0], curve_large.R_n[0], atol=1e-8)
    np.testing.assert_allclose(rpz[:, 2], 0.0, atol=1e-8)

    # smaller curve is out of larger equilibrium, so it should fail
    with pytest.raises(match="Mapping from lab to flux coordinates failed"):
        _, _ = particles.init_particles(model, eq_large)
