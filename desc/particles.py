"""Functions for tracing particles in magnetic fields."""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

import equinox as eqx
import numpy as np
from diffrax import (
    AbstractTerm,
    ConstantStepSize,
    DiscreteTerminatingEvent,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from numpy.typing import ArrayLike
from scipy.constants import (
    Boltzmann,
    electron_mass,
    elementary_charge,
    epsilon_0,
    hbar,
    proton_mass,
)

from desc.backend import jit, jnp, tree_map, vmap
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.derivatives import Derivative
from desc.equilibrium import Equilibrium
from desc.geometry import Curve, Surface
from desc.grid import Grid
from desc.io import IOAble
from desc.magnetic_fields import _MagneticField
from desc.utils import cross, dot, safediv

JOULE_PER_EV = 11606 * Boltzmann
EV_PER_JOULE = 1 / JOULE_PER_EV


class AbstractTrajectoryModel(AbstractTerm, ABC):
    """Abstract base class for particle trajectory models.

    Subclasses should implement the ``vf`` method to compute the RHS of the ODE,
    as well as the properties `frame`, `vcoords`, `args`. ``vf`` method corresponds to
    the ``vf`` method in diffrax.AbstractTerm class and must have the same name and
    signature.
    """

    # as opposed to other classes in DESC which inherit from IOAble, this class
    # is a subclass of diffrax.AbstractTerm which is an Equinox.Module. The following
    # attributes need to be defined as static fields for JAX transformation.
    _frame: str = eqx.field(static=True)
    vcoords: list[str] = eqx.field(static=True)
    args: list[str] = eqx.field(static=True)

    @property
    @abstractmethod
    def frame(self):
        """One of "flux" or "lab", indicating which frame the model is defined in.

        "flux" traces particles in (rho, theta, zeta) magnetic coordinates
        "lab" traces particles in (R, phi, Z) lab frame

        """
        pass

    @property
    @abstractmethod
    def vcoords(self):  # noqa : F811
        """Velocity coordinates used by the model, in order.

        Options are:
        "v" : modulus of velocity
        "vpar" : velocity in direction of local magnetic field.
        "vperp" : modulus of velocity perpendicular to local magnetic field.
        "vR" : velocity in lab frame R direction
        "vP" : velocity in lab frame phi direction
        "vZ" : velocity in lab frame Z direction
        """
        pass

    @property
    @abstractmethod
    def args(self):  # noqa : F811
        """Additional arguments needed by the model.

        Eg, "m", "q", "mu", for mass, charge, magnetic moment (mv⊥²/2|B|).
        """
        pass

    @abstractmethod
    def vf(self, t, x, args):
        """RHS of the particle trajectory ODE."""
        pass

    def contr(self, t0, t1, **kwargs):
        """Needed by diffrax."""
        return t1 - t0

    def prod(self, vf, control):
        """Needed by diffrax."""

        def _mul(v):
            return control * v

        return tree_map(_mul, vf)


class VacuumGuidingCenterTrajectory(AbstractTrajectoryModel):
    """Guiding center trajectories in vacuum, conserving energy and mu.

    Solves the following ODEs,

    # TODO: Is this correct? Check (m v∥² / q B²) term

    d𝐑/dt = v∥ 𝐛 + (m / q B²) ⋅ (v∥² + 1/2 v⊥²) ( 𝐛 × ∇B )

    dv∥/dt = − (v⊥² / 2B) ( 𝐛 ⋅ ∇B )

    where 𝐁 is the magnetic field vector at position 𝐑, B is the magnitude of
    the magnetic field and 𝐛 is the unit magnetic field 𝐁/B.

    Parameters
    ----------
    frame : {"lab", "flux"}
        Which coordinate frame is used for tracing particles. 'lab' corresponds to
        {R, phi, Z} coordinates, 'flux' corresponds to {rho, theta, zeta} coordinates.
        Frame must be compatible with the source of the field, i.e. if tracing in an
        Equilibrium, set frame="flux" or if tracing in a MagneticField, choose "lab".

        Although particles can be traced in "lab" frame using an Equilibrium, it is
        not recommended, since it requires coodinate mapping at each step of the
        integration. Thus, it is not implemented. For that case, we recommend converting
        the final output to "lab" frame after the integration is done using
        Equilibrium.map_coordinates method.
    """

    def __init__(self, frame):
        assert frame in ["lab", "flux"]
        self._frame = frame

    @property
    def frame(self):
        """Which frame the model is defined in."""
        return self._frame

    @property
    def vcoords(self):
        """Which velocity coordinates the model uses."""
        return ["vpar"]

    @property
    def args(self):
        """Which additional args are needed by the model."""
        return ["m", "q", "mu"]

    @jit
    def vf(self, t, x, args):
        """RHS of guiding center trajectories without collisions or slowing down.

        ``vf`` method corresponds to the ``vf`` method in diffrax.AbstractTerm class
        and must have the same name and signature.

        Parameters
        ----------
        t : float
            Time to evaluate RHS at.
        x : jax.Array, shape(4,)
            Position of particle in phase space [rho, theta, zeta, vpar] or
            [R, phi, Z, vpar].
        args : tuple
            Additional arguments needed by model, (m, q, mu, eq_or_field, kwargs).
            kwargs will be passed to the field.compute_magnetic_field method.
            mu is the v⊥²/|B|.

        Returns
        -------
        dx : jax.Array, shape(N,4)
            Velocity of particles in phase space.
        """
        x = x.squeeze()
        m, q, mu, eq_or_field, kwargs = args
        if self.frame == "flux":
            eq_or_field = eqx.error_if(
                eq_or_field,
                not isinstance(eq_or_field, Equilibrium),
                "Integration in flux coordinates requires a MagneticField.",
            )
            return self._compute_flux_coordinates(x, eq_or_field, m, q, mu, **kwargs)
        elif self.frame == "lab":
            eq_or_field = eqx.error_if(
                eq_or_field,
                not isinstance(eq_or_field, _MagneticField),
                "Integration in lab coordinates requires a MagneticField. If using an "
                "Equilibrium, we recommend setting frame='flux' and converting the "
                "output to lab coordinates only at the end by the helper function "
                "Equilibrium.map_coordinates.",
            )
            return self._compute_lab_coordinates(x, eq_or_field, m, q, mu, **kwargs)

    def _compute_flux_coordinates(self, x, eq, m, q, mu, **kwargs):
        """ODE equation for vacuum guiding center in flux coordinates.

        This function is written for vmap, so it expects x to be a coordinate of a
        single particle, and args to be a tuple of (m, q, mu, eq, kwargs) with
        m, q and mu (mv⊥²/2|B|) being scalars.
        """
        rho, theta, zeta, vpar = x
        grid = Grid(
            jnp.array([rho, theta, zeta]).T,
            spacing=jnp.zeros((3,)).T,
            jitable=True,
            sort=False,
        )
        data_keys = [
            "B",
            "|B|",
            "grad(|B|)",
            "e^rho",
            "e^theta",
            "e^zeta",
            "b",
        ]

        transforms = get_transforms(data_keys, eq, grid, jitable=True)
        profiles = get_profiles(data_keys, eq, grid)
        data = compute_fun(eq, data_keys, eq.params_dict, transforms, profiles)

        # derivative of the guiding center position in R, phi, Z coordinates
        # TODO: Is this correct? Check
        Rdot = vpar * data["b"] + (
            (m / q / data["|B|"] ** 2)
            * ((mu * data["|B|"] / m) + vpar**2)
            * cross(data["b"], data["grad(|B|)"])
        )
        rhodot = dot(Rdot, data["e^rho"])
        thetadot = dot(Rdot, data["e^theta"])
        zetadot = dot(Rdot, data["e^zeta"])
        vpardot = -mu / m * dot(data["b"], data["grad(|B|)"])
        dxdt = jnp.array([rhodot, thetadot, zetadot, vpardot]).reshape(x.shape)
        return dxdt.squeeze()

    def _compute_lab_coordinates(self, x, field, m, q, mu, **kwargs):
        """Compute the RHS of the ODE using MagneticField.

        This function is written for vmap, so it expects x to be a coordinate of a
        single particle, and args to be a tuple of (m, q, mu, field, kwargs) with
        m, q and mu being scalars.
        """
        # this is the one implemented in simsopt for method="gc_vac"
        # should be equivalent to full lagrangian from Cary & Brizard in vacuum
        vpar = x[-1]
        coord = x[:-1]

        field_compute = lambda y: jnp.linalg.norm(
            field.compute_magnetic_field(y, **kwargs), axis=-1
        ).squeeze()

        # magnetic field vector in R, phi, Z coordinates
        B = field.compute_magnetic_field(coord, **kwargs)
        grad_B = Derivative(field_compute, mode="grad")(coord)

        modB = jnp.linalg.norm(B, axis=-1)
        b = B / modB
        # factor of R from grad in cylindrical coordinates
        grad_B = grad_B.at[1].set(safediv(grad_B[1], coord[0]))
        # TODO: Is this correct? Check
        Rdot = vpar * b + (m / q / modB**2 * (mu * modB / m + vpar**2)) * cross(
            b, grad_B
        )

        vpardot = jnp.atleast_2d(-mu / m * dot(b, grad_B))
        dxdt = jnp.hstack([Rdot, vpardot.T]).reshape(x.shape)
        return dxdt.squeeze()


class SlowingDownGuidingCenterTrajectory(AbstractTrajectoryModel):
    """Guiding center trajectories with slowing down on electrons and main ions.

    Solves the following ODEs,

    # TODO: Is this correct? Check (m v∥² / q B²) term

    d𝐑/dt = v∥ 𝐛 + (m / q B²) ⋅ (v∥² + 1/2 v⊥²) ( 𝐛 × ∇B )

    dv∥/dt = − ((v² - v∥²) / 2B) ( 𝐛 ⋅ ∇B )

    dv/dt = - v / τₛ (1 + v_c³ / v³)

    where 𝐁 is the magnetic field vector at position 𝐑, B is the magnitude of
    the magnetic field, 𝐛 is the unit magnetic field 𝐁/B, τₛ is the Spitzer
    ion–electron momentum exchange time and v_c is the critical velocity associated
    with the critical energy at which the velocity reduction transitions from
    nearly exponential (drag on electrons) to significantly steeper (drag on ions).
    τₛ and v_c are defined as follows:

          mᵢ (4πϵ₀)² 3mₑ¹ᐟ² Tₑ³ᐟ²
    τₛ = ───────────────────────────
          mₑ 4√(2π) nₑ Zᵢ² e⁴ lnΛ

    v_c = [ (3√π / 4) (mₑ / mᵢ) ]¹ᐟ³ v_{Tₑ}

    v_{Tₑ} = √(2 Tₑ / mₑ)

    See ref [1] Eq. (1-4) for definitions, and other references for the details.

    References
    ----------
    [1] McMillan M, Lazerson S A. "BEAMS3D neutral beam injection model"
    Plasma Physics and Controlled Fusion (2014).
    [2] Callen J D, "Fundamentals of Plasma Physics (Lecture Notes)" (Madison, WI:
    University of Wisconsin Press) (2003)
    [3] Fowler R H, Morris R N, Rome J A and Hanatani K, "Neutral beam injection
    benchmark studies for stellarators/heliotrons", Nucl. Fusion 30 997–1010 (1990)
    [4] Rosenbluth M N, MacDonald W M and Judd D L, "Fokker–Planck equation for an
    inverse-square force", Phys.Rev. 107 1–6 (1957)

    Works only in flux coordinates corresponding to {rho, theta, zeta}. Particle
    tracing can be performed with an Equilibrium object, which must have electron
    temperature Te and electron density ne defined.

    Parameters
    ----------
    m_eff : float
        Effective mass of the plasma main ions, in units of proton mass. Default is 2.5,
        for a 50/50 DT plasma
    Z_eff : float
        Effective charge of the plasma main ions, in units of elementary charge.
        Default is 1, for H/D/T plasmas.
    """

    def __init__(self, frame="flux", m_eff=2.5, Z_eff=1):
        assert frame == "flux"
        self.Z_eff = Z_eff
        self.m_eff = m_eff

    @property
    def frame(self):
        """Which frame the model is defined in."""
        return "flux"

    @property
    def vcoords(self):
        """Which velocity coordinates the model uses."""
        return ["vpar", "v"]

    @property
    def args(self):
        """Which additional args are needed by the model."""
        return ["m", "q"]

    @jit
    def vf(self, t, x, args):
        """RHS of guiding center trajectories without collisions or slowing down.

        ``vf`` method corresponds to the ``vf`` method in diffrax.AbstractTerm class
        and must have the same name and signature.

        Parameters
        ----------
        t : float
            Time to evaluate RHS at.
        x : jax.Array, shape(N,5)
            Position of particle in phase space (rho, theta, zeta, vpar, v).
        args : tuple
            Additional arguments needed by model, ( m, q, mu, eq, kwargs).
            mu and kwargs are not used for this model.

        Returns
        -------
        dx : jax.Array, shape(N,5)
            Velocity of particles in phase space.
        """
        rho, theta, zeta, vpar, v = x
        m, q, _, eq, _ = args

        assert (
            eq.Te_l.size > 0
        ), "Equilibrium must have electron temperature Te defined."
        assert eq.ne_l.size > 0, "Equilibrium must have electron density ne defined."

        grid = Grid(
            jnp.array([rho, theta, zeta]).T,
            spacing=jnp.zeros((3,)).T,
            jitable=True,
            sort=False,
        )
        data_keys = [
            "B",
            "|B|",
            "grad(|B|)",
            "e^rho",
            "e^theta",
            "e^zeta",
            "b",
            "Te",
            "ne",
        ]

        transforms = get_transforms(data_keys, eq, grid, jitable=True)
        profiles = get_profiles(data_keys, eq, grid)
        data = compute_fun(
            eq,
            data_keys,
            eq.params_dict,
            transforms,
            profiles,
        )

        # slowing eqns from McMillan, Matthew, and Samuel A. Lazerson. "BEAMS3D
        # neutral beam injection model." Plasma Physics and Controlled Fusion (2014)
        tau_s = slowing_down_time(data["Te"], data["ne"], self.m_eff, self.Z_eff)
        vc = slowing_down_critical_velocity(data["Te"], self.m_eff)

        # derivative of the guiding center position in R, phi, Z coordinates
        # TODO: Is this correct? Check
        Rdot = vpar * data["b"] + (
            (m / q / data["|B|"] ** 2)
            * (vpar**2 + 0.5 * (v**2 - vpar**2))
            * cross(data["b"], data["grad(|B|)"])
        )
        rhodot = dot(Rdot, data["e^rho"])
        thetadot = dot(Rdot, data["e^theta"])
        zetadot = dot(Rdot, data["e^zeta"])
        vpardot = (
            -(v**2 - vpar**2) / (2 * data["|B|"]) * dot(data["b"], data["grad(|B|)"])
        )
        vdot = -v / tau_s * (1 + vc**3 / v**3)
        dxdt = jnp.array([rhodot, thetadot, zetadot, vpardot, vdot]).reshape(x.shape)
        return dxdt.squeeze()


class AbstractParticleInitializer(IOAble, ABC):
    """Abstract base class for initial distribution of particles for tracing.

    Subclasses should implement the `init_particles` method.
    """

    @abstractmethod
    def init_particles(
        self, model: AbstractTrajectoryModel, field: Union[Equilibrium, _MagneticField]
    ) -> tuple[jnp.ndarray, tuple]:
        """Initialize a distribution of particles.

        Should return two things:
        - an NxD array of initial particle positions and velocities,
        where N is the number of particles and D is the dimensionality of the
        trajectory model (4, 5, or 6).
        - a tuple of additional arguments requested by the model, eg mass, charge,
        magnetic moment of each particle.
        """
        pass

    def _return_particles(self, x, v, vpar, model, field):
        """Return the particles in a common format.

        Parameters
        ----------
        x : jax.Array, shape(N,3)
            Initial particle positions in either flux (rho, theta, zeta) coordinates or
            cylindirical (lab) coordinates, shape (N, 3), where N is the number of
            particles.
        v : ArrayLike, shape(N,)
            Initial particle speeds
        vpar : ArrayLike, shape(N,)
            Initial particle parallel velocities, in the direction of local magnetic
            field.
        model : AbstractTrajectoryModel
            Model to use for tracing particles, which defines the frame and
            velocity coordinates.
        field : Equilibrium or _MagneticField
            Source of magnetic field to use for tracing particles.

        Returns
        -------
        x0 : jax.Array, shape(N,D)
            Initial particle positions and velocities, where D is the dimensionality of
            the trajectory model, which includes 3D spatial dimensions and depending on
            the model parallel velocity and total velocity.
        args : tuple
            Additional arguments needed by the model, such as mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle.
        """
        vs = []
        for vcoord in model.vcoords:
            if vcoord == "vpar":
                vs.append(vpar)
            elif vcoord == "v":
                vs.append(v)
            else:
                raise NotImplementedError
        v0 = jnp.array(vs).T

        args = []
        for arg in model.args:
            if arg == "m":
                args += [self.m]
            elif arg == "q":
                args += [self.q]
            elif arg == "mu":
                vperp2 = v**2 - vpar**2
                modB = _compute_modB(x, field)
                args += [self.m * vperp2 / (2 * modB)]

        return jnp.hstack([x, v0]), tuple(args)


def _compute_modB(x, field, **kwargs):
    if isinstance(field, Equilibrium):
        # if Equilibrium doesn't have an iota profile, this will give bad results
        grid = Grid(
            x.T,
            spacing=jnp.zeros_like(x),
            sort=False,
        )
        return field.compute("|B|", grid=grid)["|B|"]
    return jnp.linalg.norm(field.compute_magnetic_field(x, **kwargs), axis=-1)


class ManualParticleInitializerFlux(AbstractParticleInitializer):
    """Manually specify particle starting positions and energy in flux coordinates.

    Parameters
    ----------
    rho0 : array-like
        Initial radial coordinates
    theta0 : array-like
        Initial poloidal coordinates
    zeta0 : array-like
        Initial toroidal coordinates
    xi0 : array-like
        Initial normalized parallel velocity, xi=vpar/v
    E : array-like
        Initial particle kinetic energy, in eV
    m : float
        Particle mass, in proton masses
    q : float
        Particle charge, in units of elementary charge.
    eq : Equilibrium, optional
        Used to map initial flux coordinates to lab frame, if tracing particles in
        lab frame.
    """

    def __init__(
        self,
        rho0: ArrayLike,
        theta0: ArrayLike,
        zeta0: ArrayLike,
        xi0: ArrayLike,
        E: ArrayLike = 3.5e6,
        m: float = 4,
        q: float = 2,
        eq: Optional[Equilibrium] = None,
    ):
        m = m * proton_mass
        q = q * elementary_charge
        E = E * JOULE_PER_EV
        rho0, theta0, zeta0, xi0, E, m, q = map(
            jnp.atleast_1d, (rho0, theta0, zeta0, xi0, E, m, q)
        )
        rho0, theta0, zeta0, xi0, E, m, q = jnp.broadcast_arrays(
            rho0, theta0, zeta0, xi0, E, m, q
        )
        self.m = m
        self.q = q
        self.rho0 = rho0
        self.theta0 = theta0
        self.zeta0 = zeta0
        self.v0 = jnp.sqrt(2 * E / self.m)
        self.vpar0 = xi0 * self.v0
        self.eq = eq

    def init_particles(
        self, model: AbstractTrajectoryModel, field: Union[Equilibrium, _MagneticField]
    ) -> tuple[jnp.ndarray, tuple]:
        """Initialize particles for a given trajectory model."""
        x = jnp.array([self.rho0, self.theta0, self.zeta0]).T
        if model.frame == "flux":
            x = x
        elif model.frame == "lab":
            if self.eq is None:
                raise ValueError(
                    "Mapping flux coordinates to real space requires an Equilibrium. "
                    "Please provide an Equilibrium object when constructing this class!"
                )
            warnings.warn(
                "The input coordinates are in flux coordinates, but the model operates "
                "in lab coordinates. Converting the given coordinates to lab frame."
            )
            grid = Grid(x)
            x = self.eq.compute("x", grid=grid)["x"]
        else:
            raise NotImplementedError

        return super()._return_particles(
            x=x, v=self.v0, vpar=self.vpar0, model=model, field=field
        )


class ManualParticleInitializerLab(AbstractParticleInitializer):
    """Manually specify particle starting positions and energy in lab coordinates.

    Parameters
    ----------
    R0 : array-like
        Initial radial coordinates
    phi0 : array-like
        Initial toroidal coordinates
    Z0 : array-like
        Initial vertical coordinates
    xi0 : array-like
        Initial normalized parallel velocity, xi=vpar/v
    E : array-like
        Initial particle kinetic energy, in eV
    m : float
        Particle mass, in proton masses
    q : float
        Particle charge, in units of elementary charge.
    eq : Equilibrium, optional
        Used to map initial lab coordinates to flux frame, if tracing particles in
        flux frame.
    """

    def __init__(
        self,
        R0,
        phi0,
        Z0,
        xi0,
        E=3.5e6,
        m=4,
        q=2,
        eq=None,
    ):
        m = m * proton_mass
        q = q * elementary_charge
        E = E * JOULE_PER_EV
        R0, phi0, Z0, xi0, E, m, q = map(jnp.atleast_1d, (R0, phi0, Z0, xi0, E, m, q))
        R0, phi0, Z0, xi0, E, m, q = jnp.broadcast_arrays(R0, phi0, Z0, xi0, E, m, q)
        self.m = m
        self.q = q
        self.R0 = R0
        self.phi0 = phi0
        self.Z0 = Z0
        self.v0 = jnp.sqrt(2 * E / self.m)
        self.vpar0 = xi0 * self.v0
        self.eq = eq

    def init_particles(
        self, model: AbstractTrajectoryModel, field: Union[Equilibrium, _MagneticField]
    ) -> tuple[jnp.ndarray, tuple]:
        """Initialize particles for a given trajectory model."""
        x = jnp.array([self.R0, self.phi0, self.Z0]).T
        if model.frame == "flux":
            if self.eq is None:
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please provide an Equilibrium object when constructing this class!"
                )
            warnings.warn(
                "The input coordinates are in lab coordinates, but the model operates "
                "in flux coordinates. Converting the given coordinates to flux frame."
            )
            x = self.eq.map_coordinates(
                coords=x,
                inbasis=("R", "phi", "Z"),
                outbasis=("rho", "theta", "zeta"),
            )
        elif model.frame == "lab":
            x = x
        else:
            raise NotImplementedError

        return super()._return_particles(
            x=x, v=self.v0, vpar=self.vpar0, model=model, field=field
        )


class CurveParticleInitializer(AbstractParticleInitializer):
    """Randomly sample particles starting on a curve.

    Parameters
    ----------
    curve : desc.geometry.Curve
        Curve object to initialize samples on.
    N : int
        Number of particles to generate.
    E : float
        Initial particle kinetic energy, in eV
    m : float
        Mass of particles, in proton masses.
    q : float
        charge of particles, in units of elementary charge.
    xi_min, xi_max : float
        Minimum and maximum values for randomly sampled normalized parallel velocity.
        xi = vpar/v.
    grid : Grid
        Grid used to discretize curve.
    seed : int
        Seed for rng.

    """

    def __init__(
        self,
        curve: Curve,
        N: int,
        E: float = 3.5e6,
        m: float = 4,
        q: float = 2,
        xi_min: float = -1,
        xi_max: float = 1,
        grid: Grid = None,
        seed: int = 0,
    ):
        self.curve = curve
        self.E = jnp.full(N, E * JOULE_PER_EV)
        self.m = jnp.full(N, m * proton_mass)
        self.q = jnp.full(N, q * elementary_charge)
        self.grid = grid
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.N = N
        self.seed = seed

    def init_particles(
        self, model: AbstractTrajectoryModel, field: Union[Equilibrium, _MagneticField]
    ) -> tuple[jnp.ndarray, tuple]:
        """Initialize particles for a given trajectory model."""
        data = self.curve.compute(["x_s", "s", "ds"], grid=self.grid)
        sqrtg = jnp.linalg.norm(data["x_s"], axis=-1) * data["ds"]
        idxs = _find_random_indices(sqrtg, self.N, seed=self.seed)

        zeta = data["s"][idxs]
        theta = jnp.zeros_like(zeta)
        rho = jnp.zeros_like(zeta)

        if model.frame == "flux":
            x = jnp.array([rho, theta, zeta]).T
        elif model.frame == "lab":
            x = jnp.array([rho, theta, zeta]).T
            grid = Grid(x)
            x = self.curve.compute("x", grid=grid)["x"]

        v = jnp.sqrt(2 * self.E / self.m)
        vpar = np.random.uniform(self.xi_min, self.xi_max, v.size) * v

        return super()._return_particles(x=x, v=v, vpar=vpar, model=model, field=field)


class SurfaceParticleInitializer(AbstractParticleInitializer):
    """Randomly sample particles starting on a surface.

    Parameters
    ----------
    surface : desc.geometry.Surface
        Surface object to initialize samples on.
    N : int
        Number of particles to generate.
    E : float
        Initial particle kinetic energy, in eV
    m : float
        Mass of particles, in proton masses.
    q : float
        charge of particles, in units of elementary charge.
    xi_min, xi_max : float
        Minimum and maximum values for randomly sampled normalized parallel velocity.
        xi = vpar/v.
    grid : Grid
        Grid used to discretize curve.
    seed : int
        Seed for rng.

    """

    def __init__(
        self,
        surface: Surface,
        N: int,
        E: float = 3.5e6,
        m: float = 4,
        q: float = 2,
        xi_min: float = -1,
        xi_max: float = 1,
        grid: Grid = None,
        seed: int = 0,
    ):
        self.surface = surface
        self.E = jnp.full(N, E * JOULE_PER_EV)
        self.m = jnp.full(N, m * proton_mass)
        self.q = jnp.full(N, q * elementary_charge)
        self.grid = grid
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.N = N
        self.seed = seed

    def init_particles(
        self, model: AbstractTrajectoryModel, field: Union[Equilibrium, _MagneticField]
    ) -> tuple[jnp.ndarray, tuple]:
        """Initialize particles for a given trajectory model."""
        data = self.surface.compute(
            ["|e_theta x e_zeta|", "theta", "zeta"], grid=self.grid
        )
        sqrtg = data["|e_theta x e_zeta|"]
        idxs = _find_random_indices(sqrtg, self.N, seed=self.seed)

        zeta = data["zeta"][idxs]
        theta = data["theta"][idxs]
        rho = self.surface.rho * jnp.ones_like(zeta)

        if model.frame == "flux":
            x = jnp.array([rho, theta, zeta]).T
        elif model.frame == "lab":
            x = jnp.array([rho, theta, zeta]).T
            grid = Grid(x)
            x = self.surface.compute("x", grid=grid)["x"]

        v = jnp.sqrt(2 * self.E / self.m)
        vpar = np.random.uniform(self.xi_min, self.xi_max, v.size) * v

        return super()._return_particles(x=x, v=v, vpar=vpar, model=model, field=field)


def _find_random_indices(sqrtg, N, seed):
    """Find random indices for sampling particles on a surface or curve."""
    # probability of particle generation in a given grid point is proportional to
    # its volume/area/length, which is sqrtg. Normalize sqrtg for random number
    # generation limit of 1
    sqrtg /= sqrtg.max()
    nattempts = 5 * N
    rng = np.random.default_rng(seed=seed)
    accept = None
    # loop until choosing exactly N distinct indices
    while len(np.unique(accept)) < N or accept is None:
        # generated random integers might repeat if nattempts is large
        idxs = np.unique(rng.integers(0, sqrtg.shape[0], size=(nattempts,)))
        # note: probability of selecting a number <0.3 in range [0, 1] is 30%
        accept = np.where(rng.uniform(0, 1, size=(idxs.shape[0],)) < sqrtg[idxs])[0]
        # choose random N of the accepted indices (might end up choosing less)
        accept = accept[np.unique(rng.integers(0, accept.shape[0], size=(N,)))]
        # increase the attempt count if not enough particles were accepted
        nattempts = int((nattempts / N + 5) * N)
        idxs = idxs[accept]

    return np.sort(idxs)


def trace_particles(
    field,
    y0,
    ms,
    qs,
    mus,
    ts,
    model,
    rtol=1e-8,
    atol=1e-8,
    max_steps=1000,
    min_step_size=1e-8,
    solver=Tsit5(),
    adjoint=RecursiveCheckpointAdjoint(),
    bounds_R=(0, np.inf),
    bounds_Z=(-np.inf, np.inf),
    **kwargs,
):
    """Trace charged particles in an equilibrium or external magnetic field.

    Parameters
    ----------
    field : MagneticField or Equilibrium
        Source of magnetic field to integrate
    y0 : array-like
        Initial particle positions and velocities, stacked in horizontally [x0, v0].
        The first output of `ParticleInitializer.init_particles`.
    ms : array-like
        Particle masses, must be broadcastable to the shape of y0.
    qs : array-like
        Particle charges, must be broadcastable to the shape of y0.
    mus : array-like
        Particle magnetic moments (v⊥²/|B|), must be broadcastable to the shape of
        y0. For the slowndown model, it won't be used, but must be provided for
        consistency.
    ts : array-like
        Strictly increasing array of time values where output is desired.
    model : AbstractTrajectoryModel
        Trajectory model to integrate with.
    rtol, atol : float
        relative and absolute tolerances for ode integration
    max_steps : int
        maximum number of steps between different output times
    min_step_size: float
        minimum step size (in t) that the integration can take. Defaults to 1e-8
    solver: diffrax.AbstractSolver
        diffrax Solver object to use in integration. Defaults to Tsit5(), a RK45
        explicit solver.
    adjoint : diffrax.AbstractAdjoint
        How to take derivatives of the trajectories. ``RecursiveCheckpointAdjoint``
        supports reverse mode AD and tends to be the most efficient. For forward mode AD
        use ``diffrax.ForwardMode()``.
    bounds_R : tuple of (float,float), optional
        R bounds for particle tracing bounding box. Trajectories that leave this
        box will be stopped, and NaN returned for points outside the box. When tracing
        in flux coordinates, this corresponds to rho bounds and the upper value must
        be 1. Defaults to (0,np.inf)
    bounds_Z : tuple of (float,float), optional
        Z bounds for particle tracing bounding box. Trajectories that leave this
        box will be stopped, and NaN returned for points outside the box. When tracing
        in flux coordinates, this corresponds to zeta bounds.
        Defaults to (-np.inf,np.inf).
    kwargs : dict, optional
        Additional keyword arguments to pass to the field computation, such as
            - source_grid: Grid
                Source grid to use for field computation.

    Returns
    -------
    x : ndarray, shape(num_particles, num_timesteps, 3)
        Position of each particle at each requested time, in
        either r,phi,z or rho,theta,zeta depending ``model.frame``.
    v : ndarray
        Velocity of each particle at specified times. The exact number of meaning
        will depend on ``model.vcoords``.

    """
    stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=min_step_size)
    # Euler method does not support adavtive step size controller
    stepsize_controller = (
        ConstantStepSize()
        if solver.__class__.__name__ == "Euler"
        else stepsize_controller
    )

    saveat = SaveAt(ts=ts)

    def default_terminating_event_fxn(state, **kwargs):
        R_out = jnp.logical_or(state.y[0] < bounds_R[0], state.y[0] > bounds_R[1])
        Z_out = jnp.logical_or(state.y[2] < bounds_Z[0], state.y[2] > bounds_Z[1])
        return jnp.logical_or(R_out, Z_out)

    event = DiscreteTerminatingEvent(default_terminating_event_fxn)
    intfun = lambda x, m, q, mu: diffeqsolve(
        model,
        solver,
        y0=x,
        args=[m, q, mu, field, kwargs],
        t0=ts[0],
        t1=ts[-1],
        saveat=saveat,
        max_steps=int(max(max_steps, (ts[1] - ts[0]) / min_step_size) * len(ts)),
        dt0=min_step_size,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        discrete_terminating_event=event,
    ).ys

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unhashable type")
        warnings.filterwarnings("ignore", message="`diffrax.*discrete_terminating")
        yt = vmap(intfun)(y0, ms, qs, mus)

    yt = jnp.where(jnp.isinf(yt), jnp.nan, yt)

    x = yt[:, :, :3]
    v = yt[:, :, 3:]

    return x, v


def gc_radius(vperp, modB, m, q):
    """Radius of guiding center orbit.

    Parameters
    ----------
    vperp : array-like
        Magnitude of perpendicular velocity, in m/s
    modB : array-like
        Magnitude of magnetic field, in T.
    m : array-like
        Mass of particle, in kg.
    q : array-like
        Charge of particle, in C.

    Returns
    -------
    rho : jax.Array
        Gyroradius, in meters.
    """
    return m * vperp / (jnp.abs(q) * modB)


def slowing_down_time(Te, ne, m_eff=2.5, Z_eff=1):
    """Slowing down time for fast ion friction with electrons.

    Parameters
    ----------
    Te : array-like
        Electron temperature, in eV
    ne : array-like
        Electron density, in particles/m^3
    m_eff : array-like
        Effective mass of the plasma main ions, in units of proton mass. Default is 2.5,
        for a 50/50 DT plasma
    Z_eff : array-like
        Effective charge of the plasma main ions, in units of elementary charge.
        Default is 1, for H/D/T plasmas.

    Returns
    -------
    tau_s : jax.Array
        Slowing down time, in seconds.
    """
    me = electron_mass
    mi = m_eff * proton_mass

    lnlambda = coulomb_logarithm(ne, ne / Z_eff, Te, Te, m_eff, Z_eff)

    tau_s = (
        (mi / me)
        * (4 * jnp.pi * epsilon_0) ** 2
        / (4 * jnp.sqrt(2 * jnp.pi))
        * (3 * me ** (1 / 2) * (Te * JOULE_PER_EV) ** (3 / 2))
        / (ne * Z_eff**2 * elementary_charge**4 * lnlambda)
    )
    return tau_s


def slowing_down_critical_velocity(Te, m_eff=2.5):
    """Critical velocity for transition from slowing down on electrons to ions.

    For fast ion speeds above the critical velocity the particles primarily slow down
    due to friction with electrons, which reduces the particle speed exponentially.
    Below the critical velocity friction with the ions dominates and the slowing down
    is super-exponential.

    Parameters
    ----------
    Te : array-like
        Electron temperature, in eV
    m_eff : array-like
        Effective mass of the plasma main ions, in units of proton mass. Default is 2.5,
        for a 50/50 DT plasma

    Returns
    -------
    vc : jax.Array
        Critical velocity.
    """
    me = electron_mass
    mi = m_eff * proton_mass
    vTe = jnp.sqrt(2 * Te * JOULE_PER_EV / me)
    vc = (3 * jnp.sqrt(jnp.pi) / 4 * me / mi) ** (1 / 3) * vTe
    return vc


def coulomb_logarithm(ne, ni, Te, Ti, m_eff, Z_eff) -> float:
    """Coulomb logarithm for collisions between species a and b.

    Parameters
    ----------
    ne, ni : float
        Density of electrons and ions in particles/m^3.
    Te, Ti : float
        Temperature of electrons and ions in eV.
    m_eff : float
        Effective mass of ions, in units of proton mass.
    Z_eff : float
        Effective charge of ions, in units of elementary charge.

    Returns
    -------
    log(lambda) : float

    """
    bmin, bmax = impact_parameter(ne, ni, Te, Ti, m_eff, Z_eff)
    return jnp.log(bmax / bmin)


def impact_parameter(ne, ni, Te, Ti, m_eff, Z_eff) -> float:
    """Impact parameters for classical Coulomb collision.

    Parameters
    ----------
    ne, ni : float
        Density of electrons and ions in particles/m^3.
    Te, Ti : float
        Temperature of electrons and ions in eV.
    m_eff : float
        Effective mass of ions, in units of proton mass.
    Z_eff : float
        Effective charge of ions, in units of elementary charge.
    """
    vte = jnp.sqrt(2 * Te * JOULE_PER_EV / electron_mass)
    vti = jnp.sqrt(2 * Ti * JOULE_PER_EV / (m_eff * proton_mass))
    bmin = jnp.maximum(
        impact_parameter_perp(m_eff, Z_eff, vte, vti),
        debroglie_length(m_eff, vte, vti),
    )
    bmax = debye_length(ne, ni, Te, Ti, Z_eff)
    return bmin, bmax


def impact_parameter_perp(m_eff, Z_eff, vte, vti) -> float:
    """Distance of the closest approach for a 90° Coulomb collision.

    Parameters
    ----------
    m_eff : float
        Effective mass of ions, in units of proton mass.
    Z_eff : float
        Effective charge of ions, in units of elementary charge.
    vte, vti : float
        Thermal speeds of electrons and ions in m/s.
    """
    me = electron_mass
    mi = m_eff * proton_mass
    m_reduced = (mi * me) / (me + mi)
    qe = -elementary_charge
    qi = Z_eff * elementary_charge
    return qe * qi / (4 * jnp.pi * epsilon_0 * m_reduced * (vte**2 + vti**2))


def debroglie_length(m_eff, vte, vti) -> float:
    """Thermal DeBroglie wavelength.

    Parameters
    ----------
    m_eff : float
        Effective mass of ions, in units of proton mass.
    vte, vti : float
        Thermal speeds of electrons and ions in m/s.
    """
    me = electron_mass
    mi = m_eff * proton_mass
    m_reduced = (mi * me) / (me + mi)
    v_th = jnp.sqrt(vte**2 + vti**2)
    return hbar / (2 * m_reduced * v_th)


def debye_length(ne, ni, Te, Ti, Z_eff) -> float:
    """Scale length for charge screening.

    Parameters
    ----------
    ne, ni : float
        Density of electrons and ions in particles/m^3
    Te, Ti : float
        Temperature of electrons and ions in eV
    Z_eff : float
        Effective charge of ions, in units of elementary charge.
    """
    den = ne * elementary_charge**2 / (Te * JOULE_PER_EV)
    den += ni * Z_eff**2 * elementary_charge**2 / (Ti * JOULE_PER_EV)
    return jnp.sqrt(epsilon_0 / den)
