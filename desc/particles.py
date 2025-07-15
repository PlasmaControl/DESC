"""Functions for tracing particles in magnetic fields."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from diffrax import (
    AbstractSolver,
    AbstractTerm,
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
from desc.utils import cross, dot

JOULE_PER_EV = 11606 * Boltzmann
EV_PER_JOULE = 1 / JOULE_PER_EV


class AbstractTrajectoryModel(AbstractTerm, ABC):
    """Abstract base class for particle trajectory models.

    Subclasses should implement the ``vf`` method to compute the RHS of the ODE,
    as well as the properties `frame`, `vcoords`, `args`. ``vf`` method corresponds to
    the ``vf`` method in diffrax.AbstractTerm class and must have the same name and
    signature.
    """

    @property
    @abstractmethod
    def frame(self) -> str:
        """One of "flux" or "lab", indicating which frame the model is defined in.

        "flux" traces particles in (rho, theta, zeta) magnetic coordinates
        "lab" traces particles in (R, phi, Z) lab frame

        """
        pass

    @property
    @abstractmethod
    def vcoords(self) -> list[str]:
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
    def args(self) -> list[str]:
        """Additional arguments needed by the model.

        Eg, "m", "q", "mu", for mass, charge, magnetic moment.
        """
        pass

    @abstractmethod
    def vf(self, t: float, x: jnp.ndarray, args: tuple) -> jnp.ndarray:
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

    Parameters
    ----------
    frame : {"lab", "flux"}
        Which coordinate frame is used for tracing particles. Should correspond to the
        source of the field. If tracing in an Equilibrium, set frame="flux". If tracing
        in a CoilSet or MagneticField, choose "lab".
    """

    _frame: str

    def __init__(self, frame: str):
        assert frame in ["lab", "flux"]
        self._frame = frame

    @property
    def frame(self) -> str:
        """Which frame the model is defined in."""
        return self._frame

    @property
    def vcoords(self) -> list[str]:
        """Which velocity coordinates the model uses."""
        return ["vpar"]

    @property
    def args(self) -> list[str]:
        """Which additional args are needed by the model."""
        return ["m", "q", "mu"]

    def vf(self, t: float, x: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """RHS of guiding center trajectories without collisions or slowing down.

        ``vf`` method corresponds to the ``vf`` method in diffrax.AbstractTerm class
        and must have the same name and signature.

        Parameters
        ----------
        t : float
            Time to evaluate RHS at.
        x : jax.Array, shape(4,)
            Position of particle in phase space [psi_n, theta, zeta, vpar] or
            [R, phi, Z, vpar].
        args : tuple
            Additional arguments needed by model, (eq_or_field, m, q, mu, field_kwargs).

        Returns
        -------
        dx : jax.Array, shape(N,4)
            Velocity of particles in phase space.
        """
        eq_or_field, m, q, mu, kwargs = args
        if self.frame == "flux":
            assert isinstance(eq_or_field, Equilibrium)
            return self._compute_flux_coordinates(x, eq_or_field, m, q, mu, **kwargs)
        elif self.frame == "lab":
            assert isinstance(eq_or_field, _MagneticField)
            return self._compute_lab_coordinates(x, eq_or_field, m, q, mu, **kwargs)

    def _compute_flux_coordinates(self, x, eq, m, q, mu, **kwargs):
        """Compute the RHS of the ODE using Equilibrium."""
        assert eq.iota is not None
        # TODO: (yigit) I prefer having rho instead of psi, but this is how it was
        # implemented in the past, so keeping it for now.
        psi, theta, zeta, vpar = x.T
        grid = Grid(
            jnp.array([jnp.sqrt(psi), theta, zeta]).T,
            spacing=jnp.zeros((3,)).T,
            jitable=True,
            sort=False,
        )
        data_keys = [
            "B",
            "|B|",
            "grad(|B|)",
            "grad(psi)",
            "e^theta",
            "e^zeta",
            "b",
        ]

        transforms = get_transforms(data_keys, eq, grid, jitable=True)
        profiles = get_profiles(data_keys, eq, grid)
        data = compute_fun(
            eq, data_keys, eq.params_dict, transforms, profiles, **kwargs
        )

        # derivative of the guiding center position in R, phi, Z coordinates
        Rdot = vpar * data["b"] + (
            (m / q / data["|B|"] ** 2)
            * ((mu * data["|B|"]) + vpar**2)
            * cross(data["b"], data["grad(|B|)"])
        )
        # TODO: not sure why we use psi, and where the factors come from
        psidot = dot(Rdot, data["grad(psi)"]) * (2 * jnp.pi / eq.Psi)
        thetadot = dot(Rdot, data["e^theta"])
        zetadot = dot(Rdot, data["e^zeta"])
        vpardot = -mu * dot(data["b"], data["grad(|B|)"])
        dxdt = jnp.array([psidot, thetadot, zetadot, vpardot]).T.reshape(x.shape)
        return dxdt

    def _compute_lab_coordinates(self, x, field, m, q, mu, **kwargs):
        """Compute the RHS of the ODE using MagneticField."""
        # this is the one implemented in simsopt for method="gc_vac"
        # should be equivalent to full lagrangian from Cary & Brizard in vacuum
        vpar = x[-1]
        coords = x[:-1]
        field_compute = lambda y: jnp.linalg.norm(
            field.compute_magnetic_field(y, **kwargs), axis=-1
        )

        # magnetic field vector in R, phi, Z coordinates
        B = field.compute_magnetic_field(coords, **kwargs)
        grad_B = jnp.vectorize(
            Derivative(
                field_compute,
                mode="fwd",
            ),
            signature="(n)->(n,n)",
        )(coords).squeeze()

        modB = jnp.linalg.norm(B, axis=-1)
        b = B / modB
        # factor of R from grad in cylindrical coordinates
        grad_B = grad_B.at[:, 1].divide(coords[0])

        Rdot = vpar * b + (m / q / modB**2 * (mu * modB + vpar**2)) * cross(b, grad_B)
        # TODO: I don't think this is correct
        # d1, d2, d3 = Rdot               # noqa: E800
        # d2 /= coords[0]                 # noqa: E800
        # Rdot = jnp.array([d1, d2, d3])  # noqa: E800

        vpardot = -mu * dot(b, grad_B)
        dxdt = jnp.concatenate([Rdot, vpardot]).reshape(x.shape)
        return dxdt


class SlowingDownGuidingCenterTrajectory(AbstractTrajectoryModel):
    """Guiding center trajectories with slowing down on electrons and main ions.

    Works only in flux coordinates.
    """

    def __init__(self, frame: str = "flux"):
        assert frame == "flux"

    @property
    def frame(self) -> "str":
        """Which frame the model is defined in."""
        return "flux"

    @property
    def vcoords(self) -> list[str]:
        """Which velocity coordinates the model uses."""
        return ["vpar", "v"]

    @property
    def args(self) -> list[str]:
        """Which additional args are needed by the model."""
        return ["m", "q"]

    def vf(self, t: float, x: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """RHS of guiding center trajectories without collisions or slowing down.

        ``vf`` method corresponds to the ``vf`` method in diffrax.AbstractTerm class
        and must have the same name and signature.

        Parameters
        ----------
        t : float
            Time to evaluate RHS at.
        x : jax.Array, shape(N,5)
            Position of particle in phase space (psi_n, theta, zeta, vpar, v).
        args : tuple
            Additional arguments needed by model, (eq, m, q, kwargs).

        Returns
        -------
        dx : jax.Array, shape(N,5)
            Velocity of particles in phase space.
        """
        eq, m, q, kwargs = args
        assert eq.iota is not None
        assert eq.electron_temperature is not None

        psi, theta, zeta, vpar, v = x.T
        grid = Grid(
            jnp.array([jnp.sqrt(psi), theta, zeta]).T,
            spacing=jnp.zeros((3,)).T,
            jitable=True,
            sort=False,
        )
        data_keys = [
            "B",
            "|B|",
            "grad(|B|)",
            "grad(psi)",
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
            **kwargs,
        )

        # slowing eqns from McMillan, Matthew, and Samuel A. Lazerson. "BEAMS3D
        # neutral beam injection model." Plasma Physics and Controlled Fusion (2014)
        tau_s = slowing_down_time(data["Te"], data["ne"])
        vc = slowing_down_critical_velocity(data["Te"])

        # derivative of the guiding center position in R, phi, Z coordinates
        Rdot = vpar * data["b"] + (
            (m / q / data["|B|"] ** 2) * (v**2) * cross(data["b"], data["grad(|B|)"])
        )
        # TODO: not sure why we use psi, and where the factors come from
        psidot = dot(Rdot, data["grad(psi)"]) * (2 * jnp.pi / eq.Psi)
        thetadot = dot(Rdot, data["e^theta"])
        zetadot = dot(Rdot, data["e^zeta"])
        vpardot = (
            -(v**2 - vpar**2) / (2 * data["|B|"]) * dot(data["b"], data["grad(|B|)"])
        )
        vdot = -v / tau_s * (1 + vc**3 / v**3)
        dxdt = jnp.array([psidot, thetadot, zetadot, vpardot, vdot]).T.reshape(x.shape)
        return dxdt


class AbstractParticleInitializer(IOAble, ABC):
    """ABC for initial distribution of particles for tracing.

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


def _compute_modB(x, field, **kwargs):
    if isinstance(field, Equilibrium):
        psi, theta, zeta = x.T
        grid = Grid(
            jnp.array([jnp.sqrt(psi), theta, zeta]).T,
            spacing=jnp.zeros(
                (
                    x.shape[0],
                    3,
                )
            ).T,
            sort=False,
        )
        return field.compute("|B|", grid=grid, **kwargs)["|B|"]
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
        Initial particle energy, in eV
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
            jnp.asarray, (rho0, theta0, zeta0, xi0, E, m, q)
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
        """Initialize N random particles for a given trajectory model."""
        x = jnp.array([self.rho0, self.theta0, self.zeta0]).T
        if model.frame == "flux":
            x = x
        elif model.frame == "lab":
            if self.eq is None:
                raise ValueError(
                    "Mapping flux coordinates to real space requires an Equilibrium"
                )
            grid = Grid(x)
            x = self.eq.compute("x", grid=grid)["x"]
        else:
            raise NotImplementedError

        vs = []
        for vcoord in model.vcoords:
            if vcoord == "vpar":
                vs.append(self.vpar0)
            elif vcoord == "v":
                vs.append(self.v0)
            else:
                raise NotImplementedError
        v = jnp.array(vs).T

        args = []
        for arg in model.args:
            if arg == "m":
                args += [self.m]
            elif arg == "q":
                args += [self.q]
            elif arg == "mu":
                vperp2 = self.v0**2 - self.vpar0**2
                modB = _compute_modB(x, field)
                args += [vperp2 / modB]

        return jnp.hstack([x, v]), tuple(args)


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
        Initial particle energy, in eV
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
        R0: ArrayLike,
        phi0: ArrayLike,
        Z0: ArrayLike,
        xi0: ArrayLike,
        E: ArrayLike = 3.5e6,
        m: float = 4,
        q: float = 2,
        eq: Optional[Equilibrium] = None,
    ):
        m = m * proton_mass
        q = q * elementary_charge
        E = E * JOULE_PER_EV
        R0, phi0, Z0, xi0, E, m, q = map(jnp.asarray, (R0, phi0, Z0, xi0, E, m, q))
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
        """Initialize N random particles for a given trajectory model."""
        x = jnp.array([self.R0, self.phi0, self.Z0]).T
        if model.frame == "flux":
            x = x
        elif model.frame == "lab":
            if self.eq is None:
                raise ValueError(
                    "Mapping lab coordinates to flux frame requires an Equilibrium"
                )
            x, _ = self.eq.map_coordinates(
                x,
                inbasis=["R", "phi", "Z"],
                outbasis=["rho", "theta", "zeta"],
                period=[np.inf, 2 * np.pi / self.eq.NFP, np.inf],
            )
        else:
            raise NotImplementedError

        vs = []
        for vcoord in model.vcoords:
            if vcoord == "vpar":
                vs.append(self.vpar0)
            elif vcoord == "v":
                vs.append(self.v0)
            else:
                raise NotImplementedError
        v = jnp.array(vs).T

        args = []
        for arg in model.args:
            if arg == "m":
                args += [self.m]
            elif arg == "q":
                args += [self.q]
            elif arg == "mu":
                vperp2 = self.v0**2 - self.vpar0**2
                modB = _compute_modB(x, field)
                args += [vperp2 / modB]

        return jnp.hstack([x, v]), tuple(args)


class CurveParticleInitializer(AbstractParticleInitializer):
    """Randomly sample particles starting on a curve.

    Parameters
    ----------
    curve : desc.geometry.Curve
        Curve object to initialize samples on.
    N : int
        Number of samples to generate.
    E : float
        Energy of particles, in eV.
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
        self.E = E * JOULE_PER_EV
        self.m = m * proton_mass
        self.q = q * elementary_charge
        self.grid = grid
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.N = N
        self.seed = seed

    def init_particles(
        self, model: AbstractTrajectoryModel, field: Union[Equilibrium, _MagneticField]
    ) -> tuple[jnp.ndarray, tuple]:
        """Initialize N random particles for a given trajectory model."""
        data = self.curve.compute(["x_s", "s", "ds"], grid=self.grid)
        sqrtg = jnp.linalg.norm(data["x_s"], axis=-1) * data["ds"]
        sqrtg /= sqrtg.max()
        nattempts = 10 * self.N  # 10x seems plenty in practice, but should fix
        # rejection sampling according to pdf~sqrtg to get samples
        # roughly equally distributed in real space
        # TODO: use jax rng?
        rng = np.random.default_rng(seed=self.seed)
        idxs = rng.integers(0, sqrtg.shape[0], size=(nattempts,))
        accept = np.where(rng.uniform(low=0, high=1, size=(nattempts,)) < sqrtg[idxs])[
            0
        ]
        # TODO: figure out what to do if this fails, maybe iterate until enough samples?
        assert len(accept) > self.N
        idxs = np.sort(idxs[accept[: self.N]])
        zeta = data["s"][idxs]
        theta = jnp.zeros_like(zeta)
        rho = jnp.zeros_like(zeta)

        if model.frame == "flux":
            x = jnp.array([rho, theta, zeta]).T
        elif model.frame == "lab":
            x = jnp.array([rho, theta, zeta]).T
            grid = Grid(x)
            x = self.curve.compute("x", grid=grid)["x"]

        v = jnp.sqrt(2 * self.E / self.m) * jnp.ones_like(zeta)
        vpar = np.random.uniform(self.xi_min, self.xi_max, v.size) * v
        vs = []
        for vcoord in model.vcoords:
            if vcoord == "vpar":
                vs.append(vpar)
            elif vcoord == "v":
                vs.append(v)
            else:
                raise NotImplementedError
        v = jnp.array(vs).T

        args = []
        for arg in model.args:
            if arg == "m":
                args += [jnp.full(x.shape[0], self.m)]
            elif arg == "q":
                args += [jnp.full(x.shape[0], self.q)]
            elif arg == "mu":
                vperp2 = v**2 - vpar**2
                modB = _compute_modB(x, field)
                args += [vperp2 / modB]

        return jnp.hstack([x, v]), tuple(args)


class SurfaceParticleInitializer(AbstractParticleInitializer):
    """Randomly sample particles starting on a surface.

    Parameters
    ----------
    surface : desc.geometry.Surface
        Surface object to initialize samples on.
    N : int
        Number of samples to generate.
    E : float
        Energy of particles, in eV.
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
        self.E = E * JOULE_PER_EV
        self.m = m * proton_mass
        self.q = q * elementary_charge
        self.grid = grid
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.N = N
        self.seed = seed

    def init_particles(
        self, model: AbstractTrajectoryModel, field: Union[Equilibrium, _MagneticField]
    ) -> tuple[jnp.ndarray, tuple]:
        """Initialize N random particles for a given trajectory model."""
        data = self.surface.compute(
            ["|e_theta x e_zeta|", "theta", "zeta"], grid=self.grid
        )
        sqrtg = data["|e_theta x e_zeta|"]
        sqrtg /= sqrtg.max()
        nattempts = 10 * self.N  # 10x seems plenty in practice, but should fix
        # rejection sampling according to pdf~sqrtg to get samples
        # roughly equally distributed in real space
        # TODO: use jax rng?
        rng = np.random.default_rng(seed=self.seed)
        idxs = rng.integers(0, sqrtg.shape[0], size=(nattempts,))
        accept = np.where(rng.uniform(low=0, high=1, size=(nattempts,)) < sqrtg[idxs])[
            0
        ]
        # TODO: figure out what to do if this fails, maybe iterate until enough samples?
        assert len(accept) > self.N
        idxs = np.sort(idxs[accept[: self.N]])
        zeta = data["zeta"][idxs]
        theta = data["theta"][idxs]
        rho = self.surface.rho * jnp.ones_like(zeta)

        if model.frame == "flux":
            x = jnp.array([rho, theta, zeta]).T
        elif model.frame == "lab":
            x = jnp.array([rho, theta, zeta]).T
            grid = Grid(x)
            x = self.surface.compute("x", grid=grid)["x"]

        v = jnp.sqrt(2 * self.E / self.m) * jnp.ones_like(zeta)
        vpar = np.random.uniform(self.xi_min, self.xi_max, v.size) * v
        vs = []
        for vcoord in model.vcoords:
            if vcoord == "vpar":
                vs.append(vpar)
            elif vcoord == "v":
                vs.append(v)
            else:
                raise NotImplementedError
        v = jnp.array(vs).T

        args = []
        for arg in model.args:
            if arg == "m":
                args += [jnp.full(x.shape[0], self.m)]
            elif arg == "q":
                args += [jnp.full(x.shape[0], self.q)]
            elif arg == "mu":
                vperp2 = v**2 - vpar**2
                modB = _compute_modB(x, field)
                args += [vperp2 / modB]

        return jnp.hstack([x, v]), tuple(args)


def trace_particles(
    field: Union[Equilibrium, _MagneticField],
    y0: ArrayLike,
    args: tuple,
    ts: ArrayLike,
    model: AbstractTrajectoryModel,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    maxstep: int = 1000,
    min_step_size: float = 1e-8,
    solver: AbstractSolver = Tsit5(),
    adjoint=RecursiveCheckpointAdjoint(),
    source_grid: Grid = None,
    bounds_R=(0, np.inf),
    bounds_Z=(-np.inf, np.inf),
):
    """Trace charged particles in an equilibrium or external magnetic field.

    Parameters
    ----------
    field : MagneticField or Equilibrium
        Source of magnetic field to integrate
    y0 : array-like
        Initial particle positions and velocities, stacked in horizontally [x0, v0].
        The first output of `AbstractParticleInitializer.init_particles`.
    args : tuple
        Additional arguments needed by the model, such as mass, charge, and
        magnetic moment of each particle. The second output of
        `AbstractParticleInitializer.init_particles`.
    initializer : AbstractParticleInitializer
        Object to initialize particle distribution.
    ts : array-like
        Strictly increasing array of times where output is desired.
    model : AbstractTrajectoryModel
        Trajectory model to integrate.
    rtol, atol : float
        relative and absolute tolerances for ode integration
    maxstep : int
        maximum number of steps between different output times
    min_step_size: float
        minimum step size (in t) that the integration can take. default is 1e-8
    solver: diffrax.AbstractSolver
        diffrax Solver object to use in integration,
        defaults to Tsit5(), a RK45 explicit solver.
    adjoint : diffrax.AbstractAdjoint
        How to take derivatives of the trajectories. ``RecursiveCheckpointAdjoint``
        supports reverse mode AD and tends to be the most efficient. For forward mode AD
        use ``diffrax.ForwardMode()``.
    source_grid : Grid, optional
        Grid to use to discretize field
    bounds_R : tuple of (float,float), optional
        R bounds for particle tracing bounding box. Trajectories that leave this
        box will be stopped, and NaN returned for points outside the box.
        Defaults to (0,np.inf)
    bounds_Z : tuple of (float,float), optional
        Z bounds for particle tracing bounding box. Trajectories that leave this
        box will be stopped, and NaN returned for points outside the box.
        Defaults to (-np.inf,np.inf)

    Returns
    -------
    x : ndarray, shape(num_particles, num_timesteps, 3)
        Position of each particle at each requested time, in
        either r,phi,z or rho,theta,zeta depending ``model.frame``.
    v : ndarray
        Velocity of each particle at specified times. The exact number of meaning
        will depend on ``model.vcoords``.

    """
    if isinstance(field, Equilibrium):
        assert model.frame == "flux"
    elif isinstance(field, _MagneticField):
        assert model.frame == "lab"
    else:
        raise NotImplementedError

    kwargs = {}
    if source_grid:
        kwargs["source_grid"] = source_grid

    stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=min_step_size)

    saveat = SaveAt(ts=ts)

    def default_terminating_event_fxn(state, **kwargs):
        R_out = jnp.logical_or(state.y[0] < bounds_R[0], state.y[0] > bounds_R[1])
        Z_out = jnp.logical_or(state.y[2] < bounds_Z[0], state.y[2] > bounds_Z[1])
        return jnp.logical_or(R_out, Z_out)

    discrete_terminating_event = DiscreteTerminatingEvent(default_terminating_event_fxn)

    intfun = lambda x, args: diffeqsolve(
        model,
        solver,
        y0=x,
        t0=ts[0],
        t1=ts[-1],
        saveat=saveat,
        max_steps=maxstep * len(ts),
        dt0=min_step_size,
        args=(field, *args, kwargs),
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        event=discrete_terminating_event,
    ).ys

    yt = jit(vmap(intfun))(y0, args)

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
        (2 * mi / me)
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
