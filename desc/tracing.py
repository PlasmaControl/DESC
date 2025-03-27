"""Functions for tracing particles in magnetic fields."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.constants import (
    Boltzmann,
    electron_mass,
    elementary_charge,
    epsilon_0,
    proton_mass,
)

from desc.backend import jnp
from desc.compute import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import Grid
from desc.io import IOAble
from desc.utils import cross, dot

JOULE_PER_EV = 11606 * Boltzmann
EV_PER_JOULE = 1 / JOULE_PER_EV


class AbstractTrajectoryModel(IOAble, ABC):
    """Abstract base class for particle trajectory models.

    Subclasses should implement the ``compute`` method to compute the RHS of the ODE,
    """

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
    def vcoords(self):
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

    @abstractmethod
    def compute(self, x, eq):
        """RHS of the particle trajectory ODE."""
        pass


class CollisionlessGuidingCenterTrajectory(AbstractTrajectoryModel):
    """Guiding center trajectories without collisions, conserving energy and mu.

    Parameters
    ----------
    m : float or array-like, shape(num_particles,)
        Mass of particles, in units of proton masses
    q : float or array-like, shape(num_particles,)
        Charge of particles, in units of elementary charge.
    """

    _data_keys = [
        "B",
        "|B|",
        "grad(|B|)",
        "grad(psi)",
        "e^theta",
        "e^zeta",
        "b",
    ]

    def __init__(self, m=4, q=2):
        self.m = m * proton_mass
        self.q = q * elementary_charge

    @property
    def frame(self):
        """Which frame the model is defined in."""
        return "flux"

    @property
    def vcoords(self):
        """Which velocity coordinates the model uses."""
        return ["vpar"]

    def compute(self, x, eq):
        """RHS of guiding center trajectories without collisions or slowing down.

        Parameters
        ----------
        x : jax.Array, shape(N,4)
            Position of particle in phase space (psi_n, theta, zeta, vpar).
        eq : Equilibrium
            Equilibrium object particles are being traced in.

        Returns
        -------
        dx : jax.Array, shape(N,4)
            Velocity of particles in phase space.
        """
        assert eq.iota is not None
        psi, theta, zeta, vpar, v = x.T
        grid = Grid(
            jnp.array([jnp.sqrt(psi), theta, zeta]).T,
            spacing=jnp.zeros((3,)).T,
            jitable=True,
            sort=False,
        )
        transforms = get_transforms(self._data_keys, eq, grid, jitable=True)
        profiles = get_profiles(self._data_keys, eq, grid)
        data = compute_fun(
            eq,
            self._data_keys,
            eq.params_dict,
            transforms,
            profiles,
        )

        psidot = (
            dot(cross(data["B"], data["grad(|B|)"]), data["grad(psi)"])
            * ((self.m / self.q / data["|B|"] ** 3) * (v**2))
        ) * (2 * jnp.pi / eq.Psi)
        thetadot = vpar / data["|B|"] * dot(data["B"], data["e^theta"]) + (
            self.m / self.q / data["|B|"] ** 3
        ) * (v**2) * dot(cross(data["B"], data["grad(|B|)"]), data["e^theta"])
        zetadot = vpar / data["|B|"] * dot(data["B"], data["e^zeta"]) + (
            self.m / self.q / data["|B|"] ** 3
        ) * (v**2) * dot(cross(data["B"], data["grad(|B|)"]), data["e^zeta"])
        vpardot = (
            -(v**2 - vpar**2) / (2 * data["|B|"]) * dot(data["b"], data["grad(|B|)"])
        )
        vdot = jnp.zeros_like(v)
        dx = jnp.array([psidot, thetadot, zetadot, vpardot, vdot]).T
        return dx


class SlowingDownGuidingCenterTrajectory(AbstractTrajectoryModel):
    """Guiding center trajectories with slowing down on electrons and main ions.

    Parameters
    ----------
    m_q : jax.Array
        Mass/charge ratio of particles. Defaults to that of an alpha particle.
    """

    _data_keys = [
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

    def __init__(self, m=4, q=2):
        self.m = m * proton_mass
        self.q = q * elementary_charge

    @property
    def frame(self):
        """Which frame the model is defined in."""
        return "flux"

    @property
    def vcoords(self):
        """Which velocity coordinates the model uses."""
        return ["vpar", "v"]

    def compute(self, x, eq):
        """RHS of guiding center trajectories without collisions or slowing down.

        Parameters
        ----------
        x : jax.Array, shape(N,5)
            Position of particle in phase space (psi_n, theta, zeta, vpar, v).
        eq : Equilibrium
            Equilibrium object particles are being traced in.

        Returns
        -------
        dx : jax.Array, shape(N,5)
            Velocity of particles in phase space.
        """
        assert eq.iota is not None
        assert eq.electron_temperature is not None

        psi, theta, zeta, vpar, v = x.T
        grid = Grid(
            jnp.array([jnp.sqrt(psi), theta, zeta]).T,
            spacing=jnp.zeros((3,)).T,
            jitable=True,
            sort=False,
        )
        transforms = get_transforms(self._data_keys, eq, grid, jitable=True)
        profiles = get_profiles(self._data_keys, eq, grid)
        data = compute_fun(
            eq,
            self._data_keys,
            eq.params_dict,
            transforms,
            profiles,
        )

        # slowing eqns from McMillan, Matthew, and Samuel A. Lazerson. "BEAMS3D
        # neutral beam injection model." Plasma Physics and Controlled Fusion (2014)
        me = electron_mass
        mi = 2 * proton_mass  # assuming deuterium
        Z = 1  # assuming hydrogenic species
        Te = data["Te"] * JOULE_PER_EV  # in Joules
        ne = data["ne"]  # in particles/m^3

        vTe = jnp.sqrt(2 * Te / electron_mass)
        vc = (3 * jnp.sqrt(jnp.pi) / 4 * me / mi) ** (1 / 3) * vTe

        lnlambda = jnp.where(
            data["Te"] > 10 * Z**2,
            24 - jnp.log(jnp.sqrt(data["ne"]) / data["Te"]),
            23 - jnp.log(jnp.sqrt(data["ne"] * Z * data["Te"] ** (-3 / 2))),
        )

        tau_s = (
            (2 * proton_mass / electron_mass)
            * (4 * jnp.pi * epsilon_0) ** 2
            / (4 * jnp.sqrt(2 * jnp.pi))
            * (3 * me ** (1 / 2) * Te ** (3 / 2))
            / (ne * Z**2 * elementary_charge**4 * lnlambda)
        )

        psidot = (
            dot(cross(data["B"], data["grad(|B|)"]), data["grad(psi)"])
            * ((self.m / self.q / data["|B|"] ** 3) * (v**2))
        ) * (2 * jnp.pi / eq.Psi)
        thetadot = vpar / data["|B|"] * dot(data["B"], data["e^theta"]) + (
            self.m / self.q / data["|B|"] ** 3
        ) * (v**2) * dot(cross(data["B"], data["grad(|B|)"]), data["e^theta"])
        zetadot = vpar / data["|B|"] * dot(data["B"], data["e^zeta"]) + (
            self.m / self.q / data["|B|"] ** 3
        ) * (v**2) * dot(cross(data["B"], data["grad(|B|)"]), data["e^zeta"])
        vpardot = (
            -(v**2 - vpar**2) / (2 * data["|B|"]) * dot(data["b"], data["grad(|B|)"])
        )
        vdot = -v / tau_s * (1 + vc**3 / v**3)
        dx = jnp.array([psidot, thetadot, zetadot, vpardot, vdot]).T
        return dx


class AbstractParticleInitializer(ABC, IOAble):
    """ABC for initial distribution of particles for tracing."""

    def __init__(self, m=4, q=2):
        self.m = m * proton_mass
        self.q = q * elementary_charge

    @abstractmethod
    def init_particles(self, N):
        """Initialize a distribution of N particles."""
        pass


class ManualParticleInitializerFlux(AbstractParticleInitializer):
    """Manually specify particle starting positions and energy in flux coordinates."""

    def __init__(self, rho0, theta0, zeta0, lambda0, E=3.5e6, m=4, q=2, eq=None):
        super().__init__(m, q)
        self.E = E * JOULE_PER_EV
        self.rho0 = jnp.asarray(rho0)
        self.theta0 = jnp.asarray(theta0)
        self.zeta0 = jnp.asarray(zeta0)
        self.v0 = jnp.sqrt(2 * self.E / self.m)
        vperp0 = jnp.sqrt(lambda0) * self.v0
        self.vpar0 = jnp.sqrt(self.v0**2 - vperp0**2)
        self.eq = eq

    def init_particles(self, N, model, seed=0):
        """Initialize N random particles for a given trajectory model."""
        x = jnp.array([self.rho0, self.theta0, self.zeta0]).T
        assert N is None or N == x.shape[0], "got wrong number of requested particles"
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
        return jnp.hstack([x, v])


class CurveParticleInitializer(AbstractParticleInitializer):
    """Randomly sample particles starting on a curve.

    Parameters
    ----------
    curve : desc.geometry.Curve
        Curve object to initialize samples on
    E : float
        Energy of particles, in eV
    m : float
        Mass of particles, in proton masses
    q : float
        charge of particles, in units of elementary charge.
    xi_min, xi_max : float
        Minimum and maximum values for randomly sampled normalized parallel velocity.
        xi = vpar/v
    grid : Grid
        Grid used to discretize curve.

    """

    def __init__(self, curve, E=3.5e6, m=4, q=2, xi_min=-1, xi_max=1, grid=None):
        self.curve = curve
        self.E = E * JOULE_PER_EV
        self.m = m * proton_mass
        self.q = q * elementary_charge
        self.grid = grid
        self.xi_min = xi_min
        self.xi_max = xi_max

    def init_particles(self, N, model, seed=0):
        """Initialize N random particles for a given trajectory model."""
        data = self.curve.compute(["x_s", "s", "ds"], grid=self.grid)
        sqrtg = jnp.linalg.norm(data["x_s"]) * data["ds"]
        sqrtg /= sqrtg.max()
        nattempts = 10 * N  # 10x seems plenty in practice, but should fix
        # rejection sampling according to pdf~sqrtg to get samples
        # roughly equally distributed in real space
        # TODO: use jax rng?
        rng = np.random.default_rng(seed=seed)
        idxs = rng.randint(0, sqrtg.shape[0], size=(nattempts,))
        accept = np.where(rng.uniform(low=0, high=1, size=(nattempts,)) < sqrtg[idxs])[
            0
        ]
        # TODO: figure out what to do if this fails, maybe iterate until enough samples?
        assert len(accept) > N
        idxs = np.sort(idxs[accept[:N]])
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
        return jnp.hstack([x, v])
