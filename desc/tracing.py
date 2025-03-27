"""Functions for tracing particles in magnetic fields."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.constants import (
    Boltzmann,
    electron_mass,
    elementary_charge,
    epsilon_0,
    hbar,
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

    @property
    @abstractmethod
    def args(self):
        """Additional arguments needed by the model.

        Eg, "m", "q", "mu", for mass, charge, magnetic moment.
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

    @property
    def frame(self):
        """Which frame the model is defined in."""
        return "flux"

    @property
    def vcoords(self):
        """Which velocity coordinates the model uses."""
        return ["vpar"]

    @property
    def args(self):
        """Which additional args are needed by the model."""
        return ["m", "q", "mu"]

    def compute(self, x, eq, m, q, mu):
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
        psi, theta, zeta, vpar = x.T
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
            * ((m / q / data["|B|"] ** 3) * (mu * data["|B|"]) + vpar**2)
        ) * (2 * jnp.pi / eq.Psi)
        thetadot = vpar / data["|B|"] * dot(data["B"], data["e^theta"]) + (
            m / q / data["|B|"] ** 3
        ) * ((mu * data["|B|"]) + vpar**2) * dot(
            cross(data["B"], data["grad(|B|)"]), data["e^theta"]
        )
        zetadot = vpar / data["|B|"] * dot(data["B"], data["e^zeta"]) + (
            m / q / data["|B|"] ** 3
        ) * ((mu * data["|B|"].T).T + vpar**2) * dot(
            cross(data["B"], data["grad(|B|)"]), data["e^zeta"]
        )
        vpardot = -mu * dot(data["b"], data["grad(|B|)"])
        dx = jnp.array([psidot, thetadot, zetadot, vpardot]).T
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

    def compute(self, x, eq, m, q):
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
        tau_s = slowing_down_time(data["Te"], data["ne"])
        vc = slowing_down_critical_velocity(data["Te"])

        psidot = (
            dot(cross(data["B"], data["grad(|B|)"]), data["grad(psi)"])
            * ((m / q / data["|B|"] ** 3) * (v**2))
        ) * (2 * jnp.pi / eq.Psi)
        thetadot = vpar / data["|B|"] * dot(data["B"], data["e^theta"]) + (
            m / q / data["|B|"] ** 3
        ) * (v**2) * dot(cross(data["B"], data["grad(|B|)"]), data["e^theta"])
        zetadot = vpar / data["|B|"] * dot(data["B"], data["e^zeta"]) + (
            m / q / data["|B|"] ** 3
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

    def __init__(self, rho0, theta0, zeta0, xi0, E=3.5e6, m=4, q=2, eq=None):
        super().__init__(m, q)
        E = E * JOULE_PER_EV
        rho0, theta0, zeta0, xi0, E = map(jnp.asarray, (rho0, theta0, zeta0, xi0, E))
        rho0, theta0, zeta0, xi0, E = jnp.broadcast_arrays(rho0, theta0, zeta0, xi0, E)
        self.rho0 = rho0
        self.theta0 = theta0
        self.zeta0 = zeta0
        self.v0 = jnp.sqrt(2 * E / self.m)
        self.vpar0 = xi0 * self.v0
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

    def __init__(self, R0, phi0, Z0, xi0, E=3.5e6, m=4, q=2, eq=None):
        super().__init__(m, q)
        E = E * JOULE_PER_EV
        R0, phi0, Z0, xi0, E = map(jnp.asarray, (R0, phi0, Z0, xi0, E))
        R0, phi0, Z0, xi0, E = jnp.broadcast_arrays(R0, phi0, Z0, xi0, E)
        self.R0 = R0
        self.phi0 = phi0
        self.Z0 = Z0
        self.v0 = jnp.sqrt(2 * E / self.m)
        self.vpar0 = xi0 * self.v0
        self.eq = eq

    def init_particles(self, N, model, seed=0):
        """Initialize N random particles for a given trajectory model."""
        x = jnp.array([self.R0, self.phi0, self.Z0]).T
        assert N is None or N == x.shape[0], "got wrong number of requested particles"
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
