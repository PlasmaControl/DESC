"""Functions for tracing particles in magnetic fields."""

import warnings
from abc import ABC, abstractmethod

import equinox as eqx
import numpy as np
from diffrax import (
    AbstractTerm,
    Event,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
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
from desc.utils import cross, dot, errorif, safediv

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
        return self._frame

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

    vcoords = ["vpar"]
    args = ["m", "q", "mu"]

    def __init__(self, frame):
        assert frame in ["lab", "flux"]
        self._frame = frame

    @property
    def frame(self):
        """Coordinate frame of the model."""
        return self._frame

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
            Should include the arguments needed by the model, (m, q, mu) as
            an array, Equilibrium or MagneticField object, params and any additional
            keyword arguments needed for magnetic field computation, such as iota
            profile for the Equilibrium, and source_grid for the MagneticField.

        Returns
        -------
        dx : jax.Array, shape(N,4)
            Velocity of particles in phase space.
        """
        x = x.squeeze()
        model_args, eq_or_field, params, kwargs = args
        m, q, mu = model_args
        if self.frame == "flux":
            assert isinstance(
                eq_or_field, Equilibrium
            ), "Integration in flux coordinates requires an Equilibrium."

            return self._compute_flux_coordinates(
                x, eq_or_field, params, m, q, mu, **kwargs
            )
        elif self.frame == "lab":
            assert isinstance(eq_or_field, _MagneticField), (
                "Integration in lab coordinates requires a MagneticField. If using an "
                "Equilibrium, we recommend setting frame='flux' and converting the "
                "output to lab coordinates only at the end by the helper function "
                "Equilibrium.map_coordinates."
            )

            return self._compute_lab_coordinates(
                x, eq_or_field, params, m, q, mu, **kwargs
            )

    def _compute_flux_coordinates(self, x, eq, params, m, q, mu, **kwargs):
        """ODE equation for vacuum guiding center in flux coordinates.

        This function is written for vmap, so it expects x to be a coordinate of a
        single particle, and args to be a tuple of (m, q, mu, eq, params, kwargs) with
        m, q and mu (mv⊥²/2|B|) being scalars. If the Equilibrium does not have
        iota profile, it must be passed as a keyword argument in kwargs. In that case,
        params should also contain i_l, which is the iota profile parameters.
        """
        rho, theta, zeta, vpar = x
        iota = kwargs.get("iota", None)
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
        if iota is not None:
            profiles["iota"] = iota
        data = compute_fun(eq, data_keys, params, transforms, profiles)

        # derivative of the guiding center position in R, phi, Z coordinates
        Rdot = vpar * data["b"] + (
            (m / q / data["|B|"] ** 2)
            * ((mu * data["|B|"] / m) + vpar**2)
            * cross(data["b"], data["grad(|B|)"])
        )
        # take dot product for rho, theta and zeta coordinates
        rhodot = dot(Rdot, data["e^rho"])
        thetadot = dot(Rdot, data["e^theta"])
        zetadot = dot(Rdot, data["e^zeta"])
        vpardot = -mu / m * dot(data["b"], data["grad(|B|)"])
        dxdt = jnp.array([rhodot, thetadot, zetadot, vpardot]).reshape(x.shape)
        return dxdt.squeeze()

    def _compute_lab_coordinates(self, x, field, params, m, q, mu, **kwargs):
        """Compute the RHS of the ODE using MagneticField.

        This function is written for vmap, so it expects x to be a coordinate of a
        single particle, and args to be a tuple of (m, q, mu, field, params, kwargs)
        with m, q and mu (mv⊥²/2|B|) being scalars.

        kwargs can contain source_grid for the magnetic field computation.
        """
        vpar = x[-1]
        coord = x[:-1]

        field_norm_compute = lambda y: jnp.linalg.norm(
            field.compute_magnetic_field(y, params=params, **kwargs), axis=-1
        ).squeeze()

        # magnetic field vector in R, phi, Z coordinates
        B = field.compute_magnetic_field(coord, params=params, **kwargs).squeeze()
        grad_B = Derivative(field_norm_compute, mode="grad")(coord)

        modB = jnp.linalg.norm(B, axis=-1)
        b = B / modB
        # factor of R from grad in cylindrical coordinates
        grad_B = grad_B.at[1].set(safediv(grad_B[1], coord[0]))
        Rdot = vpar * b + (m / q / modB**2 * (mu * modB / m + vpar**2)) * cross(
            b, grad_B
        )
        # velocity and angular velocity are related by the radial coordinate
        # v_phi = R * phi_dot, so phi_dot = v_phi / R
        Rdot = Rdot.at[1].set(safediv(Rdot[1], coord[0]))

        vpardot = jnp.atleast_1d(-mu / m * dot(b, grad_B))
        dxdt = jnp.concatenate([Rdot, vpardot]).reshape(x.shape)
        return dxdt.squeeze()


class SlowingDownGuidingCenterTrajectory(AbstractTrajectoryModel):
    """Guiding center trajectories with slowing down on electrons and main ions.

    Solves the following ODEs,

    d𝐑/dt = v∥ 𝐛 + (m / q B²) ⋅ (v∥² + 1/2 v⊥²) ( 𝐛 × ∇B )

    dv∥/dt = − ((v² - v∥²) / 2B) ( 𝐛 ⋅ ∇B )

    dv/dt = - v / τₛ (1 + v_c³ / v³)

    where 𝐁 is the magnetic field vector at position 𝐑, B is the magnitude of
    the magnetic field, 𝐛 is the unit magnetic field 𝐁/B, τₛ is the Spitzer
    ion–electron momentum exchange time and v_c is the critical velocity associated
    with the critical energy at which the velocity reduction transitions from
    nearly exponential (drag on electrons) to significantly steeper (drag on ions).
    τₛ and v_c are defined as follows:

    τₛ = (mᵢ (4πϵ₀)² 3mₑ¹ᐟ² Tₑ³ᐟ²) / (mₑ 4√(2π) nₑ Zᵢ² e⁴ lnΛ)

    v_c = [ (3√π / 4) (mₑ / mᵢ) ]¹ᐟ³ v_Tₑ

    v_Tₑ = √(2 Tₑ / mₑ)

    See ref [1]_ Eq. (1-4) for definitions, and other references [2]_ [3]_ [4]_
    for the details.

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

    References
    ----------
    .. [1] McMillan M, Lazerson S A. "BEAMS3D neutral beam injection model"
       Plasma Physics and Controlled Fusion (2014).
    .. [2] Callen J D, "Fundamentals of Plasma Physics (Lecture Notes)" (Madison, WI:
       University of Wisconsin Press) (2003)
    .. [3] Fowler R H, Morris R N, Rome J A and Hanatani K, "Neutral beam injection
       benchmark studies for stellarators/heliotrons", Nucl. Fusion 30 997–1010 (1990)
    .. [4] Rosenbluth M N, MacDonald W M and Judd D L, "Fokker–Planck equation for an
       inverse-square force", Phys.Rev. 107 1–6 (1957)

    """

    _Z_eff: float
    _m_eff: float
    vcoords = ["vpar", "v"]
    args = ["m", "q"]

    def __init__(self, frame="flux", m_eff=2.5, Z_eff=1.0):
        assert frame == "flux"
        self._frame = frame
        self._Z_eff = Z_eff
        self._m_eff = m_eff

    @property
    def Z_eff(self):
        """Coordinate frame of the model."""
        return self._Z_eff

    @property
    def m_eff(self):
        """Coordinate frame of the model."""
        return self._m_eff

    @property
    def frame(self):
        """Coordinate frame of the model."""
        return self._frame

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
            Should include the arguments needed by the model, (m, q) as
            an array, Equilibrium object, params and any additional keyword
            arguments needed for magnetic field computation, i.e. iota profile for
            the Equilibrium. Note: if Equilibrium does not have iota profile,
            params dictionary must contain i_l, which is the iota profile parameters.

        Returns
        -------
        dx : jax.Array, shape(N,5)
            Velocity of particles in phase space.
        """
        rho, theta, zeta, vpar, v = x
        model_args, eq, params, kwargs = args
        m, q = model_args
        iota = kwargs.get("iota", None)

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
        if iota is not None:
            profiles["iota"] = iota
        data = compute_fun(
            eq,
            data_keys,
            params,
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
        # take dot product for rho, theta and zeta coordinates
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
    def init_particles(self, model, field):
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
            the model, parallel velocity and total velocity. The initial positions are
            in the frame of the model.
        args : jax.Array, shape(N,M)
            Additional arguments needed by the model, such as mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle. M is the number of arguments
            requested by the model which is equal to len(model.args). N is the number
            of particles.
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

        args = jnp.array(args).T
        return jnp.hstack([x, v0]), args


def _compute_modB(x, field, params, **kwargs):
    if isinstance(field, Equilibrium):
        grid = Grid(
            x.T,
            spacing=jnp.zeros_like(x),
            sort=False,
            NFP=field.NFP,
        )
        profiles = get_profiles("|B|", field, grid)
        if "iota" in kwargs:
            profiles["iota"] = kwargs["iota"]
        return field.compute("|B|", params=params, grid=grid, profiles=profiles)["|B|"]
    return jnp.linalg.norm(
        field.compute_magnetic_field(x, params=params, **kwargs), axis=-1
    )


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
    """

    def __init__(
        self,
        rho0,
        theta0,
        zeta0,
        xi0,
        E=3.5e6,
        m=4,
        q=2,
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

        errorif(
            any(self.rho0 > 1.0 or self.rho0 < 0.0),
            ValueError,
            "Flux coordinate rho must be between 0 and 1.",
        )

    def init_particles(self, model, field, **kwargs):
        """Initialize particles for a given trajectory model.

        Parameters
        ----------
        model : AbstractTrajectoryModel
            Model to use for tracing particles, which defines the frame and
            velocity coordinates.
        field : Equilibrium or _MagneticField
            Source of magnetic field to use for tracing particles.
        kwargs : dict, optional
            source_grid for the magnetic field computation, if using a MagneticField
            object, can be passed as a keyword argument.

        Returns
        -------
        x0 : jax.Array, shape(N,D)
            Initial particle positions and velocities, where D is the dimensionality of
            the trajectory model, which includes 3D spatial dimensions and depending on
            the model, parallel velocity and total velocity. The initial positions are
            in the frame of the model.
        args : tuple
            Additional arguments needed by the model, mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle.
        """
        x = jnp.array([self.rho0, self.theta0, self.zeta0]).T
        params = field.params_dict
        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please use Equilibrium object with the model."
                )
            x = x
            if field.iota is None:
                iota = field.get_profile("iota")
                params["i_l"] = iota.params
                kwargs["iota"] = iota
        elif model.frame == "lab":
            if isinstance(field, Equilibrium):
                raise NotImplementedError(
                    "If you have an Equilibrium object, you should use the model "
                    "in flux frame. Since trying to integrate in lab frame will "
                    "require multiple coordinate mapping, it is not implemented."
                )
            elif isinstance(field, _MagneticField):
                raise NotImplementedError(
                    "If you have a MagneticField object, you cannot use input with "
                    "flux coordinates since there is no easy mapping between the two."
                )
        else:
            raise NotImplementedError

        return super()._return_particles(
            x=x,
            v=self.v0,
            vpar=self.vpar0,
            model=model,
            field=field,
            params=params,
            **kwargs
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

    def init_particles(self, model, field, **kwargs):
        """Initialize particles for a given trajectory model.

        Parameters
        ----------
        model : AbstractTrajectoryModel
            Model to use for tracing particles, which defines the frame and
            velocity coordinates.
        field : Equilibrium or _MagneticField
            Source of magnetic field to use for tracing particles.
        kwargs : dict, optional
            source_grid for the magnetic field computation, if using a MagneticField
            object, can be passed as a keyword argument.

        Returns
        -------
        x0 : jax.Array, shape(N,D)
            Initial particle positions and velocities, where D is the dimensionality of
            the trajectory model, which includes 3D spatial dimensions and depending on
            the model, parallel velocity and total velocity. The initial positions are
            in the frame of the model.
        args : tuple
            Additional arguments needed by the model, mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle.
        """
        x = jnp.array([self.R0, self.phi0, self.Z0]).T
        params = field.params_dict
        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please use Equilibrium object with the model."
                )
            x = field.map_coordinates(
                coords=x,
                inbasis=("R", "phi", "Z"),
                outbasis=("rho", "theta", "zeta"),
            )
            if field.iota is None:
                iota = field.get_profile("iota")
                params["i_l"] = iota.params
                kwargs["iota"] = iota
        elif model.frame == "lab":
            if isinstance(field, Equilibrium):
                raise NotImplementedError(
                    "If you have an Equilibrium object, you should use the model "
                    "in flux frame. Since trying to integrate in lab frame will "
                    "require smultiple coordinate mapping, it is not implemented."
                )
            x = x
        else:
            raise NotImplementedError

        return super()._return_particles(
            x=x,
            v=self.v0,
            vpar=self.vpar0,
            model=model,
            field=field,
            params=params,
            **kwargs
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

    def init_particles(self, model, field, **kwargs):
        """Initialize particles for a given trajectory model.

        Parameters
        ----------
        model : AbstractTrajectoryModel
            Model to use for tracing particles, which defines the frame and
            velocity coordinates.
        field : Equilibrium or _MagneticField
            Source of magnetic field to use for tracing particles.
        kwargs : dict, optional
            source_grid for the magnetic field computation, if using a MagneticField
            object, can be passed as a keyword argument.

        Returns
        -------
        x0 : jax.Array, shape(N,D)
            Initial particle positions and velocities, where D is the dimensionality of
            the trajectory model, which includes 3D spatial dimensions and depending on
            the model, parallel velocity and total velocity. The initial positions are
            in the frame of the model.
        args : tuple
            Additional arguments needed by the model, mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle.
        """
        data = self.curve.compute(["x_s", "s", "ds"], grid=self.grid)
        sqrtg = jnp.linalg.norm(data["x_s"], axis=-1) * data["ds"]
        idxs = _find_random_indices(sqrtg, self.N, seed=self.seed)

        zeta = data["s"][idxs]
        theta = jnp.zeros_like(zeta)
        rho = jnp.zeros_like(zeta)
        params = field.params_dict

        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please use Equilibrium object with the model."
                )
            x = jnp.array([rho, theta, zeta]).T
            if field.iota is None:
                iota = field.get_profile("iota")
                params["i_l"] = iota.params
                kwargs["iota"] = iota
        elif model.frame == "lab":
            if isinstance(field, Equilibrium):
                raise NotImplementedError(
                    "If you have an Equilibrium object, you should use the model "
                    "in flux frame. Since trying to integrate in lab frame will "
                    "require multiple coordinate mapping, it is not implemented."
                )
            x = jnp.array([rho, theta, zeta]).T
            grid = Grid(x)
            x = self.curve.compute("x", grid=grid)["x"]

        v = jnp.sqrt(2 * self.E / self.m)
        vpar = np.random.uniform(self.xi_min, self.xi_max, v.size) * v

        return super()._return_particles(
            x=x, v=v, vpar=vpar, model=model, field=field, params=params, **kwargs
        )


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

    def init_particles(self, model, field, **kwargs):
        """Initialize particles for a given trajectory model.

        Parameters
        ----------
        model : AbstractTrajectoryModel
            Model to use for tracing particles, which defines the frame and
            velocity coordinates.
        field : Equilibrium or _MagneticField
            Source of magnetic field to use for tracing particles.
        kwargs : dict, optional
            source_grid for the magnetic field computation, if using a MagneticField
            object, can be passed as a keyword argument.

        Returns
        -------
        x0 : jax.Array, shape(N,D)
            Initial particle positions and velocities, where D is the dimensionality of
            the trajectory model, which includes 3D spatial dimensions and depending on
            the model, parallel velocity and total velocity. The initial positions are
            in the frame of the model.
        args : tuple
            Additional arguments needed by the model, mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle.
        """
        data = self.surface.compute(
            ["|e_theta x e_zeta|", "theta", "zeta"], grid=self.grid
        )
        sqrtg = data["|e_theta x e_zeta|"]
        idxs = _find_random_indices(sqrtg, self.N, seed=self.seed)

        zeta = data["zeta"][idxs]
        theta = data["theta"][idxs]
        rho = self.surface.rho * jnp.ones_like(zeta)
        params = field.params_dict

        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please use Equilibrium object with the model."
                )
            x = jnp.array([rho, theta, zeta]).T
            if field.iota is None:
                iota = field.get_profile("iota")
                params["i_l"] = iota.params
                kwargs["iota"] = iota
        elif model.frame == "lab":
            if isinstance(field, Equilibrium):
                raise NotImplementedError(
                    "If you have an Equilibrium object, you should use the model "
                    "in flux frame. Since trying to integrate in lab frame will "
                    "require multiple coordinate mapping, it is not implemented."
                )
            x = jnp.array([rho, theta, zeta]).T
            grid = Grid(x)
            x = self.surface.compute("x", grid=grid)["x"]

        v = jnp.sqrt(2 * self.E / self.m)
        vpar = np.random.uniform(self.xi_min, self.xi_max, v.size) * v

        return super()._return_particles(
            x=x, v=v, vpar=vpar, model=model, field=field, parmas=params, **kwargs
        )


def _find_random_indices(sqrtg, N, seed):
    """Find random indices for sampling particles on a surface or curve."""
    # probability of particle generation in a given grid point is proportional to
    # its volume/area/length, which is sqrtg. Normalize sqrtg for random number
    # generation limit of 1
    sqrtg /= sqrtg.max()
    nattempts = 10 * N
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
        idxs = idxs[accept]

    return np.sort(idxs)


def trace_particles(
    field,
    y0,
    model,
    model_args,
    ts,
    params=None,
    stepsize_controller=None,
    saveat=None,
    rtol=1e-8,
    atol=1e-8,
    max_steps=1000,
    min_step_size=1e-10,
    solver=Tsit5(),
    adjoint=RecursiveCheckpointAdjoint(),
    bounds=None,
    event=None,
    options={},
):
    """Trace charged particles in an equilibrium or external magnetic field.

    Parameters
    ----------
    field : MagneticField or Equilibrium
        Source of magnetic field to integrate
    y0 : array-like
        Initial particle positions and velocities, stacked in horizontally [x0, v0].
        The first output of ``ParticleInitializer.init_particles``.
    model_args : array-like
        Additional arguments needed by the model, such as mass, charge, and
        magnetic moment (mv⊥²/2|B|) of each particle. The second output of
        ``ParticleInitializer.init_particles``.
    ts : array-like
        Strictly increasing array of time values where output is desired.
    model : AbstractTrajectoryModel
        Trajectory model to integrate with.
    params : dict, optional
        Parameters of the field object, needed for automatic differentiation.
        Defaults to field.params_dict.
    rtol, atol : float, optional
        relative and absolute tolerances for PID stepsize controller. Not used if
        stepsize_controller is provided.
    stepsize_controller : diffrax.AbstractStepsizeController, optional
        Stepsize controller to use for the integration. If not provided, a
        PIDController with the given rtol and atol will be used.
    saveat : diffrax.SaveAt, optional
        SaveAt object to specify where to save the output. If not provided, will
        save at the specified times in ts.
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
    bounds : array of shape(3, 2), optional
        Bounds for particle tracing bounding box. Trajectories that leave this
        box will be stopped, and NaN returned for points outside the box. When tracing
        an Equilibrium, the default bounds are set to
        [[0, 1], [-inf, inf], [-inf, inf]] for rho, theta, zeta coordinates.
        When tracing a MagneticField, the default bounds are set to
        [[0, inf], [-inf, inf], [-inf, inf]] for R, phi, Z coordinates.
    event : diffrax.Event, optional
        Custom event function to stop integration. If not provided, the default
        event function will be used, which stops integration when particles leave the
        bounds. If integration is stopped by the event, the output will contain NaN
        values for the points outside the bounds.
    options : dict, optional
        Additional keyword arguments to pass to the field computation, such as
            - iota : Profile
                Iota profile of the Equilibrium, if it does not have one.
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
    if not params:
        params = field.params_dict

    stepsize_controller = (
        stepsize_controller
        if stepsize_controller is not None
        else PIDController(rtol=rtol, atol=atol, dtmin=min_step_size)
    )

    saveat = saveat if saveat is not None else SaveAt(ts=ts)
    if bounds is None:
        bounds = jnp.array([[0, jnp.inf], [-jnp.inf, jnp.inf], [-jnp.inf, jnp.inf]])
        if isinstance(field, Equilibrium):
            bounds = bounds.at[0, 1].set(1.0)  # rho bounds for flux coordinates

    def default_terminating_event(t, y, args, **kwargs):
        i_out = jnp.logical_or(y[0] < bounds[0, 0], y[0] > bounds[0, 1])
        j_out = jnp.logical_or(y[1] < bounds[1, 0], y[1] > bounds[1, 1])
        k_out = jnp.logical_or(y[2] < bounds[2, 0], y[2] > bounds[2, 1])
        return jnp.logical_or(i_out, jnp.logical_or(j_out, k_out))

    event = Event(default_terminating_event) if event is None else event

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unhashable type")
        # we only want to map over initial positions and particle arguments,
        # other arguments are there to prevent closing over them, and hence
        # reduce the number of recompilations in, for example, an optimization
        # loop. Note: vmap with keyword arguments is weird, not using it for now
        yt = vmap(_intfun_wrapper, in_axes=(0, 0) + 12 * (None,))(
            y0,
            model_args,
            field,
            params,
            ts,
            max_steps,
            min_step_size,
            solver,
            stepsize_controller,
            adjoint,
            event,
            model,
            saveat,
            options,
        )

    yt = jnp.where(jnp.isinf(yt), jnp.nan, yt)

    x = yt[:, :, :3]
    v = yt[:, :, 3:]

    return x, v


def _intfun_wrapper(
    x,
    model_args,
    field,
    params,
    ts,
    max_steps,
    min_step_size,
    solver,
    stepsize_controller,
    adjoint,
    event,
    model,
    saveat,
    options,
):
    """Wrapper for the integration function for vectorized inputs.

    Defining a lambda function inside the ``trace_particles`` function leads
    to recompilations.
    """
    return diffeqsolve(
        terms=model,
        solver=solver,
        y0=x,
        args=[model_args, field, params, options],
        t0=ts[0],
        t1=ts[-1],
        saveat=saveat,
        max_steps=int(max(max_steps, (ts[1] - ts[0]) / min_step_size) * len(ts)),
        dt0=min_step_size,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        event=event,
    ).ys


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
