"""Functions for tracing particles in magnetic fields."""

import warnings
from abc import ABC, abstractmethod

import equinox as eqx
from diffrax import (
    AbstractTerm,
    Event,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from scipy.constants import Boltzmann, elementary_charge, proton_mass

from desc.backend import jax, jit, jnp, tree_map
from desc.batching import vmap_chunked
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.derivatives import Derivative
from desc.equilibrium import Equilibrium
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

        If anytime during the integration, the particle's rho coordinate becomes
        smaller than 1e-6, the magnetic field is computed at rho = 1e-6 to avoid
        numerical issues with the rho = 0. Note that particle position doesn't have
        discontinuity due to this, only the magnetic field is computed at a different
        point.
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
                "Equilibrium.compute('x')."
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
        xp, yp, zeta, vpar = x
        # we use cartesian-like coordinates xp and yp which are x=rho*cos(theta) and
        # y=rho*sin(theta) for integration, but convert them to flux coordinates for
        # compute functions. This way, we don't have the problem of terminating the
        # integration when rho<0, but actually the particle is still in the plasma
        # volume.
        rho = jnp.sqrt(xp**2 + yp**2)
        theta = jnp.arctan2(yp, xp)
        # compute functions are not correct for very small rho
        rho = jnp.where(rho < 1e-6, 1e-6, rho)
        iota = kwargs.get("iota", None)
        grid = Grid(
            jnp.array([rho, theta, zeta]).T,
            spacing=jnp.zeros((3,)).T,
            jitable=True,
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
        # get the derivative for cartesian-like coordinates
        xpdot = rhodot * jnp.cos(theta) - rho * thetadot * jnp.sin(theta)
        ypdot = rhodot * jnp.sin(theta) + rho * thetadot * jnp.cos(theta)
        # derivative the parallel velocity
        vpardot = -mu / m * dot(data["b"], data["grad(|B|)"])
        dxdt = jnp.array([xpdot, ypdot, zetadot, vpardot]).reshape(x.shape)
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

    def _return_particles(self, x, v, vpar, model, field, params=None, **kwargs):
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
                modB = _compute_modB(x, field, params, **kwargs)
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
            jitable=True,
        )
        profiles = get_profiles("|B|", field, grid)
        transforms = get_transforms("|B|", field, grid, jitable=True)
        if "iota" in kwargs:
            profiles["iota"] = kwargs["iota"]
        return compute_fun(
            field,
            "|B|",
            params=params,
            grid=grid,
            profiles=profiles,
            transforms=transforms,
        )["|B|"]
    source_grid = kwargs.pop("source_grid", None)
    return jnp.linalg.norm(
        field.compute_magnetic_field(x, params=params, source_grid=source_grid), axis=-1
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
    m : array-like
        Particle mass, in proton masses
    q : array-like
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
        rho0, theta0, zeta0, xi0, E, m, q = map(
            jnp.atleast_1d, (rho0, theta0, zeta0, xi0, E, m, q)
        )
        rho0, theta0, zeta0, xi0, E, m, q = jnp.broadcast_arrays(
            rho0, theta0, zeta0, xi0, E, m, q
        )
        self.m = m * proton_mass
        self.q = q * elementary_charge
        self.rho0 = rho0
        self.theta0 = theta0
        self.zeta0 = zeta0
        self.v0 = jnp.sqrt(2 * E * JOULE_PER_EV / self.m)
        self.vpar0 = xi0 * self.v0

        errorif(
            any(jnp.logical_or(self.rho0 > 1.0, self.rho0 < 0.0)),
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
            If you are trying to initialize particles in lab coordinates for a
            MagneticField with flux coordinates, you must pass the Equilibrium
            as keyword argument "eq" in kwargs.

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
                eq = kwargs.pop("eq", None)
                if isinstance(eq, Equilibrium):
                    grid = Grid(
                        x.T,
                        spacing=jnp.zeros_like(x),
                        sort=False,
                        NFP=eq.NFP,
                        jitable=True,
                    )
                    x = eq.compute("x", grid=grid)["x"]
                else:
                    raise NotImplementedError(
                        "If you have a MagneticField object, you cannot use input with "
                        "flux coordinates since there is no easy mapping between the "
                        "two. You can pass the Equilibrium as a keyword argument 'eq' "
                        "in kwargs to enable the mapping."
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
            **kwargs,
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
    m : array-like
        Particle mass, in proton masses
    q : array-like
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
        R0, phi0, Z0, xi0, E, m, q = map(jnp.atleast_1d, (R0, phi0, Z0, xi0, E, m, q))
        R0, phi0, Z0, xi0, E, m, q = jnp.broadcast_arrays(R0, phi0, Z0, xi0, E, m, q)
        self.m = m * proton_mass
        self.q = q * elementary_charge
        self.R0 = R0
        self.phi0 = phi0
        self.Z0 = Z0
        self.v0 = jnp.sqrt(2 * E * JOULE_PER_EV / self.m)
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
        args : jax.Array, shape(N,M)
            Additional arguments needed by the model, such as mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle. M is the number of arguments
            requested by the model which is equal to len(model.args). N is the number
            of particles.
        """
        x = jnp.array([self.R0, self.phi0, self.Z0]).T
        params = field.params_dict
        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please use Equilibrium object with the model."
                )
            # there is not much we can do for the guess here, but we need the
            # guess for jit purposes
            x_guess = jnp.zeros(x.shape)
            x_guess = x_guess.at[:, 2].set(self.phi0)
            x_guess = x_guess.at[:, 1].set(
                jnp.arctan2(self.Z0, self.R0 - field.compute("R0")["R0"])
            )
            tol = 1e-8
            x, out = field.map_coordinates(
                coords=x,
                inbasis=("R", "phi", "Z"),
                outbasis=("rho", "theta", "zeta"),
                maxiter=200,
                guess=x_guess,
                tol=tol,
                full_output=True,
            )
            x = eqx.error_if(
                x,
                (out[0] > tol).any(),
                "Mapping from lab to flux coordinates failed to achieve tolerance "
                f"{tol:.2e}. Make sure the points lie in the equilibrium.",
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
                    "require multiple coordinate mappings, it is not implemented."
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
            **kwargs,
        )


class CurveParticleInitializer(AbstractParticleInitializer):
    """Randomly sample particles starting on a curve.

    Parameters
    ----------
    curve : desc.geometry.Curve
        Curve object to initialize samples on.
    N : int
        Number of particles to generate.
    E : array-like
        Initial particle kinetic energy, in eV
    m : array-like
        Mass of particles, in proton masses.
    q : array-like
        charge of particles, in units of elementary charge.
    xi_min, xi_max : float
        Minimum and maximum values for randomly sampled normalized parallel velocity.
        xi = vpar/v.
    grid : Grid
        Grid used to discretize curve.
    seed : int
        Seed for rng.
    is_curve_magnetic_axis : bool
        Whether user ensures the given curve is the magnetic axis of the equilibrium.
        If True, additional coordinate mapping is not performed, and particles are
        initialized directly on the magnetic axis. Defaults to False. Before setting
        this to True, please make sure the curve is indeed the magnetic axis of the
        equilibrium, and obtained through an equivalent function such as
        ``eq.get_axis()``.
    """

    _static_attrs = ["N", "is_curve_magnetic_axis"]

    def __init__(
        self,
        curve,
        N,
        E=3.5e6,
        m=4,
        q=2,
        xi_min=-1,
        xi_max=1,
        grid=None,
        seed=0,
        is_curve_magnetic_axis=False,
    ):
        self.curve = curve
        E, m, q = map(jnp.atleast_1d, (E, m, q))
        self.E = jnp.broadcast_to(E, (N,)) * JOULE_PER_EV
        self.m = jnp.broadcast_to(m, (N,)) * proton_mass
        self.q = jnp.broadcast_to(q, (N,)) * elementary_charge
        self.grid = grid
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.N = N
        self.seed = seed
        self.is_curve_magnetic_axis = is_curve_magnetic_axis

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
        args : jax.Array, shape(N,M)
            Additional arguments needed by the model, such as mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle. M is the number of arguments
            requested by the model which is equal to len(model.args). N is the number
            of particles.
        """
        data = self.curve.compute(["x_s", "x", "ds", "phi"], grid=self.grid)
        # length of the line segment at each grid point
        sqrtg = jnp.linalg.norm(data["x_s"], axis=-1) * data["ds"]
        self._chosen_idxs = jax.random.choice(
            key=jax.random.PRNGKey(self.seed),
            a=sqrtg.shape[0],
            shape=(self.N,),
            replace=True,
            p=sqrtg / sqrtg.sum(),
        )

        # positions of the selected nodes in R, phi, Z coordinates
        x = jnp.take(data["x"], self._chosen_idxs, axis=0)
        zeta = jnp.take(data["phi"], self._chosen_idxs, axis=0)
        # this is not the best guess, but most likely scenario for this class
        # is to initialize particles on the magnetic axis, for sake of jitable
        # implementation, we use rho=0, theta=0 as the guess
        x_guess = jnp.array([jnp.zeros(self.N), jnp.zeros(self.N), zeta]).T
        params = field.params_dict

        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please use Equilibrium object with the model."
                )
            if not self.is_curve_magnetic_axis:
                tol = 1e-8
                x, out = field.map_coordinates(
                    coords=x,
                    inbasis=("R", "phi", "Z"),
                    outbasis=("rho", "theta", "zeta"),
                    maxiter=200,
                    guess=x_guess,
                    tol=tol,
                    full_output=True,
                )
                x = eqx.error_if(
                    x,
                    (out[0] > tol).any(),
                    "Mapping from lab to flux coordinates failed to achieve tolerance "
                    f"{tol:.2e}. Make sure the curve lies in the equilibrium.",
                )
            else:
                x = x_guess
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

        v = jnp.sqrt(2 * self.E / self.m)
        vpar = jax.random.uniform(
            key=jax.random.PRNGKey(self.seed),
            shape=(v.size,),
            minval=self.xi_min * v,
            maxval=self.xi_max * v,
        )

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
    is_surface_from_eq : bool
        Whether user ensures the given surface is obtained through
        ``eq.get_surface_at(rho=...)``. If True, additional coordinate mapping is not
        performed, and particles are initialized directly on the flux surface. Defaults
        to False. Before setting this to True, make sure the surface and Equilibrium
        have the same theta and rho definitions.
    """

    _static_attrs = ["N", "is_surface_from_eq"]

    def __init__(
        self,
        surface,
        N,
        E=3.5e6,
        m=4,
        q=2,
        xi_min=-1,
        xi_max=1,
        grid=None,
        seed=0,
        is_surface_from_eq=False,
    ):
        self.surface = surface
        E, m, q = map(jnp.atleast_1d, (E, m, q))
        self.E = jnp.broadcast_to(E, (N,)) * JOULE_PER_EV
        self.m = jnp.broadcast_to(m, (N,)) * proton_mass
        self.q = jnp.broadcast_to(q, (N,)) * elementary_charge
        self.grid = grid
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.N = N
        self.seed = seed
        self.is_surface_from_eq = is_surface_from_eq

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
        args : jax.Array, shape(N,M)
            Additional arguments needed by the model, such as mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle. M is the number of arguments
            requested by the model which is equal to len(model.args). N is the number
            of particles.
        """
        data = self.surface.compute(
            ["|e_theta x e_zeta|", "x", "theta", "zeta"], grid=self.grid
        )
        # surface area for each grid point
        # include spacing (dt*dz) to account for non-uniform grids
        sqrtg = data["|e_theta x e_zeta|"] * self.grid.spacing[:, 1:].prod(axis=-1)
        self._chosen_idxs = jax.random.choice(
            key=jax.random.PRNGKey(self.seed),
            a=sqrtg.shape[0],
            shape=(self.N,),
            replace=True,
            p=sqrtg / sqrtg.sum(),
        )
        # eq and surface might not have the same theta definition, so we will do a
        # root finding to find the correct theta and zeta coordinates from R, phi, Z
        # coordinates. We will use the surface's rho, theta, zeta coordinates as an
        # initial guess for the root finding.
        zeta = jnp.take(data["zeta"], self._chosen_idxs, axis=0)
        theta = jnp.take(data["theta"], self._chosen_idxs, axis=0)
        rho = self.surface.rho * jnp.ones_like(zeta)
        x_guess = jnp.array([rho, theta, zeta]).T
        # positions of the selected nodes in R, phi, Z coordinates
        x = jnp.take(data["x"], self._chosen_idxs, axis=0)

        params = field.params_dict

        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Mapping from lab to flux coordinates requires an Equilibrium. "
                    "Please use Equilibrium object with the model."
                )
            if not self.is_surface_from_eq:
                tol = 1e-8
                x, out = field.map_coordinates(
                    coords=x,
                    inbasis=("R", "phi", "Z"),
                    outbasis=("rho", "theta", "zeta"),
                    guess=x_guess,
                    maxiter=200,
                    tol=tol,
                    full_output=True,
                )
                x = eqx.error_if(
                    x,
                    (out[0] > tol).any(),
                    "Mapping from lab to flux coordinates failed to achieve tolerance "
                    f"{tol:.2e}. Make sure the surface lies in the equilibrium.",
                )
            else:
                x = x_guess
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

        v = jnp.sqrt(2 * self.E / self.m)
        vpar = jax.random.uniform(
            key=jax.random.PRNGKey(self.seed),
            shape=(v.size,),
            minval=self.xi_min * v,
            maxval=self.xi_max * v,
        )

        return super()._return_particles(
            x=x, v=v, vpar=vpar, model=model, field=field, params=params, **kwargs
        )


def trace_particles(
    field,
    initializer,
    model,
    ts,
    params=None,
    stepsize_controller=None,
    saveat=None,
    rtol=1e-8,
    atol=1e-8,
    max_steps=None,
    min_step_size=1e-8,
    solver=Tsit5(),
    adjoint=RecursiveCheckpointAdjoint(),
    bounds=None,
    event=None,
    chunk_size=None,
    options=None,
):
    """Trace charged particles in an equilibrium or external magnetic field.

    For jit friendly version of this function, see `_trace_particles`.

    Parameters
    ----------
    field : MagneticField or Equilibrium
        Source of magnetic field to integrate
    initializer : AbstractParticleInitializer
        Particle initializer
    ts : array-like
        Strictly increasing array of time values where output is desired.
    model : AbstractTrajectoryModel
        Trajectory model to integrate with.
    params : dict, optional
        Parameters of the field object, needed for automatic differentiation.
        Defaults to field.params_dict.
    rtol, atol : float, optional
        relative and absolute tolerances for PID stepsize controller. Not used if
        ``stepsize_controller`` is provided.
    stepsize_controller : diffrax.AbstractStepsizeController, optional
        Stepsize controller to use for the integration. If not provided, a
        PIDController with the given ``rtol`` and ``atol`` will be used.
    saveat : diffrax.SaveAt, optional
        SaveAt object to specify where to save the output. If not provided, will
        save at the specified times in ts.
    max_steps : int
        maximum number of steps for whole integration. This will be passed
        to the diffrax.diffeqsolve function. Defaults to
        (ts[1] - ts[0]) * 100 / min_step_size
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
        [[0, inf], [-inf, inf], [-inf, inf]] for R, phi, Z coordinates. Not used if
        ``event`` is provided.
    event : diffrax.Event, optional
        Custom event function to stop integration. If not provided, the default
        event function will be used, which stops integration when particles leave the
        bounds. If integration is stopped by the event, the output will contain NaN
        values for the points outside the bounds.
    chunk_size : int, optional
        Chunk size for integration over particles. If not provided, the integration will
        be done over all particles at once without chunking.
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
        Velocity of each particle at specified times. The exact number of columns
        will depend on ``model.vcoords``.

    """
    if not params:
        params = field.params_dict
    if not options:
        options = {}

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

    def default_event(t, y, args, **kwargs):
        if isinstance(field, Equilibrium):
            i = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
            j = jnp.arctan2(y[1], y[0])
        else:
            i = y[0]
            j = y[1]
        i_out = jnp.logical_or(i < bounds[0, 0], i > bounds[0, 1])
        j_out = jnp.logical_or(j < bounds[1, 0], j > bounds[1, 1])
        k_out = jnp.logical_or(y[2] < bounds[2, 0], y[2] > bounds[2, 1])
        return jnp.logical_or(i_out, jnp.logical_or(j_out, k_out))

    event = Event(default_event) if event is None else event
    max_steps = (
        max_steps
        if max_steps is not None
        else int((ts[1] - ts[0]) / min_step_size * 100)
    )

    y0, model_args = initializer.init_particles(model, field)
    return _trace_particles(
        field=field,
        y0=y0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=params,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=max_steps,
        min_step_size=min_step_size,
        solver=solver,
        adjoint=adjoint,
        event=event,
        chunk_size=chunk_size,
        options=options,
    )


def _trace_particles(
    field,
    y0,
    model,
    model_args,
    ts,
    params,
    stepsize_controller,
    saveat,
    max_steps,
    min_step_size,
    solver,
    adjoint,
    event,
    chunk_size,
    options,
):
    """Trace charged particles in an equilibrium or external magnetic field.

    This is the jit friendly version of the `trace_particles` function. For full
    documentation, see `trace_particles`. This function takes the outputs of
    `initializer.init_particles` as inputs, rather than the particle initializer
    itself. There won't be any checks on the y0 and model_args inputs, so make sure
    they are in the correct format. One can use this function in an objective, where
    the initial positions and velocities of particles are computed in the `build`
    method. If the objective requires initialization of particles at each iteration,
    make sure that the initializer can work under jit compilation which is not the
    case for all of them. All the arguments must be passed with a value, see
    ``trace_particles`` for default values for common use.

    Parameters
    ----------
    y0 : array-like
        Initial particle positions and velocities, stacked in horizontally [x0, v0].
        The first output of ``initializer.init_particles``.
    model_args : array-like
        Additional arguments needed by the model, such as mass, charge, and
        magnetic moment (mv⊥²/2|B|) of each particle. The second output of
        ``initializer.init_particles``.
    """
    # convert cartesian-like for integration in flux coordinates
    if isinstance(field, Equilibrium):
        xp = y0[:, 0] * jnp.cos(y0[:, 1])
        yp = y0[:, 0] * jnp.sin(y0[:, 1])
        y0 = y0.at[:, 0].set(xp)
        y0 = y0.at[:, 1].set(yp)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unhashable type")
        # we only want to map over initial positions and particle arguments
        # Note: vmap with keyword arguments is weird, not using it for now
        yt = vmap_chunked(
            _intfun_wrapper, in_axes=(0, 0) + 12 * (None,), chunk_size=chunk_size
        )(
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

    # convert back to flux coordinates
    if isinstance(field, Equilibrium):
        rho = jnp.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
        theta = jnp.arctan2(x[:, :, 1], x[:, :, 0])
        x = x.at[:, :, 0].set(rho)
        x = x.at[:, :, 1].set(theta)

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
    to recompilations, so instead we define the wrapper here.
    """
    return diffeqsolve(
        terms=model,
        solver=solver,
        y0=x,
        args=[model_args, field, params, options],
        t0=ts[0],
        t1=ts[-1],
        saveat=saveat,
        max_steps=max_steps,
        dt0=min_step_size,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        event=event,
    ).ys
