"""Objectives for optimizing the equilibrium from tracing particles using Diffrax solver"""

import warnings

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import Grid
from jax.experimental.ode import odeint as jax_odeint
from functools import partial
from jax import jit, vmap
from .objective_funs import _Objective

class ParticleTracer(_Objective):
    """Particle Tracer using Guiding Center equations of motion.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    output_time : ndarray
        Values of time where the system is evaluated.
    initial_conditions : tuple, array
        Initial conditions (psi, theta, zeta, vpar) to solve the system of equations.
        Starting state of the system.
    initial_parameters : tuple, array
        Parameters needed in the system, such as the magnetic momentum, mu, and the mass-charge ratio, m_q.
    compute_option: str
        Select the compute() output. Can be "optimization" for the optimization metric; "tracer" for the full 
        solution of the system; "average psi/theta/zeta/vpar" for the mean value of psi/theta/zeta/vpar in the 
        computed time.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    name : str
        Name of the objective function.
    """
    
    _scalar = False
    _linear = False
    _units = ""
    _print_value_fmt = "System solution: {:10.3e}"

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        output_time=None,
        initial_conditions=None,
        initial_parameters=None,
        compute_option=None,
        tolerance = 1.4e-8,
        deriv_mode = "rev",
        name="Particle Tracer"
    ):
        self.output_time = output_time
        self.initial_conditions=jnp.asarray(initial_conditions) 
        self.initial_parameters=jnp.asarray(initial_parameters)
        self.compute_option=compute_option
        self.tolerance = tolerance
        
        if target is None and bounds is None:
            target = 0
        
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            deriv_mode=deriv_mode,
            name=name,
        )

        self._print_value_fmt = (
            "System solution for initial conditions"
        )

    def build(self, eq=None, use_jit=True, verbose=1):
        
        self._data_keys = ["psidot", "thetadot", "zetadot", "vpardot"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )
        
        self.charge = 1.6e-19
        self.mass = 1.673e-27
        self.Energy = 3.52e6*self.charge 
        eq = eq or self._things[0]

        if self.compute_option == "optimization":
            self._dim_f = 1
        elif self.compute_option == "tracer":
            self._dim_f = [len(self.output_time), 4]
        elif self.compute_option.startswith("average "):
            self._dim_f = 1

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):

        constants = constants or self.constants

        @jit
        def system(initial_conditions = self.initial_conditions, t = self.output_time, initial_parameters = self.initial_parameters):
            psi, theta, zeta, vpar = initial_conditions
            grid = Grid(jnp.array([jnp.sqrt(psi), theta, zeta]).T, spacing=jnp.zeros((3,)).T, jitable=True, sort=False)
            transforms = get_transforms(self._data_keys, self._things[0], grid, jitable=True)
            profiles = get_profiles(self._data_keys, self._things[0], grid)
            data = compute_fun("desc.equilibrium.equilibrium.Equilibrium", self._data_keys, params, transforms, profiles, 
                               mu=initial_parameters[0], m_q=initial_parameters[1], vpar=vpar)

            return jnp.array([data[f"{key}dot"] for key in ['psi', 'theta', 'zeta', 'vpar']])

        initial_conditions_jax = jnp.array(self.initial_conditions, dtype=jnp.float64)
        initial_parameters_jax = jnp.array(self.initial_parameters, dtype=jnp.float64)
        
        
        #solution = jax_odeint(partial(system_jit, initial_parameters=self.initial_parameters), initial_conditions_jax, self.output_time, rtol=self.tolerance)
        intfun = lambda initial_conditions_jax, initial_parameters_jax: jax_odeint(partial(system, initial_parameters=initial_parameters_jax), 
                                                           initial_conditions_jax, self.output_time, rtol=self.tolerance)

        solution = vmap(intfun)(initial_conditions_jax, initial_parameters_jax)

        if self.compute_option == "optimization":
            new = jnp.repeat(solution[:, :, 0][:, 0:1], solution[:, :, 0].shape[1], axis=1)
            return jnp.sum(jnp.mean((solution[:, :, 0] - new)**2, axis=-1), axis=-1)
        elif self.compute_option == "tracer":
            return solution
        elif self.compute_option.startswith("average "):
            index = ["psi", "theta", "zeta", "vpar"].index(self.compute_option.split()[-1])
            return jnp.mean(solution[:, index])