"""Profile the FreeSurfaceError objective.

Profiling requires python < 3.14.
  - pip install xprof tensorboard tensorboard_plugin_profile
  - cd DESC/publications/unalmis2025
  - python free_surface_error_profile.py
  - tensorboard --logdir=/tmp/profile-data

"""

import numpy as np

from desc.backend import jax
from desc.examples import get
from desc.grid import LinearGrid
from desc.magnetic_fields import FreeSurfaceOuterField, ToroidalMagneticField
from desc.objectives import ForceBalance, FreeSurfaceError, ObjectiveFunction
from desc.optimize import ProximalProjection

eq = get("W7-X")
grid = LinearGrid(rho=np.array([1.0]), M=8, N=8, NFP=eq.NFP, sym=False)
eval_grid = grid
B_coil = ToroidalMagneticField(5, 1)

field = FreeSurfaceOuterField(eq.surface, M=8, N=8, B_coil=B_coil)
obj = ObjectiveFunction(
    [
        FreeSurfaceError(
            eq,
            field,
            grid=grid,
            eval_grid=eval_grid,
            solve_method="gmres",
            deriv_mode="fwd",
        )
    ]
)
constraint = ObjectiveFunction([ForceBalance(eq)])
prox = ProximalProjection(
    obj, constraint, eq, solve_options={"solve_during_proximal_build": False}
)
prox.build()
x = prox.x(eq)

err = prox.compute_scaled_error(x, prox.constants).block_until_ready()

with jax.profiler.trace("/tmp/profile-data"):
    with jax.profiler.TraceAnnotation("Benchmarking FreeSurfaceError"):
        err = prox.compute_scaled_error(x, prox.constants).block_until_ready()
