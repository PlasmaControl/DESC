"""Script to profile effective ripple objective.

pip install xprof tensorflow

python effective_ripple_profile.py
tensorboard --logdir=/tmp/profile-data
"""

import numpy as np

from desc.backend import jax
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import EffectiveRipple, ObjectiveFunction

eq = get("W7-X")
rho = np.linspace(0.1, 1, 10)
grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)

num_transit = 15
obj = ObjectiveFunction(
    [
        EffectiveRipple(
            eq,
            grid=grid,
            X=32,
            Y=32,
            Y_B=100,
            num_transit=num_transit,
            num_well=16 * num_transit,
            num_quad=32,
            num_pitch=101,
        )
    ]
)
obj.build()
x = obj.x(eq)
eps = obj.jac_scaled_error(x).block_until_ready()

with jax.profiler.trace("/tmp/profile-data"):
    with jax.profiler.TraceAnnotation("Benchmarking the Jacobian"):
        eps = obj.jac_scaled_error(x).block_until_ready()
