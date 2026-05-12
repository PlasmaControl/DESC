"""Profile bounce integral objective.

Dynamic shape support jax-finufft.
  - cd DESC
  - cd ../
  - git clone git@github.com:unalmis/jax-finufft.git
  - cd jax-finufft
  - git switch ku/dynamic
  - cd ../DESC
  - conda install -c conda-forge fftw cxx-compiler
  - pip install ../jax-finufft
  - Set JF_BUG = False in desc/integrals/_interp_utils.py
  - Build GPU stuff (or the open mp stuff for CPU)

Profiling requires python < 3.14.
  - pip install xprof tensorboard tensorboard_plugin_profile
  - cd DESC/publications/unalmis2025
  - python effective_ripple_profile.py
  - tensorboard --logdir=/tmp/profile-data

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
