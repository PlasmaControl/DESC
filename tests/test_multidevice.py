"""Tests for the multidevice capabilities."""

# This file has to run on a separate process because it changes the number of CPUs
from desc import _set_cpu_count, set_device

num_device = 1
_set_cpu_count(num_device)
set_device(kind="cpu", num_device=num_device)

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction


@pytest.mark.xfail(reason="This test is not working right now")
@pytest.mark.unit
def test_multidevice_jac():
    """Test that the Jacobian is the same for a single and multi device."""
    eq = get("HELIOTRON")
    eq.change_resolution(3, 3, 3, 6, 6, 6)

    # TODO: This doesn't work right now because grid order is not the same
    grid1 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.2, 0.4]), sym=True
    )
    grid2 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.6, 0.8]), sym=True
    )
    grid1 = Grid(grid1.nodes, weights=grid1.weights, spacing=grid1.spacing)
    grid2 = Grid(grid2.nodes, weights=grid2.weights, spacing=grid2.spacing)
    grid3 = Grid(
        jnp.vstack([grid1.nodes, grid2.nodes]),
        weights=jnp.hstack([grid1.weights, grid2.weights]),
        spacing=jnp.hstack([grid1.spacing, grid2.spacing]),
    )

    objective1 = ForceBalance(eq, grid=grid1, device_id=0)
    objective2 = ForceBalance(eq, grid=grid2, device_id=0)
    objective3 = ForceBalance(eq, grid=grid3, device_id=0)

    for obj in [objective1, objective2, objective3]:
        obj.build()
        obj = jax.device_put(obj, device=obj._device)
        obj.things[0] = eq

    obj1 = ObjectiveFunction([objective1, objective2])
    obj2 = ObjectiveFunction(objective3)
    obj1.build(use_jit=False)
    obj2.build(use_jit=False)

    np.testing.assert_allclose(obj1.x(eq), obj2.x(eq))

    np.testing.assert_allclose(
        grid3.nodes, jnp.vstack([grid1.nodes, grid2.nodes]), atol=1e-12, rtol=1e-12
    )
    err1 = objective1.jac_scaled_error(obj1.x(eq))
    err2 = objective2.jac_scaled_error(obj2.x(eq))

    np.testing.assert_allclose(err1, err2)
