"""Tests for the multidevice capabilities."""

import warnings

# This file has to run on a separate process because it changes the number of CPUs
from desc import _set_cpu_count, set_device

num_device = 2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _set_cpu_count(num_device)
    set_device(kind="cpu", num_device=num_device)

import numpy as np
import pytest

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("mpi4py is not installed, skipping MPI tests.")
    pytest.skip("mpi4py is not installed, skipping MPI tests.", allow_module_level=True)

from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction


@pytest.mark.mpi
def test_multidevice_jac():
    """Test that the Jacobian is the same for a single and multi device."""
    eq = get("HELIOTRON")
    eq.change_resolution(6, 6, 3, 12, 12, 6)
    eq1 = eq.copy()
    eq2 = eq.copy()

    grid1 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.2]), sym=True
    )
    grid2 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.6, 0.8]), sym=True
    )
    grid3 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.2, 0.6]), sym=True
    )
    grid4 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.4, 0.8, 0.9]), sym=True
    )

    objective1 = ForceBalance(eq1, grid=grid1, device_id=0)
    objective2 = ForceBalance(eq1, grid=grid2, device_id=1)
    objective3 = ForceBalance(eq2, grid=grid3, device_id=0)
    objective4 = ForceBalance(eq2, grid=grid4, device_id=0)

    # need to pass MPI communicator to the ObjectiveFunction
    with pytest.raises(ValueError):
        # this one is multi-device, and grids have different sizes
        obj1 = ObjectiveFunction([objective1, objective2])

    # deriv_mode will be set to "blocked" automatically
    with pytest.warns(UserWarning, match="When using multiple devices"):
        obj1 = ObjectiveFunction([objective1, objective2], mpi=MPI)
    # this one is single device, and grids have different sizes
    obj2 = ObjectiveFunction([objective3, objective4])
    obj1.build()
    obj2.build()

    assert obj1._is_mpi
    assert not obj2._is_mpi

    np.testing.assert_allclose(obj1.x(eq1), obj2.x(eq2))

    # multi-device objective must be blocked
    assert obj1._deriv_mode == "blocked"
    assert obj2._deriv_mode == "batched"

    # creating grids like grid3 = [grid1, grid2] doesn't give the same
    # node, spacing and weight ordering, so we can't compare the Jacobians
    # or the objective values directly. Instead, we compare the objective
    # values before and after a single iteration of the solver. This should
    # always decrease the objective value.
    error_init1 = obj1.compute_scalar(obj1.x(eq1))
    error_init2 = obj2.compute_scalar(obj2.x(eq2))

    eq1.solve(objective=obj1, maxiter=1)
    eq2.solve(objective=obj2, maxiter=1)

    error_final1 = obj1.compute_scalar(obj1.x(eq1))
    error_final2 = obj2.compute_scalar(obj2.x(eq2))

    assert error_final1 < error_init1
    assert error_final2 < error_init2
