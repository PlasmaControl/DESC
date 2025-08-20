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

from desc import config as desc_config
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction


@pytest.mark.mpi_setup
def test_set_cpu_count():
    """Test that _set_cpu_count."""
    # we already called the function, just check the desc_config
    assert desc_config["num_device"] == num_device
    assert len(desc_config["devices"]) == num_device


@pytest.mark.mpi_setup
def test_multidevice_objective():
    """Test that objective function have proper attributes."""
    eq = get("HELIOTRON")
    with pytest.warns(UserWarning, match="Reducing radial (L) resolution"):
        eq.change_resolution(6, 6, 3, 12, 12, 6)
    eq1 = eq.copy()
    eq2 = eq.copy()

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6, 0.8], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2, 0.6], sym=True)
    grid4 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.4, 0.8, 0.9], sym=True)

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
        obj1.build()
    # this one is single device, and grids have different sizes
    obj2 = ObjectiveFunction([objective3, objective4])
    obj2.build()

    assert obj1._is_mpi
    assert not obj2._is_mpi

    np.testing.assert_allclose(obj1.x(eq1), obj2.x(eq2))

    # multi-device objective must be blocked
    assert obj1._deriv_mode == "blocked"
    assert obj2._deriv_mode == "batched"


@pytest.mark.mpi_run
def test_multidevice_jac():
    """Test that the Jacobian is the same for a single and multi device."""
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        print(f"====== TOTAL OF {size} RANKS ======")

    eq = get("HELIOTRON")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(6, 6, 3, 12, 12, 6)

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6, 0.8], sym=True)

    objective1 = ForceBalance(eq, grid=grid1, device_id=0)
    objective2 = ForceBalance(eq, grid=grid2, device_id=1)

    # deriv_mode will be set to "blocked" automatically
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = ObjectiveFunction([objective1, objective2], mpi=MPI)
        obj.build()

    # creating grids like grid3 = [grid1, grid2] doesn't give the same
    # node, spacing and weight ordering, so we can't compare the Jacobians
    # or the objective values directly. Instead, we compare the objective
    # values before and after a single iteration of the solver. This should
    # always decrease the objective value.
    with obj:
        if rank == 0:
            f0 = obj.compute_scalar(obj.x(eq))
            eq.solve(objective=obj, maxiter=2, verbose=3)
            f1 = obj.compute_scalar(obj.x(eq))

            assert f1 < f0
