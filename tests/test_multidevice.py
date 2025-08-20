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
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer


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
def test_multidevice_eq_solve():
    """Test that eq.solve still reduces force error."""
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    assert size == 2
    assert rank < 2

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
            f0 = obj.compute_scalar(obj.x(eq)).block_until_ready()
            eq.solve(objective=obj, maxiter=2, verbose=3)
            f1 = obj.compute_scalar(obj.x(eq))

            assert f1 < f0


@pytest.mark.mpi_run
def test_multidevice_eq_optimize():
    """Test that eq.optimize still reduces error."""
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    assert size == 2
    assert rank < 2

    eq = get("precise_QA")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(M=3, N=2, M_grid=6, N_grid=4)

    # create two grids with different rho values, this will effectively separate
    # the quasisymmetry objective into two parts
    grid1 = LinearGrid(
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        rho=np.linspace(0.2, 0.5, 2),
        sym=True,
    )
    grid2 = LinearGrid(
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        rho=np.linspace(0.6, 1.0, 3),
        sym=True,
    )

    # when using parallel objectives, the user needs to supply the device_id
    obj1 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid1, device_id=0)
    obj2 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid2, device_id=1)
    obj3 = AspectRatio(eq=eq, target=8, weight=100, device_id=0)
    objs = [obj1, obj2, obj3]

    objective = ObjectiveFunction(
        objs, deriv_mode="blocked", mpi=MPI, rank_per_objective=np.array([0, 1, 0])
    )
    objective.build()

    # we will fix some modes as usual
    k = 1
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    constraints = (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
        FixCurrent(eq=eq),
    )
    optimizer = Optimizer("proximal-lsq-exact")

    with objective as objective:
        if rank == 0:
            f0 = objective.compute_scalar(objective.x(eq))
            eq.optimize(
                objective=objective,
                constraints=constraints,
                optimizer=optimizer,
                maxiter=1,
                verbose=3,
            )
            f1 = objective.compute_scalar(objective.x(eq))
            assert f1 < f0
