"""Tests for the multidevice capabilities."""

import warnings

# This file has to run on a separate process because it changes the number of CPUs
from desc import _set_cpu_count, set_device

num_device = 3
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
    get_fixed_boundary_constraints,
)
from desc.optimize import LinearConstraintProjection, Optimizer, ProximalProjection


@pytest.mark.mpi_setup
def test_set_cpu_count():
    """Test that _set_cpu_count works."""
    # we already called the function, just check the desc_config
    assert desc_config["kind"] == "cpu"
    assert desc_config["num_device"] == num_device
    assert len(desc_config["devices"]) == num_device
    assert len(desc_config["avail_mems"]) == num_device


@pytest.mark.mpi_run
def test_multidevice_objective_attributes():
    """Test that objective attributes are same."""
    eq = get("precise_QH")

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.8], sym=True)

    obj1 = ObjectiveFunction(
        [
            ForceBalance(eq, grid=grid1),
            ForceBalance(eq, grid=grid2),
            ForceBalance(eq, grid=grid3),
        ],
        deriv_mode="blocked",
    )
    obj1.build()

    # deriv_mode will be set to "blocked" automatically
    with pytest.warns(UserWarning, match="When using multiple devices"):
        obj2 = ObjectiveFunction(
            [
                ForceBalance(eq, grid=grid1, device_id=0),
                ForceBalance(eq, grid=grid2, device_id=1),
                ForceBalance(eq, grid=grid3, device_id=2),
            ],
            mpi=MPI,
        )
        obj2.build()

    for obj1i, obj2i in zip(obj1.objectives, obj2.objectives):
        assert obj1i._loss_function == obj2i._loss_function
        np.testing.assert_allclose(obj1i._weight, obj2i._weight)
        np.testing.assert_allclose(obj1i._target, obj2i._target)
        np.testing.assert_allclose(obj1i._normalization, obj2i._normalization)
        np.testing.assert_allclose(obj1i._dim_f, obj2i._dim_f)
        key = "quad_weights"
        np.testing.assert_allclose(
            obj1i._constants[key], obj2i._constants[key], err_msg=key
        )


@pytest.mark.mpi_run
def test_multidevice_compute():
    """Test that objective compute gives same results."""
    rank = MPI.COMM_WORLD.Get_rank()
    eq = get("precise_QH")

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.8], sym=True)

    obj1 = ObjectiveFunction(
        [
            ForceBalance(eq, grid=grid1),
            ForceBalance(eq, grid=grid2),
            ForceBalance(eq, grid=grid3),
        ],
        deriv_mode="blocked",
    )
    obj1.build()

    # deriv_mode will be set to "blocked" automatically
    with pytest.warns(UserWarning, match="When using multiple devices"):
        obj2 = ObjectiveFunction(
            [
                ForceBalance(eq, grid=grid1, device_id=0),
                ForceBalance(eq, grid=grid2, device_id=1),
                ForceBalance(eq, grid=grid3, device_id=2),
            ],
            mpi=MPI,
        )
        obj2.build()

    f1 = obj1.compute_scalar(obj1.x(eq))
    with obj2:
        if rank == 0:
            f2 = obj2.compute_scalar(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = obj1.compute_unscaled(obj1.x(eq))
            f2 = obj2.compute_unscaled(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = obj1.compute_scaled(obj1.x(eq))
            f2 = obj2.compute_scaled(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = obj1.compute_scaled_error(obj1.x(eq))
            f2 = obj2.compute_scaled_error(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)


@pytest.mark.mpi_run
def test_multidevice_derivatives():
    """Test that objective derivatives gives same results."""
    rank = MPI.COMM_WORLD.Get_rank()
    eq = get("precise_QH")

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.8], sym=True)

    obj1 = ObjectiveFunction(
        [
            ForceBalance(eq, grid=grid1),
            ForceBalance(eq, grid=grid2),
            ForceBalance(eq, grid=grid3),
        ],
        deriv_mode="blocked",
    )
    obj1.build()

    # deriv_mode will be set to "blocked" automatically
    with pytest.warns(UserWarning, match="When using multiple devices"):
        obj2 = ObjectiveFunction(
            [
                ForceBalance(eq, grid=grid1, device_id=0),
                ForceBalance(eq, grid=grid2, device_id=1),
                ForceBalance(eq, grid=grid3, device_id=2),
            ],
            mpi=MPI,
        )
        obj2.build()

    with obj2:
        if rank == 0:
            with pytest.raises(NotImplementedError):
                _ = obj2.grad(obj2.x(eq))

            f1 = obj1.jac_unscaled(obj1.x(eq))
            f2 = obj2.jac_unscaled(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = obj1.jac_scaled(obj1.x(eq))
            f2 = obj2.jac_scaled(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = obj1.jac_scaled_error(obj1.x(eq))
            f2 = obj2.jac_scaled_error(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)


@pytest.mark.mpi_run
def test_multidevice_linear_proj_derivatives():
    """Test that linear projection derivatives gives same results."""
    rank = MPI.COMM_WORLD.Get_rank()
    eq = get("precise_QH")

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.8], sym=True)

    objf1 = ObjectiveFunction(
        [
            ForceBalance(eq, grid=grid1),
            ForceBalance(eq, grid=grid2),
            ForceBalance(eq, grid=grid3),
        ],
        deriv_mode="blocked",
    )
    objf1.build()

    # deriv_mode will be set to "blocked" automatically
    with pytest.warns(UserWarning, match="When using multiple devices"):
        objf2 = ObjectiveFunction(
            [
                ForceBalance(eq, grid=grid1, device_id=0),
                ForceBalance(eq, grid=grid2, device_id=1),
                ForceBalance(eq, grid=grid3, device_id=2),
            ],
            mpi=MPI,
        )
        objf2.build()

    cons = get_fixed_boundary_constraints(eq)
    cons = ObjectiveFunction(cons)
    obj1 = LinearConstraintProjection(objective=objf1, constraint=cons)
    obj2 = LinearConstraintProjection(objective=objf2, constraint=cons)
    obj1.build()
    obj2.build()

    with objf2:
        if rank == 0:
            with pytest.raises(NotImplementedError):
                _ = obj2.grad(obj2.x(eq))

            f1 = obj1.jac_unscaled(obj1.x(eq))
            f2 = obj2.jac_unscaled(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = obj1.jac_scaled(obj1.x(eq))
            f2 = obj2.jac_scaled(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = obj1.jac_scaled_error(obj1.x(eq))
            f2 = obj2.jac_scaled_error(obj2.x(eq))
            np.testing.assert_allclose(f2, f1, atol=1e-8)


@pytest.mark.mpi_run
def test_multidevice_proximal_derivatives():
    """Test that proximal derivatives gives same results."""
    rank = MPI.COMM_WORLD.Get_rank()
    eq = get("precise_QH")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(1, 1, 1, 2, 2, 2)

    eq1 = eq.copy()
    eq2 = eq.copy()

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6, 0.8], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.9], sym=True)

    obj1 = QuasisymmetryTwoTerm(eq=eq1, helicity=(1, eq.NFP), grid=grid1)
    obj2 = QuasisymmetryTwoTerm(eq=eq1, helicity=(1, eq.NFP), grid=grid2)
    obj3 = QuasisymmetryTwoTerm(eq=eq1, helicity=(1, eq.NFP), grid=grid3)
    objs = [obj1, obj2, obj3]

    objective1 = ObjectiveFunction(objs, deriv_mode="blocked")
    objective1.build(verbose=0)

    con1 = ObjectiveFunction(ForceBalance(eq1))
    con1.build(verbose=0)

    obj1 = QuasisymmetryTwoTerm(eq=eq2, helicity=(1, eq.NFP), grid=grid1, device_id=0)
    obj2 = QuasisymmetryTwoTerm(eq=eq2, helicity=(1, eq.NFP), grid=grid2, device_id=1)
    obj3 = QuasisymmetryTwoTerm(eq=eq2, helicity=(1, eq.NFP), grid=grid3, device_id=2)
    objs = [obj1, obj2, obj3]

    objective2 = ObjectiveFunction(
        objs, deriv_mode="blocked", mpi=MPI, rank_per_objective=np.array([0, 1, 2])
    )
    objective2.build(verbose=0)
    con2 = ObjectiveFunction(ForceBalance(eq2))
    con2.build(verbose=0)

    perturb_options = {"order": 1}
    solve_options = {"maxiter": 1}
    prox1 = ProximalProjection(
        objective=objective1,
        constraint=con1,
        eq=eq1,
        solve_options=solve_options,
        perturb_options=perturb_options,
    )
    prox2 = ProximalProjection(
        objective=objective2,
        constraint=con2,
        eq=eq2,
        solve_options=solve_options,
        perturb_options=perturb_options,
    )
    prox1.build()
    prox2.build()

    with objective2:
        if rank == 0:
            f1 = prox1.grad(prox1.x(eq1))
            f2 = prox2.grad(prox2.x(eq2))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = prox1.jac_unscaled(prox1.x(eq1))
            f2 = prox2.jac_unscaled(prox2.x(eq2))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = prox1.jac_scaled(prox1.x(eq1))
            f2 = prox2.jac_scaled(prox2.x(eq2))
            np.testing.assert_allclose(f2, f1, atol=1e-8)

            f1 = prox1.jac_scaled_error(prox1.x(eq1))
            f2 = prox2.jac_scaled_error(prox2.x(eq2))
            np.testing.assert_allclose(f2, f1, atol=1e-8)


@pytest.mark.mpi_run
def test_multidevice_objective_build():
    """Test that objective function build works fine."""
    eq = get("HELIOTRON")

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6, 0.8], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2, 0.6], sym=True)
    grid4 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.4, 0.8, 0.9], sym=True)

    objective1 = ForceBalance(eq, grid=grid1, device_id=0)
    objective2 = ForceBalance(eq, grid=grid2, device_id=1)
    objective3 = ForceBalance(eq, grid=grid3, device_id=2)
    objective4 = ForceBalance(eq, grid=grid4, device_id=0)

    # need to pass MPI communicator to the ObjectiveFunction
    with pytest.raises(ValueError, match="MPI communicator"):
        # this one is multi-device
        obj1 = ObjectiveFunction([objective1, objective2, objective3])

    # need to use multiple ranks if using multiple devices
    with pytest.raises(ValueError, match="Requested number of ranks is"):
        # this one is multi-device
        obj1 = ObjectiveFunction(
            [objective1, objective2, objective3], mpi=MPI, rank_per_objective=[0, 0, 0]
        )

    # need to have same device for the same rank objectives
    with pytest.raises(ValueError, match="Same rank objectives should"):
        # this one is multi-device
        obj1 = ObjectiveFunction(
            [objective1, objective2, objective3, objective4],
            mpi=MPI,
            rank_per_objective=[0, 1, 2, 2],
        )

    obj1 = ObjectiveFunction([objective1, objective2, objective3], mpi=MPI)
    # deriv_mode will be set to "blocked" automatically
    with pytest.warns(UserWarning, match="When using multiple devices"):
        obj1.build()

    # this one is single device, and grids have different sizes
    obj2 = ObjectiveFunction([objective1, objective4])
    obj2.build()

    assert obj1._is_mpi
    assert not obj2._is_mpi

    np.testing.assert_allclose(obj1.x(eq), obj2.x(eq))

    # multi-device objective must be blocked
    assert obj1._deriv_mode == "blocked"
    assert obj2._deriv_mode == "batched"


@pytest.mark.mpi_run
def test_multidevice_eq_solve():
    """Test that eq.solve still reduces force error."""
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    assert size == num_device
    assert rank < num_device

    eq = get("HELIOTRON")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(6, 6, 3, 12, 12, 6)

    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6, 0.8], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.9], sym=True)

    obj1 = ForceBalance(eq, grid=grid1, device_id=0)
    obj2 = ForceBalance(eq, grid=grid2, device_id=1)
    obj3 = ForceBalance(eq, grid=grid3, device_id=2)

    # deriv_mode will be set to "blocked" automatically
    with pytest.warns(UserWarning, match="When using multiple devices"):
        obj = ObjectiveFunction([obj1, obj2, obj3], mpi=MPI)
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
    assert size == num_device
    assert rank < num_device

    eq = get("precise_QA")
    eq.change_resolution(M=3, N=2, M_grid=6, N_grid=4)

    # create two grids with different rho values, this will effectively separate
    # the quasisymmetry objective into two parts
    gM = eq.M_grid
    gN = eq.N_grid
    grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
    grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6, 0.8], sym=True)
    grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.9], sym=True)

    # when using parallel objectives, the user needs to supply the device_id
    obj1 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid1, device_id=0)
    obj2 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid2, device_id=1)
    obj3 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid3, device_id=2)
    obj4 = AspectRatio(eq=eq, target=8, weight=100, device_id=0)
    objs = [obj1, obj2, obj3, obj4]

    objective = ObjectiveFunction(
        objs, deriv_mode="blocked", mpi=MPI, rank_per_objective=np.array([0, 1, 2, 0])
    )
    objective.build()

    # we will fix some modes as usual
    k = 1
    sRm = eq.surface.R_basis.modes
    sZm = eq.surface.Z_basis.modes
    R_modes = np.vstack(([0, 0, 0], sRm[np.max(np.abs(sRm), 1) > k, :]))
    Z_modes = sZm[np.max(np.abs(sZm), 1) > k, :]
    constraints = (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
        FixCurrent(eq=eq),
    )
    optimizer = Optimizer("proximal-lsq-exact")

    with objective:
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
