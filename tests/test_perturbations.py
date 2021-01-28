import unittest
import numpy as np

from desc.backend import jnp
from desc.utils import unpack_state
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.compute_funs import compute_toroidal_coords
from desc.equilibrium import Equilibrium
from desc.objective_funs import ObjectiveFunction
from desc.boundary_conditions import BoundaryConstraint


class DummyFunLinear(ObjectiveFunction):
    """A dummy linear objective function."""

    @property
    def name(self):
        return "BC"

    @property
    def scalar(self):
        return False

    @property
    def derivatives(self):
        derivatives = np.array([[0, 0, 0],])
        return derivatives

    def compute(self, y, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):

        if self.BC_constraint is not None:
            x = self.BC_constraint.recover_from_bdry(y, Rb_mn, Zb_mn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes,
        )

        toroidal_coords = compute_toroidal_coords(
            Psi,
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_transform,
            self.i_transform,
            zeta_ratio,
        )

        axis = self.R_transform.grid.axis
        R0_idx = jnp.where((self.Rb_transform.basis.modes == [0, 0, 0]).all(axis=1))[0]

        # f = R0_x / Psi - R0_b
        residual = toroidal_coords["R"][axis] / Psi - Rb_mn[R0_idx]
        return residual * jnp.ones_like(y)

    def compute_scalar(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        pass

    def callback(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0) -> bool:
        pass


class TestPerturbations(unittest.TestCase):
    """Tests for pertubations."""

    def test_perturb_1D(self):
        """Linear test function where perturb order=1 is exact."""

        inputs = {
            "sym": True,
            "NFP": 1,
            "Psi": 1.0,
            "L": 2,
            "M": 2,
            "N": 1,
            "profiles": np.zeros((1, 3)),
            "boundary": np.array([[-1, 0, 0, 2], [0, 0, 3, 0], [1, 0, 1, 0],]),
        }
        eq_old = Equilibrium(inputs=inputs)
        grid = LinearGrid(NFP=eq_old.NFP, rho=0)
        R_transform = Transform(grid, eq_old.R_basis)
        Z_transform = Transform(grid, eq_old.Z_basis)
        L_transform = Transform(grid, eq_old.L_basis)
        Rb_transform = Transform(grid, eq_old.Rb_basis)
        Zb_transform = Transform(grid, eq_old.Zb_basis)
        p_transform = Transform(grid, eq_old.p_basis)
        i_transform = Transform(grid, eq_old.i_basis)
        obj_fun = DummyFunLinear(
            R_transform=R_transform,
            Z_transform=Z_transform,
            L_transform=L_transform,
            Rb_transform=Rb_transform,
            Zb_transform=Zb_transform,
            p_transform=p_transform,
            i_transform=i_transform,
            BC_constraint=BoundaryConstraint(
                eq_old.R_basis,
                eq_old.Z_basis,
                eq_old.L_basis,
                eq_old.Rb_basis,
                eq_old.Zb_basis,
                eq_old.Rb_mn,
                eq_old.Zb_mn,
            ),
        )
        eq_old.objective = obj_fun
        y = eq_old.objective.BC_constraint.project(eq_old.x)
        args = (
            y,
            eq_old.Rb_mn,
            eq_old.Zb_mn,
            eq_old.p_l,
            eq_old.i_l,
            eq_old.Psi,
            eq_old.zeta_ratio,
        )
        res_old = eq_old.objective.compute(*args)

        deltas = {
            "Rb_mn": np.zeros((eq_old.Rb_basis.num_modes,)),
            "Zb_mn": np.zeros((eq_old.Zb_basis.num_modes,)),
            "Psi": np.array([0.2]),
        }
        idx_R = np.where((eq_old.Rb_basis.modes == [0, 2, 1]).all(axis=1))[0]
        idx_Z = np.where((eq_old.Zb_basis.modes == [0, -2, 1]).all(axis=1))[0]
        deltas["Rb_mn"][idx_R] = 0.5
        deltas["Zb_mn"][idx_Z] = -0.3

        eq_new = eq_old.perturb(deltas, order=1)
        y = eq_new.objective.BC_constraint.project(eq_new.x)
        args = (
            y,
            eq_new.Rb_mn,
            eq_new.Zb_mn,
            eq_new.p_l,
            eq_new.i_l,
            eq_new.Psi,
            eq_new.zeta_ratio,
        )

        res_new = eq_new.objective.compute(*args)

        # tolerance could be lower if only testing with JAX
        np.testing.assert_allclose(res_old, 0, atol=1e-6)
        np.testing.assert_allclose(res_new, 0, atol=1e-6)

    def test_perturb_2D(self):
        """Nonlinear test function to check perturb convergence rates."""
        pass
