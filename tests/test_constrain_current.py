import numpy as np

import desc.io
from desc.compute import data_index
from desc.compute._core import compute_rotational_transform
from desc.compute.utils import compress
from desc.grid import QuadratureGrid, ConcentricGrid, LinearGrid
from desc.transform import Transform


class _ExactValueProfile:
    """Monkey patches the compute method of desc.Profile for testing."""

    def __init__(self, eq):
        self.eq = eq

    def compute(self, params=None, grid=None, dr=0, *args):
        # unused arguments required to override profile.compute
        if dr == 0:
            return self.eq.compute("current", grid)["current"]
        if dr == 1:
            return self.eq.compute("current_r", grid)["current_r"]


class TestConstrainCurrent:
    """Tests for running DESC with a fixed current profile."""

    def test_compute_rotational_transform(self, DSHAPE, HELIOTRON):
        """
        Test that compute_rotational_transform recovers iota and iota_r
        when the current is fixed to the current computed on an equilibrium
        solved with iota fixed.
        """

        def test(stellarator, grid_type):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            if grid_type == "quadrature":
                grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            else:
                f = ConcentricGrid if grid_type == "concentric" else LinearGrid
                grid = f(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

            R_transform = Transform(
                grid, eq.R_basis, derivs=data_index["iota_r"]["R_derivs"], build=True
            )
            Z_transform = Transform(
                grid, eq.Z_basis, derivs=data_index["iota_r"]["R_derivs"], build=True
            )
            L_transform = Transform(
                grid, eq.L_basis, derivs=data_index["iota_r"]["L_derivs"], build=True
            )
            data = compute_rotational_transform(
                eq.R_lmn,
                eq.Z_lmn,
                eq.L_lmn,
                i_l=None,
                c_l=None,
                Psi=eq.Psi,
                R_transform=R_transform,
                Z_transform=Z_transform,
                L_transform=L_transform,
                iota=None,
                current=_ExactValueProfile(eq),
            )
            benchmark_data = eq.compute("iota_r", grid)

            if grid_type == "linear":
                # ignore axis
                np.testing.assert_allclose(
                    compress(grid, data["iota"])[1:],
                    compress(grid, benchmark_data["iota"])[1:],
                )
                np.testing.assert_allclose(
                    compress(grid, data["iota_r"])[1:],
                    compress(grid, benchmark_data["iota_r"])[1:],
                )
            else:
                np.testing.assert_allclose(data["iota"], benchmark_data["iota"])
                np.testing.assert_allclose(data["iota_r"], benchmark_data["iota_r"])

        for grid_type in ("quadrature", "concentric", "linear"):
            test(DSHAPE, grid_type)
            test(HELIOTRON, grid_type)
