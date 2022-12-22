"""Tests for computing iota from fixed current profile and vice versa."""

import numpy as np
import pytest

import desc.io
from desc.compute import compute as compute_fun
from desc.compute import get_transforms
from desc.compute.utils import compress
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid


class _ExactValueProfile:
    """Monkey patches the compute method of desc.Profile for testing."""

    def __init__(self, eq, grid):
        self.eq = eq
        self.grid = grid

    def compute(self, params, dr, *args, **kwargs):
        if dr == 0:
            return self.eq.compute("current", grid=self.grid)["current"]
        if dr == 1:
            return self.eq.compute("current_r", grid=self.grid)["current_r"]


class TestConstrainCurrent:
    """Tests for running DESC with a fixed current profile."""

    @pytest.mark.unit
    @pytest.mark.solve
    def test_compute_rotational_transform(self, DSHAPE_current, HELIOTRON_vac):
        """Test that compute_rotational_transform recovers iota and iota_r.

        When the current is fixed to the current computed on an equilibrium
        solved with iota fixed.

        This tests that compute_rotational_transform is correct, among other things.
        """

        def test(stellarator, grid_type):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            if grid_type == "quadrature":
                grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            else:
                f = ConcentricGrid if grid_type == "concentric" else LinearGrid
                grid = f(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

            transforms = get_transforms("iota_r", eq=eq, grid=grid)
            profiles = {"iota": None, "current": _ExactValueProfile(eq, grid)}
            params = {
                "R_lmn": eq.R_lmn,
                "Z_lmn": eq.Z_lmn,
                "L_lmn": eq.L_lmn,
                "i_l": None,
                "c_l": None,
                "Psi": eq.Psi,
            }
            data = compute_fun(
                ["iota", "iota_r"],
                params=params,
                transforms=transforms,
                profiles=profiles,
            )
            benchmark_data = eq.compute("iota_r", grid=grid)

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

        for e in ("quadrature", "concentric", "linear"):
            # works with any stellarators in desc/examples
            test(DSHAPE_current, e)
            test(HELIOTRON_vac, e)
