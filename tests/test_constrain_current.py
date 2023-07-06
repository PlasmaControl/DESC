"""Tests for computing iota from fixed current profile and vice versa."""

import numpy as np
import pytest

import desc.io
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid


class _ExactValueProfile:
    """Monkey patches the compute method of desc.Profile for testing."""

    def __init__(self, eq, grid):
        self.eq = eq
        self.grid = grid
        self.transforms = get_transforms("current_rr", eq=eq, grid=grid)
        self.profiles = get_profiles("current_rr", eq=eq, grid=grid)
        self.params = get_params("current_rr", eq=eq, has_axis=grid.axis.size)

    def compute(self, params, dr, *args, **kwargs):
        if dr == 0:
            # returns the surface average of B_theta in amperes
            return compute_fun(
                "current",
                params=self.params,
                transforms=self.transforms,
                profiles=self.profiles,
            )["current"]
        if dr == 1:
            # returns the surface average of B_theta_r in amperes
            return compute_fun(
                "current_r",
                params=self.params,
                transforms=self.transforms,
                profiles=self.profiles,
            )["current_r"]
        if dr == 2:
            # returns the surface average of B_theta_rr in amperes
            return compute_fun(
                "current_rr",
                params=self.params,
                transforms=self.transforms,
                profiles=self.profiles,
            )["current_rr"]


class TestConstrainCurrent:
    """Tests for running DESC with a fixed current profile."""

    @pytest.mark.unit
    @pytest.mark.solve
    def test_compute_rotational_transform(self, DSHAPE, HELIOTRON_vac):
        """Test that compute functions recover iota and radial derivatives.

        When the current is fixed to the current computed on an equilibrium
        solved with iota fixed.

        This tests that rotational transform computations from fixed-current profiles
        are correct, among other things. For example, this test will fail if
        the compute functions for iota from a current profile are incorrect.
        """

        def test(stellarator, grid_type):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            if grid_type == "Quadrature":
                grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            elif grid_type == "Concentric":
                grid = ConcentricGrid(
                    L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
                )
            else:
                # Todo: set axis to true when every quantity's limit at the axis
                #  can be computed.
                grid = LinearGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                )

            params = {
                "R_lmn": eq.R_lmn,
                "Z_lmn": eq.Z_lmn,
                "L_lmn": eq.L_lmn,
                "i_l": None,
                "c_l": None,
                "Psi": eq.Psi,
            }
            transforms = get_transforms("iota_rr", eq=eq, grid=grid)
            profiles = {"iota": None, "current": _ExactValueProfile(eq, grid)}
            # compute rotational transform from the above current profile, which
            # is monkey patched to return a surface average of B_theta in amps
            data = compute_fun(
                ["iota", "iota_r", "iota_rr"],
                params=params,
                transforms=transforms,
                profiles=profiles,
            )
            # compute rotational transform using the equilibrium's default
            # profile (directly from the power series which defines iota
            # if the equilibrium fixes iota)
            benchmark_data = compute_fun(
                ["iota", "iota_r", "iota_rr"],
                params=get_params("iota_rr", eq=eq, has_axis=grid.axis.size),
                transforms=transforms,
                profiles=get_profiles("iota_rr", eq=eq, grid=grid),
            )
            np.testing.assert_allclose(data["iota"], benchmark_data["iota"])
            np.testing.assert_allclose(data["iota_r"], benchmark_data["iota_r"])
            np.testing.assert_allclose(data["iota_rr"], benchmark_data["iota_rr"])

        for e in ("Quadrature", "Concentric", "Linear"):
            # works with any stellarators in desc/examples with fixed iota profiles
            test(DSHAPE, e)
            test(HELIOTRON_vac, e)
