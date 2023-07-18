"""Tests for computing iota from fixed current profile and vice versa."""

import numpy as np
import pytest

import desc.io
from desc.compute import compute as compute_fun, data_index
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid


class _ExactValueProfile:
    """Monkey patches the compute method of desc.Profile for testing."""

    def __init__(self, eq, grid, profile_name="current"):
        self.profile_name = profile_name
        keys = []
        i = 0
        while self._get_name(i) in data_index:
            keys += [self._get_name(i)]
            i += 1
        self.params = get_params(keys=keys, eq=eq, has_axis=grid.axis.size)
        self.transforms = get_transforms(keys=keys, eq=eq, grid=grid)
        self.profiles = get_profiles(keys=keys, eq=eq, grid=grid)

    def _get_name(self, dr):
        return self.profile_name + "_" * bool(dr) + "r" * dr

    def compute(self, params, dr, *args, **kwargs):
        name = self._get_name(dr)
        if name in data_index:
            return compute_fun(
                names=name,
                params=self.params,
                transforms=self.transforms,
                profiles=self.profiles,
            )[name]
        return np.nan


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

        iotas = ["iota", "iota_r"]
        while iotas[-1] + "r" in data_index:
            iotas += [iotas[-1] + "r"]

        def test(stellarator, grid_type):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            kwargs = {"L": eq.L_grid, "M": eq.M_grid, "N": eq.N_grid, "NFP": eq.NFP}
            if grid_type != QuadratureGrid:
                kwargs["sym"] = eq.sym
            if grid_type == LinearGrid:
                kwargs["axis"] = True
            grid = grid_type(**kwargs)

            monkey_patched_current = _ExactValueProfile(eq, grid)
            params = monkey_patched_current.params.copy()
            params["i_l"] = None
            params["c_l"] = None
            # compute rotational transform from the above current profile, which
            # is monkey patched to return a surface average of B_theta in amps
            data = compute_fun(
                names=iotas,
                params=params,
                transforms=monkey_patched_current.transforms,
                profiles={"iota": None, "current": monkey_patched_current},
            )
            # compute rotational transform using the equilibrium's default
            # profile (directly from the power series which defines iota
            # if the equilibrium fixes iota)
            benchmark_data = compute_fun(
                names=iotas,
                params=monkey_patched_current.params,
                transforms=monkey_patched_current.transforms,
                profiles=monkey_patched_current.profiles,
            )
            # Evaluating the order n derivative of the rotational transform at
            # the magnetic axis using a given toroidal current profile requires
            # the order n + 2 derivative of that toroidal current profile.
            # Computing the order n derivative of the resulting toroidal current
            # as a function of rotational transform requires the order n
            # derivative of the rotational transform.
            # Therefore, this test cannot be used to cross-validate the magnetic
            # axis limit of the two highest order derivatives of the rotational
            # transform in data_index.
            for iota in iotas:
                skip_axis = grid.axis.size and iota in iotas[-2:]
                np.testing.assert_allclose(
                    actual=grid.compress(data[iota])[skip_axis:],
                    desired=grid.compress(benchmark_data[iota])[skip_axis:],
                    equal_nan=False,
                    err_msg=iota,
                )

        # works with any stellarators in desc/examples with fixed iota profiles
        test(DSHAPE, QuadratureGrid)
        test(DSHAPE, ConcentricGrid)
        test(DSHAPE, LinearGrid)
        test(HELIOTRON_vac, QuadratureGrid)
        test(HELIOTRON_vac, ConcentricGrid)
        test(HELIOTRON_vac, LinearGrid)
