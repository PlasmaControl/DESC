"""Tests for computing iota from fixed current profile and vice versa."""

import numpy as np
import pytest

import desc.io
from desc.compute import compute as compute_fun
from desc.compute import data_index, get_params, get_profiles, get_transforms
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid


class _ExactValueProfile:
    """Monkey patches the compute method of desc.Profile for testing."""

    def __init__(self, eq, grid, profile_name):
        self.profile_name = profile_name
        keys = []
        i = 0
        while self._get_name(i) in data_index:
            keys += [self._get_name(i)]
            i += 1
        self.max_dr = i - 1
        self.params = get_params(keys=keys, eq=eq, has_axis=grid.axis.size)
        self.transforms = get_transforms(keys=keys, eq=eq, grid=grid)
        self.profiles = get_profiles(keys=keys, eq=eq, grid=grid)

    def _get_name(self, dr):
        return self.profile_name + "_" * bool(dr) + "r" * dr

    def compute(self, params, dr, *args, **kwargs):
        if dr > self.max_dr:
            return np.nan
        name = self._get_name(dr)
        return compute_fun(
            names=name,
            params=self.params,
            transforms=self.transforms,
            profiles=self.profiles,
        )[name]


class TestConstrainCurrent:
    """Tests for computing iota from fixed current profile and vice versa."""

    # Todo: add test_compute_toroidal_current

    @pytest.mark.unit
    @pytest.mark.solve
    def test_compute_rotational_transform(self, DSHAPE, HELIOTRON_vac):
        """Validate the computation of rotational transform from net toroidal current.

        Cross-validates the computation of the rotational transform by feeding
        the resulting net toroidal current of an equilibrium with a known
        rotational transform to a method which computes the rotational transform
        as a function of net toroidal current.

        This tests that rotational transform computations from known toroidal
        current profiles are correct, among other things. For example, this test
        will fail if the compute functions for iota given a current profile are
        incorrect.
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

            params = get_params(keys=iotas, eq=eq, has_axis=grid.axis.size)
            transforms = get_transforms(keys=iotas, eq=eq, grid=grid)
            profiles = get_profiles(keys=iotas, eq=eq, grid=grid)
            monkey_patched_current = _ExactValueProfile(
                eq, grid, profile_name="current"
            )
            # Compute rotational transform from the above current profile, which
            # is monkey patched to return a surface average of B_theta in amps
            # as the input net toroidal current.
            data = compute_fun(
                names=iotas,
                params=dict(params, i_l=None, c_l=None),
                transforms=transforms,
                profiles=dict(profiles, iota=None, current=monkey_patched_current),
            )
            # Compute rotational transform using the equilibrium's default
            # profile. For this test, this should be directly from the power
            # series which defines the given rotational transform.
            benchmark_data = compute_fun(
                names=iotas, params=params, transforms=transforms, profiles=profiles
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

        # Only makes sense to test on configurations with fixed iota profiles.
        test(DSHAPE, QuadratureGrid)
        test(DSHAPE, ConcentricGrid)
        test(DSHAPE, LinearGrid)
        test(HELIOTRON_vac, QuadratureGrid)
        test(HELIOTRON_vac, ConcentricGrid)
        test(HELIOTRON_vac, LinearGrid)
