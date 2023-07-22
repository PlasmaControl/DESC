"""Tests for computing iota from fixed current profile and vice versa."""

import numpy as np
import pytest
from tests.test_axis_limits import not_implemented_limits

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

    @pytest.mark.regression
    @pytest.mark.solve
    def test_compute_rotational_transform(self, DSHAPE, HELIOTRON_vac):
        """Validate the computation of rotational transform from net toroidal current.

        Cross-validates the computation of the rotational transform by feeding
        the resulting net enclosed toroidal current of an equilibrium with a
        known rotational transform to a method which computes the rotational
        transform as a function of net enclosed toroidal current.

        This tests that rotational transform computations from known toroidal
        current profiles are correct, among other things. For example, this test
        will fail if the compute functions for iota given a current profile are
        incorrect.
        """
        names = ["iota", "iota_r"]
        while names[-1] + "r" in data_index:
            names += [names[-1] + "r"]

        def test(stellarator, grid_type):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            kwargs = {"L": eq.L_grid, "M": eq.M_grid, "N": eq.N_grid, "NFP": eq.NFP}
            if grid_type != QuadratureGrid:
                kwargs["sym"] = eq.sym
            if grid_type == LinearGrid:
                kwargs["axis"] = True
            grid = grid_type(**kwargs)

            params = get_params(keys=names, eq=eq, has_axis=grid.axis.size)
            transforms = get_transforms(keys=names, eq=eq, grid=grid)
            profiles = get_profiles(keys=names, eq=eq, grid=grid)
            # Compute rotational transform using the equilibrium's default
            # profile. For this test, this should be directly from the power
            # series which defines the given rotational transform.
            benchmark_data = compute_fun(
                names=names, params=params, transforms=transforms, profiles=profiles
            )
            # Compute rotational transform from the below current profile, which
            # is monkey patched to return a surface average of B_theta in amps
            # as the input net toroidal current.
            monkey_patched_profile = _ExactValueProfile(
                eq=eq, grid=grid, profile_name="current"
            )
            data = compute_fun(
                names=names,
                params=dict(params, i_l=None, c_l=None),
                transforms=transforms,
                profiles=dict(profiles, iota=None, current=monkey_patched_profile),
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
            at_axis = grid.nodes[:, 0] == 0
            for index, name in enumerate(names):
                validate_axis = (
                    (index < len(names) - 2)
                    # don't try to validate if we can't compute it yet
                    and name not in not_implemented_limits.get("all", {})
                    and name not in not_implemented_limits.get("fix_current", {})
                )
                mask = (validate_axis & at_axis) | ~at_axis
                np.testing.assert_allclose(
                    actual=data[name][mask],
                    desired=benchmark_data[name][mask],
                    equal_nan=False,
                    err_msg=name,
                )

        # Only makes sense to test on configurations with fixed iota profiles.
        test(DSHAPE, QuadratureGrid)
        test(DSHAPE, ConcentricGrid)
        test(DSHAPE, LinearGrid)
        test(HELIOTRON_vac, QuadratureGrid)
        test(HELIOTRON_vac, ConcentricGrid)
        test(HELIOTRON_vac, LinearGrid)

    @pytest.mark.regression
    @pytest.mark.solve
    def test_compute_toroidal_current(self, DSHAPE_current, HELIOTRON_vac2):
        """Validate the computation of net toroidal current from rotational transform.

        Cross-validates the computation of the net enclosed toroidal current by
        feeding the resulting rotational transform of an equilibrium with a
        known toroidal current profile to a method which computes the net
        enclosed toroidal current as a function of rotational transform.

        This tests that toroidal current computations from known rotational
        transform profiles are correct, among other things. For example, this
        test will fail if the compute functions for current given an iota
        profile are incorrect.
        """
        names = ["current", "current_r"]
        while names[-1] + "r" in data_index:
            names += [names[-1] + "r"]

        def test(stellarator, grid_type):
            eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
            kwargs = {"L": eq.L_grid, "M": eq.M_grid, "N": eq.N_grid, "NFP": eq.NFP}
            if grid_type != QuadratureGrid:
                kwargs["sym"] = eq.sym
            if grid_type == LinearGrid:
                kwargs["axis"] = True
            grid = grid_type(**kwargs)

            params = get_params(keys=names, eq=eq, has_axis=grid.axis.size)
            transforms = get_transforms(keys=names, eq=eq, grid=grid)
            profiles = get_profiles(keys=names, eq=eq, grid=grid)
            # Compute toroidal current using the equilibrium's default
            # profile. For this test, this should be directly from the power
            # series which defines the given toroidal current.
            benchmark_data = compute_fun(
                names=names, params=params, transforms=transforms, profiles=profiles
            )
            # Compute toroidal current from the below rotational transform profile.
            monkey_patched_profile = _ExactValueProfile(
                eq=eq, grid=grid, profile_name="iota"
            )
            data = compute_fun(
                names=names,
                params=dict(params, i_l=None, c_l=None),
                transforms=transforms,
                profiles=dict(profiles, iota=monkey_patched_profile, current=None),
            )

            at_axis = grid.nodes[:, 0] == 0
            for name in names:
                validate_axis = (
                    # don't try to validate if we can't compute it yet
                    name not in not_implemented_limits.get("all", {})
                    and name not in not_implemented_limits.get("fix_iota", {})
                )
                mask = (validate_axis & at_axis) | ~at_axis
                np.testing.assert_allclose(
                    actual=data[name][mask],
                    desired=benchmark_data[name][mask],
                    equal_nan=False,
                    err_msg=name,
                )

        # Only makes sense to test on configurations with fixed current profiles.
        test(DSHAPE_current, QuadratureGrid)
        test(DSHAPE_current, ConcentricGrid)
        test(DSHAPE_current, LinearGrid)
        test(HELIOTRON_vac2, QuadratureGrid)
        test(HELIOTRON_vac2, ConcentricGrid)
        test(HELIOTRON_vac2, LinearGrid)
