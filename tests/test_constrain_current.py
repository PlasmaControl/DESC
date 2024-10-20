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
        names = []
        dr = 0
        while (
            profile_name + "_" * bool(dr) + "r" * dr
            in data_index["desc.equilibrium.equilibrium.Equilibrium"]
        ):
            names += [profile_name + "_" * bool(dr) + "r" * dr]
            dr += 1
        data = compute_fun(
            parameterization="desc.equilibrium.equilibrium.Equilibrium",
            names=names,
            params=get_params(keys=names, obj=eq, has_axis=grid.axis.size),
            transforms=get_transforms(keys=names, obj=eq, grid=grid),
            profiles=get_profiles(keys=names, obj=eq, grid=grid),
        )
        self.output = {dr: data[name] for dr, name in enumerate(names)}

    def compute(self, grid, params, dr, *args, **kwargs):
        return self.output.get(dr, np.nan)


class TestConstrainCurrent:
    """Tests for computing iota from fixed current profile and vice versa.

    The equations referenced in the descriptions below refer to the document
    attached to the description of GitHub pull request #556.
    """

    @pytest.mark.unit
    def test_iota_to_current_and_back(self):
        """Test we can recover the rotational transform in simple cases.

        Given ``iota``, compute ``I(iota)`` as defined in equation 6.
        Use that ``I(iota)`` to compute ``iota(I)`` as defined in equation 18.
        Check that this output matches the input.

        Cross-validates the computation of the rotational transform by feeding
        the resulting enclosed net toroidal current of an equilibrium with a
        known rotational transform to a method which computes the rotational
        transform as a function of enclosed net toroidal current.

        This tests that rotational transform computations from known toroidal
        current profiles are correct, among other things. For example, this test
        will fail if the compute functions for iota given a current profile are
        incorrect.
        """
        names = ["iota", "iota_r"]
        while names[-1] + "r" in data_index["desc.equilibrium.equilibrium.Equilibrium"]:
            names += [names[-1] + "r"]

        def test(eq, grid_type):
            kwargs = {"L": eq.L_grid, "M": eq.M_grid, "N": eq.N_grid, "NFP": eq.NFP}
            if grid_type != QuadratureGrid:
                kwargs["sym"] = eq.sym
            if grid_type == LinearGrid:
                kwargs["axis"] = True
            grid = grid_type(**kwargs)

            params = get_params(keys=names, obj=eq, has_axis=grid.axis.size)
            transforms = get_transforms(keys=names, obj=eq, grid=grid)
            profiles = get_profiles(keys=names, obj=eq, grid=grid)
            # Compute rotational transform directly from the power series which
            # defines it.
            desired = {
                name: profiles[names[0]].compute(grid, params=params["i_l"], dr=dr)
                for dr, name in enumerate(names)
            }
            # Compute rotational transform using the below toroidal current,
            # which is monkey patched to return equation 6 in amps.
            monkey_patched_profile = _ExactValueProfile(
                eq=eq, grid=grid, profile_name="current"
            )
            actual = compute_fun(
                parameterization="desc.equilibrium.equilibrium.Equilibrium",
                names=names,
                params=dict(params, c_l=None, i_l=None),
                transforms=transforms,
                profiles=dict(profiles, current=monkey_patched_profile, iota=None),
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
                # don't try to validate if we can't compute axis limit yet
                validate_axis = (
                    index < len(names) - 2
                ) and name not in not_implemented_limits
                mask = (validate_axis & at_axis) | ~at_axis
                np.testing.assert_allclose(
                    actual=actual[name][mask],
                    desired=desired[name][mask],
                    equal_nan=False,
                    err_msg=name,
                )

        eq = desc.examples.get("DSHAPE")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(3, 3, 0, 6, 6, 0)
        # Only makes sense to test on configurations with fixed iota profiles.
        test(eq, QuadratureGrid)
        test(eq, ConcentricGrid)
        test(eq, LinearGrid)

    @pytest.mark.unit
    def test_current_to_iota_and_back(self):
        """Test we can recover the enclosed net toroidal current in simple cases.

        Given ``I``, compute ``iota(I)`` as defined in equation 18.
        Use that ``iota(I)`` to compute ``I(iota)`` as defined in equation 6.
        Check that this output matches the input.

        Cross-validates the computation of the enclosed net toroidal current by
        feeding the resulting rotational transform of an equilibrium with a
        known toroidal current profile to a method which computes the enclosed
        net toroidal current as a function of rotational transform.

        This tests that toroidal current computations from known rotational
        transform profiles are correct, among other things. For example, this
        test will fail if the compute functions for current as a function of
        iota are incorrect.
        """
        names = ["current", "current_r"]
        while names[-1] + "r" in data_index["desc.equilibrium.equilibrium.Equilibrium"]:
            names += [names[-1] + "r"]

        def test(eq, grid_type):
            kwargs = {"L": eq.L_grid, "M": eq.M_grid, "N": eq.N_grid, "NFP": eq.NFP}
            if grid_type != QuadratureGrid:
                kwargs["sym"] = eq.sym
            if grid_type == LinearGrid:
                kwargs["axis"] = True
            grid = grid_type(**kwargs)

            params = get_params(keys=names, obj=eq, has_axis=grid.axis.size)
            transforms = get_transforms(keys=names, obj=eq, grid=grid)
            profiles = get_profiles(keys=names, obj=eq, grid=grid)
            # Compute toroidal current directly from the power series which
            # defines it. Recall that vacuum objective sets a current profile.
            desired = {
                name: profiles[names[0]].compute(grid, params=params["c_l"], dr=dr)
                for dr, name in enumerate(names)
            }
            # Compute toroidal current using the below rotational transform,
            # which is monkey patched to return equation 18.
            monkey_patched_profile = _ExactValueProfile(
                eq=eq, grid=grid, profile_name="iota"
            )
            actual = compute_fun(
                parameterization="desc.equilibrium.equilibrium.Equilibrium",
                names=names,
                params=dict(params, c_l=None, i_l=None),
                transforms=transforms,
                profiles=dict(profiles, current=None, iota=monkey_patched_profile),
            )

            at_axis = grid.nodes[:, 0] == 0
            for name in names:
                # don't try to validate if we can't compute axis limit yet
                validate_axis = name not in not_implemented_limits
                mask = (validate_axis & at_axis) | ~at_axis
                np.testing.assert_allclose(
                    actual=actual[name][mask],
                    desired=desired[name][mask],
                    atol=1e-7,
                    equal_nan=False,
                    err_msg=name,
                )

        # Only makes sense to test on configurations with fixed current profiles.
        eq = desc.examples.get("ESTELL")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(3, 3, 3, 6, 6, 6)
        test(eq, QuadratureGrid)
        test(eq, ConcentricGrid)
        test(eq, LinearGrid)
