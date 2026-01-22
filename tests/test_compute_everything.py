"""Test that the computations on this branch agree with those on master."""

import pickle
import warnings

import numpy as np
import pytest

from desc.coils import (
    FourierPlanarCoil,
    FourierRZCoil,
    FourierXYCoil,
    FourierXYZCoil,
    SplineXYZCoil,
)
from desc.compute import data_index
from desc.compute.utils import _grow_seeds
from desc.examples import get
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierRZToroidalSurface,
    FourierXYCurve,
    FourierXYZCurve,
    ZernikeRZToroidalSection,
)
from desc.grid import LinearGrid
from desc.magnetic_fields import (
    CurrentPotentialField,
    FourierCurrentPotentialField,
    OmnigenousField,
)
from desc.utils import ResolutionWarning, errorif, xyz2rpz, xyz2rpz_vec


def _compare_against_master(
    p, data, master_data, error=False, update_master_data=False
):

    for name in data[p]:
        if p in master_data and name in master_data[p]:
            if np.isnan(master_data[p][name]).all():
                mean = 1.0
            else:
                mean = np.nanmean(np.atleast_1d(np.abs(master_data[p][name])))
            try:
                np.testing.assert_allclose(
                    actual=data[p][name],
                    desired=master_data[p][name],
                    atol=1e-8 * mean + 1e-9,  # add 1e-9 for basically-zero things
                    rtol=1e-8,
                    err_msg=f"Parameterization: {p}. Name: {name}.",
                )
            except AssertionError as e:
                error = True
                print(e)
        else:  # update master data with new compute quantity
            update_master_data = True

    return error, update_master_data


def _xyz_to_rpz(data, name):
    if name in ["x", "center"]:
        res = xyz2rpz(data[name])
    else:
        res = xyz2rpz_vec(data[name], phi=data["phi"])
    return res


def _compare_against_rpz(p, data, data_rpz, coordinate_conversion_func):
    for name in data:
        if data_index[p][name]["dim"] != 3:
            continue
        res = coordinate_conversion_func(data, name) - data_rpz[name]
        errorif(
            not np.all(
                (
                    np.isclose(res, 0, rtol=1e-8, atol=1e-8)
                    | np.isclose(np.abs(res[:, 1]), 2 * np.pi, rtol=1e-8, atol=1e-8)[
                        :, np.newaxis
                    ]
                )
                | (~np.isfinite(data_rpz[name]))
            ),
            AssertionError,
            msg=f"Parameterization: {p}. Name: {name}. Residual {res}",
        )


@pytest.mark.unit
def test_compute_everything():
    """Test that the computations on this branch agree with those on master.

    Also make sure we can compute everything without errors.

    Notes
    -----
    This test will fail if the benchmark file has been updated on both
    the local and upstream branches and git cannot resolve the merge
    conflict. In that case, please regenerate the benchmark file.
    Here are instructions for convenience.

    1. Prepend true to the line near the end of this test.
        ``if True or (not error_rpz and update_master_data_rpz):``
    2. Run pytest -k test_compute_everything
    3. Revert 1.
    4. git add tests/inputs/master_compute_data_rpz.pkl

    """
    elliptic_cross_section_with_torsion = {
        "R_lmn": [10, 1, 0.2],
        "Z_lmn": [-2, -0.2],
        "modes_R": [[0, 0], [1, 0], [0, 1]],
        "modes_Z": [[-1, 0], [0, -1]],
    }
    things = {
        # equilibria
        "desc.equilibrium.equilibrium.Equilibrium": get("W7-X"),
        # curves
        "desc.geometry.curve.FourierXYZCurve": FourierXYZCurve(
            X_n=[5, 10, 2], Y_n=[1, 2, 3], Z_n=[-4, -5, -6]
        ),
        "desc.geometry.curve.FourierRZCurve": FourierRZCurve(
            R_n=[10, 1, 0.2], Z_n=[-2, -0.2], modes_R=[0, 1, 2], modes_Z=[-1, -2], NFP=2
        ),
        "desc.geometry.curve.FourierPlanarCurve": FourierPlanarCurve(
            center=[10, 1, 3], normal=[1, 2, 3], r_n=[1, 2, 3], modes=[0, 1, 2]
        ),
        "desc.geometry.curve.FourierXYCurve": FourierXYCurve(
            center=[10, 1, 3], normal=[1, 2, 3], X_n=[0, 2], Y_n=[-3, 1], modes=[-1, 1]
        ),
        "desc.geometry.curve.SplineXYZCurve": FourierXYZCurve(
            X_n=[5, 10, 2], Y_n=[1, 2, 3], Z_n=[-4, -5, -6]
        ).to_SplineXYZ(grid=LinearGrid(N=50)),
        # surfaces
        "desc.geometry.surface.FourierRZToroidalSurface": FourierRZToroidalSurface(
            **elliptic_cross_section_with_torsion
        ),
        "desc.geometry.surface.ZernikeRZToroidalSection": ZernikeRZToroidalSection(
            **elliptic_cross_section_with_torsion
        ),
        # magnetic fields
        "desc.magnetic_fields._current_potential.CurrentPotentialField": CurrentPotentialField(  # noqa:E501
            **elliptic_cross_section_with_torsion,
            potential=lambda theta, zeta, G: G * zeta / 2 / np.pi,
            potential_dtheta=lambda theta, zeta, G: np.zeros_like(theta),
            potential_dzeta=lambda theta, zeta, G: G * np.ones_like(theta) / 2 / np.pi,
            params={"G": 1e7},
        ),
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField": (
            FourierCurrentPotentialField(
                **elliptic_cross_section_with_torsion, I=0, G=1e7
            )
        ),
        "desc.magnetic_fields._core.OmnigenousField": OmnigenousField(
            L_B=0,
            M_B=4,
            L_x=0,
            M_x=1,
            N_x=1,
            NFP=2,
            helicity=(0, 2),
            B_lm=np.array([0.8, 0.9, 1.1, 1.2]),
            x_lmn=np.array([0, -np.pi / 8, 0, np.pi / 8, 0, np.pi / 4]),
        ),
        # coils
        "desc.coils.FourierRZCoil": FourierRZCoil(
            R_n=[10, 1, 0.2], Z_n=[-2, -0.2], modes_R=[0, 1, 2], modes_Z=[-1, -2], NFP=2
        ),
        "desc.coils.FourierXYZCoil": FourierXYZCoil(
            X_n=[5, 10, 2], Y_n=[1, 2, 3], Z_n=[-4, -5, -6]
        ),
        "desc.coils.FourierPlanarCoil": FourierPlanarCoil(
            current=5,
            center=[10, 1, 3],
            normal=[1, 2, 3],
            r_n=[1, 2, 3],
            modes=[0, 1, 2],
        ),
        "desc.coils.FourierXYCoil": FourierXYCoil(
            current=5,
            center=[10, 1, 3],
            normal=[1, 2, 3],
            X_n=[0, 2],
            Y_n=[-3, 1],
            modes=[-1, 1],
        ),
        "desc.coils.SplineXYZCoil": SplineXYZCoil(
            current=5, X=[5, 10, 2, 5], Y=[1, 2, 3, 1], Z=[-4, -5, -6, -4]
        ),
    }
    assert things.keys() == data_index.keys(), (
        f"Missing the parameterization {data_index.keys() - things.keys()}"
        f" to test against master."
    )
    # use this low resolution grid for equilibria to reduce file size
    eqgrid = LinearGrid(
        L=9,
        M=5,
        N=5,
        NFP=things["desc.equilibrium.equilibrium.Equilibrium"].NFP,
        sym=things["desc.equilibrium.equilibrium.Equilibrium"].sym,
        axis=True,
    )
    curvegrid1 = LinearGrid(N=10)
    curvegrid2 = LinearGrid(N=10, NFP=2)
    fieldgrid = LinearGrid(
        L=2,
        M=4,
        N=5,
        NFP=things["desc.magnetic_fields._core.OmnigenousField"].NFP,
        sym=False,
        axis=True,
    )
    grid = {
        "desc.equilibrium.equilibrium.Equilibrium": {"grid": eqgrid},
        "desc.geometry.curve.FourierXYZCurve": {"grid": curvegrid1},
        "desc.geometry.curve.FourierRZCurve": {"grid": curvegrid2},
        "desc.geometry.curve.FourierPlanarCurve": {"grid": curvegrid1},
        "desc.geometry.curve.FourierXYCurve": {"grid": curvegrid1},
        "desc.geometry.curve.SplineXYZCurve": {"grid": curvegrid1},
        "desc.magnetic_fields._core.OmnigenousField": {"grid": fieldgrid},
    }

    with open("tests/inputs/master_compute_data_rpz.pkl", "rb") as file:
        master_data_rpz = pickle.load(file)
    this_branch_data_rpz = {}
    update_master_data_rpz = False
    error_rpz = False

    # some things can't compute "phi" and therefore can't convert to XYZ basis
    no_xyz_things = ["desc.magnetic_fields._core.OmnigenousField"]

    with warnings.catch_warnings():
        # Max resolution of master_compute_data_rpz.pkl limited by GitHub file
        # size cap at 100 mb, so can't hit suggested resolution for some things.
        warnings.filterwarnings("ignore", category=ResolutionWarning)
        warnings.filterwarnings("ignore", category=UserWarning, message="Redl")

        for p in things:

            names = set(data_index[p].keys())

            def need_special(name):
                return bool(data_index[p][name]["source_grid_requirement"]) or bool(
                    data_index[p][name]["grid_requirement"]
                )

            names -= _grow_seeds(p, set(filter(need_special, names)), names)

            this_branch_data_rpz[p] = things[p].compute(
                list(names), **grid.get(p, {}), basis="rpz"
            )
            # make sure we can compute everything
            assert this_branch_data_rpz[p].keys() == names, (
                f"Parameterization: {p}. Can't compute "
                + f"{names - this_branch_data_rpz[p].keys()}."
            )
            # compare data against master branch
            error_rpz, update_master_data_rpz = _compare_against_master(
                p,
                this_branch_data_rpz,
                master_data_rpz,
                error_rpz,
                update_master_data_rpz,
            )

            # test compute in XYZ basis
            if p in no_xyz_things:
                continue
            # remove quantities that are not implemented in the XYZ basis
            # TODO (#1110): generalize this instead of hard-coding for
            #  the quantities "grad(B)" & dependencies
            names_xyz = (
                names - {"grad(B)", "|grad(B)|", "L_grad(B)"}
                if "grad(B)" in names
                else names
            )
            this_branch_data_xyz = things[p].compute(
                list(names_xyz), **grid.get(p, {}), basis="xyz"
            )
            assert this_branch_data_xyz.keys() == names_xyz, (
                f"Parameterization: {p}. Can't compute "
                + f"{names_xyz - this_branch_data_xyz.keys()}."
            )
            _compare_against_rpz(
                p, this_branch_data_xyz, this_branch_data_rpz[p], _xyz_to_rpz
            )

    if not error_rpz and update_master_data_rpz:
        # then update the master compute data
        with open("tests/inputs/master_compute_data_rpz.pkl", "wb") as file:
            # remember to git commit this file
            pickle.dump(this_branch_data_rpz, file)

    assert not error_rpz
