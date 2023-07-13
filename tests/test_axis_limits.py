"""Tests for compute functions evaluated at limits."""
import inspect
import re

import numpy as np
import pytest

import desc.compute
from desc.compute import data_index
from desc.compute.utils import surface_integrals_map
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid

not_finite_limit_keys = {
    "D_Mercier",
    "D_current",
    "D_geodesic",
    "D_shear",  # may not exist for all configurations
    "D_well",
    "J^theta",
    "e^helical",
    "e^theta",
    "e^theta_r",
    "e^theta_t",
    "e^theta_z",
    "g^rt",
    "g^rt_r",
    "g^rt_t",
    "g^rt_z",
    "g^tt",
    "g^tt_r",
    "g^tt_t",
    "g^tt_z",
    "g^tz",
    "g^tz_r",
    "g^tz_t",
    "g^tz_z",
    "grad(alpha)",
    "|e^helical|",
    "|grad(theta)|",
    "<J*B> Redl",  # may not exist for all configurations
}


def continuity(
    eq,
    names,
    delta=1e-5,
    rtol=1e-4,
    atol=1e-5,
    desired_at_axis=None,
):
    """
    Test that the rho=0 axis limits of names are continuous extensions.

    Parameters
    ----------
    eq : Equilibrium
        The equilibrium object used to compute.
    names : iterable, str
        A list of names of the quantities to test for continuity.
    delta: float, optional
        Spacing between grid points.
        Smaller values accumulate finite precision error.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    desired_at_axis : float, optional
        If not provided, the values are extrapolated with a polynomial fit.

    """
    # TODO: remove when boozer transform works with multiple surfaces
    if isinstance(names, str):
        names = [names]
    names = [x for x in names if not ("Boozer" in x or "_mn" in x or x == "B modes")]

    rho = np.linspace(0, 1, 10) * delta
    grid = LinearGrid(rho=rho, M=8, N=8, NFP=eq.NFP, sym=eq.sym)
    assert grid.axis.size

    data = eq.compute(names, grid=grid)
    integrate = surface_integrals_map(grid, expand_out=False)

    should_compute_fit = desired_at_axis is None
    for name in names:
        if data_index[name]["coordinates"] == "":
            # useless to check if global scalar is continuous
            continue
        quantity = (
            grid.compress(data[name])
            if data_index[name]["coordinates"] == "r"
            else integrate(data[name])
        )
        if should_compute_fit:
            if np.ndim(data_index[name]["dim"]):
                # can't polyfit tensor arrays like grad(B)
                continue
            else:
                # fit the data to a polynomial to extrapolate to axis
                poly = np.polyfit(rho[1:], quantity[1:], 6)
                # constant term is the same as evaluating polynomial at rho=0
                desired_at_axis = poly[-1]

        np.testing.assert_allclose(
            quantity[0], desired_at_axis, atol=atol, rtol=rtol, err_msg=name
        )


class TestAxisLimits:
    """Tests for compute functions evaluated at limits."""

    # includes a fixed iota and fixed current equilibrium
    eqs = (get("W7-X"), get("QAS"))

    @pytest.mark.unit
    def test_data_index_deps(self):
        """Ensure developers do not add extra (or forget needed) dependencies."""

        def get_vars(fun, pattern):
            src = inspect.getsource(fun)
            # remove comments
            src = "\n".join(line.partition("#")[0] for line in src.splitlines())
            variables = re.findall(pattern, src)
            variables = set(s.replace("'", "").replace('"', "") for s in variables)
            return variables

        queried_deps = {}
        for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
            if module_name[0] == "_":
                for _, fun in inspect.getmembers(module, inspect.isfunction):
                    # quantities that this function computes
                    keys = get_vars(fun, r"(?<!_)data\[(.*?)\] =")
                    # dependencies queried in source code of this function
                    deps = get_vars(fun, r"(?<!_)data\[(.*?)\]") - keys
                    for key in keys:
                        queried_deps[key] = deps

        for key, val in data_index.items():
            deps = val["dependencies"]
            data_deps = set(deps["data"])
            assert len(data_deps) == len(deps["data"]), key
            axis_limit_deps = set(deps["axis_limit_data"])
            assert len(axis_limit_deps) == len(deps["axis_limit_data"]), key
            assert data_deps.isdisjoint(axis_limit_deps), key
            assert queried_deps[key] == data_deps | axis_limit_deps, key

    @pytest.mark.unit
    def test_axis_limit_api(self):
        """Test that axis limit dependencies are computed only when necessary."""
        eq = Equilibrium()
        grid = LinearGrid(L=2, M=2, N=2, axis=False)
        assert not grid.axis.size
        data = eq.compute("B0", grid=grid)
        assert "B0" in data and "psi_r" in data and "sqrt(g)" in data
        # assert axis limit dependencies are not in data
        assert "psi_rr" not in data and "sqrt(g)_r" not in data
        grid = LinearGrid(L=2, M=2, N=2, axis=True)
        assert grid.axis.size
        data = eq.compute("B0", grid=grid)
        assert "B0" in data and "psi_r" in data and "sqrt(g)" in data
        # assert axis limit dependencies are in data
        assert "psi_rr" in data and "sqrt(g)_r" in data
        assert np.all(np.isfinite(data["B0"]))

    @pytest.mark.unit
    def test_limit_existence(self):
        """Test that only quantities which lack limits do not evaluate at axis."""

        def skip_atomic_profile(eq, name):
            return (
                (eq.atomic_number is None and "Zeff" in name)
                or (eq.electron_temperature is None and "Te" in name)
                or (eq.electron_density is None and "ne" in name)
                or (eq.ion_temperature is None and "Ti" in name)
                or (eq.pressure is not None and "<J*B> Redl" in name)
            )

        for eq in TestAxisLimits.eqs:
            grid = LinearGrid(L=2, M=2, N=2, sym=eq.sym, NFP=eq.NFP, axis=True)
            assert grid.axis.size
            data = eq.compute(list(data_index.keys()), grid=grid)
            is_axis = grid.nodes[:, 0] == 0
            for key in data_index:
                if skip_atomic_profile(eq, key):
                    continue
                is_finite = np.isfinite(data[key])
                if key in not_finite_limit_keys:
                    assert np.all(is_finite.T ^ is_axis), key
                else:
                    assert np.all(is_finite), key

    @pytest.mark.unit
    def test_zero_limits(self):
        """Test limits of basic quantities that should be 0 at magnetic axis."""
        for eq in TestAxisLimits.eqs:
            continuity(
                eq,
                names=["rho", "psi", "psi_r", "e_theta", "sqrt(g)"],
                desired_at_axis=0,
            )

    @pytest.mark.unit
    def test_continuous_limits(self):
        """Heuristic to test correctness of all quantities with limits."""
        finite_discontinuous = {"curvature_k1", "curvature_k2"}
        continuous = data_index.keys() - not_finite_limit_keys - finite_discontinuous
        searching = True
        while searching:
            size = len(finite_discontinuous)
            for key in continuous:
                if not finite_discontinuous.isdisjoint(
                    data_index[key]["full_dependencies"]["data"]
                ):
                    finite_discontinuous.add(key)
            searching = len(finite_discontinuous) > size
        continuous -= finite_discontinuous

        weak_tolerance = {
            "B^zeta_rr": {"rtol": 5e-02},
            "F_rho": {"atol": 1e00},
            "|B|_rr": {"rtol": 5e-02},
            "F": {"atol": 1e00},
            "iota_zero_current_den": {"atol": 1e02},  # fix
            "B_rho_rr": {"rtol": 5e-02},
            "B_zeta_rr": {"rtol": 5e-02},
            "G_rr": {"rtol": 5e-02},
            "iota_zero_current_num": {"atol": 1e01},  # fix
            "J": {"atol": 1e00},
            "B0_rr": {"rtol": 5e-02},
            "B_rr": {"atol": 1e00},
            "(J sqrt(g))_r": {"atol": 1e00},
            "iota_zero_current_num_rr": {"atol": 1e-02},  # fix
            "B^theta_rr": {"rtol": 5e-02},
            "J_R": {"atol": 1e00},
        }
        for eq in TestAxisLimits.eqs:
            continuity(eq, names=continuous - weak_tolerance.keys())
            for key in weak_tolerance:
                continuity(eq, names=key, **weak_tolerance[key])
