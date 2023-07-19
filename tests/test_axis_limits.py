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

# Unless mentioned (in the source code of the compute function or elsewhere),
# the only assumptions made to compute the magnetic axis limit of
# quantities are that these functions tend toward zero as the magnetic axis
# is approached, and that the limit of their rho derivatives is not zero.
# Also d^n𝜓/(d𝜌)^n for n > 3 is assumed zero everywhere.
zero_limit_keys = {"rho", "psi", "psi_r", "e_theta", "sqrt(g)"}

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

# Todo: these limits exist, but may currently evaluate as nan.
todo_keys = {
    "iota_num_rrr",  # requires sqrt(g)_rrrr
    "iota_den_rrr",  # requires sqrt(g)_rrrr
    "iota_rr",  # already done, just needs limits of above two.
}


def _skip_profile(eq, name):
    return (
        (eq.atomic_number is None and "Zeff" in name)
        or (eq.electron_temperature is None and "Te" in name)
        or (eq.electron_density is None and "ne" in name)
        or (eq.ion_temperature is None and "Ti" in name)
        or (eq.pressure is not None and "<J*B> Redl" in name)
        or (eq.current is None and ("iota_num" in name or "iota_den" in name))
        or (eq.iota is None and name in todo_keys)
    )


def assert_is_continuous(
    eq,
    names,
    delta=1e-5,
    rtol=1e-4,
    atol=1e-5,
    desired_at_axis=None,
    kwargs=None,
):
    """
    Asserts that the rho=0 axis limits of names are continuous extensions.

    Parameters
    ----------
    eq : Equilibrium
        The equilibrium object used for the computation.
    names : list, str
        A list of names of the quantities to test for continuity.
    delta: float, optional
        Max distance from magnetic axis.
        Smaller values accumulate finite precision error.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    desired_at_axis : float, optional
        If not provided, the values are extrapolated with a polynomial fit.
    kwargs : dict, optional
        Keyword arguments to override the parameters above for specific names.
        The dictionary should have the following structure:
        {
            "name1": {
                "rtol": custom_rtol1,
                "atol": custom_atol1,
                "desired_at_axis": custom_desired_at_axis1
            },
            "name2": {"rtol": custom_rtol2},
            ...
        }

    """
    if kwargs is None:
        kwargs = {}
    if isinstance(names, str):
        names = [names]
    # TODO: remove when boozer transform works with multiple surfaces
    names = [x for x in names if not ("Boozer" in x or "_mn" in x or x == "B modes")]

    rho = np.linspace(0, 1, 10) * delta
    grid = LinearGrid(rho=rho, M=8, N=8, NFP=eq.NFP, sym=eq.sym)
    assert grid.axis.size
    integrate = surface_integrals_map(grid, expand_out=False)
    data = eq.compute(names, grid=grid)

    for name in names:
        if data_index[name]["coordinates"] == "" or _skip_profile(eq, name):
            continue
        # make single variable function of rho
        profile = (
            grid.compress(data[name])
            if data_index[name]["coordinates"] == "r"
            else integrate(data[name])
        )
        fit = kwargs.get(name, {}).get("desired_at_axis", desired_at_axis)
        if fit is None:
            if np.ndim(data_index[name]["dim"]):
                # can't polyfit tensor arrays like grad(B)
                fit = (profile[0] + profile[1]) / 2
            else:
                # fit the data to a polynomial to extrapolate to axis
                poly = np.polyfit(rho[1:], profile[1:], 6)
                # constant term is the same as evaluating polynomial at rho=0
                fit = poly[-1]
        np.testing.assert_allclose(
            actual=profile[0],
            desired=fit,
            rtol=kwargs.get(name, {}).get("rtol", rtol),
            atol=kwargs.get(name, {}).get("atol", atol),
            equal_nan=False,
            err_msg=name,
        )


def get_matches(fun, pattern):
    """Return all matches of ``pattern`` in source code of function ``fun``."""
    src = inspect.getsource(fun)
    # attempt to remove any decorator functions
    # (currently works without this filter, but better to be defensive)
    src = src.partition("def ")[2]
    # attempt to remove comments
    src = "\n".join(line.partition("#")[0] for line in src.splitlines())
    matches = pattern.findall(src)
    matches = set(s.replace("'", "").replace('"', "") for s in matches)
    return matches


class TestAxisLimits:
    """Tests for compute functions evaluated at limits."""

    @pytest.mark.unit
    def test_data_index_deps(self):
        """Ensure developers do not add extra (or forget needed) dependencies."""
        queried_deps = {}
        pattern_keys = re.compile(r"(?<!_)data\[(.*?)\] =")
        pattern_data = re.compile(r"(?<!_)data\[(.*?)\]")
        pattern_profiles = re.compile(r"profiles\[(.*?)\]")
        pattern_params = re.compile(r"params\[(.*?)\]")
        for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
            if module_name[0] == "_":
                for _, fun in inspect.getmembers(module, inspect.isfunction):
                    # quantities that this function computes
                    keys = get_matches(fun, pattern_keys)
                    # dependencies queried in source code of this function
                    deps = {
                        "data": get_matches(fun, pattern_data) - keys,
                        "profiles": get_matches(fun, pattern_profiles),
                        "params": get_matches(fun, pattern_params),
                    }
                    for key in keys:
                        queried_deps[key] = deps

        for key, val in data_index.items():
            deps = val["dependencies"]
            data = set(deps["data"])
            axis_limit_data = set(deps["axis_limit_data"])
            profiles = set(deps["profiles"])
            params = set(deps["params"])
            # assert no duplicate dependencies
            assert len(data) == len(deps["data"]), key
            assert len(axis_limit_data) == len(deps["axis_limit_data"]), key
            assert data.isdisjoint(axis_limit_data), key
            assert len(profiles) == len(deps["profiles"]), key
            assert len(params) == len(deps["params"]), key
            # assert correct dependencies are queried
            assert queried_deps[key]["data"] == data | axis_limit_data, key
            assert queried_deps[key]["profiles"] == profiles, key
            assert queried_deps[key]["params"] == params, key

    @pytest.mark.unit
    def test_axis_limit_api(self):
        """Test that axis limit dependencies are computed only when necessary."""
        name = "B0"
        deps = {"psi_r", "sqrt(g)"}
        axis_limit_deps = {"psi_rr", "sqrt(g)_r"}
        eq = Equilibrium()
        grid = LinearGrid(L=2, M=2, N=2, axis=False)
        assert not grid.axis.size
        data = eq.compute(name, grid=grid).keys()
        assert name in data and deps < data and axis_limit_deps.isdisjoint(data)
        grid = LinearGrid(L=2, M=2, N=2, axis=True)
        assert grid.axis.size
        data = eq.compute(name, grid=grid)
        assert name in data and deps | axis_limit_deps < data.keys()
        assert np.all(np.isfinite(data[name]))

    @pytest.mark.unit
    def test_limit_existence(self):
        """Test that only quantities which lack limits do not evaluate at axis."""

        def test(eq):
            grid = LinearGrid(L=2, M=2, N=2, sym=eq.sym, NFP=eq.NFP, axis=True)
            at_axis = grid.nodes[:, 0] == 0
            assert at_axis.any()
            data = eq.compute(list(data_index.keys()), grid=grid)
            for key in data_index:
                if _skip_profile(eq, key):
                    continue
                is_finite = np.isfinite(data[key])
                if key in not_finite_limit_keys:
                    assert np.all(is_finite.T ^ at_axis), key
                else:
                    assert np.all(is_finite), key

        test(get("W7-X"))  # fixed iota
        test(get("QAS"))  # fixed current

    @pytest.mark.unit
    def test_continuous_limits(self):
        """Heuristic to test correctness of all quantities with limits."""
        # It is possible for a discontinuity to propagate across dependencies,
        # so this test does not make sense for keys that rely on discontinuous
        # keys as dependencies.
        finite_discontinuous = {"curvature_k1", "curvature_k2"}
        # Subtract out not_finite_limit_keys before search to avoid false
        # positives for discontinuous keys. (Recall that nan and inf always
        # propagate up dependencies, so we do not need to search up the
        # dependency trees of keys in that set).
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

        # The need for a weaker tolerance on these keys may be due to large
        # derivatives near axis, finite precision error, or a subpar polynomial
        # regression fit against which the axis limit is compared.
        weaker_tolerance = {
            "B^zeta_rr": {"rtol": 5e-02},
            "F_rho": {"atol": 1e00},
            "|B|_rr": {"rtol": 5e-02},
            "F": {"atol": 1e00},
            "B_rho_rr": {"rtol": 5e-02},
            "B_zeta_rr": {"rtol": 5e-02},
            "G_rr": {"rtol": 5e-02},
            "J": {"atol": 1e00},
            "B0_rr": {"rtol": 5e-02},
            "B_rr": {"atol": 1e00},
            "(J sqrt(g))_r": {"atol": 1e00},
            "B^theta_rr": {"rtol": 5e-02},
            "J_R": {"atol": 1e00},
        }
        zero_limit = dict.fromkeys(zero_limit_keys, {"desired_at_axis": 0})
        kwargs = weaker_tolerance | zero_limit
        # fixed iota
        assert_is_continuous(get("W7-X"), names=continuous, kwargs=kwargs)
        # fixed current
        assert_is_continuous(get("QAS"), names=continuous, kwargs=kwargs)
