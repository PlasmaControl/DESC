"""Tests for compute functions evaluated at limits."""
import inspect
import re

import numpy as np
import pytest

import desc.compute
from desc.compute import data_index
from desc.compute.data_index import _class_inheritance
from desc.compute.utils import surface_integrals_map
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

# Unless mentioned in the source code of the compute function, the assumptions
# made to compute the magnetic axis limit can be reduced to assuming that these
# functions tend toward zero as the magnetic axis is approached and that
# d^2𝜓/(d𝜌)^2 and 𝜕√𝑔/𝜕𝜌 are both finite nonzero at the magnetic axis.
# Also d^n𝜓/(d𝜌)^n for n > 3 is assumed zero everywhere.
zero_limits = {"rho", "psi", "psi_r", "e_theta", "sqrt(g)", "B_t"}

not_finite_limits = {
    "D_Mercier",
    "D_current",
    "D_geodesic",
    "D_shear",  # may not exist for all configurations
    "D_well",
    "J^theta",
    "curvature_H_rho",
    "curvature_H_zeta",
    "curvature_K_rho",
    "curvature_K_zeta",
    "curvature_k1_rho",
    "curvature_k1_zeta",
    "curvature_k2_rho",
    "curvature_k2_zeta",
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

# reliant limits will be added to this set automatically
not_implemented_limits = {
    "iota_num_rrr",
    "iota_den_rrr",
}


def grow_seeds(seeds, search_space):
    """Traverse the dependency DAG for keys in search space dependent on seeds.

    Parameters
    ----------
    seeds : set
        Keys to find paths toward.
    search_space : iterable
        Additional keys to consider returning.

    Returns
    -------
    out : set
        All keys in search space with any path in the dependency DAG to any seed.

    """
    out = seeds.copy()
    for key in search_space:
        deps = data_index["desc.equilibrium.equilibrium.Equilibrium"][key][
            "full_with_axis_dependencies"
        ]["data"]
        if not seeds.isdisjoint(deps):
            out.add(key)
    return out


not_implemented_limits = grow_seeds(
    not_implemented_limits,
    data_index["desc.equilibrium.equilibrium.Equilibrium"].keys() - not_finite_limits,
)


def _skip_this(eq, name):
    return (
        name in not_implemented_limits
        or (eq.atomic_number is None and "Zeff" in name)
        or (eq.electron_temperature is None and "Te" in name)
        or (eq.electron_density is None and "ne" in name)
        or (eq.ion_temperature is None and "Ti" in name)
        or (eq.pressure is not None and "<J*B> Redl" in name)
        or (eq.current is None and ("iota_num" in name or "iota_den" in name))
    )


def assert_is_continuous(
    eq,
    names,
    delta=5e-5,
    rtol=1e-4,
    atol=1e-6,
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
        Smaller values accumulate finite precision error and fitting issues.
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

    num_points = 15
    rho = np.linspace(start=0, stop=delta, num=num_points)
    grid = LinearGrid(rho=rho, M=5, N=5, NFP=eq.NFP, sym=eq.sym)
    assert grid.axis.size
    integrate = surface_integrals_map(grid, expand_out=False)
    data = eq.compute(names, grid=grid)

    data_index_eq = data_index["desc.equilibrium.equilibrium.Equilibrium"]
    for name in names:
        if _skip_this(eq, name) or data_index_eq[name]["coordinates"] == "":
            # can't check continuity of global scaler quantity
            continue
        # make single variable function of rho
        if data_index_eq[name]["coordinates"] == "r":
            # already single variable function of rho
            profile = grid.compress(data[name])
        else:
            # integrate out theta and zeta dependence
            profile = np.where(
                # True if integrand has nan on a given surface.
                integrate(np.isnan(data[name])).astype(bool),
                # The integration below replaces nan with 0; put them back.
                np.nan,
                # Norms and integrals are continuous functions, so their composition
                # cannot disrupt existing continuity. Note that the absolute value
                # before the integration ensures that a discontinuous integrand does
                # not become continuous once integrated.
                integrate(np.abs(data[name])),
            )
        fit = kwargs.get(name, {}).get("desired_at_axis", desired_at_axis)
        if fit is None:
            if np.ndim(data_index_eq[name]["dim"]):
                # can't polyfit tensor arrays like grad(B)
                fit = (profile[0] + profile[1]) / 2
            else:
                # fit the data to a polynomial to extrapolate to axis
                poly = np.polynomial.polynomial.polyfit(
                    rho[1:], profile[1:], deg=min(5, num_points // 3)
                )
                # constant term is the same as evaluating polynomial at rho=0
                fit = poly[0]
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
    matches = {s.strip().strip('"') for s in matches}
    return matches


def get_parameterization(fun, default="desc.equilibrium.equilibrium.Equilibrium"):
    """Get parameterization of this function."""
    pattern_parameterization = re.compile(
        r'parameterization=\[\s*([^]]+?)\s*\]|parameterization="([^"]+)"'
    )
    decorator = inspect.getsource(fun).partition("def ")[0]
    matches = pattern_parameterization.findall(decorator)
    # if list was found, split strings in list, else string was found so just get that
    matches = [match[0].split(",") if match[0] else [match[1]] for match in matches]
    # flatten the list
    matches = {match.strip().strip('"') for sublist in matches for match in sublist}
    matches.discard("")
    return matches if matches else {default}


class TestAxisLimits:
    """Tests for compute functions evaluated at limits."""

    @pytest.mark.unit
    def test_data_index_deps(self):
        """Ensure developers do not add extra (or forget needed) dependencies."""
        queried_deps = {}

        pattern_names = re.compile(r"(?<!_)data\[(.*?)\] =")
        pattern_data = re.compile(r"(?<!_)data\[(.*?)\]")
        pattern_profiles = re.compile(r"profiles\[(.*?)\]")
        pattern_params = re.compile(r"params\[(.*?)\]")
        for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
            if module_name[0] == "_":
                for _, fun in inspect.getmembers(module, inspect.isfunction):
                    # quantities that this function computes
                    names = get_matches(fun, pattern_names)
                    parameterization = get_parameterization(fun)
                    # dependencies queried in source code of this function
                    deps = {
                        "data": get_matches(fun, pattern_data) - names,
                        "profiles": get_matches(fun, pattern_profiles),
                        "params": get_matches(fun, pattern_params),
                    }
                    # some functions compute multiple things, e.g. curvature
                    for name in names:
                        # same logic as desc.compute.data_index.py
                        for p in parameterization:
                            for base_class, superclasses in _class_inheritance.items():
                                if p in superclasses or p == base_class:
                                    queried_deps.setdefault(base_class, {})[name] = deps

        for parameterization in data_index:
            for name, val in data_index[parameterization].items():
                err_msg = f"Parameterization: {parameterization}. Name: {name}."
                deps = val["dependencies"]
                data = set(deps["data"])
                axis_limit_data = set(deps["axis_limit_data"])
                profiles = set(deps["profiles"])
                params = set(deps["params"])
                # assert no duplicate dependencies
                assert len(data) == len(deps["data"]), err_msg
                assert len(axis_limit_data) == len(deps["axis_limit_data"]), err_msg
                assert data.isdisjoint(axis_limit_data), err_msg
                assert len(profiles) == len(deps["profiles"]), err_msg
                assert len(params) == len(deps["params"]), err_msg
                # assert correct dependencies are queried
                assert (
                    queried_deps[parameterization][name]["data"]
                    == data | axis_limit_data
                ), err_msg
                assert (
                    queried_deps[parameterization][name]["profiles"] == profiles
                ), err_msg
                assert queried_deps[parameterization][name]["params"] == params, err_msg

    @pytest.mark.unit
    def test_axis_limit_api(self):
        """Test that axis limit dependencies are computed only when necessary."""
        name = "B0"
        deps = {"psi_r", "sqrt(g)"}
        axis_limit_deps = {"psi_rr", "sqrt(g)_r"}
        eq = Equilibrium()
        grid = LinearGrid(L=2, M=1, N=1, axis=False)
        assert not grid.axis.size
        data = eq.compute(name, grid=grid).keys()
        assert name in data and deps < data and axis_limit_deps.isdisjoint(data)
        grid = LinearGrid(L=2, M=1, N=1, axis=True)
        assert grid.axis.size
        data = eq.compute(name, grid=grid)
        assert name in data and deps | axis_limit_deps < data.keys()
        assert np.isfinite(data[name]).all()

    @pytest.mark.unit
    def test_limit_existence(self):
        """Test that only quantities which lack limits do not evaluate at axis."""

        def test(eq):
            grid = LinearGrid(L=1, M=1, N=1, sym=eq.sym, NFP=eq.NFP, axis=True)
            at_axis = grid.nodes[:, 0] == 0
            assert at_axis.any() and not at_axis.all()
            data = eq.compute(
                list(data_index["desc.equilibrium.equilibrium.Equilibrium"].keys()),
                grid=grid,
            )
            for key in data_index["desc.equilibrium.equilibrium.Equilibrium"]:
                if _skip_this(eq, key):
                    continue
                is_finite = np.isfinite(data[key])
                if key in not_finite_limits:
                    assert np.all(is_finite.T ^ at_axis), key
                else:
                    assert np.all(is_finite), key

        # fixed iota
        # test(get("W7-X"))  # noqa: E800
        # fixed current
        # test(get("QAS"))  # noqa: E800

    @pytest.mark.unit
    def test_continuous_limits(self):
        """Heuristic to test correctness of all quantities with limits."""
        # It is possible for a discontinuity to propagate across dependencies,
        # so this test does not make sense for keys that rely on discontinuous
        # keys as dependencies.
        finite_discontinuous = set()
        continuous = (
            data_index["desc.equilibrium.equilibrium.Equilibrium"].keys()
            - not_finite_limits
        )
        continuous -= grow_seeds(finite_discontinuous, continuous)

        # The need for a weaker tolerance on these keys may be due to large
        # derivatives near axis, finite precision error, or a subpar polynomial
        # regression fit against which the axis limit is compared.
        rtol = "rtol"
        atol = "atol"
        weaker_tolerance = {
            "B0_rr": {rtol: 5e-03},
            "iota_r": {atol: 1e-4},
            "iota_num_rr": {atol: 5e-3},
            "alpha_r": {rtol: 1e-3},
        }
        zero_map = dict.fromkeys(zero_limits, {"desired_at_axis": 0})
        # same as 'weaker_tolerance | zero_limit', but works on Python 3.8 (PEP 584)
        kwargs = dict(weaker_tolerance, **zero_map)  # noqa: F841
        # fixed iota
        # assert_is_continuous(get("W7-X"), names=continuous, kwargs=kwargs) noqa: E800
        # fixed current
        # assert_is_continuous(get("QAS"), names=continuous, kwargs=kwargs) noqa: E800
