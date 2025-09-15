"""Tests for things related to data_index."""

import inspect
import re

import pytest

import desc.compute
from desc.compute import data_index
from desc.compute.data_index import _class_inheritance
from desc.utils import errorif, getsource


def _get_matches(fun, pattern, ignore_comments=True):
    """Return all matches of ``pattern`` in source code of function ``fun``."""
    src = getsource(fun)
    if ignore_comments:
        # remove any decorator functions
        src = src.partition("def ")[2]
        # remove comments
        src = "\n".join(line.partition("#")[0] for line in src.splitlines())
    matches = pattern.findall(src)
    matches = {s.strip().strip('"') for s in matches}
    return matches


def _get_parameterization(fun, default="desc.equilibrium.equilibrium.Equilibrium"):
    """Get parameterization of thing computed by function ``fun``."""
    pattern = re.compile(r'parameterization=(?:\[([^]]+)]|"([^"]+)")')
    decorator = getsource(fun).partition("def ")[0]
    matches = pattern.findall(decorator)
    # if list was found, split strings in list, else string was found so get that
    matches = [match[0].split(",") if match[0] else [match[1]] for match in matches]
    # flatten the list
    matches = {s.strip().strip('"') for sublist in matches for s in sublist}
    matches.discard("")
    return matches if matches else {default}


@pytest.mark.unit
def test_data_index_deps():
    """Ensure developers do not add extra (or forget needed) dependencies.

    The regular expressions used in this test will fail to detect the data
    dependencies in the source code of compute functions if the query to
    the key in the data dictionary is split across multiple lines.
    To avoid failing this test unnecessarily in this case, try to refactor
    code by wrapping the query to the key in the data dictionary inside a
    parenthesis.

    Examples
    --------
    .. code-block:: python

        # Don't do this.
        x_square = data[
                           "x"
                       ] ** 2
        # Either do this
        x_square = (
               data["x"]
           ) ** 2
        # or do this
        x_square = data["x"] ** 2

    """
    queried_deps = {p: {} for p in _class_inheritance}

    pattern_name = re.compile(r"(?<!_)name=\"(.*?)\"")
    pattern_computed = re.compile(r"(?<!_)data\[(.*?)] = ")
    pattern_data = re.compile(r"(?<!_)data\[(.*?)]")
    pattern_profiles = re.compile(r"profiles\[(.*?)]")
    pattern_params = re.compile(r"params\[(.*?)]")
    pattern_dep_ignore = re.compile("noqa: unused dependency")
    for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
        if module_name[0] == "_":
            # JITed functions are not functions according to inspect,
            # so just check if callable and in the right module
            filt = lambda x: callable(x) and x.__module__ == module.__name__
            for _, fun in inspect.getmembers(module, filt):
                register_name = _get_matches(fun, pattern_name, ignore_comments=False)
                if not register_name:
                    continue
                else:
                    (register_name,) = register_name
                deps = {
                    "data": _get_matches(fun, pattern_data)
                    - _get_matches(fun, pattern_computed),
                    "profiles": _get_matches(fun, pattern_profiles),
                    "params": _get_matches(fun, pattern_params),
                    "ignore": bool(
                        _get_matches(fun, pattern_dep_ignore, ignore_comments=False)
                    ),
                }
                parameterization = _get_parameterization(fun)
                # same logic as desc.compute.data_index.py
                for p in parameterization:
                    for base_class, superclasses in _class_inheritance.items():
                        if p in superclasses or p == base_class:
                            # if it was already registered from a parent class, we
                            # prefer the child class.
                            inheritance_order = [base_class] + superclasses
                            if inheritance_order.index(p) > inheritance_order.index(
                                data_index[base_class][register_name][
                                    "parameterization"
                                ]
                            ):
                                continue
                            queried_deps[base_class][register_name] = deps
                            aliases = data_index[base_class][register_name]["aliases"]
                            for alias in aliases:
                                queried_deps[base_class][alias] = deps

    for p in data_index:
        for name, val in data_index[p].items():
            err_msg = f"Parameterization: {p}. Name: {name}."
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
            errorif(
                name not in queried_deps[p],
                AssertionError,
                "Did you reuse the function name (i.e. def_...) for"
                f" '{name}' for some other quantity?" + "\n" + err_msg,
            )
            # assert correct dependencies are queried
            if not queried_deps[p][name]["ignore"]:
                assert queried_deps[p][name]["data"] == data | axis_limit_data, err_msg
            assert queried_deps[p][name]["profiles"] == profiles, err_msg
            assert queried_deps[p][name]["params"] == params, err_msg
