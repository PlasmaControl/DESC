"""Tests for things related to data_index."""

import inspect
import re

import pytest

import desc.compute
from desc.compute import data_index
from desc.compute.data_index import _class_inheritance
from desc.utils import errorif


class TestDataIndex:
    """Tests for things related to data_index."""

    @staticmethod
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

    @staticmethod
    def get_parameterization(fun, default="desc.equilibrium.equilibrium.Equilibrium"):
        """Get parameterization of thing computed by function ``fun``."""
        pattern = re.compile(r'parameterization=(?:\[([^]]+)]|"([^"]+)")')
        decorator = inspect.getsource(fun).partition("def ")[0]
        matches = pattern.findall(decorator)
        # if list was found, split strings in list, else string was found so get that
        matches = [match[0].split(",") if match[0] else [match[1]] for match in matches]
        # flatten the list
        matches = {s.strip().strip('"') for sublist in matches for s in sublist}
        matches.discard("")
        return matches if matches else {default}

    @pytest.mark.unit
    def test_data_index_deps(self):
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

        pattern_names = re.compile(r"(?<!_)data\[(.*?)] = ")
        pattern_data = re.compile(r"(?<!_)data\[(.*?)]")
        pattern_profiles = re.compile(r"profiles\[(.*?)]")
        pattern_params = re.compile(r"params\[(.*?)]")
        for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
            if module_name[0] == "_":
                # JITed functions are not functions according to inspect,
                # so just check if callable.
                for _, fun in inspect.getmembers(module, callable):
                    # quantities that this function computes
                    names = self.get_matches(fun, pattern_names)
                    # dependencies queried in source code of this function
                    deps = {
                        "data": self.get_matches(fun, pattern_data) - names,
                        "profiles": self.get_matches(fun, pattern_profiles),
                        "params": self.get_matches(fun, pattern_params),
                    }
                    parameterization = self.get_parameterization(fun)
                    # some functions compute multiple things, e.g. curvature
                    for name in names:
                        # same logic as desc.compute.data_index.py
                        for p in parameterization:
                            for base_class, superclasses in _class_inheritance.items():
                                if p in superclasses or p == base_class:
                                    queried_deps[base_class][name] = deps
                                    aliases = data_index[base_class][name]["aliases"]
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
                # assert correct dependencies are queried
                # TODO: conversion from rpz to xyz is taken out of actual function
                #       registration because of this data["phi"] is not queried in
                #       the source code but actually needed for the computation. This
                #       is a temporary fix until we have a better way to automatically
                #       handle this.
                assert queried_deps[p][name]["data"].issubset(
                    data | axis_limit_data
                ), err_msg
                errorif(
                    name not in queried_deps[p],
                    AssertionError,
                    "Did you reuse the function name (i.e. def_...) for"
                    f" '{name}' for some other quantity?",
                )
                assert queried_deps[p][name]["profiles"] == profiles, err_msg
                assert queried_deps[p][name]["params"] == params, err_msg
