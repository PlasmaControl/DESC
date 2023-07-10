"""Tests for compute functions evaluated at limits."""

import numpy as np
import pytest

from desc.compute import data_index
from desc.compute.utils import surface_integrals
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid

not_finite_limit_keys = {
    "D_Mercier",
    "D_current",
    "D_shear",  # may exist for some configurations
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
    "<J*B> Redl",  # may exist for some configurations
}


def continuity(eq, name, expected_at_axis=None):
    """Test that the rho=0 axis limit of name is a continuous extension."""
    if data_index[name]["coordinates"] == "":
        return
    delta = 1e-6  # any smaller accumulates finite precision errors
    epsilon = 1e-5
    rho = np.linspace(0, 1, 10) * delta
    grid = LinearGrid(rho=rho, M=7, N=7, NFP=eq.NFP, sym=eq.sym)
    assert grid.axis.size

    quantity = eq.compute(name, grid=grid)[name]
    # FIXME: issues with boozer transform quantities limited to single surface
    if data_index[name]["coordinates"] == "r":
        quantity = grid.compress(quantity)
    else:
        quantity = surface_integrals(grid, np.abs(quantity), expand_out=False)
    # check continuity
    np.testing.assert_allclose(
        quantity[:-1], quantity[1:], atol=epsilon, rtol=epsilon, err_msg=name
    )
    # check expected value at axis
    if expected_at_axis is None:
        # fit the data (except axis pt) to a polynomial to extrapolate to axis
        poly = np.polyfit(rho[1:], quantity[1:], 6)
        expected_at_axis = poly[-1]  # constant term is same as eval poly at rho=0
    np.testing.assert_allclose(
        quantity[0], expected_at_axis, atol=epsilon, rtol=epsilon, err_msg=name
    )


def skip_atomic_profile(eq, name):
    """Return true if atomic profile associated with quantity is not set on eq."""
    return (
        (eq.atomic_number is None and "Zeff" in name)
        or (eq.electron_temperature is None and "Te" in name)
        or (eq.electron_density is None and "ne" in name)
        or (eq.ion_temperature is None and "Ti" in name)
        or (eq.pressure is not None and "<J*B> Redl" in name)
    )


class TestAxisLimits:
    """Tests for compute functions evaluated at limits."""

    # includes a fixed iota and fixed current equilibrium
    eqs = (get("W7-X"), get("QAS"))

    @pytest.mark.unit
    def test_data_index_deps_is_clean(self):
        """Ensure that developers do not add unnecessary dependencies."""
        # Todo: maybe also parse compute fun and also check all dependencies
        #    are requested from data dictionary.
        for key in data_index.keys():
            deps = data_index[key]["dependencies"]
            data = set(deps["data"])
            axis_limit_data = set(deps["axis_limit_data"])
            assert data.isdisjoint(axis_limit_data), key
            assert len(data) == len(deps["data"]), key
            assert len(axis_limit_data) == len(deps["axis_limit_data"]), key

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
        for eq in TestAxisLimits.eqs:
            grid = LinearGrid(L=2, M=2, N=2, sym=eq.sym, NFP=eq.NFP, axis=True)
            assert grid.axis.size
            data = eq.compute(list(data_index.keys()), grid=grid)
            is_axis = grid.nodes[:, 0] == 0
            for key in data_index.keys():
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
            for key in ("rho", "psi", "psi_r", "e_theta", "sqrt(g)"):
                continuity(eq, key, expected_at_axis=0)

    @pytest.mark.unit
    def test_finite_limits(self):
        """Heuristic to test correctness of all quantities with limits."""
        finite_limit_keys = data_index.keys() - not_finite_limit_keys
        for eq in TestAxisLimits.eqs:
            for key in finite_limit_keys:
                continuity(eq, key)
