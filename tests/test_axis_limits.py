"""Tests for compute functions evaluated at limits."""

import numpy as np
import pytest

from desc.compute import data_index
from desc.compute.utils import dot, surface_integrals_map
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid

# Unless mentioned in the source code of the compute function, the assumptions
# made to compute the magnetic axis limit can be reduced to assuming that these
# functions tend toward zero as the magnetic axis is approached and that
# d²ψ/(dρ)² and 𝜕√𝑔/𝜕𝜌 are both finite nonzero at the magnetic axis.
# Also, dⁿψ/(dρ)ⁿ for n > 3 is assumed zero everywhere.
zero_limits = {"rho", "psi", "psi_r", "e_theta", "sqrt(g)", "B_t"}
not_finite_limits = {
    "D_Mercier",
    "D_geodesic",
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
not_implemented_limits = {
    # reliant limits will be added to this set automatically
    "iota_num_rrr",
    "iota_den_rrr",
    "D_current",
}


def grow_seeds(
    seeds, search_space, parameterization="desc.equilibrium.equilibrium.Equilibrium"
):
    """Traverse the dependency DAG for keys in search space dependent on seeds.

    Parameters
    ----------
    seeds : set
        Keys to find paths toward.
    search_space : iterable
        Additional keys to consider returning.
    parameterization: str or list of str
        Name of desc types the method is valid for. eg 'desc.geometry.FourierXYZCurve'
        or `desc.equilibrium.Equilibrium`.

    Returns
    -------
    out : set
        All keys in search space with any path in the dependency DAG to any seed.

    """
    out = seeds.copy()
    for key in search_space:
        deps = data_index[parameterization][key]["full_with_axis_dependencies"]["data"]
        if not seeds.isdisjoint(deps):
            out.add(key)
    return out


not_implemented_limits = grow_seeds(
    not_implemented_limits,
    data_index["desc.equilibrium.equilibrium.Equilibrium"].keys() - not_finite_limits,
)
not_implemented_limits.discard("D_Mercier")


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
    names=data_index["desc.equilibrium.equilibrium.Equilibrium"].keys(),
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
    names : list of str
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
    # TODO: remove when boozer transform works with multiple surfaces
    names = [
        name
        for name in names
        if not (
            "Boozer" in name
            or "_mn" in name
            or name == "B modes"
            or _skip_this(eq, name)
        )
    ]

    num_points = 12
    rho = np.linspace(start=0, stop=delta, num=num_points)
    grid = LinearGrid(rho=rho, M=5, N=5, NFP=eq.NFP, sym=eq.sym)
    axis = grid.nodes[:, 0] == 0
    assert axis.any() and not axis.all()
    integrate = surface_integrals_map(grid, expand_out=False)
    data = eq.compute(names=names, grid=grid)

    p = "desc.equilibrium.equilibrium.Equilibrium"
    for name in names:
        if name in not_finite_limits:
            assert (np.isfinite(data[name]).T != axis).all(), name
            continue
        else:
            assert np.isfinite(data[name]).all(), name
        if data_index[p][name]["coordinates"] == "":
            # can't check continuity of global scalar
            continue
        # make single variable function of rho
        if data_index[p][name]["coordinates"] == "r":
            # already single variable function of rho
            profile = grid.compress(data[name])
        else:
            # integrate out theta and zeta dependence
            # Norms and integrals are continuous functions, so their composition
            # cannot disrupt existing continuity. Note that the absolute value
            # before the integration ensures that a discontinuous integrand does
            # not become continuous once integrated.
            profile = integrate(np.abs(data[name]))
        fit = kwargs.get(name, {}).get("desired_at_axis", desired_at_axis)
        if fit is None:
            if np.ndim(data_index[p][name]["dim"]):
                # can't polyfit tensor arrays like grad(B)
                fit = profile[1]
            else:
                # fit the data to a polynomial to extrapolate to axis
                poly = np.polynomial.polynomial.polyfit(
                    rho[1:], profile[1:], deg=min(4, num_points // 3)
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


class TestAxisLimits:
    """Tests for compute functions evaluated at limits."""

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
    def test_limit_continuity(self):
        """Heuristic to test correctness of all quantities with limits."""
        # The need for a weaker tolerance on these keys may be due to a subpar
        # polynomial regression fit against which the axis limit is compared.
        weaker_tolerance = {
            "B0_rr": {"rtol": 5e-03},
            "iota_r": {"atol": 1e-4},
            "iota_num_rr": {"atol": 5e-3},
            "alpha_r": {"rtol": 1e-3},
        }
        zero_map = dict.fromkeys(zero_limits, {"desired_at_axis": 0})
        # same as 'weaker_tolerance | zero_limit', but works on Python 3.8 (PEP 584)
        kwargs = dict(weaker_tolerance, **zero_map)
        # fixed iota
        assert_is_continuous(get("W7-X"), kwargs=kwargs)
        # fixed current
        assert_is_continuous(get("QAS"), kwargs=kwargs)

    @pytest.mark.unit
    def test_magnetic_field_is_physical(self):
        """Test direction of magnetic field at axis limit."""

        def test(eq):
            grid = LinearGrid(rho=0, M=5, N=5, NFP=eq.NFP, sym=eq.sym)
            assert grid.axis.size
            data = eq.compute(
                ["b", "n_theta", "n_rho", "e_zeta", "g_zz", "B"], grid=grid
            )
            # For the rotational transform to be finite at the magnetic axis,
            # the magnetic field must satisfy 𝐁 ⋅ 𝐞_ζ × 𝐞ᵨ = 0. This is also
            # required for 𝐁^θ component of the field to be physical.
            np.testing.assert_allclose(dot(data["b"], data["n_theta"]), 0, atol=1e-15)
            # and be orthogonal with 𝐞^ρ because 𝐞^ρ is multivalued at the
            # magnetic axis. 𝐁^ρ = 𝐁 ⋅ 𝐞^ρ must be single-valued for the
            # magnetic field to be physical. (The direction of the vector needs
            # to be unique).
            np.testing.assert_allclose(dot(data["b"], data["n_rho"]), 0, atol=1e-15)
            # and collinear with 𝐞_ζ near ρ=0
            np.testing.assert_allclose(
                # |𝐁_ζ| == ‖𝐁‖ ‖𝐞_ζ‖
                np.abs(dot(data["b"], (data["e_zeta"].T / np.sqrt(data["g_zz"])).T)),
                1,
            )
            # Explicitly check 𝐁 is single-valued at the magnetic axis.
            for B in data["B"].reshape((grid.num_zeta, -1, 3)):
                np.testing.assert_allclose(B[:, 0], B[0, 0])
                np.testing.assert_allclose(B[:, 1], B[0, 1])
                np.testing.assert_allclose(B[:, 2], B[0, 2])

        test(get("W7-X"))
        test(get("QAS"))
