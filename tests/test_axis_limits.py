"""Tests for compute functions evaluated at limits.

If a new quantity is added to the compute functions whose limit is not finite
(or does not exist), simply add it to the ``not_finite_limits`` set below.
If the limit has yet to be derived, add it to the ``not_implemented_limits`` set.
"""

import functools

import numpy as np
import pytest

from desc.compute import data_index
from desc.compute.utils import _grow_seeds
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals import surface_integrals_map
from desc.objectives import GenericObjective, ObjectiveFunction
from desc.utils import dot, errorif, getsource

# Unless mentioned in the source code of the compute function, the assumptions
# made to compute the magnetic axis limit can be reduced to assuming that these
# functions tend toward zero as the magnetic axis is approached and that
# d²ψ/(dρ)² and 𝜕√𝑔/𝜕𝜌 are both finite nonzero at the magnetic axis.
# Also, dⁿψ/(dρ)ⁿ for n > 3 is assumed zero everywhere.
zero_limits = {"rho", "psi", "psi_r", "psi_rrr", "e_theta", "sqrt(g)", "B_t"}

# These compute quantities require kinetic profiles, which are not defined for all
# configurations (giving NaN values). Gamma_c is 0 on axis.
not_continuous_limits = {"current Redl", "P_ISS04", "P_fusion", "<sigma*nu>", "Gamma_c"}

not_finite_limits = {
    "D_Mercier",
    "D_geodesic",
    "D_well",
    "J^theta",
    "J^theta_t",
    "J^theta_z",
    "curvature_H_rho",
    "curvature_H_zeta",
    "curvature_K_rho",
    "curvature_K_zeta",
    "curvature_k1_rho",
    "curvature_k1_zeta",
    "curvature_k2_rho",
    "curvature_k2_zeta",
    "cvdrift",
    "e^helical",
    "e^theta",
    "e^theta_r",
    "e^theta_t",
    "e^theta_z",
    "e^vartheta",
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
    "g^aa",
    "g^ra",
    "g^rv",
    "(g^rv_p)|PEST",
    "gbdrift",
    "grad(alpha)",
    "grad(alpha) (periodic)",
    "gbdrift (periodic)",
    "cvdrift (periodic)",
    "|e^helical|",
    "|grad(theta)|",
    "<J*B> Redl",  # may not exist for all configurations
    "current Redl",
    "J^theta_PEST",
    "(J^theta_PEST_v)|PEST",
    "(J^theta_PEST_p)|PEST",
    "(e^vartheta_v)|PEST",
    "(e^vartheta_p)|PEST",
}
not_implemented_limits = {
    # reliant limits will be added to this set automatically
    "D_current",
    "n_rho_z",
    "|e_theta x e_zeta|_z",
    "e^rho_rr",
    "e^theta_rr",
    "e^zeta_rr",
    "e^rho_rt",
    "e^rho_tt",
    "e^theta_rt",
    "e^theta_tt",
    "e^zeta_rt",
    "e^zeta_tt",
    "e^rho_rz",
    "e^rho_tz",
    "e^rho_zz",
    "e^theta_rz",
    "e^theta_tz",
    "e^theta_zz",
    "e^zeta_rz",
    "e^zeta_tz",
    "e^zeta_zz",
    "(e^zeta_v)|PEST",
    "(e^zeta_z)|PEST",
    "(e^rho_v)|PEST",
    "(e^rho_z)|PEST",
    "J^zeta_t",
    "J^zeta_z",
    "K_vc",  # only defined on surface
    "iota_num_rrr",
    "iota_den_rrr",
    "gds2",
    "(B*grad) grad(rho)",
}


def add_all_aliases(names):
    """Add aliases to limits."""
    all_aliases = []
    for name in names:
        for base_class in data_index.keys():
            if name in data_index[base_class].keys():
                all_aliases.append(data_index[base_class][name]["aliases"])

    # flatten
    all_aliases = [name for sublist in all_aliases for name in sublist]
    names.update(all_aliases)

    return names


zero_limits = add_all_aliases(zero_limits)
not_finite_limits = add_all_aliases(not_finite_limits)
not_implemented_limits = add_all_aliases(not_implemented_limits)
not_implemented_limits = _grow_seeds(
    "desc.equilibrium.equilibrium.Equilibrium",
    not_implemented_limits,
    data_index["desc.equilibrium.equilibrium.Equilibrium"].keys() - not_finite_limits,
    has_axis=True,
)


def _skip_this(eq, name):
    return (
        name in not_implemented_limits
        or (eq.atomic_number is None and "Zeff" in name)
        or (eq.electron_temperature is None and "Te" in name)
        or (eq.electron_density is None and "ne" in name)
        or (eq.ion_temperature is None and "Ti" in name)
        or (eq.electron_density is None and "ni" in name)
        or (eq.anisotropy is None and "beta_a" in name)
        or (eq.pressure is not None and "<J*B> Redl" in name)
        or (eq.current is None and "iota_num" in name)
        or bool(
            data_index["desc.equilibrium.equilibrium.Equilibrium"][name][
                "source_grid_requirement"
            ]
        )
        or bool(
            data_index["desc.equilibrium.equilibrium.Equilibrium"][name][
                "grid_requirement"
            ]
        )
    )


def assert_is_continuous(
    eq,
    names=data_index["desc.equilibrium.equilibrium.Equilibrium"].keys(),
    delta=1e-4,
    rtol=1e-5,
    atol=5e-7,
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
    p = "desc.equilibrium.equilibrium.Equilibrium"
    if kwargs is None:
        kwargs = {}
    names = set(names)
    names -= _grow_seeds(
        parameterization=p,
        seeds={
            name
            for name in names
            if _skip_this(eq, name)
            # TODO (#671): Boozer axis limits are not yet implemented
            or ("Boozer" in name or "_mn" in name or name == "B modes")
        },
        search_space=names,
    )
    names = list(names)

    num_points = 12
    rho = np.linspace(start=0, stop=delta, num=num_points)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    axis = grid.nodes[:, 0] == 0
    assert axis.any() and not axis.all()
    integrate = surface_integrals_map(grid, expand_out=False)
    data = eq.compute(names=names, grid=grid)

    for name in names:
        if name in not_continuous_limits:
            continue
        elif name in not_finite_limits:
            errorif(np.any(np.isfinite(data[name]).T == axis), AssertionError, msg=name)
            continue
        else:
            errorif(not np.isfinite(data[name]).all(), AssertionError, msg=name)

        if (
            data_index[p][name]["coordinates"] == ""
            or data_index[p][name]["coordinates"] == "z"
        ):
            # can't check radial continuity of scalar or function of toroidal angle
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
        name = "psi_r/sqrt(g)"
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
            "iota_r": {"atol": 1e-6},
            "iota_num_rr": {"atol": 5e-5},
            "grad(B)": {"rtol": 1e-4},
            "alpha_r (secular)": {"atol": 1e-4},
            "grad(alpha) (secular)": {"atol": 2e-4},
            "gbdrift (secular)": {"atol": 1e-4},
            "gbdrift (secular)/phi": {"atol": 1e-4},
            "(psi_r/sqrt(g))_rr": {"rtol": 2e-5},
        }
        zero_map = dict.fromkeys(zero_limits, {"desired_at_axis": 0})
        kwargs = weaker_tolerance | zero_map
        # fixed iota
        eq = get("W7-X")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)
        assert_is_continuous(eq, kwargs=kwargs)
        # fixed current
        eq = get("NCSX")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)
        assert_is_continuous(eq, kwargs=kwargs)

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

        eq = get("W7-X")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)
        test(eq)
        eq = get("NCSX")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)
        test(eq)


def _reverse_mode_unsafe_names():
    names = data_index["desc.equilibrium.equilibrium.Equilibrium"].keys()
    eq = get("ESTELL")

    def isalias(name):
        return isinstance(
            data_index["desc.equilibrium.equilibrium.Equilibrium"][name]["fun"],
            functools.partial,
        )

    def get_source(name):
        return "".join(
            getsource(
                data_index["desc.equilibrium.equilibrium.Equilibrium"][name]["fun"]
            ).split("def ")[1:]
        )

    names = [
        name
        for name in names
        if not (
            "Boozer" in name
            or "_mn" in name
            or name == "B modes"
            or _skip_this(eq, name)
            or name in not_finite_limits
            or isalias(name)
        )
    ]

    unsafe_names = []  # things that might have nan gradient but shouldn't
    for name in names:
        source = get_source(name)
        if "replace_at_axis" in source:
            unsafe_names.append(name)

    unsafe_names = sorted(unsafe_names)
    return unsafe_names


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("name", _reverse_mode_unsafe_names())
def test_reverse_mode_ad_axis(name):
    """Asserts that the rho=0 axis limits are reverse mode differentiable."""
    eq = get("ESTELL")
    grid = LinearGrid(rho=0.0, M=2, N=2, NFP=eq.NFP, sym=eq.sym)
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(2, 2, 2, 4, 4, 4)

    obj = ObjectiveFunction(GenericObjective(name, eq, grid=grid), use_jit=False)
    obj.build(verbose=0)
    g = obj.grad(obj.x())
    assert not np.any(np.isnan(g))
    print(np.count_nonzero(g), name)
