"""Tests for the generalized toroidal angle: phi = zeta + omega(rho,theta,zeta).

Mostly self contained analytic surfaces and small equilibria. The exception is
the solved stellarator-mirror hybrid in
``tests/inputs/SLAM0_mirror_hybrid_omega.h5``, used where a real converged
equilibrium with a nontrivial omega is needed.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import mu_0

from desc.coils import FourierRZCoil
from desc.equilibrium import Equilibrium
from desc.geometry import (
    FourierRZCurve,
    FourierRZToroidalSurface,
    ZernikeRZToroidalSection,
)
from desc.grid import Grid, LinearGrid
from desc.io import load
from desc.objectives import (
    AxisWSelfConsistency,
    FixAxisW,
    FixBoundaryW,
    FixOmegaGauge,
    FixOmegaInterior,
    FixZetaSFL,
)
from desc.plotting import plot_boundary, plot_section, plot_surfaces


@pytest.fixture(scope="module")
def eq_omega():
    """Stellarator-mirror hybrid: NFP=2, sym, 246 omega modes to max 0.37 rad."""
    return load("./tests/inputs/SLAM0_mirror_hybrid_omega.h5")


def _make_synthetic_surface(NFP=1):
    """Analytic surface with nonzero omega for ground truth.

    R = 10 + cos(t) + 0.05 cos(t - NFP z), Z = -sin(t) - 0.05 sin(t - NFP z),
    omega = 0.10 sin(NFP z) + 0.05 sin(t - NFP z): R even, Z and omega odd, with
    the Z signs giving the right handed (theta, zeta) system DESC requires.
    """
    return FourierRZToroidalSurface(
        R_lmn=np.array([10.0, 1.0, 0.05]),
        Z_lmn=np.array([-1.0, -0.05]),
        modes_R=np.array([[0, 0], [1, 0], [1, 1]]),
        modes_Z=np.array([[-1, 0], [-1, 1]]),
        NFP=NFP,
        sym=True,
        W_lmn=np.array([0.10, 0.05]),
        modes_W=np.array([[0, -1], [-1, 1]]),
    )


def _grid(rho, theta, zeta):
    """Unsorted Grid from broadcastable (rho, theta, zeta) arrays."""
    r, t, z = np.broadcast_arrays(rho, theta, zeta)
    return Grid(np.vstack([np.ravel(r), np.ravel(t), np.ravel(z)]).T, sort=False)


def _sample_surface(surf, ntheta=40, nzeta=41):
    """Sample physical (R, phi, Z) from a surface with omega."""
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / surf.NFP, nzeta, endpoint=False)
    T, ZE = map(np.ravel, np.meshgrid(theta, zeta, indexing="ij"))
    data = surf.compute(["R", "Z", "phi"], grid=_grid(1.0, T, ZE))
    return T, ZE, np.array([data["R"], data["phi"], data["Z"]]).T


class TestSurfaceOmegaState:
    """State, serialization, and resolution handling of the generalized surface."""

    @pytest.mark.unit
    def test_omega_sym_parity(self):
        """Omega is odd under stellarator symmetry (sin basis, like Z)."""
        surf = _make_synthetic_surface()
        assert surf.sym and surf.W_basis.sym == "sin"
        grid1 = LinearGrid(M=6, N=6, NFP=surf.NFP)
        t, z = grid1.nodes[:, 1], grid1.nodes[:, 2]
        w1 = surf.compute("omega", grid=grid1)["omega"]
        w2 = surf.compute("omega", grid=_grid(1.0, -t, -z))["omega"]
        np.testing.assert_allclose(w1, -w2, atol=1e-14)

    @pytest.mark.unit
    def test_change_resolution_preserves_omega(self):
        """Changing (M, N, Mw, Nw) up or down must preserve the coefficients."""
        surf = _make_synthetic_surface()
        g = LinearGrid(M=4, N=4)
        w_old = surf.compute("omega", grid=g)["omega"]
        for M, N, Mw, Nw in [(5, 5, 4, 4), (5, 5, 1, 1)]:
            surf.change_resolution(M=M, N=N, Mw=Mw, Nw=Nw)
            assert (surf.Mw, surf.Nw) == (Mw, Nw)
            np.testing.assert_allclose(surf.compute("omega", grid=g)["omega"], w_old)
        # changing only M, N must not touch omega
        surf.change_resolution(M=8, N=8)
        assert (surf.Mw, surf.Nw) == (1, 1)
        np.testing.assert_allclose(surf.compute("omega", grid=g)["omega"], w_old)

    @pytest.mark.unit
    def test_serialization_roundtrip(self, tmpdir):
        """Save/load preserves omega and its basis."""
        surf = _make_synthetic_surface()
        path = os.path.join(tmpdir, "surf.h5")
        surf.save(path)
        surf2 = load(path)
        np.testing.assert_allclose(
            np.asarray(surf2.W_lmn), np.asarray(surf.W_lmn), atol=1e-14
        )
        assert surf2.W_basis.equiv(surf.W_basis)

    @pytest.mark.unit
    def test_flip_orientation_flips_omega(self):
        """Flipping theta orientation must flip m<0 omega modes."""
        surf = _make_synthetic_surface()
        w0 = np.asarray(surf.W_lmn).copy()
        surf._flip_orientation()
        expected = np.where(surf.W_basis.modes[:, 1] < 0, -w0, w0)
        np.testing.assert_allclose(np.asarray(surf.W_lmn), expected)


class TestSurfaceFitting:
    """Fitting a generalized surface from sampled physical coordinates."""

    @pytest.mark.unit
    @pytest.mark.parametrize("NFP", [1, 3])
    def test_fit_recovers_synthetic_surface(self, NFP):
        """Fit sampled points of an analytic omega surface; recover everything."""
        truth = _make_synthetic_surface(NFP)
        T, ZE, coords = _sample_surface(truth)
        surf = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mw=2, Nw=2, NFP=NFP, sym=True
        )
        # evaluate both surfaces on a *different* grid
        tt = np.linspace(0.1, 2 * np.pi, 27, endpoint=False)
        zz = np.linspace(0.05, 2 * np.pi / NFP, 25, endpoint=False)
        T2, Z2 = map(np.ravel, np.meshgrid(tt, zz, indexing="ij"))
        g2 = _grid(1.0, T2, Z2)
        keys = ["R", "Z", "omega", "phi", "x", "e_theta", "e_zeta", "n_rho"]
        dt = truth.compute(keys, grid=g2, basis="xyz")
        df = surf.compute(keys, grid=g2, basis="xyz")
        for key in keys:
            np.testing.assert_allclose(df[key], dt[key], atol=1e-9, err_msg=key)

    @pytest.mark.unit
    def test_fit_basis_vectors_vs_finite_differences(self):
        """Basis vectors of the fitted map agree with finite differences."""
        T, ZE, coords = _sample_surface(_make_synthetic_surface())
        surf = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mw=2, Nw=2, NFP=1, sym=True
        )
        t0, z0 = np.array([0.7, 1.9, 4.1]), np.array([0.3, 2.2, 5.0])
        eps = 1e-6

        def evalx(t, z):
            return surf.compute("x", grid=_grid(1.0, t, z), basis="xyz")["x"]

        d0 = surf.compute(["e_theta", "e_zeta"], grid=_grid(1.0, t0, z0), basis="xyz")
        fd_et = (evalx(t0 + eps, z0) - evalx(t0 - eps, z0)) / (2 * eps)
        fd_ez = (evalx(t0, z0 + eps) - evalx(t0, z0 - eps)) / (2 * eps)
        np.testing.assert_allclose(d0["e_theta"], fd_et, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(d0["e_zeta"], fd_ez, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    def test_omega_extraction_from_angles(self):
        """Omega = phi - zeta is branch-cut immune, and zero when zeta = phi."""
        T, ZE, coords = _sample_surface(_make_synthetic_surface())
        kw = dict(M=4, N=4, Mw=2, Nw=2, NFP=1, sym=True)
        surf1 = FourierRZToroidalSurface.from_values(coords, T, zeta=ZE, **kw)
        # wrapping either angle into arbitrary 2*pi branches cannot change the
        # fit: omega comes from the periodic difference
        rng = np.random.default_rng(0)
        wrapped = coords.copy()
        wrapped[:, 1] += 2 * np.pi * rng.integers(-3, 4, size=coords.shape[0])
        for other in [
            FourierRZToroidalSurface.from_values(wrapped, T, zeta=ZE, **kw),
            FourierRZToroidalSurface.from_values(coords, T, zeta=ZE + 2 * np.pi, **kw),
        ]:
            np.testing.assert_allclose(
                np.asarray(surf1.W_lmn), np.asarray(other.W_lmn), atol=1e-10
            )
        # parameterizing by the physical angle gives omega = 0 exactly, and
        # omitting zeta gives no omega modes at all
        surf = FourierRZToroidalSurface.from_values(
            coords, T, zeta=coords[:, 1], M=6, N=6, Mw=3, Nw=3, NFP=1, sym=True
        )
        np.testing.assert_allclose(np.asarray(surf.W_lmn), 0, atol=1e-12)
        no_zeta = FourierRZToroidalSurface.from_values(coords, T, M=6, N=6, sym=True)
        assert no_zeta.W_basis.num_modes == 0

    @pytest.mark.unit
    def test_fit_xyz_basis(self):
        """Cartesian (X, Y, Z) input gives the same fit as cylindrical."""
        T, ZE, coords = _sample_surface(_make_synthetic_surface())
        R, phi = coords[:, 0], coords[:, 1]
        xyz = np.array([R * np.cos(phi), R * np.sin(phi), coords[:, 2]]).T
        kw = dict(M=4, N=4, Mw=2, Nw=2, NFP=1, sym=True)
        surf_rpz = FourierRZToroidalSurface.from_values(coords, T, zeta=ZE, **kw)
        surf_xyz = FourierRZToroidalSurface.from_values(
            xyz, T, zeta=ZE, basis="xyz", **kw
        )
        for attr in ["R_lmn", "Z_lmn", "W_lmn"]:
            np.testing.assert_allclose(
                np.asarray(getattr(surf_xyz, attr)),
                np.asarray(getattr(surf_rpz, attr)),
                atol=1e-10,
                err_msg=attr,
            )

    @pytest.mark.unit
    def test_invalid_toroidal_map_detected(self):
        """A map with d(phi)/d(zeta) <= 0 somewhere must be rejected."""
        # omega = 1.2 sin(zeta) => 1 + omega_zeta = 1 + 1.2 cos(zeta) < 0
        bad = FourierRZToroidalSurface(
            W_lmn=np.array([1.2]), modes_W=np.array([[0, -1]])
        )
        with pytest.raises(ValueError, match="not a valid toroidal"):
            bad.check_toroidal_map()
        assert _make_synthetic_surface().check_toroidal_map() > 0.5

    @pytest.mark.unit
    def test_constant_offset_surface_with_omega(self):
        """The offset surface inherits the base surface's toroidal chart.

        Offset points sit at the cylindrical angle the base surface assigns to
        its own (theta, zeta), so the fit must reproduce them at those labels.
        Carrying omega is what makes that true: the same R, Z fit with omega
        dropped misses by ~1 m.
        """
        base = _make_synthetic_surface()
        offset = 0.5
        # explicit non-symmetric grid: the default uses sym=base_surface.sym,
        # which halves the fitted coefficients
        grid = LinearGrid(M=12, N=12, NFP=1, sym=False)
        surf, data, _ = base.constant_offset_surface(
            offset, grid, M=10, N=10, full_output=True
        )
        np.testing.assert_allclose(
            np.asarray(surf.W_lmn), np.asarray(base.W_lmn), atol=1e-14
        )
        assert surf.check_toroidal_map() > 0
        sep = np.linalg.norm(
            np.asarray(data["x"]) - np.asarray(data["x_offset_surface"]), axis=1
        )
        np.testing.assert_allclose(sep, offset, rtol=1e-4)

        def _xyz(s):
            d = s.compute(["R", "Z", "phi"], grid=Grid(grid.nodes, sort=False))
            R, p, Z = (np.asarray(d[k]) for k in ("R", "phi", "Z"))
            return np.stack([R * np.cos(p), R * np.sin(p), Z], axis=1)

        rpz = np.asarray(data["x_offset_surface"])
        target = np.stack(
            [rpz[:, 0] * np.cos(rpz[:, 1]), rpz[:, 0] * np.sin(rpz[:, 1]), rpz[:, 2]],
            axis=1,
        )
        np.testing.assert_allclose(_xyz(surf), target, atol=1e-9)
        # negative control: the same R, Z without omega is far off, so the
        # assertion above really tests the inherited chart
        dropped = FourierRZToroidalSurface(
            surf.R_lmn,
            surf.Z_lmn,
            surf.R_basis.modes[:, 1:],
            surf.Z_basis.modes[:, 1:],
            base.NFP,
            base.sym,
            check_orientation=False,
        )
        assert np.linalg.norm(_xyz(dropped) - target, axis=1).max() > 0.1

    @pytest.mark.unit
    def test_constant_offset_surface_zero_omega_unchanged(self):
        """With omega = 0 the offset algorithm is the classical one."""
        surf = FourierRZToroidalSurface()
        off = surf.constant_offset_surface(1.0, LinearGrid(M=6, N=2), M=1, N=1)
        assert off.W_basis.num_modes == 0
        R = np.asarray(off.R_lmn)
        np.testing.assert_allclose(R[off.R_basis.get_idx(M=0, N=0)], 10)
        np.testing.assert_allclose(R[off.R_basis.get_idx(M=1, N=0)], 2)


class TestEquilibriumOmega:
    """Generalized equilibrium coordinates."""

    @pytest.mark.unit
    def test_zero_omega_regression(self):
        """An eq with zero-valued omega modes matches one with no modes."""
        eq0 = Equilibrium(L=4, M=4, N=2, NFP=3, sym=True)
        eq1 = Equilibrium(L=4, M=4, N=2, NFP=3, sym=True, Lw=2, Mw=2, Nw=2)
        assert eq0.W_basis.num_modes == 0 and eq1.W_basis.num_modes > 0
        np.testing.assert_allclose(np.asarray(eq1.W_lmn), 0)
        grid = LinearGrid(L=4, M=8, N=8, NFP=3)
        keys = ["R", "Z", "phi", "|B|", "sqrt(g)", "|F|", "g_tt", "g_zz", "B_zeta"]
        d0, d1 = eq0.compute(keys, grid=grid), eq1.compute(keys, grid=grid)
        for key in keys:
            np.testing.assert_allclose(d0[key], d1[key], atol=1e-14, err_msg=key)

    @pytest.mark.unit
    def test_old_equilibrium_loads(self):
        """Equilibria saved before omega existed load with omega = 0."""
        import desc.examples

        eq = desc.examples.get("DSHAPE")
        assert eq.W_basis.num_modes == 0 and eq.W_lmn.size == 0
        assert eq.Lw == eq.Mw == eq.Nw == 0
        data = eq.compute(["omega", "phi", "|B|"], grid=LinearGrid(L=2, M=4, N=0))
        np.testing.assert_allclose(data["omega"], 0)
        assert np.all(np.isfinite(data["|B|"]))

    @pytest.mark.unit
    def test_phi_derivative_identities(self):
        """Derivatives of phi match those of omega, up to the 1 from zeta."""
        eq = Equilibrium(L=4, M=4, N=2, NFP=2, sym=True, Lw=2, Mw=2, Nw=2)
        rng = np.random.default_rng(3)
        eq.W_lmn = 0.02 * rng.standard_normal(eq.W_basis.num_modes)
        derivs = ["r", "t", "z", "rr", "tt", "zz", "rt", "rz", "tz"]
        keys = ["phi", "zeta", "omega"]
        keys += [f"{q}_{d}" for d in derivs for q in ("phi", "omega")]
        d = eq.compute(keys, grid=LinearGrid(L=3, M=6, N=6, NFP=2))
        np.testing.assert_allclose(d["phi"], d["zeta"] + d["omega"], atol=1e-14)
        for s in derivs:
            np.testing.assert_allclose(
                d[f"phi_{s}"],
                (1 if s == "z" else 0) + d[f"omega_{s}"],
                atol=1e-14,
                err_msg=s,
            )
        assert np.max(np.abs(d["omega"])) > 0  # actually nonzero

    @pytest.mark.unit
    def test_omega_derivatives_vs_finite_differences(self):
        """Transform-based omega derivatives agree with finite differences."""
        eq = Equilibrium(L=4, M=4, N=2, NFP=2, sym=True, Lw=2, Mw=2, Nw=2)
        rng = np.random.default_rng(5)
        eq.W_lmn = 0.02 * rng.standard_normal(eq.W_basis.num_modes)
        r = np.array([0.5, 0.7, 0.9])
        t = np.array([0.4, 2.0, 5.1])
        z = np.array([0.2, 1.1, 2.8])
        eps = 1e-6

        def w(r, t, z):
            return eq.compute("omega", grid=_grid(r, t, z))["omega"]

        d = eq.compute(["omega_r", "omega_t", "omega_z"], grid=_grid(r, t, z))
        for key, plus, minus in [
            ("omega_r", (r + eps, t, z), (r - eps, t, z)),
            ("omega_t", (r, t + eps, z), (r, t - eps, z)),
            ("omega_z", (r, t, z + eps), (r, t, z - eps)),
        ]:
            fd = (w(*plus) - w(*minus)) / (2 * eps)
            np.testing.assert_allclose(d[key], fd, rtol=1e-5, atol=1e-8, err_msg=key)

    @pytest.mark.unit
    def test_compute_everything_nonzero_omega(self):
        """All standard rtz quantities stay finite with nonzero omega."""
        from desc.compute import data_index, get_data_deps

        eq = Equilibrium(L=4, M=4, N=2, NFP=3, sym=True, Lw=2, Mw=2, Nw=2)
        rng = np.random.default_rng(7)
        eq.W_lmn = 0.01 * rng.standard_normal(eq.W_basis.num_modes)
        assert eq.is_nested()
        # off axis: e^theta ~ 1/sqrt(g) is singular at rho=0 regardless of omega
        grid = LinearGrid(rho=np.linspace(0.2, 1.0, 5), M=8, N=8, NFP=3)
        p = "desc.equilibrium.equilibrium.Equilibrium"

        def _plain(name):
            """Computable on a plain rtz grid: no special grid anywhere in deps."""
            if data_index[p][name]["coordinates"] != "rtz":
                return False
            return not any(
                data_index[p][dep][req]
                for dep in [name] + get_data_deps(name, p)
                for req in (
                    "source_grid_requirement",
                    "grid_requirement",
                    "resolution_requirement",
                )
            )

        names = [name for name in data_index[p] if _plain(name)]
        assert len(names) > 200, f"expected to sweep many quantities, got {len(names)}"
        data = eq.compute(names, grid=grid)

        # the same equilibrium with omega zeroed: anything non-finite there is
        # non-finite for reasons unrelated to omega, which isolates omega
        eq0 = eq.copy()
        eq0.W_lmn = np.zeros(eq0.W_basis.num_modes)
        data0 = eq0.compute(names, grid=grid)
        bad = {n for n in names if not np.all(np.isfinite(np.asarray(data[n])))}
        bad0 = {n for n in names if not np.all(np.isfinite(np.asarray(data0[n])))}
        assert bad == bad0, (
            "omega changed which quantities are finite. Non-finite only with "
            f"omega != 0: {sorted(bad - bad0)}; only with omega == 0: "
            f"{sorted(bad0 - bad)}"
        )
        # beta_a and friends are NaN by design without an anisotropy profile
        assert bad0 <= {"beta_a", "beta_a_r", "beta_a_t", "beta_a_z", "grad(beta_a)"}
        # and omega actually changed the answers, so this was a real test
        assert not np.allclose(data["sqrt(g)"], data0["sqrt(g)"])
        assert np.all(data["sqrt(g)"] > 0)

    @pytest.mark.unit
    def test_axisymmetric_coordinate_invariance(self):
        """Pure-zeta omega reparameterizes an axisymmetric equilibrium.

        zeta -> phi = zeta + omega(zeta) sweeps out the identical physical
        field with unchanged R_lmn, Z_lmn, L_lmn, so every coordinate invariant
        must match.
        """
        from desc.utils import copy_coeffs

        eq0 = Equilibrium(L=4, M=4, N=0, sym=True)
        eq0.solve(verbose=0, maxiter=25, ftol=1e-6)

        eq1 = Equilibrium(L=4, M=4, N=2, sym=True, Lw=0, Mw=0, Nw=2)
        for attr, basis in [("R_lmn", "R_basis"), ("Z_lmn", "Z_basis")]:
            setattr(
                eq1,
                attr,
                copy_coeffs(
                    getattr(eq0, attr),
                    getattr(eq0, basis).modes,
                    getattr(eq1, basis).modes,
                ),
            )
        eq1.L_lmn = copy_coeffs(eq0.L_lmn, eq0.L_basis.modes, eq1.L_basis.modes)
        eq1.pressure.params = eq0.pressure.params.copy()
        eq1.current.params = eq0.current.params.copy()
        # omega = 0.1 sin(zeta) + 0.03 sin(2 zeta): no rho or theta dependence
        W = np.zeros(eq1.W_basis.num_modes)
        W[eq1.W_basis.get_idx(0, 0, -1)] = 0.1
        W[eq1.W_basis.get_idx(0, 0, -2)] = 0.03
        eq1.W_lmn = W
        eq1.surface = eq1.get_surface_at(rho=1.0)
        eq1.axis = eq1.get_axis()

        grid = LinearGrid(L=6, M=10, N=8)
        keys = ["|B|", "|F|", "sqrt(g)", "V", "S"]
        d0 = eq0.compute(keys, grid=LinearGrid(L=6, M=10, N=8))
        d1 = eq1.compute(keys, grid=grid)
        # |B| depends only on (rho, theta) here, so it matches node for node.
        # |F| is a residual of large cancelling terms, so the two charts differ
        # at roundoff amplified to ~1e-7 relative.
        np.testing.assert_allclose(d1["|B|"], d0["|B|"], rtol=1e-10)
        np.testing.assert_allclose(d1["|F|"], d0["|F|"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(d1["V"], d0["V"], rtol=1e-10)
        np.testing.assert_allclose(d1["S"], d0["S"], rtol=1e-10)
        # sqrt(g) is NOT invariant (it scales by 1 + omega_zeta): proof that
        # this test would catch a real chart effect
        assert not np.allclose(d1["sqrt(g)"], d0["sqrt(g)"], rtol=1e-3)
        # same physical torus: x(rho,theta,zeta') of eq1 is x of eq0 at phi(zeta')
        d1x = eq1.compute(["x", "phi"], grid=Grid(grid.nodes, sort=False), basis="xyz")
        nodes0 = grid.nodes.copy()
        nodes0[:, 2] = d1x["phi"]
        d0x = eq0.compute("x", grid=Grid(nodes0, sort=False), basis="xyz")
        np.testing.assert_allclose(d1x["x"], d0x["x"], atol=1e-10)
        assert eq1.is_nested()

    @pytest.mark.unit
    def test_solve_with_omega_fixed(self):
        """A solve with omega_b = 0.1 sin(zeta) matches the conventional one."""
        eq0 = Equilibrium(L=4, M=4, N=0, sym=True)
        eq0.solve(verbose=0, maxiter=30, ftol=1e-6)

        surf = FourierRZToroidalSurface(
            R_lmn=[10, 1],
            Z_lmn=[0, -1],
            modes_R=[[0, 0], [1, 0]],
            modes_Z=[[0, 0], [-1, 0]],
            sym=True,
            W_lmn=np.array([0.1]),
            modes_W=np.array([[0, -1]]),
        )
        eq1 = Equilibrium(L=4, M=4, N=2, sym=True, surface=surf, Lw=0, Mw=0, Nw=1)
        assert np.max(np.abs(np.asarray(eq1.W_lmn))) > 0  # initial guess has omega
        eq1.solve(verbose=0, maxiter=30, ftol=1e-6)

        g = LinearGrid(L=4, M=8, N=8)
        d1 = eq1.compute(["omega", "|B|", "|F|", "V"], grid=g)
        d0 = eq0.compute(["|B|", "|F|", "V"], grid=LinearGrid(L=4, M=8, N=8))
        assert np.max(np.abs(d1["omega"])) > 0.05  # preserved by the solve
        np.testing.assert_allclose(d1["V"], d0["V"], rtol=1e-4)
        np.testing.assert_allclose(d1["|B|"], d0["|B|"], rtol=2e-3)
        assert np.mean(d1["|F|"]) < 2 * np.mean(d0["|F|"]) + 1e-3


class TestCurveOmega:
    """Generalized angle support on FourierRZCurve."""

    @pytest.mark.unit
    def test_curve_omega_positions(self):
        """Curve x with W matches analytic phi = s + W(s)."""
        curve = FourierRZCurve(
            R_n=[0, 10, 1],
            Z_n=[0, 0, -1],
            NFP=1,
            sym=False,
            W_n=np.array([0.2]),
            modes_W=np.array([-1]),
        )
        s = np.linspace(0, 2 * np.pi, 17, endpoint=False)
        d = curve.compute(["x", "x_s", "x_ss"], grid=_grid(0.0, 0.0, s), basis="xyz")
        # modes [-1, 0, 1] = [sin(s), 1, cos(s)], so R = 10 + cos(s), Z = -cos(s)
        R, phi = 10 + np.cos(s), s + 0.2 * np.sin(s)
        x_true = np.array([R * np.cos(phi), R * np.sin(phi), -np.cos(s)]).T
        np.testing.assert_allclose(d["x"], x_true, atol=1e-12)

        def evalx(sv):
            return curve.compute("x", grid=_grid(0.0, 0.0, sv), basis="xyz")["x"]

        eps = 1e-6
        fd1 = (evalx(s + eps) - evalx(s - eps)) / (2 * eps)
        np.testing.assert_allclose(d["x_s"], fd1, rtol=1e-5, atol=1e-6)
        # a second difference divides by eps**2, so it needs a much larger step:
        # at eps = 1e-6 the roundoff floor alone is ~1e-2 here
        eps2 = 1e-4
        fd2 = (evalx(s + eps2) - 2 * evalx(s) + evalx(s - eps2)) / eps2**2
        np.testing.assert_allclose(d["x_ss"], fd2, rtol=1e-4, atol=1e-5)

    @pytest.mark.unit
    def test_curve_zero_omega_unchanged(self):
        """Curves without W behave exactly as before."""
        curve = FourierRZCurve(R_n=[0, 10, 1], Z_n=[0, 0, -1], NFP=1, sym=False)
        assert curve.W_basis.num_modes == 0  # no omega modes at all
        np.testing.assert_allclose(np.asarray(curve.W_n), 0)
        s = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        d = curve.compute(["x"], grid=_grid(0.0, 0.0, s), basis="xyz")
        R = 10 + np.cos(s)
        x_true = np.array([R * np.cos(s), R * np.sin(s), -np.cos(s)]).T
        np.testing.assert_allclose(d["x"], x_true, atol=1e-12)


class TestNoSpuriousOmegaDOF:
    """Objects without a generalized angle carry no omega degree of freedom.

    An empty omega basis is what "omega == 0" means: a non-symmetric basis at
    zero resolution still carries its constant mode, and left in that mode is a
    live optimizable parameter on every asymmetric object.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("sym", [True, False])
    def test_omega_free_objects_have_empty_W_basis(self, sym):
        """Default objects have no W modes in either symmetry.

        Coils are here because FourierRZCoil subclasses FourierRZCurve, but a
        coil is a lab-frame filament with no generalized angle.
        """
        coil = FourierRZCoil(current=1e6, R_n=np.array([10.0]), sym=sym)
        surf = FourierRZToroidalSurface(sym=sym)
        eq = Equilibrium(L=4, M=4, N=2, sym=sym)
        for thing, key in [
            (surf, "W_lmn"),
            (FourierRZCurve(sym=sym), "W_n"),
            (coil, "W_n"),
            (eq, "W_lmn"),
        ]:
            assert thing.W_basis.num_modes == 0
            assert thing.dimensions[key] == 0
        # a section is a constant-zeta object: omega is not a parameter at all
        sect = ZernikeRZToroidalSection(sym=sym)
        assert sect.W_basis.num_modes == 0 and "W_lmn" not in sect.dimensions
        assert eq.Lw == eq.Mw == eq.Nw == 0
        np.testing.assert_allclose(
            surf.compute("omega", grid=LinearGrid(M=4, N=4))["omega"], 0
        )
        # re-resolving must not resurrect the constant mode
        surf.change_resolution(M=4, N=3)
        eq.change_resolution(L=6, M=6, N=3)
        coil.change_resolution(N=4)
        for thing in [surf, eq, coil]:
            assert thing.W_basis.num_modes == 0

    @pytest.mark.unit
    def test_omega_can_still_be_requested(self):
        """Asking for omega resolution still builds the full asymmetric basis."""
        # asymmetric omega bases are twice the size of the symmetric ones
        assert FourierRZToroidalSurface(sym=False, Mw=2, Nw=2).W_basis.num_modes == 25
        assert FourierRZToroidalSurface(sym=True, Mw=2, Nw=2).W_basis.num_modes == 12
        # opting in after construction works too
        surf2 = FourierRZToroidalSurface(sym=False)
        surf2.change_resolution(M=2, N=2, Mw=2, Nw=2)
        assert surf2.W_basis.num_modes == 25
        eq = Equilibrium(L=4, M=4, N=2, sym=False, Lw=2, Mw=1, Nw=1)
        n_before = eq.W_basis.num_modes
        assert n_before > 0
        eq.change_resolution(Lw=4, Mw=2, Nw=2)
        assert eq.W_basis.num_modes > n_before

    @pytest.mark.unit
    @pytest.mark.parametrize("sym", [True, False])
    def test_old_files_load_without_omega(self, sym):
        """The _set_up back-compat path builds an empty basis in both symmetries."""
        for thing in [
            FourierRZToroidalSurface(sym=sym),
            ZernikeRZToroidalSection(sym=sym),
            FourierRZCurve(sym=sym),
        ]:
            old = thing.copy()
            name = "_W_n" if isinstance(old, FourierRZCurve) else "_W_lmn"
            delattr(old, name)
            del old._W_basis
            old._set_up()
            assert old.W_basis.num_modes == 0
            assert getattr(old, name[1:]).size == 0

    @pytest.mark.unit
    def test_coil_field_is_classical(self):
        """A circular coil reproduces the analytic loop field on its axis.

        A spurious omega is not inert: it moves the filament, and a constant one
        moved this coil's on-axis |B| by 24x before the basis was made empty.
        """
        R0, I = 2.0, 1e6
        coil = FourierRZCoil(current=I, R_n=np.array([R0]), sym=False)
        z = np.array([0.0, 0.5, 1.0, 2.0])
        coords = np.array([np.zeros_like(z), np.zeros_like(z), z]).T
        B = np.asarray(coil.compute_magnetic_field(coords, basis="rpz"))
        Bz = mu_0 * I * R0**2 / (2 * (R0**2 + z**2) ** 1.5)
        np.testing.assert_allclose(B[:, :2], 0, atol=1e-12)
        np.testing.assert_allclose(B[:, 2], Bz, rtol=1e-10)


def _make_elongated_surface(NFP=1, omega=True, elong=2.5):
    """Elongated cross-section, optionally with nonzero omega.

    Deliberately not near-circular: Ramanujan's inversion is singular as
    a/b -> 1, where a 0.05 % perimeter change swings elongation by tens of
    percent and the comparison measures nothing.
    """
    kw = {}
    if omega:
        kw = dict(W_lmn=np.array([0.10, 0.05]), modes_W=np.array([[0, -1], [-1, 1]]))
    return FourierRZToroidalSurface(
        R_lmn=np.array([10.0, 1.0, 0.05]),
        Z_lmn=np.array([-elong, -0.05]),
        modes_R=np.array([[0, 0], [1, 0], [1, 1]]),
        modes_Z=np.array([[-1, 0], [-1, 1]]),
        NFP=NFP,
        sym=True,
        **kw,
    )


@pytest.mark.unit
def test_area_on_solved_omega_equilibrium(eq_omega):
    """A(z), perimeter(z), elongation match a direct measurement, real omega solve."""
    theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    grid = _grid(1.0, theta, 0.3)
    d = eq_omega.compute(["R", "Z", "phi"], grid=grid)
    R, Z, phi = np.asarray(d["R"]), np.asarray(d["Z"]), np.asarray(d["phi"])
    xyz = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=1)
    xyz -= xyz.mean(axis=0)
    # Best-fit plane through the (centered) cross-section: SVD of the point
    # matrix gives xyz = U @ diag(S) @ Vt, where Vt's rows are an orthonormal
    # basis ordered by how much of the points' spread each direction explains
    # (largest singular value first). A flat ring of points has ~0 spread
    # along its normal, so that direction gets the smallest singular value --
    # Vt[2] (dropped here) is the fitted plane's normal, and Vt[0], Vt[1] are
    # two orthonormal directions spanning the plane itself. Projecting the
    # points onto those two directions (xyz @ Vt[0], xyz @ Vt[1]) gives their
    # 2D coordinates in that plane, ready for the shoelace formula below.
    _, _, Vt = np.linalg.svd(xyz, full_matrices=False)
    u, v = xyz @ Vt[0], xyz @ Vt[1]
    direct_area = 0.5 * abs(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v))
    direct_perim = np.sum(np.hypot(np.roll(u, -1) - u, np.roll(v, -1) - v))
    # Ramanujan elongation, matching desc.compute._geometry
    a = (
        np.sqrt(3)
        * (
            np.sqrt(8 * np.pi * direct_area + direct_perim**2)
            + np.sqrt(np.abs(
                2 * np.sqrt(3) * direct_perim
                * np.sqrt(8 * np.pi * direct_area + direct_perim**2)
                - 40 * np.pi * direct_area + 4 * direct_perim**2
            ))
        )
        + 3 * direct_perim
    ) / (12 * np.pi)
    direct_elong = a / (direct_area / (np.pi * a))

    # LinearGrid + override_grid=False is required: QuadratureGrid (the
    # default) integrates |e_rho x e_theta| over the non-planar constant-zeta
    # sheet itself, not the enclosed cross-sectional area, and is off by tens
    # of percent when omega != 0; override_grid=True silently swaps back to
    # that QuadratureGrid path even if a LinearGrid is passed in.
    lg = LinearGrid(
        L=eq_omega.L_grid, theta=512, zeta=0.3, NFP=eq_omega.NFP, sym=False
    )
    kw = dict(grid=lg, override_grid=False)
    area = eq_omega.compute("A(z)", **kw)["A(z)"][0]
    perim = eq_omega.compute("perimeter(z)", **kw)["perimeter(z)"][0]
    elong = eq_omega.compute("a_major/a_minor", **kw)["a_major/a_minor"][0]

    assert abs(area / direct_area - 1) < 0.05
    assert abs(perim / direct_perim - 1) < 0.05
    assert abs(elong / direct_elong - 1) < 0.05


class TestPlottingWithOmega:
    """Constant-phi plots on an equilibrium whose zeta is not phi.

    The plotters invert phi = zeta + omega before evaluating. A failed
    inversion is masked with NaN and warned about, leaving holes in the figure,
    so "no warning and no NaN" is the assertion.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "call, keys",
        [
            (
                lambda eq: plot_section(
                    eq, "|B|", phi=0.4, return_data=True, figsize=(4, 4)
                ),
                ["R", "Z", "|B|"],
            ),
            (  # a user supplied grid takes the same inversion path as phi=
                lambda eq: plot_section(
                    eq,
                    "|B|",
                    grid=LinearGrid(
                        rho=np.linspace(0.2, 1.0, 5), theta=12, zeta=np.array([0.4])
                    ),
                    return_data=True,
                    figsize=(4, 4),
                ),
                ["R", "Z", "|B|"],
            ),
            (
                lambda eq: plot_surfaces(
                    eq, rho=4, theta=0, phi=2, return_data=True, figsize=(4, 4)
                ),
                ["rho_R_coords", "rho_Z_coords"],
            ),
            (  # theta != 0 adds the vartheta contours, which invert phi too
                lambda eq: plot_surfaces(
                    eq, rho=4, theta=6, phi=2, return_data=True, figsize=(4, 4)
                ),
                ["rho_R_coords", "rho_Z_coords", "vartheta_R_coords"],
            ),
            (
                lambda eq: plot_boundary(eq, phi=3, return_data=True),
                ["R", "Z"],
            ),
        ],
    )
    def test_plotters_invert_phi(self, eq_omega, call, keys):
        """Each plotter inverts phi without failing or leaving NaN holes."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = call(eq_omega)
        plt.close(out[0])
        assert not [w for w in record if "phi -> zeta inversion" in str(w.message)]
        for key in keys:
            assert np.all(np.isfinite(np.asarray(out[-1][key]))), key

    @pytest.mark.unit
    def test_plot_section_is_not_the_constant_zeta_cut(self, eq_omega):
        """Zeroing omega moves the section, so an inversion really happened."""
        kw = dict(phi=0.4, return_data=True, figsize=(4, 4))
        fig, _, data = plot_section(eq_omega, "|B|", **kw)
        eq0 = eq_omega.copy()
        eq0.W_lmn = np.zeros(eq0.W_basis.num_modes)
        fig0, _, data0 = plot_section(eq0, "|B|", **kw)
        plt.close(fig)
        plt.close(fig0)
        assert np.max(np.abs(np.asarray(data["R"]) - np.asarray(data0["R"]))) > 1e-3


class TestOmegaConstraints:
    """Construction of the linear constraints that handle omega."""

    @staticmethod
    def _eq(sym=True):
        return Equilibrium(L=4, M=4, N=2, NFP=2, sym=sym, Lw=2, Mw=2, Nw=2)

    @pytest.mark.unit
    def test_fix_boundary_and_axis_W_with_explicit_modes(self):
        """Passing a mode list fixes exactly those coefficients."""
        eq = self._eq()
        # modes are full [l, m, n] triples, per the constraint docstrings
        modes = eq.surface.W_basis.modes[:3]
        con = FixBoundaryW(eq=eq, modes=modes)
        con.build()
        assert con.dim_f == len(modes)
        np.testing.assert_allclose(
            con.compute(eq.params_dict), np.asarray(eq.Wb_lmn)[:3], atol=1e-14
        )
        con_all = FixBoundaryW(eq=eq)
        con_all.build()
        assert con_all.dim_f == eq.surface.W_basis.num_modes

        # the axis omega basis must be sized from the equilibrium's Nw, or every
        # axis omega constraint below is a silent no-op
        assert eq.axis.W_basis.num_modes > 0
        con_axis = FixAxisW(eq=eq, modes=eq.axis.W_basis.modes[:2])
        con_axis.build()
        assert con_axis.dim_f == 2
        con_axis_all = FixAxisW(eq=eq)
        con_axis_all.build()
        assert con_axis_all.dim_f == eq.axis.W_basis.num_modes

    @pytest.mark.unit
    def test_fix_zeta_sfl_targets_zero_omega(self):
        """Every omega coefficient is driven to zero, recovering zeta = phi."""
        eq = self._eq()
        rng = np.random.default_rng(2)
        eq.W_lmn = 0.02 * rng.standard_normal(eq.W_basis.num_modes)
        con = FixZetaSFL(eq=eq)
        con.build()
        assert con.dim_f == eq.W_basis.num_modes
        np.testing.assert_allclose(con.target, 0)
        # the residual is the coefficients themselves, so it vanishes with omega
        np.testing.assert_allclose(
            con.compute(eq.params_dict), np.asarray(eq.W_lmn), atol=1e-14
        )
        eq.W_lmn = np.zeros(eq.W_basis.num_modes)
        np.testing.assert_allclose(con.compute(eq.params_dict), 0, atol=1e-14)

    @pytest.mark.unit
    @pytest.mark.parametrize("sym", [True, False])
    def test_fix_omega_gauge(self, sym):
        """The gauge constraint removes all n=0 omega modes, any (l, m)."""
        eq = self._eq(sym=sym)
        con = FixOmegaGauge(eq=eq)
        con.build()
        modes = eq.W_basis.modes
        expected = np.sum(modes[:, 2] == 0)
        # a sin basis still has no (m=0, n=0) mode, but can have other n=0
        # modes at m != 0, which symmetry alone does not remove
        assert expected > 0
        assert con.dim_f == expected

    @pytest.mark.unit
    def test_fix_omega_interior_keeps_one_radial_mode_per_mn(self):
        """All but the lowest radial mode per (m, n) is fixed.

        That leaves one free radial mode per (m, n) for
        BoundaryWSelfConsistency to determine from the boundary.
        """
        eq = self._eq()
        con = FixOmegaInterior(eq=eq)
        con.build()
        n_mn = len(np.unique(eq.W_basis.modes[:, 1:], axis=0))
        assert con.dim_f == eq.W_basis.num_modes - n_mn
        # an omega-free equilibrium gets an empty constraint, not an error
        con0 = FixOmegaInterior(eq=Equilibrium(L=4, M=4, N=2, sym=True))
        con0.build()
        assert con0.dim_f == 0

    @pytest.mark.unit
    def test_axis_W_self_consistency_matrix(self):
        """The axis constraint evaluates the omega basis at rho = 0.

        Every m != 0 mode vanishes there, and an m = 0 Zernike radial
        polynomial of degree l evaluates to (-1)^(l/2).
        """
        eq = self._eq()
        con = AxisWSelfConsistency(eq=eq)
        con.build()
        A = con._A
        assert A.shape == (eq.axis.W_basis.num_modes, eq.W_basis.num_modes)
        for i, (l, m, n) in enumerate(eq.W_basis.modes):
            if m != 0:
                np.testing.assert_allclose(A[:, i], 0, err_msg=f"mode {(l, m, n)}")
            else:
                j = np.argwhere(n == eq.axis.W_basis.modes[:, 2])
                np.testing.assert_allclose(A[j, i], (-1) ** (l // 2))
        np.testing.assert_allclose(con.compute(eq.params_dict), 0, atol=1e-14)


class TestOmegaSmallBranches:
    """Guards and reporting paths that only omega workflows reach."""

    @pytest.mark.unit
    def test_setters_reject_wrong_length(self):
        """W_n and W_lmn must match their basis size."""
        curve = FourierRZCurve(
            R_n=np.array([10.0, 1.0]),
            Z_n=np.array([-1.0]),
            modes_R=np.array([0, 1]),
            modes_Z=np.array([-1]),
            W_n=np.array([0.1]),
            modes_W=np.array([1]),
            sym=False,
        )
        # asymmetric, so the basis spans n = -1, 0, 1 rather than just n = 1
        assert curve.W_basis.num_modes == 3
        with pytest.raises(ValueError, match="W_n should have the same size"):
            curve.W_n = np.array([0.1, 0.2])
        surf = _make_synthetic_surface()
        with pytest.raises(ValueError, match="W_lmn should have the same size"):
            surf.W_lmn = np.append(np.asarray(surf.W_lmn), 0.0)

    @pytest.mark.unit
    def test_check_toroidal_map_without_omega(self):
        """With no omega basis the map is the identity, so phi_zeta == 1."""
        surf = FourierRZToroidalSurface()
        assert surf.W_basis.num_modes == 0
        # exactly 1.0: returned directly, never sampled on a grid
        assert surf.check_toroidal_map() == 1.0

    @pytest.mark.unit
    def test_a_of_z_is_sqrt_area_over_pi(self):
        """a(z) is the effective minor radius of the constant zeta section."""
        surf = _make_elongated_surface(NFP=1, omega=True)
        grid = LinearGrid(theta=64, zeta=6, NFP=1, sym=False)
        data = surf.compute(["a(z)", "A(z)"], grid=grid)
        np.testing.assert_allclose(
            data["a(z)"], np.sqrt(np.asarray(data["A(z)"]) / np.pi), rtol=1e-14
        )
        assert np.all(np.asarray(data["a(z)"]) > 0)

    @pytest.mark.unit
    def test_fixed_axis_constraints_include_omega_interior(self):
        """Interior omega is held fixed only when the equilibrium has omega."""
        from desc.objectives import get_fixed_axis_constraints

        eq = Equilibrium(L=4, M=4, N=2, NFP=2, Lw=2, Mw=2, Nw=2)
        names = [type(con).__name__ for con in get_fixed_axis_constraints(eq)]
        assert "FixOmegaInterior" in names and "FixAxisW" in names
        eq0 = Equilibrium(L=4, M=4, N=2, NFP=2)
        names0 = [type(con).__name__ for con in get_fixed_axis_constraints(eq0)]
        assert "FixOmegaInterior" not in names0

    @pytest.mark.unit
    def test_surface_setter_grows_omega_resolution(self):
        """A richer boundary grows the equilibrium basis instead of truncating."""
        eq = Equilibrium(L=4, M=4, N=2, NFP=2, sym=True, Lw=2, Mw=1, Nw=1)
        surf = FourierRZToroidalSurface(
            R_lmn=np.array([10.0, 1.0]),
            Z_lmn=np.array([-1.0]),
            modes_R=np.array([[0, 0], [1, 0]]),
            modes_Z=np.array([[-1, 0]]),
            NFP=2,
            sym=True,
            Mw=3,
            Nw=2,
        )
        # mark the highest mode of the richer basis so the resize can be traced
        mode = surf.W_basis.modes[-1]
        W = np.asarray(surf.W_lmn).copy()
        W[-1] = 0.05
        surf.W_lmn = W

        eq.surface = surf
        assert (eq.Mw, eq.Nw) == (3, 2)
        assert eq.Lw == 3  # ansi indexing ties radial degree to poloidal
        idx = eq.surface.W_basis.get_idx(*mode)
        np.testing.assert_allclose(np.asarray(eq.surface.W_lmn)[idx], 0.05)

    @pytest.mark.unit
    def test_resolution_summary_reports_omega(self, capsys):
        """The omega resolution line appears only when omega modes exist."""
        Equilibrium(L=4, M=4, N=2, NFP=2, Lw=2, Mw=2, Nw=2).resolution_summary()
        assert "Omega spectral resolution (Lw,Mw,Nw)=(2,2,2)" in capsys.readouterr().out
        Equilibrium(L=4, M=4, N=2, NFP=2).resolution_summary()
        assert "Omega spectral resolution" not in capsys.readouterr().out

    @pytest.mark.unit
    def test_get_deltas_picks_up_omega(self):
        """Perturbation deltas include the boundary and axis omega changes."""
        from desc.perturbations import get_deltas

        s1 = _make_synthetic_surface()
        s2 = _make_synthetic_surface()
        W = np.asarray(s2.W_lmn).copy()
        W[0] += 0.01
        s2.W_lmn = W
        deltas = get_deltas({"surface": s1}, {"surface": s2})
        np.testing.assert_allclose(deltas["Wb_lmn"], np.asarray(s2.W_lmn) - s1.W_lmn)

        eq = Equilibrium(L=4, M=4, N=2, NFP=2, sym=True, Lw=2, Mw=2, Nw=2)
        a1, a2 = eq.axis, eq.axis.copy()
        assert a2.W_n.size > 0
        a2.W_n = np.asarray(a2.W_n) + 0.01
        deltas = get_deltas({"axis": a1}, {"axis": a2})
        np.testing.assert_allclose(deltas["Wa_n"], np.asarray(a2.W_n) - a1.W_n)

    @pytest.mark.unit
    def test_perturb_boundary_omega(self):
        """A boundary omega delta perturbs the interior omega with it."""
        from desc.perturbations import get_deltas

        eq = Equilibrium(L=3, M=3, N=1, NFP=2, sym=True, Lw=2, Mw=1, Nw=1)
        s2 = eq.surface.copy()
        W = np.asarray(s2.W_lmn).copy()
        W[0] += 0.01
        s2.W_lmn = W
        deltas = get_deltas({"surface": eq.surface}, {"surface": s2})
        assert list(deltas) == ["Wb_lmn"]
        eq2 = eq.perturb(deltas=deltas, order=1, verbose=0, copy=True)
        # the boundary moved, and BoundaryWSelfConsistency carried it inward
        np.testing.assert_allclose(np.asarray(eq2.Wb_lmn), np.asarray(s2.W_lmn))
        assert np.max(np.abs(np.asarray(eq2.W_lmn) - np.asarray(eq.W_lmn))) > 1e-3


class TestContinuationWithOmega:
    """Continuation on a boundary carrying a generalized toroidal angle."""

    @pytest.mark.unit
    def test_boundary_omega_is_not_doubled(self):
        """Continuation converges to the requested boundary omega.

        The axisymmetric stage has to start omega-free. If it keeps the full
        boundary omega, _add_shaping ramps omega from zero on top of it and the
        solve lands on twice the requested value with no error raised.
        """
        from desc.continuation import solve_continuation_automatic

        surf = FourierRZToroidalSurface(
            R_lmn=np.array([10.0, 1.0, 0.1]),
            Z_lmn=np.array([-1.0, -0.1]),
            modes_R=np.array([[0, 0], [1, 0], [1, 1]]),
            modes_Z=np.array([[-1, 0], [-1, 1]]),
            NFP=3,
            sym=True,
            W_lmn=np.array([0.3]),
            modes_W=np.array([[0, -1]]),
        )
        # omega = 0.3 sin(3 zeta) leaves min(1 + omega_zeta) = 0.126, so
        # doubling it would invert the chart outright rather than merely
        # overshoot. It also converges without backtracking, unlike small omega.
        eq = Equilibrium(L=4, M=4, N=1, NFP=3, sym=True, surface=surf, Nw=1)
        fam = solve_continuation_automatic(eq.copy(), verbose=0, maxiter=10, ftol=1e-3)
        final = fam[-1]
        # the boundary is fixed by the constraints, so this is exact
        np.testing.assert_allclose(
            np.asarray(final.Wb_lmn), np.asarray(eq.Wb_lmn), rtol=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(final.Rb_lmn)[final.surface.R_basis.get_idx(M=1, N=1)],
            0.1,
            rtol=1e-8,
        )
        assert final.is_nested()
        assert final.get_surface_at(rho=1.0).check_toroidal_map() > 0
