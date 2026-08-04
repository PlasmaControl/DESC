"""Tests for the generalized toroidal angle (omega / W_lmn).

The computational toroidal coordinate zeta no longer has to equal the
cylindrical laboratory angle phi. Instead phi = zeta + omega(rho,theta,zeta)
with omega a periodic toroidal stream function (spectral coefficients W_lmn).

Phase 1 tests cover the generalized FourierRZToroidalSurface (state,
serialization, fitting from sampled coordinates, geometry).
Phase 2 tests cover the generalized Equilibrium (compute graph, coordinate
invariance, solves).

All tests here are self contained: they build analytic surfaces and small
equilibria in-process and need no external data files. Data-driven tests
against sampled PEST/Boozer surface data live outside the repository, in
~/generalized_toroidal/tests/test_generalized_toroidal_data.py.
"""

import os

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZCurve, FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid


def _wrap(angle):
    """Map angle difference into (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _make_synthetic_surface(NFP=1):
    """Analytic surface with nonzero omega for ground truth.

    R = 10 + cos(theta) + 0.05 cos(theta - NFP*zeta)
    Z = -sin(theta) - 0.05 sin(theta - NFP*zeta)
    omega = 0.10 sin(NFP*zeta) + 0.05 sin(theta - NFP*zeta)

    All terms respect stellarator symmetry (R even, Z and omega odd). The
    signs of Z are chosen to give a right handed (theta, zeta) system, as
    DESC requires, matching the default FourierRZToroidalSurface.
    """
    surf = FourierRZToroidalSurface(
        R_lmn=np.array([10.0, 1.0, 0.05]),
        Z_lmn=np.array([-1.0, -0.05]),
        modes_R=np.array([[0, 0], [1, 0], [1, 1]]),
        modes_Z=np.array([[-1, 0], [-1, 1]]),
        NFP=NFP,
        sym=True,
        W_lmn=np.array([0.10, 0.05]),
        modes_W=np.array([[0, -1], [-1, 1]]),
    )
    return surf


def _sample_surface(surf, ntheta=40, nzeta=41):
    """Sample physical (R, phi, Z) from a surface with omega."""
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / surf.NFP, nzeta, endpoint=False)
    T, ZE = map(np.ravel, np.meshgrid(theta, zeta, indexing="ij"))
    grid = Grid(np.vstack([np.ones_like(T), T, ZE]).T, sort=False)
    data = surf.compute(["R", "Z", "omega", "phi"], grid=grid)
    coords = np.array([data["R"], data["phi"], data["Z"]]).T
    return T, ZE, coords, data


class TestSurfaceOmegaState:
    """State, serialization, and resolution handling of the generalized surface."""

    @pytest.mark.unit
    def test_default_surface_has_zero_omega(self):
        """Default surfaces must have exactly zero omega and no W modes."""
        surf = FourierRZToroidalSurface()
        assert surf.W_basis.num_modes == 0
        assert surf.W_lmn.size == 0
        assert surf.Mz == 0 and surf.Nz == 0
        data = surf.compute(["omega", "phi"], grid=LinearGrid(M=4, N=4))
        np.testing.assert_allclose(data["omega"], 0)

    @pytest.mark.unit
    def test_omega_sym_parity(self):
        """Omega is odd under stellarator symmetry (sin basis, like Z)."""
        surf = _make_synthetic_surface()
        assert surf.sym
        assert surf.W_basis.sym == "sin"
        # evaluate omega at (theta, zeta) and (-theta, -zeta): must be odd
        grid1 = LinearGrid(M=6, N=6, NFP=surf.NFP)
        t, z = grid1.nodes[:, 1], grid1.nodes[:, 2]
        grid2 = Grid(np.vstack([np.ones_like(t), -t, -z]).T, sort=False)
        w1 = surf.compute("omega", grid=grid1)["omega"]
        w2 = surf.compute("omega", grid=grid2)["omega"]
        np.testing.assert_allclose(w1, -w2, atol=1e-14)

    @pytest.mark.unit
    def test_change_resolution_preserves_omega(self):
        """Changing (M, N, Mz, Nz) must preserve existing coefficients."""
        surf = _make_synthetic_surface()
        w_old = surf.compute("omega", grid=LinearGrid(M=4, N=4))["omega"]
        surf.change_resolution(M=5, N=5, Mz=4, Nz=4)
        assert surf.Mz == 4 and surf.Nz == 4
        w_new = surf.compute("omega", grid=LinearGrid(M=4, N=4))["omega"]
        np.testing.assert_allclose(w_old, w_new, atol=1e-14)
        # shrinking back to the original resolution also preserves the modes
        surf.change_resolution(M=5, N=5, Mz=1, Nz=1)
        w_new = surf.compute("omega", grid=LinearGrid(M=4, N=4))["omega"]
        np.testing.assert_allclose(w_old, w_new, atol=1e-14)
        # changing only M, N must not touch omega
        surf.change_resolution(M=8, N=8)
        assert surf.Mz == 1 and surf.Nz == 1
        w_new = surf.compute("omega", grid=LinearGrid(M=4, N=4))["omega"]
        np.testing.assert_allclose(w_old, w_new, atol=1e-14)

    @pytest.mark.unit
    def test_serialization_roundtrip(self, tmpdir):
        """Save/load must preserve omega; old files load with zero omega."""
        surf = _make_synthetic_surface()
        path = os.path.join(tmpdir, "surf.h5")
        surf.save(path)
        from desc.io import load

        surf2 = load(path)
        np.testing.assert_allclose(
            np.asarray(surf2.W_lmn), np.asarray(surf.W_lmn), atol=1e-14
        )
        assert surf2.W_basis.equiv(surf.W_basis)
        # simulate an old file: delete the W attributes and rerun _set_up
        surf3 = surf.copy()
        del surf3._W_lmn
        del surf3._W_basis
        surf3._set_up()
        assert surf3.W_basis.num_modes == 0
        assert surf3.W_lmn.size == 0

    @pytest.mark.unit
    def test_optimizable_params_include_W(self):
        """W_lmn is part of the surface parameter dictionary."""
        surf = _make_synthetic_surface()
        assert "W_lmn" in surf.params_dict
        np.testing.assert_allclose(
            np.asarray(surf.params_dict["W_lmn"]), np.asarray(surf.W_lmn)
        )

    @pytest.mark.unit
    def test_flip_orientation_flips_omega(self):
        """Flipping theta orientation must flip m<0 omega modes."""
        surf = _make_synthetic_surface()
        w0 = surf.W_lmn.copy()
        surf._flip_orientation()
        m = surf.W_basis.modes[:, 1]
        expected = np.where(m < 0, -np.asarray(w0), np.asarray(w0))
        np.testing.assert_allclose(np.asarray(surf.W_lmn), expected)


class TestSurfaceFitting:
    """Fitting a generalized surface from sampled physical coordinates."""

    @pytest.mark.unit
    @pytest.mark.parametrize("NFP", [1, 3])
    def test_fit_recovers_synthetic_surface(self, NFP):
        """Fit sampled points of an analytic omega surface; recover everything."""
        truth = _make_synthetic_surface(NFP)
        T, ZE, coords, _ = _sample_surface(truth)
        surf = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=NFP, sym=True
        )
        # evaluate both surfaces on a *different* grid
        tt = np.linspace(0.1, 2 * np.pi, 27, endpoint=False)
        zz = np.linspace(0.05, 2 * np.pi / NFP, 25, endpoint=False)
        T2, Z2 = map(np.ravel, np.meshgrid(tt, zz, indexing="ij"))
        g2 = Grid(np.vstack([np.ones_like(T2), T2, Z2]).T, sort=False)
        keys = ["R", "Z", "omega", "phi", "x", "e_theta", "e_zeta", "n_rho"]
        dt = truth.compute(keys, grid=g2, basis="xyz")
        df = surf.compute(keys, grid=g2, basis="xyz")
        for key in ["R", "Z", "omega", "phi"]:
            np.testing.assert_allclose(df[key], dt[key], atol=1e-10, err_msg=key)
        # Cartesian position, tangent and normal vectors
        for key in ["x", "e_theta", "e_zeta", "n_rho"]:
            np.testing.assert_allclose(df[key], dt[key], atol=1e-9, err_msg=key)

    @pytest.mark.unit
    def test_fit_derivatives_vs_finite_differences(self):
        """First derivatives of the fitted map agree with finite differences."""
        truth = _make_synthetic_surface()
        T, ZE, coords, _ = _sample_surface(truth)
        surf = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True
        )
        t0 = np.array([0.7, 1.9, 4.1])
        z0 = np.array([0.3, 2.2, 5.0])
        eps = 1e-6

        def evalx(t, z):
            g = Grid(np.vstack([np.ones_like(t), t, z]).T, sort=False)
            return surf.compute("x", grid=g, basis="xyz")["x"]

        g0 = Grid(np.vstack([np.ones_like(t0), t0, z0]).T, sort=False)
        d = surf.compute(["e_theta", "e_zeta", "omega_t", "omega_z", "omega"], grid=g0)
        d0 = surf.compute(["e_theta", "e_zeta"], grid=g0, basis="xyz")
        fd_et = (evalx(t0 + eps, z0) - evalx(t0 - eps, z0)) / (2 * eps)
        fd_ez = (evalx(t0, z0 + eps) - evalx(t0, z0 - eps)) / (2 * eps)
        np.testing.assert_allclose(d0["e_theta"], fd_et, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(d0["e_zeta"], fd_ez, rtol=1e-5, atol=1e-6)

        # omega derivatives vs finite differences of omega itself
        def evalw(t, z):
            g = Grid(np.vstack([np.ones_like(t), t, z]).T, sort=False)
            return surf.compute("omega", grid=g)["omega"]

        fd_wt = (evalw(t0 + eps, z0) - evalw(t0 - eps, z0)) / (2 * eps)
        fd_wz = (evalw(t0, z0 + eps) - evalw(t0, z0 - eps)) / (2 * eps)
        np.testing.assert_allclose(d["omega_t"], fd_wt, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(d["omega_z"], fd_wz, rtol=1e-6, atol=1e-8)

    @pytest.mark.unit
    def test_fit_surface_area_and_orientation(self):
        """Fitted surface reproduces area and right-handed orientation."""
        truth = _make_synthetic_surface()
        T, ZE, coords, _ = _sample_surface(truth)
        surf = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True
        )
        grid = LinearGrid(M=24, N=24, NFP=1)
        at = truth.compute("S", grid=grid)["S"]
        af = surf.compute("S", grid=grid)["S"]
        np.testing.assert_allclose(af, at, rtol=1e-8)
        assert surf._compute_orientation() == truth._compute_orientation() == 1

    @pytest.mark.unit
    def test_fit_across_branch_cuts(self):
        """Wrapping phi into arbitrary 2*pi branches must not change the fit."""
        truth = _make_synthetic_surface()
        T, ZE, coords, _ = _sample_surface(truth)
        rng = np.random.default_rng(0)
        shifts = 2 * np.pi * rng.integers(-3, 4, size=coords.shape[0])
        coords_wrapped = coords.copy()
        coords_wrapped[:, 1] = coords[:, 1] + shifts
        surf1 = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True
        )
        surf2 = FourierRZToroidalSurface.from_values(
            coords_wrapped, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True
        )
        np.testing.assert_allclose(
            np.asarray(surf1.W_lmn), np.asarray(surf2.W_lmn), atol=1e-10
        )
        # also wrap zeta into another branch: same surface, since omega is
        # computed from the periodic difference
        surf3 = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE + 2 * np.pi, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True
        )
        np.testing.assert_allclose(
            np.asarray(surf1.W_lmn), np.asarray(surf3.W_lmn), atol=1e-10
        )

    @pytest.mark.unit
    def test_fit_zeta_equals_phi_gives_zero_omega(self):
        """Supplying zeta = phi recovers omega = 0 exactly."""
        truth = _make_synthetic_surface()
        T, ZE, coords, _ = _sample_surface(truth)
        # parameterize by the physical angle itself
        surf = FourierRZToroidalSurface.from_values(
            coords, T, zeta=coords[:, 1], M=6, N=6, Mz=3, Nz=3, NFP=1, sym=True
        )
        np.testing.assert_allclose(np.asarray(surf.W_lmn), 0, atol=1e-12)
        # and omitting zeta entirely gives a surface with no omega modes
        surf2 = FourierRZToroidalSurface.from_values(coords, T, M=6, N=6, sym=True)
        assert surf2.W_basis.num_modes == 0

    @pytest.mark.unit
    def test_fit_xyz_basis(self):
        """Cartesian (X, Y, Z) input gives the same fit as cylindrical."""
        truth = _make_synthetic_surface()
        T, ZE, coords, _ = _sample_surface(truth)
        X = coords[:, 0] * np.cos(coords[:, 1])
        Y = coords[:, 0] * np.sin(coords[:, 1])
        xyz = np.array([X, Y, coords[:, 2]]).T
        surf_rpz = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True
        )
        surf_xyz = FourierRZToroidalSurface.from_values(
            xyz, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True, basis="xyz"
        )
        for attr in ["R_lmn", "Z_lmn", "W_lmn"]:
            np.testing.assert_allclose(
                np.asarray(getattr(surf_xyz, attr)),
                np.asarray(getattr(surf_rpz, attr)),
                atol=1e-10,
                err_msg=attr,
            )

    @pytest.mark.unit
    def test_weighted_fit(self):
        """Weighted fit with uniform weights equals the unweighted fit."""
        truth = _make_synthetic_surface()
        T, ZE, coords, _ = _sample_surface(truth)
        surf_u = FourierRZToroidalSurface.from_values(
            coords, T, zeta=ZE, M=4, N=4, Mz=2, Nz=2, NFP=1, sym=True
        )
        surf_w = FourierRZToroidalSurface.from_values(
            coords,
            T,
            zeta=ZE,
            M=4,
            N=4,
            Mz=2,
            Nz=2,
            NFP=1,
            sym=True,
            w=np.ones(coords.shape[0]),
        )
        for attr in ["R_lmn", "Z_lmn", "W_lmn"]:
            np.testing.assert_allclose(
                np.asarray(getattr(surf_w, attr)),
                np.asarray(getattr(surf_u, attr)),
                atol=1e-9,
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
        T, ZE, coords, _ = _sample_surface(bad)
        with pytest.raises(ValueError, match="not a valid toroidal"):
            FourierRZToroidalSurface.from_values(
                coords, T, zeta=ZE, M=4, N=4, Mz=1, Nz=1, NFP=1, sym=True
            )
        # a healthy map passes and returns min(1 + omega_zeta)
        good = _make_synthetic_surface()
        assert good.check_toroidal_map() > 0.5


class TestEquilibriumOmega:
    """Phase 2: generalized equilibrium coordinates."""

    @pytest.mark.unit
    def test_zero_omega_regression(self):
        """An eq with zero-valued omega modes matches one with no modes."""
        eq0 = Equilibrium(L=4, M=4, N=2, NFP=3, sym=True)
        eq1 = Equilibrium(L=4, M=4, N=2, NFP=3, sym=True, Lz=2, Mz=2, Nz=2)
        assert eq0.W_basis.num_modes == 0
        assert eq1.W_basis.num_modes > 0
        np.testing.assert_allclose(np.asarray(eq1.W_lmn), 0)
        grid = LinearGrid(L=4, M=8, N=8, NFP=3)
        keys = ["R", "Z", "phi", "|B|", "sqrt(g)", "|F|", "g_tt", "g_zz", "B_zeta"]
        d0 = eq0.compute(keys, grid=grid)
        d1 = eq1.compute(keys, grid=grid)
        for key in keys:
            np.testing.assert_allclose(d0[key], d1[key], atol=1e-14, err_msg=key)

    @pytest.mark.unit
    def test_old_equilibrium_loads(self):
        """Equilibria saved before omega existed load with omega = 0."""
        import desc.examples

        eq = desc.examples.get("DSHAPE")
        assert eq.W_basis.num_modes == 0
        assert eq.W_lmn.size == 0
        assert eq.Lz == eq.Mz == eq.Nz == 0
        data = eq.compute(["omega", "phi", "|B|"], grid=LinearGrid(L=2, M=4, N=0))
        np.testing.assert_allclose(data["omega"], 0)
        assert np.all(np.isfinite(data["|B|"]))

    @pytest.mark.unit
    def test_phi_derivative_identities(self):
        """phi_r = omega_r, phi_t = omega_t, phi_z = 1 + omega_z, and higher."""
        eq = Equilibrium(L=4, M=4, N=2, NFP=2, sym=True, Lz=2, Mz=2, Nz=2)
        rng = np.random.default_rng(3)
        eq.W_lmn = 0.02 * rng.standard_normal(eq.W_basis.num_modes)
        grid = LinearGrid(L=3, M=6, N=6, NFP=2)
        keys = [
            "phi",
            "zeta",
            "omega",
            "phi_r",
            "omega_r",
            "phi_t",
            "omega_t",
            "phi_z",
            "omega_z",
            "phi_rr",
            "omega_rr",
            "phi_tt",
            "omega_tt",
            "phi_zz",
            "omega_zz",
            "phi_rt",
            "omega_rt",
            "phi_rz",
            "omega_rz",
            "phi_tz",
            "omega_tz",
        ]
        d = eq.compute(keys, grid=grid)
        np.testing.assert_allclose(d["phi"], d["zeta"] + d["omega"], atol=1e-14)
        np.testing.assert_allclose(d["phi_r"], d["omega_r"], atol=1e-14)
        np.testing.assert_allclose(d["phi_t"], d["omega_t"], atol=1e-14)
        np.testing.assert_allclose(d["phi_z"], 1 + d["omega_z"], atol=1e-14)
        for a, b in [
            ("phi_rr", "omega_rr"),
            ("phi_tt", "omega_tt"),
            ("phi_zz", "omega_zz"),
            ("phi_rt", "omega_rt"),
            ("phi_rz", "omega_rz"),
            ("phi_tz", "omega_tz"),
        ]:
            np.testing.assert_allclose(d[a], d[b], atol=1e-14, err_msg=a)
        assert np.max(np.abs(d["omega"])) > 0  # actually nonzero

    @pytest.mark.unit
    def test_omega_derivatives_vs_finite_differences(self):
        """Transform-based omega derivatives agree with finite differences."""
        eq = Equilibrium(L=4, M=4, N=2, NFP=2, sym=True, Lz=2, Mz=2, Nz=2)
        rng = np.random.default_rng(5)
        eq.W_lmn = 0.02 * rng.standard_normal(eq.W_basis.num_modes)
        rho = np.array([0.5, 0.7, 0.9])
        theta = np.array([0.4, 2.0, 5.1])
        zeta = np.array([0.2, 1.1, 2.8])
        eps = 1e-6

        def w(r, t, z):
            g = Grid(np.vstack([r, t, z]).T, sort=False)
            return eq.compute("omega", grid=g)["omega"]

        g0 = Grid(np.vstack([rho, theta, zeta]).T, sort=False)
        d = eq.compute(["omega_r", "omega_t", "omega_z"], grid=g0)
        np.testing.assert_allclose(
            d["omega_r"],
            (w(rho + eps, theta, zeta) - w(rho - eps, theta, zeta)) / (2 * eps),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            d["omega_t"],
            (w(rho, theta + eps, zeta) - w(rho, theta - eps, zeta)) / (2 * eps),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            d["omega_z"],
            (w(rho, theta, zeta + eps) - w(rho, theta, zeta - eps)) / (2 * eps),
            rtol=1e-5,
            atol=1e-8,
        )

    @pytest.mark.unit
    def test_compute_everything_nonzero_omega(self):
        """All standard rtz quantities are finite with nonzero omega."""
        from desc.compute import data_index, get_data_deps

        eq = Equilibrium(L=4, M=4, N=2, NFP=3, sym=True, Lz=2, Mz=2, Nz=2)
        rng = np.random.default_rng(7)
        eq.W_lmn = 0.01 * rng.standard_normal(eq.W_basis.num_modes)
        assert eq.is_nested()
        # off axis: contravariant quantities such as e^theta ~ 1/sqrt(g) are
        # genuinely singular at rho=0 regardless of omega
        grid = LinearGrid(rho=np.linspace(0.2, 1.0, 5), M=8, N=8, NFP=3)
        p = "desc.equilibrium.equilibrium.Equilibrium"

        def _plain(name):
            """Quantity computable on a plain rtz grid, with all its deps."""
            entry = data_index[p][name]
            if entry["coordinates"] != "rtz":
                return False
            # a quantity is only computable here if neither it nor anything it
            # depends on needs a special grid (field line source grids,
            # Boozer resolution, flux surface integration, ...)
            for dep in [name] + get_data_deps(name, p):
                d = data_index[p][dep]
                if (
                    d["source_grid_requirement"]
                    or d["grid_requirement"]
                    or d["resolution_requirement"]
                ):
                    return False
            return True

        names = [name for name in data_index[p] if _plain(name)]
        assert len(names) > 200, f"expected to sweep many quantities, got {len(names)}"

        data = eq.compute(names, grid=grid)
        bad = [n for n in names if not np.all(np.isfinite(np.asarray(data[n])))]

        # the same equilibrium with omega set back to zero: any quantity that
        # is non-finite there is non-finite for reasons unrelated to omega, so
        # requiring the two sets to agree isolates omega as the cause
        eq0 = eq.copy()
        eq0.W_lmn = np.zeros(eq0.W_basis.num_modes)
        data0 = eq0.compute(names, grid=grid)
        bad0 = [n for n in names if not np.all(np.isfinite(np.asarray(data0[n])))]

        # A few quantities are non-finite by design regardless of omega, e.g.
        # beta_a and its derivatives are set to NaN when the equilibrium has no
        # anisotropy profile assigned. What matters is that omega introduces no
        # new ones.
        assert set(bad) == set(bad0), (
            "omega changed which quantities are finite. Non-finite only with "
            f"omega != 0: {sorted(set(bad) - set(bad0))}; only with omega == 0: "
            f"{sorted(set(bad0) - set(bad))}"
        )
        assert set(bad0) <= {
            "beta_a",
            "beta_a_r",
            "beta_a_t",
            "beta_a_z",
            "grad(beta_a)",
        }, f"unexpected NaN baseline: {bad0}"
        # and omega actually changed the answers, so this was a real test
        assert not np.allclose(data["sqrt(g)"], data0["sqrt(g)"])
        # coordinate jacobian stays positive
        assert np.all(data["sqrt(g)"] > 0)

    @pytest.mark.unit
    def test_axisymmetric_coordinate_invariance(self):
        """Pure-zeta omega reparameterizes an axisymmetric equilibrium.

        For an axisymmetric equilibrium (R, Z, lambda independent of zeta),
        the map zeta -> phi = zeta + omega(zeta) sweeps out the identical
        physical field, with unchanged R_lmn, Z_lmn, L_lmn. All coordinate
        invariants must match; |B|, pressure etc. depend only on (rho, theta)
        so they may be compared at identical computational nodes.
        """
        eq0 = Equilibrium(L=4, M=4, N=0, sym=True)
        eq0.solve(verbose=0, maxiter=25, ftol=1e-6)

        eq1 = Equilibrium(L=4, M=4, N=2, sym=True, Lz=0, Mz=0, Nz=2)
        # copy the solved axisymmetric state (axisym modes only)
        from desc.utils import copy_coeffs

        eq1.R_lmn = copy_coeffs(eq0.R_lmn, eq0.R_basis.modes, eq1.R_basis.modes)
        eq1.Z_lmn = copy_coeffs(eq0.Z_lmn, eq0.Z_basis.modes, eq1.Z_basis.modes)
        eq1.L_lmn = copy_coeffs(eq0.L_lmn, eq0.L_basis.modes, eq1.L_basis.modes)
        eq1.pressure.params = eq0.pressure.params.copy()
        eq1.current.params = eq0.current.params.copy()
        # omega = 0.1 sin(zeta) + 0.03 sin(2 zeta), a pure toroidal
        # reparameterization (no rho or theta dependence)
        W = np.zeros(eq1.W_basis.num_modes)
        W[eq1.W_basis.get_idx(0, 0, -1)] = 0.1
        W[eq1.W_basis.get_idx(0, 0, -2)] = 0.03
        eq1.W_lmn = W
        eq1.surface = eq1.get_surface_at(rho=1.0)
        eq1.axis = eq1.get_axis()

        grid = LinearGrid(L=6, M=10, N=8)
        keys = ["|B|", "|F|", "p", "iota", "sqrt(g)", "V", "S"]
        d0 = eq0.compute(keys, grid=LinearGrid(L=6, M=10, N=8))
        d1 = eq1.compute(keys, grid=grid)
        # |B|, |F|, p, iota depend only on (rho, theta) for axisymmetric
        # physics: identical at identical (rho, theta) nodes
        np.testing.assert_allclose(d1["|B|"], d0["|B|"], rtol=1e-10)
        # |F| is a residual of large, nearly cancelling terms (grad(p) against
        # the J x B force), so it carries far fewer significant digits than the
        # quantities it is built from. The two charts evaluate those terms in a
        # different order, which shows up as roundoff at the 1e-7 relative
        # level; |B| above still matches to 1e-10.
        np.testing.assert_allclose(d1["|F|"], d0["|F|"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(d1["p"], d0["p"], rtol=1e-12)
        np.testing.assert_allclose(d1["iota"], d0["iota"], rtol=1e-12, atol=1e-12)
        # global invariants
        np.testing.assert_allclose(d1["V"], d0["V"], rtol=1e-10)
        np.testing.assert_allclose(d1["S"], d0["S"], rtol=1e-10)
        # jacobian is *not* invariant (it scales by 1 + omega_zeta): sanity
        # check that this test would catch an actual omega (chart) effect
        assert not np.allclose(d1["sqrt(g)"], d0["sqrt(g)"], rtol=1e-3)
        # Cartesian positions: same physical torus. x(rho,theta,zeta') of eq1
        # equals x(rho,theta,zeta) of eq0 evaluated at zeta = phi(zeta')
        nodes = grid.nodes.copy()
        d1x = eq1.compute(["x", "phi"], grid=Grid(nodes, sort=False), basis="xyz")
        nodes0 = nodes.copy()
        nodes0[:, 2] = d1x["phi"]
        d0x = eq0.compute("x", grid=Grid(nodes0, sort=False), basis="xyz")
        np.testing.assert_allclose(d1x["x"], d0x["x"], atol=1e-10)
        # magnetic axis position: same circle (compare the n=0 coefficient,
        # the two axis curves have different numbers of modes)
        ax0, ax1 = eq0.get_axis(), eq1.get_axis()
        np.testing.assert_allclose(
            np.asarray(ax1.R_n)[ax1.R_basis.get_idx(N=0)],
            np.asarray(ax0.R_n)[ax0.R_basis.get_idx(N=0)],
            rtol=1e-10,
        )
        assert eq1.is_nested()

    @pytest.mark.unit
    def test_solve_with_omega_fixed(self):
        """A solve with nonzero omega matches the standard solve.

        The same physical boundary is described with a generalized toroidal
        angle, so the solved equilibrium must agree physically with the
        conventional one.
        """
        # standard axisymmetric solve
        eq0 = Equilibrium(L=4, M=4, N=0, sym=True)
        eq0.solve(verbose=0, maxiter=30, ftol=1e-6)

        # same physical boundary, but described with a generalized toroidal
        # angle: omega_b = 0.1 sin(zeta) on the boundary
        surf = FourierRZToroidalSurface(
            R_lmn=[10, 1],
            Z_lmn=[0, -1],
            modes_R=[[0, 0], [1, 0]],
            modes_Z=[[0, 0], [-1, 0]],
            sym=True,
            W_lmn=np.array([0.1]),
            modes_W=np.array([[0, -1]]),
        )
        eq1 = Equilibrium(L=4, M=4, N=2, sym=True, surface=surf, Lz=0, Mz=0, Nz=1)
        assert np.max(np.abs(np.asarray(eq1.W_lmn))) > 0  # initial guess has omega
        eq1.solve(verbose=0, maxiter=30, ftol=1e-6)

        # omega is preserved by the solve (fixed by the boundary constraints)
        g = LinearGrid(L=4, M=8, N=8)
        d1 = eq1.compute(["omega", "|B|", "|F|", "V"], grid=g)
        assert np.max(np.abs(d1["omega"])) > 0.05
        d0 = eq0.compute(["|B|", "|F|", "V"], grid=LinearGrid(L=4, M=8, N=8))
        # same physical solution: volume and field strength agree
        np.testing.assert_allclose(d1["V"], d0["V"], rtol=1e-4)
        np.testing.assert_allclose(d1["|B|"], d0["|B|"], rtol=2e-3)
        # both solved to comparable force balance
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
        grid = Grid(np.vstack([np.zeros_like(s), np.zeros_like(s), s]).T, sort=False)
        d = curve.compute(["x", "x_s", "x_ss", "x_sss"], grid=grid, basis="xyz")
        # R_n/Z_n map to modes [-1, 0, 1] = [sin(s), 1, cos(s)], so
        # R = 10 + cos(s), Z = -cos(s); phi = s + 0.2 sin(s)
        R = 10 + 1 * np.cos(s)
        Z = -1 * np.cos(s)
        phi = s + 0.2 * np.sin(s)
        x_true = np.array([R * np.cos(phi), R * np.sin(phi), Z]).T
        np.testing.assert_allclose(d["x"], x_true, atol=1e-12)
        # derivatives vs finite differences
        eps = 1e-6

        def evalx(sv):
            g = Grid(
                np.vstack([np.zeros_like(sv), np.zeros_like(sv), sv]).T, sort=False
            )
            return curve.compute("x", grid=g, basis="xyz")["x"]

        fd1 = (evalx(s + eps) - evalx(s - eps)) / (2 * eps)
        np.testing.assert_allclose(d["x_s"], fd1, rtol=1e-5, atol=1e-6)
        # a second difference divides by eps**2, so it needs a much larger step
        # than the first derivative to stay above roundoff: at eps=1e-6 the
        # roundoff floor alone is ~1e-2 here
        eps2 = 1e-4
        fd2 = (evalx(s + eps2) - 2 * evalx(s) + evalx(s - eps2)) / eps2**2
        np.testing.assert_allclose(d["x_ss"], fd2, rtol=1e-4, atol=1e-5)

    @pytest.mark.unit
    def test_curve_zero_omega_unchanged(self):
        """Curves without W behave exactly as before."""
        curve = FourierRZCurve(R_n=[0, 10, 1], Z_n=[0, 0, -1], NFP=1, sym=False)
        assert curve.W_basis.num_modes == 1  # constant mode, coefficient zero
        np.testing.assert_allclose(np.asarray(curve.W_n), 0)
        s = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        grid = Grid(np.vstack([np.zeros_like(s), np.zeros_like(s), s]).T, sort=False)
        d = curve.compute(["x"], grid=grid, basis="xyz")
        # modes [-1, 0, 1] = [sin(s), 1, cos(s)] so R = 10 + cos(s), Z = -cos(s)
        R = 10 + np.cos(s)
        x_true = np.array([R * np.cos(s), R * np.sin(s), -np.cos(s)]).T
        np.testing.assert_allclose(d["x"], x_true, atol=1e-12)
