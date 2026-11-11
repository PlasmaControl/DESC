"""
Tests for dipoles and dipolesets.
"""

from pathlib import Path

import numpy as np
import pytest

from desc.backend import jnp
from desc.dipole import (
    _Dipole,
    DipoleSet,
    export_dipoles,
    import_dipoles,
    initialize_axisymmetric_dipole_grid,
    initialize_dipole_grid,
    magnetic_dipole_field,
    magnetic_dipole_vector_field,
)
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.magnetic_fields._core import dipole_field, dipole_vector_potential

TEST_DIR = (Path(__file__).parent / "inputs").resolve()


def unit_dipole(**kwargs):
    """Make an isolated dipole with a near-unit normalized moment."""
    defaults = dict(X=0., Y=0., Z=0., phi=0., theta=0., m0=1., rho=1.)
    defaults.update(kwargs)
    return _Dipole(**defaults)


def cartesian_moment(d):
    return np.asarray(d.m_xyz, dtype=float)


@pytest.mark.unit
class TestDipole:
    @pytest.mark.parametrize(
        "theta,phi,moment", [(0., 0., [0, 0, 1]),
                             (np.pi / 2, 0., [1, 0, 0]),
                             (np.pi / 2, np.pi / 2, [0, 1, 0])]
    )
    def test_analytic_field_and_potential(self, theta, phi, moment):
        """Both direct dipole kernels match the Cartesian point-dipole kernel."""
        pts = np.array([[0., 0., 1.], [1., 1., 2.]])
        src = np.zeros((1, 3))
        B = magnetic_dipole_field(pts, src, phi, theta, 1.)
        A = magnetic_dipole_vector_field(pts, src, phi, theta, 1.)
        np.testing.assert_allclose(B, dipole_field(pts, src, np.asarray(moment)[None, :]), atol=1e-14)
        np.testing.assert_allclose(A, dipole_vector_potential(pts, src, np.asarray(moment)[None, :]), atol=1e-14)
        if theta == 0:
            np.testing.assert_allclose(B[0], [0, 0, 2e-7], atol=1e-13)

    def test_properties_and_strength(self):
        d = unit_dipole(X=1, Y=2, Z=3, phi=.3, theta=.8, m0=4., rho=0.)
        np.testing.assert_allclose(d.position, [1, 2, 3])
        np.testing.assert_allclose(d.rho_tilde, 0.)
        np.testing.assert_allclose(d.M0, 0.)
        d.rho_tilde = -.25
        np.testing.assert_allclose(d.rho_tilde, -.25, rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(cartesian_moment(d)), 1., rtol=1e-6)
        d.X = 6
        d.Y = 7
        d.Z = 8
        d.m0 = 2
        np.testing.assert_allclose(d.position, [6, 7, 8])
        np.testing.assert_allclose(d.M0, -.5, rtol=1e-6)

    def test_translation_copy_and_rotation(self):
        d = unit_dipole(X=1., theta=np.pi / 2)
        original = d.copy()
        d.translate([1, 2, 3])
        np.testing.assert_allclose(d.position, [2, 2, 3])
        np.testing.assert_allclose(original.position, [1, 0, 0])
        d.rotate([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(d.position, [-2, 2, 3], atol=1e-12)
        np.testing.assert_allclose(cartesian_moment(d), [0, np.tanh(1), 0], atol=1e-12)

    @pytest.mark.parametrize("rho", [0., 2., -2.])
    def test_rotation_preserves_orientation_for_zero_or_signed_rho(self, rho):
        """Regression: angular transformation must not depend on moment strength."""
        d = unit_dipole(X=1., theta=np.pi / 2, rho=rho)
        d.rotate([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(d.position, [0, 1, 0], atol=1e-12)
        assert np.isfinite(float(d.phi)) and np.isfinite(float(d.theta))
        np.testing.assert_allclose([np.sin(d.theta) * np.cos(d.phi),
                                    np.sin(d.theta) * np.sin(d.phi), np.cos(d.theta)],
                                   [0, 1, 0], atol=1e-12)

    def test_reflection(self):
        d = unit_dipole(X=1., Y=2., Z=3., theta=0., rho=0.)
        d.flip([0, 0, 1])
        np.testing.assert_allclose(d.position, [1, 2, -3])
        assert np.isfinite(float(d.theta))

    def test_compute_field(self):
        d = unit_dipole(X=.2, theta=np.pi / 2, rho=.7)
        pos = np.array([[0, 0, 2], [.5, 1, 2]])
        expected = dipole_field(pos, np.asarray(d.position)[None], cartesian_moment(d)[None])
        np.testing.assert_allclose(d.compute_magnetic_field(pos, basis="xyz"), expected, rtol=1e-10, atol=1e-14)


@pytest.mark.unit
class TestDipoleSet:
    def test_linspaced_linear(self):
        template = unit_dipole(X=1, Y=2, Z=3, rho=.1)
        group = DipoleSet.linspaced_linear(template, displacement=[0, 0, 4], n=4)
        np.testing.assert_allclose(np.asarray([d.position for d in group]),
                                   [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6]])
        assert len(group) == 4 and group[0] is not template
        np.testing.assert_allclose(group.rho, .1)
        group[0].rho = .4
        np.testing.assert_allclose(template.rho, .1)

    def test_linspaced_linear_endpoint_and_strengths(self):
        d = unit_dipole()
        group = DipoleSet.linspaced_linear(d, displacement=[3, 0, 0],
                                            rho=[0, .2, -.3, .4], n=4, endpoint=True)
        np.testing.assert_allclose([x.X for x in group], [0, 1, 2, 3])
        np.testing.assert_allclose(group.rho, [0, .2, -.3, .4])

    def test_linspaced_angular(self):
        d = unit_dipole(X=2, theta=np.pi / 2, rho=0.)
        group = DipoleSet.linspaced_angular(d, n=4, rho=[0, 1, -1, 2])
        np.testing.assert_allclose([x.position for x in group],
                                   [[2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0]], atol=1e-12)
        np.testing.assert_allclose(group.rho, [0, 1, -1, 2])
        for item, pos in zip(group, [x.position for x in group]):
            direction = [np.sin(item.theta)*np.cos(item.phi),
                         np.sin(item.theta)*np.sin(item.phi), np.cos(item.theta)]
            np.testing.assert_allclose(direction, np.asarray(pos)/2, atol=1e-12)

    def test_linspaced_angular_endpoint(self):
        group = DipoleSet.linspaced_angular(unit_dipole(X=1), n=3, angle=np.pi,
                                            endpoint=True, NFP=2, sym=True)
        np.testing.assert_allclose([d.position for d in group],
                                   [[1, 0, 0], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
        assert group.NFP == 2 and group.sym and group.num_dipoles == 12

    @pytest.mark.parametrize("constructor", ["linspaced_linear", "linspaced_angular"])
    def test_linspaced_input_validation(self, constructor):
        factory = getattr(DipoleSet, constructor)
        with pytest.raises(ValueError):
            factory(unit_dipole(), n=0)
        with pytest.raises(TypeError):
            factory(DipoleSet(unit_dipole()))
        with pytest.raises(ValueError):
            factory(unit_dipole(), n=3, rho=[1, 2])

    def test_properties_setter_and_collection_methods(self):
        ds = DipoleSet(unit_dipole(X=1), unit_dipole(X=2), NFP=3, sym=True)
        assert len(ds) == 2 and ds.num_dipoles == 12
        ds.rho = [0.25, -.5]
        np.testing.assert_allclose(ds.rho, [.25, -.5])
        np.testing.assert_allclose(ds.rho_tilde, np.tanh([.25, -.5]))
        ds.name = "set"
        assert ds.name == "set"
        ds.insert(1, unit_dipole(X=3))
        np.testing.assert_allclose(ds.X, [1, 3, 2])
        ds[1] = unit_dipole(X=4)
        np.testing.assert_allclose(ds.X, [1, 4, 2])
        del ds[1]
        np.testing.assert_allclose(ds.X, [1, 2])
        with pytest.raises(TypeError):
            ds.insert(0, "not a dipole")
        with pytest.raises(TypeError):
            ds[0] = "not a dipole"

    def test_copy_is_independent(self):
        ds = DipoleSet(unit_dipole(X=1), unit_dipole(X=2))
        cp = ds.copy()
        assert cp is not ds and cp[0] is not ds[0]
        cp.rho = [2, 3]
        np.testing.assert_allclose(ds.rho, 1.)

    def test_collection_transformations(self):
        ds = DipoleSet(unit_dipole(X=1), unit_dipole(Y=1))
        ds.translate([0, 0, 2])
        np.testing.assert_allclose(ds.Z, 2)
        ds.rotate([0, 0, 1], np.pi/2)
        np.testing.assert_allclose(ds.X, [0, -1], atol=1e-12)

    def test_vector_potential_is_sum_of_individual_moments(self):
        ds = DipoleSet(unit_dipole(X=.4, rho=.6), unit_dipole(Y=.2, rho=-.4))
        pts = np.array([[0., 0., 2.], [.1, .5, 1.]])
        expected = dipole_vector_potential(pts,
            np.asarray([d.position for d in ds]),
            np.asarray([cartesian_moment(d) for d in ds]))
        np.testing.assert_allclose(ds.compute_magnetic_vector_potential(pts, basis="xyz"),
                                   expected, rtol=1e-10, atol=1e-14)

    def test_compute_registry(self):
        ds = DipoleSet(unit_dipole(X=1, Y=2, Z=3, m0=4, rho=.2),
                       unit_dipole(X=2, Y=3, Z=4, m0=5, rho=-.3))
        vals = ds.compute(["X", "Y", "Z", "rho"])
        assert len(vals) == 2
        for i, d in enumerate(ds):
            for key in ("X", "Y", "Z", "rho"):
                np.testing.assert_allclose(vals[i][key], getattr(d, key))

    def test_field_is_sum_of_individual_moments(self):
        ds = DipoleSet(unit_dipole(X=0, rho=.5, m0=3),
                       unit_dipole(X=1, rho=-.2, theta=np.pi/2, m0=2))
        evaluation = np.array([[0, 0, 3], [2, 0, 3]])
        expected = dipole_field(evaluation,
                                np.array([d.position for d in ds]),
                                np.array([cartesian_moment(d) for d in ds]))
        np.testing.assert_allclose(ds.compute_magnetic_field(evaluation, basis="xyz"),
                                   expected, rtol=1e-10, atol=1e-13)

    @pytest.mark.parametrize("NFP,sym", [(1, False), (1, True), (2, False), (2, True)])
    def test_symmetry_equals_explicit_images(self, NFP, sym):
        ds = DipoleSet(unit_dipole(X=1, Y=.2, Z=.1, phi=.5,
                                   theta=1.1, rho=.7), NFP=NFP, sym=sym)
        full = DipoleSet.from_symmetry(ds, NFP=NFP, sym=sym)
        eval_pts = np.array([[0.25, -.6, 1.5], [1.8, .2, .5]])
        np.testing.assert_allclose(ds.compute_magnetic_field(eval_pts, basis="xyz"),
                                   full.compute_magnetic_field(eval_pts, basis="xyz"),
                                   rtol=1e-9, atol=1e-13)
        assert len(full) == NFP * (2 if sym else 1)

    def test_dunder_methods(self):
        a = DipoleSet(unit_dipole(X=1))
        b = DipoleSet(unit_dipole(X=2))
        c = a + b
        assert len(c) == 2 and c[0] is a[0]
        d = a + [unit_dipole(X=3)]
        assert len(d) == 2
        assert isinstance(a.__add__("bad"), type(NotImplemented))

    def test_import_export_csv(self, tmp_path):
        ds = DipoleSet(unit_dipole(X=1, rho=.5), unit_dipole(Y=2, rho=-.3))
        path = tmp_path / "magnets.csv"
        export_dipoles(ds, path)
        loaded = import_dipoles(NFP=1, sym=False, filename=path)
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded.rho_tilde, ds.rho_tilde, atol=1e-12)
        np.testing.assert_allclose(loaded.X, ds.X)
        np.testing.assert_allclose(loaded.Y, ds.Y)

    def test_import_focus_and_skip_inactive(self, tmp_path):
        path = tmp_path / "magnets.focus"
        path.write_text("# Total number of coils, momentq\n2 1\n"
                        "# coiltype, symmetry, coilname, ox, oy, oz, Ic, M_0, pho, Lc, mp, mt\n"
                        "3,1,pm0,1,0,0,1,2,0.5,1,0,1.5707963267948966\n"
                        "3,1,pm1,2,0,0,0,2,0.5,1,0,1.5707963267948966\n")
        loaded = import_dipoles(NFP=1, sym=False, filename=path)
        assert len(loaded) == 1
        np.testing.assert_allclose(loaded[0].X, 1.)
        np.testing.assert_allclose(loaded[0].rho_tilde, .5)
        assert path.with_suffix(".csv").exists()


@pytest.mark.unit
class TestGridInitialization:
    def test_axisymmetric_circle_grid(self):
        """Circular axis has zero torsion; magnets must point toward the circle."""
        R0, a = 4., .5
        phi = np.linspace(0, np.pi/2, 16)
        points = np.column_stack((R0*np.cos(phi), R0*np.sin(phi), np.zeros_like(phi)))
        ds = initialize_dipole_grid(points, a=a, target_size=.38, L=2,
                                     layer_thick=.12, wall_thick=.015, gap=.02,
                                     NFP=2, sym=True, n_fine=128)
        assert len(ds) > 0 and ds.NFP == 2 and ds.sym
        for d in ds:
            R = np.hypot(float(d.X), float(d.Y))
            z = float(d.Z)
            radial_distance = np.hypot(R - R0, z)
            assert radial_distance >= a + .02
            assert radial_distance <= a + .02 + 2*.12 + 1e-6
            to_axis = np.array([-(R-R0)*float(d.X)/R,
                                -(R-R0)*float(d.Y)/R, -z])
            if np.linalg.norm(to_axis) > 1e-12:
                direction = np.array([np.sin(float(d.theta))*np.cos(float(d.phi)),
                                      np.sin(float(d.theta))*np.sin(float(d.phi)),
                                      np.cos(float(d.theta))])
                np.testing.assert_allclose(direction, to_axis/np.linalg.norm(to_axis), atol=2e-3)

    def test_layer_counts_and_moment(self):
        phi = np.linspace(0, np.pi/2, 15)
        axis = np.column_stack((5*np.cos(phi), 5*np.sin(phi), np.zeros_like(phi)))
        ds1 = initialize_dipole_grid(axis, .6, .45, L=1, layer_thick=.1, m0=4., rho=.8)
        ds2 = initialize_dipole_grid(axis, .6, .45, L=2, layer_thick=.1, m0=4., rho=.8)
        assert len(ds2) >= len(ds1)
        np.testing.assert_allclose(ds2.m0, 4.)
        np.testing.assert_allclose(ds2.rho, .8)

    def test_ports_exclude_towers(self):
        phi = np.linspace(0, np.pi/2, 12)
        axis = np.column_stack((5*np.cos(phi), 5*np.sin(phi), np.zeros_like(phi)))
        full = initialize_dipole_grid(axis, .6, .45, n_fine=64)
        removed = initialize_dipole_grid(axis, .6, .45, n_fine=64,
            ports=[dict(theta_port=0., s_port=.5, port_radius=.9)])
        assert 0 < len(removed) < len(full)

    def test_invalid_geometry(self):
        axis = np.array([[2., 0, 0], [1., 1, 0], [0., 2, 0], [-1, 1, 0]])
        with pytest.raises(ValueError):
            initialize_dipole_grid(axis, a=3, target_size=.2)
        with pytest.raises(ValueError):
            initialize_dipole_grid(axis, a=.2, target_size=-.1)

    def test_axisymmetric_equilibrium_clearance(self, monkeypatch):
        """Use a circular synthetic LCFS to test derived R0/max-minor radius."""
        # Only the geometry data needed by the initializer; no equilibrium solve.
        class FakeEq:
            NFP = 2
            sym = True
            def compute(self, keys, grid):
                if np.max(np.asarray(grid.nodes)[:, 0]) < .5:
                    return {"R": np.full(grid.num_nodes, 5.),
                            "Z": np.full(grid.num_nodes, .1)}
                theta = np.asarray(grid.nodes)[:, 1]
                return {"R": 5 + .4*np.cos(theta),
                        "Z": .1 + .4*np.sin(theta)}
        eq = FakeEq()
        ds = initialize_axisymmetric_dipole_grid(
            eq, target_size=.35, clearance=.15, L=1,
            layer_thick=.1, n_axis=20, M_surface=24, N_surface=8,
        )
        assert len(ds) > 0 and ds.NFP == 2 and ds.sym
        for d in ds:
            assert np.hypot(np.hypot(float(d.X), float(d.Y)) - 5., float(d.Z)-.1) >= .55
        with pytest.raises(ValueError, match="inside sampled plasma"):
            initialize_axisymmetric_dipole_grid(eq, target_size=.35,
                clearance=.15, a=.45, n_axis=20)


@pytest.mark.unit
def test_calc_g_matches_direct_field():
    """Validate precomputed normal-field matrix against direct Biot-Savart field."""
    eq = Equilibrium(L=2, M=2, N=0, NFP=1, sym=False)
    ds = DipoleSet(unit_dipole(X=12, Y=0, Z=0, theta=np.pi/2, rho=.4),
                   unit_dipole(X=11, Y=1, Z=.5, theta=1., phi=.3, rho=-.7))
    grid = LinearGrid(rho=1., M=3, N=3, NFP=1, sym=False)
    G = np.asarray(ds.calc_g(eq, grid=grid))
    data = eq.compute(["x", "n_rho"], grid=grid, basis="xyz")
    xyz = np.asarray(data["x"])
    normals = np.asarray(data["n_rho"])
    B = np.asarray(ds.compute_magnetic_field(xyz, basis="xyz"))
    expected = np.einsum("ij,ij->i", B, normals)
    predicted = G @ (np.asarray(ds.m0) * np.tanh(np.asarray(ds.rho)))
    np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-11)


@pytest.mark.unit
@pytest.mark.parametrize("NFP,sym,expected", [
    (1, False, [-.00803107, -.01251551, .00783013]),
    (1, True, [-.01055722, -.02013324, -.00170434]),
    (2, False, [-.0053455, -.01181211, .00882978]),
    (2, True, [-.01015419, -.01831055, -.00213807]),
])
def test_optional_muse_field_regression(NFP, sym, expected):
    """Compare to independent SIMSOPT reference if optional input is available."""
    file = TEST_DIR / "muse_dipoles_desc.csv"
    if not file.exists():
        pytest.skip("MUSE CSV fixture not available")
    ds = import_dipoles(NFP=NFP, sym=sym, filename=file)
    result = np.asarray(ds.compute_magnetic_field([[.1, 0, .1]], basis="xyz"))[0]
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-9)


@pytest.mark.unit
def test_dipole_objective_residuals():
    """Check definitions of DipoleDiscreteness and DipoleVolume directly."""
    from desc.objectives._dipole import DipoleDiscreteness, DipoleVolume
    rho_raw = np.array([-1.5, 0., .1, 3.])
    ds = DipoleSet(*[unit_dipole(rho=r) for r in rho_raw])
    strength = np.tanh(rho_raw)
    d = np.abs(strength) * (1 - np.abs(strength))
    bin_obj = DipoleDiscreteness(ds)
    vol_obj = DipoleVolume(ds)

    bin_obj.build(use_jit=False, verbose=0)
    vol_obj.build(use_jit=False, verbose=0)
    np.testing.assert_allclose(bin_obj.compute({"rho": jnp.asarray(rho_raw)}),
                               [np.linalg.norm(d)], atol=1e-12)
    np.testing.assert_allclose(vol_obj.compute({"rho": jnp.asarray(rho_raw)}),
                               (strength**2 + 1e-12)**.25, atol=1e-12)
