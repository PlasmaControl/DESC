"""Tests for the RidgeEquilibrium: basis, grids, tied lambda, constraints, distance."""

import warnings

import h5py
import numpy as np
import pytest
from scipy.linalg import null_space

from desc.backend import jax, jnp
from desc.basis import DoubleFourierRidgeSeries, FourierZernikeRidgeBasis
from desc.compute._ridge import (
    _RING_KEYS,
    RidgeSubtraction,
    SeparableRingEvaluator,
    _deriv_key,
    _derivs_of,
    clustered_ring_matrices,
    default_ring_rule,
)
from desc.compute.utils import _compute as compute_fun
from desc.equilibrium import Equilibrium, RidgeEquilibrium
from desc.geometry import FourierRZToroidalRidgeSurface
from desc.grid import Grid, SharpConcentricGrid, SharpSymConcentricGrid
from desc.io import load
from desc.objectives import (
    BoundaryDistance,
    FixBoundaryR,
    FixBoundaryZ,
    FixCornerFlatness,
    FixCorners,
    FixXLineLambda,
    ObjectiveFunction,
    SeparatrixDistanceFields,
    get_fixed_boundary_constraints,
    get_soft_boundary_constraints,
)
from desc.objectives.getters import maybe_add_self_consistency
from desc.objectives.utils import factorize_linear_constraints
from desc.profiles import PowerSeriesProfile
from desc.transform import Transform

NFP, M_B, N_B, GAMMA = 5, 5, -1, 0.75
KW = dict(NFP=NFP, m_b=M_B, n_b=N_B, gamma=GAMMA)


def ridge_surface(scale=1.0, seed=0):
    """Ridge surface with corners, the whole scaled by ``scale``.

    The smooth block R = 10 + cos θ - cos(6θ + 5ζ)/6, Z = -sin θ - sin(6θ + 5ζ)/6 is
    flat on the ridges θ = 2πk/5 - ζ, the sharp blocks are 0.3 Re Q_γ and 0.3 Im Q_γ,
    and a small random part is projected onto ∂_θ A = 0 on the ridges.
    """
    surf = FourierRZToroidalRidgeSurface(
        **KW, sym=True, M=6, N=1, M_b=2, N_b=1, check_orientation=False
    )
    rng = np.random.default_rng(seed)
    coeffs = []
    ## cos(6θ + 5ζ) = cos 6θ cos 5ζ - sin 6θ sin 5ζ, the modes (6, 1) and (-6, -1)
    shapes = (
        (surf.R_basis, [(0, 0, 0, 10.0), (1, 0, 0, 1.0), (6, 1, 0, -1 / 6)]),
        (surf.R_basis, [(-6, -1, 0, 1 / 6), (0, 0, 1, 0.3)]),
        (surf.Z_basis, [(-1, 0, 0, -1.0), (-6, 1, 0, -1 / 6), (6, -1, 0, -1 / 6)]),
        (surf.Z_basis, [(0, 0, 2, 0.3)]),
    )
    for k, X in enumerate(("R", "Z")):
        basis = getattr(surf, X + "_basis")
        c = 0.005 * rng.standard_normal(basis.num_modes)
        for _, terms in shapes[2 * k : 2 * k + 2]:
            for m, n, block, value in terms:
                c[basis.get_idx(0, m, n, block=block)] += value
        P = null_space(surf.corner_constraint_matrix(basis))
        coeffs.append(scale * P @ (P.T @ c))
    surf.R_lmn, surf.Z_lmn = coeffs
    return surf


@pytest.fixture(scope="module")
def eq():
    """Small RidgeEquilibrium, made flat at the corners.

    It is projected onto the corner flatness and the self-consistency, with the boundary
    free; the pins (FixCorners) then hold its own X-line values.
    """
    eq = RidgeEquilibrium(ridge_surface(), iota=PowerSeriesProfile([-0.5, 0, -0.5]))
    return project(eq, ())


def project(eq, constraints):
    """Project eq onto the constraints: x - A⁺ (A x - b)."""
    con = ObjectiveFunction(maybe_add_self_consistency(eq, constraints))
    con.build(verbose=0)
    x = np.asarray(con.x(eq))
    A = np.asarray(con.jac_scaled(x))
    x = x - np.linalg.lstsq(A, np.asarray(con.compute_scaled_error(x)), rcond=1e-12)[0]
    eq.params_dict = con.unpack_state(x, False)[0]
    return eq


@pytest.mark.unit
def test_basis_derivatives():
    """Test the ridge basis derivatives against central differences, off the corners."""
    basis = FourierZernikeRidgeBasis(4, 4, 2, 3, 3, **KW, sym="sin")
    rng = np.random.default_rng(0)
    nodes = np.column_stack(
        [
            rng.uniform(0.2, 0.9, 40),
            rng.uniform(0, 2 * np.pi, 80)[:40],
            rng.uniform(0, 1, 40),
        ]
    )
    h = 1e-5
    for d in [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 1),
        (0, 0, 2),
        (2, 0, 1),
    ]:
        d = np.array(d)
        k = np.argmax(d)
        lower, e = d.copy(), np.zeros(3)
        lower[k] -= 1
        e[k] = h
        fd = (
            np.asarray(basis.evaluate(nodes + e, lower))
            - np.asarray(basis.evaluate(nodes - e, lower))
        ) / (2 * h)
        exact = np.asarray(basis.evaluate(nodes, d))
        np.testing.assert_allclose(fd, exact, atol=1e-7 * np.max(np.abs(exact)))


@pytest.mark.unit
def test_basis_trace():
    """Test that on ρ = 1 every mode is the surface mode of the same block."""
    basis = FourierZernikeRidgeBasis(4, 4, 2, 3, 3, **KW, sym="cos")
    surface = DoubleFourierRidgeSeries(4, 2, 3, **KW, sym="cos")
    rng = np.random.default_rng(1)
    nodes = np.column_stack(
        [np.ones(50), rng.uniform(0, 2 * np.pi, 50), rng.uniform(0, 2 * np.pi, 50)]
    )
    V, S = np.asarray(basis.evaluate(nodes)), np.asarray(surface.evaluate(nodes))
    for j, (_, m, n) in enumerate(basis.modes):
        i = surface.get_idx(0, m, n, block=basis.blocks[j])
        np.testing.assert_allclose(V[:, j], S[:, i], atol=1e-14)


@pytest.mark.unit
def test_iota_must_match_corners():
    """Test that ι(1) ≠ ι_b raises: the X-lines must be field lines."""
    with pytest.raises(ValueError):
        RidgeEquilibrium(ridge_surface(), iota=PowerSeriesProfile([-0.5, 0, -0.4]))


@pytest.mark.unit
def test_sharp_sym_grid_mirror(eq):
    """Test that the force on the dropped half of the sharp grid mirrors the kept half.

    The mirror of every node of the symmetric half is a node of the full grid, and for a
    stellarator-symmetric state |F| there equals |F| on the kept node (the tied λ keeps
    its parity on mirrored rings).
    """
    full = SharpConcentricGrid(L=8, M=8, N=4, NFP=NFP, m_b=M_B, n_b=N_B)
    half = SharpSymConcentricGrid(L=8, M=8, N=4, NFP=NFP, m_b=M_B, n_b=N_B)
    np.testing.assert_allclose(np.sum(full.weights), np.sum(half.weights))
    T = 2 * np.pi / NFP
    mirror = np.column_stack(
        [
            half.nodes[:, 0],
            np.mod(-half.nodes[:, 1], 2 * np.pi),
            np.mod(-half.nodes[:, 2], T),
        ]
    )
    key = lambda n: np.round(  # noqa: E731
        np.column_stack([n[:, 0], np.cos(n[:, 1]), np.sin(n[:, 1]), n[:, 2]]), 10
    )
    index = {tuple(k): i for i, k in enumerate(key(np.asarray(full.nodes)))}
    j = np.array([index[tuple(k)] for k in key(mirror)])
    F = eq.compute("|F|", grid=full)["|F|"]
    F_half = eq.compute("|F|", grid=half)["|F|"]
    np.testing.assert_allclose(F[j], F_half, rtol=1e-9)


@pytest.mark.unit
def test_constraints_after_projection(eq):
    """Test that every linear constraint of the fixed-boundary solve holds."""
    cons = maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq))
    cons += (FixCorners(eq),)
    for con in cons:
        con.build(verbose=0)
        np.testing.assert_allclose(
            con.compute_unscaled(*con.xs(eq)), con.target, atol=1e-10
        )
    assert any(isinstance(c, FixCornerFlatness) for c in cons)
    assert any(isinstance(c, FixXLineLambda) for c in cons)


def dense_ring_data(eq, rho, alpha, zetas, shifts):
    """Compute the R, Z keys on the separable evaluator's nodes with DESC transforms."""
    theta = np.mod(alpha[None, :] + shifts[:, None], 2 * np.pi).ravel()
    nodes = np.column_stack(
        [np.full(theta.size, rho), theta, np.repeat(zetas, alpha.size)]
    )
    grid = Grid(nodes, NFP=NFP, sort=False, jitable=True)
    derivs = _derivs_of(_RING_KEYS)
    tr = {"grid": grid}
    tr.update(
        {X: Transform(grid, getattr(eq, X + "_basis"), derivs=derivs[X]) for X in "RZ"}
    )
    data = {
        _deriv_key(X, d): tr[X].transform(eq.params_dict[X + "_lmn"], *d)
        for X in "RZ"
        for d in derivs[X]
    }
    return data, tr


@pytest.mark.unit
def test_separable_evaluator(eq):
    """Test the separable ring evaluator against DESC transforms on the same nodes.

    Every R, Z key, the ring means and antiderivatives, and one Jacobian-vector product.
    """
    rho, zetas = 0.93, np.array([0.0, 0.2, -0.3])
    n, eps = default_ring_rule(rho, eq.M)
    targets = np.array([0.1, 1.0, 2.5])
    alpha, wG, A, rowA = clustered_ring_matrices(targets, n, eps, M_B)
    shifts = eq.iota_b * zetas
    sep = SeparableRingEvaluator(eq, rho, alpha, zetas, shifts)
    params = eq.params_dict
    fast = sep.compute(params)
    dense, tr = dense_ring_data(eq, rho, alpha, zetas, shifts)
    for key, value in dense.items():
        np.testing.assert_allclose(
            fast[key], value, rtol=0, atol=1e-12 * np.max(np.abs(value))
        )

    ## the ring means and antiderivatives from the two evaluations
    def ring(R_lmn, Z_lmn, separable):
        p = dict(params, R_lmn=R_lmn, Z_lmn=Z_lmn)
        if separable:
            data = sep.compute(p)
        else:
            data = {
                _deriv_key(X, d): tr[X].transform(p[X + "_lmn"], *d)
                for X in "RZ"
                for d in _derivs_of(_RING_KEYS)[X]
            }
        g = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            "sqrt(g)",
            params=p,
            transforms={"grid": tr["grid"]},
            profiles={},
            data=data,
        )["sqrt(g)"].reshape(len(zetas), n)
        G = g @ wG
        S = A @ g.T - rowA[:, None] * G[None, :]  # every target on every ring
        return jnp.concatenate([G, S.ravel()])

    args = (jnp.asarray(params["R_lmn"]), jnp.asarray(params["Z_lmn"]))
    np.testing.assert_allclose(ring(*args, True), ring(*args, False), rtol=1e-12)
    rng = np.random.default_rng(2)
    tangents = tuple(jnp.asarray(rng.standard_normal(a.size)) for a in args)
    jvp = [
        jax.jvp(lambda R, Z: ring(R, Z, s), args, tangents)[1] for s in (True, False)
    ]
    np.testing.assert_allclose(jvp[0], jvp[1], rtol=1e-10, atol=1e-12)


def direct_lambda_1(eq, rho, theta, zeta, n=4096):
    """Compute λ_1 = S/G and ∂_θ λ_1 on one ring from 4096 uniform nodes.

    The mean is the trapezoidal rule and the antiderivative is taken by the FFT.
    """
    t = 2 * np.pi * np.arange(n) / n
    grid = Grid(np.column_stack([np.full(n, rho), t, np.full(n, zeta)]), sort=False)
    g = np.asarray(eq.compute("sqrt(g)", grid=grid)["sqrt(g)"])
    G = g.mean()
    c = np.fft.rfft(g - G) / n
    m = np.arange(c.size)
    S = np.zeros_like(theta)
    for k in range(1, c.size - 1):
        S += 2 * np.real(c[k] * np.exp(1j * m[k] * theta) / (1j * m[k]))
    gt = np.asarray(
        eq.compute(
            "sqrt(g)",
            grid=Grid(
                np.column_stack(
                    [np.full_like(theta, rho), theta, np.full_like(theta, zeta)]
                ),
                sort=False,
            ),
        )["sqrt(g)"]
    )
    return S / G, (gt - G) / G


@pytest.mark.unit
def test_tied_lambda_direct(eq):
    """Test the tied λ_1 of the compute path against a direct evaluation.

    It is also single-valued: λ_1(θ = 2π) = λ_1(0).
    """
    rho, zeta = 0.95, 0.3
    theta = np.array([0.05, 1.3, 2.9, 4.4, 2 * np.pi - 0.05])
    grid = Grid(np.column_stack([np.full(5, rho), theta, np.full(5, zeta)]), sort=False)
    data = eq.compute(["lambda_1", "lambda_1_t"], grid=grid)
    lam, lam_t = direct_lambda_1(eq, rho, theta, zeta)
    np.testing.assert_allclose(data["lambda_1"], lam, atol=1e-5)
    np.testing.assert_allclose(data["lambda_1_t"], lam_t, atol=1e-5)
    ends = Grid(np.array([[rho, 0.0, zeta], [rho, 2 * np.pi, zeta]]), sort=False)
    lam = eq.compute("lambda_1", grid=ends)["lambda_1"]
    np.testing.assert_allclose(lam[0], lam[1], atol=1e-12)


def separatrix_data(surf, planes, n=400):
    """Sample the surface on the planes, uniform in α_b (corners included)."""
    α = 2 * np.pi * np.arange(n) / n
    R, Z, A, Zt = [], [], [], []
    for z in planes:
        nodes = np.column_stack(
            [np.ones(n), np.mod(α + surf.iota_b * z, 2 * np.pi), np.full(n, z)]
        )
        d = surf.compute(["R", "Z"], grid=Grid(nodes, sort=False))
        R.append(d["R"]), Z.append(d["Z"]), A.append(α), Zt.append(np.full(n, z))
    return map(np.concatenate, (R, Z, A, Zt))


@pytest.mark.unit
def test_distance_scaling():
    """Test that the boundary distance is unchanged when all is scaled by 10.

    The equilibrium and the separatrix (a different ridge surface) are scaled together.
    """
    planes = np.array([0.0, 0.3, 0.6])
    f = []
    for scale in (1.0, 10.0):
        eq = RidgeEquilibrium(
            ridge_surface(scale), iota=PowerSeriesProfile([-0.5, 0, -0.5])
        )
        R, Z, α, ζ = separatrix_data(ridge_surface(scale, seed=3), planes)
        fields = SeparatrixDistanceFields(R, Z, α, ζ, planes, np.ones(3), M_B)
        obj = BoundaryDistance(eq, fields)
        obj.build(verbose=0)
        f.append(np.asarray(obj.compute_scaled_error(*obj.xs(eq))))
    assert np.max(np.abs(f[0])) > 1e-4
    np.testing.assert_allclose(f[0], f[1], rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_distance_reader():
    """Test the distance read from the fields against a dense polyline of the edge.

    To ~1e-6 m, at points within half the band of the interior of an edge.
    """
    planes = np.array([0.0, 0.3])
    R, Z, α, ζ = separatrix_data(ridge_surface(), planes)
    fields = SeparatrixDistanceFields(R, Z, α, ζ, planes, np.ones(2), M_B)
    rng = np.random.default_rng(4)
    band = fields.arrays["band"]
    for f, curve in enumerate(fields.curves):
        u = rng.uniform(0.1 * curve.L, 0.9 * curve.L, 50)
        T = curve.d1(u)
        normal = (
            np.column_stack([T[:, 1], -T[:, 0]]) / np.linalg.norm(T, axis=1)[:, None]
        )
        pts = curve.point(u) + rng.uniform(-band / 2, band / 2, 50)[:, None] * normal
        d, inside = fields.distance(pts[:, 0], pts[:, 1], np.full(50, f))
        dense = curve.point(np.linspace(-curve.ext, curve.L + curve.ext, 400001))
        direct = np.min(np.linalg.norm(dense[None] - pts[:, None], axis=2), axis=1)
        assert np.all(np.asarray(inside))
        np.testing.assert_allclose(np.abs(np.asarray(d)), direct, atol=1e-6)


def with_lambda_resolution(eq, M_lambda):
    """Copy eq's map into a RidgeEquilibrium with λ_0 at the resolution M_lambda."""
    new = RidgeEquilibrium(
        eq.surface,
        iota=PowerSeriesProfile([-0.5, 0, -0.5]),
        L_lambda=M_lambda,
        M_lambda=M_lambda,
    )
    new.R_lmn, new.Z_lmn, new.Ra_n, new.Za_n = eq.R_lmn, eq.Z_lmn, eq.Ra_n, eq.Za_n
    return new


@pytest.mark.unit
def test_ridge_subtraction_xline(eq):
    """Test that on ρ = 1 the tied part minus E is constant along every X-line."""
    ridge = RidgeSubtraction(eq)
    tied = np.asarray(ridge.ridge_lambda_1(eq.params_dict))  # (n_zeta, m_b)
    theta = 2 * np.pi * np.arange(M_B)[None, :] / M_B + eq.iota_b * ridge.zetas[:, None]
    ridge_nodes = np.column_stack(
        [
            np.ones(theta.size),
            np.mod(theta, 2 * np.pi).ravel(),
            np.repeat(ridge.zetas, M_B),
        ]
    )
    E_ridge = np.asarray(
        ridge.compute(eq.params_dict, ridge.node_constants(ridge_nodes))[:, 0]
    ).reshape(tied.shape)
    left = tied - E_ridge
    np.testing.assert_allclose(left, np.mean(left), atol=1e-12)


@pytest.mark.unit
def test_lambda_resolution(eq):
    """Test that λ_0 at its own resolution keeps the λ constraints."""
    e = with_lambda_resolution(eq, 6)
    assert (e.L_basis.L, e.L_basis.M, e.L_basis.N) == (6, 6, eq.N)
    e = project(e, ())
    for con in maybe_add_self_consistency(e, get_fixed_boundary_constraints(e)):
        con.build(verbose=0)
        if con.dim_f:
            np.testing.assert_allclose(
                con.compute_unscaled(*con.xs(e)), con.target, atol=1e-10
            )


@pytest.mark.unit
def test_change_resolution(eq):
    """Test that raising the resolution keeps the map, λ_0 and the constraints.

    λ itself changes at the level of the ring error only: the rings of the tied λ_1
    follow the resolution M (default_ring_rule). Lowering the resolution back recovers
    the coefficients exactly.
    """
    e = eq.copy()
    e.L_lmn = np.random.default_rng(5).normal(scale=1e-3, size=e.L_basis.num_modes)
    e = project(e, ())
    grid = Grid(
        np.column_stack(
            [
                np.linspace(0.05, 0.95, 40),
                np.linspace(0, 2 * np.pi, 40),
                np.linspace(0, 1.2, 40),
            ]
        ),
        sort=False,
    )
    keys = ["R", "Z", "lambda", "sqrt(g)"]
    before = e.compute(keys, grid=grid)
    big = e.copy()
    big.change_resolution(
        L=e.L + 2, M=e.M + 1, L_b=e.L_b + 2, M_b=e.M_b + 2, M_lambda=7
    )
    assert (big.M, big.M_b, big.L_basis.M) == (e.M + 1, e.M_b + 2, 7)
    assert (big.L_b, big.M_b, big.N_b) == (e.L_b + 2, e.M_b + 2, e.N_b)
    for basis in (big.R_basis, big.Z_basis):
        assert (basis.L_b, basis.M_b, basis.N_b) == (big.L_b, big.M_b, big.N_b)
    assert (big.surface.M_b, big.surface.N_b) == (big.M_b, big.N_b)
    assert big.M_grid == 2 * max(big.M, big.M_b)
    after = big.compute(keys, grid=grid)
    for k in ("R", "Z", "sqrt(g)"):
        np.testing.assert_allclose(after[k], before[k], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(after["lambda"], before["lambda"], atol=1e-6)
    for con in maybe_add_self_consistency(big, get_fixed_boundary_constraints(big)):
        con.build(verbose=0)
        if con.dim_f:
            np.testing.assert_allclose(
                con.compute_unscaled(*con.xs(big)), con.target, atol=1e-10
            )
    big.change_resolution(L=e.L, M=e.M, L_b=e.L_b, M_b=e.M_b, M_lambda=e.L_basis.M)
    assert (big.L_b, big.M_b, big.N_b) == (e.L_b, e.M_b, e.N_b)
    for X in ("R_lmn", "Z_lmn", "L_lmn", "Rb_lmn", "Zb_lmn"):
        np.testing.assert_allclose(getattr(big, X), getattr(e, X), atol=1e-14)


@pytest.mark.unit
def test_sharp_resolution(eq, tmp_path):
    """Test the resolution of the sharp blocks: its defaults, and that it is saved.

    An equilibrium saved before L_b, M_b and N_b were saved gets them from R_basis.
    """
    surface = ridge_surface()
    iota = PowerSeriesProfile([-0.5, 0, -0.5])
    e = RidgeEquilibrium(surface, iota=iota)
    assert (e.L_b, e.M_b, e.N_b) == (surface.M_b, surface.M_b, surface.N_b)
    e = RidgeEquilibrium(surface, iota=iota, L_b=3, M_b=1, N_b=0)
    assert (e.L_b, e.M_b, e.N_b) == (3, 1, 0)
    assert (e.R_basis.L_b, e.R_basis.M_b, e.R_basis.N_b) == (3, 1, 0)
    assert "L_b=3, M_b=1, N_b=0" in repr(e)

    for old in (False, True):
        path = str(tmp_path / f"eq_{old}.h5")
        eq.save(path)
        with h5py.File(path, "r+") as f:
            assert all(name in f for name in ("_L_b", "_M_b", "_N_b"))
            if old:
                for name in ("_L_b", "_M_b", "_N_b"):
                    del f[name]
        if old:
            with pytest.warns(RuntimeWarning, match="was not loaded"):
                loaded = load(path)
        else:
            loaded = load(path)
        assert (loaded.L_b, loaded.M_b, loaded.N_b) == (eq.L_b, eq.M_b, eq.N_b)
        assert all(type(getattr(loaded, a)) is int for a in ("_L_b", "_M_b", "_N_b"))


@pytest.mark.unit
def test_ridge_constraints_compatible(eq):
    """Test that the linear constraints of the ridge solves give no warning.

    The fixed-boundary set (with the pins) and the soft set (without FixBoundaryR/Z).
    """
    fixed = get_fixed_boundary_constraints(eq) + (FixCorners(eq),)
    soft = tuple(c for c in fixed if not isinstance(c, (FixBoundaryR, FixBoundaryZ)))
    for constraints in (fixed, soft):
        con = ObjectiveFunction(maybe_add_self_consistency(eq, constraints))
        con.build(verbose=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*is not satisfied.*")
            factorize_linear_constraints(con, con)


@pytest.mark.unit
def test_soft_boundary_constraints(eq):
    """Test that the soft-boundary getter is the fixed-boundary set minus the boundary.

    That is, the fixed-boundary set without FixBoundaryR/Z, plus FixCorners for a
    RidgeEquilibrium, class by class and target by target, also with a corner target.
    """
    standard = Equilibrium(
        L=3, M=3, N=1, NFP=5, iota=PowerSeriesProfile([1, 0, 0.5]), pressure=1e3
    )
    con = FixCorners(eq)
    con.build(verbose=0)
    corner_target = con.target + 1e-3
    cases = (
        (standard, None, ()),
        (eq, None, (FixCorners(eq),)),
        (eq, corner_target, (FixCorners(eq, target=corner_target),)),
    )
    for e, target, corners in cases:
        old = (
            tuple(
                c
                for c in get_fixed_boundary_constraints(e)
                if not isinstance(c, (FixBoundaryR, FixBoundaryZ))
            )
            + corners
        )
        new = get_soft_boundary_constraints(e, corner_target=target)
        assert [type(c) for c in new] == [type(c) for c in old]
        for c_new, c_old in zip(new, old):
            c_new.build(verbose=0)
            c_old.build(verbose=0)
            np.testing.assert_array_equal(c_new.target, c_old.target)
            np.testing.assert_array_equal(c_new.weight, c_old.weight)
            np.testing.assert_array_equal(c_new.normalization, c_old.normalization)
