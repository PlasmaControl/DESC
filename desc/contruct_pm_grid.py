"""Generate a plasma-following or axis-tube permanent-magnet grid."""

import numpy as np

from desc.dipole import DipoleSet, _Dipole


def _make_axis_spline_frame(points):
    """Build arc-length splines and a stable normal frame along an axis."""
    from scipy.interpolate import CubicSpline

    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s_nodes = np.r_[0.0, np.cumsum(segments)]
    xyz_spline = [CubicSpline(s_nodes, points[:, i]) for i in range(3)]

    def center_at(s):
        return np.column_stack([f(s) for f in xyz_spline])

    def frame_at(s):
        pos = center_at(s)
        tangent = np.column_stack([f(s, 1) for f in xyz_spline])
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        angle = np.arctan2(pos[:, 1], pos[:, 0])
        radial = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(pos))))
        e1 = radial - np.sum(radial * tangent, axis=1)[:, None] * tangent
        norm = np.linalg.norm(e1, axis=1)
        bad = norm < 1e-9
        if np.any(bad):
            z = np.tile([0.0, 0.0, 1.0], (int(bad.sum()), 1))
            e1[bad] = z - np.sum(z * tangent[bad], axis=1)[:, None] * tangent[bad]
            norm = np.linalg.norm(e1, axis=1)
        if np.any(norm < 1e-9):
            raise ValueError("Could not construct local normal plane")
        e1 /= norm[:, None]
        return e1, np.cross(tangent, e1)

    return s_nodes[-1], xyz_spline, center_at, frame_at


def _remove_short_line_runs(exists, min_run):
    """Remove contiguous true-runs shorter than ``min_run`` in-place."""
    for side in (+1, -1):
        for k in range(exists[side].shape[0]):
            mask = exists[side][k]
            edges = np.flatnonzero(
                np.diff(np.concatenate([[0], mask.astype(int), [0]]))
            )
            for start, stop in zip(edges[::2], edges[1::2]):
                if stop - start < min_run:
                    mask[start:stop] = False


def _boxes_overlap(cA, FA, cB, FB, half_widths):
    """Vectorized separating-axis test for pairs of oriented rectangular boxes."""
    displacement = cB - cA
    separated = np.zeros(len(displacement), dtype=bool)
    axes = (
        [FA[:, i] for i in range(3)]
        + [FB[:, i] for i in range(3)]
        + [np.cross(FA[:, i], FB[:, j]) for i in range(3) for j in range(3)]
    )
    for axis in axes:
        radius_a = sum(
            half_widths[i] * np.abs(np.sum(FA[:, i] * axis, axis=1)) for i in range(3)
        )
        radius_b = sum(
            half_widths[i] * np.abs(np.sum(FB[:, i] * axis, axis=1)) for i in range(3)
        )
        separated |= (
            np.abs(np.sum(displacement * axis, axis=1)) > radius_a + radius_b + 1e-12
        )
    return ~separated


def initialize_dipole_grid(
    axis_xyz=None,
    a=None,
    target_size=None,
    L=1,
    *,
    layer_thick=None,
    wall_thick=0.0,
    gap=0.0,
    magnetization=1.0e6,
    m0=None,
    rho=0.0,
    ports=(),
    NFP=1,
    sym=False,
    n_fine=512,
    eq=None,
    M_surface=48,
    N_surface=96,
    d_offset=0.02,
    seed_theta=0.0,
    L_straight=0.0,
    theta_line_samples=720,
    z_step=0.002,
    verbose=False,
):
    """Generate a plasma-following or axis-tube permanent-magnet grid.

    With ``eq`` supplied (or as the first positional argument), the magnet
    towers follow the actual sampled LCFS, including noncircular and
    nonaxisymmetric shaping. Without ``eq``, the original circular
    axis-normal tube is retained for backwards compatibility.

    The legacy path is based on ``make_circular_normal_PM_grid_global.py``
    from Djin; The input ``axis_xyz`` samples a *single* toroidal domain.
    Towers use arc-length spacing along each displaced toroidal curve, and magnets
    point inward toward the axis.

    Parameters
    ----------
    axis_xyz : array-like, shape (k, 3), or Equilibrium, optional
        Legacy centerline samples; alternatively supply an equilibrium here.
    a : float, optional
        Circular vessel radius [m] for the legacy centerline mode. Not used
        when ``eq`` is supplied.
    eq : Equilibrium, optional
        DESC equilibrium defining the LCFS to follow. Its surface normal is
        used to set magnet positions and inward dipole directions.
    M_surface, N_surface : int
        Poloidal and toroidal resolutions for sampling the plasma boundary.
    target_size : float
        Square magnet footprint width [m].
    L : int
        Maximum number of radial magnet layers per tower.
    layer_thick : float, optional
        Magnet radial thickness [m]; defaults to ``target_size``.
    wall_thick : float
        Extra footprint pitch [m] between towers.
    gap : float
        Surface-to-first-magnet-face clearance [m].
    magnetization : float
        Magnetization [A/m], giving m0 = magnetization * volume.
    m0 : float, optional
        Explicit moment [A m^2], overrides magnetization-derived moment.
    rho : float
        Initial *raw* strength, e.g. 0 for a cold start.
    ports : sequence of mappings
        Port definitions ``theta_port``, ``s_port`` in [0, 1], and
        ``port_radius`` [m]. Blocked towers are OMITTED since DipoleSet
        has no Ic parameter (unlike FAMUS/FOCUS).
    NFP, sym : int, bool
        Virtual symmetries; only generate unique-domain dipoles.
    d_offset : float
        Vessel-to-plasma normal offset [m].
    seed_theta, L_straight : float
        Line seed angle and optional end-cut setback, respectively.
    theta_line_samples, z_step : int, float
        Line packing integration resolutions.
    n_fine : int
        Axis interpolation samples for arc-length measurements.

    Returns
    -------
    DipoleSet
        Stored dipoles in tower-major order. Radial orientation is inward.

    Notes
    -----
    The plasma-following mode positions the *inner magnet faces* at ``gap``
    from sampled LCFS points along the local outward unit normal. Its
    distance test does not certify clearance everywhere on the continuous
    LCFS or detect oriented-box intersections.

    Inter-tower clearance is checked using a KD-tree of magnet centers,
    excluding pairs in the same tower. This is a center-distance proxy,
    not a full oriented box-intersection test. Also verify that the chosen
    surface lies outside the actual plasma before manufacturing magnets.
    """
    from scipy.spatial import cKDTree

    if eq is None and hasattr(axis_xyz, "compute") and hasattr(axis_xyz, "NFP"):
        eq, axis_xyz = axis_xyz, None
    if eq is not None:
        if a is not None:
            raise ValueError("a describes a circular tube; omit a when using eq")
        if target_size is None:
            raise ValueError("target_size is required")
        return _initialize_dipole_grid_from_equilibrium_lines(
            eq,
            target_size,
            L=L,
            layer_thick=layer_thick,
            wall_thick=wall_thick,
            gap=gap,
            magnetization=magnetization,
            m0=m0,
            rho=rho,
            ports=ports,
            M_surface=M_surface,
            N_surface=N_surface,
            d_offset=d_offset,
            seed_theta=seed_theta,
            L_straight=L_straight,
            theta_line_samples=theta_line_samples,
            z_step=z_step,
            verbose=verbose,
        )
    if a is None or target_size is None:
        raise ValueError("axis_xyz, a, and target_size required without eq")
    points = np.asarray(axis_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("axis_xyz must have shape (k,3) with k >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError("axis_xyz contains non-finite coordinates")
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError("L must be a positive integer")
    if not isinstance(n_fine, (int, np.integer)) or n_fine < 16:
        raise ValueError("n_fine must be an integer >= 16")
    layer_thick = target_size if layer_thick is None else layer_thick
    if min(a, target_size, layer_thick) <= 0 or min(wall_thick, gap) < 0:
        raise ValueError(
            "radii, footprint, and thickness must be positive; gaps nonnegative"
        )
    if NFP < 1:
        raise ValueError("NFP must be positive")
    pitch = target_size + wall_thick
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segments < 1e-12):
        raise ValueError("axis_xyz has duplicate consecutive samples")
    axis_length, xyz_spline, center_at, frame_at = _make_axis_spline_frame(points)

    fine_s = np.linspace(0, axis_length, n_fine)
    fine_center = center_at(fine_s)
    fine_e1, fine_e2 = frame_at(fine_s)
    # Warn/raise if the normal tube locally folds around the axis curvature.
    d1 = np.column_stack([f(fine_s, 1) for f in xyz_spline])
    d2 = np.column_stack([f(fine_s, 2) for f in xyz_spline])
    curvature = (
        np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1) ** 3
    )
    if a >= 1.0 / max(float(curvature.max()), 1e-12):
        raise ValueError("a exceeds the sampled axis minimum curvature radius")

    n_poloidal = max(1, int(np.floor(2 * np.pi * a / pitch)))
    bases, directions, theta_values, s_values = [], [], [], []
    for theta in np.linspace(0, 2 * np.pi, n_poloidal, endpoint=False):
        out_f = np.cos(theta) * fine_e1 + np.sin(theta) * fine_e2
        curve = fine_center + a * out_f
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        n_toroidal = max(1, int(np.floor(lengths[-1] / pitch)))
        arc_targets = (np.arange(n_toroidal) + 0.5) * lengths[-1] / n_toroidal
        s_samples = np.interp(arc_targets, lengths, fine_s)
        e1, e2 = frame_at(s_samples)
        outward = np.cos(theta) * e1 + np.sin(theta) * e2
        bases.extend(center_at(s_samples) + a * outward)
        directions.extend(outward)
        theta_values.extend([theta] * n_toroidal)
        s_values.extend(s_samples)

    bases = np.asarray(bases)
    directions = np.asarray(directions)
    theta_values, s_values = np.asarray(theta_values), np.asarray(s_values)
    active = np.ones(len(bases), dtype=bool)
    for port in ports:
        theta_delta = np.abs(
            (theta_values - float(port["theta_port"]) + np.pi) % (2 * np.pi) - np.pi
        )
        s_delta = np.abs(s_values - float(port["s_port"]) * axis_length)
        active &= np.hypot(a * theta_delta, s_delta) > float(port["port_radius"])

    # Grow towers simultaneously; reject candidate points too close to
    # a *different* tower (not the previous layers of their own stack).
    placed, owners = [], []
    growing = active.copy()
    for layer in range(L):
        indices = np.flatnonzero(growing)
        if not len(indices):
            break
        candidates = bases[indices] + directions[indices] * (
            gap + (layer + 0.5) * layer_thick
        )
        acceptable = np.ones(len(indices), dtype=bool)
        if placed:
            old_points = np.asarray(placed)
            # Exact exclusion of same-tower points: search all nearby
            # candidates, not just k nearest, to avoid missing cross-tower hits.
            tree = cKDTree(old_points)
            for i, neighbors in enumerate(
                tree.query_ball_point(
                    candidates, r=target_size + 0.5 * wall_thick - 1e-10
                )
            ):
                if any(owners[j] != indices[i] for j in neighbors):
                    acceptable[i] = False
        if len(indices) > 1:
            pairs = cKDTree(candidates).query_pairs(
                r=target_size + 0.5 * wall_thick - 1e-10, output_type="ndarray"
            )
            if len(pairs):
                acceptable[np.unique(pairs.ravel())] = False
        for i in np.flatnonzero(acceptable):
            placed.append(candidates[i])
            owners.append(int(indices[i]))
        growing[indices[~acceptable]] = False
    if not placed:
        raise ValueError("No dipoles fit; decrease target_size/wall_thick or adjust a")

    positions = np.asarray(placed)
    owners = np.asarray(owners)
    moments = magnetization * target_size**2 * layer_thick if m0 is None else float(m0)
    result = []
    for k in np.lexsort((np.arange(len(owners)), owners)):
        outward = directions[owners[k]]
        inward = -outward
        result.append(
            _Dipole(
                X=positions[k, 0],
                Y=positions[k, 1],
                Z=positions[k, 2],
                phi=np.arctan2(inward[1], inward[0]),
                theta=np.arccos(np.clip(inward[2], -1, 1)),
                m0=moments,
                rho=rho,
            )
        )
    return DipoleSet(*result, NFP=NFP, sym=sym)


def _initialize_dipole_grid_from_equilibrium_lines(
    eq,
    target_size,
    *,
    L=1,
    layer_thick=None,
    wall_thick=0.0,
    gap=0.0,
    magnetization=1e6,
    m0=None,
    rho=0.0,
    ports=(),
    M_surface=96,
    N_surface=128,
    d_offset=0.02,
    seed_theta=0.0,
    L_straight=0.0,
    theta_line_samples=720,
    z_step=0.002,
    verbose=False,
):
    """Pack square magnets on an outward normal-offset LCFS using continuous lines.

    Adapted from ``constant_offset_QI.py``: surface-spline normals, opposing
    geodesic-distance fronts, arc-length tower placement, half-period axis
    end-cuts, and cross-tower / symmetry-image proximity and oriented-box tests.
    Unlike sampled-node thinning, the number of candidate sites is determined
    by ``target_size + wall_thick`` and the physical vessel area, not by the
    number of equilibrium sample nodes.

    ``d_offset`` positions the vessel outside the plasma; ``gap`` separates the
    vessel from the first magnet inner face. The inward direction is ``-normal``.
    ``rho`` is the RAW optimization parameter, not normalized tanh(rho).
    Ports are omitted from the returned DipoleSet (no FOCUS Ic field).
    """
    from scipy.interpolate import CubicSpline, RectBivariateSpline
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from scipy.spatial import cKDTree

    from desc.grid import LinearGrid

    if not eq.sym:
        raise ValueError("line packing currently requires eq.sym=True")
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError("L must be a positive integer")
    if not isinstance(M_surface, (int, np.integer)) or M_surface < 8:
        raise ValueError("M_surface must be >= 8")
    if not isinstance(N_surface, (int, np.integer)) or N_surface < 8:
        raise ValueError("N_surface must be >= 8")
    layer_thick = target_size if layer_thick is None else layer_thick
    if (
        target_size <= 0
        or layer_thick <= 0
        or wall_thick < 0
        or gap < 0
        or d_offset <= 0
    ):
        raise ValueError("Invalid magnet footprint, thickness, or clearance")
    if L_straight < 0 or theta_line_samples < 64 or z_step <= 0:
        raise ValueError("Invalid end cut or packing resolution")
    if m0 is None and magnetization <= 0:
        raise ValueError("magnetization must be positive")
    NFP = eq.NFP
    footprint = float(target_size)
    L_requested = int(L)
    # Sample one half-period, with periodic theta but inclusive zeta end planes.
    th_nodes = np.linspace(0, 2 * np.pi, int(M_surface), endpoint=False)
    ze_nodes = np.linspace(0, np.pi / NFP, int(N_surface), endpoint=True)
    surf_grid = LinearGrid(rho=[1.0], theta=th_nodes, zeta=ze_nodes, NFP=NFP, sym=False)
    surf_data = eq.compute(["x"], grid=surf_grid, basis="xyz")
    surf_pts = np.asarray(surf_grid.meshgrid_reshape(surf_data["x"], "rtz"))[0]
    if surf_pts.shape != (len(th_nodes), len(ze_nodes), 3):
        raise ValueError(f"Unexpected LCFS shape {surf_pts.shape}")

    # Axis serves ONLY to define the two transverse cutting planes.
    axis_grid = LinearGrid(
        rho=[0.0],
        theta=[0.0],
        zeta=np.linspace(0, np.pi / NFP, 100),
        NFP=NFP,
        sym=False,
    )
    axis_xyz = np.asarray(eq.compute(["x"], grid=axis_grid, basis="xyz")["x"])
    if axis_xyz.shape != (100, 3):
        raise ValueError(f"Unexpected magnetic-axis shape {axis_xyz.shape}")
    PORTS = []
    for p in ports:
        if "theta_port" not in p or "zeta_port" not in p or "port_radius" not in p:
            raise ValueError(
                "Each port needs theta_port, zeta_port, port_radius (angles in radians)"
            )
        if p["port_radius"] < 0:
            raise ValueError("Port radius must be nonnegative")
        PORTS.append(p)
    # The vessel is the normal offset of the actual plasma surface.
    cell_spacing = footprint + wall_thick  # center-to-center tower spacing (pitch)
    collision_dist = footprint + wall_thick / 2.0

    # --- Spline the axis (only used for the two end cuts) ---
    seg = np.linalg.norm(np.diff(axis_xyz, axis=0), axis=1)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    L_axis = s_raw[-1]

    csx = CubicSpline(s_raw, axis_xyz[:, 0])
    csy = CubicSpline(s_raw, axis_xyz[:, 1])
    csz = CubicSpline(s_raw, axis_xyz[:, 2])
    d1x, d1y, d1z = csx.derivative(1), csy.derivative(1), csz.derivative(1)

    # --- Stop the magnets L_straight (in x) back from the end of the axis ---
    x_end = axis_xyz[-1, 0]
    s_chk = np.linspace(0.0, L_axis, 20000)
    dx = np.abs(csx(s_chk) - x_end)
    if L_straight > 0:
        far = np.where(dx >= L_straight)[0]
        if len(far) == 0:
            raise ValueError(
                f"ERROR: L_straight ({L_straight:.4f} m) is longer ",
                "than axis's x extent.",
            )
        i = far[-1]
        s_cut = np.interp(L_straight, [dx[i + 1], dx[i]], [s_chk[i + 1], s_chk[i]])
    else:
        s_cut = L_axis

    def axis_plane(s):
        """Point and unit tangent of the axis at s.

        The plane through them, perpendicular to the axis, is a domain end.
        """
        c = np.array([csx(s), csy(s), csz(s)])
        T = np.array([d1x(s), d1y(s), d1z(s)])
        return c, T / np.linalg.norm(T)

    c0, T0 = axis_plane(0.0)  # nose: symmetry plane
    c1, T1 = axis_plane(
        s_cut
    )  # L_straight cut (a symmetry plane only if L_straight = 0)

    def in_domain(P):
        """f0 >= 0 and f1 <= 0 inside the domain (between the two end planes)."""
        return (P - c0) @ T0, (P - c1) @ T1

    # =============================================================================
    # PLASMA SURFACE AND THE OFFSET (VACUUM VESSEL) SURFACE
    # =============================================================================

    surf_pts = np.asarray(surf_pts, dtype=float)
    if surf_pts.ndim != 3 or surf_pts.shape[2] != 3:
        raise ValueError(
            "ERROR: surf_pts must have shape (n_theta, n_zeta, 3), "
            f"got {surf_pts.shape}."
        )
    n_th, n_ze = surf_pts.shape[:2]

    def stell_flip(p):
        """Stellarator-symmetry map (x, y, z) -> (x, -y, -z)."""
        return p * np.array([1.0, -1.0, -1.0])

    def rot_z(p, ang):
        c, s = np.cos(ang), np.sin(ang)
        return np.stack(
            [c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]],
            axis=-1,
        )

    # --- Check the (theta, zeta) layout with stellarator symmetry ---
    jm = (-np.arange(n_th)) % n_th  # index of -theta
    size = np.ptp(surf_pts[:, 0, :], axis=0).max()
    err_0 = np.abs(surf_pts[jm, 0] - stell_flip(surf_pts[:, 0])).max()
    err_end = np.abs(
        surf_pts[jm, -1] - rot_z(stell_flip(surf_pts[:, -1]), 2 * np.pi / NFP)
    ).max()
    if max(err_0, err_end) > 1e-3 * size:
        raise ValueError(
            "ERROR: surf_pts does not look like (n_theta, n_zeta, 3) over "
            f"zeta = 0 .. pi/NFP (symmetry error "
            f"{max(err_0, err_end)*1000:.3f} mm). "
            "Check the reshape."
        )

    # --- Rebuild the surface over zeta = -pi/NFP .. 2*pi/NFP from the half period ---
    zeta_half = np.linspace(0.0, np.pi / NFP, n_ze)
    period = np.concatenate(
        [stell_flip(surf_pts[jm, n_ze - 2 : 0 : -1]), surf_pts], axis=1
    )
    zeta_period = np.concatenate([-zeta_half[n_ze - 2 : 0 : -1], zeta_half])
    surf_win = np.concatenate([period, rot_z(period, 2 * np.pi / NFP)], axis=1)
    zeta_win = np.concatenate([zeta_period, zeta_period + 2 * np.pi / NFP])

    # Periodic in theta: pad a few columns on each side, then fit cubic splines.
    th_grid = 2 * np.pi * np.arange(n_th) / n_th
    PAD = 4
    th_pad = np.concatenate(
        [th_grid[-PAD:] - 2 * np.pi, th_grid, th_grid[:PAD] + 2 * np.pi]
    )
    sp = [
        RectBivariateSpline(
            th_pad,
            zeta_win,
            np.concatenate(
                [surf_win[-PAD:, :, i], surf_win[:, :, i], surf_win[:PAD, :, i]], axis=0
            ),
        )
        for i in range(3)
    ]

    def _ev(th, ze, dth, dze):
        th = np.mod(th, 2 * np.pi)
        return np.stack([f.ev(th, ze, dx=dth, dy=dze) for f in sp], axis=-1)

    # Outward orientation of X_theta x X_zeta (decided once, from the given surface).
    _X = _ev(np.repeat(th_grid, 5), np.tile(np.linspace(0.1, 1.4, 5) / NFP, n_th), 0, 0)
    _n = np.cross(
        _ev(np.repeat(th_grid, 5), np.tile(np.linspace(0.1, 1.4, 5) / NFP, n_th), 1, 0),
        _ev(np.repeat(th_grid, 5), np.tile(np.linspace(0.1, 1.4, 5) / NFP, n_th), 0, 1),
    )
    _cen = _X.reshape(n_th, 5, 3).mean(axis=0)
    N_SIGN = 1.0 if np.sum(_n * (_X - np.tile(_cen, (n_th, 1)))) > 0 else -1.0

    def vessel(th, ze):
        """Vacuum-vessel (offset) surface at plasma coordinates (theta, zeta).

        Returns P (point), P_th, P_ze (tangent vectors) and n (outward unit normal,
        the same for the plasma surface and the offset surface).
        """
        X, Xt = _ev(th, ze, 0, 0), _ev(th, ze, 1, 0)
        Xz, Xtt = _ev(th, ze, 0, 1), _ev(th, ze, 2, 0)
        Xtz, Xzz = _ev(th, ze, 1, 1), _ev(th, ze, 0, 2)
        nv = N_SIGN * np.cross(Xt, Xz)
        nm = np.linalg.norm(nv, axis=-1, keepdims=True)
        n = nv / nm
        nt = N_SIGN * (np.cross(Xtt, Xz) + np.cross(Xt, Xtz))
        nz = N_SIGN * (np.cross(Xtz, Xz) + np.cross(Xt, Xzz))
        n_t = (nt - n * np.sum(n * nt, axis=-1, keepdims=True)) / nm
        n_z = (nz - n * np.sum(n * nz, axis=-1, keepdims=True)) / nm
        return X + d_offset * n, Xt + d_offset * n_t, Xz + d_offset * n_z, n

    # =============================================================================
    # WHERE THE DOMAIN SITS IN ZETA, AND THE OFFSET-FOLD CHECK
    # =============================================================================

    N_T = int(theta_line_samples)  # theta samples per cross-line
    th_f = seed_theta + np.linspace(0.0, 2 * np.pi, N_T + 1)
    ze_c = np.linspace(-np.pi / NFP, 2 * np.pi / NFP, 600)
    Pc, Ptc, Pzc, nc = vessel(
        np.repeat(th_f[:-1:4], len(ze_c)), np.tile(ze_c, N_T // 4)
    )
    f0, f1 = in_domain(Pc)
    inside = ((f0 >= 0) & (f1 <= 0)).reshape(N_T // 4, len(ze_c))
    # keep only the patch connected to the given half period
    # (planes also cut far-away parts)
    cols = np.where(
        inside.any(axis=0) & (ze_c > -0.5 * np.pi / NFP) & (ze_c < 1.5 * np.pi / NFP)
    )[0]
    if len(cols) == 0:
        raise ValueError(
            "ERROR: the end planes do not enclose any of the vessel surface."
        )
    ze_lo = ze_c[max(cols.min() - 2, 0)]
    ze_hi = ze_c[min(cols.max() + 2, len(ze_c) - 1)]
    if ze_lo <= ze_c[0] or ze_hi >= ze_c[-1]:
        raise ValueError(
            "ERROR: the domain reaches the edge of the rebuilt surface window."
        )

    jac = N_SIGN * np.sum(np.cross(Ptc, Pzc) * nc, axis=-1)
    patch = ((f0 >= 0) & (f1 <= 0)).reshape(N_T // 4, len(ze_c)) & (
        (ze_c >= ze_lo) & (ze_c <= ze_hi)
    )[None, :]
    patch = patch.ravel()
    if np.any(jac[patch] <= 0):
        raise ValueError(
            f"ERROR: vessel surface folds over itself at d_offset = {d_offset} m ",
            "(the plasma is more concave than d_offset somewhere). Reduce d_offset.",
        )

    # =============================================================================
    # LINES OF SQUARES
    # =============================================================================
    # Every line follows the vessel surface the long way around. The SEED line is
    # the constant-theta line theta = seed_theta. Lines are added outward from it in
    # both directions, each at least one pitch (perpendicular, measured on the
    # surface) from the previous one. The two fronts meet on the far side; there a
    # line only exists where it still fits, so lines start and stop individually
    # exactly where room appears or disappears.
    #
    # At each zeta the cross-line coordinate w (distance across constant-theta
    # lines) is integrated around the vessel from the seed. Lines are placed at
    # chosen w; because a line can tilt relative to the constant-theta lines, the
    # true perpendicular spacing is then measured in 3D and the w-gaps are widened
    # wherever it falls below one pitch (repeated until it does not).

    ZE_STEP = float(z_step)  # m along the vessel, approx.
    R_typ = np.linalg.norm(Pc[:, :2], axis=-1).max()
    N_Z = int(np.ceil((ze_hi - ze_lo) * R_typ / ZE_STEP)) + 1
    ze_l = np.linspace(ze_lo, ze_hi, N_Z)

    # w(theta) at every zeta of the line grid
    TH = np.tile(th_f, (N_Z, 1))
    ZE = np.repeat(ze_l[:, None], N_T + 1, axis=1)
    _, Pt, Pz, _ = vessel(TH.ravel(), ZE.ravel())
    Pt, Pz = Pt.reshape(N_Z, N_T + 1, 3), Pz.reshape(N_Z, N_T + 1, 3)
    h = np.linalg.norm(np.cross(Pt, Pz), axis=-1) / np.linalg.norm(Pz, axis=-1)
    w = np.concatenate(
        [
            np.zeros((N_Z, 1)),
            np.cumsum(0.5 * (h[:, 1:] + h[:, :-1]) * np.diff(th_f)[None, :], axis=1),
        ],
        axis=1,
    )
    W = w[:, -1]  # once around the vessel

    K = int(np.ceil(W.max() / (2 * cell_spacing))) + 10  # candidate lines per side
    SMOOTH_N = 2 * int(np.ceil(2 * cell_spacing / ZE_STEP)) + 1  # ~3 cm along the line

    def line_points(wpos):
        """(theta, P) of lines at cross-line positions wpos (K, N_Z)."""
        th = np.empty_like(wpos)
        for j in range(N_Z):
            th[:, j] = np.interp(np.clip(wpos[:, j], 0.0, W[j]), w[j], th_f)
        P, _, _, _ = vessel(th.ravel(), np.tile(ze_l, wpos.shape[0]))
        return th, P.reshape(wpos.shape[0], N_Z, 3)

    def dist_to_line(Q, R, m=8):
        """Distance from each point Q[k, j] to the polyline R[k, :].

        (only the segments around index j are searched).
        """
        best = np.full(Q.shape[:2], np.inf)
        J = Q.shape[1]
        for o in range(-m, m):
            ia = np.clip(np.arange(J) + o, 0, J - 1)
            ib = np.clip(np.arange(J) + o + 1, 0, J - 1)
            a, b = R[:, ia], R[:, ib]
            ab = b - a
            t = np.clip(
                np.sum((Q - a) * ab, axis=-1)
                / np.maximum(np.sum(ab * ab, axis=-1), 1e-30),
                0,
                1,
            )
            best = np.minimum(best, np.linalg.norm(Q - a - t[..., None] * ab, axis=-1))
        return best

    _, seed_P = line_points(np.zeros((1, N_Z)))
    gaps = {+1: np.full((K, N_Z), cell_spacing), -1: np.full((K, N_Z), cell_spacing)}
    for it in range(30):
        worst = 0.0
        lines = {}
        for side in (+1, -1):
            pos = np.cumsum(gaps[side], axis=0)
            wpos = pos if side > 0 else W[None, :] - pos
            th_k, P_k = line_points(wpos)
            inner = np.concatenate([seed_P, P_k[:-1]], axis=0)
            D = dist_to_line(P_k, inner)
            ok = pos < 0.6 * W[None, :]  # past ~halfway the other front is there first
            fac = np.where(
                ok,
                np.maximum(1.0, cell_spacing * (1 + 1e-4) / np.maximum(D, 1e-9)),
                1.0,
            )
            # widen smoothly along each line (never less than needed):
            # running max, then running mean
            fac = uniform_filter1d(
                maximum_filter1d(fac, SMOOTH_N, axis=1, mode="nearest"),
                SMOOTH_N // 2 + 1,
                axis=1,
                mode="nearest",
            )
            worst = max(worst, fac.max())
            gaps[side] *= fac
            lines[side] = (pos, th_k, P_k)
        if worst <= 1 + 2e-4:
            break
    else:
        raise ValueError("ERROR: the line spacing did not converge.")

    # --- Where each line exists: alternate sides until the two fronts meet ---
    exists = {+1: np.zeros((K, N_Z), bool), -1: np.zeros((K, N_Z), bool)}
    for j in range(N_Z):
        pos_p, pos_m = lines[+1][0][:, j], lines[-1][0][:, j]
        npl = nmi = 0
        while True:
            added = False
            if npl < K:
                b_out = W[j] - (pos_m[nmi - 1] if nmi else 0.0)
                if b_out - pos_p[npl] >= gaps[+1][npl, j]:
                    npl += 1
                    added = True
            if nmi < K:
                a_out = pos_p[npl - 1] if npl else 0.0
                if (W[j] - pos_m[nmi]) - a_out >= gaps[-1][nmi, j]:
                    nmi += 1
                    added = True
            if not added:
                break
        exists[+1][:npl, j] = True
        exists[-1][:nmi, j] = True

    # Check the meeting gap in 3D and drop lines that come too close.
    jj = np.clip(np.arange(N_Z)[:, None] + np.arange(-8, 9)[None, :], 0, N_Z - 1)
    for _ in range(10):
        npl = exists[+1].sum(axis=0)
        nmi = exists[-1].sum(axis=0)
        kp, km = np.maximum(npl - 1, 0), np.maximum(nmi - 1, 0)
        Pp = np.where((npl > 0)[:, None], lines[+1][2][kp, np.arange(N_Z)], seed_P[0])
        Rm = np.where(
            (nmi > 0)[:, None, None], lines[-1][2][km[:, None], jj], seed_P[0][jj]
        )
        a, ab = Rm[:, :-1], Rm[:, 1:] - Rm[:, :-1]
        t = np.clip(
            np.sum((Pp[:, None] - a) * ab, axis=-1)
            / np.maximum(np.sum(ab * ab, axis=-1), 1e-30),
            0,
            1,
        )
        D = np.linalg.norm(Pp[:, None] - a - t[..., None] * ab, axis=-1).min(axis=1)
        bad = (D < cell_spacing) & ((npl > 0) | (nmi > 0))
        if not bad.any():
            break
        for j in np.where(bad)[0]:
            if npl[j] >= nmi[j]:
                exists[+1][npl[j] - 1, j] = False
            else:
                exists[-1][nmi[j] - 1, j] = False

    # Drop line pieces too short to hold a tower without flicker.
    MIN_RUN = int(np.ceil(3 * cell_spacing / ZE_STEP))
    _remove_short_line_runs(exists, MIN_RUN)

    lines_all = [(seed_P[0], np.full(N_Z, seed_theta), np.ones(N_Z, bool))]
    for side in (+1, -1):
        for k in range(K):
            if exists[side][k].any():
                lines_all.append(
                    (lines[side][2][k], lines[side][1][k], exists[side][k])
                )

    # =============================================================================
    # TOWERS ALONG THE LINES
    # =============================================================================

    bases, outs, tans, th_all, ze_all, line_of = [], [], [], [], [], []
    for li, (Pl, thl, ex) in enumerate(lines_all):
        g0, g1 = in_domain(Pl)
        keep = ex & (g0 >= 0) & (g1 <= 0)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            # polyline in index space; extend to the exact plane crossings
            # at the domain ends
            idx = np.arange(a, b, dtype=float)
            for end, nb in ((a, a - 1), (b - 1, b)):
                if 0 <= nb < N_Z and ex[nb]:
                    for g in (g0, -g1):
                        if g[end] >= 0 > g[nb]:
                            f = g[end] / (g[end] - g[nb])
                            idx = np.append(idx, end + f * (nb - end))
            idx = np.sort(idx)
            P_run = np.stack(
                [np.interp(idx, np.arange(N_Z), Pl[:, i]) for i in range(3)], axis=1
            )
            th_run = np.interp(idx, np.arange(N_Z), np.unwrap(thl))
            ze_run = np.interp(idx, np.arange(N_Z), ze_l)
            cum = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(P_run, axis=0), axis=1))]
            )
            N = int(np.floor(cum[-1] / cell_spacing))
            if N < 1:
                continue
            tgt = (np.arange(N) + 0.5) * (cum[-1] / N)  # half-step from both ends
            th_t, ze_t = np.interp(tgt, cum, th_run), np.interp(tgt, cum, ze_run)
            P_t, _, _, n_t = vessel(th_t, ze_t)
            seg_i = np.clip(np.searchsorted(cum, tgt) - 1, 0, len(cum) - 2)
            t_dir = P_run[seg_i + 1] - P_run[seg_i]
            t_dir = t_dir - n_t * np.sum(t_dir * n_t, axis=1, keepdims=True)
            t_dir /= np.linalg.norm(t_dir, axis=1, keepdims=True)
            bases += list(P_t)
            outs += list(n_t)
            tans += list(t_dir)
            th_all += list(th_t)
            ze_all += list(ze_t)
            line_of += [li] * N

    bases, outs, tans = np.asarray(bases), np.asarray(outs), np.asarray(tans)
    th_all, ze_all, line_of = (
        np.asarray(th_all),
        np.asarray(ze_all),
        np.asarray(line_of),
    )
    T = len(bases)
    if T == 0:
        raise ValueError("ERROR: no towers fit in the domain.")

    # =============================================================================
    # PORTS
    # =============================================================================

    port_centers = (
        vessel(
            np.array([p["theta_port"] for p in PORTS]),
            np.array([p["zeta_port"] for p in PORTS]),
        )[0]
        if PORTS
        else np.zeros((0, 3))
    )
    port_radius = np.array([p["port_radius"] for p in PORTS])
    if PORTS:
        port_blocked = (
            np.linalg.norm(bases[:, None, :] - port_centers[None], axis=-1)
            <= port_radius[None, :]
        ).any(axis=1)
    else:
        port_blocked = np.zeros(T, dtype=bool)

    # =============================================================================
    # SYMMETRY IMAGES
    # =============================================================================
    # The nose plane is a stellarator-symmetry plane: the next half period is the
    # flipped copy. The far end is one too only when L_straight = 0.

    IMAGE_OPS = [lambda p: stell_flip(p)]
    if L_straight == 0:
        IMAGE_OPS.append(lambda p: rot_z(stell_flip(p), 2 * np.pi / NFP))

    def images_of(pts):
        if len(pts) == 0:
            return np.zeros((0, 3))
        return np.vstack([op(pts) for op in IMAGE_OPS])

    # =============================================================================
    # SQUARE-MAGNET BOX TEST
    # =============================================================================
    # Each magnet is a footprint x footprint x layer_thick box: one side along its
    # line, the other across it, the thickness along the surface normal. Boxes are
    # grown by wall/4 on each side across the surface, so two boxes "overlap" when
    # the wall between them would be thinner than wall/2 -- the same allowance as
    # collision_dist. Separating-axis test for pairs of boxes.

    BOX_HALF = np.array(
        [
            footprint / 2 + wall_thick / 4,
            footprint / 2 + wall_thick / 4,
            layer_thick / 2,
        ]
    )
    BOX_R = 2 * np.linalg.norm(BOX_HALF)  # boxes farther apart than this cannot touch

    def box_frames(t):
        n = outs[t]
        a = tans[t]
        return np.stack([a, np.cross(n, a), n], axis=1)  # (.., 3 axes, 3)

    def boxes_overlap(cA, FA, cB, FB):
        return _boxes_overlap(cA, FA, cB, FB, BOX_HALF)

    def image_frames(F, k):
        """Box axes of the k-th symmetry image (the symmetry maps are linear)."""
        return np.stack([IMAGE_OPS[k](F[:, i]) for i in range(3)], axis=1)

    # =============================================================================
    # GRID GENERATION  (global layer-by-layer growth)
    # =============================================================================
    # Every tower is grown together, one layer at a time. Before a layer is placed,
    # we check each still-growing tower's candidate magnet against ALL magnets already
    # placed (any tower), the other candidates being placed on this same layer, and
    # the symmetry images of both. If a candidate would come within
    # (footprint + wall/2) of any of them, or its square box would overlap any of
    # their boxes, that tower is frozen -- it stops at its current height and
    # places no more layers.

    K_NEIGH = int(np.ceil(collision_dist / layer_thick)) + 3

    placed_xyz, placed_tower, placed_layer = [], [], []
    L_tower = np.zeros(T, dtype=int)
    growing = np.ones(T, dtype=bool)

    for l in range(L_requested):
        if not growing.any():
            break

        idx = np.where(growing)[0]
        cand = bases[idx] + outs[idx] * (gap + (l + 0.5) * layer_thick)
        ok = np.ones(len(idx), dtype=bool)

        # placed magnets of OTHER towers
        if placed_xyz:
            placed_arr = np.asarray(placed_xyz)
            placed_tw = np.asarray(placed_tower)
            kq = min(len(placed_arr), K_NEIGH)
            dq, iq = cKDTree(placed_arr).query(cand, k=kq)
            if kq == 1:
                dq = dq[:, None]
                iq = iq[:, None]
            dq = np.where(placed_tw[iq] == idx[:, None], np.inf, dq)
            ok &= dq.min(axis=1) >= collision_dist

        # other towers' candidates on this layer (both members of a clash are rejected)
        if len(idx) > 1:
            pairs = cKDTree(cand).query_pairs(r=collision_dist, output_type="ndarray")
            if len(pairs) > 0:
                ok[np.unique(pairs.ravel())] = False

        # symmetry images of placed magnets and of this layer's candidates
        src = np.vstack([np.asarray(placed_xyz).reshape(-1, 3), cand])
        di, _ = cKDTree(images_of(src)).query(cand, k=1)
        ok &= di >= collision_dist

        # --- the same three checks with the actual square boxes ---
        Fc = box_frames(idx)
        if placed_xyz:
            nb = cKDTree(placed_arr).query_ball_point(cand, r=BOX_R)
            ci = np.repeat(np.arange(len(idx)), [len(v) for v in nb])
            pj = (
                np.concatenate([np.asarray(v, dtype=int) for v in nb])
                if len(ci)
                else np.zeros(0, int)
            )
            m = placed_tw[pj] != idx[ci]
            ci, pj = ci[m], pj[m]
            if len(ci):
                hit = boxes_overlap(
                    cand[ci], Fc[ci], placed_arr[pj], box_frames(placed_tw[pj])
                )
                ok[np.unique(ci[hit])] = False
        if len(idx) > 1:
            pairs = cKDTree(cand).query_pairs(r=BOX_R, output_type="ndarray")
            if len(pairs) > 0:
                hit = boxes_overlap(
                    cand[pairs[:, 0]],
                    Fc[pairs[:, 0]],
                    cand[pairs[:, 1]],
                    Fc[pairs[:, 1]],
                )
                ok[np.unique(pairs[hit].ravel())] = False
        src_tw = np.concatenate([np.asarray(placed_tower, dtype=int), idx])
        src_F = box_frames(src_tw)
        for kop in range(len(IMAGE_OPS)):
            img = IMAGE_OPS[kop](src)
            nb = cKDTree(img).query_ball_point(cand, r=BOX_R)
            ci = np.repeat(np.arange(len(idx)), [len(v) for v in nb])
            if len(ci):
                pj = np.concatenate([np.asarray(v, dtype=int) for v in nb])
                hit = boxes_overlap(
                    cand[ci], Fc[ci], img[pj], image_frames(src_F[pj], kop)
                )
                ok[np.unique(ci[hit])] = False

        for t in idx[ok]:
            placed_xyz.append(bases[t] + outs[t] * (gap + (l + 0.5) * layer_thick))
            placed_tower.append(int(t))
            placed_layer.append(l)
            L_tower[t] = l + 1
        growing[idx[~ok]] = False

    placed_xyz = np.asarray(placed_xyz).reshape(-1, 3)
    placed_tower = np.asarray(placed_tower, dtype=int)
    placed_layer = np.asarray(placed_layer, dtype=int)
    if len(placed_xyz) == 0:
        raise ValueError("No magnets fit; reduce footprint/wall or change offset")
    # Output only magnets from unblocked towers. Keep the same orientation across
    # the entire radial stack, just as in the FOCUS generator.
    order = np.lexsort((placed_layer, placed_tower))
    moment = magnetization * target_size**2 * layer_thick if m0 is None else float(m0)
    dipoles = []
    for k in order:
        t = int(placed_tower[k])
        if port_blocked[t]:
            continue
        p = placed_xyz[k]
        inward = -outs[t]
        dipoles.append(
            _Dipole(
                X=float(p[0]),
                Y=float(p[1]),
                Z=float(p[2]),
                phi=float(np.arctan2(inward[1], inward[0])),
                theta=float(np.arccos(np.clip(inward[2], -1.0, 1.0))),
                m0=float(moment),
                rho=float(rho),
            )
        )
    if verbose:
        print(f"Surface-following grid: {T:,} towers, {len(dipoles):,} stored magnets")
        print(
            f"Layers per tower: min={L_tower.min()}, max={L_tower.max()}, "
            f"mean={L_tower.mean():.2f}"
        )

        print(f"Pitch: {cell_spacing*1000:.2f} mm; offset: {d_offset*1000:.2f} mm")
        print(f"Estimated full symmetry-expanded magnets: {len(dipoles)*2*NFP:,}")
    return DipoleSet(*dipoles, NFP=NFP, sym=True)


def initialize_axisymmetric_dipole_grid(
    eq,
    target_size,
    L=1,
    *,
    a=None,
    clearance=0.0,
    layer_thick=None,
    wall_thick=0.0,
    gap=0.0,
    magnetization=1.0e6,
    m0=None,
    rho=0.0,
    n_axis=64,
    M_surface=32,
    N_surface=32,
):
    """Build a circular-axis (zero-torsion) dipole grid around a DESC equilibrium.

    Estimate the circular centerline using the magnetic-axis mean major radius
    and mean Z. Estimate a conservative required minor radius as the largest
    distance of LCFS R,Z samples to that circle's (R0,Z0) center. Set the
    tube radius to that bound plus ``clearance``, or the user-provided ``a``
    if it is *at least* as large. The resulting centerline is a planar circle
    with torsion zero, not a nonaxisymmetric approximation to the actual axis.

    This guarantees separation at the sampled LCFS nodes in R,Z, subject to
    positive gap/layer thickness. It is not a continuous-surface clearance
    certificate between sample points, nor a box collision certificate.
    """
    from desc.grid import LinearGrid

    if not isinstance(n_axis, (int, np.integer)) or n_axis < 4:
        raise ValueError("n_axis must be an integer >= 4")
    if clearance < 0:
        raise ValueError("clearance must be nonnegative")
    # Use axis major-radius and vertical centers, averaged over a full turn.
    axis_grid = LinearGrid(rho=[0.0], M=0, N=n_axis, NFP=eq.NFP, sym=False)
    axis_data = eq.compute(["R", "Z"], grid=axis_grid)
    R0 = float(np.mean(np.asarray(axis_data["R"])))
    Z0 = float(np.mean(np.asarray(axis_data["Z"])))
    surf_grid = LinearGrid(rho=[1.0], M=M_surface, N=N_surface, NFP=eq.NFP, sym=False)
    surf_data = eq.compute(["R", "Z"], grid=surf_grid)
    Rp = np.asarray(surf_data["R"])
    Zp = np.asarray(surf_data["Z"])
    a_required = float(np.max(np.hypot(Rp - R0, Zp - Z0))) + clearance
    a = a_required if a is None else float(a)
    if a < a_required:
        raise ValueError(
            f"Requested radius a={a:g} is inside sampled plasma bound {a_required:g}"
        )
    extent = np.pi / eq.NFP if eq.sym else 2 * np.pi / eq.NFP
    angles = np.linspace(0, extent, n_axis)
    axis_xyz = np.column_stack(
        (R0 * np.cos(angles), R0 * np.sin(angles), np.full(n_axis, Z0))
    )
    return initialize_dipole_grid(
        axis_xyz,
        a,
        target_size,
        L=L,
        layer_thick=layer_thick,
        wall_thick=wall_thick,
        gap=gap,
        magnetization=magnetization,
        m0=m0,
        rho=rho,
        NFP=eq.NFP,
        sym=eq.sym,
    )
