"""The distance of a sharp boundary to a separatrix, read from per-edge distance fields.
We include this penalty for force error minimization as the `SharpEquilibrium` 
needs to allow the parameterization of its boundary to slide.

The separatrix of an equilibrium with m_b corners on its boundary is given as points
(R, Z, α_b, ζ) on a set of toroidal planes, with the corners (the X-points) at
α_b = 2πk/m_b. On every plane, each edge between two corners becomes a smooth curve,
and its signed distance is sampled once on a narrow band around it. The boundary of the
equilibrium is then compared with the separatrix by reading these fields, which is
smooth in (R, Z) and cheap to differentiate. Every length is a fraction of the
separatrix's effective minor radius a, so nothing depends on the size of the device.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid
from desc.io import IOAble
from desc.utils import errorif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


def separatrix_edge(R, Z, alpha, k, m_b, dmin, corner_gap):
    """The ordered and thinned data of edge k (2πk/m_b ≤ α_b ≤ 2π(k+1)/m_b) of a plane.

    The traced data are not clean near the corners, where the two manifold branches
    interleave on a scale far below the spacing of the data elsewhere. So the points are
    ordered by their arc position on a coarse polyline (every 20th point in α_b, far
    beyond that scale), only the corner itself is kept within ``corner_gap`` of a
    corner, and the points are thinned to a spacing of at least ``dmin``, which keeps
    their scatter from turning into curvature in the spline.

    Parameters
    ----------
    R, Z, alpha : ndarray
        The separatrix points of one plane and their helical angles α_b, with the
        corners at α_b = 2πk/m_b.
    k : int
        The edge.
    m_b : int
        Number of corners.
    dmin, corner_gap : float
        The thinning spacing and the gap at the corners, in m.

    Returns
    -------
    R, Z : ndarray
        The points of the edge, from corner k to corner k + 1.

    """
    alpha = np.mod(alpha, 2 * np.pi)
    lo, hi = 2 * np.pi * k / m_b, 2 * np.pi * (k + 1) / m_b
    on = (alpha >= lo - 1e-12) & (alpha <= hi + 1e-12)
    o = np.argsort(alpha[on])
    x, y, t = R[on][o], Z[on][o], alpha[on][o]
    if not np.isclose(t[-1], hi):  # the closing corner of the last edge is at α = 0
        c = np.argmin(np.abs(np.angle(np.exp(1j * (alpha - hi)))))
        x, y = np.append(x, R[c]), np.append(y, Z[c])
    ## the arc position on the coarse polyline; the two corners stay at the ends
    idx = np.unique(np.r_[0 : len(x) : 20, len(x) - 1])
    P = np.column_stack([x[idx], y[idx]])
    seg = P[1:] - P[:-1]
    L2 = np.maximum(np.sum(seg**2, 1), 1e-300)
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt(L2))])
    pts = np.column_stack([x, y])
    tt = np.clip(np.sum((pts[:, None] - P[None, :-1]) * seg[None], 2) / L2, 0, 1)
    dd = np.sum((P[None, :-1] + tt[..., None] * seg[None] - pts[:, None]) ** 2, 2)
    j = np.argmin(dd, 1)
    pos = arc[j] + tt[np.arange(len(x)), j] * np.sqrt(L2[j])
    pos[0], pos[-1] = -1.0, arc[-1] + 1.0
    o = np.argsort(pos, kind="stable")
    x, y = x[o], y[o]
    ## thin, and keep only the corners within corner_gap of them
    keep = [0]
    for i in range(1, len(x) - 1):
        last = np.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]])
        gap = min(
            np.hypot(x[i] - x[0], y[i] - y[0]), np.hypot(x[i] - x[-1], y[i] - y[-1])
        )
        if last >= dmin and gap >= corner_gap:
            keep.append(i)
    keep.append(len(x) - 1)
    return x[keep], y[keep]


class EdgeCurve:
    """One edge of the separatrix as a smooth curve.

    A parametric cubic spline through the ordered edge points in chord length u,
    u ∈ [0, L], extended along its end tangents to u ∈ [-extend, L + extend].
    ``project`` gives the closest point (a coarse KD-tree start, then Newton on the
    spline) and the signed distance, positive on the right of the curve.

    Parameters
    ----------
    x, y : ndarray
        The ordered points of the edge (R, Z).
    extend : float
        Extension of the curve past its ends.
    coarse : float
        Spacing of the coarse points of the curve.

    """

    def __init__(self, x, y, extend, coarse):
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        self.sx = CubicSpline(s, x, bc_type="not-a-knot")
        self.sy = CubicSpline(s, y, bc_type="not-a-knot")
        self.L, self.ext = s[-1], extend
        t0 = np.array([self.sx(0.0, 1), self.sy(0.0, 1)])
        t1 = np.array([self.sx(self.L, 1), self.sy(self.L, 1)])
        self.t0, self.t1 = t0 / np.linalg.norm(t0), t1 / np.linalg.norm(t1)
        n = int(np.ceil((self.L + 2 * extend) / coarse)) + 1
        self.ucoarse = np.linspace(-extend, self.L + extend, n)
        self.Pcoarse = self.point(self.ucoarse)
        self.tree = cKDTree(self.Pcoarse)

    def point(self, u):
        """The point of the curve at u."""
        u = np.asarray(u, float)
        uc = np.clip(u, 0, self.L)
        P = np.stack([self.sx(uc), self.sy(uc)], -1)
        return (
            P
            + np.where(u < 0, u, 0)[..., None] * self.t0
            + np.where(u > self.L, u - self.L, 0)[..., None] * self.t1
        )

    def d1(self, u):
        """The tangent of the curve at u."""
        uc = np.clip(u, 0, self.L)
        T = np.stack([self.sx(uc, 1), self.sy(uc, 1)], -1)
        T = np.where((u < 0)[..., None], self.t0, T)
        return np.where((u > self.L)[..., None], self.t1, T)

    def d2(self, u):
        """The second derivative of the curve at u."""
        uc = np.clip(u, 0, self.L)
        S = np.stack([self.sx(uc, 2), self.sy(uc, 2)], -1)
        return np.where(((u < 0) | (u > self.L))[..., None], 0.0, S)

    def project(self, pts, newton=10, chunk=200000):
        """The signed distance of the points to the curve, and their closest u."""
        d, us = [], []
        h = self.ucoarse[1] - self.ucoarse[0]
        for i in range(0, len(pts), chunk):
            p = pts[i : i + chunk]
            u = self.ucoarse[self.tree.query(p)[1]]
            for _ in range(newton):
                r = self.point(u) - p
                T, S = self.d1(u), self.d2(u)
                g, H = np.sum(r * T, 1), np.sum(T * T, 1) + np.sum(r * S, 1)
                step = np.clip(g / np.where(H > 0, H, 1), -h, h)
                u = np.clip(u - step, -self.ext, self.L + self.ext)
            c, T = self.point(u), self.d1(u)
            dist = np.linalg.norm(p - c, axis=1)
            cross = T[:, 0] * (p[:, 1] - c[:, 1]) - T[:, 1] * (p[:, 0] - c[:, 0])
            d.append(np.where(cross > 0, -dist, dist))
            us.append(u)
        return np.concatenate(d), np.concatenate(us)


def _catmull_rom_weights(t):
    t2, t3 = t * t, t * t * t
    return jnp.stack(
        [
            (-t3 + 2 * t2 - t) / 2,
            (3 * t3 - 5 * t2 + 2) / 2,
            (-3 * t3 + 4 * t2 + t) / 2,
            (t3 - t2) / 2,
        ]
    )


def tiled_distance(R, Z, fid, fields):
    """The signed distance at (R, Z) from the field ``fid`` of each point.

    Inside the stored tiles, a C1 bicubic (Catmull-Rom) interpolation of the sampled
    signed distance; outside, the distance to the nearest coarse point of the field's
    curve, signed by its tangent.

    Parameters
    ----------
    R, Z, fid : ndarray, shape(k,)
        The points and the field of each point.
    fields : dict
        The arrays of ``SeparatrixDistanceFields`` (``arrays``): ``origin`` (nf, 2)
        the lower-left grid corner of each field, ``index`` (nf, ni, nj) the tile of
        each tile position (-1: not stored), ``tiles`` (ntiles, T + 3, T + 3) the node
        values with a halo of one node below and two above, ``coarse`` and
        ``coarse_tan`` (nf, nc, 2) the coarse curve and its tangents, and the spacing
        ``h`` and tile size ``T``.

    Returns
    -------
    d : ndarray, shape(k,)
        Signed distance.
    inside : ndarray of bool, shape(k,)
        Where the tiles were used.

    """
    h, T = fields["h"], fields["T"]
    index = fields["index"]
    ni, nj = index.shape[1], index.shape[2]
    o = fields["origin"][fid]
    fx, fy = (R - o[:, 0]) / h, (Z - o[:, 1]) / h
    i, j = jnp.floor(fx).astype(jnp.int32), jnp.floor(fy).astype(jnp.int32)
    ti, tj = i // T, j // T
    ok = (ti >= 0) & (tj >= 0) & (ti < ni) & (tj < nj)
    tile = jnp.where(
        ok, index[fid, jnp.clip(ti, 0, ni - 1), jnp.clip(tj, 0, nj - 1)], -1
    )
    li, lj = i - ti * T, j - tj * T
    a = jnp.arange(4)
    vals = fields["tiles"][
        jnp.maximum(tile, 0)[:, None, None],
        li[:, None, None] + a[None, :, None],
        lj[:, None, None] + a[None, None, :],
    ]
    wx, wy = _catmull_rom_weights(fx - i), _catmull_rom_weights(fy - j)
    v = jnp.einsum("ak,bk,kab->k", wx, wy, vals)
    ## the fallback: the nearest coarse point of the curve, signed by its tangent
    P, Tn = fields["coarse"][fid], fields["coarse_tan"][fid]
    dx, dy = R[:, None] - P[..., 0], Z[:, None] - P[..., 1]
    r2 = dx * dx + dy * dy
    k = jnp.argmin(r2, axis=1)
    sel = lambda A: jnp.take_along_axis(A, k[:, None], 1)[:, 0]  # noqa: E731
    cross = sel(Tn[..., 0]) * sel(dy) - sel(Tn[..., 1]) * sel(dx)
    fallback = jnp.where(cross > 0, -1.0, 1.0) * jnp.sqrt(sel(r2))
    inside = tile >= 0
    return jnp.where(inside, v, fallback), inside


class SeparatrixDistanceFields(IOAble):
    """The signed distance to every edge of a separatrix, sampled on narrow bands.

    One field per (plane, edge): an ``EdgeCurve`` through the edge's data
    (``separatrix_edge``), whose signed distance is sampled on the band |d| ≤ ``band``
    with spacing ``h`` and stored in square tiles of ``tile`` x ``tile`` cells (with
    a halo for the 4 x 4 Catmull-Rom stencil). Field f = p m_b + k is edge k of
    plane p. Every length parameter is a fraction of a = sqrt(<A_sep>_ζ / π), the
    effective minor radius of the separatrix, with A_sep the area inside it on a plane.

    Parameters
    ----------
    R, Z, alpha, zeta : ndarray
        The separatrix points, their helical angles α_b and planes ζ. The corners
        (X-points) are the points with α_b = 2πk/m_b.
    planes : ndarray
        The planes ζ to use.
    plane_weights : ndarray
        The weight of each plane, normalized to sum to 1.
    m_b : int
        Number of corners.
    band, extend, h, dmin, corner_gap : float
        The band half-width, the extension of the curves past the corners, the grid
        spacing, the thinning spacing and the gap kept free of data at the corners,
        all fractions of a.
    tile : int
        Tile size, in cells.

    """

    _io_attrs_ = ["_arrays", "_a", "_m_b", "_planes", "_weights", "_perimeters"]

    def __init__(
        self,
        R,
        Z,
        alpha,
        zeta,
        planes,
        plane_weights,
        m_b,
        band=0.05,
        extend=0.05,
        h=0.002,
        dmin=0.002,
        corner_gap=0.001,
        tile=32,
    ):
        R, Z, alpha, zeta = map(np.asarray, (R, Z, alpha, zeta))
        self._m_b = int(m_b)
        self._planes = np.asarray(planes, dtype=float)
        self._weights = np.asarray(plane_weights, dtype=float) / np.sum(plane_weights)
        on_plane = [np.isclose(zeta, z, rtol=0, atol=1e-9) for z in self._planes]

        ## a from the area inside the (unthinned) separatrix of each plane
        areas = []
        for on in on_plane:
            edges = [
                separatrix_edge(R[on], Z[on], alpha[on], k, m_b, 0.0, 0.0)
                for k in range(m_b)
            ]
            x = np.concatenate([e[0][:-1] for e in edges])
            y = np.concatenate([e[1][:-1] for e in edges])
            areas.append(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2)
        self._a = a = float(np.sqrt(np.sum(self._weights * np.array(areas)) / np.pi))

        ## the curves and their tiles
        h, band, T = h * a, band * a, int(tile)
        self.curves, origins, indices, tiles = [], [], [], []
        n_tiles = 0
        for on in on_plane:
            for k in range(m_b):
                x, y = separatrix_edge(
                    R[on], Z[on], alpha[on], k, m_b, dmin * a, corner_gap * a
                )
                curve = EdgeCurve(x, y, extend * a, coarse=h)
                origin, index, values = self._tile(curve, band, h, T)
                indices.append(np.where(index >= 0, index + n_tiles, -1))
                n_tiles += values.shape[0]
                self.curves.append(curve)
                origins.append(origin)
                tiles.append(values)
        ni = max(ix.shape[0] for ix in indices)
        nj = max(ix.shape[1] for ix in indices)
        index = -np.ones((len(indices), ni, nj), dtype=np.int32)
        for f, ix in enumerate(indices):
            index[f, : ix.shape[0], : ix.shape[1]] = ix
        nc = 400
        u = [np.linspace(-c.ext, c.L + c.ext, nc) for c in self.curves]
        self._arrays = dict(
            origin=np.stack(origins),
            index=index,
            tiles=np.concatenate(tiles),
            coarse=np.stack([c.point(v) for c, v in zip(self.curves, u)]),
            coarse_tan=np.stack([c.d1(v) for c, v in zip(self.curves, u)]),
            h=h,
            T=T,
            band=band,
        )
        perimeters = np.array([c.L for c in self.curves]).reshape(-1, m_b).sum(1)
        self._perimeters = perimeters

    @staticmethod
    def _tile(curve, band, h, T):
        """The tiles of the signed distance of one curve on |d| ≤ band."""
        P = curve.Pcoarse
        lo, hi = P.min(0) - band - 2 * T * h, P.max(0) + band + 2 * T * h
        nt = np.ceil((hi - lo) / (T * h)).astype(int)
        ci, cj = np.meshgrid(np.arange(nt[0]), np.arange(nt[1]), indexing="ij")
        centers = lo + (np.stack([ci.ravel(), cj.ravel()], 1) + 0.5) * T * h
        ## the tiles whose center is within band + half a tile diagonal of the curve
        reach = band + T * h * np.sqrt(2) / 2 + 3 * h
        keep = curve.tree.query(centers)[0] <= reach
        index = -np.ones(nt[0] * nt[1], dtype=np.int32)
        index[keep] = np.arange(keep.sum())
        a = np.arange(-1, T + 2)
        ti, tj = ci.ravel()[keep], cj.ravel()[keep]
        gx = lo[0] + (ti[:, None, None] * T + a[None, :, None]) * h + 0 * a[None, None]
        gy = lo[1] + (tj[:, None, None] * T + a[None, None, :]) * h + 0 * a[:, None]
        d, _ = curve.project(np.stack([gx.ravel(), gy.ravel()], 1))
        return lo, index.reshape(nt), d.reshape(len(ti), T + 3, T + 3)

    @property
    def a(self):
        """float: Effective minor radius of the separatrix, sqrt(<A_sep>_ζ / π)."""
        return self._a

    @property
    def m_b(self):
        """int: Number of corners."""
        return self._m_b

    @property
    def planes(self):
        """ndarray: The planes ζ."""
        return self._planes

    @property
    def weights(self):
        """ndarray: The weights of the planes, summing to 1."""
        return self._weights

    @property
    def perimeters(self):
        """ndarray: The perimeter of the separatrix on each plane."""
        return self._perimeters

    @property
    def arrays(self):
        """dict: The stacked arrays of the fields, as read by ``tiled_distance``."""
        return self._arrays

    @property
    def nbytes(self):
        """int: Memory of the stacked arrays, in bytes."""
        return int(
            sum(v.nbytes for v in self._arrays.values() if isinstance(v, np.ndarray))
        )

    def distance(self, R, Z, fid):
        """Signed distance at (R, Z) from the field ``fid``, and whether it is tiled.

        See ``tiled_distance``.
        """
        fields = {
            k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
            for k, v in self._arrays.items()
        }
        return tiled_distance(
            jnp.asarray(R), jnp.asarray(Z), jnp.asarray(fid, dtype=jnp.int32), fields
        )


def distance_nodes(m_b, n_uni=30, n_geo=5, skip=1e-4):
    """The helical angles of the nodes of ``BoundaryDistance`` on one plane.

    Per edge, n_uni nodes uniform in α_b, plus n_geo geometric offsets
    (1e-4 ... 5e-2) on each side of each corner; the nodes closer than ``skip`` to a
    corner are dropped (the corners are held by ``FixCorners``).

    Returns
    -------
    alpha : ndarray
        Helical angles α_b of the nodes.
    dalpha : ndarray
        Logical spacing Δα_b of each node.
    edge : ndarray of int
        Edge of each node.

    """
    width = 2 * np.pi / m_b
    alpha, dalpha, edge = [], [], []
    for k in range(m_b):
        lo = width * k
        near = np.geomspace(1e-4, 5e-2, n_geo)
        al = np.concatenate(
            [
                lo + (np.arange(n_uni) + 0.5) / n_uni * width,
                lo + near,
                lo + width - near,
            ]
        )
        al = np.unique(al)
        al = al[np.minimum(al - lo, lo + width - al) >= skip * (1 - 1e-9)]
        e = np.concatenate([[lo], al, [lo + width]])
        alpha.append(al)
        dalpha.append((e[2:] - e[:-2]) / 2)
        edge.append(np.full(al.size, k))
    return np.concatenate(alpha), np.concatenate(dalpha), np.concatenate(edge)


class BoundaryDistance(_Objective):
    """Distance of the boundary of a sharp equilibrium to a separatrix.

    At the nodes j of the boundary (ρ = 1, α_b from ``distance_nodes`` on every plane
    ζ of the fields, θ = α_b + ι_b ζ), the residual is

        f_j = sqrt(ω_ζ Δα_j |∂_θ X_j| 2π / L_sep(ζ_j)) d_e(j)(X_j),

    with X = (R, Z), d_e(j) the signed distance to the edge e(j) of the node's label
    (read from ``SeparatrixDistanceFields``), ω_ζ the plane weights (summing to 1) and
    L_sep(ζ) the separatrix's perimeter on the plane. |∂_θ X_j| is computed from the
    current map, so Σ f_j² ≈ Σ_ζ ω_ζ (2π / L_sep) ∮ d² ds is weighted by arclength and
    does not depend on how the boundary is labelled. The node may slide along its
    edge. Normalized by the equilibrium's minor radius scale ``a``, the residual does
    not change when the equilibrium and the separatrix are scaled together.

    Parameters
    ----------
    eq : RidgeEquilibrium
        Equilibrium whose boundary is compared.
    fields : SeparatrixDistanceFields
        The distance fields of the separatrix.
    n_uni, n_geo : int
        Nodes per edge uniform in α_b, and geometric ones at each side of a corner.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _scalar = False
    _linear = False
    _units = "(m)"
    _print_value_fmt = "Boundary distance error: "
    _static_attrs = _Objective._static_attrs + [
        "_n_uni",
        "_n_geo",
        "_fields_m_b",
    ]

    def __init__(
        self,
        eq,
        fields,
        n_uni=30,
        n_geo=5,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="boundary distance",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        ## only the arrays of the fields: their curves are not part of the objective
        self._fields = fields.arrays
        self._planes, self._plane_weights = fields.planes, fields.weights
        self._perimeters, self._fields_m_b = fields.perimeters, fields.m_b
        self._n_uni, self._n_geo = int(n_uni), int(n_geo)
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        errorif(
            eq.m_b != self._fields_m_b,
            ValueError,
            f"The equilibrium has m_b = {eq.m_b}, the fields {self._fields_m_b}.",
        )
        alpha, dalpha, edge = distance_nodes(eq.m_b, self._n_uni, self._n_geo)
        P, n = len(self._planes), len(alpha)
        zeta = np.repeat(self._planes, n)
        theta = np.mod(np.tile(alpha, P) + eq.iota_b * zeta, 2 * np.pi)
        nodes = np.column_stack([np.ones_like(theta), theta, zeta])
        grid = Grid(nodes, NFP=eq.NFP, sort=False)
        self._nodes = nodes
        self._dim_f = grid.num_nodes
        self._data_keys = ["R", "Z", "R_t", "Z_t"]
        sw = np.repeat(self._plane_weights * 2 * np.pi / self._perimeters, n)
        self._constants = {
            "transforms": get_transforms(self._data_keys, obj=eq, grid=grid),
            "profiles": get_profiles(self._data_keys, obj=eq, grid=grid),
            "fid": np.repeat(np.arange(P) * eq.m_b, n) + np.tile(edge, P),
            "logical_weights": sw * np.tile(dalpha, P),
            "fields": {
                k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                for k, v in self._fields.items()
            },
        }
        if self._normalize:
            self._normalization = compute_scaling_factors(eq)["a"]
        super().build(use_jit=use_jit, verbose=verbose)

    def distances(self, params):
        """The signed distances (m) of the nodes, and whether each lies in its tiles.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg ``eq.params_dict``.

        Returns
        -------
        d : ndarray, shape(dim_f,)
            Signed distance of each node to its edge.
        inside : ndarray of bool, shape(dim_f,)
            Whether the node lies in the tiles of its field.
        data : dict
            R, Z, R_t, Z_t at the nodes.

        """
        return self._distances(params, self._constants)

    def _distances(self, params, constants):
        """The distances of ``distances``, with the constants given."""
        data = compute_fun(
            self.things[0],
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        d, inside = tiled_distance(
            data["R"], data["Z"], constants["fid"], constants["fields"]
        )
        return d, inside, data

    def compute(self, params, constants=None):
        """Compute the arclength-weighted signed distances of the boundary nodes.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg ``eq.params_dict``.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray, shape(dim_f,)
            Weighted signed distances (m).

        """
        constants = self._get_deprecated_constants(constants)
        d, _, data = self._distances(params, constants)
        speed = jnp.sqrt(data["R_t"] ** 2 + data["Z_t"] ** 2)
        return jnp.sqrt(constants["logical_weights"] * speed) * d
