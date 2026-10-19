"""
Stellarator Dipole Grid Generator -- cuts NORMAL to the axis
============================================================
Populates a permanent-magnet dipole grid on a tube of constant circular
cross-section wrapped around a magnetic axis that is a general space curve
(it shifts in both R and Z as it goes around). Magnets DO point normal to the
axis: each cross-section is taken perpendicular to the local axis tangent, so a
tower's build direction lies in the normal plane and its magnets are oriented
along the true axis normal.

GEOMETRY
--------
The axis is supplied as xyz points. At each point along it, the vacuum-vessel
cross-section is a CIRCLE of constant minor radius `a`, drawn in the plane
PERPENDICULAR to the axis tangent at that point, and centered on the axis.

The circle is spanned by an orthonormal pair (e1, e2) in that normal plane:

    T  = unit axis tangent
    e1 = normalize(e_R - (e_R . T) T)     # e_R projected into the normal plane
    e2 = T x e1

e1 is built from e_R, so theta = 0 still points OUTBOARD, exactly as in the RZ
version. The frame is purely LOCAL -- it depends only on the tangent and e_R at
that point, with nothing carried along the curve. That matters for symmetry: a
parallel-transport (Bishop) frame would accumulate twist over the domain and
generally would not match its own image at the seam, whereas this frame has no
holonomy and so joins cleanly. Where the tangent is already perpendicular to
e_R this reduces exactly to the (R,Z) circle.

Because `a` is constant and every cross-section is a circle centered on the
axis, the poloidal angle theta labels the same point on every cross-section.
That lets the tube surface be described as a bundle of M TOROIDAL CURVES 

A TOROIDAL CURVE is the curve you get by taking the whole axis 
and displacing every point of it by `a` in the theta direction

    theta = 0    -> displaced outward -> larger radius -> LONGER curve
    theta = pi   -> displaced inward  -> smaller radius -> SHORTER curve
    theta = pi/2 -> displaced along e2 -> roughly unchanged length

NOTE: unlike the RZ version, the displacement is NOT purely radial -- tilting the
circle into the normal plane gives it a small toroidal component, so an offset
curve no longer sits at exactly constant phi. The outboard curve is still the
longer one, but a tilt term is now mixed in. The code measures each curve's
length numerically, so this changes the reasoning, not the result.

TOWERS AND LAYERS
-----------------
Each grid site holds a TOWER: a stack of magnets built radially OUTWARD from
the VV surface. Every magnet in every tower is oriented INWARD, pointing at its
own cross-section center (i.e. at the axis). Each magnet in a tower is a LAYER.

GRID COUNTS
-----------
  M      poloidal count, GLOBAL.  M = floor(2*pi*a / cell_spacing).
         Uniform because the cross-section radius `a` is constant everywhere.

  N_m    toroidal count, PER CURVE.  N_m = floor(S_m / cell_spacing), where S_m
         is the arc length of toroidal curve m. Since the outboard curves are
         longer, they carry MORE towers than the inboard ones -- denser outboard
         packing, each curve packed at cell_spacing along its own length.

  L      radial layer count, PER TOWER.  Found by direct measurement: build the
         tower one layer at a time and stop when a layer's magnet center comes
         within one `footprint` of the same-layer magnet of a neighboring tower
         on the same toroidal curve.

PORTS
-----
Each port is a circular keep-out region on the VV surface, given as

    {"theta_port": <poloidal angle, rad>,
     "s_port":     <position along axis, normalized to [0, 1]>,
     "port_radius":<m>}

A port at s_port = 0 or 1 sits on a domain end, so only half of it lands on
towers -- it closes into a full port against its own image under symmetry.

A tower whose base falls inside a port has its WHOLE stack disabled: the dipoles
are still written to the .focus file, but with Ic = 0 so the optimizer cannot
use them. Distance is measured in local surface arc length (flat approximation),
poloidally as a*dtheta and toroidally along the axis.

End CLEARANCE
--------------
Towers are placed at half-step offsets along each curve, so the first and last
sit half a pitch inside the curve ends. Thus, when applying symmetry the end poloidal slice will
be one full pitch apart from it's neighbor.

OUTPUT
------
FAMUS .focus file readable by ReadFAMUS / the PM optimizer.
"""

import numpy as np
import sys
from scipy.interpolate import CubicSpline

# =============================================================================
# USER INPUTS
# =============================================================================

# Magnetic axis (xyz) -- this is the domain the grid is generated on.

# example: generate in DESC with eq

#zeta = np.linspace(0, np.pi / eq.NFP, 20, endpoint=True) 
#grid = LinearGrid(rho=[0.0], M=0, zeta=zeta, NFP=eq.NFP)
#data = eq.axis.compute(["X", "Y","Z"], grid=grid)
#pts = np.array([data["X"],data["Y"],data["Z"]]).T
from desc.io import load
from desc.backend import print_backend_info
from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    MixedCoilSet,
)
import desc.examples
from desc.equilibrium import Equilibrium
from desc.plotting import plot_2d, plot_3d, plot_coils, plot_comparison
from desc.grid import LinearGrid
from desc.optimize import Optimizer
from desc.integrals import compute_B_plasma
import numpy as np
import pyvista as pv
from desc.dipole import _Dipole, DipoleSet, import_dipoles
from desc.io import load
import matplotlib.pyplot as plt
import jax.numpy as jnp
from desc.grid import LinearGrid
eq = load('qipm3.h5')
zeta = np.linspace(0, np.pi / eq.NFP, 20, endpoint=True) 
grid = LinearGrid(rho=[0.0], M=0, zeta=zeta, NFP=eq.NFP)
data = eq.axis.compute(["X", "Y","Z"], grid=grid)
pts = np.array([data["X"],data["Y"],data["Z"]]).T


axis_xyz=pts
# axis_xyz = np.array([[ 3.17498242e-01,  0.00000000e+00,  0.00000000e+00],
#        [ 3.17507983e-01,  1.01818630e-02, -3.77117453e-03],
#        [ 3.17531059e-01,  2.03861703e-02, -7.53515690e-03],
#        [ 3.17549494e-01,  3.06336177e-02, -1.12862057e-02],
#        [ 3.17534795e-01,  4.09415781e-02, -1.50212449e-02],
#        [ 3.17449860e-01,  5.13228316e-02, -1.87406276e-02],
#        [ 3.17251154e-01,  6.17846566e-02, -2.24482892e-02],
#        [ 3.16890818e-01,  7.23282613e-02, -2.61512196e-02],
#        [ 3.16318472e-01,  8.29484715e-02, -2.98583047e-02],
#        [ 3.15482571e-01,  9.36335710e-02, -3.35786996e-02],
#        [ 3.14331364e-01,  1.04365203e-01, -3.73199711e-02],
#        [ 3.12813590e-01,  1.15118293e-01, -4.10862520e-02],
#        [ 3.10879100e-01,  1.25860999e-01, -4.48765967e-02],
#        [ 3.08479610e-01,  1.36554773e-01, -4.86836496e-02],
#        [ 3.05569690e-01,  1.47154607e-01, -5.24926673e-02],
#        [ 3.02108029e-01,  1.57609565e-01, -5.62809129e-02],
#        [ 2.98058906e-01,  1.67863649e-01, -6.00174480e-02],
#        [ 2.93393737e-01,  1.77857008e-01, -6.36633675e-02],
#        [ 2.88092550e-01,  1.87527469e-01, -6.71725209e-02],
#        [ 2.82145235e-01,  1.96812302e-01, -7.04927312e-02],
#        [ 2.75552489e-01,  2.05650167e-01, -7.35674698e-02],
#        [ 2.68326397e-01,  2.13983161e-01, -7.63378951e-02],
#        [ 2.60490648e-01,  2.21758922e-01, -7.87451363e-02],
#        [ 2.52080371e-01,  2.28932723e-01, -8.07327036e-02],
#        [ 2.43141586e-01,  2.35469495e-01, -8.22489129e-02],
#        [ 2.33730231e-01,  2.41345653e-01, -8.32492224e-02],
#        [ 2.23910697e-01,  2.46550562e-01, -8.36983727e-02],
#        [ 2.13753865e-01,  2.51087453e-01, -8.35722166e-02],
#        [ 2.03334650e-01,  2.54973586e-01, -8.28591311e-02],
#        [ 1.92729195e-01,  2.58239563e-01, -8.15609251e-02],
#        [ 1.82011908e-01,  2.60927758e-01, -7.96931832e-02],
#        [ 1.71252609e-01,  2.63089996e-01, -7.72850046e-02],
#        [ 1.60514048e-01,  2.64784711e-01, -7.43780973e-02],
#        [ 1.49850019e-01,  2.66073883e-01, -7.10251878e-02],
#        [ 1.39304193e-01,  2.67020058e-01, -6.72877229e-02],
#        [ 1.28909688e-01,  2.67683725e-01, -6.32329036e-02],
#        [ 1.18689304e-01,  2.68121204e-01, -5.89301987e-02],
#        [ 1.08656301e-01,  2.68383165e-01, -5.44476135e-02],
#        [ 9.88155441e-02,  2.68513756e-01, -4.98480843e-02],
#        [ 8.91648420e-02,  2.68550298e-01, -4.51864039e-02],
#        [ 7.96963460e-02,  2.68523435e-01, -4.05070215e-02],
#        [ 7.03978738e-02,  2.68457603e-01, -3.58429348e-02],
#        [ 6.12540871e-02,  2.68371690e-01, -3.12157217e-02],
#        [ 5.22474815e-02,  2.68279775e-01, -2.66365967e-02],
#        [ 4.33591824e-02,  2.68191875e-01, -2.21082382e-02],
#        [ 3.45695602e-02,  2.68114683e-01, -1.76270427e-02],
#        [ 2.58586818e-02,  2.68052287e-01, -1.31854244e-02],
#        [ 1.72066127e-02,  2.68006881e-01, -8.77380174e-03],
#        [ 8.59357887e-03,  2.67979434e-01, -4.38200550e-03],
#        [ 1.64084464e-17,  2.67970265e-01, -8.36689392e-18]])




# PM GRID Geometry that can be tuned

#a = 0.085                        # circular vacuum vessel radius

a=0.065
# a=0.11

# Magnet geometry (in meters)
footprint   = .25  * 0.0254      # Length * Width = footprint, e.x 0.25 in -> .25 * 0.0254 
layer_thick = (1/16) * 0.0254   # Radial thickness per layer, e.x: 1/16 in -> (1/16) * 0.0254  
wall_thick  = 0.001             # Wall thickness between adjacent cells, m, e.x: .001 = 1mm
gap         = 0.000             # Radial gap from VV surface to first magnet face in case support structure is needed

# Radial layers requested (each tower is capped at or below this)
L_requested = 15

# --- Ports (optional) ---
# theta_port  : poloidal angle of the port center (rad)
#               0 -> outboard (+R), pi/2 -> top (+Z), pi -> inboard (-R)
# s_port      : position along the axis, NORMALIZED to [0, 1]
#               0 -> phi_start, 1 -> phi_end. A port at 0 or 1 is a HALF port;
#               it closes into a full port against its own image under symmetry.
# port_radius : port radius on the VV surface (m)
target_size = .305
R0_before = eq.compute('R0')['R0']
scale = target_size 
print(scale)
PORTS = [
    # {"theta_port": 0.0,   "s_port": 0.0, "port_radius": 0.1*scale},     # outboard port at phi_start 
    # {"theta_port": np.pi, "s_port": 0.5, "port_radius": 0.1*scale},     # inboard port, midway
    # {"theta_port": np.pi/2,   "s_port": 1.0, "port_radius": 0.1*scale}, # top port at, at phi_end
]


# Neodymium magnetization: N52
Br    = 1.465
mu0   = 4 * np.pi * 1e-7
M_mag = Br / mu0                # A/m

# Output file
output_file = "torsion_grid_circular.focus"


# =============================================================================
# DERIVED GEOMETRY
# =============================================================================

cell_spacing = footprint + wall_thick   # center-to-center tower spacing
e_Z          = np.array([0.0, 0.0, 1.0])

# --- Spline the axis, parameterized by its own arc length ---
seg    = np.linalg.norm(np.diff(axis_xyz, axis=0), axis=1)
s_raw  = np.concatenate([[0.0], np.cumsum(seg)])
L_axis = s_raw[-1]

csx = CubicSpline(s_raw, axis_xyz[:, 0])
csy = CubicSpline(s_raw, axis_xyz[:, 1])
csz = CubicSpline(s_raw, axis_xyz[:, 2])

# First derivatives -> axis tangent (used to build the normal-plane frame)
d1x = csx.derivative(1)
d1y = csy.derivative(1)
d1z = csz.derivative(1)

# --- M: poloidal tower count (global; constant cross-section radius) ---
M = int(np.floor(2 * np.pi * a / cell_spacing))
if M < 1:
    sys.exit(f"ERROR: cell_spacing ({cell_spacing*1000:.2f} mm) too large, M={M}.")
theta_arr = np.linspace(0, 2 * np.pi, M, endpoint=False)

# --- Fine sampling of the axis, used to measure each toroidal curve's length ---
N_FINE = 4000
s_fine = np.linspace(0.0, L_axis, N_FINE)
ax_f   = np.stack([csx(s_fine), csy(s_fine), csz(s_fine)], axis=1)


def normal_frame(s):
    """Orthonormal pair (e1, e2) spanning the plane PERPENDICULAR to the axis
    tangent at axis arc length s.

    e1 is e_R projected into the normal plane and renormalized, so theta = 0
    still points outboard. e2 = T x e1 completes the right-handed pair. Purely
    local -> no accumulated twist, no seam holonomy under symmetry.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)

    T = np.stack([d1x(s), d1y(s), d1z(s)], axis=1)
    T /= np.linalg.norm(T, axis=1, keepdims=True)

    pos = np.stack([csx(s), csy(s), csz(s)], axis=1)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    e_R = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)

    # Project e_R into the normal plane.
    e1  = e_R - np.sum(e_R * T, axis=1, keepdims=True) * T
    mag = np.linalg.norm(e1, axis=1, keepdims=True)

    # Degenerate only if the axis runs purely radially (T parallel to e_R).
    bad = mag[:, 0] < 1e-9
    if np.any(bad):
        ez_b     = np.broadcast_to(e_Z, (int(bad.sum()), 3))
        e1[bad]  = ez_b - np.sum(ez_b * T[bad], axis=1, keepdims=True) * T[bad]
        mag      = np.linalg.norm(e1, axis=1, keepdims=True)

    e1 = e1 / mag
    e2 = np.cross(T, e1)
    return e1, e2


def radial_dir(theta, e1, e2):
    """Outward unit direction at poloidal angle theta, in the normal plane.

    theta = 0    -> outboard (e_R projected into the normal plane)
    theta = pi/2 -> e2
    theta = pi   -> inboard
    """
    return np.cos(theta) * e1 + np.sin(theta) * e2


# Normal-plane frame along the fine axis sampling (needs normal_frame above).
e1_f, e2_f = normal_frame(s_fine)

# --- Self-intersection guard (specific to normal-plane cuts) ---
# A tube of radius r swept in the normal plane folds in on itself on the concave
# side wherever r exceeds the local radius of curvature rho = 1/kappa. The RZ
# version cannot do this; the normal-plane version can, so check it.
_d2x, _d2y, _d2z = csx.derivative(2), csy.derivative(2), csz.derivative(2)
_r1  = np.stack([d1x(s_fine),  d1y(s_fine),  d1z(s_fine)],  axis=1)
_r2  = np.stack([_d2x(s_fine), _d2y(s_fine), _d2z(s_fine)], axis=1)
_sp  = np.linalg.norm(_r1, axis=1)
_kap = np.linalg.norm(np.cross(_r1, _r2), axis=1) / _sp**3
rho_min = 1.0 / max(_kap.max(), 1e-12)

r_outer = a + gap + L_requested * layer_thick     # outermost possible layer face
if a >= rho_min:
    sys.exit(f"ERROR: minor radius a ({a:.4f} m) exceeds the minimum radius of "
             f"curvature ({rho_min:.4f} m); the VV surface self-intersects.")
if r_outer >= rho_min:
    print(f"*** WARNING: outermost layer radius {r_outer:.4f} m exceeds the minimum "
          f"radius of curvature {rho_min:.4f} m -- towers fold over on the concave side ***")


def build_toroidal_curve(theta):
    """Lay out the towers of one toroidal curve (one poloidal angle theta).

    Displace the whole axis by `a` in the theta direction to form the toroidal
    curve, measure its arc length S, fit N = floor(S / cell_spacing) towers
    along it at half-step offsets, and return each tower's base point on the
    VV surface plus its outward build direction.

    Returns
    -------
    base : (N,3)  tower base points, on the VV surface
    out  : (N,3)  outward build direction at each tower (unit)
    S    : float  arc length of this toroidal curve
    s_at : (N,)   axis arc length at each tower (common toroidal coordinate)
    """
    # The toroidal curve, and its cumulative arc length.
    curve  = ax_f + a * radial_dir(theta, e1_f, e2_f)
    d      = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(d)])
    S      = cumlen[-1]

    # How many towers fit, and where they sit along the curve.
    N       = max(1, int(np.floor(S / cell_spacing)))
    targets = (np.arange(N) + 0.5) * (S / N)      # half-step -> seam clearance
    s_at    = np.interp(targets, cumlen, s_fine)  # back to axis arc-length

    # Rebuild the normal-plane frame at those exact locations.
    cen      = np.stack([csx(s_at), csy(s_at), csz(s_at)], axis=1)
    e1, e2   = normal_frame(s_at)
    out      = radial_dir(theta, e1, e2)
    base     = cen + a * out
    return base, out, S, s_at


# =============================================================================
# PORT HELPERS
# =============================================================================

def angle_diff(a1, a2):
    """Minimum angular distance between two angles, result in [0, pi]."""
    d = abs(a1 - a2) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


def tower_in_port(theta, s):
    """Exclude a tower if its (theta, s) location is within port_radius of a
    port center, measured in local surface arc-length (flat approximation).
    `s` is the axis arc length at the tower; `s_port` is NORMALIZED to [0,1]."""
    for p in PORTS:
        dtheta = angle_diff(theta, p["theta_port"])
        ds     = abs(s - p["s_port"] * L_axis)
        dist   = np.sqrt((a * dtheta) ** 2 + ds ** 2)
        if dist <= p["port_radius"]:
            return True
    return False


# =============================================================================
# GRID GENERATION  (global layer-by-layer growth)
# =============================================================================
# Every tower is grown together, one layer at a time. Before a layer is placed,
# we check each still-growing tower's candidate magnet against ALL magnets already
# placed (any tower) plus the other candidates being placed on this same layer.
# If a candidate would come within (footprint + wall/2) of any of them, that
# tower is frozen -- it stops at its current height and places no more layers.
# A per-tower "growing" mask is updated every layer. This guarantees the final
# grid has no cross-tower collisions, and each tower is as tall as actually fits.

from scipy.spatial import cKDTree

vol_per_dipole = footprint * footprint * layer_thick
M_0            = M_mag * vol_per_dipole      # max dipole moment (A*m^2)
collision_dist = footprint + wall_thick / 2.0

# --- Collect every tower base + outward dir + (theta, s) metadata ---
bases   = []
outs    = []
th_all  = []
s_all   = []
N_per_curve = np.zeros(M, dtype=int)

for m_idx, theta in enumerate(theta_arr):
    base, out, S, s_at = build_toroidal_curve(theta)
    N_per_curve[m_idx] = len(base)
    for i in range(len(base)):
        bases.append(base[i]); outs.append(out[i])
        th_all.append(theta);  s_all.append(s_at[i])

bases  = np.asarray(bases)          # (T,3) tower base points
outs   = np.asarray(outs)           # (T,3) outward unit dirs
th_all = np.asarray(th_all)         # (T,)
s_all  = np.asarray(s_all)          # (T,)
T      = len(bases)

# --- Grow all towers together, layer by layer ---
placed_xyz   = []                   # centers of all placed magnets
placed_tower = []                   # tower index of each placed magnet
placed_layer = []                   # layer index of each placed magnet

L_tower = np.zeros(T, dtype=int)    # final height of each tower
growing = np.ones(T, dtype=bool)    # towers still allowed to add a layer

for l in range(L_requested):
    if not growing.any():
        break

    idx  = np.where(growing)[0]                       # towers trying to add layer l
    cand = bases[idx] + outs[idx] * (gap + (l + 0.5) * layer_thick)   # (K,3)

    # Distances from candidates to already-placed magnets in OTHER towers.
    # (A tower's own stack sits directly below its candidate -- layer_thick away --
    # and must never count as a collision, so exclude same-tower placed magnets.)
    ok = np.ones(len(idx), dtype=bool)
    if placed_xyz:
        placed_arr    = np.asarray(placed_xyz)
        placed_towarr = np.asarray(placed_tower)
        tree = cKDTree(placed_arr)
        kq = min(len(placed_arr), 8)
        dq, iq = tree.query(cand, k=kq)
        if kq == 1:
            dq = dq[:, None]; iq = iq[:, None]
        cand_tower = idx                              # tower id of each candidate
        same       = placed_towarr[iq] == cand_tower[:, None]
        dq_other   = np.where(same, np.inf, dq)       # ignore same-tower magnets
        dmin_other = dq_other.min(axis=1)
        ok &= dmin_other >= collision_dist

    # Candidates on THIS layer can also collide with each other. A candidate is
    # only its own tower's continuation of the stack directly below it (that
    # neighbor is exactly one layer_thick away and is fine); the conflict is with
    # OTHER towers' candidates on this same layer.
    if len(idx) > 1:
        ctree = cKDTree(cand)
        pairs = ctree.query_pairs(r=collision_dist, output_type='ndarray')
        if len(pairs) > 0:
            # both members of each too-close candidate pair are rejected this layer
            bad = np.unique(pairs.ravel())
            ok[bad] = False

    place = idx[ok]                                   # towers that place layer l
    stop  = idx[~ok]                                  # towers frozen at layer l

    for t in place:
        p = bases[t] + outs[t] * (gap + (l + 0.5) * layer_thick)
        placed_xyz.append(p); placed_tower.append(int(t)); placed_layer.append(l)
        L_tower[t] = l + 1

    growing[stop] = False          # frozen towers never grow again

L_all = L_tower.copy()

# --- Emit dipoles from the grown towers ---
dipoles  = []
tower_id = []
layer_id = []
n_blocked_towers  = 0
n_blocked_dipoles = 0

placed_xyz   = np.asarray(placed_xyz)
placed_tower = np.asarray(placed_tower)
placed_layer = np.asarray(placed_layer)

# port status per tower
port_blocked = np.array([tower_in_port(th_all[t], s_all[t]) for t in range(T)])
n_blocked_towers = int(port_blocked.sum())

# order magnets by tower then layer for a tidy file
order = np.lexsort((placed_layer, placed_tower))
for k in order:
    t = int(placed_tower[k]); l = int(placed_layer[k])
    u = outs[t]
    ox, oy, oz = -u
    mt = np.arccos(np.clip(oz, -1.0, 1.0))
    mp = np.arctan2(oy, ox)
    x, y, z = placed_xyz[k]
    Ic = 0 if port_blocked[t] else 1
    if port_blocked[t]:
        n_blocked_dipoles += 1
    dipoles.append((x, y, z, Ic, M_0, 0.0, 1, mp, mt))
    tower_id.append(t)
    layer_id.append(l)

# =============================================================================
# SUMMARY
# =============================================================================
print("=== Torsioned Circular-Axis Dipole Grid Generator ===")
print(f"Axis arc length: {L_axis:.4f} m")
print(f"Minor radius a:  {a*100:.1f} cm")
print(f"Min radius curv: {rho_min:.4f} m  (rho_min/a = {rho_min/a:.2f})")
print(f"Magnet:          footprint={footprint*1000:.3f} mm, layer={layer_thick*1000:.4f} mm")
print(f"Pitch:           {cell_spacing*1000:.3f} mm  (footprint + wall)")
print(f"Gap offset:      {gap*1000:.2f} mm")
print(f"M poloidal:      {M}  (global, from 2*pi*a / cell_spacing)")
print(f"N per curve:     {N_per_curve.min()} (inboard) to {N_per_curve.max()} (outboard)")
print(f"L per tower:     {L_all.min()} to {L_all.max()}  (mean {L_all.mean():.2f}, requested {L_requested})")
print(f"M_0 per dipole:  {M_0:.4e} A*m^2")
print(f"Volume/dipole:   {vol_per_dipole:.4e} m^3")
print(f"Total dipoles:   {len(dipoles)}")
print(f"Ports defined:   {len(PORTS)}")
print(f"Port-blocked towers:   {n_blocked_towers}")
print(f"Port-blocked dipoles:  {n_blocked_dipoles}  (Ic set to 0)")
print()


# =============================================================================
# NEAREST-NEIGHBOR COLLISION CHECK  (post-hoc, over the whole grid)
# =============================================================================
# The per-curve build check (curve_layer_counts) only compares a tower to its
# along-curve neighbors at the same layer. This is an independent, global check
# of the finished grid: for every magnet, look at its true nearest neighbors and
# flag any that belong to a DIFFERENT tower and sit closer than the collision
# distance. Same-tower magnets are identified and skipped by the fact that their
# separation is an integer multiple of layer_thick (stacked on the same line).
#
# Collision distance is footprint + wall_thick/2, i.e. half the wall may be
# consumed before it counts. The half-wall margin sits well outside the
# arc-vs-chord numerical deficit (~tens of microns) from placing towers at
# cell_spacing along curved toroidal curves, so it does not false-trigger on
# neighboring towers that are actually correctly spaced.

from scipy.spatial import cKDTree

xyz_all  = np.array([(d[0], d[1], d[2]) for d in dipoles])
tower_id = np.asarray(tower_id)
layer_id = np.asarray(layer_id)

collision_dist = footprint + wall_thick / 2.0

print("=== Nearest-neighbor collision check ===")
print(f"Collision distance:   {collision_dist*1000:.3f} mm  (footprint + wall/2)")

if len(xyz_all) < 2:
    print("Fewer than 2 magnets; nothing to check.")
    n_collisions = 0
    colliding_magnets = np.zeros(len(xyz_all), dtype=bool)
else:
    tree  = cKDTree(xyz_all)
    pairs = tree.query_pairs(r=collision_dist, output_type='ndarray')

    if len(pairs) == 0:
        n_collisions = 0
        colliding_magnets = np.zeros(len(xyz_all), dtype=bool)
    else:
        A, B = pairs[:, 0], pairs[:, 1]
        # A pair is a real collision only if the two magnets are in DIFFERENT
        # towers. (Same-tower pairs are the stacked layers of one tower; their
        # separation is a clean multiple of layer_thick and is never a problem.)
        cross_tower = tower_id[A] != tower_id[B]
        bad_pairs   = pairs[cross_tower]

        n_collisions = len(bad_pairs)
        colliding_magnets = np.zeros(len(xyz_all), dtype=bool)
        colliding_magnets[bad_pairs[:, 0]] = True
        colliding_magnets[bad_pairs[:, 1]] = True

if n_collisions == 0:
    print("No cross-tower collisions found. Grid is clear.")
else:
    d_bad = np.linalg.norm(xyz_all[bad_pairs[:, 0]] - xyz_all[bad_pairs[:, 1]], axis=1)
    w = int(np.argmin(d_bad))
    print(f"*** {n_collisions} cross-tower magnet pairs within the collision distance ***")
    print(f"    magnets involved:  {int(colliding_magnets.sum())}")
    print(f"    closest pair:      {d_bad.min()*1000:.3f} mm "
          f"(towers {tower_id[bad_pairs[w,0]]} and {tower_id[bad_pairs[w,1]]})")
print()


# =============================================================================
# WRITE FAMUS .focus FILE
# =============================================================================

N_total  = len(dipoles)
q        = 1
coiltype = 3
symmetry = 1

with open(output_file, 'w') as f:
    f.write('# Total number of coils,  momentq \n')
    f.write(f'     {N_total}     {q}\n')
    f.write('# coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt \n')
    for i, (x, y, z, Ic, M0, pho, Lc, mp, mt) in enumerate(dipoles):
        name = f"pm_{i:06d}"
        line = (f'{coiltype}, {symmetry}, {name}, '
                f'{x:.6e}, {y:.6e}, {z:.6e}, '
                f'{Ic}, {M0:.6e}, {pho:.6e}, '
                f'{Lc}, {mp:.6e}, {mt:.6e} \n')
        f.write(line)

print(f"Wrote {N_total} dipoles to '{output_file}'  "
      f"({N_total - n_blocked_dipoles} active, {n_blocked_dipoles} disabled by ports)")