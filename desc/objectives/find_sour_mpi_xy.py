from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from .sources_dipoles_utils import omega_sour

from mpi4py import MPI
import numpy as np
import math

# ---------------- Helper ----------------
def _split_indices(n_items, rank, size):
    """Return start/end indices for a contiguous block for 'rank' among 'size' pieces."""
    if size <= 0:
        return 0, 0
    base, rem = divmod(n_items, size)
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end


def _factorize_2d(size):
    """Return P_M, P_N such that P_M * P_N = size and P_M ≈ P_N."""
    if size <= 0:
        return 1, 1
    P_M = int(np.floor(np.sqrt(size)))
    while P_M > 1 and size % P_M != 0:
        P_M -= 1
    P_N = size // P_M
    return max(P_M, 1), max(P_N, 1)


# ---------------- Biot-Savart core ----------------
def biot_savart_general_vec(re_xyz, rs_xyz, J_local, dV_local):
    """
    Core Biot-Savart accumulation.

    Inputs
    ------
    re_xyz     : (R, 3) evaluation points (XYZ)
    rs_xyz     : (M_loc, 3) source points (XYZ)
    J_local    : (N_loc, M_loc, 3) current density at sources (XYZ components)
    dV_local   : (M_loc,) volume/line element weights

    Returns
    -------
    B_local    : (N_loc, 3, R)
    """
    # Normalize shapes/types
    re_xyz = jnp.asarray(re_xyz, dtype=jnp.float64)
    rs_xyz = jnp.asarray(rs_xyz, dtype=jnp.float64)
    J_local = jnp.asarray(J_local, dtype=jnp.float64)
    dV_local = jnp.asarray(dV_local, dtype=jnp.float64)

    N_loc = J_local.shape[0]
    R = re_xyz.shape[0]
    M_loc = rs_xyz.shape[0]

    # Handle empty work quickly (rank may have no M or no N)
    if N_loc == 0 or M_loc == 0 or R == 0:
        return jnp.zeros((N_loc, 3, R), dtype=jnp.float64)

    # Pre-scale currents by dV
    JdV = J_local * dV_local[None, :, None]           # (N_loc, M_loc, 3)
    B = jnp.zeros((N_loc, 3, R), dtype=jnp.float64)

    def body(i, Bacc):
        # r_vec = r_eval - r_src_i  -> (R, 3)
        r = re_xyz - rs_xyz[i, :]
        # cross(J_i*dV_i, r) for all N_loc and all R
        JdV_i = JdV[:, i, :][:, None, :]              # (N_loc, 1, 3)
        num = jnp.cross(JdV_i, r, axis=-1)            # (N_loc, R, 3)
        num = jnp.transpose(num, (0, 2, 1))           # (N_loc, 3, R)
        den = jnp.linalg.norm(r, axis=-1) ** 3        # (R,)
        contrib = jnp.where(den[None, None, :] == 0, 0.0, num / den[None, None, :])
        return Bacc + contrib

    B = 1e-7 * fori_loop(0, M_loc, body, B)
    return B


# ---------------- Field from volumetric/surface current J (K_sour/AAA) ----------------
def _compute_magnetic_field_from_Current_vec(
    Kgrid, K_at_grid, surface, data, coords,
    basis="rpz", mpi_comm=None,
):
    """
    Compute B from general current on a grid, splitting along M (sources) and N (expansion index).

    Contract (per rank):
      returns B_local in XYZ basis shaped (3, coords.shape[0], N_local)

    Inputs
    ------
    K_at_grid : (N, M, 3)  J in **RPZ** components if basis=='rpz', else XYZ comps if basis!='rpz'
    coords    : grid-like or (R,3) RPZ points if basis=='rpz', else XYZ
    data      : dict with keys:
                  'x' -> (M,3) RPZ source locations
                  '|e_theta x e_zeta|' and tangent vectors for weights, etc.

    Splitting
    ---------
    World size is factorized as P_M * P_N. Rank index is mapped to (i_M, i_N).
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    # Normalize coords array
    coords_rpz = coords.nodes if hasattr(coords, "nodes") else coords
    coords_rpz = jnp.atleast_2d(coords_rpz)

    # Standardize input J array to 3D
    if K_at_grid.ndim == 2:
        K_at_grid = K_at_grid[None, :, :]     # (1, M, 3)
    K_at_grid = jnp.asarray(K_at_grid, dtype=jnp.float64)
    N, M, _ = K_at_grid.shape

    # Source & evaluation positions
    grid_rpz = data["x"]                       # (M, 3) in RPZ
    grid_xyz = rpz2xyz(grid_rpz)              # (M, 3)
    if basis == "rpz":
        coords_xyz = rpz2xyz(coords_rpz)      # (R, 3)
    else:
        coords_xyz = jnp.asarray(coords_rpz, dtype=jnp.float64)

    # Convert input current to XYZ if provided in RPZ basis
    if basis == "rpz":
        # 'phi' will vary by NFP later; here we keep base angles from grid_rpz
        K_xyz_full = rpz2xyz_vec(K_at_grid, phi=grid_rpz[:, 1])  # (N, M, 3)
    else:
        K_xyz_full = K_at_grid                                    # already XYZ

    # Geometric weight for the Biot-Savart integral
    dV_full = Kgrid.weights * data["|e_theta x e_zeta|"] / Kgrid.NFP  # (M,)

    # 2D MPI decomposition
    P_M, P_N = _factorize_2d(size)
    i_M = rank % P_M
    i_N = rank // P_M

    # Split along sources (M)
    sM, eM = _split_indices(M, i_M, P_M)
    M_loc = eM - sM
    rs_rpz_local = grid_rpz[sM:eM, :]             # (M_loc, 3)
    dV_local = dV_full[sM:eM]                     # (M_loc,)
    K_xyz_M = K_xyz_full[:, sM:eM, :]             # (N, M_loc, 3)

    # Split along expansion index (N)
    sN, eN = _split_indices(N, i_N, P_N)
    N_loc = eN - sN
    K_xyz = K_xyz_M[sN:eN, :, :]                  # (N_loc, M_loc, 3)

    # Handle empty work paths
    R = coords_xyz.shape[0]
    if N_loc == 0 or M_loc == 0:
        return jnp.zeros((3, R, 0), dtype=jnp.float64)

    # NFP loop: rotate phi, accumulate contribution
    def nfp_body(j, B_acc_xyz):
        phi_local = (rs_rpz_local[:, 1] + j * 2 * jnp.pi / Kgrid.NFP) % (2 * jnp.pi)
        rs_rpz_j = jnp.stack([rs_rpz_local[:, 0], phi_local, rs_rpz_local[:, 2]], axis=1)
        rs_xyz = rpz2xyz(rs_rpz_j)                              # (M_loc, 3)
        K_xyz_j = rpz2xyz_vec(K_xyz, phi=phi_local)             # (N_loc, M_loc, 3)
        B_Nloc_3_R = biot_savart_general_vec(coords_xyz, rs_xyz, K_xyz_j, dV_local)  # (N_loc,3,R)
        return B_acc_xyz + B_Nloc_3_R

    B_xyz = fori_loop(0, Kgrid.NFP, nfp_body, jnp.zeros((N_loc, 3, R), dtype=jnp.float64))
    # Return as (3, R, N_loc)
    return jnp.transpose(B_xyz, (1, 2, 0))


# ---------------- Field from contour current (AAA) ----------------
def _compute_magnetic_field_from_Current_Contour_vec(
    Kgrid, K_at_grid, surface, data, coords,
    basis="rpz", mpi_comm=None,
):
    """
    Same contract and splitting as _compute_magnetic_field_from_Current_vec,
    but with contour-element weight (||e_theta||) instead of area element.

    Returns per rank: (3, coords.shape[0], N_local)
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    coords_rpz = coords.nodes if hasattr(coords, "nodes") else coords
    coords_rpz = jnp.atleast_2d(coords_rpz)

    if K_at_grid.ndim == 2:
        K_at_grid = K_at_grid[None, :, :]
    K_at_grid = jnp.asarray(K_at_grid, dtype=jnp.float64)
    N, M, _ = K_at_grid.shape

    grid_rpz = data["x"]                 # (M, 3) RPZ points along contour
    grid_xyz = rpz2xyz(grid_rpz)
    if basis == "rpz":
        coords_xyz = rpz2xyz(coords_rpz)
    else:
        coords_xyz = jnp.asarray(coords_rpz, dtype=jnp.float64)

    if basis == "rpz":
        K_xyz_full = rpz2xyz_vec(K_at_grid, phi=grid_rpz[:, 1])   # (N, M, 3)
    else:
        K_xyz_full = K_at_grid

    # Contour metric (line element magnitude) instead of area element
    dl_full = Kgrid.weights * jnp.sqrt(dot(data["e_theta"], data["e_theta"])) / Kgrid.NFP  # (M,)

    # 2D split
    P_M, P_N = _factorize_2d(size)
    i_M = rank % P_M
    i_N = rank // P_M

    sM, eM = _split_indices(M, i_M, P_M)
    M_loc = eM - sM
    rs_rpz_local = grid_rpz[sM:eM, :]
    dl_local = dl_full[sM:eM]
    K_xyz_M = K_xyz_full[:, sM:eM, :]

    sN, eN = _split_indices(N, i_N, P_N)
    N_loc = eN - sN
    K_xyz = K_xyz_M[sN:eN, :, :]

    R = coords_xyz.shape[0]
    if N_loc == 0 or M_loc == 0:
        return jnp.zeros((3, R, 0), dtype=jnp.float64)

    def nfp_body(j, B_acc_xyz):
        phi_local = (rs_rpz_local[:, 1] + j * 2 * jnp.pi / Kgrid.NFP) % (2 * jnp.pi)
        rs_rpz_j = jnp.stack([rs_rpz_local[:, 0], phi_local, rs_rpz_local[:, 2]], axis=1)
        rs_xyz = rpz2xyz(rs_rpz_j)
        K_xyz_j = rpz2xyz_vec(K_xyz, phi=phi_local)
        B_Nloc_3_R = biot_savart_general_vec(coords_xyz, rs_xyz, K_xyz_j, dl_local)
        return B_acc_xyz + B_Nloc_3_R

    B_xyz = fori_loop(0, Kgrid.NFP, nfp_body, jnp.zeros((N_loc, 3, R), dtype=jnp.float64))
    return jnp.transpose(B_xyz, (1, 2, 0))  # (3, R, N_loc)


# ---------------- Sticks ----------------

def stick2(p2_, p1_, plasma_points, surface_grid, basis="rpz"):
    """
    Compute field of straight sticks (wires) replicated across NFP.
    Compatible with MPI-distributed computation.
    Returns array of shape (3, R, W_loc) in XYZ.
    """
    R = plasma_points.shape[0]      # number of target coordinates
    W_loc = p1_.shape[0]            # number of wires in this local slice

    def nfp_loop(j, f):
        # replicate along NFP
        phi2 = (p2_[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)
        p2s = jnp.stack([p2_[:, 0], phi2, p2_[:, 2]], axis=1)

        # convert to XYZ if basis=='rpz'
        if basis == "rpz":
            p2s = rpz2xyz(p2s)

        # compute vectors
        a_s = p2s[:, None, :] - p1_[:, None, :]           # (W_loc,1,3)
        b_s = p1_[:, None, :] - plasma_points[None, :, :] # (W_loc,R,3)
        c_s = p2s[:, None, :] - plasma_points[None, :, :] # (W_loc,R,3)
        c_sxa_s = cross(c_s, a_s)                         # (W_loc,R,3)

        # Biot-Savart factor
        denom = jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=2), a_min=1e-8) \
                * jnp.sqrt(jnp.sum(c_s * c_s, axis=2))   # (W_loc,R)
        factor = 1e-7 * ((jnp.sum(a_s * c_s, axis=2) - jnp.sum(a_s * b_s, axis=2)) / denom)
        f += factor[:, :, None] * c_sxa_s
        return f

    # run NFP loop
    B_sticks = fori_loop(0, surface_grid.NFP, nfp_loop,
                         jnp.zeros((W_loc, R, 3), dtype=jnp.float64))  # (W_loc,R,3)

    # transpose to match (3, R, W_loc) for consistency with B_sour and B_wire_cont
    B_sticks = jnp.transpose(B_sticks, (2, 1, 0))  # (3, R, W_loc)

    return B_sticks

def stick(p2_, p1_, plasma_points, surface_grid, basis="rpz"):
    """
    Field of straight sticks (wires) replicated across NFP.
    Returns (W_loc, R, 3) in XYZ if basis=='rpz' at coords.
    Original entries preserved.
    """
    R = plasma_points.shape[0]

    def nfp_loop(j, f):
        phi2 = (p2_[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)
        p2s = jnp.stack([p2_[:, 0], phi2, p2_[:, 2]], axis=1)
        p2s = rpz2xyz(p2s)
        a_s = p2s[:, None, :] - p1_[:, None, :]                  # (W_loc,1,3)
        b_s = p1_[:, None, :] - plasma_points[None, :, :]        # (W_loc,R,3)
        c_s = p2s[:, None, :] - plasma_points[None, :, :]        # (W_loc,R,3)
        c_sxa_s = cross(c_s, a_s)                                # (W_loc,R,3)

        denom = jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=2), a_min=1e-8) \
                * (jnp.sum(c_s * c_s, axis=2) ** 0.5)            # (W_loc,R)
        factor = 1e-7 * ((jnp.sum(a_s * c_s, axis=2) - jnp.sum(a_s * b_s, axis=2)) / denom)
        f += factor[:, :, None] * c_sxa_s
        return f

    b_stick = fori_loop(
        0, surface_grid.NFP, nfp_loop,
        jnp.zeros((p1_.shape[0], R, 3), dtype=jnp.float64)
    )
    return b_stick  # (W_loc, R, 3) XYZ


def B_sticks_vec(stick_data, coords, surface_grid, mpi_comm=None):
    """
    MPI-ready vectorized sticks computation.
    Original argument names preserved.
    Returns array of shape (3, coords.shape[0], num_wires)
    """
    comm = mpi_comm or MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine local coordinate slice
    N_coords = coords.shape[0]
    coords_per_rank = N_coords // size
    start = rank * coords_per_rank
    end = (rank + 1) * coords_per_rank if rank != size - 1 else N_coords
    coords_local = coords[start:end]

    # Compute local sticks contribution
    B_local = stick(
        stick_data['x'],
        0*stick_data['x'],
        coords_local,
        surface_grid,
        basis='rpz'
    )  # (num_wires_local, len(coords_local), 3)

    # Transpose to (3, coords_local, num_wires)
    B_local = jnp.transpose(B_local, (2, 1, 0))

    # Optional: gather local shapes/norms for verification
    local_shape = B_local.shape
    local_norm = jnp.linalg.norm(B_local)
    all_shapes = comm.gather(local_shape, root=0)
    all_norms = comm.gather(local_norm.item(), root=0)

    # Barrier to ensure all ranks finished
    comm.Barrier()

    return B_local#, all_shapes, all_norms

# ---------------- K_sour ----------------
def K_sour_vec(sdata1, sdata2, sdata3, sgrid, surface, N, d_0, tdata, ss_data):
    """
    Produce K_sour in RPZ components of shape (N, M, 3) **after** transpose in caller.
    """
    omega_sour_fun = (omega_sour(sdata1, ss_data["u_iso"], ss_data["v_iso"], N, d_0)
                      + omega_sour(sdata2, ss_data["u_iso"], ss_data["v_iso"], N, d_0)
                      + omega_sour(sdata3, ss_data["u_iso"], ss_data["v_iso"], N, d_0))
    # RPZ components
    K_sour_total = ((-jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_v'][:, :, None]
                     +  jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_u'][:, :, None])
                    * (sdata1["lambda_iso"] ** -1)[:, None, None])
    return K_sour_total  # (M, 3) injected along second axis; caller transposes to (N, M, 3)


# ---------------- MPI-aware bn_res_vec (2D split over M and N) ----------------
def bn_res_vec_mpi_2d(
    sdata1, sdata2, sdata3, sgrid, surface, N, d_0, coords, tdata,
    contour_data, stick_data, contour_grid, ss_data, AAA,
    mpi_comm=None,
):
    """
    Build the residual matrix columns on each rank and return **local slice** only.

    Per-rank outputs:
      B_local_flat : (coords.shape[0] * 3, N_local)

    Assembly is done by the caller (rank 0) via concatenation along axis=1.
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    # ---------- currents from sources (K_sour) ----------
    K_sour = K_sour_vec(sdata1, sdata2, sdata3, sgrid, surface, N, d_0, tdata, ss_data)
    # Your previous call used transpose (2,0,1) -> target (N, M, 3)
    K_sour_N_M_3 = jnp.transpose(K_sour, (2, 0, 1))

    B_sour0 = _compute_magnetic_field_from_Current_vec(
        sgrid, K_sour_N_M_3, surface, sdata1, coords,
        basis="rpz", mpi_comm=mpi_comm
    )  # (3, R, N_loc)

    # ---------- contour currents (AAA) ----------
    AAA_N_M_3 = jnp.transpose(AAA, (2, 0, 1))   # ensure (N, M, 3)
    B_wire_cont = _compute_magnetic_field_from_Current_Contour_vec(
        contour_grid, AAA_N_M_3, surface, contour_data, coords,
        basis="rpz", mpi_comm=mpi_comm
    )  # (3, R, N_loc)

    B_sticks0 = B_sticks_vec(#sgrid, 
                             stick_data, coords, surface,
                             mpi_comm=mpi_comm,)# split_targets=split_targets)
    
    B_sour0 = jnp.transpose(B_sour0, (0,2,1))      # (3, coords, N)
    B_wire_cont = jnp.transpose(B_wire_cont, (0,2,1))
    B_sticks0 = jnp.transpose(B_sticks0, (0,2,1))
    
    # --- Combine locally ---
    B_local = B_sour0 + B_wire_cont + B_sticks0   # still (3, coords, N)
    
    # --- Optional: compute local norms for verification ---
    local_shape = B_local.shape
    local_norm = jnp.linalg.norm(B_local)
    
    # --- Gather only shapes and norms, not full matrix ---
    all_shapes = mpi_comm.gather(local_shape, root=0)
    all_norms = mpi_comm.gather(local_norm.item(), root=0)
    
    # --- Flatten/reshape to standard 3D for consistency ---
    # For example, keep (3, coords.shape[0], N) if needed
    #B_local = jnp.reshape(B_local, (3, coords.shape[0], N))
    
    # --- Barrier to ensure all ranks finished ---
    if mpi_comm is not None:
        mpi_comm.Barrier()
    
    # --- Return the local array along with shapes/norms for verification ---
    return B_local, all_shapes, all_norms