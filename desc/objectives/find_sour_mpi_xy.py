from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

<<<<<<< HEAD
from .sources_dipoles_utils import (#_compute_magnetic_field_from_Current, 
                                    #_compute_magnetic_field_from_Current_Contour,
                                    omega_sour,
                                    compute_mask,
                                    )

from mpi4py import MPI
import numpy as np

# ---------------- Helper ----------------
def _split_indices(n_items, rank, size):
    """Return start/end indices for a contiguous block for MPI rank."""
=======
from .sources_dipoles_utils import omega_sour

from mpi4py import MPI
import numpy as np
import math

# ---------------- Helper ----------------
def _split_indices(n_items, rank, size):
    """Return start/end indices for a contiguous block for 'rank' among 'size' pieces."""
    if size <= 0:
        return 0, 0
>>>>>>> 0376ee84ca0189129de4820c23aaa9f9f2ccf1ba
    base, rem = divmod(n_items, size)
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end

<<<<<<< HEAD
# ---------------- Biot-Savart ----------------
def biot_savart_general_vec(re, rs, J, dV):
    """Compute Biot-Savart integral at evaluation points."""
    if J.ndim == 2:
        J = J[None, :, :]
        vectorized = False
    else:
        vectorized = True

    N = J.shape[0]
    re, rs, J, dV = map(lambda x: jnp.asarray(x, dtype=jnp.float64), (re, rs, J, dV))
    JdV = J * dV[:, None]
    B = jnp.zeros((N, 3, re.shape[0]), dtype=jnp.float64)

    def body(i, B):
        r = re - rs[i, :]
        JdV_i = JdV[:, i, :][:, None, :]
        num = jnp.cross(JdV_i, r, axis=-1)
        num = jnp.transpose(num, (0, 2, 1))
        den = jnp.linalg.norm(r, axis=-1) ** 3
        contrib = jnp.where(den[None, None, :] == 0, 0, num / den[None, None, :])
        return B + contrib

    B = 1e-7 * fori_loop(0, rs.shape[0], body, B)

    if not vectorized:
        B = B[0]
    return B

# ---------------- MPI-Aware Magnetic Field ----------------
def _compute_magnetic_field_from_Current_vec(Kgrid, K_at_grid, surface, data, coords,
                                             basis="rpz", mpi_comm=None, split_targets=True):
    """MPI-aware Biot-Savart with optional splitting of both sources and targets."""
=======

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
>>>>>>> 0376ee84ca0189129de4820c23aaa9f9f2ccf1ba
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

<<<<<<< HEAD
    coords = jnp.atleast_2d(coords.nodes if hasattr(coords, "nodes") else coords)
    if K_at_grid.ndim == 2:
        K_at_grid = K_at_grid[None, :, :]
        vectorized = False
    else:
        vectorized = True

    N = K_at_grid.shape[0]
    grid_rpz = data["x"]
    coords_xyz = rpz2xyz(coords) if basis=="rpz" else coords
    grid_xyz = rpz2xyz(grid_rpz)

    if basis != "rpz":
        K_at_grid = xyz2rpz_vec(K_at_grid, x=grid_xyz[:,0], y=grid_xyz[:,1])

    _rs = grid_rpz
    _K = K_at_grid
    _dV = Kgrid.weights * data["|e_theta x e_zeta|"] / Kgrid.NFP

    # --- Split sources ---
    start_src, end_src = _split_indices(_rs.shape[0], rank, size)
    local_rs = _rs[start_src:end_src, :]
    local_dV = _dV[start_src:end_src]
    local_K = _K[:, start_src:end_src, :]

    # --- Split targets optionally ---
    if split_targets:
        start_tgt, end_tgt = _split_indices(coords_xyz.shape[0], rank, size)
        local_coords = coords_xyz[start_tgt:end_tgt, :]
    else:
        local_coords = coords_xyz

    def nfp_loop_local(j, f):
        phi = (local_rs[:,1] + j * 2 * jnp.pi / Kgrid.NFP) % (2 * jnp.pi)
        rs_rpz = jnp.vstack((local_rs[:,0], phi, local_rs[:,2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(local_K, phi=phi)
        f += biot_savart_general_vec(local_coords, rs_xyz, K_xyz, local_dV)
        return f

    B_local = fori_loop(0, Kgrid.NFP, nfp_loop_local,
                        jnp.zeros((N,3,local_coords.shape[0])))

    B_local_np = np.asarray(B_local, dtype=np.float64)
    if split_targets:
        all_B = mpi_comm.allgather(B_local_np)
        B_global_np = np.concatenate(all_B, axis=-1)
    else:
        B_global_np = np.empty_like(B_local_np)
        mpi_comm.Allreduce(B_local_np, B_global_np, op=MPI.SUM)

    B = jnp.asarray(B_global_np)
    if basis=="rpz":
        B = xyz2rpz_vec(jnp.transpose(B,(0,2,1)), x=coords_xyz[:,0], y=coords_xyz[:,1])
    if not vectorized:
        B = B[0]
    return B

def _compute_magnetic_field_from_Current_Contour_vec(Kgrid, K_at_grid, surface, data, coords,
                                                     basis="rpz", mpi_comm=None, split_targets=True):
    """MPI-aware Biot-Savart from contour currents with 2D parallelization."""
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    coords = jnp.atleast_2d(coords.nodes if hasattr(coords, "nodes") else coords)
    if K_at_grid.ndim==2:
        K_at_grid = K_at_grid[None,:,:]
        vectorized=False
    else:
        vectorized=True
    N = K_at_grid.shape[0]

    grid_rpz = data["x"]
    coords_xyz = rpz2xyz(coords) if basis=="rpz" else coords
    grid_xyz = rpz2xyz(grid_rpz)
    if basis != "rpz":
        K_at_grid = xyz2rpz_vec(K_at_grid, x=grid_xyz[:,0], y=grid_xyz[:,1])

    _rs = grid_rpz
    _K = K_at_grid
    _dV = Kgrid.weights * jnp.sqrt(dot(data["e_theta"], data["e_theta"])) / Kgrid.NFP

    # --- Split sources ---
    start_src, end_src = _split_indices(_rs.shape[0], rank, size)
    local_rs = _rs[start_src:end_src,:]
    local_dV = _dV[start_src:end_src]
    local_K = _K[:, start_src:end_src,:]

    # --- Split targets optionally ---
    if split_targets:
        start_tgt, end_tgt = _split_indices(coords_xyz.shape[0], rank, size)
        local_coords = coords_xyz[start_tgt:end_tgt, :]
    else:
        local_coords = coords_xyz

    def nfp_loop_local(j,f):
        phi = (local_rs[:,1] + j*2*jnp.pi/Kgrid.NFP) % (2*jnp.pi)
        rs_rpz = jnp.vstack((local_rs[:,0], phi, local_rs[:,2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(local_K, phi=phi)
        f += biot_savart_general_vec(local_coords, rs_xyz, K_xyz, local_dV)
        return f

    B_local = fori_loop(0, Kgrid.NFP, nfp_loop_local,
                        jnp.zeros((N,3,local_coords.shape[0])))
    B_local_np = np.asarray(B_local, dtype=np.float64)

    if split_targets:
        all_B = mpi_comm.allgather(B_local_np)
        B_global_np = np.concatenate(all_B, axis=-1)
    else:
        B_global_np = np.empty_like(B_local_np)
        mpi_comm.Allreduce(B_local_np, B_global_np, op=MPI.SUM)

    B = jnp.asarray(B_global_np)
    if basis=="rpz":
        B = xyz2rpz_vec(jnp.transpose(B,(0,2,1)), x=coords_xyz[:,0], y=coords_xyz[:,1])
    if not vectorized:
        B = B[0]
    return B

# ---------------- Stick / B_sticks_vec ----------------
def stick(p2_, p1_, plasma_points, surface_grid, basis="rpz"):
    def nfp_loop(j,f):
        phi2 = (p2_[:,2] + j*2*jnp.pi/surface_grid.NFP) % (2*jnp.pi)
        p2s = jnp.stack([p2_[:,0], phi2, p2_[:,2]], axis=1)
        p2s = rpz2xyz(p2s)
        a_s = p2s[:,None,:] - p1_[:,None,:]
        b_s = p1_[:,None,:] - plasma_points[None,:,:]
        c_s = p2s[:,None,:] - plasma_points[None,:,:]
        c_sxa_s = cross(c_s, a_s)
        f += (1e-7 * ( ( jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=2), a_min=1e-8) *
                         jnp.sum(c_s*c_s, axis=2)**0.5 )**-1 *
                        (jnp.sum(a_s*c_s, axis=2)-jnp.sum(a_s*b_s, axis=2)) )[:,:,None] * c_sxa_s )
        return f
    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop,
                        jnp.zeros((p1_.shape[0], plasma_points.shape[0], plasma_points.shape[1])))
    if basis=="rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:,0], y=plasma_points[:,1])
    return b_stick

def B_sticks_vec(sgrid, surface, coords, ss_data, mpi_comm=None, split_targets=True):
    """MPI-aware magnetic field from sticks."""
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    pls_points = rpz2xyz(coords)
    N_wires = ss_data["x"].shape[0]
    N_coords = pls_points.shape[0]

    # Split wires
    start_idx, end_idx = _split_indices(N_wires, rank, size)
    p1_local = 0 * ss_data["x"][start_idx:end_idx]
    p2_local = ss_data["x"][start_idx:end_idx]

    b_local = stick(p2_local, p1_local, pls_points, sgrid, basis="rpz")

    # Split targets if needed
    if split_targets:
        start_tgt, end_tgt = _split_indices(N_coords, rank, size)
        b_local_np = np.asarray(b_local)[:, start_tgt:end_tgt, :]
        all_B = mpi_comm.allgather(b_local_np)
        b_global_np = np.concatenate(all_B, axis=1)
    else:
        b_local_full = np.zeros((N_wires, N_coords, 3), dtype=np.float64)
        b_local_full[start_idx:end_idx, :, :] = np.asarray(b_local)
        b_global_np = np.empty_like(b_local_full)
        mpi_comm.Allreduce(b_local_full, b_global_np, op=MPI.SUM)

    return jnp.asarray(b_global_np)

# ---------------- K_sour_vec ----------------
def K_sour_vec(sdata1,sdata2,sdata3,sgrid,surface,N,d_0,tdata,ss_data):
    from .sources_dipoles_utils import omega_sour
    omega_sour_fun = (omega_sour(sdata1, ss_data["u_iso"], ss_data["v_iso"], N, d_0)
                      + omega_sour(sdata2, ss_data["u_iso"], ss_data["v_iso"], N, d_0)
                      + omega_sour(sdata3, ss_data["u_iso"], ss_data["v_iso"], N, d_0))
    K_sour_total = ((-jnp.imag(omega_sour_fun)[:,None,:]*sdata1['e_v'][:,:,None]
                     + jnp.imag(omega_sour_fun)[:,None,:]*sdata1['e_u'][:,:,None])
                     * (sdata1["lambda_iso"]**-1)[:,None,None])
    return K_sour_total

# ---------------- MPI-aware bn_res_vec ----------------
def bn_res_vec_mpi(sdata1,sdata2,sdata3,sgrid,surface,N,d_0,coords,tdata,
                   contour_data,stick_data,contour_grid,ss_data,AAA, mpi_comm=None, split_targets=True):
    """MPI-aware Biot-Savart residual vector."""
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD

    B_sour0 = _compute_magnetic_field_from_Current_vec(
        sgrid,
        jnp.transpose(K_sour_vec(sdata1,sdata2,sdata3,sgrid,surface,N,d_0,tdata,ss_data),(2,0,1)),
        surface,
        sdata1,
        coords,
        basis="rpz",
        mpi_comm=mpi_comm,
        split_targets=split_targets
    )

    B_wire_cont = _compute_magnetic_field_from_Current_Contour_vec(
        contour_grid,
        jnp.transpose(AAA,(2,0,1)),
        surface,
        contour_data,
        coords,
        basis="rpz",
        mpi_comm=mpi_comm,
        split_targets=split_targets
    )

    B_sticks0 = B_sticks_vec(sgrid, surface, coords, stick_data, mpi_comm=mpi_comm, split_targets=split_targets)

    B_total = jnp.transpose(B_sour0 + B_wire_cont + B_sticks0, (1,2,0))
    return jnp.concatenate((B_total[:,0,:], B_total[:,1,:], B_total[:,2,:]))
=======
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
        K_xyz_full = rpz2xyz_vec(K_at_grid, phi=grid_rpz[:, 1])  # (N, M, 3)
    else:
        K_xyz_full = K_at_grid                                    # already XYZ

    # Geometric weight for the Biot-Savart integral
    dV_full = Kgrid.weights * data["|e_theta x e_zeta|"] / Kgrid.NFP  # (M,)

    # Split along sources (M) using 2D decomposition
    P_M, P_N = _factorize_2d(size)
    i_M = rank % P_M
    i_N = rank // P_M  # Not used for N splitting

    sM, eM = _split_indices(M, i_M, P_M)
    M_loc = eM - sM
    rs_rpz_local = grid_rpz[sM:eM, :]             # (M_loc, 3)
    dV_local = dV_full[sM:eM]                     # (M_loc,)
    K_xyz_M = K_xyz_full[:, sM:eM, :]             # (N, M_loc, 3)

    # Split along N using all ranks (1D decomposition)
    sN, eN = _split_indices(N, rank, size)  # Use size instead of P_N
    N_loc = eN - sN
    K_xyz = K_xyz_M[sN:eN, :, :]                  # (N_loc, M_loc, 3)

    # Debug print
    print(f"[Rank {rank}] N split: sN={sN}, eN={eN}, N_loc={N_loc}, total N={N}")

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
    return jnp.transpose(B_xyz, (1, 2, 0))  # (3, R, N_loc)


# ---------------- Field from contour current (AAA) ----------------
def _compute_magnetic_field_from_Current_Contour_vec(
    Kgrid, K_at_grid, surface, data, coords,
    basis="rpz", mpi_comm=None,
):
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

    # Split along sources (M) using 2D decomposition
    P_M, P_N = _factorize_2d(size)
    i_M = rank % P_M
    i_N = rank // P_M  # Not used for N splitting

    sM, eM = _split_indices(M, i_M, P_M)
    M_loc = eM - sM
    rs_rpz_local = grid_rpz[sM:eM, :]
    dl_local = dl_full[sM:eM]
    K_xyz_M = K_xyz_full[:, sM:eM, :]

    # Split along N using all ranks (1D decomposition)
    sN, eN = _split_indices(N, rank, size)  # Use size instead of P_N
    N_loc = eN - sN
    K_xyz = K_xyz_M[sN:eN, :, :]

    # Debug print
    print(f"[Rank {rank}] N split: sN={sN}, eN={eN}, N_loc={N_loc}, total N={N}")

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

def stick(
    p2_,  # second point of the stick
    p1_,  # first point of the stick
    plasma_points,  # points on the plasma surface
    surface_grid,  # Kgrid,
    basis="rpz",
):
    """Computes the magnetic field on the plasma surface due to a unit current on the source wires.
    
        p2_: numpy.ndarray of dimension (N, 3)
        p1_: numpy.ndarray of dimension (N, 3)
        plasma_point: numpy.ndarray of dimension (M, 3)
    
    """
    
    #basis="rpz"
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi2 = (p2_[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)

        # TODO: Make sure p2s has the shape (N, 3)
        p2s = jnp.stack([p2_[:, 0], phi2, p2_[:, 2]], axis=1)

        #print(p2s.shape)
        p2s = rpz2xyz(p2s)

        # a_s.shape = b_s.shape = c_s.shape = (N, M, 3)
        a_s = p2s[:, None, :] - p1_[:, None, :]
        b_s = p1_[:, None, :] - plasma_points[None, :, :]
        c_s = p2s[:, None, :] - plasma_points[None, :, :]

        # if c_s and a_s are (N, 3), will work fine
        c_sxa_s = cross(c_s, a_s)

        f += (
            1e-7
            * ( #(
                (
                    jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=2), a_min=1e-8, a_max=None)
                    * jnp.sum(c_s * c_s, axis=2) ** (1 / 2)
                )
                ** (-1)
                * (jnp.sum(a_s * c_s, axis=2) - jnp.sum(a_s * b_s, axis=2)) )[:, :, None]
                * c_sxa_s#.T
            #).T
        ) # (N, M, 3)
        
        return f

    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros((p1_.shape[0], plasma_points.shape[0], plasma_points.shape[1])))

    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:, 0], y=plasma_points[:, 1])

    return b_stick

def B_sticks_vec(sgrid,
    #surface,
    #y,
    coords,
    ss_data,
):

    pls_points = rpz2xyz(coords)  # eq_surf.compute(["x"], grid=Bgrid, basis="xyz")["x"]

    #r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    b_stick_fun = stick(ss_data["x"],  # Location of the wire at the theta = pi cut, variable zeta position
                                            0 * ss_data["x"],  # All wires at the center go to the origin
                                            pls_points,
                                            sgrid,
                                            basis="rpz",
                                            )

    return b_stick_fun
    
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


def bn_res_vec_mpi_2d(
    sdata1, sdata2, sdata3, sgrid, surface, N, d_0, coords, tdata,
    contour_data, stick_data, contour_grid, ss_data, AAA,
    mpi_comm=None,
):
    """
    MPI-ready Biot-Savart computation: returns local slices.
    Final B_total shape (3, coords.shape[0], N)
    """

    from mpi4py import MPI
    import jax.numpy as jnp

    comm = mpi_comm or MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --------- 1. Compute currents from sources ----------
    K_sour = K_sour_vec(sdata1, sdata2, sdata3, sgrid, surface, N, d_0, tdata, ss_data)
    K_sour_N_M_3 = jnp.transpose(K_sour, (2, 0, 1))  # (N, M, 3)

    B_sour0 = _compute_magnetic_field_from_Current_vec(
        sgrid, K_sour_N_M_3, surface, sdata1, coords,
        basis="rpz", mpi_comm=mpi_comm
    )  # (3, coords_local, N_slice)

    # --------- 2. Contour currents ----------
    AAA_N_M_3 = jnp.transpose(AAA, (2, 0, 1))
    B_wire_cont = _compute_magnetic_field_from_Current_Contour_vec(
        contour_grid, AAA_N_M_3, surface, contour_data, coords,
        basis="rpz", mpi_comm=mpi_comm
    )  # (3, coords_local, N_slice)

    # --------- 3. Sticks contribution ----------
    B_sticks0 = np.transpose(B_sticks_vec(sgrid,coords, stick_data,),(2,1,0)) #mpi_comm=mpi_comm)  # (3, coords, N)

    #print('Shape of Bsour: ' + str(B_sour0.shape))
    print('Shape of Bstick: ' + str(B_sticks0.shape))
    
    B_local = B_sour0 + B_wire_cont #+ B_sticks0 # (3, coords_local, N)
    
    # ---------- gather along N-axis ----------
    B_parts = comm.gather(np.array(B_local), root=0)

    if rank == 0:
        print(f"[Rank 0] B_parts shapes: {[part.shape for part in B_parts]}")
        B_full = np.concatenate(B_parts, axis=2)  # (3, coords.shape[0], N_fields)
        print(f"[Rank 0] Final B_full shape: {B_full.shape}, expected (3, {coords.shape[0]}, {K_sour.shape[2]})")
        return B_full + B_sticks0
    else:
        return None
>>>>>>> 0376ee84ca0189129de4820c23aaa9f9f2ccf1ba
