from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from .sources_dipoles_utils import (omega_sour, compute_mask)

from mpi4py import MPI
import numpy as np
import math

# ---------------- Helpers ----------------
def _split_indices(n_items, rank_index, n_parts):
    """Return start,end indices for a contiguous partition index in [0..n_parts-1]."""
    base, rem = divmod(n_items, n_parts)
    if rank_index < rem:
        start = rank_index * (base + 1)
        end = start + (base + 1)
    else:
        start = rem * (base + 1) + (rank_index - rem) * base
        end = start + base
    return int(start), int(end)

def _factorize_2d(num_ranks):
    """Choose P_M,P_N such that P_M * P_N = num_ranks and P_M <= P_N and close to sqrt."""
    if num_ranks <= 1:
        return 1, 1
    P_M = int(math.floor(math.sqrt(num_ranks)))
    while P_M > 1 and (num_ranks % P_M) != 0:
        P_M -= 1
    if num_ranks % P_M != 0:
        P_M = 1
    P_N = num_ranks // P_M
    return int(P_M), int(P_N)

# ---------------- Biot-Savart kernel (unchanged) ----------------
def biot_savart_general_vec(re, rs, J, dV):
    """Compute Biot-Savart integral at evaluation points.
    re: (T,3), rs: (M_local,3), J: (N_local,M_local,3) or (1,M_local,3), dV: (M_local,)
    Returns B: (N_local,3,T_local)
    """
    if J.ndim == 2:
        J = J[None, :, :]
        vectorized = False
    else:
        vectorized = True

    re, rs, J, dV = map(lambda x: jnp.asarray(x, dtype=jnp.float64), (re, rs, J, dV))
    JdV = J * dV[:, None]  # (N_local, M_local, 3)
    B = jnp.zeros((J.shape[0], 3, re.shape[0]), dtype=jnp.float64)

    def body(i, B):
        r = re - rs[i, :]  # (T_local,3)
        JdV_i = JdV[:, i, :][:, None, :]  # (N_local,1,3)
        num = jnp.cross(JdV_i, r, axis=-1)  # (N_local, T_local, 3)
        num = jnp.transpose(num, (0, 2, 1))  # (N_local,3,T_local)
        den = jnp.linalg.norm(r, axis=-1) ** 3  # (T_local,)
        contrib = jnp.where(den[None, None, :] == 0, 0, num / den[None, None, :])
        return B + contrib

    B = 1e-7 * fori_loop(0, rs.shape[0], body, B)
    if not vectorized:
        B = B[0]
    return B

# ---------------- MPI-aware 2D-split functions ----------------
def _compute_magnetic_field_from_Current_vec_2d(
    Kgrid, K_at_grid, surface, data, coords,
    basis="rpz", mpi_comm=None, split_targets=True
):
    """
    Full 2D decomposition: split both M (sources) and N (expansion) across a P_M x P_N grid of ranks.
    Root (rank 0) collects and assembles full B_global (N,3,T) by summing contributions from all M partitions.
    Returns:
      - on rank 0: jnp.asarray(B_global) with shape (N,3,T)
      - on other ranks: None
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    world_rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()

    # coords -> array (T,3)
    if hasattr(coords, "nodes"):
        coords_arr = jnp.atleast_2d(coords.nodes)
    else:
        coords_arr = jnp.atleast_2d(coords)
    T = int(coords_arr.shape[0])

    K = jnp.asarray(K_at_grid)
    if K.ndim == 2:
        K = K[None, :, :]
        vectorized = False
    else:
        vectorized = True

    N = int(K.shape[0])  # total expansion length
    M = int(K.shape[1])  # total number of source nodes

    grid_rpz = data["x"]
    grid_xyz = rpz2xyz(grid_rpz)
    coords_xyz = rpz2xyz(coords_arr) if basis == "rpz" else coords_arr

    if basis != "rpz":
        # Convert incoming K (in xyz) to rpz basis keyed to grid_xyz
        K = xyz2rpz_vec(K, x=grid_xyz[:, 0], y=grid_xyz[:, 1])

    _rs = grid_rpz  # (M,3)
    _dV = Kgrid.weights * data["|e_theta x e_zeta|"] / Kgrid.NFP  # (M,)
    _K = K  # (N,M,3)

    # Factorize ranks into P_M x P_N
    P_M, P_N = _factorize_2d(world_size)  # P_M * P_N == world_size
    if P_M * P_N != world_size:
        # fallback to 1D splitting if factorization failed
        P_M, P_N = world_size, 1

    # Map world_rank -> (i_M, i_N)
    i_M = world_rank % P_M
    i_N = world_rank // P_M

    # Each rank has contiguous M_local (sources) and contiguous N_local (expansion indices)
    start_M, end_M = _split_indices(M, i_M, P_M)
    start_N, end_N = _split_indices(N, i_N, P_N)
    M_local = end_M - start_M
    N_local = end_N - start_N

    # Targets: split by world_rank if requested (simple 1D split)
    if split_targets:
        start_T, end_T = _split_indices(T, world_rank, world_size)
        T_local = end_T - start_T
        coords_local = coords_xyz[start_T:end_T, :]
    else:
        start_T, end_T = 0, T
        T_local = T
        coords_local = coords_xyz

    # Handle empty-slice cases: produce zero (N_local,3,T_local)
    if M_local <= 0 or N_local <= 0 or T_local <= 0:
        B_local_np = np.zeros((N_local, 3, T_local), dtype=np.float64)
        # prepare metadata for root assembly
        meta = (start_N, end_N, start_T, end_T, world_rank)
        gathered = mpi_comm.gather((meta, B_local_np), root=0)
        if world_rank == 0:
            # assemble from gathered blocks
            B_global = np.zeros((N, 3, T), dtype=np.float64)
            for (meta_k, arr_k) in gathered:
                sN, eN, sT, eT, rnk = meta_k
                if arr_k.size == 0:
                    continue
                B_global[sN:eN, :, sT:eT] += arr_k
            return jnp.asarray(B_global)
        else:
            return None

    # Extract local slices
    rs_local = _rs[start_M:end_M, :]                 # (M_local,3)
    dV_local = np.asarray(_dV[start_M:end_M], dtype=np.float64)  # (M_local,)
    K_local = _K[start_N:end_N, start_M:end_M, :]    # (N_local, M_local, 3)

    # compute local contribution: for each j in NFP do phi shift, compute Biot-Savart
    def nfp_loop_local(j, f):
        phi = (rs_local[:, 1] + j * 2 * jnp.pi / Kgrid.NFP) % (2 * jnp.pi)
        rs_rpz = jnp.vstack((rs_local[:, 0], phi, rs_local[:, 2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(K_local, phi=phi)  # (N_local, M_local, 3)
        f += biot_savart_general_vec(coords_local, rs_xyz, K_xyz, dV_local)
        return f

    B_local = fori_loop(0, Kgrid.NFP, nfp_loop_local, jnp.zeros((N_local, 3, T_local)))
    B_local_np = np.asarray(B_local, dtype=np.float64)

    # Gather local blocks to root and assemble
    meta = (start_N, end_N, start_T, end_T, world_rank)
    gathered = mpi_comm.gather((meta, B_local_np), root=0)

    if world_rank == 0:
        # Root assembles full array by summing contributions into the right N and T slots.
        B_global = np.zeros((N, 3, T), dtype=np.float64)
        for (meta_k, arr_k) in gathered:
            sN, eN, sT, eT, rnk = meta_k
            if arr_k.size == 0:
                continue
            # arr_k shape should be (eN-sN, 3, eT-sT)
            B_global[sN:eN, :, sT:eT] += arr_k
        return jnp.asarray(B_global)
    else:
        return None


def _compute_magnetic_field_from_Current_Contour_vec_2d(
    Kgrid, K_at_grid, surface, data, coords,
    basis="rpz", mpi_comm=None, split_targets=True
):
    """
    2D-split version for contour currents. Same semantics as _compute_magnetic_field_from_Current_vec_2d.
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    world_rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()

    if hasattr(coords, "nodes"):
        coords_arr = jnp.atleast_2d(coords.nodes)
    else:
        coords_arr = jnp.atleast_2d(coords)
    T = int(coords_arr.shape[0])

    K = jnp.asarray(K_at_grid)
    if K.ndim == 2:
        K = K[None, :, :]
        vectorized = False
    else:
        vectorized = True

    N = int(K.shape[0])
    Mgrid = int(data["x"].shape[0])

    grid_rpz = data["x"]
    grid_xyz = rpz2xyz(grid_rpz)
    coords_xyz = rpz2xyz(coords_arr) if basis == "rpz" else coords_arr

    if basis != "rpz":
        K = xyz2rpz_vec(K, x=grid_xyz[:, 0], y=grid_xyz[:, 1])

    _rs = grid_rpz
    _K = K
    _dV = Kgrid.weights * jnp.sqrt(dot(data["e_theta"], data["e_theta"])) / Kgrid.NFP

    # factorize ranks into P_M x P_N for contour (Pairs behave same)
    P_M, P_N = _factorize_2d(world_size)
    if P_M * P_N != world_size:
        P_M, P_N = world_size, 1

    i_M = world_rank % P_M
    i_N = world_rank // P_M

    start_M, end_M = _split_indices(Mgrid, i_M, P_M)
    start_N, end_N = _split_indices(N, i_N, P_N)
    M_local = end_M - start_M
    N_local = end_N - start_N

    if split_targets:
        start_T, end_T = _split_indices(T, world_rank, world_size)
        coords_local = coords_xyz[start_T:end_T, :]
        T_local = end_T - start_T
    else:
        start_T, end_T = 0, T
        coords_local = coords_xyz
        T_local = T

    if M_local <= 0 or N_local <= 0 or T_local <= 0:
        B_local_np = np.zeros((N_local, 3, T_local), dtype=np.float64)
        meta = (start_N, end_N, start_T, end_T, world_rank)
        gathered = mpi_comm.gather((meta, B_local_np), root=0)
        if world_rank == 0:
            B_global = np.zeros((N, 3, T), dtype=np.float64)
            for meta_k, arr_k in gathered:
                sN, eN, sT, eT, rnk = meta_k
                if arr_k.size == 0:
                    continue
                B_global[sN:eN, :, sT:eT] += arr_k
            return jnp.asarray(B_global)
        else:
            return None

    rs_local = _rs[start_M:end_M, :]
    dV_local = np.asarray(_dV[start_M:end_M], dtype=np.float64)
    K_local = _K[start_N:end_N, start_M:end_M, :]

    def nfp_loop_local(j, f):
        phi = (rs_local[:, 1] + j * 2 * jnp.pi / Kgrid.NFP) % (2 * jnp.pi)
        rs_rpz = jnp.vstack((rs_local[:, 0], phi, rs_local[:, 2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(K_local, phi=phi)
        f += biot_savart_general_vec(coords_local, rs_xyz, K_xyz, dV_local)
        return f

    B_local = fori_loop(0, Kgrid.NFP, nfp_loop_local, jnp.zeros((N_local, 3, T_local)))
    B_local_np = np.asarray(B_local, dtype=np.float64)

    meta = (start_N, end_N, start_T, end_T, world_rank)
    gathered = mpi_comm.gather((meta, B_local_np), root=0)

    if world_rank == 0:
        B_global = np.zeros((N, 3, T), dtype=np.float64)
        for meta_k, arr_k in gathered:
            sN, eN, sT, eT, rnk = meta_k
            if arr_k.size == 0:
                continue
            B_global[sN:eN, :, sT:eT] += arr_k
        return jnp.asarray(B_global)
    else:
        return None

# ---------------- Sticks (unchanged semantics but root-assembly) ----------------
def stick(p2_, p1_, plasma_points, surface_grid, basis="rpz"):
    def nfp_loop(j, f):
        phi2 = (p2_[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)
        p2s = jnp.stack([p2_[:, 0], phi2, p2_[:, 2]], axis=1)
        p2s = rpz2xyz(p2s)
        a_s = p2s[:, None, :] - p1_[:, None, :]
        b_s = p1_[:, None, :] - plasma_points[None, :, :]
        c_s = p2s[:, None, :] - plasma_points[None, :, :]
        c_sxa_s = cross(c_s, a_s)
        term = (
            1e-7
            * (
                (
                    jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=2), a_min=1e-8) *
                    jnp.sum(c_s * c_s, axis=2) ** 0.5
                ) ** (-1)
                * (jnp.sum(a_s * c_s, axis=2) - jnp.sum(a_s * b_s, axis=2))
            )[:, :, None]
            * c_sxa_s
        )
        f += term
        return f

    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop,
                        jnp.zeros((p1_.shape[0], plasma_points.shape[0], plasma_points.shape[1])))
    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:, 0], y=plasma_points[:, 1])
    return b_stick

def B_sticks_vec_2d(sgrid, surface, coords, ss_data, mpi_comm=None, split_targets=True):
    """
    2D-aware sticks. We split wires (M_wires) across P_M parts and expansion N is not relevant here.
    We still gather to root and assemble final (N_wires, T, 3) with root receiving full array.
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    world_rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()

    pls_points = rpz2xyz(coords)
    N_wires = ss_data["x"].shape[0]
    T = pls_points.shape[0]

    start_w, end_w = _split_indices(N_wires, world_rank, world_size)
    M_local = end_w - start_w

    if split_targets:
        start_T, end_T = _split_indices(T, world_rank, world_size)
        T_local = end_T - start_T
        coords_local = pls_points[start_T:end_T, :]
    else:
        start_T, end_T = 0, T
        T_local = T
        coords_local = pls_points

    if M_local <= 0 or T_local <= 0:
        local_arr = np.zeros((N_wires, T_local, 3), dtype=np.float64)
        meta = (start_w, end_w, start_T, end_T, world_rank)
        gathered = mpi_comm.gather((meta, local_arr), root=0)
        if world_rank == 0:
            out = np.zeros((N_wires, T, 3), dtype=np.float64)
            for (meta_k, arr_k) in gathered:
                sW, eW, sT, eT, rnk = meta_k
                if arr_k.size == 0:
                    continue
                out[sW:eW, sT:eT, :] += arr_k[sW - sW:sW - sW + (eW - sW), :, :]
            return jnp.asarray(out)
        else:
            return None

    p1_local = 0 * ss_data["x"][start_w:end_w]
    p2_local = ss_data["x"][start_w:end_w]
    b_local_full = stick(p2_local, p1_local, pls_points, sgrid, basis="rpz")  # (M_local, T, 3)

    # If split_targets, take the correct T slice
    if split_targets:
        local_block = np.asarray(b_local_full)[:, start_T:end_T, :]  # (M_local, T_local, 3)
        # place into full-wire coordinates for root-friendly addition later:
        arr_for_send = np.zeros((N_wires, T_local, 3), dtype=np.float64)
        arr_for_send[start_w:end_w, :, :] = local_block
    else:
        arr_for_send = np.zeros((N_wires, T, 3), dtype=np.float64)
        arr_for_send[start_w:end_w, :, :] = np.asarray(b_local_full)

    meta = (start_w, end_w, start_T, end_T, world_rank)
    gathered = mpi_comm.gather((meta, arr_for_send), root=0)

    if world_rank == 0:
        out = np.zeros((N_wires, T, 3), dtype=np.float64)
        for (meta_k, arr_k) in gathered:
            sW, eW, sT, eT, rnk = meta_k
            if arr_k.size == 0:
                continue
            out[sW:eW, sT:eT, :] += arr_k[sW:eW, :, :]
        return jnp.asarray(out)
    else:
        return None

# ---------------- K_sour_vec (unchanged) ----------------
def K_sour_vec(sdata1, sdata2, sdata3, sgrid, surface, N, d_0, tdata, ss_data):
    omega_sour_fun = (omega_sour(sdata1, ss_data["u_iso"], ss_data["v_iso"], N, d_0)
                      + omega_sour(sdata2, ss_data["u_iso"], ss_data["v_iso"], N, d_0)
                      + omega_sour(sdata3, ss_data["u_iso"], ss_data["v_iso"], N, d_0))
    K_sour_total = ((-jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_v'][:, :, None]
                     + jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_u'][:, :, None])
                    * (sdata1["lambda_iso"] ** -1)[:, None, None])
    return K_sour_total

# ---------------- bn_res_vec_mpi assembling on root (rank 0) ----------------
def bn_res_vec_mpi_2d(
    sdata1, sdata2, sdata3,
    sgrid, surface, N, d_0, coords,
    tdata, contour_data, stick_data,
    contour_grid, ss_data, AAA,
    mpi_comm=None, split_targets=True
):
    """
    MPI-aware Biot-Savart residual vector with full 2D decomposition
    (split across both source points and expansion indices).

    Returns:
        concatenated residual vector (1D array)
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD

    # ---------------- Magnetic field from distributed sources ----------------
    K_sour_total = jnp.transpose(
        K_sour_vec(sdata1, sdata2, sdata3, sgrid, surface, N, d_0, tdata, ss_data),
        (2, 0, 1)
    )

    B_sour0 = _compute_magnetic_field_from_Current_vec_2d(
        Kgrid=sgrid,
        K_at_grid=K_sour_total,
        surface=surface,
        data=sdata1,
        coords=coords,
        basis="rpz",
        mpi_comm=mpi_comm,
        split_targets=split_targets
    )

    # ---------------- Magnetic field from contour currents ----------------
    B_wire_cont = _compute_magnetic_field_from_Current_Contour_vec_2d(
        Kgrid=contour_grid,
        K_at_grid=jnp.transpose(AAA, (2, 0, 1)),
        surface=surface,
        data=contour_data,
        coords=coords,
        basis="rpz",
        mpi_comm=mpi_comm,
        split_targets=split_targets
    )

    # ---------------- Magnetic field from sticks ----------------
    B_sticks0 = B_sticks_vec_2d(
        sgrid=sgrid,
        surface=surface,
        coords=coords,
        ss_data=stick_data,
        mpi_comm=mpi_comm,
        split_targets=split_targets
    )

    # ---------------- Ensure compatible shapes ----------------
    # All arrays should have shape (N_expansion, 3, N_coords)
    def _reshape_B(B):
        if B.shape[1] != 3:
            return jnp.transpose(B, (0, 2, 1))
        return B

    B_sour0    = _reshape_B(B_sour0)
    B_wire_cont = _reshape_B(B_wire_cont)
    B_sticks0  = _reshape_B(B_sticks0)

    # Sum contributions
    B_total = B_sour0 + B_wire_cont + B_sticks0

    # Final transpose for concatenation: (3, N_coords, N_expansion)
    B_total = jnp.transpose(B_total, (1, 2, 0))

    # Concatenate components for residual vector
    return jnp.concatenate((B_total[0, :, :], B_total[1, :, :], B_total[2, :, :]))