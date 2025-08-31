from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from .sources_dipoles_utils import (_compute_magnetic_field_from_Current, 
                                    _compute_magnetic_field_from_Current_Contour,
                                    omega_sour,
                                    compute_mask,
                                    )

# add these imports at top of your file
from mpi4py import MPI
import numpy as np

from mpi4py import MPI
import numpy as np
from desc.backend import fori_loop, jax, jnp
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz_vec
from desc.utils import cross, dot

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

# ---------------- Compute Magnetic Field (MPI-Aware) ----------------
def _compute_magnetic_field_from_Current_vec(Kgrid, K_at_grid, surface, data, coords,
                                             basis="rpz", mpi_comm=None):
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)

    if K_at_grid.ndim == 2:
        K_at_grid = K_at_grid[None, :, :]
        vectorized = False
    else:
        vectorized = True
    N = K_at_grid.shape[0]

    grid_rpz = data["x"]
    grid_xyz = rpz2xyz(grid_rpz)
    coords_xyz = rpz2xyz(coords) if basis=="rpz" else coords
    if basis != "rpz":
        K_at_grid = xyz2rpz_vec(K_at_grid, x=grid_xyz[:,0], y=grid_xyz[:,1])

    surface_grid = Kgrid
    _rs = grid_rpz
    _K = K_at_grid
    _dV = surface_grid.weights * data["|e_theta x e_zeta|"] / surface_grid.NFP

    # Split source nodes
    start_idx, end_idx = _split_indices(_rs.shape[0], rank, size)
    local_rs = _rs[start_idx:end_idx, :]
    local_dV = _dV[start_idx:end_idx]
    local_K = _K[:, start_idx:end_idx, :]

    def nfp_loop_local(j, f):
        phi = (local_rs[:,1] + j*2*jnp.pi/surface_grid.NFP) % (2*jnp.pi)
        rs_rpz = jnp.vstack((local_rs[:,0], phi, local_rs[:,2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(local_K, phi=phi)
        f += biot_savart_general_vec(coords_xyz, rs_xyz, K_xyz, local_dV)
        return f

    B_local = fori_loop(0, surface_grid.NFP, nfp_loop_local, jnp.zeros((N,3,coords_xyz.shape[0])))
    B_local_np = np.asarray(B_local, dtype=np.float64)
    B_global_np = np.empty_like(B_local_np)
    mpi_comm.Allreduce(B_local_np, B_global_np, op=MPI.SUM)
    B = jnp.asarray(B_global_np)

    if basis=="rpz":
        B = xyz2rpz_vec(jnp.transpose(B,(0,2,1)), x=coords_xyz[:,0], y=coords_xyz[:,1])
    if not vectorized:
        B = B[0]
    return B

def _compute_magnetic_field_from_Current_Contour_vec(Kgrid, K_at_grid, surface, data, coords,
                                                     basis="rpz", mpi_comm=None):
    """MPI-Aware version for contour currents."""
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)

    if K_at_grid.ndim==2:
        K_at_grid = K_at_grid[None,:,:]
        vectorized=False
    else:
        vectorized=True
    N = K_at_grid.shape[0]

    grid_rpz = data["x"]
    grid_xyz = rpz2xyz(grid_rpz)
    coords_xyz = rpz2xyz(coords) if basis=="rpz" else coords
    if basis != "rpz":
        K_at_grid = xyz2rpz_vec(K_at_grid, x=grid_xyz[:,0], y=grid_xyz[:,1])

    surface_grid = Kgrid
    _rs = grid_rpz
    _K = K_at_grid
    _dV = surface_grid.weights * jnp.sqrt(dot(data["e_theta"], data["e_theta"])) / surface_grid.NFP

    start_idx, end_idx = _split_indices(_rs.shape[0], rank, size)
    local_rs = _rs[start_idx:end_idx,:]
    local_dV = _dV[start_idx:end_idx]
    local_K = _K[:, start_idx:end_idx,:]

    def nfp_loop_local(j,f):
        phi = (local_rs[:,1] + j*2*jnp.pi/surface_grid.NFP) % (2*jnp.pi)
        rs_rpz = jnp.vstack((local_rs[:,0], phi, local_rs[:,2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(local_K, phi=phi)
        f += biot_savart_general_vec(coords_xyz, rs_xyz, K_xyz, local_dV)
        return f

    B_local = fori_loop(0, surface_grid.NFP, nfp_loop_local, jnp.zeros((N,3,coords_xyz.shape[0])))
    B_local_np = np.asarray(B_local, dtype=np.float64)
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
    
#######################
# --- Utility: balanced splitting of indices across ranks ---
def _split_indices(N, rank, size):
    """Return start, end indices for this rank when splitting N items across size ranks."""
    n_per_rank = N // size
    remainder = N % size
    if rank < remainder:
        start = rank * (n_per_rank + 1)
        end = start + n_per_rank + 1
    else:
        start = rank * n_per_rank + remainder
        end = start + n_per_rank
    return start, end


# --- Biot–Savart contribution from volume sources ---
def B_sour_vec(sgrid, surface, coords, J, dV, rs, *, mpi_comm=None):
    """
    MPI-parallelized computation of B field from volume current sources.
    Distributes *sources* (J, dV, rs) across ranks.
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    N_sources = J.shape[0]
    start, end = _split_indices(N_sources, rank, size)

    J_chunk = J[start:end]
    dV_chunk = dV[start:end]
    rs_chunk = rs[start:end]

    # Each rank computes its partial contribution
    b_local = biot_savart_general_vec(coords, rs_chunk, J_chunk, dV_chunk)

    # Convert to numpy, then reduce across ranks
    b_local_np = np.asarray(b_local, dtype=np.float64)
    b_global_np = np.empty_like(b_local_np)
    mpi_comm.Allreduce(b_local_np, b_global_np, op=MPI.SUM)

    return jnp.asarray(b_global_np)


# --- Biot–Savart contribution from theta-contours (same pattern as above) ---
def B_theta_contours_vec(sgrid, surface, coords, J, dV, rs, *, mpi_comm=None):
    """
    MPI-parallelized computation of B field from theta-contour sources.
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    N_sources = J.shape[0]
    start, end = _split_indices(N_sources, rank, size)

    J_chunk = J[start:end]
    dV_chunk = dV[start:end]
    rs_chunk = rs[start:end]

    b_local = biot_savart_general_vec(coords, rs_chunk, J_chunk, dV_chunk)

    b_local_np = np.asarray(b_local, dtype=np.float64)
    b_global_np = np.empty_like(b_local_np)
    mpi_comm.Allreduce(b_local_np, b_global_np, op=MPI.SUM)

    return jnp.asarray(b_global_np)


# --- Sticks contribution (already working, just cleaned) ---
def B_sticks_vec(sgrid, surface, coords, ss_data, *, mpi_comm=None):
    """
    MPI-parallelized computation of B field from discrete sticks (wires).
    Distributes sticks across ranks.
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    N_sticks = ss_data.shape[0]
    start, end = _split_indices(N_sticks, rank, size)

    sticks_chunk = ss_data[start:end]

    # Local computation
    b_local = biot_savart_general_vec_sticks(coords, sticks_chunk)

    # Convert to numpy, pad if needed (because different ranks may compute different #sticks)
    b_local_np = np.asarray(b_local, dtype=np.float64)
    b_global_np = np.zeros_like(b_local_np)
    mpi_comm.Allreduce(b_local_np, b_global_np, op=MPI.SUM)

    return jnp.asarray(b_global_np)


# --- Top-level residual calculation ---
def bn_res_vec_mpi(sgrid, surface, coords, J, dV, rs, ss_data, *, mpi_comm=None):
    """
    Full MPI-parallelized residual calculation.
    Distributes *sources* across ranks for B_sour and B_theta_contours,
    and distributes sticks for B_sticks.
    """
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD

    # Contributions
    B1 = B_sour_vec(sgrid, surface, coords, J, dV, rs, mpi_comm=mpi_comm)
    B2 = B_theta_contours_vec(sgrid, surface, coords, J, dV, rs, mpi_comm=mpi_comm)
    B3 = B_sticks_vec(sgrid, surface, coords, ss_data, mpi_comm=mpi_comm)

    # Combine
    B_total = B1 + B2 + B3
    return B_total