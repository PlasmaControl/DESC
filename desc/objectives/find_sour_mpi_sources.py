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

"""
MPI-aware Biot-Savart computation for DESC
Splits source contributions (J * dV) across MPI ranks.
Requires: mpi4py, jax, numpy, desc package
"""


# ---------------- Helper ----------------
def _split_indices(n_items, rank, size):
    """Return start/end indices for a contiguous block for MPI rank."""
    base = n_items // size
    rem = n_items % size
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end

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
                                             basis="rpz", mpi_comm=None):
    """MPI-aware Biot-Savart from volumetric currents."""
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

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

    # Split sources across ranks
    start_idx, end_idx = _split_indices(_rs.shape[0], rank, size)
    local_rs = _rs[start_idx:end_idx, :]
    local_dV = _dV[start_idx:end_idx]
    local_K = _K[:, start_idx:end_idx, :]

    def nfp_loop_local(j, f):
        phi = (local_rs[:,1] + j*2*jnp.pi/Kgrid.NFP) % (2*jnp.pi)
        rs_rpz = jnp.vstack((local_rs[:,0], phi, local_rs[:,2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(local_K, phi=phi)
        f += biot_savart_general_vec(coords_xyz, rs_xyz, K_xyz, local_dV)
        return f

    B_local = fori_loop(0, Kgrid.NFP, nfp_loop_local, jnp.zeros((N,3,coords_xyz.shape[0])))
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
    """MPI-aware Biot-Savart from contour currents."""
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

    start_idx, end_idx = _split_indices(_rs.shape[0], rank, size)
    local_rs = _rs[start_idx:end_idx,:]
    local_dV = _dV[start_idx:end_idx]
    local_K = _K[:, start_idx:end_idx,:]

    def nfp_loop_local(j,f):
        phi = (local_rs[:,1] + j*2*jnp.pi/Kgrid.NFP) % (2*jnp.pi)
        rs_rpz = jnp.vstack((local_rs[:,0], phi, local_rs[:,2])).T
        rs_xyz = rpz2xyz(rs_rpz)
        K_xyz = rpz2xyz_vec(local_K, phi=phi)
        f += biot_savart_general_vec(coords_xyz, rs_xyz, K_xyz, local_dV)
        return f

    B_local = fori_loop(0, Kgrid.NFP, nfp_loop_local, jnp.zeros((N,3,coords_xyz.shape[0])))
    B_local_np = np.asarray(B_local, dtype=np.float64)
    B_global_np = np.empty_like(B_local_np)
    mpi_comm.Allreduce(B_local_np, B_global_np, op=MPI.SUM)
    B = jnp.asarray(B_global_np)

    if basis=="rpz":
        B = xyz2rpz_vec(jnp.transpose(B,(0,2,1)), x=coords_xyz[:,0], y=coords_xyz[:,1])
    if not vectorized:
        B = B[0]
    return B

# ---------------- High-Level MPI Wrappers ----------------
def B_sour_vec(sdata1,sdata2,sdata3,sgrid,surface,N,d_0,coords,tdata,ss_data,mpi_comm=None):
    return _compute_magnetic_field_from_Current_vec(
        sgrid,
        jnp.transpose(K_sour_vec(sdata1,sdata2,sdata3,sgrid,surface,N,d_0,tdata,ss_data),(2,0,1)),
        surface,
        sdata1,
        coords,
        basis="rpz",
        mpi_comm=mpi_comm
    )

def B_theta_contours_vec(surface, coords, ss_data, ss_grid, AAA, mpi_comm=None):
    return _compute_magnetic_field_from_Current_Contour_vec(
        ss_grid,
        jnp.transpose(AAA,(2,0,1)),
        surface,
        ss_data,
        coords,
        basis="rpz",
        mpi_comm=mpi_comm
    )

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

def B_sticks_vec(sgrid, surface, coords, ss_data, mpi_comm=None):
    """MPI-aware magnetic field from sticks."""
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    pls_points = rpz2xyz(coords)
    N_wires = ss_data["x"].shape[0]
    N_coords = pls_points.shape[0]

    start_idx, end_idx = _split_indices(N_wires, rank, size)
    p1_local = 0 * ss_data["x"][start_idx:end_idx]
    p2_local = ss_data["x"][start_idx:end_idx]

    b_local = stick(p2_local, p1_local, pls_points, sgrid, basis="rpz")

    full_shape = (N_wires, N_coords, 3)
    b_local_full = np.zeros(full_shape, dtype=np.float64)
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
                   contour_data,stick_data,contour_grid,ss_data,AAA, mpi_comm=None):
    """MPI-aware Biot-Savart residual vector."""
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD

    B_sour0 = B_sour_vec(sdata1,sdata2,sdata3,sgrid,surface,N,d_0,coords,tdata,ss_data, mpi_comm=mpi_comm)
    B_wire_cont = B_theta_contours_vec(surface, coords, contour_data, contour_grid, AAA, mpi_comm=mpi_comm)
    B_sticks0 = B_sticks_vec(sgrid, surface, coords, stick_data, mpi_comm=mpi_comm)

    B_total = jnp.transpose(B_sour0 + B_wire_cont + B_sticks0, (1,2,0))
    return jnp.concatenate((B_total[:,0,:], B_total[:,1,:], B_total[:,2,:]))

# ---------------- Example MPI usage ----------------
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f"Running MPI with {size} ranks")