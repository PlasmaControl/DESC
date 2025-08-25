from interpax import interp2d

from desc.backend import fori_loop, jax, jit, jnp
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

#from desc.find_dips import biot_savart_general

#@jax.jit
def f_sour(data_or, u1_, v1_, N, d_0): 

    w1 = comp_loc(
        u1_,
        v1_,
    )

    v_1_num = v1_eval(w1, N, d_0, data_or)

    f1_reg = f_reg(w1, d_0, data_or)

    return (
        jnp.log(v_1_num)
        - 2
        * jnp.pi
        * jnp.real(w1)
        / (data_or["omega_1"] ** 2 * data_or["tau_2"])
        * data_or["w"]
        - 1 / 2 * f1_reg
    )

#@jax.jit
def omega_sour(
    data_or,
    u1_,
    v1_,
    N,
    d_0,
):

    w1 = comp_loc( u1_, v1_ )

    v_1_num = v1_eval(w1, N, d_0, data_or)
    v_1_num_prime = v1_prime_eval(w1, N, d_0, data_or)

    chi_reg_1 = chi_reg(w1, d_0, data_or)

    omega = (
        v_1_num_prime / v_1_num  # Regularized near the vortex cores
        - 2 * jnp.pi * jnp.real(w1) / (data_or["omega_1"] ** 2 * data_or["tau_2"])
        + 1/2 * (chi_reg_1)  # Additional terms with regularization close to the vortex core
    )

    return omega


#@jax.jit
def v1_prime_eval(w0, N, d_0, data_or):

    gamma = data_or["omega_1"] / jnp.pi
    p = data_or["tau"]
    
    def body_fun(n, carry):
        _product = carry
        term = ( ((-1) ** n) * (p ** (n**2 + n)) ) * ( ( (2 * n + 1) / gamma ) * jnp.cos((2 * n + 1) * ( data_or["w"][:,None] - w0[None,:]) / gamma ) )
        return _product + term
    
    test = fori_loop( 0, N, body_fun, jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) + jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) )*1j)
    
    return test


#@jax.jit
def v1_eval(w0, N, d_0, data_or):
    
    gamma = data_or["omega_1"] / jnp.pi
    p = data_or["tau"]

    product_ = 0

    #for n in range(0, N):

    #    product_ = product_ + (
    #        (((-1) ** n) * (p ** (n**2 + n)))
    #        * jnp.sin((2 * n + 1) * (data_or["w"] - w0) / gamma)
    #    )

    def body_fun(n,carry):
        product_ = carry
        term = product_ + (  ( ( (-1) ** n) * (p ** (n**2 + n) ) ) * jnp.sin( (2 * n + 1) * (data_or["w"][:,None] - w0[None,:]) / gamma) )
        
        return product_ + term
        
    return jnp.where(
        jnp.abs(data_or["w"][:,None] - w0[None,:]) > d_0,
        2 * p ** (1 / 4) * fori_loop(0, N, body_fun, 
                                     jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) + jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) )*1j ),
        1,  # Arbitraty value of 1 inside the circle around the vortex core
    )

    
#@jax.jit
def chi_reg(w0, d_0, data_or):  # location of the vortex

    return jnp.where(
        jnp.abs(data_or["w"][:,None] - w0[None,:]) < d_0,
        (- ( data_or["lambda_u"] / data_or["lambda_iso"])
        + (data_or["lambda_v"] / data_or["lambda_iso"]) * 1j)[:,None],
        0,
    )


#@jax.jit
def f_reg(w0, d_0, data_or):  # location of the vortex

    return jnp.where(
        jnp.abs(data_or["w"] - w0) < d_0, jnp.log(data_or["lambda_iso"]), 0
    )


#@jax.jit
def comp_loc(
    theta_0_,
    phi_0_,
):
    return theta_0_ + phi_0_ * 1j


# Interpolate isothermal coordinates and interpolate on a different grid
#@jax.jit
def iso_coords_interp(tdata, _data, sgrid):

    # Temporary grid
    tgrid = LinearGrid(M=60, N=60)
    Phi = tdata["Phi_iso"]
    Psi = tdata["Psi_iso"]
    b0 = tdata["b_iso"]
    lamb_ratio = tdata["lambda_ratio"]

    _data["omega_1"] = tdata["omega_1"]
    _data["omega_2"] = tdata["omega_1"]

    _data["tau"] = tdata["tau"]
    _data["tau_1"] = tdata["tau_1"]
    _data["tau_2"] = tdata["tau_2"]

    lamb_u = tdata["lambda_iso_u"]
    lamb_v = tdata["lambda_iso_v"]

    # Data on plasma surface
    u_t = tdata["u_t"]
    u_z = tdata["u_z"]
    v_t = tdata["v_t"]
    v_z = tdata["v_z"]

    # Build new grids to allow interpolation between last grid points and theta = 2*pi or zeta = 2*pi
    m_size = tgrid.M * 2 + 1
    n_size = tgrid.N * 2 + 1

    # Rearrange variables
    # Add extra rows and columns to represent theta = 2pi or zeta = 2pi
    theta_mod = add_extra_coords(tdata["theta"], n_size, m_size, 0)
    zeta_mod = add_extra_coords(tdata["zeta"], n_size, m_size, 1)
    u_mod = zeta_mod - add_extra_periodic(Phi, n_size, m_size)
    v_mod = lamb_ratio * (
        theta_mod - add_extra_periodic(Psi, n_size, m_size) + b0 * u_mod
    )
    u_t_mod = add_extra_periodic(u_t, n_size, m_size)
    u_z_mod = add_extra_periodic(u_z, n_size, m_size)
    v_t_mod = add_extra_periodic(v_t, n_size, m_size)
    v_z_mod = add_extra_periodic(v_z, n_size, m_size)
    lamb_u_mod = add_extra_periodic(lamb_u, n_size, m_size)
    lamb_v_mod = add_extra_periodic(lamb_v, n_size, m_size)

    # Interpolate on theta_mod, zeta_mod
    points = jnp.array((zeta_mod.flatten(), theta_mod.flatten())).T

    # Interpolate isothermal coordinates
    _data["u_iso"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        u_mod,
        method="cubic",
    )
    _data["v_iso"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        v_mod,
        method="cubic",
    )

    _data["lambda_u"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        lamb_u_mod,
        method="cubic",
    )
    _data["lambda_v"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        lamb_v_mod,
        method="cubic",
    )

    # Interpolate derivatives of isothermal coordinates
    u0_t = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        u_t_mod,
        method="cubic",
    )
    u0_z = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        u_z_mod,
        method="cubic",
    )
    v0_t = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        v_t_mod,
        method="cubic",
    )
    v0_z = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        v_z_mod,
        method="cubic",
    )

    # Build harmonic vectors with interpolated data
    grad1 = (u0_t * _data["e^theta_s"].T + u0_z * _data["e^zeta_s"].T).T
    grad2 = (v0_t * _data["e^theta_s"].T + v0_z * _data["e^zeta_s"].T).T

    _data["e^u_s"] = grad1
    _data["e^v_s"] = grad2

    _data["e_u"] = (dot(grad1, grad1) ** (-1) * grad1.T).T
    _data["e_v"] = (dot(grad2, grad2) ** (-1) * grad2.T).T

    # Define the parameter "lambda" according to the paper
    _data["lambda_iso"] = dot(_data["e_u"], _data["e_u"]) ** (1 / 2)

    _data["w"] = comp_loc(_data["u_iso"], _data["v_iso"])

    return _data


# Load isothermal coordinates on the construction grid
def iso_coords_load(name, eq):

    # Temporary grid
    tgrid = LinearGrid(M=60, N=60)

    # Data on plasma surface
    tdata = eq.compute(["theta", "zeta"], grid=tgrid)

    tdata["u_iso"] = jnp.load(name + "u.npy")
    tdata["v_iso"] = jnp.load(name + "v.npy")
    tdata["Phi_iso"] = jnp.load(name + "Phi.npy")
    tdata["Psi_iso"] = jnp.load(name + "Psi.npy")
    tdata["b_iso"] = jnp.load(name + "b.npy")
    tdata["lambda_ratio"] = jnp.load(name + "ratio.npy")

    tdata["omega_1"] = jnp.load(name + "omega_1.npy")
    tdata["omega_2"] = jnp.load(name + "omega_2.npy")

    tdata["tau"] = jnp.load(name + "tau.npy")
    tdata["tau_1"] = jnp.load(name + "tau_1.npy")
    tdata["tau_2"] = jnp.load(name + "tau_2.npy")

    tdata["lambda_iso_u"] = jnp.load(name + "lambda_u.npy")
    tdata["lambda_iso_v"] = jnp.load(name + "lambda_v.npy")

    tdata["u_t"] = jnp.load(name + "u_t.npy")
    tdata["u_z"] = jnp.load(name + "u_z.npy")
    tdata["v_t"] = jnp.load(name + "v_t.npy")
    tdata["v_z"] = jnp.load(name + "v_z.npy")

    return tdata


def interp_grid(theta, zeta, w_surface, tdata):

    # Find grids for dipoles
    s_grid = alt_grid(theta, zeta)

    # Evaluate data on grids of dipoles
    s_data = w_surface.compute(
        [
            "theta",
            "zeta",
            "e^theta_s",
            "e^zeta_s",
            "x",
            "e_theta",  # extra vector needed for the poloidal wire contours
            '|e_theta x e_zeta|',
        ],
        grid=s_grid,
    )

    return iso_coords_interp(tdata, s_data, w_surface)


#@jax.jit
def add_extra(data_, n_size, m_size):

    _mod = data_.reshape((n_size, m_size)).T
    _mod = jnp.column_stack([_mod, _mod[0:m_size, 0]])
    _mod = jnp.vstack([_mod, 2 * jnp.pi * jnp.ones(_mod.shape[1])])

    return _mod


#@jax.jit
def add_extra_periodic(data_, n_size, m_size):

    _mod = data_.reshape((n_size, m_size)).T
    _mod = jnp.column_stack([_mod, _mod[:, 0]])
    _mod = jnp.vstack([_mod, _mod[0, :]])

    return _mod


#@jax.jit
def add_extra_coords(data_, n_size, m_size, ind):

    _mod = data_.reshape((n_size, m_size)).T
    # _mod = jnp.vstack( [ _mod, _mod[0:m_size,0] ] )

    if ind == 0:
        _mod = jnp.column_stack([_mod, _mod[:, 0]])
        _mod = jnp.vstack([_mod, 2 * jnp.pi * jnp.ones(_mod.shape[1])])

    if ind == 1:
        _mod = jnp.column_stack([_mod, 2 * jnp.pi * jnp.ones_like(_mod[:, 0])])
        _mod = jnp.vstack([_mod, _mod[0, :]])

    return _mod

#@jax.jit
def alt_grid(theta, zeta):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten()

    return Grid(
        jnp.stack((jnp.ones_like(theta_flat), theta_flat, zeta_flat)).T, jitable=True
    )

#@jax.jit
def alt_grid_sticks(theta, zeta, sgrid):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten()

    return Grid(
        jnp.stack((jnp.ones_like(theta_flat), theta_flat, zeta_flat)).T,
        weights=jnp.ones_like(theta_flat),
        NFP=sgrid.NFP,
        jitable=True,
    )

#@jax.jit
def densify_linspace(arr, points_per_interval=1):
    """
    Given a jnp.linspace array, return a new array with additional points
    between each pair of original points while keeping all original points.

    Args:
        arr (jnp.ndarray): Original 1D array (typically from jnp.linspace)
        points_per_interval (int): Number of points to insert between each pair

    Returns:
        jnp.ndarray: New array with original + additional interpolated points
    """
    if arr.ndim != 1:
        raise ValueError("Only 1D arrays supported")

    new_points = []
    for i in range(len(arr) - 1):
        start = arr[i]
        end = arr[i + 1]

        # Include original point
        new_points.append(start)

        # Generate internal points (excluding end to avoid duplication)
        if points_per_interval > 0:
            inter_points = jnp.linspace(start, end, points_per_interval + 2)[1:-1]
            new_points.append(inter_points)

    new_points.append(arr[-1])  # Don't forget the last point!

    return jnp.concatenate([jnp.atleast_1d(p) for p in new_points])


#@jax.jit
def _compute_magnetic_field_from_Current(
    Kgrid, K_at_grid, surface, data,
    coords,
    #basis="rpz"
):
    """Compute magnetic field at a set of points.

    Parameters
    ----------
    K_at_grid : ndarray, shape (num_nodes,3)
        Surface current evaluated at points on a grid, which you want to calculate
        B from, should be in cartesian ("xyz") or cylindrical ("rpz") specifiec
        by "basis" argument
    surface : FourierRZToroidalSurface
        surface object upon which the surface current K_at_grid lies
    coords : array-like shape(N,3) or Grid
        cylindrical or cartesian coordinates to evlauate B at
    grid : Grid,
        source grid upon which to evaluate the surface current density K
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field

    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field at specified points

    """

    #Bdata = eq.compute(["R", "phi", "Z", "n_rho"], grid=Bgrid)
    #coords = jnp.vstack([Bdata["R"], Bdata["phi"], Bdata["Z"]]).T

    basis="rpz"
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])

    surface_grid = Kgrid

    # compute and store grid quantities
    # needed for integration
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    #data = surface.compute(["x", "|e_theta x e_zeta|"], grid=surface_grid, basis="xyz")

    _rs = data["x"] # Coordinates passed to this function are already in rpz format
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    _dV = surface_grid.weights * data["|e_theta x e_zeta|"] / surface_grid.NFP

    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (
            2 * jnp.pi
        )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general(
            coords,
            rs,
            K,
            _dV,
        )
        f += fj
        return f

    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))

    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])

    return B

#@jax.jit
def _compute_magnetic_field_from_Current_Contour(
    Kgrid, K_at_grid, surface, data, coords,
    #basis="rpz"
):
    """Compute magnetic field at a set of points.

    Parameters
    ----------
    K_at_grid : ndarray, shape (num_nodes,3)
        Surface current evaluated at points on a grid, which you want to calculate
        B from, should be in cartesian ("xyz") or cylindrical ("rpz") specifiec
        by "basis" argument
    surface : FourierRZToroidalSurface
        surface object upon which the surface current K_at_grid lies
    coords : array-like shape(N,3) or Grid
        cylindrical or cartesian coordinates to evlauate B at
    grid : Grid,
        source grid upon which to evaluate the surface current density K
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field

    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field at specified points

    """

    basis="rpz"
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])

    surface_grid = Kgrid

    _rs = data["x"] # Coordinates passed to this function must be in rpz format
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    # _dV = surface_grid.weights * data["|e_theta x e_zeta|"] / surface_grid.NFP
    # _dV = surface_grid.weights * 1 / surface_grid.NFP
    _dV = (
        surface_grid.weights
        * dot(data["e_theta"], data["e_theta"]) ** (1 / 2)
        / surface_grid.NFP
    )

    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (
            2 * jnp.pi
        )

        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general(
            coords,
            rs,
            K,
            _dV,
        )
        f += fj
        return f

    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))

    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])

    return B

#@jax.jit
def biot_savart_general(re, rs, J, dV):

    #print("rs.shape =", rs.shape)
    #print("J.shape =", J.shape)
    
    #re, rs, J, dV = map(jnp.asarray, (re, rs, J, dV))
    #assert J.shape == rs.shape
    #JdV = J * dV[:, None]
    #B = jnp.zeros_like(re)

    re, rs, J, dV = map(lambda x: jnp.asarray(x, dtype=jnp.float64), (re, rs, J, dV))
    JdV = J * dV[:, None]
    B = jnp.zeros_like(re, dtype=jnp.float64)
    
    def body(i, B):
        r = re - rs[i, :]
        num = jnp.cross(JdV[i, :], r, axis=-1)
        den = jnp.linalg.norm(r, axis=-1) ** 3
        B = B + jnp.where(den[:, None] == 0, 0, num / den[:, None])
        return B

    #return 1e-7 * fori_loop(0, J.shape[0], body, B)
    return 1e-7 * fori_loop(0, rs.shape[0], body, B)


##### Functions for dipoles
#@jax.jit
def shift_grid(theta, zeta, dt, dz, w_surface, name):
    
    l_zeta = zeta - dz/2
    r_zeta = zeta + dz/2
    d_theta = theta - dt/2
    u_theta = theta + dt/2
        
    # Find grids for dipoles
    l_grid = alt_grid(theta,l_zeta)
    r_grid = alt_grid(theta,r_zeta)

    d_grid = alt_grid(d_theta,zeta)
    u_grid = alt_grid(u_theta,zeta)
    
    # Evaluate data on grids of dipoles
    l_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = l_grid)
    r_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = r_grid)
    d_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = d_grid)
    u_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = u_grid)
    
    return (iso_coords_interp(name, l_data, l_grid, w_surface),
            iso_coords_interp(name, r_data, r_grid, w_surface),
            iso_coords_interp(name, d_data, d_grid, w_surface),
            iso_coords_interp(name, u_data, u_grid, w_surface)
           )

#@jax.jit
def f_pair(data_or,
           u1_, v1_, # first dipole
           u2_, v2_, # second dipole
           N, d_0):
    
    # Evaluate the dipoles on the grid of vortices
    w1 = comp_loc(u1_, v1_,)
    w2 = comp_loc(u2_, v2_,)
    
    v_1_num = v1_eval(w1, N, d_0, data_or)
    v_1_den = v1_eval(w2, N, d_0, data_or)
    
    f1_reg = f_reg(w1, d_0, data_or)
    f2_reg = f_reg(w2, d_0, data_or)

    return ( jnp.log( v_1_num ) - jnp.log( v_1_den )
            - 2 * jnp.pi * jnp.real( w1 - w2 ) / ( data_or["omega_1"] ** 2 * data_or["tau_2"] ) * data_or["w"] 
            - 1/2 * ( f1_reg - f2_reg )
           )

#@jax.jit
def omega_pair(data_or,
               u1_, v1_, # first dipole
               u2_, v2_, # second dipole
               N,
               d_0,
              ):
    
    w1 = comp_loc(u1_, v1_,)
    w2 = comp_loc(u2_, v2_,)
    
    v_1_num = v1_eval(w1, N,  d_0, data_or)
    v_1_den = v1_eval(w2, N, d_0, data_or) 
    v_1_num_prime = v1_prime_eval(w1, N, d_0, data_or)
    v_1_den_prime = v1_prime_eval(w2, N, d_0, data_or)
    
    chi_reg_1 = chi_reg(w1, d_0, data_or)
    chi_reg_2 = chi_reg(w2, d_0, data_or)
    
    omega = ( ( v_1_num_prime / v_1_num - v_1_den_prime / v_1_den ) # Regularized near the vortex cores
             - 2 * jnp.pi * jnp.real( w1 - w2 ) / ( data_or["omega_1"] ** 2 * data_or["tau_2"]) 
             + 1 / 2 * ( chi_reg_1 - chi_reg_2 ) # Additional terms with regularization close to the vortex core
            )
    
    return omega

#@jax.jit
def compute_mask(contour_data, theta_coarse, zeta_coarse):
    """
    Compute a binary mask matrix matching the column order of the original loop.
    Each block of m columns corresponds to one zeta_coarse value.
    
    Parameters:
        contour_data (dict): Contains 1D arrays "theta" (shape (n,)) and "zeta" (shape (n,)).
        theta_coarse (ndarray): 1D array of shape (m,).
        zeta_coarse (ndarray): 1D array of shape (k,).
    
    Returns:
        ndarray: Binary mask of shape (n, m*k), with columns ordered by zeta_coarse.
    """
    theta_cond = contour_data["theta"][:, None] >= theta_coarse[None, :]  # Shape (n, m)
    zeta_cond = contour_data["zeta"][:, None] == zeta_coarse[None, :]    # Shape (n, k)
    mask = jnp.where(theta_cond[:, :, None] & zeta_cond[:, None, :], 1, 0)  # Shape (n, m, k)
    # Transpose to (n, k, m) so zeta_coarse varies fastest
    mask = jnp.transpose(mask, (0, 2, 1))
    # Reshape to (n, m*k)
    return mask.reshape(mask.shape[0], -1)