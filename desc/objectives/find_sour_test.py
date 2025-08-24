from interpax import interp2d

from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.find_dips import biot_savart_general

# from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot


def bn_res(
    p_M,
    p_N,
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    y,
    N,
    d_0,
    coords,
    tdata,
    contour_data,
    stick_data,
    contour_grid,
):

    B_sour0 = B_sour(
        p_M, p_N, sdata1, sdata2, sdata3, sgrid, surface, y, N, d_0, coords, tdata
    )

    B_wire_cont = B_theta_contours(
        p_M, p_N, sdata1, sgrid, surface, y, coords, contour_data, contour_grid
    )

    B_sticks0 = B_sticks(
        p_M,
        p_N,
        sgrid,
        surface,
        y,
        coords,
        stick_data,
    )

    B_total = B_sour0 + B_wire_cont + B_sticks0

    return B_total


def B_sour(
    p_M,
    p_N,
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    y,
    N,
    d_0,
    coords,  # eq,Bgrid,
    tdata,
):

    return _compute_magnetic_field_from_Current(
        sgrid,
        K_sour(
            p_M,
            p_N,
            sdata1,
            sdata2,
            sdata3,
            sgrid,
            surface,
            y,
            # dt, dz,
            N,
            d_0,
            tdata,
        ),
        surface,
        sdata1,
        coords,
        basis="rpz",
    )


# @jax.jit
def B_theta_contours(
    p_M,
    p_N,
    sdata,
    sgrid,
    surface,
    y,
    coords,
    ss_data,
    ss_grid,
):

    r_t = p_M * 2 + 1  # theta_coarse.shape[0]
    r_z = p_N * 2 + 1  # zeta_coarse.shape[0]

    theta_coarse = jnp.linspace(
        2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1 / 2,
        2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1 / 2),
        p_M * 2 + 1,
    )

    zeta_coarse = jnp.linspace(
        2 * jnp.pi / surface.NFP * (1 / (p_N * 2 + 1)) * 1 / 2,
        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2 + 1) * 1 / 2),
        p_N * 2 + 1,
    )

    sign_vals = jnp.where(ss_data["theta"] < jnp.pi, -1, 0) + jnp.where(
        ss_data["theta"] > jnp.pi, 1, 0
    )

    def outer_body(i, K_cont):
        def inner_body(j, K_cont_inner):
            k_fix = jnp.where(
                (ss_data["zeta"] == zeta_coarse[i])
                & (ss_data["theta"] > theta_coarse[j]),
                1,
                0,
            )
            return (
                K_cont_inner
                + (
                    y[i * r_t + j]
                    * sign_vals
                    * k_fix
                    * dot(ss_data["e_theta"], ss_data["e_theta"]) ** (-1 / 2)
                    * ss_data["e_theta"].T
                ).T
            )

        return fori_loop(0, r_t, inner_body, K_cont)

    K_cont = fori_loop(0, r_z, outer_body, jnp.zeros_like(ss_data["e_theta"]))

    return _compute_magnetic_field_from_Current_Contour(
        ss_grid, K_cont, surface, ss_data, coords, basis="rpz"
    )


def B_sticks(
    p_M,
    p_N,
    sgrid,
    surface,
    y,
    coords,
    ss_data,
):

    pls_points = rpz2xyz(coords)  # eq_surf.compute(["x"], grid=Bgrid, basis="xyz")["x"]

    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    def sticks_fun(carry, x):
        i = x
        b_stick_fun = carry + y[i] * stick(
            ss_data["x"][
                i
            ],  # Location of the wire at the theta = pi cut, variable zeta position
            0 * ss_data["x"][i],  # All wires at the center go to the origin
            pls_points,
            sgrid,
            basis="rpz",
        )
        return b_stick_fun

    def sticks_map(i):
        return scan(sticks_fun, init=jnp.array([0]), xs=i)

    len0 = len(y)
    i = jnp.arange(0, len0)
    sticks_total = vmap(lambda j: sticks_map(j))(i)

    return sticks_total


def stick(
    p2_,  # second point of the stick
    p1_,  # first point of the stick
    plasma_points,  # points on the plasma surface
    surface_grid,  # Kgrid,
    basis="rpz",
):

    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi2 = (p2_[2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)
        # phi1 = ( p1_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        # (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )

        p2s = jnp.vstack((p2_[0], phi2, p2_[2])).T
        p2s = rpz2xyz(p2s)

        a_s = p2s - p1_
        b_s = p1_ - plasma_points
        c_s = p2s - plasma_points

        c_sxa_s = cross(c_s, a_s)

        f += (
            1e-7
            * (
                (
                    jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=1), a_min=1e-8, a_max=None)
                    * jnp.sum(c_s * c_s, axis=1) ** (1 / 2)
                )
                ** (-1)
                * (jnp.sum(a_s * c_s) - jnp.sum(a_s * b_s))
                * c_sxa_s.T
            ).T
        )

        return f

    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(plasma_points))

    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:, 0], y=plasma_points[:, 1])

    return b_stick


# @jax.jit
# @jax.jit(static_argnums=(0,1))
def K_sour(
    p_M,
    p_N,
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    y,
    N,
    d_0,
    tdata,
):

    theta = jnp.linspace(
        2 * jnp.pi * (1 / (p_M * 2)) * 1 / 2,
        2 * jnp.pi * (1 - 1 / (p_M * 2) * 1 / 2),
        p_M * 2,
    )

    zeta = jnp.linspace(
        2 * jnp.pi / surface.NFP * (1 / (p_N * 2)) * 1 / 2,
        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2) * 1 / 2),
        p_N * 2,
    )

    ss_data = interp_grid(theta, zeta, surface, tdata)

    assert (p_M * 2) * (p_N * 2) == ss_data["theta"].shape[
        0
    ], "Check that the sources coincide with the number of sources/sinks"

    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    ss_data["u_iso"] = jnp.asarray(ss_data["u_iso"])
    ss_data["v_iso"] = jnp.asarray(ss_data["v_iso"])

    # @jax.jit
    def body_fun1(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = jax.lax.dynamic_index_in_dim(y, i, axis=0)

        omega_s1 = omega_sour(sdata1, ss_data["u_iso"][i], ss_data["v_iso"][i], N, d_0)

        omega_total_real += y_ * jnp.real(omega_s1)
        omega_total_imag += y_ * jnp.imag(omega_s1)
        return omega_total_real, omega_total_imag

    # @jax.jit
    def body_fun2(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = jax.lax.dynamic_index_in_dim(y, i, axis=0)

        omega_s2 = omega_sour(sdata2, ss_data["u_iso"][i], ss_data["v_iso"][i], N, d_0)

        omega_total_real += y_ * jnp.real(omega_s2)
        omega_total_imag += y_ * jnp.imag(omega_s2)
        return omega_total_real, omega_total_imag

    # @jax.jit
    def body_fun3(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = jax.lax.dynamic_index_in_dim(y, i, axis=0)

        # Need to evlauate three omegas
        omega_s3 = omega_sour(sdata3, ss_data["u_iso"][i], ss_data["v_iso"][i], N, d_0)

        omega_total_real += y_ * jnp.real(omega_s3)
        omega_total_imag += y_ * jnp.imag(omega_s3)
        return omega_total_real, omega_total_imag

    omega_total_real1, omega_total_imag1 = fori_loop(
        0,
        r,
        body_fun1,
        (jnp.zeros_like(sdata1["theta"]), jnp.zeros_like(sdata1["theta"])),
    )

    omega_total_real2, omega_total_imag2 = fori_loop(
        0,
        r,
        body_fun2,
        (jnp.zeros_like(sdata2["theta"]), jnp.zeros_like(sdata2["theta"])),
    )

    omega_total_real3, omega_total_imag3 = fori_loop(
        0,
        r,
        body_fun3,
        (jnp.zeros_like(sdata3["theta"]), jnp.zeros_like(sdata3["theta"])),
    )

    # Assume periodicity on the values of the sources so
    return (
        (sdata1["lambda_iso"] ** (-1))
        * (
            -omega_total_imag1
            * sdata1["e_v"].T  # - cross(sdata["n_rho"],sdata["e_u"]).T
            - omega_total_imag2 * sdata2["e_v"].T
            - omega_total_imag2 * sdata3["e_v"].T
            + omega_total_real1
            * sdata1["e_u"].T  # - cross(sdata["n_rho"],sdata["e_v"]).T
            + omega_total_real2 * sdata2["e_u"].T
            + omega_total_real2 * sdata3["e_u"].T
        )
    ).T


# @jax.jit
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


def omega_sour(
    data_or,
    u1_,
    v1_,
    N,
    d_0,
):

    w1 = comp_loc(
        u1_,
        v1_,
    )

    v_1_num = v1_eval(w1, N, d_0, data_or)
    v_1_num_prime = v1_prime_eval(w1, N, d_0, data_or)

    chi_reg_1 = chi_reg(w1, d_0, data_or)

    omega = (
        v_1_num_prime / v_1_num  # Regularized near the vortex cores
        - 2 * jnp.pi * jnp.real(w1) / (data_or["omega_1"] ** 2 * data_or["tau_2"])
        + 1
        / 2
        * (chi_reg_1)  # Additional terms with regularization close to the vortex core
    )

    return omega


def v1_eval(w0, N, d_0, data_or):

    gamma = data_or["omega_1"] / jnp.pi
    p = data_or["tau"]

    product_ = 0

    # for n in range(0, N):

    #    product_ = product_ + (
    #        (((-1) ** n) * (p ** (n**2 + n)))
    #        * jnp.sin((2 * n + 1) * (data_or["w"] - w0) / gamma)
    #    )

    def body_fun(n, carry):
        product_ = carry
        term = product_ + (
            (((-1) ** n) * (p ** (n**2 + n)))
            * jnp.sin((2 * n + 1) * (data_or["w"] - w0) / gamma)
        )

        return product_ + term

    return jnp.where(
        jnp.abs(data_or["w"] - w0) > d_0,
        2 * p ** (1 / 4) * fori_loop(0, N, body_fun, jnp.zeros_like(data_or["w"])),
        1,  # Arbitraty value of 1 inside the circle around the vortex core
    )


# @jax.jit
def chi_reg(w0, d_0, data_or):  # location of the vortex

    return jnp.where(
        jnp.abs(data_or["w"] - w0) < d_0,
        -(data_or["lambda_u"] / data_or["lambda_iso"])
        + (data_or["lambda_v"] / data_or["lambda_iso"]) * 1j,
        0,
    )


# @jax.jit
def f_reg(w0, d_0, data_or):  # location of the vortex

    return jnp.where(
        jnp.abs(data_or["w"] - w0) < d_0, jnp.log(data_or["lambda_iso"]), 0
    )


def v1_prime_eval(w0, N, d_0, data_or):

    gamma = data_or["omega_1"] / jnp.pi
    p = data_or["tau"]

    def body_fun(n, carry):
        _product = carry
        term = (((-1) ** n) * (p ** (n**2 + n))) * (
            ((2 * n + 1) / gamma) * jnp.cos((2 * n + 1) * (data_or["w"] - w0) / gamma)
        )
        return _product + term

    return fori_loop(0, N, body_fun, jnp.zeros_like(data_or["w"]))


def comp_loc(
    theta_0_,
    phi_0_,
):
    return theta_0_ + phi_0_ * 1j


# Interpolate isothermal coordinates and interpolate on a different grid
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
            "|e_theta x e_zeta|",
        ],
        grid=s_grid,
    )

    return iso_coords_interp(tdata, s_data, w_surface)


def add_extra(data_, n_size, m_size):

    _mod = data_.reshape((n_size, m_size)).T
    _mod = jnp.column_stack([_mod, _mod[0:m_size, 0]])
    _mod = jnp.vstack([_mod, 2 * jnp.pi * jnp.ones(_mod.shape[1])])

    return _mod


def add_extra_periodic(data_, n_size, m_size):

    _mod = data_.reshape((n_size, m_size)).T
    _mod = jnp.column_stack([_mod, _mod[:, 0]])
    _mod = jnp.vstack([_mod, _mod[0, :]])

    return _mod


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


def alt_grid(theta, zeta):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten()

    return Grid(
        jnp.stack((jnp.ones_like(theta_flat), theta_flat, zeta_flat)).T, jitable=True
    )


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


def _compute_magnetic_field_from_Current(
    Kgrid, K_at_grid, surface, data, coords, basis="rpz"
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

    # Bdata = eq.compute(["R", "phi", "Z", "n_rho"], grid=Bgrid)
    # coords = jnp.vstack([Bdata["R"], Bdata["phi"], Bdata["Z"]]).T

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
    # data = surface.compute(["x", "|e_theta x e_zeta|"], grid=surface_grid, basis="xyz")

    _rs = data["x"]  # xyz2rpz(data["x"])
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


def _compute_magnetic_field_from_Current_Contour(
    Kgrid, K_at_grid, surface, data, coords, basis="rpz"
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

    # Bdata = eq.compute(["R", "phi", "Z", "n_rho"], grid=Bgrid)
    # coords = jnp.vstack([Bdata["R"], Bdata["phi"], Bdata["Z"]]).T

    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])

    surface_grid = Kgrid

    _rs = data["x"]  # xyz2rpz(data["x"])
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
