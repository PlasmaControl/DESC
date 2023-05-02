"""High order method for singular surface integrals, from Malhotra 2019."""
import numpy as np
import scipy

from desc.backend import fori_loop, jit, jnp, put
from desc.basis import DoubleFourierSeries
from desc.geometry.utils import rpz2xyz, rpz2xyz_vec


def _get_quadrature_nodes(q):
    Nr = Nw = q
    r, dr = scipy.special.roots_legendre(Nr)
    # integrate seperately over [-1,0] and [0,1]
    r1 = 1 / 2 * r - 1 / 2
    r2 = 1 / 2 * r + 1 / 2
    r = jnp.concatenate([r1, r2])
    dr = jnp.concatenate([dr, dr]) / 2
    w = np.linspace(0, np.pi, Nw, endpoint=False)
    dw = jnp.ones_like(w) * np.pi / Nw
    r, w = jnp.meshgrid(r, w)
    r = r.flatten()
    w = w.flatten()
    dr, dw = jnp.meshgrid(dr, dw)
    dr = dr.flatten()
    dw = dw.flatten()
    return r, w, dr, dw


def get_fourier_interp_matrix(eval_grid, src_grid, s, q):
    """Fourier interpolation matrix required for high order singular integration."""
    r, w, dr, dw = _get_quadrature_nodes(q)

    src_dtheta = src_grid.spacing[:, 1]
    src_dzeta = src_grid.spacing[:, 2]
    eval_theta = eval_grid.nodes[:, 1]
    eval_zeta = eval_grid.nodes[:, 2]

    h_t = jnp.mean(src_dtheta)
    h_z = jnp.mean(src_dzeta)

    theta_q = eval_theta[:, None] + s / 2 * h_t * r * jnp.sin(w)
    zeta_q = eval_zeta[:, None] + s / 2 * h_z * r * jnp.cos(w)

    basis = DoubleFourierSeries(M=src_grid.M, N=src_grid.N, NFP=src_grid.NFP)
    A = basis.evaluate(src_grid.nodes)
    Ainv = jnp.linalg.pinv(A, rcond=None)

    B = jnp.zeros((*theta_q.shape, basis.num_modes))

    def body(i, B):
        x = jnp.array(
            [
                jnp.zeros_like(theta_q[i]),
                theta_q[i],
                zeta_q[i],
            ]
        ).T
        Bi = basis.evaluate(jnp.atleast_2d(x))
        B = put(B, i, Bi)
        return B

    B = fori_loop(0, B.shape[0], body, B)
    return B @ Ainv


@jit
def _chi(rho):
    return jnp.exp(-36 * jnp.abs(rho) ** 8)


@jit
def _rho(theta, zeta, theta0, zeta0, dtheta, dzeta, M):
    dt = abs(theta - theta0)
    dz = abs(zeta - zeta0)
    dt = jnp.minimum(dt, 2 * np.pi - dt)
    dz = jnp.minimum(dz, 2 * np.pi - dz)
    return 2 / M * jnp.sqrt((dt / dtheta) ** 2 + (dz / dzeta) ** 2)


def _nonsingular_part(eval_data, eval_grid, src_data, src_grid, M, kernel):

    src_theta = src_grid.nodes[:, 1]
    src_zeta = src_grid.nodes[:, 2]
    src_dtheta = src_grid.spacing[:, 1]
    src_dzeta = src_grid.spacing[:, 2]
    eval_theta = eval_grid.nodes[:, 1]
    eval_zeta = eval_grid.nodes[:, 2]
    rho = _rho(
        src_theta,
        src_zeta,
        eval_theta[:, None],
        eval_zeta[:, None],
        src_dtheta,
        src_dzeta,
        M,
    )
    eta = _chi(rho)
    w = src_grid.weights * src_data["|e_theta x e_zeta|"]
    f = kernel(eval_data, src_data)
    if f.ndim == 2:
        f = f[:, :, None]
    A = (1 - eta[:, :, None]) * f
    return jnp.sum(A * w[None, :, None], axis=1)


def _singular_part_loop_eval(eval_data, eval_grid, src_data, src_grid, M, kernel, q, R):
    src_dtheta = src_grid.spacing[:, 1]
    src_dzeta = src_grid.spacing[:, 2]
    eval_theta = jnp.asarray(eval_grid.nodes[:, 1])
    eval_zeta = jnp.asarray(eval_grid.nodes[:, 2])
    eval_data = {key: jnp.asarray(val) for key, val in eval_data.items()}

    neval = eval_grid.num_nodes

    r, w, dr, dw = _get_quadrature_nodes(q)

    h_t = jnp.mean(src_dtheta)
    h_z = jnp.mean(src_dzeta)

    eta = _chi(r)
    v = eta * M**2 * h_t * h_z / 4 * abs(r) * dr * dw

    keys = [
        "R",
        "R_t",
        "R_z",
        "Z",
        "Z_t",
        "Z_z",
        "e^rho",
        "|e_theta x e_zeta|",
        "zeta",
        "theta",
    ]
    keys_interp = ["R", "R_t", "R_z", "Z", "Z_t", "Z_z"]
    fsrc = jnp.array([src_data[key] for key in keys_interp]).T

    def body(i, f):
        theta_q = eval_theta[i] + M / 2 * h_t * r * jnp.sin(w)
        zeta_q = eval_zeta[i] + M / 2 * h_z * r * jnp.cos(w)

        fq = R[i] @ fsrc

        src_data_polar = {key: val for key, val in zip(keys_interp, fq.T)}
        src_data_polar["zeta"] = zeta_q
        src_data_polar["theta"] = theta_q
        x_t = jnp.array(
            [
                src_data_polar["R_t"],
                jnp.zeros_like(src_data_polar["R"]),
                src_data_polar["Z_t"],
            ]
        ).T
        x_z = jnp.array(
            [src_data_polar["R_z"], src_data_polar["R"], src_data_polar["Z_z"]]
        ).T

        src_data_polar["e^rho"] = jnp.cross(x_t, x_z, axis=-1)
        src_data_polar["|e_theta x e_zeta|"] = jnp.linalg.norm(
            (src_data_polar["e^rho"]), axis=-1
        )

        k = kernel(
            {key: val[i] for key, val in eval_data.items() if key in keys},
            src_data_polar,
        )
        if k.ndim == 2:
            k = k[:, :, None]
        fi = jnp.sum(
            k * (v * src_data_polar["|e_theta x e_zeta|"])[None, :, None], axis=1
        )
        return f.at[i].set(fi.squeeze())

    f = fori_loop(0, neval, body, jnp.zeros((neval, kernel.ndim)))
    return f


def singular_integral(eval_data, eval_grid, src_data, src_grid, M, kernel, q, R):
    """Evaluate a singular integral on a surface."""
    if isinstance(kernel, str):
        kernel = kernels[kernel]
    out1 = _nonsingular_part(eval_data, eval_grid, src_data, src_grid, M, kernel)
    out2 = _singular_part_loop_eval(
        eval_data, eval_grid, src_data, src_grid, M, kernel, q, R
    )
    return out1 + out2


def kernel_nr_over_r3(eval_data, src_data):
    """n*r/|r|^2."""
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    dx = eval_x[:, None] - src_x[None]
    n = rpz2xyz_vec(src_data["e^rho"], phi=src_data["zeta"])
    n = n / jnp.linalg.norm(n, axis=-1)[:, None]
    r = jnp.linalg.norm(dx, axis=-1)
    return jnp.where(r < np.finfo(r.dtype).eps, 0, jnp.sum(n * dx, axis=-1) / r**3)


kernel_nr_over_r3.ndim = 1
kernel_nr_over_r3.keys = ["R", "zeta", "Z", "e^rho"]


def kernel_1_over_r(eval_data, src_data):
    """1/|r|."""
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    dx = eval_x[:, None] - src_x[None]
    r = jnp.linalg.norm(dx, axis=-1)
    return jnp.where(r < np.finfo(r.dtype).eps, 0, 1 / r)


kernel_1_over_r.ndim = 1
kernel_1_over_r.keys = ["R", "zeta", "Z"]


kernels = {"1_over_r": kernel_1_over_r, "nr_over_r3": kernel_nr_over_r3}
