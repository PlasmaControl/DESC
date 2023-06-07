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
    src_dzeta = src_grid.spacing[:, 2] / src_grid.NFP
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
def _rho(theta, zeta, theta0, zeta0, dtheta, dzeta, s):
    dt = abs(theta - theta0)
    dz = abs(zeta - zeta0)
    dt = jnp.minimum(dt, 2 * np.pi - dt)
    dz = jnp.minimum(dz, 2 * np.pi - dz)
    return 2 / s * jnp.sqrt((dt / dtheta) ** 2 + (dz / dzeta) ** 2)


def _nonsingular_part(eval_data, eval_grid, src_data, src_grid, s, kernel):

    src_theta = src_grid.nodes[:, 1]
    src_zeta = src_grid.nodes[:, 2]
    src_dtheta = src_grid.spacing[:, 1]
    src_dzeta = src_grid.spacing[:, 2] / src_grid.NFP
    eval_theta = jnp.asarray(eval_grid.nodes[:, 1])
    eval_zeta = jnp.asarray(eval_grid.nodes[:, 2])
    w = src_grid.weights * src_data["|e_theta x e_zeta|"] / src_grid.NFP
    keys = kernel.keys

    def body2(j, f_data):
        f, src_data = f_data
        src_data["zeta"] = (src_zeta + j * 2 * np.pi / src_grid.NFP) % (2 * np.pi)

        # nest this def to avoid having to pass the modified src_data around the loop
        # easier to just close over it and let JAX figure it out
        def body1(i, fj):
            k = kernel(
                {key: val[i] for key, val in eval_data.items() if key in keys},
                src_data,
            )
            rho = _rho(
                src_theta,
                src_data["zeta"],
                eval_theta[i],
                eval_zeta[i],
                src_dtheta,
                src_dzeta,
                s,
            )
            if k.ndim == 2:  # so that broadcasting works correctly for biot savart
                k = k[:, :, None]
            eta = _chi(rho)
            A = (1 - eta)[None, :, None] * k
            f_temp = jnp.sum(A * w[None, :, None], axis=1)
            return fj.at[i].set(f_temp.squeeze())

        fj = jnp.zeros((eval_grid.num_nodes, kernel.ndim))
        fj = fori_loop(0, eval_grid.num_nodes, body1, fj)
        f += fj
        return f, src_data

    f = jnp.zeros((eval_grid.num_nodes, kernel.ndim))
    f, _ = fori_loop(0, int(src_grid.NFP), body2, (f, src_data))

    return f


def _singular_part(
    eval_data, eval_grid, src_data, src_grid, s, q, kernel, interp_matrix
):
    src_dtheta = src_grid.spacing[:, 1]
    src_dzeta = src_grid.spacing[:, 2] / src_grid.NFP
    eval_theta = jnp.asarray(eval_grid.nodes[:, 1])
    eval_zeta = jnp.asarray(eval_grid.nodes[:, 2])

    r, w, dr, dw = _get_quadrature_nodes(q)

    h_t = jnp.mean(src_dtheta)
    h_z = jnp.mean(src_dzeta)

    eta = _chi(r)
    v = eta * s**2 * h_t * h_z / 4 * abs(r) * dr * dw

    keys = list(set(["|e_theta x e_zeta|"] + kernel.keys))
    fsrc = [src_data[key] for key in keys]

    def body(i, f):
        theta_q = eval_theta[i] + s / 2 * h_t * r * jnp.sin(w)
        zeta_q = eval_zeta[i] + s / 2 * h_z * r * jnp.cos(w)

        src_data_polar = {key: interp_matrix[i] @ val for key, val in zip(keys, fsrc)}
        src_data_polar["zeta"] = zeta_q
        src_data_polar["theta"] = theta_q

        k = kernel(
            {key: val[i] for key, val in eval_data.items() if key in keys},
            src_data_polar,
        )
        if k.ndim == 2:  # so that broadcasting works correctly for biot savart
            k = k[:, :, None]
        fi = jnp.sum(
            k * (v * src_data_polar["|e_theta x e_zeta|"])[None, :, None], axis=1
        )
        return f.at[i].set(fi.squeeze())

    f = fori_loop(
        0, eval_grid.num_nodes, body, jnp.zeros((eval_grid.num_nodes, kernel.ndim))
    )
    return f


def singular_integral(
    eval_data, eval_grid, src_data, src_grid, s, q, kernel, interp_matrix
):
    """Evaluate a singular integral on a surface."""
    # sanitize inputs, we need everything as jax arrays so they can be indexed
    # properly in the loops
    src_data = {key: jnp.asarray(val) for key, val in src_data.items()}
    eval_data = {key: jnp.asarray(val) for key, val in eval_data.items()}

    if isinstance(kernel, str):
        kernel = kernels[kernel]

    out2 = _singular_part(
        eval_data, eval_grid, src_data, src_grid, s, q, kernel, interp_matrix
    )
    out1 = _nonsingular_part(eval_data, eval_grid, src_data, src_grid, s, kernel)
    return out1 + out2


def _kernel_nr_over_r3(eval_data, src_data):
    # n * r / |r|^3
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


_kernel_nr_over_r3.ndim = 1
_kernel_nr_over_r3.keys = ["R", "zeta", "Z", "e^rho"]


def _kernel_1_over_r(eval_data, src_data):
    # 1/ |r|
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    dx = eval_x[:, None] - src_x[None]
    r = jnp.linalg.norm(dx, axis=-1)
    return jnp.where(r < np.finfo(r.dtype).eps, 0, 1 / r)


_kernel_1_over_r.ndim = 1
_kernel_1_over_r.keys = ["R", "zeta", "Z"]


def _kernel_biot_savart(eval_data, src_data):
    # K x r / |r|^3
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    dx = eval_x[:, None] - src_x[None]
    K = rpz2xyz_vec(src_data["K_vc"], phi=src_data["zeta"])
    num = jnp.cross(K, dx, axis=-1)
    r = jnp.linalg.norm(dx, axis=-1)[:, :, None]
    return 1e-7 * jnp.where(r < np.finfo(r.dtype).eps, 0, num / (r * r * r))


_kernel_biot_savart.ndim = 3
_kernel_biot_savart.keys = ["R", "zeta", "Z", "K_vc"]


def _kernel_biot_savart_A(eval_data, src_data):
    # K  / |r|
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    dx = eval_x[:, None] - src_x[None]
    K = rpz2xyz_vec(src_data["K_vc"], phi=src_data["zeta"])
    r = jnp.linalg.norm(dx, axis=-1)[:, :, None]
    return 1e-7 * jnp.where(r < np.finfo(r.dtype).eps, 0, K / r)


_kernel_biot_savart_A.ndim = 3
_kernel_biot_savart_A.keys = ["R", "zeta", "Z", "K_vc"]


kernels = {
    "1_over_r": _kernel_1_over_r,
    "nr_over_r3": _kernel_nr_over_r3,
    "biot_savart": _kernel_biot_savart,
    "biot_savart_A": _kernel_biot_savart_A,
}
