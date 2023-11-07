"""High order method for singular surface integrals, from Malhotra 2019."""
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import scipy

from desc.backend import custom_jvp, fori_loop, jnp, put, vmap
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz, rpz2xyz_vec
from desc.interpolate import fft_interp2d
from desc.io import IOAble
from desc.utils import isalmostequal, islinspaced


def _get_quadrature_nodes(q):
    """Polar nodes for quadrature around singular point.

    Parameters
    ----------
    q : int
        Order of quadrature in radial and azimuthal directions.

    Returns
    -------
    r, w : ndarray
        Radial and azimuthal coordinates
    dr, dw : ndarray
        Radial and azimuthal spacing/quadrature weights

    """
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


class _BIESTInterpolator(IOAble, ABC):
    """Base class for interpolators from cartesian to polar domain.

    Used for singular integral calculations.

    Parameters
    ----------
    eval_grid, src_grid : Grid
        Evaluation and source points for the integral transform.
        src_grid should be a LinearGrid
    s : int
        Extent of polar grid in number of src grid points. Same as "M" in the
        original Malhotra papers.
    q : int
        Order of quadrature in polar domain

    """

    _io_attrs_ = ["_eval_grid", "_src_grid", "_q", "_s"]

    @abstractmethod
    def __init__(self, eval_grid, src_grid, s, q):
        pass

    @property
    def s(self):
        """int: Extent of polar grid in number of src grid points."""
        return self._s

    @property
    def q(self):
        """int: Order of quadrature in polar domain."""
        return self._q

    @abstractmethod
    def __call__(self, f, i):
        """Interpolate data to polar grid points.

        Parameters
        ----------
        f : ndarray
            Data at source grid points to interpolate.
        i : int
            Index of polar node.

        Returns
        -------
        fi : ndarray
            Source data interpolated to ith polar node.
        """


class FFTInterpolator(_BIESTInterpolator):
    """FFT interpolation operator required for high order singular integration.

    Parameters
    ----------
    eval_grid, src_grid : Grid
        Evaluation and source points for the integral transform.
        src_grid should be a LinearGrid
    s : int
        Extent of polar grid in number of src grid points. Same as "M" in the
        original Malhotra papers.
    q : int
        Order of quadrature in polar domain

    """

    _io_attrs_ = _BIESTInterpolator._io_attrs_ + ["_h_t", "_h_z", "_st", "_sz"]

    def __init__(self, eval_grid, src_grid, s, q):
        # need src_grid to be linearly spaced in theta, zeta,
        # and contain only 1 rho value
        assert isalmostequal(
            src_grid.nodes[:, 0]
        ), "singular integration requires source grid on a single surface"
        assert src_grid.num_nodes == (
            src_grid.num_theta * src_grid.num_zeta
        ), "singular integration requires a tensor product grid in theta and zeta"
        src_theta = src_grid.nodes[:, 1].reshape(
            (src_grid.num_zeta, src_grid.num_theta)
        )
        src_zeta = src_grid.nodes[:, 2].reshape((src_grid.num_zeta, src_grid.num_theta))
        assert isalmostequal(
            src_theta, axis=0
        ), "singular integration requires rectangular source grid in theta and zeta"
        assert isalmostequal(
            src_zeta, axis=1
        ), "singular integration requires rectangular source grid in theta and zeta"
        assert islinspaced(
            src_theta, axis=1
        ), "singular integration requires source nodes be equally spaced in theta"
        assert islinspaced(
            src_zeta, axis=0
        ), "singular integration requires source nodes be equally spaced in zeta"

        # need eval_grid to be linearly spaced in theta, zeta,
        # and contain only 1 rho value
        assert isalmostequal(
            eval_grid.nodes[:, 0]
        ), "singular integration requires eval grid on a single surface"
        assert eval_grid.num_nodes == (
            eval_grid.num_theta * eval_grid.num_zeta
        ), "singular integration requires a tensor product grid in theta and zeta"
        eval_theta = eval_grid.nodes[:, 1].reshape(
            (eval_grid.num_zeta, eval_grid.num_theta)
        )
        eval_zeta = eval_grid.nodes[:, 2].reshape(
            (eval_grid.num_zeta, eval_grid.num_theta)
        )
        assert isalmostequal(
            eval_theta, axis=0
        ), "singular integration requires rectangular eval grid in theta and zeta"
        assert isalmostequal(
            eval_zeta, axis=1
        ), "singular integration requires rectangular eval grid in theta and zeta"
        assert islinspaced(
            eval_theta, axis=1
        ), "singular integration requires eval nodes be equally spaced in theta"
        assert islinspaced(
            eval_zeta, axis=0
        ), "singular integration requires eval nodes be equally spaced in zeta"

        self._eval_grid = eval_grid
        self._src_grid = src_grid
        self._s = s
        self._q = q

        r, w, dr, dw = _get_quadrature_nodes(q)

        src_dtheta = src_grid.spacing[:, 1]
        src_dzeta = src_grid.spacing[:, 2] / src_grid.NFP

        self._h_t = jnp.mean(src_dtheta)
        self._h_z = jnp.mean(src_dzeta)

        self._st = s / 2 * self._h_t * r * jnp.sin(w)
        self._sz = s / 2 * self._h_z * r * jnp.cos(w)

    def __call__(self, f, i):
        """Interpolate data to polar grid points.

        Parameters
        ----------
        f : ndarray
            Data at source grid points to interpolate.
        i : int
            Index of polar node.

        Returns
        -------
        fi : ndarray
            Source data interpolated to ith polar node.
        """
        shp = f.shape[1:]
        f = f.reshape(
            (self._src_grid.num_theta, self._src_grid.num_zeta, -1), order="F"
        )
        g = fft_interp2d(
            f,
            self._eval_grid.num_theta,
            self._eval_grid.num_zeta,
            sx=self._st[i],
            sy=self._sz[i],
            dx=self._h_t,
            dy=self._h_z,
        )
        return g.reshape((self._eval_grid.num_nodes, *shp), order="F")


class DFTInterpolator(_BIESTInterpolator):
    """Fourier interpolation matrix required for high order singular integration.

    Parameters
    ----------
    eval_grid, src_grid : Grid
        Evaluation and source points for the integral transform.
        src_grid should be a LinearGrid
    s : int
        Extent of polar grid in number of src grid points. Same as "M" in the
        original Malhotra papers.
    q : int
        Order of quadrature in polar domain

    """

    _io_attrs_ = _BIESTInterpolator._io_attrs_ + ["_mat"]

    def __init__(self, eval_grid, src_grid, s, q):
        # need src_grid to be linearly spaced in theta, zeta,
        # and contain only 1 rho value
        assert isalmostequal(
            src_grid.nodes[:, 0]
        ), "singular integration requires source grid on a single surface"
        assert src_grid.num_nodes == (
            src_grid.num_theta * src_grid.num_zeta
        ), "singular integration requires a tensor product grid in theta and zeta"
        src_theta = src_grid.nodes[:, 1].reshape(
            (src_grid.num_zeta, src_grid.num_theta)
        )
        src_zeta = src_grid.nodes[:, 2].reshape((src_grid.num_zeta, src_grid.num_theta))
        assert isalmostequal(
            src_theta, axis=0
        ), "singular integration requires rectangular source grid in theta and zeta"
        assert isalmostequal(
            src_zeta, axis=1
        ), "singular integration requires rectangular source grid in theta and zeta"
        assert islinspaced(
            src_theta, axis=1
        ), "singular integration requires source nodes be equally spaced in theta"
        assert islinspaced(
            src_zeta, axis=0
        ), "singular integration requires source nodes be equally spaced in zeta"

        self._eval_grid = eval_grid
        self._src_grid = src_grid
        self._s = s
        self._q = q

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
        self._mat = B @ Ainv

    def __call__(self, f, i):
        """Interpolate data to polar grid points.

        Parameters
        ----------
        f : ndarray
            Data at source grid points to interpolate.
        i : int
            Index of polar node.

        Returns
        -------
        fi : ndarray
            Source data interpolated to ith polar node.
        """
        return self._mat[:, i] @ f


def _chi(rho):
    """Partition of unity function."""
    return jnp.exp(-36 * jnp.abs(rho) ** 8)


def _rho(theta, zeta, theta0, zeta0, dtheta, dzeta, s):
    """Polar grid radial coordinate."""
    dt = abs(theta - theta0)
    dz = abs(zeta - zeta0)
    dt = jnp.minimum(dt, 2 * np.pi - dt)
    dz = jnp.minimum(dz, 2 * np.pi - dz)
    return 2 / s * jnp.sqrt((dt / dtheta) ** 2 + (dz / dzeta) ** 2)


def _nonsingular_part(
    eval_data, eval_grid, src_data, src_grid, s, kernel, mask=True, loop=False
):
    """Integrate kernel over non-singular points."""
    assert isinstance(src_grid.NFP, int)
    src_theta = src_grid.nodes[:, 1]
    src_zeta = src_grid.nodes[:, 2]
    src_dtheta = src_grid.spacing[:, 1]
    src_dzeta = src_grid.spacing[:, 2] / src_grid.NFP
    eval_theta = jnp.asarray(eval_grid.nodes[:, 1])
    eval_zeta = jnp.asarray(eval_grid.nodes[:, 2])
    w = src_grid.weights * src_data["|e_theta x e_zeta|"] / src_grid.NFP
    h_t = jnp.mean(src_dtheta)
    h_z = jnp.mean(src_dzeta)

    keys = kernel.keys

    def nfp_loop(j, f_data):
        # calculate effects at all eval pts from all src pts on a single field period,
        # summing over field periods
        f, src_data = f_data
        src_data["zeta"] = (src_zeta + j * 2 * np.pi / src_grid.NFP) % (2 * np.pi)

        # nest this def to avoid having to pass the modified src_data around the loop
        # easier to just close over it and let JAX figure it out
        def eval_pt_vmap(i):
            # this calculates the effect at a single evaluation point, from all others
            # in a single field period. vmap this to get all pts
            k = kernel(
                {key: val[i] for key, val in eval_data.items() if key in keys},
                src_data,
            )
            if kernel.ndim == 1:  # so that broadcasting works correctly
                k = k[:, :, None]

            if mask:
                rho = _rho(
                    src_theta,
                    src_data["zeta"],  # to account for different field periods
                    eval_theta[i],
                    eval_zeta[i],
                    h_t,
                    h_z,
                    s,
                )

                eta = _chi(rho)
                k = (1 - eta)[None, :, None] * k
            f_temp = jnp.sum(k * w[None, :, None], axis=1)
            return f_temp.squeeze()

        def eval_pt_loop(i, fj):
            # this calculates the effect at a single evaluation point, from all others
            # in a single field period. loop this to get all pts
            k = kernel(
                {key: val[i] for key, val in eval_data.items() if key in keys},
                src_data,
            )
            if kernel.ndim == 1:  # so that broadcasting works correctly
                k = k[:, :, None]

            if mask:
                rho = _rho(
                    src_theta,
                    src_data["zeta"],  # to account for different field periods
                    eval_theta[i],
                    eval_zeta[i],
                    h_t,
                    h_z,
                    s,
                )

                eta = _chi(rho)
                k = (1 - eta)[None, :, None] * k
            f_temp = jnp.sum(k * w[None, :, None], axis=1)
            return put(fj, i, f_temp.squeeze())

        # vmap for inner part found more efficient than fori_loop, especially on gpu,
        # but for jacobian looped seems to be better and less memory
        if loop:
            fj = fori_loop(0, eval_grid.num_nodes, eval_pt_loop, jnp.zeros_like(f))
        else:
            fj = vmap(eval_pt_vmap)(jnp.arange(eval_grid.num_nodes))

        f += fj
        return f, src_data

    f = jnp.zeros((eval_grid.num_nodes, kernel.ndim))
    f, _ = fori_loop(0, src_grid.NFP, nfp_loop, (f, src_data))

    return f


def _singular_part(
    eval_data, eval_grid, src_data, src_grid, s, q, kernel, interpolator
):
    """Integrate singular point by interpolating to polar grid."""
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

    def polar_pt_loop(i):
        # evaluate the effect from a single polar node around each singular point
        # on that singular point. Polar grids from other singularities have no effect
        dt = s / 2 * h_t * r[i] * jnp.sin(w[i])
        dz = s / 2 * h_z * r[i] * jnp.cos(w[i])
        theta_i = eval_theta + dt
        zeta_i = eval_zeta + dz

        # data interpolated to each eval pt offset by dt,dz
        src_data_polar = {key: interpolator(val, i) for key, val in zip(keys, fsrc)}

        src_data_polar["zeta"] = zeta_i
        src_data_polar["theta"] = theta_i

        # eval pts x src pts for 1 polar grid offset
        # only need diagonal term because polar grid points
        # don't contribute to other eval pts
        k = kernel(
            {key: val for key, val in eval_data.items() if key in keys},
            src_data_polar,
            diag=True,
        )
        if kernel.ndim == 1:  # so that broadcasting works correctly
            k = k[:, None]
        dS = (v[i] * src_data_polar["|e_theta x e_zeta|"])[:, None]
        fi = k * dS
        return fi

    # vmap found more efficient than fori_loop, esp on gpu
    return vmap(polar_pt_loop)(jnp.arange(v.size)).sum(axis=0)


def _singular_integral_exact(
    eval_data, eval_grid, src_data, src_grid, kernel, interpolator
):

    s, q = interpolator.s, interpolator.q

    out2 = _singular_part(
        eval_data, eval_grid, src_data, src_grid, s, q, kernel, interpolator
    )
    out1 = _nonsingular_part(eval_data, eval_grid, src_data, src_grid, s, kernel)
    return out1 + out2


@partial(custom_jvp, nondiff_argnums=(4, 5))
def _singular_integral_approx(
    eval_data, eval_grid, src_data, src_grid, kernel, interpolator
):
    return _singular_integral_exact(
        eval_data, eval_grid, src_data, src_grid, kernel, interpolator
    )


@_singular_integral_approx.defjvp
def _singular_integral_jvp(kernel, interpolator, primals, tangents):
    import jax

    def foo(*args):
        return _nonsingular_part(
            *args, s=interpolator.s, kernel=kernel, mask=False, loop=True
        )

    return jax.jvp(foo, primals, tangents)


def singular_integral(
    eval_data, eval_grid, src_data, src_grid, kernel, interpolator, approxdf=True
):
    """Evaluate a singular integral transform on a surface.

    eg f(θ, ζ) = ∫ ∫ K(θ, ζ, θ', ζ') g(θ', ζ') dθ' dζ'

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points. Keys should be those required by
        kernel as kernel.keys
    eval_grid : Grid
        Points where integral transform is to be evaluated (eg unprimed coordinates).
    src_data : dict
        Dictionary of data at source points. Keys should be those required by
        kernel as kernel.keys
    src_grid : LinearGrid
        Source points for integral (eg primed coordinates). Should be linearly spaced
        rectangular grid in both theta, zeta.
    kernel : str or callable
        Kernel function to evaluate. Should take 3 arguments:
            eval_data : dict of data at evaluation points
            src_data : dict of data at source points
            diag : boolean, whether to evaluate full cross interations or just diagonal
        If a callable, should also have the attributes ``ndim`` and ``keys`` defined.
        ``ndim`` is an integer representing the dimensionality of the output function f,
        1 if f is scalar, 3 if f is a vector, etc.
        ``keys`` is a list of strings of what data is required to evaluate the kernel.
        The kernel will be called with dictionaries containing this data at source and
        evaluation points
    interpolator : callable
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
    approxdf : bool
        Whether to use approximate derivative when calculating jacobian. Use of an
        approximate derivative is significantly faster

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, kernel.ndim)
        Integral transform evaluated at eval_grid.

    """
    # sanitize inputs, we need everything as jax arrays so they can be indexed
    # properly in the loops
    src_data = {key: jnp.asarray(val) for key, val in src_data.items()}
    eval_data = {key: jnp.asarray(val) for key, val in eval_data.items()}

    if isinstance(kernel, str):
        kernel = kernels[kernel]

    if approxdf:
        return _singular_integral_approx(
            eval_data, eval_grid, src_data, src_grid, kernel, interpolator
        )
    else:
        return _singular_integral_exact(
            eval_data, eval_grid, src_data, src_grid, kernel, interpolator
        )


def _kernel_nr_over_r3(eval_data, src_data, diag=False):
    # n * r / |r|^3
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - src_x
    else:
        dx = eval_x[:, None] - src_x[None]
    n = rpz2xyz_vec(src_data["e^rho"], phi=src_data["zeta"])
    n = n / jnp.linalg.norm(n, axis=-1)[:, None]
    r = jnp.linalg.norm(dx, axis=-1)
    return jnp.where(r < np.finfo(r.dtype).eps, 0, jnp.sum(n * dx, axis=-1) / r**3)


_kernel_nr_over_r3.ndim = 1
_kernel_nr_over_r3.keys = ["R", "zeta", "Z", "e^rho"]


def _kernel_1_over_r(eval_data, src_data, diag=False):
    # 1/ |r|
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - src_x
    else:
        dx = eval_x[:, None] - src_x[None]
    r = jnp.linalg.norm(dx, axis=-1)
    return jnp.where(r < np.finfo(r.dtype).eps, 0, 1 / r)


_kernel_1_over_r.ndim = 1
_kernel_1_over_r.keys = ["R", "zeta", "Z"]


def _kernel_biot_savart(eval_data, src_data, diag=False):
    # K x r / |r|^3
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - src_x
    else:
        dx = eval_x[:, None] - src_x[None]
    K = rpz2xyz_vec(src_data["K_vc"], phi=src_data["zeta"])
    num = jnp.cross(K, dx, axis=-1)
    r = jnp.linalg.norm(dx, axis=-1)
    if diag:
        r = r[:, None]
    else:
        r = r[:, :, None]
    return 1e-7 * jnp.where(r < np.finfo(r.dtype).eps, 0, num / (r * r * r))


_kernel_biot_savart.ndim = 3
_kernel_biot_savart.keys = ["R", "zeta", "Z", "K_vc"]


def _kernel_biot_savart_A(eval_data, src_data, diag=False):
    # K  / |r|
    src_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([src_data["R"], src_data["zeta"], src_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - src_x
    else:
        dx = eval_x[:, None] - src_x[None]
    K = rpz2xyz_vec(src_data["K_vc"], phi=src_data["zeta"])
    r = jnp.linalg.norm(dx, axis=-1)
    if diag:
        r = r[:, None]
    else:
        r = r[:, :, None]
    return 1e-7 * jnp.where(r < np.finfo(r.dtype).eps, 0, K / r)


_kernel_biot_savart_A.ndim = 3
_kernel_biot_savart_A.keys = ["R", "zeta", "Z", "K_vc"]


kernels = {
    "1_over_r": _kernel_1_over_r,
    "nr_over_r3": _kernel_nr_over_r3,
    "biot_savart": _kernel_biot_savart,
    "biot_savart_A": _kernel_biot_savart_A,
}


def virtual_casing_biot_savart(
    eval_data, eval_grid, src_data, src_grid, interpolator, approxdf=True
):
    """Evaluate magnetic field on surface due to sheet current on surface.

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points. Keys should be those required by
        kernel as kernel.keys
    eval_grid : Grid
        Points where integral transform is to be evaluated (eg unprimed coordinates).
    src_data : dict
        Dictionary of data at source points. Keys should be those required by
        kernel as kernel.keys
    src_grid : LinearGrid
        Source points for integral (eg primed coordinates). Should be linearly spaced
        rectangular grid in both theta, zeta.
    interpolator : callable
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
    approxdf : bool
        Whether to use approximate derivative when calculating jacobian. Use of an
        approximate derivative is significantly faster

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, kernel.ndim)
        Integral transform evaluated at eval_grid.

    """
    return singular_integral(
        eval_data,
        eval_grid,
        src_data,
        src_grid,
        _kernel_biot_savart,
        interpolator,
        approxdf,
    )
