"""High order method for singular surface integrals, from Malhotra 2019."""

from abc import ABC, abstractmethod

import scipy
from interpax import fft_interp2d
from scipy.constants import mu_0

from desc.backend import fori_loop, jnp, rfft2, vmap
from desc.basis import DoubleFourierSeries
from desc.batching import batch_map
from desc.compute.geom_utils import rpz2xyz, rpz2xyz_vec, xyz2rpz_vec
from desc.grid import LinearGrid
from desc.io import IOAble
from desc.utils import errorif, safediv, safenorm, warnif


def _get_default_sq(grid):
    k = max(min(grid.num_theta, grid.num_zeta * grid.NFP), 2)
    s = k - 1
    q = k // 2 + int(jnp.sqrt(k))
    return s, q


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
    # integrate separately over [-1,0] and [0,1]
    r1 = 1 / 2 * r - 1 / 2
    r2 = 1 / 2 * r + 1 / 2
    r = jnp.concatenate([r1, r2])
    dr = jnp.concatenate([dr, dr]) / 2
    w = jnp.linspace(0, jnp.pi, Nw, endpoint=False)
    dw = jnp.ones_like(w) * jnp.pi / Nw
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
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
    s : int
        Extent of polar grid in number of source grid points.
        Same as ``M`` in the original Malhotra papers.
    q : int
        Order of quadrature in polar domain.

    """

    _io_attrs_ = ["_eval_grid", "_source_grid", "_q", "_s"]

    def __init__(self, eval_grid, source_grid, s, q):
        assert (
            eval_grid.num_rho == source_grid.num_rho == 1
        ), "Singular integration requires grids on a single surface."
        assert (
            source_grid.nodes[0, 0] == eval_grid.nodes[0, 0]
        ), "Singular integration requires grids on the same surface."
        assert s <= source_grid.num_theta, (
            "Polar grid is invalid. "
            f"Got s = {s} > {source_grid.num_theta} = source_grid.num_theta."
        )
        assert s <= source_grid.num_zeta * source_grid.NFP, (
            "Polar grid is invalid. "
            f"Got s = {s} > {source_grid.num_zeta * source_grid.NFP} = "
            f"source_grid.num_zeta * source_grid.NFP."
        )
        self._eval_grid = eval_grid
        self._source_grid = source_grid
        self._s = s
        self._q = q

    @property
    def s(self):
        """int: Extent of polar grid in number of source grid points."""
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
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
        Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
        ``eval_grid`` resolution should at least match ``source_grid``.
    s : int
        Extent of polar grid in number of source grid points.
        Same as ``M`` in the original Malhotra papers.
    q : int
        Order of quadrature in polar domain

    """

    _io_attrs_ = _BIESTInterpolator._io_attrs_ + ["_h_t", "_h_z", "_st", "_sz"]

    def __init__(self, eval_grid, source_grid, s, q):
        assert eval_grid.can_fft2, "Got False for eval_grid.can_fft2."
        assert source_grid.can_fft2, "Got False for source_grid.can_fft2."
        # Otherwise frequency spectrum is truncated.
        assert eval_grid.num_theta >= source_grid.num_theta, (
            f"Got eval_grid.num_theta = {eval_grid.num_theta} < "
            f"{source_grid.num_theta} = source_grid.num_theta."
        )
        assert eval_grid.num_zeta >= source_grid.num_zeta, (
            f"Got eval_grid.num_zeta = {eval_grid.num_zeta} < "
            f"{source_grid.num_zeta} = source_grid.num_zeta."
        )
        # NFP may be different only if there is no toroidal variation.
        assert (eval_grid.NFP == source_grid.NFP) or eval_grid.num_zeta == 1, (
            "NFP does not match. "
            f"Got eval_grid.NFP={eval_grid.NFP} and source_grid.NFP={source_grid.NFP}."
        )
        super().__init__(eval_grid, source_grid, s, q)

        self._h_t = 2 * jnp.pi / source_grid.num_theta
        self._h_z = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
        # Change of variable requires s = M for the mapping to be bijective.
        # For s < M, the partition of unity c.o.v. ensures
        # the portion of the integral that is lost is negligible.
        r, w, _, _ = _get_quadrature_nodes(q)
        self._st = s / 2 * self._h_t * r * jnp.sin(w)
        self._sz = s / 2 * self._h_z * r * jnp.cos(w)

    def vander_polar(self, i):
        """Return Vandermonde matrix for ith polar node."""
        pass

    def __call__(self, f, i, is_fourier=False, vander=None):
        """Interpolate data to polar grid points.

        Parameters
        ----------
        f : ndarray
            Data at source grid points to interpolate.
        i : int
            Index of polar node.
        is_fourier : bool
            Whether ``f`` holds Fourier coefficients as returned by
            ``self.fourier``. Default is false.

        Returns
        -------
        fi : ndarray
            Source data interpolated to ith polar node.

        """
        errorif(is_fourier, NotImplementedError)
        shape = f.shape[1:]
        f = self._source_grid.meshgrid_reshape(f, "rtz")[0]
        return fft_interp2d(
            f,
            self._eval_grid.num_theta,
            self._eval_grid.num_zeta,
            sx=self._st[i],
            sy=self._sz[i],
            dx=self._h_t,
            dy=self._h_z,
        ).reshape(self._eval_grid.num_nodes, *shape, order="F")


class DFTInterpolator(_BIESTInterpolator):
    """Fourier interpolation matrix required for high order singular integration.

    Parameters
    ----------
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
        ``source_grid`` must be a tensor-product grid in (ρ, θ, ζ) with
        uniformly spaced nodes (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
    s : int
        Extent of polar grid in number of source grid points.
        Same as ``M`` in the original Malhotra papers.
    q : int
        Order of quadrature in polar domain

    """

    _io_attrs_ = _BIESTInterpolator._io_attrs_ + ["_basis", "_st", "_sz"]

    def __init__(self, eval_grid, source_grid, s, q):
        assert source_grid.can_fft2, "Got False for source_grid.can_fft2."
        super().__init__(eval_grid, source_grid, s, q)

        h_t = 2 * jnp.pi / source_grid.num_theta
        h_z = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
        # Change of variable requires s = M for the mapping to be bijective.
        # For s < M, the partition of unity c.o.v. ensures
        # the portion of the integral that is lost is negligible.
        r, w, _, _ = _get_quadrature_nodes(q)
        self._st = s / 2 * h_t * r * jnp.sin(w)
        self._sz = s / 2 * h_z * r * jnp.cos(w)
        self._basis = DoubleFourierSeries(
            M=source_grid.M, N=source_grid.N, NFP=source_grid.NFP, complex_vander=True
        )

    def fourier(self, f):
        """Return real FFT of ``f``."""
        return rfft2(
            self._source_grid.meshgrid_reshape(f, "rtz")[0],
            axes=(0, 1),
            norm="forward",
        ).reshape(-1, *f.shape[1:])

    def vander_polar(self, i):
        """Return Vandermonde matrix for ith polar node."""
        theta = self._eval_grid.nodes[:, 1] + self._st[i]
        zeta = self._eval_grid.nodes[:, 2] + self._sz[i]
        x = jnp.column_stack([jnp.zeros(self._eval_grid.num_nodes), theta, zeta])
        return self._basis._evaluate_complex_vander(x)

    def __call__(self, f, i, is_fourier=False, vander=None):
        """Interpolate data to polar grid points.

        Parameters
        ----------
        f : ndarray
            Data at source grid points to interpolate.
        i : int
            Index of polar node.
        is_fourier : bool
            Whether ``f`` holds Fourier coefficients as returned by
            ``self.fourier``. Default is false.
        vander : jnp.ndarray
            Cached value for ``self.vander_polar(i)``.

        Returns
        -------
        fi : ndarray
            Source data interpolated to ith polar node.

        """
        if not is_fourier:
            f = self.fourier(f)
        if vander is None:
            vander = self.vander_polar(i)
        return jnp.real(vander @ f)


def _chi(rho):
    """Partition of unity function. Eq 39 in [2]."""
    return jnp.exp(-36 * jnp.abs(rho) ** 8)


def _rho(theta, zeta, theta0, zeta0, dtheta, dzeta, s):
    """Polar grid radial coordinate. Argument of Chi in eq. 36 in [2]."""
    dt = jnp.abs(theta - theta0)
    dz = jnp.abs(zeta - zeta0)
    dt = jnp.minimum(dt, 2 * jnp.pi - dt)
    dz = jnp.minimum(dz, 2 * jnp.pi - dz)
    return 2 / s * jnp.sqrt((dt / dtheta) ** 2 + (dz / dzeta) ** 2)


def _nonsingular_part(
    eval_data, eval_grid, source_data, source_grid, s, kernel, loop=False
):
    """Integrate kernel over non-singular points.

    Generally follows sec 3.2.1 of [2].
    """
    assert source_grid.NFP == int(source_grid.NFP)
    source_theta = source_grid.nodes[:, 1]
    # make sure source dict has zeta and phi to avoid
    # adding keys to dict during iteration
    source_zeta = source_data.setdefault("zeta", source_grid.nodes[:, 2])
    source_phi = source_data["phi"]
    eval_theta = jnp.asarray(eval_grid.nodes[:, 1])
    eval_zeta = jnp.asarray(eval_grid.nodes[:, 2])
    h_t = 2 * jnp.pi / source_grid.num_theta
    h_z = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
    w = source_data["|e_theta x e_zeta|"][jnp.newaxis] * h_t * h_z

    keys = kernel.keys

    def nfp_loop(j, f_data):
        # calculate effects at all eval pts from all source pts on a single field
        # period, summing over field periods
        f, source_data = f_data
        source_data["zeta"] = (source_zeta + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )
        source_data["phi"] = (source_phi + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )

        # nest this def to avoid having to pass the modified source_data around the loop
        # easier to just close over it and let JAX figure it out
        def eval_pt(i):
            # this calculates the effect at a single evaluation point, from all others
            # in a single field period. vmap this to get all pts
            k = kernel({key: eval_data[key][i] for key in keys}, source_data).reshape(
                -1, source_grid.num_nodes, kernel.ndim
            )
            rho = _rho(
                source_theta,
                source_data["zeta"],  # to account for different field periods
                eval_theta[i][:, jnp.newaxis],
                eval_zeta[i][:, jnp.newaxis],
                h_t,
                h_z,
                s,
            )
            eta = _chi(rho)  # from eq 36 of [2]
            return jnp.sum(k * (w * (1 - eta))[..., jnp.newaxis], axis=1)

        # vmap for inner part found more efficient than loop, especially on gpu,
        # but for jacobian looped seems to be better and less memory
        f += batch_map(
            eval_pt,
            jnp.arange(eval_grid.num_nodes),
            1 if loop else None,
        ).reshape(eval_grid.num_nodes, kernel.ndim)
        return f, source_data

    f = jnp.zeros((eval_grid.num_nodes, kernel.ndim))
    f, _ = fori_loop(0, int(source_grid.NFP), nfp_loop, (f, source_data))

    # undo rotation of source_zeta
    source_data["zeta"] = source_zeta
    source_data["phi"] = source_phi
    # we sum distance vectors, so they need to be in xyz for that to work
    # but then need to convert vectors back to rpz
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def _singular_part(
    eval_data,
    eval_grid,
    source_data,
    source_grid,
    s,
    q,
    kernel,
    interpolator,
    loop=False,
):
    """Integrate singular point by interpolating to polar grid.

    Generally follows sec 3.2.2 of [2], with the following differences:

    - hyperparameter M replaced by s
    - density sigma / function f is absorbed into kernel.
    """
    h_t = 2 * jnp.pi / source_grid.num_theta
    h_z = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
    eval_theta = jnp.asarray(eval_grid.nodes[:, 1])
    eval_zeta = jnp.asarray(eval_grid.nodes[:, 2])

    # could maybe cache these in the interpolator
    r, w, dr, dw = _get_quadrature_nodes(q)
    eta = _chi(r)
    # integrand of eq 38 in [2] except stuff that needs to be interpolated
    v = eta * s**2 * h_t * h_z / 4 * jnp.abs(r) * dr * dw
    dt = s / 2 * h_t * r * jnp.sin(w)
    dz = s / 2 * h_z * r * jnp.cos(w)
    keys = set(["|e_theta x e_zeta|"] + kernel.keys)
    if "phi" in keys:
        keys.remove("phi")
        keys.add("omega")
    keys = list(keys)
    fsource = [source_data[key] for key in keys]
    if isinstance(interpolator, DFTInterpolator):
        fsource = [interpolator.fourier(val) for val in fsource]
        is_fourier = True
    else:
        # TODO: https://github.com/f0uriest/interpax/issues/53.
        is_fourier = False

    def polar_pt(i):
        """See sec 3.2.2 of [2].

        Evaluate the effect from a single polar node around each eval point
        on that eval point. Polar grids from other singularities have no effect.
        """
        # The ``FFTInterpolator`` interpolates to polar node i of the source grid, while
        # the ``DFTInterpolator`` interpolates to polar node i of the eval   grid.
        # When the source grid and eval grid have different NFP, the values taken
        # by a function that is periodic with source grid NFP will in general
        # be different at these points. For functions with no toroidal variation
        # there will be no difference, and that is the only time the
        # source grid and eval grid may have different NFP.
        vander = interpolator.vander_polar(i)
        source_data_polar = {
            key: interpolator(val, i, is_fourier, vander)
            for key, val in zip(keys, fsource)
        }
        # The (θ, ζ) coordinates at which the maps above were evaluated.
        source_data_polar["theta"] = eval_theta + dt[i]
        source_data_polar["zeta"] = eval_zeta + dz[i]
        # ϕ is not periodic map of θ, ζ.
        if "omega" in keys:
            source_data_polar["phi"] = (
                source_data_polar["zeta"] + source_data_polar["omega"]
            )

        # eval pts x source pts for 1 polar grid offset
        # only need diagonal term because polar grid points
        # don't contribute to other eval pts due to the c.o.v.
        k = kernel(eval_data, source_data_polar, diag=True).reshape(
            eval_grid.num_nodes, kernel.ndim
        )
        dS = (v[i] * source_data_polar["|e_theta x e_zeta|"])[:, jnp.newaxis]
        fi = k * dS
        return fi

    def polar_pt_loop(i, f):
        return f + polar_pt(i)

    # vmap found more efficient than fori_loop, esp on gpu, but uses more memory
    if loop:
        f = fori_loop(
            0, v.size, polar_pt_loop, jnp.zeros((eval_grid.num_nodes, kernel.ndim))
        )
    else:
        f = vmap(polar_pt)(jnp.arange(v.size)).sum(axis=0)

    # we sum distance vectors, so they need to be in xyz for that to work
    # but then need to convert vectors back to rpz
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def singular_integral(
    eval_data,
    source_data,
    kernel,
    interpolator,
    loop=False,
):
    """Evaluate a singular integral transform on a surface.

    eg f(θ, ζ) = ∫ ∫ K(θ, ζ, θ', ζ') g(θ', ζ') dθ' dζ'

    Where K(θ, ζ, θ', ζ') is the (singular) kernel and g(θ', ζ') is the metric on the
    surface. See eq. 3.7 in [1]_, but we have absorbed the density σ into K

    Uses method by Malhotra et. al. [1]_ [2]_

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points (eval_grid passed to interpolator).
        Keys should be those required by kernel as kernel.keys. Vector data should be
        in rpz basis.
    source_data : dict
        Dictionary of data at source points (source_grid passed to interpolator). Keys
        should be those required by kernel as kernel.keys. Vector data should be in
        rpz basis.
    kernel : str or callable
        Kernel function to evaluate. If str, one of the following:
            '1_over_r' : 1 / |𝐫 − 𝐫'|
            'nr_over_r3' : 𝐧'⋅(𝐫 − 𝐫') / |𝐫 − 𝐫'|³
            'biot_savart' : μ₀/4π 𝐊'×(𝐫 − 𝐫') / |𝐫 − 𝐫'|³
            'biot_savart_A' : μ₀/4π 𝐊' / |𝐫 − 𝐫'|
        If callable, should take 3 arguments:
            eval_data : dict of data at evaluation points (primed)
            source_data : dict of data at source points (unprimed)
            diag : boolean, whether to evaluate full cross interactions or just diagonal
        If a callable, should also have the attributes ``ndim`` and ``keys`` defined.
        ``ndim`` is an integer representing the dimensionality of the output function f,
        1 if f is scalar, 3 if f is a vector, etc.
        ``keys`` is a list of strings of what data is required to evaluate the kernel.
        The kernel will be called with dictionaries containing this data at source and
        evaluation points.
        If vector valued, the input to the kernel function will be in rpz and output
        should be in xyz.
    interpolator : callable
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
    loop : bool
        If True, evaluate integral using loops, as opposed to vmap. Slower, but uses
        less memory.

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, kernel.ndim)
        Integral transform evaluated at eval_grid. Vectors are in rpz basis.

    References
    ----------
    .. [1] Malhotra, Dhairya, et al. "Efficient high-order singular quadrature schemes
       in magnetic fusion." Plasma Physics and Controlled Fusion 62.2 (2019): 024004.
    .. [2] Malhotra, Dhairya, et al. "Taylor states in stellarators: A fast high-order
       boundary integral solver." Journal of Computational Physics 397 (2019): 108791.

    """
    # sanitize inputs, we need everything as jax arrays so they can be indexed
    # properly in the loops
    source_data = {key: jnp.asarray(val) for key, val in source_data.items()}
    eval_data = {key: jnp.asarray(val) for key, val in eval_data.items()}

    if isinstance(kernel, str):
        kernel = kernels[kernel]

    s, q = interpolator.s, interpolator.q
    eval_grid, source_grid = interpolator._eval_grid, interpolator._source_grid

    out2 = _singular_part(
        eval_data, eval_grid, source_data, source_grid, s, q, kernel, interpolator, loop
    )
    out1 = _nonsingular_part(
        eval_data, eval_grid, source_data, source_grid, s, kernel, loop
    )
    return out1 + out2


def _kernel_nr_over_r3(eval_data, source_data, diag=False):
    # n * r / |r|^3
    source_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([source_data["R"], source_data["phi"], source_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - source_x
    else:
        dx = eval_x[:, None] - source_x[None]
    n = rpz2xyz_vec(source_data["e^rho"], phi=source_data["phi"])
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    r = safenorm(dx, axis=-1)
    return safediv(jnp.sum(n * dx, axis=-1), r**3)


_kernel_nr_over_r3.ndim = 1
_kernel_nr_over_r3.keys = ["R", "phi", "Z", "e^rho"]


def _kernel_1_over_r(eval_data, source_data, diag=False):
    # 1/|r|
    source_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([source_data["R"], source_data["phi"], source_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - source_x
    else:
        dx = eval_x[:, None] - source_x[None]
    r = safenorm(dx, axis=-1)
    return safediv(1, r)


_kernel_1_over_r.ndim = 1
_kernel_1_over_r.keys = ["R", "phi", "Z"]


def _kernel_biot_savart(eval_data, source_data, diag=False):
    # K x r / |r|^3
    source_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([source_data["R"], source_data["phi"], source_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - source_x
    else:
        dx = eval_x[:, None] - source_x[None]
    K = rpz2xyz_vec(source_data["K_vc"], phi=source_data["phi"])
    num = jnp.cross(K, dx, axis=-1)
    r = safenorm(dx, axis=-1)[..., None]
    return mu_0 / 4 / jnp.pi * safediv(num, r**3)


_kernel_biot_savart.ndim = 3
_kernel_biot_savart.keys = ["R", "phi", "Z", "K_vc"]


def _kernel_biot_savart_A(eval_data, source_data, diag=False):
    # K  / |r|
    source_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([source_data["R"], source_data["phi"], source_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - source_x
    else:
        dx = eval_x[:, None] - source_x[None]
    r = safenorm(dx, axis=-1)[..., None]
    K = rpz2xyz_vec(source_data["K_vc"], phi=source_data["phi"])
    return mu_0 / 4 / jnp.pi * safediv(K, r)


_kernel_biot_savart_A.ndim = 3
_kernel_biot_savart_A.keys = ["R", "phi", "Z", "K_vc"]


kernels = {
    "1_over_r": _kernel_1_over_r,
    "nr_over_r3": _kernel_nr_over_r3,
    "biot_savart": _kernel_biot_savart,
    "biot_savart_A": _kernel_biot_savart_A,
}


def virtual_casing_biot_savart(eval_data, source_data, interpolator, loop=True):
    """Evaluate magnetic field on surface due to sheet current on surface.

    The magnetic field due to the plasma current can be written as a Biot-Savart
    integral over the plasma volume:

    𝐁ᵥ(𝐫) = μ₀/4π ∫ 𝐉(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d³𝐫'

    Where 𝐉 is the plasma current density, 𝐫 is a point on the plasma surface, and 𝐫' is
    a point in the plasma volume.

    This 3D integral can be converted to a 2D integral over the plasma boundary using
    the virtual casing principle [1]_

    𝐁ᵥ(𝐫) = μ₀/4π ∫ (𝐧' ⋅ 𝐁(𝐫')) (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
            + μ₀/4π ∫ (𝐧' × 𝐁(𝐫') × (𝐫 − 𝐫')/ |𝐫 − 𝐫'|³ d²𝐫'
            + 𝐁(𝐫)/2

    Where 𝐁 is the total field on the surface and 𝐧' is the outward surface normal.
    Because the total field is tangent, the first term in the integrand is zero leaving

    𝐁ᵥ(𝐫) = μ₀/4π ∫ K_vc(𝐫') × (𝐫 − 𝐫')/ |𝐫 − 𝐫'|³ d²𝐫' + 𝐁(𝐫)/2

    Where we have defined the virtual casing sheet current K_vc = 𝐧' × 𝐁(𝐫')

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points (eval_grid passed to interpolator).
        Keys should be those required by kernel as kernel.keys. Vector data should be
        in rpz basis.
    source_data : dict
        Dictionary of data at source points (source_grid passed to interpolator). Keys
        should be those required by kernel as kernel.keys. Vector data should be in
        rpz basis.
    interpolator : callable
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
    loop : bool
        If True, evaluate integral using loops, as opposed to vmap. Slower, but uses
        less memory.

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, kernel.ndim)
        Integral transform evaluated at eval_grid. Vectors are in rpz basis.

    References
    ----------
    .. [1] Hanson, James D. "The virtual-casing principle and Helmholtz’s theorem."
       Plasma Physics and Controlled Fusion 57.11 (2015): 115006.

    """
    return singular_integral(
        eval_data,
        source_data,
        _kernel_biot_savart,
        interpolator,
        loop,
    )


def compute_B_plasma(eq, eval_grid, source_grid=None, normal_only=False):
    """Evaluate magnetic field on surface due to enclosed plasma currents.

    The magnetic field due to the plasma current can be written as a Biot-Savart
    integral over the plasma volume:

    𝐁ᵥ(𝐫) = μ₀/4π ∫ 𝐉(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d³𝐫'

    Where 𝐉 is the plasma current density, 𝐫 is a point on the plasma surface, and 𝐫' is
    a point in the plasma volume.

    This 3D integral can be converted to a 2D integral over the plasma boundary using
    the virtual casing principle [1]_

    𝐁ᵥ(𝐫) = μ₀/4π ∫ (𝐧' ⋅ 𝐁(𝐫')) (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
            + μ₀/4π ∫ (𝐧' × 𝐁(𝐫') × (𝐫 − 𝐫')/ |𝐫 − 𝐫'|³ d²𝐫'
            + 𝐁(𝐫)/2

    Where 𝐁 is the total field on the surface and 𝐧' is the outward surface normal.
    Because the total field is tangent, the first term in the integrand is zero leaving

    𝐁ᵥ(𝐫) = μ₀/4π ∫ K_vc(𝐫') × (𝐫 − 𝐫')/ |𝐫 − 𝐫'|³ d²𝐫' + 𝐁(𝐫)/2

    Where we have defined the virtual casing sheet current K_vc = 𝐧' × 𝐁(𝐫')

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that is the source of the plasma current.
    eval_grid : Grid
        Evaluation points for the magnetic field.
    source_grid : Grid, optional
        Source points for integral.
    normal_only : bool
        If True, only compute and return the normal component of the plasma field 𝐁ᵥ⋅𝐧

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, 3) or shape(eval_grid.num_nodes,)
        Magnetic field evaluated at eval_grid.
        If normal_only=False, vector B is in rpz basis.

    References
    ----------
    .. [1] Hanson, James D. "The virtual-casing principle and Helmholtz’s theorem."
       Plasma Physics and Controlled Fusion 57.11 (2015): 115006.

    """
    if source_grid is None:
        source_grid = LinearGrid(
            rho=jnp.array([1.0]),
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )

    data_keys = ["K_vc", "B", "R", "phi", "Z", "e^rho", "n_rho", "|e_theta x e_zeta|"]
    eval_data = eq.compute(data_keys, grid=eval_grid)
    source_data = eq.compute(data_keys, grid=source_grid)
    s, q = _get_default_sq(source_grid)
    try:
        interpolator = FFTInterpolator(eval_grid, source_grid, s, q)
    except AssertionError as e:
        warnif(
            True,
            msg="Could not build fft interpolator, switching to dft which is slow."
            "\nReason: " + str(e),
        )
        interpolator = DFTInterpolator(eval_grid, source_grid, s, q)
    if hasattr(eq.surface, "Phi_mn"):
        source_data["K_vc"] += eq.surface.compute("K", grid=source_grid)["K"]
    Bplasma = virtual_casing_biot_savart(eval_data, source_data, interpolator)
    # need extra factor of B/2 bc we're evaluating on plasma surface
    Bplasma = Bplasma + eval_data["B"] / 2
    if normal_only:
        Bplasma = jnp.sum(Bplasma * eval_data["n_rho"], axis=1)
    return Bplasma
