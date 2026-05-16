"""High order method for singular surface integrals, from Malhotra 2019."""

import warnings
from abc import ABC, abstractmethod
from functools import partial

from interpax_fft import rfft2_modes, rfft2_vander, rfft_interp2d
from scipy.constants import mu_0

from desc.backend import jit, jnp, rfft2
from desc.batching import batch_map, vmap_chunked
from desc.grid import LinearGrid  # noqa: F401
from desc.integrals.quad_utils import (
    _best_params,
    _best_ratio,
    _get_polar_quadrature,
    chi,
    eta,
    nfp_loop,
)
from desc.io import IOAble
from desc.utils import (
    apply,
    check_posint,
    dot,
    parse_argname_change,
    rpz2xyz,
    rpz2xyz_vec,
    safediv,
    safenorm,
    setdefault,
    warnif,
    xyz2rpz_vec,
)


def get_interpolator(
    eval_grid,
    source_grid,
    source_data,
    st=None,
    sz=None,
    q=None,
    *,
    use_dft=False,
    warn_fft=True,
    **kwargs,
):
    """Get interpolator from Cartesian to polar domain.

    Parameters
    ----------
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
    source_data : dict[str, jnp.ndarray]
        Dictionary of data evaluated on single flux surface grid that ``can_fft2``
        with keys ``|e_theta x e_zeta|``, ``e_theta``, and ``e_zeta``.
    use_dft : bool
        Whether to use matrix multiplication transform from spectral to physical domain
        instead of inverse fast Fourier transform.
    warn_fft : bool
        Whether to warn if the interpolation will be lossy. Default is ``True``.

    Returns
    -------
    f : _BIESTInterpolator
        Interpolator that uses the specified method.

    """
    if st is None or sz is None or q is None:
        _st, _sz, _q = _best_params(source_grid, _best_ratio(source_data))
        st = setdefault(st, _st)
        sz = setdefault(sz, _sz)
        q = setdefault(q, _q)

    if use_dft:
        f = DFTInterpolator(eval_grid, source_grid, st, sz, q)
    else:
        try:
            f = FFTInterpolator(eval_grid, source_grid, st, sz, q, warn_fft=warn_fft)
        except AssertionError as e:
            warnings.warn(
                "Could not build FFT interpolator because:\n"
                + str(e)
                + "\nSwitching to DFT interpolator which is more expensive.",
            )
            f = DFTInterpolator(eval_grid, source_grid, st, sz, q)
            use_dft = True

    # TODO (#1599).
    warnif(
        use_dft,
        RuntimeWarning,
        msg="Computations may be performed incorrectly for large matrices "
        "due to open issues with JAX. Until this is fixed, it is recommended to "
        "validate results against computations with a small choice for chunk size.",
    )
    return f


class _BIESTInterpolator(IOAble, ABC):
    """Base class for interpolators from cartesian to polar domain.

    Used for singular integral calculations.

    Parameters
    ----------
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
    st, sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.
    q : int
        Order of quadrature in polar domain.

    """

    _io_attrs_ = [
        "_eval_grid",
        "_source_grid",
        "_st",
        "_sz",
        "_q",
        "_ht",
        "_hz",
        "_shift_t",
        "_shift_z",
    ]

    _static_attrs = ["_q"]

    def __init__(self, eval_grid, source_grid, st, sz, q):
        check_posint(eval_grid.NFP)
        check_posint(source_grid.NFP)
        assert source_grid.can_fft2, "Got False for source_grid.can_fft2."
        # NFP may be different only if there is no toroidal variation.
        assert (eval_grid.NFP == source_grid.NFP) or source_grid.num_zeta == 1, (
            "NFP does not match. "
            f"Got eval_grid.NFP={eval_grid.NFP} and source_grid.NFP={source_grid.NFP}."
        )
        assert (
            eval_grid.num_rho == source_grid.num_rho == 1
        ), "Singular integration requires grids on a single surface."
        assert (
            source_grid.nodes[0, 0] == eval_grid.nodes[0, 0]
        ), "Singular integration requires grids on the same surface."
        assert st <= source_grid.num_theta, (
            "Polar grid is invalid. "
            f"Got st = {st} > {source_grid.num_theta} = source_grid.num_theta."
        )
        assert sz <= source_grid.num_zeta * source_grid.NFP, (
            "Polar grid is invalid. "
            f"Got sz = {sz} > {source_grid.num_zeta * source_grid.NFP} = "
            "source_grid.num_zeta * source_grid.NFP."
        )
        self._eval_grid = eval_grid
        self._source_grid = source_grid
        self._st = st
        self._sz = sz
        self._q = q
        self._ht = 2 * jnp.pi / source_grid.num_theta
        self._hz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
        r, w, _, _ = _get_polar_quadrature(q)
        self._shift_t = self._ht * st / 2 * r * jnp.sin(w)
        self._shift_z = self._hz * sz / 2 * r * jnp.cos(w)

    @property
    def eval_grid(self):
        """Evaluation points."""
        return self._eval_grid

    @property
    def source_grid(self):
        """Source points for quadrature of kernels."""
        return self._source_grid

    @property
    def st(self):
        """Extent of polar grid support.

        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.
        """
        return self._st

    @property
    def sz(self):
        """Extent of polar grid support.

        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.
        """
        return self._sz

    @property
    def q(self):
        """int: Order of quadrature in polar domain."""
        return self._q

    @property
    def ht(self):
        """float: Source grid θ spacing."""
        return self._ht

    @property
    def hz(self):
        """float: Source grid ζ spacing."""
        return self._hz

    @property
    def shift_t(self):
        """jnp.ndarray: θ shift to polar nodes."""
        return self._shift_t

    @property
    def shift_z(self):
        """jnp.ndarray: ζ shift to polar nodes."""
        return self._shift_z

    def vander_polar(self, i):
        """Return Vandermonde matrix for ith polar node."""
        pass

    def fourier(self, f):
        """Return Fourier transform of ``f`` as expected by this interpolator."""
        return f

    @abstractmethod
    def __call__(self, f, i, *, vander=None):
        """Interpolate ``f`` to polar node ``i`` around evaluation grid.

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
    """FFT interpolation operator for high order polar quadrature.

    Parameters
    ----------
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
        Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
        ``eval_grid`` resolution should at least match ``source_grid``.
    st, sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.
    q : int
        Order of quadrature in polar domain.
    warn_fft : bool
        Whether to warn if the interpolation will be lossy. Default is ``True``.

    """

    def __init__(self, eval_grid, source_grid, st, sz, q, *, warn_fft=True, **kwargs):
        st = parse_argname_change(st, kwargs, "s", "st")
        assert eval_grid.can_fft2, "Got False for eval_grid.can_fft2."
        warnif(
            warn_fft and eval_grid.num_theta < (source_grid.num_theta // 2 + 1),
            msg="Frequency spectrum of FFT interpolation will be truncated.\n"
            f"Got eval_grid.num_theta = {eval_grid.num_theta} < "
            f"{source_grid.num_theta // 2 + 1} = source_grid.num_theta // 2 + 1.",
        )
        warnif(
            warn_fft and eval_grid.num_zeta < (source_grid.num_zeta // 2 + 1),
            msg="Frequency spectrum of FFT interpolation will be truncated.\n"
            f"Got eval_grid.num_zeta = {eval_grid.num_zeta} < "
            f"{source_grid.num_zeta // 2 + 1} = source_grid.num_zeta // 2 + 1.",
        )
        super().__init__(eval_grid, source_grid, st, sz, q)

    def fourier(self, f):
        """Return Fourier transform of ``f`` as expected by this interpolator."""
        return self.source_grid.meshgrid_reshape(f, "rtz")[0]

    def __call__(self, f, i, *, is_fourier=False, vander=None):
        """Interpolate ``f`` to polar node ``i`` around evaluation grid.

        Notes
        -----
        This actually interpolates ``f`` to polar node ``i`` around the source grid.
        This is different from the expected point when the grids differ in the
        number of field periods. Functions without toroidal variation take the
        same value at these different points. Hence, the only case where the
        source grid and the evaluation grid may differ in the number of field
        periods is when the source grid can only capture functions without
        toroidal variation.

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
        if not is_fourier:
            f = self.fourier(f)
        return rfft_interp2d(
            f,
            n1=self.eval_grid.num_theta,
            n2=self.eval_grid.num_zeta,
            sx=self._shift_t[i],
            sy=self._shift_z[i],
            dx=self._ht,
            dy=self._hz,
        ).reshape(self.eval_grid.num_nodes, *f.shape[2:], order="F")


class DFTInterpolator(_BIESTInterpolator):
    """MMT interpolation operator for high order polar quadrature.

    Parameters
    ----------
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
        ``source_grid`` must be a tensor-product grid in (ρ, θ, ζ) with
        uniformly spaced nodes (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
    st, sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.
    q : int
        Order of quadrature in polar domain.

    """

    _io_attrs_ = _BIESTInterpolator._io_attrs_ + ["_modes_fft", "_modes_rfft"]

    def __init__(self, eval_grid, source_grid, st, sz, q, **kwargs):
        st = parse_argname_change(st, kwargs, "s", "st")
        super().__init__(eval_grid, source_grid, st, sz, q)
        self._modes_fft, self._modes_rfft = rfft2_modes(
            source_grid.num_theta,
            source_grid.num_zeta,
            domain_rfft=(0, 2 * jnp.pi / source_grid.NFP),
        )

    def fourier(self, f):
        """Return Fourier transform of ``f`` as expected by this interpolator."""
        i = (0, -1) if (self.source_grid.num_zeta % 2 == 0) else 0
        return 2 * rfft2(
            self.source_grid.meshgrid_reshape(f, "rtz")[0],
            axes=(0, 1),
            norm="forward",
        ).at[:, i].divide(2).reshape(-1, *f.shape[1:])

    def vander_polar(self, i):
        """Return Vandermonde matrix for ith polar node."""
        return rfft2_vander(
            self.eval_grid.unique_theta + self._shift_t[i],
            self.eval_grid.unique_zeta + self._shift_z[i],
            self._modes_fft,
            self._modes_rfft,
            inverse_idx_fft=self.eval_grid.inverse_theta_idx,
            inverse_idx_rfft=self.eval_grid.inverse_zeta_idx,
        ).reshape(self.eval_grid.num_nodes, -1)

    def __call__(self, f, i, *, is_fourier=False, vander=None):
        """Interpolate ``f`` to polar node ``i`` around evaluation grid.

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


def _prune_data(eval_data, eval_grid, source_data, source_grid, kernel):
    """Returns new dictionaries with only required data."""
    keys = ["R", "phi", "Z", "theta", "zeta"]
    if hasattr(kernel, "eval_keys"):
        keys = keys + kernel.eval_keys

    eval_data = apply(eval_data, jnp.asarray, keys)
    if eval_grid is not None:
        # Casting to JAX arrays reduces memory usage.
        if "theta" not in eval_data:
            eval_data["theta"] = jnp.asarray(eval_grid.nodes[:, 1])
        if "zeta" not in eval_data:
            eval_data["zeta"] = jnp.asarray(eval_grid.nodes[:, 2])

    # Can't prune ω because ω is need to interpolate ϕ in _singular_part.
    keys = kernel.keys + ["omega", "theta", "zeta"]
    source_data = apply(source_data, jnp.asarray, keys)
    # to avoid adding keys to dictionary during iteration
    if "theta" not in source_data:
        source_data["theta"] = jnp.asarray(source_grid.nodes[:, 1])
    if "zeta" not in source_data:
        source_data["zeta"] = jnp.asarray(source_grid.nodes[:, 2])

    return eval_data, source_data


def _nonsingular_part(
    eval_data,
    eval_grid,
    source_data,
    source_grid,
    st,
    sz,
    kernel,
    *,
    ndim=None,
    chunk_size=None,
):
    """Integrate kernel over non-singular points.

    Generally follows sec 3.2.1 of [2].
    If ``eval_grid`` is ``None``, then takes η = 0.
    """
    assert source_grid.can_fft2
    ht = 2 * jnp.pi / source_grid.num_theta
    hz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP

    ndim = setdefault(ndim, kernel.ndim)

    source_zeta = source_data["zeta"]
    source_phi = source_data["phi"]

    def func(zeta_j):
        source_data["zeta"] = zeta_j
        source_data["phi"] = zeta_j  # TODO (#465)

        # nest this def and let JAX figure it out
        def eval_pt(eval_data_i):
            _eta = (
                0
                if eval_grid is None
                else eta(
                    source_data["theta"],
                    source_data["zeta"],
                    eval_data_i["theta"][:, jnp.newaxis],
                    eval_data_i["zeta"][:, jnp.newaxis],
                    ht,
                    hz,
                    st,
                    sz,
                )
            )
            # absorbing (1 - eta) into ds to reduce number of flops by factor of ndim
            return (
                kernel(eval_data_i, source_data, (ht * hz) * (1 - _eta))
                .reshape(-1, source_grid.num_nodes, ndim)
                .sum(-2)
            )

        return batch_map(eval_pt, eval_data, chunk_size).reshape(-1, ndim)

    f = nfp_loop(source_grid, func, jnp.zeros((eval_data["phi"].size, ndim)))
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    # undo rotation of ζ and ϕ
    source_data["zeta"] = source_zeta
    source_data["phi"] = source_phi

    return f


def _singular_part(
    eval_data, source_data, interpolator, kernel, *, known_map=None, chunk_size=None
):
    """Integrate singular point by interpolating to polar grid.

    Generally follows sec 3.2.2 of [2], with the following differences:

    - hyperparameter M replaced by ``st`` and ``sz``.
    - density sigma / function f is absorbed into kernel.

    TODO (#465): For nonzero ω, the quadrature may not be symmetric about the
       singular point. Hence the quadrature may not converge for Cauchy
       principal values. Prove otherwise or remove singularity.

    """
    eval_grid = interpolator.eval_grid
    eval_theta = eval_grid.unique_theta
    eval_zeta = eval_grid.unique_zeta

    r, w, dr, dw = _get_polar_quadrature(interpolator.q)
    r = jnp.abs(r)
    # integrand of eq 38 in [2] except stuff that needs to be interpolated
    v = interpolator.ht * interpolator.hz * interpolator.st * interpolator.sz / 4
    v = v * (chi(r) * r * dr * dw)

    keys = set(kernel.keys)
    if "phi" in keys:
        keys.remove("phi")  # ϕ is not a periodic map of θ, ζ.
        keys.add("omega")
    if known_map is not None:
        map_name, map_fun = known_map
        keys.remove(map_name)
    # It is necessary to take the Fourier transforms of the
    # vector components of the orthonormal polar basis vectors R̂, ϕ̂, Ẑ.
    # Vector components of the Cartesian basis are not NFP periodic.
    fsource = [
        (key, interpolator.fourier(source_data[key]))
        for key in keys
        if key in source_data
    ]

    def polar_pt(i):
        """See sec 3.2.2 of [2].

        Evaluate the effect from a single polar node around each eval point
        on that eval point. Polar grids from other singularities have no effect,
        so only the diagonal term of the kernel is needed.
        """
        vander = interpolator.vander_polar(i)
        source_data_polar = {
            key: interpolator(val, i, is_fourier=True, vander=vander)
            for key, val in fsource
        }
        # coordinates of the polar nodes around the evaluation point
        source_data_polar["theta"] = eval_data["theta"] + interpolator.shift_t[i]
        source_data_polar["zeta"] = eval_data["zeta"] + interpolator.shift_z[i]
        if "omega" in keys:
            source_data_polar["phi"] = (
                source_data_polar["zeta"] + source_data_polar["omega"]
            )
        if known_map is not None:
            source_data_polar[map_name] = map_fun(
                eval_grid,
                t=eval_theta + interpolator.shift_t[i],
                z=eval_zeta + interpolator.shift_z[i],
            )
        return kernel(eval_data, source_data_polar, v[i], diag=True)

    f = vmap_chunked(
        polar_pt,
        chunk_size=chunk_size,
        reduction=jnp.add,
        chunk_reduction=_add_reduce,
    )(jnp.arange(v.size)).reshape(eval_grid.num_nodes, -1)

    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def _add_reduce(x):
    return x.sum(0)


def singular_integral(
    eval_data,
    source_data,
    interpolator,
    kernel,
    *,
    known_map=None,
    ndim=None,
    chunk_size=None,
    **kwargs,
):
    """Evaluate a singular integral transform on a surface.

    eg f(θ, ζ) = ∫ ∫ K(θ, ζ, θ', ζ') g(θ', ζ') dθ' dζ'

    Where K(θ, ζ, θ', ζ') is the (singular) kernel and g(θ', ζ') is the metric on the
    surface. See eq. 3.7 in [1]_, but we have absorbed the density σ into K

    Uses method by Malhotra et al. [1]_ [2]_

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points (``interpolator.eval_grid``).
        Should store (R, ϕ, Z) coordinates to evaluate field and any keys
        in ``kernel.eval_keys``.
        Vector data should be in rpz basis.
    source_data : dict
        Dictionary of data at source points (``interpolatr.source_grid``). Keys
        should be those required by kernel as ``kernel.keys``.
        Vector data should be in rpz basis.
    interpolator : _BIESTInterpolator
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
    kernel : str or callable
        Kernel function to evaluate. If str, one of the following:
            '1_over_r'        : 1 / |𝐫 − 𝐫'| dS
            'nr_over_r3'      : 𝐧'⋅(𝐫 − 𝐫') / |𝐫 − 𝐫'|³ dS
            'biot_savart'     : μ₀/4π 𝐊'×(𝐫 − 𝐫') / |𝐫 − 𝐫'|³ dS
            'biot_savart_A'   : μ₀/4π 𝐊' / |𝐫 − 𝐫'| dS
        If callable, should take 4 arguments:
            eval_data   : dict of data at evaluation points (primed)
            source_data : dict of data at source points (unprimed)
            ds          : Surface area element (not weighted by ‖e_θ × e_ζ‖ Jacobian).
                          Broadcasts with shape
                          (eval_grid.num_nodes, source_grid.num_nodes).
            diag        : boolean, whether to evaluate full cross interactions
                          or just diagonal
        If a callable, should also have the attributes ``ndim`` and ``keys`` defined.
        ``ndim`` is an integer representing the dimensionality of the output function f,
        1 if f is scalar, 3 if f is a vector, etc.
        ``keys`` is a list of strings of what data is required to evaluate the kernel.
        The kernel will be called with dictionaries containing this data at source and
        evaluation points.
        If vector valued, the input to the kernel function will be in rpz and output
        should be in xyz.
    known_map : (str, callable)
        Optional. If map used in kernel of singular integral is known,
        then it is more efficient to provide a callable to compute it
        rather than interpolating and evaluating a Fourier series.
        First index should store the name of the map used in the kernel
        e.g. "Phi (periodic)", and the second index should store the Python
        callable that accepts a grid argument.
        Should broadcast with shapes (..., source_grid.num_nodes, ndim).
    ndim : int
        Default is kernel.ndim.
        In some applications it, may be useful to supply other values for batching.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

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
    chunk_size = parse_argname_change(chunk_size, kwargs, "loop", "chunk_size")
    chunk_size = None if (chunk_size == 0) else chunk_size

    if isinstance(kernel, str):
        kernel = kernels[kernel]

    eval_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid
    if kwargs.get("_prune_data", True):
        eval_data, source_data = _prune_data(
            eval_data,
            eval_grid,
            source_data,
            source_grid,
            kernel,
        )
    out1 = _singular_part(
        eval_data,
        source_data,
        interpolator,
        kernel,
        known_map=known_map,
        chunk_size=chunk_size,
    )
    out2 = _nonsingular_part(
        eval_data,
        eval_grid,
        source_data,
        source_grid,
        interpolator.st,
        interpolator.sz,
        kernel,
        ndim=ndim,
        chunk_size=chunk_size,
    )
    return out1 + out2


def _dx(eval_data, source_data, diag=False):
    """Compute distance vector between eval and source points.

    Parameters
    ----------
    eval_data : dict[str, jnp.ndarray]
        x data evaluated on eval grid.
    source_data : dict[str, jnp.ndarray]
        y data evaluated on source grid.
    diag : bool
        Set to True to bypass outer product.

    Returns
    -------
    dx : jnp.ndarray
        The vector x-y where y is a source point and x is eval point,
        in Cartesian coordinates.
        Shape (num eval, num source, 3).

    """
    source_x = rpz2xyz(
        jnp.column_stack([source_data["R"], source_data["phi"], source_data["Z"]])
    )
    eval_x = rpz2xyz(
        jnp.column_stack([eval_data["R"], eval_data["phi"], eval_data["Z"]])
    )
    if not diag:
        eval_x = eval_x[:, jnp.newaxis]
    return eval_x - source_x


_dx.keys = ["R", "phi", "Z"]


def _G(dx, keepdims=False):
    """Fundamental solution to the Laplacian in ℝ³.

    Parameters
    ----------
    dx : jnp.ndarray
        The vector x-y where y is a source point and x is eval point,
        in Cartesian coordinates.
        Shape (num eval, num source, 3).

    Returns
    -------
    G : jnp.ndarray
        G(x-y) = -1/(4π ‖x−y‖).
        Shape (num eval, num source).

    """
    return safediv(-1, 4 * jnp.pi * safenorm(dx, axis=-1, keepdims=keepdims))


def _grad_G(dx):
    """∇_x G(x−y) where G is the fundamental solution to the Laplacian in ℝ³.

    Parameters
    ----------
    dx : jnp.ndarray
        The vector x-y where y is a source point and x is eval point,
        in Cartesian coordinates.
        Shape (num eval, num source, 3).

    Returns
    -------
    grad_G : jnp.ndarray
        ∇_x G(x−y) = (4π)⁻¹ ‖x−y‖⁻³ (x-y).
        Shape (num eval, num source).

    """
    return safediv(dx, 4 * jnp.pi * safenorm(dx, axis=-1, keepdims=True) ** 3)


def _kernel_1_over_r(eval_data, source_data, ds, diag=False):
    """Returns -4π da(y) G(x-y) = ‖e_θ × e_ζ‖ dθ dζ ‖x−y‖⁻¹."""
    return (
        (-4 * jnp.pi * ds)
        * source_data["|e_theta x e_zeta|"]
        * _G(_dx(eval_data, source_data, diag))
    )


_kernel_1_over_r.ndim = 1
_kernel_1_over_r.keys = _dx.keys + ["|e_theta x e_zeta|"]


def _kernel_nr_over_r3(eval_data, source_data, ds, diag=False):
    """Returns 4π ds(y) ⋅ ∇_x G(x−y) = ds(y) ⋅ ‖x−y‖⁻³ (x-y)."""
    return (4 * jnp.pi * ds) * dot(
        rpz2xyz_vec(source_data["e_theta x e_zeta"], phi=source_data["phi"]),
        _grad_G(_dx(eval_data, source_data, diag)),
    )


_kernel_nr_over_r3.ndim = 1
_kernel_nr_over_r3.keys = _dx.keys + ["e_theta x e_zeta"]


def _kernel_biot_savart(eval_data, source_data, ds, diag=False):
    """Returns (μ₀ K(y) x ∇_x G(x−y)) da(y) = (μ₀/4π) K(y) da(y) × (x-y) ‖x−y‖⁻³."""
    if jnp.ndim(ds) > 0:
        ds = ds[..., jnp.newaxis]
    K = rpz2xyz_vec(source_data["K_vc"], phi=source_data["phi"])
    return ds * jnp.cross(
        mu_0 * K * source_data["|e_theta x e_zeta|"][:, jnp.newaxis],
        _grad_G(_dx(eval_data, source_data, diag)),
    )


_kernel_biot_savart.ndim = 3
_kernel_biot_savart.keys = _dx.keys + ["K_vc", "|e_theta x e_zeta|"]


def _kernel_biot_savart_A(eval_data, source_data, ds, diag=False):
    """Returns ds(y) (-μ₀K)(y) G(x−y) = (μ₀/4π) ds(y) K(y) ‖x−y‖⁻¹."""
    if jnp.ndim(ds) > 0:
        ds = ds[..., jnp.newaxis]
    K = rpz2xyz_vec(source_data["K_vc"], phi=source_data["phi"])
    return (
        ds
        * source_data["|e_theta x e_zeta|"][:, jnp.newaxis]
        * (-mu_0 * K)
        * _G(_dx(eval_data, source_data, diag), keepdims=True)
    )


_kernel_biot_savart_A.ndim = 3
_kernel_biot_savart_A.keys = _dx.keys + ["K_vc", "|e_theta x e_zeta|"]


def _kernel_BS_plus_grad_S(eval_data, source_data, ds, diag=False):
    """Returns K(y) (Tesla) x ∇_x G(x−y) da(y) + ∇_x G(x−y) Bₙ(y) da(y)."""
    if jnp.ndim(ds) > 0:
        ds = ds[..., jnp.newaxis]
    K = rpz2xyz_vec(source_data["K_vc (periodic)"], phi=source_data["phi"])
    a = source_data["|e_theta x e_zeta|"]
    grad_G = _grad_G(_dx(eval_data, source_data, diag))
    return ds * (
        jnp.cross(K * a[:, jnp.newaxis], grad_G)
        + grad_G * (source_data["B0*n"] * a)[:, jnp.newaxis]
    )


_kernel_BS_plus_grad_S.ndim = 3
_kernel_BS_plus_grad_S.keys = _dx.keys + [
    "K_vc (periodic)",
    "B0*n",
    "|e_theta x e_zeta|",
]


def _kernel_monopole(eval_data, source_data, ds, diag=False):
    """Kernel of single layer operator S[B0*n]: (B0*n)(y) G(x-y) da(y)."""
    return (
        ds
        * (source_data["|e_theta x e_zeta|"] * source_data["B0*n"])
        * _G(_dx(eval_data, source_data, diag))
    )


_kernel_monopole.ndim = 1
_kernel_monopole.keys = _dx.keys + ["B0*n", "|e_theta x e_zeta|"]


def _kernel_dipole(eval_data, source_data, ds, diag=False):
    """Kernel of double layer operator D[Φ]: Φ(y)〈∇_x G(x−y),n(y)〉da(y)."""
    out = ds * dot(
        rpz2xyz_vec(source_data["e_theta x e_zeta"], phi=source_data["phi"]),
        _grad_G(_dx(eval_data, source_data, diag)),
    )
    if source_data["Phi (periodic)"].ndim > 1:
        out = out[..., jnp.newaxis]
    # Do operation with Φ at the end, so that the following
    # outer product plus reduction is more likely to be fused.
    return source_data["Phi (periodic)"] * out


_kernel_dipole.ndim = 1
_kernel_dipole.keys = _dx.keys + ["e_theta x e_zeta", "Phi (periodic)"]


def _kernel_dipole_plus_half(eval_data, source_data, ds, diag=False):
    """Kernel of operator (D[Φ] + Φ/2)(x)."""
    eval_Phi = eval_data["Phi(x) (periodic)"]
    if not diag:
        eval_Phi = eval_Phi[:, jnp.newaxis]
    out = ds * dot(
        rpz2xyz_vec(source_data["e_theta x e_zeta"], phi=source_data["phi"]),
        _grad_G(_dx(eval_data, source_data, diag)),
    )
    if source_data["Phi (periodic)"].ndim > 1:
        out = out[..., jnp.newaxis]
    # Do operation with Φ at the end, so that the following
    # outer product plus reduction is more likely to be fused.
    return (source_data["Phi (periodic)"] - eval_Phi) * out


_kernel_dipole_plus_half.ndim = 1
_kernel_dipole_plus_half.keys = _dx.keys + ["e_theta x e_zeta", "Phi (periodic)"]
_kernel_dipole_plus_half.eval_keys = ["Phi(x) (periodic)"]

kernels = {
    "1_over_r": _kernel_1_over_r,
    "nr_over_r3": _kernel_nr_over_r3,
    "biot_savart": _kernel_biot_savart,
    "biot_savart_A": _kernel_biot_savart_A,
    "biot_savart_grad_S": _kernel_BS_plus_grad_S,
    "monopole": _kernel_monopole,
    "dipole": _kernel_dipole,
    "dipole_plus_half": _kernel_dipole_plus_half,
}


@partial(jit, static_argnames=["chunk_size", "loop"])
def virtual_casing_biot_savart(
    eval_data, source_data, interpolator, chunk_size=None, **kwargs
):
    """Evaluate magnetic field on surface due to sheet current on surface.

    The magnetic field due to the plasma current can be written as a Biot-Savart
    integral over the plasma volume:

    𝐁ᵥ(𝐫) = μ₀/4π ∫ 𝐉(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d³𝐫'

    Where 𝐉 is the plasma current density, 𝐫 is a point on the plasma surface, and 𝐫' is
    a point in the plasma volume.

    This 3D integral can be converted to a 2D integral over the plasma boundary using
    the virtual casing principle [1]_

    𝐁ᵥ(𝐫) = μ₀/4π ∫ (𝐧' ⋅ 𝐁(𝐫')) * (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + μ₀/4π ∫ (𝐧' × 𝐁(𝐫')) × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + 𝐁(𝐫)/2

    Where 𝐁 is the total field on the surface and 𝐧' is the outward surface normal.
    Because the total field is tangent, the first term in the integrand is zero leaving

    𝐁ᵥ(𝐫) = μ₀/4π ∫ K_vc(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫' + 𝐁(𝐫)/2

    Where we have defined the virtual casing sheet current K_vc = 𝐧' × 𝐁(𝐫')

    References
    ----------
       [1] Hanson, James D. "The virtual-casing principle and Helmholtz’s theorem."
       Plasma Physics and Controlled Fusion 57.11 (2015): 115006.

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points (``interpolator.eval_grid``).
        Should store (R, ϕ, Z) coordinates to evaluate field and any keys
        in ``kernel.eval_keys``.
        Vector data should be in rpz basis.
    source_data : dict
        Dictionary of data at source points (``interpolatr.source_grid``). Keys
        should be those required by kernel as ``kernel.keys``.
        Vector data should be in rpz basis.
    interpolator : _BIESTInterpolator
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
    chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, kernel.ndim)
        Integral transform evaluated at eval_grid. Vectors are in rpz basis.

    """
    return singular_integral(
        eval_data,
        source_data,
        interpolator,
        _kernel_biot_savart,
        chunk_size=chunk_size,
        **kwargs,
    )


def compute_B_plasma(
    eq, eval_grid, source_grid=None, normal_only=False, chunk_size=None
):
    """Evaluate magnetic field on surface due to enclosed plasma currents.

    The magnetic field due to the plasma current can be written as a Biot-Savart
    integral over the plasma volume:

    𝐁ᵥ(𝐫) = μ₀/4π ∫ 𝐉(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d³𝐫'

    Where 𝐉 is the plasma current density, 𝐫 is a point on the plasma surface, and 𝐫' is
    a point in the plasma volume.

    This 3D integral can be converted to a 2D integral over the plasma boundary using
    the virtual casing principle [1]_

    𝐁ᵥ(𝐫) = μ₀/4π ∫ (𝐧' ⋅ 𝐁(𝐫')) * (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + μ₀/4π ∫ (𝐧' × 𝐁(𝐫')) × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + 𝐁(𝐫)/2

    Where 𝐁 is the total field on the surface and 𝐧' is the outward surface normal.
    Because the total field is tangent, the first term in the integrand is zero leaving

    𝐁ᵥ(𝐫) = μ₀/4π ∫ K_vc(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫' + 𝐁(𝐫)/2

    Where we have defined the virtual casing sheet current K_vc = 𝐧' × 𝐁(𝐫')

    References
    ----------
       [1] Hanson, James D. "The virtual-casing principle and Helmholtz’s theorem."
       Plasma Physics and Controlled Fusion 57.11 (2015): 115006.

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
    chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, 3) or shape(eval_grid.num_nodes,)
        Magnetic field evaluated at eval_grid.
        If normal_only=False, vector B is in rpz basis.

    """
    if source_grid is None:
        source_grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )

    eval_data = eq.compute(_dx.keys + ["B", "n_rho"], grid=eval_grid)
    source_data = eq.compute(
        _kernel_biot_savart.keys + ["|e_theta x e_zeta|"], grid=source_grid
    )
    if hasattr(eq.surface, "Phi_mn"):
        source_data = eq.surface.compute("K", grid=source_grid, data=source_data)
        source_data["K_vc"] += source_data["K"]

    interpolator = get_interpolator(eval_grid, source_grid, source_data)
    Bplasma = virtual_casing_biot_savart(
        eval_data, source_data, interpolator, chunk_size=chunk_size
    )
    # need extra factor of B/2 bc we're evaluating on plasma surface
    Bplasma += eval_data["B"] / 2
    if normal_only:
        Bplasma = dot(Bplasma, eval_data["n_rho"])
    return Bplasma
