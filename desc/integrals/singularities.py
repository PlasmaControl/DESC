"""High order method for singular surface integrals, from Malhotra 2019."""

from abc import ABC, abstractmethod
from functools import partial

from scipy.constants import mu_0

from desc.backend import jit, jnp, rfft2
from desc.batching import batch_map, vmap_chunked
from desc.compute.geom_utils import rpz2xyz, rpz2xyz_vec, xyz2rpz_vec
from desc.grid import LinearGrid
from desc.integrals._interp_utils import rfft2_modes, rfft2_vander
from desc.integrals.quad_utils import _get_polar_quadrature, chi, eta, nfp_loop
from desc.io import IOAble
from desc.utils import (
    check_posint,
    dot,
    parse_argname_change,
    safediv,
    safenorm,
    setdefault,
    warnif,
)

from ._interpax_mod import fft_interp2d


def _vanilla_params(grid):
    """Parameters for support size and quadrature resolution.

    These parameters do not account for grid anisotropy.

    Parameters
    ----------
    grid : LinearGrid
        Grid that can fft2.

    Returns
    -------
    st : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    q : int
        Order of quadrature in radial and azimuthal directions.

    """
    Nt = grid.num_theta
    Nz = grid.num_zeta * grid.NFP
    q = int(jnp.sqrt(grid.num_nodes) / 2)
    s = min(q, Nt, Nz)
    return s, s, q


def best_params(grid, ratio):
    """Parameters for heuristic support size and quadrature resolution.

    These parameters account for global grid anisotropy which ensures
    more robust convergence across a wider aspect ratio range.

    Parameters
    ----------
    grid : LinearGrid
        Grid that can fft2.
    ratio : float or jnp.ndarray
        Best ratio.

    Returns
    -------
    st : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    q : int
        Order of quadrature in radial and azimuthal directions.

    """
    assert grid.can_fft2
    Nt = grid.num_theta
    Nz = grid.num_zeta * grid.NFP
    if grid.num_zeta > 1:  # actually has toroidal resolution
        q = int(jnp.sqrt(grid.num_nodes) / 2)
    else:  # axisymmetry
        q = int(jnp.sqrt(Nt * Nz) / 2)
    s = min(q, Nt, Nz)
    # Size of singular region in real space = s * h * |e_.|
    # For it to be a circle, choose radius ~ equal
    # s_t * h_t * |e_t| = s_z * h_z * |e_z|
    # s_z / s_t = h_t / h_z  |e_t| / |e_z| = Nz*NFP/Nt |e_t| / |e_z|
    # Denote ratio = < |e_z| / |e_t| > and
    #      s_ratio = s_z / s_t = Nz*NFP/Nt / ratio
    # Also want sqrt(s_z*s_t) ~ s = q.
    s_ratio = jnp.sqrt(Nz / Nt / ratio)
    st = jnp.clip(jnp.ceil(s / s_ratio).astype(int), None, Nt)
    sz = jnp.clip(jnp.ceil(s * s_ratio).astype(int), None, Nz)
    if s_ratio.size == 1:
        st = int(st)
        sz = int(sz)
    return st, sz, q


def best_ratio(data, local=False):
    """Ratio to make singular integration partition ~circle in real space.

    Parameters
    ----------
    data : dict[str, jnp.ndarray]
        Dictionary of data evaluated on single flux surface grid that ``can_fft2``
        with keys ``|e_theta x e_zeta|``, ``e_theta``, and ``e_zeta``.
    local : bool
        Whether to average with local aspect ratio.

    """
    scale = jnp.linalg.norm(data["e_zeta"], axis=-1) / jnp.linalg.norm(
        data["e_theta"], axis=-1
    )
    mean = jnp.mean(scale * data["|e_theta x e_zeta|"]) / jnp.mean(
        data["|e_theta x e_zeta|"]
    )
    return (0.5 * (mean + scale)) if local else mean


def get_interpolator(
    eval_grid,
    source_grid,
    src_data,
    use_dft=False,
    *,
    st=None,
    sz=None,
    q=None,
    warn_dft=True,
    warn_fft=True,
    **kwargs,
):
    """Get interpolator from Cartesian to polar domain.

    Parameters
    ----------
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
    src_data : dict[str, jnp.ndarray]
        Dictionary of data evaluated on single flux surface grid that ``can_fft2``
        with keys ``|e_theta x e_zeta|``, ``e_theta``, and ``e_zeta``.
    use_dft : bool
        Whether to use matrix multiplication transform from spectral to physical domain
        instead of inverse fast Fourier transform.
    warn_dft : bool
        Set to ``False`` to turn off warnings about using DFT.
    warn_fft : bool
        Set to ``False`` to turn off warnings about FFT frequency truncation.

    Returns
    -------
    f : _BIESTInterpolator
        Interpolator that uses the specified method.

    """
    if st is None or sz is None or q is None:
        _st, _sz, _q = best_params(source_grid, best_ratio(src_data))
        st = setdefault(st, _st)
        sz = setdefault(sz, _sz)
        q = setdefault(q, _q)
    if use_dft:
        f = DFTInterpolator(eval_grid, source_grid, st, sz, q)
    else:
        try:
            f = FFTInterpolator(eval_grid, source_grid, st, sz, q, warn_fft=warn_fft)
        except AssertionError as e:
            warnif(
                warn_dft,
                msg="Could not build fft interpolator because:\n"
                + str(e)
                + "\nThe DFT interpolator is much less performant."
                "\nIn some cases when the real domain grid is sparser than the "
                "spectral domain grid because the DFT interpolator may be useful "
                "as it is exact while FFT truncates higher frequencies.",
            )
            f = DFTInterpolator(eval_grid, source_grid, st, sz, q)
            use_dft = True
    # TODO (#1599).
    warnif(
        use_dft and warn_dft,
        RuntimeWarning,
        msg="Matrix multiplication may be performed incorrectly for large matrices. "
        "Until this is fixed, it is recommended to validate results small chunk size "
        "when using the DFT interpolator.",
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
    """FFT interpolation operator required for high order singular integration.

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

    """

    def __init__(self, eval_grid, source_grid, st, sz, q, **kwargs):
        st = parse_argname_change(st, kwargs, "s", "st")
        assert eval_grid.can_fft2, "Got False for eval_grid.can_fft2."
        warn = kwargs.get("warn_fft", True)
        warnif(
            warn and eval_grid.num_theta < source_grid.num_theta,
            msg="Frequency spectrum of FFT interpolation will be truncated because "
            "the evaluation grid has less resolution than the source grid.\n"
            f"Got eval_grid.num_theta = {eval_grid.num_theta} < "
            f"{source_grid.num_theta} = source_grid.num_theta.",
        )
        warnif(
            warn and eval_grid.num_zeta < source_grid.num_zeta,
            msg="Frequency spectrum of FFT interpolation will be truncated because "
            "the evaluation grid has less resolution than the source grid.\n"
            f"Got eval_grid.num_zeta = {eval_grid.num_zeta} < "
            f"{source_grid.num_zeta} = source_grid.num_zeta.",
        )
        super().__init__(eval_grid, source_grid, st, sz, q)

    def fourier(self, f):
        """Return Fourier transform of ``f`` as expected by this interpolator."""
        # TODO (#1206)
        return jnp.fft.ifft2(
            self._source_grid.meshgrid_reshape(f, "rtz")[0], axes=(0, 1)
        )

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
        return fft_interp2d(
            f,
            n1=self._eval_grid.num_theta,
            n2=self._eval_grid.num_zeta,
            sx=self._shift_t[i],
            sy=self._shift_z[i],
            dx=self._ht,
            dy=self._hz,
            is_fourier=True,
        ).reshape(self._eval_grid.num_nodes, *f.shape[2:], order="F")


class DFTInterpolator(_BIESTInterpolator):
    """Fourier interpolation matrix required for high order singular integration.

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
        if (self._source_grid.num_zeta % 2) == 0:
            i = (0, -1)
        else:
            i = 0
        return 2 * rfft2(
            self._source_grid.meshgrid_reshape(f, "rtz")[0],
            axes=(0, 1),
            norm="forward",
        ).at[:, i].divide(2).reshape(-1, *f.shape[1:])

    def vander_polar(self, i):
        """Return Vandermonde matrix for ith polar node."""
        return rfft2_vander(
            self._eval_grid.unique_theta + self._shift_t[i],
            self._eval_grid.unique_zeta + self._shift_z[i],
            self._modes_fft,
            self._modes_rfft,
            inverse_idx_fft=self._eval_grid.inverse_theta_idx,
            inverse_idx_rfft=self._eval_grid.inverse_zeta_idx,
        ).reshape(self._eval_grid.num_nodes, -1)

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


def _nonsingular_part(
    eval_data,
    eval_grid,
    source_data,
    source_grid,
    st,
    sz,
    kernel,
    chunk_size=None,
    _eta=eta,
):
    """Integrate kernel over non-singular points.

    Generally follows sec 3.2.1 of [2].
    """
    assert source_grid.can_fft2
    source_data.setdefault("theta", source_grid.nodes[:, 1])
    # make sure source dict has zeta and phi to avoid
    # adding keys to dict during iteration
    source_zeta = source_data.setdefault("zeta", source_grid.nodes[:, 2])
    source_phi = source_data["phi"]

    # slim down to skip batching quantities that aren't used
    eval_data = {key: eval_data[key] for key in kernel.keys if key in eval_data}
    eval_data["theta"] = jnp.asarray(eval_grid.nodes[:, 1])
    eval_data["zeta"] = jnp.asarray(eval_grid.nodes[:, 2])

    ht = 2 * jnp.pi / source_grid.num_theta
    hz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
    w = source_data["|e_theta x e_zeta|"][jnp.newaxis] * ht * hz

    def func(zeta_j):
        source_data["zeta"] = zeta_j
        source_data["phi"] = zeta_j  # TODO (#465)

        # nest this def and let JAX figure it out
        def eval_pt(eval_data_i):
            k = kernel(eval_data_i, source_data).reshape(
                -1, source_grid.num_nodes, kernel.ndim
            )
            e = 1 - _eta(
                source_data["theta"],
                source_data["zeta"],
                eval_data_i["theta"][:, jnp.newaxis],
                eval_data_i["zeta"][:, jnp.newaxis],
                ht,
                hz,
                st,
                sz,
            )
            return jnp.sum(k * (e * w)[..., jnp.newaxis], axis=-2)

        return batch_map(eval_pt, eval_data, chunk_size).reshape(
            eval_grid.num_nodes, kernel.ndim
        )

    f = nfp_loop(source_grid, func, jnp.zeros((eval_grid.num_nodes, kernel.ndim)))

    # undo rotation of source_zeta
    source_data["zeta"] = source_zeta
    source_data["phi"] = source_phi
    # we sum vectors at different points, so they need to be in xyz for that to work
    # but then need to convert vectors back to rpz
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def _singular_part(
    eval_data, source_data, interpolator, *, kernel, known_map=None, chunk_size=None
):
    """Integrate singular point by interpolating to polar grid.

    Generally follows sec 3.2.2 of [2], with the following differences:

    - hyperparameter M replaced by ``st`` and ``sz``.
    - density sigma / function f is absorbed into kernel.
    """
    eval_grid = interpolator._eval_grid
    eval_theta = eval_grid.unique_theta
    eval_zeta = eval_grid.unique_zeta

    r, w, dr, dw = _get_polar_quadrature(interpolator.q)
    r = jnp.abs(r)
    # integrand of eq 38 in [2] except stuff that needs to be interpolated
    v = (
        chi(r)
        * (interpolator.ht * interpolator.hz)
        * (interpolator.st * interpolator.sz / 4)
        * r
        * dr
        * dw
    )

    keys = set(["|e_theta x e_zeta|"] + kernel.keys)
    if "phi" in keys:
        keys.remove("phi")  # ϕ is not a periodic map of θ, ζ.
        keys.add("omega")
    if known_map is not None:
        map_name, map_fun = known_map
        keys.remove(map_name)
    # Note that it is necessary to take the Fourier transforms of the
    # vector components of the orthonormal polar basis vectors R̂, ϕ̂, Ẑ.
    # Vector components of the Cartesian basis are not NFP periodic.
    fsource = [(key, interpolator.fourier(source_data[key])) for key in keys]

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
        # Coordinates of the polar nodes around the evaluation point.
        t = eval_theta + interpolator.shift_t[i]
        z = eval_zeta + interpolator.shift_z[i]
        if known_map is not None:
            source_data_polar[map_name] = map_fun((t, z, eval_grid))
        source_data_polar["theta"] = t[eval_grid.inverse_theta_idx]
        source_data_polar["zeta"] = z[eval_grid.inverse_zeta_idx]
        if "omega" in keys:
            source_data_polar["phi"] = (
                source_data_polar["zeta"] + source_data_polar["omega"]
            )
            # TODO (#465): For nonzero ω, the quadrature may not be symmetric about the
            #  singular point for hypersingular kernels such as the Biot-Savart kernel.
            #  Hence the quadrature may not converge to the Hadamard finite part.
            #  Prove otherwise or use uniform grid in θ, ϕ and map coordinates.

        k = kernel(eval_data, source_data_polar, diag=True).reshape(
            eval_grid.num_nodes, kernel.ndim
        )
        dS = v[i] * source_data_polar["|e_theta x e_zeta|"]
        fi = k * dS[:, jnp.newaxis]
        return fi

    f = vmap_chunked(
        polar_pt,
        chunk_size=chunk_size,
        reduction=jnp.add,
        chunk_reduction=_add_reduce,
    )(jnp.arange(v.size))
    assert f.shape == (eval_grid.num_nodes, kernel.ndim)

    # we sum vectors at different points, so they need to be in xyz for that to work
    # but then need to convert vectors back to rpz
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def _add_reduce(x):
    # https://github.com/jax-ml/jax/issues/23493
    return x.sum(axis=0)


def singular_integral(
    eval_data,
    source_data,
    interpolator,
    *,
    kernel,
    known_map=None,
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
        Dictionary of data at evaluation points (eval_grid passed to interpolator).
        Keys should be those required by kernel as kernel.keys. Vector data should be
        in rpz basis.
    source_data : dict
        Dictionary of data at source points (source_grid passed to interpolator). Keys
        should be those required by kernel as kernel.keys. Vector data should be in
        rpz basis.
    interpolator : _BIESTInterpolator
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
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
    known_map : (str, callable)
        Optional. If map used in kernel of singular integral is known,
        then may provide a callable to compute to avoid inefficient
        interpolation and function approximation.
        First index should store the name of the map used in the kernel
        e.g. "Phi", and the second index should store the Python callable
        that accepts a grid argument.
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
    if chunk_size == 0:
        chunk_size = None
    # sanitize inputs, we need everything as jax arrays so they can be indexed
    # properly in the loops
    source_data = {key: jnp.asarray(val) for key, val in source_data.items()}
    eval_data = {key: jnp.asarray(val) for key, val in eval_data.items()}

    if isinstance(kernel, str):
        kernel = kernels[kernel]

    return _nonsingular_part(
        eval_data,
        interpolator._eval_grid,
        source_data,
        interpolator._source_grid,
        interpolator.st,
        interpolator.sz,
        kernel,
        chunk_size,
    ) + _singular_part(
        eval_data,
        source_data,
        interpolator,
        kernel=kernel,
        known_map=known_map,
        chunk_size=chunk_size,
    )


def _dx(eval_data, source_data, diag=False):
    """Returns dx = x−x'."""
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


def _kernel_1_over_r(eval_data, source_data, diag=False):
    """Returns G(x,x') = |x−x'|⁻¹."""
    dx = _dx(eval_data, source_data, diag)
    return safediv(1, safenorm(dx, axis=-1))


_kernel_1_over_r.ndim = 1
_kernel_1_over_r.keys = _dx.keys


def _kernel_nr_over_r3(eval_data, source_data, diag=False):
    """Returns n ⋅ −∇G(x,x') = n ⋅ (x−x')|x−x'|⁻³."""
    dx = _dx(eval_data, source_data, diag)
    # Need to use instead of n_rho to pass Green's ID test;
    # Fourier spectrum is much more concentrated for some reason.
    n = (
        source_data["e_theta x e_zeta"]
        / source_data["|e_theta x e_zeta|"][:, jnp.newaxis]
    )
    n = rpz2xyz_vec(n, phi=source_data["phi"])
    return safediv(dot(n, dx), safenorm(dx, axis=-1) ** 3)


_kernel_nr_over_r3.ndim = 1
_kernel_nr_over_r3.keys = _dx.keys + ["e_theta x e_zeta", "|e_theta x e_zeta|"]


def _kernel_biot_savart(eval_data, source_data, diag=False):
    """Returns (μ₀/4π) K × −∇G(x,x') = (μ₀/4π) K × (x−x')|x−x'|⁻³."""
    dx = _dx(eval_data, source_data, diag)
    K = rpz2xyz_vec(source_data["K_vc"], phi=source_data["phi"])
    return safediv(
        mu_0 / (4 * jnp.pi) * jnp.cross(K, dx),
        safenorm(dx, axis=-1, keepdims=True) ** 3,
    )


_kernel_biot_savart.ndim = 3
_kernel_biot_savart.keys = _dx.keys + ["K_vc"]


def _kernel_biot_savart_A(eval_data, source_data, diag=False):
    """Returns (μ₀/4π) K G(x,x') = (μ₀/4π) K |x−x'|⁻¹."""
    dx = _dx(eval_data, source_data, diag)
    K = rpz2xyz_vec(source_data["K_vc"], phi=source_data["phi"])
    return safediv(
        mu_0 / (4 * jnp.pi) * K,
        safenorm(dx, axis=-1, keepdims=True),
    )


_kernel_biot_savart_A.ndim = 3
_kernel_biot_savart_A.keys = _dx.keys + ["K_vc"]


def _kernel_Bn_over_r(eval_data, source_data, diag=False):
    """Returns Bₙ G(x,x') = Bₙ |x−x'|⁻¹."""
    dx = _dx(eval_data, source_data, diag)
    return safediv(source_data["Bn"], safenorm(dx, axis=-1))


_kernel_Bn_over_r.ndim = 1
_kernel_Bn_over_r.keys = _dx.keys + ["Bn"]


def _kernel_Phi_dGp_dn(eval_data, source_data, diag=False):
    """Returns Φ n ⋅ −∇G(x,x') = Φ n ⋅ (x−x')|x−x'|⁻³. Phi has units Tesla-meters."""
    dx = _dx(eval_data, source_data, diag)
    # Using n_rho works better than normalized e^rho*sqrt(g) here.
    n = rpz2xyz_vec(source_data["n_rho"], phi=source_data["phi"])
    if diag:
        numerator = source_data["Phi"] * dot(n, dx)
    else:
        numerator = dot(source_data["Phi"][..., jnp.newaxis] * n, dx)
    return safediv(numerator, safenorm(dx, axis=-1) ** 3)


_kernel_Phi_dGp_dn.ndim = 1
_kernel_Phi_dGp_dn.keys = _dx.keys + ["n_rho", "Phi"]


def _kernel_biot_savart_coulomb(eval_data, source_data, diag=False):
    """Returns [ K (Tesla) × −∇G(x,x') - Bₙ ∇G(x,x') ] / 4π."""
    dx = _dx(eval_data, source_data, diag)
    K = rpz2xyz_vec(source_data["K_vc"], phi=source_data["phi"])
    numerator = jnp.cross(K, dx) + source_data["Bn"][:, jnp.newaxis] * dx
    return safediv(
        numerator / (4 * jnp.pi),
        safenorm(dx, axis=-1, keepdims=True) ** 3,
    )


_kernel_biot_savart_coulomb.ndim = 3
_kernel_biot_savart_coulomb.keys = _dx.keys + ["K_vc", "Bn"]


kernels = {
    "1_over_r": _kernel_1_over_r,
    "nr_over_r3": _kernel_nr_over_r3,
    "biot_savart": _kernel_biot_savart,
    "biot_savart_A": _kernel_biot_savart_A,
    "Bn_over_r": _kernel_Bn_over_r,
    "Phi_dGp_dn": _kernel_Phi_dGp_dn,
    "biot_savart_coulomb": _kernel_biot_savart_coulomb,
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

    References
    ----------
    .. [1] Hanson, James D. "The virtual-casing principle and Helmholtz’s theorem."
       Plasma Physics and Controlled Fusion 57.11 (2015): 115006.

    """
    return singular_integral(
        eval_data,
        source_data,
        interpolator=interpolator,
        kernel=_kernel_biot_savart,
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
    chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

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
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )

    eval_data = eq.compute(_dx.keys + ["B", "n_rho"], grid=eval_grid)
    source_data = eq.compute(
        _kernel_biot_savart_A.keys + ["|e_theta x e_zeta|"], grid=source_grid
    )
    if hasattr(eq.surface, "Phi_mn"):
        source_data = eq.surface.compute("K", grid=source_grid, data=source_data)
        source_data["K_vc"] += source_data["K"]

    interpolator = get_interpolator(eval_grid, source_grid, source_data)
    Bplasma = virtual_casing_biot_savart(
        eval_data, source_data, interpolator, chunk_size
    )
    # need extra factor of B/2 bc we're evaluating on plasma surface
    Bplasma += eval_data["B"] / 2
    if normal_only:
        Bplasma = dot(Bplasma, eval_data["n_rho"])
    return Bplasma
