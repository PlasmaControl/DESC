"""High order method for singular surface integrals, from Malhotra 2019."""

from abc import ABC, abstractmethod

import numpy as np
import scipy
from interpax import fft_interp2d
from scipy.constants import mu_0

from desc.backend import fori_loop, jnp, rfft2
from desc.batching import batch_map, vmap_chunked
from desc.grid import LinearGrid
from desc.integrals._interp_utils import rfft2_modes, rfft2_vander
from desc.io import IOAble
from desc.utils import (
    check_posint,
    errorif,
    parse_argname_change,
    rpz2xyz,
    rpz2xyz_vec,
    safediv,
    safenorm,
    warnif,
    xyz2rpz_vec,
)


def _chi(r):
    """Partition of unity function in polar coordinates. Eq 39 in [2].

    Parameters
    ----------
    r : jnp.ndarray
        Absolute value of radial coordinate in polar domain.

    """
    return jnp.exp(-36 * jnp.abs(r) ** 8)


def _eta(theta, zeta, theta0, zeta0, ht, hz, st, sz):
    """Partition of unity function in rectangular coordinates.

    Consider the mapping from
    (θ,ζ) ∈ [-π, π) × [-π/NFP, π/NFP) to (ρ,ω) ∈ [−1, 1] × [0, 2π)
    defined by
    θ − θ₀ = h₁ s₁/2 ρ sin ω
    ζ − ζ₀ = h₂ s₂/2 ρ cos ω
    with Jacobian determinant norm h₁h₂ s₁s₂/4 |ρ|.

    In general in dimensions higher than one, the mapping that determines a
    change of variable for integration must be bijective. This is satisfied
    only if s₁ = 2π/h₁ and s₂ = (2π/NFP)/h₂. In the particular case the
    integrand is nonzero in a subset of the domain, then the change of variable
    need only be a bijective map where the function does not vanish, more
    precisely, its set of compact support.

    The functions we integrate are proportional to η₀(θ,ζ) = χ₀(r) far from the
    singularity at (θ₀,ζ₀). Therefore, the support matches χ₀(r)'s, assuming
    this region is sufficiently large compared to the singular region.
    Here χ₀(r) has support where the argument r lies in [0, 1]. The map r
    defines a coordinate mapping between the toroidal domain and a polar domain
    such that the integration region in the polar domain (ρ,ω) ∈ [−1, 1] × [0, 2π)
    equals the compact support, and furthermore is a circular region around the
    singular point in (θ,ζ) geometry when s₁ × s₂ denote the number of grid points
    on a uniformly discretized toroidal domain (θ,ζ) ∈ [0, 2π)².
      χ₀ : r ↦ exp(−36r⁸)

      r : ρ, ω ↦ |ρ|

      r : θ, ζ ↦ 2 [ (θ−θ₀)²/(h₁s₁)² + (ζ−ζ₀)²/(h₂s₂)² ]⁰ᐧ⁵

    Hence, r ≥ 1 (r ≤ 1) outside (inside) the integration domain.

    The choice for the size of the support is determined by s₁ and s₂.
    The optimal choice is dependent on the nature of the singularity e.g. if the
    integrand decays quickly then the elliptical grid determined by s₁ and s₂
    can be made smaller and the integration will have higher resolution for the
    same number of quadrature points.

    With the above definitions the support lies on an s₁ × s₂ subset
    of a field period which has ``num_theta`` × ``num_zeta`` nodes total.
    Since kernels are 2π periodic, the choice for s₂ should be multiplied by NFP.
    Then the support lies on an s₁ × s₂ subset of the full domain. For large NFP
    devices such as Heliotron or tokamaks it is typical that s₁ ≪ s₂.

    Parameters
    ----------
    theta, zeta : jnp.ndarray
        Coordinates of points to evaluate partition function η₀(θ,ζ).
    theta0, zeta0 : jnp.ndarray
        Origin (θ₀,ζ₀) where the partition η₀ is unity.
    ht, hz : float
        Grid step size in θ and ζ.
    st, sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.

    """
    dt = jnp.abs(theta - theta0)
    dz = jnp.abs(zeta - zeta0)
    # The distance spans (dθ,dζ) ∈ [0, π]², independent of NFP.
    dt = jnp.minimum(dt, 2 * jnp.pi - dt)
    dz = jnp.minimum(dz, 2 * jnp.pi - dz)
    r = 2 * jnp.hypot(dt / (ht * st), dz / (hz * sz))
    return _chi(r)


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
    ratio : float
        Mean best ratio.

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
    st = min(Nt, int(jnp.ceil(s / s_ratio)))
    sz = min(Nz, int(jnp.ceil(s * s_ratio)))
    return st, sz, q


def _local_params(grid, ratio):
    """Parameters for heuristic support size and quadrature resolution.

    These parameters account for local grid anisotropy to ensure
    more robust convergence across stronger geometric shaping.

    Parameters
    ----------
    grid : LinearGrid
        Grid that can fft2.
    ratio : tuple
        Mean best ratio and local ratio

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
    ratio = (ratio[0] + ratio[1]) / 2
    # same logic as heuristic params
    s_ratio = jnp.sqrt(Nz / Nt / ratio)
    st = min(Nt, int(jnp.ceil(s / s_ratio)))
    sz = min(Nz, int(jnp.ceil(s * s_ratio)))
    return st, sz, q


def best_ratio(data, return_local=False):
    """Ratio to make singular integration partition ~circle in real space.

    Parameters
    ----------
    data : dict[str, jnp.ndarray]
        Dictionary of data evaluated on single flux surface grid that
        ``can_fft2`` with keys ``|e_theta x e_zeta|``, ``e_theta``, and ``e_zeta``.
    return_local : bool
        Whether to return the local ratio as well as the mean global ratio.

    Returns
    -------
    mean : float
        Mean best ratio.

    """
    local = jnp.linalg.norm(data["e_zeta"], axis=-1) / jnp.linalg.norm(
        data["e_theta"], axis=-1
    )
    mean = jnp.mean(local * data["|e_theta x e_zeta|"]) / jnp.mean(
        data["|e_theta x e_zeta|"]
    )
    return (mean, local) if return_local else mean


def _get_quadrature_nodes(q):
    """Polar nodes for quadrature around singular point.

    Parameters
    ----------
    q : int
        Order of quadrature in radial and azimuthal directions.

    Returns
    -------
    r, w : ndarray
        Radial and azimuthal coordinates.
    dr, dw : ndarray
        Radial and azimuthal spacing and quadrature weights.

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
        r, w, _, _ = _get_quadrature_nodes(q)
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
        warnif(
            eval_grid.num_theta < source_grid.num_theta,
            msg="Frequency spectrum of FFT interpolation will be truncated because "
            "the evaluation grid has less resolution than the source grid.\n"
            f"Got eval_grid.num_theta = {eval_grid.num_theta} < "
            f"{source_grid.num_theta} = source_grid.num_theta.",
        )
        warnif(
            eval_grid.num_zeta < source_grid.num_zeta,
            msg="Frequency spectrum of FFT interpolation will be truncated because "
            "the evaluation grid has less resolution than the source grid.\n"
            f"Got eval_grid.num_zeta = {eval_grid.num_zeta} < "
            f"{source_grid.num_zeta} = source_grid.num_zeta.",
        )
        super().__init__(eval_grid, source_grid, st, sz, q)

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
        # Would need to add interpax code to DESC
        # https://github.com/f0uriest/interpax/issues/53
        # for is_fourier to do anything.
        shape = f.shape[1:]
        return fft_interp2d(
            self._source_grid.meshgrid_reshape(f, "rtz")[0],
            n1=self._eval_grid.num_theta,
            n2=self._eval_grid.num_zeta,
            sx=self._shift_t[i],
            sy=self._shift_z[i],
            dx=self._ht,
            dy=self._hz,
        ).reshape(self._eval_grid.num_nodes, *shape, order="F")


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
        Order of quadrature in polar domain

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
        theta = self._eval_grid.nodes[:, 1] + self._shift_t[i]
        zeta = self._eval_grid.nodes[:, 2] + self._shift_z[i]
        return rfft2_vander(theta, zeta, self._modes_fft, self._modes_rfft).reshape(
            self._eval_grid.num_nodes, -1
        )

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
):
    """Integrate kernel over non-singular points.

    Generally follows sec 3.2.1 of [2].
    """
    source_theta = source_grid.nodes[:, 1]
    # make sure source dict has zeta and phi to avoid
    # adding keys to dict during iteration
    source_zeta = source_data.setdefault("zeta", source_grid.nodes[:, 2])
    source_phi = source_data["phi"]

    eval_data = {key: eval_data[key] for key in kernel.keys if key in eval_data}
    eval_data["theta"] = jnp.asarray(eval_grid.nodes[:, 1])
    eval_data["zeta"] = jnp.asarray(eval_grid.nodes[:, 2])

    ht = 2 * jnp.pi / source_grid.num_theta
    hz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
    w = source_data["|e_theta x e_zeta|"][jnp.newaxis] * ht * hz

    def nfp_loop(j, f_data):
        """Calculate effects from source points on a single field period.

        The surface integral is computed on the full domain because the kernels of
        interest have toroidal variation and are not NFP periodic. To that end, the
        integral is computed on every field period and summed. The ``source_grid`` is
        the first field period because DESC truncates the computational domain to
        ζ ∈ [0, 2π/grid.NFP) and changes variables to the spectrally condensed
        ζ* = basis.NFP ζ. Therefore, we shift the domain to the next field period by
        incrementing the toroidal coordinate of the grid by 2π/NFP. For an axisymmetric
        configuration, it is most efficient for ``source_grid`` to be a single toroidal
        cross-section. To capture toroidal effects of the kernels on those grids for
        axisymmetric configurations, we set a dummy value for NFP to an integer larger
        than 1 so that the toroidal increment can move to a new spot.
        """
        f, source_data = f_data
        source_data["zeta"] = (source_zeta + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )
        source_data["phi"] = (source_phi + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )

        # nest this def to avoid having to pass the modified source_data around the loop
        # easier to just close over it and let JAX figure it out
        def eval_pt(eval_data_i):
            k = kernel(eval_data_i, source_data).reshape(
                -1, source_grid.num_nodes, kernel.ndim
            )
            eta = _eta(
                source_theta,
                source_data["zeta"],
                eval_data_i["theta"][:, jnp.newaxis],
                eval_data_i["zeta"][:, jnp.newaxis],
                ht,
                hz,
                st,
                sz,
            )
            return jnp.sum(k * (w * (1 - eta))[..., jnp.newaxis], axis=1)

        f += batch_map(eval_pt, eval_data, chunk_size).reshape(
            eval_grid.num_nodes, kernel.ndim
        )
        return f, source_data

    # This error should be raised earlier since this is not the only place
    # we need the higher dummy NFP value, but the error message is more
    # helpful with the nfp loop docstring.
    errorif(
        source_grid.num_zeta == 1 and source_grid.NFP == 1,
        msg="Source grid cannot compute toroidal effects.\n"
        "Increase NFP of source grid to e.g. 64.\n"
        "This is required to " + nfp_loop.__doc__,
    )
    f = jnp.zeros((eval_grid.num_nodes, kernel.ndim))
    f, _ = fori_loop(0, source_grid.NFP, nfp_loop, (f, source_data))

    # undo rotation of source_zeta
    source_data["zeta"] = source_zeta
    source_data["phi"] = source_phi
    # we sum vectors at different points, so they need to be in xyz for that to work
    # but then need to convert vectors back to rpz
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def _singular_part(eval_data, source_data, kernel, interpolator, chunk_size=None):
    """Integrate singular point by interpolating to polar grid.

    Generally follows sec 3.2.2 of [2], with the following differences:

    - hyperparameter M replaced by ``st`` and ``sz``.
    - density sigma / function f is absorbed into kernel.
    """
    eval_grid = interpolator._eval_grid
    eval_theta = jnp.asarray(eval_grid.nodes[:, 1])
    eval_zeta = jnp.asarray(eval_grid.nodes[:, 2])

    r, w, dr, dw = _get_quadrature_nodes(interpolator.q)
    r = jnp.abs(r)
    # integrand of eq 38 in [2] except stuff that needs to be interpolated
    v = (
        _chi(r)
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
    keys = list(keys)
    # Note that it is necessary to take the Fourier transforms of the
    # vector components of the orthonormal polar basis vectors R̂, ϕ̂, Ẑ.
    # Vector components of the Cartesian basis are not NFP periodic.
    fsource = [interpolator.fourier(source_data[key]) for key in keys]

    def polar_pt(i):
        """See sec 3.2.2 of [2].

        Evaluate the effect from a single polar node around each eval point
        on that eval point. Polar grids from other singularities have no effect,
        so only the diagonal term of the kernel is needed.
        """
        vander = interpolator.vander_polar(i)
        source_data_polar = {
            key: interpolator(val, i, is_fourier=True, vander=vander)
            for key, val in zip(keys, fsource)
        }
        # Coordinates of the polar nodes around the evaluation point.
        source_data_polar["theta"] = eval_theta + interpolator.shift_t[i]
        source_data_polar["zeta"] = eval_zeta + interpolator.shift_z[i]
        if "omega" in keys:
            source_data_polar["phi"] = (
                source_data_polar["zeta"] + source_data_polar["omega"]
            )
            # TODO (#465): For nonzero ω, the quadrature may not be symmetric about the
            #  singular point for hypersingular kernels such as the Biot-Savart
            #  kernel. (Recall the singularity is in real space). Hence the quadrature
            #  may not converge to the desired Hadamard finite part. Prove otherwise or
            #  use uniform grid in θ, ϕ and map coordinates before starting the singular
            #  integral routine.

        # eval pts x source pts for 1 polar grid offset
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
        # TODO (#1386): Infer jnp.add.reduce from reduction.
        #  https://github.com/jax-ml/jax/issues/23493.
        chunk_reduction=lambda x: x.sum(axis=0),
    )(jnp.arange(v.size))
    assert f.shape == (eval_grid.num_nodes, kernel.ndim)

    # we sum vectors at different points, so they need to be in xyz for that to work
    # but then need to convert vectors back to rpz
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def singular_integral(
    eval_data,
    source_data,
    kernel,
    interpolator,
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
    interpolator : _BIESTInterpolator
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
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

    out1 = _singular_part(eval_data, source_data, kernel, interpolator, chunk_size)
    out2 = _nonsingular_part(
        eval_data,
        interpolator._eval_grid,
        source_data,
        interpolator._source_grid,
        interpolator.st,
        interpolator.sz,
        kernel,
        chunk_size,
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
        _kernel_biot_savart,
        interpolator,
        chunk_size,
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
            rho=np.array([1.0]),
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )

    data_keys = ["K_vc", "B", "R", "phi", "Z", "e^rho", "n_rho", "|e_theta x e_zeta|"]
    eval_data = eq.compute(data_keys, grid=eval_grid)
    source_data = eq.compute(data_keys, grid=source_grid)
    st, sz, q = best_params(source_grid, best_ratio(source_data))
    try:
        interpolator = FFTInterpolator(eval_grid, source_grid, st, sz, q)
    except AssertionError as e:
        warnif(
            True,
            msg="Could not build fft interpolator, switching to dft which is slow."
            "\nReason: " + str(e),
        )
        interpolator = DFTInterpolator(eval_grid, source_grid, st, sz, q)
    if hasattr(eq.surface, "Phi_mn"):
        source_data["K_vc"] += eq.surface.compute("K", grid=source_grid)["K"]
    Bplasma = virtual_casing_biot_savart(
        eval_data, source_data, interpolator, chunk_size
    )
    # need extra factor of B/2 bc we're evaluating on plasma surface
    Bplasma = Bplasma + eval_data["B"] / 2
    if normal_only:
        Bplasma = jnp.sum(Bplasma * eval_data["n_rho"], axis=1)
    return Bplasma
