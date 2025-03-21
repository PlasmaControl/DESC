"""High order accurate vacuum field solver."""

from functools import partial

from matplotlib import pyplot as plt

from desc.backend import fixed_point, irfft2, jit, jnp, rfft2
from desc.basis import DoubleFourierSeries
from desc.grid import LinearGrid
from desc.integrals.quad_utils import eta_zero
from desc.integrals.singularities import (
    _kernel_biot_savart_coulomb,
    _kernel_Bn_over_r,
    _kernel_Phi_dGp_dn,
    _nonsingular_part,
    get_interpolator,
    singular_integral,
)
from desc.io import IOAble
from desc.transform import Transform
from desc.utils import errorif, setdefault, warnif


def _to_real_coef(grid, f):
    f = rfft2(
        f.reshape(grid.num_theta, grid.num_zeta),
        norm="forward",
        axes=(0, 1),
    ).ravel()
    return jnp.concatenate([f.real, f.imag])


def _to_rfft(grid, f):
    f = f[: f.size // 2] + 1j * f[f.size // 2 :]
    f = f.reshape(grid.num_theta, grid.num_zeta // 2 + 1)
    return f


def _nonhomogenous_part(self, chunk_size):
    return -singular_integral(
        self._data["Phi"],
        self._data["src"],
        interpolator=self._interpolator["Phi"],
        kernel=_kernel_Bn_over_r,
        chunk_size=chunk_size,
    ).squeeze(axis=-1) / (2 * jnp.pi)


@partial(jit, static_argnames=["chunk_size"])
def _lsmr_compute_Phi(self, basis=None, *, chunk_size=None):
    """Compute Fourier harmonics Φ_mn by solving least squares system.

    This is an Adomian Decomposition method, but because the
    integral equation is linear under Fourier decomposition
    the formula for the coefficients is known (i.e. not recursive).
    """
    if "Phi_mn" in self._data["Phi"]:
        return self._data

    g = _nonhomogenous_part(self, chunk_size)

    basis = setdefault(basis, self.basis)
    evl_Phi = basis.evaluate(self.Phi_grid)
    src_data = self._data["src"].copy()
    src_data["Phi"] = (
        evl_Phi if self._same_grid_phi_src else basis.evaluate(self.src_grid)
    )
    F = evl_Phi + singular_integral(
        eval_data=self._data["Phi"],
        source_data=src_data,
        interpolator=self._interpolator["Phi"],
        kernel=_kernel_Phi_dGp_dn,
        known_map=("Phi", basis.evaluate),
        ndim=basis.num_modes,
        chunk_size=chunk_size,
    ) / (2 * jnp.pi)
    assert F.shape == (self.Phi_grid.num_nodes, basis.num_modes)

    # Solving overdetermined system useful to reduce size of F while
    # retaining FFT interpolation accuracy in the singular integrals.
    # TODO: https://github.com/patrick-kidger/lineax/pull/86
    #  JAX doesn't have lsmr yet, but apparently Rory is working on it.
    self._data["Phi"]["Phi_mn"] = (
        jnp.linalg.solve(F, g)
        if (self.Phi_grid.num_nodes == basis.num_modes)
        else jnp.linalg.lstsq(F, g)[0]
    )
    return self._data


def _fredholm_operator(Phi_k, g, self, chunk_size=None):
    """Compute Fredholm integral operator such that T(Φ_mn) → Φ_mn.

    Note that the iteration is on the Fourier coefficients to
    bypass approximation error.

    Parameters
    ----------
    Phi_k : jnp.ndarray
        Φ_mn values from ``self._Phi_grid``.
    g : jnp.ndarray
        Non-homogenous term on ``self._Phi_grid``.

    Returns
    -------
    Phi_k+1 : jnp.ndarray
        Fredholm integral operator computed on ``self._Phi_grid``.

    """
    # Phi_k = _to_rfft(self.Phi_grid, Phi_k)  # noqa

    src_data = self._data["src"].copy()
    src_data["Phi"] = self._upsample_to_source(Phi_k, is_fourier=False)

    # TODO: Don't need to re-interpolate Phi since we already have it.
    #       Requires resolving issue described in _interpax_mod.py.
    Phi_k = g - singular_integral(
        eval_data=self._data["Phi"],
        source_data=src_data,
        interpolator=self._interpolator["Phi"],
        kernel=_kernel_Phi_dGp_dn,
        chunk_size=chunk_size,
    ).squeeze(axis=-1) / (2 * jnp.pi)

    # Phi_k = _to_real_coef(self.Phi_grid, Phi_k)  # noqa
    return Phi_k


@partial(jit, static_argnames=["tol", "maxiter", "method", "chunk_size"])
def _fixed_point_compute_Phi(
    self,
    Phi_0=None,
    *,
    tol=1e-6,
    maxiter=20,
    method="del2",
    chunk_size=None,
):
    assert self.Phi_grid.can_fft2
    if "Phi_mn" in self._data["Phi"]:
        return self._data

    if Phi_0 is None:
        basis = DoubleFourierSeries(
            M=min(self.basis.M, 3),
            N=min(self.basis.N, 3),
            NFP=self.basis.NFP,
            sym=self.basis.sym,
        )
        self._data = _lsmr_compute_Phi(self, basis, chunk_size=chunk_size)
        Phi_0 = basis.evaluate(self.Phi_grid) @ self._data["Phi"]["Phi_mn"]
    # Phi_0 = _to_real_coef(self.Phi_grid, Phi_0)   # noqa
    Phi = fixed_point(
        _fredholm_operator,
        Phi_0,
        (_nonhomogenous_part(self, chunk_size), self, chunk_size),
        tol,
        maxiter,
        method,
        scalar=True,
    )
    # Phi = irfft2(   # noqa
    #     _to_rfft(self.Phi_grid, Phi),  # noqa
    #     s=(self.Phi_grid.num_theta, self.Phi_grid.num_zeta),  # noqa
    #     norm="forward",  # noqa
    #     axes=(0, 1),  # noqa
    # ).reshape(self.Phi_grid.num_nodes, order="F")  # noqa

    self._data["Phi"]["Phi_mn"] = self._phi_transform.fit(Phi)
    return self._data


class VacuumSolver(IOAble):
    """Compute vacuum field that satisfies LCFS boundary condition.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - 𝐁 ⋅ 𝐧 = 0       on ∂D
    - ∇ × 𝐁₀ = μ₀ 𝐉   on D ∪ D^∁
    - ∇ ⋅ 𝐁₀ = 0      on D ∪ D^∁
    - ∇²Φ = 0         on D

    That is, given a magnetic field 𝐁₀ due to volume current sources,
    finds the unique vacuum field ∇Φ such that 𝐁 ⋅ 𝐧 = 0 without assuming
    nested flux surfaces.

    Examples
    --------
    In a vacuum, the magnetic field may be written 𝐁 = ∇𝛷. The solution to
    ∇²𝛷 = 0, under a homogenous boundary condition 𝐁 ⋅ 𝐧 = 0, is 𝛷 = 0. To
    obtain a non-trivial solution, the boundary condition may be modified.
    Let 𝐁 = 𝐁₀ + ∇Φ. If 𝐁₀ ≠ 0 and satisfies ∇ × 𝐁₀ = 0, then ∇²Φ = 0 solved
    under an inhomogeneous boundary condition yields a non-trivial solution.
    If 𝐁₀ ≠ -∇Φ, then 𝐁 ≠ 0.

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂D.
    B0 : _MagneticField
        Magnetic field such that ∇ × 𝐁₀ = μ₀ 𝐉
        where 𝐉 is the current in amperes everywhere.
    evl_grid : Grid
        Evaluation points on D for the magnetic field.
    src_grid : Grid
        Source points on ∂D for quadrature of kernels.
        Default resolution is ``src_grid.M=surface.M*4`` and ``src_grid.N=surface.N*4``.
    Phi_grid : Grid
        Interpolation points on ∂D.
        Resolution determines accuracy of Φ interpolation.
        Default resolution is ``Phi_grid.M=surface.M*2`` and ``Phi_grid.N=surface.N*2``.
    Phi_M : int
        Poloidal Fourier resolution to interpolate Φ on ∂D.
        Should be at most ``Phi_grid.M``.
    Phi_N : int
        Toroidal Fourier resolution to interpolate Φ on ∂D.
        Should be at most ``Phi_grid.N``.
    sym
        Symmetry for interpolation basis.
    interior : bool
        If true, it is assumed the evaluation grid is subset of D.
        If false, it is assumed the evaluation grid is subset of ∂D.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    B0n : jnp.ndarray
        Optional, 𝐁₀⋅𝐧 on ``src_grid``.
    use_dft : bool
        Whether to use matrix multiplication transform from spectral to physical domain
        instead of inverse fast Fourier transform.

    """

    def __init__(
        self,
        surface,
        B0,
        evl_grid,
        src_grid=None,
        Phi_grid=None,
        Phi_M=None,
        Phi_N=None,
        sym=None,
        interior=True,
        *,
        chunk_size=None,
        B0n=None,
        use_dft=False,
        **kwargs,
    ):
        # TODO (#1206)
        if src_grid is None:
            src_grid = LinearGrid(
                M=surface.M * 4,
                N=surface.N * 4,
                NFP=surface.NFP if surface.N > 0 else 64,
            )
        if Phi_grid is None:
            Phi_grid = LinearGrid(
                M=surface.M * 2,
                N=surface.N * 2,
                NFP=surface.NFP if surface.N > 0 else 64,
            )
        self._same_grid_phi_src = src_grid.equiv(Phi_grid)

        errorif(
            Phi_M is not None and Phi_M > Phi_grid.M,
            msg=f"Got Phi_M={Phi_M} > {Phi_grid.M}=Phi_grid.M.",
        )
        errorif(
            Phi_N is not None and Phi_N > Phi_grid.N,
            msg=f"Got Phi_N={Phi_N} > {Phi_grid.N}=Phi_grid.N.",
        )
        basis = DoubleFourierSeries(
            M=setdefault(Phi_M, Phi_grid.M),
            N=setdefault(Phi_N, Phi_grid.N),
            NFP=surface.NFP,
            # TODO: Reviewer should update this.
            sym=setdefault(sym, False) and surface.sym,
        )

        # Compute data on source grid.
        position = ["R", "phi", "Z"]
        self._src_transform = Transform(src_grid, basis, derivs=1, build_pinv=True)
        src_data = surface.compute(
            position + ["n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"],
            grid=src_grid,
        )
        # Compute data on Phi grid.
        if self._same_grid_phi_src:
            Phi_data = src_data
            self._src_transform.build_pinv()
            self._phi_transform = self._src_transform
        else:
            Phi_data = surface.compute(position, grid=Phi_grid)
            self._phi_transform = Transform(
                Phi_grid, basis, build=False, build_pinv=True
            )

        # Compute data on evaluation grid.
        if evl_grid.equiv(Phi_grid):
            evl_data = Phi_data
        elif not self._same_grid_phi_src and evl_grid.equiv(src_grid):
            evl_data = src_data
        else:
            evl_data = surface.compute(position, grid=evl_grid)

        errorif(B0 is None and B0n is None)
        src_data["Bn"] = (
            B0n
            if B0n is not None
            else B0.compute_Bnormal(
                surface,
                eval_grid=src_grid,
                source_grid=src_grid,
                vc_source_grid=src_grid,
                chunk_size=chunk_size,
            )[0]
        )

        self._B0 = B0
        self._interior = interior
        self._data = {"evl": evl_data, "Phi": Phi_data, "src": src_data}
        self._interpolator = {
            "evl": get_interpolator(evl_grid, src_grid, src_data, use_dft, **kwargs),
            "Phi": get_interpolator(Phi_grid, src_grid, src_data, use_dft, **kwargs),
        }

    @property
    def evl_grid(self):
        """Return the evaluation grid used by this solver."""
        return self._interpolator["evl"]._eval_grid

    @property
    def src_grid(self):
        """Return the source grid used by this solver."""
        return self._interpolator["evl"]._source_grid

    @property
    def Phi_grid(self):
        """Return the source grid used by this solver."""
        return self._interpolator["Phi"]._eval_grid

    def _upsample_to_source(self, x, is_fourier=False):
        if not self._same_grid_phi_src:
            if not is_fourier:
                x = self.Phi_grid.meshgrid_reshape(x, "rtz")[0]
                x = rfft2(x, norm="forward", axes=(0, 1))
            x = irfft2(
                x,
                s=(self.src_grid.num_theta, self.src_grid.num_zeta),
                norm="forward",
                axes=(0, 1),
            ).reshape(self.src_grid.num_nodes, *x.shape[2:], order="F")
        return x

    @property
    def basis(self):
        """Return the DoubleFourierBasis used by this solver."""
        return self._src_transform.basis

    def compute_Phi(
        self, chunk_size=None, maxiter=0, tol=1e-6, method="del2", Phi_0=None, **kwargs
    ):
        """Compute Fourier coefficients of vacuum potential Φ on ∂D.

        Parameters
        ----------
        chunk_size : int or None
            Size to split singular integral computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``.
        maxiter : int
            Maximum number of fixed point iterations.
            Set to zero to invert the system instead.
        tol : float
            Stopping tolerance for iteration.
        method : {"del2", "simple"}
            Method of finding the fixed-point, defaults to ``del2``,
            which uses Steffensen's acceleration method.
            The former typically converges quadratically and the latter converges
            linearly.
        Phi_0 : jnp.ndarray
            Initial guess for Φ on ``self.Phi_grid`` for iteration.
            In general, it is best to select the initial guess as truncated
            Fourier series. Default is a fit to a low resolution solution.

        Returns
        -------
        data : dict
             Fourier coefficients of Φ on ∂D stored in ``data["Phi"]["Phi_mn"]``.

        """
        warnif(kwargs.get("warn", True) and (maxiter > 0), msg="Still debugging")
        self._data = (
            _fixed_point_compute_Phi(
                self,
                Phi_0,
                tol=tol,
                maxiter=maxiter,
                method=method,
                chunk_size=chunk_size,
            )
            if (maxiter > 0)
            else _lsmr_compute_Phi(self, chunk_size=chunk_size)
        )
        return self._data

    def _compute_virtual_current(self):
        """𝐊_vc = -𝐧 × ∇Φ.

        This is the vacuum portion of the virtual casing current.
        """
        if "K_vc" in self._data["src"]:
            return self._data

        Phi_mn = self._data["Phi"]["Phi_mn"]
        data = self._data["src"]
        data["Phi_t"] = self._src_transform.transform(Phi_mn, dt=1)
        data["Phi_z"] = self._src_transform.transform(Phi_mn, dz=1)
        data["K^theta"] = data["Phi_z"] / data["|e_theta x e_zeta|"]
        data["K^zeta"] = -data["Phi_t"] / data["|e_theta x e_zeta|"]
        data["K_vc"] = (
            data["K^theta"][:, jnp.newaxis] * data["e_theta"]
            + data["K^zeta"][:, jnp.newaxis] * data["e_zeta"]
        )
        return self._data

    def compute_vacuum_field(self, chunk_size=None):
        """Compute magnetic field due to vacuum potential Φ.

        Parameters
        ----------
        chunk_size : int or None
            Size to split computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        data : dict
             Vacuum field ∇Φ stored in ``data["evl"]["grad(Phi)"]``.

        """
        if "grad(Phi)" in self._data["evl"]:
            return self._data

        self._data = self.compute_Phi(chunk_size)
        self._data = self._compute_virtual_current()
        self._data["evl"]["grad(Phi)"] = (
            _nonsingular_part(
                self._data["evl"],
                self.evl_grid,
                self._data["src"],
                self.src_grid,
                st=jnp.nan,
                sz=jnp.nan,
                kernel=_kernel_biot_savart_coulomb,
                chunk_size=chunk_size,
                _eta=eta_zero,
            )
            if self._interior
            else 2
            * singular_integral(
                self._data["evl"],
                self._data["src"],
                interpolator=self._interpolator["evl"],
                kernel=_kernel_biot_savart_coulomb,
                chunk_size=chunk_size,
            )
        )
        return self._data

    def compute_current_field(self, chunk_size=None):
        """Compute magnetic field 𝐁₀ due to volume current sources.

        Parameters
        ----------
        chunk_size : int or None
            Size to split computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        data : dict
            𝐁₀ stored in ``data["evl"]["B0"]``.

        """
        if "B0" in self._data["evl"]:
            return self._data

        data = self._data["evl"]
        data["B0"] = self._B0.compute_magnetic_field(
            coords=jnp.column_stack([data["R"], data["phi"], data["Z"]]),
            source_grid=self.src_grid,
            chunk_size=chunk_size,
        )
        return self._data

    def compute_magnetic_field(self, chunk_size=None):
        """Compute magnetic field 𝐁 = 𝐁₀ + ∇Φ.

        Parameters
        ----------
        chunk_size : int or None
            Size to split computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        data : dict
            𝐁 stored in ``data["evl"]["B0+grad(Phi)"]``.

        """
        if "B0+grad(Phi)" in self._data["evl"]:
            return self._data

        self._data = self.compute_current_field(chunk_size)
        self._data = self.compute_vacuum_field(chunk_size)
        self._data["evl"]["B0+grad(Phi)"] = (
            self._data["evl"]["B0"] + self._data["evl"]["grad(Phi)"]
        )
        return self._data

    def plot_Bn_error(self, Bn):
        """Plot 𝐁 ⋅ 𝐧 error on ∂D.

        Parameters
        ----------
        Bn : jnp.ndarray
            𝐁 ⋅ 𝐧 on the evaluation grid.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        errorif(self._interior)
        grid = self.evl_grid
        theta = grid.meshgrid_reshape(grid.nodes[:, 1], "rtz")[0]
        zeta = grid.meshgrid_reshape(grid.nodes[:, 2], "rtz")[0]
        Bn = grid.meshgrid_reshape(Bn, "rtz")[0]

        fig, ax = plt.subplots()
        contour = ax.contourf(theta, zeta, Bn)
        fig.colorbar(contour, ax=ax)
        ax.set_title(r"$(B_0 + \nabla \Phi) \cdot n$ on $\partial D$")
        return fig, ax
