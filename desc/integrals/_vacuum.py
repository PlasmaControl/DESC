"""High order accurate vacuum field solver."""

from functools import partial

from matplotlib import pyplot as plt

from desc.backend import fixed_point, irfft2, jit, jnp, rfft2
from desc.basis import DoubleFourierSeries
from desc.batching import vmap_chunked
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


def _nonhomogenous_part(self, chunk_size):
    return jnp.atleast_1d(
        -singular_integral(
            self._data["Phi"],
            self._data["src"],
            interpolator=self._interpolator["Phi"],
            kernel=_kernel_Bn_over_r,
            chunk_size=chunk_size,
        ).squeeze()
        / (2 * jnp.pi)
    )


@partial(vmap_chunked, in_axes=(None, 0, 0, None), chunk_size=None)
def _fourier_operator(self, m, n, chunk_size):
    # Finite-dimensional approximation of linear operator is
    # _fourier_operator(Φₘₙ) = F @ Φₘₙ.
    Phi = self.basis._get_fun_mode(m, n)
    evl_Phi = Phi(self.Phi_grid)
    src_data = self._data["src"].copy()
    src_data["Phi"] = evl_Phi if self._same_grid_phi_src else Phi(self.src_grid)

    return evl_Phi + singular_integral(
        eval_data=self._data["Phi"],
        source_data=src_data,
        interpolator=self._interpolator["Phi"],
        kernel=_kernel_Phi_dGp_dn,
        known_map=("Phi", Phi),
        chunk_size=chunk_size,
    ).squeeze() / (2 * jnp.pi)


@partial(jit, static_argnames=["chunk_size"])
def _fourier_compute_Phi(self, *, chunk_size=None):
    if "Phi_mn" in self._data["Phi"]:
        return self._data

    g = _nonhomogenous_part(self, chunk_size)
    _, m, n = self.basis.modes.T
    F = _fourier_operator(self, m, n, chunk_size).T
    assert F.shape == (self.Phi_grid.num_nodes, self.basis.num_modes)
    # Solving overdetermined system useful to reduce size of F while
    # retaining FFT interpolation accuracy in the singular integrals.
    # TODO: use ``jax.scipy.sparse.linalg.cg``
    self._data["Phi"]["Phi_mn"] = (
        jnp.linalg.solve(F, g)
        if (self.Phi_grid.num_nodes == self.basis.num_modes)
        else jnp.linalg.lstsq(F, g)[0]
    )
    return self._data


# TODO: iterate on Phi_mn
def _fredholm_operator(Phi_k, g, self, chunk_size=None):
    """Compute Fredholm integral operator Tf such that Tf = f.

    Parameters
    ----------
    Phi_k : jnp.ndarray
        Φ values on ``self._Phi_grid``.
    g : jnp.ndarray
        Non-homogenous term on ``self._Phi_grid``.

    Returns
    -------
    Tf : jnp.ndarray
        Fredholm integral operator computed on ``self._Phi_grid``.

    """
    src_data = self._data["src"].copy()
    src_data["Phi"] = self._upsample_to_source(Phi_k)
    return g - singular_integral(
        eval_data=self._data["Phi"],
        source_data=src_data,
        interpolator=self._interpolator["Phi"],
        kernel=_kernel_Phi_dGp_dn,
        chunk_size=chunk_size,
    ).squeeze() / (2 * jnp.pi)


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

    g = _nonhomogenous_part(self, chunk_size)
    if Phi_0 is None:
        # TODO: Add option to set initial guess to first k
        #  terms of Fourier fit (output of ``_compute_Phi_mn``).
        #  Useful if k is small.
        _, m, n = self.basis.modes.T
        Phi_0 = self.basis._get_fun_mode(m[1], n[1])(self.Phi_grid)
    self._data["Phi"]["Phi"] = fixed_point(
        _fredholm_operator,
        Phi_0,
        (g, self, chunk_size),
        tol,
        maxiter,
        method,
        scalar=True,
    )
    self._data["Phi"]["Phi_mn"] = self._phi_transform.fit(self._data["Phi"]["Phi"])
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

    def _upsample_to_source(self, x):
        if not self._same_grid_phi_src:
            x = self.Phi_grid.meshgrid_reshape(x, "rtz")[0]
            x = irfft2(
                rfft2(x, norm="forward", axes=(0, 1)),
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
        self, chunk_size=None, maxiter=0, tol=1e-6, method="del2", Phi_0=None
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
            Fourier series. Default is Fourier series with unit coefficients
            for two modes (that have frequency zero and one).

        Returns
        -------
        data : dict
             Fourier coefficients of Φ on ∂D stored in ``data["Phi"]["Phi_mn"]``.

        """
        warnif(maxiter > 0, msg="Still debugging")
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
            else _fourier_compute_Phi(self, chunk_size=chunk_size)
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
