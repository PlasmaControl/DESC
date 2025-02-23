"""High order accurate vacuum field solver."""

from functools import partial

from matplotlib import pyplot as plt

from desc.backend import jit, jnp
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
from desc.utils import errorif, setdefault


@partial(vmap_chunked, in_axes=(None, 0, None), chunk_size=None)
def _green(self, Phi_mn, chunk_size):
    """Compute green(Φₘₙ).

    green is linear map of Φₘₙ.
    Finite-dimensional approximation gives green(Φₘₙ) = A @ Φₘₙ.
    """
    src_data = self._data["src"].copy()
    src_data["Phi"] = self._transform["src"].transform(Phi_mn)
    Phi = (
        src_data["Phi"]
        if self._same_grid_phi_src
        else self._transform["Phi"].transform(Phi_mn)
    )
    # TODO: Can generalize this to be one singular integral without interpolation.
    return Phi + singular_integral(
        self._data["Phi"],
        src_data,
        _kernel_Phi_dGp_dn,
        self._interpolator["Phi"],
        chunk_size,
    ).squeeze() / (2 * jnp.pi)


@partial(jit, static_argnames=["chunk_size"])
def _compute_Phi_mn(self, *, chunk_size=None):
    if "Phi_mn" not in self._data["Phi"]:
        num_modes = self._transform["Phi"].num_modes
        num_nodes = self._transform["Phi"].grid.num_nodes
        full_rank = num_modes == num_nodes

        b = -singular_integral(
            self._data["Phi"],
            self._data["src"],
            _kernel_Bn_over_r,
            self._interpolator["Phi"],
            chunk_size,
        ).squeeze() / (2 * jnp.pi)
        # Least squares method can significantly reduce size of A while
        # retaining FFT interpolation accuracy in the singular integrals.
        # Green is expensive, so constructing Jacobian A then solving
        # better than iterative methods like ``jax.scipy.sparse.linalg.cg``.
        A = _green(self, jnp.eye(num_modes), chunk_size).T
        self._data["Phi"]["Phi_mn"] = (
            jnp.linalg.solve(A, b) if full_rank else jnp.linalg.lstsq(A, b)[0]
        )
    return self._data


class VacuumSolver(IOAble):
    """Compute vacuum field that satisfies LCFS boundary condition in plasma interior.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - 𝐁 ⋅ 𝐧 = 0      on ∂D
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
    Phi_grid : Grid
        Interpolation points on ∂D.
        Resolution determines accuracy of Φ interpolation.
        Default resolution is ``Phi_grid.M=surface.M*2`` and ``Phi_grid.N=surface.N*2``.
    src_grid : Grid
        Source points on ∂D for quadrature of kernels.
        Default resolution is ``src_grid.M=surface.M*4`` and ``src_grid.N=surface.N*4``.
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
        instead of inverse fast Fourier transform. This is useful when the real domain
        grid is much sparser than the spectral domain grid because the MMT is exact
        while the FFT will truncate higher frequencies.

    """

    def __init__(
        self,
        surface,
        B0,
        evl_grid,
        Phi_grid=None,
        src_grid=None,
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
        # TODO (#1599).
        errorif(
            use_dft,
            RuntimeError,
            msg="JAX performs matrix multiplication incorrectly for large matrices. "
            "Until this is fixed, the inversion to obtain Phi cannot be done "
            "with an interpolator that uses matrix multiplication transforms.",
        )
        # TODO (#1206)
        if Phi_grid is None:
            Phi_grid = LinearGrid(
                M=surface.M * 2,
                N=surface.N * 2,
                NFP=surface.NFP if surface.N > 0 else 64,
            )
        if src_grid is None:
            src_grid = Phi_grid = LinearGrid(
                M=surface.M * 4,
                N=surface.N * 4,
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
        src_transform = Transform(src_grid, basis, derivs=1)
        src_data = surface.compute(
            position + ["n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"],
            grid=src_grid,
        )
        # Compute data on Phi grid.
        if self._same_grid_phi_src:
            Phi_transform = src_transform
            Phi_data = src_data
        else:
            Phi_transform = Transform(Phi_grid, basis)
            Phi_data = surface.compute(position, grid=Phi_grid)
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
        self._evl_grid = evl_grid
        self._interior = interior
        self._data = {"evl": evl_data, "Phi": Phi_data, "src": src_data}
        self._transform = {"Phi": Phi_transform, "src": src_transform}
        self._interpolator = {
            "evl": get_interpolator(evl_grid, src_grid, src_data, use_dft, **kwargs),
            "Phi": get_interpolator(Phi_grid, src_grid, src_data, use_dft, **kwargs),
        }

    def compute_Phi_mn(self, chunk_size=None):
        """Compute Fourier coefficients of vacuum potential Φ on ∂D.

        Parameters
        ----------
        chunk_size : int or None
            Size to split computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        data : dict
             Fourier coefficients of Φ on ∂D stored in ``data["Phi"]["Phi_mn"]``.

        """
        return _compute_Phi_mn(self, chunk_size=chunk_size)

    def _compute_virtual_current(self):
        """𝐊_vc = -𝐧 × ∇Φ.

        This is the vacuum portion of the virtual casing current.
        """
        if "K_vc" in self._data["src"]:
            return self._data

        Phi_mn = self._data["Phi"]["Phi_mn"]
        data = self._data["src"]
        data["Phi_t"] = self._transform["src"].transform(Phi_mn, dt=1)
        data["Phi_z"] = self._transform["src"].transform(Phi_mn, dz=1)
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
        if "grad(phi)" in self._data["evl"]:
            return self._data

        self._data = self.compute_Phi_mn(chunk_size)
        self._data = self._compute_virtual_current()
        self._data["evl"]["grad(Phi)"] = (
            _nonsingular_part(
                self._data["evl"],
                self._evl_grid,
                self._data["src"],
                self._transform["src"].grid,
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
                kernel=_kernel_biot_savart_coulomb,
                interpolator=self._interpolator["evl"],
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
            source_grid=self._transform["src"].grid,
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
        grid = self._evl_grid
        theta = grid.meshgrid_reshape(grid.nodes[:, 1], "rtz")[0]
        zeta = grid.meshgrid_reshape(grid.nodes[:, 2], "rtz")[0]
        Bn = grid.meshgrid_reshape(Bn, "rtz")[0]

        fig, ax = plt.subplots()
        contour = ax.contourf(theta, zeta, Bn)
        fig.colorbar(contour, ax=ax)
        ax.set_title(r"$(B_0 + \nabla \Phi) \cdot n$ on $\partial D$")
        return fig, ax
