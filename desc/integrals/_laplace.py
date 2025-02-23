"""High order accurate vacuum field solver."""

from functools import partial

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.batching import vmap_chunked
from desc.grid import LinearGrid
from desc.integrals.quad_utils import eta_zero
from desc.integrals.singularities import (
    _get_interpolator,
    _kernel_biot_savart_coulomb,
    _kernel_Bn_over_r,
    _kernel_Phi_dGp_dn,
    _nonsingular_part,
    singular_integral,
)
from desc.io import IOAble
from desc.transform import Transform


@partial(vmap_chunked, in_axes=(None, 0, None), chunk_size=1)
def _green(self, Phi_mn, chunk_size):
    """Compute green(Φₘₙ).

    After Fourier transform, ``green`` is linear in the spectral coefficients
    Φₘₙ. Finite-dimensional approximation enables writing green(Φₘₙ) = A @ Φₘₙ.
    """
    src_data = self._data["src"].copy()
    src_data["Phi"] = self._transform["src"].transform(Phi_mn)
    Phi = (
        src_data["Phi"]
        if self._same_transform
        else self._transform["Phi"].transform(Phi_mn)
    )
    return Phi + singular_integral(
        self._data["Phi"],
        src_data,
        _kernel_Phi_dGp_dn,
        self._interpolator["Phi"],
        chunk_size,
    ).squeeze() / (2 * jnp.pi)


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
    src_grid : Grid
        Source points on ∂D for quadrature of kernels.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    interior : bool
        If true, it is assumed the evaluation grid is subset of D.
        If false, it is assumed the evaluation grid is subset of ∂D.
    B0n : jnp.ndarray
        Optional, 𝐁₀⋅𝐧 on ``src_grid``.

    """

    def __init__(
        self,
        surface,
        B0,
        evl_grid,
        Phi_grid=None,
        src_grid=None,
        chunk_size=None,
        interior=True,
        B0n=None,
        **kwargs,
    ):
        position = ["R", "phi", "Z"]
        evl_names = position + kwargs.get("evl_names", [])
        phi_names = position
        src_names = position + ["n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"]

        Phi_grid = Phi_grid or LinearGrid(
            M=surface.M_grid,
            N=surface.N_grid,
            NFP=surface.NFP if surface.N > 0 else 64,
        )
        src_grid = src_grid or Phi_grid
        basis = DoubleFourierSeries(
            M=Phi_grid.M, N=Phi_grid.N, NFP=surface.NFP, sym=False and surface.sym
        )

        # Compute source grid data.
        src_transform = Transform(src_grid, basis, derivs=1)
        src_data = surface.compute(src_names, grid=src_grid)
        # Compute Phi grid data.
        self._same_transform = src_grid.equiv(Phi_grid)
        if self._same_transform:
            Phi_transform = src_transform
            Phi_data = src_data
        else:
            Phi_transform = Transform(Phi_grid, basis)
            Phi_data = surface.compute(phi_names, grid=Phi_grid)
        # Compute eval grid data.
        if evl_grid.equiv(Phi_grid):
            evl_data = Phi_data
        elif not self._same_transform and evl_grid.equiv(src_grid):
            evl_data = src_data
        else:
            evl_data = surface.compute(evl_names, grid=evl_grid)

        if B0n is not None:
            src_data["Bn"] = B0n
        elif B0 is not None:
            src_data["Bn"], _ = B0.compute_Bnormal(
                surface,
                eval_grid=src_grid,
                source_grid=src_grid,
                vc_source_grid=src_grid,
                chunk_size=chunk_size,
            )

        self._B0 = B0
        self._evl_grid = evl_grid
        self._interior = interior
        self._data = {"evl": evl_data, "Phi": Phi_data, "src": src_data}
        self._transform = {"Phi": Phi_transform, "src": src_transform}
        self._interpolator = {
            "evl": _get_interpolator(evl_grid, src_grid, src_data, **kwargs),
            "Phi": _get_interpolator(Phi_grid, src_grid, src_data, **kwargs),
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
        if "Phi_mn" not in self._data["Phi"]:
            b = -singular_integral(
                self._data["Phi"],
                self._data["src"],
                _kernel_Bn_over_r,
                self._interpolator["Phi"],
                chunk_size,
            ).squeeze() / (2 * jnp.pi)

            # green is expensive, so better to construct Jacobian then solve
            # rather than iterative methods like ``jax.scipy.sparse.linalg.cg``
            A = _green(self, jnp.eye(self._transform["Phi"].num_modes), chunk_size).T
            self._data["Phi"]["Phi_mn"] = jnp.linalg.solve(A, b)

        return self._data

    def _compute_virtual_current(self):
        """𝐊_vc = -𝐧 × ∇Φ."""
        # Note this is not a virtual casing current.
        data = self._data["src"]
        if "K_vc" not in data:
            Phi_mn = self._data["Phi"]["Phi_mn"]
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
        self._data = self.compute_Phi_mn(chunk_size)
        self._data = self._compute_virtual_current()

        if "grad(Phi)" not in self._data["evl"]:
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
        data = self._data["evl"]
        if "B0" not in data:
            coords = jnp.column_stack([data["R"], data["phi"], data["Z"]])
            data["B0"] = self._B0.compute_magnetic_field(
                coords=coords,
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
            𝐁 stored in ``data["evl"]["B"]``.

        """
        if "B" not in self._data["evl"]:
            self._data = self.compute_current_field(chunk_size)
            self._data = self.compute_vacuum_field(chunk_size)
            self._data["evl"]["B"] = (
                self._data["evl"]["B0"] + self._data["evl"]["grad(Phi)"]
            )
        return self._data
