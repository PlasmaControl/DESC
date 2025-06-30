from functools import partial

from desc.backend import jit, jnp
from desc.basis import DoubleFourierSeries
from desc.grid import LinearGrid
from desc.integrals._vacuum import _lsmr_Phi, _surface_gradient
from desc.integrals.singularities import (
    _dx,
    _kernel_biot_savart,
    _kernel_biot_savart_A,
    get_interpolator,
    singular_integral,
)
from desc.io import IOAble
from desc.transform import Transform
from desc.utils import cross, dot, errorif, setdefault


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


# TODO: (kaya) Section 5.2.2 Diagonal and smooth potential formulation


class FreeBoundarySolver(IOAble):
    """Compute exterior field for free boundary problem.

    References
    ----------
       [1] Unalmis et al. New high-order accurate free surface stellarator
           equilibria optimization and boundary integral methods in DESC.

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂𝒳.
    evl_grid : Grid
        Evaluation points on ∂𝒳.
    src_grid : Grid
        Source points on ∂𝒳 for quadrature of kernels.
        Default resolution is ``src_grid.M=surface.M*4`` and ``src_grid.N=surface.N*4``.
    Phi_grid : Grid
        Interpolation points on ∂𝒳.
        Resolution determines accuracy of interpolation for quadrature.
        Default is ``src_grid``; lower often slows convergence.
    Phi_M : int
        Poloidal Fourier resolution to interpolate Φ on ∂𝒳.
        Should be at most ``Phi_grid.M``.
    Phi_N : int
        Toroidal Fourier resolution to interpolate Φ on ∂𝒳.
        Should be at most ``Phi_grid.N``.
    sym
        Symmetry for basis which interpolates Φ.
        Default assumes no symmetry.
    B_coil : _MagneticField
        Magnetic field produced by coils.
    Y_coil : float or None
        Net poloidal coil current.
        If given ``None`` will be computed via A.3 in [1].
    Y_plasma : float
        Net poloidal plasma current.
        May be computed via A.3 in [1] with V = B_plasma.
        For parallel free boundary computations this parameter must be consistent
        with the rotational transform and flux given to the inner free boundary
        solver. See section 5.2.3 in [1].
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
        Recommend to verify computation with ``chunk_size`` set to a small number
        due to bugs in Google's JAX or the XLA software.
    use_dft : bool
        Whether to use matrix multiplication transform from spectral to physical domain
        instead of inverse fast Fourier transform.

    """

    def __init__(
        self,
        surface,
        evl_grid,
        src_grid=None,
        Phi_grid=None,
        Phi_M=None,
        Phi_N=None,
        sym=None,
        *,
        B_coil,
        Y_coil,
        Y_plasma,
        chunk_size=None,
        use_dft=False,
        **kwargs,
    ):
        self._exterior = True
        self._I = 0
        self._Y = Y_coil + Y_plasma
        errorif(
            evl_grid.nodes[0, 0] < evl_grid.num_rho,
            msg="Evaluation grid must be on boundary.",
        )
        if src_grid is None:
            src_grid = LinearGrid(
                M=surface.M * 4,
                N=surface.N * 4,
                NFP=surface.NFP if surface.N > 0 else 64,
                sym=False,  # TODO (#1206)
            )
        if Phi_grid is None:
            Phi_grid = src_grid
            self._src_grid_equals_Phi_grid = True
        else:
            self._src_grid_equals_Phi_grid = src_grid.equiv(Phi_grid)

        assert evl_grid.can_fft2
        assert src_grid.can_fft2
        assert Phi_grid.can_fft2

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
            sym=setdefault(sym, False),
        )

        # Compute data on source grid.
        names = [
            "x",
            "n_rho",
            "|e_theta x e_zeta|",
            "e_theta",
            "e_zeta",
            "n_rho x grad(theta)",
            "n_rho x grad(zeta)",
        ]
        src_data = surface.compute(names, grid=src_grid)
        # Compute data on Phi grid.
        if self._src_grid_equals_Phi_grid:
            Phi_data = src_data
        else:
            Phi_data = surface.compute(names, grid=Phi_grid)
        self._phi_transform = Transform(Phi_grid, basis, derivs=1, build_pinv=True)

        Bcoil = B_coil.compute_magnetic_field(
            coords=Phi_data["x"], source_grid=src_grid, chunk_size=chunk_size
        )
        Phi_data["n_rho x B_coil"] = cross(Phi_data["n_rho"], Bcoil)
        if Y_coil is None:
            # A.3 in [1] averaged over all χ_θ for increased accuracy since we
            # only have discrete interpolation to true B_coil.
            # (l2 norm error of fourier series better than max pointwise).
            self._Y_coil = dot(Bcoil, Phi_data["e_zeta"]).mean()
        else:
            self._Y_coil = Y_coil

        # Compute data on evaluation grid.
        if evl_grid.equiv(Phi_grid):
            evl_data = Phi_data
            self._evl_transform = self._phi_transform
        else:
            self._evl_transform = Transform(evl_grid, basis, derivs=1)
            if not self._src_grid_equals_Phi_grid and evl_grid.equiv(src_grid):
                evl_data = src_data
            else:
                evl_data = surface.compute(names, grid=evl_grid)

        self._data = {"evl": evl_data, "Phi": Phi_data, "src": src_data}
        self._interpolator = {
            "Phi": get_interpolator(Phi_grid, src_grid, src_data, use_dft, **kwargs),
            "evl": get_interpolator(evl_grid, src_grid, src_data, use_dft, **kwargs),
        }

    @property
    def evl_grid(self):
        """Return the evaluation grid used by this solver."""
        return self._interpolator["evl"]._eval_grid

    @property
    def src_grid(self):
        """Return the source grid used by this solver."""
        return self._interpolator["Phi"]._source_grid

    @property
    def Phi_grid(self):
        """Return the source grid used by this solver."""
        return self._interpolator["Phi"]._eval_grid

    @property
    def I(self):  # noqa: E743
        """Net toroidal current in 𝒳 which is a source for the field ∇Φ."""
        return self._I

    @property
    def Y(self):
        """Net poloidal current outside closure(𝒳).

        That is a source for the field ∇Φ.
        """
        return self._Y

    @property
    def basis(self):
        """Return the DoubleFourierBasis used by this solver."""
        return self._phi_transform.basis

    def compute_Phi(self, chunk_size=None):
        """Compute Fourier coefficients of vacuum potential Φ on ∂𝒳.

        Parameters
        ----------
        chunk_size : int or None
            Size to split singular integral computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``.

        Returns
        -------
        data : dict
             Fourier coefficients of Φ on ∂𝒳 stored in ``data["Phi"]["Phi_mn"]``.

        """
        self._data = _lsmr_Phi(self, bc=_free_boundary_bc, chunk_size=chunk_size)
        return self._data

    def compute_B2(self, chunk_size=None):
        """Compute ‖B_out‖² on the evaluation grid.

        Returns
        -------
        data : dict
            ‖B_out‖² stored in ``data["evl"]["|B_out|^2"]``.

        """
        if "|B_out|^2" in self._data["evl"]:
            return self._data
        self._data = self.compute_Phi(chunk_size)
        B_out_tan = _surface_gradient(
            self._data["evl"], self._evl_transform, self._data["Phi"]["Phi_mn"]
        )
        self._data["evl"]["|B_out|^2"] = dot(B_out_tan, B_out_tan)
        return self._data


def _sg_mat(data, basis, grid):
    """Returns n × ∇ in shape (num nodes * 3, num modes)."""
    _t = basis.evaluate(grid, [0, 1, 0])[:, jnp.newaxis]
    _z = basis.evaluate(grid, [0, 0, 1])[:, jnp.newaxis]
    sg = (
        _t * data["n_rho x grad(theta)"][..., jnp.newaxis]
        + _z * data["n_rho x grad(zeta)"][..., jnp.newaxis]
    )
    return sg.reshape(grid.num_nodes * 3, basis.num_modes)


def _free_boundary_bc(self, chunk_size=None):
    """Returns γ = (n × ∇)⁻¹ (n × B_coil)."""
    data = self._data["Phi"]
    if "gamma" in data:
        return self._data

    sg_gamma_periodic = (
        data["n_rho x B_coil"] - self._Y_coil * data["n_rho x grad(zeta)"]
    ).ravel()
    gamma_periodic = jnp.linalg.lstsq(
        _sg_mat(data, self.basis, self.Phi_grid), sg_gamma_periodic
    )[0]
    gamma_periodic = self._phi_transform.transform(gamma_periodic)
    gamma_secular = self._Y_coil * self.Phi_grid.nodes[:, 2]
    self._data["Phi"]["gamma"] = gamma_periodic + gamma_secular
    return self._data
