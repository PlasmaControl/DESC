from functools import partial

from desc.backend import fixed_point, irfft2, jit, jnp, rfft2
from desc.basis import DoubleFourierSeries
from desc.batching import vmap_chunked
from desc.grid import LinearGrid
from desc.integrals._vacuum import _H
from desc.integrals.singularities import (
    _dx,
    _kernel_biot_savart,
    _kernel_biot_savart_A,
    get_interpolator,
    singular_integral,
)
from desc.io import IOAble
from desc.transform import Transform
from desc.utils import cross, dot, errorif, setdefault, warnif


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

    𝐁ᵥ(𝐫) = μ₀/4π ∫ (𝐧' ⋅ 𝐁(𝐫')) * (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + μ₀/4π ∫ (𝐧' × 𝐁(𝐫')) × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + 𝐁(𝐫)/2

    Where 𝐁 is the total field on the surface and 𝐧' is the outward surface normal.
    Because the total field is tangent, the first term in the integrand is zero leaving

    𝐁ᵥ(𝐫) = μ₀/4π ∫ K_vc(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫' + 𝐁(𝐫)/2

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


class FreeBoundarySolver(IOAble):
    """Compute exterior field for free boundary problem.

    See shared article for detailed description.

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂𝒳.
    B_coil : _MagneticField
        Magnetic field produced by coils.
    evl_grid : Grid
        Evaluation points on ∂𝒳 for the magnetic field.
    src_grid : Grid
        Source points on ∂𝒳 for quadrature of kernels.
        Default resolution is ``src_grid.M=surface.M*4`` and ``src_grid.N=surface.N*4``.
    Phi_grid : Grid
        Interpolation points on ∂𝒳.
        Resolution determines accuracy of Φ interpolation.
        Default resolution is ``Phi_grid.M=surface.M*2`` and ``Phi_grid.N=surface.N*2``.
    Phi_M : int
        Poloidal Fourier resolution to interpolate Φ on ∂𝒳.
        Should be at most ``Phi_grid.M``.
    Phi_N : int
        Toroidal Fourier resolution to interpolate Φ on ∂𝒳.
        Should be at most ``Phi_grid.N``.
    sym
        Symmetry for basis which interpolates Φ.
        Default assumes no symmetry.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    use_dft : bool
        Whether to use matrix multiplication transform from spectral to physical domain
        instead of inverse fast Fourier transform.

    """

    def __init__(
        self,
        surface,
        B_coil,
        evl_grid,
        src_grid=None,
        Phi_grid=None,
        Phi_M=None,
        Phi_N=None,
        sym=None,
        *,
        chunk_size=None,
        use_dft=False,
        **kwargs,
    ):
        errorif(
            evl_grid.nodes[0, 0] < evl_grid.num_rho,
            msg="Evaluation grid must be on boundary.",
        )
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
            sym=setdefault(sym, False) and surface.sym,
        )

        # Compute data on source grid.
        geometric_names = ["x", "n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"]
        src_data = surface.compute(geometric_names, grid=src_grid)
        # Compute data on Phi grid.
        if self._same_grid_phi_src:
            Phi_data = src_data
        else:
            Phi_data = surface.compute(geometric_names, grid=Phi_grid)
        self._phi_transform = Transform(Phi_grid, basis, derivs=1, build_pinv=True)
        Phi_data["n x B_coil"] = cross(
            Phi_data["n_rho"],
            B_coil.compute_magnetic_field(
                coords=Phi_data["x"], source_grid=src_grid, chunk_size=chunk_size
            ),
        )
        # Compute data on evaluation grid.
        if evl_grid.equiv(Phi_grid):
            evl_data = Phi_data
            self._evl_transform = self._phi_transform
        else:
            self._evl_transform = Transform(evl_grid, basis, derivs=1)
            if not self._same_grid_phi_src and evl_grid.equiv(src_grid):
                evl_data = src_data
            else:
                evl_data = surface.compute(geometric_names, grid=evl_grid)

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
        return self._phi_transform.basis

    def compute_Phi(
        self, chunk_size=None, maxiter=0, tol=1e-6, method="del2", Phi_0=None, **kwargs
    ):
        """Compute Fourier coefficients of vacuum potential Φ on ∂𝒳.

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
             Fourier coefficients of Φ on ∂𝒳 stored in ``data["Phi"]["Phi_mn"]``.

        """
        warnif(kwargs.get("warn", True) and (maxiter > 0), msg="Still debugging")
        self._data = (
            _fixed_point_Phi(
                self,
                Phi_0,
                tol=tol,
                maxiter=maxiter,
                method=method,
                chunk_size=chunk_size,
            )
            if (maxiter > 0)
            else _lsmr_Phi(self, chunk_size=chunk_size)
        )
        return self._data

    def compute_B_out(self):
        """Compute ‖B_out‖² on the evaluation grid.

        Returns
        -------
        data : dict
            ‖B_out‖² stored in ``data["evl"]["|B_out|^2"]``.

        """
        if "|B_out|^2" in self._data["evl"]:
            return self._data
        B_out_tan = _surface_gradient(
            self._data["evl"], self._evl_transform, self._data["Phi"]["Phi_mn"]
        )
        self._data["evl"]["|B_out|^2"] = dot(B_out_tan, B_out_tan)
        return self._data


@partial(vmap_chunked, in_axes=(None, None, 0), chunk_size=None)
def _surface_gradient(data, transform, c):
    # TODO (#1531): Can make this more efficient O(N) instead of O(N^2).
    assert c.size == transform.basis.num_modes
    f_t = transform.transform(c, dt=1)[:, jnp.newaxis]
    f_z = transform.transform(c, dz=1)[:, jnp.newaxis]
    return (f_t * data["e_zeta"] - f_z * data["e_theta"]) / data["|e_theta x e_zeta|"][
        :, jnp.newaxis
    ]


def _boundary_condition(self):
    """Returns γ = (n × ∇)⁻¹ (n × B_coil)."""
    if "gamma" in self._data["Phi"]:
        return self._data

    A = _surface_gradient(
        self._data["Phi"], self._phi_transform, jnp.eye(self.basis.num_modes)
    ).T
    # check dims here
    assert A.shape == (3, self.Phi_grid.num_nodes, self.basis.num_modes)
    gamma_mn = (
        jnp.linalg.solve(A, self._data["Phi"]["n x B_coil"].T)
        if (self.Phi_grid.num_nodes == self.basis.num_modes)
        else jnp.linalg.lstsq(A, self._data["Phi"]["n x B_coil"].T)[0]
    )
    self._data["Phi"]["gamma"] = self._phi_transform.transform(gamma_mn)

    return self._data


@partial(jit, static_argnames=["chunk_size"])
def _lsmr_Phi(self, basis=None, *, chunk_size=None):
    """Compute Fourier harmonics Φ̃ by solving least squares system."""
    if "Phi_mn" in self._data["Phi"]:
        return self._data

    self._data = _boundary_condition(self)
    gamma = self._data["Phi"]["gamma"]

    basis = setdefault(basis, self.basis)
    evl_Phi = basis.evaluate(self.Phi_grid)
    src_data = self._data["src"].copy()
    src_data["Phi"] = (
        evl_Phi if self._same_grid_phi_src else basis.evaluate(self.src_grid)
    )
    A = evl_Phi / 2 + _H(self, src_data, chunk_size, basis)
    assert A.shape == (self.Phi_grid.num_nodes, basis.num_modes)

    # Solving overdetermined system useful to reduce size of A while
    # retaining FFT interpolation accuracy in the singular integrals.
    self._data["Phi"]["Phi_mn"] = (
        jnp.linalg.solve(A, gamma)
        if (self.Phi_grid.num_nodes == basis.num_modes)
        else jnp.linalg.lstsq(A, gamma)[0]
    )
    return self._data


def _iteration_operator(Phi_k, self, chunk_size=None):
    """Compute iteration operator T(Φ).

    Parameters
    ----------
    Phi_k : jnp.ndarray
        Φ values on ``self._Phi_grid``.

    Returns
    -------
    Phi_k+1 : jnp.ndarray
        Fredholm integral operator computed on ``self._Phi_grid``.

    """
    src_data = self._data["src"].copy()
    src_data["Phi"] = self._upsample_to_source(Phi_k, is_fourier=False)
    gamma = self._data["Phi"]["gamma"]
    H = _H(self, src_data, chunk_size).squeeze(axis=-1)
    return -H + 0.5 * Phi_k + gamma


@partial(jit, static_argnames=["tol", "maxiter", "method", "chunk_size"])
def _fixed_point_Phi(
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

    self._data = _boundary_condition(self)

    if Phi_0 is None:
        basis = DoubleFourierSeries(
            M=min(self.basis.M, 3),
            N=min(self.basis.N, 3),
            NFP=self.basis.NFP,
            sym=self.basis.sym,
        )
        self._data = _lsmr_Phi(self, basis, chunk_size=chunk_size)
        Phi_0 = basis.evaluate(self.Phi_grid) @ self._data["Phi"]["Phi_mn"]

    Phi = fixed_point(
        _iteration_operator,
        Phi_0,
        (self, chunk_size),
        tol,
        maxiter,
        method,
        scalar=True,
    )

    self._data["Phi"]["Phi_mn"] = self._phi_transform.fit(Phi)
    return self._data
