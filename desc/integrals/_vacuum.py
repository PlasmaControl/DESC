"""High order accurate vacuum field solver."""

from functools import partial

from matplotlib import pyplot as plt

from desc.backend import fixed_point, irfft2, jit, jnp, rfft2
from desc.basis import DoubleFourierSeries
from desc.grid import LinearGrid, _Grid
from desc.integrals.singularities import (
    _kernel_biot_savart_coulomb,
    _kernel_dipole,
    _kernel_dipole_smooth,
    _kernel_monopole,
    _nonsingular_part,
    get_interpolator,
    singular_integral,
)
from desc.io import IOAble
from desc.transform import Transform
from desc.utils import errorif, setdefault, warnif


class VacuumSolver(IOAble):
    """Compute vacuum field that satisfies LCFS boundary condition.

    Let 𝒳 be an open set with smooth closed boundary ∂𝒳.
    Computes the magnetic field B in units of Tesla such that

    -                   ∆Φ(x) = 0    x ∈ 𝒳
    -        (B - ∇Φ - B₀)(x) = 0    x ∈ 𝒳
    - <n,B>(x) = <n,∇Φ+B₀>(x) = 0    x ∈ ∂𝒳
    -         ∇ × (B - B₀)(x) = 0    x ∉ ∂𝒳
    -                <∇,B>(x) = 0    ∀x

    References
    ----------
       [1] Unalmis et al. New high-order accurate free surface stellarator
           equilibria optimization and boundary integral methods in DESC.

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂𝒳.
    evl_grid : Grid or jnp.ndarray
        Evaluation points on ∂𝒳.
        Or ``coords`` array of evaluation points in 𝒳 in R,phi,Z coords.
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
    exterior : bool
        Whether to solve the exterior Neumann problem instead of the interior.
        If true, then 𝒳 is exterior of plasma.
        Default is false.
    I : float
        Poloidal secular coefficient for Φ.
    Y : float
        Toroidal secular coefficient for Φ.
    B0 : _MagneticField
        Assumes that ``B0.compute_Bnormal()`` computes <n,B₀> with the
        normal that points to the exterior.
    B0n : jnp.ndarray
        Optional, <n,B₀> on ``src_grid``.
        Assumes <n,B₀> is computed with the
        normal that points to the exterior.
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
        exterior=False,
        I=0,
        Y=0,
        B0=None,
        B0n=None,
        chunk_size=None,
        use_dft=False,
        **kwargs,
    ):
        warnif(I != 0 or Y != 0, msg="This option needs more testing.")
        self._exterior = bool(exterior)
        self._I = I
        self._Y = Y
        self._B0 = B0

        self._evl_grid = evl_grid
        if src_grid is None:
            src_grid = LinearGrid(
                M=surface.M * 4,
                N=surface.N * 4,
                NFP=surface.NFP if surface.N > 0 else 64,
                sym=False,
            )
        if Phi_grid is None:
            Phi_grid = src_grid
            self._src_grid_equals_Phi_grid = True
        else:
            self._src_grid_equals_Phi_grid = src_grid.equiv(Phi_grid)

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
        self._src_transform = Transform(src_grid, basis, derivs=1)
        src_data = surface.compute(
            [
                "x",
                "n_rho",
                "|e_theta x e_zeta|",
                "e_theta",
                "e_zeta",
                "n_rho x grad(theta)",
                "n_rho x grad(zeta)",
            ],
            grid=src_grid,
        )
        if B0n is not None:
            src_data["Bn"] = B0n
            self._B0 = True  # Mark it as not None to avoid redundant computation later.
        elif B0 is not None:
            src_data["Bn"] = B0.compute_Bnormal(
                surface,
                eval_grid=src_grid,
                source_grid=src_grid,
                vc_source_grid=src_grid,
                chunk_size=chunk_size,
            )[0]
        if self._exterior and "Bn" in src_data:
            src_data["Bn"] = -src_data["Bn"]

        # Compute data on Phi grid.
        if self._src_grid_equals_Phi_grid:
            Phi_data = src_data
            self._src_transform.build_pinv()
            self._phi_transform = self._src_transform
        else:
            Phi_data = surface.compute("x", grid=Phi_grid)
            self._phi_transform = Transform(
                Phi_grid, basis, build=False, build_pinv=True
            )

        # Compute data on evaluation grid.
        if self._evaluate_in_X():
            R, phi, Z = evl_grid.T
            evl_data = {"R": R, "phi": phi, "Z": Z, "x": evl_grid}
        else:
            if evl_grid.equiv(Phi_grid):
                evl_data = Phi_data
            elif not self._src_grid_equals_Phi_grid and evl_grid.equiv(src_grid):
                evl_data = src_data
            else:
                evl_data = surface.compute("x", grid=evl_grid)

        self._data = {"evl": evl_data, "Phi": Phi_data, "src": src_data}
        self._data_evl_old = self._data["evl"].copy()
        self._interpolator = {
            "Phi": get_interpolator(Phi_grid, src_grid, src_data, use_dft, **kwargs),
        }
        if not self._evaluate_in_X():
            self._interpolator["evl"] = get_interpolator(
                evl_grid, src_grid, src_data, use_dft, **kwargs
            )

    def _evaluate_in_X(self, coords=None):
        # if coords was supplied assume in X else check if eval grid was
        # originally some coords array.
        return coords is not None or not isinstance(self.evl_grid, _Grid)

    @property
    def evl_grid(self):
        """Return the evaluation grid used by this solver."""
        return self._evl_grid

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
        return self._src_transform.basis

    def compute_Phi(
        self,
        chunk_size=None,
        maxiter=0,
        tol=1e-6,
        method="simple",
        Phi_0=None,
        **kwargs,
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
            Method of finding the fixed-point, defaults to ``simple``.
        Phi_0 : jnp.ndarray
            Initial guess for Φ on ``self.Phi_grid`` for iteration.
            In general, it is best to select the initial guess as truncated
            Fourier series. Default is a fit to a low resolution solution.

        Returns
        -------
        data : dict
             Fourier coefficients of Φ on ∂𝒳 stored in ``data["Phi"]["Phi_mn"]``.

        """
        errorif(
            (not self._exterior) and (maxiter > 0),
            msg="Fixed point method will not convergence for interior problem.",
        )
        warnif(kwargs.get("warn", True) and (maxiter > 0), msg="This is experimental.")
        self._data = (
            _fixed_point_Phi(
                self,
                bc=_vacuum_bc,
                tol=tol,
                maxiter=maxiter,
                method=method,
                chunk_size=chunk_size,
                Phi_0=Phi_0,
            )
            if (maxiter > 0)
            else _lsmr_Phi_smoother(self, bc=_vacuum_bc, chunk_size=chunk_size)
        )
        return self._data

    def _compute_virtual_current(self):
        """𝐊_vc = -n × ∇Φ for interior and n × ∇Φ for exterior.

        This is the vacuum portion of the virtual surface current.
        """
        if "K_vc" in self._data["src"]:
            return self._data

        data = self._data["src"]
        data["K_vc"] = -_surface_gradient(
            data,
            self._src_transform,
            self._data["Phi"]["Phi_mn"],
            self._I,
            self._Y,
        )
        if self._exterior:
            data["K_vc"] = -data["K_vc"]
        return self._data

    def _set_new_evl_coords(self, coords):
        R, phi, Z = coords.T
        self._data["evl"]["R"] = R
        self._data["evl"]["phi"] = phi
        self._data["evl"]["Z"] = Z
        self._data["evl"]["x"] = coords

    def _set_old_evl_coords(self):
        self._data["evl"]["R"] = self._data_evl_old["R"]
        self._data["evl"]["phi"] = self._data_evl_old["phi"]
        self._data["evl"]["Z"] = self._data_evl_old["Z"]
        self._data["evl"]["x"] = self._data_evl_old["x"]

    def compute_vacuum_field(self, chunk_size=None, coords=None):
        """Compute magnetic field due to vacuum potential Φ.

        Parameters
        ----------
        chunk_size : int or None
            Size to split computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.
        coords : jnp.ndarray
            Optional, evaluation points in 𝒳 in coordinates of R, phi, Z basis.

        Returns
        -------
        data : dict
             Vacuum field ∇Φ stored in ``data["evl"]["grad(Phi)"]``.

        """
        if coords is not None:
            self._set_new_evl_coords(coords)
        elif "grad(Phi)" in self._data["evl"]:
            return self._data

        self._data = self.compute_Phi(chunk_size)
        self._data = self._compute_virtual_current()

        if self._evaluate_in_X(coords):
            self._data["evl"]["grad(Phi)"] = _nonsingular_part(
                self._data["evl"],
                None,
                self._data["src"],
                self.src_grid,
                st=jnp.nan,
                sz=jnp.nan,
                kernel=_kernel_biot_savart_coulomb,
                chunk_size=chunk_size,
            )
        else:
            self._data["evl"]["grad(Phi)"] = 2 * singular_integral(
                self._data["evl"],
                self._data["src"],
                interpolator=self._interpolator["evl"],
                kernel=_kernel_biot_savart_coulomb,
                chunk_size=chunk_size,
            )
        if coords is not None:
            self._set_old_evl_coords()
        return self._data

    def compute_current_field(self, chunk_size=None, coords=None):
        """Compute magnetic field B₀ due to volume current sources.

        Parameters
        ----------
        chunk_size : int or None
            Size to split computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.
        coords : jnp.ndarray
            Optional, evaluation points in 𝒳 in coordinates of R,phi,Z basis.

        Returns
        -------
        data : dict
            B₀ stored in ``data["evl"]["B0"]``.

        """
        if coords is not None:
            self._set_new_evl_coords(coords)
        elif "B0" in self._data["evl"]:
            return self._data

        data = self._data["evl"]

        data["B0"] = (
            0
            if self._B0 is None
            else self._B0.compute_magnetic_field(
                coords=data["x"], source_grid=self.src_grid, chunk_size=chunk_size
            )
        )
        if coords is not None:
            self._set_old_evl_coords()
        return self._data

    def compute_magnetic_field(self, chunk_size=None, coords=None):
        """Compute magnetic field B = B₀ + ∇Φ.

        Parameters
        ----------
        chunk_size : int or None
            Size to split computation into chunks.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.
        coords : jnp.ndarray
            Optional, evaluation points in 𝒳 in coordinates of R,phi,Z basis.

        Returns
        -------
        data : dict
            B stored in ``data["evl"]["B0+grad(Phi)"]``.

        """
        if coords is not None:
            self._set_new_evl_coords(coords)
        elif "B0+grad(Phi)" in self._data["evl"]:
            return self._data

        self._data = self.compute_current_field(chunk_size)
        self._data = self.compute_vacuum_field(chunk_size)
        self._data["evl"]["B0+grad(Phi)"] = (
            self._data["evl"]["B0"] + self._data["evl"]["grad(Phi)"]
        )
        if coords is not None:
            self._set_old_evl_coords()
        return self._data

    def plot_Bn_error(self, Bn):
        """Plot 𝐁 ⋅ 𝐧 error on ∂𝒳.

        Parameters
        ----------
        Bn : jnp.ndarray
            𝐁 ⋅ 𝐧 on the evaluation grid.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        errorif(self._evaluate_in_X())
        grid = self.evl_grid
        theta = grid.meshgrid_reshape(grid.nodes[:, 1], "rtz")[0]
        zeta = grid.meshgrid_reshape(grid.nodes[:, 2], "rtz")[0]
        Bn = grid.meshgrid_reshape(Bn, "rtz")[0]

        fig, ax = plt.subplots()
        contour = ax.contourf(theta, zeta, Bn)
        fig.colorbar(contour, ax=ax)
        ax.set_title(r"$B \cdot n$ on $\partial X$")
        return fig, ax


def _surface_gradient(data, transform, c, I=0, Y=0):
    assert c.size == transform.basis.num_modes
    f_t = (transform.transform(c, dt=1) + I)[:, jnp.newaxis]
    f_z = (transform.transform(c, dz=1) + Y)[:, jnp.newaxis]
    return f_t * data["n_rho x grad(theta)"] + f_z * data["n_rho x grad(zeta)"]


def _vacuum_bc(self, chunk_size):
    """Returns gamma = S[<B₀,n>] for interior and -S[<B₀,n>] for exterior."""
    if "gamma" in self._data["Phi"]:
        return self._data

    self._data["Phi"]["gamma"] = (
        0
        if self._B0 is None
        else singular_integral(
            self._data["Phi"],
            self._data["src"],
            interpolator=self._interpolator["Phi"],
            kernel=_kernel_monopole,
            chunk_size=chunk_size,
        ).squeeze(axis=-1)
    )
    return self._data


def _double_layer(self, src_data, chunk_size, basis=None):
    """Compute D[Φ](x) = ∫_y Φ(y)〈∇_x G(x−y),ds(y)〉or, if basis is supplied, D[Φ₁].

    If ``basis`` is not supplied, then computes D[Φ].
    If ``basis`` is supplied, then computes D[Φ₁] = ℱ⁻¹(ℱD)(ℱΦ₁)
    where ℱ Φ₁ = [1, ..., 1] are the coefficients of the given orthogonal
    basis that interpolates Φ₁(x) = ∑ₘ (ℱΦ₁)ᵐ fₘ(x) = ∑ₘ fₘ(x) for x ∈ ∂𝒳.

    Parameters
    ----------
    basis : DoubleFourierSeries
        Optional. If supplied changes the meaning of the output. See note.

    """
    kwargs = {}
    if basis is not None:
        kwargs["known_map"] = ("Phi", partial(basis.evaluate, secular=True))
        kwargs["ndim"] = basis.num_modes + 2
    D = singular_integral(
        eval_data=self._data["Phi"],
        source_data=src_data,
        interpolator=self._interpolator["Phi"],
        kernel=_kernel_dipole,
        chunk_size=chunk_size,
        **kwargs,
    )
    return -D if self._exterior else D


def _double_layer_smoother(self, src_data, chunk_size, basis=None):
    """Compute (D[Φ] - D[1]Φ)(x) = ∫_y (Φ(y)-Φ(x))〈∇_x G(x−y),ds(y).

    Parameters
    ----------
    basis : DoubleFourierSeries
        Optional. If supplied changes the meaning of the output. See note.

    """
    kwargs = {}
    if basis is not None:
        kwargs["known_map"] = ("Phi", partial(basis.evaluate, secular=True))
        kwargs["ndim"] = basis.num_modes + 2
    D = singular_integral(
        eval_data=self._data["Phi"],
        source_data=src_data,
        interpolator=self._interpolator["Phi"],
        kernel=_kernel_dipole_smooth,
        chunk_size=chunk_size,
        **kwargs,
    )
    if self._exterior:
        D = -D
    return D


@partial(jit, static_argnames=["bc", "chunk_size", "interior_dirichlet"])
def _lsmr_Phi_smoother(
    self,
    *,
    bc,
    chunk_size=None,
    interior_dirichlet=False,
):
    """Compute Fourier harmonics Φ̃ by solving least squares system.

    Parameters
    ----------
    self : VacuumSolver or FreeBoundarySolver
    bc : callable
        Signature should match bc(self, chunk_size)
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    data : dict[str, jnp.ndarray]
        Returns ``self._data`` with ``Phi_mn`` in ``self._data["Phi"]``.

    """
    if "Phi_mn" in self._data["Phi"]:
        return self._data

    self._data = bc(self, chunk_size)
    gamma = self._data["Phi"]["gamma"]

    evl_Phi = self.basis.evaluate(self.Phi_grid, secular=True)
    self._data["Phi"]["evl_Phi"] = evl_Phi
    src_data = self._data["src"].copy()
    src_data["Phi"] = (
        evl_Phi
        if self._src_grid_equals_Phi_grid
        else self.basis.evaluate(self.src_grid, secular=True)
    )
    A = _double_layer_smoother(self, src_data, chunk_size, self.basis)
    if interior_dirichlet:
        A = A - evl_Phi
    elif self._exterior:
        A = A + evl_Phi
    assert A.shape == (self.Phi_grid.num_nodes, self.basis.num_modes + 2)

    # TODO: check this with our new sign conventions
    gamma = gamma - (self.I * A[:, -2] + self.Y * A[:, -1])
    A = A[:, :-2]
    # Solving overdetermined system useful to reduce size of A while
    # retaining FFT interpolation accuracy in the singular integrals.
    # TODO: https://github.com/patrick-kidger/lineax/pull/86
    self._data["Phi"]["Phi_mn"] = (
        jnp.linalg.solve(A, gamma)
        if (self.Phi_grid.num_nodes == self.basis.num_modes)
        else jnp.linalg.lstsq(A, gamma)[0]
    )
    return self._data


@partial(jit, static_argnames=["bc", "chunk_size", "interior_dirichlet"])
def _lsmr_Phi(
    self,
    *,
    bc,
    chunk_size=None,
    interior_dirichlet=False,
):
    """Compute Fourier harmonics Φ̃ by solving least squares system.

    Parameters
    ----------
    self : VacuumSolver or FreeBoundarySolver
    bc : callable
        Signature should match bc(self, chunk_size)
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    data : dict[str, jnp.ndarray]
        Returns ``self._data`` with ``Phi_mn`` in ``self._data["Phi"]``.

    """
    if "Phi_mn" in self._data["Phi"]:
        return self._data

    self._data = bc(self, chunk_size)
    gamma = self._data["Phi"]["gamma"]

    evl_Phi = self.basis.evaluate(self.Phi_grid, secular=True)
    src_data = self._data["src"].copy()
    src_data["Phi"] = (
        evl_Phi
        if self._src_grid_equals_Phi_grid
        else self.basis.evaluate(self.src_grid, secular=True)
    )
    A = (-evl_Phi if interior_dirichlet else evl_Phi) / 2 + _double_layer(
        self, src_data, chunk_size, self.basis
    )
    assert A.shape == (self.Phi_grid.num_nodes, self.basis.num_modes + 2)

    # TODO: check this with our new sign conventions
    gamma = gamma - (self.I * A[:, -2] + self.Y * A[:, -1])
    A = A[:, :-2]
    # Solving overdetermined system useful to reduce size of A while
    # retaining FFT interpolation accuracy in the singular integrals.
    # TODO: https://github.com/patrick-kidger/lineax/pull/86
    self._data["Phi"]["Phi_mn"] = (
        jnp.linalg.solve(A, gamma)
        if (self.Phi_grid.num_nodes == self.basis.num_modes)
        else jnp.linalg.lstsq(A, gamma)[0]
    )
    return self._data


def _iteration_operator(Phi_k, self, chunk_size=None, interior_dirichlet=False):
    """Compute iteration operator T(Φ).

    Parameters
    ----------
    Phi_k : jnp.ndarray
        Φ values on ``self._Phi_grid``.
    self : VacuumSolver or FreeBoundarySolver
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    Phi_k+1 : jnp.ndarray
        Iteration operator applied to input.

    """
    src_data = self._data["src"].copy()
    src_data["Phi"] = _upsample_to_source(self, Phi_k, is_fourier=False)
    gamma = self._data["Phi"]["gamma"]
    D = _double_layer(self, src_data, chunk_size).squeeze(axis=-1)
    if interior_dirichlet:
        gamma = -gamma
    else:
        # doing exterior Neumann and D was negated already
        D = -D
    return D + 0.5 * Phi_k + gamma


@partial(
    jit,
    static_argnames=[
        "bc",
        "tol",
        "maxiter",
        "method",
        "chunk_size",
        "interior_dirichlet",
    ],
)
def _fixed_point_Phi(
    self,
    *,
    bc,
    tol=1e-6,
    maxiter=20,
    method="del2",
    chunk_size=None,
    Phi_0=None,
    interior_dirichlet=False,
):
    assert self.Phi_grid.can_fft2
    if "Phi_mn" in self._data["Phi"]:
        return self._data

    self._data = bc(self, chunk_size)

    if Phi_0 is None:
        basis = DoubleFourierSeries(
            M=min(self.basis.M, 3),
            N=min(self.basis.N, 3),
            NFP=self.basis.NFP,
            sym=self.basis.sym,
        )
        self._data = _lsmr_Phi(self, bc=bc, basis=basis, chunk_size=chunk_size)
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


def _upsample_to_source(self, x, is_fourier=False):
    if not self._src_grid_equals_Phi_grid:
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
