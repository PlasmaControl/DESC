"""High order accurate source free field solver.

References
----------
    [1] Unalmis et al. New high-order accurate free surface stellarator
        equilibria optimization and boundary integral methods in DESC.
"""

from desc.basis import DoubleFourierSeries
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.integrals.singularities import get_interpolator
from desc.utils import errorif, setdefault


class SourceFreeField(FourierRZToroidalSurface):
    """Compute source free magnetic fields.

    Let 𝒳 be an open set with smooth closed boundary ∂𝒳.
    Computes the magnetic field B in units of Tesla such that

    -                   ∆Φ(x) = 0    x ∈ 𝒳
    -        (B - ∇Φ - B₀)(x) = 0    x ∈ 𝒳
    - <n,B>(x) = <n,∇Φ+B₀>(x) = 0    x ∈ ∂𝒳
    -         ∇ × (B - B₀)(x) = 0    x ∉ ∂𝒳
    -                <∇,B>(x) = 0    ∀x

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂𝒳.
    grid : Grid
        Points on ∂𝒳 for quadrature and interpolation
        used to solve for the potential.
    M : int
        Poloidal Fourier resolution to interpolate potential on ∂𝒳.
        Default is ``grid.M``.
    N : int
        Toroidal Fourier resolution to interpolate potential on ∂𝒳.
        Default is ``grid.N``.
    sym : str
        Symmetry for Fourier basis interpolating the periodic part of the potential.
        Default assumes no symmetry.
    I : float
        Net toroidal current parameter.
        Default is zero.
    Y : float
        Net poloidal current parameter.
        Default is zero.

    """

    _immediate_attributes_ = ["_surface", "_grid", "_Phi_basis", "I", "Y"]

    def __init__(
        self,
        surface,
        grid=None,
        M=None,
        N=None,
        sym=False,
        I=0,  # noqa: E741
        Y=0,
        **kwargs,
    ):
        self._surface = surface
        if grid is None:
            grid = LinearGrid(
                M=surface.M * 4,
                N=surface.N * 4,
                NFP=surface.NFP if surface.N > 0 else 64,
                # TODO(for reviewers) set this based off surface or Phi sym?
                sym=False,
            )
        errorif(not grid.can_fft2)
        self._grid = grid
        M = setdefault(M, grid.M)
        N = setdefault(N, grid.N)
        errorif(M > grid.M, msg=f"Got M={M} > {grid.M}=grid.M.")
        errorif(N > grid.N, msg=f"Got N={N} > {grid.N}=grid.N.")
        self._Phi_basis = DoubleFourierSeries(M=M, N=N, NFP=surface.NFP, sym=sym)
        self.I = I
        self.Y = Y

    def __getattr__(self, attr):
        return getattr(self._surface, attr)

    def __setattr__(self, name, value):
        if name in SourceFreeField._immediate_attributes_:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_surface"), name, value)

    def __hasattr__(self, attr):
        return hasattr(self, attr) or hasattr(self._surface, attr)

    @property
    def surface(self):
        """Surface geometry defining boundary."""
        return self._surface

    @property
    def grid(self):
        """Points on boundary for quadrature and interpolation."""
        return self._grid

    @property
    def Phi_basis(self):
        """DoubleFourierSeries: Basis for periodic part of potential."""
        return self._Phi_basis

    @property
    def sym_Phi(self):
        """str: Type of symmetry of periodic part of Phi (no symmetry if False)."""
        return self._Phi_basis.sym

    @property
    def M_Phi(self):
        """int: Poloidal resolution of periodic part of Phi."""
        return self._Phi_basis.M

    @property
    def N_Phi(self):
        """int: Toroidal resolution of periodic part of Phi."""
        return self._Phi_basis.N

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        data=None,
        override_grid=True,
        *,
        RpZ_coords=None,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        RpZ_coords : dict[str, jnp.ndarray]
            (R, ϕ, Z) coordinates at which to evaluate
            quantities that support evaluation off ``self.grid``.
            Should store the three entries ``"R"``, ``"phi"``, and ``"Z"``.
            The precise behavior is stated for each quantity
            in the public documentation for the list of variables.
        grid : Grid
            Optional grid of coordinates at which to evaluate
            quantities that support evaluation off ``self.grid``.
            The precise behavior is stated for each quantity
            in the public documentation for the list of variables.
            If given, then ``RpZ_coords`` may be ignored.
            If ``grid`` is not given and ``RpZ_coords`` is not given,
            then ``grid`` will default to ``self.grid``.
        params : dict[str, jnp.ndarray]
            Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
            Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from ``self.grid``.
        data : dict[str, jnp.ndarray]
            Data computed so far, generally output from other compute functions.
            Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
            v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
            of the cylindrical coordinates R, ϕ, Z.
        override_grid : bool
            If True, override ``self.grid`` if necessary and use a full
            resolution grid to compute quantities and then downsample to ``self.grid``.
            If False, uses only ``self.grid``, which may lead to
            inaccurate values for surface or volume averages.

        Returns
        -------
        data : dict[str, jnp.ndarray]
            Computed quantities and intermediate variables on ``self.grid``.
            May also store data evaluated on ``RpZ_coords`` or ``grid``
            for those quantities that support evaluation off ``self.grid``.
            The precise behavior is stated for each quantity
            in the public documentation for the list of variables.

        """
        if grid is not None and "eval_interpolator" not in kwargs:
            warn_fft = kwargs.pop("warn_fft", True)
            kwargs["eval_interpolator"] = get_interpolator(
                eval_grid=grid,
                source_grid=self._grid,
                source_data=super().compute(
                    ["|e_theta x e_zeta|", "e_theta", "e_zeta"],
                    self._grid,
                    params,
                    transforms,
                    # Do not pass in data since the expectation
                    # is that quantities computed on self.grid.
                    # whereas input data may have things on eval
                    # grid as well.
                    override_grid=override_grid,
                    **kwargs,
                ),
                warn_fft=warn_fft,
                **kwargs,
            )
        if RpZ_coords is None and grid is not None and not self._grid.equiv(grid):
            # then user forgot to supply RpZ coords for evaluation grid
            RpZ = super().compute(
                ["R", "phi", "Z"],
                grid,
                params,
                transforms,
                override_grid=override_grid,
                **kwargs,
            )
            RpZ_coords = {"R": RpZ["R"], "phi": RpZ["phi"], "Z": RpZ["Z"]}
        if RpZ_coords is not None:
            kwargs["RpZ_coords"] = RpZ_coords
        if "B0" in kwargs:
            kwargs.setdefault("surface", self._surface)

        return super().compute(
            names, self._grid, params, transforms, data, override_grid, **kwargs
        )


class FreeSurfaceOuterField(SourceFreeField):
    """Compute field on outer plasma for free surface.

    Implements the interior Dirichlet formulation described in [1].

    Parameters
    ----------
    Y_coil : float
        Net poloidal current in coils parameter.
        Default is None.

    """

    _immediate_attributes_ = ["Y_coil"]

    def __init__(
        self,
        surface,
        grid=None,
        M=None,
        N=None,
        sym=False,
        I=0,  # noqa: E741
        Y=0,
        Y_coil=None,
        **kwargs,
    ):
        super().__init__(surface, grid, M, N, sym, I, Y, **kwargs)
        self.Y_coil = Y_coil

    def __setattr__(self, name, value):
        if (
            name in FreeSurfaceOuterField._immediate_attributes_
            or name in SourceFreeField._immediate_attributes_
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_surface"), name, value)

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        data=None,
        override_grid=True,
        *,
        RpZ_coords=None,
        **kwargs,
    ):
        """Compute the quantity given by name on grid."""
        if self.Y_coil is not None and "Y_coil" not in data:
            data["Y_coil"] = self.Y_coil
        return super().compute(
            names,
            grid,
            params,
            transforms,
            data,
            override_grid,
            RpZ_coords=RpZ_coords,
            **kwargs,
        )
