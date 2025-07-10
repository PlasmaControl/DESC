"""High order accurate source free field solver.

References
----------
    [1] Unalmis et al. New high-order accurate free surface stellarator
        equilibria optimization and boundary integral methods in DESC.
"""

from desc.basis import DoubleFourierSeries
from desc.geometry import FourierRZToroidalSurface
from desc.integrals.singularities import get_interpolator
from desc.magnetic_fields import ToroidalMagneticField
from desc.utils import errorif


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
    B0 : _MagneticField
        Magnetic field due to currents in 𝒳 and net currents outside 𝒳
        which are not accounted for ``I`` and ``Y``.

    """

    _immediate_attributes_ = ["_surface", "_Phi_basis", "I", "Y", "_B0"]

    def __init__(
        self,
        surface,
        M,
        N,
        sym=False,
        I=0,  # noqa: E741
        Y=0,
        B0=ToroidalMagneticField(0, 1),
    ):
        self._surface = surface
        self._Phi_basis = DoubleFourierSeries(M=M, N=N, NFP=surface.NFP, sym=sym)
        self.I = I
        self.Y = Y
        self._B0 = B0

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
        grid,
        params=None,
        transforms=None,
        data=None,
        RpZ_data=None,
        RpZ_grid=None,
        override_grid=True,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid
            Grid of coordinates on which to perform computation.
        params : dict[str, jnp.ndarray]
            Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
            Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from ``grid``.
        data : dict[str, jnp.ndarray]
            Data computed so far, generally output from other compute functions.
            Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
            v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
            of the cylindrical coordinates R, ϕ, Z.
        RpZ_data : dict[str, jnp.ndarray]
            Data evaluated so far on the (R, ϕ, Z) coordinates in this dictionary.
            Should store the three entries ``"R"``, ``"phi"``, and ``"Z"``
            if the intention is to compute something at these coordinates.
            If not given, then computes from ``RpZ_grid``.
        RpZ_grid : Grid
            Grid of coordinates on which to evaluate quantities that support
            evaluation off of ``grid``.
            If not given, then default is ``grid``.
        override_grid : bool
            If True, override ``self.grid`` if necessary and use a full
            resolution grid to compute quantities and then downsample to ``self.grid``.
            If False, uses only ``self.grid``, which may lead to
            inaccurate values for surface or volume averages.

        Returns
        -------
        data : dict[str, jnp.ndarray]
            Quantities and intermediate variables computed on ``grid``.
        RpZ_data : dict[str, jnp.ndarray]
            Quantities and intermediate variables computed on the
            (R, ϕ, Z) coordinates in ``RpZ_data``.

        """
        errorif(self.M_Phi > grid.M, msg=f"Got M_Phi={self.M_Phi} > {grid.M}=grid.M.")
        errorif(self.N_Phi > grid.N, msg=f"Got N_Phi={self.N_Phi} > {grid.N}=grid.N.")

        # cludge until all magnetic field classes use new API
        kwargs.setdefault("B0", self._B0)
        kwargs.setdefault("surface", self._surface)

        # to simplify computation of a singular integral for grad(Phi)
        if kwargs.get("on_boundary", False) and "eval_interpolator" not in kwargs:
            if RpZ_grid is None:
                errorif(RpZ_data is not None, msg="Please supply RpZ_grid.")
                # grad(Phi) computation will proceed assuming RpZ_grid is grid.
            else:
                warn_fft = kwargs.pop("warn_fft", True)
                kwargs["eval_interpolator"] = get_interpolator(
                    eval_grid=RpZ_grid,
                    source_grid=grid,
                    source_data=super().compute(
                        ["|e_theta x e_zeta|", "e_theta", "e_zeta"],
                        grid,
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

        if RpZ_data is None:
            if RpZ_grid is None:
                RpZ_grid = grid
                RpZ_data = data
            RpZ_data = super().compute(
                ["R", "phi", "Z"],
                RpZ_grid,
                params,
                transforms,
                data=RpZ_data,
                override_grid=override_grid,
                **kwargs,
            )

        return super().compute(
            names,
            grid,
            params,
            transforms,
            data,
            override_grid,
            RpZ_data=RpZ_data,
            **kwargs,
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

    _immediate_attributes_ = ["_B_coil", "_Y_coil"]

    def __init__(
        self,
        surface,
        B_coil,
        M=None,
        N=None,
        sym=False,
        I=0,  # noqa: E741
        Y=0,
        Y_coil=None,
    ):
        super().__init__(surface, M, N, sym, I, Y)
        self._B_coil
        self._Y_coil = Y_coil

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
        grid,
        params=None,
        transforms=None,
        data=None,
        RpZ_data=None,
        RpZ_grid=None,
        override_grid=True,
        **kwargs,
    ):
        """Compute the quantity given by name on grid."""
        if self._Y_coil is not None and "Y_coil" not in data:
            data["Y_coil"] = self._Y_coil
        kwargs.setdefault("B_coil", self._B_coil)
        return super().compute(
            names,
            grid,
            params,
            transforms,
            data,
            RpZ_data,
            RpZ_grid,
            override_grid,
            **kwargs,
        )
