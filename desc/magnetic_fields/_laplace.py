"""High order accurate multiply connected geometry Laplace solver.

References
----------
    [1] Unalmis et al. New high-order accurate free surface stellarator
        equilibria optimization and boundary integral methods in DESC.

"""

from desc.basis import DoubleFourierSeries
from desc.geometry import FourierRZToroidalSurface
from desc.integrals.singularities import get_interpolator
from desc.magnetic_fields import ToroidalMagneticField
from desc.utils import errorif, setdefault, warnif


class SourceFreeField(FourierRZToroidalSurface):
    """Compute source free magnetic fields.

    Implements the Neumann formulation in multiply connected
    geometry described in [1].

    Let 𝒳 be an open set with continuously differentiable
    closed boundary ∂𝒳. This class solves the following
    partial differential equation for
    varphi = φ = Φ (periodic) = ``Phi (periodic)``.

    -                  ∆φ(x) = 0   x ∈ 𝒳
    -       (B - ∇φ - B₀)(x) = 0   x ∈ 𝒳
    -     n dot (∇φ + B₀)(x) = 0   x ∈ ∂𝒳
    -             n dot B(x) = 0   x ∈ ∂𝒳
    -       curl (B - B₀)(x) = 0   x ∉ ∂𝒳
    -               div B(x) = 0   ∀x

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂𝒳.
    M : int
        Poloidal Fourier resolution to interpolate potential on ∂𝒳.
    N : int
        Toroidal Fourier resolution to interpolate potential on ∂𝒳.
    NFP : int
        Field periodicity of potential on ∂𝒳.
        Default is ``surface.NFP`` which is correct only if
        the globally defined part of ``B0`` produces an NFP periodic
        field.
    sym : str
        Symmetry for Fourier basis interpolating the periodic part of the
        potential. Default is ``False``.
    B0 : _MagneticField
        Magnetic field due to currents in 𝒳 and net currents outside 𝒳
    I : float
        Net toroidal current determining a circulation of Φ (not φ).
        Default is zero.
    Y : float
        Net poloidal current determining a circulation of Φ (not φ).
        Default is zero.

    """

    _immediate_attributes_ = ["_surface", "_Phi_basis", "_B0", "I", "Y"]

    def __init__(
        self,
        surface,
        M,
        N,
        NFP=None,
        sym=False,
        B0=None,
        I=0.0,  # noqa: E741
        Y=0.0,
    ):
        self._surface = surface
        self._Phi_basis = DoubleFourierSeries(
            M=M, N=N, NFP=setdefault(NFP, surface.NFP), sym=sym
        )
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
            If True, override ``grid`` if necessary and use a full
            resolution grid to compute quantities and then downsample to ``grid``.
            If False, uses only ``grid``, which may lead to
            inaccurate values for surface or volume averages.

        Returns
        -------
        data : dict[str, jnp.ndarray]
            Quantities and intermediate variables computed on ``grid``.
        RpZ_data : dict[str, jnp.ndarray]
            Quantities and intermediate variables computed on the
            (R, ϕ, Z) coordinates in ``RpZ_data``.

        """
        errorif(
            self.M_Phi > grid.M, msg=f"Got M_Phi = {self.M_Phi} > {grid.M} = grid.M."
        )
        errorif(
            self.N_Phi > grid.N, msg=f"Got N_Phi = {self.N_Phi} > {grid.N} = grid.N."
        )

        kwargs.setdefault("B0", self._B0)

        # to simplify computation of a singular integral for ∇φ
        if kwargs.get("on_boundary", False) and "eval_interpolator" not in kwargs:
            if RpZ_grid is None:
                errorif(RpZ_data is not None, msg="Please supply RpZ_grid.")
            else:
                kwargs["eval_interpolator"] = get_interpolator(
                    eval_grid=RpZ_grid,
                    source_grid=grid,
                    source_data=super().compute(
                        ["|e_theta x e_zeta|", "e_theta", "e_zeta"],
                        grid,
                        params,
                        transforms,
                        data,
                        override_grid,
                        **kwargs,
                    ),
                    **kwargs,
                )

        if RpZ_data is None:
            if RpZ_grid is None:
                RpZ_grid = grid
                RpZ_data = data
                same_grid = True
            else:
                same_grid = False
            RpZ_data = super().compute(
                ["R", "phi", "Z"],
                RpZ_grid,
                params,
                transforms,
                data=RpZ_data,
                override_grid=override_grid,
                **kwargs,
            )
            if same_grid:
                data = RpZ_data

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

    Implements the interior Dirichlet formulation in multiply connected
    geometry described in [1].

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂𝒳.
    M : int
        Poloidal Fourier resolution to interpolate potential on ∂𝒳.
    N : int
        Toroidal Fourier resolution to interpolate potential on ∂𝒳.
    sym : str
        Symmetry for Fourier basis interpolating the periodic part of the
        potential. Default is ``sin`` when the surface is stellarator
        symmetric and ``False`` otherwise.
    M_coil : int
        Poloidal Fourier resolution to interpolate coil potential on ∂𝒳.
        Default is ``M``.
    N_coil : int
        Poloidal Fourier resolution to interpolate coil potential on ∂𝒳.
        Default is ``N``.
    sym_coil : str
        Symmetry for Fourier basis interpolating the periodic part of the
        coil potential. Default is ``sym``.
    B_coil : _MagneticField
        Magnetic field from coil current sources.
        This must be smooth and divergence free for correctness.
    Y_coil : float
        Net poloidal current determining circulation of coil field.
        Default is to compute from ``B_coil``.
    I_plasma : float
        Net toroidal plasma current determining a circulation of Φ.
        Default is zero.
    I_sheet : float
        Net toroidal sheet current determining a circulation of Φ.
        Default is zero.

    """

    _immediate_attributes_ = ["_Phi_coil_basis", "_B_coil"]

    def __init__(
        self,
        surface,
        M,
        N,
        sym=None,
        M_coil=None,
        N_coil=None,
        sym_coil=None,
        B_coil=None,
        Y_coil=None,
        I_plasma=0.0,
        I_sheet=0.0,
    ):
        sym = setdefault(sym, "sin" if surface.sym else False)
        I = I_plasma + I_sheet  # noqa: E741

        super().__init__(
            surface,
            M,
            N,
            surface.NFP,
            sym,
            FreeSurfaceOuterField._B0(I, Y_coil),
            I,
            Y_coil,
        )
        if M_coil is None and N_coil is None and sym_coil is None:
            self._Phi_coil_basis = self._Phi_basis
        else:
            self._Phi_coil_basis = DoubleFourierSeries(
                M=setdefault(M_coil, M),
                N=setdefault(N_coil, N),
                NFP=surface.NFP,
                sym=setdefault(sym_coil, sym),
            )
        self._B_coil = B_coil

    @staticmethod
    def _B0(I, Y):  # noqa: E741
        """Returns ∇(Φ (secular))."""
        warnif(
            I != 0,
            NotImplementedError,
            "Must supply B0 as kwarg in compute method for correctness.",
        )
        return ToroidalMagneticField(setdefault(Y, 0), 1)

    def __setattr__(self, name, value):
        if (
            name in FreeSurfaceOuterField._immediate_attributes_
            or name in SourceFreeField._immediate_attributes_
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_surface"), name, value)

    @property
    def Phi_coil_basis(self):
        """DoubleFourierSeries: Basis for periodic part of coil potential."""
        return self._Phi_coil_basis

    @property
    def sym_Phi_coil(self):
        """str: Symmetry of periodic part of Phi_coil (no symmetry if False)."""
        return self._Phi_coil_basis.sym

    @property
    def M_Phi_coil(self):
        """int: Poloidal resolution of periodic part of Phi_coil."""
        return self._Phi_coil_basis.M

    @property
    def N_Phi_coil(self):
        """int: Toroidal resolution of periodic part of Phi_coil."""
        return self._Phi_coil_basis.N

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
        errorif(
            self.M_Phi_coil > grid.M,
            msg=f"Got M_Phi_coil = {self.M_Phi_coil} > {grid.M} = grid.M.",
        )
        errorif(
            self.N_Phi_coil > grid.N,
            msg=f"Got N_Phi_coil = {self.N_Phi_coil} > {grid.N} = grid.N.",
        )
        kwargs.setdefault("B_coil", self._B_coil)
        if self.Y is None and (params is None or "Y" not in params):
            data, RpZ_data = super().compute(
                "Y_coil",
                grid,
                params,
                transforms,
                data,
                RpZ_data,
                RpZ_grid,
                override_grid,
                **kwargs,
            )
            params = setdefault(params, {})
            params["Y"] = data["Y_coil"]
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
