"""High order accurate multiply connected geometry Laplace solver as described in [1]_.

References
----------
.. [1] Unalmis et al. New high-order accurate free surface stellarator
       equilibria optimization and boundary integral methods in DESC.

"""

from math import pi

from scipy.constants import mu_0

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.integrals.singularities import get_interpolator
from desc.utils import errorif, setdefault

from ._core import ToroidalMagneticField, _MagneticField


def _scalar_period(value, name, allow_none=False):
    """Return a scalar period coefficient with consistent validation."""
    if value is None:
        if allow_none:
            return None
        value = 0.0
    value = jnp.asarray(value)
    if value.size != 1:
        raise TypeError(f"{name} must be a scalar, got shape {value.shape}.")
    return value.squeeze()


def _axis_current_field(axis, num_nodes):
    """Build the unit-I harmonic field from a linked toroidal filament."""
    from desc.coils import FourierRZCoil

    field = FourierRZCoil(
        current=2 * pi / mu_0,
        R_n=axis.R_n,
        Z_n=axis.Z_n,
        modes_R=axis.R_basis.modes[:, 2],
        modes_Z=axis.Z_basis.modes[:, 2],
        NFP=axis.NFP,
        sym=axis.sym,
    )
    field.rotmat = axis.rotmat
    field.shift = axis.shift
    # A coil source grid must span the complete closed filament.
    source_grid = LinearGrid(zeta=int(num_nodes), NFP=1)
    return field, source_grid


class _SecularPotentialField(_MagneticField):
    """Base field plus physical harmonic representatives of the I and Y periods."""

    _io_attrs_ = _MagneticField._io_attrs_ + [
        "_base_field",
        "_I_field",
        "_I_num_nodes",
        "_Y_field",
        "_I",
        "_Y",
    ]

    def __init__(self, base_field, I_field, I_source_grid, I=0.0, Y=0.0):  # noqa: E741
        self._base_field = base_field
        self._I_field = I_field
        self._I_num_nodes = I_source_grid.num_nodes
        self._Y_field = ToroidalMagneticField(1.0, 1.0)
        self.I = I
        self.Y = Y
        self._set_up()

    def _set_up(self):
        """Reconstruct runtime-only filament quadrature data after loading."""
        self._I_source_grid = LinearGrid(zeta=int(self._I_num_nodes), NFP=1)

    @property
    def base_field(self):
        """MagneticField: Field supplied independently of the selected periods."""
        return self._base_field

    @property
    def I(self):  # noqa: E743
        """float: Toroidal-current period in T m."""
        return self._I

    @I.setter
    def I(self, new):  # noqa: E743
        self._I = _scalar_period(new, "I")

    @property
    def Y(self):
        """Scalar or None: Poloidal-current period in T m."""
        return self._Y

    @Y.setter
    def Y(self, new):
        self._Y = _scalar_period(new, "Y", allow_none=True)

    def _compute_A_or_B(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        compute_A_or_B="B",
        chunk_size=None,
    ):
        """Evaluate the base field and both physical period representatives."""
        params = {} if params is None else params
        I = params.get("I", self.I)  # noqa: E741
        Y = params.get("Y", self.Y)
        Y = 0.0 if Y is None else Y
        op = {
            "A": "compute_magnetic_vector_potential",
            "B": "compute_magnetic_field",
        }[compute_A_or_B]

        coords = jnp.atleast_2d(jnp.asarray(coords))
        field = jnp.zeros_like(coords, dtype=jnp.float64)
        if self._base_field is not None:
            field = field + getattr(self._base_field, op)(
                coords,
                basis=basis,
                source_grid=source_grid,
                transforms=transforms,
                chunk_size=chunk_size,
            )

        # Avoid an unnecessary close-filament Biot-Savart evaluation for the
        # overwhelmingly common zero-I interior problem. Explicit parameter
        # overrides may be tracers, in which case the evaluation is retained.
        if "I" in params:
            include_I = True
        else:
            include_I = bool(self.I != 0)
        if include_I:
            field = field + I * getattr(self._I_field, op)(
                coords,
                basis=basis,
                source_grid=self._I_source_grid,
                chunk_size=chunk_size,
            )

        field = field + getattr(self._Y_field, op)(
            coords,
            params={"B0": Y, "R0": 1.0},
            basis=basis,
            chunk_size=chunk_size,
        )
        return field

    def compute_magnetic_field(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute the physical magnetic field."""
        return self._compute_A_or_B(
            coords,
            params,
            basis,
            source_grid,
            transforms,
            "B",
            chunk_size,
        )

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute a magnetic vector potential for the physical field."""
        return self._compute_A_or_B(
            coords,
            params,
            basis,
            source_grid,
            transforms,
            "A",
            chunk_size,
        )


class SourceFreeField(FourierRZToroidalSurface):
    """Compute source free magnetic fields.

    Implements the Neumann formulation in multiply connected
    geometry described in [1]_.

    Let 𝒳 be an open set with continuously differentiable
    closed boundary ∂𝒳. This class solves the following
    partial differential equation for the globally defined harmonic remainder
    ϕ, represented in the solver by ``Phi_tilde``. The physical harmonic
    field carrying the prescribed periods is ``B0``.

    -                  ∆φ(x) = 0   x ∈ 𝒳
    -       (B - ∇φ - B₀)(x) = 0   x ∈ 𝒳
    -     n dot (∇φ + B₀)(x) = 0   x ∈ ∂𝒳
    -             n dot B(x) = 0   x ∈ ∂𝒳
    -       curl (B - B₀)(x) = 0   x ∈ 𝒳
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
        Symmetry for the Fourier basis interpolating the globally defined
        potential remainder. Default is ``sin`` when the surface is stellarator
        symmetric and ``False`` otherwise. Pass ``False`` explicitly when the
        boundary data do not share the surface symmetry.
    B0 : _MagneticField
        Magnetic field due to sources other than the selected ``I`` and ``Y``
        harmonic representatives. The complete auxiliary field is exposed as
        :attr:`B0`; the field supplied here is exposed as :attr:`B0_base`.
    I : float
        Net toroidal current. Its physical harmonic representative is the field
        of a unit-period toroidal filament on an axis inferred from ``surface``.
        The filament lies inside the surface and is therefore intended for an
        exterior source-free domain. Default is zero.
    Y : float
        Net poloidal current. Its physical harmonic representative is
        ``Y * grad(phi)``, implemented as a toroidal magnetic field with
        magnitude ``Y / R``. Default is zero.

    """

    _io_attrs_ = ["_surface", "_Phi_tilde_basis", "_B0", "_I", "_Y"]
    _immediate_attributes_ = [
        "_surface",
        "_Phi_tilde_basis",
        "_B0",
        "_I",
        "_Y",
        "I",
        "Y",
    ]

    def __init__(
        self,
        surface,
        M,
        N,
        NFP=None,
        sym=None,
        B0=None,
        I=0.0,  # noqa: E741
        Y=0.0,
    ):
        self._surface = surface
        sym = setdefault(sym, "sin" if surface.sym else False)
        self._Phi_tilde_basis = DoubleFourierSeries(
            M=M,
            N=N,
            NFP=setdefault(NFP, surface.NFP),
            sym=sym,
            stop_gradient=True,
        )
        self._I = _scalar_period(I, "I")
        self._Y = _scalar_period(Y, "Y", allow_none=True)
        # The linked loop need only remain inside the surface; the physical
        # solution is invariant to this representative after the single-valued
        # correction is solved consistently. Oversample the close filament so
        # its normal and tangential traces are accurate on typical BIE grids.
        axis = surface.get_axis()
        num_axis_nodes = max(
            128,
            8 * (2 * surface.M + 1),
            8 * (2 * surface.N + 1) * surface.NFP,
        )
        I_field, I_source_grid = _axis_current_field(axis, num_axis_nodes)
        self._B0 = _SecularPotentialField(B0, I_field, I_source_grid, self._I, self._Y)

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
    def B0(self):
        """MagneticField: Complete auxiliary field, including the I and Y periods."""
        return self._B0

    @property
    def B0_base(self):
        """MagneticField: User-supplied field excluding the built period fields."""
        return self._B0.base_field

    @property
    def I(self):  # noqa: E743
        """float: Net toroidal-current period in T m."""
        return self._I

    @I.setter
    def I(self, new):  # noqa: E743
        self._I = _scalar_period(new, "I")
        if hasattr(self, "_B0") and isinstance(self._B0, _SecularPotentialField):
            self._B0.I = self._I

    @property
    def Y(self):
        """Scalar or None: Net poloidal-current period in T m."""
        return self._Y

    @Y.setter
    def Y(self, new):
        self._Y = _scalar_period(new, "Y", allow_none=True)
        if hasattr(self, "_B0") and isinstance(self._B0, _SecularPotentialField):
            self._B0.Y = self._Y

    @property
    def Phi_tilde_basis(self):
        """DoubleFourierSeries: Basis for the globally defined potential remainder."""
        return self._Phi_tilde_basis

    @property
    def sym_Phi_tilde(self):
        """str: Symmetry of the potential remainder (no symmetry if False)."""
        return self._Phi_tilde_basis.sym

    @property
    def M_Phi_tilde(self):
        """int: Poloidal resolution of the potential remainder."""
        return self._Phi_tilde_basis.M

    @property
    def N_Phi_tilde(self):
        """int: Toroidal resolution of the potential remainder."""
        return self._Phi_tilde_basis.N

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
            self.M_Phi_tilde > grid.M,
            msg=f"Got M_Phi_tilde = {self.M_Phi_tilde} > {grid.M} = grid.M.",
        )
        errorif(
            self.N_Phi_tilde > grid.N,
            msg=f"Got N_Phi_tilde = {self.N_Phi_tilde} > {grid.N} = grid.N.",
        )

        if "B0" not in kwargs:
            kwargs["B0"] = self._B0
            if params is not None:
                B0_params = {key: params[key] for key in ("I", "Y") if key in params}
                if B0_params:
                    kwargs["B0_params"] = B0_params

        # to simplify computation of the singular integral for B_remainder
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
    geometry described in [1]_.
    For this formulation, ``Phi_tilde`` represents the globally
    defined boundary density Φ̃ used in the Dirichlet integral equation, while
    ``B0`` supplies the physical harmonic field carrying the prescribed periods.

    Parameters
    ----------
    surface : Surface
        Geometry defining ∂𝒳.
    M : int
        Poloidal Fourier resolution to interpolate potential on ∂𝒳.
    N : int
        Toroidal Fourier resolution to interpolate potential on ∂𝒳.
    sym : str
        Symmetry for the Fourier basis interpolating the globally defined
        boundary density. Default is ``sin`` when the surface is stellarator
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
        Default is zero. The physical representative is built from a linked
        filament inferred from ``surface``.
    I_sheet : float
        Net toroidal sheet current determining a circulation of Φ.
        Default is zero. The physical representative is built from a linked
        filament inferred from ``surface``.

    """

    _io_attrs_ = SourceFreeField._io_attrs_ + ["_varphi_basis", "_B_coil"]
    _immediate_attributes_ = ["_varphi_basis", "_B_coil"]

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
            None,
            I,
            Y_coil,
        )
        if M_coil is None and N_coil is None and sym_coil is None:
            self._varphi_basis = self._Phi_tilde_basis
        else:
            self._varphi_basis = DoubleFourierSeries(
                M=setdefault(M_coil, M),
                N=setdefault(N_coil, N),
                NFP=surface.NFP,
                sym=setdefault(sym_coil, sym),
                stop_gradient=True,
            )
        self._B_coil = B_coil

    def __setattr__(self, name, value):
        if (
            name in FreeSurfaceOuterField._immediate_attributes_
            or name in SourceFreeField._immediate_attributes_
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_surface"), name, value)

    @property
    def varphi_basis(self):
        """DoubleFourierSeries: Basis for periodic part of coil potential."""
        return self._varphi_basis

    @property
    def sym_varphi(self):
        """str: Symmetry of periodic part of varphi (no symmetry if False)."""
        return self._varphi_basis.sym

    @property
    def M_varphi(self):
        """int: Poloidal resolution of periodic part of varphi."""
        return self._varphi_basis.M

    @property
    def N_varphi(self):
        """int: Toroidal resolution of periodic part of varphi."""
        return self._varphi_basis.N

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
            self.M_varphi > grid.M,
            msg=f"Got M_varphi = {self.M_varphi} > {grid.M} = grid.M.",
        )
        errorif(
            self.N_varphi > grid.N,
            msg=f"Got N_varphi = {self.N_varphi} > {grid.N} = grid.N.",
        )
        kwargs.setdefault("B_coil", self._B_coil)
        if self.Y is None and (params is None or params.get("Y", None) is None):
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
