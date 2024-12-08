"""Magnetic field due to sheet current on a winding surface."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
from scipy.constants import mu_0

from desc.backend import cho_factor, cho_solve, fori_loop, jnp
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz_vec
from desc.compute.utils import _compute as compute_fun
from desc.derivatives import Derivative
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.integrals import compute_B_plasma
from desc.optimizable import optimizable_parameter
from desc.utils import (
    Timer,
    check_posint,
    copy_coeffs,
    dot,
    errorif,
    safediv,
    setdefault,
    warnif,
)

from ._core import (
    _MagneticField,
    biot_savart_general,
    biot_savart_general_vector_potential,
)


class CurrentPotentialField(_MagneticField, FourierRZToroidalSurface):
    """Magnetic field due to a surface current potential on a toroidal surface.

    Surface current K is assumed given by K = n x ∇ Φ
    where:

        - n is the winding surface unit normal.
        - Phi is the current potential function, which is a function of theta and zeta.

    This function then uses biot-savart to find the B field from this current
    density K on the surface.

    Parameters
    ----------
    potential : callable
        function to compute the current potential. Should have a signature of
        the form potential(theta,zeta,**params) -> ndarray.
        theta,zeta are poloidal and toroidal angles on the surface.
        Assumed to have units of Amperes.
    potential_dtheta: callable
        function to compute the theta derivative of the current potential
    potential_dzeta: callable
        function to compute the zeta derivative of the current potential
    params : dict, optional
        default parameters to pass to potential function (and its derivatives)
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for winding surface R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry for the surface geometry.
        Default is "auto" which enforces if modes are symmetric. If True,
        non-symmetric modes will be truncated.
    M, N: int or None
        Maximum poloidal and toroidal mode numbers. Defaults to maximum from modes_R
        and modes_Z.
    name : str
        name for this field
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

    """

    _io_attrs_ = (
        _MagneticField._io_attrs_
        + FourierRZToroidalSurface._io_attrs_
        + [
            "_params",
        ]
    )

    def __init__(
        self,
        potential,
        potential_dtheta,
        potential_dzeta,
        params=None,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        M=None,
        N=None,
        name="",
        check_orientation=True,
    ):
        assert callable(potential), "Potential must be callable!"
        assert callable(potential_dtheta), "Potential derivative must be callable!"
        assert callable(potential_dzeta), "Potential derivative must be callable!"

        self._potential = potential
        self._potential_dtheta = potential_dtheta
        self._potential_dzeta = potential_dzeta
        self._params = params

        super().__init__(
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            M=M,
            N=N,
            name=name,
            check_orientation=check_orientation,
        )

    # TODO: make this an optimizable parameter so the potential may be optimized
    @property
    def params(self):
        """Dict of parameters to pass to potential function and its derivatives."""
        return self._params

    @params.setter
    def params(self, new):
        warnif(
            len(new) != len(self._params),
            UserWarning,
            "Length of new params is different from length of current params! "
            "May cause errors unless potential function is also changed.",
        )
        self._params = new

    @property
    def potential(self):
        """Potential function, signature (theta,zeta,**params) -> potential value."""
        return self._potential

    @potential.setter
    def potential(self, new):
        if new != self._potential:
            assert callable(new), "Potential must be callable!"
            self._potential = new

    @property
    def potential_dtheta(self):
        """Phi poloidal deriv. function, signature (theta,zeta,**params) -> value."""
        return self._potential_dtheta

    @potential_dtheta.setter
    def potential_dtheta(self, new):
        if new != self._potential_dtheta:
            assert callable(new), "Potential derivative must be callable!"
            self._potential_dtheta = new

    @property
    def potential_dzeta(self):
        """Phi toroidal deriv. function, signature (theta,zeta,**params) -> value."""
        return self._potential_dzeta

    @potential_dzeta.setter
    def potential_dzeta(self, new):
        if new != self._potential_dzeta:
            assert callable(new), "Potential derivative must be callable!"
            self._potential_dzeta = new

    def save(self, file_name, file_format=None, file_mode="w"):
        """Save the object.

        **Not supported for this object!**

        Parameters
        ----------
        file_name : str file path OR file instance
            location to save object
        file_format : str (Default hdf5)
            format of save file. Only used if file_name is a file path
        file_mode : str (Default w - overwrite)
            mode for save file. Only used if file_name is a file path

        """
        raise OSError(
            "Saving CurrentPotentialField is not supported,"
            " as the potential function cannot be serialized."
        )

    def _compute_A_or_B(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        compute_A_or_B="B",
    ):
        """Compute magnetic field or vector potential at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from source_grid
        compute_A_or_B: {"A", "B"}, optional
            whether to compute the magnetic vector potential "A" or the magnetic field
            "B". Defaults to "B"

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field or vector potential at specified points

        """
        source_grid = source_grid or LinearGrid(
            M=30 + 2 * self.M,
            N=30 + 2 * self.N,
            NFP=self.NFP,
        )
        return _compute_A_or_B_from_CurrentPotentialField(
            field=self,
            coords=coords,
            params=params,
            basis=basis,
            source_grid=source_grid,
            transforms=transforms,
            compute_A_or_B=compute_A_or_B,
        )

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None, transforms=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from source_grid

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        return self._compute_A_or_B(coords, params, basis, source_grid, transforms, "B")

    def compute_magnetic_vector_potential(
        self, coords, params=None, basis="rpz", source_grid=None, transforms=None
    ):
        """Compute magnetic vector potential at a set of points.

        This assumes the Coulomb gauge.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate vector potential at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic vector potential.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from source_grid

        Returns
        -------
        A : ndarray, shape(N,3)
            Magnetic vector potential at specified points.

        """
        return self._compute_A_or_B(coords, params, basis, source_grid, transforms, "A")

    @classmethod
    def from_surface(
        cls,
        surface,
        potential,
        potential_dtheta,
        potential_dzeta,
        params=None,
    ):
        """Create CurrentPotentialField using geometry provided by given surface.

        Parameters
        ----------
        surface: FourierRZToroidalSurface, optional, default None
            Existing FourierRZToroidalSurface object to create a
            CurrentPotentialField with.
        potential : callable
            function to compute the current potential. Should have a signature of
            the form potential(theta,zeta,**params) -> ndarray.
            theta,zeta are poloidal and toroidal angles on the surface
        potential_dtheta: callable
            function to compute the theta derivative of the current potential
        potential_dzeta: callable
            function to compute the zeta derivative of the current potential
        params : dict, optional
            default parameters to pass to potential function (and its derivatives)

        """
        errorif(
            not isinstance(surface, FourierRZToroidalSurface),
            TypeError,
            "Expected type FourierRZToroidalSurface for argument surface, "
            f"instead got type {type(surface)}",
        )

        R_lmn = surface.R_lmn
        Z_lmn = surface.Z_lmn
        modes_R = surface._R_basis.modes[:, 1:]
        modes_Z = surface._Z_basis.modes[:, 1:]
        NFP = surface.NFP
        sym = surface.sym
        name = surface.name

        return cls(
            potential,
            potential_dtheta,
            potential_dzeta,
            params,
            R_lmn,
            Z_lmn,
            modes_R,
            modes_Z,
            NFP,
            sym,
            name=name,
            check_orientation=False,
        )


class FourierCurrentPotentialField(_MagneticField, FourierRZToroidalSurface):
    """Magnetic field due to a surface current potential on a toroidal surface.

    Surface current K is assumed given by

    K = n x ∇ Φ

    Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    where:

        - n is the winding surface unit normal.
        - Phi is the current potential function, which is a function of theta and zeta,
          and is given as a secular linear term in theta/zeta and a double Fourier
          series in theta/zeta.

    This class then uses biot-savart to find the B field from this current
    density K on the surface.

    Parameters
    ----------
    Phi_mn : ndarray
        Fourier coefficients of the double FourierSeries part of the current potential.
        Has units of Amperes.
    modes_Phi : array-like, shape(k,2)
        Poloidal and Toroidal mode numbers corresponding to passed-in Phi_mn
        coefficients.
    I : float
        Net current linking the plasma and the surface toroidally
        Denoted I in the algorithm, has units of Amperes.
    G : float
        Net current linking the plasma and the surface poloidally
        Denoted G in the algorithm, has units of Amperes.
        NOTE: a negative G will tend to produce a positive toroidal magnetic field
        B in DESC, as in DESC the poloidal angle is taken to be positive
        and increasing when going in the clockwise direction, which with the
        convention n x grad(phi) will result in a toroidal field in the negative
        toroidal direction.
    sym_Phi :  {False,"cos","sin"}
        whether to enforce a given symmetry for the DoubleFourierSeries part of the
        current potential.
    M_Phi, N_Phi: int or None
        Maximum poloidal and toroidal mode numbers for the single valued part of the
        current potential.
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for winding surface R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry for the surface geometry.
        Default is "auto" which enforces if modes are symmetric. If True,
        non-symmetric modes will be truncated.
    M, N: int or None
        Maximum poloidal and toroidal mode numbers. Defaults to maximum from modes_R
        and modes_Z.
    name : str
        name for this field
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

    """

    _io_attrs_ = (
        _MagneticField._io_attrs_
        + FourierRZToroidalSurface._io_attrs_
        + ["_Phi_mn", "_I", "_G", "_Phi_basis", "_M_Phi", "_N_Phi", "_sym_Phi"]
    )

    def __init__(
        self,
        Phi_mn=np.array([0.0]),
        modes_Phi=np.array([[0, 0]]),
        I=0,
        G=0,
        sym_Phi=False,
        M_Phi=None,
        N_Phi=None,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        M=None,
        N=None,
        name="",
        check_orientation=True,
    ):
        Phi_mn, modes_Phi = map(np.asarray, (Phi_mn, modes_Phi))
        assert (
            Phi_mn.size == modes_Phi.shape[0]
        ), "Phi_mn size and modes_Phi.shape[0] must be the same size!"

        assert np.issubdtype(modes_Phi.dtype, np.integer)

        M_Phi = setdefault(M_Phi, np.max(abs(modes_Phi[:, 0])))
        N_Phi = setdefault(N_Phi, np.max(abs(modes_Phi[:, 1])))

        self._M_Phi = M_Phi
        self._N_Phi = N_Phi

        self._sym_Phi = sym_Phi
        self._Phi_basis = DoubleFourierSeries(M=M_Phi, N=N_Phi, NFP=NFP, sym=sym_Phi)
        self._Phi_mn = copy_coeffs(Phi_mn, modes_Phi, self._Phi_basis.modes[:, 1:])

        self._I = float(np.squeeze(I))
        self._G = float(np.squeeze(G))

        super().__init__(
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            M=M,
            N=N,
            name=name,
            check_orientation=check_orientation,
        )

    @optimizable_parameter
    @property
    def I(self):  # noqa: E743
        """Net current linking the plasma and the surface toroidally."""
        return self._I

    @I.setter
    def I(self, new):  # noqa: E743
        self._I = float(np.squeeze(new))

    @optimizable_parameter
    @property
    def G(self):
        """Net current linking the plasma and the surface poloidally."""
        return self._G

    @G.setter
    def G(self, new):
        self._G = float(np.squeeze(new))

    @optimizable_parameter
    @property
    def Phi_mn(self):
        """Fourier coefficients describing single-valued part of potential."""
        return self._Phi_mn

    @Phi_mn.setter
    def Phi_mn(self, new):
        if len(new) == self.Phi_basis.num_modes:
            self._Phi_mn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Phi_mn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Phi_basis.num_modes} modes."
            )

    @property
    def Phi_basis(self):
        """DoubleFourierSeries: Spectral basis for Phi."""
        return self._Phi_basis

    @property
    def sym_Phi(self):
        """str: Type of symmetry of periodic part of Phi (no symmetry if False)."""
        return self._sym_Phi

    @property
    def M_Phi(self):
        """int: Poloidal resolution of periodic part of Phi."""
        return self._M_Phi

    @property
    def N_Phi(self):
        """int: Toroidal resolution of periodic part of Phi."""
        return self._N_Phi

    def change_Phi_resolution(self, M=None, N=None, NFP=None, sym_Phi=None):
        """Change the maximum poloidal and toroidal resolution for Phi.

        Parameters
        ----------
        M : int
            Poloidal resolution to change Phi basis to.
            If None, defaults to current self.Phi_basis poloidal resolution
        N : int
            Toroidal resolution to change Phi basis to.
            If None, defaults to current self.Phi_basis toroidal resolution
        NFP : int
            Number of field periods for surface and Phi basis.
            If None, defaults to current NFP.
            Note: will change the NFP of the surface geometry as well as the
            Phi basis.
        sym_Phi :  {"auto","cos","sin",False}
            whether to enforce a given symmetry for the DoubleFourierSeries part of the
            current potential. Default is "auto" which enforces if modes are symmetric.
            If True, non-symmetric modes will be truncated.

        """
        M = M or self._M_Phi
        N = N or self._M_Phi
        NFP = NFP or self.NFP
        sym_Phi = sym_Phi or self.sym_Phi

        Phi_modes_old = self.Phi_basis.modes
        self.Phi_basis.change_resolution(M=M, N=N, NFP=self.NFP, sym=sym_Phi)

        self._Phi_mn = copy_coeffs(self.Phi_mn, Phi_modes_old, self.Phi_basis.modes)
        self._M_Phi = M
        self._N_Phi = N
        self._sym_Phi = sym_Phi
        self.change_resolution(
            NFP=NFP
        )  # make sure surface and Phi basis NFP are the same

    def _compute_A_or_B(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        compute_A_or_B="B",
    ):
        """Compute magnetic field or vector potential at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from source_grid
        compute_A_or_B: {"A", "B"}, optional
            whether to compute the magnetic vector potential "A" or the magnetic field
            "B". Defaults to "B"

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field or vector potential at specified points

        """
        source_grid = source_grid or LinearGrid(
            M=30 + 2 * max(self.M, self.M_Phi),
            N=30 + 2 * max(self.N, self.N_Phi),
            NFP=self.NFP,
        )
        return _compute_A_or_B_from_CurrentPotentialField(
            field=self,
            coords=coords,
            params=params,
            basis=basis,
            source_grid=source_grid,
            transforms=transforms,
            compute_A_or_B=compute_A_or_B,
        )

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None, transforms=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from source_grid

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        return self._compute_A_or_B(coords, params, basis, source_grid, transforms, "B")

    def compute_magnetic_vector_potential(
        self, coords, params=None, basis="rpz", source_grid=None, transforms=None
    ):
        """Compute magnetic vector potential at a set of points.

        This assumes the Coulomb gauge.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate vector potential at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic vector potential.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from source_grid

        Returns
        -------
        A : ndarray, shape(N,3)
            Magnetic vector potential at specified points.

        """
        return self._compute_A_or_B(coords, params, basis, source_grid, transforms, "A")

    @classmethod
    def from_surface(
        cls,
        surface,
        Phi_mn=np.array([0.0]),
        modes_Phi=np.array([[0, 0]]),
        I=0,
        G=0,
        sym_Phi="auto",
        M_Phi=None,
        N_Phi=None,
    ):
        """Create FourierCurrentPotentialField using geometry of given surface.

        Parameters
        ----------
        surface: FourierRZToroidalSurface, optional, default None
            Existing FourierRZToroidalSurface object to create a
            CurrentPotentialField with.
        Phi_mn : ndarray
            Fourier coefficients of the double FourierSeries of the current potential.
            Should correspond to the given DoubleFourierSeries basis object passed in.
        modes_Phi : array-like, shape(k,2)
            Poloidal and Toroidal mode numbers corresponding to passed-in Phi_mn
            coefficients
        I : float
            Net current linking the plasma and the surface toroidally
            Denoted I in the algorithm
        G : float
            Net current linking the plasma and the surface poloidally
            Denoted G in the algorithm
            NOTE: a negative G will tend to produce a positive toroidal magnetic field
            B in DESC, as in DESC the poloidal angle is taken to be positive
            and increasing when going in the clockwise direction, which with the
            convention n x grad(phi) will result in a toroidal field in the negative
            toroidal direction.
        sym_Phi :  {"auto", "cos","sin", False}
            whether to enforce a given symmetry for the DoubleFourierSeries part of the
            current potential. If "auto", assumes sin symmetry if the surface is
            symmetric, else False.
        M_Phi, N_Phi: int or None
            Maximum poloidal and toroidal mode numbers for the single valued part of the
            current potential.

        """
        if not isinstance(surface, FourierRZToroidalSurface):
            raise TypeError(
                "Expected type FourierRZToroidalSurface for argument surface, "
                f"instead got type {type(surface)}"
            )
        R_lmn = surface.R_lmn
        Z_lmn = surface.Z_lmn
        modes_R = surface._R_basis.modes[:, 1:]
        modes_Z = surface._Z_basis.modes[:, 1:]
        NFP = surface.NFP
        sym = surface.sym
        name = surface.name
        if sym_Phi == "auto":
            sym_Phi = "sin" if surface.sym else False

        return cls(
            Phi_mn=Phi_mn,
            modes_Phi=modes_Phi,
            I=I,
            G=G,
            sym_Phi=sym_Phi,
            M_Phi=M_Phi,
            N_Phi=N_Phi,
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            name=name,
            check_orientation=False,
        )

    def to_CoilSet(  # noqa: C901 - FIXME: simplify this
        self,
        num_coils,
        step=1,
        spline_method="cubic",
        show_plots=False,
        npts=128,
        stell_sym=False,
        plot_kwargs={"figsize": (8, 6)},
    ):
        """Find helical or modular coils from this surface current potential.

        Surface current K is assumed given by

        K = n x ∇ Φ

        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

        where n is the winding surface unit normal, Φ is the current potential
        function, which is a function of theta and zeta, and is given as a
        secular linear term in theta (I)  and zeta (G) and a double Fourier
        series in theta/zeta.

        NOTE: The function is not jit/AD compatible

        Parameters
        ----------
        num_coils : int, optional
            Number of coils to discretize the surface current with.
            If the coils are modular (i.e. I=0), then this is the number of
            coils per field period. If the coils are stellarator-symmetric, then this
            is the number of coils per half field-period. The coils returned always
            have a coil which passes through the theta=0 zeta=0 point of the surface.
        step : int, optional
            Amount of points to skip by when saving the coil geometry spline
            by default 1, meaning that every point will be saved
            if higher, less points will be saved e.g. 3 saves every 3rd point
        spline_method : str, optional
            method of fitting to use for the spline, by default ``"cubic"``
            see ``SplineXYZCoil`` for more info
        show_plots : bool, optional,
            whether to show plots of the contours chosen for coils, by default False
        npts : int, optional
            Number of zeta points over one field period to use to discretize the surface
            when finding constant current potential contours.
        stell_sym : bool
            whether the coils are stellarator-symmetric or not. Defaults to False. Only
            matters for modular coils (currently)
        plot_kwargs : dict
            dict of kwargs to use when plotting the contour plots if ``show_plots=True``
            ``figsize`` is used for the figure size, and the rest are passed to
            ``plt.contourf``

        Returns
        -------
        coils : CoilSet
            DESC `CoilSet` of `SplineXYZCoil` coils that are a discretization of
            the surface current on the given winding surface.

        """
        check_posint(num_coils, "num_coils", False)
        check_posint(step, "step", False)
        check_posint(npts, "npts", False)
        nfp = self.Phi_basis.NFP

        net_toroidal_current = self.I
        net_poloidal_current = self.G
        helicity = safediv(
            net_poloidal_current, net_toroidal_current * nfp, threshold=1e-8
        )
        coil_type = "modular" if jnp.isclose(helicity, 0) else "helical"
        # determine current per coil
        if coil_type == "helical":
            # helical coils
            coil_current = jnp.abs(net_toroidal_current) / num_coils
        else:  # modular coils
            coil_current = net_poloidal_current / num_coils / nfp
            if stell_sym:  # num_coils is num coils per half period, so
                # need to account for the extra factor of 2
                coil_current = coil_current / 2
        assert not jnp.isclose(net_toroidal_current, 0) or not jnp.isclose(
            net_poloidal_current, 0
        ), (
            "Detected both net toroidal and poloidal current are both zero, "
            "this function cannot find windowpane coils"
        )

        contour_theta, contour_zeta = _find_current_potential_contours(
            self,
            num_coils,
            npts,
            show_plots,
            stell_sym,
            net_poloidal_current,
            net_toroidal_current,
            helicity,
            plot_kwargs=plot_kwargs,
        )

        ################################################################
        # Find the XYZ points in real space of the coil contours
        ################################################################

        contour_X, contour_Y, contour_Z = _find_XYZ_points(
            contour_theta,
            contour_zeta,
            self,
        )
        ################################################################
        # Create CoilSet object
        ################################################################
        # local imports to avoid circular imports
        from desc.coils import CoilSet, SplineXYZCoil

        coils = []
        for j in range(num_coils):
            if coil_type == "helical":
                # helical coils
                # make sure that the sign of the coil current is correct
                # by dotting K with the vector along the contour
                # TODO: probably could use helicity sign and just check the slope of
                # the contours to see which way they are going, but this is easy for
                # now and not too expensive
                contour_vector = jnp.array(
                    [
                        contour_X[j][1] - contour_X[j][0],
                        contour_Y[j][1] - contour_Y[j][0],
                        contour_Z[j][1] - contour_Z[j][0],
                    ]
                )
                K = self.compute(
                    "K",
                    grid=Grid(
                        jnp.array([[0, contour_theta[j][0], contour_zeta[j][0]]])
                    ),
                    basis="xyz",
                )["K"]
                current_sign = jnp.sign(jnp.dot(contour_vector, K[0, :]))
                thisCurrent = current_sign * jnp.abs(coil_current)
            else:
                # modular coils
                # make sure that the sign of the coil current is correct
                # don't need to dot with K here because we know the direction
                # based off the direction of the theta contour and sign of G
                # (extra negative sign because a positive G -> negative toroidal B
                # but we always have a right-handed coord system, and so current flowing
                # in positive poloidal direction creates a positive toroidal B)
                # for modular coils, easiest way to check contour direction is to see
                # direction of the contour thetas
                sign_of_theta_contours = jnp.sign(
                    contour_theta[0][-1] - contour_theta[0][0]
                )
                current_sign = -sign_of_theta_contours * jnp.sign(net_poloidal_current)
                thisCurrent = jnp.abs(coil_current) * current_sign
            coil = SplineXYZCoil(
                thisCurrent,
                jnp.append(contour_X[j][0::step], contour_X[j][0]),
                jnp.append(contour_Y[j][0::step], contour_Y[j][0]),
                jnp.append(contour_Z[j][0::step], contour_Z[j][0]),
                method=spline_method,
            )
            coils.append(coil)
        # check_intersection is False here as these coils by construction
        # cannot intersect eachother (they are contours of the current potential
        # which cannot self-intersect by definition)
        # unless stell_sym is true, then the full coilset might have
        # self intersection depending on if the coils cross the
        # symmetry plane, in which case we will check
        if coil_type == "modular":
            final_coilset = CoilSet(
                *coils, NFP=nfp, sym=stell_sym, check_intersection=stell_sym
            )
        else:
            # TODO: once winding surface curve is implemented, enforce sym for
            # helical as well
            final_coilset = CoilSet(*coils, check_intersection=False)
        return final_coilset


def _compute_A_or_B_from_CurrentPotentialField(
    field,
    coords,
    source_grid,
    params=None,
    basis="rpz",
    transforms=None,
    compute_A_or_B="B",
    data=None,
):
    """Compute magnetic field or vector potential at a set of points.

    Parameters
    ----------
    field : CurrentPotentialField or FourierCurrentPotentialField
        current potential field object from which to compute magnetic field.
    coords : array-like shape(N,3)
        cylindrical or cartesian coordinates
    source_grid : Grid,
        source grid upon which to evaluate the surface current density K
    params : dict, optional
        parameters to pass to compute function
        should include the potential
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field
    compute_A_or_B: {"A", "B"}, optional
        whether to compute the magnetic vector potential "A" or the magnetic field
        "B". Defaults to "B"
    data : dict
        if provided, do not compute any dependency data, but instead use provided
        dictionary corresponding to the source_grid.
        data dictionary requires keys `"K", "| e_theta x e_zeta |"` and `"x"`,
        with `"K"` and `"x"` in rpz basis.


    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field or vector potential at specified points

    """
    errorif(
        compute_A_or_B not in ["A", "B"],
        ValueError,
        f'Expected "A" or "B" for compute_A_or_B, instead got {compute_A_or_B}',
    )
    assert basis.lower() in ["rpz", "xyz"]
    coords = jnp.atleast_2d(jnp.asarray(coords))
    if basis == "rpz":
        coords = rpz2xyz(coords)
    op = {"B": biot_savart_general, "A": biot_savart_general_vector_potential}[
        compute_A_or_B
    ]
    # compute surface current, and store grid quantities
    # needed for integration in class
    if data is None:
        if not params or not transforms:
            data = field.compute(
                ["K", "x"],
                grid=source_grid,
                basis="rpz",
                params=params,
                transforms=transforms,
                jitable=True,
            )
        else:
            data = compute_fun(
                field,
                names=["K", "x"],
                params=params,
                transforms=transforms,
                profiles={},
            )

    _rs = data["x"]
    _K = data["K"]

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    _dV = source_grid.weights * data["|e_theta x e_zeta|"] / source_grid.NFP

    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (source_grid.nodes[:, 2] + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = op(
            coords,
            rs,
            K,
            _dV,
        )
        f += fj
        return f

    B = fori_loop(0, source_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
    return B


def solve_regularized_surface_current(  # noqa: C901 fxn too complex
    field,
    eq,
    lambda_regularization=1e-30,
    current_helicity=(1, 0),
    vacuum=False,
    regularization_type="regcoil",
    source_grid=None,
    eval_grid=None,
    vc_source_grid=None,
    external_field=None,
    external_field_grid=None,
    verbose=1,
):
    """Runs REGCOIL-like algorithm to find the current potential for the surface.

    NOTE: The function is not jit/AD compatible

    Follows algorithm of [1]_ to find the current potential Phi on the surface,
    given a surface current::

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    The algorithm minimizes the quadratic flux on the plasma surface due to the
    surface current (B_Phi_SV for field from the single valued part Φₛᵥ, and
    B_GI for that from the secular terms I and G), plasma current, and external
    fields::

        Bn = ∫ ∫ (B . n)^2 dA
        B = B_plasma + B_external + B_Phi_SV + B_GI

    G is fixed by the equilibrium magnetic field strength, and I is determined
    by the desired coil topology (given by ``current_helicity``), with zero
    helicity corresponding to modular coils, and non-zero helicity corresponding
    to helical coils. The algorithm then finds the single-valued part of Φ
    by minimizing the quadratic flux on the plasma surface along with a
    regularization term on the surface current magnitude::

        min_Φₛᵥ  ∫ ∫ (B . n)^2 dA + λ ∫ ∫ ||K||^2 dA

    where λ is the regularization parameter, smaller `lambda_regularization`
    corresponds to less regularization (consequently, lower Bn error but more
    complex and large surface currents) and larger `lambda_regularization`
    corresponds to more regularization (consequently, higher Bn error but simpler
    and smaller surface currents).

    If the ``simple`` regularization is used, the problem instead becomes::

        min_Φₛᵥ  (B . n)^2 + λ  ||Φ_mn||^2

    Parameters
    ----------
    field : FourierCurrentPotentialField
        ``FourierCurrentPotentialField`` to run REGCOIL algorithm with.
    eq : Equilibrium
        Equilibrium to minimize the quadratic flux (plus regularization) on.
    lambda_regularization : float or ndarray, optional
        regularization parameter, >= 0, regularizes minimization of Bn
        on plasma surface with minimization of current density mag K on winding
        surface i.e. larger lambda_regularization, simpler coilset and smaller
        currents, but worse Bn. If a float, only runs REGCOIL for that single value
        and returns a list with the single FourierCurrentPotentialField and the
        associated data.
        If an array is passed, will run REGCOIL for each lambda_regularization in
        that array and return a list of FourierCurrentPotentialFields, and the
        associated data.
    current_helicity : tuple of size 2, optional
        Tuple of ``(M_coil, N_coil)`` used to determine coil topology, where`` M_coil``
        is the number of poloidal transits a coil makes before closing back on itself
        and ``N_coil`` is the number of toroidal transits a coil makes before
        returning back to itself.
        if ``N_coil`` is zero and ``M_coil`` nonzero, it corresponds to modular
        coil topology.
        If both ``N_coil``,``M_coil`` are nonzero, it corresponds to helical coils.
        If ``N_coil``,``M_coil`` are both zero, it corresponds to windowpane coils.
        The net toroidal current (when ``M_coil`` is nonzero) is set as
        ``I = N_coil(G-G_ext)/M_coil``
        As an example, if helical coils which make one poloidal transit per field period
        and close on themselves after one full toroidal transit are desired, that
        corresponds to ``current_helicity = (1*NFP, 1)``
    vacuum : bool, optional
        if True, will not include the contribution to the normal field from the
        plasma currents.
    regularization_type : {"simple","regcoil"}
        whether to use a simple regularization based off of just the single-valued
        part of Phi, or to use the full REGCOIL regularization penalizing | K | ^ 2.
        Defaults to ``"regcoil"``
    source_grid : Grid, optional
        Source grid upon which to evaluate the surface current when calculating
        the normal field on the plasma surface. Defaults to
        LinearGrid(M=max(3 * current_potential_field.M_Phi, 30),
        N=max(3 * current_potential_field.N_Phi, 30), NFP=eq.NFP)
    eval_grid : Grid, optional
        Grid upon which to evaluate the normal field on the plasma surface, and
        at which the normal field is minimized.
        Defaults to
        `LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)`
    vc_source_grid : LinearGrid
        LinearGrid to use for the singular integral for the virtual casing
        principle to calculate the component of the normal field from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0
    external_field: _MagneticField,
        DESC `_MagneticField` object giving the magnetic field
        provided by any coils/fields external to the winding surface.
        e.g. can provide a TF coilset to calculate the surface current
        which is needed to minimize Bn given this external coilset providing
        the bulk of the required net toroidal magnetic flux, by default None
    external_field_grid : Grid, optional
        Source grid with which to evaluate the external field when calculating
        its contribution to the normal field on the plasma surface (if it is a type
        that requires a source, like a `CoilSet` or a `CurrentPotentialField`).
        By default None, which will use the default grid for the given
        external field type.
    verbose : int, optional
        level of verbosity, if 0 will print nothing.
        1 will display Bn max,min,average and chi^2 values for each
        lambda_regularization.
        2 will display jacobian timing info

    Returns
    -------
    fields  : list of FourierCurrentPotentialField
        A FourierCurrentPotentialField with the Phi_mn set to the
        optimized current potential. This is a list of length
        lambda_regularization.size with the optimized fields
        for each parameter value lambda_regularization.
    data : dict
        Dictionary with the following keys,::

            lambda_regularization : regularization parameter the algorithm was ran
                    with, a array of passed-in `lambda_regularization`
                    corresponding to the list of `Phi_mn`.
            Phi_mn : the single-valued current potential coefficients which
                    minimize the Bn at the given eval_grid on the plasma, subject
                    to regularization on the surface current magnitude governed by
                    lambda_regularization.
                    A list of arrays of length `self.Phi_basis.num_modes` if passed-in
                    `lambda_regularization`,
                    with list length `lambda_regularization.size`, corresponding to the
                    list of regularization parameters `lambda_regularization`.
            I : float, net toroidal current (in Amperes) on the winding surface.
                    Governed by the `current_helicity` parameters, and is zero for
                    modular coils (when `p=current_helicity[0]=0`).
            G : float, net poloidal current (in Amperes) on the winding surface.
                    Determined by the equilibrium toroidal magnetic field, as well as
                    the given external field.
            chi^2_B : quadratic flux squared, integrated over the plasma surface.
                    list of float of  length `lambda_regularization.size`,
                    corresponding to the array of `lambda_regularization` values.
            chi^2_K : Current density magnitude squared, integrated over winding
                    surface. a list of float of length `lambda_regularization.size`,
                    corresponding to the array of `lambda_regularization`.
            ||K|| : Current density magnitude on winding surface, evaluated at the
                    given `source_grid`. A list of arrays, with list length
                    `lambda_regularization.size`, corresponding to the array
                    of `lambda_regularization`.
            eval_grid: Grid object that Bn was evaluated at.
            source_grid: Grid object that Phi and K were evaluated at.

    References
    ----------
    .. [1] Landreman, Matt. "An improved current potential method for fast computation
      of stellarator coil shapes." Nuclear Fusion 57 (2017): 046003.

    """
    errorif(
        len(current_helicity) != 2,
        ValueError,
        "current_helicity must be a length-two tuple",
    )
    errorif(
        any(int(hel) != hel for hel in current_helicity),
        ValueError,
        "Helicity values must be integer",
    )

    errorif(
        regularization_type not in ["simple", "regcoil"],
        ValueError,
        'regularization_type must be "simple" or "regcoil"',
    )
    errorif(
        not isinstance(field, FourierCurrentPotentialField),
        ValueError,
        "Expected FourierCurrentPotentialField for field argument, instead got type "
        f"{type(field)}",
    )
    warnif(
        field.sym_Phi == "cos" and field.sym is True and eq.sym is True,
        UserWarning,
        "Detected incompatible Phi symmetry (cos) for symmetric"
        " equilibrium and surface geometry, it is recommended to switch to"
        " sin symmetry for Phi.",
    )

    current_potential_field = field.copy()  # copy field so we can modify freely
    M_coil = current_helicity[0]  # poloidal transits before coil returns to itself
    N_coil = current_helicity[1]  # toroidal transits before coil returns to itself

    # maybe it is an EquilibriaFamily
    errorif(hasattr(eq, "__len__"), ValueError, "Expected a single equilibrium")

    if vacuum:
        # check if vacuum flag should be True or not
        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure appears to be non-zero (max {pres} Pa), "
            + "vacuum flag should probably be set to False.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current appears to be non-zero (max {curr} A), "
            + "vacuum flag should probably be set to False.",
        )

    data = {}
    if external_field:  # ensure given field is an instance of _MagneticField
        assert hasattr(external_field, "compute_magnetic_field"), (
            "Expected MagneticField for argument external_field, "
            f"got type {type(external_field)} "
        )
        data["external_field"] = external_field
        data["external_field_grid"] = external_field_grid

    if source_grid is None:
        source_grid = LinearGrid(
            M=max(3 * current_potential_field.M_Phi, 30),
            N=max(3 * current_potential_field.N_Phi, 30),
            NFP=int(eq.NFP),
        )
    if eval_grid is None:
        eval_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=int(eq.NFP))
    B_eq_surf = eq.compute("|B|", eval_grid)["|B|"]
    # just need it for normalization, so do a simple mean
    normalization_B = jnp.mean(B_eq_surf)

    data["eval_grid"] = eval_grid
    data["source_grid"] = source_grid

    # plasma surface normal vector magnitude on eval grid
    ne_mag = eq.compute(["|e_theta x e_zeta|"], eval_grid)["|e_theta x e_zeta|"]
    # winding surface normal vector magnitude on source grid
    ns_mag = current_potential_field.compute(["|e_theta x e_zeta|"], source_grid)[
        "|e_theta x e_zeta|"
    ]

    # calculate net enclosed poloidal and toroidal currents
    G_tot = -(eq.compute("G", grid=source_grid)["G"][0] / mu_0 * 2 * jnp.pi)

    if external_field:
        G_ext = _G_from_external_field(external_field, eq, external_field_grid)
    else:
        G_ext = 0

    # G needed by surface current is the total G minus the external contribution
    G = G_tot - G_ext
    # calculate I, net toroidal current on winding surface
    if N_coil == 0 and M_coil == 0:  # windowpane coils
        I = G = 0
    elif N_coil == 0:  # modular coils
        I = 0
    elif M_coil == 0:  # only toroidally closed coils, like PF coils
        I = N_coil * G_tot  # give some toroidal current corr. to N_coil
        G = 0  # because I==0
    else:  # helical coils
        I = N_coil * G / M_coil

    # define functions which will be differentiated
    def Bn_from_K(phi_mn, I, G):
        """B from K from REGCOIL eqn 4."""
        params = current_potential_field.params_dict
        params["Phi_mn"] = phi_mn
        params["I"] = I
        params["G"] = G
        Bn, _ = current_potential_field.compute_Bnormal(
            eq.surface, eval_grid=eval_grid, source_grid=source_grid, params=params
        )
        return Bn

    def K(phi_mn, I, G):
        params = current_potential_field.params_dict
        params["Phi_mn"] = phi_mn
        params["I"] = I
        params["G"] = G
        data = current_potential_field.compute("K", grid=source_grid, params=params)
        return data["K"]

    def surfint_Ksqd(phi_mn, I, G):
        K_vec = K(phi_mn, I, G)
        K_mag_sqd = dot(K_vec, K_vec, axis=1)
        integrand = (K_mag_sqd * ns_mag * source_grid.weights).T
        return jnp.sum(integrand).squeeze()

    def surfint_Bnsqd(phi_mn, I, G):
        integrand = ((Bn_from_K(phi_mn, I, G) ** 2) * ne_mag * eval_grid.weights).T
        return jnp.sum(integrand).squeeze()

    if regularization_type == "regcoil":
        # also need these gradients for the RHS if using regcoil regularization
        # set jacobian deriv mode based on the matrix dimensions,  which is the output
        # size (grid nodes, which is eval_grid for Bn and 3*source_grid for K)
        # by the input size (the number of Phi modes)
        deriv_mode = (
            "fwd"
            if eval_grid.num_nodes >= 0.5 * current_potential_field.Phi_basis.num_modes
            else "rev"
        )
        grad_Bn = Derivative(Bn_from_K, mode=deriv_mode).compute(
            current_potential_field.Phi_mn, 0.0, 0.0
        )
        # TODO: likely can make the grad_Ksv one into a compute fxn instead of using
        # JAX to compute dK/dPhimn, but this works for now
        deriv_mode = (
            "fwd"
            if 3 * source_grid.num_nodes
            >= 0.5 * current_potential_field.Phi_basis.num_modes
            else "rev"
        )
        grad_Ksv = Derivative(K).compute(current_potential_field.Phi_mn, 0.0, 0.0)

    timer = Timer()
    # calculate the Jacobian matrix A for  Bn_SV = A*Phi_mn
    timer.start("Jacobian Calculation")
    if regularization_type == "regcoil":
        A1 = Derivative(surfint_Bnsqd, mode="hess").compute(
            current_potential_field.Phi_mn, 0.0, 0.0
        )
        A2 = Derivative(surfint_Ksqd, mode="hess").compute(
            current_potential_field.Phi_mn, 0.0, 0.0
        )

    else:
        # set jacobian deriv mode based on the matrix dimensions
        deriv_mode = (
            "fwd"
            if eval_grid.num_nodes >= 0.5 * current_potential_field.Phi_basis.num_modes
            else "rev"
        )
        A = (
            Derivative(Bn_from_K).compute(current_potential_field.Phi_mn, 0.0, 0.0).T
            * ne_mag
            * eval_grid.weights
        ).T
    timer.stop("Jacobian Calculation")
    if verbose > 1:
        timer.disp("Jacobian Calculation")

    current_potential_field.I = float(I)
    current_potential_field.G = float(G)

    # find the normal field from the secular part of the current potential
    # also mutliply by necessary weights and normal vector magnitude
    Bn_GI = (
        Bn_from_K(jnp.zeros_like(current_potential_field.Phi_mn), I, G)
        * ne_mag
        * eval_grid.weights
    )
    if not vacuum:  # get Bn from plasma contribution
        Bn_plasma = compute_B_plasma(eq, eval_grid, vc_source_grid, normal_only=True)
        Bn_plasma = Bn_plasma * ne_mag * eval_grid.weights

    else:
        Bn_plasma = jnp.zeros_like(Bn_GI)  # from plasma current, currently assume is 0
    # find external field's Bnormal contribution
    if external_field:
        Bn_ext, _ = external_field.compute_Bnormal(
            eq.surface, eval_grid=eval_grid, source_grid=external_field_grid
        )
        Bn_ext = Bn_ext * ne_mag * eval_grid.weights

    else:
        Bn_ext = jnp.zeros_like(Bn_GI)

    rhs = Bn_plasma + Bn_ext + Bn_GI
    if regularization_type == "regcoil":
        rhs_B = -2 * ((grad_Bn.T * rhs).T).sum(axis=0)
        dotted_K_d_K_d_Phimn = dot(
            K(
                jnp.zeros_like(current_potential_field.Phi_mn),
                current_potential_field.I,
                current_potential_field.G,
            )[:, :, jnp.newaxis],
            grad_Ksv,
            axis=1,
        )
        rhs_K = -2 * (dotted_K_d_K_d_Phimn.T * ns_mag * source_grid.weights).T.sum(
            axis=0
        )
    lambda_regularizations = np.atleast_1d(lambda_regularization)

    chi2Bs = []
    chi2Ks = []
    K_mags = []
    phi_mns = []
    Bn_arrs = []
    fields = []

    # calculate the Phi_mn which minimizes
    # (chi^2_B + lambda_regularization*chi^2_K) for each lambda_regularization
    # pre-calculate the SVD
    if regularization_type == "simple":
        u, s, vh = jnp.linalg.svd(A, full_matrices=False)
        s_uT = (u * s).T
        s_uT_b = -s_uT @ rhs
        vht = vh.T

    for lambda_regularization in lambda_regularizations:
        printstring = (
            "Calculating Phi_SV for "
            + f"lambda_regularization = {lambda_regularization:1.5e}"
        )
        if verbose > 0:
            print(
                "#" * len(printstring)
                + "\n"
                + printstring
                + "\n"
                + "#" * len(printstring)
            )

        if regularization_type == "simple":
            # calculate Phi_mn with SVD inverse plus the regularization
            phi_mn_opt = vht @ ((1 / (s**2 + lambda_regularization)) * s_uT_b)
        else:
            # solve linear system
            matrix = A1 + lambda_regularization * A2
            rhs = rhs_B + lambda_regularization * rhs_K
            cho_fac_c_and_lower = cho_factor(matrix)
            phi_mn_opt = cho_solve(cho_fac_c_and_lower, rhs)
            if jnp.any(jnp.isnan(phi_mn_opt)):
                print(
                    "Singular linear system encountered at "
                    f"lambda={lambda_regularization}, retrying with pseudoinverse"
                )
                # failed to solve, likely bc matrix is singular, use lstsq instead
                phi_mn_opt = jnp.linalg.lstsq(matrix, rhs)[0]

        phi_mns.append(phi_mn_opt)

        Bn_SV = Bn_from_K(phi_mn_opt, 0.0, 0.0) * ne_mag * eval_grid.weights
        Bn_tot = Bn_SV + Bn_plasma + Bn_GI + Bn_ext

        chi_B = jnp.sum(Bn_tot * Bn_tot / ne_mag / eval_grid.weights)
        chi2Bs.append(chi_B)

        current_potential_field.Phi_mn = phi_mn_opt
        fields.append(current_potential_field.copy())
        K = current_potential_field.compute(["K"], grid=source_grid)["K"]
        K_mag = jnp.linalg.norm(K, axis=-1)
        chi_K = jnp.sum(K_mag * K_mag * ns_mag * source_grid.weights)
        chi2Ks.append(chi_K)
        K_mags.append(K_mag)
        Bn_print = Bn_tot
        Bn_print_normalized = Bn_tot / normalization_B
        Bn_arrs.append(Bn_tot)
        if verbose > 0:
            units = " (T)"
            printstring = f"chi^2 B = {chi_B:1.5e}"
            print(printstring)
            printstring = f"min Bnormal = {jnp.min(np.abs(Bn_print)):1.5e}"
            printstring += units
            print(printstring)
            printstring = f"Max Bnormal = {jnp.max(jnp.abs(Bn_print)):1.5e}"
            printstring += units
            print(printstring)
            printstring = f"Avg Bnormal = {jnp.mean(jnp.abs(Bn_print)):1.5e}"
            printstring += units
            print(printstring)
            units = " (unitless)"
            printstring = f"min Bnormal = {jnp.min(np.abs(Bn_print_normalized)):1.5e}"
            printstring += units
            print(printstring)
            printstring = f"Max Bnormal = {jnp.max(jnp.abs(Bn_print_normalized)):1.5e}"
            printstring += units
            print(printstring)
            printstring = f"Avg Bnormal = {jnp.mean(jnp.abs(Bn_print_normalized)):1.5e}"
            printstring += units
            print(printstring)
    data["lambda_regularization"] = lambda_regularizations
    data["Phi_mn"] = phi_mns
    data["I"] = I
    data["G"] = G
    data["chi^2_B"] = chi2Bs
    data["chi^2_K"] = chi2Ks
    data["|K|"] = K_mags
    data["Bn_total"] = Bn_arrs

    return fields, data


# TODO: replace contour finding with optimizing Winding surface curves
# once that is implemented
def _find_current_potential_contours(
    surface_current_field,
    num_coils,
    npts=128,
    show_plots=False,
    stell_sym=False,
    net_poloidal_current=None,
    net_toroidal_current=None,
    helicity=None,
    plot_kwargs={},
):
    """Find contours of constant current potential (i.e. coils).

    Surface current K is assumed given by

    K = n x ∇ Φ

    Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    where:

        - n is the winding surface unit normal.
        - Φ is the current potential function, which is a function of theta and zeta,
          and is given as a secular linear term in theta (I)  and zeta (G) and a double
          Fourier series in theta/zeta.

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current potential to find contours of
    num_coils : int
        number of contours desired
    npts : int
        number of points to discretize the current potential with in the
        zeta direction. The discretization in theta will be proportional to this.
        The discretized current potential is then passed to the
        ``skimage.measure.find_contours`` which finds contours using a
        marching squares algorithm.
    show_plots : bool, optional
        whether to plot the contours, useful for debugging or seeing why contour finding
        fails, by default False
    stell_sym : bool, optional
        if the modular coilset has stellarator symmetry, by default False.
        Does nothing for helical coils.
    net_poloidal_current, net_toroidal_current : float, optional
        the net poloidal (toroidal) current flowing on the surface. If None, will
         attempt to infer from the given surface_current_field's attributes.
    helicity : int, optional
        the helicity of the coil currents, should be consistent with the passed-in
        net currents. If None, will use the correct ratio of net poloidal and net
        toroidal currents.
    plot_kwargs : tuple
            figsize to pass to matplotlib figure call, to control size of figure
            if ``show_plots=True``

    Returns
    -------
    contour_theta, contour_zeta: list of 1D arrays
        list of length num_coils containing arrays of the theta
        and zeta values describing each contour found.

    """
    ################################################################
    # find current helicity
    ################################################################
    # we know that I = -(G - G_ext) / (helicity * NFP)
    # if net_toroidal_current is zero, then we have modular coils,
    # and just make helicity zero
    net_poloidal_current = setdefault(
        net_poloidal_current, surface_current_field.G, net_poloidal_current
    )
    net_toroidal_current = setdefault(
        net_toroidal_current, surface_current_field.I, net_toroidal_current
    )
    nfp = surface_current_field.NFP
    helicity = setdefault(
        helicity,
        safediv(net_poloidal_current, net_toroidal_current * nfp, threshold=1e-8),
        helicity,
    )
    coil_type = "modular" if jnp.isclose(helicity, 0) else "helical"
    dz = 2 * np.pi / nfp / npts
    if coil_type == "helical":
        # helical coils
        zeta_full = jnp.linspace(
            0, 2 * jnp.pi / nfp, round(2 * jnp.pi / nfp / dz), endpoint=True
        )
        # ensure we have always have points at least from -2pi, 2pi as depending
        # on sign of I, the contours from Phi = [0, abs(I)] may have their starting
        # points (the theta value at zeta=0) be positive or negative theta values,
        # and we want to ensure we catch the start and end of the contours
        theta0 = jnp.sign(helicity) * 2 * jnp.pi
        theta1 = -jnp.sign(helicity) * (2 * np.pi * int(np.abs(helicity) + 1))
        theta_full = jnp.linspace(
            theta0,
            theta1,
            round(abs(theta1 - theta0) / dz),
            endpoint=True,
        )

        theta_full = jnp.sort(theta_full)
    else:
        # modular coils
        theta_full = jnp.linspace(0, 2 * jnp.pi, round(2 * jnp.pi / dz))
        # we start below 0 for zeta to allow for contours which may go in/out of
        # the zeta=0 plane
        zeta_full = jnp.linspace(
            -jnp.pi / nfp, (2 + 1) * jnp.pi / nfp, round(4 * jnp.pi / dz)
        )

    ################################################################
    # find contours of constant phi
    ################################################################
    # make linspace contours
    if coil_type == "helical":
        # helical coils
        # we start them on zeta=0 plane, so we will find contours
        # going from 0 to I (corresponding to zeta=0, and theta*sign(I) increasing)
        contours = jnp.linspace(
            0, jnp.abs(net_toroidal_current), num_coils + 1, endpoint=True
        )
        contours = jnp.sort(contours)
    else:
        # modular coils
        # go from zero to G/nfp
        # or G/nfp/2 if stell_sym is True
        max_curr = (
            jnp.abs(net_poloidal_current) / nfp / 2
            if stell_sym
            else jnp.abs(net_poloidal_current) / nfp
        )
        # For stell_sym, must pick these carefully
        # so that the coilset has coils that are in order
        # of ascending phi, otherwise we may get the
        # problem that the first coil is over the zeta=0 sym line
        # and the last coil is on the zeta=pi/NFP sym line
        # use a small offset to avoid this issue
        offset = max_curr / num_coils / 2 if stell_sym else 0
        contours = jnp.linspace(
            offset,
            max_curr + offset,
            num_coils + 1,
            endpoint=True,
        ) * jnp.sign(net_poloidal_current)
        contours = jnp.sort(contours)

    theta_full_2D, zeta_full_2D = jnp.meshgrid(theta_full, zeta_full, indexing="ij")

    grid = Grid(
        jnp.vstack(
            (
                jnp.zeros_like(theta_full_2D.flatten(order="F")),
                theta_full_2D.flatten(order="F"),
                zeta_full_2D.flatten(order="F"),
            )
        ).T,
        sort=False,
    )
    phi_total_full = surface_current_field.compute("Phi", grid=grid)["Phi"].reshape(
        theta_full.size, zeta_full.size, order="F"
    )

    # list of arrays of the zeta and theta coordinates
    # of each constant potential contour
    contour_zeta = []
    contour_theta = []
    # list of arrays of the current potential contours,
    # given as indices in theta,zeta
    contours_indices = []
    for contour in contours:
        this_contour = skimage.measure.find_contours(
            np.asarray(jnp.transpose(phi_total_full)), level=contour
        )
        warnif(
            len(this_contour) > 1,
            UserWarning,
            "Detected multiple current potential contours for the same value,"
            + " indicates there may be unclosed coils or window-pane-like structures"
            " for which coil-cutting is not currently supported",
        )
        contours_indices.append(this_contour[0])
    # from the indices, calculate the actual zeta and theta values of the contours
    contours_theta_zeta_not_uniform = [
        np.array(
            [
                np.interp(contour[:, 0], np.arange(zeta_full.size), zeta_full),
                np.interp(contour[:, 1], np.arange(theta_full.size), theta_full),
            ]
        ).T
        for contour in contours_indices
    ]
    # make all the contours the same length by interpolating
    # any shorter ones to match the length of the longest
    Npts = np.max([c.shape[0] for c in contours_indices])
    s = np.linspace(0, 2 * np.pi, Npts)
    contours_theta_zeta = [
        np.array(
            [
                np.interp(s, np.linspace(0, 2 * np.pi, c[:, 0].size), c[:, 0]),
                np.interp(s, np.linspace(0, 2 * np.pi, c[:, 1].size), c[:, 1]),
            ]
        ).T
        for c in contours_theta_zeta_not_uniform
    ]
    # reverse so that our coils start at zeta=0, this is just a choice
    contours_theta_zeta.reverse()

    numCoils = 0
    # to be used to check closure conditions on the coils
    ## closure condition in zeta for modular is returns to same zeta,
    ## while for helical is that the contour dzeta = 2pi/NFP
    zeta_diff = 2 * jnp.pi / nfp if coil_type == "helical" else 0.0
    ## closure condition in theta for modular is that dtheta = 2pi,
    ## while for helical the dtheta = 2pi*abs(helicity)
    theta_diff = (
        2 * jnp.pi * jnp.abs(helicity) if coil_type == "helical" else 2 * jnp.pi
    )

    if show_plots:
        plt.figure(figsize=plot_kwargs.pop("figsize", (8, 6)))
        plt.contourf(
            zeta_full_2D.T,
            theta_full_2D.T,
            jnp.transpose(phi_total_full),
            levels=100,
            **plot_kwargs,
        )
        plt.xlabel(r"$\zeta$")
        plt.ylabel(r"$\theta$")
        plt.xlim([np.min(zeta_full), np.max(zeta_full)])
        plt.ylim([np.min(theta_full), np.max(theta_full)])

    for j in range(num_coils):
        contour_zeta.append(contours_theta_zeta[j][:, 0])
        contour_theta.append(contours_theta_zeta[j][:, 1])
        # check if closed and if not throw warning

        if not jnp.isclose(
            jnp.abs(contour_zeta[-1][-1] - contour_zeta[-1][0]), zeta_diff, rtol=1e-4
        ) or not jnp.isclose(
            jnp.abs(contour_theta[-1][-1] - contour_theta[-1][0]), theta_diff, rtol=1e-4
        ):
            warnings.warn(
                f"Detected a coil contour (coil index {j}) that may not be "
                "closed, this may lead to incorrect coils, "
                "check that the surface current potential contours do not contain "
                "any local maxima or window-pane-like structures,"
                " and that the current potential contours do not go across "
                "The edges of the zeta extent used for the plotting:"
                "the zeta=0 or zeta=2pi/NFP planes for helical coils or the"
                " zeta=-pi/NFP and zeta=2pi+pi/NFP planes, for modular coils. "
                "Use `show_plots=True` to visualize the contours.",
                UserWarning,
            )
            print(f"zeta diff = {jnp.abs(contour_zeta[-1][-1] - contour_zeta[-1][0])}")
            print(f"expected zeta diff = {zeta_diff}")
            print(
                f"theta diff = {jnp.abs(contour_theta[-1][-1] - contour_theta[-1][0])}"
            )
            print(f"expected theta diff = {theta_diff}")

        numCoils += 1
        if show_plots:
            plt.plot(contour_zeta[-1], contour_theta[-1], "-r", linewidth=1)
            if j > 0:
                plt.plot(contour_zeta[-1][-1], contour_theta[-1][-1], "sk")
            else:
                plt.plot(
                    contour_zeta[-1][-1],
                    contour_theta[-1][-1],
                    "sk",
                    label="start of contour",
                )
    if coil_type == "helical":
        # right now these are only over 1 FP
        # so must tile them s.t. they are full coils, by repeating them
        #  with a 2pi/NFP shift in zeta
        # and a -2pi*helicity shift in theta
        # we could alternatively wait until we are in real space and then
        # rotate the coils there, but this also works
        for i_contour in range(num_coils):
            # check if the contour is arranged with zeta=0 at the start
            # or at the end, easiest to do this tiling if we assume
            # the first index is at zeta=0
            zeta_starts_at_zero = (
                contour_zeta[i_contour][-1] > contour_zeta[i_contour][0]
            )
            orig_theta = contour_theta[i_contour]
            orig_zeta = contour_zeta[i_contour]
            if not zeta_starts_at_zero:
                # flip so that the contour starts at zeta=0
                orig_theta = jnp.flip(orig_theta)
                orig_zeta = jnp.flip(orig_zeta)
            orig_endpoint_theta = orig_theta[-1]

            # dont need last points here since we will shift the whole
            # curve over, and we know the last point must be
            # (zeta0+2pi/NFP, theta0+2pi*abs(helicity)),
            # so easiest to just not include them initially and shift whole curve
            orig_theta = jnp.atleast_1d(orig_theta[:-1])
            orig_zeta = jnp.atleast_1d(orig_zeta[:-1])

            contour_theta[i_contour] = jnp.atleast_1d(orig_theta)
            contour_zeta[i_contour] = jnp.atleast_1d(orig_zeta)

            theta_shift = -2 * np.pi * helicity

            zeta_shift = 2 * jnp.pi / nfp - orig_zeta[0]

            for i in range(1, nfp):
                contour_theta[i_contour] = jnp.concatenate(
                    [contour_theta[i_contour], orig_theta + theta_shift * i]
                )
                contour_zeta[i_contour] = jnp.concatenate(
                    [contour_zeta[i_contour], orig_zeta + zeta_shift * i]
                )
            contour_theta[i_contour] = jnp.append(
                contour_theta[i_contour],
                nfp * (orig_endpoint_theta - contour_theta[i_contour][0])
                + contour_theta[i_contour][0],
            )
            contour_zeta[i_contour] = jnp.append(contour_zeta[i_contour], 2 * jnp.pi)
    if show_plots:
        plt.legend()

    return contour_theta, contour_zeta


def _find_XYZ_points(
    theta_pts,
    zeta_pts,
    surface,
):
    contour_X = []
    contour_Y = []
    contour_Z = []

    for thetas, zetas in zip(theta_pts, zeta_pts):
        coords = surface.compute(
            "x",
            grid=Grid(
                jnp.vstack((jnp.zeros_like(thetas), thetas, zetas)).T,
                sort=False,
            ),
            basis="xyz",
        )["x"]
        contour_X.append(coords[:, 0])
        contour_Y.append(coords[:, 1])
        contour_Z.append(coords[:, 2])

    return contour_X, contour_Y, contour_Z


def _G_from_external_field(external_field, eq, external_field_grid):
    # calculate the portion of G provided by external field
    # by integrating external toroidal field along a curve of constant theta
    try:
        G_ext = external_field.G
    except AttributeError:
        curve_grid = LinearGrid(
            N=int(eq.NFP) * 50,
            theta=jnp.array(jnp.pi),  # does not matter which theta we choose
            rho=jnp.array(1.0),
            endpoint=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # ignore warning from unequal NFP for grid and basis,
            # as we don't know a-priori if the external field
            # shares the same discrete symmetry as the equilibrium,
            # so we will use a grid with NFP=1 to be safe
            curve_data = eq.compute(
                ["R", "phi", "Z", "e_zeta"],
                grid=curve_grid,
            )
            curve_coords = jnp.vstack(
                (curve_data["R"], curve_data["phi"], curve_data["Z"])
            ).T
            ext_field_along_curve = external_field.compute_magnetic_field(
                curve_coords, basis="rpz", source_grid=external_field_grid
            )
        # calculate covariant B_zeta = B dot e_zeta from external field
        ext_field_B_zeta = dot(ext_field_along_curve, curve_data["e_zeta"], axis=-1)

        # negative sign here because with REGCOIL convention, negative G makes
        # positive toroidal B
        G_ext = -jnp.sum(ext_field_B_zeta) * 2 * jnp.pi / curve_grid.num_nodes / mu_0
    return G_ext
