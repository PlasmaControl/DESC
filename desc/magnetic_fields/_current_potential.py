"""Magnetic field due to sheet current on a winding surface."""

import numpy as np

from desc.backend import fori_loop, jnp
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.compute.utils import _compute as compute_fun
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.optimizable import Optimizable, optimizable_parameter
from desc.utils import copy_coeffs, errorif, setdefault, warnif

from ._core import _MagneticField, biot_savart_general


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
        theta,zeta are poloidal and toroidal angles on the surface
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
        source_grid = source_grid or LinearGrid(
            M=30 + 2 * self.M,
            N=30 + 2 * self.N,
            NFP=self.NFP,
        )
        return _compute_magnetic_field_from_CurrentPotentialField(
            field=self,
            coords=coords,
            params=params,
            basis=basis,
            source_grid=source_grid,
            transforms=transforms,
        )

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


class FourierCurrentPotentialField(
    _MagneticField, FourierRZToroidalSurface, Optimizable
):
    """Magnetic field due to a surface current potential on a toroidal surface.

    Surface current K is assumed given by

    K = n x ∇ Φ

    Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    where:

        - n is the winding surface unit normal.
        - Phi is the current potential function, which is a function of theta and zeta,
          and is given as a secular linear term in theta/zeta and a double Fourier
          series in theta/zeta.

    This function then uses biot-savart to find the B field from this current
    density K on the surface.

    Parameters
    ----------
    Phi_mn : ndarray
        Fourier coefficients of the double FourierSeries part of the current potential.
    modes_Phi : array-like, shape(k,2)
        Poloidal and Toroidal mode numbers corresponding to passed-in Phi_mn
        coefficients.
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
        source_grid = source_grid or LinearGrid(
            M=30 + 2 * max(self.M, self.M_Phi),
            N=30 + 2 * max(self.N, self.N_Phi),
            NFP=self.NFP,
        )
        return _compute_magnetic_field_from_CurrentPotentialField(
            field=self,
            coords=coords,
            params=params,
            basis=basis,
            source_grid=source_grid,
            transforms=transforms,
        )

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


def _compute_magnetic_field_from_CurrentPotentialField(
    field, coords, source_grid, params=None, basis="rpz", transforms=None
):
    """Compute magnetic field at a set of points.

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


    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field at specified points

    """
    assert basis.lower() in ["rpz", "xyz"]
    coords = jnp.atleast_2d(jnp.asarray(coords))
    if basis == "rpz":
        coords = rpz2xyz(coords)

    # compute surface current, and store grid quantities
    # needed for integration in class
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    if not params or not transforms:
        data = field.compute(
            ["K", "x"],
            grid=source_grid,
            basis="xyz",
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
            basis="xyz",
        )

    _rs = xyz2rpz(data["x"])
    _K = xyz2rpz_vec(data["K"], phi=source_grid.nodes[:, 2])

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
        fj = biot_savart_general(
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
