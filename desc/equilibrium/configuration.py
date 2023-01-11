"""Base class for Equilibrium."""

import copy
import numbers
import warnings
from abc import ABC

import numpy as np
from termcolor import colored

from desc.backend import jnp
from desc.basis import FourierZernikeBasis, fourier, zernike_radial
from desc.compute import compute as compute_fun
from desc.compute.utils import compress, get_params, get_profiles, get_transforms
from desc.geometry import (
    FourierRZCurve,
    FourierRZToroidalSurface,
    Surface,
    ZernikeRZToroidalSection,
)
from desc.grid import LinearGrid, QuadratureGrid
from desc.io import IOAble
from desc.profiles import PowerSeriesProfile, Profile, SplineProfile
from desc.utils import copy_coeffs

from .coords import compute_flux_coords, compute_theta_coords, is_nested, to_sfl
from .initial_guess import set_initial_guess


class _Configuration(IOAble, ABC):
    """Configuration is an abstract base class for equilibrium information.

    It contains information about a plasma state, including the
    shapes of flux surfaces and profile inputs. It can compute additional
    information, such as the magnetic field and plasma currents.

    Parameters
    ----------
    Psi : float (optional)
        total toroidal flux (in Webers) within LCFS. Default 1.0
    NFP : int (optional)
        number of field periods Default surface.NFP or 1
    L : int (optional)
        Radial resolution. Default 2*M for `spectral_indexing`==fringe, else M
    M : int (optional)
        Poloidal resolution. Default surface.M or 1
    N : int (optional)
        Toroidal resolution. Default surface.N or 0
    pressure : Profile or ndarray shape(k,2) (optional)
        Pressure profile or array of mode numbers and spectral coefficients.
        Default is a PowerSeriesProfile with zero pressure
    iota : Profile or ndarray shape(k,2) (optional)
        Rotational transform profile or array of mode numbers and spectral coefficients
    current : Profile or ndarray shape(k,2) (optional)
        Toroidal current profile or array of mode numbers and spectral coefficients
        Default is a PowerSeriesProfile with zero toroidal current
    surface: Surface or ndarray shape(k,5) (optional)
        Fixed boundary surface shape, as a Surface object or array of
        spectral mode numbers and coefficients of the form [l, m, n, R, Z].
        Default is a FourierRZToroidalSurface with major radius 10 and minor radius 1
    axis : Curve or ndarray shape(k,3) (optional)
        Initial guess for the magnetic axis as a Curve object or ndarray
        of mode numbers and spectral coefficients of the form [n, R, Z].
        Default is the centroid of the surface.
    sym : bool (optional)
        Whether to enforce stellarator symmetry. Default surface.sym or False.
    spectral_indexing : str (optional)
        Type of Zernike indexing scheme to use. Default ``'ansi'``

    """

    _io_attrs_ = [
        "_sym",
        "_R_sym",
        "_Z_sym",
        "_Psi",
        "_NFP",
        "_L",
        "_M",
        "_N",
        "_R_lmn",
        "_Z_lmn",
        "_L_lmn",
        "_R_basis",
        "_Z_basis",
        "_L_basis",
        "_surface",
        "_axis",
        "_pressure",
        "_iota",
        "_current",
        "_spectral_indexing",
        "_bdry_mode",
    ]

    def __init__(  # noqa: C901 - FIXME: break this up into simpler pieces
        self,
        Psi=1.0,
        NFP=None,
        L=None,
        M=None,
        N=None,
        pressure=None,
        iota=None,
        current=None,
        surface=None,
        axis=None,
        sym=None,
        spectral_indexing=None,
        **kwargs,
    ):

        assert spectral_indexing in [None, "ansi", "fringe",], (
            "spectral_indexing should be one of 'ansi', 'fringe', None, got "
            + f"{spectral_indexing}"
        )
        if spectral_indexing is None and hasattr(surface, "spectral_indexing"):
            self._spectral_indexing = surface.spectral_indexing
        elif spectral_indexing is None:
            self._spectral_indexing = "ansi"
        else:
            self._spectral_indexing = spectral_indexing

        assert isinstance(
            Psi, numbers.Real
        ), f"Psi should be a real integer or float, got {type(Psi)}"
        self._Psi = float(Psi)

        assert (NFP is None) or (
            isinstance(NFP, numbers.Real) and int(NFP) == NFP and NFP > 0
        ), f"NFP should be a positive integer, got {type(NFP)}"
        if NFP is not None:
            self._NFP = NFP
        elif hasattr(surface, "NFP"):
            self._NFP = surface.NFP
        elif hasattr(axis, "NFP"):
            self._NFP = axis.NFP
        else:
            self._NFP = 1

        assert sym in [
            None,
            True,
            False,
        ], f"sym should be one of True, False, None, got {sym}"
        if sym is None and hasattr(surface, "sym"):
            self._sym = surface.sym
        elif sym is None:
            self._sym = False
        else:
            self._sym = sym
        # stellarator symmetry for bases
        if self.sym:
            self._R_sym = "cos"
            self._Z_sym = "sin"
        else:
            self._R_sym = False
            self._Z_sym = False

        # resolution
        assert (L is None) or (
            isinstance(L, numbers.Real) and (L == int(L)) and (L >= 0)
        ), f"L should be a non-negative integer or None, got {L}"
        assert (M is None) or (
            isinstance(M, numbers.Real) and (M == int(M)) and (M >= 0)
        ), f"M should be a non-negative integer or None, got {M}"
        assert (N is None) or (
            isinstance(N, numbers.Real) and (N == int(N)) and (N >= 0)
        ), f"N should be a non-negative integer or None, got {N}"
        if N is not None:
            self._N = int(N)
        elif hasattr(surface, "N"):
            self._N = surface.N
        else:
            self._N = 0

        if M is not None:
            self._M = int(M)
        elif hasattr(surface, "M"):
            self._M = surface.M
        else:
            self._M = 1

        if L is not None:
            self._L = int(L)
        elif hasattr(surface, "L") and (surface.L > 0):
            self._L = surface.L
        else:
            self._L = self.M if (self.spectral_indexing == "ansi") else 2 * self.M

        # bases
        self._R_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._R_sym,
            spectral_indexing=self.spectral_indexing,
        )
        self._Z_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._Z_sym,
            spectral_indexing=self.spectral_indexing,
        )
        self._L_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._Z_sym,
            spectral_indexing=self.spectral_indexing,
        )

        # surface
        if surface is None:
            self._surface = FourierRZToroidalSurface(NFP=self.NFP, sym=self.sym)
            self._bdry_mode = "lcfs"
        elif isinstance(surface, Surface):
            self._surface = surface
            if isinstance(surface, FourierRZToroidalSurface):
                self._bdry_mode = "lcfs"
            if isinstance(surface, ZernikeRZToroidalSection):
                self._bdry_mode = "poincare"
        elif isinstance(surface, (np.ndarray, jnp.ndarray)):
            if np.all(surface[:, 0] == 0):
                self._bdry_mode = "lcfs"
            elif np.all(surface[:, 2] == 0):
                self._bdry_mode = "poincare"
            else:
                raise ValueError("boundary should either have l=0 or n=0")
            if self.bdry_mode == "lcfs":
                self._surface = FourierRZToroidalSurface(
                    surface[:, 3],
                    surface[:, 4],
                    surface[:, 1:3].astype(int),
                    surface[:, 1:3].astype(int),
                    self.NFP,
                    self.sym,
                )
            elif self.bdry_mode == "poincare":
                self._surface = ZernikeRZToroidalSection(
                    surface[:, 3],
                    surface[:, 4],
                    surface[:, :2].astype(int),
                    surface[:, :2].astype(int),
                    self.spectral_indexing,
                    self.sym,
                )
        else:
            raise TypeError("Got unknown surface type {}".format(surface))
        self._surface.change_resolution(self.L, self.M, self.N)

        # magnetic axis
        if isinstance(axis, FourierRZCurve):
            self._axis = axis
        elif isinstance(axis, (np.ndarray, jnp.ndarray)):
            self._axis = FourierRZCurve(
                axis[:, 1],
                axis[:, 2],
                axis[:, 0].astype(int),
                NFP=self.NFP,
                sym=self.sym,
                name="axis",
            )
        elif axis is None:  # use the center of surface
            # TODO: make this method of surface, surface.get_axis()?
            if isinstance(self.surface, FourierRZToroidalSurface):
                self._axis = FourierRZCurve(
                    R_n=self.surface.R_lmn[
                        np.where(self.surface.R_basis.modes[:, 1] == 0)
                    ],
                    Z_n=self.surface.Z_lmn[
                        np.where(self.surface.Z_basis.modes[:, 1] == 0)
                    ],
                    modes_R=self.surface.R_basis.modes[
                        np.where(self.surface.R_basis.modes[:, 1] == 0)[0], -1
                    ],
                    modes_Z=self.surface.Z_basis.modes[
                        np.where(self.surface.Z_basis.modes[:, 1] == 0)[0], -1
                    ],
                    NFP=self.NFP,
                )
            elif isinstance(self.surface, ZernikeRZToroidalSection):
                # FIXME: include m=0 l!=0 modes
                self._axis = FourierRZCurve(
                    R_n=self.surface.R_lmn[
                        np.where(
                            (self.surface.R_basis.modes[:, 0] == 0)
                            & (self.surface.R_basis.modes[:, 1] == 0)
                        )
                    ].sum(),
                    Z_n=self.surface.Z_lmn[
                        np.where(
                            (self.surface.Z_basis.modes[:, 0] == 0)
                            & (self.surface.Z_basis.modes[:, 1] == 0)
                        )
                    ].sum(),
                    modes_R=[0],
                    modes_Z=[0],
                    NFP=self.NFP,
                )
        else:
            raise TypeError("Got unknown axis type {}".format(axis))

        # profiles
        self._pressure = None
        self._iota = None
        self._current = None

        # pressure
        if isinstance(pressure, Profile):
            self._pressure = pressure
        elif isinstance(pressure, (np.ndarray, jnp.ndarray)):
            self._pressure = PowerSeriesProfile(
                modes=pressure[:, 0], params=pressure[:, 1], name="pressure"
            )
        elif pressure is None:
            self._pressure = PowerSeriesProfile(
                modes=np.array([0]), params=np.array([0]), name="pressure"
            )
        else:
            raise TypeError("Got unknown pressure profile {}".format(pressure))

        # default profile
        if iota is None and current is None:
            self._current = PowerSeriesProfile(
                modes=np.array([0]), params=np.array([0]), name="current"
            )
        elif iota is not None and current is not None:
            raise ValueError("Cannot specify both iota and current profiles.")

        # iota
        if isinstance(iota, Profile):
            self.iota = iota
        elif isinstance(iota, (np.ndarray, jnp.ndarray)):
            self._iota = PowerSeriesProfile(
                modes=iota[:, 0], params=iota[:, 1], name="iota"
            )
        elif iota is not None:
            raise TypeError("Got unknown iota profile {}".format(iota))

        # current
        if isinstance(current, Profile):
            self.current = current
        elif isinstance(current, (np.ndarray, jnp.ndarray)):
            self._current = PowerSeriesProfile(
                modes=current[:, 0], params=current[:, 1], name="current"
            )
        elif current is not None:
            raise TypeError("Got unknown current profile {}".format(current))

        # ensure profiles have the right resolution
        for profile in ["pressure", "iota", "current"]:
            p = getattr(self, profile)
            if hasattr(p, "change_resolution"):
                p.change_resolution(max(p.basis.L, self.L))
            if isinstance(p, PowerSeriesProfile) and p.sym != "even":
                warnings.warn(
                    colored(f"{profile} profile is not an even power series.", "yellow")
                )

        # ensure number of field periods agree before setting guesses
        eq_NFP = self.NFP
        surf_NFP = self.surface.NFP if hasattr(self.surface, "NFP") else self.NFP
        axis_NFP = self._axis.NFP

        if not (eq_NFP == surf_NFP == axis_NFP):
            raise ValueError(
                "Unequal number of field periods for equilirium "
                + f"{eq_NFP}, surface {surf_NFP}, and axis {axis_NFP}"
            )

        # make sure symmetry agrees
        assert (
            self.sym == self.surface.sym
        ), "Surface and Equilibrium must have the same symmetry"

        self._R_lmn = np.zeros(self.R_basis.num_modes)
        self._Z_lmn = np.zeros(self.Z_basis.num_modes)
        self._L_lmn = np.zeros(self.L_basis.num_modes)
        self.set_initial_guess()
        if "R_lmn" in kwargs:
            self.R_lmn = kwargs.pop("R_lmn")
        if "Z_lmn" in kwargs:
            self.Z_lmn = kwargs.pop("Z_lmn")
        if "L_lmn" in kwargs:
            self.L_lmn = kwargs.pop("L_lmn")

    # TODO: allow user to pass in arrays for surface, axis? or R_lmn etc?
    # TODO: make this kwargs instead?
    def _set_up(self):
        """Set unset attributes after loading.

        To ensure object has all properties needed for current DESC version.
        Allows for backwards-compatibility with equilibria saved/ran with older
        DESC versions.
        """
        for attribute in self._io_attrs_:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

    def set_initial_guess(self, *args):
        """Set the initial guess for the flux surfaces, eg R_lmn, Z_lmn, L_lmn.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to initialize
        args :
            either:
              - No arguments, in which case eq.surface will be scaled for the guess.
              - Another Surface object, which will be scaled to generate the guess.
                Optionally a Curve object may also be supplied for the magnetic axis.
              - Another Equilibrium, whose flux surfaces will be used.
              - File path to a VMEC or DESC equilibrium, which will be loaded and used.
              - Grid and 2-3 ndarrays, specifying the flux surface locations (R, Z, and
                optionally lambda) at fixed flux coordinates. All arrays should have the
                same length. Optionally, an ndarray of shape(k,3) may be passed instead
                of a grid.

        Examples
        --------
        Use existing equil.surface and scales down for guess:

        >>> equil.set_initial_guess()

        Use supplied Surface and scales down for guess. Assumes axis is centroid
        of user supplied surface:

        >>> equil.set_initial_guess(surface)

        Optionally, an interior surface may be scaled by giving the surface a
        flux label:

        >>> surf = FourierRZToroidalSurface(rho=0.7)
        >>> equil.set_initial_guess(surf)

        Use supplied Surface and a supplied Curve for axis and scales between
        them for guess:

        >>> equil.set_initial_guess(surface, curve)

        Use the flux surfaces from an existing Equilibrium:

        >>> equil.set_initial_guess(equil2)

        Use flux surfaces from existing Equilibrium or VMEC output stored on disk:

        >>> equil.set_initial_guess(path_to_saved_DESC_or_VMEC_output)

        Use flux surfaces specified by points:
        nodes should either be a Grid or an ndarray, shape(k,3) giving the locations
        in rho, theta, zeta coordinates. R, Z, and optionally lambda should be
        array-like, shape(k,) giving the corresponding real space coordinates

        >>> equil.set_initial_guess(nodes, R, Z, lambda)

        """
        set_initial_guess(self, *args)

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this equilibrium."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new

    def change_resolution(self, L=None, M=None, N=None, NFP=None, *args, **kwargs):
        """Set the spectral resolution.

        Parameters
        ----------
        L : int
            maximum radial zernike mode number
        M : int
            maximum poloidal fourier mode number
        N : int
            maximum toroidal fourier mode number
        NFP : int
            Number of field periods.

        """
        L_change = M_change = N_change = NFP_change = False
        if L is not None and L != self.L:
            L_change = True
            self._L = L
        if M is not None and M != self.M:
            M_change = True
            self._M = M
        if N is not None and N != self.N:
            N_change = True
            self._N = N
        if NFP is not None and NFP != self.NFP:
            NFP_change = True
            self._NFP = NFP

        if not np.any([L_change, M_change, N_change, NFP_change]):
            return

        old_modes_R = self.R_basis.modes
        old_modes_Z = self.Z_basis.modes
        old_modes_L = self.L_basis.modes

        self.R_basis.change_resolution(self.L, self.M, self.N, self.NFP)
        self.Z_basis.change_resolution(self.L, self.M, self.N, self.NFP)
        self.L_basis.change_resolution(self.L, self.M, self.N, self.NFP)

        if L_change and hasattr(self.pressure, "change_resolution"):
            self.pressure.change_resolution(L=max(L, self.pressure.basis.L))
        if L_change and hasattr(self.iota, "change_resolution"):
            self.iota.change_resolution(L=max(L, self.iota.basis.L))
        if L_change and hasattr(self.current, "change_resolution"):
            self.current.change_resolution(L=max(L, self.current.basis.L))

        self.surface.change_resolution(self.L, self.M, self.N, NFP=self.NFP)

        self._R_lmn = copy_coeffs(self.R_lmn, old_modes_R, self.R_basis.modes)
        self._Z_lmn = copy_coeffs(self.Z_lmn, old_modes_Z, self.Z_basis.modes)
        self._L_lmn = copy_coeffs(self.L_lmn, old_modes_L, self.L_basis.modes)

    def get_surface_at(self, rho=None, theta=None, zeta=None):
        """Return a representation for a given coordinate surface.

        Parameters
        ----------
        rho, theta, zeta : float or None
            radial, poloidal, or toroidal coordinate for the surface. Only
            one may be specified.

        Returns
        -------
        surf : Surface
            object representing the given surface, either a FourierRZToroidalSurface
            for surfaces of constant rho, or a ZernikeRZToroidalSection for
            surfaces of constant zeta.

        """
        if (rho is not None) and (theta is None) and (zeta is None):
            assert (rho >= 0) and (rho <= 1)
            surface = FourierRZToroidalSurface(sym=self.sym, NFP=self.NFP, rho=rho)
            surface.change_resolution(self.M, self.N)

            AR = np.zeros((surface.R_basis.num_modes, self.R_basis.num_modes))
            AZ = np.zeros((surface.Z_basis.num_modes, self.Z_basis.num_modes))

            for i, (l, m, n) in enumerate(self.R_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.R_basis.modes[:, 1] == m,
                        surface.R_basis.modes[:, 2] == n,
                    )
                )
                AR[j, i] = zernike_radial(rho, l, m)

            for i, (l, m, n) in enumerate(self.Z_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.Z_basis.modes[:, 1] == m,
                        surface.Z_basis.modes[:, 2] == n,
                    )
                )
                AZ[j, i] = zernike_radial(rho, l, m)
            Rb = AR @ self.R_lmn
            Zb = AZ @ self.Z_lmn
            surface.R_lmn = Rb
            surface.Z_lmn = Zb
            surface.grid = LinearGrid(
                rho=rho, M=2 * surface.M, N=2 * surface.N, endpoint=True, NFP=self.NFP
            )
            return surface

        if (rho is None) and (theta is None) and (zeta is not None):
            assert (zeta >= 0) and (zeta <= 2 * np.pi)
            surface = ZernikeRZToroidalSection(sym=self.sym, zeta=zeta)
            surface.change_resolution(self.L, self.M)

            AR = np.zeros((surface.R_basis.num_modes, self.R_basis.num_modes))
            AZ = np.zeros((surface.Z_basis.num_modes, self.Z_basis.num_modes))

            for i, (l, m, n) in enumerate(self.R_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.R_basis.modes[:, 0] == l,
                        surface.R_basis.modes[:, 1] == m,
                    )
                )
                AR[j, i] = fourier(zeta, n, self.NFP)

            for i, (l, m, n) in enumerate(self.Z_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.Z_basis.modes[:, 0] == l,
                        surface.Z_basis.modes[:, 1] == m,
                    )
                )
                AZ[j, i] = fourier(zeta, n, self.NFP)
            Rb = AR @ self.R_lmn
            Zb = AZ @ self.Z_lmn
            surface.R_lmn = Rb
            surface.Z_lmn = Zb
            surface.grid = LinearGrid(
                L=2 * surface.L,
                M=2 * surface.M,
                zeta=zeta,
                endpoint=True,
            )
            return surface
        if (rho is None) and (theta is not None) and (zeta is None):
            raise NotImplementedError(
                "Constant theta surfaces have not been implemented yet"
            )
        else:
            raise ValueError(
                "Only one coordinate can be specified, got {}, {}, {}".format(
                    rho, theta, zeta
                )
            )

    def get_profile(self, name, grid=None, **kwargs):
        """Return a SplineProfile of the desired quantity.

        Parameters
        ----------
        name : str
            Name of the quantity to compute.
        grid : Grid, optional
            Grid of coordinates to evaluate at. Defaults to the quadrature grid.
            Note profile will only be a function of the radial coordinate.

        Returns
        -------
        profile : SplineProfile
            Radial profile of the desired quantity.

        """
        if grid is None:
            grid = QuadratureGrid(self.L_grid, self.M_grid, self.N_grid, self.NFP)
        data = self.compute(name, grid=grid, **kwargs)
        x = data[name]
        x = compress(grid, x, surface_label="rho")
        return SplineProfile(
            x, grid.nodes[grid.unique_rho_idx, 0], grid=grid, name=name
        )

    @property
    def surface(self):
        """Surface: Geometric surface defining boundary conditions."""
        return self._surface

    @surface.setter
    def surface(self, new):
        if isinstance(new, Surface):
            assert (
                self.sym == new.sym
            ), "Surface and Equilibrium must have the same symmetry"
            new.change_resolution(self.L, self.M, self.N)
            self._surface = new
        else:
            raise TypeError(
                f"surfaces should be of type Surface or a subclass, got {new}"
            )

    @property
    def spectral_indexing(self):
        """str: Type of indexing used for the spectral basis."""
        return self._spectral_indexing

    @property
    def sym(self):
        """bool: Whether this equilibrium is stellarator symmetric."""
        return self._sym

    @property
    def bdry_mode(self):
        """str: Method for specifying boundary condition."""
        return self._bdry_mode

    @property
    def Psi(self):
        """float: Total toroidal flux within the last closed flux surface in Webers."""
        return self._Psi

    @Psi.setter
    def Psi(self, Psi):
        self._Psi = float(Psi)

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @NFP.setter
    def NFP(self, NFP):
        assert (
            isinstance(NFP, numbers.Real) and (NFP == int(NFP)) and (NFP > 0)
        ), f"NFP should be a positive integer, got {type(NFP)}"
        self.change_resolution(NFP=NFP)

    @property
    def L(self):
        """int: Maximum radial mode number."""
        return self._L

    @L.setter
    def L(self, L):
        assert (
            isinstance(L, numbers.Real) and (L == int(L)) and (L >= 0)
        ), f"L should be a non-negative integer got {L}"
        self.change_resolution(L=L)

    @property
    def M(self):
        """int: Maximum poloidal fourier mode number."""
        return self._M

    @M.setter
    def M(self, M):
        assert (
            isinstance(M, numbers.Real) and (M == int(M)) and (M >= 0)
        ), f"M should be a non-negative integer got {M}"
        self.change_resolution(M=M)

    @property
    def N(self):
        """int: Maximum toroidal fourier mode number."""
        return self._N

    @N.setter
    def N(self, N):
        assert (
            isinstance(N, numbers.Real) and (N == int(N)) and (N >= 0)
        ), f"N should be a non-negative integer got {N}"
        self.change_resolution(N=N)

    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients of R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, R_lmn):
        self._R_lmn[:] = R_lmn

    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients of Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, Z_lmn):
        self._Z_lmn[:] = Z_lmn

    @property
    def L_lmn(self):
        """ndarray: Spectral coefficients of lambda."""
        return self._L_lmn

    @L_lmn.setter
    def L_lmn(self, L_lmn):
        self._L_lmn[:] = L_lmn

    @property
    def Rb_lmn(self):
        """ndarray: Spectral coefficients of R at the boundary."""
        return self.surface.R_lmn

    @Rb_lmn.setter
    def Rb_lmn(self, Rb_lmn):
        self.surface.R_lmn = Rb_lmn

    @property
    def Zb_lmn(self):
        """ndarray: Spectral coefficients of Z at the boundary."""
        return self.surface.Z_lmn

    @Zb_lmn.setter
    def Zb_lmn(self, Zb_lmn):
        self.surface.Z_lmn = Zb_lmn

    @property
    def Ra_n(self):
        """ndarray: R coefficients for axis Fourier series."""
        return self.axis.R_n

    @property
    def Za_n(self):
        """ndarray: Z coefficients for axis Fourier series."""
        return self.axis.Z_n

    @property
    def axis(self):
        """Curve: object representing the magnetic axis."""
        # value of Zernike polynomials at rho=0 for unique radial modes (+/-1)
        sign_l = np.atleast_2d(((np.arange(0, self.L + 1, 2) / 2) % 2) * -2 + 1).T
        # indices where m=0
        idx0_R = np.where(self.R_basis.modes[:, 1] == 0)[0]
        idx0_Z = np.where(self.Z_basis.modes[:, 1] == 0)[0]
        # indices where l=0 & m=0
        idx00_R = np.where((self.R_basis.modes[:, :2] == [0, 0]).all(axis=1))[0]
        idx00_Z = np.where((self.Z_basis.modes[:, :2] == [0, 0]).all(axis=1))[0]
        # this reshaping assumes the FourierZernike bases are sorted
        R_n = np.sum(
            sign_l * np.reshape(self.R_lmn[idx0_R], (-1, idx00_R.size), order="F"),
            axis=0,
        )
        modes_R = self.R_basis.modes[idx00_R, 2]
        if len(idx00_Z):
            Z_n = np.sum(
                sign_l * np.reshape(self.Z_lmn[idx0_Z], (-1, idx00_Z.size), order="F"),
                axis=0,
            )
            modes_Z = self.Z_basis.modes[idx00_Z, 2]
        else:  # catch cases such as axisymmetry with stellarator symmetry
            Z_n = 0
            modes_Z = 0
        self._axis = FourierRZCurve(R_n, Z_n, modes_R, modes_Z, NFP=self.NFP)
        return self._axis

    @property
    def pressure(self):
        """Profile: Pressure profile."""
        return self._pressure

    @pressure.setter
    def pressure(self, new):
        if isinstance(new, Profile):
            self._pressure = new
        else:
            raise TypeError(
                f"pressure profile should be of type Profile or a subclass, got {new} "
            )

    @property
    def p_l(self):
        """ndarray: Coefficients of pressure profile."""
        return self.pressure.params

    @p_l.setter
    def p_l(self, p_l):
        self.pressure.params = p_l

    @property
    def iota(self):
        """Profile: Rotational transform (iota) profile."""
        return self._iota

    @iota.setter
    def iota(self, new):
        if isinstance(new, Profile) or (new is None):
            self._iota = new
        else:
            raise TypeError(
                f"iota profile should be of type Profile or a subclass, got {new} "
            )

    @property
    def i_l(self):
        """ndarray: Coefficients of iota profile."""
        return np.empty(0) if self.iota is None else self.iota.params

    @i_l.setter
    def i_l(self, i_l):
        if self.iota is None:
            raise ValueError(
                "Attempt to set rotational transform on an equilibrium "
                + "with fixed toroidal current"
            )
        self.iota.params = i_l

    @property
    def current(self):
        """Profile: Toroidal current profile (I)."""
        return self._current

    @current.setter
    def current(self, new):
        if isinstance(new, Profile) or (new is None):
            self._current = new
        else:
            raise TypeError(
                f"current profile should be of type Profile or a subclass, got {new} "
            )

    @property
    def c_l(self):
        """ndarray: Coefficients of current profile."""
        return np.empty(0) if self.current is None else self.current.params

    @c_l.setter
    def c_l(self, c_l):
        if self.current is None:
            raise ValueError(
                "Attempt to set toroidal current on an equilibrium with "
                + "fixed rotational transform"
            )
        self.current.params = c_l

    @property
    def R_basis(self):
        """FourierZernikeBasis: Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """FourierZernikeBasis: Spectral basis for Z."""
        return self._Z_basis

    @property
    def L_basis(self):
        """FourierZernikeBasis: Spectral basis for lambda."""
        return self._L_basis

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        profiles=None,
        data=None,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid, optional
            Grid of coordinates to evaluate at. Defaults to the quadrature grid.
        params : dict of ndarray
            Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
            Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from grid
        profiles : dict of Profile
            Profile objects for pressure, iota, current, etc. Defaults to attributes
            of self
        data : dict of ndarray
            Data computed so far, generally output from other compute functions

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        # TODO: default to returning just desired qty? options to return_all?
        # TODO: use get_params method? need to break up compute functions first
        if grid is None:
            grid = QuadratureGrid(self.L_grid, self.M_grid, self.N_grid, self.NFP)
        if params is None:
            params = get_params(names, eq=self)
        if profiles is None:
            profiles = get_profiles(names, eq=self, grid=grid)
        if transforms is None:
            transforms = get_transforms(names, eq=self, grid=grid, **kwargs)

        data = compute_fun(
            names,
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
            **kwargs,
        )
        return data

    def compute_theta_coords(self, flux_coords, L_lmn=None, tol=1e-6, maxiter=20):
        """Find geometric theta for given straight field line theta.

        Parameters
        ----------
        flux_coords : ndarray, shape(k,3)
            2d array of flux coordinates [rho,theta*,zeta]. Each row is a different
            coordinate.
        L_lmn : ndarray
            spectral coefficients for lambda. Defaults to eq.L_lmn
        tol : float
            Stopping tolerance.
        maxiter : int > 0
            maximum number of Newton iterations

        Returns
        -------
        coords : ndarray, shape(k,3)
            coordinates [rho,theta,zeta]. If Newton method doesn't converge for
            a given coordinate nan will be returned for those values

        """
        return compute_theta_coords(self, flux_coords, L_lmn, tol, maxiter)

    def compute_flux_coords(
        self, real_coords, R_lmn=None, Z_lmn=None, tol=1e-6, maxiter=20, rhomin=1e-6
    ):
        """Find the (rho, theta, zeta) that correspond to given (R, phi, Z).

        Parameters
        ----------
        real_coords : ndarray, shape(k,3)
            2D array of real space coordinates [R,phi,Z]. Each row is a different
            coordinate.
        R_lmn, Z_lmn : ndarray
            spectral coefficients for R and Z. Defaults to eq.R_lmn, eq.Z_lmn
        tol : float
            Stopping tolerance. Iterations stop when sqrt((R-Ri)**2 + (Z-Zi)**2) < tol
        maxiter : int > 0
            maximum number of Newton iterations
        rhomin : float
            minimum allowable value of rho (to avoid singularity at rho=0)

        Returns
        -------
        flux_coords : ndarray, shape(k,3)
            flux coordinates [rho,theta,zeta]. If Newton method doesn't converge for
            a given coordinate (often because it is outside the plasma boundary),
            nan will be returned for those values

        """
        return compute_flux_coords(
            self, real_coords, R_lmn, Z_lmn, tol, maxiter, rhomin
        )

    def is_nested(self, grid=None, R_lmn=None, Z_lmn=None, msg=None):
        """Check that an equilibrium has properly nested flux surfaces in a plane.

        Does so by checking coordianate Jacobian (sqrt(g)) sign.
        If coordinate Jacobian switches sign somewhere in the volume, this
        indicates that it is zero at some point, meaning surfaces are touching and
        the equilibrium is not nested.

        NOTE: If grid resolution used is too low, or the solution is just barely
        unnested, this function may fail to return the correct answer.

        Parameters
        ----------
        grid  :  Grid, optional
            Grid on which to evaluate the coordinate Jacobian and check for the sign.
            (Default to QuadratureGrid with eq's current grid resolutions)
        R_lmn, Z_lmn : ndarray, optional
            spectral coefficients for R and Z. Defaults to eq.R_lmn, eq.Z_lmn
        msg : {None, "auto", "manual"}
            Warning to throw if unnested.

        Returns
        -------
        is_nested : bool
            whether the surfaces are nested

        """
        return is_nested(self, grid, R_lmn, Z_lmn, msg)

    def to_sfl(
        self,
        L=None,
        M=None,
        N=None,
        L_grid=None,
        M_grid=None,
        N_grid=None,
        rcond=None,
        copy=False,
    ):
        """Transform this equilibrium to use straight field line coordinates.

        Uses a least squares fit to find FourierZernike coefficients of R, Z, Rb, Zb
        with respect to the straight field line coordinates, rather than the boundary
        coordinates. The new lambda value will be zero.

        NOTE: Though the converted equilibrium will have the same flux surfaces,
        the force balance error will likely be higher than the original equilibrium.

        Parameters
        ----------
        L : int, optional
            radial resolution to use for SFL equilibrium. Default = 1.5*eq.L
        M : int, optional
            poloidal resolution to use for SFL equilibrium. Default = 1.5*eq.M
        N : int, optional
            toroidal resolution to use for SFL equilibrium. Default = 1.5*eq.N
        L_grid : int, optional
            radial spatial resolution to use for fit to new basis. Default = 2*L
        M_grid : int, optional
            poloidal spatial resolution to use for fit to new basis. Default = 2*M
        N_grid : int, optional
            toroidal spatial resolution to use for fit to new basis. Default = 2*N
        rcond : float, optional
            cutoff for small singular values in least squares fit.
        copy : bool, optional
            Whether to update the existing equilibrium or make a copy (Default).

        Returns
        -------
        eq_sfl : Equilibrium
            Equilibrium transformed to a straight field line coordinate representation.

        """
        return to_sfl(self, L, M, N, L_grid, M_grid, N_grid, rcond, copy)
