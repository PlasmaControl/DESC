import numpy as np
import os
import copy
import numbers

from termcolor import colored
from abc import ABC
from shapely.geometry import LineString, MultiLineString
from inspect import signature

from desc.backend import jnp, jit, put, while_loop
from desc.io import IOAble, load
from desc.utils import copy_coeffs, Index, unpack_state
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.transform import Transform
from desc.profiles import Profile, PowerSeriesProfile
from desc.geometry import (
    FourierRZToroidalSurface,
    ZernikeRZToroidalSection,
    FourierRZCurve,
    Surface,
)
from desc.basis import DoubleFourierSeries, FourierZernikeBasis, fourier, zernike_radial
import desc.compute as compute_funs
from desc.compute import arg_order, data_index


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
        Default is a PowerSeriesProfile with zero rotational transform
    surface: Surface or ndarray shape(k,5) (optional)
        Fixed boundary surface shape, as a Surface object or array of
        spectral mode numbers and coefficients of the form [l, m, n, R, Z].
        Default is a FourierRZToroidalSurface with major radius 10 and
        minor radius 1
    axis : Curve or ndarray shape(k,3) (optional)
        Initial guess for the magnetic axis as a Curve object or ndarray
        of mode numbers and spectral coefficints of the form [n, R, Z].
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
        "_spectral_indexing",
        "_bdry_mode",
    ]

    def __init__(
        self,
        Psi=1.0,
        NFP=None,
        L=None,
        M=None,
        N=None,
        pressure=None,
        iota=None,
        surface=None,
        axis=None,
        sym=None,
        spectral_indexing=None,
        **kwargs,
    ):

        assert spectral_indexing in [None, "ansi", "fringe",], (
            f"spectral_indexing should be one of 'ansi', 'fringe', None, got "
            + "{spectral_indexing}"
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

        # surface and axis
        if surface is None:
            self._surface = FourierRZToroidalSurface(NFP=self.NFP)
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

        # make sure field periods agree
        eqNFP = self.NFP
        surfNFP = self.surface.NFP if hasattr(self.surface, "NFP") else self.NFP
        axNFP = self.axis.NFP
        if not (eqNFP == surfNFP == axNFP):
            raise ValueError(
                "Unequal number of field periods for equilirium "
                + f"{eqNFP}, surface {surfNFP}, and axis {axNFP}"
            )

        # profiles
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

        if isinstance(iota, Profile):
            self.iota = iota
        elif isinstance(iota, (np.ndarray, jnp.ndarray)):
            self._iota = PowerSeriesProfile(
                modes=iota[:, 0], params=iota[:, 1], name="iota"
            )
        elif iota is None:
            self._iota = PowerSeriesProfile(
                modes=np.array([0]), params=np.array([0]), name="iota"
            )
        else:
            raise TypeError("Got unknown iota profile {}".format(iota))

        # keep track of where it came from
        self._parent = None
        self._children = []

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
    def set_initial_guess(self, *args):
        """Set the initial guess for the flux surfaces, eg R_lmn, Z_lmn, L_lmn.

        Parameters
        ----------
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

        Optionally, an interior surface may be scaled by giving the surface a flux label:

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
        nargs = len(args)
        if nargs > 4:
            raise ValueError(
                "set_initial_guess should be called with 4 or fewer arguments."
            )
        if nargs == 0:
            if hasattr(self, "_surface"):
                # use whatever surface is already assigned
                if hasattr(self, "_axis"):
                    axisR = np.array([self.axis.R_basis.modes[:, -1], self.axis.R_n]).T
                    axisZ = np.array([self.axis.Z_basis.modes[:, -1], self.axis.Z_n]).T
                else:
                    axisR = None
                    axisZ = None
                self.R_lmn = self._initial_guess_surface(
                    self.R_basis, self.Rb_lmn, self.surface.R_basis, axisR
                )
                self.Z_lmn = self._initial_guess_surface(
                    self.Z_basis, self.Zb_lmn, self.surface.Z_basis, axisZ
                )
            else:
                raise ValueError(
                    "set_initial_guess called with no arguments, "
                    + "but no surface is assigned."
                )
        else:  # nargs > 0
            if isinstance(args[0], Surface):
                surface = args[0]
                if nargs > 1:
                    if isinstance(args[1], FourierRZCurve):
                        axis = args[1]
                        axisR = np.array([axis.R_basis.modes[:, -1], axis.R_n]).T
                        axisZ = np.array([axis.Z_basis.modes[:, -1], axis.Z_n]).T
                    else:
                        raise TypeError(
                            "Don't know how to initialize from object type {}".format(
                                type(args[1])
                            )
                        )
                else:
                    axisR = None
                    axisZ = None
                coord = surface.rho if hasattr(surface, "rho") else None
                self.R_lmn = self._initial_guess_surface(
                    self.R_basis,
                    surface.R_lmn,
                    surface.R_basis,
                    axisR,
                    coord=coord,
                )
                self.Z_lmn = self._initial_guess_surface(
                    self.Z_basis,
                    surface.Z_lmn,
                    surface.Z_basis,
                    axisZ,
                    coord=coord,
                )
            elif isinstance(args[0], _Configuration):
                eq = args[0]
                if nargs > 1:
                    raise ValueError(
                        "set_initial_guess got unknown additional argument {}.".format(
                            args[1]
                        )
                    )
                self.R_lmn = copy_coeffs(eq.R_lmn, eq.R_basis.modes, self.R_basis.modes)
                self.Z_lmn = copy_coeffs(eq.Z_lmn, eq.Z_basis.modes, self.Z_basis.modes)
                self.L_lmn = copy_coeffs(eq.L_lmn, eq.L_basis.modes, self.L_basis.modes)
            elif isinstance(args[0], (str, os.PathLike)):
                # from file
                path = args[0]
                file_format = None
                if nargs > 1:
                    if isinstance(args[1], str):
                        file_format = args[1]
                    else:
                        raise ValueError(
                            "set_initial_guess got unknown additional argument "
                            + "{}.".format(args[1])
                        )
                try:  # is it desc?
                    eq = load(path, file_format)
                except:
                    try:  # maybe its vmec
                        from desc.vmec import VMECIO

                        eq = VMECIO.load(path)
                    except:  # its neither
                        raise ValueError(
                            "Could not load equilibrium from path {}, ".format(path)
                            + "please make sure it is a valid DESC or VMEC equilibrium."
                        )
                if not isinstance(eq, _Configuration):
                    if hasattr(eq, "equilibria"):  # its a family!
                        eq = eq[-1]
                    else:
                        raise TypeError(
                            "Cannot initialize equilibrium from loaded object of type "
                            + "{}".format(type(eq))
                        )
                self.R_lmn = copy_coeffs(eq.R_lmn, eq.R_basis.modes, self.R_basis.modes)
                self.Z_lmn = copy_coeffs(eq.Z_lmn, eq.Z_basis.modes, self.Z_basis.modes)
                self.L_lmn = copy_coeffs(eq.L_lmn, eq.L_basis.modes, self.L_basis.modes)

            elif nargs > 2:  # assume we got nodes and ndarray of points
                grid = args[0]
                R = args[1]
                self.R_lmn = self._initial_guess_points(grid, R, self.R_basis)
                Z = args[2]
                self.Z_lmn = self._initial_guess_points(grid, Z, self.Z_basis)
                if nargs > 3:
                    lmbda = args[3]
                    self.L_lmn = self._initial_guess_points(grid, lmbda, self.L_basis)
                else:
                    self.L_lmn = jnp.zeros(self.L_basis.num_modes)

            else:
                raise ValueError(
                    "Can't initialize equilibrium from args {}.".format(args)
                )

    def _initial_guess_surface(
        self, x_basis, b_lmn, b_basis, axis=None, mode=None, coord=None
    ):
        """Create an initial guess from the boundary coefficients and magnetic axis guess.

        Parameters
        ----------
        x_basis : FourierZernikeBais
            basis of the flux surfaces (for R, Z, or Lambda).
        b_lmn : ndarray, shape(b_basis.num_modes,)
            vector of boundary coefficients associated with b_basis.
        b_basis : Basis
            basis of the boundary surface (for Rb or Zb)
        axis : ndarray, shape(num_modes,2)
            coefficients of the magnetic axis. axis[i, :] = [n, x0].
            Only used for 'lcfs' boundary mode. Defaults to m=0 modes of boundary
        mode : str
            One of 'lcfs', 'poincare'.
            Whether the boundary condition is specified by the last closed flux surface
            (rho=1) or the Poincare section (zeta=0).
        coord : float or None
            Surface label (ie, rho, zeta etc) for supplied surface.

        Returns
        -------
        x_lmn : ndarray
            vector of flux surface coefficients associated with x_basis.

        """
        x_lmn = np.zeros((x_basis.num_modes,))
        if mode is None:
            # auto detect based on mode numbers
            if np.all(b_basis.modes[:, 0] == 0):
                mode = "lcfs"
            elif np.all(b_basis.modes[:, 2] == 0):
                mode = "poincare"
            else:
                raise ValueError("Surface should have either l=0 or n=0")
        if mode == "lcfs":
            if coord is None:
                coord = 1.0
            if axis is None:
                axidx = np.where(b_basis.modes[:, 1] == 0)[0]
                axis = np.array([b_basis.modes[axidx, 2], b_lmn[axidx]]).T
            for k, (l, m, n) in enumerate(b_basis.modes):
                scale = zernike_radial(coord, abs(m), m)
                # index of basis mode with lowest radial power (l = |m|)
                idx0 = np.where((x_basis.modes == [np.abs(m), m, n]).all(axis=1))[0]
                if m == 0:  # magnetic axis only affects m=0 modes
                    # index of basis mode with second lowest radial power (l = |m| + 2)
                    idx2 = np.where(
                        (x_basis.modes == [np.abs(m) + 2, m, n]).all(axis=1)
                    )[0]
                    ax = np.where(axis[:, 0] == n)[0]
                    if ax.size:
                        a_n = axis[ax[0], 1]  # use provided axis guess
                    else:
                        a_n = b_lmn[k]  # use boundary centroid as axis
                    x_lmn[idx0] = (b_lmn[k] + a_n) / 2 / scale
                    x_lmn[idx2] = (b_lmn[k] - a_n) / 2 / scale
                else:
                    x_lmn[idx0] = b_lmn[k] / scale

        elif mode == "poincare":
            for k, (l, m, n) in enumerate(b_basis.modes):
                idx = np.where((x_basis.modes == [l, m, n]).all(axis=1))[0]
                x_lmn[idx] = b_lmn[k]

        else:
            raise ValueError("Boundary mode should be either 'lcfs' or 'poincare'.")

        return x_lmn

    def _initial_guess_points(self, nodes, x, x_basis):
        """Create an initial guess based on locations of flux surfaces in real space

        Parameters
        ----------
        nodes : Grid or ndarray, shape(k,3)
            Locations in flux coordinates where real space coordinates are given.
        x : ndarray, shape(k,)
            R, Z or lambda values at specified nodes.
        x_basis : Basis
            Spectral basis for x (R, Z or lambda)

        Returns
        -------
        x_lmn : ndarray
            Vector of flux surface coefficients associated with x_basis.

        """
        if not isinstance(nodes, Grid):
            nodes = Grid(nodes, sort=False)
        transform = Transform(nodes, x_basis, build=False, build_pinv=True)
        x_lmn = transform.fit(x)
        return x_lmn

    @property
    def parent(self):
        """Pointer to the equilibrium this was derived from."""
        return self.__dict__.setdefault("_parent", None)

    @property
    def children(self):
        """List of configurations that were derived from this one."""
        return self.__dict__.setdefault("_children", [])

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this equilibrium."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        new._parent = self
        self.children.append(new)
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
            self.pressure.change_resolution(L=L)
        if L_change and hasattr(self.iota, "change_resolution"):
            self.iota.change_resolution(L=L)

        self.axis.change_resolution(self.N, NFP=self.NFP)
        self.surface.change_resolution(self.L, self.M, self.N, NFP=self.NFP)

        self._R_lmn = copy_coeffs(self.R_lmn, old_modes_R, self.R_basis.modes)
        self._Z_lmn = copy_coeffs(self.Z_lmn, old_modes_Z, self.Z_basis.modes)
        self._L_lmn = copy_coeffs(self.L_lmn, old_modes_L, self.L_basis.modes)

        self._make_labels()

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
                rho=rho,
                M=4 * surface.M + 1,
                N=4 * surface.N + 1,
                endpoint=True,
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
                L=2 * surface.L + 1,
                M=4 * surface.M + 1,
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

    @property
    def surface(self):
        """Geometric surface defining boundary conditions."""
        return self._surface

    @surface.setter
    def surface(self, new):
        if isinstance(new, Surface):
            new.change_resolution(self.L, self.M, self.N)
            self._surface = new
        else:
            raise TypeError(
                f"surfaces should be of type Surface or a subclass, got {new}"
            )

    @property
    def spectral_indexing(self):
        """Type of indexing used for the spectral basis (str)."""
        return self._spectral_indexing

    @property
    def sym(self):
        """Whether this equilibrium is stellarator symmetric (bool)."""
        return self._sym

    @property
    def bdry_mode(self):
        """Mode for specifying plasma boundary (str)."""
        return self._bdry_mode

    @property
    def Psi(self):
        """Total toroidal flux within the last closed flux surface in Webers (float)."""
        return self._Psi

    @Psi.setter
    def Psi(self, Psi):
        self._Psi = float(Psi)

    @property
    def NFP(self):
        """Number of (toroidal) field periods (int)."""
        return self._NFP

    @NFP.setter
    def NFP(self, NFP):
        assert (
            isinstance(NFP, numbers.Real) and (NFP == int(NFP)) and (NFP > 0)
        ), f"NFP should be a positive integer, got {type(NFP)}"
        self.change_resolution(NFP=NFP)

    @property
    def L(self):
        """Maximum radial mode number (int)."""
        return self._L

    @L.setter
    def L(self, L):
        assert (
            isinstance(L, numbers.Real) and (L == int(L)) and (L >= 0)
        ), f"L should be a non-negative integer got {L}"
        self.change_resolution(L=L)

    @property
    def M(self):
        """Maximum poloidal fourier mode number (int)."""
        return self._M

    @M.setter
    def M(self, M):
        assert (
            isinstance(M, numbers.Real) and (M == int(M)) and (M >= 0)
        ), f"M should be a non-negative integer got {M}"
        self.change_resolution(M=M)

    @property
    def N(self):
        """Maximum toroidal fourier mode number (int)."""
        return self._N

    @N.setter
    def N(self, N):
        assert (
            isinstance(N, numbers.Real) and (N == int(N)) and (N >= 0)
        ), f"N should be a non-negative integer got {N}"
        self.change_resolution(N=N)

    @property
    def R_lmn(self):
        """Spectral coefficients of R (ndarray)."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, R_lmn):
        self._R_lmn[:] = R_lmn

    @property
    def Z_lmn(self):
        """Spectral coefficients of Z (ndarray)."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, Z_lmn):
        self._Z_lmn[:] = Z_lmn

    @property
    def L_lmn(self):
        """Spectral coefficients of lambda (ndarray)."""
        return self._L_lmn

    @L_lmn.setter
    def L_lmn(self, L_lmn):
        self._L_lmn[:] = L_lmn

    @property
    def Rb_lmn(self):
        """Spectral coefficients of R at the boundary (ndarray)."""
        return self.surface.R_lmn

    @Rb_lmn.setter
    def Rb_lmn(self, Rb_lmn):
        self.surface.R_lmn = Rb_lmn

    @property
    def Zb_lmn(self):
        """Spectral coefficients of Z at the boundary (ndarray)."""
        return self.surface.Z_lmn

    @Zb_lmn.setter
    def Zb_lmn(self, Zb_lmn):
        self.surface.Z_lmn = Zb_lmn

    @property
    def Ra_n(self):
        """R coefficients for axis Fourier series."""
        return self.axis.R_n

    @property
    def Za_n(self):
        """Z coefficients for axis Fourier series."""
        return self.axis.Z_n

    @property
    def axis(self):
        """Curve object representing the magnetic axis."""
        # TODO: return the current axis by evaluating at rho=0
        return self._axis

    @axis.setter
    def axis(self, new):
        if isinstance(new, FourierRZCurve):
            new.change_resolution(self.N)
            self._axis = new
        else:
            raise TypeError(
                f"axis should be of type FourierRZCurve or a subclass, got {new}"
            )

    @property
    def pressure(self):
        """Pressure profile."""
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
        """Coefficients of pressure profile (ndarray)."""
        return self.pressure.params

    @p_l.setter
    def p_l(self, p_l):
        self.pressure.params = p_l

    @property
    def iota(self):
        """Rotational transform (iota) profile."""
        return self._iota

    @iota.setter
    def iota(self, new):
        if isinstance(new, Profile):
            self._iota = new
        else:
            raise TypeError(
                f"iota profile should be of type Profile or a subclass, got {new} "
            )

    @property
    def i_l(self):
        """Coefficients of iota profile (ndarray)."""
        return self.iota.params

    @i_l.setter
    def i_l(self, i_l):
        self.iota.params = i_l

    @property
    def R_basis(self):
        """Spectral basis for R (FourierZernikeBasis)."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z (FourierZernikeBasis)."""
        return self._Z_basis

    @property
    def L_basis(self):
        """Spectral basis for lambda (FourierZernikeBasis)."""
        return self._L_basis

    # FIXME: update this now that x is no longer a property of Configuration
    def _make_labels(self):
        R_label = ["R_{},{},{}".format(l, m, n) for l, m, n in self.R_basis.modes]
        Z_label = ["Z_{},{},{}".format(l, m, n) for l, m, n in self.Z_basis.modes]
        L_label = ["L_{},{},{}".format(l, m, n) for l, m, n in self.L_basis.modes]
        return None

    def get_xlabel_by_idx(self, idx):
        """Find which mode corresponds to a given entry in x.

        Parameters
        ----------
        idx : int or array-like of int
            index into optimization vector x

        Returns
        -------
        label : str or list of str
            label for the coefficient at index idx, eg R_0,1,3 or L_4,3,0

        """
        self._make_labels()
        idx = np.atleast_1d(idx)
        labels = [self.xlabel.get(i, None) for i in idx]
        return labels

    def get_idx_by_xlabel(self, labels):
        """Find which index of x corresponds to a given mode.

        Parameters
        ----------
        label : str or list of str
            label for the coefficient at index idx, eg R_0,1,3 or L_4,3,0

        Returns
        -------
        idx : int or array-like of int
            index into optimization vector x

        """
        self._make_labels()
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        idx = [self.rev_xlabel.get(label, None) for label in labels]
        return np.array(idx)

    def compute(self, name, grid=None, data=None, **kwargs):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        name : str
            Name of the quantity to compute.
        grid : Grid, optional
            Grid of coordinates to evaluate at. Defaults to the quadrature grid.

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        if name not in data_index:
            raise ValueError("Unrecognized value '{}'.".format(name))
        if grid is None:
            grid = QuadratureGrid(self.L_grid, self.M_grid, self.N_grid, self.NFP)
        M_booz = kwargs.pop("M_booz", 2 * self.M)
        N_booz = kwargs.pop("N_booz", 2 * self.N)
        if len(kwargs) > 0 and not set(kwargs.keys()).issubset(["helicity"]):
            raise ValueError("Unrecognized argument(s).")

        fun = getattr(compute_funs, data_index[name]["fun"])
        sig = signature(fun)

        inputs = {"data": data}
        for arg in sig.parameters.keys():
            if arg in arg_order:
                inputs[arg] = getattr(self, arg)
            elif arg == "R_transform":
                inputs[arg] = Transform(
                    grid, self.R_basis, derivs=data_index[name]["R_derivs"]
                )
            elif arg == "Z_transform":
                inputs[arg] = Transform(
                    grid, self.Z_basis, derivs=data_index[name]["R_derivs"]
                )
            elif arg == "L_transform":
                inputs[arg] = Transform(
                    grid, self.L_basis, derivs=data_index[name]["L_derivs"]
                )
            elif arg == "B_transform":
                inputs[arg] = Transform(
                    grid,
                    DoubleFourierSeries(
                        M=M_booz, N=N_booz, sym=self.R_basis.sym, NFP=self.NFP
                    ),
                    derivs=0,
                    build_pinv=True,
                )
            elif arg == "w_transform":
                inputs[arg] = Transform(
                    grid,
                    DoubleFourierSeries(
                        M=M_booz, N=N_booz, sym=self.Z_basis.sym, NFP=self.NFP
                    ),
                    derivs=1,
                )
            elif arg == "pressure":
                inputs[arg] = self.pressure.copy()
                inputs[arg].grid = grid
            elif arg == "iota":
                inputs[arg] = self.iota.copy()
                inputs[arg].grid = grid

        return fun(**inputs, **kwargs)

    def compute_theta_coords(self, flux_coords, L_lmn=None, tol=1e-6, maxiter=20):
        """Find the theta coordinates (rho, theta, phi) that correspond to a set of
        straight field-line coordinates (rho, theta*, zeta).

        Parameters
        ----------
        flux_coords : ndarray, shape(k,3)
            2d array of flux coordinates [rho,theta*,zeta]. Each row is a different
            coordinate.
        L_lmn : ndarray
            spectral coefficients for lambda. Defaults to self.L_lmn
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
        if L_lmn is None:
            L_lmn = self.L_lmn
        rho, theta_star, zeta = flux_coords.T
        if maxiter <= 0:
            raise ValueError(f"maxiter must be a positive integer, got{maxiter}")
        if jnp.any(rho < 0):
            raise ValueError("rho values must be positive")

        # Note: theta* (also known as vartheta) is the poloidal straight field-line
        # angle in PEST-like flux coordinates

        nodes = flux_coords.copy()
        A0 = self.L_basis.evaluate(nodes, (0, 0, 0))

        # theta* = theta + lambda
        lmbda = jnp.dot(A0, L_lmn)
        k = 0

        def cond_fun(nodes_k_lmbda):
            nodes, k, lmbda = nodes_k_lmbda
            theta_star_k = nodes[:, 1] + lmbda
            err = theta_star - theta_star_k
            return jnp.any(jnp.abs(err) > tol) & (k < maxiter)

        # Newton method for root finding
        def body_fun(nodes_k_lmbda):
            nodes, k, lmbda = nodes_k_lmbda
            A1 = self.L_basis.evaluate(nodes, (0, 1, 0))
            lmbda_t = jnp.dot(A1, L_lmn)
            f = theta_star - nodes[:, 1] - lmbda
            df = -1 - lmbda_t
            nodes = put(nodes, Index[:, 1], nodes[:, 1] - f / df)
            A0 = self.L_basis.evaluate(nodes, (0, 0, 0))
            lmbda = jnp.dot(A0, L_lmn)
            k += 1
            return (nodes, k, lmbda)

        nodes, k, lmbda = jit(while_loop, static_argnums=(0, 1))(
            cond_fun, body_fun, (nodes, k, lmbda)
        )
        theta_star_k = nodes[:, 1] + lmbda
        err = theta_star - theta_star_k
        noconverge = jnp.abs(err) > tol
        nodes = jnp.where(noconverge[:, np.newaxis], jnp.nan, nodes)

        return nodes

    def compute_flux_coords(
        self, real_coords, R_lmn=None, Z_lmn=None, tol=1e-6, maxiter=20, rhomin=1e-6
    ):
        """Find the flux coordinates (rho, theta, zeta) that correspond to a set of
        real space coordinates (R, phi, Z).

        Parameters
        ----------
        real_coords : ndarray, shape(k,3)
            2D array of real space coordinates [R,phi,Z]. Each row is a different coordinate.
        R_lmn, Z_lmn : ndarray
            spectral coefficients for R and Z. Defaults to self.R_lmn, self.Z_lmn
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
        if maxiter <= 0:
            raise ValueError(f"maxiter must be a positive integer, got{maxiter}")
        if R_lmn is None:
            R_lmn = self.R_lmn
        if Z_lmn is None:
            Z_lmn = self.Z_lmn

        R, phi, Z = real_coords.T
        R = jnp.abs(R)

        # nearest neighbor search on coarse grid for initial guess
        nodes = ConcentricGrid(L=20, M=10, N=0).nodes
        AR = self.R_basis.evaluate(nodes)
        AZ = self.Z_basis.evaluate(nodes)
        Rg = jnp.dot(AR, R_lmn)
        Zg = jnp.dot(AZ, Z_lmn)
        distance = (R[:, np.newaxis] - Rg) ** 2 + (Z[:, np.newaxis] - Zg) ** 2
        idx = jnp.argmin(distance, axis=1)

        rhok = nodes[idx, 0]
        thetak = nodes[idx, 1]
        Rk = Rg[idx]
        Zk = Zg[idx]
        k = 0

        def cond_fun(k_rhok_thetak_Rk_Zk):
            k, rhok, thetak, Rk, Zk = k_rhok_thetak_Rk_Zk
            return jnp.any(((R - Rk) ** 2 + (Z - Zk) ** 2) > tol ** 2) & (k < maxiter)

        def body_fun(k_rhok_thetak_Rk_Zk):
            k, rhok, thetak, Rk, Zk = k_rhok_thetak_Rk_Zk
            nodes = jnp.array([rhok, thetak, phi]).T
            ARr = self.R_basis.evaluate(nodes, (1, 0, 0))
            Rr = jnp.dot(ARr, R_lmn)
            AZr = self.Z_basis.evaluate(nodes, (1, 0, 0))
            Zr = jnp.dot(AZr, Z_lmn)
            ARt = self.R_basis.evaluate(nodes, (0, 1, 0))
            Rt = jnp.dot(ARt, R_lmn)
            AZt = self.Z_basis.evaluate(nodes, (0, 1, 0))
            Zt = jnp.dot(AZt, Z_lmn)

            tau = Rt * Zr - Rr * Zt
            eR = R - Rk
            eZ = Z - Zk
            thetak += (Zr * eR - Rr * eZ) / tau
            rhok += (Rt * eZ - Zt * eR) / tau
            # negative rho -> rotate theta instead
            thetak = jnp.where(
                rhok < 0, (thetak + np.pi) % (2 * np.pi), thetak % (2 * np.pi)
            )
            rhok = jnp.abs(rhok)
            rhok = jnp.clip(rhok, rhomin, 1)
            nodes = jnp.array([rhok, thetak, phi]).T

            AR = self.R_basis.evaluate(nodes, (0, 0, 0))
            Rk = jnp.dot(AR, R_lmn)
            AZ = self.Z_basis.evaluate(nodes, (0, 0, 0))
            Zk = jnp.dot(AZ, Z_lmn)
            k += 1
            return (k, rhok, thetak, Rk, Zk)

        k, rhok, thetak, Rk, Zk = while_loop(
            cond_fun, body_fun, (k, rhok, thetak, Rk, Zk)
        )

        noconverge = (R - Rk) ** 2 + (Z - Zk) ** 2 > tol ** 2
        rho = jnp.where(noconverge, jnp.nan, rhok)
        theta = jnp.where(noconverge, jnp.nan, thetak)
        phi = jnp.where(noconverge, jnp.nan, phi)

        return jnp.vstack([rho, theta, phi]).T

    def is_nested(self, nsurfs=10, ntheta=20, nzeta=None, Nt=45, Nr=20):
        """Check that an equilibrium has properly nested flux surfaces in a plane.

        Parameters
        ----------
        nsurfs : int, optional
            number of radial surfaces to check (Default value = 10)
        ntheta : int, optional
            number of sfl poloidal contours to check (Default value = 20)
        nzeta : int, optional
            Number of toroidal planes to check, by default checks the zeta=0
            plane for axisymmetric equilibria and 5 planes evenly spaced in
            zeta between 0 and 2pi/NFP for non-axisymmetric, otherwise uses
            nzeta planes linearly spaced  in zeta between 0 and 2pi/NFP
        Nt : int, optional
            number of theta points to use for the r contours (Default value = 45)
        Nr : int, optional
            number of r points to use for the theta contours (Default value = 20)

        Returns
        -------
        is_nested : bool
            whether or not the surfaces are nested

        """
        planes_nested_bools = []
        if nzeta is None:
            zetas = (
                [0]
                if self.N == 0
                else np.linspace(0, 2 * np.pi / self.NFP, 5, endpoint=False)
            )
        else:
            zetas = np.linspace(0, 2 * np.pi / self.NFP, nzeta, endpoint=False)

        for zeta in zetas:
            r_grid = LinearGrid(L=nsurfs, M=Nt, zeta=zeta, endpoint=True)
            t_grid = LinearGrid(L=Nr, M=ntheta, zeta=zeta, endpoint=False)

            r_coords = self.compute("R", r_grid)
            t_coords = self.compute("lambda", t_grid)

            v_nodes = t_grid.nodes
            v_nodes[:, 1] = t_grid.nodes[:, 1] - t_coords["lambda"]
            v_grid = Grid(v_nodes)
            v_coords = self.compute("R", v_grid)

            # rho contours
            Rr = r_coords["R"].reshape((r_grid.L, r_grid.M, r_grid.N))[:, :, 0]
            Zr = r_coords["Z"].reshape((r_grid.L, r_grid.M, r_grid.N))[:, :, 0]

            # theta contours
            Rv = v_coords["R"].reshape((t_grid.L, t_grid.M, t_grid.N))[:, :, 0]
            Zv = v_coords["Z"].reshape((t_grid.L, t_grid.M, t_grid.N))[:, :, 0]

            rline = MultiLineString(
                [LineString(np.array([R, Z]).T) for R, Z in zip(Rr, Zr)]
            )
            vline = MultiLineString(
                [LineString(np.array([R, Z]).T) for R, Z in zip(Rv.T, Zv.T)]
            )

            planes_nested_bools.append(rline.is_simple and vline.is_simple)
        return np.all(planes_nested_bools)

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
            radial resolution to use for SFL equilibrium. Default = self.L
        M : int, optional
            poloidal resolution to use for SFL equilibrium. Default = self.M
        N : int, optional
            toroidal resolution to use for SFL equilibrium. Default = self.N
        L_grid : int, optional
            radial spatial resolution to use for fit to new basis. Default = 4*self.L+1
        M_grid : int, optional
            poloidal spatial resolution to use for fit to new basis. Default = 4*self.M+1
        N_grid : int, optional
            toroidal spatial resolution to use for fit to new basis. Default = 4*self.N+1
        rcond : float, optional
            cutoff for small singular values in least squares fit.
        copy : bool, optional
            Whether to update the existing equilibrium or make a copy (Default).

        Returns
        -------
        eq_sfl : Equilibrium
            Equilibrium transformed to a straight field line coordinate representation.

        """
        L = L or int(1.5 * self.L)
        M = M or int(1.5 * self.M)
        N = N or int(1.5 * self.N)
        L_grid = L_grid or L
        M_grid = M_grid or M
        N_grid = N_grid or N

        grid = ConcentricGrid(L_grid, M_grid, N_grid, node_pattern="ocs")
        bdry_grid = LinearGrid(rho=1, M=2 * M + 1, N=2 * N + 1)

        toroidal_coords = self.compute("R", grid)
        theta = grid.nodes[:, 1]
        vartheta = theta + self.compute("lambda", grid)["lambda"]
        sfl_grid = grid
        sfl_grid.nodes[:, 1] = vartheta

        bdry_coords = self.compute("R", bdry_grid)
        bdry_theta = bdry_grid.nodes[:, 1]
        bdry_vartheta = bdry_theta + self.compute("lambda", bdry_grid)["lambda"]
        bdry_sfl_grid = bdry_grid
        bdry_sfl_grid.nodes[:, 1] = bdry_vartheta

        if copy:
            eq_sfl = self.copy()
        else:
            eq_sfl = self
        eq_sfl.change_resolution(L, M, N)

        R_sfl_transform = Transform(
            sfl_grid, eq_sfl.R_basis, build=False, build_pinv=True, rcond=rcond
        )
        R_lmn_sfl = R_sfl_transform.fit(toroidal_coords["R"])
        del R_sfl_transform  # these can take up a lot of memory so delete when done.

        Z_sfl_transform = Transform(
            sfl_grid, eq_sfl.Z_basis, build=False, build_pinv=True, rcond=rcond
        )
        Z_lmn_sfl = Z_sfl_transform.fit(toroidal_coords["Z"])
        del Z_sfl_transform
        L_lmn_sfl = np.zeros_like(eq_sfl.L_lmn)

        R_sfl_bdry_transform = Transform(
            bdry_sfl_grid,
            eq_sfl.surface.R_basis,
            build=False,
            build_pinv=True,
            rcond=rcond,
        )
        Rb_lmn_sfl = R_sfl_bdry_transform.fit(bdry_coords["R"])
        del R_sfl_bdry_transform

        Z_sfl_bdry_transform = Transform(
            bdry_sfl_grid,
            eq_sfl.surface.Z_basis,
            build=False,
            build_pinv=True,
            rcond=rcond,
        )
        Zb_lmn_sfl = Z_sfl_bdry_transform.fit(bdry_coords["Z"])
        del Z_sfl_bdry_transform

        eq_sfl.Rb_lmn = Rb_lmn_sfl
        eq_sfl.Zb_lmn = Zb_lmn_sfl
        eq_sfl.R_lmn = R_lmn_sfl
        eq_sfl.Z_lmn = Z_lmn_sfl
        eq_sfl.L_lmn = L_lmn_sfl

        return eq_sfl
