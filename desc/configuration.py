import numpy as np
import copy
from termcolor import colored
from abc import ABC
from shapely.geometry import LineString, MultiLineString

from desc.backend import jnp, put
from desc.io import IOAble
from desc.utils import copy_coeffs
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.transform import Transform
from desc.profiles import Profile, PowerSeriesProfile
from desc.geometry import (
    FourierRZToroidalSurface,
    ZernikeRZToroidalSection,
    FourierRZCurve,
)
from desc.basis import (
    DoubleFourierSeries,
    FourierZernikeBasis,
)
from desc.compute_funs import (
    compute_toroidal_flux,
    compute_rotational_transform,
    compute_pressure,
    compute_lambda,
    compute_toroidal_coords,
    compute_cartesian_coords,
    compute_jacobian,
    compute_magnetic_field_magnitude,
    compute_magnetic_pressure_gradient,
    compute_magnetic_tension,
    compute_contravariant_current_density,
    compute_force_error,
    compute_volume,
    compute_energy,
)
from desc.vmec_utils import (
    ptolemy_identity_rev,
    zernike_to_fourier,
)


class _Configuration(IOAble, ABC):
    """Configuration is an abstract base class for equilibrium information.

    It contains information about a plasma state, including the
    shapes of flux surfaces and profile inputs. It can compute additional
    information, such as the magnetic field and plasma currents.
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

    def __init__(self, inputs):
        """Initialize a Configuration.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs with the following required keys:
                Psi : float, total toroidal flux (in Webers) within LCFS
                NFP : int, number of field periods
                L : int, radial resolution
                M : int, poloidal resolution
                N : int, toroidal resolution
                profiles : ndarray, array of profile coeffs [l, p_l, i_l]
                boundary : ndarray, array of boundary coeffs [m, n, Rb_lmn, Zb_lmn]
            And the following optional keys:
                sym : bool, is the problem stellarator symmetric or not, default is False
                spectral_indexing : str, type of Zernike indexing scheme to use, default is ``'ansi'``
                bdry_mode : {``'lcfs'``, ``'poincare'``}, where the BC are enforced
                axis : ndarray, array of magnetic axis coeffs [n, R0_n, Z0_n]
                R_lmn : ndarray, spectral coefficients of R
                Z_lmn : ndarray, spectral coefficients of Z
                L_lmn : ndarray, spectral coefficients of lambda

        """
        self.inputs = inputs
        try:
            self._Psi = float(inputs["Psi"])
            self._NFP = inputs["NFP"]
            self._L = inputs["L"]
            self._M = inputs["M"]
            self._N = inputs["N"]
            profiles = inputs["profiles"]
            boundary = inputs["boundary"]
        except:
            raise ValueError(colored("input dict does not contain proper keys", "red"))

        # optional inputs
        self._sym = inputs.get("sym", False)
        self._spectral_indexing = inputs.get("spectral_indexing", "fringe")
        self._bdry_mode = inputs.get("bdry_mode", "lcfs")

        # keep track of where it came from
        self._parent = None
        self._children = []

        # stellarator symmetry for bases
        if self.sym:
            self._R_sym = "cos"
            self._Z_sym = "sin"
        else:
            self._R_sym = False
            self._Z_sym = False

        # create bases
        self._set_basis()

        # format profiles
        self._pressure = PowerSeriesProfile(
            modes=profiles[:, 0], params=profiles[:, 1], name="pressure"
        )
        self._iota = PowerSeriesProfile(
            modes=profiles[:, 0], params=profiles[:, 2], name="iota"
        )
        self._pressure.change_resolution(self._L)
        self._iota.change_resolution(self._L)

        # format boundary
        if self.bdry_mode == "lcfs":
            self._surface = FourierRZToroidalSurface(
                boundary[:, 3],
                boundary[:, 4],
                boundary[:, 1:3].astype(int),
                boundary[:, 1:3].astype(int),
                self.NFP,
                self.sym,
            )
        elif self.bdry_mode == "poincare":
            self._surface = ZernikeRZToroidalSection(
                boundary[:, 3],
                boundary[:, 4],
                boundary[:, :2].astype(int),
                boundary[:, :2].astype(int),
                self.spectral_indexing,
                self.sym,
            )
        else:
            raise ValueError("boundary should either have l=0 or n=0")
        self._surface.change_resolution(self._L, self._M, self._N)

        axis = inputs.get("axis", boundary[np.where(boundary[:, 1] == 0)[0], 2:])
        self._axis = FourierRZCurve(
            axis[:, 1],
            axis[:, 2],
            axis[:, 0].astype(int),
            NFP=self.NFP,
            sym=self.sym,
            name="axis",
        )

        # default initial guess
        self._R_lmn = initial_guess(
            self.R_basis, self.Rb_lmn, self.surface.R_basis, axis[:, 0:-1]
        )
        self._Z_lmn = initial_guess(
            self.Z_basis, self.Zb_lmn, self.surface.Z_basis, axis[:, (0, -1)]
        )
        self._L_lmn = np.zeros((self.L_basis.num_modes,))

        # this makes sure the axis has the correct coefficients
        self.axis.change_resolution(self.N)
        self._axis = self.axis

    def _set_basis(self):

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

    @property
    def parent(self):
        """Pointer to the equilibrium this was derived from."""
        if not hasattr(self, "_parent"):
            self._parent = None
        return self._parent

    @property
    def children(self):
        """List of configurations that were derived from this one."""
        if not hasattr(self, "_children"):
            self._children = []
        return self._children

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this equilibrium."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        new._parent = self
        self.children.append(new)
        return new

    def change_resolution(self, L=None, M=None, N=None, *args, **kwargs):
        """Set the spectral resolution.

        Parameters
        ----------
        L : int
            maximum radial zernike mode number
        M : int
            maximum poloidal fourier mode number
        N : int
            maximum toroidal fourier mode number

        """
        L_change = M_change = N_change = False
        if L is not None and L != self.L:
            L_change = True
            self._L = L
        if M is not None and M != self.M:
            M_change = True
            self._M = M
        if N is not None and N != self.N:
            N_change = True
            self._N = N

        if not np.any([L_change, M_change, N_change]):
            return

        old_modes_R = self.R_basis.modes
        old_modes_Z = self.Z_basis.modes
        old_modes_L = self.L_basis.modes

        self._set_basis()

        if L_change:
            self._pressure.change_resolution(L=L)
            self._iota.change_resolution(L=L)

        if N_change:
            self.axis.change_resolution(self.N)

        self.surface.change_resolution(self.L, self.M, self.N)

        self._R_lmn = copy_coeffs(self.R_lmn, old_modes_R, self.R_basis.modes)
        self._Z_lmn = copy_coeffs(self.Z_lmn, old_modes_Z, self.Z_basis.modes)
        self._L_lmn = copy_coeffs(self.L_lmn, old_modes_L, self.L_basis.modes)

        self._make_labels()

    @property
    def surface(self):
        """Geometric surface defining boundary conditions."""
        return self._surface

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
        if self.NFP != NFP:
            self._NFP = NFP

    @property
    def L(self):
        """Maximum radial mode number (int)."""
        return self._L

    @L.setter
    def L(self, L):
        if self.L != L:
            self._L = L
            self.change_resolution(L=L)

    @property
    def M(self):
        """Maximum poloidal fourier mode number (int)."""
        return self._M

    @M.setter
    def M(self, M):
        if self.M != M:
            self._M = M
            self.change_resolution(M=M)

    @property
    def N(self):
        """Maximum toroidal fourier mode number (int)."""
        return self._N

    @N.setter
    def N(self, N):
        if self.N != N:
            self._N = N
            self.change_resolution(N=N)

    @property
    def R_lmn(self):
        """Spectral coefficients of R (ndarray)."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, R_lmn):
        self._R_lmn = R_lmn

    @property
    def Z_lmn(self):
        """Spectral coefficients of Z (ndarray)."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, Z_lmn):
        self._Z_lmn = Z_lmn

    @property
    def L_lmn(self):
        """Spectral coefficients of lambda (ndarray)."""
        return self._L_lmn

    @L_lmn.setter
    def L_lmn(self, L_lmn):
        self._L_lmn = L_lmn

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

    @property
    def major_radius(self):
        """Major radius (m)."""
        V = self.compute_volume()
        A = self.compute_cross_section_area()
        return V / (2 * np.pi * A)

    @property
    def minor_radius(self):
        """Minor radius (m)."""
        A = self.compute_cross_section_area()
        return np.sqrt(A / np.pi)

    @property
    def aspect_ratio(self):
        """Aspect ratio = major radius / minor radius."""
        V = self.compute_volume()
        A = self.compute_cross_section_area()
        return V / (2 * np.sqrt(np.pi * A ** 3))

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

    def compute_toroidal_flux(self, grid=None, deriv=1, data=None):
        """Compute Compute toroidal magnetic flux profile.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of toroidal magnetic flux profile.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        iota = self.iota.copy()
        iota.grid = grid

        return compute_toroidal_flux(self.Psi, iota, dr=deriv, data=data)

    def compute_iota(self, grid=None, deriv=1, data=None):
        """Compute rotational transform profile.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of rotational transform profile.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        iota = self.iota.copy()
        iota.grid = grid

        return compute_rotational_transform(self.iota.params, iota, dr=deriv, data=data)

    def compute_pressure(self, grid=None, deriv=1, data=None):
        """Compute pressure profile.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of pressure profile.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        pressure = self.pressure.copy()
        pressure.grid = grid

        return compute_pressure(pressure.params, pressure, dr=deriv, data=data)

    def compute_lambda(self, grid=None, deriv=0, data=None):
        """Compute lambda such that theta* = theta + lambda is a sfl coordinate.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of lambda values.
            Keys are of the form 'lambda_x' meaning the derivative of lambda wrt to x.

        """
        L_transform = Transform(grid, self.L_basis, derivs=deriv, build=True)

        return compute_lambda(
            self.L_lmn, L_transform, dr=deriv, dt=deriv, dz=deriv, data=data
        )

    def compute_toroidal_coords(self, grid=None, deriv=0, data=None):
        """Compute toroidal coordinates (R, phi, Z).

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
            Keys are of the form 'X_y' meaning the derivative of X wrt y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=deriv, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=deriv, build=True)

        return compute_toroidal_coords(
            self.R_lmn,
            self.Z_lmn,
            R_transform,
            Z_transform,
            dr=deriv,
            dt=deriv,
            dz=deriv,
            data=data,
        )

    def compute_cartesian_coords(self, grid=None, data=None):
        """Compute Cartesian coordinates (X, Y, Z).

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of Cartesian coordinates.

        """
        R_transform = Transform(grid, self.R_basis, derivs=0, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=0, build=True)

        return compute_cartesian_coords(
            self.R_lmn, self.Z_lmn, R_transform, Z_transform, data=data
        )

    def compute_jacobian(self, grid=None, deriv=0, data=None):
        """Compute coordinate system Jacobian.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of coordinate system Jacobian.
            Keys are of the form 'sqrt(g)_x', meaning the x derivative of the coordinate
            system Jacobian sqrt(g).

        """
        R_transform = Transform(grid, self.R_basis, derivs=deriv + 1, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=deriv + 1, build=True)

        return compute_jacobian(
            self.R_lmn,
            self.Z_lmn,
            R_transform,
            Z_transform,
            dr=deriv,
            dt=deriv,
            dz=deriv,
            data=data,
        )

    def compute_magnetic_field(self, grid=None, deriv=0, data=None):
        """Compute magnetic field components.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of magnetic field magnitude.
            Keys are of the form '|B|_x', meaning the x derivative of the
            magnetic field magnitude |B|.

        """
        R_transform = Transform(grid, self.R_basis, derivs=deriv + 1, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=deriv + 1, build=True)
        L_transform = Transform(grid, self.L_basis, derivs=deriv + 1, build=True)
        iota = self.iota.copy()
        iota.grid = grid

        return compute_magnetic_field_magnitude(
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.iota.params,
            self.Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            dr=deriv,
            dt=deriv,
            dz=deriv,
            data=data,
        )

    def compute_current_density(self, grid=None, deriv=0, data=None):
        """Compute contravariant current density components.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of contravariant current density
            components. Keys are of the form 'J^x_y', meaning the x contravariant (J^x)
            component of the current density J, differentiated wrt y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=deriv + 2, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=deriv + 2, build=True)
        L_transform = Transform(grid, self.L_basis, derivs=deriv + 2, build=True)
        iota = self.iota.copy()
        iota.grid = grid

        return compute_contravariant_current_density(
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.iota.params,
            self.Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            dr=deriv,
            dt=deriv,
            dz=deriv,
            data=data,
        )

    def compute_magnetic_pressure_gradient(self, grid=None, data=None):
        """Compute magnetic pressure gradient.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of magnetic pressure gradient
            components and magnitude. Keys are of the form 'grad(|B|^2)_x', meaning the x
            covariant component of the magnetic pressure gradient grad(|B|^2).

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=2, build=True)
        L_transform = Transform(grid, self.L_basis, derivs=2, build=True)
        iota = self.iota.copy()
        iota.grid = grid

        return compute_magnetic_pressure_gradient(
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.iota.params,
            self.Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            data=data,
        )

    def compute_magnetic_tension(self, grid=None, data=None):
        """Compute magnetic tension.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of magnetic tension vector components
            and magnitude. Keys are of the form '((B*grad(|B|))B)^x', meaning the x
            contravariant component of the magnetic tension vector (B*grad(|B|))B.

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=2, build=True)
        L_transform = Transform(grid, self.L_basis, derivs=2, build=True)
        iota = self.iota.copy()
        iota.grid = grid

        return compute_magnetic_tension(
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.iota.params,
            self.Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            data=data,
        )

    def compute_force_error(self, grid=None, data=None):
        """Compute force error magnitude.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        data : dict
            Dictionary of ndarray, shape(num_nodes,) of force error magnitudes.

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=2, build=True)
        L_transform = Transform(grid, self.L_basis, derivs=2, build=True)
        iota = self.iota.copy()
        pressure = self.pressure.copy()
        iota.grid = grid
        pressure.grid = grid

        return compute_force_error(
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.iota.params,
            self.pressure.params,
            self.Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            pressure,
            data=data,
        )

    def compute_volume(self):
        """Compute plasma volume.

        Returns
        -------
        volume : float
            Plasma volume (m^3).

        """
        grid = QuadratureGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP)

        R_transform = Transform(grid, self.R_basis, derivs=0, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=0, build=True)

        data = compute_volume(self.R_lmn, self.Z_lmn, R_transform, Z_transform)
        return data["V"]

    def compute_cross_section_area(self):
        """Compute toroidally averaged cross-section area.

        Returns
        -------
        area : float
            Cross section area (m^2).

        """
        grid = QuadratureGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP)

        R_transform = Transform(grid, self.R_basis, derivs=0, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=0, build=True)

        data = compute_jacobian(self.R_lmn, self.Z_lmn, R_transform, Z_transform)

        N = np.unique(grid.nodes[:, -1]).size  # number of toroidal angles
        weights = R_transform.grid.weights / (2 * np.pi / N)  # remove toroidal weights
        return np.mean(
            np.sum(
                np.reshape(  # sqrt(g) / R * weight = dArea
                    np.abs(data["g"] / data["R"]) * weights, (N, -1)
                ),
                axis=1,
            )
        )

    def compute_energy(self, gamma=0):
        """Compute MHD energy.

        :math:`W=\int_V dV(\\frac{B^2}{2\mu_0} + \\frac{p}{\gamma - 1})`

        Parameters
        ----------
        gamma : float
            Adiabatic (compressional) index.

        Returns
        -------
        energy : float
            Total MHD energy (J).

        """
        grid = QuadratureGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP)

        R_transform = Transform(grid, self.R_basis, derivs=1, build=True)
        Z_transform = Transform(grid, self.Z_basis, derivs=1, build=True)
        L_transform = Transform(grid, self.L_basis, derivs=1, build=True)
        iota = self.iota.copy()
        pressure = self.pressure.copy()
        iota.grid = grid
        pressure.grid = grid

        data = compute_energy(
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.iota.params,
            self.pressure.params,
            self.Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            pressure,
            gamma,
        )
        return data["W"]

    def compute_theta_coords(self, flux_coords, tol=1e-6, maxiter=20):
        """Find the theta coordinates (rho, theta, phi) that correspond to a set of
        straight field-line coordinates (rho, theta*, zeta).

        Parameters
        ----------
        flux_coords : ndarray, shape(k,3)
            2d array of flux coordinates [rho,theta*,zeta]. Each row is a different
            coordinate.
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
        rho = flux_coords[:, 0]
        theta_star = flux_coords[:, 1]
        zeta = flux_coords[:, 2]
        if maxiter <= 0:
            raise ValueError(f"maxiter must be a positive integer, got{maxiter}")
        if jnp.any(rho) <= 0:
            raise ValueError("rho values must be positive")

        # Note: theta* (also known as vartheta) is the poloidal straight field-line
        # angle in PEST-like flux coordinates

        theta_k = theta_star
        grid = Grid(jnp.vstack([rho, theta_k, zeta]).T, sort=False)

        transform = Transform(
            grid,
            self.L_basis,
            derivs=np.array([[0, 0, 0], [0, 1, 0]]),
            method="direct1",
            build=True,
        )

        # theta* = theta + lambda
        theta_star_k = theta_k + transform.transform(self.L_lmn)
        err = theta_star - theta_star_k

        # Newton method for root finding
        k = 0
        while jnp.any(abs(err) > tol) and k < maxiter:
            lmbda = transform.transform(self.L_lmn, 0, 0, 0)
            lmbda_t = transform.transform(self.L_lmn, 0, 1, 0)
            f = theta_star - theta_k - lmbda
            df = -1 - lmbda_t

            theta_k = theta_k - f / df

            grid = Grid(jnp.vstack([rho, theta_k, zeta]).T, sort=False)
            transform = Transform(
                grid,
                self.L_basis,
                derivs=np.array([[0, 0, 0], [0, 1, 0]]),
                method="direct1",
                build=True,
            )

            theta_star_k = theta_k + transform.transform(self.L_lmn)
            err = theta_star - theta_star_k
            k += 1

        if k >= maxiter:  # didn't converge for all, mark those as nan
            i = np.where(abs(err) > tol)
            rho = put(rho, i, np.nan)
            theta_k = put(theta_k, i, np.nan)
            zeta = put(zeta, i, np.nan)

        return jnp.vstack([rho, theta_k, zeta]).T

    def compute_flux_coords(self, real_coords, tol=1e-6, maxiter=20, rhomin=1e-6):
        """Find the flux coordinates (rho, theta, zeta) that correspond to a set of
        real space coordinates (R, phi, Z).

        Parameters
        ----------
        real_coords : ndarray, shape(k,3)
            2d array of real space coordinates [R,phi,Z]. Each row is a different coordinate.
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
        R = real_coords[:, 0]
        phi = real_coords[:, 1]
        Z = real_coords[:, 2]
        if maxiter <= 0:
            raise ValueError(f"maxiter must be a positive integer, got{maxiter}")
        if jnp.any(R) <= 0:
            raise ValueError("R values must be positive")

        R0, phi0, Z0 = self.axis.compute_coordinates(grid=phi).T
        theta = jnp.arctan2(Z - Z0, R - R0)
        rho = 0.5 * jnp.ones_like(theta)  # TODO: better initial guess
        grid = Grid(jnp.vstack([rho, theta, phi]).T, sort=False)

        R_transform = Transform(grid, self.R_basis, derivs=1, method="direct1")
        Z_transform = Transform(grid, self.Z_basis, derivs=1, method="direct1")

        Rk = R_transform.transform(self.R_lmn)
        Zk = Z_transform.transform(self.Z_lmn)
        eR = R - Rk
        eZ = Z - Zk

        k = 0
        while jnp.any(jnp.sqrt((eR) ** 2 + (eZ) ** 2) > tol) and k < maxiter:
            Rr = R_transform.transform(self.R_lmn, 1, 0, 0)
            Rt = R_transform.transform(self.R_lmn, 0, 1, 0)
            Zr = Z_transform.transform(self.Z_lmn, 1, 0, 0)
            Zt = Z_transform.transform(self.Z_lmn, 0, 1, 0)

            tau = Rt * Zr - Rr * Zt
            theta += (Zr * eR - Rr * eZ) / tau
            rho += (Rt * eZ - Zt * eR) / tau
            # negative rho -> rotate theta instead
            theta = jnp.where(rho < 0, -theta % (2 * np.pi), theta % (2 * np.pi))
            rho = jnp.clip(rho, rhomin, 1)

            grid = Grid(jnp.vstack([rho, theta, phi]).T, sort=False)
            R_transform = Transform(grid, self.R_basis, derivs=1, method="direct1")
            Z_transform = Transform(grid, self.Z_basis, derivs=1, method="direct1")

            Rk = R_transform.transform(self.R_lmn)
            Zk = Z_transform.transform(self.Z_lmn)
            eR = R - Rk
            eZ = Z - Zk
            k += 1

        if k >= maxiter:  # didn't converge for all, mark those as nan
            i = np.where(jnp.sqrt((eR) ** 2 + (eZ) ** 2) > tol)
            rho = put(rho, i, np.nan)
            theta = put(theta, i, np.nan)
            phi = put(phi, i, np.nan)

        return jnp.vstack([rho, theta, phi]).T

    def is_nested(self, nsurfs=10, ntheta=20, zeta=0, Nt=45, Nr=20):
        """Check that an equilibrium has properly nested flux surfaces in a plane.

        Parameters
        ----------
        nsurfs : int, optional
            number of radial surfaces to check (Default value = 10)
        ntheta : int, optional
            number of sfl poloidal contours to check (Default value = 20)
        zeta : float, optional
            toroidal plane to check (Default value = 0)
        Nt : int, optional
            number of theta points to use for the r contours (Default value = 45)
        Nr : int, optional
            number of r points to use for the theta contours (Default value = 20)

        Returns
        -------
        is_nested : bool
            whether or not the surfaces are nested

        """
        r_grid = LinearGrid(L=nsurfs, M=Nt, zeta=zeta, endpoint=True)
        t_grid = LinearGrid(L=Nr, M=ntheta, zeta=zeta, endpoint=False)

        r_coords = self.compute_lambda(r_grid)
        t_coords = self.compute_lambda(t_grid)
        r_coords = self.compute_toroidal_coords(r_grid, data=r_coords)
        t_coords = self.compute_toroidal_coords(t_grid, data=t_coords)

        v_nodes = t_grid.nodes
        v_nodes[:, 1] = t_grid.nodes[:, 1] - t_coords["lambda"]
        v_grid = Grid(v_nodes)
        v_coords = self.compute_toroidal_coords(v_grid)

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

        return rline.is_simple and vline.is_simple

    def to_sfl(
        self,
        L=None,
        M=None,
        N=None,
        L_grid=None,
        M_grid=None,
        N_grid=None,
        rcond=None,
        in_place=False,
    ):
        """Transform this equilibrium to use straight field line coordinates.

        Uses a least squares fit to find FourierZernike coefficients of R,Z,Rb,Zb with
        respect to the straight field line coordinates, rather than the boundary coordinates.
        The new lambda value will be zero.

        NOTE: Though the converted equilibrium will have flux surfaces that look correct, the
        force balance error will likely be significantly higher than the original equilibrium.

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
        in_place : bool, optional
            whether to return a new equilibriumor modify in place

        Returns
        -------
        eq_sfl : Equilibrium
            Equilibrium transformed to a straight field line coordinate representation.
            Only returned if "copy" is True, otherwise modifies the current equilibrium in place.

        """
        L = L or int(1.5 * self.L)
        M = M or int(1.5 * self.M)
        N = N or int(1.5 * self.N)
        L_grid = L_grid or L
        M_grid = M_grid or M
        N_grid = N_grid or N

        grid = ConcentricGrid(L_grid, M_grid, N_grid, node_pattern="ocs")
        bdry_grid = LinearGrid(rho=1, M=2 * M + 1, N=2 * N + 1)

        toroidal_coords = self.compute_toroidal_coords(grid)
        theta = grid.nodes[:, 1]
        vartheta = theta + toroidal_coords["lambda"]
        sfl_grid = grid
        sfl_grid.nodes[:, 1] = vartheta

        bdry_coords = self.compute_toroidal_coords(bdry_grid)
        bdry_theta = bdry_grid.nodes[:, 1]
        bdry_vartheta = bdry_theta + bdry_coords["lambda"]
        bdry_sfl_grid = bdry_grid
        bdry_sfl_grid.nodes[:, 1] = bdry_vartheta

        if in_place:
            eq_sfl = self
        else:
            eq_sfl = self.copy()
        eq_sfl.change_resolution(L, M, N)

        R_sfl_transform = Transform(
            sfl_grid, eq_sfl.R_basis, build_pinv=True, rcond=rcond
        )
        R_lmn_sfl = R_sfl_transform.fit(toroidal_coords["R"])
        del R_sfl_transform  # these can take up a lot of memory so delete when done.

        Z_sfl_transform = Transform(
            sfl_grid, eq_sfl.Z_basis, build_pinv=True, rcond=rcond
        )
        Z_lmn_sfl = Z_sfl_transform.fit(toroidal_coords["Z"])
        del Z_sfl_transform
        L_lmn_sfl = np.zeros_like(eq_sfl.L_lmn)

        R_sfl_bdry_transform = Transform(
            bdry_sfl_grid, eq_sfl.surface.R_basis, build_pinv=True, rcond=rcond
        )
        Rb_lmn_sfl = R_sfl_bdry_transform.fit(bdry_coords["R"])
        del R_sfl_bdry_transform

        Z_sfl_bdry_transform = Transform(
            bdry_sfl_grid, eq_sfl.surface.Z_basis, build_pinv=True, rcond=rcond
        )
        Zb_lmn_sfl = Z_sfl_bdry_transform.fit(bdry_coords["Z"])
        del Z_sfl_bdry_transform

        eq_sfl.Rb_lmn = Rb_lmn_sfl
        eq_sfl.Zb_lmn = Zb_lmn_sfl
        eq_sfl.R_lmn = R_lmn_sfl
        eq_sfl.Z_lmn = Z_lmn_sfl
        eq_sfl.L_lmn = L_lmn_sfl

        if not in_place:
            return eq_sfl

    def run_booz_xform(
        self,
        M_nyq=None,
        N_nyq=None,
        M_boz=None,
        N_boz=None,
        rho=None,
        filename=None,
        verbose=True,
    ):
        """Convert to Boozer coordinates by running booz_xform.

        Parameters
        ----------
        M_nyq : int
            Poloidal resolution for derived quantities. Default is M.
        N_nyq : int
            Toroidal resolution for derived quantities. Default is N.
        M_boz : int
            Poloidal resolution of Boozer spectrum. Default is 2*M.
        N_boz : int
            Toroidal resolution of Boozer spectrum. Default is 2*N.
        rho : ndarray
            Radial coordinates of the flux surfaces to evaluate at.
        filename : str, Optional
            If given, saves the results to a NetCDF file.
        verbose : bool
            Set False to suppress output of Booz_xform calculations.

        Returns
        -------
        b : Booz_xform
            Booz_xform object that contains the transformed quantities.

        """
        try:
            import booz_xform as bx
        except ImportError as exc:
            raise ImportError(
                colored(
                    "booz_xform not installed, installation details can be found at "
                    + "https://hiddensymmetries.github.io/booz_xform/getting_started.html",
                    "red",
                )
            ) from exc

        if M_nyq is None:
            M_nyq = self.M
        if N_nyq is None:
            N_nyq = self.N
        if M_boz is None:
            M_boz = 2 * self.M
        if N_boz is None:
            N_boz = 2 * self.N if self.N > 0 else 0
        if rho is None:
            rho = np.linspace(0.01, 1, num=100)

        # Booz_xform object
        b = bx.Booz_xform()
        b.verbose = verbose
        b.asym = not self.sym
        b.nfp = int(self.NFP)
        b.ns_in = len(rho)
        b.s_in = rho ** 2
        b.compute_surfs = np.arange(0, b.ns_in)

        # equilibrium resolution
        b.mpol = self.M + 1
        b.ntor = self.N
        b.mnmax = (2 * self.N + 1) * self.M + self.N + 1
        b.xm = np.tile(
            np.linspace(0, self.M, self.M + 1), (2 * self.N + 1, 1)
        ).T.flatten()[-b.mnmax :]
        b.xn = np.tile(
            np.linspace(-self.N, self.N, 2 * self.N + 1) * self.NFP, self.M + 1
        )[-b.mnmax :]

        # Nyquist resolution
        b.mpol_nyq = int(M_nyq)
        b.ntor_nyq = int(N_nyq)
        b.mnmax_nyq = (2 * N_nyq + 1) * M_nyq + N_nyq + 1
        b.xm_nyq = np.tile(
            np.linspace(0, M_nyq, M_nyq + 1), (2 * N_nyq + 1, 1)
        ).T.flatten()[-b.mnmax :]
        b.xn_nyq = np.tile(
            np.linspace(-N_nyq, N_nyq, 2 * N_nyq + 1) * self.NFP, M_nyq + 1
        )[-b.mnmax :]

        # Boozer resolution
        b.mboz = int(M_boz)
        b.nboz = int(N_boz)

        # R, Z, lambda
        m, n, R_mn = zernike_to_fourier(self.R_lmn, basis=self.R_basis, rho=rho)
        m, n, Z_mn = zernike_to_fourier(self.Z_lmn, basis=self.Z_basis, rho=rho)
        m, n, L_mn = zernike_to_fourier(self.L_lmn, basis=self.L_basis, rho=rho)
        xm, xn, R_s, R_c = ptolemy_identity_rev(m, n, R_mn)
        xm, xn, Z_s, Z_c = ptolemy_identity_rev(m, n, Z_mn)
        xm, xn, L_s, L_c = ptolemy_identity_rev(m, n, L_mn)
        b.rmnc = R_c.T
        b.zmns = Z_s.T
        b.lmns = L_s.T
        if not self.sym:
            b.rmns = R_s.T
            b.zmnc = Z_c.T
            b.lmnc = L_c.T

        # Nyquist grid for computing derived quantities
        grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=self.NFP)
        if self.sym:
            basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=self.NFP, sym="cos")
        else:
            basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=self.NFP, sym=None)
        transform = Transform(grid, basis, build_pinv=True)
        m = basis.modes[:, 1]
        n = basis.modes[:, 2]

        # |B|, B_theta, B_zeta
        B_mn = np.zeros((b.ns_in, b.mnmax_nyq))
        Bt_mn = np.zeros((b.ns_in, b.mnmax_nyq))
        Bz_mn = np.zeros((b.ns_in, b.mnmax_nyq))
        for k in range(b.ns_in):
            grid = LinearGrid(
                M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=self.NFP, rho=rho[k]
            )
            data = self.compute_magnetic_field(grid)
            B_mn[k, :] = transform.fit(data["|B|"])
            Bt_mn[k, :] = transform.fit(data["B_theta"])
            Bz_mn[k, :] = transform.fit(data["B_zeta"])
        xm, xn, B_s, B_c = ptolemy_identity_rev(m, n, B_mn)
        xm, xn, Bt_s, Bt_c = ptolemy_identity_rev(m, n, Bt_mn)
        xm, xn, Bz_s, Bz_c = ptolemy_identity_rev(m, n, Bz_mn)
        b.bmnc = B_c.T
        b.bsubumnc = Bt_c.T
        b.bsubvmnc = Bz_c.T
        if not self.sym:
            b.bmns = B_s.T
            b.bsubumns = Bt_s.T
            b.bsubvmns = Bz_s.T

        # rotational transform
        b.iota = self.iota(rho)

        # run booz_xform
        b.run()
        if filename is not None:
            b.write_boozmn(filename)
        return b


def initial_guess(x_basis, b_lmn, b_basis, axis, mode="lcfs"):
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
        Only used for 'lcfs' boundary mode.
    mode : str
        One of 'lcfs', 'poincare'.
        Whether the boundary condition is specified by the last closed flux surface
        (rho=1) or the Poincare section (zeta=0).

    Returns
    -------
    x_lmn : ndarray
        vector of flux surface coefficients associated with x_basis.

    """
    x_lmn = np.zeros((x_basis.num_modes,))

    if mode == "lcfs":
        for k, (l, m, n) in enumerate(b_basis.modes):
            # index of basis mode with lowest radial power (l = |m|)
            idx0 = np.where((x_basis.modes == [np.abs(m), m, n]).all(axis=1))[0]
            if m == 0:  # magnetic axis only affects m=0 modes
                # index of basis mode with second lowest radial power (l = |m| + 2)
                idx2 = np.where((x_basis.modes == [np.abs(m) + 2, m, n]).all(axis=1))[0]
                ax = np.where(axis[:, 0] == n)[0]
                if ax.size:
                    a_n = axis[ax[0], 1]  # use provided axis guess
                else:
                    a_n = b_lmn[k]  # use boundary centroid as axis
                x_lmn[idx0] = (b_lmn[k] + a_n) / 2
                x_lmn[idx2] = (b_lmn[k] - a_n) / 2
            else:
                x_lmn[idx0] = b_lmn[k]

    elif mode == "poincare":
        for k, (l, m, n) in enumerate(b_basis.modes):
            idx = np.where((x_basis.modes == [l, m, n]).all(axis=1))[0]
            x_lmn[idx] = b_lmn[k]

    else:
        raise ValueError("Boundary mode should be either 'lcfs' or 'poincare'.")

    return x_lmn
