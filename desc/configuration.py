import numpy as np
from collections.abc import MutableSequence

from desc.backend import TextColors, put
from desc.utils import Tristate, unpack_state
from desc.basis import Basis, PowerSeries, FourierSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.grid import Grid, LinearGrid, ConcentricGrid
from desc.transform import Transform
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry
from desc.objective_funs import ObjectiveFunction
from desc.equilibrium_io import IOAble

"""
from desc.compute_funs import compute_coordinates, compute_coordinate_derivatives
from desc.compute_funs import compute_covariant_basis, compute_contravariant_basis
from desc.compute_funs import compute_jacobian, compute_magnetic_field, compute_plasma_current
from desc.compute_funs import compute_magnetic_field_magnitude, compute_force_magnitude
"""

class Configuration(IOAble):
    """Configuration contains information about a plasma state, including the
       shapes of flux surfaces and profile inputs. It can compute additional
       information, such as the magnetic field and plasma currents.
    """

    _io_attrs_ = ['_R0_n', '_Z0_n', '_r_lmn', '_l_lmn', '_R1_mn', '_Z1_mn',
                  '_p_l', '_i_l', '_Psi', '_NFP',
                  '_R0_basis', '_Z0_basis', '_r_basis', '_l_basis',
                  '_R1_basis', '_Z1_basis', '_p_basis', '_i_basis']

    _object_lib_ = {'PowerSeries'         : PowerSeries,
                    'DoubleFourierSeries' : DoubleFourierSeries,
                    'FourierZernikeBasis' : FourierZernikeBasis,
                    'LinearGrid'          : LinearGrid,
                    'ConcentricGrid'      : ConcentricGrid}

    def __init__(self, inputs:dict=None, load_from=None,
                 file_format:str='hdf5', obj_lib=None) -> None:
        """Initializes a Configuration

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs with the following required keys:
                L : int, radial resolution
                M : int, poloidal resolution
                N : int, toroidal resolution
                p_l : ndarray, spectral coefficients of pressure profile
                i_l : ndarray, spectral coefficients of rotational transform
                Psi : float, total toroidal flux (in Webers) within LCFS
                NFP : int, number of field periods
                bdry : ndarray, array of fourier coeffs [m,n,Rcoeff, Zcoeff]
            And the following optional keys:
                sym : bool, is the problem stellarator symmetric or not, default is False
                index : str, type of Zernike indexing scheme to use, default is 'ansi'
                bdry_mode : str, how to calculate error at bdry, default is 'spectral'
                bdry_ratio : float, Multiplier on the 3D boundary modes. Default = 1.0.
                pres_ratio : float, Multiplier on the pressure profile. Default = 1.0.
                zeta_ratio : float, Multiplier on the toroidal derivatives. Default = 1.0.
                errr_ratio : float, Weight on the force balance equations, relative to the boundary condition equations. Default = 1e-8.
                axis : ndarray, Fourier coefficients for initial guess for the axis
                x : ndarray, state vector of spectral coefficients
                R0_n : ndarray, spectral coefficients of R0
                Z0_n : ndarray, spectral coefficients of Z0
                r_lmn : ndarray, spectral coefficients of r
                l_lmn : ndarray, spectral coefficients of lambda
        load_from : str file path OR file instance
            file to initialize from
        file_format : str 
            file format of file initializing from. Default is 'hdf5'

        Returns
        -------
        None

        """
        self._file_format_ = file_format
        if load_from is None:
            self._init_from_inputs_(inputs=inputs)
        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib)

    def _init_from_inputs_(self, inputs:dict=None) -> None:
        # required inputs
        try:
            self._NFP = inputs['NFP']
            self._Psi = inputs['Psi']
            self._L = inputs['L']
            self._M = inputs['M']
            self._N = inputs['N']
            profiles = inputs['profiles']
            boundary = inputs['boundary']
        except:
            raise ValueError(TextColors.FAIL +
                             "input dict does not contain proper keys"
                           + TextColors.ENDC)

        # optional inputs
        self._sym = inputs.get('sym', False)
        self._index = inputs.get('index', 'ansi')
        self.bdry_mode = inputs.get('bdry_mode', 'spectral')
        self.bdry_ratio = inputs.get('bdry_ratio', 1.0)
        self.pres_ratio = inputs.get('pres_ratio', 1.0)
        self.zeta_ratio = inputs.get('zeta_ratio', 1.0)
        self.errr_ratio = inputs.get('errr_ratio', 1e-8)

        # stellarator symmetry for bases
        if self._sym:
            self._R_sym = Tristate(True)
            self._Z_sym = Tristate(False)
        else:
            self._R_sym = Tristate(None)
            self._Z_sym = Tristate(None)

        # create bases
        self._R0_basis = FourierSeries(
            N=self._N, NFP=self._NFP, sym=self._R_sym)
        self._Z0_basis = FourierSeries(
            N=self._N, NFP=self._NFP, sym=self._Z_sym)
        self._r_basis = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N,
            NFP=self._NFP, sym=self._R_sym, index=self._index)
        self._l_basis = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N,
            NFP=self._NFP, sym=self._Z_sym, index=self._index)
        self._R1_basis = DoubleFourierSeries(
            M=self._M, N=self._N, NFP=self._NFP, sym=self._R_sym)
        self._Z1_basis = DoubleFourierSeries(
            M=self._M, N=self._N, NFP=self._NFP, sym=self._Z_sym)
        self._p_basis = PowerSeries(L=self._p_l.size-1)
        self._i_basis = PowerSeries(L=self._i_l.size-1)

        # format profiles
        self._p_l, self._i_l = format_profiles(profiles, self._p_basis, self._i_basis)

        # format boundary
        self._R1_n, self._Z1_n = format_boundary(
            boundary, self._R1_basis, self._Z1_basis, self.bdry_mode)
        ratio_R1 = np.where(self._R1_basis.modes[:, 2] != 0, self.bdry_ratio, 1)
        ratio_Z1 = np.where(self._Z1_basis.modes[:, 2] != 0, self.bdry_ratio, 1)
        self._R1_n *= ratio_R1
        self._Z1_n *= ratio_Z1

        # solution, if provided
        try:
            self._x = inputs['x']
            self._R0_n, self._Z0_n, self._r_lmn, self._l_lmn = unpack_state(
                self._x, self._R0_basis.num_modes, self._Z0_basis.num_modes,
                self._r_basis.num_modes, self._l_basis.num_modes)
        except:
            try:
                self._R0_n = inputs['R0_n']
                self._Z0_n = inputs['Z0_n']
                self._r_lmn = inputs['r_lmn']
                self._l_lmn = inputs['l_lmn']
            except:
                # create initial guess
                axis = inputs.get('axis', boundary[np.where(boundary[:, 0] == 0)[0], 1:])
                self._R0_n, self._Z0_n = format_axis(axis, self._R0_basis, self._Z0_basis)
                self._r_lmn = np.zeros((self._r_basis.num_modes,))
                self._l_lmn = np.zeros((self._l_basis.num_modes,))
                # TODO: set r_lmn coefficients for initial guess
            self._x = np.concatenate([self._R0_n, self._Z0_n, self._r_lmn, self._l_lmn])

    def change_resolution(self, L:int=None, M:int=None, N:int=None) -> None:
        # TODO: check if resolution actually changes

        if L is not None:
            self._L = L
        if M is not None:
            self._M = M
        if N is not None:
            self._N = N

        old_modes_R = self._R_basis.modes
        old_modes_Z = self._Z_basis.modes
        old_modes_L = self._L_basis.modes
        old_modes_R1 = self._R1_basis.modes
        old_modes_Z1 = self._Z1_basis.modes

        # create bases
        self._R_basis = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N,
            NFP=self._NFP, sym=self._R_sym, index=self._index)
        self._Z_basis = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N,
            NFP=self._NFP, sym=self._Z_sym, index=self._index)
        self._L_basis = DoubleFourierSeries(
            M=self._M, N=self._N, NFP=self._NFP, sym=self._L_sym)
        self._R1_basis = DoubleFourierSeries(
            M=self._M, N=self._N, NFP=self._NFP, sym=self._R_sym)
        self._Z1_basis = DoubleFourierSeries(
            M=self._M, N=self._N, NFP=self._NFP, sym=self._Z_sym)

        def copy_coeffs(c_old, modes_old, modes_new):
            num_modes = modes_new.shape[0]
            c_new = np.zeros((num_modes,))
            for i in range(num_modes):
                idx = np.where(np.all(np.array([
                    np.array(modes_old[:, 0] == modes_new[i, 0]),
                    np.array(modes_old[:, 1] == modes_new[i, 1]),
                    np.array(modes_old[:, 2] == modes_new[i, 2])]), axis=0))[0]
                if len(idx):
                    c_new[i] = c_old[idx[0]]
            return c_new

        self._R0_n = copy_coeffs(self._R0_n, old_modes_R, self._R_basis.modes)
        self._Z0_n = copy_coeffs(self._Z0_n, old_modes_Z, self._Z_basis.modes)
        self._cL = copy_coeffs(self._cL, old_modes_L, self._L_basis.modes)
        self._R1_n = copy_coeffs(self._R1_n, old_modes_R1, self._R1_basis.modes)
        self._Z1_n = copy_coeffs(self._Z1_n, old_modes_Z1, self._Z1_basis.modes)

        # state vector
        self._x = np.concatenate([self._R0_n, self._Z0_n, self._cL])

    @property
    def sym(self) -> bool:
        return self._sym

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x) -> None:
        self._x = x
        self._R0_n, self._Z0_n, self._cL = unpack_state(
            self._x, self._R_basis.num_modes, self._Z_basis.num_modes)

    @property
    def R0_n(self):
        """ spectral coefficients of R """
        return self._R0_n

    @R0_n.setter
    def R0_n(self, R0_n) -> None:
        self._R0_n = R0_n
        self._x = np.concatenate([self._R0_n, self._Z0_n, self._cL])

    @property
    def Z0_n(self):
        """ spectral coefficients of Z """
        return self._Z0_n

    @Z0_n.setter
    def Z0_n(self, Z0_n) -> None:
        self._Z0_n = Z0_n
        self._x = np.concatenate([self._R0_n, self._Z0_n, self._cL])

    @property
    def cL(self):
        """ spectral coefficients of L """
        return self._cL

    @cL.setter
    def cL(self, cL) -> None:
        self._cL = cL
        self._x = np.concatenate([self._R0_n, self._Z0_n, self._cL])

    @property
    def R1_n(self):
        """ spectral coefficients of R at the boundary"""
        return self._R1_n

    @R1_n.setter
    def R1_n(self, R1_n) -> None:
        self._R1_n = R1_n

    @property
    def Z1_n(self):
        """ spectral coefficients of Z at the boundary"""
        return self._Z1_n

    @Z1_n.setter
    def Z1_n(self, Z1_n) -> None:
        self._Z1_n = Z1_n

    @property
    def cP(self):
        """ spectral coefficients of pressure """
        return self._cP

    @cP.setter
    def cP(self, cP) -> None:
        self._cP = cP

    @property
    def cI(self):
        """ spectral coefficients of iota """
        return self._cI

    @cI.setter
    def cI(self, cI) -> None:
        self._cI = cI

    @property
    def Psi(self) -> float:
        """ float, total toroidal flux (in Webers) within LCFS"""
        return self._Psi

    @Psi.setter
    def Psi(self, Psi) -> None:
        self._Psi = Psi

    @property
    def NFP(self) -> int:
        """ int, number of field periods"""
        return self._NFP

    @NFP.setter
    def NFP(self, NFP) -> None:
        self._NFP = NFP

    @property
    def R_basis(self) -> Basis:
        """
        Spectral basis for R

        Returns
        -------
        Basis

        """
        return self._R_basis

    @R_basis.setter
    def R_basis(self, R_basis:Basis) -> None:
        self._R_basis = R_basis

    @property
    def Z_basis(self) -> Basis:
        """
        Spectral basis for Z

        Returns
        -------
        Basis

        """
        return self._Z_basis

    @Z_basis.setter
    def Z_basis(self, Z_basis:Basis) -> None:
        self._Z_basis = Z_basis

    @property
    def L_basis(self) -> Basis:
        """
        Spectral basis for L

        Returns
        -------
        Basis

        """
        return self._L_basis

    @L_basis.setter
    def L_basis(self, L_basis:Basis) -> None:
        self._L_basis = L_basis

    @property
    def R1_basis(self) -> Basis:
        """
        Spectral basis for R at the boundary

        Returns
        -------
        Basis

        """
        return self._R1_basis

    @R1_basis.setter
    def R1_basis(self, R1_basis:Basis) -> None:
        self._R1_basis = R1_basis

    @property
    def Z1_basis(self) -> Basis:
        """
        Spectral basis for Z at the boundary

        Returns
        -------
        Basis

        """
        return self._Z1_basis

    @Z1_basis.setter
    def Z1_basis(self, Z1_basis:Basis) -> None:
        self._Z1_basis = Z1_basis

    @property
    def P_basis(self) -> Basis:
        """
        Spectral basis for pressure

        Returns
        -------
        Basis

        """
        return self._P_basis

    @P_basis.setter
    def P_basis(self, P_basis:Basis) -> None:
        self._P_basis = P_basis

    @property
    def I_basis(self) -> Basis:
        """
        Spectral basis for iota

        Returns
        -------
        Basis

        """
        return self._I_basis

    @I_basis.setter
    def I_basis(self, I_basis:Basis) -> None:
        self._I_basis = I_basis

    def compute_coordinates(self, grid: Grid) -> dict:
        """Converts from spectral to real space by calling :func:`desc.configuration.compute_coordinates` 

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes at which to evaluate R and Z.

        Returns
        -------
        coords : dict
            dictionary of ndarray, shape(N_nodes,) of coordinates evaluated at node locations.
            keys are of the form 'X_y' meaning the derivative of X wrt to y

        """

        R_transform = Transform(grid, self._R_basis, derivs=0)
        Z_transform = Transform(grid, self._Z_basis, derivs=0)
        coords = compute_coordinates(self._R0_n, self._Z0_n, R_transform,
                                     Z_transform)
        return coords

    def compute_coordinate_derivatives(self, grid: Grid) -> dict:
        """Converts from spectral to real space and evaluates derivatives of R,Z wrt to SFL coords by calling :func:`desc.configuration.compute_coordinate_derivatives`

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes at which to evaluate derivatives.

        Returns
        -------
        coord_der : dict
            dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations.
            keys are of the form 'X_y' meaning the derivative of X wrt to y

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        return coord_der

    def compute_covariant_basis(self, grid: Grid) -> dict:
        """Computes covariant basis vectors at grid points by calling :func:`desc.configuration.compute_covariant_basis`


        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes at which to find the covariant basis vectors.

        Returns
        -------
        cov_basis : dict
            dictionary of ndarray containing covariant basis
            vectors and derivatives at each node. Keys are of the form 'e_x_y',
            meaning the unit vector in the x direction, differentiated wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        return cov_basis

    def compute_contravariant_basis(self, grid: Grid) -> dict:
        """Computes contravariant basis vectors and jacobian elements by calling :func:`desc.configuration.compute_contravariant_basis`

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of 
            the nodes at which to find the contravariant basis vectors and the 
            jacobian elements.

        Returns
        -------
        con_basis : dict
            dictionary of ndarray containing contravariant basis vectors and jacobian elements

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        con_basis = compute_contravariant_basis(
            coord_der, cov_basis, jacobian, axis=grid.axis)
        return con_basis

    def compute_jacobian(self, grid: Grid) -> dict:
        """Computes coordinate jacobian and derivatives by calling :func:`desc.configuration.compute_jacobian`

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of 
            the nodes at which to find the coordinate jacobian elements and its
            partial derivatives.

        Returns
        -------
        jacobian : dict
            dictionary of ndarray, shape(N_nodes,) of coordinate
            jacobian and partial derivatives. Keys are of the form `g_x` meaning
            the x derivative of the coordinate jacobian g

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        return jacobian

    def compute_magnetic_field(self, grid: Grid) -> dict:
        """Computes magnetic field components at node locations by calling :func:`desc.configuration.compute_magnetic_field`

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of 
            the nodes at which to evaluate the magnetic field components

        Returns
        -------
        magnetic_field: dict
            dictionary of ndarray, shape(N_nodes,) of magnetic field
            and derivatives. Keys are of the form 'B_x_y' or 'B^x_y', meaning the
            covariant (B_x) or contravariant (B^x) component of the magnetic field, with the derivative wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        I_transform = Transform(grid, self._I_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self._cI,
                                                self._Psi, I_transform)
        return magnetic_field

    def compute_plasma_current(self, grid: Grid) -> dict:
        """Computes current density field at node locations by calling :func:`desc.configuration.compute_plasma_current`

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of 
            the nodes at which to evaluate the plasma current components

        Returns
        -------
        plasma_current : dict
            dictionary of ndarray, shape(N_nodes,) of current field.
            Keys are of the form 'J^x_y' meaning the contravariant (J^x)
            component of the current, with the derivative wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        I_transform = Transform(grid, self._I_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self._cI,
                                                self._Psi, I_transform)
        plasma_current = compute_plasma_current(coord_der, cov_basis, jacobian,
                                        magnetic_field, self._cI, I_transform)
        return plasma_current

    def compute_magnetic_field_magnitude(self, grid: Grid) -> dict:
        """Computes magnetic field magnitude at node locations by calling :func:`desc.configuration.compute_magnetic_field_magnitude`

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of 
            the nodes at which to evaluate the magnetic field magnitude and derivatives

        Returns
        -------
        magnetic_field_mag : dict
            dictionary of ndarray, shape(N_nodes,) of magnetic field magnitude and derivatives

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        I_transform = Transform(grid, self._I_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self._cI,
                                                self._Psi, I_transform)
        magnetic_field_mag = compute_magnetic_field_magnitude(cov_basis,
                                      magnetic_field, self._cI, I_transform)
        return magnetic_field_mag

    def compute_force_magnitude(self, grid: Grid) -> dict:
        """Computes force error magnitude at node locations by calling :func:`desc.configuration.compute_force_magnitude`

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of 
            the nodes at which to evaluate the force error magnitudes

        Returns
        -------
        force_mag : dict
            dictionary of ndarray, shape(N_nodes,) of force magnitudes

        """
        R_transform = Transform(grid, self._R_basis, derivs=3)
        Z_transform = Transform(grid, self._Z_basis, derivs=3)
        I_transform = Transform(grid, self._I_basis, derivs=1)
        P_transform = Transform(grid, self._P_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self._R0_n, self._Z0_n,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        con_basis = compute_contravariant_basis(coord_der, cov_basis, jacobian,
                                                axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self._cI,
                                                self._Psi, I_transform)
        plasma_current = compute_plasma_current(coord_der, cov_basis, jacobian,
                                        magnetic_field, self._cI, I_transform)
        force_mag = compute_force_magnitude(coord_der, cov_basis, con_basis,
            jacobian, magnetic_field, plasma_current, self._cP, P_transform)
        return force_mag


class Equilibrium(Configuration, IOAble):
    """Equilibrium is a decorator design pattern on top of Configuration.
       It adds information about how the equilibrium configuration was solved.
    """
    _io_attrs_ = Configuration._io_attrs_ \
               + ['_initial', '_objective', '_optimizer', '_solved']
    _object_lib_ = Configuration._object_lib_
    _object_lib_.update({'Configuration' : Configuration,
                         'ObjectiveFunction' : ObjectiveFunction})

    def __init__(self, inputs:dict=None, load_from=None,
                 file_format:str='hdf5', obj_lib=None) -> None:
        super().__init__(inputs=inputs, load_from=load_from, file_format=file_format, obj_lib=obj_lib)

    def _init_from_inputs_(self, inputs:dict=None) -> None:
        super()._init_from_inputs_(inputs=inputs)
        self._x0 = self._x
        self._objective = inputs.get('objective', None)
        self._optimizer = inputs.get('optimizer', None)
        self.optimizer_results = {}
        self._solved = False

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0) -> None:
        self._x0 = x0

    @property
    def solved(self) -> bool:
        return self._solved

    @solved.setter
    def solved(self, solved) -> None:
        self._solved = solved

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        self._objective = objective
        self.solved = False

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.solved = False

    @property
    def initial(self) -> Configuration:
        """
        Initial Configuration from which the Equilibrium was solved

        Returns
        -------
        Configuration

        """
        bdryR = np.array([self._R1_basis.modes[:, 1:2],
                          self._R1_n, np.zeros_like(self._R1_n)]).T
        bdryZ = np.array([self._Z1_basis.modes[:, 1:2],
                          np.zeros_like(self._R1_n), self._Z1_n]).T
        inputs = {'L': self._L,
                  'M': self._M,
                  'N': self._N,
                  'cP': self._cP,
                  'cI': self._cI,
                  'Psi': self._Psi,
                  'NFP': self._NFP,
                  'bdry': np.vstack((bdryR, bdryZ)),
                  'sym': self._sym,
                  'index': self._index,
                  'x': self._x0
                 }
        return Configuration(inputs=inputs)

    def optimize(self):
        pass

    def solve(self):
        if self._optimizer is None or self._objective is None:
            raise AttributeError(
                "Equilibrium must have objective and optimizer defined before solving.")
        args = (self.R1_n, self.Z1_n, self.cP, self.cI, self.Psi,
                self.bdry_ratio, self.pres_ratio, self.zeta_ratio, self.errr_ratio)

        result = self._optimizer.optimize(self._objective, self.x, args=args)
        self.optimizer_results = result
        self.x = result['x']
        self.solved = True  # TODO: do we still call it solved if the solver exited early?
        return result

# XXX: Should this (also) inherit from Equilibrium?
class EquilibriaFamily(IOAble, MutableSequence):
    """EquilibriaFamily stores a list of Equilibria
    """
    _io_attrs_ = ['equilibria']
    _object_lib_ = Equilibrium._object_lib_
    _object_lib_.update({'Equilibrium': Equilibrium})

    def __init__(self, inputs=None, load_from=None, file_format='hdf5',
                 obj_lib=None) -> None:
        self._file_format_ = file_format
        if load_from is None:
            self._init_from_inputs_(inputs=inputs)
        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib)

    def _init_from_inputs_(self, inputs=None):
        self._equilibria = []
        self.append(Equilibrium(inputs=inputs))
        return None

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self._equilibria[i]

    def __setitem__(self, i, new_item):
        if isinstance(new_item, Configuration):
            self._equilibria[i] = new_item
        else:
            raise ValueError(
                "Members of EquilibriaFamily should be of type Configuration or a subclass")

    def __delitem__(self, i):
        del self._equilibria[i]

    def __len__(self):
        return len(self._equilibria)

    def insert(self, i, new_item):
        self._equilibria.insert(i, new_item)

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    @property
    def equilibria(self):
        return self._equilibria

    @equilibria.setter
    def equilibria(self, eq):
        self._equilibria = eq

    def __slice__(self, idx):
        if idx is None:
            theslice = slice(None, None)
        elif type(idx) is int:
            theslice = idx
        elif type(idx) is list:
            try:
                theslice = slice(idx[0], idx[1], idx[2])
            except IndexError:
                theslice = slice(idx[0], idx[1])
        else:
            raise TypeError('index is not a valid type.')
        return theslice

# TODO: overwrite all Equilibrium methods and default to self._equilibria[-1]

# these functions are needed to format the input arrays

def format_profiles(profiles, p_basis:PowerSeries, i_basis:PowerSeries):
    """Formats profile input arrays

    Parameters
    ----------
    profiles : ndarray, shape(Nbdry,3)
        array of fourier coeffs [l, p, i]
    p_basis : PowerSeries
        spectral basis for p_l coefficients
    i_basis : PowerSeries
        spectral basis for i_l coefficients

    Returns
    -------
    p_l : ndarray
        spectral coefficients for pressure profile
    i_l : ndarray
        spectral coefficients for rotational transform profile

    """
    p_l = np.zeros((p_basis.num_modes,))
    i_l = np.zeros((i_basis.num_modes,))

    for l, p, i in profiles:
        idx_p = np.where(p_basis.modes[:, 0] == int(l))[0]
        idx_i = np.where(i_basis.modes[:, 0] == int(l))[0]
        p_l = put(p_l, idx_p, p)
        i_l = put(i_l, idx_i, i)

    return p_l, i_l


def format_boundary(boundary, R1_basis:DoubleFourierSeries,
                    Z1_basis:DoubleFourierSeries, mode:str='spectral'):
    """Formats boundary arrays and converts between real and fourier representations

    Parameters
    ----------
    boundary : ndarray, shape(Nbdry,4)
        array of fourier coeffs [m, n, R1, Z1]
        or array of real space coordinates, [theta, phi, R, Z]
    R1_basis : DoubleFourierSeries
        spectral basis for R1_mn coefficients
    Z1_basis : DoubleFourierSeries
        spectral basis for Z1_mn coefficients
    mode : str
        one of 'real', 'spectral'. Whether bdry is specified in real or spectral space.

    Returns
    -------
    R1_mn : ndarray
        spectral coefficients for R boundary
    Z1_mn : ndarray
        spectral coefficients for Z boundary

    """
    if mode == 'real':
        theta = boundary[:, 0]
        phi = boundary[:, 1]
        rho = np.ones_like(theta)

        nodes = np.array([rho, theta, phi]).T
        grid = Grid(nodes)
        R1_tform = Transform(grid, R1_basis)
        Z1_tform = Transform(grid, Z1_basis)

        # fit real data to spectral coefficients
        R1_mn = R1_tform.fit(boundary[:, 2])
        Z1_mn = Z1_tform.fit(boundary[:, 3])

    else:
        R1_mn = np.zeros((R1_basis.num_modes,))
        Z1_mn = np.zeros((Z1_basis.num_modes,))

        for m, n, R1, Z1 in boundary:
            idx_R = np.where(np.logical_and(R1_basis.modes[:, 1] == int(m),
                                            R1_basis.modes[:, 2] == int(n)))[0]
            idx_Z = np.where(np.logical_and(Z1_basis.modes[:, 1] == int(m),
                                            Z1_basis.modes[:, 2] == int(n)))[0]
            R1_mn = put(R1_mn, idx_R, R1)
            Z1_mn = put(Z1_mn, idx_Z, Z1)

    return R1_mn, Z1_mn


def format_axis(axis, R0_basis:FourierSeries, Z0_basis:FourierSeries):
    """Formats magnetic axis input arrays

    Parameters
    ----------
    axis : ndarray, shape(Nbdry,3)
        array of fourier coeffs [l, p, i]
    R0_basis : FourierSeries
        spectral basis for R0_n coefficients
    Z0_basis : PowerSeries
        spectral basis for Z0_n coefficients

    Returns
    -------
    R0_n : ndarray
        spectral coefficients for magnetic axis R coordinate
    Z0_n : ndarray
        spectral coefficients for magnetic axis Z coordinate

    """
    R0_n = np.zeros((R0_basis.num_modes,))
    Z0_n = np.zeros((Z0_basis.num_modes,))

    for n, R0, Z0 in axis:
        idx_R0 = np.where(R0_basis.modes[:, 0] == int(n))[0]
        idx_Z0 = np.where(Z0_basis.modes[:, 0] == int(n))[0]
        R0_n = put(R0_n, idx_R0, R0)
        Z0_n = put(Z0_n, idx_Z0, Z0)

    return R0_n, Z0_n