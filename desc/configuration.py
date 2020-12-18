import numpy as np
from collections.abc import MutableSequence

from desc.backend import TextColors, Tristate, unpack_state
from desc.basis import Basis, PowerSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.grid import Grid, LinearGrid, ConcentricGrid
from desc.transform import Transform
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry
from desc.objective_funs import ObjectiveFunction
from desc.equilibrium_io import IOAble
from desc.compute_funs import compute_coordinates, compute_coordinate_derivatives
from desc.compute_funs import compute_covariant_basis, compute_contravariant_basis
from desc.compute_funs import compute_jacobian, compute_magnetic_field, compute_plasma_current
from desc.compute_funs import compute_magnetic_field_magnitude, compute_force_magnitude


class Configuration(IOAble):
    """Configuration contains information about a plasma state, including the
       shapes of flux surfaces and profile inputs. It can compute additional
       information, such as the magnetic field and plasma currents.
    """

    _io_attrs_ = ['_cR', '_cZ', '_cL', '_cRb', '_cZb', '_cP', '_cI', '_Psi', '_NFP',
                  '_R_basis', '_Z_basis', '_L_basis', '_Rb_basis', '_Zb_basis',
                  '_P_basis', '_I_basis']
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
                cP : ndarray, pressure spectral coefficients, indexed as (lm,n) flattened in row major order
                cI : ndarray, iota spectral coefficients, indexed as (lm,n) flattened in row major order
                Psi : float, total toroidal flux (in Webers) within LCFS
                NFP : int, number of field periods
                bdry : ndarray, array of fourier coeffs [m,n,Rcoeff, Zcoeff]
            And the following optional keys:
                sym : bool, is the problem stellarator symmetric or not, default is False
                index : str, type of Zernike indexing scheme to use, default is 'ansi'
                bdry_mode : str, how to calculate error at bdry, default is 'spectral'
                bdry_ratio :
                axis :
                cR : ndarray, spectral coefficients of R
                cZ : ndarray, spectral coefficients of Z
                cL : ndarray, spectral coefficients of L
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
            self._L = inputs['L']
            self._M = inputs['M']
            self._N = inputs['N']
            self._cP = inputs['cP']
            self._cI = inputs['cI']
            self._Psi = inputs['Psi']
            self._NFP = inputs['NFP']
            bdry = inputs['bdry']
        except:
            raise ValueError(TextColors.FAIL +
                             "input dict does not contain proper keys"
                           + TextColors.ENDC)

        # optional inputs
        self._sym = inputs.get('sym', False)
        self._index = inputs.get('index', 'ansi')
        bdry_mode = inputs.get('bdry_mode', 'spectral')
        bdry_ratio = inputs.get('bdry_ratio', 1.0)
        axis = inputs.get('axis', bdry[np.where(bdry[:, 0] == 0)[0], 1:])

        # stellarator symmetry for bases
        if self._sym:
            self._R_sym = Tristate(True)
            self._Z_sym = Tristate(False)
            self._L_sym = Tristate(False)
        else:
            self._R_sym = Tristate(None)
            self._Z_sym = Tristate(None)
            self._L_sym = Tristate(None)

        # create bases
        self._R_basis = FourierZernikeBasis(
                    L=self._L, M=self._M, N=self._N,
                    NFP=self._NFP, sym=self._R_sym, index=self._index)
        self._Z_basis = FourierZernikeBasis(
                    L=self._L, M=self._M, N=self._N,
                    NFP=self._NFP, sym=self._Z_sym, index=self._index)
        self._L_basis = DoubleFourierSeries(
                    M=self._M, N=self._N, NFP=self._NFP, sym=self._L_sym)
        self._Rb_basis = DoubleFourierSeries(
                    M=self._M, N=self._N, NFP=self._NFP, sym=self._R_sym)
        self._Zb_basis = DoubleFourierSeries(
                    M=self._M, N=self._N, NFP=self._NFP, sym=self._Z_sym)
        self._P_basis = PowerSeries(L=self._cP.size-1)
        self._I_basis = PowerSeries(L=self._cI.size-1)

        # format boundary
        self._cRb, self._cZb = format_bdry(
                            bdry, self._Rb_basis, self._Zb_basis, bdry_mode)
        ratio_Rb = np.where(self._Rb_basis.modes[:, 2] != 0, bdry_ratio, 1)
        ratio_Zb = np.where(self._Zb_basis.modes[:, 2] != 0, bdry_ratio, 1)
        self._cRb *= ratio_Rb
        self._cZb *= ratio_Zb

        # solution, if provided
        try:
            self._cR = inputs['cR']
            self._cZ = inputs['cZ']
            self._cL = inputs['cL']
        except:
            self._cR, self._cZ = get_initial_guess_scale_bdry(
                        axis, bdry, bdry_ratio, self._R_basis, self._Z_basis)
            self._cL = np.zeros((self._L_basis.num_modes,))

        # state vector
        self._x = np.concatenate([self._cR, self._cZ, self._cL])

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
        old_modes_Rb = self._Rb_basis.modes
        old_modes_Zb = self._Zb_basis.modes

        # create bases
        self._R_basis = FourierZernikeBasis(
                    L=self._L, M=self._M, N=self._N,
                    NFP=self._NFP, sym=self._R_sym, index=self._index)
        self._Z_basis = FourierZernikeBasis(
                    L=self._L, M=self._M, N=self._N,
                    NFP=self._NFP, sym=self._Z_sym, index=self._index)
        self._L_basis = DoubleFourierSeries(
                    M=self._M, N=self._N, NFP=self._NFP, sym=self._L_sym)
        self._Rb_basis = DoubleFourierSeries(
                    M=self._M, N=self._N, NFP=self._NFP, sym=self._R_sym)
        self._Zb_basis = DoubleFourierSeries(
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

        self._cR = copy_coeffs(self._cR, old_modes_R, self._R_basis.modes)
        self._cZ = copy_coeffs(self._cZ, old_modes_Z, self._Z_basis.modes)
        self._cL = copy_coeffs(self._cL, old_modes_L, self._L_basis.modes)
        self._cRb = copy_coeffs(self._cRb, old_modes_Rb, self._Rb_basis.modes)
        self._cZb = copy_coeffs(self._cZb, old_modes_Zb, self._Zb_basis.modes)

        # state vector
        self._x = np.concatenate([self._cR, self._cZ, self._cL])

    @property
    def sym(self) -> bool:
        return self._sym

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x) -> None:
        self._x = x
        self._cR, self._cZ, self._cL = unpack_state(
                self._x, self._R_basis.num_modes, self._Z_basis.num_modes)

    @property
    def cR(self):
        """ spectral coefficients of R """
        return self._cR

    @cR.setter
    def cR(self, cR) -> None:
        self._cR = cR

    @property
    def cZ(self):
        """ spectral coefficients of Z """
        return self._cZ

    @cZ.setter
    def cZ(self, cZ) -> None:
        self._cZ = cZ
        
    @property
    def cL(self):
        """ spectral coefficients of L """
        return self._cL

    @cL.setter
    def cL(self, cL) -> None:
        self._cL = cL

    @property
    def cRb(self):
        """ spectral coefficients of R at the boundary"""
        return self._cRb

    @cRb.setter
    def cRb(self, cRb) -> None:
        self._cRb = cRb

    @property
    def cZb(self):
        """ spectral coefficients of Z at the boundary"""
        return self._cZb

    @cZb.setter
    def cZb(self, cZb) -> None:
        self._cZb = cZb

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
    def Rb_basis(self) -> Basis:
        """
        Spectral basis for R at the boundary

        Returns
        -------
        Basis

        """
        return self._Rb_basis

    @Rb_basis.setter
    def Rb_basis(self, Rb_basis:Basis) -> None:
        self._Rb_basis = Rb_basis

    @property
    def Zb_basis(self) -> Basis:
        """
        Spectral basis for Z at the boundary

        Returns
        -------
        Basis

        """
        return self._Zb_basis

    @Zb_basis.setter
    def Zb_basis(self, Zb_basis:Basis) -> None:
        self._Zb_basis = Zb_basis

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

    def compute_coordinates(self, grid:Grid) -> dict:
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
        coords = compute_coordinates(self._cR, self._cZ, R_transform,
                                     Z_transform)
        return coords

    def compute_coordinate_derivatives(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
                                                   R_transform, Z_transform)
        return coord_der

    def compute_covariant_basis(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        return cov_basis

    def compute_contravariant_basis(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        con_basis = compute_contravariant_basis(coord_der, cov_basis, jacobian, axis=grid.axis)
        return con_basis

    def compute_jacobian(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        return jacobian

    def compute_magnetic_field(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self._cI,
                                                self._Psi, I_transform)
        return magnetic_field

    def compute_plasma_current(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self._cI,
                                                self._Psi, I_transform)
        plasma_current = compute_plasma_current(coord_der, cov_basis, jacobian,
                                        magnetic_field, self._cI, I_transform)
        return plasma_current

    def compute_magnetic_field_magnitude(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self._cI,
                                                self._Psi, I_transform)
        magnetic_field_mag = compute_magnetic_field_magnitude(cov_basis,
                                      magnetic_field, self._cI, I_transform)
        return magnetic_field_mag

    def compute_force_magnitude(self, grid:Grid) -> dict:
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
        coord_der = compute_coordinate_derivatives(self._cR, self._cZ,
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


class Equilibrium(Configuration,IOAble):
    """Equilibrium is a decorator design pattern on top of Configuration.
       It adds information about how the equilibrium configuration was solved.
    """
    _io_attrs_ = Configuration._io_attrs_ + ['_initial', '_objective', '_optimizer', '_solved']
    _object_lib_ = Configuration._object_lib_
    _object_lib_.update({'Configuration' : Configuration,
                         'ObjectiveFunction' : ObjectiveFunction})

    def __init__(self, inputs:dict=None, load_from=None,
                 file_format:str='hdf5', obj_lib=None) -> None:
        super().__init__(inputs=inputs, load_from=load_from, file_format=file_format, obj_lib=obj_lib)

    def _init_from_inputs_(self, inputs:dict=None) -> None:
        super()._init_from_inputs_(inputs=inputs)
        self._initial = Configuration(inputs=inputs)
        self._objective = inputs.get('objective', None)
        self._optimizer = inputs.get('optimizer', None)
        self._solved = False

    @property
    def solved(self) -> bool:
        return self._solved

    @solved.setter
    def solved(self, issolved):
        self._solved = issolved

    @property
    def initial(self) -> Configuration:
        """
        Initial Configuration from which the Equilibrium was solved

        Returns
        -------
        Configuration

        """
        return self._initial

    @initial.setter
    def initial(self, config:Configuration) -> None:
        self._initial = config

    @property
    def x(self):
        """ State vector of (cR,cZ,cL) """
        return self._x

    @x.setter
    def x(self, x) -> None:
        self._x = x
        self._cR, self._cZ, self._cL = \
            unpack_state(self._x,
                         self._R_basis.num_modes,
                         self._Z_basis.num_modes)
        self._solved = True

    @property
    def solved(self) -> bool:
        """Boolean, if the Equilibrium has been solved or not"""
        return self._solved

    @property
    def initial(self) -> Configuration:
        return self._initial

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x) -> None:
        self._x = x
        self._cR, self._cZ, self._cL = \
            unpack_state(self._x,
                         self._R_basis.num_modes,
                         self._Z_basis.num_modes)
        self._solved = True

    def optimize(self):
        pass

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


# XXX: Should this (also) inherit from Equilibrium?
class EquilibriaFamily(IOAble, MutableSequence):
    """EquilibriaFamily stores a list of Equilibria
    """
    _io_attrs_ = ['equilibria']
    _object_lib_ = Equilibrium._object_lib_
    _object_lib_.update({'Equilibrium' : Equilibrium})

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
        # add type checking
        self._equilibria[i] = new_item

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
            theslice = slice(None,None)
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
