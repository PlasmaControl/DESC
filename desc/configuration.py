import numpy as np
from collections.abc import MutableSequence

from desc.backend import jnp, put, opsindex, cross, dot, TextColors, Tristate
from desc.basis import Basis, PowerSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.grid import Grid, LinearGrid, ConcentricGrid
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry
#from desc import equilibrium_io as eq_io
from desc.equilibrium_io import IOAble, reader_factory, writer_factory


def unpack_state(x, nR, nZ):
    """Unpacks the optimization state vector x into cR, cZ, cL components

    Parameters
    ----------
    x : ndarray
        vector to unpack: x = [cR, cZ, cL]
    nR : int
        number of cR coefficients
    nZ : int
        number of cZ coefficients

    Returns
    -------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    cL : ndarray
        spectral coefficients of lambda

    """

    cR = x[:nR]
    cZ = x[nR:nR+nZ]
    cL = x[nR+nZ:]
    return cR, cZ, cL

class Configuration(IOAble):
    """Configuration constains information about a plasma state, including the
       shapes of flux surfaces and profile inputs. It can compute additional
       information, such as the magnetic field and plasma currents.
    """
    _save_attrs_ = ['cR', 'cZ', 'cL', 'cRb', 'cZb', 'cP',
                             'cI', 'Psi', 'NFP', 'R_basis',
                             'Z_basis', 'L_basis', 'Rb_basis',
                             'Zb_basis', 'P_basis', 'I_basis']
    _object_lib_ = {'PowerSeries' : PowerSeries,
                            'DoubleFourierSeries'   : DoubleFourierSeries,
                            'FourierZernikeBasis'   : FourierZernikeBasis,
                            'LinearGrid'            : LinearGrid,
                            'ConcentricGrid'        : ConcentricGrid}

    def __init__(self, inputs:dict=None, load_from=None, file_format:str='hdf5', obj_lib=None) -> None:
        """Initializes a Configuration

        Parameters
        ----------
        inputs : dict
            DESCRIPTION
        load_from : str OR ...
            DESCRIPTION
        file_format : str
            DESCRIPTION

        Returns
        -------
        None

        """
        #self._def_save_attrs_()

        self.inputs = inputs
        self.load_from = load_from
        if inputs is not None:
            self._init_from_inputs_()
        elif load_from is not None:
            if file_format is None:
                raise RuntimeError('file_format argument must be included when loading from file.')
            self._file_format_ = file_format
            self._init_from_file_(obj_lib=obj_lib)
        else:
            raise RuntimeError('inputs or load_from must be specified.')

    def _def_save_attrs_(self) -> None:
        """Defines attributes to save

        Returns
        -------
        None

        """
        self._save_attrs_ = ['cR', 'cZ', 'cL', 'cRb', 'cZb', 'cP',
                             'cI', 'Psi', 'NFP', 'R_basis',
                             'Z_basis', 'L_basis', 'Rb_basis',
                             'Zb_basis', 'P_basis', 'I_basis']

    def _init_from_inputs_(self, inputs:dict=None) -> None:
        """

        Parameters
        ----------
        inputs : dict, optional
            Dictionary of inputs with the following required keys:
                L :
                M :
                N :
                cP :
                cI :
                Psi :
                NFP :
                bdry :
            And the following optional keys:
                sym :
                index :
                bdry_mode :
                bdry_ratio :
                axis :
                cR :
                cZ :
                cL :

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None

        """
        if inputs is None:
            inputs = self.inputs

        # required keys
        try:
            self.__L = inputs['L']
            self.__M = inputs['M']
            self.__N = inputs['N']
            self.__cP = inputs['cP']
            self.__cI = inputs['cI']
            self.__Psi = inputs['Psi']
            self.__NFP = inputs['NFP']
            bdry = inputs['bdry']
        except:
            raise ValueError(TextColors.FAIL +
                             "input dict does not contain proper keys"
                           + TextColors.ENDC)

        # optional keys
        self.__sym = inputs.get('sym', False)
        self.__index = inputs.get('index', 'ansi')
        bdry_mode = inputs.get('bdry_mode', 'spectral')
        bdry_ratio = inputs.get('bdry_ratio', 1.0)
        axis = inputs.get('axis', bdry[np.where(bdry[:, 0] == 0)[0], 1:])

        # stellarator symmetry for bases
        if self.__sym:
            self.__R_sym = Tristate(True)
            self.__Z_sym = Tristate(False)
            self.__L_sym = Tristate(False)
        else:
            self.__R_sym = Tristate(None)
            self.__Z_sym = Tristate(None)
            self.__L_sym = Tristate(None)

        # create bases
        self.__R_basis = FourierZernikeBasis(
                    L=self.__L, M=self.__M, N=self.__N,
                    NFP=self.__NFP, sym=self.__R_sym, index=self.__index)
        self.__Z_basis = FourierZernikeBasis(
                    L=self.__L, M=self.__M, N=self.__N,
                    NFP=self.__NFP, sym=self.__Z_sym, index=self.__index)
        self.__L_basis = DoubleFourierSeries(
                    M=self.__M, N=self.__N, NFP=self.__NFP, sym=self.__L_sym)
        self.__Rb_basis = DoubleFourierSeries(
                    M=self.__M, N=self.__N, NFP=self.__NFP, sym=self.__R_sym)
        self.__Zb_basis = DoubleFourierSeries(
                    M=self.__M, N=self.__N, NFP=self.__NFP, sym=self.__Z_sym)
        self.__P_basis = PowerSeries(L=self.__cP.size-1)
        self.__I_basis = PowerSeries(L=self.__cI.size-1)

        # format boundary
        self.__cRb, self.__cZb = format_bdry(
                            bdry, self.__Rb_basis, self.__Zb_basis, bdry_mode)
        ratio_Rb = np.where(self.__Rb_basis.modes[:, 2] != 0, bdry_ratio, 1)
        ratio_Zb = np.where(self.__Zb_basis.modes[:, 2] != 0, bdry_ratio, 1)
        self.__cRb *= ratio_Rb
        self.__cZb *= ratio_Zb

        # solution, if provided
        try:
            self.__cR = inputs['cR']
            self.__cZ = inputs['cZ']
            self.__cL = inputs['cL']
        except:
            self.__cR, self.__cZ = get_initial_guess_scale_bdry(
                        axis, bdry, bdry_ratio, self.__R_basis, self.__Z_basis)
            self.__cL = np.zeros((self.__L_basis.num_modes,))

        # state vector
        self.__x = np.concatenate([self.__cR, self.__cZ, self.__cL])

    def change_resolution(self, L:int=None, M:int=None, N:int=None) -> None:
        # TODO: check if resolution actually changes

        if L is not None:
            self.__L = L
        if M is not None:
            self.__M = M
        if N is not None:
            self.__N = N

        old_modes_R = self.__R_basis.modes
        old_modes_Z = self.__Z_basis.modes
        old_modes_L = self.__L_basis.modes
        old_modes_Rb = self.__Rb_basis.modes
        old_modes_Zb = self.__Zb_basis.modes

        # create bases
        self.__R_basis = FourierZernikeBasis(
                    L=self.__L, M=self.__M, N=self.__N,
                    NFP=self.__NFP, sym=self.__R_sym, index=self.__index)
        self.__Z_basis = FourierZernikeBasis(
                    L=self.__L, M=self.__M, N=self.__N,
                    NFP=self.__NFP, sym=self.__Z_sym, index=self.__index)
        self.__L_basis = DoubleFourierSeries(
                    M=self.__M, N=self.__N, NFP=self.__NFP, sym=self.__L_sym)
        self.__Rb_basis = DoubleFourierSeries(
                    M=self.__M, N=self.__N, NFP=self.__NFP, sym=self.__R_sym)
        self.__Zb_basis = DoubleFourierSeries(
                    M=self.__M, N=self.__N, NFP=self.__NFP, sym=self.__Z_sym)

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

        self.__cR = copy_coeffs(self.__cR, old_modes_R, self.__R_basis.modes)
        self.__cZ = copy_coeffs(self.__cZ, old_modes_Z, self.__Z_basis.modes)
        self.__cL = copy_coeffs(self.__cL, old_modes_L, self.__L_basis.modes)
        self.__cRb = copy_coeffs(self.__cRb, old_modes_Rb, self.__Rb_basis.modes)
        self.__cZb = copy_coeffs(self.__cZb, old_modes_Zb, self.__Zb_basis.modes)

        # state vector
        self.__x = np.concatenate([self.__cR, self.__cZ, self.__cL])

    @property
    def sym(self) -> bool:
        return self.__sym

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x) -> None:
        self.__x = x
        self.__cR, self.__cZ, self.__cL = unpack_state(
                self.__x, self.__R_basis.num_modes, self.__Z_basis.num_modes)

    @property
    def cR(self):
        return self.__cR

    @cR.setter
    def cR(self, cR) -> None:
        self.__cR = cR

    @property
    def cZ(self):
        return self.__cZ

    @cZ.setter
    def cZ(self, cZ) -> None:
        self.__cZ = cZ

    @property
    def cRb(self):
        return self.__cRb

    @cRb.setter
    def cRb(self, cRb) -> None:
        self.__cRb = cRb

    @property
    def cZb(self):
        return self.__cZb

    @cZb.setter
    def cZb(self, cZb) -> None:
        self.__cZb = cZb

    @property
    def cP(self):
        return self.__cP

    @cP.setter
    def cP(self, cP) -> None:
        self.__cP = cP

    @property
    def cI(self):
        return self.__cI

    @cI.setter
    def cI(self, cI) -> None:
        self.__cI = cI

    @property
    def Psi(self) -> float:
        return self.__Psi

    @Psi.setter
    def Psi(self, Psi) -> None:
        self.__Psi = Psi

    @property
    def NFP(self) -> int:
        return self.__NFP

    @NFP.setter
    def NFP(self, NFP) -> None:
        self.__NFP = NFP

    @property
    def R_basis(self) -> Basis:
        return self.__R_basis

    @R_basis.setter
    def R_basis(self, R_basis:Basis) -> None:
        self.__R_basis = R_basis

    @property
    def Z_basis(self) -> Basis:
        return self.__Z_basis

    @Z_basis.setter
    def Z_basis(self, Z_basis:Basis) -> None:
        self.__Z_basis = Z_basis

    @property
    def L_basis(self) -> Basis:
        return self.__L_basis

    @L_basis.setter
    def L_basis(self, L_basis:Basis) -> None:
        self.__L_basis = L_basis

    @property
    def Rb_basis(self) -> Basis:
        return self.__Rb_basis

    @Rb_basis.setter
    def Rb_basis(self, Rb_basis:Basis) -> None:
        self.__Rb_basis = Rb_basis

    @property
    def Zb_basis(self) -> Basis:
        return self.__Zb_basis

    @Zb_basis.setter
    def Zb_basis(self, Zb_basis:Basis) -> None:
        self.__Zb_basis = Zb_basis

    @property
    def P_basis(self) -> Basis:
        return self.__P_basis

    @P_basis.setter
    def P_basis(self, P_basis:Basis) -> None:
        self.__P_basis = P_basis

    @property
    def I_basis(self) -> Basis:
        return self.__I_basis

    @I_basis.setter
    def I_basis(self, I_basis:Basis) -> None:
        self.__I_basis = I_basis

    def compute_coordinates(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=0)
        Z_transform = Transform(grid, self.__Z_basis, derivs=0)
        coords = compute_coordinates(self.__cR, self.__cZ, R_transform,
                                     Z_transform)
        return coords

    def compute_coordinate_derivatives(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        return coord_der

    def compute_covariant_basis(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        return cov_basis

    def compute_contravariant_basis(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        con_basis = compute_contravariant_basis(coord_der, cov_basis, jacobian, axis=grid.axis)
        return con_basis

    def compute_jacobian(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        return jacobian

    def compute_magnetic_field(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        I_transform = Transform(grid, self.__I_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self.__cI,
                                                self.__Psi_lcfs, I_transform)
        return magnetic_field

    def compute_plasma_current(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        I_transform = Transform(grid, self.__I_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self.__cI,
                                                self.__Psi_lcfs, I_transform)
        plasma_current = compute_plasma_current(coord_der, cov_basis, jacobian,
                                        magnetic_field, self.__cI, I_transform)
        return plasma_current

    def compute_magnetic_field_magnitude(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        I_transform = Transform(grid, self.__I_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self.__cI,
                                                self.__Psi_lcfs, I_transform)
        magnetic_field_mag = compute_magnetic_field_magnitude(cov_basis,
                                      magnetic_field, self.__cI, I_transform)
        return magnetic_field_mag

    def compute_force_magnitude(self, grid:Grid) -> dict:
        R_transform = Transform(grid, self.__R_basis, derivs=3)
        Z_transform = Transform(grid, self.__Z_basis, derivs=3)
        I_transform = Transform(grid, self.__I_basis, derivs=1)
        P_transform = Transform(grid, self.__P_basis, derivs=1)
        coord_der = compute_coordinate_derivatives(self.__cR, self.__cZ,
                                                   R_transform, Z_transform)
        cov_basis = compute_covariant_basis(coord_der, axis=grid.axis)
        jacobian = compute_jacobian(coord_der, cov_basis, axis=grid.axis)
        con_basis = compute_contravariant_basis(coord_der, cov_basis, jacobian,
                                                axis=grid.axis)
        magnetic_field = compute_magnetic_field(cov_basis, jacobian, self.__cI,
                                                self.__Psi_lcfs, I_transform)
        plasma_current = compute_plasma_current(coord_der, cov_basis, jacobian,
                                        magnetic_field, self.__cI, I_transform)
        force_mag = compute_force_magnitude(coord_der, cov_basis, con_basis,
            jacobian, magnetic_field, plasma_current, self.__cP, P_transform)
        return force_mag

    #def save(self, save_to, file_format:str='hdf5', file_mode:str='w'):
        """Saves the configuration to file.

        Parameters
        __________
        save_to : str or file instance
            Object to save to. May be a string file path or file instance.
        file_format : str
            Format of file referenced by save_to. (Default = 'hdf5')
        file_mode : str
            File mode for file referenced by save_to. Only applicable if
            save_to is a string file path. (Default = 'w')

        Returns
        _______
        None

        """
    #    writer = eq_io.writer_factory(save_to, file_format=file_format,
    #            file_mode=file_mode)
    #    writer.write_obj(self)
    #    writer.close()
    #    return None


class Equilibrium(Configuration,IOAble):
    """Equilibrium is a decorator design pattern on top of Configuration.
       It adds information about how the equilibrium configuration was solved.
    """
    _save_attrs_ = Configuration._save_attrs_ + ['initial', 'objective', 'optimizer', 'solved']
    _object_lib_ = Configuration._object_lib_
    _object_lib_.update({'Configuration' : Configuration})

    def __init__(self, inputs:dict=None, load_from=None, file_format:str='hdf5', obj_lib=None) -> None:
        super().__init__(inputs=inputs, load_from=load_from, file_format=file_format, obj_lib=obj_lib)


    def _init_from_inputs_(self, inputs:dict=None) -> None:
        if inputs is None:
            inputs = self.inputs

        super()._init_from_inputs_(inputs=inputs)
        self.__initial = Configuration(inputs=inputs)
        self.__objective = inputs.get('objective', None)
        self.__optimizer = inputs.get('optimizer', None)
        self.__solved = False

    #def _init_from_file_(self, load_from=None, file_format:str=None) -> None:
    #    if load_from is None:
    #        load_from = self.load_from
    #    if file_format is None:
    #        file_format = self._file_format_
    #    reader = eq_io.reader_factory(load_from, file_format)
    #    self.initial = Configuration(load_from=reader.sub('initial'), file_format=file_format)
   #     self._save_attrs_ = self.initial._save_attrs_ + self.__addl_save_attrs__
  #      reader.read_obj(self)

    @property
    def solved(self) -> bool:
        return self.__solved

    @solved.setter
    def solved(self, issolved):
        self.__solved = issolved

    @property
    def initial(self) -> Configuration:
        return self.__initial

    @initial.setter
    def initial(self, conf:Configuration) -> None:
        self.__initial = conf

    @property
    def x(self):
        return self._Configuration__x

    @x.setter
    def x(self, x) -> None:
        self._Configuration__x = x
        self._Configuration__cR, self._Configuration__cZ, self.__cL = \
            unpack_state(self._Configuration__x,
                         self._Configuration__R_basis.num_modes,
                         self._Configuration__Z_basis.num_modes)
        self.__solved = True

    def optimize(self):
        pass

    @property
    def objective(self):
        return self.__objective

    @objective.setter
    def objective(self, objective):
        self.__objective = objective
        self.solved = False

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self.solved = False

    #def save(self, save_to, file_format='hdf5', file_mode='w'):
    #    writer = eq_io.writer_factory(save_to, file_format=file_format,
    #            file_mode=file_mode)
    #    writer.write_obj(self)
    #    writer.write_obj(self.initial, where=writer.sub('initial'))
    #    writer.close()
    #    return None


# XXX: Should this (also) inherit from Equilibrium?
class EquilibriaFamily(MutableSequence,IOAble):
    """EquilibriaFamily stores a list of Equilibria
    """
    _save_attrs_ = ['inputs', 'equilibria']
    _object_lib_ = Equilibrium._object_lib_
    _object_lib_.update({'Equilibrium' : Equilibrium})

    # FIXME: This should not have the same signiture as Configuration if it does not inherit from it
    def __init__(self, inputs=None, load_from=None, file_format='hdf5') -> None:
        self.__equilibria = []
        self.inputs = inputs
        self.load_from = load_from
        self._file_format_ = file_format
        self._file_mode_ = 'a'
        if inputs is not None:
            self._init_from_inputs_()
        elif load_from is not None:
            if file_format is None:
                raise RuntimeError('file_format argument must be included when loading from file.')
            self._init_from_file_(load_from, file_format=file_format)
        else:
            # hack
            pass #raise RuntimeError('inputs or load_from must be specified.')

    def _init_from_inputs_(self, inputs=None):
        if inputs is None:
            inputs = self.inputs
        writer = writer_factory(self.inputs['output_path'],
                file_format=self._file_format_, file_mode='w')
        writer.close()
        #self.append(Equilibrium(inputs=self.inputs))
        return None

    #def _init_from_file_(self, load_from=None, file_format=None):
    #    if load_from is None:
    #        load_from = self.load_from
    #    if file_format is None:
    #        file_format = self._file_format_
    #    reader = reader_factory(self.load_from, file_format=file_format)
    #    idx = 0
    #    while str(idx) in reader.groups():
    #        self.append(Equilibrium(load_from=reader.sub(str(idx))))
    #        idx += 1
    #    return None

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.__equilibria[i]

    def __setitem__(self, i, new_item):
        # add type checking
        self.__equilibria[i] = new_item

    def __delitem__(self, i):
        del self.__equilibria[i]

    def insert(self, i, new_item):
        self.__equilibria.insert(i, new_item)

    def __len__(self):
        return len(self.__equilibria)

    @property
    def solver(self):
        return self.__solver

    @solver.setter
    def solver(self, solver):
        self.__solver = solver

    @property
    def equilibria(self):
        return self.__equilibria

    @equilibria.setter
    def equilibria(self, eq):
        self.__equilibria = eq

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

    def save(self, save_to=None, file_format=None) -> None:
        #theslice = self.__slice__(idx)
        if save_to is None:
            save_to = self.inputs['output_path']
        if file_format is None:
            file_format = self._file_format_

        super().save(save_to, file_format=file_format)
        #writer = writer_factory(self.inputs['output_path'],
        #        file_format=file_format, file_mode=self._file_mode_)
        #writer.write_dict(self.inputs, where=writer.sub('inputs'))
        #for i in range(len(self[theslice])):
        #    print('saving index {}'.format(i))
        #    self[i].save(writer.sub(str(idx)), file_format=file_format,
        #        file_mode=self._file_mode_)
        #writer.close()

# TODO: overwrite all Equilibrium methods and default to self.__equilibria[-1]

def compute_coordinates(cR, cZ, R_transform, Z_transform):
    """Converts from spectral to real space

    Parameters
    ----------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    R_transform : Transform
        transforms R coefficients to real space
    Z_transform : Transform
        transforms Z coefficients to real space

    Returns
    -------
    coords : dict
        dictionary of ndarray, shape(N_nodes,) of coordinates evaluated at node locations
        keys are of the form 'X_y' meaning the derivative of X wrt to y

    """
    coords = {}
    coords['R'] = R_transform.transform(cR)
    coords['Z'] = Z_transform.transform(cZ)
    coords['phi'] = R_transform.grid.nodes[:, 2]    # phi = zeta
    coords['X'] = coords['R']*np.cos(coords['phi'])
    coords['Y'] = coords['R']*np.sin(coords['phi'])

# TODO: eliminate unnecessary derivatives for speedup (eg. R_rrr)
def compute_coordinate_derivatives(cR, cZ, R_transform, Z_transform, zeta_ratio=1.0):
    """Converts from spectral to real space and evaluates derivatives of R,Z wrt to SFL coords

    Parameters
    ----------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    R_transform : Transform
        transforms R coefficients to real space
    Z_transform : Transform
        transforms Z coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero
        effectively solves for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    coord_der : dict
        dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        keys are of the form 'X_y' meaning the derivative of X wrt to y

    """
    # notation: X_y means derivative of X wrt y
    coord_der = {}
    coord_der['R'] = R_transform.transform(cR, 0, 0, 0)
    coord_der['Z'] = Z_transform.transform(cZ, 0, 0, 0)
    coord_der['0'] = jnp.zeros_like(coord_der['R'])

    coord_der['R_r'] = R_transform.transform(cR, 1, 0, 0)
    coord_der['Z_r'] = Z_transform.transform(cZ, 1, 0, 0)
    coord_der['R_v'] = R_transform.transform(cR, 0, 1, 0)
    coord_der['Z_v'] = Z_transform.transform(cZ, 0, 1, 0)
    coord_der['R_z'] = R_transform.transform(cR, 0, 0, 1) * zeta_ratio
    coord_der['Z_z'] = Z_transform.transform(cZ, 0, 0, 1) * zeta_ratio

    coord_der['R_rr'] = R_transform.transform(cR, 2, 0, 0)
    coord_der['Z_rr'] = Z_transform.transform(cZ, 2, 0, 0)
    coord_der['R_rv'] = R_transform.transform(cR, 1, 1, 0)
    coord_der['Z_rv'] = Z_transform.transform(cZ, 1, 1, 0)
    coord_der['R_rz'] = R_transform.transform(cR, 1, 0, 1) * zeta_ratio
    coord_der['Z_rz'] = Z_transform.transform(cZ, 1, 0, 1) * zeta_ratio
    coord_der['R_vv'] = R_transform.transform(cR, 0, 2, 0)
    coord_der['Z_vv'] = Z_transform.transform(cZ, 0, 2, 0)
    coord_der['R_vz'] = R_transform.transform(cR, 0, 1, 1) * zeta_ratio
    coord_der['Z_vz'] = Z_transform.transform(cZ, 0, 1, 1) * zeta_ratio
    coord_der['R_zz'] = R_transform.transform(cR, 0, 0, 2) * zeta_ratio
    coord_der['Z_zz'] = Z_transform.transform(cZ, 0, 0, 2) * zeta_ratio

    # axis or QS terms
    if R_transform.grid.axis.size > 0 or R_transform.derivs == 'qs':
        coord_der['R_rrr'] = R_transform.transform(cR, 3, 0, 0)
        coord_der['Z_rrr'] = Z_transform.transform(cZ, 3, 0, 0)
        coord_der['R_rrv'] = R_transform.transform(cR, 2, 1, 0)
        coord_der['Z_rrv'] = Z_transform.transform(cZ, 2, 1, 0)
        coord_der['R_rrz'] = R_transform.transform(cR, 2, 0, 1) * zeta_ratio
        coord_der['Z_rrz'] = Z_transform.transform(cZ, 2, 0, 1) * zeta_ratio
        coord_der['R_rvv'] = R_transform.transform(cR, 1, 2, 0)
        coord_der['Z_rvv'] = Z_transform.transform(cZ, 1, 2, 0)
        coord_der['R_rvz'] = R_transform.transform(cR, 1, 1, 1) * zeta_ratio
        coord_der['Z_rvz'] = Z_transform.transform(cZ, 1, 1, 1) * zeta_ratio
        coord_der['R_rzz'] = R_transform.transform(cR, 1, 0, 2) * zeta_ratio
        coord_der['Z_rzz'] = Z_transform.transform(cZ, 1, 0, 2) * zeta_ratio
        coord_der['R_vvv'] = R_transform.transform(cR, 0, 3, 0)
        coord_der['Z_vvv'] = Z_transform.transform(cZ, 0, 3, 0)
        coord_der['R_vvz'] = R_transform.transform(cR, 0, 2, 1) * zeta_ratio
        coord_der['Z_vvz'] = Z_transform.transform(cZ, 0, 2, 1) * zeta_ratio
        coord_der['R_vzz'] = R_transform.transform(cR, 0, 1, 2) * zeta_ratio
        coord_der['Z_vzz'] = Z_transform.transform(cZ, 0, 1, 2) * zeta_ratio
        coord_der['R_zzz'] = R_transform.transform(cR, 0, 0, 3) * zeta_ratio
        coord_der['Z_zzz'] = Z_transform.transform(cZ, 0, 0, 3) * zeta_ratio

        coord_der['R_rrvv'] = R_transform.transform(cR, 2, 2, 0)
        coord_der['Z_rrvv'] = Z_transform.transform(cZ, 2, 2, 0)

    return coord_der


def compute_covariant_basis(coord_der, axis=jnp.array([]), derivs='force'):
    """Computes covariant basis vectors at grid points

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing the coordinate
        derivatives at each node, such as computed by ``compute_coordinate_derivatives``
    axis : ndarray, optional
        indicies of axis nodes
    derivs : str
        type of calculation being performed
        ``'force'``: all of the derivatives needed to calculate an
        equilibrium from the force balance equations
        ``'qs'``: all of the derivatives needed to calculate quasi-
        symmetry from the triple-product equation

    Returns
    -------
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node. Keys are of the form 'e_x_y',
        meaning the unit vector in the x direction, differentiated wrt to y.

    """
    # notation: subscript word is direction of unit vector, subscript letters denote partial derivatives
    # eg, e_rho_v is the v derivative of the covariant basis vector in the rho direction
    cov_basis = {}
    cov_basis['e_rho'] = jnp.array(
        [coord_der['R_r'],  coord_der['0'],   coord_der['Z_r']])
    cov_basis['e_theta'] = jnp.array(
        [coord_der['R_v'],  coord_der['0'],   coord_der['Z_v']])
    cov_basis['e_zeta'] = jnp.array(
        [coord_der['R_z'],  coord_der['R'],   coord_der['Z_z']])

    cov_basis['e_rho_r'] = jnp.array(
        [coord_der['R_rr'], coord_der['0'],   coord_der['Z_rr']])
    cov_basis['e_rho_v'] = jnp.array(
        [coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_rho_z'] = jnp.array(
        [coord_der['R_rz'], coord_der['0'],   coord_der['Z_rz']])

    cov_basis['e_theta_r'] = jnp.array(
        [coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_theta_v'] = jnp.array(
        [coord_der['R_vv'], coord_der['0'],   coord_der['Z_vv']])
    cov_basis['e_theta_z'] = jnp.array(
        [coord_der['R_vz'], coord_der['0'],   coord_der['Z_vz']])

    cov_basis['e_zeta_r'] = jnp.array(
        [coord_der['R_rz'], coord_der['R_r'], coord_der['Z_rz']])
    cov_basis['e_zeta_v'] = jnp.array(
        [coord_der['R_vz'], coord_der['R_v'], coord_der['Z_vz']])
    cov_basis['e_zeta_z'] = jnp.array(
        [coord_der['R_zz'], coord_der['R_z'], coord_der['Z_zz']])

    # axis or QS terms
    if len(axis) or derivs == 'qs':
        cov_basis['e_rho_rr'] = jnp.array(
            [coord_der['R_rrr'], coord_der['0'],   coord_der['Z_rrr']])
        cov_basis['e_rho_rv'] = jnp.array(
            [coord_der['R_rrv'], coord_der['0'],   coord_der['Z_rrv']])
        cov_basis['e_rho_rz'] = jnp.array(
            [coord_der['R_rrz'], coord_der['0'],   coord_der['Z_rrz']])
        cov_basis['e_rho_vv'] = jnp.array(
            [coord_der['R_rvv'], coord_der['0'],   coord_der['Z_rvv']])
        cov_basis['e_rho_vz'] = jnp.array(
            [coord_der['R_rvz'], coord_der['0'],   coord_der['Z_rvz']])
        cov_basis['e_rho_zz'] = jnp.array(
            [coord_der['R_rzz'], coord_der['0'],   coord_der['Z_rzz']])

        cov_basis['e_theta_rr'] = jnp.array(
            [coord_der['R_rrv'], coord_der['0'],   coord_der['Z_rrv']])
        cov_basis['e_theta_rv'] = jnp.array(
            [coord_der['R_rvv'], coord_der['0'],   coord_der['Z_rvv']])
        cov_basis['e_theta_rz'] = jnp.array(
            [coord_der['R_rvz'], coord_der['0'],   coord_der['Z_rvz']])
        cov_basis['e_theta_vv'] = jnp.array(
            [coord_der['R_vvv'], coord_der['0'],   coord_der['Z_vvv']])
        cov_basis['e_theta_vz'] = jnp.array(
            [coord_der['R_vvz'], coord_der['0'],   coord_der['Z_vvz']])
        cov_basis['e_theta_zz'] = jnp.array(
            [coord_der['R_vzz'], coord_der['0'],   coord_der['Z_vzz']])

        cov_basis['e_zeta_rr'] = jnp.array(
            [coord_der['R_rrz'], coord_der['R_rr'], coord_der['Z_rrz']])
        cov_basis['e_zeta_rv'] = jnp.array(
            [coord_der['R_rvz'], coord_der['R_rv'], coord_der['Z_rvz']])
        cov_basis['e_zeta_rz'] = jnp.array(
            [coord_der['R_rzz'], coord_der['R_rz'], coord_der['Z_rzz']])
        cov_basis['e_zeta_vv'] = jnp.array(
            [coord_der['R_vvz'], coord_der['R_vv'], coord_der['Z_vvz']])
        cov_basis['e_zeta_vz'] = jnp.array(
            [coord_der['R_vzz'], coord_der['R_vz'], coord_der['Z_vzz']])
        cov_basis['e_zeta_zz'] = jnp.array(
            [coord_der['R_zzz'], coord_der['R_zz'], coord_der['Z_zzz']])

    return cov_basis


def compute_contravariant_basis(coord_der, cov_basis, jacobian, axis=jnp.array([])):
    """Computes contravariant basis vectors and jacobian elements

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing coordinate derivatives
        evaluated at node locations, such as computed by ``compute_coordinate_derivatives``
    cov_basis : dict
        dictionary of ndarray containing covariant basis vectors
        and derivatives at each node, such as computed by ``compute_covariant_basis``
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian
        and partial derivatives, such as computed by ``compute_jacobian``
        axis : ndarray, optional
        indicies of axis nodes
    axis : ndarray, optional
        indicies of axis nodes

    Returns
    -------
    con_basis : dict
        dictionary of ndarray containing contravariant basis vectors and jacobian elements

    """
    # subscripts (superscripts) denote covariant (contravariant) basis vectors
    con_basis = {}

    # contravariant basis vectors
    con_basis['e^rho'] = cross(
        cov_basis['e_theta'], cov_basis['e_zeta'], 0)/jacobian['g']
    con_basis['e^theta'] = cross(
        cov_basis['e_zeta'], cov_basis['e_rho'], 0)/jacobian['g']
    con_basis['e^zeta'] = jnp.array(
        [coord_der['0'], 1/coord_der['R'], coord_der['0']])

    # axis terms
    if len(axis):
        con_basis['e^rho'] = put(con_basis['e^rho'], opsindex[:, axis], (cross(
            cov_basis['e_theta_r'][:, axis], cov_basis['e_zeta'][:, axis], 0)/jacobian['g_r'][axis]))
        # e^theta = infinite at the axis

    # metric coefficients
    con_basis['g^rr'] = dot(con_basis['e^rho'],   con_basis['e^rho'],   0)
    con_basis['g^rv'] = dot(con_basis['e^rho'],   con_basis['e^theta'], 0)
    con_basis['g^rz'] = dot(con_basis['e^rho'],   con_basis['e^zeta'],  0)
    con_basis['g^vv'] = dot(con_basis['e^theta'], con_basis['e^theta'], 0)
    con_basis['g^vz'] = dot(con_basis['e^theta'], con_basis['e^zeta'],  0)
    con_basis['g^zz'] = dot(con_basis['e^zeta'],  con_basis['e^zeta'],  0)

    return con_basis


def compute_jacobian(coord_der, cov_basis, axis=jnp.array([]), derivs='force'):
    """Computes coordinate jacobian and derivatives

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing of coordinate
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    axis : ndarray, optional
        indicies of axis nodes
    derivs : str
        type of calculation being performed
        ``'force'``: all of the derivatives needed to calculate an
        equilibrium from the force balance equations
        ``'qs'``: all of the derivatives needed to calculate quasi-
        symmetry from the triple-product equation

    Returns
    -------
    jacobian : dict
        dictionary of ndarray, shape(N_nodes,) of coordinate
        jacobian and partial derivatives. Keys are of the form `g_x` meaning
        the x derivative of the coordinate jacobian g

    """
    # notation: subscripts denote partial derivatives
    jacobian = {}
    jacobian['g'] = coord_der['R']*(coord_der['R_v']*coord_der['Z_r'] \
                                  - coord_der['R_r']*coord_der['Z_v'])
    jacobian['g_r'] = coord_der['R']*(coord_der['R_rv']*coord_der['Z_r']
                                    + coord_der['R_v']*coord_der['Z_rr']
                                    - coord_der['R_rr']*coord_der['Z_v']
                                    - coord_der['R_r']*coord_der['Z_rv']) \
                    + coord_der['R_r']*(coord_der['R_v']*coord_der['Z_r']
                                      - coord_der['R_r']*coord_der['Z_v'])
    jacobian['g_v'] = coord_der['R']*(coord_der['R_vv']*coord_der['Z_r']
                                    + coord_der['R_v']*coord_der['Z_rv']
                                    - coord_der['R_rv']*coord_der['Z_v']
                                    - coord_der['R_r']*coord_der['Z_vv']) \
                    + coord_der['R_v']*(coord_der['R_v']*coord_der['Z_r']
                                      - coord_der['R_r']*coord_der['Z_v'])
    jacobian['g_z'] = coord_der['R']*(coord_der['R_vz']*coord_der['Z_r']
                                    + coord_der['R_v']*coord_der['Z_rz']
                                    - coord_der['R_rz']*coord_der['Z_v']
                                    - coord_der['R_r']*coord_der['Z_vz']) \
                    + coord_der['R_z']*(coord_der['R_v']*coord_der['Z_r']
                                      - coord_der['R_r']*coord_der['Z_v'])

    """
    jacobian['g'] = dot(cov_basis['e_rho'],
                    cross(cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0)
    jacobian['g_r'] = dot(cov_basis['e_rho_r'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta_r'], 0), 0)
    jacobian['g_v'] = dot(cov_basis['e_rho_v'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0), 0)
    jacobian['g_z'] = dot(cov_basis['e_rho_z'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0), 0)
    """

    # axis or QS terms
    if len(axis) or derivs == 'qs':
        jacobian['g_rr'] = dot(cov_basis['e_rho_rr'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0)*2 \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rr'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_r'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_rr'], 0), 0)
        jacobian['g_rv'] = dot(cov_basis['e_rho_rv'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rv'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_rv'], 0), 0)
        jacobian['g_rz'] = dot(cov_basis['e_rho_rz'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rz'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_rz'], 0), 0)

        jacobian['g_vv'] = dot(cov_basis['e_rho_vv'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0)*2 \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_vv'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_v'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_vv'], 0), 0)
        jacobian['g_vz'] = dot(cov_basis['e_rho_vz'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_vz'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_vz'], 0), 0)
        jacobian['g_zz'] = dot(cov_basis['e_rho_zz'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0)*2 \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_zz'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_z'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_zz'], 0), 0)

    return jacobian


def compute_magnetic_field(cov_basis, jacobian, cI, Psi_lcfs, I_transform, derivs='force'):
    """Computes magnetic field components at node locations

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian
        and partial derivatives, such as computed by ``compute_jacobian``.
    cI : ndarray
        coefficients to pass to rotational transform function
    Psi_lcfs : float
        total toroidal flux (in Webers) within LCFS
    I_transform : Transform
        object with transform method to go from spectral to physical space with derivatives
    derivs : str
        type of calculation being performed
        ``'force'``: all of the derivatives needed to calculate an
        equilibrium from the force balance equations
        ``'qs'``: all of the derivatives needed to calculate quasi-
        symmetry from the triple-product equation

    Returns
    -------
    magnetic_field: dict
        dictionary of ndarray, shape(N_nodes,) of magnetic field
        and derivatives. Keys are of the form 'B_x_y' or 'B^x_y', meaning the
        covariant (B_x) or contravariant (B^x) component of the magnetic field, with the derivative wrt to y.

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    magnetic_field = {}
    r = I_transform.grid.nodes[:, 0]
    axis = I_transform.grid.axis
    iota = I_transform.transform(cI, 0)
    iota_r = I_transform.transform(cI, 1)

    # toroidal flux
    magnetic_field['psi'] = Psi_lcfs*r**2
    magnetic_field['psi_r'] = 2*Psi_lcfs*r
    magnetic_field['psi_rr'] = 2*Psi_lcfs*jnp.ones_like(r)

    # contravariant B components
    magnetic_field['B^rho'] = jnp.zeros_like(r)
    magnetic_field['B^zeta'] = magnetic_field['psi_r'] / \
        (2*jnp.pi*jacobian['g'])
    if len(axis):
        magnetic_field['B^zeta'] = put(
            magnetic_field['B^zeta'], axis, magnetic_field['psi_rr'][axis] / (2*jnp.pi*jacobian['g_r'][axis]))
    magnetic_field['B^theta'] = iota * magnetic_field['B^zeta']
    magnetic_field['B_con'] = magnetic_field['B^rho']*cov_basis['e_rho'] + magnetic_field['B^theta'] * \
        cov_basis['e_theta'] + magnetic_field['B^zeta']*cov_basis['e_zeta']

    # covariant B components
    magnetic_field['B_rho'] = magnetic_field['B^zeta'] * \
        dot(iota*cov_basis['e_theta'] +
            cov_basis['e_zeta'], cov_basis['e_rho'], 0)
    magnetic_field['B_theta'] = magnetic_field['B^zeta'] * \
        dot(iota*cov_basis['e_theta'] +
            cov_basis['e_zeta'], cov_basis['e_theta'], 0)
    magnetic_field['B_zeta'] = magnetic_field['B^zeta'] * \
        dot(iota*cov_basis['e_theta'] +
            cov_basis['e_zeta'], cov_basis['e_zeta'], 0)

    # B^{zeta} derivatives
    magnetic_field['B^zeta_r'] = magnetic_field['psi_rr'] / (2*jnp.pi*jacobian['g']) - \
        (magnetic_field['psi_r']*jacobian['g_r']) / (2*jnp.pi*jacobian['g']**2)
    magnetic_field['B^zeta_v'] = - \
        (magnetic_field['psi_r']*jacobian['g_v']) / (2*jnp.pi*jacobian['g']**2)
    magnetic_field['B^zeta_z'] = - \
        (magnetic_field['psi_r']*jacobian['g_z']) / (2*jnp.pi*jacobian['g']**2)

    # axis terms
    if len(axis):
        magnetic_field['B^zeta_r'] = put(magnetic_field['B^zeta_r'], axis, -(magnetic_field['psi_rr']
                                                                            [axis]*jacobian['g_rr'][axis]) / (4*jnp.pi*jacobian['g_r'][axis]**2))
        magnetic_field['B^zeta_v'] = put(magnetic_field['B^zeta_v'], axis, 0)
        magnetic_field['B^zeta_z'] = put(magnetic_field['B^zeta_z'], axis, -(magnetic_field['psi_rr']
                                                                            [axis]*jacobian['g_rz'][axis]) / (2*jnp.pi*jacobian['g_r'][axis]**2))

    # QS terms
    if derivs == 'qs':
        magnetic_field['B^zeta_vv'] = - (magnetic_field['psi_r']*jacobian['g_vv']) / (2*jnp.pi*jacobian['g']**2) \
            + (magnetic_field['psi_r']*jacobian['g_v']
               ** 2) / (jnp.pi*jacobian['g']**3)
        magnetic_field['B^zeta_vz'] = - (magnetic_field['psi_r']*jacobian['g_vz']) / (2*jnp.pi*jacobian['g']**2) \
            + (magnetic_field['psi_r']*jacobian['g_v']*jacobian['g_z']) / \
            (jnp.pi*jacobian['g']**3)
        magnetic_field['B^zeta_zz'] = - (magnetic_field['psi_r']*jacobian['g_zz']) / (2*jnp.pi*jacobian['g']**2) \
            + (magnetic_field['psi_r']*jacobian['g_z']
               ** 2) / (jnp.pi*jacobian['g']**3)

    # covariant B component derivatives
    magnetic_field['B_theta_r'] = magnetic_field['B^zeta_r']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_theta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota_r*cov_basis['e_theta']+iota*cov_basis['e_rho_v']+cov_basis['e_zeta_r'], cov_basis['e_theta'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_v'], 0))
    magnetic_field['B_zeta_r'] = magnetic_field['B^zeta_r']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota_r*cov_basis['e_theta']+iota*cov_basis['e_rho_v']+cov_basis['e_zeta_r'], cov_basis['e_zeta'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta_r'], 0))
    magnetic_field['B_rho_v'] = magnetic_field['B^zeta_v']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_v']+cov_basis['e_zeta_v'], cov_basis['e_rho'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_v'], 0))
    magnetic_field['B_zeta_v'] = magnetic_field['B^zeta_v']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_v']+cov_basis['e_zeta_v'], cov_basis['e_zeta'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))
    magnetic_field['B_rho_z'] = magnetic_field['B^zeta_z']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_z']+cov_basis['e_zeta_z'], cov_basis['e_rho'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_z'], 0))
    magnetic_field['B_theta_z'] = magnetic_field['B^zeta_z']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_theta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_z']+cov_basis['e_zeta_z'], cov_basis['e_theta'], 0)
                                    + dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_theta_z'], 0))

    return magnetic_field


def compute_plasma_current(coord_der, cov_basis, jacobian, magnetic_field, cI, I_transform):
    """Computes current density field at node locations

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian
        and partial derivatives, such as computed by ``compute_jacobian``.
    coord_der : dict
        dictionary of ndarray containing of coordinate
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives,
        such as computed by ``compute_magnetic_field``.
    cI : ndarray
        coefficients to pass to rotational transform function.
    I_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    plasma_current : dict
        dictionary of ndarray, shape(N_nodes,) of current field.
        Keys are of the form 'J^x_y' meaning the contravariant (J^x)
        component of the current, with the derivative wrt to y.

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    plasma_current = {}
    mu0 = 4*jnp.pi*1e-7
    axis = I_transform.grid.axis
    iota = I_transform.transform(cI, 0)

    # axis terms
    if len(axis):
        g_rrv = 2*coord_der['R_rv']*(coord_der['Z_r']*coord_der['R_rv'] - coord_der['R_r']*coord_der['Z_rv']) \
            + 2*coord_der['R_r']*(coord_der['Z_r']*coord_der['R_rvv'] - coord_der['R_r']*coord_der['Z_rvv']) \
            + coord_der['R']*(2*coord_der['Z_rr']*coord_der['R_rvv'] - 2*coord_der['R_rr']*coord_der['Z_rvv']
                              + coord_der['R_rv']*coord_der['Z_rrv'] -
                              coord_der['Z_rv']*coord_der['R_rrv']
                              + coord_der['Z_r']*coord_der['R_rrvv'] - coord_der['R_r']*coord_der['Z_rrvv'])
        Bsup_zeta_rv = magnetic_field['psi_rr']*(2*jacobian['g_rr']*jacobian['g_rv'] -
                                                 jacobian['g_r']*g_rrv) / (4*jnp.pi*jacobian['g_r']**3)
        Bsub_zeta_rv = Bsup_zeta_rv*dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0) + magnetic_field['B^zeta']*dot(
            iota*cov_basis['e_rho_vv'] + 2*cov_basis['e_zeta_rv'], cov_basis['e_zeta'], 0)
        Bsub_theta_rz = magnetic_field['B^zeta_z']*dot(cov_basis['e_zeta'], cov_basis['e_rho_v'], 0) + magnetic_field['B^zeta']*(
            dot(cov_basis['e_zeta_z'], cov_basis['e_rho_v'], 0) + dot(cov_basis['e_zeta'], cov_basis['e_rho_vz'], 0))

    # contravariant J components
    plasma_current['J^rho'] = (magnetic_field['B_zeta_v'] -
                               magnetic_field['B_theta_z']) / (mu0*jacobian['g'])
    plasma_current['J^theta'] = (magnetic_field['B_rho_z'] -
                                 magnetic_field['B_zeta_r']) / (mu0*jacobian['g'])
    plasma_current['J^zeta'] = (magnetic_field['B_theta_r'] -
                                magnetic_field['B_rho_v']) / (mu0*jacobian['g'])

    # axis terms
    if len(axis):
        plasma_current['J^rho'] = put(plasma_current['J^rho'], axis,
                                      (Bsub_zeta_rv[axis] - Bsub_theta_rz[axis]) / (jacobian['g_r'][axis]))

    plasma_current['J_con'] = plasma_current['J^rho']*cov_basis['e_rho'] + plasma_current['J^theta'] * \
        cov_basis['e_theta'] + plasma_current['J^zeta']*cov_basis['e_zeta']

    return plasma_current


def compute_magnetic_field_magnitude(cov_basis, magnetic_field, cI, I_transform):
    """Computes magnetic field magnitude at node locations

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives,
        such as computed by ``compute_magnetic_field``.
    cI : ndarray
        coefficients to pass to rotational transform function
    I_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    magnetic_field_mag : dict
        dictionary of ndarray, shape(N_nodes,) of magnetic field magnitude and derivatives

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    magnetic_field_mag = {}
    iota = I_transform.transform(cI, 0)

    magnetic_field_mag['|B|'] = jnp.abs(magnetic_field['B^zeta'])*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0) +
                                                              2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0) + dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))

    magnetic_field_mag['|B|_v'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    magnetic_field_mag['|B|_z'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    magnetic_field_mag['|B|_vv'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vv']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_v'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vv'], 0))+2*iota*(dot(cov_basis['e_theta_vv'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vv'], 0)+2*dot(cov_basis['e_theta_v'], cov_basis['e_zeta_v'], 0))+2*(dot(cov_basis['e_zeta_v'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vv'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))**2 \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    magnetic_field_mag['|B|_zz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_zz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_z'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_zz'], 0))+2*iota*(dot(cov_basis['e_theta_zz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_zz'], 0)+2*dot(cov_basis['e_theta_z'], cov_basis['e_zeta_z'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0))**2 \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    magnetic_field_mag['|B|_vz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vz'], 0))+2*iota*(dot(cov_basis['e_theta_vz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta_v'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_theta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vz'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    return magnetic_field_mag


def compute_force_magnitude(coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, P_transform):
    """Computes force error magnitude at node locations

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing of coordinate
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    con_basis : dict
        dictionary of ndarray containing contravariant basis
        vectors and metric elements at each node, such as computed by ``compute_contravariant_basis``.
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian
        and partial derivatives, such as computed by ``compute_jacobian``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives,
        such as computed by ``compute_magnetic_field``.
    plasma_current : dict
        dictionary of ndarray containing current and derivatives,
        such as computed by ``compute_plasma_current``.
    cP : ndarray
        parameters to pass to pressure function
    Psi_lcfs : float
        total toroidal flux (in Webers) within LCFS
    P_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    force_mag : dict
        dictionary of ndarray, shape(N_nodes,) of force magnitudes

    """
    force_mag = {}
    mu0 = 4*jnp.pi*1e-7
    axis = P_transform.grid.axis
    pres_r = P_transform.transform(cP, 1)

    # force balance error covariant components
    F_rho = jacobian['g']*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                           plasma_current['J^zeta']*magnetic_field['B^theta']) - pres_r
    F_theta = jacobian['g']*plasma_current['J^rho']*magnetic_field['B^zeta']
    F_zeta = -jacobian['g']*plasma_current['J^rho']*magnetic_field['B^theta']

    # axis terms
    if len(axis):
        Jsup_theta = (magnetic_field['B_rho_z'] -
                      magnetic_field['B_zeta_r']) / mu0
        Jsup_zeta = (magnetic_field['B_theta_r'] -
                     magnetic_field['B_rho_v']) / mu0
        F_rho = put(F_rho, axis, Jsup_theta[axis]*magnetic_field['B^zeta']
                    [axis] - Jsup_zeta[axis]*magnetic_field['B^theta'][axis])
        grad_theta = cross(cov_basis['e_zeta'], cov_basis['e_rho'], 0)
        gsup_vv = dot(grad_theta, grad_theta, 0)
        gsup_rv = dot(con_basis['e^rho'], grad_theta, 0)
        gsup_vz = dot(grad_theta, con_basis['e^zeta'], 0)
        F_theta = put(
            F_theta, axis, plasma_current['J^rho'][axis]*magnetic_field['B^zeta'][axis])
        F_zeta = put(F_zeta, axis, -plasma_current['J^rho']
                     [axis]*magnetic_field['B^theta'][axis])
        con_basis['g^vv'] = put(con_basis['g^vv'], axis, gsup_vv[axis])
        con_basis['g^rv'] = put(con_basis['g^rv'], axis, gsup_rv[axis])
        con_basis['g^vz'] = put(con_basis['g^vz'], axis, gsup_vz[axis])

    # F_i*F_j*g^ij terms
    Fg_rr = F_rho * F_rho * con_basis['g^rr']
    Fg_vv = F_theta*F_theta*con_basis['g^vv']
    Fg_zz = F_zeta * F_zeta * con_basis['g^zz']
    Fg_rv = F_rho * F_theta*con_basis['g^rv']
    Fg_rz = F_rho * F_zeta * con_basis['g^rz']
    Fg_vz = F_theta*F_zeta * con_basis['g^vz']

    # magnitudes
    force_mag['|F|'] = jnp.sqrt(Fg_rr + Fg_vv + Fg_zz + 2*Fg_rv + 2*Fg_rz + 2*Fg_vz)
    force_mag['|grad(p)|'] = jnp.sqrt(pres_r*pres_r*con_basis['g^rr'])

    return force_mag
