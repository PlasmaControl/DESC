import numpy as np
from collections.abc import MutableSequence

from desc.backend import jnp, put, opsindex, cross, dot, presfun, iotafun, TextColors, unpack_x
from desc.init_guess import get_initial_guess_scale_bdry
from desc.zernike import symmetric_x
from desc.boundary_conditions import format_bdry
from desc import equilibrium_io as eq_io


class Configuration():

    # TODO: replace zern_idx & lambda_idx with Transform objects
    def __init__(self, inputs=None, load_from=None, file_format='hdf5'): #bdry, cP, cI, Psi, NFP, zern_idx, lambda_idx, sym=False, x=None, axis=None) -> None:
        """Initializes a configuration

        Parameters
        ----------
        bdry : ndarray of float, shape(Nbdry,4)
            array of boundary Fourier coeffs [m,n,Rcoeff, Zcoeff]
            OR
            array of real space coordinates, [theta,phi,R,Z]
        cP : ndarray
            spectral coefficients of the pressure profile (Pascals)
        cI : ndarray
            spectral coefficients of the rotational transform profile
        Psi : float
            toroidal flux within the last closed flux surface (Webers)
        NFP : int
            number of toroidal field periods
        zern_idx : ndarray of int, shape(N_coeffs,3)
            indices for spectral basis, ie an array of [l,m,n] for each spectral coefficient
        lambda_idx : ndarray of int, shape(Nmodes,2)
            poloidal and toroidal mode numbers [m,n]
        sym : bool
            True for stellarator symmetry, False otherwise
        x : ndarray
            state vector of independent variables: [cR, cZ, cL]. If not supplied,
            the flux surfaces are scaled from the boundary and magnetic axis
        axis : ndarray, shape(Naxis,3)
            array of axis Fourier coeffs [n,Rcoeff, Zcoeff]
        """

        self._save_attrs_ = ['__bdry', '__cP', '__cI', '__Psi', '__NFP',
                '__zern_idx','__lambda_idx', '__sym', 'x', 'axis', '__cR',
                '__cZ', '__cL']

        self.inputs = inputs
        self.load_from = load_from
        if inputs is not None:
            self._init_from_inputs_()
        elif load_from is not None:
            if file_format is None:
                raise RuntimeError('file_format argument must be included when loading from file.')
            self._file_format_ = file_format
            self._init_from_file_()
        else:
            raise RuntimeError('inputs or load_from must be specified.')

    def _init_from_inputs_(self, inputs=None):
        if inputs is None:
            inputs = self.inputs
        self.__bdry = inputs['bdry']
        self.__cP = inputs['cP']
        self.__cI = inputs['cI']
        self.__Psi = inputs['Psi']
        self.__NFP = inputs['NFP']
        self.__zern_idx = inputs['zern_idx']
        self.__lambda_idx = inputs['lambda_idx']
        self.__sym = inputs['sym']

        self.__bdryM, self.__bdryN, self.__bdryR, self.__bdryZ = format_bdry(
            np.max(self.__lambda_idx[:,0]), np.max(self.__lambda_idx[:,1]), self.__NFP, self.__bdry, 'spectral', 'spectral')

            # embed this if-else into the symmetric_x function
        if sym:
            # TODO: move symmetric_x inside configuration.py
            sym_mat = symmetric_x(self.__zern_idx, self.__lambda_idx)
        else:
            sym_mat = np.eye(2*self.__zern_idx.shape[0] + self.__lambda_idx.shape[0])

        if inputs['x'] is None:
            # TODO: change input reader to make the default axis=None
            if inputs['axis'] is None:
                axis = bdry[np.where(inputs['bdry'][:, 0] == 0)[0], 1:]
            self.__cR, self.__cZ = get_initial_guess_scale_bdry(
                axis, inputs['bdry'], 1.0, inputs['zern_idx'], inputs['NFP'], mode='spectral', rcond=1e-6)
            self.__cL = np.zeros(len(inputs['lambda_idx']))
            self.__x = np.concatenate([self.__cR, self.__cZ, self.__cL])
            self.__x = np.matmul(sym_mat.T, inputs['x'])#self.__x)
        else:
            self.__x = inputs['x']
            try:
                # TODO: move unpack_x inside configuration.py
                self.__cR, self.__cZ, self.__cL = unpack_x(np.matmul(sym_mat, self.__x), len(inputs['zern_idx']))
            except:
                raise ValueError(TextColors.FAIL +
                    "State vector dimension is incompatible with other parameters" + TextColors.ENDC)
        return None

    def _init_from_file_(self, load_from=None, file_format=None):
        if load_from is None:
            load_from = self.load_from
        if file_format is None:
            file_format = self._file_format_
        reader = eq_io.reader_factory(load_from, file_format)
        reader.read_obj(self)
        return None

    @property
    def bdry(self):
        return self.__bdry

    # I have not included __bdry{M,N,R,Z} in _save_attrs_ since the setter sets them via __bdry.
    @bdry.setter
    def bdry(self, bdry):
        self.__bdry = bdry
        self.__bdryM, self.__bdryN, self.__bdryR, self.__bdryZ = format_bdry(
            np.max(self.__lambda_idx[:,0]), np.max(self.__lambda_idx[:,1]), self.__NFP, self.__bdry, 'spectral', 'spectral')

    @property
    def cP(self):
        return self.__cP

    @cP.setter
    def cP(self, cP):
        self.__cP = cP

    @property
    def cI(self):
        return self.__cI

    @cI.setter
    def cI(self, cI):
        self.__cI = cI

    @property
    def Psi(self):
        return self.__Psi

    @Psi.setter
    def Psi(self, Psi):
        self.__Psi = Psi

    @property
    def NFP(self):
        return self.__NFP

    @NFP.setter
    def NFP(self, NFP):
        self.__NFP = NFP

    @property
    def sym(self):
        return self.__sym

    @sym.setter
    def sym(self, sym):
        self.__sym = sym
        if sym:
            # TODO: move symmetric_x inside configuration.py
            sym_mat = symmetric_x(self.__zern_idx, self.__lambda_idx)
        else:
            sym_mat = np.eye(2*self.__zern_idx.shape[0] + self.__lambda_idx.shape[0])
        self.__x = np.matmul(sym_mat.T, self.__x)

    def attributes(self):
        return (self.x, self.bdryR, self.bdryZ, self.cP, self.cI, self.Psi)

    def compute_coordinate_derivatives(self):
        pass
        # return compute_coordinate_derivatives(self.cR, self.cZ, zernike_transform, zeta_ratio=1.0, mode='equil')

    def compute_covariant_basis(self):
        pass
        # return compute_covariant_basis(coord_der, zernike_transform, mode='equil')

    def compute_contravariant_basis(self):
        pass
        # return compute_contravariant_basis(coord_der, cov_basis, jacobian, zernike_transform)

    def compute_jacobian(self):
        pass
        # return compute_jacobian(coord_der, cov_basis, zernike_transform, mode='equil')

    def compute_magnetic_field(self):
        pass
        # return compute_magnetic_field(cov_basis, jacobian, cI, Psi_lcfs, zernike_transform, mode='equil')

    def compute_plasma_current(self):
        pass
        # return compute_plasma_current(coord_der, cov_basis, jacobian, magnetic_field, cI, Psi_lcfs, zernike_transform)

    def compute_magnetic_field_magnitude(self):
        pass
        # return def compute_magnetic_field_magnitude(cov_basis, magnetic_field, cI, zernike_transform)

    def compute_force_magnitude(self):
        pass
        # return def compute_force_magnitude(coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, cI, Psi_lcfs, zernike_transform):

    def save(self, save_to, file_format='hdf5', file_mode='w'):
        writer = eq_io.writer_factory(save_to, file_format=file_format,
                file_mode=file_mode)
        writer.write_obj(self)
        writer.close()
        return None


class Equilibrium(Configuration):

    def __init__(self, inputs=None, load_from=None, file_format='hdf5'):#bdry, cP, cI, Psi, NFP, zern_idx, lambda_idx, sym=False, x=None, axis=None, objective=None, optimizer=None) -> None:
        self.__addl_save_attrs__ = ['objective', 'optimizer', 'solved']
        self.inputs = inputs
        self.load_from = load_from
        if inputs is not None:
            self._init_from_inputs_()
        elif load_from is not None:
            if file_format is None:
                raise RuntimeError('file_format argument must be included when loading from file.')
            self._file_format_ = file_format
            self._init_from_file_()
        else:
            raise RuntimeError('inputs or load_from must be specified.')


        #self.initial = Configuration(self, bdry, cP, cI, Psi, NFP, zern_idx, lambda_idx, sym=False, x=None, axis=None)
        #self._save_attrs_ = self.initial._save_attrs_ + ['objective', 'optimizer', 'solved']
        #self.__objective = objective
        #self.__optimizer = optimizer
        #self.solved = False

    def _init_from_inputs_(self, inputs=None):
        if inputs is None:
            inputs = self.inputs
        self.initial = Configuration(inputs=inputs)
        self._save_attrs_ = self.initial._save_attrs_ + self.__addl_save_attrs__
        self.__objective = inputs['objective']
        self.__optimizer = inputs['optimizer']
        self.solved = False
        return None

    def _init_from_file_(self, load_from=None, file_format=None):
        if load_from is None:
            load_from = self.load_from
        if file_format is None:
            file_format = self._file_format_
        reader = eq_io.reader_factory(load_from, file_format)
        self.initial = Configuration(load_from=reader.sub('initial'), file_format=file_format)
        self._save_attrs_ = self.initial._save_attrs_ + self.__addl_save_attrs__
        reader.read_obj(self)
        return None

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

    def save(self, save_to, file_format='hdf5', file_mode='w'):
        writer = eq_io.writer_factory(save_to, file_format=file_format,
                file_mode=file_mode)
        writer.write_obj(self)
        writer.write_obj(self.initial, where=writer.sub('initial'))
        writer.close()
        return None

    # TODO: Does this inherit from Equilibrium? I don't think so because they have different constructors
class EquilibriaFamily(MutableSequence):

    def __init__(self, inputs=None, load_from=None, file_format='hdf5'):#equilibria, solver=None) -> None:
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
            self._file_format_ = file_format
            self._init_from_file_()
        else:
            raise RuntimeError('inputs or load_from must be specified.')

        #self.__equilibria = equilibria
        #self.__solver = solver
        #self.output_path = inputs['output_path'] #need some way to integrate this
        #check if file exists - option to overwrite

    def _init_from_inputs_(self, inputs=None):
        if inputs is None:
            inputs = self.inputs
        writer = eq_io.writer_factory(self.inputs['output_path'],
                file_format=self._file_format_, file_mode='w')
        writer.write_dict(self.inputs, where=writer.sub('inputs'))
        writer.close()
        self.append(Equilibrium(inputs=self.inputs))
        return None

    def _init_from_file_(self, load_from=None, file_format=None):
        if load_from is None:
            load_from = self.load_from
        if file_format is None:
            file_format = self._file_format_
        reader = eq_io.reader_factory(self.load_from, file_format=file_format)
        idx = 0
        while str(idx) in reader.groups():
            self.append(Equilibrium(load_from=reader.sub(str(idx))))
            idx += 1
        return None

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.__eqilibria[i]

    def __setitem__(self, i, new_item):
        # add type checking
        self.__equilibria[i] = new_item

    def __delitem__(self, i):
        del self.__equilibrium[i]

    def insert(self, i, new_item):
        self.__equilibrium.insert(i, new_item)

    def __len__(self):
        return len(self.__equilibria)

    #@property
    #def equilibria(self):
    #    return self.__equilibria

    #@equilibria.setter
    #def equilibria(self, equilibria):
    #    self.__equilibria = equilibria

    @property
    def solver(self):
        return self.__solver

    @solver.setter
    def solver(self, solver):
        self.__solver = solver

    def save(self, idx, save_to=None, file_format=None):
        if type(idx) is not int:
            # implement fancier indexing later
            raise NotImplementedError('idx must be a single integer index')

        if save_to is None:
            save_to = self.inputs['output_path']
        if file_format is None:
            file_format = self._file_format_

        writer = eq_io.writer_factory(self.inputs['output_path'],
                file_format=file_format, file_mode=self.file_mode)
        self[idx].save(writer.sub(str(idx)), file_format=file_format,
                file_mode=self._file_mode_)
        writer.close()
        return None

    #def load(self, filename):
    #    reader = eq_io.reader_factory(filename, file_format)
    #    self.inputs = {}
    #    reader.read_dict(self.inputs, where=reader.sub('inputs'))
    #    self.__equilibria = []
    #    neqilibria = reader.count()
    #    for i in range(nequilibria):
    #        self.append(reader.load_equilibrium(where=reader.sub(str(i))))
    #    return None
        #self.__equilibria = []
        #for
# TODO: overwrite all Equilibrium methods and default to self.__equilibria[-1]


def compute_coordinate_derivatives(cR, cZ, zernike_transform, zeta_ratio=1.0, mode='equil'):
    """Converts from spectral to real space and evaluates derivatives of R,Z wrt to SFL coords

    Parameters
    ----------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero
        effectively solves for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)
    mode : str
        one of 'equil' or 'qs'. Whether to compute field terms for equilibrium or quasisymmetry optimization (Default value = 'equil')

    Returns
    -------
    coord_der : dict
        dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        keys are of the form 'X_y' meaning the derivative of X wrt to y

    """
    # notation: X_y means derivative of X wrt y
    coord_der = {}
    coord_der['R'] = zernike_transform.transform(cR, 0, 0, 0)
    coord_der['Z'] = zernike_transform.transform(cZ, 0, 0, 0)
    coord_der['0'] = jnp.zeros_like(coord_der['R'])

    coord_der['R_r'] = zernike_transform.transform(cR, 1, 0, 0)
    coord_der['Z_r'] = zernike_transform.transform(cZ, 1, 0, 0)
    coord_der['R_v'] = zernike_transform.transform(cR, 0, 1, 0)
    coord_der['Z_v'] = zernike_transform.transform(cZ, 0, 1, 0)
    coord_der['R_z'] = zernike_transform.transform(cR, 0, 0, 1) * zeta_ratio
    coord_der['Z_z'] = zernike_transform.transform(cZ, 0, 0, 1) * zeta_ratio

    coord_der['R_rr'] = zernike_transform.transform(cR, 2, 0, 0)
    coord_der['Z_rr'] = zernike_transform.transform(cZ, 2, 0, 0)
    coord_der['R_rv'] = zernike_transform.transform(cR, 1, 1, 0)
    coord_der['Z_rv'] = zernike_transform.transform(cZ, 1, 1, 0)
    coord_der['R_rz'] = zernike_transform.transform(cR, 1, 0, 1) * zeta_ratio
    coord_der['Z_rz'] = zernike_transform.transform(cZ, 1, 0, 1) * zeta_ratio
    coord_der['R_vv'] = zernike_transform.transform(cR, 0, 2, 0)
    coord_der['Z_vv'] = zernike_transform.transform(cZ, 0, 2, 0)
    coord_der['R_vz'] = zernike_transform.transform(cR, 0, 1, 1) * zeta_ratio
    coord_der['Z_vz'] = zernike_transform.transform(cZ, 0, 1, 1) * zeta_ratio
    coord_der['R_zz'] = zernike_transform.transform(cR, 0, 0, 2) * zeta_ratio
    coord_der['Z_zz'] = zernike_transform.transform(cZ, 0, 0, 2) * zeta_ratio

    # axis or QS terms
    if len(zernike_transform.axn) or mode == 'qs':
        coord_der['R_rrr'] = zernike_transform.transform(cR, 3, 0, 0)
        coord_der['Z_rrr'] = zernike_transform.transform(cZ, 3, 0, 0)
        coord_der['R_rrv'] = zernike_transform.transform(cR, 2, 1, 0)
        coord_der['Z_rrv'] = zernike_transform.transform(cZ, 2, 1, 0)
        coord_der['R_rrz'] = zernike_transform.transform(
            cR, 2, 0, 1) * zeta_ratio
        coord_der['Z_rrz'] = zernike_transform.transform(
            cZ, 2, 0, 1) * zeta_ratio
        coord_der['R_rvv'] = zernike_transform.transform(cR, 1, 2, 0)
        coord_der['Z_rvv'] = zernike_transform.transform(cZ, 1, 2, 0)
        coord_der['R_rvz'] = zernike_transform.transform(
            cR, 1, 1, 1) * zeta_ratio
        coord_der['Z_rvz'] = zernike_transform.transform(
            cZ, 1, 1, 1) * zeta_ratio
        coord_der['R_rzz'] = zernike_transform.transform(
            cR, 1, 0, 2) * zeta_ratio
        coord_der['Z_rzz'] = zernike_transform.transform(
            cZ, 1, 0, 2) * zeta_ratio
        coord_der['R_vvv'] = zernike_transform.transform(cR, 0, 3, 0)
        coord_der['Z_vvv'] = zernike_transform.transform(cZ, 0, 3, 0)
        coord_der['R_vvz'] = zernike_transform.transform(
            cR, 0, 2, 1) * zeta_ratio
        coord_der['Z_vvz'] = zernike_transform.transform(
            cZ, 0, 2, 1) * zeta_ratio
        coord_der['R_vzz'] = zernike_transform.transform(
            cR, 0, 1, 2) * zeta_ratio
        coord_der['Z_vzz'] = zernike_transform.transform(
            cZ, 0, 1, 2) * zeta_ratio
        coord_der['R_zzz'] = zernike_transform.transform(
            cR, 0, 0, 3) * zeta_ratio
        coord_der['Z_zzz'] = zernike_transform.transform(
            cZ, 0, 0, 3) * zeta_ratio

        coord_der['R_rrvv'] = zernike_transform.transform(cR, 2, 2, 0)
        coord_der['Z_rrvv'] = zernike_transform.transform(cZ, 2, 2, 0)

    return coord_der


def compute_covariant_basis(coord_der, zernike_transform, mode='equil'):
    """Computes covariant basis vectors at grid points

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing the coordinate
        derivatives at each node, such as computed by ``compute_coordinate_derivatives``
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives
    mode : str
        one of 'equil' or 'qs'. Whether to compute field terms for equilibrium or quasisymmetry optimization (Default value = 'equil')

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
    if len(zernike_transform.axn) or mode == 'qs':
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


def compute_contravariant_basis(coord_der, cov_basis, jacobian, zernike_transform):
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
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives

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
    if len(zernike_transform.axn):
        axn = zernike_transform.axn
        con_basis['e^rho'] = put(con_basis['e^rho'], opsindex[:, axn], (cross(
            cov_basis['e_theta_r'][:, axn], cov_basis['e_zeta'][:, axn], 0)/jacobian['g_r'][axn]))
        # e^theta = infinite at the axis

    # metric coefficients
    con_basis['g^rr'] = dot(con_basis['e^rho'],  con_basis['e^rho'],  0)
    con_basis['g^rv'] = dot(con_basis['e^rho'],  con_basis['e^theta'], 0)
    con_basis['g^rz'] = dot(con_basis['e^rho'],  con_basis['e^zeta'], 0)
    con_basis['g^vv'] = dot(con_basis['e^theta'], con_basis['e^theta'], 0)
    con_basis['g^vz'] = dot(con_basis['e^theta'], con_basis['e^zeta'], 0)
    con_basis['g^zz'] = dot(con_basis['e^zeta'], con_basis['e^zeta'], 0)

    return con_basis


def compute_jacobian(coord_der, cov_basis, zernike_transform, mode='equil'):
    """Computes coordinate jacobian and derivatives

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing of coordinate
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives
    mode : str
        one of 'equil' or 'qs'. Whether to compute field terms for equilibrium or quasisymmetry optimization (Default value = 'equil')

    Returns
    -------
    jacobian : dict
        dictionary of ndarray, shape(N_nodes,) of coordinate
        jacobian and partial derivatives. Keys are of the form `g_x` meaning
        the x derivative of the coordinate jacobian g

    """
    # notation: subscripts denote partial derivatives
    jacobian = {}
    jacobian['g'] = dot(cov_basis['e_rho'], cross(
        cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0)

    jacobian['g_r'] = dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                          cov_basis['e_zeta_r'], 0), 0)
    jacobian['g_v'] = dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                          cov_basis['e_zeta_v'], 0), 0)
    jacobian['g_z'] = dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                          cov_basis['e_zeta_z'], 0), 0)

    # axis or QS terms
    if len(zernike_transform.axn) or mode == 'qs':
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


def compute_magnetic_field(cov_basis, jacobian, cI, Psi_lcfs, zernike_transform, mode='equil'):
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
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives
    mode : str
        one of 'equil' or 'qs'. Whether to compute field terms for equilibrium or quasisymmetry optimization (Default value = 'equil')

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
    r = zernike_transform.nodes[0]
    axn = zernike_transform.axn
    iota = iotafun(r, 0, cI)
    iota_r = iotafun(r, 1, cI)

    # toroidal flux
    magnetic_field['psi'] = Psi_lcfs*r**2
    magnetic_field['psi_r'] = 2*Psi_lcfs*r
    magnetic_field['psi_rr'] = 2*Psi_lcfs*jnp.ones_like(r)

    # contravariant B components
    magnetic_field['B^rho'] = jnp.zeros_like(r)
    magnetic_field['B^zeta'] = magnetic_field['psi_r'] / \
        (2*jnp.pi*jacobian['g'])
    if len(axn):
        magnetic_field['B^zeta'] = put(
            magnetic_field['B^zeta'], axn, magnetic_field['psi_rr'][axn] / (2*jnp.pi*jacobian['g_r'][axn]))
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
    if len(axn):
        magnetic_field['B^zeta_r'] = put(magnetic_field['B^zeta_r'], axn, -(magnetic_field['psi_rr']
                                                                            [axn]*jacobian['g_rr'][axn]) / (4*jnp.pi*jacobian['g_r'][axn]**2))
        magnetic_field['B^zeta_v'] = put(magnetic_field['B^zeta_v'], axn, 0)
        magnetic_field['B^zeta_z'] = put(magnetic_field['B^zeta_z'], axn, -(magnetic_field['psi_rr']
                                                                            [axn]*jacobian['g_rz'][axn]) / (2*jnp.pi*jacobian['g_r'][axn]**2))

    # QS terms
    if mode == 'qs':
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


def compute_plasma_current(coord_der, cov_basis, jacobian, magnetic_field, cI, Psi_lcfs, zernike_transform):
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
    Psi_lcfs : float
        total toroidal flux (in Webers) within LCFS.
    zernike_transform : ZernikeTransform
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
    r = zernike_transform.nodes[0]
    axn = zernike_transform.axn
    iota = iotafun(r, 0, cI)

    # axis terms
    if len(axn):
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
    if len(axn):
        plasma_current['J^rho'] = put(plasma_current['J^rho'], axn,
                                      (Bsub_zeta_rv[axn] - Bsub_theta_rz[axn]) / (jacobian['g_r'][axn]))

    plasma_current['J_con'] = plasma_current['J^rho']*cov_basis['e_rho'] + plasma_current['J^theta'] * \
        cov_basis['e_theta'] + plasma_current['J^zeta']*cov_basis['e_zeta']

    return plasma_current


def compute_magnetic_field_magnitude(cov_basis, magnetic_field, cI, zernike_transform):
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
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    B_mag : dict
        dictionary of ndarray, shape(N_nodes,) of magnetic field magnitude and derivatives

    """

    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    B_mag = {}
    r = zernike_transform.nodes[0]
    iota = iotafun(r, 0, cI)

    B_mag['|B|'] = jnp.abs(magnetic_field['B^zeta'])*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0) +
                                                              2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0) + dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))

    B_mag['|B|_v'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    B_mag['|B|_z'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    B_mag['|B|_vv'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vv']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_v'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vv'], 0))+2*iota*(dot(cov_basis['e_theta_vv'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vv'], 0)+2*dot(cov_basis['e_theta_v'], cov_basis['e_zeta_v'], 0))+2*(dot(cov_basis['e_zeta_v'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vv'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))**2 \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    B_mag['|B|_zz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_zz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_z'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_zz'], 0))+2*iota*(dot(cov_basis['e_theta_zz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_zz'], 0)+2*dot(cov_basis['e_theta_z'], cov_basis['e_zeta_z'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0))**2 \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    B_mag['|B|_vz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vz'], 0))+2*iota*(dot(cov_basis['e_theta_vz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta_v'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_theta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vz'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    return B_mag


def compute_force_magnitude(coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, cI, Psi_lcfs, zernike_transform):
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
    cI : ndarray
        parameters to pass to rotational transform function
    Psi_lcfs : float
        total toroidal flux (in Webers) within LCFS
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    force_magnitude : ndarray, shape(N_nodes,)
        force error magnitudes at each node.
    p_mag : ndarray, shape(N_nodes,)
        magnitude of pressure gradient at each node.

    """

    mu0 = 4*jnp.pi*1e-7
    r = zernike_transform.nodes[0]
    axn = zernike_transform.axn
    pres_r = presfun(r, 1, cP)

    # force balance error covariant components
    F_rho = jacobian['g']*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                           plasma_current['J^zeta']*magnetic_field['B^theta']) - pres_r
    F_theta = jacobian['g']*plasma_current['J^rho']*magnetic_field['B^zeta']
    F_zeta = -jacobian['g']*plasma_current['J^rho']*magnetic_field['B^theta']

    # axis terms
    if len(axn):
        Jsup_theta = (magnetic_field['B_rho_z'] -
                      magnetic_field['B_zeta_r']) / mu0
        Jsup_zeta = (magnetic_field['B_theta_r'] -
                     magnetic_field['B_rho_v']) / mu0
        F_rho = put(F_rho, axn, Jsup_theta[axn]*magnetic_field['B^zeta']
                    [axn] - Jsup_zeta[axn]*magnetic_field['B^theta'][axn])
        grad_theta = cross(cov_basis['e_zeta'], cov_basis['e_rho'], 0)
        gsup_vv = dot(grad_theta, grad_theta, 0)
        gsup_rv = dot(con_basis['e^rho'], grad_theta, 0)
        gsup_vz = dot(grad_theta, con_basis['e^zeta'], 0)
        F_theta = put(
            F_theta, axn, plasma_current['J^rho'][axn]*magnetic_field['B^zeta'][axn])
        F_zeta = put(F_zeta, axn, -plasma_current['J^rho']
                     [axn]*magnetic_field['B^theta'][axn])
        con_basis['g^vv'] = put(con_basis['g^vv'], axn, gsup_vv[axn])
        con_basis['g^rv'] = put(con_basis['g^rv'], axn, gsup_rv[axn])
        con_basis['g^vz'] = put(con_basis['g^vz'], axn, gsup_vz[axn])

    # F_i*F_j*g^ij terms
    Fg_rr = F_rho * F_rho * con_basis['g^rr']
    Fg_vv = F_theta*F_theta*con_basis['g^vv']
    Fg_zz = F_zeta * F_zeta * con_basis['g^zz']
    Fg_rv = F_rho * F_theta*con_basis['g^rv']
    Fg_rz = F_rho * F_zeta * con_basis['g^rz']
    Fg_vz = F_theta*F_zeta * con_basis['g^vz']

    # magnitudes
    force_magnitude = jnp.sqrt(
        Fg_rr + Fg_vv + Fg_zz + 2*Fg_rv + 2*Fg_rz + 2*Fg_vz)
    p_mag = jnp.sqrt(pres_r*pres_r*con_basis['g^rr'])

    return force_magnitude, p_mag
