from ._current_potential import FourierCurrentPotentialField # CurrentPotentialField 
import numpy as np
from functools import partial
from desc.backend import jnp
from desc.optimizable import optimizable_parameter
from quadcoil.quadcoil import _input_checking, _parse_objectives, _parse_constraints
from quadcoil import QuadcoilParams, SurfaceRZFourierJAX, merge_callables, gen_winding_surface_arc
from desc.vmec_utils import ptolemy_linear_transform
from desc.grid import LinearGrid
from jax import jit, flatten_util


class QuadcoilField(FourierCurrentPotentialField):
    """
    
    Attributes:
    eq
    mpol
    ntor
    quadpoints_phi
    quadpoints_theta
    plasma_quadpoints_phi
    plasma_quadpoints_theta
    winding_mpol
    winding_ntor
    winding_quadpoints_phi
    winding_quadpoints_theta
    constraint_name
    constraint_type
    constraint_unit
    constraint_value
    objective_name
    objective_unit
    objective_weight

    plasma_coil_distance: 
    A scalar. If None, then the winding surface is a reference to another optimizable 
    surface, and the accompanying QuadcoilObjective.things and QuadcoilConstraints.things
    will include an optimizable surface.

    _ptolemy_Phi_A
    _ptolemy_Phi_c_indices
    _ptolemy_Phi_s_indices
    _ptolemy_R_plasma_A
    _ptolemy_R_plasma_c_indices
    _ptolemy_R_plasma_s_indices
    _ptolemy_Z_plasma_A
    _ptolemy_Z_plasma_c_indices
    _ptolemy_Z_plasma_s_indices
    _ptolemy_R_winding_A
    _ptolemy_R_winding_c_indices
    _ptolemy_R_winding_s_indices
    _ptolemy_Z_winding_A
    _ptolemy_Z_winding_c_indices
    _ptolemy_Z_winding_s_indices
    """

    _io_attrs_ = (
        FourierCurrentPotentialField._io_attrs_
        + []
    )

    _static_attrs = (
        FourierCurrentPotentialField._static_attrs
        + [
            '_plasma_quadpoints_phi_native',
            '_plasma_quadpoints_theta_native',
            '_plasma_quadpoints_phi',
            '_plasma_quadpoints_theta',
            '_ptolemy_R_winding_A',
            '_ptolemy_R_winding_c_indices',
            '_ptolemy_R_winding_s_indices',
            '_ptolemy_Z_winding_A',
            '_ptolemy_Z_winding_c_indices',
            '_ptolemy_Z_winding_s_indices',
            '_winding_quadpoints_phi',
            '_winding_quadpoints_theta',
            '_quadpoints_phi',
            '_quadpoints_theta',
            '_ptolemy_Phi_A',
            '_ptolemy_Phi_c_indices',
            '_ptolemy_Phi_s_indices',
            '_ptolemy_R_plasma_A',
            '_ptolemy_R_plasma_c_indices',
            '_ptolemy_R_plasma_s_indices',
            '_ptolemy_Z_plasma_A',
            '_ptolemy_Z_plasma_c_indices',
            '_ptolemy_Z_plasma_s_indices',
            'unravel_aux_dofs',
            '_aux_dofs_flat',
            '_f_quadcoil',
            '_g_quadcoil',
            '_h_quadcoil',
        ]
    )

    def __init__(
        self,
        eq,
        # By default, the plasma surface has quadpoints
        # number based on DESC quadrature points.
        plasma_quadpoints_phi_interp:int=1, 
        plasma_quadpoints_theta_interp:int=1, 
        quadpoints_phi=None,
        quadpoints_theta=None,
        # Winding surface optimization mode 
        winding_surface=None,
        # These are when winding surfaces are not provided
        plasma_coil_distance=None,
        M=6, # Number of poloidal harmonics in the winding surface. Equivalent to mpol in simsopt.
        N=5, # Number of toroidal harmonics in the winding surface. Equivalent to ntor in simsopt.
        winding_quadpoints_phi=None,
        winding_quadpoints_theta=None,
        # These are for winding surface optimization
        winding_surface_generator=gen_winding_surface_arc,
        objective_name='f_B',
        objective_weight=1.,
        objective_unit=None,
        constraint_name=(),
        constraint_type=(),
        constraint_unit=(),
        constraint_value=jnp.array([]),
        # Args for FourierCurrentPotentialField
        Phi_mn=np.array([0.0]),
        modes_Phi=np.array([[0, 0]]),
        I=0,
        G=0,
        sym_Phi=False,
        M_Phi=None,
        N_Phi=None,
        name="",
        check_orientation=True,
    ):
        # Checking if constraints and objective inputs are legal
        _input_checking(
            objective_name=objective_name,
            objective_weight=objective_weight,
            objective_unit=objective_unit,
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value,
        )
        self.eq = eq
        self._winding_surface_generator = winding_surface_generator

        # ----- Setting plasma quantities -----
        plasma_grid = LinearGrid(
            NFP=eq.surface.NFP,
            # If we set this to sym it'll only evaluate 
            # theta from 0 to pi.
            sym=False, 
            M=eq.surface.M,#Poloidal grid resolution.
            N=eq.surface.N,
            rho=1.0
        )
        self._plasma_quadpoints_phi_native = plasma_grid.nodes[plasma_grid.unique_zeta_idx, 2]/jnp.pi/2
        self._plasma_quadpoints_theta_native = plasma_grid.nodes[plasma_grid.unique_theta_idx, 1]/jnp.pi/2
        self._plasma_quadpoints_phi = interpolate_array(
            self._plasma_quadpoints_phi_native, 
            plasma_quadpoints_phi_interp, 
            1/eq.surface.NFP
        )
        self._plasma_quadpoints_theta = interpolate_array(
            self._plasma_quadpoints_theta_native, 
            plasma_quadpoints_theta_interp, 
            1.
        )
        # ----- Treating the winding surface -----
        # In the default behavior of FourierCurrentPotential,
        # the winding surface is part of params and can be optimized.
        # This is not always True in QUADCOIL. Sometimes we want to 
        # use an automatically generated winding surface instead.
        # So, before initializing the superclass, we must
        # handle the winding surface first.
        print('Phi_mn',Phi_mn)
        self._plasma_coil_distance = plasma_coil_distance
        if plasma_coil_distance is None:
            if winding_surface is None:
                raise ValueError('One of plasma_coil_distance and winding_surface must be provided.')
            # Plasma-blanket optimization. The winding surface and the plasma are both allowed to change.
            # ----- Building the desc surf -> quadcoil (simsopt) surf map -----
            (
                self._ptolemy_R_winding_A, 
                self._ptolemy_R_winding_c_indices, 
                self._ptolemy_R_winding_s_indices
            ) = ptolemy_identity_rev_precompute(
                self.R_basis.modes[:,1], 
                self.R_basis.modes[:,2]
            )
            (
                self._ptolemy_Z_winding_A, 
                self._ptolemy_Z_winding_c_indices, 
                self._ptolemy_Z_winding_s_indices
            ) = ptolemy_identity_rev_precompute(
                self.Z_basis.modes[:,1], 
                self.Z_basis.modes[:,2]
            )
            # ----- Initializing the superclass -----
            R_lmn = winding_surface.R_lmn
            Z_lmn = winding_surface.Z_lmn
            modes_R = winding_surface._R_basis.modes[:, 1:]
            modes_Z = winding_surface._Z_basis.modes[:, 1:]
            NFP = winding_surface.NFP
            sym = winding_surface.sym
            name = winding_surface.name
            winding_grid = LinearGrid(
                NFP=winding_surface.NFP,
                # If we set this to sym it'll only evaluate 
                # theta from 0 to pi.
                sym=False, 
                M=winding_surface.M,#Poloidal grid resolution.
                N=winding_surface.N,
                rho=1.0
            )
            winding_quadpoints_phi_1fp = winding_grid.nodes[winding_grid.unique_zeta_idx, 2]/jnp.pi/2
            # The quadpoints for the winding surface needs to cover all field periods. 
            # this repeats the quadpoints over all NFP.
            self._winding_quadpoints_phi = (
                jnp.repeat(jnp.linspace(0, 1, NFP, endpoint=False), len(winding_quadpoints_phi_1fp))
                + jnp.tile(winding_quadpoints_phi_1fp, NFP)
            )
            self._winding_quadpoints_theta = winding_grid.nodes[winding_grid.unique_theta_idx, 1]/jnp.pi/2
        else:
            self._ptolemy_R_winding_A = None
            self._ptolemy_R_winding_c_indices = None
            self._ptolemy_R_winding_s_indices = None
            self._ptolemy_Z_winding_A = None
            self._ptolemy_Z_winding_c_indices = None
            self._ptolemy_Z_winding_s_indices = None
            NFP = eq.surface.NFP
            sym = eq.surface.sym
            if winding_quadpoints_phi is None:
                winding_quadpoints_phi = jnp.linspace(0, 1, 32*NFP, endpoint=False)
            if winding_quadpoints_theta is None:
                winding_quadpoints_theta = jnp.linspace(0, 1, 34, endpoint=False)
            self._winding_quadpoints_phi = winding_quadpoints_phi
            self._winding_quadpoints_theta = winding_quadpoints_theta
            R_lmn = None # winding_surface.R_lmn
            Z_lmn = None # winding_surface.Z_lmn
            modes_R = None # winding_surface._R_basis.modes[:, 1:]
            modes_Z = None # winding_surface._Z_basis.modes[:, 1:]
            name = 'Automated winding surface'
            if winding_quadpoints_phi is None or winding_quadpoints_theta is None:
                winding_quadpoints_phi = jnp.linspace(0, 1, 32*NFP, endpoint=False)
                winding_quadpoints_theta = jnp.linspace(0, 1, 34, endpoint=False)

        if quadpoints_phi is None or quadpoints_theta is None:
            quadpoints_phi = jnp.linspace(0, 1/NFP, 32, endpoint=False)
            quadpoints_theta = jnp.linspace(0, 1, 34, endpoint=False)
        self._quadpoints_phi, self._quadpoints_theta = quadpoints_phi, quadpoints_theta

        if sym_Phi == "auto":
            sym_Phi = "sin" if sym else False
        print('modes_R', type(modes_R), 'modes_Z', type(modes_Z))
        print('modes_R', modes_R, 'modes_Z', modes_Z)
        super().__init__(
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
            # NFP=NFP,
            # sym=sym,
            NFP=NFP,
            sym=sym,
            M=M,
            N=N,
            name=name,
            check_orientation=check_orientation,
        )

        # ----- Building the operators for converting plasma Fourier coefficients -----
        (
            self._ptolemy_Phi_A, 
            self._ptolemy_Phi_c_indices, 
            self._ptolemy_Phi_s_indices
        ) = ptolemy_identity_rev_precompute(
            self.Phi_basis.modes[:,1], 
            self.Phi_basis.modes[:,2]
        )
        (
            self._ptolemy_R_plasma_A, 
            self._ptolemy_R_plasma_c_indices, 
            self._ptolemy_R_plasma_s_indices
        ) = ptolemy_identity_rev_precompute(
            eq.surface.R_basis.modes[:,1], 
            eq.surface.R_basis.modes[:,2]
        )
        (
            self._ptolemy_Z_plasma_A, 
            self._ptolemy_Z_plasma_c_indices, 
            self._ptolemy_Z_plasma_s_indices
        ) = ptolemy_identity_rev_precompute(
            eq.surface.Z_basis.modes[:,1], 
            eq.surface.Z_basis.modes[:,2]
        )

        # ----- Building the QUADCOIL callables -----
        # TODO: finish function parsing.
        f_obj, g_obj_list, h_obj_list, aux_dofs_obj = _parse_objectives(
            objective_name=objective_name, 
            objective_unit=objective_unit,
            objective_weight=objective_weight, 
        )
        g_cons_list, h_cons_list, aux_dofs_cons = _parse_constraints(
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value,
        )
        # Merging constraints and aux dofs from different sources
        g_list = g_obj_list + g_cons_list
        h_list = h_obj_list + h_cons_list
        # Auxiliary dofs (or slack variables)
        # For compatibility with params parsing in DESC, we must store aux_dof as a
        # jax numpy array aux_dofs_flat.
        aux_dofs_dict = aux_dofs_obj | aux_dofs_cons
        aux_dofs_flat, unravel_aux_dofs = flatten_util.ravel_pytree(aux_dofs_dict)
        self.unravel_aux_dofs = unravel_aux_dofs
        self._aux_dofs_flat = aux_dofs_flat

        self._f_quadcoil = lambda qp, x, f_obj=f_obj: f_obj(qp, x)
        self._g_quadcoil = lambda qp, x, g_list=g_list: merge_callables(g_list)(qp, x)
        self._h_quadcoil = lambda qp, x, h_list=h_list: merge_callables(h_list)(qp, x)
        # n_g = len(g_list)
        # n_h = len(h_list)

    # @property
    # def params_dict(self):
    #     """dict: dictionary of arrays of optimizable parameters."""
    #     return {
    #         key: jnp.atleast_1d(jnp.asarray(getattr(self, key))).copy()
    #         for key in self.optimizable_params
    #     }

    # @params_dict.setter
    # def params_dict(self, d):
    #     for key, val in d.items():
    #         if jnp.asarray(val).size:
    #             setattr(self, key, val)
    #     if self._auto_surface:
    #         params_auto_surface = self._generate_auto_surface_coeffs()
    #         setattr(self, "R_lmn", params_auto_surface["R_lmn"])
    #         setattr(self, "Z_lmn", params_auto_surface["Z_lmn"])

    @optimizable_parameter
    @property
    def aux_dofs_flat(self):
        """ Flattened slack variables for QUADCOIL """
        return self._aux_dofs_flat
    
    @aux_dofs_flat.setter
    def aux_dofs_flat(self, new):
        """ Flattened slack variables for QUADCOIL """
        self._aux_dofs_flat = new
    
    def plasma_surface_quadcoil(self):
        return self.params_to_plasma_surface_quadcoil(self.eq.params_dict)
    
    def winding_surface_quadcoil(self):
        return self.params_to_winding_surface_quadcoil(self.eq.params_dict, self.params_dict)
      
    @property
    def plasma_coil_distance(self):
        """ The plasma coil distance, if applicable. """
        return self._plasma_coil_distance
    
    # ----- Logics -----
    # In QUADCOIL, the form of the opimizable depends on the choice of 
    # objectives and constraints. This is because QUADCOIL targets non-smooth
    # objectives and constraints by automatically adding slack variables.
    # Because of this, we decided to put most logics that would usually be in 
    # Objective.build() and Objective.compute() into this Optimizable instead.
    # These includes conversion from DESC params into quadcoil objects 
    # and parsing objective and constraint functions.

    def params_to_dofs(self, params_qf):
        """ Converts params (input to QuadcoilObjective.compute()) into a quadcoil dofs dictionary """
        Phi_s_raw, Phi_c_raw = ptolemy_identity_rev_compute(self._ptolemy_Phi_A, self._ptolemy_Phi_c_indices, self._ptolemy_Phi_s_indices, params_qf['Phi_mn'])
        # Stellsym SurfaceRZFourier's dofs consists of 
        # [rc, zs]
        # Non-stellsym SurfaceRZFourier's dofs consists of 
        # [rc, rs, zc, zs]
        # Because rs, zs from ptolemy_identity_rev shares the same m, n 
        # arrays as rc, zc, they both have a zero as the first element 
        # that need to be removed.
        Phi_c = Phi_c_raw.flatten()
        Phi_s = Phi_s_raw.flatten()[1:]
        if self.sym_Phi:
            phi_quadcoil = Phi_c
        else:
            phi_quadcoil = jnp.concatenate([Phi_c, Phi_s])
        # Converting the flattened aux dofs dictionary back into a dict.
        return {'phi': phi_quadcoil} | self.unravel_aux_dofs(params_qf['aux_dofs_flat'])
    
    def params_to_plasma_surface_quadcoil(self, params_eq):
        """ Reads plasma surface info from params and creates a quadcoil.SurfaceRZFourierJAX object """
        r_s_raw, r_c_raw = ptolemy_identity_rev_compute(self._ptolemy_R_plasma_A, self._ptolemy_R_plasma_c_indices, self._ptolemy_R_plasma_s_indices, params_eq['Rb_lmn'])
        z_s_raw, z_c_raw = ptolemy_identity_rev_compute(self._ptolemy_Z_plasma_A, self._ptolemy_Z_plasma_c_indices, self._ptolemy_Z_plasma_s_indices, params_eq['Zb_lmn'])
            # Stellsym SurfaceRZFourier's dofs consists of 
        # [rc, zs]
        # Non-stellsym SurfaceRZFourier's dofs consists of 
        # [rc, rs, zc, zs]
        # Because rs, zs from ptolemy_identity_rev shares the same m, n 
        # arrays as rc, zc, they both have a zero as the first element 
        # that need to be removed.
        rc = r_c_raw.flatten()
        rs = r_s_raw.flatten()[1:]
        zc = z_c_raw.flatten()
        zs = z_s_raw.flatten()[1:]
        if self.eq.surface.sym:
            plasma_dofs = jnp.concatenate([rc, zs])
        else:
            plasma_dofs = jnp.concatenate([rc, rs, zc, zs])
        
        return SurfaceRZFourierJAX(
            nfp=self.NFP, stellsym=self.sym, 
            mpol=self.eq.surface.M, ntor=self.eq.surface.N, 
            quadpoints_phi=self._plasma_quadpoints_phi, 
            quadpoints_theta=self._plasma_quadpoints_theta,
            dofs=plasma_dofs
        )

    def params_to_winding_surface_quadcoil(self, params_eq, params_qf):
        """ Reads winding surface surface info from params and creates a quadcoil.SurfaceRZFourierJAX object """
        # One mode generates the winding surface automatically from the 
        # plasma surface
        if self.plasma_coil_distance is not None:
            plasma_surface = self.params_to_plasma_surface_quadcoil(params_eq)
            winding_dofs = self._winding_surface_generator(
                plasma_gamma=plasma_surface.gamma(), 
                d_expand=-self.plasma_coil_distance, # This is DESC's sign convention 
                nfp=self.NFP, 
                stellsym=self.sym,
                mpol=self.M,
                ntor=self.N,
            )
        # Another mode keeps the winding surface as an optimizable
        # (It will also appear in QuadcoilObjective.things)
        else:
            r_s_raw, r_c_raw = ptolemy_identity_rev_compute(self._ptolemy_R_winding_A, self._ptolemy_R_winding_c_indices, self._ptolemy_R_winding_s_indices, params_qf['R_lmn'])
            z_s_raw, z_c_raw = ptolemy_identity_rev_compute(self._ptolemy_Z_winding_A, self._ptolemy_Z_winding_c_indices, self._ptolemy_Z_winding_s_indices, params_qf['Z_lmn'])
                # Stellsym SurfaceRZFourier's dofs consists of 
            # [rc, zs]
            # Non-stellsym SurfaceRZFourier's dofs consists of 
            # [rc, rs, zc, zs]
            # Because rs, zs from ptolemy_identity_rev shares the same m, n 
            # arrays as rc, zc, they both have a zero as the first element 
            # that need to be removed.
            rc = r_c_raw.flatten()
            rs = r_s_raw.flatten()[1:]
            zc = z_c_raw.flatten()
            zs = z_s_raw.flatten()[1:]
            if self.sym:
                winding_dofs = jnp.concatenate([rc, zs])
            else:
                winding_dofs = jnp.concatenate([rc, rs, zc, zs])

        return SurfaceRZFourierJAX(
            nfp=self.NFP, stellsym=self.sym, 
            mpol=self.M, ntor=self.N, 
            quadpoints_phi=self._winding_quadpoints_phi, 
            quadpoints_theta=self._winding_quadpoints_theta,
            dofs=winding_dofs
        )

    def params_to_qp(self, params_eq, params_qf):
        """ Converts params (input to QuadcoilObjective.compute()) into a quadcoil.QuadcoilParams object"""
        plasma_surface = self.params_to_plasma_surface_quadcoil(params_eq)
        winding_surface = self.params_to_winding_surface_quadcoil(params_eq, params_qf)
        qp_out = QuadcoilParams(
            plasma_surface=plasma_surface, 
            winding_surface=winding_surface, 
            net_poloidal_current_amperes=params_qf['G'], 
            net_toroidal_current_amperes=params_qf['I'],
            # TODO: for free boundary optimization, support for 
            # Bnormal plasma and other coils are not available yet.
            Bnormal_plasma=None, 
            mpol=self.M, 
            ntor=self.N, 
            quadpoints_phi=self._quadpoints_phi,
            quadpoints_theta=self._quadpoints_theta, 
            stellsym=self.sym_Phi
        )
        return(qp_out)
        
# ----- Helper functions -----
def ptolemy_identity_rev_precompute(m_1, n_1):
    """
    We have split ptolemy_identity_rev into two parts:
    ``ptolemy_identity_rev_precompute`` and
    ``ptolemy_identity_rev_compute``. The ``original ptolemy_identity_rev``
    relies on numpy boolean indexing. Even when we set m_1, n_1 to static, 
    they will still be converted to traced arrays once jit happens, and the 
    numpy boolean indexing will break. Because of that, we perform all numpy
    operations in ``ptolemy_identity_rev_precompute`` during ``build()``,
    store the results as static, and then perform all jaxable operations in
    ``ptolemy_identity_rev_compute`` during ``compute()``.

    .. code-block:: python
    
        desc_to_vmec_surf_R = ptolemy_identity_rev_jit_precomputation(
            tuple(eq.surface.R_basis.modes[:,1]), 
            tuple(eq.surface.R_basis.modes[:,2])
        )
        desc_to_vmec_surf_Z = ptolemy_identity_rev_jit_precomputation(
            tuple(eq.surface.Z_basis.modes[:,1]), 
            tuple(eq.surface.Z_basis.modes[:,2])
        )
        rs_raw, rc_raw = desc_to_vmec_surf_R(eq.surface.R_lmn)
        zs_raw, zc_raw = desc_to_vmec_surf_Z(eq.surface.Z_lmn)# Stellsym SurfaceRZFourier's dofs consists of 
        # [rc, zs]
        # Non-stellsym SurfaceRZFourier's dofs consists of 
        # [rc, rs, zc, zs]
        # Because rs, zs from ptolemy_identity_rev shares the same m, n 
        # arrays as rc, zc, they both have a zero as the first element 
        # that need to be removed.
        rc = rc_raw.flatten()
        rs = rs_raw.flatten()[1:]
        zc = zc_raw.flatten()
        zs = zs_raw.flatten()[1:]
        if eq.sym:
            dofs = jnp.concatenate([rc, zs])
        else:
            dofs = jnp.concatenate([rc, rs, zc, zs])
    
    Converts from a double Fourier series of the form:
        ss * sin(m𝛉) * sin(n𝛟) + sc * sin(m𝛉) * cos(n𝛟) +
        cs * cos(m𝛉) * sin(n𝛟) + cc * cos(m𝛉) * cos(n𝛟)
    to the double-angle form:
        s * sin(m𝛉-n𝛟) + c * cos(m𝛉-n𝛟)
    using Ptolemy's sum and difference formulas.

    Parameters
    ----------
    m_1 : ndarray, shape(num_modes,)
    n_1 : ndarray, shape(num_modes,)
        ``R_basis_modes[:,1], R_basis_modes[:,2]`` or ``Z_basis_modes[:,1], Z_basis_modes[:,2]`` 

    Returns
    -------
    A, c_indices, s_indices : tuples
        .. code-block:: python

            # For calculating rs_raw, rc_raw,
            x = np.atleast_2d(desc_surf.R_lmn)
            y = (A @ x.T).T
            rs_raw, rc_raw = _modes_x_to_mnsc(vmec_modes, y)

    """
    try:
        from desc.backend import jnp, sign
        # from desc.vmec_utils import ptolemy_linear_transform # , _modes_x_to_mnsc
    except:
        raise ModuleNotFoundError('desc.backend.jnp and desc.backend.sign unavailable.')

    # Precomputing linear operators
    m_1, n_1 = map(np.atleast_1d, (m_1, n_1))
    desc_modes = np.vstack([np.zeros_like(m_1), m_1, n_1]).T
    A, vmec_modes = ptolemy_linear_transform(desc_modes)
    cmask = vmec_modes[:, 0] == 1
    smask = vmec_modes[:, 0] == -1
    c_indices = np.where(cmask)[0]
    s_indices = np.where(smask)[0]
    # A is 2d, and this converts 2d arr to tuple.
    A = tuple(map(tuple, A.tolist()))
    c_indices = tuple(c_indices.tolist())
    s_indices = tuple(s_indices.tolist())
    return A, c_indices, s_indices

@partial(jit, static_argnames=['A', 'c_indices', 's_indices', ])
def ptolemy_identity_rev_compute(A, c_indices, s_indices, x):
    A = jnp.array(A)
    y = (A @ x.T).T
    if len(c_indices):
        c = (y.T[jnp.array(c_indices)]).T
    if len(s_indices):
        s = (y.T[jnp.array(s_indices)]).T
        # if there are sin terms, add a zero for the m=n=0 mode
        s = jnp.concatenate([jnp.zeros_like(s.T[:1]), s.T]).T
    
    if not len(s_indices):
        s = jnp.zeros_like(c) 
    if not len(c_indices):
        c = jnp.zeros_like(s)
    assert len(s.T) == len(c.T)
    return s, c

# For interpolating quadpoints_theta and quadpoints_phi
def interpolate_array(x, k:int, period):
    x_roll = jnp.append(x, period+x[0])
    # differences between adjacent values
    dx = jnp.diff(x_roll)
    # interpolation weights: 0, 1/k, 2/k, ..., (k-1)/k
    w = jnp.linspace(0, 1, k, endpoint=False)
    # broadcast and add
    blocks = x[:, None] + dx[:, None] * w[None, :]
    # flatten and append the last element
    return blocks.ravel()
