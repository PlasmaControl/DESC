
from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.integrals import virtual_casing_biot_savart
from scipy.constants import mu_0
from desc.vmec_utils import ptolemy_linear_transform
from jax import jit
from functools import partial
from desc.vmec_utils import ptolemy_identity_fwd
from quadcoil import make_rzfourier_mc_ms_nc_ns
import numpy as np
# Used in create_source_grid only
from ..integrals.singularities import best_params, best_ratio
from desc.grid import LinearGrid
from desc.compute import get_profiles, get_transforms
from desc.utils import Timer, warnif
from desc.integrals import DFTInterpolator, FFTInterpolator

# Data keys needed to calculate Bnormal_plasma.
_BPLASMA_DATA_KEYS = ['K_vc', 'B', 'R', 'phi', 'Z', 'e^rho', 'n_rho', '|e_theta x e_zeta|']
# Data keys needed to calculate Bnormal from external coils.
_BCOIL_DATA_KEYS = ['R', 'Z', 'n_rho', 'phi', '|e_theta x e_zeta|']

# ----- Helper functions -----
def compute_Bnormal_plasma(constants, params_eq, _bplasma_chunk_size):
    # Using the stored transforms to calculate B_normal_plasma
    source_data = compute_fun(
        'desc.equilibrium.equilibrium.Equilibrium',
        _BPLASMA_DATA_KEYS,
        params=params_eq,
        transforms=constants['source_transforms'],
        profiles=constants['source_profiles'],
    )
    eval_data = compute_fun(
        'desc.equilibrium.equilibrium.Equilibrium',
        _BPLASMA_DATA_KEYS,
        params=params_eq,
        transforms=constants['eval_transforms'],
        profiles=constants['eval_profiles'],
    )
    Bplasma = virtual_casing_biot_savart(
        eval_data,
        source_data,
        constants['interpolator'],
        chunk_size=_bplasma_chunk_size,
    )
    # need extra factor of B/2 bc we're evaluating on plasma surface
    Bplasma = Bplasma + eval_data['B'] / 2
    Bnormal_plasma = jnp.sum(Bplasma * eval_data['n_rho'], axis=1)
    return Bnormal_plasma

def compute_eval_data_coils(constants, params_eq):
    eval_data_coils = compute_fun(
        'desc.equilibrium.equilibrium.Equilibrium',
        _BCOIL_DATA_KEYS,
        params=params_eq,
        transforms=constants['eval_transforms'],
        profiles=constants['eval_profiles'],
    )
    coils_x = jnp.array([eval_data_coils['R'], eval_data_coils['phi'], eval_data_coils['Z']]).T
    coils_n_rho = eval_data_coils['n_rho']
    return coils_x, coils_n_rho

def compute_Bnormal_ext(constants, params_field, _bs_chunk_size):
    coils_nrho = constants['coils_n_rho']
    coils_x = constants['coils_x']
    B_ext = constants['sum_field'].compute_magnetic_field(
        coils_x,
        source_grid=constants['field_grid'],
        basis='rpz',
        params=params_field,
        chunk_size=_bs_chunk_size,
    )
    B_ext = jnp.sum(B_ext * coils_nrho, axis=-1)
    return B_ext

def compute_G(params_eq, constants):
    G_data = compute_fun(
        'desc.equilibrium.equilibrium.Equilibrium',
        ['G'],                
        params=params_eq,                   
        transforms=constants['net_poloidal_current_transforms'],
        profiles=constants['net_poloidal_current_profiles'],
    )
    net_poloidal_current_amperes = - G_data['G'][0] / mu_0 * 2 * jnp.pi
    return net_poloidal_current_amperes

def ptolemy_identity_rev_precompute(m_1, n_1):
    '''
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

    '''
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

def compute_Bnormal(
        field, constants, Bnormal_shape,
        enable_Bnormal_plasma,
        eq_fixed, field_fixed, 
        params_eq, params_field, 
        bs_chunk_size, bplasma_chunk_size):
    
    Bnormal = jnp.zeros(Bnormal_shape)

    if field:
        if eq_fixed:
            if field_fixed:
                Bnormal += constants['Bnormal_ext'].reshape(Bnormal.shape) # eq and field fixed
            else:
                Bnormal += compute_Bnormal_ext(constants, params_field, bs_chunk_size).reshape(Bnormal.shape) # neither fixed, field fixed only.
        else:
            coils_x, coils_n_rho = compute_eval_data_coils(constants, params_eq)
            constants['coils_x'] = coils_x
            constants['coils_n_rho'] = coils_n_rho
            Bnormal += compute_Bnormal_ext(constants, params_field, bs_chunk_size).reshape(Bnormal.shape) # neither fixed, field fixed only.

    # Plasma fields
    if enable_Bnormal_plasma:
        if eq_fixed:
            Bnormal += constants['Bnormal_plasma'].reshape(Bnormal.shape)
        else:
            Bnormal += compute_Bnormal_plasma(constants, params_eq, bplasma_chunk_size).reshape(Bnormal.shape)

    return Bnormal

# Flipping the current potential of a quadcoil 
# phi Fourier array in the toroidal direction
def toroidal_flip(phi, m, n):
    m = np.array(m)
    n = np.array(n)
    phi_swapped = np.array(phi)
    index_map = {(mi, ni): i for i, (mi, ni) in enumerate(zip(m, n))}
    for i, (mi, ni) in enumerate(zip(m, n)):
        # skip n=0 (it’s its own opposite)
        if ni == 0:
            continue
        if mi != 0:
            j = index_map.get((mi, -ni))
            if j is not None:
                phi_swapped[i] = phi[j]
        else:
            phi_swapped[i] = -phi[i]
    return phi_swapped

def quadcoil_phi_to_desc_phi(phi_mn_quadcoil, stellsym, mpol, ntor):
    '''Converts quadcoil phi to desc phi.'''
    if stellsym:
        # The dofs contain rc, zs, and rc has one more element than zs.
        phis = phi_mn_quadcoil
        phic = jnp.zeros(len(phis) + 1)
    else:
        # The dofs contain rc, zs, and rc has one more element than zs.
        len_sin = len(phi_mn_quadcoil)//2
        phis = phi_mn_quadcoil[-len_sin:]
        phic = phi_mn_quadcoil[:-len_sin]

    phis = jnp.insert(phis, 0, 0.)
    mc, _, nc, _ = make_rzfourier_mc_ms_nc_ns(mpol, ntor)
    modes_M, modes_N, Phi_mn = ptolemy_identity_fwd(mc, nc, phis, phic)
    Phi_mn = Phi_mn.flatten()
    return Phi_mn, modes_M, modes_N


def create_source(eq, source_grid_in, plasma_grid_in):
    if source_grid_in is None:
        # for axisymmetry we still need to know about toroidal effects, so its
        # cheapest to pretend there are extra field periods
        source_grid = LinearGrid(
            rho=np.array([1.0]),
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )
        source_profiles = get_profiles(_BPLASMA_DATA_KEYS, obj=eq, grid=source_grid)
        source_transforms = get_transforms(_BPLASMA_DATA_KEYS, obj=eq, grid=source_grid)
    else:
        source_grid = source_grid_in
    # Creating interpolator for Bnormal_plasma
    ratio_data = eq.compute(
        ["|e_theta x e_zeta|", "e_theta", "e_zeta"], grid=source_grid
    )
    st, sz, q = best_params(source_grid, best_ratio(ratio_data))
    try:
        interpolator = FFTInterpolator(plasma_grid_in, source_grid, st, sz, q)
    except AssertionError as e:
        warnif(
            True,
            msg="Could not build fft interpolator, switching to dft which is slow."
            "\nReason: " + str(e),
        )
        interpolator = DFTInterpolator(plasma_grid_in, source_grid, st, sz, q)
    return (
        source_profiles,
        source_transforms,
        interpolator
    )