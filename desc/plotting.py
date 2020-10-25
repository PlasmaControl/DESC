import os
from matplotlib import rcParams, cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from desc.nodes import get_nodes_grid
from desc.zernike import ZernikeTransform, axis_posn
from desc.backend import get_needed_derivatives, iotafun, presfun
from desc.input_output import vmec_interpolate, read_desc
from desc.field_components import compute_coordinate_derivatives, compute_covariant_basis
from desc.field_components import compute_contravariant_basis, compute_jacobian
from desc.field_components import compute_magnetic_field, compute_plasma_current, compute_force_magnitude


colorblind_colors = [(0.0000, 0.4500, 0.7000),  # blue
                     (0.8359, 0.3682, 0.0000),  # vermillion
                     (0.0000, 0.6000, 0.5000),  # bluish green
                     (0.9500, 0.9000, 0.2500),  # yellow
                     (0.3500, 0.7000, 0.9000),  # sky blue
                     (0.8000, 0.6000, 0.7000),  # reddish purple
                     (0.9000, 0.6000, 0.0000)]  # orange
dashes = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # solid
          (3.7, 1.6, 0.0, 0.0, 0.0, 0.0),  # dashed
          (1.0, 1.6, 0.0, 0.0, 0.0, 0.0),  # dotted
          (6.4, 1.6, 1.0, 1.6, 0.0, 0.0),  # dot dash
          (3.0, 1.6, 1.0, 1.6, 1.0, 1.6),  # dot dot dash
          (6.0, 4.0, 0.0, 0.0, 0.0, 0.0),  # long dash
          (1.0, 1.6, 3.0, 1.6, 3.0, 1.6)]  # dash dash dot
matplotlib.rcdefaults()
rcParams['font.family'] = 'DejaVu Serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 10
rcParams['figure.facecolor'] = (1, 1, 1, 1)
rcParams['figure.figsize'] = (6, 4)
rcParams['figure.dpi'] = 141
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.labelsize'] = 'small'
rcParams['axes.titlesize'] = 'medium'
rcParams['lines.linewidth'] = 1
rcParams['lines.solid_capstyle'] = 'round'
rcParams['lines.dash_capstyle'] = 'round'
rcParams['lines.dash_joinstyle'] = 'round'
rcParams['xtick.labelsize'] = 'x-small'
rcParams['ytick.labelsize'] = 'x-small'
# rcParams['text.usetex']=True
color_cycle = cycler(color=colorblind_colors)
dash_cycle = cycler(dashes=dashes)
rcParams['axes.prop_cycle'] = color_cycle


def print_coeffs(cR, cZ, cL, zern_idx, lambda_idx):
    """prints coeffs to the terminal

    Args:
        cR,cZ,cL (array-like): spectral coefficients for R, Z, and lambda
        zern_idx, lambda_idx (array-like): mode numbers for zernike and fourier spectral basis.
    """

    print('Fourier-Zernike coefficients:')
    for k, lmn in enumerate(zern_idx):
        print('l: {:3d}, m: {:3d}, n: {:3d}, cR: {:16.8e}, cZ: {:16.8e}'.format(
            lmn[0], lmn[1], lmn[2], cR[k], cZ[k]))

    print('Lambda coefficients')
    for k, mn in enumerate(lambda_idx):
        print('m: {:3d}, n: {:3d}, cL: {:16.8e}'.format(mn[0], mn[1], cL[k]))


def plot_coeffs(cR, cZ, cL, zern_idx, lambda_idx, cR_init=None, cZ_init=None, cL_init=None, **kwargs):
    """Scatter plots of zernike and lambda coefficients, before and after solving

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cL (ndarray, shape(2M+1)*(2N+1)): spectral coefficients of lambda
        zern_idx (ndarray, shape(N_coeffs,3)): array of (l,m,n) indices for each spectral R,Z coeff
        lambda_idx (ndarray, shape(Nlambda,2)): indices for lambda spectral basis, ie an array of [m,n] for each spectral coefficient
        cR_init (ndarray, shape(N_coeffs,)): initial spectral coefficients of R
        cZ_init (ndarray, shape(N_coeffs,)): initial spectral coefficients of Z
        cL_init (ndarray, shape(2M+1)*(2N+1)): initial spectral coefficients of lambda

    Returns:
        fig (matplotlib.figure): handle to the figure
        ax (ndarray of matplotlib.axes): handle to axes
    """
    nRZ = len(cR)
    nL = len(cL)
    fig, ax = plt.subplots(1, 3, figsize=(cR.size//5, 6))
    ax = ax.flatten()

    ax[0].scatter(cR, np.arange(nRZ), s=2, label='Final')
    if cR_init is not None:
        ax[0].scatter(cR_init, np.arange(nRZ), s=2, label='Init')
    ax[0].set_yticks(np.arange(nRZ))
    ax[0].set_yticklabels([str(i) for i in zern_idx])
    ax[0].set_xlabel('$R$')
    ax[0].set_ylabel('[l,m,n]')
    ax[0].axvline(0, c='k', lw=.25)
    ax[0].legend(loc='upper right')

    ax[1].scatter(cZ, np.arange(nRZ), s=2, label='Final')
    if cZ_init is not None:
        ax[1].scatter(cZ_init, np.arange(nRZ), s=2, label='Init')
    ax[1].set_yticks(np.arange(nRZ))
    ax[1].set_yticklabels([str(i) for i in zern_idx])
    ax[1].set_xlabel('$Z$')
    ax[1].set_ylabel('[l,m,n]')
    ax[1].axvline(0, c='k', lw=.25)
    ax[1].legend()

    ax[2].scatter(cL, np.arange(nL), s=2, label='Final')
    if cL_init is not None:
        ax[2].scatter(cL_init, np.arange(nL), s=2, label='Init')
    ax[2].set_yticks(np.arange(nL))
    ax[2].set_yticklabels([str(i) for i in lambda_idx])
    ax[2].set_xlabel('$\lambda$')
    ax[2].set_ylabel('[m,n]')
    ax[2].axvline(0, c='k', lw=.25)
    ax[2].legend()

    plt.subplots_adjust(wspace=.5)

    return fig, ax


def plot_coord_surfaces(cR, cZ, zern_idx, NFP, nr=10, nt=12, ax=None, bdryR=None, bdryZ=None, **kwargs):
    """Plots solutions (currently only zeta=0 plane)

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zern_idx (ndarray, shape(Nc,3)): indices for R,Z spectral basis, ie an array of [l,m,n] for each spectral coefficient
        NFP (int): number of field periods
        nr (int): number of flux surfaces to show
        nt (int): number of theta lines to show
        ax (matplotlib.axes): axes to plot on. If None, a new figure is created.

    Returns:
        ax (matplotlib.axes): handle to axes used for the plot
    """

    Nr = 100
    Nt = 361
    rstep = Nr//nr
    tstep = Nt//nt
    zeta = kwargs.get('zeta', 0)
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2*np.pi, Nt)
    r, t = np.meshgrid(r, t, indexing='ij')
    r = r.flatten()
    t = t.flatten()
    z = zeta*np.ones_like(r)
    zernike_transform = ZernikeTransform([r, t, z], zern_idx, NFP)

    R = zernike_transform.transform(cR, 0, 0, 0).reshape((Nr, Nt))
    Z = zernike_transform.transform(cZ, 0, 0, 0).reshape((Nr, Nt))

    if ax is None:
        fig, ax = plt.subplots()
    # plot desired bdry
    if (bdryR is not None) and (bdryZ is not None):
        ax.plot(
            bdryR, bdryZ, color=colorblind_colors[1], lw=2, alpha=.5, dashes=(None, None))
    # plot r contours
    ax.plot(R.T[:, ::rstep], Z.T[:, ::rstep],
            color=colorblind_colors[0], lw=.5, dashes=(None, None))
    # plot actual bdry
    ax.plot(R.T[:, -1], Z.T[:, -1], color=colorblind_colors[0],
            lw=.5, dashes=(None, None))
    # plot theta contours
    ax.plot(R[:, ::tstep], Z[:, ::tstep],
            color=colorblind_colors[0], lw=.5, dashes=dashes[2])
    ax.axis('equal')
    ax.set_xlabel('$R$')
    ax.set_ylabel('$Z$')
    ax.set_title(kwargs.get('title'))
    return ax


def plot_IC(cR_init, cZ_init, zern_idx, NFP, nodes, cP, cI, **kwargs):
    """Plot initial conditions, such as the initial guess for flux surfaces,
    node locations, and profiles.

    Args:
        cR_init (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ_init (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zern_idx (ndarray, shape(N_coeffs,3)): array of (l,m,n) indices for each spectral R,Z coeff
        NFP (int): number of field periods
        cI (array-like): paramters to pass to rotational transform function
        cP (array-like): parameters to pass to pressure function

    Returns:
        fig (matplotlib.figure): handle to figure used for plotting
        ax (ndarray of matplotlib.axes): handles to axes used for plotting
    """

    fig = plt.figure(figsize=kwargs.get('figsize', (9, 3)))
    gs = matplotlib.gridspec.GridSpec(2, 3)
    ax0 = plt.subplot(gs[:, 0])
    ax1 = plt.subplot(gs[:, 1], projection='polar')
    ax2 = plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[1, 2])

    # coordinate surfaces
    plot_coord_surfaces(cR_init, cZ_init, zern_idx, NFP, nr=10, nt=12, ax=ax0)
    ax0.axis('equal')
    ax0.set_title(r'Initial guess, $\zeta=0$ plane')

    # node locations
    ax1.scatter(nodes[1], nodes[0], s=1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([0, np.pi/4, np.pi/2, 3/4*np.pi,
                    np.pi, 5/4*np.pi, 3/2*np.pi, 7/4*np.pi])
    ax1.set_xticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                         r'$\pi$', r'$\frac{4\pi}{4}$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax1.set_yticklabels([])
    ax1.set_title(r'Node locations, $\zeta=0$ plane', pad=20)

    # profiles
    xx = np.linspace(0, 1, 100)
    ax2.plot(xx, presfun(xx, 0, cP), lw=1)
    ax2.set_ylabel(r'$p$')
    ax2.set_xticklabels([])
    ax2.set_title('Profiles')
    ax3.plot(xx, iotafun(xx, 0, cI), lw=1)
    ax3.set_ylabel(r'$\iota$')
    ax3.set_xlabel(r'$\rho$')
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    ax = np.array([ax0, ax1, ax2, ax3])

    return fig, ax


def plot_fb_err(equil, domain='real', normalize='local', log=True, cmap='plasma', **kwargs):
    """Plots force balance error

    Args:
        equil (dict): dictionary of equilibrium solution quantities
        domain (str): one of 'real', 'sfl'. What basis to use for plotting, 
            real (R,Z) coordinates or straight field line (rho,vartheta)
        normalize (str, bool): Whether and how to normalize values
            None, False - no normalization, values plotted are force error in Newtons/m^3
            'local' - normalize by local pressure gradient
            'global' - normalize by pressure gradient at rho=0.5
            True - same as 'global'
        log (bool): plot logarithm of error or absolute value
        cmap (str,matplotlib.colors.Colormap): colormap to use

    Returns:
        Nothing, makes plot
    """

    cR = equil['cR']
    cZ = equil['cZ']
    cP = equil['cP']
    cI = equil['cI']
    Psi_lcfs = equil['Psi_lcfs']
    NFP = equil['NFP']
    zern_idx = equil['zern_idx']

    if np.max(zern_idx[:, 2]) == 0:
        Nz = 1
        rows = 1
    else:
        Nz = 6
        rows = 2

    Nr = kwargs.get('Nr', 51)
    Nv = kwargs.get('Nv', 90)
    Nlevels = kwargs.get('Nlevels', 100)

    nodes, vols = get_nodes_grid(NFP, nr=Nr, nt=Nv, nz=Nz)
    derivatives = get_needed_derivatives('all')
    zernike_transform = ZernikeTransform(nodes, zern_idx, NFP, derivatives)

    # compute fields components
    coord_der = compute_coordinate_derivatives(cR, cZ, zernike_transform)
    cov_basis = compute_covariant_basis(coord_der, zernike_transform)
    jacobian = compute_jacobian(coord_der, cov_basis, zernike_transform)
    con_basis = compute_contravariant_basis(
        coord_der, cov_basis, jacobian, zernike_transform)
    magnetic_field = compute_magnetic_field(cov_basis, jacobian, cI,
                                            Psi_lcfs, zernike_transform)
    plasma_current = compute_plasma_current(coord_der, cov_basis,
                                            jacobian, magnetic_field, cI, Psi_lcfs, zernike_transform)
    force_magnitude, p_mag = compute_force_magnitude(
        coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, cI, Psi_lcfs, zernike_transform)

    if domain == 'real':
        xlabel = r'R'
        ylabel = r'Z'
        R = zernike_transform.transform(cR, 0, 0, 0).reshape((Nr, Nv, Nz))
        Z = zernike_transform.transform(cZ, 0, 0, 0).reshape((Nr, Nv, Nz))
    elif domain == 'sfl':
        xlabel = r'$\vartheta$'
        ylabel = r'$\rho$'
        R = nodes[1].reshape((Nr, Nv, Nz))
        Z = nodes[0].reshape((Nr, Nv, Nz))
    else:
        raise ValueError("domain must be either 'real' or 'sfl'")

    if normalize == 'local':
        label = r'||F||/$\nabla$p'
        norm_errF = force_magnitude / p_mag
    elif normalize == 'global':
        label = r'||F||/$\nabla$p($\rho$=0.5)'
        halfn = np.where(nodes[0] == 0.5)[0][0]
        norm_errF = force_magnitude / p_mag[halfn]
    else:
        label = r'||F||'
        norm_errF = force_magnitude

    if log:
        label = r'$\mathregular{log}_{10}$('+label+')'
        norm_errF = np.log10(norm_errF)

    norm_errF = norm_errF.reshape((Nr, Nv, Nz))

    plt.figure()
    for k in range(Nz):
        ax = plt.subplot(rows, int(Nz/rows), k+1)
        cf = ax.contourf(R[:, :, k], Z[:, :, k], norm_errF[:, :, k],
                         cmap=cmap, extend='both', levels=Nlevels)
        if domain == 'real':
            ax.axis('equal')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = plt.colorbar(cf)
        if k == 0:
            cbar.ax.set_ylabel(label)
    plt.show()


def plot_comparison(equil0, equil1, label0='x0', label1='x1', **kwargs):
    """Plots force balance error

    Args:
        equil0, equil1 (dict): dictionary of two equilibrium solution quantities
        label0, label1 (str): labels for each equilibria

    Returns:
        Nothing, makes plot
    """

    cR0 = equil0['cR']
    cZ0 = equil0['cZ']
    NFP0 = equil0['NFP']
    zern_idx0 = equil0['zern_idx']

    cR1 = equil1['cR']
    cZ1 = equil1['cZ']
    NFP1 = equil1['NFP']
    zern_idx1 = equil1['zern_idx']

    if NFP0 == NFP1:
        NFP = NFP0
    else:
        raise Exception("NFP must be the same for both solutions")

    if max(np.max(zern_idx0[:, 2]), np.max(zern_idx1[:, 2])) == 0:
        Nz = 1
        rows = 1
    else:
        Nz = 6
        rows = 2

    Nr = kwargs.get('Nr', 8)
    Nv = kwargs.get('Nv', 13)

    NNr = 100
    NNv = 360

    # constant rho surfaces
    nodes_r, vols = get_nodes_grid(NFP, nr=Nr, nt=NNv, nz=Nz)
    zernike_transform0_r = ZernikeTransform(nodes_r, zern_idx0, NFP)
    zernike_transform1_r = ZernikeTransform(nodes_r, zern_idx1, NFP)

    # constant theta surfaces
    nodes_v, vols = get_nodes_grid(NFP, nr=NNr, nt=Nv, nz=Nz)
    zernike_transform0_v = ZernikeTransform(nodes_v, zern_idx0, NFP)
    zernike_transform1_v = ZernikeTransform(nodes_v, zern_idx1, NFP)

    R0r = zernike_transform0_r.transform(cR0, 0, 0, 0).reshape((Nr, NNv, Nz))
    Z0r = zernike_transform0_r.transform(cZ0, 0, 0, 0).reshape((Nr, NNv, Nz))
    R1r = zernike_transform1_r.transform(cR1, 0, 0, 0).reshape((Nr, NNv, Nz))
    Z1r = zernike_transform1_r.transform(cZ1, 0, 0, 0).reshape((Nr, NNv, Nz))

    R0v = zernike_transform0_v.transform(cR0, 0, 0, 0).reshape((NNr, Nv, Nz))
    Z0v = zernike_transform0_v.transform(cZ0, 0, 0, 0).reshape((NNr, Nv, Nz))
    R1v = zernike_transform1_v.transform(cR1, 0, 0, 0).reshape((NNr, Nv, Nz))
    Z1v = zernike_transform1_v.transform(cZ1, 0, 0, 0).reshape((NNr, Nv, Nz))

    plt.figure()
    for k in range(Nz):
        ax = plt.subplot(rows, int(Nz/rows), k+1)

        ax.plot(R0r[0, 0, k], Z0r[0, 0, k], 'bo')
        s0 = ax.plot(R0r[:, :, k].T, Z0r[:, :, k].T, 'b-')
        ax.plot(R0v[:, :, k], Z0v[:, :, k], 'b:')

        ax.plot(R1r[0, 0, k], Z1r[0, 0, k], 'ro')
        s1 = ax.plot(R1r[:, :, k].T, Z1r[:, :, k].T, 'r-')
        ax.plot(R1v[:, :, k], Z1v[:, :, k], 'r:')

        ax.axis('equal')
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        if k == 0:
            s0[0].set_label(label0)
            s1[0].set_label(label1)
            ax.legend(fontsize='xx-small')
    plt.show()


def plot_vmec_comparison(vmec_data, equil):
    """Plots comparison of VMEC and DESC solutions

    Args:
        vmec_data (dict): dictionary of vmec solution quantities.
        equil (dict): dictionary of equilibrium solution quantities.

    Returns:

    """

    cR = equil['cR']
    cZ = equil['cZ']
    NFP = equil['NFP']
    zern_idx = equil['zern_idx']

    Nr = 8
    Nv = 360
    if np.max(zern_idx[:, 2]) == 0:
        Nz = 1
        rows = 1
    else:
        Nz = 6
        rows = 2

    Nr_vmec = vmec_data['rmnc'].shape[0]-1
    s_idx = Nr_vmec % np.floor(Nr_vmec/(Nr-1))
    idxes = np.linspace(s_idx, Nr_vmec, Nr).astype(int)
    if s_idx != 0:
        idxes = np.pad(idxes, (1, 0), mode='constant')
    r = np.sqrt(idxes/Nr_vmec)
    v = np.linspace(0, 2*np.pi, Nv)
    z = np.linspace(0, 2*np.pi/NFP, Nz)
    rr, vv, zz = np.meshgrid(r, v, z, indexing='ij')
    rr = rr.flatten()
    vv = vv.flatten()
    zz = zz.flatten()
    nodes = [rr, vv, zz]
    zernike_transform = ZernikeTransform(nodes, zern_idx, NFP)

    R_desc = zernike_transform.transform(cR, 0, 0, 0).reshape((r.size, Nv, Nz))
    Z_desc = zernike_transform.transform(cZ, 0, 0, 0).reshape((r.size, Nv, Nz))

    R_vmec, Z_vmec = vmec_interpolate(
        vmec_data['rmnc'][idxes], vmec_data['zmns'][idxes], vmec_data['xm'], vmec_data['xn'], v, z)

    plt.figure()
    for k in range(Nz):
        ax = plt.subplot(rows, int(Nz/rows), k+1)
        ax.plot(R_vmec[0, 0, k], Z_vmec[0, 0, k], 'bo')
        s_vmec = ax.plot(R_vmec[:, :, k].T, Z_vmec[:, :, k].T, 'b-')
        ax.plot(R_desc[0, 0, k], Z_desc[0, 0, k], 'ro')
        s_desc = ax.plot(R_desc[:, :, k].T, Z_desc[:, :, k].T, 'r--')
        ax.axis('equal')
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        if k == 0:
            s_vmec[0].set_label('VMEC')
            s_desc[0].set_label('DESC')
            ax.legend(fontsize='xx-small')
    plt.show()


def plot_logo(savepath=None, **kwargs):
    """Plots the DESC logo

    Args:
        savepath (str or path-like, optional): path to save the figure to. 
            File format is inferred from the filename
        kwargs: options include 'Dcolor', 'Dcolor_rho', 'Dcolor_theta', 
            'Ecolor', 'Scolor', 'Ccolor', 'BGcolor', 'fig_width'

    Returns:
        fig (matplotlib.figure): handle to the figure used for plotting
        ax (matplotlib.axes): handle to the axis used for plotting
    """

    Dcolor = kwargs.get('Dcolor', 'xkcd:neon purple')
    Dcolor_rho = kwargs.get('Dcolor_rho', 'xkcd:neon pink')
    Dcolor_theta = kwargs.get('Dcolor_theta', 'xkcd:neon pink')
    Ecolor = kwargs.get('Ecolor', 'deepskyblue')
    Scolor = kwargs.get('Scolor', 'deepskyblue')
    Ccolor = kwargs.get('Ccolor', 'deepskyblue')
    BGcolor = kwargs.get('BGcolor', 'white')
    fig_width = kwargs.get('fig_width', 3)
    contour_lw_ratio = kwargs.get('contour_lw_ratio', 0.3)
    lw = fig_width**.5

    transparent = False
    if BGcolor == 'dark':
        BGcolor = 'xkcd:charcoal grey'
    elif BGcolor == 'light':
        BGcolor = 'white'
    elif BGcolor == 'clear':
        BGcolor = 'white'
        transparent = True

    path = os.path.dirname(os.path.abspath(__file__))
    equil = read_desc(path + '/../examples/DESC/outputs/LOGO_m12x18_n0x0')

    fig = plt.figure(figsize=(fig_width, fig_width/2))
    ax = fig.add_axes([0.1, 0.1, .8, .8])
    ax.axis('equal')
    ax.axis('off')
    ax.set_facecolor(BGcolor)
    fig.set_facecolor(BGcolor)
    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    bottom = 0
    top = 10
    Dleft = 0
    Dw = 8
    Dh = top-bottom + 2
    DX = Dleft + Dw/2
    DY = (top-bottom)/2
    Dright = Dleft + Dw

    Eleft = Dright + 0.5
    Eright = Eleft + 4

    Soffset = 1
    Sleft = Eright + 0.5
    Sw = 5
    Sright = Sleft + Sw

    Ctheta = np.linspace(np.pi/4, 2*np.pi-np.pi/4, 1000)
    Cleft = Sright + 0.75
    Cw = 4
    Ch = 11
    Cx0 = Cleft + Cw/2
    Cy0 = (top-bottom)/2

    # D
    cR = equil['cR']
    cZ = equil['cZ']
    zern_idx = equil['zern_idx']
    NFP = equil['NFP']
    R0, Z0 = axis_posn(cR, cZ, zern_idx, NFP)

    nr = kwargs.get('nr', 5)
    nt = kwargs.get('nt', 8)
    Nr = 100
    Nt = 361
    rstep = Nr//nr
    tstep = Nt//nt
    zeta = 0
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2*np.pi, Nt)
    r, t = np.meshgrid(r, t, indexing='ij')
    r = r.flatten()
    t = t.flatten()
    z = zeta*np.ones_like(r)
    zernike_transform = ZernikeTransform([r, t, z], zern_idx, NFP)
    bdry_nodes = np.array(
        [np.ones(Nt), np.linspace(0, 2*np.pi, Nt), np.ones(Nt)])
    bdry_zernike_transform = ZernikeTransform(bdry_nodes, zern_idx, NFP)

    R = zernike_transform.transform(cR, 0, 0, 0).reshape((Nr, Nt))
    Z = zernike_transform.transform(cZ, 0, 0, 0).reshape((Nr, Nt))
    bdryR = bdry_zernike_transform.transform(cR, 0, 0, 0)
    bdryZ = bdry_zernike_transform.transform(cZ, 0, 0, 0)

    R = (R-R0)/(R.max()-R.min())*Dw + DX
    Z = (Z-Z0)/(Z.max()-Z.min())*Dh + DY
    bdryR = (bdryR-R0)/(bdryR.max()-bdryR.min())*Dw + DX
    bdryZ = (bdryZ-Z0)/(bdryZ.max()-bdryZ.min())*Dh + DY

    # plot r contours
    ax.plot(R.T[:, ::rstep], Z.T[:, ::rstep],
            color=Dcolor_rho, lw=lw*contour_lw_ratio, ls='-')
    # plot theta contours
    ax.plot(R[:, ::tstep], Z[:, ::tstep],
            color=Dcolor_theta, lw=lw*contour_lw_ratio, ls='-')
    ax.plot(bdryR, bdryZ, color=Dcolor, lw=lw)

    # E
    ax.plot([Eleft, Eleft+1], [bottom, top],
            lw=lw, color=Ecolor, linestyle='-')
    ax.plot([Eleft, Eright], [bottom, bottom],
            lw=lw, color=Ecolor, linestyle='-')
    ax.plot([Eleft+1/2, Eright], [bottom+(top+bottom)/2, bottom +
                                  (top+bottom)/2], lw=lw, color=Ecolor, linestyle='-')
    ax.plot([Eleft+1, Eright], [top, top], lw=lw, color=Ecolor, linestyle='-')

    # S
    Sy = np.linspace(bottom, top+Soffset, 1000)
    Sx = Sw*np.cos(Sy*3/2*np.pi/(Sy.max()-Sy.min())-np.pi)**2 + Sleft
    ax.plot(Sx, Sy[::-1]-Soffset/2, lw=lw, color=Scolor, linestyle='-')

    # C
    Cx = Cw/2*np.cos(Ctheta)+Cx0
    Cy = Ch/2*np.sin(Ctheta)+Cy0
    ax.plot(Cx, Cy, lw=lw, color=Ccolor, linestyle='-')

    if savepath is not None:
        fig.savefig(savepath, facecolor=fig.get_facecolor(), edgecolor='none')

    return fig, ax
