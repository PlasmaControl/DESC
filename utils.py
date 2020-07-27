import numpy as np
from zernike import ZernikeTransform
import matplotlib.pyplot as plt




colorblind_colors = [(0.0000, 0.4500, 0.7000), # blue
                     (0.8359, 0.3682, 0.0000), # vermillion
                     (0.0000, 0.6000, 0.5000), # bluish green
                     (0.9500, 0.9000, 0.2500), # yellow
                     (0.3500, 0.7000, 0.9000), # sky blue
                     (0.8000, 0.6000, 0.7000), # reddish purple
                     (0.9000, 0.6000, 0.0000)] # orange
dashes = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), # solid
          (3.7, 1.6, 0.0, 0.0, 0.0, 0.0), # dashed
          (1.0, 1.6, 0.0, 0.0, 0.0, 0.0), # dotted
          (6.4, 1.6, 1.0, 1.6, 0.0, 0.0), # dot dash
          (3.0, 1.6, 1.0, 1.6, 1.0, 1.6), # dot dot dash
          (6.0, 4.0, 0.0, 0.0, 0.0, 0.0), # long dash
          (1.0, 1.6, 3.0, 1.6, 3.0, 1.6)] # dash dash dot
import matplotlib
from matplotlib import rcParams, cycler
matplotlib.rcdefaults()
rcParams['font.family'] = 'DejaVu Serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 10
rcParams['figure.facecolor'] = (1,1,1,1)
rcParams['figure.figsize'] = (6,4)
rcParams['figure.dpi'] = 141
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.labelsize'] =  'small'
rcParams['axes.titlesize'] = 'medium'
rcParams['lines.linewidth'] = 2.5
rcParams['lines.solid_capstyle'] = 'round'
rcParams['lines.dash_capstyle'] = 'round'
rcParams['lines.dash_joinstyle'] = 'round'
rcParams['xtick.labelsize'] = 'x-small'
rcParams['ytick.labelsize'] = 'x-small'
# rcParams['text.usetex']=True
color_cycle = cycler(color=colorblind_colors)
dash_cycle = cycler(dashes=dashes)
rcParams['axes.prop_cycle'] =  color_cycle


def dot(a,b,axis=-1):
    """wrapper for vector dot product"""
    return np.sum(a*b,axis=axis,keepdims=False)

def cross(a,b,axis=-1):
    """wrapper for vector cross product with some error handling"""
    if a.size and b.size:
        return np.cross(a,b,axis=axis)
    else:
        return np.array([])
    
def rms(x):
    """Compute rms value of an array"""
    return np.sqrt(np.mean(x**2))

def findifjac(x0,fun,rel_step=np.finfo(np.float64).eps**(1/3)):
    """Compute Jacobian matrix using 2nd order centered finite differences
    
    Args:
        x0 (array-like): location where to evaluate jacobian
        fun (callable): function to take jacobian of
        rel_step (float): relative step size for finite difference. h = rel_step*max(1,abs(x0))
    
    """
    
    f0 = fun(x0)
    m = f0.size
    n = x0.size
    J_transposed = np.empty((n, m))
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1
    h = rel_step * sign_x0 * np.maximum(1.0, np.abs(x0))
    h_vecs = np.diag(h)
    for i in range(h.size):
        x1 = x0 - h_vecs[i]
        x2 = x0 + h_vecs[i]
        dx = x2[i] - x1[i]
        f1 = fun(x1)
        f2 = fun(x2)
        df = f2 - f1
        J_transposed[i] = df / dx

    if m == 1:
        J_transposed = np.ravel(J_transposed)

    return J_transposed.T




def plotter(cR,cZ,M,N,NFP,bdryR=None,bdryZ=None,nr=20,ntheta=30,ax=None):
    """Plots solutions (currently only zeta=0 plane)

    Args:
        cR (array-like): spectral coefficients of R
        cZ (array-like): spectral coefficients of Z
        M (int): maximum poloidal mode number
        N (int): maximum toroidal mode number
        NFP (int): number of field periods
        bdryR (array-like): R coordinates of desired boundary surface
        bdryZ (array-like): Z coordinates of desired boundary surface
        nr (int): number of flux surfaces to show
        ntheta (int): number of theta lines to show
        ax (matplotlib.axes): axes to plot on. If None, a new figure is created.
    
    Returns:
        ax (matplotlib.axes): handle to axes used for the plot
    """

    Nr = 100
    Ntheta = 100
    rstep = Nr//nr
    thetastep = Ntheta//ntheta
    r = np.linspace(0,1,Nr)
    theta = np.linspace(0,2*np.pi,Ntheta)
    rr,tt = np.meshgrid(r,theta,indexing='ij')
    rr = rr.flatten()
    tt = tt.flatten()
    zz = np.zeros_like(rr)
    zernt = ZernikeTransform([rr,tt,zz],M,N,NFP)

    R = zernt.transform(cR).reshape((Nr,Ntheta))
    Z = zernt.transform(cZ).reshape((Nr,Ntheta))

    foo = ax if ax else plt
    # plot desired bdry
    if bdryR is not None and bdryZ is not None:
        foo.plot(bdryR,bdryZ,color=colorblind_colors[1])
    # plot r contours
    foo.plot(R.T[:,::rstep],Z.T[:,::rstep],color=colorblind_colors[0],lw=.5)
    # plot actual bdry
    foo.plot(R.T[:,-1],Z.T[:,-1],color=colorblind_colors[0],lw=.5)
    # plot theta contours
    foo.plot(R[:,::thetastep],Z[:,::thetastep],color=colorblind_colors[0],lw=.5,ls='--');
    foo.axis('equal')

    return plt.gca()