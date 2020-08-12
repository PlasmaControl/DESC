import numpy as np
import functools

try:
#     raise
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import jax
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)
    x = jnp.linspace(0,5)
    y = jnp.exp(1)
    use_jax = True
    print('Using JAX',x.dtype)
except:
    jnp = np
    use_jax = False
    print('JAX borked, using numpy')


if use_jax:
    jit = jax.jit
    fori_loop = jax.lax.fori_loop
    @jit
    def put(arr,inds,vals):
        """basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff"""
        return jax.ops.index_update(arr,inds,vals)
    @jit
    def factorial(n):
        def body_fun(i,n):
            return n*i
        y = fori_loop(1.,n+1,body_fun,jnp.ones_like(n,dtype=jnp.float64))
        return y*(n>=0)

else:
    jit = lambda func, *args, **kwargs: func
    from scipy.special import factorial

    # we divide by zero in a few places but then overwrite with the 
    # correct asmptotic values, so lets suppress annoying warnings about that
    np.seterr(divide='ignore', invalid='ignore')
    arange = np.arange
    def put(arr,inds,vals):
        """basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff"""
        if isinstance(inds,tuple):
            inds = np.ravel_multi_index(inds,arr.shape)
        np.put(arr,inds,vals)
        return arr
    def fori_loop(lower, upper, body_fun, init_val):
        val = init_val
        for i in np.arange(lower, upper):
            val = body_fun(i, val)
        return val



def conditional_decorator(dec, condition, *args, **kwargs):
    """apply arbitrary decorator to a function if condition is met"""
    @functools.wraps(dec)
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func,*args,**kwargs)
    return decorator


def dot(a,b,axis):
    """wrapper for vector dot product"""
    return jnp.sum(a*b,axis=axis,keepdims=False)

def sign(x):
    """sign function, but returns 1 for x==0"""
    y = jnp.sign(x)
    y = put(y,jnp.where(y.flatten()==0)[0],1)
    return y

def cross(a,b,axis):
    """wrapper for vector cross product with some error handling"""
    return jnp.cross(a,b,axis=axis)
  
def rms(x):
    """Compute rms value of an array"""
    return jnp.sqrt(jnp.mean(x**2))
    
def iotafun(rho,nu, params):
    """Rotational transform
    
    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
        params (array-like): parameters to use for calculating profile
    """
    
    return jnp.polyval(jnp.polyder(params[::-1],nu),rho)
        
def pressfun(rho,nu,params):
    """Plasma pressure
    
    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
        params (array-like): parameters to use for calculating profile
    """

    return jnp.polyval(jnp.polyder(params[::-1],nu),rho)


def get_needed_derivatives(mode):
    if mode == 'force':
        return np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                         [2,0,0],[1,1,0],[1,0,1],[0,2,0],
                         [0,1,1],[0,0,2],[2,1,0],[1,2,0],
                         [1,1,1],[2,2,0]])
    else:
        raise NotImplementedError


def unpack_x(x,nRZ):
    """Unpacks the optimization state vector x into cR,cZ,cL components
    
    Args:
        x (ndarray): vector to unpack
        nRZ (int): number of R,Z coeffs        
    Returns:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cL (ndarray, shape(2M+1)*(2N+1)): spectral coefficients of lambda           
    """
    
    cR = x[:nRZ]
    cZ = x[nRZ:2*nRZ]
    cL = x[2*nRZ:]
    return cR,cZ,cL


def expand_resolution(x,zern_idx_old,zern_idx_new,lambda_idx_old,lambda_idx_new):
    """Expands solution to a higher resolution by filling with zeros
    
    Args:
        x (ndarray): solution at initial resolution
        zern_idx_old (ndarray of int, size(nRZ_old,3)): mode indices corresponding to initial R,Z
        zern_idx_new (ndarray of int, size(nRZ_new,3)): mode indices corresponding to new R,Z
        lambda_idx_old (ndarray of int, size(nL_old,2)): mode indices corresponding to initial lambda
        lambda_idx_new (ndarray of int, size(nL_new,2)): mode indices corresponding to new lambda
    
    Returns:
        x_new (ndarray): solution expanded to new resolution
    """
    
    
    cR,cZ,cL = unpack_x(x,len(zern_idx_old))
    
    cR_new = np.zeros(len(zern_idx_new))
    cZ_new = np.zeros(len(zern_idx_new))
    cL_new = np.zeros(len(lambda_idx_new))
    old_in_new = np.where((zern_idx_new[:, None] == zern_idx_old).all(-1))[0]
    cR_new[old_in_new] = cR
    cR_new[old_in_new] = cR
    old_in_new = np.where((lambda_idx_new[:, None] == lambda_idx_old).all(-1))[0]
    cL_new[old_in_new] = cL
    x_new = np.concatenate([cR_new,cZ_new,cL_new])
    
    return x_new


class FiniteDifferenceJacobian():
    def __init__(self, fun, rel_step=jnp.finfo(jnp.float64).eps**(1/3)):
        self.fun = fun
        self.rel_step = rel_step
    @conditional_decorator(functools.partial(jit,static_argnums=np.arange(0,2)), use_jax)
    def __call__(self,x0,*args):
        f0 = self.fun(x0,*args)
        m = f0.size
        n = x0.size
        J_transposed = jnp.empty((n, m))
        idx = jnp.arange(m).astype(jnp.int64)
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = self.rel_step * sign_x0 * jnp.maximum(1.0, jnp.abs(x0))
        h_vecs = jnp.diag(h)
        for i in range(h.size):
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            dx = x2[i] - x1[i]
            f1 = self.fun(x1,*args)
            f2 = self.fun(x2,*args)
            df = f2 - f1
            dfdx = df / dx
            put(J_transposed,i*m+idx,dfdx)
        if m == 1:
            J_transposed = jnp.ravel(J_transposed)
        return J_transposed.T
    
if use_jax:
    jacfwd = jax.jacfwd
    jacrev = jax.jacrev
else:
    jacfwd = FiniteDifferenceJacobian
    jacrev = FiniteDifferenceJacobian
