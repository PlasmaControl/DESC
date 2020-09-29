import numpy as np
import functools
import warnings

try:
    #     raise
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import jax
    import jaxlib
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)
    x = jnp.linspace(0, 5)
    y = jnp.exp(1)
    use_jax = True
    print('Using JAX, version={}, jaxlib version={}, dtype={}'.format(
        jax.__version__,jaxlib.__version__, x.dtype))
except:
    jnp = np
    use_jax = False
    warnings.warn('Failed to load JAX, using numpy instead')


if use_jax:
    jit = jax.jit
    fori_loop = jax.lax.fori_loop
    @jit
    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing"

        basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff.


        Args:
            arr (array-like): Array to populate
            inds (array-like of int): Indices to populate
            vals (array-like): Values to insert

        Returns:
            arr (array-like). Input array with vals inserted at inds.
        """

        return jax.ops.index_update(arr, inds, vals)

    @jit
    def factorial(n):
        """Factorial function for jax backend

        Args:
            n (int): input values. if n<0, returns 0

        Returns:
            n! (float): factorial of n

        """

        def body_fun(i, n):
            return n*i
        y = fori_loop(1., n+1, body_fun, jnp.ones_like(n, dtype=jnp.float64))
        return y*(n >= 0)

else:
    jit = lambda func, *args, **kwargs: func
    from scipy.special import factorial

    # we divide by zero in a few places but then overwrite with the
    # correct asmptotic values, so lets suppress annoying warnings about that
    np.seterr(divide='ignore', invalid='ignore')
    arange = np.arange

    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing"

        basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff.


        Args:
            arr (array-like): Array to populate
            inds (array-like of int): Indices to populate
            vals (array-like): Values to insert

        Returns:
            arr (array-like). Input array with vals inserted at inds.
        """

        if isinstance(inds, tuple):
            inds = np.ravel_multi_index(inds, arr.shape)
        np.put(arr, inds, vals)
        return arr

    def fori_loop(lower, upper, body_fun, init_val):
        """Loop from lower to upper, applying body_fun to init_val

        This version is for the numpy backend, for jax backend see jax.fori_loop
        The semantics of ``fori_loop`` are given by this Python implementation::

            def fori_loop(lower, upper, body_fun, init_val):
                val = init_val
                for i in range(lower, upper):
                    val = body_fun(i, val)
                return val
        Args:
            lower: an integer representing the loop index lower bound (inclusive)
            upper: an integer representing the loop index upper bound (exclusive)
            body_fun: function of type ``(int, a) -> a``.
            init_val: initial loop carry value of type ``a``.

        Returns:
            Loop value from the final iteration, of type ``a``.
        """
        val = init_val
        for i in np.arange(lower, upper):
            val = body_fun(i, val)
        return val


def conditional_decorator(dec, condition, *args, **kwargs):
    """Apply arbitrary decorator to a function if condition is met

    Args:
        dec (decorator): Decorator to apply
        condition (bool): condition that must be met for decorator to be applied
        args: Arguments to pass to decorator
        kwargs: Keyword arguments to pass to decorator

    Returns:
       cond_dec (decorator): Decorator that acts like ``dec`` if ``condition``, 
       otherwise does nothing.
    """
    @functools.wraps(dec)
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func, *args, **kwargs)
    return decorator


def dot(a, b, axis):
    """Batched vector dot product

    Args:
        a (array-like): first array of vectors
        b (array-like): second array of vectors        
        axis (int): axis along which vectors are stored

    Returns:
        y (array-like): y = sum(a*b, axis=axis)
    """
    return jnp.sum(a*b, axis=axis, keepdims=False)


def sign(x):
    """Sign function, but returns 1 for x==0

    Args:
        x (array-like): array of input values

    Returns 
        y (array-like): 1 where x>=0, -1 where x<0
    """
    y = jnp.sign(x)
    y = put(y, jnp.where(y.flatten() == 0)[0], 1)
    return y


def cross(a, b, axis):
    """Batched vector cross product

    Args:
        a (array-like): first array of vectors
        b (array-like): second array of vectors        
        axis (int): axis along which vectors are stored

    Returns:
        y (array-like): y = a x b
    """
    return jnp.cross(a, b, axis=axis)


def rms(x):
    """Compute rms value of an array

    Args:
        x (array-like): input array

    Returns:
        y (float): rms value of x, eg sqrt(sum(x**2))
    """
    return jnp.sqrt(jnp.mean(x**2))


def iotafun(rho, nu, params):
    """Rotational transform

    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
        params (array-like): parameters to use for calculating profile

    Returns:
        iota (array-like): iota profile (or derivative) evaluated at rho
    """
    return jnp.polyval(jnp.polyder(params[::-1], nu), rho)


def presfun(rho, nu, params):
    """Plasma pressure

    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
        params (array-like): parameters to use for calculating profile

    Returns:
        pres (array-like): pressure profile (or derivative) evaluated at rho
    """
    return jnp.polyval(jnp.polyder(params[::-1], nu), rho)


def get_needed_derivatives(mode):
    """Get array of derivatives needed for calculating objective function

    Args:
        mode (str): one of 'force'`, 'accel', or 'all'

    Returns:
        derivs (array, shape (N,3)): combinations of derivatives of R,Z needed
            to compute objective function. Each row is one set, columns represent
            the order of derivative for [rho, theta, zeta].
    """

    if mode == 'force' or mode == 'accel':
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                         [0, 1, 1], [0, 0, 2], [2, 1, 0], [1, 2, 0],
                         [1, 1, 1], [2, 2, 0]])
    elif mode == 'all':
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                         [0, 1, 1], [0, 0, 2], [3, 0, 0], [2, 1, 0],
                         [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2],
                         [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3],
                         [2, 2, 0]])
    else:
        raise NotImplementedError


def unpack_x(x, nRZ):
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
    return cR, cZ, cL


class FiniteDifferenceJacobian():
    """Class that wraps a function and computes its jacobian using 2nd order centered finite differences

    Args:
        fun (callable): function to wrap
        rel_step (float): relative step size for finite differences. 
            step_size = rel_step * x0 * max(1,abs(x0))

    Returns:
       jac_fun (callable): object that computes the jacobian of fun.
"""

    def __init__(self, fun, rel_step=jnp.finfo(jnp.float64).eps**(1/3)):
        self.fun = fun
        self.rel_step = rel_step

    @conditional_decorator(functools.partial(jit, static_argnums=np.arange(0, 2)), use_jax)
    def __call__(self, x0, *args):
        """Evaluate the jacobian of fun at x0.

        Args:
            x0 (array-like): point to evaluate jacobian
            args: additional arguments passed to fun.

        Returns:
            dF/dx (array-like): Jacobian of fun at x0.
        """
        f0 = self.fun(x0, *args)
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
            f1 = self.fun(x1, *args)
            f2 = self.fun(x2, *args)
            df = f2 - f1
            dfdx = df / dx
            put(J_transposed, i*m+idx, dfdx)
        if m == 1:
            J_transposed = jnp.ravel(J_transposed)
        return J_transposed.T


def polyder_vec(p, m=1, pad=True):
    """Vectorized version of polyder for differentiating multiple polynomials of the same degree

    Args:
        p (ndarray, shape(N,M)): polynomial coefficients. Each row is 1 polynomial, in descending powers of x,
            each column is a power of x
        m (int >=0): order of derivative
        pad (bool): whether to pad output with zeros to be the same shape as input

    Returns:
        der (ndarray, shape(N,M) if pad, else shape(N,M-m)): polynomial coefficients for derivative in descending order
    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive")

    p = np.atleast_2d(p)
    n = p.shape[1] - 1  # order of polynomials
    y = p[:, :-1] * np.arange(n, 0, -1)
    if pad:
        y = np.pad(y, ((0, 0), (1, 0)))
    if m == 0:
        val = p
    else:
        val = polyder_vec(y, m - 1, pad)
    return val


def polyval_vec(p, x):
    """Evaluate a polynomial at specific values, 
    vectorized for evaluating multiple polynomials of the same degree.

    Parameters:
        p (ndarray, shape(N,M)): Array of coefficient for N polynomials of order M. 
            Each row is one polynomial, given in descending powers of x. 
        x (array-like, len(K,)): A number, or 1d array of numbers at
            which to evaluate p. If greater than 1d it is flattened.

    Returns:
        y (ndarray, shape(N,K)): polynomials evaluated at x.
            Each row corresponds to a polynomial, each column to a value of x

    Notes:
        Horner's scheme is used to evaluate the polynomial. Even so,
        for polynomials of high degree the values may be inaccurate due to
        rounding errors. Use carefully.
    """
    p = np.atleast_2d(p)
    npoly = p.shape[0]
    order = p.shape[1]
    x = np.asarray(x).flatten()
    nx = len(x)
    y = np.zeros((npoly, nx))
    for i in range(order):
        y = y * x + p[:, i][:, np.newaxis]
    return y


# TODO: this stuff doesn't work without JAX
if use_jax:
    jacfwd = jax.jacfwd
    jacrev = jax.jacrev
else:
    jacfwd = FiniteDifferenceJacobian
    jacrev = FiniteDifferenceJacobian
