import numpy as np
import functools
import warnings

import os
os.environ["JAX_PLATFORM_NAME"] = 'cpu'

if os.environ.get('DESC_USE_NUMPY'):
    jnp = np
    use_jax = False
    print('Using numpy backend, version={}, dtype={}'.format(
        np.__version__, np.linspace(0, 1).dtype))
else:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jax
            import jaxlib
            import jax.numpy as jnp
            from jax.config import config
            config.update("jax_enable_x64", True)
            x = jnp.linspace(0, 5)
            y = jnp.exp(x)
        use_jax = True
        print('Using JAX backend, jax version={}, jaxlib version={}, dtype={}'.format(
            jax.__version__, jaxlib.__version__, x.dtype))
    except:
        jnp = np
        use_jax = False
        warnings.warn('Failed to load JAX')
        print('Using numpy backend, version={}, dtype={}'.format(
            np.__version__, np.linspace(0, 1).dtype))

if use_jax:
    jit = jax.jit
    fori_loop = jax.lax.fori_loop

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
            n (int,array-like): input values. if n<0, returns 0

        Returns:
            n! (float): factorial of n

        """
        x = jnp.asarray(n+1)
        y = jnp.exp(jax.scipy.special.gammaln(x))
        y = jnp.where(x < 1, 0, y)
        return y

else:
    jit = lambda func, *args, **kwargs: func
    from scipy.special import factorial

    # we divide by zero in a few places but then overwrite with the
    # correct asmptotic values, so lets suppress annoying warnings about that
    np.seterr(divide='ignore', invalid='ignore')

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


class _Indexable():
    """Helper object for building indexes for indexed update functions.
    This is a singleton object that overrides the :code:`__getitem__` method
    to return the index it is passed.
    >>> jax.ops.index[1:2, 3, None, ..., ::2]
    (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
    """
    __slots__ = ()

    def __getitem__(self, index):
        return index


#: Index object singleton
opsindex = _Indexable()


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


def issorted(x, axis=None, tol=1e-12):
    """Checks if an array is sorted, within a given tolerance

    Checks whether x[i+1] - x[i] > tol

    Args:
        x (array-like): input values
        axis (int): axis along which to check if the array is sorted.
            If None, the flattened array is used.
        tol (float): tolerance for determining order. Array is still considered sorted
            if the difference between adjacent values is greater than -tol

    Returns:
        bool: whether the array is sorted
    """
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(np.diff(x, axis=axis) >= -tol)


def isalmostequal(x, axis=-1, tol=1e-12):
    """Checks if all values of an array are equal, to within a given tolerance

    Args:
        x (array-like): input values
        axis (int): axis along which to make comparison. If None, the flattened array is used
        tol (float): tolerance for comparison. Array is considered equal if std(x)*len(x)< tol along axis

    Returns:
        bool: whether the array is equal
    """
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(x.std(axis=axis)*x.shape[axis] < tol)


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
    x = jnp.atleast_1d(x)
    y = jnp.sign(x)
    x0 = jnp.where(y.flatten() == 0)[0]
    y = put(y, x0, 1)
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


def get_needed_derivatives(mode, axis=True):
    """Get array of derivatives needed for calculating objective function

    Args:
        mode (str): one of ``None``, ``'force'``, ``'accel'``, ``'qs'``, or ``'all'``
        axis (bool): whether to include terms needed for axis expansion

    Returns:
        derivs (array, shape (N,3)): combinations of derivatives of R,Z needed
            to compute objective function. Each row is one set, columns represent
            the order of derivative for [rho, theta, zeta].
    """
    equil_derivs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                             [0, 1, 1], [0, 0, 2]])
    axis_derivs = np.array([[2, 1, 0], [1, 2, 0], [1, 1, 1], [2, 2, 0]])
    qs_derivs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                          [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                          [0, 1, 1], [0, 0, 2], [3, 0, 0], [2, 1, 0],
                          [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2],
                          [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3],
                          [2, 2, 0]])
    if mode is None:
        return np.array([[0, 0, 0]])
    elif mode.lower() in ['force', 'accel']:
        if axis:
            return np.vstack([equil_derivs, axis_derivs])
        else:
            return equil_derivs
    elif mode.lower() in ['all', 'qs']:
        return qs_derivs
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

    def __init__(self, fun, rel_step=np.finfo(np.float64).eps**(1/3)):
        self.fun = fun
        self.rel_step = rel_step

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
        J_transposed = np.empty((n, m))
        idx = np.arange(m).astype(jnp.int64)
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = self.rel_step * sign_x0 * np.maximum(1.0, jnp.abs(x0))
        h_vecs = np.diag(h)
        for i in range(h.size):
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            dx = x2[i] - x1[i]
            f1 = self.fun(x1, *args)
            f2 = self.fun(x2, *args)
            df = f2 - f1
            dfdx = df / dx
            J_transposed = put(J_transposed, i*m+idx, dfdx)
        if m == 1:
            J_transposed = np.ravel(J_transposed)
        return J_transposed.T


class SPSAJacobian():
    """Class for computing jacobian simultaneous perturbation stochastic approximation

    Args:
        fun (callable): function to be differentiated
        rel_step (float): relative step size for finite difference
        N (int): number of samples to take
    """

    def __init__(self, fun, rel_step=1e-6, N=100):

        self.fun = fun
        self.rel_step = rel_step
        self.N = N

    def __call__(self, x0, *args, **kwargs):
        """Update and get the jacobian"""

        f0 = self.fun(x0, *args)
        m = f0.size
        n = x0.size

        J = np.zeros((m, n))
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = self.rel_step * sign_x0 * np.maximum(1.0, np.abs(x0))

        for i in range(self.N):
            dx = (np.random.binomial(1, .5, x0.shape)*2-1)*h
            x1 = x0 + dx
            x2 = x0 - dx
            dx = (x1 - x2).flatten()[np.newaxis]
            f1 = np.atleast_1d(self.fun(x1, *args))
            f2 = np.atleast_1d(self.fun(x2, *args))
            df = (f1-f2).flatten()[:, np.newaxis]
            dfdx = df/dx
            J += dfdx
        return J/self.N


class BroydenJacobian():
    """Class for computing jacobian using rank 1 updates

    Args:
        fun (callable): function to be differentiated
        x0 (array-like): starting point
        f0 (array-like): function evaluated at starting point
        J0 (array-like): estimate of jacobian at starting point
            If not given, the identity matrix is used
        minstep (float): minimum step size for updating the jacobian
    """

    def __init__(self, fun, x0, f0, J0=None, minstep=1e-12):

        self.fun = fun
        self.x0 = x0
        self.f0 = f0
        self.shape = (f0.size, x0.size)
        self.J = J0 if J0 is not None else np.eye(*self.shape)
        self.minstep = minstep
        self.x1 = self.x0
        self.f1 = self.f0

    def __call__(self, x, *args, **kwargs):
        """Update and get the jacobian"""

        self.x0 = self.x1
        self.f0 = self.f1
        self.x1 = x
        dx = self.x1-self.x0
        step = np.linalg.norm(dx)
        if step < self.minstep:
            return self.J
        else:
            self.f1 = self.fun(x, *args)
            df = self.f1 - self.f0
            update = (df - self.J.dot(dx))/step**2
            update = update[:, np.newaxis]*dx[np.newaxis, :]
            self.J += update
            return self.J


@conditional_decorator(functools.partial(jit), use_jax)
def polyder_vec(p, m):
    """Vectorized version of polyder for differentiating multiple polynomials of the same degree

    Args:
        p (ndarray, shape(N,M)): polynomial coefficients. Each row is 1 polynomial, in descending powers of x,
            each column is a power of x
        m (int >=0): order of derivative

    Returns:
        der (ndarray, shape(N,M)): polynomial coefficients for derivative in descending order
    """
    m = jnp.asarray(m, dtype=int)  # order of derivative
    p = jnp.atleast_2d(p)
    l = p.shape[0]               # number of polynomials
    n = p.shape[1] - 1           # order of polynomials

    D = jnp.arange(n, -1, -1)
    D = factorial(D)/factorial(D-m)

    p = jnp.roll(D*p, m, axis=1)
    idx = jnp.arange(p.shape[1])
    p = jnp.where(idx < m, 0, p)

    return p


@conditional_decorator(functools.partial(jit), use_jax)
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
    p = jnp.atleast_2d(p)
    npoly = p.shape[0]
    order = p.shape[1]
    x = jnp.asarray(x).flatten()
    nx = len(x)
    y = jnp.zeros((npoly, nx))

    def body_fun(k, y):
        return y * x + p[:, k][:, jnp.newaxis]
    y = fori_loop(0, order, body_fun, y)

    return y


# TODO: this stuff doesn't work without JAX
if use_jax:
    jacfwd = jax.jacfwd
    jacrev = jax.jacrev
    grad = jax.grad
else:
    jacfwd = FiniteDifferenceJacobian
    jacrev = FiniteDifferenceJacobian
    grad = FiniteDifferenceJacobian
