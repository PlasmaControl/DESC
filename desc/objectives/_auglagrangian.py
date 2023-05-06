import jax
import numpy as np

from desc.backend import jnp

from .objective_funs import ObjectiveFunction


class AugLagrangianLS(ObjectiveFunction):
    """Augmented Lagrangian function for least-squares optimization

    Parameters
    ----------
    func : objective function
    constr : constraint functions
    """

    def __init__(self, func, constr):
        self.func = func
        self.constr = constr

    def scalar(self):
        return False

    def name(self):
        return "least squares augmented lagrangian"

    def derivatives(self):
        return

    def compute(self, x, lmbda, mu):
        L = self.func(x)
        c = self.compute_constraints(x)
        #        c = -lmbda * c + mu / 2 * c**2
        c = 1 / (np.sqrt(2 * mu)) * (-lmbda + mu * c)
        L = jnp.concatenate((L, c), axis=None)
        return L

    def compute_scalar(self, x, lmbda, mu):
        return np.linalg.norm(self.compute(x, lmbda, mu))

    def callback(self, x, lmbda, mu):
        L = self.compute(x, lmbda, mu)
        print("The Least Squares Lagrangian is " + str(L))

    def compute_constraints(self, x):
        c = jnp.array([])
        for i in range(len(self.constr)):
            c = jnp.concatenate((c, self.constr[i].fun(x)), axis=None)
        return c


class AugLagrangian(ObjectiveFunction):
    def __init__(self, func, constr):
        self.func = func
        self.constr = constr

    def scalar(self):
        return True

    def name(self):
        return "augmented lagrangian"

    def derivatives(self):
        return

    def compute(self, x, lmbda, mu):
        L = self.func(x)
        c = self.compute_constraints(x)
        # jax.debug.print("lmbda term is " + str(jnp.dot(lmbda,c)))
        # jax.debug.print("mu term is " + str(mu/2*jnp.dot(c,c)))
        # jax.debug.print("L is " + str(L))
        return L - jnp.dot(lmbda, c) + mu / 2 * jnp.dot(c, c)

    def compute_scalar(self, x, lmbda, mu):
        return self.compute(x, lmbda, mu)

    def callback(self, x, lmbda, mu):
        L = self.compute(x, lmbda, mu)
        print("The Lagrangian is " + str(L))

    def compute_constraints(self, x):
        c = jnp.array([])
        for i in range(len(self.constr)):
            c = jnp.concatenate((c, self.constr[i](x)), axis=None)
        return c
