import jax.numpy as jno
import numpy as np
from jax import core

multiply_add_p = core.Primitive("multiply_add")

def multiply_add_impl(x,y,z):
    return np.add(np.multiply(x,y),z)

multiply_add_p.def_impl(multiply_add_impl)

def multiply_add_prim(x,y,z):
    return multiply_add_p.bind(x,y,z)

assert multiply_add_prim(1,1,2) == 3
