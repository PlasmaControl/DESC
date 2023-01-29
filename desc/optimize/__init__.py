"""Functions for minimization and wrappers for scipy methods."""

from . import _desc_wrappers, _scipy_wrappers
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .optimizer import Optimizer, optimizers, register_optimizer
from .stochastic import sgd
