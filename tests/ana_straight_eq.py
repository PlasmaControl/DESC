from scipy.constants import mu_0
from dataclasses import dataclass
from .ana_util import modes_gen
from .ana_base import ana_equilibrium
import numpy as np


@dataclass
class model_theta_pinch_eq1(ana_equilibrium):
    R: float
    a: float
    Psi0: float

    def _get_params(self):
        A = ((2 * self.Psi0) / (np.pi * self.a**2 * (np.sqrt(2) + 1))) ** 2
        p_coeff = A / 2 / mu_0
        return {
            "Rb_lmn": {"modes_R": ((0, 0), (1, 0)), "R_lmn": (self.R, self.a)},
            "Zb_lmn": {"modes_Z": ((0, 0), (-1, 0)), "Z_lmn": (0, -self.a)},
            "pressure": {"params": (p_coeff, -p_coeff), "modes": (0, 2)},
            "sym": False,
        }

    def get_modes(self, L, M, N):
        raise NotImplementedError("Getting mode not implemented")

    def radial(self, rtz):
        rho = rtz[:, 0]
        a = self.a
        return np.sqrt(np.sqrt(2) + 1) * a * np.sqrt(np.sqrt(1 + rho**2) - 1)

    def inverse_rep(self, rtz):
        t = rtz[:, 1]
        return np.stack(
            (self.radial(rtz) * np.cos(t), t, -self.radial(rtz) * np.sin(t)), axis=1
        )
