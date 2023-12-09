import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ana_model(ABC):
    """Base class for analytic model for equilibriums with nested surfaces

    Args:
        ABC (_type_): _description_
    """
    def __post_init__(self):
        self._modes = self._get_modes()

    @property
    def modes(self):
        return self._modes

    @abstractmethod
    def _get_modes(self):
        pass

    @abstractmethod
    def j_vec_ana_cal(self, rtz):
        pass

    @abstractmethod
    def B_vec_ana_cal(self, rtz):
        pass

    @abstractmethod
    def gradp_vec_ana_cal(self,rtz):
        pass

    def jxB(self, rtz):
        return np.cross(self.j_vec_ana_cal(rtz), self.B_vec_ana_ca(rtz), axis=1)
