import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ana_model(ABC):
    def __post_init__(self):
        self._modes = self._get_modes()
    @property
    def modes(self):
        return self._modes
    @abstractmethod
    def _get_modes(self):
        pass
    @abstractmethod
    def j_vec_ana_cal(self):
        pass
    @abstractmethod
    def B_vec_ana_cal(self):
        pass
    @abstractmethod
    def gradp_vec_ana_cal(self):
        pass