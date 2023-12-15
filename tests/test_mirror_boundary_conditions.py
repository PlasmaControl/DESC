"""Test boundary conditions for mirror"""

import numpy as np
import pytest
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from dataclasses import dataclass