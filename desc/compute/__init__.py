"""Functions for computing field and plasma quantities from an equilibrium."""

from ._core import (
    compute_cartesian_coords,
    compute_contravariant_basis,
    compute_contravariant_metric_coefficients,
    compute_covariant_basis,
    compute_covariant_metric_coefficients,
    compute_flux_coords,
    compute_geometry,
    compute_jacobian,
    compute_lambda,
    compute_pressure,
    compute_rotational_transform,
    compute_toroidal_coords,
    compute_toroidal_flux,
    compute_toroidal_flux_gradient,
)
from ._equil import compute_energy, compute_force_error
from ._field import (
    compute_B_dot_gradB,
    compute_boozer_magnetic_field,
    compute_contravariant_current_density,
    compute_contravariant_magnetic_field,
    compute_covariant_magnetic_field,
    compute_magnetic_field_magnitude,
    compute_magnetic_pressure_gradient,
    compute_magnetic_tension,
)
from ._qs import compute_boozer_coordinates, compute_quasisymmetry_error
from ._stability import compute_magnetic_well, compute_mercier_stability
from .data_index import data_index

# defines the order in which objective arguments get concatenated into the state vector
arg_order = ("R_lmn", "Z_lmn", "L_lmn", "p_l", "i_l", "c_l", "Psi", "Rb_lmn", "Zb_lmn")
