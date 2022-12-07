"""Functions for computing field and plasma quantities from an equilibrium.

All compute functions take the following arguments:

Parameters
----------
params : dict of ndarray
    Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
transforms : dict of Transform
    Transforms for R, Z, lambda, etc
profiles : dict of Profile
    Profile objects for pressure, iota, current, etc
data : dict of ndarray
    Data computed so far, generally output from other compute functions
kwargs : dict
    Other arguments needed by some functions, such as helicity

Returns
-------
data : dict of ndarray
    Dictionary of ndarray, shape(num_nodes,) of computed quantities.
    Keys are of the form 'X_y' meaning the derivative of X wrt y.

"""

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
    compute_pressure_gradient,
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
from .utils import get_data_deps, get_derivs, get_params, get_profiles, get_transforms

# defines the order in which objective arguments get concatenated into the state vector
arg_order = ("R_lmn", "Z_lmn", "L_lmn", "p_l", "i_l", "c_l", "Psi", "Rb_lmn", "Zb_lmn")


def _build_data_index():
    for key in data_index.keys():
        full = {}
        full["data"] = get_data_deps(key)
        full["transforms"] = get_derivs(key)
        full["params"] = get_params(key)
        full["profiles"] = get_profiles(key)
        data_index[key]["full_dependencies"] = full


_build_data_index()
