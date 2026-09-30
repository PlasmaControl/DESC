"""Tests for desc.external.quadcoil."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.examples import get
from desc.external.quadcoil import QuadcoilProxy
from desc.magnetic_fields import solve_regularized_surface_current
from desc.objectives import QuadraticFlux, SurfaceCurrentRegularization


@pytest.mark.unit
def test_quadcoil():
    """Test the QUADCOIL proxy."""
    # Setting testing thresholds
    # 5% normalized error in Phi Fourier coefficients
    thres_phi = 0.01
    thres_G = 0.001  # to be decreased

    # ----- Test 1: NESCOIL, value only -----
    def run_regcoil(vacuum):
        # Loading equilibrium and resolutions
        # ARIES-CS
        if not vacuum:
            quadcoil_test_eq = get("ARIES-CS")
        else:
            quadcoil_test_eq = get("precise_QA")

        # Settings
        mpol = 8  # Num. poloidal modes in the current potential
        ntor = 8  # Num. toroidal modes in the current potential
        # Controls the resolution of the plasma surface integration.
        # Integration in quadcoil is naively performed using summation
        # so we recommend at least 16 here.
        # This corresponds to a (33 x 33) grid.
        minor_radius = quadcoil_test_eq.compute("a")["a"]
        plasma_coil_distance = minor_radius * 0.75
        # Resolution for sampling objectives
        quadpoints_phi = jnp.linspace(0, 1 / quadcoil_test_eq.NFP, 33, endpoint=False)
        quadpoints_theta = jnp.linspace(0, 1, 33, endpoint=False)
        quadcoil_kwargs_basic = {
            "mpol": mpol,
            "ntor": ntor,
            "quadpoints_phi": quadpoints_phi,
            "quadpoints_theta": quadpoints_theta,
            "plasma_coil_distance": plasma_coil_distance,
        }
        quadcoil_kwargs_nescoil = quadcoil_kwargs_basic | {
            "objective_name": "f_B",  # The NESCOIL problem only contains
            # The NESCOIL problem is simple enough to need no normalization
            # constants. In the next example we will discuss how to choose
            # this constant.
            "objective_unit": None,
        }

        # Define a QuadcoilProxy with the simplest possible
        # signature
        objective_nescoil = QuadcoilProxy(
            eq=quadcoil_test_eq,
            quadcoil_kwargs=quadcoil_kwargs_nescoil,
            vacuum=vacuum,
        )
        objective_nescoil.build()
        scf_nescoil = objective_nescoil.solve_quadcoil_surface_current(
            # Like Objective.compute(), the xs of the equilibrium
            # must be passed in as the *arg.
            *objective_nescoil.xs(quadcoil_test_eq)
        )
        source_grid = objective_nescoil._constants["source_grid"]
        eval_grid = objective_nescoil._constants["eval_grid"]
        fields, data = solve_regularized_surface_current(
            scf_nescoil,
            eq=quadcoil_test_eq,
            source_grid=source_grid,
            eval_grid=eval_grid,
            current_helicity=(
                1,
                0,
            ),
            lambda_regularization=jnp.array([0]),
            vacuum=vacuum,
            regularization_type="regcoil",
            chunk_size=40,
        )
        # Comparing Phi coefficients
        phi1 = fields[0].compute("Phi")["Phi"]
        phi2 = scf_nescoil.compute("Phi")["Phi"]
        norm_error = jnp.average(jnp.abs(phi1 - phi2)) / jnp.max(jnp.abs(phi1))
        assert norm_error <= thres_phi
        # Testing net poloidal current
        assert np.abs(scf_nescoil.G - fields[0].G) <= np.abs(thres_G * fields[0].G)
        # Solving the NESCOIL problem using DESC's built in REGCOIL.
        # Please see the REGCOIL-like coil optimization tutorial
        fields_regcoil, data_regcoil = solve_regularized_surface_current(
            scf_nescoil,
            eq=quadcoil_test_eq,
            source_grid=source_grid,
            eval_grid=eval_grid,
            current_helicity=(
                1,
                0,
            ),
            lambda_regularization=jnp.array([1e-15]),
            vacuum=vacuum,
            regularization_type="regcoil",
            chunk_size=40,
        )
        obj_flux = QuadraticFlux(
            field=fields[0],
            eq=quadcoil_test_eq,
            eval_grid=eval_grid,
            field_grid=source_grid,
            vacuum=False,
            bs_chunk_size=10,
            normalize=False,
            normalize_target=False,
        )
        obj_flux.build()
        chi_2_B_regcoil = obj_flux.compute_scalar(*obj_flux.xs(fields_regcoil[0]))
        obj_regularization = SurfaceCurrentRegularization(
            surface_current_field=fields[0],
            source_grid=source_grid,
            weight=1,
            normalize=False,
            # don't use normalizations, to match the REGCOIL problem exactly
            normalize_target=False,
        )  # we will set it to the sqrt of optimal weight from above,
        # since DESC will square the weight when making the overall cost fxn
        obj_regularization.build()
        chi_2_K_regcoil = obj_regularization.compute_scalar(
            *obj_regularization.xs(fields_regcoil[0])
        )
        C_B = chi_2_B_regcoil / (4 * jnp.pi**2)
        C_K = chi_2_K_regcoil / (4 * jnp.pi**2)
        quadcoil_kwargs_regcoil = quadcoil_kwargs_basic | {
            "objective_name": "f_K",
            "objective_unit": C_K,
            "constraint_name": ("f_B",),
            "constraint_type": ("<=",),
            "constraint_value": jnp.array([C_B]),
            "constraint_unit": jnp.array([C_B]),
        }
        # Define a QuadcoilProxy with the simplest possible
        # signature.
        objective_regcoil = QuadcoilProxy(
            eq=quadcoil_test_eq,
            quadcoil_kwargs=quadcoil_kwargs_regcoil,
            vacuum=vacuum,
        )
        objective_regcoil.build()
        scf_regcoil = objective_regcoil.solve_quadcoil_surface_current(
            # Like Objective.compute(), the xs of the equilibrium
            # must be passed in as the *arg.
            *objective_regcoil.xs(quadcoil_test_eq)
        )
        phi1_regcoil = fields_regcoil[0].compute("Phi")["Phi"]
        phi2_regcoil = scf_regcoil.compute("Phi")["Phi"]
        norm_error = jnp.average(jnp.abs(phi1_regcoil - phi2_regcoil)) / jnp.max(
            jnp.abs(phi1_regcoil)
        )
        assert norm_error <= thres_phi

    run_regcoil(True)
    run_regcoil(False)
