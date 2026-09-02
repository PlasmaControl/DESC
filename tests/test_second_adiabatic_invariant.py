"""Tests for SecondAdiabaticInvariantAlphaDerivative and SoftConnectivity objectives.

References
----------
.. [1] Chen, H., Lu, Z., Xu, G., et al. (2026). "Direct Optimization
   of Stellarator Omnigenity from the Second Adiabatic Invariant."
   arXiv:2608.02418.
"""

from __future__ import annotations

import numpy as np
import pytest

from desc.backend import jnp
from desc.compute._omnigenity import (
    _smoothmax_logsumexp,
    _softplus_relu,
    _softplus_relu_sigmoid,
)
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    SecondAdiabaticInvariantAlphaDerivative,
    SoftConnectivity,
)
from desc.optimize import Optimizer


class TestSecondAdiabaticInvariant:
    """Test suite for direct second adiabatic invariant optimization."""

    @pytest.mark.unit
    def test_softplus_relu_and_sigmoid(self):
        """Test mathematical properties of the softplus smoothing kernel."""
        x = np.linspace(-5.0, 5.0, 100)
        beta = 50.0
        y = _softplus_relu(x, beta=beta)
        dy = _softplus_relu_sigmoid(x, beta=beta)

        # For positive x, softplus(x) ≈ x
        np.testing.assert_allclose(y[x > 0.5], x[x > 0.5], atol=1e-2)
        # For negative x, softplus(x) ≈ 0
        np.testing.assert_allclose(y[x < -0.5], 0.0, atol=1e-2)
        # Verify derivative bounds in [0, 1]
        assert np.all(dy >= 0.0) and np.all(dy <= 1.0)

    @pytest.mark.unit
    def test_smoothmax_logsumexp(self):
        """Test Log-Sum-Exp differentiable extrema approximation."""
        data = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, -1.0]])
        tau = 0.01
        smooth_max = _smoothmax_logsumexp(data, axis=1, tau=tau).squeeze()
        exact_max = np.max(data, axis=1)
        np.testing.assert_allclose(smooth_max, exact_max, atol=0.05)

    @pytest.mark.unit
    def test_second_adiabatic_invariant_objective_build_and_compute(self):
        """Test objective construction, compute, and residual evaluation."""
        eq = get("DSHAPE")
        obj = SecondAdiabaticInvariantAlphaDerivative(
            eq=eq,
            num_alpha=16,
            nzeta=32,
            M_booz=4,
            N_booz=0,
        )
        obj.build()
        residuals = obj.compute(eq.params_dict)

        assert np.all(np.isfinite(residuals))
        assert residuals.ndim == 1
        assert len(residuals) > 0

    @pytest.mark.unit
    def test_second_adiabatic_invariant_jacobian_and_grad(self):
        """Test gradient and Jacobian derivative actions of SecondAdiabaticInvariant."""
        eq = get("DSHAPE")
        obj_fun = ObjectiveFunction(
            SecondAdiabaticInvariantAlphaDerivative(
                eq=eq,
                num_alpha=8,
                nzeta=20,
                M_booz=2,
                N_booz=0,
            )
        )
        obj_fun.build()

        x = obj_fun.x()
        grad = obj_fun.grad(x)
        assert np.all(np.isfinite(grad))
        assert not np.any(np.isnan(grad))
        assert grad.shape == (obj_fun.dim_x,)

        jac = obj_fun.jac_scaled(x)
        assert np.all(np.isfinite(jac))
        assert not np.any(np.isnan(jac))
        assert jac.shape == (obj_fun.dim_f, obj_fun.dim_x)

        tangent = np.random.default_rng(42).normal(size=x.shape)
        jvp_action = obj_fun.jvp_scaled(tangent, x)
        assert np.all(np.isfinite(jvp_action))
        np.testing.assert_allclose(jvp_action, jac @ tangent, rtol=1e-8, atol=1e-8)

        cotangent = np.random.default_rng(43).normal(size=(obj_fun.dim_f,))
        vjp_action = obj_fun.vjp_scaled(cotangent, x)
        assert np.all(np.isfinite(vjp_action))
        np.testing.assert_allclose(vjp_action, jac.T @ cotangent, rtol=1e-8, atol=1e-8)

        # Verify gradient-Jacobian consistency: grad = J^T * f
        f = obj_fun.compute_scaled_error(x)
        expected_grad = jac.T @ f
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-8, atol=1e-8)

    @pytest.mark.unit
    def test_soft_connectivity_objective_build_and_compute(self):
        """Test SoftConnectivity objective build and evaluation."""
        from desc.magnetic_fields import SplineZeta

        eq = get("DSHAPE")
        spline = SplineZeta(n_control=4, NFP=eq.NFP)
        obj = SoftConnectivity(
            eq=eq,
            spline=spline,
            num_alpha=12,
            M_booz=4,
            N_booz=0,
        )
        obj.build()
        residuals = obj.compute(eq.params_dict, spline.params_dict)

        assert np.all(np.isfinite(residuals))
        assert residuals.ndim == 1
        assert len(residuals) > 0

    @pytest.mark.unit
    def test_spline_zeta_reduced_knots_and_symmetry(self):
        """Test SplineZeta parameter extraction and symmetry behavior."""
        from desc.magnetic_fields import SplineZeta

        spline_sym = SplineZeta(n_control=4, NFP=2, symmetry=True)
        assert spline_sym.symmetry is True
        knots_sym = spline_sym.reduced_knots()
        assert knots_sym["reduced_alpha_knots"].size == 4
        assert knots_sym["zeta_min_knots"].size == 4
        assert knots_sym["zeta_max_knots"].size == 4
        assert spline_sym.alpha_knots_full.size == 8

        spline_asym = SplineZeta(n_control=4, NFP=2, symmetry=False)
        assert spline_asym.symmetry is False
        knots_asym = spline_asym.reduced_knots()
        assert knots_asym["reduced_alpha_knots"].size == 4
        assert spline_asym.alpha_knots_full.size == 4

    @pytest.mark.unit
    def test_single_step_optimization(self):
        """Test single-step equilibrium optimization using J* objective."""
        from desc.objectives import ForceBalance, get_fixed_boundary_constraints

        eq = get("DSHAPE")
        objective = ObjectiveFunction(
            SecondAdiabaticInvariantAlphaDerivative(
                eq=eq,
                num_alpha=8,
                nzeta=20,
                M_booz=2,
                N_booz=0,
            )
        )
        constraints = (ForceBalance(eq), *get_fixed_boundary_constraints(eq))
        optimizer = Optimizer("proximal-lsq-exact")
        eq_opt, result = eq.optimize(
            objective=objective,
            constraints=constraints,
            optimizer=optimizer,
            maxiter=1,
            verbose=0,
        )
        assert result.success or result.nfev >= 1
        assert np.all(np.isfinite(eq_opt.R_lmn))
        assert np.all(np.isfinite(eq_opt.Z_lmn))

    @pytest.mark.unit
    def test_soft_connectivity_joint_objective_optimization(self):
        """Test joint optimization with eq and SplineZeta."""
        from desc.magnetic_fields import SplineZeta
        from desc.objectives import (
            FixParameters,
            ForceBalance,
            get_fixed_boundary_constraints,
        )

        eq = get("DSHAPE")
        spline = SplineZeta(n_control=4, NFP=eq.NFP)
        objective = ObjectiveFunction(
            SoftConnectivity(
                eq=eq,
                spline=spline,
                num_alpha=8,
                M_booz=2,
                N_booz=0,
            )
        )
        constraints = (
            ForceBalance(eq),
            *get_fixed_boundary_constraints(eq),
            FixParameters(spline, {"zeta_max_knots": True}),
        )
        optimizer = Optimizer("proximal-lsq-exact")
        (eq_opt, spline_opt), result = optimizer.optimize(
            things=(eq, spline),
            objective=objective,
            constraints=constraints,
            maxiter=1,
            verbose=0,
        )
        assert result.success or result.nfev >= 1
        assert np.all(np.isfinite(eq_opt.R_lmn))
        assert np.all(np.isfinite(spline_opt.zeta_min_knots))
