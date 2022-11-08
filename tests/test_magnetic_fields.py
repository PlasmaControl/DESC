"""Tests for magnetic field classes."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.magnetic_fields import (
    PoloidalMagneticField,
    ScalarPotentialField,
    SplineMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
    field_line_integrate,
)


def phi_lm(R, phi, Z, a, m):
    """Scalar potential test function."""
    CNm0 = (R**m - R**-m) / (2 * m)
    Nm1 = CNm0 * Z
    CDm0 = (R**m + R**-m) / 2
    c1 = -m * (m - 1)
    c2 = (m + 1) * (m - 2)
    c3 = m * (m + 1)
    c4 = -(m + 2) * (m - 1)
    CDm1 = (c1 * R ** (m + 2) + c2 * R ** (m) + c3 * R ** (-m + 2) + c4 * R ** (-m)) / (
        8 * m * (m**2 - 1)
    )
    Dm2 = CDm0 * Z**2 / 2 + CDm1
    return phi + a * Dm2 * jnp.sin(m * phi) + a * Nm1 * jnp.cos(m * phi)


a = -1.489
m = 5
args = {"a": a, "m": m}


class TestMagneticFields:
    """Tests for MagneticField classes."""

    @pytest.mark.unit
    def test_basic_fields(self):
        """Tests for basic field types (toroidal, vertical, poloidal)."""
        tfield = ToroidalMagneticField(2, 1)
        vfield = VerticalMagneticField(1)
        pfield = PoloidalMagneticField(2, 1, 2)
        np.testing.assert_allclose(tfield([1, 0, 0]), [[0, 2, 0]])
        np.testing.assert_allclose((4 * tfield)([2, 0, 0]), [[0, 4, 0]])
        np.testing.assert_allclose((tfield + vfield)([1, 0, 0]), [[0, 2, 1]])
        np.testing.assert_allclose(
            (tfield + vfield - pfield)([1, 0, 0.1]), [[0.4, 2, 1]]
        )

    @pytest.mark.unit
    def test_scalar_field(self):
        """Test scalar potential magnetic field against analytic result."""
        field = ScalarPotentialField(phi_lm, args)
        np.testing.assert_allclose(
            field.compute_magnetic_field([1.0, 0, 0]), [[0, 1, 0]]
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([1.0, np.pi / 4, 0]), [[0, 1, 0]]
        )

    @pytest.mark.slow
    @pytest.mark.unit
    def test_spline_field(self):
        """Test accuracy of spline magnetic field."""
        field1 = ScalarPotentialField(phi_lm, args)
        R = np.linspace(0.5, 1.5, 20)
        Z = np.linspace(-1.5, 1.5, 20)
        p = np.linspace(0, 2 * np.pi / 5, 40)
        field2 = SplineMagneticField.from_field(field1, R, p, Z, period=2 * np.pi / 5)

        np.testing.assert_allclose(
            field1([1.0, 1.0, 1.0]), field2([1.0, 1.0, 1.0]), rtol=1e-2, atol=1e-2
        )

        extcur = [4700.0, 1000.0]
        mgrid = "tests/inputs/mgrid_test.nc"
        field3 = SplineMagneticField.from_mgrid(mgrid, extcur)

        np.testing.assert_allclose(
            field3([0.70, 0, 0]), [[0, -0.671, 0.0858]], rtol=1e-3, atol=1e-8
        )

    @pytest.mark.unit
    def test_spline_field_axisym(self):
        """Test computing axisymmetric magnetic field using SplineMagneticField."""
        extcur = [
            -1.370985e03,
            -1.609154e03,
            -2.751331e03,
            -2.524384e03,
            -3.435372e03,
            -3.466123e03,
            3.670919e03,
            3.450196e03,
            2.908027e03,
            3.404695e03,
            -4.148967e03,
            -4.294406e03,
            -3.059939e03,
            -2.990609e03,
            3.903818e03,
            3.727301e03,
            -3.049484e03,
            -3.086940e03,
            -1.488703e07,
            -2.430716e04,
            -2.380229e04,
        ]
        field = SplineMagneticField.from_mgrid(
            "tests/inputs/mgrid_d3d.nc", extcur=extcur
        )
        # make sure field is invariant to shift in phi
        B1 = field.compute_magnetic_field(np.array([1.75, 0.0, 0.0]))
        B2 = field.compute_magnetic_field(np.array([1.75, 1.0, 0.0]))
        np.testing.assert_allclose(B1, B2)

    @pytest.mark.unit
    def test_field_line_integrate(self):
        """Test field line integration."""
        # q=4, field line should rotate 1/4 turn after 1 toroidal transit
        # from outboard midplane to top center
        field = ToroidalMagneticField(2, 10) + PoloidalMagneticField(2, 10, 0.25)
        r0 = [10.001]
        z0 = [0.0]
        phis = [0, 2 * np.pi]
        r, z = field_line_integrate(r0, z0, phis, field)
        np.testing.assert_allclose(r[-1], 10, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z[-1], 0.001, rtol=1e-6, atol=1e-6)
