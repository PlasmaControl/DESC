"""Tests for magnetic field classes."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.magnetic_fields import (
    CD_m_k,
    CN_m_k,
    DommaschkPotentialField,
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


@pytest.mark.unit
def test_dommaschk_CN_CD_m_0():
    """Test of CD_m_k and CN_m_k when k=0."""
    # based off eqn 8 and 9 of Dommaschk paper
    # https://doi.org/10.1016/0010-4655(86)90109-8
    for m in range(1, 6):
        # test of CD_m_k based off eqn 8
        R = np.linspace(0.1, 1, 100)
        res1 = CD_m_k(R, m, 0)
        res2 = 0.5 * (R**m + R ** (-m))
        np.testing.assert_allclose(res1, res2)

        # test of CN_m_k based off eqn 9
        res1 = CN_m_k(R, m, 0)
        res2 = 0.5 * (R**m - R ** (-m)) / m
        np.testing.assert_allclose(res1, res2, atol=1e-15)


@pytest.mark.unit
def test_dommaschk_field_errors():
    """Test the assert statements of the DommaschkField function."""
    ms = [1]
    ls = [1]
    a_arr = [1]
    b_arr = [1]
    c_arr = [1]
    d_arr = [1, 1]  # length is not equal to the rest
    with pytest.raises(AssertionError):
        DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)
    d_arr = [1]
    ms = [-1]  # negative modenumber
    with pytest.raises(AssertionError):
        DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)


@pytest.mark.unit
def test_dommaschk_radial_field():
    """Test the Dommaschk potential for a pure toroidal (Bphi~1/R) field."""
    phi = np.linspace(0, 2 * np.pi, 10)
    R = np.linspace(0.1, 1.5, 50)
    Z = np.linspace(-0.05, 0.5, 50)
    R, phi, Z = np.meshgrid(R, phi, Z)
    coords = np.vstack((R.flatten(), phi.flatten(), Z.flatten())).T

    ms = [0]
    ls = [0]
    a_arr = [0]
    b_arr = [0]
    c_arr = [0]
    d_arr = [0]
    B = DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)
    B_dom = B.compute_magnetic_field(coords)
    np.testing.assert_allclose(B_dom[:, 0], 0)
    np.testing.assert_array_equal(B_dom[:, 1], 1 / R.flatten())
    np.testing.assert_allclose(B_dom[:, 2], 0)


@pytest.mark.unit
def test_dommaschk_vertical_field():
    """Test the Dommaschk potential for a 1/R toroidal + pure vertical field."""
    phi = jnp.linspace(0, 2 * jnp.pi, 10)
    R = jnp.linspace(0.1, 1.5, 50)
    Z = jnp.linspace(-0.5, 0.5, 50)
    R, phi, Z = jnp.meshgrid(R, phi, Z)
    coords = jnp.vstack((R.flatten(), phi.flatten(), Z.flatten())).T

    ms = [0]
    ls = [1]
    a_arr = [1]
    b_arr = [0]
    c_arr = [0]
    d_arr = [0]
    B = DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)
    B_dom = B.compute_magnetic_field(coords)
    ones = jnp.ones_like(B_dom[:, 0])
    jnp.testing.assert_allclose(B_dom[:, 0], 0, atol=1e-14)
    jnp.testing.assert_allclose(B_dom[:, 1], 1 / R.flatten(), atol=1e-14)
    jnp.testing.assert_allclose(B_dom[:, 2], ones, atol=5e-15)


@pytest.mark.unit
def test_dommaschk_fit_toroidal_field():
    """Test the Dommaschk potential fit for a 1/R toroidal scaled to 2 T."""
    phi = np.linspace(0, 2 * jnp.pi, 3)
    R = np.linspace(0.1, 1.5, 3)
    Z = np.linspace(-0.5, 0.5, 3)
    R, phi, Z = np.meshgrid(R, phi, Z)
    coords = np.vstack((R.flatten(), phi.flatten(), Z.flatten())).T

    max_l = 1
    max_m = 1
    B0 = 2  # scale strength for to 1/R field to fit

    def B0_over_R(coord):
        B_R = np.zeros_like(coord[:, 0])
        B_phi = B0 / coord[:, 0]
        B_Z = np.zeros_like(coord[:, 0])
        return np.vstack((B_R, B_phi, B_Z)).T

    B = DommaschkPotentialField.fit_magnetic_field(B0_over_R, coords, max_m, max_l)

    B_dom = B.compute_magnetic_field(coords)
    np.testing.assert_allclose(B_dom[:, 0], 0, atol=4e-15)
    np.testing.assert_allclose(B_dom[:, 1], B0 / R.flatten(), atol=1e-15)
    np.testing.assert_allclose(B_dom[:, 2], jnp.zeros_like(R.flatten()), atol=1e-15)

    # only nonzero coefficient of the field should be the B0
    np.testing.assert_allclose(B._params["B0"], B0, atol=1e-15)
    for coef in ["a_arr", "b_arr", "c_arr", "d_arr"]:
        np.testing.assert_allclose(B._params[coef], 0, atol=1e-15)


@pytest.mark.unit
def test_dommaschk_fit_vertical_and_toroidal_field():
    """Test the Dommaschk potential fit for a toroidal and a vertical field."""
    phi = np.linspace(0, 2 * np.pi, 3)
    R = np.linspace(0.1, 1.5, 3)
    Z = np.linspace(-0.5, 0.5, 3)
    R, phi, Z = np.meshgrid(R, phi, Z)
    coords = np.vstack((R.flatten(), phi.flatten(), Z.flatten())).T

    max_l = 1
    max_m = 1
    B0 = 2  # scale strength for to 1/R field to fit
    B0_Z = 1  # scale strength for to uniform vertical field to fit
    field = ToroidalMagneticField(B0=B0, R0=1) + VerticalMagneticField(B0=B0_Z)

    B = DommaschkPotentialField.fit_magnetic_field(field, coords, max_m, max_l)

    B_dom = B.compute_magnetic_field(coords)
    np.testing.assert_allclose(B_dom[:, 0], 0, atol=4e-15)
    np.testing.assert_allclose(B_dom[:, 1], B0 / R.flatten(), atol=1e-15)
    np.testing.assert_allclose(B_dom[:, 2], B0_Z, atol=1e-15)

    np.testing.assert_allclose(B._params["B0"], B0)

    # only nonzero coefficient of the field should be the B0 and a_ml = a_01
    np.testing.assert_allclose(B._params["B0"], B0, atol=1e-15)
    for coef, m, l in zip(B._params["a_arr"], B._params["ms"], B._params["ls"]):
        if m == 0 and l == 1:
            np.testing.assert_allclose(coef, B0_Z)
        else:
            np.testing.assert_allclose(coef, 0, atol=1e-15)
    for name in ["b_arr", "c_arr", "d_arr"]:
        np.testing.assert_allclose(B._params[name], 0, atol=1e-15)
