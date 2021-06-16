import numpy as np

from desc.equilibrium import Equilibrium


def test_magnetic_axis_guess(DummyStellarator):
    """Test that the magnetic axis initial guess is used correctly."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )
    zeta = np.linspace(0, 2 * np.pi, num=33, endpoint=False)

    # axis guess for Dummy Stellarator:
    R0 = 3.4 + 0.2 * np.cos(eq.NFP * zeta)
    Z0 = -0.2 * np.sin(eq.NFP * zeta)

    # axis location as input
    R0_eq, Z0_eq = eq.compute_axis_location(zeta)

    np.testing.assert_allclose(R0_eq, R0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(Z0_eq, Z0, rtol=0, atol=1e-6)
