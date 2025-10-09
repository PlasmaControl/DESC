"""Various checks to automate installation issues and decrease user support requests.

These tests run when DESC boots to notify users of potential issues
or to replace warning messages in downstream libraries with more
verbose warnings.
"""

import warnings

import numpy as np
from jax import grad

from desc.backend import jnp, rfft
from desc.integrals._interp_utils import nufft1d2r


def _c_1d(x):
    return jnp.cos(7 * x) + jnp.sin(x) - 33.2


@grad
def _true_g_c_1d(xq):
    return _c_1d(xq).sum()


def check_jax_finufft(func=_c_1d, g_func=_true_g_c_1d):
    """Runs tests/test_interp_utils.py::TestFastInterp::test_non_uniform_real_FFT."""
    n = 15
    f = 2 * rfft(func(jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)), norm="forward")
    f = f.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)
    xq = jnp.array([7.34, 1.10134, 2.28])

    msg = (
        "If you want to use NUFFTs, follow the DESC installation instructions.\n"
        "Otherwise you must pass the parameter nufft_eps=0.\n"
        "This applies to effective ripple, Gamma_c, and any other\n"
        "computations that involve bounce integrals.\n"
    )
    # https://github.com/unalmis/jax-finufft/blob/main/tests/interpolation_test.py#L13
    RTOL = 2e-6

    try:
        np.testing.assert_allclose(nufft1d2r(xq, f), func(xq), rtol=RTOL)

        @grad
        def g(xq):
            return nufft1d2r(xq, f, eps=1e-7).sum()

        np.testing.assert_allclose(g(xq), g_func(xq), rtol=RTOL)
    except NameError:
        warnings.warn("\njax-finufft is not installed.\n" + msg)
    except NotImplementedError:
        warnings.warn("\njax-finufft is not installed on GPU.\n" + msg)
    except AssertionError as e:
        import jax_finufft

        warnings.warn(
            f"\njax-finufft version <= {jax_finufft.__version__} has incorrect maths.\n"
            + "\n\n"
            + e
            + "\n\n"
            + msg
        )
