"""Objectives for ITG turbulence optimization."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import setdefault

from .objective_funs import _Objective, collect_docs


class ITGProxy(_Objective):
    """ITG turbulence proxy from Landreman et al. 2025.

    This objective computes the analytical proxy for Ion Temperature Gradient
    (ITG) turbulence heat flux. The proxy is defined as:

    f_Q = mean((sigmoid(cvdrift) + 0.2) * |grad_x|^3 / B)

    Lower values indicate configurations with reduced ITG turbulence.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to be optimized.
    rho : float or array-like
        Flux surface(s) to evaluate on.
    alpha : float or array-like, optional
        Field line labels to evaluate. Default is 8 field lines in [0, pi]
        for non-axisymmetric cases, or alpha=0 for axisymmetric.
    nturns : int, optional
        Number of toroidal transits per field line. Default is 3.
    nzetaperturn : int, optional
        Number of points per toroidal transit. Default is 100.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_iota_keys",
        "_nturns",
        "_nzetaperturn",
        "_add_lcfs",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "ITG Proxy f_Q: "

    def __init__(
        self,
        eq,
        rho,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        alpha=None,
        nturns=3,
        nzetaperturn=100,
        name="ITG Proxy",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        self._nturns = nturns
        self._nzetaperturn = nzetaperturn
        self._rho = np.atleast_1d(rho)
        self._add_lcfs = np.all(self._rho < 0.97)
        self._alpha = setdefault(
            alpha,
            (
                jnp.linspace(0, (1 + eq.sym) * jnp.pi, (1 + eq.sym) * 8)
                if eq.N
                else jnp.array([0])
            ),
        )

        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        self._iota_keys = ["iota", "iota_r", "shear", "a"]

        eq = self.things[0]
        iota_grid = LinearGrid(
            rho=np.append(self._rho, 1) if self._add_lcfs else self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
        )
        assert not iota_grid.axis.size
        self._dim_f = iota_grid.num_rho - self._add_lcfs
        transforms = get_transforms(self._iota_keys, eq, iota_grid)
        profiles = get_profiles(
            self._iota_keys + ["ITG proxy integrand"], eq, iota_grid
        )
        self._constants = {
            "rho": self._rho,
            "alpha": self._alpha,
            "zeta": jnp.linspace(
                -self._nturns * jnp.pi,
                +self._nturns * jnp.pi,
                +self._nturns * self._nzetaperturn,
            ),
            "iota_transforms": transforms,
            "profiles": profiles,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the ITG turbulence proxy.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom.
        constants : dict
            Dictionary of constant data. Defaults to self.constants.

        Returns
        -------
        f_Q : ndarray
            ITG proxy values for each flux surface.

        """
        if constants is None:
            constants = self.constants
        eq = self.things[0]

        # Compute iota on the iota grid
        iota_data = compute_fun(
            eq,
            self._iota_keys,
            params,
            constants["iota_transforms"],
            constants["profiles"],
        )
        iota_grid = constants["iota_transforms"]["grid"]

        def get(key):
            x = iota_grid.compress(iota_data[key])
            return x[:-1] if self._add_lcfs else x

        iota = get("iota")

        # Create field-aligned grid
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=iota,
            params=params,
        )

        # Prepare data for the compute function
        data = {
            key: grid.expand(get(key))
            for key in self._iota_keys
            if (key != "iota" and key != "a")
        }
        data["iota"] = grid.expand(iota)
        data["a"] = iota_data["a"]

        # Compute the ITG proxy integrand
        data = compute_fun(
            eq,
            ["ITG proxy integrand"],
            params,
            transforms=get_transforms(["ITG proxy integrand"], eq, grid, jitable=True),
            profiles=constants["profiles"],
            data=data,
        )

        # Average over field line to get per-flux-surface values
        integrand = data["ITG proxy integrand"]
        num_rho = len(constants["rho"])
        num_alpha = len(constants["alpha"])
        num_zeta = len(constants["zeta"])

        integrand_reshaped = integrand.reshape(num_zeta, num_rho, num_alpha)
        f_Q = jnp.mean(integrand_reshaped, axis=(0, 2))

        return f_Q
