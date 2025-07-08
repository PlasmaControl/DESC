from functools import partial

from desc.backend import jit
from desc.grid import LinearGrid
from desc.integrals.singularities import (
    _dx,
    _kernel_biot_savart,
    _kernel_biot_savart_A,
    get_interpolator,
    singular_integral,
)
from desc.utils import dot


@partial(jit, static_argnames=["chunk_size", "loop"])
def virtual_casing_biot_savart(
    eval_data, source_data, interpolator, chunk_size=None, **kwargs
):
    """Evaluate magnetic field on surface due to sheet current on surface.

    The magnetic field due to the plasma current can be written as a Biot-Savart
    integral over the plasma volume:

    𝐁ᵥ(𝐫) = μ₀/4π ∫ 𝐉(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d³𝐫'

    Where 𝐉 is the plasma current density, 𝐫 is a point on the plasma surface, and 𝐫' is
    a point in the plasma volume.

    This 3D integral can be converted to a 2D integral over the plasma boundary using
    the virtual casing principle [1]_

    𝐁ᵥ(𝐫) = μ₀/4π ∫ (𝐧' ⋅ 𝐁(𝐫')) * (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + μ₀/4π ∫ (𝐧' × 𝐁(𝐫')) × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + 𝐁(𝐫)/2

    Where 𝐁 is the total field on the surface and 𝐧' is the outward surface normal.
    Because the total field is tangent, the first term in the integrand is zero leaving

    𝐁ᵥ(𝐫) = μ₀/4π ∫ K_vc(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫' + 𝐁(𝐫)/2

    Where we have defined the virtual casing sheet current K_vc = 𝐧' × 𝐁(𝐫')

    References
    ----------
       [1] Hanson, James D. "The virtual-casing principle and Helmholtz’s theorem."
       Plasma Physics and Controlled Fusion 57.11 (2015): 115006.

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points (eval_grid passed to interpolator).
        Keys should be those required by kernel as kernel.keys. Vector data should be
        in rpz basis.
    source_data : dict
        Dictionary of data at source points (source_grid passed to interpolator). Keys
        should be those required by kernel as kernel.keys. Vector data should be in
        rpz basis.
    interpolator : _BIESTInterpolator
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator`` or
        ``DFTInterpolator``
    chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, kernel.ndim)
        Integral transform evaluated at eval_grid. Vectors are in rpz basis.

    """
    return singular_integral(
        eval_data,
        source_data,
        interpolator=interpolator,
        kernel=_kernel_biot_savart,
        chunk_size=chunk_size,
        **kwargs,
    )


def compute_B_plasma(
    eq, eval_grid, source_grid=None, normal_only=False, chunk_size=None
):
    """Evaluate magnetic field on surface due to enclosed plasma currents.

    The magnetic field due to the plasma current can be written as a Biot-Savart
    integral over the plasma volume:

    𝐁ᵥ(𝐫) = μ₀/4π ∫ 𝐉(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d³𝐫'

    Where 𝐉 is the plasma current density, 𝐫 is a point on the plasma surface, and 𝐫' is
    a point in the plasma volume.

    This 3D integral can be converted to a 2D integral over the plasma boundary using
    the virtual casing principle [1]_

    𝐁ᵥ(𝐫) = μ₀/4π ∫ (𝐧' ⋅ 𝐁(𝐫')) * (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + μ₀/4π ∫ (𝐧' × 𝐁(𝐫')) × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫'
          + 𝐁(𝐫)/2

    Where 𝐁 is the total field on the surface and 𝐧' is the outward surface normal.
    Because the total field is tangent, the first term in the integrand is zero leaving

    𝐁ᵥ(𝐫) = μ₀/4π ∫ K_vc(𝐫') × (𝐫 − 𝐫')/|𝐫 − 𝐫'|³ d²𝐫' + 𝐁(𝐫)/2

    Where we have defined the virtual casing sheet current K_vc = 𝐧' × 𝐁(𝐫')

    References
    ----------
       [1] Hanson, James D. "The virtual-casing principle and Helmholtz’s theorem."
       Plasma Physics and Controlled Fusion 57.11 (2015): 115006.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that is the source of the plasma current.
    eval_grid : Grid
        Evaluation points for the magnetic field.
    source_grid : Grid, optional
        Source points for integral.
    normal_only : bool
        If True, only compute and return the normal component of the plasma field 𝐁ᵥ⋅𝐧
    chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, 3) or shape(eval_grid.num_nodes,)
        Magnetic field evaluated at eval_grid.
        If normal_only=False, vector B is in rpz basis.

    """
    if source_grid is None:
        source_grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )

    eval_data = eq.compute(_dx.keys + ["B", "n_rho"], grid=eval_grid)
    source_data = eq.compute(
        _kernel_biot_savart_A.keys + ["|e_theta x e_zeta|"], grid=source_grid
    )
    if hasattr(eq.surface, "Phi_mn"):
        source_data = eq.surface.compute("K", grid=source_grid, data=source_data)
        source_data["K_vc"] += source_data["K"]

    interpolator = get_interpolator(eval_grid, source_grid, source_data)
    Bplasma = virtual_casing_biot_savart(
        eval_data, source_data, interpolator, chunk_size
    )
    # need extra factor of B/2 bc we're evaluating on plasma surface
    Bplasma += eval_data["B"] / 2
    if normal_only:
        Bplasma = dot(Bplasma, eval_data["n_rho"])
    return Bplasma
