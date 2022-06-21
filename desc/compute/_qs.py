"""Compute functions for quasisymmetry objectives, ie Boozer, Two-Term, and Triple Product."""

from desc.backend import jnp, put, sign

from ._core import check_derivs
from ._field import (
    compute_magnetic_field_magnitude,
    compute_covariant_magnetic_field,
    compute_B_dot_gradB,
)


def compute_boozer_coords(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    B_transform,
    w_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute Boozer coordinates.

    Assumes transform grids are uniform spacing on single flux surface without symmetry.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface, in Webers.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    B_transform : Transform
        Transforms spectral coefficients of B(rho,theta,zeta) to real space.
        B_transform.basis should be of type DoubleFourierSeries.
    w_transform : Transform
        Transforms spectral coefficients of w(rho,theta,zeta) to real space.
        w_transform.basis should be of type DoubleFourierSeries.
    iota : Profile
        Transforms i_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Boozer harmonics.

    """
    data = compute_magnetic_field_magnitude(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data=data,
    )
    # TODO: can remove this call if compute_|B| changed to use B_covariant
    data = compute_covariant_magnetic_field(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data=data,
    )

    NFP = w_transform.basis.NFP
    if not check_derivs("nu", R_transform, Z_transform, L_transform):
        return data

    # covariant Boozer components: I = B_theta, G = B_zeta (in Boozer coordinates)
    idx0 = B_transform.basis.get_idx(M=0, N=0)
    B_theta_mn = B_transform.fit(data["B_theta"])
    B_zeta_mn = B_transform.fit(data["B_zeta"])
    data["I"] = B_theta_mn[idx0]
    data["G"] = B_zeta_mn[idx0]

    # w (RHS of eq 10 in Hirshman 1995 "Transformation from VMEC to Boozer Coordinates")
    w_mn = jnp.zeros((w_transform.basis.num_modes,))
    for k, (l, m, n) in enumerate(w_transform.basis.modes):
        if m != 0:
            idx = B_transform.basis.get_idx(M=-m, N=n)
            w_mn = put(w_mn, k, (sign(n) * B_theta_mn[idx] / jnp.abs(m))[0])
        elif n != 0:
            idx = B_transform.basis.get_idx(M=m, N=-n)
            w_mn = put(w_mn, k, (sign(m) * B_zeta_mn[idx] / jnp.abs(NFP * n))[0])

    # transform to real space
    w = w_transform.transform(w_mn)
    w_t = w_transform.transform(w_mn, dr=0, dt=1, dz=0)
    w_z = w_transform.transform(w_mn, dr=0, dt=0, dz=1)

    # nu = zeta_Boozer - zeta
    GI = data["G"] + data["iota"] * data["I"]
    data["nu"] = (w - data["I"] * data["lambda"]) / GI
    data["nu_t"] = (w_t - data["I"] * data["lambda_t"]) / GI
    data["nu_z"] = (w_z - data["I"] * data["lambda_z"]) / GI

    # Boozer angles
    data["theta_B"] = data["theta"] + data["lambda"] + data["iota"] * data["nu"]
    data["zeta_B"] = data["zeta"] + data["nu"]

    # Jacobian of Boozer coordinates wrt (theta,zeta) coordinates
    data["sqrt(g)_B"] = (1 + data["lambda_t"]) * (1 + data["nu_z"]) + (
        data["iota"] - data["lambda_z"]
    ) * data["nu_t"]

    # Riemann sum integration
    nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
    norm = 2 ** (3 - jnp.sum((B_transform.basis.modes == 0), axis=1))
    data["|B|_mn"] = (
        norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
        * jnp.matmul(
            B_transform.basis.evaluate(nodes).T, data["sqrt(g)_B"] * data["|B|"]
        )
        / B_transform.grid.num_nodes
    )
    data["B modes"] = B_transform.basis.modes

    return data


def compute_quasisymmetry_error(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    helicity=(1, 0),
    data=None,
    **kwargs,
):
    """Compute quasi-symmetry triple product and two-term errors.

    f_C computation assumes transform grids are a single flux surface.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface, in Webers.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    helicity : tuple, int
        Type of quasi-symmetry (M, N).

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of quasi-symmetry errors.
        Key "QS_FF" is the flux function metric, key "QS_TP" is the triple product.

    """
    data = compute_B_dot_gradB(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data=data,
    )
    # TODO: can remove this call if compute_|B| changed to use B_covariant
    data = compute_covariant_magnetic_field(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data=data,
    )

    M = helicity[0]
    N = helicity[1]

    # covariant Boozer components: I = B_theta, G = B_zeta (in Boozer coordinates)
    if check_derivs("I", R_transform, Z_transform, L_transform):
        data["I"] = jnp.mean(data["B_theta"])
        data["G"] = jnp.mean(data["B_zeta"])

    # QS two-term (T^3)
    if check_derivs("f_C", R_transform, Z_transform, L_transform):
        data["f_C"] = (M * data["iota"] - N) * (data["psi_r"] / data["sqrt(g)"]) * (
            data["B_zeta"] * data["|B|_t"] - data["B_theta"] * data["|B|_z"]
        ) - (M * data["G"] + N * data["I"]) * data["B*grad(|B|)"]
    # QS triple product (T^4/m^2)
    if check_derivs("f_T", R_transform, Z_transform, L_transform):
        data["f_T"] = (data["psi_r"] / data["sqrt(g)"]) * (
            data["|B|_t"] * data["(B*grad(|B|))_z"]
            - data["|B|_z"] * data["(B*grad(|B|))_t"]
        )

    return data
