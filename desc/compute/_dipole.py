from desc.backend import jnp

from .data_index import register_compute_fun

_DIPOLE_PARAMETERIZATION = "desc.dipole._Dipole"


@register_compute_fun(
    name="X",
    label="X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate of the dipole position",
    dim=0,
    params=["X"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _X_Dipole(params, transforms, profiles, data, **kwargs):
    data["X"] = params["X"]
    return data


@register_compute_fun(
    name="Y",
    label="Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate of the dipole position",
    dim=0,
    params=["Y"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _Y_Dipole(params, transforms, profiles, data, **kwargs):
    data["Y"] = params["Y"]
    return data


@register_compute_fun(
    name="Z",
    label="Z",
    units="m",
    units_long="meters",
    description="Cartesian Z coordinate of the dipole position",
    dim=0,
    params=["Z"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _Z_Dipole(params, transforms, profiles, data, **kwargs):
    data["Z"] = params["Z"]
    return data


@register_compute_fun(
    name="phi",
    label="\\phi",
    units="rad",
    units_long="radians",
    description="Azimuthal orientation angle of the dipole moment",
    dim=0,
    params=["phi"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _phi_Dipole(params, transforms, profiles, data, **kwargs):
    data["phi"] = params["phi"]
    return data


@register_compute_fun(
    name="theta",
    label="\\theta",
    units="rad",
    units_long="radians",
    description="Polar orientation angle of the dipole moment",
    dim=0,
    params=["theta"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _theta_Dipole(params, transforms, profiles, data, **kwargs):
    data["theta"] = params["theta"]
    return data


@register_compute_fun(
    name="m0",
    label="m_0",
    units="A \\cdot m^2",
    units_long="Ampere meters squared",
    description="Magnitude of the magnetic dipole moment",
    dim=0,
    params=["m0"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _m0_Dipole(params, transforms, profiles, data, **kwargs):
    data["m0"] = params["m0"]
    return data


@register_compute_fun(
    name="rho",
    label="\\rho",
    units="~",
    units_long="None",
    description="Dimensionless optimization parameter in range (-1, 1) that sets "
    "the radial direction and magnitude of the dipole moment",
    dim=0,
    params=["rho"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _rho_Dipole(params, transforms, profiles, data, **kwargs):
    data["rho"] = params["rho"]
    return data


@register_compute_fun(
    name="M0",
    label="M_0",
    units="A \\cdot m^2",
    units_long="Ampere meters squared",
    description="Effective dipole moment strength, with radial direction",
    dim=0,
    params=["m0", "rho"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _M0_Dipole(params, transforms, profiles, data, **kwargs):
    data["M0"] = params["m0"] * params["rho"]
    return data


@register_compute_fun(
    name="position",
    label="\\mathbf{x}_0",
    units="m",
    units_long="meters",
    description="Position of the dipole in Cartesian [X, Y, Z] coordinates",
    dim=3,
    params=["X", "Y", "Z"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _position_Dipole(params, transforms, profiles, data, **kwargs):
    data["position"] = jnp.array([params["X"], params["Y"], params["Z"]])
    return data


@register_compute_fun(
    name="m_xyz",
    label="\\mathbf{m}",
    units="A \\cdot m^2",
    units_long="Ampere meters squared",
    description="Dipole moment vector in Cartesian [X, Y, Z] components",
    dim=3,
    params=["m0", "rho", "phi", "theta"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization=_DIPOLE_PARAMETERIZATION,
)
def _m_xyz_Dipole(params, transforms, profiles, data, **kwargs):
    M0 = params["m0"] * params["rho"]
    m_hat = jnp.array(
        [
            jnp.sin(params["theta"]) * jnp.cos(params["phi"]),
            jnp.sin(params["theta"]) * jnp.sin(params["phi"]),
            jnp.cos(params["theta"]),
        ]
    )
    data["m_xyz"] = M0 * m_hat
    return data