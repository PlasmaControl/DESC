"""data_index contains all the quantities calculated by the compute functions.

label = (str) Title of the quantity in LaTeX format.
units = (str) Units of the quantity in LaTeX format.
units_long (str) Full units without abbreviations.
description (str) Description of the quantity.
fun = (str) Function name in compute_funs.py that computes the quantity.
dim = (int) Dimension of the quantity: 0-D (global qty), 1-D (local scalar qty),
    or 3-D (local vector qty).
dependencies : dictionary of things required to compute the quantity
    params : parameters of equilibrium needed, eg R_lmn, Z_lmn
    transforms : dict of keys anrsd derivative orders [rho, theta, zeta] for R, Z, etc.
    profiles : profiles needed, eg iota, pressure
    data : other items in the data index needed to compute qty
NOTE: should only list *direct* dependencies. The full dependencies will be built
recursively at runtime using each quantities direct dependencies.
"""

data_index = {}

# flux coordinates
data_index["rho"] = {
    "label": "\\rho",
    "units": "~",
    "units_long": "None",
    "description": "Radial coordinate, proportional to the square root of the toroidal flux",
    "fun": "compute_flux_coords",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [],
    },
}
data_index["theta"] = {
    "label": "\\theta",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal angular coordinate (geometric, not magnetic)",
    "fun": "compute_flux_coords",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [],
    },
}
data_index["zeta"] = {
    "label": "\\zeta",
    "units": "rad",
    "units_long": "radians",
    "description": "Toroidal angular coordinate, equal to the geometric toroidal angle",
    "fun": "compute_flux_coords",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [],
    },
}

# toroidal flux
data_index["psi"] = {
    "label": "\\psi = \\Psi / (2 \\pi)",
    "units": "Wb",
    "units_long": "Webers",
    "description": "Toroidal flux",
    "fun": "compute_toroidal_flux",
    "dim": 1,
    "dependencies": {
        "params": ["Psi"],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [],
    },
}
data_index["psi_r"] = {
    "label": "\\psi' = \\partial_{\\rho} \\Psi / (2 \\pi)",
    "units": "Wb",
    "units_long": "Webers",
    "description": "Toroidal flux, first radial derivative",
    "fun": "compute_toroidal_flux",
    "dim": 1,
    "dependencies": {
        "params": ["Psi"],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [],
    },
}
data_index["psi_rr"] = {
    "label": "\\psi'' = \\partial_{\\rho\\rho} \\Psi / (2 \\pi)",
    "units": "Wb",
    "units_long": "Webers",
    "description": "Toroidal flux, second radial derivative",
    "fun": "compute_toroidal_flux",
    "dim": 1,
    "dependencies": {
        "params": ["Psi"],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [],
    },
}
data_index["grad(psi)"] = {
    "label": "\\nabla\\psi",
    "units": "Wb / m",
    "units_long": "Webers per meter",
    "description": "Toroidal flux gradient",
    "fun": "compute_toroidal_flux_gradient",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "e^rho"],
    },
}
data_index["|grad(psi)|^2"] = {
    "label": "|\\nabla\\psi|^{2}",
    "units": "Wb / m",
    "units_long": "Webers squared per square meter",
    "description": "Toroidal flux gradient magnitude squared",
    "fun": "compute_toroidal_flux_gradient",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "g^rr"],
    },
}
data_index["|grad(psi)|"] = {
    "label": "|\\nabla\\psi|",
    "units": "Wb / m",
    "units_long": "Webers per meter",
    "description": "Toroidal flux gradient magnitude",
    "fun": "compute_toroidal_flux_gradient",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["|grad(psi)|^2"],
    },
}
# R
data_index["R"] = {
    "label": "R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_r"] = {
    "label": "\\partial_{\\rho} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, first radial derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[1, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_t"] = {
    "label": "\\partial_{\\theta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, first poloidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_z"] = {
    "label": "\\partial_{\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, first toroidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rr"] = {
    "label": "\\partial_{\\rho\\rho} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, second radial derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[2, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_tt"] = {
    "label": "\\partial_{\\theta\\theta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, second poloidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 2, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, second toroidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 0, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rt"] = {
    "label": "\\partial_{\\rho\\theta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, second derivative wrt radius and poloidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[1, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rz"] = {
    "label": "\\partial_{\\rho\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, second derivative wrt radius and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[1, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_tz"] = {
    "label": "\\partial_{\\theta\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 1, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rrr"] = {
    "label": "\\partial_{\\rho\\rho\\rho} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third radial derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[3, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_ttt"] = {
    "label": "\\partial_{\\theta\\theta\\theta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third poloidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 3, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_zzz"] = {
    "label": "\\partial_{\\zeta\\zeta\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third toroidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 0, 3]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rrt"] = {
    "label": "\\partial_{\\rho\\rho\\theta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third derivative, wrt radius twice and poloidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[2, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rtt"] = {
    "label": "\\partial_{\\rho\\theta\\theta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third derivative wrt radius and poloidal angle twice",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[1, 2, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rrz"] = {
    "label": "\\partial_{\\rho\\rho\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third derivative wrt radius twice and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[2, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rzz"] = {
    "label": "\\partial_{\\rho\\zeta\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third derivative wrt radius and toroidal angle twice",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[1, 0, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_ttz"] = {
    "label": "\\partial_{\\theta\\theta\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third derivative wrt poloidal angle twice and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 2, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_tzz"] = {
    "label": "\\partial_{\\theta\\zeta\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third derivative wrt poloidal angle  and toroidal angle twice",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[0, 1, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["R_rtz"] = {
    "label": "\\partial_{\\rho\\theta\\zeta} R",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius in lab frame, third derivative wrt radius, poloidal angle, and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["R_lmn"],
        "transforms": {"R": [[1, 1, 1]]},
        "profiles": [],
        "data": [],
    },
}

# Z
data_index["Z"] = {
    "label": "Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_r"] = {
    "label": "\\partial_{\\rho} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, first radial derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[1, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_t"] = {
    "label": "\\partial_{\\theta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, first poloidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_z"] = {
    "label": "\\partial_{\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, first toroidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rr"] = {
    "label": "\\partial_{\\rho\\rho} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, second radial derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[2, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_tt"] = {
    "label": "\\partial_{\\theta\\theta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, second poloidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 2, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, second toroidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 0, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rt"] = {
    "label": "\\partial_{\\rho\\theta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, second derivative wrt radius and poloidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[1, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rz"] = {
    "label": "\\partial_{\\rho\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, second derivative wrt radius and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[1, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_tz"] = {
    "label": "\\partial_{\\theta\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 1, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rrr"] = {
    "label": "\\partial_{\\rho\\rho\\rho} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third radial derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[3, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_ttt"] = {
    "label": "\\partial_{\\theta\\theta\\theta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third poloidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 3, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_zzz"] = {
    "label": "\\partial_{\\zeta\\zeta\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third toroidal derivative",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 0, 3]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rrt"] = {
    "label": "\\partial_{\\rho\\rho\\theta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third derivative, wrt radius twice and poloidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[2, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rtt"] = {
    "label": "\\partial_{\\rho\\theta\\theta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third derivative wrt radius and poloidal angle twice",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[1, 2, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rrz"] = {
    "label": "\\partial_{\\rho\\rho\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third derivative wrt radius twice and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[2, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rzz"] = {
    "label": "\\partial_{\\rho\\zeta\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third derivative wrt radius and toroidal angle twice",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[1, 0, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_ttz"] = {
    "label": "\\partial_{\\theta\\theta\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third derivative wrt poloidal angle twice and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 2, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_tzz"] = {
    "label": "\\partial_{\\theta\\zeta\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third derivative wrt poloidal angle  and toroidal angle twice",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[0, 1, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["Z_rtz"] = {
    "label": "\\partial_{\\rho\\theta\\zeta} Z",
    "units": "m",
    "units_long": "meters",
    "description": "Vertical coordinate in lab frame, third derivative wrt radius, poloidal angle, and toroidal angle",
    "fun": "compute_toroidal_coords",
    "dim": 1,
    "dependencies": {
        "params": ["Z_lmn"],
        "transforms": {"Z": [[1, 1, 1]]},
        "profiles": [],
        "data": [],
    },
}

# cartesian coordinates
data_index["phi"] = {
    "label": "\\phi = \\zeta",
    "units": "rad",
    "units_long": "radians",
    "description": "Toroidal angle in lab frame",
    "fun": "compute_cartesian_coords",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["zeta"],
    },
}
data_index["X"] = {
    "label": "X = R \\cos{\\phi}",
    "units": "m",
    "units_long": "meters",
    "description": "Cartesian X coordinate",
    "fun": "compute_cartesian_coords",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R", "phi"],
    },
}
data_index["Y"] = {
    "label": "Y = R \\sin{\\phi}",
    "units": "m",
    "units_long": "meters",
    "description": "Cartesian Y coordinate",
    "fun": "compute_cartesian_coords",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R", "phi"],
    },
}

# lambda
data_index["lambda"] = {
    "label": "\\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_r"] = {
    "label": "\\partial_{\\rho} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, first radial derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[1, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_t"] = {
    "label": "\\partial_{\\theta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, first poloidal derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_z"] = {
    "label": "\\partial_{\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, first toroidal derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rr"] = {
    "label": "\\partial_{\\rho\\rho} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, second radial derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[2, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_tt"] = {
    "label": "\\partial_{\\theta\\theta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, second poloidal derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 2, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, second toroidal derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 0, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rt"] = {
    "label": "\\partial_{\\rho\\theta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, second derivative wrt radius and poloidal angle",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[1, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rz"] = {
    "label": "\\partial_{\\rho\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, second derivative wrt radius and toroidal angle",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[1, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_tz"] = {
    "label": "\\partial_{\\theta\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 1, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rrr"] = {
    "label": "\\partial_{\\rho\\rho\\rho} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third radial derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[3, 0, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_ttt"] = {
    "label": "\\partial_{\\theta\\theta\\theta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third poloidal derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 3, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_zzz"] = {
    "label": "\\partial_{\\zeta\\zeta\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third toroidal derivative",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 0, 3]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rrt"] = {
    "label": "\\partial_{\\rho\\rho\\theta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third derivative, wrt radius twice and poloidal angle",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[2, 1, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rtt"] = {
    "label": "\\partial_{\\rho\\theta\\theta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third derivative wrt radius and poloidal angle twice",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[1, 2, 0]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rrz"] = {
    "label": "\\partial_{\\rho\\rho\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third derivative wrt radius twice and toroidal angle",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[2, 0, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rzz"] = {
    "label": "\\partial_{\\rho\\zeta\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third derivative wrt radius and toroidal angle twice",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[1, 0, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_ttz"] = {
    "label": "\\partial_{\\theta\\theta\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third derivative wrt poloidal angle twice and toroidal angle",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 2, 1]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_tzz"] = {
    "label": "\\partial_{\\theta\\zeta\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third derivative wrt poloidal angle  and toroidal angle twice",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[0, 1, 2]]},
        "profiles": [],
        "data": [],
    },
}
data_index["lambda_rtz"] = {
    "label": "\\partial_{\\rho\\theta\\zeta} \\lambda",
    "units": "rad",
    "units_long": "radians",
    "description": "Poloidal stream function, third derivative wrt radius, poloidal angle, and toroidal angle",
    "fun": "compute_lambda",
    "dim": 1,
    "dependencies": {
        "params": ["L_lmn"],
        "transforms": {"L": [[1, 1, 1]]},
        "profiles": [],
        "data": [],
    },
}

# pressure
data_index["p"] = {
    "label": "p",
    "units": "Pa",
    "units_long": "Pascal",
    "description": "Pressure",
    "fun": "compute_pressure",
    "dim": 1,
    "dependencies": {
        "params": ["p_l"],
        "transforms": {},
        "profiles": ["pressure"],
        "data": [],
    },
}
data_index["p_r"] = {
    "label": "\\partial_{\\rho} p",
    "units": "Pa",
    "units_long": "Pascal",
    "description": "Pressure, first radial derivative",
    "fun": "compute_pressure",
    "dim": 1,
    "dependencies": {
        "params": ["p_l"],
        "transforms": {},
        "profiles": ["pressure"],
        "data": [],
    },
}
data_index["grad(p)"] = {
    "label": "\\nabla p",
    "units": "N \\cdot m^{-3}",
    "units_long": "Newtons / cubic meter",
    "description": "Pressure gradient",
    "fun": "compute_pressure_gradient",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["p_r", "e^rho"],
    },
}
data_index["|grad(p)|"] = {
    "label": "|\\nabla p|",
    "units": "N \\cdot m^{-3}",
    "units_long": "Newtons / cubic meter",
    "description": "Magnitude of pressure gradient",
    "fun": "compute_pressure_gradient",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["p_r", "|grad(rho)|"],
    },
}
data_index["<|grad(p)|>_vol"] = {
    "label": "<|\\nabla p|>_{vol}",
    "units": "N \\cdot m^{-3}",
    "units_long": "Newtons / cubic meter",
    "description": "Volume average of magnitude of pressure gradient",
    "fun": "compute_pressure_gradient",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["|grad(p)|", "sqrt(g)", "V"],
    },
}

# rotational transform
data_index["iota"] = {
    "label": "\\iota",
    "units": "~",
    "units_long": "None",
    "description": "Rotational transform",
    "fun": "compute_rotational_transform",
    "dim": 1,
    "dependencies": {
        "params": ["i_l", "c_l"],
        "transforms": {},
        "profiles": ["iota", "current"],
        "data": ["psi_r", "lambda_t", "lambda_z", "g_tt", "g_tz", "sqrt(g)"],
    },
}
data_index["iota_r"] = {
    "label": "\\partial_{\\rho} \\iota",
    "units": "~",
    "units_long": "None",
    "description": "Rotational transform, first radial derivative",
    "fun": "compute_rotational_transform",
    "dim": 1,
    "dependencies": {
        "params": ["i_l", "c_l"],
        "transforms": {},
        "profiles": ["iota", "current"],
        "data": [
            "psi_r",
            "psi_rr",
            "lambda_t",
            "lambda_z",
            "lambda_rt",
            "lambda_rz",
            "g_tt",
            "g_tt_r",
            "g_tz",
            "g_tz_r",
            "sqrt(g)",
            "sqrt(g)_r",
        ],
    },
}

# covariant basis
data_index["e_rho"] = {
    "label": "\\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_r", "Z_r"],
    },
}
data_index["e_theta"] = {
    "label": "\\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_t", "Z_t"],
    },
}
data_index["e_zeta"] = {
    "label": "\\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R", "R_z", "Z_z"],
    },
}
data_index["e_rho_r"] = {
    "label": "\\partial_{\\rho} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, derivative wrt radial coordinate",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rr", "Z_rr"],
    },
}
data_index["e_rho_t"] = {
    "label": "\\partial_{\\theta} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, derivative wrt poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rt", "Z_rt"],
    },
}
data_index["e_rho_z"] = {
    "label": "\\partial_{\\zeta} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, derivative wrt toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rz", "Z_rz"],
    },
}
data_index["e_theta_r"] = {
    "label": "\\partial_{\\rho} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, derivative wrt radial coordinate",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rt", "Z_rt"],
    },
}
data_index["e_theta_t"] = {
    "label": "\\partial_{\\theta} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, derivative wrt poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_tt", "Z_tt"],
    },
}
data_index["e_theta_z"] = {
    "label": "\\partial_{\\zeta} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, derivative wrt toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_tz", "Z_tz"],
    },
}
data_index["e_zeta_r"] = {
    "label": "\\partial_{\\rho} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, derivative wrt radial coordinate",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rz", "R_r", "Z_rz"],
    },
}
data_index["e_zeta_t"] = {
    "label": "\\partial_{\\theta} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, derivative wrt poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_tz", "R_t", "Z_tz"],
    },
}
data_index["e_zeta_z"] = {
    "label": "\\partial_{\\zeta} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, derivative wrt toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_zz", "R_z", "Z_zz"],
    },
}
data_index["e_rho_rr"] = {
    "label": "\\partial_{\\rho\\rho} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, second derivative wrt radial coordinate",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rrr", "Z_rrr"],
    },
}
data_index["e_rho_tt"] = {
    "label": "\\partial_{\\theta\\theta} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, second derivative wrt poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rtt", "Z_rtt"],
    },
}
data_index["e_rho_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, second derivative wrt toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rzz", "Z_rzz"],
    },
}
data_index["e_rho_rt"] = {
    "label": "\\partial_{\\rho\\theta} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, second derivative wrt radial coordinate and poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rrt", "Z_rrt"],
    },
}
data_index["e_rho_rz"] = {
    "label": "\\partial_{\\rho\\zeta} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, second derivative wrt radial coordinate and toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rrz", "Z_rrz"],
    },
}
data_index["e_rho_tz"] = {
    "label": "\\partial_{\\theta\\zeta} \\mathbf{e}_{\\rho}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant radial basis vector, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rtz", "Z_rtz"],
    },
}
data_index["e_theta_rr"] = {
    "label": "\\partial_{\\rho\\rho} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, second derivative wrt radial coordinate",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rrt", "Z_rrt"],
    },
}
data_index["e_theta_tt"] = {
    "label": "\\partial_{\\theta\\theta} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, second derivative wrt poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_ttt", "Z_ttt"],
    },
}
data_index["e_theta_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, second derivative wrt toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_tzz", "Z_tzz"],
    },
}
data_index["e_theta_rt"] = {
    "label": "\\partial_{\\rho\\theta} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, second derivative wrt radial coordinate and poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rtt", "Z_rtt"],
    },
}
data_index["e_theta_rz"] = {
    "label": "\\partial_{\\rho\\zeta} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, second derivative wrt radial coordinate and toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rtz", "Z_rtz"],
    },
}
data_index["e_theta_tz"] = {
    "label": "\\partial_{\\theta\\zeta} \\mathbf{e}_{\\theta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant poloidal basis vector, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_ttz", "Z_ttz"],
    },
}
data_index["e_zeta_rr"] = {
    "label": "\\partial_{\\rho\\rho} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, second derivative wrt radial coordinate",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rrz", "R_rr", "Z_rrz"],
    },
}
data_index["e_zeta_tt"] = {
    "label": "\\partial_{\\theta\\theta} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, second derivative wrt poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_ttz", "R_tt", "Z_ttz"],
    },
}
data_index["e_zeta_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, second derivative wrt toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_zzz", "R_zz", "Z_zzz"],
    },
}
data_index["e_zeta_rt"] = {
    "label": "\\partial_{\\rho\\theta} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, second derivative wrt radial coordinate and poloidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rtz", "R_rt", "Z_rtz"],
    },
}
data_index["e_zeta_rz"] = {
    "label": "\\partial_{\\rho\\zeta} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, second derivative wrt radial coordinate and toroidal angle",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_rzz", "R_rz", "Z_rzz"],
    },
}
data_index["e_zeta_tz"] = {
    "label": "\\partial_{\\theta\\zeta} \\mathbf{e}_{\\zeta}",
    "units": "m",
    "units_long": "meters",
    "description": "Covariant toroidal basis vector, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_covariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R_tzz", "R_tz", "Z_tzz"],
    },
}

# contravariant basis
data_index["e^rho"] = {
    "label": "\\mathbf{e}^{\\rho}",
    "units": "m^{-1}",
    "units_long": "inverse meters",
    "description": "Contravariant radial basis vector",
    "fun": "compute_contravariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_theta", "e_zeta", "sqrt(g)"],
    },
}
data_index["e^theta"] = {
    "label": "\\mathbf{e}^{\\theta}",
    "units": "m^{-1}",
    "units_long": "inverse meters",
    "description": "Contravariant poloidal basis vector",
    "fun": "compute_contravariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_zeta", "e_rho", "sqrt(g)"],
    },
}
data_index["e^zeta"] = {
    "label": "\\mathbf{e}^{\\zeta}",
    "units": "m^{-1}",
    "units_long": "inverse meters",
    "description": "Contravariant toroidal basis vector",
    "fun": "compute_contravariant_basis",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R"],
    },
}

# Jacobian
data_index["sqrt(g)"] = {
    "label": "\\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho", "e_theta", "e_zeta"],
    },
}
data_index["|e_theta x e_zeta|"] = {
    "label": "|e_{\\theta} \\times e_{\\zeta}|",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "2D jacobian determinant for constant rho surface",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_theta", "e_zeta"],
    },
}
data_index["|e_zeta x e_rho|"] = {
    "label": "|e_{\\zeta} \\times e_{\\rho}|",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "2D jacobian determinant for constant theta surface",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_zeta", "e_rho"],
    },
}
data_index["|e_rho x e_theta|"] = {
    "label": "|e_{\\rho} \\times e_{\\theta}|",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "2D jacobian determinant for constant zeta surface",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho", "e_theta"],
    },
}
data_index["sqrt(g)_r"] = {
    "label": "\\partial_{\\rho} \\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant, derivative wrt radial coordinate",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho", "e_theta", "e_zeta", "e_rho_r", "e_theta_r", "e_zeta_r"],
    },
}
data_index["sqrt(g)_t"] = {
    "label": "\\partial_{\\theta} \\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant, derivative wrt poloidal angle",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho", "e_theta", "e_zeta", "e_rho_t", "e_theta_t", "e_zeta_t"],
    },
}
data_index["sqrt(g)_z"] = {
    "label": "\\partial_{\\zeta} \\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant, derivative wrt toroidal angle",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho", "e_theta", "e_zeta", "e_rho_z", "e_theta_z", "e_zeta_z"],
    },
}
data_index["sqrt(g)_rr"] = {
    "label": "\\partial_{\\rho\\rho} \\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant, second derivative wrt radial coordinate",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "e_rho",
            "e_theta",
            "e_zeta",
            "e_rho_r",
            "e_theta_r",
            "e_zeta_r",
            "e_rho_rr",
            "e_theta_rr",
            "e_zeta_rr",
        ],
    },
}
data_index["sqrt(g)_tt"] = {
    "label": "\\partial_{\\theta\\theta} \\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant, second derivative wrt poloidal angle",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "e_rho",
            "e_theta",
            "e_zeta",
            "e_rho_t",
            "e_theta_t",
            "e_zeta_t",
            "e_rho_tt",
            "e_theta_tt",
            "e_zeta_tt",
        ],
    },
}
data_index["sqrt(g)_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} \\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant, second derivative wrt toroidal angle",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "e_rho",
            "e_theta",
            "e_zeta",
            "e_rho_z",
            "e_theta_z",
            "e_zeta_z",
            "e_rho_zz",
            "e_theta_zz",
            "e_zeta_zz",
        ],
    },
}
data_index["sqrt(g)_tz"] = {
    "label": "\\partial_{\\theta\\zeta} \\sqrt{g}",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Jacobian determinant, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_jacobian",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "e_rho",
            "e_theta",
            "e_zeta",
            "e_rho_z",
            "e_theta_z",
            "e_zeta_z",
            "e_rho_t",
            "e_theta_t",
            "e_zeta_t",
            "e_rho_tz",
            "e_theta_tz",
            "e_zeta_tz",
        ],
    },
}

# covariant metric coefficients
data_index["g_rr"] = {
    "label": "g_{\\rho\\rho}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Radial/Radial element of covariant metric tensor",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho"],
    },
}
data_index["g_tt"] = {
    "label": "g_{\\theta\\theta}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Poloidal/Poloidal element of covariant metric tensor",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_theta"],
    },
}
data_index["g_zz"] = {
    "label": "g_{\\zeta\\zeta}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Toroidal/Toroidal element of covariant metric tensor",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_zeta"],
    },
}
data_index["g_rt"] = {
    "label": "g_{\\rho\\theta}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Radial/Poloidal element of covariant metric tensor",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho", "e_theta"],
    },
}
data_index["g_rz"] = {
    "label": "g_{\\rho\\zeta}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Radial/Toroidal element of covariant metric tensor",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_rho", "e_zeta"],
    },
}
data_index["g_tz"] = {
    "label": "g_{\\theta\\zeta}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Poloidal/Toroidal element of covariant metric tensor",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_theta", "e_zeta"],
    },
}
data_index["g_tt_r"] = {
    "label": "\\partial_{\\rho} g_{\\theta\\theta}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Poloidal/Poloidal element of covariant metric tensor, derivative wrt rho",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_theta", "e_theta_r"],
    },
}
data_index["g_tz_r"] = {
    "label": "\\partial_{\\rho} g_{\\theta\\zeta}",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Poloidal/Toroidal element of covariant metric tensor, derivative wrt rho",
    "fun": "compute_covariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e_theta", "e_zeta", "e_theta_r", "e_zeta_r"],
    },
}

# contravariant metric coefficients
data_index["g^rr"] = {
    "label": "g^{\\rho\\rho}",
    "units": "m^{-2}",
    "units_long": "inverse square meters",
    "description": "Radial/Radial element of contravariant metric tensor",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e^rho"],
    },
}
data_index["g^tt"] = {
    "label": "g^{\\theta\\theta}",
    "units": "m^{-2}",
    "units_long": "inverse square meters",
    "description": "Poloidal/Poloidal element of contravariant metric tensor",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e^theta"],
    },
}
data_index["g^zz"] = {
    "label": "g^{\\zeta\\zeta}",
    "units": "m^{-2}",
    "units_long": "inverse square meters",
    "description": "Toroidal/Toroidal element of contravariant metric tensor",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e^zeta"],
    },
}
data_index["g^rt"] = {
    "label": "g^{\\rho\\theta}",
    "units": "m^{-2}",
    "units_long": "inverse square meters",
    "description": "Radial/Poloidal element of contravariant metric tensor",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e^rho", "e^theta"],
    },
}
data_index["g^rz"] = {
    "label": "g^{\\rho\\zeta}",
    "units": "m^{-2}",
    "units_long": "inverse square meters",
    "description": "Radial/Toroidal element of contravariant metric tensor",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e^rho", "e^zeta"],
    },
}
data_index["g^tz"] = {
    "label": "g^{\\theta\\zeta}",
    "units": "m^{-2}",
    "units_long": "inverse square meters",
    "description": "Poloidal/Toroidal element of contravariant metric tensor",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["e^theta", "e^zeta"],
    },
}
data_index["|grad(rho)|"] = {
    "label": "|\\nabla \\rho|",
    "units": "m^{-1}",
    "units_long": "inverse meters",
    "description": "Magnitude of contravariant radial basis vector",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["g^rr"],
    },
}
data_index["|grad(theta)|"] = {
    "label": "|\\nabla \\theta|",
    "units": "m^{-1}",
    "units_long": "inverse meters",
    "description": "Magnitude of contravariant poloidal basis vector",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["g^tt"],
    },
}
data_index["|grad(zeta)|"] = {
    "label": "|\\nabla \\zeta|",
    "units": "m^{-1}",
    "units_long": "inverse meters",
    "description": "Magnitude of contravariant toroidal basis vector",
    "fun": "compute_contravariant_metric_coefficients",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["g^zz"],
    },
}

# geometry
data_index["V"] = {
    "label": "V",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Volume",
    "fun": "compute_geometry",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["sqrt(g)"],
    },
}
data_index["A"] = {
    "label": "A",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Cross-sectional area",
    "fun": "compute_geometry",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["sqrt(g)", "R"],
    },
}
data_index["R0"] = {
    "label": "R_{0}",
    "units": "m",
    "units_long": "meters",
    "description": "Major radius",
    "fun": "compute_geometry",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["V", "A"],
    },
}
data_index["a"] = {
    "label": "a",
    "units": "m",
    "units_long": "meters",
    "description": "Minor radius",
    "fun": "compute_geometry",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["A"],
    },
}
data_index["R0/a"] = {
    "label": "R_{0} / a",
    "units": "~",
    "units_long": "None",
    "description": "Aspect ratio",
    "fun": "compute_geometry",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["R0", "a"],
    },
}
data_index["V(r)"] = {
    "label": "V(\\rho)",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Volume enclosed by flux surfaces",
    "fun": "compute_geometry",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["e_theta", "e_zeta", "Z"],
    },
}
data_index["V_r(r)"] = {
    "label": "\\partial_{\\rho} V(\\rho)",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Volume enclosed by flux surfaces, derivative wrt radial coordinate",
    "fun": "compute_geometry",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["sqrt(g)"],
    },
}
data_index["V_rr(r)"] = {
    "label": "\\partial_{\\rho\\rho} V(\\rho)",
    "units": "m^{3}",
    "units_long": "cubic meters",
    "description": "Volume enclosed by flux surfaces, second derivative wrt radial coordinate",
    "fun": "compute_geometry",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["sqrt(g)_r", "sqrt(g)"],
    },
}
data_index["S(r)"] = {
    "label": "S(\\rho)",
    "units": "m^{2}",
    "units_long": "square meters",
    "description": "Surface area of flux surfaces",
    "fun": "compute_geometry",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["|e_theta x e_zeta|"],
    },
}

# contravariant magnetic field
data_index["B0"] = {
    "label": "\\psi' / \\sqrt{g}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "sqrt(g)"],
    },
}
data_index["B^rho"] = {
    "label": "B^{\\rho}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant radial component of magnetic field",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [],
    },
}
data_index["B^theta"] = {
    "label": "B^{\\theta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant poloidal component of magnetic field",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "iota", "lambda_z"],
    },
}
data_index["B^zeta"] = {
    "label": "B^{\\zeta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant toroidal component of magnetic field",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "lambda_t"],
    },
}
data_index["B"] = {
    "label": "\\mathbf{B}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnetic field",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B^theta", "e_theta", "B^zeta", "e_zeta"],
    },
}
data_index["B_R"] = {
    "label": "B_{R}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Radial component of magnetic field in lab frame",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B"],
    },
}
data_index["B_phi"] = {
    "label": "B_{\\phi}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Toroidal component of magnetic field in lab frame",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B"],
    },
}
data_index["B_Z"] = {
    "label": "B_{Z}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Vertical component of magnetic field in lab frame",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B"],
    },
}
data_index["B0_r"] = {
    "label": "\\psi'' / \\sqrt{g} - \\psi' \\partial_{\\rho} \\sqrt{g} / g",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "psi_rr", "sqrt(g)", "sqrt(g)_r"],
    },
}
data_index["B^theta_r"] = {
    "label": "\\partial_{\\rho} B^{\\theta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant poloidal component of magnetic field, derivative wrt radial coordinate",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0_r", "iota", "lambda_z", "B0", "iota_r", "lambda_rz"],
    },
}
data_index["B^zeta_r"] = {
    "label": "\\partial_{\\rho} B^{\\zeta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant toroidal component of magnetic field, derivative wrt radial coordinate",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0_r", "lambda_t", "B0", "lambda_rt"],
    },
}
data_index["B_r"] = {
    "label": "\\partial_{\\rho} \\mathbf{B}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnetic field, derivative wrt radial coordinate",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta_r",
            "B^theta",
            "B^zeta_r",
            "B^zeta",
            "e_theta",
            "e_theta_r",
            "e_zeta",
            "e_zeta_r",
        ],
    },
}
data_index["B0_t"] = {
    "label": "-\\psi' \\partial_{\\theta} \\sqrt{g} / g",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "sqrt(g)_t", "sqrt(g)"],
    },
}
data_index["B^theta_t"] = {
    "label": "\\partial_{\\theta} B^{\\theta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant poloidal component of magnetic field, derivative wrt poloidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_t", "iota", "lambda_z", "lambda_tz"],
    },
}
data_index["B^zeta_t"] = {
    "label": "\\partial_{\\theta} B^{\\zeta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant toroidal component of magnetic field, derivative wrt poloidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_t", "lambda_t", "lambda_tt"],
    },
}
data_index["B_t"] = {
    "label": "\\partial_{\\theta} \\mathbf{B}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnetic field, derivative wrt poloidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta_t",
            "B^theta",
            "B^zeta_t",
            "B^zeta",
            "e_theta",
            "e_theta_t",
            "e_zeta",
            "e_zeta_t",
        ],
    },
}
data_index["B0_z"] = {
    "label": "-\\psi' \\partial_{\\zeta} \\sqrt{g} / g",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "sqrt(g)", "sqrt(g)_z"],
    },
}
data_index["B^theta_z"] = {
    "label": "\\partial_{\\zeta} B^{\\theta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant poloidal component of magnetic field, derivative wrt toroidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_z", "iota", "lambda_z", "lambda_zz"],
    },
}
data_index["B^zeta_z"] = {
    "label": "\\partial_{\\zeta} B^{\\zeta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant toroidal component of magnetic field, derivative wrt toroidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_z", "lambda_t", "lambda_tz"],
    },
}
data_index["B_z"] = {
    "label": "\\partial_{\\zeta} \\mathbf{B}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnetic field, derivative wrt toroidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta_z",
            "B^theta",
            "B^zeta_z",
            "B^zeta",
            "e_theta",
            "e_theta_z",
            "e_zeta",
            "e_zeta_z",
        ],
    },
}
data_index["B0_tt"] = {
    "label": "-\\psi' \\partial_{\\theta\\theta} \\sqrt{g} / g + "
    + "2 \\psi' (\\partial_{\\theta} \\sqrt{g})^2 / (\\sqrt{g})^{3}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "sqrt(g)", "sqrt(g)_t", "sqrt(g)_tt"],
    },
}
data_index["B^theta_tt"] = {
    "label": "\\partial_{\\theta\\theta} B^{\\theta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant poloidal component of magnetic field, second derivative wrt poloidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_t", "B0_tt", "iota", "lambda_z", "lambda_tz", "lambda_ttz"],
    },
}
data_index["B^zeta_tt"] = {
    "label": "\\partial_{\\theta\\theta} B^{\\zeta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant toroidal component of magnetic field, second derivative wrt poloidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_t", "B0_tt", "lambda_t", "lambda_tt", "lambda_ttt"],
    },
}
data_index["B0_zz"] = {
    "label": "-\\psi' \\partial_{\\zeta\\zeta} \\sqrt{g} / g + "
    + "2 \\psi' (\\partial_{\\zeta} \\sqrt{g})^2 / (\\sqrt{g})^{3}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "sqrt(g)", "sqrt(g)_z", "sqrt(g)_zz"],
    },
}
data_index["B^theta_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} B^{\\theta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant poloidal component of magnetic field, second derivative wrt toroidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_z", "B0_zz", "iota", "lambda_z", "lambda_zz", "lambda_zzz"],
    },
}
data_index["B^zeta_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} B^{\\zeta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant toroidal component of magnetic field, second derivative wrt toroidal angle",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B0", "B0_z", "B0_zz", "lambda_t", "lambda_tz", "lambda_tzz"],
    },
}
data_index["B0_tz"] = {
    "label": "-\\psi' \\partial_{\\theta\\zeta} \\sqrt{g} / g + "
    + "2 \\psi' \\partial_{\\theta} \\sqrt{g} \\partial_{\\zeta} \\sqrt{g} / "
    + "(\\sqrt{g})^{3}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["psi_r", "sqrt(g)", "sqrt(g)_t", "sqrt(g)_z", "sqrt(g)_tz"],
    },
}
data_index["B^theta_tz"] = {
    "label": "\\partial_{\\theta\\zeta} B^{\\theta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant poloidal component of magnetic field, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B0",
            "B0_t",
            "B0_z",
            "B0_tz",
            "iota",
            "lambda_z",
            "lambda_zz",
            "lambda_tz",
            "lambda_tzz",
        ],
    },
}
data_index["B^zeta_tz"] = {
    "label": "\\partial_{\\theta\\zeta} B^{\\zeta}",
    "units": "T \\cdot m^{-1}",
    "units_long": "Tesla / meters",
    "description": "Contravariant toroidal component of magnetic field, second derivative wrt poloidal and toroidal angles",
    "fun": "compute_contravariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B0",
            "B0_t",
            "B0_z",
            "B0_tz",
            "lambda_t",
            "lambda_tt",
            "lambda_tz",
            "lambda_ttz",
        ],
    },
}

# covariant magnetic field
data_index["B_rho"] = {
    "label": "B_{\\rho}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant radial component of magnetic field",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "e_rho"],
    },
}
data_index["B_theta"] = {
    "label": "B_{\\theta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant poloidal component of magnetic field",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "e_theta"],
    },
}
data_index["B_zeta"] = {
    "label": "B_{\\zeta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant toroidal component of magnetic field",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "e_zeta"],
    },
}
data_index["B_rho_r"] = {
    "label": "\\partial_{\\rho} B_{\\rho}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant radial component of magnetic field, derivative wrt radial coordinate",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_r", "e_rho", "e_rho_r"],
    },
}
data_index["B_theta_r"] = {
    "label": "\\partial_{\\rho} B_{\\theta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant poloidal component of magnetic field, derivative wrt radial coordinate",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_r", "e_theta", "e_theta_r"],
    },
}
data_index["B_zeta_r"] = {
    "label": "\\partial_{\\rho} B_{\\zeta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant toroidal component of magnetic field, derivative wrt radial coordinate",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_r", "e_zeta", "e_zeta_r"],
    },
}
data_index["B_rho_t"] = {
    "label": "\\partial_{\\theta} B_{\\rho}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant radial component of magnetic field, derivative wrt poloidal angle",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_t", "e_rho", "e_rho_t"],
    },
}
data_index["B_theta_t"] = {
    "label": "\\partial_{\\theta} B_{\\theta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant poloidal component of magnetic field, derivative wrt poloidal angle",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_t", "e_theta", "e_theta_t"],
    },
}
data_index["B_zeta_t"] = {
    "label": "\\partial_{\\theta} B_{\\zeta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant toroidal component of magnetic field, derivative wrt poloidal angle",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_t", "e_zeta", "e_zeta_t"],
    },
}
data_index["B_rho_z"] = {
    "label": "\\partial_{\\zeta} B_{\\rho}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant radial component of magnetic field, derivative wrt toroidal angle",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_z", "e_rho", "e_rho_z"],
    },
}
data_index["B_theta_z"] = {
    "label": "\\partial_{\\zeta} B_{\\theta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant poloidal component of magnetic field, derivative wrt toroidal angle",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_z", "e_theta", "e_theta_z"],
    },
}
data_index["B_zeta_z"] = {
    "label": "\\partial_{\\zeta} B_{\\zeta}",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Covariant toroidal component of magnetic field, derivative wrt toroidal angle",
    "fun": "compute_covariant_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B", "B_z", "e_zeta", "e_zeta_z"],
    },
}

# magnetic field magnitude
data_index["|B|"] = {
    "label": "|\\mathbf{B}|",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnitude of magnetic field",
    "fun": "compute_magnetic_field_magnitude",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["|B|^2"],
    },
}
data_index["|B|^2"] = {
    "label": "|\\mathbf{B}|^{2}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnitude of magnetic field, squared",
    "fun": "compute_magnetic_field_magnitude",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B^theta", "B^zeta", "g_tt", "g_tz", "g_zz"],
    },
}
data_index["|B|_t"] = {
    "label": "\\partial_{\\theta} |\\mathbf{B}|",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnitude of magnetic field, derivative wrt poloidal angle",
    "fun": "compute_magnetic_field_magnitude",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_t",
            "B^zeta_t",
            "g_tt",
            "g_tz",
            "g_zz",
            "e_theta",
            "e_zeta",
            "e_theta_t",
            "e_zeta_t",
            "|B|",
        ],
    },
}
data_index["|B|_z"] = {
    "label": "\\partial_{\\zeta} |\\mathbf{B}|",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnitude of magnetic field, derivative wrt toroidal angle",
    "fun": "compute_magnetic_field_magnitude",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_z",
            "B^zeta_z",
            "g_tt",
            "g_tz",
            "g_zz",
            "e_theta",
            "e_zeta",
            "e_theta_z",
            "e_zeta_z",
            "|B|",
        ],
    },
}
data_index["|B|_tt"] = {
    "label": "\\partial_{\\theta\\theta} |\\mathbf{B}|",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnitude of magnetic field, second derivative wrt poloidal angle",
    "fun": "compute_magnetic_field_magnitude",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_t",
            "B^zeta_t",
            "B^theta_tt",
            "B^zeta_tt",
            "g_tt",
            "g_tz",
            "g_zz",
            "e_theta",
            "e_zeta",
            "e_theta_t",
            "e_zeta_t",
            "e_theta_tt",
            "e_zeta_tt",
            "|B|",
            "|B|_t",
        ],
    },
}
data_index["|B|_zz"] = {
    "label": "\\partial_{\\zeta\\zeta} |\\mathbf{B}|",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnitude of magnetic field, second derivative wrt toroidal angle",
    "fun": "compute_magnetic_field_magnitude",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_z",
            "B^zeta_z",
            "B^theta_zz",
            "B^zeta_zz",
            "g_tt",
            "g_tz",
            "g_zz",
            "e_theta",
            "e_zeta",
            "e_theta_z",
            "e_zeta_z",
            "e_theta_zz",
            "e_zeta_zz",
            "|B|",
            "|B|_z",
        ],
    },
}
data_index["|B|_tz"] = {
    "label": "\\partial_{\\theta\\zeta} |\\mathbf{B}|",
    "units": "T",
    "units_long": "Tesla",
    "description": "Magnitude of magnetic field, derivative wrt poloidal and toroidal angles",
    "fun": "compute_magnetic_field_magnitude",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_t",
            "B^zeta_t",
            "B^theta_z",
            "B^zeta_z",
            "B^theta_tz",
            "B^zeta_tz",
            "g_tt",
            "g_tz",
            "g_zz",
            "e_theta",
            "e_zeta",
            "e_theta_t",
            "e_zeta_t",
            "e_theta_z",
            "e_zeta_z",
            "e_theta_tz",
            "e_zeta_tz",
            "|B|",
            "|B|_t",
            "|B|_z",
        ],
    },
}

# magnetic pressure gradient
data_index["grad(|B|^2)_rho"] = {
    "label": "(\\nabla B^{2})_{\\rho}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant radial component of magnetic pressure gradient",
    "fun": "compute_magnetic_pressure_gradient",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_r",
            "B^zeta_r",
            "B_theta",
            "B_zeta",
            "B_theta_r",
            "B_zeta_r",
        ],
    },
}
data_index["grad(|B|^2)_theta"] = {
    "label": "(\\nabla B^{2})_{\\theta}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant poloidal component of magnetic pressure gradient",
    "fun": "compute_magnetic_pressure_gradient",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_t",
            "B^zeta_t",
            "B_theta",
            "B_zeta",
            "B_theta_t",
            "B_zeta_t",
        ],
    },
}
data_index["grad(|B|^2)_zeta"] = {
    "label": "(\\nabla B^{2})_{\\zeta}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant toroidal component of magnetic pressure gradient",
    "fun": "compute_magnetic_pressure_gradient",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_z",
            "B^zeta_z",
            "B_theta",
            "B_zeta",
            "B_theta_z",
            "B_zeta_z",
        ],
    },
}
data_index["grad(|B|^2)"] = {
    "label": "\\nabla B^{2}",
    "units": "T^{2} \\cdot m^{-1}",
    "units_long": "Tesla squared / meters",
    "description": "Magnetic pressure gradient",
    "fun": "compute_magnetic_pressure_gradient",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "grad(|B|^2)_rho",
            "grad(|B|^2)_theta",
            "grad(|B|^2)_zeta",
            "e^rho",
            "e^theta",
            "e^zeta",
        ],
    },
}
data_index["|grad(|B|^2)|/2mu0"] = {
    "label": "|\\nabla B^{2}/(2\\mu_0)|",
    "units": "N \\cdot m^{-3}",
    "units_long": "Newton / cubic meter",
    "description": "Magnitude of magnetic pressure gradient",
    "fun": "compute_magnetic_pressure_gradient",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "grad(|B|^2)_rho",
            "grad(|B|^2)_theta",
            "grad(|B|^2)_zeta",
            "g^rr",
            "g^tt",
            "g^zz",
            "g^rt",
            "g^rz",
            "g^tz",
        ],
    },
}

# magnetic tension
data_index["(curl(B)xB)_rho"] = {
    "label": "((\\nabla \\times \\mathbf{B}) \\times \\mathbf{B})_{\\rho}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant radial component of Lorentz force",
    "fun": "compute_magnetic_tension",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B^theta", "B^zeta", "J^theta", "J^zeta"],
    },
}
data_index["(curl(B)xB)_theta"] = {
    "label": "((\\nabla \\times \\mathbf{B}) \\times \\mathbf{B})_{\\theta}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant poloidal component of Lorentz force",
    "fun": "compute_magnetic_tension",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B^zeta", "J^rho"],
    },
}
data_index["(curl(B)xB)_zeta"] = {
    "label": "((\\nabla \\times \\mathbf{B}) \\times \\mathbf{B})_{\\zeta}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant toroidal component of Lorentz force",
    "fun": "compute_magnetic_tension",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B^theta", "J^rho"],
    },
}
data_index["curl(B)xB"] = {
    "label": "(\\nabla \\times \\mathbf{B}) \\times \\mathbf{B}",
    "units": "T^{2} \\cdot m^{-1}",
    "units_long": "Tesla squared / meters",
    "description": "Lorentz force",
    "fun": "compute_magnetic_tension",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "(curl(B)xB)_rho",
            "(curl(B)xB)_theta",
            "(curl(B)xB)_zeta",
            "e^rho",
            "e^theta",
            "e^zeta",
        ],
    },
}
data_index["(B*grad)B"] = {
    "label": "(\\mathbf{B} \\cdot \\nabla) \\mathbf{B}",
    "units": "T^{2} \\cdot m^{-1}",
    "units_long": "Tesla squared / meters",
    "description": "Magnetic tension",
    "fun": "compute_magnetic_tension",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["curl(B)xB", "grad(|B|^2)"],
    },
}
data_index["((B*grad)B)_rho"] = {
    "label": "((\\mathbf{B} \\cdot \\nabla) \\mathbf{B})_{\\rho}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant radial component of magnetic tension",
    "fun": "compute_magnetic_tension",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["(B*grad)B", "e_rho"],
    },
}
data_index["((B*grad)B)_theta"] = {
    "label": "((\\mathbf{B} \\cdot \\nabla) \\mathbf{B})_{\\theta}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant poloidal component of magnetic tension",
    "fun": "compute_magnetic_tension",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["(B*grad)B", "e_theta"],
    },
}
data_index["((B*grad)B)_zeta"] = {
    "label": "((\\mathbf{B} \\cdot \\nabla) \\mathbf{B})_{\\zeta}",
    "units": "T^{2}",
    "units_long": "Tesla squared",
    "description": "Covariant toroidal component of magnetic tension",
    "fun": "compute_magnetic_tension",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["(B*grad)B", "e_zeta"],
    },
}
data_index["|(B*grad)B|"] = {
    "label": "|(\\mathbf{B} \\cdot \\nabla) \\mathbf{B}|",
    "units": "T^2 \\cdot m^{-1}",
    "units_long": "Tesla squared / meters",
    "description": "Magnitude of magnetic tension",
    "fun": "compute_magnetic_tension",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "((B*grad)B)_rho",
            "((B*grad)B)_theta",
            "((B*grad)B)_zeta",
            "g^rr",
            "g^tt",
            "g^zz",
            "g^rt",
            "g^rz",
            "g^tz",
        ],
    },
}

# B dot grad(B)
data_index["B*grad(|B|)"] = {
    "label": "\\mathbf{B} \\cdot \\nabla B",
    "units": "T^2 \\cdot m^{-1}",
    "units_long": "Tesla squared / meters",
    "description": "",
    "fun": "compute_B_dot_gradB",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B^theta", "B^zeta", "|B|_t", "|B|_z"],
    },
}
data_index["(B*grad(|B|))_t"] = {
    "label": "\\partial_{\\theta} (\\mathbf{B} \\cdot \\nabla B)",
    "units": "T^2 \\cdot m^{-1}",
    "units_long": "Tesla squared / meters",
    "description": "",
    "fun": "compute_B_dot_gradB",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_t",
            "B^zeta_t",
            "|B|_t",
            "|B|_z",
            "|B|_tt",
            "|B|_tz",
        ],
    },
}
data_index["(B*grad(|B|))_z"] = {
    "label": "\\partial_{\\zeta} (\\mathbf{B} \\cdot \\nabla B)",
    "units": "T^2 \\cdot m^{-1}",
    "units_long": "Tesla squared / meters",
    "description": "",
    "fun": "compute_B_dot_gradB",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "B^theta",
            "B^zeta",
            "B^theta_z",
            "B^zeta_z",
            "|B|_t",
            "|B|_z",
            "|B|_tz",
            "|B|_zz",
        ],
    },
}

# Boozer magnetic field
data_index["I"] = {
    "label": "I",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Boozer toroidal current",
    "fun": "compute_boozer_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["B_theta"],
    },
}
data_index["current"] = {
    "label": "\\frac{2\\pi}{\\mu_0} I",
    "units": "A",
    "units_long": "Amperes",
    "description": "Net toroidal current",
    "fun": "compute_boozer_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["I"],
    },
}
data_index["I_r"] = {
    "label": "\\partial_{\\rho} I",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Boozer toroidal current, derivative wrt radial coordinate",
    "fun": "compute_boozer_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["B_theta_r"],
    },
}
data_index["current_r"] = {
    "label": "\\frac{2\\pi}{\\mu_0} \\partial_{\\rho} I",
    "units": "A",
    "units_long": "Amperes",
    "description": "Net toroidal current, derivative wrt radial coordinate",
    "fun": "compute_boozer_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["I_r"],
    },
}
data_index["G"] = {
    "label": "G",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Boozer poloidal current",
    "fun": "compute_boozer_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["B_zeta"],
    },
}
data_index["G_r"] = {
    "label": "\\partial_{\\rho} G",
    "units": "T \\cdot m",
    "units_long": "Tesla * meters",
    "description": "Boozer poloidal current, derivative wrt radial coordinate",
    "fun": "compute_boozer_magnetic_field",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["B_zeta_r"],
    },
}

# contravariant current density
data_index["J^rho"] = {
    "label": "J^{\\rho}",
    "units": "A \\cdot m^{-3}",
    "units_long": "Amperes / cubic meter",
    "description": "Contravariant radial component of plasma current density",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B_zeta_t", "B_theta_z"],
    },
}
data_index["J^theta"] = {
    "label": "J^{\\theta}",
    "units": "A \\cdot m^{-3}",
    "units_long": "Amperes / cubic meter",
    "description": "Contravariant poloidal component of plasma current density",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B_rho_z", "B_zeta_r"],
    },
}
data_index["J^zeta"] = {
    "label": "J^{\\zeta}",
    "units": "A \\cdot m^{-3}",
    "units_long": "Amperes / cubic meter",
    "description": "Contravariant toroidal component of plasma current density",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B_theta_r", "B_rho_t"],
    },
}
data_index["J"] = {
    "label": "\\mathbf{J}",
    "units": "A \\cdot m^{-2}",
    "units_long": "Amperes / square meter",
    "description": "Plasma current density",
    "fun": "compute_contravariant_current_density",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["J^rho", "J^theta", "J^zeta", "e_rho", "e_theta", "e_zeta"],
    },
}
data_index["J_R"] = {
    "label": "J_{R}",
    "units": "A \\cdot m^{-2}",
    "units_long": "Amperes / square meter",
    "description": "Radial componenet of plasma current density in lab frame",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["J"],
    },
}
data_index["J_phi"] = {
    "label": "J_{\\phi}",
    "units": "A \\cdot m^{-2}",
    "units_long": "Amperes / square meter",
    "description": "Toroidal componenet of plasma current density in lab frame",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["J"],
    },
}
data_index["J_Z"] = {
    "label": "J_{Z}",
    "units": "A \\cdot m^{-2}",
    "units_long": "Amperes / square meter",
    "description": "Vertical componenet of plasma current density in lab frame",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["J"],
    },
}
data_index["|J|"] = {
    "label": "|\\mathbf{J}|",
    "units": "A \\cdot m^{-2}",
    "units_long": "Amperes / square meter",
    "description": "Magnitue of plasma current density",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "J^rho",
            "J^theta",
            "J^zeta",
            "g_rr",
            "g_tt",
            "g_zz",
            "g_rt",
            "g_rz",
            "g_tz",
        ],
    },
}
data_index["J_parallel"] = {
    "label": "\\mathbf{J} \\cdot \\mathbf{b}",
    "units": "A \\cdot m^{-2}",
    "units_long": "Amperes / square meter",
    "description": "Plasma current density parallel to magnetic field",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["J^rho", "J^theta", "J^zeta", "B_rho", "B_theta", "B_zeta", "|B|"],
    },
}
data_index["<J dot B>"] = {
    "label": "\\langle \\mathbf{J} \\cdot \\mathbf{B} \\rangle",
    "units": "T A \\cdot m^{-2}",
    "units_long": "Tesla Amperes / square meter",
    "description": "Current density parallel to magnetic field, flux surface averaged.",
    "fun": "compute_contravariant_current_density",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["J_parallel"],
    },
}
data_index["div(J_perp)"] = {
    "label": "\\nabla \\cdot \\mathbf{J}_{\\perp}",
    "units": "A \\cdot m^{-3}",
    "units_long": "Amperes / cubic meter",
    "description": "Divergence of plasma current density perpendicular to magnetic field",
    "fun": "compute_force_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["J^rho", "p_r", "|B|"],
    },
}

# energy
data_index["W"] = {
    "label": "W",
    "units": "J",
    "units_long": "Joules",
    "description": "Plasma total energy",
    "fun": "compute_energy",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["W_B", "W_p"],
    },
}
data_index["W_B"] = {
    "label": "W_B",
    "units": "J",
    "units_long": "Joules",
    "description": "Plasma magnetic energy",
    "fun": "compute_energy",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["|B|", "sqrt(g)"],
    },
}
data_index["W_p"] = {
    "label": "W_p",
    "units": "J",
    "units_long": "Joules",
    "description": "Plasma thermodynamic energy",
    "fun": "compute_energy",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["p", "sqrt(g)"],
        "kwargs": ["gamma"],
    },
}

# force error
data_index["F_rho"] = {
    "label": "F_{\\rho}",
    "units": "N \\cdot m^{-2}",
    "units_long": "Newtons / square meter",
    "description": "Covariant radial component of force balance error",
    "fun": "compute_force_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["p_r", "sqrt(g)", "B^theta", "B^zeta", "J^theta", "J^zeta"],
    },
}
data_index["F_theta"] = {
    "label": "F_{\\theta}",
    "units": "N \\cdot m^{-2}",
    "units_long": "Newtons / square meter",
    "description": "Covariant poloidal component of force balance error",
    "fun": "compute_force_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B^zeta", "J^rho"],
    },
}
data_index["F_zeta"] = {
    "label": "F_{\\zeta}",
    "units": "N \\cdot m^{-2}",
    "units_long": "Newtons / square meter",
    "description": "Covariant toroidal component of force balance error",
    "fun": "compute_force_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "B^theta", "J^rho"],
    },
}
data_index["F_beta"] = {
    "label": "F_{\\beta}",
    "units": "A",
    "units_long": "Amperes",
    "description": "Covariant helical component of force balance error",
    "fun": "compute_force_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["sqrt(g)", "J^rho"],
    },
}
data_index["F"] = {
    "label": "\\mathbf{J} \\times \\mathbf{B} - \\nabla p",
    "units": "N \\cdot m^{-3}",
    "units_long": "Newtons / cubic meter",
    "description": "Force balance error",
    "fun": "compute_force_error",
    "dim": 3,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["F_rho", "F_theta", "F_zeta", "e^rho", "e^theta", "e^zeta"],
    },
}
data_index["|F|"] = {
    "label": "|\\mathbf{J} \\times \\mathbf{B} - \\nabla p|",
    "units": "N \\cdot m^{-3}",
    "units_long": "Newtons / cubic meter",
    "description": "Magnitude of force balance error",
    "fun": "compute_force_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "F_rho",
            "F_theta",
            "F_zeta",
            "g^rr",
            "g^tt",
            "g^zz",
            "g^rt",
            "g^rz",
            "g^tz",
        ],
    },
}
data_index["<|F|>_vol"] = {
    "label": "<|\\mathbf{J} \\times \\mathbf{B} - \\nabla p|>_{vol}",
    "units": "N \\cdot m^{-3}",
    "units_long": "Newtons / cubic meter",
    "description": "Volume average of magnitude of force balance error",
    "fun": "compute_force_error",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["|F|", "sqrt(g)", "V"],
    },
}
data_index["|beta|"] = {
    "label": "|B^{\\theta} \\nabla \\zeta - B^{\\zeta} \\nabla \\theta|",
    "units": "T \\cdot m^{-2}",
    "units_long": "Tesla / square meter",
    "description": "Magnitude of helical basis vector",
    "fun": "compute_force_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["B^theta", "B^zeta", "g^tt", "g^zz", "g^tz"],
    },
}

# quasi-symmetry
# TODO: add w_mn, etc from boozer stuff
data_index["nu"] = {
    "label": "\\nu = \\zeta_{B} - \\zeta",
    "units": "rad",
    "units_long": "radians",
    "description": "Boozer toroidal stream function",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"w": [[0, 0, 0]], "B": [[0, 0, 0]]},
        "profiles": [],
        "data": ["lambda", "B_theta", "B_zeta"],
    },
}
data_index["nu_t"] = {
    "label": "\\partial_{\\theta} \\nu",
    "units": "rad",
    "units_long": "radians",
    "description": "Boozer toroidal stream function, derivative wrt poloidal angle",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"w": [[0, 1, 0]], "B": [[0, 0, 0]]},
        "profiles": [],
        "data": ["lambda_t", "B_theta", "B_zeta"],
    },
}
data_index["nu_z"] = {
    "label": "\\partial_{\\zeta} \\nu",
    "units": "rad",
    "units_long": "radians",
    "description": "Boozer toroidal stream function, derivative wrt toroidal angle",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"w": [[0, 0, 1]], "B": [[0, 0, 0]]},
        "profiles": [],
        "data": ["lambda_z", "B_theta", "B_zeta"],
    },
}
data_index["theta_B"] = {
    "label": "\\theta_{B}",
    "units": "rad",
    "units_long": "radians",
    "description": "Boozer poloidal angular coordinate",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["theta", "lambda", "iota", "nu"],
    },
}
data_index["zeta_B"] = {
    "label": "\\zeta_{B}",
    "units": "rad",
    "units_long": "radians",
    "description": "Boozer toroidal angular coordinate",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["zeta", "nu"],
    },
}
data_index["sqrt(g)_B"] = {
    "label": "\\sqrt{g}_{B}",
    "units": "~",
    "units_long": "None",
    "description": "Jacobian determinant of Boozer coordinates",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["lambda_t", "lambda_z", "nu_t", "nu_z", "iota"],
    },
}
data_index["|B|_mn"] = {
    "label": "B_{mn}^{Boozer}",
    "units": "T",
    "units_long": "Tesla",
    "description": "Boozer harmonics of magnetic field",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"B": [[0, 0, 0]]},
        "profiles": [],
        "data": ["sqrt(g)_B", "|B|", "rho", "theta_B", "zeta_B"],
    },
}
data_index["B modes"] = {
    "label": "Boozer modes",
    "units": "",
    "units_long": "None",
    "description": "Boozer harmonics",
    "fun": "compute_boozer_coordinates",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"B": [[0, 0, 0]]},
        "profiles": [],
        "data": ["|B|_mn"],
    },
}
data_index["f_C"] = {
    "label": "(\\mathbf{B} \\times \\nabla \\psi) \\cdot \\nabla B - "
    + "(M G + N I) / (M \\iota - N) \\mathbf{B} \\cdot \\nabla B",
    "units": "T^{3}",
    "units_long": "Tesla cubed",
    "description": "Two-term quasisymmetry metric",
    "fun": "compute_quasisymmetry_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "iota",
            "psi_r",
            "sqrt(g)",
            "B_theta",
            "B_zeta",
            "|B|_t",
            "|B|_z",
            "G",
            "I",
            "B*grad(|B|)",
        ],
        "kwargs": ["helicity"],
    },
}
data_index["f_T"] = {
    "label": "\\nabla \\psi \\times \\nabla B \\cdot \\nabla "
    + "(\\mathbf{B} \\cdot \\nabla B)",
    "units": "T^{4} \\cdot m^{-2}",
    "units_long": "Tesla quarted / square meters",
    "description": "Triple product quasisymmetry metric",
    "fun": "compute_quasisymmetry_error",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": [
            "psi_r",
            "sqrt(g)",
            "|B|_t",
            "|B|_z",
            "(B*grad(|B|))_t",
            "(B*grad(|B|))_z",
        ],
    },
}

# stability
data_index["D_Mercier"] = {
    "label": "D_{Mercier}",
    "units": "~",
    "units_long": "None",
    "description": "Mercier stability criterion",
    "fun": "compute_mercier_stability",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["D_shear", "D_current", "D_well", "D_geodesic"],
    },
}
data_index["D_shear"] = {
    "label": "D_{shear}",
    "units": "~",
    "units_long": "None",
    "description": "Mercier stability criterion magnetic shear term",
    "fun": "compute_mercier_stability",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {},
        "profiles": [],
        "data": ["iota_r", "psi_r"],
    },
}
data_index["D_current"] = {
    "label": "D_{current}",
    "units": "~",
    "units_long": "None",
    "description": "Mercier stability criterion toroidal current term",
    "fun": "compute_mercier_stability",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [
            "G",
            "iota_r",
            "psi_r",
            "B",
            "J",
            "I_r",
            "sqrt(g)",
            "|grad(rho)|",
            "|grad(psi)|",
        ],
    },
}
data_index["D_well"] = {
    "label": "D_{well}",
    "units": "~",
    "units_long": "None",
    "description": "Mercier stability criterion magnetic well term",
    "fun": "compute_mercier_stability",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [
            "p_r",
            "psi",
            "psi_r",
            "psi_rr",
            "V_rr(r)",
            "V_r(r)",
            "|B|^2",
            "|grad(psi)|",
            "sqrt(g)",
            "|grad(rho)|",
        ],
    },
}
data_index["D_geodesic"] = {
    "label": "D_{geodesic}",
    "units": "~",
    "units_long": "None",
    "description": "Mercier stability criterion geodesic curvature term",
    "fun": "compute_mercier_stability",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["|grad(psi)|", "sqrt(g)", "|grad(rho)|", "J", "B", "|B|^2"],
    },
}
data_index["<B^2>"] = {
    "label": "\\langle B^2 \\rangle",
    "units": "T^2",
    "units_long": "Tesla squared",
    "description": "Flux surface average magnetic field squared",
    "fun": "compute_magnetic_well",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [
            "sqrt(g)",
            "|B|^2",
            "V_r(r)",
        ],
    },
}
data_index["<B^2>_r"] = {
    "label": "\\partial_{\\rho} \\langle B^2 \\rangle",
    "units": "T^2",
    "units_long": "Tesla squared",
    "description": "Flux surface average magnetic field squared, radial derivative",
    "fun": "compute_magnetic_well",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [
            "sqrt(g)",
            "sqrt(g)_r",
            "B",
            "B_r",
            "|B|^2",
            "<B^2>",
            "V_r(r)",
            "V_rr(r)",
        ],
    },
}
data_index["magnetic well"] = {
    "label": "Magnetic Well",
    "units": "~",
    "units_long": "None",
    "description": "Magnetic well proxy for MHD stability",
    "fun": "compute_magnetic_well",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [
            "V(r)",
            "V_r(r)",
            "p_r",
            "<B^2>_r",
        ],
    },
}
# Quantities related to bootstrap current
data_index["<J dot B> Redl"] = {
    "label": "\\langle\\mathbf{J}\\cdot\\mathbf{B}\\rangle_{Redl}",
    "units": "T A m^{-2}",
    "units_long": "Tesla Ampere / meter^2",
    "description": "Bootstrap current profile, Redl model for quasisymmetry",
    "fun": "compute_J_dot_B_Redl",
    "dim": 1,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": [
            "|B|",
            "sqrt(g)",
            "G",
            "I",
            "iota",
        ],
    },
}
data_index["vol avg |B|"] = {
    "label": "vol avg |B|",
    "units": "T",
    "units_long": "Tesla",
    "description": "Volume-averaged (root-mean-square) |B|",
    "fun": "compute_avg_B",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["|B|", "sqrt(g)", "V"],
    },
}
data_index["vol avg beta"] = {
    "label": "vol avg beta",
    "units": "dimensionless",
    "units_long": "dimensionless",
    "description": "Volume average beta",
    "fun": "compute_avg_beta",
    "dim": 0,
    "dependencies": {
        "params": [],
        "transforms": {"grid": []},
        "profiles": [],
        "data": ["p", "sqrt(g)", "V", "vol avg |B|"],
    },
}
