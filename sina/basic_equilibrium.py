import os
import pathlib
import desc.equilibrium
import desc.plotting
import desc.profiles
import desc.geometry
import desc.continuation

# ======================================================================================
# The shape of the last-closed-flux-surface (LCFS) of the plasma.
# ======================================================================================
# Q: Is there no plasma outside the LCFS?
# Q: Is a surface object from DESC, always a 2D surface in 3D space? Or is it just a 1D
#    curve in 2D space, which is a cross-section of a 3D plasma?
# Q: Can I plot the LCFS easily to see what it looks like?
# Q: What does number of field periods (NFP) mean?
# ======================================================================================
last_closed_flux_surface = desc.geometry.FourierRZToroidalSurface(
    # Fourier coefficients for R in cylindrical coordinates:
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    # Fourier coefficients for Z in cylindrical coordinates:
    Z_lmn=[1, -0.3, -0.3],
    # Poloidal and toroidal mode numbers [m,n] for R_lmn:
    modes_R=[
        (0, 0),
        (1, 0),
        (1, 1),
        (-1, -1),
    ],
    # Poloidal and toroidal mode numbers [m,n] for Z_lmn: Q: Is this statement correct?
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    # Number of field periods (toroidal mode number) for the surface:
    NFP=19,
)
# ======================================================================================
# ======================================================================================


# ======================================================================================
# Pressure profile is the pressure of the plasma as a function of the flux surface
# label, rho, which can be any real scalar value between 0 and 1.
# ======================================================================================
# Q: Why pressure is only a function of rho? How come it's not 2D or 3D?
# ======================================================================================

# pressure_profile(x) = 1.8e4 - 3.6e4 x^2 + 1.8e4 x^4
pressure_profile = desc.profiles.PowerSeriesProfile([1.8e4, 0, -3.6e4, 0, 1.8e4])

# ======================================================================================
# ======================================================================================


# ======================================================================================
# Rotational transform is the average number of times a magnetic field line wraps
# around the torus poloidally per toroidal turn.
# ======================================================================================

# rotational_transform(x) = 1 + 1.5 x^2
rotational_transform = desc.profiles.PowerSeriesProfile([1, 0, 1.5])

# ======================================================================================
# ======================================================================================


def solve_the_equilibrium(file_name: str = "equilibrium"):
    # ======================================================================================
    # Answer this question: What is the unique magnetic field in 3D space that will hold
    # the plasma with the given LCFS, pressure profile, and rotational transform in
    # equilibrium?
    # ======================================================================================
    # Q:
    # ======================================================================================
    equilibrium = desc.equilibrium.Equilibrium(
        L=8,  # radial resolution
        M=8,  # poloidal resolution
        N=3,  # toroidal resolution
        surface=last_closed_flux_surface,
        pressure=pressure_profile,
        iota=rotational_transform,
        # the total toroidal magnetic field flux through the surface enclosed by the LCFS
        # in Wb: Q: Is this statement correct?
        Psi=1.0,
    )

    family_of_equilibria: desc.equilibrium.EquilibriaFamily = (
        desc.continuation.solve_continuation_automatic(equilibrium, verbose=3)
    )
    assert isinstance(family_of_equilibria, desc.equilibrium.EquilibriaFamily)

    the_best_equilibrium = family_of_equilibria[-1]
    the_best_equilibrium.save(file_name)

    equilibra_family = desc.continuation.solve_continuation_automatic(equilibrium)
    the_best_equilibrium = family_of_equilibria[-1]
    the_best_equilibrium.save(file_name)

    return the_best_equilibrium

    # ======================================================================================
    # ======================================================================================


def plot_the_equilibrium(equilibrium: desc.equilibrium.Equilibrium):
    desc.plotting.plot_2d(equilibrium, "flux_surfaces")


def read_equilibrium_from_file(file_name: str):
    return desc.equilibrium.Equilibrium.load(file_name)


if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.absolute())
    equilibrium = solve_the_equilibrium("equilibrium.pkl")
    # equilibrium = read_equilibrium_from_file("equilibrium.pkl")
    plot_the_equilibrium(equilibrium)
