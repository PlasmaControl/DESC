import os
import pathlib
import desc.equilibrium
import desc.objectives
import desc.optimize
import desc.plotting
import desc.profiles
import desc.grid
import desc.geometry
import desc.continuation
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================================
# The shape of the last-closed-flux-surface (LCFS) of the plasma.
# ======================================================================================
# Q: Is there no plasma outside the LCFS?
# A:
# Q: Is a surface object from DESC, always a 2D surface in 3D space? Or is it just a 1D
#    curve in 2D space, which is a cross-section of a 3D plasma?
# A:
# Q: Can I plot the LCFS easily to see what it looks like?
# A:
# Q: What does number of field periods (NFP) mean?
# A:
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

    return the_best_equilibrium

    # ======================================================================================
    # ======================================================================================


def read_equilibrium_from_file(file_name: str) -> desc.equilibrium.Equilibrium:
    return desc.equilibrium.Equilibrium.load(file_name)


if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.absolute())
    # equilibrium = solve_the_equilibrium("equilibrium.pkl")
    equilibrium = read_equilibrium_from_file("equilibrium.pkl")

    # # plot modes of |B| in Boozer coordinates
    # desc.plotting.plot_boozer_modes(equilibrium, num_modes=8, rho=10)

    # # plot |B| contours in Boozer coordinates on a surface (default is rho=1)
    # desc.plotting.plot_boozer_surface(equilibrium)

    # # plot normalized QS metrics
    # desc.plotting.plot_qs_error(equilibrium, helicity=(1, equilibrium.NFP), rho=10)

    # Optimize the equilibrium to minimize the triple-product quasi-symmetry objective
    # function. This will return a new equilibrium object.
    original_equilibrium = equilibrium.copy()  # make a copy of the original one
    optimizer = desc.optimize.Optimizer("proximal-lsq-exact")

    # We will adjust Rcc, Rss, Zsc, and Zcs Fourier coefficients to minimize the
    # quasi-symmetry objective function.
    index_of_Rcc = equilibrium.surface.R_basis.get_idx(M=1, N=2)
    index_of_Rss = equilibrium.surface.R_basis.get_idx(M=-1, N=-2)
    index_of_Zsc = equilibrium.surface.Z_basis.get_idx(M=-1, N=2)
    index_of_Zcs = equilibrium.surface.Z_basis.get_idx(M=1, N=-2)

    # boundary modes to constrain
    # Q: Why do we delete stuff?
    R_modes = np.delete(
        equilibrium.surface.R_basis.modes, [index_of_Rcc, index_of_Rss], axis=0
    )
    Z_modes = np.delete(
        equilibrium.surface.Z_basis.modes, [index_of_Zsc, index_of_Zcs], axis=0
    )

    # constraints
    constraints = (
        desc.objectives.ForceBalance(
            eq=equilibrium
        ),  # enforce JxB-grad(p)=0 during optimization
        desc.objectives.FixBoundaryR(
            eq=equilibrium, modes=R_modes
        ),  # fix specified R boundary modes
        desc.objectives.FixBoundaryZ(
            eq=equilibrium, modes=Z_modes
        ),  # fix specified Z boundary modes
        desc.objectives.FixPressure(eq=equilibrium),  # fix pressure profile
        desc.objectives.FixIota(eq=equilibrium),  # fix rotational transform profile
        desc.objectives.FixPsi(eq=equilibrium),  # fix total toroidal magnetic flux
    )
    grid_vol = desc.grid.ConcentricGrid(
        L=equilibrium.L_grid,
        M=equilibrium.M_grid,
        N=equilibrium.N_grid,
        NFP=equilibrium.NFP,
        sym=equilibrium.sym,
    )

    objective_to_minimize = desc.objectives.ObjectiveFunction(
        desc.objectives.QuasisymmetryTripleProduct(eq=equilibrium, grid=grid_vol)
    )

    equilibrium, optimize_result = equilibrium.optimize(
        objective=objective_to_minimize,
        constraints=constraints,
        optimizer=optimizer,
        ftol=5e-2,  # stopping tolerance on the function value
        xtol=1e-6,  # stopping tolerance on the step size
        gtol=1e-6,  # stopping tolerance on the gradient
        maxiter=50,  # maximum number of iterations
        options={
            "perturb_options": {
                "order": 2,
                "verbose": 0,
            },  # use 2nd-order perturbations
            "solve_options": {
                "ftol": 5e-3,
                "xtol": 1e-6,
                "gtol": 1e-6,
                "verbose": 0,
            },  # for equilibrium subproblem
        },
        copy=False,  # copy=False we will overwrite the eq_qs_T object with the optimized result
        verbose=3,
    )

    desc.plotting.plot_comparison(
        [original_equilibrium, equilibrium], labels=["Original", "Optimized"]
    ) # Q: No difference between the two plots?
    plt.show()
