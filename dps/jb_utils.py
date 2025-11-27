from desc.backend import jnp
import matplotlib.pyplot as plt
from desc.grid import Grid
import desc.io

def set_mu(psi_i, theta_i, zeta_i, vpar_i, eq, Energy_SI, Mass):
    grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
    data = eq.compute("|B|", grid=grid)
    mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])
    return mu

def output_to_file(sol, name):
    print(sol.shape)
    list1 = sol[:, 0]
    list2 = sol[:, 1]
    list3 = sol[:, 2]
    list4 = sol[:, 3]

    combined_lists = zip(list1, list2, list3, list4)
    
    file_name = f'{name}.txt'

    with open(file_name, 'w') as file:        
        for row in combined_lists:
            row_str = '\t'.join(map(str, row))
            file.write(row_str + '\n')


def savetxt(data, name):
    print(data.shape)

    combined_lists = zip(data)
    
    file_name = f'{name}.txt'

    with open(file_name, 'w') as file:        
        for row in combined_lists:
            row_str = '\t'.join(map(str, row))
            file.write(row_str + '\n')

def Trajectory_Plot(solution, save_name="Trajectory_Plot.png"):
    fig, ax = plt.subplots()
    ax.plot(jnp.sqrt(solution[:, 0]) * jnp.cos(solution[:, 1]), jnp.sqrt(solution[:, 0]) * jnp.sin(solution[:, 1]))
    ax.set_aspect("equal", adjustable='box')
    plt.xlabel(r'$\sqrt{\psi}cos(\theta)$')
    plt.ylabel(r'$\sqrt{\psi}sin(\theta)$')
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Trajectory Plot Saved: {save_name}")