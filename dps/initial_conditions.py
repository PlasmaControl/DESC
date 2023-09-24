import jax.numpy as jnp
import jax.random
import time
import matplotlib.pyplot as plt
import jax

def histplot(hist):
    plt.hist(hist)
    plt.show()

def starting_ensamble(size):
    key = jax.random.PRNGKey(int(time.time()))

    v_init = jax.random.maxwell(key, (size,))

    psi_init = jax.random.uniform(key, (size,), minval=1e-4, maxval=1-1e-4)
    zeta_init = 0.2
    theta_init = 0.2

    ini_cond = [[float(psi_init[i]), theta_init, zeta_init, float(v_init[i])] for i in range(0, size)]
    return ini_cond

a = starting_ensamble(10)
print(a)
i=10

print(f"output_{i}.png")
