import jax
import jax.numpy as jnp
import time

# Create large test matrix
A = jnp.ones((31000, 10000), dtype=jnp.float32)

# Force CPU device (optional)
A = jax.device_put(A, jax.devices("cpu")[0])

start = time.time()
A_pinv = jnp.linalg.pinv(A).block_until_ready()
print(f"Computed pseudo-inverse in {time.time() - start:.2f} seconds")
