import jax
from desc.backend import jnp, put
import numpy as np
def tree2arr(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    idx = np.cumsum([foo.size for foo in leaves])
    return np.concatenate(leaves), treedef, idx[:-1]

def arr2tree(arr, treedef, idx):
    leaves = np.split(arr, idx)
    print("leaves are " + str(leaves))
    return jax.tree_util.tree_unflatten(treedef, leaves)


a = jnp.array([1.0])
b = jnp.array([2.0, 3.0, 4.0])
c = jnp.array([5.0, 6.0])

d = (a,b,c)
print("d is " + str(d))

idx = jnp.zeros(len(d)).astype('int')
count = 0
for i in range(len(d)):
    count = count + len(d[i])
    idx = put(idx,i,count)

print("idx is " + str(idx))

arr = tree2arr(d)
print("arr is " + str(arr))

e = arr2tree(arr[0],arr[1],arr[2])
print("e is " + str(e))

for i in range(len(e)):
    print(jnp.array(e[i]))
