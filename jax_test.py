import jax


class Objective:
    def __init__(self, c):
        self.c = c
        self.compute = jax.jit(self.compute)

    def compute(self, x, c=None):
        if c is None:
            c = self.c
        return c * x

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        self._c = c


objective = Objective(2)
x = 1

f0 = objective.compute(x)

objective.c = 3

f1 = objective.compute(x)
f2 = objective.compute(x, objective.c)

print("f0 = {}".format(f0))
print("f1 = {}".format(f1))
print("f2 = {}".format(f2))
