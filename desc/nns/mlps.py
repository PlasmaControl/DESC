import abc
from typing import Sequence
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from desc.equilibrium import Equilibrium


class SomeNN(nn.Module):
    apply_scale: bool  # whether to use scaling

    @staticmethod
    @abc.abstractmethod
    def create_input(eq: Equilibrium) -> jax.Array:
        """

        Args:
            eq: desc.equilibrium.Equilibrium
        Returns: the input of same shape as input which can be used to init the model

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def configure(eq: Equilibrium, module_config: dict) -> dict:
        """
        Take module_config and configure rest
        """
        raise NotImplementedError


class MultiMLP(SomeNN):
    """
    three MLP's one for each of Rlmn, llmn, Zlmn
        * Rb_lmn -> R_lmn
        * random_uniform_like(Zb_lmn) * 0.05 -> L_lmn
        * Zb_lmn -> Z_lmn

    """

    rlmn_layers: Sequence[int]
    llmn_layers: Sequence[int]
    zlmn_layers: Sequence[int]

    # projection of deterministic initial guess from DESC
    x_init: jax.Array

    # projection of scaling factors
    x_scale: jax.Array

    # shape of y prediction which desc handles s.t.
    # boundary is always satisfied
    dimy: int
    name: str = "multiMLP"

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        # assert inputs.ndim inputs.shape ...
        out = []
        x_r = inputs[0]
        x_l = inputs[1]
        x_z = inputs[2]
        for ident, x in [("rlmn", x_r), ("llmn", x_l), ("zlmn", x_z)]:
            dep_layers = getattr(self, f"{ident}_layers")
            for j, feat in enumerate(dep_layers):
                if ident == "rlmn":
                    kernel_init = nn.initializers.normal(stddev=1e-2)
                elif ident == "zlmn":
                    kernel_init = nn.initializers.normal(stddev=1e-2)
                else:  # llmn
                    kernel_init = nn.initializers.normal(stddev=1e-2)

                x = nn.Dense(
                    feat,
                    name=f"{ident}_layer_{j}",
                    use_bias=True,
                    kernel_init=kernel_init,
                    dtype=x.dtype,
                )(x)

                if j != len(dep_layers) - 1:  # last layer has no activation
                    act_fn = nn.tanh  # make configurable
                    x = act_fn(x)
            out.append(x)
        out = jnp.concat((out[0], out[1], out[2]))

        # add spectral initial guess and opt. scaling
        if self.apply_scale:
            out = out * self.x_scale + self.x_init
        else:
            out = out + self.x_init
        return out

    @staticmethod
    def create_input(eq: Equilibrium, seed: int = 42) -> list:
        module_input = [
            eq.Rb_lmn,
            random.uniform(key=random.PRNGKey(seed), shape=eq.Zb_lmn.shape) * 0.05,
            eq.Zb_lmn,
        ]

        return module_input

    @staticmethod
    def configure(eq: Equilibrium, module_config: dict) -> dict:
        dimr = eq.R_lmn.size - eq.Rb_lmn.size
        dimz = eq.Z_lmn.size - eq.Zb_lmn.size
        diml = eq.L_lmn.size
        dimy = dimr + dimz + diml

        mlp_layers = module_config.pop("mlp_layers")
        module_config.update(
            {
                "name": MultiMLP.name,
                "rlmn_layers": [*mlp_layers, dimr],
                "llmn_layers": [*mlp_layers, diml],
                "zlmn_layers": [*mlp_layers, dimz],
                "dimy": dimy,
            }
        )

        return module_config


class SingleMLP(SomeNN):
    """
    one single MLP one for each of Rlmn, llmn, Zlmn

    * flattened(Rb_lmn, random_uniform_like(Zb_lmn)*0.05, Zb_lmn)
        -> flattened(R_lmn, L_lmn, Z_lmn)

    this is done because objective.x() might shuffle
      the coefficients so we need to relax
    the explicit connection between boundary and whole-volume modes
    """

    all_layers: Sequence[int]
    # deterministic initial guess from DESC
    x_init: jax.Array
    # projection of scaling factors
    x_scale: jax.Array
    dimy: int
    name: str = "singleMLP"

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        x = inputs
        for j, feat in enumerate(self.all_layers):
            kernel_init = nn.initializers.normal(stddev=1e-2)

            x = nn.Dense(
                feat,
                name=f"layer_{j}",
                use_bias=True,
                kernel_init=kernel_init,
                dtype=x.dtype,
            )(x)

            if j != len(self.all_layers) - 1:  # last layer has no activation
                act_fn = nn.tanh  # make configurable
                x = act_fn(x)

        # add spectral initial guess
        if self.apply_scale:
            out = x * self.x_scale + self.x_init
        else:
            out = x + self.x_init
        return out

    @staticmethod
    def create_input(eq: Equilibrium, seed: int = 42) -> jax.Array:
        module_input = jnp.concat(
            (
                eq.Rb_lmn,
                random.uniform(key=random.PRNGKey(seed), shape=eq.Zb_lmn.shape) * 0.05,
                eq.Zb_lmn,
            )
        )
        return module_input

    @staticmethod
    def configure(eq: Equilibrium, module_config: dict) -> dict:
        dimr = eq.R_lmn.size - eq.Rb_lmn.size
        dimz = eq.Z_lmn.size - eq.Zb_lmn.size
        diml = eq.L_lmn.size
        dimy = dimr + dimz + diml

        mlp_layers = module_config.pop("mlp_layers")
        module_config.update(
            {
                "name": SingleMLP.name,
                "all_layers": [*mlp_layers, dimy],
                "dimy": dimy,
            }
        )
        return module_config


# class stackedMLP(SomeNN):  # predict lower order modes and
# higher order modes with separate nns todo
