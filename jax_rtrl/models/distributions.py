"""Distributions implemented with distrax."""

import distrax
import jax


class NormalTanh(distrax.Transformed):
    """A Normal distribution followed by a Tanh transformation."""

    def __init__(self, loc, scale):
        super().__init__(
            distrax.Independent(distrax.Normal(loc, scale), 1),
            distrax.Block(distrax.Tanh(), 1),
        )

    def mode(self) -> jax.Array:
        raw_sample = self.distribution.mode()
        return self.bijector.forward(raw_sample)

    def entropy(self) -> jax.Array:
        print("WARNING: Using base distribution's entropy in place of the true one!")
        return self.distribution.entropy()
