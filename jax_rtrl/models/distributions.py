"""Distributions implemented with distrax."""

import distrax
import jax


class NormalTanh(distrax.Transformed):
    """A Normal distribution followed by a Tanh transformation."""
    
    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._normal, name)

    def __init__(self, loc, scale):
        self._normal = distrax.Normal(loc, scale)
        super().__init__(
            distrax.Independent(self._normal, 1),
            distrax.Block(distrax.Tanh(), 1),
        )

    def mode(self) -> jax.Array:
        raw_sample = self.distribution.mode()
        return self.bijector.forward(raw_sample)

    def entropy(self) -> jax.Array:
        print("WARNING: Using base distribution's entropy in place of the true one!")
        return self.distribution.entropy()
