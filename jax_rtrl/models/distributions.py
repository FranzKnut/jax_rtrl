"""Distributions implemented with distrax."""

import distrax
import jax


class UniformMixture(distrax.MixtureSameFamily):
    """A uniform mixture of distributions."""

    def __init__(self, components):
        # HACK: Distrax MixtureSameFamily expects batch_dim in the last dimension? so we swap axes here.
        components = jax.tree.map(lambda d: jax.numpy.swapaxes(d, 0, -1), components)
        super().__init__(
            distrax.Categorical(logits=jax.numpy.zeros(components.batch_shape)),
            components,
        )

    def mode(self):
        """Return the mode of the mixture distribution."""
        return self.mean()


class NormalTanh(distrax.Transformed):
    """A Normal distribution followed by a Tanh transformation."""

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._normal, name)

    def __init__(self, loc, scale):
        self._normal = distrax.Normal(loc, scale)
        super().__init__(
            self._normal,
            # distrax.Independent(self._normal, 1),
            distrax.Tanh(),
            # distrax.Block(distrax.Tanh(), 1),
        )

    def mode(self) -> jax.Array:
        sample = self.distribution.mode()
        return self.bijector.forward(sample)

    def entropy(self) -> jax.Array:
        print("WARNING: Using base distribution's entropy in place of the true one!")
        return self.distribution.entropy()

    def variance(self) -> jax.Array:
        print("WARNING: Using base distribution's variance in place of the true one!")
        return self.distribution.variance()
