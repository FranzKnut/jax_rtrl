"""Distributions implemented with distrax."""

from typing import Any

from chex import PRNGKey
import distrax
import jax
import jax.numpy as jnp


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
        try:
            return self.mean()
        except NotImplementedError:
            print(
                "WARNING: Mean is not implemented for this mixture distribution. Using mean of modes."
            )
            return jax.numpy.mean(self.components_distribution.mode(), axis=-1)


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


class JointVectorDist(distrax.Joint):
    """A joint distribution where events are concatenated to be vectors."""

    def log_prob(self, value):
        """Compute joint log probability."""
        return super().log_prob(jnp.split(value, value.shape[-1], axis=-1))

    def variance(self) -> Any:
        """Compute joint variance."""
        def _variance(leaf):
            if hasattr(leaf, "variance"):
                try:
                    return leaf.variance()
                except NotImplementedError:
                    pass
            if hasattr(leaf, "distribution") and hasattr(leaf.distribution, "variance"):
                try:
                    return leaf.distribution.variance()
                except NotImplementedError:
                    pass
            else:
                raise NotImplementedError(
                    f"Variance not implemented for distribution {type(leaf)}"
                )
        return jnp.stack([_variance(leaf) for leaf in self._distributions], axis=-1)

    def sample(self, seed: PRNGKey, sample_shape: tuple[int, ...] = ()):
        """Sample from the joint distribution."""
        return jnp.stack(
            distrax.Joint.sample(self, seed=seed, sample_shape=sample_shape), axis=-1
        )
