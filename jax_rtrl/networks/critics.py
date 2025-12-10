from dataclasses import field
import flax.linen as nn

from jax_rtrl.models.feedforward import MLP, FADense
from jax_rtrl.util.jax_util import get_normalization_fn


class Critic(nn.Module):
    """Critic network."""

    layers: list[int] = field(default_factory=list)
    f_align: bool = False
    norm: str | None = None  # Normalization type, e.g., "layer", "batch"

    @nn.compact
    def __call__(self, x, training=True):
        """Compute value from latent."""
        if self.layers:
            x = MLP(
                self.layers,
                f_align=self.f_align,
                name="mlp",
                norm=self.norm,
            )(x)
        x = get_normalization_fn(self.norm, training=training)(x)

        return FADense(
            1,
            # kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="critic_head",
        )(x)
