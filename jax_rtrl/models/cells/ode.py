from dataclasses import field
from typing import Literal
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_rl_util.util.logging_util import tree_stack

from jax_rtrl.models.wirings import make_mask_initializer


class ODECell(nn.RNNCellBase):
    """Simple CTRNN cell."""

    num_units: int
    dt: float = 1.0
    T: float = 1.0
    solver: Literal["euler"] = "euler"
    return_sequences: bool = False
    wiring: str | None = None
    wiring_kwargs: dict = field(default_factory=dict)

    def _f(self, h, x):
        """Compute the derivative of the state."""
        raise NotImplementedError("_f must be implemented in subclasses.")

    def _make_params(self, x):
        """Create parameters for ODECell."""
        # Define params
        w_shape = (self.num_units, x.shape[-1] + self.num_units + 1)
        if self.wiring is not None:
            print("Using wiring:", self.wiring)
            self.variable(
                "wiring",
                "mask",
                make_mask_initializer(self.wiring, **self.wiring_kwargs),
                self.make_rng() if self.has_rng("params") else None,
                w_shape,
                int,
            ).value

    def solve(self, h, x):
        """Solve ODE over time T with step dt."""
        outs = []
        if self.solver == "euler":
            # Euler integration steps with dt
            for _step in jnp.arange(0, self.T, self.dt):
                h_dot = self._f(h, x)
                h = jax.tree.map(lambda a, b: a + b * self.dt, h, h_dot)
                if self.return_sequences:
                    outs.append(h)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        if self.return_sequences:
            return tree_stack(outs)
        return h

    @nn.compact
    def __call__(self, h, x):  # noqa
        """Call ODE solver."""
        if h is None:
            h = self.initialize_carry(self.make_rng(), x.shape)

        self._make_params(x)

        # Solve
        out = self.solve(h, x)
        return out, out

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1
