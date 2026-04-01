"""Hopfield Network cell implementation.

Implements both:
- Classical Hopfield Network (Hopfield, 1982): bipolar patterns with a
  sign-based activation, framed as a continuous-time ODE.
- Modern Hopfield Network (Ramsauer et al., 2020): continuous patterns with
  softmax-based retrieval equivalent to the attention mechanism.

Both variants are exposed as ODECell subclasses and follow the same
structural conventions as CTRNNCell in ctrnn.py.

References:
    Hopfield, J. J. (1982). Neural networks and physical systems with emergent
    collective computational abilities. PNAS, 79(8), 2554-2558.

    Ramsauer, H., et al. (2020). Hopfield Networks is All You Need.
    https://arxiv.org/abs/2008.02217
"""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
from chex import PRNGKey

from jax_rtrl.models.cells.ode import ODECell

## Hopfield ODE functions


def classical_hopfield(params, h, x):
    """Compute Classical Hopfield ODE step.

    dh/dt = (-h + sign(W @ [x, h, 1])) / tau

    The weight matrix W combines input (W_in), recurrent (W_rec), and bias
    connections, analogously to the CTRNN parameterization.  The sign
    activation produces bipolar fixed-points.
    """
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    u = y @ params["W"].T
    act = jnp.sign(u)
    return (act - h) / params["tau"]


def modern_hopfield(params, h, x, beta=1.0):
    """Compute Modern Hopfield ODE step.

    dh/dt = (-h + patterns^T softmax(beta * patterns @ W @ [x, h, 1])) / tau

    W projects the concatenated input/state to pattern space as a query.
    The stored patterns serve as both keys and values for the attention
    operation (Ramsauer et al., 2020).
    """
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    query = y @ params["W"].T
    patterns = params["patterns"]
    similarities = beta * (query @ patterns.T)
    act = jax.nn.softmax(similarities, axis=-1) @ patterns
    return (act - h) / params["tau"]


class HopfieldCell(ODECell):
    """Hopfield Network cell.

    Can be configured as a classical (Hopfield, 1982) or modern
    (Ramsauer et al., 2020) Hopfield Network, both framed as
    continuous-time ODEs compatible with the ODECell solver framework.

    In classical mode the cell uses a sign activation analogous to a CTRNN
    with tanh replaced by sign.  In modern mode the cell uses a softmax-based
    update equivalent to single-head attention over a learned pattern matrix.

    The number of ODE integration steps is controlled by the inherited
    ``T`` (total time) and ``dt`` (step size) attributes.

    Attributes:
        num_units: Dimension of the hidden state / pattern space.
        ode_type: ``"classical"`` (Hopfield 1982) or ``"modern"``
            (Ramsauer et al. 2020).
        beta: Inverse temperature for modern mode.  Higher values produce
            sharper, more winner-take-all retrieval.
        num_stored_patterns: Number of stored patterns for modern mode.
            Defaults to ``num_units`` when ``None``.
    """

    ode_type: str = "modern"
    beta: float = 1.0
    num_stored_patterns: int | None = None

    def _make_params(self, x):
        ODECell._make_params(self, x)
        w_shape = (self.num_units, x.shape[-1] + self.num_units + 1)

        def _initializer(key, *_):
            _w_in = nn.initializers.lecun_uniform(in_axis=-1, out_axis=-2)(
                key, (self.num_units, x.shape[-1])
            )
            _w_rec = nn.initializers.lecun_uniform(in_axis=-1, out_axis=-2)(
                key, (self.num_units, self.num_units)
            )
            _bias = jnp.zeros((self.num_units, 1))
            return jnp.concatenate([_w_in, _w_rec, _bias], axis=-1)

        self.param("W", _initializer, w_shape)
        self.param(
            "tau", partial(jrand.uniform, minval=1, maxval=8), (self.num_units,)
        )
        if self.ode_type == "modern":
            num_stored = self.num_stored_patterns or self.num_units
            self.param(
                "patterns",
                nn.initializers.normal(stddev=0.1),
                (num_stored, self.num_units),
            )

    def _f(self, h, x):
        """Compute the derivative of the state."""
        params = self.variables["params"]
        if self.ode_type == "classical":
            return classical_hopfield(params, h, x)
        elif self.ode_type == "modern":
            return modern_hopfield(params, h, x, beta=self.beta)
        else:
            raise ValueError(f"Unknown update_type: {self.ode_type!r}")

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize neuron states."""
        return jnp.zeros(tuple(input_shape)[:-1] + (self.num_units,))

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1
