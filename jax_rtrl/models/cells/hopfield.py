"""Hopfield Network layer implementations.

Implements both:
- Classical Hopfield Network (Hopfield, 1982): bipolar patterns with Hebbian
  learning and synchronous sign-based update.
- Modern Hopfield Network (Ramsauer et al., 2020): continuous patterns with
  softmax-based retrieval equivalent to the attention mechanism.

References:
    Hopfield, J. J. (1982). Neural networks and physical systems with emergent
    collective computational abilities. PNAS, 79(8), 2554-2558.

    Ramsauer, H., et al. (2020). Hopfield Networks is All You Need.
    https://arxiv.org/abs/2008.02217
"""

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp


def hebbian_weights(patterns: jnp.ndarray) -> jnp.ndarray:
    """Compute Hebbian weight matrix from stored bipolar patterns.

    W = (1/N) * sum_i p_i p_i^T,  with zero diagonal.

    Args:
        patterns: Stored bipolar patterns of shape (num_patterns, num_units).

    Returns:
        Symmetric weight matrix of shape (num_units, num_units).
    """
    num_patterns = patterns.shape[0]
    W = jnp.einsum("pi,pj->ij", patterns, patterns) / num_patterns
    return W * (1 - jnp.eye(W.shape[0]))


def classical_update(x: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    """One synchronous update step of the classical Hopfield network.

    Computes the sign of the weighted sum of the current state.

    Args:
        x: Current bipolar state, shape (..., num_units).
        W: Symmetric weight matrix, shape (num_units, num_units).

    Returns:
        Updated bipolar state (values in {-1, +1}), same shape as x.
    """
    return jnp.where(x @ W.T >= 0, 1.0, -1.0)


def modern_update(
    x: jnp.ndarray, patterns: jnp.ndarray, beta: float = 1.0
) -> jnp.ndarray:
    """One update step of the modern Hopfield network.

    Implements the fixed-point update rule from Ramsauer et al. (2020)::

        x_new = patterns^T * softmax(beta * patterns * x)

    This update is equivalent to a single-head attention operation where
    the query is the current state and keys/values are the stored patterns.

    Args:
        x: Query state, shape (..., num_units).
        patterns: Stored patterns, shape (num_patterns, num_units).
        beta: Inverse temperature.  Higher values produce sharper (more
            winner-take-all) retrieval.

    Returns:
        Updated state, shape (..., num_units).
    """
    similarities = beta * (x @ patterns.T)
    attn = jax.nn.softmax(similarities, axis=-1)
    return attn @ patterns


class HopfieldLayer(nn.Module):
    """Hopfield Network layer.

    Can be configured as a classical (Hopfield, 1982) or modern
    (Ramsauer et al., 2020) Hopfield Network.

    **Classical mode** uses Hebbian learning to store bipolar {-1, +1}
    patterns and a synchronous sign-based update rule for retrieval.
    Patterns may be supplied at call time (via the ``patterns`` argument)
    or the weight matrix may be learned as a parameter.

    **Modern mode** uses a softmax-based update rule (equivalent to
    single-head attention), enabling continuous states and exponentially
    higher storage capacity.  Stored patterns may be passed at call time
    or learned as parameters.

    Attributes:
        num_units: Dimension of the state / pattern space.
        mode: ``"classical"`` (Hopfield 1982) or ``"modern"``
            (Ramsauer et al. 2020).
        beta: Inverse temperature used in modern mode.  Higher values
            produce sharper, more winner-take-all retrieval.
        num_steps: Number of iterative retrieval steps.
        num_stored_patterns: Number of learnable stored patterns in modern
            mode.  Ignored when ``patterns`` is provided at call time.
            Defaults to ``num_units`` when ``None``.
    """

    num_units: int
    mode: Literal["classical", "modern"] = "modern"
    beta: float = 1.0
    num_steps: int = 1
    num_stored_patterns: int | None = None

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, patterns: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Perform Hopfield retrieval.

        Args:
            x: Query state of shape ``(..., num_units)``.
            patterns: Optional stored patterns to use during retrieval.

                - Classical mode: bipolar {-1, +1} array of shape
                  ``(num_patterns, num_units)``.
                - Modern mode: continuous array of shape
                  ``(num_patterns, num_units)``.

                If ``None``, learned parameters are used (``W`` for
                classical mode, ``patterns`` for modern mode).

        Returns:
            Retrieved state of shape ``(..., num_units)``.
        """
        if self.mode == "classical":
            return self._classical_retrieve(x, patterns)
        return self._modern_retrieve(x, patterns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classical_retrieve(
        self, x: jnp.ndarray, patterns: jnp.ndarray | None
    ) -> jnp.ndarray:
        """Classical Hopfield retrieval with synchronous sign updates."""
        if patterns is not None:
            W = hebbian_weights(patterns)
        else:
            W_raw = self.param(
                "W",
                nn.initializers.glorot_uniform(),
                (self.num_units, self.num_units),
            )
            W = (W_raw + W_raw.T) / 2
            W = W * (1 - jnp.eye(self.num_units))

        h = x
        for _ in range(self.num_steps):
            h = classical_update(h, W)
        return h

    def _modern_retrieve(
        self, x: jnp.ndarray, patterns: jnp.ndarray | None
    ) -> jnp.ndarray:
        """Modern Hopfield retrieval with softmax updates."""
        if patterns is None:
            num_stored = self.num_stored_patterns or self.num_units
            patterns = self.param(
                "patterns",
                nn.initializers.normal(stddev=0.1),
                (num_stored, self.num_units),
            )

        h = x
        for _ in range(self.num_steps):
            h = modern_update(h, patterns, self.beta)
        return h
