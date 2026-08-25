"""Multi-head self-attention cells built with Jax.

author: jlemmel
"""

import jax.numpy as jnp
from flax import linen as nn


def resets_to_mask(resets):
    """Build a self-attention mask that blocks attention across episode boundaries.

    A ``True`` value in ``resets[t]`` marks the start of a new episode: position
    ``t`` (and every position up to the next reset) may then only attend to
    positions from ``t`` onward, mirroring the reset semantics of the LRU
    cell's parallel scan (see ``binary_operator_reset`` in ``seq_util.py``).

    Args:
        resets: boolean array of shape (steps,).

    Returns:
        A (1, steps, steps) boolean mask, True where attention is allowed.
    """
    segment_ids = jnp.cumsum(resets.astype(jnp.int32))
    return nn.make_attention_mask(segment_ids, segment_ids, jnp.equal, dtype=bool)


class AttentionCell(nn.Module):
    """Bidirectional multi-head self-attention over a full input sequence.

    Unlike the recurrent cells in this package, attention has no meaningful
    fixed-size hidden state: every output position depends on every input
    position (subject to episode resets). ``carry`` is therefore unused and
    simply passed through unchanged, so the cell can still be dropped in
    wherever a ``(carry, inputs, resets=None) -> (carry, outputs)`` cell is
    expected. Only BPTT (i.e. plain backprop through the attention op) is
    supported -- there is no online/eligibility-trace variant.
    """

    d_hidden: int
    num_heads: int = 1
    causal: bool = False

    def setup(self):
        """Create the underlying multi-head attention module."""
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_hidden,
            out_features=self.d_hidden,
            deterministic=True,
        )

    def __call__(self, carry, inputs, resets=None):
        """Compute self-attention over a sequence.

        Assumes the input is a sequence of shape (steps, input_dim), or a single
        vector of shape (input_dim,) treated as a length-1 sequence.
        """
        _is_sequence = len(inputs.shape) > 1
        x = jnp.reshape(inputs, (-1, inputs.shape[-1]))

        mask = None
        if self.causal:
            mask = nn.make_causal_mask(x[..., 0], dtype=bool)
        if resets is not None:
            reset_mask = resets_to_mask(jnp.reshape(resets, (-1)))
            mask = reset_mask if mask is None else mask & reset_mask

        out = self.attn(x, mask=mask)
        return carry, out if _is_sequence else out[0]

    def initialize_carry(self, rng, input_shape):
        """Attention has no carry, no None."""
        return None

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


class CausalAttentionCell(AttentionCell):
    """Causal multi-head self-attention cell.

    Each position only attends to itself and earlier positions, making it
    suitable for autoregressive/decoder-style use.
    """

    causal: bool = True
