"""RNN wirings for JAX."""

import jax
import jax.numpy as jnp
import jax.random as jrandom


def fully_connected(output_size: int, input_size: int, **_):
    """Create a fully connected mask.

    Args:
        output_size (int): output size
        input_size (int): input size

    Returns:
        array: mask of ones with shape (output, input_size)
    """
    return jnp.ones((output_size, input_size))


def fully_connected_no_self(output_size: int, input_size: int, **_):
    """Like fully_connected but with zeros on the diagonal."""
    mask = fully_connected(output_size, input_size)
    assert output_size < input_size, (
        f"output_size {output_size} unexpectedly larger than input_size {input_size}"
    )
    rem = input_size - output_size
    return mask - jnp.concatenate(
        [jnp.zeros((output_size, rem)), jnp.eye(output_size)], axis=1
    )


def random(output_size: int, input_size: int, key=None, sparsity=0.5, **_):
    """Randomly sparse mask.

    Args:
        output_size (int): laten space size
        input_size (int): input size
        key (_type_, optional): jax random key. Defaults to None.
        sparsity (float, optional): probability for element turning out 0. Defaults to 0.5.

    Returns:
        _type_: mask of ones and zeros with shape (num_units, num_units+input_size)
    """
    if key is None:
        key = jrandom.PRNGKey(0)
    mask = jrandom.bernoulli(key, 1 - sparsity, shape=(output_size, input_size))
    mask = jnp.array(mask, dtype=float)
    return mask


def ncp(
    num_units: int,
    input_size: int,
    output_neurons: int = None,
    key=None,
    sparsity=0.3,
    has_bias=True,
    **_,
):
    """
    Neural Circuit Policies (NCP) wiring.

    Hidden units are ordered as output neurons first, interneurons last.
    Inputs only reach interneurons, and every unit receives only from
    interneurons.

    Parameters
    ----------
    num_units : int
        Number of hidden units.
    input_size : int
        Number of mask columns, laid out as ``[x, h, bias]``.
    output_neurons : int, optional
        Number of output neurons. Defaults to ``num_units // 2``.
    key : PRNGKey, optional
        Random key for sampling connections.
    sparsity : float
        Probability of dropping an allowed connection.
    has_bias : bool
        Whether the last column is a bias column.

    Returns
    -------
    mask : ndarray
        Mask of shape ``(num_units, input_size)``.
    """
    if output_neurons is None:
        output_neurons = num_units // 2
    assert 0 <= output_neurons < num_units, (
        f"output_neurons ({output_neurons}) must be in [0, num_units={num_units})"
    )
    if has_bias:
        input_size -= 1
    n_inputs = input_size - num_units
    interneurons = num_units - output_neurons
    if key is None:
        key = jrandom.PRNGKey(0)
    key_in, key_rec = jrandom.split(key)

    input_mask = jnp.zeros((num_units, n_inputs))
    input_mask = input_mask.at[output_neurons:].set(
        jrandom.bernoulli(key_in, 1 - sparsity, shape=(interneurons, n_inputs))
    )
    recurrent_mask = jnp.zeros((num_units, num_units))
    recurrent_mask = recurrent_mask.at[:, output_neurons:].set(
        jrandom.bernoulli(key_rec, 1 - sparsity, shape=(num_units, interneurons))
    )
    mask = jnp.concatenate([input_mask, recurrent_mask], axis=1)
    if has_bias:
        mask = jnp.concatenate([mask, jnp.ones((num_units, 1))], axis=1)
    return mask


def unconnected(num_units: int, input_size: int, has_bias=True, **_):
    """Unconnected hidden units, fully connected input."""
    if has_bias:
        input_size -= 1
    input_fc = jnp.ones((num_units, input_size - num_units))
    recurrent_diag = jnp.zeros((num_units, num_units))
    mask = jnp.concatenate([input_fc, recurrent_diag], axis=1)
    if has_bias:
        mask = jnp.concatenate([mask, jnp.ones((num_units, 1))], axis=1)
    return mask


def diagonal(num_units: int, input_size: int, has_bias=True, **_):
    """Diagonally connected hidden units, fully connected input."""
    if has_bias:
        input_size -= 1
    input_fc = jnp.ones((num_units, input_size - num_units))
    recurrent_diag = jnp.eye(num_units)
    mask = jnp.concatenate([input_fc, recurrent_diag], axis=1)
    if has_bias:
        mask = jnp.concatenate([mask, jnp.ones((num_units, 1))], axis=1)
    return mask


def block_diagonal(num_units: int, input_size: int, num_blocks: int = 4, **_):
    """Unconnected blocks for hidden units and fully connected input."""
    assert num_units % num_blocks == 0, (
        f"num_units ({num_units}) must be divisible by num_blocks ({num_blocks})"
    )
    input_mask = jnp.ones((num_units, input_size - num_units))
    _blocks = [
        jnp.ones((num_units // num_blocks, num_units // num_blocks))
        for _ in range(num_blocks)
    ]
    return jnp.concatenate([input_mask, jax.scipy.linalg.block_diag(*_blocks)], axis=1)


def make_mask_initializer(wiring_name: str, bias=True, **kwargs):
    """Create a mask for given wiring name."""

    def make_mask(key, shape, dtype):
        mask = globals()[wiring_name](*shape, key=key, **kwargs)
        if bias:
            # Force bias to be visible
            mask = mask.at[:, -1].set(1.0)
        return mask

    return make_mask
