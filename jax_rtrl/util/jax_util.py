"""Utility functions for JAX models."""

import re
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.linen as nn
import jax.random as jrand
import optax
from pprint import pprint


def pprint_params(params):
    def mask_leaves(d):
        if isinstance(d, dict):
            return {k: mask_leaves(v) for k, v in d.items()}
        return d.shape  # Replace all non-dict leaves

    pprint(mask_leaves(params))


class JaxRng:
    """Base class that facilitates jax PRNG managment."""

    def __init__(self, rng) -> None:
        """Set up internal rng."""
        self._rng = rng

    @property
    def rng(self):
        """Split internal rng."""
        self._rng, rng = jax.jit(jrandom.split)(self._rng)
        return rng


def symlog(x):
    """Symmetric log."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x):
    """Inverse of symmetric log."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def sigmoid_between(x, lower, upper):
    """Map input to sigmoid that goes from lower to upper."""
    return (upper - lower) * jax.nn.sigmoid(x) + lower


@jax.jit
def preprocess_img(img):
    """Make grayscale from RGB Image."""
    import dm_pix as pix

    return pix.rgb_to_grayscale(jnp.array(img / 255.0, dtype=jnp.float32))


@partial(jax.jit, static_argnames=["batch_size"])
def zeros_like_tree(tree, batch_size=None):
    """Create pytree of zeros with batchsize."""
    if batch_size is not None:
        return jax.tree.map(lambda x: jnp.zeros((batch_size,) + x.shape), tree)
    else:
        return jax.tree.map(lambda x: jnp.zeros_like(x), tree)


def mse_loss(y_hat, y):
    """Mean squared error."""
    return jnp.mean((y - y_hat) ** 2)


def bce_loss(y_hat, y):
    """Binary cross-entropy."""
    return optax.sigmoid_binary_cross_entropy(y_hat, y)


def mae_loss(y_hat, y):
    """Mean absoluted error."""
    return jnp.mean(jnp.abs(y - y_hat))


def rbf_kernel(X, gamma=None):
    """Radial Basis Function Kernel."""
    if gamma is None:
        gamma = 1.0 / X.shape[-1]
    XX = (X**2).sum(1)
    XY = X @ X.T
    sq_distances = XX[:, None] + XX - 2 * XY
    return jnp.exp(-gamma * sq_distances)


def g_slow_loss(x_before, x_t, x_next):
    """Change in first derivative error.

    For the start of a trajectory, where x_before = x_t, simply pass in x_before=x_t, x_t=x_t
    For the end of a trajectory, where x_next = x_t, simply pass in x_t=x_t, x_next=x_t
    """
    return jnp.mean((x_next - x_t - (x_t - x_before)) ** 2)


def make_vmap_model(_model, **kwargs):
    """Create a vectorized version of the given model's apply function using JAX's vmap and jit.

    Args:
        _model: The model object which contains the apply function to be vectorized.
        **kwargs: Additional keyword arguments to be passed to the model's apply function.

    Returns:
        A function that takes parameters (_p) and input data (_input), and applies the vectorized model's apply function to the input data.
    """

    @jax.jit
    def _vmap_model(_p, _input):
        return jax.vmap(jax.tree_util.Partial(_model.apply, _p, **kwargs))(_input)

    return _vmap_model


def make_validate(_model, test_data, **kwargs):
    """Create a validation function for a given model and test data.

    Args:
        _model: The model to be validated.
        test_data: The data to be used for validation.
        **kwargs: Additional keyword arguments to be passed to the vmap model creation function.

    Returns:
        A function that takes model parameters as input and returns the mean squared error loss
        between the model predictions and the test data.
    """
    vmap_model = make_vmap_model(_model, **kwargs)

    @jax.jit
    def _validate(_p):
        y_hat, _ = vmap_model(_p, test_data)
        return mse_loss(y_hat, test_data)

    return _validate


def get_keystr(k):
    """Even prettier key string."""

    def _str(_k):
        if hasattr(_k, "key"):
            return _k.key
        return str(_k)

    return "/".join(map(_str, k))


def get_matching_leaves(tree, pattern):
    """Get leaves of tree that match pattern."""
    flattened, _ = jax.tree_util.tree_flatten_with_path(tree)
    flattened = {get_keystr(k): v for k, v in flattened}
    return [flattened[k] for k in flattened if re.fullmatch(pattern, k)]


def get_subtree(tree, prefix):
    """Get subtree of tree that matches pattern."""
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    tree = {get_keystr(k): v for k, v in tree}
    return {k: v for k, v in tree.items() if re.fullmatch(prefix + ".*", k)}


def set_matching_leaves(tree, pattern, new_values):
    """Set leaves of tree that match pattern."""
    flattened, treedef = jax.tree_util.tree_flatten_with_path(tree)
    key_matches = [re.fullmatch(pattern, jax.tree_util.keystr(k)) for k, _ in flattened]
    index = 0
    for i, k in enumerate(key_matches):
        if k:
            flattened[i] = new_values[index]
            index += 1
        else:
            flattened[i] = flattened[i][1]
    return jax.tree_util.tree_unflatten(treedef, flattened)


def get_normalization_fn(norm_type, training=True, **kwargs):
    """Get normalization function based on type."""
    if norm_type is None:
        return lambda x: x
    if norm_type == "layer":
        return nn.LayerNorm(**kwargs)
    elif norm_type == "batch":
        return nn.BatchNorm(
            use_running_average=not training, axis_name="batch", **kwargs
        )
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


def majority_vote(outputs):
    """Return the most popular discrete output."""
    return jnp.argmax(jnp.bincount(outputs, length=outputs.shape[-1]), axis=-2)


def tree_stack(trees, axis=0):
    """Take a list of trees and stack every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function. Taken from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves, treedef = jax.tree.flatten(trees[0])
    leaves_list = [leaves]
    for tree in trees[1:]:
        leaves, _ = jax.tree.flatten(tree)
        leaves_list.append(leaves)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaf, axis=axis) for leaf in grouped_leaves]
    return treedef.unflatten(result_leaves)


def symmetric_uniform_init(lim, dtype=jnp.float_):
    def init(key, shape, dtype=dtype, out_sharding=None):
        # dtype = dtypes.canonicalize_dtype(dtype)
        return jrand.uniform(
            key, shape, dtype, out_sharding=out_sharding, minval=-lim, maxval=lim
        )

    return init
