"""Utility functions for JAX models."""

import json
import os
import re
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import orbax.checkpoint


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


def restore_params_and_config(path):
    """Restore params and config from checkpoint."""
    path = os.path.abspath(path)
    hparams_file_path = os.path.join(path, "hparams.json")
    orbax_path = os.path.join(path, "ckpt")

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = checkpointer.restore(
        orbax_path,
        restore_args=jax.tree.map(
            lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray),
            checkpointer.metadata(orbax_path),
        ),
    )

    if os.path.exists(hparams_file_path):
        with open(hparams_file_path) as f:
            restored_hparams = json.load(f)
    else:
        restored_hparams = {}
    return params, restored_hparams


def checkpointing(path, fresh=False, hparams: dict = None):
    """Set up checkpointing at given path.

    Parameters
    ----------
    path : str
        Path to the checkpoint directory.
    fresh : bool, optional
        If True, overwrite existing checkpoint. Default is False.
    hparams : dict, optional
        Hyper-parameters to be saved alongside model params.

    Returns
    -------
    tuple
        A tuple containing:
            - params : PyTree or None
                Restored parameters, or None if no checkpoint found or fresh is True.
            - hparams : dict
                Restored or provided hyper-parameters.
        save_model : Callable
            Function (PyTree -> None) for saving given PyTree.
    """
    path = os.path.abspath(path)
    hparams_file_path = os.path.join(path, "hparams.json")

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_path = os.path.join(path, "ckpt")

    def save_model(_params):
        _params = jax.tree.map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), _params)
        return checkpointer.save(orbax_path, _params, force=True)

    restored_params = None
    restored_hparams = {}
    print(path, end=": ")
    exists = os.path.exists(path)
    if not exists:
        print("No checkpoint found")
    else:
        if fresh:
            print("Overwriting existing checkpoint")
        else:
            restored_params, restored_hparams = restore_params_and_config(path)
            print("Restored checkpoint")

    if (not exists or fresh) and hparams is not None:
        os.makedirs(path, exist_ok=True)
        with open(hparams_file_path, "w") as f:
            json.dump(hparams, f)

    return (restored_params, restored_hparams), save_model


def mse_loss(y_hat, y):
    """Mean squared error."""
    return jnp.mean((y - y_hat) ** 2)


def bce_loss(y_hat, y):
    """Binary cross-entropy."""
    return optax.sigmoid_binary_cross_entropy(y_hat, y)


def mae_loss(y_hat, y):
    """Mean absoluted error."""
    return jnp.mean(jnp.abs(y - y_hat))


def sparsity_log_penalty(x):
    """Penalty to encourage output Sparse Coding.

    See: http://ufldl.stanford.edu/tutorial/unsupervised/SparseCoding/#:~:text=Sparse%20coding%20is%20a%20class,1ai%CF%95i
    """

    def _f(x):
        return jnp.log(1 + jnp.abs(x) ** 2).mean()

    if isinstance(x, jnp.ndarray):
        return _f(x)
    else:
        return jax.tree.reduce(lambda _x, _y: _x + _f(_y), x, initializer=0)


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
