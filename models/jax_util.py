"""Utility functions for JAX models."""
from functools import partial
import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
import orbax.checkpoint
from jax.tree_util import tree_map, tree_reduce
import jax.tree_util as jtu
import dm_pix as pix


class JAX_RNG:
    """Base class that facilitates jax PRNG managment."""

    def __init__(self, rng) -> None:
        """Set up internal rng."""
        self._rng = rng

    @property
    def rng(self):
        """Split internal rng."""
        self._rng, rng = jrandom.split(self._rng)
        return rng


def preprocess_img(img):
    """Make grayscale from RGB Image."""
    return pix.rgb_to_grayscale(jnp.array(img/255.0, dtype=jnp.float32))


def tree_norm(tree):
    """Sum of the norm of all elements in the tree."""
    return tree_reduce(lambda x, y: x + jnp.linalg.norm(y), tree, initializer=0)


def leaf_norms(tree):
    """Return Dict of leaf names and their norms."""
    flattened, _ = jtu.tree_flatten_with_path(tree)
    flattened = {jtu.keystr(k): v for k, v in flattened}
    return {k: tree_reduce(lambda x, y: x + jnp.linalg.norm(y), v, initializer=0)
            for k, v in flattened.items()}


@partial(jax.jit, static_argnames=['batch_size'])
def zeros_like_tree(tree, batch_size=None):
    """Create pytree of zeros with batchsize."""
    if batch_size is not None:
        return tree_map(lambda x: jnp.zeros((batch_size,) + x.shape), tree)
    else:
        return tree_map(lambda x: jnp.zeros_like(x), tree)


def checkpointing(path, fresh=False):
    """Set up checkpointing at given path.

    Returns:
        params : PyTree
                Restored parameters or None if no checkpoint found or fresh is True.
        save_model : Callable
                Function (PyTree->None) for saving given PyTree
    """
    path = os.path.abspath(path)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    def save_model(_params): return checkpointer.save(path, _params, force=True) if save_model else None
    params = None
    if not os.path.exists(path):
        print("No checkpoint found.")
    else:
        if fresh:
            print("Overwriting existing checkpoint")
        else:
            params = checkpointer.restore(path)
            print("Restored model from checkpoint")
    return params, save_model
