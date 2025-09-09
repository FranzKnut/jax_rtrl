"""Regularization functions for RNNs."""

import jax.numpy as jnp
import jax


def norm_preserving(de_dh, dh_dh):
    """Incentivice preserving the norm in the direction of the error gradient (Pascanu, 2012)."""
    return (jnp.linalg.norm(de_dh * dh_dh) / jnp.linalg.norm(de_dh) - 1) ** 2


def _maybe_reduce_tree(fun, tree):
    """Apply a function to a JAX tree or a single value."""
    if isinstance(tree, jnp.ndarray):
        return fun(tree)
    else:
        return jax.tree.reduce(lambda _x, _y: _x + fun(_y), tree, initializer=0)


def R_SLNI(H, sigma: float | None = None):
    """
    Sparse Coding through Local Neural Inhibition (SLNI).
    (Aljundi, Rohrbach and Tuytelaars, 2019)

    Args:
        H (jnp.ndarray): Shape (..., N), where
                         N = number of hidden units,
        sigma (float): Gaussian width parameter.

    Returns:
        float: The computed R_SLNI value.
    """

    def _f(x):
        N = x.shape[-1]
        _sigma = sigma or (N / 6)
        # Compute Gaussian weights for all i, j
        idx = jnp.arange(N)
        diff = idx[:, None] - idx[None, :]  # shape (N, N)
        weights = jnp.exp(-(diff**2) / (2 * _sigma**2))  # shape (N, N)

        # Dot products between all pairs: H H^T
        x = jnp.abs(x)
        dot_products = jax.vmap(jax.vmap(jnp.outer))(x, x)  # shape (N, N), entries are h_i * h_j

        # Exclude diagonal terms (i != j)
        mask = 1 - jnp.eye(N)

        # Apply mask, weights, and normalize
        R = (weights * dot_products * mask).sum()
        return R

    return _maybe_reduce_tree(_f, H)


def sparsity_log_penalty(H):
    """Penalty to encourage output Sparse Coding.

    See: http://ufldl.stanford.edu/tutorial/unsupervised/SparseCoding/#:~:text=Sparse%20coding%20is%20a%20class,1ai%CF%95i
    """

    def _f(x):
        return jnp.log(1 + jnp.abs(x) ** 2).mean()

    return _maybe_reduce_tree(_f, H)
