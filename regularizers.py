"""Regularization functions for RNNs."""
from jax import numpy as jnp


def norm_preserving(de_dh, dh_dh):
    """Incentivice preserving the norm in the direction of the error gradient (Pascanu, 2012)."""
    return (jnp.linalg.norm(de_dh*dh_dh)/jnp.linalg.norm(de_dh)-1)**2
