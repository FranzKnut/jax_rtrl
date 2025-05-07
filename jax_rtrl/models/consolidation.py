"""Weight consolidation for continual learning."""

import operator
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax_rtrl.models.jax_util import zeros_like_tree


@dataclass
class WeightConsolidationState:
    """Configuration for Elastic Weight Consolidation."""

    factor: float
    omega: jax.Array
    reg_strength: jax.Array
    theta_ref: jax.Array
    decay: float = 0.9


def init_weight_consolidation_state(factor, decay, theta):
    """Initialize the state for weight consolidation."""
    return WeightConsolidationState(
        factor=factor,
        decay=decay,
        omega=zeros_like_tree(theta),
        reg_strength=zeros_like_tree(theta),
        theta_ref=theta,
    )


def update_reg_strength(
    state: WeightConsolidationState,
    new_theta,
    reset: bool = False,
    xi: float = 1e-6,
):
    """Update the regularization strength for weight consolidation.

    Args:
        state: The current consolidation state.
        new_theta: The new parameter value.
        reset: Whether to reset omega.
        xi: A small positive value to prevent division by zero.
    """

    def _update_reg_strength(reg_st, omega, t, t_ref):
        return state.decay * reg_st + (1 - state.decay) * omega / (
            (t - t_ref) ** 2 + xi
        )

    new_reg_strength = jax.tree.map(
        _update_reg_strength,
        state.reg_strength,
        state.omega,
        new_theta,
        state.theta_ref,
    )
    return state.replace(
        omega=zeros_like_tree(new_theta) if reset else state.omega,
        reg_strength=new_reg_strength,
        theta_ref=new_theta,
    )


def update_omega(state: WeightConsolidationState, dL_dtheta, dtheta_dt):
    """Update the consolidation coefficient.

    Args:
        omega: The consolidation coefficient.
        dL_dtheta: The gradient of the loss with respect to the parameter.
        dtheta_dt: The update that was applied to the parameter.
    """
    return state.replace(
        omega=jax.tree.map(lambda o, g, d: o - g * d, state.omega, dL_dtheta, dtheta_dt)
    )


def compute_weight_consolidation_loss(theta, state: WeightConsolidationState):
    """Compute the weight consolidation loss."""

    def _wc_loss(strength, t, t_ref):
        return state.factor * jnp.sum(strength * (t_ref - t) ** 2)

    return jax.tree.reduce(
        operator.add,
        jax.tree.map(
            _wc_loss,
            state.reg_strength,
            theta,
            state.theta_ref,
        ),
    )
