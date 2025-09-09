"""Weight consolidation for online continual learning.

This module implements online weight consolidation for continual learning scenarios
where task boundaries are not known explicitly. Based on:
- "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al. (2017)
- "Continual Learning Through Synaptic Intelligence" by Zenke et al. (2017)

Example usage for online continual learning:
    # Initialize consolidation state
    state = init_weight_consolidation_state(decay=0.95, theta=initial_params)

    # Online learning loop (no explicit task boundaries):
    for step, batch in enumerate(data_stream):
        # Compute loss with consolidation regularization
        def loss_with_consolidation(params):
            task_loss = your_task_loss(params, batch)
            consolidation_loss = compute_weight_consolidation_loss(state, params)
            return task_loss + consolidation_loss

        # Get gradients and update parameters
        grads = jax.grad(loss_with_consolidation)(params)
        old_params = params
        params = optimizer.update(grads, params)
        param_update = jax.tree.map(lambda new, old: new - old, params, old_params)

        # Update omega (importance weights) using gradients and parameter changes
        state = update_omega(state, grads, param_update)

        # Update regularization strength periodically or when distribution shifts
        if step % update_interval == 0:
            state = update_reg_strength(state, params, reset=False)
"""

from functools import partial
import operator
from typing import Literal
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax_rtrl.models.jax_util import zeros_like_tree
from jaxtyping import PyTree


@dataclass
class WeightConsolidationState:
    """Configuration for online weight consolidation."""

    omega: PyTree  # Importance weights (approximates Fisher Information)
    reg_strength: jax.Array  # Regularization strength per parameter
    theta_ref: jax.Array  # Reference parameters from last consolidation


class WeightConsolidation(object):
    """Online weight consolidation for continual learning."""

    def __init__(self, cons_type: Literal["ewc", "si"] | None, decay: float):
        self.cons_type = cons_type
        self.decay = decay

    def init(self, theta: jax.Array) -> WeightConsolidationState:
        if self.cons_type is None:
            return None
        return init_weight_consolidation_state(theta)

    def step(
        self, state, dL_dtheta: jax.Array, dtheta_dt: jax.Array
    ) -> WeightConsolidationState:
        """Update the consolidation state with new parameters and gradients."""
        if self.cons_type == "ewc":
            return update_fisher_information(state, dL_dtheta, self.decay)
        elif self.cons_type == "si":
            return update_omega(state, dL_dtheta, dtheta_dt)
        elif self.cons_type is not None:
            raise ValueError(f"Unknown consolidation type: {self.cons_type}")

    def episode(self, state, new_theta: jax.Array) -> WeightConsolidationState:
        """Update the reference parameters at the end of an episode."""
        if self.cons_type == "si":
            return update_reg_strength(state, new_theta, self.decay, reset_omega=True)
        else:
            return set_theta_ref(state, new_theta)

    def loss(self, theta, state: WeightConsolidationState):
        return compute_weight_consolidation_loss(theta, state)


def compute_weight_consolidation_loss(theta, state: WeightConsolidationState):
    """Compute the weight consolidation loss for online continual learning.

    The consolidation loss is: L_consolidation = λ * Σ reg_strength_i * (θ_i - θ*_i)²
    """

    def _consolidation_loss(strength, t, t_ref):
        return jnp.mean(strength * (t - t_ref) ** 2)

    return jax.tree.reduce(
        operator.add,
        jax.tree.map(
            _consolidation_loss,
            state.reg_strength,
            theta,
            state.theta_ref,
        ),
    )


def init_weight_consolidation_state(theta):
    """Initialize the state for online weight consolidation.

    Args:
        theta: Initial parameters.
    """
    return WeightConsolidationState(
        omega=zeros_like_tree(theta),
        reg_strength=zeros_like_tree(theta),
        theta_ref=theta,
    )


def set_theta_ref(state: WeightConsolidationState, new_theta_ref: jax.Array):
    """Set the reference parameters (theta_ref) for online weight consolidation.

    Args:
        state: The current consolidation state.
        new_theta_ref: The new reference parameters.
    """
    return state.replace(theta_ref=new_theta_ref)


@partial(jax.jit, static_argnames=("reset_omega",))
def update_reg_strength(
    state: WeightConsolidationState,
    new_theta,
    decay: float,
    reset_omega: bool = True,
    xi: float = 1e-6,
):
    """Update the regularization strength for synaptic intelligence.

    Args:
        state: The current consolidation state.
        new_theta: The new parameter value.
        reset: Whether to reset omega (useful for periodic consolidation).
        xi: A small positive value to prevent division by zero.

    Note:
        For online learning, this should be called periodically rather than
        after every update to avoid computational overhead.
    """

    def _update_reg_strength(reg_st, omega, t, t_ref):
        # Online update: balance between accumulated importance (omega) and parameter drift
        param_drift = (t - t_ref) ** 2 + xi
        importance_contribution = omega / param_drift
        return decay * reg_st + importance_contribution

    new_reg_strength = jax.tree.map(
        _update_reg_strength,
        state.reg_strength,
        state.omega,
        new_theta,
        state.theta_ref,
    )
    return state.replace(
        omega=zeros_like_tree(new_theta) if reset_omega else state.omega,
        reg_strength=new_reg_strength,
        theta_ref=new_theta,
    )


@jax.jit
def update_omega(state: WeightConsolidationState, dL_dtheta, dtheta_dt):
    """Update the importance weights (omega) for online continual learning.

    Args:
        state: The current consolidation state.
        dL_dtheta: The gradient of the loss with respect to the parameter.
        dtheta_dt: The update that was applied to the parameter.

    Note:
        This function approximates the Fisher Information Matrix using the
        relationship between gradients and parameter updates. For online learning,
        we use exponential moving average to adapt to distribution changes.
    """

    # Synaptic Intelligence approximation
    return state.replace(
        omega=jax.tree.map(lambda o, g, d: o - g * d, state.omega, dL_dtheta, dtheta_dt)
    )


@jax.jit
def update_fisher_information(state: WeightConsolidationState, dL_dtheta, decay: float):
    """Update the approximation of the Fisher Information Matrix for online elastic weight consolidation.

    Args:
        state: The current consolidation state.
        dL_dtheta: The gradient of the loss with respect to the parameter.
        dtheta_dt: The update that was applied to the parameter.

    Note:
        This function approximates the Fisher Information Matrix using the
        relationship between gradients and parameter updates. For online learning,
        we use exponential moving average to adapt to distribution changes.
    """

    # Online Synaptic Intelligence approximation
    def _update_fi(old_strength, grad):
        # Compute importance as interaction between gradient and parameter change
        new_importance = grad**2
        # if use_abs:
        #     new_importance = jax.tree.map(jnp.abs, new_importance)
        # Use exponential moving average for online adaptation
        return decay * old_strength + new_importance
        # return new_importance + old_omega

    return state.replace(
        reg_strength=jax.tree.map(_update_fi, state.reg_strength, dL_dtheta)
    )


def should_consolidate(
    state: WeightConsolidationState,
    current_theta,
    threshold: float = 0.01,
    min_steps: int = 100,
) -> bool:
    """Determine if consolidation should occur based on parameter drift.

    Args:
        state: Current consolidation state.
        current_theta: Current parameters.
        threshold: Relative change threshold for triggering consolidation.
        min_steps: Minimum steps before allowing consolidation.

    Returns:
        Boolean indicating whether consolidation should occur.
    """
    # Need minimum history before consolidating
    if hasattr(state, "step_count") and state.step_count < min_steps:
        return False

    # Compute relative parameter change magnitude
    def _relative_change(theta, theta_ref):
        diff_norm = jnp.linalg.norm(theta - theta_ref)
        ref_norm = jnp.linalg.norm(theta_ref)
        return diff_norm / (ref_norm + 1e-8)

    changes = jax.tree.map(_relative_change, current_theta, state.theta_ref)
    avg_change = jnp.mean(jnp.array(jax.tree.leaves(changes)))

    return avg_change > threshold
