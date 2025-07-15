"""Weight consolidation for online continual learning.

This module implements online weight consolidation for continual learning scenarios
where task boundaries are not known explicitly. Based on:
- "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al. (2017)
- "Continual Learning Through Synaptic Intelligence" by Zenke et al. (2017)

Example usage for online continual learning:
    # Initialize consolidation state
    state = init_weight_consolidation_state(factor=1000.0, decay=0.95, theta=initial_params)

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

import operator
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax_rtrl.models.jax_util import zeros_like_tree
from jaxtyping import PyTree


@dataclass
class WeightConsolidationState:
    """Configuration for online weight consolidation.

    TODO: Consider renaming to OnlineConsolidationState for clarity.
    """

    factor: float  # Consolidation strength (lambda)
    omega: PyTree  # Importance weights (approximates Fisher Information)
    reg_strength: jax.Array  # Regularization strength per parameter
    theta_ref: jax.Array  # Reference parameters from last consolidation
    decay: float = 0.9  # Decay factor for online updates


def init_weight_consolidation_state(factor, decay, theta):
    """Initialize the state for online weight consolidation.

    TODO: Consider renaming to init_online_consolidation_state for clarity.

    Args:
        factor: Consolidation strength (lambda).
        decay: Decay factor for exponential moving averages.
        theta: Initial parameters.
    """
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
    """Update the regularization strength for online weight consolidation.

    TODO: Consider renaming to update_online_reg_strength for clarity.

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
        importance_contribution = jnp.abs(omega) / (param_drift + xi)
        return state.decay * reg_st + (1 - state.decay) * importance_contribution

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
    """Update the importance weights (omega) for online continual learning.

    TODO: Consider renaming to update_importance_weights for clarity.

    Args:
        state: The current consolidation state.
        dL_dtheta: The gradient of the loss with respect to the parameter.
        dtheta_dt: The update that was applied to the parameter.

    Note:
        This function approximates the Fisher Information Matrix using the
        relationship between gradients and parameter updates. For online learning,
        we use exponential moving average to adapt to distribution changes.
    """

    # Online Fisher Information approximation with exponential decay
    # omega approximates the diagonal Fisher Information: F_ii ≈ |∇L|² / |Δθ|²
    def _update_omega_online(old_omega, grad, param_update):
        # Compute importance as interaction between gradient and parameter change
        new_importance = jnp.abs(grad * param_update)
        # Use exponential moving average for online adaptation
        return state.decay * old_omega + (1 - state.decay) * new_importance

    return state.replace(
        omega=jax.tree.map(_update_omega_online, state.omega, dL_dtheta, dtheta_dt)
    )


def compute_weight_consolidation_loss(theta, state: WeightConsolidationState):
    """Compute the weight consolidation loss for online continual learning.

    TODO: Consider renaming to compute_consolidation_penalty for clarity.

    The consolidation loss is: L_consolidation = λ * Σ reg_strength_i * (θ_i - θ*_i)²
    where reg_strength incorporates both importance (omega) and parameter drift.
    """

    def _consolidation_loss(strength, t, t_ref):
        return state.factor * jnp.sum(strength * (t - t_ref) ** 2)

    return jax.tree.reduce(
        operator.add,
        jax.tree.map(
            _consolidation_loss,
            state.reg_strength,
            theta,
            state.theta_ref,
        ),
    )


def should_consolidate(
    state: WeightConsolidationState, current_theta, threshold: float = 0.01, min_steps: int = 100
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


def consolidate_online(state: WeightConsolidationState, new_theta_ref):
    """Consolidate parameters for online continual learning.

    This wraps update_reg_strength with reset=True for explicit consolidation.
    Should be called when significant parameter drift is detected or periodically.

    Args:
        state: Current consolidation state.
        new_theta_ref: Parameters to use as new reference.

    Returns:
        Updated state with consolidated parameters.
    """
    return update_reg_strength(state, new_theta_ref, reset=True)


def get_consolidation_stats(state: WeightConsolidationState):
    """Get statistics about the current consolidation state.

    Returns:
        Dictionary with consolidation statistics for monitoring.
    """
    omega_stats = jax.tree.map(
        lambda x: {"mean": jnp.mean(x), "std": jnp.std(x), "max": jnp.max(x), "min": jnp.min(x)},
        state.omega,
    )

    reg_stats = jax.tree.map(
        lambda x: {"mean": jnp.mean(x), "std": jnp.std(x), "max": jnp.max(x), "min": jnp.min(x)},
        state.reg_strength,
    )

    return {
        "omega_stats": omega_stats,
        "reg_strength_stats": reg_stats,
        "factor": state.factor,
        "decay": state.decay,
    }
