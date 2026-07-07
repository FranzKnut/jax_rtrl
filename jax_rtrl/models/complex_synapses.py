"""Benna-Fusi model of synaptic plasticity with connected vessel dynamics.

The connected vessels model approximates a power-law memory kernel using a chain of
communicating dynamic variables (vessels). Synaptic weight is read from the first
vessel, while hidden vessels regularize it with the history of modifications.

Reference: Benna & Fusi (2016); Kaplanis et al. (2018)
"""

import copy

import jax
import jax.numpy as jnp
from typing import Any, Literal
from flax import struct


@struct.dataclass
class BennaFusiConfig:
    """Configuration for Benna-Fusi model."""

    n_vessels: int = 3  # Number of hidden state variables in the chain.
    g_factor: float = 1e-5  # Scaling factor for the conductances.
    dt: float = 1.0  # Time step for updates.
    decay: bool = True  # Whether to let the last vessel decay to 0.
    decay_to_initial: bool = False  # Whether to decay to initial value instead of 0.
    w_index: int = 0  # Index of the weight dimension in the parameter arrays.
    initialization: Literal["all_w", "first_w", "zero"] = (
        "all_w"  # How to initialize hidden vessels.
    )


@struct.dataclass
class BennaFusiState:
    """State of the Benna-Fusi model for a single parameter."""

    steps: int  # Number of updates applied to this parameter (for tracking time).
    vessels: Any  # Pytree with leaves of shape (..., n_vessels)


class BennaFusi:
    """Connected vessel model of complex synapses.

    Implements a chain of N coupled dynamic variables where:
    - u_1 is the observable weight
    - u_2, ..., u_N are hidden states
    - Dynamics: C_k * du_k/dt = g_{k-1,k}(u_{k-1} - u_k) + g_{k,k+1}(u_{k+1} - u_k)
    Optimal parameters: C_k = 2^(k-1), g_{k,k+1} ∝ 2^(-(k+2))
    """

    def __init__(
        self,
        config: BennaFusiConfig = BennaFusiConfig(),
    ):
        """Initialize the Benna-Fusi model.

        Args:
            config: BennaFusiConfig object with model parameters.
        """
        self.config = config

        # Capacitances: C_k = 2^(k-1)
        self.C = jnp.array([2.0 ** (k - 1) for k in range(1, config.n_vessels + 1)])

        # Conductances: g_{k,k+1} ∝ 2^(-k-2)
        self.g = jnp.array(
            [config.g_factor * 2.0 ** (-k - 2) for k in range(1, config.n_vessels + 1)]
        )

    def init_state(self, weights: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Initialize hidden state given weight pytree.

        Args:
            weights: Pytree of weights (arbitrary nested dict structure).

        Returns:
            State pytree where each weight has 'vessels': array of shape (..., n_vessels).
            All vessels initialized to w (since u_1 = w and others start empty).
        """

        if self.config.decay_to_initial:
            # If decaying to initial value, we need to store the initial weights as well
            self.initial_weights = copy.deepcopy(weights)

        def init_weight_state(w):
            # Initialize all vessels to w
            if self.config.initialization == "all_w":
                return jnp.tile(
                    w[..., None], (1,) * len(w.shape) + (self.config.n_vessels,)
                )
            elif self.config.initialization == "first_w":
                # Only first vessel is w, others start at 0
                return jnp.concatenate(
                    [
                        w[..., None],
                        jnp.zeros((*w.shape, self.config.n_vessels - 1)),
                    ],
                    axis=-1,
                )
            elif self.config.initialization == "zero":
                # All vessels start at 0
                return jnp.zeros((*w.shape, self.config.n_vessels))
            else:
                raise ValueError(
                    f"Unknown initialization method: {self.config.initialization}. "
                    "Choose from 'all_w', 'first_w', or 'zero'."
                )

        return BennaFusiState(steps=0, vessels=jax.tree.map(init_weight_state, weights))

    def update(
        self,
        state: BennaFusiState,
        dw: dict[str, Any],
        consolidation_factor: dict[str, Any] | None = None,
    ) -> BennaFusiState:
        """Update hidden state given weight change.

        Solves: C_k * du_k/dt = g_{k-1,k}(u_{k-1} - u_k) + g_{k,k+1}(u_{k+1} - u_k)

        Where u_0 = u_1 + dw_ext/dt (first vessel receives external input).

        Args:
            state: Current hidden state (output from init_state or previous call).
            dw: Weight update for each parameter.
            dt: Time step.
            consolidation_factor: Pytree with same prefix as dw of factors for consolidating updates.
                If provided, dw is scaled by this factor to simulate consolidation effects.
        Returns:
            Updated state with same structure as input state.
        """

        def update_weight_state(
            vessels, weight_update, initial_weight=None, cons_factor=None
        ):
            # vessels shape: (..., n_vessels)
            # weight_update shape: (...)

            # Compute derivatives for each vessel using Euler method
            # du_k/dt = (g_{k-1,k}(u_{k-1} - u_k) + g_{k,k+1}(u_{k+1} - u_k)) / C_k

            derivatives = jnp.zeros_like(vessels)

            for k in range(self.config.n_vessels):
                # Left coupling: g_{k-1,k} * (u_{k-1} - u_k)
                if k == 0:
                    left_term = weight_update
                else:
                    left_term = self.g[k - 1] * (vessels[..., k - 1] - vessels[..., k])
                    if cons_factor is not None:
                        left_term *= cons_factor

                # Right coupling: g_{k,k+1} * (u_{k+1} - u_k)
                if k < self.config.n_vessels - 1:
                    right_term = self.g[k] * (vessels[..., k + 1] - vessels[..., k])
                elif self.config.decay:
                    # Last vessel: u_{N+1} = 0 (leak)
                    targ = initial_weight if self.config.decay_to_initial else 0.0
                    right_term = self.g[k] * (targ - vessels[..., k])

                # Right term only active after steps > 2^k / g_1 to allow initial consolidation
                # right_term *= state.steps > ((2**k) / self.g[0])

                derivatives = derivatives.at[..., k].set(
                    (left_term + right_term) / self.C[k]
                )

            # Euler integration
            return vessels + self.config.dt * derivatives

        _iw = self.initial_weights if self.config.decay_to_initial else None
        _args = (
            (_iw, consolidation_factor) if consolidation_factor is not None else (_iw,)
        )
        return state.replace(
            steps=state.steps + 1,
            vessels=jax.tree.map(update_weight_state, state.vessels, dw, *_args),
        )

    def get_weights(self, state: BennaFusiState) -> dict[str, Any]:
        """Extract effective weights from hidden state (first vessel u_1).

        Args:
            state: Hidden state dictionary.

        Returns:
            Dictionary of effective weights matching original structure.
        """

        def get_first_vessel(vessels):
            return vessels[..., self.config.w_index]

        return jax.tree.map(get_first_vessel, state.vessels)
