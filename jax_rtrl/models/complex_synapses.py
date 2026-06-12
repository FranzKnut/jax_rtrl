"""Benna-Fusi model of synaptic plasticity with connected vessel dynamics.

The connected vessels model approximates a power-law memory kernel using a chain of
communicating dynamic variables (vessels). Synaptic weight is read from the first
vessel, while hidden vessels regularize it with the history of modifications.

Reference: Benna & Fusi (2016); Kaplanis et al. (2018)
"""

import jax
import jax.numpy as jnp
from typing import Any


class BennaFusi:
    """Connected vessel model of complex synapses.

    Implements a chain of N coupled dynamic variables where:
    - u_1 is the observable weight
    - u_2, ..., u_N are hidden states
    - Dynamics: C_k * du_k/dt = g_{k-1,k}(u_{k-1} - u_k) + g_{k,k+1}(u_{k+1} - u_k)

    Optimal parameters: C_k = 2^(k-1), g_{k,k+1} ∝ 2^(-(k+2))
    """

    def __init__(self, n_vessels: int = 3, g_factor: float = 1e-5):
        """Initialize the Benna-Fusi model.

        Args:
            n_vessels: Number of hidden state variables in the chain.
            g_factor: Scaling factor for the conductances.
        """
        self.n_vessels = n_vessels

        # Capacitances: C_k = 2^(k-1)
        self.C = jnp.array([2.0 ** (k - 1) for k in range(1, n_vessels + 1)])

        # Conductances: g_{k,k+1} ∝ 2^(-(k+2))
        self.g = jnp.array(
            [g_factor * 2.0 ** (-(k + 2)) for k in range(1, n_vessels + 1)]
        )

    def init_state(self, weights: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Initialize hidden state given weight pytree.

        Args:
            weights: Pytree of weights (arbitrary nested dict structure).

        Returns:
            State pytree where each weight has 'vessels': array of shape (..., n_vessels).
            All vessels initialized to w (since u_1 = w and others start empty).
        """

        def init_weight_state(w):
            # Initialize all vessels to w for the first vessel u_1 = w
            return jnp.tile(w[..., None], (1,) * len(w.shape) + (self.n_vessels,))

            # return jnp.concatenate(
            #     [
            #         w[..., None],
            #         jnp.zeros((*w.shape, self.n_vessels - 1)),
            #     ],
            #     axis=-1,
            # )

        return jax.tree.map(init_weight_state, weights)

    def update(
        self, state: dict[str, dict[str, Any]], dw: dict[str, Any], dt: float = 1.0
    ) -> dict[str, dict[str, Any]]:
        """Update hidden state given weight change.

        Solves: C_k * du_k/dt = g_{k-1,k}(u_{k-1} - u_k) + g_{k,k+1}(u_{k+1} - u_k)

        Where u_0 = u_1 + dw_ext/dt (first vessel receives external input).

        Args:
            state: Current hidden state (output from init_state or previous call).
            dw: Weight update for each parameter.
            dt: Time step.

        Returns:
            Updated state with same structure as input state.
        """

        def update_weight_state(vessels, weight_update):
            # vessels shape: (..., n_vessels)
            # weight_update shape: (...)

            # Compute derivatives for each vessel using Euler method
            # du_k/dt = (g_{k-1,k}(u_{k-1} - u_k) + g_{k,k+1}(u_{k+1} - u_k)) / C_k

            derivatives = jnp.zeros_like(vessels)

            for k in range(self.n_vessels):
                # Left coupling: g_{k-1,k} * (u_{k-1} - u_k)
                if k == 0:
                    left_term = weight_update
                else:
                    left_term = self.g[k - 1] * (vessels[..., k - 1] - vessels[..., k])

                # Right coupling: g_{k,k+1} * (u_{k+1} - u_k)
                if k < self.n_vessels - 1:
                    right_term = self.g[k] * (vessels[..., k + 1] - vessels[..., k])
                else:
                    # Last vessel: u_{N+1} = 0 (leak)
                    right_term = self.g[k] * (0.0 - vessels[..., k])

                derivatives = derivatives.at[..., k].set(
                    (left_term + right_term) / self.C[k]
                )

            # Euler integration
            return vessels + dt * derivatives

        return jax.tree.map(update_weight_state, state, dw)

    def get_weights(self, state: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Extract effective weights from hidden state (first vessel u_1).

        Args:
            state: Hidden state dictionary.

        Returns:
            Dictionary of effective weights matching original structure.
        """

        def get_first_vessel(vessels):
            return vessels[..., 0]

        return jax.tree.map(get_first_vessel, state)
