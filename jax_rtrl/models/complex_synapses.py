"""Benna-Fusi model of synaptic plasticity with connected vessel dynamics.

The connected vessels model approximates a power-law memory kernel using a chain of
communicating dynamic variables (vessels). Synaptic weight is read from the first
vessel, while hidden vessels regularize it with the history of modifications.

Reference: Benna & Fusi (2016); Kaplanis et al. (2018)

Copilot Edit:
Coupled dynamics: Each vessel u_k is connected to its neighbors with conductances g, allowing liquid-like flow between vessels
Optimal parameters: C_k = 2^(k-1) (exponentially increasing capacities) and g_{k,k+1} ∝ 2^(-(k+2)) (exponentially decreasing coupling)
External input: Weight updates feed into the first vessel only: C_1 du_1/dt = dw_ext/dt + g_{1,2}(u_2 - u_1)
Observable weight: w is simply u_1 (the first vessel), while u_2,...,u_N are hidden regularization variables
Leak term: The last vessel leaks to zero (u_{N+1} = 0), creating a dissipative boundary
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
    
    def __init__(self, n_vessels: int = 3):
        """Initialize the Benna-Fusi model.
        
        Args:
            n_vessels: Number of hidden state variables in the chain.
        """
        self.n_vessels = n_vessels
        
        # Capacitances: C_k = 2^(k-1)
        self.C = jnp.array([2.0 ** (k - 1) for k in range(1, n_vessels + 1)])
        
        # Conductances: g_{k,k+1} ∝ 2^(-(k+2))
        self.g = jnp.array([2.0 ** (-(k + 2)) for k in range(n_vessels)])
    
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
            return {'vessels': jnp.tile(w[..., None], (1,) * len(w.shape) + (self.n_vessels,))}
        
        return jax.tree.map(init_weight_state, weights)
    
    def __call__(
        self, 
        state: dict[str, dict[str, Any]], 
        dw: dict[str, Any],
        dt: float = 1.0
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
                    # First vessel: u_0 = u_1 + dw_ext/dt
                    u_left = vessels[..., 0] + weight_update
                else:
                    u_left = vessels[..., k - 1]
                
                left_term = self.g[k - 1] * (u_left - vessels[..., k]) if k > 0 else self.g[0] * (weight_update)
                
                # Right coupling: g_{k,k+1} * (u_{k+1} - u_k)
                if k < self.n_vessels - 1:
                    right_term = self.g[k] * (vessels[..., k + 1] - vessels[..., k])
                else:
                    # Last vessel: u_{N+1} = 0 (leak)
                    right_term = self.g[k] * (0.0 - vessels[..., k])
                
                derivatives = derivatives.at[..., k].set((left_term + right_term) / self.C[k])
            
            # Euler integration
            new_vessels = vessels + dt * derivatives
            return {'vessels': new_vessels}
        
        return jax.tree.map(update_weight_state, state, dw)
    
    def get_weights(self, state: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Extract effective weights from hidden state (first vessel u_1).
        
        Args:
            state: Hidden state dictionary.
        
        Returns:
            Dictionary of effective weights matching original structure.
        """
        def get_first_vessel(vessels_dict):
            return vessels_dict['vessels'][..., 0]
        
        return jax.tree.map(get_first_vessel, state)
    
    