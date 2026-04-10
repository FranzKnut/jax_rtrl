"""Leaky Integrate-and-Fire (LIF) and adaptive LIF (adLIF) cell implementations.

This module provides:
- ``spike``: a spiking function with a piecewise-linear surrogate gradient via
  ``jax.custom_vjp``, used for backpropagation through spiking networks.
- ``LIFCell``: a classical LIF or adLIF RNN cell that inherits from ``ODECell``,
  supporting a configurable integration step size ``dt`` and total integration
  time ``T`` per external timestep.  The forward pass uses the surrogate spike
  function, enabling gradient-based training (BPTT).
- ``eprop_lif`` / ``eprop_adlif``: RFLO-style eligibility-trace update functions
  that approximate the e-prop learning rule for LIF and adLIF networks.
- ``OnlineLIFCell``: wraps ``LIFCell`` with ``flax.linen.custom_vjp`` to replace
  BPTT gradients with e-prop eligibility traces for online (time-local) learning.

LIF dynamics (one sub-step of size ``dt``, soft reset)::

    alpha     = sigmoid(log_alpha) ** dt          # dt-scaled membrane decay
    I[t]      = W_in @ x[t] + W_rec @ s[t-1] + bias
    v[t]      = alpha * v[t-1] + (1 - alpha) * I[t] - threshold * s[t-1]
    s[t]      = spike(v[t] - threshold)

``T / dt`` sub-steps are executed per external timestep, all receiving the same
input ``x``.  With the default ``dt=T=1.0`` the behaviour is identical to a
single-step discrete-time LIF.

adLIF adds an adaptation variable::

    rho       = sigmoid(log_rho) ** dt            # dt-scaled adaptation decay
    a[t]      = rho * a[t-1] + (1 - rho) * s[t-1]
    I[t]      = W_in @ x[t] + W_rec @ s[t-1] + bias
    v[t]      = alpha * v[t-1] + (1-alpha)*(I[t] - beta*a[t]) - threshold*s[t-1]
    s[t]      = spike(v[t] - threshold)

where ``sigmoid(log_alpha)`` / ``sigmoid(log_rho)`` are the per-unit-time decay
factors and are scaled to ``** dt`` for sub-step integration.
"""

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import PRNGKey

from jax_rtrl.models.cells.ode import ODECell, OnlineODECell
from jax_rtrl.util.jax_util import tree_stack


# ============================================================
# Surrogate Gradient Spike Function
# ============================================================


@jax.custom_vjp
def spike(v: jnp.ndarray) -> jnp.ndarray:
    """Spiking function: ``H(v >= 0)`` in the forward pass.

    In the backward pass a piecewise-linear surrogate gradient is used::

        d(spike) / dv  ≈  max(0, 1 - |v|)

    Args:
        v: Pre-spike value (membrane potential minus threshold).

    Returns:
        Spike train – 1.0 where ``v >= 0``, 0.0 otherwise.
    """
    return (v >= 0.0).astype(jnp.float32)


def _spike_fwd(v):
    return (v >= 0.0).astype(jnp.float32), v


def _spike_bwd(v, g):
    surr = jnp.maximum(0.0, 1.0 - jnp.abs(v))
    return (g * surr,)


spike.defvjp(_spike_fwd, _spike_bwd)


def _psi(v_new, threshold):
    """Piecewise-linear pseudo-derivative (surrogate gradient) of the spike fn.

    Args:
        v_new:     Membrane potential after the current update.
        threshold: Per-neuron spike threshold.

    Returns:
        Surrogate derivative: ``max(0, 1 - |v_new - threshold|)``.
    """
    return jnp.maximum(0.0, 1.0 - jnp.abs(v_new - threshold))


# ============================================================
# LIF Cell
# ============================================================


class LIFCell(ODECell):
    """Leaky Integrate-and-Fire (LIF) or adaptive-LIF (adLIF) RNN cell.

    Inherits ``dt``, ``T``, ``solver``, ``wiring``, and ``wiring_kwargs`` from
    :class:`~jax_rtrl.models.cells.ode.ODECell`.  Each call to ``__call__``
    unrolls ``T / dt`` internal sub-steps, all receiving the same input ``x``.

    Parameters are initialised as learnable Flax parameters:

    * ``W_in``  – input weights ``(num_units, n_in)``.
    * ``W_rec``  – recurrent weights ``(num_units, num_units)``.
    * ``bias``   – bias ``(num_units,)``.
    * ``log_alpha`` – per-unit-time membrane decay in logit space;
      ``alpha_dt = sigmoid(log_alpha) ** dt``.
    * ``v_threshold`` – per-neuron spike threshold ``(num_units,)``.

    For ``lif_type="adlif"`` two additional parameters are created:

    * ``log_rho`` – per-unit-time adaptation decay in logit space;
      ``rho_dt = sigmoid(log_rho) ** dt``.
    * ``beta``    – per-neuron adaptation strength ``(num_units,)``.

    Carry structure:

    * LIF:   ``(v, s)``      – membrane potential, previous spikes.
    * adLIF: ``(v, a, s)``   – membrane potential, adaptation, previous spikes.

    The cell output at every external timestep is the spike train ``s`` produced
    by the **last** internal sub-step.

    Attributes:
        lif_type:      ``"lif"`` (classical) or ``"adlif"`` (adaptive).
        v_threshold:   Initial spike threshold value.
        tau_v_init:    Initial membrane time constant (in time steps).
        tau_a_init:    Initial adaptation time constant for adLIF (in time steps).
        beta_init:     Initial adaptation strength for adLIF.
    """

    # num_units is inherited from ODECell
    lif_type: Literal["lif", "adlif"] = "lif"
    v_threshold: float = 1.0
    tau_v_init: float = 20.0
    tau_a_init: float = 100.0
    beta_init: float = 1.6

    def _make_params(self, x):
        """Declare all Flax parameters for this cell."""
        n_in = x.shape[-1]
        n_h = self.num_units

        self.param("W_in", nn.initializers.lecun_uniform(), (n_h, n_in))
        # Small recurrent initialisation for stability
        self.param(
            "W_rec",
            lambda key, shape: nn.initializers.lecun_uniform()(key, shape) * 0.1,
            (n_h, n_h),
        )
        self.param("bias", nn.initializers.zeros, (n_h,))

        # Membrane decay: alpha = sigmoid(log_alpha) ≈ exp(-1/tau_v)
        alpha_init = jnp.exp(-1.0 / self.tau_v_init)
        log_alpha_init = jnp.log(alpha_init / (1.0 - alpha_init))
        self.param(
            "log_alpha",
            lambda key, shape: jnp.full(shape, log_alpha_init),
            (n_h,),
        )

        # Per-neuron learnable threshold
        self.param(
            "v_threshold",
            lambda key, shape: jnp.full(shape, self.v_threshold),
            (n_h,),
        )

        if self.lif_type == "adlif":
            rho_init = jnp.exp(-1.0 / self.tau_a_init)
            log_rho_init = jnp.log(rho_init / (1.0 - rho_init))
            self.param(
                "log_rho",
                lambda key, shape: jnp.full(shape, log_rho_init),
                (n_h,),
            )
            self.param(
                "beta",
                lambda key, shape: jnp.full(shape, self.beta_init),
                (n_h,),
            )

    def _forward_step(self, h, x, params):
        """One LIF / adLIF sub-step (no parameter-init side-effects).

        Uses ``alpha_dt = sigmoid(log_alpha) ** self.dt`` so that the membrane
        decay automatically scales with the integration step size.  For
        ``dt=1.0`` (the default) this is identical to ``sigmoid(log_alpha)``.

        Args:
            h:      Current carry – ``(v, s)`` for LIF, ``(v, a, s)`` for adLIF.
            x:      Input vector.
            params: Parameter dict (from ``self.variables["params"]``).

        Returns:
            ``(new_carry, spikes)`` – new hidden state and spike output.
        """
        alpha = jax.nn.sigmoid(params["log_alpha"])
        alpha_dt = alpha ** self.dt          # dt-scaled membrane decay
        threshold = params["v_threshold"]
        i_syn = x @ params["W_in"].T

        if self.lif_type == "lif":
            v, s = h
            i_syn = i_syn + s @ params["W_rec"].T + params["bias"]
            v_new = alpha_dt * v + (1.0 - alpha_dt) * i_syn - threshold * s
            s_new = spike(v_new - threshold)
            return (v_new, s_new), s_new

        elif self.lif_type == "adlif":
            v, a, s = h
            rho = jax.nn.sigmoid(params["log_rho"])
            rho_dt = rho ** self.dt          # dt-scaled adaptation decay
            a_new = rho_dt * a + (1.0 - rho_dt) * s
            i_syn = i_syn + s @ params["W_rec"].T + params["bias"]
            v_new = (
                alpha_dt * v
                + (1.0 - alpha_dt) * (i_syn - params["beta"] * a_new)
                - threshold * s
            )
            s_new = spike(v_new - threshold)
            return (v_new, a_new, s_new), s_new

        else:
            raise ValueError(f"Unknown lif_type: {self.lif_type!r}")

    def solve(self, h, x, return_sequences=False):
        """Run ``T / dt`` LIF sub-steps, all with the same input ``x``.

        Args:
            h:                Current carry.
            x:                Input vector (held constant for all sub-steps).
            return_sequences: If ``True``, stack and return the spike output of
                              every sub-step instead of only the last.

        Returns:
            ``(final_carry, spikes)`` – or stacked sequences if
            ``return_sequences=True``.
        """
        outs = []
        spikes = None
        if self.solver == "euler":
            for _step in jnp.arange(0, self.T, self.dt):
                h, spikes = self._forward_step(h, x, self.variables["params"])
                if return_sequences:
                    outs.append(spikes)
        else:
            raise ValueError(f"Unknown solver: {self.solver!r}")

        if return_sequences:
            return tree_stack(outs)
        return h, spikes

    @nn.compact
    def __call__(self, carry, x):
        """Forward pass.

        Args:
            carry: Previous state (or ``None`` for automatic initialisation).
            x:     Input vector.

        Returns:
            ``(new_carry, spikes)`` tuple where ``spikes`` is the output of the
            last internal sub-step.
        """
        if carry is None:
            carry = self.initialize_carry(self.make_rng(), x.shape)
        self._make_params(x)
        return self.solve(carry, x)

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple):
        """Return zero-initialised carry.

        Args:
            rng:         PRNG key (unused, kept for interface compatibility).
            input_shape: Shape of the input tensor, e.g. ``(n_in,)`` or ``(B, n_in)``.

        Returns:
            Initial carry (all zeros).
        """
        batch_shape = input_shape[:-1]
        v = jnp.zeros(batch_shape + (self.num_units,))
        s = jnp.zeros(batch_shape + (self.num_units,))
        if self.lif_type == "lif":
            return (v, s)
        # adlif
        a = jnp.zeros(batch_shape + (self.num_units,))
        return (v, a, s)

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


# ============================================================
# E-prop Eligibility Trace Functions
# ============================================================


def eprop_lif(cell: LIFCell, carry, params, x):
    """RFLO-style e-prop eligibility-trace update for LIF.

    Approximates the e-prop learning rule (Bellec et al., 2020) using a
    first-order low-pass filter over the instantaneous Jacobian of each output
    spike with respect to the corresponding weight.  The update rule for weight
    *W* is::

        jp_W[t+1] = alpha_dt * jp_W[t]  +  d(s_new)/d(W)|_{immediate}

    where ``alpha_dt = sigmoid(log_alpha) ** cell.dt`` is the dt-scaled membrane
    decay.  For ``cell.dt = 1`` this is identical to the previous formulation.

    This matches BPTT (with surrogate gradients) exactly for a single sub-step
    and provides a causal, time-local approximation for longer sequences.

    Args:
        cell:   ``LIFCell`` instance (provides ``num_units`` and ``dt``).
        carry:  Full online carry ``(h, jp, jx)`` where ``h = (v, s)``.
        params: Parameter dict from ``self.variables["params"]``.
        x:      Current input vector.

    Returns:
        ``(jp_new, jx_new)`` – updated eligibility traces.
    """
    h, jp, jx = carry
    v, s = h

    alpha = jax.nn.sigmoid(params["log_alpha"])
    alpha_dt = alpha ** cell.dt          # dt-scaled decay (trace filter coefficient)
    threshold = params["v_threshold"]

    # Recompute forward quantities for the trace
    i_syn = x @ params["W_in"].T + s @ params["W_rec"].T + params["bias"]
    v_new = alpha_dt * v + (1.0 - alpha_dt) * i_syn - threshold * s

    # Pseudo-derivative: psi_i = max(0, 1 - |v_new_i - threshold_i|)
    psi = _psi(v_new, threshold)  # (n_h,)

    # Immediate Jacobians  d(s_new[i])/d(W[i,j]) = psi[i] * d(v_new[i])/d(W[i,j])
    # W_in:   d(v_new)/d(W_in[i,j])  = (1 - alpha_dt[i]) * x[j]
    jp_W_in = (
        alpha_dt[:, None] * jp["W_in"]
        + psi[:, None] * (1.0 - alpha_dt)[:, None] * x[None, :]
    )
    # W_rec:  d(v_new)/d(W_rec[i,j]) = (1 - alpha_dt[i]) * s[j]
    jp_W_rec = (
        alpha_dt[:, None] * jp["W_rec"]
        + psi[:, None] * (1.0 - alpha_dt)[:, None] * s[None, :]
    )
    # bias:   d(v_new)/d(bias[i])    = (1 - alpha_dt[i])
    jp_bias = alpha_dt * jp["bias"] + psi * (1.0 - alpha_dt)

    # log_alpha:  d(alpha_dt)/d(log_alpha) = dt * alpha_dt * (1 - alpha)
    #             d(v_new)/d(log_alpha)     = (v - I) * dt * alpha_dt * (1 - alpha)
    jp_log_alpha = (
        alpha_dt * jp["log_alpha"]
        + psi * (v - i_syn) * cell.dt * alpha_dt * (1.0 - alpha)
    )

    # v_threshold: d(s_new[i])/d(threshold[i])
    #              = psi[i] * d(v_new)/d(threshold) + d(spike)/d(threshold)
    #              = psi[i] * (-s[i])                + (-psi[i])
    #              = -psi[i] * (s[i] + 1)
    jp_v_threshold = (
        alpha_dt * jp["v_threshold"]
        - psi * (s + 1.0)
    )

    jp_new = {
        "W_in": jp_W_in,
        "W_rec": jp_W_rec,
        "bias": jp_bias,
        "log_alpha": jp_log_alpha,
        "v_threshold": jp_v_threshold,
    }

    # Input gradient: d(s_new[i])/d(x[j]) = psi[i] * (1 - alpha_dt[i]) * W_in[i,j]
    jx_new = (
        alpha_dt[:, None] * jx
        + psi[:, None] * (1.0 - alpha_dt)[:, None] * params["W_in"]
    )

    return jp_new, jx_new


def eprop_adlif(cell: LIFCell, carry, params, x):
    """RFLO-style e-prop eligibility-trace update for adLIF.

    Extends ``eprop_lif`` to handle the adaptation variable.  The adaptation
    affects the recurrent and bias traces through the effective input current
    ``I_eff = I - beta * a_new``.

    Args:
        cell:   ``LIFCell`` instance (``lif_type="adlif"``).
        carry:  Full online carry ``(h, jp, jx)`` where ``h = (v, a, s)``.
        params: Parameter dict from ``self.variables["params"]``.
        x:      Current input vector.

    Returns:
        ``(jp_new, jx_new)`` – updated eligibility traces.
    """
    h, jp, jx = carry
    v, a, s = h

    alpha = jax.nn.sigmoid(params["log_alpha"])
    alpha_dt = alpha ** cell.dt          # dt-scaled membrane decay
    rho = jax.nn.sigmoid(params["log_rho"])
    rho_dt = rho ** cell.dt              # dt-scaled adaptation decay
    threshold = params["v_threshold"]
    beta = params["beta"]

    # Recompute forward quantities
    a_new = rho_dt * a + (1.0 - rho_dt) * s
    i_syn = x @ params["W_in"].T + s @ params["W_rec"].T + params["bias"]
    i_eff = i_syn - beta * a_new
    v_new = alpha_dt * v + (1.0 - alpha_dt) * i_eff - threshold * s

    psi = _psi(v_new, threshold)  # (n_h,)

    # W_in: same as LIF (adaptation doesn't affect input directly)
    jp_W_in = (
        alpha_dt[:, None] * jp["W_in"]
        + psi[:, None] * (1.0 - alpha_dt)[:, None] * x[None, :]
    )

    # W_rec: d(v_new[i])/d(W_rec[i,j]) = (1 - alpha_dt[i]) * s[j]
    jp_W_rec = (
        alpha_dt[:, None] * jp["W_rec"]
        + psi[:, None] * (1.0 - alpha_dt)[:, None] * s[None, :]
    )

    jp_bias = alpha_dt * jp["bias"] + psi * (1.0 - alpha_dt)

    # log_alpha:  d(alpha_dt)/d(log_alpha) = dt * alpha_dt * (1 - alpha)
    #             d(v_new)/d(log_alpha)     = (v - I_eff) * dt * alpha_dt * (1 - alpha)
    jp_log_alpha = (
        alpha_dt * jp["log_alpha"]
        + psi * (v - i_eff) * cell.dt * alpha_dt * (1.0 - alpha)
    )

    # v_threshold: same formula as LIF
    jp_v_threshold = (
        alpha_dt * jp["v_threshold"]
        - psi * (s + 1.0)
    )

    # log_rho: d(rho_dt)/d(log_rho) = dt * rho_dt * (1 - rho)
    #          d(a_new)/d(log_rho)   = (a - s) * dt * rho_dt * (1 - rho)
    #          d(v_new)/d(log_rho)   = -(1-alpha_dt)*beta*(a-s)*dt*rho_dt*(1-rho)
    jp_log_rho = (
        alpha_dt * jp["log_rho"]
        + psi * (-(1.0 - alpha_dt) * beta * (a - s) * cell.dt * rho_dt * (1.0 - rho))
    )

    # beta: d(v_new[i])/d(beta[i]) = -(1 - alpha_dt[i]) * a_new[i]
    jp_beta = (
        alpha_dt * jp["beta"]
        + psi * (-(1.0 - alpha_dt) * a_new)
    )

    jp_new = {
        "W_in": jp_W_in,
        "W_rec": jp_W_rec,
        "bias": jp_bias,
        "log_alpha": jp_log_alpha,
        "v_threshold": jp_v_threshold,
        "log_rho": jp_log_rho,
        "beta": jp_beta,
    }

    # Input gradient: same formula as LIF
    jx_new = (
        alpha_dt[:, None] * jx
        + psi[:, None] * (1.0 - alpha_dt)[:, None] * params["W_in"]
    )

    return jp_new, jx_new


# ============================================================
# Online LIF Cell (E-prop)
# ============================================================


class OnlineLIFCell(LIFCell):
    """LIF / adLIF cell with online e-prop learning.

    Wraps ``LIFCell`` with a ``flax.linen.custom_vjp`` backward pass that
    replaces BPTT gradients with causal eligibility traces (e-prop).

    In ``plasticity="bptt"`` mode the cell behaves exactly like ``LIFCell``
    (with surrogate gradients through ``spike``).  In ``plasticity="eprop"``
    mode the backward pass computes::

        grads_W[i, j] = learning_signal[i] * eligibility_trace_W[i, j]

    where the learning signal is the gradient of the loss with respect to the
    current sub-step's output spikes.

    Multiple internal sub-steps (``T / dt``) are handled inside ``solve()``,
    mirroring the structure of :class:`~jax_rtrl.models.cells.ode.OnlineODECell`.

    Online carry structure:

    * BPTT: ``(v, s)`` or ``(v, a, s)`` – same as ``LIFCell``.
    * eprop: ``(h, jp, jx)`` where ``h`` is the LIF carry, ``jp`` is a dict of
      eligibility traces matching the parameter structure, and ``jx`` is the
      input eligibility trace.

    Attributes:
        plasticity: ``"bptt"`` or ``"eprop"``.
    """

    plasticity: str = "eprop"

    def solve(self, carry, x, return_sequences=False):
        """Run ``T / dt`` e-prop sub-steps, each with its own backward pass.

        In ``plasticity="bptt"`` mode this delegates to the parent ``solve()``.

        Args:
            carry:            Current online carry.
            x:                Input vector (held constant across sub-steps).
            return_sequences: Stack spike outputs from every sub-step.

        Returns:
            ``(final_carry, spikes)`` or stacked sequence.
        """
        if self.plasticity == "bptt":
            return super().solve(carry, x, return_sequences)

        outs = []

        def f(mdl, _carry, _x):
            """Step function used when not differentiating."""
            h = _carry[0]
            new_h, out = mdl._forward_step(h, _x, mdl.variables["params"])
            return (new_h, *_carry[1:]), out

        def fwd(mdl, _carry, _x):
            """Forward pass: compute next state and update eligibility traces."""
            h = _carry[0]
            new_h, out = mdl._forward_step(h, _x, mdl.variables["params"])
            # Skip trace computation during parameter initialisation (traces None)
            if all(c is None for c in _carry[1:]):
                traces = _carry[1:]
            else:
                traces = mdl._trace_update(_carry, mdl.variables["params"], _x)
            new_carry = (new_h, *traces)
            return (new_carry, out), (new_h, *traces)

        def bwd(tmp, y_bar):
            """Backward pass: replace BPTT gradients with e-prop gradients."""
            # Learning signal = gradient of loss w.r.t. output spikes
            df_dy = y_bar[-1]
            # Also include gradient flowing through the carry's spike component
            h_bar = y_bar[-2][0]   # grad w.r.t. new h = (v_bar, [a_bar,] s_bar)
            df_dy = df_dy + h_bar[-1]  # s is always the last element of h

            grads_p, grads_x = OnlineODECell.online_gradient(
                tmp, df_dy, plasticity="rflo"
            )
            carry_zeros = jax.tree.map(jnp.zeros_like, tmp)
            return ({"params": grads_p}, carry_zeros, grads_x)

        if self.solver == "euler":
            f_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
            for _step in jnp.arange(0, self.T, self.dt):
                carry, spikes = f_grad(self, carry, x)
                if return_sequences:
                    outs.append(spikes)
        else:
            raise ValueError(f"Unknown solver: {self.solver!r}")

        if return_sequences:
            return tree_stack(outs)
        return carry, spikes

    @nn.compact
    def __call__(self, carry, x):
        """Forward pass.

        Args:
            carry: Previous online carry (or ``None`` for auto-init).
            x:     Input vector.

        Returns:
            ``(new_carry, spikes)`` tuple.
        """
        if carry is None:
            carry = self.initialize_carry(self.make_rng(), x.shape)
        self._make_params(x)
        return self.solve(carry, x)

    def _trace_update(self, carry, params, x):
        """Dispatch eligibility-trace update based on ``lif_type``.

        Args:
            carry:  Full online carry ``(h, jp, jx)``.
            params: Parameter dict.
            x:      Current input.

        Returns:
            ``(jp_new, jx_new)``.
        """
        if self.plasticity != "eprop":
            raise ValueError(
                f"Plasticity {self.plasticity!r} is not supported for LIFCell."
                " Use 'bptt' or 'eprop'."
            )
        if self.lif_type == "lif":
            return eprop_lif(self, carry, params, x)
        elif self.lif_type == "adlif":
            return eprop_adlif(self, carry, params, x)
        else:
            raise ValueError(f"Unknown lif_type: {self.lif_type!r}")

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple):
        """Initialise carry, including zero eligibility traces for online mode.

        Args:
            rng:         PRNG key.
            input_shape: Shape of the input tensor.

        Returns:
            Initial carry (structure depends on ``plasticity``).
        """
        h = super().initialize_carry(rng, input_shape)
        if self.plasticity == "bptt":
            return h

        batch_shape = input_shape[:-1]
        n_in = input_shape[-1]

        # Input eligibility trace: shape (n_h, n_in)
        jx = jnp.zeros(batch_shape + (self.num_units, n_in))

        # Initialise the module once to discover parameter shapes
        _h = h
        for _ in batch_shape:
            # Reduce to a single (non-batched) example for the init call
            _h = jax.tree.map(lambda t: t[0], _h)

        params = self.lazy_init(
            rng, (_h, None, None), jnp.zeros(input_shape[-1:])
        )

        # Zero eligibility traces matching each parameter's shape
        jp = jax.tree.map(
            lambda p: jnp.zeros(batch_shape + p.shape), params["params"]
        )
        return h, jp, jx
