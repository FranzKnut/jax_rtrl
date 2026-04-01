"""Parallel sequence solver using DEER (quasi-Newton) iteration.

Based on the DEER framework:
  Lim et al. (2024) "DEER: Differentiable Newton Iteration for Recurrent Networks"
  Reference implementation:
  https://github.com/MoniFarsang/LrcSSM/blob/82b07c3bd4c14ae6c44e5e9d4b6262a83dbd3131/elk/algs/deer.py

The key idea: instead of computing h[t+1] = func(h[t], x[t]) sequentially in O(T)
steps, use Newton iterations where each step solves a *linear* recurrence via
jax.lax.associative_scan in O(log T) depth.

For LRC the Jacobian d(func)/dh is exactly diagonal (all h-dependencies are
element-wise), so the diagonal (quasi) DEER approximation is exact.
"""

import jax
import jax.numpy as jnp


def _diagonal_binary_operator(q_i, q_j):
    """Associative binary operator for a diagonal linear recurrence.

    Combines two segments (g_i, b_i) and (g_j, b_j) where each segment
    encodes h[t] = g[t] * h[t-1] + b[t] (element-wise, diagonal case).

    The combined segment represents the composition:
      h[j] = g_j * (g_i * h_start + b_i) + b_j = (g_j * g_i) * h_start + (g_j * b_i + b_j)
    """
    g_i, b_i = q_i
    g_j, b_j = q_j
    return g_j * g_i, g_j * b_i + b_j


def _solve_diagonal_recurrence(G, b, h0):
    """Solve the diagonal linear recurrence h[t] = G[t] * h[t-1] + b[t] in parallel.

    Uses jax.lax.associative_scan for O(log T) depth computation.

    Args:
        G: (T, ny) diagonal state-transition coefficients.
        b: (T, ny) bias terms.
        h0: (ny,) initial hidden state.

    Returns:
        (T, ny) sequence of hidden states h[1], ..., h[T].
    """
    # Prepend identity element for h0 so the scan includes the initial condition.
    ones = jnp.ones_like(G[:1])  # (1, ny) – acts as identity (g=1, h[0]=h0)
    G_aug = jnp.concatenate([ones, G], axis=0)  # (T+1, ny)
    b_aug = jnp.concatenate([h0[None], b], axis=0)  # (T+1, ny)

    _, yt = jax.lax.associative_scan(_diagonal_binary_operator, (G_aug, b_aug))
    return yt[1:]  # Drop the initial h0, return (T, ny)


def seq1d(func_step, h0, xs, params, max_iter=10):
    """Solve h[t+1] = func_step(h[t], x[t], params) for a whole sequence in parallel.

    Applies the DEER quasi-Newton iteration: each Newton step linearises the
    nonlinear recurrence around the current iterate and solves the resulting
    *diagonal* linear system with jax.lax.associative_scan.

    For LRC, d(func_step)/dh is exactly diagonal, so this is not an
    approximation – it converges to the correct solution as max_iter increases.

    The outer Newton loop is implemented via jax.lax.scan so that gradients
    are propagated correctly through all iterations.

    Args:
        func_step: Callable(h, x, params) -> h_next.
            The per-step recurrence. Must be differentiable w.r.t. h (argnums=0).
        h0: (ny,) initial hidden state.
        xs: (T, nx) input sequence.
        params: Model parameters (pytree). Gradients are computed through these.
        max_iter: Number of Newton iterations. 10 is sufficient for typical LRC
            models; increase for longer sequences or tighter tolerances.

    Returns:
        (T, ny) hidden state sequence h[1], ..., h[T].

    References:
        Lim et al. (2024). "DEER: Differentiable Newton Iteration for
        Recurrent Networks." Based on the implementation in MoniFarsang/LrcSSM.
    """
    ny = h0.shape[-1]
    h_init = jnp.zeros((xs.shape[0], ny), dtype=h0.dtype)

    def newton_step(h_k, _):
        """One quasi-Newton (DEER) iteration.

        Linearises func_step around h_k and solves the resulting diagonal
        linear recurrence in parallel.
        """
        # Shifted iterate: h_prev[t] = h[t-1], with h_prev[0] = h0
        h_prev = jnp.concatenate([h0[None], h_k[:-1]], axis=0)  # (T, ny)

        # Function values at current iterate
        f_vals = jax.vmap(lambda h, x: func_step(h, x, params))(h_prev, xs)  # (T, ny)

        # Diagonal Jacobians G[t] = diag(d func_step / dh |_{h_prev[t], x[t]})
        # Use standard-basis JVPs (forward-mode) to avoid materialising the
        # full n×n Jacobian matrix; only the j-th element of each JVP
        # column is retained, giving the diagonal element G[j,j] directly.
        def _diag_jac(h, x):
            eye = jnp.eye(h.shape[0], dtype=h.dtype)

            def _jvp_diag_j(e_j, j):
                _, col_j = jax.jvp(
                    lambda h_: func_step(h_, x, params), (h,), (e_j,)
                )
                return col_j[j]

            return jax.vmap(_jvp_diag_j)(eye, jnp.arange(h.shape[0]))

        G = jax.vmap(_diag_jac)(h_prev, xs)  # (T, ny)

        # Newton RHS: b[t] = f[t] - G[t] * h_prev[t]
        b = f_vals - G * h_prev  # (T, ny)

        # Solve h_new[t] = G[t] * h_new[t-1] + b[t] via parallel scan
        h_new = _solve_diagonal_recurrence(G, b, h0)  # (T, ny)
        return h_new, None

    h_sol, _ = jax.lax.scan(newton_step, h_init, None, length=max_iter)
    return h_sol
