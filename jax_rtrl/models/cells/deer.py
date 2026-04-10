"""Concrete DEER cell subclasses for common ODECell types.

Each class combines :class:`~jax_rtrl.models.cells.ode.DEERCell` (which
provides the DEER parallel sequence solver) with a concrete
:class:`~jax_rtrl.models.cells.ode.ODECell` subclass via multiple
inheritance.  The MRO ensures that:

* ``__call__`` comes from :class:`DEERCell` (parallel DEER solve).
* ``_make_params``, ``_ode``, ``initialize_carry``, and all other
  cell-specific logic come from the concrete cell class.

Usage example::

    cell = DEERCTRNNCell(num_units=64, ode_type="murray", max_deer_iter=10)
    params = cell.init(jax.random.PRNGKey(0), None, jnp.zeros(8))
    h0 = cell.apply(params, jax.random.PRNGKey(1), (8,),
                    method=cell.initialize_carry)
    # Process a full sequence (T=20, input_dim=8) in parallel
    h_final, all_h = cell.apply(params, h0, jnp.zeros((20, 8)))
"""

from jax_rtrl.models.cells.ctrnn import CTRNNCell
from jax_rtrl.models.cells.lrc import LRCCell
from jax_rtrl.models.cells.ltc import LTCCell
from jax_rtrl.models.cells.ode import DEERCell


class DEERLRCCell(DEERCell, LRCCell):
    """LRC cell using the DEER parallel sequence solver.

    Combines :class:`~jax_rtrl.models.cells.ode.DEERCell` with
    :class:`~jax_rtrl.models.cells.lrc.LRCCell`.

    Because the LRC ODE has an exactly diagonal Jacobian w.r.t. the hidden
    state, :func:`~jax_rtrl.models.cells.deer_util.seq1d_full` converges in
    very few iterations (often 1–2 for typical parameters).

    See :class:`~jax_rtrl.models.cells.ode.DEERCell` for solver attributes
    (``max_deer_iter``).
    """


class DEERCTRNNCell(DEERCell, CTRNNCell):
    """CTRNN cell using the DEER parallel sequence solver.

    Combines :class:`~jax_rtrl.models.cells.ode.DEERCell` with
    :class:`~jax_rtrl.models.cells.ctrnn.CTRNNCell`.

    The CTRNN ODE has off-diagonal Jacobians (the hidden-to-hidden weight
    matrix ``W`` couples all neurons), so the full-matrix DEER iteration is
    required here.

    See :class:`~jax_rtrl.models.cells.ode.DEERCell` for solver attributes
    (``max_deer_iter``).
    """


class DEERLTCCell(DEERCell, LTCCell):
    """LTC cell using the DEER parallel sequence solver.

    Combines :class:`~jax_rtrl.models.cells.ode.DEERCell` with
    :class:`~jax_rtrl.models.cells.ltc.LTCCell`.

    See :class:`~jax_rtrl.models.cells.ode.DEERCell` for solver attributes
    (``max_deer_iter``).
    """
