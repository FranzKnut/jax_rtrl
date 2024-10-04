"""Plasticity Rules for online learning."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import tree_map
from jax.nn import initializers


class Plasticity:
    """A generic plasticity rule that serves as a template for other plasticity rules."""

    def __init__(self, cell, feedback_alignment=False, key=None):
        """Set Cell."""
        self.cell = cell
        self.feedback_alignment = feedback_alignment
        if self.feedback_alignment:
            self.B = initializers.normal()(key, (cell.units, cell.units))

    def build(self, traces, batch_size=1):
        """Add the necessary traces to the cell's trace dictionary.

        @param traces:
        @return: traces so calls can be chained
        """
        batched_units = (batch_size, self.cell.units) if batch_size else (self.cell.units,)
        # batched_out = (batch_size, self.cell.out_size) if batch_size else (
        #     self.cell.out_size,)
        # P = dh/dw
        traces["P"] = tree_map(lambda x: jnp.zeros(batched_units + x.shape), eqx.filter(self.cell, eqx.is_array))
        # if self.cell.output_mapping:
        #     # P_out = (dy/dw_out, dy/db_out)
        #     traces['P_out'] = (jnp.zeros(batched_out + self.cell.w_out.shape),
        #                        jnp.zeros(batched_out + self.cell.b_out.shape))
        if self.cell.with_dh_h:
            # Q = dh/dh
            traces["Q"] = jnp.zeros(batched_units + (self.cell.input_size + self.cell.units,))
        return traces

    @staticmethod
    def call(cell, traces, ctx):
        """Return the update to the traces.

        @param traces:
        @param ctx:
        @return:
        """
        return {}

    @staticmethod
    def loss_mapping(cell, dout):
        """Map the loss to the hidden state.

        Applies the output mapping to the loss gradient.
        """
        if cell.output_mapping == "linear":
            # dy/dh = W^T
            dloss_dh = jnp.einsum("ij,...j->...i", cell.w_out.T, dout)
        else:
            # dy/dh = W
            dloss_dh = jnp.concatenate(
                [
                    jnp.zeros(dout.shape[:-1] + (cell.units - cell.out_size,)),
                    dout * (cell.w_out if cell.output_mapping == "affine" else 1),
                ],
                axis=-1,
            )
        return dloss_dh

    def update(self, cell, traces, dout, with_dy_dx=False):
        """Compute the approximate jacobian.

        @param traces:
        @param dout:
        @return: Dictionary where key is the name of the weight and value is the update
        """
        gradients = tree_map(lambda t: jnp.einsum("H,H...->...", dout, t), traces["P"])

        if with_dy_dx:
            dy_dx = jnp.einsum("H,H...->...", dout, traces["Q"])[..., : cell.input_size]
            return gradients, dy_dx

        # if cell.output_mapping:
        #     w_out_grad = tree_map(lambda t: dout @ t, traces['P_out'])
        #     gradients = eqx.tree_at(lambda t: (
        #         t.w_out, t.b_out), gradients, w_out_grad)

        return gradients


class LocalMSE(Plasticity):
    """Corresponds to SnAp-0.

    NOTE: It might be faster to just compute the gradient of the loss for one step directly

    "Hebbian" loss = (post(pre)-pre)²/2
    Gradient of hebbian loss dHl/dw

    df/dw = d(x(t+1)-x(t))/dw
    x(t+1)=x(t) + dt * f(x(t))
    """

    @staticmethod
    def call(cell, traces, ctx):
        """Compute the immediate jacobian of the cell params wrt. previous state.

        Args:
            cell (_type_): _description_
            traces (_type_): _description_
            ctx (_type_): _description_

        Returns:
            _type_: _description_
        """
        del traces
        pre = ctx["pre"]
        post = ctx["post"]
        out = {}
        if cell.output_mapping:
            jac_out = jax.jacrev(cell.map_output, argnums=0)(cell, post)
            out["P_out"] = (jac_out.w_out, jac_out.b_out)

        if cell.with_dh_h:
            jac_cell, jac_pre = jax.jacrev(cell.f, argnums=[0, 2], has_aux=True)(cell, [0], pre)[0]
            out["Q"] = jac_pre[-cell.units :, -cell.units :]
        else:
            jac_cell = jax.jacrev(cell.f, argnums=0, has_aux=True)(cell, [0], pre)[0]
        p = tree_map(lambda x: x[-cell.units :], jac_cell)
        out["P"] = p
        return out


class RTRL(Plasticity):
    """Real-time Recurrent Learning."""

    def __init__(self, cell, key):
        """Real-time Recurrent Learning needs state-state jacobian."""
        del key
        super().__init__(cell)
        self.cell.with_dh_h = True

    @staticmethod
    def call(cell, traces, ctx):
        """Compute the RTRL step.

        Computes immediate jacobian as in LocalMSE and also the state-state jacobian.
        Then computes the RTRL step by multiplying the immediate jacobian with the state-state.
        Args:
            cell (_type_): self
            traces (_type_): previous rtrl traces
            ctx (_type_): RNN cell step context

        Returns:
            _type_: _description_
        """
        P = traces["P"]
        # Q = traces['Q']
        pre = ctx["pre"]
        # post = ctx['post']

        def rtrl_step(p, rec, dh):
            return p + (rec + dh[-cell.units :]) * cell.dt

        # immediate jacobian (this step)
        df_dw, df_dh = jax.jacrev(cell.f, argnums=[0, 2], has_aux=True)(cell, [0], pre)[0]

        # dh/dh = d(h + f(h) * dt)/dh = I + df/dh * dt
        identitiy = jnp.concatenate([jnp.zeros((cell.units, cell.input_size)), jnp.identity(cell.units)], axis=-1)
        dh_dh = identitiy + df_dh[-cell.units :]
        # / cell.dt

        # jacobian trace (previous step * dh_h)
        comm = jax.tree_map(lambda p: jnp.tensordot(df_dh[..., -cell.units :, -cell.units :], p, axes=1), P)

        # Update dh_dw approximation
        dh_dw = jax.tree_map(rtrl_step, P, comm, df_dw)

        # if cell.output_mapping:
        #     jac_out = jax.jacrev(cell.map_output, argnums=0)(cell, post)
        #     dy_dout = (jac_out.w_out, jac_out.b_out)
        #     return {'P': dh_dw,
        #             'P_out': dy_dout,
        #             'Q': dh_dh}
        return {"P": dh_dw, "Q": dh_dh}

    # y = W_out x
    # dL/dx = B (y-yhat)


class RFLO(Plasticity):
    """Random Feedback Local Online learning.

    Inspired by https://github.com/omarschall/vanilla-rtrl.
    """

    def __init__(self, cell, key):
        """Initialize random feedback matrix B."""
        super().__init__(cell, feedback_alignment=True, key=key)

    @staticmethod
    def call(cell, traces, ctx):
        """Update P by one time step of temporal filtration via the inverse time constant alpha (see Eq. 1)."""
        if "u" not in ctx:
            raise Exception("RFLO ist not implemented for this cell type.")

        P = traces["P"].w
        S = traces["P"].tau
        pre = ctx["pre"]
        post = ctx["post"]
        u = ctx["u"]

        # derivative of activation phi'
        df_dh = jax.jacrev(cell.activation)(u)

        # Outer product the get Immediate Jacobian
        M_immediate = jnp.einsum("ij,k", df_dh, jnp.concatenate([pre, jnp.ones(1)]))  # Add one for bias

        # Update eligibility traces
        P = P + jnp.einsum("i,ijk->ijk", 1 / cell.tau, M_immediate - P)
        dh_dtau = ((pre[-cell.units :] - post) / cell.tau) * jnp.eye(cell.units) - S
        S = S + jnp.einsum("i,ik->ik", 1 / cell.tau, dh_dtau)

        out = eqx.tree_at(lambda t: (t.w, t.tau), traces["P"], (P, S))
        # if cell.output_mapping:
        #     jac_out = jax.jacrev(cell.map_output, argnums=0)(cell, post)
        #     p_out = (jac_out.w_out, jac_out.b_out)
        #     return {'P': out,
        #             'P_out': p_out}
        return {"P": out}


class UORO(Plasticity):
    """Unbiased Online Recurrent Optimization."""

    @staticmethod
    def call(cell, traces, ctx):
        """Update the traces of UORO."""
        pre = ctx["h"][0]
        act = ctx["act"]
        traces["T"] += -jnp.multiply(cell.tau, traces["T"].T).T + act[None, :]
        traces["S"] += -cell.tau * traces["S"] - pre
        return traces


class Infomax(Plasticity):
    """Information theoretic plasticity rule that maximizes the mutual information between the input and the output."""

    def build(self, traces, batch_size=1):
        """Add the necessary traces to the cell's trace dictionary.

        @param traces:
        @return: traces so calls can be chained
        """
        pass

    def call(self, traces, ctx):
        """Update the traces.

        @param traces:
        @param ctx:
        @return:
        """
        if self.type == "ax+b":
            if self.activation == "sigm":
                da = x * (1 - 2 * y) + 1 / (self.a + 1e-8)
                db = 1 - 2 * y

            else:
                da = -2 * x * y + 1 / (self.a + 1e-8)
                db = -2 * y
        else:
            if self.activation == "sigm":
                da = -1 / self.a + (x - self.b) * (2 * y - 1) / self.a**2
                db = (2 * y - 1) / self.a

            else:
                da = -1 / self.a + 2 * (x - self.b) * y / self.a**2
                db = 2 * y / self.a  # 4*eta*np.mean(new_x*new_y**2)
        return traces

    def update(self, traces, dout):
        """
        Updates the weights
        @param traces:
        @param dout:
        @return: Dictionary where key is the name of the weight and value is the update
        """
        self.a += self.lr * jnp.mean(self.da, axis=0)
        self.b += self.lr * jnp.mean(self.db, axis=0)
        return {}, {}


class Hebbian(Plasticity):
    """Hebbian associative learning rule."""

    def call(self, traces, ctx):
        """Update the traces"""
        pre = ctx["pre"][:, -1]
        post = ctx["h_new"][0, -1]
        dw = jnp.outer(post, pre)
        return {"dw": dw}, {}


class Oja(Plasticity):
    def call(self, traces, ctx):
        pre = ctx["pre"][:, -1]
        post = ctx["h_new"][0, -1]
        dw = post.T * pre - (post**2).T * self.cell.params["w"]  # * old_trace
        return {"dw": dw}, {}


class BCM(Plasticity):
    def __init__(self, cell, eta=0.7):
        super().__init__(cell)
        self.eta = eta

    def build(self, traces, batch_size=1):
        # theta = <y^2>
        traces["theta"] = jnp.zeros(self.cell.units)

    def call(self, traces, ctx):
        post = ctx["h_new"][0]
        traces["theta"] += self.eta * (-traces["theta"] + post**2)

    def update(self, traces, dout):
        pre = ctx["pre"][0]
        post = ctx["h_new"][0, -1]
        dw = (post * (post - traces["theta"])).T * pre * dout[:, None]
        return {"dw": dw}, {}

    # * self.syn_mask * self.Erev) - self._trace_decay * old_trace
    # * self.Erev
