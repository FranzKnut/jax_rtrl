from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core.frozen_dict import unfreeze


class GRUOnlineCell(nn.recurrent.RNNCellBase, metaclass=nn.recurrent.RNNCellCompatibilityMeta):
    """
    GRU layer that updates internal elegibility traces to allow online learning.
    """

    d_hidden: int  # hidden state dimension
    training_mode: str
    activation_fn = jnp.tanh
    gate_fn = nn.sigmoid
    dtype = None

    def setup(self):
        dense_h = partial(
            nn.Dense,
            features=self.d_hidden,
            use_bias=False,
        )
        self.dense_hr = dense_h(name="hr")
        self.dense_hz = dense_h(name="hz")
        # add bias because the linear transformations aren't directly summed.
        self.dense_hn = dense_h(name="hn", use_bias=True)

        dense_i = partial(
            nn.Dense,
            features=self.d_hidden,
            dtype=self.dtype,
            use_bias=True,
        )
        self.dense_ir = dense_i(name="ir")
        self.dense_iz = dense_i(name="iz")
        self.dense_in = dense_i(name="in")

    def __call__(self, h, i):
        def sig_prime(x):
            return nn.sigmoid(x) * nn.sigmoid(1 - x)

        def tanh_prime(x):
            return 1 - jnp.tanh(x) ** 2

        if self.training_mode == "online_spatial":
            R = self.dense_ir(i) + jax.lax.stop_gradient(self.dense_hr(h))
        else:
            R = self.dense_ir(i) + self.dense_hr(h)
        r = nn.sigmoid(R)

        if self.training_mode == "online_spatial":
            Z = self.dense_iz(i) + jax.lax.stop_gradient(self.dense_hz(h))
        else:
            Z = self.dense_iz(i) + self.dense_hz(h)
        z = nn.sigmoid(Z)

        if self.training_mode == "online_spatial":
            hn = jax.lax.stop_gradient(self.dense_hn(h))
        else:
            hn = self.dense_hn(h)
        N = self.dense_in(i) + r * hn
        n = jnp.tanh(N)

        new_h = (1.0 - z) * h + z * n

        if self.training_mode in ["online_snap1"]:
            diag_dr_dh = sig_prime(R) * jnp.diag(self.dense_hr.variables["params"]["kernel"])
            diag_dz_dh = sig_prime(Z) * jnp.diag(self.dense_hz.variables["params"]["kernel"])
            diag_dn_dh = tanh_prime(N) * (diag_dr_dh * hn + r * jnp.diag(self.dense_hr.variables["params"]["kernel"]))
            diag_dh_dh = diag_dz_dh * (n - h) + diag_dn_dh * z + (1.0 - z)

            # Even though vscode tells you these are not used, they are.
            diag_dh_dz = (n - h) * sig_prime(Z)
            diag_dh_dn = z * tanh_prime(N)
            diag_dh_dr = z * tanh_prime(N) * hn * sig_prime(R)

            # Trick: use the diag of d.../dh as an error signal and then use vjp to compute
            # partial[i, i, :]
            grad = jax.tree.map(jnp.zeros_like, unfreeze(self.variables["params"]))
            grad = unfreeze(grad)
            for beg in ["h", "i"]:
                for end in ["r", "z", "n"]:
                    dense = self.__getattr__("dense_" + beg + end)
                    inputs = eval(beg)
                    if beg == "h" and end == "n":
                        errors = errors = eval("diag_dh_d%s" % end) * r
                    else:
                        errors = eval("diag_dh_d%s" % end)

                    grad[beg + end] = jax.vjp(
                        lambda p: dense.apply({"params": p}, inputs),
                        unfreeze(dense.variables["params"]),
                    )[1](errors)[0]

            return new_h, (new_h, diag_dh_dh, {"params": grad})
        else:
            return new_h, new_h


class GRU(nn.Module):
    """
    GRU layer that updates internal elegibility traces to allow online learning.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # NOT USE HERE input and output dimensions
    seq_length: int  # time sequence length
    training_mode: str = "bptt"  # which learning algorithm that will be used
    training: bool = False  # TODO remove, for debugging purposes
    activation_fn = jnp.tanh  # NOT USED
    dtype = None  # needed for obscure Flax reasons

    def setup(self):
        def update_bptt(cell, carry, inputs):
            return cell.apply({"params": cell.variables["params"]}, carry, inputs)

        def update_tbptt_and_spatial(cell, carry, inputs):
            return cell.apply({"params": cell.variables["params"]}, jax.lax.stop_gradient(carry), inputs)

        def update_snap(cell, carry, inputs):
            hiddens, jac, t = carry
            new_hiddens, diag_rec_jac, partial_param = cell.apply(
                {"params": cell.variables["params"]}, hiddens, inputs
            )[1]
            new_hiddens += self.pert_hidden_states.value[t]

            def _jac_update(old, new):
                if len(old.shape) == 1:
                    diag = diag_rec_jac
                elif len(old.shape) == 2:
                    diag = diag_rec_jac[:, None]
                return new + diag * old

            new_jac = jax.tree.map(_jac_update, jac, partial_param)

            return (new_hiddens, new_jac, t + 1), (new_hiddens, new_jac)

        self.cell = GRUOnlineCell(self.d_hidden, self.training_mode)
        if self.training_mode in ["bptt", "online_reservoir"]:
            update_fn = update_bptt
        elif self.training_mode in ["online_1truncated", "online_spatial"]:
            update_fn = update_tbptt_and_spatial
        elif self.training_mode == "online_snap1":
            self.pert_hidden_states = self.variable(
                "perturbations",
                "hidden_states",
                jnp.zeros,
                (self.seq_length, self.d_hidden),
            )
            update_fn = update_snap
        else:
            raise ValueError("Training mode not implemented:", self.training_mode)

        if self.is_initializing():
            _ = self.cell(jnp.zeros((self.d_hidden,)), jnp.zeros((self.d_hidden)))

        if self.training_mode == "online_snap1":
            # initialize all the traces
            traces_attr = []
            for e in jax.tree_util.tree_flatten_with_path(self.cell.variables["params"])[0]:
                path, val = e
                name_attr = "_".join([p.key for p in path])
                traces_attr += [name_attr]

                def _init(v):
                    return jnp.diagonal(
                        jnp.zeros(
                            [
                                self.seq_length,
                            ]
                            + [self.d_hidden] * len(v)
                        )
                    )

                val_attr = self.variable("traces", name_attr, _init, val.shape)
                self.__setattr__(name_attr, val_attr)
            self.traces_attr = traces_attr

        self.scan = nn.scan(
            update_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=self.seq_length,
        )

        self.update = update_bptt

    def __call__(self, inputs):
        if self.training_mode in [
            "bptt",
            "online_reservoir",
            "online_1truncated",
            "online_spatial",
        ]:
            init_carry = jnp.zeros((self.d_hidden,))
            return self.scan(self.cell, init_carry, inputs)[1]
        elif self.training_mode == "online_snap1":
            init_carry = (
                jnp.zeros((self.d_hidden,)),
                {
                    "params": jax.tree.map(
                        lambda p: jnp.diagonal(jnp.zeros((self.d_hidden,) + p.shape)),
                        unfreeze(self.cell.variables["params"]),
                    )
                },
                0,
            )
            new_hiddens, traces = self.scan(self.cell, init_carry, inputs)[1]

            # Store the traces for gradient computation
            for e in jax.tree_util.tree_flatten_with_path(traces["params"])[0]:
                path, trace = e
                name_attr = "_".join([p.key for p in path])
                self.__getattr__(name_attr).value = trace

            return new_hiddens

    def update_gradients(self, grad):
        # Sanity check, gradient should be computed through autodiff for these methods
        if self.training_mode in [
            "bptt",
            "online_1truncated",
            "online_spatial",
            "online_reservoir",
        ]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        delta = self.pert_hidden_states.value

        # Otherwise grad[i] = delta[i] * trace[i]
        for attr in self.traces_attr:
            path = attr.split("_")
            trace = self.__getattr__(attr).value
            # for biases
            if len(trace.shape) == 2:
                g = delta * trace
            # for weights
            else:
                g = delta[:, None] * trace
            grad["cell"][path[0]][path[1]] = jnp.sum(g, axis=0)

        return grad
