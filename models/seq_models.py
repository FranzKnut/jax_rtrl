from chex import PRNGKey
from dataclasses import dataclass, field
from typing import Tuple
import distrax
from flax import linen as nn
import jax
import jax.numpy as jnp
from models.jax_util import zeros_like_tree
from models.mlp import MLP, DistributionLayer, FADense


class SequenceLayer(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    seq: nn.Module  # seq module
    d_output: int  # output layer sizes
    dropout: float = 0.0  # dropout probability
    norm: str = "layer"  # which normalization to use
    training: bool = True  # in training mode (dropout in trainign mode only)

    @nn.compact
    def __call__(self, hidden, inputs, **kwargs):
        """Applies, layer norm, seq model, dropout and GLU in that order."""
        out1 = nn.Dense(self.d_output)
        out2 = nn.Dense(self.d_output)
        if self.norm in ["layer"]:
            normalization = nn.LayerNorm()
        else:
            normalization = nn.BatchNorm(use_running_average=not self.training, axis_name="batch")
        drop = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not self.training)

        x = normalization(inputs)  # pre normalization
        hidden, x = self.seq(hidden, x, **kwargs)  # call seq model
        x = drop(nn.gelu(x))
        x = out1(x) * jax.nn.sigmoid(out2(x))  # GLU
        x = drop(x)
        return hidden, inputs + x  # skip connection


class MultiLayerRNN(nn.RNNCellBase):
    """Multilayer RNN."""

    layers: list
    rnn_cls: nn.RNNCellBase
    rnn_kwargs: dict = field(default_factory=dict)

    def make_rnns(self):
        return [
            SequenceLayer(
                self.rnn_cls(size, **self.rnn_kwargs, name=f"rnn_{i}"),
                size,
            )
            for i, size in enumerate(self.layers)
        ]

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        batch_size = input_shape[0:1] if len(input_shape) > 1 else ()
        shapes = zip(self.layers, [self.layers[:1], *[(s,) for s in self.layers[:-1]]])
        return [
            self.rnn_cls(size, **self.rnn_kwargs).initialize_carry(rng, batch_size + in_size)
            for size, in_size in shapes
        ]

    def setup(self):
        self.rnns = [
            SequenceLayer(
                self.rnn_cls(size, **self.rnn_kwargs, name=f"rnn_{i}"),
                size,
            )
            for i, size in enumerate(self.layers)
        ]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @nn.compact
    def __call__(self, carries, x, **kwargs):
        """Call MLP."""
        x = FADense(self.layers[0], name="input")(x)
        for i, rnn in enumerate(self.rnns):
            carries[i], x = rnn(carries[i], x, **kwargs)
        return carries, x


class FAMultiLayerRNN(MultiLayerRNN):
    """MultiLayer RNN with different modes of backpropagating error signals."""

    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)
    fa_type: str = "bp"

    @nn.compact
    def __call__(self, carries, x, **kwargs):
        if self.fa_type == "bp":
            return MultiLayerRNN.__call__(self, carries, x, **kwargs)
        elif self.fa_type == "dfa":
            Bs = [
                self.variable(
                    "falign",
                    f"B{i}",
                    self.kernel_init,
                    self.make_rng() if self.has_rng("params") else None,
                    (size, self.layers[-1]),
                ).value
                for i, size in enumerate(self.layers[:-1])
            ]
        else:
            raise ValueError("unknown fa_type: " + self.fa_type)

        def f(mdl, _carries, x, _Bs):
            return MultiLayerRNN.__call__(mdl, _carries, x, **kwargs)

        def fwd(mdl: MultiLayerRNN, _carries, x, _Bs):
            """Forward pass with tmp for backward pass."""
            vjps = []
            for i, rnn in enumerate(mdl.rnns):
                (_carries[i], x), vjp_func = jax.vjp(rnn.apply, rnn.variables, _carries[i], x, **kwargs)
                vjps.append(vjp_func)

            return (_carries, x), (vjps, _Bs)

        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            vjps, _Bs = tmp
            grads = zeros_like_tree(self.variables["params"])
            grads_inputs = []
            for i in range(len(self.rnns)):
                y_t = _Bs[i] @ y_bar[1] if i < len(self.rnns) - 1 else y_bar[1]
                params_t, *inputs_t = vjps[i]((y_bar[0][i], y_t))
                grads[f"rnn_{i}"] = params_t["params"]
                grads_inputs.append(inputs_t[0])
                if i == 0:
                    x_grad = inputs_t[1]

            return ({"params": grads}, grads_inputs, x_grad, zeros_like_tree(_Bs))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)

        return fa_grad(self, carries, x, Bs)


@dataclass
class RNNEnsembleConfig:
    num_modules: int
    layers: tuple[int]
    model: type = nn.RNNCellBase
    out_size: int | None = None
    out_dist: str | None = None
    input_layers: tuple[int] | None = None  # TODO
    output_layers: tuple[int] | None = None
    fa_type: str = "bp"
    rnn_kwargs: dict = field(default_factory=dict)
    skip_connection: bool = False


class RNNEnsemble(nn.RNNCellBase):
    """Ensemble of RNN cells."""

    config: RNNEnsembleConfig

    def setup(self):
        self.ensembles = [
            FAMultiLayerRNN(
                self.config.layers,
                rnn_cls=self.config.model,
                rnn_kwargs=self.config.rnn_kwargs,
                fa_type=self.config.fa_type,
                name=f"ensembles_{i}",
            )
            for i in range(self.config.num_modules)
        ]
        if self.config.output_layers:
            self.mlps_out = [
                MLP(self.config.output_layers, f_align=self.config.rnn_kwargs.get("f_align", False), name=f"mlps_{i}")
                for i in range(self.config.num_modules)
            ]
        if self.config.out_size is not None:
            # Make distribution for each submodule
            self.dists = [
                DistributionLayer(self.config.out_size, self.config.out_dist, name=f"dists_{i}")
                for i in range(self.config.num_modules)
            ]

    def _postprocessing(self, outs, x):
        for i in range(self.config.num_modules):
            out = outs[i]
            # Optional Skip connection
            if self.config.skip_connection:
                out = jnp.concatenate([x, out], axis=-1)
            # Outup FF layers
            if self.config.output_layers:
                out = self.mlps_out[i](out)
            if self.config.out_size is not None:
                # Make distribution for each submodule
                outs[i] = self.dists[i](out)

        # Aggregate outputs
        if not self.config.out_dist:
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-2), *outs)
        else:
            # Last dim is batch in distrax
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-1), *outs)
            outs = distrax.MixtureSameFamily(distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs)
        return outs

    def __call__(self, h: jax.Array | None = None, x: jax.Array = None, rng=None, **call_args):  # noqa
        """Call submodules and concatenate output.

        If out_dist is not None, the output will be distribution(s),

        Parameters
        ----------
        h : List
            of rnn submodule states
        x : Array
            input
        rng : PRNGKey, optional,
            if given, returns one value per submodule in order to train them independently,
            If None, mean of submodules or a Mixed Distribution is returned.

        Returns
        -------
        _type_
            _description_
        """
        if h is None:
            h = self.initialize_carry(jax.random.key(0), x.shape)
        outs = []
        carry_out = []
        for i in range(self.config.num_modules):
            # Loop over rnn submodules
            carry, out = self.ensembles[i](h[i], x, **call_args)
            carry_out.append(carry)
            outs.append(out)

        # Post-process and aggregate outputs
        outs = self._postprocessing(outs, x)

        if rng is not None:
            outs = outs.sample(seed=rng) if self.config.out_dist else outs.mean(axis=0)
        return carry_out, outs

    def _loss(self, hidden, loss_fn, target, x):
        # Post-process and aggregate outputs
        outs = self._postprocessing(hidden, x)
        return loss_fn(outs, target)

    @nn.nowrap
    def loss_and_grad(self, params, loss_fn, carry, hidden, x, target):
        # Compute gradient of loss wrt to network output
        loss, (grads, df_dh) = jax.value_and_grad(self.apply, argnums=[0, 1])(
            params, [c[-1][0].real for c in carry], loss_fn, target, x, method=self._loss
        )

        # Compute parameter gradients for each RNN
        for i, (c, d) in enumerate(zip(carry, df_dh)):
            grads["params"][f"rnns_{i}"] = self.config.model.rtrl_gradient(
                c, d, plasticity=self.config.rnn_kwargs["plasticity"]
            )[0]
        return loss, grads

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        return [
            FAMultiLayerRNN(
                self.config.layers, rnn_cls=self.config.model, rnn_kwargs=self.config.rnn_kwargs
            ).initialize_carry(rng, input_shape)
            for _ in range(self.config.num_modules)
        ]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @staticmethod
    def clip_tau(params):
        """HACK: clip tau to > 1.0"""
        for k in params["params"]["rnn"]:
            for _l in params["params"]["rnn"][k]:
                params["params"]["rnn"][k][_l]["tau"] = jnp.clip(params["params"]["rnn"][k][_l]["tau"], min=1.0)
        return params
